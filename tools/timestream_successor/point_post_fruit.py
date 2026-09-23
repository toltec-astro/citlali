"""Post-FRUIT extensions of the additive companion, preserving saved estimators.

No native execution. Covariances describe a model transferred from background,
not an exact conditional detector-noise law or a feedback uncertainty product.
"""
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from scipy.stats import chi2
from point_additive import draw_noise, variance, make_context, pair_covariance
from point_empirical import coordinates, patch, stats
from point_admission import fit_diagnostic, projection
from point_empirical import aperture_measurements
from inspect_forward_map import require, read_yaml

NAMES = ('peak', 'template', 'aperture1', 'aperture1.5', 'aperture2')
UNITS = ('mJy/nominal beam',)*2 + ('mJy/nominal beam arcsec^2',)*3


def frozen_masks(images, header, geometries, factor=4.):
    shape = images[0]['PRIMARY'].shape
    require(all(i['PRIMARY'].shape == shape for i in images), 'map shape changed')
    common = np.logical_and.reduce([i['SCI_SUPPORT'].astype(bool) & np.isfinite(i['PRIMARY']) &
                                    np.isfinite(i['NORMALIZATION']) & (i['NORMALIZATION'] > 0) for i in images])
    x, y = coordinates(shape, header)
    source = np.zeros(shape, bool)
    circles = []
    for p in geometries:
        radius = factor*max(p[3:5])*np.sqrt(8*np.log(2))
        source |= np.hypot(x-p[1], y-p[2]) <= radius
        circles.append(dict(center_east_north_arcsec=list(p[1:3]), radius_arcsec=radius))
    return common, source, dict(factor=factor, circles=circles, common_pixels=int(common.sum()),
                               background_pixels=int((common & ~source).sum()), rule='union, fixed across cycles')


def measurement_values(record):
    good = record['fit']['status'] == 'success'
    return [record['fit']['parameters'][0] if good else None, record['frozen_template_projection'],
            *[r['value_mJy_per_beam_arcsec2'] for r in record['apertures']]]


def operators(fixed):
    return np.stack([fixed.weights, *[4.*(fixed.radius <= f*fixed.major) for f in (1., 1.5, 2.)]])


def context_for(signal, coverage, support, header, fixed, parameters, model):
    c = make_context(signal, coverage, support, header, fixed.beam, parameters, model)
    c['q'] = fixed.weights.copy()
    c['weights'] = operators(fixed)
    c['original_parameters'] = fixed.parameters.copy()
    c['major'] = fixed.major
    c['measurement_domain'] = c['roi'] | np.any(c['weights'] != 0, axis=0)
    require(not np.any(c['measurement_domain'] & ~c['support']), 'measurement union incomplete')
    c['noiseless_record'] = saved_measure(c['baseline'], c)
    c['noiseless'] = measurement_values(c['noiseless_record'])
    require(all(v is not None and np.isfinite(v) for v in c['noiseless']), 'noiseless estimator unavailable')
    return c


def saved_measure(data, c):
    # Same calls and ordering as FixedMeasurements.measure; all nuisance
    # parameters free, same fit initialization, exact original fixed geometry.
    fit = fit_diagnostic(data, c['support'], c['header'], c['beam'])
    if fit['status'] != 'success':
        return dict(fit=fit, frozen_template_projection=None,
                    apertures=[dict(value_mJy_per_beam_arcsec2=None) for _ in range(3)])
    rel = data-fit['parameters'][6]
    x, y = coordinates(data.shape, c['header'])
    return dict(fit=fit, frozen_template_projection=projection(rel, c['support'], c['q']),
                apertures=aperture_measurements(rel, c['support'], x, y,
                    c['original_parameters'][1:3], c['major'], c['beam'], 4.))


def trial(payload):
    c, seed, identity, b = payload
    wall = time.perf_counter(); cpu = time.process_time()
    noise = draw_noise(c['psd'], c['scale'], seed)
    generated = time.perf_counter()-wall
    linear = [float(np.sum(w*noise)) for w in c['weights']]
    t = time.perf_counter()
    record = saved_measure(np.where(c['support'], c['baseline']+noise, np.nan), c)
    good = record['fit']['status'] == 'success'
    return dict(trial=b, seed=list(seed), identity=identity, status='success' if good else 'failed',
                values=measurement_values(record), fit=record['fit'], linear_noise=linear,
                normalized_noise_second_moment=float(np.mean((noise[c['measurement_domain']]/c['scale'][c['measurement_domain']])**2)),
                generation_wall_seconds=generated, fitting_wall_seconds=time.perf_counter()-t,
                cpu_seconds=time.process_time()-cpu)


def joint_summary(records, reference, observed=None):
    success = [r for r in records if r['status'] == 'success']
    vals = np.array([r['values'] for r in success], dtype=float).reshape(-1, 5)
    complete = len(success) == len(records)
    measurements = []
    for i, name in enumerate(NAMES):
        a = vals[:, i]; n = len(a)
        sd = float(a.std(ddof=1)) if n > 1 else None
        measurements.append(dict(name=name, units=UNITS[i], observed=None if observed is None else observed[i],
            noiseless_reference=reference[i], scatter=sd if complete else None,
            relative_scatter=sd/abs(observed[i]) if complete and observed is not None and observed[i] else None,
            partial_survivor_scatter=sd if not complete else None,
            mean_shift=float(a.mean()-reference[i]) if complete else None,
            partial_survivor_mean_shift=float(a.mean()-reference[i]) if n and not complete else None,
            quantiles=np.quantile(a, [.025, .16, .5, .84, .975]) if n else None,
            quantile_scope='complete conditional model distribution' if complete else 'partial survivors only',
            MC_SE_mean=sd/np.sqrt(n) if sd is not None else None,
            approximate_MC_relative_SE_scatter=1/np.sqrt(2*(n-1)) if n > 1 else None))
    covariance = np.cov(vals, rowvar=False, ddof=1) if complete and len(vals)>1 else None
    correlation = np.corrcoef(vals, rowvar=False) if covariance is not None else None
    return dict(trials=len(records), failures=len(records)-len(success),
                bound_limited=sum(r['fit'].get('constrained', False) for r in success),
                primary_available=complete, measurements=measurements,
                covariance=covariance, correlation=correlation, order=NAMES,
                covariance_units='outer product of listed measurement units',
                scope='conditional additive scatter; no bias correction; sensitivity intervals are MC precision only')


def ensemble(c, count, seed, array, model, workers=4):
    # Model identities are explicit; variants deliberately use the shared base
    # stream for matched comparisons. Scheduling never enters the random key.
    identity = dict(array=array, cycle=10, model=model, comparison_stream='base-T10-common-random-v1')
    payload = [(c, (seed, int(array[1:]), 10, 0, b), identity, b) for b in range(count)]
    if workers == 1:
        return list(map(trial, payload))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(trial, payload, chunksize=8))


def dense_variance(psd, scale, weights):
    use = weights != 0
    loc = np.argwhere(use); a = (weights*scale)[use]
    dy = loc[:, 0, None]-loc[None, :, 0]; dx = loc[:, 1, None]-loc[None, :, 1]
    rho = np.fft.ifft2(psd).real
    return float(np.sum(a[:, None]*rho[dy % len(rho), dx % len(rho)]*a[None, :]))


def larger_same_covariance(psd):
    # Embed identical discrete modes at the same physical frequencies, NOT a
    # newly sampled/repaired spectrum. This enlarges only the generation grid.
    # The represented periodic covariance (including its limitations) is exact.
    out = np.zeros(tuple(2*n for n in psd.shape))
    out[::2, ::2] = 4*psd
    return out


def propagation_checks(c, records, seed, array):
    B = len(records); bounds = chi2.ppf([.0005, .9995], B-1)/(B-1)
    linear = np.array([r['linear_noise'] for r in records])
    larger = larger_same_covariance(c['psd'])
    extra = np.array([[np.sum(w*draw_noise(larger, c['scale'], (seed, int(array[1:]), 10, 999, b)))
                       for w in c['weights']] for b in range(B)])
    checks=[]
    for i, w in enumerate(c['weights']):
        v = variance(c['psd'], c['scale'], w); dense = dense_variance(c['psd'], c['scale'], w)
        vl = variance(larger, c['scale'], w); sample = linear[:, i].var(ddof=1); big = extra[:, i].var(ddof=1)
        require(np.isclose(v, dense, rtol=1e-11) and np.isclose(v, vl, rtol=1e-11), 'dense/domain covariance mismatch')
        checks.append(dict(name=NAMES[i+1], analytic_variance=v, dense_variance=dense,
            diagonal_variance=float(np.sum((w*c['scale'])**2)), simulated_variance=sample,
            variance_ratio=sample/v, MC999_ratio_interval=bounds, sampling_check=bool(bounds[0]<=sample/v<=bounds[1]),
            larger_domain_variance=vl, larger_simulated_variance=big,
            larger_sampling_check=bool(bounds[0]<=big/v<=bounds[1]),
            scope='fixed-background linear operator; actual refitted-background estimator is in joint ensemble'))
    # Normalized pixel second moments after cropping AND amplitude modulation;
    # sampling uncertainty uses tr(R^2), accounting for spatial dependence.
    loc = np.argwhere(c['measurement_domain']); rho = np.fft.ifft2(c['psd']).real
    dy = loc[:, 0, None]-loc[None, :, 0]; dx = loc[:, 1, None]-loc[None, :, 1]
    R = rho[dy % len(rho), dx % len(rho)]
    se = np.sqrt(2*np.sum(R*R)/(B*len(loc)**2))
    moment = float(np.mean([r['normalized_noise_second_moment'] for r in records]))
    return dict(operators=checks, modulation_second_moment=moment, expected_second_moment=1.,
        second_moment_MC_SE=se, second_moment_check=bool(abs(moment-1)<=5*se),
        larger_domain='same discrete modes at same physical frequencies on twice grid; exact covariance, no new inference/PSD repair; periodic representation remains; no physical long-mode claim',
        larger_random_coupling='independent streams; comparison tests sampling consistency only')


def paired_comparison(base, variant, reference, seed):
    b = joint_summary(base, reference); v = joint_summary(variant, reference)
    if not b['primary_available'] or not v['primary_available']:
        return dict(base=b, variant=v, MC_ratio=None)
    x = np.array([r['values'] for r in base]); y = np.array([r['values'] for r in variant])
    idx = np.random.default_rng(seed).integers(0, len(x), (1000, len(x)))
    ratios = y[idx].std(axis=1, ddof=1)/x[idx].std(axis=1, ddof=1)
    return dict(base=b, variant=v, scatter_ratio=y.std(axis=0, ddof=1)/x.std(axis=0, ddof=1),
                paired_bootstrap_MC95_ratio=np.quantile(ratios, [.025,.975], axis=0),
                coupling='same real-white inputs on same grid; model identity differs; all four variants retain grid',
                interval_scope='finite Monte Carlo precision, not observational confidence or model uncertainty')


def contribution_maps(root, plan, shape, hits):
    op=np.memmap(root/'operator-pixel.i32', dtype='<i4', mode='r'); maps={}
    for col in plan['ordered_column_domain']:
        pub=read_yaml(root/col['publication_file']); n=str(pub['network'])
        pixels=op[col['first_input_row']:col['first_input_row']+col['input_rows']]
        pixels=pixels[pixels>=0]
        maps.setdefault(n, np.zeros(shape, np.int64))
        maps[n] += np.bincount(pixels, minlength=int(np.prod(shape))).reshape(shape)
    require(np.array_equal(sum(maps.values()), hits), 'network contribution maps fail HITS reconciliation')
    return maps


def applicability(signal, coverage, header, original, model, contributions, weights):
    x,y=coordinates(signal.shape, header); bg=model['background_mask']; mean=model['summary']['mean_removed']
    n=y>=original[2]; source=np.hypot(x,y)<=40
    edges=np.array([r['coverage_min'] for r in model['summary']['bins']] +
                   [model['summary']['bins'][-1]['coverage_max_exclusive']])
    rows=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        for label, region in [('north', n), ('south', ~n)]:
            mask=bg & region & (coverage>=lo)&(coverage<hi)
            cv, pairs=pair_covariance(model['z'],mask,8,50)
            center=cv[8,8]; corr=cv/center if np.isfinite(center) and center>0 else cv
            totals={net:int(a[mask].sum()) for net,a in contributions.items()}; den=sum(totals.values())
            rows.append(dict(bin=i, region=label, coverage_range=[lo,hi], **stats(signal[mask]),
                second_moment_global_reference=float(np.mean((signal[mask]-mean)**2)) if mask.any() else None,
                normalized_second_moment=float(np.mean(model['z'][mask]**2)) if mask.any() else None,
                correlation=corr, pairs=pairs, network_fractions={net:v/den for net,v in totals.items()} if den else {}))
    regional={}
    for label, mask in [('source_fit', source), ('background', bg), ('north', bg&n), ('south', bg&~n)]:
        totals={net:int(a[mask].sum()) for net,a in contributions.items()}; den=sum(totals.values())
        regional[label]=dict(pixels=int(mask.sum()),coverage_quantiles=np.quantile(coverage[mask],[0,.1,.5,.9,1]) if mask.any() else None,
            network_fractions={net:v/den for net,v in totals.items()} if den else {},
            background=stats(signal[mask]) if label!='source_fit' else None)
    # Per-pixel network fractions are descriptive covariates, not variance sums.
    total=sum(contributions.values()); mix={k:np.divide(v,total,out=np.zeros(total.shape),where=total>0) for k,v in contributions.items()}
    half=weights.shape[-1]//2; row=int(header['CRPIX2']-1); col=int(header['CRPIX1']-1)
    weighted={net:float(np.sum(patch(v,row,col,half)*weights[0]**2)/np.sum(weights[0]**2)) for net,v in mix.items()}
    for label,mask in [('background',bg),('source_fit',source),('north',bg&n),('south',bg&~n)]:
        regional[label]['pixel_mixture_quantiles']={net:np.quantile(a[mask],[0,.1,.5,.9,1]) if mask.any() else None for net,a in mix.items()}
    fit_depth=coverage[source]; training=coverage[bg]
    return dict(regions=regional, overlapping_coverage_bins=rows, template_squared_weighted_source_mixture=weighted,
        source_coverage_background_percentiles=np.searchsorted(np.sort(training), fit_depth)/len(training),
        source_depth_valid_pixels=int((source&model['valid']).sum()), source_fit_pixels=int(source.sum()),
        network_statement='accepted contribution counts only; not independent network noise components; cross-network noise covariance unavailable',
        mixture_statement='coverage, position and mixture can be confounded; no causal attribution or mixture-aware model fitted',
        transfer_assumption='background amplitude/shared normalized covariance apply at source despite adaptive feedback; not demonstrated by these maps')


def blank_remeasure(signal, coverage, support, model, c, prior):
    half=c['q'].shape[0]//2; rows=[]; positions=[]; roi=np.any(c['weights']!=0,axis=0)
    for r in prior['trials']:
        y,x=r['row'],r['column']; bg=patch(model['background_mask'],y,x,half)
        ok=bg.shape==roi.shape and bool(np.all(bg[roi]))
        rec=dict(row=y,column=x,available=ok)
        if ok:
            d=patch(signal,y,x,half); scale=patch(model['scale'],y,x,half); valid=patch(model['valid'],y,x,half)
            rec.update(values=[float(np.sum(w*np.nan_to_num(d))) for w in c['weights']],
                predictions=[variance(model['psd'],scale,w) for w in c['weights']],
                amplitude_supported=bool(np.all(valid[roi])),
                template_squared_coverage=float(np.sum(c['q']**2*patch(coverage,y,x,half))/np.sum(c['q']**2)))
            positions.append((y,x))
        rows.append(rec)
    overlap=[]
    loc=np.argwhere(roi)-half
    sets=[set(map(tuple,loc+np.array(p))) for p in positions]
    for i in range(len(sets)):
        for j in range(i):
            overlap.append(dict(i=i,j=j,shared_pixels=len(sets[i]&sets[j])))
    selected=[r for r in rows if r['available']]
    return dict(locations=rows,available=len(selected),overlap=overlap,
        measured=[stats([r['values'][i] for r in selected]) for i in range(4)],
        predicted_RMS_scatter=[float(np.sqrt(np.mean([r['predictions'][i] for r in selected]))) if selected else None for i in range(4)],
        method='original empirical raw frozen projection/apertures, remeasured on this saved cycle; no source-fit background for pure-noise blanks',
        scope='fixed-background linear diagnostic only, distinct from actual source estimators with trial-refitted constant; overlapping placements not independent')


def verify_cycle_contributions(product, original):
    """Check network-resolved contribution maps, not only total HITS equality."""
    receipt=read_yaml(product/'receipt.yaml');shape=next(iter(original.values())).shape
    op=np.memmap(product/'operator-pixel.i32',dtype='<i4',mode='r');maps={}
    for s in receipt['segments']:
        if s['retained']==0:
            continue  # no MAP row interval is published for the inherited failed segment
        n=str(s['source_segment']['network']);start=s['MAP_first_input_row']
        p=op[start:start+s['retained']];p=p[p>=0]
        maps.setdefault(n,np.zeros(shape,np.int64))
        maps[n]+=np.bincount(p,minlength=int(np.prod(shape))).reshape(shape)
    require(set(maps)==set(original) and all(np.array_equal(maps[n],original[n]) for n in maps),
            'network contribution mixture differs across saved cycles')
    return dict(networks=sorted(maps,key=int),per_pixel_network_contributions_exact=True)
