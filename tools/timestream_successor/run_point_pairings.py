"""Reconcile POINT processing chunks and compare every balanced reconstruction.

This is a read-only successor empirical companion extension, not a cleaning run.
"""
import argparse,json,platform,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy,astropy
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspect_forward_map import digest,require
from point_empirical_io import load_parent,write_fits,dump
from point_pairings import balanced_partitions,footprints,normalized_sum,measured,compared,population_summary,population_comparison
from point_pairings_io import reconcile_request,replay_scans


def label(scans):return '+'.join(map(str,scans))


def plots(dest,array,groups,pairs,products,header,regions):
    singles=[g for g in groups if len(g['scans'])==1]
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    any_measurements=False
    for key,marker in [('fit','o'),('fixed_template','s')]:
        vals=[g['measurement']['fit'].get('fitted_amplitude_above_constant_mJy_per_nominal_beam',np.nan) if key=='fit' else (g['measurement'][key] or {}).get('amplitude',np.nan) for g in singles]
        ax[0].plot([g['scans'][0] for g in singles],vals,marker+'-',label=key.replace('_',' '))
        any_measurements |= bool(np.isfinite(vals).any())
    ax[0].set(xlabel='Processing chunk ID',ylabel='mJy / nominal beam',title='Complete-footprint individual measurements');ax[0].legend()
    ax[0].set(xticks=[g['scans'][0] for g in singles],xlim=(-.2,len(singles)-.8))
    if not any_measurements:
        ax[0].set(yticks=[])
        ax[0].text(.5,.5,'No individual chunk fully supports\nthe fit or fixed-template footprint',ha='center',va='center',transform=ax[0].transAxes)
    for name in ['core','template','fit','fit_periphery']:
        ax[1].plot([g['scans'][0] for g in singles],[g['measurement']['coverage'][name]['supported_fraction'] for g in singles],'o-',label=name.replace('_',' '))
    ax[1].set(xlabel='Processing chunk ID',ylabel='Supported fraction',ylim=(0,1.03),title='Each chunk’s own coverage cut');ax[1].legend(fontsize=8)
    fig.suptitle(f'{array} · first-pass reconstruction · missing measurements stay unavailable',fontsize=12)
    fig.savefig(dest/'individual-chunks.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(10,17),layout='constrained')
    shape=next(iter(products.values()))['signal'].shape
    dx=header['CDELT1']*3600;dy=header['CDELT2']*3600
    extent=[(.5-header['CRPIX1'])*dx,(shape[1]+.5-header['CRPIX1'])*dx,(.5-header['CRPIX2'])*dy,(shape[0]+.5-header['CRPIX2'])*dy]
    differences=[np.where(products[p['A']]['support']&products[p['B']]['support'],products[p['A']]['signal']-products[p['B']]['signal'],np.nan) for p in pairs]
    zoom=regions['fit'];vals=np.concatenate([abs(d[zoom&np.isfinite(d)]) for d in differences]);limit=np.percentile(vals,99) if len(vals) else 1
    for ax,p,d in zip(axes.ravel(),pairs,differences):
        im=ax.imshow(d,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ok=p['comparison']['fit_comparison_available'];v=p['comparison']['fixed_template_A_minus_B']
        text=f"template Δ={v['amplitude']:.1f}" if v else 'template unavailable'
        ax.set(title=f"{p['A']} − {p['B']}\nfit {'supported' if ok else 'unavailable'}; {text}",xlim=(40,-40),ylim=(-40,40),xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
        ax.set_title(ax.get_title(),fontsize=10)
    for ax in axes.ravel()[len(pairs):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axes.ravel()),label='A − B (mJy / nominal beam)',shrink=.7)
    fig.suptitle(f'{array} · all balanced processing-chunk pairings\nFirst-pass reconstruction differences; relative cleaning attenuation unresolved',fontsize=12)
    fig.savefig(dest/'balanced-differences.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,9),layout='constrained');names=['core','template','fit','fit_periphery'];subsets=[g for g in groups if len(g['scans'])>1]
    missing=np.array([[g['measurement']['coverage'][name]['expected_pixels']-g['measurement']['coverage'][name]['supported_pixels'] for name in names] for g in subsets])
    median=np.array([[g['measurement']['coverage'][name]['normalization_quantiles_including_zero'][2] for name in names] for g in subsets])
    for panel,values,title in zip(ax,[missing,median],['Unsupported pixels in fixed footprints','Median normalization / hits']):
        im=panel.imshow(values,aspect='auto',cmap='YlOrRd' if panel==ax[0] else 'viridis',**({'vmin':0,'vmax':max(1,missing.max())} if panel==ax[0] else {}));panel.set(xticks=range(len(names)),xticklabels=[n.replace('_','\n') for n in names],yticks=range(len(subsets)),yticklabels=[g['name'] for g in subsets],title=title)
        fig.colorbar(im,ax=panel,shrink=.7)
        if panel==ax[0]:
            for i in range(len(subsets)):
                for j in range(len(names)):panel.text(j,i,str(missing[i,j]),ha='center',va='center',fontsize=8,color='black' if missing[i,j]<.6*max(1,missing.max()) else 'white')
    fig.suptitle(f'{array} · local footprint coverage · complete support is necessary, not a response-equivalence test',fontsize=11)
    fig.savefig(dest/'local-coverage.png',dpi=150);plt.close(fig)


def run(args):
    start=time.perf_counter();cpu=time.process_time();out=args.output.resolve()
    require(not out.exists() and not out.is_relative_to(args.maps.resolve()),'output must be new and separate from MAP inputs')
    reconciliation=reconcile_request(args.processing_request)
    saved=json.loads(args.measurement.read_text());require(saved['schema']=='citlali-point-source-description-v1' and saved['observation']==152389,'foreign saved description')
    byarray={d['array']:d for d in saved['arrays']};require(set(byarray)=={'a1100','a1400','a2000'} and len(saved['arrays'])==3,'three arrays required')
    out.mkdir(parents=True);source=Path(__file__).parent;revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    names=['run_point_pairings.py','point_pairings.py','point_pairings_io.py','point_empirical.py','point_empirical_io.py','inspect_point_source.py','inspect_forward_map.py']
    provenance=dict(schema='citlali-point-balanced-reconstructions-v1',source_revision=revision,code_sha256={n:digest(source/n) for n in names},
        measurement=dict(path=str(args.measurement.resolve()),sha256=digest(args.measurement)),maps=str(args.maps.resolve()),profile=args.profile,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),
        definitions=dict(group_map='sum original admitted numerators / sum original admitted uniform normalizations; never mean normalized images',
            support='parent coverage-cut policy evaluated from each subset own normalization; complete declared measurement footprints required',
            fit='exact saved seven-parameter Gaussian plus constant and original bounds/initialization/objective; full 40 arcsec AST-centered disk required',
            template='full-map fitted shape/centroid held fixed; raw sum(t*M)/sum(t*t) on radius 3 measured major FWHM',
            apertures='full-map centroid/radii held fixed at 1,1.5,2 measured major FWHM; raw sum * pixel area; no background subtraction or flux conversion',
            interpretation='first-pass reconstruction differences; unequal background/coverage and unresolved different cleaning responses; no final uncertainty or significance'))
    dump(out/'provenance.json',provenance);dump(out/'scan-identity.json',reconciliation);allarrays=[];rows=[];timing=[];originals={}
    for array in ('a1100','a1400','a2000'):
        root=args.maps/array;dest=out/array;dest.mkdir();full,h,bundle,plan,desc,p,verification=load_parent(root,byarray[array]);originals[array]=digest(root/'map.fits')
        regions,template,major,x,y=footprints(full['PRIMARY'].shape,h,p);n,q,scans,audit=replay_scans(root,plan,full,regions,reconciliation)
        partitions=balanced_partitions(scans);timing.extend(audit['timing']);dump(dest/'replay-audit.json',audit);dump(dest/'parent-verification.json',verification)
        write_fits(dest/'measurement-footprints.fits',h,[(name.upper(),r.astype('u1'),'1') for name,r in regions.items()])
        subsets=[(s,) for s in scans]+sorted({s for pair in partitions for s in pair});groups=[];products={};measurements={};populations={}
        for subset in subsets:
            name=label(subset);product=normalized_sum(n,q,[scans.index(s) for s in subset],bundle['coverage_cut'])
            m=measured(product,regions,template,h,bundle['nominal_beam_fwhm_arcsec'],p,major,x,y)
            pop=population_summary(audit['members'],subset,regions)
            for region in regions:
                require(sum(v['contributions'] for v in pop[region].values())==m['coverage'][region]['normalization_total'],'local population does not recover hits')
            group=dict(name=name,scans=list(subset),measurement=m,populations=pop,
                missing_parent_segments=[dict(network=s['network'],scan=s['scan'],cause=s['cause']) for s in audit['missing_segments'] if s['scan'] in subset])
            groups.append(group);products[name]=product;measurements[name]=m;populations[name]=pop
            write_fits(dest/f'subset-{name}.fits',h,[('SIGNAL',product['signal'],'mJy/beam'),('NUMERATOR',product['numerator'],'mJy/beam'),('NORMALIZATION',product['normalization'],'1'),('SCI_SUPPORT',product['support'].astype('u1'),'1')])
            rows.append(dict(array=array,subset=name,kind='individual' if len(subset)==1 else 'balanced-half',
                fit_amplitude=m['fit'].get('fitted_amplitude_above_constant_mJy_per_nominal_beam',np.nan),fit_available=m['fit']['status']=='success',
                fit_background=m['fit'].get('fitted_constant_mJy_per_nominal_beam',np.nan),fixed_amplitude=(m['fixed_template'] or {}).get('amplitude',np.nan),
                fit_supported_pixels=m['coverage']['fit']['supported_pixels'],fit_expected_pixels=m['coverage']['fit']['expected_pixels'],
                fit_median_hits=m['coverage']['fit']['normalization_quantiles_including_zero'][2],template_supported_pixels=m['coverage']['template']['supported_pixels']))
        pairs=[]
        for a,b in partitions:
            na,nb=label(a),label(b);d,common,comparison=compared(products[na],products[nb],measurements[na],measurements[nb],regions,template)
            comparison['local_populations']=population_comparison(populations[na],populations[nb])
            pairs.append(dict(A=na,B=nb,comparison=comparison))
            qa,qb=products[na]['normalization'],products[nb]['normalization'];ratio=np.full(qa.shape,np.nan);np.divide(qa,qb,out=ratio,where=qb>0)
            write_fits(dest/f'difference-{na}-vs-{nb}.fits',h,[('A_MINUS_B',d,'mJy/beam'),('COMMON_SUPPORT',common.astype('u1'),'1'),('Q_A',qa,'1'),('Q_B',qb,'1'),('Q_A_OVER_Q_B',ratio,'1')])
        dump(dest/'groups.json',groups);dump(dest/'comparisons.json',pairs);plots(dest,array,groups,pairs,products,h,regions)
        allarrays.append(dict(array=array,scans=scans,pairings=pairs,groups=[{k:v for k,v in g.items() if k!='populations'} for g in groups]))
        require(digest(root/'map.fits')==originals[array],'input map mutated')
        print(f'{array}: {len(scans)} processing chunks, {len(partitions)} complementary partitions, {sum(g["measurement"]["fit"]["status"]=="success" for g in groups)}/{len(groups)} complete-footprint fits',flush=True)
    Table(rows=rows,meta={'amplitude_unit':'mJy/nominal beam','scope':provenance['definitions']['interpretation']}).write(out/'source-measurements.ecsv',format='ascii.ecsv')
    # Network rows are retained separately: native clocks/intervals need not be
    # silently treated as identical across networks.
    dump(out/'scan-timing.json',timing);dump(out/'summary.json',allarrays)
    elapsed=time.perf_counter()-start;dump(out/'RUN.json',dict(source_revision=revision,wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),command=sys.argv,parent_maps_unchanged=originals,
        scope='replay verified saved contributions, measurements and rendering; no RTC/PTC relearning'))
    report=['# POINT 152389: balanced first-pass reconstruction comparisons','',provenance['definitions']['interpretation']+'.','',
        'The bound request forces 10-second processing chunks. The realized data have six processing IDs (0–5) within telescope observation 152389_0_2 (raw ScanNum=2). These are not six separately identified telescope observations. No four-scan grouping is identified in the bound telescope/processing records; no two chunks were discarded. All ten complementary three-versus-three pairings are included.','',
        'UTC ranges below are first/last native science samples across networks; exact per-network rows/times and unquantified absolute-epoch uncertainty are retained in scan-timing.json. PTC slot membership is checked against the actual RTC factor/phase (a2000 uses factor 2); the replay audit records its first/last RTC sample times separately.','',
        '| Chunk | First science sample UTC | Last science sample UTC |','|---|---|---|']
    for s in allarrays[0]['scans']:
        ts=[t for t in timing if t['scan']==s];report.append(f"| {s} | {min(t['first_sample_UTC'] for t in ts)} | {max(t['last_sample_UTC'] for t in ts)} |")
    report+=['','Each subset uses its own coverage cut. The fixed 40″ fit disk must be completely supported; an unsupported fit is unavailable, never silently refitted on an intersection. Template/aperture footprints retain the full-map centroid and measured radii. Complete support does not establish equal precision or equal cleaning response. The peripheral fit-region diagnostics are not an annulus subtraction.','',
        'All amplitudes are mJy/nominal beam. Apertures in groups.json remain mJy/nominal beam × arcsec². Each subset FITS stores signal, numerator, normalization and support. Each difference FITS stores A−B, common support, both normalizations and their ratio. Measurement masks are stored separately.','']
    for array in allarrays:
        a=array['array'];report += [f'## {a}','',f"[Individual chunks]({out/a/'individual-chunks.png'}) · [All differences]({out/a/'balanced-differences.png'}) · [Local coverage]({out/a/'local-coverage.png'})",'',
            '| Subset | Fitted amplitude | Fixed template | Fit support | Fit Q min / median / max |','|---|---:|---:|---:|---|']
        for g in array['groups']:
            m=g['measurement'];f=m['fit'];fit=f"{f['fitted_amplitude_above_constant_mJy_per_nominal_beam']:.2f}" if f['status']=='success' else 'unavailable';fixed=f"{m['fixed_template']['amplitude']:.2f}" if m['fixed_template'] else 'unavailable';c=m['coverage']['fit'];qs=c['normalization_quantiles_including_zero']
            report.append(f"| [{g['name']}]({out/a/('subset-'+g['name']+'.fits')}) | {fit} | {fixed} | {c['supported_pixels']}/{c['expected_pixels']} | {qs[0]:g} / {qs[2]:g} / {qs[4]:g} |")
        report+=['','| A versus B | Fitted A−B | Template A−B | Local fit QA/QB (10/50/90%) | Core detectors A/B/common |','|---|---:|---:|---|---|']
        for p in array['pairings']:
            c=p['comparison'];fit=f"{c['fitted_amplitude_A_minus_B']:.2f}" if c['fit_comparison_available'] else 'unavailable';f=c['fixed_template_A_minus_B'];fixed=f"{f['amplitude']:.2f}" if f else 'unavailable';q=c['local_coverage']['fit']['ratio_QA_QB_positive_pixels']
            pop=c['local_populations']['core']
            report.append(f"| [{p['A']} versus {p['B']}]({out/a/('difference-'+p['A']+'-vs-'+p['B']+'.fits')}) | {fit} | {fixed} | {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} | {pop['detectors_A']}/{pop['detectors_B']}/{pop['detectors_both']} |")
        report += ['',f"[Full measurements and local detector populations]({out/a/'groups.json'}) · [Pair coverage and availability]({out/a/'comparisons.json'}) · [Exact replay, detector identities, exclusions and timing]({out/a/'replay-audit.json'})",'']
    report += ['Existing parent exclusions are preserved: raw networks 6/10 were not supplied, and nw3 processing chunk 5 failed its PTC fit. Per-detector PTC cause-code counts are whole-segment facts; excluded samples have no serialized MAP coordinates, so no local source-position attribution is invented. Per-footprint contribution counts describe accepted samples exactly.','',
        'These pairings reuse the same observation. Different inherited PTC operators can have different source responses; the maps do not measure that attenuation. No noise-only null, final flux uncertainty, calibration instability, variability, or independence claim follows. No demand for pre-FRUIT agreement is imposed.','',f'Companion-only runtime: {elapsed:.3f} s. Original maps, calibration, cleaning, coefficients and FRUIT remain unchanged. Exact identities and environment are in provenance.json and RUN.json.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    files=sorted(f for f in out.rglob('*') if f.is_file());(out/'FILES_SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.relative_to(out)}\n' for f in files))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--maps',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True)
    p.add_argument('--processing-request',type=Path,required=True);p.add_argument('--profile',choices=['saved-descriptive-v1'],required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
if __name__=='__main__':main()
