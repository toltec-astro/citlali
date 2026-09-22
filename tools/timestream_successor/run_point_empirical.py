"""Reproducible empirical companion for the saved first-forward POINT measurement.

Example: python run_point_empirical.py --maps MAP_PRODUCTS --measurement
source-measurements.json --profile saved-descriptive-v1 --output NEW_DIRECTORY
"""
import argparse
from dataclasses import asdict
import json,os,platform,resource,sys,time,subprocess
from pathlib import Path
import numpy as np
import scipy,astropy
from astropy.table import Table
from point_empirical import Request,background,source_experiment,METHOD
from point_empirical_io import load_parent,split_maps,write_fits,dump
from point_empirical_plots import draw_background,draw_measurements,draw_splits
from inspect_forward_map import digest,require


def rows_for_measurements(array,description,exp):
    f=exp['fixed_template'];r=exp['refit'];rows=[]
    def add(name,value,unit,s,available,definition,bias=None):
        rows.append(dict(array=array,measurement=name,value=value,unit=unit,
            empirical_scatter=s['std'] if available else np.nan,
            empirical_bias=s['mean'] if bias is None and available else bias if bias is not None else np.nan,
            blank_count=s['n'],scatter_available=available,definition=definition))
    add('fixed_template_amplitude',f['source']['amplitude'] if f['source'] else np.nan,'mJy/beam',f['blank'],f['scatter_available'],
        'Raw sum(template * signal) / sum(template**2); no offset subtraction; shape and centroid frozen to saved descriptive fit; signed blank amplitudes. Bias column is blank mean.')
    add('descriptive_fit_amplitude',description['fitted_amplitude_above_constant_mJy_per_nominal_beam'],'mJy/beam',r['error'],r['scatter_available'],
        'Exact saved seven-parameter Gaussian plus constant; blank patch plus fitted source, all seven parameters refitted. Bias is mean recovered minus injected amplitude.')
    for a in exp['apertures']:
        add(f"aperture_{a['source']['radius_measured_fwhm']:g}_fwhm",a['source']['value_mJy_per_beam_arcsec2'] if a['source']['available'] else np.nan,'mJy arcsec2/beam',a['blank'],a['scatter_available'],
            'Pixel-center circular raw sum; radius in measured major FWHM; multiply pixel angular area; retain mJy/nominal-beam arcsec^2 because integrated-flux conversion is not authorized; no annulus subtraction or aperture correction. Bias column is blank mean.')
    return rows


def run(args):
    start=time.perf_counter();cpu=time.process_time();out=args.output.resolve()
    require(not out.exists(),'output already exists; companion never overwrites products')
    require(not out.is_relative_to(args.maps.resolve()),'companion output must be separate from input MAP products')
    if args.request:
        request=Request(**json.loads(args.request.read_text()))
    else:request=Request()
    measurement=json.loads(args.measurement.read_text())
    require(measurement['schema']=='citlali-point-source-description-v1' and measurement['observation']==152389,'foreign descriptive measurement')
    arrays={x['array']:x for x in measurement['arrays']}
    require(len(measurement['arrays'])==3 and set(arrays)=={'a1100','a1400','a2000'},'three per-array measurements required')
    out.mkdir(parents=True);dump(out/'request.json',asdict(request))
    code={f.name:digest(f) for f in Path(__file__).parent.glob('*point*.py')}
    code['inspect_forward_map.py']=digest(Path(__file__).parent/'inspect_forward_map.py')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    provenance=dict(method=METHOD,profile=args.profile,source_revision=revision,code_sha256=code,
        source_measurement_path=str(args.measurement.resolve()),source_measurement_sha256=digest(args.measurement),
        maps=str(args.maps.resolve()),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__),
        fit_authority='Owner-requested reuse of exact saved descriptive measurement; not the unavailable six-parameter SCI-POINT numerical method',
        fit_definition='Unweighted elliptical Gaussian plus constant; finite SCI_SUPPORT pixels within 40 arcsec of AST center; positive peak initialization within 20 arcsec; centroid bounds +/-20 arcsec, sigma bounds 0.5..20 arcsec, angle +/-pi, A>=0; trf exact dense solver, linear loss, 2-point Jacobian, x_scale=jac, tolerances1e-12,max_nfev3000',
        noise_scope='Empirical background and conditional map-domain measurement scatter; no covariance map, significance, inverse-variance weights, FRUIT decision or PTC recovery validation')
    dump(out/'provenance.json',provenance);table=[];summaries=[];parents={}
    for array in ('a1100','a1400','a2000'):
        dest=out/array;dest.mkdir();root=args.maps/array
        data,h,bundle,plan,desc,p,verified=load_parent(root,arrays[array]);parents[array]=digest(root/'map.fits')
        beam=bundle['nominal_beam_fwhm_arcsec'];support=data['SCI_SUPPORT'].astype(bool)
        major=max(p[3:5])*np.sqrt(8*np.log(2))
        bg=background(data['PRIMARY'],data['NORMALIZATION'],support,h,p[1:3],major,request)
        exp=source_experiment(data['PRIMARY'],data['NORMALIZATION'],support,h,beam,p,bg['background_mask'],request)
        splits=split_maps(root,plan,data,h,beam,p)
        bgreport={k:bg[k] for k in ['records','sensitivity','spatial']}
        bgreport.update(method='coverage-binned sample standard deviation (ddof=1), median and 1.482602218505602*MAD; no signal clipping',
            source_mask_radius_arcsec=request.source_mask_fwhm*major,units='mJy/nominal beam',
            background_pixels=int(bg['background_mask'].sum()),scale_valid_pixels=int(bg['valid'].sum()),
            validity='SCI_SUPPORT, finite input and coverage within adequately populated background bins; coverage-conditioned spatial pooling, not a per-pixel calibrated variance',
            source_parent=bundle['product_identity'])
        dump(dest/'background.json',bgreport);dump(dest/'measurements.json',exp);dump(dest/'split-report.json',splits['report']);dump(dest/'parent-verification.json',verified)
        write_fits(dest/'background-companion.fits',h,[('BG_STD',bg['scale'],'mJy/beam'),('BG_ROBUST',bg['robust_scale'],'mJy/beam'),('BG_MEDIAN',bg['median'],'mJy/beam'),
            ('VALID',bg['valid'].astype('u1'),'1'),('SOURCE_MASK',bg['source_mask'].astype('u1'),'1'),('BACKGROUND_MASK',bg['background_mask'].astype('u1'),'1'),('COVERAGE_BIN',bg['bin_index'].astype('i2'),'1')])
        write_fits(dest/'split-maps.fits',h,[('SPLIT_A',splits['signals'][0],'mJy/beam'),('SPLIT_B',splits['signals'][1],'mJy/beam'),('A_MINUS_B',splits['difference'],'mJy/beam'),
            ('HITS_A',splits['counts'][0],'count'),('HITS_B',splits['counts'][1],'count'),('A_FRACTION',splits['balance'],'1'),('COMMON_SUPPORT',splits['common'].astype('u1'),'1')])
        blankrows=[dict(row=t['row'],column=t['column'],coverage_ratio=t['coverage_ratio'],support_fraction=t['fit_support_fraction'],
            fixed_amplitude=t['fixed']['amplitude'] if t['fixed'] else np.nan) for t in exp['trials']]
        if exp['refit_trials']:Table(rows=[dict(row=t['row'],column=t['column'],coverage_ratio=t['coverage_ratio'],amplitude_error=t.get('amplitude_error',np.nan),status=t['status']) for t in exp['refit_trials']]).write(dest/'refit-locations.ecsv',format='ascii.ecsv')
        if blankrows:Table(rows=blankrows,meta={'definition':'Pixel rows/columns are zero-based; all amplitudes mJy/nominal beam; full trial measurements are in measurements.json.'}).write(dest/'blank-locations.ecsv',format='ascii.ecsv')
        table.extend(rows_for_measurements(array,desc,exp))
        draw_background(dest/'background.png',array,data,bg,beam);draw_measurements(dest/'measurements.png',array,exp,bg);draw_splits(dest/'splits.png',array,splits,h)
        summaries.append(dict(array=array,parent=bundle['product_identity'],description=desc,selection=exp['selection'],
            fixed_template=exp['fixed_template'],refit=exp['refit'],apertures=exp['apertures'],split=splits['report']))
        require(parents[array]==digest(root/'map.fits'),'original map changed during companion')
        print(f"{array}: {exp['selection']['selected_count']} blank locations, {exp['refit']['failed']} refit failures",flush=True)
    Table(rows=table,meta={'scope':provenance['noise_scope'],'method':METHOD}).write(out/'source-measurements.ecsv',format='ascii.ecsv')
    dump(out/'summary.json',summaries)
    elapsed=time.perf_counter()-start
    dump(out/'RUN.json',dict(exit_code=0,wall_seconds=elapsed,process_cpu_seconds=time.process_time()-cpu,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        runtime_scope='companion verification, PTC replay splits, empirical trials and rendering only; no RTC/PTC rerun',command=sys.argv,source_revision=revision,parent_maps_unchanged=parents))
    report=['# POINT 152389 empirical companion','',provenance['noise_scope']+'.','',
        'The existing six-plane maps are unchanged. The companion reuses the saved Gaussian-plus-constant descriptive measurement; it does not instantiate the unavailable SCI-POINT numerical method.','',
        '| Array | Measurement | Value | Empirical scatter | Blank mean / refit bias | Unit | Locations |','|---|---|---:|---:|---:|---|---:|']
    for row in table:
        scatter=f"{row['empirical_scatter']:.4g}" if row['scatter_available'] else 'unavailable'
        bias=f"{row['empirical_bias']:.4g}" if row['scatter_available'] else 'unavailable'
        report.append(f"| {row['array']} | {row['measurement']} | {row['value']:.4g} | {scatter} | {bias} | {row['unit']} | {row['blank_count']} |")
    report+=['','Scatter is a sample standard deviation, not a confidence interval or detection significance. Blank patches may overlap. The source-refit bias is relative to an injected copy of the fitted model; it does not measure bias relative to the true astronomical source. Apertures retain integrated brightness in mJy/nominal-beam arcsec²; no beam-area conversion, background subtraction or aperture correction is applied.','']
    for s in summaries:
        a=s['array'];sel=s['selection']
        report += [f'## {a}','',f"Selected {sel['selected_count']} locations from {sel['candidate_count']} coverage-matched candidates. A greedy subset has {sel['fit_footprint_disjoint_subset_count']} disjoint fixed-template footprints. These counts expose dependence; no effective independent sample count is claimed. The full 40 arcsec refit has {s['refit']['selection']['selected_count']} complete matched blank footprints.",'',
            f"[Background]({out/a/'background.png'}) · [Measurement diagnostics]({out/a/'measurements.png'}) · [Split maps]({out/a/'splits.png'})",'',
            f"[Background FITS]({out/a/'background-companion.fits'}) · [Split FITS]({out/a/'split-maps.fits'}) · [Full blank/refit records]({out/a/'measurements.json'})",'']
    report += [f'Companion wall time: {elapsed:.3f} s. Full numerical settings, code digests, dependency versions and source identities are in provenance.json and request.json.','',
        'Split A uses scans 0,2,4 and B uses 1,3,5 from existing accepted PTC products. Their raw difference is on common support. Coverage imbalance and source-shaped difference are reported; no independent/equal-depth noise scaling is assumed. Existing nw3 scan 5 failure remains absent.','',
        'Background-scale validity assumes spatial exchangeability within coverage bins; quadrant and mask-size diagnostics expose departures. No map weight, calibration, source-feedback model or FRUIT threshold is changed. Map-level validation does not establish source recovery through PTC.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    files=sorted(f for f in out.rglob('*') if f.is_file());(out/'FILES_SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.relative_to(out)}\n' for f in files))
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True);parser.add_argument('--measurement',type=Path,required=True)
    parser.add_argument('--profile',choices=['saved-descriptive-v1'],required=True)
    parser.add_argument('--request',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as e:
        print(f'POINT empirical companion failed: {e}',file=sys.stderr);raise
if __name__=='__main__':main()
