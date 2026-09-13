"""Exactly 48 truth-assisted linear readouts; no new reduction or nonlinear fit."""
import datetime
import platform
import resource
import subprocess
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from common import HERE,PRIOR,OUT,SEEDS,ARRAYS,read,write,digest,preserve,verify_freeze

FWHM=2*np.sqrt(2*np.log(2))

def saved_gaussian(p,x,y):
    amp,cx,cy,lx,ly,angle,b0,bx,by=p
    c,s=np.cos(angle),np.sin(angle)
    u=c*(x-cx)+s*(y-cy);v=-s*(x-cx)+c*(y-cy)
    return amp*np.exp(-.5*((u/(np.exp(lx)/FWHM))**2+(v/(np.exp(ly)/FWHM))**2))+b0+bx*x/90+by*y/90

def main():
    assert not (HERE/'START.json').exists()
    verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
    start=time.monotonic();rows=[];calls=0
    write(HERE/'START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),freeze_sha256=digest(HERE/'FREEZE.json'),
        python=platform.python_version(),numpy=np.__version__,linear_fit_limit=48))
    try:
        with threadpool_limits(limits=4):
            write(HERE/'THREAD_POOLS.json',threadpool_info())
            with np.load(OUT/'geometry.npz') as z:x,y,central,D=z['x'],z['y'],z['central'],z['D']
            np.testing.assert_array_equal(central,D&(np.hypot(x,y)<=60))
            stored={(r['case'],r['arm'],r['pass_index']):r for r in read(OUT/'RECEIPTS.json')}
            definitions=read(PRIOR/'CASES.json')['states']
            for name in ['H','D']:
                with np.load(OUT/(name+'_truth.npz')) as z:truth=z['truth']
                for seed in SEEDS:
                    for arm in ['P','C']:
                        case=f'{name}_{seed}';old=stored[case,arm,6]
                        with np.load(OUT/case/arm/'pass06_maps.npz') as z:maps=z['total']
                        for a,array in enumerate(ARRAYS):
                            expected=definitions[name]['peak'][a]
                            template=truth[a]/expected
                            m=old['measurement'][a]
                            for radius in [60,52]:
                                mask=central[a]&(np.hypot(x,y)<=radius)&np.isfinite(maps[a])
                                design=np.column_stack([template[mask],np.ones(int(mask.sum())),x[mask]/90,y[mask]/90])
                                values=maps[a,mask]
                                fit=m['original']['fit'] if radius==60 else m['fixed_inner_domain_probe']['fit']
                                comparator=saved_gaussian(fit['parameters'],x[mask],y[mask])
                                comparator_sse=float(np.sum((values-comparator)**2))
                                np.testing.assert_allclose(comparator_sse,fit['sse'],rtol=2e-13,atol=1e-8)
                                row=dict(case=case,arm=arm,array=array,pass_index=6,radius_arcsec=radius,pixels=int(mask.sum()),
                                    expected_peak=expected,template_sampled_max=float(np.max(template[mask])),
                                    truth_template_origin=definitions[name],
                                    original_peak_usable=m['judgments']['peak_response_usable'],original_judgments=m['judgments'],
                                    original_probe_available=m['fixed_inner_domain_probe']['available'],
                                    original_peak_score=m['original']['empirical_peak_score'],
                                    outer_background=m['original']['background'],free_fit=fit,free_fit_sse_reproduced=comparator_sse,
                                    free_relative_peak_error=fit['peak']/expected-1)
                                calls+=1;assert calls<=48
                                t=time.monotonic()
                                try:
                                    coeff,residuals,rank,singular=np.linalg.lstsq(design,values,rcond=None)
                                    reconstructed=np.sum(design*coeff,axis=1)
                                    residual=values-reconstructed
                                    normal=np.sum(design*residual[:,None],axis=0)
                                    scale=np.linalg.norm(design,ord='fro')*np.linalg.norm(values)
                                    normal_relative=float(np.linalg.norm(normal)/max(scale,1e-30))
                                    finite=bool(np.isfinite(coeff).all() and np.isfinite(singular).all() and np.isfinite(residual).all())
                                    available=finite and rank==4
                                    row.update(available=bool(available),reason='finite_full_rank' if available else 'nonfinite_or_rank_deficient',
                                        coefficients=coeff,rank=int(rank),singular_values=singular,
                                        condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                                        residual_norm=float(np.linalg.norm(residual)),sse=float(np.sum(residual**2)),
                                        normal_equation_residual=normal,relative_normal_equation_residual=normal_relative,
                                        reported_residual_sum=residuals,seconds=time.monotonic()-t)
                                    if available:
                                        row.update(amplitude=float(coeff[0]),relative_peak_error=float(coeff[0]/expected-1),
                                            background=coeff[1:],fitted_background_at_truth_center=float(coeff[1]+coeff[2]*definitions[name]['centroid'][0]/90+coeff[3]*definitions[name]['centroid'][1]/90),
                                            free_background_at_truth_center=float(fit['background'][0]+fit['background'][1]*definitions[name]['centroid'][0]/90+fit['background'][2]*definitions[name]['centroid'][1]/90),
                                            free_SSE_minus_fixed_SSE=float(comparator_sse-np.sum(residual**2)))
                                        assert normal_relative<1e-12
                                except np.linalg.LinAlgError as err:
                                    row.update(available=False,reason='linear_algebra_failure',error=str(err),seconds=time.monotonic()-t)
                                rows.append(row)
            assert calls==48 and len(rows)==48
            write(HERE/'READOUTS.json',rows)
            verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
            write(HERE/'COMPLETE.json',dict(linear_fits=calls,available=sum(r['available'] for r in rows),
                wall_seconds=time.monotonic()-start,sum_linear_fit_seconds=sum(r['seconds'] for r in rows),peak_rss_bytes=rss))
            write(HERE/'VERIFICATION.json',dict(linear_fit_calls=calls,saved_free_fit_cost_checks=48,
                original_labels_unchanged=True,normal_equations_checked=sum(r['available'] for r in rows),
                maximum_relative_normal_residual=max(r.get('relative_normal_equation_residual',0) for r in rows),
                previous_results_preserved=True,verification_refits=0,PTC_calls=0,feedback_fits=0,free_Gaussian_refits=0,
                pre_PTC_reads=0,threshold_changes=0,reserved_observations=0,candidate='parked_for_POINT'))
            print(read(HERE/'COMPLETE.json'))
    except BaseException as err:
        write(HERE/'FAILURE.json',dict(error=str(err),traceback=traceback.format_exc(),linear_fits_started=calls))
        write(HERE/'PARTIAL_READOUTS.json',rows)
        raise
if __name__=='__main__':main()
