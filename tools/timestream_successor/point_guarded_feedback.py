"""Fixed guarded measurement/admission boundary for the bounded POINT campaign.

No PTC, noise learning or historical fitter. Brightness mJy/nominal beam;
coordinates and widths arcsec; fixed apertures retain original M0 geometry.
"""
import numpy as np
from point_expected import infer_expected, METHOD
from point_corr_admission import select
from point_admission import projection
from point_empirical import aperture_measurements, stats
from point_response_cd import save_candidate, vector_measurement, MEASUREMENT_NAMES, RATIO_FLOOR
from point_fruit_trajectory import bound, candidate_guard
from point_empirical_io import write_fits
from point_jinc_campaign import exact_header
from point_additive_diagnostics import save_json
from inspect_forward_map import require

POLICIES=('BASE-POINT','CORR','DIAG')
BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder')


def measurement(c,signal,support,guard):
    """Describe the already executed guard fit; never fit a second time."""
    f=c['fixed'];p=guard['accepted_parameters'];good=guard['accepted']
    fit=dict(status='success' if good else guard['status'],method=METHOD,
             cause=guard['cause'],constrained=bool(guard['diagnostic'] is not None and any(guard['diagnostic']['active_mask'])))
    if good:fit['parameters']=p
    relative=signal-p[6] if good else None
    aperture=aperture_measurements(relative,support,f.x,f.y,f.parameters[1:3],f.major,f.beam,4.) if good else [dict(value_mJy_per_beam_arcsec2=None) for _ in range(3)]
    raw=aperture_measurements(signal,support,f.x,f.y,f.parameters[1:3],f.major,f.beam,4.)
    return dict(fit=fit,guard=guard,frozen_template_projection=projection(relative,support,f.weights) if good else None,
        apertures=aperture,raw_apertures=raw,
        aperture_background_terms=[r['value_mJy_per_beam_arcsec2']-v['value_mJy_per_beam_arcsec2'] if good and r['value_mJy_per_beam_arcsec2'] is not None and v['value_mJy_per_beam_arcsec2'] is not None else None for r,v in zip(raw,aperture)],
        background=stats(signal[support&f.background]),
        profile=[float(np.mean(signal[m])) if m.any() else None for m in [support&(f.radius>=lo)&(f.radius<hi) for lo,hi in zip(f.edges[:-1],f.edges[1:])]],
        uncertainty_scope='guard conditional full frozen C0 reference-scale sigma_A; not endpoint empirical uncertainty')


def infer(c,signal,support):
    prepared=infer_expected(signal,support,c['fixed'],c['neighborhood'],c['scale'],c['psd'])
    return measurement(c,signal,support,prepared[0]),prepared


def publish(dest,c,signal,prepared,policy,tau):
    dest.mkdir(exist_ok=True,parents=True);guard,base,fields=prepared
    require(policy in POLICIES,'unknown guarded policy')
    require(policy=='BASE-POINT' or tau is not None and np.isfinite(tau) and tau>=3,'missing frozen threshold')
    bad=fields['policies'][policy]['numerical_failures'] if fields is not None and policy!='BASE-POINT' else 0
    result=base if policy=='BASE-POINT' else select(base,fields,c['k0'],policy,tau)
    selected=result['selected'];valid=result['valid'];h=exact_header(c['header'])
    if guard['accepted']:
        require(np.array_equal(result['candidate'][selected],(signal-guard['accepted_parameters'][6])[selected]),'candidate changed measured brightness')
    require(np.all(result['candidate'][valid&~selected]==0),'excluded valid pixels not intentional zero')
    path=dest/'candidate.fits';save_candidate(path,h,result)
    summary=dict(method=METHOD,policy=policy,threshold=tau,guard=guard,BASE=base['summary'],candidate=result['summary'],
        numerical_failures=bad,known=guard['complete'] and bad==0,
        selected_pixels=int(selected.sum()),brightness_area=float(np.sum(result['candidate'][selected])*4),
        baseline_pixels=int(base['selected'].sum()),proposed_pixels=int(result.get('proposed_extra',np.zeros_like(selected)).sum()),
        added_pixels=int(result.get('added',np.zeros_like(selected)).sum()),
        extra_brightness_area=float(np.sum(result['candidate'][result.get('added',np.zeros_like(selected))])*4))
    planes=[('MEASURED',signal,'mJy/nominal beam'),('BASE_POINT',base['selected'].astype('u1'),'1'),('K0',c['k0'].astype('u1'),'1'),('BRIGHTNESS',base['brightness'],'mJy/nominal beam'),('VALID',valid.astype('u1'),'1'),('ADMITTED',selected.astype('u1'),'1')]
    for name,key in [('PROPOSED','proposed_extra'),('ADDED','added')]:planes.append((name,result.get(key,np.zeros_like(selected)).astype('u1'),'1'))
    if 'noise_high_extrapolated' in c:
        planes.extend([('NOISE_EXTRAP',c['noise_high_extrapolated'].astype('u1'),'1'),('COV_EXTRAP',c['covariance_high_extrapolated'].astype('u1'),'1')])
    if fields is not None:
        planes.extend([('NEFF_W',fields['neff_weights'],'1'),('EVIDENCE',fields['evidence'],'mJy/nominal beam')])
        for pol in ('CORR','DIAG'):
            q=fields['policies'][pol];planes.extend([(pol+'_SCORE',q['score'],'1'),(pol+'_VAR',q['variance'],'(mJy/nominal beam)^2'),(pol+'_VALID',q['available'].astype('u1'),'1')])
    write_fits(dest/'admission.fits',h,planes,companion='POINT-GUARDED-DIAGNOSTIC');save_json(dest/'admission.json',summary)
    return path,summary


def continuation_reason(guard,summary,candidate,header):
    if not guard['complete']:return 'computational_unavailable',guard['message']
    if not guard['accepted']:return 'source_retention_miss',guard['message']
    if summary['numerical_failures']:return 'computational_unavailable','supplemental neighborhood numerical failure'
    try:candidate_guard(candidate,header)
    except ValueError as exc:return 'candidate_boundary_or_support',str(exc)
    return None,None


def ideal_comparison(c,reference,actual,ideal):
    """Compare stored full-map measurements; caller performs only the ideal fit."""
    b,a,i=map(vector_measurement,(reference,actual,ideal))
    rows=None
    if all(v is not None for v in (b,a,i)):
        rows=[dict(name=n,actual_increment=float(da),ideal_increment=float(di),absolute_discrepancy=float(da-di),
                   ratio=float(da/di) if np.isfinite(da+di) and abs(di)>RATIO_FLOOR else None)
              for n,da,di in zip(MEASUREMENT_NAMES,a-b,i-b)]
    raw=[]
    for j,factor in enumerate((1.,1.5,2.)):
        q=[m['raw_apertures'][j]['value_mJy_per_beam_arcsec2'] for m in (reference,actual,ideal)]
        bg=[m['aperture_background_terms'][j] for m in (reference,actual,ideal)]
        raw.append(dict(factor=factor,actual_raw_increment=q[1]-q[0] if None not in q[:2] else None,
            ideal_raw_increment=q[2]-q[0] if q[0] is not None and q[2] is not None else None,
            actual_background_increment=bg[1]-bg[0] if None not in bg[:2] else None,
            ideal_background_increment=bg[2]-bg[0] if bg[0] is not None and bg[2] is not None else None))
    return dict(comparisons=rows,reference=reference,actual=actual,ideal_map_addition=ideal,
                raw_aperture_and_background_increments=raw,ratio_floor_native_units=RATIO_FLOOR)
