"""Independent saved-parameter checks; deliberately disables additional least squares."""
import numpy as np
from common import HERE,OUT,DIAG,ARRAYS,read,write,preserve,verify_freeze
from run_readout import saved_gaussian

def forbidden(*args,**kwargs):raise RuntimeError('additional fits forbidden in verification')
np.linalg.lstsq=forbidden
verify_freeze();assert preserve()==read(HERE/'PRESERVATION_START.json')
rows=read(HERE/'READOUTS.json')
with np.load(OUT/'geometry.npz') as z:x,y,central=z['x'],z['y'],z['central']
witnesses=[];normal=[]
for row in rows:
    a=ARRAYS.index(row['array']);name=row['case'].split('_')[0]
    with np.load(OUT/row['case']/row['arm']/'pass06_maps.npz') as z:total=z['total'][a]
    with np.load(OUT/(name+'_truth.npz')) as z:truth=z['truth'][a]
    mask=central[a]&(np.hypot(x,y)<=row['radius_arcsec'])&np.isfinite(total)
    d=row['truth_template_origin'];coeff=np.array(row['coefficients'])
    p=np.r_[coeff[0],d['centroid'],np.log(d['widths']),np.deg2rad(25),coeff[1:]]
    assert len(p)==9 and np.isfinite(p).all() and max(abs(p[1:3]))<80 and np.all((np.exp(p[3:5])>4)&(np.exp(p[3:5])<60))
    predicted=saved_gaussian(p,x[mask],y[mask])
    template=truth[mask]/row['expected_peak']
    design=np.column_stack([template,np.ones(len(template)),x[mask]/90,y[mask]/90])
    np.testing.assert_allclose(predicted,np.sum(design*coeff,axis=1),rtol=1e-13,atol=1e-12)
    res=total[mask]-predicted
    sse=float(np.sum(res**2))
    np.testing.assert_allclose(sse,row['sse'],rtol=1e-13,atol=1e-8)
    normal_relative=np.linalg.norm(np.sum(design*res[:,None],axis=0))/max(np.linalg.norm(design)*np.linalg.norm(total[mask]),1e-30)
    assert normal_relative<1e-12
    normal.append(float(normal_relative))
    # Both models are feasible in the same Gaussian-plus-plane family.
    # Retain raw cost comparison, rather than refitting a nonlinear competitor.
    if row['free_fit']['sse']>sse:
        witnesses.append(dict(case=row['case'],arm=row['arm'],array=row['array'],radius_arcsec=row['radius_arcsec'],
            free_SSE=row['free_fit']['sse'],feasible_fixed_SSE=sse,relative_improvement=(row['free_fit']['sse']-sse)/row['free_fit']['sse'],
            original_peak_usable=row['original_peak_usable'],feasible_parameters=p))
oldtraces=read(DIAG/'TRACES.json')
failures={(q['case'],q['arm'],q['array']) for q in oldtraces if q['pass_index']==6 and q['case'].split('_')[0] in ['H','D'] and 'peak_domain_change_gt_1pct' in q['withholding_reasons']}
inner_witnesses={(q['case'],q['arm'],q['array']) for q in witnesses if q['radius_arcsec']==52}
write(HERE/'FIT_OBJECTIVE_WITNESSES.json',dict(witnesses=witnesses,all_H_D_domain_rejection_states_have_inner_witness=failures<=inner_witnesses,
    H_D_domain_rejection_states=len(failures),inner_witnesses=len(inner_witnesses),meaning='Known feasible lower-cost model; original fits and labels unchanged. No nonlinear refit was performed.'))
v=read(HERE/'VERIFICATION.json');v.update(independent_feasible_model_checks=48,independent_normal_equation_max=max(normal),
    saved_free_fit_lower_cost_witnesses=len(witnesses),inner_fit_witnesses=len(inner_witnesses),H_D_domain_rejections_with_witness=len(failures&inner_witnesses),
    linear_fits_after_verification=read(HERE/'COMPLETE.json')['linear_fits'])
write(HERE/'VERIFICATION.json',v)
print(v)
