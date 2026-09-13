"""Descriptive metrics only; no estimator or acceptance-rule changes."""
import numpy as np
from scipy.ndimage import label
from common import b, candidate, source_core

def energy(x):
    value = float(np.sum(np.asarray(x,float)**2))
    if not np.isfinite(value):
        raise ValueError('nonfinite diagnostic energy')
    return value

def fraction(num, den):
    return float(num/den) if den > 0 else None

def maxima(data,e,u,original):
    peak=float(np.max(u[e.D]))
    if peak == 0:
        return dict(peak=0.,position=None,reason='zero_model_has_no_unique_source_maximum')
    ids=np.flatnonzero(e.D & (u==peak));i=int(ids[0])
    center=original.get('fit',{}).get('centroid')
    return dict(peak=peak,position=[float(data.x[i]),float(data.ygrid[i])],ties=len(ids),
        pixel_index=i,coverage=float(data.Q[e.array,i]),
        coverage_percentile_in_domain=float(np.mean(data.Q[e.array,e.D]<=data.Q[e.array,i])),
        radius_arcsec=float(np.hypot(data.x[i],data.ygrid[i])),
        separation_from_original_fit_arcsec=None if center is None else float(np.hypot(data.x[i]-center[0],data.ygrid[i]-center[1])),
        complete_wavelet_footprint=e.valid[:,i])

def image_problem(data,e,total,n,t,original,pair):
    delta=t-n;D=e.D;norm=energy(delta[D]);core=source_core(data,e,original)
    Y,beta=e.residual(total)
    sigma=e.sigmas[:,e.stratum]
    target=candidate.analysis(Y.reshape(e.shape)).reshape(5,-1)
    omega=e.valid&D&(abs(target)>=5*sigma)
    wn=candidate.analysis(n.reshape(e.shape)).reshape(5,-1)
    wt=candidate.analysis(t.reshape(e.shape)).reshape(5,-1)
    wd=candidate.analysis(delta.reshape(e.shape)).reshape(5,-1)
    constrained_target=np.where(omega,target/sigma,0.)
    constrained_delta=np.where(omega,wd/sigma,0.)
    fn=.5*energy(np.where(omega,(wn-target)/sigma,0.))
    ft=.5*energy(np.where(omega,(wt-target)/sigma,0.))
    f0=.5*energy(constrained_target)
    if not pair['empty']:
        np.testing.assert_allclose([fn,ft],[pair['snapshots'][k]['objective'] for k in ['declared','tight']],rtol=1e-9,atol=1e-9)
    top_count=max(1,int(np.ceil(.01*D.sum())))
    top=np.partition(delta[D]**2,-top_count)[-top_count:]
    bright=D&(abs(delta)>=.1*np.max(abs(delta[D]))) if norm>0 else np.zeros_like(D)
    components,count=label(bright.reshape(e.shape))
    sizes=np.bincount(components.ravel())[1:]
    rim=D&(np.hypot(data.x,data.ygrid)>52)
    low=D&(e.stratum==0)
    return dict(empty=pair['empty'],original_fit=original.get('fit'),background=beta,
        nominal_maximum=maxima(data,e,n,original),tight_maximum=maxima(data,e,t,original),
        difference_L2=float(np.sqrt(norm)),relative_image_L2=fraction(np.sqrt(norm),np.sqrt(energy(t[D]))),
        difference_energy=norm,core_pixels=int(core.sum()),core_energy_fraction=fraction(energy(delta[core]),norm),
        complement_energy_fraction=fraction(energy(delta[D&~core]),norm),rim_energy_fraction=fraction(energy(delta[rim]),norm),
        low_coverage_energy_fraction=fraction(energy(delta[low]),norm),top_one_percent_pixel_energy_fraction=fraction(float(top.sum()),norm),
        change_components=dict(threshold_fraction=.1,connectivity=4,components=count,
            single_pixel_components=int(np.sum(sizes==1)),largest_component_pixels=int(sizes.max()) if count else 0),
        scale_bands=[dict(band=j+1,full_domain_difference_energy=energy(wd[j,D]),
            source_core_difference_energy=energy(wd[j,core]),selected_coefficients=int(omega[j].sum()),
            selected_scaled_difference_energy=energy(constrained_delta[j])) for j in range(5)],
        selected_target_norm=float(np.sqrt(2*f0)),selected_change_norm=float(np.sqrt(energy(constrained_delta))),
        selected_change_relative_to_target=fraction(np.sqrt(energy(constrained_delta)),np.sqrt(2*f0)),
        objective_nominal=fn,objective_tight=ft,objective_zero=f0,objective_reduction=fn-ft,
        remaining_objective_fraction_removed=fraction(fn-ft,fn),objective_reduction_relative_to_zero=fraction(fn-ft,f0),
        selected=omega)

def projection_groups(data,a,delta,nominal,core_samples,previous_state):
    cols=data.cols[a];valid=data.valid[:,cols];d=delta[:,cols];n=nominal[:,cols]
    cross=core_samples[:,cols]>0
    overall=dict(valid_samples=int(valid.sum()),nonzero_samples=int(np.count_nonzero(d)),
        difference_energy=energy(d),nominal_energy=energy(n),maximum_absolute_difference=float(np.max(abs(d))),
        source_crossing_samples=int(cross.sum()),source_core_difference_energy=energy(d[cross]),
        source_core_nominal_energy=energy(n[cross]))
    overall['relative_projected_L2']=fraction(np.sqrt(overall['difference_energy']),np.sqrt(overall['nominal_energy']))
    overall['source_core_energy_fraction']=fraction(overall['source_core_difference_energy'],overall['difference_energy'])
    groups=[];waveforms={}
    for c,g,lo,hi,cc,mask,counts,den in data.groups:
        if data.ar[cc[0]] != a:
            continue
        x=delta[lo:hi,cc];active=mask!=0
        centered=np.where(active,x-np.sum(x,axis=0)/counts,0.)
        count=mask.sum(axis=1)
        mean=np.divide(x.sum(axis=1),count,out=np.zeros(len(count)),where=count>0)
        crossing=core_samples[lo:hi,cc]>0;nc=crossing.sum(axis=1)
        source_mean=np.divide(np.where(crossing,x,0).sum(axis=1),nc,out=np.zeros(len(nc)),where=nc>0)
        prefix=f'c{c:02d}_nw{g:02d}'
        np.testing.assert_array_equal(previous_state[prefix+'_columns'],cc)
        A=previous_state[prefix+'_basis']
        en=energy(x);ec=energy(centered);core_en=energy(x[crossing])
        groups.append(dict(group=prefix,array=b.ARRAYS[a],lo=lo,hi=hi,detectors=len(cc),
            difference_energy=en,centered_difference_energy=ec,source_core_difference_energy=core_en,
            coherent_time_mean_energy=float(np.sum(count*mean**2)),
            coherent_energy_fraction=fraction(float(np.sum(count*mean**2)),en),
            source_crossing_coherent_energy_fraction=fraction(float(np.sum(nc*source_mean**2)),core_en),
            preceding_basis_coupling_fraction=fraction(energy(b.mm(centered,A)),ec),
            basis_coupling_meaning='Euclidean energy in preceding rank-5 basis of valid-filled centered difference; not a frozen-basis replay or masked cleaning response.',
            maximum_detector_difference_energy=float(np.max(np.sum(x*x,axis=0)))))
        waveforms[prefix]=np.array([np.arange(lo,hi),count,mean,nc,source_mean])
    return overall,groups,waveforms

def subspace_distance(A,B):
    cosines=np.clip(np.linalg.svd(b.mm(A.T,B),compute_uv=False),0.,1.)
    return dict(cosines=cosines,principal_angles_degrees=np.rad2deg(np.arccos(cosines)),
        maximum_principal_angle_degrees=float(np.rad2deg(np.arccos(cosines.min()))),
        normalized_projector_distance=float(np.sqrt(max(0.,1.-np.mean(cosines**2)))))

def select_states(projection_rows):
    ordinary=('H_20260911',0);real=('real123424',0)
    grouped={}
    for r in projection_rows:
        grouped.setdefault((r['case'],r['pass_index']),[]).append(r)
    candidates=[]
    for key,rr in grouped.items():
        if key==ordinary or key[0].split('_')[0] not in ['H','H-shift','D','T']:
            continue
        if len(rr)==3 and all(r['available'] for r in rr):
            candidates.append((max(r['projection']['source_core_difference_energy'] for r in rr),key))
    candidates.sort(key=lambda q:(-q[0],q[1][0],q[1][1]))
    worst=candidates[0][1]
    assert all(len(grouped[k])==3 and all(r['available'] for r in grouped[k]) for k in [ordinary,worst,real])
    return [dict(role=role,case=k[0],pass_index=k[1]) for role,k in [('ordinary',ordinary),('largest_source_region_change',worst),('real',real)]],candidates
