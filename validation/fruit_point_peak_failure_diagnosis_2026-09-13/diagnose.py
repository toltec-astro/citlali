"""Read saved JSON/NPZ products only: no scientific code, fitter or solver imports."""
from pathlib import Path
import json
import hashlib
import re
import subprocess
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'fruit_point_nominal_utility_2026-09-13'
OUT=Path('/private/tmp/sci-fruit-point-nominal-utility-20260913-r0.1')
ARRAYS=['a1100','a1400','a2000'];SEEDS=[20260911,20260912]

def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def native(x):
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,x):Path(path).write_text(json.dumps(native(x),indent=2)+'\n')
def preserve():
    p=read(PRIOR/'PRESERVATION_START.json')
    manifests=[Path(r['path']) for r in p['packets']]+[PRIOR/'RESULT_MANIFEST.json']
    packet=[]
    for m in manifests:
        rows=read(m)['files']
        for r in rows:assert digest(m.parent/r['path'])==r['sha256'],r['path']
        packet.append(dict(path=str(m),sha256=digest(m),payloads=len(rows)))
    ext=[]
    for root in [Path(r['root']) for r in p['external_products']]+[OUT]:
        rows=read(root/'PRODUCT_MANIFEST.json')['files']
        for r in rows:assert digest(root/r['path'])==r['sha256'],r['path']
        ext.append(dict(root=str(root),manifest_sha256=digest(root/'PRODUCT_MANIFEST.json'),payloads=len(rows)))
    f=ROOT/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    rows=re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|$',(f/'PACKET_MANIFEST.md').read_text(),re.M)
    assert len(rows)==57
    for name,_,sha in rows:assert digest(f/name)==sha
    for w in p['protected_worktrees']:
        assert subprocess.check_output(['git','-C',w['path'],'status','--short'],text=True)==w['status']
    return dict(packets=packet,external=ext,frozen_authority_payloads=57,protected_worktrees=p['protected_worktrees'],archives='status only; opaque, unopened and unhashed')

def withholding(m):
    j=m['judgments'];p=m['fixed_inner_domain_probe'];o=m['original']
    reasons=list(j['limitations'])
    if not p['available']:reasons.append('inner_probe_unavailable')
    elif p['peak_relative_difference']>.01:reasons.append('peak_domain_change_gt_1pct')
    if o.get('empirical_peak_score') is None:reasons.append('peak_score_unavailable')
    elif o['empirical_peak_score']<=5:reasons.append('peak_score_le_5')
    assert (not reasons)==j['peak_response_usable']
    return reasons

def fit_readout(f):
    if not f:return None
    x,y=f['centroid'];beta=f['background']
    return dict(peak=f['peak'],centroid=f['centroid'],widths=f['widths'],angle_rad=f['angle_rad'],
        gaussian_integral=f['gaussian_integral'],background=beta,background_at_fitted_center=beta[0]+beta[1]*x/90+beta[2]*y/90,
        sse=f['sse'],nfev=f['nfev'],solver_status=f['solver_status'],successful_starts=f['successful_starts'])

def main():
    prior=preserve();write(HERE/'PRESERVATION_START.json',prior)
    rows=read(OUT/'RECEIPTS.json');assert len(rows)==238
    traces=[];lookup={}
    for r in rows:
        for a,m in enumerate(r['measurement']):
            o=m['original'];p=m['fixed_inner_domain_probe'];f=fit_readout(o.get('fit'))
            j=m['judgments'];reasons=withholding(m)
            outer=o['background']
            outer_at_center=None if f is None else outer[0]+outer[1]*f['centroid'][0]/90+outer[2]*f['centroid'][1]/90
            q=dict(case=r['case'],arm=r['arm'],array=ARRAYS[a],pass_index=r['pass_index'],wall_seconds=r['cumulative_wall_seconds'],
                fit=f,inner_fit=fit_readout(p.get('fit')),outer_background=outer,outer_background_at_fitted_center=outer_at_center,
                inner_probe_available=p['available'],peak_domain_change=p.get('peak_relative_difference'),centroid_domain_change_arcsec=p.get('centroid_difference_arcsec'),
                peak_score=o.get('empirical_peak_score'),withholding_reasons=reasons,judgments=j,
                fit_shape_error=o.get('shape_error'),source_support_fraction=o.get('source_support_fraction'),exterior_RMS=o['exterior_rms'],
                source_truth_score=None if r['truth_score'] is None else r['truth_score'][a],feedback_available=r['decisions'][a]['available'])
            traces.append(q);lookup[r['case'],r['arm'],a,r['pass_index']]=q
    ratios=[]
    for arm in ['P','C']:
        for a in range(3):
            for k in range(7):
                for n,d,target in [('H','D',1/.9),('D','H',.9),('T','H',.8 if a==1 else 1)]:
                    for i in SEEDS:
                        for j in SEEDS:
                            nm,dm=lookup[f'{n}_{i}',arm,a,k],lookup[f'{d}_{j}',arm,a,k]
                            ratio=nm['fit']['peak']/dm['fit']['peak'] if nm['fit'] and dm['fit'] and dm['fit']['peak']!=0 else None
                            error=ratio/target-1 if ratio is not None else None
                            usable=not nm['withholding_reasons'] and not dm['withholding_reasons']
                            ratios.append(dict(arm=arm,array=ARRAYS[a],pass_index=k,family=n+'/'+d,numerator_seed=i,denominator_seed=j,
                                pairing='matched' if i==j else 'crossed',ratio=ratio,expected=target,relative_error=error,available=usable,
                                numeric_within_5pct=error is not None and abs(error)<=.05,
                                numerator_withholding=nm['withholding_reasons'],denominator_withholding=dm['withholding_reasons']))
    seed_changes=[]
    for arm in ['P','C']:
        for a in range(3):
            for k in range(7):
                h1,h2=[lookup[f'H_{s}',arm,a,k] for s in SEEDS];d1,d2=[lookup[f'D_{s}',arm,a,k] for s in SEEDS]
                if not all(v['fit'] and v['fit']['peak']>0 for v in [h1,h2,d1,d2]):continue
                h=h2['fit']['peak']/h1['fit']['peak'];d=d2['fit']['peak']/d1['fit']['peak']
                gains=[next(r for r in ratios if r['arm']==arm and r['array']==ARRAYS[a] and r['pass_index']==k and r['family']=='H/D' and r['numerator_seed']==i and r['denominator_seed']==j)['ratio'] for i,j in [(SEEDS[0],SEEDS[0]),(SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0]),(SEEDS[1],SEEDS[1])]]
                np.testing.assert_allclose(gains[1]/gains[0],1/d,rtol=1e-13)
                np.testing.assert_allclose(gains[2]/gains[3],d,rtol=1e-13)
                def changes(one,two):
                    f,g=one['fit'],two['fit']
                    return dict(peak_ratio=g['peak']/f['peak'],width_ratios=np.array(g['widths'])/f['widths'],
                        width_product_ratio=np.prod(g['widths'])/np.prod(f['widths']),integral_ratio=g['gaussian_integral']/f['gaussian_integral'],
                        center_background_difference=g['background_at_fitted_center']-f['background_at_fitted_center'],
                        outer_center_background_difference=two['outer_background_at_fitted_center']-one['outer_background_at_fitted_center'],
                        available=not one['withholding_reasons'] and not two['withholding_reasons'])
                seed_changes.append(dict(arm=arm,array=ARRAYS[a],pass_index=k,H=changes(h1,h2),D=changes(d1,d2),
                    common_seed_log_response=.5*(np.log(h)+np.log(d)),differential_seed_log_response=np.log(h/d),
                    matched_gain_ratio_seed2_over_seed1=h/d))
    scaling=[]
    with np.load(OUT/'geometry.npz') as z:domains={name:z[name] for name in ['S','D','O','central']}
    map_paths=[]
    for arm in ['P','C']:
        for seed in SEEDS:
            for k in range(7):
                hp=OUT/f'H_{seed}'/arm/f'pass{k:02d}_maps.npz';tp=OUT/f'T_{seed}'/arm/f'pass{k:02d}_maps.npz';map_paths.extend([hp,tp])
                with np.load(hp) as hz,np.load(tp) as tz:
                    for a in range(3):
                        scale=.8 if a==1 else 1.
                        hf,tf=lookup[f'H_{seed}',arm,a,k]['fit'],lookup[f'T_{seed}',arm,a,k]['fit']
                        stat={}
                        for region,mask in domains.items():
                            m=mask[a];h,t=hz['total'][a,m],tz['total'][a,m]
                            assert np.isfinite(h).all() and np.isfinite(t).all()
                            norm=np.linalg.norm(scale*h)
                            stat[region]=dict(relative_L2=float(np.linalg.norm(t-scale*h)/norm),maximum_abs=float(np.max(abs(t-scale*h))))
                        scaling.append(dict(arm=arm,seed=seed,array=ARRAYS[a],pass_index=k,expected_scale=scale,total_scaling=stat,
                            feedback_relative_L2=float(np.linalg.norm(tz['next_model'][a]-scale*hz['next_model'][a])/max(np.linalg.norm(scale*hz['next_model'][a]),1e-30)),
                            peak_scale_error=tf['peak']/(scale*hf['peak'])-1 if hf and tf and hf['peak']!=0 else None,
                            width_ratios=np.array(tf['widths'])/hf['widths'] if hf and tf else None,
                            fit_SSE_over_scaled_H_SSE=tf['sse']/(scale*scale*hf['sse']) if hf and tf else None,
                            fit_background_scaling_difference=np.array(tf['background'])-scale*np.array(hf['background']) if hf and tf else None))
    summary=[]
    for arm in ['P','C']:
        for label,names in [('primary_HDT',['H','D','T']),('all_compact',['H','H-shift','D','T']),('all_cases',None)]:
            qq=[q for q in traces if q['arm']==arm and q['pass_index']==6 and (names is None or q['case'].rsplit('_',1)[0] in names)]
            usable=[not q['withholding_reasons'] for q in qq]
            errs=[q['source_truth_score'].get('peak_relative_error') if q['source_truth_score'] else None for q in qq]
            summary.append(dict(arm=arm,population=label,required=len(qq),usable=sum(usable),withheld=len(qq)-sum(usable),
                reason_occurrences=dict(Counter(x for q in qq for x in q['withholding_reasons'])),
                reason_combinations=dict(Counter('+'.join(q['withholding_reasons']) for q in qq if q['withholding_reasons'])),
                accuracy_availability=dict(usable_accurate=sum(ok and er is not None and abs(er)<=.05 for ok,er in zip(usable,errs)),
                    usable_inaccurate=sum(ok and er is not None and abs(er)>.05 for ok,er in zip(usable,errs)),
                    withheld_accurate=sum(not ok and er is not None and abs(er)<=.05 for ok,er in zip(usable,errs)),
                    withheld_inaccurate=sum(not ok and er is not None and abs(er)>.05 for ok,er in zip(usable,errs)))))
    ratio_summary=[]
    for arm in ['P','C']:
        for fam in ['H/D','D/H','T/H']:
            for pairing in ['matched','crossed']:
                rr=[r for r in ratios if r['arm']==arm and r['pass_index']==6 and r['family']==fam and r['pairing']==pairing]
                ratio_summary.append(dict(arm=arm,family=fam,pairing=pairing,required=len(rr),numeric_within_5pct=sum(r['numeric_within_5pct'] for r in rr),
                    available=sum(r['available'] for r in rr),usable_and_accurate=sum(r['available'] and r['numeric_within_5pct'] for r in rr),
                    max_abs_error=max(abs(r['relative_error']) for r in rr)))
    write(HERE/'TRACES.json',traces);write(HERE/'RATIOS.json',ratios);write(HERE/'SEED_RESPONSE.json',seed_changes);write(HERE/'SCALING_CHECKS.json',scaling)
    write(HERE/'SUMMARY.json',dict(availability=summary,ratios=ratio_summary))
    bindings=[OUT/'RECEIPTS.json',OUT/'geometry.npz',PRIOR/'RESULT_MANIFEST.json',PRIOR/'CASES.json',HERE/'SCOPE.md',HERE/'diagnose.py']+map_paths
    write(HERE/'INPUT_BINDINGS.json',dict(starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=[dict(path=str(p),sha256=digest(p)) for p in sorted(set(bindings))]))
    assert preserve()==prior
    write(HERE/'VERIFICATION.json',dict(saved_array_pass_labels_verified=len(traces),ratios=len(ratios),scaling_array_pass_checks=len(scaling),
        exact_ratio_identities_checked=len(seed_changes)*2,prior_products_preserved=True,pre_PTC_reads=0,PTC_calls=0,refits=0,optimizations=0,
        threshold_changes=0,selected_iterations=0,reserved_observations=0,candidate_status='parked_for_POINT'))
    print(json.dumps(native(dict(availability=summary,ratios=ratio_summary)),indent=2))
if __name__=='__main__':main()
