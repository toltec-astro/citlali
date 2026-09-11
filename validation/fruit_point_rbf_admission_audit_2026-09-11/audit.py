#!/usr/bin/env python3
"""Retained-product bookkeeping and a declared correlated-noise oracle diagnostic."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent;PREV=H.parent/'fruit_point_rbf_feedback_2026-09-11'
sys.path.insert(0,str(PREV));import rbf
b=rbf.base
read=lambda p:json.loads(Path(p).read_text())
TOUCHED={}
def receipt(p):TOUCHED[str(p)]=b.digest(p);return read(p)
def archive(p):TOUCHED[str(p)]=b.digest(p);return np.load(p)
def covariance_detector(features,expected):
    C=np.cov(features,rowvar=False,ddof=1);Cs=.9*C+.1*np.diag(np.diag(C));cond=float(np.linalg.cond(Cs))
    if not np.isfinite(Cs).all() or not np.all(np.linalg.eigvalsh(Cs)>0) or cond>1e8:raise ValueError('unusable projection covariance')
    v=np.linalg.solve(Cs,expected);w=v/np.dot(expected,v);sigma=float(np.sqrt(np.dot(w,Cs@w)))
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('invalid amplitude scale')
    return w,sigma,C,Cs,cond

def main():
    # This receipt predates diagnostic scoring and binds the exact definition.
    b.write(H/'AUDIT_START.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=b.digest(H/'AUDIT_PLAN.md'),code_sha256=b.digest(Path(__file__))))
    gates=[];timelines=[];sources=['gauss20260911','gauss20260912','mismatch20260911','coma_bright','coma_half'];roots={v:Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-'+v) for v in ['r0.1','r0.2']}
    for version,root in roots.items():
        for case in receipt(root/'CASES.json'):
            name=case['case'];events=[[] for _ in range(3)];first=archive(root/name/'R/pass00_maps.npz')['total'];last=None;maps_equal=np.ones(3,dtype=bool)
            for k in range(7):
                rec=receipt(root/name/'R'/f'pass{k:02d}.json');maps=archive(root/name/'R'/f'pass{k:02d}_maps.npz')
                if last is not None:np.testing.assert_array_equal(maps['applied_model'],last)
                for a,array in enumerate(b.ARRAYS):
                    d=rec['decision'][a];score_pass=[v is not None and v>=5 for v in d['heldout_scores']];cos_pass=d['fold_cosine'] is not None and d['fold_cosine']>=.8
                    assert (all(score_pass) and cos_pass)==d['admitted']
                    event=dict(version=version,case=name,array=array,map_number=k+1,heldout_scores=d['heldout_scores'],fold_cosine=d['fold_cosine'],score_pass=score_pass,cosine_pass=cos_pass,
                        failed_gates=[n for n,ok in zip(['score_0','score_1','cosine'],score_pass+[cos_pass]) if not ok],admitted=d['admitted'],next_nonzero=bool(np.any(maps['next_model'][a])),applied_nonzero=bool(np.any(maps['applied_model'][a])))
                    gates.append(event);events[a].append(event);maps_equal[a]&=np.array_equal(first[a],maps['total'][a],equal_nan=True)
                last=maps['next_model'].copy()
            for a,array in enumerate(b.ARRAYS):
                ee=events[a];timelines.append(dict(version=version,case=name,array=array,first_admission_map=next((e['map_number'] for e in ee if e['admitted']),None),
                    first_application_map=next((e['map_number'] for e in ee if e['applied_nonzero']),None),admitted_map_count=sum(e['admitted'] for e in ee),applied_map_count=sum(e['applied_nonzero'] for e in ee),all_outputs_equal_first=bool(maps_equal[a]),
                    first_gates=ee[0],final_gates=ee[-1]))
    root=roots['r0.1'];g=archive(root/'geometry.npz');x=g['x'];y=g['y'];Q=g['Q'];S=g['S'];D=g['D'];O=g['O'];shape=tuple(g['shape']);B=np.column_stack([np.ones(len(x)),x/90,y/90]);rad=np.hypot(x-17,y+11);oracle=[];features_saved={}
    lookup={(int(xx),int(yy)):i for i,(xx,yy) in enumerate(zip(x,y))}
    for case in sources:
        null='null20260912' if case=='gauss20260912' else 'null20260911';reference='null20260911' if null.endswith('12') else 'null20260912'
        observed=archive(root/case/'R/pass00_maps.npz')['total'];nuisance=archive(root/null/'R/pass00_maps.npz')['total'];refnoise=archive(root/reference/'R/pass00_maps.npz')['total']
        # At bootstrap these maps are shared by all arms and both candidates.
        for version,rr in roots.items():
            for arm in ['P','G','R']:np.testing.assert_array_equal(observed,archive(rr/case/arm/'pass00_maps.npz')['total'])
        for a,array in enumerate(b.ARRAYS):
            key=case+'_'+array;meta=dict(case=case,array=array,paired_null=null,reference_null=reference)
            try:
                q=np.sqrt(Q[a]/np.median(Q[a,S[a]]));q[~S[a]]=0
                def residual(z):
                    beta=np.linalg.lstsq(B[O[a]],z[O[a]],rcond=None)[0]
                    return np.where(S[a],q*(z-b.mm(B,beta)),0)
                Y=residual(observed[a]);N=residual(nuisance[a]);R=residual(refnoise[a]);T=Y-N
                masks=[D[a]&(rad>lo)&(rad<=hi) for lo,hi in [(-1,15),(15,35),(35,60)]]
                footprint=np.flatnonzero(np.logical_or.reduce(masks));norms=np.array([np.linalg.norm(T[m]) for m in masks]);assert np.all(norms>0)
                filters=np.array([np.where(m,T/n,0) for m,n in zip(masks,norms)]);f=b.mm(filters,Y);fn=b.mm(filters,N);targetQ=np.median(Q[a,footprint]);vectors=[];positions=[];ratios=[]
                for dy in range(-80,81,20):
                    for dx in range(-80,81,20):
                        shifted=np.array([lookup.get((int(x[i]+dx),int(y[i]+dy)),-1) for i in footprint])
                        if np.any(shifted<0) or not np.all(S[a,shifted]):continue
                        ratio=np.median(Q[a,shifted])/targetQ
                        if not .5<=ratio<=2:continue
                        vectors.append(b.mm(filters[:,footprint],R[shifted]));positions.append((dx,dy));ratios.append(ratio)
                if len(vectors)<16:raise ValueError(f'only {len(vectors)} complete coverage-compatible placements')
                features=np.array(vectors);w,sigma,C,Cs,cond=covariance_detector(features,norms);mean=features.mean(axis=0)
                amp=float(np.dot(w,f-mean));nullamp=float(np.dot(w,fn-mean));paired=amp-nullamp;assert np.isclose(paired,1,rtol=1e-10,atol=1e-10)
                scores=b.mm(features-mean,w)/sigma
                oracle.append(dict(**meta,available=True,placements=len(vectors),condition_number=cond,expected=norms,covariance=C,shrunk_covariance=Cs,weights=w,
                    amplitude=amp,paired_null_amplitude=nullamp,paired_amplitude=paired,empirical_amplitude_scale=sigma,source_score=amp/sigma,paired_null_score=nullamp/sigma,processed_response_score=paired/sigma,
                    reference_score_range=[float(scores.min()),float(scores.max())],reference_score_rms=float(np.sqrt(np.mean(scores**2))),five_empirical_scale_marker=bool(amp/sigma>=5),
                    qualification='Oracle realized processed template; three-projection covariance and overlapping spatial placements; not a calibrated significance or independent detection validation'))
                features_saved[key+'_reference_features']=features;features_saved[key+'_shifts']=np.array(positions);features_saved[key+'_coverage_ratios']=np.array(ratios);features_saved[key+'_filters']=filters
            except (ValueError,AssertionError) as e:oracle.append(dict(**meta,available=False,reason=str(e)))
    np.savez_compressed(H/'DIAGNOSTIC_FEATURES.npz',**features_saved)
    # Frozen RBF packet and all files actually used remain unchanged.
    packet=read(PREV/'RESULT_MANIFEST.json')
    for r in packet['files']:assert b.digest(PREV/r['path'])==r['sha256'],r['path']
    for name,h in TOUCHED.items():assert b.digest(name)==h,name
    b.write(H/'AUDIT_RESULTS.json',dict(gates=gates,timelines=timelines,processed_template_diagnostic=oracle,checks=dict(previous_packet_files_unchanged=len(packet['files']),retained_files_read=len(TOUCHED),all_used_hashes_unchanged=True,first_maps_identical_across_arms=True,next_to_applied_exact=True,new_cleaning_passes=0,independent_nuisance_realizations=2),input_files=[dict(path=p,sha256=h) for p,h in sorted(TOUCHED.items())]))
    for r in oracle:print(r['case'],r['array'],[(k,round(r[k],3)) for k in ['source_score','paired_null_score','processed_response_score']] if r['available'] else r['reason'])
    print('Applied compact cases',[(r['version'],r['case'],r['array'],r['first_admission_map'],r['first_application_map'],r['applied_map_count']) for r in timelines if r['case'].startswith('gauss') and r['first_admission_map']])
if __name__=='__main__':main()
