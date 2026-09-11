#!/usr/bin/env python3
"""Check retained products, positional policy, unchanged control and stored Apply."""
from pathlib import Path
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
import experiment as e

def read(p):return json.loads(Path(p).read_text())

def verify(root):
    freeze=e.verify_freeze();start=read(root/'RUN_START.json')
    assert freeze==start['freeze'] and freeze['utc']<start['utc']
    manifest=read(root/'PRODUCT_MANIFEST.json')
    for r in manifest['files']:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and e.b.digest(p)==r['sha256'],str(p)
    assert manifest['bytes']<4*2**30
    failures=list(root.rglob('*FAILURE.json'));assert not failures,failures
    cases=read(root/'CASES.json');assert len(cases)==11
    assert len(list(root.glob('*/*/COMPLETE.json')))==22
    g=np.load(root/'fixed_geometry.npz');D=g['D'];S=g['normalization_support']
    count=0
    for case in cases:
        name=case['case']
        for arm in ['G','J']:
            last=np.zeros_like(D,dtype=float)
            for k in range(7):
                p=root/name/arm/f'pass{k:02d}';m=np.load(str(p)+'_maps.npz');r=read(str(p)+'.json');count+=1
                np.testing.assert_array_equal(m['applied_model'],last)
                assert np.isfinite(m['total'][S]).all() and np.all(m['next_model'][~D]==0)
                for a in range(3):
                    dec=r['decision'][a];f=r['policy_fit'][a]
                    if dec['reason']=='coherent_source':
                        assert f['peak']>3*dec['scale'] and not f['boundary_rejection']
                        z=e.b.gaussian(f['parameters'],g['x'][D[a]],g['y'][D[a]])
                        np.testing.assert_allclose(m['next_model'][a,D[a]],z,rtol=1e-13,atol=1e-12)
                    else:assert not m['next_model'][a].any()
                if arm=='J':
                    np.testing.assert_array_equal(r['policy_fit'][0]['centroid'],r['policy_fit'][1]['centroid'])
                    radius=np.linalg.norm(np.array(r['policy_fit'][2]['centroid'])-r['prior']['common_centroid'])
                    assert radius<=2+1e-10
                    if not r['prior']['reference_pair_anchor'] or r['prior']['a2000_prior_tension']:assert not m['next_model'][2].any()
                last=m['next_model']
        np.testing.assert_array_equal(np.load(root/name/'G/pass00_maps.npz')['total'],np.load(root/name/'J/pass00_maps.npz')['total'])
    assert count==154
    previous=read(e.PREV/'RESULT_MANIFEST.json')
    for r in previous['files']:assert e.b.digest(e.PREV/r['path'])==r['sha256']
    old=Path('/private/tmp/sci-fruit-point-coherent-feedback-20260911-r0.2');max_control=0.
    old_cases=['real123424','null20260911','null20260912','gauss20260911','gauss20260912','mismatch20260911']
    for case in old_cases:
        for k in range(7):
            x=np.load(old/case/'G'/f'pass{k:02d}_maps.npz');y=np.load(root/case/'G'/f'pass{k:02d}_maps.npz')
            for field in ['total','applied_model','next_model']:
                np.testing.assert_array_equal(x[field],y[field]);max_control=max(max_control,float(np.nanmax(abs(x[field]-y[field]))))
    data=e.b.Data(read(e.PREV/'INPUT_PREFLIGHT.json'));errors={}
    for arm in ['G','J']:
        p=root/'real123424'/arm;maps=np.load(p/'pass06_maps.npz');st=np.load(p/'pass06_state.npz')
        projected=data.project(maps['applied_model']);rho=data.y-projected;clean=np.full_like(data.y,np.nan)
        for c,nw,lo,hi,cols,mask,counts,den in data.groups:
            key=f'c{c:02d}_nw{nw:02d}';A=st[key+'_basis'];lam=st[key+'_location'];xc=rho[lo:hi,cols]-lam
            np.testing.assert_array_equal(cols,st[key+'_columns'])
            np.testing.assert_allclose(lam,np.sum(np.where(mask!=0,rho[lo:hi,cols],0),axis=0)/counts,rtol=1e-13,atol=1e-9)
            patterns,inverse=np.unique(mask,axis=0,return_inverse=True);coef=np.zeros((hi-lo,5))
            for i,pattern in enumerate(patterns):
                rows=np.flatnonzero(inverse==i);used=pattern!=0
                if not used.any():continue
                B=A[used];coef[rows]=np.linalg.solve(e.b.mm(B.T,B),e.b.mm(xc[rows][:,used],B).T).T
            clean[lo:hi,cols]=np.where(mask!=0,xc-e.b.mm(coef,A.T),np.nan)
        rebuilt=data.grid(clean+projected);np.testing.assert_allclose(rebuilt,maps['total'],rtol=1e-10,atol=1e-8,equal_nan=True)
        errors[arm]=float(np.nanmax(abs(rebuilt-maps['total'])))
    result=dict(status='PASS',passes=count,trajectories=22,files=len(manifest['files']),bytes=manifest['bytes'],freeze_before_run=True,
        all_product_hashes_match=True,bootstrap_identical=True,replacement_no_background_and_own_array_admission=True,
        relative_position_and_tension_rules=True,prior_experiment_payload_hashes_unchanged=len(previous['files']),
        unchanged_control_cases=6,max_control_difference=max_control,stored_state_map_max_abs_error=errors,input_hash_unchanged=True,reserved_129081='not opened')
    e.b.write(e.HERE/'VERIFICATION.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    with threadpool_limits(limits=4):verify(a.root)
