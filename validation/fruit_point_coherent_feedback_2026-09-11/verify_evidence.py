#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
from threadpoolctl import threadpool_limits
import experiment as e

H=Path(__file__).resolve().parent

def verify():
    roots=[Path('/private/tmp/sci-fruit-point-coherent-feedback-20260911-r0.1'),Path('/private/tmp/sci-fruit-point-coherent-feedback-20260911-r0.2')]
    records=[]
    for freeze in ['FREEZE.json','FREEZE_R0.2.json']:
        for row in json.loads((H/freeze).read_text())['files']:
            assert e.digest(H/row['path'])==row['sha256'],row['path']
    for root in roots:
        manifest=json.loads((root/'PRODUCT_MANIFEST.json').read_text())
        for f in manifest['files']:
            p=root/f['path'];assert p.stat().st_size==f['bytes'] and e.digest(p)==f['sha256'],str(p)
        assert manifest['bytes']<4*2**30
        assert not list(root.rglob('*FAILURE.json'))
        completes=list(root.glob('*/*/COMPLETE.json'));assert len(completes)==12
        geom=np.load(root/'fixed_geometry.npz');D=geom['D'];S=geom['normalization_support'];cases=[p.name for p in root.iterdir() if p.is_dir()]
        for case in cases:
            for arm in ['P','G']:
                last=np.zeros_like(D,dtype=float)
                for k in range(7):
                    m=np.load(root/case/arm/f'pass{k:02d}_maps.npz');r=json.loads((root/case/arm/f'pass{k:02d}.json').read_text())
                    assert np.array_equal(m['applied_model'],last)
                    assert np.all(m['next_model'][~D]==0) and np.all(m['applied_model'][~D]==0)
                    assert np.isfinite(m['total'][S]).all()
                    if arm=='G':
                        for a in range(3):
                            if r['decision'][a]['reason']=='coherent_source':
                                predicted=e.gaussian(r['fit'][a]['parameters'],geom['x'][D[a]],geom['y'][D[a]])
                                np.testing.assert_allclose(m['next_model'][a,D[a]],predicted,rtol=1e-13,atol=1e-12)
                            else:assert not np.any(m['next_model'][a])
                    last=m['next_model']
            p=np.load(root/case/'P/pass00_maps.npz')['total'];g=np.load(root/case/'G/pass00_maps.npz')['total']
            np.testing.assert_array_equal(p,g)
        records.append(dict(root=str(root),files=len(manifest['files']),bytes=manifest['bytes'],complete_trajectories=len(completes),passes=84,all_hashes_match=True,bootstrap_identical=True,replacement_and_no_background_checked=True))
    maxdiff=0.
    for case in [p.name for p in roots[0].iterdir() if p.is_dir()]:
        for k in range(7):
            p=np.load(roots[0]/case/'P'/f'pass{k:02d}_maps.npz');q=np.load(roots[1]/case/'P'/f'pass{k:02d}_maps.npz')
            for field in ['total','applied_model','next_model']:
                np.testing.assert_array_equal(p[field],q[field]);v=np.abs(p[field]-q[field]);maxdiff=max(maxdiff,float(np.nanmax(v)))
    # Independent application of stored state: no fitting or replay of learning.
    record=json.loads((H/'INPUT_PREFLIGHT.json').read_text());data=e.Data(record);reconstruction={}
    for arm in ['P','G']:
        root=roots[1]/'real123424'/arm;maps=np.load(root/'pass06_maps.npz');state=np.load(root/'pass06_state.npz')
        proj=data.project(maps['applied_model']);rho=data.y-proj;clean=np.full_like(data.y,np.nan)
        for c,g,lo,hi,cols,mask,counts,den in data.groups:
            key=f'c{c:02d}_nw{g:02d}';A=state[key+'_basis'];lam=state[key+'_location']
            np.testing.assert_array_equal(cols,state[key+'_columns'])
            np.testing.assert_allclose(lam,np.sum(np.where(mask!=0,rho[lo:hi,cols],0),axis=0)/counts,rtol=1e-13,atol=1e-9)
            xc=rho[lo:hi,cols]-lam
            # Direct per-distinct-mask solve checks the stored Apply independently.
            patterns,inverse=np.unique(mask,axis=0,return_inverse=True);coef=np.zeros((hi-lo,5))
            for i,pattern in enumerate(patterns):
                rows=np.flatnonzero(inverse==i);used=pattern!=0
                if not used.any():continue
                B=A[used];normal=e.mm(B.T,B);rhs=e.mm(xc[rows][:,used],B)
                coef[rows]=np.linalg.solve(normal,rhs.T).T
            clean[lo:hi,cols]=np.where(mask!=0,xc-e.mm(coef,A.T),np.nan)
        rebuilt=data.grid(clean+proj);diff=float(np.nanmax(abs(rebuilt-maps['total'])))
        np.testing.assert_allclose(rebuilt,maps['total'],rtol=1e-10,atol=1e-8,equal_nan=True);reconstruction[arm]=diff
    for row in json.loads((H/'OG_BENCHMARK.json').read_text())['files']:
        assert e.digest(row['path'])==row['sha256'],row['path']
    e.write(H/'VERIFICATION.json',dict(status='PASS',runs=records,reference_maps_reproduce_exactly_across_revision=True,max_reference_difference=maxdiff,independent_stored_state_map_max_abs_error=reconstruction,input_hash_unchanged=True,OG_hashes_unchanged=True,production_or_upstream_changes=False,reserved_129081='not opened'))
    print('PASS',records,'reference max difference',maxdiff,'stored-state map errors',reconstruction)
if __name__=='__main__':
    with threadpool_limits(limits=4):verify()
