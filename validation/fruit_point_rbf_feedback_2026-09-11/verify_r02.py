#!/usr/bin/env python3
"""Verify immutable sources/products, recurrence, coefficients and stored Apply."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
import rbf_r02 as rbf
import run_screen_r02 as run_screen
b=rbf.base;H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def verify(root,dest):
    freeze=run_screen.verify_freeze();start=read(root/'RUN_START.json');assert freeze==start['freeze'] and freeze['utc']<start['utc']
    manifest=read(root/'PRODUCT_MANIFEST.json')
    for r in manifest['files']:
        p=root/r['path'];assert p.stat().st_size==r['bytes'] and b.digest(p)==r['sha256'],str(p)
    assert manifest['bytes']<4*2**30;assert not list(root.rglob('*FAILURE.json'))
    cases=read(root/'CASES.json');assert len(cases)==9;assert len(list(root.glob('*/*/COMPLETE.json')))==27
    g=np.load(root/'geometry.npz');D=g['D'];S=g['S'];matrices=[]
    for array in b.ARRAYS:
        f=np.load(root/f'basis_{array}.npz');phi=sparse.csr_matrix((f['phi_data'],f['phi_indices'],f['phi_indptr']),shape=f['phi_shape']);HH=sparse.csr_matrix((f['H_data'],f['H_indices'],f['H_indptr']),shape=f['H_shape']);matrices.append((phi,f['b'],HH))
    count=0;max_integral_error=0.
    for case in cases:
        name=case['case']
        for arm in ['P','G','R']:
            last=np.zeros_like(D,dtype=float)
            for k in range(7):
                p=root/name/arm/f'pass{k:02d}';m=np.load(str(p)+'_maps.npz');rec=read(str(p)+'.json');count+=1
                np.testing.assert_array_equal(m['applied_model'],last);assert np.isfinite(m['total'][S]).all() and np.all(m['next_model'][~D]==0)
                if arm=='R':
                    coef=np.load(str(p)+'_coefficients.npz')
                    for a,array in enumerate(b.ARRAYS):
                        aa=coef[array];assert np.all(aa>=0);phi,bb,HH=matrices[a];model=phi@aa;decision=rec['decision'][a]
                        admit=all(v is not None and v>=5 for v in decision['heldout_scores']) and decision['fold_cosine'] is not None and decision['fold_cosine']>=.8
                        assert admit==decision['admitted'];np.testing.assert_allclose(m['next_model'][a],model if admit else np.zeros_like(model),atol=1e-10,rtol=1e-13)
                        F=np.dot(bb,aa);power=np.dot(aa,HH@aa);directF=4*model[D[a]].sum();directPower=4*np.dot(model[D[a]],model[D[a]])
                        np.testing.assert_allclose([F,power],[directF,directPower],rtol=1e-11,atol=1e-9)
                        if F>0 and power>0:
                            area=F*F/power;gridarea=directF**2/directPower;max_integral_error=max(max_integral_error,abs(area-gridarea)/area)
                last=m['next_model']
        for arm in ['G','R']:np.testing.assert_array_equal(np.load(root/name/'P/pass00_maps.npz')['total'],np.load(root/name/arm/'pass00_maps.npz')['total'])
    assert count==189
    old=Path('/private/tmp/sci-fruit-point-coherent-feedback-20260911-r0.2');oldcases=['real123424','null20260911','null20260912','gauss20260911','gauss20260912','mismatch20260911']
    for case in oldcases:
        for arm in ['P','G']:
            for k in range(7):
                x=np.load(old/case/arm/f'pass{k:02d}_maps.npz');y=np.load(root/case/arm/f'pass{k:02d}_maps.npz')
                for field in ['total','applied_model','next_model']:np.testing.assert_array_equal(x[field],y[field])
    initial=Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1')
    for case in cases:
        for arm in ['P','G']:
            for k in range(7):
                xx=np.load(initial/case['case']/arm/f'pass{k:02d}_maps.npz');yy=np.load(root/case['case']/arm/f'pass{k:02d}_maps.npz')
                for field in ['total','applied_model','next_model']:np.testing.assert_array_equal(xx[field],yy[field])
    previous_files=0
    for folder in [rbf.PREV,H.parent/'fruit_point_alignment_prior_2026-09-11']:
        for r in read(folder/'RESULT_MANIFEST.json')['files']:
            assert b.digest(folder/r['path'])==r['sha256'];previous_files+=1
    data=b.Data(read(rbf.PREV/'INPUT_PREFLIGHT.json'));errors={}
    for arm in ['P','G','R']:
        path=root/'real123424'/arm;maps=np.load(path/'pass06_maps.npz');st=np.load(path/'pass06_state.npz')
        projected=data.project(maps['applied_model']);rho=data.y-projected;clean=np.full_like(data.y,np.nan)
        for c,nw,lo,hi,cols,mask,counts,den in data.groups:
            key=f'c{c:02d}_nw{nw:02d}';A=st[key+'_basis'];lam=st[key+'_location'];xc=rho[lo:hi,cols]-lam
            np.testing.assert_array_equal(cols,st[key+'_columns']);np.testing.assert_allclose(lam,np.sum(np.where(mask!=0,rho[lo:hi,cols],0),axis=0)/counts,rtol=1e-13,atol=1e-9)
            patterns,inverse=np.unique(mask,axis=0,return_inverse=True);coef=np.zeros((hi-lo,5))
            for i,pattern in enumerate(patterns):
                rows=np.flatnonzero(inverse==i);used=pattern!=0
                if not used.any():continue
                Aused=A[used];coef[rows]=np.linalg.solve(b.mm(Aused.T,Aused),b.mm(xc[rows][:,used],Aused).T).T
            clean[lo:hi,cols]=np.where(mask!=0,xc-b.mm(coef,A.T),np.nan)
        rebuilt=data.grid(clean+projected);np.testing.assert_allclose(rebuilt,maps['total'],rtol=1e-10,atol=1e-8,equal_nan=True);errors[arm]=float(np.nanmax(abs(rebuilt-maps['total'])))
    result=dict(status='PASS',passes=count,trajectories=27,files=len(manifest['files']),bytes=manifest['bytes'],freeze_before_run=True,all_product_hashes_match=True,bootstrap_identical=True,
        immutable_parent_replacement_and_stored_relearning=True,nonnegative_coefficients_background_excluded=True,admission_rule_verified=True,max_relative_overlap_integral_error=max_integral_error,
        unchanged_control_trajectories=18,previous_packet_hashes_unchanged=previous_files,stored_state_map_max_abs_error=errors,input_hash_unchanged=True,
        reserved_129081='historically characterized; no new raw scientific read or feedback trajectory')
    b.write(dest,result);print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dest',type=Path);a=p.parse_args()
    with threadpool_limits(limits=4):verify(a.root,a.dest)
