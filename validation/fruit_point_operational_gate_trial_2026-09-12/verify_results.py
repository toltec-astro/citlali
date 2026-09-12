"""Read-only product, recurrence, objective and preservation checks."""
import hashlib
import re
import subprocess
import numpy as np
from common import b, read, verify_freeze, HERE, ROOT, OUT, PREVIOUS, OLD
import candidate
from run_trial import make_estimators
from types import SimpleNamespace

def check_learned_states(rows):
    """Recompute selected covariance identities, never call clean or replay."""
    from run_trial import array_hash
    data=b.Data(read(PREVIOUS/'INPUT_PREFLIGHT.json'))
    old=np.load(OLD/'geometry.npz')
    checked=[]
    selected_cases=['real123424','H_20260911','T_20260911','H_20260912']
    groups=[]
    for a in range(3):
        g=[g for g in data.groups if data.ar[g[4][0]]==a]
        groups.extend([g[0],g[-1]])
    for case in selected_cases:
        if case=='real123424':parent=data.y
        else:
            name,seed=case.rsplit('_',1)
            noise=b.nuisance(data,old['scales'],int(seed))
            truth=np.load(OUT/(name+'_truth.npz'))['truth']
            parent=noise+data.project(truth)
            if name=='T':parent[:,data.ar==1]-=.2*noise[:,data.ar==1]
            del noise
        for arm in ['P','C']:
            rr=[r for r in rows if r['case']==case and r['arm']==arm]
            if not rr:continue
            assert array_hash(parent)==read(OUT/case/arm/'START.json')['parent_sha256_float64_C']
            for k in sorted({rr[0]['pass_index'],rr[-1]['pass_index']}):
                maps=np.load(OUT/case/arm/f'pass{k:02d}_maps.npz')
                model=maps['applied_model'];state=np.load(OUT/case/arm/f'pass{k:02d}_state.npz')
                for c,g,lo,hi,cols,mask,counts,den in groups:
                    projected=model[data.ar[cols][None,:],data.pixel[lo:hi,cols]]
                    raw=parent[lo:hi,cols]-np.where(mask!=0,projected,0)
                    location=np.sum(np.where(mask!=0,raw,0),axis=0)/counts
                    centered=np.where(mask!=0,raw-location,0)
                    cov=np.divide(b.mm(centered.T,centered),den,out=np.zeros_like(den),where=den>0)
                    prefix=f'c{c:02d}_nw{g:02d}'
                    np.testing.assert_array_equal(cols,state[prefix+'_columns'])
                    np.testing.assert_allclose(location,state[prefix+'_location'],rtol=1e-12,atol=1e-10)
                    A=state[prefix+'_basis'];ev=state[prefix+'_eigenvalues']
                    residual=float(np.linalg.norm(b.mm(cov,A)-A*ev)/max(np.linalg.norm(cov),1e-30))
                    assert residual<1e-10,(case,arm,k,prefix,residual)
                    np.testing.assert_allclose(b.mm(A.T,A),np.eye(5),rtol=1e-10,atol=1e-10)
                    checked.append(dict(case=case,arm=arm,pass_index=k,group=prefix,eigen_residual=residual))
        del parent
    b.write(HERE/'LEARNED_STATE_CHECKS.json',dict(checks=checked,cleaning_calls=0,
        meaning='Selected original-parent residual means/covariance eigen-identities; no new cleaning/replay'))
    return len(checked)

def main():
    verify_freeze()
    manifest=read(OUT/'PRODUCT_MANIFEST.json')
    for row in manifest['files']:
        assert b.digest(OUT/row['path'])==row['sha256'],row['path']
    z=np.load(OUT/'geometry.npz')
    data=SimpleNamespace(x=z['x'],ygrid=z['y'],shape=tuple(z['shape']),S=z['S'],D=z['D'],O=z['O'],Q=z['Q'])
    estimators,_=make_estimators(data)
    rows=read(OUT/'RECEIPTS.json')
    seen={};objective_checks=0;selection_checks=0;recurrence_checks=0;bootstrap_checks=0
    for r in rows:
        name,arm,k=r['case'],r['arm'],r['pass_index']
        p=np.load(OUT/name/arm/f'pass{k:02d}_maps.npz')
        if k==0:
            assert np.all(p['applied_model']==0)
            other=seen.get((name,'C' if arm=='P' else 'P',0))
            if other is not None:
                np.testing.assert_array_equal(p['total'],other)
                bootstrap_checks+=1
        else:
            prev=np.load(OUT/name/arm/f'pass{k-1:02d}_maps.npz')
            np.testing.assert_array_equal(p['applied_model'],prev['next_model'])
            recurrence_checks+=1
        seen[name,arm,k]=p['total'].copy()
        for a,e in enumerate(estimators):
            assert np.isfinite(p['total'][a,e.S]).all()
            assert np.isnan(p['total'][a,~e.S]).all()
            d=r['decisions'][a]
            if arm=='P':
                R=candidate.mad(p['total'][a,e.O])
                expected=np.where(data.D[a]&(p['total'][a]>3*R),p['total'][a],0)
                np.testing.assert_array_equal(p['next_model'][a],expected)
                continue
            Y,_=e.residual(p['total'][a]);W=candidate.analysis(Y.reshape(e.shape)).reshape(5,-1)
            sigma=e.sigmas[:,e.stratum]
            omega=e.valid&e.D&(abs(W)>=5*sigma)
            np.testing.assert_array_equal(omega,p['selected'][a]);selection_checks+=1
            assert np.all(p['next_model'][a]>=0) and not np.any(p['next_model'][a,~e.D])
            if omega.any():
                scale=d['solver_scale'];u=p['solver_trial'][a]/scale
                diff=candidate.analysis(u.reshape(e.shape))-candidate.analysis((Y/scale).reshape(e.shape))
                weight=np.where(omega,(scale/sigma)**2,0).reshape((5,)+e.shape)
                objective=.5*np.sum(weight*diff**2)
                grad=candidate.adjoint(weight*diff).ravel()[e.D]
                g0=candidate.adjoint(-weight*candidate.analysis((Y/scale).reshape(e.shape))).ravel()[e.D]
                pg=np.where((u[e.D]<=0)&(grad>0),0,grad)
                relative=max(abs(pg))/max(max(abs(g0)),1e-12)
                np.testing.assert_allclose(objective,d['objective'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(relative,d['relative_projected_gradient'],rtol=1e-9,atol=1e-12)
                objective_checks+=1
                if d['available']:
                    assert d['solver_success'] and d['finite'] and relative<=1e-4
                    np.testing.assert_array_equal(p['next_model'][a],p['solver_trial'][a])
                else:assert not np.any(p['next_model'][a])
            else:
                assert d['available'] and not d['admitted']
                assert not np.any(p['next_model'][a])
    starts=list(OUT.glob('*/*/START.json'))
    for p in starts:
        d=p.parent
        terminal=d/'COMPLETE.json' if (d/'COMPLETE.json').exists() else d/'FAILURE.json'
        assert read(terminal)['parent_unchanged']
    start=read(HERE/'PRESERVATION_START.json')
    packets=[]
    for record in start['packets']:
        p=record['path'];assert b.digest(p)==record['sha256']
        payloads=read(p)['files']
        from pathlib import Path
        for row in payloads:assert b.digest(Path(p).parent/row['path'])==row['sha256']
        packets.append(len(payloads))
    frozen=ROOT/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    records=re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|$',(frozen/'PACKET_MANIFEST.md').read_text(),re.M)
    assert len(records)==57
    for name,_,sha in records:assert b.digest(frozen/name)==sha
    for record in start['protected_worktrees']:
        assert subprocess.check_output(['git','-C',record['path'],'status','--short'],text=True)==record['status']
    learned_checks=check_learned_states(rows)
    result=dict(frozen_files=len(read(HERE/'FREEZE.json')['files']),run_payloads=len(manifest['files']),
        retained_passes=len(rows),cleaning_calls=read(OUT/'CLEANING_CALLS.json')['started'],
        matched_bootstraps=bootstrap_checks,applied_model_continuity_checks=recurrence_checks,
        selected_support_checks=selection_checks,objective_gradient_checks=objective_checks,
        immutable_parent_receipts=len(starts),sampled_learned_state_checks=learned_checks,
        verification_cleaning_calls=0,preserved_packet_payloads=packets,frozen_ordinary_MAP_payloads=57,
        protected_worktrees_unchanged=True,archives='presence/status only; not read, hashed or unpacked')
    b.write(HERE/'VERIFICATION.json',result)
    print(result)

if __name__=='__main__':main()
