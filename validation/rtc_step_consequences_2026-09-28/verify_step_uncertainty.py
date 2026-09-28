import sys,json,numpy as np
from pathlib import Path
repo=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(repo/'tools/timestream_successor'))
from point_reference import context,jacobian
from point_additive import variance
root=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928');records=[]
for out in sorted(root.glob('*-*')):
 if not (out/'LAUNCH.json').exists():continue
 base=out/'run/BASE-POINT/REF/a1100';s=json.loads((base/'SUMMARY.json').read_text());c=context(out/'freeze/a1100');f=c['fixed'];p=np.asarray(s['histories'][-1]['measurement']['guard']['accepted_parameters']);roi=f.disk;J=jacobian(p,f.x[roi],f.y[roi]);U,sv,Vt=np.linalg.svd(J,full_matrices=False);rank=int(np.sum(sv>np.finfo(float).eps*max(J.shape)*sv[0]));pinv=np.einsum('ik,kj->ij',Vt[:rank].T/sv[:rank],U[:,:rank].T,optimize=False)
 W=[]
 for row in pinv:
  z=np.zeros(roi.shape);z[roi]=row;W.append(z)
 ops=np.asarray(W+[q-np.sum(q)*W[6] for q in [f.weights,*[4*(f.radius<=a*f.major) for a in (1,1.5,2)]]]);saved=np.load(base/'endpoint/model.npz');old=saved['operators'];assert np.isfinite(ops).all() and np.isfinite(old).all();err=np.max(abs(ops-old));assert np.allclose(ops,old,rtol=1e-12,atol=1e-12)
 sig=[float(np.sqrt(variance(saved['psd'],saved['source_scale'],q))) for q in ops];reported=[r['conditional_sigma'] for r in s['endpoint_uncertainty']];assert all(v is not None and np.isfinite(v) for v in reported);assert np.allclose(sig,reported,rtol=1e-12,atol=1e-12)
 records.append(dict(run=out.name,operators_finite=True,operator_max_absolute_difference=float(err),sigmas_finite=True,sigma_max_absolute_difference=float(np.max(abs(np.array(sig)-reported)))))
(root/'UNCERTAINTY_WARNING_CHECK.json').write_text(json.dumps(dict(pass_=True,scope='same saved endpoint covariance/parameters; independently contract SVD factors without BLAS matmul; no model learning or uncertainty ensemble',records=records),indent=2)+'\n');print(records)
