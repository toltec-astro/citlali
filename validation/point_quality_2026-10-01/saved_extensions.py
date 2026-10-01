"""Saved-only support geometry and re-bound endpoint covariance; no native passes."""
from pathlib import Path
import os,sys,json,re,time
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/point-quality-mpl',XDG_CACHE_HOME='/private/tmp/point-quality-cache')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
D=Path(__file__).resolve().parent;W=D.parents[1];sys.path.insert(0,str(W/'tools/timestream_successor'))
import numpy as np,yaml
from astropy.io import fits
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from point_reference import context,endpoint
from point_guarded_feedback import infer_base
from point_fruit_diagnostics import patch_map
from point_empirical import coordinates
L=Path('/Users/gwilson/work_toltec/local_data/citlali-validation')
def main():
 start=time.monotonic();root=L/'successor-rtc-point-transfer-completion-20260930/B-REF';prod=root/'M0/products/a1100'
 plan=yaml.safe_load((prod/'plan.yaml').read_text());pix=np.fromfile(prod/'operator-pixel.i32',dtype='<i4')
 with fits.open(prod/'map.fits') as h:
  shape=h[0].data.shape;x,y=coordinates(shape,h[0].header);disk=np.hypot(x,y)<=40
 groups=[dict(name='even',scans=[0,2,4]),dict(name='odd',scans=[1,3,5])];maps=[]
 for group in groups:
  counts=np.zeros(np.prod(shape),dtype=np.int64)
  for col in plan['ordered_column_domain']:
   nw,sid=map(int,re.search(r'network-(\d+)-segment-(\d+)',col['publication_file']).groups())
   if sid in group['scans']:
    p=pix[col['first_input_row']:col['first_input_row']+col['input_rows']];counts+=np.bincount(p[p>=0],minlength=len(counts))
  z=counts.reshape(shape);maps.append(z);group.update(contributions=int(z.sum()),source_disk_contributions=int(z[disk].sum()),reached_pixels=int((z>0).sum()),source_disk_reached=int(((z>0)&disk).sum()),source_disk_expected=int(disk.sum()))
 intervals=[]
 for nw in range(1,5):
  p=root/f'upstream/products/network{nw}/donor-continuity/ptc/receipt.yaml';r=yaml.safe_load(p.read_text())
  intervals.append(dict(network=nw,binding=bound(p),scans=[{k:s[k] for k in ('scan','native_interval','scheduled_times','retained')} for s in r['segments']]))
 save_json(D/'results/subset-geometry.json',dict(anchor=152420,selection='Single preregistered alternating six10s scheduled segments, even0/2/4 vs odd1/3/5; no outcome search. Groups alternate time to balance drift and repeated coverage. Original physical gaps must be retained.',groups=groups,common_reached_pixels=int(((maps[0]>0)&(maps[1]>0)).sum()),bindings=[bound(prod/'plan.yaml'),bound(prod/'operator-pixel.i32'),bound(prod/'map.fits')],intervals=intervals,qualification='Geometry/support audit from saved S M0 operator, conditional on full-observation decisions. No subset map or independent reduction generated. Same split is specified for R/S; current legacy interface cannot supply subset-relearned R, retained R samples purged. Stability not established; geometry is not claimed impossible.',new_starts=0))
 root=L/'successor-rtc-two-stage-20260928/B-REF-rep1';path=root/'run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits';c=context(root/'freeze/a1100');s,v,_,_=patch_map(path);m,_=infer_base(c,s,v)
 dest=D/'results/152430-rebound-endpoint';rows=endpoint(path,c,m,dest)
 save_json(D/'results/152430-rebound-measurement.json',dict(map=bound(path),context=[bound(p) for p in sorted((root/'freeze/a1100').iterdir()) if p.is_file()],measurement=m,uncertainties=rows,uncertainty_model=bound(dest/'model.json'),operator_binding=bound(dest/'model.npz'),qualification='Repaired free fit on executed B map; endpoint covariance recomputed through its current local Jacobian. Conditional transferred background covariance, not independent repeat or paired R/S significance. No model-estimation/calibration error coverage; no sigma_A/g_A until compatible response estimator binding.',new_starts=0,evaluation_seconds=time.monotonic()-start))
if __name__=='__main__':main()
