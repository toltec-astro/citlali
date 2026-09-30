"""Compact local provenance and saved accounting recovery, no new trajectory."""
import gzip,json,sys,platform
from pathlib import Path
import numpy as np,scipy,astropy
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.sparse.linalg import svds
from calibration_audit import D,R,H,S,HST,yread,bound,save,digest
P=H.parents[2]/'pointings/reduced/redu09'
def main():
 cfg=P/'citlali_o152389_0_2_c8.yaml';log=P/'citlali.log.gz';lines=gzip.open(log,'rt').read().splitlines()
 selected=[dict(line=i+1,text=line) for i,line in enumerate(lines) if ('starting fruit loops iteration' in line or ('152420/raw/' in line and ('reading in ' in line or 'MEDRMS' in line)))]
 accounting=[]
 for k in range(1,6):
  p=R/f'run/BASE-POINT/REF/a1100/cycle-{k:02d}/native/products/receipt.yaml';r=yread(p)
  accounting.append(dict(cycle=k,receipt=bound(p),selected_model=r['selected_model'],decomposition_max_abs_mJy=r['decomposition_max_abs_mJy_per_nominal_beam']))
 previous=D.parent/'point152420_map_comparison_2026-09-29/redu09/results/metrics.json';old=json.loads(previous.read_text())
 for b in old['inputs']:assert digest(b['path'])==b['sha256']
 metadata=[cfg,log,P/'index.yaml',H,S,HST,previous,R/'inputs/effective-point.yaml',R/'inputs/ast-source-binding.json',R/'inputs/observation.json']
 metadata += sorted(P.glob('learning_iter_*.csv'))
 metadata += [p for p in S.parent.glob('*baseline-manifest.ecsv')]+[p for p in S.parent.glob('*baseline-apt.ecsv')]
 sourcecfg=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/v2/beammaps/3c273/reduced/redu00/citlali_o148670_0_2_c1.yaml');metadata.append(sourcecfg)
 save(D/'results/provenance-audit.json',dict(inputs=[bound(p) for p in metadata],historical_iteration_log=selected,historical_resolved_config=yread(cfg),successor_addback_accounting=accounting,previous_sampling_check=dict(binding=bound(previous),verified_inputs=len(old['inputs']),fits=old['fits']),limits=['Historical save_all_iters=false and repeated writes to same 152420/raw path: initial/iter1-3 map bytes not recovered. Other redu directories are not proved parents.','Present-day historical file hashes establish this audit binding, not independent historical authenticity.','Native AltAz versus FK5 coadds are not pixelwise registered.','Historical row weights and flags establish some contribution, not exact native source-crossing support.']))
 assert abs(least_squares(lambda x:x-3,[0.]).x[0]-3)<1e-8
 np.testing.assert_allclose(fftconvolve([1.,2.],[3.,4.]),[3,10,8]);np.testing.assert_allclose(svds(np.diag([1.,2.,3.]),k=1,solver='propack')[1],[3.])
 wheel=Path('/private/tmp/point-amplitude-environment/scipy-1.17.1-cp313-cp313-macosx_14_0_arm64.whl')
 save(D/'results/environment-repair.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,old_scipy='1.15.2',failure='dyld invalid __DATA/__thread_bss offset in _spropack; all four installed PROPACK module hashes matched wheel RECORD',wheel=bound(wheel),installation='isolated-target smoke, then /Users/gwilson/tolteca/bin/pip install --no-deps --force-reinstall pinned wheel; NumPy unchanged',smoke=['least_squares','fftconvolve','svds solver=propack'],machine_wide_packages_modified=False))
 print('Verified saved provenance, five successor add-back closures, historical sampling evidence, and environment smoke.')
if __name__=='__main__':main()
