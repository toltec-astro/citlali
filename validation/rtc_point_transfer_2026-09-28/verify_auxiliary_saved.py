import os,sys,json,tempfile
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
w=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(w/'tools/timestream_successor'),str(w/'validation/point152420_map_comparison_2026-09-29')]
import numpy as np
import compare as old
from point_auxiliary import geometry,weights,sample,projection,from_map
from point_reference import initialize,endpoint
from point_guarded_feedback import measurement
from point_empirical import coordinates
from point_additive import linear_weights
from inspect_point_source import gaussian
r=old.BASE/'successor-rtc-point-transfer-runtime-repair-20260929'
redu=Path('/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings/reduced/redu09/152420/raw/toltec_commissioning_a1100_pointing_152420_citlali.fits')
records={}
for label,path,legacy in [('current',old.NEW,False),('earlier_contained',old.OLD,True),('redu09',redu,True)]:
 data=old.load(path,legacy);fit=old.fit(data);h=data['header'].copy()
 if legacy:
  h['CDELT1']/=3600;h['CDELT2']/=3600;h['CUNIT1']=h['CUNIT2']='deg' # Unit adapter only, axes remain native AltAz offsets.
 step=abs(h['CDELT1'])*3600;half=int(np.ceil(40/step));ph=h.copy();ph['CRPIX1']=ph['CRPIX2']=half+1
 g=geometry(ph,(2*half+1,)*2,fit['parameters']);q,roi=weights(g);a=sample(data['signal'],data['support'],h,g)
 assert a['available'],(label,a)
 value,info=projection(dict(auxiliary=g),a,fit['constant'])
 gx,gy=coordinates(q.shape,g['header']);unit=np.array(fit['parameters']);unit[0]=1.;unit[6]=0
 assert np.isclose(np.sum(q*gaussian(unit,gx,gy)),1,rtol=2e-15)
 rejected=False
 try:oldq,_=linear_weights(ph,(2*half+1,)*2,fit['parameters'])
 except ValueError:rejected=True
 if label=='earlier_contained':assert np.array_equal(oldq,q)
 else:assert rejected
 records[label]=dict(input=old.bound(path),fit=fit,shape=g['shape'],projection=value,support=info,legacy_rejected=rejected,template_pixels=int(roi.sum()),outside_feedback_pixels=int(np.sum(roi&(np.hypot(gx,gy)>40))),normalization=float(np.sum(q*gaussian(unit,gx,gy))))
with tempfile.TemporaryDirectory(prefix='point-aux-saved-',dir='/private/tmp') as td:
 d=Path(td);c,guard=initialize(old.NEW,d/'context',beam=5.,auxiliary=True)
 previous=json.loads(old.GUARD.read_text())
 assert guard==previous,'guard changed'
 m=measurement(c,c['original'],c['support'],guard,from_map(old.NEW,c));assert m['auxiliary_support']['available']
 uncertainties=endpoint(old.NEW,c,m,d/'endpoint')
 assert all(q['conditional_sigma'] is not None for q in uncertainties),uncertainties
 records['current_pipeline']=dict(guard_exact=True,auxiliary=c['auxiliary'],measurement=m,uncertainties=uncertainties)
Path(sys.argv[1]).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:{key:v[key] for key in ('shape','projection','outside_feedback_pixels','normalization')} for k,v in records.items() if k!='current_pipeline'},indent=2))
