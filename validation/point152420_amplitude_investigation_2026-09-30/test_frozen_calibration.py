import unittest
import numpy as np
from frozen_calibration import apply_basis,normalization,segments,checkpoint
class FrozenCalibrationTests(unittest.TestCase):
 def test_observed_projection_and_scale_linearity(self):
  rng=np.random.default_rng(159);b=rng.normal(size=(9,2));v=rng.normal(size=(30,9));eligible=np.ones_like(v,bool);eligible[:10,0]=False;v[~eligible]=np.nan
  clean,ok,_=apply_basis(v,eligible,b)
  means=np.nanmean(v,axis=0)
  for t in range(30):
   j=eligible[t];z=v[t,j]-means[j];coeff=np.linalg.lstsq(b[j],z,rcond=1e-5)[0]
   np.testing.assert_allclose(clean[t,j],z-b[j]@coeff,atol=1e-13)
  scaled,ok2,_=apply_basis(v*3,eligible,b);np.testing.assert_array_equal(ok,ok2);np.testing.assert_allclose(scaled[ok],3*clean[ok],atol=1e-13)
 def test_population_and_rank_loss(self):
  b=np.eye(3);v=np.arange(12.).reshape(4,3);mask=np.ones_like(v,bool);mask[0,0]=False
  _,ok,d=apply_basis(v,mask,b);self.assertFalse(ok[0].any());self.assertEqual(d['rank_failed_times'],1)
 def test_exact_native_normalization_order_statistic(self):
  count=np.array([0,1,2,3,4,5,6,7,100]);support,ref=normalization(count)
  self.assertEqual(ref,100);np.testing.assert_array_equal(support,[False]*8+[True])
 def test_one_local_native_segment_replay(self):
  nw,sid,p,s,col=next(segments());a=checkpoint(sid,p,s)
  clean,valid,d=apply_basis(a['y'],a['eligible'],a['basis']);native=a['native'][:,a['cols']];old=a['causes'][:,a['cols']]==0
  np.testing.assert_array_equal(valid,old);np.testing.assert_allclose(clean[old],native[old],atol=1e-6,rtol=1e-10)
if __name__=='__main__':unittest.main()
