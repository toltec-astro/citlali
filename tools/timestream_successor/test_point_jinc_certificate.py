"""Exact rational arithmetic tests of the actual native bound propagation.

The fixture's exact target coefficients are constructed from the binary64 cache
and an explicit error enclosure, so Fraction is an exact oracle. Analytic cache
proof is separately supplied by point_jinc_interval, never inferred here.
"""
import json,os,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from inspect_forward_map import digest,read_yaml
BINARY=Path(os.environ.get('CITLALI_JINC_DIAGNOSTIC',str(Path(__file__).resolve().parents[2]/'build/bin/citlali_jinc_diagnostic')))
class NativeCertificate(unittest.TestCase):
 def test_exact_rational_targets_signed_and_parallel(self):
  self.assertTrue(BINARY.is_file(),'required native fixture missing')
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);spec=dict(rows=7,columns=7,phases=16,pixel_matrix_rad=[-1,0,0,1],scale_rad=1.,a=1.1,b=1.67,c=2.,rmax=3.)
   def bound(p):return dict(path=str(p),sha256=digest(p))
   def run(q,name):
    p=root/(name+'.json');p.write_text(json.dumps(q));subprocess.run([str(BINARY),str(p),str(root/name)],check=True,capture_output=True);return root/name
   q=dict(schema='point-152389-jinc-diagnostic-v1',mode='cache',spec=spec);cache=run(q,'cache')/'cache.f64';k=np.fromfile(cache,dtype='<f8').reshape(16,16,7,7)
   ek=np.full(k.shape,1e-14);ep=root/'errors.f64';ek.tofile(ep)
   samples=np.array([[3.,3.,1.25,1.,8.],[3.5,4.25,-.75,2.,16.],[4.25,3.25,3.125,.5,4.],[np.nextafter(2.5,0),2.,-.0625,1.,32.]])
   sp=root/'samples.f64';samples.tofile(sp)
   q.update(mode='accumulate',profile='POINT-152389-JINC-NUMERICAL-ADEQUACY-v1',coefficient_errors=bound(ep),cache_sha256=digest(cache),sample_payload=bound(sp),samples=len(samples),frozen_peak=1000.)
   exact=[[F(0) for _ in range(4)] for _ in range(49)]
   for x,y,t,w,fs in samples:
    fx,fy=F(float(x)),F(float(y));cx=(fx+F(1,2)).numerator//(fx+F(1,2)).denominator;cy=(fy+F(1,2)).numerator//(fy+F(1,2)).denominator
    qx=int(16*(fx-cx+F(1,2)));qy=int(16*(fy-cy+F(1,2)))
    for py in range(1,8):
     for px in range(1,8):
      dx,dy=px-cx,py-cy
      if abs(dx)>3 or abs(dy)>3:continue
      # Every constructed target lies inside its declared coefficient bound.
      kk=F(float(k[qy,qx,dy+3,dx+3]))+F(float(ek[qy,qx,dy+3,dx+3]))/2
      z=exact[(py-1)*7+px-1];z[0]+=F(w)*kk*F(t);z[1]+=F(w)*kk;z[2]+=F(w)*kk*kk;z[3]+=kk*kk/F(fs)
   for workers in (1,3):
    q['workers']=workers;out=run(q,'grid'+str(workers));planes=[np.fromfile(out/(n+'.f64'),dtype='<f8') for n in ('N','C','Q','T')];errors=[np.fromfile(out/f'error-{i}.f64',dtype='<f8') for i in range(5)];m=np.fromfile(out/'m.f64',dtype='<f8');causes=np.fromfile(out/'causes.u8',dtype='u1')
    for p,row in enumerate(exact):
     for i in range(4):self.assertLessEqual(abs(F(float(planes[i][p]))-row[i]),F(float(errors[i][p])))
     if causes[p]==0:
      self.assertLessEqual(abs(F(float(m[p]))-row[0]/row[1]),F(float(errors[4][p])))
      self.assertLessEqual(F(float(errors[4][p])),F(1,1000000)*(1000+abs(F(float(m[p])))))
    self.assertGreater(np.count_nonzero(causes==0),0)

   # Deliberate small denominator must fail separation, without an arbitrary rho.
   samples=np.array([[3.,3.,1.,1.,8.],[4.,3.,2.,-k[8,8,3,3]/k[8,8,3,2],8.]])
   self.assertGreater(samples[1,3],0);samples.tofile(sp);q.update(sample_payload=bound(sp),samples=2,workers=1)
   out=run(q,'cancel');self.assertTrue(np.fromfile(out/'causes.u8',dtype='u1')[16]&2)
   samples=np.array([[3.,3.,1e-320,1.,8.]]);samples.tofile(sp);q.update(sample_payload=bound(sp),samples=1)
   out=run(q,'subnormal');causes=np.fromfile(out/'causes.u8',dtype='u1');self.assertTrue(np.all(causes!=0));self.assertTrue(np.any(causes&1))
