import unittest
from decimal import Decimal as D,localcontext
import math
from point_jinc_interval import I,pi,jinc,first_root,kernel_radius
class SelectedJincIntervals(unittest.TestCase):
 def test_pi_root_and_zero(self):
  # Published decimal prefixes are only regression checks; proof is the series.
  p=pi();self.assertLess(p.lo,D('3.14159265358979323846264338327950288419716939937510582097494459'));self.assertGreater(p.hi,D('3.14159265358979323846264338327950288419716939937510582097494460'))
  root=first_root();self.assertGreater(jinc(I(root.lo)).lo,0);self.assertLess(jinc(I(root.hi)).hi,0)
  v=jinc(I(0));self.assertEqual(v.lo,1);self.assertEqual(v.hi,1)
 def test_directed_operations_and_ambient_context(self):
  v=I(1)/3;self.assertLess(v.lo*3, D('1.000000000000000000000000001'))
  with localcontext() as c:
   c.prec=7
   r=kernel_radius(I('1.23'),'1.1','1.67')
  s=kernel_radius(I('1.23'),'1.1','1.67');self.assertEqual((r.lo,r.hi),(s.lo,s.hi))
  with self.assertRaises(ValueError):I(1)/I(-1,1)
 def test_signed_kernel_and_enclosure_width(self):
  for r in ('0.1','0.7','1.3','2.7','3.9','4.7'):
   v=kernel_radius(I(r),'1.1','1.67');self.assertLess(v.hi-v.lo,D('1e-35'))
  self.assertLess(kernel_radius(I('.8'),'1.1','1.67').hi,0)
