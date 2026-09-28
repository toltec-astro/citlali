import unittest,tempfile,json
from pathlib import Path
from point_step_experiment import consume,consume_consequences,LIMITS
class CounterTest(unittest.TestCase):
 def test_absolute_and_duplicate_limits(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)
   for i in range(5):consume(r,'parent',dict(run=str(i)))
   with self.assertRaises(Exception):consume(r,'parent',dict(run='5'))
   (r/'DEFECT-REPLACEMENT.json').write_text(json.dumps(dict(implementation_defect='fixture routing',corrected_source='fixture',maximum_replacements=1)))
   consume(r,'parent',dict(run='5'))
   with self.assertRaises(Exception):consume(r,'parent',dict(run='6'))
   consume(r,'rtc_only',dict(run='a'))
   with self.assertRaises(Exception):consume(r,'rtc_only',dict(run='a'))
   consume(r,'rtc_only',dict(run='b'))
   with self.assertRaises(Exception):consume(r,'rtc_only',dict(run='c'))
 def test_consequences_caps(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)
   for kind,cap in dict(parent=4,trajectory=4,feedback=20,endpoint=4,rtc_only=1).items():
    for i in range(cap):consume_consequences(r,kind,dict(run=str(i)))
    with self.assertRaises(Exception):consume_consequences(r,kind,dict(run='extra'))
   (r/'DEFECT-REPLACEMENT.json').write_text(json.dumps(dict(implementation_defect='fixture',corrected_source='fixture',maximum_replacements=1)))
   for kind,cap in dict(parent=1,trajectory=1,feedback=5,endpoint=1,rtc_only=0).items():
    for i in range(cap):consume_consequences(r,kind,dict(run='replacement'+str(i)))
    with self.assertRaises(Exception):consume_consequences(r,kind,dict(run='over'))
 def test_native_grid(self):
  from point_rtc_response import native_rows
  import numpy as np
  for factor,phase in ((1,0),(2,0),(2,1)):
   receipt={'donor_continuity':{'realized_detectors':[dict(factor=factor,phase_native_rows=phase)]}}
   np.testing.assert_array_equal(native_rows(receipt,[0,1,7]),np.array([0,1,7])*factor+phase)
 def test_no_legacy_budget(self):
  self.assertEqual(LIMITS,dict(parent=5,trajectory=5,feedback=25,endpoint=5,rtc_only=2))
if __name__=='__main__':unittest.main()
