import unittest,tempfile,json
from pathlib import Path
from point_step_experiment import consume,LIMITS
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
 def test_no_legacy_budget(self):
  self.assertEqual(LIMITS,dict(parent=5,trajectory=5,feedback=25,endpoint=5,rtc_only=2))
if __name__=='__main__':unittest.main()
