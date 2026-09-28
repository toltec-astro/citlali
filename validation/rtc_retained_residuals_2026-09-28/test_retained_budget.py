import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/timestream_successor'))
from point_step_experiment import consume_retained
class Budget(unittest.TestCase):
 def test_no_sixth_parent_or_duplicate(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for i in range(5):consume_retained(root,'parent',dict(run=i))
   with self.assertRaises(Exception):consume_retained(root,'parent',dict(run=5))
   with self.assertRaises(Exception):consume_retained(root,'parent',dict(run=0))
   self.assertEqual(len((root/'attempts.jsonl').read_text().splitlines()),5)
 def test_no_replacement_or_rtc_only(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with self.assertRaises(Exception):consume_retained(root,'rtc_only',dict(run=0))
   (root/'DEFECT-REPLACEMENT.json').write_text('{}')
   with self.assertRaises(Exception):consume_retained(root,'parent',dict(run=0))
 def test_exact_feedback_cap(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for i in range(25):consume_retained(root,'feedback',dict(cycle=i))
   with self.assertRaises(Exception):consume_retained(root,'feedback',dict(cycle=25))
if __name__=='__main__':unittest.main()
