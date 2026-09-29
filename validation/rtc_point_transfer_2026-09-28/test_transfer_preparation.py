"""Small nonobservational admission/accounting tests."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_transfer_experiment import consume,LIMITS,ROLES,CAP,ACTIVE,ROLE_CAP,CAMPAIGN_METADATA,REPORT_CAP
from point_processing import rtc_experiment,feedback_count
class Preparation(unittest.TestCase):
 def test_exact_budget_and_no_duplicates(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for role in ROLES:
    consume(root,'parent',dict(run=role));consume(root,'trajectory',dict(run=role,branch='REF',cycle=0))
    for k in range(1,6):consume(root,'feedback',dict(run=role,branch='REF',cycle=k))
    consume(root,'endpoint',dict(run=role,cycle=5))
   for kind in LIMITS:
    with self.assertRaises(Exception):consume(root,kind,dict(run=ROLES[0]))
   for kind in ('rtc_only','replacement','noise'):
    with self.assertRaises(Exception):consume(root,kind,dict(run=ROLES[0]))
 def test_foreign_or_duplicate_role(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaises(Exception):consume(root,'parent',dict(run='B-REF-rep2'))
   consume(root,'parent',dict(run='B-REF'))
   with self.assertRaises(Exception):consume(root,'parent',dict(run='B-REF'))
   with self.assertRaises(Exception):consume(root,'feedback',dict(run='B-REF',branch='REF',cycle=6))
 def test_full_capacity_arithmetic(self):
  prior=1_856_251_212
  self.assertEqual(prior+7*ROLE_CAP+ACTIVE+CAMPAIGN_METADATA,29_968_705_628)
  self.assertLessEqual(prior+8*ROLE_CAP+REPORT_CAP+CAMPAIGN_METADATA,CAP)
 def test_new_route_is_rtc_and_five_cycles(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'freeze').mkdir();(root/'freeze/prospective.json').write_text(json.dumps(dict(schema='citlali-point-speed-pca-v1',campaign_scope='rtc-point-transfer-v1',feedback_count=5)))
   self.assertTrue(rtc_experiment(root));self.assertEqual(feedback_count(root),5)
if __name__=='__main__':unittest.main()
