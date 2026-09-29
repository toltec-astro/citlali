"""Small nonobservational admission/accounting tests."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
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
class TerminalEvidence(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  spec=importlib.util.spec_from_file_location('transfer_runner',Path(__file__).with_name('run_campaign.py'))
  cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)
 def test_partial_evidence_is_unknown_and_success_is_strict(self):
  r=self.runner
  with tempfile.TemporaryDirectory() as d,patch.object(r,'ROOT',Path(d)):
   receipt=Path(d)/'B-REF/upstream/products/network1/receipt.yaml';receipt.parent.mkdir(parents=True);receipt.write_text('')
   (Path(d)/'attempts.jsonl').write_text(json.dumps(dict(kind='parent'))+'\n{"kind":')
   with self.assertRaises(Exception):r.accounting()
   counts=r.accounting(strict=False)
   self.assertEqual(counts['charged_attempts']['parent'],1)
   self.assertFalse(counts['accounting_complete']);self.assertEqual(counts['completed_native_RTC_Apply_receipts'],0)
   self.assertTrue(any(q.get('line')==2 for q in counts['unknown_evidence']))
   self.assertTrue(any(q['path']==str(receipt) for q in counts['unknown_evidence']))
   (Path(d)/'attempts.jsonl').write_text(json.dumps(dict(kind='parent'))+'\n')
   with self.assertRaises(Exception):r.accounting() # Empty receipt remains a strict failure.
 def test_terminal_status_survives_accounting_and_log_failure(self):
  r=self.runner
  for state in ('INTERRUPTED','PROCESS_FAILED','REQUIRED_GATE_FAILED'):
   with self.subTest(state=state),tempfile.TemporaryDirectory() as d,patch.object(r,'ROOT',Path(d)),patch.object(r,'STATE',{}),patch.object(r,'accounting',side_effect=OSError('partial evidence')):
    (Path(d)/'failure.log').mkdir() # Force the independent logging path to fail too.
    self.assertEqual(r.terminal_failure(state,130,'synthetic interruption',RuntimeError('fixture')),130)
    saved=json.loads((Path(d)/'STATUS.json').read_text())
    self.assertEqual(saved['state'],state);self.assertEqual(len(saved['terminal_evidence_errors']),2)
 def test_actual_handlers_publish_terminal_status(self):
  r=self.runner
  for failure,state,code in ((KeyboardInterrupt(),'INTERRUPTED',130),(r.ProcessFailure('synthetic'),'PROCESS_FAILED',1),(ValueError('synthetic'),'REQUIRED_GATE_FAILED',1)):
   with self.subTest(state=state),tempfile.TemporaryDirectory() as d:
    root=Path(d);review=root/'PREFLIGHT_REVIEW.json';review.write_text(json.dumps(dict(source_revision='fixture',verdict='pass')))
    def git(args,**kwargs):return 'fixture' if args[1]=='rev-parse' else ''
    with patch.object(r,'ROOT',root),patch.object(r,'STATE',{}),patch.object(r.subprocess,'check_output',side_effect=git),patch.object(r,'capacity',return_value={}),patch.object(r,'bound',side_effect=lambda p:dict(path=str(p))),patch.object(r,'command',side_effect=failure),patch('psutil.Process') as process,patch.object(sys,'argv',['run_campaign.py','--review',str(review)]):
     process.return_value.children.return_value=[]
     # A partial native receipt left by an interrupted writer must not mask status.
     original=r.accounting
     def partial_accounting(strict=True):
      receipt=root/'B-REF/upstream/products/network1/receipt.yaml';receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text('Apply_performed: [')
      (root/'attempts.jsonl').write_text('{')
      return original(strict=strict)
     with patch.object(r,'accounting',side_effect=partial_accounting):self.assertEqual(r.main(),code)
     self.assertEqual(json.loads((root/'STATUS.json').read_text())['state'],state)
     self.assertFalse(json.loads((root/'ACCOUNTING.json').read_text())['accounting_complete'])
 def test_preflight_only_stops_before_any_science(self):
  r=self.runner
  with tempfile.TemporaryDirectory() as d,patch.object(r,'ROOT',Path(d)),patch.object(r,'STATE',{}),patch.object(r.subprocess,'check_output',side_effect=lambda args,**kw:'fixture' if args[1]=='rev-parse' else ''),patch.object(r,'capacity',return_value={}),patch.object(r,'bound',side_effect=lambda p:dict(path=str(p))),patch.object(r,'run_preflight') as gates,patch.object(r,'prepare_role') as prepare,patch.object(r,'launch') as launch,patch.object(sys,'argv',['run_campaign.py','--preflight-only']):
   self.assertEqual(r.main(),0);gates.assert_called_once_with('fixture');prepare.assert_not_called();launch.assert_not_called()
   root=Path(d)/'preflight-only/fixture';self.assertEqual(json.loads((root/'STATUS.json').read_text())['state'],'PREFLIGHT_PASSED_NO_SCIENCE')
   self.assertEqual(json.loads((root/'ACCOUNTING.json').read_text())['charged_attempts'],dict.fromkeys(LIMITS,0))
   self.assertFalse((Path(d)/'LAUNCHED.json').exists())
 def test_configure_precedes_requested_build_targets(self):
  r=self.runner;seen=[]
  def stop_after_build(name,args):
   seen.append((name,args))
   if name=='build':raise RuntimeError('fixture stop')
  with patch.object(r,'command',side_effect=stop_after_build):
   with self.assertRaisesRegex(RuntimeError,'fixture stop'):r.run_preflight('fixture')
  self.assertEqual(seen[0],('configure',['cmake','-S','.', '-B','build']))
  self.assertEqual(seen[1][0],'build');self.assertIn('citlali_rtc_point_select',seen[1][1])
 def test_limits_precede_scientific_imports(self):
  script="""import builtins,os,runpy,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
 if name=='point_transfer_experiment':
  assert 'numpy' not in sys.modules
  for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):assert os.environ[k]=='1'
  assert os.environ['MPLBACKEND']=='Agg'
  raise SystemExit(0)
 return original(name,*args,**kwargs)
builtins.__import__=checked
runpy.run_path(sys.argv[1],run_name='import_order_fixture')
raise AssertionError('scientific helper import was not reached')
"""
  env=dict(os.environ,OMP_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',MPLBACKEND='MacOSX')
  subprocess.run([sys.executable,'-c',script,str(Path(__file__).with_name('run_campaign.py'))],env=env,check=True)
if __name__=='__main__':unittest.main()
