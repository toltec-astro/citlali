"""Exact scientific comparison and interruption recovery without a reduction."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import yaml
import compare_successor_outputs as old
import verify_successor_campaign as recovery

GOLDEN='c2e379ed8cb8cbf5e58c8300f0888f96b142ae791fea2a38ca532dfc46a1aa15'
DOCUMENT='a: [1, true, 1.5, null, hello]\nb: {z: -0.0, a: .nan}\nwall_seconds: 4.3\nwindows: [{usable: true, duration_seconds: 4}]\n'

class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'data';self.root.mkdir()
        (self.root/'receipt.yaml').write_text(DOCUMENT)
        (self.root/'x.f64').write_bytes(b'\x00'*16)
        self.out=self.base/'evidence'
    def snap(self,root=None,out=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return recovery.snapshot(root or self.root,out or self.out)
    def parse(self,details=True):return old.Parser(self.root/'receipt.yaml',self.root,details).parse()
    def test_preoptimization_golden_digest_and_optional_details(self):
        full=self.parse();small=self.parse(False)
        self.assertEqual(full['semantic_sha256'],GOLDEN)
        self.assertEqual({k:v for k,v in full.items() if k!='subtrees'},
                         {k:v for k,v in small.items() if k!='subtrees'})
        self.assertTrue(full['subtrees']);self.assertEqual(small['subtrees'],{})
    def test_scalar_cache_bounded_and_cleared(self):
        p=old.Parser(self.root/'receipt.yaml',self.root)
        for i in range(9000):p.scalar_hash('str',str(i))
        self.assertEqual(p._scalar_hash.cache_info().currsize,8192)
        p.scalar_hash('str','x'*1025)
        self.assertEqual(p._scalar_hash.cache_info().currsize,8192)
        p.parse();self.assertEqual(p._scalar_hash.cache_info().currsize,0)
    def test_only_declared_normalizations(self):
        before=self.parse()['semantic_sha256']
        (self.root/'receipt.yaml').write_text('windows:\n - duration_seconds: 4\n   usable: true\nwall_seconds: 10\nb: {a: .nan, z: -0.0}\na: [1, true, 1.5, null, hello]\n')
        self.assertEqual(self.parse()['semantic_sha256'],before)
        (self.root/'receipt.yaml').write_text(DOCUMENT.replace('duration_seconds: 4','duration_seconds: 5'))
        self.assertNotEqual(self.parse()['semantic_sha256'],before)
    def test_deep_validity_difference_retained_without_subtree_details(self):
        a=self.snap();(self.root/'receipt.yaml').write_text('windows: [{a: {b: {c: {usable: false}}}}]\n')
        b=self.snap(out=self.base/'candidate')
        self.assertFalse(recovery.compare_snapshots(a,b)['pass_equivalence'])
    def test_binary_difference_detected(self):
        a=self.snap();(self.root/'x.f64').write_bytes(b'\x01'*16)
        b=self.snap(out=self.base/'candidate')
        self.assertFalse(recovery.compare_snapshots(a,b)['pass_equivalence'])
    def test_inventory_difference_detected(self):
        a=self.snap();(self.root/'extra.u8').write_bytes(b'1');b=self.snap(out=self.base/'candidate')
        with self.assertRaisesRegex(ValueError,'inventory'):recovery.compare_snapshots(a,b)
    def test_reuse_rehashes_and_does_not_parse_again(self):
        a=self.snap()
        with patch.object(old.Parser,'parse',side_effect=AssertionError('parsed again')),patch.object(old,'sha',wraps=old.sha) as hashing:
            b=self.snap()
        self.assertTrue(recovery.compare_snapshots(a,b)['pass_equivalence'])
        self.assertEqual(b['checkpoints_reused'],2)
        self.assertIn(self.root/'x.f64',[c.args[0] for c in hashing.call_args_list])
        self.assertEqual(len(list(self.out.rglob('*.json'))),5)
    def test_changed_input_blocks_reuse(self):
        self.snap();(self.root/'receipt.yaml').write_text('changed: true\n')
        with self.assertRaisesRegex(ValueError,'input content changed'):self.snap()
    def test_changed_tool_blocks_reuse(self):
        self.snap()
        with patch.object(recovery,'tool_binding',return_value={'changed':1}):
            with self.assertRaisesRegex(ValueError,'source/tool/inventory'):self.snap()
    def test_corrupt_checkpoint_blocks_reuse(self):
        self.snap();p=next((self.out/'yaml').glob('*.json'));v=json.loads(p.read_text())
        v['record']['yaml']['semantic_sha256']='forged';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'checkpoint integrity'):self.snap()
    def test_interrupted_snapshot_resumes_completed_document(self):
        (self.root/'z.yaml').write_text('a: 3\n')
        original=old.Parser.parse
        def interrupt(parser):
            if parser.path.name=='z.yaml':raise RuntimeError('interrupted')
            return original(parser)
        with patch.object(old.Parser,'parse',interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.snap()
        self.assertFalse((self.out/'snapshot.json').exists())
        (self.out/'unused.tmp-1').write_text('incomplete')
        seen=[]
        def record(parser):seen.append(parser.path.name);return original(parser)
        with patch.object(old.Parser,'parse',record):self.snap()
        self.assertEqual(seen,['z.yaml'])
    def test_input_changed_during_parse_fails(self):
        original=old.Parser.parse
        def change(parser):
            result=original(parser);parser.path.write_text(DOCUMENT+'changed: true\n');return result
        with patch.object(old.Parser,'parse',change):
            with self.assertRaisesRegex(ValueError,'input changed while'):self.snap()
    def test_cal_parent_still_verified_and_dependency_bound_on_resume(self):
        cal=self.root/'donor-continuity/cal/receipt.yaml';cal.parent.mkdir(parents=True);cal.write_text('a: 3\n')
        (self.root/'receipt.yaml').write_text('source_CAL_receipt_sha256: '+old.sha(cal)+'\n')
        self.snap();cal.write_text('a: 4\n')
        with self.assertRaisesRegex(ValueError,'source/tool/inventory'):self.snap()
        with self.assertRaisesRegex(AssertionError,'CAL receipt digest'):self.snap(out=self.base/'candidate')
    def test_bad_yaml_not_converted_to_pass(self):
        (self.root/'receipt.yaml').write_text('a: 1\na: 2\n')
        with self.assertRaisesRegex(AssertionError,'duplicate key'):self.snap()
    def test_output_cannot_be_inside_input(self):
        with self.assertRaisesRegex(ValueError,'inside scientific'):self.snap(out=self.root/'verification')
    def test_source_arrays_and_yaml_unchanged(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()};self.snap()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

class CompletedRunBinding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.expected=dict(runtime_revision='5cd3e28396a08917986f3ec73314a69d3605bd0d',executable_sha256='binary',networks=[0,4],runs={})
        for n in recovery.SETTINGS:
            r=self.root/f'workers-{n}';(r/'products').mkdir(parents=True)
            (r/'RUN.json').write_text(json.dumps(dict(workers=n,exit_code=2,executable_sha256='binary',command=['never-execute',str(r/'development.yaml')])))
            receipt=dict(source_revision='5cd3e2839',exit_code=2,requested_network_workers=n,requested_internal_threads=1,observed_Eigen_threads=1,started_network_workers=min(n,2),networks=[dict(network=i,input={'sha256':str(i)},exit_code=2 if i==4 else 0,state='partial' if i==4 else 'completed',output=str(r/'products'/f'network{i}')) for i in (0,4)])
            (r/'products/observation-receipt.yaml').write_text(yaml.safe_dump(receipt));self.rebind(n)
    def rebind(self,n):
        r=self.root/f'workers-{n}'
        self.expected['runs'][str(n)]={name:old.sha(r/name) for name in ('RUN.json','products/observation-receipt.yaml')}
    def test_completed_partial_disposition_and_exact_published_spelling(self):
        networks,roots=recovery.preflight(self.root,self.expected)
        self.assertEqual(networks,[0,4]);self.assertEqual(roots['12'],str(self.root/'workers-12'))
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        self.assertEqual(recovery.preflight(alias,self.expected),(networks,roots))
    def test_changed_receipt_rejected(self):
        p=self.root/'workers-2/products/observation-receipt.yaml';p.write_text(p.read_text()+'unknown: 1\n')
        with self.assertRaisesRegex(ValueError,'run binding changed'):recovery.preflight(self.root,self.expected)
    def test_population_or_disposition_difference_rejected_even_with_binding(self):
        p=self.root/'workers-2/products/observation-receipt.yaml';d=yaml.safe_load(p.read_text());d['networks'][1]['exit_code']=0;p.write_text(yaml.safe_dump(d));self.rebind(2)
        with self.assertRaisesRegex(ValueError,'population/input/disposition'):recovery.preflight(self.root,self.expected)
    def test_wrong_binary_rejected(self):
        self.expected['executable_sha256']='different'
        with self.assertRaisesRegex(ValueError,'binary identity'):recovery.preflight(self.root,self.expected)
    def test_complete_network_compares_all_four_without_executing_runtime(self):
        roots={str(n):str(self.root/f'workers-{n}') for n in recovery.SETTINGS}
        for n in recovery.SETTINGS:
            p=Path(roots[str(n)])/'products/network0';p.mkdir();(p/'x.f64').write_bytes(b'1234')
        with contextlib.redirect_stdout(io.StringIO()):
            result=recovery.verify_network(self.root,self.root/'verification',0,roots)
        self.assertEqual([r['workers'] for r in result['comparisons']],[2,4,8,12])
        self.assertTrue(all(r['pass_equivalence'] for r in result['comparisons']))
    def test_parallel_cli_resume_and_scientific_failure(self):
        for n in recovery.SETTINGS:
            for network in (0,4):
                p=self.root/f'workers-{n}/products/network{network}';p.mkdir()
                (p/'x.f64').write_bytes(b'1234')
                (p/'receipt.yaml').write_text('science: [1, 2, 3]\n')
        expected=self.root/'expected.json';expected.write_text(json.dumps(self.expected))
        output=self.root/'verification'
        command=[sys.executable,recovery.__file__,'--campaign',str(self.root),'--output',str(output),
                 '--expected',str(expected),'--workers','2']
        for _ in range(2):
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
        status=json.loads((output/'STATUS.json').read_text())
        self.assertEqual(status['state'],'PASS');self.assertEqual(len(status['networks_completed']),2)
        (self.root/'workers-12/products/network4/x.f64').write_bytes(b'different')
        # A fresh verification must detect a difference; an existing checkpoint
        # must also reject input mutation rather than reusing yesterday's pass.
        for target in (output,self.root/'new-verification'):
            command[command.index('--output')+1]=str(target)
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,1,run.stderr)
            status=json.loads((target/'STATUS.json').read_text())
            self.assertEqual(status['state'],'FAIL');self.assertEqual(status['failures'][0]['network'],4)
            self.assertEqual(status['networks_completed'][0]['network'],0)

if __name__=='__main__':unittest.main()
