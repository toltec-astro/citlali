import importlib.util
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('cleanup',Path(__file__).with_name('cleanup_native.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Safety(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir='/private/tmp');self.root=Path(self.tmp.name).resolve();p=self.root/'B-REF-rep1/upstream/a.f64';p.parent.mkdir(parents=True);p.write_bytes(b'original');self.p=p
  self.row=dict(path=str(p),bytes=p.stat().st_size,sha256=m.digest(p));self.plan=dict(root=str(self.root),files=[self.row],bytes=self.row['bytes'])
 def tearDown(self):self.tmp.cleanup()
 def test_exact_valid_entry_and_hash(self):
  self.assertEqual(m.validate_entries(self.plan,{}),{str(self.p)});m.verify_native([self.plan]);self.assertTrue(self.p.exists())
 def test_foreign_stage_and_protected_rejected(self):
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{str(self.p):self.row['sha256']})
  q=self.root/'B-REF-rep1/compact/a.f64';q.parent.mkdir();q.write_bytes(b'original');self.plan['files']=[dict(self.row,path=str(q))]
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{})
 def test_symlink_and_outside_root_rejected(self):
  q=self.p.with_name('link.f64');q.symlink_to(self.p);self.plan['files']=[dict(self.row,path=str(q))]
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{})
  self.plan['files']=[dict(self.row,path='/private/tmp/outside.f64')]
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{})
 def test_duplicate_and_missing_rejected(self):
  self.plan['files']*=2;self.plan['bytes']*=2
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{})
  self.plan['files']=[dict(self.row,path=str(self.p.with_name('missing.f64')))];self.plan['bytes']=self.row['bytes']
  with self.assertRaises(RuntimeError):m.validate_entries(self.plan,{})
 def test_hash_failure_changes_nothing(self):
  q=self.p.with_name('b.f64');q.write_bytes(b'other');self.plan['files'].append(dict(path=str(q),bytes=5,sha256='0'*64));self.plan['bytes']+=5
  with self.assertRaises(RuntimeError):m.verify_native([self.plan])
  self.assertEqual(self.p.read_bytes(),b'original');self.assertEqual(q.read_bytes(),b'other')
if __name__=='__main__':unittest.main()
