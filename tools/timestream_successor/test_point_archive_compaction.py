import tempfile, unittest
from pathlib import Path
from point_archive_compaction import digest,identity,changed_rejection,validate_closure,check_scope,resolve,restore,importance_ratio,fraction,explain_missing

class ArchiveTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.r=self.root/'owned';self.p=self.r/'run/a/native/products/cleaned.f64';self.q=self.r/'run/b/native/products/cleaned.f64';self.p.parent.mkdir(parents=True);self.q.parent.mkdir(parents=True);self.p.write_bytes(b'original');self.q.write_bytes(b'original');self.a=dict(path=str(self.p),retained=str(self.q),sha256=digest(self.p),bytes=8)
 def tearDown(self):self.tmp.cleanup()
 def test_units_and_denominators(self):
  self.assertAlmostEqual(importance_ratio(-.103601,6.49,'mJy/beam','mJy/beam','conditional_reference'),.0159631741,places=8)
  self.assertAlmostEqual(fraction(776.46,85320,'own_reference_source'),.9100563,places=5)
  with self.assertRaises(ValueError):importance_ratio(1,2,'brightness-area','mJy/beam','historical_context')
  with self.assertRaises(ValueError):fraction(1,2,'whole-source-or-increment')
  self.assertIsNone(importance_ratio(1,None,'x','x','unavailable'))
 def test_protected_and_changed_rejection(self):
  with self.assertRaises(ValueError):check_scope(self.root/'original/cleaned.f64',(self.r,))
  s=identity(self.p);self.p.write_bytes(b'altered')
  with self.assertRaises(ValueError):changed_rejection(self.p,s)
 def test_dependency_closure(self):
  self.assertTrue(validate_closure([self.a],(self.r,)))
  b=dict(self.a,path=str(self.q),retained=str(self.p))
  with self.assertRaises(ValueError):validate_closure([self.a,b],(self.r,))
 def test_report_access_and_explicit_restore(self):
  self.p.unlink();r=resolve(self.p,[self.a]);self.assertEqual(r['path'],str(self.q));self.assertFalse(self.p.exists())
  restore(self.p,[self.a]);self.assertEqual(self.p.read_bytes(),b'original')
  with self.assertRaises(FileNotFoundError):resolve(self.root/'unrecorded',[self.a])
 def test_corrupt_replacement_rejection(self):
  self.p.unlink();self.q.write_bytes(b'corrupt')
  with self.assertRaises(ValueError):resolve(self.p,[self.a])
 def test_corrupt_existing_artifact_rejection(self):
  self.p.write_bytes(b'corrupt')
  with self.assertRaises(ValueError):resolve(self.p,[self.a])
 def test_symlink_rejection(self):
  self.p.unlink();self.p.symlink_to(self.q)
  with self.assertRaises(ValueError):identity(self.p)
 def test_explicit_missing_status_without_inference(self):
  self.p.unlink()
  with self.assertRaisesRegex(FileNotFoundError,'INTENTIONALLY COMPACTED'):explain_missing(self.p,[self.a])
  with self.assertRaisesRegex(FileNotFoundError,'missing/corrupt'):explain_missing(self.root/'unknown',[self.a])
  self.assertFalse(self.p.exists())

if __name__=='__main__':unittest.main()
