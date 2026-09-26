import tempfile,unittest
from pathlib import Path
import netCDF4,numpy as np
import local_data_purge as p
class PurgeSafetyTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.previous=p.ROOT;p.ROOT=self.root
 def tearDown(self):p.ROOT=self.previous;self.temp.cleanup()
 def test_outside_and_symlink_rejected(self):
  a=self.root/'data';a.write_bytes(b'raw');s=self.root/'link';s.symlink_to(a)
  with self.assertRaises(ValueError):p.identity(s)
  with self.assertRaises(ValueError):p.identity(Path('/etc/hosts'))
 def test_changed_file_rejected(self):
  f=self.root/'x';f.write_bytes(b'abc');r=dict(path='x',**p.identity(f));f.write_bytes(b'changed')
  with self.assertRaises(ValueError):p.unchanged(r)
 def fixture(self):
  f=self.root/'mixed.nc'
  with netCDF4.Dataset(f,'w')as d:
   d.createDimension('n_pts',5);d.createDimension('n_dets',2);d.createDimension('rank',1)
   d.title='test';d.createVariable('signal','f4',('n_pts','n_dets'))[:]=np.arange(10).reshape(5,2)
   a=d.createVariable('basis','f8',('n_dets','rank'));a[:]=[[np.nan],[-0.0]];a.units='dimensionless'
   d.createVariable('apt_gain','f8',('n_dets',))[:]=[2,3]
   d.createVariable('producer',str,('rank',))[0]='fixture'
  return dict(path=f.name,**p.identity(f),sample_dimension='n_pts',removed_variables=['signal'],retained_variables=['basis','apt_gain','producer'],retained_companion='mixed_purge_retained.nc')
 def test_mixed_preserves_basis_calibration_nan_and_strings(self):
  row=self.fixture();result=p.separate_one(row);self.assertEqual(result['status'],'pass',result);self.assertTrue((self.root/row['path']).exists())
  with netCDF4.Dataset(self.root/row['retained_companion'])as d:
   self.assertNotIn('signal',d.variables);self.assertEqual(set(d.variables),{'basis','apt_gain','producer'});self.assertTrue(np.isnan(d['basis'][0,0]));self.assertTrue(np.signbit(d['basis'][1,0]));self.assertEqual(d['basis'].units,'dimensionless');self.assertEqual(d['producer'][0],'fixture');np.testing.assert_array_equal(d['apt_gain'][:],[2,3])
 def test_no_overwrite_or_implicit_source_deletion(self):
  row=self.fixture();f=self.root/row['retained_companion'];f.write_bytes(b'protected')
  result=p.separate_one(row);self.assertEqual(result['status'],'retained mixed-container exception');self.assertEqual(f.read_bytes(),b'protected');self.assertTrue((self.root/row['path']).exists())
 def test_purged_checkpoint_requires_explicit_reprocessing(self):
  paths=self.root/'paths.json.gz';p.save(paths,['old/cleaned.f64'])
  p.save(self.root/'LOCAL_DATA_PURGE_2026-09-26.json',dict(status='completed',paths_file=paths.name,paths_sha256=p.digest(paths),durable_git_record='record'))
  with self.assertRaisesRegex(FileNotFoundError,'INTENTIONALLY PURGED'):p.explain_purged(self.root/'old/cleaned.f64')
  self.assertIsNone(p.explain_purged(self.root/'unrelated'))
  self.assertFalse((self.root/'old/cleaned.f64').exists())
 def test_schema_mismatch_retains_original(self):
  row=self.fixture();row['retained_variables'].remove('basis');result=p.separate_one(row)
  self.assertEqual(result['status'],'retained mixed-container exception');self.assertTrue((self.root/row['path']).exists());self.assertFalse((self.root/row['retained_companion']).exists())
if __name__=='__main__':unittest.main()
