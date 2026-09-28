"""Saved two-stage invariants; no native processing, fitting or noise ensembles."""
from pathlib import Path
import json,gzip,unittest,importlib.util,hashlib
import numpy as np
D=Path(__file__).resolve().parent;C=D/'compact';R=C/'report'
def read(p):return json.loads(p.read_text())
class Saved(unittest.TestCase):
 def test_full_B_gate(self):
  r=read(C/'REGRESSION.json');self.assertTrue(r['pass_']);self.assertTrue(r['endpoint_exact']);self.assertEqual(len(r['planes']),60);self.assertEqual(len(r['decisions']),37);self.assertEqual(r['maximum_absolute_plane_difference'],0)
  r=read(C/'NATIVE_REGRESSION.json');self.assertTrue(r['pass_']);self.assertEqual(r['science_hashes'],25584);self.assertTrue(all(q['exact'] for q in r['checks']))
 def test_complete_population_preserves_unavailability(self):
  for relative,retained in [('',1386),('Q/',1385)]:
   pop=read(R/(relative+'population.json'));self.assertEqual(len(pop),2069);self.assertEqual(len({(q['network'],q['detector'],q['physical_run']) for q in pop}),2069)
   self.assertEqual(sum(q['stages']['T5']['retained']>0 for q in pop),retained)
   for q in pop:
    self.assertEqual(set(q['stages']),{'CAL','M0','T5'})
    if not q['stages']['T5']['retained']:self.assertNotIn('maximum',q['stages']['T5'])
   check=read(R/(relative+'projection-check.json'));self.assertTrue(check['pass_']);self.assertLess(check['maximum_absolute_difference'],1e-9)
 def test_single_frozen_policy_and_realized_loss(self):
  p=D/'QUARANTINE.json';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'a8edbfecdc297e06fd8625fbd1b435ff1b43bae60de852043cac0fcda995f1c5');m=read(p)['members'];self.assertEqual([(q['network'],q['detector'],q['native_interval']) for q in m],[(2,88,[0,7821])])
  rows=read(R/'B-Q-comparison.json')['native_support']
  for q in rows:
   self.assertEqual(q['RTC_paired_gained'],0);self.assertEqual(q['MAP_centers_gained'],0)
   self.assertTrue(all(q['network']==2 and c['channel']==88 and c['gained']==0 for c in q['changed']))
   self.assertTrue(q['events_exact']);self.assertTrue(q['decisions_exact'])
 def test_actual_counts_and_no_retry(self):
  r=read(C/'EXECUTION_ACCOUNTING.json');self.assertEqual((r['original_parents'],r['charged_trajectories'],r['feedback_attempts'],r['feedback_completed'],r['endpoints']),(5,5,25,25,5));self.assertEqual(r['Q_trajectories'],4);self.assertEqual(r['replacements'],0);self.assertEqual(r['initial_network_Apply_completions'],20);self.assertEqual(r['unexpected_error_messages'],[])
 def test_fixed_windows_and_source_context(self):
  rows=read(R/'new-windows.json');self.assertEqual(len(rows),16);z=np.load(R/'new-windows.npz')
  for q in rows:self.assertEqual(len(z[q['key']+'_T5']),q['past_last']-q['first'])
  self.assertTrue(np.isfinite(z['n2_new1_raw_x']).all());self.assertTrue(np.isfinite(z['n2_new1_raw_r']).all());self.assertGreater(float(z['n2_new1_distance_arcsec'].min()),60)
  old=read(R/'old-window-projections.json')['records'];self.assertEqual(len(old),32)
  for q in old:
   if not q['actual_MAP_contributions']:self.assertIsNone(q['peak_native_row']);self.assertIsNone(q['T5_residual_max'])
 def test_B_domain_loss_is_explicit(self):
  a=np.load(R/'B-Q-maps.npz');b=a['B_support'];q=a['Q_support'];r=read(R/'B-Q-comparison.json')['cycles'][-1]
  self.assertEqual(r['lost_B_science_pixels'],int((b&~q).sum()));self.assertEqual(r['gained_science_pixels'],int((q&~b).sum()));self.assertTrue(np.isnan(a['difference'][~(b&q)]).all())
  self.assertEqual([r['name'] for r in r['measurements']],['peak','cx','cy','FWHM_major','FWHM_minor','major_angle','constant','template','aperture1','aperture1.5','aperture2'])
 def test_response_controls_and_original_validity(self):
  r=read(R/'response-Q/source-responses.json');self.assertEqual(len(r),6);self.assertEqual({q['cycle'] for q in r},{0,5});self.assertEqual({q['branch'] for q in r},{'C-core-plus','C-core-minus','C-shoulder'})
  for q in read(R/'response-Q/native-source-accounting.json'):self.assertTrue(q['input_r_unchanged']);self.assertTrue(q['input_validity_unchanged']);self.assertTrue(q['own_RTC_learning'])
 def test_no_scratch_cleanup(self):
  p=json.loads(gzip.decompress((D/'NATIVE_SCRATCH_PROPOSAL.json.gz').read_bytes()));self.assertFalse(p['applied']);self.assertFalse(p['owner_approval']);self.assertEqual(sum(q['bytes'] for q in p['files']),p['bytes']);protected={q['path'] for q in read(D/'PROTECTED_INPUTS.json')}
  for q in p['files']:self.assertTrue(Path(q['path']).is_relative_to(p['root']));self.assertNotIn(q['path'],protected)
 def test_matched_footprint_and_missing_extreme(self):
  z=np.load(R/'B-Q-maps.npz');a=read(R/'B-Q-comparison.json')['final_residuals'];common=z['B_event_footprint']&z['B_support']&z['Q_support'];lost=z['B_event_footprint']&z['B_support']&~z['Q_support']
  self.assertEqual(int(common.sum()),58);self.assertEqual(int(lost.sum()),1)
  for name,key in [('B_on_common_event_footprint','B_residual'),('Q_event_footprint','Q_residual')]:self.assertAlmostEqual(a[name]['std'],float(np.std(z[key][common],ddof=1)),places=10)
  self.assertAlmostEqual(a['lost_event_pixels'][0]['B_residual'],float(z['B_residual'][lost][0]),places=10)
 def test_compact_integrity(self):
  m=read(D/'COMPACT_MANIFEST.json');self.assertEqual(len(m['files']),2336)
  for row in m['files']:
   b=(C/row['path']).read_bytes();self.assertEqual(len(b),row['bytes']);self.assertEqual(hashlib.sha256(b).hexdigest(),row['sha256']);original=gzip.decompress(b) if row['encoding']=='gzip' else b;self.assertEqual(hashlib.sha256(original).hexdigest(),row['source_sha256'])
 def test_statistics_do_not_bridge_gaps(self):
  spec=importlib.util.spec_from_file_location('two_stage_population',D/'two_stage_population.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);z=np.r_[np.zeros(40),np.full(40,np.nan),np.ones(40)*1000];q=m.summarize(z,np.ones(len(z),bool),0,len(z));self.assertEqual(q['adjacent_difference_MAD'],0);self.assertNotIn('max_half_second_contrast',q)
if __name__=='__main__':unittest.main()
