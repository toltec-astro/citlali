"""Saved evidence invariants: no new native observation or noise model."""
from pathlib import Path
import json,gzip,unittest,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;C=D/'compact';R=C/'report'
def read(p):return json.loads(p.read_text())
class Saved(unittest.TestCase):
 def test_complete_population_and_unavailability(self):
  pop=read(R/'population.json');nets=read(R/'population-summary.json')['networks'];self.assertEqual(len(pop),2069);self.assertEqual(len({(q['network'],q['detector'],q['physical_run']) for q in pop}),2069)
  for n in nets:
   rr=[q for q in pop if q['network']==n['network']];self.assertEqual(len(rr),n['detectors']);self.assertEqual(sum(q['stages']['M0']['retained']>0 for q in rr),n['retained_detectors']);self.assertEqual(sum(q['MAP_contributions'] for q in rr),n['MAP_contributions'])
  self.assertEqual(sum(n['MAP_contributions'] for n in nets),9805999)
  for q in pop:
   self.assertEqual(set(q['stages']),{'CAL','M0'});self.assertIn('unavailable',q['T5'])
   if not q['stages']['M0']['retained']:self.assertNotIn('maximum',q['stages']['M0'])
 def test_only_partial_regression_claim(self):
  r=read(C/'REGRESSION.json');self.assertTrue(r['completed_M0_exact']);self.assertTrue(r['candidate_blocked']);self.assertIn('unavailable',r['full_M0_T5_equivalence']);self.assertEqual(r['endpoint'],'not executed');r=read(C/'NATIVE_REGRESSION.json');self.assertTrue(r['pass_']);self.assertEqual(r['science_hashes'],25142);self.assertTrue(all(q['exact'] for q in r['checks']))
 def test_truthful_failed_execution(self):
  r=read(C/'EXECUTION_ACCOUNTING.json');self.assertEqual((r['original_parents'],r['charged_trajectories'],r['feedback_attempts']),(1,1,1));self.assertEqual((r['feedback_completed'],r['endpoints'],r['Q_trajectories'],r['replacements']),(0,0,0,0));self.assertEqual(r['initial_network_Apply_completions'],4)
  log=(C/'B-REF-rep1/run/BASE-POINT/REF/a1100/cycle-01/native/run.log').read_text();self.assertIn('FRUIT foreign array/observation',log)
 def test_bounded_windows_preserve_missing_support(self):
  rows=read(R/'new-windows.json');self.assertEqual(len(rows),12);self.assertEqual({q['network'] for q in rows},{1,2,3,4});z=np.load(R/'new-windows.npz')
  for q in rows:
   self.assertIn('unavailable',q['T5']);p=q['key'];self.assertEqual(len(z[p+'_row']),q['past_last']-q['first']);self.assertTrue(np.all(np.diff(z[p+'_row'])==1));self.assertEqual(int((z[p+'_map_pixel']>=0).sum()),q['MAP_contributions'])
  self.assertFalse(any(k.endswith('_T5') for k in z.files));self.assertEqual(sum(bool(q['learn_trace']) for q in rows),1)
 def test_actual_map_and_old_window_identity(self):
  p=read(R/'projection-verification.json');self.assertTrue(p['old_T5_pixel_and_causes_authenticated']);self.assertTrue(p['old_T5_unit_normalization_equals_hits']);self.assertLess(p['M0_signal_max_absolute_difference'],1e-9)
  q=read(R/'old-window-projections.json');self.assertEqual(len(q['records']),32);w=next(w for w in q['records'] if w['key']=='n1_w5');self.assertEqual((w['actual_MAP_contributions'],w['science_supported_MAP_contributions'],w['source_disk_MAP_contributions']),(367,0,0));self.assertFalse(w['peak_pixel_science_supported']);self.assertGreater(w['max_abs_projection_any_mapped_mJy_beam'],0)
 def test_cleanup_proposal_does_not_remove_originals(self):
  p=json.loads(gzip.decompress((D/'NATIVE_REMOVALS.json.gz').read_bytes()));self.assertFalse(p['applied']);root=Path(p['root']);self.assertEqual(len(p['files']),25142);self.assertEqual(sum(q['bytes'] for q in p['files']),p['bytes']);protected={q['path'] for q in read(D/'PROTECTED_INPUTS.json')}
  for q in p['files']:self.assertTrue(Path(q['path']).is_relative_to(root));self.assertNotIn(q['path'],protected)
 def test_descriptive_stats_do_not_bridge_unavailable_pairs(self):
  spec=importlib.util.spec_from_file_location('saved_population',D/'retained_population.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  z=np.r_[np.zeros(40),np.nan*np.ones(40),np.ones(40)*1000];q=m.summarize(z,np.ones(len(z),bool),0,len(z));self.assertEqual(q['adjacent_difference_MAD'],0);self.assertNotIn('max_half_second_contrast',q)
if __name__=='__main__':unittest.main()
