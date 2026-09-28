"""Reporting invariants over retained evidence and small ambiguous-match fixtures."""
import unittest,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from report_step_campaign import stats
from extract_step_response import match_events
D=Path(__file__).parent/'compact'
class Reporting(unittest.TestCase):
 def test_missing_centers_not_zero_or_adjacent(self):
  v=np.array([0.,np.nan,2.,np.nan,4.,np.nan,np.nan,np.nan,100.]);r=stats(v,np.arange(9),4)
  self.assertIsNone(r['adjacent_difference_MAD']);self.assertIsNone(r['post_minus_pre'])
 def test_ambiguous_physical_event_matches_stay_unmatched(self):
  a=[dict(detector=3,seed_earlier_row=4,trial_exclusion=[2,9])];b=[dict(detector=3,seed_earlier_row=k,trial_exclusion=[3,8]) for k in (5,6)]
  pairs,left,right,amb=match_events(a,b);self.assertEqual(pairs,[]);self.assertEqual(left,[0]);self.assertEqual(right,[0,1]);self.assertEqual(len(amb),1)
 def test_exposure_union_and_valid_denominators(self):
  for name in ('N14-REF-rep2',):
   p=D/'report'/name;z=np.load(p/'population-masks.npz')
   for n in json.loads((p/'population.json').read_text()):
    nt,nd=n['shape'];get=lambda k:np.unpackbits(z[f"n{n['network']}_{k}"],axis=1)[:,:nt].astype(bool)
    vx=get('valid_x');rtc=get('RTC_x');self.assertEqual(int(vx.sum()),n['input_valid_x_cells']);self.assertEqual(int(rtc.sum()),n['RTC_retained_x_cells'])
    self.assertTrue(np.array_equal(rtc,get('planned_filtered')));self.assertFalse(np.any(rtc&get('jump_scan')))
    self.assertEqual(int((vx&~rtc).sum()),sum(q['incremental_preceding_eligible_cells'] for q in n['exposure_ledger']))
    self.assertEqual(int(get('CAL_eligible').sum()),n['CAL_eligible_scheduled_cells'])
 def test_seed_free_controls_do_not_imply_no_candidates(self):
  w={q['key']:q for q in json.loads((D/'report/N14-REF-rep2/windows.json').read_text())}
  for key,count in [('n3_w4',15),('n4_w4',13)]:
   self.assertEqual(w[key]['kind'],'unflagged');self.assertEqual(sum(len(t['raw_candidates_in_window']) for t in w[key]['learn_trace']),count)
   self.assertTrue(any(t['consider_class']=='unresolved-extent' for t in w[key]['learn_trace']))
 def test_active_changes_preserve_unavailable_support(self):
  a=json.loads((D/'report/response/active-selection-summary.json').read_text())
  for branch in ('C-core-plus','C-shoulder'):
   n=next(q for q in a if q['branch']==branch and q['network']==1)
   self.assertEqual(n['appeared'],1);self.assertEqual(n['RTC_lost_cells'],1188)
  for q in json.loads((D/'report/response/source-responses.json').read_text()):
   if q['branch']!='C-core-minus':
    self.assertEqual(q['unsupported_regions']['RTC_reference_contributions_unavailable'],1188)
    self.assertEqual(q['unsupported_regions']['actual_lost_science_pixels'],1)
 def test_ideal_retains_complete_reference_domain(self):
  for r in json.loads((D/'report/response/source-responses.json').read_text()):
   self.assertEqual(r['ideal_accounting']['reference_contributions'],r['ideal_accounting']['available_contributions']);self.assertEqual(r['ideal_accounting']['lost_contributions'],0)
   self.assertGreaterEqual(r['unsupported_regions']['actual_lost_science_pixels'],0)
   self.assertGreaterEqual(r['unsupported_regions']['actual_gained_science_pixels'],0)
   self.assertLessEqual(r['unsupported_regions']['reference_supported_comparison_fraction'],1.)
if __name__=='__main__':unittest.main()
