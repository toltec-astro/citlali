"""Interpretation guards for the bounded health report, not a new estimator."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('census', Path(__file__).parents[1]/'tools/timestream_successor/rtc_common_mode_census.py')
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def fits(rows):
    return pd.DataFrame(rows, columns=['detector', 'interval', 'first', 'past_last', 'cause', 'gain', 'offset', 'correlation', 'residual_scatter', 'reference_scatter', 'relative_gain'])


class CensusTests(unittest.TestCase):
    def test_matched_seed_health_flag_is_not_identity_ambiguity(self):
        relation = {'disposition': 'matched', 'seed_occurrence': 'exact-seed',
                    'seed_uid': '418', 'is_good_match': 'false'}
        self.assertTrue(census.accepted_relation(relation))
        self.assertFalse(census.accepted_relation(dict(relation, disposition='ambiguous')))
        self.assertFalse(census.accepted_relation(dict(relation, seed_uid='')))

    def test_identity_needs_full_qualified_artifact_key_and_uniqueness(self):
        a = pd.DataFrame({'channel': [221, 426], 'longitudinal_key': ['artifact-A|seed219', None]})
        b = pd.DataFrame({'channel': [218, 221], 'longitudinal_key': ['artifact-A|seed219', 'artifact-B|seed219']})
        self.assertEqual(census.matched_channel(a, b, 221), 218)
        self.assertIsNone(census.matched_channel(a, b, 426))
        self.assertIsNone(census.matched_channel(a, pd.concat([b, b.iloc[:1]]), 221))
        self.assertIsNone(census.matched_channel(a, b.iloc[1:], 221))

    def test_duration_weighting_does_not_let_short_segments_dominate(self):
        self.assertEqual(census.weighted_median([1, 9, 9, 9], [100, 1, 1, 1]), 1)
        self.assertEqual(census.weighted_median([1, 1, 9], [50, 50, 3]), 1)
        self.assertTrue(np.isnan(census.weighted_median([np.nan], [1])))

    def test_pair_support_and_reference_scale_cancel_in_gain_ratio(self):
        a = fits([[0, 0, 0, 100, 0, 2., 10., 1., 0., 1., 2.],
                  [0, 0, 100, 160, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        b = fits([[1, 0, 40, 140, 0, 4., -5., 1., 0., 1., 4.]])
        c = np.sin(np.arange(160)*.1)
        result, pieces = census.matched_comparison(a, b, 10+2*c, -5+4*c, c, .01)
        self.assertAlmostEqual(result['common_fit_seconds'], .6)
        self.assertEqual(pieces[['first', 'past_last']].values.tolist(), [[40, 100]])
        self.assertTrue(np.isnan(result['gain_ratio_duration_weighted_median']))
        self.assertEqual(result['identical_fit_seconds'], 0.)
        self.assertAlmostEqual(pieces.frozen_coefficient_ratio.iloc[0], .5)
        # Common metric overlap is shorter than the inherited64-sample guard.
        self.assertTrue(np.isnan(result['target_correlation_duration_weighted_median']))
        a2, b2 = a.copy(), b.copy()
        a2['gain'] /= -7
        b2['gain'] /= -7
        _, pieces2 = census.matched_comparison(a2, b2, 10+2*c, -5+4*c, -7*c, .01)
        self.assertEqual(pieces2.frozen_coefficient_ratio.iloc[0], pieces.frozen_coefficient_ratio.iloc[0])

    def test_gain_ratios_require_identical_fits_under_all_references(self):
        a = fits([[d, 0, lo, hi, 0, gain, 0., 1., 0., 1., gain]
                  for d, gain in [(0, 2.), (1, 4.)] for lo, hi in [(0, 100), (100, 200)]])
        b = a.copy()
        b.loc[(b.detector == 1) & (b['first'] == 100), 'first'] = 120
        shared = census.same_support_fits({'a': a, 'b': b}, 0, 1)
        c = np.sin(np.arange(200)*.1)
        for frame in shared.values():
            self.assertEqual(frame['first'].tolist(), [0, 0])
            result, _ = census.matched_comparison(frame[frame.detector == 0], frame[frame.detector == 1], 2*c, 4*c, c, .01)
            self.assertEqual(result['identical_fit_seconds'], 1.)
            self.assertAlmostEqual(result['gain_ratio_duration_weighted_median'], .5)

    def test_common_support_metrics_ignore_a_feature_outside_overlap(self):
        a = fits([[0, 0, 0, 200, 0, 2., 0., 1., 0., 1., 2.]])
        b = fits([[1, 0, 100, 200, 0, 1., 0., 1., 0., 1., 1.]])
        c = np.sin(np.arange(200)*.1)
        target = 2*c
        target[:100] += 1000
        result, _ = census.matched_comparison(a, b, target, c, c, .01)
        self.assertAlmostEqual(result['target_correlation_duration_weighted_median'], 1.)
        self.assertAlmostEqual(result['target_residual_mad_duration_weighted_median'], 0.)
        self.assertEqual(result['common_fit_seconds'], 1.)

    def test_negative_duration_never_bridges_invalid_or_processing_boundaries(self):
        a = fits([[0, 0, 0, 10, 0, -1., 0., -1., 1., 1., -1.],
                  [0, 0, 20, 30, 0, -1., 0., -1., 1., 1., -1.],
                  [0, 1, 30, 40, 0, -1., 0., -1., 1., 1., -1.],
                  [0, 2, 40, 50, 0, 1., 0., 1., 1., 1., 1.]])
        summary, _ = census.fit_summary(a, .1)
        self.assertEqual(summary['negative_fit_seconds'], 3.)
        self.assertEqual(summary['longest_contiguous_negative_fit_seconds'], 1.)
        self.assertEqual(summary['longest_consecutive_negative_interval_count'], 2)
        self.assertEqual(summary['negative_interval_count'], 2)

    def test_unavailable_stays_unavailable(self):
        a = fits([[0, 0, 0, 200, 2, np.nan, np.nan, np.nan, np.nan, 0., np.nan]])
        summary, rows = census.fit_summary(a, .01)
        self.assertEqual(summary['fit_seconds'], 0.)
        self.assertTrue(np.isnan(summary['gain_duration_weighted_median']))
        self.assertEqual(rows, [])
        self.assertIn('weak_reference', summary['unavailable_fit_reasons'])


if __name__ == '__main__':
    unittest.main()
