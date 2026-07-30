import unittest

import numpy as np

from tools.diagnostics import science_iq_temperature_survey as survey


class ScienceIqTemperatureSurveyTests(unittest.TestCase):
    def test_scan_intervals_reconstruct_forced_chunks_and_edge_trim(self):
        telescope_time = 1000.0 + np.arange(51) * 0.1
        durations = np.asarray([0.8, 0.9, 0.9, 0.9, 0.8])
        intervals = survey._scan_intervals_from_times(
            telescope_time,
            durations,
        )
        self.assertEqual(len(intervals), 5)
        self.assertAlmostEqual(intervals[0]["start_time_unix_sec"], 1000.1)
        self.assertAlmostEqual(intervals[0]["end_time_unix_sec"], 1000.9)
        self.assertAlmostEqual(intervals[1]["start_time_unix_sec"], 1001.0)
        self.assertAlmostEqual(intervals[1]["end_time_unix_sec"], 1001.9)
        self.assertAlmostEqual(intervals[4]["start_time_unix_sec"], 1004.0)
        self.assertAlmostEqual(intervals[4]["end_time_unix_sec"], 1004.8)

    def test_nearest_housekeeping_match_honors_age_limit(self):
        query = np.asarray([99.0, 101.0, 151.0, 220.0, np.nan])
        sample = np.asarray([100.0, 160.0])
        indices, ages = survey._nearest_sample_indices(
            query,
            sample,
            max_age_sec=20.0,
        )
        np.testing.assert_array_equal(indices, [0, 0, 1, -1, -1])
        np.testing.assert_allclose(ages[:3], [1.0, 1.0, 9.0])
        self.assertTrue(np.isnan(ages[3]))
        self.assertTrue(np.isnan(ages[4]))

    def test_benjamini_hochberg_is_monotone_in_rank(self):
        adjusted = survey._benjamini_hochberg(
            [0.01, 0.04, 0.03, None]
        )
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertIsNone(adjusted[3])

    def test_affected_control_overlap_is_not_implicit(self):
        self.assertFalse(
            set(survey.DEFAULT_AFFECTED_NETWORKS)
            & set(survey.DEFAULT_CONTROL_NETWORKS)
        )

    def test_night_chronology_preserves_type_and_group_contrast(self):
        science_rows = [
            {"obsnum": 10, "network": 1, "step_detector_fraction": 0.4},
            {"obsnum": 10, "network": 2, "step_detector_fraction": 0.6},
            {"obsnum": 10, "network": 0, "step_detector_fraction": 0.1},
        ]
        pointing_rows = [
            {"obsnum": 9, "network": 1, "step_detector_fraction": 0.02},
            {"obsnum": 9, "network": 0, "step_detector_fraction": 0.01},
        ]
        hk_rows = []
        for obsnum, sample_time, temperature in (
            (9, 900.0, 0.06),
            (10, 1000.0, 0.07),
        ):
            hk_rows.append(
                {
                    "obsnum": obsnum,
                    "channel_id": "T8",
                    "sample_time_unix_sec": sample_time,
                    "value": temperature,
                }
            )
        rows = survey._night_chronology_rows(
            science_rows=science_rows,
            pointing_rows=pointing_rows,
            hk_rows=hk_rows,
            affected_networks={1, 2},
            control_networks={0},
        )
        self.assertEqual(
            [row["observation_type"] for row in rows],
            ["pointing", "science"],
        )
        self.assertAlmostEqual(
            rows[1]["affected_minus_control_step_fraction"],
            0.4,
        )
        self.assertAlmostEqual(rows[1]["T8"], 0.07)


if __name__ == "__main__":
    unittest.main()
