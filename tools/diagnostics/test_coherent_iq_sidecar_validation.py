#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.diagnostics.coherent_iq_sidecar_validation import (
    _boolean_series,
    closest_unique_matches,
    compare_scores,
    maximum_match_count,
)


class CoherentIqSidecarValidationTest(unittest.TestCase):
    def test_boolean_csv_values_are_parsed_by_value(self) -> None:
        parsed = _boolean_series(pd.Series(["True", "False", "1", "0"]))
        self.assertEqual(parsed.tolist(), [True, False, True, False])
        with self.assertRaises(ValueError):
            _boolean_series(pd.Series(["maybe"]))

    def test_matching_is_one_to_one_and_closest_first(self) -> None:
        reference = pd.DataFrame({"time": [10.0, 10.2, 12.0]})
        candidates = pd.DataFrame(
            {
                "candidate_id": ["c0", "c1", "c2"],
                "event_time_unix_sec": [10.04, 10.18, 13.0],
            }
        )
        matches = closest_unique_matches(
            reference,
            candidates,
            reference_time_column="time",
            tolerance_sec=0.1,
        )
        self.assertEqual(matches["candidate_id"].tolist(), ["c1", "c0"])
        self.assertEqual(
            set(matches["reference_row_zero_based"].tolist()), {0, 1}
        )
        self.assertEqual(
            maximum_match_count(
                reference["time"].to_numpy(),
                candidates["event_time_unix_sec"].to_numpy(),
                0.1,
            ),
            2,
        )

    def test_runtime_and_offline_scores_remain_distinct(self) -> None:
        scores = pd.DataFrame(
            {
                "candidate_id": ["candidate-1", "candidate-1"],
                "network": [1, 2],
                "status": ["scored", "scored"],
                "projection_amplitude_mrad": [8.0, -3.0],
                "absolute_cosine_similarity": [0.8, 0.4],
                "explained_energy_fraction": [0.64, 0.16],
                "descriptive_mode_selected": [True, False],
            }
        )
        known_events = pd.DataFrame(
            {
                "event_cluster_id": ["event-1"],
                "networks": ["1 2"],
            }
        )
        matches = pd.DataFrame(
            {
                "reference_row_zero_based": [0],
                "candidate_id": ["candidate-1"],
                "candidate_minus_reference_time_sec": [-0.2],
            }
        )
        offline = pd.DataFrame(
            {
                "event_cluster_id": ["event-1", "event-1"],
                "network": [1, 2],
                "status": ["scored", "scored"],
                "projection_amplitude_mrad": [10.0, -7.0],
                "absolute_cosine_similarity": [0.9, 0.7],
                "explained_energy_fraction": [0.81, 0.49],
            }
        )
        comparison, by_network, summary = compare_scores(
            scores,
            known_events,
            matches,
            offline,
            minimum_absolute_cosine=0.6,
            minimum_absolute_amplitude_mrad=5.0,
        )
        self.assertEqual(len(comparison), 2)
        self.assertEqual(len(by_network), 2)
        self.assertEqual(summary["runtime_selected_count"], 1)
        self.assertEqual(summary["offline_selected_count"], 2)
        self.assertEqual(summary["both_selected_count"], 1)
        self.assertAlmostEqual(summary["runtime_recall_of_offline_selected"], 0.5)
        self.assertTrue(
            np.allclose(
                comparison["projection_amplitude_mrad_runtime"], [8.0, -3.0]
            )
        )


if __name__ == "__main__":
    unittest.main()
