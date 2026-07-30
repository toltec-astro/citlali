import unittest

import numpy as np
import pandas as pd

from tools.diagnostics import science_iq_electronics_localization as analysis


class ScienceIqElectronicsLocalizationTests(unittest.TestCase):
    def test_mode_recovers_stable_uid_loading(self):
        rows = []
        loading = np.asarray([0.5, -1.0, 2.0])
        for event, amplitude in enumerate([-2.0, -1.0, 1.0, 3.0]):
            for uid, value in enumerate(loading):
                rows.append(
                    {
                        "event_cluster_id": f"e{event}",
                        "uid": uid,
                        "phase_change_mrad": 1.0e3 * amplitude * value,
                    }
                )
        mode = analysis._mode_for_observation(pd.DataFrame(rows))
        self.assertAlmostEqual(
            mode.singular_values[0] ** 2
            / np.sum(mode.singular_values**2),
            1.0,
        )
        self.assertAlmostEqual(
            analysis._cosine(mode.loading[0], loading),
            1.0,
        )

    def test_uid_model_survives_slot_reassignment(self):
        rows = []
        loading = {10: -1.0, 11: 0.5, 12: 2.0}
        for event, amplitude in enumerate([1.0, -2.0, 3.0, -1.0]):
            obsnum = 1 if event < 2 else 2
            slot_order = [10, 11, 12] if obsnum == 1 else [12, 10, 11]
            for slot, uid in enumerate(slot_order):
                rows.append(
                    {
                        "event_cluster_id": f"e{event}",
                        "obsnum": obsnum,
                        "network": 8,
                        "uid": uid,
                        "tone_slot_zero_based": slot,
                        "phase_change_mrad": (
                            1.0e3 * amplitude * loading[uid]
                        ),
                        "tone_offset_frequency_hz": uid * 1.0e6,
                        "probe_frequency_hz": 700.0e6 + uid * 1.0e6,
                    }
                )
        frame = pd.DataFrame(rows)
        train = frame[frame["obsnum"] == 1]
        test = frame[frame["obsnum"] == 2]
        uid_rows, _ = analysis._exact_mode_predictions(
            train,
            test,
            coordinate="uid",
            rank=1,
            model_name="empirical_uid_rank1",
        )
        slot_rows, _ = analysis._exact_mode_predictions(
            train,
            test,
            coordinate="tone_slot_zero_based",
            rank=1,
            model_name="tone_slot_rank1",
        )
        self.assertGreater(
            np.median([row["zero_baseline_r2"] for row in uid_rows]),
            0.99,
        )
        self.assertLess(
            np.median([row["zero_baseline_r2"] for row in slot_rows]),
            0.5,
        )
        paired = analysis._paired_uid_slot_predictions(
            train,
            test,
            rank=1,
        )
        paired_frame = pd.DataFrame(paired)
        counts = paired_frame.groupby("model")["test_tone_count"].sum()
        self.assertEqual(int(counts.iloc[0]), int(counts.iloc[1]))
        scores = paired_frame.groupby("model")["zero_baseline_r2"].median()
        self.assertGreater(
            scores["empirical_uid_rank1_shared_tones"],
            0.99,
        )
        self.assertLess(scores["tone_slot_rank1_shared_tones"], 0.5)

    def test_event_separation_is_deterministic(self):
        rows = pd.DataFrame(
            {
                "event_absolute_sec": [1.0, 1.2, 1.7, 2.4],
                "value": [1, 2, 3, 4],
            }
        )
        selected = analysis._select_separated_events(
            rows,
            minimum_separation_sec=0.6,
        )
        self.assertEqual(selected["value"].tolist(), [1, 3, 4])

    def test_template_projection_is_cosine_squared(self):
        y = np.asarray([1.0, -2.0, 3.0])
        design = y[:, None]
        _, _, r2, rho = analysis._fit_loading(y, design)
        self.assertAlmostEqual(r2, 1.0)
        self.assertAlmostEqual(rho, 1.0)

    def test_pointing_population_comparison_keeps_nulls(self):
        rows = []
        for vector, score in enumerate([0.8, 0.9]):
            rows.append(
                {
                    "network": 8,
                    "population": "pointing_event",
                    "obsnum": 10,
                    "vector_id": f"event-{vector}",
                    "template_zero_baseline_r2": score,
                }
            )
        for vector, score in enumerate([0.0, 0.1]):
            rows.append(
                {
                    "network": 8,
                    "population": "clean_pointing_fixed_epoch",
                    "obsnum": 11,
                    "vector_id": f"null-{vector}",
                    "template_zero_baseline_r2": score,
                }
            )
        comparison = analysis._pointing_population_comparison_rows(
            rows,
            networks=[8],
        )[0]
        self.assertEqual(comparison["event_vector_count"], 2)
        self.assertEqual(comparison["null_vector_count"], 2)
        self.assertAlmostEqual(comparison["event_null_pairwise_auc"], 1.0)

    def test_decode_null_terminated_header(self):
        value = np.frombuffer(b"roach8\0junk", dtype="S1")
        self.assertEqual(analysis._decode_chars(value), "roach8")


if __name__ == "__main__":
    unittest.main()
