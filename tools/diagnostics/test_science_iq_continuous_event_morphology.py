import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from tools.diagnostics import science_iq_continuous_event_morphology as analysis


class ScienceIqContinuousEventMorphologyTests(unittest.TestCase):
    def test_telegraph_example_selects_densest_window(self):
        time = np.arange(0.0, 100.0, 0.1)
        projection = analysis.Projection(
            obsnum=1,
            network=8,
            raw_path=Path("raw.nc"),
            apt_path=Path("apt.ecsv"),
            template_source="test",
            template_training_obsnums=(1,),
            template_tone_count=10,
            sample_frequency_hz=10.0,
            time_unix_sec=time,
            projected_phase_rad=np.sin(time),
            step_change_rad=np.zeros_like(time),
            step_score=np.zeros_like(time),
            step_center_rad=0.0,
            step_sigma_rad=1.0,
        )
        events = [
            {
                "event_id": f"e{index}",
                "primary_event_candidate": True,
                "networks": "1 8 9",
                "refined_event_time_unix_sec": value,
                "dominant_projected_step_sign": (
                    "positive" if index % 2 else "negative"
                ),
                "quality_tier": "B_cross_rack_3to4",
                "network_count": 3,
            }
            for index, value in enumerate([5.0, 50.0, 51.0, 52.0])
        ]
        example = analysis._telegraph_example_candidate(
            projection=projection,
            event_rows=events,
            window_sec=10.0,
        )
        self.assertIsNotNone(example)
        self.assertEqual(example["event_count"], 3)
        self.assertEqual(len(example["marker_rows"]), 3)

    def test_sign_sequence_metrics_detect_alternation_excess(self):
        rows = [
            {
                "event_time_unix_sec": float(index),
                "dominant_projected_step_sign": sign,
            }
            for index, sign in enumerate(
                ["positive", "negative", "positive", "negative"]
            )
        ]
        metrics = analysis._sign_sequence_metrics(rows)
        self.assertEqual(
            metrics["primary_adjacent_sign_alternation_fraction"],
            1.0,
        )
        self.assertEqual(
            metrics["primary_iid_sign_alternation_expectation"],
            0.5,
        )
        self.assertEqual(
            metrics["primary_maximum_same_sign_run_length"],
            1,
        )

    def test_missing_exact_apt_is_inventoried_and_excluded(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "apt_1_matched.ecsv").touch()
            raw = {
                1: {1: Path("one.nc")},
                2: {1: Path("two.nc")},
            }
            analyzed, inventory = analysis._partition_observations_by_apt(
                raw,
                apt_root=root,
            )
        self.assertEqual(sorted(analyzed), [1])
        by_obsnum = {row["obsnum"]: row for row in inventory}
        self.assertEqual(by_obsnum[1]["analysis_status"], "analyzed")
        self.assertEqual(
            by_obsnum[2]["analysis_status"],
            "excluded_missing_exact_matched_apt",
        )

    def test_symmetric_step_filter_recovers_step(self):
        values = np.zeros(1000)
        values[500:] = 2.0
        filtered = analysis._symmetric_step_filter(
            values,
            sample_frequency_hz=100.0,
            window_sec=0.20,
            guard_sec=0.05,
        )
        peak = int(np.nanargmax(filtered))
        self.assertLessEqual(abs(peak - 500), 5)
        self.assertAlmostEqual(float(filtered[peak]), 2.0)

    def test_cross_rack_cluster_tiers(self):
        rows = []
        ordinal = 0
        for time, networks in (
            (10.0, [1, 2, 8, 9]),
            (20.0, [1, 2, 3, 4, 8]),
        ):
            for offset, network in enumerate(networks):
                ordinal += 1
                rows.append(
                    {
                        "candidate_id": f"c{ordinal}",
                        "obsnum": 1,
                        "network": network,
                        "candidate_time_unix_sec": time + 0.01 * offset,
                        "candidate_time_since_start_sec": time,
                        "signed_step_change_rad_per_rms_loading": 0.1,
                        "signed_step_score": 10.0 + network,
                        "absolute_step_score": 10.0 + network,
                    }
                )
        events, members = analysis._cluster_candidate_rows(
            rows,
            obsnum=1,
            coincidence_sec=0.25,
            consume_sec=0.50,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event["quality_tier"] for event in events],
            ["B_cross_rack_3to4", "A_cross_rack_5plus"],
        )
        self.assertTrue(all(event["primary_event_candidate"] for event in events))
        self.assertTrue(
            all(
                event["network_member_sign_is_unanimous"]
                for event in events
            )
        )
        self.assertEqual(len(members), 9)

    def test_leave_one_observation_out_template(self):
        rows = []
        loading_values = np.linspace(-2.0, 2.0, 10)
        for obsnum in (1, 2, 3):
            for event in range(3):
                for uid, loading in enumerate(loading_values):
                    rows.append(
                        {
                            "obsnum": obsnum,
                            "network": 8,
                            "event_cluster_id": f"{obsnum}-{event}",
                            "uid": uid,
                            "phase_change_mrad": (
                                1.0e3 * (event + 1) * loading
                            ),
                        }
                    )
        template = analysis._load_template(
            obsnum=2,
            network=8,
            event_tones=pd.DataFrame(rows),
            fixed_templates=pd.DataFrame(),
            event_rich_obsnums=[1, 2, 3],
        )
        self.assertEqual(template.training_obsnums, (1, 3))
        self.assertIn("leave_one", template.source)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(template.loading**2))),
            1.0,
        )

    def test_exponential_recovery_fit(self):
        time = np.linspace(0.2, 6.0, 300)
        value = 0.25 + 0.75 * np.exp(-time / 1.6)
        fit = analysis._fit_recovery(time, value)
        self.assertEqual(fit["recovery_fit_status"], "fit")
        self.assertAlmostEqual(fit["recovery_tau_sec"], 1.6, places=2)
        self.assertAlmostEqual(
            fit["recovery_asymptote_fraction"],
            0.25,
            places=2,
        )
        self.assertGreater(fit["recovery_fit_r2"], 0.999)

    def test_onset_refinement_uses_projected_derivative(self):
        time = np.arange(0.0, 10.0, 0.01)
        onset = 5.12
        projected = 0.5 * (1.0 + np.tanh((time - onset) / 0.02))
        projection = analysis.Projection(
            obsnum=1,
            network=8,
            raw_path=Path("raw.nc"),
            apt_path=Path("apt.ecsv"),
            template_source="test",
            template_training_obsnums=(1,),
            template_tone_count=10,
            sample_frequency_hz=100.0,
            time_unix_sec=time,
            projected_phase_rad=projected,
            step_change_rad=np.zeros_like(time),
            step_score=np.zeros_like(time),
            step_center_rad=0.0,
            step_sigma_rad=1.0,
        )
        refined = analysis._refine_onset_time(projection, 5.20)
        self.assertLess(abs(refined - onset), 0.03)

    def test_waveform_summary_preserves_event_count(self):
        grid = np.asarray([-1.0, 0.0, 1.0])
        records = [
            {
                "event_id": "a",
                "obsnum": 1,
                "network": 8,
                "quality_tier": "A_cross_rack_5plus",
                "step_snr": 10.0,
                "available_post_sec": 6.0,
                "time_sec": grid,
                "normalized": np.asarray([0.0, 1.0, 0.5]),
            },
            {
                "event_id": "b",
                "obsnum": 1,
                "network": 8,
                "quality_tier": "A_cross_rack_5plus",
                "step_snr": 12.0,
                "available_post_sec": 6.0,
                "time_sec": grid,
                "normalized": np.asarray([0.0, 1.0, 0.3]),
            },
        ]
        stack, examples = analysis._summarize_waveforms(
            records,
            networks=[8],
        )
        at_one = next(
            row
            for row in stack
            if row["time_from_network_onset_sec"] == 1.0
        )
        self.assertEqual(at_one["contributing_event_count"], 2)
        self.assertAlmostEqual(
            at_one["median_normalized_projected_phase"],
            0.4,
        )
        self.assertEqual(
            {row["event_id"] for row in examples},
            {"a", "b"},
        )


if __name__ == "__main__":
    unittest.main()
