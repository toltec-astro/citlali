import unittest
from pathlib import Path

import numpy as np

from tools.diagnostics import science_iq_hidden_state_analysis as analysis
from tools.diagnostics.science_iq_continuous_event_morphology import Projection


class ScienceIqHiddenStateAnalysisTests(unittest.TestCase):
    def test_forward_backward_probabilities_are_normalized(self):
        values = np.asarray([[-1.0], [-0.9], [1.0], [1.1]])
        means = np.asarray([[-1.0], [1.0]])
        variances = np.asarray([[0.1], [0.1]])
        emission = analysis._log_emission(values, means, variances)
        posterior, transition_sum, likelihood = analysis._forward_backward(
            emission,
            np.asarray([0.5, 0.5]),
            np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        )
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
        self.assertEqual(transition_sum.shape, (2, 2))
        self.assertTrue(np.isfinite(likelihood))

    def test_two_state_hmm_recovers_synthetic_levels(self):
        rng = np.random.default_rng(10)
        states = np.repeat([0, 1, 0, 1], 30)
        values = np.where(states == 0, -2.0, 2.0)
        values = values + rng.normal(0.0, 0.20, len(values))
        fits = [
            analysis._fit_gaussian_hmm(
                values,
                n_states=count,
                n_initializations=4,
                random_seed=100,
                maximum_iterations=100,
                convergence_tolerance=1.0e-8,
                minimum_occupancy_fraction=0.03,
                minimum_center_separation_sigma=1.0,
            )
            for count in (1, 2, 3)
        ]
        selected = analysis._select_hmm_model(
            fits,
            bic_parsimony_tolerance=6.0,
        )
        self.assertEqual(selected.n_states, 2)
        np.testing.assert_allclose(selected.means[:, 0], [-2.0, 2.0], atol=0.1)
        self.assertGreater(
            adjusted_accuracy(selected.decoded, states),
            0.98,
        )

    def test_interval_measurement_preserves_uid_projection_units(self):
        time = np.arange(0.0, 10.0, 0.01)
        projected = np.where(time < 5.0, -1.0, 2.0)
        projection = Projection(
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
        events = [
            {
                "event_id": "e1",
                "primary_event_candidate": True,
                "refined_event_time_unix_sec": 5.0,
            }
        ]
        intervals, measurements = analysis._interval_measurements(
            obsnum=1,
            projections={8: projection},
            event_rows=events,
            transition_guard_sec=0.35,
            minimum_samples=8,
        )
        self.assertEqual(len(intervals), 2)
        self.assertEqual(len(measurements), 2)
        self.assertAlmostEqual(
            measurements[0][
                "projected_phase_level_rad_per_rms_loading"
            ],
            -1.0,
        )
        self.assertAlmostEqual(
            measurements[1][
                "projected_phase_level_rad_per_rms_loading"
            ],
            2.0,
        )

    def test_dwell_runs_merge_unchanged_catalog_intervals(self):
        intervals = [
            {
                "interval_start_unix_sec": float(index),
                "interval_end_unix_sec": float(index + 1),
            }
            for index in range(5)
        ]
        decoded = np.asarray([0, 0, 1, 1, 0])
        posterior = np.full((5, 2), 0.1)
        posterior[np.arange(5), decoded] = 0.9
        runs = analysis._dwell_run_rows(
            obsnum=1,
            model_scope="network",
            network=8,
            interval_rows=intervals,
            decoded=decoded,
            posterior=posterior,
        )
        self.assertEqual([row["dwell_duration_sec"] for row in runs], [2, 2, 1])
        self.assertTrue(runs[0]["left_censored_by_observation_start"])
        self.assertTrue(runs[-1]["right_censored_by_observation_end"])

    def test_coordinate_scale_tolerates_sparse_constant_network(self):
        values = np.asarray(
            [
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        )
        scale = analysis._robust_coordinate_scales(values)
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertGreater(scale[0], 0.0)
        self.assertEqual(scale[1], 1.0)


def adjusted_accuracy(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    direct = np.mean(first == second)
    inverted = np.mean((1 - first) == second)
    return float(max(direct, inverted))


if __name__ == "__main__":
    unittest.main()
