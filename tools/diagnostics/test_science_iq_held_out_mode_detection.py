import unittest
from pathlib import Path

import numpy as np

from tools.diagnostics import science_iq_held_out_mode_detection as analysis
from tools.diagnostics.science_iq_continuous_event_morphology import Projection
from tools.diagnostics.science_iq_hidden_state_analysis import HmmFit


def synthetic_projection(
    *,
    network: int,
    values: np.ndarray,
    sample_frequency_hz: float = 20.0,
) -> Projection:
    time = np.arange(len(values), dtype=float) / sample_frequency_hz
    return Projection(
        obsnum=1,
        network=network,
        raw_path=Path(f"nw{network}.nc"),
        apt_path=Path("apt.ecsv"),
        template_source="synthetic",
        template_training_obsnums=(1,),
        template_tone_count=10,
        sample_frequency_hz=sample_frequency_hz,
        time_unix_sec=time,
        projected_phase_rad=np.asarray(values, dtype=float),
        step_change_rad=np.zeros(len(values), dtype=float),
        step_score=np.zeros(len(values), dtype=float),
        step_center_rad=0.0,
        step_sigma_rad=1.0,
    )


class ScienceIqHeldOutModeDetectionTests(unittest.TestCase):
    def test_fixed_bins_use_common_time_and_network_identity(self):
        samples = 1000
        first = synthetic_projection(
            network=1,
            values=np.where(np.arange(samples) < 500, -1.0, 2.0),
        )
        second = synthetic_projection(
            network=8,
            values=np.where(np.arange(samples) < 500, 3.0, -4.0),
        )
        time, levels, counts, rows = analysis._fixed_bin_measurements(
            {1: first, 8: second},
            networks=[1, 8],
            bin_width_sec=1.0,
            minimum_samples_per_network=10,
        )
        self.assertEqual(levels.shape, (49, 2))
        self.assertEqual(counts.shape, levels.shape)
        self.assertEqual(len(rows), 98)
        self.assertTrue(np.all(np.diff(time) > 0.0))
        np.testing.assert_allclose(levels[:25, 0], -1.0)
        np.testing.assert_allclose(levels[25:, 0], 2.0)
        self.assertEqual(
            sorted({int(row["network"]) for row in rows}),
            [1, 8],
        )

    def test_target_normalization_preserves_amplitude_against_frozen_scale(self):
        time = np.arange(400, dtype=float)
        state = np.repeat([-1.0, 1.0, -1.0, 1.0], 100)
        training = np.column_stack([state, 2.0 * state])
        target = 0.25 * training
        train_standardized, train_norm = analysis._normalize_segment(
            time,
            training,
        )
        target_standardized, target_norm = analysis._normalize_segment(
            time,
            target,
            applied_scales=(
                train_norm.applied_scale_rad_per_rms_loading
            ),
        )
        self.assertGreater(np.std(train_standardized[:, 0]), 0.5)
        self.assertLess(
            np.std(target_standardized[:, 0]),
            0.4 * np.std(train_standardized[:, 0]),
        )
        np.testing.assert_allclose(
            target_norm.applied_scale_rad_per_rms_loading,
            train_norm.applied_scale_rad_per_rms_loading,
        )
        np.testing.assert_allclose(
            target_norm.intrinsic_residual_scale_rad_per_rms_loading
            / train_norm.intrinsic_residual_scale_rad_per_rms_loading,
            0.25,
            atol=0.02,
        )

    def test_canonicalization_finds_transition_hub_and_network_groups(self):
        means = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, -1.0, -1.0, -1.0, 2.0],
                [-1.0, -1.0, 2.0, 2.0, 2.0, -1.0],
            ]
        )
        decoded = np.asarray([0, 1, 0, 2, 0, 1, 0, 2, 0])
        posterior = np.full((len(decoded), 3), 0.01)
        posterior[np.arange(len(decoded)), decoded] = 0.98
        fit = HmmFit(
            n_states=3,
            means=means,
            variances=np.ones_like(means),
            transition=np.full((3, 3), 1.0 / 3.0),
            initial=np.full(3, 1.0 / 3.0),
            posterior=posterior,
            decoded=decoded,
            log_likelihood=0.0,
            bic=0.0,
            aic=0.0,
            parameter_count=1,
            iterations=1,
            converged=True,
            minimum_center_separation_sigma=1.0,
            minimum_posterior_occupancy_fraction=0.1,
            selection_eligible=True,
            ineligibility_reason=None,
        )
        result, names, audit = analysis._canonicalize_two_mode_fit(
            fit,
            networks=[1, 2, 3, 4, 8, 9],
        )
        self.assertEqual(
            names,
            ["baseline_hub", "mode_129_relative", "mode_348_relative"],
        )
        np.testing.assert_allclose(result.means[0], means[0])
        np.testing.assert_allclose(result.means[1], means[1])
        np.testing.assert_allclose(result.means[2], means[2])
        self.assertEqual(audit["baseline_neighbor_count"], 2)

    def test_one_to_one_matching_maximizes_count_then_minimizes_residual(self):
        matches = analysis._one_to_one_time_matches(
            np.asarray([0.0, 1.0]),
            np.asarray([0.6]),
            tolerance_sec=0.7,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].prediction_index, 1)
        self.assertAlmostEqual(matches[0].residual_sec, 0.4)

        two = analysis._one_to_one_time_matches(
            np.asarray([0.0, 0.9]),
            np.asarray([0.5, 1.4]),
            tolerance_sec=0.6,
        )
        self.assertEqual(len(two), 2)

    def test_frozen_model_decodes_synthetic_state_path(self):
        values = np.asarray(
            [
                [-2.0, -2.0],
                [-1.9, -2.1],
                [2.0, -1.0],
                [2.1, -0.9],
                [-1.0, 2.0],
                [-0.9, 2.1],
            ]
        )
        means = np.asarray([[-2.0, -2.0], [2.0, -1.0], [-1.0, 2.0]])
        fit = HmmFit(
            n_states=3,
            means=means,
            variances=np.full_like(means, 0.05),
            transition=np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ]
            ),
            initial=np.full(3, 1.0 / 3.0),
            posterior=np.empty((0, 3)),
            decoded=np.empty(0, dtype=int),
            log_likelihood=0.0,
            bic=0.0,
            aic=0.0,
            parameter_count=1,
            iterations=1,
            converged=True,
            minimum_center_separation_sigma=1.0,
            minimum_posterior_occupancy_fraction=0.1,
            selection_eligible=True,
            ineligibility_reason=None,
        )
        decoded, posterior, likelihood = analysis._decode_frozen_model(
            values,
            fit,
        )
        np.testing.assert_array_equal(decoded, [0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(likelihood))


if __name__ == "__main__":
    unittest.main()
