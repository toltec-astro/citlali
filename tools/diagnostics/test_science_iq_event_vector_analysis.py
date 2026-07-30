import unittest

import numpy as np

from tools.diagnostics import science_iq_event_vector_analysis as analysis


class ScienceIqEventVectorAnalysisTests(unittest.TestCase):
    def test_local_complex_derivative_recovers_linear_sweep(self):
        offsets = np.arange(-4, 5, dtype=float)[:, None]
        probes = np.asarray([100.0, 200.0])
        frequency = probes[None, :] + offsets
        intercept = np.asarray([2.0 + 3.0j, -1.0 + 0.5j])
        slope = np.asarray([0.2 - 0.1j, -0.4 + 0.3j])
        complex_iq = intercept[None, :] + offsets * slope[None, :]
        z0, derivative, valid = analysis._local_complex_derivative(
            frequency,
            complex_iq,
            probes,
            half_window_steps=3,
        )
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(z0, intercept)
        np.testing.assert_allclose(derivative, slope)

    def test_mode_fit_identifies_frequency_like_change(self):
        direction = np.asarray(
            [1.0 + 0.2j, -0.4 + 0.8j, 0.3 - 0.7j, -0.9 - 0.1j]
        ) * 1.0e-5
        fractional_change = 2300.0 * direction
        fit = analysis._fit_event_modes(
            fractional_change,
            direction,
            np.ones(direction.size, dtype=bool),
        )
        self.assertEqual(fit["best_single_mode"], "frequency")
        self.assertAlmostEqual(fit["frequency_r2"], 1.0)
        self.assertAlmostEqual(
            fit["frequency_shift_hz_frequency_only"],
            2300.0,
        )

    def test_mode_fit_identifies_common_phase_rotation(self):
        direction = np.asarray(
            [1.0 + 0.2j, -0.4 + 0.8j, 0.3 - 0.7j, -0.9 - 0.1j]
        ) * 1.0e-5
        fractional_change = np.full(direction.shape, 0.0 + 0.012j)
        fit = analysis._fit_event_modes(
            fractional_change,
            direction,
            np.ones(direction.size, dtype=bool),
        )
        self.assertEqual(fit["best_single_mode"], "phase")
        self.assertAlmostEqual(fit["phase_r2"], 1.0)
        self.assertAlmostEqual(fit["phase_rad_phase_only"], 0.012)

    def test_candidate_clustering_requires_affected_networks(self):
        rows = [
            {
                "network": network,
                "raw_event_time_unix_sec": time,
                "raw_coherent_same_sign_fraction": fraction,
            }
            for network, time, fraction in (
                (1, 10.00, 0.4),
                (2, 10.08, 0.5),
                (8, 10.12, 0.3),
                (0, 10.15, 0.2),
                (3, 11.00, 0.5),
                (5, 11.02, 0.5),
            )
        ]
        clusters = analysis._cluster_candidates(
            rows,
            affected_networks={1, 2, 3, 4, 8, 9},
            minimum_fraction=0.1,
            tolerance_sec=0.35,
            minimum_affected_networks=3,
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            {row["network"] for row in clusters[0]},
            {0, 1, 2, 8},
        )

    def test_rayleigh_summary_uniform_quadrants(self):
        summary = analysis._rayleigh_summary(
            [0.0, 0.25, 0.5, 0.75],
            period=1.0,
        )
        self.assertAlmostEqual(summary["resultant_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
