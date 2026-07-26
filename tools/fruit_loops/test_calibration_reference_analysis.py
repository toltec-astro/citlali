from __future__ import annotations

import unittest

import numpy as np

from tools.fruit_loops import analyze_calibration_reference as analysis
from tools.fruit_loops import compare_feedback_ablation as ablation


class CalibrationReferenceAnalysisTest(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict]:
        rows = []
        for array in analysis.ARRAYS:
            for iteration, recovery, map_change in (
                (9, 0.90, float("nan")),
                (10, 0.945, 0.04),
                (11, 0.95445, 0.009),
            ):
                rows.append(
                    {
                        "array": array,
                        "iteration": iteration,
                        "kernel_normalized_amplitude_recovery_fraction":
                            recovery,
                        "major_fwhm_over_kernel": 1.0,
                        "minor_fwhm_over_kernel": 1.0,
                        "successive_transfer_delta_relative_rms":
                            map_change,
                        "injected_fit_s2n": 20.0,
                        "centroid_error_arcsec": 0.02,
                    }
                )
        return rows

    def test_thresholds_require_two_successive_transitions(self) -> None:
        assessed = analysis.threshold_assessment(self.rows())
        a1100 = {
            row["tolerance_percent"]: row
            for row in assessed
            if row["array"] == "a1100"
        }

        self.assertFalse(a1100[1]["all_candidate_diagnostics_pass"])
        self.assertTrue(a1100[5]["all_candidate_diagnostics_pass"])
        self.assertEqual(a1100[1]["window_transitions"], "10;11")

    def test_trajectory_summary_does_not_assume_exponential_form(self) -> None:
        summary = analysis.trajectory_summary(self.rows())

        self.assertEqual(len(summary), 3)
        self.assertTrue(summary[0]["monotonic_non_decreasing"])
        self.assertAlmostEqual(
            summary[0]["last_three_iteration_span"], 0.05445
        )

    def test_robust_background_sigma_uses_mad(self) -> None:
        values = np.tile(np.asarray([-1.0, 1.0]), (10, 5))

        sigma = ablation.robust_background_sigma(
            values, 4.5, 4.5, 1.0
        )

        self.assertAlmostEqual(sigma, 1.4826)


if __name__ == "__main__":
    unittest.main()
