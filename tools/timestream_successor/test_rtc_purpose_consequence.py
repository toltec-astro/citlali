import unittest
import numpy as np
from rtc_purpose_consequence import crossing, PHASES
from analyze_rtc_purpose_consequence import projection_noise_sigma


class PurposeConsequence(unittest.TestCase):
    def test_shared_output_noise_uses_original_support_covariance(self):
        source = np.ones(7)
        # Two overlapping 3-tap averages, each weighted by1/2. Shared middle
        # input has weight1/3, all four side inputs1/6: variance2/9, not1/6.
        sigma = projection_noise_sigma(source, np.array([2, 4]), np.ones(3) / 3, 1)
        self.assertAlmostEqual(sigma**2, 2 / 9)
        self.assertNotAlmostEqual(sigma**2, 1 / 6)

    def test_missing_principal_crossing_is_not_tail_recovery(self):
        t = np.linspace(0, 20, 201)
        xy = np.column_stack([t, np.ones_like(t) * 20])
        _, reason = crossing(t, xy, np.array([10, 0]), 10, 5, 3)
        self.assertIn("no principal", reason)

    def test_window_includes_full_crossing_not_search_crop(self):
        t = np.linspace(0, 40, 401)
        xy = np.column_stack([t, np.zeros_like(t)])
        (lo, hi), reason = crossing(t, xy, np.array([20, 0]), 20, 12, 3)
        self.assertFalse(reason)
        self.assertLess(t[lo], 10)
        self.assertGreater(t[hi - 1], 30)

    def test_four_phases_are_fixed_quadrature_not_random_trials(self):
        np.testing.assert_allclose(np.sum(np.exp(1j * PHASES)), 0, atol=1e-15)
        self.assertEqual(len(PHASES), 4)


if __name__ == "__main__":
    unittest.main()
