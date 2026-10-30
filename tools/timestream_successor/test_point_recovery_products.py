import unittest
import numpy as np
from point_recovery_products import signals, response


class RecoveryProducts(unittest.TestCase):
    def test_signed_frozen_shapes_and_taper(self):
        p = [400, 8, 1, 3, 4, 0.2, 99]
        s = signals(p, 9)
        np.testing.assert_array_equal(s["CORE-"], -s["CORE+"])
        np.testing.assert_array_equal(s["REF"], np.zeros((41, 41)))
        y, x = np.indices((41, 41))
        outside = np.hypot((x - 20) * 2, (y - 20) * 2) >= 36
        self.assertTrue(all(np.all(v[outside] == 0) for v in s.values()))
        p[-1] = -999
        for k, v in signals(p, 9).items():
            np.testing.assert_array_equal(v, s[k])

    def test_fixed_domain_and_signed_metrics(self):
        truth = np.array([1.0, -2.0, 0.0])
        valid = np.ones(3, bool)
        self.assertEqual(response(truth, truth, valid)["gain"], 1.0)
        self.assertEqual(response(-truth, -truth, valid)["distortion"], 0.0)
        self.assertEqual(response(2 * truth, truth, valid)["gain"], 2.0)
        valid[-1] = False
        self.assertFalse(response(truth, truth, valid)["available"])

    def test_invalid_truth(self):
        with self.assertRaises(ValueError):
            signals([np.nan] * 7, 9)
        with self.assertRaises(ValueError):
            response(np.zeros(2), np.zeros(2), np.ones(2, bool))


class MeasurementTest(unittest.TestCase):
    def test_frozen_footprints_constant_accounting_and_lost_support(self):
        from point_recovery_products import geometry, measurements, embed, comparison

        g = geometry(
            dict(CDELT1=-2, CDELT2=2, CRPIX1=41, CRPIX2=41),
            (81, 81),
            [10, 8, 0, 3, 4, 0, 2],
        )
        s = np.full((81, 81), 2.0)
        v = np.ones_like(s, bool)
        m = measurements(s, v, g, dict(accepted_parameters=[10, 8, 0, 3, 4, 0, 2]))
        self.assertTrue(all(a["background_subtracted"] == 0 for a in m["apertures"]))
        self.assertAlmostEqual(m["template_background_subtracted"], 0)
        truth = embed(signals([10, 8, 0, 3, 4, 0, 99], 10)["CORE+"], g)
        r, _ = comparison(s + truth, v, s, v, truth, g)
        self.assertAlmostEqual(r["gain"], 1)
        v[40, 40] = False
        r, _ = comparison(s + truth, v, s, np.ones_like(v), truth, g)
        self.assertFalse(r["available"])
        self.assertEqual(r["lost_benchmark_pixels"], 1)

    def test_bad_m0_width(self):
        for major in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                signals([10, 0, 0, 2, 3, 0, 0], major)
