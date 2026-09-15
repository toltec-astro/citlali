import unittest
import numpy as np
from rtc_coherent_trial import CoherentEvidence, CoherentPlan, erode, centered
from rtc_coherent_assessment import fit_coupling, group_prediction


class CoherentTests(unittest.TestCase):
    def setup_case(self):
        t = np.arange(6000) * 0.01
        q = (1 + 0.5 * np.sin(0.08 * t)) * np.cos(
            2 * np.pi * 11 * t + 0.03 * np.sin(0.2 * t)
        )
        x = np.zeros((len(t), 4, 2))
        for d in range(4):
            x[:, d, 0] = (d + 1) * q + 0.01 * np.sin(0.7 * t + d)
            x[:, d, 1] = 0.2 * (d + 1) * q + 0.01 * np.cos(0.9 * t + d)
        good = np.ones(x.shape[:2], bool)
        train = t < 20
        e = CoherentEvidence.learn(x, t, good, [[0, 1]], train, 11, 20, "VAL0")
        p = CoherentPlan.consider(e, x, t, good, [[2, 3]], "VAL0", "test-plan")
        return x, t, good, e, p

    def test_heldout_prediction_and_no_self_donation(self):
        x, t, g, e, p = self.setup_case()
        y, s = p.apply(x, "VAL0")
        self.assertLess(np.std(y[(t > 30) & s[:, 2], 2, 0]), 0.05)
        with self.assertRaisesRegex(ValueError, "own donors"):
            CoherentPlan.consider(e, x, t, g, [[0]], "VAL0", "self")
        self.assertFalse(p.scientific_admission)
        self.assertFalse(p.five_second_projection)

    def test_exact_bindings_and_immutability(self):
        x, t, g, e, p = self.setup_case()
        before = x.copy()
        p.apply(x, "VAL0")
        np.testing.assert_array_equal(x, before)
        with self.assertRaises(ValueError):
            p.original[0, 0, 0] = 1
        with self.assertRaises(ValueError):
            p.apply(x, "VAL1")
        changed = x.copy()
        changed[10, 2, 0] += 1
        with self.assertRaises(ValueError):
            p.apply(changed, "VAL0")
        with self.assertRaises(ValueError):
            CoherentPlan.consider(e, x, t + 0.01, g, [[2, 3]], "VAL0", "stale")

    def test_template_projection_and_relearning_are_different(self):
        x, t, g, e, p = self.setup_case()
        delta = np.zeros_like(x)
        delta[:, 2, 0] = np.cos(2 * np.pi * 11 * t) * 0.5
        y, _ = p.apply(x, "VAL0")
        fixed, _ = p.apply(x, "VAL0", injection=delta)
        np.testing.assert_allclose(fixed - y, delta, atol=2e-15)
        projected, _ = p.apply(x, "VAL0", injection=delta, response="projection")
        self.assertGreater(np.max(abs(projected - fixed)), 0.1)
        # r has its own coefficients and the identical diagonal fit operator.
        np.testing.assert_array_equal(projected[:, :, 1], y[:, :, 1])
        donor_sky = delta.copy()
        donor_sky[:, 0, 0] = np.cos(2 * np.pi * 11 * t) * 2
        new = CoherentEvidence.learn(
            x + donor_sky, t, g, [[0, 1]], t < 20, 11, 20, "VAL0"
        )
        self.assertFalse(np.array_equal(e.bases[0], new.bases[0]))

    def test_gap_support_and_nonfinite_contract(self):
        x, t, g, e, p = self.setup_case()
        g[3000, 0] = False
        x[3000, 0, :] = np.nan
        new = CoherentEvidence.learn(x, t, g, [[0, 1]], t < 20, 11, 20, "VAL0")
        self.assertFalse(new.basis_good[0][2980:3021].any())
        g[3000, 0] = True
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            CoherentEvidence.learn(x, t, g, [[0, 1]], t < 20, 11, 20, "VAL0")

    def test_training_heldout_isolation_and_complex_oracle(self):
        rng = np.random.default_rng(43)
        q = rng.normal(size=100) + 1j * rng.normal(size=100)
        y = q[:, None] * np.array([1 + 2j, -2 + 0.4j, 3 - 1j])[None, :]
        tr = np.arange(100) < 40
        ev = np.arange(100) > 45
        with np.errstate(all="raise"):
            np.testing.assert_allclose(
                fit_coupling(q, y), [1 + 2j, -2 + 0.4j, 3 - 1j], atol=2e-15
            )
            a, _ = group_prediction(y, tr, ev, [0], [1, 2])
            z = y.copy()
            z[ev, 1] *= 2
            b, _ = group_prediction(z, tr, ev, [0], [1, 2])
        self.assertEqual(a["complex_coefficients"], b["complex_coefficients"])
        self.assertLess(max(a["heldout_complex_power_error_ratio"]), 1e-28)
        self.assertGreater(b["heldout_complex_power_error_ratio"][0], 0.2)

    def test_filter_support_and_impulse(self):
        g = np.ones(50, bool)
        g[23] = False
        e = erode(g, 3)
        self.assertFalse(e[20:27].any())
        self.assertFalse(e[:3].any())
        x = np.zeros((50, 2))
        x[10] = [1, 2]
        h = np.array([0.2, 0.6, 0.2])
        y = centered(x, h)
        np.testing.assert_allclose(y[9:12], h[:, None] * [1, 2])
        with self.assertRaises(ValueError):
            centered(x, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
