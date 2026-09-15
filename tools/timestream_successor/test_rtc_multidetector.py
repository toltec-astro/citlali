import copy
import unittest

from prepare_rtc_multidetector import AFFECTED, CONTROLS, combine, review_template


class MultiDetectorPreparationTests(unittest.TestCase):
    def configs(self):
        # Deliberately tiny coefficients test only exact binding, not science.
        common = dict(observation=152390, network=12, lowpass_identity="fixture",
                      fir=[0.25, 0.5, 0.25],
                      trials=[dict(id="w1-t3", requested_width_hz=1,
                                   requested_span_seconds=3, finite_notch_identity="fixture",
                                   finite_notch=[-0.1, 1.2, -0.1])])
        for key in ("raw", "tune", "manifest", "audit_receipt", "telescope", "effective_config", "ast_acceptance"):
            common[key] = dict(path=key, sha256="a" * 64)
        return {d: dict(copy.deepcopy(common), channel=d,
                        samples=dict(path=f"samples-{d}.f64", sha256="b" * 64))
                for d in AFFECTED + CONTROLS}

    def test_complete_simultaneous_population_and_explicit_filter_choices(self):
        inputs = self.configs()
        before = copy.deepcopy(inputs)
        combined = combine(inputs)
        self.assertEqual(inputs, before)
        self.assertEqual([d["channel"] for d in combined["detectors"]], sorted(inputs))
        self.assertEqual(sum(d["filter"] == "w1-t3" for d in combined["detectors"]), 8)
        self.assertNotIn("events", combined)

    def test_mixed_parents_scope_and_design_fail(self):
        for key, value in (("network", 11), ("observation", 152391),
                           ("fir", [1.0]), ("channel", 999),
                           ("manifest", dict(path="foreign", sha256="c" * 64))):
            with self.subTest(key=key):
                configs = self.configs()
                configs[193][key] = value
                with self.assertRaises(ValueError):
                    combine(configs)
        configs = self.configs()
        del configs[193]
        with self.assertRaises(ValueError):
            combine(configs)
        configs = self.configs()
        configs[269]["trials"][0]["requested_width_hz"] = 2
        with self.assertRaises(ValueError):
            combine(configs)

    def test_template_never_accepts_candidates_or_assumes_stability(self):
        receipt = dict(schema="rtc-multidetector-learning-v1", Apply_performed=False,
                       learning_binding="exact", VAL_generation=0,
                       detectors=[dict(channel=1, occurrence="one")])
        template = review_template(receipt)
        self.assertFalse(template["approved"])
        self.assertEqual(template["events"], [])
        self.assertEqual(template["detectors"][0]["stable_segments"], [])
        self.assertIsNone(template["existing_scans"])
        receipt["Apply_performed"] = True
        with self.assertRaises(ValueError):
            review_template(receipt)


if __name__ == "__main__":
    unittest.main()
