#!/usr/bin/env python3
"""Focused tests for the evidence-only WP-7 numerical-policy calculator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "analyze_rtc_scan_array_policy_candidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_rtc_scan_array_policy_candidate", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
calculator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calculator
SPEC.loader.exec_module(calculator)


class CandidatePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = calculator.make_report()

    def result(self, speed: float, array: str) -> dict[str, object]:
        for entry in self.report["synthetic_speed_sweep"]:
            if entry["speed_arcsec_per_sec"] == speed:
                return entry["arrays"][array]
        self.fail(f"speed {speed} not found")

    def test_array_models_use_calibration_software_frequencies(self) -> None:
        models = self.report["array_models"]
        self.assertEqual(models["a1100"]["center_frequency_ghz"], 272.0)
        self.assertEqual(models["a1400"]["center_frequency_ghz"], 214.0)
        self.assertEqual(models["a2000"]["center_frequency_ghz"], 150.0)
        self.assertAlmostEqual(
            models["a1100"]["airy_intensity_fwhm_arcsec"],
            4.678641378758944,
        )

    def test_fifty_arcsec_per_sec_prototype_is_array_specific(self) -> None:
        self.assertEqual(self.result(50.0, "a1100")["selected_factor"], 2)
        self.assertEqual(self.result(50.0, "a1400")["selected_factor"], 3)
        self.assertEqual(self.result(50.0, "a2000")["selected_factor"], 4)

    def test_factor_does_not_increase_with_speed(self) -> None:
        for array in calculator.ARRAY_CENTER_FREQUENCY_GHZ:
            factors = [
                self.result(speed, array)["selected_factor"]
                for speed in calculator.SPEEDS_ARCSEC_PER_SEC
            ]
            self.assertTrue(
                all(a >= b for a, b in zip(factors, factors[1:])), factors
            )

    def test_fastest_short_wavelength_case_falls_back_to_identity(self) -> None:
        result = self.result(100.0, "a1100")
        self.assertEqual(result["disposition"], "identity_fallback_candidate")
        self.assertEqual(result["selected_factor"], 1)

    def test_every_decimated_prototype_meets_candidate_bounds(self) -> None:
        policy = self.report["candidate_policy"]
        for speed in self.report["synthetic_speed_sweep"]:
            for result in speed["arrays"].values():
                if result["selected_factor"] == 1:
                    continue
                self.assertGreaterEqual(
                    result["output_samples_per_airy_fwhm"],
                    policy["minimum_samples_per_airy_fwhm"],
                )
                self.assertLessEqual(
                    result["half_support_sec_estimate"],
                    policy["maximum_filter_half_support_sec"],
                )
                self.assertGreater(
                    result["alias_stopband_start_hz"],
                    result["science_passband_hz"],
                )

    def test_report_does_not_claim_authority_or_certification(self) -> None:
        self.assertEqual(
            self.report["status"], "evidence_only_not_scientific_authority"
        )
        self.assertIn(
            "feasibility estimates only",
            self.report["method"]["filter_warning"],
        )
        self.assertIn(
            "no accepted AST-valid v_max",
            self.report["representative_observation"]["disposition"],
        )


if __name__ == "__main__":
    unittest.main()
