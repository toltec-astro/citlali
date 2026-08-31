#!/usr/bin/env python3
"""Focused fail-closed tests for the WP-7 fixture-census orchestrator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).with_name("run_rtc_filter_fixture_census.py")
SPEC = importlib.util.spec_from_file_location(
    "run_rtc_filter_fixture_census", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FixtureCensusRunnerTest(unittest.TestCase):
    @staticmethod
    def valid_record() -> dict:
        factors = []
        for factor in range(1, 257):
            factors.append(
                {
                    "factor": factor,
                    "upper_speed_ceiling_arcsec_per_sec": 100.0 / factor,
                    "upper_boundary_inclusive": True,
                    "upper_speed_typed_cause": "scan_speed_above_mode_support",
                    "occurrence_admission_by_network": [
                        {
                            "base_admitted_count": 5,
                            "upper_speed_admitted_count": 4,
                            "scan_speed_above_mode_support_count": 1,
                        }
                    ],
                    "support_erosion": {
                        "status": (
                            "exact-occurrence-local-m1-no-filter"
                            if factor == 1
                            else "pending-exact-filter-coefficients-and-half-support"
                        ),
                        "support_eroded_output_count": 0 if factor == 1 else None,
                    },
                }
            )
        return {
            "schema": runner.RESULT_SCHEMA,
            "numerical_policy_id": runner.NUMERICAL_POLICY_ID,
            "speed_admission_policy_id": runner.SPEED_ADMISSION_POLICY_ID,
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "common_analysis_grid_requested": False,
            "rtc_route_activated": False,
            "automatic_factor_selection_authorized": False,
            "source_clean_asserted": True,
            "apt_bundle": {
                "bundle_kind": "baseline",
                "canonical_bundle_verified": True,
                "detector_raw_inventory_complete": True,
                "matched_detector_relation_available": False,
            },
            "d0_fixture_identity_ready": True,
            "telescope_ast": {
                "policy_id": runner.AST_POLICY_ID,
                "route_profile": "science-lissajous",
                "maximum_available": True,
                "maximum_causes": 0,
                "physical_scan_member_count": 100,
                "physical_segment_count": 1,
                "chunk_record_mismatch_count": 0,
                "chunk_summary_matches": True,
            },
            "mapping_checks": {
                "identity_mismatch_count": 0,
                "missing_support_count": 0,
            },
            "candidate_mode_domains": [
                {
                    "automatic_factor_selection_authorized": False,
                    "factor_candidates": factors,
                }
            ],
        }

    def test_manifest_requires_unique_case_ids(self) -> None:
        document = {
            "schema": "citlali-wp7-rtc-filter-fixture-cases-v1",
            "cases": [{"id": "same"}, {"id": "same"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.yaml"
            path.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(RuntimeError, "not unique"):
                runner.load_cases(path)

    def test_auxiliary_inventory_fails_closed(self) -> None:
        case = {
            "id": "fixture",
            "auxiliary_inputs": [
                {
                    "role": "housekeeping",
                    "pattern": "data/hk-*.nc",
                    "minimum_count": 2,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "hk-one.nc").write_bytes(b"one")
            with self.assertRaisesRegex(RuntimeError, "requires at least 2"):
                runner.auxiliary_inputs(root, case)
            (root / "data" / "hk-two.nc").write_bytes(b"two")
            self.assertEqual(len(runner.auxiliary_inputs(root, case)), 2)

    def test_result_rejects_common_grid_or_scope_substitution(self) -> None:
        case = {
            "id": "fixture",
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "ast_route_profile": "science-lissajous",
        }
        record = self.valid_record()
        runner.validate_case_result(case, record)
        record["common_analysis_grid_requested"] = True
        with self.assertRaisesRegex(RuntimeError, "common grid"):
            runner.validate_case_result(case, record)
        record["common_analysis_grid_requested"] = False
        record["observation"] = 11
        with self.assertRaisesRegex(RuntimeError, "output scope"):
            runner.validate_case_result(case, record)

    def test_result_rejects_selection_or_guessed_filter_support(self) -> None:
        case = {
            "id": "fixture",
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "ast_route_profile": "science-lissajous",
        }
        record = self.valid_record()
        record["automatic_factor_selection_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "factor selection"):
            runner.validate_case_result(case, record)

        record = self.valid_record()
        record["candidate_mode_domains"][0]["factor_candidates"][1][
            "support_erosion"
        ]["support_eroded_output_count"] = 3
        with self.assertRaisesRegex(RuntimeError, "guessed M>1"):
            runner.validate_case_result(case, record)

    def test_verified_baseline_bundle_does_not_require_matched_relation(self) -> None:
        case = {
            "id": "fixture",
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "ast_route_profile": "science-lissajous",
        }
        record = self.valid_record()
        self.assertFalse(
            record["apt_bundle"]["matched_detector_relation_available"]
        )
        runner.validate_case_result(case, record)

    def test_result_rejects_unverified_or_incomplete_apt_inventory(self) -> None:
        case = {
            "id": "fixture",
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "ast_route_profile": "science-lissajous",
        }
        record = self.valid_record()
        record["apt_bundle"]["canonical_bundle_verified"] = False
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            runner.validate_case_result(case, record)

        record = self.valid_record()
        record["apt_bundle"]["detector_raw_inventory_complete"] = False
        with self.assertRaisesRegex(RuntimeError, "inventory is incomplete"):
            runner.validate_case_result(case, record)


if __name__ == "__main__":
    unittest.main()
