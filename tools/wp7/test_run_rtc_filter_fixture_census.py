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
        }
        record = {
            "observation": 10,
            "subobservation": 0,
            "scan": 2,
            "common_analysis_grid_requested": False,
            "rtc_route_activated": False,
            "mapping_checks": {
                "identity_mismatch_count": 0,
                "missing_support_count": 0,
            },
        }
        runner.validate_case_result(case, record)
        record["common_analysis_grid_requested"] = True
        with self.assertRaisesRegex(RuntimeError, "common grid"):
            runner.validate_case_result(case, record)
        record["common_analysis_grid_requested"] = False
        record["observation"] = 11
        with self.assertRaisesRegex(RuntimeError, "output scope"):
            runner.validate_case_result(case, record)


if __name__ == "__main__":
    unittest.main()
