#!/usr/bin/env python3
"""Focused tests for the WP-7 representative-data record validator."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_identity_rtc_acceptance.py")
SPEC = importlib.util.spec_from_file_location(
    "verify_identity_rtc_acceptance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def valid_record() -> dict[str, object]:
    source_revision = "0123456789abcdef0123456789abcdef01234567"
    return {
        "schema": validator.SCHEMA,
        "source_revision": source_revision,
        "executable_revision": f"candidate-g{source_revision[:9]}",
        "executable_sha256": "a" * 64,
        "kidscpp_revision": validator.KIDSCPP_REVISION,
        "kidscpp_build_patch_sha256": validator.KIDSCPP_PATCH_SHA256,
        "tula_revision": validator.TULA_REVISION,
        "tula_build_patch_sha256": validator.TULA_PATCH_SHA256,
        "design_commit": validator.DESIGN_COMMIT,
        "align_repair_commit": validator.ALIGN_REPAIR_COMMIT,
        "design_is_ancestor": True,
        "align_repair_is_ancestor": True,
        "owner_run": True,
        "real_paired_data": True,
        "apt_bundle_verified": True,
        "raw_sources_verified": True,
        "tune_bindings_verified": True,
        "tune_accumulation_explicit": True,
        "product_inspected_in_memory": True,
        "publication_complete": True,
        "representative_dataset_id": "SCI_ALIGN_STAGE7_NGC4449_152390",
        "observation": 152390,
        "first_native_row": 20000,
        "native_row_count": 2048,
        "mapping_instance_id": "sha256:mapping",
        "producer_interface_id": validator.PRODUCER_INTERFACE,
        "producer_interface_sha256": validator.PRODUCER_SHA256,
        "occurrence_support_assignment_schema": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_SCHEMA
        ),
        "occurrence_support_assignment_id": "wp7-provisional-center:152390:1",
        "occurrence_support_assignment_sha256": "b" * 64,
        "occurrence_support_assignment_status": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_STATUS
        ),
        "occurrence_support_assigned_by": "project-owner",
        "occurrence_support_assigned_at_utc": (
            "2026-08-28T12:00:00Z"
        ),
        "occurrence_support_calibration_pending": True,
        "occurrence_support_calibration_disposition": (
            validator.OCCURRENCE_SUPPORT_CALIBRATION_DISPOSITION
        ),
        "occurrence_support_event_time_role": "integration_center",
        "occurrence_support_duration_relation": (
            validator.OCCURRENCE_SUPPORT_DURATION_RELATION
        ),
        "terminal_state": "complete",
        "terminal_failure_cause": "none",
        "terminal_failure_detail": "",
        "metrics": {
            "network_count": 2,
            "detector_count": 10,
            "native_occurrence_count": 4096,
            "native_detector_occurrence_count": 20480,
            "paired_numeric_payload_bytes": 327680,
            "paired_member_state_bytes": 40960,
            "paired_occurrence_interval_bytes": 65536,
            "paired_detector_axis_bytes": 400,
            "paired_identity_text_bytes": 100,
            "paired_logical_owned_bytes": 434676,
            "referenced_native_axis_count": 2,
            "aligned_cell_count": 100,
            "mapped_cell_count": 90,
            "evidence_event_count": 5,
            "direct_x_event_count": 3,
            "direct_r_event_count": 4,
            "x_and_r_event_count": 2,
            "alignment_absence_event_count": 0,
            "pair_ineligible_cell_count": 5,
            "x_numerically_valid_cell_count": 80,
            "r_numerically_valid_cell_count": 80,
            "derived_evidence_bytes": 80,
            "derived_plan_bytes": 40,
            "paired_value_comparison_count": 180,
            "identity_comparison_count": 100,
            "support_comparison_count": 100,
            "assigned_support_binding_count": 4096,
            "pair_decision_comparison_count": 100,
            "pair_causal_evidence_comparison_count": 100,
            "chunk_partition_count": 2,
            "wall_time_sec": 1.0,
            "cpu_time_sec": 0.5,
            "peak_rss_bytes": 1024,
            "rtc_owned_numeric_bytes": 0,
            "x_bitwise_mismatch_count": 0,
            "r_bitwise_mismatch_count": 0,
            "identity_mismatch_count": 0,
            "support_mismatch_count": 0,
            "assigned_support_binding_mismatch_count": 0,
            "pair_decision_mismatch_count": 0,
            "pair_causal_evidence_mismatch_count": 0,
            "member_cause_mismatch_count": 0,
            "chunk_scientific_mismatch_count": 0,
            "selected_time_mismatch_count": 0,
            "representative_native_mismatch_count": 0,
            "ast_interpolation_call_count": 0,
            "cal_call_count": 0,
            "val_call_count": 0,
            "ptc_call_count": 0,
            "map_call_count": 0,
            "unexpected_error_count": 0,
            "unexpected_critical_count": 0,
        },
    }


class AcceptanceValidatorTest(unittest.TestCase):
    def test_accepts_complete_zero_mismatch_record(self) -> None:
        validator.validate(valid_record())

    def test_rejects_unchecked_pair_causal_evidence(self) -> None:
        record = valid_record()
        record["metrics"]["pair_causal_evidence_comparison_count"] = 99
        with self.assertRaisesRegex(
            validator.AcceptanceError,
            "pair_causal_evidence_comparison_count",
        ):
            validator.validate(record)

    def test_rejects_diagnostic_smoke_slice_as_acceptance_evidence(self) -> None:
        record = valid_record()
        record["native_row_count"] = 16
        with self.assertRaisesRegex(
            validator.AcceptanceError, "native_row_count"
        ):
            validator.validate(record)

    def test_rejects_selected_time_or_native_identity_mismatch(self) -> None:
        for name in (
            "selected_time_mismatch_count",
            "representative_native_mismatch_count",
        ):
            with self.subTest(name=name):
                record = copy.deepcopy(valid_record())
                record["metrics"][name] = 1
                with self.assertRaisesRegex(
                    validator.AcceptanceError, f"{name} must be zero"
                ):
                    validator.validate(record)

    def test_rejects_assignment_that_hides_pending_calibration(self) -> None:
        record = valid_record()
        record["occurrence_support_calibration_pending"] = False
        with self.assertRaisesRegex(
            validator.AcceptanceError,
            "occurrence_support_calibration_pending must be true",
        ):
            validator.validate(record)

    def test_rejects_incomplete_assigned_support_binding(self) -> None:
        record = valid_record()
        record["metrics"]["assigned_support_binding_count"] = 4095
        with self.assertRaisesRegex(
            validator.AcceptanceError,
            "assigned_support_binding_count",
        ):
            validator.validate(record)


if __name__ == "__main__":
    unittest.main()
