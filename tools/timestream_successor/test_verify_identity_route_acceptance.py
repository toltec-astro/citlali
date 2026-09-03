#!/usr/bin/env python3
"""Focused tests for the Timestream Successor acceptance validator."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_identity_route_acceptance.py")
SPEC = importlib.util.spec_from_file_location(
    "verify_identity_route_acceptance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def valid_record() -> dict[str, object]:
    source_revision = "0123456789abcdef0123456789abcdef01234567"
    network_count = 11
    detector_count = 22
    native_row_count = 2048
    native_occurrences = network_count * native_row_count
    detector_occurrences = detector_count * native_row_count
    paired_numeric_bytes = 2 * detector_occurrences * 8
    paired_coordinate_state_bytes = 4 * detector_occurrences
    paired_occurrence_axis_bytes = 32 * native_occurrences
    paired_detector_axis_bytes = 88 * detector_count
    paired_identity_text_bytes = 100
    return {
        "schema": validator.SCHEMA,
        "subject_candidate_revision": validator.SUBJECT_CANDIDATE_REVISION,
        "subject_candidate_tree": validator.SUBJECT_CANDIDATE_TREE,
        "tooling_revision": source_revision,
        "source_revision": source_revision,
        "executable_revision": source_revision[:9],
        "executable_version": f"candidate-g{source_revision[:9]}",
        "citlali_source_clean": True,
        "executable_sha256": "a" * 64,
        "build_environment": "spack",
        "build_profile": "unity-gcc13",
        "spack_environment_sha256": "c" * 64,
        "spack_environment_byte_count": 512,
        "spack_environment_retained": True,
        "spack_lock_sha256": "b" * 64,
        "spack_lock_byte_count": 1024,
        "spack_lock_retained": True,
        "spack_root_dag": "a" * 32,
        "dependency_state_verified": True,
        "kidscpp_version": "3.1.0",
        "tula_version": "3.1.0",
        "owner_run": True,
        "real_paired_data": True,
        "apt_bundle_verified": True,
        "raw_sources_verified": True,
        "tune_bindings_verified": True,
        "tune_accumulation_explicit": True,
        "product_inspected_in_memory": True,
        "publication_complete": True,
        "route_context_state": "map_facing_context_complete",
        "route_activated": False,
        "ordinary_route_changed": False,
        "canonical_integration_performed": False,
        "representative_science_claim": False,
        "representative_dataset_id": validator.REPRESENTATIVE_DATASET,
        "observation": 152390,
        "subobservation": 0,
        "scan": 2,
        "first_native_row": 20000,
        "native_row_count": native_row_count,
        "mapping_instance_id": "sha256:mapping",
        "telescope_filename": validator.TELESCOPE_FILENAME,
        "telescope_sha256": validator.TELESCOPE_SHA256,
        "telescope_byte_count": validator.TELESCOPE_BYTE_COUNT,
        "telescope_record_count": validator.TELESCOPE_RECORD_COUNT,
        "apt_manifest_sha256": "d" * 64,
        "apt_bundle_semantic_sha256": "sha256:" + "e" * 64,
        "apt_bundle_envelope_sha256": "sha256:" + "f" * 64,
        "config_sha256": "1" * 64,
        "producer_interface_id": validator.PRODUCER_INTERFACE,
        "producer_interface_sha256": validator.PRODUCER_SHA256,
        "occurrence_support_assignment_schema": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_SCHEMA
        ),
        "occurrence_support_assignment_id": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_ID
        ),
        "occurrence_support_assignment_sha256": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_SHA256
        ),
        "occurrence_support_assignment_status": (
            validator.OCCURRENCE_SUPPORT_ASSIGNMENT_STATUS
        ),
        "occurrence_support_assigned_by": "project-owner",
        "occurrence_support_assigned_at_utc": "2026-08-28T12:00:00Z",
        "occurrence_support_calibration_pending": True,
        "occurrence_support_calibration_disposition": (
            validator.OCCURRENCE_SUPPORT_CALIBRATION_DISPOSITION
        ),
        "occurrence_support_event_time_role": "integration_center",
        "occurrence_support_duration_relation": (
            validator.OCCURRENCE_SUPPORT_DURATION_RELATION
        ),
        "ast_present_in_rtc_input_context": True,
        "identity_rtc_ast_dependency": "not_applicable",
        "val_initial_generation": 0,
        "val_committed_finding_count": 0,
        "val_exact_snapshot_bound": True,
        "calibration_product_state": "unavailable_component_not_admitted",
        "calibration_for_ptc_val_evaluation_state": (
            "unavailable_calibration_product_absent"
        ),
        "ptc_product_state": "unavailable_component_not_admitted",
        "ptc_for_map_val_evaluation_state": "unavailable_ptc_product_absent",
        "map_admission_state": "unavailable_calibration_and_ptc_products",
        "map_action_performed": False,
        "terminal_state": "complete",
        "terminal_failure_cause": "none",
        "terminal_failure_detail": "",
        "metrics": {
            "network_count": network_count,
            "detector_count": detector_count,
            "native_occurrence_count": native_occurrences,
            "native_detector_occurrence_count": detector_occurrences,
            "paired_numeric_payload_bytes": paired_numeric_bytes,
            "paired_coordinate_state_bytes": paired_coordinate_state_bytes,
            "paired_occurrence_axis_bytes": paired_occurrence_axis_bytes,
            "paired_detector_axis_bytes": paired_detector_axis_bytes,
            "paired_identity_text_bytes": paired_identity_text_bytes,
            "paired_logical_owned_bytes": (
                paired_numeric_bytes
                + paired_coordinate_state_bytes
                + paired_occurrence_axis_bytes
                + paired_detector_axis_bytes
                + paired_identity_text_bytes
            ),
            "referenced_native_axis_count": network_count,
            "rtc_native_occurrence_count": native_occurrences,
            "rtc_detector_occurrence_count": detector_occurrences,
            "evidence_event_count": 5,
            "direct_x_event_count": 3,
            "direct_r_event_count": 4,
            "x_and_r_event_count": 2,
            "pair_ineligible_cell_count": 5,
            "x_payload_available_cell_count": detector_occurrences,
            "r_payload_available_cell_count": detector_occurrences,
            "x_numerically_valid_cell_count": detector_occurrences - 3,
            "r_numerically_valid_cell_count": detector_occurrences - 4,
            "derived_evidence_bytes": 80,
            "derived_plan_bytes": 0,
            "paired_ingress_value_comparison_count": 2 * detector_occurrences,
            "paired_ingress_identity_comparison_count": detector_count,
            "paired_ingress_member_state_comparison_count": 2 * detector_occurrences,
            "rtc_product_value_comparison_count": 2 * detector_occurrences,
            "identity_comparison_count": detector_occurrences,
            "support_comparison_count": native_occurrences,
            "native_time_comparison_count": native_occurrences,
            "representative_native_comparison_count": native_occurrences,
            "assigned_support_binding_count": native_occurrences,
            "pair_decision_comparison_count": detector_occurrences,
            "pair_causal_evidence_comparison_count": detector_occurrences,
            "chunk_partition_count": 2,
            "chunk_realized_operator_comparison_count": 1,
            "chunk_scientific_comparison_count": detector_occurrences,
            "route_occurrence_binding_count": native_occurrences,
            "ast_mapped_occurrence_count": native_occurrences,
            "ast_available_occurrence_count": native_occurrences - 528,
            "ast_unavailable_occurrence_count": 528,
            "ast_support_count": native_occurrences - 528,
            "val_binding_comparison_count": 8,
            "ast_raw_owned_bytes": 1000000,
            "ast_mapped_owned_bytes": 100000,
            "align_owned_bytes": 0,
            "rtc_input_owned_bytes": 264,
            "rtc_output_owned_bytes": 0,
            "val_owned_bytes": 0,
            "wall_time_sec": 1.0,
            "cpu_time_sec": 0.5,
            "process_peak_rss_bytes": 1024,
            "rtc_owned_numeric_bytes": 0,
            "x_bitwise_mismatch_count": 0,
            "r_bitwise_mismatch_count": 0,
            "paired_ingress_identity_mismatch_count": 0,
            "paired_ingress_member_state_mismatch_count": 0,
            "identity_mismatch_count": 0,
            "support_mismatch_count": 0,
            "assigned_support_binding_mismatch_count": 0,
            "pair_decision_mismatch_count": 0,
            "pair_causal_evidence_mismatch_count": 0,
            "member_cause_mismatch_count": 0,
            "chunk_realized_operator_mismatch_count": 0,
            "chunk_scientific_mismatch_count": 0,
            "selected_time_mismatch_count": 0,
            "representative_native_mismatch_count": 0,
            "route_occurrence_binding_mismatch_count": 0,
            "ast_identity_mismatch_count": 0,
            "ast_support_mismatch_count": 0,
            "val_binding_mismatch_count": 0,
            "native_admission_entry_count": 1,
            "learn_entry_count": 1,
            "consider_entry_count": 1,
            "apply_entry_count": 1,
            "finalization_entry_count": 1,
            "publication_entry_count": 1,
            "unexpected_error_count": 0,
            "unexpected_critical_count": 0,
        },
    }


class AcceptanceValidatorTest(unittest.TestCase):
    def test_accepts_complete_zero_mismatch_record(self) -> None:
        validator.validate(valid_record())

    def test_rejects_wrong_source_or_environment_identity(self) -> None:
        for name, value in (
            ("subject_candidate_revision", "f" * 40),
            ("executable_revision", "f" * 9),
            ("executable_version", "candidate-dirty"),
            ("build_environment", "local-fallback"),
            ("build_profile", "different-profile"),
            ("spack_environment_sha256", "0" * 40),
            ("spack_lock_sha256", "0" * 40),
            ("spack_root_dag", "not-a-concrete-hash"),
            ("apt_manifest_sha256", "0" * 40),
            ("apt_bundle_semantic_sha256", "e" * 64),
            ("apt_bundle_envelope_sha256", "sha256:" + "G" * 64),
            ("config_sha256", "0" * 40),
        ):
            with self.subTest(name=name):
                record = valid_record()
                record[name] = value
                with self.assertRaises(validator.AcceptanceError):
                    validator.validate(record)

    def test_rejects_missing_ast_val_or_truthful_terminal_state(self) -> None:
        for name, value in (
            ("ast_present_in_rtc_input_context", False),
            ("identity_rtc_ast_dependency", "absent"),
            ("val_initial_generation", 1),
            ("val_exact_snapshot_bound", False),
            ("calibration_product_state", "identity"),
            ("map_action_performed", True),
        ):
            with self.subTest(name=name):
                record = valid_record()
                record[name] = value
                with self.assertRaisesRegex(validator.AcceptanceError, name):
                    validator.validate(record)

    def test_rejects_incomplete_ast_or_val_route_binding(self) -> None:
        for name, value in (
            ("route_occurrence_binding_count", 0),
            ("ast_mapped_occurrence_count", 0),
            ("ast_support_count", 0),
            ("val_binding_comparison_count", 7),
            ("ast_support_mismatch_count", 1),
        ):
            with self.subTest(name=name):
                record = copy.deepcopy(valid_record())
                record["metrics"][name] = value
                with self.assertRaisesRegex(validator.AcceptanceError, name):
                    validator.validate(record)

        no_ast_support = copy.deepcopy(valid_record())
        native_occurrences = no_ast_support["metrics"]["native_occurrence_count"]
        no_ast_support["metrics"]["ast_available_occurrence_count"] = 0
        no_ast_support["metrics"]["ast_unavailable_occurrence_count"] = (
            native_occurrences
        )
        no_ast_support["metrics"]["ast_support_count"] = 0
        with self.assertRaisesRegex(
            validator.AcceptanceError, "ast_available_occurrence_count"
        ):
            validator.validate(no_ast_support)

    def test_rejects_different_support_assignment(self) -> None:
        record = valid_record()
        record["occurrence_support_assignment_id"] = "different"
        with self.assertRaisesRegex(
            validator.AcceptanceError, "occurrence_support_assignment_id"
        ):
            validator.validate(record)

    def test_rejects_partial_or_mismatched_scientific_comparison(self) -> None:
        for name, value in (
            ("rtc_product_value_comparison_count", 40958),
            ("pair_causal_evidence_comparison_count", 20479),
            ("chunk_scientific_comparison_count", 20479),
            ("identity_mismatch_count", 1),
        ):
            with self.subTest(name=name):
                record = copy.deepcopy(valid_record())
                record["metrics"][name] = value
                with self.assertRaisesRegex(validator.AcceptanceError, name):
                    validator.validate(record)

    def test_rejects_inconsistent_evidence_summary(self) -> None:
        record = copy.deepcopy(valid_record())
        record["metrics"]["x_numerically_valid_cell_count"] = 0
        with self.assertRaisesRegex(validator.AcceptanceError, "direct_x"):
            validator.validate(record)

    def test_rejects_missing_lifecycle_stage_or_owned_rtc_plane(self) -> None:
        for name, value in (
            ("finalization_entry_count", 0),
            ("publication_entry_count", 0),
            ("rtc_owned_numeric_bytes", 8),
        ):
            with self.subTest(name=name):
                record = copy.deepcopy(valid_record())
                record["metrics"][name] = value
                with self.assertRaisesRegex(validator.AcceptanceError, name):
                    validator.validate(record)

    def test_rejects_smoke_slice_as_representative_evidence(self) -> None:
        record = valid_record()
        record["native_row_count"] = 16
        with self.assertRaisesRegex(validator.AcceptanceError, "native_row_count"):
            validator.validate(record)


if __name__ == "__main__":
    unittest.main()
