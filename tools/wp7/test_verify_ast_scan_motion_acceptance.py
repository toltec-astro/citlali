#!/usr/bin/env python3
"""Focused mutation tests for the WP-7 AST evidence validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_ast_scan_motion_acceptance.py")
SPEC = importlib.util.spec_from_file_location(
    "verify_ast_scan_motion_acceptance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"
EXECUTABLE_SHA256 = "a" * 64


def encode_record(record: dict[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def record_sha256(record_bytes: bytes) -> str:
    return hashlib.sha256(record_bytes).hexdigest()


def participant(network: int) -> dict[str, object]:
    return {
        "network": network,
        "filename": f"toltec{network}_152390_000_0002_2026_02_19_09_00_00.nc",
        "sha256": f"{network + 1:064x}",
        "byte_count": 1000 + network,
        "occurrence_count": 100000,
        "packet_discontinuity_count": 0,
        "available_count": 99990,
        "unavailable_count": 10,
        "telemetry_defect_cause_count": 2,
        "first_time_unix_sec": 1000.0 + network * 0.0025,
        "last_time_unix_sec": 2000.0 + network * 0.0025,
    }


def valid_record() -> dict[str, object]:
    participants = [participant(network) for network in range(11)]
    return {
        "schema": validator.SCHEMA,
        "source_revision": SOURCE_REVISION,
        "executable_revision": SOURCE_REVISION,
        "executable_version": "candidate",
        "executable_sha256": EXECUTABLE_SHA256,
        "citlali_source_clean": True,
        "citlali_ignored_source_state_verified": True,
        "dependency_state_verified": True,
        "kidscpp_revision": validator.KIDSCPP_REVISION,
        "kidscpp_build_patch_sha256": validator.KIDSCPP_PATCH_SHA256,
        "kidscpp_tree": validator.KIDSCPP_TREE,
        "tula_revision": validator.TULA_REVISION,
        "tula_build_patch_sha256": validator.TULA_PATCH_SHA256,
        "tula_tree": validator.TULA_TREE,
        "design_commit": validator.DESIGN_COMMIT,
        "align_repair_commit": validator.ALIGN_REPAIR_COMMIT,
        "design_is_ancestor": True,
        "align_repair_is_ancestor": True,
        "owner_run": True,
        "representative_data": True,
        "authority_policy_id": validator.POLICY_ID,
        "product_role": validator.PRODUCT_ROLE,
        "representative_dataset_id": validator.DATASET_ID,
        "observation": 152390,
        "subobservation": 0,
        "scan": 2,
        "common_analysis_grid_requested": False,
        "persistent_ast_product_published": False,
        "product_inspected_in_memory": True,
        "telescope": {
            "filename": validator.TELESCOPE_FILENAME,
            "sha256": validator.TELESCOPE_SHA256,
            "byte_count": validator.TELESCOPE_BYTE_COUNT,
            "record_count": validator.TELESCOPE_RECORD_COUNT,
            "time_field": validator.TIME_FIELD,
            "ra_field": validator.RA_FIELD,
            "dec_field": validator.DEC_FIELD,
            "observation_goal": "Science",
            "observation_program": "Lissajous",
            "scan_file_valid": 1,
            "source_epoch": 2000.0,
            "source_coordinate_system": 0,
            "nominal_cadence_hz": 50.0,
            "minimum_interval_sec": 0.019,
            "maximum_interval_sec": 0.021,
            "direct_adjacent_maximum_arcsec_per_sec": 1500.0,
            "direct_adjacent_maximizing_record": 2504,
        },
        "apt_bundle": {
            "manifest_sha256": "b" * 64,
            "semantic_sha256": "sha256:" + "c" * 64,
            "envelope_sha256": "sha256:" + "d" * 64,
            "participant_network_count": 11,
        },
        "identity_binding": {
            "requested": 1523900001,
            "effective": 1523900002,
            "observation_resolved": 1523900003,
            "realized": 1523900004,
        },
        "raw_product": {
            "route_profile": "science-lissajous",
            "physical_scan_member_count": 62109,
            "physical_segment_count": 1,
            "raw_direction_valid_count": 62109,
            "quality_classified_count": 62099,
            "telemetry_defect_count": 2,
            "telemetry_defect_records": [2504, 12971],
            "realized_direction_valid_count": 62097,
            "derivative_valid_count": 62067,
            "maximum_available": True,
            "maximum_causes": 0,
            "maximum_speed_arcsec_per_sec": 215.0,
            "maximizing_record": 30000,
            "continuity_run_count": 1,
            "admitted_candidate_count": 61000,
            "derived_record_bytes": 1000000,
            "referenced_source_axis_count": 1,
            "referenced_source_direction_plane_count": 2,
            "referenced_beammap_membership_plane_count": 0,
            "physical_segment_directory_bytes": 8,
        },
        "chunk_invariance": {
            "partition_count": 3,
            "record_mismatch_count": 0,
            "telemetry_support_mismatch_count": 0,
            "derivative_support_mismatch_count": 0,
            "summary_mismatch_count": 0,
        },
        "network_mapping": {
            "timing_scope": "network-specific",
            "total_occurrence_count": 1100000,
            "available_count": 1099890,
            "unavailable_count": 110,
            "support_count": 1099890,
            "identity_mismatch_count": 0,
            "support_mismatch_count": 0,
            "value_mismatch_count": 0,
            "missing_unavailable_cause_count": 0,
            "mapped_owned_bytes": 1000000,
            "nw0_first_time_unix_sec": 1000.0,
            "nw7_first_time_unix_sec": 1000.0175,
            "nw0_nw7_times_distinct": True,
            "participants": participants,
        },
        "performance": {
            "raw_product_wall_time_sec": 0.1,
            "raw_product_cpu_time_sec": 0.1,
            "network_mapping_wall_time_sec": 0.2,
            "network_mapping_cpu_time_sec": 0.2,
            "process_peak_rss_bytes": 1024,
        },
        "unexpected_error_count": 0,
    }


class AcceptanceValidatorTest(unittest.TestCase):
    def test_accepts_complete_record_and_exact_package(self) -> None:
        record = valid_record()
        validator.validate(record)
        encoded = encode_record(record)
        validator.validate_exact_package(
            encoded,
            expected_record_sha256=record_sha256(encoded),
            expected_source_revision=SOURCE_REVISION,
            expected_executable_sha256=EXECUTABLE_SHA256,
        )

    def test_rejects_source_or_executable_substitution(self) -> None:
        for name, value in (
            ("source_revision", "f" * 40),
            ("executable_sha256", "e" * 64),
        ):
            with self.subTest(name=name):
                record = valid_record()
                record[name] = value
                if name == "source_revision":
                    record["executable_revision"] = value
                encoded = encode_record(record)
                with self.assertRaisesRegex(
                    validator.AcceptanceError, "exact package expectation"
                ):
                    validator.validate_exact_package(
                        encoded,
                        expected_record_sha256=record_sha256(encoded),
                        expected_source_revision=SOURCE_REVISION,
                        expected_executable_sha256=EXECUTABLE_SHA256,
                    )

    def test_exact_package_verifies_executable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "runner"
            executable.write_bytes(b"exact AST runner")
            digest = hashlib.sha256(executable.read_bytes()).hexdigest()
            record = valid_record()
            record["executable_sha256"] = digest
            encoded = encode_record(record)
            validator.validate_exact_package(
                encoded,
                expected_record_sha256=record_sha256(encoded),
                expected_source_revision=SOURCE_REVISION,
                expected_executable_sha256=digest,
                executable=str(executable),
            )
            executable.write_bytes(b"substituted")
            with self.assertRaisesRegex(
                validator.AcceptanceError, "executable file"
            ):
                validator.validate_exact_package(
                    encoded,
                    expected_record_sha256=record_sha256(encoded),
                    expected_source_revision=SOURCE_REVISION,
                    expected_executable_sha256=digest,
                    executable=str(executable),
                )

    def test_exact_package_rejects_material_record_substitution(self) -> None:
        trusted = valid_record()
        trusted_bytes = encode_record(trusted)
        trusted_sha256 = record_sha256(trusted_bytes)
        mutations = (
            ("raw maximum", "raw_product", "maximum_speed_arcsec_per_sec", 216.0),
            ("maximizing record", "raw_product", "maximizing_record", 30001),
            ("derived bytes", "raw_product", "derived_record_bytes", 1),
            ("APT manifest", "apt_bundle", "manifest_sha256", "e" * 64),
            (
                "APT semantic identity",
                "apt_bundle",
                "semantic_sha256",
                "sha256:" + "e" * 64,
            ),
            (
                "APT envelope identity",
                "apt_bundle",
                "envelope_sha256",
                "sha256:" + "f" * 64,
            ),
            ("mapped bytes", "network_mapping", "mapped_owned_bytes", 1),
        )
        for name, section, field, value in mutations:
            with self.subTest(name=name):
                record = copy.deepcopy(trusted)
                record[section][field] = value
                validator.validate(record)
                with self.assertRaisesRegex(
                    validator.AcceptanceError, "record bytes disagree"
                ):
                    validator.validate_exact_package(
                        encode_record(record),
                        expected_record_sha256=trusted_sha256,
                        expected_source_revision=SOURCE_REVISION,
                        expected_executable_sha256=EXECUTABLE_SHA256,
                    )

        participant_mutations = (
            ("participant SHA", "sha256", "f" * 64),
            ("participant bytes", "byte_count", 999999),
        )
        for name, field, value in participant_mutations:
            with self.subTest(name=name):
                record = copy.deepcopy(trusted)
                record["network_mapping"]["participants"][0][field] = value
                validator.validate(record)
                with self.assertRaisesRegex(
                    validator.AcceptanceError, "record bytes disagree"
                ):
                    validator.validate_exact_package(
                        encode_record(record),
                        expected_record_sha256=trusted_sha256,
                        expected_source_revision=SOURCE_REVISION,
                        expected_executable_sha256=EXECUTABLE_SHA256,
                    )

    def test_rejects_telescope_or_scientific_boundary_mutation(self) -> None:
        mutations = (
            ("telescope", "sha256", "0" * 64),
            ("telescope", "maximum_interval_sec", 0.031),
            ("telescope", "direct_adjacent_maximum_arcsec_per_sec", 1400.0),
            ("raw_product", "telemetry_defect_records", [2504]),
            ("raw_product", "derivative_valid_count", 62068),
            ("raw_product", "maximum_speed_arcsec_per_sec", 230.0),
        )
        for section, name, value in mutations:
            with self.subTest(section=section, name=name):
                record = valid_record()
                record[section][name] = value
                with self.assertRaises(validator.AcceptanceError):
                    validator.validate(record)

    def test_rejects_chunk_or_mapping_mismatch(self) -> None:
        mutations = (
            ("chunk_invariance", "record_mismatch_count", 1),
            ("network_mapping", "identity_mismatch_count", 1),
            ("network_mapping", "support_mismatch_count", 1),
            ("network_mapping", "value_mismatch_count", 1),
            ("network_mapping", "missing_unavailable_cause_count", 1),
        )
        for section, name, value in mutations:
            with self.subTest(section=section, name=name):
                record = valid_record()
                record[section][name] = value
                with self.assertRaises(validator.AcceptanceError):
                    validator.validate(record)

    def test_rejects_cross_network_or_persistent_product_claim(self) -> None:
        for name, value in (
            ("common_analysis_grid_requested", True),
            ("persistent_ast_product_published", True),
            ("product_inspected_in_memory", False),
        ):
            with self.subTest(name=name):
                record = valid_record()
                record[name] = value
                with self.assertRaises(validator.AcceptanceError):
                    validator.validate(record)

    def test_rejects_network_time_collapse_or_participant_sum_error(self) -> None:
        record = valid_record()
        record["network_mapping"]["nw7_first_time_unix_sec"] = 1000.0
        with self.assertRaisesRegex(
            validator.AcceptanceError, "distinct network times"
        ):
            validator.validate(record)

        record = valid_record()
        record["network_mapping"]["participants"][0]["available_count"] -= 1
        with self.assertRaises(validator.AcceptanceError):
            validator.validate(record)


if __name__ == "__main__":
    unittest.main()
