#!/usr/bin/env python3
"""Focused contract tests for the SCI-ALIGN-001 3C273 per-map runner."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from tools.diagnostics import run_sci_align_001_3c273_beammap as runner
    from tools.diagnostics.sci_align_001_3c273_common import (
        AnalysisProducts,
        AnalysisProtocol,
        ContractError,
        RawLinkageError,
        RawMapping,
        ReductionInputs,
        RUNNER_SCHEMA,
        _finite_or_none,
        atomic_write_json,
        atomic_write_text,
        build_common_support,
        build_row_mapping,
        canonical_json,
        checksum_lines,
        classify_scan_direction,
        explicit_missing_network_rows,
        fit_timing,
        group_selected_scans,
        linear_predictor_diagnostic,
        parse_manifest,
        raw_counter_diagnostics,
        reconstruct_legacy_timestamp,
        resolve_network_t0_session,
        resume_binding_digest,
        resume_is_valid,
        sha256_file,
        source_write_guard,
    )
except ModuleNotFoundError:  # direct execution from tools/diagnostics
    import run_sci_align_001_3c273_beammap as runner  # type: ignore[no-redef]
    from sci_align_001_3c273_common import (  # type: ignore[no-redef]
        AnalysisProducts,
        AnalysisProtocol,
        ContractError,
        RawLinkageError,
        RawMapping,
        ReductionInputs,
        RUNNER_SCHEMA,
        _finite_or_none,
        atomic_write_json,
        atomic_write_text,
        build_common_support,
        build_row_mapping,
        canonical_json,
        checksum_lines,
        classify_scan_direction,
        explicit_missing_network_rows,
        fit_timing,
        group_selected_scans,
        linear_predictor_diagnostic,
        parse_manifest,
        raw_counter_diagnostics,
        reconstruct_legacy_timestamp,
        resolve_network_t0_session,
        resume_binding_digest,
        resume_is_valid,
        sha256_file,
        source_write_guard,
    )


REPO = Path(__file__).resolve().parents[2]
FROZEN_PROTOCOL = (
    REPO
    / "validation/sci_align_001_3c273_corpus_tooling_2026-08-03"
    / "frozen_analysis_protocol.json"
)


def _signed_u32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.uint32).view(np.int32).astype(np.int64)


def _fake_products(candidate: str = "fixture") -> AnalysisProducts:
    summary = {
        "schema": RUNNER_SCHEMA,
        "map_id": candidate,
        "observation_number": 148670,
        "analysis_mode": "core",
        "status": "success",
        "quality": True,
    }
    return AnalysisProducts(
        map_summary=summary,
        map_result={
            "schema": RUNNER_SCHEMA,
            "summary": summary,
            "scope": {"enhanced_analysis_complete": False},
        },
        network_rows=[],
        timing_rows=[],
        fit_control_rows=[],
        fit_controls={},
        scan_registry=[],
        raw_linkage_rows=[],
        raw_counter_rows=[],
        raw_phase_rows=[],
    )


class DirectionAndSupportTests(unittest.TestCase):
    def test_direction_classifier_is_signal_independent(self) -> None:
        valid = np.ones(100, dtype=bool)
        self.assertEqual(
            classify_scan_direction(np.full(100, 3.0), valid, 1.0)[:2],
            ("right", "selected"),
        )
        self.assertEqual(
            classify_scan_direction(np.full(100, -3.0), valid, 1.0)[:2],
            ("left", "selected"),
        )
        self.assertEqual(
            classify_scan_direction(np.full(100, 0.5), valid, 1.0)[0],
            "excluded",
        )
        valid[50] = False
        self.assertEqual(
            classify_scan_direction(np.full(100, 3.0), valid, 1.0)[1],
            "hold_invalid_or_transition_ambiguous",
        )

    def test_registry_grouping_uses_compatibility_order(self) -> None:
        rows = [
            {
                "stable_scan_id": 9,
                "compatibility_ordinal_1based": 3,
                "classification": "right",
            },
            {
                "stable_scan_id": 7,
                "compatibility_ordinal_1based": 1,
                "classification": "left",
            },
            {
                "stable_scan_id": 8,
                "compatibility_ordinal_1based": 2,
                "classification": "excluded",
            },
        ]
        self.assertEqual(
            group_selected_scans(rows),
            {"left": [7], "right": [9], "excluded": [8]},
        )

    def test_raw_slot_linkage_is_one_to_one_and_half_cell_exclusive(self) -> None:
        slots, assigned, residual, row_for_slot = build_row_mapping(
            np.asarray([0.001, 0.011, 0.021]), 0.0, 0.01, 4, iq_rows=3, q_rows=3
        )
        np.testing.assert_array_equal(slots, [0, 1, 2])
        np.testing.assert_allclose(assigned, [0.0, 0.01, 0.02])
        np.testing.assert_allclose(residual, [0.001, 0.001, 0.001])
        np.testing.assert_array_equal(row_for_slot, [0, 1, 2, -1])
        with self.assertRaisesRegex(ContractError, "multiple native detector rows"):
            build_row_mapping(np.asarray([0.001, 0.002]), 0.0, 0.01, 3)
        with self.assertRaisesRegex(ContractError, "row order"):
            build_row_mapping(np.asarray([0.011, 0.001]), 0.0, 0.01, 3)
        with self.assertRaisesRegex(ContractError, "cardinality"):
            build_row_mapping(
                np.asarray([0.001, 0.011]), 0.0, 0.01, 3, iq_rows=2, q_rows=3
            )
        with self.assertRaisesRegex(ContractError, "exclusive half cell"):
            build_row_mapping(np.asarray([0.005, 0.015]), 0.0, 0.01, 3)

    def test_common_support_applies_all_row_guards(self) -> None:
        mapping = RawMapping(
            interface="toltec0",
            network=0,
            path=Path("raw.nc"),
            times=np.arange(5, dtype=float),
            slots=np.arange(5, dtype=np.int64),
            assigned=np.arange(5, dtype=float),
            residual=np.zeros(5),
            row_for_slot=np.arange(5, dtype=np.int64),
            packet_gap_events=0,
            sample_rate_hz=1.0,
            cadence_sec=1.0,
            fpga_hz=1.0,
            accumulation_ticks=1,
            timestamp_fields=np.zeros((5, 6), dtype=np.int64),
            counter_transitions=[],
            phase_summary={},
        )
        np.testing.assert_array_equal(
            build_common_support(mapping, 5, (-1, 0, 1)),
            [False, True, True, True, False],
        )


class CounterAndProtocolTests(unittest.TestCase):
    def test_packet_counter_wrap_and_gaps_are_transport_diagnostics(self) -> None:
        fields = np.zeros((3, 6), dtype=np.int64)
        fields[:, 0] = 100
        fields[:, 2] = [0, 1, 2]
        fields[:, 3] = [2**31 - 1, -(2**31), -(2**31) + 1]
        np.testing.assert_allclose(
            reconstruct_legacy_timestamp(fields, fpga_hz=1.0),
            [99.0, 100.0, 101.0],
        )
        fields[2, 3] += 1
        np.testing.assert_allclose(
            reconstruct_legacy_timestamp(fields, fpga_hz=1.0),
            [99.0, 100.0, 101.0],
        )

    def test_deterministic_json_rejects_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ContractError, r"non-finite.*\$\.value"):
            canonical_json({"value": np.float64(np.nan)})

    def test_unavailable_fit_diagnostics_use_null_without_permitting_nan(self) -> None:
        self.assertIsNone(_finite_or_none(np.nan))
        self.assertIsNone(_finite_or_none(np.inf))
        self.assertEqual(canonical_json({"optional": _finite_or_none(np.nan)}), '{"optional":null}')
        self.assertEqual(
            fit_timing({"quality": True}, {"quality": True}, np.array([1.0, 0.0]), None, 1.0),
            {"quality": False, "reason": "direction_speed_unavailable"},
        )

    def test_linear_predictor_reports_constant_response_as_unavailable(self) -> None:
        diagnostic = linear_predictor_diagnostic(
            [
                {
                    "available": True,
                    "native_to_assigned_slot_residual_sec": value,
                    "timing_residual_sec": 0.0,
                }
                for value in (-0.001, 0.0, 0.001)
            ],
            "native_to_assigned_slot_residual_sec",
        )
        self.assertFalse(diagnostic["available"])
        self.assertEqual(diagnostic["reason"], "response_has_no_network_leverage")

    def test_counter_inventory_preserves_t0_and_pairs_transitions(self) -> None:
        fpga_hz = 1_000_000.0
        accumulation = 8192
        transition_rows = np.floor(
            np.arange(1, 130, dtype=float) * 15625.0 / 128.0
        ).astype(int)
        row_count = int(transition_rows[-1] + 4)
        fields = np.zeros((row_count, 6), dtype=np.int64)
        fields[:, 0] = 1_768_000_000
        fields[:, 1] = np.cumsum(np.isin(np.arange(row_count), transition_rows))
        clock_u32 = (
            np.uint64(2**32 - 50_000)
            + np.arange(row_count, dtype=np.uint64) * np.uint64(accumulation)
        ) % np.uint64(2**32)
        fields[:, 2] = _signed_u32(clock_u32)
        fields[:, 3] = np.arange(100, 100 + row_count)
        pps_time_transition_rows = transition_rows.copy()
        pps_time_transition_rows[1] += 1
        pps_time_u32 = (
            np.uint64(3_000_000_000)
            + np.cumsum(
                np.isin(np.arange(row_count), pps_time_transition_rows),
                dtype=np.uint64,
            )
            * np.uint64(int(fpga_hz))
        ) % np.uint64(2**32)
        fields[:, 4] = _signed_u32(pps_time_u32)
        fields[:, 5] = 125_000_000

        rows, summary, anomalies = raw_counter_diagnostics(
            fields, network=0, fpga_hz=fpga_hz, accumulation_ticks=accumulation
        )
        self.assertEqual(len(rows), len(transition_rows))
        self.assertEqual(summary["t0_integer_sec"], 1_768_000_000)
        self.assertEqual(summary["clock_time_nanosec_values_json"], "[125000000]")
        self.assertTrue(summary["pps_spacing_all_122_or_123"])
        self.assertEqual(summary["repeat_128_interval_mismatch_count"], 0)
        self.assertEqual(summary["clock_increment_mismatch_count"], 0)
        self.assertEqual(summary["packet_increment_mismatch_count"], 0)
        self.assertEqual(summary["pps_time_increment_mismatch_count"], 0)
        self.assertEqual(anomalies, [])
        self.assertEqual(
            summary["pps_time_transition_pairing_status"],
            "unique_ordered_same_or_adjacent_row_bijection",
        )
        self.assertEqual(summary["pps_time_transition_offset_zero_count"], len(rows) - 1)
        self.assertEqual(summary["pps_time_transition_offset_plus_one_count"], 1)
        self.assertTrue(summary["variable_metadata_capture_or_isr_latency_observed"])
        self.assertEqual(
            rows[1]["native_frame_phase_row_zero_based"], transition_rows[1] + 1
        )
        self.assertFalse(rows[1]["count_row_geometry_is_native_frame_phase"])
        self.assertTrue(rows[1]["native_frame_phase_available"])
        self.assertFalse(rows[0]["metadata_to_integration_association_proved"])

        misaligned = fields.copy()
        misaligned_transition_rows = transition_rows.copy()
        misaligned_transition_rows[1] += 5
        misaligned_pps_time_u32 = (
            np.uint64(3_000_000_000)
            + np.cumsum(
                np.isin(np.arange(row_count), misaligned_transition_rows),
                dtype=np.uint64,
            )
            * np.uint64(int(fpga_hz))
        ) % np.uint64(2**32)
        misaligned[:, 4] = _signed_u32(misaligned_pps_time_u32)
        misaligned_rows, misaligned_summary, _ = raw_counter_diagnostics(
            misaligned,
            network=0,
            fpga_hz=fpga_hz,
            accumulation_ticks=accumulation,
        )
        self.assertEqual(
            misaligned_summary["pps_time_transition_pairing_status"],
            "ambiguous_transition_geometry",
        )
        self.assertIsNone(misaligned_summary["native_frame_phase_mean_sec"])
        self.assertFalse(misaligned_summary["variable_latency_inference_authorized"])
        self.assertTrue(
            all(not row["native_frame_phase_available"] for row in misaligned_rows)
        )

        anomalous = fields.copy()
        anomalous_pps_time = pps_time_u32.copy()
        anomalous_pps_time[int(pps_time_transition_rows[10]):] += np.uint64(1)
        anomalous[:, 4] = _signed_u32(anomalous_pps_time)
        _, anomalous_summary, anomaly_rows = raw_counter_diagnostics(
            anomalous,
            network=9,
            fpga_hz=fpga_hz,
            accumulation_ticks=accumulation,
        )
        self.assertEqual(anomalous_summary["pps_time_increment_mismatch_count"], 1)
        self.assertEqual(anomalous_summary["pps_time_increment_eligible_count"], len(transition_rows) - 1)
        self.assertEqual(len(anomaly_rows), 1)
        self.assertEqual(anomaly_rows[0]["network_id"], 9)
        self.assertEqual(anomaly_rows[0]["signed_tick_residual"], 1)
        self.assertFalse(anomaly_rows[0]["metadata_to_integration_association_proved"])

    def test_frozen_protocol_binds_authority_and_limits_fixture_exclusions(self) -> None:
        protocol = AnalysisProtocol.from_json(FROZEN_PROTOCOL)
        self.assertEqual(
            protocol.authority_schema_version,
            "sci-align-001-3c273-corpus-protocol-v2",
        )
        self.assertEqual(protocol.authority_document_sha256, sha256_file(FROZEN_PROTOCOL))
        self.assertEqual(len(protocol.enhanced_models), 4)
        self.assertEqual(protocol.excluded_uids_for_observation(148671), ())
        self.assertEqual(
            protocol.excluded_uids_for_observation(148670),
            (0, 5, 10, 15, 20, 25, 30, 35),
        )

        with tempfile.TemporaryDirectory() as temporary:
            changed = json.loads(FROZEN_PROTOCOL.read_text())
            changed["fit_quality"]["minimum_matched_detectors"] = 99
            path = Path(temporary) / "changed-protocol.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "differs from implemented"):
                AnalysisProtocol.from_json(path)

    def test_missing_network_rows_are_explicit(self) -> None:
        rows = explicit_missing_network_rows(
            [0, 1, 2],
            {1: {"map_id": "m", "network_id": 1, "timing_residual_sec": 0.1}},
            "m",
        )
        self.assertEqual([row["network_id"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]["status"], "missing_network")
        self.assertTrue(rows[1]["available"])


class ManifestResumeAndRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_inputs(self, *, enhanced: bool = False) -> ReductionInputs:
        source = self.root / "source"
        reduction = source / "reduced" / "redu00" / "148670"
        config = source / "config" / "realized.yaml"
        products = {
            "detector_tod": reduction / "raw" / "source_crossing_tod" / "tod.nc",
            "output_apt": reduction / "raw" / "apt_fixture_citlali.ecsv",
            "provenance": reduction / "timestream_output_provenance.yaml",
            "telescope": source / "data" / "tel.nc",
            "raw": source / "data" / "toltec0_148670.nc",
        }
        for path in [config, *products.values()]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        config.write_text(
            json.dumps(
                {
                    "inputs": [
                        {
                            "data_items": [
                                {
                                    "filepath": str(products["telescope"]),
                                    "meta": {"interface": "lmt"},
                                },
                                {
                                    "filepath": str(products["raw"]),
                                    "meta": {"interface": "toltec0"},
                                },
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return ReductionInputs(
            candidate_id="fixture-148670",
            observation_number=148670,
            reduction_path=reduction,
            config_path=config,
            detector_tod_path=products["detector_tod"],
            output_apt_path=products["output_apt"],
            provenance_path=products["provenance"],
            telescope_path=products["telescope"],
            project_path=None,
            raw_by_network={0: products["raw"]} if enhanced else {},
            core_eligible=True,
            enhanced_eligible=enhanced,
        )

    def test_selected_manifest_schema_aliases_are_accepted(self) -> None:
        inputs = self.make_inputs(enhanced=True)
        manifest = self.root / "selected.json"
        network_t0_vector = [{"network": 0, "t0": 1765472667}]
        network_t0_digest = hashlib.sha256(
            canonical_json(network_t0_vector).encode("utf-8")
        ).hexdigest()
        manifest_base = {
            "schema_version": "sci-align-001-3c273-selected-manifest-v2",
            "source_inventory_sha256": "a" * 64,
            "owner_selection_sha256": "b" * 64,
            "owner_selection_format": "csv",
            "obsnum_allowlist_sha256": "c" * 64,
            "obsnum_allowlist_schema_version": "sci-align-001-3c273-obsnum-allowlist-v1",
            "obsnum_allowlist_filename": "allowlist.json",
            "rows": [
                        {
                            "candidate_id": inputs.candidate_id,
                            "observation_number": inputs.observation_number,
                            "reduction_path": str(inputs.reduction_path),
                            "config_path": str(inputs.config_path),
                            "detector_tod_path": str(inputs.detector_tod_path),
                            "output_apt_path": str(inputs.output_apt_path),
                            "provenance_path": str(inputs.provenance_path),
                            "telescope_path": str(inputs.telescope_path),
                            "analysis_role": "primary",
                            "raw_files": [
                                {
                                    "network": 0,
                                    "path": str(inputs.raw_by_network[0]),
                                    "sha256": sha256_file(inputs.raw_by_network[0]),
                                    "digest_status": "sha256",
                                    "size_bytes": inputs.raw_by_network[0].stat().st_size,
                                }
                            ],
                            "network_t0_status": "complete_unambiguous",
                            "network_t0_vector": network_t0_vector,
                            "network_t0_vector_sha256": network_t0_digest,
                            "core_eligible": "true",
                            "enhanced_eligible": "true",
                        }
                    ],
        }
        (self.root / "allowlist.json").write_text("{}\n", encoding="utf-8")
        # Bind the artificial selected manifest to a real byte payload.
        manifest_base["obsnum_allowlist_sha256"] = sha256_file(self.root / "allowlist.json")
        manifest_document = {
            **manifest_base,
            "manifest_sha256": hashlib.sha256(
                canonical_json(manifest_base).encode("utf-8")
            ).hexdigest(),
        }
        manifest.write_text(
            json.dumps(manifest_document),
            encoding="utf-8",
        )
        parsed = parse_manifest(manifest)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].candidate_id, inputs.candidate_id)
        self.assertEqual(parsed[0].analysis_role, "primary")
        self.assertEqual(
            {key: value.resolve() for key, value in parsed[0].raw_by_network.items()},
            {key: value.resolve() for key, value in inputs.raw_by_network.items()},
        )
        self.assertEqual(
            parsed[0].supplied_sha256[str(parsed[0].raw_by_network[0])],
            sha256_file(inputs.raw_by_network[0]),
        )
        self.assertEqual(parsed[0].network_t0_vector, ((0, 1765472667),))
        self.assertEqual(parsed[0].network_t0_vector_sha256, network_t0_digest)

        duplicate = json.loads(json.dumps(manifest_document["rows"][0]))
        duplicate["candidate_id"] += "-second-primary"
        manifest_document["rows"].append(duplicate)
        manifest_document["manifest_sha256"] = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in manifest_document.items()
                    if key != "manifest_sha256"
                }
            ).encode("utf-8")
        ).hexdigest()
        manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "more than one primary"):
            parse_manifest(manifest)

        manifest_document["rows"] = manifest_document["rows"][:1]
        manifest_document["rows"][0]["analysis_role"] = "sensitivity"
        manifest_document["manifest_sha256"] = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in manifest_document.items()
                    if key != "manifest_sha256"
                }
            ).encode("utf-8")
        ).hexdigest()
        manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "lacks exactly one primary"):
            parse_manifest(manifest)

        manifest_document["rows"][0]["analysis_role"] = "primary"
        manifest_document["rows"][0]["observation_number"] += 1
        manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "internal digest mismatch"):
            parse_manifest(manifest)

    def test_manifest_t0_authority_matches_raw_canonical_session_identity(self) -> None:
        inputs = self.make_inputs(enhanced=True)
        authoritative_vector = [{"network": 0, "t0": 1765472667}]
        authoritative_digest = hashlib.sha256(
            canonical_json(authoritative_vector).encode("utf-8")
        ).hexdigest()
        inputs.network_t0_vector = ((0, 1765472667),)
        inputs.network_t0_vector_sha256 = authoritative_digest
        inputs.network_t0_status = "complete_unambiguous"
        raw_summary = [
            {
                "network_id": 0,
                "t0_integer_sec": 1765472667,
                "t0_integer_value_count": 1,
            }
        ]

        session = resolve_network_t0_session(inputs, raw_summary, [0], True)
        self.assertEqual(session["network_t0_vector"], authoritative_vector)
        self.assertEqual(
            session["network_t0_vector_sha256"], authoritative_digest
        )
        self.assertEqual(
            session["raw_recomputed_network_t0_vector_sha256"],
            authoritative_digest,
        )
        self.assertTrue(session["manifest_authority_validated_against_raw"])
        self.assertNotIn("digest_sha256", session)

        mismatched_raw_summary = [
            {
                "network_id": 0,
                "t0_integer_sec": 1765472668,
                "t0_integer_value_count": 1,
            }
        ]
        with self.assertRaisesRegex(
            RawLinkageError, "differs from selected-manifest authority"
        ):
            resolve_network_t0_session(
                inputs, mismatched_raw_summary, [0], True
            )

    def test_digest_cache_hashes_current_file_and_rejects_stale_authority(self) -> None:
        source = self.root / "current-input.bin"
        source.write_bytes(b"first")
        cache_root = self.root / "digest-output"
        cache_root.mkdir()
        cache = runner.DigestCache(cache_root)
        first = sha256_file(source)
        measured, digest_source = cache.digest(source, first)
        self.assertEqual(measured, first)
        self.assertEqual(
            digest_source, "runner_sha256_validated_against_supplied_authority"
        )
        cache.publish()

        source.write_bytes(b"other")
        with self.assertRaisesRegex(ContractError, "differs from supplied authority"):
            cache.digest(source, first)

    def test_resume_digest_is_sensitive_and_checksum_bound(self) -> None:
        protocol = {"name": "frozen", "value": 1}
        inputs = [{"role": "config", "path": "/x", "sha256": "a" * 64}]
        tools = {"runner.py": "b" * 64}
        first = resume_binding_digest(protocol, inputs, tools)
        self.assertNotEqual(
            first,
            resume_binding_digest({**protocol, "value": 2}, inputs, tools),
        )
        self.assertNotEqual(
            first,
            resume_binding_digest(
                protocol, [{**inputs[0], "sha256": "c" * 64}], tools
            ),
        )

        output = self.root / "checksummed"
        output.mkdir()
        atomic_write_json(
            output / "resume_binding.json", {"binding_sha256": first}
        )
        atomic_write_json(output / "map_result.json", {"status": "success"})
        atomic_write_text(output / "SHA256SUMS", checksum_lines(output))
        self.assertTrue(resume_is_valid(output, first))
        atomic_write_json(output / "map_result.json", {"status": "changed"})
        self.assertFalse(resume_is_valid(output, first))

    def test_source_write_guard_rejects_every_product_parent(self) -> None:
        inputs = self.make_inputs(enhanced=True)
        with self.assertRaisesRegex(ContractError, "inside source directory"):
            source_write_guard(inputs, inputs.detector_tod_path.parent / "diagnostic")
        source_write_guard(inputs, self.root / "owner-output")

    def test_run_one_records_selected_manifest_digest_and_direct_null(self) -> None:
        inputs = self.make_inputs()
        selected = self.root / "selected-manifest.json"
        selected.write_text('{"frozen":true}\n', encoding="utf-8")
        output = self.root / "run-output"
        output.mkdir()
        cache = runner.DigestCache(output)
        with mock.patch.object(
            runner, "analyze_reduction", return_value=_fake_products(inputs.candidate_id)
        ):
            status = runner.run_one(
                inputs,
                output,
                AnalysisProtocol(),
                "core",
                False,
                cache,
                selected,
            )
        self.assertEqual(status, "completed")
        candidate_output = output / inputs.candidate_id
        expected = sha256_file(selected)
        binding = json.loads((candidate_output / "resume_binding.json").read_text())
        result = json.loads((candidate_output / "map_result.json").read_text())
        self.assertEqual(binding["selected_manifest_sha256"], expected)
        self.assertEqual(result["selected_manifest_sha256"], expected)
        self.assertEqual(
            binding["protocol"]["candidate_identity"]["analysis_role"], "primary"
        )
        self.assertEqual(result["summary"]["analysis_role"], "primary")

        direct_output = self.root / "direct-output"
        direct_output.mkdir()
        with mock.patch.object(
            runner, "analyze_reduction", return_value=_fake_products(inputs.candidate_id)
        ):
            runner.run_one(
                inputs,
                direct_output,
                AnalysisProtocol(),
                "core",
                False,
                runner.DigestCache(direct_output),
                None,
            )
        direct_directory = direct_output / inputs.candidate_id
        direct_binding = json.loads(
            (direct_directory / "resume_binding.json").read_text()
        )
        direct_result = json.loads((direct_directory / "map_result.json").read_text())
        self.assertIsNone(direct_binding["selected_manifest_sha256"])
        self.assertIsNone(direct_result["selected_manifest_sha256"])

    def test_enhanced_linkage_failure_retains_core_and_exits_partial(self) -> None:
        inputs = self.make_inputs(enhanced=True)
        output = self.root / "enhanced-output"
        output.mkdir()
        with mock.patch.object(
            runner,
            "analyze_reduction",
            side_effect=[
                RawLinkageError("synthetic raw ambiguity"),
                _fake_products(inputs.candidate_id),
            ],
        ):
            status = runner.run_one(
                inputs,
                output,
                AnalysisProtocol(),
                "enhanced",
                False,
                runner.DigestCache(output),
                None,
            )
        self.assertEqual(status, "partial_core_success_enhanced_failed")
        candidate_output = output / inputs.candidate_id
        failure = json.loads((candidate_output / "enhanced_failure.json").read_text())
        summary = json.loads((candidate_output / "map_summary.json").read_text())
        self.assertEqual(failure["stage"], "raw_linkage_proof")
        self.assertEqual(summary["analysis_mode"], "core")
        self.assertEqual(summary["enhanced_status"], "failed")
        self.assertTrue((candidate_output / "map_result.json").is_file())


if __name__ == "__main__":
    unittest.main()
