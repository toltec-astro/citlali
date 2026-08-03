#!/usr/bin/env python3
"""Synthetic tests for the bounded SCI-ALIGN-001 corpus aggregator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "aggregate_sci_align_001_3c273_corpus.py"
SPEC = importlib.util.spec_from_file_location("sci_align_aggregate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
agg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agg
SPEC.loader.exec_module(agg)


NETWORKS = (0, 7, 11)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def t0_vector(value: int) -> list[dict[str, int]]:
    return [{"network": network, "t0": value} for network in NETWORKS]


def t0_digest(value: list[dict[str, int]]) -> str:
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


class SyntheticCorpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = root / "selected_manifest.json"
        self.owner_selection = root / "owner_selection.json"
        self.inventory = root / "candidate_inventory.json"
        self.allowlist = root / "allowlist.json"
        self.template = root / "runner_protocol.json"
        self.freeze_dir = root / "freeze"
        self.maps = root / "maps"
        self.output = root / "aggregate"
        self.rows: list[dict[str, Any]] = []
        self.owner_selection.write_text(json.dumps({
            "schema_version": "sci-align-001-3c273-selection-v2",
            "selection": "synthetic-test-authority",
        }, indent=2, sort_keys=True) + "\n")
        inventory_base = {
            "schema_version": agg.INVENTORY_SCHEMA,
            "rows": [],
            "scope": "synthetic aggregation test authority",
            "obsnum_allowlist": {
                "filename": self.allowlist.name,
                "sha256": "0" * 64,
                "schema_version": "sci-align-001-3c273-obsnum-allowlist-v1",
            },
        }
        self.inventory.write_text(json.dumps({
            **inventory_base,
            "inventory_sha256": agg._semantic_digest(inventory_base),
        }, indent=2, sort_keys=True) + "\n")
        self.template.write_text(json.dumps({
            "schema_version": "sci-align-001-3c273-corpus-protocol-v2",
            "status": "FROZEN_BEFORE_CORPUS_TIMING_RESULTS",
        }, indent=2, sort_keys=True) + "\n")

    def add_manifest_map(
        self,
        map_id: str,
        obsnum: int,
        date: str,
        *,
        session: int | None = None,
        selected: bool = True,
        enhanced: bool = False,
        status: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "candidate_id": map_id,
            "map_id": map_id,
            "observation_number": obsnum,
            "obsnum": obsnum,
            "observation_start_utc": f"{date}T01:02:03Z",
            "observation_date": date,
            "reduction_id": f"redu-{map_id}",
            "duplicate_group_id": f"obs:{obsnum}",
            "selected": selected,
            "analysis_role": "primary" if selected else "sensitivity",
            "selection_status": "canonical" if selected else "duplicate_sensitivity",
            "core_eligible": True,
            "enhanced_eligible": enhanced,
        }
        if session is not None:
            vector = t0_vector(session)
            row.update({
                "session_id": f"roach-t0:{t0_digest(vector)[:20]}",
                "session_status": "network_t0_vector",
                "network_t0_vector": vector,
                "network_t0_vector_sha256": t0_digest(vector),
                "network_t0_status": status or "complete_unambiguous",
            })
        else:
            row.update({"session_id": f"date:{date}", "session_status": "date_group_fallback"})
        self.rows.append(row)

    def freeze(self) -> dict[str, Any]:
        self.allowlist.write_text(json.dumps({
            "schema_version": "sci-align-001-3c273-obsnum-allowlist-v1",
            "corpus_id": "synthetic-aggregation",
            "selection_authority": "test fixture",
            "obsnums": sorted({int(row["obsnum"]) for row in self.rows}),
        }, indent=2, sort_keys=True) + "\n")
        inventory_base = {
            "schema_version": agg.INVENTORY_SCHEMA,
            "rows": self.rows,
            "scope": "synthetic aggregation test authority",
            "obsnum_allowlist": {
                "filename": self.allowlist.name,
                "sha256": agg.sha256_file(self.allowlist),
                "schema_version": "sci-align-001-3c273-obsnum-allowlist-v1",
            },
        }
        self.inventory.write_text(json.dumps({
            **inventory_base,
            "inventory_sha256": agg._semantic_digest(inventory_base),
        }, indent=2, sort_keys=True) + "\n")
        source_inventory_sha = json.loads(
            self.inventory.read_text()
        )["inventory_sha256"]
        self.owner_selection.write_text(json.dumps({
            "schema_version": agg.SELECTION_SCHEMA,
            "source_inventory_sha256": source_inventory_sha,
            "selection_rule": "synthetic provenance-only owner selection",
            "rows": [
                {
                    "candidate_id": row["candidate_id"],
                    "observation_number": row["observation_number"],
                    "selected": bool(row["selected"]),
                    "owner_note": "synthetic test",
                }
                for row in self.rows
            ],
        }, indent=2, sort_keys=True) + "\n")
        base = {
            "schema_version": agg.SELECTED_MANIFEST_SCHEMA,
            "source_inventory_sha256": source_inventory_sha,
            "owner_selection_sha256": agg.sha256_file(self.owner_selection),
            "owner_selection_format": "json",
            "obsnum_allowlist_sha256": agg.sha256_file(self.allowlist),
            "obsnum_allowlist_schema_version": "sci-align-001-3c273-obsnum-allowlist-v1",
            "obsnum_allowlist_filename": self.allowlist.name,
            "rows": self.rows,
        }
        document = {**base, "manifest_sha256": agg._semantic_digest(base)}
        self.manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        agg.write_checksums(
            self.root, (self.manifest, self.owner_selection, self.inventory, self.allowlist),
        )
        code = agg.main([
            "freeze", "--selected-manifest", str(self.manifest),
            "--protocol-template", str(self.template),
            "--output", str(self.freeze_dir),
        ])
        if code != 0:
            raise AssertionError("freeze failed")
        return json.loads((self.freeze_dir / "frozen_analysis_protocol.json").read_text())

    def add_map_output(
        self,
        map_id: str,
        timing: dict[int, float],
        *,
        timing_se: float = 1.0e-4,
        slot: dict[int, float] | None = None,
        phase: dict[int, float] | None = None,
        half_difference: float | None = None,
        half_difference_se: float | None = None,
        counter_anomaly: bool = False,
        nw9_mismatch_count: int = 0,
        partial: bool = False,
        shuffle: bool = False,
        covariance: list[list[float]] | None = None,
    ) -> Path:
        manifest_row = next(row for row in self.rows if row["map_id"] == map_id)
        directory = self.maps / map_id
        directory.mkdir(parents=True, exist_ok=True)
        network_rows = []
        raw_rows = []
        for network in sorted(timing):
            item = {
                "map_id": map_id,
                "observation_number": manifest_row["obsnum"],
                "network_id": network,
                "array": "a1100" if network == 0 else "a1400" if network == 7 else "a2000",
                "available": True,
                "status": "available",
                "detector_count": 50,
                "timing_residual_sec": timing[network],
                "timing_se_sec": timing_se,
                "scan_speed_abs_arcsec_s": 50.0,
                "left_major_fwhm_arcsec": 6.0,
                "left_minor_fwhm_arcsec": 5.0,
                "right_major_fwhm_arcsec": 6.0,
                "right_minor_fwhm_arcsec": 5.0,
            }
            if slot is not None and network in slot:
                item["native_to_assigned_slot_residual_sec"] = slot[network]
            network_rows.append(item)
            if phase is not None and network in phase:
                raw_rows.append({
                    "network_id": network,
                    "raw_linkage_status": "config_proven",
                    "native_frame_phase_mean_sec": phase[network],
                    "native_frame_phase_std_sec": 2.0e-7,
                    "pps_transition_count": 128,
                    "native_to_assigned_mean_sec": slot[network] if slot and network in slot else None,
                    "pps_spacing_other_count": 1 if counter_anomaly else 0,
                    "repeat_128_interval_mismatch_count": 1 if counter_anomaly else 0,
                    "clock_increment_mismatch_count": 0,
                    "packet_increment_mismatch_count": 0,
                    "pps_time_increment_mismatch_count": 0,
                    "pps_time_increment_eligible_count": 127,
                    "pps_time_increment_mismatch_rate": (
                        nw9_mismatch_count / 127 if network == 9 else 0.0
                    ),
                    "pps_time_transition_offset_other_count": 0,
                    "variable_metadata_capture_or_isr_latency_observed": counter_anomaly,
                })
                raw_rows[-1]["pps_time_increment_mismatch_count"] = (
                    nw9_mismatch_count if network == 9 else 0
                )
        if shuffle:
            network_rows = list(reversed(network_rows))
            raw_rows = list(reversed(raw_rows))
        pooled = sum(timing.values()) / len(timing)
        status = "partial_core_success_enhanced_failed" if partial else "success"
        summary = {
            "schema": "sci-align-001-3c273-map-result-v1",
            "map_id": map_id,
            "candidate_id": map_id,
            "observation_number": manifest_row["obsnum"],
            "status": status,
            "quality": True,
            "analysis_mode": "core" if partial or slot is None else "enhanced",
            "timing_residual_sec": pooled,
            "timing_se_sec": timing_se,
            "scan_speed_abs_median_arcsec_s": 50.0,
            "left_major_fwhm_arcsec": 6.0,
            "left_minor_fwhm_arcsec": 5.0,
            "right_major_fwhm_arcsec": 6.0,
            "right_minor_fwhm_arcsec": 5.0,
            "first_second_half_difference_sec": half_difference,
            "first_second_half_difference_se_sec": half_difference_se,
        }
        template_sha = agg.sha256_file(self.template)
        manifest_sha = agg.sha256_file(self.manifest)
        result = {
            "schema": "sci-align-001-3c273-map-result-v1",
            "selected_manifest_sha256": manifest_sha,
            "identity": {"candidate_id": map_id},
            "protocol": {"authority_document_sha256": template_sha},
            "summary": summary,
            "network_results": network_rows,
            "raw_phase_summary": raw_rows,
            "timing_models": [],
        }
        agg.write_json(directory / "map_result.json", result)
        agg.write_json(directory / "map_summary.json", summary)
        agg.write_csv(directory / "network_map_results.csv", network_rows, agg._fields(network_rows))
        if raw_rows:
            agg.write_csv(directory / "raw_phase_summary.csv", raw_rows, agg._fields(raw_rows))
        agg.write_json(directory / "resume_binding.json", {
            "schema": "test-binding",
            "selected_manifest_sha256": manifest_sha,
            "protocol": {"authority_document_sha256": template_sha},
        })
        if covariance is not None:
            covariance_rows = []
            ordered = sorted(timing)
            for i, left in enumerate(ordered):
                for j, right in enumerate(ordered[i:], start=i):
                    covariance_rows.append({
                        "network_i": left,
                        "network_j": right,
                        "covariance_sec2": covariance[i][j],
                        "replicate_count": 10,
                    })
            agg.write_csv(directory / "measurement_covariance.csv", covariance_rows, agg._fields(covariance_rows))
        agg.write_checksums(directory)
        return directory

    def run(self, *, output: Path | None = None) -> tuple[int, dict[str, Any] | None]:
        destination = output or self.output
        code = agg.main([
            "run", "--selected-manifest", str(self.manifest),
            "--frozen-protocol", str(self.freeze_dir / "frozen_analysis_protocol.json"),
            "--map-output-root", str(self.maps),
            "--output", str(destination),
        ])
        summary = json.loads((destination / "corpus_summary.json").read_text()) if code == 0 else None
        return code, summary


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_dates(self, count: int = 4) -> SyntheticCorpus:
        corpus = SyntheticCorpus(self.root)
        for index in range(count):
            corpus.add_manifest_map(
                f"map-{index}", 1000 + index, f"2026-01-{index + 1:02d}",
                enhanced=True,
            )
        protocol = corpus.freeze()
        self.assertEqual(protocol["grouping_kind"], "observing_date" if count >= 3 else "observation_number")
        return corpus

    def test_a_global_stable(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "A")
        self.assertFalse(summary["production_correction_authorized"])
        self.assertTrue((corpus.output / "SHA256SUMS").is_file())
        agg._verify_checksum_file(corpus.output)
        agg._verify_checksum_file(corpus.freeze_dir)

    def test_b_network_stable_with_missing_network(self) -> None:
        corpus = self.make_dates()
        offsets = {0: -0.014, 7: -0.012, 11: -0.010}
        for index in range(4):
            values = dict(offsets)
            if index == 0:
                values.pop(11)
            corpus.add_map_output(f"map-{index}", values)
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "B")

    def test_d_slot_predictable_beta_minus_one(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            slot = {network: (network / 10000.0) + index * (network + 1) * 2.0e-4 for network in NETWORKS}
            timing = {network: -0.008 - slot[network] for network in NETWORKS}
            corpus.add_map_output(f"map-{index}", timing, slot=slot, phase=slot)
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "D")
        self.assertTrue(
            summary["decision"]["structural_followup_supported_by_native_phase_or_slot"]
        )
        beta = summary["predictor_regressions"]["native_to_assigned_slot_residual"]
        self.assertAlmostEqual(beta["beta"], -1.0, places=8)
        self.assertTrue(beta["beta_consistent_with_minus_one_95"])

    def test_d_predictive_beta_not_minus_one(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            slot = {network: -0.003 + network * 1.0e-4 + index * (network + 1) * 1.0e-4 for network in NETWORKS}
            timing = {network: -0.010 + 2.0 * slot[network] for network in NETWORKS}
            corpus.add_map_output(f"map-{index}", timing, slot=slot, phase=slot)
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "D")
        beta = summary["predictor_regressions"]["native_to_assigned_slot_residual"]
        self.assertAlmostEqual(beta["beta"], 2.0, places=8)
        self.assertFalse(beta["beta_consistent_with_minus_one_95"])

    def test_c_t0_session_anchor_and_phase_stability(self) -> None:
        corpus = SyntheticCorpus(self.root)
        offsets = {200: -0.016, 300: -0.012, 400: -0.008, 500: -0.004}
        map_index = 0
        for session_index, session in enumerate(offsets):
            for within in range(2):
                corpus.add_manifest_map(
                    f"map-{map_index}", 2000 + map_index,
                    f"2026-02-{session_index + 1:02d}", session=session, enhanced=True,
                )
                map_index += 1
        protocol = corpus.freeze()
        self.assertEqual(protocol["grouping_kind"], "t0_clocktime_vector")
        map_index = 0
        for session in offsets:
            phase = {network: 0.001 + network * 1.0e-5 for network in (*NETWORKS, 9)}
            for _ in range(2):
                corpus.add_map_output(
                    f"map-{map_index}", {network: offsets[session] for network in NETWORKS},
                    phase=phase,
                )
                map_index += 1
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "C")
        self.assertEqual(summary["session_effects"]["session_count"], 4)
        self.assertFalse(
            summary["decision"]["structural_followup_supported_by_native_phase_or_slot"]
        )
        self.assertIsNone(summary["decision"]["structural_followup"])
        self.assertIn(
            "session timing alone does not support",
            (corpus.output / "REPORT.md").read_text(),
        )

    def test_e_quantified_within_observation_variation_not_clock_drift(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
                half_difference=0.002, half_difference_se=0.0001,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "E")
        self.assertFalse(summary["within_observation_timing_variation"]["clock_drift_claimed"])

    def test_f_unpredictable_and_outlier_retained(self) -> None:
        corpus = self.make_dates()
        patterns = (
            {0: -0.020, 7: -0.010, 11: 0.000},
            {0: 0.001, 7: -0.021, 11: -0.009},
            {0: -0.011, 7: 0.002, 11: -0.022},
            {0: 0.030, 7: -0.013, 11: 0.004},
        )
        for index, values in enumerate(patterns):
            corpus.add_map_output(f"map-{index}", values)
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "F")
        predictions = (corpus.output / "heldout_predictions.csv").read_text()
        self.assertIn("0.029999999999999999", predictions)

    def test_g_insufficient_groups(self) -> None:
        corpus = self.make_dates(3)
        for index in range(3):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "G")

    def test_duplicate_inherits_observation_fold(self) -> None:
        corpus = self.make_dates(3)
        corpus.add_manifest_map(
            "map-duplicate", 1000, "2026-01-01", selected=False,
        )
        protocol = corpus.freeze()
        registry = {row["map_id"]: row for row in protocol["partition_rows"]}
        self.assertEqual(
            registry["map-duplicate"]["validation_group_id"],
            registry["map-0"]["validation_group_id"],
        )
        self.assertEqual(registry["map-duplicate"]["analysis_role"], "duplicate_sensitivity")

    def test_incomplete_t0_falls_to_date_not_provenance_session(self) -> None:
        corpus = SyntheticCorpus(self.root)
        for index in range(3):
            corpus.add_manifest_map(
                f"map-{index}", 3000 + index, f"2026-03-{index + 1:02d}",
                session=500 + index, status="incomplete",
            )
        protocol = corpus.freeze()
        self.assertEqual(protocol["grouping_kind"], "observing_date")

    def test_partial_core_enhanced_failure_keeps_core_and_marks_predictor_inapplicable(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS}, partial=index == 0,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["decision"]["code"], "A")
        self.assertTrue(any("enhanced" in row["limitation"] for row in summary["limitations"]))

    def test_counter_anomaly_exposed_without_clock_drift_claim(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            phase = {network: 0.001 + network * 1.0e-5 for network in NETWORKS}
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
                phase=phase, counter_anomaly=index == 2,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["within_observation_timing_variation"]["counter_anomaly_map_count"], 1)
        self.assertFalse(summary["within_observation_timing_variation"]["clock_drift_claimed"])

    def test_nw9_anomaly_occurrence_denominator_and_leave_out_effect_are_reported(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            phase = {network: 0.001 + network * 1.0e-5 for network in (*NETWORKS, 9)}
            timing = {0: -0.012, 7: -0.012, 9: -0.012 + index * 1.0e-4, 11: -0.012}
            corpus.add_map_output(
                f"map-{index}", timing, phase=phase, nw9_mismatch_count=index,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        nw9 = summary["nw9_pps_time_anomaly"]
        self.assertEqual(nw9["nw9_total_mismatch_count"], 6)
        self.assertEqual(nw9["nw9_total_eligible_increment_count"], 4 * 127)
        sensitivity = (corpus.output / "nw9_timing_sensitivity.csv").read_text()
        self.assertIn("leave_nw9_out_pooled_timing_sec", sensitivity)
        occurrence = (corpus.output / "pps_time_increment_occurrence.csv").read_text()
        self.assertIn("eligible_increment_count", occurrence)

    def test_invalid_psd_covariance_rejected(self) -> None:
        corpus = self.make_dates(3)
        invalid = [
            [1e-8, 2e-8, 0.0],
            [2e-8, 1e-8, 0.0],
            [0.0, 0.0, 1e-8],
        ]
        for index in range(3):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
                covariance=invalid if index == 0 else None,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 2)
        self.assertIsNone(summary)

    def test_manifest_digest_mismatch_rejected(self) -> None:
        corpus = self.make_dates(3)
        for index in range(3):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        document = json.loads(corpus.manifest.read_text())
        document["note"] = "changed after freeze"
        corpus.manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        code, summary = corpus.run()
        self.assertEqual(code, 2)
        self.assertIsNone(summary)

    def test_predictive_block_score_uses_covariance_log_volume_and_parameter_count(self) -> None:
        opposed = agg._predictive_block_metrics(
            agg.np.asarray([1.0, -1.0]),
            agg.np.asarray([[1.0, 0.9], [0.9, 1.0]]),
        )
        self.assertGreater(opposed["mahalanobis"], 10.0)
        narrow = agg._predictive_block_metrics(
            agg.np.zeros(2), agg.np.eye(2) * 1.0e-8,
        )
        inflated = agg._predictive_block_metrics(
            agg.np.zeros(2), agg.np.eye(2) * 1.0e-2,
        )
        self.assertLess(
            narrow["negative_log_predictive_density_per_observation"],
            inflated["negative_log_predictive_density_per_observation"],
        )

        corpus = self.make_dates()
        covariance = [
            [1.0e-8, 0.8e-8, 0.4e-8],
            [0.8e-8, 1.0e-8, 0.5e-8],
            [0.4e-8, 0.5e-8, 1.0e-8],
        ]
        for index in range(4):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
                covariance=covariance,
            )
        self.assertEqual(corpus.run()[0], 0)
        candidates = agg._read_table(corpus.output / "candidate_model_results.csv")
        m0 = next(row for row in candidates if row["model_id"] == "M0_GLOBAL" and row["validation_regime"] == "outer_logo")
        m1 = next(row for row in candidates if row["model_id"] == "M1_NETWORK" and row["validation_regime"] == "outer_logo")
        self.assertEqual(int(m0["fitted_parameter_count_max"]), 1)
        self.assertEqual(int(m1["fitted_parameter_count_max"]), len(NETWORKS))
        predictions = agg._read_table(corpus.output / "heldout_predictions.csv")
        supported = next(
            row for row in predictions
            if row["model_id"] == "M0_GLOBAL" and row["supported"] == "true"
        )
        self.assertGreater(
            float(supported["predictive_block_parameter_covariance_trace_sec2"]), 0.0,
        )
        self.assertIn("jackknife_full", supported["predictive_block_measurement_covariance_sources"])

    def test_freeze_binds_registry_and_tool_and_run_rejects_tool_mismatch(self) -> None:
        corpus = self.make_dates(3)
        protocol_path = corpus.freeze_dir / "frozen_analysis_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        self.assertEqual(
            protocol["candidate_model_registry_sha256"], agg._candidate_registry_digest(),
        )
        self.assertEqual(
            protocol["aggregation_tool"]["script_sha256"], agg.sha256_file(SCRIPT),
        )
        protocol["aggregation_tool"]["script_sha256"] = "0" * 64
        protocol["protocol_sha256"] = agg._semantic_digest(
            protocol, ("protocol_sha256",),
        )
        agg.write_json(protocol_path, protocol)
        agg.write_checksums(corpus.freeze_dir)
        corpus.maps.mkdir(parents=True, exist_ok=True)
        code, summary = corpus.run()
        self.assertEqual(code, 2)
        self.assertIsNone(summary)
        self.assertFalse(corpus.output.exists())

    def test_run_rejects_frozen_candidate_registry_mismatch(self) -> None:
        corpus = self.make_dates(3)
        protocol_path = corpus.freeze_dir / "frozen_analysis_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["candidate_models"][0]["id"] = "M0_TAMPERED"
        protocol["candidate_model_registry_sha256"] = agg._candidate_registry_digest(
            protocol["candidate_models"],
        )
        protocol["aggregation_tool"]["candidate_model_registry_sha256"] = (
            protocol["candidate_model_registry_sha256"]
        )
        protocol["protocol_sha256"] = agg._semantic_digest(
            protocol, ("protocol_sha256",),
        )
        agg.write_json(protocol_path, protocol)
        agg.write_checksums(corpus.freeze_dir)
        corpus.maps.mkdir(parents=True, exist_ok=True)
        code, summary = corpus.run()
        self.assertEqual(code, 2)
        self.assertIsNone(summary)

    def test_exact_manifest_boundary_rejects_legacy_and_invalid_authority(self) -> None:
        legacy_root = self.root / "legacy"
        legacy_root.mkdir()
        legacy = legacy_root / "selected_manifest.json"
        legacy.write_text(json.dumps([]) + "\n")
        with self.assertRaises(agg.AggregateError):
            agg._selected_manifest_document(legacy)

        strict_root = self.root / "strict"
        strict_root.mkdir(parents=True, exist_ok=True)
        corpus = SyntheticCorpus(strict_root)
        corpus.add_manifest_map("map-0", 5000, "2026-04-01")
        corpus.freeze()
        document = json.loads(corpus.manifest.read_text())
        document["source_inventory_sha256"] = "not-a-digest"
        document["manifest_sha256"] = agg._semantic_digest(
            document, ("manifest_sha256",),
        )
        corpus.manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        with self.assertRaises(agg.AggregateError):
            agg._selected_manifest_document(corpus.manifest)

        document["source_inventory_sha256"] = json.loads(
            corpus.inventory.read_text()
        )["inventory_sha256"]
        document["rows"][0]["analysis_role"] = "sensitivity"
        document["rows"][0]["selected"] = False
        document["manifest_sha256"] = agg._semantic_digest(
            document, ("manifest_sha256",),
        )
        corpus.manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        with self.assertRaises(agg.AggregateError):
            agg._selected_manifest_document(corpus.manifest)

    def test_correlated_map_repeats_do_not_pseudoreplicate_drift_gate(self) -> None:
        corpus = SyntheticCorpus(self.root)
        map_index = 0
        for session_index, map_count in enumerate((8, 1, 1, 1)):
            for _ in range(map_count):
                corpus.add_manifest_map(
                    f"map-{map_index}", 6000 + map_index,
                    f"2026-05-{session_index + 1:02d}",
                    session=700 + session_index,
                )
                map_index += 1
        corpus.freeze()
        for index in range(map_index):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
                half_difference=2.0e-4 if index < 8 else 0.0,
                half_difference_se=1.0e-4,
            )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        drift = summary["within_observation_timing_variation"]
        self.assertEqual(drift["quantified_map_count"], 11)
        self.assertEqual(drift["quantified_independent_group_count"], 4)
        self.assertFalse(drift["within_observation_timing_variation_resolved"])

    def test_session_gate_adjusts_persistent_network_and_independent_groups(self) -> None:
        corpus = SyntheticCorpus(self.root)
        for index in range(4):
            corpus.add_manifest_map(
                f"map-{index}", 7000 + index, f"2026-06-{index + 1:02d}",
                session=900 + index,
            )
        corpus.freeze()
        corpus.add_map_output("map-0", {0: -0.020})
        corpus.add_map_output("map-1", {7: -0.012})
        corpus.add_map_output("map-2", {11: -0.004})
        corpus.add_map_output(
            "map-3", {0: -0.020, 7: -0.012, 11: -0.004},
        )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        session = summary["session_effects"]
        self.assertTrue(session["available"])
        self.assertEqual(session["independent_group_count"], 4)
        self.assertFalse(session["session_effect_resolved"])
        self.assertLess(session["session_network_adjusted_mean_range_sec"], 1.0e-10)

    def test_sensitivity_role_is_compared_but_never_fitted(self) -> None:
        corpus = self.make_dates()
        corpus.add_manifest_map(
            "map-sensitivity", 1000, "2026-01-01", selected=False,
        )
        protocol = corpus.freeze()
        registry = {row["map_id"]: row for row in protocol["partition_rows"]}
        self.assertEqual(
            registry["map-sensitivity"]["analysis_role"], "duplicate_sensitivity",
        )
        for index in range(4):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS},
            )
        corpus.add_map_output(
            "map-sensitivity", {network: -0.0115 for network in NETWORKS},
        )
        code, summary = corpus.run()
        self.assertEqual(code, 0)
        assert summary is not None
        self.assertEqual(summary["map_count"], 4)
        sensitivity = summary["duplicate_reduction_sensitivity"]
        self.assertEqual(sensitivity["sensitivity_reduction_count"], 1)
        self.assertEqual(sensitivity["paired_network_comparison_count"], 3)
        self.assertFalse(sensitivity["used_for_model_fitting_or_classification"])
        rows = agg._read_table(corpus.output / "duplicate_reduction_sensitivity.csv")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["used_for_model_fitting_or_classification"] == "false" for row in rows))

    def test_shuffle_preserves_scientific_outputs(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        first = self.root / "aggregate-first"
        code, _ = corpus.run(output=first)
        self.assertEqual(code, 0)
        for index in range(4):
            corpus.add_map_output(
                f"map-{index}", {network: -0.012 for network in NETWORKS}, shuffle=True,
            )
        second = self.root / "aggregate-second"
        code, _ = corpus.run(output=second)
        self.assertEqual(code, 0)
        for name in (
            "candidate_model_results.csv", "heldout_predictions.csv",
            "variance_components.csv", "network_repeatability.csv",
            "corpus_summary.json",
        ):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_run_dry_run_validates_but_writes_nothing(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        destination = self.root / "dry-run-output"
        code = agg.main([
            "run", "--selected-manifest", str(corpus.manifest),
            "--frozen-protocol", str(corpus.freeze_dir / "frozen_analysis_protocol.json"),
            "--map-output-root", str(corpus.maps),
            "--output", str(destination), "--dry-run",
        ])
        self.assertEqual(code, 0)
        self.assertFalse(destination.exists())

    def test_serial_replay_is_byte_identical(self) -> None:
        corpus = self.make_dates()
        for index in range(4):
            corpus.add_map_output(f"map-{index}", {network: -0.012 for network in NETWORKS})
        first = self.root / "serial-first"
        second = self.root / "serial-second"
        self.assertEqual(corpus.run(output=first)[0], 0)
        self.assertEqual(corpus.run(output=second)[0], 0)
        first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
        second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes(), str(relative))


if __name__ == "__main__":
    unittest.main()
