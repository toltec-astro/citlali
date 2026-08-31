#!/usr/bin/env python3
"""Focused tests for the WP-7 D2 PSD/line evidence tool."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.wp7 import rtc_filter_psd_line_evidence as evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PsdLineEvidenceTest(unittest.TestCase):
    def make_input(
        self,
        root: Path,
        *,
        timing_domain: str = evidence.NATIVE_TIMING_DOMAIN,
        stage: str = "native_post_cleaning_residual",
        protected: bool = True,
        second_start: float = 20.0,
    ) -> Path:
        fs = 64.0
        run_length = 640
        time = np.concatenate(
            (
                np.arange(run_length) / fs + 1000.0,
                np.arange(run_length) / fs + 1000.0 + second_start,
            )
        )
        occurrence = np.arange(time.size, dtype=np.int64) + 5000
        run_id = np.repeat(np.asarray([10, 11], dtype=np.int64), run_length)
        detector_id = np.asarray([100, 101, 102, 103], dtype=np.int64)
        phase = np.asarray([0.0, 0.2, 0.4, 0.6])
        signal = np.sin(2.0 * np.pi * 18.0 * time[:, None] + phase)
        signal += 0.05 * np.sin(2.0 * np.pi * 3.0 * time[:, None] + phase)
        valid = np.ones(signal.shape, dtype=bool)
        source_excluded = np.zeros(time.shape, dtype=bool)
        source_excluded[100:110] = True

        arrays = {
            "occurrence_id": occurrence,
            "time_sec": time,
            "physical_run_id": run_id,
            "detector_id": detector_id,
            "signal": signal,
            "valid": valid,
            "source_excluded": source_excluded,
        }
        declarations: dict[str, str] = {}
        for name, value in arrays.items():
            path = root / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            declarations[name] = path.name
        credited_protection = protected and timing_domain == evidence.NATIVE_TIMING_DOMAIN
        intervals = []
        if credited_protection:
            intervals.append(
                {
                    "interval_id": "established-18hz",
                    "low_hz": 17.5,
                    "high_hz": 18.5,
                    "effective_before_decimation": True,
                    "operator_evidence_id": "synthetic-predecimation-notch-v1",
                }
            )
        manifest = {
            "schema": evidence.INPUT_SCHEMA,
            "identity": {
                "case_id": "synthetic-two-run",
                "route_family": "science",
                "observation": 1,
                "subobservation": 0,
                "scan": 0,
                "network": 7,
                "array": "a1100",
                "stream_stage": stage,
                "timing_domain": timing_domain,
                "signal_units": "adu",
                "cadence_domain_id": "synthetic-64hz-v1",
            },
            "cadence_domain": {
                "nominal_interval_sec": 1.0 / fs,
                "maximum_fractional_deviation": 1.0e-9,
            },
            "source_mask": {
                "policy_id": "synthetic-source-mask-v1",
                "status": (
                    "applied"
                    if timing_domain == evidence.NATIVE_TIMING_DOMAIN
                    else "absent_discovery"
                ),
                "meaning": "true values are excluded",
            },
            "line_mask": {
                "policy_id": "synthetic-established-line-mask-v1",
                "strategy_id": evidence.ESTABLISHED_LINE_STRATEGY,
                "status": "applied" if credited_protection else "pending",
                "intervals_hz": intervals,
            },
            "arrays": declarations,
        }
        path = root / "input.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return path

    def test_native_residual_builds_psd_envelope_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence.build_evidence(
                evidence.load_input(self.make_input(root)), root / "out"
            )
            self.assertEqual(result["disposition"], "residual_psd_envelope_candidate")
            self.assertEqual(result["identity"]["timing_domain"], "network_native")
            self.assertEqual(result["native_axis"]["physical_run_count"], 2)
            self.assertEqual(result["detector_summary"]["accepted_count"], 4)
            self.assertEqual(
                result["detector_summary"]["aggregate_row_order"],
                ["median", "q90", "q95", "q99", "maximum"],
            )

    def test_legacy_rectangular_input_is_discovery_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence.build_evidence(
                evidence.load_input(
                    self.make_input(
                        root,
                        timing_domain=evidence.DISCOVERY_TIMING_DOMAIN,
                        stage="legacy_ptc_output",
                    )
                ),
                root / "out",
            )
            self.assertEqual(result["disposition"], "discovery_only_non_native_timing")

    def test_missing_line_mask_withholds_residual_envelope_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence.build_evidence(
                evidence.load_input(self.make_input(root, protected=False)),
                root / "out",
            )
            self.assertEqual(
                result["disposition"],
                "measurement_complete_envelope_pending_line_mask",
            )

    def test_complete_no_lines_can_supply_an_empty_established_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root, protected=False)
            document = json.loads(manifest.read_text())
            document["line_mask"]["status"] = "complete_no_lines"
            manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            result = evidence.build_evidence(
                evidence.load_input(manifest), root / "out"
            )
            self.assertEqual(result["disposition"], "residual_psd_envelope_candidate")

    def test_line_inventory_uses_established_detector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence.build_evidence(
                evidence.load_input(
                    self.make_input(root, stage="native_prefilter")
                ),
                root / "out",
            )
            self.assertEqual(
                result["line_inventory"]["strategy_id"],
                evidence.ESTABLISHED_LINE_STRATEGY,
            )
            lines = result["line_inventory"]["lines"]
            self.assertTrue(any(abs(row["center_hz"] - 18.0) < 0.3 for row in lines))

    def test_foldable_line_requires_effective_predecimation_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            protected_root = Path(tmp) / "protected"
            protected_root.mkdir()
            protected = evidence.build_evidence(
                evidence.load_input(
                    self.make_input(
                        protected_root, protected=True, stage="native_prefilter"
                    )
                ),
                protected_root / "out",
            )
            unprotected_root = Path(tmp) / "unprotected"
            unprotected_root.mkdir()
            unprotected = evidence.build_evidence(
                evidence.load_input(
                    self.make_input(
                        unprotected_root, protected=False, stage="native_prefilter"
                    )
                ),
                unprotected_root / "out",
            )
            protected_factor2 = protected["line_inventory"]["factor_summary"][1]
            unprotected_factor2 = unprotected["line_inventory"]["factor_summary"][1]
            self.assertGreater(protected_factor2["foldable_line_count"], 0)
            self.assertEqual(protected_factor2["line_gate"], "not_blocked_by_inventory")
            self.assertEqual(unprotected_factor2["line_gate"], "withhold")

    def test_run_separator_prevents_a_welch_window_crossing_gap(self) -> None:
        signal = np.arange(8, dtype=float)
        valid = np.ones(8, dtype=bool)
        run_id = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
        separated_signal, separated_valid = evidence._insert_run_separators(
            signal, valid, run_id
        )
        self.assertEqual(separated_signal.size, 9)
        self.assertFalse(separated_valid[4])

    def test_postcleaning_lines_cannot_claim_predecimation_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evidence.build_evidence(
                evidence.load_input(self.make_input(root)), root / "out"
            )
            self.assertEqual(
                result["line_inventory"]["ordering_relevance"],
                "diagnostic_only_postcleaning_stream",
            )
            self.assertEqual(result["line_inventory"]["factor_summary"], [])

    def test_distinct_network_native_times_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root0 = Path(tmp) / "nw0"
            root7 = Path(tmp) / "nw7"
            root0.mkdir()
            root7.mkdir()
            path0 = self.make_input(root0, second_start=20.0)
            path7 = self.make_input(root7, second_start=20.0025)
            time7 = np.load(root7 / "time_sec.npy", allow_pickle=False)
            time7 += 0.0025
            np.save(root7 / "time_sec.npy", time7, allow_pickle=False)
            result0 = evidence.build_evidence(evidence.load_input(path0), root0 / "out")
            result7 = evidence.build_evidence(evidence.load_input(path7), root7 / "out")
            self.assertNotEqual(
                result0["native_axis"]["first_time_sec"],
                result7["native_axis"]["first_time_sec"],
            )

    def test_outputs_are_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root)
            evidence.build_evidence(evidence.load_input(manifest), root / "out1")
            evidence.build_evidence(evidence.load_input(manifest), root / "out2")
            names = sorted(path.name for path in (root / "out1").iterdir())
            self.assertEqual(names, sorted(path.name for path in (root / "out2").iterdir()))
            for name in names:
                self.assertEqual(_sha256(root / "out1" / name), _sha256(root / "out2" / name))

    def test_noncontiguous_reused_run_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root)
            run_id = np.load(root / "physical_run_id.npy", allow_pickle=False)
            run_id[-10:] = run_id[0]
            np.save(root / "physical_run_id.npy", run_id, allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "not contiguous"):
                evidence.load_input(manifest)

    def test_effective_line_protection_requires_operator_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root, stage="native_prefilter")
            document = json.loads(manifest.read_text())
            del document["line_mask"]["intervals_hz"][0]["operator_evidence_id"]
            manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            loaded = evidence.load_input(manifest)
            with self.assertRaisesRegex(RuntimeError, "operator evidence"):
                evidence.build_evidence(loaded, root / "out")

    def test_cadence_outside_declared_domain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root)
            time = np.load(root / "time_sec.npy", allow_pickle=False)
            time[200:] += 1.0e-4
            np.save(root / "time_sec.npy", time, allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "cadence domain"):
                evidence.load_input(manifest)

    def test_source_mask_policy_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_input(root)
            document = json.loads(manifest.read_text())
            document["source_mask"]["policy_id"] = ""
            manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "source_mask.policy_id"):
                evidence.load_input(manifest)

    def test_invalid_estimator_controls_fail_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = evidence.load_input(self.make_input(root))
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                evidence.build_evidence(
                    loaded,
                    root / "out",
                    evidence.EstimatorConfig(overlap_frac=1.0),
                )
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
