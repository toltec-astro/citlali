#!/usr/bin/env python3
"""Tests for the SCI-ALIGN-001 same-T0 cadence-lattice diagnostic."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name(
    "analyze_sci_align_001_same_t0_cadence_lattice.py"
)
SPEC = importlib.util.spec_from_file_location("same_t0_lattice", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SameT0CadenceLatticeTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        aggregate = root / "aggregate"
        aggregate.mkdir()
        obs_maps = {
            101: "map:a",
            102: "map:b",
            103: "map:c",
        }
        group = "roach-t0:test"
        networks = [0, 1]
        cadence = 0.008192

        (aggregate / "corpus_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "sci-align-001-3c273-aggregate-v2",
                    "grouping_kind": "t0_clocktime_vector",
                }
            )
            + "\n"
        )
        session_rows = []
        map_rows = []
        network_rows = []
        occurrence_rows = []
        anomaly_rows = []
        timing_phase_rows = []
        timing_by_obs = {101: 0.0, 102: 0.010, 103: 0.005}
        slot_by_obs = {101: 0.0, 102: -0.001808, 103: -0.000904}
        phase_by_obs = {101: 0.00400, 102: 0.00402, 103: 0.00398}
        for obs, map_id in obs_maps.items():
            session_rows.append(
                {
                    "map_id": map_id,
                    "obsnum": obs,
                    "analysis_role": "primary",
                    "validation_group_id": group,
                    "core_eligible": "true",
                    "enhanced_eligible": "true",
                }
            )
            map_rows.append(
                {
                    "map_id": map_id,
                    "observation_number": obs,
                    "validation_group_id": group,
                    "cadence_sec": cadence,
                    "analysis_mode": "enhanced",
                    "status": "success",
                    "timing_residual_sec": timing_by_obs[obs],
                }
            )
            timing_phase_rows.append(
                {
                    "record_type": "timing_model",
                    "map_id": map_id,
                    "observation_number": obs,
                    "network_id": "",
                    "model_id": "assigned_slot_k+0_phi+0.0",
                    "timing_residual_sec": timing_by_obs[obs],
                    "native_frame_phase_mean_sec": "",
                    "native_to_assigned_mean_sec": "",
                }
            )
            for network in networks:
                timing = timing_by_obs[obs] + network * 0.001
                phase = phase_by_obs[obs] + network * 0.00001
                slot = slot_by_obs[obs] + network * 0.0001
                mismatch = int(obs == 102 and network == 1)
                network_rows.append(
                    {
                        "map_id": map_id,
                        "observation_number": obs,
                        "validation_group_id": group,
                        "network_id": network,
                        "timing_residual_sec": timing,
                        "timing_se_sec": 0.0005,
                        "native_frame_phase_mean_sec": phase,
                        "native_to_assigned_slot_residual_sec": slot,
                        "status": "available",
                        "available": "true",
                        "pps_transition_count": 10,
                        "pps_time_transition_offset_zero_count": 10,
                        "pps_time_transition_offset_minus_one_count": 0,
                        "pps_time_transition_offset_plus_one_count": 0,
                        "pps_time_transition_offset_other_count": 0,
                        "pps_time_transition_different_row_count": 0,
                        "pps_time_transition_same_row_count": 10,
                        "pps_time_transition_pairing_status": (
                            "unique_ordered_same_or_adjacent_row_bijection"
                        ),
                        "pps_time_increment_mismatch_count": mismatch,
                        "variable_metadata_capture_or_isr_latency_observed": (
                            "false"
                        ),
                        "raw_linkage_status": "proved_original_row_one_to_one",
                        "raw_timestamp_physical_semantics": "unresolved",
                        "t0_integer_sec": 1000 + network,
                    }
                )
                occurrence_rows.append(
                    {
                        "map_id": map_id,
                        "observation_number": obs,
                        "network_id": network,
                        "t0_session_id": group,
                        "mismatch_count": mismatch,
                        "isolated_count": mismatch,
                        "consecutive_count": 0,
                        "eligible_increment_count": 9,
                        "mismatch_rate": mismatch / 9,
                    }
                )
                timing_phase_rows.append(
                    {
                        "record_type": "raw_phase_summary",
                        "map_id": map_id,
                        "observation_number": obs,
                        "network_id": network,
                        "model_id": "",
                        "timing_residual_sec": "",
                        "native_frame_phase_mean_sec": phase,
                        "native_to_assigned_mean_sec": slot,
                    }
                )
                if mismatch:
                    anomaly_rows.append(
                        {
                            "map_id": map_id,
                            "observation_number": obs,
                            "network_id": network,
                            "t0_session_id": group,
                            "cluster_class": "isolated",
                        }
                    )

        files = {
            "map_summary.csv": map_rows,
            "network_map_results.csv": network_rows,
            "pps_time_increment_occurrence.csv": occurrence_rows,
            "raw_pps_time_increment_anomalies.csv": anomaly_rows,
            "session_registry.csv": session_rows,
            "timing_phase_results.csv": timing_phase_rows,
        }
        for name, rows in files.items():
            write_csv(aggregate / name, rows)
        sums_lines = []
        for path in sorted(aggregate.iterdir()):
            if path.name == "SHA256SUMS":
                continue
            sums_lines.append(f"{MODULE.sha256(path)}  {path.name}\n")
        (aggregate / "SHA256SUMS").write_text("".join(sums_lines))

        input_files = []
        for path in sorted(aggregate.iterdir()):
            if path.name == "SHA256SUMS":
                continue
            input_files.append(
                {"name": path.name, "role": "test", "sha256": MODULE.sha256(path)}
            )
        protocol = {
            "schema": "sci-align-001-same-t0-cadence-lattice-protocol-v1",
            "analysis_scope": "test",
            "cadence_sec": cadence,
            "half_cadence_sec": cadence / 2,
            "input_sha256sums_sha256": MODULE.sha256(aggregate / "SHA256SUMS"),
            "input_files": input_files,
            "frozen_group": {
                "group_id": group,
                "maps": [
                    {"observation_number": obs, "map_id": map_id}
                    for obs, map_id in obs_maps.items()
                ],
                "expected_network_ids": networks,
            },
            "pair_order": [[101, 102], [101, 103], [102, 103]],
            "classification_policy": {"universal_correction": "prohibited"},
        }
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        return aggregate, protocol_path

    def test_complete_run_and_pairwise_math(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            aggregate, protocol = self.make_fixture(root)
            output = root / "output"
            MODULE.run(aggregate, protocol, output)
            summary = json.loads((output / "diagnostic_summary.json").read_text())
            self.assertEqual(summary["joined_record_count"], 6)
            self.assertEqual(summary["pairwise_record_count"], 6)
            self.assertEqual(summary["association_class_counts"], {"same_row_only": 6})
            self.assertEqual(summary["increment_anomaly_class_counts"]["isolated_only"], 1)
            with (output / "pairwise_network_differences.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            row = next(
                item
                for item in rows
                if item["observation_a"] == "101"
                and item["observation_b"] == "102"
                and item["network_id"] == "0"
            )
            self.assertAlmostEqual(float(row["delta_timing_sec"]), 0.010)
            self.assertAlmostEqual(float(row["delta_slot_residual_sec"]), -0.001808)
            self.assertAlmostEqual(float(row["fixed_minus_one_residual_sec"]), 0.008192)
            self.assertEqual(int(row["nearest_full_cadence_index"]), 1)
            self.assertAlmostEqual(float(row["full_cadence_remainder_sec"]), 0.0)
            state = summary["transitive_half_cadence_state"]
            self.assertTrue(state["transitive"])
            self.assertEqual(
                state["state_half_cadence_indices"],
                {"101": 0, "102": 2, "103": 1},
            )
            self.assertTrue((output / "SHA256SUMS").is_file())

    def test_digest_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            aggregate, protocol = self.make_fixture(root)
            with (aggregate / "map_summary.csv").open("a") as stream:
                stream.write("corruption\n")
            output = root / "output"
            with self.assertRaises(MODULE.ContractError):
                MODULE.run(aggregate, protocol, output)
            self.assertFalse(output.exists())

    def test_wrapped_delta_and_nearest_integer(self) -> None:
        self.assertAlmostEqual(MODULE.wrapped_delta(0.007, 0.008), -0.001)
        self.assertEqual(MODULE.nearest_integer(1.5), 2)
        self.assertEqual(MODULE.nearest_integer(-1.5), -2)


if __name__ == "__main__":
    unittest.main()
