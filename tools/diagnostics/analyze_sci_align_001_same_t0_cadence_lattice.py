#!/usr/bin/env python3
"""Run the frozen SCI-ALIGN-001 same-T0 cadence-lattice comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a frozen input or identity contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_bool(value: str, *, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ContractError(f"{field} must be true or false, got {value!r}")


def require_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ContractError(f"required field {field} is unavailable")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"required field {field} is non-finite")
    return result


def require_int(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ContractError(f"required field {field} is unavailable")
    return int(value)


def nearest_integer(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def wrapped_delta(value: float, period: float) -> float:
    result = (value + 0.5 * period) % period - 0.5 * period
    if result == 0.5 * period:
        return -0.5 * period
    return result


def rms(values: list[float]) -> float:
    if not values:
        raise ContractError("cannot compute RMS of an empty collection")
    return math.sqrt(sum(value * value for value in values) / len(values))


def association_class(row: dict[str, str]) -> str:
    transition_count = require_int(row, "pps_transition_count")
    zero = require_int(row, "pps_time_transition_offset_zero_count")
    minus = require_int(row, "pps_time_transition_offset_minus_one_count")
    plus = require_int(row, "pps_time_transition_offset_plus_one_count")
    other = require_int(row, "pps_time_transition_offset_other_count")
    different = require_int(row, "pps_time_transition_different_row_count")
    if zero + minus + plus + other != transition_count:
        raise ContractError("PPS transition-offset counts do not sum")
    if different != minus + plus + other:
        raise ContractError("PPS different-row count disagrees with offsets")
    if other:
        return "other_offset_present"
    if zero == transition_count:
        return "same_row_only"
    if plus == transition_count:
        return "plus_one_row_only"
    if minus == transition_count:
        return "minus_one_row_only"
    return "mixed_same_or_adjacent"


def anomaly_class(row: dict[str, str]) -> str:
    mismatch = require_int(row, "mismatch_count")
    isolated = require_int(row, "isolated_count")
    consecutive = require_int(row, "consecutive_count")
    if isolated + consecutive != mismatch:
        raise ContractError("PPS increment anomaly-class counts do not sum")
    if mismatch == 0:
        return "none"
    if isolated and consecutive:
        return "mixed_isolated_and_consecutive"
    if isolated:
        return "isolated_only"
    return "consecutive_only"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_and_verify_inputs(
    aggregate_root: Path, protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    sums_path = aggregate_root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ContractError(f"missing aggregate SHA256SUMS: {sums_path}")
    expected_sums_digest = protocol["input_sha256sums_sha256"]
    if sha256(sums_path) != expected_sums_digest:
        raise ContractError("aggregate SHA256SUMS identity mismatch")
    listed: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split(None, 1)
        listed[name.strip()] = digest
    identities = []
    for item in protocol["input_files"]:
        path = aggregate_root / item["name"]
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ContractError(f"frozen input digest mismatch: {item['name']}")
        if listed.get(item["name"]) != actual:
            raise ContractError(f"aggregate manifest disagrees: {item['name']}")
        identities.append(
            {
                **item,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
            }
        )
    return identities


def prepare_joined_records(
    aggregate_root: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    frozen = protocol["frozen_group"]
    group_id = frozen["group_id"]
    map_by_obs = {
        int(item["observation_number"]): str(item["map_id"])
        for item in frozen["maps"]
    }
    obs_by_map = {map_id: obs for obs, map_id in map_by_obs.items()}
    expected_networks = [int(value) for value in frozen["expected_network_ids"]]

    corpus = json.loads((aggregate_root / "corpus_summary.json").read_text())
    if corpus.get("schema_version") != "sci-align-001-3c273-aggregate-v2":
        raise ContractError("unsupported aggregate schema")
    if corpus.get("grouping_kind") != "t0_clocktime_vector":
        raise ContractError("aggregate did not use T0-vector grouping")

    session_rows = [
        row
        for row in read_csv(aggregate_root / "session_registry.csv")
        if row["map_id"] in obs_by_map
    ]
    if len(session_rows) != len(map_by_obs):
        raise ContractError("frozen session registry does not contain exactly three maps")
    session_by_obs: dict[int, dict[str, Any]] = {}
    for row in session_rows:
        obs = int(row["obsnum"])
        if map_by_obs.get(obs) != row["map_id"]:
            raise ContractError("session registry map/observation identity mismatch")
        if row["validation_group_id"] != group_id:
            raise ContractError("map is not in the frozen T0-vector group")
        if row["analysis_role"] != "primary":
            raise ContractError("frozen map is not primary")
        if not parse_bool(row["core_eligible"], field="core_eligible"):
            raise ContractError("frozen map is not core eligible")
        if not parse_bool(row["enhanced_eligible"], field="enhanced_eligible"):
            raise ContractError("frozen map is not enhanced eligible")
        session_by_obs[obs] = row

    map_rows = {
        int(row["observation_number"]): row
        for row in read_csv(aggregate_root / "map_summary.csv")
        if row["map_id"] in obs_by_map
    }
    if set(map_rows) != set(map_by_obs):
        raise ContractError("map summary identity is incomplete")
    cadence = float(protocol["cadence_sec"])
    for obs, row in map_rows.items():
        if row["map_id"] != map_by_obs[obs]:
            raise ContractError("map summary identity mismatch")
        if row["validation_group_id"] != group_id:
            raise ContractError("map summary group identity mismatch")
        if not math.isclose(float(row["cadence_sec"]), cadence, abs_tol=1e-15):
            raise ContractError("map cadence disagrees with frozen cadence")
        if row["analysis_mode"] != "enhanced" or row["status"] != "success":
            raise ContractError("frozen map is not a successful enhanced result")

    network_rows = [
        row
        for row in read_csv(aggregate_root / "network_map_results.csv")
        if row["map_id"] in obs_by_map
    ]
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in network_rows:
        map_id = row["map_id"]
        obs = int(row["observation_number"])
        network = int(row["network_id"])
        if obs_by_map.get(map_id) != obs or map_by_obs.get(obs) != map_id:
            raise ContractError("network result map identity mismatch")
        if row["validation_group_id"] != group_id:
            raise ContractError("network result group identity mismatch")
        if row["status"] != "available" or not parse_bool(
            row["available"], field="available"
        ):
            raise ContractError("frozen network timing is unavailable")
        key = (map_id, network)
        if key in by_key:
            raise ContractError("duplicate map/network result")
        by_key[key] = row
    for obs, map_id in map_by_obs.items():
        actual = sorted(network for key_map, network in by_key if key_map == map_id)
        if actual != expected_networks:
            raise ContractError(f"network identity mismatch for observation {obs}")

    occurrence_rows = [
        row
        for row in read_csv(aggregate_root / "pps_time_increment_occurrence.csv")
        if row["map_id"] in obs_by_map
    ]
    occurrence_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in occurrence_rows:
        key = (row["map_id"], int(row["network_id"]))
        if key in occurrence_by_key:
            raise ContractError("duplicate PPS occurrence row")
        if row["t0_session_id"] != group_id:
            raise ContractError("PPS occurrence group identity mismatch")
        occurrence_by_key[key] = row
    if set(occurrence_by_key) != set(by_key):
        raise ContractError("PPS occurrence support does not match network results")

    raw_anomaly_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in read_csv(aggregate_root / "raw_pps_time_increment_anomalies.csv"):
        if row["map_id"] not in obs_by_map:
            continue
        if row["t0_session_id"] != group_id:
            raise ContractError("raw anomaly group identity mismatch")
        raw_anomaly_counts[(row["map_id"], int(row["network_id"]))][
            row["cluster_class"]
        ] += 1

    raw_phase_rows = {}
    pooled_timing_rows = {}
    for row in read_csv(aggregate_root / "timing_phase_results.csv"):
        if row["map_id"] not in obs_by_map:
            continue
        if row["record_type"] == "raw_phase_summary":
            key = (row["map_id"], int(row["network_id"]))
            if key in raw_phase_rows:
                raise ContractError("duplicate raw-phase summary")
            raw_phase_rows[key] = row
        elif row["record_type"] == "timing_model" and row["model_id"] == (
            "assigned_slot_k+0_phi+0.0"
        ):
            if row["map_id"] in pooled_timing_rows:
                raise ContractError("duplicate pooled baseline timing row")
            pooled_timing_rows[row["map_id"]] = row
    if set(raw_phase_rows) != set(by_key):
        raise ContractError("raw-phase support does not match network results")
    if set(pooled_timing_rows) != set(obs_by_map):
        raise ContractError("pooled timing cross-check is incomplete")

    joined = []
    for obs in sorted(map_by_obs):
        map_id = map_by_obs[obs]
        pooled = pooled_timing_rows[map_id]
        if float(pooled["timing_residual_sec"]) != float(
            map_rows[obs]["timing_residual_sec"]
        ):
            raise ContractError("pooled timing cross-check failed")
        for network in expected_networks:
            key = (map_id, network)
            row = by_key[key]
            occurrence = occurrence_by_key[key]
            raw_phase = raw_phase_rows[key]
            phase = require_float(row, "native_frame_phase_mean_sec")
            slot = require_float(row, "native_to_assigned_slot_residual_sec")
            if phase != require_float(raw_phase, "native_frame_phase_mean_sec"):
                raise ContractError("native phase cross-check failed")
            if slot != require_float(raw_phase, "native_to_assigned_mean_sec"):
                raise ContractError("slot residual cross-check failed")
            mismatch = require_int(occurrence, "mismatch_count")
            if mismatch != require_int(row, "pps_time_increment_mismatch_count"):
                raise ContractError("PPS mismatch count cross-check failed")
            anomaly_counts = raw_anomaly_counts[key]
            if sum(anomaly_counts.values()) != mismatch:
                raise ContractError("raw anomaly rows do not match mismatch count")
            joined.append(
                {
                    "map_id": map_id,
                    "observation_number": obs,
                    "network_id": network,
                    "timing_residual_sec": require_float(row, "timing_residual_sec"),
                    "timing_se_sec": require_float(row, "timing_se_sec"),
                    "native_frame_phase_mean_sec": phase,
                    "native_to_assigned_slot_residual_sec": slot,
                    "association_class": association_class(row),
                    "pairing_status": row["pps_time_transition_pairing_status"],
                    "same_row_transition_count": require_int(
                        row, "pps_time_transition_same_row_count"
                    ),
                    "different_row_transition_count": require_int(
                        row, "pps_time_transition_different_row_count"
                    ),
                    "increment_anomaly_class": anomaly_class(occurrence),
                    "increment_eligible_count": require_int(
                        occurrence, "eligible_increment_count"
                    ),
                    "increment_mismatch_count": mismatch,
                    "increment_mismatch_rate": require_float(
                        occurrence, "mismatch_rate"
                    ),
                    "isolated_anomaly_count": int(anomaly_counts["isolated"]),
                    "consecutive_anomaly_count": int(
                        anomaly_counts["consecutive"]
                    ),
                    "variable_metadata_latency_observed": parse_bool(
                        row["variable_metadata_capture_or_isr_latency_observed"],
                        field="variable_metadata_capture_or_isr_latency_observed",
                    ),
                    "raw_linkage_status": row["raw_linkage_status"],
                    "raw_timestamp_physical_semantics": row[
                        "raw_timestamp_physical_semantics"
                    ],
                    "t0_integer_sec": require_int(row, "t0_integer_sec"),
                }
            )
    return joined, map_rows


def pairwise_records(
    joined: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (int(row["observation_number"]), int(row["network_id"])): row
        for row in joined
    }
    cadence = float(protocol["cadence_sec"])
    half = float(protocol["half_cadence_sec"])
    rows = []
    for obs_a, obs_b in protocol["pair_order"]:
        for network in protocol["frozen_group"]["expected_network_ids"]:
            a = by_key[(int(obs_a), int(network))]
            b = by_key[(int(obs_b), int(network))]
            delta_timing = b["timing_residual_sec"] - a["timing_residual_sec"]
            delta_se = math.hypot(a["timing_se_sec"], b["timing_se_sec"])
            delta_phase = wrapped_delta(
                b["native_frame_phase_mean_sec"]
                - a["native_frame_phase_mean_sec"],
                cadence,
            )
            delta_slot = (
                b["native_to_assigned_slot_residual_sec"]
                - a["native_to_assigned_slot_residual_sec"]
            )
            prediction = -delta_slot
            residual = delta_timing - prediction
            full_index = nearest_integer(residual / cadence)
            full_remainder = residual - full_index * cadence
            half_index = nearest_integer(residual / half)
            half_remainder = residual - half_index * half
            rows.append(
                {
                    "observation_a": int(obs_a),
                    "observation_b": int(obs_b),
                    "map_id_a": a["map_id"],
                    "map_id_b": b["map_id"],
                    "network_id": int(network),
                    "timing_a_sec": a["timing_residual_sec"],
                    "timing_b_sec": b["timing_residual_sec"],
                    "delta_timing_sec": delta_timing,
                    "delta_timing_diagonal_se_sec": delta_se,
                    "delta_timing_diagonal_z": delta_timing / delta_se,
                    "native_phase_a_sec": a["native_frame_phase_mean_sec"],
                    "native_phase_b_sec": b["native_frame_phase_mean_sec"],
                    "delta_native_phase_wrapped_sec": delta_phase,
                    "slot_residual_a_sec": a[
                        "native_to_assigned_slot_residual_sec"
                    ],
                    "slot_residual_b_sec": b[
                        "native_to_assigned_slot_residual_sec"
                    ],
                    "delta_slot_residual_sec": delta_slot,
                    "fixed_minus_one_prediction_sec": prediction,
                    "fixed_minus_one_residual_sec": residual,
                    "nearest_full_cadence_index": full_index,
                    "full_cadence_remainder_sec": full_remainder,
                    "full_cadence_remainder_diagonal_z": full_remainder / delta_se,
                    "full_cadence_within_1p96_diagonal_se": abs(full_remainder)
                    <= 1.96 * delta_se,
                    "nearest_half_cadence_index": half_index,
                    "nearest_half_cadence_index_parity": (
                        "even" if half_index % 2 == 0 else "odd"
                    ),
                    "half_cadence_remainder_sec": half_remainder,
                    "half_cadence_remainder_diagonal_z": half_remainder / delta_se,
                    "half_cadence_within_1p96_diagonal_se": abs(half_remainder)
                    <= 1.96 * delta_se,
                    "association_class_a": a["association_class"],
                    "association_class_b": b["association_class"],
                    "association_class_changed": a["association_class"]
                    != b["association_class"],
                    "increment_anomaly_class_a": a["increment_anomaly_class"],
                    "increment_anomaly_class_b": b["increment_anomaly_class"],
                    "increment_anomaly_class_changed": a[
                        "increment_anomaly_class"
                    ]
                    != b["increment_anomaly_class"],
                    "variable_latency_observed_a": a[
                        "variable_metadata_latency_observed"
                    ],
                    "variable_latency_observed_b": b[
                        "variable_metadata_latency_observed"
                    ],
                }
            )
    return rows


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["observation_a"], row["observation_b"])].append(row)
    summaries = []
    for pair, group in sorted(grouped.items()):
        residuals = [row["fixed_minus_one_residual_sec"] for row in group]
        full_remainders = [row["full_cadence_remainder_sec"] for row in group]
        half_remainders = [row["half_cadence_remainder_sec"] for row in group]
        clean = [
            row
            for row in group
            if row["increment_anomaly_class_a"] == "none"
            and row["increment_anomaly_class_b"] == "none"
        ]
        half_index_counts = Counter(
            row["nearest_half_cadence_index"] for row in group
        )
        ranked_half_indices = sorted(
            half_index_counts.items(), key=lambda item: (-item[1], item[0])
        )
        modal_half_index, modal_half_count = ranked_half_indices[0]
        modal_half_unique = (
            len(ranked_half_indices) == 1
            or modal_half_count > ranked_half_indices[1][1]
        )
        summaries.append(
            {
                "observation_a": pair[0],
                "observation_b": pair[1],
                "network_count": len(group),
                "median_delta_timing_sec": statistics.median(
                    row["delta_timing_sec"] for row in group
                ),
                "median_abs_delta_timing_sec": statistics.median(
                    abs(row["delta_timing_sec"]) for row in group
                ),
                "median_abs_delta_native_phase_wrapped_sec": statistics.median(
                    abs(row["delta_native_phase_wrapped_sec"]) for row in group
                ),
                "median_delta_slot_residual_sec": statistics.median(
                    row["delta_slot_residual_sec"] for row in group
                ),
                "median_fixed_minus_one_residual_sec": statistics.median(
                    residuals
                ),
                "fixed_minus_one_residual_rms_sec": rms(residuals),
                "full_cadence_remainder_rms_sec": rms(full_remainders),
                "half_cadence_remainder_rms_sec": rms(half_remainders),
                "full_cadence_within_1p96_diagonal_se_count": sum(
                    row["full_cadence_within_1p96_diagonal_se"] for row in group
                ),
                "half_cadence_within_1p96_diagonal_se_count": sum(
                    row["half_cadence_within_1p96_diagonal_se"] for row in group
                ),
                "nearest_full_cadence_index_counts_json": json.dumps(
                    dict(
                        sorted(
                            Counter(
                                row["nearest_full_cadence_index"] for row in group
                            ).items()
                        )
                    ),
                    sort_keys=True,
                ),
                "nearest_half_cadence_index_counts_json": json.dumps(
                    dict(sorted(half_index_counts.items())),
                    sort_keys=True,
                ),
                "modal_half_cadence_index": modal_half_index,
                "modal_half_cadence_index_count": modal_half_count,
                "modal_half_cadence_index_fraction": modal_half_count / len(group),
                "modal_half_cadence_index_unique": modal_half_unique,
                "odd_half_cadence_index_count": sum(
                    row["nearest_half_cadence_index_parity"] == "odd"
                    for row in group
                ),
                "association_class_changed_count": sum(
                    row["association_class_changed"] for row in group
                ),
                "increment_anomaly_class_changed_count": sum(
                    row["increment_anomaly_class_changed"] for row in group
                ),
                "clean_control_network_count": len(clean),
                "clean_control_median_abs_fixed_minus_one_residual_sec": (
                    statistics.median(
                        abs(row["fixed_minus_one_residual_sec"]) for row in clean
                    )
                    if clean
                    else None
                ),
            }
        )
    return summaries


def build_transitive_half_cadence_state(
    protocol: dict[str, Any], pair_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    observations = [
        int(row["observation_number"])
        for row in protocol["frozen_group"]["maps"]
    ]
    if not observations:
        raise ContractError("frozen group contains no observations")
    reference = observations[0]
    states: dict[int, int] = {reference: 0}
    pair_modes: list[dict[str, Any]] = []
    for row in pair_summaries:
        obs_a = int(row["observation_a"])
        obs_b = int(row["observation_b"])
        mode = int(row["modal_half_cadence_index"])
        pair_modes.append(
            {
                "observation_a": obs_a,
                "observation_b": obs_b,
                "modal_half_cadence_index": mode,
                "support_count": int(row["modal_half_cadence_index_count"]),
                "network_count": int(row["network_count"]),
                "unique": bool(row["modal_half_cadence_index_unique"]),
            }
        )
        if obs_a == reference:
            states[obs_b] = mode

    complete = set(states) == set(observations)
    all_modes_unique = all(row["unique"] for row in pair_modes)
    pair_checks = []
    for row in pair_modes:
        a = row["observation_a"]
        b = row["observation_b"]
        predicted = states[b] - states[a] if a in states and b in states else None
        pair_checks.append(
            {
                **row,
                "state_predicted_half_cadence_index": predicted,
                "consistent": predicted == row["modal_half_cadence_index"],
            }
        )
    transitive = complete and all_modes_unique and all(
        row["consistent"] for row in pair_checks
    )
    return {
        "reference_observation": reference,
        "state_half_cadence_indices": {
            str(obs): states[obs] for obs in observations if obs in states
        },
        "complete": complete,
        "all_pair_modes_unique": all_modes_unique,
        "transitive": transitive,
        "pair_checks": pair_checks,
    }


def build_summary(
    protocol: dict[str, Any],
    protocol_sha: str,
    input_identities: list[dict[str, Any]],
    joined: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    tool_path: Path,
) -> dict[str, Any]:
    association_counts = Counter(row["association_class"] for row in joined)
    anomaly_counts = Counter(row["increment_anomaly_class"] for row in joined)
    residuals = [row["fixed_minus_one_residual_sec"] for row in pairs]
    phase_changes = [abs(row["delta_native_phase_wrapped_sec"]) for row in pairs]
    timing_changes = [abs(row["delta_timing_sec"]) for row in pairs]
    half_cadence_state = build_transitive_half_cadence_state(
        protocol, pair_summaries
    )
    return {
        "schema": "sci-align-001-same-t0-cadence-lattice-result-v1",
        "analysis_scope": protocol["analysis_scope"],
        "protocol_sha256": protocol_sha,
        "tool": {"path": str(tool_path), "sha256": sha256(tool_path)},
        "input_identities": input_identities,
        "frozen_group": protocol["frozen_group"],
        "cadence_sec": protocol["cadence_sec"],
        "half_cadence_sec": protocol["half_cadence_sec"],
        "joined_record_count": len(joined),
        "pairwise_record_count": len(pairs),
        "association_class_counts": dict(sorted(association_counts.items())),
        "increment_anomaly_class_counts": dict(sorted(anomaly_counts.items())),
        "association_class_change_count": sum(
            row["association_class_changed"] for row in pairs
        ),
        "increment_anomaly_class_change_count": sum(
            row["increment_anomaly_class_changed"] for row in pairs
        ),
        "variable_latency_observed_record_count": sum(
            row["variable_metadata_latency_observed"] for row in joined
        ),
        "median_abs_pairwise_timing_change_sec": statistics.median(timing_changes),
        "median_abs_pairwise_native_phase_change_sec": statistics.median(
            phase_changes
        ),
        "fixed_minus_one_residual_rms_sec": rms(residuals),
        "full_cadence_remainder_rms_sec": rms(
            [row["full_cadence_remainder_sec"] for row in pairs]
        ),
        "half_cadence_remainder_rms_sec": rms(
            [row["half_cadence_remainder_sec"] for row in pairs]
        ),
        "full_cadence_within_1p96_diagonal_se_count": sum(
            row["full_cadence_within_1p96_diagonal_se"] for row in pairs
        ),
        "half_cadence_within_1p96_diagonal_se_count": sum(
            row["half_cadence_within_1p96_diagonal_se"] for row in pairs
        ),
        "odd_half_cadence_index_count": sum(
            row["nearest_half_cadence_index_parity"] == "odd" for row in pairs
        ),
        "transitive_half_cadence_state": half_cadence_state,
        "pair_summaries": pair_summaries,
        "interpretation_boundaries": protocol["classification_policy"],
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SCI-ALIGN-001 same-T0 cadence-lattice comparison",
        "",
        "## Frozen identity",
        "",
        f"- Group: `{summary['frozen_group']['group_id']}`",
        "- Observations: "
        + ", ".join(
            str(row["observation_number"])
            for row in summary["frozen_group"]["maps"]
        ),
        "- Networks: "
        + ", ".join(
            str(value) for value in summary["frozen_group"]["expected_network_ids"]
        ),
        f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
        f"- Joined records: {summary['joined_record_count']}",
        f"- Pairwise records: {summary['pairwise_record_count']}",
        "",
        "## Descriptive results",
        "",
        "| Obs A | Obs B | median delta timing (ms) | median abs phase delta (ms) | median -1-slot residual (ms) | full-lattice RMS (ms) | half-lattice RMS (ms) | modal half step | modal support | full within 1.96 SE | half within 1.96 SE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["pair_summaries"]:
        lines.append(
            "| {observation_a} | {observation_b} | {timing:.6f} | {phase:.6f} | "
            "{residual:.6f} | {full:.6f} | {half:.6f} | {half_mode:+d} | "
            "{half_mode_n}/{count} | {full_n}/{count} | "
            "{half_n}/{count} |".format(
                observation_a=row["observation_a"],
                observation_b=row["observation_b"],
                timing=1000 * row["median_delta_timing_sec"],
                phase=1000 * row["median_abs_delta_native_phase_wrapped_sec"],
                residual=1000 * row["median_fixed_minus_one_residual_sec"],
                full=1000 * row["full_cadence_remainder_rms_sec"],
                half=1000 * row["half_cadence_remainder_rms_sec"],
                half_mode=row["modal_half_cadence_index"],
                half_mode_n=row["modal_half_cadence_index_count"],
                full_n=row["full_cadence_within_1p96_diagonal_se_count"],
                half_n=row["half_cadence_within_1p96_diagonal_se_count"],
                count=row["network_count"],
            )
        )
    state = summary["transitive_half_cadence_state"]
    lines.extend(
        [
            "",
            "## Transitive half-cadence state check",
            "",
            f"Reference observation: {state['reference_observation']}.",
            "Map states in 4.096-ms units: `"
            + json.dumps(state["state_half_cadence_indices"], sort_keys=True)
            + "`.",
            f"All pair modes unique: {str(state['all_pair_modes_unique']).lower()}.",
            f"Pair modes transitive: {str(state['transitive']).lower()}.",
            "",
            "Association classes: `"
            + json.dumps(summary["association_class_counts"], sort_keys=True)
            + "`.",
            "",
            "Increment-anomaly classes: `"
            + json.dumps(summary["increment_anomaly_class_counts"], sort_keys=True)
            + "`.",
            "",
            f"Delivered association-class changes across pairs: {summary['association_class_change_count']}.",
            f"Increment-anomaly-class changes across pairs: {summary['increment_anomaly_class_change_count']}.",
            f"Records with observed variable metadata latency: {summary['variable_latency_observed_record_count']}.",
            "",
            "## Interpretation boundary",
            "",
            "This is a descriptive comparison of delivered compact evidence. "
            "Diagonal timing SE is not cross-map covariance. Same-row delivered "
            "PPS/PpsTime pairing does not prove FPGA metadata-to-integration "
            "association. A common start/end/centroid convention cancels in "
            "pairwise differences. No result authorizes a timing correction.",
            "",
        ]
    )
    return "\n".join(lines)


JOINED_FIELDS = [
    "map_id",
    "observation_number",
    "network_id",
    "timing_residual_sec",
    "timing_se_sec",
    "native_frame_phase_mean_sec",
    "native_to_assigned_slot_residual_sec",
    "association_class",
    "pairing_status",
    "same_row_transition_count",
    "different_row_transition_count",
    "increment_anomaly_class",
    "increment_eligible_count",
    "increment_mismatch_count",
    "increment_mismatch_rate",
    "isolated_anomaly_count",
    "consecutive_anomaly_count",
    "variable_metadata_latency_observed",
    "raw_linkage_status",
    "raw_timestamp_physical_semantics",
    "t0_integer_sec",
]

PAIR_FIELDS = [
    "observation_a",
    "observation_b",
    "map_id_a",
    "map_id_b",
    "network_id",
    "timing_a_sec",
    "timing_b_sec",
    "delta_timing_sec",
    "delta_timing_diagonal_se_sec",
    "delta_timing_diagonal_z",
    "native_phase_a_sec",
    "native_phase_b_sec",
    "delta_native_phase_wrapped_sec",
    "slot_residual_a_sec",
    "slot_residual_b_sec",
    "delta_slot_residual_sec",
    "fixed_minus_one_prediction_sec",
    "fixed_minus_one_residual_sec",
    "nearest_full_cadence_index",
    "full_cadence_remainder_sec",
    "full_cadence_remainder_diagonal_z",
    "full_cadence_within_1p96_diagonal_se",
    "nearest_half_cadence_index",
    "nearest_half_cadence_index_parity",
    "half_cadence_remainder_sec",
    "half_cadence_remainder_diagonal_z",
    "half_cadence_within_1p96_diagonal_se",
    "association_class_a",
    "association_class_b",
    "association_class_changed",
    "increment_anomaly_class_a",
    "increment_anomaly_class_b",
    "increment_anomaly_class_changed",
    "variable_latency_observed_a",
    "variable_latency_observed_b",
]

PAIR_SUMMARY_FIELDS = [
    "observation_a",
    "observation_b",
    "network_count",
    "median_delta_timing_sec",
    "median_abs_delta_timing_sec",
    "median_abs_delta_native_phase_wrapped_sec",
    "median_delta_slot_residual_sec",
    "median_fixed_minus_one_residual_sec",
    "fixed_minus_one_residual_rms_sec",
    "full_cadence_remainder_rms_sec",
    "half_cadence_remainder_rms_sec",
    "full_cadence_within_1p96_diagonal_se_count",
    "half_cadence_within_1p96_diagonal_se_count",
    "nearest_full_cadence_index_counts_json",
    "nearest_half_cadence_index_counts_json",
    "modal_half_cadence_index",
    "modal_half_cadence_index_count",
    "modal_half_cadence_index_fraction",
    "modal_half_cadence_index_unique",
    "odd_half_cadence_index_count",
    "association_class_changed_count",
    "increment_anomaly_class_changed_count",
    "clean_control_network_count",
    "clean_control_median_abs_fixed_minus_one_residual_sec",
]


def run(aggregate_root: Path, protocol_path: Path, output: Path) -> Path:
    if output.exists():
        raise ContractError(f"refusing existing output: {output}")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != (
        "sci-align-001-same-t0-cadence-lattice-protocol-v1"
    ):
        raise ContractError("unsupported frozen protocol schema")
    if not math.isclose(
        float(protocol["half_cadence_sec"]),
        0.5 * float(protocol["cadence_sec"]),
        abs_tol=1e-15,
    ):
        raise ContractError("half cadence is not exactly half the cadence")
    input_identities = load_and_verify_inputs(aggregate_root, protocol)
    joined, _ = prepare_joined_records(aggregate_root, protocol)
    pairs = pairwise_records(joined, protocol)
    pair_summaries = summarize_pairs(pairs)
    summary = build_summary(
        protocol,
        sha256(protocol_path),
        input_identities,
        joined,
        pairs,
        pair_summaries,
        Path(__file__).resolve(),
    )
    output.mkdir(parents=True)
    write_csv(output / "joined_network_records.csv", joined, JOINED_FIELDS)
    write_csv(output / "pairwise_network_differences.csv", pairs, PAIR_FIELDS)
    write_csv(output / "pair_summary.csv", pair_summaries, PAIR_SUMMARY_FIELDS)
    write_json(output / "diagnostic_summary.json", summary)
    write_json(
        output / "input_identity.json",
        {
            "schema": "sci-align-001-same-t0-input-identity-v1",
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": sha256(protocol_path),
            "aggregate_root": str(aggregate_root.resolve()),
            "aggregate_sha256sums_sha256": protocol[
                "input_sha256sums_sha256"
            ],
            "inputs": input_identities,
            "frozen_group": protocol["frozen_group"],
        },
    )
    (output / "REPORT.md").write_text(render_report(summary))
    output_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_files)
    )
    return output / "diagnostic_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            args.aggregate_root.resolve(), args.protocol.resolve(), args.output.resolve()
        )
    except (ContractError, KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
