#!/usr/bin/env python3
"""Fit authenticated SCI-ALIGN event centroids by observation and network."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

import analyze_sci_align_001_lissajous_timestream as analysis
import sci_align_001_lissajous_event_centroids as centroids


class NetworkAuditError(RuntimeError):
    """A campaign input or derived network result violates its contract."""


MODEL_NAMES = ("constant", "lag", "hysteresis", "joint")


def cadence_index(value_ms: float, cadence_ms: float) -> int:
    """Return the nearest cadence index with deterministic half-away rounding."""
    scaled = float(value_ms) / float(cadence_ms)
    if scaled >= 0.0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def cadence_fields(prefix: str, value_ms: float, cadence_ms: float) -> dict[str, Any]:
    index = cadence_index(value_ms, cadence_ms)
    return {
        f"{prefix}_minus_one_cadence_ms": float(value_ms - cadence_ms),
        f"{prefix}_nearest_cadence_index": index,
        f"{prefix}_nearest_cadence_residual_ms": float(
            value_ms - index * cadence_ms
        ),
    }


def improvement(constant: dict[str, Any], fit: dict[str, Any]) -> float:
    baseline = float(constant["objective"])
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise NetworkAuditError("constant-model objective is not positive finite")
    return (baseline - float(fit["objective"])) / baseline


def fit_network_rows(
    centroid_rows: Table,
    protocol: dict[str, Any],
    obsnum: int,
    pooled_lag_tau_ms: float,
    cadence_ms: float,
) -> list[dict[str, Any]]:
    """Fit every retained network using the frozen event qualification."""
    threshold = float(protocol["source_quality"][
        "primary_minimum_correlation"
    ])
    minimum = int(protocol["source_quality"][
        "minimum_scored_samples_per_event"
    ])
    networks = sorted(set(map(int, centroid_rows["network"])))
    result: list[dict[str, Any]] = []
    for network in networks:
        mask = np.asarray(centroid_rows["network"], dtype=int) == network
        rows = centroid_rows[mask]
        qualified = centroids.qualified_mask(rows, threshold, minimum)
        base: dict[str, Any] = {
            "obsnum": int(obsnum),
            "network": int(network),
            "array": "a1100" if 0 <= network <= 6 else (
                "a1400" if 7 <= network <= 10 else "a2000"
            ),
            "status": "insufficient_support",
            "assessed_event_count": len(rows),
            "qualified_event_count": int(np.count_nonzero(qualified)),
            "qualified_detector_count": len(set(map(int, rows[qualified]["uid"]))),
            "pooled_lag_tau_ms": float(pooled_lag_tau_ms),
            "common_robust_scale_arcsec": math.nan,
            "constant_objective_arcsec2": math.nan,
            "lag_tau_ms": math.nan,
            "lag_objective_improvement_fraction": math.nan,
            "lag_design_condition_number": math.nan,
            "lag_boundary": False,
            "lag_minus_pooled_ms": math.nan,
            "lag_minus_one_cadence_ms": math.nan,
            "lag_nearest_cadence_index": 0,
            "lag_nearest_cadence_residual_ms": math.nan,
            "hysteresis_az_half_offset_arcsec": math.nan,
            "hysteresis_el_half_offset_arcsec": math.nan,
            "hysteresis_objective_improvement_fraction": math.nan,
            "hysteresis_design_condition_number": math.nan,
            "hysteresis_boundary": False,
            "joint_tau_ms": math.nan,
            "joint_az_half_offset_arcsec": math.nan,
            "joint_el_half_offset_arcsec": math.nan,
            "joint_objective_improvement_fraction": math.nan,
            "joint_design_condition_number": math.nan,
            "joint_boundary": False,
            "joint_minus_one_cadence_ms": math.nan,
            "joint_nearest_cadence_index": 0,
            "joint_nearest_cadence_residual_ms": math.nan,
        }
        try:
            fitted = centroids.fit_centroid_models(rows, protocol, threshold)
        except centroids.EventCentroidError as error:
            base["status_detail"] = str(error)
            result.append(base)
            continue
        models = fitted["models"]
        if set(models) != set(MODEL_NAMES):
            raise NetworkAuditError(
                f"ObsNum {obsnum} network {network} model set changed"
            )
        constant = models["constant"]
        lag = models["lag"]
        hysteresis = models["hysteresis"]
        joint = models["joint"]
        lag_tau = float(lag["tau_ms"])
        joint_tau = float(joint["tau_ms"])
        base.update({
            "status": "success",
            "status_detail": "",
            "common_robust_scale_arcsec": float(
                fitted["common_robust_scale_arcsec"]
            ),
            "constant_objective_arcsec2": float(constant["objective"]),
            "lag_tau_ms": lag_tau,
            "lag_objective_improvement_fraction": improvement(constant, lag),
            "lag_design_condition_number": float(
                lag["design_condition_number"]
            ),
            "lag_boundary": bool(lag["boundary"]),
            "lag_minus_pooled_ms": float(lag_tau - pooled_lag_tau_ms),
            "hysteresis_az_half_offset_arcsec": float(
                hysteresis["parameters"]["h_az_arcsec"]
            ),
            "hysteresis_el_half_offset_arcsec": float(
                hysteresis["parameters"]["h_el_arcsec"]
            ),
            "hysteresis_objective_improvement_fraction": improvement(
                constant, hysteresis
            ),
            "hysteresis_design_condition_number": float(
                hysteresis["design_condition_number"]
            ),
            "hysteresis_boundary": bool(hysteresis["boundary"]),
            "joint_tau_ms": joint_tau,
            "joint_az_half_offset_arcsec": float(
                joint["parameters"]["h_az_arcsec"]
            ),
            "joint_el_half_offset_arcsec": float(
                joint["parameters"]["h_el_arcsec"]
            ),
            "joint_objective_improvement_fraction": improvement(
                constant, joint
            ),
            "joint_design_condition_number": float(
                joint["design_condition_number"]
            ),
            "joint_boundary": bool(joint["boundary"]),
        })
        base.update(cadence_fields("lag", lag_tau, cadence_ms))
        base.update(cadence_fields("joint", joint_tau, cadence_ms))
        result.append(base)
    return result


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), value))


def observation_rows(
    campaign_rows: Table,
    network_rows: list[dict[str, Any]],
    cadence_ms: float,
) -> list[dict[str, Any]]:
    result = []
    for source in campaign_rows:
        row = {name: source[name] for name in campaign_rows.colnames}
        if str(row["status"]) == "complete":
            row.update(cadence_fields(
                "lag", float(row["lag_tau_ms"]), cadence_ms
            ))
            row.update(cadence_fields(
                "joint", float(row["joint_tau_ms"]), cadence_ms
            ))
        obsnum = int(row["obsnum"])
        selected = [
            value for value in network_rows
            if int(value["obsnum"]) == obsnum and value["status"] == "success"
        ]
        lag = [float(value["lag_tau_ms"]) for value in selected]
        joint = [float(value["joint_tau_ms"]) for value in selected]
        row.update({
            "successful_network_count": len(selected),
            "network_lag_tau_p16_ms": percentile(lag, 16.0) if lag else math.nan,
            "network_lag_tau_median_ms": (
                percentile(lag, 50.0) if lag else math.nan
            ),
            "network_lag_tau_p84_ms": percentile(lag, 84.0) if lag else math.nan,
            "network_lag_tau_range_ms": (
                float(max(lag) - min(lag)) if lag else math.nan
            ),
            "network_lag_minus_one_cadence_median_ms": (
                percentile([value - cadence_ms for value in lag], 50.0)
                if lag else math.nan
            ),
            "network_lag_nearest_plus_one_count": sum(
                cadence_index(value, cadence_ms) == 1 for value in lag
            ),
            "network_joint_tau_p16_ms": (
                percentile(joint, 16.0) if joint else math.nan
            ),
            "network_joint_tau_median_ms": (
                percentile(joint, 50.0) if joint else math.nan
            ),
            "network_joint_tau_p84_ms": (
                percentile(joint, 84.0) if joint else math.nan
            ),
        })
        result.append(row)
    return result


def network_summary_rows(
    rows: list[dict[str, Any]], cadence_ms: float,
) -> list[dict[str, Any]]:
    result = []
    networks = sorted({int(row["network"]) for row in rows})
    for network in networks:
        selected = [
            row for row in rows
            if int(row["network"]) == network and row["status"] == "success"
        ]
        lag = [float(row["lag_tau_ms"]) for row in selected]
        joint = [float(row["joint_tau_ms"]) for row in selected]
        result.append({
            "network": network,
            "array": selected[0]["array"] if selected else "unknown",
            "successful_observation_count": len(selected),
            "lag_tau_p16_ms": percentile(lag, 16.0) if lag else math.nan,
            "lag_tau_median_ms": percentile(lag, 50.0) if lag else math.nan,
            "lag_tau_p84_ms": percentile(lag, 84.0) if lag else math.nan,
            "lag_minus_one_cadence_median_ms": (
                percentile([value - cadence_ms for value in lag], 50.0)
                if lag else math.nan
            ),
            "lag_nearest_plus_one_fraction": (
                float(np.mean([
                    cadence_index(value, cadence_ms) == 1 for value in lag
                ])) if lag else math.nan
            ),
            "joint_tau_p16_ms": percentile(joint, 16.0) if joint else math.nan,
            "joint_tau_median_ms": percentile(joint, 50.0) if joint else math.nan,
            "joint_tau_p84_ms": percentile(joint, 84.0) if joint else math.nan,
            "qualified_event_count_median": (
                percentile([
                    float(row["qualified_event_count"]) for row in selected
                ], 50.0) if selected else math.nan
            ),
            "qualified_detector_count_median": (
                percentile([
                    float(row["qualified_detector_count"]) for row in selected
                ], 50.0) if selected else math.nan
            ),
            "lag_design_condition_median": (
                percentile([
                    float(row["lag_design_condition_number"])
                    for row in selected
                ], 50.0) if selected else math.nan
            ),
            "joint_design_condition_median": (
                percentile([
                    float(row["joint_design_condition_number"])
                    for row in selected
                ], 50.0) if selected else math.nan
            ),
        })
    return result


def authenticate_campaign_audit(path: Path) -> tuple[dict[str, Any], Table]:
    analysis.verify_sha256s(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != (
        "sci-align-001-lissajous-event-centroid-fit-campaign-audit-v1"
    ):
        raise NetworkAuditError("unsupported campaign audit schema")
    rows = Table.read(path / "event_fit_campaign_status.ecsv")
    return manifest, rows


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise NetworkAuditError(f"output already exists: {output}")
    cadence_ms = float(args.cadence_ms)
    if not math.isfinite(cadence_ms) or cadence_ms <= 0.0:
        raise NetworkAuditError("cadence must be positive finite")
    centroid_protocol_path = args.centroid_protocol.resolve()
    centroid_protocol = json.loads(centroid_protocol_path.read_text())
    if centroid_protocol.get("schema") != (
        "sci-align-001-lissajous-event-centroid-protocol-v1"
    ):
        raise NetworkAuditError("unsupported centroid protocol schema")
    centroid_protocol_sha = analysis.sha256_file(centroid_protocol_path)
    campaign_manifest, campaign_rows = authenticate_campaign_audit(
        args.campaign_audit.resolve()
    )
    if any(str(row["status"]) != "complete" for row in campaign_rows):
        raise NetworkAuditError("campaign audit is not complete")

    network_rows: list[dict[str, Any]] = []
    fit_root = args.fit_root.resolve()
    for source in campaign_rows:
        obsnum = int(source["obsnum"])
        directory = fit_root / f"o{obsnum}"
        analysis.verify_sha256s(directory, "FIT_GATE_SHA256SUMS")
        gate_path = directory / "fit_gate.json"
        gate = json.loads(gate_path.read_text())
        if gate.get("schema") != (
            "sci-align-001-lissajous-event-centroid-fit-gate-v1"
        ):
            raise NetworkAuditError(f"unsupported fit gate for ObsNum {obsnum}")
        if analysis.sha256_file(gate_path) != str(source["fit_gate_sha256"]):
            raise NetworkAuditError(f"fit identity changed for ObsNum {obsnum}")
        if gate["input"]["centroid_protocol_sha256"] != centroid_protocol_sha:
            raise NetworkAuditError(
                f"centroid protocol changed for ObsNum {obsnum}"
            )
        centroid_rows = Table.read(directory / "event_centroids.ecsv")
        derived = fit_network_rows(
            centroid_rows,
            centroid_protocol,
            obsnum,
            float(source["lag_tau_ms"]),
            cadence_ms,
        )
        for row in derived:
            row["input_fit_gate_sha256"] = str(source["fit_gate_sha256"])
            row["input_fit_gate_sha256s_sha256"] = str(
                source["fit_gate_sha256s_sha256"]
            )
        network_rows.extend(derived)

    observation = observation_rows(campaign_rows, network_rows, cadence_ms)
    summary = network_summary_rows(network_rows, cadence_ms)
    successful = sum(row["status"] == "success" for row in network_rows)
    if args.require_complete and successful != len(network_rows):
        failed = [
            f"{row['obsnum']}:nw{row['network']}:{row['status_detail']}"
            for row in network_rows if row["status"] != "success"
        ]
        raise NetworkAuditError(
            "one or more network fits are incomplete: " + ", ".join(failed)
        )
    output.mkdir(parents=True)
    Table(rows=observation).write(
        output / "observation_results.ecsv", format="ascii.ecsv"
    )
    Table(rows=network_rows).write(
        output / "observation_network_results.ecsv", format="ascii.ecsv"
    )
    Table(rows=summary).write(
        output / "network_summary.ecsv", format="ascii.ecsv"
    )
    manifest = {
        "schema": (
            "sci-align-001-lissajous-event-centroid-network-audit-v1"
        ),
        "cadence_ms": cadence_ms,
        "campaign_audit_path": str(args.campaign_audit.resolve()),
        "campaign_audit_manifest_sha256": analysis.sha256_file(
            args.campaign_audit.resolve() / "manifest.json"
        ),
        "campaign_audit_sha256s_sha256": analysis.sha256_file(
            args.campaign_audit.resolve() / "SHA256SUMS"
        ),
        "campaign_selection_sha256": campaign_manifest["selection_sha256"],
        "campaign_protocol_sha256": campaign_manifest["protocol_sha256"],
        "centroid_protocol_path": str(centroid_protocol_path),
        "centroid_protocol_sha256": centroid_protocol_sha,
        "fit_root": str(fit_root),
        "observation_count": len(observation),
        "network_row_count": len(network_rows),
        "successful_network_row_count": successful,
        "network_count": len(summary),
        "estimator": (
            "the frozen primary event qualification is retained; each network "
            "is then fit independently with the frozen four-model robust "
            "centroid estimator and network-specific constant-model MAD scale"
        ),
        "interpretation_boundary": {
            "permitted": (
                "descriptive pooled and per-network point estimates, cadence "
                "residuals, support, objective improvements, and conditioning"
            ),
            "prohibited": (
                "formal significance, universal correction, causal origin, or "
                "native detector-frame phase inference"
            ),
        },
        "native_phase_status": (
            "unavailable in retained event-centroid products; a raw-counter "
            "phase join is a separate diagnostic"
        ),
        "uncertainty_status": "deferred_during_baseline_algorithm_validation",
    }
    analysis.write_json(output / "manifest.json", manifest)
    names = [
        "manifest.json", "network_summary.ecsv",
        "observation_network_results.ecsv", "observation_results.ecsv",
    ]
    analysis.write_checksums(output, names)
    analysis.verify_sha256s(output)
    print(
        f"network audit complete: observations={len(observation)} "
        f"network_rows={len(network_rows)} successful={successful} "
        f"output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--centroid-protocol", type=Path, required=True)
    result.add_argument("--campaign-audit", type=Path, required=True)
    result.add_argument("--fit-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cadence-ms", type=float, default=8.192)
    result.add_argument("--require-complete", action="store_true")
    return result


def main() -> int:
    try:
        run(parser().parse_args())
    except (
        NetworkAuditError,
        analysis.ContractError,
        centroids.EventCentroidError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
