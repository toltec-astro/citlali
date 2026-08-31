#!/usr/bin/env python3
"""Aggregate WP-7 D2 PSD/line artifacts without choosing an envelope or factor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.wp7.rtc_filter_psd_line_evidence import (
    NATIVE_TIMING_DOMAIN,
    RESULT_SCHEMA,
    sha256_file,
)
from tools.blank_sky.rtc_line_audit import _contiguous_runs


CORPUS_SCHEMA = "citlali-wp7-rtc-filter-psd-line-corpus-v1"
AGGREGATE_ROW_ORDER = ("median", "q90", "q95", "q99", "maximum")


def _load_array(report_path: Path, declaration: object, label: str) -> np.ndarray:
    if not isinstance(declaration, dict):
        raise RuntimeError(f"{label} declaration is missing")
    filename = declaration.get("file")
    expected_hash = declaration.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_hash, str):
        raise RuntimeError(f"{label} declaration is malformed")
    path = (report_path.parent / filename).resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"{label} artifact is absent or changed: {path}")
    value = np.load(path, allow_pickle=False)
    if (
        list(value.shape) != declaration.get("shape")
        or str(value.dtype) != declaration.get("dtype")
    ):
        raise RuntimeError(f"{label} artifact shape or type changed")
    return value


def load_report(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open() as stream:
        report = json.load(stream)
    if report.get("schema") != RESULT_SCHEMA:
        raise RuntimeError(f"unsupported D2 evidence schema: {path}")
    report["_path"] = path
    report["_sha256"] = sha256_file(path)
    return report


def _group_key(report: dict[str, Any]) -> tuple[str, str, str]:
    identity = report["identity"]
    return (
        str(identity["array"]),
        str(identity["cadence_domain_id"]),
        str(identity["signal_units"]),
    )


def _safe_group_name(key: tuple[str, str, str]) -> str:
    raw = "__".join(key)
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in raw
    )


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, value, allow_pickle=False)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _aggregate(values: np.ndarray) -> np.ndarray:
    result = np.full((len(AGGREGATE_ROW_ORDER), values.shape[1]), np.nan)
    populated = np.any(np.isfinite(values), axis=0)
    if not np.any(populated):
        return result
    selected = values[:, populated]
    result[:, populated] = np.vstack(
        (
            np.nanmedian(selected, axis=0),
            np.nanquantile(selected, 0.90, axis=0),
            np.nanquantile(selected, 0.95, axis=0),
            np.nanquantile(selected, 0.99, axis=0),
            np.nanmax(selected, axis=0),
        )
    )
    return result


def _integrated_power(frequency: np.ndarray, psd: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    runs = [bounds for bounds in _contiguous_runs(eligible) if bounds[1] - bounds[0] >= 2]
    if not runs:
        return np.full(psd.shape[0], np.nan)
    return np.asarray(
        [
            sum(
                float(np.trapezoid(row[first:last], frequency[first:last]))
                for first, last in runs
            )
            for row in psd
        ],
        dtype=np.float64,
    )


def _build_residual_group(
    key: tuple[str, str, str],
    reports: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    frequency: np.ndarray | None = None
    all_psds: list[np.ndarray] = []
    all_eligible_psds: list[np.ndarray] = []
    per_case: list[dict[str, Any]] = []
    for report in reports:
        path = report["_path"]
        artifacts = report["artifacts"]
        current_frequency = _load_array(path, artifacts["frequency_hz"], "frequency_hz")
        psd = _load_array(path, artifacts["psd"], "psd").astype(np.float64)
        accepted = _load_array(
            path, artifacts["detector_accepted"], "detector_accepted"
        ).astype(bool)
        eligible = _load_array(
            path,
            artifacts["broadband_frequency_eligible"],
            "broadband_frequency_eligible",
        ).astype(bool)
        if frequency is None:
            frequency = current_frequency
        elif not np.array_equal(frequency, current_frequency):
            raise RuntimeError(f"frequency grids differ within cadence group {key}")
        accepted_psd = psd[accepted]
        eligible_psd = accepted_psd.copy()
        eligible_psd[:, ~eligible] = np.nan
        all_psds.append(accepted_psd)
        all_eligible_psds.append(eligible_psd)
        power = _integrated_power(current_frequency, accepted_psd, eligible)
        finite_power = power[np.isfinite(power)]
        identity = report["identity"]
        per_case.append(
            {
                "case_id": identity["case_id"],
                "route_family": identity["route_family"],
                "observation": identity["observation"],
                "network": identity["network"],
                "detector_count": int(accepted_psd.shape[0]),
                "line_mask_policy_id": report["line_mask"]["policy_id"],
                "eligible_frequency_bin_count": int(np.count_nonzero(eligible)),
                "integrated_broadband_power_median": (
                    float(np.median(finite_power)) if finite_power.size else None
                ),
                "integrated_broadband_power_q95": (
                    float(np.quantile(finite_power, 0.95)) if finite_power.size else None
                ),
                "integrated_broadband_power_maximum": (
                    float(np.max(finite_power)) if finite_power.size else None
                ),
                "disposition": report["disposition"],
            }
        )
    assert frequency is not None
    values = np.concatenate(all_psds, axis=0)
    eligible_values = np.concatenate(all_eligible_psds, axis=0)
    aggregate = _aggregate(values)
    broadband_aggregate = _aggregate(eligible_values)
    contribution_count = np.sum(np.isfinite(eligible_values), axis=0).astype(np.int64)
    q95 = broadband_aggregate[2]
    maximum = broadband_aggregate[4]
    ratio = np.divide(
        maximum,
        q95,
        out=np.full(maximum.shape, np.nan),
        where=np.isfinite(q95) & (q95 > 0),
    )
    finite_ratio = ratio[np.isfinite(ratio)]
    prefix = _safe_group_name(key)
    artifacts = {
        "frequency_hz": _write_array(output_dir / f"{prefix}__frequency_hz.npy", frequency),
        "aggregate_psd": _write_array(output_dir / f"{prefix}__aggregate_psd.npy", aggregate),
        "broadband_aggregate_psd": _write_array(
            output_dir / f"{prefix}__broadband_aggregate_psd.npy", broadband_aggregate
        ),
        "broadband_contributing_detector_count": _write_array(
            output_dir / f"{prefix}__broadband_contributing_detector_count.npy",
            contribution_count,
        ),
    }
    candidate = all(
        report["disposition"] == "residual_psd_envelope_candidate"
        for report in reports
    )
    return {
        "array": key[0],
        "cadence_domain_id": key[1],
        "signal_units": key[2],
        "status": (
            "aggregation_sensitivity_candidate"
            if candidate
            else "aggregation_measurement_pending_input_line_mask"
        ),
        "artifact_count": len(reports),
        "detector_count": int(values.shape[0]),
        "route_families": sorted({str(report["identity"]["route_family"]) for report in reports}),
        "aggregate_row_order": list(AGGREGATE_ROW_ORDER),
        "maximum_to_q95_ratio": {
            "median_over_frequency": (
                float(np.median(finite_ratio)) if finite_ratio.size else None
            ),
            "q90_over_frequency": (
                float(np.quantile(finite_ratio, 0.90)) if finite_ratio.size else None
            ),
            "maximum_over_frequency": (
                float(np.max(finite_ratio)) if finite_ratio.size else None
            ),
        },
        "per_case": sorted(
            per_case,
            key=lambda row: (
                str(row["route_family"]),
                int(row["observation"]),
                int(row["network"]),
            ),
        ),
        "artifacts": artifacts,
    }


def _line_gate_summary(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for factor in range(1, 257):
        rows = []
        for report in reports:
            summaries = report["line_inventory"]["factor_summary"]
            if summaries:
                rows.append(summaries[factor - 1])
        blockers = [
            report["identity"]["case_id"]
            for report in reports
            if report["line_inventory"]["factor_summary"]
            and report["line_inventory"]["factor_summary"][factor - 1]["line_gate"]
            == "withhold"
        ]
        result.append(
            {
                "factor": factor,
                "prefilter_artifact_count": len(rows),
                "foldable_line_count_sum": int(
                    sum(row["foldable_line_count"] for row in rows)
                ),
                "unprotected_foldable_line_count_sum": int(
                    sum(row["unprotected_foldable_line_count"] for row in rows)
                ),
                "blocking_case_ids": sorted(set(blockers)),
                "line_gate": "withhold" if blockers else "not_blocked_by_available_inventory",
            }
        )
    return result


def build_corpus(report_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    reports = [load_report(path) for path in report_paths]
    if not reports:
        raise RuntimeError("no D2 evidence artifacts were supplied")
    identities = [
        (
            report["identity"]["case_id"],
            report["identity"]["network"],
            report["identity"]["stream_stage"],
        )
        for report in reports
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError("D2 corpus repeats a case/network/stage identity")

    native_reports = [
        report
        for report in reports
        if report["identity"]["timing_domain"] == NATIVE_TIMING_DOMAIN
    ]
    residual_reports = [
        report
        for report in native_reports
        if report["identity"]["stream_stage"] == "native_post_cleaning_residual"
    ]
    prefilter_reports = [
        report
        for report in native_reports
        if report["identity"]["stream_stage"] == "native_prefilter"
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for report in residual_reports:
        groups.setdefault(_group_key(report), []).append(report)
    output_dir.mkdir(parents=True)
    group_results = [
        _build_residual_group(key, sorted(value, key=lambda row: row["_sha256"]), output_dir)
        for key, value in sorted(groups.items())
    ]

    route_families = sorted(
        {str(report["identity"]["route_family"]) for report in native_reports}
    )
    required_routes = {"beammap", "science", "oof"}
    missing_routes = sorted(required_routes - set(route_families))
    pending_groups = [
        group
        for group in group_results
        if group["status"] != "aggregation_sensitivity_candidate"
    ]
    if missing_routes:
        disposition = "incomplete_required_route_family_evidence"
    elif not residual_reports or not prefilter_reports:
        disposition = "incomplete_residual_or_prefilter_evidence"
    elif pending_groups:
        disposition = "measurement_complete_owner_envelope_input_pending"
    else:
        disposition = "measurement_complete_owner_envelope_choice_not_selected"

    result: dict[str, Any] = {
        "schema": CORPUS_SCHEMA,
        "status": "evidence_only_no_envelope_factor_or_filter_selected",
        "disposition": disposition,
        "input_artifacts": [
            {
                "sha256": report["_sha256"],
                "case_id": report["identity"]["case_id"],
                "route_family": report["identity"]["route_family"],
                "observation": report["identity"]["observation"],
                "network": report["identity"]["network"],
                "stream_stage": report["identity"]["stream_stage"],
                "timing_domain": report["identity"]["timing_domain"],
                "disposition": report["disposition"],
            }
            for report in sorted(reports, key=lambda row: row["_sha256"])
        ],
        "required_route_families": sorted(required_routes),
        "present_native_route_families": route_families,
        "missing_native_route_families": missing_routes,
        "legacy_discovery_artifact_count": len(reports) - len(native_reports),
        "residual_groups": group_results,
        "prefilter_line_gate_by_factor": _line_gate_summary(prefilter_reports),
        "limitations": [
            (
                "The five aggregation rows are alternatives for sensitivity "
                "study, not a selected envelope."
            ),
            (
                "A factor is withheld by available unprotected foldable-line "
                "evidence but is never selected here."
            ),
            (
                "Legacy rectangular timing artifacts remain discovery-only "
                "and never enter native evidence groups."
            ),
        ],
    }
    (output_dir / "corpus.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = build_corpus(args.evidence, args.output_dir.resolve())
    print(json.dumps({"disposition": result["disposition"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
