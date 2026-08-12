#!/usr/bin/env python3
"""Authenticate and summarize an event-support Lissajous fit campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from astropy.table import Table

import analyze_sci_align_001_lissajous_timestream as analysis


class CampaignAuditError(RuntimeError):
    """An event campaign result violates its identity contract."""


def parameter(fit: dict[str, Any], name: str) -> float:
    return float(fit.get("parameters", {}).get(name, math.nan))


def audit_row(
    selected: dict[str, Any],
    fit_root: Path,
    review_root: Path,
) -> dict[str, Any]:
    obsnum = int(selected["pointing_obsnum"])
    fit_directory = fit_root / f"o{obsnum}"
    review_directory = review_root / f"o{obsnum}"
    row: dict[str, Any] = {
        "obsnum": obsnum,
        "beammap_obsnum": int(selected["beammap_obsnum"]),
        "snr_a1100": float(selected["snr_a1100"]),
        "mean_elevation_deg": float(selected["mean_elevation_deg"]),
        "status": "missing",
    }
    if not (fit_directory / "fit_gate.json").is_file():
        return row
    analysis.verify_sha256s(fit_directory, "FIT_GATE_SHA256SUMS")
    gate = json.loads((fit_directory / "fit_gate.json").read_text())
    if gate.get("schema") != "sci-align-001-lissajous-event-fit-gate-v1":
        raise CampaignAuditError(f"unsupported event gate for ObsNum {obsnum}")
    identity = gate["input"]
    if identity["ptc_sha256"] != selected["ptc_sha256"]:
        raise CampaignAuditError(f"PTC identity changed for ObsNum {obsnum}")
    if identity["ppt_sha256"] != selected["ppt_sha256"]:
        raise CampaignAuditError(f"PPT identity changed for ObsNum {obsnum}")
    if not (review_directory / "manifest.json").is_file():
        row["status"] = "fit_complete_review_missing"
        return row
    analysis.verify_sha256s(review_directory)
    review = json.loads((review_directory / "manifest.json").read_text())
    if review.get("schema") != "sci-align-001-lissajous-event-fit-review-v1":
        raise CampaignAuditError(f"unsupported event review for ObsNum {obsnum}")
    fit_sha = analysis.sha256_file(fit_directory / "fit_gate.json")
    if review["fit_gate_sha256"] != fit_sha:
        raise CampaignAuditError(f"review fit identity changed for ObsNum {obsnum}")
    fits = gate["point_model_results"]
    census = gate["crossing_support"]["census"]
    constant = fits["constant"]
    lag = fits["lag"]
    hysteresis = fits["hysteresis"]
    joint = fits["joint"]
    base_objective = float(constant["objective"])

    def improvement(fit: dict[str, Any]) -> float:
        return (base_objective - float(fit["objective"])) / base_objective

    row.update({
        "status": "complete",
        "geometric_event_count": int(census["geometric_event_count"]),
        "complete_event_count": int(census["accepted_event_count"]),
        "retained_detector_scan_count": int(
            census["retained_detector_scan_count"]
        ),
        "retained_unique_detector_count": int(
            census["retained_unique_detector_count"]
        ),
        "scored_value_count": int(census["retained_scored_sample_count"]),
        "lag_tau_ms": float(lag["tau_ms"]),
        "lag_objective_improvement_fraction": improvement(lag),
        "hysteresis_az_half_offset_arcsec": parameter(
            hysteresis, "h_az_arcsec"
        ),
        "hysteresis_el_half_offset_arcsec": parameter(
            hysteresis, "h_el_arcsec"
        ),
        "hysteresis_objective_improvement_fraction": improvement(hysteresis),
        "joint_tau_ms": float(joint["tau_ms"]),
        "joint_az_half_offset_arcsec": parameter(joint, "h_az_arcsec"),
        "joint_el_half_offset_arcsec": parameter(joint, "h_el_arcsec"),
        "joint_objective_improvement_fraction": improvement(joint),
        "fit_gate_sha256": fit_sha,
        "fit_gate_sha256s_sha256": analysis.sha256_file(
            fit_directory / "FIT_GATE_SHA256SUMS"
        ),
        "review_manifest_sha256": analysis.sha256_file(
            review_directory / "manifest.json"
        ),
        "review_sha256s_sha256": analysis.sha256_file(
            review_directory / "SHA256SUMS"
        ),
    })
    return row


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise CampaignAuditError(f"audit output already exists: {output}")
    protocol = analysis.load_protocol(args.protocol.resolve())
    selection = analysis.load_selection(
        args.selection.resolve(),
        protocol["input_authority"]["selection_manifest_sha256"],
    )
    rows = [
        audit_row(row, args.fit_root.resolve(), args.review_root.resolve())
        for row in selection["rows"]
    ]
    complete = sum(row["status"] == "complete" for row in rows)
    if args.require_complete and complete != len(rows):
        missing = [
            f"{row['obsnum']}:{row['status']}"
            for row in rows if row["status"] != "complete"
        ]
        raise CampaignAuditError(
            "event campaign is incomplete: " + ", ".join(missing)
        )
    output.mkdir(parents=True)
    Table(rows=rows).write(
        output / "event_fit_campaign_status.ecsv", format="ascii.ecsv"
    )
    manifest = {
        "schema": "sci-align-001-lissajous-event-fit-campaign-audit-v1",
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": analysis.sha256_file(args.selection.resolve()),
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": analysis.sha256_file(args.protocol.resolve()),
        "fit_root": str(args.fit_root.resolve()),
        "review_root": str(args.review_root.resolve()),
        "observation_count": len(rows),
        "complete_observation_count": complete,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "uncertainty_status": "deferred_during_baseline_algorithm_validation",
        "interpretation": (
            "authenticated point estimates and structural review only; no "
            "formal significance, universal correction, or causal identification"
        ),
    }
    analysis.write_json(output / "manifest.json", manifest)
    analysis.write_checksums(
        output, ["event_fit_campaign_status.ecsv", "manifest.json"]
    )
    analysis.verify_sha256s(output)
    print(
        f"event campaign audit complete: observations={len(rows)} "
        f"complete={complete} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--fit-root", type=Path, required=True)
    result.add_argument("--review-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--require-complete", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (
        CampaignAuditError,
        analysis.ContractError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
