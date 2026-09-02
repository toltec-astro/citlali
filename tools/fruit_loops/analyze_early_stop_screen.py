#!/usr/bin/env python3
"""Analyze the prospective SCI-FRUIT EL-F2 early-stop screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from tools.fruit_loops.analyze_compact_relaxation_screen import (
    ARRAYS,
    analyze_pair,
    select_row,
    write_csv,
)


def require_iteration_set(
    iterations: list[int],
    *,
    start_iteration: int,
    stop_iteration_exclusive: int,
    context: str,
) -> None:
    expected = list(range(start_iteration, stop_iteration_exclusive))
    if sorted(iterations) != expected:
        raise ValueError(
            f"{context} iterations differ: expected={expected} "
            f"actual={sorted(iterations)}"
        )


def read_execution(path: Path, manifest: dict) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4:
        raise ValueError(f"expected four primary execution rows; found {len(rows)}")

    expected: dict[tuple[float, bool], int] = {}
    for spec in manifest["methods"].values():
        alpha = float(spec["alpha"])
        count = int(spec["stop_iteration_exclusive"]) - int(
            manifest["trajectory_start_iteration"]
        )
        expected[(alpha, False)] = count
        expected[(alpha, True)] = count

    observed: set[tuple[float, bool]] = set()
    for row in rows:
        key = (float(row["alpha"]), row["injection"].lower() == "true")
        if key not in expected:
            raise ValueError(f"unexpected execution trajectory: {key}")
        if key in observed:
            raise ValueError(f"duplicate execution trajectory: {key}")
        observed.add(key)
        if row["status"] != "completed":
            raise ValueError(f"trajectory did not complete: {row['trajectory']}")
        if int(row["completed_iterations"]) != expected[key]:
            raise ValueError(
                f"trajectory iteration count differs: {row['trajectory']}"
            )
        if int(row["error_or_critical_messages"]) != 0:
            raise ValueError(
                f"trajectory has error/critical messages: {row['trajectory']}"
            )
        wall = float(row["wall_seconds"])
        if wall <= 0.0:
            raise ValueError(f"invalid wall time: {row['trajectory']}")
    if observed != set(expected):
        raise ValueError("primary execution trajectory set is incomplete")
    return rows


def pair_mean_wall_seconds(rows: list[dict], alpha: float) -> float:
    values = [
        float(row["wall_seconds"])
        for row in rows
        if float(row["alpha"]) == alpha
    ]
    if len(values) != 2:
        raise ValueError(f"expected two wall times for alpha={alpha}")
    return sum(values) / len(values)


def classify(rows: list[dict], execution_rows: list[dict], manifest: dict) -> dict:
    screen = manifest["screen"]
    reference = manifest["methods"]["reference"]
    candidate = manifest["methods"]["candidate"]
    reference_alpha = float(reference["alpha"])
    candidate_alpha = float(candidate["alpha"])
    reference_iteration = int(reference["terminal_iteration"])
    candidate_iteration = int(candidate["terminal_iteration"])
    max_error_degradation = float(
        screen["max_absolute_recovery_error_degradation"]
    )
    max_width_error = float(screen["max_width_fractional_error"])
    max_centroid = float(screen["max_centroid_error_arcsec"])
    max_residual_ratio = float(screen["max_residual_ratio_to_reference"])

    array_checks: dict[str, dict[str, bool]] = {}
    for array in ARRAYS:
        ref = select_row(rows, reference_alpha, reference_iteration, array)
        cand = select_row(rows, candidate_alpha, candidate_iteration, array)
        array_checks[array] = {
            "terminal_recovery_error_within_allowance": abs(
                cand["kernel_normalized_central_recovery"] - 1.0
            )
            <= abs(ref["kernel_normalized_central_recovery"] - 1.0)
            + max_error_degradation,
            "terminal_major_width_within_limit": abs(
                cand["major_fwhm_over_kernel"] - 1.0
            )
            <= max_width_error,
            "terminal_minor_width_within_limit": abs(
                cand["minor_fwhm_over_kernel"] - 1.0
            )
            <= max_width_error,
            "terminal_centroid_within_limit": (
                cand["centroid_error_arcsec"] <= max_centroid
            ),
            "terminal_annular_residual_within_limit": (
                cand["annular_residual_over_truth"]
                <= max_residual_ratio * ref["annular_residual_over_truth"]
            ),
            "terminal_kernel_residual_within_limit": (
                cand["kernel_residual_relative_rms"]
                <= max_residual_ratio * ref["kernel_residual_relative_rms"]
            ),
        }

    scientific_pass = all(
        passed
        for checks in array_checks.values()
        for passed in checks.values()
    )
    reference_wall = pair_mean_wall_seconds(execution_rows, reference_alpha)
    candidate_wall = pair_mean_wall_seconds(execution_rows, candidate_alpha)
    improvement = (reference_wall - candidate_wall) / reference_wall
    performance_pass = improvement >= float(
        screen["minimum_pair_mean_wall_time_improvement_fraction"]
    )

    if not scientific_pass:
        classification = "does_not_replicate"
    elif not performance_pass:
        classification = "scientifically_replicates_but_misses_performance_target"
    else:
        classification = "promising_early_stop_result"

    return {
        "test_id": manifest["test_id"],
        "valid_primary_screen": True,
        "reference": {
            "alpha": reference_alpha,
            "terminal_iteration": reference_iteration,
            "pair_mean_wall_seconds": reference_wall,
        },
        "candidate": {
            "alpha": candidate_alpha,
            "terminal_iteration": candidate_iteration,
            "pair_mean_wall_seconds": candidate_wall,
            "wall_time_improvement_fraction": improvement,
        },
        "array_checks": array_checks,
        "scientific_protections_pass": scientific_pass,
        "performance_target_pass": performance_pass,
        "classification": classification,
        "restart_required": classification == "promising_early_stop_result",
    }


def write_report(path: Path, rows: list[dict], result: dict) -> None:
    lines = [
        "# SCI-FRUIT EL-F2 early-stop screen result",
        "",
        f"Primary classification: **{result['classification']}**",
        "",
        "This is development evidence only and is not a method qualification.",
        "",
        "## Terminal comparison",
        "",
        "| Array | Reference recovery | Candidate recovery | "
        "Reference annular residual | Candidate annular residual | Checks |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    ref = result["reference"]
    cand = result["candidate"]
    for array in ARRAYS:
        ref_row = select_row(rows, ref["alpha"], ref["terminal_iteration"], array)
        cand_row = select_row(
            rows, cand["alpha"], cand["terminal_iteration"], array
        )
        failed = [
            name
            for name, passed in result["array_checks"][array].items()
            if not passed
        ]
        lines.append(
            f"| {array} | "
            f"{ref_row['kernel_normalized_central_recovery']:.6f} | "
            f"{cand_row['kernel_normalized_central_recovery']:.6f} | "
            f"{ref_row['annular_residual_over_truth']:.8g} | "
            f"{cand_row['annular_residual_over_truth']:.8g} | "
            + ("PASS" if not failed else "FAIL — " + ", ".join(failed))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Reference pair-mean wall time: {ref['pair_mean_wall_seconds']:.3f} s",
            f"- Candidate pair-mean wall time: {cand['pair_mean_wall_seconds']:.3f} s",
            "- Candidate wall-time improvement: "
            f"{100.0 * cand['wall_time_improvement_fraction']:.3f}%",
            f"- Ten-percent target: {'PASS' if result['performance_target_pass'] else 'FAIL'}",
            "",
            "## Follow-up",
            "",
            "- Exact restart replay required: "
            + ("yes" if result["restart_required"] else "no"),
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    start = int(manifest["trajectory_start_iteration"])
    rows: list[dict] = []
    for role, spec in manifest["methods"].items():
        stop = int(spec["stop_iteration_exclusive"])
        require_iteration_set(
            list(range(start, stop)),
            start_iteration=start,
            stop_iteration_exclusive=stop,
            context=role,
        )
        rows.extend(
            analyze_pair(
                alpha_label=f"{float(spec['alpha']):.2f}",
                alpha=float(spec["alpha"]),
                control_root=Path(spec["control"]),
                injected_root=Path(spec["injected"]),
                manifest=manifest,
                stop_iteration_exclusive=stop,
            )
        )
    rows.sort(key=lambda row: (row["alpha"], row["iteration"], row["array"]))
    execution_rows = read_execution(args.execution, manifest)
    write_csv(args.output, rows)
    result = classify(rows, execution_rows, manifest)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.result.with_suffix(".md"), rows, result)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote screen result to {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
