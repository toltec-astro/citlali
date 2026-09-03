#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F4 feedback-bypass screen."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from netCDF4 import Dataset

from tools.fruit_loops.analyze_compact_relaxation_screen import (
    IMAGE_EXTENSIONS,
    analyze_pair,
    iteration_dirs,
    select_row,
    trajectory_config,
)
from tools.fruit_loops.compare_injected_source_pair import ARRAYS, product_path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def require_exact_trajectory_images(
    expected_root: Path,
    actual_root: Path,
    obsnum: int,
    stop_iteration_exclusive: int,
) -> int:
    compared = 0
    for iteration in range(stop_iteration_exclusive):
        expected_redu = expected_root / f"redu{iteration:02d}"
        actual_redu = actual_root / f"redu{iteration:02d}"
        for array in ARRAYS:
            expected_path = product_path(expected_redu, obsnum, array)
            actual_path = product_path(actual_redu, obsnum, array)
            with fits.open(expected_path, memmap=True) as expected, fits.open(
                actual_path, memmap=True
            ) as actual:
                for extension in IMAGE_EXTENSIONS:
                    expected_data = expected[extension].data
                    actual_data = actual[extension].data
                    if (
                        expected_data.shape != actual_data.shape
                        or expected_data.dtype != actual_data.dtype
                        or expected_data.tobytes() != actual_data.tobytes()
                    ):
                        raise ValueError(
                            "default-off image is not bitwise compatible: "
                            f"obs={obsnum} iter={iteration} array={array} "
                            f"extension={extension}"
                        )
                    compared += 1
    return compared


def require_candidate_iteration_zero_exact(
    control_root: Path, injected_root: Path, obsnum: int
) -> int:
    return require_exact_trajectory_images(
        control_root, injected_root, obsnum, 1
    )


def read_reference_metrics(path: Path) -> dict[tuple[float, int, str], dict]:
    result: dict[tuple[float, int, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[(float(row["alpha"]), int(row["iteration"]), row["array"])] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"alpha", "iteration", "array"}
            }
    return result


def mapdiag_penalties(root: Path, terminal_iteration: int) -> list[dict]:
    path = root / f"redu{terminal_iteration:02d}" / (
        f"learning_iter_{terminal_iteration}.csv"
    )
    penalties: list[dict] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["record_type"] != "detector_penalty"
                or row["producer"] != "mapdiag:raw_obs"
                or row["reason"] != "map_pixel_outlier_detector_dominance"
            ):
                continue
            penalties.append(
                {
                    "iteration": int(row["iter"]),
                    "scan": int(row["scan"]),
                    "uid": int(row["uid"]),
                    "array": int(row["array"]),
                    "score": float(row["score"]),
                    "factor": float(row["factor"]),
                    "scan_local": row["scan_local"] == "1",
                }
            )
    return penalties


def compare_penalties(rows: list[dict]) -> list[dict]:
    """Classify exact additions, removals, retentions, and timing changes."""
    comparisons: list[dict] = []
    pair_keys = sorted({(row["obsnum"], row["variant"]) for row in rows})
    for obsnum, variant in pair_keys:
        methods: dict[str, dict[tuple[int, int, int, int], dict]] = {}
        for method in ("complete_map", "feedback_excluded"):
            methods[method] = {
                (
                    row["iteration"],
                    row["scan"],
                    row["uid"],
                    row["array"],
                ): row
                for row in rows
                if row["obsnum"] == obsnum
                and row["variant"] == variant
                and row["evidence_view"] == method
            }
        complete = methods["complete_map"]
        candidate = methods["feedback_excluded"]
        exact = set(complete) & set(candidate)

        def append(disposition: str, left: dict | None, right: dict | None) -> None:
            record = left if left is not None else right
            assert record is not None
            comparisons.append(
                {
                    "obsnum": obsnum,
                    "variant": variant,
                    "disposition": disposition,
                    "scan": record["scan"],
                    "uid": record["uid"],
                    "array": record["array"],
                    "complete_iteration": (
                        left["iteration"] if left is not None else ""
                    ),
                    "candidate_iteration": (
                        right["iteration"] if right is not None else ""
                    ),
                    "complete_score": left["score"] if left is not None else "",
                    "candidate_score": (
                        right["score"] if right is not None else ""
                    ),
                }
            )

        for key in sorted(exact):
            left = complete[key]
            right = candidate[key]
            disposition = (
                "retained"
                if left["score"] == right["score"]
                and left["factor"] == right["factor"]
                else "retained_value_changed"
            )
            append(disposition, left, right)

        complete_unmatched = [complete[key] for key in set(complete) - exact]
        candidate_unmatched = [candidate[key] for key in set(candidate) - exact]
        base_keys = sorted(
            {
                (row["scan"], row["uid"], row["array"])
                for row in complete_unmatched + candidate_unmatched
            }
        )
        for base_key in base_keys:
            left_rows = sorted(
                (
                    row
                    for row in complete_unmatched
                    if (row["scan"], row["uid"], row["array"]) == base_key
                ),
                key=lambda row: row["iteration"],
            )
            right_rows = sorted(
                (
                    row
                    for row in candidate_unmatched
                    if (row["scan"], row["uid"], row["array"]) == base_key
                ),
                key=lambda row: row["iteration"],
            )
            common_count = min(len(left_rows), len(right_rows))
            for index in range(common_count):
                append("timing_changed", left_rows[index], right_rows[index])
            for row in left_rows[common_count:]:
                append("removed", row, None)
            for row in right_rows[common_count:]:
                append("added", None, row)
    return comparisons


def require_candidate_provenance(
    root: Path,
    obsnum: int,
    terminal_iteration: int,
    expected_enabled: bool,
) -> None:
    config = trajectory_config(iteration_dirs(root, obsnum))
    actual = bool(
        config["timestream"]["learning"]
        ["map_pixel_outlier_detector_exclusion_feedback_bypass_enabled"]
    )
    if actual != expected_enabled:
        raise ValueError(
            f"merged config bypass state differs for obs={obsnum}: {actual}"
        )

    redu = root / f"redu{terminal_iteration:02d}"
    checkpoint = redu / "citlali_restart_checkpoint.nc"
    with Dataset(checkpoint) as dataset:
        policy = yaml.safe_load(str(dataset["learning_policy_yaml"][0]))
    if (
        bool(
            policy[
                "map_pixel_outlier_detector_exclusion_feedback_bypass_enabled"
            ]
        )
        != expected_enabled
    ):
        raise ValueError(f"checkpoint bypass policy differs: {checkpoint}")

    fits_path = product_path(redu, obsnum, "a1400")
    with fits.open(fits_path, memmap=True) as hdul:
        fits_value = bool(
            hdul[0].header[
                "CONFIG.LEARNING.MAP_OUTLIER_DET_FB_BYPASS"
            ]
        )
    if fits_value != expected_enabled:
        raise ValueError(f"FITS bypass provenance differs: {fits_path}")


def reversal_fraction(
    candidate: float, failed: float, preceding: float, higher_is_better: bool
) -> float:
    if higher_is_better:
        return (candidate - failed) / (preceding - failed)
    return (failed - candidate) / (failed - preceding)


def regression_failures(
    complete: dict, candidate: dict, screen: dict
) -> list[str]:
    failures: list[str] = []
    if (
        abs(candidate["kernel_normalized_central_recovery"] - 1.0)
        - abs(complete["kernel_normalized_central_recovery"] - 1.0)
        > float(screen["maximum_central_recovery_absolute_error_increase"])
    ):
        failures.append("central_recovery")
    for label, key in (
        ("major_width", "major_fwhm_over_kernel"),
        ("minor_width", "minor_fwhm_over_kernel"),
    ):
        if abs(candidate[key] - 1.0) - abs(complete[key] - 1.0) > float(
            screen[f"maximum_{label}_absolute_error_increase"]
        ):
            failures.append(label)
    if candidate["centroid_error_arcsec"] - complete[
        "centroid_error_arcsec"
    ] > float(screen["maximum_centroid_error_increase_arcsec"]):
        failures.append("centroid")
    if candidate["annular_residual_over_truth"] > float(
        screen["maximum_annular_residual_ratio"]
    ) * complete["annular_residual_over_truth"]:
        failures.append("annular_residual")
    if candidate["kernel_residual_relative_rms"] > float(
        screen["maximum_kernel_residual_ratio"]
    ) * complete["kernel_residual_relative_rms"]:
        failures.append("kernel_residual")
    return failures


def regression_failure_details(
    complete: dict, candidate: dict, screen: dict
) -> list[dict]:
    """Return the measured comparison behind each registered failure."""
    failures = set(regression_failures(complete, candidate, screen))
    details: list[dict] = []
    metric_specs = (
        (
            "central_recovery",
            "kernel_normalized_central_recovery",
            "absolute_error_increase",
            float(screen["maximum_central_recovery_absolute_error_increase"]),
        ),
        (
            "major_width",
            "major_fwhm_over_kernel",
            "absolute_error_increase",
            float(screen["maximum_major_width_absolute_error_increase"]),
        ),
        (
            "minor_width",
            "minor_fwhm_over_kernel",
            "absolute_error_increase",
            float(screen["maximum_minor_width_absolute_error_increase"]),
        ),
        (
            "centroid",
            "centroid_error_arcsec",
            "absolute_increase_arcsec",
            float(screen["maximum_centroid_error_increase_arcsec"]),
        ),
        (
            "annular_residual",
            "annular_residual_over_truth",
            "candidate_over_complete_ratio",
            float(screen["maximum_annular_residual_ratio"]),
        ),
        (
            "kernel_residual",
            "kernel_residual_relative_rms",
            "candidate_over_complete_ratio",
            float(screen["maximum_kernel_residual_ratio"]),
        ),
    )
    for label, key, comparison, limit in metric_specs:
        if label not in failures:
            continue
        complete_value = float(complete[key])
        candidate_value = float(candidate[key])
        if comparison == "absolute_error_increase":
            measured = abs(candidate_value - 1.0) - abs(complete_value - 1.0)
        elif comparison == "absolute_increase_arcsec":
            measured = candidate_value - complete_value
        else:
            measured = candidate_value / complete_value
        details.append(
            {
                "metric": label,
                "complete_value": complete_value,
                "candidate_value": candidate_value,
                "comparison": comparison,
                "measured": measured,
                "limit": limit,
            }
        )
    return details


def inherited_failures(
    candidate: dict, reference: dict, screen: dict
) -> list[str]:
    failures: list[str] = []
    if abs(candidate["kernel_normalized_central_recovery"] - 1.0) > abs(
        reference["kernel_normalized_central_recovery"] - 1.0
    ) + float(screen["maximum_central_recovery_absolute_error_increase"]):
        failures.append("central_recovery")
    for label, key in (
        ("major_width", "major_fwhm_over_kernel"),
        ("minor_width", "minor_fwhm_over_kernel"),
    ):
        if abs(candidate[key] - 1.0) > float(
            screen["maximum_width_fractional_error"]
        ):
            failures.append(label)
    if candidate["centroid_error_arcsec"] > float(
        screen["maximum_centroid_error_arcsec"]
    ):
        failures.append("centroid")
    if candidate["annular_residual_over_truth"] > float(
        screen["maximum_residual_ratio"]
    ) * reference["annular_residual_over_truth"]:
        failures.append("annular_residual")
    if candidate["kernel_residual_relative_rms"] > float(
        screen["maximum_residual_ratio"]
    ) * reference["kernel_residual_relative_rms"]:
        failures.append("kernel_residual")
    return failures


def read_execution(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    error_pattern = re.compile(r"\[(?:error|critical)\]|(?:error|critical):", re.I)
    for path in sorted(log_dir.glob("*.log")):
        text = path.read_text(encoding="utf-8")
        wall_match = re.search(r"^\s*([0-9.]+) real\s", text, re.M)
        rss_match = re.search(
            r"^\s*([0-9]+)\s+maximum resident set size$", text, re.M
        )
        if wall_match is None or rss_match is None:
            raise ValueError(f"resource record is incomplete: {path}")
        errors = sum(bool(error_pattern.search(line)) for line in text.splitlines())
        complete = "citlali is done!" in text
        if not complete or errors:
            raise ValueError(
                f"trajectory completion/log check failed: {path} "
                f"complete={complete} errors={errors}"
            )
        candidate = "feedback-excluded" in path.name
        if candidate:
            evidence = re.findall(
                r"EL-F4 mapdiag detector-penalty evidence .*?"
                r"iter=(\d+) map=(\d+) evidence_view=([a-z_]+)",
                text,
            )
            expected_iters = 6 if "123424" in path.name else 7
            if len(evidence) != expected_iters * len(ARRAYS):
                raise ValueError(f"candidate evidence trace incomplete: {path}")
            for iteration_text, _, view in evidence:
                iteration = int(iteration_text)
                expected_view = (
                    "complete_map_no_feedback"
                    if iteration == 0
                    else "feedback_excluded_map"
                )
                if view != expected_view:
                    raise ValueError(f"candidate evidence view differs: {path}")
        rows.append(
            {
                "trajectory": path.stem,
                "status": "completed",
                "wall_seconds": float(wall_match.group(1)),
                "maximum_resident_bytes": int(rss_match.group(1)),
                "error_or_critical_messages": errors,
                "evidence_trace_verified": candidate,
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected eight execution logs; found {len(rows)}")
    return rows


def analyze(
    manifest: dict, repo_root: Path
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    metrics: list[dict] = []
    penalty_inventory: list[dict] = []
    regressions: list[dict] = []
    compatibility_planes = 0
    iteration_zero_planes = 0

    for obsnum_text, case in manifest["cases"].items():
        obsnum = int(obsnum_text)
        stop = int(case["stop_iteration_exclusive"])
        pair_manifest = dict(manifest)
        pair_manifest.update(
            {
                "obsnum": obsnum,
                "stop_iteration_exclusive": stop,
            }
        )
        method_rows: dict[str, list[dict]] = {}
        for method, prefix in (
            ("complete_map", "complete"),
            ("feedback_excluded", "candidate"),
        ):
            control = Path(case[f"{prefix}_control"])
            injected = Path(case[f"{prefix}_injected"])
            method_rows[method] = analyze_pair(
                alpha_label=f"{manifest['alpha']}-{method}-{obsnum}",
                alpha=float(manifest["alpha"]),
                control_root=control,
                injected_root=injected,
                manifest=pair_manifest,
                stop_iteration_exclusive=stop,
            )
            for row in method_rows[method]:
                metrics.append(
                    {"obsnum": obsnum, "evidence_view": method, **row}
                )
            require_candidate_provenance(
                control,
                obsnum,
                int(case["terminal_iteration"]),
                method == "feedback_excluded",
            )
            require_candidate_provenance(
                injected,
                obsnum,
                int(case["terminal_iteration"]),
                method == "feedback_excluded",
            )
            for variant, root in (("control", control), ("injected", injected)):
                for penalty in mapdiag_penalties(
                    root, int(case["terminal_iteration"])
                ):
                    penalty_inventory.append(
                        {
                            "obsnum": obsnum,
                            "evidence_view": method,
                            "variant": variant,
                            **penalty,
                        }
                    )

        for variant in ("control", "injected"):
            compatibility_planes += require_exact_trajectory_images(
                Path(case[f"historical_complete_{variant}"]),
                Path(case[f"complete_{variant}"]),
                obsnum,
                stop,
            )
        iteration_zero_planes += require_candidate_iteration_zero_exact(
            Path(case["candidate_control"]),
            Path(case["candidate_injected"]),
            obsnum,
        )

        for iteration in range(int(manifest["injection_start_iteration"]), stop):
            for array in ARRAYS:
                complete = select_row(
                    method_rows["complete_map"],
                    float(manifest["alpha"]),
                    iteration,
                    array,
                )
                candidate = select_row(
                    method_rows["feedback_excluded"],
                    float(manifest["alpha"]),
                    iteration,
                    array,
                )
                failures = regression_failures(
                    complete, candidate, manifest["regression_screen"]
                )
                if failures:
                    regressions.append(
                        {
                            "obsnum": obsnum,
                            "iteration": iteration,
                            "array": array,
                            "failures": failures,
                            "failure_details": regression_failure_details(
                                complete,
                                candidate,
                                manifest["regression_screen"],
                            ),
                            "complete_annular_residual": complete[
                                "annular_residual_over_truth"
                            ],
                            "candidate_annular_residual": candidate[
                                "annular_residual_over_truth"
                            ],
                            "annular_residual_ratio": candidate[
                                "annular_residual_over_truth"
                            ]
                            / complete["annular_residual_over_truth"],
                        }
                    )

    primary_case = manifest["cases"][manifest["primary_gate"]["obsnum"]]
    primary_row = next(
        row
        for row in metrics
        if row["obsnum"] == int(manifest["primary_gate"]["obsnum"])
        and row["evidence_view"] == "feedback_excluded"
        and row["iteration"] == int(primary_case["terminal_iteration"])
        and row["array"] == "a1400"
    )
    gate = manifest["primary_gate"]
    recovery_reversal = reversal_fraction(
        primary_row["kernel_normalized_central_recovery"],
        float(gate["original_iteration_5_recovery"]),
        float(gate["original_iteration_4_recovery"]),
        True,
    )
    annular_reversal = reversal_fraction(
        primary_row["annular_residual_over_truth"],
        float(gate["original_iteration_5_annular_residual"]),
        float(gate["original_iteration_4_annular_residual"]),
        False,
    )
    primary_pass = (
        recovery_reversal >= float(gate["minimum_reversal_fraction"])
        and annular_reversal >= float(gate["minimum_reversal_fraction"])
    )

    target = gate["target_penalty"]
    candidate_target_records = [
        row
        for row in penalty_inventory
        if row["obsnum"] == int(gate["obsnum"])
        and row["evidence_view"] == "feedback_excluded"
        and row["variant"] == "injected"
        and all(row[key] == value for key, value in target.items())
    ]
    target_absent = not candidate_target_records
    primary_pass = primary_pass and target_absent

    inherited: list[dict] = []
    for obsnum_text, case in manifest["cases"].items():
        refs = read_reference_metrics(repo_root / case["inherited_reference_metrics"])
        for array in ARRAYS:
            candidate = next(
                row
                for row in metrics
                if row["obsnum"] == int(obsnum_text)
                and row["evidence_view"] == "feedback_excluded"
                and row["iteration"] == int(case["terminal_iteration"])
                and row["array"] == array
            )
            reference = refs[
                (
                    float(case["inherited_reference_alpha"]),
                    int(case["inherited_reference_iteration"]),
                    array,
                )
            ]
            inherited.append(
                {
                    "obsnum": int(obsnum_text),
                    "array": array,
                    "failures": inherited_failures(
                        candidate, reference, manifest["inherited_screen"]
                    ),
                }
            )

    execution = read_execution(Path(manifest["output_root"]) / "logs")
    performance_comparison: list[dict] = []
    for obsnum_text in manifest["cases"]:
        complete_times = [
            row["wall_seconds"]
            for row in execution
            if f"point-{obsnum_text}-complete-map" in row["trajectory"]
        ]
        candidate_times = [
            row["wall_seconds"]
            for row in execution
            if f"point-{obsnum_text}-feedback-excluded" in row["trajectory"]
        ]
        if len(complete_times) != 2 or len(candidate_times) != 2:
            raise ValueError(f"incomplete performance pair: obs={obsnum_text}")
        complete_mean = float(np.mean(complete_times))
        candidate_mean = float(np.mean(candidate_times))
        performance_comparison.append(
            {
                "obsnum": int(obsnum_text),
                "complete_pair_mean_wall_seconds": complete_mean,
                "candidate_pair_mean_wall_seconds": candidate_mean,
                "candidate_wall_time_change_percent": 100.0
                * (candidate_mean / complete_mean - 1.0),
            }
        )
    penalty_comparison = compare_penalties(penalty_inventory)
    penalty_dispositions = {
        disposition: sum(
            row["disposition"] == disposition for row in penalty_comparison
        )
        for disposition in (
            "retained",
            "retained_value_changed",
            "timing_changed",
            "removed",
            "added",
        )
    }
    output_bytes = sum(
        path.stat().st_size
        for path in Path(manifest["output_root"]).rglob("*")
        if path.is_file()
    )
    if not primary_pass:
        disposition = "do_not_advance"
    elif regressions:
        disposition = "mechanism_helpful_but_regressive"
    else:
        disposition = "advance_to_broader_policy_testing"
    result = {
        "test_id": manifest["test_id"],
        "valid_complete_matrix": True,
        "default_off_bitwise_compatible_planes": compatibility_planes,
        "candidate_iteration_zero_bitwise_equal_planes": iteration_zero_planes,
        "primary_gate": {
            "target_penalty_absent": target_absent,
            "a1400_iteration_5_recovery": primary_row[
                "kernel_normalized_central_recovery"
            ],
            "a1400_iteration_5_annular_residual": primary_row[
                "annular_residual_over_truth"
            ],
            "recovery_reversal_fraction": recovery_reversal,
            "annular_residual_reversal_fraction": annular_reversal,
            "full_reversal": recovery_reversal >= float(gate["full_reversal_fraction"])
            and annular_reversal >= float(gate["full_reversal_fraction"]),
            "passed": primary_pass,
        },
        "new_protected_regression_count": len(regressions),
        "new_protected_regressions": regressions,
        "inherited_absolute_screen": inherited,
        "all_array_inherited_absolute_screen_pass": all(
            not row["failures"] for row in inherited
        ),
        "execution": {
            "trajectory_count": len(execution),
            "aggregate_wall_seconds": sum(row["wall_seconds"] for row in execution),
            "maximum_resident_bytes": max(
                row["maximum_resident_bytes"] for row in execution
            ),
            "retained_output_bytes": output_bytes,
        },
        "penalty_comparison": penalty_dispositions,
        "performance_comparison": performance_comparison,
        "disposition": disposition,
        "claim_scope": "two exposed bright compact-source pointing observations",
    }
    return metrics, penalty_inventory, penalty_comparison, execution, result


def write_report(path: Path, metrics: list[dict], penalties: list[dict], result: dict) -> None:
    primary = result["primary_gate"]
    lines = [
        "# SCI-FRUIT EL-F4 feedback-model-bypass result",
        "",
        f"Disposition: **{result['disposition']}**",
        "",
        "This is development evidence only; it does not qualify a FRUIT method.",
        "",
        "## Primary mechanism gate",
        "",
        f"- UID 4460 penalty absent: `{primary['target_penalty_absent']}`",
        f"- a1400 iteration-5 recovery: `{primary['a1400_iteration_5_recovery']:.9f}`",
        f"- a1400 iteration-5 annular residual: `{primary['a1400_iteration_5_annular_residual']:.9g}`",
        f"- recovery reversal fraction: `{primary['recovery_reversal_fraction']:.9f}`",
        f"- annular-residual reversal fraction: `{primary['annular_residual_reversal_fraction']:.9f}`",
        f"- full reversal: `{primary['full_reversal']}`",
        "",
        "## New protected regressions",
        "",
    ]
    if result["new_protected_regressions"]:
        lines.extend(
            [
                "| Obs | Iter | Array | Failed metric | Complete | Candidate | Registered comparison |",
                "|---:|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in result["new_protected_regressions"]:
            for detail in row["failure_details"]:
                lines.append(
                    f"| {row['obsnum']} | {row['iteration']} | "
                    f"{row['array']} | {detail['metric']} | "
                    f"{detail['complete_value']:.9g} | "
                    f"{detail['candidate_value']:.9g} | "
                    f"{detail['measured']:.6f} "
                    f"({detail['comparison']}; limit {detail['limit']:.6f}) |"
                )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Terminal candidate versus complete-map response",
            "",
            "| Obs | Array | Complete recovery | Candidate recovery | Complete annular | Candidate annular |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    by_key = {
        (row["obsnum"], row["evidence_view"], row["iteration"], row["array"]): row
        for row in metrics
    }
    terminals = {123424: 5, 152389: 6}
    for obsnum, iteration in terminals.items():
        for array in ARRAYS:
            complete = by_key[(obsnum, "complete_map", iteration, array)]
            candidate = by_key[(obsnum, "feedback_excluded", iteration, array)]
            lines.append(
                f"| {obsnum} | {array} | "
                f"{complete['kernel_normalized_central_recovery']:.6f} | "
                f"{candidate['kernel_normalized_central_recovery']:.6f} | "
                f"{complete['annular_residual_over_truth']:.8g} | "
                f"{candidate['annular_residual_over_truth']:.8g} |"
            )
    lines.extend(
        [
            "",
            "## Inherited absolute screen",
            "",
            "This screen is reported for context and does not decide whether the bypass caused a new regression.",
            "",
            "| Obs | Array | Result | Failed protections |",
            "|---:|---|---|---|",
        ]
    )
    for row in result["inherited_absolute_screen"]:
        failures = ", ".join(row["failures"]) if row["failures"] else "none"
        outcome = "pass" if not row["failures"] else "fail"
        lines.append(
            f"| {row['obsnum']} | {row['array']} | {outcome} | {failures} |"
        )
    lines.extend(
        [
            "",
            "## Execution performance",
            "",
            "| Obs | Complete pair mean (s) | Candidate pair mean (s) | Candidate change |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["performance_comparison"]:
        lines.append(
            f"| {row['obsnum']} | "
            f"{row['complete_pair_mean_wall_seconds']:.3f} | "
            f"{row['candidate_pair_mean_wall_seconds']:.3f} | "
            f"{row['candidate_wall_time_change_percent']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Validity and bounds",
            "",
            f"- Default-off bitwise-compatible image planes: `{result['default_off_bitwise_compatible_planes']}`",
            f"- Candidate iteration-0 bitwise-equal planes: `{result['candidate_iteration_zero_bitwise_equal_planes']}`",
            f"- Completed first-attempt trajectories: `{result['execution']['trajectory_count']}`",
            f"- Aggregate wall time: `{result['execution']['aggregate_wall_seconds']:.2f} s`",
            f"- Maximum resident memory: `{result['execution']['maximum_resident_bytes'] / 2**30:.3f} GiB`",
            f"- Retained output: `{result['execution']['retained_output_bytes'] / 2**30:.3f} GiB`",
            "- Error/critical messages: `0`",
            "- Configuration, FITS provenance, checkpoint policy, and candidate evidence traces: verified",
            "- Penalty comparison (candidate versus complete-map evidence): "
            + ", ".join(
                f"{key} `{value}`"
                for key, value in result["penalty_comparison"].items()
            ),
            "",
            "## Interpretation",
            "",
            "The bypass reproduces the earlier causal rescue on observation 123424, but the wholesale policy is too broad. It introduces three preregistered a2000 kernel-residual regressions on observation 123424 and raises the observation-152389 a1100 iteration-4 annular residual by more than the preregistered ten-percent allowance. The registered disposition therefore stops advancement of this exact policy. The result supports a narrower future hypothesis that distinguishes injection-sensitive feedback-driven penalties from repeatable ordinary detector evidence; it does not authorize another run.",
            "",
            f"The terminal effective map-diagnostic penalty inventory contains `{len(penalties)}` records across the eight trajectories; see `PENALTY_INVENTORY_R0.1.csv` for exact identities.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--penalties", required=True, type=Path)
    parser.add_argument("--penalty-comparison", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    metrics, penalties, penalty_comparison, execution, result = analyze(
        manifest, repo_root
    )
    write_csv(args.metrics, metrics)
    write_csv(args.penalties, penalties)
    write_csv(args.penalty_comparison, penalty_comparison)
    write_csv(args.execution, execution)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.report, metrics, penalties, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
