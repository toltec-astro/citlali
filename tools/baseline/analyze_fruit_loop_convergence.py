#!/usr/bin/env python3
"""Measure saved fruit-loop iterations without changing production behavior."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits


MANIFEST_SCHEMA_VERSION = "citlali-fruit-loop-convergence-study-v1"
RESULT_SCHEMA_VERSION = "citlali-fruit-loop-convergence-result-v1"
LEARNING_RE = re.compile(
    r"reduction learning finalize:.*?"
    r"phase=(?P<phase>\S+).*?"
    r"effective_sample_mask_intervals=(?P<masks>\d+).*?"
    r"effective_detector_penalties=(?P<penalties>\d+)"
)
RUNTIME_RE = re.compile(
    r"profile stage=reduction\.iteration .*?elapsed_s=(?P<seconds>[0-9.]+)"
)
VERSION_RE = re.compile(r"citlali version:\s*(?P<version>\S+)", re.IGNORECASE)
WCS_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
)


class StudyError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StudyError(f"{path}: expected a JSON object")
    return value


def resolve_path(manifest_path: Path, raw: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not expanded.is_absolute():
        expanded = manifest_path.parent / expanded
    return expanded.resolve()


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(flatten(item, f"{prefix}[{index}]"))
    else:
        result[prefix] = json.dumps(value, sort_keys=True, default=str)
    return result


def reduction_config(reduction_dir: Path) -> tuple[Path, dict[str, str]]:
    configs = sorted(
        reduction_dir.glob("citlali*.yaml"),
        key=lambda path: ("merged" in path.name, path.name),
    )
    if not configs:
        raise StudyError(f"{reduction_dir}: no citlali*.yaml config found")
    with configs[0].open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise StudyError(f"{configs[0]}: expected a YAML mapping")
    return configs[0], flatten(value)


def config_differences(
    baseline: dict[str, str],
    candidate: dict[str, str],
    ignored_paths: list[str],
) -> list[dict[str, Any]]:
    differences = []
    for path in sorted(set(baseline) | set(candidate)):
        if any(fnmatch.fnmatch(path, pattern) for pattern in ignored_paths):
            continue
        if baseline.get(path) != candidate.get(path):
            differences.append(
                {
                    "path": path,
                    "baseline": baseline.get(path),
                    "candidate": candidate.get(path),
                }
            )
    return differences


def read_log_summary(reduction_dir: Path) -> dict[str, Any]:
    paths = [reduction_dir / "citlali.log.gz", reduction_dir / "citlali.log"]
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return {
            "available": False,
            "path": None,
            "citlali_version": None,
            "learning": None,
            "iteration_runtime_seconds": None,
        }
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            text = stream.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    learning_matches = list(LEARNING_RE.finditer(text))
    runtime_matches = list(RUNTIME_RE.finditer(text))
    version_matches = list(VERSION_RE.finditer(text))
    learning = None
    if learning_matches:
        match = learning_matches[-1]
        learning = {
            "phase": match.group("phase"),
            "effective_sample_mask_intervals": int(match.group("masks")),
            "effective_detector_penalties": int(match.group("penalties")),
        }
    return {
        "available": True,
        "path": str(path),
        "citlali_version": (
            version_matches[-1].group("version") if version_matches else None
        ),
        "learning": learning,
        "iteration_runtime_seconds": (
            float(runtime_matches[-1].group("seconds")) if runtime_matches else None
        ),
    }


def pixel_scale_arcsec(header: fits.Header) -> float:
    for key in ("CDELT1", "CD1_1"):
        value = header.get(key)
        if value is not None and math.isfinite(float(value)) and float(value) != 0.0:
            return abs(float(value)) * 3600.0
    raise StudyError("signal HDU has no usable CDELT1 or CD1_1 pixel scale")


def relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.sum(reference * reference, dtype=np.float64))
    if denominator <= 0.0:
        return math.inf
    numerator = float(np.sum(delta * delta, dtype=np.float64))
    return math.sqrt(numerator / denominator)


def fractional_change(previous: float, current: float) -> float | None:
    if not math.isfinite(previous) or not math.isfinite(current) or previous == 0.0:
        return None
    return abs(current - previous) / abs(previous)


def load_map(path: Path, product: dict[str, Any]) -> dict[str, Any]:
    signal_name = str(product.get("signal_hdu", "signal_I"))
    weight_name = str(product.get("weight_hdu", "weight_I"))
    coverage_name = str(product.get("coverage_hdu", "coverage_bool_I"))
    with fits.open(path, memmap=True) as hdus:
        try:
            signal_hdu = hdus[signal_name]
            weight_hdu = hdus[weight_name]
        except KeyError as exc:
            raise StudyError(f"{path}: required HDU missing: {exc}") from exc
        signal = np.asarray(signal_hdu.data, dtype=np.float64).squeeze()
        weight = np.asarray(weight_hdu.data, dtype=np.float64).squeeze()
        coverage = (
            np.asarray(hdus[coverage_name].data).squeeze()
            if coverage_name in hdus
            else np.ones(signal.shape, dtype=bool)
        )
        signal_unit = signal_hdu.header.get("BUNIT")
        weight_unit = weight_hdu.header.get("BUNIT")
        scale_arcsec = pixel_scale_arcsec(signal_hdu.header)
        wcs_identity = {
            key: signal_hdu.header[key]
            for key in WCS_KEYS
            if key in signal_hdu.header
        }
    if signal.ndim != 2 or weight.shape != signal.shape or coverage.shape != signal.shape:
        raise StudyError(
            f"{path}: expected matching 2-D signal, weight, and coverage planes"
        )
    return {
        "signal": signal,
        "weight": weight,
        "coverage": coverage,
        "signal_unit": signal_unit,
        "weight_unit": weight_unit,
        "pixel_scale_arcsec": scale_arcsec,
        "wcs_identity": wcs_identity,
    }


def map_path(reduction_dir: Path, pattern: str, array: str) -> Path:
    matches = sorted(reduction_dir.glob(pattern.format(array=array)))
    if len(matches) != 1:
        raise StudyError(
            f"{reduction_dir}: pattern {pattern!r} for {array} matched "
            f"{len(matches)} files; expected exactly one"
        )
    return matches[0]


def compare_maps(
    previous_path: Path,
    current_path: Path,
    product: dict[str, Any],
) -> dict[str, Any]:
    previous = load_map(previous_path, product)
    current = load_map(current_path, product)
    if previous["signal"].shape != current["signal"].shape:
        raise StudyError(
            f"map shape changed: {previous_path} {previous['signal'].shape} != "
            f"{current_path} {current['signal'].shape}"
        )
    for key in (
        "signal_unit",
        "weight_unit",
        "pixel_scale_arcsec",
        "wcs_identity",
    ):
        if previous[key] != current[key]:
            raise StudyError(
                f"map identity changed for {key}: {previous[key]!r} != {current[key]!r}"
            )

    previous_signal = previous["signal"]
    current_signal = current["signal"]
    previous_weight = previous["weight"]
    current_weight = current["weight"]
    previous_valid = (
        np.isfinite(previous_signal)
        & np.isfinite(previous_weight)
        & (previous_weight > 0.0)
        & np.asarray(previous["coverage"], dtype=bool)
    )
    current_valid = (
        np.isfinite(current_signal)
        & np.isfinite(current_weight)
        & (current_weight > 0.0)
        & np.asarray(current["coverage"], dtype=bool)
    )
    union = previous_valid | current_valid
    common = previous_valid & current_valid
    if not common.any():
        raise StudyError(f"{previous_path} and {current_path}: no common valid pixels")

    radius_arcsec = float(product["aperture_radius_arcsec"])
    rows, columns = previous_signal.shape
    y, x = np.indices(previous_signal.shape)
    center_y = (rows - 1) / 2.0
    center_x = (columns - 1) / 2.0
    radius_pixels = radius_arcsec / float(previous["pixel_scale_arcsec"])
    aperture = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius_pixels**2
    aperture_common = common & aperture
    if not aperture_common.any():
        raise StudyError(f"{previous_path}: source aperture has no common valid pixels")

    delta = current_signal - previous_signal
    map_delta = delta[common]
    aperture_delta = delta[aperture_common]
    difference_weight = (
        previous_weight[common]
        * current_weight[common]
        / (previous_weight[common] + current_weight[common])
    )
    previous_peak = float(np.max(previous_signal[aperture_common]))
    current_peak = float(np.max(current_signal[aperture_common]))
    previous_sum = float(np.sum(previous_signal[aperture_common], dtype=np.float64))
    current_sum = float(np.sum(current_signal[aperture_common], dtype=np.float64))

    return {
        "previous_path": str(previous_path),
        "current_path": str(current_path),
        "shape": [rows, columns],
        "signal_unit": previous["signal_unit"],
        "weight_unit": previous["weight_unit"],
        "pixel_scale_arcsec": previous["pixel_scale_arcsec"],
        "common_valid_pixels": int(common.sum()),
        "support_jaccard": float(common.sum() / union.sum()) if union.any() else 1.0,
        "map_relative_l2_delta": relative_l2(
            map_delta, previous_signal[common]
        ),
        "aperture_relative_l2_delta": relative_l2(
            aperture_delta, previous_signal[aperture_common]
        ),
        "peak_fractional_change": fractional_change(previous_peak, current_peak),
        "aperture_sum_fractional_change": fractional_change(
            previous_sum, current_sum
        ),
        "weight_relative_l2_delta": relative_l2(
            current_weight[common] - previous_weight[common],
            previous_weight[common],
        ),
        "noise_scaled_delta_rms": float(
            np.sqrt(
                np.mean(
                    map_delta * map_delta * difference_weight,
                    dtype=np.float64,
                )
            )
        ),
    }


def learning_is_stable(
    previous: dict[str, Any] | None, current: dict[str, Any] | None
) -> bool | None:
    if previous is None or current is None:
        return None
    return (
        previous["effective_sample_mask_intervals"]
        == current["effective_sample_mask_intervals"]
        and previous["effective_detector_penalties"]
        == current["effective_detector_penalties"]
        and current["phase"] == "apply"
    )


def threshold_pass(metric: float | None, maximum: float) -> bool:
    return metric is not None and math.isfinite(metric) and metric <= maximum


def evaluate_rule(
    rule: dict[str, Any],
    transitions: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
) -> dict[str, Any]:
    required = int(rule["consecutive_passes"])
    minimum = int(rule["minimum_completed_iteration_id"])
    consecutive = 0
    stop_iteration_id = None
    evaluations = []
    for transition in transitions:
        array_results = []
        for array, metrics in transition["arrays"].items():
            failures = []
            if metrics["support_jaccard"] < float(rule["support_jaccard_min"]):
                failures.append("support_jaccard")
            for key in (
                "map_relative_l2_delta",
                "aperture_relative_l2_delta",
                "peak_fractional_change",
                "weight_relative_l2_delta",
            ):
                if not threshold_pass(metrics[key], float(rule[f"{key}_max"])):
                    failures.append(key)
            array_results.append(
                {"array": array, "passed": not failures, "failures": failures}
            )
        learning_ok = (
            transition["learning_state_stable"] is True
            if rule.get("require_learning_state_stable", True)
            else True
        )
        eligible = transition["current_iteration_id"] >= minimum
        passed = eligible and learning_ok and all(
            result["passed"] for result in array_results
        )
        consecutive = consecutive + 1 if passed else 0
        evaluations.append(
            {
                "current_iteration_id": transition["current_iteration_id"],
                "eligible": eligible,
                "learning_passed": learning_ok,
                "arrays": array_results,
                "passed": passed,
                "consecutive_passes": consecutive,
            }
        )
        if consecutive >= required:
            stop_iteration_id = transition["current_iteration_id"]
            break

    saved = [
        iteration
        for iteration in iterations
        if stop_iteration_id is not None
        and iteration["iteration_id"] > stop_iteration_id
    ]
    saved_seconds_values = [
        iteration["log"]["iteration_runtime_seconds"]
        for iteration in saved
        if iteration["log"]["iteration_runtime_seconds"] is not None
    ]
    return {
        "rule_id": rule["rule_id"],
        "definition": rule,
        "status": "candidate_only_not_production_approved",
        "stop_iteration_id": stop_iteration_id,
        "saved_iteration_count": len(saved),
        "estimated_saved_seconds": (
            float(sum(saved_seconds_values))
            if len(saved_seconds_values) == len(saved)
            else None
        ),
        "evaluations": evaluations,
    }


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise StudyError(f"{manifest_path}: unsupported manifest schema")
    product = manifest.get("product")
    entries = manifest.get("iterations")
    if not isinstance(product, dict) or not isinstance(entries, list) or len(entries) < 2:
        raise StudyError("manifest requires product settings and at least two iterations")
    arrays = product.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise StudyError("product.arrays must be a non-empty list")
    if product.get("aperture_center") != "map_center":
        raise StudyError("product.aperture_center must explicitly be 'map_center'")

    ignored_paths = [str(value) for value in manifest.get("config_ignore_paths", [])]
    iterations = []
    baseline_config: dict[str, str] | None = None
    baseline_version: str | None = None
    protocol_errors = []
    seen_ids: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise StudyError("iteration entries must be objects")
        iteration_id = int(entry["iteration_id"])
        if iteration_id in seen_ids:
            raise StudyError(f"duplicate iteration_id {iteration_id}")
        seen_ids.add(iteration_id)
        reduction_dir = resolve_path(manifest_path, str(entry["reduction_dir"]))
        if not reduction_dir.is_dir():
            raise StudyError(f"{reduction_dir}: reduction directory not found")
        config_path, config = reduction_config(reduction_dir)
        differences = (
            []
            if baseline_config is None
            else config_differences(baseline_config, config, ignored_paths)
        )
        if differences:
            protocol_errors.append(
                f"iteration {iteration_id}: config differs from first iteration"
            )
        baseline_config = config if baseline_config is None else baseline_config
        log = read_log_summary(reduction_dir)
        version = log["citlali_version"]
        if version is None:
            protocol_errors.append(
                f"iteration {iteration_id}: Citlali version unavailable"
            )
        elif baseline_version is None:
            baseline_version = version
        elif version != baseline_version:
            protocol_errors.append(
                f"iteration {iteration_id}: Citlali version differs from first iteration"
            )
        iterations.append(
            {
                "iteration_id": iteration_id,
                "reduction_dir": str(reduction_dir),
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "config_differences": differences,
                "log": log,
            }
        )
    if [item["iteration_id"] for item in iterations] != sorted(seen_ids):
        raise StudyError("iterations must be ordered by increasing iteration_id")

    transitions = []
    pattern = str(product["path_pattern"])
    for previous, current in zip(iterations, iterations[1:]):
        array_metrics = {}
        for array in arrays:
            previous_path = map_path(Path(previous["reduction_dir"]), pattern, str(array))
            current_path = map_path(Path(current["reduction_dir"]), pattern, str(array))
            array_metrics[str(array)] = compare_maps(
                previous_path, current_path, product
            )
        transitions.append(
            {
                "previous_iteration_id": previous["iteration_id"],
                "current_iteration_id": current["iteration_id"],
                "learning_state_stable": learning_is_stable(
                    previous["log"]["learning"], current["log"]["learning"]
                ),
                "arrays": array_metrics,
            }
        )

    rules = manifest.get("candidate_rules") or []
    if not isinstance(rules, list):
        raise StudyError("candidate_rules must be a list")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "manifest_path": str(manifest_path),
        "disposition": "offline_evidence_only",
        "production_change_authorized": False,
        "protocol_complete": not protocol_errors,
        "protocol_errors": protocol_errors,
        "config_ignore_paths": ignored_paths,
        "product": product,
        "iterations": iterations,
        "transitions": transitions,
        "candidate_rule_results": [
            evaluate_rule(rule, transitions, iterations) for rule in rules
        ],
    }


def markdown_report(result: dict[str, Any]) -> str:
    def metric(value: float | None) -> str:
        return "not_assessable" if value is None else f"{value:.6g}"

    lines = [
        f"# Fruit-Loop Convergence Study: {result['study_id']}",
        "",
        f"- Disposition: `{result['disposition']}`",
        f"- Protocol complete: `{str(result['protocol_complete']).lower()}`",
        "- Production change authorized: `false`",
        "",
    ]
    if result["protocol_errors"]:
        lines.extend(["## Protocol Errors", ""])
        lines.extend(f"- {error}" for error in result["protocol_errors"])
        lines.append("")
    lines.extend(
        [
            "## Consecutive Iteration Metrics",
            "",
            "| Transition | Array | Support Jaccard | Map relative L2 | "
            "Aperture relative L2 | Peak fractional | Weight relative L2 | "
            "Learning stable |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for transition in result["transitions"]:
        label = (
            f"{transition['previous_iteration_id']} -> "
            f"{transition['current_iteration_id']}"
        )
        for array, metrics in transition["arrays"].items():
            lines.append(
                f"| {label} | {array} | {metrics['support_jaccard']:.6g} | "
                f"{metrics['map_relative_l2_delta']:.6g} | "
                f"{metrics['aperture_relative_l2_delta']:.6g} | "
                f"{metric(metrics['peak_fractional_change'])} | "
                f"{metrics['weight_relative_l2_delta']:.6g} | "
                f"{transition['learning_state_stable']} |"
            )
    lines.extend(["", "## Candidate Rules", ""])
    if not result["candidate_rule_results"]:
        lines.append("No candidate rules were evaluated.")
    for rule in result["candidate_rule_results"]:
        lines.extend(
            [
                f"### {rule['rule_id']}",
                "",
                "- Status: `candidate_only_not_production_approved`",
                f"- Simulated stop iteration: `{rule['stop_iteration_id']}`",
                f"- Saved iterations: `{rule['saved_iteration_count']}`",
                f"- Estimated saved seconds: `{rule['estimated_saved_seconds']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "These measurements describe one saved iteration sequence. They do not "
            "approve a scientific stopping threshold. A production rule requires "
            "scientific-owner approval and evidence across representative source "
            "morphologies, brightnesses, arrays, and learning modes.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.manifest.resolve())
    except (OSError, StudyError, ValueError, KeyError) as exc:
        print(f"fruit-loop convergence study failed: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if args.report_out:
        args.report_out.write_text(markdown_report(result), encoding="utf-8")
    return 0 if result["protocol_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
