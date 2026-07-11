#!/usr/bin/env python3
"""Apply the accepted scale-aware science reduction equivalence gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from astropy.io import fits


SCHEMA_VERSION = "citlali-science-equivalence-result-v1"
PROFILE_SCHEMA_VERSION = "citlali-science-equivalence-profile-v1"


def product_map(root: Path, pattern: str) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in root.rglob(pattern)}


def rms_relative(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline_values = np.asarray(baseline).reshape(-1)
    candidate_values = np.asarray(candidate).reshape(-1)
    finite = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    if not finite.any():
        return 0.0
    baseline_values = baseline_values[finite]
    difference = candidate_values[finite] - baseline_values
    baseline_rms = float(np.sqrt(np.mean(np.abs(baseline_values) ** 2)))
    difference_rms = float(np.sqrt(np.mean(np.abs(difference) ** 2)))
    return difference_rms / max(baseline_rms, 1.0e-300)


def fits_metrics(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    baseline_products = product_map(baseline_root, "*.fits")
    candidate_products = product_map(candidate_root, "*.fits")
    sets_exact = baseline_products.keys() == candidate_products.keys()
    layer_metrics: list[float] = []
    compared_layers = 0
    for name in sorted(baseline_products.keys() & candidate_products.keys()):
        with fits.open(baseline_products[name], memmap=True) as baseline_hdus, fits.open(
            candidate_products[name], memmap=True
        ) as candidate_hdus:
            if len(baseline_hdus) != len(candidate_hdus):
                sets_exact = False
                continue
            for baseline_hdu, candidate_hdu in zip(
                baseline_hdus[1:], candidate_hdus[1:]
            ):
                if baseline_hdu.name != candidate_hdu.name:
                    sets_exact = False
                    continue
                if baseline_hdu.data is None or candidate_hdu.data is None:
                    continue
                layer_metrics.append(
                    rms_relative(baseline_hdu.data, candidate_hdu.data)
                )
                compared_layers += 1
    values = np.asarray(layer_metrics, dtype=float)
    return {
        "fits_product_sets_exact": sets_exact,
        "fits_product_count": len(baseline_products),
        "map_layer_count": compared_layers,
        "map_rms_relative_max": float(values.max(initial=0.0)),
        "map_rms_relative_p99": (
            float(np.quantile(values, 0.99)) if values.size else 0.0
        ),
    }


def float_metrics(
    name: str, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    baseline_values = np.asarray(baseline, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    finite = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    if not finite.any():
        return {"rms_relative": 0.0, "max_absolute": 0.0, "max_fractional": 0.0}
    baseline_values = baseline_values[finite]
    candidate_values = candidate_values[finite]
    difference = np.abs(candidate_values - baseline_values)
    fractional = difference / np.maximum(np.abs(baseline_values), 1.0e-12)
    return {
        "rms_relative": rms_relative(baseline_values, candidate_values),
        "max_absolute": float(difference.max(initial=0.0)),
        "max_fractional": float(fractional.max(initial=0.0)),
    }


def netcdf_metrics(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    baseline_products = product_map(baseline_root, "*.nc")
    candidate_products = product_map(candidate_root, "*.nc")
    sets_exact = baseline_products.keys() == candidate_products.keys()
    integer_exact = True
    integer_mismatches: list[str] = []
    ptc_weight_rms = 0.0
    median_absolute = 0.0
    median_fractional = 0.0
    other_rms = 0.0
    compared_variables = 0

    for product in sorted(baseline_products.keys() & candidate_products.keys()):
        with netCDF4.Dataset(baseline_products[product]) as baseline_file, netCDF4.Dataset(
            candidate_products[product]
        ) as candidate_file:
            baseline_names = set(baseline_file.variables)
            candidate_names = set(candidate_file.variables)
            if baseline_names != candidate_names:
                sets_exact = False
            for name in sorted(baseline_names & candidate_names):
                baseline = np.asarray(baseline_file[name][:])
                candidate = np.asarray(candidate_file[name][:])
                if baseline.shape != candidate.shape:
                    sets_exact = False
                    continue
                if baseline.dtype.kind not in "biufc?" or baseline.size == 0:
                    continue
                compared_variables += 1
                if baseline.dtype.kind in "biu?":
                    if not np.array_equal(baseline, candidate):
                        integer_exact = False
                        integer_mismatches.append(f"{product}:{name}")
                    continue
                metrics = float_metrics(name, baseline, candidate)
                if name == "ptc_detector_weight":
                    ptc_weight_rms = max(ptc_weight_rms, metrics["rms_relative"])
                elif name == "ptc_detector_median":
                    median_absolute = max(median_absolute, metrics["max_absolute"])
                    median_fractional = max(
                        median_fractional, metrics["max_fractional"]
                    )
                else:
                    other_rms = max(other_rms, metrics["rms_relative"])

    return {
        "netcdf_product_sets_exact": sets_exact,
        "netcdf_product_count": len(baseline_products),
        "netcdf_variable_count": compared_variables,
        "integer_diagnostics_exact": integer_exact,
        "integer_diagnostic_mismatches": integer_mismatches,
        "ptc_weight_rms_relative_max": ptc_weight_rms,
        "detector_median_absolute_max": median_absolute,
        "detector_median_fractional_max": median_fractional,
        "other_diagnostic_rms_relative_max": other_rms,
    }


def evaluate(metrics: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    requirements = profile["requirements"]
    thresholds = profile["thresholds"]
    for name, required in requirements.items():
        if required and metrics.get(name) is not True:
            failures.append(f"{name}: required true, got {metrics.get(name)!r}")
    for name, threshold in thresholds.items():
        if metrics[name] > threshold:
            failures.append(f"{name}: {metrics[name]:.8g} > {threshold:.8g}")
    return failures


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Science Scientific Equivalence",
        "",
        f"- Profile: `{result['profile_id']}`",
        f"- Baseline: `{result['baseline']}`",
        f"- Candidate: `{result['candidate']}`",
        f"- Verdict: **{result['verdict']}**",
        "",
        "## Metrics",
        "",
    ]
    for key, value in result["metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in result["failures"])
    if not result["failures"]:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("validation/profiles/science_scientific_equivalence_v1.json"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported science equivalence profile schema")
    metrics = {
        **fits_metrics(args.baseline, args.candidate),
        **netcdf_metrics(args.baseline, args.candidate),
    }
    failures = evaluate(metrics, profile)
    result = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "metrics": metrics,
        "failures": failures,
        "verdict": "accepted" if not failures else "rejected",
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    report = render_markdown(result)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
