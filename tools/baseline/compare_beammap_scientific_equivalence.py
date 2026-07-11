#!/usr/bin/env python3
"""Apply the accepted scale-aware Beammap scientific equivalence gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table


SCHEMA_VERSION = "citlali-beammap-equivalence-result-v1"
PROFILE_SCHEMA_VERSION = "citlali-beammap-equivalence-profile-v1"
LAYER_NAMES = ("signal", "weight", "kernel")
DET_RE = re.compile(r"_det_(\d+)_")


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def relative_difference(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    floor = np.maximum(np.abs(baseline), 1.0e-12)
    return np.abs(candidate - baseline) / floor


def apt_metrics(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    pattern = "apt_*_beammap_*_citlali.ecsv"
    baseline_path = find_one(baseline_root, pattern)
    candidate_path = find_one(candidate_root, pattern)
    baseline = Table.read(baseline_path)
    candidate = Table.read(candidate_path)

    baseline_uid = np.asarray(baseline["uid"])
    candidate_uid = np.asarray(candidate["uid"])
    identity_exact = np.array_equal(baseline_uid, candidate_uid)
    flags_exact = identity_exact and all(
        np.array_equal(np.asarray(baseline[name]), np.asarray(candidate[name]))
        for name in ("flag", "flag2")
    )

    def max_relative(name: str) -> float:
        values = relative_difference(
            np.asarray(candidate[name], dtype=float),
            np.asarray(baseline[name], dtype=float),
        )
        finite = values[np.isfinite(values)]
        return float(finite.max(initial=0.0))

    def max_absolute(names: tuple[str, ...]) -> float:
        maxima = []
        for name in names:
            values = np.abs(
                np.asarray(candidate[name], dtype=float)
                - np.asarray(baseline[name], dtype=float)
            )
            finite = values[np.isfinite(values)]
            maxima.append(float(finite.max(initial=0.0)))
        return max(maxima, default=0.0)

    return {
        "baseline_product": str(baseline_path.relative_to(baseline_root)),
        "candidate_product": str(candidate_path.relative_to(candidate_root)),
        "detector_count": len(baseline),
        "detector_identity_exact": identity_exact,
        "detector_flags_exact": flags_exact,
        "sensitivity_relative_max": max_relative("sens"),
        "signal_to_noise_relative_max": max_relative("sig2noise"),
        "position_absolute_arcsec_max": max_absolute(("x_t", "y_t")),
        "fwhm_absolute_arcsec_max": max_absolute(("a_fwhm", "b_fwhm")),
        "good_uids": [
            int(uid)
            for uid, flag in zip(baseline_uid, np.asarray(baseline["flag"]))
            if int(flag) == 0
        ],
    }


def layer_rms_relative(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline_values = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate_values = np.asarray(candidate, dtype=np.float64).reshape(-1)
    finite = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    if not finite.any():
        return 0.0
    baseline_values = baseline_values[finite]
    difference = candidate_values[finite] - baseline_values
    baseline_rms = float(np.sqrt(np.mean(baseline_values * baseline_values)))
    difference_rms = float(np.sqrt(np.mean(difference * difference)))
    return difference_rms / max(baseline_rms, 1.0e-300)


def map_metrics(
    baseline_root: Path, candidate_root: Path, good_uids: set[int]
) -> dict[str, Any]:
    baseline_products = {
        path.name: path for path in baseline_root.rglob("*_citlali_flag*.fits")
    }
    candidate_products = {
        path.name: path for path in candidate_root.rglob("*_citlali_flag*.fits")
    }
    product_sets_exact = baseline_products.keys() == candidate_products.keys()

    values: dict[str, dict[str, list[float]]] = {
        quality: {layer: [] for layer in LAYER_NAMES}
        for quality in ("good", "bad")
    }
    for name in sorted(baseline_products.keys() & candidate_products.keys()):
        with fits.open(
            baseline_products[name], memmap=True, lazy_load_hdus=True
        ) as baseline_hdus, fits.open(
            candidate_products[name], memmap=True, lazy_load_hdus=True
        ) as candidate_hdus:
            if len(baseline_hdus) != len(candidate_hdus):
                product_sets_exact = False
                continue
            for baseline_hdu, candidate_hdu in zip(
                baseline_hdus[1:], candidate_hdus[1:]
            ):
                if baseline_hdu.name != candidate_hdu.name:
                    product_sets_exact = False
                    continue
                layer = baseline_hdu.name.split("_", 1)[0].lower()
                match = DET_RE.search(baseline_hdu.name)
                if layer not in LAYER_NAMES or match is None:
                    continue
                uid = int(match.group(1))
                quality = "good" if uid in good_uids else "bad"
                values[quality][layer].append(
                    layer_rms_relative(baseline_hdu.data, candidate_hdu.data)
                )

    summary: dict[str, Any] = {"product_sets_exact": product_sets_exact}
    for quality, layers in values.items():
        for layer, layer_values in layers.items():
            array = np.asarray(layer_values, dtype=float)
            prefix = f"{quality}_{layer}_rms_relative"
            summary[f"{prefix}_count"] = int(array.size)
            summary[f"{prefix}_max"] = float(array.max(initial=0.0))
            summary[f"{prefix}_p99"] = (
                float(np.quantile(array, 0.99)) if array.size else 0.0
            )
    return summary


def evaluate(metrics: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    requirements = profile["requirements"]
    thresholds = profile["thresholds"]
    apt = metrics["apt"]
    maps = metrics["maps"]
    for name in (
        "detector_identity_exact",
        "detector_flags_exact",
        "product_sets_exact",
    ):
        value = apt.get(name) if name in apt else maps.get(name)
        if requirements[name] and value is not True:
            failures.append(f"{name}: required true, got {value!r}")

    direct_metrics = (
        "sensitivity_relative_max",
        "signal_to_noise_relative_max",
        "position_absolute_arcsec_max",
        "fwhm_absolute_arcsec_max",
    )
    for name in direct_metrics:
        if apt[name] > thresholds[name]:
            failures.append(f"{name}: {apt[name]:.8g} > {thresholds[name]:.8g}")

    for quality in ("good", "bad"):
        for layer in LAYER_NAMES:
            metric = f"{quality}_{layer}_rms_relative_max"
            threshold = f"{quality}_{layer}_rms_relative_max"
            if maps[metric] > thresholds[threshold]:
                failures.append(
                    f"{metric}: {maps[metric]:.8g} > {thresholds[threshold]:.8g}")
    for layer in LAYER_NAMES:
        metric = f"good_{layer}_rms_relative_p99"
        threshold = thresholds["good_layer_rms_relative_p99_max"]
        if maps[metric] > threshold:
            failures.append(f"{metric}: {maps[metric]:.8g} > {threshold:.8g}")
    return failures


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Beammap Scientific Equivalence",
        "",
        f"- Profile: `{result['profile_id']}`",
        f"- Baseline: `{result['baseline']}`",
        f"- Candidate: `{result['candidate']}`",
        f"- Verdict: **{result['verdict']}**",
        "",
        "## APT Metrics",
        "",
    ]
    for key, value in result["metrics"]["apt"].items():
        if key != "good_uids":
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Map Metrics", ""])
    for key, value in result["metrics"]["maps"].items():
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
        default=Path("validation/profiles/beammap_scientific_equivalence_v1.json"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported Beammap equivalence profile schema")
    apt = apt_metrics(args.baseline, args.candidate)
    maps = map_metrics(args.baseline, args.candidate, set(apt["good_uids"]))
    failures = evaluate({"apt": apt, "maps": maps}, profile)
    apt.pop("good_uids")
    result = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "metrics": {"apt": apt, "maps": maps},
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
