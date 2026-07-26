#!/usr/bin/env python3
"""Stratify the 108-pointing population before fruit-loop experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.stats import rankdata


ARRAYS = ("a1100", "a1400", "a2000")
EXISTING_FRUIT_LOOP_OBSNUMS = {
    133410, 144176, 148434, 151718, 153481,
}
INJECTION_CANDIDATES = {
    133410: "normal_existing_checkpoint_v2",
    151718: "marginal_high_tau",
    142578: "stress_representative",
}
BADNESS_FIELDS = (
    "low_fit_s2n_badness",
    "low_amplitude_to_background_badness",
    "high_roughness_badness",
    "fwhm_kernel_mismatch_badness",
    "axis_ratio_departure_badness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hero-metrics", required=True, type=Path)
    parser.add_argument("--kernel-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def badness_percentile(
    values: np.ndarray, *, higher_is_better: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.ones(values.shape, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    bad_direction = -values[finite] if higher_is_better else values[finite]
    if finite.sum() == 1:
        result[finite] = 0.0
        return result
    result[finite] = (
        rankdata(bad_direction, method="average") - 1.0
    ) / (finite.sum() - 1.0)
    return result


def absolute_log_ratio(numerator: float, denominator: float = 1.0) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        return math.nan
    return abs(math.log(numerator / denominator))


def load_array_rows(
    hero_metrics_path: Path, kernel_metrics_path: Path,
) -> list[dict]:
    hero = Table.read(hero_metrics_path)
    kernels = Table.read(kernel_metrics_path)
    kernels = kernels[np.asarray(kernels["release"]).astype(str) == "rc1"]
    kernel_lookup = {
        (int(row["obsnum"]), str(row["array"])): row
        for row in kernels
    }
    rows = []
    for row in hero:
        obsnum = int(row["obsnum"])
        array = str(row["array"])
        key = (obsnum, array)
        if key not in kernel_lookup:
            raise ValueError(f"missing RC1 kernel metrics for {key}")
        kernel = kernel_lookup[key]
        measured_fwhm = float(row["fit_fwhm_geomean_arcsec"])
        kernel_fwhm = float(kernel["kernel_fit_fwhm_geomean_arcsec"])
        axis_ratio = float(row["fit_fwhm_axis_ratio"])
        rows.append(
            {
                "obsnum": obsnum,
                "source": str(row["source"]),
                "observation_date": str(row["observation_date"]),
                "mjd": float(row["mjd"]),
                "array": array,
                "fit_s2n": float(row["fit_sig2noise"]),
                "amplitude_to_background_sigma": float(
                    row["fit_amplitude_to_background_sigma"]
                ),
                "map_roughness_fraction": float(
                    row["map_roughness_fraction"]
                ),
                "measured_fwhm_arcsec": measured_fwhm,
                "kernel_fwhm_arcsec": kernel_fwhm,
                "measured_to_kernel_fwhm_ratio":
                    measured_fwhm / kernel_fwhm
                    if kernel_fwhm != 0.0 else math.nan,
                "fwhm_kernel_log_mismatch": absolute_log_ratio(
                    measured_fwhm, kernel_fwhm
                ),
                "fit_axis_ratio": axis_ratio,
                "axis_ratio_log_departure": absolute_log_ratio(axis_ratio),
                "fit_x_arcsec": float(row["fit_x_t_arcsec"]),
                "fit_y_arcsec": float(row["fit_y_t_arcsec"]),
                "map_background_sigma_mjy": float(
                    row["map_background_sigma_mjy"]
                ),
                "map_median_weight": float(row["map_median_weight"]),
                "kernel_fit_success": bool(kernel["kernel_fit_success"]),
            }
        )
    expected = len({int(row["obsnum"]) for row in hero}) * len(ARRAYS)
    if len(rows) != expected:
        raise ValueError(
            f"expected exactly three array rows per observation; "
            f"found {len(rows)} rows, expected {expected}"
        )
    return rows


def add_array_badness(rows: list[dict]) -> None:
    definitions = (
        ("fit_s2n", "low_fit_s2n_badness", True),
        (
            "amplitude_to_background_sigma",
            "low_amplitude_to_background_badness",
            True,
        ),
        (
            "map_roughness_fraction",
            "high_roughness_badness",
            False,
        ),
        (
            "fwhm_kernel_log_mismatch",
            "fwhm_kernel_mismatch_badness",
            False,
        ),
        (
            "axis_ratio_log_departure",
            "axis_ratio_departure_badness",
            False,
        ),
    )
    for array in ARRAYS:
        selected = [row for row in rows if row["array"] == array]
        for source, destination, higher_is_better in definitions:
            badness = badness_percentile(
                np.asarray([row[source] for row in selected]),
                higher_is_better=higher_is_better,
            )
            for row, value in zip(selected, badness, strict=True):
                row[destination] = float(value)
        for row in selected:
            row["array_quality_badness"] = float(
                np.mean([row[field] for field in BADNESS_FIELDS])
            )


def observation_rows(array_rows: list[dict]) -> list[dict]:
    rows = []
    obsnums = sorted({int(row["obsnum"]) for row in array_rows})
    for obsnum in obsnums:
        selected = [
            row for row in array_rows if int(row["obsnum"]) == obsnum
        ]
        if {str(row["array"]) for row in selected} != set(ARRAYS):
            raise ValueError(f"incomplete array coverage for obsnum {obsnum}")
        array_badness = np.asarray(
            [row["array_quality_badness"] for row in selected], dtype=float
        )
        x = np.asarray([row["fit_x_arcsec"] for row in selected], dtype=float)
        y = np.asarray([row["fit_y_arcsec"] for row in selected], dtype=float)
        centroid_scatter = float(
            np.hypot(np.std(x), np.std(y))
        )
        rows.append(
            {
                "obsnum": obsnum,
                "source": selected[0]["source"],
                "observation_date": selected[0]["observation_date"],
                "mjd": selected[0]["mjd"],
                "median_array_badness": float(np.median(array_badness)),
                "worst_array_badness": float(np.max(array_badness)),
                "map_quality_badness": float(
                    0.5 * np.median(array_badness)
                    + 0.5 * np.max(array_badness)
                ),
                "cross_array_centroid_rms_arcsec": centroid_scatter,
                "minimum_fit_s2n": float(
                    min(row["fit_s2n"] for row in selected)
                ),
                "minimum_amplitude_to_background_sigma": float(
                    min(
                        row["amplitude_to_background_sigma"]
                        for row in selected
                    )
                ),
                "maximum_fwhm_kernel_fractional_mismatch": float(
                    max(
                        abs(row["measured_to_kernel_fwhm_ratio"] - 1.0)
                        for row in selected
                    )
                ),
                "maximum_axis_ratio": float(
                    max(row["fit_axis_ratio"] for row in selected)
                ),
                "maximum_map_roughness_fraction": float(
                    max(row["map_roughness_fraction"] for row in selected)
                ),
            }
        )
    map_rank = badness_percentile(
        np.asarray([row["map_quality_badness"] for row in rows]),
        higher_is_better=False,
    )
    centroid_rank = badness_percentile(
        np.asarray(
            [row["cross_array_centroid_rms_arcsec"] for row in rows]
        ),
        higher_is_better=False,
    )
    for row, map_value, centroid_value in zip(
        rows, map_rank, centroid_rank, strict=True,
    ):
        row["map_quality_rank_badness"] = float(map_value)
        row["centroid_scatter_rank_badness"] = float(centroid_value)
        row["quality_score"] = float(
            0.8 * map_value + 0.2 * centroid_value
        )

    rows.sort(key=lambda row: (row["quality_score"], row["obsnum"]))
    normal_count = len(rows) // 2
    marginal_count = math.ceil(len(rows) * 0.35)
    for rank, row in enumerate(rows, start=1):
        row["quality_rank"] = rank
        if rank <= normal_count:
            row["quality_stratum"] = "normal"
        elif rank <= normal_count + marginal_count:
            row["quality_stratum"] = "marginal"
        else:
            row["quality_stratum"] = "stress"
        obsnum = int(row["obsnum"])
        row["existing_fruit_loop_sequence"] = (
            obsnum in EXISTING_FRUIT_LOOP_OBSNUMS
        )
        row["injection_candidate"] = obsnum in INJECTION_CANDIDATES
        row["injection_candidate_reason"] = INJECTION_CANDIDATES.get(
            obsnum, ""
        )
    return rows


def sentinel_obsnums(rows: list[dict]) -> dict[int, str]:
    selected: dict[int, str] = {}
    existing = {
        int(row["obsnum"]) for row in rows
        if row["existing_fruit_loop_sequence"]
    }
    for stratum in ("normal", "marginal", "stress"):
        candidates = [
            row for row in rows
            if row["quality_stratum"] == stratum
            and int(row["obsnum"]) not in existing
        ]
        for fraction, label in (
            (0.15, "lower_badness_anchor"),
            (0.50, "median_anchor"),
            (0.85, "higher_badness_anchor"),
        ):
            index = round(fraction * (len(candidates) - 1))
            row = candidates[index]
            selected[int(row["obsnum"])] = f"{stratum}_{label}"

    represented_sources = {
        str(row["source"]) for row in rows
        if int(row["obsnum"]) in existing | set(selected)
    }
    for source in sorted({str(row["source"]) for row in rows}):
        if source in represented_sources:
            continue
        candidates = [row for row in rows if row["source"] == source]
        source_median = float(
            np.median([item["quality_score"] for item in candidates])
        )
        candidates.sort(
            key=lambda row: abs(
                float(row["quality_score"]) - source_median
            )
        )
        row = candidates[0]
        selected[int(row["obsnum"])] = f"source_coverage_{source}"
    return selected


def population_run_matrix(rows: list[dict]) -> list[dict]:
    sentinels = sentinel_obsnums(rows)
    result = []
    for row in sorted(rows, key=lambda item: int(item["quality_rank"])):
        obsnum = int(row["obsnum"])
        if row["existing_fruit_loop_sequence"]:
            phase = "sentinel_extension_first"
            selection_reason = (
                "existing_study_rerun_for_common_binary_and_10_iterations"
            )
        elif obsnum in sentinels:
            phase = "sentinel_extension_first"
            selection_reason = sentinels[obsnum]
        else:
            phase = "population_after_sentinel_gate"
            selection_reason = "complete_108_observation_population"
        result.append(
            {
                "obsnum": obsnum,
                "source": row["source"],
                "quality_rank": row["quality_rank"],
                "quality_stratum": row["quality_stratum"],
                "quality_score": row["quality_score"],
                "phase": phase,
                "selection_reason": selection_reason,
                "historical_five_iteration_evidence":
                    row["existing_fruit_loop_sequence"],
                "real_source_iterations_requested": 10,
                "save_all_iterations": True,
                "checkpoint_schema_required": "v2",
                "unique_output_workspace_required": True,
                "controlled_injection_candidate":
                    row["injection_candidate"],
                "controlled_injection_reason":
                    row["injection_candidate_reason"],
                "unity_status": "not_requested",
            }
        )
    return result


def stratum_summary(rows: list[dict]) -> list[dict]:
    result = []
    for stratum in ("normal", "marginal", "stress"):
        selected = [
            row for row in rows if row["quality_stratum"] == stratum
        ]
        result.append(
            {
                "quality_stratum": stratum,
                "observations": len(selected),
                "quality_rank_min": min(
                    row["quality_rank"] for row in selected
                ),
                "quality_rank_max": max(
                    row["quality_rank"] for row in selected
                ),
                "quality_score_min": min(
                    row["quality_score"] for row in selected
                ),
                "quality_score_median": float(
                    np.median(
                        [row["quality_score"] for row in selected]
                    )
                ),
                "quality_score_max": max(
                    row["quality_score"] for row in selected
                ),
                "existing_fruit_loop_observations": sum(
                    bool(row["existing_fruit_loop_sequence"])
                    for row in selected
                ),
                "sources": ";".join(
                    sorted({str(row["source"]) for row in selected})
                ),
            }
        )
    return result


def plot_quality_rank(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "3c273": "#1f77b4",
        "Uranus": "#ff7f0e",
        "Neptune": "#2ca02c",
        "3c279": "#d62728",
        "3c345": "#9467bd",
        "3c84": "#8c564b",
    }
    fig, axis = plt.subplots(figsize=(12, 6))
    for source in sorted({str(row["source"]) for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        axis.scatter(
            [row["quality_rank"] for row in selected],
            [row["quality_score"] for row in selected],
            label=source,
            color=colors.get(source),
            s=34,
            alpha=0.85,
        )
    axis.axvline(54.5, color="0.35", linestyle="--", linewidth=1)
    axis.axvline(92.5, color="0.35", linestyle="--", linewidth=1)
    axis.set_xlabel("Quality rank (lower is better)")
    axis.set_ylabel("Composite quality badness score")
    axis.set_title(
        "108 RC1 pointings: independent quality baseline"
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "quality_rank_by_source.png", dpi=180)
    plt.close(fig)


def plot_quality_plane(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "normal": "#2ca02c",
        "marginal": "#ffbf00",
        "stress": "#d62728",
    }
    fig, axis = plt.subplots(figsize=(8.5, 7))
    for stratum in ("normal", "marginal", "stress"):
        selected = [
            row for row in rows if row["quality_stratum"] == stratum
        ]
        axis.scatter(
            [row["worst_array_badness"] for row in selected],
            [
                row["cross_array_centroid_rms_arcsec"]
                for row in selected
            ],
            label=stratum,
            color=colors[stratum],
            s=38,
            alpha=0.8,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Worst-array diagnostic badness")
    axis.set_ylabel("Cross-array centroid RMS (arcsec)")
    axis.set_title("Quality plane; strata are descriptive quantiles")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "quality_plane.png", dpi=180)
    plt.close(fig)


def plot_component_heatmap(
    array_rows: list[dict], observation_rows_: list[dict], output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(
        observation_rows_, key=lambda row: int(row["quality_rank"])
    )
    lookup = {
        (int(row["obsnum"]), str(row["array"])): row
        for row in array_rows
    }
    columns = []
    labels = []
    short_names = (
        ("low_fit_s2n_badness", "low S/N"),
        (
            "low_amplitude_to_background_badness",
            "low contrast",
        ),
        ("high_roughness_badness", "roughness"),
        ("fwhm_kernel_mismatch_badness", "FWHM/kernel"),
        ("axis_ratio_departure_badness", "axis ratio"),
    )
    for array in ARRAYS:
        for field, label in short_names:
            columns.append(
                [
                    lookup[(int(row["obsnum"]), array)][field]
                    for row in ordered
                ]
            )
            labels.append(f"{array} {label}")
    matrix = np.asarray(columns, dtype=float).T
    fig, axis = plt.subplots(figsize=(14, 10))
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest",
        cmap="magma_r", vmin=0.0, vmax=1.0,
    )
    axis.set_xlabel("Independent baseline diagnostic")
    axis.set_ylabel("Observation quality rank")
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_yticks(
        [0, 26, 53, 72, 91, 107],
        [1, 27, 54, 73, 92, 108],
    )
    axis.axhline(53.5, color="cyan", linewidth=1)
    axis.axhline(91.5, color="cyan", linewidth=1)
    fig.colorbar(image, ax=axis, label="Badness percentile")
    axis.set_title("Quality components ordered by composite rank")
    fig.tight_layout()
    fig.savefig(output / "quality_component_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = load_array_rows(args.hero_metrics, args.kernel_metrics)
    add_array_badness(arrays)
    observations = observation_rows(arrays)
    matrix = population_run_matrix(observations)
    summary = stratum_summary(observations)
    write_csv(args.output / "array_quality_metrics.csv", arrays)
    write_csv(
        args.output / "observation_quality_inventory.csv", observations
    )
    write_csv(args.output / "quality_stratum_summary.csv", summary)
    write_csv(args.output / "population_run_matrix.csv", matrix)
    plot_quality_rank(observations, args.output)
    plot_quality_plane(observations, args.output)
    plot_component_heatmap(arrays, observations, args.output)
    manifest = {
        "schema_version": "citlali-fruit-loop-quality-stratification-v1",
        "hero_metrics_path": str(args.hero_metrics),
        "hero_metrics_sha256": sha256(args.hero_metrics),
        "kernel_metrics_path": str(args.kernel_metrics),
        "kernel_metrics_sha256": sha256(args.kernel_metrics),
        "observation_count": len(observations),
        "array_row_count": len(arrays),
        "quality_definition": {
            "array_components": list(BADNESS_FIELDS),
            "array_component_weighting": "equal",
            "map_quality_badness":
                "0.5 * median array badness + 0.5 * worst array badness",
            "quality_score":
                "0.8 * map-quality rank + 0.2 * cross-array-centroid "
                "scatter rank",
            "strata":
                "rank quantiles: first 50% normal, next 35% marginal, "
                "last 15% stress",
            "interpretation":
                "descriptive experiment-design labels, not data rejection",
        },
        "existing_fruit_loop_observation_count": sum(
            bool(row["existing_fruit_loop_sequence"])
            for row in observations
        ),
        "files": sorted(
            {path.name for path in args.output.iterdir()}
            | {"manifest.json"}
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(observations)}-observation quality stratification "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
