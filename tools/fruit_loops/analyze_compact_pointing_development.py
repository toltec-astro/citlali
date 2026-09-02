#!/usr/bin/env python3
"""Summarize one development-only compact-source FRUIT trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits

from tools.fruit_loops.analyze_population_stage import source_free_map_metrics
from tools.fruit_loops.compare_feedback_ablation import reduction_rows


ARRAYS = ("a1100", "a1400", "a2000")
ARRAY_COLORS = {
    "a1100": "#1f77b4",
    "a1400": "#ff7f0e",
    "a2000": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced",
        required=True,
        type=Path,
        help="Directory containing contiguous redu00, redu01, ... products.",
    )
    parser.add_argument("--obsnum", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--control-id",
        required=True,
        help="Human-readable identity for the operational control run.",
    )
    parser.add_argument(
        "--software-id",
        required=True,
        help="Exact executable/version identity reported by the run.",
    )
    parser.add_argument(
        "--executable-snapshot",
        required=True,
        type=Path,
        help="Preserved executable used to produce the control run.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def map_path(reduced: Path, obsnum: int, iteration: int, array: str) -> Path:
    return (
        reduced
        / f"redu{iteration:02d}"
        / str(obsnum)
        / "raw"
        / f"toltec_commissioning_{array}_pointing_{obsnum}_citlali.fits"
    )


def table_path(reduced: Path, obsnum: int, iteration: int) -> Path:
    return (
        reduced
        / f"redu{iteration:02d}"
        / str(obsnum)
        / "raw"
        / f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv"
    )


def scalar_map_and_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with fits.open(path, memmap=True) as hdul:
        signal = np.asarray(hdul["signal_I"].data, dtype=float).squeeze()
        coverage = (
            np.asarray(hdul["coverage_bool_I"].data).squeeze() > 0.5
        )
        image_header = hdul["signal_I"].header
        primary_header = hdul[0].header
        geometry = {
            "apt_reference_major_fwhm_arcsec": float(
                primary_header["BMAJ"]
            ),
            "apt_reference_minor_fwhm_arcsec": float(
                primary_header["BMIN"]
            ),
            "crpix1": float(image_header["CRPIX1"]),
            "crpix2": float(image_header["CRPIX2"]),
            "crval1": float(image_header["CRVAL1"]),
            "crval2": float(image_header["CRVAL2"]),
            "cdelt1": float(image_header["CDELT1"]),
            "cdelt2": float(image_header["CDELT2"]),
        }
    return signal, coverage, geometry


def radial_mask(
    shape: tuple[int, int], geometry: dict, center_x: float, center_y: float,
    inner_radius_arcsec: float, outer_radius_arcsec: float,
) -> np.ndarray:
    ny, nx = shape
    x = (
        np.arange(nx, dtype=float) + 1.0 - geometry["crpix1"]
    ) * geometry["cdelt1"] + geometry["crval1"]
    y = (
        np.arange(ny, dtype=float) + 1.0 - geometry["crpix2"]
    ) * geometry["cdelt2"] + geometry["crval2"]
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx - center_x, yy - center_y)
    return (radius >= inner_radius_arcsec) & (radius <= outer_radius_arcsec)


def relative_delta_rms(
    current: np.ndarray, previous: np.ndarray, selected: np.ndarray,
) -> float:
    valid = selected & np.isfinite(current) & np.isfinite(previous)
    if not valid.any():
        return math.nan
    denominator = float(np.sqrt(np.mean(np.square(previous[valid]))))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    numerator = float(
        np.sqrt(np.mean(np.square(current[valid] - previous[valid])))
    )
    return numerator / denominator


def prepare_rows(
    reduced: Path, obsnum: int, control_id: str,
) -> tuple[list[dict], list[Path]]:
    extracted = reduction_rows(control_id, reduced, obsnum)
    extracted.sort(key=lambda row: (str(row["array"]), int(row["iteration"])))
    iterations = sorted({int(row["iteration"]) for row in extracted})
    expected = list(range(len(iterations)))
    if iterations != expected:
        raise ValueError(f"expected contiguous iterations {expected}; got {iterations}")
    if len(extracted) != len(ARRAYS) * len(iterations):
        raise ValueError("one or more iteration/array products are missing")

    result: list[dict] = []
    inputs: list[Path] = []
    groups = {
        array: [row for row in extracted if row["array"] == array]
        for array in ARRAYS
    }
    for array, group in groups.items():
        group.sort(key=lambda row: int(row["iteration"]))
        seed = group[0]
        seed_x = float(seed["x_t_arcsec"])
        seed_y = float(seed["y_t_arcsec"])
        seed_amplitude = float(seed["amplitude"])
        seed_fit_snr = float(seed["fit_sig2noise"])
        previous_signal: np.ndarray | None = None
        for index, base in enumerate(group):
            iteration = int(base["iteration"])
            fits_path = map_path(reduced, obsnum, iteration, array)
            pointing_path = table_path(reduced, obsnum, iteration)
            inputs.extend((fits_path, pointing_path))
            signal, coverage, geometry = scalar_map_and_geometry(fits_path)
            source_x = float(base["x_t_arcsec"])
            source_y = float(base["y_t_arcsec"])
            source_major = float(base["major_fwhm_arcsec"])
            source_minor = float(base["minor_fwhm_arcsec"])
            kernel_major = float(base["kernel_major_fwhm_arcsec"])
            kernel_minor = float(base["kernel_minor_fwhm_arcsec"])
            apt_major = geometry["apt_reference_major_fwhm_arcsec"]
            apt_minor = geometry["apt_reference_minor_fwhm_arcsec"]
            source_geomean = math.sqrt(source_major * source_minor)
            kernel_geomean = math.sqrt(kernel_major * kernel_minor)
            apt_geomean = math.sqrt(apt_major * apt_minor)
            background = source_free_map_metrics(
                fits_path,
                fit_x_arcsec=source_x,
                fit_y_arcsec=source_y,
                kernel_major_fwhm_arcsec=kernel_major,
                kernel_minor_fwhm_arcsec=kernel_minor,
                include_empirical_point_source_snr=False,
            )
            core_mask = radial_mask(
                signal.shape, geometry, seed_x, seed_y, 0.0, 40.0
            ) & coverage
            background_mask = radial_mask(
                signal.shape, geometry, seed_x, seed_y, 40.0, 120.0
            ) & coverage
            if previous_signal is None:
                core_delta = math.nan
                background_delta = math.nan
                centroid_step = math.nan
            else:
                core_delta = relative_delta_rms(
                    signal, previous_signal, core_mask
                )
                background_delta = relative_delta_rms(
                    signal, previous_signal, background_mask
                )
                previous = group[index - 1]
                centroid_step = math.hypot(
                    source_x - float(previous["x_t_arcsec"]),
                    source_y - float(previous["y_t_arcsec"]),
                )
            result.append(
                {
                    "control_id": control_id,
                    "obsnum": obsnum,
                    "source": base["source"],
                    "iteration": iteration,
                    "array": array,
                    "amplitude_mjy_beam": float(base["amplitude"]),
                    "amplitude_error_mjy_beam": float(
                        base["amplitude_error"]
                    ),
                    "amplitude_ratio_seed": float(base["amplitude"])
                    / seed_amplitude,
                    "fit_sig2noise": float(base["fit_sig2noise"]),
                    "fit_sig2noise_ratio_seed": float(base["fit_sig2noise"])
                    / seed_fit_snr,
                    "peak_over_full_map_rms": float(
                        base["legacy_peak_over_full_map_rms"]
                    ),
                    "source_major_fwhm_arcsec": source_major,
                    "source_minor_fwhm_arcsec": source_minor,
                    "source_geomean_fwhm_arcsec": source_geomean,
                    "source_axis_ratio": source_major / source_minor,
                    "kernel_major_fwhm_arcsec": kernel_major,
                    "kernel_minor_fwhm_arcsec": kernel_minor,
                    "kernel_geomean_fwhm_arcsec": kernel_geomean,
                    "source_over_kernel_fwhm": source_geomean
                    / kernel_geomean,
                    "apt_reference_major_fwhm_arcsec": apt_major,
                    "apt_reference_minor_fwhm_arcsec": apt_minor,
                    "apt_reference_geomean_fwhm_arcsec": apt_geomean,
                    "source_over_apt_reference_fwhm": source_geomean
                    / apt_geomean,
                    "x_t_arcsec": source_x,
                    "y_t_arcsec": source_y,
                    "centroid_step_arcsec": centroid_step,
                    "centroid_shift_from_seed_arcsec": math.hypot(
                        source_x - seed_x, source_y - seed_y
                    ),
                    "background_median_mjy_beam": background[
                        "map_background_median_mjy"
                    ],
                    "background_sigma_mjy_beam": background[
                        "map_background_sigma_mjy"
                    ],
                    "pixel_roughness_mjy_beam": background[
                        "map_pixel_roughness_mjy"
                    ],
                    "pixel_roughness_over_background_sigma": background[
                        "map_roughness_fraction"
                    ],
                    "core_0_40arcsec_successive_delta_relative_rms":
                        core_delta,
                    "background_40_120arcsec_successive_delta_relative_rms":
                        background_delta,
                }
            )
            previous_signal = signal
    result.sort(key=lambda row: (int(row["iteration"]), str(row["array"])))
    return result, sorted(set(inputs))


def add_seed_ratios(rows: list[dict]) -> None:
    for array in ARRAYS:
        group = sorted(
            (row for row in rows if row["array"] == array),
            key=lambda row: int(row["iteration"]),
        )
        seed_background = float(group[0]["background_sigma_mjy_beam"])
        for row in group:
            row["background_sigma_ratio_seed"] = (
                float(row["background_sigma_mjy_beam"]) / seed_background
            )


def plot_summary(rows: list[dict], output: Path, obsnum: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), constrained_layout=True)
    panels = axes.ravel()
    for array in ARRAYS:
        group = sorted(
            (row for row in rows if row["array"] == array),
            key=lambda row: int(row["iteration"]),
        )
        iteration = np.asarray([row["iteration"] for row in group], dtype=int)
        color = ARRAY_COLORS[array]
        panels[0].plot(
            iteration,
            [row["amplitude_ratio_seed"] for row in group],
            marker="o",
            color=color,
            label=array,
        )
        panels[1].plot(
            iteration,
            [row["source_over_kernel_fwhm"] for row in group],
            marker="o",
            color=color,
            label=f"{array}: processed kernel",
        )
        panels[1].plot(
            iteration,
            [row["source_over_apt_reference_fwhm"] for row in group],
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"{array}: APT header",
        )
        panels[2].plot(
            iteration,
            [row["centroid_shift_from_seed_arcsec"] for row in group],
            marker="o",
            color=color,
            label=array,
        )
        panels[3].plot(
            iteration,
            [row["background_sigma_ratio_seed"] for row in group],
            marker="o",
            color=color,
            label=array,
        )

    panels[0].axhline(1.0, color="0.5", linewidth=1.0)
    panels[0].set_title("Recovered peak amplitude")
    panels[0].set_ylabel("amplitude / iteration-0 amplitude")
    panels[0].legend(frameon=False)

    panels[1].axhline(1.0, color="0.35", linewidth=1.0)
    panels[1].set_title("Recovered Gaussian width versus references")
    panels[1].set_ylabel("geometric-mean FWHM ratio")
    panels[1].legend(frameon=False, fontsize=8, ncol=2)

    panels[2].set_title("Centroid stability")
    panels[2].set_ylabel("shift from iteration 0 (arcsec)")

    panels[3].axhline(1.0, color="0.5", linewidth=1.0)
    panels[3].set_title("Background structure, 40–120 arcsec")
    panels[3].set_ylabel("robust sigma / iteration-0 sigma")

    for axis in panels:
        axis.set_xlabel("saved FRUIT iteration")
        axis.set_xticks(sorted({int(row["iteration"]) for row in rows}))
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Pointing {obsnum}: established Citlali FRUIT trajectory\n"
        "exploratory development evidence — not qualification",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, analysis_inputs = prepare_rows(
        args.reduced.resolve(), args.obsnum, args.control_id
    )
    add_seed_ratios(rows)
    metrics_path = args.output / "iteration_metrics.csv"
    plot_path = args.output / f"point_{args.obsnum}_iteration_summary.png"
    write_csv(metrics_path, rows)
    plot_summary(rows, plot_path, args.obsnum)

    terminal = args.reduced / f"redu{max(row['iteration'] for row in rows):02d}"
    supporting_inputs = [
        terminal / "citlali_merged_config.yaml",
        terminal / "runtime_provenance.yaml",
        terminal / "config_source_manifest.yaml",
    ]
    for path in supporting_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis_inputs.extend(supporting_inputs)
    config = yaml.safe_load((terminal / "citlali_merged_config.yaml").read_text())
    manifest = {
        "schema_version": "sci-fruit-compact-development-analysis-v1",
        "role": "exploratory-development-only",
        "qualification_use_authorized": False,
        "control_id": args.control_id,
        "control_executable": {
            "path": str(args.executable_snapshot.resolve()),
            "size_bytes": args.executable_snapshot.stat().st_size,
            "sha256": sha256(args.executable_snapshot),
            "version": args.software_id,
        },
        "obsnum": args.obsnum,
        "source_root": str(args.reduced.resolve()),
        "iteration_count": len({int(row["iteration"]) for row in rows}),
        "mapmaking_method": config["mapmaking"]["method"],
        "fruit_loop_configuration": {
            "max_iters": config["timestream"]["fruit_loops"]["max_iters"],
            "type": config["timestream"]["fruit_loops"]["type"],
            "mode": config["timestream"]["fruit_loops"]["mode"],
            "array_flux_limit": config["timestream"]["fruit_loops"][
                "array_flux_limit"
            ],
            "sig2noise_limit": config["timestream"]["fruit_loops"][
                "sig2noise_limit"
            ],
        },
        "reference_interpretation": {
            "processed_kernel": (
                "primary same-run response comparator; not independent "
                "diffraction truth"
            ),
            "apt_header_beam": (
                "secondary reduction-carried beam geometry; not independent "
                "diffraction truth"
            ),
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(analysis_inputs))
        ],
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (metrics_path, plot_path)
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} metric rows to {metrics_path}")
    print(f"wrote summary plot to {plot_path}")
    print(f"wrote input/output manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
