#!/usr/bin/env python3
"""Reconstruct unthresholded split-direction naive maps from a full PTC TOD.

This SCI-ALIGN-001 follow-up applies Citlali's detector-grouped naive mapper
rules to one retained detector without the later coverage-support threshold.
It is intentionally descriptive: the full PTC is a separate single-pass
replay, so the result is not represented as a reversible de-thresholding of
the final multi-iteration Beammap product.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from analyze_sci_align_001_ptc_sampling import (  # noqa: E402
    PtcDetector,
    cxx_llround,
    load_ptc_detector,
    map_tables,
    output_checksums,
    pixel_size_rad_from_wcs,
    sha256_file,
    write_json,
)
from analyze_sci_align_001_selected_sampling_join import (  # noqa: E402
    classify_full_ptc_scan_direction,
    load_full_ptc_append_bounds,
    load_map_pointing,
)
from analyze_sci_align_001_split_direction_transfer import (  # noqa: E402
    fit_gaussian_core,
)
from render_sci_align_001_split_direction_beammaps import (  # noqa: E402
    ARRAY_IDS,
    MODES,
    ContractError,
    FitsProduct,
    crop_mask,
    discover_fits,
    discover_raw_dir,
    image_coordinates,
    masked_signal,
    normalized_profile,
    pixel_edges,
    robust_limits,
    row_values,
    scan_summary,
    spatial_wcs,
    wcs_signature,
)


RAD_TO_ARCSEC = 206264.80624709636


@dataclass(frozen=True)
class RegistryScan:
    scan_id: int
    sample_count: int
    direction: str


@dataclass
class ReconstructedMap:
    mode: str
    weighted_signal: np.ndarray
    weight: np.ndarray
    hit_count: np.ndarray
    signal: np.ndarray
    scan_count: int
    selected_sample_count: int
    accepted_sample_count: int
    flagged_sample_count: int
    nonfinite_sample_count: int
    outside_sample_count: int


def load_registry(path: Path) -> dict[int, RegistryScan]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"direction registry is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_index", "sample_count", "direction", "mode"}
    if not rows or not required.issubset(rows[0]):
        raise ContractError("direction registry is empty or lacks required columns")
    result: dict[int, RegistryScan] = {}
    for row in rows:
        if row["mode"] != "all":
            raise ContractError(f"direction registry mode is {row['mode']!r}, not 'all'")
        scan_id = int(row["scan_index"]) + 1
        direction = row["direction"]
        if direction not in {"left", "right"}:
            raise ContractError(f"scan {scan_id} has invalid direction {direction!r}")
        if scan_id in result:
            raise ContractError(f"direction registry duplicates scan {scan_id}")
        result[scan_id] = RegistryScan(
            scan_id=scan_id,
            sample_count=int(row["sample_count"]),
            direction=direction,
        )
    return result


def empty_map(mode: str, shape: tuple[int, int]) -> ReconstructedMap:
    return ReconstructedMap(
        mode=mode,
        weighted_signal=np.zeros(shape, dtype=float),
        weight=np.zeros(shape, dtype=float),
        hit_count=np.zeros(shape, dtype=np.int64),
        signal=np.full(shape, np.nan, dtype=float),
        scan_count=0,
        selected_sample_count=0,
        accepted_sample_count=0,
        flagged_sample_count=0,
        nonfinite_sample_count=0,
        outside_sample_count=0,
    )


def reconstruct_unthresholded_maps(
    ptc: PtcDetector,
    bounds: np.ndarray,
    registry: dict[int, RegistryScan],
    map_lat: np.ndarray,
    map_lon: np.ndarray,
    shape: tuple[int, int],
    pixel_size_rad: float,
) -> dict[str, ReconstructedMap]:
    if bounds.shape != (ptc.output_scan_index.size, 2):
        raise ContractError("scan bounds do not match the full-PTC scan identity axis")
    if map_lat.shape != ptc.signal.shape or map_lon.shape != ptc.signal.shape:
        raise ContractError("map pointing does not match the detector signal timebase")
    if not math.isfinite(pixel_size_rad) or pixel_size_rad <= 0.0:
        raise ContractError("map pixel size is not finite and positive")
    if ptc.apt_flag != 0:
        raise ContractError(f"uid={ptc.uid} has nonzero full-PTC APT flag {ptc.apt_flag}")

    result = {mode: empty_map(mode, shape) for mode in MODES}
    seen: set[int] = set()
    for scan_row, ((start, end), raw_scan_id) in enumerate(
        zip(bounds, ptc.output_scan_index, strict=True)
    ):
        scan_id = int(raw_scan_id)
        if scan_id in seen:
            raise ContractError(f"full PTC duplicates scan identity {scan_id}")
        seen.add(scan_id)
        if scan_id not in registry:
            raise ContractError(f"full PTC scan {scan_id} is absent from the registry")
        reg = registry[scan_id]
        start_i, end_i = int(start), int(end)
        n_samples = end_i - start_i + 1
        if n_samples != reg.sample_count:
            raise ContractError(
                f"scan {scan_id} full-PTC/registry length mismatch: "
                f"{n_samples} != {reg.sample_count}"
            )
        derived_direction = classify_full_ptc_scan_direction(
            ptc, start_i, end_i, scan_id
        )
        if derived_direction != reg.direction:
            raise ContractError(
                f"scan {scan_id} trajectory direction {derived_direction} "
                f"disagrees with registry {reg.direction}"
            )
        scan_weight = float(ptc.weights[scan_row])
        if not math.isfinite(scan_weight) or scan_weight <= 0.0:
            raise ContractError(
                f"scan {scan_id} has non-finite or nonpositive detector weight"
            )

        slc = slice(start_i, end_i + 1)
        signal = ptc.signal[slc]
        flags = ptc.flags[slc]
        row_float = map_lat[slc] / pixel_size_rad + (shape[0] - 1) / 2.0
        col_float = map_lon[slc] / pixel_size_rad + (shape[1] - 1) / 2.0
        finite_position = np.isfinite(row_float) & np.isfinite(col_float)
        row = np.full(n_samples, -1, dtype=np.int64)
        col = np.full(n_samples, -1, dtype=np.int64)
        row[finite_position] = cxx_llround(row_float[finite_position])
        col[finite_position] = cxx_llround(col_float[finite_position])
        inside = (
            finite_position
            & (row >= 0)
            & (row < shape[0])
            & (col >= 0)
            & (col < shape[1])
        )
        finite_signal = np.isfinite(signal)
        good_flag = flags == 0
        accepted = good_flag & finite_signal & inside

        for mode in ("standard", reg.direction):
            target = result[mode]
            target.scan_count += 1
            target.selected_sample_count += n_samples
            target.accepted_sample_count += int(np.sum(accepted))
            target.flagged_sample_count += int(np.sum(~good_flag))
            target.nonfinite_sample_count += int(np.sum(~finite_signal))
            target.outside_sample_count += int(np.sum(~inside))
            rr, cc = row[accepted], col[accepted]
            values = signal[accepted]
            np.add.at(target.weight, (rr, cc), scan_weight)
            np.add.at(target.weighted_signal, (rr, cc), values * scan_weight)
            np.add.at(target.hit_count, (rr, cc), 1)

    if seen != set(registry):
        missing = sorted(set(registry) - seen)
        raise ContractError(f"full PTC does not cover registry scans: {missing[:20]}")
    for mode, target in result.items():
        positive = np.isfinite(target.weight) & (target.weight > 0.0)
        if target.scan_count == 0 or not np.any(positive):
            raise ContractError(f"{mode} reconstruction has no positive support")
        target.signal[positive] = (
            target.weighted_signal[positive] / target.weight[positive]
        )
    return result


def fit_reconstructions(
    maps: dict[str, ReconstructedMap],
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    fit_half_width: float,
) -> dict[str, dict[str, Any]]:
    fits: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        fit = fit_gaussian_core(
            maps[mode].signal, x, y, center_x, center_y, fit_half_width
        )
        if fit.get("status") != "success":
            raise ContractError(f"{mode} unthresholded fit failed: {fit}")
        fits[mode] = fit
    return fits


def displacement_row(
    family: str,
    centers: dict[str, tuple[float, float]],
    scan: Any,
) -> dict[str, Any]:
    left_x, left_y = centers["left"]
    right_x, right_y = centers["right"]
    dx, dy = right_x - left_x, right_y - left_y
    parallel = dx * scan.axis_x + dy * scan.axis_y
    perpendicular = dx * scan.cross_x + dy * scan.cross_y
    rate_difference = scan.right_rate_arcsec_s - scan.left_rate_arcsec_s
    if not math.isfinite(rate_difference) or rate_difference <= 0.0:
        raise ContractError("registry scan-rate difference is not finite and positive")
    return {
        "family": family,
        "left_x_arcsec": left_x,
        "left_y_arcsec": left_y,
        "right_x_arcsec": right_x,
        "right_y_arcsec": right_y,
        "delta_x_right_minus_left_arcsec": dx,
        "delta_y_right_minus_left_arcsec": dy,
        "delta_parallel_right_minus_left_arcsec": parallel,
        "delta_perpendicular_right_minus_left_arcsec": perpendicular,
        "left_rate_arcsec_per_sec": scan.left_rate_arcsec_s,
        "right_rate_arcsec_per_sec": scan.right_rate_arcsec_s,
        "rate_difference_arcsec_per_sec": rate_difference,
        "timing_equivalent_ms": 1000.0 * parallel / rate_difference,
    }


def metrics_rows(
    maps: dict[str, ReconstructedMap], fits: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        target, fit = maps[mode], fits[mode]
        rows.append({
            "mode": mode,
            "scan_count": target.scan_count,
            "selected_sample_count": target.selected_sample_count,
            "accepted_sample_count": target.accepted_sample_count,
            "flagged_sample_count": target.flagged_sample_count,
            "nonfinite_signal_sample_count": target.nonfinite_sample_count,
            "outside_map_sample_count": target.outside_sample_count,
            "positive_weight_pixel_count": int(np.sum(target.weight > 0.0)),
            "fit_x_arcsec": fit["x_arcsec"],
            "fit_y_arcsec": fit["y_arcsec"],
            "fit_major_fwhm_arcsec": fit["major_fwhm_arcsec"],
            "fit_minor_fwhm_arcsec": fit["minor_fwhm_arcsec"],
            "fit_angle_rad": fit["angle_rad"],
            "fit_amplitude_native": fit["amplitude_native"],
            "fit_residual_rms_fraction_peak": fit["residual_rms_fraction_peak"],
            "fit_pixel_count": fit["n_pixels"],
        })
    return rows


def native_crop(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    half_width: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ys, xs = crop_mask(x, y, center_x, center_y, half_width)
    xedge, yedge = pixel_edges(x[xs]), pixel_edges(y[ys])
    # Display positive azimuth to the left, matching the retained FITS WCS.
    return image[ys, xs][:, ::-1], (xedge[1], xedge[0], yedge[0], yedge[1])


def retained_crop(
    image: np.ndarray,
    header: Any,
    center_x: float,
    center_y: float,
    half_width: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    wcs = spatial_wcs(header)
    x, y = image_coordinates(wcs, image.shape)
    ys, xs = crop_mask(x, y, center_x, center_y, half_width)
    xedge, yedge = pixel_edges(x[xs]), pixel_edges(y[ys])
    return image[ys, xs], (xedge[0], xedge[1], yedge[0], yedge[1])


def native_profile(
    image: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    axis_x: float,
    axis_y: float,
    offsets: np.ndarray,
    cross_half_width_arcsec: float,
) -> np.ndarray:
    if image.shape != weight.shape or image.shape != (y.size, x.size):
        raise ContractError("native profile map/weight/coordinate geometry differs")
    pixel_arcsec = float(np.median(np.diff(x)))
    if not math.isfinite(pixel_arcsec) or pixel_arcsec <= 0.0:
        raise ContractError("native profile pixel scale is not finite and positive")
    xx, yy = np.meshgrid(x, y)
    dx, dy = xx - center_x, yy - center_y
    parallel = dx * axis_x + dy * axis_y
    perpendicular = -dx * axis_y + dy * axis_x
    base = (
        np.isfinite(image) & np.isfinite(weight) & (weight > 0.0)
        & (np.abs(perpendicular) <= cross_half_width_arcsec)
    )
    result = np.full(offsets.shape, np.nan, dtype=float)
    for index, offset in enumerate(offsets):
        chosen = base & (np.abs(parallel - offset) <= 0.5 * pixel_arcsec + 1.0e-12)
        if np.any(chosen):
            result[index] = float(np.average(image[chosen], weights=weight[chosen]))
    return result


def render_pdf(
    path: Path,
    ptc: PtcDetector,
    maps: dict[str, ReconstructedMap],
    fits: dict[str, dict[str, Any]],
    retained: dict[str, tuple[np.ndarray, np.ndarray, Any]],
    apt_values: dict[str, dict[str, Any]],
    displacement: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    pixel_arcsec: float,
    scan: Any,
    plot_half_width: float,
) -> None:
    cx, cy = apt_values["standard"]["x_t_raw"], apt_values["standard"]["y_t_raw"]
    reconstructed_crops = [
        native_crop(maps[mode].signal, x, y, cx, cy, plot_half_width)
        for mode in MODES
    ]
    retained_crops = [
        retained_crop(
            masked_signal(retained[mode][0], retained[mode][1]),
            retained[mode][2], cx, cy, plot_half_width,
        )
        for mode in MODES
    ]
    recon_limits = robust_limits([item[0] for item in reconstructed_crops])
    retained_limits = robust_limits([item[0] for item in retained_crops])

    with PdfPages(path, metadata={
        "Title": f"SCI-ALIGN-001 unthresholded PTC maps: Obs 150819 UID {ptc.uid}",
        "Subject": "Full-PTC naive reconstruction before map support thresholding",
    }) as pdf:
        figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.0))
        for column, mode in enumerate(MODES):
            recon_image, recon_extent = reconstructed_crops[column]
            retained_image, retained_extent = retained_crops[column]
            axes[0, column].imshow(
                recon_image, origin="lower", extent=recon_extent,
                interpolation="nearest", cmap="viridis",
                vmin=recon_limits[0], vmax=recon_limits[1], aspect="equal",
            )
            axes[0, column].plot(
                fits[mode]["x_arcsec"], fits[mode]["y_arcsec"], "+",
                color="red", markersize=9, markeredgewidth=1.5,
            )
            axes[0, column].set_title(
                f"{mode}: unthresholded full-PTC reconstruction\n"
                f"fit=({fits[mode]['x_arcsec']:.3f}, {fits[mode]['y_arcsec']:.3f}) arcsec",
                fontsize=9,
            )
            axes[1, column].imshow(
                retained_image, origin="lower", extent=retained_extent,
                interpolation="nearest", cmap="viridis",
                vmin=retained_limits[0], vmax=retained_limits[1], aspect="equal",
            )
            axes[1, column].plot(
                apt_values[mode]["x_t_raw"], apt_values[mode]["y_t_raw"],
                "o", markerfacecolor="none", markeredgecolor="red", markersize=7,
            )
            axes[1, column].set_title(
                f"{mode}: retained thresholded Citlali map\n"
                f"APT=({apt_values[mode]['x_t_raw']:.3f}, "
                f"{apt_values[mode]['y_t_raw']:.3f}) arcsec",
                fontsize=9,
            )
            for row in range(2):
                axes[row, column].set_xlabel("Az offset (arcsec)")
                axes[row, column].set_ylabel("El offset (arcsec)")
                axes[row, column].tick_params(labelsize=8)
        figure.suptitle(
            f"Obs 150819 a1100 UID {ptc.uid}: support-threshold comparison\n"
            "top: all positive-weight full-PTC hits; bottom: final Citlali support",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        offsets = np.arange(-20.0, 20.0 + 0.5 * pixel_arcsec, pixel_arcsec)
        colors = {"standard": "#333333", "left": "#1f77b4", "right": "#ff7f0e"}
        for mode in MODES:
            center = fits[mode]
            profile = native_profile(
                maps[mode].signal, maps[mode].weight, x, y,
                center["x_arcsec"], center["y_arcsec"],
                scan.axis_x, scan.axis_y, offsets,
                cross_half_width_arcsec=3.0,
            )
            axes[0].plot(offsets, normalized_profile(profile), color=colors[mode], label=mode)
        axes[0].axvline(0.0, color="0.7", linewidth=0.8)
        axes[0].set_xlabel("Fast-axis offset from each reconstructed centroid (arcsec)")
        axes[0].set_ylabel("Baseline-subtracted peak-normalized signal")
        axes[0].set_title("Unthresholded reconstructed core profiles")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        axes[1].axis("off")
        lines = [
            "Right-minus-left displacement",
            "",
        ]
        for row in displacement:
            lines.extend([
                row["family"],
                f"  parallel = {row['delta_parallel_right_minus_left_arcsec']:+.6f} arcsec",
                f"  perpendicular = {row['delta_perpendicular_right_minus_left_arcsec']:+.6f} arcsec",
                f"  timing equivalent = {row['timing_equivalent_ms']:+.6f} ms",
                "",
            ])
        lines.extend([
            "Interpretation boundary:",
            "The reconstruction omits Citlali's coverage support threshold.",
            "Its full PTC is a separate single-pass replay, not the retained",
            "multi-iteration map's pre-threshold buffer.",
        ])
        axes[1].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
        figure.suptitle(
            f"Obs 150819 UID {ptc.uid}: unthresholded directional displacement",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(figure)
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.uid < 0 or args.fit_half_width_arcsec <= 0.0 or args.plot_half_width_arcsec <= 0.0:
        raise ContractError("uid and fit/plot half-widths must be positive")
    if args.plot_half_width_arcsec <= args.fit_half_width_arcsec:
        raise ContractError("plot half-width must exceed fit half-width")
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"refusing existing output directory: {output}")

    ptc = load_ptc_detector(args.full_ptc_tod, args.uid)
    if ptc.array != ARRAY_IDS[args.array]:
        raise ContractError(
            f"uid={args.uid} full-PTC array {ptc.array} disagrees with {args.array}"
        )
    bounds = load_full_ptc_append_bounds(ptc)
    recomputed_lat, recomputed_lon = load_map_pointing(ptc)
    residual = np.hypot(ptc.det_lat - recomputed_lat, ptc.det_lon - recomputed_lon)
    finite_residual = residual[np.isfinite(residual)]
    if finite_residual.size != residual.size or float(np.max(finite_residual)) > 1.0e-12:
        raise ContractError("stored detector pointing disagrees with telescope-plus-offset pointing")

    raw_dir = discover_raw_dir(args.map_reduction_root)
    registry_path = raw_dir / "beammap_direction_scan_registry_all.csv"
    registry = load_registry(registry_path)
    scan = scan_summary(registry_path)
    apt_paths, apt_tables, apt_indices = map_tables(raw_dir)
    for mode in MODES:
        if args.uid not in apt_indices[mode]:
            raise ContractError(f"uid={args.uid} is absent from the {mode} APT")
    apt_values = {
        mode: row_values(apt_tables[mode], apt_indices[mode][args.uid])
        for mode in MODES
    }
    if int(apt_values["standard"]["array"]) != ptc.array or int(
        apt_values["standard"]["nw"]
    ) != ptc.nw:
        raise ContractError("full PTC and retained APT detector identities disagree")

    retained: dict[str, tuple[np.ndarray, np.ndarray, Any]] = {}
    used_fits: list[Path] = []
    with contextlib.ExitStack() as stack:
        for mode in MODES:
            paths = discover_fits(raw_dir, args.array, mode)
            product = FitsProduct(paths, stack)
            retained[mode] = product.planes(
                apt_indices[mode][args.uid], int(apt_values[mode]["flag"])
            )
            used_fits.extend(paths)
        signatures = {mode: wcs_signature(retained[mode][2]) for mode in MODES}
        if any(signatures[mode] != signatures["standard"] for mode in MODES[1:]):
            raise ContractError("retained standard/left/right maps do not share one WCS")
        shape = retained["standard"][0].shape
        if any(retained[mode][0].shape != shape for mode in MODES):
            raise ContractError("retained standard/left/right maps have different shapes")
        wcs = spatial_wcs(retained["standard"][2])
        pixel_size_rad = pixel_size_rad_from_wcs(wcs)
        pixel_arcsec = pixel_size_rad * RAD_TO_ARCSEC
        x = (np.arange(shape[1], dtype=float) - (shape[1] - 1) / 2.0) * pixel_arcsec
        y = (np.arange(shape[0], dtype=float) - (shape[0] - 1) / 2.0) * pixel_arcsec
        maps = reconstruct_unthresholded_maps(
            ptc, bounds, registry, ptc.det_lat, ptc.det_lon, shape, pixel_size_rad
        )
        fits = fit_reconstructions(
            maps, x, y,
            apt_values["standard"]["x_t_raw"],
            apt_values["standard"]["y_t_raw"],
            args.fit_half_width_arcsec,
        )
        reconstructed_centers = {
            mode: (fits[mode]["x_arcsec"], fits[mode]["y_arcsec"])
            for mode in MODES
        }
        apt_centers = {
            mode: (apt_values[mode]["x_t_raw"], apt_values[mode]["y_t_raw"])
            for mode in MODES
        }
        displacement = [
            displacement_row("unthresholded_full_ptc_reconstruction", reconstructed_centers, scan),
            displacement_row("thresholded_retained_citlali_apt", apt_centers, scan),
        ]
        output.mkdir(parents=True)
        pdf_path = output / f"unthresholded_ptc_maps_o150819_uid{args.uid}.pdf"
        render_pdf(
            pdf_path, ptc, maps, fits, retained, apt_values, displacement,
            x, y, pixel_arcsec, scan, args.plot_half_width_arcsec,
        )

    metrics_path = output / "reconstruction_metrics.ecsv"
    displacement_path = output / "displacement_comparison.ecsv"
    maps_path = output / "unthresholded_maps.npz"
    manifest_path = output / "manifest.json"
    Table(rows=metrics_rows(maps, fits)).write(metrics_path, format="ascii.ecsv")
    Table(rows=displacement).write(displacement_path, format="ascii.ecsv")
    np.savez_compressed(maps_path, **{
        **{f"{mode}_signal": maps[mode].signal for mode in MODES},
        **{f"{mode}_weight": maps[mode].weight for mode in MODES},
        **{f"{mode}_hit_count": maps[mode].hit_count for mode in MODES},
        "native_x_arcsec": x,
        "native_y_arcsec": y,
    })
    manifest = {
        "schema": "sci-align-001-unthresholded-full-ptc-map-reconstruction-v1",
        "observation_number": 150819,
        "uid": args.uid,
        "array": args.array,
        "network": ptc.nw,
        "full_ptc_fruitloops_iter": ptc.fruitloops_iter,
        "full_ptc_sample_count": int(ptc.signal.size),
        "full_ptc_scan_count": int(ptc.output_scan_index.size),
        "pixel_size_arcsec": pixel_arcsec,
        "fit_half_width_arcsec": args.fit_half_width_arcsec,
        "plot_half_width_arcsec": args.plot_half_width_arcsec,
        "pointing_crosscheck_max_residual_arcsec": float(np.max(finite_residual)) * RAD_TO_ARCSEC,
        "mapmaking_contract": (
            "nearest-pixel C++ llround-equivalent binning; per-scan detector weight; "
            "flag==0; finite signal; detector grouping uses stored det_lat/det_lon; "
            "no coverage support threshold"
        ),
        "interpretation_boundary": (
            "The full PTC is a separate single-pass replay. This tests whether the "
            "directional displacement exists before support thresholding in that replay; "
            "it is not a reversible de-thresholding of the retained multi-iteration map."
        ),
        "reconstruction_metrics": metrics_rows(maps, fits),
        "displacement_comparison": displacement,
        "inputs": [{
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        } for role, path in [
            ("full_ptc_tod", ptc.path),
            ("direction_registry", registry_path),
            *[(f"{mode}_apt", apt_paths[mode]) for mode in MODES],
            *[("retained_detector_map_fits", path) for path in sorted(set(used_fits))],
        ]],
        "outputs": [
            pdf_path.name, metrics_path.name, displacement_path.name, maps_path.name,
        ],
    }
    write_json(manifest_path, manifest)
    output_checksums(output, [
        pdf_path.name, metrics_path.name, displacement_path.name,
        maps_path.name, manifest_path.name,
    ])

    print("===== UNTHRESHOLDED FULL-PTC MAP RECONSTRUCTION =====")
    print(
        f"obs=150819 uid={args.uid} samples={ptc.signal.size} "
        f"scans={ptc.output_scan_index.size} pixel_arcsec={pixel_arcsec:.9f}"
    )
    for row in metrics_rows(maps, fits):
        print(
            f"mode={row['mode']} scans={row['scan_count']} "
            f"accepted={row['accepted_sample_count']} "
            f"pixels={row['positive_weight_pixel_count']} "
            f"fit=({row['fit_x_arcsec']:+.6f},{row['fit_y_arcsec']:+.6f})"
        )
    for row in displacement:
        print(
            f"family={row['family']} "
            f"parallel_arcsec={row['delta_parallel_right_minus_left_arcsec']:+.6f} "
            f"perpendicular_arcsec={row['delta_perpendicular_right_minus_left_arcsec']:+.6f} "
            f"timing_ms={row['timing_equivalent_ms']:+.6f}"
        )
    print(f"output={output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--full-ptc-tod", required=True, type=Path)
    result.add_argument("--map-reduction-root", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--uid", type=int, default=199)
    result.add_argument("--array", choices=sorted(ARRAY_IDS), default="a1100")
    result.add_argument("--fit-half-width-arcsec", type=float, default=12.0)
    result.add_argument("--plot-half-width-arcsec", type=float, default=25.0)
    return result


def main() -> int:
    try:
        run(parser().parse_args())
    except (ContractError, OSError, ValueError, KeyError, IndexError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
