#!/usr/bin/env python3
"""Render detector-resolved standard/left/right Beammap comparisons.

This diagnostic is read-only.  It consumes one completed Citlali
``beammap.direction_mode: all`` reduction and writes a multipage PDF plus
machine-readable selection, metric, and provenance products.  Detector
selection is based only on the standard APT (and an optional pre-existing UID
allowlist); directional displacements never influence cohort membership.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


ARRAY_IDS = {"a1100": 0, "a1400": 1, "a2000": 2}
MODES = ("standard", "left", "right")
RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi
REQUIRED_APT_COLUMNS = {
    "uid", "array", "nw", "flag", "amp", "x_t", "x_t_raw", "x_t_err",
    "y_t", "y_t_raw", "y_t_err", "a_fwhm", "b_fwhm", "sig2noise",
}


class ContractError(RuntimeError):
    """A retained product does not satisfy the visualization contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def finite_scalar(value: Any) -> float:
    if np.ma.is_masked(value):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def int_scalar(value: Any, default: int = -1) -> int:
    number = finite_scalar(value)
    return int(round(number)) if math.isfinite(number) else default


def require_unique(paths: Iterable[Path], label: str) -> Path:
    values = sorted(set(paths))
    if len(values) != 1:
        raise ContractError(
            f"expected exactly one {label}; found {len(values)}: "
            f"{[str(path) for path in values]}"
        )
    return values[0]


def is_standard_apt(path: Path) -> bool:
    stem = path.stem
    return (
        stem.startswith("apt_")
        and stem.endswith("_citlali")
        and "_fit_qc" not in stem
        and "_psf" not in stem
    )


def discover_raw_dir(root: Path) -> Path:
    root = root.resolve()
    direct = [path for path in root.glob("apt_*.ecsv") if is_standard_apt(path)]
    if len(direct) == 1 and root.name in {"raw", "filtered"}:
        return root
    candidates = {
        path.parent
        for path in root.rglob("apt_*.ecsv")
        if is_standard_apt(path) and path.parent.name == "raw"
    }
    if len(candidates) != 1:
        raise ContractError(
            "could not resolve one raw Beammap product directory beneath "
            f"{root}; candidates={sorted(str(path) for path in candidates)}"
        )
    return next(iter(candidates))


def product_apt_path(standard: Path, mode: str) -> Path:
    suffix = "" if mode == "standard" else f"_{mode}"
    return standard.with_name(standard.stem + suffix + standard.suffix)


def fit_qc_path(standard: Path, mode: str) -> Path:
    suffix = "" if mode == "standard" else f"_{mode}"
    return standard.with_name(standard.stem + suffix + "_fit_qc.ecsv")


def fits_mode(path: Path) -> str:
    name = path.name
    if "_citlali_left" in name:
        return "left"
    if "_citlali_right" in name:
        return "right"
    return "standard"


def discover_fits(raw_dir: Path, array_name: str, mode: str) -> list[Path]:
    candidates = [
        path for path in raw_dir.glob(f"*_{array_name}_beammap_*.fits")
        if fits_mode(path) == mode and "noise" not in path.name
    ]
    if not candidates:
        raise ContractError(
            f"no {array_name} {mode} detector-map FITS under {raw_dir}"
        )
    return sorted(candidates)


def require_columns(table: Table, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(table.colnames))
    if missing:
        raise ContractError(f"{label} lacks required columns: {missing}")


def uid_index(table: Table, label: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, value in enumerate(table["uid"]):
        uid = int_scalar(value)
        if uid < 0 or uid in result:
            raise ContractError(f"{label} has invalid or duplicate uid={uid}")
        result[uid] = index
    return result


def read_uid_allowlist(path: Path | None) -> tuple[list[int] | None, str]:
    if path is None:
        return None, "standard_apt_quality_only"
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"hero UID selection is missing: {path}")
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text())
        values = document.get("uids", document) if isinstance(document, dict) else document
        if not isinstance(values, list):
            raise ContractError("JSON hero selection must be a list or {'uids': [...]} mapping")
        uids = [int(value) for value in values]
    else:
        table = Table.read(path)
        if "uid" not in table.colnames:
            raise ContractError(f"hero selection table lacks uid column: {path}")
        uids = [int_scalar(value) for value in table["uid"]]
    if any(uid < 0 for uid in uids) or len(set(uids)) != len(uids):
        raise ContractError("hero selection contains invalid or duplicate UIDs")
    return uids, str(path)


def standard_hero_eligible(row: Any, array_id: int) -> bool:
    required_finite = (
        "amp", "x_t", "x_t_err", "y_t", "y_t_err", "a_fwhm",
        "b_fwhm", "sig2noise",
    )
    if int_scalar(row["array"]) != array_id or int_scalar(row["flag"]) != 0:
        return False
    if "flag2" in row.colnames and int_scalar(row["flag2"]) != 0:
        return False
    values = {name: finite_scalar(row[name]) for name in required_finite}
    return (
        all(math.isfinite(value) for value in values.values())
        and values["amp"] > 0.0
        and values["x_t_err"] >= 0.0
        and values["y_t_err"] >= 0.0
        and values["a_fwhm"] > 0.0
        and values["b_fwhm"] > 0.0
        and values["sig2noise"] > 0.0
    )


def select_detectors(
    standard: Table,
    array_id: int,
    maximum: int,
    allowlist: list[int] | None,
) -> list[dict[str, Any]]:
    index = uid_index(standard, "standard APT")
    eligible: list[dict[str, Any]] = []
    allow_order = {uid: rank for rank, uid in enumerate(allowlist or [])}
    for row_index, row in enumerate(standard):
        uid = int_scalar(row["uid"])
        if not standard_hero_eligible(row, array_id):
            continue
        if allowlist is not None and uid not in allow_order:
            continue
        eligible.append({
            "uid": uid,
            "standard_row_index": row_index,
            "nw": int_scalar(row["nw"]),
            "standard_sig2noise": finite_scalar(row["sig2noise"]),
            "allowlist_rank": allow_order.get(uid, -1),
        })
    if allowlist is not None:
        missing = sorted(set(allowlist) - set(index))
        if missing:
            raise ContractError(f"hero selection UIDs absent from standard APT: {missing}")
        ineligible = sorted(set(allowlist) - {row["uid"] for row in eligible})
        if ineligible:
            raise ContractError(
                "hero selection UIDs fail the standard-only a1100 quality rules: "
                f"{ineligible}"
            )
        eligible.sort(key=lambda row: (row["allowlist_rank"], row["uid"]))
        selected = eligible[:maximum]
    else:
        by_network: dict[int, list[dict[str, Any]]] = {}
        for row in eligible:
            by_network.setdefault(row["nw"], []).append(row)
        for rows in by_network.values():
            rows.sort(key=lambda row: (-row["standard_sig2noise"], row["uid"]))
        selected = []
        depth = 0
        networks = sorted(by_network)
        while len(selected) < maximum:
            added = False
            for network in networks:
                rows = by_network[network]
                if depth < len(rows):
                    selected.append(rows[depth])
                    added = True
                    if len(selected) == maximum:
                        break
            if not added:
                break
            depth += 1
        selected.sort(key=lambda row: (row["nw"], -row["standard_sig2noise"], row["uid"]))
    if not selected:
        raise ContractError("standard-only hero selection produced no detectors")
    for rank, row in enumerate(selected):
        row["selection_rank"] = rank
    return selected


@dataclass
class ScanSummary:
    scan_angle_rad: float
    axis_x: float
    axis_y: float
    cross_x: float
    cross_y: float
    left_count: int
    right_count: int
    left_rate_arcsec_s: float
    right_rate_arcsec_s: float


def scan_summary(path: Path) -> ScanSummary:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ContractError(f"empty scan registry: {path}")
    angles = np.asarray([float(row["scan_angle_rad"]) for row in rows])
    if not np.all(np.isfinite(angles)) or not np.allclose(angles, angles[0], atol=0.0, rtol=0.0):
        raise ContractError("scan registry does not have one exact finite scan angle")
    angle = float(angles[0])
    axis = np.asarray([math.cos(angle), math.sin(angle)])
    summaries: dict[str, tuple[int, float]] = {}
    for direction in ("left", "right"):
        chosen = [row for row in rows if row["direction"] == direction]
        if not chosen:
            raise ContractError(f"scan registry has no {direction} legs")
        duration = np.asarray([float(row["duration_sec"]) for row in chosen])
        rate = np.asarray([
            float(row["signed_fast_axis_rate_rad_per_sec"]) * RAD_TO_ARCSEC
            for row in chosen
        ])
        if np.any(~np.isfinite(duration)) or np.any(duration <= 0.0) or np.any(~np.isfinite(rate)):
            raise ContractError(f"scan registry has invalid {direction} durations or rates")
        summaries[direction] = (len(chosen), float(np.average(rate, weights=duration)))
    if not (summaries["left"][1] < 0.0 < summaries["right"][1]):
        raise ContractError("scan registry signed rates do not bracket zero")
    return ScanSummary(
        scan_angle_rad=angle,
        axis_x=float(axis[0]), axis_y=float(axis[1]),
        cross_x=float(-axis[1]), cross_y=float(axis[0]),
        left_count=summaries["left"][0], right_count=summaries["right"][0],
        left_rate_arcsec_s=summaries["left"][1],
        right_rate_arcsec_s=summaries["right"][1],
    )


class FitsProduct:
    def __init__(self, paths: Sequence[Path], stack: contextlib.ExitStack):
        self.paths = list(paths)
        self.by_flag: dict[int | None, fits.HDUList] = {}
        for path in self.paths:
            hdus = stack.enter_context(fits.open(path, mode="readonly", memmap=True))
            split = hdus[0].header.get("BEAMMAP.SPLIT_VALUE")
            key = int(split) if split is not None else None
            if key in self.by_flag:
                raise ContractError(f"duplicate detector-map FITS flag group {key}: {paths}")
            self.by_flag[key] = hdus

    def planes(self, map_index: int, flag: int) -> tuple[np.ndarray, np.ndarray, fits.Header]:
        signal_name = f"signal_det_{map_index}_I"
        weight_name = f"weight_det_{map_index}_I"
        candidates = []
        if flag in self.by_flag:
            candidates.append(self.by_flag[flag])
        if None in self.by_flag:
            candidates.append(self.by_flag[None])
        candidates.extend(hdus for key, hdus in self.by_flag.items() if key not in {flag, None})
        for hdus in candidates:
            try:
                signal_hdu = hdus[signal_name]
                weight_hdu = hdus[weight_name]
            except (KeyError, IndexError):
                continue
            signal = np.asarray(signal_hdu.data, dtype=float).squeeze()
            weight = np.asarray(weight_hdu.data, dtype=float).squeeze()
            if signal.ndim != 2 or signal.shape != weight.shape:
                raise ContractError(f"invalid detector plane geometry for {signal_name}")
            return signal, weight, signal_hdu.header.copy()
        raise ContractError(
            f"detector map extensions {signal_name}/{weight_name} are missing"
        )


def wcs_signature(header: fits.Header) -> dict[str, Any]:
    keys = (
        "NAXIS1", "NAXIS2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
        "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    )
    return {key: header.get(key) for key in keys}


def spatial_wcs(header: fits.Header) -> WCS:
    wcs = WCS(header).sub([1, 2])
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    if matrix.shape != (2, 2) or np.any(~np.isfinite(matrix)):
        raise ContractError("invalid spatial WCS matrix")
    if abs(matrix[0, 1]) > 1.0e-12 or abs(matrix[1, 0]) > 1.0e-12:
        raise ContractError("visualizer currently requires axis-aligned AltAz offset WCS")
    return wcs


def image_coordinates(wcs: WCS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    xpix = np.arange(shape[1], dtype=float)
    ypix = np.arange(shape[0], dtype=float)
    xworld, _ = wcs.pixel_to_world_values(xpix, np.zeros_like(xpix))
    _, yworld = wcs.pixel_to_world_values(np.zeros_like(ypix), ypix)
    return np.asarray(xworld, dtype=float), np.asarray(yworld, dtype=float)


def pixel_edges(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        raise ContractError("map axis has fewer than two pixels")
    delta = float(np.median(np.diff(values)))
    return float(values[0] - 0.5 * delta), float(values[-1] + 0.5 * delta)


def masked_signal(signal: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(signal) & np.isfinite(weight) & (weight > 0.0), signal, np.nan)


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (
        np.isfinite(x) & np.isfinite(y)
        & (x0 >= 0) & (y0 >= 0)
        & (x1 < image.shape[1]) & (y1 < image.shape[0])
    )
    result = np.full(x.shape, np.nan, dtype=float)
    if not np.any(valid):
        return result
    xv, yv = x[valid], y[valid]
    x0v, x1v, y0v, y1v = x0[valid], x1[valid], y0[valid], y1[valid]
    values = np.column_stack([
        image[y0v, x0v], image[y0v, x1v], image[y1v, x0v], image[y1v, x1v]
    ])
    finite = np.all(np.isfinite(values), axis=1)
    dx, dy = xv - x0v, yv - y0v
    interpolated = (
        values[:, 0] * (1.0 - dx) * (1.0 - dy)
        + values[:, 1] * dx * (1.0 - dy)
        + values[:, 2] * (1.0 - dx) * dy
        + values[:, 3] * dx * dy
    )
    target = np.flatnonzero(valid)
    result[target[finite]] = interpolated[finite]
    return result


def row_values(table: Table, index: int) -> dict[str, Any]:
    row = table[index]
    result = {name: finite_scalar(row[name]) for name in (
        "uid", "array", "nw", "flag", "flag2", "amp", "amp_err",
        "x_t", "x_t_raw", "x_t_err", "y_t", "y_t_raw", "y_t_err",
        "a_fwhm", "b_fwhm", "angle", "sig2noise",
    ) if name in table.colnames}
    return result


def detector_metrics(
    uid: int,
    tables: dict[str, Table],
    indices: dict[str, dict[int, int]],
    scan: ScanSummary,
) -> dict[str, Any]:
    values = {mode: row_values(tables[mode], indices[mode][uid]) for mode in MODES}
    left, right = values["left"], values["right"]
    # Detector-map WCS and the scan registry are both in the raw AltAz map
    # frame.  x_t/y_t are subsequently reference-subtracted and may be
    # derotated by process_apt(), so they cannot be projected onto this scan
    # axis or used to locate a fitted source in the FITS image.
    dx = right["x_t_raw"] - left["x_t_raw"]
    dy = right["y_t_raw"] - left["y_t_raw"]
    parallel = dx * scan.axis_x + dy * scan.axis_y
    perpendicular = dx * scan.cross_x + dy * scan.cross_y
    variance_x = left["x_t_err"] ** 2 + right["x_t_err"] ** 2
    variance_y = left["y_t_err"] ** 2 + right["y_t_err"] ** 2
    parallel_sigma = math.sqrt(scan.axis_x ** 2 * variance_x + scan.axis_y ** 2 * variance_y)
    perpendicular_sigma = math.sqrt(scan.cross_x ** 2 * variance_x + scan.cross_y ** 2 * variance_y)
    denominator = scan.right_rate_arcsec_s - scan.left_rate_arcsec_s
    timing_ms = 1000.0 * parallel / denominator
    timing_sigma_ms = 1000.0 * parallel_sigma / abs(denominator)
    result: dict[str, Any] = {
        "uid": uid,
        "array": int(values["standard"]["array"]),
        "nw": int(values["standard"]["nw"]),
        "position_frame": "raw_altaz_detector_map",
        "delta_x_raw_right_minus_left_arcsec": dx,
        "delta_y_raw_right_minus_left_arcsec": dy,
        "delta_parallel_right_minus_left_arcsec": parallel,
        "delta_parallel_fit_sigma_arcsec": parallel_sigma,
        "delta_perpendicular_right_minus_left_arcsec": perpendicular,
        "delta_perpendicular_fit_sigma_arcsec": perpendicular_sigma,
        "left_rate_arcsec_s": scan.left_rate_arcsec_s,
        "right_rate_arcsec_s": scan.right_rate_arcsec_s,
        "rate_difference_arcsec_s": denominator,
        "timing_equivalent_ms": timing_ms,
        "timing_equivalent_fit_sigma_ms": timing_sigma_ms,
        "uncertainty_scope": "diagonal_left_right_beam_fit_covariance_only",
    }
    for mode in MODES:
        for key in (
            "flag", "flag2", "sig2noise", "x_t", "x_t_raw", "x_t_err",
            "y_t", "y_t_raw", "y_t_err", "a_fwhm", "b_fwhm", "amp",
            "amp_err",
        ):
            if key in values[mode]:
                result[f"{mode}_{key}"] = values[mode][key]
    return result


def robust_limits(images: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([image[np.isfinite(image)] for image in images])
    if values.size == 0:
        raise ContractError("selected detector has no finite positive-weight map pixels")
    low, high = np.percentile(values, [2.0, 99.5])
    if not high > low:
        center = float(np.median(values))
        width = max(abs(center) * 1.0e-6, 1.0e-12)
        return center - width, center + width
    return float(low), float(high)


def crop_mask(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, half: float,
) -> tuple[slice, slice]:
    ix = np.flatnonzero(np.abs(x - cx) <= half)
    iy = np.flatnonzero(np.abs(y - cy) <= half)
    if ix.size < 5 or iy.size < 5:
        raise ContractError(f"map does not cover requested {half}-arcsec centroid crop")
    return slice(int(iy[0]), int(iy[-1] + 1)), slice(int(ix[0]), int(ix[-1] + 1))


def normalized_profile(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if np.sum(finite) < 5:
        return np.full_like(values, np.nan)
    edge = np.r_[values[finite][:5], values[finite][-5:]]
    baseline = float(np.nanmedian(edge))
    shifted = values - baseline
    peak = float(np.nanmax(shifted))
    return shifted / peak if peak > 0.0 else np.full_like(values, np.nan)


def draw_map(
    ax: Any, image: np.ndarray, extent: tuple[float, float, float, float],
    title: str, limits: tuple[float, float], centroid: tuple[float, float],
    marker: str, color: str,
) -> None:
    ax.imshow(
        image, origin="lower", extent=extent, interpolation="nearest",
        cmap="viridis", vmin=limits[0], vmax=limits[1], rasterized=True,
        aspect="equal",
    )
    ax.plot(centroid[0], centroid[1], marker=marker, color=color, markersize=6,
            markerfacecolor="none", markeredgewidth=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Az offset (arcsec)", fontsize=8)
    ax.set_ylabel("El offset (arcsec)", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_direction_arrow(
    ax: Any, scan: ScanSummary,
    extent: tuple[float, float, float, float],
) -> None:
    xmin, xmax = sorted(extent[:2])
    ymin, ymax = sorted(extent[2:])
    start = np.asarray([xmin + 0.16 * (xmax - xmin), ymin + 0.13 * (ymax - ymin)])
    length = 0.18 * min(xmax - xmin, ymax - ymin)
    stop = start + length * np.asarray([scan.axis_x, scan.axis_y])
    ax.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.2})
    ax.text(start[0], start[1], "+scan", fontsize=7, ha="left", va="top")


def render_detector(
    axes: np.ndarray,
    uid: int,
    maps: dict[str, tuple[np.ndarray, np.ndarray, fits.Header]],
    apt_values: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    scan: ScanSummary,
    half_width: float,
) -> None:
    standard_header = maps["standard"][2]
    signatures = {mode: wcs_signature(maps[mode][2]) for mode in MODES}
    if (
        signatures["left"] != signatures["standard"]
        or signatures["right"] != signatures["standard"]
    ):
        raise ContractError(f"uid={uid} standard/left/right WCS signatures differ")
    wcs = spatial_wcs(standard_header)
    shape = maps["standard"][0].shape
    if any(maps[mode][0].shape != shape for mode in MODES):
        raise ContractError(f"uid={uid} standard/left/right shapes differ")
    x, y = image_coordinates(wcs, shape)
    cx = apt_values["standard"]["x_t_raw"]
    cy = apt_values["standard"]["y_t_raw"]
    ys, xs = crop_mask(x, y, cx, cy, half_width)
    cropped = {
        mode: masked_signal(maps[mode][0], maps[mode][1])[ys, xs]
        for mode in MODES
    }
    xcrop, ycrop = x[xs], y[ys]
    xedges, yedges = pixel_edges(xcrop), pixel_edges(ycrop)
    extent = (xedges[0], xedges[1], yedges[0], yedges[1])
    limits = robust_limits(list(cropped.values()))
    style = {
        "standard": ("+", "0.2"),
        "left": ("o", "tab:blue"),
        "right": ("s", "tab:orange"),
    }
    for column, mode in enumerate(MODES):
        value = apt_values[mode]
        draw_map(
            axes[0, column], cropped[mode], extent,
            f"{mode}: flag={int(value.get('flag', -1))} S/N={value.get('sig2noise', math.nan):.1f}",
            limits, (value["x_t_raw"], value["y_t_raw"]), *style[mode],
        )
        draw_direction_arrow(axes[0, column], scan, extent)

    overlay = axes[1, 0]
    contour_colors = {"standard": "0.25", "left": "tab:blue", "right": "tab:orange"}
    for mode in MODES:
        image = cropped[mode]
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            continue
        baseline = float(np.percentile(finite, 10.0))
        peak = float(np.max(finite) - baseline)
        if peak <= 0.0:
            continue
        overlay.contour(
            xcrop, ycrop, image - baseline,
            levels=np.asarray([0.3, 0.5, 0.7, 0.9]) * peak,
            colors=contour_colors[mode], linewidths=0.8,
        )
        marker, color = style[mode]
        value = apt_values[mode]
        overlay.plot(value["x_t_raw"], value["y_t_raw"], marker=marker, color=color,
                     markersize=6, markerfacecolor="none", markeredgewidth=1.3,
                     label=mode)
    overlay.set_title("Unrecentered contours", fontsize=9)
    overlay.set_xlabel("Az offset (arcsec)", fontsize=8)
    overlay.set_ylabel("El offset (arcsec)", fontsize=8)
    overlay.tick_params(labelsize=7)
    overlay.set_aspect("equal")
    overlay.legend(fontsize=7, loc="best")
    draw_direction_arrow(overlay, scan, extent)

    difference = cropped["left"] - cropped["right"]
    finite_difference = np.abs(difference[np.isfinite(difference)])
    diff_limit = float(np.percentile(finite_difference, 99.0)) if finite_difference.size else 1.0
    diff_limit = max(diff_limit, np.finfo(float).eps)
    axes[1, 1].imshow(
        difference, origin="lower", extent=extent, interpolation="nearest",
        cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit,
        rasterized=True, aspect="equal",
    )
    axes[1, 1].set_title("Left - right (common WCS)", fontsize=9)
    axes[1, 1].set_xlabel("Az offset (arcsec)", fontsize=8)
    axes[1, 1].set_ylabel("El offset (arcsec)", fontsize=8)
    axes[1, 1].tick_params(labelsize=7)
    draw_direction_arrow(axes[1, 1], scan, extent)

    profile_ax = axes[1, 2]
    offsets = np.linspace(-half_width, half_width, 241)
    world_x = cx + offsets * scan.axis_x
    world_y = cy + offsets * scan.axis_y
    pixel_x, pixel_y = wcs.world_to_pixel_values(world_x, world_y)
    for mode in MODES:
        image = masked_signal(maps[mode][0], maps[mode][1])
        profile = normalized_profile(bilinear(image, pixel_x, pixel_y))
        profile_ax.plot(offsets, profile, color=style[mode][1], label=mode, lw=1.2)
        center = (
            (apt_values[mode]["x_t_raw"] - cx) * scan.axis_x
            + (apt_values[mode]["y_t_raw"] - cy) * scan.axis_y
        )
        profile_ax.axvline(center, color=style[mode][1], lw=0.8, alpha=0.7)
    profile_ax.axhline(0.0, color="0.7", lw=0.6)
    profile_ax.set_xlim(-half_width, half_width)
    profile_ax.set_xlabel("Fast-axis offset from standard centroid (arcsec)", fontsize=8)
    profile_ax.set_ylabel("Baseline-subtracted peak-normalized signal", fontsize=8)
    profile_ax.set_title(
        f"Along-scan slice: dt={metrics['timing_equivalent_ms']:+.2f} +/- "
        f"{metrics['timing_equivalent_fit_sigma_ms']:.2f} ms (fit-only)",
        fontsize=9,
    )
    profile_ax.tick_params(labelsize=7)
    profile_ax.grid(alpha=0.2)
    profile_ax.legend(fontsize=7, loc="best")


def make_pdf(
    output: Path,
    selected: list[dict[str, Any]],
    products: dict[str, FitsProduct],
    tables: dict[str, Table],
    indices: dict[str, dict[int, int]],
    metrics: dict[int, dict[str, Any]],
    scan: ScanSummary,
    detectors_per_page: int,
    half_width: float,
    observation: int,
    array_name: str,
) -> None:
    with PdfPages(output) as pdf:
        for page_start in range(0, len(selected), detectors_per_page):
            page_rows = selected[page_start:page_start + detectors_per_page]
            figure = plt.figure(
                figsize=(11.0, 8.5 * len(page_rows)),
                constrained_layout=False,
            )
            outer = figure.add_gridspec(len(page_rows), 1, hspace=0.22)
            for page_offset, selection in enumerate(page_rows):
                uid = int(selection["uid"])
                inner = outer[page_offset].subgridspec(2, 3, hspace=0.32, wspace=0.28)
                axes = np.asarray([
                    [figure.add_subplot(inner[row, column]) for column in range(3)]
                    for row in range(2)
                ])
                apt_values = {
                    mode: row_values(tables[mode], indices[mode][uid])
                    for mode in MODES
                }
                maps = {}
                for mode in MODES:
                    row_index = indices[mode][uid]
                    flag = int(apt_values[mode].get("flag", -1))
                    maps[mode] = products[mode].planes(row_index, flag)
                render_detector(
                    axes, uid, maps, apt_values, metrics[uid], scan, half_width,
                )
                axes[0, 0].text(
                    0.0, 1.10,
                    f"Obs {observation} {array_name} uid={uid} nw={selection['nw']} | "
                    "right-left: parallel="
                    f"{metrics[uid]['delta_parallel_right_minus_left_arcsec']:+.3f} "
                    "arcsec, perpendicular="
                    f"{metrics[uid]['delta_perpendicular_right_minus_left_arcsec']:+.3f} "
                    "arcsec",
                    transform=axes[0, 0].transAxes, fontsize=9, ha="left", va="bottom",
                )
            figure.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.96)
            pdf.savefig(figure, dpi=150)
            plt.close(figure)
        info = pdf.infodict()
        info["Title"] = (
            "SCI-ALIGN-001 split-direction Beammap visualization: "
            f"{observation} {array_name}"
        )
        info["Subject"] = "Unrecentered standard, left, and right detector Beammap products"


def output_checksums(output_dir: Path, names: Sequence[str]) -> None:
    lines = [f"{sha256_file(output_dir / name)}  {name}\n" for name in sorted(names)]
    (output_dir / "SHA256SUMS").write_text("".join(lines))


def run(args: argparse.Namespace) -> None:
    if args.detectors_per_page not in (1, 2):
        raise ContractError("detectors-per-page must be one or two")
    if args.max_detectors <= 0 or args.half_width_arcsec <= 0.0:
        raise ContractError("max-detectors and half-width-arcsec must be positive")
    raw_dir = discover_raw_dir(args.reduction_root)
    standard_apt = require_unique(
        [path for path in raw_dir.glob("apt_*.ecsv") if is_standard_apt(path)],
        "standard Beammap APT",
    )
    apt_paths = {mode: product_apt_path(standard_apt, mode) for mode in MODES}
    qc_paths = {mode: fit_qc_path(standard_apt, mode) for mode in MODES}
    for label, path in {**{f"{m}_apt": p for m, p in apt_paths.items()},
                        **{f"{m}_fit_qc": p for m, p in qc_paths.items()}}.items():
        if not path.is_file():
            raise ContractError(f"missing {label}: {path}")
    tables = {mode: Table.read(path, format="ascii.ecsv") for mode, path in apt_paths.items()}
    for mode, table in tables.items():
        require_columns(table, REQUIRED_APT_COLUMNS, f"{mode} APT")
    indices = {mode: uid_index(table, f"{mode} APT") for mode, table in tables.items()}
    common_uids = set.intersection(*(set(value) for value in indices.values()))
    if set(indices["standard"]) != common_uids:
        missing = sorted(set(indices["standard"]) - common_uids)
        raise ContractError(f"directional APTs lack standard detector UIDs: {missing[:20]}")
    observation = int_scalar(tables["standard"].meta.get("obsnum"))
    if observation < 0:
        raise ContractError("standard APT lacks a valid obsnum metadata value")
    for mode in MODES:
        realized = tables[mode].meta.get("beammap_direction_mode")
        if realized is not None and str(realized) != mode:
            raise ContractError(f"{mode} APT declares beammap_direction_mode={realized!r}")
    registry = require_unique(
        raw_dir.parent.rglob("beammap_direction_scan_registry_all.csv"),
        "all-mode direction scan registry",
    )
    scan = scan_summary(registry)
    allowlist, selection_authority = read_uid_allowlist(args.hero_selection)
    selected = select_detectors(
        tables["standard"], ARRAY_IDS[args.array], args.max_detectors, allowlist,
    )
    metric_rows = [
        detector_metrics(int(row["uid"]), tables, indices, scan)
        for row in selected
    ]
    metric_by_uid = {int(row["uid"]): row for row in metric_rows}
    fits_paths = {
        mode: discover_fits(raw_dir, args.array, mode)
        for mode in MODES
    }
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise ContractError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    selection_path = output_dir / "selected_detectors.ecsv"
    metrics_path = output_dir / "detector_metrics.ecsv"
    Table(rows=selected).write(selection_path, format="ascii.ecsv", overwrite=False)
    Table(rows=metric_rows).write(metrics_path, format="ascii.ecsv", overwrite=False)
    pdf_path = output_dir / f"split_direction_beammaps_o{observation}_{args.array}.pdf"
    with contextlib.ExitStack() as stack:
        products = {
            mode: FitsProduct(paths, stack)
            for mode, paths in fits_paths.items()
        }
        make_pdf(
            pdf_path, selected, products, tables, indices, metric_by_uid, scan,
            args.detectors_per_page, args.half_width_arcsec, observation, args.array,
        )
    input_paths = [*apt_paths.values(), *qc_paths.values(), registry]
    for paths in fits_paths.values():
        input_paths.extend(paths)
    if args.hero_selection is not None:
        input_paths.append(args.hero_selection.resolve())
    manifest = {
        "schema": "sci-align-001-split-direction-beammap-visualization-v2",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "observation_number": observation,
        "array": args.array,
        "selection": {
            "authority": selection_authority,
            "uses_directional_displacement": False,
            "standard_rules": [
                "array matches requested array",
                "standard flag == 0",
                "standard flag2 == 0 when present",
                "finite positive standard amplitude, widths, and S/N",
                "finite nonnegative standard centroid uncertainties",
            ],
            "automatic_order": "network-balanced round-robin of descending standard sig2noise",
            "requested_maximum": args.max_detectors,
            "selected_count": len(selected),
        },
        "layout": {
            "detectors_per_page": args.detectors_per_page,
            "maximum_supported_detectors_per_page": 2,
            "half_width_arcsec": args.half_width_arcsec,
            "maps_recentered": False,
            "map_color_scale": "common per detector across standard/left/right",
        },
        "scan": scan.__dict__,
        "uncertainty_scope": (
            "diagonal left/right beam-fit covariance only; "
            "no pixel independence claim"
        ),
        "position_authority": {
            "frame": "raw_altaz_detector_map",
            "centroid_columns": ["x_t_raw", "y_t_raw"],
            "fit_uncertainty_columns": ["x_t_err", "y_t_err"],
            "reason": (
                "detector FITS WCS and scan direction share the raw AltAz map frame; "
                "x_t/y_t are reference-subtracted and may be derotated"
            ),
        },
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(input_paths))
        ],
        "outputs": [pdf_path.name, selection_path.name, metrics_path.name],
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    output_checksums(
        output_dir,
        [pdf_path.name, selection_path.name, metrics_path.name, manifest_path.name],
    )
    print(
        f"visualization complete: obs={observation} array={args.array} "
        f"detectors={len(selected)} pages={math.ceil(len(selected) / args.detectors_per_page)} "
        f"output={output_dir}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--reduction-root", required=True, type=Path,
        help="Completed all-mode reduction root, observation directory, or raw directory",
    )
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--array", choices=sorted(ARRAY_IDS), default="a1100")
    result.add_argument("--max-detectors", type=int, default=100)
    result.add_argument(
        "--detectors-per-page", type=int, choices=(1, 2), default=1,
        help="One detector per page by default; never more than two",
    )
    result.add_argument("--half-width-arcsec", type=float, default=20.0)
    result.add_argument(
        "--hero-selection", type=Path,
        help=(
            "Optional ECSV/CSV table with uid column or JSON UID list fixed "
            "before split inspection"
        ),
    )
    return result


def main() -> int:
    try:
        run(parser().parse_args())
    except (ContractError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
