#!/usr/bin/env python3
"""Audit naive Beammap pixel support against one retained full PTC TOD.

This read-only SCI-ALIGN-001 diagnostic answers a deliberately narrow
question: are apparently disjoint or white neighboring pixels in a detector
Beammap explained by the accepted detector samples and Citlali's exact naive
nearest-pixel accumulation rule?

Scan direction is derived self-consistently from the PTC ``az_phys`` and
``TelTime`` trajectories.  No earlier split-direction registry is consumed.
The full PTC may come from a separate, provenance-bound replay of the same
observation; the manifest records that cross-run limitation explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from render_sci_align_001_split_direction_beammaps import (  # noqa: E402
    ARRAY_IDS,
    MODES,
    ContractError,
    FitsProduct,
    crop_mask,
    discover_fits,
    discover_raw_dir,
    image_coordinates,
    is_standard_apt,
    masked_signal,
    pixel_edges,
    product_apt_path,
    require_unique,
    robust_limits,
    row_values,
    spatial_wcs,
    uid_index,
    wcs_signature,
)


RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi
REQUIRED_PTC_VARS = {
    "signal", "flags", "weights", "det_lat", "det_lon", "TelTime",
    "TelUTC", "alt_phys", "az_phys", "scan_indices",
    "output_scan_index", "apt_uid", "apt_array", "apt_nw",
}
REQUIRED_APT_COLUMNS = {
    "uid", "array", "nw", "flag", "x_t_raw", "y_t_raw",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_columns(table: Table, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(table.colnames))
    if missing:
        raise ContractError(f"{label} lacks required columns: {missing}")


def read_float(variable: Any, key: Any = slice(None)) -> np.ndarray:
    value = variable[key]
    return np.asarray(np.ma.filled(value, np.nan), dtype=float)


def read_int(variable: Any, key: Any = slice(None)) -> np.ndarray:
    value = variable[key]
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ContractError(f"required integer variable {variable.name} is masked")
    result = np.asarray(value, dtype=np.int64)
    return result


def cxx_llround(values: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        raise ContractError("pixel rounding received non-finite coordinates")
    return np.where(
        values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5)
    ).astype(np.int64)


@dataclass
class PtcDetector:
    path: Path
    detector_index: int
    uid: int
    array: int
    nw: int
    apt_flag: int
    fruitloops_iter: int
    signal: np.ndarray
    flags: np.ndarray
    det_lat: np.ndarray
    det_lon: np.ndarray
    tel_time: np.ndarray
    tel_utc: np.ndarray
    alt_phys: np.ndarray
    az_phys: np.ndarray
    scan_indices: np.ndarray
    output_scan_index: np.ndarray
    weights: np.ndarray


def unique_integer(values: np.ndarray, label: str) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        raise ContractError(f"{label} contains non-finite values")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=0.0, rtol=0.0):
        raise ContractError(f"{label} contains non-integral values")
    if np.unique(rounded).size != rounded.size:
        raise ContractError(f"{label} contains duplicate identities")
    return rounded


def load_ptc_detector(path: Path, uid: int) -> PtcDetector:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"full PTC TOD is missing: {path}")
    with netCDF4.Dataset(path, mode="r") as dataset:
        missing = sorted(REQUIRED_PTC_VARS - set(dataset.variables))
        if missing:
            raise ContractError(f"full PTC TOD lacks required variables: {missing}")
        uids_raw = read_float(dataset.variables["apt_uid"])
        uids = unique_integer(uids_raw, "apt_uid")
        matches = np.flatnonzero(uids == uid)
        if matches.size != 1:
            raise ContractError(f"expected exactly one apt_uid={uid}; found {matches.size}")
        detector_index = int(matches[0])
        arrays = read_float(dataset.variables["apt_array"])
        networks = read_float(dataset.variables["apt_nw"])
        apt_flag = 0
        if "apt_flag" in dataset.variables:
            apt_flag = int(round(float(read_float(
                dataset.variables["apt_flag"], detector_index
            ))))
        fruitloops_iter = -1
        if "FRUITLOOPS_ITER" in dataset.variables:
            fruitloops_iter = int(read_int(
                dataset.variables["FRUITLOOPS_ITER"]
            ).reshape(-1)[0])
        signal = read_float(dataset.variables["signal"], (slice(None), detector_index))
        flags = read_float(dataset.variables["flags"], (slice(None), detector_index))
        det_lat = read_float(dataset.variables["det_lat"], (slice(None), detector_index))
        det_lon = read_float(dataset.variables["det_lon"], (slice(None), detector_index))
        tel_time = read_float(dataset.variables["TelTime"])
        tel_utc = read_float(dataset.variables["TelUTC"])
        alt_phys = read_float(dataset.variables["alt_phys"])
        az_phys = read_float(dataset.variables["az_phys"])
        scan_indices = read_int(dataset.variables["scan_indices"])
        output_scan_index = read_int(dataset.variables["output_scan_index"])
        weights = read_float(dataset.variables["weights"], (slice(None), detector_index))
    n_samples = signal.size
    for label, values in {
        "flags": flags, "det_lat": det_lat, "det_lon": det_lon,
        "TelTime": tel_time, "TelUTC": tel_utc,
        "alt_phys": alt_phys, "az_phys": az_phys,
    }.items():
        if values.shape != (n_samples,):
            raise ContractError(
                f"{label} shape {values.shape} does not match signal {(n_samples,)}"
            )
    if scan_indices.shape != (weights.size, 2):
        raise ContractError(
            f"scan_indices shape {scan_indices.shape} does not match weights {weights.size}"
        )
    if output_scan_index.shape != (weights.size,):
        raise ContractError("output_scan_index does not match the scan dimension")
    return PtcDetector(
        path=path, detector_index=detector_index, uid=uid,
        array=int(round(float(arrays[detector_index]))),
        nw=int(round(float(networks[detector_index]))), apt_flag=apt_flag,
        fruitloops_iter=fruitloops_iter, signal=signal, flags=flags,
        det_lat=det_lat, det_lon=det_lon, tel_time=tel_time,
        tel_utc=tel_utc, alt_phys=alt_phys, az_phys=az_phys,
        scan_indices=scan_indices, output_scan_index=output_scan_index,
        weights=weights,
    )


@dataclass
class ScanClassification:
    rows: list[dict[str, Any]]
    sample_direction: np.ndarray


def classify_scans(ptc: PtcDetector) -> ScanClassification:
    sample_direction = np.full(ptc.signal.size, "outside", dtype="U8")
    rows: list[dict[str, Any]] = []
    previous_end = -1
    for scan_row, ((start, end), original_scan) in enumerate(zip(
        ptc.scan_indices, ptc.output_scan_index, strict=True
    )):
        start_i, end_i = int(start), int(end)
        if start_i < 0 or end_i < start_i or end_i >= ptc.signal.size:
            raise ContractError(
                f"scan row {scan_row} has invalid inclusive bounds {start_i}:{end_i}"
            )
        if start_i != previous_end + 1:
            raise ContractError(
                f"scan row {scan_row} is not contiguous after sample {previous_end}"
            )
        previous_end = end_i
        slc = slice(start_i, end_i + 1)
        time = ptc.tel_time[slc]
        az = ptc.az_phys[slc]
        if np.any(~np.isfinite(time)) or np.any(~np.isfinite(az)):
            raise ContractError(f"scan row {scan_row} has non-finite time or azimuth")
        dt = np.diff(time)
        if np.any(dt <= 0.0):
            raise ContractError(f"scan row {scan_row} has non-increasing TelTime")
        centered_time = time - float(np.mean(time))
        denominator = float(np.dot(centered_time, centered_time))
        if denominator <= 0.0:
            raise ContractError(f"scan row {scan_row} has no time support")
        slope = float(np.dot(centered_time, az - float(np.mean(az))) / denominator)
        displacement = float(az[-1] - az[0])
        if slope == 0.0 or displacement == 0.0 or math.copysign(1.0, slope) != math.copysign(1.0, displacement):
            raise ContractError(
                f"scan row {scan_row} has ambiguous azimuth direction: "
                f"slope={slope} displacement={displacement}"
            )
        direction = "right" if slope > 0.0 else "left"
        sample_direction[slc] = direction
        rows.append({
            "scan_row_zero_based": scan_row,
            "output_scan_index_one_based": int(original_scan),
            "start_sample_inclusive": start_i,
            "end_sample_inclusive": end_i,
            "sample_count": end_i - start_i + 1,
            "start_time_sec": float(time[0]),
            "end_time_sec": float(time[-1]),
            "duration_sec": float(time[-1] - time[0]),
            "az_displacement_arcsec": displacement * RAD_TO_ARCSEC,
            "az_rate_arcsec_per_sec": slope * RAD_TO_ARCSEC,
            "direction": direction,
            "detector_weight": float(ptc.weights[scan_row]),
        })
    if previous_end != ptc.signal.size - 1:
        raise ContractError(
            f"scan table ends at {previous_end}, before final sample {ptc.signal.size - 1}"
        )
    if not {"left", "right"}.issubset(set(sample_direction)):
        raise ContractError("self-contained scan classification lacks left or right scans")
    return ScanClassification(rows=rows, sample_direction=sample_direction)


def map_tables(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Table], dict[str, dict[int, int]]]:
    standard = require_unique(
        [path for path in raw_dir.glob("apt_*.ecsv") if is_standard_apt(path)],
        "standard Beammap APT",
    )
    paths = {mode: product_apt_path(standard, mode) for mode in MODES}
    for mode, path in paths.items():
        if not path.is_file():
            raise ContractError(f"missing {mode} APT: {path}")
    tables = {mode: Table.read(path, format="ascii.ecsv") for mode, path in paths.items()}
    for mode, table in tables.items():
        require_columns(table, REQUIRED_APT_COLUMNS, f"{mode} APT")
    indices = {mode: uid_index(table, f"{mode} APT") for mode, table in tables.items()}
    return paths, tables, indices


def mapmaker_pixel_coordinates(
    det_lat: np.ndarray, det_lon: np.ndarray,
    shape: tuple[int, int], pixel_size_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    row = det_lat / pixel_size_rad + (shape[0] - 1) / 2.0
    column = det_lon / pixel_size_rad + (shape[1] - 1) / 2.0
    return row, column


def pixel_size_rad_from_wcs(wcs: Any) -> float:
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    scales = np.abs(np.diag(matrix))
    if scales.size != 2 or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ContractError("invalid detector map pixel scale")
    if not np.allclose(scales, scales[0], rtol=0.0, atol=1.0e-12):
        raise ContractError(f"non-square detector map pixels are unsupported: {scales}")
    unit = str(wcs.wcs.cunit[0]).lower()
    scale = float(scales[0])
    if unit in {"arcsec", "arcsecond", "arcseconds"}:
        return scale / RAD_TO_ARCSEC
    if unit in {"deg", "degree", "degrees"}:
        return math.radians(scale)
    if unit in {"rad", "radian", "radians"}:
        return scale
    raise ContractError(f"unsupported detector map WCS unit: {unit!r}")


def scan_weight_per_sample(ptc: PtcDetector) -> np.ndarray:
    result = np.full(ptc.signal.size, np.nan, dtype=float)
    for scan_row, (start, end) in enumerate(ptc.scan_indices):
        result[int(start):int(end) + 1] = ptc.weights[scan_row]
    return result


def shifted_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    src_y0, src_y1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
    src_x0, src_x1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return result
    result[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = mask[
        src_y0:src_y1, src_x0:src_x1
    ]
    return result


def overlap_metrics(hit_support: np.ndarray, map_support: np.ndarray) -> dict[str, Any]:
    intersection = int(np.sum(hit_support & map_support))
    union = int(np.sum(hit_support | map_support))
    hit_count = int(np.sum(hit_support))
    map_count = int(np.sum(map_support))
    return {
        "hit_supported_pixels": hit_count,
        "map_supported_pixels": map_count,
        "intersection_pixels": intersection,
        "hit_only_pixels": int(np.sum(hit_support & ~map_support)),
        "map_only_pixels": int(np.sum(~hit_support & map_support)),
        "jaccard": intersection / union if union else math.nan,
        "map_support_recall": intersection / map_count if map_count else math.nan,
        "hit_support_precision": intersection / hit_count if hit_count else math.nan,
    }


def best_registration(
    hit_support: np.ndarray, map_support: np.ndarray, radius: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_score: tuple[float, int, int, int] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = shifted_mask(hit_support, dy, dx)
            metrics = overlap_metrics(shifted, map_support)
            candidate = {
                "row_shift_pixels": dy, "column_shift_pixels": dx,
                **metrics,
            }
            score = (
                -math.inf if not math.isfinite(metrics["jaccard"])
                else metrics["jaccard"],
                -abs(dy) - abs(dx), -abs(dy), -abs(dx),
            )
            if best_score is None or score > best_score:
                best = candidate
                best_score = score
    if best is None:
        raise ContractError("registration search produced no candidates")
    return best


@dataclass
class ModeAudit:
    mode: str
    signal: np.ndarray
    weight: np.ndarray
    header: Any
    hit_count: np.ndarray
    accepted: np.ndarray
    pixel_row: np.ndarray
    pixel_column: np.ndarray
    metrics: dict[str, Any]


def audit_mode(
    mode: str, ptc: PtcDetector, direction: np.ndarray,
    signal: np.ndarray, weight: np.ndarray, header: Any,
    registration_radius: int,
) -> ModeAudit:
    wcs = spatial_wcs(header)
    pixel_size_rad = pixel_size_rad_from_wcs(wcs)
    row_float, column_float = mapmaker_pixel_coordinates(
        ptc.det_lat, ptc.det_lon, signal.shape, pixel_size_rad
    )
    finite_position = np.isfinite(row_float) & np.isfinite(column_float)
    row = np.full(row_float.shape, -1, dtype=np.int64)
    column = np.full(column_float.shape, -1, dtype=np.int64)
    row[finite_position] = cxx_llround(row_float[finite_position])
    column[finite_position] = cxx_llround(column_float[finite_position])
    inside = (
        finite_position & (row >= 0) & (row < signal.shape[0])
        & (column >= 0) & (column < signal.shape[1])
    )
    per_sample_weight = scan_weight_per_sample(ptc)
    mode_selected = np.ones(ptc.signal.size, dtype=bool)
    if mode != "standard":
        mode_selected = direction == mode
    accepted = (
        mode_selected & (ptc.apt_flag == 0) & (ptc.flags == 0.0)
        & np.isfinite(ptc.signal) & np.isfinite(per_sample_weight)
        & (per_sample_weight > 0.0) & inside
    )
    hit_count = np.zeros(signal.shape, dtype=np.int64)
    np.add.at(hit_count, (row[accepted], column[accepted]), 1)
    map_support = np.isfinite(signal) & np.isfinite(weight) & (weight > 0.0)
    raw_metrics = overlap_metrics(hit_count > 0, map_support)
    registered = best_registration(hit_count > 0, map_support, registration_radius)
    metrics = {
        "mode": mode,
        "total_samples": int(ptc.signal.size),
        "mode_selected_samples": int(np.sum(mode_selected)),
        "accepted_in_map_samples": int(np.sum(accepted)),
        "flagged_mode_samples": int(np.sum(mode_selected & (ptc.flags != 0.0))),
        "nonfinite_signal_mode_samples": int(np.sum(mode_selected & ~np.isfinite(ptc.signal))),
        "nonpositive_weight_mode_samples": int(np.sum(
            mode_selected & (~np.isfinite(per_sample_weight) | (per_sample_weight <= 0.0))
        )),
        "outside_map_mode_samples": int(np.sum(mode_selected & ~inside)),
        "raw": raw_metrics,
        "best_integer_registration": registered,
        "pixel_size_arcsec": pixel_size_rad * RAD_TO_ARCSEC,
    }
    return ModeAudit(
        mode=mode, signal=signal, weight=weight, header=header,
        hit_count=hit_count, accepted=accepted,
        pixel_row=row_float, pixel_column=column_float, metrics=metrics,
    )


def flattened_metrics(audits: dict[str, ModeAudit]) -> list[dict[str, Any]]:
    rows = []
    for mode in MODES:
        value = audits[mode].metrics
        row = {
            key: item for key, item in value.items()
            if key not in {"raw", "best_integer_registration"}
        }
        row.update({f"raw_{key}": item for key, item in value["raw"].items()})
        row.update({
            f"registered_{key}": item
            for key, item in value["best_integer_registration"].items()
        })
        rows.append(row)
    return rows


def support_category(hit: np.ndarray, map_support: np.ndarray) -> np.ndarray:
    # 0 neither, 1 agreement, 2 retained-hit only, 3 map-support only.
    return (
        (hit & map_support).astype(np.int8)
        + 2 * (hit & ~map_support).astype(np.int8)
        + 3 * (~hit & map_support).astype(np.int8)
    )


def render_map_page(
    pdf: PdfPages, ptc: PtcDetector, audits: dict[str, ModeAudit],
    apt_values: dict[str, dict[str, Any]], half_width: float,
) -> None:
    signatures = {mode: wcs_signature(audits[mode].header) for mode in MODES}
    if any(signatures[mode] != signatures["standard"] for mode in MODES[1:]):
        raise ContractError("standard/left/right detector maps do not share one WCS")
    shape = audits["standard"].signal.shape
    wcs = spatial_wcs(audits["standard"].header)
    x, y = image_coordinates(wcs, shape)
    cx, cy = apt_values["standard"]["x_t_raw"], apt_values["standard"]["y_t_raw"]
    ys, xs = crop_mask(x, y, cx, cy, half_width)
    xcrop, ycrop = x[xs], y[ys]
    xedge, yedge = pixel_edges(xcrop), pixel_edges(ycrop)
    extent = (xedge[0], xedge[1], yedge[0], yedge[1])
    images = [masked_signal(audits[mode].signal, audits[mode].weight)[ys, xs] for mode in MODES]
    limits = robust_limits(images)
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 11.2))
    category_cmap = ListedColormap(["white", "#202020", "#e69f00", "#cc79a7"])
    category_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], category_cmap.N)
    for column, mode in enumerate(MODES):
        audit = audits[mode]
        image = images[column]
        axes[0, column].imshow(
            image, origin="lower", extent=extent, interpolation="nearest",
            cmap="viridis", vmin=limits[0], vmax=limits[1], aspect="equal",
            rasterized=True,
        )
        chosen = audit.accepted
        world_x, world_y = wcs.pixel_to_world_values(
            audit.pixel_column[chosen], audit.pixel_row[chosen]
        )
        in_crop = (
            np.isfinite(world_x) & np.isfinite(world_y)
            & (np.abs(world_x - cx) <= half_width)
            & (np.abs(world_y - cy) <= half_width)
        )
        axes[0, column].plot(
            np.asarray(world_x)[in_crop], np.asarray(world_y)[in_crop], ".",
            color="white", markeredgecolor="black", markeredgewidth=0.15,
            markersize=1.8, alpha=0.65,
        )
        axes[0, column].plot(
            apt_values[mode]["x_t_raw"], apt_values[mode]["y_t_raw"],
            marker="o", markerfacecolor="none", markeredgecolor="red",
            markersize=6,
        )
        axes[0, column].set_title(
            f"{mode}: naive map + accepted sample positions", fontsize=9
        )
        map_support = (
            np.isfinite(audit.signal) & np.isfinite(audit.weight)
            & (audit.weight > 0.0)
        )
        category = support_category(audit.hit_count > 0, map_support)[ys, xs]
        axes[1, column].imshow(
            category, origin="lower", extent=extent, interpolation="nearest",
            cmap=category_cmap, norm=category_norm, aspect="equal",
            rasterized=True,
        )
        raw = audit.metrics["raw"]
        axes[1, column].set_title(
            f"raw support: J={raw['jaccard']:.3f}; "
            f"hit-only={raw['hit_only_pixels']} map-only={raw['map_only_pixels']}",
            fontsize=8,
        )
        hit_crop = audit.hit_count[ys, xs]
        axes[2, column].imshow(
            np.where(hit_crop > 0, np.log10(hit_crop), np.nan),
            origin="lower", extent=extent, interpolation="nearest",
            cmap="magma", aspect="equal", rasterized=True,
        )
        registered = audit.metrics["best_integer_registration"]
        axes[2, column].set_title(
            "log10 accepted hits; best support shift "
            f"(row,col)=({registered['row_shift_pixels']:+d},"
            f"{registered['column_shift_pixels']:+d}), J={registered['jaccard']:.3f}",
            fontsize=8,
        )
        for row in range(3):
            axes[row, column].set_xlabel("Az offset (arcsec)", fontsize=8)
            axes[row, column].set_ylabel("El offset (arcsec)", fontsize=8)
            axes[row, column].tick_params(labelsize=7)
    figure.suptitle(
        f"Obs 150819 UID {ptc.uid} (nw={ptc.nw}) naive sampling audit\n"
        "support colors: black=hit+map, orange=hit only, magenta=map only, white=neither",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(figure, dpi=160)
    plt.close(figure)


def robust_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else math.nan


def render_trajectory_page(
    pdf: PdfPages, ptc: PtcDetector, classification: ScanClassification,
) -> dict[str, Any]:
    lon_offset = (ptc.det_lon - ptc.az_phys) * RAD_TO_ARCSEC
    lat_offset = (ptc.det_lat - ptc.alt_phys) * RAD_TO_ARCSEC
    lon_center = float(np.nanmedian(lon_offset))
    lat_center = float(np.nanmedian(lat_offset))
    lon_residual = lon_offset - lon_center
    lat_residual = lat_offset - lat_center
    time0 = float(ptc.tel_time[0])
    rel_time = ptc.tel_time - time0
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    colors = np.where(classification.sample_direction == "left", "tab:blue", "tab:orange")
    stride = max(1, ptc.signal.size // 25000)
    sample = slice(None, None, stride)
    axes[0, 0].scatter(
        (ptc.az_phys[sample] - np.nanmedian(ptc.az_phys)) * RAD_TO_ARCSEC,
        (ptc.alt_phys[sample] - np.nanmedian(ptc.alt_phys)) * RAD_TO_ARCSEC,
        s=1.0, c=colors[sample], alpha=0.45, rasterized=True,
    )
    axes[0, 0].set_title("Reported telescope raster (median-centered)")
    axes[0, 0].set_xlabel("az_phys offset (arcsec)")
    axes[0, 0].set_ylabel("alt_phys offset (arcsec)")
    axes[0, 0].set_aspect("equal")
    axes[0, 1].scatter(
        (ptc.det_lon[sample] - np.nanmedian(ptc.det_lon)) * RAD_TO_ARCSEC,
        (ptc.det_lat[sample] - np.nanmedian(ptc.det_lat)) * RAD_TO_ARCSEC,
        s=1.0, c=colors[sample], alpha=0.45, rasterized=True,
    )
    axes[0, 1].set_title(f"Final detector trajectory, UID {ptc.uid} (median-centered)")
    axes[0, 1].set_xlabel("det_lon offset (arcsec)")
    axes[0, 1].set_ylabel("det_lat offset (arcsec)")
    axes[0, 1].set_aspect("equal")
    axes[1, 0].plot(rel_time, lon_residual, lw=0.45, label="lon - az_phys")
    axes[1, 0].plot(rel_time, lat_residual, lw=0.45, label="lat - alt_phys")
    axes[1, 0].set_title("Detector-minus-telescope residual after constant offset")
    axes[1, 0].set_xlabel("seconds from first PTC sample")
    axes[1, 0].set_ylabel("residual (arcsec)")
    axes[1, 0].legend(fontsize=8)
    along_steps = []
    detector_lon_steps = []
    detector_lat_steps = []
    pointing_step_residuals = []
    time_steps = []
    for start, end in ptc.scan_indices:
        slc = slice(int(start), int(end) + 1)
        az_step = np.diff(ptc.az_phys[slc]) * RAD_TO_ARCSEC
        alt_step = np.diff(ptc.alt_phys[slc]) * RAD_TO_ARCSEC
        det_lon_step = np.diff(ptc.det_lon[slc]) * RAD_TO_ARCSEC
        det_lat_step = np.diff(ptc.det_lat[slc]) * RAD_TO_ARCSEC
        along_steps.append(az_step)
        detector_lon_steps.append(det_lon_step)
        detector_lat_steps.append(det_lat_step)
        pointing_step_residuals.append(np.hypot(
            det_lon_step - az_step, det_lat_step - alt_step
        ))
        time_steps.append(np.diff(ptc.tel_time[slc]) * 1000.0)
    along_step = np.concatenate(along_steps)
    detector_lon_step = np.concatenate(detector_lon_steps)
    detector_lat_step = np.concatenate(detector_lat_steps)
    pointing_step_residual = np.concatenate(pointing_step_residuals)
    time_step = np.concatenate(time_steps)
    axes[1, 1].hist(along_step[np.isfinite(along_step)], bins=100, alpha=0.6, label="az step (arcsec)")
    twin = axes[1, 1].twinx()
    twin.hist(time_step[np.isfinite(time_step)], bins=100, alpha=0.35, color="tab:green", label="dt (ms)")
    axes[1, 1].set_title("Within-scan sample cadence and azimuth step")
    axes[1, 1].set_xlabel("azimuth step (arcsec); green axis is cadence")
    axes[1, 1].set_ylabel("az-step count")
    twin.set_ylabel("cadence count")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=8)
    figure.suptitle(
        "PTC detector pointing versus reported telescope trajectory\n"
        "blue=left scans; orange=right scans",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pdf.savefig(figure, dpi=160)
    plt.close(figure)
    return {
        "detector_minus_telescope_lon_median_arcsec": lon_center,
        "detector_minus_telescope_lat_median_arcsec": lat_center,
        "lon_residual_p16_arcsec": robust_percentile(lon_residual, 16.0),
        "lon_residual_p50_arcsec": robust_percentile(lon_residual, 50.0),
        "lon_residual_p84_arcsec": robust_percentile(lon_residual, 84.0),
        "lat_residual_p16_arcsec": robust_percentile(lat_residual, 16.0),
        "lat_residual_p50_arcsec": robust_percentile(lat_residual, 50.0),
        "lat_residual_p84_arcsec": robust_percentile(lat_residual, 84.0),
        "within_scan_dt_median_ms": robust_percentile(time_step, 50.0),
        "within_scan_dt_p99_ms": robust_percentile(time_step, 99.0),
        "within_scan_abs_az_step_median_arcsec": robust_percentile(np.abs(along_step), 50.0),
        "within_scan_abs_az_step_p99_arcsec": robust_percentile(np.abs(along_step), 99.0),
        "within_scan_abs_detector_lon_step_p99_arcsec": robust_percentile(
            np.abs(detector_lon_step), 99.0
        ),
        "within_scan_abs_detector_lat_step_p99_arcsec": robust_percentile(
            np.abs(detector_lat_step), 99.0
        ),
        "detector_minus_telescope_step_residual_p50_arcsec": robust_percentile(
            pointing_step_residual, 50.0
        ),
        "detector_minus_telescope_step_residual_p99_arcsec": robust_percentile(
            pointing_step_residual, 99.0
        ),
        "detector_minus_telescope_step_residual_max_arcsec": robust_percentile(
            pointing_step_residual, 100.0
        ),
    }


def output_checksums(output: Path, names: Sequence[str]) -> None:
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256_file(output / name)}  {name}\n" for name in sorted(names)
    ))


def run(args: argparse.Namespace) -> None:
    if args.uid < 0 or args.half_width_arcsec <= 0.0 or args.registration_radius_pixels < 0:
        raise ContractError("uid, half-width, and registration radius are outside their domains")
    ptc = load_ptc_detector(args.ptc_tod, args.uid)
    classification = classify_scans(ptc)
    raw_dir = discover_raw_dir(args.map_reduction_root)
    apt_paths, tables, indices = map_tables(raw_dir)
    for mode in MODES:
        if args.uid not in indices[mode]:
            raise ContractError(f"uid={args.uid} is absent from the {mode} APT")
    apt_values = {
        mode: row_values(tables[mode], indices[mode][args.uid])
        for mode in MODES
    }
    standard = apt_values["standard"]
    if ptc.array != ARRAY_IDS[args.array]:
        raise ContractError(
            f"requested array {args.array} has index {ARRAY_IDS[args.array]}, "
            f"but uid={args.uid} has PTC array index {ptc.array}"
        )
    if int(standard["array"]) != ptc.array or int(standard["nw"]) != ptc.nw:
        raise ContractError(
            "PTC and map APT disagree on detector array/network identity: "
            f"ptc=({ptc.array},{ptc.nw}) map=({standard['array']},{standard['nw']})"
        )
    fits_paths = {
        mode: discover_fits(raw_dir, args.array, mode) for mode in MODES
    }
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    audits: dict[str, ModeAudit] = {}
    used_fits: list[Path] = []
    with contextlib.ExitStack() as stack:
        products = {
            mode: FitsProduct(paths, stack) for mode, paths in fits_paths.items()
        }
        for mode in MODES:
            map_index = indices[mode][args.uid]
            flag = int(apt_values[mode]["flag"])
            signal, weight, header = products[mode].planes(map_index, flag)
            audits[mode] = audit_mode(
                mode, ptc, classification.sample_direction,
                signal, weight, header, args.registration_radius_pixels,
            )
            used_fits.extend(products[mode].paths)
        pdf_path = output / f"ptc_sampling_audit_o150819_uid{args.uid}.pdf"
        with PdfPages(pdf_path) as pdf:
            render_map_page(pdf, ptc, audits, apt_values, args.half_width_arcsec)
            trajectory_metrics = render_trajectory_page(pdf, ptc, classification)
            info = pdf.infodict()
            info["Title"] = f"SCI-ALIGN-001 PTC sampling audit: Obs 150819 UID {args.uid}"
            info["Subject"] = "Naive pixel support and telescope/detector sample trajectory"
    scan_path = output / "scan_classification.ecsv"
    support_path = output / "mode_support_metrics.ecsv"
    hit_path = output / "hit_counts.npz"
    Table(rows=classification.rows).write(scan_path, format="ascii.ecsv")
    Table(rows=flattened_metrics(audits)).write(support_path, format="ascii.ecsv")
    np.savez_compressed(hit_path, **{
        f"{mode}_hit_count": audits[mode].hit_count for mode in MODES
    })
    manifest = {
        "schema": "sci-align-001-ptc-naive-sampling-audit-v1",
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "observation_number": 150819,
        "uid": ptc.uid,
        "ptc_detector_index_zero_based": ptc.detector_index,
        "array": ptc.array,
        "network": ptc.nw,
        "fruitloops_iter": ptc.fruitloops_iter,
        "sample_count": int(ptc.signal.size),
        "scan_count": len(classification.rows),
        "left_scan_count": sum(row["direction"] == "left" for row in classification.rows),
        "right_scan_count": sum(row["direction"] == "right" for row in classification.rows),
        "scan_direction_authority": (
            "self-contained least-squares az_phys rate plus endpoint displacement, "
            "requiring strictly increasing TelTime and agreeing nonzero signs"
        ),
        "pixel_assignment_authority": (
            "Citlali NaiveMapmaker: llround(det_lat/pixel_size + (n_rows-1)/2), "
            "llround(det_lon/pixel_size + (n_cols-1)/2)"
        ),
        "cross_run_limitation": (
            "PTC sample/pointing authority and directional map products may come "
            "from separate replays of ObsNum 150819; identities and input hashes "
            "are bound here, and raw plus best constant-integer registration are "
            "reported without treating registration as physical motion"
        ),
        "mode_support_metrics": {mode: audits[mode].metrics for mode in MODES},
        "trajectory_metrics": trajectory_metrics,
        "inputs": [
            {
                "role": role, "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in [
                ("full_ptc_tod", ptc.path),
                *[(f"{mode}_apt", apt_paths[mode]) for mode in MODES],
                *[("detector_map_fits", path) for path in sorted(set(used_fits))],
            ]
        ],
        "outputs": [pdf_path.name, scan_path.name, support_path.name, hit_path.name],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    output_checksums(
        output,
        [pdf_path.name, scan_path.name, support_path.name, hit_path.name, manifest_path.name],
    )
    print("===== PTC SAMPLING AUDIT =====")
    print(
        f"obs=150819 uid={ptc.uid} detector_index={ptc.detector_index} "
        f"samples={ptc.signal.size} scans={len(classification.rows)} "
        f"left={manifest['left_scan_count']} right={manifest['right_scan_count']}"
    )
    for mode in MODES:
        metrics = audits[mode].metrics
        raw = metrics["raw"]
        registered = metrics["best_integer_registration"]
        print(
            f"mode={mode} accepted={metrics['accepted_in_map_samples']} "
            f"raw_jaccard={raw['jaccard']:.6f} "
            f"raw_hit_only={raw['hit_only_pixels']} raw_map_only={raw['map_only_pixels']} "
            f"best_shift=({registered['row_shift_pixels']:+d},"
            f"{registered['column_shift_pixels']:+d}) "
            f"registered_jaccard={registered['jaccard']:.6f}"
        )
    print(f"output={output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ptc-tod", required=True, type=Path)
    result.add_argument("--map-reduction-root", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--uid", type=int, default=199)
    result.add_argument("--array", choices=("a1100", "a1400", "a2000"), default="a1100")
    result.add_argument("--half-width-arcsec", type=float, default=25.0)
    result.add_argument("--registration-radius-pixels", type=int, default=8)
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
