#!/usr/bin/env python3
"""Audit selected same-run PTC samples against naive detector-map support.

This read-only SCI-ALIGN-001 diagnostic joins three explicit authorities for
one detector:

* telescope pointing and pointing offsets from a retained full PTC TOD;
* signal and flags from the map reduction's detector-specific PTC TOD; and
* per-scan detector weights from that same map reduction's PTC diagnostics.

The join uses the documented one-based original scan identity.  For detector
map grouping Citlali intentionally suppresses focal-plane detector offsets
during map accumulation, so pixel coordinates are reconstructed from
``az_phys``/``alt_phys`` plus the retained pointing offsets.  Only the scans
retained in the detector-specific TOD are tested; map support from other scans
is explicitly untested rather than classified as disagreement.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
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

from analyze_sci_align_001_ptc_sampling import (  # noqa: E402
    RAD_TO_ARCSEC,
    PtcDetector,
    classify_scans,
    cxx_llround,
    load_ptc_detector,
    map_tables,
    output_checksums,
    pixel_size_rad_from_wcs,
    read_float,
    read_int,
    sha256_file,
    write_json,
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
    pixel_edges,
    robust_limits,
    row_values,
    spatial_wcs,
    wcs_signature,
)


REQUIRED_SELECTED_VARS = {
    "detector_tod_uid", "detector_tod_array", "detector_tod_network",
    "detector_tod_fit_good", "detector_tod_fit_x_t_arcsec",
    "detector_tod_fit_y_t_arcsec", "detector_tod_slot_kind",
    "detector_tod_scan_index", "detector_tod_n_samples",
    "detector_tod_scan_inner_start_sample",
    "detector_tod_scan_inner_end_sample", "signal", "flags",
}
REQUIRED_DIAG_VARS = {
    "output_scan_index", "ptc_diag_uid", "ptc_detector_weight",
}
REQUIRED_REGISTRY_COLUMNS = {
    "scan_index", "direction", "selected", "mode",
}


def require_variables(dataset: netCDF4.Dataset, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(dataset.variables))
    if missing:
        raise ContractError(f"{label} lacks required variables: {missing}")


def unique_index(values: np.ndarray, identity: int, label: str) -> int:
    if values.ndim != 1 or values.size == 0:
        raise ContractError(f"{label} identity axis is empty or not one-dimensional")
    if len(set(int(value) for value in values)) != values.size:
        raise ContractError(f"{label} identity axis is non-unique")
    match = np.flatnonzero(values == identity)
    if match.size != 1:
        raise ContractError(
            f"expected exactly one {label} identity {identity}; found {match.size}"
        )
    return int(match[0])


@dataclass
class SelectedDetectorTod:
    path: Path
    detector_index: int
    uid: int
    array: int
    network: int
    fit_good: int
    fit_x_arcsec: float
    fit_y_arcsec: float
    slot_kind: np.ndarray
    scan_index: np.ndarray
    n_samples: np.ndarray
    inner_start: np.ndarray
    inner_end: np.ndarray
    signal: np.ndarray
    flags: np.ndarray


def load_selected_detector_tod(path: Path, uid: int) -> SelectedDetectorTod:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"selected detector TOD is missing: {path}")
    with netCDF4.Dataset(path, mode="r") as dataset:
        require_variables(dataset, REQUIRED_SELECTED_VARS, "selected detector TOD")
        uids = read_int(dataset.variables["detector_tod_uid"])
        detector_index = unique_index(uids, uid, "detector_tod_uid")
        selected = (detector_index, slice(None))
        slot_kind = read_int(dataset.variables["detector_tod_slot_kind"], selected)
        scan_index = read_int(dataset.variables["detector_tod_scan_index"], selected)
        n_samples = read_int(dataset.variables["detector_tod_n_samples"], selected)
        inner_start = read_int(
            dataset.variables["detector_tod_scan_inner_start_sample"], selected
        )
        inner_end = read_int(
            dataset.variables["detector_tod_scan_inner_end_sample"], selected
        )
        signal = read_float(
            dataset.variables["signal"], (detector_index, slice(None), slice(None))
        )
        flags = read_int(
            dataset.variables["flags"], (detector_index, slice(None), slice(None))
        )
        arrays = read_int(dataset.variables["detector_tod_array"])
        networks = read_int(dataset.variables["detector_tod_network"])
        fit_good = read_int(dataset.variables["detector_tod_fit_good"])
        fit_x = read_float(dataset.variables["detector_tod_fit_x_t_arcsec"])
        fit_y = read_float(dataset.variables["detector_tod_fit_y_t_arcsec"])
    slot_shape = slot_kind.shape
    for label, values in {
        "scan_index": scan_index, "n_samples": n_samples,
        "inner_start": inner_start, "inner_end": inner_end,
    }.items():
        if values.shape != slot_shape:
            raise ContractError(f"selected detector TOD {label} shape differs")
    if signal.ndim != 2 or flags.shape != signal.shape or signal.shape[0] != slot_shape[0]:
        raise ContractError("selected detector TOD signal/flag slot geometry differs")
    if np.any(n_samples > signal.shape[1]):
        raise ContractError("selected detector TOD n_samples exceeds n_samples_max")
    populated = n_samples > 0
    retained_lengths = inner_end[populated] - inner_start[populated] + 1
    if np.any(retained_lengths != n_samples[populated]):
        raise ContractError(
            "selected detector TOD inner-scan bounds disagree with n_samples"
        )
    return SelectedDetectorTod(
        path=path, detector_index=detector_index, uid=uid,
        array=int(arrays[detector_index]), network=int(networks[detector_index]),
        fit_good=int(fit_good[detector_index]),
        fit_x_arcsec=float(fit_x[detector_index]),
        fit_y_arcsec=float(fit_y[detector_index]),
        slot_kind=slot_kind, scan_index=scan_index, n_samples=n_samples,
        inner_start=inner_start, inner_end=inner_end,
        signal=signal, flags=flags,
    )


@dataclass
class SameRunWeights:
    path: Path
    scan_ids: np.ndarray
    weights: np.ndarray


def load_same_run_weights(path: Path, uid: int) -> SameRunWeights:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"same-run PTC diagnostics are missing: {path}")
    with netCDF4.Dataset(path, mode="r") as dataset:
        require_variables(dataset, REQUIRED_DIAG_VARS, "same-run PTC diagnostics")
        uids = read_int(dataset.variables["ptc_diag_uid"])
        detector_index = unique_index(uids, uid, "ptc_diag_uid")
        scan_ids = read_int(dataset.variables["output_scan_index"])
        weights = read_float(
            dataset.variables["ptc_detector_weight"],
            (slice(None), detector_index),
        )
    if scan_ids.ndim != 1 or weights.shape != scan_ids.shape:
        raise ContractError("same-run PTC diagnostic scan/weight geometry differs")
    if len(set(int(value) for value in scan_ids)) != scan_ids.size:
        raise ContractError("same-run PTC diagnostic scan identities are non-unique")
    return SameRunWeights(path=path, scan_ids=scan_ids, weights=weights)


def load_map_pointing(ptc: PtcDetector) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(ptc.path, mode="r") as dataset:
        for name in ("pointing_offset_alt", "pointing_offset_az"):
            if name not in dataset.variables:
                raise ContractError(f"full PTC TOD lacks required {name}")
            unit = str(getattr(dataset.variables[name], "units", "")).lower()
            if "arcsec" not in unit:
                raise ContractError(f"full PTC TOD {name} does not declare arcsec units")
        offset_alt = read_float(dataset.variables["pointing_offset_alt"])
        offset_az = read_float(dataset.variables["pointing_offset_az"])
    if offset_alt.shape != ptc.alt_phys.shape or offset_az.shape != ptc.az_phys.shape:
        raise ContractError("full PTC pointing-offset geometry differs from telescope pointing")
    return (
        ptc.alt_phys + offset_alt / RAD_TO_ARCSEC,
        ptc.az_phys + offset_az / RAD_TO_ARCSEC,
    )


@dataclass
class RegistryRow:
    scan_id: int
    direction: str
    selected: bool


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ContractError(f"invalid {label} boolean: {value!r}")


def load_direction_registry(path: Path) -> dict[int, RegistryRow]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"same-run direction registry is missing: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_REGISTRY_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ContractError(f"direction registry lacks columns: {missing}")
        rows = list(reader)
    result: dict[int, RegistryRow] = {}
    for row in rows:
        if row["mode"] != "all":
            raise ContractError(f"direction registry row has mode={row['mode']!r}")
        scan_id = int(row["scan_index"]) + 1
        direction = row["direction"]
        if direction not in {"left", "right"}:
            raise ContractError(f"scan {scan_id} has unsupported direction {direction!r}")
        if scan_id in result:
            raise ContractError(f"direction registry duplicates scan {scan_id}")
        result[scan_id] = RegistryRow(
            scan_id=scan_id, direction=direction,
            selected=parse_bool(row["selected"], "selected"),
        )
    if not result:
        raise ContractError("direction registry is empty")
    return result


def discover_selected_inputs(raw_dir: Path) -> tuple[Path, Path, Path]:
    tod_dir = raw_dir / "source_crossing_tod"
    selected = sorted(tod_dir.glob("*_ptc_detector_tod.nc"))
    diagnostics = sorted(tod_dir.glob("*_ptcdiag.nc"))
    registry = raw_dir / "beammap_direction_scan_registry_all.csv"
    if len(selected) != 1:
        raise ContractError(f"expected one selected detector TOD; found {len(selected)}")
    if len(diagnostics) != 1:
        raise ContractError(f"expected one same-run PTC diagnostic; found {len(diagnostics)}")
    return selected[0], diagnostics[0], registry


@dataclass
class JoinedScan:
    scan_id: int
    slot: int
    duplicate_slot_count: int
    slot_kind: int
    direction: str
    start: int
    end: int
    n_samples: int
    same_run_weight: float
    signal: np.ndarray
    flags: np.ndarray


def arrays_identical(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left, right, equal_nan=True)


def join_selected_scans(
    ptc: PtcDetector,
    selected: SelectedDetectorTod,
    weights: SameRunWeights,
    registry: dict[int, RegistryRow],
    sample_direction: np.ndarray,
) -> tuple[list[JoinedScan], int]:
    full_rows = {int(scan_id): row for row, scan_id in enumerate(ptc.output_scan_index)}
    if len(full_rows) != ptc.output_scan_index.size:
        raise ContractError("full PTC output scan identities are non-unique")
    weight_rows = {int(scan_id): row for row, scan_id in enumerate(weights.scan_ids)}
    by_scan: dict[int, JoinedScan] = {}
    duplicate_count = 0
    for slot in range(selected.scan_index.size):
        scan_id = int(selected.scan_index[slot])
        n_samples = int(selected.n_samples[slot])
        kind = int(selected.slot_kind[slot])
        if scan_id <= 0 or n_samples <= 0 or kind not in {1, 2}:
            continue
        if scan_id not in full_rows or scan_id not in weight_rows or scan_id not in registry:
            raise ContractError(f"selected scan {scan_id} lacks a join authority")
        registry_row = registry[scan_id]
        if not registry_row.selected:
            raise ContractError(f"selected TOD scan {scan_id} is disabled in map registry")
        full_row = full_rows[scan_id]
        start, end = (int(value) for value in ptc.scan_indices[full_row])
        full_length = end - start + 1
        if full_length != n_samples:
            raise ContractError(
                f"scan {scan_id} selected/full PTC length mismatch: "
                f"selected={n_samples} full={full_length}"
            )
        directions = set(sample_direction[start:end + 1])
        if directions != {registry_row.direction}:
            raise ContractError(
                f"scan {scan_id} registry/full-PTC direction mismatch: "
                f"registry={registry_row.direction} full={sorted(directions)}"
            )
        signal = np.asarray(selected.signal[slot, :n_samples], dtype=float)
        flags = np.asarray(selected.flags[slot, :n_samples], dtype=np.int64)
        joined = JoinedScan(
            scan_id=scan_id, slot=slot, duplicate_slot_count=1,
            slot_kind=kind, direction=registry_row.direction,
            start=start, end=end, n_samples=n_samples,
            same_run_weight=float(weights.weights[weight_rows[scan_id]]),
            signal=signal, flags=flags,
        )
        if scan_id in by_scan:
            previous = by_scan[scan_id]
            if not arrays_identical(previous.signal, signal) or not np.array_equal(
                previous.flags, flags
            ):
                raise ContractError(f"duplicate selected slots disagree for scan {scan_id}")
            previous.duplicate_slot_count += 1
            duplicate_count += 1
        else:
            by_scan[scan_id] = joined
    joined_scans = [by_scan[key] for key in sorted(by_scan)]
    if not joined_scans:
        raise ContractError("selected detector TOD has no populated slots")
    if not {row.direction for row in joined_scans}.issuperset({"left", "right"}):
        raise ContractError("selected detector TOD does not retain both directions")
    return joined_scans, duplicate_count


@dataclass
class ModeAudit:
    mode: str
    signal: np.ndarray
    weight: np.ndarray
    header: Any
    hit_count: np.ndarray
    accepted_lat: np.ndarray
    accepted_lon: np.ndarray
    metrics: dict[str, Any]


def map_support_metrics(hit_count: np.ndarray, map_support: np.ndarray) -> dict[str, Any]:
    hit = hit_count > 0
    hit_pixels = int(np.sum(hit))
    supported = int(np.sum(hit & map_support))
    return {
        "selected_hit_pixels": hit_pixels,
        "selected_hit_and_map_pixels": supported,
        "selected_hit_only_pixels": int(np.sum(hit & ~map_support)),
        "selected_hit_supported_fraction": supported / hit_pixels if hit_pixels else math.nan,
        "map_supported_pixels": int(np.sum(map_support)),
        "map_supported_not_tested_pixels": int(np.sum(map_support & ~hit)),
    }


def audit_mode(
    mode: str,
    ptc: PtcDetector,
    joined_scans: Sequence[JoinedScan],
    map_lat: np.ndarray,
    map_lon: np.ndarray,
    apt_flag: int,
    signal: np.ndarray,
    weight: np.ndarray,
    header: Any,
) -> ModeAudit:
    pixel_size = pixel_size_rad_from_wcs(spatial_wcs(header))
    hit_count = np.zeros(signal.shape, dtype=np.int64)
    accepted_lat: list[np.ndarray] = []
    accepted_lon: list[np.ndarray] = []
    selected_samples = accepted_samples = flagged_samples = 0
    nonfinite_samples = outside_samples = nonpositive_weight_samples = 0
    selected_scans = 0
    for joined in joined_scans:
        if mode != "standard" and joined.direction != mode:
            continue
        selected_scans += 1
        slc = slice(joined.start, joined.end + 1)
        lat = map_lat[slc]
        lon = map_lon[slc]
        row_float = lat / pixel_size + (signal.shape[0] - 1) / 2.0
        col_float = lon / pixel_size + (signal.shape[1] - 1) / 2.0
        finite_position = np.isfinite(row_float) & np.isfinite(col_float)
        row = np.full(joined.n_samples, -1, dtype=np.int64)
        col = np.full(joined.n_samples, -1, dtype=np.int64)
        row[finite_position] = cxx_llround(row_float[finite_position])
        col[finite_position] = cxx_llround(col_float[finite_position])
        inside = (
            finite_position & (row >= 0) & (row < signal.shape[0])
            & (col >= 0) & (col < signal.shape[1])
        )
        finite_signal = np.isfinite(joined.signal)
        good_flag = joined.flags == 0
        positive_weight = math.isfinite(joined.same_run_weight) and joined.same_run_weight > 0.0
        accepted = good_flag & finite_signal & inside & (apt_flag == 0) & positive_weight
        selected_samples += joined.n_samples
        accepted_samples += int(np.sum(accepted))
        flagged_samples += int(np.sum(~good_flag))
        nonfinite_samples += int(np.sum(~finite_signal))
        outside_samples += int(np.sum(~inside))
        if not positive_weight:
            nonpositive_weight_samples += joined.n_samples
        np.add.at(hit_count, (row[accepted], col[accepted]), 1)
        accepted_lat.append(lat[accepted])
        accepted_lon.append(lon[accepted])
    if selected_scans == 0 or accepted_samples == 0:
        raise ContractError(f"mode {mode} has no selected scan/sample support")
    map_support = np.isfinite(signal) & np.isfinite(weight) & (weight > 0.0)
    metrics = {
        "mode": mode,
        "selected_scan_count": selected_scans,
        "selected_sample_count": selected_samples,
        "accepted_sample_count": accepted_samples,
        "flagged_sample_count": flagged_samples,
        "nonfinite_signal_sample_count": nonfinite_samples,
        "outside_map_sample_count": outside_samples,
        "nonpositive_weight_sample_count": nonpositive_weight_samples,
        "pixel_size_arcsec": pixel_size * RAD_TO_ARCSEC,
        **map_support_metrics(hit_count, map_support),
    }
    return ModeAudit(
        mode=mode, signal=signal, weight=weight, header=header,
        hit_count=hit_count,
        accepted_lat=np.concatenate(accepted_lat),
        accepted_lon=np.concatenate(accepted_lon),
        metrics=metrics,
    )


def support_category(hit: np.ndarray, map_support: np.ndarray) -> np.ndarray:
    # 0 neither, 1 selected hit + map, 2 selected hit only, 3 map not tested.
    return (
        (hit & map_support).astype(np.int8)
        + 2 * (hit & ~map_support).astype(np.int8)
        + 3 * (~hit & map_support).astype(np.int8)
    )


def render_support_page(
    pdf: PdfPages,
    ptc: PtcDetector,
    audits: dict[str, ModeAudit],
    apt_values: dict[str, dict[str, Any]],
    half_width: float,
) -> None:
    signatures = {mode: wcs_signature(audits[mode].header) for mode in MODES}
    if any(signatures[mode] != signatures["standard"] for mode in MODES[1:]):
        raise ContractError("standard/left/right maps do not share one WCS")
    shape = audits["standard"].signal.shape
    wcs = spatial_wcs(audits["standard"].header)
    x, y = image_coordinates(wcs, shape)
    cx = float(apt_values["standard"]["x_t_raw"])
    cy = float(apt_values["standard"]["y_t_raw"])
    ys, xs = crop_mask(x, y, cx, cy, half_width)
    xedge, yedge = pixel_edges(x[xs]), pixel_edges(y[ys])
    extent = (xedge[0], xedge[1], yedge[0], yedge[1])
    images = [
        masked_signal(audits[mode].signal, audits[mode].weight)[ys, xs]
        for mode in MODES
    ]
    limits = robust_limits(images)
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 11.2))
    category_cmap = ListedColormap(["white", "#202020", "#e69f00", "#cc79a7"])
    category_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], category_cmap.N)
    for column, mode in enumerate(MODES):
        audit = audits[mode]
        axes[0, column].imshow(
            images[column], origin="lower", extent=extent,
            interpolation="nearest", cmap="viridis", vmin=limits[0],
            vmax=limits[1], aspect="equal", rasterized=True,
        )
        # The map WCS world values are the retained tangent coordinates; plot
        # those values directly after converting radians to arcseconds.
        sample_x = audit.accepted_lon * RAD_TO_ARCSEC
        sample_y = audit.accepted_lat * RAD_TO_ARCSEC
        in_crop = (
            np.isfinite(sample_x) & np.isfinite(sample_y)
            & (np.abs(sample_x - cx) <= half_width)
            & (np.abs(sample_y - cy) <= half_width)
        )
        axes[0, column].plot(
            sample_x[in_crop], sample_y[in_crop], ".", color="white",
            markeredgecolor="black", markeredgewidth=0.15,
            markersize=2.0, alpha=0.75,
        )
        axes[0, column].plot(
            apt_values[mode]["x_t_raw"], apt_values[mode]["y_t_raw"],
            marker="o", markerfacecolor="none", markeredgecolor="red",
            markersize=6,
        )
        axes[0, column].set_title(
            f"{mode}: map + accepted same-run selected samples", fontsize=8.5
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
        metric = audit.metrics
        axes[1, column].set_title(
            f"selected-hit support={metric['selected_hit_supported_fraction']:.4f}; "
            f"hit-only={metric['selected_hit_only_pixels']}", fontsize=8.5,
        )
        hit_crop = audit.hit_count[ys, xs]
        axes[2, column].imshow(
            np.where(hit_crop > 0, np.log10(hit_crop), np.nan),
            origin="lower", extent=extent, interpolation="nearest",
            cmap="magma", aspect="equal", rasterized=True,
        )
        axes[2, column].set_title(
            f"log10 hits from {metric['selected_scan_count']} distinct retained scans",
            fontsize=8.5,
        )
        for row in range(3):
            axes[row, column].set_xlabel("Az offset (arcsec)", fontsize=8)
            axes[row, column].set_ylabel("El offset (arcsec)", fontsize=8)
            axes[row, column].tick_params(labelsize=7)
    figure.suptitle(
        f"Obs 150819 UID {ptc.uid}: same-run selected-scan naive support\n"
        "black=selected hit + map; orange=selected hit only; "
        "magenta=map support from untested or selected scans; white=neither",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(figure, dpi=160)
    plt.close(figure)


def render_join_page(
    pdf: PdfPages,
    ptc: PtcDetector,
    joined_scans: Sequence[JoinedScan],
    map_lat: np.ndarray,
    map_lon: np.ndarray,
    audits: dict[str, ModeAudit],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    colors = {"left": "tab:blue", "right": "tab:orange"}
    rows = []
    for joined in joined_scans:
        slc = slice(joined.start, joined.end + 1)
        good = (joined.flags == 0) & np.isfinite(joined.signal)
        axes[0, 0].plot(
            map_lon[slc][good] * RAD_TO_ARCSEC,
            map_lat[slc][good] * RAD_TO_ARCSEC,
            ".", color=colors[joined.direction], markersize=1.8,
            alpha=0.7, rasterized=True,
        )
        rows.append({
            "scan_id": joined.scan_id,
            "direction": joined.direction,
            "accepted_fraction": float(np.mean(good)),
            "flagged_fraction": float(np.mean(joined.flags != 0)),
            "weight": joined.same_run_weight,
        })
    axes[0, 0].set_title("Joined map-coordinate tracks (blue=left, orange=right)")
    axes[0, 0].set_xlabel("Az offset (arcsec)")
    axes[0, 0].set_ylabel("El offset (arcsec)")
    axes[0, 0].set_aspect("equal")
    scan_id = np.asarray([row["scan_id"] for row in rows])
    accepted_fraction = np.asarray([row["accepted_fraction"] for row in rows])
    bar_colors = [colors[row["direction"]] for row in rows]
    axes[0, 1].bar(scan_id, accepted_fraction, color=bar_colors, width=1.5)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Unflagged finite fraction by retained scan")
    axes[0, 1].set_xlabel("one-based scan identity")
    axes[0, 1].set_ylabel("fraction")
    weights = np.asarray([row["weight"] for row in rows])
    axes[1, 0].plot(scan_id, weights, "o", ms=4)
    axes[1, 0].set_title("Same-run detector map weight by retained scan")
    axes[1, 0].set_xlabel("one-based scan identity")
    axes[1, 0].set_ylabel("PTC detector weight")
    modes = list(MODES)
    fraction = [audits[mode].metrics["selected_hit_supported_fraction"] for mode in modes]
    hit_only = [audits[mode].metrics["selected_hit_only_pixels"] for mode in modes]
    bars = axes[1, 1].bar(modes, fraction, color=["0.3", "tab:blue", "tab:orange"])
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Selected hit pixels retained by corresponding map")
    axes[1, 1].set_ylabel("supported fraction")
    for bar, count in zip(bars, hit_only, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.02, bar.get_height() + 0.025),
            f"hit-only={count}", ha="center", va="bottom", fontsize=8,
        )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=8)
    figure.suptitle(
        f"Explicit scan-identity join QA: Obs 150819 UID {ptc.uid}\n"
        "signal/flags/weights are from the map reduction; pointing is from the full PTC replay",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pdf.savefig(figure, dpi=160)
    plt.close(figure)


def joined_scan_rows(joined_scans: Sequence[JoinedScan]) -> list[dict[str, Any]]:
    return [{
        "scan_id_one_based": row.scan_id,
        "retained_slot_zero_based": row.slot,
        "duplicate_slot_count": row.duplicate_slot_count,
        "slot_kind": row.slot_kind,
        "direction": row.direction,
        "full_ptc_start_sample_inclusive": row.start,
        "full_ptc_end_sample_inclusive": row.end,
        "sample_count": row.n_samples,
        "same_run_weight": row.same_run_weight,
        "flagged_sample_count": int(np.sum(row.flags != 0)),
        "finite_signal_sample_count": int(np.sum(np.isfinite(row.signal))),
    } for row in joined_scans]


def run(args: argparse.Namespace) -> None:
    if args.uid < 0 or args.half_width_arcsec <= 0.0:
        raise ContractError("uid and half-width are outside their domains")
    ptc = load_ptc_detector(args.full_ptc_tod, args.uid)
    classification = classify_scans(ptc)
    map_lat, map_lon = load_map_pointing(ptc)
    raw_dir = discover_raw_dir(args.map_reduction_root)
    selected_path, diag_path, registry_path = discover_selected_inputs(raw_dir)
    selected = load_selected_detector_tod(selected_path, args.uid)
    weights = load_same_run_weights(diag_path, args.uid)
    registry = load_direction_registry(registry_path)
    if selected.array != ARRAY_IDS[args.array]:
        raise ContractError(
            f"requested array {args.array} disagrees with selected TOD array {selected.array}"
        )
    if (selected.array, selected.network) != (ptc.array, ptc.nw):
        raise ContractError("full PTC and selected TOD detector identities disagree")
    joined_scans, duplicate_slots = join_selected_scans(
        ptc, selected, weights, registry, classification.sample_direction
    )
    apt_paths, apt_tables, apt_indices = map_tables(raw_dir)
    apt_values = {}
    for mode in MODES:
        if args.uid not in apt_indices[mode]:
            raise ContractError(f"uid={args.uid} is absent from {mode} APT")
        apt_values[mode] = row_values(
            apt_tables[mode], apt_indices[mode][args.uid]
        )
    standard = apt_values["standard"]
    if int(standard["array"]) != selected.array or int(standard["nw"]) != selected.network:
        raise ContractError("selected TOD and map APT detector identities disagree")
    fits_paths = {mode: discover_fits(raw_dir, args.array, mode) for mode in MODES}
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    audits: dict[str, ModeAudit] = {}
    used_fits: list[Path] = []
    with contextlib.ExitStack() as stack:
        products = {mode: FitsProduct(fits_paths[mode], stack) for mode in MODES}
        for mode in MODES:
            map_index = apt_indices[mode][args.uid]
            signal, map_weight, header = products[mode].planes(
                map_index, int(apt_values[mode]["flag"])
            )
            audits[mode] = audit_mode(
                mode, ptc, joined_scans, map_lat, map_lon,
                int(standard["flag"]), signal, map_weight, header,
            )
            used_fits.extend(products[mode].paths)
        pdf_path = output / f"selected_sampling_join_o150819_uid{args.uid}.pdf"
        with PdfPages(pdf_path) as pdf:
            render_support_page(
                pdf, ptc, audits, apt_values, args.half_width_arcsec
            )
            render_join_page(pdf, ptc, joined_scans, map_lat, map_lon, audits)
            info = pdf.infodict()
            info["Title"] = f"SCI-ALIGN-001 selected sampling join: Obs 150819 UID {args.uid}"
            info["Subject"] = "Same-run selected PTC support versus naive maps"
    scan_path = output / "selected_scan_join.ecsv"
    metrics_path = output / "mode_selected_support.ecsv"
    hit_path = output / "selected_hit_counts.npz"
    Table(rows=joined_scan_rows(joined_scans)).write(scan_path, format="ascii.ecsv")
    Table(rows=[audits[mode].metrics for mode in MODES]).write(
        metrics_path, format="ascii.ecsv"
    )
    np.savez_compressed(hit_path, **{
        f"{mode}_selected_hit_count": audits[mode].hit_count for mode in MODES
    })
    manifest = {
        "schema": "sci-align-001-selected-sampling-join-v1",
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "observation_number": 150819,
        "uid": args.uid,
        "array": selected.array,
        "network": selected.network,
        "full_ptc_sample_count": int(ptc.signal.size),
        "full_ptc_scan_count": int(ptc.output_scan_index.size),
        "selected_slot_count": int(np.sum(selected.n_samples > 0)),
        "distinct_joined_scan_count": len(joined_scans),
        "duplicate_selected_slot_count": duplicate_slots,
        "joined_left_scan_count": sum(row.direction == "left" for row in joined_scans),
        "joined_right_scan_count": sum(row.direction == "right" for row in joined_scans),
        "identity_contract": (
            "one-based detector_tod_scan_index joined to one-based full-PTC and "
            "PTC-diagnostic output_scan_index; same-run registry zero-based scan_index "
            "is converted explicitly to one-based identity"
        ),
        "map_pointing_contract": (
            "detector grouping suppresses detector focal-plane offsets during map accumulation; "
            "map lat/lon are full-PTC alt_phys/az_phys plus retained arcsec pointing offsets"
        ),
        "scope_limitation": (
            "Only distinct scans retained in the map reduction's detector-specific TOD are tested. "
            "Map-supported pixels without a selected hit may be supported by any unretained scan and "
            "are not disagreement. Pointing comes from a separate replay of the same observation."
        ),
        "mode_support_metrics": {mode: audits[mode].metrics for mode in MODES},
        "inputs": [{
            "role": role, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        } for role, path in [
            ("full_ptc_tod", ptc.path),
            ("same_run_selected_detector_tod", selected.path),
            ("same_run_ptc_diagnostics", weights.path),
            ("same_run_direction_registry", registry_path.resolve()),
            *[(f"{mode}_apt", apt_paths[mode]) for mode in MODES],
            *[("detector_map_fits", path) for path in sorted(set(used_fits))],
        ]],
        "outputs": [pdf_path.name, scan_path.name, metrics_path.name, hit_path.name],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    output_checksums(
        output,
        [pdf_path.name, scan_path.name, metrics_path.name, hit_path.name, manifest_path.name],
    )
    print("===== SAME-RUN SELECTED SAMPLING JOIN =====")
    print(
        f"obs=150819 uid={args.uid} selected_slots={manifest['selected_slot_count']} "
        f"distinct_scans={len(joined_scans)} duplicates={duplicate_slots} "
        f"left={manifest['joined_left_scan_count']} right={manifest['joined_right_scan_count']}"
    )
    for mode in MODES:
        metric = audits[mode].metrics
        print(
            f"mode={mode} scans={metric['selected_scan_count']} "
            f"accepted={metric['accepted_sample_count']} "
            f"selected_hit_pixels={metric['selected_hit_pixels']} "
            f"hit_only={metric['selected_hit_only_pixels']} "
            f"supported_fraction={metric['selected_hit_supported_fraction']:.6f}"
        )
    print(f"output={output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--full-ptc-tod", required=True, type=Path)
    result.add_argument("--map-reduction-root", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--uid", type=int, default=199)
    result.add_argument("--array", choices=sorted(ARRAY_IDS), default="a1100")
    result.add_argument("--half-width-arcsec", type=float, default=25.0)
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
