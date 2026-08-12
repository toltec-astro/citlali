#!/usr/bin/env python3
"""Test time-lag and direction-sign models with retained Lissajous pointings.

This read-only SCI-ALIGN-001 diagnostic reconstructs velocity-sector maps from
retained pointing PTC timestreams.  It compares four empirical map-centroid
models without changing Citlali products or prescribing a correction:

* constant centroid;
* scalar time lag, ``c = c0 + tau * v``;
* independent map-axis direction-sign terms; and
* the joint lag plus direction-sign model.

The inventory command freezes product identities before sector centroids are
measured.  The run command accepts only that checksum-bound inventory.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import multiprocessing
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from analyze_sci_align_001_split_direction_transfer import (  # noqa: E402
    fit_gaussian_core,
)


RAD_TO_ARCSEC = 206264.80624709636
SECTOR_COUNT = 8
MODEL_NAMES = ("constant", "time_lag", "axis_sign", "joint")
MODEL_FIELDS = (
    "model", "sector_count", "coordinate_count", "parameter_count", "rank",
    "condition_number", "rms_arcsec", "bic", "x0_arcsec", "y0_arcsec",
    "tau_sec", "tau_ms", "h_az_arcsec", "h_el_arcsec",
)
PDF_METADATA = {
    "Creator": "SCI-ALIGN-001 Lissajous pointing diagnostic",
    "CreationDate": datetime.datetime(2026, 8, 10, tzinfo=datetime.timezone.utc),
    "ModDate": datetime.datetime(2026, 8, 10, tzinfo=datetime.timezone.utc),
}


class ContractError(RuntimeError):
    """An input or fitted product violates the diagnostic contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_checksums(root: Path, names: Iterable[str]) -> None:
    lines = [f"{sha256_file(root / name)}  {name}" for name in sorted(names)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def read_scalar(variable: Any, index: int | None = None) -> float:
    value = variable[:] if index is None else variable[index]
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ContractError(f"expected scalar from {variable.name}")
    return float(array.reshape(-1)[0])


def ppt_a1100(path: Path) -> dict[str, float]:
    table = Table.read(path)
    rows = table[np.asarray(table["array"], dtype=int) == 0]
    if len(rows) != 1:
        raise ContractError(f"{path}: expected exactly one a1100 PPT row")
    return {
        "snr_a1100": float(rows[0]["sig2noise"]),
        "ppt_x_arcsec": float(rows[0]["x_t"]),
        "ppt_y_arcsec": float(rows[0]["y_t"]),
    }


def ptc_summary(path: Path) -> dict[str, float | int | str]:
    with netCDF4.Dataset(path) as dataset:
        required = {
            "signal", "flags", "weights", "scan_indices", "apt_array",
            "apt_flag", "apt_x_t", "apt_y_t", "TelTime", "TelElAct",
            "az_phys", "alt_phys", "pointing_offset_az",
            "pointing_offset_alt",
        }
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ContractError(f"{path}: missing PTC variables {missing}")
        time = np.asarray(dataset.variables["TelTime"][:], dtype=float)
        elevation = np.asarray(dataset.variables["TelElAct"][:], dtype=float)
        scans = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        if time.ndim != 1 or elevation.shape != time.shape:
            raise ContractError(f"{path}: invalid telescope geometry")
        if scans.ndim != 2 or scans.shape[1] != 2:
            raise ContractError(f"{path}: invalid scan_indices geometry")
        speeds: list[np.ndarray] = []
        angles: list[np.ndarray] = []
        az = np.asarray(dataset.variables["az_phys"][:], dtype=float)
        alt = np.asarray(dataset.variables["alt_phys"][:], dtype=float)
        for start, stop in scans:
            sl = slice(int(start), int(stop) + 1)
            t = time[sl]
            if t.size < 3 or np.any(np.diff(t) <= 0):
                raise ContractError(f"{path}: invalid time support in scan")
            vx = np.gradient(az[sl] * RAD_TO_ARCSEC, t)
            vy = np.gradient(alt[sl] * RAD_TO_ARCSEC, t)
            speed = np.hypot(vx, vy)
            good = np.isfinite(speed) & (speed > 5.0)
            speeds.append(speed[good])
            angles.append(np.arctan2(vy[good], vx[good]))
        speed = np.concatenate(speeds)
        angle = np.concatenate(angles)
        sectors = sector_index(angle)
        return {
            "ptc_sample_count": int(time.size),
            "ptc_scan_count": int(scans.shape[0]),
            "detector_count": int(dataset.dimensions["n_dets"].size),
            "mean_elevation_deg": float(np.degrees(np.mean(elevation))),
            "median_speed_arcsec_s": float(np.median(speed)),
            "p95_speed_arcsec_s": float(np.percentile(speed, 95.0)),
            "populated_velocity_sector_count": int(np.unique(sectors).size),
        }


def sector_index(angle_rad: np.ndarray) -> np.ndarray:
    width = 2.0 * math.pi / SECTOR_COUNT
    return np.floor(np.mod(angle_rad + 0.5 * width, 2.0 * math.pi) / width).astype(int)


def discover_products(root: Path, obsnum: int) -> tuple[Path, Path]:
    ptc = sorted(root.rglob(f"toltec_commissioning_pointing_{obsnum}_ptc_timestream.nc"))
    ppt = sorted(root.rglob(f"ppt_commissioning_pointing_{obsnum}_citlali.ecsv"))
    if len(ptc) != 1 or len(ppt) != 1:
        raise ContractError(
            f"obs {obsnum}: expected one PTC and one PPT, found {len(ptc)} and {len(ppt)}"
        )
    return ptc[0].resolve(), ppt[0].resolve()


def inventory(args: argparse.Namespace) -> None:
    root = args.standard_trial_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    pairs = [
        (131925, 131920), (131925, 131926),
        (133543, 133542), (133543, 133544),
        (135397, 135396), (135397, 135398),
        (136279, 136278), (136279, 136280),
        (150819, 150818), (150819, 150820),
        (151126, 151125), (151126, 151127),
        (151600, 151599), (151600, 151601),
        (151950, 151949), (151950, 151951),
        (152451, 152450), (152451, 152452),
        (152882, 152881), (152882, 152883),
        (148670, 148669), (148670, 148671),
    ]
    rows: list[dict[str, Any]] = []
    for beammap, pointing in pairs:
        ptc, ppt = discover_products(root, pointing)
        row: dict[str, Any] = {
            "beammap_obsnum": beammap,
            "pointing_obsnum": pointing,
            "selection_role": (
                "anchor" if beammap in {148670, 150819} else "complete_bracket"
            ),
            "ptc_path": str(ptc),
            "ptc_sha256": sha256_file(ptc),
            "ppt_path": str(ppt),
            "ppt_sha256": sha256_file(ppt),
        }
        row.update(ppt_a1100(ppt))
        row.update(ptc_summary(ptc))
        row["brightness_stratum"] = (
            "high_snr" if row["snr_a1100"] >= 50.0 else "secondary"
        )
        rows.append(row)
    fieldnames = list(rows[0])
    with (output / "selected_pointings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "sci-align-001-lissajous-pointing-selection-v1",
        "selection_rule": (
            "both immediately bracketing 3C273 pointings for every analyzed "
            "beammap having complete retained standard-trial PTC and PPT products"
        ),
        "selection_is_independent_of_directional_centroids": True,
        "standard_trial_root": str(root),
        "row_count": len(rows),
        "rows": rows,
    }
    write_json(output / "selected_pointings.json", manifest)
    protocol = {
        "schema": "sci-align-001-lissajous-pointing-protocol-v1",
        "array": "a1100",
        "map_pixel_arcsec": 2.0,
        "map_half_width_arcsec": 80.0,
        "fit_half_width_arcsec": 15.0,
        "velocity_sector_count": SECTOR_COUNT,
        "minimum_speed_arcsec_s": 5.0,
        "model_names": list(MODEL_NAMES),
        "time_lag_model": "centroid = intercept + tau * map_velocity",
        "axis_sign_model": (
            "x=x0+h_az*sign(vx), y=y0+h_el*sign(vy); empirical map axes"
        ),
        "interpretation_limits": [
            "descriptive model comparison; sector-centroid covariance is unavailable",
            "does not identify a physical encoder, secondary, or timestamp mechanism",
            "does not prescribe a pointing or timing correction",
        ],
    }
    write_json(output / "frozen_protocol.json", protocol)
    write_checksums(output, [
        "frozen_protocol.json", "selected_pointings.csv", "selected_pointings.json"
    ])
    print(f"selection frozen: pointings={len(rows)} output={output}")


@dataclass
class SectorMap:
    weighted_signal: np.ndarray
    weight: np.ndarray
    hit_count: np.ndarray
    velocity_weight: float = 0.0
    velocity_x_sum: float = 0.0
    velocity_y_sum: float = 0.0
    accepted: int = 0


def empty_sector(shape: tuple[int, int]) -> SectorMap:
    return SectorMap(
        np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int64)
    )


def scan_velocity(time: np.ndarray, az: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ContractError("scan telescope time is not strictly increasing")
    return (
        np.gradient(az * RAD_TO_ARCSEC, time),
        np.gradient(alt * RAD_TO_ARCSEC, time),
    )


def reconstruct_sectors(
    ptc_path: Path,
    pixel_arcsec: float,
    half_width_arcsec: float,
    minimum_speed: float,
) -> tuple[list[SectorMap], dict[str, Any]]:
    n_side = int(round(2.0 * half_width_arcsec / pixel_arcsec)) + 1
    shape = (n_side, n_side)
    sectors = [empty_sector(shape) for _ in range(SECTOR_COUNT)]
    full = empty_sector(shape)
    with netCDF4.Dataset(ptc_path) as dataset:
        scans = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        time_all = np.asarray(dataset.variables["TelTime"][:], dtype=float)
        az_all = np.asarray(dataset.variables["az_phys"][:], dtype=float)
        alt_all = np.asarray(dataset.variables["alt_phys"][:], dtype=float)
        elevation_all = np.asarray(dataset.variables["TelElAct"][:], dtype=float)
        po_az_all = np.asarray(dataset.variables["pointing_offset_az"][:], dtype=float)
        po_alt_all = np.asarray(dataset.variables["pointing_offset_alt"][:], dtype=float)
        apt_array = np.asarray(dataset.variables["apt_array"][:], dtype=int)
        apt_flag = np.asarray(dataset.variables["apt_flag"][:], dtype=int)
        apt_x = np.asarray(dataset.variables["apt_x_t"][:], dtype=float)
        apt_y = np.asarray(dataset.variables["apt_y_t"][:], dtype=float)
        det_keep = (apt_array == 0) & (apt_flag == 0) & np.isfinite(apt_x) & np.isfinite(apt_y)
        det_indices = np.flatnonzero(det_keep)
        # NetCDF4 orthogonal indexing of thousands of disjoint detector
        # indices can create a very large gather plan.  Read the compact
        # retained matrices contiguously, then select detectors in NumPy.
        weights_all = np.asarray(dataset.variables["weights"][:, :], dtype=float)[
            :, det_indices
        ]
        pointing_residual_max = 0.0
        for scan_row, (start, stop) in enumerate(scans):
            sl = slice(int(start), int(stop) + 1)
            time = time_all[sl]
            az = az_all[sl]
            alt = alt_all[sl]
            elevation = elevation_all[sl]
            vx, vy = scan_velocity(time, az, alt)
            speed = np.hypot(vx, vy)
            sector_for_time = sector_index(np.arctan2(vy, vx))
            signal = np.asarray(dataset.variables["signal"][sl, :], dtype=float)[
                :, det_indices
            ]
            flags = np.asarray(dataset.variables["flags"][sl, :], dtype=float)[
                :, det_indices
            ]
            scan_weights = weights_all[scan_row]
            ct = np.cos(elevation)[:, None]
            st = np.sin(elevation)[:, None]
            lon = (
                az[:, None] * RAD_TO_ARCSEC
                + ct * apt_x[det_indices][None, :]
                - st * apt_y[det_indices][None, :]
                + po_az_all[sl, None]
            )
            lat = (
                alt[:, None] * RAD_TO_ARCSEC
                + ct * apt_y[det_indices][None, :]
                + st * apt_x[det_indices][None, :]
                + po_alt_all[sl, None]
            )
            # Radius is invariant under the elevation rotation; this also
            # catches accidental unit or offset-contract changes.
            observed = np.hypot(
                lon - az[:, None] * RAD_TO_ARCSEC - po_az_all[sl, None],
                lat - alt[:, None] * RAD_TO_ARCSEC - po_alt_all[sl, None],
            )
            expected = np.hypot(apt_x[det_indices], apt_y[det_indices])[None, :]
            pointing_residual_max = max(
                pointing_residual_max, float(np.nanmax(np.abs(observed - expected)))
            )
            col = np.floor((lon + half_width_arcsec) / pixel_arcsec + 0.5).astype(int)
            row = np.floor((lat + half_width_arcsec) / pixel_arcsec + 0.5).astype(int)
            base_good = (
                np.isfinite(signal) & np.isfinite(flags) & (flags == 0.0)
                & np.isfinite(scan_weights)[None, :] & (scan_weights[None, :] > 0.0)
                & (col >= 0) & (col < n_side) & (row >= 0) & (row < n_side)
                & np.isfinite(speed)[:, None] & (speed[:, None] >= minimum_speed)
            )
            for item_index in range(-1, SECTOR_COUNT):
                item = full if item_index < 0 else sectors[item_index]
                good = base_good if item_index < 0 else (
                    base_good & (sector_for_time[:, None] == item_index)
                )
                rr, dd = np.nonzero(good)
                if rr.size == 0:
                    continue
                cc = col[rr, dd]
                yy = row[rr, dd]
                ww = scan_weights[dd]
                ss = signal[rr, dd]
                np.add.at(item.weight, (yy, cc), ww)
                np.add.at(item.weighted_signal, (yy, cc), ww * ss)
                np.add.at(item.hit_count, (yy, cc), 1)
                item.accepted += int(rr.size)
                near = np.hypot(lon[rr, dd], lat[rr, dd]) <= 20.0
                if np.any(near):
                    vw = ww[near]
                    vr = rr[near]
                    item.velocity_weight += float(np.sum(vw))
                    item.velocity_x_sum += float(np.sum(vw * vx[vr]))
                    item.velocity_y_sum += float(np.sum(vw * vy[vr]))
    return [full, *sectors], {
        "selected_detector_count": int(det_indices.size),
        "pointing_radius_contract_max_residual_arcsec": pointing_residual_max,
    }


def signal_image(item: SectorMap) -> np.ndarray:
    result = np.full(item.weight.shape, np.nan)
    good = item.weight > 0.0
    result[good] = item.weighted_signal[good] / item.weight[good]
    return result


def fit_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = [row for row in rows if row["status"] == "success"]
    if len(good) < 6:
        raise ContractError("fewer than six velocity sectors have successful fits")
    x = np.asarray([row["x_arcsec"] for row in good])
    y = np.asarray([row["y_arcsec"] for row in good])
    vx = np.asarray([row["velocity_x_arcsec_s"] for row in good])
    vy = np.asarray([row["velocity_y_arcsec_s"] for row in good])
    sx = np.sign(vx)
    sy = np.sign(vy)
    target = np.ravel(np.column_stack([x, y]))
    designs: dict[str, np.ndarray] = {}
    n = len(good)
    constant = np.zeros((2 * n, 2))
    constant[0::2, 0] = 1.0
    constant[1::2, 1] = 1.0
    designs["constant"] = constant
    time_lag = np.zeros((2 * n, 3))
    time_lag[:, :2] = constant
    time_lag[0::2, 2] = vx
    time_lag[1::2, 2] = vy
    designs["time_lag"] = time_lag
    axis_sign = np.zeros((2 * n, 4))
    axis_sign[:, :2] = constant
    axis_sign[0::2, 2] = sx
    axis_sign[1::2, 3] = sy
    designs["axis_sign"] = axis_sign
    joint = np.zeros((2 * n, 5))
    joint[:, :2] = constant
    joint[0::2, 2] = vx
    joint[1::2, 2] = vy
    joint[0::2, 3] = sx
    joint[1::2, 4] = sy
    designs["joint"] = joint
    results: list[dict[str, Any]] = []
    parameter_names = {
        "constant": ("x0_arcsec", "y0_arcsec"),
        "time_lag": ("x0_arcsec", "y0_arcsec", "tau_sec"),
        "axis_sign": ("x0_arcsec", "y0_arcsec", "h_az_arcsec", "h_el_arcsec"),
        "joint": (
            "x0_arcsec", "y0_arcsec", "tau_sec", "h_az_arcsec", "h_el_arcsec"
        ),
    }
    for name in MODEL_NAMES:
        design = designs[name]
        beta, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ beta
        rss = float(np.sum(residual ** 2))
        count = int(target.size)
        row: dict[str, Any] = {
            "model": name,
            "sector_count": n,
            "coordinate_count": count,
            "parameter_count": int(beta.size),
            "rank": int(rank),
            "condition_number": float(singular[0] / singular[-1]),
            "rms_arcsec": float(np.sqrt(rss / count)),
            "bic": float(count * math.log(max(rss / count, 1.0e-30)) + beta.size * math.log(count)),
        }
        row.update(zip(parameter_names[name], map(float, beta), strict=True))
        if "tau_sec" in row:
            row["tau_ms"] = 1000.0 * row["tau_sec"]
        results.append(row)
    return results


def analyze_pointing(row: dict[str, Any], protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    items, contract = reconstruct_sectors(
        Path(row["ptc_path"]),
        float(protocol["map_pixel_arcsec"]),
        float(protocol["map_half_width_arcsec"]),
        float(protocol["minimum_speed_arcsec_s"]),
    )
    pixel = float(protocol["map_pixel_arcsec"])
    half = float(protocol["map_half_width_arcsec"])
    axis = np.linspace(-half, half, items[0].weight.shape[0])
    full_image = signal_image(items[0])
    full_fit = fit_gaussian_core(
        full_image, axis, axis, float(row["ppt_x_arcsec"]),
        float(row["ppt_y_arcsec"]), float(protocol["fit_half_width_arcsec"]),
    )
    if full_fit["status"] != "success":
        raise ContractError(f"obs {row['pointing_obsnum']}: full-map fit failed")
    sector_rows: list[dict[str, Any]] = []
    for index, item in enumerate(items[1:]):
        image = signal_image(item)
        fit = fit_gaussian_core(
            image, axis, axis, float(full_fit["x_arcsec"]),
            float(full_fit["y_arcsec"]), float(protocol["fit_half_width_arcsec"]),
        )
        entry: dict[str, Any] = {
            "sector": index,
            "center_angle_deg": 360.0 * index / SECTOR_COUNT,
            "accepted_contribution_count": item.accepted,
            "velocity_weight": item.velocity_weight,
            "velocity_x_arcsec_s": (
                item.velocity_x_sum / item.velocity_weight
                if item.velocity_weight > 0.0 else math.nan
            ),
            "velocity_y_arcsec_s": (
                item.velocity_y_sum / item.velocity_weight
                if item.velocity_weight > 0.0 else math.nan
            ),
        }
        entry.update(fit)
        sector_rows.append(entry)
    model_rows = fit_models(sector_rows)
    output.mkdir(parents=True)
    Table(rows=sector_rows).write(output / "sector_centroids.ecsv", format="ascii.ecsv")
    Table(
        rows=[[row.get(name, math.nan) for name in MODEL_FIELDS] for row in model_rows],
        names=MODEL_FIELDS,
    ).write(output / "model_results.ecsv", format="ascii.ecsv")
    images = np.stack([signal_image(item) for item in items])
    np.savez_compressed(output / "sector_maps.npz", images=images, axis_arcsec=axis)
    pdf_name = f"lissajous_velocity_sectors_o{row['pointing_obsnum']}.pdf"
    with PdfPages(output / pdf_name, metadata=PDF_METADATA) as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        for plot_index, (ax, item, image) in enumerate(zip(axes.ravel(), items, images)):
            label = "full" if plot_index == 0 else f"sector {plot_index - 1}"
            ax.imshow(image, origin="lower", extent=[-half, half, -half, half])
            ax.set_title(label)
            ax.set_xlim(full_fit["x_arcsec"] - 25, full_fit["x_arcsec"] + 25)
            ax.set_ylim(full_fit["y_arcsec"] - 25, full_fit["y_arcsec"] + 25)
            if plot_index > 0 and sector_rows[plot_index - 1]["status"] == "success":
                rr = sector_rows[plot_index - 1]
                ax.plot(rr["x_arcsec"], rr["y_arcsec"], "+", color="white", ms=10)
        fig.suptitle(f"Obs {row['pointing_obsnum']} a1100 velocity-sector maps")
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
        ax.scatter(
            [r["velocity_x_arcsec_s"] for r in sector_rows],
            [r["velocity_y_arcsec_s"] for r in sector_rows],
            c=np.arange(SECTOR_COUNT), cmap="hsv", s=80,
        )
        for r in sector_rows:
            ax.annotate(str(r["sector"]), (r["velocity_x_arcsec_s"], r["velocity_y_arcsec_s"]))
        ax.axhline(0, color="0.7")
        ax.axvline(0, color="0.7")
        ax.set_aspect("equal")
        ax.set_xlabel("effective map x velocity (arcsec/s)")
        ax.set_ylabel("effective map y velocity (arcsec/s)")
        pdf.savefig(fig)
        plt.close(fig)
    result = {
        "schema": "sci-align-001-lissajous-pointing-result-v1",
        "input": row,
        "protocol": protocol,
        "pointing_contract": contract,
        "full_fit": full_fit,
        "sector_results": sector_rows,
        "model_results": model_rows,
    }
    write_json(output / "result.json", result)
    write_checksums(output, [
        "model_results.ecsv", pdf_name, "result.json", "sector_centroids.ecsv",
        "sector_maps.npz",
    ])
    return result


def analyze_pointing_worker(
    row: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    messages: Any,
) -> None:
    """Run one observation in an isolated process to bound library RSS."""
    try:
        analyze_pointing(row, protocol, output)
    except BaseException:  # pragma: no cover - returned to parent verbatim
        messages.put({"ok": False, "traceback": traceback.format_exc()})
        return
    messages.put({"ok": True})


def analyze_pointing_isolated(
    row: dict[str, Any], protocol: dict[str, Any], output: Path
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    process = context.Process(
        target=analyze_pointing_worker,
        args=(row, protocol, output, messages),
    )
    process.start()
    process.join()
    try:
        message = messages.get(timeout=2.0)
    except queue.Empty as error:
        raise ContractError(
            f"obs {row['pointing_obsnum']}: isolated worker exited "
            f"with code {process.exitcode} without a result"
        ) from error
    finally:
        messages.close()
        messages.join_thread()
    if not message["ok"]:
        raise ContractError(
            f"obs {row['pointing_obsnum']}: isolated worker failed\n"
            f"{message['traceback']}"
        )
    if process.exitcode != 0:
        raise ContractError(
            f"obs {row['pointing_obsnum']}: isolated worker exit code "
            f"{process.exitcode}"
        )
    process.close()
    return json.loads((output / "result.json").read_text())


def load_frozen_selection(
    selection_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checksums = selection_dir / "SHA256SUMS"
    if not checksums.is_file():
        raise ContractError("selection directory lacks SHA256SUMS")
    for line in checksums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        if sha256_file(selection_dir / name) != expected:
            raise ContractError(f"selection checksum mismatch: {name}")
    selection = json.loads((selection_dir / "selected_pointings.json").read_text())
    protocol = json.loads((selection_dir / "frozen_protocol.json").read_text())
    return selection, protocol


def run(args: argparse.Namespace) -> None:
    selection_dir = args.selection_dir.resolve()
    selection, protocol = load_frozen_selection(selection_dir)
    rows = selection["rows"]
    if args.obsnum:
        wanted = set(args.obsnum)
        rows = [row for row in rows if int(row["pointing_obsnum"]) in wanted]
        if {int(row["pointing_obsnum"]) for row in rows} != wanted:
            raise ContractError("one or more requested obsnums are absent from selection")
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for row in rows:
        obsnum = int(row["pointing_obsnum"])
        if sha256_file(Path(row["ptc_path"])) != row["ptc_sha256"]:
            raise ContractError(f"obs {obsnum}: PTC identity changed")
        if sha256_file(Path(row["ppt_path"])) != row["ppt_sha256"]:
            raise ContractError(f"obs {obsnum}: PPT identity changed")
        analyze_pointing_isolated(row, protocol, output / f"o{obsnum}")
        print(f"analyzed pointing obs={obsnum}")
    aggregate_results(selection_dir, output, rows)
    print(f"analysis complete: pointings={len(rows)} output={output}")


def run_one(args: argparse.Namespace) -> None:
    """Analyze exactly one frozen observation into a shared corpus root.

    Each Slurm array task owns a distinct ``o<obsnum>`` directory.  The
    corpus-level files are deliberately left to the existing ``aggregate``
    command after every array task has completed.
    """
    selection_dir = args.selection_dir.resolve()
    selection, protocol = load_frozen_selection(selection_dir)
    rows = [
        row for row in selection["rows"]
        if int(row["pointing_obsnum"]) == args.obsnum
    ]
    if len(rows) != 1:
        raise ContractError(
            f"selection does not contain exactly one obs {args.obsnum}"
        )
    row = rows[0]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    observation_output = output / f"o{args.obsnum}"
    if observation_output.exists():
        raise ContractError(f"output already exists: {observation_output}")
    if sha256_file(Path(row["ptc_path"])) != row["ptc_sha256"]:
        raise ContractError(f"obs {args.obsnum}: PTC identity changed")
    if sha256_file(Path(row["ppt_path"])) != row["ppt_sha256"]:
        raise ContractError(f"obs {args.obsnum}: PPT identity changed")
    analyze_pointing_isolated(row, protocol, observation_output)
    print(f"analyzed pointing obs={args.obsnum} output={observation_output}")


def aggregate_results(
    selection_dir: Path,
    output: Path,
    rows: list[dict[str, Any]],
) -> None:
    corpus: list[dict[str, Any]] = []
    for row in rows:
        obsnum = int(row["pointing_obsnum"])
        result_path = output / f"o{obsnum}" / "result.json"
        if not result_path.is_file():
            raise ContractError(f"obs {obsnum}: result.json is missing")
        result = json.loads(result_path.read_text())
        if int(result["input"]["pointing_obsnum"]) != obsnum:
            raise ContractError(f"obs {obsnum}: result identity mismatch")
        if result["input"]["ptc_sha256"] != row["ptc_sha256"]:
            raise ContractError(f"obs {obsnum}: result PTC identity mismatch")
        for model in result["model_results"]:
            corpus.append({
                "beammap_obsnum": row["beammap_obsnum"],
                "pointing_obsnum": obsnum,
                "selection_role": row["selection_role"],
                "brightness_stratum": row["brightness_stratum"],
                "snr_a1100": row["snr_a1100"],
                "mean_elevation_deg": row["mean_elevation_deg"],
                **model,
            })
    corpus_fields = (
        "beammap_obsnum", "pointing_obsnum", "selection_role",
        "brightness_stratum", "snr_a1100", "mean_elevation_deg", *MODEL_FIELDS,
    )
    table = Table(
        rows=[[row.get(name, math.nan) for name in corpus_fields] for row in corpus],
        names=corpus_fields,
    )
    table.write(
        output / "corpus_model_results.ecsv", format="ascii.ecsv", overwrite=True
    )
    plot_corpus_summary(table, output / "corpus_model_summary.pdf")
    manifest = {
        "schema": "sci-align-001-lissajous-pointing-corpus-result-v1",
        "selection_manifest_sha256": sha256_file(selection_dir / "selected_pointings.json"),
        "frozen_protocol_sha256": sha256_file(selection_dir / "frozen_protocol.json"),
        "pointing_count": len(rows),
        "pointing_obsnums": [int(row["pointing_obsnum"]) for row in rows],
    }
    write_json(output / "manifest.json", manifest)
    write_checksums(output, [
        "corpus_model_results.ecsv", "corpus_model_summary.pdf", "manifest.json"
    ])


def verify_observation_results(output: Path, rows: list[dict[str, Any]]) -> None:
    """Authenticate every isolated observation before corpus aggregation."""
    for row in rows:
        obsnum = int(row["pointing_obsnum"])
        root = output / f"o{obsnum}"
        checksums = root / "SHA256SUMS"
        if not checksums.is_file():
            raise ContractError(f"obs {obsnum}: SHA256SUMS is missing")
        for line in checksums.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            path = root / name.strip()
            if sha256_file(path) != expected:
                raise ContractError(f"obs {obsnum}: checksum mismatch: {path.name}")


def plot_corpus_summary(table: Table, path: Path) -> None:
    by_model = {
        model: table[np.asarray(table["model"], dtype=str) == model]
        for model in MODEL_NAMES
    }
    lag = by_model["time_lag"]
    sign = by_model["axis_sign"]
    constant = by_model["constant"]
    joint = by_model["joint"]
    pointings = list(np.asarray(constant["pointing_obsnum"], dtype=int))
    with PdfPages(path, metadata=PDF_METADATA) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
        scatter = axes[0, 0].scatter(
            lag["mean_elevation_deg"], lag["tau_ms"], c=lag["snr_a1100"],
            cmap="viridis", edgecolor="black",
        )
        axes[0, 0].axhline(0.0, color="0.7")
        axes[0, 0].set_xlabel("mean telescope elevation (deg)")
        axes[0, 0].set_ylabel("descriptive scalar lag (ms)")
        fig.colorbar(scatter, ax=axes[0, 0], label="a1100 PPT S/N")
        axes[0, 1].scatter(
            sign["mean_elevation_deg"], sign["h_az_arcsec"], label="map x sign term"
        )
        axes[0, 1].scatter(
            sign["mean_elevation_deg"], sign["h_el_arcsec"], label="map y sign term"
        )
        axes[0, 1].axhline(0.0, color="0.7")
        axes[0, 1].set_xlabel("mean telescope elevation (deg)")
        axes[0, 1].set_ylabel("direction-sign half-separation (arcsec)")
        axes[0, 1].legend()
        index = np.arange(len(pointings))
        axes[1, 0].plot(index, np.asarray(lag["bic"] - constant["bic"]), "o-", label="time lag")
        axes[1, 0].plot(index, np.asarray(sign["bic"] - constant["bic"]), "o-", label="axis sign")
        axes[1, 0].plot(index, np.asarray(joint["bic"] - constant["bic"]), "o-", label="joint")
        axes[1, 0].axhline(0.0, color="0.4")
        axes[1, 0].set_xticks(index, [str(value) for value in pointings], rotation=90)
        axes[1, 0].set_ylabel("BIC minus constant model (descriptive)")
        axes[1, 0].legend()
        axes[1, 1].plot(index, constant["rms_arcsec"], "o-", label="constant")
        axes[1, 1].plot(index, lag["rms_arcsec"], "o-", label="time lag")
        axes[1, 1].plot(index, sign["rms_arcsec"], "o-", label="axis sign")
        axes[1, 1].plot(index, joint["rms_arcsec"], "o-", label="joint")
        axes[1, 1].set_xticks(index, [str(value) for value in pointings], rotation=90)
        axes[1, 1].set_ylabel("sector-centroid residual RMS (arcsec)")
        axes[1, 1].legend()
        fig.suptitle(
            "SCI-ALIGN-001 retained Lissajous pointing model comparison\n"
            "descriptive only; sector-centroid covariance unavailable"
        )
        pdf.savefig(fig)
        plt.close(fig)


def aggregate(args: argparse.Namespace) -> None:
    selection_dir = args.selection_dir.resolve()
    selection, _ = load_frozen_selection(selection_dir)
    rows = selection["rows"]
    output = args.output.resolve()
    if args.existing_observation_root:
        source = args.existing_observation_root.resolve()
        if output != source:
            raise ContractError(
                "--output must equal --existing-observation-root"
            )
    elif output.exists():
        raise ContractError(f"output already exists: {output}")
    else:
        output.mkdir(parents=True)
    verify_observation_results(output, rows)
    aggregate_results(selection_dir, output, rows)
    print(f"aggregate complete: pointings={len(rows)} output={output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    inventory_parser = sub.add_parser("inventory")
    inventory_parser.add_argument("--standard-trial-root", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.set_defaults(function=inventory)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--selection-dir", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--obsnum", type=int, action="append")
    run_parser.set_defaults(function=run)
    run_one_parser = sub.add_parser("run-one")
    run_one_parser.add_argument("--selection-dir", type=Path, required=True)
    run_one_parser.add_argument("--output-root", type=Path, required=True)
    run_one_parser.add_argument("--obsnum", type=int, required=True)
    run_one_parser.set_defaults(function=run_one)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--selection-dir", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--existing-observation-root", type=Path)
    aggregate_parser.set_defaults(function=aggregate)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (ContractError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
