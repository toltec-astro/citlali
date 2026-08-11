#!/usr/bin/env python3
"""Fit Beammap timing directly from a retained full-PTC timestream.

This is the Beammap analogue of
``analyze_sci_align_001_lissajous_timestream.py``.  It preserves that tool's
``t + tau`` convention and scan-local interpolation, but applies the
detector-grouped Beammap coordinate contract: telescope pointing plus the
configured pointing offset, with physical detector APT offsets suppressed.

The per-detector source centers and beam shapes are taken only from the
direction-blind standard APT.  Directional APTs are neither accepted nor read.
The tool is diagnostic-only and never modifies an input reduction product.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import analyze_sci_align_001_lissajous_timestream as pointing  # noqa: E402
from render_sci_align_001_split_direction_beammaps import (  # noqa: E402
    select_detectors,
    uid_index,
)


RAD_TO_ARCSEC = pointing.RAD_TO_ARCSEC
FWHM_TO_SIGMA = pointing.FWHM_TO_SIGMA
MODELS = ("constant", "lag", "scan_hysteresis", "joint")


class ContractError(RuntimeError):
    """An input or result violates the frozen diagnostic contract."""


@dataclass(frozen=True)
class DetectorGeometry:
    uid: np.ndarray
    network: np.ndarray
    center_x_arcsec: np.ndarray
    center_y_arcsec: np.ndarray
    major_fwhm_arcsec: np.ndarray
    minor_fwhm_arcsec: np.ndarray
    angle_rad: np.ndarray

    def subset(self, keep: np.ndarray) -> "DetectorGeometry":
        return DetectorGeometry(*(
            np.asarray(value)[keep]
            for value in (
                self.uid,
                self.network,
                self.center_x_arcsec,
                self.center_y_arcsec,
                self.major_fwhm_arcsec,
                self.minor_fwhm_arcsec,
                self.angle_rad,
            )
        ))


@dataclass
class PreparedScan:
    scan_row: int
    output_scan_index: int
    full_time: np.ndarray
    full_az: np.ndarray
    full_alt: np.ndarray
    full_pointing_az: np.ndarray
    full_pointing_alt: np.ndarray
    full_velocity_x: np.ndarray
    full_velocity_y: np.ndarray
    recorded_time: np.ndarray
    geometry: DetectorGeometry
    ptc_weight: np.ndarray
    score_mask: np.ndarray
    residual_signal: np.ndarray
    reference_x: np.ndarray
    reference_y: np.ndarray

    def coordinates(
        self, tau_sec: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = self.recorded_time + tau_sec
        az = pointing.interpolate_unwrapped(query, self.full_time, self.full_az)
        alt = pointing.interpolate_unwrapped(query, self.full_time, self.full_alt)
        pointing_az = pointing.interpolate_linear(
            query, self.full_time, self.full_pointing_az
        )
        pointing_alt = pointing.interpolate_linear(
            query, self.full_time, self.full_pointing_alt
        )
        velocity_x = pointing.interpolate_linear(
            query, self.full_time, self.full_velocity_x
        )
        velocity_y = pointing.interpolate_linear(
            query, self.full_time, self.full_velocity_y
        )
        # Detector-grouped Beammap maps intentionally suppress physical APT
        # offsets.  Do not use det_lon/det_lat or rotate apt_x_t/apt_y_t here.
        x = az * RAD_TO_ARCSEC + pointing_az
        y = alt * RAD_TO_ARCSEC + pointing_alt
        return x, y, velocity_x, velocity_y


@dataclass
class PreparedObservation:
    obsnum: int
    ptc_path: Path
    standard_apt_path: Path
    scans: list[PreparedScan]
    scan_axis_x: float
    scan_axis_y: float
    selected_detector_count: int
    used_detector_count: int
    eligible_networks: tuple[int, ...]
    common_support_sample_count: int
    scored_value_count: int
    protocol: dict[str, Any]


def sha256_file(path: Path) -> str:
    return pointing.sha256_file(path)


def write_json(path: Path, value: Any) -> None:
    pointing.write_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != "sci-align-001-beammap-timestream-protocol-v1":
        raise ContractError("unsupported Beammap timestream protocol schema")
    return document


def require_standard_apt(path: Path, obsnum: int) -> Table:
    if "_left" in path.stem or "_right" in path.stem:
        raise ContractError("directional APTs are prohibited as fit authority")
    table = Table.read(path, format="ascii.ecsv")
    required = {
        "uid", "array", "nw", "flag", "amp", "sig2noise", "x_t_raw",
        "y_t_raw", "a_fwhm", "b_fwhm", "angle",
    }
    missing = sorted(required - set(table.colnames))
    if missing:
        raise ContractError(f"standard APT lacks columns: {missing}")
    realized = int(table.meta.get("obsnum", -1))
    if realized != obsnum:
        raise ContractError(
            f"standard APT obsnum {realized} does not match requested {obsnum}"
        )
    uid_index(table, "standard APT")
    return table


def freeze(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    ptc = args.ptc.resolve()
    apt = args.standard_apt.resolve()
    if not ptc.is_file() or not apt.is_file():
        raise ContractError("full PTC and standard APT must both exist")
    require_standard_apt(apt, args.obsnum)
    with netCDF4.Dataset(ptc) as dataset:
        required = {
            "signal", "flags", "weights", "scan_indices", "raw_scan_indices",
            "output_scan_index", "apt_array", "apt_uid", "apt_nw", "TelTime",
            "az_phys", "alt_phys", "pointing_offset_az", "pointing_offset_alt",
        }
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ContractError(f"full PTC lacks variables: {missing}")
        sample_count = int(dataset.dimensions["n_pts"].size)
        scan_count = int(dataset.dimensions["n_scans"].size)
        detector_count = int(dataset.dimensions["n_dets"].size)
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    document = {
        "schema": "sci-align-001-beammap-timestream-input-v1",
        "observation_number": args.obsnum,
        "array": "a1100",
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "full_ptc": {"path": str(ptc), "sha256": sha256_file(ptc), "size_bytes": ptc.stat().st_size},
            "standard_apt": {"path": str(apt), "sha256": sha256_file(apt), "size_bytes": apt.stat().st_size},
        },
        "input_census": {
            "sample_count": sample_count,
            "scan_count": scan_count,
            "detector_count": detector_count,
        },
        "selection": {
            "maximum_detectors": int(protocol["selection"]["maximum_detectors"]),
            "uses_directional_products": False,
        },
    }
    write_json(output / "frozen_input.json", document)
    pointing.write_checksums(output, ["frozen_input.json"])
    print(f"Beammap timestream input frozen: obs={args.obsnum} output={output}")


def load_frozen(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(path.read_text())
    if document.get("schema") != "sci-align-001-beammap-timestream-input-v1":
        raise ContractError("unsupported frozen Beammap input schema")
    protocol_path = Path(document["protocol"]["path"])
    if sha256_file(protocol_path) != document["protocol"]["sha256"]:
        raise ContractError("frozen protocol identity changed")
    protocol = load_protocol(protocol_path)
    for role in ("full_ptc", "standard_apt"):
        item = document["inputs"][role]
        path_item = Path(item["path"])
        if sha256_file(path_item) != item["sha256"]:
            raise ContractError(f"frozen {role} identity changed")
    if sha256_file(Path(__file__).resolve()) != document["tool"]["sha256"]:
        raise ContractError("diagnostic tool identity changed after freeze")
    return document, protocol


def detector_geometry(
    standard: Table, selected: Sequence[dict[str, Any]]
) -> DetectorGeometry:
    rows = [standard[int(item["standard_row_index"])] for item in selected]
    major = np.asarray([max(float(row["a_fwhm"]), float(row["b_fwhm"])) for row in rows])
    minor = np.asarray([min(float(row["a_fwhm"]), float(row["b_fwhm"])) for row in rows])
    result = DetectorGeometry(
        uid=np.asarray([int(row["uid"]) for row in rows], dtype=np.int64),
        network=np.asarray([int(row["nw"]) for row in rows], dtype=np.int64),
        center_x_arcsec=np.asarray([float(row["x_t_raw"]) for row in rows]),
        center_y_arcsec=np.asarray([float(row["y_t_raw"]) for row in rows]),
        major_fwhm_arcsec=major,
        minor_fwhm_arcsec=minor,
        angle_rad=np.asarray([float(row["angle"]) for row in rows]),
    )
    values = np.concatenate([
        result.center_x_arcsec, result.center_y_arcsec, major, minor,
        result.angle_rad,
    ])
    if np.any(~np.isfinite(values)) or np.any(major <= 0.0) or np.any(minor <= 0.0):
        raise ContractError("selected standard-APT geometry is invalid")
    return result


def scan_axis(displacements: Sequence[tuple[float, float]]) -> tuple[float, float]:
    vectors = np.asarray(displacements, dtype=float)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    keep = np.isfinite(lengths) & (lengths > 0.0)
    if np.count_nonzero(keep) < 2:
        raise ContractError("insufficient scan displacement support")
    unit = vectors[keep] / lengths[keep, None]
    covariance = unit.T @ unit
    values, columns = np.linalg.eigh(covariance)
    axis = columns[:, int(np.argmax(values))]
    if (abs(axis[0]) >= abs(axis[1]) and axis[0] < 0.0) or (
        abs(axis[1]) > abs(axis[0]) and axis[1] < 0.0
    ):
        axis *= -1.0
    return float(axis[0]), float(axis[1])


def prepare(frozen: dict[str, Any], protocol: dict[str, Any]) -> PreparedObservation:
    obsnum = int(frozen["observation_number"])
    ptc_path = Path(frozen["inputs"]["full_ptc"]["path"])
    apt_path = Path(frozen["inputs"]["standard_apt"]["path"])
    standard = require_standard_apt(apt_path, obsnum)
    maximum = int(protocol["selection"]["maximum_detectors"])
    selected = select_detectors(standard, 0, maximum, None)
    selected_geometry = detector_geometry(standard, selected)
    support = protocol["support"]
    max_tau = float(support["maximum_abs_tau_sec"])
    scans_out: list[PreparedScan] = []
    displacements: list[tuple[float, float]] = []
    used_uids: set[int] = set()
    used_networks: set[int] = set()
    support_count = 0
    scored_count = 0

    with netCDF4.Dataset(ptc_path) as dataset:
        ptc_uid = np.asarray(dataset.variables["apt_uid"][:], dtype=np.int64)
        ptc_array = np.asarray(dataset.variables["apt_array"][:], dtype=np.int64)
        ptc_nw = np.asarray(dataset.variables["apt_nw"][:], dtype=np.int64)
        if len(set(map(int, ptc_uid))) != ptc_uid.size:
            raise ContractError("full PTC contains duplicate detector UIDs")
        by_uid = {int(uid): index for index, uid in enumerate(ptc_uid)}
        missing = sorted(set(map(int, selected_geometry.uid)) - set(by_uid))
        if missing:
            raise ContractError(f"full PTC lacks selected standard-APT UIDs: {missing}")
        detector_indices = np.asarray(
            [by_uid[int(uid)] for uid in selected_geometry.uid], dtype=np.int64
        )
        if np.any(ptc_array[detector_indices] != 0):
            raise ContractError("selected detector array identity differs in full PTC")
        if not np.array_equal(ptc_nw[detector_indices], selected_geometry.network):
            raise ContractError("selected detector network identity differs in full PTC")

        bounds = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        raw = np.asarray(dataset.variables["raw_scan_indices"][:], dtype=np.int64)
        if raw.shape != (bounds.shape[0], 4) or not np.array_equal(raw[:, :2], bounds) or not np.array_equal(raw[:, 2:], bounds):
            raise ContractError("full PTC lacks repaired same-timebase scan metadata")
        output_scan = np.asarray(dataset.variables["output_scan_index"][:], dtype=np.int64)
        if output_scan.shape != (bounds.shape[0],) or len(set(map(int, output_scan))) != output_scan.size:
            raise ContractError("full PTC output scan identities are invalid")
        time_all = np.asarray(dataset.variables["TelTime"][:], dtype=float)
        az_all = np.asarray(dataset.variables["az_phys"][:], dtype=float)
        alt_all = np.asarray(dataset.variables["alt_phys"][:], dtype=float)
        paz_all = np.asarray(dataset.variables["pointing_offset_az"][:], dtype=float)
        palt_all = np.asarray(dataset.variables["pointing_offset_alt"][:], dtype=float)
        weights_all = np.asarray(dataset.variables["weights"][:, :], dtype=float)[:, detector_indices]

        for scan_row, (start, stop) in enumerate(bounds):
            slc = slice(int(start), int(stop) + 1)
            full_time = time_all[slc]
            if full_time.size < 3 or np.any(~np.isfinite(full_time)) or np.any(np.diff(full_time) <= 0.0):
                raise ContractError(f"scan {scan_row} has invalid time support")
            full_az = pointing.unwrap_finite(az_all[slc])
            full_alt = pointing.unwrap_finite(alt_all[slc])
            full_paz = paz_all[slc]
            full_palt = palt_all[slc]
            vx, vy = pointing.scan_velocity(full_time, full_az, full_alt)
            displacements.append((
                float((full_az[-1] - full_az[0]) * RAD_TO_ARCSEC),
                float((full_alt[-1] - full_alt[0]) * RAD_TO_ARCSEC),
            ))
            common = (
                (full_time - max_tau >= full_time[0])
                & (full_time + max_tau <= full_time[-1])
            )
            if np.count_nonzero(common) < 10:
                continue
            recorded = full_time[common]
            x_ref = full_az[common] * RAD_TO_ARCSEC + full_paz[common]
            y_ref = full_alt[common] * RAD_TO_ARCSEC + full_palt[common]
            radius = np.hypot(
                x_ref[:, None] - selected_geometry.center_x_arcsec[None, :],
                y_ref[:, None] - selected_geometry.center_y_arcsec[None, :],
            )
            # Read contiguous detector matrices first; disjoint NetCDF gather
            # indexing is prohibitively expensive for this product geometry.
            signal = np.asarray(dataset.variables["signal"][slc, :], dtype=float)[common][:, detector_indices]
            flags = np.asarray(dataset.variables["flags"][slc, :], dtype=float)[common][:, detector_indices]
            weight = weights_all[scan_row]
            valid = (
                np.isfinite(signal) & np.isfinite(flags) & (flags == 0.0)
                & np.isfinite(weight)[None, :] & (weight[None, :] > 0.0)
            )
            score = valid & (radius <= float(support["source_scoring_radius_arcsec"]))
            offsource = valid & (radius >= float(support["baseline_training_min_radius_arcsec"]))
            detector_keep = (
                np.sum(score, axis=0) >= int(support["minimum_scored_samples_per_detector_scan"])
            ) & (
                np.sum(offsource, axis=0) >= int(support["minimum_baseline_samples_per_detector_scan"])
            )
            if not np.any(detector_keep):
                continue
            signal = signal[:, detector_keep]
            valid = valid[:, detector_keep]
            score = score[:, detector_keep]
            offsource = offsource[:, detector_keep]
            geometry = selected_geometry.subset(detector_keep)
            u = pointing.normalized_scan_time(recorded)
            residual, _ = pointing.fit_offsource_baseline(
                signal, valid, offsource, u, "constant"
            )
            scans_out.append(PreparedScan(
                scan_row=scan_row,
                output_scan_index=int(output_scan[scan_row]),
                full_time=full_time,
                full_az=full_az,
                full_alt=full_alt,
                full_pointing_az=full_paz,
                full_pointing_alt=full_palt,
                full_velocity_x=vx,
                full_velocity_y=vy,
                recorded_time=recorded,
                geometry=geometry,
                ptc_weight=weight[detector_keep],
                score_mask=score,
                residual_signal=residual,
                reference_x=x_ref,
                reference_y=y_ref,
            ))
            used_uids.update(map(int, geometry.uid))
            used_networks.update(map(int, geometry.network))
            support_count += recorded.size
            scored_count += int(np.count_nonzero(score))
    if len(scans_out) < int(support["minimum_retained_scan_count"]):
        raise ContractError(f"only {len(scans_out)} scans retain source-crossing support")
    axis_x, axis_y = scan_axis(displacements)
    return PreparedObservation(
        obsnum=obsnum,
        ptc_path=ptc_path,
        standard_apt_path=apt_path,
        scans=scans_out,
        scan_axis_x=axis_x,
        scan_axis_y=axis_y,
        selected_detector_count=len(selected_geometry.uid),
        used_detector_count=len(used_uids),
        eligible_networks=tuple(sorted(used_networks)),
        common_support_sample_count=support_count,
        scored_value_count=scored_count,
        protocol=protocol,
    )


def coordinate_gate(observation: PreparedObservation) -> dict[str, Any]:
    maximum = 0.0
    for scan in observation.scans:
        x, y, _, _ = scan.coordinates(0.0)
        maximum = max(
            maximum,
            float(np.max(np.abs(x - scan.reference_x))),
            float(np.max(np.abs(y - scan.reference_y))),
        )
    if maximum > 1.0e-9:
        raise ContractError(f"zero-lag detector-map coordinate residual is {maximum}")
    return {
        "status": "pass",
        "maximum_absolute_residual_arcsec": maximum,
        "semantics": "telescope tangent pointing plus pointing offsets; detector APT offsets suppressed",
    }


def detector_template(
    x: np.ndarray,
    y: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    geometry: DetectorGeometry,
) -> np.ndarray:
    sigma_major = geometry.major_fwhm_arcsec * FWHM_TO_SIGMA
    sigma_minor = geometry.minor_fwhm_arcsec * FWHM_TO_SIGMA
    ct = np.cos(geometry.angle_rad)[None, :]
    st = np.sin(geometry.angle_rad)[None, :]
    dx = x[:, None] - center_x
    dy = y[:, None] - center_y
    major = ct * dx + st * dy
    minor = -st * dx + ct * dy
    return np.exp(-0.5 * (
        (major / sigma_major[None, :]) ** 2
        + (minor / sigma_minor[None, :]) ** 2
    ))


def parameter_names(model: str) -> tuple[str, ...]:
    if model == "constant":
        return ("delta_x_arcsec", "delta_y_arcsec")
    if model == "lag":
        return ("delta_x_arcsec", "delta_y_arcsec", "tau_ms")
    if model == "scan_hysteresis":
        return ("delta_x_arcsec", "delta_y_arcsec", "h_scan_arcsec")
    if model == "joint":
        return ("delta_x_arcsec", "delta_y_arcsec", "tau_ms", "h_scan_arcsec")
    raise ContractError(f"unsupported model {model}")


def decode(values: np.ndarray, model: str) -> dict[str, float]:
    return dict(zip(parameter_names(model), map(float, values), strict=True))


def scan_objective(
    scan: PreparedScan,
    parameters: dict[str, float],
    model: str,
    axis_x: float,
    axis_y: float,
) -> tuple[float, float, int]:
    tau_sec = 0.001 * parameters.get("tau_ms", 0.0)
    x, y, vx, vy = scan.coordinates(tau_sec)
    center_x = scan.geometry.center_x_arcsec[None, :] + parameters["delta_x_arcsec"]
    center_y = scan.geometry.center_y_arcsec[None, :] + parameters["delta_y_arcsec"]
    if model in {"scan_hysteresis", "joint"}:
        sign = np.sign(vx * axis_x + vy * axis_y)[:, None]
        center_x = center_x + parameters["h_scan_arcsec"] * sign * axis_x
        center_y = center_y + parameters["h_scan_arcsec"] * sign * axis_y
    template = detector_template(x, y, center_x, center_y, scan.geometry)
    mask = scan.score_mask
    bs = np.sum(np.where(mask, template * scan.residual_signal, 0.0), axis=0)
    bb = np.sum(np.where(mask, template * template, 0.0), axis=0)
    amplitude = np.maximum(np.divide(bs, bb, out=np.zeros_like(bs), where=bb > 1.0e-16), 0.0)
    residual = scan.residual_signal - template * amplitude[None, :]
    detector_sse = np.sum(np.where(mask, residual * residual, 0.0), axis=0)
    count = np.sum(mask, axis=0)
    return (
        float(np.sum(scan.ptc_weight * detector_sse)),
        float(np.sum(scan.ptc_weight * count)),
        int(np.sum(count)),
    )


def objective(values: np.ndarray, observation: PreparedObservation, model: str) -> float:
    parameters = decode(values, model)
    sse = 0.0
    weight = 0.0
    for scan in observation.scans:
        item_sse, item_weight, _ = scan_objective(
            scan, parameters, model,
            observation.scan_axis_x, observation.scan_axis_y,
        )
        sse += item_sse
        weight += item_weight
    if not math.isfinite(sse) or weight <= 0.0:
        return math.inf
    return sse / weight


def bounds_starts(
    observation: PreparedObservation, model: str
) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    models = observation.protocol["models"]
    center = float(models["center_correction_bound_arcsec"])
    bounds: list[tuple[float, float]] = [(-center, center), (-center, center)]
    base = [0.0, 0.0]
    tau_starts: Sequence[float] = (0.0,)
    h_starts: Sequence[float] = (0.0,)
    if model in {"lag", "joint"}:
        tau_bounds = tuple(map(float, models["lag_search_bounds_ms"]))
        bounds.append(tau_bounds)
        base.append(0.0)
        tau_starts = (-25.0, 0.0, 25.0)
    if model in {"scan_hysteresis", "joint"}:
        h_bounds = tuple(map(float, models["scan_hysteresis_bounds_arcsec"]))
        bounds.append(h_bounds)
        base.append(0.0)
        h_starts = (-2.0, 0.0, 2.0)
    starts = []
    for tau in tau_starts:
        for h_value in h_starts:
            value = np.asarray(base, dtype=float)
            if model == "lag":
                value[2] = tau
            elif model == "scan_hysteresis":
                value[2] = h_value
            elif model == "joint":
                value[2:] = (tau, h_value)
            starts.append(value)
    return bounds, starts


def fit_model(observation: PreparedObservation, model: str) -> dict[str, Any]:
    bounds, starts = bounds_starts(observation, model)
    results = [
        minimize(
            objective, start, args=(observation, model), method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1.0e-12, "gtol": 1.0e-8, "eps": 1.0e-4},
        )
        for start in starts
    ]
    finite = [item for item in results if math.isfinite(float(item.fun))]
    if not finite:
        return {"model": model, "status": "fit_failed"}
    converged = [item for item in finite if bool(item.success)]
    best = min(converged or finite, key=lambda item: float(item.fun))
    parameters = decode(np.asarray(best.x), model)
    per_scan = []
    for scan in observation.scans:
        sse, weight, count = scan_objective(
            scan, parameters, model,
            observation.scan_axis_x, observation.scan_axis_y,
        )
        per_scan.append({
            "scan_row": scan.scan_row,
            "output_scan_index": scan.output_scan_index,
            "weighted_mse": sse / weight if weight > 0.0 else math.nan,
            "scored_sample_count": count,
        })
    tau_ms = parameters.get("tau_ms", 0.0)
    lag_bounds = observation.protocol["models"]["lag_search_bounds_ms"]
    boundary = model in {"lag", "joint"} and (
        tau_ms <= float(lag_bounds[0]) + 1.0
        or tau_ms >= float(lag_bounds[1]) - 1.0
    )
    return {
        "model": model,
        "status": "boundary_failure" if boundary else "success",
        "objective": float(best.fun),
        "parameters": parameters,
        "tau_ms": float(tau_ms),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimizer_iterations": int(best.nit),
        "optimizer_function_evaluations": int(getattr(best, "nfev", -1)),
        "multistart_count": len(starts),
        "per_scan": per_scan,
    }


def profile_tau(
    observation: PreparedObservation, lag_fit: dict[str, Any]
) -> list[dict[str, float | bool]]:
    spec = observation.protocol["models"]["objective_profile_tau_grid_ms"]
    grid = np.linspace(float(spec["minimum"]), float(spec["maximum"]), int(spec["count"]))
    center_bound = float(observation.protocol["models"]["center_correction_bound_arcsec"])
    initial = np.asarray([
        lag_fit["parameters"]["delta_x_arcsec"],
        lag_fit["parameters"]["delta_y_arcsec"],
    ])
    rows = []
    for tau in grid:
        result = minimize(
            lambda xy: objective(np.asarray([xy[0], xy[1], tau]), observation, "lag"),
            initial,
            method="L-BFGS-B",
            bounds=[(-center_bound, center_bound)] * 2,
            options={"maxiter": 200, "ftol": 1.0e-12, "gtol": 1.0e-8, "eps": 1.0e-4},
        )
        rows.append({
            "tau_ms": float(tau),
            "objective": float(result.fun),
            "delta_x_arcsec": float(result.x[0]),
            "delta_y_arcsec": float(result.x[1]),
            "optimizer_success": bool(result.success),
        })
    return rows


def render_pdf(
    path: Path,
    observation: PreparedObservation,
    fits: list[dict[str, Any]],
    profile: list[dict[str, Any]],
) -> None:
    with PdfPages(path, metadata={
        "Title": f"SCI-ALIGN-001 Beammap direct timestream fit: Obs {observation.obsnum}",
        "Subject": "Direct full-PTC lag versus scan-axis hysteresis diagnostic",
    }) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
        tau = np.asarray([row["tau_ms"] for row in profile])
        value = np.asarray([row["objective"] for row in profile])
        axes[0].plot(tau, value - np.nanmin(value), marker="o", markersize=3)
        axes[0].axvline(fits[1]["tau_ms"], color="tab:red", linestyle="--", label="lag optimum")
        axes[0].set_xlabel("tau in complete coordinate at t+tau (ms)")
        axes[0].set_ylabel("profile objective - minimum")
        axes[0].set_title("Lag objective profile")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        names = [fit["model"] for fit in fits]
        objectives = np.asarray([fit.get("objective", np.nan) for fit in fits])
        axes[1].bar(names, objectives - np.nanmin(objectives))
        axes[1].set_ylabel("objective - best objective")
        axes[1].set_title("Point-estimate model comparison")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(axis="y", alpha=0.25)
        figure.suptitle(
            f"Obs {observation.obsnum}: {observation.used_detector_count} detectors, "
            f"{len(observation.scans)} scans\n"
            f"scan axis=({observation.scan_axis_x:+.4f}, {observation.scan_axis_y:+.4f})",
            fontsize=13,
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    frozen, protocol = load_frozen(args.frozen_input.resolve())
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    observation = prepare(frozen, protocol)
    gate = coordinate_gate(observation)
    fits = [fit_model(observation, model) for model in MODELS]
    if any(fit["status"] != "success" for fit in fits):
        raise ContractError(f"one or more model fits failed: {[fit['status'] for fit in fits]}")
    profile = profile_tau(observation, fits[1])
    selected_uids = sorted({int(uid) for scan in observation.scans for uid in scan.geometry.uid})
    selected_rows = []
    for uid in selected_uids:
        networks = {int(scan.geometry.network[np.flatnonzero(scan.geometry.uid == uid)[0]]) for scan in observation.scans if np.any(scan.geometry.uid == uid)}
        selected_rows.append({"uid": uid, "nw": networks.pop()})
    Table(rows=selected_rows).write(output / "selected_detectors.ecsv", format="ascii.ecsv")
    Table(rows=[{
        "model": fit["model"],
        "status": fit["status"],
        "objective": fit["objective"],
        "tau_ms": fit["tau_ms"],
        **fit["parameters"],
    } for fit in fits]).write(output / "model_results.ecsv", format="ascii.ecsv")
    Table(rows=profile).write(output / "objective_profile.ecsv", format="ascii.ecsv")
    render_pdf(output / f"beammap_timestream_fit_o{observation.obsnum}.pdf", observation, fits, profile)
    manifest = {
        "schema": "sci-align-001-beammap-timestream-result-v1",
        "observation_number": observation.obsnum,
        "frozen_input": {"path": str(args.frozen_input.resolve()), "sha256": sha256_file(args.frozen_input)},
        "protocol_sha256": frozen["protocol"]["sha256"],
        "coordinate_gate": gate,
        "coordinate_contract": "detector-grouped Beammap telescope-plus-pointing coordinates at t+tau; physical APT offsets suppressed",
        "center_and_shape_authority": "direction-blind standard APT only; per-detector x_t_raw/y_t_raw/a_fwhm/b_fwhm/angle",
        "support": {
            "selected_detector_count": observation.selected_detector_count,
            "used_detector_count": observation.used_detector_count,
            "eligible_networks": observation.eligible_networks,
            "retained_scan_count": len(observation.scans),
            "common_support_sample_count": observation.common_support_sample_count,
            "scored_value_count": observation.scored_value_count,
        },
        "scan_axis": {"x": observation.scan_axis_x, "y": observation.scan_axis_y},
        "model_results": fits,
        "uncertainty_status": "not estimated in baseline-validation stage",
        "interpretation_boundary": "tests registration within delivered full-PTC signal and telescope-plus-pointing coordinates; does not identify upstream cause or prescribe a correction",
        "outputs": [
            f"beammap_timestream_fit_o{observation.obsnum}.pdf",
            "model_results.ecsv", "objective_profile.ecsv", "selected_detectors.ecsv",
        ],
    }
    write_json(output / "result.json", manifest)
    pointing.write_checksums(output, manifest["outputs"] + ["result.json"])
    print(
        f"Beammap timestream fit complete: obs={observation.obsnum} "
        f"lag_ms={fits[1]['tau_ms']:+.6f} "
        f"h_scan_arcsec={fits[2]['parameters']['h_scan_arcsec']:+.6f} "
        f"output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--ptc", type=Path, required=True)
    freeze_parser.add_argument("--standard-apt", type=Path, required=True)
    freeze_parser.add_argument("--obsnum", type=int, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--frozen-input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "freeze":
            freeze(args)
        elif args.command == "run":
            run(args)
        else:  # pragma: no cover
            raise ContractError(f"unsupported command {args.command}")
    except (ContractError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
