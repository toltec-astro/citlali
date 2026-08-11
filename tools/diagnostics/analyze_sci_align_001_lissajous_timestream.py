#!/usr/bin/env python3
"""Direct PTC-timestream timing diagnostic for SCI-ALIGN-001.

The primary estimator evaluates the complete detector coordinate at ``t+tau``
inside each retained scan and profiles detector-scan amplitudes and baselines.
It is deliberately separate from production Citlali and never modifies an
input reduction product.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
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
from scipy.optimize import minimize  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.stats import gaussian_kde, theilslopes  # noqa: E402

import analyze_sci_align_001_lissajous_pointing as map_space  # noqa: E402


RAD_TO_ARCSEC = 206264.80624709636
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
MODEL_NAMES = ("constant", "lag", "hysteresis", "joint")
BASELINE_NAMES = ("constant", "linear")


class ContractError(RuntimeError):
    """An input or result violates the frozen diagnostic contract."""


def map_centroid_tau_to_coordinate_shift_ms(tau_ms: float) -> float:
    """Convert map centroid-versus-velocity slope to t+tau convention."""
    return -float(tau_ms)


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


def verify_sha256s(root: Path) -> None:
    path = root / "SHA256SUMS"
    if not path.is_file():
        raise ContractError(f"checksum manifest is missing: {path}")
    for line in path.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        actual = sha256_file(root / name)
        if actual != expected:
            raise ContractError(
                f"checksum mismatch for {root / name}: {actual} != {expected}"
            )


def unwrap_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ContractError("angular interpolation input is not a finite vector")
    return np.unwrap(values)


def interpolate_unwrapped(
    query_time: np.ndarray,
    sample_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a scan-local continuous angular representation."""
    if np.any(query_time < sample_time[0]) or np.any(query_time > sample_time[-1]):
        raise ContractError("interpolation query escapes its scan")
    return np.interp(query_time, sample_time, unwrap_finite(values))


def interpolate_linear(
    query_time: np.ndarray,
    sample_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != sample_time.shape or np.any(~np.isfinite(values)):
        raise ContractError("linear interpolation input geometry is invalid")
    if np.any(query_time < sample_time[0]) or np.any(query_time > sample_time[-1]):
        raise ContractError("interpolation query escapes its scan")
    return np.interp(query_time, sample_time, values)


def scan_velocity(
    sample_time: np.ndarray,
    az_unwrapped: np.ndarray,
    alt_unwrapped: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_time.size < 3 or np.any(np.diff(sample_time) <= 0.0):
        raise ContractError("scan time is not strictly increasing")
    return (
        np.gradient(az_unwrapped * RAD_TO_ARCSEC, sample_time),
        np.gradient(alt_unwrapped * RAD_TO_ARCSEC, sample_time),
    )


def fit_offsource_baseline(
    signal: np.ndarray,
    valid: np.ndarray,
    offsource: np.ndarray,
    normalized_time: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit detector baselines only on fixed off-source samples."""
    if mode not in BASELINE_NAMES:
        raise ContractError(f"unsupported baseline mode: {mode}")
    mask = valid & offsource
    count = np.sum(mask, axis=0).astype(float)
    sy = np.sum(np.where(mask, signal, 0.0), axis=0)
    if mode == "constant":
        intercept = np.divide(
            sy, count, out=np.full_like(sy, np.nan), where=count > 0.0
        )
        baseline = np.broadcast_to(intercept[None, :], signal.shape)
        return signal - baseline, np.column_stack([intercept])

    u = normalized_time[:, None]
    su = np.sum(np.where(mask, u, 0.0), axis=0)
    suu = np.sum(np.where(mask, u * u, 0.0), axis=0)
    suy = np.sum(np.where(mask, u * signal, 0.0), axis=0)
    determinant = count * suu - su * su
    intercept = np.divide(
        sy * suu - su * suy,
        determinant,
        out=np.full_like(sy, np.nan),
        where=determinant > 0.0,
    )
    slope = np.divide(
        count * suy - su * sy,
        determinant,
        out=np.full_like(sy, np.nan),
        where=determinant > 0.0,
    )
    baseline = intercept[None, :] + u * slope[None, :]
    return signal - baseline, np.column_stack([intercept, slope])


@dataclass(frozen=True)
class BeamGeometry:
    major_fwhm_arcsec: float
    minor_fwhm_arcsec: float
    angle_rad: float


@dataclass
class PreparedScan:
    scan_row: int
    output_scan_index: int
    full_time: np.ndarray
    full_az: np.ndarray
    full_alt: np.ndarray
    full_elevation: np.ndarray
    full_pointing_az: np.ndarray
    full_pointing_alt: np.ndarray
    full_velocity_x: np.ndarray
    full_velocity_y: np.ndarray
    recorded_time: np.ndarray
    apt_x: np.ndarray
    apt_y: np.ndarray
    detector_uid: np.ndarray
    detector_network: np.ndarray
    ptc_weight: np.ndarray
    valid: np.ndarray
    score_mask: np.ndarray
    offsource_mask: np.ndarray
    residual_by_baseline: dict[str, np.ndarray]
    baseline_coefficients: dict[str, np.ndarray]
    reference_x: np.ndarray
    reference_y: np.ndarray

    def coordinates(
        self, tau_sec: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = self.recorded_time + tau_sec
        az = interpolate_unwrapped(query, self.full_time, self.full_az)
        alt = interpolate_unwrapped(query, self.full_time, self.full_alt)
        elevation = interpolate_unwrapped(
            query, self.full_time, self.full_elevation
        )
        pointing_az = interpolate_linear(
            query, self.full_time, self.full_pointing_az
        )
        pointing_alt = interpolate_linear(
            query, self.full_time, self.full_pointing_alt
        )
        velocity_x = interpolate_linear(
            query, self.full_time, self.full_velocity_x
        )
        velocity_y = interpolate_linear(
            query, self.full_time, self.full_velocity_y
        )
        ct = np.cos(elevation)[:, None]
        st = np.sin(elevation)[:, None]
        x = (
            az[:, None] * RAD_TO_ARCSEC
            + pointing_az[:, None]
            + ct * self.apt_x[None, :]
            - st * self.apt_y[None, :]
        )
        y = (
            alt[:, None] * RAD_TO_ARCSEC
            + pointing_alt[:, None]
            + ct * self.apt_y[None, :]
            + st * self.apt_x[None, :]
        )
        return x, y, velocity_x, velocity_y


@dataclass
class PreparedObservation:
    obsnum: int
    ptc_path: Path
    ppt_path: Path
    ppt_x_arcsec: float
    ppt_y_arcsec: float
    beam: BeamGeometry
    scans: list[PreparedScan]
    eligible_uid_count: int
    eligible_networks: tuple[int, ...]
    common_support_sample_count: int
    scored_value_count: int
    protocol: dict[str, Any]


def coordinate_reconstruction_gate(
    observation: PreparedObservation,
) -> dict[str, float | int | str]:
    maximum = 0.0
    for scan in observation.scans:
        x, y, _, _ = scan.coordinates(0.0)
        if x.shape != scan.reference_x.shape or y.shape != scan.reference_y.shape:
            raise ContractError("zero-lag coordinate geometry changed")
        maximum = max(
            maximum,
            float(np.max(np.abs(x - scan.reference_x))),
            float(np.max(np.abs(y - scan.reference_y))),
        )
    if maximum > 1.0e-9:
        raise ContractError(
            f"zero-lag coordinate reconstruction residual is {maximum} arcsec"
        )
    return {
        "status": "pass",
        "scan_count": len(observation.scans),
        "maximum_absolute_coordinate_residual_arcsec": maximum,
    }


def gaussian_beam(
    x: np.ndarray,
    y: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    beam: BeamGeometry,
) -> np.ndarray:
    sigma_major = beam.major_fwhm_arcsec * FWHM_TO_SIGMA
    sigma_minor = beam.minor_fwhm_arcsec * FWHM_TO_SIGMA
    if sigma_major <= 0.0 or sigma_minor <= 0.0:
        raise ContractError("beam widths must be positive")
    ct = math.cos(beam.angle_rad)
    st = math.sin(beam.angle_rad)
    dx = x - center_x
    dy = y - center_y
    major = ct * dx + st * dy
    minor = -st * dx + ct * dy
    return np.exp(-0.5 * (
        (major / sigma_major) ** 2 + (minor / sigma_minor) ** 2
    ))


def ppt_a1100(path: Path) -> dict[str, float]:
    table = Table.read(path)
    rows = table[np.asarray(table["array"], dtype=int) == 0]
    if len(rows) != 1:
        raise ContractError(f"{path}: expected exactly one a1100 PPT row")
    row = rows[0]
    required = ("x_t", "y_t", "a_fwhm", "b_fwhm", "angle")
    result = {name: float(row[name]) for name in required}
    if any(not math.isfinite(value) for value in result.values()):
        raise ContractError(f"{path}: a1100 PPT geometry is non-finite")
    return result


def load_protocol(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "sci-align-001-lissajous-timestream-protocol-v1":
        raise ContractError("unsupported frozen protocol schema")
    return doc


def load_selection(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ContractError("frozen selection manifest identity changed")
    doc = json.loads(path.read_text())
    if doc.get("schema") != "sci-align-001-lissajous-pointing-selection-v1":
        raise ContractError("unsupported selection manifest schema")
    return doc


def normalized_scan_time(time: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (time[0] + time[-1])
    half_span = 0.5 * (time[-1] - time[0])
    if half_span <= 0.0:
        raise ContractError("scan has nonpositive time span")
    return (time - midpoint) / half_span


def prepare_observation(
    row: dict[str, Any],
    protocol: dict[str, Any],
    baseline_modes: Sequence[str] = BASELINE_NAMES,
) -> PreparedObservation:
    ptc_path = Path(row["ptc_path"])
    ppt_path = Path(row["ppt_path"])
    if sha256_file(ptc_path) != row["ptc_sha256"]:
        raise ContractError(f"obs {row['pointing_obsnum']}: PTC identity changed")
    if sha256_file(ppt_path) != row["ppt_sha256"]:
        raise ContractError(f"obs {row['pointing_obsnum']}: PPT identity changed")
    ppt = ppt_a1100(ppt_path)
    support = protocol["common_support"]
    eligibility = protocol["eligibility"]
    max_tau = float(support["maximum_abs_tau_sec"])
    prepared: list[PreparedScan] = []
    all_uids: set[int] = set()
    all_networks: set[int] = set()
    support_samples = 0
    scored_values = 0

    with netCDF4.Dataset(ptc_path) as dataset:
        required = {
            "signal", "flags", "weights", "scan_indices", "output_scan_index",
            "apt_array", "apt_flag", "apt_uid", "apt_nw", "apt_x_t", "apt_y_t",
            "TelTime", "TelElAct", "az_phys", "alt_phys",
            "pointing_offset_az", "pointing_offset_alt",
        }
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ContractError(f"{ptc_path}: missing variables {missing}")
        time_all = np.asarray(dataset.variables["TelTime"][:], dtype=float)
        az_all = np.asarray(dataset.variables["az_phys"][:], dtype=float)
        alt_all = np.asarray(dataset.variables["alt_phys"][:], dtype=float)
        elevation_all = np.asarray(dataset.variables["TelElAct"][:], dtype=float)
        pointing_az_all = np.asarray(
            dataset.variables["pointing_offset_az"][:], dtype=float
        )
        pointing_alt_all = np.asarray(
            dataset.variables["pointing_offset_alt"][:], dtype=float
        )
        scans = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        output_scan = np.asarray(
            dataset.variables["output_scan_index"][:], dtype=np.int64
        )
        apt_array = np.asarray(dataset.variables["apt_array"][:], dtype=int)
        apt_flag = np.asarray(dataset.variables["apt_flag"][:], dtype=int)
        apt_uid = np.asarray(dataset.variables["apt_uid"][:], dtype=int)
        apt_nw = np.asarray(dataset.variables["apt_nw"][:], dtype=int)
        apt_x = np.asarray(dataset.variables["apt_x_t"][:], dtype=float)
        apt_y = np.asarray(dataset.variables["apt_y_t"][:], dtype=float)
        detector_base = (
            (apt_array == int(eligibility["array_id"]))
            & (apt_flag == int(eligibility["apt_flag_required"]))
            & np.isfinite(apt_x) & np.isfinite(apt_y)
        )
        detector_indices = np.flatnonzero(detector_base)
        weights_all = np.asarray(dataset.variables["weights"][:, :], dtype=float)

        for scan_row, (start, stop) in enumerate(scans):
            full_slice = slice(int(start), int(stop) + 1)
            full_time = time_all[full_slice]
            if full_time.size < 3 or np.any(np.diff(full_time) <= 0.0):
                raise ContractError(f"obs {row['pointing_obsnum']}: invalid scan time")
            common = (
                (full_time - max_tau >= full_time[0])
                & (full_time + max_tau <= full_time[-1])
            )
            if np.count_nonzero(common) < 10:
                raise ContractError("common support leaves too few scan samples")
            full_az = unwrap_finite(az_all[full_slice])
            full_alt = unwrap_finite(alt_all[full_slice])
            full_elevation = unwrap_finite(elevation_all[full_slice])
            full_paz = pointing_az_all[full_slice]
            full_palt = pointing_alt_all[full_slice]
            full_vx, full_vy = scan_velocity(full_time, full_az, full_alt)
            recorded_time = full_time[common]
            elevation = full_elevation[common]
            ct = np.cos(elevation)[:, None]
            st = np.sin(elevation)[:, None]
            x_ref = (
                full_az[common, None] * RAD_TO_ARCSEC
                + full_paz[common, None]
                + ct * apt_x[detector_indices][None, :]
                - st * apt_y[detector_indices][None, :]
            )
            y_ref = (
                full_alt[common, None] * RAD_TO_ARCSEC
                + full_palt[common, None]
                + ct * apt_y[detector_indices][None, :]
                + st * apt_x[detector_indices][None, :]
            )
            radius = np.hypot(
                x_ref - ppt["x_t"], y_ref - ppt["y_t"]
            )
            signal = np.asarray(
                dataset.variables["signal"][full_slice, :], dtype=float
            )[common][:, detector_indices]
            flags = np.asarray(
                dataset.variables["flags"][full_slice, :], dtype=float
            )[common][:, detector_indices]
            weight = weights_all[scan_row, detector_indices]
            valid = (
                np.isfinite(signal) & np.isfinite(flags)
                & (flags == float(eligibility["sample_flag_required"]))
                & np.isfinite(weight)[None, :] & (weight[None, :] > 0.0)
            )
            score_mask = valid & (
                radius <= float(eligibility["source_scoring_radius_arcsec"])
            )
            offsource_mask = valid & (
                radius >= float(eligibility["baseline_training_min_radius_arcsec"])
            )
            detector_keep = (
                np.sum(score_mask, axis=0)
                >= int(eligibility["minimum_scored_samples_per_detector_scan"])
            ) & (
                np.sum(offsource_mask, axis=0)
                >= int(eligibility["minimum_baseline_samples_per_detector_scan"])
            )
            if not np.any(detector_keep):
                raise ContractError(
                    f"obs {row['pointing_obsnum']} scan {scan_row}: no eligible detectors"
                )
            signal = signal[:, detector_keep]
            valid = valid[:, detector_keep]
            score_mask = score_mask[:, detector_keep]
            offsource_mask = offsource_mask[:, detector_keep]
            x_ref = x_ref[:, detector_keep]
            y_ref = y_ref[:, detector_keep]
            selected = detector_indices[detector_keep]
            residuals: dict[str, np.ndarray] = {}
            coefficients: dict[str, np.ndarray] = {}
            u = normalized_scan_time(recorded_time)
            for mode in baseline_modes:
                residual, coefficient = fit_offsource_baseline(
                    signal, valid, offsource_mask, u, mode
                )
                residuals[mode] = residual
                coefficients[mode] = coefficient
            prepared.append(PreparedScan(
                scan_row=scan_row,
                output_scan_index=int(output_scan[scan_row]),
                full_time=full_time,
                full_az=full_az,
                full_alt=full_alt,
                full_elevation=full_elevation,
                full_pointing_az=full_paz,
                full_pointing_alt=full_palt,
                full_velocity_x=full_vx,
                full_velocity_y=full_vy,
                recorded_time=recorded_time,
                apt_x=apt_x[selected],
                apt_y=apt_y[selected],
                detector_uid=apt_uid[selected],
                detector_network=apt_nw[selected],
                ptc_weight=weight[detector_keep],
                valid=valid,
                score_mask=score_mask,
                offsource_mask=offsource_mask,
                residual_by_baseline=residuals,
                baseline_coefficients=coefficients,
                reference_x=x_ref,
                reference_y=y_ref,
            ))
            all_uids.update(map(int, apt_uid[selected]))
            all_networks.update(map(int, apt_nw[selected]))
            support_samples += int(recorded_time.size)
            scored_values += int(np.count_nonzero(score_mask))

    angle = float(ppt["angle"])
    # PPT angle is stored in radians by the active pointing product contract.
    beam = BeamGeometry(
        major_fwhm_arcsec=max(ppt["a_fwhm"], ppt["b_fwhm"]),
        minor_fwhm_arcsec=min(ppt["a_fwhm"], ppt["b_fwhm"]),
        angle_rad=angle,
    )
    return PreparedObservation(
        obsnum=int(row["pointing_obsnum"]),
        ptc_path=ptc_path,
        ppt_path=ppt_path,
        ppt_x_arcsec=ppt["x_t"],
        ppt_y_arcsec=ppt["y_t"],
        beam=beam,
        scans=prepared,
        eligible_uid_count=len(all_uids),
        eligible_networks=tuple(sorted(all_networks)),
        common_support_sample_count=support_samples,
        scored_value_count=scored_values,
        protocol=protocol,
    )


def model_parameter_names(model: str, beam_mode: str) -> tuple[str, ...]:
    if model == "constant":
        names = ("x0_arcsec", "y0_arcsec")
    elif model == "lag":
        names = ("x0_arcsec", "y0_arcsec", "tau_sec")
    elif model == "hysteresis":
        names = ("x0_arcsec", "y0_arcsec", "h_az_arcsec", "h_el_arcsec")
    elif model == "joint":
        names = (
            "x0_arcsec", "y0_arcsec", "tau_sec", "h_az_arcsec", "h_el_arcsec"
        )
    else:
        raise ContractError(f"unsupported model: {model}")
    if beam_mode == "free":
        names += ("major_fwhm_arcsec", "minor_fwhm_arcsec", "beam_angle_rad")
    elif beam_mode != "fixed":
        raise ContractError(f"unsupported beam mode: {beam_mode}")
    return names


def parameter_dict(
    values: np.ndarray, model: str, beam_mode: str
) -> dict[str, float]:
    names = model_parameter_names(model, beam_mode)
    if len(values) != len(names):
        raise ContractError("model parameter geometry mismatch")
    parameters = dict(zip(names, map(float, values), strict=True))
    # Optimize tau in milliseconds so its numerical scale is comparable to
    # the arcsecond position coordinates.  The forward model and public
    # result remain in seconds.
    if "tau_sec" in parameters:
        parameters["tau_sec"] /= 1000.0
    return parameters


def parameter_bounds_and_starts(
    observation: PreparedObservation,
    model: str,
    beam_mode: str,
) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    protocol = observation.protocol
    center_bound = float(
        protocol["source_model"]["source_center_bounds_relative_to_ppt_arcsec"]
    )
    bounds: list[tuple[float, float]] = [
        (observation.ppt_x_arcsec - center_bound, observation.ppt_x_arcsec + center_bound),
        (observation.ppt_y_arcsec - center_bound, observation.ppt_y_arcsec + center_bound),
    ]
    base = [observation.ppt_x_arcsec, observation.ppt_y_arcsec]
    tau_positions: list[int] = []
    if model in {"lag", "joint"}:
        lag_bounds = protocol["models"]["lag_search_bounds_ms"]
        bounds.append((float(lag_bounds[0]), float(lag_bounds[1])))
        base.append(0.0)
        tau_positions.append(len(base) - 1)
    if model in {"hysteresis", "joint"}:
        h_bounds = protocol["models"]["hysteresis_half_separation_bounds_arcsec"]
        bounds.extend([(float(h_bounds[0]), float(h_bounds[1]))] * 2)
        base.extend([0.0, 0.0])
    if beam_mode == "free":
        fwhm_bounds = protocol["source_model"]["free_beam_fwhm_bounds_arcsec"]
        bounds.extend([
            (float(fwhm_bounds[0]), float(fwhm_bounds[1])),
            (float(fwhm_bounds[0]), float(fwhm_bounds[1])),
            (-0.5 * math.pi, 0.5 * math.pi),
        ])
        base.extend([
            observation.beam.major_fwhm_arcsec,
            observation.beam.minor_fwhm_arcsec,
            observation.beam.angle_rad,
        ])
    tau_starts = (-25.0, 0.0, 25.0) if tau_positions else (0.0,)
    sign_starts = (
        ((0.0, 0.0), (-1.0, -1.0), (-1.0, 1.0),
         (1.0, -1.0), (1.0, 1.0))
        if model in {"hysteresis", "joint"} else ((0.0, 0.0),)
    )
    starts = []
    for tau_ms in tau_starts:
        for h_az, h_el in sign_starts:
            value = np.asarray(base, dtype=float)
            if tau_positions:
                value[tau_positions[0]] = tau_ms
            if model == "hysteresis":
                value[2:4] = (h_az, h_el)
            elif model == "joint":
                value[3:5] = (h_az, h_el)
            starts.append(value)
    return bounds, starts


def optimizer_finite_difference_steps(
    model: str, beam_mode: str
) -> np.ndarray:
    """Return explicit steps in each optimizer coordinate's native unit."""
    steps = []
    for name in model_parameter_names(model, beam_mode):
        if name == "tau_sec":
            # tau's optimizer coordinate is milliseconds: 0.01 ms = 10 us.
            steps.append(0.01)
        elif name == "beam_angle_rad":
            steps.append(1.0e-5)
        else:
            steps.append(1.0e-4)
    return np.asarray(steps, dtype=float)


def beam_from_parameters(
    parameters: dict[str, float],
    fixed: BeamGeometry,
    beam_mode: str,
) -> BeamGeometry:
    if beam_mode == "fixed":
        return fixed
    major = parameters["major_fwhm_arcsec"]
    minor = parameters["minor_fwhm_arcsec"]
    angle = parameters["beam_angle_rad"]
    # Major/minor label exchange is harmless to the profile but canonicalize
    # it for stable reporting.
    if minor > major:
        major, minor = minor, major
        angle += 0.5 * math.pi
    angle = ((angle + 0.5 * math.pi) % math.pi) - 0.5 * math.pi
    return BeamGeometry(major, minor, angle)


def scan_profiled_objective(
    scan: PreparedScan,
    parameters: dict[str, float],
    model: str,
    beam: BeamGeometry,
    baseline_mode: str,
    network_include: set[int] | None = None,
) -> tuple[float, float, int, dict[str, float]]:
    tau = parameters.get("tau_sec", 0.0)
    x, y, velocity_x, velocity_y = scan.coordinates(tau)
    center_x = np.full(scan.recorded_time.shape, parameters["x0_arcsec"])
    center_y = np.full(scan.recorded_time.shape, parameters["y0_arcsec"])
    if model in {"hysteresis", "joint"}:
        center_x += parameters["h_az_arcsec"] * np.sign(velocity_x)
        center_y += parameters["h_el_arcsec"] * np.sign(velocity_y)
    template = gaussian_beam(
        x, y, center_x[:, None], center_y[:, None], beam
    )
    mask = scan.score_mask.copy()
    if network_include is not None:
        mask &= np.isin(scan.detector_network, list(network_include))[None, :]
    residual_signal = scan.residual_by_baseline[baseline_mode]
    weighted_template_signal = np.sum(
        np.where(mask, template * residual_signal, 0.0), axis=0
    )
    template_square = np.sum(np.where(mask, template * template, 0.0), axis=0)
    amplitude = np.divide(
        weighted_template_signal,
        template_square,
        out=np.zeros_like(weighted_template_signal),
        where=template_square > 1.0e-16,
    )
    amplitude = np.maximum(amplitude, 0.0)
    residual = residual_signal - template * amplitude[None, :]
    detector_sse = np.sum(np.where(mask, residual * residual, 0.0), axis=0)
    counts = np.sum(mask, axis=0)
    weighted_sse = float(np.sum(scan.ptc_weight * detector_sse))
    weight_count = float(np.sum(scan.ptc_weight * counts))
    sample_count = int(np.sum(counts))
    direction_positive = velocity_x >= 0.0
    positive_mask = mask & direction_positive[:, None]
    negative_mask = mask & ~direction_positive[:, None]
    diagnostics = {
        "positive_weighted_sse": float(np.sum(
            scan.ptc_weight * np.sum(
                np.where(positive_mask, residual * residual, 0.0), axis=0
            )
        )),
        "positive_weight_count": float(np.sum(
            scan.ptc_weight * np.sum(positive_mask, axis=0)
        )),
        "negative_weighted_sse": float(np.sum(
            scan.ptc_weight * np.sum(
                np.where(negative_mask, residual * residual, 0.0), axis=0
            )
        )),
        "negative_weight_count": float(np.sum(
            scan.ptc_weight * np.sum(negative_mask, axis=0)
        )),
    }
    return weighted_sse, weight_count, sample_count, diagnostics


def observation_objective(
    values: np.ndarray,
    observation: PreparedObservation,
    model: str,
    beam_mode: str,
    baseline_mode: str,
    scan_multiplicity: np.ndarray | None = None,
    network_include: set[int] | None = None,
) -> float:
    parameters = parameter_dict(values, model, beam_mode)
    beam = beam_from_parameters(parameters, observation.beam, beam_mode)
    if scan_multiplicity is None:
        scan_multiplicity = np.ones(len(observation.scans), dtype=float)
    if scan_multiplicity.shape != (len(observation.scans),):
        raise ContractError("scan multiplicity geometry mismatch")
    total_sse = 0.0
    total_weight = 0.0
    for multiplicity, scan in zip(scan_multiplicity, observation.scans, strict=True):
        if multiplicity <= 0.0:
            continue
        sse, weight, _, _ = scan_profiled_objective(
            scan, parameters, model, beam, baseline_mode, network_include
        )
        total_sse += float(multiplicity) * sse
        total_weight += float(multiplicity) * weight
    if not math.isfinite(total_sse) or total_weight <= 0.0:
        return math.inf
    return total_sse / total_weight


def fit_observation_model(
    observation: PreparedObservation,
    model: str,
    beam_mode: str = "fixed",
    baseline_mode: str = "constant",
    scan_multiplicity: np.ndarray | None = None,
    network_include: set[int] | None = None,
    initial: np.ndarray | None = None,
) -> dict[str, Any]:
    bounds, starts = parameter_bounds_and_starts(observation, model, beam_mode)
    if initial is not None:
        starts = [np.asarray(initial, dtype=float)]
    results = []
    for start in starts:
        result = minimize(
            observation_objective,
            start,
            args=(
                observation, model, beam_mode, baseline_mode,
                scan_multiplicity, network_include,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 500,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "eps": optimizer_finite_difference_steps(model, beam_mode),
            },
        )
        if math.isfinite(float(result.fun)):
            results.append(result)
    if not results:
        return {"status": "fit_failed", "message": "no finite optimizer result"}
    best = min(results, key=lambda item: float(item.fun))
    parameters = parameter_dict(np.asarray(best.x), model, beam_mode)
    beam = beam_from_parameters(parameters, observation.beam, beam_mode)
    lag_bounds = observation.protocol["models"]["lag_search_bounds_ms"]
    margin = float(observation.protocol["models"]["boundary_failure_margin_ms"])
    tau_ms = 1000.0 * parameters.get("tau_sec", 0.0)
    boundary = model in {"lag", "joint"} and (
        tau_ms <= float(lag_bounds[0]) + margin
        or tau_ms >= float(lag_bounds[1]) - margin
    )
    per_scan = []
    for scan in observation.scans:
        sse, weight, count, diagnostics = scan_profiled_objective(
            scan, parameters, model, beam, baseline_mode, network_include
        )
        per_scan.append({
            "scan_row": scan.scan_row,
            "output_scan_index": scan.output_scan_index,
            "weighted_sse": sse,
            "weight_count": weight,
            "weighted_mse": sse / weight if weight > 0.0 else math.nan,
            "scored_sample_count": count,
            **diagnostics,
        })
    return {
        "status": "boundary_failure" if boundary else "success",
        "model": model,
        "beam_mode": beam_mode,
        "baseline_mode": baseline_mode,
        "objective": float(best.fun),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimizer_iterations": int(best.nit),
        "parameters": parameters,
        "tau_ms": tau_ms,
        "boundary": boundary,
        "beam": {
            "major_fwhm_arcsec": beam.major_fwhm_arcsec,
            "minor_fwhm_arcsec": beam.minor_fwhm_arcsec,
            "angle_rad": beam.angle_rad,
        },
        "per_scan": per_scan,
    }


def derivative_tau_estimate(
    observation: PreparedObservation,
    no_lag_fit: dict[str, Any],
    baseline_mode: str = "constant",
) -> dict[str, float | str]:
    if no_lag_fit["status"] != "success":
        return {"status": "unavailable_no_lag_fit"}
    parameters = dict(no_lag_fit["parameters"])
    parameters["tau_sec"] = 0.0
    beam = observation.beam
    normal = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    for scan in observation.scans:
        x, y, vx, vy = scan.coordinates(0.0)
        cx = parameters["x0_arcsec"]
        cy = parameters["y0_arcsec"]
        template = gaussian_beam(x, y, cx, cy, beam)
        signal = scan.residual_by_baseline[baseline_mode]
        mask = scan.score_mask
        sum_bs = np.sum(np.where(mask, template * signal, 0.0), axis=0)
        sum_bb = np.sum(np.where(mask, template * template, 0.0), axis=0)
        amplitude = np.maximum(np.divide(
            sum_bs, sum_bb, out=np.zeros_like(sum_bs), where=sum_bb > 1.0e-16
        ), 0.0)
        sigma_major = beam.major_fwhm_arcsec * FWHM_TO_SIGMA
        sigma_minor = beam.minor_fwhm_arcsec * FWHM_TO_SIGMA
        ct = math.cos(beam.angle_rad)
        st = math.sin(beam.angle_rad)
        dx = x - cx
        dy = y - cy
        major = ct * dx + st * dy
        minor = -st * dx + ct * dy
        dlog_dx = -(ct * major / sigma_major ** 2 - st * minor / sigma_minor ** 2)
        dlog_dy = -(st * major / sigma_major ** 2 + ct * minor / sigma_minor ** 2)
        g_tau = amplitude[None, :] * template * (
            dlog_dx * vx[:, None] + dlog_dy * vy[:, None]
        )
        g_x = -amplitude[None, :] * template * dlog_dx
        g_y = -amplitude[None, :] * template * dlog_dy
        residual = signal - amplitude[None, :] * template
        # Profile the first-order template against the detector-scan amplitude
        # nuisance before solving the global [dx,dy,tau] normal equations.
        template_norm = np.sum(np.where(mask, template * template, 0.0), axis=0)
        columns = [g_x, g_y, g_tau]
        projected = []
        for column in columns:
            coefficient = np.divide(
                np.sum(np.where(mask, template * column, 0.0), axis=0),
                template_norm,
                out=np.zeros(template_norm.shape),
                where=template_norm > 1.0e-16,
            )
            projected.append(column - template * coefficient[None, :])
        residual_coefficient = np.divide(
            np.sum(np.where(mask, template * residual, 0.0), axis=0),
            template_norm,
            out=np.zeros(template_norm.shape),
            where=template_norm > 1.0e-16,
        )
        projected_residual = residual - template * residual_coefficient[None, :]
        weights = scan.ptc_weight[None, :]
        for i, first in enumerate(projected):
            rhs[i] += float(np.sum(np.where(
                mask, weights * first * projected_residual, 0.0
            )))
            for j, second in enumerate(projected):
                normal[i, j] += float(np.sum(np.where(
                    mask, weights * first * second, 0.0
                )))
    if np.linalg.matrix_rank(normal) < 3:
        return {"status": "nonpositive_template_norm"}
    solution = np.linalg.solve(normal, rhs)
    return {
        "status": "success",
        "delta_x_arcsec": float(solution[0]),
        "delta_y_arcsec": float(solution[1]),
        "tau_sec": float(solution[2]),
        "tau_ms": float(1000.0 * solution[2]),
        "template_norm": float(normal[2, 2]),
        "normal_condition_number": float(np.linalg.cond(normal)),
    }


def bootstrap_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ContractError("bootstrap contains no finite values")
    quantiles = np.percentile(values, [2.5, 16.0, 50.0, 84.0, 97.5])
    multimodal = False
    peak_count = 0
    if values.size >= 50 and np.ptp(values) > 0.0:
        central = values[
            (values >= np.percentile(values, 0.5))
            & (values <= np.percentile(values, 99.5))
        ]
        if np.unique(central).size > 2:
            grid = np.linspace(float(np.min(central)), float(np.max(central)), 256)
            density = gaussian_kde(central)(grid)
            peaks, properties = find_peaks(
                density, height=0.2 * float(np.max(density))
            )
            peak_count = int(peaks.size)
            multimodal = peak_count > 1
    return {
        "successful_count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(quantiles[2]),
        "interval_68": [float(quantiles[1]), float(quantiles[3])],
        "interval_95": [float(quantiles[0]), float(quantiles[4])],
        "p_negative": float(np.mean(values < 0.0)),
        "multimodal": multimodal,
        "kde_peak_count": peak_count,
    }


def fit_to_optimizer_vector(
    fit: dict[str, Any], model: str, beam_mode: str
) -> np.ndarray:
    parameters = fit["parameters"]
    values = []
    for name in model_parameter_names(model, beam_mode):
        value = float(parameters[name])
        if name == "tau_sec":
            value *= 1000.0
        values.append(value)
    return np.asarray(values, dtype=float)


def fit_at_fixed_tau(
    observation: PreparedObservation,
    tau_ms: float,
    initial_fit: dict[str, Any],
    *,
    baseline_mode: str = "constant",
) -> dict[str, Any]:
    """Profile x0/y0 while holding exact-shift tau fixed."""
    full_bounds, _ = parameter_bounds_and_starts(observation, "lag", "fixed")
    initial = fit_to_optimizer_vector(initial_fit, "lag", "fixed")[:2]

    def objective(xy: np.ndarray) -> float:
        return observation_objective(
            np.asarray([xy[0], xy[1], tau_ms]),
            observation, "lag", "fixed", baseline_mode,
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=full_bounds[:2],
        options={
            "maxiter": 300,
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "eps": np.asarray([1.0e-4, 1.0e-4]),
        },
    )
    return {
        "tau_ms": float(tau_ms),
        "objective": float(result.fun),
        "x0_arcsec": float(result.x[0]),
        "y0_arcsec": float(result.x[1]),
        "optimizer_success": bool(result.success),
    }


def objective_profile(
    observation: PreparedObservation, lag_fit: dict[str, Any]
) -> list[dict[str, Any]]:
    spec = observation.protocol["models"]["objective_profile_tau_grid_ms"]
    grid = np.linspace(
        float(spec["minimum"]), float(spec["maximum"]), int(spec["count"])
    )
    return [fit_at_fixed_tau(observation, tau, lag_fit) for tau in grid]


def heldout_model_comparison(
    observation: PreparedObservation,
    full_fits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Leave one complete PTC scan row out and score held-out residuals."""
    rows = []
    winners = {name: 0 for name in MODEL_NAMES}
    n_scan = len(observation.scans)
    for heldout in range(n_scan):
        multiplicity = np.ones(n_scan, dtype=float)
        multiplicity[heldout] = 0.0
        scores: dict[str, float] = {}
        for model in MODEL_NAMES:
            initial = fit_to_optimizer_vector(full_fits[model], model, "fixed")
            training = fit_observation_model(
                observation, model, scan_multiplicity=multiplicity,
                initial=initial,
            )
            if training["status"] not in {"success", "boundary_failure"}:
                scores[model] = math.inf
                continue
            parameters = training["parameters"]
            beam = beam_from_parameters(parameters, observation.beam, "fixed")
            sse, weight, _, _ = scan_profiled_objective(
                observation.scans[heldout], parameters, model, beam, "constant"
            )
            scores[model] = sse / weight if weight > 0.0 else math.inf
        finite = {key: value for key, value in scores.items() if math.isfinite(value)}
        winner = min(finite, key=finite.get) if finite else "none"
        if winner in winners:
            winners[winner] += 1
        rows.append({
            "heldout_scan_row": heldout,
            "output_scan_index": observation.scans[heldout].output_scan_index,
            "winner": winner,
            **{f"{name}_score": scores[name] for name in MODEL_NAMES},
        })
    return {
        "fold_count": n_scan,
        "winner_counts": winners,
        "winner_frequencies": {
            name: count / n_scan for name, count in winners.items()
        },
        "folds": rows,
    }


def convergence_change(
    earlier: dict[str, Any], later: dict[str, Any]
) -> dict[str, float]:
    return {
        "median_change": abs(later["median"] - earlier["median"]),
        "interval_68_endpoint_change": max(
            abs(later["interval_68"][0] - earlier["interval_68"][0]),
            abs(later["interval_68"][1] - earlier["interval_68"][1]),
        ),
        "p_negative_change": abs(later["p_negative"] - earlier["p_negative"]),
    }


def bootstrap_is_converged(
    values: np.ndarray, protocol: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    if values.size < 500:
        return False, {"status": "insufficient_count"}
    earlier = bootstrap_summary(values[: values.size - 250])
    later = bootstrap_summary(values)
    change = convergence_change(earlier, later)
    limits = protocol["bootstrap"]["timestream_convergence"]
    passed = (
        not later["multimodal"]
        and change["median_change"] <= float(limits["median_change_max_ms"])
        and change["interval_68_endpoint_change"]
        <= float(limits["interval_endpoint_change_max_ms"])
        and change["p_negative_change"]
        <= float(limits["p_tau_negative_change_max"])
    )
    return passed, {
        "status": "pass" if passed else "extend",
        "multimodal": bool(later["multimodal"]),
        "kde_peak_count": int(later["kde_peak_count"]),
        **change,
    }


@dataclass
class MapScanAccumulator:
    scan_row: int
    items: list[Any]


def map_scan_accumulators(
    ptc_path: Path, protocol: dict[str, Any]
) -> list[MapScanAccumulator]:
    """Reconstruct the committed map estimator's additive terms by scan."""
    pixel = 2.0
    half = 80.0
    minimum_speed = 5.0
    n_side = int(round(2.0 * half / pixel)) + 1
    shape = (n_side, n_side)
    result: list[MapScanAccumulator] = []
    with netCDF4.Dataset(ptc_path) as dataset:
        scans = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        time_all = np.asarray(dataset.variables["TelTime"][:], dtype=float)
        az_all = np.asarray(dataset.variables["az_phys"][:], dtype=float)
        alt_all = np.asarray(dataset.variables["alt_phys"][:], dtype=float)
        elevation_all = np.asarray(dataset.variables["TelElAct"][:], dtype=float)
        po_az_all = np.asarray(
            dataset.variables["pointing_offset_az"][:], dtype=float
        )
        po_alt_all = np.asarray(
            dataset.variables["pointing_offset_alt"][:], dtype=float
        )
        apt_array = np.asarray(dataset.variables["apt_array"][:], dtype=int)
        apt_flag = np.asarray(dataset.variables["apt_flag"][:], dtype=int)
        apt_x = np.asarray(dataset.variables["apt_x_t"][:], dtype=float)
        apt_y = np.asarray(dataset.variables["apt_y_t"][:], dtype=float)
        keep = (
            (apt_array == 0) & (apt_flag == 0)
            & np.isfinite(apt_x) & np.isfinite(apt_y)
        )
        detector = np.flatnonzero(keep)
        weights_all = np.asarray(dataset.variables["weights"][:, :], dtype=float)[
            :, detector
        ]
        for scan_row, (start, stop) in enumerate(scans):
            items = [map_space.empty_sector(shape) for _ in range(9)]
            sl = slice(int(start), int(stop) + 1)
            time = time_all[sl]
            az = az_all[sl]
            alt = alt_all[sl]
            elevation = elevation_all[sl]
            vx, vy = map_space.scan_velocity(time, az, alt)
            speed = np.hypot(vx, vy)
            sector_for_time = map_space.sector_index(np.arctan2(vy, vx))
            signal = np.asarray(dataset.variables["signal"][sl, :], dtype=float)[
                :, detector
            ]
            flags = np.asarray(dataset.variables["flags"][sl, :], dtype=float)[
                :, detector
            ]
            scan_weights = weights_all[scan_row]
            ct = np.cos(elevation)[:, None]
            st = np.sin(elevation)[:, None]
            lon = (
                az[:, None] * RAD_TO_ARCSEC
                + ct * apt_x[detector][None, :]
                - st * apt_y[detector][None, :]
                + po_az_all[sl, None]
            )
            lat = (
                alt[:, None] * RAD_TO_ARCSEC
                + ct * apt_y[detector][None, :]
                + st * apt_x[detector][None, :]
                + po_alt_all[sl, None]
            )
            col = np.floor((lon + half) / pixel + 0.5).astype(int)
            row_index = np.floor((lat + half) / pixel + 0.5).astype(int)
            base_good = (
                np.isfinite(signal) & np.isfinite(flags) & (flags == 0.0)
                & np.isfinite(scan_weights)[None, :]
                & (scan_weights[None, :] > 0.0)
                & (col >= 0) & (col < n_side)
                & (row_index >= 0) & (row_index < n_side)
                & np.isfinite(speed)[:, None]
                & (speed[:, None] >= minimum_speed)
            )
            for item_index, item in enumerate(items):
                good = base_good if item_index == 0 else (
                    base_good & (sector_for_time[:, None] == item_index - 1)
                )
                rr, dd = np.nonzero(good)
                if rr.size == 0:
                    continue
                cc = col[rr, dd]
                yy = row_index[rr, dd]
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
            result.append(MapScanAccumulator(scan_row, items))
    return result


def aggregate_map_scans(
    scans: list[MapScanAccumulator], multiplicity: np.ndarray
) -> list[Any]:
    if multiplicity.shape != (len(scans),):
        raise ContractError("map scan multiplicity geometry mismatch")
    shape = scans[0].items[0].weight.shape
    total = [map_space.empty_sector(shape) for _ in range(9)]
    for count, scan in zip(multiplicity, scans, strict=True):
        if count <= 0:
            continue
        for target, source in zip(total, scan.items, strict=True):
            target.weighted_signal += count * source.weighted_signal
            target.weight += count * source.weight
            target.hit_count += int(count) * source.hit_count
            target.velocity_weight += count * source.velocity_weight
            target.velocity_x_sum += count * source.velocity_x_sum
            target.velocity_y_sum += count * source.velocity_y_sum
            target.accepted += int(count) * source.accepted
    return total


def fit_map_resample(
    scans: list[MapScanAccumulator],
    multiplicity: np.ndarray,
    ppt_x: float,
    ppt_y: float,
) -> dict[str, Any]:
    items = aggregate_map_scans(scans, multiplicity)
    axis = np.linspace(-80.0, 80.0, items[0].weight.shape[0])
    full_fit = map_space.fit_gaussian_core(
        map_space.signal_image(items[0]), axis, axis, ppt_x, ppt_y, 15.0
    )
    if full_fit["status"] != "success":
        return {"status": "full_map_fit_failed"}
    sectors = []
    for index, item in enumerate(items[1:]):
        fit = map_space.fit_gaussian_core(
            map_space.signal_image(item), axis, axis,
            float(full_fit["x_arcsec"]), float(full_fit["y_arcsec"]), 15.0,
        )
        row = {
            "sector": index,
            "status": fit["status"],
            "velocity_x_arcsec_s": (
                item.velocity_x_sum / item.velocity_weight
                if item.velocity_weight > 0.0 else math.nan
            ),
            "velocity_y_arcsec_s": (
                item.velocity_y_sum / item.velocity_weight
                if item.velocity_weight > 0.0 else math.nan
            ),
        }
        row.update(fit)
        sectors.append(row)
    try:
        models = map_space.fit_models(sectors)
    except (map_space.ContractError, ValueError, np.linalg.LinAlgError):
        return {"status": "sector_model_fit_failed"}
    lag = next(row for row in models if row["model"] == "time_lag")
    raw_tau_ms = float(lag["tau_ms"])
    return {
        "status": "success",
        "raw_centroid_slope_tau_ms": raw_tau_ms,
        "coordinate_shift_tau_ms": map_centroid_tau_to_coordinate_shift_ms(
            raw_tau_ms
        ),
    }


def authenticated_map_result(map_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    verify_sha256s(map_root)
    obsnum = int(row["pointing_obsnum"])
    obs_root = map_root / f"o{obsnum}"
    verify_sha256s(obs_root)
    result = json.loads((obs_root / "result.json").read_text())
    if result.get("schema") != "sci-align-001-lissajous-pointing-result-v1":
        raise ContractError(f"obs {obsnum}: unsupported authenticated map result")
    map_input = result["input"]
    if (
        int(map_input["pointing_obsnum"]) != obsnum
        or map_input["ptc_sha256"] != row["ptc_sha256"]
        or map_input["ppt_sha256"] != row["ppt_sha256"]
    ):
        raise ContractError(f"obs {obsnum}: map/timestream input identity mismatch")
    lag = next(
        item for item in result["model_results"] if item["model"] == "time_lag"
    )
    raw = float(lag["tau_ms"])
    return {
        "result_path": str((obs_root / "result.json").resolve()),
        "result_sha256": sha256_file(obs_root / "result.json"),
        "obs_sha256s_sha256": sha256_file(obs_root / "SHA256SUMS"),
        "raw_centroid_slope_tau_ms": raw,
        "coordinate_shift_tau_ms": map_centroid_tau_to_coordinate_shift_ms(raw),
    }


def paired_convergence(
    timestream: np.ndarray,
    map_values: np.ndarray,
    count: int,
    protocol: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if count < 250:
        return False, {"status": "insufficient_count"}

    def metrics(n: int) -> tuple[dict[str, Any], float]:
        good = np.isfinite(timestream[:n]) & np.isfinite(map_values[:n])
        delta = timestream[:n][good] - map_values[:n][good]
        if delta.size < 2:
            raise ContractError("paired bootstrap has fewer than two finite pairs")
        covariance = float(np.cov(
            timestream[:n][good], map_values[:n][good], ddof=1
        )[0, 1])
        return bootstrap_summary(delta), covariance

    earlier, cov_earlier = metrics(count - 50)
    later, cov_later = metrics(count)
    endpoint = max(
        abs(later["interval_68"][0] - earlier["interval_68"][0]),
        abs(later["interval_68"][1] - earlier["interval_68"][1]),
    )
    relative_covariance = abs(cov_later - cov_earlier) / max(
        abs(cov_later), abs(cov_earlier), 1.0e-12
    )
    limits = protocol["bootstrap"]["paired_map_convergence"]
    passed = (
        abs(later["median"] - earlier["median"])
        <= float(limits["delta_median_change_max_ms"])
        and endpoint <= float(limits["delta_interval_endpoint_change_max_ms"])
        and relative_covariance
        <= float(limits["covariance_relative_change_max"])
    )
    return passed, {
        "status": "pass" if passed else "extend",
        "delta_median_change_ms": abs(later["median"] - earlier["median"]),
        "delta_interval_endpoint_change_ms": endpoint,
        "covariance_relative_change": relative_covariance,
    }


def bootstrap_observation(
    observation: PreparedObservation,
    lag_fit: dict[str, Any],
    map_scans: list[MapScanAccumulator],
    output: Path,
) -> dict[str, Any]:
    spec = observation.protocol["bootstrap"]
    n_scan = len(observation.scans)
    seed = 2026081000 + observation.obsnum
    rng = np.random.default_rng(seed)
    max_ts = int(spec["maximum_timestream_replicates"])
    max_map = int(spec["paired_map_maximum"])
    max_draw = max(max_ts, max_map)
    draw_indices = rng.integers(0, n_scan, size=(max_draw, n_scan))
    multiplicities = np.asarray([
        np.bincount(draw, minlength=n_scan) for draw in draw_indices
    ], dtype=np.int16)
    work_path = output / "bootstrap_work.npz"
    timestream = np.full(max_ts, np.nan)
    map_values = np.full(max_map, np.nan)
    if work_path.exists():
        with np.load(work_path) as work:
            if int(work["seed"]) != seed or int(work["n_scan"]) != n_scan:
                raise ContractError("bootstrap checkpoint identity mismatch")
            timestream[: len(work["timestream"])] = work["timestream"]
            map_values[: len(work["map_values"])] = work["map_values"]
    initial = fit_to_optimizer_vector(lag_fit, "lag", "fixed")

    def checkpoint() -> None:
        np.savez_compressed(
            work_path, seed=seed, n_scan=n_scan,
            timestream=timestream, map_values=map_values,
        )

    ts_target = int(spec["minimum_successful_timestream_replicates"])
    ts_convergence: dict[str, Any] = {"status": "not_evaluated"}
    while True:
        completed = int(np.count_nonzero(np.isfinite(timestream)))
        for index in range(max_ts):
            if np.isfinite(timestream[index]):
                continue
            fit = fit_observation_model(
                observation, "lag",
                scan_multiplicity=multiplicities[index].astype(float),
                initial=initial,
            )
            if fit["status"] == "success":
                timestream[index] = float(fit["tau_ms"])
            if (index + 1) % 25 == 0:
                checkpoint()
            completed = int(np.count_nonzero(np.isfinite(timestream)))
            if completed >= ts_target:
                break
        finite_ts = timestream[np.isfinite(timestream)]
        if finite_ts.size < ts_target:
            raise ContractError("timestream bootstrap cannot reach target")
        converged, ts_convergence = bootstrap_is_converged(
            finite_ts[:ts_target], observation.protocol
        )
        if converged or ts_target >= max_ts:
            break
        ts_target = min(
            max_ts, ts_target + int(spec["timestream_increment_if_not_converged"])
        )

    map_target = int(spec["paired_map_initial_target"])
    map_convergence: dict[str, Any] = {"status": "not_evaluated"}
    while True:
        for index in range(map_target):
            if np.isfinite(map_values[index]):
                continue
            result = fit_map_resample(
                map_scans, multiplicities[index].astype(float),
                observation.ppt_x_arcsec, observation.ppt_y_arcsec,
            )
            if result["status"] == "success":
                map_values[index] = float(result["coordinate_shift_tau_ms"])
            if (index + 1) % 25 == 0:
                checkpoint()
        paired_good = (
            np.isfinite(timestream[:map_target])
            & np.isfinite(map_values[:map_target])
        )
        if np.count_nonzero(paired_good) < int(spec["paired_map_minimum_successful"]):
            if map_target >= max_map:
                raise ContractError("paired map bootstrap cannot reach minimum")
            map_target = min(max_map, map_target + 50)
            continue
        converged, map_convergence = paired_convergence(
            timestream, map_values, map_target, observation.protocol
        )
        if converged or map_target >= max_map:
            break
        map_target = min(
            max_map, map_target + int(spec["paired_map_increment_if_not_converged"])
        )
    checkpoint()
    ts_values = timestream[np.isfinite(timestream)][:ts_target]
    paired_good = (
        np.isfinite(timestream[:map_target])
        & np.isfinite(map_values[:map_target])
    )
    ts_pair = timestream[:map_target][paired_good]
    map_pair = map_values[:map_target][paired_good]
    delta = ts_pair - map_pair
    covariance = float(np.cov(ts_pair, map_pair, ddof=1)[0, 1])
    correlation = float(np.corrcoef(ts_pair, map_pair)[0, 1])
    return {
        "seed": seed,
        "whole_scan_resampling_unit": observation.protocol["common_support"][
            "whole_scan_resampling_unit"
        ],
        "timestream_target_count": ts_target,
        "timestream_summary": bootstrap_summary(ts_values),
        "timestream_convergence": ts_convergence,
        "paired_map_target_count": map_target,
        "paired_successful_count": int(delta.size),
        "map_coordinate_shift_summary": bootstrap_summary(map_pair),
        "paired_delta_summary": bootstrap_summary(delta),
        "timestream_map_covariance_ms2": covariance,
        "timestream_map_correlation": correlation,
        "paired_convergence": map_convergence,
    }


def sensitivity_fits(
    observation: PreparedObservation, primary: dict[str, Any]
) -> dict[str, Any]:
    fixed_initial = fit_to_optimizer_vector(primary, "lag", "fixed")
    linear = fit_observation_model(
        observation, "lag", baseline_mode="linear", initial=fixed_initial
    )
    free_initial = np.concatenate([
        fixed_initial,
        np.asarray([
            observation.beam.major_fwhm_arcsec,
            observation.beam.minor_fwhm_arcsec,
            observation.beam.angle_rad,
        ]),
    ])
    free = fit_observation_model(
        observation, "lag", beam_mode="free", initial=free_initial
    )
    return {"linear_baseline_fixed_beam": linear, "constant_baseline_free_beam": free}


def network_sensitivity(
    observation: PreparedObservation, primary: dict[str, Any]
) -> list[dict[str, Any]]:
    spec = observation.protocol["sensitivities"]
    initial = fit_to_optimizer_vector(primary, "lag", "fixed")
    networks = set(observation.eligible_networks)
    rows = []
    for network in sorted(networks):
        detector_uids = set()
        scored = 0
        for scan in observation.scans:
            selected = scan.detector_network == network
            detector_uids.update(map(int, scan.detector_uid[selected]))
            scored += int(np.count_nonzero(scan.score_mask[:, selected]))
        adequate = (
            len(detector_uids) >= int(spec["per_network_minimum_detector_count"])
            and scored >= int(spec["per_network_minimum_scored_sample_count"])
        )
        for kind, included in (
            ("single_network", {network}),
            ("leave_one_network_out", networks - {network}),
        ):
            if kind == "single_network" and not adequate:
                rows.append({
                    "kind": kind, "network": network, "status": "insufficient",
                    "detector_count": len(detector_uids),
                    "scored_sample_count": scored, "tau_ms": math.nan,
                })
                continue
            fit = fit_observation_model(
                observation, "lag", network_include=set(included), initial=initial
            )
            rows.append({
                "kind": kind,
                "network": network,
                "status": fit["status"],
                "detector_count": len(detector_uids),
                "scored_sample_count": scored,
                "tau_ms": float(fit.get("tau_ms", math.nan)),
            })
    return rows


def model_sensitivity_status(
    primary: dict[str, Any],
    sensitivities: dict[str, Any],
    bootstrap: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    interval = bootstrap["timestream_summary"]["interval_68"]
    limit = float(protocol["sensitivities"]["model_sensitive_tau_change_ms"])
    primary_tau = float(primary["tau_ms"])
    entries = []
    sensitive = False
    for name, fit in sensitivities.items():
        tau = float(fit.get("tau_ms", math.nan))
        change = abs(tau - primary_tau) if math.isfinite(tau) else math.inf
        outside = not (float(interval[0]) <= tau <= float(interval[1]))
        failed = fit.get("status") != "success"
        item_sensitive = failed or change > limit or outside
        sensitive |= item_sensitive
        entries.append({
            "name": name, "status": fit.get("status"), "tau_ms": tau,
            "absolute_change_ms": change,
            "outside_primary_bootstrap_68": outside,
            "model_sensitive": item_sensitive,
        })
    return {"model_sensitive": sensitive, "comparisons": entries}


def write_observation_plots(
    output: Path,
    obsnum: int,
    profile: list[dict[str, Any]],
    primary: dict[str, Any],
    bootstrap: dict[str, Any],
) -> str:
    pdf_name = f"lissajous_timestream_diagnostic_o{obsnum}.pdf"
    with PdfPages(output / pdf_name) as pdf:
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        tau = np.asarray([row["tau_ms"] for row in profile])
        objective = np.asarray([row["objective"] for row in profile])
        ax.plot(tau, objective - np.nanmin(objective), "o-")
        ax.axvline(float(primary["tau_ms"]), color="C1", label="exact fit")
        ax.set_xlabel("exact coordinate-shift tau (ms)")
        ax.set_ylabel("profile objective - minimum")
        ax.set_title(f"Obs {obsnum}: objective profile")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        per_scan = primary["per_scan"]
        axes[0].plot(
            [row["scan_row"] for row in per_scan],
            [row["weighted_mse"] for row in per_scan], "o-",
        )
        axes[0].set_xlabel("complete PTC scan row")
        axes[0].set_ylabel("profiled weighted MSE")
        axes[0].set_title("Residual diagnostic by scan")
        with np.load(output / "bootstrap_work.npz") as work:
            ts = np.asarray(work["timestream"], dtype=float)
            mp = np.asarray(work["map_values"], dtype=float)
        good = np.isfinite(ts)
        axes[1].hist(ts[good], bins=30, alpha=0.6, label="timestream")
        n = min(ts.size, mp.size)
        pair = np.isfinite(ts[:n]) & np.isfinite(mp[:n])
        axes[1].hist(mp[:n][pair], bins=30, alpha=0.6, label="map, sign-aligned")
        axes[1].set_xlabel("coordinate-shift tau (ms)")
        axes[1].set_ylabel("whole-scan bootstrap count")
        axes[1].legend()
        axes[1].set_title(
            f"paired covariance={bootstrap['timestream_map_covariance_ms2']:.3g} ms²"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_name


def analyze_observation(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    selection = load_selection(
        args.selection, protocol["input_authority"]["selection_manifest_sha256"]
    )
    row = selected_row(selection, args.obsnum)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ContractError(f"completed output already exists: {output}")
    observation = prepare_observation(row, protocol)
    coordinate_gate = coordinate_reconstruction_gate(observation)
    map_result = authenticated_map_result(args.map_root.resolve(), row)
    full_fits = {
        model: fit_observation_model(observation, model) for model in MODEL_NAMES
    }
    primary = full_fits["lag"]
    if primary["status"] != "success":
        raise ContractError(f"obs {args.obsnum}: primary lag fit failed")
    profile = objective_profile(observation, primary)
    derivative = derivative_tau_estimate(observation, full_fits["constant"])
    blocked = heldout_model_comparison(observation, full_fits)
    sensitivities = sensitivity_fits(observation, primary)
    network_rows = network_sensitivity(observation, primary)
    map_scans = map_scan_accumulators(observation.ptc_path, protocol)
    if len(map_scans) != len(observation.scans):
        raise ContractError("map and timestream scan-row counts differ")
    bootstrap = bootstrap_observation(observation, primary, map_scans, output)
    sensitivity_status = model_sensitivity_status(
        primary, sensitivities, bootstrap, protocol
    )
    Table(rows=profile).write(output / "objective_profile.ecsv", format="ascii.ecsv")
    Table(rows=blocked["folds"]).write(
        output / "blocked_prediction.ecsv", format="ascii.ecsv"
    )
    Table(rows=network_rows).write(
        output / "network_sensitivity.ecsv", format="ascii.ecsv"
    )
    model_rows = []
    for model, fit in full_fits.items():
        model_rows.append({
            "model": model, "status": fit["status"],
            "objective": fit["objective"], "tau_ms": fit["tau_ms"],
            "x0_arcsec": fit["parameters"]["x0_arcsec"],
            "y0_arcsec": fit["parameters"]["y0_arcsec"],
            "h_az_arcsec": fit["parameters"].get("h_az_arcsec", math.nan),
            "h_el_arcsec": fit["parameters"].get("h_el_arcsec", math.nan),
        })
    Table(rows=model_rows).write(
        output / "point_model_results.ecsv", format="ascii.ecsv"
    )
    pdf_name = write_observation_plots(
        output, observation.obsnum, profile, primary, bootstrap
    )
    result = {
        "schema": "sci-align-001-lissajous-timestream-observation-result-v1",
        "obsnum": observation.obsnum,
        "beammap_obsnum": int(row["beammap_obsnum"]),
        "brightness_stratum": row["brightness_stratum"],
        "input": {
            "ptc_path": str(observation.ptc_path),
            "ptc_sha256": row["ptc_sha256"],
            "ppt_path": str(observation.ppt_path),
            "ppt_sha256": row["ppt_sha256"],
            "protocol_sha256": sha256_file(args.protocol),
            "selection_sha256": sha256_file(args.selection),
            "map_result": map_result,
        },
        "support": {
            "scan_count": len(observation.scans),
            "eligible_uid_count": observation.eligible_uid_count,
            "eligible_networks": observation.eligible_networks,
            "common_support_sample_count": observation.common_support_sample_count,
            "scored_value_count": observation.scored_value_count,
            "map_support_difference": "committed map estimator retains its full scan support and map eligibility; exact timestream estimator uses frozen +/-50-ms common edge trim and source-scoring eligibility",
        },
        "coordinate_gate": coordinate_gate,
        "point_model_results": full_fits,
        "derivative_crosscheck": derivative,
        "blocked_prediction": blocked,
        "sensitivity_fits": sensitivities,
        "model_sensitivity": sensitivity_status,
        "bootstrap": bootstrap,
        "primary_tau_ms": float(primary["tau_ms"]),
        "map_coordinate_shift_tau_ms": map_result["coordinate_shift_tau_ms"],
        "point_difference_ms": (
            float(primary["tau_ms"]) - map_result["coordinate_shift_tau_ms"]
        ),
        "sign_agreement": bool(
            np.sign(primary["tau_ms"])
            == np.sign(map_result["coordinate_shift_tau_ms"])
        ),
    }
    write_json(output / "result.json", result)
    names = [
        "blocked_prediction.ecsv", "bootstrap_work.npz",
        "network_sensitivity.ecsv", "objective_profile.ecsv",
        "point_model_results.ecsv", pdf_name, "result.json",
    ]
    write_checksums(output, names)
    print(
        f"observation complete: obs={observation.obsnum} "
        f"tau_ms={primary['tau_ms']:.6f} output={output}"
    )


def extend_observation_bootstrap(args: argparse.Namespace) -> None:
    """Extend a checksum-valid completed observation without refitting gates."""
    protocol = load_protocol(args.protocol)
    selection = load_selection(
        args.selection, protocol["input_authority"]["selection_manifest_sha256"]
    )
    row = selected_row(selection, args.obsnum)
    output = args.output.resolve()
    verify_sha256s(output)
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    if int(result["obsnum"]) != args.obsnum:
        raise ContractError("extension output observation identity mismatch")
    predecessor_protocol_sha256 = (
        "5366dd8cfe963e29bf273a7c764637f9b85586f211963920acdc95b2610f9ad1"
    )
    current_protocol_sha256 = sha256_file(args.protocol)
    original_protocol_sha256 = result["input"]["protocol_sha256"]
    if original_protocol_sha256 not in {
        predecessor_protocol_sha256, current_protocol_sha256,
    }:
        raise ContractError("extension protocol identity is not an allowed predecessor")
    if not (output / "result_initial_500.json").exists():
        shutil.copy2(result_path, output / "result_initial_500.json")
        shutil.copy2(output / "SHA256SUMS", output / "SHA256SUMS_initial_500")
    observation = prepare_observation(row, protocol)
    coordinate_reconstruction_gate(observation)
    authenticated_map_result(args.map_root.resolve(), row)
    primary = result["point_model_results"]["lag"]
    map_scans = map_scan_accumulators(observation.ptc_path, protocol)
    bootstrap = bootstrap_observation(observation, primary, map_scans, output)
    result["bootstrap"] = bootstrap
    result["model_sensitivity"] = model_sensitivity_status(
        primary, result["sensitivity_fits"], bootstrap, protocol
    )
    result["bootstrap_extension"] = {
        "reason": "500-replicate timestream distribution was multimodal",
        "point_fit_protocol_sha256": original_protocol_sha256,
        "extension_protocol_sha256": current_protocol_sha256,
        "preserved_initial_result": "result_initial_500.json",
        "preserved_initial_checksums": "SHA256SUMS_initial_500",
    }
    table = Table.read(output / "objective_profile.ecsv", format="ascii.ecsv")
    profile = [
        {
            name: (
                table_row[name].item()
                if hasattr(table_row[name], "item") else table_row[name]
            )
            for name in table.colnames
        }
        for table_row in table
    ]
    pdf_name = write_observation_plots(
        output, observation.obsnum, profile, primary, bootstrap
    )
    write_json(result_path, result)
    names = [
        "blocked_prediction.ecsv", "bootstrap_work.npz",
        "network_sensitivity.ecsv", "objective_profile.ecsv",
        "point_model_results.ecsv", pdf_name, "result.json",
        "result_initial_500.json", "SHA256SUMS_initial_500",
    ]
    write_checksums(output, names)
    print(
        f"bootstrap extension complete: obs={observation.obsnum} "
        f"target={bootstrap['timestream_target_count']} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-input")
    inspect.add_argument("--protocol", type=Path, required=True)
    inspect.add_argument("--selection", type=Path, required=True)
    inspect.add_argument("--obsnum", type=int, required=True)
    fit = sub.add_parser("fit-anchor")
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--selection", type=Path, required=True)
    fit.add_argument("--obsnum", type=int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze-observation")
    analyze.add_argument("--protocol", type=Path, required=True)
    analyze.add_argument("--selection", type=Path, required=True)
    analyze.add_argument("--map-root", type=Path, required=True)
    analyze.add_argument("--obsnum", type=int, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    extend = sub.add_parser("extend-bootstrap")
    extend.add_argument("--protocol", type=Path, required=True)
    extend.add_argument("--selection", type=Path, required=True)
    extend.add_argument("--map-root", type=Path, required=True)
    extend.add_argument("--obsnum", type=int, required=True)
    extend.add_argument("--output", type=Path, required=True)
    return result


def selected_row(
    selection: dict[str, Any], obsnum: int
) -> dict[str, Any]:
    rows = [
        row for row in selection["rows"]
        if int(row["pointing_obsnum"]) == obsnum
    ]
    if len(rows) != 1:
        raise ContractError(f"selection does not contain exactly one obs {obsnum}")
    return rows[0]


def inspect_input(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    selection = load_selection(
        args.selection, protocol["input_authority"]["selection_manifest_sha256"]
    )
    observation = prepare_observation(selected_row(selection, args.obsnum), protocol)
    print(json.dumps({
        "obsnum": observation.obsnum,
        "scan_count": len(observation.scans),
        "eligible_uid_count": observation.eligible_uid_count,
        "eligible_networks": observation.eligible_networks,
        "common_support_sample_count": observation.common_support_sample_count,
        "scored_value_count": observation.scored_value_count,
        "coordinate_gate": coordinate_reconstruction_gate(observation),
    }, indent=2))


def fit_anchor(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    selection = load_selection(
        args.selection, protocol["input_authority"]["selection_manifest_sha256"]
    )
    expected_anchor = int(protocol["anchor_gate"]["pointing_obsnum"])
    if args.obsnum != expected_anchor:
        raise ContractError(
            f"fit-anchor is restricted to frozen anchor {expected_anchor}"
        )
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    observation = prepare_observation(selected_row(selection, args.obsnum), protocol)
    coordinate_gate = coordinate_reconstruction_gate(observation)
    results = []
    for model in MODEL_NAMES:
        results.append(fit_observation_model(observation, model))
    derivative = derivative_tau_estimate(observation, results[0])
    manifest = {
        "schema": "sci-align-001-lissajous-timestream-anchor-v1",
        "obsnum": args.obsnum,
        "protocol_sha256": sha256_file(args.protocol),
        "selection_sha256": sha256_file(args.selection),
        "support": {
            "scan_count": len(observation.scans),
            "eligible_uid_count": observation.eligible_uid_count,
            "eligible_networks": observation.eligible_networks,
            "common_support_sample_count": observation.common_support_sample_count,
            "scored_value_count": observation.scored_value_count,
        },
        "coordinate_gate": coordinate_gate,
        "model_results": results,
        "derivative_crosscheck": derivative,
    }
    write_json(output / "anchor_result.json", manifest)
    write_checksums(output, ["anchor_result.json"])
    print(f"anchor fit complete: obs={args.obsnum} output={output}")


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "inspect-input":
            inspect_input(args)
        elif args.command == "fit-anchor":
            fit_anchor(args)
        elif args.command == "analyze-observation":
            analyze_observation(args)
        elif args.command == "extend-bootstrap":
            extend_observation_bootstrap(args)
        else:  # pragma: no cover
            raise ContractError(f"unsupported command: {args.command}")
    except (ContractError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
