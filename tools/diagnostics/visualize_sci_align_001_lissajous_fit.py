#!/usr/bin/env python3
"""Render authenticated SCI-ALIGN Lissajous fit and model-adequacy evidence."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402


class VisualizationError(RuntimeError):
    """An input or generated visualization violates the audit contract."""


@dataclass(frozen=True)
class FitUnit:
    unit_id: str
    scan_row: int
    output_scan_index: int
    detector_index: int
    uid: int
    network: int
    segment_index: int
    start: int
    stop: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise VisualizationError("segment mask must be one-dimensional")
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(map(int, starts), map(int, stops), strict=True))


def fit_vector(fit: dict[str, Any], model: str = "lag") -> np.ndarray:
    return analysis.fit_to_optimizer_vector(fit, model, "fixed")


def profile_fixed_tau(
    observation: analysis.PreparedObservation,
    tau_ms: float,
    initial_fit: dict[str, Any],
) -> dict[str, Any]:
    """Profile x/y at fixed tau with deterministic numerical fallback starts."""
    full_bounds, _ = analysis.parameter_bounds_and_starts(
        observation, "lag", "fixed"
    )
    primary = fit_vector(initial_fit)[:2]
    ppt = np.asarray([
        observation.ppt_x_arcsec, observation.ppt_y_arcsec
    ])
    offset = 0.25 * min(
        observation.beam.major_fwhm_arcsec,
        observation.beam.minor_fwhm_arcsec,
    )
    starts = [
        primary,
        ppt,
        ppt + np.asarray([offset, 0.0]),
        ppt + np.asarray([-offset, 0.0]),
        ppt + np.asarray([0.0, offset]),
        ppt + np.asarray([0.0, -offset]),
    ]

    def objective(xy: np.ndarray) -> float:
        return analysis.observation_objective(
            np.asarray([xy[0], xy[1], tau_ms]),
            observation, "lag", "fixed", "constant",
        )

    results = []
    finite_results = []
    attempted = 0
    for start_index, start in enumerate(starts):
        attempted += 1
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=full_bounds[:2],
            options={
                "maxiter": 300,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "eps": np.asarray([1.0e-4, 1.0e-4]),
            },
        )
        if math.isfinite(float(result.fun)):
            finite_results.append(result)
        if bool(result.success) and math.isfinite(float(result.fun)):
            results.append(result)
            if start_index == 0:
                break
    candidates = results or finite_results
    if not candidates:
        raise VisualizationError(f"fixed-tau profile failed at tau={tau_ms:g} ms")
    best = min(candidates, key=lambda item: float(item.fun))
    return {
        "tau_ms": float(tau_ms),
        "objective": float(best.fun),
        "x0_arcsec": float(best.x[0]),
        "y0_arcsec": float(best.x[1]),
        "optimizer_success": bool(results),
        "optimizer_attempt_count": attempted,
        "optimizer_converged_count": len(results),
    }


def model_components(
    scan: analysis.PreparedScan,
    parameters: dict[str, float],
    beam: analysis.BeamGeometry,
    *,
    fixed_amplitude: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    tau = float(parameters.get("tau_sec", 0.0))
    x, y, vx, vy = scan.coordinates(tau)
    cx = np.full(scan.recorded_time.shape, float(parameters["x0_arcsec"]))
    cy = np.full(scan.recorded_time.shape, float(parameters["y0_arcsec"]))
    if "h_az_arcsec" in parameters:
        cx += float(parameters["h_az_arcsec"]) * np.sign(vx)
        cy += float(parameters["h_el_arcsec"]) * np.sign(vy)
    template = analysis.gaussian_beam(x, y, cx[:, None], cy[:, None], beam)
    data = scan.residual_by_baseline["constant"]
    mask = scan.score_mask
    if fixed_amplitude is None:
        numerator = np.sum(np.where(mask, template * data, 0.0), axis=0)
        denominator = np.sum(np.where(mask, template * template, 0.0), axis=0)
        amplitude = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1.0e-16,
        )
        amplitude = np.maximum(amplitude, 0.0)
    else:
        amplitude = np.asarray(fixed_amplitude, dtype=float)
    source = template * amplitude[None, :]
    residual = data - source
    return {
        "x": x,
        "y": y,
        "velocity_x": vx,
        "velocity_y": vy,
        "template": template,
        "amplitude": amplitude,
        "source": source,
        "residual": residual,
    }


def support_sha256(observation: analysis.PreparedObservation) -> str:
    digest = hashlib.sha256()
    for scan in observation.scans:
        digest.update(np.asarray([scan.scan_row], dtype="<i8").tobytes())
        digest.update(np.asarray(scan.detector_uid, dtype="<i8").tobytes())
        digest.update(np.asarray(scan.score_mask, dtype=np.uint8).tobytes())
        digest.update(np.asarray(scan.ptc_weight, dtype="<f8").tobytes())
    return digest.hexdigest()


def result_is_complete(root: Path, result: dict[str, Any]) -> None:
    analysis.verify_sha256s(root)
    if result.get("schema") != (
        "sci-align-001-lissajous-timestream-observation-result-v1"
    ):
        raise VisualizationError("unsupported or incomplete result schema")
    convergence = result.get("bootstrap", {}).get("timestream_convergence", {})
    if convergence.get("status") != "pass":
        raise VisualizationError("timestream bootstrap did not complete its gate")
    state_path = root / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("status") != "complete":
            raise VisualizationError("instrumented result run_state is not complete")


def scan_direction(vx: float, vy: float) -> str:
    if abs(vx) >= abs(vy):
        return "az_positive" if vx >= 0.0 else "az_negative"
    return "el_positive" if vy >= 0.0 else "el_negative"


def unit_metrics(
    observation: analysis.PreparedObservation,
    best: list[dict[str, np.ndarray]],
    tau0: list[dict[str, np.ndarray]],
) -> tuple[list[FitUnit], list[dict[str, Any]]]:
    units: list[FitUnit] = []
    rows: list[dict[str, Any]] = []
    total_sse = 0.0
    for scan, components in zip(observation.scans, best, strict=True):
        total_sse += float(np.sum(
            scan.ptc_weight[None, :]
            * np.where(scan.score_mask, components["residual"] ** 2, 0.0)
        ))
    for scan, components, zero in zip(
        observation.scans, best, tau0, strict=True
    ):
        for detector_index, (uid, network) in enumerate(zip(
            scan.detector_uid, scan.detector_network, strict=True
        )):
            for segment_index, (start, stop) in enumerate(
                contiguous_true_segments(scan.score_mask[:, detector_index])
            ):
                unit = FitUnit(
                    unit_id=(
                        f"s{scan.scan_row:02d}_uid{int(uid)}_seg{segment_index:02d}"
                    ),
                    scan_row=scan.scan_row,
                    output_scan_index=scan.output_scan_index,
                    detector_index=detector_index,
                    uid=int(uid),
                    network=int(network),
                    segment_index=segment_index,
                    start=start,
                    stop=stop,
                )
                sl = slice(start, stop)
                weight = float(scan.ptc_weight[detector_index])
                residual = components["residual"][sl, detector_index]
                zero_residual = zero["residual"][sl, detector_index]
                times = scan.recorded_time[sl]
                x0 = scan.reference_x[sl, detector_index]
                y0 = scan.reference_y[sl, detector_index]
                radius = np.hypot(
                    x0 - observation.ppt_x_arcsec,
                    y0 - observation.ppt_y_arcsec,
                )
                vx = components["velocity_x"][sl]
                vy = components["velocity_y"][sl]
                source = components["source"][sl, detector_index]
                # A fixed-nuisance Fisher-like curvature proxy in the exact
                # weighted SSE numerator. It is not promoted to an uncertainty.
                # Use the already evaluated source and its trajectory derivative.
                speed = np.hypot(vx, vy)
                sigma = 0.5 * (
                    observation.beam.major_fwhm_arcsec
                    + observation.beam.minor_fwhm_arcsec
                ) * analysis.FWHM_TO_SIGMA
                leverage = float(np.sum(
                    weight * (source * speed / max(sigma, 1.0e-12) / 1000.0) ** 2
                ))
                sse = float(weight * np.sum(residual ** 2))
                zero_sse = float(weight * np.sum(zero_residual ** 2))
                context_start = max(0, start - 25)
                context_stop = min(scan.recorded_time.size, stop + 25)
                nearby_excluded = int(np.count_nonzero(
                    ~scan.score_mask[context_start:context_stop, detector_index]
                ))
                row = {
                    "unit_id": unit.unit_id,
                    "scan_row": unit.scan_row,
                    "output_scan_index": unit.output_scan_index,
                    "uid": unit.uid,
                    "network": unit.network,
                    "segment_index": unit.segment_index,
                    "sample_count": stop - start,
                    "nearby_excluded_sample_count": nearby_excluded,
                    "time_span_ms": 1000.0 * float(times[-1] - times[0]),
                    "closest_nominal_approach_arcsec": float(np.min(radius)),
                    "mean_velocity_x_arcsec_s": float(np.mean(vx)),
                    "mean_velocity_y_arcsec_s": float(np.mean(vy)),
                    "local_velocity_x_arcsec_s": float(vx[np.argmin(radius)]),
                    "local_velocity_y_arcsec_s": float(vy[np.argmin(radius)]),
                    "mean_speed_arcsec_s": float(np.mean(speed)),
                    "direction": scan_direction(float(np.mean(vx)), float(np.mean(vy))),
                    "weight": weight,
                    "weight_min": weight,
                    "weight_median": weight,
                    "weight_max": weight,
                    "residual_rms_native": float(np.sqrt(np.mean(residual ** 2))),
                    "sqrt_weight_scaled_residual_rms": float(
                        np.sqrt(weight * np.mean(residual ** 2))
                    ),
                    "timing_leverage_proxy": leverage,
                    "weighted_sse": sse,
                    "profiled_tau0_weighted_sse": zero_sse,
                    "weighted_sse_improvement_over_profiled_tau0": zero_sse - sse,
                    "objective_numerator_fraction": (
                        sse / total_sse if total_sse > 0.0 else math.nan
                    ),
                    "valid_sample_count": int(np.count_nonzero(
                        scan.valid[sl, detector_index]
                    )),
                    "scored_sample_count": int(np.count_nonzero(
                        scan.score_mask[sl, detector_index]
                    )),
                    "flagged_or_invalid_sample_count": int(np.count_nonzero(
                        ~scan.valid[sl, detector_index]
                    )),
                }
                units.append(unit)
                rows.append(row)
    if not units:
        raise VisualizationError("fit support produced no contiguous units")
    return units, rows


def deterministic_selection(
    units: list[FitUnit],
    rows: list[dict[str, Any]],
    highlights: tuple[int, ...],
    target: int,
) -> tuple[list[FitUnit], dict[str, Any]]:
    by_id = {unit.unit_id: unit for unit in units}
    row_by_id = {row["unit_id"]: row for row in rows}
    chosen: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(unit_id: str, reason: str) -> None:
        if unit_id not in chosen:
            chosen.append(unit_id)
        reasons.setdefault(unit_id, []).append(reason)

    for scan_row in highlights:
        candidates = [row for row in rows if row["scan_row"] == scan_row]
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda row: row["unit_id"])
        leverage = sorted(
            candidates,
            key=lambda row: (-row["timing_leverage_proxy"], row["unit_id"]),
        )
        residual = sorted(
            candidates,
            key=lambda row: (row["sqrt_weight_scaled_residual_rms"], row["unit_id"]),
        )
        add(leverage[0]["unit_id"], f"scan_{scan_row}_maximum_leverage")
        add(
            residual[len(residual) // 2]["unit_id"],
            f"scan_{scan_row}_median_residual",
        )
        high_residual_with_leverage = [
            row for row in reversed(residual)
            if row["timing_leverage_proxy"]
            >= np.median([item["timing_leverage_proxy"] for item in candidates])
        ]
        add(
            high_residual_with_leverage[0]["unit_id"],
            f"scan_{scan_row}_high_residual_with_leverage",
        )

    add(
        max(rows, key=lambda row: (row["timing_leverage_proxy"], row["unit_id"]))[
            "unit_id"
        ],
        "global_maximum_leverage",
    )
    add(
        max(rows, key=lambda row: (
            row["sqrt_weight_scaled_residual_rms"], row["unit_id"]
        ))["unit_id"],
        "global_maximum_scaled_residual",
    )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["scan_row"],
            -row["timing_leverage_proxy"],
            row["sqrt_weight_scaled_residual_rms"],
            row["unit_id"],
        ),
    )
    scan_order = sorted({row["scan_row"] for row in ranked})
    offset = 0
    while len(chosen) < target:
        added = False
        for scan_row in scan_order:
            candidates = [row for row in ranked if row["scan_row"] == scan_row]
            if offset < len(candidates):
                before = len(chosen)
                add(candidates[offset]["unit_id"], "round_robin_scan_leverage")
                added |= len(chosen) > before
                if len(chosen) >= target:
                    break
        if not added:
            break
        offset += 1
    chosen = chosen[:target]
    return [by_id[item] for item in chosen], {
        "schema": "sci-align-001-lissajous-fit-unit-selection-v1",
        "selection_uses_fitted_tau_value": False,
        "target_count": target,
        "highlight_scan_rows": list(highlights),
        "algorithm": (
            "deterministic maximum-leverage, median-residual, and "
            "high-residual-with-at-least-median-leverage examples for each "
            "highlight scan; global extrema; then round-robin scan-stratified "
            "leverage order with unit_id tie breaks"
        ),
        "selected": [
            {"unit_id": item, "reasons": reasons[item], **row_by_id[item]}
            for item in chosen
        ],
    }


def detailed_figures(
    output: Path,
    observation: analysis.PreparedObservation,
    selected: list[FitUnit],
    row_by_id: dict[str, dict[str, Any]],
    best: list[dict[str, np.ndarray]],
    zero: list[dict[str, np.ndarray]],
    fixed_zero: list[dict[str, np.ndarray]],
    primary: dict[str, Any],
    tau0_fit: dict[str, Any],
) -> str:
    name = f"source_crossing_validation_o{observation.obsnum}.pdf"
    plots = output / "plots"
    plots.mkdir()
    with PdfPages(output / name) as pdf:
        for page_index, unit in enumerate(selected):
            scan = observation.scans[unit.scan_row]
            metric = row_by_id[unit.unit_id]
            det = unit.detector_index
            closest = int(np.argmin(np.hypot(
                scan.reference_x[:, det] - primary["parameters"]["x0_arcsec"],
                scan.reference_y[:, det] - primary["parameters"]["y0_arcsec"],
            )))
            t0 = scan.recorded_time[closest]
            context_start = max(0, unit.start - 25)
            context_stop = min(scan.recorded_time.size, unit.stop + 25)
            context = np.arange(context_start, context_stop)
            contributing = scan.score_mask[context, det]
            t_ms = 1000.0 * (scan.recorded_time[context] - t0)
            data = scan.residual_by_baseline["constant"][context, det]
            fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=True)
            grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 0.8])
            ax_data = fig.add_subplot(grid[0, :2])
            ax_resid = fig.add_subplot(grid[1, 0])
            ax_geom = fig.add_subplot(grid[1, 1])
            ax_text = fig.add_subplot(grid[:, 2])
            ax_data.plot(
                t_ms[~contributing], data[~contributing], "o", ms=3,
                mfc="none", mec="0.65", label="nearby excluded",
            )
            ax_data.plot(
                t_ms[contributing], data[contributing], "o", ms=4,
                label="objective support",
            )
            ax_data.plot(t_ms, best[unit.scan_row]["source"][context, det], lw=2,
                         label="best-fit model")
            ax_data.plot(t_ms, zero[unit.scan_row]["source"][context, det], lw=2,
                         label="profiled tau=0")
            ax_data.plot(
                t_ms, fixed_zero[unit.scan_row]["source"][context, det],
                lw=1, ls="--", label="fixed-nuisance tau=0",
            )
            ax_data.axhline(0.0, color="0.5", lw=0.8, label="removed baseline")
            ax_data.set(xlabel="Time from nominal tau=0 closest approach (ms)",
                        ylabel="Baseline-subtracted PTC signal (native units)")
            ax_data.legend(ncol=2, fontsize=8)
            ax_data.set_title(unit.unit_id)
            ax_resid.plot(
                t_ms[contributing],
                best[unit.scan_row]["residual"][context, det][contributing],
                "o-", ms=3, label="best fit",
            )
            ax_resid.plot(
                t_ms[contributing],
                zero[unit.scan_row]["residual"][context, det][contributing],
                "o-", ms=3, label="profiled tau=0",
            )
            ax_resid.axhline(0.0, color="0.5", lw=0.8)
            ax_resid.set(xlabel="Time (ms)", ylabel="Native-unit residual")
            ax_resid.legend(fontsize=8)
            ax_geom.plot(
                scan.reference_x[context, det], scan.reference_y[context, det],
                color="0.55", lw=1.2, label="tau=0 trajectory",
            )
            ax_geom.plot(
                best[unit.scan_row]["x"][context, det],
                best[unit.scan_row]["y"][context, det],
                lw=1.5, label="best-tau trajectory",
            )
            ax_geom.scatter(
                best[unit.scan_row]["x"][context, det][contributing],
                best[unit.scan_row]["y"][context, det][contributing], s=8,
            )
            ax_geom.plot(
                primary["parameters"]["x0_arcsec"],
                primary["parameters"]["y0_arcsec"], "+", ms=12, mew=2,
                label="fitted centroid",
            )
            ax_geom.add_patch(Ellipse(
                (primary["parameters"]["x0_arcsec"],
                 primary["parameters"]["y0_arcsec"]),
                observation.beam.major_fwhm_arcsec,
                observation.beam.minor_fwhm_arcsec,
                angle=np.degrees(observation.beam.angle_rad),
                fill=False, lw=1.2,
            ))
            ax_geom.set_aspect("equal")
            ax_geom.set(xlabel="Az tangent-plane offset (arcsec)",
                        ylabel="El tangent-plane offset (arcsec)")
            ax_geom.legend(fontsize=7)
            speed = metric["mean_speed_arcsec_s"]
            displacement = speed * abs(float(primary["tau_ms"])) / 1000.0
            text = (
                f"ObsNum {observation.obsnum}\n"
                f"scan row {unit.scan_row}; output scan {unit.output_scan_index}\n"
                f"UID {unit.uid}; network {unit.network}\n"
                f"support samples {metric['sample_count']}\n"
                f"velocity ({metric['mean_velocity_x_arcsec_s']:.2f}, "
                f"{metric['mean_velocity_y_arcsec_s']:.2f}) arcsec/s\n"
                f"speed {speed:.2f} arcsec/s\n"
                f"observation tau {primary['tau_ms']:.3f} ms\n"
                f"local |v tau| {displacement:.3f} arcsec\n"
                f"native residual RMS {metric['residual_rms_native']:.4g}\n"
                f"sqrt(weight)-scaled RMS "
                f"{metric['sqrt_weight_scaled_residual_rms']:.4g}\n"
                f"timing leverage proxy {metric['timing_leverage_proxy']:.4g}\n"
                f"weighted SSE improvement vs profiled tau=0 "
                f"{metric['weighted_sse_improvement_over_profiled_tau0']:.4g}\n"
                f"profiled tau=0 objective {tau0_fit['objective']:.6g}\n\n"
                "Weight is a retained detector-scan weight, not an "
                "authenticated per-sample uncertainty."
            )
            ax_text.axis("off")
            ax_text.text(0.0, 1.0, text, va="top", fontsize=9)
            png = plots / f"detail_{page_index:02d}_{unit.unit_id}.png"
            fig.savefig(png, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    return name


def residual_atlas(
    output: Path,
    observation: analysis.PreparedObservation,
    units: list[FitUnit],
    best: list[dict[str, np.ndarray]],
) -> str:
    name = f"all_crossing_residual_atlas_o{observation.obsnum}.pdf"
    grids: list[tuple[int, np.ndarray, list[FitUnit]]] = []
    edges = np.linspace(-1500.0, 1500.0, 122)
    for scan_row, scan in enumerate(observation.scans):
        scan_units = sorted(
            [unit for unit in units if unit.scan_row == scan_row],
            key=lambda item: (item.network, item.uid, item.segment_index),
        )
        raster = np.full((len(scan_units), edges.size - 1), np.nan)
        for row_index, unit in enumerate(scan_units):
            det = unit.detector_index
            center = int(np.argmin(np.hypot(
                scan.reference_x[:, det] - observation.ppt_x_arcsec,
                scan.reference_y[:, det] - observation.ppt_y_arcsec,
            )))
            indices = np.arange(unit.start, unit.stop)
            relative = 1000.0 * (
                scan.recorded_time[indices] - scan.recorded_time[center]
            )
            residual = (
                best[scan_row]["residual"][indices, det]
                * math.sqrt(float(scan.ptc_weight[det]))
            )
            bins = np.digitize(relative, edges) - 1
            for bin_index in np.unique(bins[(bins >= 0) & (bins < raster.shape[1])]):
                selected = bins == bin_index
                raster[row_index, bin_index] = float(np.mean(residual[selected]))
        grids.append((scan_row, raster, scan_units))
    finite = np.concatenate([grid[np.isfinite(grid)] for _, grid, _ in grids])
    limit = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    with PdfPages(output / name) as pdf:
        for scan_row, raster, scan_units in grids:
            fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
            image = ax.imshow(
                raster, origin="lower", aspect="auto", interpolation="none",
                extent=[edges[0], edges[-1], 0, len(scan_units)],
                cmap="coolwarm", vmin=-limit, vmax=limit,
            )
            ax.set(
                xlabel="Time from nominal closest approach (ms)",
                ylabel="Fit units sorted by network, UID, segment",
                title=f"Obs {observation.obsnum} scan row {scan_row}: residual atlas",
            )
            fig.colorbar(image, ax=ax, label="sqrt(weight)-scaled residual")
            pdf.savefig(fig)
            plt.close(fig)
    return name


def residual_footprints(
    output: Path,
    observation: analysis.PreparedObservation,
    best: list[dict[str, np.ndarray]],
    highlights: tuple[int, ...],
) -> str:
    name = f"source_centered_residuals_by_scan_o{observation.obsnum}.pdf"
    edges = np.linspace(-45.0, 45.0, 91)

    def accumulate(scan_rows: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        weighted = np.zeros((90, 90))
        weight = np.zeros((90, 90))
        for scan_row in scan_rows:
            scan = observation.scans[scan_row]
            comp = best[scan_row]
            rr, dd = np.nonzero(scan.score_mask)
            x = comp["x"][rr, dd] - observation.ppt_x_arcsec
            y = comp["y"][rr, dd] - observation.ppt_y_arcsec
            w = scan.ptc_weight[dd]
            residual = comp["residual"][rr, dd]
            weighted += np.histogram2d(y, x, bins=(edges, edges), weights=w * residual)[0]
            weight += np.histogram2d(y, x, bins=(edges, edges), weights=w)[0]
        mean = np.divide(weighted, weight, out=np.full_like(weighted, np.nan), where=weight > 0)
        return mean, weight

    products = [("full observation", *accumulate(range(len(observation.scans))))]
    products.extend([
        (f"scan row {scan_row}", *accumulate([scan_row]))
        for scan_row in range(len(observation.scans))
    ])
    finite = np.concatenate([mean[np.isfinite(mean)] for _, mean, _ in products])
    limit = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    with PdfPages(output / name) as pdf:
        for label, mean, weight in products:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
            im = axes[0].imshow(
                mean, origin="lower", extent=[-45, 45, -45, 45],
                cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="none",
            )
            axes[0].set_title(f"{label}: weighted residual diagnostic")
            log_weight = np.full_like(weight, np.nan)
            positive_weight = weight > 0.0
            log_weight[positive_weight] = np.log10(weight[positive_weight])
            axes[1].imshow(
                log_weight, origin="lower",
                extent=[-45, 45, -45, 45], cmap="viridis", interpolation="none",
            )
            axes[1].set_title("log10 retained weight footprint")
            for ax in axes:
                ax.set(xlabel="Az relative to PPT centroid (arcsec)",
                       ylabel="El relative to PPT centroid (arcsec)")
            fig.colorbar(im, ax=axes[0], label="weighted mean native residual")
            pdf.savefig(fig)
            plt.close(fig)
        available = [item for item in highlights if item < len(observation.scans)]
        if len(available) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
            for ax, scan_row in zip(axes, available[:2], strict=True):
                _, mean, _ = next(item for item in products if item[0] == f"scan row {scan_row}")
                ax.imshow(mean, origin="lower", extent=[-45, 45, -45, 45],
                          cmap="coolwarm", vmin=-limit, vmax=limit,
                          interpolation="none")
                ax.set_title(f"scan row {scan_row}")
                ax.set(xlabel="Az relative to PPT centroid (arcsec)",
                       ylabel="El relative to PPT centroid (arcsec)")
            fig.suptitle("Highlighted scans on identical residual scale")
            pdf.savefig(fig)
            plt.close(fig)
    return name


def model_adequacy(
    output: Path,
    observation: analysis.PreparedObservation,
    rows: list[dict[str, Any]],
    highlights: tuple[int, ...],
) -> str:
    name = f"model_adequacy_and_leverage_o{observation.obsnum}.pdf"
    leverage = np.asarray([row["timing_leverage_proxy"] for row in rows])
    residual = np.asarray([row["sqrt_weight_scaled_residual_rms"] for row in rows])
    contribution = np.asarray([row["objective_numerator_fraction"] for row in rows])
    scan = np.asarray([row["scan_row"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    axes[0, 0].scatter(leverage, residual, c=scan, s=10, cmap="viridis")
    axes[0, 0].set(xlabel="Timing leverage proxy", ylabel="sqrt(weight)-scaled residual RMS")
    axes[0, 1].scatter(leverage, contribution, c=scan, s=10, cmap="viridis")
    axes[0, 1].set(xlabel="Timing leverage proxy", ylabel="Objective numerator fraction")
    scan_leverage = np.asarray([
        np.sum(leverage[scan == row]) for row in sorted(set(scan))
    ])
    order = np.argsort(scan_leverage)[::-1]
    axes[1, 0].step(
        np.arange(1, order.size + 1),
        np.cumsum(scan_leverage[order]) / np.sum(scan_leverage), where="mid",
    )
    axes[1, 0].set(xlabel="Scans ordered by leverage", ylabel="Cumulative leverage fraction")
    ordered = np.sort(leverage)[::-1]
    axes[1, 1].step(
        np.arange(1, ordered.size + 1), np.cumsum(ordered) / np.sum(ordered),
        where="mid",
    )
    axes[1, 1].set(xlabel="Fit units ordered by leverage", ylabel="Cumulative leverage fraction")
    fig.suptitle(
        f"Obs {observation.obsnum} model adequacy; highlighted scans {list(highlights)}"
    )
    fig.savefig(output / name)
    plt.close(fig)
    return name


def scan_profiles(
    output: Path,
    observation: analysis.PreparedObservation,
    primary: dict[str, Any],
    best_components: list[dict[str, np.ndarray]],
    full_profile_path: Path,
    highlights: tuple[int, ...],
) -> tuple[str, list[dict[str, Any]]]:
    name = f"scan_profiled_objectives_o{observation.obsnum}.pdf"
    spec = observation.protocol["models"]["objective_profile_tau_grid_ms"]
    grid = np.linspace(float(spec["minimum"]), float(spec["maximum"]), int(spec["count"]))
    all_rows: list[dict[str, Any]] = []
    with PdfPages(output / name) as pdf:
        full = Table.read(full_profile_path, format="ascii.ecsv")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        full_obj = np.asarray(full["objective"], dtype=float)
        ax.plot(full["tau_ms"], full_obj - np.min(full_obj), label="full observation")
        ax.axvline(0.0, color="0.5", lw=1)
        ax.axvline(primary["tau_ms"], color="tab:red", lw=1, ls="--")
        ax.set(xlabel="tau (ms)", ylabel="Objective minus own minimum",
               title="Full-observation profiled objective")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
        scan_curves = []
        for scan_row, scan in enumerate(observation.scans):
            print(
                f"visualization progress stage=scan_objective_profiles "
                f"scan_row={scan_row} target={len(observation.scans)}",
                file=sys.stderr,
                flush=True,
            )
            curve = []
            for tau in grid:
                parameters = dict(primary["parameters"])
                parameters["tau_sec"] = float(tau) / 1000.0
                components = model_components(
                    scan,
                    parameters,
                    observation.beam,
                    fixed_amplitude=best_components[scan_row]["amplitude"],
                )
                weighted_sse = float(np.sum(
                    scan.ptc_weight[None, :]
                    * np.where(
                        scan.score_mask, components["residual"] ** 2, 0.0
                    )
                ))
                weight_count = float(np.sum(
                    scan.ptc_weight[None, :] * scan.score_mask
                ))
                curve.append({
                    "tau_ms": float(tau),
                    "objective": weighted_sse / weight_count,
                    "x0_arcsec": float(primary["parameters"]["x0_arcsec"]),
                    "y0_arcsec": float(primary["parameters"]["y0_arcsec"]),
                    "optimizer_success": True,
                    "optimizer_attempt_count": 0,
                    "optimizer_converged_count": 0,
                    "scan_row": scan_row,
                    "output_scan_index": scan.output_scan_index,
                    "profile_kind": "fixed_nuisance",
                })
            all_rows.extend(curve)
            scan_curves.append(curve)
        for start in range(0, len(scan_curves), 4):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            for ax, scan_row in zip(axes.ravel(), range(start, min(start + 4, len(scan_curves)))):
                curve = scan_curves[scan_row]
                objective = np.asarray([row["objective"] for row in curve])
                tau = np.asarray([row["tau_ms"] for row in curve])
                ax.plot(tau, objective - np.min(objective))
                ax.axvline(0.0, color="0.5", lw=1)
                ax.axvline(primary["tau_ms"], color="tab:red", lw=1, ls="--")
                ax.set_title(
                    f"scan row {scan_row} ({curve[0]['profile_kind']})"
                )
                ax.set(xlabel="tau (ms)", ylabel="Objective - minimum")
            for ax in axes.ravel()[len(range(start, min(start + 4, len(scan_curves)))):]:
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        available = [item for item in highlights if item < len(scan_curves)]
        if len(available) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for scan_row in available[:2]:
                curve = scan_curves[scan_row]
                objective = np.asarray([row["objective"] for row in curve])
                ax.plot(
                    grid, objective - np.min(objective),
                    label=f"scan row {scan_row} ({curve[0]['profile_kind']})",
                )
            ax.axvline(0.0, color="0.5", lw=1)
            ax.axvline(primary["tau_ms"], color="tab:red", lw=1, ls="--")
            ax.set(xlabel="tau (ms)", ylabel="Objective - own minimum",
                   title="Highlighted scan objective profiles")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
    return name, all_rows


def map_context(
    output: Path,
    observation: analysis.PreparedObservation,
    map_path: Path,
    primary: dict[str, Any],
) -> str:
    name = f"standard_map_context_o{observation.obsnum}.pdf"
    with fits.open(map_path) as hdul:
        hdu = hdul["signal_I"]
        image = np.asarray(hdu.data, dtype=float).squeeze()
        header = hdu.header
    ny, nx = image.shape
    x = (np.arange(nx) + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"])
    y = (np.arange(ny) + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"])
    xx, yy = np.meshgrid(x, y)
    template = analysis.gaussian_beam(
        xx, yy,
        np.asarray(float(primary["parameters"]["x0_arcsec"])),
        np.asarray(float(primary["parameters"]["y0_arcsec"])),
        observation.beam,
    )
    core = np.hypot(xx - observation.ppt_x_arcsec, yy - observation.ppt_y_arcsec) <= 20.0
    finite = core & np.isfinite(image)
    amplitude = float(np.sum(image[finite] * template[finite]) / np.sum(template[finite] ** 2))
    compact = amplitude * template
    residual = image - compact
    finite_image = image[np.isfinite(image)]
    low, high = np.percentile(finite_image, [2, 99.7])
    residual_limit = np.percentile(np.abs(residual[np.isfinite(residual)]), 99.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].imshow(image, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   vmin=low, vmax=high, cmap="viridis")
    axes[0].set_title("Authenticated standard a1100 map")
    axes[1].imshow(compact, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis")
    axes[1].set_title("PPT-beam compact model (context)")
    axes[2].imshow(residual, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                   vmin=-residual_limit, vmax=residual_limit, cmap="coolwarm")
    axes[2].set_title("Map minus compact context model")
    for ax in axes:
        ax.plot(primary["parameters"]["x0_arcsec"], primary["parameters"]["y0_arcsec"], "+", ms=10, mew=2)
        ax.set(xlabel="Az offset (arcsec)", ylabel="El offset (arcsec)", xlim=(-80, 80), ylim=(-80, 80))
    fig.suptitle("Map context is not the timestream timing objective")
    fig.savefig(output / name)
    plt.close(fig)
    return name


def code_reference(function: Any) -> dict[str, Any]:
    lines, start = inspect.getsourcelines(function)
    return {
        "path": str(Path(inspect.getsourcefile(function) or "").resolve()),
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "function": function.__name__,
    }


def write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    def progress(stage: str) -> None:
        print(f"visualization progress stage={stage}", file=sys.stderr, flush=True)

    progress("authenticate_inputs")
    repo = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    selection_path = args.selection.resolve()
    result_root = args.result_root.resolve()
    map_path = args.map_fits.resolve()
    output = args.output.resolve()
    if output.exists():
        raise VisualizationError(f"output already exists: {output}")
    protocol = analysis.load_protocol(protocol_path)
    selection = analysis.load_selection(
        selection_path, protocol["input_authority"]["selection_manifest_sha256"]
    )
    row = analysis.selected_row(selection, args.obsnum)
    result = json.loads((result_root / "result.json").read_text())
    result_is_complete(result_root, result)
    if int(result["obsnum"]) != args.obsnum:
        raise VisualizationError("result observation identity mismatch")
    if (
        result["input"]["ptc_sha256"] != row["ptc_sha256"]
        or result["input"]["ppt_sha256"] != row["ppt_sha256"]
        or result["input"]["protocol_sha256"] != sha256_file(protocol_path)
    ):
        raise VisualizationError("result input identities do not match frozen inputs")
    if not map_path.is_file():
        raise VisualizationError(f"standard map is missing: {map_path}")
    output.mkdir(parents=True)
    progress("prepare_observation_and_exact_objective")
    observation = analysis.prepare_observation(row, protocol)
    coordinate_gate = analysis.coordinate_reconstruction_gate(observation)
    primary = result["point_model_results"]["lag"]
    primary_vector = fit_vector(primary)
    reconstructed = analysis.observation_objective(
        primary_vector, observation, "lag", "fixed", "constant"
    )
    tolerance = 1.0e-10 * max(1.0, abs(float(primary["objective"])))
    if abs(reconstructed - float(primary["objective"])) > tolerance:
        raise VisualizationError("reconstructed best-fit objective changed")
    tau0_fit = profile_fixed_tau(observation, 0.0, primary)
    if not tau0_fit["optimizer_success"]:
        raise VisualizationError("profiled tau=0 comparator did not converge")
    zero_parameters = dict(primary["parameters"])
    zero_parameters.update({
        "x0_arcsec": float(tau0_fit["x0_arcsec"]),
        "y0_arcsec": float(tau0_fit["y0_arcsec"]),
        "tau_sec": 0.0,
    })
    fixed_zero_parameters = dict(primary["parameters"])
    fixed_zero_parameters["tau_sec"] = 0.0
    best = [model_components(scan, primary["parameters"], observation.beam) for scan in observation.scans]
    zero = [model_components(scan, zero_parameters, observation.beam) for scan in observation.scans]
    fixed_zero = [
        model_components(
            scan, fixed_zero_parameters, observation.beam,
            fixed_amplitude=best[index]["amplitude"],
        )
        for index, scan in enumerate(observation.scans)
    ]
    units, metrics = unit_metrics(observation, best, zero)
    selected, selection_doc = deterministic_selection(
        units, metrics, tuple(args.highlight_scan_row), args.selected_count
    )
    Table(rows=metrics).write(output / "crossing_metrics.ecsv", format="ascii.ecsv")
    write_json(output / "selected_crossings.json", selection_doc)
    support_digest = support_sha256(observation)
    source_path = Path(analysis.__file__).resolve()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    identity = {
        "schema": "sci-align-001-lissajous-fit-visualization-identity-v1",
        "obsnum": args.obsnum,
        "repository": str(repo),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "result_root": str(result_root),
        "result_sha256": sha256_file(result_root / "result.json"),
        "result_sha256s_sha256": sha256_file(result_root / "SHA256SUMS"),
        "ptc_path": row["ptc_path"],
        "ptc_sha256": row["ptc_sha256"],
        "ppt_path": row["ppt_path"],
        "ppt_sha256": row["ppt_sha256"],
        "standard_map_path": str(map_path),
        "standard_map_sha256": sha256_file(map_path),
        "fit_support_sha256": support_digest,
        "profiled_tau0_support_sha256": support_digest,
        "coordinate_gate": coordinate_gate,
        "best_objective_recorded": float(primary["objective"]),
        "best_objective_reconstructed": reconstructed,
        "objective_tolerance": tolerance,
        "profiled_tau0_objective": float(tau0_fit["objective"]),
        "fixed_nuisance_tau0_objective": analysis.observation_objective(
            fit_vector({**primary, "parameters": fixed_zero_parameters}),
            observation, "lag", "fixed", "constant",
        ),
        "analysis_source_path": str(source_path),
        "analysis_source_sha256": sha256_file(source_path),
    }
    write_json(output / "execution_identity.json", identity)
    audit = {
        "schema": "sci-align-001-lissajous-fit-support-audit-v1",
        "fit_consumes_all_detector_samples": False,
        "support_is_fixed_before_fit": True,
        "support_definition": {
            "array": "a1100 only",
            "apt_flag": 0,
            "sample_flag": 0,
            "finite_positive_detector_scan_weight": True,
            "common_tau_edge_trim_seconds": float(protocol["common_support"]["maximum_abs_tau_sec"]),
            "source_scoring_radius_arcsec": float(protocol["eligibility"]["source_scoring_radius_arcsec"]),
            "baseline_training_min_radius_arcsec": float(protocol["eligibility"]["baseline_training_min_radius_arcsec"]),
            "minimum_scored_samples_per_detector_scan": int(protocol["eligibility"]["minimum_scored_samples_per_detector_scan"]),
            "minimum_baseline_samples_per_detector_scan": int(protocol["eligibility"]["minimum_baseline_samples_per_detector_scan"]),
        },
        "preprocessing": (
            "constant detector-scan baseline fitted only on fixed off-source "
            "samples and subtracted before the source objective; no per-unit normalization"
        ),
        "upstream_processing_boundary": (
            "the delivered PTC signal, flags, weights, and telescope-coordinate "
            "trajectory are treated as retained inputs; filtering, cleaning, "
            "calibration, and metadata association performed before this product "
            "are neither reconstructed nor independently authenticated here"
        ),
        "weight_semantics": (
            "one retained PTC weight per detector and scan; the available "
            "evidence does not authenticate it as a per-sample inverse variance, "
            "so residuals are labeled sqrt(weight)-scaled rather than standardized"
        ),
        "parameters": {
            "global": ["x0_arcsec", "y0_arcsec", "tau_sec"],
            "profiled_per_detector_scan": ["nonnegative source amplitude"],
            "preprocessed_per_detector_scan": ["constant off-source baseline"],
        },
        "positive_tau_convention": "detector coordinates are evaluated at recorded_time + tau",
        "scan_objective_curve_kind": (
            "fixed-nuisance: observation-level x0/y0 and the best-fit "
            "detector-scan amplitudes are held fixed while the exact trajectory "
            "and source model are re-evaluated at each tau; no scan-specific "
            "timing estimator is introduced"
        ),
        "visualization_unit": (
            "one contiguous True block in the fixed score_mask for one detector "
            "and one complete PTC scan row"
        ),
        "counts": {
            "scan_count": len(observation.scans),
            "fit_unit_count": len(units),
            "scored_value_count": observation.scored_value_count,
        },
        "code_references": {
            "prepare_observation": code_reference(analysis.prepare_observation),
            "objective": code_reference(analysis.observation_objective),
            "profiled_scan_objective": code_reference(analysis.scan_profiled_objective),
        },
    }
    write_json(output / "FIT_SUPPORT_AUDIT.json", audit)
    (output / "FIT_SUPPORT_AUDIT.md").write_text(
        f"# Fit support audit for ObsNum {args.obsnum}\n\n"
        "The fit does not consume every sample. It uses a support mask frozen "
        "at recorded time before tau is fit. The mask requires a1100 APT flag "
        "0, sample flag 0, finite positive detector-scan weight, the frozen "
        f"+/-{1000 * protocol['common_support']['maximum_abs_tau_sec']:.0f}-ms "
        "edge trim, and the frozen source radius. Constant detector-scan "
        "baselines are learned only from fixed off-source samples and removed.\n\n"
        "The objective weight is one retained PTC weight per detector and scan. "
        "It is not authenticated here as a per-sample inverse variance; figures "
        "therefore use native or sqrt(weight)-scaled residuals, never sigma units.\n\n"
        "The lag model has global x0, y0, and tau. A nonnegative source amplitude "
        "is profiled for every detector-scan. A visualization unit is a contiguous "
        "True block of the fixed score mask for one detector in one scan. "
        "Individual-scan objective curves are explicitly fixed-nuisance curves; "
        "they do not define independent scan timing fits. The delivered PTC "
        "arrays already include upstream processing that this renderer does not "
        "reconstruct or independently authenticate.\n"
    )
    row_by_id = {row["unit_id"]: row for row in metrics}
    progress("detailed_fit_units")
    detailed_figures(
        output, observation, selected, row_by_id, best, zero, fixed_zero,
        primary, tau0_fit,
    )
    progress("residual_atlas")
    residual_atlas(output, observation, units, best)
    progress("source_centered_residual_footprints")
    residual_footprints(
        output, observation, best, tuple(args.highlight_scan_row)
    )
    progress("model_adequacy_and_leverage")
    model_adequacy(
        output, observation, metrics, tuple(args.highlight_scan_row)
    )
    progress("scan_objective_profiles")
    profile_name, scan_profile_rows = scan_profiles(
        output, observation, primary, best,
        result_root / "objective_profile.ecsv", tuple(args.highlight_scan_row),
    )
    Table(rows=scan_profile_rows).write(
        output / "scan_profiled_objectives.ecsv", format="ascii.ecsv"
    )
    progress("standard_map_context")
    map_context(output, observation, map_path, primary)
    shutil.copy2(Path(__file__), output / "make_visualization.py")
    leverage = np.asarray([row["timing_leverage_proxy"] for row in metrics])
    scan_leverage = {
        str(scan_row): float(np.sum([
            row["timing_leverage_proxy"] for row in metrics
            if row["scan_row"] == scan_row
        ]))
        for scan_row in range(len(observation.scans))
    }
    largest_scan = max(scan_leverage, key=scan_leverage.get)
    top_fraction = float(np.sum(np.sort(leverage)[::-1][:max(1, len(leverage) // 20)]) / np.sum(leverage))
    report = (
        f"# Lissajous fit visualization report: ObsNum {args.obsnum}\n\n"
        f"The exact fit uses {observation.scored_value_count} values in "
        f"{len(units)} contiguous detector-scan support units across "
        f"{len(observation.scans)} scans. Support is fixed before fitting.\n\n"
        f"The recorded best objective ({primary['objective']:.12g}) was "
        f"reconstructed as {reconstructed:.12g}. The profiled tau=0 objective "
        f"is {tau0_fit['objective']:.12g}; this is an objective change, not a "
        "chi-square claim.\n\n"
        f"Scan row {largest_scan} has the largest summed timing-leverage proxy. "
        f"The top 5% of fit units carry {100 * top_fraction:.1f}% of that proxy. "
        "Residual structure and source-model adequacy must be judged from the "
        "authenticated pages; this renderer does not prescribe a correction.\n\n"
        "The standard map is context only and uses a different objective. "
        "All footprint maps are diagnostic weighted bins, not Citlali maps.\n"
        "Individual-scan objective curves hold the observation-level centroid "
        "and best-fit detector-scan amplitudes fixed and are labeled "
        "fixed-nuisance; only the completed full-observation curve is a profiled "
        "objective.\n"
    )
    (output / "REPORT.md").write_text(report)
    progress("checksums")
    write_checksums(output)
    analysis.verify_sha256s(output)
    print(
        f"visualization complete: obs={args.obsnum} units={len(units)} "
        f"selected={len(selected)} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--result-root", type=Path, required=True)
    result.add_argument("--map-fits", type=Path, required=True)
    result.add_argument("--obsnum", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--highlight-scan-row", type=int, action="append", default=[])
    result.add_argument("--selected-count", type=int, default=16, choices=range(12, 21))
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.highlight_scan_row:
        args.highlight_scan_row = [6, 7]
    try:
        run(args)
    except (
        VisualizationError,
        analysis.ContractError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
