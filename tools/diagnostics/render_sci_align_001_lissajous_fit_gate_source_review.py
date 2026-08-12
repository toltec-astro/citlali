#!/usr/bin/env python3
"""Render detector-aligned source evidence for a completed fit gate.

This supplementary renderer never refits a timing model.  It authenticates a
completed fit gate, reconstructs its exact lag objective, then aligns each
eligible detector crossing by signed distance along its own local trajectory.
It exists because averaging focal-plane detectors at a common timestamp
smears crossings that occur at different times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402
import visualize_sci_align_001_lissajous_fit as visualization  # noqa: E402


class ReviewError(RuntimeError):
    """A source-aligned review input or output violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_checksums(root: Path) -> None:
    names = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text("".join(
        f"{sha256_file(root / name)}  {name}\n" for name in names
    ))


def beam_normalized_radius(
    dx: np.ndarray, dy: np.ndarray, beam: analysis.BeamGeometry,
) -> np.ndarray:
    ct = math.cos(beam.angle_rad)
    st = math.sin(beam.angle_rad)
    major = ct * dx + st * dy
    minor = -st * dx + ct * dy
    return np.sqrt(
        (major / beam.major_fwhm_arcsec) ** 2
        + (minor / beam.minor_fwhm_arcsec) ** 2
    )


def source_aligned_samples(
    observation: analysis.PreparedObservation,
    components: list[dict[str, np.ndarray]],
    parameters: dict[str, float],
    maximum_closest_fwhm_radius: float = 0.5,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Return per-sample profiles aligned by each detector's local crossing."""
    sample_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "along_arcsec", "cross_arcsec", "normalized_data",
            "normalized_model", "weight", "direction_index", "scan_row",
            "uid", "network",
        )
    }
    events: list[dict[str, Any]] = []
    direction_names = (
        "az_positive", "az_negative", "el_positive", "el_negative"
    )
    direction_to_index = {
        name: index for index, name in enumerate(direction_names)
    }
    center_x = float(parameters["x0_arcsec"])
    center_y = float(parameters["y0_arcsec"])
    for scan, comp in zip(observation.scans, components, strict=True):
        for detector_index, (uid, network) in enumerate(zip(
            scan.detector_uid, scan.detector_network, strict=True
        )):
            for segment_index, (start, stop) in enumerate(
                visualization.contiguous_true_segments(
                    scan.score_mask[:, detector_index]
                )
            ):
                indices = np.arange(start, stop)
                dx = comp["x"][indices, detector_index] - center_x
                dy = comp["y"][indices, detector_index] - center_y
                normalized_radius = beam_normalized_radius(
                    dx, dy, observation.beam
                )
                closest_local = int(np.argmin(normalized_radius))
                closest = float(normalized_radius[closest_local])
                if closest > maximum_closest_fwhm_radius:
                    continue
                amplitude = float(comp["amplitude"][detector_index])
                detector_weight = float(scan.ptc_weight[detector_index])
                if not (
                    math.isfinite(amplitude) and amplitude > 0.0
                    and math.isfinite(detector_weight) and detector_weight > 0.0
                ):
                    continue
                vx = float(comp["velocity_x"][indices[closest_local]])
                vy = float(comp["velocity_y"][indices[closest_local]])
                speed = math.hypot(vx, vy)
                if not math.isfinite(speed) or speed <= 0.0:
                    continue
                ux = vx / speed
                uy = vy / speed
                along = dx * ux + dy * uy
                cross = -dx * uy + dy * ux
                direction = visualization.scan_direction(vx, vy)
                data = (
                    scan.residual_by_baseline["constant"][indices, detector_index]
                    / amplitude
                )
                model = comp["source"][indices, detector_index] / amplitude
                weight = np.full(
                    indices.size, detector_weight * amplitude * amplitude
                )
                sample_parts["along_arcsec"].append(along)
                sample_parts["cross_arcsec"].append(cross)
                sample_parts["normalized_data"].append(data)
                sample_parts["normalized_model"].append(model)
                sample_parts["weight"].append(weight)
                sample_parts["direction_index"].append(np.full(
                    indices.size, direction_to_index[direction], dtype=np.int64
                ))
                sample_parts["scan_row"].append(np.full(
                    indices.size, scan.scan_row, dtype=np.int64
                ))
                sample_parts["uid"].append(np.full(
                    indices.size, int(uid), dtype=np.int64
                ))
                sample_parts["network"].append(np.full(
                    indices.size, int(network), dtype=np.int64
                ))
                events.append({
                    "event_id": (
                        f"s{scan.scan_row:02d}_uid{int(uid)}_seg{segment_index:02d}"
                    ),
                    "scan_row": scan.scan_row,
                    "output_scan_index": scan.output_scan_index,
                    "uid": int(uid),
                    "network": int(network),
                    "segment_index": segment_index,
                    "sample_count": int(indices.size),
                    "closest_fwhm_radius": closest,
                    "closest_distance_arcsec": float(
                        math.hypot(dx[closest_local], dy[closest_local])
                    ),
                    "crossing_angle_deg": float(
                        math.degrees(math.atan2(vy, vx)) % 360.0
                    ),
                    "direction": direction,
                    "speed_arcsec_s": speed,
                    "amplitude_native": amplitude,
                    "detector_scan_weight": detector_weight,
                })
    if not events:
        raise ReviewError("no detector trajectory crosses within 0.5 FWHM")
    samples = {
        name: np.concatenate(parts) for name, parts in sample_parts.items()
    }
    samples["direction_names"] = np.asarray(direction_names)
    return samples, events


def binned_weighted_stack(
    coordinate: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    weight: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    coordinate = np.asarray(coordinate, dtype=float)
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    weight = np.asarray(weight, dtype=float)
    valid = (
        np.isfinite(coordinate) & np.isfinite(data) & np.isfinite(model)
        & np.isfinite(weight) & (weight > 0.0)
    )
    denominator = np.histogram(coordinate[valid], edges, weights=weight[valid])[0]
    count = np.histogram(coordinate[valid], edges)[0]
    data_sum = np.histogram(
        coordinate[valid], edges, weights=weight[valid] * data[valid]
    )[0]
    model_sum = np.histogram(
        coordinate[valid], edges, weights=weight[valid] * model[valid]
    )[0]
    data_mean = np.divide(
        data_sum, denominator, out=np.full(denominator.shape, np.nan),
        where=denominator > 0.0,
    )
    model_mean = np.divide(
        model_sum, denominator, out=np.full(denominator.shape, np.nan),
        where=denominator > 0.0,
    )
    return {
        "center_arcsec": 0.5 * (edges[:-1] + edges[1:]),
        "data_mean": data_mean,
        "model_mean": model_mean,
        "weight_sum": denominator,
        "sample_count": count,
    }


def write_aligned_stack_pdf(
    output: Path,
    observation: analysis.PreparedObservation,
    samples: dict[str, np.ndarray],
    events: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    name = f"source_aligned_crossing_stacks_o{observation.obsnum}.pdf"
    half_width = 2.0 * observation.beam.major_fwhm_arcsec
    edges = np.linspace(-half_width, half_width, 81)
    groups = [("all crossings", np.ones(samples["weight"].size, dtype=bool))]
    groups.extend([
        (str(direction), samples["direction_index"] == index)
        for index, direction in enumerate(samples["direction_names"])
    ])
    rows: list[dict[str, Any]] = []
    with PdfPages(output / name) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), constrained_layout=True)
        for ax, (label, selected) in zip(axes.flat[:5], groups, strict=True):
            stack = binned_weighted_stack(
                samples["along_arcsec"][selected],
                samples["normalized_data"][selected],
                samples["normalized_model"][selected],
                samples["weight"][selected], edges,
            )
            if np.any(selected):
                ax.plot(
                    stack["center_arcsec"], stack["data_mean"], "o-", ms=3,
                    label="source-aligned data",
                )
                ax.plot(
                    stack["center_arcsec"], stack["model_mean"], lw=2,
                    label="exact fitted model",
                )
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5, 0.5, "No qualifying crossings", ha="center",
                    va="center", transform=ax.transAxes,
                )
            ax.axvline(0.0, color="0.5", lw=0.8)
            ax.set(
                title=label.replace("_", " "),
                xlabel="Signed along-trajectory offset (arcsec)",
                ylabel="Per-detector amplitude-normalized signal",
                xlim=(-half_width, half_width),
            )
            for index in range(stack["center_arcsec"].size):
                rows.append({
                    "group": label,
                    "bin_center_arcsec": float(stack["center_arcsec"][index]),
                    "data_mean": float(stack["data_mean"][index]),
                    "model_mean": float(stack["model_mean"][index]),
                    "weight_sum": float(stack["weight_sum"][index]),
                    "sample_count": int(stack["sample_count"][index]),
                })
        angles = np.asarray([row["crossing_angle_deg"] for row in events])
        axes.flat[5].hist(angles, bins=np.linspace(0.0, 360.0, 25))
        axes.flat[5].set(
            xlabel="Local crossing angle (deg)", ylabel="Crossing count",
            title=f"{len(events)} detector-scan crossing events",
        )
        fig.suptitle(
            f"Obs {observation.obsnum}: detector-aligned source crossings; "
            "no common-timestamp detector averaging"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return name, rows


def fixed_nuisance_tau_profile(
    observation: analysis.PreparedObservation,
    primary: dict[str, Any],
) -> list[dict[str, float]]:
    spec = observation.protocol["models"]["objective_profile_tau_grid_ms"]
    grid = np.linspace(
        float(spec["minimum"]), float(spec["maximum"]), int(spec["count"])
    )
    parameters = dict(primary["parameters"])
    rows = []
    for tau_ms in grid:
        parameters["tau_sec"] = float(tau_ms) / 1000.0
        values = analysis.fit_to_optimizer_vector(
            {**primary, "parameters": parameters}, "lag", "fixed"
        )
        rows.append({
            "tau_ms": float(tau_ms),
            "objective": float(analysis.observation_objective(
                values, observation, "lag", "fixed", "constant"
            )),
            "profile_kind": "fixed_centroid_profiled_detector_amplitudes",
        })
    return rows


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise ReviewError(f"output already exists: {output}")
    fit_root = args.fit_gate_root.resolve()
    analysis.verify_sha256s(fit_root, "FIT_GATE_SHA256SUMS")
    gate = json.loads((fit_root / "fit_gate.json").read_text())
    if gate.get("schema") != "sci-align-001-lissajous-fit-gate-v1":
        raise ReviewError("unsupported fit-gate schema")
    if int(gate["obsnum"]) != args.obsnum:
        raise ReviewError("fit-gate observation identity changed")
    protocol = analysis.load_protocol(args.protocol.resolve())
    selection = analysis.load_selection(
        args.selection.resolve(),
        protocol["input_authority"]["selection_manifest_sha256"],
    )
    row = analysis.selected_row(selection, args.obsnum)
    observation = analysis.prepare_observation(row, protocol)
    coordinate_gate = analysis.coordinate_reconstruction_gate(observation)
    if coordinate_gate != gate["coordinate_gate"]:
        raise ReviewError("fit-gate coordinate identity changed")
    if analysis.observation_support_summary(observation) != gate["support"]:
        raise ReviewError("fit-gate support identity changed")
    expected_input = analysis.fit_gate_input_identity(
        args, row, gate["input"]["map_result"]
    )
    if expected_input != gate["input"]:
        raise ReviewError("fit-gate input or numerical identity changed")
    primary = gate["point_model_results"]["lag"]
    primary_values = analysis.fit_to_optimizer_vector(primary, "lag", "fixed")
    reconstructed = analysis.observation_objective(
        primary_values, observation, "lag", "fixed", "constant"
    )
    tolerance = 1.0e-10 * max(1.0, abs(float(primary["objective"])))
    if abs(reconstructed - float(primary["objective"])) > tolerance:
        raise ReviewError("fit-gate lag objective reconstruction changed")
    constant = gate["point_model_results"]["constant"]
    zero_parameters = dict(constant["parameters"])
    zero_parameters["tau_sec"] = 0.0
    fixed_zero_parameters = dict(primary["parameters"])
    fixed_zero_parameters["tau_sec"] = 0.0
    best = [
        visualization.model_components(
            scan, primary["parameters"], observation.beam
        ) for scan in observation.scans
    ]
    zero = [
        visualization.model_components(scan, zero_parameters, observation.beam)
        for scan in observation.scans
    ]
    fixed_zero = [
        visualization.model_components(
            scan, fixed_zero_parameters, observation.beam,
            fixed_amplitude=best[index]["amplitude"],
        ) for index, scan in enumerate(observation.scans)
    ]
    samples, events = source_aligned_samples(
        observation, best, primary["parameters"]
    )
    accepted_event_ids = {row["event_id"] for row in events}
    all_units, all_metrics = visualization.unit_metrics(observation, best, zero)
    units = [unit for unit in all_units if unit.unit_id in accepted_event_ids]
    metrics = [
        row for row in all_metrics if row["unit_id"] in accepted_event_ids
    ]
    if len(units) < args.selected_count:
        raise ReviewError(
            "too few qualifying source crossings for individual review: "
            f"{len(units)} < {args.selected_count}"
        )
    selected, selection_doc = visualization.deterministic_selection(
        units, metrics, (), args.selected_count
    )
    selection_doc["candidate_restriction"] = (
        "closest fitted trajectory point is within 0.5 elliptical FWHM"
    )
    selection_doc["selection_uses_fitted_tau_value"] = True
    output.mkdir(parents=True)
    Table(rows=metrics).write(
        output / "crossing_metrics.ecsv", format="ascii.ecsv"
    )
    write_json(output / "selected_crossings.json", selection_doc)
    visualization.detailed_figures(
        output, observation, selected,
        {row["unit_id"]: row for row in metrics},
        best, zero, fixed_zero, primary, constant,
    )
    Table(rows=events).write(
        output / "source_crossing_events.ecsv", format="ascii.ecsv"
    )
    _, stack_rows = write_aligned_stack_pdf(
        output, observation, samples, events
    )
    Table(rows=stack_rows).write(
        output / "source_aligned_stacks.ecsv", format="ascii.ecsv"
    )
    profile_rows = fixed_nuisance_tau_profile(observation, primary)
    Table(rows=profile_rows).write(
        output / "fixed_nuisance_tau_profile.ecsv", format="ascii.ecsv"
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    tau = np.asarray([row["tau_ms"] for row in profile_rows])
    objective = np.asarray([row["objective"] for row in profile_rows])
    ax.plot(tau, objective - np.min(objective), "o-")
    ax.axvline(0.0, color="0.5", lw=1)
    ax.axvline(float(primary["tau_ms"]), color="tab:red", ls="--",
               label="fitted lag")
    ax.set(xlabel="tau (ms)", ylabel="Objective - grid minimum",
           title="Fixed-centroid lag objective; detector amplitudes profiled")
    ax.legend()
    fig.savefig(output / f"fixed_nuisance_tau_profile_o{args.obsnum}.pdf")
    plt.close(fig)
    renderer_path = Path(__file__).resolve()
    repo = renderer_path.parents[2]
    manifest = {
        "schema": "sci-align-001-lissajous-fit-gate-source-review-v1",
        "obsnum": args.obsnum,
        "fit_gate_path": str(fit_root),
        "fit_gate_sha256": sha256_file(fit_root / "fit_gate.json"),
        "fit_gate_sha256s_sha256": sha256_file(
            fit_root / "FIT_GATE_SHA256SUMS"
        ),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "selection_sha256": sha256_file(args.selection.resolve()),
        "ptc_sha256": row["ptc_sha256"],
        "ppt_sha256": row["ppt_sha256"],
        "numerical_implementation_sha256": sha256_file(
            Path(analysis.__file__).resolve()
        ),
        "renderer_sha256": sha256_file(renderer_path),
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "coordinate_gate": coordinate_gate,
        "lag_objective_recorded": float(primary["objective"]),
        "lag_objective_reconstructed": reconstructed,
        "objective_tolerance": tolerance,
        "crossing_definition": (
            "one contiguous fixed-score-mask detector/scan segment whose "
            "closest fitted trajectory point is within 0.5 elliptical FWHM"
        ),
        "alignment_definition": (
            "signed tangent-plane projection on the local detector trajectory "
            "velocity at closest approach to the fitted source centroid"
        ),
        "stack_weight": "retained detector-scan weight times fitted amplitude squared",
        "crossing_event_count": len(events),
        "selected_individual_crossing_count": len(selected),
        "timestamp_detector_averaging_used": False,
        "numerical_fit_changed": False,
        "existing_fit_gate_modified": False,
        "disposition": "supplementary_owner_review_before_remaining_campaign",
    }
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        f"# Source-aligned fit-gate review for ObsNum {args.obsnum}\n\n"
        "This supplementary artifact does not refit the observation. It "
        "authenticates and exactly reconstructs the completed lag objective. "
        "Unlike the original fit-gate page, it never averages detectors at a "
        "common timestamp. Detector/scan crossings are aligned by signed "
        "distance along their own local source trajectories before stacking.\n"
    )
    write_checksums(output)
    analysis.verify_sha256s(output)
    print(
        f"source review complete: obs={args.obsnum} crossings={len(events)} "
        f"selected={len(selected)} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--fit-gate-root", type=Path, required=True)
    result.add_argument("--obsnum", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--selected-count", type=int, default=16,
                        choices=range(12, 21))
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (
        ReviewError, analysis.ContractError, OSError, ValueError, KeyError
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
