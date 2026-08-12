#!/usr/bin/env python3
"""Render event-faithful review evidence for a Lissajous event-fit gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402
import render_sci_align_001_lissajous_fit_gate_source_review as legacy  # noqa: E402
import sci_align_001_lissajous_crossings as crossings  # noqa: E402
import visualize_sci_align_001_lissajous_fit as visualization  # noqa: E402


class EventReviewError(RuntimeError):
    """Event-fit review input or output violates its contract."""


def _scan_and_detector(
    observation: analysis.PreparedObservation,
    event: Any,
) -> tuple[analysis.PreparedScan, int]:
    matches = [
        scan for scan in observation.scans
        if int(scan.scan_row) == int(event["scan_row"])
    ]
    if len(matches) != 1:
        raise EventReviewError("event scan identity is not unique")
    scan = matches[0]
    detector = np.flatnonzero(
        np.asarray(scan.detector_uid, dtype=int) == int(event["uid"])
    )
    if detector.size != 1:
        raise EventReviewError("event detector identity is not unique")
    return scan, int(detector[0])


def event_metrics_and_samples(
    observation: analysis.PreparedObservation,
    events: Table,
    best: list[dict[str, np.ndarray]],
    zero: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    scan_position = {
        int(scan.scan_row): index for index, scan in enumerate(observation.scans)
    }
    rows: list[dict[str, Any]] = []
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "along_arcsec", "data", "model", "weight", "direction_index"
        )
    }
    direction_names = (
        "az_positive", "az_negative", "el_positive", "el_negative"
    )
    direction_index = {name: index for index, name in enumerate(direction_names)}
    retained_detector_scans = {
        (int(scan.scan_row), int(uid))
        for scan in observation.scans
        for uid in scan.detector_uid
    }
    for event in events[np.asarray(events["accepted"], dtype=bool)]:
        if (int(event["scan_row"]), int(event["uid"])) not in retained_detector_scans:
            continue
        scan, detector = _scan_and_detector(observation, event)
        position = scan_position[int(scan.scan_row)]
        comp = best[position]
        zero_comp = zero[position]
        start = int(event["fit_window_start"])
        stop = int(event["fit_window_stop_exclusive"])
        indices = np.arange(start, stop)
        contributing = scan.score_mask[indices, detector]
        selected = indices[contributing]
        if selected.size == 0:
            continue
        closest = int(event["closest_sample"])
        vx = float(event["velocity_x_arcsec_per_sec"])
        vy = float(event["velocity_y_arcsec_per_sec"])
        speed = math.hypot(vx, vy)
        ux = vx / speed
        uy = vy / speed
        cx = float(comp["x"][closest, detector])
        cy = float(comp["y"][closest, detector])
        source_x = np.asarray(comp["x"][selected, detector], dtype=float)
        source_y = np.asarray(comp["y"][selected, detector], dtype=float)
        along = (source_x - cx) * ux + (source_y - cy) * uy
        amplitude = float(comp["amplitude"][detector])
        detector_weight = float(scan.ptc_weight[detector])
        data = scan.residual_by_baseline["constant"][selected, detector]
        model = comp["source"][selected, detector]
        residual = comp["residual"][selected, detector]
        zero_residual = zero_comp["residual"][selected, detector]
        if data.size >= 2 and np.std(data) > 0.0 and np.std(model) > 0.0:
            correlation = float(np.corrcoef(data, model)[0, 1])
        else:
            correlation = math.nan
        sigma = 0.5 * (
            observation.beam.major_fwhm_arcsec
            + observation.beam.minor_fwhm_arcsec
        ) * analysis.FWHM_TO_SIGMA
        leverage = float(np.sum(
            detector_weight
            * (model * speed / max(sigma, 1.0e-12) / 1000.0) ** 2
        ))
        direction = visualization.scan_direction(vx, vy)
        if math.isfinite(amplitude) and amplitude > 0.0:
            weight = np.full(
                data.size, detector_weight * amplitude * amplitude
            )
            normalized_data = data / amplitude
            normalized_model = model / amplitude
        else:
            # Keep zero-amplitude events in the metric/accounting table while
            # giving them exactly zero influence on amplitude-normalized
            # stacks. Event membership itself remains geometry-only.
            weight = np.zeros(data.size)
            normalized_data = np.zeros(data.size)
            normalized_model = np.zeros(data.size)
        parts["along_arcsec"].append(along)
        parts["data"].append(normalized_data)
        parts["model"].append(normalized_model)
        parts["weight"].append(weight)
        parts["direction_index"].append(np.full(
            data.size, direction_index[direction], dtype=np.int64
        ))
        rows.append({
            "event_id": str(event["event_id"]),
            "scan_row": int(scan.scan_row),
            "output_scan_index": int(scan.output_scan_index),
            "uid": int(event["uid"]),
            "network": int(event["network"]),
            "event_index": int(event["detector_event_index"]),
            "half_power_start": int(event["half_power_start"]),
            "half_power_stop_exclusive": int(
                event["half_power_stop_exclusive"]
            ),
            "fit_window_start": start,
            "fit_window_stop_exclusive": stop,
            "scored_sample_count": int(selected.size),
            "closest_elliptical_fwhm_radius": float(
                event["closest_elliptical_fwhm_radius"]
            ),
            "crossing_angle_deg": float(event["directed_crossing_angle_deg"]),
            "velocity_x_arcsec_per_sec": vx,
            "velocity_y_arcsec_per_sec": vy,
            "speed_arcsec_per_sec": speed,
            "direction": direction,
            "amplitude_native": amplitude,
            "detector_scan_weight": detector_weight,
            "local_data_model_correlation": correlation,
            "residual_rms_native": float(np.sqrt(np.mean(residual**2))),
            "sqrt_weight_scaled_residual_rms": float(np.sqrt(
                detector_weight * np.mean(residual**2)
            )),
            "weighted_sse": float(detector_weight * np.sum(residual**2)),
            "weighted_sse_improvement_over_tau0": float(
                detector_weight
                * (np.sum(zero_residual**2) - np.sum(residual**2))
            ),
            "timing_leverage_proxy": leverage,
        })
    if not rows:
        raise EventReviewError("event review retained no accepted event")
    samples = {name: np.concatenate(values) for name, values in parts.items()}
    samples["direction_names"] = np.asarray(direction_names)
    return rows, samples


def deterministic_event_selection(
    rows: list[dict[str, Any]], target: int
) -> tuple[list[str], dict[str, Any]]:
    chosen: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        event_id = row["event_id"]
        if event_id not in chosen:
            chosen.append(event_id)
        reasons.setdefault(event_id, []).append(reason)

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scan_row"], row["uid"]), []).append(row)
    multi_event_groups = sorted(
        (identity, members)
        for identity, members in grouped.items()
        if len(members) > 1
    )
    if multi_event_groups:
        _, members = multi_event_groups[0]
        for row in sorted(members, key=lambda item: item["event_index"]):
            add(row, "first_multi_event_detector_scan_distinct_passage")

    add(max(rows, key=lambda row: row["timing_leverage_proxy"]),
        "global_maximum_timing_leverage")
    add(max(rows, key=lambda row: row["sqrt_weight_scaled_residual_rms"]),
        "global_maximum_scaled_residual")
    finite_correlation = [
        row for row in rows if math.isfinite(row["local_data_model_correlation"])
    ]
    add(min(finite_correlation, key=lambda row: row["local_data_model_correlation"]),
        "global_minimum_data_model_correlation")
    add(max(finite_correlation, key=lambda row: row["local_data_model_correlation"]),
        "global_maximum_data_model_correlation")
    scans = sorted({row["scan_row"] for row in rows})
    for scan_row in scans:
        candidates = [row for row in rows if row["scan_row"] == scan_row]
        add(max(candidates, key=lambda row: row["timing_leverage_proxy"]),
            f"scan_{scan_row}_maximum_timing_leverage")
        if len(chosen) >= target:
            break
    ranked = sorted(
        rows,
        key=lambda row: (
            row["scan_row"], -row["timing_leverage_proxy"], row["event_id"]
        ),
    )
    offset = 0
    maximum_candidates_per_scan = max(
        sum(row["scan_row"] == scan_row for row in ranked)
        for scan_row in scans
    )
    while len(chosen) < target:
        added = False
        for scan_row in scans:
            candidates = [row for row in ranked if row["scan_row"] == scan_row]
            if offset < len(candidates):
                before = len(chosen)
                add(candidates[offset], "round_robin_scan_leverage")
                added |= len(chosen) > before
                if len(chosen) >= target:
                    break
        offset += 1
        if not added and offset >= maximum_candidates_per_scan:
            break
    chosen = chosen[:target]
    by_id = {row["event_id"]: row for row in rows}
    return chosen, {
        "schema": "sci-align-001-lissajous-event-review-selection-v1",
        "event_inclusion_uses_fitted_parameter": False,
        "target_count": target,
        "selected": [
            {**by_id[event_id], "selection_reasons": reasons[event_id]}
            for event_id in chosen
        ],
    }


def write_stack_pdf(
    output: Path,
    observation: analysis.PreparedObservation,
    samples: dict[str, np.ndarray],
    metrics: list[dict[str, Any]],
    census: dict[str, Any],
) -> str:
    name = f"event_source_aligned_stacks_o{observation.obsnum}.pdf"
    half_width = 2.0 * observation.beam.major_fwhm_arcsec
    edges = np.linspace(-half_width, half_width, 81)
    groups = [("all accepted events", np.ones(samples["weight"].size, bool))]
    groups.extend([
        (str(name), samples["direction_index"] == index)
        for index, name in enumerate(samples["direction_names"])
    ])
    with PdfPages(output / name) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for ax, (label, selected) in zip(axes.flat[:5], groups, strict=True):
            stack = legacy.binned_weighted_stack(
                samples["along_arcsec"][selected], samples["data"][selected],
                samples["model"][selected], samples["weight"][selected], edges,
            )
            ax.plot(stack["center_arcsec"], stack["data_mean"], "o-", ms=3,
                    label="event-aligned data")
            ax.plot(stack["center_arcsec"], stack["model_mean"], lw=2,
                    label="fitted model")
            ax.axvline(0.0, color="0.5", lw=0.8)
            ax.set(title=label.replace("_", " "),
                   xlabel="Along-event trajectory offset (arcsec)",
                   ylabel="Amplitude-normalized signal",
                   xlim=(-half_width, half_width))
            ax.legend(fontsize=8)
        angles = np.asarray([row["crossing_angle_deg"] for row in metrics])
        axes.flat[5].hist(angles, bins=np.linspace(0.0, 360.0, 25))
        axes.flat[5].set(xlabel="Directed crossing angle (deg)",
                         ylabel="Accepted events")
        fig.suptitle(
            f"Obs {observation.obsnum}: {census['accepted_event_count']} accepted "
            f"of {census['geometric_event_count']} geometric events"
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        speed = np.asarray([row["speed_arcsec_per_sec"] for row in metrics])
        correlation = np.asarray([
            row["local_data_model_correlation"] for row in metrics
        ])
        residual = np.asarray([
            row["sqrt_weight_scaled_residual_rms"] for row in metrics
        ])
        leverage = np.asarray([row["timing_leverage_proxy"] for row in metrics])
        axes[0, 0].hist(speed, bins=40)
        axes[0, 0].set(xlabel="Local projected speed (arcsec/s)", ylabel="Events")
        axes[0, 1].hist(correlation[np.isfinite(correlation)], bins=40)
        axes[0, 1].set(xlabel="Local data/model correlation", ylabel="Events")
        axes[1, 0].scatter(leverage, residual, s=8, alpha=0.5)
        axes[1, 0].set(xlabel="Timing leverage proxy",
                       ylabel="sqrt(weight)-scaled residual RMS")
        labels = list(census["disposition_counts"])
        axes[1, 1].barh(
            labels, [census["disposition_counts"][label] for label in labels]
        )
        axes[1, 1].set(xlabel="Geometric event count", title="Event disposition")
        fig.suptitle("Event support and model-adequacy census")
        pdf.savefig(fig)
        plt.close(fig)
    return name


def write_detail_pdf(
    output: Path,
    observation: analysis.PreparedObservation,
    events: Table,
    selected_ids: list[str],
    metrics: list[dict[str, Any]],
    best: list[dict[str, np.ndarray]],
    zero: list[dict[str, np.ndarray]],
    primary: dict[str, Any],
    selection_document: dict[str, Any],
) -> str:
    name = f"event_crossing_validation_o{observation.obsnum}.pdf"
    by_event = {str(row["event_id"]): row for row in events}
    by_metric = {row["event_id"]: row for row in metrics}
    selection_reasons = {
        row["event_id"]: row["selection_reasons"]
        for row in selection_document["selected"]
    }
    scan_position = {
        int(scan.scan_row): index for index, scan in enumerate(observation.scans)
    }
    with PdfPages(output / name) as pdf:
        for event_id in selected_ids:
            event = by_event[event_id]
            metric = by_metric[event_id]
            scan, detector = _scan_and_detector(observation, event)
            position = scan_position[int(scan.scan_row)]
            start = int(event["fit_window_start"])
            stop = int(event["fit_window_stop_exclusive"])
            context_start = max(0, start - 25)
            context_stop = min(scan.recorded_time.size, stop + 25)
            context = np.arange(context_start, context_stop)
            support = scan.score_mask[context, detector]
            closest = int(event["closest_sample"])
            time_ms = 1000.0 * (
                scan.recorded_time[context] - scan.recorded_time[closest]
            )
            data = scan.residual_by_baseline["constant"][context, detector]
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            grid = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.0, 0.8])
            ax_data = fig.add_subplot(grid[0, :2])
            ax_residual = fig.add_subplot(grid[1, 0])
            ax_geometry = fig.add_subplot(grid[1, 1])
            ax_text = fig.add_subplot(grid[:, 2])
            ax_data.plot(time_ms[~support], data[~support], "o", ms=3,
                         mfc="none", mec="0.65", label="nearby excluded")
            ax_data.plot(time_ms[support], data[support], "o", ms=4,
                         label="event objective support")
            ax_data.plot(time_ms, best[position]["source"][context, detector],
                         lw=2, label="best-fit model")
            ax_data.plot(time_ms, zero[position]["source"][context, detector],
                         lw=1.5, label="tau=0 model")
            ax_data.axhline(0.0, color="0.5", lw=0.8)
            ax_data.set(xlabel="Time from tau=0 event closest approach (ms)",
                        ylabel="Baseline-subtracted PTC signal (native units)",
                        title=event_id)
            ax_data.legend(fontsize=8, ncol=2)
            ax_residual.plot(
                time_ms[support],
                best[position]["residual"][context, detector][support],
                "o-", ms=3, label="best fit",
            )
            ax_residual.plot(
                time_ms[support],
                zero[position]["residual"][context, detector][support],
                "o-", ms=3, label="tau=0",
            )
            ax_residual.axhline(0.0, color="0.5", lw=0.8)
            ax_residual.set(xlabel="Time (ms)", ylabel="Native residual")
            ax_residual.legend(fontsize=8)
            ax_geometry.plot(
                scan.reference_x[context, detector],
                scan.reference_y[context, detector], color="0.55",
                label="tau=0 trajectory",
            )
            ax_geometry.plot(
                best[position]["x"][context, detector],
                best[position]["y"][context, detector], label="best-tau trajectory",
            )
            ax_geometry.scatter(
                scan.reference_x[context[support], detector],
                scan.reference_y[context[support], detector], s=10,
                label="event support",
            )
            ax_geometry.plot(
                observation.ppt_x_arcsec, observation.ppt_y_arcsec,
                "x", ms=9, mew=2, label="PPT event center",
            )
            ax_geometry.plot(
                primary["parameters"]["x0_arcsec"],
                primary["parameters"]["y0_arcsec"],
                "+", ms=10, mew=2, label="fitted center",
            )
            ax_geometry.add_patch(Ellipse(
                (observation.ppt_x_arcsec, observation.ppt_y_arcsec),
                observation.beam.major_fwhm_arcsec,
                observation.beam.minor_fwhm_arcsec,
                angle=math.degrees(observation.beam.angle_rad),
                fill=False, lw=1.2,
            ))
            ax_geometry.set_aspect("equal")
            ax_geometry.set(xlabel="Az tangent offset (arcsec)",
                            ylabel="El tangent offset (arcsec)")
            ax_geometry.legend(fontsize=6)
            ax_text.axis("off")
            ax_text.text(0.0, 1.0, (
                f"ObsNum {observation.obsnum}\n"
                f"scan row {metric['scan_row']}; output scan "
                f"{metric['output_scan_index']}\n"
                f"UID {metric['uid']}; network {metric['network']}\n"
                f"event index {metric['event_index']}\n"
                f"support samples {metric['scored_sample_count']}\n"
                f"closest radius {metric['closest_elliptical_fwhm_radius']:.3f} FWHM\n"
                f"velocity ({metric['velocity_x_arcsec_per_sec']:.2f}, "
                f"{metric['velocity_y_arcsec_per_sec']:.2f}) arcsec/s\n"
                f"speed {metric['speed_arcsec_per_sec']:.2f} arcsec/s\n"
                f"angle {metric['crossing_angle_deg']:.1f} deg\n"
                f"fit tau {primary['tau_ms']:+.3f} ms\n"
                f"data/model correlation "
                f"{metric['local_data_model_correlation']:.3f}\n"
                f"scaled residual RMS "
                f"{metric['sqrt_weight_scaled_residual_rms']:.4g}\n"
                f"timing leverage {metric['timing_leverage_proxy']:.4g}\n"
                f"SSE improvement vs tau=0 "
                f"{metric['weighted_sse_improvement_over_tau0']:.4g}\n\n"
                "Selected because:\n"
                + "\n".join(
                    f"- {reason}" for reason in selection_reasons[event_id]
                )
                + "\n\n"
                "Event inclusion was frozen at tau=0 around the PPT center."
            ), va="top", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    return name


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise EventReviewError(f"event review is already complete: {output}")
    fit_root = args.fit_gate_root.resolve()
    analysis.verify_sha256s(fit_root, "FIT_GATE_SHA256SUMS")
    gate = json.loads((fit_root / "fit_gate.json").read_text())
    if gate.get("schema") != "sci-align-001-lissajous-event-fit-gate-v1":
        raise EventReviewError("unsupported event fit-gate schema")
    protocol = analysis.load_protocol(args.protocol.resolve())
    crossing_protocol = crossings.load_crossing_protocol(
        args.crossing_protocol.resolve()
    )
    crossings.authenticate_base_protocol(protocol, crossing_protocol)
    selection = analysis.load_selection(
        args.selection.resolve(),
        protocol["input_authority"]["selection_manifest_sha256"],
    )
    row = analysis.selected_row(selection, args.obsnum)
    base = analysis.prepare_observation(row, protocol)
    events = crossings.catalog_crossing_events(base, crossing_protocol)
    observation, support = crossings.restrict_to_crossing_support(
        base, events, crossing_protocol
    )
    if analysis.observation_support_summary(observation) != gate["support"]:
        raise EventReviewError("event fit support identity changed")
    if crossings.event_census(events, support) != gate["crossing_support"]["census"]:
        raise EventReviewError("event census identity changed")
    primary = gate["point_model_results"]["lag"]
    values = analysis.fit_to_optimizer_vector(primary, "lag", "fixed")
    reconstructed = analysis.observation_objective(
        values, observation, "lag", "fixed", "constant"
    )
    tolerance = 1.0e-10 * max(1.0, abs(float(primary["objective"])))
    if abs(reconstructed - float(primary["objective"])) > tolerance:
        raise EventReviewError("event-fit objective reconstruction changed")
    best = [
        visualization.model_components(scan, primary["parameters"], observation.beam)
        for scan in observation.scans
    ]
    zero_parameters = dict(primary["parameters"])
    zero_parameters["tau_sec"] = 0.0
    zero = [
        visualization.model_components(scan, zero_parameters, observation.beam)
        for scan in observation.scans
    ]
    metrics, samples = event_metrics_and_samples(observation, events, best, zero)
    selected, selection_doc = deterministic_event_selection(
        metrics, args.selected_count
    )
    output.mkdir(parents=True, exist_ok=True)
    Table(rows=metrics).write(
        output / "event_metrics.ecsv", format="ascii.ecsv", overwrite=True
    )
    events.write(
        output / "event_catalog.ecsv", format="ascii.ecsv", overwrite=True
    )
    analysis.write_json(output / "selected_events.json", selection_doc)
    census = crossings.event_census(events, support)
    stack_pdf = write_stack_pdf(output, observation, samples, metrics, census)
    detail_pdf = write_detail_pdf(
        output, observation, events, selected, metrics, best, zero, primary,
        selection_doc,
    )
    profile_rows = legacy.fixed_nuisance_tau_profile(observation, primary)
    Table(rows=profile_rows).write(
        output / "fixed_nuisance_tau_profile.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    tau = np.asarray([item["tau_ms"] for item in profile_rows])
    objective = np.asarray([item["objective"] for item in profile_rows])
    ax.plot(tau, objective - np.min(objective), "o-")
    ax.axvline(0.0, color="0.5", lw=1)
    ax.axvline(float(primary["tau_ms"]), color="tab:red", ls="--",
               label="fitted lag")
    ax.set(xlabel="tau (ms)", ylabel="Objective - grid minimum",
           title="Event-support fixed-centroid lag objective")
    ax.legend()
    profile_pdf = f"event_tau_profile_o{observation.obsnum}.pdf"
    fig.savefig(output / profile_pdf)
    plt.close(fig)
    manifest = {
        "schema": "sci-align-001-lissajous-event-fit-review-v1",
        "obsnum": args.obsnum,
        "fit_gate_sha256": analysis.sha256_file(fit_root / "fit_gate.json"),
        "fit_gate_sha256s_sha256": analysis.sha256_file(
            fit_root / "FIT_GATE_SHA256SUMS"
        ),
        "protocol_sha256": analysis.sha256_file(args.protocol.resolve()),
        "crossing_protocol_sha256": analysis.sha256_file(
            args.crossing_protocol.resolve()
        ),
        "selection_sha256": analysis.sha256_file(args.selection.resolve()),
        "lag_objective_recorded": float(primary["objective"]),
        "lag_objective_reconstructed": reconstructed,
        "objective_tolerance": tolerance,
        "crossing_census": census,
        "selected_individual_event_count": len(selected),
        "event_inclusion_uses_fitted_parameter": False,
        "timestamp_detector_averaging_used": False,
        "numerical_fit_changed": False,
        "disposition": "owner_review_before_66_observation_campaign_release",
        "outputs": [stack_pdf, detail_pdf, profile_pdf],
    }
    analysis.write_json(output / "manifest.json", manifest)
    analysis.write_checksums(output, [
        "event_catalog.ecsv", "event_metrics.ecsv",
        "fixed_nuisance_tau_profile.ecsv", "manifest.json",
        "selected_events.json", stack_pdf, detail_pdf, profile_pdf,
    ])
    analysis.verify_sha256s(output)
    print(
        f"event review complete: obs={args.obsnum} "
        f"events={census['accepted_event_count']} selected={len(selected)} "
        f"output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--crossing-protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--fit-gate-root", type=Path, required=True)
    result.add_argument("--obsnum", type=int, required=True)
    result.add_argument("--selected-count", type=int, default=16)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (
        EventReviewError,
        analysis.ContractError,
        crossings.CrossingContractError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
