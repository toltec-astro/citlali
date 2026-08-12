#!/usr/bin/env python3
"""Render event-centroid validation evidence for one Lissajous pointing."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
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
import sci_align_001_lissajous_crossings as crossings  # noqa: E402
import sci_align_001_lissajous_event_centroids as centroids  # noqa: E402


class EventCentroidReviewError(RuntimeError):
    """The event-centroid review violates its evidence contract."""


def _event_lookup(events: Table) -> dict[str, Any]:
    return {str(row["event_id"]): row for row in events}


def _centroid_lookup(rows: Table) -> dict[str, Any]:
    return {str(row["event_id"]): row for row in rows}


def _direction(row: Any) -> str:
    vx = float(row["velocity_x_arcsec_per_sec"])
    vy = float(row["velocity_y_arcsec_per_sec"])
    if abs(vx) >= abs(vy):
        return "az_positive" if vx >= 0.0 else "az_negative"
    return "el_positive" if vy >= 0.0 else "el_negative"


def deterministic_selection(
    rows: Table, prediction: np.ndarray, count: int,
) -> dict[str, Any]:
    records = []
    for row, expected in zip(rows, prediction, strict=True):
        measured = float(row["peak_shift_arcsec"])
        records.append({
            "event_id": str(row["event_id"]),
            "quality_disposition": str(row["quality_disposition"]),
            "quality_qualified": bool(row["quality_qualified"]),
            "peak_correlation": float(row["peak_correlation"]),
            "absolute_global_residual_arcsec": (
                abs(measured - float(expected))
                if math.isfinite(measured) else math.inf
            ),
        })
    chosen: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(record: dict[str, Any], reason: str) -> None:
        event_id = record["event_id"]
        if event_id not in chosen:
            chosen.append(event_id)
        reasons.setdefault(event_id, []).append(reason)

    qualified = [row for row in records if row["quality_qualified"]]
    finite = [row for row in records if math.isfinite(row["peak_correlation"])]
    if qualified:
        add(max(qualified, key=lambda row: row["peak_correlation"]),
            "highest_qualified_correlation")
        add(min(qualified, key=lambda row: row["peak_correlation"]),
            "lowest_qualified_correlation")
        add(max(qualified, key=lambda row: row["absolute_global_residual_arcsec"]),
            "largest_qualified_global_model_residual")
    if finite:
        add(min(finite, key=lambda row: row["peak_correlation"]),
            "lowest_finite_correlation")
    for disposition in sorted({row["quality_disposition"] for row in records}):
        candidates = [
            row for row in records if row["quality_disposition"] == disposition
        ]
        add(max(candidates, key=lambda row: row["absolute_global_residual_arcsec"]),
            f"representative_{disposition}")
    multi: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row, source in zip(records, rows, strict=True):
        multi.setdefault((int(source["scan_row"]), int(source["uid"])), []).append(row)
    groups = [members for members in multi.values() if len(members) > 1]
    if groups:
        for row in groups[0]:
            add(row, "first_multi_event_detector_scan")
    for row in sorted(
        records,
        key=lambda item: (
            not item["quality_qualified"],
            -item["absolute_global_residual_arcsec"],
            item["event_id"],
        ),
    ):
        add(row, "residual_rank_fill")
        if len(chosen) >= count:
            break
    chosen = chosen[:count]
    return {
        "schema": "sci-align-001-event-centroid-review-selection-v1",
        "selection_uses_fitted_tau": False,
        "selection_uses_model_residual_for_diagnostic_coverage": True,
        "selected": [
            {
                **next(row for row in records if row["event_id"] == event_id),
                "selection_reasons": reasons[event_id],
            }
            for event_id in chosen
        ],
    }


def _find_scan(observation: analysis.PreparedObservation, scan_row: int):
    matches = [scan for scan in observation.scans if scan.scan_row == scan_row]
    if len(matches) != 1:
        raise EventCentroidReviewError("scan identity is not unique")
    return matches[0]


def write_detail_pdf(
    output: Path,
    observation: analysis.PreparedObservation,
    events: Table,
    rows: Table,
    fit: dict[str, Any],
    selection: dict[str, Any],
) -> str:
    name = f"event_centroid_validation_o{observation.obsnum}.pdf"
    event_by_id = _event_lookup(events)
    row_by_id = _centroid_lookup(rows)
    prediction = centroids.centroid_prediction(rows, fit)
    predicted_by_id = {
        str(row["event_id"]): float(value)
        for row, value in zip(rows, prediction, strict=True)
    }
    with PdfPages(output / name) as pdf:
        for selected in selection["selected"]:
            event_id = selected["event_id"]
            event = event_by_id[event_id]
            row = row_by_id[event_id]
            samples = centroids.event_profile_samples(observation, event, row)
            scan = _find_scan(observation, int(row["scan_row"]))
            detector = int(samples["detector_index"])
            context = np.asarray(samples["context_indices"], dtype=int)
            score = np.asarray(samples["score_mask"], dtype=bool)
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 0.9])
            ax_data = fig.add_subplot(grid[0, :2])
            ax_residual = fig.add_subplot(grid[1, 0])
            ax_geometry = fig.add_subplot(grid[1, 1])
            ax_text = fig.add_subplot(grid[:, 2])
            time_ms = np.asarray(samples["time_ms"], dtype=float)
            data = np.asarray(samples["data"], dtype=float)
            model = np.asarray(samples["model"], dtype=float)
            ax_data.plot(time_ms[~score], data[~score], "o", ms=3, mfc="none",
                         mec="0.65", label="nearby excluded")
            ax_data.plot(time_ms[score], data[score], "o", ms=4,
                         label="fixed event support")
            if np.all(np.isfinite(model)):
                ax_data.plot(time_ms, model, lw=2, label="local matched filter")
            ax_data.axhline(float(row["profiled_intercept_native"]),
                            color="0.5", lw=0.8)
            ax_data.set(
                title=event_id,
                xlabel="Time from nominal closest approach (ms)",
                ylabel="Baseline-subtracted PTC signal (native units)",
            )
            ax_data.legend(fontsize=8)
            if np.all(np.isfinite(model)):
                ax_residual.plot(time_ms[score], (data - model)[score], "o-")
            ax_residual.axhline(0.0, color="0.5", lw=0.8)
            ax_residual.set(xlabel="Time (ms)", ylabel="Local residual")
            x = np.asarray(scan.reference_x[context, detector], dtype=float)
            y = np.asarray(scan.reference_y[context, detector], dtype=float)
            ax_geometry.plot(x, y, color="0.55", label="tau=0 trajectory")
            ax_geometry.scatter(x[score], y[score], s=10, label="event support")
            ax_geometry.plot(observation.ppt_x_arcsec,
                             observation.ppt_y_arcsec, "x", ms=9, mew=2,
                             label="PPT center")
            ax_geometry.plot(float(samples["center_x_arcsec"]),
                             float(samples["center_y_arcsec"]), "o", ms=7,
                             mfc="none", mew=2, label="local centroid")
            predicted = predicted_by_id[event_id]
            ax_geometry.plot(
                observation.ppt_x_arcsec + predicted * float(row["unit_x"]),
                observation.ppt_y_arcsec + predicted * float(row["unit_y"]),
                "+", ms=10, mew=2, label="global lag prediction",
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
            measured = float(row["peak_shift_arcsec"])
            disposition_text = textwrap.fill(
                str(row["quality_disposition"]).replace("_", " "), width=33
            )
            reason_text = "\n".join(
                "- " + textwrap.fill(
                    reason.replace("_", " "), width=31,
                    subsequent_indent="  ",
                )
                for reason in selected["selection_reasons"]
            )
            ax_text.text(0.0, 1.0, (
                f"ObsNum {observation.obsnum}\n"
                f"scan row {int(row['scan_row'])}; UID {int(row['uid'])}\n"
                f"network {int(row['network'])}; {_direction(row)}\n"
                f"samples {int(row['scored_sample_count'])}\n"
                f"speed {float(row['speed_arcsec_per_sec']):.2f} arcsec/s\n"
                f"effective FWHM {float(row['effective_fwhm_arcsec']):.2f} arcsec\n\n"
                f"correlation {float(row['peak_correlation']):.3f}\n"
                f"amplitude {float(row['profiled_amplitude_native']):.4g}\n"
                f"local shift {measured:+.3f} arcsec\n"
                f"global prediction {predicted:+.3f} arcsec\n"
                f"residual {measured-predicted:+.3f} arcsec\n"
                f"grid coordinate {float(row['peak_shift_effective_fwhm']):+.3f} FWHM\n\n"
                f"quality:\n{disposition_text}\n"
                f"qualified: {bool(row['quality_qualified'])}\n\n"
                "Selected because:\n" + reason_text + "\n\n"
                "Quality uses local compact-source\n"
                "morphology only; global timing is\n"
                "not a quality gate."
            ), va="top", fontsize=8.5)
            pdf.savefig(fig)
            plt.close(fig)
    return name


def _stack(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.digitize(x, edges) - 1
    center = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(center.size, np.nan)
    count = np.zeros(center.size, dtype=int)
    for bin_index in range(center.size):
        selected = (index == bin_index) & np.isfinite(y)
        count[bin_index] = int(np.count_nonzero(selected))
        if count[bin_index]:
            mean[bin_index] = float(np.mean(y[selected]))
    return center, mean, count


def stack_samples(
    observation: analysis.PreparedObservation,
    events: Table,
    rows: Table,
    fit: dict[str, Any],
) -> dict[str, np.ndarray]:
    events_by_id = _event_lookup(events)
    prediction = centroids.centroid_prediction(rows, fit)
    parts: dict[str, list] = {
        "local_x": [], "global_x": [], "signal": [], "qualified": [],
        "direction": [], "event_id": [],
    }
    for row, predicted in zip(rows, prediction, strict=True):
        amplitude = float(row["profiled_amplitude_native"])
        intercept = float(row["profiled_intercept_native"])
        shift = float(row["peak_shift_arcsec"])
        if not all(map(math.isfinite, (amplitude, intercept, shift))) or amplitude <= 0:
            continue
        event = events_by_id[str(row["event_id"])]
        samples = centroids.event_profile_samples(observation, event, row)
        score = np.asarray(samples["score_mask"], dtype=bool)
        along = np.asarray(samples["along_arcsec"], dtype=float)[score]
        signal = (
            np.asarray(samples["data"], dtype=float)[score] - intercept
        ) / amplitude
        parts["local_x"].append(along - shift)
        parts["global_x"].append(along - float(predicted))
        parts["signal"].append(signal)
        parts["qualified"].append(np.full(signal.size, bool(row["quality_qualified"])))
        parts["direction"].append(np.full(signal.size, _direction(row), dtype=object))
        parts["event_id"].append(np.full(signal.size, str(row["event_id"]), dtype=object))
    return {name: np.concatenate(values) for name, values in parts.items()}


def write_summary_pdf(
    output: Path,
    observation: analysis.PreparedObservation,
    events: Table,
    rows: Table,
    gate: dict[str, Any],
) -> str:
    name = f"event_centroid_stacks_o{observation.obsnum}.pdf"
    fit = gate["point_model_results"]["lag"]
    samples = stack_samples(observation, events, rows, fit)
    half = 2.0 * observation.beam.major_fwhm_arcsec
    edges = np.linspace(-half, half, 81)
    prediction = centroids.centroid_prediction(rows, fit)
    measured = np.asarray(rows["peak_shift_arcsec"], dtype=float)
    qualified = np.asarray(rows["quality_qualified"], dtype=bool)
    with PdfPages(output / name) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        groups = [
            ("qualified: locally aligned", samples["qualified"], "local_x"),
            ("rejected: locally aligned", ~samples["qualified"], "local_x"),
            ("qualified: global lag aligned", samples["qualified"], "global_x"),
        ]
        for ax, (label, selected, x_name) in zip(axes.flat[:3], groups, strict=True):
            center, mean, count = _stack(
                samples[x_name][selected], samples["signal"][selected], edges
            )
            ax.plot(center, mean, "o-", ms=3)
            ax.axvline(0.0, color="0.5", lw=0.8)
            ax.set(title=f"{label}\n{len(set(samples['event_id'][selected]))} events",
                   xlabel="Along-trajectory offset (arcsec)",
                   ylabel="Amplitude-normalized signal", xlim=(-half, half))
        ax = axes.flat[3]
        for direction in (
            "az_positive", "az_negative", "el_positive", "el_negative"
        ):
            selected = samples["qualified"] & (samples["direction"] == direction)
            center, mean, count = _stack(
                samples["global_x"][selected], samples["signal"][selected], edges
            )
            ax.plot(
                center, mean, "o-", ms=2.5,
                label=(f"{direction.replace('_', ' ')} "
                       f"({len(set(samples['event_id'][selected]))})"),
            )
        ax.axvline(0.0, color="0.5", lw=0.8)
        ax.set(title="Qualified global-aligned direction stacks",
               xlabel="Offset from global lag prediction (arcsec)",
               ylabel="Amplitude-normalized signal", xlim=(-half, half))
        ax.legend(fontsize=7)
        fig.suptitle(
            f"Obs {observation.obsnum}: local morphology and global registration"
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        axes[0, 0].scatter(prediction[qualified], measured[qualified], s=8, alpha=0.5)
        limit = np.nanmax(np.abs(np.r_[prediction[qualified], measured[qualified]]))
        axes[0, 0].plot([-limit, limit], [-limit, limit], color="0.5")
        axes[0, 0].set(xlabel="Global lag prediction (arcsec)",
                       ylabel="Measured local centroid (arcsec)")
        axes[0, 1].scatter(
            np.asarray(rows["speed_arcsec_per_sec"], float)[qualified],
            (measured - prediction)[qualified], s=8, alpha=0.5,
        )
        axes[0, 1].axhline(0.0, color="0.5")
        axes[0, 1].set(xlabel="Crossing speed (arcsec/s)",
                       ylabel="Lag-model residual (arcsec)")
        corr = np.asarray(rows["peak_correlation"], float)
        axes[1, 0].hist(corr[np.isfinite(corr)], bins=40)
        axes[1, 0].axvline(gate["primary_minimum_correlation"],
                           color="tab:red", ls="--")
        axes[1, 0].set(xlabel="Local matched-filter correlation", ylabel="Events")
        dispositions = gate["centroid_census"]["quality_disposition_counts"]
        axes[1, 1].barh(list(dispositions), list(dispositions.values()))
        axes[1, 1].set(xlabel="Assessed events", title="Primary disposition")
        census = gate["crossing_census"]
        cc = gate["centroid_census"]
        fig.suptitle(
            f"Geometric {census['geometric_event_count']}; complete "
            f"{census['accepted_event_count']}; assessed {cc['assessed_event_count']}; "
            f"qualified {cc['primary_qualified_event_count']}; rejected "
            f"{cc['primary_rejected_event_count']}"
        )
        pdf.savefig(fig)
        plt.close(fig)
    return name


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise EventCentroidReviewError(f"review output exists: {output}")
    fit_root = args.fit_gate_root.resolve()
    analysis.verify_sha256s(fit_root, "FIT_GATE_SHA256SUMS")
    gate = json.loads((fit_root / "fit_gate.json").read_text())
    if gate.get("schema") != "sci-align-001-lissajous-event-centroid-fit-gate-v1":
        raise EventCentroidReviewError("unsupported event-centroid gate schema")
    protocol = analysis.load_protocol(args.protocol.resolve())
    crossing_protocol = crossings.load_crossing_protocol(
        args.crossing_protocol.resolve()
    )
    crossings.authenticate_base_protocol(protocol, crossing_protocol)
    centroid_protocol = centroids.load_event_centroid_protocol(
        args.centroid_protocol.resolve(), args.crossing_protocol.resolve()
    )
    selection = analysis.load_selection(
        args.selection.resolve(),
        protocol["input_authority"]["selection_manifest_sha256"],
    )
    row = analysis.selected_row(selection, args.obsnum)
    observation = analysis.prepare_observation(row, protocol)
    events = crossings.catalog_crossing_events(observation, crossing_protocol)
    rows = centroids.catalog_event_centroids(
        observation, events, centroid_protocol
    )
    if centroids.centroid_census(rows, centroid_protocol) != gate["centroid_census"]:
        raise EventCentroidReviewError("reconstructed centroid census changed")
    recorded = Table.read(fit_root / "event_centroids.ecsv")
    if list(map(str, recorded["event_id"])) != list(map(str, rows["event_id"])):
        raise EventCentroidReviewError("centroid event identity/order changed")
    for name in ("peak_shift_arcsec", "peak_correlation"):
        if not np.allclose(
            np.asarray(recorded[name], float), np.asarray(rows[name], float),
            rtol=0.0, atol=1.0e-12, equal_nan=True,
        ):
            raise EventCentroidReviewError(f"reconstructed {name} changed")
    fit = gate["point_model_results"]["lag"]
    prediction = centroids.centroid_prediction(rows, fit)
    selection_document = deterministic_selection(
        rows, prediction, args.selected_count
    )
    output.mkdir(parents=True)
    analysis.write_json(output / "selected_events.json", selection_document)
    detail = write_detail_pdf(
        output, observation, events, rows, fit, selection_document
    )
    summary = write_summary_pdf(output, observation, events, rows, gate)
    profile_rows = Table.read(fit_root / "centroid_tau_profile.ecsv")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    tau = np.asarray(profile_rows["tau_ms"], float)
    objective = np.asarray(profile_rows["objective"], float)
    ax.plot(tau, objective - np.min(objective), "o-")
    ax.axvline(0.0, color="0.5")
    ax.axvline(float(fit["tau_ms"]), color="tab:red", ls="--",
               label=f"robust lag {float(fit['tau_ms']):+.3f} ms")
    ax.set(xlabel="Fixed tau (ms)", ylabel="Robust objective - minimum",
           title=f"Obs {args.obsnum}: profiled robust centroid lag")
    ax.legend()
    profile_name = f"event_centroid_tau_profile_o{args.obsnum}.pdf"
    fig.savefig(output / profile_name)
    plt.close(fig)
    manifest = {
        "schema": "sci-align-001-lissajous-event-centroid-review-v1",
        "obsnum": int(args.obsnum),
        "fit_gate_sha256": analysis.sha256_file(fit_root / "fit_gate.json"),
        "fit_gate_sha256s_sha256": analysis.sha256_file(
            fit_root / "FIT_GATE_SHA256SUMS"
        ),
        "protocol_sha256": analysis.sha256_file(args.protocol.resolve()),
        "crossing_protocol_sha256": analysis.sha256_file(
            args.crossing_protocol.resolve()
        ),
        "centroid_protocol_sha256": analysis.sha256_file(
            args.centroid_protocol.resolve()
        ),
        "selection_sha256": analysis.sha256_file(args.selection.resolve()),
        "crossing_census": gate["crossing_census"],
        "centroid_census": gate["centroid_census"],
        "selected_individual_event_count": len(selection_document["selected"]),
        "event_qualification_uses_fitted_parameter": False,
        "timestamp_detector_averaging_used": False,
        "disposition": "owner_review_before_remaining_65_observations",
        "outputs": [detail, summary, profile_name],
    }
    analysis.write_json(output / "manifest.json", manifest)
    analysis.write_checksums(output, [
        detail, "manifest.json", profile_name, "selected_events.json", summary,
    ])
    analysis.verify_sha256s(output)
    print(
        f"event centroid review complete: obs={args.obsnum} "
        f"selected={len(selection_document['selected'])} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--crossing-protocol", type=Path, required=True)
    result.add_argument("--centroid-protocol", type=Path, required=True)
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
        EventCentroidReviewError,
        analysis.ContractError,
        crossings.CrossingContractError,
        centroids.EventCentroidError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
