#!/usr/bin/env python3
"""Run the spatial event-centroid SCI-ALIGN-001 Lissajous fit gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from astropy.table import Table

import analyze_sci_align_001_lissajous_timestream as analysis
import sci_align_001_lissajous_crossings as crossings
import sci_align_001_lissajous_event_centroids as centroids


class EventFitGateError(RuntimeError):
    """The event-centroid fit gate violates its frozen contract."""


def _input_identity(
    args: argparse.Namespace,
    row: dict,
    map_result: dict,
) -> dict:
    return {
        "pointing_obsnum": int(row["pointing_obsnum"]),
        "ptc_path": row["ptc_path"],
        "ptc_sha256": row["ptc_sha256"],
        "ppt_path": row["ppt_path"],
        "ppt_sha256": row["ppt_sha256"],
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": analysis.sha256_file(args.selection.resolve()),
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": analysis.sha256_file(args.protocol.resolve()),
        "crossing_protocol_path": str(args.crossing_protocol.resolve()),
        "crossing_protocol_sha256": analysis.sha256_file(
            args.crossing_protocol.resolve()
        ),
        "centroid_protocol_path": str(args.centroid_protocol.resolve()),
        "centroid_protocol_sha256": analysis.sha256_file(
            args.centroid_protocol.resolve()
        ),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": analysis.sha256_file(Path(__file__).resolve()),
        "centroid_implementation_path": str(Path(centroids.__file__).resolve()),
        "centroid_implementation_sha256": analysis.sha256_file(
            Path(centroids.__file__).resolve()
        ),
        "map_result": map_result,
    }


def _model_rows(result: dict) -> list[dict]:
    rows = []
    for model in centroids.MODEL_NAMES:
        fit = result["models"][model]
        parameters = fit["parameters"]
        rows.append({
            "model": model,
            "status": fit["status"],
            "objective_arcsec2": fit["objective"],
            "x0_arcsec": parameters["x0_arcsec"],
            "y0_arcsec": parameters["y0_arcsec"],
            "tau_ms": parameters.get("tau_ms", 0.0),
            "h_az_arcsec": parameters.get("h_az_arcsec", math.nan),
            "h_el_arcsec": parameters.get("h_el_arcsec", math.nan),
            "event_count": fit["event_count"],
            "detector_count": fit["detector_count"],
            "robust_scale_arcsec": fit["robust_scale_arcsec"],
            "residual_rms_arcsec": fit["residual_rms_arcsec"],
            "residual_mad_scale_arcsec": fit["residual_mad_scale_arcsec"],
            "iterations": fit["iterations"],
            "design_parameter_count": fit["design_parameter_count"],
            "design_rank": fit["design_rank"],
            "design_condition_number": fit["design_condition_number"],
            "boundary": fit["boundary"],
        })
    return rows


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise EventFitGateError(f"event-centroid output already exists: {output}")
    output.mkdir(parents=True)
    monitor = analysis.RunMonitor(output, args.maximum_wall_seconds)
    monitor.emit("run_start", stage="event_centroid_fit_gate", obsnum=args.obsnum)
    try:
        with monitor.stage("authenticate_inputs"):
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
        with monitor.stage("prepare_observation"):
            observation = analysis.prepare_observation(row, protocol)
        with monitor.stage("coordinate_reconstruction_gate"):
            coordinate_gate = analysis.coordinate_reconstruction_gate(observation)
        with monitor.stage("authenticate_map_result"):
            map_result = analysis.authenticated_map_result(
                args.map_root.resolve(), row
            )
        with monitor.stage("catalog_geometric_crossings"):
            events = crossings.catalog_crossing_events(
                observation, crossing_protocol
            )
            _, support = crossings.restrict_to_crossing_support(
                observation, events, crossing_protocol
            )
            crossing_census = crossings.event_census(events, support)
            events.write(
                output / "crossing_events.ecsv", format="ascii.ecsv"
            )
            support.write(
                output / "crossing_support.ecsv", format="ascii.ecsv"
            )
        with monitor.stage("profile_event_centroids"):
            centroid_rows = centroids.catalog_event_centroids(
                observation, events, centroid_protocol
            )
            centroid_rows.write(
                output / "event_centroids.ecsv", format="ascii.ecsv"
            )
            centroid_census = centroids.centroid_census(
                centroid_rows, centroid_protocol
            )
        with monitor.stage("fit_robust_centroid_models"):
            threshold_results = []
            for threshold in centroid_protocol["source_quality"][
                "correlation_sensitivity_thresholds"
            ]:
                monitor.check_deadline(f"centroid_threshold_{threshold}")
                threshold_results.append(centroids.fit_centroid_models(
                    centroid_rows, centroid_protocol, float(threshold)
                ))
            primary_threshold = float(centroid_protocol["source_quality"][
                "primary_minimum_correlation"
            ])
            primary = next(
                result for result in threshold_results
                if float(result["threshold"]) == primary_threshold
            )
            profile = centroids.robust_tau_profile(
                centroid_rows, centroid_protocol, primary
            )
        with monitor.stage("write_results"):
            Table(rows=_model_rows(primary)).write(
                output / "centroid_model_results.ecsv", format="ascii.ecsv"
            )
            Table(rows=[{
                "minimum_correlation": result["threshold"],
                "qualified_event_count": result["qualified_event_count"],
                "qualified_detector_count": result["qualified_detector_count"],
                "common_robust_scale_arcsec": result[
                    "common_robust_scale_arcsec"
                ],
                "lag_tau_ms": result["models"]["lag"]["tau_ms"],
                "joint_tau_ms": result["models"]["joint"]["tau_ms"],
                "joint_h_az_arcsec": result["models"]["joint"][
                    "parameters"
                ]["h_az_arcsec"],
                "joint_h_el_arcsec": result["models"]["joint"][
                    "parameters"
                ]["h_el_arcsec"],
            } for result in threshold_results]).write(
                output / "centroid_threshold_sensitivity.ecsv",
                format="ascii.ecsv",
            )
            Table(rows=profile).write(
                output / "centroid_tau_profile.ecsv", format="ascii.ecsv"
            )
            gate = {
                "schema": "sci-align-001-lissajous-event-centroid-fit-gate-v1",
                "obsnum": int(args.obsnum),
                "beammap_obsnum": int(row["beammap_obsnum"]),
                "brightness_stratum": row["brightness_stratum"],
                "input": _input_identity(args, row, map_result),
                "base_support": analysis.observation_support_summary(observation),
                "coordinate_gate": coordinate_gate,
                "crossing_census": crossing_census,
                "centroid_census": centroid_census,
                "primary_minimum_correlation": primary_threshold,
                "point_model_results": primary["models"],
                "threshold_sensitivity": threshold_results,
                "quality_gate": {
                    "automatic_structural_status": "pass",
                    "tau_used_as_gate": False,
                    "owner_review_required": True,
                    "qualification_uses_global_model": False,
                    "all_primary_models_successful": all(
                        fit["status"] == "success"
                        for fit in primary["models"].values()
                    ),
                    "all_primary_objectives_finite": all(
                        math.isfinite(float(fit["objective"]))
                        for fit in primary["models"].values()
                    ),
                    "all_primary_designs_full_rank": all(
                        fit["design_rank"] == fit["design_parameter_count"]
                        for fit in primary["models"].values()
                    ),
                    "coordinate_reconstruction": (
                        coordinate_gate["status"] == "pass"
                    ),
                },
                "uncertainty_status": (
                    "deferred_during_baseline_algorithm_validation"
                ),
                "interpretation_boundary": centroid_protocol["interpretation"],
            }
            analysis.write_json(output / "fit_gate.json", gate)
        monitor.emit(
            "run_complete", stage="event_centroid_fit_gate",
            status="fit_gate_complete", obsnum=args.obsnum,
        )
        analysis.write_json(output / "run_state.json", monitor.state(
            "fit_gate_complete", obsnum=args.obsnum,
            current_stage="awaiting_owner_event_centroid_review",
        ))
        analysis.write_checksums(output, [
            "centroid_model_results.ecsv",
            "centroid_tau_profile.ecsv",
            "centroid_threshold_sensitivity.ecsv",
            "crossing_events.ecsv",
            "crossing_support.ecsv",
            "event_centroids.ecsv",
            "fit_gate.json",
            "progress.jsonl",
            "run_state.json",
        ], "FIT_GATE_SHA256SUMS")
        analysis.verify_sha256s(output, "FIT_GATE_SHA256SUMS")
    except BaseException as error:
        analysis.write_json(output / "run_state.json", monitor.state(
            "failed", obsnum=args.obsnum, error_type=type(error).__name__,
            error_message=str(error),
        ))
        raise
    print(
        f"event centroid gate complete: obs={args.obsnum} "
        f"qualified={centroid_census['primary_qualified_event_count']} "
        f"lag_ms={primary['models']['lag']['tau_ms']:+.6f} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--crossing-protocol", type=Path, required=True)
    result.add_argument("--centroid-protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--map-root", type=Path, required=True)
    result.add_argument("--obsnum", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--maximum-wall-seconds", type=float, default=1200.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (
        EventFitGateError,
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
