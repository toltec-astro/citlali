#!/usr/bin/env python3
"""Run a checkpointed event-support SCI-ALIGN-001 Lissajous fit gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import analyze_sci_align_001_lissajous_timestream as analysis
import run_sci_align_001_lissajous_fit_gate_checkpointed as checkpoint
import sci_align_001_lissajous_crossings as crossings


class EventFitGateError(RuntimeError):
    """The event-fit gate violates its frozen lifecycle contract."""


def _manifest_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        _, name = line.split(maxsplit=1)
        names.append(name.strip())
    return names


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if (output / "fit_gate.json").exists():
        raise EventFitGateError(f"event fit gate is already complete: {output}")
    output.mkdir(parents=True, exist_ok=True)
    monitor = analysis.RunMonitor(output, args.maximum_wall_seconds)
    checkpoint.hydrate_monitor_counts(monitor)
    monitor.emit(
        "run_start", stage="event_fit_gate", obsnum=args.obsnum,
        maximum_wall_seconds=args.maximum_wall_seconds,
    )
    try:
        with monitor.stage("authenticate_inputs"):
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
        with monitor.stage("prepare_base_observation"):
            base_observation = analysis.prepare_observation(row, protocol)
        with monitor.stage("freeze_crossing_events"):
            events = crossings.catalog_crossing_events(
                base_observation, crossing_protocol
            )
            observation, support = crossings.restrict_to_crossing_support(
                base_observation, events, crossing_protocol
            )
            events.write(
                output / "crossing_events.ecsv",
                format="ascii.ecsv",
                overwrite=True,
            )
            support.write(
                output / "crossing_support.ecsv",
                format="ascii.ecsv",
                overwrite=True,
            )
            crossing_census = crossings.event_census(events, support)
            crossing_identity = {
                "schema": "sci-align-001-lissajous-realized-crossing-support-v1",
                "crossing_protocol_sha256": analysis.sha256_file(
                    args.crossing_protocol.resolve()
                ),
                "crossing_events_sha256": analysis.sha256_file(
                    output / "crossing_events.ecsv"
                ),
                "crossing_support_sha256": analysis.sha256_file(
                    output / "crossing_support.ecsv"
                ),
                "census": crossing_census,
            }
            analysis.write_json(output / "crossing_support.json", crossing_identity)
            analysis.write_checksums(
                output,
                ["crossing_events.ecsv", "crossing_support.ecsv",
                 "crossing_support.json"],
                "CROSSING_SUPPORT_SHA256SUMS",
            )
        with monitor.stage("coordinate_reconstruction_gate"):
            coordinate_gate = analysis.coordinate_reconstruction_gate(observation)
        with monitor.stage("authenticate_map_result"):
            map_result = analysis.authenticated_map_result(
                args.map_root.resolve(), row
            )
        identity = checkpoint.checkpoint_identity(
            args, row, observation, coordinate_gate, map_result
        )
        identity["schema"] = "sci-align-001-event-fit-gate-checkpoint-identity-v1"
        identity["orchestrator_path"] = str(Path(__file__).resolve())
        identity["orchestrator_sha256"] = analysis.sha256_file(
            Path(__file__).resolve()
        )
        identity["checkpoint_library_path"] = str(
            Path(checkpoint.__file__).resolve()
        )
        identity["checkpoint_library_sha256"] = analysis.sha256_file(
            Path(checkpoint.__file__).resolve()
        )
        identity["crossing_implementation_path"] = str(
            Path(crossings.__file__).resolve()
        )
        identity["crossing_implementation_sha256"] = analysis.sha256_file(
            Path(crossings.__file__).resolve()
        )
        identity["crossing_support"] = crossing_identity
        with monitor.stage("authenticate_model_checkpoints"):
            completed = checkpoint.load_checkpoint(output, identity)
        for model in analysis.MODEL_NAMES:
            if model in completed:
                monitor.emit(
                    "model_checkpoint_reused", stage=f"event.{model}",
                    model=model, status="success",
                )
                continue
            with monitor.stage(f"event_model_fit.{model}"):
                completed[model] = analysis.fit_observation_model(
                    observation, model, monitor=monitor,
                    fit_label=f"event.{model}",
                )
                checkpoint.save_checkpoint(output, identity, completed)
                monitor.emit(
                    "model_checkpoint_saved", stage=f"event.{model}",
                    model=model, status="success", completed=len(completed),
                    target=len(analysis.MODEL_NAMES),
                )
        with monitor.stage("write_event_fit_gate"):
            gate = analysis.write_fit_gate_checkpoint(
                args, output, observation, row, coordinate_gate,
                map_result, completed, monitor,
            )
            gate["schema"] = "sci-align-001-lissajous-event-fit-gate-v1"
            gate["crossing_support"] = crossing_identity
            gate["base_support"] = analysis.observation_support_summary(
                base_observation
            )
            analysis.write_json(output / "fit_gate.json", gate)
            names = _manifest_names(output / "FIT_GATE_SHA256SUMS")
            names.extend([
                "CROSSING_SUPPORT_SHA256SUMS",
                "crossing_events.ecsv",
                "crossing_support.ecsv",
                "crossing_support.json",
                checkpoint.CHECKPOINT_NAME,
                checkpoint.CHECKSUM_NAME,
            ])
            analysis.write_checksums(
                output, sorted(set(names)), "FIT_GATE_SHA256SUMS"
            )
            analysis.verify_sha256s(output, "FIT_GATE_SHA256SUMS")
    except BaseException as error:
        analysis.write_json(output / "run_state.json", monitor.state(
            "failed", obsnum=args.obsnum, error_type=type(error).__name__,
            error_message=str(error),
        ))
        raise
    monitor.emit(
        "run_complete", stage="event_fit_gate", status="fit_gate_complete",
        obsnum=args.obsnum,
    )
    analysis.write_json(output / "run_state.json", monitor.state(
        "fit_gate_complete", obsnum=args.obsnum,
        current_stage="awaiting_owner_event_review",
        automatic_structural_status=(
            gate["quality_gate"]["automatic_structural_status"]
        ),
    ))
    print(
        f"event fit gate complete: obs={args.obsnum} "
        f"events={crossing_census['accepted_event_count']} "
        f"tau_ms={completed['lag']['tau_ms']:+.6f} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--crossing-protocol", type=Path, required=True)
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
        checkpoint.ContractError,
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
