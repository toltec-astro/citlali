#!/usr/bin/env python3
"""Run a SCI-ALIGN-001 fit gate with atomic per-model checkpoints.

This is an orchestration-only wrapper around the frozen numerical fitter in
``analyze_sci_align_001_lissajous_timestream.py``.  It does not change model
arithmetic, optimizer starts, bounds, objective functions, or final fit-gate
schema.  A completed model is checksum-bound immediately and reused only when
the complete input, support, coordinate, map, numerical implementation, and
wrapper identities still match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import analyze_sci_align_001_lissajous_timestream as target


CHECKPOINT_NAME = "FIT_GATE_MODEL_CHECKPOINT.json"
CHECKSUM_NAME = "FIT_GATE_MODEL_CHECKPOINT_SHA256SUMS"


class ContractError(RuntimeError):
    """A checkpointed fit-gate lifecycle contract was violated."""


def payload_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_checkpoint_checksum(output: Path) -> None:
    temporary = output / f".{CHECKSUM_NAME}.tmp"
    temporary.write_text(
        f"{target.sha256_file(output / CHECKPOINT_NAME)}  {CHECKPOINT_NAME}\n"
    )
    temporary.replace(output / CHECKSUM_NAME)


def checkpoint_identity(
    args: argparse.Namespace,
    row: dict[str, Any],
    observation: Any,
    coordinate_gate: dict[str, Any],
    map_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "sci-align-001-fit-gate-model-checkpoint-identity-v1",
        "obsnum": int(args.obsnum),
        "ptc_sha256": row["ptc_sha256"],
        "ppt_sha256": row["ppt_sha256"],
        "protocol_sha256": target.sha256_file(args.protocol),
        "selection_sha256": target.sha256_file(args.selection),
        "map_result": map_result,
        "support": target.observation_support_summary(observation),
        "coordinate_gate": coordinate_gate,
        "numerical_implementation_path": str(Path(target.__file__).resolve()),
        "numerical_implementation_sha256": target.sha256_file(
            Path(target.__file__).resolve()
        ),
        "orchestrator_path": str(Path(__file__).resolve()),
        "orchestrator_sha256": target.sha256_file(Path(__file__).resolve()),
    }


def load_checkpoint(
    output: Path, identity: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    path = output / CHECKPOINT_NAME
    manifest = output / CHECKSUM_NAME
    if not path.exists() and not manifest.exists():
        return {}
    if not path.exists():
        raise ContractError("fit-gate checkpoint manifest lacks its payload")
    document = json.loads(path.read_text())
    if document.get("schema") != "sci-align-001-fit-gate-model-checkpoint-v1":
        raise ContractError("unsupported fit-gate model checkpoint schema")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ContractError("fit-gate model checkpoint payload is invalid")
    if document.get("payload_sha256") != payload_sha256(payload):
        raise ContractError("fit-gate model checkpoint payload digest changed")
    try:
        target.verify_sha256s(output, CHECKSUM_NAME)
    except (OSError, target.ContractError):
        # The JSON is the atomic authority. A termination after its replacement
        # but before the companion manifest replacement leaves a valid,
        # self-digested payload and a stale or absent convenience manifest.
        write_checkpoint_checksum(output)
        target.verify_sha256s(output, CHECKSUM_NAME)
    if payload.get("identity") != identity:
        raise ContractError("fit-gate model checkpoint identity changed")
    fits = payload.get("completed_model_fits", {})
    completed = payload.get("completed_models", [])
    if completed != [name for name in target.MODEL_NAMES if name in fits]:
        raise ContractError("fit-gate model checkpoint order is invalid")
    return {name: fits[name] for name in target.MODEL_NAMES if name in fits}


def save_checkpoint(
    output: Path,
    identity: dict[str, Any],
    fits: dict[str, dict[str, Any]],
) -> None:
    ordered = [name for name in target.MODEL_NAMES if name in fits]
    payload = {
        "identity": identity,
        "completed_models": ordered,
        "completed_model_fits": {name: fits[name] for name in ordered},
    }
    document = {
        "schema": "sci-align-001-fit-gate-model-checkpoint-v1",
        "payload_sha256": payload_sha256(payload),
        "payload": payload,
    }
    target.write_json_atomic(output / CHECKPOINT_NAME, document)
    write_checkpoint_checksum(output)
    target.verify_sha256s(output, CHECKSUM_NAME)


def hydrate_monitor_counts(monitor: target.RunMonitor) -> None:
    """Restore cumulative counters when appending to a prior progress log."""
    if not monitor.progress_path.is_file():
        return
    attempts = 0
    fallbacks = 0
    for line in monitor.progress_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        attempts += record.get("event") == "optimizer_attempt_start"
        fallbacks += record.get("event") == "optimizer_fallback"
    monitor.optimizer_attempt_count = int(attempts)
    monitor.optimizer_fallback_count = int(fallbacks)


def protect_checkpoint_with_fit_gate(output: Path) -> None:
    """Add operational checkpoints to the immutable fit-gate hash layer."""
    manifest = output / "FIT_GATE_SHA256SUMS"
    names = []
    for line in manifest.read_text().splitlines():
        _, name = line.split(maxsplit=1)
        names.append(name.strip())
    names.extend([CHECKPOINT_NAME, CHECKSUM_NAME])
    target.write_checksums(output, sorted(set(names)), "FIT_GATE_SHA256SUMS")
    target.verify_sha256s(output, "FIT_GATE_SHA256SUMS")


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if (output / "fit_gate.json").exists():
        raise ContractError(f"fit gate is already complete: {output}")
    output.mkdir(parents=True, exist_ok=True)
    monitor = target.RunMonitor(output, args.maximum_wall_seconds)
    hydrate_monitor_counts(monitor)
    monitor.emit(
        "run_start", stage="fit_gate_checkpointed", obsnum=args.obsnum,
        maximum_wall_seconds=args.maximum_wall_seconds,
    )
    try:
        with monitor.stage("authenticate_inputs"):
            protocol = target.load_protocol(args.protocol)
            selection = target.load_selection(
                args.selection,
                protocol["input_authority"]["selection_manifest_sha256"],
            )
            row = target.selected_row(selection, args.obsnum)
        with monitor.stage("prepare_observation"):
            observation = target.prepare_observation(row, protocol)
        with monitor.stage("coordinate_reconstruction_gate"):
            coordinate_gate = target.coordinate_reconstruction_gate(observation)
        with monitor.stage("authenticate_map_result"):
            map_result = target.authenticated_map_result(
                args.map_root.resolve(), row
            )
        identity = checkpoint_identity(
            args, row, observation, coordinate_gate, map_result
        )
        with monitor.stage("authenticate_model_checkpoints"):
            full_fits = load_checkpoint(output, identity)
        for model in target.MODEL_NAMES:
            if model in full_fits:
                monitor.emit(
                    "model_checkpoint_reused", stage=f"full.{model}",
                    model=model, status="success",
                )
                continue
            with monitor.stage(f"full_model_fit.{model}"):
                full_fits[model] = target.fit_observation_model(
                    observation, model, monitor=monitor,
                    fit_label=f"full.{model}",
                )
                save_checkpoint(output, identity, full_fits)
                monitor.emit(
                    "model_checkpoint_saved", stage=f"full.{model}",
                    model=model, status="success",
                    completed=len(full_fits), target=len(target.MODEL_NAMES),
                )
        with monitor.stage("write_fit_gate"):
            gate = target.write_fit_gate_checkpoint(
                args, output, observation, row, coordinate_gate,
                map_result, full_fits, monitor,
            )
            protect_checkpoint_with_fit_gate(output)
    except BaseException as error:
        target.write_json(output / "run_state.json", monitor.state(
            "failed", obsnum=args.obsnum, error_type=type(error).__name__,
            error_message=str(error),
        ))
        raise
    monitor.emit(
        "run_complete", stage="fit_gate_checkpointed",
        status="fit_gate_complete", obsnum=args.obsnum,
    )
    target.write_json(output / "run_state.json", monitor.state(
        "fit_gate_complete", obsnum=args.obsnum,
        current_stage="awaiting_owner_review",
        automatic_structural_status=(
            gate["quality_gate"]["automatic_structural_status"]
        ),
    ))
    print(
        f"fit gate complete: obs={args.obsnum} "
        f"review={output / f'lissajous_fit_gate_o{args.obsnum}.pdf'} "
        f"output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--map-root", type=Path, required=True)
    result.add_argument("--obsnum", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--maximum-wall-seconds", type=float, default=2700.0)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (ContractError, target.ContractError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
