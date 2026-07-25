#!/usr/bin/env python3
"""Prepare a restart-matched control/injected fruit-loop transfer pair."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import yaml


def nested(config: dict, *path: str) -> dict:
    node: object = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"required config path is absent: {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(f"config path is not a mapping: {'.'.join(path)}")
    return node


def require_source_config(config: dict) -> None:
    runtime = nested(config, "runtime")
    fruit = nested(config, "timestream", "fruit_loops")
    if runtime.get("reduction_type") != "pointing":
        raise ValueError("injected-source transfer test requires pointing mode")
    if fruit.get("enabled") is not True:
        raise ValueError("fruit loops must be enabled")
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("transfer test requires exactly one observation")


def output_config(
    source: dict,
    *,
    enabled: bool,
    restart_path: str,
    output_dir: str,
    start_iteration: int,
    stop_iteration: int,
    amplitudes: list[float],
) -> dict:
    config = copy.deepcopy(source)
    config["runtime"]["output_dir"] = output_dir
    raw = nested(config, "timestream", "raw_time_chunk")
    raw["kernel"]["enabled"] = True
    fruit = nested(config, "timestream", "fruit_loops")
    fruit["path"] = None
    fruit["restart_path"] = restart_path
    fruit["max_iters"] = stop_iteration
    fruit["save_all_iters"] = True
    fruit["diagnostics_enabled"] = True
    fruit["injected_source_test"] = {
        "enabled": enabled,
        "start_iteration": start_iteration,
        "array_amplitude_mjy_beam": amplitudes,
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--restart-path",
        required=True,
        help="Completed reduNN immediately before the injection start",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-output-root",
        required=True,
        help="Unity root under which control/reduced and injected/reduced live",
    )
    parser.add_argument("--start-iteration", required=True, type=int)
    parser.add_argument(
        "--additional-iterations",
        type=int,
        default=5,
        help="Number of paired iterations to run (default: 5)",
    )
    parser.add_argument(
        "--amplitudes-mjy-beam",
        required=True,
        nargs=3,
        type=float,
        metavar=("A1100", "A1400", "A2000"),
    )
    args = parser.parse_args()

    if args.start_iteration < 1:
        parser.error("--start-iteration must be at least 1")
    if args.additional_iterations < 2:
        parser.error("--additional-iterations must be at least 2")
    if not all(math.isfinite(value) for value in args.amplitudes_mjy_beam):
        parser.error("injected amplitudes must be finite")
    if not any(value > 0.0 for value in args.amplitudes_mjy_beam):
        parser.error("at least one injected amplitude must be positive")
    if any(value < 0.0 for value in args.amplitudes_mjy_beam):
        parser.error("injected amplitudes cannot be negative")

    source = yaml.safe_load(args.input.read_text())
    if not isinstance(source, dict):
        raise ValueError("input must contain one YAML mapping")
    require_source_config(source)

    root = args.runtime_output_root.rstrip("/")
    stop_iteration = args.start_iteration + args.additional_iterations
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for label, enabled in (("control", False), ("injected", True)):
        config = output_config(
            source,
            enabled=enabled,
            restart_path=args.restart_path,
            output_dir=f"{root}/{label}/reduced/",
            start_iteration=args.start_iteration,
            stop_iteration=stop_iteration,
            amplitudes=list(args.amplitudes_mjy_beam),
        )
        filename = f"citlali_injected_source_{label}.yaml"
        (args.output_dir / filename).write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        variants[label] = {
            "config": filename,
            "output_dir": f"{root}/{label}/reduced/",
            "injection_enabled": enabled,
        }

    manifest = {
        "schema_version": "citlali-fruit-loop-injected-source-pair-v1",
        "source_config": str(args.input.resolve()),
        "restart_path": args.restart_path,
        "start_iteration": args.start_iteration,
        "stop_iteration_exclusive": stop_iteration,
        "additional_iterations": args.additional_iterations,
        "array_order": ["a1100", "a1400", "a2000"],
        "array_amplitude_mjy_beam": list(args.amplitudes_mjy_beam),
        "variants": variants,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    print(f"wrote control/injected pair to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
