#!/usr/bin/env python3
"""Prepare the uninterrupted reference for an injected-source transfer test."""

from __future__ import annotations

import argparse
import copy
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


def prepare_reference_config(
    source: dict, *, output_dir: str, iterations: int
) -> dict:
    config = copy.deepcopy(source)
    runtime = nested(config, "runtime")
    fruit = nested(config, "timestream", "fruit_loops")
    if runtime.get("reduction_type") != "pointing":
        raise ValueError("injected-source reference requires pointing mode")
    if fruit.get("enabled") is not True:
        raise ValueError("fruit loops must be enabled")
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("reference requires exactly one observation")

    runtime["output_dir"] = output_dir
    nested(config, "timestream", "raw_time_chunk")["kernel"]["enabled"] = True
    fruit["path"] = None
    fruit["restart_path"] = None
    fruit["max_iters"] = iterations
    fruit["save_all_iters"] = True
    fruit["diagnostics_enabled"] = True
    fruit["injected_source_test"] = {
        "enabled": False,
        "start_iteration": 1,
        "array_amplitude_mjy_beam": [0.0, 0.0, 0.0],
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-output-root",
        required=True,
        help="Fresh Unity directory under which reference/reduced will live",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of uninterrupted iterations (default: 10)",
    )
    args = parser.parse_args()

    if args.iterations < 10:
        parser.error("--iterations must be at least 10")

    source = yaml.safe_load(args.input.read_text())
    if not isinstance(source, dict):
        raise ValueError("input must contain one YAML mapping")

    root = args.runtime_output_root.rstrip("/")
    runtime_output_dir = f"{root}/reference/reduced/"
    config = prepare_reference_config(
        source, output_dir=runtime_output_dir, iterations=args.iterations
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "citlali_injected_source_reference.yaml"
    (args.output_dir / filename).write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    manifest = {
        "schema_version": "citlali-fruit-loop-injected-source-reference-v1",
        "source_config": str(args.input.resolve()),
        "iterations": args.iterations,
        "config": filename,
        "runtime_output_dir": runtime_output_dir,
        "restart_source_iteration": 8,
        "continuation_reference_iteration": 9,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    print(f"wrote uninterrupted reference config to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
