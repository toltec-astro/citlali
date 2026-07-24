#!/usr/bin/env python3
"""Create focused multi-observation Citlali configs from an effective YAML."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def observation_number(item: dict[str, Any]) -> str:
    name = str(item.get("meta", {}).get("name", ""))
    if not name:
        raise ValueError("input is missing meta.name")
    return name.split("_", 1)[0]


def find_observation(inputs: list[dict[str, Any]], obsnum: str) -> int:
    matches = [
        index
        for index, item in enumerate(inputs)
        if observation_number(item) == obsnum
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one input for observation {obsnum}, "
            f"found {len(matches)}"
        )
    return matches[0]


def select_history(
    inputs: list[dict[str, Any]], terminal_obsnum: str, history_count: int
) -> list[dict[str, Any]]:
    terminal_index = find_observation(inputs, terminal_obsnum)
    if history_count < 0 or history_count > terminal_index:
        raise ValueError(
            f"history count {history_count} is outside 0..{terminal_index}"
        )
    return copy.deepcopy(
        inputs[terminal_index - history_count : terminal_index + 1]
    )


def select_repeated_history(
    inputs: list[dict[str, Any]],
    repeated_obsnum: str,
    repeat_count: int,
    terminal_obsnum: str,
) -> list[dict[str, Any]]:
    if repeat_count < 0:
        raise ValueError("repeat count must be non-negative")
    repeated = inputs[find_observation(inputs, repeated_obsnum)]
    terminal = inputs[find_observation(inputs, terminal_obsnum)]
    return [copy.deepcopy(repeated) for _ in range(repeat_count)] + [
        copy.deepcopy(terminal)
    ]


def make_subset(
    config: dict[str, Any],
    *,
    terminal_obsnum: str,
    output_dir: Path,
    history_count: int | None = None,
    repeated_obsnum: str | None = None,
    repeat_count: int | None = None,
) -> dict[str, Any]:
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("config inputs must be a non-empty sequence")

    if history_count is not None:
        selected = select_history(inputs, terminal_obsnum, history_count)
    elif repeated_obsnum is not None and repeat_count is not None:
        selected = select_repeated_history(
            inputs, repeated_obsnum, repeat_count, terminal_obsnum
        )
    else:
        raise ValueError("select either contiguous or repeated history")

    result = copy.deepcopy(config)
    result["inputs"] = selected
    runtime = result.setdefault("runtime", {})
    runtime["output_dir"] = str(output_dir.resolve()) + "/"
    runtime["use_subdir"] = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--terminal-obsnum", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--history-count",
        type=int,
        help="number of immediately preceding observations to retain",
    )
    selection.add_argument(
        "--repeat-obsnum",
        help="observation to repeat before the terminal observation",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        help="number of copies used with --repeat-obsnum",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.repeat_obsnum is None) != (args.repeat_count is None):
        raise SystemExit(
            "--repeat-obsnum and --repeat-count must be supplied together"
        )

    with args.source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    subset = make_subset(
        config,
        terminal_obsnum=args.terminal_obsnum,
        output_dir=args.output_dir,
        history_count=args.history_count,
        repeated_obsnum=args.repeat_obsnum,
        repeat_count=args.repeat_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(subset, handle, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
