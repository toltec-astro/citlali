#!/usr/bin/env python3
"""Round-trip an old low-level/TolTECA config through compact translation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import compare_lowlevel_yaml
import expand_compact_config
import lowlevel_to_compact_config


SCHEMA_VERSION = "citlali-config-translation-roundtrip-v1"


class RoundTripError(RuntimeError):
    """Raised for user-correctable round-trip errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def safe_stem(path: Path) -> str:
    text = path.stem
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def write_markdown(result: dict[str, Any], out_path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Config Translation Round Trip",
        "",
        "## Summary",
        "",
        f"- Input: `{result['input_config']}`",
        f"- Mode: `{result['mode']}`",
        f"- Profile: `{result['profile']}`",
        f"- Compact output: `{result['compact_config']}`",
        f"- Expanded low-level output: `{result['expanded_low_level']}`",
        f"- Expanded TolTECA output: `{result['expanded_tolteca']}`",
        f"- Diff count: {summary['diff_count']}",
        "",
        "## Leaves",
        "",
        "| Tree | Leaves |",
        "| --- | ---: |",
        f"| Input | {summary['baseline_leaf_count']} |",
        f"| Round-trip candidate | {summary['candidate_leaf_count']} |",
        "",
        "## Differences By Kind",
        "",
        "| Kind | Count |",
        "| --- | ---: |",
    ]
    for kind, count in summary["diff_count_by_kind"].items():
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(["", "## Differences By Top-Level Node", "", "| Node | Count |", "| --- | ---: |"])
    for top, count in summary["diff_count_by_top"].items():
        lines.append(f"| `{top}` | {count} |")
    lines.extend(["", "## Compact Translation", ""])
    lines.append(f"- Compact mapped entries: {result['compact_mapping_count']}")
    lines.append(f"- Preserved expert leaf count: {result['preserved_expert_leaf_count']}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_round_trip(args: argparse.Namespace) -> dict[str, Any]:
    input_path = resolve_path(args.input_config, Path.cwd())
    work_dir = resolve_path(args.work_dir, Path.cwd())
    work_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = resolve_path(args.profiles_dir, Path.cwd())
    rules_path = resolve_path(args.classification_rules, Path.cwd()) if args.classification_rules else None

    source_data = load_yaml(input_path)
    low_level = compare_lowlevel_yaml.extract_low_level(source_data)
    if not isinstance(low_level, dict):
        raise RoundTripError("input low-level config must be a mapping")

    mode = lowlevel_to_compact_config.infer_mode(low_level, args.mode)
    profile = args.profile or lowlevel_to_compact_config.PASSTHROUGH_PROFILE_BY_MODE[mode]
    expand_compact_config.load_profile(profiles_dir, profile)

    prefix = args.output_prefix or safe_stem(input_path)
    compact_path = work_dir / f"{prefix}.compact.yaml"
    low_level_path = work_dir / f"{prefix}.roundtrip.low_level.yaml"
    tolteca_path = work_dir / f"{prefix}.roundtrip.tolteca.yaml"
    compare_json_path = work_dir / f"{prefix}.compare.json"
    summary_json_path = work_dir / f"{prefix}.roundtrip.json"
    markdown_path = work_dir / f"{prefix}.roundtrip.md"

    compact, mappings = lowlevel_to_compact_config.build_compact(
        low_level,
        mode=mode,
        profile=profile,
        include_output_dir=args.include_output_dir,
        preserve_unmapped=True,
        classification_rules=rules_path,
        compact_path=compact_path,
    )
    compact_path.write_text(expand_compact_config.dump_yaml(compact), encoding="utf-8")

    expanded, expansion_summary = expand_compact_config.expand_config(
        compact_path,
        None,
        profiles_dir,
        None,
    )
    low_level_output = expand_compact_config.to_low_level_config(expanded)
    tolteca_output = expand_compact_config.format_expanded_output(expanded, "tolteca")
    low_level_path.write_text(expand_compact_config.dump_yaml(low_level_output), encoding="utf-8")
    tolteca_path.write_text(expand_compact_config.dump_yaml(tolteca_output), encoding="utf-8")

    compare_result = compare_lowlevel_yaml.compare(source_data, low_level_output, args.ignore)
    compare_json_path.write_text(json.dumps(compare_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    preservation_record = next((item for item in mappings if item.get("compact") == "expert"), {})
    preserved_expert_leaf_count = int(preservation_record.get("preserved_leaf_count", 0))
    compact_mapping_count = sum(1 for item in mappings if item.get("compact") != "expert")
    result = {
        "schema": SCHEMA_VERSION,
        "input_config": str(input_path),
        "mode": mode,
        "profile": profile,
        "compact_config": str(compact_path),
        "expanded_low_level": str(low_level_path),
        "expanded_tolteca": str(tolteca_path),
        "compare_json": str(compare_json_path),
        "summary_markdown": str(markdown_path),
        "include_output_dir": args.include_output_dir,
        "ignore_patterns": args.ignore,
        "compact_mapping_count": compact_mapping_count,
        "preserved_expert_leaf_count": preserved_expert_leaf_count,
        "expansion_warnings": expansion_summary.get("warnings", []),
        "summary": compare_result["summary"],
    }
    summary_json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, markdown_path)
    result["summary_json"] = str(summary_json_path)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_config", help="Old full low-level YAML or TolTECA YAML with reduce.steps.*.config.low_level.")
    parser.add_argument("--mode", choices=("pointing", "oof", "beammap", "science"), default=None)
    parser.add_argument("--profile", default=None, help="Compact profile name. Defaults to the mode compatibility profile.")
    parser.add_argument("--profiles-dir", default="tools/config/profiles")
    parser.add_argument(
        "--classification-rules",
        default=str(Path(__file__).with_name("config_key_classification.yaml")),
        help="Rules file used to classify mapped and preserved paths in the compact summary.",
    )
    parser.add_argument("--work-dir", default="/tmp/citlali_config_translation_roundtrip")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--include-output-dir", action="store_true", help="Map runtime.output_dir to output.dir instead of preserving it under expert.")
    parser.add_argument("--ignore", action="append", default=[], help="Dotted-path glob to ignore during final comparison.")
    parser.add_argument("--expected-diff-count", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = run_round_trip(args)
    except (OSError, yaml.YAMLError, RoundTripError, lowlevel_to_compact_config.ConvertError, expand_compact_config.ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = result["summary"]
    print(
        "translation round trip: "
        f"diffs={summary['diff_count']} "
        f"baseline_leaves={summary['baseline_leaf_count']} "
        f"candidate_leaves={summary['candidate_leaf_count']}"
    )
    print(f"compact={result['compact_config']}")
    print(f"expanded_low_level={result['expanded_low_level']}")
    if summary["diff_count"] != args.expected_diff_count:
        print(
            f"error: diff_count expected {args.expected_diff_count}, got {summary['diff_count']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
