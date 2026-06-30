#!/usr/bin/env python3
"""Compare two Citlali low-level YAML trees."""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "citlali-lowlevel-yaml-comparison-v1"


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def step_values(steps: Any) -> list[Any]:
    if isinstance(steps, list):
        return steps
    if isinstance(steps, dict):
        def step_sort_key(item: Any) -> tuple[int, Any]:
            text = str(item)
            if text.isdigit():
                return (0, int(text))
            return (1, text)

        return [steps[key] for key in sorted(steps, key=step_sort_key)]
    return []


def extract_low_level(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    reduce_section = data.get("reduce")
    if not isinstance(reduce_section, dict):
        return data
    for step in step_values(reduce_section.get("steps", [])):
        if not isinstance(step, dict):
            continue
        config = step.get("config", {})
        if isinstance(config, dict) and "low_level" in config:
            return config["low_level"] or {}
    return data


def path_to_string(parts: tuple[str, ...]) -> str:
    result = ""
    for part in parts:
        if part.startswith("["):
            result += part
        elif result:
            result += "." + part
        else:
            result = part
    return result


def normalize_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("[]" if part.startswith("[") else part for part in parts)


def value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def value_preview(value: Any, limit: int = 140) -> str:
    text = value_key(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def walk_leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, child in value.items():
            rows.extend(walk_leaves(child, prefix + (str(key),)))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(walk_leaves(child, prefix + (f"[{index}]",)))
        return rows
    return [
        {
            "path": path_to_string(prefix),
            "normalized_path": path_to_string(normalize_parts(prefix)),
            "top": prefix[0] if prefix else "",
            "value": value,
            "value_key": value_key(value),
            "value_preview": value_preview(value),
            "value_type": type(value).__name__,
        }
    ]


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: collections.Counter[str] = collections.Counter()
    for row in rows:
        path = row["normalized_path"]
        if path in result:
            duplicates[path] += 1
            path = f"{path}#{duplicates[path] + 1}"
        result[path] = row
    return result


def is_ignored(path: str, ignore_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns)


def compare(
    baseline: Any,
    candidate: Any,
    ignore_patterns: list[str],
) -> dict[str, Any]:
    baseline_rows = row_map(walk_leaves(extract_low_level(baseline)))
    candidate_rows = row_map(walk_leaves(extract_low_level(candidate)))

    baseline_paths = set(baseline_rows)
    candidate_paths = set(candidate_rows)
    all_paths = sorted(baseline_paths | candidate_paths)

    diffs: list[dict[str, Any]] = []
    for path in all_paths:
        if is_ignored(path, ignore_patterns):
            continue
        baseline_row = baseline_rows.get(path)
        candidate_row = candidate_rows.get(path)
        if baseline_row is None:
            diffs.append(
                {
                    "kind": "extra_candidate_path",
                    "path": path,
                    "top": candidate_row["top"] if candidate_row else "",
                    "baseline": None,
                    "candidate": candidate_row["value_preview"] if candidate_row else None,
                }
            )
            continue
        if candidate_row is None:
            diffs.append(
                {
                    "kind": "missing_candidate_path",
                    "path": path,
                    "top": baseline_row["top"],
                    "baseline": baseline_row["value_preview"],
                    "candidate": None,
                }
            )
            continue
        if baseline_row["value_key"] != candidate_row["value_key"]:
            diffs.append(
                {
                    "kind": "changed_value",
                    "path": path,
                    "top": baseline_row["top"],
                    "baseline": baseline_row["value_preview"],
                    "candidate": candidate_row["value_preview"],
                }
            )

    counts_by_kind = collections.Counter(diff["kind"] for diff in diffs)
    counts_by_top = collections.Counter(diff["top"] for diff in diffs)
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "baseline_leaf_count": len(baseline_rows),
            "candidate_leaf_count": len(candidate_rows),
            "diff_count": len(diffs),
            "diff_count_by_kind": dict(sorted(counts_by_kind.items())),
            "diff_count_by_top": dict(sorted(counts_by_top.items())),
            "ignore_patterns": ignore_patterns,
        },
        "diffs": diffs,
    }


def write_markdown(result: dict[str, Any], out_path: Path, max_rows: int) -> None:
    summary = result["summary"]
    lines = [
        "# Citlali Low-Level YAML Comparison",
        "",
        "## Summary",
        "",
        f"- Baseline leaf keys: {summary['baseline_leaf_count']}",
        f"- Candidate leaf keys: {summary['candidate_leaf_count']}",
        f"- Differences: {summary['diff_count']}",
        "",
        "Ignored patterns:",
        "",
    ]
    for pattern in summary["ignore_patterns"]:
        lines.append(f"- `{pattern}`")

    lines.extend(["", "## Differences By Kind", "", "| Kind | Count |", "| --- | ---: |"])
    for kind, count in summary["diff_count_by_kind"].items():
        lines.append(f"| `{kind}` | {count} |")

    lines.extend(["", "## Differences By Top-Level Node", "", "| Node | Count |", "| --- | ---: |"])
    for top, count in sorted(summary["diff_count_by_top"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{top}` | {count} |")

    lines.extend(
        [
            "",
            f"## First {max_rows} Differences",
            "",
            "| Kind | Path | Baseline | Candidate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for diff in result["diffs"][:max_rows]:
        lines.append(
            f"| `{diff['kind']}` | `{markdown_cell(diff['path'])}` | "
            f"`{markdown_cell(diff['baseline'])}` | `{markdown_cell(diff['candidate'])}` |"
        )
    if len(result["diffs"]) > max_rows:
        lines.append(f"| _{len(result['diffs']) - max_rows} more omitted_ |  |  |  |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline YAML file.")
    parser.add_argument("candidate", help="Candidate YAML file.")
    parser.add_argument("--ignore", action="append", default=[], help="Dotted-path glob to ignore.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown output path.")
    parser.add_argument("--max-markdown-rows", type=int, default=120, help="Maximum diff rows in Markdown output.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = compare(
        load_yaml(Path(args.baseline).expanduser().resolve()),
        load_yaml(Path(args.candidate).expanduser().resolve()),
        list(args.ignore),
    )
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(result, Path(args.markdown_out).expanduser(), args.max_markdown_rows)

    diff_count = result["summary"]["diff_count"]
    print(f"baseline_leaf_count={result['summary']['baseline_leaf_count']}")
    print(f"candidate_leaf_count={result['summary']['candidate_leaf_count']}")
    print(f"diff_count={diff_count}")
    return 0 if diff_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
