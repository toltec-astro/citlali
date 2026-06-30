#!/usr/bin/env python3
"""Compare two Citlali output manifests."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_IGNORE_PATTERNS = [
    "generated_utc",
    "run.output_dir",
    "run.command",
    "run.wall_time_sec",
    "run.peak_rss_kb",
    "run.environment",
    "run.git_sha",
    "run.branch",
    "files.*.mtime_ns",
]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def ignored(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def values_close(a: int | float, b: int | float, atol: float, rtol: float) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def add_diff(
    diffs: list[dict[str, Any]],
    path: str,
    kind: str,
    baseline: Any,
    candidate: Any,
    max_diffs: int,
) -> None:
    if len(diffs) >= max_diffs:
        return
    diffs.append(
        {
            "path": path,
            "kind": kind,
            "baseline": baseline,
            "candidate": candidate,
        }
    )


def compare_values(
    path: str,
    baseline: Any,
    candidate: Any,
    diffs: list[dict[str, Any]],
    ignore_patterns: list[str],
    atol: float,
    rtol: float,
    max_diffs: int,
) -> None:
    if len(diffs) >= max_diffs or ignored(path, ignore_patterns):
        return

    if type(baseline) is not type(candidate):
        if is_number(baseline) and is_number(candidate):
            if not values_close(baseline, candidate, atol, rtol):
                add_diff(diffs, path, "numeric_changed", baseline, candidate, max_diffs)
            return
        add_diff(diffs, path, "type_changed", type(baseline).__name__, type(candidate).__name__, max_diffs)
        return

    if isinstance(baseline, dict):
        base_keys = set(baseline)
        cand_keys = set(candidate)
        for key in sorted(base_keys - cand_keys):
            key_path = f"{path}.{key}" if path else str(key)
            if not ignored(key_path, ignore_patterns):
                add_diff(diffs, key_path, "missing_key", baseline[key], None, max_diffs)
        for key in sorted(cand_keys - base_keys):
            key_path = f"{path}.{key}" if path else str(key)
            if not ignored(key_path, ignore_patterns):
                add_diff(diffs, key_path, "extra_key", None, candidate[key], max_diffs)
        for key in sorted(base_keys & cand_keys):
            key_path = f"{path}.{key}" if path else str(key)
            compare_values(
                key_path,
                baseline[key],
                candidate[key],
                diffs,
                ignore_patterns,
                atol,
                rtol,
                max_diffs,
            )
            if len(diffs) >= max_diffs:
                return
        return

    if isinstance(baseline, list):
        if len(baseline) != len(candidate):
            add_diff(diffs, path, "list_length_changed", len(baseline), len(candidate), max_diffs)
        for index, (base_value, cand_value) in enumerate(zip(baseline, candidate)):
            compare_values(
                f"{path}[{index}]",
                base_value,
                cand_value,
                diffs,
                ignore_patterns,
                atol,
                rtol,
                max_diffs,
            )
            if len(diffs) >= max_diffs:
                return
        return

    if is_number(baseline):
        if not values_close(baseline, candidate, atol, rtol):
            add_diff(diffs, path, "numeric_changed", baseline, candidate, max_diffs)
        return

    if baseline != candidate:
        add_diff(diffs, path, "value_changed", baseline, candidate, max_diffs)


def file_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in manifest.get("files", []):
        if isinstance(record, dict) and "path" in record:
            result[str(record["path"])] = record
    return result


def compare_file_sets(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    diffs: list[dict[str, Any]],
    ignore_patterns: list[str],
    atol: float,
    rtol: float,
    max_diffs: int,
) -> None:
    base_files = file_map(baseline)
    cand_files = file_map(candidate)
    base_paths = set(base_files)
    cand_paths = set(cand_files)

    for relpath in sorted(base_paths - cand_paths):
        add_diff(diffs, f"files.{relpath}", "missing_file", base_files[relpath], None, max_diffs)
        if len(diffs) >= max_diffs:
            return
    for relpath in sorted(cand_paths - base_paths):
        add_diff(diffs, f"files.{relpath}", "extra_file", None, cand_files[relpath], max_diffs)
        if len(diffs) >= max_diffs:
            return
    for relpath in sorted(base_paths & cand_paths):
        compare_values(
            f"files.{relpath}",
            base_files[relpath],
            cand_files[relpath],
            diffs,
            ignore_patterns,
            atol,
            rtol,
            max_diffs,
        )
        if len(diffs) >= max_diffs:
            return


def compare_manifests(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_manifest(Path(args.baseline))
    candidate = load_manifest(Path(args.candidate))
    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS) + list(args.ignore)
    if args.ignore_sha256:
        ignore_patterns.extend(["files.*.sha256", "files.*.summary.*.data_sha256", "files.*.summary.*.*.data_sha256"])

    diffs: list[dict[str, Any]] = []
    compare_values(
        "schema_version",
        baseline.get("schema_version"),
        candidate.get("schema_version"),
        diffs,
        ignore_patterns,
        args.atol,
        args.rtol,
        args.max_diffs,
    )
    compare_values(
        "run.case",
        baseline.get("run", {}).get("case"),
        candidate.get("run", {}).get("case"),
        diffs,
        ignore_patterns,
        args.atol,
        args.rtol,
        args.max_diffs,
    )
    compare_values(
        "aggregate",
        baseline.get("aggregate", {}),
        candidate.get("aggregate", {}),
        diffs,
        ignore_patterns,
        args.atol,
        args.rtol,
        args.max_diffs,
    )
    compare_file_sets(
        baseline,
        candidate,
        diffs,
        ignore_patterns,
        args.atol,
        args.rtol,
        args.max_diffs,
    )

    truncated = len(diffs) >= args.max_diffs
    return {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "atol": args.atol,
        "rtol": args.rtol,
        "ignore_patterns": ignore_patterns,
        "diff_count": len(diffs),
        "truncated": truncated,
        "diffs": diffs,
    }


def print_report(result: dict[str, Any]) -> None:
    diff_count = int(result["diff_count"])
    if diff_count == 0:
        print("manifests match")
        return
    print(f"manifest differences: {diff_count}")
    if result.get("truncated"):
        print("difference report reached --max-diffs; rerun with a larger limit for full detail")
    for diff in result["diffs"]:
        print(f"- {diff['kind']}: {diff['path']}")
        print(f"  baseline: {short_value(diff['baseline'])}")
        print(f"  candidate: {short_value(diff['candidate'])}")


def short_value(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline JSON manifest.")
    parser.add_argument("candidate", help="Candidate JSON manifest.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for floating values.")
    parser.add_argument("--rtol", type=float, default=1.0e-10, help="Relative tolerance for floating values.")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Additional dotted-path glob to ignore, for example 'files.*.summary.hdus[0].header.DATE'.",
    )
    parser.add_argument(
        "--ignore-sha256",
        action="store_true",
        help="Ignore full-file and array SHA-256 differences.",
    )
    parser.add_argument("--max-diffs", type=int, default=200, help="Maximum differences to report.")
    parser.add_argument("--json-out", default="", help="Optional path to write machine-readable diff JSON.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = compare_manifests(args)
    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print_report(result)
    return 0 if int(result["diff_count"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
