#!/usr/bin/env python3
"""Run compact config equivalence cases against TolTECA low-level baselines."""

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


SCHEMA_VERSION = "citlali-compact-compatibility-results-v1"
SUITE_SCHEMA = "citlali-compact-compatibility-suite-v1"


class SuiteError(RuntimeError):
    """Raised for user-correctable compatibility-suite errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def status_for_case(result: dict[str, Any], require_all: bool) -> str:
    if result.get("missing_inputs"):
        return "failed" if require_all else "skipped"
    if result.get("errors"):
        return "failed"
    return "passed" if result.get("passed") else "failed"


def evaluate_expectations(case: dict[str, Any], compare_result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = compare_result["summary"]
    expected_diff_count = case.get("expected_diff_count", 0)
    if summary["diff_count"] != expected_diff_count:
        errors.append(f"diff_count expected {expected_diff_count}, got {summary['diff_count']}")

    expected_baseline_leaf_count = case.get("expected_baseline_leaf_count")
    if expected_baseline_leaf_count is not None and summary["baseline_leaf_count"] != expected_baseline_leaf_count:
        errors.append(
            "baseline_leaf_count expected "
            f"{expected_baseline_leaf_count}, got {summary['baseline_leaf_count']}"
        )

    expected_candidate_leaf_count = case.get("expected_candidate_leaf_count")
    if expected_candidate_leaf_count is not None and summary["candidate_leaf_count"] != expected_candidate_leaf_count:
        errors.append(
            "candidate_leaf_count expected "
            f"{expected_candidate_leaf_count}, got {summary['candidate_leaf_count']}"
        )
    return errors


def run_case(
    case: dict[str, Any],
    *,
    suite_dir: Path,
    profiles_dir: Path,
    work_dir: Path | None,
    require_all: bool,
) -> dict[str, Any]:
    name = str(case.get("name", "unnamed"))
    compact_value = case.get("compact_config")
    base_value = case.get("base_config")
    if not isinstance(compact_value, str) or not isinstance(base_value, str):
        return {
            "name": name,
            "status": "failed",
            "passed": False,
            "errors": ["case must define string compact_config and base_config"],
        }

    compact_path = resolve_path(compact_value, suite_dir)
    base_path = resolve_path(base_value, suite_dir)
    missing_inputs = []
    if not compact_path.exists():
        missing_inputs.append(str(compact_path))
    if not base_path.exists():
        missing_inputs.append(str(base_path))
    if missing_inputs:
        result = {
            "name": name,
            "intent": case.get("intent", ""),
            "compact_config": str(compact_path),
            "base_config": str(base_path),
            "missing_inputs": missing_inputs,
            "errors": [],
            "passed": False,
        }
        result["status"] = status_for_case(result, require_all)
        return result

    ignore_patterns = case.get("ignore", [])
    if not isinstance(ignore_patterns, list) or not all(isinstance(item, str) for item in ignore_patterns):
        return {
            "name": name,
            "status": "failed",
            "passed": False,
            "errors": ["ignore must be a list of strings"],
        }

    try:
        expanded, expansion_summary = expand_compact_config.expand_config(
            compact_path,
            base_path,
            profiles_dir,
            case.get("profile_override"),
        )
        candidate = expand_compact_config.to_low_level_config(expanded)
        baseline = load_yaml(base_path)
        compare_result = compare_lowlevel_yaml.compare(baseline, candidate, ignore_patterns)
        errors = evaluate_expectations(case, compare_result)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "name": name,
            "intent": case.get("intent", ""),
            "compact_config": str(compact_path),
            "base_config": str(base_path),
            "status": "failed",
            "passed": False,
            "errors": [str(exc)],
        }

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        expanded_path = work_dir / f"{safe_name}.low_level.yaml"
        compare_json_path = work_dir / f"{safe_name}.compare.json"
        expanded_path.write_text(expand_compact_config.dump_yaml(candidate), encoding="utf-8")
        compare_json_path.write_text(json.dumps(compare_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        expanded_path = None
        compare_json_path = None

    summary = compare_result["summary"]
    result = {
        "name": name,
        "intent": case.get("intent", ""),
        "compact_config": str(compact_path),
        "base_config": str(base_path),
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "baseline_leaf_count": summary["baseline_leaf_count"],
        "candidate_leaf_count": summary["candidate_leaf_count"],
        "diff_count": summary["diff_count"],
        "diff_count_by_kind": summary["diff_count_by_kind"],
        "diff_count_by_top": summary["diff_count_by_top"],
        "ignore_patterns": ignore_patterns,
        "warnings": expansion_summary.get("warnings", []),
        "errors": errors,
    }
    if expanded_path is not None:
        result["expanded_low_level"] = str(expanded_path)
        result["compare_json"] = str(compare_json_path)
    return result


def write_markdown(results: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Compact Config Compatibility Results",
        "",
        "## Summary",
        "",
        f"- Passed: {results['summary']['passed']}",
        f"- Failed: {results['summary']['failed']}",
        f"- Skipped: {results['summary']['skipped']}",
        "",
        "| Status | Case | Intent | Baseline Leaves | Candidate Leaves | Diffs |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in results["cases"]:
        lines.append(
            f"| `{case['status']}` | `{case['name']}` | `{case.get('intent', '')}` | "
            f"{case.get('baseline_leaf_count', '')} | {case.get('candidate_leaf_count', '')} | "
            f"{case.get('diff_count', '')} |"
        )

    failed_or_skipped = [case for case in results["cases"] if case["status"] != "passed"]
    if failed_or_skipped:
        lines.extend(["", "## Diagnostics", ""])
        for case in failed_or_skipped:
            lines.append(f"### {case['name']}")
            for missing in case.get("missing_inputs", []):
                lines.append(f"- Missing input: `{missing}`")
            for error in case.get("errors", []):
                lines.append(f"- Error: {error}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_suite(path: Path) -> dict[str, Any]:
    suite = load_yaml(path)
    if not isinstance(suite, dict):
        raise SuiteError("compatibility suite must be a mapping")
    if suite.get("schema") != SUITE_SCHEMA:
        raise SuiteError(f"compatibility suite must declare schema: {SUITE_SCHEMA}")
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise SuiteError("compatibility suite must contain a cases list")
    return suite


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("compact_compatibility_cases.yaml")),
        help="Compatibility cases YAML file.",
    )
    parser.add_argument(
        "--profiles-dir",
        default=str(repo_root / "tools/config/profiles"),
        help="Compact profile directory.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Optional directory for expanded low-level YAML and per-case comparison JSON.",
    )
    parser.add_argument("--json-out", default="", help="Optional suite JSON report path.")
    parser.add_argument("--markdown-out", default="", help="Optional suite Markdown report path.")
    parser.add_argument("--require-all", action="store_true", help="Treat missing baseline files as failures.")
    parser.add_argument("--allow-empty", action="store_true", help="Return success when no case can run.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cases_path = Path(args.cases).expanduser().resolve()
    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else None

    try:
        suite = load_suite(cases_path)
    except (OSError, yaml.YAMLError, SuiteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    case_results = [
        run_case(
            case,
            suite_dir=cases_path.parent,
            profiles_dir=profiles_dir,
            work_dir=work_dir,
            require_all=args.require_all,
        )
        for case in suite["cases"]
    ]

    status_counts = {
        "passed": sum(1 for case in case_results if case["status"] == "passed"),
        "failed": sum(1 for case in case_results if case["status"] == "failed"),
        "skipped": sum(1 for case in case_results if case["status"] == "skipped"),
    }
    results = {
        "schema": SCHEMA_VERSION,
        "cases_file": str(cases_path),
        "profiles_dir": str(profiles_dir),
        "summary": status_counts,
        "cases": case_results,
    }

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(results, Path(args.markdown_out).expanduser())

    print(
        "compact compatibility: "
        f"passed={status_counts['passed']} failed={status_counts['failed']} skipped={status_counts['skipped']}"
    )
    for case in case_results:
        detail = ""
        if case["status"] == "passed":
            detail = (
                f" leaves={case['baseline_leaf_count']}/{case['candidate_leaf_count']}"
                f" diffs={case['diff_count']}"
            )
        elif case.get("missing_inputs"):
            detail = f" missing={len(case['missing_inputs'])}"
        elif case.get("errors"):
            detail = f" error={case['errors'][0]}"
        print(f"{case['status']}: {case['name']}{detail}")

    if status_counts["failed"]:
        return 1
    if not args.allow_empty and status_counts["passed"] == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
