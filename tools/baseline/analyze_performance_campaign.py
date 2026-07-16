#!/usr/bin/env python3
"""Analyze a controlled paired Citlali performance campaign."""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CAMPAIGN_SCHEMA_VERSION = "citlali-performance-campaign-v1"
RUN_SCHEMA_VERSION = "citlali-performance-run-v1"
RESULT_SCHEMA_VERSION = "citlali-performance-campaign-result-v1"
ROLES = ("baseline", "candidate")


class CampaignError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CampaignError(f"{path}: expected JSON object")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, Any]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "q1": q1,
        "q3": q3,
        "iqr": None if q1 is None or q3 is None else q3 - q1,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0.0):
        return None
    return candidate / baseline


def resolve_run_path(campaign_path: Path, entry: Any) -> Path:
    raw = entry.get("metadata") if isinstance(entry, dict) else entry
    if not isinstance(raw, str) or not raw:
        raise CampaignError("campaign run entry requires a metadata path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (campaign_path.parent / path).resolve()


def load_runs(campaign_path: Path, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    entries = campaign.get("runs")
    if not isinstance(entries, list):
        raise CampaignError("campaign.runs must be a list")
    result = []
    for entry in entries:
        path = resolve_run_path(campaign_path, entry)
        run = load_json(path)
        if run.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CampaignError(f"{path}: unsupported performance run schema")
        run["_metadata_path"] = str(path)
        result.append(run)
    return result


def leaf_map(run: dict[str, Any]) -> dict[str, str]:
    reduction = run.get("reduction") or {}
    rows = reduction.get("config_leaves") or []
    return {
        str(row["path"]): str(row["value_key"])
        for row in rows
        if isinstance(row, dict) and "path" in row and "value_key" in row
    }


def compare_configs(
    baseline: dict[str, Any], candidate: dict[str, Any], ignore: list[str]
) -> dict[str, Any]:
    baseline_map = leaf_map(baseline)
    candidate_map = leaf_map(candidate)
    differences = []
    for path in sorted(set(baseline_map) | set(candidate_map)):
        if any(fnmatch.fnmatch(path, pattern) for pattern in ignore):
            continue
        if baseline_map.get(path) != candidate_map.get(path):
            differences.append(
                {
                    "path": path,
                    "baseline": baseline_map.get(path),
                    "candidate": candidate_map.get(path),
                }
            )
    return {
        "equivalent": not differences,
        "baseline_leaf_count": len(baseline_map),
        "candidate_leaf_count": len(candidate_map),
        "ignored_paths": ignore,
        "differences": differences,
    }


def input_map(run: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates = []
    reduction = run.get("reduction") or {}
    for record in reduction.get("inputs") or []:
        basename = str(record.get("basename", ""))
        if not basename:
            continue
        if basename in result:
            duplicates.append(basename)
        result[basename] = record
    return result, duplicates


def compare_inputs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_map, baseline_duplicates = input_map(baseline)
    candidate_map, candidate_duplicates = input_map(candidate)
    differences = []
    for name in sorted(set(baseline_map) | set(candidate_map)):
        left = baseline_map.get(name)
        right = candidate_map.get(name)
        if left is None or right is None:
            differences.append({"basename": name, "reason": "missing input"})
            continue
        if not left.get("exists") or not right.get("exists"):
            differences.append({"basename": name, "reason": "input unavailable"})
            continue
        if left.get("size_bytes") != right.get("size_bytes"):
            differences.append({"basename": name, "reason": "size differs"})
            continue
        same_path = left.get("path") == right.get("path")
        left_hash = left.get("sha256")
        right_hash = right.get("sha256")
        if not same_path and (not left_hash or not right_hash):
            differences.append(
                {
                    "basename": name,
                    "reason": "different paths without complete hashes",
                }
            )
            continue
        if left_hash and right_hash and left_hash != right_hash:
            differences.append({"basename": name, "reason": "sha256 differs"})
    return {
        "equivalent": not differences and not baseline_duplicates and not candidate_duplicates,
        "input_count": len(baseline_map),
        "baseline_duplicate_basenames": baseline_duplicates,
        "candidate_duplicate_basenames": candidate_duplicates,
        "differences": differences,
    }


def measured_value(run: dict[str, Any], name: str) -> float | None:
    reduction = run.get("reduction") or {}
    gnu_time = run.get("gnu_time") or {}
    values = {
        "citlali_total_log_seconds": reduction.get("citlali_total_log_seconds"),
        "external_wall_seconds": gnu_time.get("elapsed_wall_seconds"),
        "peak_rss_kb": gnu_time.get("maximum_resident_set_kb"),
        "filesystem_inputs": gnu_time.get("filesystem_inputs"),
        "filesystem_outputs": gnu_time.get("filesystem_outputs"),
    }
    value = values[name]
    return float(value) if value is not None else None


def profile_stages(run: dict[str, Any]) -> dict[str, float]:
    reduction = run.get("reduction") or {}
    profile = reduction.get("profile") or {}
    return {
        str(name): float(value)
        for name, value in (profile.get("stage_totals_seconds") or {}).items()
    }


def version_matches(run: dict[str, Any], expected: str) -> bool:
    reduction = run.get("reduction") or {}
    version = str((reduction.get("versions") or {}).get("citlali") or "")
    executable_output = str((run.get("executable") or {}).get("version_output") or "")
    return bool(expected) and (expected in version or expected in executable_output)


def run_errors(
    run: dict[str, Any], campaign: dict[str, Any], expected_version: str
) -> list[str]:
    case_id = run.get("case_id", "unknown")
    errors = []
    if run.get("campaign_id") != campaign["campaign_id"]:
        errors.append(f"{case_id}: campaign id mismatch")
    if run.get("build_type") != campaign.get("build_type"):
        errors.append(f"{case_id}: build type mismatch")
    if run.get("role") not in ROLES:
        errors.append(f"{case_id}: invalid role")
    if run.get("phase") not in {"warmup", "measured"}:
        errors.append(f"{case_id}: invalid phase")
    if run.get("command_exit_code") != 0:
        errors.append(f"{case_id}: command failed")
    if run.get("structure_ok") is not True:
        errors.append(f"{case_id}: reduction-directory structure invalid")
    if run.get("measurement_ok") is not True:
        errors.append(f"{case_id}: GNU time measurement incomplete")
    if not isinstance(run.get("reduction"), dict):
        errors.append(f"{case_id}: reduction evidence missing")
    if not version_matches(run, expected_version):
        errors.append(f"{case_id}: Citlali version does not contain {expected_version!r}")
    versions = (run.get("reduction") or {}).get("versions") or {}
    executable_dependencies = (run.get("executable") or {}).get("dependencies") or {}
    for dependency, expected in (campaign.get("required_dependencies") or {}).items():
        actual = executable_dependencies.get(dependency) or versions.get(dependency)
        if actual != expected:
            errors.append(
                f"{case_id}: {dependency} version {actual!r}; "
                f"expected {expected!r}"
            )
    issue_counts = (run.get("reduction") or {}).get("log_issue_counts") or {}
    serious = sum(
        int(issue_counts.get(name, 0))
        for name in ("error", "fatal", "critical", "traceback")
    )
    if serious:
        errors.append(f"{case_id}: reduction log contains {serious} serious issues")
    for metric in (
        "citlali_total_log_seconds",
        "external_wall_seconds",
        "peak_rss_kb",
    ):
        if measured_value(run, metric) is None:
            errors.append(f"{case_id}: metric {metric} unavailable")
    return errors


def host_signature(run: dict[str, Any]) -> tuple[Any, ...]:
    host = run.get("host") or {}
    storage = run.get("storage") or {}
    return (
        host.get("hostname"),
        host.get("platform"),
        host.get("affinity_cpu_count"),
        json.dumps(host.get("environment") or {}, sort_keys=True),
        storage.get("device"),
    )


def timestamp(run: dict[str, Any], name: str) -> datetime | None:
    value = run.get(name)
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def timestamp_sort_value(run: dict[str, Any]) -> float:
    value = timestamp(run, "started_utc")
    return value.timestamp() if value is not None else float("inf")


def runtime_signature(run: dict[str, Any]) -> str:
    reduction = run.get("reduction") or {}
    value = reduction.get("runtime_signature")
    return json.dumps(value, sort_keys=True, default=str)


def pair_metrics(
    pair_index: int, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    metrics = {}
    for name in (
        "citlali_total_log_seconds",
        "external_wall_seconds",
        "peak_rss_kb",
        "filesystem_inputs",
        "filesystem_outputs",
    ):
        baseline_value = measured_value(baseline, name)
        candidate_value = measured_value(candidate, name)
        metrics[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "ratio": ratio(candidate_value, baseline_value),
        }
    return {
        "pair_index": pair_index,
        "first_role": min(
            (baseline, candidate), key=lambda run: str(run.get("started_utc", ""))
        )["role"],
        "baseline_case_id": baseline["case_id"],
        "candidate_case_id": candidate["case_id"],
        "metrics": metrics,
    }


def analyze(campaign_path: Path) -> dict[str, Any]:
    campaign = load_json(campaign_path)
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignError("unsupported performance campaign schema")
    campaign_id = str(campaign.get("campaign_id") or "")
    if not campaign_id:
        raise CampaignError("campaign_id is required")
    protocol = campaign.get("protocol")
    budgets = campaign.get("budgets")
    roles = campaign.get("roles")
    if not isinstance(protocol, dict) or not isinstance(budgets, dict):
        raise CampaignError("protocol and budgets are required")
    if not isinstance(roles, dict) or any(role not in roles for role in ROLES):
        raise CampaignError("baseline and candidate role definitions are required")
    runs = load_runs(campaign_path, campaign)
    errors: list[str] = []
    case_ids = [str(run.get("case_id", "")) for run in runs]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case_id values must be unique")
    for run in runs:
        role = str(run.get("role", ""))
        expected = str((roles.get(role) or {}).get("version_contains", ""))
        errors.extend(run_errors(run, campaign, expected))
    for role in ROLES:
        hashes = {
            (run.get("executable") or {}).get("sha256")
            for run in runs
            if run.get("role") == role
        }
        if None in hashes or len(hashes) != 1:
            errors.append(f"{role} runs do not use one identified executable")

    warmups = [run for run in runs if run.get("phase") == "warmup"]
    measured = [run for run in runs if run.get("phase") == "measured"]
    if protocol.get("require_warmup_each_role", True):
        for role in ROLES:
            if not any(run.get("role") == role for run in warmups):
                errors.append(f"missing {role} warmup")
    measured_starts = [timestamp(run, "started_utc") for run in measured]
    measured_starts = [value for value in measured_starts if value is not None]
    warmup_ends = [timestamp(run, "ended_utc") for run in warmups]
    warmup_ends = [value for value in warmup_ends if value is not None]
    if measured and (len(measured_starts) != len(measured)):
        errors.append("one or more measured runs have invalid start timestamps")
    if warmups and (len(warmup_ends) != len(warmups)):
        errors.append("one or more warmups have invalid end timestamps")
    if measured_starts and warmup_ends and max(warmup_ends) > min(measured_starts):
        errors.append("all warmups must finish before measured runs begin")
    chronological = sorted(
        runs,
        key=timestamp_sort_value,
    )
    for previous, current in zip(chronological, chronological[1:]):
        previous_end = timestamp(previous, "ended_utc")
        current_start = timestamp(current, "started_utc")
        if previous_end is None or current_start is None:
            continue
        if previous_end > current_start:
            errors.append(
                f"runs {previous.get('case_id')} and {current.get('case_id')} overlap"
            )
    host_signatures = {host_signature(run) for run in runs}
    if protocol.get("require_same_host", True) and len(host_signatures) != 1:
        errors.append("runs do not share one host/platform/CPU-affinity signature")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in measured:
        grouped[int(run.get("pair_index", -1))].append(run)
    pair_indices = sorted(grouped)
    if pair_indices and pair_indices != list(range(len(pair_indices))):
        errors.append(f"measured pair indices are not contiguous: {pair_indices}")
    minimum_pairs = int(protocol.get("minimum_measured_pairs", 3))
    if len(grouped) < minimum_pairs:
        errors.append(f"measured pairs={len(grouped)} below required {minimum_pairs}")

    pair_results = []
    config_results = []
    input_results = []
    first_role = str(protocol.get("first_measured_role", "baseline"))
    ignore_paths = [str(item) for item in protocol.get("config_ignore_paths", [])]
    for ordinal, pair_index in enumerate(sorted(grouped)):
        pair_runs = grouped[pair_index]
        by_role = {role: [run for run in pair_runs if run.get("role") == role] for role in ROLES}
        if any(len(by_role[role]) != 1 for role in ROLES):
            errors.append(f"pair {pair_index}: requires exactly one run per role")
            continue
        baseline = by_role["baseline"][0]
        candidate = by_role["candidate"][0]
        pair = pair_metrics(pair_index, baseline, candidate)
        pair_results.append(pair)
        if protocol.get("require_alternating_first_role", True):
            expected_first = first_role if ordinal % 2 == 0 else (
                "candidate" if first_role == "baseline" else "baseline"
            )
            if pair["first_role"] != expected_first:
                errors.append(
                    f"pair {pair_index}: first role {pair['first_role']!r}; "
                    f"expected {expected_first!r}"
                )
        config_result = compare_configs(baseline, candidate, ignore_paths)
        config_result["pair_index"] = pair_index
        config_results.append(config_result)
        if not config_result["equivalent"]:
            errors.append(f"pair {pair_index}: low-level configs differ")
        input_result = compare_inputs(baseline, candidate)
        input_result["pair_index"] = pair_index
        input_results.append(input_result)
        if not input_result["equivalent"]:
            errors.append(f"pair {pair_index}: input identities differ")
        if protocol.get("require_runtime_signature_match", True):
            left = runtime_signature(baseline)
            right = runtime_signature(candidate)
            if left != "null" and right != "null" and left != right:
                errors.append(f"pair {pair_index}: runtime provenance differs")

    metric_summaries = {}
    for name in (
        "citlali_total_log_seconds",
        "external_wall_seconds",
        "peak_rss_kb",
        "filesystem_inputs",
        "filesystem_outputs",
    ):
        ratios = [
            pair["metrics"][name]["ratio"]
            for pair in pair_results
            if pair["metrics"][name]["ratio"] is not None
        ]
        metric_summaries[name] = summary(ratios)

    stage_names = sorted(
        set().union(*(profile_stages(run).keys() for run in measured))
        if measured
        else set()
    )
    stages = []
    for name in stage_names:
        baseline_values = [
            profile_stages(run)[name]
            for run in measured
            if run.get("role") == "baseline" and name in profile_stages(run)
        ]
        candidate_values = [
            profile_stages(run)[name]
            for run in measured
            if run.get("role") == "candidate" and name in profile_stages(run)
        ]
        baseline_summary = summary(baseline_values)
        candidate_summary = summary(candidate_values)
        stages.append(
            {
                "stage": name,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "median_ratio": ratio(
                    candidate_summary["median"], baseline_summary["median"]
                ),
            }
        )

    budget_failures = []
    pending_qualifications = []
    budget_metrics = {
        "citlali_total_log_seconds": "median_citlali_runtime_ratio_max",
        "peak_rss_kb": "median_peak_rss_ratio_max",
    }
    for metric, budget_name in budget_metrics.items():
        measured_median = metric_summaries[metric]["median"]
        configured_limit = budgets[budget_name]
        if configured_limit is None:
            pending_qualifications.append(
                f"{budget_name}: policy pending; {metric} remains measured"
            )
            continue
        limit = float(configured_limit)
        if measured_median is not None and measured_median > limit:
            budget_failures.append(
                f"{metric} median ratio {measured_median:.6f} exceeds {limit:.6f}"
            )

    profiler = campaign.get("profiler_overhead_evidence") or {}
    if protocol.get("require_profiler_overhead_evidence", False):
        if profiler.get("status") not in {"accepted", "approved_exception"}:
            errors.append("profiler overhead evidence is required")

    protocol_complete = not errors
    verdict = (
        "incomplete"
        if not protocol_complete
        else "rejected"
        if budget_failures
        else "pending_policy"
        if pending_qualifications
        else "accepted"
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "validation_epoch_id": campaign.get("validation_epoch_id"),
        "validation_profile_id": campaign.get("validation_profile_id"),
        "mode": campaign.get("mode"),
        "verdict": verdict,
        "protocol_complete": protocol_complete,
        "protocol_errors": errors,
        "budget_failures": budget_failures,
        "pending_qualifications": pending_qualifications,
        "budgets": budgets,
        "run_count": len(runs),
        "warmup_count": len(warmups),
        "measured_pair_count": len(pair_results),
        "pair_results": pair_results,
        "metric_ratio_summaries": metric_summaries,
        "config_results": config_results,
        "input_results": input_results,
        "stage_summaries": stages,
        "profiler_overhead_evidence": profiler,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        cells = " | ".join(
            value.ljust(widths[col]) for col, value in enumerate(row)
        )
        lines.append(f"| {cells} |")
        if index == 0:
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any], top_stages: int) -> str:
    lines = [
        "# Citlali Performance Campaign",
        "",
        f"- Campaign: `{result['campaign_id']}`",
        f"- Validation epoch: `{result['validation_epoch_id']}`",
        f"- Validation profile: `{result['validation_profile_id']}`",
        f"- Mode: `{result['mode']}`",
        f"- Verdict: **{result['verdict']}**",
        f"- Warmups: `{result['warmup_count']}`",
        f"- Measured pairs: `{result['measured_pair_count']}`",
        "",
        "## Paired Measurements",
        "",
    ]
    rows = [["pair", "first", "Citlali ratio", "external ratio", "RSS ratio"]]
    for pair in result["pair_results"]:
        metrics = pair["metrics"]
        rows.append(
            [
                str(pair["pair_index"]),
                str(pair["first_role"]),
                fmt(metrics["citlali_total_log_seconds"]["ratio"]),
                fmt(metrics["external_wall_seconds"]["ratio"]),
                fmt(metrics["peak_rss_kb"]["ratio"]),
            ]
        )
    lines.append(markdown_table(rows))
    lines.extend(["", "## Ratio Summary", ""])
    summary_rows = [["metric", "median", "IQR", "min", "max"]]
    for name, values in result["metric_ratio_summaries"].items():
        summary_rows.append(
            [
                name,
                fmt(values["median"]),
                fmt(values["iqr"]),
                fmt(values["minimum"]),
                fmt(values["maximum"]),
            ]
        )
    lines.append(markdown_table(summary_rows))

    ranked_stages = sorted(
        result["stage_summaries"],
        key=lambda row: row["candidate"]["median"] or 0.0,
        reverse=True,
    )[:top_stages]
    lines.extend(["", f"## Top {top_stages} Candidate Stages", ""])
    stage_rows = [["stage", "baseline median s", "candidate median s", "ratio"]]
    for row in ranked_stages:
        stage_rows.append(
            [
                row["stage"],
                fmt(row["baseline"]["median"], 3),
                fmt(row["candidate"]["median"], 3),
                fmt(row["median_ratio"]),
            ]
        )
    lines.append(markdown_table(stage_rows))
    lines.extend(["", "## Protocol Errors", ""])
    lines.extend(f"- {error}" for error in result["protocol_errors"])
    if not result["protocol_errors"]:
        lines.append("None.")
    lines.extend(["", "## Budget Failures", ""])
    lines.extend(f"- {failure}" for failure in result["budget_failures"])
    if not result["budget_failures"]:
        lines.append("None.")
    lines.extend(["", "## Pending Qualifications", ""])
    lines.extend(f"- {item}" for item in result["pending_qualifications"])
    if not result["pending_qualifications"]:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--top-stages", type=int, default=20)
    return parser.parse_args(argv)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = analyze(args.campaign.expanduser().resolve())
    except (OSError, json.JSONDecodeError, CampaignError, KeyError, TypeError) as error:
        print(f"performance campaign invalid: {error}", file=sys.stderr)
        return 2
    report = render_markdown(result, args.top_stages)
    if args.json_out:
        write_text(
            args.json_out.expanduser(),
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.report_out:
        write_text(args.report_out.expanduser(), report)
    print(report, end="")
    if result["verdict"] == "accepted":
        return 0
    return 1 if result["verdict"] == "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
