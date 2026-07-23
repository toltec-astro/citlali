#!/usr/bin/env python3
"""Analyze a heterogeneous corpus of Citlali Beammap performance runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline import analyze_performance_campaign as performance  # noqa: E402
from tools.baseline import audit_reduction_run  # noqa: E402


CORPUS_SCHEMA_VERSION = "citlali-beammap-performance-corpus-v1"
RESULT_SCHEMA_VERSION = "citlali-beammap-performance-corpus-result-v1"
RUN_SCHEMA_VERSION = "citlali-performance-run-v1"
METRICS = (
    "citlali_total_log_seconds",
    "external_wall_seconds",
    "peak_rss_kb",
    "filesystem_inputs",
    "filesystem_outputs",
)
WORKLOAD_FIELDS = (
    "detector_count",
    "map_count",
    "scan_count",
    "iteration_count",
    "active_map_pass_count",
    "map_scan_pass_count",
    "maximum_detector_tod_samples",
    "output_file_count",
    "output_bytes",
)


class CorpusError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CorpusError(f"{path}: expected JSON object")
    return value


def resolve_path(manifest_path: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CorpusError(f"{label} requires a path")
    path = Path(raw).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def load_run(manifest_path: Path, raw: Any, label: str) -> tuple[dict[str, Any], Path]:
    path = resolve_path(manifest_path, raw, label)
    run = load_json(path)
    if run.get("schema_version") != RUN_SCHEMA_VERSION:
        raise CorpusError(f"{path}: unsupported performance run schema")
    return run, path


def serious_issue_count(run: dict[str, Any]) -> int:
    counts = (run.get("reduction") or {}).get("log_issue_counts") or {}
    return sum(
        int(counts.get(name, 0)) for name in ("error", "fatal", "critical", "traceback")
    )


def basic_run_errors(run: dict[str, Any], label: str) -> list[str]:
    errors = []
    if run.get("command_exit_code") != 0:
        errors.append(f"{label}: reduction command failed")
    if run.get("structure_ok") is not True:
        errors.append(f"{label}: reduction-directory structure invalid")
    if run.get("measurement_ok") is not True:
        errors.append(f"{label}: GNU Time measurement incomplete")
    if not isinstance(run.get("reduction"), dict):
        errors.append(f"{label}: reduction evidence missing")
    serious = serious_issue_count(run)
    if serious:
        errors.append(f"{label}: reduction log contains {serious} serious issues")
    for metric in ("citlali_total_log_seconds", "external_wall_seconds", "peak_rss_kb"):
        if performance.measured_value(run, metric) is None:
            errors.append(f"{label}: metric {metric} unavailable")
    return errors


def release_errors(
    run: dict[str, Any], release: dict[str, Any], label: str
) -> list[str]:
    errors = []
    if run.get("build_type") != release.get("build_type"):
        errors.append(
            f"{label}: build type {run.get('build_type')!r}; "
            f"expected {release.get('build_type')!r}"
        )
    expected_version = str(release.get("version_contains") or "")
    if not expected_version or not performance.version_matches(run, expected_version):
        errors.append(f"{label}: Citlali version does not contain {expected_version!r}")
    versions = (run.get("reduction") or {}).get("versions") or {}
    executable_dependencies = (run.get("executable") or {}).get("dependencies") or {}
    for dependency, expected in (release.get("required_dependencies") or {}).items():
        actual = executable_dependencies.get(dependency) or versions.get(dependency)
        if actual != expected:
            errors.append(
                f"{label}: {dependency} version {actual!r}; expected {expected!r}"
            )
    return errors


def current_evidence_errors(
    run: dict[str, Any], label: str, corpus_id: str
) -> list[str]:
    errors = []
    reduction = run.get("reduction") or {}
    if run.get("campaign_id") != corpus_id:
        errors.append(
            f"{label}: performance campaign id {run.get('campaign_id')!r}; "
            f"expected {corpus_id!r}"
        )
    if not (run.get("executable") or {}).get("sha256"):
        errors.append(f"{label}: executable SHA-256 is unavailable")
    if not reduction.get("config_sha256"):
        errors.append(f"{label}: effective config SHA-256 is unavailable")
    if not isinstance(reduction.get("runtime_signature"), dict):
        errors.append(f"{label}: runtime policy signature is unavailable")
    if not reduction.get("inputs"):
        errors.append(f"{label}: input identity records are unavailable")
    if not (run.get("host") or {}).get("hostname"):
        errors.append(f"{label}: host identity is unavailable")
    if (run.get("storage") or {}).get("device") is None:
        errors.append(f"{label}: storage device identity is unavailable")
    profile = reduction.get("profile") or {}
    if profile.get("present") is not True or not profile.get("stage_totals_seconds"):
        errors.append(f"{label}: profile stage evidence is unavailable")
    for metric in METRICS:
        if performance.measured_value(run, metric) is None:
            errors.append(f"{label}: metric {metric} unavailable")
    return errors


def available_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value") if value.get("available") is True else None
    return value


def resolve_reduction_path(
    manifest_path: Path,
    entry: dict[str, Any],
    run: dict[str, Any],
    metadata_path: Path,
) -> Path | None:
    if entry.get("reduction_path"):
        return resolve_path(manifest_path, entry["reduction_path"], "reduction_path")
    if (metadata_path.parent / "beammap_provenance.yaml").is_file():
        return metadata_path.parent
    raw = (run.get("reduction") or {}).get("path")
    if isinstance(raw, str) and Path(raw).expanduser().is_dir():
        return Path(raw).expanduser().resolve()
    return None


def integer_field(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusError(f"{name} must be a nonnegative integer")
    return value


def extract_workload(
    manifest_path: Path,
    entry: dict[str, Any],
    run: dict[str, Any],
    metadata_path: Path,
    observation_id: int,
) -> tuple[dict[str, int | None], Path | None, list[str]]:
    errors: list[str] = []
    workload: dict[str, int | None] = {name: None for name in WORKLOAD_FIELDS}
    reduction_path = resolve_reduction_path(manifest_path, entry, run, metadata_path)
    provenance_path = (
        reduction_path / "beammap_provenance.yaml"
        if reduction_path is not None
        else None
    )
    if provenance_path is None or not provenance_path.is_file():
        errors.append(
            f"observation {observation_id}: beammap provenance unavailable; "
            "set reduction_path beside the downloaded products"
        )
    else:
        data = audit_reduction_run.load_yaml(provenance_path)
        observations = data.get("observations") if isinstance(data, dict) else None
        if not isinstance(observations, list) or len(observations) != 1:
            errors.append(
                f"observation {observation_id}: Beammap provenance must contain "
                "exactly one observation"
            )
        else:
            observation = observations[0]
            actual = (
                observation.get("obsnum") if isinstance(observation, dict) else None
            )
            if actual != observation_id:
                errors.append(
                    f"observation {observation_id}: provenance identifies {actual!r}"
                )
            else:
                iterations = observation.get("iterations") or []
                workload.update(
                    {
                        "detector_count": observation.get("detector_count"),
                        "map_count": observation.get("map_count"),
                        "scan_count": observation.get("scan_count"),
                        "iteration_count": len(iterations),
                        "active_map_pass_count": sum(
                            int(item.get("active_map_count", 0))
                            * int(item.get("mapmaking_pass_count", 0))
                            for item in iterations
                            if isinstance(item, dict)
                        ),
                        "maximum_detector_tod_samples": available_value(
                            (observation.get("detector_tod") or {}).get(
                                "maximum_sample_count"
                            )
                        ),
                    }
                )
                if (
                    workload["scan_count"] is not None
                    and workload["active_map_pass_count"] is not None
                ):
                    workload["map_scan_pass_count"] = int(workload["scan_count"]) * int(
                        workload["active_map_pass_count"]
                    )
    if reduction_path is not None and reduction_path.is_dir():
        files = [path for path in reduction_path.rglob("*") if path.is_file()]
        workload["output_file_count"] = len(files)
        workload["output_bytes"] = sum(path.stat().st_size for path in files)

    overrides = entry.get("workload") or {}
    if not isinstance(overrides, dict):
        errors.append(
            f"observation {observation_id}: workload override must be an object"
        )
        overrides = {}
    for name, raw_value in overrides.items():
        if name not in WORKLOAD_FIELDS:
            errors.append(
                f"observation {observation_id}: unknown workload field {name!r}"
            )
            continue
        try:
            value = integer_field(raw_value, f"workload.{name}")
        except CorpusError as error:
            errors.append(f"observation {observation_id}: {error}")
            continue
        existing = workload.get(name)
        if existing is not None and existing != value:
            errors.append(
                f"observation {observation_id}: workload override {name}={value} "
                f"conflicts with provenance value {existing}"
            )
            continue
        workload[name] = value
    return workload, reduction_path, errors


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def linear_relationship(xs: list[float], ys: list[float]) -> dict[str, Any]:
    if len(xs) != len(ys):
        raise CorpusError("relationship vectors have different lengths")
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {
            "count": len(xs),
            "pearson_r": None,
            "slope": None,
            "intercept": None,
            "r_squared": None,
        }
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    xx = sum((value - x_mean) ** 2 for value in xs)
    yy = sum((value - y_mean) ** 2 for value in ys)
    xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = xy / xx
    intercept = y_mean - slope * x_mean
    correlation = xy / math.sqrt(xx * yy)
    return {
        "count": len(xs),
        "pearson_r": correlation,
        "slope": slope,
        "intercept": intercept,
        "r_squared": correlation * correlation,
    }


def group_observations(records: list[dict[str, Any]], key) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[str(key(record))].append(record["observation_id"])
    return [
        {"signature": signature, "observation_ids": sorted(observation_ids)}
        for signature, observation_ids in sorted(grouped.items())
    ]


def comparison_result(
    observation_id: int,
    current: dict[str, Any],
    comparison: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    metrics = {}
    for name in METRICS:
        old = performance.measured_value(comparison, name)
        new = performance.measured_value(current, name)
        metrics[name] = {
            "comparison": old,
            "current": new,
            "ratio": performance.ratio(new, old),
        }
    config = performance.compare_configs(comparison, current, [])
    inputs = performance.compare_inputs(comparison, current)
    shared_stages = sorted(
        set(performance.profile_stages(comparison))
        & set(performance.profile_stages(current))
    )
    stages = [
        {
            "stage": stage,
            "comparison_seconds": performance.profile_stages(comparison)[stage],
            "current_seconds": performance.profile_stages(current)[stage],
            "ratio": performance.ratio(
                performance.profile_stages(current)[stage],
                performance.profile_stages(comparison)[stage],
            ),
        }
        for stage in shared_stages
    ]
    return {
        "observation_id": observation_id,
        "label": label,
        "metrics": metrics,
        "config_equivalent": config["equivalent"],
        "config_differences": config["differences"],
        "inputs_equivalent": inputs["equivalent"],
        "input_differences": inputs["differences"],
        "shared_stage_count": len(stages),
        "stages": stages,
    }


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise CorpusError("unsupported Beammap corpus schema")
    corpus_id = str(manifest.get("corpus_id") or "")
    if not corpus_id:
        raise CorpusError("corpus_id is required")
    release = manifest.get("release")
    protocol = manifest.get("protocol")
    entries = manifest.get("observations")
    if not isinstance(release, dict) or not isinstance(protocol, dict):
        raise CorpusError("release and protocol objects are required")
    if not isinstance(entries, list):
        raise CorpusError("observations must be a list")
    expected_raw = protocol.get("expected_observation_ids")
    if not isinstance(expected_raw, list):
        raise CorpusError("protocol.expected_observation_ids must be a list")
    expected = []
    for raw in expected_raw:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise CorpusError("expected observation IDs must be positive integers")
        expected.append(raw)

    errors: list[str] = []
    if not release.get("label"):
        errors.append("release label is required")
    if not release.get("version_contains"):
        errors.append("release version token is required")
    if not release.get("build_type"):
        errors.append("release build type is required")
    required_dependencies = release.get("required_dependencies")
    if not isinstance(required_dependencies, dict) or not required_dependencies:
        errors.append("release dependency identity is required")
    elif any(not str(value) for value in required_dependencies.values()):
        errors.append("release dependency revisions must not be empty")
    if not expected:
        errors.append("expected observation list is empty")
    if len(expected) != len(set(expected)):
        errors.append("expected observation IDs are duplicated")

    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"observation entry {index} must be an object")
            continue
        observation_id = entry.get("observation_id")
        if (
            isinstance(observation_id, bool)
            or not isinstance(observation_id, int)
            or observation_id <= 0
        ):
            errors.append(f"observation entry {index} has invalid observation_id")
            continue
        if observation_id in seen:
            errors.append(f"observation {observation_id}: duplicate current record")
            continue
        seen.add(observation_id)
        label = f"observation {observation_id}"
        try:
            run, metadata_path = load_run(
                manifest_path, entry.get("metadata"), f"{label} metadata"
            )
        except (OSError, json.JSONDecodeError, CorpusError) as error:
            errors.append(str(error))
            continue
        run_errors = basic_run_errors(run, label)
        run_errors.extend(release_errors(run, release, label))
        run_errors.extend(current_evidence_errors(run, label, corpus_id))
        workload, reduction_path, workload_errors = extract_workload(
            manifest_path,
            entry,
            run,
            metadata_path,
            observation_id,
        )
        run_errors.extend(workload_errors)
        for name in (
            "detector_count",
            "map_count",
            "scan_count",
            "iteration_count",
            "active_map_pass_count",
            "map_scan_pass_count",
            "output_file_count",
            "output_bytes",
        ):
            if workload.get(name) in (None, 0):
                run_errors.append(f"{label}: workload {name} is unavailable")
        errors.extend(run_errors)
        metrics = {name: performance.measured_value(run, name) for name in METRICS}
        runtime = metrics["citlali_total_log_seconds"]
        scan_count = workload.get("scan_count")
        map_scan_pass_count = workload.get("map_scan_pass_count")
        normalized = {
            "citlali_seconds_per_scan": (
                runtime / scan_count if runtime is not None and scan_count else None
            ),
            "citlali_seconds_per_billion_map_scan_passes": (
                runtime * 1_000_000_000.0 / map_scan_pass_count
                if runtime is not None and map_scan_pass_count
                else None
            ),
        }
        record = {
            "observation_id": observation_id,
            "metadata_path": str(metadata_path),
            "reduction_path": str(reduction_path) if reduction_path else None,
            "notes": entry.get("notes"),
            "valid": not run_errors,
            "errors": run_errors,
            "metrics": metrics,
            "workload": workload,
            "normalized": normalized,
            "config_sha256": (run.get("reduction") or {}).get("config_sha256"),
            "runtime_signature": (run.get("reduction") or {}).get("runtime_signature"),
            "executable_sha256": (run.get("executable") or {}).get("sha256"),
            "host": (run.get("host") or {}).get("hostname"),
            "storage_device": (run.get("storage") or {}).get("device"),
            "profile_stages": performance.profile_stages(run),
            "_run": run,
        }
        records.append(record)

        comparison_labels: set[str] = set()
        raw_comparisons = entry.get("comparisons") or []
        if not isinstance(raw_comparisons, list):
            errors.append(f"{label}: comparisons must be a list")
            continue
        for comparison_index, comparison_entry in enumerate(raw_comparisons):
            if not isinstance(comparison_entry, dict):
                errors.append(
                    f"{label}: comparison {comparison_index} must be an object"
                )
                continue
            comparison_label = str(comparison_entry.get("label") or "")
            if not comparison_label:
                errors.append(f"{label}: comparison {comparison_index} needs a label")
                continue
            if comparison_label in comparison_labels:
                errors.append(
                    f"{label}: duplicate comparison label {comparison_label!r}"
                )
                continue
            comparison_labels.add(comparison_label)
            try:
                comparison_run, comparison_path = load_run(
                    manifest_path,
                    comparison_entry.get("metadata"),
                    f"{label} comparison {comparison_label}",
                )
            except (OSError, json.JSONDecodeError, CorpusError) as error:
                errors.append(str(error))
                continue
            comparison_errors = basic_run_errors(
                comparison_run, f"{label} comparison {comparison_label}"
            )
            _, _, identity_errors = extract_workload(
                manifest_path,
                comparison_entry,
                comparison_run,
                comparison_path,
                observation_id,
            )
            comparison_errors.extend(identity_errors)
            errors.extend(comparison_errors)
            if not comparison_errors:
                comparisons.append(
                    comparison_result(
                        observation_id,
                        run,
                        comparison_run,
                        comparison_label,
                    )
                )

    expected_set = set(expected)
    observed_set = {record["observation_id"] for record in records}
    for observation_id in sorted(expected_set - observed_set):
        errors.append(f"expected observation {observation_id} has no current record")
    for observation_id in sorted(observed_set - expected_set):
        errors.append(f"observation {observation_id} is not in the expected list")

    valid_records = [record for record in records if record["valid"]]
    if valid_records and protocol.get("require_single_executable", True):
        hashes = {record["executable_sha256"] for record in valid_records}
        if None in hashes or len(hashes) != 1:
            errors.append("current records do not use one identified executable")
    if valid_records and protocol.get("require_single_runtime_signature", True):
        signatures = {
            json.dumps(record["runtime_signature"], sort_keys=True, default=str)
            for record in valid_records
        }
        if "null" in signatures or len(signatures) != 1:
            errors.append("current records do not share one runtime signature")

    metric_summaries = {
        name: performance.summary(
            [
                value
                for record in valid_records
                if (value := finite_number(record["metrics"][name])) is not None
            ]
        )
        for name in METRICS
    }
    normalized_fields = (
        "citlali_seconds_per_scan",
        "citlali_seconds_per_billion_map_scan_passes",
    )
    normalized_summaries = {
        name: performance.summary(
            [
                value
                for record in valid_records
                if (value := finite_number(record["normalized"][name])) is not None
            ]
        )
        for name in normalized_fields
    }

    relationships = []
    for workload_name in WORKLOAD_FIELDS:
        for metric_name in ("citlali_total_log_seconds", "peak_rss_kb"):
            pairs = [
                (
                    finite_number(record["workload"][workload_name]),
                    finite_number(record["metrics"][metric_name]),
                )
                for record in valid_records
            ]
            complete = [(x, y) for x, y in pairs if x is not None and y is not None]
            xs = [x for x, _ in complete]
            ys = [y for _, y in complete]
            relationships.append(
                {
                    "workload": workload_name,
                    "metric": metric_name,
                    **linear_relationship(xs, ys),
                }
            )

    stage_names = sorted(
        set().union(*(record["profile_stages"].keys() for record in valid_records))
        if valid_records
        else set()
    )
    stage_summaries = [
        {
            "stage": name,
            **performance.summary(
                [
                    record["profile_stages"][name]
                    for record in valid_records
                    if name in record["profile_stages"]
                ]
            ),
        }
        for name in stage_names
    ]
    rankings = {}
    for name, source in (
        ("citlali_total_log_seconds", "metrics"),
        ("peak_rss_kb", "metrics"),
        ("citlali_seconds_per_scan", "normalized"),
        ("citlali_seconds_per_billion_map_scan_passes", "normalized"),
    ):
        rankings[name] = [
            {
                "observation_id": record["observation_id"],
                "value": record[source][name],
            }
            for record in sorted(
                valid_records,
                key=lambda item: finite_number(item[source][name]) or float("-inf"),
                reverse=True,
            )
            if finite_number(record[source][name]) is not None
        ]

    executable_groups = group_observations(
        valid_records, lambda record: record["executable_sha256"]
    )
    runtime_groups = group_observations(
        valid_records,
        lambda record: json.dumps(
            record["runtime_signature"], sort_keys=True, default=str
        ),
    )
    config_groups = group_observations(
        valid_records, lambda record: record["config_sha256"]
    )
    host_groups = group_observations(valid_records, lambda record: record["host"])
    storage_groups = group_observations(
        valid_records, lambda record: record["storage_device"]
    )

    public_records = [
        {key: value for key, value in record.items() if key != "_run"}
        for record in records
    ]
    complete = (
        bool(expected)
        and observed_set == expected_set
        and len(valid_records) == len(expected)
        and not errors
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "release": release,
        "verdict": "complete" if complete else "incomplete",
        "complete": complete,
        "errors": errors,
        "expected_observation_count": len(expected),
        "current_record_count": len(records),
        "valid_current_record_count": len(valid_records),
        "comparison_count": len(comparisons),
        "records": public_records,
        "comparisons": comparisons,
        "metric_summaries": metric_summaries,
        "normalized_summaries": normalized_summaries,
        "relationships": relationships,
        "stage_summaries": stage_summaries,
        "rankings": rankings,
        "identity_groups": {
            "executables": executable_groups,
            "runtime_signatures": runtime_groups,
            "configs": config_groups,
            "hosts": host_groups,
            "storage_devices": storage_groups,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append(
            "| "
            + " | ".join(
                value.ljust(widths[column]) for column, value in enumerate(row)
            )
            + " |"
        )
        if index == 0:
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any], top: int) -> str:
    lines = [
        "# Citlali Beammap Performance Corpus",
        "",
        f"- Corpus: `{result['corpus_id']}`",
        f"- Release: `{result['release'].get('label')}`",
        f"- Verdict: **{result['verdict']}**",
        f"- Expected observations: `{result['expected_observation_count']}`",
        f"- Valid current records: `{result['valid_current_record_count']}`",
        f"- Same-observation comparisons: `{result['comparison_count']}`",
        "",
        "## Current Population",
        "",
    ]
    rows = [
        [
            "obs",
            "scans",
            "detectors",
            "iterations",
            "Citlali s",
            "RSS GB",
            "s/scan",
        ]
    ]
    for record in sorted(result["records"], key=lambda row: row["observation_id"]):
        workload = record["workload"]
        metrics = record["metrics"]
        rows.append(
            [
                str(record["observation_id"]),
                str(workload["scan_count"] or ""),
                str(workload["detector_count"] or ""),
                str(workload["iteration_count"] or ""),
                fmt(metrics["citlali_total_log_seconds"]),
                fmt(
                    metrics["peak_rss_kb"] / (1024.0 * 1024.0)
                    if metrics["peak_rss_kb"] is not None
                    else None
                ),
                fmt(record["normalized"]["citlali_seconds_per_scan"]),
            ]
        )
    lines.append(markdown_table(rows))

    lines.extend(["", "## Population Summary", ""])
    summary_rows = [["metric", "count", "median", "IQR", "min", "max"]]
    for name, values in {
        **result["metric_summaries"],
        **result["normalized_summaries"],
    }.items():
        summary_rows.append(
            [
                name,
                str(values["count"]),
                fmt(values["median"]),
                fmt(values["iqr"]),
                fmt(values["minimum"]),
                fmt(values["maximum"]),
            ]
        )
    lines.append(markdown_table(summary_rows))

    lines.extend(["", f"## Highest-Cost Observations (Top {top})", ""])
    ranking_rows = [["metric", "observation", "value"]]
    for name, rows_for_metric in result["rankings"].items():
        for row in rows_for_metric[:top]:
            ranking_rows.append([name, str(row["observation_id"]), fmt(row["value"])])
    lines.append(markdown_table(ranking_rows))

    informative = [
        row for row in result["relationships"] if row["pearson_r"] is not None
    ]
    informative.sort(key=lambda row: abs(row["pearson_r"]), reverse=True)
    lines.extend(["", f"## Workload Relationships (Top {top})", ""])
    relation_rows = [["workload", "metric", "n", "Pearson r", "R2", "slope"]]
    for row in informative[:top]:
        relation_rows.append(
            [
                row["workload"],
                row["metric"],
                str(row["count"]),
                fmt(row["pearson_r"], 4),
                fmt(row["r_squared"], 4),
                fmt(row["slope"], 6),
            ]
        )
    lines.append(markdown_table(relation_rows))

    lines.extend(["", "## Same-Observation Comparisons", ""])
    comparison_rows = [["obs", "label", "Citlali ratio", "RSS ratio", "config same"]]
    for row in result["comparisons"]:
        comparison_rows.append(
            [
                str(row["observation_id"]),
                row["label"],
                fmt(row["metrics"]["citlali_total_log_seconds"]["ratio"], 4),
                fmt(row["metrics"]["peak_rss_kb"]["ratio"], 4),
                str(row["config_equivalent"]),
            ]
        )
    lines.append(markdown_table(comparison_rows))

    lines.extend(["", "## Completeness Errors", ""])
    lines.extend(f"- {error}" for error in result["errors"])
    if not result["errors"]:
        lines.append("None.")
    lines.extend(
        [
            "",
            "Rankings and relationships are diagnostic. They do not discard "
            "outliers or establish causation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args(argv)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = analyze(args.manifest.expanduser().resolve())
    except (
        OSError,
        json.JSONDecodeError,
        CorpusError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Beammap corpus invalid: {error}", file=sys.stderr)
        return 2
    report = render_markdown(result, args.top)
    if args.json_out:
        write_text(
            args.json_out.expanduser(),
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.report_out:
        write_text(args.report_out.expanduser(), report)
    print(report, end="")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
