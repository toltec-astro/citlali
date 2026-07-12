#!/usr/bin/env python3
"""Audit a completed Citlali reduction directory without reading large arrays.

This is a fast preflight tool for validation runs.  It answers questions like
"did this run actually write under the refactor path?", "did it finish?", and
"which coarse stages consumed time?" using only the low-level config, logs, and
file inventory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - depends on validation environment
    yaml = None  # type: ignore[assignment]


REDU_RE = re.compile(r"^redu(\d+)$")
TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
VALIDATION_PATH_RE = re.compile(r"/2026-refactor/(?P<mode>[^/]+)/(?P<label>[^/]+)/reduced/?")
PRODUCT_SUFFIXES = {".fits", ".fit", ".nc", ".nc4", ".cdf", ".csv", ".ecsv"}
PROFILE_SIDECAR_NAMES = {"citlali_profile.ecsv"}
PROVENANCE_SIDECARS = {
    "runtime": {
        "filename": "runtime_provenance.yaml",
        "schema_version": "citlali-runtime-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective",),
            ("realized",),
        ),
        "allow_multiple": False,
    },
    "timestream_output": {
        "filename": "timestream_output_provenance.yaml",
        "schema_version": "citlali-timestream-output-provenance-v1",
        "required_paths": (
            ("requested",),
            ("effective",),
            ("realized",),
        ),
        "allow_multiple": True,
    },
    "processed_timestream": {
        "filename": "processed_timestream_provenance.yaml",
        "schema_version": "citlali-processed-timestream-provenance-v1",
        "required_paths": (
            ("initialized",),
            ("requested",),
            ("effective", "config"),
            ("effective", "resolutions"),
            ("realized",),
        ),
        "allow_multiple": False,
    },
}
LOG_MARKERS = (
    ("start", "reduction-local compressed log"),
    ("version", "citlali version:"),
    ("setup", "pipeline setup"),
    ("running_pipeline", "running pipeline"),
    ("first_mapmaking_start", "starting mapmaking"),
    ("first_mapmaking_run", "running mapmaking"),
    ("max_iteration", "max iteration reached"),
    ("ptcdiag_start", "writing ptc diagnostics sidecar chunks"),
    ("apt_start", "writing apt table"),
    ("apt_done", "done writing apt table"),
    ("fitqc_start", "writing beammap fit qc table"),
    ("fitqc_done", "done writing beammap fit qc table"),
    ("split_flag0_done", "beammap split maps (flag=0) have been written"),
    ("split_flag1_done", "beammap split maps (flag=1) have been written"),
    ("index_start", "making index files"),
    ("done", "citlali is done"),
)
INTERVALS = (
    ("total_log", "start", "done"),
    ("startup_to_setup", "start", "setup"),
    ("pipeline_to_first_mapmaking", "running_pipeline", "first_mapmaking_start"),
    ("mapmaking_to_max_iteration", "first_mapmaking_start", "max_iteration"),
    ("ptcdiag_to_apt_start", "ptcdiag_start", "apt_start"),
    ("apt_write", "apt_start", "apt_done"),
    ("fitqc_write", "fitqc_start", "fitqc_done"),
    ("fitqc_to_split_flag0_done", "fitqc_done", "split_flag0_done"),
    ("split_flag1_write", "split_flag0_done", "split_flag1_done"),
    ("index_to_done", "index_start", "done"),
)


@dataclass(frozen=True)
class TimedLine:
    timestamp: datetime
    line: str


def redu_number(path: Path) -> int | None:
    match = REDU_RE.match(path.name)
    return int(match.group(1)) if match else None


def find_latest_redu(root: Path) -> Path:
    candidates = [child for child in root.iterdir() if child.is_dir() and redu_number(child) is not None]
    if not candidates:
        raise FileNotFoundError(f"no reduNN directories under {root}")
    return max(candidates, key=lambda path: redu_number(path) or -1)


def resolve_redu_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    if redu_number(path) is not None:
        return path
    return find_latest_redu(path)


def open_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from handle


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")


def selected_log_line(line: str) -> bool:
    lowered = line.lower()
    return any(text in line for _, text in LOG_MARKERS) or any(
        token in lowered for token in ("fatal", "critical", "error", "traceback")
    )


def collect_labels_from_text(text: str) -> list[dict[str, str]]:
    result = []
    for match in VALIDATION_PATH_RE.finditer(text):
        result.append({"mode": match.group("mode"), "label": match.group("label"), "path": match.group(0)})
    return result


def load_yaml(path: Path) -> Any:
    if yaml is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_nested_key(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        found = []
        for child_key, child_value in value.items():
            if child_key == key:
                found.append(child_value)
            found.extend(find_nested_key(child_value, key))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(find_nested_key(child, key))
        return found
    return []


def has_nested_path(value: Any, path: tuple[str, ...]) -> bool:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def nested_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def processed_provenance_semantic_errors(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_records = {
        "effective.cleaner_mode":
            ("effective", "resolutions", "cleaner_mode"),
        "effective.weighting_source_mask":
            ("effective", "resolutions", "weighting_source_mask"),
        "effective.weighting_dependencies":
            ("effective", "resolutions", "weighting_dependencies"),
        "effective.fruit_loop_iterations":
            ("effective", "resolutions", "fruit_loop_iterations"),
        "effective.fruit_loop_interpolation":
            ("effective", "resolutions", "fruit_loop_interpolation"),
        "realized.source_protection":
            ("realized", "source_protection"),
        "realized.fruit_loop_iterations_completed":
            ("realized", "fruit_loop_iterations_completed"),
        "realized.fruit_loops_converged":
            ("realized", "fruit_loops_converged"),
    }
    try:
        for label, path in required_records.items():
            if nested_value(data, path).get("available") is not True:
                errors.append(f"{label} is unavailable")
        if errors:
            return errors

        requested = nested_value(data, ("requested",))
        effective = nested_value(data, ("effective", "config"))
        resolutions = nested_value(data, ("effective", "resolutions"))
        realized = nested_value(data, ("realized",))

        cleaner = resolutions["cleaner_mode"]["value"]
        if effective["processed_time_chunk"]["clean"]["active"] != cleaner["effective"]:
            errors.append("cleaner resolution does not match effective clean.active")

        source_mask = resolutions["weighting_source_mask"]["value"]
        requested_weighting = requested["processed_time_chunk"]["weighting"]
        effective_weighting = effective["processed_time_chunk"]["weighting"]
        if effective_weighting["source_mask_radius_arcsec"] != source_mask["effective"]:
            errors.append("source-mask resolution does not match effective weighting")
        if source_mask["requested_present"]:
            if requested_weighting["source_mask_radius_arcsec"] != source_mask.get("requested"):
                errors.append("source-mask resolution does not match requested weighting")
        elif not source_mask["inherited_from_cleaning"]:
            errors.append("absent source mask is not marked as inherited")

        weighting = resolutions["weighting_dependencies"]["value"]
        requested_validation = requested_weighting["validation"]["enabled"]
        effective_validation = effective_weighting["validation"]["enabled"]
        expected_validation = bool(
            requested_validation
            or weighting["validation_forced_by_weighting_type"]
        )
        if effective_validation != expected_validation:
            errors.append("weight-validation resolution does not match effective config")
        requested_busy = requested_weighting["busy_row_suppression"]["enabled"]
        effective_busy = effective_weighting["busy_row_suppression"]["enabled"]
        expected_busy = bool(
            requested_busy
            and not weighting["busy_row_disabled_without_second_pass"]
        )
        if effective_busy != expected_busy:
            errors.append("busy-row resolution does not match effective config")

        fruit = resolutions["fruit_loop_iterations"]["value"]
        effective_fruit = effective["fruit_loops"]
        if effective_fruit["max_iters"] != fruit["effective_max_iters"]:
            errors.append("iteration resolution does not match effective max_iters")
        if effective_fruit["save_all_iters"] != fruit["effective_save_all_iters"]:
            errors.append("iteration resolution does not match effective save_all_iters")
        if fruit["forced_single_iteration_while_disabled"] != (
            not requested["fruit_loops"]["enabled"]
        ):
            errors.append("disabled fruit-loop iteration decision is inconsistent")

        source = realized["source_protection"]["value"]
        requested_second_pass = requested["processed_time_chunk"]["flagging"]["second_pass_local"]
        effective_second_pass = effective["processed_time_chunk"]["flagging"]["second_pass_local"]
        expected_activation_request = bool(
            requested_second_pass["enabled"]
            and requested_second_pass["source_protection"]["enabled"]
        )
        if source["processed_activation_requested"] != expected_activation_request:
            errors.append("source-protection request record is inconsistent")
        expected_active = bool(
            expected_activation_request and source["source_aware_reduction"]
        )
        if source["processed_active"] != expected_active:
            errors.append("source-protection realization is inconsistent")
        if effective_second_pass["source_protection"]["active"] != source["processed_active"]:
            errors.append("source-protection realization does not match effective config")

        completed = realized["fruit_loop_iterations_completed"]["value"]
        if not isinstance(completed, int) or completed < 1:
            errors.append("completed iteration count must be a positive integer")
        elif completed > effective_fruit["max_iters"]:
            errors.append("completed iteration count exceeds effective max_iters")
        if not isinstance(realized["fruit_loops_converged"]["value"], bool):
            errors.append("fruit-loop convergence realization must be boolean")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot evaluate processed provenance semantics: {exc}")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_provenance_files(redu: Path, filename: str) -> list[Path]:
    return sorted(redu.rglob(filename))


def audit_provenance_sidecars(
    redu: Path, require_processed: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in PROVENANCE_SIDECARS.items():
        required = name == "processed_timestream" and require_processed
        paths = find_provenance_files(redu, str(spec["filename"]))
        record: dict[str, Any] = {
            "paths": [str(path) for path in paths],
            "count": len(paths),
            "present": bool(paths),
            "required": required,
            "valid": not required,
        }
        if not paths:
            result[name] = record
            continue
        file_records = []
        for path in paths:
            item: dict[str, Any] = {"path": str(path), "valid": False}
            try:
                data = load_yaml(path)
                if not isinstance(data, dict):
                    raise ValueError("provenance root must be a mapping")
                schema_version = data.get("schema_version")
                missing_paths = [
                    ".".join(section)
                    for section in spec["required_paths"]
                    if not has_nested_path(data, section)
                ]
                initialized_ok = data.get("initialized") is not False
                item.update(
                    {
                        "schema_version": schema_version,
                        "schema_ok":
                            schema_version == spec["schema_version"],
                        "missing_paths": missing_paths,
                        "initialized_ok": initialized_ok,
                        "sha256": sha256_file(path),
                    }
                )
                semantic_errors = (
                    processed_provenance_semantic_errors(data)
                    if name == "processed_timestream" and not missing_paths
                    else []
                )
                item["semantic_errors"] = semantic_errors
                item["valid"] = bool(
                    item["schema_ok"]
                    and not missing_paths
                    and initialized_ok
                    and not semantic_errors
                )
            except Exception as exc:
                item["error"] = str(exc)
            file_records.append(item)
        cardinality_ok = bool(spec["allow_multiple"] or len(paths) == 1)
        record.update(
            {
                "files": file_records,
                "cardinality_ok": cardinality_ok,
                "schema_version": file_records[0].get("schema_version"),
                "schema_ok": all(
                    bool(item.get("schema_ok")) for item in file_records
                ),
                "initialized_ok": all(
                    bool(item.get("initialized_ok"))
                    for item in file_records
                ),
                "missing_paths": file_records[0].get("missing_paths", [])
                    if len(file_records) == 1 else {
                        item["path"]: item.get("missing_paths", [])
                        for item in file_records
                        if item.get("missing_paths")
                    },
                "sha256": file_records[0].get("sha256", "")
                    if len(file_records) == 1 else {
                        item["path"]: item.get("sha256", "")
                        for item in file_records
                    },
            }
        )
        record["valid"] = bool(
            cardinality_ok
            and all(bool(item.get("valid")) for item in file_records)
        )
        result[name] = record
    return result


def provenance_ok(audit: dict[str, Any]) -> bool:
    return all(
        bool(record.get("valid"))
        for record in audit.get("provenance", {}).values()
    )


def provenance_hash_summary(record: dict[str, Any]) -> str:
    hashes = record.get("sha256", "")
    if isinstance(hashes, str):
        return hashes
    if not isinstance(hashes, dict):
        return ""
    unique = sorted({str(value) for value in hashes.values() if value})
    prefixes = ", ".join(value[:12] for value in unique)
    return f"{len(hashes)} files; {len(unique)} unique: {prefixes}"


def find_config(path: Path) -> Path | None:
    configs = sorted(path.glob("citlali_o*.yaml"))
    return configs[0] if configs else None


def find_log(path: Path) -> Path | None:
    candidates = sorted(path.glob("citlali.log*"))
    return candidates[0] if candidates else None


def audit_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "error": "no citlali_o*.yaml found"}
    result: dict[str, Any] = {"path": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        result["labels"] = collect_labels_from_text(text)
        data = load_yaml(path)
        if data is not None:
            output_dirs = [str(value) for value in find_nested_key(data, "output_dir")]
            result["output_dirs"] = output_dirs
            result["n_threads"] = find_nested_key(data, "n_threads")
            result["parallel_policy"] = find_nested_key(data, "parallel_policy")
            result["reduction_type"] = find_nested_key(data, "reduction_type")
    except Exception as exc:
        result["error"] = str(exc)
    return result


def audit_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "error": "no citlali.log found"}
    markers: dict[str, TimedLine] = {}
    selected: list[str] = []
    labels: list[dict[str, str]] = []
    counts = Counter()
    mapmaking_starts: list[str] = []
    mapmaking_runs: list[str] = []
    ptc_chunks = 0
    first_ptc_chunk: datetime | None = None
    last_ptc_chunk: datetime | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    try:
        for line in open_text(path):
            stripped = line.rstrip("\n")
            ts = parse_timestamp(stripped)
            if ts is not None:
                first_ts = first_ts or ts
                last_ts = ts
            labels.extend(collect_labels_from_text(stripped))
            lowered = stripped.lower()
            if "fatal" in lowered:
                counts["fatal"] += 1
            if "critical" in lowered:
                counts["critical"] += 1
            if "error" in lowered:
                counts["error"] += 1
            if "traceback" in lowered:
                counts["traceback"] += 1
            if selected_log_line(stripped):
                selected.append(stripped)
            if ts is None:
                continue
            for key, text in LOG_MARKERS:
                if text in stripped:
                    markers.setdefault(key, TimedLine(ts, stripped))
                    if key == "first_mapmaking_start":
                        mapmaking_starts.append(ts.isoformat(sep=" "))
                    if key == "first_mapmaking_run":
                        mapmaking_runs.append(ts.isoformat(sep=" "))
            if "ptc diagnostics sidecar chunk written" in stripped:
                ptc_chunks += 1
                first_ptc_chunk = first_ptc_chunk or ts
                last_ptc_chunk = ts
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}

    intervals: dict[str, float] = {}
    for name, start_key, end_key in INTERVALS:
        start = markers.get(start_key)
        end = markers.get(end_key)
        if start is not None and end is not None:
            intervals[name] = (end.timestamp - start.timestamp).total_seconds()
    if ptc_chunks and first_ptc_chunk is not None and last_ptc_chunk is not None:
        intervals["ptc_first_to_last_chunk"] = (last_ptc_chunk - first_ptc_chunk).total_seconds()
        intervals["ptc_avg_chunk_spacing"] = (
            intervals["ptc_first_to_last_chunk"] / (ptc_chunks - 1) if ptc_chunks > 1 else 0.0
        )

    return {
        "path": str(path),
        "first_timestamp": first_ts.isoformat(sep=" ") if first_ts else None,
        "last_timestamp": last_ts.isoformat(sep=" ") if last_ts else None,
        "markers": {key: {"timestamp": value.timestamp.isoformat(sep=" "), "line": value.line} for key, value in markers.items()},
        "interval_seconds": intervals,
        "mapmaking_starts": mapmaking_starts,
        "mapmaking_runs": mapmaking_runs,
        "ptc_chunk_count": ptc_chunks,
        "issue_counts": dict(sorted(counts.items())),
        "labels": labels,
        "selected_lines": selected[:500],
    }


def product_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return "fits"
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "netcdf"
    if suffix == ".ecsv":
        return "ecsv"
    if suffix == ".csv":
        return "csv"
    if suffix in {".log", ".gz"} and "log" in name:
        return "log"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "other"


def audit_products(path: Path, top: int) -> dict[str, Any]:
    files = [child for child in path.rglob("*") if child.is_file()]
    by_kind = Counter(product_kind(child) for child in files)
    comparable = [
        {
            "path": child.relative_to(path).as_posix(),
            "kind": product_kind(child),
            "size_bytes": child.stat().st_size,
        }
        for child in files
        if child.suffix.lower() in PRODUCT_SUFFIXES or product_kind(child) in {"fits", "netcdf", "ecsv", "csv"}
    ]
    comparable.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    stable_comparable = [
        row for row in comparable
        if Path(str(row["path"])).name not in PROFILE_SIDECAR_NAMES
    ]
    stable_by_kind = Counter(str(row["kind"]) for row in stable_comparable)
    profile_sidecars = [
        row for row in comparable
        if Path(str(row["path"])).name in PROFILE_SIDECAR_NAMES
    ]
    return {
        "file_count": len(files),
        "counts_by_kind": dict(sorted(by_kind.items())),
        "comparable_count": len(comparable),
        "stable_counts_by_kind": dict(sorted(stable_by_kind.items())),
        "stable_comparable_count": len(stable_comparable),
        "profile_sidecars": profile_sidecars,
        "largest_comparable": comparable[:top],
    }


def unique_labels(*sections: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for section in sections:
        for item in section.get("labels", []):
            key = (str(item.get("mode", "")), str(item.get("label", "")))
            if key in seen:
                continue
            seen.add(key)
            result.append({"mode": key[0], "label": key[1]})
    return result


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    redu = resolve_redu_path(Path(args.reduction))
    config = audit_config(find_config(redu))
    log = audit_log(find_log(redu))
    labels = unique_labels(config, log)
    result = {
        "reduction": str(redu),
        "expected_label": args.expected_label,
        "expected_mode": args.expected_mode,
        "labels": labels,
        "label_ok": None,
        "mode_ok": None,
        "config": config,
        "log": log,
        "provenance": audit_provenance_sidecars(
            redu, getattr(args, "require_processed_provenance", False)
        ),
        "products": audit_products(redu, args.top),
    }
    if args.expected_label:
        result["label_ok"] = any(item["label"] == args.expected_label for item in labels)
    if args.expected_mode:
        result["mode_ok"] = any(item["mode"] == args.expected_mode for item in labels)
    return result


def fmt_seconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Citlali Reduction Run Audit",
        "",
        f"- Reduction: `{result['reduction']}`",
        f"- Expected mode: `{result['expected_mode'] or ''}`",
        f"- Expected label: `{result['expected_label'] or ''}`",
        f"- Mode OK: `{result['mode_ok']}`",
        f"- Label OK: `{result['label_ok']}`",
        "",
        "## Identity",
        "",
    ]
    labels = result["labels"]
    if labels:
        for item in labels:
            lines.append(f"- `{item['mode']}/{item['label']}`")
    else:
        lines.append("- No validation path labels found in config or log.")

    config = result["config"]
    lines.extend(["", "## Config", ""])
    lines.append(f"- Config: `{config.get('path')}`")
    for key in ("reduction_type", "parallel_policy", "n_threads", "output_dirs"):
        if key in config:
            lines.append(f"- {key}: `{config[key]}`")
    if "error" in config:
        lines.append(f"- error: `{config['error']}`")

    log = result["log"]
    lines.extend(["", "## Log", ""])
    lines.append(f"- Log: `{log.get('path')}`")
    lines.append(f"- First timestamp: `{log.get('first_timestamp')}`")
    lines.append(f"- Last timestamp: `{log.get('last_timestamp')}`")
    lines.append(f"- PTC chunks: `{log.get('ptc_chunk_count')}`")
    lines.append(f"- Issue counts: `{log.get('issue_counts')}`")
    if "error" in log:
        lines.append(f"- error: `{log['error']}`")

    intervals = log.get("interval_seconds", {})
    if intervals:
        lines.extend(["", "## Timing", "", "| Interval | Seconds |", "| --- | ---: |"])
        for key, value in intervals.items():
            lines.append(f"| `{key}` | {fmt_seconds(value)} |")

    provenance = result["provenance"]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Sidecar | Present | Required | Valid | Schema | SHA-256 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, record in provenance.items():
        lines.append(
            f"| `{name}` | `{record['present']}` | `{record['required']}` | "
            f"`{record['valid']}` | `{record.get('schema_version', '')}` | "
            f"`{provenance_hash_summary(record)}` |"
        )
        if record.get("missing_paths"):
            lines.append(
                f"\nMissing `{name}` paths: `"
                + "`, `".join(record["missing_paths"])
                + "`"
            )
        if record.get("error"):
            lines.append(f"\n`{name}` error: `{record['error']}`")
        for item in record.get("files", []):
            if item.get("semantic_errors"):
                lines.append(
                    f"\n`{name}` semantic errors for `{item['path']}`: `"
                    + "`; `".join(item["semantic_errors"])
                    + "`"
                )

    products = result["products"]
    lines.extend(["", "## Products", ""])
    lines.append(f"- Files: `{products['file_count']}`")
    lines.append(f"- Comparable products: `{products['comparable_count']}`")
    lines.append(f"- Stable comparable products: `{products['stable_comparable_count']}`")
    lines.append(f"- Counts by kind: `{products['counts_by_kind']}`")
    lines.append(f"- Stable counts by kind: `{products['stable_counts_by_kind']}`")
    if products["profile_sidecars"]:
        lines.append(
            "- Profile sidecars: `" +
            ", ".join(str(row["path"]) for row in products["profile_sidecars"]) +
            "`"
        )
    if products["largest_comparable"]:
        lines.extend(["", "Largest comparable products:", ""])
        for row in products["largest_comparable"]:
            lines.append(f"- `{row['path']}` `{row['kind']}` {row['size_bytes']} bytes")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reduction", help="A reduNN directory or a reduced root containing reduNN children.")
    parser.add_argument("--expected-mode", default="", help="Expected validation mode, e.g. beammap.")
    parser.add_argument("--expected-label", default="", help="Expected validation label, e.g. refactor or citlali.")
    parser.add_argument("--top", type=int, default=12, help="Number of largest products to list.")
    parser.add_argument(
        "--require-processed-provenance",
        action="store_true",
        help="Fail unless processed_timestream_provenance.yaml is present and valid.",
    )
    parser.add_argument("--json-out", default="", help="Optional path for machine-readable JSON.")
    parser.add_argument("--report-out", default="", help="Optional path for Markdown output.")
    return parser.parse_args(argv)


def write_text(path: str, text: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = build_audit(args)
    report = render_markdown(result)
    if args.json_out:
        write_text(args.json_out, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.report_out:
        write_text(args.report_out, report)
    print(report, end="")
    if result["label_ok"] is False or result["mode_ok"] is False:
        return 2
    if not provenance_ok(result):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
