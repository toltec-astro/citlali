#!/usr/bin/env python3
"""Record the immutable evidence state of the stopped EL25 confirmation.

This recorder is intentionally separate from the preregistered confirmation
runner.  It never invokes AM, imports the confirmation runner, loads spectral
arrays, integrates a passband, or evaluates an operator.  It holds the cache's
shared POSIX lock while validating and hashing the stopped cache.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import struct
import tempfile
from collections import Counter
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

CACHE_BASENAME = "sci_cal_001_am12_el25_confirmation_v1_20260802_root"
DEFAULT_CACHE_DIR = Path("/private/tmp") / CACHE_BASENAME
CACHE_LOCK_NAME = ".am12_el25_confirmation.lock"

CONTEXT_NAME = "execution_context.json"
CONTEXT_SHA256 = "a867df7b05ea590c498e41932bb1b3f9520e635d2534f7c8fcc539cfd4a12ecf"
PREREGISTRATION_NAME = "am12_el25_confirmation_preregistration.json"
PROTOCOL_NAME = "AM12_EL25_CONFIRMATION_PROTOCOL.md"
RESULT_SCHEMA_NAME = "am12_el25_confirmation_result.schema.json"
RUNNER_NAME = "run_am12_el25_confirmation_study.py"

DECISION_NAME = "am12_el25_confirmation_failure_decision.json"
REPORT_NAME = "AM12_EL25_CONFIRMATION_FAILURE_REPORT.md"

PACKAGE_ID = "SCI-CAL-001"
STUDY_ID = "SCI-CAL-001-AM12-EL25-CONFIRMATION-001"
RECORD_ID = "SCI-CAL-001-AM12-EL25-CONFIRMATION-001-FAILURE-001"
PREREGISTRATION_COMMIT = "fe3b3a1f7885334c50337382d97a84121dbe57c0"
PREREGISTRATION_PARENT = "f4014d3669b94b1eceb8158da7993737efc908f2"

FULL_GRID_STAGE = "confirmation_trisection_integer_elevation_full_grid"
ANCHOR_STAGE = "anchor_225ghz_el80"
EXPECTED_CASE_COUNT = 16
EXPECTED_ELEVATIONS = tuple(range(25, 81))
EXPECTED_FULL_GRID_COUNT = 896
OBSERVED_COMPLETE_CASE_COUNT = 12
OBSERVED_FULL_GRID_COUNT = 672
OBSERVED_RAW_AND_SIDECAR_COUNT = 1953
OBSERVED_TRACE_COUNT = 13

FAILING_COORDINATE = "q50_q75_trisect_2"
FAILING_PROFILE = "LMT_DJF_50"
FAILING_TRACE_NAME = f"{FAILING_COORDINATE}_{FAILING_PROFILE}.json"
FAILING_TRACE_EVALUATION_COUNT = 99
X80_DECIMAL = Decimal("1.01538872688246729")
GUARD_THRESHOLD_SOURCE_LITERAL = "5.0e-17"

INVENTORY_ALGORITHM = "sha256(relative_path NUL file_sha256_bytes NUL)"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compact_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f17(value: float) -> str:
    return f"{value:.17e}"


def exact_float_decimal(value: float) -> str:
    return str(Decimal.from_float(value))


def relative_entry(root: Path, path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def aggregate_entries(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry["sha256"]))
        digest.update(b"\0")
        total_bytes += int(entry["size_bytes"])
    return {
        "algorithm": INVENTORY_ALGORITHM,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }


def inventory_paths(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    entries = [relative_entry(root, path) for path in sorted(paths)]
    return {**aggregate_entries(entries), "files": entries}


def direct_regular_files(directory: Path) -> list[Path]:
    require(directory.is_dir() and not directory.is_symlink(), f"missing directory: {directory}")
    result: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        require(not path.is_symlink(), f"symlink is not admitted in cache: {path}")
        require(path.is_file(), f"unexpected non-file cache entry: {path}")
        result.append(path)
    return result


@contextmanager
def shared_cache_lock(cache_dir: Path) -> Iterator[None]:
    lock_path = cache_dir / CACHE_LOCK_NAME
    require(lock_path.is_file() and not lock_path.is_symlink(), f"missing cache lock: {lock_path}")
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("confirmation cache is locked by a writer") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_canonical_json(
    path: Path, *, compact: bool = False, preserve_key_order: bool = False
) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON: {path}") from error
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    if compact:
        expected = compact_json_bytes(payload)
    elif preserve_key_order:
        expected = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    else:
        expected = json_bytes(payload)
    require(raw == expected, f"noncanonical JSON: {path}")
    return raw, payload


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON: {path}") from error
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def validate_context(
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    context_path = cache_dir / CONTEXT_NAME
    raw, context = load_canonical_json(context_path, compact=True)
    require(sha256_bytes(raw) == CONTEXT_SHA256, "execution-context SHA-256 mismatch")
    require(context.get("package_id") == PACKAGE_ID, "execution-context package mismatch")
    require(context.get("study_id") == STUDY_ID, "execution-context study mismatch")
    require(
        context.get("schema_version")
        == "sci-cal-001-am12-el25-confirmation-study-v1-execution-context-v1",
        "execution-context schema mismatch",
    )
    require(
        context.get("authorization")
        == "one_bounded_numerical_confirmation_no_operator_or_domain_authorization",
        "execution-context authorization mismatch",
    )
    binding = context.get("preregistration_git_binding", {})
    require(
        binding.get("preregistration_commit") == PREREGISTRATION_COMMIT,
        "preregistration commit mismatch",
    )
    require(
        binding.get("preregistration_parent") == PREREGISTRATION_PARENT,
        "preregistration parent mismatch",
    )
    parameters = context.get("execution_parameters", {})
    require(
        parameters.get("external_cache_basename") == CACHE_BASENAME,
        "execution-context cache basename mismatch",
    )
    require(
        parameters.get("elevations_deg") == list(EXPECTED_ELEVATIONS),
        "execution-context elevation plan mismatch",
    )
    require(parameters.get("jobs") == 8, "execution-context job count mismatch")
    require(
        parameters.get("omp_threads_per_process") == 1,
        "execution-context OMP count mismatch",
    )
    require(
        parameters.get("cache_shard_count") == 8,
        "execution-context shard count mismatch",
    )
    security = context.get("security", {})
    require(
        security
        == {
            "beammap_or_fts_access": False,
            "citlali_application_code_modified": False,
            "network_access": False,
            "sibling_repository_writes": False,
            "unity_access": False,
        },
        "execution-context security boundary mismatch",
    )

    controls = context.get("controls", {})
    expected_control_names = {
        "preregistration": PREREGISTRATION_NAME,
        "protocol": PROTOCOL_NAME,
        "result_schema": RESULT_SCHEMA_NAME,
        "runner": RUNNER_NAME,
    }
    for role, filename in expected_control_names.items():
        record = controls.get(role, {})
        require(record.get("filename") == filename, f"control filename mismatch: {role}")
        require(
            record.get("sha256") == sha256_path(PACKAGE_DIR / filename),
            f"control SHA-256 mismatch: {role}",
        )

    preregistration = load_json_object(PACKAGE_DIR / PREREGISTRATION_NAME)
    require(
        preregistration["expected_coverage"]["profile_scale_case_count"]
        == EXPECTED_CASE_COUNT,
        "preregistration case count mismatch",
    )
    require(
        preregistration["expected_coverage"]["full_direct_am_grid_count"]
        == EXPECTED_FULL_GRID_COUNT,
        "preregistration full-grid count mismatch",
    )
    require(
        preregistration["elevation_selection"]["elevations_deg"]
        == list(EXPECTED_ELEVATIONS),
        "preregistration elevation plan mismatch",
    )
    expected_cases = [
        (item["coordinate_id"], item["truth_profile"])
        for item in context.get("scale_cases", [])
    ]
    require(len(expected_cases) == EXPECTED_CASE_COUNT, "context case count mismatch")
    require(len(set(expected_cases)) == EXPECTED_CASE_COUNT, "duplicate context case")

    prereg_cases: list[tuple[str, str]] = []
    by_interval = preregistration["profile_allocation"]["by_interval"]
    for coordinate in preregistration["opacity_selection"]["coordinates"]:
        for profile in by_interval[coordinate["interval"]]:
            prereg_cases.append((coordinate["coordinate_id"], profile))
    require(expected_cases == prereg_cases, "context/preregistration case-plan mismatch")

    context_coordinates = {
        item["coordinate_id"]: item for item in context.get("coordinate_plan", [])
    }
    prereg_coordinates = {
        item["coordinate_id"]: item
        for item in preregistration["opacity_selection"]["coordinates"]
    }
    require(set(context_coordinates) == set(prereg_coordinates), "coordinate key mismatch")
    for coordinate_id, frozen in prereg_coordinates.items():
        observed = context_coordinates[coordinate_id]
        for key in (
            "requested_tau225_exact",
            "target_transmission_literal",
            "achieved_tau225_exact",
        ):
            require(observed.get(key) == frozen.get(key), f"coordinate mismatch: {coordinate_id}/{key}")

    require(
        context["atmosphere"]["executable"]["sha256"]
        == preregistration["atmosphere_execution"]["executable_sha256"],
        "AM executable binding mismatch",
    )
    context_profiles = {
        item["profile"]: item["sha256"]
        for item in context["atmosphere"]["profiles"]
    }
    require(
        context_profiles == preregistration["profile_allocation"]["profile_sha256"],
        "profile binding mismatch",
    )
    require(
        context["passband_set"]["passband_set_id"]
        == preregistration["passband_set"]["passband_set_id"],
        "passband-set binding mismatch",
    )

    context_entry = {
        "path": CONTEXT_NAME,
        "size_bytes": len(raw),
        "sha256": CONTEXT_SHA256,
    }
    return context, preregistration, context_entry


def validate_sidecars(
    cache_dir: Path,
    context: dict[str, Any],
    raw_inventory: dict[str, Any],
    sidecar_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    raw_entries = {Path(item["path"]).name: item for item in raw_inventory["files"]}
    raw_ids = {name.removesuffix(".txt") for name in raw_entries}
    sidecar_ids = {path.name.removesuffix(".run.json") for path in sidecar_paths}
    require(raw_ids == sidecar_ids, "raw-output/sidecar identity anti-join failed")

    context_sha256 = CONTEXT_SHA256
    expected_cases = {
        (item["coordinate_id"], item["truth_profile"])
        for item in context["scale_cases"]
    }
    stage_counts: Counter[str] = Counter()
    return_code_counts: Counter[str] = Counter()
    warning_histogram: Counter[str] = Counter()
    full_grid: dict[tuple[str, str, int], dict[str, Any]] = {}
    anchor_runs: dict[tuple[str, str, str], dict[str, Any]] = {}

    for path in sidecar_paths:
        _, sidecar = load_canonical_json(path)
        cache_id = path.name.removesuffix(".run.json")
        require(sidecar.get("cache_id") == cache_id, f"sidecar cache ID mismatch: {path.name}")
        require(
            sidecar.get("execution_context_sha256") == context_sha256,
            f"sidecar context mismatch: {cache_id}",
        )
        raw_name = f"{cache_id}.txt"
        require(raw_name in raw_entries, f"missing raw output: {cache_id}")
        raw_entry = raw_entries[raw_name]
        require(
            sidecar.get("combined_output_path_relative_to_cache")
            == f"raw_outputs/{raw_name}",
            f"sidecar raw path mismatch: {cache_id}",
        )
        require(
            sidecar.get("combined_output_sha256") == raw_entry["sha256"],
            f"sidecar raw SHA-256 mismatch: {cache_id}",
        )
        request = sidecar.get("request", {})
        stage = request.get("stage")
        target = request.get("target")
        profile = request.get("profile")
        require((target, profile) in expected_cases, f"unexpected sidecar case: {cache_id}")
        require(stage in {ANCHOR_STAGE, FULL_GRID_STAGE}, f"unexpected stage: {cache_id}")
        stage_counts[stage] += 1
        return_code_counts[str(sidecar.get("return_code"))] += 1

        if stage == ANCHOR_STAGE:
            require(
                request.get("f_min_centi_ghz") == 22499
                and request.get("f_max_centi_ghz") == 22501
                and request.get("step_mhz") == 10
                and request.get("zenith_angle_deg") == 10,
                f"anchor request mismatch: {cache_id}",
            )
            require(
                sidecar.get("return_code") == 0
                and sidecar.get("numeric_row_count") == 3
                and sidecar.get("unresolved_line_warning_count") is None
                and sidecar.get("unresolved_column_warning_line_count") == 0
                and sidecar.get("unresolved_summary_warning_line_count") == 0
                and sidecar.get("other_warning_line_count") == 0
                and sidecar.get("error_line_count") == 0,
                f"anchor status contract mismatch: {cache_id}",
            )
            key = (target, profile, request.get("scale_decimal"))
            require(key not in anchor_runs, f"duplicate anchor request: {key}")
            anchor_runs[key] = sidecar
            continue

        require(
            request.get("f_min_centi_ghz") == 0
            and request.get("f_max_centi_ghz") == 50000
            and request.get("step_mhz") == 10,
            f"full-grid frequency request mismatch: {cache_id}",
        )
        elevation = 90 - int(request.get("zenith_angle_deg"))
        require(elevation in EXPECTED_ELEVATIONS, f"full-grid elevation mismatch: {cache_id}")
        require(sidecar.get("numeric_row_count") == 50001, f"full-grid row count mismatch: {cache_id}")
        return_code = sidecar.get("return_code")
        if return_code == 0:
            require(
                sidecar.get("unresolved_line_warning_count") is None
                and sidecar.get("unresolved_column_warning_line_count") == 0
                and sidecar.get("unresolved_summary_warning_line_count") == 0
                and sidecar.get("other_warning_line_count") == 0
                and sidecar.get("error_line_count") == 0,
                f"status-0 full-grid warning contract mismatch: {cache_id}",
            )
        elif return_code == 1:
            warning_count = sidecar.get("unresolved_line_warning_count")
            require(
                warning_count in {86, 87, 88}
                and int(sidecar.get("unresolved_column_warning_line_count", 0)) > 0
                and sidecar.get("unresolved_summary_warning_line_count") == 1
                and sidecar.get("other_warning_line_count") == 0
                and sidecar.get("error_line_count") == 0,
                f"WARN-001 sidecar contract mismatch: {cache_id}",
            )
            warning_histogram[str(warning_count)] += 1
        else:
            raise RuntimeError(f"unadmitted full-grid status: {cache_id}/{return_code}")
        key = (target, profile, elevation)
        require(key not in full_grid, f"duplicate full-grid request: {key}")
        full_grid[key] = sidecar

    require(
        stage_counts == {ANCHOR_STAGE: 1281, FULL_GRID_STAGE: OBSERVED_FULL_GRID_COUNT},
        f"stage counts changed: {dict(stage_counts)}",
    )
    require(len(anchor_runs) == 1281, "anchor identity count mismatch")
    require(len(full_grid) == OBSERVED_FULL_GRID_COUNT, "full-grid identity count mismatch")
    validation = {
        "raw_output_count": len(raw_ids),
        "execution_sidecar_count": len(sidecar_ids),
        "matched_pair_count": len(raw_ids & sidecar_ids),
        "raw_without_sidecar_count": len(raw_ids - sidecar_ids),
        "sidecar_without_raw_count": len(sidecar_ids - raw_ids),
        "all_sidecar_raw_sha256_bindings_match": True,
        "stage_counts": dict(sorted(stage_counts.items())),
        "return_code_counts": dict(sorted(return_code_counts.items())),
        "status_1_unresolved_summary_count_histogram": {
            key: warning_histogram.get(key, 0) for key in ("86", "87", "88")
        },
        "sidecar_status_contract_pass": True,
    }
    return validation, anchor_runs, full_grid


def validate_traces(
    context: dict[str, Any],
    preregistration: dict[str, Any],
    trace_paths: Sequence[Path],
    anchor_runs: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    expected_cases = {
        (item["coordinate_id"], item["truth_profile"])
        for item in context["scale_cases"]
    }
    coordinate_literals = {
        item["coordinate_id"]: item["target_transmission_literal"]
        for item in preregistration["opacity_selection"]["coordinates"]
    }
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    trace_anchor_keys: set[tuple[str, str, str]] = set()
    evaluation_count_by_case: dict[str, int] = {}
    for path in trace_paths:
        _, trace = load_canonical_json(path)
        key = (trace.get("target"), trace.get("profile"))
        require(key in expected_cases, f"unexpected scale trace: {path.name}")
        require(key not in traces, f"duplicate scale trace: {key}")
        require(
            path.name == f"{key[0]}_{key[1]}.json",
            f"scale trace filename mismatch: {path.name}",
        )
        require(
            trace.get("execution_context_sha256") == CONTEXT_SHA256,
            f"scale trace context mismatch: {path.name}",
        )
        require(
            trace.get("target_transmission_literal") == coordinate_literals[key[0]],
            f"scale trace target mismatch: {path.name}",
        )
        require(
            trace.get("root_iterations") == 48
            and trace.get("maximum_bracket_expansions") == 64,
            f"scale trace iteration contract mismatch: {path.name}",
        )
        evaluations = trace.get("evaluations")
        require(isinstance(evaluations, list) and evaluations, f"empty scale trace: {path.name}")
        require(
            evaluations[-1].get("role") == "canonical_plateau_midpoint",
            f"missing canonical midpoint: {path.name}",
        )
        for index, evaluation in enumerate(evaluations):
            require(evaluation.get("evaluation_index") == index, f"nonsequential trace: {path.name}")
            anchor_key = (key[0], key[1], evaluation.get("scale_decimal"))
            require(anchor_key not in trace_anchor_keys, f"duplicate trace evaluation: {anchor_key}")
            trace_anchor_keys.add(anchor_key)
            require(anchor_key in anchor_runs, f"trace evaluation lacks anchor sidecar: {anchor_key}")
            sidecar = anchor_runs[anchor_key]
            for trace_field, sidecar_field in (
                ("return_code", "return_code"),
                ("numeric_text_sha256", "numeric_text_sha256"),
                ("normalized_output_sha256", "normalized_output_sha256"),
                ("unresolved_line_warning_count", "unresolved_line_warning_count"),
            ):
                require(
                    evaluation.get(trace_field) == sidecar.get(sidecar_field),
                    f"trace/sidecar mismatch: {path.name}/{index}/{trace_field}",
                )
        traces[key] = trace
        evaluation_count_by_case[f"{key[0]}/{key[1]}"] = len(evaluations)

    require(len(traces) == OBSERVED_TRACE_COUNT, "scale trace count changed")
    require(set(anchor_runs) == trace_anchor_keys, "anchor/trace identity anti-join failed")
    failing = traces.get((FAILING_COORDINATE, FAILING_PROFILE))
    require(failing is not None, "failing scale trace is absent")
    require(
        len(failing["evaluations"]) == FAILING_TRACE_EVALUATION_COUNT,
        "failing scale trace evaluation count changed",
    )
    validation = {
        "expected_trace_count": EXPECTED_CASE_COUNT,
        "observed_trace_count": len(traces),
        "missing_trace_count": EXPECTED_CASE_COUNT - len(traces),
        "trace_evaluation_count": len(trace_anchor_keys),
        "matched_anchor_sidecar_count": len(set(anchor_runs) & trace_anchor_keys),
        "anchor_sidecar_without_trace_count": len(set(anchor_runs) - trace_anchor_keys),
        "trace_evaluation_without_anchor_sidecar_count": len(trace_anchor_keys - set(anchor_runs)),
        "evaluation_count_by_case": dict(sorted(evaluation_count_by_case.items())),
        "trace_anchor_binding_pass": True,
    }
    return validation, traces


def derive_coverage(
    context: dict[str, Any],
    traces: dict[tuple[str, str], dict[str, Any]],
    full_grid: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    expected_cases = [
        (item["coordinate_id"], item["truth_profile"])
        for item in context["scale_cases"]
    ]
    expected_grid = {
        (coordinate, profile, elevation)
        for coordinate, profile in expected_cases
        for elevation in EXPECTED_ELEVATIONS
    }
    actual_grid = set(full_grid)
    require(not actual_grid - expected_grid, "unexpected full-grid coordinate")
    require(len(actual_grid) == OBSERVED_FULL_GRID_COUNT, "observed full-grid count changed")

    cases: list[dict[str, Any]] = []
    complete_cases: list[dict[str, str]] = []
    incomplete_cases: list[dict[str, Any]] = []
    for coordinate, profile in expected_cases:
        observed_elevations = sorted(
            elevation
            for target, truth_profile, elevation in actual_grid
            if target == coordinate and truth_profile == profile
        )
        missing_elevations = sorted(set(EXPECTED_ELEVATIONS) - set(observed_elevations))
        trace = traces.get((coordinate, profile))
        complete = not missing_elevations
        record = {
            "coordinate_id": coordinate,
            "truth_profile": profile,
            "scale_trace_present": trace is not None,
            "scale_trace_evaluation_count": len(trace["evaluations"]) if trace else 0,
            "expected_full_grid_count": len(EXPECTED_ELEVATIONS),
            "observed_full_grid_count": len(observed_elevations),
            "missing_full_grid_count": len(missing_elevations),
            "observed_elevations_deg": observed_elevations,
            "missing_elevations_deg": missing_elevations,
            "case_complete": complete,
        }
        if complete:
            require(trace is not None, f"complete grid lacks scale trace: {coordinate}/{profile}")
            complete_cases.append({"coordinate_id": coordinate, "truth_profile": profile})
            record["state"] = "complete_scale_trace_and_56_full_grids"
        else:
            require(not observed_elevations, f"partially populated case: {coordinate}/{profile}")
            if (coordinate, profile) == (FAILING_COORDINATE, FAILING_PROFILE):
                require(trace is not None, "failing case scale trace missing")
                record["state"] = "scale_trace_complete_99_evaluations_guard_stop_zero_full_grids"
            else:
                require(trace is None, f"unexpected trace after guard stop: {coordinate}/{profile}")
                record["state"] = "not_started_after_guard_stop"
            incomplete_cases.append(record.copy())
        cases.append(record)

    require(len(complete_cases) == OBSERVED_COMPLETE_CASE_COUNT, "complete case count changed")
    require(len(incomplete_cases) == 4, "incomplete case count changed")
    require(
        sum(item["observed_full_grid_count"] for item in cases) == OBSERVED_FULL_GRID_COUNT,
        "case-level full-grid total mismatch",
    )
    return {
        "expected_scale_case_count": EXPECTED_CASE_COUNT,
        "complete_scale_case_count": len(complete_cases),
        "incomplete_scale_case_count": len(incomplete_cases),
        "expected_elevations_per_case": len(EXPECTED_ELEVATIONS),
        "expected_full_grid_count": len(expected_grid),
        "observed_full_grid_count": len(actual_grid),
        "missing_full_grid_count": len(expected_grid - actual_grid),
        "unexpected_full_grid_count": len(actual_grid - expected_grid),
        "duplicate_full_grid_count": 0,
        "complete_cases": complete_cases,
        "incomplete_cases": incomplete_cases,
        "cases": cases,
        "coverage_complete": False,
    }


def float_ulp_distance(left: float, right: float) -> int:
    require(math.isfinite(left) and math.isfinite(right), "nonfinite ULP operand")
    require(left >= 0.0 and right >= 0.0, "ULP helper only admits nonnegative values")
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def guard_discrepancy(
    preregistration: dict[str, Any],
    traces: dict[tuple[str, str], dict[str, Any]],
    full_grid: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    coordinate = next(
        item
        for item in preregistration["opacity_selection"]["coordinates"]
        if item["coordinate_id"] == FAILING_COORDINATE
    )
    trace = traces[(FAILING_COORDINATE, FAILING_PROFILE)]
    evaluations = trace["evaluations"]
    midpoint = evaluations[-1]
    require(midpoint["evaluation_index"] == 98, "failing midpoint index changed")
    require(midpoint["role"] == "canonical_plateau_midpoint", "failing midpoint role changed")
    require(
        midpoint["transmission"] == "8.71912800000000043e-01",
        "failing midpoint transmission changed",
    )
    target_literal = coordinate["target_transmission_literal"]
    target_float = float(target_literal)
    parsed_midpoint_float = float(midpoint["transmission"])
    require(parsed_midpoint_float == target_float, "failing midpoint missed target literal")

    recomputed_tau = -math.log(target_float) / float(X80_DECIMAL)
    expected_tau = float(Decimal(coordinate["achieved_tau225_exact"]))
    signed = recomputed_tau - expected_tau
    absolute = abs(signed)
    threshold = float(GUARD_THRESHOLD_SOURCE_LITERAL)
    margin = absolute - threshold
    require(absolute > threshold, "recorded guard no longer fails")
    require(float_ulp_distance(recomputed_tau, expected_tau) == 2, "guard ULP discrepancy changed")
    require(not full_grid.keys() & {(FAILING_COORDINATE, FAILING_PROFILE, e) for e in EXPECTED_ELEVATIONS}, "failing case unexpectedly has full grids")

    runner_path = PACKAGE_DIR / RUNNER_NAME
    source_lines = runner_path.read_text(encoding="utf-8").splitlines()
    guard_line_numbers = [
        index
        for index, line in enumerate(source_lines, start=1)
        if "abs(achieved_from_literal - float(coordinate.achieved_tau)) <= 5.0e-17" in line
    ]
    require(len(guard_line_numbers) == 1, "frozen runner guard expression changed")
    exception_lines = [
        index
        for index, line in enumerate(source_lines, start=1)
        if 'f"achieved coordinate mismatch: {target}/{case.profile}"' in line
    ]
    require(len(exception_lines) == 1, "frozen runner guard exception changed")

    return {
        "coordinate_id": FAILING_COORDINATE,
        "truth_profile": FAILING_PROFILE,
        "failure_stage": "post_scale_solution_pre_full_grid_achieved_coordinate_float_guard",
        "scale_trace_path_relative_to_cache": f"scale_traces/{FAILING_TRACE_NAME}",
        "scale_trace_evaluation_count": len(evaluations),
        "canonical_midpoint_evaluation_index": midpoint["evaluation_index"],
        "canonical_midpoint_scale_decimal": midpoint["scale_decimal"],
        "target_transmission_literal": target_literal,
        "canonical_midpoint_transmission_f17": midpoint["transmission"],
        "parsed_midpoint_equals_target_float": True,
        "x80_decimal": str(X80_DECIMAL),
        "expected_achieved_tau225_exact_decimal": coordinate["achieved_tau225_exact"],
        "expected_achieved_tau225_float_f17": f17(expected_tau),
        "expected_achieved_tau225_float_hex": expected_tau.hex(),
        "expected_achieved_tau225_float_exact_decimal": exact_float_decimal(expected_tau),
        "recomputed_from_target_tau225_float_f17": f17(recomputed_tau),
        "recomputed_from_target_tau225_float_hex": recomputed_tau.hex(),
        "recomputed_from_target_tau225_float_exact_decimal": exact_float_decimal(recomputed_tau),
        "signed_discrepancy_recomputed_minus_expected_f17": f17(signed),
        "signed_discrepancy_exact_binary_float_decimal": exact_float_decimal(signed),
        "absolute_discrepancy_f17": f17(absolute),
        "absolute_discrepancy_exact_binary_float_decimal": exact_float_decimal(absolute),
        "guard_threshold_source_literal": GUARD_THRESHOLD_SOURCE_LITERAL,
        "guard_threshold_float_f17": f17(threshold),
        "guard_threshold_float_exact_decimal": exact_float_decimal(threshold),
        "excess_over_guard_f17": f17(margin),
        "excess_over_guard_exact_binary_float_decimal": exact_float_decimal(margin),
        "absolute_discrepancy_in_expected_value_ulps": 2,
        "guard_comparison_pass": False,
        "frozen_runner_sha256": sha256_path(runner_path),
        "frozen_runner_guard_line": guard_line_numbers[0],
        "frozen_runner_exception_line": exception_lines[0],
        "guard_expression": "abs(achieved_from_literal - float(coordinate.achieved_tau)) <= 5.0e-17",
        "expected_exception_message": f"achieved coordinate mismatch: {FAILING_COORDINATE}/{FAILING_PROFILE}",
        "exception_capture_status": "not_preserved_in_cache; reconstructed exactly from frozen runner expression and digest-bound trace/preregistration operands",
        "full_grid_count_for_failing_case": 0,
    }


def internal_cache_summary(cache_dir: Path) -> dict[str, Any]:
    root = cache_dir / "am_spectral_cache"
    require(root.is_dir() and not root.is_symlink(), "missing internal AM cache")
    expected_shards = [f"shard_{index:02d}" for index in range(8)]
    require(
        [path.name for path in sorted(root.iterdir())] == expected_shards,
        "internal AM cache shard set mismatch",
    )
    shard_records: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    for shard_name in expected_shards:
        shard = root / shard_name
        files = direct_regular_files(shard)
        entries = [relative_entry(cache_dir, path) for path in files]
        aggregate = aggregate_entries(entries)
        shard_records.append({"shard": shard_name, **aggregate})
        all_entries.extend(entries)
    return {
        "inventory_level": "aggregate_per_shard_only_no_internal_member_list_in_decision",
        "shard_count": len(shard_records),
        **aggregate_entries(all_entries),
        "shards": shard_records,
    }


def build_report(decision: dict[str, Any]) -> bytes:
    coverage = decision["coverage"]
    guard = decision["stop_record"]["guard_discrepancy"]
    inventory = decision["cache_inventory"]
    lines = [
        "# SCI-CAL-001 EL25 Confirmation Failure Record",
        "",
        "## Decision",
        "",
        "The preregistered confirmation is **invalid**, not a numerical pass or fail. "
        "Execution stopped at the frozen achieved-coordinate floating-point guard "
        "before complete evidence existed. Numerical representation fidelity is "
        "invalid/unavailable, observational performance was not evaluated, and this "
        "record authorizes neither operator adoption nor an operational domain.",
        "",
        "No partial band integration, candidate error metric, ranking, maximum-error "
        "search, or observational inference was performed by this recorder.",
        "",
        "## Exact Stop",
        "",
        f"- Case: `{guard['coordinate_id']}/{guard['truth_profile']}`",
        f"- Trace: `{guard['scale_trace_path_relative_to_cache']}` with "
        f"{guard['scale_trace_evaluation_count']} evaluations",
        f"- Expected achieved tau225 float: `{guard['expected_achieved_tau225_float_f17']}`",
        f"- Recomputed tau225 float: `{guard['recomputed_from_target_tau225_float_f17']}`",
        f"- Absolute discrepancy: `{guard['absolute_discrepancy_f17']}` "
        f"({guard['absolute_discrepancy_in_expected_value_ulps']} ULPs)",
        f"- Frozen guard threshold: `{guard['guard_threshold_float_f17']}`",
        f"- Excess over threshold: `{guard['excess_over_guard_f17']}`",
        "- The cache contains no captured exception log. The discrepancy is reconstructed "
        "from the SHA-bound frozen runner expression, the canonical 99-evaluation trace, "
        "and the frozen preregistration operand; it is not presented as a verbatim log.",
        "",
        "## Preserved Coverage",
        "",
        f"- Complete cases: {coverage['complete_scale_case_count']} / "
        f"{coverage['expected_scale_case_count']}",
        f"- Complete full grids: {coverage['observed_full_grid_count']} / "
        f"{coverage['expected_full_grid_count']}",
        f"- Missing full grids: {coverage['missing_full_grid_count']}",
        "- The failing case has its completed scale trace and zero full grids. Three later "
        "cases were not started.",
        "",
        "## Cache Integrity",
        "",
        f"- Execution context SHA-256: `{decision['provenance']['execution_context']['sha256']}`",
        f"- Raw outputs and matched sidecars: "
        f"{decision['cache_validation']['raw_sidecar_matching']['matched_pair_count']}",
        f"- Scale traces: {inventory['scale_traces']['file_count']}",
        f"- Rejected AM failed-attempt files: {inventory['failed_attempts']['file_count']}",
        f"- Evidence aggregate SHA-256: `{inventory['evidence_aggregate']['aggregate_sha256']}`",
        f"- Internal AM cache: {inventory['am_spectral_cache_summary']['file_count']} files "
        f"across {inventory['am_spectral_cache_summary']['shard_count']} shards, aggregate "
        f"`{inventory['am_spectral_cache_summary']['aggregate_sha256']}`",
        "",
        "The decision JSON contains the complete per-file SHA-256 inventory for raw "
        "outputs, execution sidecars, scale traces, and failed attempts. Internal AM "
        "cache members are summarized by deterministic per-shard and whole-cache "
        "aggregates.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def build_failure_record(cache_dir: Path) -> tuple[bytes, bytes]:
    require(cache_dir.is_absolute(), "cache path must be absolute")
    require(cache_dir.name == CACHE_BASENAME, "cache basename mismatch")
    require(cache_dir.is_dir() and not cache_dir.is_symlink(), "missing cache root")
    expected_root_entries = {
        CACHE_LOCK_NAME,
        "am_spectral_cache",
        CONTEXT_NAME,
        "execution_records",
        "failed_attempts",
        "raw_outputs",
        "scale_traces",
    }
    require(
        {path.name for path in cache_dir.iterdir()} == expected_root_entries,
        "cache root entry set mismatch",
    )

    with shared_cache_lock(cache_dir):
        context, preregistration, context_entry = validate_context(cache_dir)
        raw_paths = direct_regular_files(cache_dir / "raw_outputs")
        sidecar_paths = direct_regular_files(cache_dir / "execution_records")
        trace_paths = direct_regular_files(cache_dir / "scale_traces")
        failed_paths = direct_regular_files(cache_dir / "failed_attempts")
        require(all(path.suffix == ".txt" for path in raw_paths), "unexpected raw-output suffix")
        require(
            all(path.name.endswith(".run.json") for path in sidecar_paths),
            "unexpected execution-sidecar suffix",
        )
        require(all(path.suffix == ".json" for path in trace_paths), "unexpected scale-trace suffix")
        require(len(raw_paths) == OBSERVED_RAW_AND_SIDECAR_COUNT, "raw-output count changed")
        require(len(sidecar_paths) == OBSERVED_RAW_AND_SIDECAR_COUNT, "sidecar count changed")
        require(len(trace_paths) == OBSERVED_TRACE_COUNT, "scale-trace count changed")
        require(not failed_paths, "failed-attempt directory is no longer empty")

        raw_inventory = inventory_paths(cache_dir, raw_paths)
        sidecar_inventory = inventory_paths(cache_dir, sidecar_paths)
        trace_inventory = inventory_paths(cache_dir, trace_paths)
        failed_inventory = inventory_paths(cache_dir, failed_paths)
        matching, anchor_runs, full_grid = validate_sidecars(
            cache_dir, context, raw_inventory, sidecar_paths
        )
        trace_validation, traces = validate_traces(
            context, preregistration, trace_paths, anchor_runs
        )
        coverage = derive_coverage(context, traces, full_grid)
        guard = guard_discrepancy(preregistration, traces, full_grid)
        internal_summary = internal_cache_summary(cache_dir)

        evidence_entries = [context_entry]
        for category in (
            raw_inventory,
            sidecar_inventory,
            trace_inventory,
            failed_inventory,
        ):
            evidence_entries.extend(category["files"])
        evidence_aggregate = aggregate_entries(evidence_entries)

        recorder_path = Path(__file__).resolve()
        decision = {
            "schema_version": "sci-cal-001-am12-el25-confirmation-failure-record-v1",
            "record_id": RECORD_ID,
            "record_kind": "deterministic_post_stop_failure_evidence",
            "package_id": PACKAGE_ID,
            "study_id": STUDY_ID,
            "provenance": {
                "recorder": {
                    "filename": recorder_path.name,
                    "sha256": sha256_path(recorder_path),
                },
                "cache": {
                    "basename": CACHE_BASENAME,
                    "resolved_path": str(cache_dir.resolve()),
                    "access": "nonblocking_shared_POSIX_lock_read_only",
                    "lock_path_relative_to_cache": CACHE_LOCK_NAME,
                },
                "execution_context": context_entry,
                "preregistration_git_binding": context["preregistration_git_binding"],
                "frozen_controls": context["controls"],
                "AM": {
                    "model": context["atmosphere"]["model"],
                    "executable_sha256": context["atmosphere"]["executable"]["sha256"],
                    "execution_context_bound_version_identity": "am version 12.2 (build date Aug  1 2026 11:20:29)",
                },
            },
            "integrity_boundary": {
                "AM_invoked": False,
                "cache_replay_invoked": False,
                "spectral_arrays_loaded": False,
                "passband_integration_performed": False,
                "candidate_operator_evaluated": False,
                "partial_candidate_errors_computed_or_inspected": False,
                "maximum_error_computed_or_inspected": False,
                "Citlali_application_code_modified": False,
                "TolTECA_modified": False,
                "Unity_contacted": False,
                "operator_adopted": False,
                "operational_domain_authorized": False,
            },
            "cache_validation": {
                "execution_context_canonical_JSON": True,
                "execution_context_identity_pass": True,
                "root_entry_set_pass": True,
                "raw_sidecar_matching": matching,
                "trace_anchor_matching": trace_validation,
                "failed_attempt_directory_empty": True,
                "structural_validation_pass": True,
            },
            "cache_inventory": {
                "aggregate_scope": "execution_context plus raw_outputs, execution_records, scale_traces, and failed_attempts; lock and internal AM implementation cache excluded",
                "evidence_aggregate": evidence_aggregate,
                "execution_context": context_entry,
                "raw_outputs": raw_inventory,
                "execution_records": sidecar_inventory,
                "scale_traces": trace_inventory,
                "failed_attempts": failed_inventory,
                "am_spectral_cache_summary": internal_summary,
            },
            "coverage": coverage,
            "stop_record": {
                "stop_kind": "preregistered_runner_guard_failure_after_AM_scale_search_before_full_grid_for_case_13",
                "failed_AM_attempt_preserved": False,
                "reason_no_failed_AM_attempt": "AM returned admitted evidence; the subsequent confirmation-runner achieved-coordinate floating-point guard stopped execution",
                "guard_discrepancy": guard,
            },
            "decision": {
                "confirmation_verdict": "confirmation_invalid",
                "software_correctness_gate": "invalid",
                "numerical_representation_fidelity_gate": "invalid",
                "numerical_representation_fidelity_reason": "complete preregistered coverage and candidate metrics are unavailable",
                "observational_performance_gate": "not_evaluated",
                "operator_adoption_authorized": False,
                "operational_opacity_elevation_domain_authorized": False,
                "production_authorized": False,
                "repair_or_reaudit_authorized": False,
                "owner_scientific_choice_made": False,
            },
        }

    decision_raw = json_bytes(decision)
    report_raw = build_report(decision)
    return decision_raw, report_raw


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_or_check(output_dir: Path, decision_raw: bytes, report_raw: bytes, *, check: bool) -> None:
    outputs = {
        output_dir / DECISION_NAME: decision_raw,
        output_dir / REPORT_NAME: report_raw,
    }
    if check:
        for path, expected in outputs.items():
            require(path.is_file(), f"missing recorded failure artifact: {path}")
            require(path.read_bytes() == expected, f"failure artifact is not deterministic: {path}")
        return
    for path, data in outputs.items():
        atomic_write(path, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the read-only stopped SCI-CAL-001 EL25 confirmation cache."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute in memory and require existing artifacts to match byte-for-byte",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    require(output_dir == PACKAGE_DIR, "failure artifacts must be written in the task package")
    decision_raw, report_raw = build_failure_record(args.cache_dir.resolve())
    write_or_check(output_dir, decision_raw, report_raw, check=args.check)
    print(
        json.dumps(
            {
                "check": args.check,
                "decision": {
                    "filename": DECISION_NAME,
                    "size_bytes": len(decision_raw),
                    "sha256": sha256_bytes(decision_raw),
                },
                "report": {
                    "filename": REPORT_NAME,
                    "size_bytes": len(report_raw),
                    "sha256": sha256_bytes(report_raw),
                },
                "verdict": "confirmation_invalid",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
