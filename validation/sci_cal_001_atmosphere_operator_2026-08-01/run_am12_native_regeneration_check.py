#!/usr/bin/env python3
"""Run and verify the bounded SCI-CAL-001 AM 12.2 native regeneration.

The copied AM tree is treated as read-only.  By default this script first
executes the preregistered annual-q95 ZA10/ZA70 smoke gate, then (only if both
cases match exactly) executes the five annual LMT profiles at zenith angles
10--80 degrees in two-degree steps.  It uses the historical AM argv without
its Slurm ``srun`` wrapper.  Raw combined stdout/stderr and exact execution
sidecars are written only below ``--cache-dir``; cache-dependent header bytes
are not used for committed artifact identity.  Parsed
f/tau/tx/Trj/Tb samples are compared with the copied historical ``.dat`` files
using exact binary64 equality.

``--check`` never runs AM.  It validates the cached raw outputs and execution
sidecars, reconstructs the three evidence artifacts, and checks their bytes
against the files beside this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_ID = "SCI-CAL-001"
SCHEMA_VERSION = "sci-cal-001-am12-native-regeneration-v2"
EVIDENCE_DATE = "2026-08-01"
DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
PERCENTILES = (5, 25, 50, 75, 95)
ZENITH_ANGLES_DEG = tuple(range(10, 82, 2))
EXPECTED_ROWS = 50001
EXPECTED_FREQUENCY_GHZ = np.arange(EXPECTED_ROWS, dtype=np.float64) / 100.0
EXPECTED_WARNING_COUNTS = (86, 87, 88)

METRICS_NAME = "native_regeneration_metrics.csv"
MANIFEST_NAME = "native_regeneration_manifest.json"
REPORT_NAME = "NATIVE_REGENERATION_REPORT.md"
EXECUTION_CONTEXT_NAME = "execution_context.json"
CACHE_LOCK_NAME = ".native_regeneration.lock"
PINNED_LOCALE = {"LANG": "C", "LC_ALL": "C"}

HISTORICAL_LINUX_BINARY_SHA256 = (
    "3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c"
)
HISTORICAL_RUN_SCRIPT_SHA256 = (
    "02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3"
)
HISTORICAL_COMMAND_PRINTER_SHA256 = (
    "29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6"
)
HISTORICAL_PACKER_SHA256 = (
    "3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5"
)
HISTORICAL_VERSION_LINE = "am version 12.2 (build date Aug 26 2022 19:20:13)"
AM12_VERSION_PREFIX = "am version 12.2 "
HISTORICAL_COMPILER_ID = "GCC 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)"

FLOAT_TOKEN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\Z")
VERSION_LINE = re.compile(r"^# (?P<identity>am version .+)$", re.MULTILINE)
UNRESOLVED_WARNING = re.compile(
    r"^! Warning: Encountered in-band lines narrower than the frequency\n"
    r"^!          grid spacing\.  The output configuration data includes\n"
    r"^!          the unresolved line count after each column definition\n"
    r"^!          for which this occurred\.  Count: (?P<count>\d+)$",
    re.MULTILINE,
)
WARNING_LINE = re.compile(r"^! Warning: (?P<message>.*)$", re.MULTILINE)
ERROR_LINE = re.compile(r"^! Error: .*$", re.MULTILINE)
COLUMN_WARNING = re.compile(r"Column included \d+ unresolved lines\.\Z")
SUMMARY_WARNING = "Encountered in-band lines narrower than the frequency"
CACHE_INSERT_WARNING = "Unable to rename file in insert_as_mru()."
CACHE_PROMOTE_WARNING = "Unable to rename file in promote_to_mru()."

WARNING_COUNT_FIELDS = (
    "unresolved_column_warning_line_count",
    "unresolved_summary_warning_line_count",
    "cache_insert_as_mru_warning_line_count",
    "cache_promote_to_mru_warning_line_count",
    "other_warning_line_count",
)


@dataclass(frozen=True)
class Case:
    percentile: int
    zenith_angle_deg: int

    @property
    def profile_stem(self) -> str:
        return f"LMT_annual_{self.percentile}"

    @property
    def case_id(self) -> str:
        return f"{self.profile_stem}_za{self.zenith_angle_deg:02d}"

    @property
    def filename(self) -> str:
        return f"{self.profile_stem}_{self.zenith_angle_deg}.dat"

    @property
    def elevation_deg(self) -> int:
        return 90 - self.zenith_angle_deg


@dataclass(frozen=True)
class ParsedOutput:
    samples: np.ndarray
    version_identity: str
    warning_count: int | None
    numeric_text_sha256: str
    normalized_output_sha256: str
    unresolved_column_warning_line_count: int
    unresolved_summary_warning_line_count: int
    cache_insert_as_mru_warning_line_count: int
    cache_promote_to_mru_warning_line_count: int
    other_warning_line_count: int
    error_line_count: int


@dataclass(frozen=True)
class BuildIdentity:
    supplied_path: str
    resolved_path: str
    size_bytes: int
    sha256: str
    binary_format: str


@dataclass(frozen=True)
class RunResult:
    case: Case
    generated_sha256: str
    return_code: int
    parsed: ParsedOutput
    sidecar: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f64(value: float) -> str:
    return format(float(value), ".17e")


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("cannot render an empty metrics table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def detect_binary_format(path: Path) -> str:
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic == b"\x7fELF":
        return "elf"
    if magic in {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }:
        return "mach-o"
    if magic[:2] == b"MZ":
        return "pe"
    return "unknown"


def build_identity(path: Path) -> BuildIdentity:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"not a regular executable file: {resolved}")
    return BuildIdentity(
        supplied_path=str(path),
        resolved_path=str(resolved),
        size_bytes=resolved.stat().st_size,
        sha256=sha256_path(resolved),
        binary_format=detect_binary_format(resolved),
    )


def build_identity_payload(identity: BuildIdentity) -> dict[str, Any]:
    return {
        "supplied_path": identity.supplied_path,
        "resolved_path": identity.resolved_path,
        "size_bytes": identity.size_bytes,
        "sha256": identity.sha256,
        "binary_format": identity.binary_format,
    }


def host_identity() -> dict[str, str]:
    return {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
    }


def classify_regeneration_build(
    regeneration: BuildIdentity, historical_linux: BuildIdentity
) -> str:
    if regeneration.sha256 == historical_linux.sha256:
        return "copied_linux_reference_binary_reexecution"
    if regeneration.binary_format == "mach-o":
        return "native_macos_build_distinct_from_copied_linux_binary"
    return "native_or_host_build_distinct_from_copied_linux_binary"


def normalize_combined_output(text: str) -> bytes:
    lines = []
    for line in text.splitlines():
        if line.startswith("# run time "):
            lines.append("# run time <volatile>")
        elif line.startswith("# dcache hit: "):
            lines.append("# dcache counters <volatile>")
        else:
            lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_output(data: bytes, *, label: str) -> ParsedOutput:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"non-UTF-8 AM output for {label}") from error

    rows: list[list[float]] = []
    numeric_text_digest = hashlib.sha256()
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        tokens = line.split()
        if len(tokens) != 5 or not all(FLOAT_TOKEN.fullmatch(item) for item in tokens):
            continue
        rows.append([float(item) for item in tokens])
        numeric_text_digest.update(raw_line.encode("utf-8"))
    samples = np.asarray(rows, dtype=np.float64)
    if samples.shape != (EXPECTED_ROWS, 5):
        raise RuntimeError(
            f"unexpected numeric grid for {label}: {samples.shape} != "
            f"({EXPECTED_ROWS}, 5)"
        )
    if not np.all(np.isfinite(samples)):
        raise RuntimeError(f"non-finite numeric value in {label}")
    if not np.array_equal(samples[:, 0], EXPECTED_FREQUENCY_GHZ):
        raise RuntimeError(f"unexpected frequency coordinate in {label}")

    versions = VERSION_LINE.findall(text)
    if len(versions) != 1:
        raise RuntimeError(
            f"expected one AM version line in {label}, found {len(versions)}"
        )
    warnings = [int(item) for item in UNRESOLVED_WARNING.findall(text)]
    if len(warnings) > 1:
        raise RuntimeError(f"multiple unresolved-line warning summaries in {label}")
    warning_count = warnings[0] if warnings else None
    if warning_count is not None and warning_count not in EXPECTED_WARNING_COUNTS:
        raise RuntimeError(
            f"unexpected unresolved-line count in {label}: {warning_count}"
        )
    warning_messages = [match.group("message") for match in WARNING_LINE.finditer(text)]
    unresolved_column_warning_line_count = sum(
        COLUMN_WARNING.fullmatch(message) is not None for message in warning_messages
    )
    unresolved_summary_warning_line_count = warning_messages.count(SUMMARY_WARNING)
    cache_insert_as_mru_warning_line_count = warning_messages.count(
        CACHE_INSERT_WARNING
    )
    cache_promote_to_mru_warning_line_count = warning_messages.count(
        CACHE_PROMOTE_WARNING
    )
    classified_warning_count = (
        unresolved_column_warning_line_count
        + unresolved_summary_warning_line_count
        + cache_insert_as_mru_warning_line_count
        + cache_promote_to_mru_warning_line_count
    )
    other_warning_line_count = len(warning_messages) - classified_warning_count
    error_line_count = len(ERROR_LINE.findall(text))
    if unresolved_summary_warning_line_count != 1:
        raise RuntimeError(
            f"expected one unresolved-line summary warning in {label}, found "
            f"{unresolved_summary_warning_line_count}"
        )
    if unresolved_column_warning_line_count <= 0:
        raise RuntimeError(f"missing unresolved-column warnings in {label}")
    if (
        cache_insert_as_mru_warning_line_count
        or cache_promote_to_mru_warning_line_count
        or other_warning_line_count
        or error_line_count
    ):
        raise RuntimeError(
            f"unexpected AM diagnostic class in {label}: "
            f"cache_insert={cache_insert_as_mru_warning_line_count}, "
            f"cache_promote={cache_promote_to_mru_warning_line_count}, "
            f"other_warning={other_warning_line_count}, error={error_line_count}"
        )
    return ParsedOutput(
        samples=samples,
        version_identity=versions[0],
        warning_count=warning_count,
        numeric_text_sha256=numeric_text_digest.hexdigest(),
        normalized_output_sha256=sha256_bytes(normalize_combined_output(text)),
        unresolved_column_warning_line_count=unresolved_column_warning_line_count,
        unresolved_summary_warning_line_count=unresolved_summary_warning_line_count,
        cache_insert_as_mru_warning_line_count=(cache_insert_as_mru_warning_line_count),
        cache_promote_to_mru_warning_line_count=(
            cache_promote_to_mru_warning_line_count
        ),
        other_warning_line_count=other_warning_line_count,
        error_line_count=error_line_count,
    )


def validate_return_contract(
    return_code: int, parsed: ParsedOutput, *, label: str
) -> None:
    if not parsed.version_identity.startswith(AM12_VERSION_PREFIX):
        raise RuntimeError(
            f"rejected non-AM-12.2 identity for {label}: {parsed.version_identity!r}"
        )
    if return_code == 0:
        return
    if return_code == 1 and parsed.warning_count in EXPECTED_WARNING_COUNTS:
        return
    raise RuntimeError(
        f"rejected AM return status for {label}: return_code={return_code}, "
        f"unresolved_line_warning_count={parsed.warning_count}"
    )


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def case_argv(executable: Path, case: Case) -> list[str]:
    return [
        str(executable),
        f"LMT_am_inputs/{case.profile_stem}.amc",
        "0",
        "GHz",
        "500",
        "GHz",
        "10",
        "MHz",
        str(case.zenith_angle_deg),
        "deg",
        "1.0",
    ]


def sidecar_path(cache_dir: Path, case: Case) -> Path:
    return cache_dir / "execution_records" / f"{case.case_id}.run.json"


def raw_cache_path(cache_dir: Path, case: Case) -> Path:
    return cache_dir / "raw_outputs" / case.filename


def case_cache_shard_index(case: Case, shard_count: int) -> int:
    percentile_index = PERCENTILES.index(case.percentile)
    zenith_index = ZENITH_ANGLES_DEG.index(case.zenith_angle_deg)
    matrix_index = percentile_index * len(ZENITH_ANGLES_DEG) + zenith_index
    return matrix_index % shard_count


def case_am_cache_dir(am_cache_root: Path, case: Case, shard_count: int) -> Path:
    return am_cache_root / f"shard_{case_cache_shard_index(case, shard_count):02d}"


def run_case(
    case: Case,
    *,
    executable: Path,
    executable_identity: BuildIdentity,
    big_atmosphere_root: Path,
    cache_dir: Path,
    am_cache_dir: Path,
    omp_threads: int,
    execution_host: dict[str, str],
    execution_context_sha256: str,
) -> RunResult:
    argv = case_argv(executable, case)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(omp_threads)
    environment["AM_CACHE_PATH"] = str(am_cache_dir)
    environment.update(PINNED_LOCALE)
    completed = subprocess.run(
        argv,
        cwd=big_atmosphere_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    generated = completed.stdout
    parsed = parse_output(generated, label=f"generated {case.case_id}")
    validate_return_contract(completed.returncode, parsed, label=case.case_id)
    profile = big_atmosphere_root / "LMT_am_inputs" / f"{case.profile_stem}.amc"
    raw_path = raw_cache_path(cache_dir, case)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "argv": argv,
        "working_directory_role": "Big_Atmosphere",
        "working_directory_path_relative_to_am_root": big_atmosphere_root.relative_to(
            big_atmosphere_root.parent
        ).as_posix(),
        "profile_sha256": sha256_path(profile),
        "am_executable_sha256": executable_identity.sha256,
        "omp_threads": omp_threads,
        "locale": PINNED_LOCALE,
        "execution_host": execution_host,
        "execution_context_sha256": execution_context_sha256,
        "am_cache_path_relative_to_cache": str(am_cache_dir.relative_to(cache_dir)),
        "return_code": completed.returncode,
        "combined_output_path_relative_to_cache": str(raw_path.relative_to(cache_dir)),
        "combined_output_sha256": sha256_bytes(generated),
        "numeric_text_sha256": parsed.numeric_text_sha256,
        "normalized_output_sha256": parsed.normalized_output_sha256,
        "numeric_row_count": int(parsed.samples.shape[0]),
        "unresolved_line_warning_count": parsed.warning_count,
        **{field: getattr(parsed, field) for field in WARNING_COUNT_FIELDS},
        "error_line_count": parsed.error_line_count,
        "am_version_identity": parsed.version_identity,
    }
    atomic_write(raw_path, generated)
    atomic_write(sidecar_path(cache_dir, case), json_bytes(sidecar))
    return RunResult(
        case, sha256_bytes(generated), completed.returncode, parsed, sidecar
    )


def load_cached_case(
    case: Case,
    *,
    executable_identity: BuildIdentity,
    big_atmosphere_root: Path,
    cache_dir: Path,
    am_cache_dir: Path,
    omp_threads: int,
    execution_host: dict[str, str],
    execution_context_sha256: str,
) -> RunResult:
    raw_path = raw_cache_path(cache_dir, case)
    record_path = sidecar_path(cache_dir, case)
    if not raw_path.is_file() or not record_path.is_file():
        raise RuntimeError(f"missing cached AM evidence for {case.case_id}")
    generated = raw_path.read_bytes()
    sidecar = json.loads(record_path.read_text(encoding="utf-8"))
    profile = big_atmosphere_root / "LMT_am_inputs" / f"{case.profile_stem}.amc"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "working_directory_role": "Big_Atmosphere",
        "working_directory_path_relative_to_am_root": big_atmosphere_root.relative_to(
            big_atmosphere_root.parent
        ).as_posix(),
        "profile_sha256": sha256_path(profile),
        "am_executable_sha256": executable_identity.sha256,
        "omp_threads": omp_threads,
        "locale": PINNED_LOCALE,
        "execution_host": execution_host,
        "execution_context_sha256": execution_context_sha256,
        "am_cache_path_relative_to_cache": str(am_cache_dir.relative_to(cache_dir)),
        "combined_output_path_relative_to_cache": str(raw_path.relative_to(cache_dir)),
        "combined_output_sha256": sha256_bytes(generated),
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise RuntimeError(
                f"cached sidecar mismatch for {case.case_id}/{key}: "
                f"{sidecar.get(key)!r} != {value!r}"
            )
    argv = sidecar.get("argv")
    if argv != case_argv(Path(executable_identity.resolved_path), case):
        raise RuntimeError(f"cached argv mismatch for {case.case_id}")
    parsed = parse_output(generated, label=f"cached {case.case_id}")
    return_code = int(sidecar.get("return_code"))
    validate_return_contract(return_code, parsed, label=case.case_id)
    if sidecar.get("numeric_row_count") != EXPECTED_ROWS:
        raise RuntimeError(f"cached row-count mismatch for {case.case_id}")
    if sidecar.get("unresolved_line_warning_count") != parsed.warning_count:
        raise RuntimeError(f"cached warning-count mismatch for {case.case_id}")
    if sidecar.get("numeric_text_sha256") != parsed.numeric_text_sha256:
        raise RuntimeError(f"cached numeric-text digest mismatch for {case.case_id}")
    if sidecar.get("normalized_output_sha256") != parsed.normalized_output_sha256:
        raise RuntimeError(
            f"cached normalized-output digest mismatch for {case.case_id}"
        )
    for field in (*WARNING_COUNT_FIELDS, "error_line_count"):
        if sidecar.get(field) != getattr(parsed, field):
            raise RuntimeError(
                f"cached diagnostic-count mismatch for {case.case_id}/{field}"
            )
    if sidecar.get("am_version_identity") != parsed.version_identity:
        raise RuntimeError(f"cached AM-version mismatch for {case.case_id}")
    return RunResult(case, sha256_bytes(generated), return_code, parsed, sidecar)


def source_files(source_root: Path) -> list[Path]:
    named_files = {
        "_README_",
        "INSTALLING",
        "LICENSE",
        "Makefile",
        "REFERENCES",
    }
    return sorted(
        (
            path
            for path in source_root.iterdir()
            if path.is_file()
            and (path.suffix in {".c", ".h"} or path.name in named_files)
        ),
        key=lambda item: item.name,
    )


def inventory_files(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    entries = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_path(path)
        entries.append({"path": relative, "size_bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
        total_bytes += size
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "aggregate_algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "files": entries,
    }


def validate_frozen_workflow(am_root: Path) -> dict[str, Any]:
    paths = {
        "historical_run_script": (
            am_root / "Big_Atmosphere/01_do_am_runs.sh",
            HISTORICAL_RUN_SCRIPT_SHA256,
        ),
        "historical_command_printer": (
            am_root / "Big_Atmosphere/generateAmModels.py",
            HISTORICAL_COMMAND_PRINTER_SHA256,
        ),
        "historical_packer": (
            am_root / "Big_Atmosphere/make_npz.py",
            HISTORICAL_PACKER_SHA256,
        ),
    }
    result = {}
    for role, (path, expected) in paths.items():
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"frozen workflow digest mismatch for {path}")
        result[role] = {
            "path_relative_to_am_root": path.relative_to(am_root).as_posix(),
            "sha256": actual,
        }
    return result


def compiler_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_supplied",
            "note": "required for complete native-build provenance",
        }
    identity = build_identity(path)
    environment = os.environ.copy()
    environment.update(PINNED_LOCALE)
    completed = subprocess.run(
        [identity.resolved_path, "--version"],
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    version_bytes = completed.stdout
    return {
        "status": "supplied_by_operator_as_build_compiler",
        "supplied_path": identity.supplied_path,
        "resolved_path": identity.resolved_path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "version_command_return_code": completed.returncode,
        "version_output_sha256": sha256_bytes(version_bytes),
        "version_output": version_bytes.decode("utf-8", errors="replace").strip(),
    }


def build_execution_context(
    *,
    am_root: Path,
    regeneration: BuildIdentity,
    historical_linux: BuildIdentity,
    compiler: dict[str, Any],
    native_build_command: str | None,
    jobs: int,
    omp_threads: int,
    workflow: dict[str, Any],
    execution_host: dict[str, str],
    run_scope: str,
) -> dict[str, Any]:
    source_root = am_root / "am-12.2/src"
    profile_root = am_root / "Big_Atmosphere/LMT_am_inputs"
    reference_root = am_root / "Big_Atmosphere/LMT_am_outputs"
    profile_paths = [
        profile_root / f"LMT_annual_{percentile}.amc" for percentile in PERCENTILES
    ]
    reference_paths = [
        reference_root / Case(percentile, zenith_angle).filename
        for percentile in PERCENTILES
        for zenith_angle in ZENITH_ANGLES_DEG
    ]
    runner = Path(__file__).resolve()
    phase_order: list[dict[str, Any]] = [
        {
            "phase": "smoke_gate",
            "cases": ["LMT_annual_95_za10", "LMT_annual_95_za70"],
            "completion_barrier": (
                "both cases must exactly match before any subsequent phase"
            ),
        }
    ]
    if run_scope == "full_annual_matrix":
        phase_order.append(
            {
                "phase": "remaining_matrix",
                "order": (
                    "all non-smoke cases in percentile-major order with zenith "
                    "angle ascending before shard assignment"
                ),
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}-execution-context-v1",
        "runner": {
            "filename": runner.name,
            "sha256": sha256_path(runner),
        },
        "run_scope": run_scope,
        "execution_host": execution_host,
        "execution_parameters": {
            "jobs": jobs,
            "omp_threads_per_process": omp_threads,
            "locale": PINNED_LOCALE,
            "argv_template": [
                "<am-executable>",
                "LMT_am_inputs/LMT_annual_<percentile>.amc",
                "0",
                "GHz",
                "500",
                "GHz",
                "10",
                "MHz",
                "<zenith-angle-deg>",
                "deg",
                "1.0",
            ],
            "slurm_wrapper_used": False,
            "working_directory_role": "Big_Atmosphere",
            "am_cache_sharding": {
                "shard_count": jobs,
                "assignment": (
                    "percentile-major matrix index with zenith angle minor, "
                    "modulo shard_count"
                ),
                "phase_order": phase_order,
                "within_shard_order": (
                    "encounter order inside each phase; phases do not overlap"
                ),
                "process_ownership": (
                    "one ordered worker queue per shard per phase inside one process"
                ),
            },
            "cache_lock": {
                "filename": CACHE_LOCK_NAME,
                "writer_mode": "nonblocking whole-cache POSIX exclusive lock",
                "reader_mode": "nonblocking whole-cache POSIX shared lock",
            },
        },
        "builds": {
            "copied_linux_reference": build_identity_payload(historical_linux),
            "regeneration": {
                **build_identity_payload(regeneration),
                "classification": classify_regeneration_build(
                    regeneration, historical_linux
                ),
                "native_build_command": native_build_command,
                "compiler": compiler,
            },
        },
        "historical_workflow": workflow,
        "inputs": {
            "am_source_inventory": inventory_files(
                source_root, source_files(source_root)
            ),
            "annual_profile_inventory": inventory_files(profile_root, profile_paths),
            "copied_reference_output_inventory": inventory_files(
                reference_root, reference_paths
            ),
        },
        "output_normalization": {
            "purpose": (
                "preserve warning-bearing combined output while replacing only "
                "volatile runtime and dcache-counter header values"
            ),
            "algorithm": (
                "UTF-8 splitlines; replace lines beginning '# run time ' with "
                "'# run time <volatile>' and lines beginning '# dcache hit: ' "
                "with '# dcache counters <volatile>'; join with LF and append LF"
            ),
        },
        "security": {
            "uploader_logs_read": False,
            "uploader_logs_or_credentials_copied": False,
            "network_access": False,
            "unity_access": False,
        },
    }


def load_execution_context(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid cached execution context: {path}") from error
    if raw != json_bytes(payload):
        raise RuntimeError(f"noncanonical cached execution context: {path}")
    if not isinstance(payload.get("execution_host"), dict):
        raise RuntimeError(f"missing execution host in cached context: {path}")
    return payload


def acquire_cache_lock(cache_dir: Path, *, exclusive: bool) -> Any:
    lock_path = cache_dir / CACHE_LOCK_NAME
    if not exclusive and not lock_path.is_file():
        raise RuntimeError(f"cache-only operation requires lock file: {lock_path}")
    handle = lock_path.open("a+b")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        mode = "writer" if exclusive else "reader"
        raise RuntimeError(
            f"external cache is already locked; cannot acquire {mode} lock: {lock_path}"
        ) from error
    return handle


def maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def compare_case(
    run: RunResult,
    *,
    am_root: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    case = run.case
    reference_path = am_root / "Big_Atmosphere/LMT_am_outputs" / case.filename
    reference_bytes = reference_path.read_bytes()
    reference = parse_output(reference_bytes, label=f"reference {case.case_id}")
    if reference.version_identity != HISTORICAL_VERSION_LINE:
        raise RuntimeError(
            f"unexpected copied AM identity for {case.case_id}: "
            f"{reference.version_identity}"
        )
    if reference.warning_count not in EXPECTED_WARNING_COUNTS:
        raise RuntimeError(f"missing copied warning contract for {case.case_id}")

    field_names = ("frequency", "tau", "tx", "trj", "tb")
    equal = [
        bool(np.array_equal(reference.samples[:, index], run.parsed.samples[:, index]))
        for index in range(5)
    ]
    maxima = [
        maximum_absolute_difference(
            reference.samples[:, index], run.parsed.samples[:, index]
        )
        for index in range(5)
    ]
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "profile": case.profile_stem,
        "percentile": str(case.percentile),
        "zenith_angle_deg": str(case.zenith_angle_deg),
        "elevation_deg": str(case.elevation_deg),
        "reference_path_relative_to_am_root": reference_path.relative_to(
            am_root
        ).as_posix(),
        "generated_path_relative_to_cache": raw_cache_path(cache_dir, case)
        .relative_to(cache_dir)
        .as_posix(),
        "reference_sha256": sha256_bytes(reference_bytes),
        "reference_numeric_text_sha256": reference.numeric_text_sha256,
        "generated_numeric_text_sha256": run.parsed.numeric_text_sha256,
        "generated_normalized_output_sha256": (run.parsed.normalized_output_sha256),
        "numeric_data_lines_byte_equal": str(
            reference.numeric_text_sha256 == run.parsed.numeric_text_sha256
        ).lower(),
        "return_code": str(run.return_code),
        "unresolved_line_warning_count": str(run.parsed.warning_count),
        **{field: str(getattr(run.parsed, field)) for field in WARNING_COUNT_FIELDS},
        "error_line_count": str(run.parsed.error_line_count),
        "reference_am_identity": reference.version_identity,
        "regeneration_am_identity": run.parsed.version_identity,
        "reference_row_count": str(reference.samples.shape[0]),
        "generated_row_count": str(run.parsed.samples.shape[0]),
    }
    for name, is_equal, maximum in zip(field_names, equal, maxima, strict=True):
        row[f"{name}_exact_equal"] = str(is_equal).lower()
        row[f"{name}_max_abs_difference"] = f64(maximum)
    row["all_fields_exact_equal"] = str(all(equal)).lower()
    row["status"] = "exact_match" if all(equal) else "numeric_mismatch"
    return row


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_count = sum(row["all_fields_exact_equal"] == "true" for row in rows)
    data_line_byte_exact_count = sum(
        row["numeric_data_lines_byte_equal"] == "true" for row in rows
    )
    return_codes: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    regeneration_identities: dict[str, int] = {}
    for row in rows:
        return_codes[row["return_code"]] = return_codes.get(row["return_code"], 0) + 1
        warning = row["unresolved_line_warning_count"]
        warning_counts[warning] = warning_counts.get(warning, 0) + 1
        identity = row["regeneration_am_identity"]
        regeneration_identities[identity] = regeneration_identities.get(identity, 0) + 1
    maxima = {
        field: max(float(row[f"{field}_max_abs_difference"]) for row in rows)
        for field in ("frequency", "tau", "tx", "trj", "tb")
    }
    warning_class_totals = {
        field: sum(int(row[field]) for row in rows) for field in WARNING_COUNT_FIELDS
    }
    error_line_total = sum(int(row["error_line_count"]) for row in rows)
    normalized_numeric_digest = hashlib.sha256()
    normalized_full_output_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["case_id"]):
        case_id = row["case_id"].encode("utf-8")
        normalized_numeric_digest.update(case_id)
        normalized_numeric_digest.update(b"\0")
        normalized_numeric_digest.update(
            bytes.fromhex(row["generated_numeric_text_sha256"])
        )
        normalized_numeric_digest.update(b"\0")
        normalized_full_output_digest.update(case_id)
        normalized_full_output_digest.update(b"\0")
        normalized_full_output_digest.update(
            bytes.fromhex(row["generated_normalized_output_sha256"])
        )
        normalized_full_output_digest.update(b"\0")
    return {
        "case_count": len(rows),
        "exact_case_count": exact_count,
        "mismatch_case_count": len(rows) - exact_count,
        "all_cases_exact": exact_count == len(rows),
        "numeric_data_line_byte_exact_count": data_line_byte_exact_count,
        "numeric_data_line_byte_mismatch_count": (
            len(rows) - data_line_byte_exact_count
        ),
        "return_code_counts": dict(sorted(return_codes.items())),
        "unresolved_line_warning_counts": dict(sorted(warning_counts.items())),
        "regeneration_am_identity_counts": dict(
            sorted(regeneration_identities.items())
        ),
        "warning_class_totals": warning_class_totals,
        "error_line_total": error_line_total,
        "normalized_numeric_output_aggregate_sha256": (
            normalized_numeric_digest.hexdigest()
        ),
        "normalized_numeric_output_aggregate_algorithm": (
            "case_id NUL numeric_text_sha256_bytes NUL in case_id bytewise order"
        ),
        "normalized_full_output_aggregate_sha256": (
            normalized_full_output_digest.hexdigest()
        ),
        "normalized_full_output_aggregate_algorithm": (
            "case_id NUL normalized_output_sha256_bytes NUL in case_id bytewise order"
        ),
        "maximum_absolute_differences": {
            key: f64(value) for key, value in maxima.items()
        },
    }


def build_manifest(
    *,
    am_root: Path,
    cache_dir: Path,
    am_cache_dir: Path,
    regeneration: BuildIdentity,
    historical_linux: BuildIdentity,
    compiler: dict[str, Any],
    native_build_command: str | None,
    jobs: int,
    omp_threads: int,
    workflow: dict[str, Any],
    execution_context: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    metrics_bytes: bytes,
) -> bytes:
    build_class = classify_regeneration_build(regeneration, historical_linux)
    context_bytes = json_bytes(execution_context)
    context_inputs = execution_context["inputs"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "package_id": PACKAGE_ID,
            "evidence_date": EVIDENCE_DATE,
            "study": "am12_native_regeneration_exact_numeric_check",
            "scientific_scope": (
                "AM 12.2 annual-profile regeneration evidence; not the exact "
                "legacy am_q25/q50/q75/q95 lineage"
            ),
        },
        "cache_execution_context": {
            "filename": EXECUTION_CONTEXT_NAME,
            "sha256": sha256_bytes(context_bytes),
            "content": execution_context,
        },
        "scope": {
            "season": "annual",
            "water_profile_percentiles": list(PERCENTILES),
            "zenith_angle_deg": {
                "minimum": 10,
                "maximum": 80,
                "step": 2,
                "count_per_profile": len(ZENITH_ANGLES_DEG),
            },
            "derived_elevation_deg": {
                "minimum": 10,
                "maximum": 80,
                "step": 2,
            },
            "case_count": len(metrics_rows),
            "frequency_ghz": {
                "minimum": "0.00000000000000000e+00",
                "maximum": "5.00000000000000000e+02",
                "step": "1.00000000000000002e-02",
                "count": EXPECTED_ROWS,
            },
            "fields_compared": ["f_GHz", "tau_neper", "tx", "Trj_K", "Tb_K"],
        },
        "historical_workflow": workflow,
        "builds": {
            "copied_linux_reference": {
                "path_relative_to_am_root": "am-12.2/bin/am",
                "sha256": historical_linux.sha256,
                "size_bytes": historical_linux.size_bytes,
                "binary_format": historical_linux.binary_format,
                "am_identity": HISTORICAL_VERSION_LINE,
                "compiler_identity_embedded_in_binary": HISTORICAL_COMPILER_ID,
            },
            "regeneration": {
                "classification": build_class,
                "supplied_path": regeneration.supplied_path,
                "resolved_path": regeneration.resolved_path,
                "sha256": regeneration.sha256,
                "size_bytes": regeneration.size_bytes,
                "binary_format": regeneration.binary_format,
                "same_bytes_as_copied_linux_reference": (
                    regeneration.sha256 == historical_linux.sha256
                ),
                "build_command_supplied_by_operator": native_build_command,
                "compiler": compiler,
            },
        },
        "am_source_inventory": context_inputs["am_source_inventory"],
        "annual_profile_inventory": context_inputs["annual_profile_inventory"],
        "copied_reference_output_inventory": context_inputs[
            "copied_reference_output_inventory"
        ],
        "execution": {
            "argv_template": [
                "<am-executable>",
                "LMT_am_inputs/LMT_annual_<percentile>.amc",
                "0",
                "GHz",
                "500",
                "GHz",
                "10",
                "MHz",
                "<zenith-angle-deg>",
                "deg",
                "1.0",
            ],
            "slurm_wrapper_used": False,
            "working_directory_role": "Big_Atmosphere",
            "jobs": jobs,
            "omp_threads_per_process": omp_threads,
            "cache_concurrency_policy": (
                "one process holds a whole-cache exclusive POSIX lock while "
                "running; one ordered worker queue owns each deterministic "
                "AM_CACHE_PATH shard"
            ),
            "environment_overrides": {
                "OMP_NUM_THREADS": str(omp_threads),
                **PINNED_LOCALE,
                "AM_CACHE_PATH": {
                    "root_path_relative_to_cache": am_cache_dir.relative_to(
                        cache_dir
                    ).as_posix(),
                    "runtime_value_policy": (
                        "absolute resolution of --cache-dir/am_cache/"
                        "shard_<matrix-index-mod-jobs>"
                    ),
                },
            },
            "cache_path_recorded_as": "paths relative to --cache-dir",
            "am_cache_root_relative_to_cache": am_cache_dir.relative_to(
                cache_dir
            ).as_posix(),
            "am_cache_sharding": {
                "shard_count": jobs,
                "assignment": (
                    "(percentile-major matrix index with zenith angle minor) "
                    "modulo shard_count"
                ),
                "within_shard_order": (
                    "smoke-gate encounter order, completion barrier, then "
                    "remaining percentile-major/zenith-angle encounter order"
                ),
                "ownership": (
                    "one ordered worker queue per shard per phase; phases do not "
                    "overlap"
                ),
            },
            "whole_cache_lock": {
                "filename": CACHE_LOCK_NAME,
                "run_mode": "nonblocking POSIX exclusive lock",
                "cache_only_mode": "nonblocking POSIX shared lock",
            },
            "staged_smoke_gate": {
                "cases": [
                    "LMT_annual_95_za10",
                    "LMT_annual_95_za70",
                ],
                "requirement": (
                    "all five parsed fields exactly equal before remaining "
                    "178 cases execute"
                ),
            },
            "host": execution_context["execution_host"],
            "accepted_return_contract": (
                "return code 0, or return code 1 only with the canonical "
                "unresolved-line warning count 86/87/88 and 50001 rows; "
                "cache, unknown-warning, and error diagnostics are rejected"
            ),
            "committed_output_digest_policy": (
                "commit per-case numeric-text and normalized warning-bearing "
                "combined-output SHA-256 values plus aggregates; raw combined-"
                "output SHA-256 values remain in execution sidecars"
            ),
            "combined_output_normalization": execution_context["output_normalization"],
        },
        "results": summarize_metrics(metrics_rows),
        "rejected_predecessor_attempt": {
            "status": "excluded_from_canonical_evidence",
            "external_cache_basename": (
                "sci_cal_001_am12_2_native_matrix_20260801_root"
            ),
            "reason": (
                "concurrent processes shared one AM_CACHE_PATH and emitted "
                "cache mutation warnings"
            ),
            "case_count": 180,
            "all_numeric_data_lines_exact": True,
            "cases_with_cache_warning": 28,
            "cache_insert_as_mru_warning_line_count": 22,
            "cache_promote_to_mru_warning_line_count": 9,
            "canonical_artifacts_use_this_attempt": False,
        },
        "superseded_predecessor_attempt": {
            "status": "superseded_by_stronger_provenance_contract",
            "external_cache_basename": (
                "sci_cal_001_am12_2_native_matrix_clean_sharded_20260801_root"
            ),
            "reason": (
                "the matrix was numerically exact and diagnostic-clean, but "
                "its cache did not bind the runner/source/reference/compiler/"
                "host context or committed normalized warning-bearing output"
            ),
            "case_count": 180,
            "all_numeric_data_lines_exact": True,
            "canonical_artifacts_use_this_attempt": False,
        },
        "artifacts": {
            "metrics_csv": {
                "filename": METRICS_NAME,
                "sha256": sha256_bytes(metrics_bytes),
            },
            "raw_outputs": "stored below --cache-dir and not committed",
            "execution_sidecars": "stored below --cache-dir and not committed",
        },
        "security": {
            "uploader_logs_read": False,
            "uploader_logs_or_credentials_copied": False,
            "network_access": False,
            "unity_access": False,
        },
    }
    return json_bytes(payload)


def build_report(
    *,
    manifest: dict[str, Any],
    manifest_bytes: bytes,
) -> bytes:
    results = manifest["results"]
    regeneration = manifest["builds"]["regeneration"]
    compiler = regeneration["compiler"]
    verdict = (
        "all parsed fields match exactly"
        if results["all_cases_exact"]
        else "one or more parsed fields differ"
    )
    lines = [
        "# SCI-CAL-001 AM 12.2 native regeneration report",
        "",
        "## Verdict",
        "",
        f"The complete `{results['case_count']}`-case annual AM 12.2 matrix was structurally valid; {verdict}. Exact parsed-field matches: `{results['exact_case_count']}`; parsed-field mismatches: `{results['mismatch_case_count']}`. Numeric data-line byte matches: `{results['numeric_data_line_byte_exact_count']}`; byte mismatches: `{results['numeric_data_line_byte_mismatch_count']}`.",
        "",
        "This is a software/numerical regeneration check for the copied AM 12.2 annual profiles. It does not establish that these profiles are the exact legacy `am_q25/q50/q75/q95` inputs, and it does not select or authorize an atmosphere operator.",
        "",
        "A predecessor parallel attempt used one shared `AM_CACHE_PATH` and was rejected from canonical evidence after 28 of 180 cases emitted cache-mutation warnings (22 `insert_as_mru`, 9 `promote_to_mru`, with overlap). Its numeric data lines were still exact, but those warnings fail the software-execution contract. A second numerically exact sharded attempt with zero cache, unknown-warning, or error diagnostics was superseded because it did not yet bind the complete execution context or commit warning-bearing output identity. The canonical matrix reported here satisfies both requirements; its unresolved-line warnings and status 1 remain explicit and are not described as a clean software success.",
        "",
        "## Build identity",
        "",
        f"The copied Linux reference binary is SHA-256 `{manifest['builds']['copied_linux_reference']['sha256']}` and identifies itself as `{HISTORICAL_VERSION_LINE}`. The regeneration executable is classified as `{regeneration['classification']}`, format `{regeneration['binary_format']}`, SHA-256 `{regeneration['sha256']}`. Same bytes as the copied Linux binary: `{str(regeneration['same_bytes_as_copied_linux_reference']).lower()}`.",
        "",
        f"Native compiler provenance status: `{compiler['status']}`. Native build command supplied: `{regeneration['build_command_supplied_by_operator']}`.",
        "",
        "## Execution and comparison contract",
        "",
        "Each case used the historical AM argv body `profile 0 GHz 500 GHz 10 MHz ZA deg 1.0`, without `srun`, with pinned `LANG=C`, `LC_ALL=C`, the requested `OMP_NUM_THREADS`, and a deterministically assigned cache shard below `--cache-dir/am_cache`. One whole-cache POSIX writer lock excludes other processes. Within each of two nonoverlapping phases, one ordered worker queue owns each shard; the q95 ZA10/ZA70 smoke phase must complete exactly before the remaining matrix begins. Generated combined stdout/stderr and execution sidecars remain in the external cache. The comparison parses all 50,001 rows and requires exact binary64 equality independently for frequency, tau, transmission, Rayleigh-Jeans temperature, and brightness temperature.",
        "",
        "Committed identity includes both normalized numeric data text and normalized warning-bearing combined output. The latter replaces only the volatile runtime and dcache-counter header lines; it preserves the AM identity, configuration, numeric grid, warning lines, and all other output. Each sidecar binds its raw and normalized output digests to the immutable cache execution-context SHA-256.",
        "",
        "AM return code 1 is accepted only when the complete grid accompanies the canonical unresolved-narrow-line warning with count 86, 87, or 88. Cache insert/promote warnings, unknown warning classes, error lines, and other nonzero statuses fail closed.",
        "",
        "## Aggregate differences",
        "",
        "| Field | Maximum absolute difference |",
        "| --- | ---: |",
    ]
    for field, value in results["maximum_absolute_differences"].items():
        lines.append(f"| `{field}` | `{value}` |")
    lines.extend(
        [
            "",
            f"Return-code counts: `{json.dumps(results['return_code_counts'], sort_keys=True)}`. Warning-count distribution: `{json.dumps(results['unresolved_line_warning_counts'], sort_keys=True)}`.",
            "",
            f"Regeneration AM identity distribution: `{json.dumps(results['regeneration_am_identity_counts'], sort_keys=True)}`.",
            "",
            f"Warning-class totals: `{json.dumps(results['warning_class_totals'], sort_keys=True)}`. Error-line total: `{results['error_line_total']}`. Normalized numeric-output aggregate SHA-256: `{results['normalized_numeric_output_aggregate_sha256']}`. Normalized warning-bearing full-output aggregate SHA-256: `{results['normalized_full_output_aggregate_sha256']}`.",
            "",
            "## Provenance closure",
            "",
            f"`{METRICS_NAME}` SHA-256 is `{manifest['artifacts']['metrics_csv']['sha256']}`. `{MANIFEST_NAME}` SHA-256 is `{sha256_bytes(manifest_bytes)}`. The external cache execution context is `{manifest['cache_execution_context']['filename']}` SHA-256 `{manifest['cache_execution_context']['sha256']}`, and its complete content is copied into the committed manifest. It binds the runner, copied and regeneration binaries, compiler and build command, AM source, five annual profiles, all 180 copied reference grids, frozen historical scripts, argv, run scope, ordered shard topology, locale, and actual execution host.",
            "",
            "Uploader logs and credentials are deliberately excluded. No network or Unity access is part of this workflow.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def collect_runs(
    cases: list[Case],
    *,
    check: bool,
    executable: Path,
    executable_identity: BuildIdentity,
    big_atmosphere_root: Path,
    cache_dir: Path,
    am_cache_dir: Path,
    jobs: int,
    omp_threads: int,
    execution_host: dict[str, str],
    execution_context_sha256: str,
) -> list[RunResult]:
    if check:
        return [
            load_cached_case(
                case,
                executable_identity=executable_identity,
                big_atmosphere_root=big_atmosphere_root,
                cache_dir=cache_dir,
                am_cache_dir=case_am_cache_dir(am_cache_dir, case, jobs),
                omp_threads=omp_threads,
                execution_host=execution_host,
                execution_context_sha256=execution_context_sha256,
            )
            for case in cases
        ]

    def execute(case: Case) -> RunResult:
        return run_case(
            case,
            executable=executable,
            executable_identity=executable_identity,
            big_atmosphere_root=big_atmosphere_root,
            cache_dir=cache_dir,
            am_cache_dir=case_am_cache_dir(am_cache_dir, case, jobs),
            omp_threads=omp_threads,
            execution_host=execution_host,
            execution_context_sha256=execution_context_sha256,
        )

    if jobs == 1:
        return [execute(case) for case in cases]
    shard_cases: list[list[Case]] = [[] for _ in range(jobs)]
    for case in cases:
        shard_cases[case_cache_shard_index(case, jobs)].append(case)

    def execute_shard(ordered_cases: list[Case]) -> list[RunResult]:
        completed = []
        for case in ordered_cases:
            result = execute(case)
            completed.append(result)
            print(f"completed {case.case_id}", flush=True)
        return completed

    completed: dict[str, RunResult] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(jobs, sum(bool(items) for items in shard_cases))
    ) as executor:
        futures = [
            executor.submit(execute_shard, ordered_cases)
            for ordered_cases in shard_cases
            if ordered_cases
        ]
        for future in futures:
            for result in future.result():
                completed[result.case.case_id] = result
    return [completed[case.case_id] for case in cases]


def expected_artifacts(
    *,
    am_root: Path,
    cache_dir: Path,
    am_cache_dir: Path,
    regeneration: BuildIdentity,
    historical_linux: BuildIdentity,
    compiler: dict[str, Any],
    native_build_command: str | None,
    jobs: int,
    omp_threads: int,
    workflow: dict[str, Any],
    execution_context: dict[str, Any],
    runs: list[RunResult],
) -> tuple[dict[str, bytes], bool]:
    metrics_rows = [
        compare_case(run, am_root=am_root, cache_dir=cache_dir) for run in runs
    ]
    metrics_bytes = render_csv(metrics_rows)
    manifest_bytes = build_manifest(
        am_root=am_root,
        cache_dir=cache_dir,
        am_cache_dir=am_cache_dir,
        regeneration=regeneration,
        historical_linux=historical_linux,
        compiler=compiler,
        native_build_command=native_build_command,
        jobs=jobs,
        omp_threads=omp_threads,
        workflow=workflow,
        execution_context=execution_context,
        metrics_rows=metrics_rows,
        metrics_bytes=metrics_bytes,
    )
    manifest = json.loads(manifest_bytes)
    report_bytes = build_report(manifest=manifest, manifest_bytes=manifest_bytes)
    return {
        METRICS_NAME: metrics_bytes,
        MANIFEST_NAME: manifest_bytes,
        REPORT_NAME: report_bytes,
    }, bool(manifest["results"]["all_cases_exact"])


def smoke_cases() -> list[Case]:
    return [Case(95, 10), Case(95, 70)]


def report_smoke_gate(runs: list[RunResult], *, am_root: Path, cache_dir: Path) -> bool:
    rows = [compare_case(run, am_root=am_root, cache_dir=cache_dir) for run in runs]
    all_exact = all(row["all_fields_exact_equal"] == "true" for row in rows)
    for row in rows:
        print(
            "smoke gate "
            f"{row['case_id']}: parsed_fields_exact="
            f"{row['all_fields_exact_equal']}, numeric_data_lines_byte_equal="
            f"{row['numeric_data_lines_byte_equal']}"
        )
    if not all_exact:
        print(
            "preregistered q95 smoke gate failed; remaining cases were not run",
            file=sys.stderr,
        )
    return all_exact


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--am-root",
        type=Path,
        default=DEFAULT_AM_ROOT,
        help="read-only AM root containing am-12.2 and Big_Atmosphere",
    )
    parser.add_argument(
        "--am-executable",
        type=Path,
        help="AM executable to run; defaults to the copied Linux binary",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="external writable cache for generated raw outputs and sidecars",
    )
    parser.add_argument("--jobs", type=positive_integer, default=1)
    parser.add_argument("--omp-threads", type=positive_integer, default=1)
    parser.add_argument(
        "--compiler-executable",
        type=Path,
        help="compiler explicitly used to build a distinct native AM executable",
    )
    parser.add_argument(
        "--native-build-command",
        help="exact command used to build a distinct native AM executable",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate cache and checked-in artifacts without running AM",
    )
    parser.add_argument(
        "--regenerate-from-cache",
        action="store_true",
        help=(
            "validate the complete external cache and rewrite package artifacts "
            "without running AM"
        ),
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "run or check only annual q95 ZA10/ZA70; do not write canonical "
            "package artifacts"
        ),
    )
    args = parser.parse_args()
    if args.check and args.regenerate_from_cache:
        parser.error("--check and --regenerate-from-cache are mutually exclusive")
    cache_only = args.check or args.regenerate_from_cache

    am_root = args.am_root.expanduser().resolve(strict=True)
    big_root = am_root / "Big_Atmosphere"
    cache_dir = args.cache_dir.expanduser().resolve()
    package_dir = Path(__file__).resolve().parent
    if is_relative_to(cache_dir, am_root):
        raise RuntimeError("--cache-dir must not write inside the read-only AM root")
    if is_relative_to(cache_dir, package_dir):
        raise RuntimeError(
            "--cache-dir must not place raw outputs in the evidence package"
        )
    if cache_only:
        if not cache_dir.is_dir():
            raise RuntimeError(
                f"cache-only operation requires an existing cache directory: {cache_dir}"
            )
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_lock_handle = acquire_cache_lock(cache_dir, exclusive=not cache_only)
    if not cache_only:
        existing = sorted(
            path.name for path in cache_dir.iterdir() if path.name != CACHE_LOCK_NAME
        )
        if existing:
            raise RuntimeError(
                "execution requires a fresh external cache; found existing "
                f"entries below {cache_dir}: {existing}"
            )
    am_cache_dir = (cache_dir / "am_cache").resolve()
    if not is_relative_to(am_cache_dir, cache_dir):
        raise RuntimeError("--cache-dir/am_cache must resolve within --cache-dir")
    if cache_only and not am_cache_dir.is_dir():
        raise RuntimeError(
            f"cache-only operation requires the recorded AM cache directory: {am_cache_dir}"
        )
    for shard_index in range(args.jobs):
        shard = am_cache_dir / f"shard_{shard_index:02d}"
        if cache_only:
            if not shard.is_dir():
                raise RuntimeError(
                    f"cache-only operation requires the recorded AM cache shard: {shard}"
                )
        else:
            shard.mkdir(parents=True, exist_ok=True)

    executable = (
        args.am_executable.expanduser()
        if args.am_executable is not None
        else am_root / "am-12.2/bin/am"
    ).resolve(strict=True)
    regeneration = build_identity(executable)
    historical_linux = build_identity(am_root / "am-12.2/bin/am")
    if historical_linux.sha256 != HISTORICAL_LINUX_BINARY_SHA256:
        raise RuntimeError("copied historical Linux AM binary digest mismatch")
    build_class = classify_regeneration_build(regeneration, historical_linux)
    if build_class != "copied_linux_reference_binary_reexecution":
        missing = []
        if args.compiler_executable is None:
            missing.append("--compiler-executable")
        if not args.native_build_command:
            missing.append("--native-build-command")
        if missing:
            raise RuntimeError(
                "distinct native AM builds require complete provenance: missing "
                + ", ".join(missing)
            )

    workflow = validate_frozen_workflow(am_root)
    compiler = compiler_identity(args.compiler_executable)
    context_path = cache_dir / EXECUTION_CONTEXT_NAME
    run_scope = "smoke_only" if args.smoke_only else "full_annual_matrix"
    if cache_only:
        if not context_path.is_file():
            raise RuntimeError(
                f"cache-only operation requires execution context: {context_path}"
            )
        execution_context = load_execution_context(context_path)
        execution_host = execution_context["execution_host"]
        expected_context = build_execution_context(
            am_root=am_root,
            regeneration=regeneration,
            historical_linux=historical_linux,
            compiler=compiler,
            native_build_command=args.native_build_command,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            workflow=workflow,
            execution_host=execution_host,
            run_scope=run_scope,
        )
        if execution_context != expected_context:
            raise RuntimeError(
                "cached execution context does not match the current runner, "
                "inputs, references, build provenance, or requested parameters"
            )
    else:
        execution_host = host_identity()
        execution_context = build_execution_context(
            am_root=am_root,
            regeneration=regeneration,
            historical_linux=historical_linux,
            compiler=compiler,
            native_build_command=args.native_build_command,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            workflow=workflow,
            execution_host=execution_host,
            run_scope=run_scope,
        )
        atomic_write(context_path, json_bytes(execution_context))
    execution_context_sha256 = sha256_path(context_path)
    cases = [
        Case(percentile, zenith_angle)
        for percentile in PERCENTILES
        for zenith_angle in ZENITH_ANGLES_DEG
    ]
    staged_cases = smoke_cases()
    if args.smoke_only:
        runs = collect_runs(
            staged_cases,
            check=cache_only,
            executable=executable,
            executable_identity=regeneration,
            big_atmosphere_root=big_root,
            cache_dir=cache_dir,
            am_cache_dir=am_cache_dir,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            execution_host=execution_host,
            execution_context_sha256=execution_context_sha256,
        )
        return 0 if report_smoke_gate(runs, am_root=am_root, cache_dir=cache_dir) else 1

    if cache_only:
        runs = collect_runs(
            cases,
            check=True,
            executable=executable,
            executable_identity=regeneration,
            big_atmosphere_root=big_root,
            cache_dir=cache_dir,
            am_cache_dir=am_cache_dir,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            execution_host=execution_host,
            execution_context_sha256=execution_context_sha256,
        )
    else:
        staged_runs = collect_runs(
            staged_cases,
            check=False,
            executable=executable,
            executable_identity=regeneration,
            big_atmosphere_root=big_root,
            cache_dir=cache_dir,
            am_cache_dir=am_cache_dir,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            execution_host=execution_host,
            execution_context_sha256=execution_context_sha256,
        )
        if not report_smoke_gate(staged_runs, am_root=am_root, cache_dir=cache_dir):
            return 1
        staged_case_ids = {case.case_id for case in staged_cases}
        remaining_cases = [
            case for case in cases if case.case_id not in staged_case_ids
        ]
        remaining_runs = collect_runs(
            remaining_cases,
            check=False,
            executable=executable,
            executable_identity=regeneration,
            big_atmosphere_root=big_root,
            cache_dir=cache_dir,
            am_cache_dir=am_cache_dir,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            execution_host=execution_host,
            execution_context_sha256=execution_context_sha256,
        )
        runs_by_id = {run.case.case_id: run for run in staged_runs + remaining_runs}
        runs = [runs_by_id[case.case_id] for case in cases]
    artifacts, all_exact = expected_artifacts(
        am_root=am_root,
        cache_dir=cache_dir,
        am_cache_dir=am_cache_dir,
        regeneration=regeneration,
        historical_linux=historical_linux,
        compiler=compiler,
        native_build_command=args.native_build_command,
        jobs=args.jobs,
        omp_threads=args.omp_threads,
        workflow=workflow,
        execution_context=execution_context,
        runs=runs,
    )

    stale = False
    for name, expected in artifacts.items():
        path = package_dir / name
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                print(
                    f"stale or missing native-regeneration artifact: {path}",
                    file=sys.stderr,
                )
                stale = True
        else:
            atomic_write(path, expected)
            print(f"wrote {path}")
    if stale:
        return 1
    if not all_exact:
        print("native AM regeneration contains numeric mismatches", file=sys.stderr)
        return 1
    fcntl.flock(cache_lock_handle.fileno(), fcntl.LOCK_UN)
    cache_lock_handle.close()
    print("native AM regeneration exactly matches all copied parsed grids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
