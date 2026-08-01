#!/usr/bin/env python3
"""Probe the post-hoc SCI-CAL-001 AM-12.2 H2O-scale hypothesis.

This evidence-only driver implements diagnostic P1 in
``FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md``.  It never edits an AMC profile and
varies only its documented final ``Nscale troposphere h2o`` argument.  Raw AM
stdout/stderr and execution sidecars are kept below the caller-supplied cache
directory; only deterministic summaries are written beside this script.

The default mode executes AM.  ``--check`` is deliberately cache-only: it
loads and validates every raw output and sidecar needed to reconstruct the
artifacts, then checks committed artifact bytes without launching AM.  An
execution requires a fresh external cache and holds a whole-cache exclusive
POSIX lock; cache-only verification holds a shared lock.  A canonical
execution context binds the runner, host, build, inputs, locale, argv, and
concurrency settings into every sidecar.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
PACKAGE_ID = "SCI-CAL-001"
SCHEMA_VERSION = "sci-cal-001-am12-h2o-scale-hypothesis-v2"
ARTIFACT_SCHEMA_VERSION = "sci-cal-001-am12-h2o-scale-hypothesis-v3"
EVIDENCE_DATE = "2026-08-01"
EXECUTION_CONTEXT_NAME = "execution_context.json"
CACHE_LOCK_NAME = ".h2o_scale_hypothesis.lock"
PINNED_LOCALE = {"LANG": "C", "LC_ALL": "C"}

DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
DEFAULT_LEGACY_SOURCE_DIR = Path(
    "/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity"
)
SEASONS = ("annual", "DJF", "MAM", "JJA", "SON")
PERCENTILES = (5, 25, 50, 75, 95)
PROFILE_STEMS = tuple(
    f"LMT_{season}_{percentile}" for season in SEASONS for percentile in PERCENTILES
)
ELEVATIONS_DEG = np.arange(20, 82, 2, dtype=np.float64)
ZENITH_ANGLES_DEG = tuple(int(90 - value) for value in ELEVATIONS_DEG)
FREQUENCY_GHZ = np.arange(50001, dtype=np.float64) / 100.0
REFERENCE_FREQUENCY_GHZ = 225.00
BAND_FREQUENCIES_GHZ = {
    "a1100": 272.73,
    "a1400": 214.29,
    "a2000": 150.00,
}

REPAIR_BASE_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
REPAIR_LINE_EVIDENCE_HEAD = "ae99be1cef8c390d0e7490835ffca1f31da7ebc0"
CALIBRATE_REL = Path("include/citlali/core/timestream/rtc/calibrate.h")
PHASE0_SCRIPT_REL = Path(
    "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py"
)
FROZEN_REPAIR_INPUTS = {
    CALIBRATE_REL: ("d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"),
    PHASE0_SCRIPT_REL: (
        "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
    ),
}
FROZEN_PROTOCOL_INPUTS = {
    Path("FOLLOWUP_STUDY_PREREGISTRATION.md"): {
        "sha256": ("65935dbc906317e984cf2ae8b35c5868a3f216eca2ec6290f2887976892d8457"),
        "size_bytes": 8528,
    },
    Path("FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md"): {
        "sha256": ("0d47c11479a1ba0176babd3ea285e2871edbb1341406b6b044cbc53114c51a1d"),
        "size_bytes": 5236,
    },
    Path("FOLLOWUP_STUDY_DEVIATION_LOG.md"): {
        "sha256": ("a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036"),
        "size_bytes": 2066,
    },
}
EXPECTED_TARGET_TRANSMISSIONS = {
    "am_q25": "0.9500275",
    "am_q50": "0.9142065",
    "am_q75": "0.8515054",
    "am_q95": "0.7337698",
}
TARGETS = tuple(EXPECTED_TARGET_TRANSMISSIONS)

COPIED_MANIFEST_NAME = "copied_am_manifest.json"
EXPECTED_COPIED_MANIFEST_FILE_SHA256 = (
    "ef525bc5f2883f181cccd43f585b4e398d227f2a26eeca0ed90cf4f5922f520f"
)
COPIED_CANONICAL_PRODUCT_MANIFEST_SHA256 = (
    "18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac"
)
EXPECTED_PROFILE_AGGREGATE_SHA256 = (
    "b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c"
)
AM_SOURCE_CONTRACTS = {
    "am-12.2/src/config.c": (
        "6e57faf4e58a536c8fdb66291c9a186f0f3c01356ee6b00a9677eea6c7fbce79"
    ),
    "am-12.2/src/nscale.c": (
        "c00a333583988c241fc80a2648378914361e0eaf8fdd8f1fc112b7d2ff913d06"
    ),
    "Big_Atmosphere/01_do_am_runs.sh": (
        "02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3"
    ),
    "Big_Atmosphere/generateAmModels.py": (
        "29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6"
    ),
    "Big_Atmosphere/make_npz.py": (
        "3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5"
    ),
}
HISTORICAL_LINUX_BINARY_SHA256 = (
    "3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c"
)
LEGACY_RAW_SOURCES = {
    "am_q25": {
        "filename": "amLMT25.npz",
        "sha256": "6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b",
        "md5": "008d7fa69aff187a9edf419f3d961b4c",
        "tolteca_datafile_id": "454",
    },
    "am_q50": {
        "filename": "amLMT50.npz",
        "sha256": "1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81",
        "md5": "6ec393672be8af4dfa06a3f4cf9aa32e",
        "tolteca_datafile_id": "455",
    },
    "am_q75": {
        "filename": "amLMT75.npz",
        "sha256": "adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e",
        "md5": "d6cf4bb27008179ec491864388deac58",
        "tolteca_datafile_id": "456",
    },
}
MISSING_Q95 = {
    "tolteca_datafile_id": "461",
    "expected_md5": "0ca7b331823237767d26016d19bffb3d",
    "status": "registered_raw_grid_absent_not_retrieved",
}

SCALES_NAME = "h2o_scale_hypothesis_scales.csv"
METRICS_NAME = "h2o_scale_hypothesis_metrics.csv"
COEFFICIENTS_NAME = "h2o_scale_hypothesis_coefficients.csv"
MANIFEST_NAME = "h2o_scale_hypothesis_manifest.json"
REPORT_NAME = "H2O_SCALE_HYPOTHESIS_REPORT.md"
OUTPUT_NAMES = (
    SCALES_NAME,
    METRICS_NAME,
    COEFFICIENTS_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
)
INTERRUPTED_V2_DISPOSITION = {
    "status": "noncanonical_interrupted_excluded_from_v3_evidence",
    "external_cache_basename": "sci_cal_001_h2o_scale_p1_20260801_root_v2",
    "termination": (
        "stopped after cache-provenance review; no related probe or AM process remained"
    ),
    "reason": (
        "the process had no whole-cache cross-process lock and its sidecars did "
        "not bind an immutable execution context"
    ),
    "observed_partial_cache_inventory": {
        "raw_output_file_count": 12455,
        "execution_sidecar_file_count": 12455,
        "scale_trace_file_count": 100,
        "execution_sidecar_stage_counts": {
            "anchor_225ghz_el80": 9792,
            "direct_full_grid_all_hypotheses": 1764,
            "direct_full_grid_selected_transmission_rank1": 124,
            "full_grid_scale0_construction_endpoint": 775,
        },
        "direct_fitted_scale_expected_stage_counts": {
            "direct_full_grid_all_hypotheses": 2976,
            "direct_full_grid_selected_transmission_rank1": 124,
        },
        "total_direct_fitted_scale_expected_count": 3100,
        "total_direct_fitted_scale_observed_count": 1888,
        "targeted_sigint_failure_inventory": {
            "failure_sidecar_count": 3,
            "empty_combined_output_file_count": 3,
            "return_code": -2,
            "profile": "LMT_JJA_5",
            "target": "am_q25",
            "zenith_angles_deg": [10, 50, 54],
            "disposition": "termination records excluded from v3 evidence",
        },
    },
    "preservation": "external cache retained read-only; not deleted",
    "canonical_v3_artifacts_or_rankings_use_this_attempt": False,
}
INTERRUPTED_FIRST_V3_DEVELOPMENT_DISPOSITION = {
    "status": "noncanonical_development_attempt_excluded",
    "external_cache_basename": (
        "sci_cal_001_h2o_scale_p1_context_v3_final_20260801_root"
    ),
    "execution_context_sha256": (
        "b6f7f88175983b49d2113bdbe626f115e7ced1da6922d7c5dea9636b64217fdd"
    ),
    "runner_sha256": (
        "dae40f4484dead989d4cc559ea7cc52f9af844651c52949a6399214373d82625"
    ),
    "termination": (
        "stopped during anchor inference after pre-full-grid memory-retention "
        "review; no related probe or AM process remained"
    ),
    "reason": (
        "the in-process digest inventory retained complete ParsedOutput sample "
        "arrays and would have added approximately 7.75 GB across the final "
        "full-grid runs"
    ),
    "observed_partial_cache_inventory": {
        "successful_raw_output_file_count": 1811,
        "successful_execution_sidecar_file_count": 1811,
        "scale_trace_file_count": 16,
        "targeted_sigint_failure_inventory": {
            "empty_combined_output_file_count": 6,
            "complete_failure_sidecar_count": 3,
            "empty_atomic_failure_sidecar_temporary_file_count": 3,
            "return_code_where_complete": -2,
            "profiles": [
                "LMT_JJA_25",
                "LMT_JJA_50",
                "LMT_JJA_75",
                "LMT_SON_5",
                "LMT_SON_25",
                "LMT_SON_75",
            ],
            "target": "am_q25",
            "zenith_angle_deg": 10,
            "disposition": "termination records excluded from evidence",
        },
    },
    "preservation": "external cache retained untouched; never reused",
    "canonical_artifacts_or_rankings_use_this_attempt": False,
}

FLOAT_TOKEN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\Z")
VERSION_LINE = re.compile(r"^# (?P<identity>am version .+)$", re.MULTILINE)
UNRESOLVED_WARNING = re.compile(
    r"^! Warning: Encountered in-band lines narrower than the frequency\n"
    r"^!          grid spacing\.  The output configuration data includes\n"
    r"^!          the unresolved line count after each column definition\n"
    r"^!          for which this occurred\.  Count: (?P<count>\d+)$",
    re.MULTILINE,
)
WARNING_HEADER = re.compile(r"^! Warning: (?P<message>.+)$", re.MULTILINE)
ERROR_LINE = re.compile(r"^! Error: .*$", re.MULTILINE)
ROOT_ITERATIONS = 48
MAX_BRACKET_EXPANSIONS = 64
COPIED_TX_ABSOLUTE_PRINT_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class BuildIdentity:
    supplied_path: str
    resolved_path: str
    size_bytes: int
    sha256: str
    binary_format: str


@dataclass(frozen=True)
class RunSpec:
    stage: str
    profile: str
    target: str
    f_min_centi_ghz: int
    f_max_centi_ghz: int
    zenith_angle_deg: int
    scale_decimal: str

    @property
    def expected_rows(self) -> int:
        return self.f_max_centi_ghz - self.f_min_centi_ghz + 1

    @property
    def elevation_deg(self) -> int:
        return 90 - self.zenith_angle_deg

    def request_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "profile": self.profile,
            "target": self.target,
            "f_min_centi_ghz": self.f_min_centi_ghz,
            "f_max_centi_ghz": self.f_max_centi_ghz,
            "step_mhz": 10,
            "zenith_angle_deg": self.zenith_angle_deg,
            "scale_decimal": self.scale_decimal,
        }


@dataclass(frozen=True)
class ParsedOutput:
    samples: np.ndarray
    version_identity: str
    warning_count: int | None
    numeric_text_sha256: str
    normalized_output_sha256: str
    unresolved_column_warning_line_count: int
    unresolved_summary_warning_line_count: int
    other_warning_line_count: int
    error_line_count: int


@dataclass(frozen=True)
class RunResult:
    spec: RunSpec
    parsed: ParsedOutput
    return_code: int
    raw_sha256: str
    sidecar: dict[str, Any]
    cache_id: str


@dataclass(frozen=True)
class RunObservation:
    """Lightweight immutable identity retained after a run leaves local scope."""

    cache_id: str
    spec: RunSpec
    return_code: int
    am_version_identity: str
    unresolved_line_warning_count: int | None
    unresolved_column_warning_line_count: int
    unresolved_summary_warning_line_count: int
    other_warning_line_count: int
    error_line_count: int
    numeric_text_sha256: str
    normalized_output_sha256: str


@dataclass(frozen=True)
class ScaleSolution:
    target: str
    profile: str
    target_transmission: float
    target_tau: float
    scale_decimal: str
    scale_value: float
    exact_parsed_transmission_match: bool
    method: str
    scale0: RunResult
    scale1: RunResult
    fitted: RunResult
    copied_scale1_tau: float
    copied_scale1_transmission: float
    affine_initial_scale: float
    plateau_lower_outside_scale: float | None
    plateau_lower_inside_scale: float | None
    plateau_upper_inside_scale: float | None
    plateau_upper_outside_scale: float | None
    trace_relative_path: str
    trace_sha256: str
    trace_evaluation_count: int


def digest_path(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    return digest_path(path, "sha256")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f64(value: float) -> str:
    return format(float(value), ".17e")


def optional_f64(value: float | None) -> str:
    return "" if value is None else f64(value)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("cannot render an empty CSV")
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


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


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
        raise RuntimeError(f"not a regular executable: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise RuntimeError(f"not executable: {resolved}")
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


def compiler_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_supplied",
            "note": "required for complete distinct-build provenance",
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
        **build_identity_payload(identity),
        "version_command_return_code": completed.returncode,
        "version_output_sha256": sha256_bytes(version_bytes),
        "version_output": version_bytes.decode("utf-8", errors="replace").strip(),
    }


def source_files(source_root: Path) -> list[Path]:
    named_files = {"_README_", "INSTALLING", "LICENSE", "Makefile", "REFERENCES"}
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
    entries: list[dict[str, Any]] = []
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


def normalize_combined_output(data: bytes) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("cannot normalize non-UTF-8 AM output") from error
    lines = []
    for line in text.splitlines():
        if line.startswith("# run time "):
            lines.append("# run time <volatile>")
        elif line.startswith("# dcache hit: "):
            lines.append("# dcache counters <volatile>")
        else:
            lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8")


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


def load_phase0_source() -> Any:
    for relative, expected in FROZEN_REPAIR_INPUTS.items():
        actual = sha256_path(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"frozen repair input mismatch for {relative}: {actual} != {expected}"
            )
    path = REPO_ROOT / PHASE0_SCRIPT_REL
    spec = importlib.util.spec_from_file_location("sci_cal_001_h2o_scale_phase0", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen phase-0 parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = module.parse_source(REPO_ROOT)
    for target, expected in EXPECTED_TARGET_TRANSMISSIONS.items():
        if source.transmissions.get(target) != expected:
            raise RuntimeError(
                f"unexpected repair-base T225 literal for {target}: "
                f"{source.transmissions.get(target)!r} != {expected!r}"
            )
    return source


def validate_protocol_inputs() -> dict[str, Any]:
    paths = []
    for relative, expected in sorted(
        FROZEN_PROTOCOL_INPUTS.items(), key=lambda item: item[0].as_posix()
    ):
        path = PACKAGE_DIR / relative
        actual_size = path.stat().st_size
        actual_sha256 = sha256_path(path)
        if actual_size != expected["size_bytes"]:
            raise RuntimeError(
                f"frozen protocol size mismatch for {relative}: "
                f"{actual_size} != {expected['size_bytes']}"
            )
        if actual_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"frozen protocol SHA-256 mismatch for {relative}: "
                f"{actual_sha256} != {expected['sha256']}"
            )
        paths.append(path)
    return inventory_files(PACKAGE_DIR, paths)


def inventory_profiles(am_root: Path) -> dict[str, Any]:
    root = am_root / "Big_Atmosphere/LMT_am_inputs"
    expected_names = {f"{stem}.amc" for stem in PROFILE_STEMS}
    actual_names = {path.name for path in root.glob("LMT_*.amc") if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            "copied AMC set mismatch: "
            f"missing={sorted(expected_names - actual_names)!r}, "
            f"extra={sorted(actual_names - expected_names)!r}"
        )
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for name in sorted(expected_names):
        path = root / name
        data = path.read_bytes()
        text = data.decode("utf-8")
        if text.splitlines().count("Nscale troposphere h2o %9") != 1:
            raise RuntimeError(f"unexpected H2O Nscale contract in {path}")
        if "trop_h2o_scale_factor" not in text:
            raise RuntimeError(f"missing documented final scale argument in {path}")
        digest = sha256_bytes(data)
        relative = path.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
        entries.append(
            {
                "path_relative_to_profile_root": relative,
                "size_bytes": len(data),
                "sha256": digest,
            }
        )
        total_bytes += len(data)
    if aggregate.hexdigest() != EXPECTED_PROFILE_AGGREGATE_SHA256:
        raise RuntimeError(
            "copied AMC aggregate mismatch: "
            f"{aggregate.hexdigest()} != {EXPECTED_PROFILE_AGGREGATE_SHA256}"
        )
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "aggregate_algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "files": entries,
    }


def validate_am_contract_files(am_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative_string, expected in AM_SOURCE_CONTRACTS.items():
        relative = Path(relative_string)
        path = am_root / relative
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen AM source/workflow mismatch for {relative}: "
                f"{actual} != {expected}"
            )
        result[relative_string] = {"sha256": actual}
    config_text = (am_root / "am-12.2/src/config.c").read_text(encoding="utf-8")
    nscale_text = (am_root / "am-12.2/src/nscale.c").read_text(encoding="utf-8")
    if "get_nonneg_dbl_val(\n                &Nscale" not in config_text:
        raise RuntimeError("AM source no longer establishes nonnegative Nscale")
    if "ptr->tagnum == 0 || ptr->tagnum == tagnum" not in nscale_text:
        raise RuntimeError("AM source no longer establishes tag-specific Nscale")
    return result


def load_copied_inventory(am_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = PACKAGE_DIR / COPIED_MANIFEST_NAME
    actual_file_sha256 = sha256_path(manifest_path)
    if actual_file_sha256 != EXPECTED_COPIED_MANIFEST_FILE_SHA256:
        raise RuntimeError(
            "frozen copied-suite manifest file mismatch: "
            f"{actual_file_sha256} != {EXPECTED_COPIED_MANIFEST_FILE_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_canonical = manifest["copied_suite"]["canonical_manifest_sha256"]
    if actual_canonical != COPIED_CANONICAL_PRODUCT_MANIFEST_SHA256:
        raise RuntimeError(
            "frozen copied-product canonical manifest mismatch: "
            f"{actual_canonical} != {COPIED_CANONICAL_PRODUCT_MANIFEST_SHA256}"
        )
    products = {
        Path(item["filename"]).stem: item
        for item in manifest["copied_suite"]["products"]
    }
    if set(products) != set(PROFILE_STEMS):
        raise RuntimeError("copied NPZ manifest does not contain exactly 25 profiles")
    root = am_root / "Big_Atmosphere/LMT_am_npz"
    for profile in PROFILE_STEMS:
        item = products[profile]
        path = root / item["filename"]
        if sha256_path(path) != item["sha256"]:
            raise RuntimeError(f"copied NPZ SHA-256 mismatch for {path}")
        if digest_path(path, "md5") != item["md5"]:
            raise RuntimeError(f"copied NPZ MD5 mismatch for {path}")
        with np.load(path, allow_pickle=False) as archive:
            item["exact_zero_transmission_count"] = int(
                np.count_nonzero(archive["atmTtx"] == 0.0)
            )
    return products


def parse_profile(profile: str) -> tuple[str, int, str]:
    prefix, season, percentile_text = profile.split("_")
    if prefix != "LMT":
        raise RuntimeError(f"invalid profile name: {profile}")
    percentile = int(percentile_text)
    family = (
        "copied_annual_merra2_2007_2016_profile"
        if season == "annual"
        else f"copied_explicit_seasonal_{season}_merra2_2007_2016_profile"
    )
    return season, percentile, family


def load_copied_profile(am_root: Path, profile: str) -> dict[str, np.ndarray]:
    path = am_root / "Big_Atmosphere/LMT_am_npz" / f"{profile}.npz"
    with np.load(path, allow_pickle=False) as archive:
        expected_members = ["el", "atmFreq", "atmTRJ", "atmTtx", "atmTaun"]
        if archive.files != expected_members:
            raise RuntimeError(f"unexpected copied NPZ members in {path}")
        elevation = archive["el"]
        if elevation.shape != (36,) or not np.array_equal(
            elevation, np.arange(10, 82, 2, dtype=np.float64)
        ):
            raise RuntimeError(f"unexpected copied elevation coordinate in {path}")
        indices = np.asarray(
            [int(np.flatnonzero(elevation == value)[0]) for value in ELEVATIONS_DEG]
        )
        frequency = archive["atmFreq"][:, indices]
        if frequency.shape != (50001, 31):
            raise RuntimeError(f"unexpected copied frequency shape in {path}")
        if not np.all(frequency == frequency[:, [0]]) or not np.array_equal(
            frequency[:, 0], FREQUENCY_GHZ
        ):
            raise RuntimeError(f"unexpected copied frequency coordinate in {path}")
        result = {
            "tau": archive["atmTaun"][:, indices].copy(),
            "tx": archive["atmTtx"][:, indices].copy(),
            "trj": archive["atmTRJ"][:, indices].copy(),
        }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise RuntimeError(f"non-finite copied AM value in {path}")
    if not (
        np.all(result["tau"] >= 0.0)
        and np.all(result["tx"] >= 0.0)
        and np.all(result["tx"] <= 1.0)
    ):
        raise RuntimeError(f"invalid copied AM physical domain in {path}")
    if (
        np.max(np.abs(np.exp(-result["tau"]) - result["tx"]))
        > COPIED_TX_ABSOLUTE_PRINT_TOLERANCE
    ):
        raise RuntimeError(
            f"copied tau/tx inconsistency exceeds print precision: {path}"
        )
    return result


def load_legacy_sources(legacy_source_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for target, metadata in LEGACY_RAW_SOURCES.items():
        path = legacy_source_dir / metadata["filename"]
        if sha256_path(path) != metadata["sha256"]:
            raise RuntimeError(f"legacy SHA-256 mismatch for {path}")
        if digest_path(path, "md5") != metadata["md5"]:
            raise RuntimeError(f"legacy MD5 mismatch for {path}")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["el", "atmFreq", "atmTRJ", "atmTtx"]:
                raise RuntimeError(f"unexpected legacy NPZ members in {path}")
            elevation = archive["el"]
            frequency = archive["atmFreq"]
            if not np.array_equal(elevation, ELEVATIONS_DEG):
                raise RuntimeError(f"unexpected legacy elevation coordinate in {path}")
            if frequency.shape != (50001, 31):
                raise RuntimeError(f"unexpected legacy frequency shape in {path}")
            if not np.all(frequency == frequency[:, [0]]) or not np.array_equal(
                frequency[:, 0], FREQUENCY_GHZ
            ):
                raise RuntimeError(f"unexpected legacy frequency coordinate in {path}")
            result[target] = {
                "tx": archive["atmTtx"].copy(),
                "trj": archive["atmTRJ"].copy(),
            }
        if not (
            np.all(np.isfinite(result[target]["tx"]))
            and np.all(result[target]["tx"] > 0.0)
            and np.all(result[target]["tx"] <= 1.0)
            and np.all(np.isfinite(result[target]["trj"]))
        ):
            raise RuntimeError(f"invalid legacy physical domain in {path}")
    return result


def build_execution_context(
    *,
    args: argparse.Namespace,
    am_root: Path,
    legacy_source_dir: Path,
    executable: BuildIdentity,
    historical_linux: BuildIdentity,
    compiler: dict[str, Any],
    profile_inventory: dict[str, Any],
    copied_inventory: dict[str, dict[str, Any]],
    am_contracts: dict[str, dict[str, str]],
    protocol_inventory: dict[str, Any],
    execution_host: dict[str, str],
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    source_root = am_root / "am-12.2/src"
    copied_products = [
        {
            "profile": profile,
            "filename": copied_inventory[profile]["filename"],
            "size_bytes": copied_inventory[profile]["bytes"],
            "sha256": copied_inventory[profile]["sha256"],
            "md5": copied_inventory[profile]["md5"],
            "exact_zero_transmission_count": copied_inventory[profile][
                "exact_zero_transmission_count"
            ],
        }
        for profile in PROFILE_STEMS
    ]
    copied_manifest_path = PACKAGE_DIR / COPIED_MANIFEST_NAME
    legacy_inputs = []
    for target, metadata in LEGACY_RAW_SOURCES.items():
        path = legacy_source_dir / metadata["filename"]
        legacy_inputs.append(
            {
                "target": target,
                "filename": metadata["filename"],
                "size_bytes": path.stat().st_size,
                "sha256": metadata["sha256"],
                "md5": metadata["md5"],
                "tolteca_datafile_id": metadata["tolteca_datafile_id"],
            }
        )
    repair_inputs = []
    for relative, digest in sorted(
        FROZEN_REPAIR_INPUTS.items(), key=lambda item: item[0].as_posix()
    ):
        path = REPO_ROOT / relative
        repair_inputs.append(
            {
                "path_relative_to_repository": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}-execution-context-v1",
        "runner": {
            "filename": runner.name,
            "sha256": sha256_path(runner),
        },
        "execution_host": execution_host,
        "execution_parameters": {
            "jobs": args.jobs,
            "omp_threads_per_process": args.omp_threads,
            "locale": PINNED_LOCALE,
            "argv_template": [
                "<am-executable>",
                "LMT_am_inputs/<immutable-profile>.amc",
                "<fmin-binary64-17e>",
                "GHz",
                "<fmax-binary64-17e>",
                "GHz",
                "10",
                "MHz",
                "<integer-zenith-angle-deg>",
                "deg",
                "<frozen-h2o-scale-decimal>",
            ],
            "working_directory_role": "Big_Atmosphere",
            "slurm_wrapper_used": False,
            "am_cache_sharding": {
                "shard_count": args.jobs,
                "cache_id_identity": (
                    "canonical RunSpec request, AM executable/profile SHA-256, "
                    "OMP threads, shard count, and execution-context SHA-256"
                ),
                "assignment": (
                    "big-endian first 64 bits of sha256(cache_id) modulo shard_count"
                ),
                "within_process_locking": (
                    "one lock per shard around each AM subprocess"
                ),
            },
            "cache_lock": {
                "filename": CACHE_LOCK_NAME,
                "writer_mode": "nonblocking whole-cache POSIX exclusive lock",
                "reader_mode": "nonblocking whole-cache POSIX shared lock",
            },
            "in_process_observation_retention": {
                "record_type": "frozen lightweight RunObservation",
                "retained_fields": [
                    "cache_id",
                    "RunSpec",
                    "return_code",
                    "AM_version_identity",
                    "diagnostic_counts",
                    "numeric_text_sha256",
                    "normalized_output_sha256",
                ],
                "explicitly_not_retained": [
                    "ParsedOutput.samples",
                    "raw_combined_output",
                    "execution_sidecar_payload",
                ],
                "purpose": (
                    "keep final all-run digest aggregation memory-bounded without "
                    "changing its scientific identity or digest semantics"
                ),
            },
        },
        "builds": {
            "copied_linux_reference": build_identity_payload(historical_linux),
            "regeneration": {
                **build_identity_payload(executable),
                "classification": classify_regeneration_build(
                    executable, historical_linux
                ),
                "native_build_command": args.native_build_command,
                "compiler": compiler,
            },
        },
        "historical_workflow": {
            relative: {
                "path_relative_to_am_root": relative,
                "sha256": metadata["sha256"],
            }
            for relative, metadata in am_contracts.items()
            if relative.startswith("Big_Atmosphere/")
        },
        "inputs": {
            "am_source_inventory": inventory_files(
                source_root, source_files(source_root)
            ),
            "am_source_contracts": {
                relative: metadata
                for relative, metadata in am_contracts.items()
                if relative.startswith("am-12.2/src/")
            },
            "immutable_amc_profile_inventory": profile_inventory,
            "copied_scale1_npz_products": copied_products,
            "copied_suite_manifest": {
                "filename": COPIED_MANIFEST_NAME,
                "size_bytes": copied_manifest_path.stat().st_size,
                "sha256": EXPECTED_COPIED_MANIFEST_FILE_SHA256,
                "canonical_product_manifest_sha256": (
                    COPIED_CANONICAL_PRODUCT_MANIFEST_SHA256
                ),
            },
            "legacy_raw_sources": legacy_inputs,
            "missing_q95_raw_source": MISSING_Q95,
            "repair_base_inputs": repair_inputs,
            "frozen_protocol_artifact_inventory": protocol_inventory,
        },
        "protocol": {
            "study": "diagnostic_P1_documented_h2o_scale_provenance_hypothesis",
            "repair_base_sha": REPAIR_BASE_SHA,
            "repair_line_evidence_head": REPAIR_LINE_EVIDENCE_HEAD,
            "profile_stems": list(PROFILE_STEMS),
            "target_transmission_literals": EXPECTED_TARGET_TRANSMISSIONS,
            "frequency_grid_ghz": {
                "minimum": f64(0.0),
                "maximum": f64(500.0),
                "step": f64(0.01),
                "count": 50001,
            },
            "elevation_grid_deg": [int(value) for value in ELEVATIONS_DEG],
            "root_iterations": ROOT_ITERATIONS,
            "maximum_bracket_expansions": MAX_BRACKET_EXPANSIONS,
            "only_varying_parameter": (
                "Nscale troposphere h2o through immutable AMC argv %9"
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
            "citlali_application_code_modified": False,
        },
    }


def frequency_index(frequency_ghz: float) -> int:
    index = round(frequency_ghz * 100.0)
    if FREQUENCY_GHZ[index] != frequency_ghz:
        raise RuntimeError(f"frequency is not an exact 10-MHz node: {frequency_ghz}")
    return index


def parse_output(data: bytes, spec: RunSpec) -> ParsedOutput:
    label = f"{spec.stage}/{spec.profile}/{spec.target}/EL{spec.elevation_deg}"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"non-UTF-8 AM output for {label}") from error
    rows: list[list[float]] = []
    numeric_digest = hashlib.sha256()
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        tokens = line.split()
        if len(tokens) != 5 or not all(
            FLOAT_TOKEN.fullmatch(token) for token in tokens
        ):
            continue
        rows.append([float(token) for token in tokens])
        numeric_digest.update(raw_line.encode("utf-8"))
    samples = np.asarray(rows, dtype=np.float64)
    if samples.shape != (spec.expected_rows, 5):
        raise RuntimeError(
            f"unexpected AM grid for {label}: {samples.shape} != "
            f"({spec.expected_rows}, 5)"
        )
    expected_frequency = (
        np.arange(
            spec.f_min_centi_ghz,
            spec.f_max_centi_ghz + 1,
            dtype=np.float64,
        )
        / 100.0
    )
    if not np.array_equal(samples[:, 0], expected_frequency):
        raise RuntimeError(f"unexpected frequency grid for {label}")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError(f"non-finite AM output for {label}")
    if not (
        np.all(samples[:, 1] >= 0.0)
        and np.all(samples[:, 2] >= 0.0)
        and np.all(samples[:, 2] <= 1.0)
    ):
        raise RuntimeError(f"invalid tau/transmission domain for {label}")
    versions = VERSION_LINE.findall(text)
    if len(versions) != 1 or not versions[0].startswith("am version 12.2"):
        raise RuntimeError(f"unexpected AM version identity for {label}: {versions!r}")
    warnings = [int(value) for value in UNRESOLVED_WARNING.findall(text)]
    if len(warnings) > 1:
        raise RuntimeError(f"multiple unresolved-line warnings for {label}")
    warning_headers = WARNING_HEADER.findall(text)
    allowed_header = "Encountered in-band lines narrower than the frequency"
    unknown_warning_headers = [
        message
        for message in warning_headers
        if message != allowed_header
        and re.fullmatch(r"Column included \d+ unresolved lines\.", message) is None
    ]
    if unknown_warning_headers:
        raise RuntimeError(
            f"unknown AM warning class for {label}: {unknown_warning_headers!r}"
        )
    if warning_headers and not warnings:
        raise RuntimeError(f"incomplete canonical unresolved-line warning for {label}")
    error_line_count = len(ERROR_LINE.findall(text))
    if error_line_count:
        raise RuntimeError(f"AM error diagnostic present for {label}")
    return ParsedOutput(
        samples=samples,
        version_identity=versions[0],
        warning_count=warnings[0] if warnings else None,
        numeric_text_sha256=numeric_digest.hexdigest(),
        normalized_output_sha256=sha256_bytes(normalize_combined_output(data)),
        unresolved_column_warning_line_count=sum(
            re.fullmatch(r"Column included \d+ unresolved lines\.", message) is not None
            for message in warning_headers
        ),
        unresolved_summary_warning_line_count=sum(
            message == allowed_header for message in warning_headers
        ),
        other_warning_line_count=len(unknown_warning_headers),
        error_line_count=error_line_count,
    )


def validate_return_contract(
    return_code: int, parsed: ParsedOutput, label: str
) -> None:
    if return_code == 0:
        return
    if return_code == 1 and parsed.warning_count is not None:
        return
    raise RuntimeError(
        f"rejected AM status for {label}: return_code={return_code}, "
        f"warning_count={parsed.warning_count}"
    )


class Runner:
    def __init__(
        self,
        *,
        executable: BuildIdentity,
        am_root: Path,
        cache_dir: Path,
        omp_threads: int,
        cache_shard_count: int,
        execution_host: dict[str, str],
        execution_context_sha256: str,
        execute: bool,
    ) -> None:
        self.executable = executable
        self.am_root = am_root
        self.big_atmosphere_root = am_root / "Big_Atmosphere"
        self.profile_root = self.big_atmosphere_root / "LMT_am_inputs"
        self.cache_dir = cache_dir
        self.am_cache_root = cache_dir / "am_spectral_cache"
        self.omp_threads = omp_threads
        self.cache_shard_count = cache_shard_count
        self.execution_host = execution_host
        self.execution_context_sha256 = execution_context_sha256
        self.cache_shard_locks = [threading.Lock() for _ in range(cache_shard_count)]
        self.observed_runs_lock = threading.Lock()
        self._observed_runs: dict[str, RunObservation] = {}
        self.execute = execute
        for index in range(cache_shard_count):
            shard = self.am_cache_root / f"shard_{index:02d}"
            if execute:
                shard.mkdir(parents=True, exist_ok=True)
            elif not shard.is_dir():
                raise RuntimeError(
                    f"cache-only operation requires AM cache shard: {shard}"
                )

    def argv(self, spec: RunSpec) -> list[str]:
        return [
            self.executable.resolved_path,
            f"LMT_am_inputs/{spec.profile}.amc",
            f64(spec.f_min_centi_ghz / 100.0),
            "GHz",
            f64(spec.f_max_centi_ghz / 100.0),
            "GHz",
            "10",
            "MHz",
            str(spec.zenith_angle_deg),
            "deg",
            spec.scale_decimal,
        ]

    def cache_id(self, spec: RunSpec) -> str:
        profile_path = self.profile_root / f"{spec.profile}.amc"
        identity = {
            "request": spec.request_payload(),
            "am_executable_sha256": self.executable.sha256,
            "profile_sha256": sha256_path(profile_path),
            "omp_threads": self.omp_threads,
            "cache_shard_count": self.cache_shard_count,
            "execution_context_sha256": self.execution_context_sha256,
        }
        digest = sha256_bytes(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )[:24]
        semantic = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            f"{spec.stage}_{spec.profile}_{spec.target}_za{spec.zenith_angle_deg:02d}",
        ).strip("_")
        return f"{semantic}_{digest}"

    def cache_shard_index(self, cache_id: str) -> int:
        digest = hashlib.sha256(cache_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.cache_shard_count

    def am_cache_dir(self, cache_id: str) -> Path:
        return self.am_cache_root / f"shard_{self.cache_shard_index(cache_id):02d}"

    def raw_path(self, cache_id: str) -> Path:
        return self.cache_dir / "raw_outputs" / f"{cache_id}.txt"

    def sidecar_path(self, cache_id: str) -> Path:
        return self.cache_dir / "execution_records" / f"{cache_id}.run.json"

    def _sidecar_core(
        self, spec: RunSpec, cache_id: str, raw: bytes, parsed: ParsedOutput
    ) -> dict[str, Any]:
        profile_path = self.profile_root / f"{spec.profile}.amc"
        raw_path = self.raw_path(cache_id)
        shard_index = self.cache_shard_index(cache_id)
        am_cache_dir = self.am_cache_dir(cache_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "cache_id": cache_id,
            "request": spec.request_payload(),
            "argv": self.argv(spec),
            "working_directory_role": "Big_Atmosphere",
            "profile_path_relative_to_working_directory": (
                f"LMT_am_inputs/{spec.profile}.amc"
            ),
            "profile_sha256": sha256_path(profile_path),
            "am_executable_sha256": self.executable.sha256,
            "omp_threads": self.omp_threads,
            "locale": PINNED_LOCALE,
            "execution_host": self.execution_host,
            "execution_context_sha256": self.execution_context_sha256,
            "am_cache_shard_index": shard_index,
            "am_cache_shard_count": self.cache_shard_count,
            "am_cache_path_relative_to_cache": am_cache_dir.relative_to(
                self.cache_dir
            ).as_posix(),
            "combined_output_path_relative_to_cache": raw_path.relative_to(
                self.cache_dir
            ).as_posix(),
            "combined_output_sha256": sha256_bytes(raw),
            "numeric_text_sha256": parsed.numeric_text_sha256,
            "normalized_output_sha256": parsed.normalized_output_sha256,
            "numeric_row_count": int(parsed.samples.shape[0]),
            "unresolved_line_warning_count": parsed.warning_count,
            "unresolved_column_warning_line_count": (
                parsed.unresolved_column_warning_line_count
            ),
            "unresolved_summary_warning_line_count": (
                parsed.unresolved_summary_warning_line_count
            ),
            "other_warning_line_count": parsed.other_warning_line_count,
            "error_line_count": parsed.error_line_count,
            "am_version_identity": parsed.version_identity,
        }

    def _load_cached_result(
        self,
        spec: RunSpec,
        cache_id: str,
        raw_path: Path,
        record_path: Path,
    ) -> RunResult:
        if not raw_path.is_file() or not record_path.is_file():
            raise RuntimeError(f"missing cached AM evidence for {cache_id}")
        raw = raw_path.read_bytes()
        sidecar_bytes = record_path.read_bytes()
        sidecar = json.loads(sidecar_bytes)
        if sidecar_bytes != json_bytes(sidecar):
            raise RuntimeError(f"noncanonical cached sidecar for {cache_id}")
        parsed = parse_output(raw, spec)
        expected = self._sidecar_core(spec, cache_id, raw, parsed)
        for key, value in expected.items():
            if sidecar.get(key) != value:
                raise RuntimeError(
                    f"cached sidecar mismatch for {cache_id}/{key}: "
                    f"{sidecar.get(key)!r} != {value!r}"
                )
        return_code = int(sidecar.get("return_code"))
        validate_return_contract(return_code, parsed, cache_id)
        return RunResult(
            spec, parsed, return_code, sha256_bytes(raw), sidecar, cache_id
        )

    def _preserve_failed_attempt(
        self,
        *,
        spec: RunSpec,
        cache_id: str,
        raw: bytes,
        return_code: int,
        error: Exception,
    ) -> None:
        digest = sha256_bytes(raw)
        stem = f"{cache_id}_{digest[:16]}"
        raw_path = self.cache_dir / "failed_attempts" / f"{stem}.txt"
        record_path = self.cache_dir / "failed_attempts" / f"{stem}.failure.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected_am_attempt_not_used_as_evidence",
            "cache_id": cache_id,
            "request": spec.request_payload(),
            "argv": self.argv(spec),
            "working_directory_role": "Big_Atmosphere",
            "am_executable_sha256": self.executable.sha256,
            "omp_threads": self.omp_threads,
            "locale": PINNED_LOCALE,
            "execution_host": self.execution_host,
            "execution_context_sha256": self.execution_context_sha256,
            "am_cache_shard_index": self.cache_shard_index(cache_id),
            "am_cache_shard_count": self.cache_shard_count,
            "return_code": return_code,
            "combined_output_path_relative_to_cache": raw_path.relative_to(
                self.cache_dir
            ).as_posix(),
            "combined_output_sha256": digest,
            "rejection": f"{type(error).__name__}: {error}",
        }
        atomic_write(raw_path, raw)
        atomic_write(record_path, json_bytes(payload))

    @staticmethod
    def _observation(result: RunResult) -> RunObservation:
        parsed = result.parsed
        return RunObservation(
            cache_id=result.cache_id,
            spec=result.spec,
            return_code=result.return_code,
            am_version_identity=parsed.version_identity,
            unresolved_line_warning_count=parsed.warning_count,
            unresolved_column_warning_line_count=(
                parsed.unresolved_column_warning_line_count
            ),
            unresolved_summary_warning_line_count=(
                parsed.unresolved_summary_warning_line_count
            ),
            other_warning_line_count=parsed.other_warning_line_count,
            error_line_count=parsed.error_line_count,
            numeric_text_sha256=parsed.numeric_text_sha256,
            normalized_output_sha256=parsed.normalized_output_sha256,
        )

    def _record_result(self, result: RunResult) -> RunResult:
        observation = self._observation(result)
        with self.observed_runs_lock:
            previous = self._observed_runs.get(result.cache_id)
            if previous is not None and previous != observation:
                raise RuntimeError(
                    f"inconsistent repeated AM result for {result.cache_id}"
                )
            self._observed_runs[result.cache_id] = observation
        return result

    def observed_runs(self) -> list[RunObservation]:
        with self.observed_runs_lock:
            return [self._observed_runs[key] for key in sorted(self._observed_runs)]

    def run_or_load(self, spec: RunSpec) -> RunResult:
        cache_id = self.cache_id(spec)
        raw_path = self.raw_path(cache_id)
        record_path = self.sidecar_path(cache_id)
        if raw_path.is_file() != record_path.is_file():
            raise RuntimeError(f"incomplete cached AM evidence for {cache_id}")
        if raw_path.is_file() and record_path.is_file():
            return self._record_result(
                self._load_cached_result(spec, cache_id, raw_path, record_path)
            )
        if self.execute:
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = str(self.omp_threads)
            shard_index = self.cache_shard_index(cache_id)
            environment["AM_CACHE_PATH"] = str(self.am_cache_dir(cache_id))
            environment.update(PINNED_LOCALE)
            with self.cache_shard_locks[shard_index]:
                completed = subprocess.run(
                    self.argv(spec),
                    cwd=self.big_atmosphere_root,
                    env=environment,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            raw = completed.stdout
            try:
                parsed = parse_output(raw, spec)
                validate_return_contract(completed.returncode, parsed, cache_id)
            except Exception as error:
                self._preserve_failed_attempt(
                    spec=spec,
                    cache_id=cache_id,
                    raw=raw,
                    return_code=completed.returncode,
                    error=error,
                )
                raise
            sidecar = self._sidecar_core(spec, cache_id, raw, parsed)
            sidecar["return_code"] = completed.returncode
            atomic_write(raw_path, raw)
            atomic_write(record_path, json_bytes(sidecar))
            return self._record_result(
                RunResult(
                    spec,
                    parsed,
                    completed.returncode,
                    sha256_bytes(raw),
                    sidecar,
                    cache_id,
                )
            )
        return self._record_result(
            self._load_cached_result(spec, cache_id, raw_path, record_path)
        )

    def run_many(self, specs: Iterable[RunSpec], jobs: int) -> list[RunResult]:
        ordered = list(specs)
        if jobs == 1:
            return [self.run_or_load(spec) for spec in ordered]
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            return list(executor.map(self.run_or_load, ordered))


def anchor_spec(profile: str, target: str, scale_decimal: str) -> RunSpec:
    return RunSpec(
        stage="anchor_225ghz_el80",
        profile=profile,
        target=target,
        f_min_centi_ghz=22499,
        f_max_centi_ghz=22501,
        zenith_angle_deg=10,
        scale_decimal=scale_decimal,
    )


def full_grid_spec(
    stage: str,
    profile: str,
    target: str,
    zenith_angle_deg: int,
    scale_decimal: str,
) -> RunSpec:
    return RunSpec(
        stage=stage,
        profile=profile,
        target=target,
        f_min_centi_ghz=0,
        f_max_centi_ghz=50000,
        zenith_angle_deg=zenith_angle_deg,
        scale_decimal=scale_decimal,
    )


def anchor_values(run: RunResult) -> tuple[float, float, float]:
    indices = np.flatnonzero(run.parsed.samples[:, 0] == REFERENCE_FREQUENCY_GHZ)
    if indices.size != 1:
        raise RuntimeError(f"225 GHz is not exactly one node in {run.cache_id}")
    row = run.parsed.samples[int(indices[0])]
    return float(row[1]), float(row[2]), float(row[3])


def copied_anchor(am_root: Path, profile: str) -> tuple[float, float]:
    path = am_root / "Big_Atmosphere/LMT_am_npz" / f"{profile}.npz"
    with np.load(path, allow_pickle=False) as archive:
        elevation_indices = np.flatnonzero(archive["el"] == 80.0)
        if elevation_indices.size != 1:
            raise RuntimeError(f"EL80 is not exactly one copied node in {path}")
        column = int(elevation_indices[0])
        frequency_indices = np.flatnonzero(
            archive["atmFreq"][:, column] == REFERENCE_FREQUENCY_GHZ
        )
        if frequency_indices.size != 1:
            raise RuntimeError(f"225 GHz is not exactly one copied node in {path}")
        row = int(frequency_indices[0])
        tau = float(archive["atmTaun"][row, column])
        transmission = float(archive["atmTtx"][row, column])
    return tau, transmission


def solve_scale_hypothesis(
    *,
    runner: Runner,
    profile: str,
    target: str,
    scale0: RunResult,
    scale1: RunResult,
    copied_scale1_tau: float,
    copied_scale1_transmission: float,
) -> ScaleSolution:
    """Solve one nonnegative parsed-T225 plateau using no other AM degree."""

    target_transmission = float(EXPECTED_TARGET_TRANSMISSIONS[target])
    target_tau = -math.log(target_transmission)
    tau0, tx0, _ = anchor_values(scale0)
    tau1_direct, _, _ = anchor_values(scale1)
    denominator = copied_scale1_tau - tau0
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(f"nonpositive copied H2O tau increment for {profile}")
    affine_initial_scale = max(0.0, (target_tau - tau0) / denominator)

    evaluations: dict[str, tuple[RunResult, str]] = {
        f64(0.0): (scale0, "shared_direct_scale0"),
        f64(1.0): (scale1, "shared_direct_scale1_linearity_check"),
    }

    def evaluate(scale: float, role: str) -> RunResult:
        if not math.isfinite(scale) or scale < 0.0:
            raise RuntimeError(
                f"invalid inferred H2O scale for {profile}/{target}: {scale}"
            )
        decimal = f64(scale)
        if decimal not in evaluations:
            run = runner.run_or_load(anchor_spec(profile, target, decimal))
            evaluations[decimal] = (run, role)
        return evaluations[decimal][0]

    initial = evaluate(affine_initial_scale, "affine_tau_initial_candidate")
    _, initial_tx, _ = anchor_values(initial)
    exact = initial_tx == target_transmission
    inside = initial if exact else None
    lower_outside: RunResult | None = None

    if not exact and tx0 >= target_transmission:
        low = scale0
        high_scale = max(1.0, affine_initial_scale * 1.25 + 0.1)
        high = evaluate(high_scale, "entry_bracket_high")
        for _ in range(MAX_BRACKET_EXPANSIONS):
            if anchor_values(high)[1] <= target_transmission:
                break
            low = high
            high_scale = high_scale * 2.0 + 0.1
            high = evaluate(high_scale, "entry_bracket_expand")
        else:
            raise RuntimeError(f"could not bracket T225 for {profile}/{target}")

        for _ in range(ROOT_ITERATIONS):
            low_scale = float(low.spec.scale_decimal)
            high_scale = float(high.spec.scale_decimal)
            midpoint = (low_scale + high_scale) / 2.0
            if f64(midpoint) in {low.spec.scale_decimal, high.spec.scale_decimal}:
                break
            run = evaluate(midpoint, "entry_boundary_bisection")
            if anchor_values(run)[1] > target_transmission:
                low = run
            else:
                high = run
        lower_outside = low
        if anchor_values(high)[1] == target_transmission:
            inside = high
            exact = True

    plateau_lower_outside: RunResult | None = None
    plateau_lower_inside: RunResult | None = None
    plateau_upper_inside: RunResult | None = None
    plateau_upper_outside: RunResult | None = None

    if exact and inside is not None:
        inside_scale = float(inside.spec.scale_decimal)
        if lower_outside is None:
            delta = max(1.0e-8, max(1.0, inside_scale) * 1.0e-4)
            for _ in range(MAX_BRACKET_EXPANSIONS):
                candidate_scale = max(0.0, inside_scale - delta)
                candidate = evaluate(candidate_scale, "lower_plateau_bracket")
                candidate_tx = anchor_values(candidate)[1]
                if candidate_tx > target_transmission or candidate_scale == 0.0:
                    lower_outside = candidate
                    break
                if candidate_tx < target_transmission:
                    raise RuntimeError(
                        f"nonmonotone lower T225 bracket for {profile}/{target}"
                    )
                delta *= 2.0
            else:
                raise RuntimeError(
                    f"could not establish lower plateau bracket for {profile}/{target}"
                )

        if anchor_values(lower_outside)[1] == target_transmission:
            if float(lower_outside.spec.scale_decimal) != 0.0:
                raise RuntimeError(
                    f"invalid lower plateau boundary for {profile}/{target}"
                )
            plateau_lower_inside = lower_outside
        else:
            low = lower_outside
            high = inside
            for _ in range(ROOT_ITERATIONS):
                midpoint = (
                    float(low.spec.scale_decimal) + float(high.spec.scale_decimal)
                ) / 2.0
                if f64(midpoint) in {low.spec.scale_decimal, high.spec.scale_decimal}:
                    break
                run = evaluate(midpoint, "lower_plateau_bisection")
                if anchor_values(run)[1] > target_transmission:
                    low = run
                elif anchor_values(run)[1] == target_transmission:
                    high = run
                else:
                    raise RuntimeError(
                        f"nonmonotone lower plateau search for {profile}/{target}"
                    )
            plateau_lower_outside = low
            plateau_lower_inside = high

        delta = max(1.0e-8, max(1.0, inside_scale) * 1.0e-4)
        high = inside
        for _ in range(MAX_BRACKET_EXPANSIONS):
            candidate = evaluate(inside_scale + delta, "upper_plateau_bracket")
            candidate_tx = anchor_values(candidate)[1]
            if candidate_tx < target_transmission:
                plateau_upper_outside = candidate
                break
            if candidate_tx > target_transmission:
                raise RuntimeError(
                    f"nonmonotone upper T225 bracket for {profile}/{target}"
                )
            high = candidate
            delta *= 2.0
        else:
            raise RuntimeError(
                f"could not establish upper plateau bracket for {profile}/{target}"
            )
        low = high
        high = plateau_upper_outside
        for _ in range(ROOT_ITERATIONS):
            midpoint = (
                float(low.spec.scale_decimal) + float(high.spec.scale_decimal)
            ) / 2.0
            if f64(midpoint) in {low.spec.scale_decimal, high.spec.scale_decimal}:
                break
            run = evaluate(midpoint, "upper_plateau_bisection")
            if anchor_values(run)[1] == target_transmission:
                low = run
            elif anchor_values(run)[1] < target_transmission:
                high = run
            else:
                raise RuntimeError(
                    f"nonmonotone upper plateau search for {profile}/{target}"
                )
        plateau_upper_inside = low
        plateau_upper_outside = high

        canonical_value = (
            float(plateau_lower_inside.spec.scale_decimal)
            + float(plateau_upper_inside.spec.scale_decimal)
        ) / 2.0
        fitted = evaluate(canonical_value, "canonical_plateau_midpoint")
        if anchor_values(fitted)[1] != target_transmission:
            fitted = plateau_lower_inside
        method = (
            "direct_parsed_tx_plateau_midpoint_seeded_by_"
            "direct_scale0_plus_copied_scale1_tau"
        )
    else:
        fitted, _ = min(
            evaluations.values(),
            key=lambda item: (
                abs(anchor_values(item[0])[1] - target_transmission),
                abs(anchor_values(item[0])[0] - target_tau),
                float(item[0].spec.scale_decimal),
            ),
        )
        exact = False
        method = (
            "nonnegative_closest_sample_no_exact_parsed_tx_plateau_"
            "no_additional_atmospheric_degree"
        )

    ordered_evaluations = sorted(
        (
            float(decimal),
            anchor_values(run)[0],
            anchor_values(run)[1],
            decimal,
        )
        for decimal, (run, _) in evaluations.items()
    )
    for left, right in zip(ordered_evaluations, ordered_evaluations[1:]):
        if right[1] + 1.0e-12 < left[1] or right[2] > left[2] + 1.0e-15:
            raise RuntimeError(
                f"nonmonotone direct anchor evaluations for {profile}/{target}"
            )

    trace_rows = []
    for decimal, (run, role) in evaluations.items():
        tau, transmission, trj = anchor_values(run)
        trace_rows.append(
            {
                "evaluation_index": len(trace_rows),
                "role": role,
                "scale_decimal": decimal,
                "tau_los": f64(tau),
                "transmission": f64(transmission),
                "trj_k": f64(trj),
                "return_code": run.return_code,
                "unresolved_line_warning_count": run.parsed.warning_count,
                "numeric_text_sha256": run.parsed.numeric_text_sha256,
                "normalized_output_sha256": (run.parsed.normalized_output_sha256),
            }
        )
    trace = {
        "schema_version": SCHEMA_VERSION,
        "execution_context_sha256": runner.execution_context_sha256,
        "trace_kind": "nonnegative_single_parameter_parsed_t225_plateau",
        "target": target,
        "target_transmission_literal": EXPECTED_TARGET_TRANSMISSIONS[target],
        "profile": profile,
        "root_iterations": ROOT_ITERATIONS,
        "maximum_bracket_expansions": MAX_BRACKET_EXPANSIONS,
        "canonical_policy": (
            "midpoint of the innermost binary64 scales observed inside the "
            "exact parsed-transmission plateau after fixed-count bisection"
        ),
        "evaluations": trace_rows,
    }
    trace_bytes = json_bytes(trace)
    trace_relative = Path("scale_traces") / f"{target}_{profile}.json"
    trace_path = runner.cache_dir / trace_relative
    if runner.execute:
        atomic_write(trace_path, trace_bytes)
    elif not trace_path.is_file() or trace_path.read_bytes() != trace_bytes:
        raise RuntimeError(f"cached scale trace differs: {trace_path}")

    return ScaleSolution(
        target=target,
        profile=profile,
        target_transmission=target_transmission,
        target_tau=target_tau,
        scale_decimal=fitted.spec.scale_decimal,
        scale_value=float(fitted.spec.scale_decimal),
        exact_parsed_transmission_match=exact,
        method=method,
        scale0=scale0,
        scale1=scale1,
        fitted=fitted,
        copied_scale1_tau=copied_scale1_tau,
        copied_scale1_transmission=copied_scale1_transmission,
        affine_initial_scale=affine_initial_scale,
        plateau_lower_outside_scale=(
            None
            if plateau_lower_outside is None
            else float(plateau_lower_outside.spec.scale_decimal)
        ),
        plateau_lower_inside_scale=(
            None
            if plateau_lower_inside is None
            else float(plateau_lower_inside.spec.scale_decimal)
        ),
        plateau_upper_inside_scale=(
            None
            if plateau_upper_inside is None
            else float(plateau_upper_inside.spec.scale_decimal)
        ),
        plateau_upper_outside_scale=(
            None
            if plateau_upper_outside is None
            else float(plateau_upper_outside.spec.scale_decimal)
        ),
        trace_relative_path=trace_relative.as_posix(),
        trace_sha256=sha256_bytes(trace_bytes),
        trace_evaluation_count=len(trace_rows),
    )


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="linear"))


def residual_summary(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    signed = np.asarray(predicted - truth, dtype=np.float64)
    absolute = np.abs(signed)
    return {
        "minimum_signed": float(np.min(signed)),
        "maximum_signed": float(np.max(signed)),
        "maximum_absolute": float(np.max(absolute)),
        "p95_absolute": percentile(absolute, 0.95),
        "median_absolute": percentile(absolute, 0.5),
        "rms": float(np.sqrt(np.mean(np.square(signed)))),
    }


def correction_summary(
    predicted_los_tau: np.ndarray, truth_los_tau: np.ndarray
) -> tuple[dict[str, float], int]:
    delta = np.asarray(predicted_los_tau - truth_los_tau, dtype=np.float64)
    overflow = delta > math.log(sys.float_info.max)
    signed = np.empty_like(delta)
    signed[~overflow] = np.expm1(delta[~overflow])
    signed[overflow] = np.inf
    absolute = np.abs(signed)
    return (
        {
            "minimum_signed": float(np.min(signed)),
            "maximum_signed": float(np.max(signed)),
            "maximum_absolute": float(np.max(absolute)),
            "p95_absolute": percentile(absolute, 0.95),
            "median_absolute": percentile(absolute, 0.5),
            "rms": float(np.sqrt(np.mean(np.square(signed)))),
        },
        int(np.count_nonzero(overflow)),
    )


def empty_metric_row() -> dict[str, str]:
    return {
        "target_model": "",
        "target_registry_family": "",
        "source_profile": "",
        "source_profile_season": "",
        "source_profile_percentile": "",
        "source_profile_family": "",
        "h2o_scale_decimal": "",
        "anchor_exact_parsed_tx_match": "",
        "evaluation_lane": "",
        "comparison_quantity": "",
        "comparison_scope": "",
        "band": "",
        "frequency_ghz": "",
        "truth_kind": "",
        "truth_artifact": "",
        "truth_sha256": "",
        "sample_count": "",
        "transmission_or_ratio_min_signed_residual": "",
        "transmission_or_ratio_max_signed_residual": "",
        "transmission_or_ratio_max_abs_residual": "",
        "transmission_or_ratio_p95_abs_residual": "",
        "transmission_or_ratio_median_abs_residual": "",
        "transmission_or_ratio_rms_residual": "",
        "trj_min_signed_residual_k": "",
        "trj_max_signed_residual_k": "",
        "trj_max_abs_residual_k": "",
        "trj_p95_abs_residual_k": "",
        "trj_median_abs_residual_k": "",
        "trj_rms_residual_k": "",
        "fractional_correction_min_signed_error": "",
        "fractional_correction_max_signed_error": "",
        "fractional_correction_max_abs_error": "",
        "fractional_correction_p95_abs_error": "",
        "fractional_correction_median_abs_error": "",
        "fractional_correction_rms_error": "",
        "fractional_correction_overflow_count": "",
        "passes_provisional_1pct_numerical_diagnostic": "",
        "transmission_rms_rank": "",
        "trj_rms_rank": "",
        "ancillary_screening_transmission_rank1": "false",
        "direct_full_grid_evaluated": "false",
        "am_run_count": "0",
        "maximum_return_code": "",
        "warning_status_run_count": "0",
        "unresolved_line_warning_count_sum": "0",
        "interpretation": "post_hoc_provenance_hypothesis_not_custody_proof_or_operator_authorization",
    }


def summarize_metric_row(
    *,
    target: str,
    profile: str,
    solution: ScaleSolution,
    lane: str,
    comparison_quantity: str,
    scope: str,
    band: str,
    frequency: float | None,
    truth_kind: str,
    truth_artifact: str,
    truth_sha256: str,
    predicted_value: np.ndarray,
    truth_value: np.ndarray,
    predicted_tau: np.ndarray,
    truth_tau: np.ndarray,
    predicted_trj: np.ndarray | None,
    truth_trj: np.ndarray | None,
    runs: list[RunResult] | None = None,
) -> dict[str, str]:
    season, percentile_value, family = parse_profile(profile)
    value_metrics = residual_summary(predicted_value, truth_value)
    correction_metrics, overflow_count = correction_summary(predicted_tau, truth_tau)
    row = empty_metric_row()
    row.update(
        {
            "target_model": target,
            "target_registry_family": "legacy_generic_unprefixed_am_q_registry",
            "source_profile": profile,
            "source_profile_season": season,
            "source_profile_percentile": str(percentile_value),
            "source_profile_family": family,
            "h2o_scale_decimal": solution.scale_decimal,
            "anchor_exact_parsed_tx_match": str(
                solution.exact_parsed_transmission_match
            ).lower(),
            "evaluation_lane": lane,
            "comparison_quantity": comparison_quantity,
            "comparison_scope": scope,
            "band": band,
            "frequency_ghz": "" if frequency is None else f64(frequency),
            "truth_kind": truth_kind,
            "truth_artifact": truth_artifact,
            "truth_sha256": truth_sha256,
            "sample_count": str(predicted_value.size),
            "transmission_or_ratio_min_signed_residual": f64(
                value_metrics["minimum_signed"]
            ),
            "transmission_or_ratio_max_signed_residual": f64(
                value_metrics["maximum_signed"]
            ),
            "transmission_or_ratio_max_abs_residual": f64(
                value_metrics["maximum_absolute"]
            ),
            "transmission_or_ratio_p95_abs_residual": f64(
                value_metrics["p95_absolute"]
            ),
            "transmission_or_ratio_median_abs_residual": f64(
                value_metrics["median_absolute"]
            ),
            "transmission_or_ratio_rms_residual": f64(value_metrics["rms"]),
            "fractional_correction_min_signed_error": f64(
                correction_metrics["minimum_signed"]
            ),
            "fractional_correction_max_signed_error": f64(
                correction_metrics["maximum_signed"]
            ),
            "fractional_correction_max_abs_error": f64(
                correction_metrics["maximum_absolute"]
            ),
            "fractional_correction_p95_abs_error": f64(
                correction_metrics["p95_absolute"]
            ),
            "fractional_correction_median_abs_error": f64(
                correction_metrics["median_absolute"]
            ),
            "fractional_correction_rms_error": f64(correction_metrics["rms"]),
            "fractional_correction_overflow_count": str(overflow_count),
            "passes_provisional_1pct_numerical_diagnostic": str(
                bool(correction_metrics["maximum_absolute"] <= 0.01)
            ).lower(),
        }
    )
    if predicted_trj is not None and truth_trj is not None:
        trj_metrics = residual_summary(predicted_trj, truth_trj)
        row.update(
            {
                "trj_min_signed_residual_k": f64(trj_metrics["minimum_signed"]),
                "trj_max_signed_residual_k": f64(trj_metrics["maximum_signed"]),
                "trj_max_abs_residual_k": f64(trj_metrics["maximum_absolute"]),
                "trj_p95_abs_residual_k": f64(trj_metrics["p95_absolute"]),
                "trj_median_abs_residual_k": f64(trj_metrics["median_absolute"]),
                "trj_rms_residual_k": f64(trj_metrics["rms"]),
            }
        )
    if runs:
        row.update(
            {
                "am_run_count": str(len(runs)),
                "maximum_return_code": str(max(run.return_code for run in runs)),
                "warning_status_run_count": str(
                    sum(run.return_code == 1 for run in runs)
                ),
                "unresolved_line_warning_count_sum": str(
                    sum(run.parsed.warning_count or 0 for run in runs)
                ),
            }
        )
    return row


def run_full_profile(
    *,
    runner: Runner,
    jobs: int,
    stage: str,
    profile: str,
    target: str,
    scale_decimal: str,
) -> tuple[dict[str, np.ndarray], list[RunResult]]:
    specs = [
        full_grid_spec(
            stage,
            profile,
            target,
            zenith_angle,
            scale_decimal,
        )
        for zenith_angle in ZENITH_ANGLES_DEG
    ]
    runs = runner.run_many(specs, jobs)
    arrays = {
        "tau": np.stack([run.parsed.samples[:, 1] for run in runs], axis=1),
        "tx": np.stack([run.parsed.samples[:, 2] for run in runs], axis=1),
        "trj": np.stack([run.parsed.samples[:, 3] for run in runs], axis=1),
    }
    for name, value in arrays.items():
        if value.shape != (50001, 31) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid assembled {name} grid for {profile}/{target}")
    return arrays, runs


def build_scale_solutions(
    *,
    runner: Runner,
    am_root: Path,
    jobs: int,
) -> dict[tuple[str, str], ScaleSolution]:
    shared_specs = []
    for profile in PROFILE_STEMS:
        shared_specs.extend(
            [
                anchor_spec(profile, "shared", f64(0.0)),
                anchor_spec(profile, "shared", f64(1.0)),
            ]
        )
    shared_runs = runner.run_many(shared_specs, jobs)
    shared = {(run.spec.profile, run.spec.scale_decimal): run for run in shared_runs}
    copied = {profile: copied_anchor(am_root, profile) for profile in PROFILE_STEMS}

    requests = [(target, profile) for target in TARGETS for profile in PROFILE_STEMS]

    def solve(request: tuple[str, str]) -> ScaleSolution:
        target, profile = request
        copied_tau, copied_tx = copied[profile]
        return solve_scale_hypothesis(
            runner=runner,
            profile=profile,
            target=target,
            scale0=shared[profile, f64(0.0)],
            scale1=shared[profile, f64(1.0)],
            copied_scale1_tau=copied_tau,
            copied_scale1_transmission=copied_tx,
        )

    if jobs == 1:
        ordered_solutions = [solve(request) for request in requests]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            ordered_solutions = list(executor.map(solve, requests))
    return {
        (solution.target, solution.profile): solution for solution in ordered_solutions
    }


def build_scale_rows(
    solutions: dict[tuple[str, str], ScaleSolution],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for target in TARGETS:
        for profile in PROFILE_STEMS:
            solution = solutions[target, profile]
            season, percentile_value, family = parse_profile(profile)
            tau0, tx0, _ = anchor_values(solution.scale0)
            tau1_direct, tx1_direct, _ = anchor_values(solution.scale1)
            fitted_tau, fitted_tx, _ = anchor_values(solution.fitted)
            affine_copied_tau = tau0 + solution.scale_value * (
                solution.copied_scale1_tau - tau0
            )
            affine_direct_tau = tau0 + solution.scale_value * (tau1_direct - tau0)
            anchor_correction = math.expm1(fitted_tau - solution.target_tau)
            rows.append(
                {
                    "target_model": target,
                    "target_registry_family": "legacy_generic_unprefixed_am_q_registry",
                    "target_t225_source_literal": EXPECTED_TARGET_TRANSMISSIONS[target],
                    "target_los_tau_from_literal": f64(solution.target_tau),
                    "source_profile": profile,
                    "source_profile_season": season,
                    "source_profile_percentile": str(percentile_value),
                    "source_profile_family": family,
                    "fitted_h2o_scale_decimal": solution.scale_decimal,
                    "fit_method": solution.method,
                    "direct_scale0_tau_los": f64(tau0),
                    "direct_scale0_transmission": f64(tx0),
                    "copied_scale1_tau_los": f64(solution.copied_scale1_tau),
                    "copied_scale1_transmission": f64(
                        solution.copied_scale1_transmission
                    ),
                    "direct_scale1_tau_los": f64(tau1_direct),
                    "direct_scale1_transmission": f64(tx1_direct),
                    "direct_minus_copied_scale1_tau": f64(
                        tau1_direct - solution.copied_scale1_tau
                    ),
                    "direct_minus_copied_scale1_transmission": f64(
                        tx1_direct - solution.copied_scale1_transmission
                    ),
                    "affine_initial_scale_from_direct0_copied1_tau": f64(
                        solution.affine_initial_scale
                    ),
                    "plateau_lower_outside_scale": optional_f64(
                        solution.plateau_lower_outside_scale
                    ),
                    "plateau_lower_inside_scale": optional_f64(
                        solution.plateau_lower_inside_scale
                    ),
                    "plateau_upper_inside_scale": optional_f64(
                        solution.plateau_upper_inside_scale
                    ),
                    "plateau_upper_outside_scale": optional_f64(
                        solution.plateau_upper_outside_scale
                    ),
                    "direct_fitted_tau_los": f64(fitted_tau),
                    "direct_fitted_transmission": f64(fitted_tx),
                    "exact_parsed_target_transmission_match": str(
                        solution.exact_parsed_transmission_match
                    ).lower(),
                    "signed_anchor_fractional_correction_error": f64(anchor_correction),
                    "absolute_anchor_fractional_correction_error": f64(
                        abs(anchor_correction)
                    ),
                    "direct_fitted_minus_affine_copied1_tau": f64(
                        fitted_tau - affine_copied_tau
                    ),
                    "direct_fitted_minus_affine_direct1_tau": f64(
                        fitted_tau - affine_direct_tau
                    ),
                    "scale_trace_path_relative_to_cache": solution.trace_relative_path,
                    "scale_trace_sha256": solution.trace_sha256,
                    "scale_trace_evaluation_count": str(
                        solution.trace_evaluation_count
                    ),
                    "anchor_run_return_codes": ";".join(
                        str(run.return_code)
                        for run in (
                            solution.scale0,
                            solution.scale1,
                            solution.fitted,
                        )
                    ),
                    "anchor_unresolved_warning_counts": ";".join(
                        str(run.parsed.warning_count or 0)
                        for run in (
                            solution.scale0,
                            solution.scale1,
                            solution.fitted,
                        )
                    ),
                    "ancillary_screening_transmission_rank1": "false",
                    "direct_full_grid_evaluated": "false",
                    "direct_full_grid_t225_el80_transmission": "",
                    "direct_full_grid_exact_target_match": "",
                    "interpretation": "post_hoc_candidate_input_recipe_not_custody_proof",
                }
            )
    return rows


def coefficient_rows_for_surface(
    *,
    lane: str,
    target: str,
    profile: str,
    solution: ScaleSolution,
    ratios: dict[str, np.ndarray],
    source: Any,
) -> list[dict[str, str]]:
    season, percentile_value, family = parse_profile(profile)
    elevation_rad = np.deg2rad(ELEVATIONS_DEG)
    rows: list[dict[str, str]] = []
    for band, ratio in ratios.items():
        recovered = np.polyfit(elevation_rad, ratio, 6)
        rounded = np.round(recovered, 8)
        literals = source.coefficients[target][band]
        source_values = np.asarray([float(value) for value in literals])
        for index, source_literal in enumerate(literals):
            rows.append(
                {
                    "target_model": target,
                    "target_registry_family": "legacy_generic_unprefixed_am_q_registry",
                    "source_profile": profile,
                    "source_profile_season": season,
                    "source_profile_percentile": str(percentile_value),
                    "source_profile_family": family,
                    "h2o_scale_decimal": solution.scale_decimal,
                    "evaluation_lane": lane,
                    "band": band,
                    "frequency_ghz": f64(BAND_FREQUENCIES_GHZ[band]),
                    "degree_power": str(6 - index),
                    "repair_base_source_literal": source_literal,
                    "candidate_unrounded_binary64": f64(recovered[index]),
                    "candidate_rounded_8_decimals": f64(rounded[index]),
                    "absolute_unrounded_to_source_difference": f64(
                        abs(recovered[index] - source_values[index])
                    ),
                    "exact_after_8_decimal_rounding": str(
                        bool(rounded[index] == source_values[index])
                    ).lower(),
                    "interpretation": "weaker_q95_ratio_surface_provenance_hypothesis_raw_grid_absent",
                }
            )
    return rows


def add_q95_surface_rows(
    *,
    rows: list[dict[str, str]],
    coefficient_rows: list[dict[str, str]],
    lane: str,
    profile: str,
    solution: ScaleSolution,
    arrays: dict[str, np.ndarray],
    source: Any,
    runs: list[RunResult] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference_index = frequency_index(REFERENCE_FREQUENCY_GHZ)
    reference_tx = arrays["tx"][reference_index]
    ratios: dict[str, np.ndarray] = {}
    source_ratios: dict[str, np.ndarray] = {}
    elevation_rad = np.deg2rad(ELEVATIONS_DEG)
    for band, frequency in BAND_FREQUENCIES_GHZ.items():
        band_index = frequency_index(frequency)
        ratio = arrays["tx"][band_index] / reference_tx
        source_ratio = np.polyval(
            np.asarray(
                [float(value) for value in source.coefficients["am_q95"][band]],
                dtype=np.float64,
            ),
            elevation_rad,
        )
        if not (
            np.all(np.isfinite(ratio))
            and np.all(ratio > 0.0)
            and np.all(np.isfinite(source_ratio))
            and np.all(source_ratio > 0.0)
        ):
            raise RuntimeError(f"invalid q95 ratio domain for {profile}/{lane}/{band}")
        ratios[band] = ratio
        source_ratios[band] = source_ratio
        rows.append(
            summarize_metric_row(
                target="am_q95",
                profile=profile,
                solution=solution,
                lane=lane,
                comparison_quantity="nominal_frequency_transmission_ratio_to_225ghz",
                scope=f"nominal_ratio_surface_{band}",
                band=band,
                frequency=frequency,
                truth_kind="repair_base_degree6_ratio_literal_surface_raw_q95_absent",
                truth_artifact=CALIBRATE_REL.as_posix(),
                truth_sha256=FROZEN_REPAIR_INPUTS[CALIBRATE_REL],
                predicted_value=ratio,
                truth_value=source_ratio,
                predicted_tau=-np.log(ratio),
                truth_tau=-np.log(source_ratio),
                predicted_trj=None,
                truth_trj=None,
                runs=runs,
            )
        )
    combined_ratio = np.concatenate([ratios[band] for band in BAND_FREQUENCIES_GHZ])
    combined_source = np.concatenate(
        [source_ratios[band] for band in BAND_FREQUENCIES_GHZ]
    )
    rows.append(
        summarize_metric_row(
            target="am_q95",
            profile=profile,
            solution=solution,
            lane=lane,
            comparison_quantity="nominal_frequency_transmission_ratio_to_225ghz",
            scope="all_nominal_ratio_surfaces",
            band="all",
            frequency=None,
            truth_kind="repair_base_degree6_ratio_literal_surfaces_raw_q95_absent",
            truth_artifact=CALIBRATE_REL.as_posix(),
            truth_sha256=FROZEN_REPAIR_INPUTS[CALIBRATE_REL],
            predicted_value=combined_ratio,
            truth_value=combined_source,
            predicted_tau=-np.log(combined_ratio),
            truth_tau=-np.log(combined_source),
            predicted_trj=None,
            truth_trj=None,
            runs=runs,
        )
    )
    coefficient_rows.extend(
        coefficient_rows_for_surface(
            lane=lane,
            target="am_q95",
            profile=profile,
            solution=solution,
            ratios=ratios,
            source=source,
        )
    )
    return ratios, source_ratios


def rank_constructed_rows(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for target in TARGETS:
        scope = "full_grid" if target != "am_q95" else "all_nominal_ratio_surfaces"
        candidates = [
            row
            for row in rows
            if row["target_model"] == target
            and row["evaluation_lane"] == "affine_scale0_to_copied_scale1_all_profiles"
            and row["comparison_scope"] == scope
        ]
        if len(candidates) != len(PROFILE_STEMS):
            raise RuntimeError(
                f"unexpected constructed ranking population for {target}"
            )
        tx_order = sorted(
            candidates,
            key=lambda row: (
                row["anchor_exact_parsed_tx_match"] != "true",
                float(row["transmission_or_ratio_rms_residual"]),
                float(row["transmission_or_ratio_max_abs_residual"]),
                row["source_profile"],
            ),
        )
        tx_rank = {
            row["source_profile"]: index + 1 for index, row in enumerate(tx_order)
        }
        if target != "am_q95":
            trj_order = sorted(
                candidates,
                key=lambda row: (
                    row["anchor_exact_parsed_tx_match"] != "true",
                    float(row["trj_rms_residual_k"]),
                    float(row["trj_max_abs_residual_k"]),
                    row["source_profile"],
                ),
            )
            trj_rank = {
                row["source_profile"]: index + 1 for index, row in enumerate(trj_order)
            }
        else:
            trj_rank = {}
        selected[target] = tx_order[0]["source_profile"]
        for row in rows:
            if (
                row["target_model"] == target
                and row["evaluation_lane"]
                == "affine_scale0_to_copied_scale1_all_profiles"
            ):
                row["transmission_rms_rank"] = str(tx_rank[row["source_profile"]])
                if target != "am_q95":
                    row["trj_rms_rank"] = str(trj_rank[row["source_profile"]])
                row["ancillary_screening_transmission_rank1"] = str(
                    row["source_profile"] == selected[target]
                ).lower()
    return selected


def build_constructed_metrics(
    *,
    runner: Runner,
    jobs: int,
    am_root: Path,
    source: Any,
    copied_inventory: dict[str, dict[str, Any]],
    legacy: dict[str, dict[str, np.ndarray]],
    solutions: dict[tuple[str, str], ScaleSolution],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
    dict[str, dict[str, np.ndarray]],
]:
    rows: list[dict[str, str]] = []
    coefficient_rows: list[dict[str, str]] = []
    best_scores: dict[str, tuple[Any, ...]] = {}
    best_arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_index = frequency_index(REFERENCE_FREQUENCY_GHZ)

    for profile_index, profile in enumerate(PROFILE_STEMS, start=1):
        print(
            f"P1 constructed-grid profile {profile_index}/{len(PROFILE_STEMS)}: "
            f"{profile}",
            file=sys.stderr,
            flush=True,
        )
        copied = load_copied_profile(am_root, profile)
        scale0, scale0_runs = run_full_profile(
            runner=runner,
            jobs=jobs,
            stage="full_grid_scale0_construction_endpoint",
            profile=profile,
            target="shared",
            scale_decimal=f64(0.0),
        )
        if np.max(np.abs(np.exp(-scale0["tau"]) - scale0["tx"])) > 1.0e-6:
            raise RuntimeError(f"direct scale0 tau/tx mismatch for {profile}")

        for target in TARGETS:
            solution = solutions[target, profile]
            scale = solution.scale_value
            predicted_tau = scale0["tau"] + scale * (copied["tau"] - scale0["tau"])
            predicted_tx = np.exp(-predicted_tau)
            predicted_trj = scale0["trj"] + scale * (copied["trj"] - scale0["trj"])
            if not (
                np.all(np.isfinite(predicted_tau))
                and np.all(predicted_tau >= 0.0)
                and np.all(np.isfinite(predicted_tx))
                and np.all(predicted_tx >= 0.0)
                and np.all(np.isfinite(predicted_trj))
            ):
                raise RuntimeError(
                    f"invalid affine full-grid construction for {target}/{profile}"
                )
            arrays = {"tau": predicted_tau, "tx": predicted_tx, "trj": predicted_trj}

            if target in LEGACY_RAW_SOURCES:
                truth = legacy[target]
                metadata = LEGACY_RAW_SOURCES[target]
                truth_tau = -np.log(truth["tx"])
                full_row = summarize_metric_row(
                    target=target,
                    profile=profile,
                    solution=solution,
                    lane="affine_scale0_to_copied_scale1_all_profiles",
                    comparison_quantity="full_grid_transmission_and_trj",
                    scope="full_grid",
                    band="all",
                    frequency=None,
                    truth_kind="digest_bound_recovered_legacy_raw_grid",
                    truth_artifact=metadata["filename"],
                    truth_sha256=metadata["sha256"],
                    predicted_value=predicted_tx,
                    truth_value=truth["tx"],
                    predicted_tau=predicted_tau,
                    truth_tau=truth_tau,
                    predicted_trj=predicted_trj,
                    truth_trj=truth["trj"],
                    runs=scale0_runs,
                )
                rows.append(full_row)
                score = (
                    not solution.exact_parsed_transmission_match,
                    float(full_row["transmission_or_ratio_rms_residual"]),
                    float(full_row["transmission_or_ratio_max_abs_residual"]),
                    profile,
                )
                if target not in best_scores or score < best_scores[target]:
                    best_scores[target] = score
                    best_arrays[target] = {
                        "tau": predicted_tau.copy(),
                        "trj": predicted_trj.copy(),
                    }
                for band, frequency in BAND_FREQUENCIES_GHZ.items():
                    index = frequency_index(frequency)
                    rows.append(
                        summarize_metric_row(
                            target=target,
                            profile=profile,
                            solution=solution,
                            lane="affine_scale0_to_copied_scale1_all_profiles",
                            comparison_quantity="nominal_frequency_transmission_and_trj",
                            scope=f"nominal_{band}",
                            band=band,
                            frequency=frequency,
                            truth_kind="digest_bound_recovered_legacy_raw_grid",
                            truth_artifact=metadata["filename"],
                            truth_sha256=metadata["sha256"],
                            predicted_value=predicted_tx[index],
                            truth_value=truth["tx"][index],
                            predicted_tau=predicted_tau[index],
                            truth_tau=truth_tau[index],
                            predicted_trj=predicted_trj[index],
                            truth_trj=truth["trj"][index],
                            runs=scale0_runs,
                        )
                    )
            else:
                before = len(rows)
                add_q95_surface_rows(
                    rows=rows,
                    coefficient_rows=coefficient_rows,
                    lane="affine_scale0_to_copied_scale1_all_profiles",
                    profile=profile,
                    solution=solution,
                    arrays=arrays,
                    source=source,
                    runs=scale0_runs,
                )
                aggregate = next(
                    row
                    for row in rows[before:]
                    if row["comparison_scope"] == "all_nominal_ratio_surfaces"
                )
                score = (
                    not solution.exact_parsed_transmission_match,
                    float(aggregate["transmission_or_ratio_rms_residual"]),
                    float(aggregate["transmission_or_ratio_max_abs_residual"]),
                    profile,
                )
                if target not in best_scores or score < best_scores[target]:
                    best_scores[target] = score
                    best_arrays[target] = {
                        "tau": predicted_tau.copy(),
                        "trj": predicted_trj.copy(),
                    }

            constructed_anchor_tx = float(predicted_tx[reference_index, -1])
            if not math.isfinite(constructed_anchor_tx):
                raise RuntimeError(
                    f"non-finite constructed anchor for {target}/{profile}"
                )

    selected = rank_constructed_rows(rows)
    for target, profile in selected.items():
        if best_scores[target][-1] != profile:
            raise RuntimeError(f"internal best-hypothesis mismatch for {target}")
    return rows, coefficient_rows, selected, best_arrays


def numeric_run_aggregate_sha256(runs: list[RunResult]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(
        runs,
        key=lambda item: (
            item.spec.profile,
            item.spec.target,
            item.spec.f_min_centi_ghz,
            item.spec.f_max_centi_ghz,
            item.spec.zenith_angle_deg,
            item.spec.scale_decimal,
        ),
    )
    for run in ordered:
        scientific_identity = {
            "profile": run.spec.profile,
            "target": run.spec.target,
            "f_min_centi_ghz": run.spec.f_min_centi_ghz,
            "f_max_centi_ghz": run.spec.f_max_centi_ghz,
            "zenith_angle_deg": run.spec.zenith_angle_deg,
            "scale_decimal": run.spec.scale_decimal,
        }
        digest.update(
            json.dumps(
                scientific_identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(bytes.fromhex(run.parsed.numeric_text_sha256))
        digest.update(b"\0")
    return digest.hexdigest()


def execution_run_digest_summary(runs: list[RunObservation]) -> dict[str, Any]:
    ordered = sorted(
        runs,
        key=lambda item: (
            item.spec.stage,
            item.spec.profile,
            item.spec.target,
            item.spec.f_min_centi_ghz,
            item.spec.f_max_centi_ghz,
            item.spec.zenith_angle_deg,
            item.spec.scale_decimal,
        ),
    )
    numeric_digest = hashlib.sha256()
    normalized_output_digest = hashlib.sha256()
    return_code_counts: dict[str, int] = {}
    version_counts: dict[str, int] = {}
    diagnostic_totals = {
        "warning_bearing_run_count": 0,
        "unresolved_line_warning_count_sum": 0,
        "unresolved_column_warning_line_count": 0,
        "unresolved_summary_warning_line_count": 0,
        "other_warning_line_count": 0,
        "error_line_count": 0,
    }
    for run in ordered:
        identity_bytes = json.dumps(
            run.spec.request_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        for digest, value in (
            (numeric_digest, run.numeric_text_sha256),
            (normalized_output_digest, run.normalized_output_sha256),
        ):
            digest.update(identity_bytes)
            digest.update(b"\0")
            digest.update(bytes.fromhex(value))
            digest.update(b"\0")
        return_key = str(run.return_code)
        return_code_counts[return_key] = return_code_counts.get(return_key, 0) + 1
        version = run.am_version_identity
        version_counts[version] = version_counts.get(version, 0) + 1
        if run.unresolved_line_warning_count is not None:
            diagnostic_totals["warning_bearing_run_count"] += 1
            diagnostic_totals["unresolved_line_warning_count_sum"] += (
                run.unresolved_line_warning_count
            )
        for field in (
            "unresolved_column_warning_line_count",
            "unresolved_summary_warning_line_count",
            "other_warning_line_count",
            "error_line_count",
        ):
            diagnostic_totals[field] += int(getattr(run, field))
    algorithm = (
        "for each unique referenced run sorted by "
        "(stage,profile,target,f_min_centi_ghz,f_max_centi_ghz,"
        "zenith_angle_deg,scale_decimal): UTF-8 canonical JSON RunSpec request "
        "+ NUL + digest bytes + NUL"
    )
    return {
        "unique_referenced_run_count": len(ordered),
        "normalized_numeric_text_aggregate_sha256": numeric_digest.hexdigest(),
        "normalized_numeric_text_aggregate_algorithm": algorithm,
        "normalized_warning_bearing_output_aggregate_sha256": (
            normalized_output_digest.hexdigest()
        ),
        "normalized_warning_bearing_output_aggregate_algorithm": algorithm,
        "return_code_counts": dict(sorted(return_code_counts.items())),
        "am_version_identity_counts": dict(sorted(version_counts.items())),
        "diagnostic_totals": diagnostic_totals,
        "per_run_raw_and_normalized_digests": (
            "bound to execution-context SHA-256 in external execution sidecars"
        ),
    }


def add_direct_legacy_rows(
    *,
    rows: list[dict[str, str]],
    target: str,
    profile: str,
    solution: ScaleSolution,
    direct: dict[str, np.ndarray],
    runs: list[RunResult],
    legacy: dict[str, dict[str, np.ndarray]],
    lane: str,
) -> None:
    truth = legacy[target]
    truth_tau = -np.log(truth["tx"])
    metadata = LEGACY_RAW_SOURCES[target]
    rows.append(
        summarize_metric_row(
            target=target,
            profile=profile,
            solution=solution,
            lane=lane,
            comparison_quantity="full_grid_transmission_and_trj",
            scope="full_grid",
            band="all",
            frequency=None,
            truth_kind="digest_bound_recovered_legacy_raw_grid",
            truth_artifact=metadata["filename"],
            truth_sha256=metadata["sha256"],
            predicted_value=direct["tx"],
            truth_value=truth["tx"],
            predicted_tau=direct["tau"],
            truth_tau=truth_tau,
            predicted_trj=direct["trj"],
            truth_trj=truth["trj"],
            runs=runs,
        )
    )
    for band, frequency in BAND_FREQUENCIES_GHZ.items():
        index = frequency_index(frequency)
        rows.append(
            summarize_metric_row(
                target=target,
                profile=profile,
                solution=solution,
                lane=lane,
                comparison_quantity="nominal_frequency_transmission_and_trj",
                scope=f"nominal_{band}",
                band=band,
                frequency=frequency,
                truth_kind="digest_bound_recovered_legacy_raw_grid",
                truth_artifact=metadata["filename"],
                truth_sha256=metadata["sha256"],
                predicted_value=direct["tx"][index],
                truth_value=truth["tx"][index],
                predicted_tau=direct["tau"][index],
                truth_tau=truth_tau[index],
                predicted_trj=direct["trj"][index],
                truth_trj=truth["trj"][index],
                runs=runs,
            )
        )


def add_affine_direct_validation_rows(
    *,
    rows: list[dict[str, str]],
    target: str,
    profile: str,
    solution: ScaleSolution,
    affine: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
    runs: list[RunResult],
) -> None:
    aggregate_digest = numeric_run_aggregate_sha256(runs)
    affine_tx = np.exp(-affine["tau"])
    rows.append(
        summarize_metric_row(
            target=target,
            profile=profile,
            solution=solution,
            lane="affine_construction_vs_direct_am_all_hypotheses_validation",
            comparison_quantity="full_grid_transmission_and_trj",
            scope="full_grid",
            band="all",
            frequency=None,
            truth_kind="direct_am_same_profile_and_frozen_scale",
            truth_artifact="external_cache_31_direct_full_grid_runs",
            truth_sha256=aggregate_digest,
            predicted_value=affine_tx,
            truth_value=direct["tx"],
            predicted_tau=affine["tau"],
            truth_tau=direct["tau"],
            predicted_trj=affine["trj"],
            truth_trj=direct["trj"],
            runs=runs,
        )
    )
    for band, frequency in BAND_FREQUENCIES_GHZ.items():
        index = frequency_index(frequency)
        rows.append(
            summarize_metric_row(
                target=target,
                profile=profile,
                solution=solution,
                lane="affine_construction_vs_direct_am_all_hypotheses_validation",
                comparison_quantity="nominal_frequency_transmission_and_trj",
                scope=f"nominal_{band}",
                band=band,
                frequency=frequency,
                truth_kind="direct_am_same_profile_and_frozen_scale",
                truth_artifact="external_cache_31_direct_full_grid_runs",
                truth_sha256=aggregate_digest,
                predicted_value=affine_tx[index],
                truth_value=direct["tx"][index],
                predicted_tau=affine["tau"][index],
                truth_tau=direct["tau"][index],
                predicted_trj=affine["trj"][index],
                truth_trj=direct["trj"][index],
                runs=runs,
            )
        )


def build_all_direct_matrix(
    *,
    runner: Runner,
    jobs: int,
    source: Any,
    legacy: dict[str, dict[str, np.ndarray]],
    solutions: dict[tuple[str, str], ScaleSolution],
    am_root: Path,
    ancillary_screening_selected: dict[str, str],
    metric_rows: list[dict[str, str]],
    coefficient_rows: list[dict[str, str]],
    scale_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    lane = "direct_am_fitted_scale_all_25_profiles"
    reference_index = frequency_index(REFERENCE_FREQUENCY_GHZ)
    principal_rows: dict[tuple[str, str], dict[str, str]] = {}
    affine_principal_rows: dict[tuple[str, str], dict[str, str]] = {}
    run_summaries: dict[tuple[str, str], dict[str, Any]] = {}

    for profile_index, profile in enumerate(PROFILE_STEMS, start=1):
        print(
            f"P1 all-direct profile {profile_index}/{len(PROFILE_STEMS)}: {profile}",
            file=sys.stderr,
            flush=True,
        )
        copied = load_copied_profile(am_root, profile)
        scale0, _ = run_full_profile(
            runner=runner,
            jobs=jobs,
            stage="full_grid_scale0_construction_endpoint",
            profile=profile,
            target="shared",
            scale_decimal=f64(0.0),
        )
        for target in TARGETS:
            solution = solutions[target, profile]
            scale = solution.scale_value
            affine = {
                "tau": scale0["tau"] + scale * (copied["tau"] - scale0["tau"]),
                "trj": scale0["trj"] + scale * (copied["trj"] - scale0["trj"]),
            }
            stage = (
                "direct_full_grid_selected_transmission_rank1"
                if profile == ancillary_screening_selected[target]
                else "direct_full_grid_all_hypotheses"
            )
            direct, runs = run_full_profile(
                runner=runner,
                jobs=jobs,
                stage=stage,
                profile=profile,
                target=target,
                scale_decimal=solution.scale_decimal,
            )
            full_anchor_tx = float(direct["tx"][reference_index, -1])
            full_anchor_exact = full_anchor_tx == solution.target_transmission
            if not full_anchor_exact:
                raise RuntimeError(
                    f"direct full-grid T225 anchor mismatch for {target}/{profile}: "
                    f"{full_anchor_tx} != {solution.target_transmission}"
                )
            scale_row = next(
                row
                for row in scale_rows
                if row["target_model"] == target and row["source_profile"] == profile
            )
            scale_row["ancillary_screening_transmission_rank1"] = str(
                profile == ancillary_screening_selected[target]
            ).lower()
            scale_row["direct_full_grid_evaluated"] = "true"
            scale_row["direct_full_grid_t225_el80_transmission"] = f64(full_anchor_tx)
            scale_row["direct_full_grid_exact_target_match"] = "true"

            before_truth = len(metric_rows)
            if target in LEGACY_RAW_SOURCES:
                add_direct_legacy_rows(
                    rows=metric_rows,
                    target=target,
                    profile=profile,
                    solution=solution,
                    direct=direct,
                    runs=runs,
                    legacy=legacy,
                    lane=lane,
                )
                principal_scope = "full_grid"
            else:
                add_q95_surface_rows(
                    rows=metric_rows,
                    coefficient_rows=coefficient_rows,
                    lane=lane,
                    profile=profile,
                    solution=solution,
                    arrays=direct,
                    source=source,
                    runs=runs,
                )
                principal_scope = "all_nominal_ratio_surfaces"
            truth_rows = metric_rows[before_truth:]
            for row in truth_rows:
                row["direct_full_grid_evaluated"] = "true"
            principal_rows[target, profile] = next(
                row for row in truth_rows if row["comparison_scope"] == principal_scope
            )

            before_affine = len(metric_rows)
            add_affine_direct_validation_rows(
                rows=metric_rows,
                target=target,
                profile=profile,
                solution=solution,
                affine=affine,
                direct=direct,
                runs=runs,
            )
            affine_rows = metric_rows[before_affine:]
            for row in affine_rows:
                row["direct_full_grid_evaluated"] = "true"
            affine_principal_rows[target, profile] = next(
                row for row in affine_rows if row["comparison_scope"] == "full_grid"
            )
            run_summaries[target, profile] = {
                "direct_run_numeric_text_aggregate_sha256": (
                    numeric_run_aggregate_sha256(runs)
                ),
                "direct_run_count": len(runs),
                "warning_status_run_count": sum(run.return_code == 1 for run in runs),
            }

    results: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        candidates = [principal_rows[target, profile] for profile in PROFILE_STEMS]
        transmission_order = sorted(
            candidates,
            key=lambda row: (
                float(row["transmission_or_ratio_rms_residual"]),
                float(row["transmission_or_ratio_max_abs_residual"]),
                row["source_profile"],
            ),
        )
        transmission_ranks = {
            row["source_profile"]: index + 1
            for index, row in enumerate(transmission_order)
        }
        if target in LEGACY_RAW_SOURCES:
            trj_order = sorted(
                candidates,
                key=lambda row: (
                    float(row["trj_rms_residual_k"]),
                    float(row["trj_max_abs_residual_k"]),
                    row["source_profile"],
                ),
            )
            trj_ranks = {
                row["source_profile"]: index + 1 for index, row in enumerate(trj_order)
            }
        else:
            trj_order = []
            trj_ranks = {}
        for row in metric_rows:
            if row["target_model"] == target and row["evaluation_lane"] == lane:
                row["transmission_rms_rank"] = str(
                    transmission_ranks[row["source_profile"]]
                )
                if target in LEGACY_RAW_SOURCES:
                    row["trj_rms_rank"] = str(trj_ranks[row["source_profile"]])

        def summarize_rank1(row: dict[str, str]) -> dict[str, Any]:
            profile = row["source_profile"]
            solution = solutions[target, profile]
            affine_row = affine_principal_rows[target, profile]
            return {
                "profile": profile,
                "source_profile_family": parse_profile(profile)[2],
                "h2o_scale_decimal": solution.scale_decimal,
                "direct_full_grid_t225_el80_transmission": next(
                    scale_row["direct_full_grid_t225_el80_transmission"]
                    for scale_row in scale_rows
                    if scale_row["target_model"] == target
                    and scale_row["source_profile"] == profile
                ),
                "direct_full_grid_exact_parsed_target_match": True,
                "transmission_or_ratio_rms_residual": row[
                    "transmission_or_ratio_rms_residual"
                ],
                "transmission_or_ratio_max_abs_residual": row[
                    "transmission_or_ratio_max_abs_residual"
                ],
                "fractional_correction_max_abs_error": row[
                    "fractional_correction_max_abs_error"
                ],
                "trj_rms_residual_k": row["trj_rms_residual_k"],
                "trj_max_abs_residual_k": row["trj_max_abs_residual_k"],
                "affine_vs_direct_full_grid_fractional_correction_max_abs_error": (
                    affine_row["fractional_correction_max_abs_error"]
                ),
                "affine_vs_direct_full_grid_trj_rms_residual_k": affine_row[
                    "trj_rms_residual_k"
                ],
                **run_summaries[target, profile],
            }

        if target in LEGACY_RAW_SOURCES:
            results[target] = {
                "direct_transmission_rms_rank1": summarize_rank1(transmission_order[0]),
                "direct_rayleigh_jeans_rms_rank1": summarize_rank1(trj_order[0]),
                "rankings_are_separate_no_composite": True,
            }
        else:
            results[target] = {
                "direct_nominal_ratio_surface_rms_rank1": summarize_rank1(
                    transmission_order[0]
                ),
                "rayleigh_jeans_ranking": "not_applicable_raw_q95_absent",
            }
    return results


def build_report(
    *,
    metric_rows: list[dict[str, str]],
    selected: dict[str, str],
    direct_results: dict[str, dict[str, Any]],
    copied_zero_transmission_count: int,
    execution_context_sha256: str,
    execution_run_summary: dict[str, Any],
) -> bytes:
    status_one_count = execution_run_summary["return_code_counts"].get("1", 0)
    run_count = execution_run_summary["unique_referenced_run_count"]
    lines = [
        "# SCI-CAL-001 AM 12.2 H2O-scale provenance-hypothesis report",
        "",
        "## Status",
        "",
        "This is diagnostic P1: a post-hoc candidate-input-recipe search. It is not historical custody proof, a holdout test, operator authorization, an operational-domain declaration, or observational photometric validation.",
        "",
        "The legacy `am_q25/am_q50/am_q75/am_q95` targets are the generic unprefixed TolTECA registry family. The copied `annual`, `DJF`, `MAM`, `JJA`, and `SON` MERRA-2 profiles are separate explicitly named AM-12.2 families. A numerical match does not rename a copied profile as a registered generic q artifact.",
        "",
        "## All-direct P1 rank-one hypotheses",
        "",
        "The fitted-scale 0--500 GHz by 31-elevation AM grid was run directly for all 100 target/profile hypotheses. Transmission and Rayleigh-Jeans rankings are separate; no unregistered composite score or near-exact cutoff was invented. q95 has only the weaker nominal ratio-surface ranking because its registered raw grid is absent.",
        "",
        "| Generic target | Direct ranking | Copied profile hypothesis | H2O scale | Direct RMS residual | Direct max absolute residual | Max correction error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        result = direct_results[target]
        ranked = (
            [
                ("transmission RMS", result["direct_transmission_rms_rank1"]),
                (
                    "Rayleigh-Jeans RMS",
                    result["direct_rayleigh_jeans_rms_rank1"],
                ),
            ]
            if target in LEGACY_RAW_SOURCES
            else [
                (
                    "nominal ratio-surface RMS",
                    result["direct_nominal_ratio_surface_rms_rank1"],
                )
            ]
        )
        for ranking, item in ranked:
            lines.append(
                "| `{target}` | {ranking} | `{profile}` | `{scale}` | `{rms}` | "
                "`{maximum}` | `{correction}` |".format(
                    target=target,
                    ranking=ranking,
                    profile=item["profile"],
                    scale=item["h2o_scale_decimal"],
                    rms=item["transmission_or_ratio_rms_residual"]
                    if ranking != "Rayleigh-Jeans RMS"
                    else item["trj_rms_residual_k"],
                    maximum=item["transmission_or_ratio_max_abs_residual"]
                    if ranking != "Rayleigh-Jeans RMS"
                    else item["trj_max_abs_residual_k"],
                    correction=item["fractional_correction_max_abs_error"],
                )
            )
    lines.extend(
        [
            "",
            "For q25/q50/q75 the principal direct comparison is the complete 50001-by-31 legacy transmission and Rayleigh-Jeans grid. For q95, whose registered raw datafile ID 461 is absent, it is only the 93-point nominal-frequency elevation-ratio surface derived from repair-base degree-six literals; that q95 evidence is strictly weaker.",
            "",
            "## Method boundary",
            "",
            "Each immutable AMC file contains exactly one `Nscale troposphere h2o %9` statement. The scale was seeded only from direct AM scale 0 and the copied scale-1 T225 optical depth, checked against a direct scale-1 run, then located on the exact parsed-transmission plateau at 225 GHz and EL80. The canonical scale is the midpoint of the innermost observed plateau interval after a fixed 48 bisections. Every evaluation and bracket is preserved as a digest-bound external-cache trace.",
            "",
            "Frozen P1 is fulfilled by the all-direct fitted-scale lane, not by a surrogate. The earlier all-profile affine LOS-tau/Trj construction is retained only as ancillary screening and is checked against every direct grid; it is not used for P1 completion or final ranking.",
            "",
            f"The 25 copied full products contain `{copied_zero_transmission_count}` exact printed transmission zeros at opaque spectral samples. These are accepted only with finite nonnegative `atmTaun` and an absolute tau-to-transmission consistency difference no larger than `{COPIED_TX_ABSOLUTE_PRINT_TOLERANCE:.1e}`. LOS tau, not `-log` of the rounded transmission field, is authoritative for construction and fractional-correction metrics.",
            "",
            "The one-percent fields are provisional numerical diagnostics only. They do not establish 5--10% absolute flux accuracy or approximately 5% observation-to-observation repeatability, and they do not reduce common calibrator, Beammap-extinction, selector, aligned-elevation, timing, or airmass systematics.",
            "",
            "No additional atmospheric profile, scale parameter, passband, frequency, elevation, or fitting degree was introduced. Numerical rank one can narrow a post-hoc candidate recipe but cannot establish generic-q custody, because profile selection and H2O-scale inference were performed after the legacy surfaces were known.",
            "",
        ]
    )
    lines.extend(
        [
            "## Execution integrity and predecessor disposition",
            "",
            f"The canonical v3 cache binds every sidecar and scale trace to immutable execution-context SHA-256 `{execution_context_sha256}`. One process held the whole-cache exclusive POSIX lock throughout execution and artifact construction; cache-only verification uses a shared lock. `LANG=C` and `LC_ALL=C` were pinned for AM subprocesses.",
            "",
            f"Across `{run_count}` unique referenced v3 AM runs, `{status_one_count}` returned status 1 with only the accepted unresolved-narrow-line warning structure. Those warnings, their counts, and normalized warning-bearing output identities remain explicit diagnostics; this report does not call the software execution clean or warning-free.",
            "",
            "The interrupted external cache `sci_cal_001_h2o_scale_p1_20260801_root_v2` is noncanonical and excluded. It was stopped after cache-provenance review because it had neither a whole-cache cross-process lock nor immutable execution-context binding. Its retained partial inventory is 12,455 raw outputs, 12,455 execution sidecars, and 100 scale traces. It completed 1,764 general all-hypothesis plus 124 selected-rank-one direct fitted-scale grids: 1,888 of the expected 3,100 total. The targeted SIGINT also left three excluded status -2 failure sidecars and three empty outputs for `LMT_JJA_5/am_q25` at ZA 10, 50, and 54. The v2 cache is never used for v3 artifacts or rankings.",
            "",
            "The first context-bound v3 development cache `sci_cal_001_h2o_scale_p1_context_v3_final_20260801_root` is also noncanonical and excluded. A pre-full-grid runtime review found that its digest inventory retained complete parsed arrays, projecting approximately 7.75 GB of avoidable retained memory. It was stopped during anchor inference after 1,811 matched raw-output/sidecar pairs and 16 traces; the targeted SIGINT left six empty outputs, three complete status -2 failure sidecars, and three empty atomic sidecar temporaries. The cache remains untouched and is never reused. This was a software-execution provenance correction, not a scientific-protocol change; the canonical process retains only frozen lightweight run identity, digest, and diagnostic records.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_manifest(
    *,
    args: argparse.Namespace,
    executable: BuildIdentity,
    am_root: Path,
    legacy_source_dir: Path,
    cache_dir: Path,
    profile_inventory: dict[str, Any],
    copied_inventory: dict[str, dict[str, Any]],
    am_contracts: dict[str, dict[str, str]],
    scale_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    coefficient_rows: list[dict[str, str]],
    selected: dict[str, str],
    direct_results: dict[str, dict[str, Any]],
    artifact_bytes: dict[str, bytes],
    execution_context: dict[str, Any],
    execution_context_sha256: str,
    execution_run_summary: dict[str, Any],
) -> bytes:
    copied_products = []
    for profile in PROFILE_STEMS:
        item = copied_inventory[profile]
        copied_products.append(
            {
                "profile": profile,
                "filename": item["filename"],
                "size_bytes": item["bytes"],
                "sha256": item["sha256"],
                "md5": item["md5"],
                "exact_zero_transmission_count": item["exact_zero_transmission_count"],
            }
        )
    exact_anchor_count = sum(
        row["exact_parsed_target_transmission_match"] == "true" for row in scale_rows
    )
    direct_full_anchor_count = sum(
        row["direct_full_grid_exact_target_match"] == "true" for row in scale_rows
    )
    failed_records = sorted(
        cache_dir.glob("failed_attempts/*.failure.json"), key=lambda path: path.name
    )
    failed_classes: dict[str, int] = {}
    for path in failed_records:
        record = json.loads(path.read_text(encoding="utf-8"))
        rejection = str(record.get("rejection", ""))
        if "insert_as_mru" in rejection or "Unable to rename" in rejection:
            classification = "am_cache_mru_rename_race_rejected"
        elif "unknown AM warning class" in rejection:
            classification = "unknown_warning_class_rejected"
        else:
            classification = "other_failed_closed_attempt"
        failed_classes[classification] = failed_classes.get(classification, 0) + 1
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity": {
            "package_id": PACKAGE_ID,
            "evidence_date": EVIDENCE_DATE,
            "study": "diagnostic_P1_documented_h2o_scale_provenance_hypothesis",
            "study_status": "post_hoc_provenance_hypothesis",
            "repair_base_sha": REPAIR_BASE_SHA,
            "repair_line_evidence_head": REPAIR_LINE_EVIDENCE_HEAD,
            "custody_proof": False,
            "holdout_evidence": False,
            "operator_authorization": "none",
            "operational_domain_authorization": "none",
        },
        "cache_execution_context": {
            "filename": EXECUTION_CONTEXT_NAME,
            "sha256": execution_context_sha256,
            "content": execution_context,
        },
        "scope": {
            "generic_registry_targets": list(TARGETS),
            "copied_profile_families": list(PROFILE_STEMS),
            "profile_count": len(PROFILE_STEMS),
            "hypothesis_count": len(TARGETS) * len(PROFILE_STEMS),
            "frequency_grid_ghz": {
                "minimum": f64(0.0),
                "maximum": f64(500.0),
                "step": f64(0.01),
                "count": 50001,
            },
            "elevation_grid_deg": {
                "minimum": f64(20.0),
                "maximum": f64(80.0),
                "step": f64(2.0),
                "count": 31,
            },
            "nominal_monochromatic_frequencies_ghz": {
                band: f64(value) for band, value in BAND_FREQUENCIES_GHZ.items()
            },
            "reference_frequency_ghz": f64(REFERENCE_FREQUENCY_GHZ),
            "passband_integration": "none_legacy_monochromatic_contract",
        },
        "input_provenance": {
            "am_root": str(am_root),
            "legacy_source_dir": str(legacy_source_dir),
            "am_executable": {
                "supplied_path": executable.supplied_path,
                "resolved_path": executable.resolved_path,
                "size_bytes": executable.size_bytes,
                "sha256": executable.sha256,
                "binary_format": executable.binary_format,
            },
            "builds": execution_context["builds"],
            "immutable_amc_profiles": profile_inventory,
            "copied_scale1_npz_products": copied_products,
            "copied_suite_manifest": {
                "filename": COPIED_MANIFEST_NAME,
                "sha256": EXPECTED_COPIED_MANIFEST_FILE_SHA256,
                "canonical_product_manifest_sha256": (
                    COPIED_CANONICAL_PRODUCT_MANIFEST_SHA256
                ),
            },
            "am_source_and_historical_workflow_contracts": am_contracts,
            "legacy_raw_sources": LEGACY_RAW_SOURCES,
            "missing_q95_raw_source": MISSING_Q95,
            "repair_base_inputs": {
                relative.as_posix(): {"sha256": digest}
                for relative, digest in FROZEN_REPAIR_INPUTS.items()
            },
            "frozen_protocol_artifact_inventory": execution_context["inputs"][
                "frozen_protocol_artifact_inventory"
            ],
        },
        "execution": {
            "cache_dir": "external_caller_supplied_not_artifact_identity",
            "raw_outputs_and_sidecars_committed": False,
            "working_directory_role": "Big_Atmosphere",
            "argv_template": [
                "<am-executable>",
                "LMT_am_inputs/<immutable-profile>.amc",
                "<fmin>",
                "GHz",
                "<fmax>",
                "GHz",
                "10",
                "MHz",
                "<zenith-angle>",
                "deg",
                "<frozen-h2o-scale-decimal>",
            ],
            "jobs": args.jobs,
            "omp_threads_per_process": args.omp_threads,
            "environment_overrides": {
                "OMP_NUM_THREADS": str(args.omp_threads),
                **PINNED_LOCALE,
                "AM_CACHE_PATH": (
                    "external_cache/am_spectral_cache/shard_<deterministic-index>"
                ),
            },
            "am_cache_sharding": {
                "shard_count": args.jobs,
                "assignment": (
                    "big-endian first 64 bits of sha256(cache_id) modulo shard_count"
                ),
                "locking": "one in-process lock per shard around each AM subprocess",
                "purpose": "prevent concurrent AM insert_as_mru rename races",
            },
            "whole_cache_lock": {
                "filename": CACHE_LOCK_NAME,
                "run_mode": "nonblocking POSIX exclusive lock",
                "cache_only_mode": "nonblocking POSIX shared lock",
            },
            "host": execution_context["execution_host"],
            "execution_context_sha256": execution_context_sha256,
            "return_contract": (
                "status 0, or status 1 only with the canonical unresolved-"
                "narrow-line warning, its exact 'Column included <integer> "
                "unresolved lines.' subrecords, and a complete exact grid; "
                "all other warning headers are rejected"
            ),
            "check_mode": "cache_only_no_am_subprocess",
            "normalized_numeric_digest_algorithms": {
                "direct_run_numeric_text_aggregate_sha256": (
                    "sha256(for each run sorted by "
                    "(profile,target,f_min_centi_ghz,f_max_centi_ghz,"
                    "zenith_angle_deg,scale_decimal): UTF-8 "
                    "json.dumps(scientific_identity,sort_keys=True,"
                    "separators=(',',':')) + NUL + "
                    "bytes.fromhex(numeric_text_sha256) + NUL)"
                ),
                "scale_trace_sha256": (
                    "sha256(UTF-8 json.dumps(trace,indent=2,sort_keys=True) + "
                    "newline); trace binds execution-context SHA-256 and each "
                    "evaluation's numeric-text and normalized warning-bearing "
                    "output SHA-256; it excludes raw combined-output SHA/cache ID"
                ),
                "all_referenced_normalized_output_aggregate_sha256": (
                    execution_run_summary[
                        "normalized_warning_bearing_output_aggregate_algorithm"
                    ]
                ),
            },
            "rejected_attempt_inventory": {
                "count": len(failed_records),
                "classification_counts": dict(sorted(failed_classes.items())),
                "combined_output_sha256_location": (
                    "external failure sidecars only, never committed manifest"
                ),
            },
        },
        "scale_inference": {
            "only_varying_parameter": "Nscale troposphere h2o through AMC argv %9",
            "target_coordinate": "parsed AM transmission at 225 GHz and EL80/ZA10",
            "target_literals": EXPECTED_TARGET_TRANSMISSIONS,
            "nonnegative": True,
            "seed": "direct scale0 tau plus immutable copied scale1 tau",
            "direct_linearity_checks": "direct scale1 and fitted-scale anchors",
            "root_policy": (
                "fixed 48-iteration monotone bisection of the exact parsed-"
                "transmission plateau; canonical midpoint of innermost inside scales"
            ),
            "exact_parsed_anchor_count": exact_anchor_count,
            "direct_full_grid_exact_anchor_count": direct_full_anchor_count,
            "hypothesis_count": len(scale_rows),
            "no_other_atmospheric_degrees": True,
        },
        "full_grid_construction": {
            "frozen_p1_completion_lane": (
                "direct AM fitted-scale 0-500 GHz by 10 MHz at EL20-80 by "
                "2 degrees for all 4 targets times 25 profiles"
            ),
            "frozen_p1_fulfilled_by_all_direct_lane": True,
            "direct_hypothesis_count": 100,
            "ancillary_affine_screening_lane": (
                "tau(s)=tau_direct_scale0+s*(tau_copied_scale1-tau_direct_scale0)"
            ),
            "opaque_sample_policy": {
                "copied_exact_zero_transmission_count": sum(
                    item["exact_zero_transmission_count"]
                    for item in copied_inventory.values()
                ),
                "tau_authority": "finite nonnegative copied atmTaun",
                "copied_tx_domain": "0 <= atmTtx <= 1",
                "tau_tx_absolute_print_tolerance": f64(
                    COPIED_TX_ABSOLUTE_PRINT_TOLERANCE
                ),
                "fractional_correction_metrics_reconstruct_tau_from_tx": False,
            },
            "ancillary_affine_trj_lane": (
                "explicitly approximate affine-in-scale Trj construction"
            ),
            "ancillary_affine_used_for_p1_completion_or_ranking": False,
            "q25_q50_q75_truth": "complete digest-bound legacy raw grids",
            "q95_truth": (
                "repair-base nominal-frequency ratio/elevation degree-six "
                "literals because registered raw q95 is absent"
            ),
        },
        "ranking": {
            "direct_transmission_ranking": (
                "full-grid transmission RMS, then maximum transmission residual, "
                "then profile name"
            ),
            "direct_rayleigh_jeans_ranking": (
                "full-grid Rayleigh-Jeans RMS, then maximum Rayleigh-Jeans "
                "residual, then profile name; separate from transmission"
            ),
            "direct_q95_ratio_ranking": (
                "combined 93-point nominal-ratio RMS, then maximum ratio residual, "
                "then profile name; raw q95 absent"
            ),
            "unregistered_composite_score": False,
            "near_exact_threshold": "not_preregistered_not_invented",
            "ancillary_affine_screening_transmission_rank1": selected,
            "final_rank1_identities": direct_results,
        },
        "results": {
            "scale_row_count": len(scale_rows),
            "metric_row_count": len(metric_rows),
            "coefficient_row_count": len(coefficient_rows),
            "direct_hypothesis_count": 100,
            "direct_full_grid_exact_anchor_count": direct_full_anchor_count,
            "direct_rank1_results": direct_results,
            "execution_run_digest_summary": execution_run_summary,
            "provisional_one_percent_is": (
                "numerical diagnostic only, not physical per-sample accuracy"
            ),
            "observational_absolute_flux_gate": "not_tested",
            "observational_repeatability_gate": "not_tested",
        },
        "interrupted_predecessor_attempt": INTERRUPTED_V2_DISPOSITION,
        "interrupted_first_v3_development_attempt": (
            INTERRUPTED_FIRST_V3_DEVELOPMENT_DISPOSITION
        ),
        "artifacts": {
            name: {
                "sha256": sha256_bytes(content),
                **(
                    {"row_count": len(scale_rows)}
                    if name == SCALES_NAME
                    else {"row_count": len(metric_rows)}
                    if name == METRICS_NAME
                    else {"row_count": len(coefficient_rows)}
                    if name == COEFFICIENTS_NAME
                    else {}
                ),
            }
            for name, content in artifact_bytes.items()
        }
        | {
            "generator": {
                "filename": Path(__file__).name,
                "sha256": sha256_path(Path(__file__)),
            }
        },
        "security": {
            "am_tree_modified": False,
            "legacy_repository_modified": False,
            "uploader_logs_read": False,
            "uploader_logs_or_credentials_copied": False,
            "network_access": False,
            "unity_access": False,
            "citlali_application_code_modified": False,
        },
        "limitations": [
            "Scale inference is post hoc and cannot prove the historical profile choice or custody chain.",
            "The affine LOS-tau/Trj lane is ancillary screening only; frozen P1 completion and ranking use all 100 direct grids.",
            "The registered q95 raw grid is absent, so q95 is compared only with repair-base ratio-polynomial literals.",
            "No result authorizes an atmosphere operator or an opacity/elevation domain.",
            "Software correctness, numerical representation fidelity, and observational performance remain separate gates.",
        ],
    }
    return json_bytes(payload)


def prepare_cache_dir(args: argparse.Namespace) -> Path:
    am_root = args.am_root.expanduser().resolve(strict=True)
    legacy_source_dir = args.legacy_source_dir.expanduser().resolve(strict=True)
    cache_dir = args.cache_dir.expanduser().resolve()
    for forbidden in (am_root, legacy_source_dir, PACKAGE_DIR):
        if is_relative_to(cache_dir, forbidden) or is_relative_to(forbidden, cache_dir):
            raise RuntimeError(
                f"--cache-dir must be external to all read-only/input roots: {cache_dir}"
            )
    if args.check:
        cache_dir = cache_dir.resolve(strict=True)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = cache_dir.resolve(strict=True)
    return cache_dir


def build_outputs(args: argparse.Namespace, *, cache_dir: Path) -> dict[str, bytes]:
    am_root = args.am_root.expanduser().resolve(strict=True)
    legacy_source_dir = args.legacy_source_dir.expanduser().resolve(strict=True)

    executable = build_identity(args.am_executable)
    historical_linux = build_identity(am_root / "am-12.2/bin/am")
    if historical_linux.sha256 != HISTORICAL_LINUX_BINARY_SHA256:
        raise RuntimeError("copied historical Linux AM binary digest mismatch")
    build_class = classify_regeneration_build(executable, historical_linux)
    if build_class != "copied_linux_reference_binary_reexecution":
        missing = []
        if args.compiler_executable is None:
            missing.append("--compiler-executable")
        if not args.native_build_command:
            missing.append("--native-build-command")
        if missing:
            raise RuntimeError(
                "distinct AM builds require complete provenance: missing "
                + ", ".join(missing)
            )
    compiler = compiler_identity(args.compiler_executable)
    protocol_inventory = validate_protocol_inputs()
    source = load_phase0_source()
    profile_inventory = inventory_profiles(am_root)
    am_contracts = validate_am_contract_files(am_root)
    copied_inventory = load_copied_inventory(am_root)
    legacy = load_legacy_sources(legacy_source_dir)
    context_path = cache_dir / EXECUTION_CONTEXT_NAME
    if args.check:
        if not context_path.is_file():
            raise RuntimeError(
                f"cache-only operation requires execution context: {context_path}"
            )
        execution_context = load_execution_context(context_path)
        execution_host = execution_context["execution_host"]
        expected_context = build_execution_context(
            args=args,
            am_root=am_root,
            legacy_source_dir=legacy_source_dir,
            executable=executable,
            historical_linux=historical_linux,
            compiler=compiler,
            profile_inventory=profile_inventory,
            copied_inventory=copied_inventory,
            am_contracts=am_contracts,
            protocol_inventory=protocol_inventory,
            execution_host=execution_host,
        )
        if execution_context != expected_context:
            raise RuntimeError(
                "cached execution context does not match the current runner, "
                "execution/build parameters, source/workflow, copied-AM, "
                "legacy, or repair inputs"
            )
    else:
        execution_host = host_identity()
        execution_context = build_execution_context(
            args=args,
            am_root=am_root,
            legacy_source_dir=legacy_source_dir,
            executable=executable,
            historical_linux=historical_linux,
            compiler=compiler,
            profile_inventory=profile_inventory,
            copied_inventory=copied_inventory,
            am_contracts=am_contracts,
            protocol_inventory=protocol_inventory,
            execution_host=execution_host,
        )
        atomic_write(context_path, json_bytes(execution_context))
    execution_context_sha256 = sha256_path(context_path)
    runner = Runner(
        executable=executable,
        am_root=am_root,
        cache_dir=cache_dir,
        omp_threads=args.omp_threads,
        cache_shard_count=args.jobs,
        execution_host=execution_host,
        execution_context_sha256=execution_context_sha256,
        execute=not args.check,
    )

    # AM populates its shared spectral cache lazily.  Prime the exact anchor
    # interval in one process before any parallel anchor work so concurrent
    # first-use writers cannot yield an unexplained status-1/no-warning run.
    # The prewarm has its own digest-bound raw output and sidecar.  In --check
    # mode it is loaded, never executed.
    print("P1 serial AM-cache prewarm", file=sys.stderr, flush=True)
    runner.run_or_load(anchor_spec(PROFILE_STEMS[0], "serial_cache_prewarm", f64(0.0)))
    print("P1 anchor-scale inference: 100 hypotheses", file=sys.stderr, flush=True)
    solutions = build_scale_solutions(runner=runner, am_root=am_root, jobs=args.jobs)
    scale_rows = build_scale_rows(solutions)
    metric_rows, coefficient_rows, selected, _best_arrays = build_constructed_metrics(
        runner=runner,
        jobs=args.jobs,
        am_root=am_root,
        source=source,
        copied_inventory=copied_inventory,
        legacy=legacy,
        solutions=solutions,
    )
    direct_results = build_all_direct_matrix(
        runner=runner,
        jobs=args.jobs,
        source=source,
        legacy=legacy,
        solutions=solutions,
        am_root=am_root,
        ancillary_screening_selected=selected,
        metric_rows=metric_rows,
        coefficient_rows=coefficient_rows,
        scale_rows=scale_rows,
    )

    if len(scale_rows) != 100:
        raise RuntimeError(f"unexpected scale row count: {len(scale_rows)}")
    if not all(
        row["direct_full_grid_exact_target_match"] == "true" for row in scale_rows
    ):
        raise RuntimeError("not all 100 direct full-grid anchors match exactly")
    if len(metric_rows) != 1200:
        raise RuntimeError(f"unexpected metric row count: {len(metric_rows)}")
    if len(coefficient_rows) != 1050:
        raise RuntimeError(f"unexpected coefficient row count: {len(coefficient_rows)}")
    failed_records = sorted(cache_dir.glob("failed_attempts/*.failure.json"))
    if failed_records:
        raise RuntimeError(
            "canonical P1 execution contains rejected AM attempts: "
            f"{len(failed_records)}"
        )
    execution_run_summary = execution_run_digest_summary(runner.observed_runs())

    artifacts = {
        SCALES_NAME: render_csv(scale_rows),
        METRICS_NAME: render_csv(metric_rows),
        COEFFICIENTS_NAME: render_csv(coefficient_rows),
    }
    artifacts[REPORT_NAME] = build_report(
        metric_rows=metric_rows,
        selected=selected,
        direct_results=direct_results,
        copied_zero_transmission_count=sum(
            item["exact_zero_transmission_count"] for item in copied_inventory.values()
        ),
        execution_context_sha256=execution_context_sha256,
        execution_run_summary=execution_run_summary,
    )
    artifacts[MANIFEST_NAME] = build_manifest(
        args=args,
        executable=executable,
        am_root=am_root,
        legacy_source_dir=legacy_source_dir,
        cache_dir=cache_dir,
        profile_inventory=profile_inventory,
        copied_inventory=copied_inventory,
        am_contracts=am_contracts,
        scale_rows=scale_rows,
        metric_rows=metric_rows,
        coefficient_rows=coefficient_rows,
        selected=selected,
        direct_results=direct_results,
        artifact_bytes=artifacts,
        execution_context=execution_context,
        execution_context_sha256=execution_context_sha256,
        execution_run_summary=execution_run_summary,
    )
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc diagnostic P1 using only immutable copied AMC profiles "
            "and their documented tropospheric-H2O scale argument."
        )
    )
    parser.add_argument("--am-executable", type=Path, required=True)
    parser.add_argument("--am-root", type=Path, default=DEFAULT_AM_ROOT)
    parser.add_argument(
        "--legacy-source-dir", type=Path, default=DEFAULT_LEGACY_SOURCE_DIR
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--compiler-executable",
        type=Path,
        help="compiler explicitly used to build a distinct AM executable",
    )
    parser.add_argument(
        "--native-build-command",
        help="exact command used to build a distinct AM executable",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.omp_threads < 1:
        parser.error("--omp-threads must be positive")

    cache_dir = prepare_cache_dir(args)
    cache_lock_handle = acquire_cache_lock(cache_dir, exclusive=not args.check)
    try:
        if not args.check:
            existing = sorted(
                path.name
                for path in cache_dir.iterdir()
                if path.name != CACHE_LOCK_NAME
            )
            if existing:
                raise RuntimeError(
                    "execution requires a fresh external cache; found existing "
                    f"entries below {cache_dir}: {existing}"
                )
        generated = build_outputs(args, cache_dir=cache_dir)
        for name in OUTPUT_NAMES:
            content = generated[name]
            path = PACKAGE_DIR / name
            if args.check:
                if not path.is_file() or path.read_bytes() != content:
                    raise RuntimeError(f"generated artifact differs: {path}")
            else:
                atomic_write(path, content)
    finally:
        fcntl.flock(cache_lock_handle.fileno(), fcntl.LOCK_UN)
        cache_lock_handle.close()
    print(
        ("verified" if args.check else "generated")
        + " SCI-CAL-001 H2O-scale hypothesis artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
