#!/usr/bin/env python3
"""Run or cache-verify the preregistered AM 12.2 resolution diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
PROFILES = ("LMT_DJF_5", "LMT_DJF_95")
ZENITH_ANGLES_DEG = (10, 70)
STEPS_MHZ = (10, 5, 2, 1)
FREQUENCIES_GHZ = (150.00, 214.29, 225.00, 272.73)
F_MIN_GHZ = 140
F_MAX_GHZ = 280
SCHEMA_VERSION = "sci-cal-001-am12-frequency-resolution-v2"
SIDECAR_SCHEMA_VERSION = "sci-cal-001-am12-frequency-run-v1"
EXPECTED_NATIVE_BINARY_SHA256 = (
    "78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb"
)
EXPECTED_NATIVE_BINARY_BYTES = 58_435_360
EXPECTED_COPIED_BINARY_SHA256 = (
    "3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c"
)
EXPECTED_INPUTS = {
    "LMT_DJF_5": {
        "amc": {
            "path_relative_to_am_root": ("Big_Atmosphere/LMT_am_inputs/LMT_DJF_5.amc"),
            "bytes": 4837,
            "sha256": (
                "fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474"
            ),
        },
        "npz": {
            "path_relative_to_am_root": ("Big_Atmosphere/LMT_am_npz/LMT_DJF_5.npz"),
            "bytes": 57_602_678,
            "sha256": (
                "214d9fa975c73afa01a4e1b5c5f068245779989578acd8574831b7fe2b6ed6cc"
            ),
        },
    },
    "LMT_DJF_95": {
        "amc": {
            "path_relative_to_am_root": ("Big_Atmosphere/LMT_am_inputs/LMT_DJF_95.amc"),
            "bytes": 4841,
            "sha256": (
                "b87b918b302425ef3d85aeedc285863a987579923289a37b97c6de5c935175e6"
            ),
        },
        "npz": {
            "path_relative_to_am_root": ("Big_Atmosphere/LMT_am_npz/LMT_DJF_95.npz"),
            "bytes": 57_602_678,
            "sha256": (
                "3dd961143e31a8db8182c35dd55472ad9ec943a711f652f6d55d485ee5ddb42d"
            ),
        },
    },
}
OUTPUTS = (
    "frequency_resolution_metrics.csv",
    "frequency_resolution_manifest.json",
    "FREQUENCY_RESOLUTION_REPORT.md",
)
VERSION_LINE = re.compile(r"^# (?P<identity>am version .+)$", re.MULTILINE)
SUMMARY_WARNING = re.compile(
    r"^! Warning: Encountered in-band lines narrower than the frequency\n"
    r"^!          grid spacing\.  The output configuration data includes\n"
    r"^!          the unresolved line count after each column definition\n"
    r"^!          for which this occurred\.  Count: (?P<count>\d+)$",
    re.MULTILINE,
)
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+\.\d+[eE][+-]\d+\s+")
NORMALIZED_OUTPUT_ALGORITHM = (
    "decode UTF-8; normalize line endings to LF; replace '# run time ...' with "
    "'# run time <volatile>' and '# dcache hit: ...' with "
    "'# dcache counters <volatile>'; append one LF"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def f64(value: float) -> str:
    return format(float(value), ".17e")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_csv(rows: list[dict[str, str]]) -> bytes:
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


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def case_id(profile_name: str, zenith_angle_deg: int, step_mhz: int) -> str:
    return f"{profile_name}_za{zenith_angle_deg:02d}_{step_mhz}mhz"


def raw_output_path(
    cache_dir: Path, profile_name: str, zenith_angle_deg: int, step_mhz: int
) -> Path:
    return (
        cache_dir
        / "raw_outputs"
        / f"{case_id(profile_name, zenith_angle_deg, step_mhz)}.dat"
    )


def sidecar_path(
    cache_dir: Path, profile_name: str, zenith_angle_deg: int, step_mhz: int
) -> Path:
    return (
        cache_dir
        / "execution_records"
        / f"{case_id(profile_name, zenith_angle_deg, step_mhz)}.run.json"
    )


def normalize_output(raw_output: bytes) -> bytes:
    text = raw_output.decode("utf-8")
    normalized = []
    for line in text.splitlines():
        if line.startswith("# run time "):
            normalized.append("# run time <volatile>")
        elif line.startswith("# dcache hit: "):
            normalized.append("# dcache counters <volatile>")
        else:
            normalized.append(line)
    return ("\n".join(normalized) + "\n").encode("utf-8")


def warning_classes(lines: list[str]) -> dict[str, int]:
    counts = {
        "unresolved_column": 0,
        "unresolved_summary": 0,
        "cache_insert_as_mru": 0,
        "cache_promote_to_mru": 0,
        "other": 0,
    }
    for line in lines:
        if not line.startswith("! Warning:"):
            continue
        if re.fullmatch(r"! Warning: Column included \d+ unresolved lines?\.", line):
            counts["unresolved_column"] += 1
        elif line == "! Warning: Encountered in-band lines narrower than the frequency":
            counts["unresolved_summary"] += 1
        elif line == "! Warning: Unable to rename file in insert_as_mru().":
            counts["cache_insert_as_mru"] += 1
        elif line == "! Warning: Unable to rename file in promote_to_mru().":
            counts["cache_promote_to_mru"] += 1
        else:
            counts["other"] += 1
    return counts


def parse_output(
    raw_output: bytes, expected_rows: int, *, label: str
) -> dict[str, Any]:
    try:
        text = raw_output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"non-UTF-8 AM output for {label}") from error
    lines = text.splitlines()
    versions = VERSION_LINE.findall(text)
    if len(versions) != 1 or not versions[0].startswith("am version 12.2 ("):
        raise RuntimeError(f"unexpected AM identity for {label}: {versions}")
    table_lines = [line for line in lines if NUMERIC_PATTERN.match(line)]
    if len(table_lines) != expected_rows:
        raise RuntimeError(
            f"AM output has {len(table_lines)} data rows for {label}, "
            f"expected {expected_rows}"
        )
    table = np.loadtxt(io.StringIO("\n".join(table_lines)), dtype=np.float64)
    if table.shape != (expected_rows, 5) or not np.all(np.isfinite(table)):
        raise RuntimeError(f"unexpected AM table for {label}: {table.shape}")
    summary_matches = [int(value) for value in SUMMARY_WARNING.findall(text)]
    if len(summary_matches) > 1:
        raise RuntimeError(f"multiple AM summary warnings for {label}")
    classes = warning_classes(lines)
    if classes["unresolved_summary"] != len(summary_matches):
        raise RuntimeError(f"inconsistent unresolved warning structure for {label}")
    error_lines = [line for line in lines if line.startswith(("! Error:", "! Fatal:"))]
    unexpected_warning_count = (
        classes["cache_insert_as_mru"]
        + classes["cache_promote_to_mru"]
        + classes["other"]
    )
    if unexpected_warning_count or error_lines:
        raise RuntimeError(
            f"unexpected AM diagnostics for {label}: "
            f"warning_classes={classes}, errors={error_lines}"
        )
    normalized = normalize_output(raw_output)
    numeric = ("\n".join(table_lines) + "\n").encode("utf-8")
    return {
        "table": table,
        "am_identity": versions[0],
        "unresolved_line_warning_count": (summary_matches[0] if summary_matches else 0),
        "warning_class_counts": classes,
        "unexpected_warning_count": unexpected_warning_count,
        "error_line_count": len(error_lines),
        "raw_output_bytes": len(raw_output),
        "raw_output_sha256": sha256_bytes(raw_output),
        "normalized_output_sha256": sha256_bytes(normalized),
        "numeric_output_sha256": sha256_bytes(numeric),
    }


def validate_return_contract(
    return_code: int, parsed: dict[str, Any], *, label: str
) -> None:
    if return_code == 0:
        return
    if return_code == 1 and parsed["unresolved_line_warning_count"] > 0:
        return
    raise RuntimeError(
        f"unexpected AM return contract for {label}: return={return_code}, "
        f"warning_count={parsed['unresolved_line_warning_count']}"
    )


def validate_frozen_inputs(executable: Path, am_root: Path) -> dict[str, Any]:
    if executable.stat().st_size != EXPECTED_NATIVE_BINARY_BYTES:
        raise RuntimeError("native AM executable byte count changed")
    executable_digest = sha256_path(executable)
    if executable_digest != EXPECTED_NATIVE_BINARY_SHA256:
        raise RuntimeError("native AM executable digest changed")
    copied_binary = am_root / "am-12.2/bin/am"
    copied_digest = sha256_path(copied_binary)
    if copied_digest != EXPECTED_COPIED_BINARY_SHA256:
        raise RuntimeError("copied Linux AM executable digest changed")

    profiles = []
    for profile_name in PROFILES:
        profile_record: dict[str, Any] = {"profile": profile_name}
        for role in ("amc", "npz"):
            expected = EXPECTED_INPUTS[profile_name][role]
            path = am_root / expected["path_relative_to_am_root"]
            actual = {
                "path_relative_to_am_root": expected["path_relative_to_am_root"],
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            if actual != expected:
                raise RuntimeError(
                    f"frozen frequency-resolution {role} input changed for "
                    f"{profile_name}: {actual} != {expected}"
                )
            profile_record[role] = actual
        profiles.append(profile_record)
    return {
        "native_executable": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": executable_digest,
        },
        "copied_linux_executable": {
            "path_relative_to_am_root": "am-12.2/bin/am",
            "bytes": copied_binary.stat().st_size,
            "sha256": copied_digest,
        },
        "profiles": profiles,
    }


def case_argv(
    executable: Path,
    am_root: Path,
    profile_name: str,
    zenith_angle_deg: int,
    step_mhz: int,
) -> list[str]:
    profile = am_root / EXPECTED_INPUTS[profile_name]["amc"]["path_relative_to_am_root"]
    return [
        str(executable),
        str(profile),
        str(F_MIN_GHZ),
        "GHz",
        str(F_MAX_GHZ),
        "GHz",
        str(step_mhz),
        "MHz",
        str(zenith_angle_deg),
        "deg",
        "1.0",
    ]


def expected_rows(step_mhz: int) -> int:
    return (F_MAX_GHZ - F_MIN_GHZ) * 1000 // step_mhz + 1


def validate_frequency_grid(
    table: np.ndarray, step_mhz: int, *, label: str
) -> dict[float, tuple[float, float]]:
    frequency = F_MIN_GHZ + np.arange(expected_rows(step_mhz), dtype=np.float64) * (
        float(step_mhz) / 1000.0
    )
    if not np.allclose(table[:, 0], frequency, rtol=0.0, atol=5.0e-12):
        raise RuntimeError(f"frequency-grid mismatch for {label}")
    centers: dict[float, tuple[float, float]] = {}
    for target in FREQUENCIES_GHZ:
        indices = np.flatnonzero(table[:, 0] == target)
        if indices.size != 1:
            raise RuntimeError(f"{target} GHz is not exactly one grid node for {label}")
        index = int(indices[0])
        centers[target] = (float(table[index, 1]), float(table[index, 2]))
    return centers


def run_record(
    *,
    profile_name: str,
    zenith_angle_deg: int,
    step_mhz: int,
    argv: list[str],
    omp_threads: int,
    am_cache_dir: Path,
    cache_dir: Path,
    return_code: int,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    raw_path = raw_output_path(cache_dir, profile_name, zenith_angle_deg, step_mhz)
    record_path = sidecar_path(cache_dir, profile_name, zenith_angle_deg, step_mhz)
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "case_id": case_id(profile_name, zenith_angle_deg, step_mhz),
        "profile": profile_name,
        "zenith_angle_deg": zenith_angle_deg,
        "elevation_deg": 90 - zenith_angle_deg,
        "step_mhz": step_mhz,
        "argv": argv,
        "environment_overrides": {
            "OMP_NUM_THREADS": str(omp_threads),
            "AM_CACHE_PATH": str(am_cache_dir),
        },
        "return_code": return_code,
        "am_identity": parsed["am_identity"],
        "row_count": expected_rows(step_mhz),
        "unresolved_line_warning_count": parsed["unresolved_line_warning_count"],
        "warning_class_counts": parsed["warning_class_counts"],
        "unexpected_warning_count": parsed["unexpected_warning_count"],
        "error_line_count": parsed["error_line_count"],
        "raw_output_path_relative_to_cache": raw_path.relative_to(cache_dir).as_posix(),
        "sidecar_path_relative_to_cache": record_path.relative_to(cache_dir).as_posix(),
        "raw_output_bytes": parsed["raw_output_bytes"],
        "raw_output_sha256": parsed["raw_output_sha256"],
        "normalized_output_sha256": parsed["normalized_output_sha256"],
        "normalized_output_algorithm": NORMALIZED_OUTPUT_ALGORITHM,
        "numeric_output_sha256": parsed["numeric_output_sha256"],
    }


def execute_case(
    *,
    executable: Path,
    am_root: Path,
    profile_name: str,
    zenith_angle_deg: int,
    step_mhz: int,
    cache_dir: Path,
    am_cache_dir: Path,
    omp_threads: int,
) -> dict[str, Any]:
    argv = case_argv(executable, am_root, profile_name, zenith_angle_deg, step_mhz)
    environment = os.environ.copy()
    environment["AM_CACHE_PATH"] = str(am_cache_dir)
    environment["OMP_NUM_THREADS"] = str(omp_threads)
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    parsed = parse_output(
        completed.stdout,
        expected_rows(step_mhz),
        label=case_id(profile_name, zenith_angle_deg, step_mhz),
    )
    validate_return_contract(
        completed.returncode,
        parsed,
        label=case_id(profile_name, zenith_angle_deg, step_mhz),
    )
    record = run_record(
        profile_name=profile_name,
        zenith_angle_deg=zenith_angle_deg,
        step_mhz=step_mhz,
        argv=argv,
        omp_threads=omp_threads,
        am_cache_dir=am_cache_dir,
        cache_dir=cache_dir,
        return_code=completed.returncode,
        parsed=parsed,
    )
    atomic_write(
        raw_output_path(cache_dir, profile_name, zenith_angle_deg, step_mhz),
        completed.stdout,
    )
    atomic_write(
        sidecar_path(cache_dir, profile_name, zenith_angle_deg, step_mhz),
        json_bytes(record),
    )
    return {
        "record": record,
        "centers": validate_frequency_grid(
            parsed["table"], step_mhz, label=record["case_id"]
        ),
    }


def load_cached_case(
    *,
    executable: Path,
    am_root: Path,
    profile_name: str,
    zenith_angle_deg: int,
    step_mhz: int,
    cache_dir: Path,
    am_cache_dir: Path,
    omp_threads: int,
) -> dict[str, Any]:
    raw_path = raw_output_path(cache_dir, profile_name, zenith_angle_deg, step_mhz)
    record_path = sidecar_path(cache_dir, profile_name, zenith_angle_deg, step_mhz)
    if not raw_path.is_file() or not record_path.is_file():
        raise RuntimeError(
            f"missing cached frequency-resolution evidence for "
            f"{case_id(profile_name, zenith_angle_deg, step_mhz)}"
        )
    raw = raw_path.read_bytes()
    parsed = parse_output(
        raw,
        expected_rows(step_mhz),
        label=case_id(profile_name, zenith_angle_deg, step_mhz),
    )
    cached = json.loads(record_path.read_text(encoding="utf-8"))
    return_code = int(cached.get("return_code"))
    validate_return_contract(
        return_code,
        parsed,
        label=case_id(profile_name, zenith_angle_deg, step_mhz),
    )
    expected_record = run_record(
        profile_name=profile_name,
        zenith_angle_deg=zenith_angle_deg,
        step_mhz=step_mhz,
        argv=case_argv(executable, am_root, profile_name, zenith_angle_deg, step_mhz),
        omp_threads=omp_threads,
        am_cache_dir=am_cache_dir,
        cache_dir=cache_dir,
        return_code=return_code,
        parsed=parsed,
    )
    if cached != expected_record:
        raise RuntimeError(f"stale cached run sidecar: {record_path}")
    return {
        "record": expected_record,
        "centers": validate_frequency_grid(
            parsed["table"], step_mhz, label=expected_record["case_id"]
        ),
    }


def copied_centers(
    am_root: Path, profile_name: str, zenith_angle_deg: int
) -> dict[float, tuple[float, float]]:
    npz_path = (
        am_root / EXPECTED_INPUTS[profile_name]["npz"]["path_relative_to_am_root"]
    )
    with np.load(npz_path, allow_pickle=False) as data:
        if data.files != ["el", "atmFreq", "atmTRJ", "atmTtx", "atmTaun"]:
            raise RuntimeError(f"unexpected copied NPZ members: {npz_path}")
        elevation = 90.0 - float(zenith_angle_deg)
        elevation_indices = np.flatnonzero(data["el"] == elevation)
        if elevation_indices.size != 1:
            raise RuntimeError(f"copied NPZ elevation is not unique: {npz_path}")
        elevation_index = int(elevation_indices[0])
        output = {}
        for frequency in FREQUENCIES_GHZ:
            frequency_indices = np.flatnonzero(
                data["atmFreq"][:, elevation_index] == frequency
            )
            if frequency_indices.size != 1:
                raise RuntimeError(
                    f"copied NPZ frequency is not unique: {npz_path}/{frequency}"
                )
            frequency_index = int(frequency_indices[0])
            output[frequency] = (
                float(data["atmTaun"][frequency_index, elevation_index]),
                float(data["atmTtx"][frequency_index, elevation_index]),
            )
    return output


def build_outputs(args: argparse.Namespace) -> dict[str, bytes]:
    executable = args.am_executable.expanduser().resolve(strict=True)
    am_root = args.am_root.expanduser().resolve(strict=True)
    cache_dir = args.cache_dir.expanduser().resolve()
    if is_relative_to(cache_dir, am_root) or is_relative_to(cache_dir, PACKAGE_DIR):
        raise RuntimeError("--cache-dir must be outside AM inputs and the package")
    if args.check:
        if not cache_dir.is_dir():
            raise RuntimeError("--check requires an existing cache directory")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    am_cache_dir = (cache_dir / "am_cache").resolve()
    if not is_relative_to(am_cache_dir, cache_dir):
        raise RuntimeError("AM cache must resolve within --cache-dir")
    if args.check:
        if not am_cache_dir.is_dir():
            raise RuntimeError("--check requires the recorded AM cache directory")
    else:
        am_cache_dir.mkdir(parents=True, exist_ok=True)

    inputs = validate_frozen_inputs(executable, am_root)
    runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for profile_name in PROFILES:
        for zenith_angle in ZENITH_ANGLES_DEG:
            for step_mhz in STEPS_MHZ:
                kwargs = {
                    "executable": executable,
                    "am_root": am_root,
                    "profile_name": profile_name,
                    "zenith_angle_deg": zenith_angle,
                    "step_mhz": step_mhz,
                    "cache_dir": cache_dir,
                    "am_cache_dir": am_cache_dir,
                    "omp_threads": args.omp_threads,
                }
                runs[(profile_name, zenith_angle, step_mhz)] = (
                    load_cached_case(**kwargs) if args.check else execute_case(**kwargs)
                )

    rows: list[dict[str, str]] = []
    for profile_name in PROFILES:
        for zenith_angle in ZENITH_ANGLES_DEG:
            copied = copied_centers(am_root, profile_name, zenith_angle)
            reference = runs[(profile_name, zenith_angle, 1)]
            for step_mhz in STEPS_MHZ:
                run = runs[(profile_name, zenith_angle, step_mhz)]
                record = run["record"]
                for frequency in FREQUENCIES_GHZ:
                    tau, transmission = run["centers"][frequency]
                    tau_reference, transmission_reference = reference["centers"][
                        frequency
                    ]
                    tau_copied, transmission_copied = copied[frequency]
                    correction_difference = abs(math.expm1(tau - tau_reference))
                    copied_correction_difference = abs(math.expm1(tau - tau_copied))
                    exact_copied = ""
                    if step_mhz == 10:
                        exact_copied = str(
                            tau == tau_copied and transmission == transmission_copied
                        ).lower()
                    rows.append(
                        {
                            "profile": profile_name,
                            "zenith_angle_deg": str(zenith_angle),
                            "elevation_deg": str(90 - zenith_angle),
                            "step_mhz": str(step_mhz),
                            "frequency_ghz": f64(frequency),
                            "am_identity": record["am_identity"],
                            "return_code": str(record["return_code"]),
                            "unresolved_line_warning_count": str(
                                record["unresolved_line_warning_count"]
                            ),
                            "unresolved_column_warning_line_count": str(
                                record["warning_class_counts"]["unresolved_column"]
                            ),
                            "unresolved_summary_warning_line_count": str(
                                record["warning_class_counts"]["unresolved_summary"]
                            ),
                            "cache_warning_line_count": str(
                                record["warning_class_counts"]["cache_insert_as_mru"]
                                + record["warning_class_counts"]["cache_promote_to_mru"]
                            ),
                            "cache_insert_as_mru_warning_line_count": str(
                                record["warning_class_counts"]["cache_insert_as_mru"]
                            ),
                            "cache_promote_to_mru_warning_line_count": str(
                                record["warning_class_counts"]["cache_promote_to_mru"]
                            ),
                            "other_warning_line_count": str(
                                record["warning_class_counts"]["other"]
                            ),
                            "error_line_count": str(record["error_line_count"]),
                            "row_count": str(record["row_count"]),
                            "raw_output_sha256": record["raw_output_sha256"],
                            "normalized_output_sha256": record[
                                "normalized_output_sha256"
                            ],
                            "numeric_output_sha256": record["numeric_output_sha256"],
                            "tau_los": f64(tau),
                            "transmission": f64(transmission),
                            "reference_1mhz_tau_los": f64(tau_reference),
                            "reference_1mhz_transmission": f64(transmission_reference),
                            "fractional_correction_difference_vs_1mhz": f64(
                                correction_difference
                            ),
                            "fractional_correction_difference_vs_copied_0to500ghz_10mhz": f64(
                                copied_correction_difference
                            ),
                            "copied_10mhz_tau_los": f64(tau_copied),
                            "copied_10mhz_transmission": f64(transmission_copied),
                            "exact_copied_match_when_10mhz": exact_copied,
                        }
                    )

    max_by_step = {
        step: max(
            float(row["fractional_correction_difference_vs_1mhz"])
            for row in rows
            if int(row["step_mhz"]) == step
        )
        for step in STEPS_MHZ
    }
    ten_mhz_rows = [row for row in rows if row["step_mhz"] == "10"]
    all_10mhz_copied = all(
        row["exact_copied_match_when_10mhz"] == "true" for row in ten_mhz_rows
    )
    max_copied_difference_10mhz = max(
        float(row["fractional_correction_difference_vs_copied_0to500ghz_10mhz"])
        for row in ten_mhz_rows
    )
    threshold = 1.0e-3
    passes = max_by_step[10] <= threshold
    run_records = [
        runs[(profile, zenith, step)]["record"]
        for profile in PROFILES
        for zenith in ZENITH_ANGLES_DEG
        for step in STEPS_MHZ
    ]
    am_identities = sorted({record["am_identity"] for record in run_records})
    if len(am_identities) != 1:
        raise RuntimeError(
            f"frequency-resolution AM identities differ: {am_identities}"
        )
    sidecar_digests = [
        {
            "case_id": record["case_id"],
            "path_relative_to_cache": record["sidecar_path_relative_to_cache"],
            "bytes": len(json_bytes(record)),
            "sha256": sha256_bytes(json_bytes(record)),
        }
        for record in run_records
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_status": "diagnostic_not_operator_authorization",
        "preregistration": "FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md",
        "inputs": inputs,
        "grid": {
            "minimum_ghz": F_MIN_GHZ,
            "maximum_ghz": F_MAX_GHZ,
            "steps_mhz": list(STEPS_MHZ),
            "center_frequencies_ghz": list(FREQUENCIES_GHZ),
        },
        "profiles": list(PROFILES),
        "zenith_angles_deg": list(ZENITH_ANGLES_DEG),
        "h2o_scale": "1.0",
        "execution": {
            "run_count": len(run_records),
            "am_identity": am_identities[0],
            "omp_threads": args.omp_threads,
            "am_cache_path_relative_to_cache": "am_cache",
            "check_mode": "cache_only_no_process_execution_no_directory_creation",
            "known_warning_policy": (
                "return code 1 is accepted only with the complete exact grid "
                "and canonical unresolved-line summary; cache, unknown warning, "
                "and error lines fail"
            ),
        },
        "cache_evidence": {
            "raw_outputs": "stored below --cache-dir/raw_outputs and not committed",
            "execution_sidecars": (
                "stored below --cache-dir/execution_records and not committed"
            ),
            "execution_sidecar_digests": sidecar_digests,
            "execution_sidecar_digest_algorithm": (
                "SHA-256 of deterministic sorted-key UTF-8 JSON with two-space "
                "indentation and one final LF"
            ),
            "normalized_output_algorithm": NORMALIZED_OUTPUT_ALGORITHM,
            "raw_output_digest_algorithm": "SHA-256 of exact combined stdout/stderr bytes",
            "numeric_output_digest_algorithm": (
                "SHA-256 of parsed numeric lines joined by LF with one final LF"
            ),
        },
        "runs": run_records,
        "warning_class_totals": {
            name: sum(record["warning_class_counts"][name] for record in run_records)
            for name in (
                "unresolved_column",
                "unresolved_summary",
                "cache_insert_as_mru",
                "cache_promote_to_mru",
                "other",
            )
        },
        "error_line_total": sum(record["error_line_count"] for record in run_records),
        "diagnostic_threshold_fraction": f64(threshold),
        "maximum_fractional_correction_difference_by_step": {
            str(step): f64(max_by_step[step]) for step in STEPS_MHZ
        },
        "ten_mhz_exactly_matches_copied_centers": all_10mhz_copied,
        "ten_mhz_140to280_vs_copied_0to500_maximum_fractional_correction_difference": f64(
            max_copied_difference_10mhz
        ),
        "ten_mhz_resolution_diagnostic_passes": passes,
        "limitations": [
            "This study brackets two DJF profiles and two elevations; it is not a complete spectral-resolution validation.",
            "A numerical pass does not make warning-status AM output operationally clean.",
            "This study does not establish the missing historical q95 lineage or observational accuracy.",
        ],
    }

    lines = [
        "# SCI-CAL-001 AM 12.2 frequency-resolution diagnostic",
        "",
        "The SHA-256-bound native AM 12.2 build was evaluated on the preregistered 140--280 GHz grids. This is a numerical-resolution diagnostic, not an operator or domain authorization.",
        "",
        "| Grid step | Maximum correction difference from 1 MHz |",
        "| ---: | ---: |",
    ]
    for step in STEPS_MHZ:
        lines.append(f"| {step} MHz | {100.0 * max_by_step[step]:.6f}% |")
    lines.extend(
        [
            "",
            f"The 10-MHz 140--280 GHz center values {'exactly match' if all_10mhz_copied else 'do not exactly match'} the copied 0--500 GHz AM 12.2 products; the maximum range-change correction difference is {100.0 * max_copied_difference_10mhz:.6f}%. The preregistered 0.1% bounded resolution diagnostic {'passes' if passes else 'does not pass'}.",
            "",
            f"All {len(run_records)} raw combined outputs and deterministic execution sidecars remain in the external cache. Their exact and normalized SHA-256 values, AM identity, argv, return status, and all warning classes are retained in the manifest and metrics table. Cache/unknown warnings and error lines are zero; the unresolved-line summary remains explicit.",
            "",
            "A pass does not suppress AM's unresolved-line warning or convert an exit-status-1 run into a clean software-success claim. The copied 10-MHz grid remains immutable lineage evidence.",
            "",
            "This result does not recover the registered legacy q95 artifact, authorize a successor model family, or establish 5--10% absolute flux accuracy or approximately 5% repeatability.",
            "",
        ]
    )
    return {
        OUTPUTS[0]: render_csv(rows),
        OUTPUTS[1]: json_bytes(manifest),
        OUTPUTS[2]: "\n".join(lines).encode("utf-8"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--am-executable", type=Path, required=True)
    parser.add_argument("--am-root", type=Path, default=DEFAULT_AM_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--omp-threads", type=int, default=14)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate cached raw outputs, sidecars, and artifacts without running AM",
    )
    args = parser.parse_args()
    if args.omp_threads < 1:
        parser.error("--omp-threads must be positive")
    generated = build_outputs(args)
    stale = False
    for name, content in generated.items():
        path = PACKAGE_DIR / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                print(f"stale or missing generated artifact: {path}")
                stale = True
        else:
            atomic_write(path, content)
    if stale:
        return 1
    print(
        ("verified" if args.check else "generated")
        + " AM 12.2 frequency-resolution artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
