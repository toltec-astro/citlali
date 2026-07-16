#!/usr/bin/env python3
"""Run one controlled reduction and capture portable performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline import audit_reduction_run
from tools.config import compare_lowlevel_yaml


SCHEMA_VERSION = "citlali-performance-run-v1"
SELECTED_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
)
GNU_TIME_FIELDS = {
    "User time (seconds)": ("user_seconds", float),
    "System time (seconds)": ("system_seconds", float),
    "Percent of CPU this job got": ("cpu_percent", lambda value: float(value.rstrip("%"))),
    "Maximum resident set size (kbytes)": ("maximum_resident_set_kb", int),
    "Major (requiring I/O) page faults": ("major_page_faults", int),
    "Minor (reclaiming a frame) page faults": ("minor_page_faults", int),
    "Voluntary context switches": ("voluntary_context_switches", int),
    "Involuntary context switches": ("involuntary_context_switches", int),
    "File system inputs": ("filesystem_inputs", int),
    "File system outputs": ("filesystem_outputs", int),
    "Exit status": ("reported_exit_status", int),
}


class PerformanceCaseError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redu_number(path: Path) -> int | None:
    return audit_reduction_run.redu_number(path)


def reduction_directories(root: Path) -> set[Path]:
    if not root.is_dir():
        return set()
    return {
        child.resolve()
        for child in root.iterdir()
        if child.is_dir() and redu_number(child) is not None
    }


def parse_elapsed_seconds(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise PerformanceCaseError(f"unsupported GNU time elapsed value {value!r}")


def parse_gnu_time(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            result["elapsed_wall_seconds"] = parse_elapsed_seconds(
                line.rsplit(": ", 1)[1]
            )
            continue
        for label, (name, converter) in GNU_TIME_FIELDS.items():
            prefix = f"{label}: "
            if line.startswith(prefix):
                result[name] = converter(line[len(prefix) :])
                break
    return result


def verify_gnu_time(command: Path) -> None:
    completed = subprocess.run(
        [str(command), "--version"], capture_output=True, text=True, check=False
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or "gnu time" not in output.lower():
        raise PerformanceCaseError(f"GNU time not available at {command}")


def git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def executable_identity(path: Path) -> dict[str, Any]:
    executable = path.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PerformanceCaseError(f"Citlali executable is not executable: {executable}")
    completed = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PerformanceCaseError(
            f"Citlali --version failed for {executable}: {completed.stderr.strip()}"
        )
    build_root = executable.parent.parent
    dependency_names = {
        "kidscpp-src": "kids",
        "tula-src": "tula",
    }
    dependencies = {}
    for source_name, identity_name in dependency_names.items():
        matches = sorted(build_root.glob(f"**/_deps/{source_name}"))
        revisions = {revision for match in matches if (revision := git_revision(match))}
        dependencies[identity_name] = next(iter(revisions)) if len(revisions) == 1 else None
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    return {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version_output": output,
        "dependencies": dependencies,
    }


def parse_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "present": False,
            "record_count": 0,
            "size_bytes": 0,
            "stage_totals_seconds": {},
        }
    stage_totals: dict[str, float] = {}
    record_count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("index "):
            continue
        fields = shlex.split(line)
        if len(fields) != 4:
            raise PerformanceCaseError(f"invalid profile row in {path}: {line}")
        stage = fields[1]
        stage_totals[stage] = stage_totals.get(stage, 0.0) + float(fields[3])
        record_count += 1
    return {
        "present": True,
        "record_count": record_count,
        "size_bytes": path.stat().st_size,
        "stage_totals_seconds": dict(sorted(stage_totals.items())),
    }


def versions_from_log(path: Path) -> dict[str, str | None]:
    markers = {
        "citlali": "citlali version:",
        "kids": "kids version:",
        "tula": "tula version:",
    }
    result: dict[str, str | None] = {name: None for name in markers}
    for line in audit_reduction_run.open_text(path):
        for name, marker in markers.items():
            if marker in line:
                result[name] = line.split(marker, 1)[1].strip()
        if all(result.values()):
            break
    return result


def config_leaves(config_path: Path) -> list[dict[str, str]]:
    data = compare_lowlevel_yaml.load_yaml(config_path)
    rows = compare_lowlevel_yaml.row_map(
        compare_lowlevel_yaml.walk_leaves(
            compare_lowlevel_yaml.extract_low_level(data)
        )
    )
    return [
        {"path": path, "value_key": row["value_key"]}
        for path, row in sorted(rows.items())
    ]


def walk_filepaths(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            if key == "filepath" and isinstance(child, str) and child:
                result.append(child)
            result.extend(walk_filepaths(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(walk_filepaths(child))
        return result
    return []


def input_identities(config_path: Path, hash_max_bytes: int) -> list[dict[str, Any]]:
    data = compare_lowlevel_yaml.extract_low_level(
        compare_lowlevel_yaml.load_yaml(config_path)
    )
    identities = []
    for raw_path in sorted(set(walk_filepaths(data))):
        path = Path(raw_path)
        record: dict[str, Any] = {
            "path": raw_path,
            "basename": path.name,
            "exists": path.is_file(),
            "size_bytes": None,
            "sha256": None,
        }
        if path.is_file():
            record["size_bytes"] = path.stat().st_size
            if path.stat().st_size <= hash_max_bytes:
                record["sha256"] = sha256_file(path)
        identities.append(record)
    return identities


def runtime_signature(reduction: Path) -> dict[str, Any] | None:
    path = reduction / "runtime_provenance.yaml"
    data = audit_reduction_run.load_yaml(path) if path.is_file() else None
    if not isinstance(data, dict):
        return None
    requested = data.get("requested") or {}
    return {
        "requested": {
            name: requested.get(name)
            for name in (
                "verbose",
                "interp_over_gaps",
                "n_threads",
                "parallel_policy",
                "reduction_type",
                "use_subdir",
            )
        },
        "effective_threads": data.get("effective", {}).get("threads"),
        "realized_threads": data.get("realized", {}).get("threads"),
        "realized_parallel_policy": data.get("realized", {}).get(
            "parallel_policy"
        ),
        "realized_reduction_type": data.get("realized", {}).get(
            "reduction_type"
        ),
    }


def reduction_evidence(reduction: Path, hash_max_bytes: int) -> dict[str, Any]:
    config_path = audit_reduction_run.find_config(reduction)
    log_path = audit_reduction_run.find_log(reduction)
    log = audit_reduction_run.audit_log(log_path)
    profile_path = reduction / "citlali_profile.ecsv"
    return {
        "path": str(reduction),
        "name": reduction.name,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_leaves": config_leaves(config_path),
        "inputs": input_identities(config_path, hash_max_bytes),
        "log_path": str(log_path),
        "versions": versions_from_log(log_path),
        "citlali_total_log_seconds": log.get("interval_seconds", {}).get(
            "total_log"
        ),
        "log_issue_counts": log.get("issue_counts", {}),
        "runtime_signature": runtime_signature(reduction),
        "profile": parse_profile(profile_path),
    }


def affinity_cpu_count() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    return len(getter(0)) if getter is not None else None


def host_evidence() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "affinity_cpu_count": affinity_cpu_count(),
        "environment": {
            key: os.environ[key] for key in SELECTED_ENVIRONMENT if key in os.environ
        },
    }


def storage_evidence(root: Path) -> dict[str, Any]:
    stat = root.stat()
    filesystem = os.statvfs(root)
    return {
        "reduced_root": str(root),
        "device": stat.st_dev,
        "filesystem_block_size": filesystem.f_bsize,
        "filesystem_fragment_size": filesystem.f_frsize,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--phase", choices=("warmup", "measured"), required=True)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument(
        "--build-type",
        choices=("Release", "RelWithDebInfo", "Debug"),
        required=True,
    )
    parser.add_argument("--reduced-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-new-reductions", type=int, default=1)
    parser.add_argument("--gnu-time", type=Path, default=Path("/usr/bin/time"))
    parser.add_argument("--citlali-executable", type=Path, required=True)
    parser.add_argument("--hash-input-max-bytes", type=int, default=100_000_000)
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command after --, e.g. tolteca reduce."
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.pair_index < 0:
        parser.error("--pair-index must be nonnegative")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    time_report = output.with_suffix(".time.txt")
    if output.exists() or time_report.exists():
        print(
            f"performance output already exists: {output} or {time_report}",
            file=sys.stderr,
        )
        return 2
    reduced_root = args.reduced_root.expanduser().resolve()
    if not reduced_root.is_dir():
        print(f"reduced root does not exist: {reduced_root}", file=sys.stderr)
        return 2
    try:
        verify_gnu_time(args.gnu_time)
        executable = executable_identity(args.citlali_executable)
    except (OSError, PerformanceCaseError) as error:
        print(f"performance case setup failed: {error}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    before = reduction_directories(reduced_root)
    started_utc = utc_now()
    started = time.monotonic()
    command = [
        str(args.gnu_time),
        "-v",
        "-o",
        str(time_report),
        *args.command,
    ]
    completed = subprocess.run(command, check=False)
    wrapper_wall_seconds = time.monotonic() - started
    ended_utc = utc_now()
    after = reduction_directories(reduced_root)
    new_reductions = sorted(after - before, key=lambda path: redu_number(path) or -1)
    time_text = time_report.read_text(encoding="utf-8") if time_report.is_file() else ""
    try:
        gnu_time = parse_gnu_time(time_text)
    except (IndexError, ValueError, PerformanceCaseError) as error:
        gnu_time = {"parse_error": str(error)}

    structure_ok = len(new_reductions) == args.expected_new_reductions
    reduction = new_reductions[-1] if structure_ok and new_reductions else None
    evidence = None
    evidence_error = None
    if reduction is not None:
        try:
            evidence = reduction_evidence(reduction, args.hash_input_max_bytes)
        except Exception as error:
            evidence_error = str(error)
    measurement_ok = all(
        name in gnu_time
        for name in ("elapsed_wall_seconds", "maximum_resident_set_kb")
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": args.campaign_id,
        "case_id": args.case_id,
        "role": args.role,
        "phase": args.phase,
        "pair_index": args.pair_index,
        "build_type": args.build_type,
        "command": args.command,
        "executable": executable,
        "cwd": str(Path.cwd()),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "command_exit_code": completed.returncode,
        "wrapper_wall_seconds": wrapper_wall_seconds,
        "host": host_evidence(),
        "storage": storage_evidence(reduced_root),
        "gnu_time": gnu_time,
        "new_reductions": [str(path) for path in new_reductions],
        "expected_new_reductions": args.expected_new_reductions,
        "structure_ok": structure_ok,
        "measurement_ok": measurement_ok,
        "reduction": evidence,
        "evidence_error": evidence_error,
        "artifacts": {
            "metadata": str(output),
            "gnu_time_report": str(time_report),
            "attached_metadata": None,
            "attached_gnu_time_report": None,
        },
    }
    write_json(output, record)
    if reduction is not None:
        attached_metadata = reduction / "performance_run.json"
        attached_time = reduction / "performance_time.txt"
        if attached_metadata.exists() or attached_time.exists():
            print(
                f"performance attachment already exists in {reduction}",
                file=sys.stderr,
            )
            return 2
        attached_time.write_text(time_text, encoding="utf-8")
        record["artifacts"]["attached_metadata"] = str(attached_metadata)
        record["artifacts"]["attached_gnu_time_report"] = str(attached_time)
        write_json(output, record)
        write_json(attached_metadata, record)

    print(
        f"performance case {args.case_id}: exit={completed.returncode} "
        f"new_reductions={len(new_reductions)} measurement_ok={measurement_ok}"
    )
    if completed.returncode != 0:
        return completed.returncode if completed.returncode > 0 else 1
    if not structure_ok or evidence is None:
        return 2
    if not measurement_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
