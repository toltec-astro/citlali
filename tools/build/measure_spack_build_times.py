#!/usr/bin/env python3
"""Measure clean, no-op, and representative Citlali incremental builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from run_spack_citlali_dev import build_command, configure_command
from spack_citlali_common import (
    process_environment,
    spack_build_env_command,
    validate_concrete_graph,
    validate_first_party_sources,
)
from spack_citlali_profiles import PROFILES, get_profile


SCHEMA_VERSION = "citlali-build-timing-v1"
DEFAULT_INCREMENTAL_INPUT = Path("src/citlali/cli/main.cpp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def source_identity(source_root: Path) -> dict:
    """Return source revision and worktree state used by a timing campaign."""
    return {
        "revision": _capture(["git", "rev-parse", "HEAD"], cwd=source_root),
        "status": _capture(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
        ).splitlines(),
    }


def resolve_incremental_inputs(
    source_root: Path, inputs: Sequence[Path]
) -> list[Path]:
    """Resolve incremental inputs and require them to be source-tree files."""
    source_root = source_root.resolve()
    result = []
    for value in inputs:
        path = (value if value.is_absolute() else source_root / value).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"incremental input is outside the source tree: {path}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(path)
    return result


@contextmanager
def timestamp_touch(path: Path) -> Iterator[dict]:
    """Force one dependency rebuild while preserving source bytes and times."""
    original_stat = path.stat()
    original_sha256 = _sha256(path)
    touched_mtime_ns = max(time.time_ns(), original_stat.st_mtime_ns + 1_000_000_000)
    os.utime(path, ns=(original_stat.st_atime_ns, touched_mtime_ns))
    try:
        yield {
            "path": str(path),
            "sha256": original_sha256,
            "original_mtime_ns": original_stat.st_mtime_ns,
            "touched_mtime_ns": touched_mtime_ns,
        }
    finally:
        os.utime(
            path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        if _sha256(path) != original_sha256:
            raise RuntimeError(f"incremental timing changed source bytes: {path}")


def timed_command(
    *,
    name: str,
    command: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
) -> dict:
    """Run one timing stage and return its machine-readable result."""
    print(f"[{name}] {' '.join(command)}", flush=True)
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    with log_path.open("w") as stream:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    result = {
        "name": name,
        "command": list(command),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "log": log_path.name,
    }
    print(
        f"[{name}] returncode={completed.returncode} elapsed_s={elapsed:.3f}",
        flush=True,
    )
    if completed.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        print("\n".join(tail))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="macos-llvm20",
    )
    parser.add_argument(
        "--spack",
        type=Path,
        default=Path.home() / "GitHub/spack/bin/spack",
    )
    parser.add_argument(
        "--spack-python",
        type=Path,
        default=Path.home() / "tolteca/bin/python",
    )
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--source", type=Path, default=source_root)
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument(
        "--incremental-input",
        type=Path,
        action="append",
        help=(
            "source-relative input to touch for an incremental build; may be "
            "repeated (default: src/citlali/cli/main.cpp)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=source_root / "build/build-timing-results",
    )
    parser.add_argument(
        "--keep-build-dir",
        action="store_true",
        help="retain the otherwise disposable clean build tree",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record rather than reject a dirty source tree",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    profile = get_profile(args.profile)
    source_root = args.source.expanduser().resolve()
    spack = args.spack.expanduser().resolve()
    spack_python = args.spack_python.expanduser().resolve()
    environment_path = (
        args.environment.expanduser().resolve()
        if args.environment is not None
        else source_root / "spack/environments" / profile.environment_directory
    )
    output_root = args.output_root.expanduser().resolve()
    incremental_inputs = args.incremental_input or [DEFAULT_INCREMENTAL_INPUT]
    incremental_paths = resolve_incremental_inputs(
        source_root,
        incremental_inputs,
    )

    for required in (
        spack,
        spack_python,
        profile.c_compiler,
        profile.cxx_compiler,
        source_root / "cmake/spack/CMakeLists.txt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    identity = source_identity(source_root)
    if identity["status"] and not args.allow_dirty:
        raise RuntimeError(
            "source checkout is dirty; commit changes or pass --allow-dirty"
        )

    validate_first_party_sources(source_root)
    root_hash, root_spec = validate_concrete_graph(
        environment_path,
        root_compiler_term=profile.root_compiler_term,
        required_graph_packages=profile.required_graph_packages,
    )
    environment = process_environment(spack_python)

    campaign_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_dir = output_root / f"{campaign_id}-{profile.name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    work_parent = source_root / "build"
    work_parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix="spack-citlali-timing-", dir=work_parent)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "profile": profile.name,
        "source": identity,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "build": {
            "jobs": args.jobs,
            "generator": profile.cmake_generator,
            "build_type": "Release",
            "build_dir": str(build_dir),
            "spack_environment": str(environment_path),
            "spack_root_hash": root_hash,
            "spack_root_spec": root_spec,
        },
        "incremental_inputs": [str(path.relative_to(source_root)) for path in incremental_paths],
        "stages": [],
    }
    manifest_path = output_dir / "build-timing.json"

    def record(result: dict) -> None:
        manifest["stages"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run_stage(
        *,
        name: str,
        command: Sequence[str],
        metadata: dict | None = None,
    ) -> dict:
        result = timed_command(
            name=name,
            command=command,
            environment=environment,
            log_path=output_dir / f"{name}.log",
        )
        if metadata:
            result.update(metadata)
        record(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"timing stage failed: {name}")
        return result

    try:
        configure = configure_command(
            source_root=source_root,
            build_dir=build_dir,
            profile=profile,
            root_hash=root_hash,
            fresh=True,
        )
        run_stage(
            name="clean_configure",
            command=spack_build_env_command(spack, environment_path, configure),
        )
        build = spack_build_env_command(
            spack,
            environment_path,
            build_command(build_dir=build_dir, jobs=args.jobs),
        )
        run_stage(name="clean_build", command=build)
        run_stage(name="no_op_build", command=build)
        for index, incremental_path in enumerate(incremental_paths, start=1):
            with timestamp_touch(incremental_path) as touch:
                run_stage(
                    name=f"incremental_build_{index}",
                    command=build,
                    metadata={"input": touch},
                )
    finally:
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["build_dir_retained"] = args.keep_build_dir
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not args.keep_build_dir:
            shutil.rmtree(build_dir, ignore_errors=True)

    print(f"timing manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
