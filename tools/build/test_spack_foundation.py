#!/usr/bin/env python3
"""Build and run the independent consumer of the installed Tula foundation."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path
from typing import Sequence


def _run(command: Sequence[str], *, environment: dict[str, str]) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    return completed.stdout.strip()


def _deployment_target() -> str:
    version = platform.mac_ver()[0]
    if not version:
        raise RuntimeError("cannot determine the macOS deployment target")
    return f"{version.split('.', maxsplit=1)[0]}.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--environment",
        type=Path,
        default=source_root / "spack/environments/foundation-macos-llvm20",
    )
    parser.add_argument(
        "--tula-source",
        type=Path,
        default=source_root.parent / "tula",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=source_root / "build/spack-foundation-consumer",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spack = args.spack.expanduser().resolve()
    spack_python = args.spack_python.expanduser().resolve()
    environment_path = args.environment.expanduser().resolve()
    consumer_source = args.tula_source.expanduser().resolve() / "tests/installed_consumer"
    build_dir = args.build_dir.expanduser().resolve()
    llvm_cxx = Path("/opt/homebrew/opt/llvm@20/bin/clang++")

    for required in (spack, spack_python, llvm_cxx, consumer_source / "CMakeLists.txt"):
        if not required.exists():
            raise FileNotFoundError(required)

    environment = dict(os.environ)
    environment["SPACK_PYTHON"] = str(spack_python)
    prefix_output = _run(
        [str(spack), "-e", str(environment_path), "location", "-i", "tula"],
        environment=environment,
    )
    tula_prefix = Path(prefix_output.splitlines()[-1])

    _run(
        [
            str(spack),
            "-e",
            str(environment_path),
            "build-env",
            "tula",
            "--",
            "cmake",
            "--fresh",
            "-S",
            str(consumer_source),
            "-B",
            str(build_dir),
            f"-DCMAKE_CXX_COMPILER={llvm_cxx}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={_deployment_target()}",
            f"-Dtula_DIR={tula_prefix / 'lib/cmake/tula'}",
        ],
        environment=environment,
    )
    _run(["cmake", "--build", str(build_dir), "-j", "8"], environment=environment)
    _run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        environment=environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
