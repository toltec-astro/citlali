#!/usr/bin/env python3
"""Test installed Kidscpp consumers in its exact Spack build environment."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_concrete_graph(environment_path: Path) -> None:
    lock_path = environment_path / "spack.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"missing {lock_path}; concretize the Kidscpp environment first"
        )

    lock = json.loads(lock_path.read_text())
    roots = lock.get("roots", [])
    if len(roots) != 1:
        raise RuntimeError(f"expected one concrete root, found {len(roots)}")
    root_spec = roots[0].get("spec", "")
    if "kidscpp@3.1.0" not in root_spec or "%cxx=clang@20.1.8" not in root_spec:
        raise RuntimeError(f"unexpected Kidscpp root: {root_spec}")

    packages = {
        spec.get("name"): spec for spec in lock.get("concrete_specs", {}).values()
    }
    expected = {
        "kidscpp": ("3.1.0", "toltec.kidscpp"),
        "tula-perflibs": ("0.1.0", "toltec.citlali"),
        "llvm-openmp": ("20.1.8", "builtin"),
    }
    for name, (version, namespace) in expected.items():
        spec = packages.get(name)
        if spec is None:
            raise RuntimeError(f"concrete graph is missing {name}")
        actual = (str(spec.get("version")), spec.get("namespace"))
        if actual != (version, namespace):
            raise RuntimeError(
                f"unexpected {name} identity: {actual}; "
                f"expected {(version, namespace)}"
            )


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
        default=source_root / "spack/environments/kidscpp-macos-llvm20",
    )
    parser.add_argument(
        "--kidscpp-source",
        type=Path,
        default=source_root.parent / "kidscpp",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=source_root / "build/spack-kidscpp-consumers",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="raw TolTEC timestream used by the real-reader acceptance test",
    )
    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help="fail unless --fixture is supplied",
    )
    return parser.parse_args(argv)


def _configure_build_test(
    *,
    spack: Path,
    environment_path: Path,
    package_prefix: Path,
    source: Path,
    build: Path,
    process_environment: dict[str, str],
    cmake_args: Sequence[str] = (),
) -> None:
    command = [
        str(spack),
        "-e",
        str(environment_path),
        "build-env",
        "kidscpp",
        "--",
        "cmake",
        "--fresh",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={_deployment_target()}",
        f"-Dkidscpp_DIR={package_prefix / 'lib/cmake/kidscpp'}",
        *cmake_args,
    ]
    _run(command, environment=process_environment)
    _run(
        ["cmake", "--build", str(build), "-j", "8"],
        environment=process_environment,
    )
    _run(
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
        environment=process_environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(__file__).resolve().parents[2]
    spack = args.spack.expanduser().resolve()
    spack_python = args.spack_python.expanduser().resolve()
    environment_path = args.environment.expanduser().resolve()
    kidscpp_source = args.kidscpp_source.expanduser().resolve()
    build_root = args.build_root.expanduser().resolve()
    llvm_cxx = Path("/opt/homebrew/opt/llvm@20/bin/clang++")

    required = (
        spack,
        spack_python,
        llvm_cxx,
        kidscpp_source / "tests/installed_consumer/CMakeLists.txt",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.require_real_data and args.fixture is None:
        raise ValueError("--require-real-data requires --fixture")

    process_environment = dict(os.environ)
    process_environment["SPACK_PYTHON"] = str(spack_python)
    _assert_concrete_graph(environment_path)
    prefix_output = _run(
        [str(spack), "-e", str(environment_path), "location", "-i", "kidscpp"],
        environment=process_environment,
    )
    package_prefix = Path(prefix_output.splitlines()[-1])

    _configure_build_test(
        spack=spack,
        environment_path=environment_path,
        package_prefix=package_prefix,
        source=kidscpp_source / "tests/installed_consumer",
        build=build_root / "api",
        process_environment=process_environment,
    )

    if args.fixture is None:
        print("real-data test: NOT RUN (supply --fixture)")
        return 0

    fixture = args.fixture.expanduser().resolve()
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    print(f"real-data fixture: {fixture}")
    print(f"real-data fixture sha256: {_sha256(fixture)}")
    _configure_build_test(
        spack=spack,
        environment_path=environment_path,
        package_prefix=package_prefix,
        source=source_root / "spack/consumers/kidscpp_real_reader",
        build=build_root / "real-reader",
        process_environment=process_environment,
        cmake_args=(f"-DKIDS_REAL_FIXTURE={fixture}",),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
