#!/usr/bin/env python3
"""Configure, build, or test Citlali in a persistent native Spack build tree."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from spack_citlali_common import (
    deployment_target,
    process_environment,
    run,
    spack_build_env_command,
    validate_concrete_graph,
    validate_first_party_sources,
)
from spack_citlali_profiles import PROFILES, get_profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("configure", "build", "test", "all"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="macos-llvm20",
    )
    parser.add_argument(
        "--spack", type=Path, default=Path.home() / "GitHub/spack/bin/spack"
    )
    parser.add_argument(
        "--spack-python",
        type=Path,
        default=Path.home() / "tolteca/bin/python",
    )
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--source", type=Path, default=source_root)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=source_root / "build/spack-citlali-dev",
    )
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard the CMake cache during configuration",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = get_profile(args.profile)
    spack = args.spack.expanduser().resolve()
    spack_python = args.spack_python.expanduser().resolve()
    source_root = args.source.expanduser().resolve()
    environment_path = (
        args.environment.expanduser().resolve()
        if args.environment is not None
        else source_root / "spack/environments" / profile.environment_directory
    )
    build_dir = args.build_dir.expanduser().resolve()

    for required in (
        spack,
        spack_python,
        profile.c_compiler,
        profile.cxx_compiler,
        source_root / "cmake/spack/CMakeLists.txt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    validate_first_party_sources(source_root)
    root_hash, root_spec = validate_concrete_graph(
        environment_path,
        root_compiler_term=profile.root_compiler_term,
        required_graph_packages=profile.required_graph_packages,
    )
    print(f"Spack root: {root_spec}")
    print(f"Spack DAG hash: {root_hash}")
    environment = process_environment(spack_python)

    if args.action in {"configure", "all"}:
        configure = [
            "cmake",
            *(["--fresh"] if args.fresh else []),
            "-G",
            profile.cmake_generator,
            "-S",
            str(source_root / "cmake/spack"),
            "-B",
            str(build_dir),
            f"-DCMAKE_C_COMPILER={profile.c_compiler}",
            f"-DCMAKE_CXX_COMPILER={profile.cxx_compiler}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            *[
                argument.format(deployment_target=deployment_target())
                for argument in profile.cmake_platform_arguments
            ],
            "-DCITLALI_BUILD_CLI=ON",
            "-DCITLALI_BUILD_TESTS=ON",
            "-DCITLALI_ENABLE_OPENMP=ON",
            "-DCITLALI_USE_WIENER_FILTER_OMP=ON",
            f"-DCITLALI_SPACK_DAG_HASH={root_hash}",
        ]
        run(
            spack_build_env_command(spack, environment_path, configure),
            environment=environment,
        )

    if args.action in {"build", "all"}:
        run(
            spack_build_env_command(
                spack,
                environment_path,
                ["cmake", "--build", str(build_dir), "-j", str(args.jobs)],
            ),
            environment=environment,
        )

    if args.action in {"test", "all"}:
        run(
            spack_build_env_command(
                spack,
                environment_path,
                [
                    "ctest",
                    "--test-dir",
                    str(build_dir),
                    "--output-on-failure",
                ],
            ),
            environment=environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
