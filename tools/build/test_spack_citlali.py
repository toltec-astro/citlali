#!/usr/bin/env python3
"""Validate the installed full Citlali package and its independent consumer."""

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
)
from spack_citlali_profiles import PROFILES, get_profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--consumer-build-dir",
        type=Path,
        default=source_root / "build/spack-citlali-consumer",
    )
    parser.add_argument(
        "--developer-build-dir",
        type=Path,
        default=source_root / "build/spack-citlali-dev",
    )
    parser.add_argument(
        "--skip-developer-ctest",
        action="store_true",
        help="skip the complete compiled test suite (packaging diagnostics only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = get_profile(args.profile)
    source_root = Path(__file__).resolve().parents[2]
    spack = args.spack.expanduser().resolve()
    spack_python = args.spack_python.expanduser().resolve()
    environment_path = (
        args.environment.expanduser().resolve()
        if args.environment is not None
        else source_root / "spack/environments" / profile.environment_directory
    )
    consumer_build_dir = args.consumer_build_dir.expanduser().resolve()
    developer_build_dir = args.developer_build_dir.expanduser().resolve()
    consumer_source = source_root / "spack/consumers/citlali_installed"

    for required in (
        spack,
        spack_python,
        profile.cxx_compiler,
        consumer_source / "CMakeLists.txt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    root_hash, root_spec = validate_concrete_graph(
        environment_path,
        root_compiler_term=profile.root_compiler_term,
        required_graph_packages=profile.required_graph_packages,
    )
    print(f"Spack root: {root_spec}")
    print(f"Spack DAG hash: {root_hash}")
    environment = process_environment(spack_python)
    source_revision = run(
        ["git", "-C", str(source_root), "rev-parse", "--short=9", "HEAD"],
        environment=environment,
    ).splitlines()[-1]
    source_status = run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        environment=environment,
    )
    if source_status:
        raise RuntimeError("installed-artifact acceptance requires a clean source tree")
    prefix_output = run(
        [str(spack), "-e", str(environment_path), "location", "-i", "citlali"],
        environment=environment,
    )
    package_prefix = Path(prefix_output.splitlines()[-1])
    executable = package_prefix / "bin/citlali"
    if not executable.is_file():
        raise FileNotFoundError(executable)

    version_output = run([str(executable), "--version"], environment=environment)
    for required_text in (
        "v4.0.0-",
        source_revision,
        "kids 3.1.0",
        root_hash,
        profile.provenance_compiler,
        "cxx=23",
    ):
        if required_text not in version_output:
            raise RuntimeError(
                f"installed CLI version output is missing {required_text!r}"
            )
    if "-dirty" in version_output:
        raise RuntimeError("installed CLI was built from a dirty source tree")
    help_output = run([str(executable), "--help"], environment=environment)
    for required_text in (
        "--dump_config",
        "--grppiex",
        "Multiple config file are",
        "merged in order.",
    ):
        if required_text not in help_output:
            raise RuntimeError(
                f"installed CLI help is missing {required_text!r}"
            )

    configure = [
        "cmake",
        "--fresh",
        "-S",
        str(consumer_source),
        "-B",
        str(consumer_build_dir),
        f"-DCMAKE_CXX_COMPILER={profile.cxx_compiler}",
        *[
            argument.format(deployment_target=deployment_target())
            for argument in profile.cmake_platform_arguments
        ],
        f"-Dcitlali_DIR={package_prefix / 'lib/cmake/citlali'}",
    ]
    run(
        spack_build_env_command(spack, environment_path, configure),
        environment=environment,
    )
    run(
        ["cmake", "--build", str(consumer_build_dir), "-j", "8"],
        environment=environment,
    )
    run(
        [
            "ctest",
            "--test-dir",
            str(consumer_build_dir),
            "--output-on-failure",
        ],
        environment=environment,
    )

    if not args.skip_developer_ctest:
        if not (developer_build_dir / "CTestTestfile.cmake").is_file():
            raise FileNotFoundError(
                f"missing configured test tree {developer_build_dir}; "
                "run run_spack_citlali_dev.py all first"
            )
        run(
            spack_build_env_command(
                spack,
                environment_path,
                [
                    "ctest",
                    "--test-dir",
                    str(developer_build_dir),
                    "--output-on-failure",
                ],
            ),
            environment=environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
