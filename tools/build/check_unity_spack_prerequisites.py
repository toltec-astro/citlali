#!/usr/bin/env python3
"""Validate prerequisites for the user-owned Unity Spack build lane."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


EXPECTED_SPACK_VERSION = (1, 2, 2)
EXPECTED_GCC_VERSION = (13, 3, 0)
MINIMUM_PYTHON = (3, 9)
MINIMUM_CMAKE = (3, 25)
MINIMUM_MAKE = (4, 0)
REPOSITORY_MARKERS = {
    "tula_cmake": Path("spack_repo/toltec/tula_cmake/repo.yaml"),
    "tula": Path("spack_repo/toltec/tula/repo.yaml"),
    "kidscpp": Path("spack_repo/toltec/kidscpp/repo.yaml"),
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


CommandRunner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", text)
    return (
        tuple(int(value) for value in match.groups() if value is not None)
        if match
        else None
    )


def inspect_prerequisites(
    *,
    workspace_root: Path,
    citlali_source: Path,
    spack: Path,
    python: Path,
    runner: CommandRunner = _run,
    system_name: str | None = None,
    machine: str | None = None,
) -> list[CheckResult]:
    """Inspect all requirements without stopping at the first failure."""
    results: list[CheckResult] = []

    def check_version(
        name: str,
        executable: Path,
        expected: tuple[int, ...],
        *,
        minimum: bool = False,
    ) -> None:
        if not executable.is_file():
            results.append(CheckResult(name, "fail", f"missing {executable}"))
            return
        try:
            actual = _version(runner([str(executable), "--version"]))
            accepted = actual is not None and (
                actual >= expected if minimum else actual == expected
            )
            relation = ">=" if minimum else "=="
            status = "pass" if accepted else "fail"
            detail = (
                f"{executable} version={actual}; "
                f"requires {relation}{expected}"
            )
            results.append(CheckResult(name, status, detail))
        except (OSError, subprocess.CalledProcessError) as error:
            results.append(CheckResult(name, "fail", str(error)))

    host = (system_name or platform.system(), machine or platform.machine())
    results.append(
        CheckResult(
            "platform",
            "pass" if host == ("Linux", "x86_64") else "fail",
            f"{host[0]} {host[1]}",
        )
    )
    check_version("gcc", Path("/usr/bin/gcc"), EXPECTED_GCC_VERSION)
    check_version("g++", Path("/usr/bin/g++"), EXPECTED_GCC_VERSION)
    check_version("gfortran", Path("/usr/bin/gfortran"), EXPECTED_GCC_VERSION)
    check_version("cmake", Path(shutil.which("cmake") or ""), MINIMUM_CMAKE, minimum=True)
    check_version("make", Path(shutil.which("make") or ""), MINIMUM_MAKE, minimum=True)
    check_version("python", python, MINIMUM_PYTHON, minimum=True)
    check_version("spack", spack, EXPECTED_SPACK_VERSION)

    for name, marker in REPOSITORY_MARKERS.items():
        path = workspace_root / name / marker
        results.append(
            CheckResult(
                f"repository.{name}",
                "pass" if path.is_file() else "fail",
                str(path),
            )
        )
    citlali_marker = citlali_source / "spack/spack_repo/toltec/citlali/repo.yaml"
    results.append(
        CheckResult(
            "repository.citlali",
            "pass" if citlali_marker.is_file() else "fail",
            str(citlali_marker),
        )
    )

    free_gib = shutil.disk_usage(workspace_root).free / (1024**3)
    results.append(
        CheckResult(
            "disk",
            "pass" if free_gib >= 20 else "fail",
            f"{free_gib:.1f} GiB free; requires at least 20 GiB",
        )
    )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=source_root.parent)
    parser.add_argument("--citlali-source", type=Path, default=source_root)
    parser.add_argument(
        "--spack",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--spack-python",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = inspect_prerequisites(
        workspace_root=args.workspace_root.expanduser().resolve(),
        citlali_source=args.citlali_source.expanduser().resolve(),
        spack=args.spack.expanduser().resolve(),
        python=args.spack_python.expanduser().resolve(),
    )
    for result in results:
        print(f"{result.status.upper():4} {result.name}: {result.detail}")
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
