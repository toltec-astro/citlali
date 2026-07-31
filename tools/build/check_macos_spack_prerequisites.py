#!/usr/bin/env python3
"""Validate the native macOS prerequisites for the Spack build lane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


EXPECTED_LLVM_MAJOR = 20
EXPECTED_SPACK_VERSION = (1, 2, 2)
MINIMUM_SPACK_PYTHON = (3, 9)
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
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _version_tuple(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups() if value is not None)


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="pass", detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="fail", detail=detail)


def _info(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="info", detail=detail)


def inspect_prerequisites(
    *,
    workspace_root: Path,
    citlali_source: Path,
    spack_executable: Path | None,
    spack_python: Path | None,
    runner: CommandRunner = _run,
    system_name: str | None = None,
    machine: str | None = None,
    environment: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Return all prerequisite results without stopping at the first failure."""

    results: list[CheckResult] = []
    system_name = system_name or platform.system()
    machine = machine or platform.machine()
    environment = environment if environment is not None else dict(os.environ)

    if system_name == "Darwin" and machine == "arm64":
        results.append(_pass("platform", "Darwin arm64"))
    else:
        results.append(
            _fail(
                "platform",
                f"expected Darwin arm64, found {system_name} {machine}",
            )
        )

    brew = shutil.which("brew")
    if brew is None:
        results.append(_fail("homebrew", "brew is not on PATH"))
        llvm_prefix = None
    else:
        try:
            llvm_prefix = Path(runner([brew, "--prefix", "llvm@20"]))
            results.append(_pass("homebrew", f"llvm@20 prefix={llvm_prefix}"))
        except (OSError, subprocess.CalledProcessError) as error:
            llvm_prefix = None
            results.append(_fail("homebrew", f"cannot resolve llvm@20: {error}"))

    clangxx = llvm_prefix / "bin/clang++" if llvm_prefix else None
    if clangxx is None or not clangxx.is_file():
        results.append(_fail("compiler", "Homebrew llvm@20 clang++ is missing"))
    else:
        try:
            compiler_text = runner([str(clangxx), "--version"])
            compiler_version = _version_tuple(compiler_text)
            is_apple_clang = "Apple clang" in compiler_text
            if (
                compiler_version is not None
                and compiler_version[0] == EXPECTED_LLVM_MAJOR
                and not is_apple_clang
            ):
                results.append(
                    _pass(
                        "compiler",
                        f"{clangxx} reports LLVM {compiler_version[0]}",
                    )
                )
            else:
                results.append(
                    _fail(
                        "compiler",
                        "expected non-Apple LLVM 20, got "
                        + compiler_text.splitlines()[0],
                    )
                )
        except (OSError, subprocess.CalledProcessError) as error:
            results.append(_fail("compiler", f"cannot execute {clangxx}: {error}"))

    for tool, minimum in (("cmake", (3, 25)), ("ninja", (1, 10))):
        executable = shutil.which(tool)
        if executable is None:
            results.append(_fail(tool, f"{tool} is not on PATH"))
            continue
        try:
            output = runner([executable, "--version"])
            version = _version_tuple(output)
            if version is not None and version >= minimum:
                results.append(_pass(tool, f"{executable} version={version}"))
            else:
                results.append(
                    _fail(tool, f"requires >= {minimum}, found {version}")
                )
        except (OSError, subprocess.CalledProcessError) as error:
            results.append(_fail(tool, f"cannot execute {executable}: {error}"))

    if spack_python is None:
        results.append(
            _fail(
                "spack_python",
                "set SPACK_PYTHON or pass --spack-python",
            )
        )
    elif not spack_python.is_file():
        results.append(
            _fail("spack_python", f"missing interpreter {spack_python}")
        )
    else:
        try:
            output = runner([str(spack_python), "--version"])
            version = _version_tuple(output)
            if version is not None and version >= MINIMUM_SPACK_PYTHON:
                results.append(
                    _pass(
                        "spack_python",
                        f"{spack_python} version={version}",
                    )
                )
            else:
                results.append(
                    _fail(
                        "spack_python",
                        f"requires >= {MINIMUM_SPACK_PYTHON}, found {version}",
                    )
                )
        except (OSError, subprocess.CalledProcessError) as error:
            results.append(
                _fail("spack_python", f"cannot execute {spack_python}: {error}")
            )

    if spack_executable is None:
        results.append(
            _fail(
                "spack",
                "Spack is missing; pass --spack or set SPACK_ROOT",
            )
        )
    elif not spack_executable.is_file():
        results.append(_fail("spack", f"missing executable {spack_executable}"))
    else:
        try:
            output = runner([str(spack_executable), "--version"])
            version = _version_tuple(output)
            if version == EXPECTED_SPACK_VERSION:
                results.append(_pass("spack", f"{spack_executable} version={version}"))
            else:
                results.append(
                    _fail(
                        "spack",
                        f"requires {EXPECTED_SPACK_VERSION}, found {version}",
                    )
                )
        except (OSError, subprocess.CalledProcessError) as error:
            results.append(_fail("spack", f"cannot execute Spack: {error}"))

    for repository, marker in REPOSITORY_MARKERS.items():
        path = workspace_root / repository
        expected = path / marker
        if expected.is_file():
            results.append(_pass(f"repository.{repository}", str(path)))
        else:
            results.append(
                _fail(
                    f"repository.{repository}",
                    f"expected {expected}",
                )
            )

    if (citlali_source / "CMakeLists.txt").is_file():
        results.append(_pass("repository.citlali", str(citlali_source)))
    else:
        results.append(
            _fail(
                "repository.citlali",
                f"expected {citlali_source / 'CMakeLists.txt'}",
            )
        )

    contaminated = []
    for variable in ("CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS"):
        value = environment.get(variable, "")
        if "/opt/homebrew/opt/libomp" in value:
            contaminated.append(variable)
    if contaminated:
        results.append(
            _fail(
                "openmp_environment",
                "unversioned Homebrew libomp is forced through "
                + ", ".join(contaminated),
            )
        )
    else:
        results.append(
            _info(
                "openmp_environment",
                "no global Homebrew libomp flags; Spack must provide the "
                "LLVM-20-compatible runtime",
            )
        )

    return results


def _resolve_spack(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument.expanduser().resolve()
    spack_root = os.environ.get("SPACK_ROOT")
    if spack_root:
        return (Path(spack_root).expanduser() / "bin/spack").resolve()
    executable = shutil.which("spack")
    return Path(executable).resolve() if executable else None


def _resolve_spack_python(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument.expanduser().resolve()
    configured = os.environ.get("SPACK_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path.home() / "tolteca/bin/python"
    return candidate.resolve() if candidate.is_file() else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=source_root.parent,
        help="Directory containing sibling tula_cmake, tula, and kidscpp checkouts",
    )
    parser.add_argument(
        "--citlali-source",
        type=Path,
        default=source_root,
        help="Citlali build-adaptation source checkout",
    )
    parser.add_argument("--spack", type=Path, help="Path to Spack executable")
    parser.add_argument(
        "--spack-python",
        type=Path,
        help="Python interpreter used by the Spack launcher",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = inspect_prerequisites(
        workspace_root=args.workspace_root.expanduser().resolve(),
        citlali_source=args.citlali_source.expanduser().resolve(),
        spack_executable=_resolve_spack(args.spack),
        spack_python=_resolve_spack_python(args.spack_python),
    )
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            print(f"{result.status.upper():4} {result.name}: {result.detail}")
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
