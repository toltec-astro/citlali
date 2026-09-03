"""Shared checks and command helpers for the native Citlali Spack lane."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Sequence

from verify_spack_source_revisions import (
    inspect_revisions,
    load_revisions,
    require_accepted_revisions,
)


EXPECTED_PACKAGES = {
    "citlali": ("4.0.0", "toltec.citlali"),
    "kidscpp": ("3.1.0", "toltec.kidscpp"),
    "tula": ("3.1.0", "toltec.tula"),
    "tula-ccfits": ("1.0.0", "toltec.tula_cmake"),
    "tula-netcdf-cxx4": ("4.3.1", "toltec.tula_cmake"),
    "tula-perflibs": ("0.1.0", "toltec.tula_cmake"),
    "cfitsio": ("4.3.0", "builtin"),
    "hdf5": ("1.14.6", "builtin"),
}

_VERSION_SOURCE_REVISION = re.compile(
    r"^(?:(?:\S+)-\d+-g)?([0-9a-f]{7,40})(?:-dirty)?(?:\s|$)",
    re.MULTILINE,
)

_SPACK_COMPILER_ENVIRONMENT_KEYS = (
    "CC",
    "CXX",
    "SPACK_CC",
    "SPACK_CXX",
    "SPACK_COMPILER_WRAPPER_PATH",
    "SPACK_TARGET_ARGS_CC",
    "SPACK_TARGET_ARGS_CXX",
)


def require_matching_source_revision(
    version_output: str, source_revision: str
) -> str:
    """Require the CLI's Git abbreviation to identify the source commit."""
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source revision must be a full lowercase Git SHA-1")

    revisions = _VERSION_SOURCE_REVISION.findall(version_output)
    if len(revisions) != 1:
        raise RuntimeError(
            "installed CLI version output must contain exactly one Git revision"
        )

    reported_revision = revisions[0]
    if not source_revision.startswith(reported_revision):
        raise RuntimeError(
            "installed CLI source revision does not match the source tree: "
            f"reported={reported_revision} expected={source_revision}"
        )
    return reported_revision


def validate_first_party_sources(source_root: Path) -> None:
    """Require the exact clean dependency sources declared by Citlali."""
    revisions = load_revisions(source_root / "spack/upstream-revisions.json")
    results = inspect_revisions(source_root / "build/spack-sources", revisions)
    require_accepted_revisions(results)


def run(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    relay_output: bool = True,
) -> str:
    """Run a command and optionally relay its combined output."""
    suffix = "" if relay_output else " [output captured]"
    print("+", " ".join(command) + suffix, flush=True)
    try:
        completed = subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="")
        raise
    if relay_output:
        print(completed.stdout, end="")
    return completed.stdout.strip()


def deployment_target() -> str:
    """Return a stable major-version macOS deployment target."""
    version = platform.mac_ver()[0]
    if not version:
        raise RuntimeError("cannot determine the macOS deployment target")
    return f"{version.split('.', maxsplit=1)[0]}.0"


def process_environment(spack_python: Path) -> dict[str, str]:
    """Return the process environment with the supported Spack launcher."""
    environment = dict(os.environ)
    environment["SPACK_PYTHON"] = str(spack_python)
    return environment


def validate_spack_compiler_environment(
    output: str,
    *,
    expected_c_compiler: Path,
    expected_cxx_compiler: Path,
) -> dict[str, str]:
    """Validate the concrete root's compiler-wrapper environment."""
    values = {}
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator and name in _SPACK_COMPILER_ENVIRONMENT_KEYS:
            values[name] = value

    missing = [
        name for name in _SPACK_COMPILER_ENVIRONMENT_KEYS if not values.get(name)
    ]
    if missing:
        raise RuntimeError(
            f"Spack build environment is missing compiler controls: {missing}"
        )

    expected_compilers = {
        "SPACK_CC": expected_c_compiler,
        "SPACK_CXX": expected_cxx_compiler,
    }
    for name, expected in expected_compilers.items():
        actual = Path(values[name])
        if actual.resolve() != expected.resolve():
            raise RuntimeError(
                f"{name} does not identify the profile compiler: "
                f"actual={actual} expected={expected}"
            )

    wrapper_directories = [
        Path(item).resolve()
        for item in values["SPACK_COMPILER_WRAPPER_PATH"].split(os.pathsep)
        if item
    ]
    for name in ("CC", "CXX"):
        wrapper = Path(values[name]).resolve()
        if not any(
            wrapper == directory or directory in wrapper.parents
            for directory in wrapper_directories
        ):
            raise RuntimeError(
                f"{name} is not inside SPACK_COMPILER_WRAPPER_PATH: {values[name]}"
            )

    c_arguments = values["SPACK_TARGET_ARGS_CC"]
    cxx_arguments = values["SPACK_TARGET_ARGS_CXX"]
    if c_arguments != cxx_arguments:
        raise RuntimeError(
            "Spack C and CXX target arguments differ: "
            f"CC={c_arguments!r} CXX={cxx_arguments!r}"
        )

    return values


def inspect_spack_compiler_environment(
    spack: Path,
    environment_path: Path,
    *,
    environment: dict[str, str],
    expected_c_compiler: Path,
    expected_cxx_compiler: Path,
) -> dict[str, str]:
    """Inspect and report the compiler controls for the concrete Citlali root."""
    output = run(
        spack_build_env_command(spack, environment_path, ["/usr/bin/env"]),
        environment=environment,
        relay_output=False,
    )
    values = validate_spack_compiler_environment(
        output,
        expected_c_compiler=expected_c_compiler,
        expected_cxx_compiler=expected_cxx_compiler,
    )
    print(f"Spack C wrapper: {values['CC']}")
    print(f"Spack CXX wrapper: {values['CXX']}")
    print(f"Spack target arguments: {values['SPACK_TARGET_ARGS_CXX']}")
    return values


def require_spack_compiler_cache(
    build_dir: Path,
    compiler_environment: dict[str, str],
    *,
    allow_missing: bool,
) -> None:
    """Require a configured CMake tree to retain Spack wrapper compilers."""
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        if allow_missing:
            return
        raise FileNotFoundError(cache_path)

    cache_values = {}
    for line in cache_path.read_text(errors="replace").splitlines():
        name_and_type, separator, value = line.partition("=")
        if not separator:
            continue
        name = name_and_type.partition(":")[0]
        if name in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"):
            cache_values[name] = value

    expected = {
        "CMAKE_C_COMPILER": compiler_environment["CC"],
        "CMAKE_CXX_COMPILER": compiler_environment["CXX"],
    }
    for name, wrapper in expected.items():
        actual = cache_values.get(name)
        if actual is None:
            raise RuntimeError(f"CMake cache is missing {name}")
        if Path(actual).resolve() != Path(wrapper).resolve():
            raise RuntimeError(
                f"CMake cache bypasses the Spack compiler wrapper for {name}: "
                f"actual={actual} expected={wrapper}; reconfigure with --fresh"
            )


def managed_deployment_environment(
    environment: dict[str, str],
    environment_path: Path,
    *,
    profile_name: str,
    expected_root_hash: str,
) -> dict[str, str]:
    """Bind runtime deployment labels to the concrete Citlali root."""
    lock_path = environment_path / "spack.lock"
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    roots = lock.get("roots", [])
    if len(roots) != 1 or roots[0].get("hash") != expected_root_hash:
        raise RuntimeError(
            "deployment environment does not contain the accepted Citlali root"
        )
    result = dict(environment)
    result.update(
        {
            "TOLTECA_CPP_ENV": str(environment_path),
            "TOLTECA_SPACK_PROFILE": profile_name,
            "TOLTECA_SPACK_LOCK_SHA256": hashlib.sha256(lock_bytes).hexdigest(),
        }
    )
    return result


def validate_concrete_graph(
    environment_path: Path,
    *,
    root_compiler_term: str = "%cxx=clang@20.1.8",
    required_graph_packages: Sequence[str] = ("llvm-openmp",),
) -> tuple[str, str]:
    """Validate the accepted full-app graph and return root hash and spec."""
    lock_path = environment_path / "spack.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"missing {lock_path}; concretize the Citlali environment first"
        )

    lock = json.loads(lock_path.read_text())
    roots = lock.get("roots", [])
    if len(roots) != 1:
        raise RuntimeError(f"expected one concrete root, found {len(roots)}")

    root_hash = roots[0].get("hash", "")
    root_spec = roots[0].get("spec", "")
    required_root_terms = (
        "citlali@4.0.0",
        "+openmp",
        "+tests",
        "+wiener_openmp",
        root_compiler_term,
    )
    missing = [term for term in required_root_terms if term not in root_spec]
    if not root_hash or missing:
        raise RuntimeError(
            f"unexpected Citlali root {root_spec!r}; missing {missing}"
        )

    packages = {
        spec.get("name"): spec for spec in lock.get("concrete_specs", {}).values()
    }
    for name, (version, namespace) in EXPECTED_PACKAGES.items():
        spec = packages.get(name)
        if spec is None:
            raise RuntimeError(f"concrete graph is missing {name}")
        actual = (str(spec.get("version")), spec.get("namespace"))
        if actual != (version, namespace):
            raise RuntimeError(
                f"unexpected {name} identity: {actual}; "
                f"expected {(version, namespace)}"
            )
    for name in required_graph_packages:
        if name not in packages:
            raise RuntimeError(f"concrete graph is missing {name}")
    return root_hash, root_spec


def spack_build_env_command(
    spack: Path, environment_path: Path, command: Sequence[str]
) -> list[str]:
    """Wrap a command in the exact Citlali dependency environment."""
    return [
        str(spack),
        "-e",
        str(environment_path),
        "build-env",
        "citlali",
        "--",
        *command,
    ]
