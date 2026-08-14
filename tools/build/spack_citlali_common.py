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
    r"^v\S*?-g([0-9a-f]{7,40})(?:-dirty)?(?:\s|$)", re.MULTILINE
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


def run(command: Sequence[str], *, environment: dict[str, str]) -> str:
    """Run a command and relay its combined output."""
    print("+", " ".join(command), flush=True)
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
