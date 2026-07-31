"""Shared checks and command helpers for the native Citlali Spack lane."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Sequence


EXPECTED_PACKAGES = {
    "citlali": ("4.0.0", "toltec.citlali"),
    "kidscpp": ("3.1.0", "toltec.kidscpp"),
    "tula": ("3.1.0", "toltec.tula"),
    "tula-perflibs": ("0.1.0", "toltec.citlali"),
    "llvm-openmp": ("20.1.8", "builtin"),
    "cfitsio": ("4.3.0", "builtin"),
    "hdf5": ("1.14.6", "builtin"),
}


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


def validate_concrete_graph(environment_path: Path) -> tuple[str, str]:
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
        "+tests",
        "+wiener_openmp",
        "%cxx=clang@20.1.8",
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
