"""Supported host profiles for the native Citlali Spack build lane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildProfile:
    """Host-specific inputs that must not leak into the package graph."""

    name: str
    environment_directory: str
    c_compiler: Path
    cxx_compiler: Path
    root_compiler_term: str
    provenance_compiler: str
    cmake_generator: str
    required_graph_packages: tuple[str, ...] = ()
    cmake_platform_arguments: tuple[str, ...] = ()


PROFILES = {
    "macos-llvm20": BuildProfile(
        name="macos-llvm20",
        environment_directory="citlali-macos-llvm20",
        c_compiler=Path("/opt/homebrew/opt/llvm@20/bin/clang"),
        cxx_compiler=Path("/opt/homebrew/opt/llvm@20/bin/clang++"),
        root_compiler_term="%cxx=clang@20.1.8",
        provenance_compiler="compiler=Clang-20.1.8",
        cmake_generator="Ninja",
        required_graph_packages=("llvm-openmp",),
        cmake_platform_arguments=("-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",),
    ),
    "unity-gcc13": BuildProfile(
        name="unity-gcc13",
        environment_directory="citlali-unity-gcc13",
        c_compiler=Path("/usr/bin/gcc"),
        cxx_compiler=Path("/usr/bin/g++"),
        root_compiler_term="%cxx=gcc@13.3.0",
        provenance_compiler="compiler=GNU-13.3.0",
        cmake_generator="Unix Makefiles",
    ),
}


def get_profile(name: str) -> BuildProfile:
    """Return a supported build profile by its stable command-line name."""
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown build profile {name!r}") from error
