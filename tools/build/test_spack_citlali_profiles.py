from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("spack_citlali_profiles.py")
SPEC = importlib.util.spec_from_file_location("spack_citlali_profiles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SpackCitlaliProfileTest(unittest.TestCase):
    def test_macos_profile_requires_exact_llvm_openmp(self) -> None:
        profile = MODULE.get_profile("macos-llvm20")
        self.assertEqual(profile.cmake_generator, "Ninja")
        self.assertEqual(profile.required_graph_packages, ("llvm-openmp",))
        self.assertEqual(profile.provenance_compiler, "compiler=Clang-20.1.8")

    def test_unity_profile_uses_measured_system_gcc(self) -> None:
        profile = MODULE.get_profile("unity-gcc13")
        self.assertEqual(profile.cmake_generator, "Unix Makefiles")
        self.assertEqual(profile.cxx_compiler, Path("/usr/bin/g++"))
        self.assertEqual(profile.root_compiler_term, "%cxx=gcc@13.3.0")
        self.assertEqual(profile.required_graph_packages, ())

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown build profile"):
            MODULE.get_profile("not-a-profile")


if __name__ == "__main__":
    unittest.main()
