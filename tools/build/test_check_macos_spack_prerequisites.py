from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("check_macos_spack_prerequisites.py")
SPEC = importlib.util.spec_from_file_location("macos_spack_prerequisites", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MacosSpackPrerequisiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.workspace = self.root / "workspace"
        self.citlali = self.workspace / "citlali-refactor-build"
        self.citlali.mkdir(parents=True)
        (self.citlali / "CMakeLists.txt").touch()
        for repository, marker in MODULE.REPOSITORY_MARKERS.items():
            marker_path = self.workspace / repository / marker
            marker_path.parent.mkdir(parents=True)
            marker_path.touch()
        self.llvm_prefix = self.root / "llvm@20"
        (self.llvm_prefix / "bin").mkdir(parents=True)
        (self.llvm_prefix / "bin/clang++").touch()
        self.spack = self.root / "spack/bin/spack"
        self.spack.parent.mkdir(parents=True)
        self.spack.touch()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _runner(self, command: list[str]) -> str:
        if command[-2:] == ["--prefix", "llvm@20"]:
            return str(self.llvm_prefix)
        if command[0].endswith("clang++"):
            return "Homebrew clang version 20.1.8"
        if command[0].endswith("cmake"):
            return "cmake version 4.3.0"
        if command[0].endswith("ninja"):
            return "1.13.2"
        if command[0] == str(self.spack):
            return "1.2.2"
        raise AssertionError(command)

    def _inspect(self, **overrides):
        arguments = {
            "workspace_root": self.workspace,
            "citlali_source": self.citlali,
            "spack_executable": self.spack,
            "runner": self._runner,
            "system_name": "Darwin",
            "machine": "arm64",
            "environment": {},
        }
        arguments.update(overrides)
        with patch.object(MODULE.shutil, "which") as which:
            which.side_effect = lambda name: f"/mock/bin/{name}"
            return MODULE.inspect_prerequisites(**arguments)

    def test_accepts_complete_native_workspace(self) -> None:
        results = self._inspect()
        self.assertFalse([result for result in results if result.status == "fail"])

    def test_rejects_apple_clang(self) -> None:
        def runner(command: list[str]) -> str:
            if command[0].endswith("clang++"):
                return "Apple clang version 20.0.0"
            return self._runner(command)

        results = self._inspect(runner=runner)
        compiler = next(result for result in results if result.name == "compiler")
        self.assertEqual(compiler.status, "fail")

    def test_rejects_wrong_spack_version(self) -> None:
        def runner(command: list[str]) -> str:
            if command[0] == str(self.spack):
                return "1.1.0"
            return self._runner(command)

        results = self._inspect(runner=runner)
        spack = next(result for result in results if result.name == "spack")
        self.assertEqual(spack.status, "fail")

    def test_rejects_global_unversioned_libomp_flags(self) -> None:
        results = self._inspect(
            environment={"LDFLAGS": "-L/opt/homebrew/opt/libomp/lib"}
        )
        openmp = next(
            result for result in results if result.name == "openmp_environment"
        )
        self.assertEqual(openmp.status, "fail")

    def test_reports_missing_sibling_repository(self) -> None:
        os.remove(
            self.workspace / "kidscpp" / MODULE.REPOSITORY_MARKERS["kidscpp"]
        )
        results = self._inspect()
        kidscpp = next(
            result for result in results if result.name == "repository.kidscpp"
        )
        self.assertEqual(kidscpp.status, "fail")


if __name__ == "__main__":
    unittest.main()
