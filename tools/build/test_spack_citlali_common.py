from __future__ import annotations

import hashlib
import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("spack_citlali_common.py")
SPEC = importlib.util.spec_from_file_location("spack_citlali_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EXPECTED_PACKAGES = MODULE.EXPECTED_PACKAGES
validate_concrete_graph = MODULE.validate_concrete_graph
managed_deployment_environment = MODULE.managed_deployment_environment
require_matching_source_revision = MODULE.require_matching_source_revision
require_spack_compiler_cache = MODULE.require_spack_compiler_cache
validate_spack_compiler_environment = MODULE.validate_spack_compiler_environment


class SpackCitlaliGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.environment = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_lock(self, root_spec: str | None = None) -> None:
        concrete_specs = {}
        for index, (name, (version, namespace)) in enumerate(
            EXPECTED_PACKAGES.items()
        ):
            concrete_specs[str(index)] = {
                "name": name,
                "version": version,
                "namespace": namespace,
            }
        concrete_specs["macos-openmp"] = {
            "name": "llvm-openmp",
            "version": "20.1.8",
            "namespace": "builtin",
        }
        payload = {
            "roots": [
                {
                    "hash": "a" * 32,
                    "spec": root_spec
                    or "citlali@4.0.0+openmp+tests+wiener_openmp "
                    "%cxx=clang@20.1.8",
                }
            ],
            "concrete_specs": concrete_specs,
        }
        (self.environment / "spack.lock").write_text(json.dumps(payload))

    def test_accepts_expected_graph(self) -> None:
        self._write_lock()
        root_hash, _ = validate_concrete_graph(self.environment)
        self.assertEqual(root_hash, "a" * 32)

    def test_rejects_missing_performance_variant(self) -> None:
        self._write_lock("citlali@4.0.0+openmp+tests %cxx=clang@20.1.8")
        with self.assertRaisesRegex(RuntimeError, "wiener_openmp"):
            validate_concrete_graph(self.environment)

    def test_rejects_missing_pipeline_openmp_variant(self) -> None:
        self._write_lock(
            "citlali@4.0.0+tests+wiener_openmp %cxx=clang@20.1.8"
        )
        with self.assertRaisesRegex(RuntimeError, "openmp"):
            validate_concrete_graph(self.environment)

    def test_rejects_wrong_dependency_identity(self) -> None:
        self._write_lock()
        lock_path = self.environment / "spack.lock"
        payload = json.loads(lock_path.read_text())
        payload["concrete_specs"]["1"]["version"] = "2.0.0"
        lock_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "kidscpp identity"):
            validate_concrete_graph(self.environment)

    def test_accepts_unity_graph_without_llvm_openmp_package(self) -> None:
        self._write_lock(
            "citlali@4.0.0+openmp+tests+wiener_openmp %cxx=gcc@13.3.0"
        )
        lock_path = self.environment / "spack.lock"
        payload = json.loads(lock_path.read_text())
        payload["concrete_specs"] = {
            key: value
            for key, value in payload["concrete_specs"].items()
            if value["name"] != "llvm-openmp"
        }
        lock_path.write_text(json.dumps(payload))

        root_hash, _ = validate_concrete_graph(
            self.environment,
            root_compiler_term="%cxx=gcc@13.3.0",
            required_graph_packages=(),
        )
        self.assertEqual(root_hash, "a" * 32)

    def test_managed_environment_binds_profile_lock_and_root(self) -> None:
        self._write_lock()
        result = managed_deployment_environment(
            {"EXISTING": "value"},
            self.environment,
            profile_name="macos-llvm20",
            expected_root_hash="a" * 32,
        )
        lock_bytes = (self.environment / "spack.lock").read_bytes()
        self.assertEqual(result["EXISTING"], "value")
        self.assertEqual(result["TOLTECA_CPP_ENV"], str(self.environment))
        self.assertEqual(result["TOLTECA_SPACK_PROFILE"], "macos-llvm20")
        self.assertEqual(
            result["TOLTECA_SPACK_LOCK_SHA256"],
            hashlib.sha256(lock_bytes).hexdigest(),
        )

    def test_managed_environment_rejects_wrong_root(self) -> None:
        self._write_lock()
        with self.assertRaisesRegex(RuntimeError, "accepted Citlali root"):
            managed_deployment_environment(
                {},
                self.environment,
                profile_name="macos-llvm20",
                expected_root_hash="b" * 32,
            )


class SpackCitlaliSourceRevisionTest(unittest.TestCase):
    source_revision = "34b83df514847695e8c17648eb7b66e75e97b7d3"

    def test_accepts_dynamic_git_abbreviation(self) -> None:
        reported = require_matching_source_revision(
            "v4.0.0-3642-g34b83df5 (2026-08-14T11:16:05)\n"
            "kids 3.1.0 (spack-package)\n",
            self.source_revision,
        )
        self.assertEqual(reported, "34b83df5")

    def test_accepts_non_version_git_describe_tag(self) -> None:
        reported = require_matching_source_revision(
            "wp7-timestream-integration-20260826-35-g34b83df5 "
            "(2026-08-28T14:01:44)\n"
            "kids 3.1.0 (spack-package)\n",
            self.source_revision,
        )
        self.assertEqual(reported, "34b83df5")

    def test_accepts_longer_matching_abbreviation(self) -> None:
        reported = require_matching_source_revision(
            "v4.0.0-3642-g34b83df51484-dirty (2026-08-14T11:16:05)",
            self.source_revision,
        )
        self.assertEqual(reported, "34b83df51484")

    def test_accepts_tagless_matching_abbreviation(self) -> None:
        reported = require_matching_source_revision(
            "34b83df5 (2026-08-14T11:16:05)\n"
            "kids 3.1.0 (spack-package)\n",
            self.source_revision,
        )
        self.assertEqual(reported, "34b83df5")

    def test_accepts_tagless_longer_dirty_abbreviation(self) -> None:
        reported = require_matching_source_revision(
            "34b83df51484-dirty (2026-08-14T11:16:05)",
            self.source_revision,
        )
        self.assertEqual(reported, "34b83df51484")

    def test_rejects_mismatched_abbreviation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            require_matching_source_revision(
                "v4.0.0-3642-gdeadbee (2026-08-14T11:16:05)",
                self.source_revision,
            )

    def test_rejects_missing_revision(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one Git revision"):
            require_matching_source_revision(
                "v4.0.0 (2026-08-14T11:16:05)", self.source_revision
            )

    def test_rejects_multiple_revision_lines(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one Git revision"):
            require_matching_source_revision(
                "v4.0.0-3642-g34b83df5 (2026-08-14T11:16:05)\n"
                "34b83df5 (duplicate)\n",
                self.source_revision,
            )


class SpackCompilerEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.wrapper_root = self.root / "compiler-wrapper"
        self.c_wrapper = self.wrapper_root / "cc"
        self.cxx_wrapper = self.wrapper_root / "gcc/g++"
        self.c_compiler = self.root / "compiler/gcc"
        self.cxx_compiler = self.root / "compiler/g++"
        for path in (
            self.c_wrapper,
            self.cxx_wrapper,
            self.c_compiler,
            self.cxx_compiler,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _output(self, **replacements: str) -> str:
        values = {
            "CC": str(self.c_wrapper),
            "CXX": str(self.cxx_wrapper),
            "SPACK_CC": str(self.c_compiler),
            "SPACK_CXX": str(self.cxx_compiler),
            "SPACK_COMPILER_WRAPPER_PATH": str(self.wrapper_root),
            "SPACK_TARGET_ARGS_CC": "-march=cascadelake -mtune=cascadelake",
            "SPACK_TARGET_ARGS_CXX": "-march=cascadelake -mtune=cascadelake",
        }
        values.update(replacements)
        return "\n".join(f"{name}={value}" for name, value in values.items())

    def _validate(self, output: str) -> dict[str, str]:
        return validate_spack_compiler_environment(
            output,
            expected_c_compiler=self.c_compiler,
            expected_cxx_compiler=self.cxx_compiler,
        )

    def test_accepts_wrappers_and_matching_concrete_target(self) -> None:
        values = self._validate(self._output())
        self.assertEqual(
            values["SPACK_TARGET_ARGS_CXX"],
            "-march=cascadelake -mtune=cascadelake",
        )

    def test_rejects_direct_compiler_bypass(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not inside"):
            self._validate(self._output(CC=str(self.c_compiler)))

    def test_rejects_mismatched_target_arguments(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "target arguments differ"):
            self._validate(self._output(SPACK_TARGET_ARGS_CXX="-march=x86-64"))

    def test_rejects_wrong_profile_compiler(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "profile compiler"):
            self._validate(self._output(SPACK_CXX="/usr/bin/false"))

    def test_accepts_wrapper_backed_cmake_cache(self) -> None:
        build_dir = self.root / "build"
        build_dir.mkdir()
        (build_dir / "CMakeCache.txt").write_text(
            "CMAKE_C_COMPILER:FILEPATH=" + str(self.c_wrapper) + "\n"
            "CMAKE_CXX_COMPILER:FILEPATH=" + str(self.cxx_wrapper) + "\n"
        )
        require_spack_compiler_cache(
            build_dir,
            self._validate(self._output()),
            allow_missing=False,
        )

    def test_rejects_direct_compiler_cmake_cache(self) -> None:
        build_dir = self.root / "build"
        build_dir.mkdir()
        (build_dir / "CMakeCache.txt").write_text(
            "CMAKE_C_COMPILER:FILEPATH=" + str(self.c_compiler) + "\n"
            "CMAKE_CXX_COMPILER:FILEPATH=" + str(self.cxx_compiler) + "\n"
        )
        with self.assertRaisesRegex(RuntimeError, "bypasses"):
            require_spack_compiler_cache(
                build_dir,
                self._validate(self._output()),
                allow_missing=False,
            )

    def test_missing_cmake_cache_requires_explicit_allowance(self) -> None:
        build_dir = self.root / "build"
        require_spack_compiler_cache(
            build_dir,
            self._validate(self._output()),
            allow_missing=True,
        )
        with self.assertRaises(FileNotFoundError):
            require_spack_compiler_cache(
                build_dir,
                self._validate(self._output()),
                allow_missing=False,
            )


if __name__ == "__main__":
    unittest.main()
