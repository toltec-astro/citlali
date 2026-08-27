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

    def test_accepts_longer_matching_abbreviation(self) -> None:
        reported = require_matching_source_revision(
            "v4.0.0-3642-g34b83df51484-dirty (2026-08-14T11:16:05)",
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


if __name__ == "__main__":
    unittest.main()
