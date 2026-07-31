from __future__ import annotations

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
        payload = {
            "roots": [
                {
                    "hash": "a" * 32,
                    "spec": root_spec
                    or "citlali@4.0.0+tests+wiener_openmp "
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
        self._write_lock("citlali@4.0.0+tests %cxx=clang@20.1.8")
        with self.assertRaisesRegex(RuntimeError, "wiener_openmp"):
            validate_concrete_graph(self.environment)

    def test_rejects_wrong_dependency_identity(self) -> None:
        self._write_lock()
        lock_path = self.environment / "spack.lock"
        payload = json.loads(lock_path.read_text())
        payload["concrete_specs"]["1"]["version"] = "2.0.0"
        lock_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "kidscpp identity"):
            validate_concrete_graph(self.environment)


if __name__ == "__main__":
    unittest.main()
