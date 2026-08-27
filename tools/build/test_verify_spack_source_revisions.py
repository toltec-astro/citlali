from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_spack_source_revisions.py")
SPEC = importlib.util.spec_from_file_location("verify_spack_source_revisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SpackSourceRevisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.commit = "a" * 40
        self.manifest = self.root / "revisions.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repositories": {"tula": {"commit": self.commit}},
                }
            )
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_accepts_exact_clean_revision(self) -> None:
        revisions = MODULE.load_revisions(self.manifest)

        def runner(command: list[str]) -> str:
            return self.commit if command[-2:] == ["rev-parse", "HEAD"] else ""

        results = MODULE.inspect_revisions(self.root, revisions, runner=runner)
        MODULE.require_accepted_revisions(results)

    def test_rejects_revision_drift(self) -> None:
        result = MODULE.RevisionResult("tula", self.commit, "b" * 40, True)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            MODULE.require_accepted_revisions([result])

    def test_rejects_dirty_checkout(self) -> None:
        result = MODULE.RevisionResult("tula", self.commit, self.commit, False)
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            MODULE.require_accepted_revisions([result])

    def test_rejects_short_commit(self) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repositories": {"tula": {"commit": "abc"}},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "full commit"):
            MODULE.load_revisions(self.manifest)


if __name__ == "__main__":
    unittest.main()
