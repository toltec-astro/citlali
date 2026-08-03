from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_spack_sources.py")
SPEC = importlib.util.spec_from_file_location("prepare_spack_sources", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareSpackSourcesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.commit = "a" * 40
        self.source = MODULE.SourceSpec(
            "tula",
            "https://github.com/toltec-astro/tula.git",
            "v3.x_spack",
            self.commit,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loads_complete_manifest(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repositories": {
                        "tula": {
                            "url": self.source.url,
                            "branch": self.source.branch,
                            "commit": self.commit,
                        }
                    },
                }
            )
        )
        self.assertEqual(MODULE.load_sources(manifest), [self.source])

    def test_accepts_existing_exact_clean_checkout(self) -> None:
        target = self.root / "tula"
        (target / ".git").mkdir(parents=True)

        def runner(command: list[str]) -> str:
            return self.commit if command[-2:] == ["rev-parse", "HEAD"] else ""

        MODULE.prepare_source(self.root, self.source, refresh=False, runner=runner)

    def test_clones_and_detaches_at_manifest_revision(self) -> None:
        target = self.root / "tula"
        actual = "b" * 40
        commands = []

        def runner(command: list[str]) -> str:
            nonlocal actual
            commands.append(command)
            if command[1] == "clone":
                (target / ".git").mkdir(parents=True)
                return ""
            if command[-2:] == ["status", "--porcelain"]:
                return ""
            if command[-2:] == ["rev-parse", "HEAD"]:
                return actual
            if "checkout" in command:
                actual = self.commit
                return ""
            raise AssertionError(command)

        MODULE.prepare_source(self.root, self.source, refresh=False, runner=runner)
        self.assertTrue(any("clone" in command for command in commands))
        self.assertTrue(any("checkout" in command for command in commands))

    def test_rejects_dirty_checkout(self) -> None:
        target = self.root / "tula"
        (target / ".git").mkdir(parents=True)

        def runner(command: list[str]) -> str:
            if command[-2:] == ["status", "--porcelain"]:
                return " M file"
            return self.commit

        with self.assertRaisesRegex(RuntimeError, "dirty"):
            MODULE.prepare_source(self.root, self.source, refresh=True, runner=runner)

    def test_requires_explicit_refresh_for_revision_change(self) -> None:
        target = self.root / "tula"
        (target / ".git").mkdir(parents=True)

        def runner(command: list[str]) -> str:
            return "" if command[-2:] == ["status", "--porcelain"] else "b" * 40

        with self.assertRaisesRegex(RuntimeError, "--refresh"):
            MODULE.prepare_source(self.root, self.source, refresh=False, runner=runner)


if __name__ == "__main__":
    unittest.main()
