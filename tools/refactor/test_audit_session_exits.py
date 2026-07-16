from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_session_exits.py")
SPEC = importlib.util.spec_from_file_location("audit_session_exits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_session_exits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_session_exits)


class SessionExitAuditTest(unittest.TestCase):
    def make_repo(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repo = Path(temporary.name)
        (repo / "include/citlali/core/session").mkdir(parents=True)
        (repo / "include/citlali/core/detail").mkdir(parents=True)
        (repo / "include/citlali/core/session/entry.h").write_text(
            "#include <citlali/core/detail/worker.h>\n", encoding="utf-8"
        )
        (repo / "include/citlali/core/detail/worker.h").write_text(
            "inline void stop() { std::exit(EXIT_FAILURE); }\n",
            encoding="utf-8",
        )
        return repo

    def test_follows_transitive_project_headers(self) -> None:
        repo = self.make_repo()

        result = audit_session_exits.audit(
            repo, ["include/citlali/core/session/entry.h"]
        )

        self.assertEqual(result["dependency_file_count"], 2)
        self.assertEqual(result["library_exit_count"], 1)
        self.assertEqual(
            result["library_exit_counts_by_file"],
            {"include/citlali/core/detail/worker.h": 1},
        )

    def test_includes_core_library_sources(self) -> None:
        repo = self.make_repo()
        source = repo / "src/citlali/core/engine/worker.cpp"
        source.parent.mkdir(parents=True)
        source.write_text(
            "void stop_source() { exit(EXIT_FAILURE); }\n",
            encoding="utf-8",
        )

        result = audit_session_exits.audit(
            repo, ["include/citlali/core/session/entry.h"]
        )

        self.assertEqual(result["dependency_file_count"], 3)
        self.assertEqual(result["library_exit_count"], 2)
        self.assertEqual(
            result["library_exit_counts_by_file"][
                "src/citlali/core/engine/worker.cpp"
            ],
            1,
        )

    def test_baseline_rejects_growth_and_allows_reduction(self) -> None:
        repo = self.make_repo()
        result = audit_session_exits.audit(
            repo, ["include/citlali/core/session/entry.h"]
        )
        baseline = {
            "schema_version": audit_session_exits.SCHEMA_VERSION,
            "library_exit_counts_by_file": {
                "include/citlali/core/detail/worker.h": 0
            },
        }
        self.assertEqual(len(audit_session_exits.baseline_growth(result, baseline)), 1)

        baseline["library_exit_counts_by_file"] = {
            "include/citlali/core/detail/worker.h": 2
        }
        self.assertEqual(audit_session_exits.baseline_growth(result, baseline), [])


if __name__ == "__main__":
    unittest.main()
