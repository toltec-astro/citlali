from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from tools.config import audit_polarimetry_boundary as audit


class PolarimetryBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_manifest(self) -> None:
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_SCHEMA,
                "path_count": len(audit.EXPECTED_PATHS),
                "path_sha256": audit.EXPECTED_PATH_SHA256,
                "paths": audit.EXPECTED_PATHS,
            }
        )
        self.assertTrue(state["exact"])

    def test_rejects_unsorted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "manifest.json"
            source.write_text('{"paths":["z","a"]}')
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                audit.load_manifest(source)

    def test_extracts_declared_tuple_paths(self) -> None:
        source = '''
        auto a = std::tuple{"timestream", "polarimetry", "enabled"};
        auto b = std::tuple{"timestream", "polarimetry", "grouping"};
        '''
        self.assertEqual(
            audit.tuple_paths(source),
            [
                "timestream.polarimetry.enabled",
                "timestream.polarimetry.grouping",
            ],
        )

    def test_repo_boundary_is_exact(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = audit.audit(repo_root)
        self.assertFalse(result["path_or_boundary_drift"])


if __name__ == "__main__":
    unittest.main()
