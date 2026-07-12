from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.config import audit_processed_timestream_boundary as audit


class ProcessedTimestreamBoundaryAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        for source in audit.TYPED_READER_SOURCES.values():
            target = self.repo_root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
        serializer = self.repo_root / audit.PROCESSED_CONFIG_SERIALIZER_SOURCE
        serializer.parent.mkdir(parents=True, exist_ok=True)
        serializer.write_text("")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_reader(self, name: str, text: str) -> None:
        (self.repo_root / audit.TYPED_READER_SOURCES[name]).write_text(text)

    def write_manifest(self, paths: list[str]) -> Path:
        manifest = self.repo_root / audit.FROZEN_PATH_MANIFEST_SOURCE
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": audit.PATH_MANIFEST_SCHEMA_VERSION,
                    "source": "test",
                    "path_count": len(paths),
                    "path_sha256": audit.path_digest(paths),
                    "paths": paths,
                }
            )
        )
        return manifest

    def test_loads_canonical_frozen_path_manifest(self) -> None:
        paths = ["timestream.fruit_loops.enabled"]
        manifest = self.write_manifest(paths)

        source, loaded, digest = audit.load_frozen_paths(self.repo_root)

        self.assertEqual(source, manifest)
        self.assertEqual(loaded, paths)
        self.assertEqual(digest, audit.path_digest(paths))

    def test_rejects_duplicate_or_unsorted_manifest_paths(self) -> None:
        paths = [
            "timestream.fruit_loops.max_iters",
            "timestream.fruit_loops.enabled",
        ]
        self.write_manifest(paths)

        with self.assertRaisesRegex(ValueError, "unique and canonically sorted"):
            audit.load_frozen_paths(self.repo_root)

    def test_routes_second_pass_separately_from_general_flagging(self) -> None:
        self.assertEqual(
            audit.typed_reader_name(
                "timestream.processed_time_chunk.flagging.lower_tod_inv_var_factor"
            ),
            "flagging",
        )
        self.assertEqual(
            audit.typed_reader_name(
                "timestream.processed_time_chunk.flagging."
                "second_pass_local.min_spike_sigma"
            ),
            "second_pass_local",
        )

    def test_requires_leaf_key_in_declared_reader(self) -> None:
        path = (
            "timestream.processed_time_chunk.weighting."
            "source_mask_radius_arcsec"
        )
        self.write_reader("weighting", '"source_mask_radius_arcsec"')
        covered, uncovered, stale = audit.typed_reader_coverage(
            [path], self.repo_root
        )
        self.assertEqual(len(covered), 1)
        self.assertEqual(uncovered, [])
        self.assertEqual(stale, [])

        self.write_reader("weighting", '"different_key"')
        covered, uncovered, stale = audit.typed_reader_coverage(
            [path], self.repo_root
        )
        self.assertEqual(covered, [])
        self.assertEqual(uncovered, [path])
        self.assertEqual(stale, [])

    def test_reports_stale_compatibility_aliases(self) -> None:
        path = "timestream.fruit_loops.enabled"
        self.write_reader("fruit_loops", '"enabled"')
        with patch.dict(
            audit.COMPATIBILITY_ALIASES,
            {"timestream.fruit_loops.obsolete": path},
            clear=True,
        ):
            covered, uncovered, stale = audit.typed_reader_coverage(
                [path], self.repo_root
            )
        self.assertEqual(len(covered), 1)
        self.assertEqual(uncovered, [])
        self.assertEqual(
            stale, ["timestream.fruit_loops.obsolete"]
        )

    def test_requires_leaf_key_in_snapshot_serializer(self) -> None:
        path = "timestream.fruit_loops.max_iters"
        serializer = self.repo_root / audit.PROCESSED_CONFIG_SERIALIZER_SOURCE
        serializer.write_text('node["max_iters"] = value;')

        covered, uncovered = audit.serializer_coverage(
            [path], self.repo_root
        )
        self.assertEqual(covered, [path])
        self.assertEqual(uncovered, [])

        serializer.write_text('node["different_key"] = value;')
        covered, uncovered = audit.serializer_coverage(
            [path], self.repo_root
        )
        self.assertEqual(covered, [])
        self.assertEqual(uncovered, [path])

    def test_accepts_retired_compatibility_boundary(self) -> None:
        result = audit.compatibility_boundary("typed_reader(config);\n")

        self.assertTrue(result["retired"])
        self.assertFalse(result["isolated"])
        self.assertFalse(result["parser_precedes_seed"])
        self.assertEqual(result["legacy_parser_call_count"], 0)
        self.assertEqual(result["compatibility_seed_call_count"], 0)
        self.assertEqual(result["direct_mirror_call_counts"], {})

    def test_rejects_any_reintroduced_compatibility_call(self) -> None:
        result = audit.compatibility_boundary(
            "read_processor_config(ptcproc);\n"
            "mirror_processed_clean_config(clean, ptcproc);\n"
        )

        self.assertFalse(result["retired"])
        self.assertFalse(result["isolated"])
        self.assertFalse(result["parser_precedes_seed"])
        self.assertEqual(
            result["direct_mirror_call_counts"],
            {"mirror_processed_clean_config": 1},
        )


if __name__ == "__main__":
    unittest.main()
