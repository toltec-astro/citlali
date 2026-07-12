from __future__ import annotations

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
