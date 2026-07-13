from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.config import audit_raw_timestream_boundary as audit


class RawTimestreamBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_frozen_manifest(self) -> None:
        paths = [
            "timestream.polarimetry.enabled",
            "timestream.polarimetry.grouping",
            "timestream.raw_time_chunk.filter.enabled",
        ]
        with patch.object(audit, "EXPECTED_PATH_COUNT", 3), patch.object(
            audit, "EXPECTED_RAW_PATH_COUNT", 1
        ), patch.object(audit, "EXPECTED_PATH_SHA256", audit.path_digest(paths)):
            state = audit.manifest_state(
                {
                    "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                    "retired_parser_source": audit.PARSER_SOURCE,
                    "path_count": 3,
                    "path_sha256": audit.path_digest(paths),
                    "paths": paths,
                }
            )
        self.assertTrue(state["exact"])

    def test_rejects_unsorted_or_duplicate_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "manifest.json"
            source.write_text(
                '{"paths":["timestream.raw_time_chunk.z",'
                '"timestream.raw_time_chunk.a"]}'
            )
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                audit.load_frozen_manifest(source)

    def test_classifies_raw_and_adjacent_polarimetry_paths(self) -> None:
        self.assertEqual(
            audit.family("timestream.raw_time_chunk.filter.enabled"),
            "raw_timestream",
        )
        self.assertEqual(
            audit.family("timestream.polarimetry.enabled"), "polarimetry"
        )
        self.assertEqual(audit.family("mapmaking.enabled"), "unclassified")

    def test_accepts_retired_parser(self) -> None:
        self.assertTrue(audit.retired_parser_state("class RTCProc {};")["retired"])

    def test_rejects_parser_declaration_or_definition(self) -> None:
        declaration = audit.retired_parser_state("void get_config(Config &);")
        definition = audit.retired_parser_state(
            "void RTCProc::get_config(Config &) {}"
        )
        self.assertFalse(declaration["retired"])
        self.assertFalse(definition["retired"])

    def exact_boundary_lines(self) -> list[str]:
        return [
            "read_raw_timestream_request_config(config, request, diag);",
            "read_legacy_polarimetry_runtime_config(config, legacy, diag);",
            "initialize_raw_timestream_authority(request, plan, typed, rtc);",
            "adapt_legacy_polarimetry_runtime(legacy, rtc);",
        ]

    def test_accepts_exact_retired_authority_boundary(self) -> None:
        result = audit.authority_boundary("\n".join(self.exact_boundary_lines()))
        self.assertTrue(result["exact"])
        self.assertTrue(result["authority_order_exact"])
        self.assertEqual(result["legacy_parser_call_count"], 0)
        self.assertEqual(result["legacy_to_typed_mirror_call_counts"], {})

    def test_rejects_legacy_parser_mirror_or_compare(self) -> None:
        forbidden = [
            "read_processor_config(legacy, config);",
            "mirror_raw_filter_config(typed, legacy);",
            "compare_raw_timestream_authority(typed, legacy);",
        ]
        for line in forbidden:
            with self.subTest(line=line):
                result = audit.authority_boundary(
                    "\n".join(self.exact_boundary_lines() + [line])
                )
                self.assertFalse(result["exact"])

    def test_rejects_missing_or_misordered_authority_calls(self) -> None:
        missing_reader = self.exact_boundary_lines()[1:]
        self.assertFalse(
            audit.authority_boundary("\n".join(missing_reader))["exact"]
        )
        misordered = list(reversed(self.exact_boundary_lines()))
        self.assertFalse(
            audit.authority_boundary("\n".join(misordered))["exact"]
        )

    def test_typed_reader_coverage_accepts_parent_and_compatibility_alias(self) -> None:
        legacy_alias = next(iter(audit.COMPATIBILITY_ALIASES))
        typed_alias = audit.COMPATIBILITY_ALIASES[legacy_alias]
        frozen = [
            "timestream.raw_time_chunk.filter.notch",
            "timestream.raw_time_chunk.filter.notch.enabled",
            legacy_alias,
        ]
        declared = [
            "timestream.raw_time_chunk.filter.notch.enabled",
            typed_alias,
        ]
        covered, uncovered, stale = audit.typed_reader_coverage(
            frozen, declared
        )
        self.assertEqual(covered, sorted(frozen))
        self.assertEqual(uncovered, [])
        self.assertEqual(stale, [])

    def test_serializer_coverage_reports_missing_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source = repo_root / "serializer.h"
            source.write_text('node["enabled"] = true;\n')
            with patch.object(audit, "RAW_SERIALIZER_SOURCE", "serializer.h"):
                covered, uncovered = audit.serializer_coverage(
                    [
                        "timestream.raw_time_chunk.filter.enabled",
                        "timestream.raw_time_chunk.filter.freq_high_Hz",
                    ],
                    repo_root,
                )
        self.assertEqual(covered, ["timestream.raw_time_chunk.filter.enabled"])
        self.assertEqual(
            uncovered, ["timestream.raw_time_chunk.filter.freq_high_Hz"]
        )

    def test_rejects_yaml_string_view_assignment(self) -> None:
        source = '''
        node["unsafe"] = citlali::config::to_string(value);
        node["safe"] = std::string{citlali::config::to_string(value)};
        '''
        self.assertEqual(
            audit.unsafe_yaml_string_view_assignment_lines(source), [2]
        )

    def test_adapter_coverage_reports_missing_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source = repo_root / "adapter.h"
            source.write_text("target.enabled = source.enabled;\n")
            with patch.object(audit, "RAW_ADAPTER_SOURCES", ("adapter.h",)):
                covered, uncovered = audit.adapter_coverage(
                    [
                        "timestream.raw_time_chunk.filter.enabled",
                        "timestream.raw_time_chunk.filter.freq_high_Hz",
                    ],
                    repo_root,
                )
        self.assertEqual(covered, ["timestream.raw_time_chunk.filter.enabled"])
        self.assertEqual(
            uncovered, ["timestream.raw_time_chunk.filter.freq_high_Hz"]
        )


if __name__ == "__main__":
    unittest.main()
