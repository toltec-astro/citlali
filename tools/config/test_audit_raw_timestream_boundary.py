from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.config import audit_raw_timestream_boundary as audit


class RawTimestreamBoundaryAuditTest(unittest.TestCase):
    def test_extracts_sorted_unique_literal_paths(self) -> None:
        body = '''
        std::tuple{"timestream", "raw_time_chunk", "filter", "enabled"};
        std::tuple{"timestream", "raw_time_chunk", "despike", "enabled"};
        std::tuple{"timestream", "raw_time_chunk", "filter", "enabled"};
        '''
        self.assertEqual(
            audit.literal_paths(body),
            [
                "timestream.raw_time_chunk.despike.enabled",
                "timestream.raw_time_chunk.filter.enabled",
            ],
        )

    def test_extracts_only_rtc_parser_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "rtcproc.h"
            source.write_text(
                "void RTCProc::get_config() { parser(); }\n"
                "template <typename T>\nvoid next(T value) {}\n"
            )
            self.assertEqual(
                audit.parser_body(source),
                "void RTCProc::get_config() { parser(); }",
            )

    def test_classifies_raw_and_adjacent_polarimetry_paths(self) -> None:
        self.assertEqual(
            audit.family("timestream.raw_time_chunk.filter.enabled"),
            "raw_timestream",
        )
        self.assertEqual(
            audit.family("timestream.polarimetry.enabled"),
            "polarimetry",
        )
        self.assertEqual(audit.family("mapmaking.enabled"), "unclassified")

    def test_accepts_exact_typed_authority_boundary(self) -> None:
        lines = [
            "read_raw_timestream_request_config(config, request, diag);",
            "read_processor_config(rtcproc, config);",
        ]
        lines.extend(
            f"{name}(typed, rtcproc);"
            for name in audit.LEGACY_TO_TYPED_MIRROR_CALLS
        )
        lines.append(
            "initialize_raw_timestream_authority(request, plan, typed, rtcproc);"
        )
        lines.append(
            "compare_raw_timestream_authority(oracle, rtcproc);"
        )
        lines.append(
            "adapt_legacy_polarimetry_runtime(legacy, rtcproc);"
        )
        result = audit.authority_boundary("\n".join(lines))

        self.assertTrue(result["exact"])
        self.assertTrue(result["parser_precedes_mirrors"])
        self.assertTrue(result["authority_order_exact"])
        self.assertEqual(result["missing_mirror_calls"], [])
        self.assertEqual(result["unexpected_mirror_calls"], [])
        self.assertEqual(result["non_unit_mirror_call_counts"], {})

    def test_rejects_missing_repeated_or_unexpected_mirrors(self) -> None:
        first = audit.LEGACY_TO_TYPED_MIRROR_CALLS[0]
        result = audit.authority_boundary(
            "read_processor_config(rtcproc, config);\n"
            f"{first}(typed, rtcproc);\n"
            f"{first}(typed, rtcproc);\n"
            "mirror_raw_untracked_config(typed, rtcproc);\n"
        )

        self.assertFalse(result["exact"])
        self.assertIn(
            audit.LEGACY_TO_TYPED_MIRROR_CALLS[1],
            result["missing_mirror_calls"],
        )
        self.assertEqual(
            result["unexpected_mirror_calls"],
            ["mirror_raw_untracked_config"],
        )
        self.assertEqual(result["non_unit_mirror_call_counts"], {first: 2})

    def test_rejects_missing_or_misordered_typed_authority(self) -> None:
        lines = ["read_processor_config(rtcproc, config);"]
        lines.extend(
            f"{name}(typed, rtcproc);"
            for name in audit.LEGACY_TO_TYPED_MIRROR_CALLS
        )
        without_authority = audit.authority_boundary("\n".join(lines))
        self.assertFalse(without_authority["exact"])
        self.assertFalse(without_authority["authority_order_exact"])

        lines.insert(
            1,
            "read_raw_timestream_request_config(config, request, diag);",
        )
        lines.append(
            "initialize_raw_timestream_authority(request, plan, typed, rtcproc);"
        )
        lines.append(
            "compare_raw_timestream_authority(oracle, rtcproc);"
        )
        lines.append(
            "adapt_legacy_polarimetry_runtime(legacy, rtcproc);"
        )
        misordered = audit.authority_boundary("\n".join(lines))
        self.assertFalse(misordered["exact"])
        self.assertFalse(misordered["authority_order_exact"])

    def test_rejects_missing_polarimetry_runtime_adapter(self) -> None:
        lines = [
            "read_raw_timestream_request_config(config, request, diag);",
            "read_processor_config(legacy, config);",
        ]
        lines.extend(
            f"{name}(oracle, legacy);"
            for name in audit.LEGACY_TO_TYPED_MIRROR_CALLS
        )
        lines.append(
            "initialize_raw_timestream_authority(request, plan, typed, rtcproc);"
        )
        lines.append(
            "compare_raw_timestream_authority(oracle, rtcproc);"
        )

        result = audit.authority_boundary("\n".join(lines))

        self.assertFalse(result["exact"])
        self.assertFalse(result["authority_order_exact"])
        self.assertEqual(result["polarimetry_runtime_adapter_call_count"], 0)

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

        self.assertEqual(
            covered, ["timestream.raw_time_chunk.filter.enabled"]
        )
        self.assertEqual(
            uncovered,
            ["timestream.raw_time_chunk.filter.freq_high_Hz"],
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
            with patch.object(
                audit, "RAW_ADAPTER_SOURCES", ("adapter.h",)
            ):
                covered, uncovered = audit.adapter_coverage(
                    [
                        "timestream.raw_time_chunk.filter.enabled",
                        "timestream.raw_time_chunk.filter.freq_high_Hz",
                    ],
                    repo_root,
                )

        self.assertEqual(
            covered, ["timestream.raw_time_chunk.filter.enabled"]
        )
        self.assertEqual(
            uncovered,
            ["timestream.raw_time_chunk.filter.freq_high_Hz"],
        )


if __name__ == "__main__":
    unittest.main()
