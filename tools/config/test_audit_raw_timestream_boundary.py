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

    def test_accepts_exact_legacy_to_typed_boundary(self) -> None:
        lines = [
            "read_raw_timestream_request_config(config, request, diag);",
            "read_processor_config(rtcproc, config);",
        ]
        lines.extend(
            f"{name}(typed, rtcproc);"
            for name in audit.LEGACY_TO_TYPED_MIRROR_CALLS
        )
        lines.append("compare_raw_timestream_shadow(request, rtcproc);")
        result = audit.legacy_boundary("\n".join(lines))

        self.assertTrue(result["exact"])
        self.assertTrue(result["parser_precedes_mirrors"])
        self.assertTrue(result["shadow_order_exact"])
        self.assertEqual(result["missing_mirror_calls"], [])
        self.assertEqual(result["unexpected_mirror_calls"], [])
        self.assertEqual(result["non_unit_mirror_call_counts"], {})

    def test_rejects_missing_repeated_or_unexpected_mirrors(self) -> None:
        first = audit.LEGACY_TO_TYPED_MIRROR_CALLS[0]
        result = audit.legacy_boundary(
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

    def test_rejects_missing_or_misordered_typed_shadow(self) -> None:
        lines = ["read_processor_config(rtcproc, config);"]
        lines.extend(
            f"{name}(typed, rtcproc);"
            for name in audit.LEGACY_TO_TYPED_MIRROR_CALLS
        )
        without_shadow = audit.legacy_boundary("\n".join(lines))
        self.assertFalse(without_shadow["exact"])
        self.assertFalse(without_shadow["shadow_order_exact"])

        lines.insert(
            1,
            "read_raw_timestream_request_config(config, request, diag);",
        )
        lines.append("compare_raw_timestream_shadow(request, rtcproc);")
        misordered = audit.legacy_boundary("\n".join(lines))
        self.assertFalse(misordered["exact"])
        self.assertFalse(misordered["shadow_order_exact"])

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
