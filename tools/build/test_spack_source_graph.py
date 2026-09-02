#!/usr/bin/env python3
"""Keep the established and Spack production translation-unit graphs equal."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


SOURCE_PATTERN = re.compile(
    r'"(?:\$\{CITLALI_SOURCE_ROOT\}/)?(src/citlali/[^"\s]+\.cpp)"'
)


class SpackSourceGraphTest(unittest.TestCase):
    def test_production_source_graph_matches_established_cmake(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        established = set(
            SOURCE_PATTERN.findall((source_root / "CMakeLists.txt").read_text())
        )
        spack = set(
            SOURCE_PATTERN.findall(
                (source_root / "cmake/spack/CMakeLists.txt").read_text()
            )
        )

        self.assertEqual(spack, established)
        self.assertEqual(len(established), 14)

    def test_shared_test_graph_is_independent_of_top_level_entry_point(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        test_graph = (source_root / "tests/CMakeLists.txt").read_text()

        self.assertIn("CITLALI_TEST_SOURCE_ROOT", test_graph)
        self.assertNotIn("${CMAKE_SOURCE_DIR}/", test_graph)

    def test_spack_default_build_covers_every_registered_ctest_binary(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        spack_graph = (source_root / "cmake/spack/CMakeLists.txt").read_text()

        for target in (
            "citlali_test",
            "citlali_jinc_map_contract_test",
            "citlali_jinc_parallel_ownership_test",
            "citlali_safety_test",
            "citlali_sci_align_test",
            "citlali_science_map_fits_products_test",
            "citlali_timestream_successor_native_paired_readout_test",
            "citlali_timestream_successor_ast_scan_motion_test",
            "citlali_timestream_successor_identity_test",
            "citlali_timestream_successor_identity_route_context_test",
        ):
            self.assertIn(target, spack_graph)

    def test_acceptance_runner_uses_raw_timestream_compatibility_boundary(
        self,
    ) -> None:
        source_root = Path(__file__).resolve().parents[2]
        runner = (
            source_root
            / "tools/timestream_successor/identity_route_acceptance.cpp"
        ).read_text()

        self.assertIn(
            "#include <citlali/core/compat/kidscpp_raw_timestream.h>",
            runner,
        )
        self.assertNotIn("#include <kids/toltec/toltec.h>", runner)
        self.assertIn("kidscpp::get_raw_timestream_meta", runner)
        self.assertIn("kidscpp::read_raw_timestream_slice", runner)
        self.assertNotIn("kids::toltec::get_meta", runner)
        self.assertNotIn("kids::toltec::read_data_slice", runner)


if __name__ == "__main__":
    unittest.main()
