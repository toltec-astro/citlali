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
        self.assertEqual(len(established), 11)

    def test_shared_test_graph_is_independent_of_top_level_entry_point(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        test_graph = (source_root / "tests/CMakeLists.txt").read_text()

        self.assertIn("CITLALI_TEST_SOURCE_ROOT", test_graph)
        self.assertNotIn("${CMAKE_SOURCE_DIR}/", test_graph)


if __name__ == "__main__":
    unittest.main()
