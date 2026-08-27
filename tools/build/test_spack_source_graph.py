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


if __name__ == "__main__":
    unittest.main()
