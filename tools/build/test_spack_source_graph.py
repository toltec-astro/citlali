#!/usr/bin/env python3
"""Keep the established and Spack production translation-unit graphs equal."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path



class SpackSourceGraphTest(unittest.TestCase):
    def test_production_sources_and_fp_policy_have_one_owner(self) -> None:
        root = Path(__file__).resolve().parents[2]
        module = (root / "cmake/citlali_core_sources.cmake").read_text()
        core_block = re.search(r"set\(CITLALI_CORE_SOURCES(.*?)\)", module, re.S)
        strict_block = re.search(r"set\(CITLALI_STRICT_FP_SOURCES(.*?)\)", module, re.S)
        self.assertIsNotNone(core_block)
        self.assertIsNotNone(strict_block)
        sources = core_block.group(1).split()
        strict = strict_block.group(1).split()
        self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(len(strict), len(set(strict)))
        self.assertTrue(set(strict).issubset(sources))
        for path in sources:
            self.assertTrue((root / path).is_file(), path)
        for filename in (
            "jinc_discrete.cpp", "jinc_certified.cpp", "ptc_inverse_noise.cpp",
            "ptc_empirical_map_handoff.cpp", "point_optical_response.cpp", "fruit_cycle.cpp",
        ):
            self.assertIn(filename, [Path(path).name for path in strict])
        self.assertIn('COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off"', module)
        for entry in ("CMakeLists.txt", "cmake/spack/CMakeLists.txt"):
            graph = (root / entry).read_text()
            self.assertIn("cmake/citlali_core_sources.cmake", graph)
            self.assertRegex(graph, r"citlali_add_core_sources\(citlali ")
            self.assertNotRegex(graph, r"target_sources\(\s*citlali\s")
            self.assertNotIn("set_source_files_properties(", graph)

    def test_every_ordinary_test_is_in_shared_build_aggregate(self) -> None:
        root = Path(__file__).resolve().parents[2]
        graph = (root / "tests/CMakeLists.txt").read_text()
        targets = set(re.findall(r"add_executable\(\s*(citlali\w+)", graph))
        discovered = set(re.findall(r"citlali_discover_tests\(\s*(citlali\w+)", graph))
        explicit = set(re.findall(
            r"add_dependencies\(citlali_test_binaries\s+(citlali\w+)\)", graph
        ))
        separately_requested = {"citlali_science_map_truth_test", "citlali_science_map_tsan_test"}
        self.assertEqual(targets - separately_requested, discovered | (explicit & targets))
        self.assertTrue(separately_requested.isdisjoint(discovered | explicit))
        self.assertIn("citlali_safety_test", discovered)
        self.assertIn("add_custom_target(citlali_test_binaries ALL)", graph)
        self.assertIn("add_dependencies(check citlali_test_binaries)", graph)
        self.assertIn("add_dependencies(citlali_test_binaries ${target})", graph)
        self.assertIn("gtest_discover_tests(${target} ${ARGN})", graph)
        self.assertNotIn("${CMAKE_SOURCE_DIR}/", graph)
        spack = (root / "cmake/spack/CMakeLists.txt").read_text()
        self.assertNotIn("PROPERTY EXCLUDE_FROM_ALL FALSE", spack)

    def test_spack_driver_builds_cli_and_ordinary_aggregate(self) -> None:
        from run_spack_citlali_dev import build_command

        self.assertEqual(build_command(build_dir=Path("/tmp/fresh"), jobs=8), [
            "cmake", "--build", "/tmp/fresh", "--target",
            "citlali_cli", "citlali_test_binaries", "-j", "8",
        ])

    def test_developer_binary_requires_this_builds_actual_executable(self) -> None:
        from run_spack_citlali_dev import developer_binary

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            build.mkdir()
            binary = build / "citlali"
            manifest = build / "citlali_cli_path.txt"
            manifest.write_text(str(binary))
            with self.assertRaisesRegex(RuntimeError, "missing or foreign"):
                developer_binary(build)
            binary.write_text("synthetic executable")
            self.assertEqual(developer_binary(build), binary.resolve())
            foreign = root / "foreign"
            foreign.write_text("old executable")
            manifest.write_text(str(foreign))
            with self.assertRaisesRegex(RuntimeError, "missing or foreign"):
                developer_binary(build)

    def test_application_uses_shared_inputs_without_acceptance_main(self) -> None:
        root = Path(__file__).resolve().parents[2]
        execution = (root / "src/citlali/cli/rtc_pipeline_execution.cpp").read_text()
        acceptance = (root / "tools/timestream_successor/identity_route_acceptance.cpp").read_text()
        self.assertNotIn("#define main", execution)
        self.assertNotIn('"identity_route_acceptance.cpp"', execution)
        self.assertIn('#include "successor_input_bindings.h"', execution)
        self.assertIn('#include "successor_input_bindings.h"', acceptance)
        self.assertNotIn("TelescopeInput load_telescope(", acceptance)
        for entry in ("CMakeLists.txt", "cmake/spack/CMakeLists.txt"):
            graph = (root / entry).read_text()
            self.assertIn("citlali_add_successor_input_bindings(", graph)
            self.assertIn("citlali_rtc_execution PUBLIC citlali_successor_input_bindings", graph)

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
