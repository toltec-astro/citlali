from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from spack_citlali_profiles import get_profile


MODULE_PATH = Path(__file__).with_name("test_spack_citlali.py")
SPEC = importlib.util.spec_from_file_location(
    "spack_citlali_installed_acceptance", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InstalledConsumerCommandTest(unittest.TestCase):
    def test_configure_uses_spack_environment_compiler(self) -> None:
        command = MODULE.consumer_configure_command(
            consumer_source=Path("/tmp/source"),
            consumer_build_dir=Path("/tmp/build"),
            package_prefix=Path("/tmp/prefix"),
            profile=get_profile("unity-gcc13"),
        )
        self.assertNotIn("-DCMAKE_CXX_COMPILER=/usr/bin/g++", command)
        self.assertIn("--fresh", command)

    def test_build_remains_inside_spack_build_environment(self) -> None:
        command = MODULE.consumer_build_command(
            spack=Path("/opt/spack/bin/spack"),
            environment_path=Path("/tmp/environment"),
            consumer_build_dir=Path("/tmp/build"),
            jobs=3,
        )
        self.assertEqual(
            command,
            [
                "/opt/spack/bin/spack",
                "-e",
                "/tmp/environment",
                "build-env",
                "citlali",
                "--",
                "cmake",
                "--build",
                "/tmp/build",
                "-j",
                "3",
            ],
        )


if __name__ == "__main__":
    unittest.main()
