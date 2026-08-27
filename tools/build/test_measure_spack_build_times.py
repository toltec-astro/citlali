import tempfile
import unittest
from pathlib import Path

from measure_spack_build_times import (
    resolve_incremental_inputs,
    source_identity,
    timestamp_touch,
)
from run_spack_citlali_dev import build_command


class TimestampTouchTest(unittest.TestCase):
    def test_restores_timestamp_and_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.cpp"
            path.write_text("int answer = 42;\n")
            original_stat = path.stat()

            with timestamp_touch(path) as touch:
                self.assertGreater(
                    path.stat().st_mtime_ns,
                    original_stat.st_mtime_ns,
                )
                self.assertEqual(touch["path"], str(path))

            self.assertEqual(path.read_text(), "int answer = 42;\n")
            self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)


class BuildCommandTest(unittest.TestCase):
    def test_build_command_uses_requested_parallelism(self):
        self.assertEqual(
            build_command(build_dir=Path("/tmp/build"), jobs=6),
            ["cmake", "--build", "/tmp/build", "-j", "6"],
        )

    def test_source_identity_reports_revision_and_status(self):
        source_root = Path(__file__).resolve().parents[2]
        identity = source_identity(source_root)
        self.assertRegex(identity["revision"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(identity["status"], list)

    def test_incremental_inputs_must_remain_inside_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory) / "source"
            source_root.mkdir()
            source_file = source_root / "input.cpp"
            source_file.write_text("int main() {}\n")
            source_file = source_file.resolve()

            self.assertEqual(
                resolve_incremental_inputs(source_root, [Path("input.cpp")]),
                [source_file],
            )
            with self.assertRaisesRegex(ValueError, "outside the source tree"):
                resolve_incremental_inputs(source_root, [Path("../other.cpp")])


if __name__ == "__main__":
    unittest.main()
