import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.baseline import run_performance_case as performance


GNU_TIME_SAMPLE = """
        User time (seconds): 12.50
        System time (seconds): 1.25
        Percent of CPU this job got: 399%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.50
        Maximum resident set size (kbytes): 123456
        Major (requiring I/O) page faults: 7
        Minor (reclaiming a frame) page faults: 800
        Voluntary context switches: 90
        Involuntary context switches: 11
        File system inputs: 12
        File system outputs: 34
        Exit status: 0
"""


class RunPerformanceCaseTest(unittest.TestCase):
    def test_parses_gnu_time_report(self) -> None:
        result = performance.parse_gnu_time(GNU_TIME_SAMPLE)

        self.assertEqual(result["elapsed_wall_seconds"], 3723.5)
        self.assertEqual(result["maximum_resident_set_kb"], 123456)
        self.assertEqual(result["cpu_percent"], 399.0)
        self.assertEqual(result["filesystem_outputs"], 34)

    def test_parses_minute_elapsed_format(self) -> None:
        self.assertEqual(performance.parse_elapsed_seconds("2:03.25"), 123.25)

    def test_aggregates_profile_records_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "citlali_profile.ecsv"
            path.write_text(
                "# %ECSV 1.0\n"
                "index stage context elapsed_s\n"
                '0 "map.populate" "iter=0" 1.250000\n'
                '1 "map.populate" "iter=1" 2.750000\n'
                '2 "map.output" "" 0.500000\n',
                encoding="utf-8",
            )

            result = performance.parse_profile(path)

        self.assertTrue(result["present"])
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["stage_totals_seconds"]["map.populate"], 4.0)
        self.assertEqual(result["stage_totals_seconds"]["map.output"], 0.5)

    def test_missing_profile_is_explicit(self) -> None:
        result = performance.parse_profile(Path("/missing/profile.ecsv"))

        self.assertFalse(result["present"])
        self.assertEqual(result["record_count"], 0)

    def test_reads_binary_and_dependency_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "citlali.log"
            path.write_text(
                "citlali version: v4-test\n"
                "kids version: kids-test\n"
                "tula version: tula-test\n",
                encoding="utf-8",
            )

            result = performance.versions_from_log(path)

        self.assertEqual(
            result,
            {"citlali": "v4-test", "kids": "kids-test", "tula": "tula-test"},
        )

    @mock.patch("tools.baseline.run_performance_case.subprocess.run")
    def test_accepts_gnu_time_version_capitalization(
        self, run: mock.Mock
    ) -> None:
        run.return_value = mock.Mock(
            returncode=0, stdout="time (GNU Time) 1.9\n", stderr=""
        )

        performance.verify_gnu_time(Path("/usr/bin/time"))


if __name__ == "__main__":
    unittest.main()
