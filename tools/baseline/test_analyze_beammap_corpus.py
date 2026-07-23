import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.baseline import analyze_beammap_corpus as corpus


class AnalyzeBeammapCorpusTest(unittest.TestCase):
    def test_accepts_complete_heterogeneous_corpus_without_creating_pairs(self) -> None:
        with self._manifest() as path:
            result = corpus.analyze(path)

        self.assertEqual(result["verdict"], "complete")
        self.assertEqual(result["valid_current_record_count"], 3)
        self.assertEqual(result["comparison_count"], 0)
        self.assertEqual(
            result["metric_summaries"]["citlali_total_log_seconds"]["median"],
            120.0,
        )
        relationship = next(
            row
            for row in result["relationships"]
            if row["workload"] == "scan_count"
            and row["metric"] == "citlali_total_log_seconds"
        )
        self.assertAlmostEqual(relationship["pearson_r"], 1.0)
        self.assertAlmostEqual(relationship["slope"], 2.0)
        self.assertEqual(
            result["identity_groups"]["configs"][0]["observation_ids"],
            [1001, 1002, 1003],
        )

    def test_reports_explicit_same_observation_comparison(self) -> None:
        with self._manifest(with_comparison=True) as path:
            result = corpus.analyze(path)

        self.assertTrue(result["complete"])
        self.assertEqual(result["comparison_count"], 1)
        pair = result["comparisons"][0]
        self.assertEqual(pair["observation_id"], 1001)
        self.assertAlmostEqual(
            pair["metrics"]["citlali_total_log_seconds"]["ratio"], 1.25
        )

    def test_rejects_comparison_with_different_observation_identity(self) -> None:
        with self._manifest(with_comparison=True, comparison_obsnum=9999) as path:
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertEqual(result["comparison_count"], 0)
        self.assertTrue(
            any("provenance identifies 9999" in error for error in result["errors"])
        )

    def test_marks_missing_expected_observation_incomplete(self) -> None:
        with self._manifest(omit_last=True) as path:
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertTrue(
            any(
                "expected observation 1003 has no current record" in error
                for error in result["errors"]
            )
        )

    def test_rejects_mixed_current_executables(self) -> None:
        with self._manifest(mixed_executables=True) as path:
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertIn(
            "current records do not use one identified executable",
            result["errors"],
        )

    def test_rejects_workload_override_that_conflicts_with_provenance(self) -> None:
        with self._manifest() as path:
            value = json.loads(path.read_text(encoding="utf-8"))
            value["observations"][0]["workload"] = {"scan_count": 999}
            path.write_text(json.dumps(value), encoding="utf-8")
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertTrue(
            any(
                "conflicts with provenance value 50" in error
                for error in result["errors"]
            )
        )

    def test_rejects_serious_log_issue(self) -> None:
        with self._manifest() as path:
            value = json.loads(path.read_text(encoding="utf-8"))
            metadata = path.parent / value["observations"][0]["metadata"]
            run = json.loads(metadata.read_text(encoding="utf-8"))
            run["reduction"]["log_issue_counts"] = {"error": 1}
            metadata.write_text(json.dumps(run), encoding="utf-8")
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertTrue(any("serious issues" in error for error in result["errors"]))

    def test_rejects_record_from_another_corpus(self) -> None:
        with self._manifest() as path:
            value = json.loads(path.read_text(encoding="utf-8"))
            metadata = path.parent / value["observations"][0]["metadata"]
            run = json.loads(metadata.read_text(encoding="utf-8"))
            run["campaign_id"] = "another-corpus"
            metadata.write_text(json.dumps(run), encoding="utf-8")
            result = corpus.analyze(path)

        self.assertFalse(result["complete"])
        self.assertTrue(
            any("performance campaign id" in error for error in result["errors"])
        )

    class _Manifest:
        def __init__(
            self,
            *,
            with_comparison: bool,
            comparison_obsnum: int,
            omit_last: bool,
            mixed_executables: bool,
        ) -> None:
            self.with_comparison = with_comparison
            self.comparison_obsnum = comparison_obsnum
            self.omit_last = omit_last
            self.mixed_executables = mixed_executables
            self.directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.directory = tempfile.TemporaryDirectory()
            root = Path(self.directory.name)
            entries = []
            for index, observation_id in enumerate((1001, 1002, 1003)):
                if self.omit_last and observation_id == 1003:
                    continue
                reduction = root / f"redu-{observation_id}"
                self._write_provenance(
                    reduction,
                    observation_id,
                    scan_count=50 + index * 10,
                )
                metadata = self._write_run(
                    reduction,
                    observation_id,
                    runtime=100.0 + index * 20.0,
                    executable=(
                        "other-binary"
                        if self.mixed_executables and observation_id == 1003
                        else "release-binary"
                    ),
                )
                entry = {
                    "observation_id": observation_id,
                    "metadata": str(metadata.relative_to(root)),
                }
                if self.with_comparison and observation_id == 1001:
                    comparison = root / "comparison-1001"
                    self._write_provenance(
                        comparison,
                        self.comparison_obsnum,
                        scan_count=50,
                    )
                    comparison_metadata = self._write_run(
                        comparison,
                        self.comparison_obsnum,
                        runtime=80.0,
                        executable="old-binary",
                        version="old-sha",
                    )
                    entry["comparisons"] = [
                        {
                            "label": "previous-release",
                            "metadata": str(comparison_metadata.relative_to(root)),
                        }
                    ]
                entries.append(entry)
            manifest = {
                "schema_version": corpus.CORPUS_SCHEMA_VERSION,
                "corpus_id": "test-corpus",
                "release": {
                    "label": "test-release",
                    "version_contains": "release-sha",
                    "build_type": "Release",
                    "required_dependencies": {
                        "kids": "kids-sha",
                        "tula": "tula-sha",
                    },
                },
                "protocol": {
                    "expected_observation_ids": [1001, 1002, 1003],
                    "require_single_executable": True,
                    "require_single_runtime_signature": True,
                },
                "observations": entries,
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

        @staticmethod
        def _write_provenance(
            reduction: Path, observation_id: int, *, scan_count: int
        ) -> None:
            reduction.mkdir(parents=True)
            value = {
                "schema_version": "citlali-beammap-provenance-v2",
                "observations": [
                    {
                        "obsnum": observation_id,
                        "detector_count": 5000,
                        "map_count": 5000,
                        "scan_count": scan_count,
                        "iterations": [
                            {
                                "active_map_count": 5000,
                                "mapmaking_pass_count": 1,
                            },
                            {
                                "active_map_count": 5000,
                                "mapmaking_pass_count": 1,
                            },
                        ],
                        "detector_tod": {
                            "maximum_sample_count": {
                                "available": True,
                                "value": 800,
                            }
                        },
                    }
                ],
            }
            (reduction / "beammap_provenance.yaml").write_text(
                yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
            )
            (reduction / "product.fits").write_bytes(b"fits")

        @staticmethod
        def _write_run(
            reduction: Path,
            observation_id: int,
            *,
            runtime: float,
            executable: str,
            version: str = "release-sha",
        ) -> Path:
            record = {
                "schema_version": corpus.RUN_SCHEMA_VERSION,
                "campaign_id": "test-corpus",
                "case_id": f"beammap-{observation_id}",
                "role": "candidate",
                "phase": "measured",
                "pair_index": 0,
                "build_type": "Release",
                "command_exit_code": 0,
                "structure_ok": True,
                "measurement_ok": True,
                "executable": {
                    "sha256": executable,
                    "version_output": version,
                    "dependencies": {
                        "kids": "kids-sha",
                        "tula": "tula-sha",
                    },
                },
                "host": {"hostname": "node1"},
                "storage": {"device": 123},
                "gnu_time": {
                    "elapsed_wall_seconds": runtime + 5.0,
                    "maximum_resident_set_kb": 2_000_000,
                    "filesystem_inputs": 100,
                    "filesystem_outputs": 200,
                },
                "reduction": {
                    "path": str(reduction),
                    "config_sha256": "config-sha",
                    "versions": {"citlali": version},
                    "citlali_total_log_seconds": runtime,
                    "log_issue_counts": {},
                    "runtime_signature": {"threads": 16, "policy": "omp"},
                    "config_leaves": [
                        {"path": "mapmaking.method", "value_key": '"jinc"'}
                    ],
                    "inputs": [
                        {
                            "path": "/shared/input.nc",
                            "basename": "input.nc",
                            "exists": True,
                            "size_bytes": 10,
                            "sha256": "input-sha",
                        }
                    ],
                    "profile": {
                        "present": True,
                        "stage_totals_seconds": {"beammap.mapmaking": runtime * 0.5},
                    },
                },
            }
            path = reduction / "performance_run.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            return path

    def _manifest(
        self,
        *,
        with_comparison: bool = False,
        comparison_obsnum: int = 1001,
        omit_last: bool = False,
        mixed_executables: bool = False,
    ) -> _Manifest:
        return self._Manifest(
            with_comparison=with_comparison,
            comparison_obsnum=comparison_obsnum,
            omit_last=omit_last,
            mixed_executables=mixed_executables,
        )


if __name__ == "__main__":
    unittest.main()
