import json
import tempfile
import unittest
from pathlib import Path

from tools.baseline import analyze_performance_campaign as campaign


class AnalyzePerformanceCampaignTest(unittest.TestCase):
    def test_accepts_complete_paired_campaign(self) -> None:
        with self._campaign() as path:
            result = campaign.analyze(path)

        self.assertEqual(result["verdict"], "accepted")
        self.assertEqual(result["measured_pair_count"], 3)
        self.assertAlmostEqual(
            result["metric_ratio_summaries"]["citlali_total_log_seconds"]["median"],
            1.04,
        )
        self.assertFalse(result["protocol_errors"])
        self.assertFalse(result["budget_failures"])

    def test_rejects_complete_campaign_over_runtime_budget(self) -> None:
        with self._campaign(candidate_runtime=106.0) as path:
            result = campaign.analyze(path)

        self.assertEqual(result["verdict"], "rejected")
        self.assertIn("citlali_total_log_seconds", result["budget_failures"][0])

    def test_marks_missing_pair_campaign_incomplete(self) -> None:
        with self._campaign(measured_pairs=2) as path:
            result = campaign.analyze(path)

        self.assertEqual(result["verdict"], "incomplete")
        self.assertTrue(any("below required" in error for error in result["protocol_errors"]))

    def test_unallowlisted_config_difference_is_incomplete(self) -> None:
        with self._campaign(candidate_config_value="2") as path:
            result = campaign.analyze(path)

        self.assertEqual(result["verdict"], "incomplete")
        self.assertTrue(any("configs differ" in error for error in result["protocol_errors"]))

    def test_different_large_input_paths_require_hashes(self) -> None:
        baseline = {
            "reduction": {
                "inputs": [
                    {
                        "path": "/baseline/input.nc",
                        "basename": "input.nc",
                        "exists": True,
                        "size_bytes": 1_000_000_000,
                        "sha256": None,
                    }
                ]
            }
        }
        candidate = {
            "reduction": {
                "inputs": [
                    {
                        "path": "/candidate/input.nc",
                        "basename": "input.nc",
                        "exists": True,
                        "size_bytes": 1_000_000_000,
                        "sha256": None,
                    }
                ]
            }
        }

        result = campaign.compare_inputs(baseline, candidate)

        self.assertFalse(result["equivalent"])
        self.assertEqual(
            result["differences"][0]["reason"],
            "different paths without complete hashes",
        )

    class _Campaign:
        def __init__(
            self,
            candidate_runtime: float,
            measured_pairs: int,
            candidate_config_value: str,
        ) -> None:
            self.candidate_runtime = candidate_runtime
            self.measured_pairs = measured_pairs
            self.candidate_config_value = candidate_config_value
            self.directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.directory = tempfile.TemporaryDirectory()
            root = Path(self.directory.name)
            runs = []
            runs.append(self._write_run(root, "baseline-warmup", "baseline", "warmup", 0, 0))
            runs.append(self._write_run(root, "candidate-warmup", "candidate", "warmup", 0, 1))
            for pair_index in range(self.measured_pairs):
                first = "baseline" if pair_index % 2 == 0 else "candidate"
                second = "candidate" if first == "baseline" else "baseline"
                runs.append(
                    self._write_run(
                        root,
                        f"{first}-{pair_index}",
                        first,
                        "measured",
                        pair_index,
                        pair_index * 10 + 2,
                    )
                )
                runs.append(
                    self._write_run(
                        root,
                        f"{second}-{pair_index}",
                        second,
                        "measured",
                        pair_index,
                        pair_index * 10 + 3,
                    )
                )
            value = {
                "schema_version": campaign.CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": "beammap-test",
                "validation_epoch_id": "epoch",
                "validation_profile_id": "profile",
                "mode": "beammap",
                "build_type": "Release",
                "roles": {
                    "baseline": {"version_contains": "base-sha"},
                    "candidate": {"version_contains": "candidate-sha"},
                },
                "protocol": {
                    "minimum_measured_pairs": 3,
                    "preferred_measured_pairs": 5,
                    "require_warmup_each_role": True,
                    "require_same_host": True,
                    "require_alternating_first_role": True,
                    "first_measured_role": "baseline",
                    "require_runtime_signature_match": True,
                    "require_profiler_overhead_evidence": False,
                    "config_ignore_paths": ["runtime.output_dir"],
                },
                "budgets": {
                    "median_citlali_runtime_ratio_max": 1.05,
                    "median_peak_rss_ratio_max": 1.05,
                },
                "profiler_overhead_evidence": {"status": "pending"},
                "runs": runs,
            }
            path = root / "campaign.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

        def _write_run(
            self,
            root: Path,
            case_id: str,
            role: str,
            phase: str,
            pair_index: int,
            order: int,
        ) -> str:
            runtime = 100.0 if role == "baseline" else self.candidate_runtime
            config_value = "1" if role == "baseline" else self.candidate_config_value
            record = {
                "schema_version": "citlali-performance-run-v1",
                "campaign_id": "beammap-test",
                "case_id": case_id,
                "role": role,
                "phase": phase,
                "pair_index": pair_index,
                "build_type": "Release",
                "started_utc": f"2026-07-16T12:00:{order * 2:02d}+00:00",
                "ended_utc": f"2026-07-16T12:00:{order * 2 + 1:02d}+00:00",
                "command_exit_code": 0,
                "structure_ok": True,
                "measurement_ok": True,
                "executable": {
                    "sha256": "baseline-binary" if role == "baseline" else "candidate-binary",
                    "version_output": (
                        "base-sha" if role == "baseline" else "candidate-sha"
                    ),
                    "dependencies": {},
                },
                "host": {
                    "hostname": "node1",
                    "platform": "Linux-test",
                    "affinity_cpu_count": 16,
                    "environment": {},
                },
                "storage": {"device": 123},
                "gnu_time": {
                    "elapsed_wall_seconds": runtime + 5.0,
                    "maximum_resident_set_kb": 1000 if role == "baseline" else 1020,
                    "filesystem_inputs": 10,
                    "filesystem_outputs": 20,
                },
                "reduction": {
                    "versions": {
                        "citlali": (
                            "base-sha" if role == "baseline" else "candidate-sha"
                        )
                    },
                    "citlali_total_log_seconds": runtime,
                    "log_issue_counts": {},
                    "config_leaves": [
                        {"path": "runtime.output_dir", "value_key": f'"/{role}"'},
                        {"path": "mapmaking.method", "value_key": config_value},
                    ],
                    "inputs": [
                        {
                            "path": "/shared/input.nc",
                            "basename": "input.nc",
                            "exists": True,
                            "size_bytes": 100,
                            "sha256": "abc",
                        }
                    ],
                    "runtime_signature": {"threads": 16},
                    "profile": {
                        "stage_totals_seconds": {
                            "map.populate": runtime * 0.5,
                            "map.output": runtime * 0.1,
                        }
                    },
                },
            }
            path = root / f"{case_id}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            return path.name

    def _campaign(
        self,
        candidate_runtime: float = 104.0,
        measured_pairs: int = 3,
        candidate_config_value: str = "1",
    ) -> _Campaign:
        return self._Campaign(
            candidate_runtime, measured_pairs, candidate_config_value
        )


if __name__ == "__main__":
    unittest.main()
