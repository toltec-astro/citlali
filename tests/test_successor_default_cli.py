"""Exercise the actual default executable; no route flag or alternate binary."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class SuccessorDefaultCli(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="citlali-default-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"

    def invoke(self, config, *overlays):
        paths = []
        for index, content in enumerate((config, *overlays)):
            path = self.root / f"config-{index}.json"
            path.write_text(json.dumps(content))
            paths.append(str(path))
        return subprocess.run([CLI, *paths], text=True, capture_output=True, timeout=30)

    def request(self, contents):
        path = self.root / "input.json"
        path.write_text(json.dumps(contents))
        return dict(schema="citlali-development-v1", terminal="rtc-only",
                    input=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    output=str(self.output))

    def rejected(self, result, reason):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(reason, result.stdout + result.stderr)
        self.assertIn("Timestream Successor", result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_dump_config_describes_default_successor_and_actual_stop(self):
        result = subprocess.run([CLI, "--dump_config"], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("schema: citlali-development-v1", result.stdout)
        self.assertIn("terminal: cal", result.stdout)
        self.assertNotIn("mapmaking:", result.stdout)

    def test_legacy_config_cannot_silently_reach_old_processing(self):
        self.rejected(self.invoke(dict(runtime=dict(reduction_type="science"), data_items=[])),
                      "successor.input_binding_unavailable")

    def test_downstream_request_does_not_fall_back_to_rtc_completion(self):
        config = self.request({})
        config["terminal"] = "ordinary-science"
        self.rejected(self.invoke(config), "successor.PTC_not_implemented")

    def test_bound_request_tamper_is_rejected_before_execution(self):
        config = self.request({})
        Path(config["input"]["path"]).write_text("{\"changed\":true}")
        self.rejected(self.invoke(config), "successor.input_digest_mismatch")

    def test_unsupported_extra_settings_are_not_ignored(self):
        config = self.request({})
        config["legacy_fallback"] = True
        self.rejected(self.invoke(config), "successor.unsupported_configuration")

    def test_merged_configuration_uses_the_same_default(self):
        config = self.request({})
        self.rejected(self.invoke(config, dict(terminal="PTC")), "successor.PTC_not_implemented")

    def test_missing_terminal_defaults_to_cal_and_enters_the_existing_rtc_parent(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1"))
        del config["terminal"]
        self.rejected(self.invoke(config), "successor.RTC_plan_incomplete")

    def test_actual_rtc_boundary_refuses_missing_frozen_plan(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1"))
        self.rejected(self.invoke(config), "successor.RTC_plan_incomplete")

    def test_experimental_injection_not_enabled_by_default_activation(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1",
            terminal_request="rtc-only", decision_apply={}, declared_contaminant={}))
        self.rejected(self.invoke(config), "successor.unsupported_experiment")

    def test_relative_paths_cannot_acquire_ambient_binding(self):
        config = self.request({})
        config["input"]["path"] = "input.json"
        self.rejected(self.invoke(config), "successor.path_binding")

    def test_fixed_plan_injection_trials_are_not_default_reductions(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1",
            terminal_request="rtc-only", decision_apply={},
            fixed_plan_injections=[dict(amplitude=1.0)]))
        self.rejected(self.invoke(config), "successor.unsupported_experiment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    args, remaining = parser.parse_known_args()
    CLI = args.cli
    unittest.main(argv=[__file__, *remaining])
