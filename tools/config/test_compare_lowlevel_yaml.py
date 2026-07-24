import json
import tempfile
import unittest
from pathlib import Path

from tools.config import compare_lowlevel_yaml as compare


class CompareLowlevelYamlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = [
            {
                "path": "inputs[].data_items[].filepath",
                "comparison": "basename",
                "rationale": "portable input identity",
            },
            {
                "path": "runtime.output_dir",
                "comparison": "nonempty_string",
                "rationale": "deployment location",
            },
        ]

    def test_binding_paths_compare_identity_not_prefix(self) -> None:
        baseline = {
            "inputs": [{"data_items": [{"filepath": "/site-a/data/toltec0.nc"}]}],
            "runtime": {"output_dir": "/site-a/reduced"},
            "mapmaking": {"method": "jinc"},
        }
        candidate = {
            "inputs": [{"data_items": [{"filepath": "/site-b/data/toltec0.nc"}]}],
            "runtime": {"output_dir": "/site-b/reduced"},
            "mapmaking": {"method": "jinc"},
        }

        result = compare.compare(
            baseline,
            candidate,
            [],
            self.rules,
            "test-bindings",
        )

        self.assertEqual(result["summary"]["diff_count"], 0)
        self.assertEqual(result["summary"]["binding_match_count"], 2)

    def test_changed_binding_basename_is_not_hidden(self) -> None:
        baseline = {
            "inputs": [{"data_items": [{"filepath": "/site-a/data/toltec0.nc"}]}],
            "runtime": {"output_dir": "/site-a/reduced"},
        }
        candidate = {
            "inputs": [{"data_items": [{"filepath": "/site-b/data/toltec1.nc"}]}],
            "runtime": {"output_dir": "/site-b/reduced"},
        }

        result = compare.compare(baseline, candidate, [], self.rules)

        self.assertEqual(result["summary"]["diff_count"], 1)
        self.assertEqual(result["diffs"][0]["kind"], "changed_binding_identity")

    def test_repeated_input_paths_all_use_binding_identity(self) -> None:
        baseline = {
            "inputs": [
                {
                    "data_items": [
                        {"filepath": "/site-a/data/toltec0.nc"},
                        {"filepath": "/site-a/data/toltec1.nc"},
                    ]
                }
            ],
            "runtime": {"output_dir": "/site-a/reduced"},
        }
        candidate = {
            "inputs": [
                {
                    "data_items": [
                        {"filepath": "/site-b/data/toltec0.nc"},
                        {"filepath": "/site-b/data/toltec1.nc"},
                    ]
                }
            ],
            "runtime": {"output_dir": "/site-b/reduced"},
        }

        result = compare.compare(baseline, candidate, [], self.rules)

        self.assertEqual(result["summary"]["diff_count"], 0)
        self.assertEqual(result["summary"]["binding_match_count"], 3)

    def test_empty_location_binding_is_rejected(self) -> None:
        baseline = {"runtime": {"output_dir": "/site-a/reduced"}}
        candidate = {"runtime": {"output_dir": ""}}

        result = compare.compare(baseline, candidate, [], self.rules)

        self.assertEqual(result["summary"]["diff_count"], 1)
        self.assertEqual(result["diffs"][0]["kind"], "invalid_binding_value")

    def test_binding_policy_registry_rejects_duplicate_paths(self) -> None:
        registry = {
            "schema_version": compare.BINDING_POLICY_SCHEMA_VERSION,
            "policies": [
                {
                    "policy_id": "duplicate",
                    "rules": [
                        {
                            "path": "runtime.output_dir",
                            "comparison": "nonempty_string",
                            "rationale": "first",
                        },
                        {
                            "path": "runtime.output_dir",
                            "comparison": "nonempty_string",
                            "rationale": "second",
                        },
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policies.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate binding rule"):
                compare.load_binding_policy(path, "duplicate")


if __name__ == "__main__":
    unittest.main()
