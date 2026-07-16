from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.config.tolteca_mode_kit import (
    MODE_REDUCTION_TYPES,
    build_report,
    extract_low_level,
    flatten_leaves,
    merge_files,
    normalized_path,
    policy_sha256,
    recursive_update,
    validate_modes,
)


class RecursiveUpdateTest(unittest.TestCase):
    def test_recursively_overrides_dict_value(self) -> None:
        target = {"outer": {"keep": 1, "replace": 2}}

        recursive_update(target, {"outer": {"replace": 3}})

        self.assertEqual(target, {"outer": {"keep": 1, "replace": 3}})

    def test_updates_list_item_by_numeric_key(self) -> None:
        target = {"steps": [{"name": "old", "keep": True}]}

        recursive_update(target, {"steps": {0: {"name": "new"}}})

        self.assertEqual(target["steps"], [{"name": "new", "keep": True}])

    def test_replaces_entire_list_with_slice_dsl(self) -> None:
        target = {"items": [1, 2, 3]}

        recursive_update(target, {"items": {"[:]": [4, 5]}})

        self.assertEqual(target["items"], [4, 5])

    def test_appends_then_updates_current_last_item(self) -> None:
        target = {"items": [1, 2, 3]}

        recursive_update(target, {"items": {"[]": [4, 5], -1: 10}})

        self.assertEqual(target["items"], [1, 2, 3, 4, 10])

    def test_deletes_and_inserts_list_slices_in_order(self) -> None:
        target = {"items": [1, 2, 3, 4]}

        recursive_update(
            target,
            {"items": {"[1:3]": [], "[1:1]": [8, 9]}},
        )

        self.assertEqual(target["items"], [1, 8, 9, 4])

    def test_null_is_a_value_not_a_delete_operation(self) -> None:
        target = {"value": 3, "other": 4}

        recursive_update(target, {"value": None})

        self.assertEqual(target, {"value": None, "other": 4})


class MergeFilesTest(unittest.TestCase):
    def test_yaml_aliases_and_same_value_reassertion_have_final_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "70_base.yaml").write_text(
                "defaults: &defaults\n  value: 4\ncopy: *defaults\n",
                encoding="utf-8",
            )
            (root / "80_override.yaml").write_text(
                "copy:\n  value: 4\n",
                encoding="utf-8",
            )

            merged, origins, changes = merge_files(sorted(root.glob("*.yaml")))

        self.assertEqual(merged["copy"]["value"], 4)
        self.assertEqual(origins["copy.value"], "80_override.yaml")
        self.assertTrue(
            any(
                row["path"] == "copy.value" and row["kind"] == "reasserted"
                for row in changes
            )
        )

    def test_multiple_steps_are_preserved_and_index_patch_targets_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "reduce": {
                    "steps": [
                        {"name": "citlali", "config": {"low_level": {"runtime": {"n_threads": 4}}}},
                        {"name": "after", "config": {"enabled": True}},
                    ]
                }
            }
            patch = {"reduce": {"steps": {1: {"config": {"enabled": False}}}}}
            (root / "70_base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
            (root / "80_override.yaml").write_text(yaml.safe_dump(patch), encoding="utf-8")

            merged, _, _ = merge_files(sorted(root.glob("*.yaml")))

        self.assertEqual(len(merged["reduce"]["steps"]), 2)
        self.assertFalse(merged["reduce"]["steps"][1]["config"]["enabled"])
        self.assertEqual(extract_low_level(merged)["runtime"]["n_threads"], 4)


class ModeKitValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_root = cls.repo_root / "config/tolteca"
        cls.manifest = cls.config_root / "manifest.yaml"
        cls.contract = cls.repo_root / "tools/config/config_leaf_contract_resolved.json"

    def test_checked_mode_kits_match_manifest_and_leaf_contract(self) -> None:
        reports = validate_modes(
            self.config_root,
            self.manifest,
            self.contract,
            list(MODE_REDUCTION_TYPES),
        )

        self.assertEqual([report["mode"] for report in reports], list(MODE_REDUCTION_TYPES))
        self.assertTrue(
            all(report["valid"] for report in reports),
            json.dumps({report["mode"]: report["errors"] for report in reports}, indent=2),
        )

    def test_unknown_low_level_leaf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mode_dir = root / "point"
            mode_dir.mkdir()
            base = {
                "reduce": {
                    "steps": [
                        {
                            "config": {
                                "low_level": {
                                    "runtime": {"reduction_type": "pointing"},
                                    "not_a_real_domain": {"enabled": True},
                                }
                            }
                        }
                    ]
                }
            }
            for filename in (
                "70_pipeline.yaml",
                "71_runtime.yaml",
                "72_observation.yaml",
                "80_products.yaml",
                "90_user_overrides.yaml",
            ):
                value = base if filename == "70_pipeline.yaml" else {}
                (mode_dir / filename).write_text(yaml.safe_dump(value), encoding="utf-8")
            report = build_report(
                "point",
                mode_dir,
                {"policy_sha256": "intentionally-wrong"},
                {},
            )

        self.assertFalse(report["valid"])
        self.assertIn("not_a_real_domain.enabled", report["unknown_leaves"])

    def test_deliberate_expert_override_is_reported_when_drift_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mode_dir = Path(tmp) / "point"
            mode_dir.mkdir()
            policy = {
                "runtime": {"output_dir": ".", "reduction_type": "pointing"},
                "mapmaking": {"enabled": True},
            }
            base = {
                "reduce": {
                    "steps": [{"config": {"low_level": policy}}],
                }
            }
            override = {
                "reduce": {
                    "steps": {0: {"config": {"low_level": {"mapmaking": {"enabled": False}}}}}
                }
            }
            values = {
                "70_pipeline.yaml": base,
                "71_runtime.yaml": {},
                "72_observation.yaml": {},
                "80_products.yaml": {},
                "90_user_overrides.yaml": override,
            }
            for filename, value in values.items():
                (mode_dir / filename).write_text(yaml.safe_dump(value), encoding="utf-8")
            contract = {
                "runtime.output_dir": {"authority": "runtime", "owner": "runtime-owner"},
                "runtime.reduction_type": {"authority": "runtime", "owner": "runtime-owner"},
                "mapmaking.enabled": {"authority": "mapmaking", "owner": "mapmaking-owner"},
            }

            report = build_report(
                "point",
                mode_dir,
                {"policy_sha256": policy_sha256(policy)},
                contract,
                allow_policy_drift=True,
            )

        self.assertTrue(report["valid"])
        self.assertFalse(report["policy_matches_manifest"])
        self.assertEqual(len(report["expert_override_changes"]), 1)
        self.assertEqual(
            report["expert_override_changes"][0]["config_path"],
            "mapmaking.enabled",
        )
        self.assertEqual(report["expert_override_changes"][0]["authority"], "mapmaking")


class ScienceAuthoringPrototypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.prototype_root = cls.repo_root / "config/tolteca/v2"
        cls.science_dir = cls.prototype_root / "science"
        cls.manifest = cls.prototype_root / "manifest.yaml"
        cls.contract = cls.repo_root / "tools/config/config_leaf_contract_resolved.json"
        cls.rules = yaml.safe_load(
            (cls.repo_root / "tools/config/config_key_classification.yaml").read_text()
        )

    def classification(self, path: str) -> str:
        for rule in self.rules["rules"]:
            if fnmatch.fnmatchcase(path, rule["pattern"]):
                return rule["classification"]
        return self.rules["fallback"]["classification"]

    def test_science_prototype_matches_accepted_policy(self) -> None:
        report = validate_modes(
            self.prototype_root,
            self.manifest,
            self.contract,
            ["science"],
        )[0]

        self.assertTrue(report["valid"], json.dumps(report["errors"], indent=2))
        self.assertTrue(report["policy_matches_manifest"])
        self.assertEqual(report["leaf_count"], 404)
        self.assertEqual(
            report["policy_sha256"],
            "10095418b09100f15c90af173ee34ea7bfcf12260cec41d80f43f6f50473a347",
        )

    def test_operator_files_contain_only_user_facing_low_level_leaves(self) -> None:
        expected_counts = {
            "71_science_runtime.yaml": 4,
            "81_science_defaults.yaml": 27,
            "82_science_products.yaml": 30,
        }
        for filename, expected_count in expected_counts.items():
            patch = yaml.safe_load((self.science_dir / filename).read_text())
            leaves = flatten_leaves(extract_low_level(patch))
            classifications = {
                normalized_path(path): self.classification(normalized_path(path))
                for path in leaves
            }
            self.assertEqual(len(leaves), expected_count, filename)
            self.assertEqual(
                {"user-facing"},
                set(classifications.values()),
                f"{filename}: {classifications}",
            )

    def test_fruit_loop_controls_are_consolidated_in_analysis_defaults(self) -> None:
        defaults = extract_low_level(
            yaml.safe_load(
                (self.science_dir / "81_science_defaults.yaml").read_text()
            )
        )
        products = extract_low_level(
            yaml.safe_load(
                (self.science_dir / "82_science_products.yaml").read_text()
            )
        )

        self.assertEqual(
            set(defaults["timestream"]["fruit_loops"]),
            {
                "enabled",
                "max_iters",
                "sig2noise_limit",
                "array_flux_limit",
                "save_all_iters",
            },
        )
        self.assertNotIn("fruit_loops", products.get("timestream", {}))

    def test_generated_data_bindings_belong_to_observation_file(self) -> None:
        runtime = yaml.safe_load(
            (self.science_dir / "71_science_runtime.yaml").read_text()
        )
        observation = yaml.safe_load(
            (self.science_dir / "72_science_observation.yaml").read_text()
        )

        self.assertNotIn("inputs", runtime["reduce"])
        runtime_low_level = extract_low_level(runtime)
        self.assertNotIn("kids", runtime_low_level)
        self.assertEqual(observation["reduce"]["inputs"][0]["path"], "../data")
        self.assertEqual(
            extract_low_level(observation)["kids"]["solver"]["fitreportdir"],
            "../data",
        )

    def test_normal_operator_surface_remains_bounded(self) -> None:
        line_limits = {
            "71_science_runtime.yaml": 30,
            "81_science_defaults.yaml": 90,
            "82_science_products.yaml": 80,
        }
        for filename, limit in line_limits.items():
            line_count = len((self.science_dir / filename).read_text().splitlines())
            self.assertLessEqual(line_count, limit, filename)

    def test_checked_prototype_is_generator_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated_root = Path(tmp) / "v2"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        self.repo_root
                        / "tools/config/generate_science_authoring_prototype.py"
                    ),
                    "--source-root",
                    str(self.repo_root / "config/tolteca"),
                    "--output-root",
                    str(generated_root),
                ],
                check=True,
                cwd=self.repo_root,
            )
            expected_files = [
                path.relative_to(self.prototype_root)
                for path in self.prototype_root.rglob("*.yaml")
            ]
            for relative_path in expected_files:
                self.assertEqual(
                    (self.prototype_root / relative_path).read_bytes(),
                    (generated_root / relative_path).read_bytes(),
                    str(relative_path),
                )


if __name__ == "__main__":
    unittest.main()
