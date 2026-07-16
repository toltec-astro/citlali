from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.config.tolteca_mode_kit import (
    MODE_REDUCTION_TYPES,
    build_report,
    extract_low_level,
    merge_files,
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


if __name__ == "__main__":
    unittest.main()
