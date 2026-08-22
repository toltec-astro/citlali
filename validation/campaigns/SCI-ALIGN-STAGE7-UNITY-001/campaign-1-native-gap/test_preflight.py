from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[3]
PREFLIGHT_PATH = PACKAGE_DIR / "preflight.py"
SPEC = importlib.util.spec_from_file_location("stage7_native_gap_preflight", PREFLIGHT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class NativeGapCampaignPreflightTest(unittest.TestCase):
    def test_canonical_science_kit_plus_overlay_passes(self) -> None:
        merged, report = PREFLIGHT.validate(
            REPO_ROOT / "config/tolteca/v2/science",
            PACKAGE_DIR / PREFLIGHT.OVERLAY_NAME,
        )
        self.assertEqual(
            report["status"], "config_ready_observation_checks_pending"
        )
        self.assertEqual(report["campaign_overlay_leaf_count"], 26)
        self.assertEqual(
            merged["reduce"]["steps"][0]["config"]["low_level"]["mapmaking"]["method"],
            "naive",
        )

    def test_overlay_must_be_final_numbered_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mode_dir = Path(temporary)
            for source in (REPO_ROOT / "config/tolteca/v2/science").glob("*.yaml"):
                shutil.copy2(source, mode_dir / source.name)
            overlay = mode_dir / PREFLIGHT.OVERLAY_NAME
            shutil.copy2(PACKAGE_DIR / PREFLIGHT.OVERLAY_NAME, overlay)
            (mode_dir / "99_zzzz_after_campaign.yaml").write_text(
                "reduce: {}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(PREFLIGHT.PreflightError, "not the final"):
                PREFLIGHT.validate(mode_dir, overlay)

    def test_overlay_cannot_own_an_unreviewed_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            overlay = Path(temporary) / PREFLIGHT.OVERLAY_NAME
            document = yaml.safe_load(
                (PACKAGE_DIR / PREFLIGHT.OVERLAY_NAME).read_text(encoding="utf-8")
            )
            document["reduce"]["steps"][0]["config"]["low_level"]["mapmaking"][
                "coverage_cut"
            ] = 0.1
            overlay.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(PREFLIGHT.PreflightError, "unexpected leaves"):
                PREFLIGHT.validate(REPO_ROOT / "config/tolteca/v2/science", overlay)

    def test_overlay_cannot_reenable_rejected_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            overlay = Path(temporary) / PREFLIGHT.OVERLAY_NAME
            document = yaml.safe_load(
                (PACKAGE_DIR / PREFLIGHT.OVERLAY_NAME).read_text(encoding="utf-8")
            )
            document["reduce"]["steps"][0]["config"]["low_level"]["timestream"][
                "learning"
            ]["enabled"] = True
            overlay.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(PREFLIGHT.PreflightError, "learning.enabled"):
                PREFLIGHT.validate(REPO_ROOT / "config/tolteca/v2/science", overlay)


if __name__ == "__main__":
    unittest.main()
