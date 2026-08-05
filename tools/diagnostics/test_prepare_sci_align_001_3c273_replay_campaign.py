#!/usr/bin/env python3
"""Focused contracts for SCI-ALIGN's numbered-config replay campaign."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from tools.diagnostics.prepare_sci_align_001_3c273_replay_campaign import (
    CAMPAIGN,
    CampaignError,
    authority,
    direct_config,
    observation_inputs,
    selected_campaign,
)


SOURCE = Path("/Users/gwilson/work_toltec/local_data/beammaps/3c273")


class NumberedCampaignTest(unittest.TestCase):
    def test_numbered_authority_has_every_campaign_photometry_entry(self) -> None:
        low_level, astrometry, photometry = authority(SOURCE / "70_reduce.yaml", SOURCE / "72_reduce.yaml")
        self.assertIn("beammap", low_level)
        self.assertEqual(astrometry["type"], "astrometry")
        self.assertEqual([obsnum for obsnum, _batch in CAMPAIGN], [
            113862, 131925, 136279, 152882, 128588, 133543, 150819, 152451,
            129687, 134643, 151126, 151950, 130922, 135397, 151600,
        ])
        self.assertTrue(all(obsnum in photometry for obsnum, _batch in CAMPAIGN))
        self.assertNotIn("select", photometry[113862])

    def test_input_discovery_requires_one_scannum_two_raw_file_per_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reduced").mkdir()
            (root / "reduced" / "tel_toltec_2024-03-23_113862_00_0002_recomputed.nc").write_text("tel")
            (root / "reduced" / "apt_113862_000_0002_2024_03_23.ecsv").write_text("apt")
            raw = root / "raw"
            raw.mkdir()
            (raw / "toltec0_113862_000_0002_2024_03_23.nc").write_text("raw")
            values = observation_inputs(root, raw, 113862)
            self.assertEqual(sorted(values), ["matched_input_apt", "telescope", "toltec0"])
            (raw / "toltec0_113862_000_0002_duplicate.nc").write_text("raw")
            with self.assertRaisesRegex(CampaignError, "ambiguous raw network 0"):
                observation_inputs(root, raw, 113862)

    def test_explicit_subset_preserves_declared_batch_and_rejects_unknown_members(self) -> None:
        self.assertEqual(selected_campaign([113862]), ((113862, 1),))
        self.assertEqual(selected_campaign([152451, 113862]), ((113862, 1), (152451, 2)))
        with self.assertRaisesRegex(CampaignError, "at most once"):
            selected_campaign([113862, 113862])
        with self.assertRaisesRegex(CampaignError, "not a campaign member"):
            selected_campaign([148670])

    def test_direct_config_preserves_numbered_policy_and_binds_diagnostic_outputs(self) -> None:
        low_level, astrometry, photometry = authority(SOURCE / "70_reduce.yaml", SOURCE / "72_reduce.yaml")
        inputs = {
            "matched_input_apt": Path("/data/apt.ecsv"),
            "telescope": Path("/data/tel.nc"),
            "toltec0": Path("/data/toltec0_113862_000_0002_x.nc"),
        }
        config = direct_config(
            low_level=low_level, astrometry=astrometry, photometry=photometry[113862], inputs=inputs,
            prior=Path("/repo/data/beammap_priors/beammap_slot_priors_soft_v1.ecsv"), fitreport_root=Path("/data"),
            output=Path("/output/reduced"), obsnum=113862, threads=6,
        )
        self.assertTrue(config["beammap"]["detector_tod_output"]["enabled"])
        self.assertEqual(config["beammap"]["detector_tod_output"]["subdir_name"], "source_crossing_tod")
        self.assertEqual(config["kids"]["solver"]["fitreportdir"], "/data")
        self.assertEqual(config["runtime"]["output_dir"], "/output/reduced")
        self.assertTrue(config["runtime"]["crop_detector_to_telescope_support"])
        self.assertEqual(config["mapmaking"]["grouping"], "detector")
        self.assertEqual(config["inputs"][0]["cal_items"][1]["type"], "photometry")
        self.assertNotIn("select", config["inputs"][0]["cal_items"][1])
        self.assertEqual(yaml.safe_load(yaml.safe_dump(config)), config)


if __name__ == "__main__":
    unittest.main()
