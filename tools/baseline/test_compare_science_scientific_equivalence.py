from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from tools.baseline import compare_science_scientific_equivalence as science


class ScienceScientificEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.baseline = root / "baseline"
        self.candidate = root / "candidate"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def write_map(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.full((4, 4), value), name="signal_I"),
            ]
        ).writeto(path)

    def test_separates_raw_and_filtered_map_metrics(self) -> None:
        self.write_map(self.baseline / "obs/raw/map.fits", 1.0)
        self.write_map(self.candidate / "obs/raw/map.fits", 1.0 + 1.0e-9)
        self.write_map(self.baseline / "coadded/filtered/map.fits", 1.0)
        self.write_map(self.candidate / "coadded/filtered/map.fits", 1.01)

        metrics = science.fits_metrics(self.baseline, self.candidate)

        self.assertTrue(metrics["fits_product_sets_exact"])
        self.assertEqual(metrics["raw_map_layer_count"], 1)
        self.assertEqual(metrics["filtered_map_layer_count"], 1)
        self.assertLess(metrics["raw_map_rms_relative_max"], 1.0e-8)
        self.assertAlmostEqual(
            metrics["filtered_map_rms_relative_max"], 0.01
        )

    def test_v2_profile_keeps_raw_bound_strict(self) -> None:
        profile = json.loads(
            Path(
                "validation/profiles/science_scientific_equivalence_v2.json"
            ).read_text()
        )
        metrics = {
            "fits_product_sets_exact": True,
            "netcdf_product_sets_exact": True,
            "integer_diagnostics_exact": True,
            "raw_map_rms_relative_max": 2.0e-8,
            "filtered_map_rms_relative_max": 0.01,
            "ptc_weight_rms_relative_max": 0.0,
            "detector_median_absolute_max": 0.0,
            "detector_median_fractional_max": 0.0,
            "other_diagnostic_rms_relative_max": 0.0,
        }

        failures = science.evaluate(metrics, profile)

        self.assertEqual(len(failures), 1)
        self.assertIn("raw_map_rms_relative_max", failures[0])


if __name__ == "__main__":
    unittest.main()
