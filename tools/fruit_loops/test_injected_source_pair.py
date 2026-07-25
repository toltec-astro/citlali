from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from tools.fruit_loops import compare_injected_source_pair as compare


class InjectedSourcePairTest(unittest.TestCase):
    def test_discovers_absolute_iteration_from_fits_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for redu_number, fruit_iteration in ((0, 9), (1, 10)):
                product = compare.product_path(
                    root / f"redu{redu_number:02d}", 133410, "a1100"
                )
                product.parent.mkdir(parents=True)
                primary = fits.PrimaryHDU()
                primary.header["HIERARCH FRUITLOOPS_ITER"] = fruit_iteration
                primary.writeto(product)

            self.assertEqual(
                compare.iteration_dirs(root, 133410),
                {9: root / "redu00", 10: root / "redu01"},
            )

    def test_pair_contract_allows_only_output_and_enable_difference(self) -> None:
        manifest = {
            "start_iteration": 9,
            "array_amplitude_mjy_beam": [1000.0, 2000.0, 3000.0],
        }
        control = {
            "runtime": {"output_dir": "/control", "reduction_type": "pointing"},
            "mapmaking": {"pixel_size_arcsec": 1.0},
            "timestream": {
                "fruit_loops": {
                    "max_iters": 14,
                    "injected_source_test": {
                        "enabled": False,
                        "start_iteration": 9,
                        "array_amplitude_mjy_beam": [
                            1000.0,
                            2000.0,
                            3000.0,
                        ],
                    },
                }
            },
        }
        injected = copy.deepcopy(control)
        injected["runtime"]["output_dir"] = "/injected"
        injected["timestream"]["fruit_loops"]["injected_source_test"][
            "enabled"
        ] = True

        compare.require_pair_config(control, injected, manifest)

        injected["mapmaking"]["pixel_size_arcsec"] = 2.0
        with self.assertRaisesRegex(ValueError, "differ beyond"):
            compare.require_pair_config(control, injected, manifest)

    def test_recovers_known_gaussian_amplitude_width_and_center(self) -> None:
        yy, xx = np.indices((101, 101), dtype=float)
        xx -= 50.0
        yy -= 50.0
        amplitude = 3981.3
        sigma_x = 3.5
        sigma_y = 4.25
        x_center = 1.2
        y_center = -0.7
        values = 13.0 + amplitude * np.exp(
            -0.5
            * (
                np.square((xx - x_center) / sigma_x)
                + np.square((yy - y_center) / sigma_y)
            )
        )

        recovered = compare.gaussian_fit(values, 1.0)

        self.assertAlmostEqual(recovered["amplitude"], amplitude, delta=1.0e-3)
        self.assertAlmostEqual(recovered["x_arcsec"], x_center, delta=1.0e-4)
        self.assertAlmostEqual(recovered["y_arcsec"], y_center, delta=1.0e-4)
        self.assertAlmostEqual(
            recovered["major_fwhm_arcsec"],
            sigma_y * compare.FWHM_FACTOR,
            delta=1.0e-4,
        )
        self.assertAlmostEqual(
            recovered["minor_fwhm_arcsec"],
            sigma_x * compare.FWHM_FACTOR,
            delta=1.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
