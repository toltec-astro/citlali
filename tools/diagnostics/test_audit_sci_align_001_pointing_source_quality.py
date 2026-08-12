#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_sci_align_001_pointing_source_quality as target  # noqa: E402


PROTOCOL = {
    "morphology": {
        "maximum_fit_radius_arcsec": 30.0,
        "minimum_fit_radius_arcsec": 15.0,
        "display_half_width_arcsec": 35.0,
        "review_core_radius_fwhm": 0.75,
        "noise_inner_radius_fwhm": 1.5,
        "residual_smoothing_arcsec": 1.0,
        "component_sigma_threshold": 5.0,
        "component_peak_fraction_threshold": 0.1,
        "minimum_component_beam_fraction": 0.1,
        "secondary_peak_sigma_threshold": 5.0,
        "secondary_peak_fraction_threshold": 0.15,
    }
}


def synthetic_map(secondary_amplitude: float = 0.0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260812)
    x = np.arange(-45.0, 46.0)
    y = np.arange(-45.0, 46.0)
    xx, yy = np.meshgrid(x, y)
    theta = 0.45
    ct, st = np.cos(theta), np.sin(theta)
    u = ct * xx + st * yy
    v = -st * xx + ct * yy
    image = 10.0 * np.exp(-0.5 * ((u / 5.0) ** 2 + (v / 2.2) ** 2))
    image += secondary_amplitude * np.exp(
        -0.5 * (((xx - 18.0) / 2.5) ** 2 + ((yy + 4.0) / 2.5) ** 2)
    )
    image += 0.04 * rng.normal(size=image.shape)
    weight = np.ones(image.shape)
    ppt = {
        "amp": 10.0,
        "x_t": 0.0,
        "y_t": 0.0,
        "a_fwhm": target.FWHM_PER_SIGMA * 5.0,
        "b_fwhm": target.FWHM_PER_SIGMA * 2.2,
        "angle": theta,
        "sig2noise": 75.0,
    }
    return image, weight, x, y, ppt


class PointingSourceQualityTest(unittest.TestCase):
    def test_broad_elliptical_single_peak_is_not_excluded(self) -> None:
        image, weight, x, y, ppt = synthetic_map()
        metrics, _ = target.source_quality_metrics(
            image, weight, x, y, ppt, PROTOCOL
        )
        self.assertGreater(metrics["fit_axis_ratio"], 2.0)
        self.assertLess(metrics["strongest_positive_secondary_peak_fraction"], 0.05)
        self.assertEqual(metrics["positive_secondary_peak_count"], 0)

    def test_secondary_peak_is_surfaced_for_review(self) -> None:
        image, weight, x, y, ppt = synthetic_map(secondary_amplitude=4.0)
        metrics, _ = target.source_quality_metrics(
            image, weight, x, y, ppt, PROTOCOL
        )
        self.assertGreater(metrics["strongest_positive_secondary_peak_fraction"], 0.25)
        self.assertGreaterEqual(metrics["positive_secondary_peak_count"], 1)
        self.assertGreaterEqual(metrics["coherent_residual_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
