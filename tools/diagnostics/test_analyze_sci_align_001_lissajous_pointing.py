#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_lissajous_pointing as target  # noqa: E402


def sector_rows(
    tau: float = 0.0,
    h_az: float = 0.0,
    h_el: float = 0.0,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for sector in range(target.SECTOR_COUNT):
        angle = 2.0 * math.pi * sector / target.SECTOR_COUNT
        vx = 50.0 * math.cos(angle)
        vy = 50.0 * math.sin(angle)
        rows.append({
            "sector": sector,
            "status": "success",
            "x_arcsec": 1.25 + tau * vx + h_az * np.sign(vx),
            "y_arcsec": -0.75 + tau * vy + h_el * np.sign(vy),
            "velocity_x_arcsec_s": vx,
            "velocity_y_arcsec_s": vy,
        })
    return rows


class LissajousPointingDiagnosticTest(unittest.TestCase):
    def test_velocity_sector_boundaries_are_deterministic(self) -> None:
        width = 2.0 * math.pi / target.SECTOR_COUNT
        angles = np.asarray([
            0.0,
            0.49 * width,
            0.51 * width,
            -0.49 * width,
            -0.51 * width,
            math.pi,
        ])
        self.assertEqual(
            target.sector_index(angles).tolist(),
            [0, 0, 1, 0, 7, 4],
        )

    def test_time_lag_model_recovers_known_scalar_lag(self) -> None:
        results = {row["model"]: row for row in target.fit_models(
            sector_rows(tau=-0.0125)
        )}
        self.assertAlmostEqual(results["time_lag"]["tau_ms"], -12.5, places=10)
        self.assertLess(results["time_lag"]["rms_arcsec"], 1.0e-12)
        self.assertGreater(results["constant"]["rms_arcsec"], 0.3)

    def test_axis_sign_model_recovers_known_hysteresis(self) -> None:
        results = {row["model"]: row for row in target.fit_models(
            sector_rows(h_az=-1.2, h_el=0.4)
        )}
        self.assertAlmostEqual(results["axis_sign"]["h_az_arcsec"], -1.2)
        self.assertAlmostEqual(results["axis_sign"]["h_el_arcsec"], 0.4)
        self.assertLess(results["axis_sign"]["rms_arcsec"], 1.0e-12)

    def test_joint_model_recovers_both_components(self) -> None:
        results = {row["model"]: row for row in target.fit_models(
            sector_rows(tau=-0.009, h_az=-0.7, h_el=0.25)
        )}
        joint = results["joint"]
        self.assertAlmostEqual(joint["tau_ms"], -9.0, places=10)
        self.assertAlmostEqual(joint["h_az_arcsec"], -0.7, places=10)
        self.assertAlmostEqual(joint["h_el_arcsec"], 0.25, places=10)
        self.assertLess(joint["rms_arcsec"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
