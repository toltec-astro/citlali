#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from astropy.table import Table


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_sci_align_001_lissajous_event_fit_networks as target  # noqa: E402


PROTOCOL = Path(__file__).resolve().parents[2] / (
    "validation/sci_align_001_lissajous_timestream_2026-08-10/"
    "event_centroid_protocol.json"
)


def synthetic_rows() -> Table:
    rng = np.random.default_rng(20260812)
    rows = []
    for network, tau_ms in ((0, 8.192), (1, 5.0)):
        count = 180
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        speed = rng.uniform(18.0, 55.0, count)
        ux = np.cos(angle)
        uy = np.sin(angle)
        measured = (
            0.3 * ux - 0.2 * uy - speed / 1000.0 * tau_ms
            + rng.normal(0.0, 0.02, count)
        )
        for index in range(count):
            rows.append({
                "event_id": f"nw{network}_e{index}",
                "uid": network * 1000 + index % 50,
                "network": network,
                "unit_x": ux[index],
                "unit_y": uy[index],
                "speed_arcsec_per_sec": speed[index],
                "velocity_x_arcsec_per_sec": ux[index] * speed[index],
                "velocity_y_arcsec_per_sec": uy[index] * speed[index],
                "scored_sample_count": 20,
                "peak_at_grid_boundary": False,
                "peak_correlation": 0.9,
                "profiled_amplitude_native": 1.0,
                "peak_shift_arcsec": measured[index],
            })
    return Table(rows=rows)


class NetworkAuditTest(unittest.TestCase):
    def test_cadence_rounding_is_half_away_from_zero(self) -> None:
        self.assertEqual(target.cadence_index(4.096, 8.192), 1)
        self.assertEqual(target.cadence_index(-4.096, 8.192), -1)
        self.assertEqual(target.cadence_index(12.288, 8.192), 2)

    def test_network_fits_recover_independent_lags(self) -> None:
        protocol = json.loads(PROTOCOL.read_text())
        rows = target.fit_network_rows(
            synthetic_rows(), protocol, 123456, 7.0, 8.192
        )
        self.assertEqual([row["network"] for row in rows], [0, 1])
        self.assertTrue(all(row["status"] == "success" for row in rows))
        self.assertAlmostEqual(rows[0]["lag_tau_ms"], 8.192, delta=0.35)
        self.assertAlmostEqual(rows[1]["lag_tau_ms"], 5.0, delta=0.35)
        self.assertEqual(rows[0]["lag_nearest_cadence_index"], 1)
        self.assertEqual(rows[1]["lag_nearest_cadence_index"], 1)
        self.assertAlmostEqual(
            rows[0]["lag_minus_pooled_ms"],
            rows[0]["lag_tau_ms"] - 7.0,
        )

    def test_network_summary_preserves_network_identity(self) -> None:
        protocol = json.loads(PROTOCOL.read_text())
        rows = target.fit_network_rows(
            synthetic_rows(), protocol, 123456, 7.0, 8.192
        )
        summary = target.network_summary_rows(rows, 8.192)
        self.assertEqual([row["network"] for row in summary], [0, 1])
        self.assertEqual(summary[0]["successful_observation_count"], 1)
        self.assertEqual(summary[0]["lag_nearest_plus_one_fraction"], 1.0)

    def test_observation_rows_add_network_spread(self) -> None:
        protocol = json.loads(PROTOCOL.read_text())
        networks = target.fit_network_rows(
            synthetic_rows(), protocol, 123456, 7.0, 8.192
        )
        campaign = Table(rows=[{
            "obsnum": 123456,
            "status": "complete",
            "lag_tau_ms": 7.0,
            "joint_tau_ms": 6.0,
        }])
        rows = target.observation_rows(campaign, networks, 8.192)
        self.assertEqual(rows[0]["successful_network_count"], 2)
        self.assertGreater(rows[0]["network_lag_tau_range_ms"], 2.5)
        self.assertEqual(rows[0]["network_lag_nearest_plus_one_count"], 2)


if __name__ == "__main__":
    unittest.main()
