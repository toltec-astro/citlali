#!/usr/bin/env python3

from __future__ import annotations

import copy
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.table import Table


sys.path.insert(0, str(Path(__file__).resolve().parent))

import sci_align_001_lissajous_crossings as crossings  # noqa: E402
import sci_align_001_lissajous_event_centroids as target  # noqa: E402
from test_analyze_sci_align_001_lissajous_timestream import (  # noqa: E402
    synthetic_observation,
)
from test_sci_align_001_lissajous_crossings import (  # noqa: E402
    two_pass_observation,
)


ROOT = Path(__file__).resolve().parents[2] / (
    "validation/sci_align_001_lissajous_timestream_2026-08-10"
)
CROSSING = ROOT / "crossing_support_protocol.json"
CENTROID = ROOT / "event_centroid_protocol.json"


def protocol() -> dict:
    result = target.load_event_centroid_protocol(CENTROID, CROSSING)
    result = copy.deepcopy(result)
    result["centroid_estimator"]["minimum_qualified_detectors"] = 5
    return result


class EventCentroidTest(unittest.TestCase):
    def test_overlapping_windows_are_partitioned_between_passages(self) -> None:
        observation = two_pass_observation()
        crossing_protocol = crossings.load_crossing_protocol(CROSSING)
        events = crossings.catalog_crossing_events(
            observation, crossing_protocol
        )
        windows = target.effective_event_windows(events)
        first = windows["s00_uid1051_evt00"]
        second = windows["s00_uid1051_evt01"]
        self.assertEqual(first[1], second[0])

    def test_spatial_centroids_recover_injected_lag(self) -> None:
        observation = synthetic_observation(tau_sec=0.007)
        crossing_protocol = crossings.load_crossing_protocol(CROSSING)
        events = crossings.catalog_crossing_events(
            observation, crossing_protocol
        )
        rows = target.catalog_event_centroids(
            observation, events, protocol()
        )
        result = target.fit_centroid_models(rows, protocol(), 0.5)
        self.assertGreater(result["qualified_event_count"], 50)
        self.assertAlmostEqual(
            result["models"]["lag"]["tau_ms"], 7.0, delta=0.15
        )

    def test_flat_event_is_not_qualified_as_compact_source(self) -> None:
        observation = synthetic_observation(tau_sec=0.0)
        observation = replace(observation, scans=[
            replace(scan, residual_by_baseline={
                name: np.zeros_like(value)
                for name, value in scan.residual_by_baseline.items()
            })
            for scan in observation.scans
        ])
        crossing_protocol = crossings.load_crossing_protocol(CROSSING)
        events = crossings.catalog_crossing_events(
            observation, crossing_protocol
        )
        rows = target.catalog_event_centroids(
            observation, events, protocol()
        )
        self.assertEqual(
            int(np.count_nonzero(np.asarray(rows["quality_qualified"], bool))),
            0,
        )

    def test_robust_joint_model_recovers_injected_parameters(self) -> None:
        rng = np.random.default_rng(20260812)
        count = 240
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        ux = np.cos(angle)
        uy = np.sin(angle)
        speed = rng.uniform(15.0, 55.0, count)
        vx = ux * speed
        vy = uy * speed
        truth = {
            "x0_arcsec": 0.7,
            "y0_arcsec": -0.4,
            "tau_ms": 6.5,
            "h_az_arcsec": -0.8,
            "h_el_arcsec": 0.35,
        }
        measured = (
            ux * truth["x0_arcsec"] + uy * truth["y0_arcsec"]
            - speed / 1000.0 * truth["tau_ms"]
            + ux * np.sign(vx) * truth["h_az_arcsec"]
            + uy * np.sign(vy) * truth["h_el_arcsec"]
            + rng.normal(0.0, 0.035, count)
        )
        measured[::47] += rng.choice([-1.0, 1.0], measured[::47].size) * 5.0
        rows = Table({
            "event_id": [f"e{index}" for index in range(count)],
            "uid": 1000 + np.arange(count) % 40,
            "unit_x": ux,
            "unit_y": uy,
            "speed_arcsec_per_sec": speed,
            "velocity_x_arcsec_per_sec": vx,
            "velocity_y_arcsec_per_sec": vy,
            "scored_sample_count": np.full(count, 20),
            "peak_at_grid_boundary": np.zeros(count, dtype=bool),
            "peak_correlation": np.full(count, 0.9),
            "profiled_amplitude_native": np.ones(count),
            "peak_shift_arcsec": measured,
        })
        result = target.fit_centroid_models(rows, protocol(), 0.5)
        parameters = result["models"]["joint"]["parameters"]
        for name, expected in truth.items():
            tolerance = 0.40 if name == "tau_ms" else 0.03
            self.assertAlmostEqual(parameters[name], expected, delta=tolerance)
        self.assertAlmostEqual(
            result["models"]["joint"]["effective_base_weight"], 40.0
        )

    def test_fixed_tau_profile_contains_lag_minimum(self) -> None:
        rng = np.random.default_rng(4)
        count = 80
        angle = rng.uniform(0.0, 2.0 * math.pi, count)
        ux = np.cos(angle)
        uy = np.sin(angle)
        speed = rng.uniform(20.0, 50.0, count)
        tau = 5.0
        rows = Table({
            "event_id": [f"e{index}" for index in range(count)],
            "uid": 1000 + np.arange(count) % 20,
            "unit_x": ux,
            "unit_y": uy,
            "speed_arcsec_per_sec": speed,
            "velocity_x_arcsec_per_sec": ux * speed,
            "velocity_y_arcsec_per_sec": uy * speed,
            "scored_sample_count": np.full(count, 20),
            "peak_at_grid_boundary": np.zeros(count, dtype=bool),
            "peak_correlation": np.full(count, 0.9),
            "profiled_amplitude_native": np.ones(count),
            "peak_shift_arcsec": (
                0.2 * ux - 0.1 * uy - speed / 1000.0 * tau
                + rng.normal(0.0, 0.01, count)
            ),
        })
        fit = target.fit_centroid_models(rows, protocol(), 0.5)
        profile = target.robust_tau_profile(rows, protocol(), fit)
        best = min(profile, key=lambda row: row["objective"])
        self.assertEqual(best["tau_ms"], 5.0)


if __name__ == "__main__":
    unittest.main()
