#!/usr/bin/env python3
"""Tests for detector-source-aligned fit-gate review evidence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402
import render_sci_align_001_lissajous_fit_gate_source_review as target  # noqa: E402
import visualize_sci_align_001_lissajous_fit as visualization  # noqa: E402
from test_analyze_sci_align_001_lissajous_timestream import (  # noqa: E402
    synthetic_observation,
)


class SourceAlignedFitGateReviewTest(unittest.TestCase):
    def test_alignment_recovers_crossing_smeared_by_timestamp_average(self) -> None:
        time = np.linspace(-20.0, 20.0, 321)
        centers = np.asarray([-6.0, 6.0])
        profiles = np.exp(
            -0.5 * ((time[:, None] - centers[None, :]) / 2.0) ** 2
        )
        legacy_timestamp_average = np.mean(profiles, axis=1)
        coordinate = np.concatenate([
            time - centers[0], time - centers[1]
        ])
        values = np.concatenate([profiles[:, 0], profiles[:, 1]])
        edges = np.linspace(-20.0, 20.0, 161)
        aligned = target.binned_weighted_stack(
            coordinate, values, values, np.ones(values.size), edges
        )
        self.assertLess(np.max(legacy_timestamp_average), 0.51)
        self.assertGreater(np.nanmax(aligned["data_mean"]), 0.99)
        peak = aligned["center_arcsec"][np.nanargmax(aligned["data_mean"])]
        self.assertLess(abs(float(peak)), 0.2)

    def test_synthetic_crossings_align_and_preserve_exact_model(self) -> None:
        observation = synthetic_observation(tau_sec=0.006)
        parameters = {
            "x0_arcsec": 2.5,
            "y0_arcsec": -1.75,
            "tau_sec": 0.006,
        }
        components = [
            visualization.model_components(
                scan, parameters, observation.beam
            ) for scan in observation.scans
        ]
        samples, events = target.source_aligned_samples(
            observation, components, parameters
        )
        self.assertGreater(len(events), 20)
        self.assertEqual(samples["along_arcsec"].shape, samples["weight"].shape)
        self.assertTrue(np.all(np.isfinite(samples["normalized_model"])))
        edges = np.linspace(-20.0, 20.0, 81)
        stack = target.binned_weighted_stack(
            samples["along_arcsec"], samples["normalized_data"],
            samples["normalized_model"], samples["weight"], edges,
        )
        center = np.abs(stack["center_arcsec"]) <= 1.0
        wing = np.abs(stack["center_arcsec"]) >= 12.0
        self.assertGreater(
            np.nanmean(stack["model_mean"][center]),
            np.nanmean(stack["model_mean"][wing]) + 0.5,
        )
        self.assertLess(
            np.nanmedian(np.abs(
                stack["data_mean"][center] - stack["model_mean"][center]
            )),
            0.02,
        )

    def test_beam_radius_uses_elliptical_fwhm_geometry(self) -> None:
        beam = analysis.BeamGeometry(10.0, 5.0, 0.0)
        radius = target.beam_normalized_radius(
            np.asarray([5.0, 0.0]), np.asarray([0.0, 2.5]), beam
        )
        np.testing.assert_allclose(radius, [0.5, 0.5])

    def test_fixed_nuisance_profile_recovers_synthetic_lag(self) -> None:
        observation = synthetic_observation(tau_sec=0.006)
        primary = {
            "parameters": {
                "x0_arcsec": 2.5,
                "y0_arcsec": -1.75,
                "tau_sec": 0.006,
            }
        }
        rows = target.fixed_nuisance_tau_profile(observation, primary)
        best = min(rows, key=lambda row: row["objective"])
        self.assertEqual(len(rows), 41)
        self.assertAlmostEqual(best["tau_ms"], 5.0)
        self.assertLess(best["objective"], rows[20]["objective"])


if __name__ == "__main__":
    unittest.main()
