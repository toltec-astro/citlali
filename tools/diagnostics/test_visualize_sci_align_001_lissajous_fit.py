#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402
import visualize_sci_align_001_lissajous_fit as target  # noqa: E402
from test_analyze_sci_align_001_lissajous_timestream import (  # noqa: E402
    synthetic_observation,
)


class LissajousFitVisualizationTest(unittest.TestCase):
    def test_contiguous_true_segments_preserve_gaps(self) -> None:
        mask = np.asarray([False, True, True, False, True, False])
        self.assertEqual(target.contiguous_true_segments(mask), [(1, 3), (4, 5)])

    def test_support_digest_is_deterministic_and_mask_sensitive(self) -> None:
        observation = synthetic_observation()
        first = target.support_sha256(observation)
        second = target.support_sha256(observation)
        self.assertEqual(first, second)
        observation.scans[0].score_mask[0, 0] ^= True
        self.assertNotEqual(first, target.support_sha256(observation))

    def test_model_components_reconstruct_exact_profiled_objective(self) -> None:
        observation = synthetic_observation(tau_sec=0.006)
        fit = analysis.fit_observation_model(observation, "lag")
        self.assertEqual(fit["status"], "success")
        total_sse = 0.0
        total_weight = 0.0
        for scan in observation.scans:
            components = target.model_components(
                scan, fit["parameters"], observation.beam
            )
            total_sse += float(np.sum(
                scan.ptc_weight[None, :]
                * np.where(scan.score_mask, components["residual"] ** 2, 0.0)
            ))
            total_weight += float(np.sum(
                scan.ptc_weight[None, :] * scan.score_mask
            ))
        expected = analysis.observation_objective(
            analysis.fit_to_optimizer_vector(fit, "lag", "fixed"),
            observation, "lag", "fixed", "constant",
        )
        self.assertAlmostEqual(total_sse / total_weight, expected, places=12)


if __name__ == "__main__":
    unittest.main()
