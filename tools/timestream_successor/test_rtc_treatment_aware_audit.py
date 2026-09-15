import unittest
import tempfile
from pathlib import Path
from rtc_disturbance_burden import digest

import numpy as np
from prepare_rtc_population_replay import cross_spectrum
from run_rtc_notch_recovery import finite_trial

from rtc_treatment_aware_audit import (
    bool_runs,
    erode,
    fixed_weight,
    rows_for_intervals,
    simultaneous,
    split_runs,
    sealed_verifier,
)


class PopulationAccounting(unittest.TestCase):
    def test_reused_weight_and_native_cell_files_require_the_accepted_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights, cells = root / "weights.json", root / "cells.json"
            weights.write_text('{"sens": 3}')
            cells.write_text("[[0, 1, 2, 3]]")
            seal = root / "SUMS"
            seal.write_text(
                "".join(f"{digest(p)}  {p.name}\n" for p in (weights, cells))
            )
            accepted = digest(seal)
            verify, checked = sealed_verifier(root, "SUMS", accepted)
            verify(weights)
            verify(cells)
            self.assertEqual(len(checked), 3)
            for path in (weights, cells):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    verify(path)
                path.write_bytes(original)
            seal.write_text(seal.read_text() + "\n")
            with self.assertRaises(ValueError):
                sealed_verifier(root, "SUMS", accepted)

    def test_matched_coherence_known_phase_and_scale(self):
        rng = np.random.default_rng(831)
        left = rng.normal(size=(80, 12)) + 1j * rng.normal(size=(80, 12))
        coh, phase = cross_spectrum(left, 2j * left)
        np.testing.assert_allclose(coh, 1, atol=2e-15)
        np.testing.assert_allclose(phase, -np.pi / 2, atol=2e-15)
        with self.assertRaises(ValueError):
            cross_spectrum(left[:2], left[:2])
        with self.assertRaises(ValueError):
            cross_spectrum(left, np.zeros_like(left))

    def test_existing_finite_family_has_bound_complete_support(self):
        trial = finite_trial(
            0.008192062377929688, 10.756113666291341, 6, [1 / 307] * 307
        )
        self.assertEqual(len(trial["finite_notch"]), 733)
        self.assertAlmostEqual(
            trial["cumulative_half_seconds"], 519 * 0.008192062377929688
        )
        self.assertAlmostEqual(sum(trial["finite_notch"]), 1, places=13)
        self.assertEqual(trial["finite_notch"], trial["finite_notch"][::-1])

    def test_complete_support_matches_explicit_neighborhood(self):
        admitted = np.ones(30, dtype=bool)
        admitted[[7, 8, 19]] = False
        for half in (0, 1, 3, 6):
            spans = erode(bool_runs(admitted), half)
            actual = [i for a, b in spans for i in range(a, b)]
            expected = [
                i
                for i in range(half, len(admitted) - half)
                if admitted[i - half : i + half + 1].all()
            ]
            self.assertEqual(actual, expected)

    def test_physical_gap_never_becomes_elapsed_support(self):
        # Row-adjacent runs have a physical acquisition break between them.
        self.assertEqual(erode([(0, 10), (10, 20)], 2), [(2, 8), (12, 18)])
        self.assertEqual(erode([(0, 4)], 2), [])
        self.assertEqual(erode([(0, 5)], 2), [(2, 3)])
        self.assertEqual(
            split_runs([(0, 10), (10, 20)], [(0, 20)]), [(0, 10), (10, 20)]
        )

    def test_overlap_counts_each_detector_once(self):
        value = simultaneous([[(0, 4), (2, 6)], [(3, 7)]])
        self.assertEqual(value, dict(maximum=2, duration_us_by_count={1: 4, 2: 3}))
        self.assertEqual(split_runs([(0, 10)], [(2, 4), (6, 8)]), [(2, 4), (6, 8)])

    def test_exact_native_cells_and_partial_cell_rejection(self):
        cells = np.array([[0, 2, 0, 4], [1, 6, 4, 8], [2, 22, 20, 24]])
        self.assertEqual(
            rows_for_intervals([(0, 8), (20, 24)], cells), [(0, 2), (2, 3)]
        )
        with self.assertRaises(ValueError):
            rows_for_intervals([(1, 8)], cells)

    def test_weights_are_common_reference_and_unavailable_is_not_zero(self):
        self.assertEqual(fixed_weight(True, 2), 0.25)
        self.assertEqual(fixed_weight(True, 4), 0.0625)
        for v in (None, 0, -1, np.nan, np.inf):
            self.assertIsNone(fixed_weight(True, v))
        self.assertIsNone(fixed_weight(False, 2))

    def test_no_negative_or_fractional_footprint(self):
        for half in (-1, 0.5):
            with self.assertRaises(ValueError):
                erode([(0, 10)], half)


if __name__ == "__main__":
    unittest.main()
