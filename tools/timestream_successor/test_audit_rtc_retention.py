"""Independent checks for the evidence-only footprint/overlap census."""

import unittest

import numpy as np

from audit_rtc_retention import NAMES, counts, support_bits


class FootprintAccounting(unittest.TestCase):
    def test_exact_inclusive_extent_and_fixed_phase(self):
        masks = np.zeros((9, 31), bool)
        masks[3, 15] = True
        bits = support_bits(masks, 2, [(0, 31)])
        self.assertEqual(np.flatnonzero(bits & (1 << 3)).tolist(), [13, 14, 15, 16, 17])
        self.assertEqual(
            np.flatnonzero((bits == 0) & (np.arange(31) % 2 == 0)).tolist(),
            [2, 4, 6, 8, 10, 12, 18, 20, 22, 24, 26, 28],
        )

    def test_overlaps_are_unioned_and_remain_identifiable(self):
        masks = np.zeros((9, 31), bool)
        masks[3, 14] = True
        masks[7, 16] = True
        bits = support_bits(masks, 2, [(0, 31)])[2:-2]
        result = counts(bits)
        self.assertEqual(result["union"], 7)
        self.assertEqual(result["shared_multiple_causes"], 3)
        self.assertEqual(
            result["by_cause"]["pending_event"], {"inclusive": 5, "only_this_cause": 2}
        )
        self.assertEqual(
            result["by_cause"]["above_F2_sampling_speed"],
            {"inclusive": 5, "only_this_cause": 2},
        )
        self.assertEqual(sum(result["disjoint_combinations"].values()), 7)

    def test_adjacent_indices_do_not_bridge_physical_runs(self):
        masks = np.zeros((9, 30), bool)
        bits = support_bits(masks, 3, [(0, 15), (15, 30)])
        self.assertTrue(np.all(bits[12:18] != 0))
        self.assertEqual(np.count_nonzero(bits == 0), 18)

    def test_short_run_has_no_complete_footprint(self):
        masks = np.zeros((9, 13), bool)
        self.assertEqual(
            np.count_nonzero(support_bits(masks, 3, [(0, 6), (6, 13)]) == 0), 1
        )

    def test_composed_firs_match_literal_nested_support(self):
        masks = np.zeros((9, 1600), bool)
        masks[3, 770:777] = True
        masks[7, 1200:1210] = True
        notch_half, lp_half = 183, 153
        result = support_bits(masks, notch_half + lp_half, [(0, 1600)]) == 0
        admitted = ~masks.any(axis=0)
        notch = np.zeros(1600, bool)
        for q in range(notch_half, 1600 - notch_half):
            notch[q] = all(admitted[q - notch_half : q + notch_half + 1])
        composed = np.zeros(1600, bool)
        for q in range(lp_half, 1600 - lp_half):
            composed[q] = all(notch[q - lp_half : q + lp_half + 1])
        np.testing.assert_array_equal(result, composed)

    def test_direct_and_expanded_populations_do_not_double_count(self):
        masks = np.zeros((9, 50), bool)
        masks[1, 20:23] = True
        masks[4, 21:25] = True
        direct = support_bits(masks, 0, [(0, 50)])
        final = support_bits(masks, 3, [(0, 50)])
        self.assertEqual(
            np.count_nonzero(direct) + np.count_nonzero(final[direct == 0]),
            np.count_nonzero(final),
        )
        self.assertEqual(len(NAMES), 10)


if __name__ == "__main__":
    unittest.main()
