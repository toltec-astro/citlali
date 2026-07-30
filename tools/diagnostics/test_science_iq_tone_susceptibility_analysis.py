import unittest

import numpy as np
import pandas as pd

from tools.diagnostics import science_iq_tone_susceptibility_analysis as analysis


class ScienceIqToneSusceptibilityAnalysisTests(unittest.TestCase):
    def test_fixed_tone_susceptibility_exceeds_event_preserving_null(self):
        rng = np.random.default_rng(10)
        opportunity = np.ones((40, 100), dtype=bool)
        probability = np.full(100, 0.08)
        probability[:10] = 0.85
        response = rng.random(opportunity.shape) < probability[None, :]
        frequency = np.linspace(0.4e9, 0.9e9, 100)
        observed, summary = analysis._permutation_summary(
            response,
            opportunity,
            frequency,
            frequency_bins=10,
            minimum_opportunities=20,
            top_fraction=0.2,
            n_permutations=99,
            rng=np.random.default_rng(11),
        )
        self.assertGreater(observed["tone_response_rate_variance"], 0.01)
        self.assertLessEqual(
            summary["tone_response_rate_variance"]["permutation_p"],
            0.02,
        )
        self.assertGreater(
            observed["split_half_response_rate_spearman"],
            0.25,
        )

    def test_frequency_band_susceptibility_exceeds_null(self):
        rng = np.random.default_rng(20)
        opportunity = np.ones((40, 120), dtype=bool)
        probability = np.full(120, 0.05)
        probability[40:80] = 0.70
        response = rng.random(opportunity.shape) < probability[None, :]
        frequency = np.linspace(0.4e9, 0.9e9, 120)
        _, summary = analysis._permutation_summary(
            response,
            opportunity,
            frequency,
            frequency_bins=12,
            minimum_opportunities=20,
            top_fraction=0.2,
            n_permutations=99,
            rng=np.random.default_rng(21),
        )
        self.assertLessEqual(
            summary["frequency_bin_rate_variance"]["permutation_p"],
            0.02,
        )

    def test_pair_coupling_reports_opposite_frequency_sign(self):
        rows = []
        for event in range(12):
            value = -1.0 if event % 3 else 1.0
            rows.extend(
                [
                    {
                        "event_cluster_id": f"c{event:03d}",
                        "network": 2,
                        "fit_status": "fit",
                        "combined_frequency_shift_hz": value,
                        "combined_phase_rad": 0.2 * value,
                        "combined_gain_fraction": 0.1 * value,
                    },
                    {
                        "event_cluster_id": f"c{event:03d}",
                        "network": 8,
                        "fit_status": "fit",
                        "combined_frequency_shift_hz": -2.0 * value,
                        "combined_phase_rad": -0.3 * value,
                        "combined_gain_fraction": -0.2 * value,
                    },
                ]
            )
        result = analysis._pair_coupling_rows(
            pd.DataFrame(rows),
            networks=[2, 8],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["frequency_like_paired_event_count"], 12)
        self.assertAlmostEqual(
            result[0]["frequency_like_opposite_sign_fraction"],
            1.0,
        )
        self.assertAlmostEqual(result[0]["frequency_like_spearman"], -1.0)

    def test_delay_model_recovers_phase_slope(self):
        offset = np.linspace(-200.0e6, 200.0e6, 41)
        delay_sec = 3.0e-12
        phase_rad = 0.012 + 2.0 * np.pi * offset * delay_sec
        frame = pd.DataFrame(
            {
                "event_cluster_id": "c001",
                "network": 2,
                "tone_offset_frequency_hz": offset,
                "phase_change_mrad": 1.0e3 * phase_rad,
                "phase_responsive": True,
            }
        )
        rows = analysis._delay_model_rows(frame, networks=[2])
        all_tones = next(
            row for row in rows if row["population"] == "all_model_valid"
        )
        self.assertAlmostEqual(
            all_tones["delay_equivalent_sec"],
            delay_sec,
        )
        self.assertAlmostEqual(
            all_tones["phase_plus_delay_zero_baseline_r2"],
            1.0,
        )

    def test_rank_one_mode_recovers_stable_tone_transfer(self):
        event_amplitude = np.linspace(-2.0, 3.0, 12)
        tone_loading = np.asarray([0.5, -1.0, 1.5, 2.0])
        rows = []
        for event_index, amplitude in enumerate(event_amplitude):
            for tone_index, loading in enumerate(tone_loading):
                phase = amplitude * loading
                rows.append(
                    {
                        "event_cluster_id": f"c{event_index:03d}",
                        "network": 2,
                        "uid": 100 + tone_index,
                        "tone_slot_zero_based": tone_index,
                        "tone_offset_frequency_hz": (
                            tone_index - 1.5
                        )
                        * 1.0e8,
                        "probe_frequency_hz": 7.0e8
                        + (tone_index - 1.5) * 1.0e8,
                        "phase_change_mrad": 1.0e3 * phase,
                        "fractional_change_real": 0.25 * phase,
                        "fractional_change_imag": phase,
                    }
                )
        summary, tones = analysis._rank_one_mode_rows(
            pd.DataFrame(rows),
            networks=[2],
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(tones), 4)
        self.assertAlmostEqual(summary[0]["phase_rank1_energy_fraction"], 1.0)
        self.assertAlmostEqual(
            summary[0]["complex_rank1_energy_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            summary[0]["phase_rank1_split_half_loading_cosine"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
