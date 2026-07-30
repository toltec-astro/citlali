#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.diagnostics.coherent_iq_mode_observer import (
    attach_cross_network_coincidence,
    make_template,
    score_event,
)


def synthetic_template(*, rank_two: bool = False) -> dict:
    uids = np.arange(100, 112)
    offset = np.linspace(-6.0e7, 6.0e7, uids.size)
    primary = np.linspace(-1.5, 1.5, uids.size)
    modes = [primary]
    if rank_two:
        modes.append(np.cos(np.linspace(0.0, 2.0 * np.pi, uids.size)))
    return make_template(
        template_id="synthetic-nw8",
        template_version="test-1",
        network=8,
        uids=uids,
        tone_slots=np.arange(uids.size),
        tone_offsets_hz=offset,
        probe_frequencies_hz=8.0e8 + offset,
        modes=np.asarray(modes),
        training={"dataset": "unit test"},
        validation={"status": "synthetic"},
        provenance={"source": "unit test"},
        tone_offset_tolerance_hz=1.0,
        minimum_compatible_tone_fraction=0.75,
        required_metadata={"firmware": "test-fw"},
    )


def template_arrays(template: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tones = template["tone_coordinate"]["tones"]
    return (
        np.asarray([row["uid"] for row in tones], dtype=int),
        np.asarray([row["tone_offset_frequency_hz"] for row in tones]),
        np.asarray(
            [row["loadings"]["phase_mode_1"] for row in tones], dtype=float
        ),
    )


class CoherentIqModeObserverTest(unittest.TestCase):
    def score(
        self,
        template: dict,
        phase: np.ndarray,
        *,
        network: int = 8,
        uids: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        template_uids, template_offsets, _ = template_arrays(template)
        return score_event(
            template,
            network=network,
            uids=template_uids if uids is None else uids,
            tone_offsets_hz=template_offsets if offsets is None else offsets,
            phase_change_mrad=phase,
            metadata={"firmware": "test-fw"} if metadata is None else metadata,
        )

    def test_rank_one_positive_and_negative_amplitudes(self) -> None:
        template = synthetic_template()
        _, _, mode = template_arrays(template)
        positive = self.score(template, 12.5 * mode)
        negative = self.score(template, -7.0 * mode)
        self.assertEqual(positive.status, "scored")
        self.assertAlmostEqual(positive.projection_amplitude_mrad, 12.5)
        self.assertEqual(positive.sign, 1)
        self.assertAlmostEqual(positive.absolute_cosine_similarity, 1.0)
        self.assertAlmostEqual(positive.explained_energy_fraction, 1.0)
        self.assertAlmostEqual(negative.projection_amplitude_mrad, -7.0)
        self.assertEqual(negative.sign, -1)
        self.assertAlmostEqual(negative.cosine_similarity, -1.0)

    def test_missing_tones_are_explicit_and_bounded(self) -> None:
        template = synthetic_template()
        uids, offsets, mode = template_arrays(template)
        keep = np.arange(uids.size) < 10
        score = self.score(
            template,
            4.0 * mode[keep],
            uids=uids[keep],
            offsets=offsets[keep],
        )
        self.assertEqual(score.status, "scored")
        self.assertEqual(score.compatible_tone_count, 10)
        self.assertAlmostEqual(score.compatible_tone_fraction, 10 / 12)
        too_few = keep.copy()
        too_few[8:] = False
        rejected = self.score(
            template,
            4.0 * mode[too_few],
            uids=uids[too_few],
            offsets=offsets[too_few],
        )
        self.assertEqual(rejected.status, "insufficient_compatible_tones")

    def test_reordered_uid_map_is_safe_but_incompatible_frequency_fails_closed(self) -> None:
        template = synthetic_template()
        uids, offsets, mode = template_arrays(template)
        order = np.arange(uids.size)[::-1]
        reordered = self.score(
            template,
            6.0 * mode[order],
            uids=uids[order],
            offsets=offsets[order],
        )
        self.assertEqual(reordered.status, "scored")
        self.assertAlmostEqual(reordered.absolute_cosine_similarity, 1.0)
        incompatible = self.score(
            template,
            6.0 * mode,
            offsets=offsets + 10.0,
        )
        self.assertEqual(incompatible.status, "insufficient_compatible_tones")

    def test_detector_local_common_phase_and_delay_slope_are_distinguished(self) -> None:
        template = synthetic_template()
        _, offsets, mode = template_arrays(template)
        local = np.zeros_like(mode)
        local[3] = 50.0
        local_score = self.score(template, local)
        self.assertLess(local_score.explained_energy_fraction, 0.4)

        common = np.full_like(mode, 15.0)
        common_score = self.score(template, common)
        self.assertAlmostEqual(
            common_score.common_phase_explained_energy_fraction, 1.0
        )
        self.assertGreater(
            common_score.common_phase_explained_energy_fraction,
            common_score.explained_energy_fraction,
        )

        slope = 3.0 + 20.0 * (offsets - offsets.mean()) / offsets.std()
        slope_score = self.score(template, slope)
        self.assertAlmostEqual(
            slope_score.delay_slope_explained_energy_fraction, 1.0
        )
        self.assertGreater(
            slope_score.delay_slope_explained_energy_fraction,
            slope_score.explained_energy_fraction,
        )

    def test_two_mode_event_reports_combined_explanation(self) -> None:
        template = synthetic_template(rank_two=True)
        tones = template["tone_coordinate"]["tones"]
        mode1 = np.asarray([row["loadings"]["phase_mode_1"] for row in tones])
        mode2 = np.asarray([row["loadings"]["phase_mode_2"] for row in tones])
        score = self.score(template, 8.0 * mode1 + 5.0 * mode2)
        self.assertGreater(score.multi_mode_explained_energy_fraction, 0.999)
        self.assertGreater(
            score.multi_mode_explained_energy_fraction,
            score.explained_energy_fraction,
        )

    def test_null_wrong_network_and_wrong_metadata(self) -> None:
        template = synthetic_template()
        _, _, mode = template_arrays(template)
        null = self.score(template, np.zeros_like(mode))
        self.assertEqual(null.status, "zero_event_energy")
        wrong_network = self.score(template, mode, network=9)
        self.assertEqual(wrong_network.status, "incompatible_network")
        wrong_metadata = self.score(
            template, mode, metadata={"firmware": "other"}
        )
        self.assertEqual(wrong_metadata.status, "incompatible_metadata")

    def test_scoring_is_non_mutating_and_deterministic(self) -> None:
        template = synthetic_template()
        uids, offsets, mode = template_arrays(template)
        template_before = copy.deepcopy(template)
        uids_before = uids.copy()
        offsets_before = offsets.copy()
        phase = 9.0 * mode
        phase_before = phase.copy()
        first = self.score(
            template, phase, uids=uids, offsets=offsets
        ).as_dict()
        second = self.score(
            template, phase, uids=uids, offsets=offsets
        ).as_dict()
        self.assertEqual(first, second)
        self.assertEqual(template, template_before)
        np.testing.assert_array_equal(uids, uids_before)
        np.testing.assert_array_equal(offsets, offsets_before)
        np.testing.assert_array_equal(phase, phase_before)

    def test_cross_network_coincidence_counts_only_selected_records(self) -> None:
        rows = [
            {
                "event_time_unix_sec": 10.0,
                "network": 1,
                "status": "scored",
                "selected": True,
            },
            {
                "event_time_unix_sec": 10.1,
                "network": 8,
                "status": "scored",
                "selected": True,
            },
            {
                "event_time_unix_sec": 10.05,
                "network": 9,
                "status": "scored",
                "selected": False,
            },
        ]
        attach_cross_network_coincidence(
            rows, tolerance_sec=0.2, selection_field="selected"
        )
        for row in rows:
            self.assertEqual(row["cross_network_coincident_count"], 2)
            self.assertEqual(row["cross_network_coincident_networks"], "1 8")


if __name__ == "__main__":
    unittest.main()
