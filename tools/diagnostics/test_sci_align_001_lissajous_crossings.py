#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path
import json

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_lissajous_timestream as analysis  # noqa: E402
import sci_align_001_lissajous_crossings as target  # noqa: E402
import render_sci_align_001_lissajous_event_fit_gate as renderer  # noqa: E402
from test_analyze_sci_align_001_lissajous_timestream import (  # noqa: E402
    synthetic_observation,
)


PROTOCOL = Path(__file__).resolve().parents[2] / (
    "validation/sci_align_001_lissajous_timestream_2026-08-10/"
    "crossing_support_protocol.json"
)


def two_pass_observation() -> analysis.PreparedObservation:
    """Make two close passages whose fixed fit windows overlap."""
    original = synthetic_observation().scans[0]
    time = np.linspace(0.0, 10.0, 101)
    x = np.concatenate([
        np.linspace(-10.0, 6.0, 51),
        np.linspace(6.0, -10.0, 51)[1:],
    ])
    y = np.zeros(time.size)
    vx = np.gradient(x, time)
    vy = np.gradient(y, time)
    shape = (time.size, 1)
    valid = np.ones(shape, dtype=bool)
    signal = np.exp(-0.5 * (x[:, None] / 2.0) ** 2)
    scan = replace(
        original,
        scan_row=0,
        output_scan_index=1,
        full_time=time,
        full_az=x / analysis.RAD_TO_ARCSEC,
        full_alt=y,
        full_elevation=np.zeros(time.size),
        full_pointing_az=np.zeros(time.size),
        full_pointing_alt=np.zeros(time.size),
        full_velocity_x=vx,
        full_velocity_y=vy,
        recorded_time=time,
        apt_x=np.array([0.0]),
        apt_y=np.array([0.0]),
        detector_uid=np.array([1051]),
        detector_network=np.array([1]),
        ptc_weight=np.array([1.0]),
        valid=valid,
        score_mask=valid.copy(),
        offsource_mask=np.abs(x[:, None]) >= 8.0,
        residual_by_baseline={
            name: signal.copy() for name in analysis.BASELINE_NAMES
        },
        baseline_coefficients={
            name: np.zeros((1, 1 if name == "constant" else 2))
            for name in analysis.BASELINE_NAMES
        },
        reference_x=x[:, None],
        reference_y=y[:, None],
    )
    return analysis.PreparedObservation(
        obsnum=150818,
        ptc_path=Path("synthetic.nc"),
        ppt_path=Path("synthetic.ecsv"),
        ppt_x_arcsec=0.0,
        ppt_y_arcsec=0.0,
        beam=analysis.BeamGeometry(10.0, 10.0, 0.0),
        scans=[scan],
        eligible_uid_count=1,
        eligible_networks=(1,),
        common_support_sample_count=time.size,
        scored_value_count=time.size,
        protocol=synthetic_observation().protocol,
    )


class CrossingSupportTest(unittest.TestCase):
    def test_campaign_identity_changes_do_not_change_fit_core(self) -> None:
        crossing_protocol = target.load_crossing_protocol(PROTOCOL)
        base_path = PROTOCOL.with_name("frozen_protocol.json")
        base = json.loads(base_path.read_text())
        campaign = json.loads(base_path.read_text())
        campaign["scope"] = {"pointing_count": 66}
        campaign["input_authority"] = {"selection_manifest_sha256": "new"}
        campaign["corpus"] = {"primary_sets": ["all_66"]}
        campaign["campaign"] = {"schema": "campaign"}
        target.authenticate_base_protocol(base, crossing_protocol)
        target.authenticate_base_protocol(campaign, crossing_protocol)
        campaign["source_model"]["primary_baseline"] = "changed"
        with self.assertRaises(target.CrossingContractError):
            target.authenticate_base_protocol(campaign, crossing_protocol)

    def test_true_blocks_are_half_open_and_separate(self) -> None:
        blocks = target.true_blocks(np.array([
            False, True, True, False, True, False,
        ]))
        self.assertEqual(blocks, [(1, 3), (4, 5)])

    def test_two_passages_remain_distinct_when_fit_windows_overlap(self) -> None:
        protocol = target.load_crossing_protocol(PROTOCOL)
        observation = two_pass_observation()
        events = target.catalog_crossing_events(observation, protocol)
        accepted = events[np.asarray(events["accepted"], dtype=bool)]

        self.assertEqual(len(accepted), 2)
        self.assertEqual(list(accepted["event_id"]), [
            "s00_uid1051_evt00", "s00_uid1051_evt01",
        ])
        self.assertGreater(
            int(accepted[0]["fit_window_stop_exclusive"]),
            int(accepted[1]["fit_window_start"]),
        )

        restricted, support = target.restrict_to_crossing_support(
            observation, events, protocol
        )
        self.assertEqual(len(restricted.scans), 1)
        self.assertEqual(len(support), 1)
        self.assertEqual(int(support[0]["uid"]), 1051)
        self.assertEqual(
            int(support[0]["scored_sample_count"]),
            int(np.count_nonzero(restricted.scans[0].score_mask)),
        )

    def test_event_membership_does_not_depend_on_detector_signal(self) -> None:
        protocol = target.load_crossing_protocol(PROTOCOL)
        original = two_pass_observation()
        first = target.catalog_crossing_events(original, protocol)
        changed_scan = replace(
            original.scans[0],
            residual_by_baseline={
                name: -1000.0 * value
                for name, value in original.scans[0].residual_by_baseline.items()
            },
        )
        changed = replace(original, scans=[changed_scan])
        second = target.catalog_crossing_events(changed, protocol)

        columns = [
            "event_id", "half_power_start", "half_power_stop_exclusive",
            "closest_sample", "fit_window_start",
            "fit_window_stop_exclusive", "accepted", "disposition",
        ]
        for column in columns:
            self.assertEqual(list(first[column]), list(second[column]))

    def test_edge_crossing_is_retained_as_rejected_evidence(self) -> None:
        protocol = target.load_crossing_protocol(PROTOCOL)
        observation = two_pass_observation()
        scan = observation.scans[0]
        x = np.linspace(0.0, 10.0, scan.recorded_time.size)
        changed = replace(
            scan,
            full_az=x / analysis.RAD_TO_ARCSEC,
            full_velocity_x=np.gradient(x, scan.full_time),
            reference_x=x[:, None],
        )
        events = target.catalog_crossing_events(
            replace(observation, scans=[changed]), protocol
        )
        self.assertEqual(len(events), 1)
        self.assertFalse(bool(events[0]["accepted"]))
        self.assertEqual(
            str(events[0]["disposition"]),
            "half_power_block_touches_scan_edge",
        )

    def test_review_selection_preserves_both_passages_of_first_pair(self) -> None:
        rows = []
        for event_id, scan_row, uid, event_index, leverage in (
            ("single", 0, 10, 0, 100.0),
            ("pair_a", 0, 20, 0, 1.0),
            ("pair_b", 0, 20, 1, 2.0),
            ("later_a", 1, 30, 0, 3.0),
            ("later_b", 1, 30, 1, 4.0),
        ):
            rows.append({
                "event_id": event_id,
                "scan_row": scan_row,
                "uid": uid,
                "event_index": event_index,
                "timing_leverage_proxy": leverage,
                "sqrt_weight_scaled_residual_rms": leverage,
                "local_data_model_correlation": 0.5,
            })
        chosen, document = renderer.deterministic_event_selection(rows, 4)
        self.assertIn("pair_a", chosen)
        self.assertIn("pair_b", chosen)
        by_id = {row["event_id"]: row for row in document["selected"]}
        reason = "first_multi_event_detector_scan_distinct_passage"
        self.assertIn(reason, by_id["pair_a"]["selection_reasons"])
        self.assertIn(reason, by_id["pair_b"]["selection_reasons"])


if __name__ == "__main__":
    unittest.main()
