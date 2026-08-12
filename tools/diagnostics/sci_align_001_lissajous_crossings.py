#!/usr/bin/env python3
"""Frozen geometric crossing support for SCI-ALIGN-001 Lissajous fits.

Crossing discovery is deliberately independent of a measured timing or
hysteresis parameter.  Events are defined at ``tau=0`` around the retained
PPT source center, then converted to fixed sample masks that every competing
model and every visualization must reuse.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


SCHEMA = "sci-align-001-lissajous-crossing-support-protocol-v1"


class CrossingContractError(RuntimeError):
    """A crossing-support input or realized event violates its contract."""


def load_crossing_protocol(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise CrossingContractError("unsupported crossing-support protocol")
    if document.get("status") != "frozen_before_successor_real_fit_inspection":
        raise CrossingContractError("crossing-support protocol is not frozen")
    support = document.get("support", {})
    required = {
        "discovery_tau_sec",
        "crossing_radius_elliptical_fwhm",
        "fit_half_window_elliptical_fwhm",
        "minimum_scored_samples_per_detector_scan",
    }
    if required - set(support):
        raise CrossingContractError("crossing-support protocol is incomplete")
    if float(support["discovery_tau_sec"]) != 0.0:
        raise CrossingContractError("crossing discovery must use tau=0")
    if not 0.0 < float(support["crossing_radius_elliptical_fwhm"]) <= 1.0:
        raise CrossingContractError("crossing radius is invalid")
    if float(support["fit_half_window_elliptical_fwhm"]) <= 0.0:
        raise CrossingContractError("crossing fit half-window is invalid")
    if int(support["minimum_scored_samples_per_detector_scan"]) < 3:
        raise CrossingContractError("minimum crossing sample count is invalid")
    return document


def base_protocol_core_sha256(document: dict[str, Any]) -> str:
    """Digest immutable fit arithmetic while allowing campaign identities.

    Campaign freezing legitimately replaces selection/input authority and
    descriptive scope/corpus records. Those fields do not define the
    preparation arithmetic, support geometry, source model, or optimizer.
    """
    core = {
        key: value for key, value in document.items()
        if key not in {"scope", "input_authority", "corpus", "campaign"}
    }
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def authenticate_base_protocol(
    document: dict[str, Any], crossing_protocol: dict[str, Any]
) -> None:
    expected = crossing_protocol["base_timestream_protocol"].get(
        "immutable_fit_core_sha256"
    )
    if not isinstance(expected, str):
        raise CrossingContractError("base protocol core identity is absent")
    actual = base_protocol_core_sha256(document)
    if actual != expected:
        raise CrossingContractError(
            f"base timestream fit-core identity changed: {actual} != {expected}"
        )


def true_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous true blocks in a one-dimensional mask."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise CrossingContractError("crossing mask must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def beam_normalized_radius(
    dx: np.ndarray,
    dy: np.ndarray,
    major_fwhm_arcsec: float,
    minor_fwhm_arcsec: float,
    angle_rad: float,
) -> np.ndarray:
    if major_fwhm_arcsec <= 0.0 or minor_fwhm_arcsec <= 0.0:
        raise CrossingContractError("beam FWHM must be positive")
    ct = math.cos(angle_rad)
    st = math.sin(angle_rad)
    major = ct * dx + st * dy
    minor = -st * dx + ct * dy
    return np.hypot(major / major_fwhm_arcsec, minor / minor_fwhm_arcsec)


def catalog_crossing_events(
    observation: Any,
    protocol: dict[str, Any],
) -> Table:
    """Catalog fixed tau-zero/PPT-centered crossing events.

    Every contiguous passage through the half-power ellipse is a separate
    event.  The catalog retains rejected events and a deterministic reason so
    visual review can account for the complete geometric census.
    """
    spec = protocol["support"]
    discovery_tau = float(spec["discovery_tau_sec"])
    radius_limit = float(spec["crossing_radius_elliptical_fwhm"])
    window_limit = float(spec["fit_half_window_elliptical_fwhm"])
    rows: list[dict[str, Any]] = []
    event_number = 0
    for scan in observation.scans:
        x, y, velocity_x, velocity_y = scan.coordinates(discovery_tau)
        dx = x - float(observation.ppt_x_arcsec)
        dy = y - float(observation.ppt_y_arcsec)
        rho = beam_normalized_radius(
            dx,
            dy,
            float(observation.beam.major_fwhm_arcsec),
            float(observation.beam.minor_fwhm_arcsec),
            float(observation.beam.angle_rad),
        )
        time = np.asarray(scan.recorded_time, dtype=float)
        q_major_ct = math.cos(float(observation.beam.angle_rad))
        q_major_st = math.sin(float(observation.beam.angle_rad))
        q_major = (
            q_major_ct * dx + q_major_st * dy
        ) / float(observation.beam.major_fwhm_arcsec)
        q_minor = (
            -q_major_st * dx + q_major_ct * dy
        ) / float(observation.beam.minor_fwhm_arcsec)
        q_velocity_major = np.gradient(q_major, time, axis=0)
        q_velocity_minor = np.gradient(q_minor, time, axis=0)
        for detector_index, (uid, network) in enumerate(zip(
            scan.detector_uid, scan.detector_network, strict=True
        )):
            blocks = true_blocks(rho[:, detector_index] <= radius_limit)
            for detector_event_index, (start, stop) in enumerate(blocks):
                local = start + int(np.argmin(rho[start:stop, detector_index]))
                vx = float(velocity_x[local])
                vy = float(velocity_y[local])
                q_speed = float(math.hypot(
                    q_velocity_major[local, detector_index],
                    q_velocity_minor[local, detector_index],
                ))
                touches_edge = bool(start == 0 or stop == time.size)
                reason = "accepted"
                window_start = local
                window_stop = local + 1
                scored_samples = 0
                if touches_edge:
                    reason = "half_power_block_touches_scan_edge"
                elif not math.isfinite(q_speed) or q_speed <= 0.0:
                    reason = "invalid_local_elliptical_speed"
                else:
                    half_window_sec = window_limit / q_speed
                    in_window = np.abs(time - time[local]) <= half_window_sec
                    window_indices = np.flatnonzero(in_window)
                    window_start = int(window_indices[0])
                    window_stop = int(window_indices[-1]) + 1
                    scored_samples = int(np.count_nonzero(
                        in_window & scan.score_mask[:, detector_index]
                    ))
                rows.append({
                    "event_number": event_number,
                    "event_id": (
                        f"s{int(scan.scan_row):02d}_uid{int(uid)}_"
                        f"evt{detector_event_index:02d}"
                    ),
                    "scan_row": int(scan.scan_row),
                    "output_scan_index": int(scan.output_scan_index),
                    "detector_index_in_base_scan": int(detector_index),
                    "uid": int(uid),
                    "network": int(network),
                    "detector_event_index": int(detector_event_index),
                    "half_power_start": int(start),
                    "half_power_stop_exclusive": int(stop),
                    "closest_sample": int(local),
                    "fit_window_start": int(window_start),
                    "fit_window_stop_exclusive": int(window_stop),
                    "fit_window_sample_count": int(window_stop - window_start),
                    "scored_sample_count": scored_samples,
                    "touches_scan_edge": touches_edge,
                    "closest_elliptical_fwhm_radius": float(
                        rho[local, detector_index]
                    ),
                    "closest_projected_distance_arcsec": float(math.hypot(
                        dx[local, detector_index], dy[local, detector_index]
                    )),
                    "velocity_x_arcsec_per_sec": vx,
                    "velocity_y_arcsec_per_sec": vy,
                    "projected_speed_arcsec_per_sec": float(math.hypot(vx, vy)),
                    "elliptical_fwhm_speed_per_sec": q_speed,
                    "directed_crossing_angle_deg": float(
                        math.degrees(math.atan2(vy, vx)) % 360.0
                    ),
                    "accepted": reason == "accepted",
                    "disposition": reason,
                })
                event_number += 1
    if not rows:
        raise CrossingContractError("no geometric source crossing was found")
    return Table(rows=rows)


def _subset_scan(
    scan: Any, mask: np.ndarray, minimum: int
) -> Any | None:
    counts = np.sum(mask, axis=0)
    detector_keep = counts >= minimum
    if not np.any(detector_keep):
        return None
    return replace(
        scan,
        apt_x=scan.apt_x[detector_keep],
        apt_y=scan.apt_y[detector_keep],
        detector_uid=scan.detector_uid[detector_keep],
        detector_network=scan.detector_network[detector_keep],
        ptc_weight=scan.ptc_weight[detector_keep],
        valid=scan.valid[:, detector_keep],
        score_mask=mask[:, detector_keep],
        offsource_mask=scan.offsource_mask[:, detector_keep],
        residual_by_baseline={
            name: value[:, detector_keep]
            for name, value in scan.residual_by_baseline.items()
        },
        baseline_coefficients={
            name: value[detector_keep]
            for name, value in scan.baseline_coefficients.items()
        },
        reference_x=scan.reference_x[:, detector_keep],
        reference_y=scan.reference_y[:, detector_keep],
    )


def restrict_to_crossing_support(
    observation: Any,
    events: Table,
    protocol: dict[str, Any],
) -> tuple[Any, Table]:
    """Return an observation whose score masks are the fixed event windows."""
    minimum = int(
        protocol["support"]["minimum_scored_samples_per_detector_scan"]
    )
    selected_scans = []
    support_rows: list[dict[str, Any]] = []
    accepted = events[np.asarray(events["accepted"], dtype=bool)]
    for scan in observation.scans:
        mask = np.zeros_like(scan.score_mask, dtype=bool)
        scan_events = accepted[
            np.asarray(accepted["scan_row"], dtype=int) == int(scan.scan_row)
        ]
        uid_to_detector = {
            int(uid): index for index, uid in enumerate(scan.detector_uid)
        }
        for event in scan_events:
            uid = int(event["uid"])
            if uid not in uid_to_detector:
                raise CrossingContractError("crossing UID is absent from scan")
            detector = uid_to_detector[uid]
            start = int(event["fit_window_start"])
            stop = int(event["fit_window_stop_exclusive"])
            mask[start:stop, detector] = True
        mask &= scan.score_mask
        selected = _subset_scan(scan, mask, minimum)
        if selected is None:
            continue
        selected_scans.append(selected)
        for detector_index, uid in enumerate(selected.detector_uid):
            support_rows.append({
                "scan_row": int(selected.scan_row),
                "output_scan_index": int(selected.output_scan_index),
                "uid": int(uid),
                "network": int(selected.detector_network[detector_index]),
                "scored_sample_count": int(np.count_nonzero(
                    selected.score_mask[:, detector_index]
                )),
            })
    if not selected_scans:
        raise CrossingContractError("crossing support retained no scan")
    support = Table(rows=support_rows)
    restricted = replace(
        observation,
        scans=selected_scans,
        eligible_uid_count=len(set(map(int, support["uid"]))),
        eligible_networks=tuple(sorted(set(map(int, support["network"])))),
        common_support_sample_count=sum(
            scan.recorded_time.size for scan in selected_scans
        ),
        scored_value_count=sum(
            int(np.count_nonzero(scan.score_mask)) for scan in selected_scans
        ),
    )
    return restricted, support


def event_census(events: Table, support: Table) -> dict[str, Any]:
    accepted = np.asarray(events["accepted"], dtype=bool)
    dispositions: dict[str, int] = {}
    for value in events["disposition"]:
        key = str(value)
        dispositions[key] = dispositions.get(key, 0) + 1
    return {
        "geometric_event_count": len(events),
        "accepted_event_count": int(np.count_nonzero(accepted)),
        "rejected_event_count": int(np.count_nonzero(~accepted)),
        "disposition_counts": dict(sorted(dispositions.items())),
        "crossing_unique_detector_count": len(set(map(int, events["uid"]))),
        "retained_detector_scan_count": len(support),
        "retained_unique_detector_count": len(set(map(int, support["uid"]))),
        "retained_scan_count": len(set(map(int, support["scan_row"]))),
        "retained_scored_sample_count": int(sum(support["scored_sample_count"])),
    }
