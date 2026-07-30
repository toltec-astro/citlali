#!/usr/bin/env python3
"""Survey complete raw-I/Q observations for the learned event mode.

This diagnostic removes the RTC-chunk selection from the event census.  For
every observation with an exact matched APT, it projects every available raw
sample in the requested affected networks onto a fixed UID loading, applies a
symmetric pre/post step filter, and clusters independently detected network
candidates in absolute time.  Observations without an exact matched APT are
inventoried and excluded rather than assigned a surrogate detector identity.

For event-rich science observations, the loading is trained on the other two
event-rich observations.  Previously cataloged RTC-guided events are used
only after detection to measure recall.  Other observations use the fixed
all-science loading.  The result is therefore a full-duration,
template-selected catalog; it is not claimed to be complete for pathologies
orthogonal to the learned mode.

The second stage measures the projected time-domain response around every
cross-rack event candidate supported by at least three affected networks.  It
records onset lags, rise time, immediate step, persistence, and an optional
exponential-recovery fit.  Event waveforms are censored before neighboring
catalog events to avoid treating a second event as recovery.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "citlali-continuous-iq-mpl-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy import optimize, signal  # noqa: E402
from scipy.optimize import OptimizeWarning  # noqa: E402

from tools.diagnostics.science_iq_electronics_localization import (  # noqa: E402
    _mode_for_observation,
)


SCHEMA_VERSION = "citlali-science-iq-continuous-event-morphology-v1"
EVENT_VECTOR_SCHEMA = "citlali-science-iq-event-vector-v2"
TEMPLATE_SCHEMA = "citlali-science-iq-tone-susceptibility-v1"
DEFAULT_NETWORKS = (1, 2, 3, 4, 8, 9)
DEFAULT_EVENT_RICH_OBSNUMS = (152419, 152431, 152433)
RAW_PATTERN = re.compile(
    r"toltec(?P<network>\d+)_(?P<obsnum>\d{6})_000_0002_.*\.nc$"
)


@dataclass(frozen=True)
class Template:
    network: int
    source: str
    training_obsnums: tuple[int, ...]
    uid: np.ndarray
    loading: np.ndarray


@dataclass
class Projection:
    obsnum: int
    network: int
    raw_path: Path
    apt_path: Path
    template_source: str
    template_training_obsnums: tuple[int, ...]
    template_tone_count: int
    sample_frequency_hz: float
    time_unix_sec: np.ndarray
    projected_phase_rad: np.ndarray
    step_change_rad: np.ndarray
    step_score: np.ndarray
    step_center_rad: float
    step_sigma_rad: float


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else None


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return math.nan
    median = float(np.median(finite))
    return float(1.482_602_218_505_602 * np.median(np.abs(finite - median)))


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_unix_sec": float(stat.st_mtime),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _array_name(network: int) -> str:
    if 0 <= int(network) <= 6:
        return "a1100"
    if 7 <= int(network) <= 10:
        return "a1400"
    if 11 <= int(network) <= 12:
        return "a2000"
    raise ValueError(f"invalid TolTEC network {network}")


def _rack(network: int) -> str:
    return "RACKA" if int(network) <= 6 else "RACKO"


def _find_raw_files(
    data_root: Path,
    *,
    networks: list[int],
) -> dict[int, dict[int, Path]]:
    result: dict[int, dict[int, Path]] = {}
    for path in sorted(data_root.glob("toltec*_000_0002_*.nc")):
        match = RAW_PATTERN.fullmatch(path.name)
        if not match:
            continue
        network = int(match.group("network"))
        obsnum = int(match.group("obsnum"))
        if network not in networks:
            continue
        observation = result.setdefault(obsnum, {})
        if network in observation:
            raise ValueError(
                f"duplicate raw file for observation {obsnum}, nw{network}"
            )
        observation[network] = path
    if not result:
        raise FileNotFoundError(f"no raw observations found under {data_root}")
    incomplete = {
        obsnum: sorted(set(networks) - set(paths))
        for obsnum, paths in result.items()
        if set(paths) != set(networks)
    }
    if incomplete:
        raise FileNotFoundError(
            f"affected-network raw coverage is incomplete: {incomplete}"
        )
    return result


def _partition_observations_by_apt(
    raw_by_observation: dict[int, dict[int, Path]],
    *,
    apt_root: Path,
) -> tuple[
    dict[int, dict[int, Path]],
    list[dict[str, Any]],
]:
    analyzed: dict[int, dict[int, Path]] = {}
    inventory: list[dict[str, Any]] = []
    for obsnum, raw_paths in sorted(raw_by_observation.items()):
        apt_path = apt_root / f"apt_{obsnum}_matched.ecsv"
        has_apt = apt_path.is_file()
        if has_apt:
            analyzed[obsnum] = raw_paths
        inventory.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "analysis_status": (
                    "analyzed"
                    if has_apt
                    else "excluded_missing_exact_matched_apt"
                ),
                "exclusion_reason": (
                    None
                    if has_apt
                    else (
                        "no exact observation-specific matched APT; "
                        "surrogate detector identity is not allowed"
                    )
                ),
                "affected_network_raw_file_count": int(len(raw_paths)),
                "affected_networks": " ".join(
                    str(value) for value in sorted(raw_paths)
                ),
                "apt_path": str(apt_path) if has_apt else None,
                "raw_paths": " ".join(
                    str(raw_paths[value]) for value in sorted(raw_paths)
                ),
            }
        )
    return analyzed, inventory


def _load_template(
    *,
    obsnum: int,
    network: int,
    event_tones: pd.DataFrame,
    fixed_templates: pd.DataFrame,
    event_rich_obsnums: list[int],
) -> Template:
    if obsnum in event_rich_obsnums:
        training = tuple(
            value for value in event_rich_obsnums if value != obsnum
        )
        rows = event_tones[
            (event_tones["network"] == network)
            & event_tones["obsnum"].isin(training)
        ]
        mode = _mode_for_observation(
            rows,
            coordinate="uid",
            n_modes=1,
        )
        uid = mode.coordinate.astype(int)
        loading = mode.loading[0].astype(float)
        source = "leave_one_observation_out_science_uid_rank1"
    else:
        training = tuple(event_rich_obsnums)
        rows = fixed_templates[fixed_templates["network"] == network]
        uid = rows["uid"].to_numpy(dtype=int)
        loading = rows[
            "phase_rank1_loading_rms_normalized"
        ].to_numpy(dtype=float)
        source = "fixed_all_event_science_uid_rank1"
    finite = np.isfinite(loading)
    uid = uid[finite]
    loading = loading[finite]
    if len(uid) < 8:
        raise ValueError(
            f"observation {obsnum}, nw{network}: template has too few UIDs"
        )
    if len(np.unique(uid)) != len(uid):
        raise ValueError(
            f"observation {obsnum}, nw{network}: template UIDs are not unique"
        )
    rms = math.sqrt(float(np.mean(loading**2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(
            f"observation {obsnum}, nw{network}: invalid template RMS"
        )
    return Template(
        network=network,
        source=source,
        training_obsnums=training,
        uid=uid,
        loading=loading / rms,
    )


def _apt_slots_and_loading(
    apt_path: Path,
    template: Template,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    apt = Table.read(apt_path, format="ascii.ecsv")
    rows = apt[np.asarray(apt["nw"], dtype=int) == template.network]
    loading_by_uid = {
        int(uid): float(value)
        for uid, value in zip(
            template.uid,
            template.loading,
            strict=True,
        )
    }
    selected: list[tuple[int, int, float]] = []
    for row in rows:
        uid_value = float(row["uid"])
        tone_value = float(row["kids_tone"])
        if not np.isfinite(uid_value) or not np.isfinite(tone_value):
            continue
        uid = int(uid_value)
        tone = int(tone_value)
        kids_flag = (
            float(row["kids_flag"]) if "kids_flag" in rows.colnames else 0.0
        )
        map_flag = float(row["flag"]) if "flag" in rows.colnames else 0.0
        if (
            uid in loading_by_uid
            and kids_flag == 0.0
            and map_flag == 0.0
        ):
            selected.append((tone, uid, loading_by_uid[uid]))
    selected.sort()
    if len(selected) < 8:
        raise ValueError(
            f"{apt_path.name}, nw{template.network}: "
            "too few usable template tones"
        )
    slots = np.asarray([value[0] for value in selected], dtype=int)
    uid = np.asarray([value[1] for value in selected], dtype=int)
    loading = np.asarray([value[2] for value in selected], dtype=float)
    if len(np.unique(slots)) != len(slots):
        raise ValueError(
            f"{apt_path.name}, nw{template.network}: duplicate tone slots"
        )
    loading /= math.sqrt(float(np.mean(loading**2)))
    return slots, uid, loading


def _symmetric_step_filter(
    values: np.ndarray,
    *,
    sample_frequency_hz: float,
    window_sec: float,
    guard_sec: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(1, int(round(float(window_sec) * sample_frequency_hz)))
    guard = max(0, int(round(float(guard_sec) * sample_frequency_hz)))
    index = np.arange(len(values))
    valid = (
        (index >= window + guard)
        & (index + guard + window <= len(values))
    )
    result = np.full(len(values), np.nan)
    cumulative = np.concatenate(
        [np.asarray([0.0]), np.cumsum(values, dtype=np.float64)]
    )
    center = index[valid]
    pre = (
        cumulative[center - guard]
        - cumulative[center - guard - window]
    ) / window
    post = (
        cumulative[center + guard + window]
        - cumulative[center + guard]
    ) / window
    result[valid] = post - pre
    return result


def _project_network(
    *,
    obsnum: int,
    network: int,
    raw_path: Path,
    apt_path: Path,
    template: Template,
    step_window_sec: float,
    step_guard_sec: float,
) -> Projection:
    slots, _, loading = _apt_slots_and_loading(apt_path, template)
    with netCDF4.Dataset(raw_path) as raw:
        time_unix = np.asarray(
            raw.variables["Data.Toltec.RecvTime"][:],
            dtype=float,
        )
        i_data = np.ma.filled(
            np.ma.asarray(
                raw.variables["Data.Toltec.Is"][:],
                dtype=np.float32,
            )[:, slots],
            np.nan,
        )
        q_data = np.ma.filled(
            np.ma.asarray(
                raw.variables["Data.Toltec.Qs"][:],
                dtype=np.float32,
            )[:, slots],
            np.nan,
        )
        header_obsnum = int(
            np.asarray(raw.variables["Header.Toltec.ObsNum"][...]).item()
        )
        header_network = int(
            np.asarray(raw.variables["Header.Toltec.RoachIndex"][...]).item()
        )
    if header_obsnum != obsnum or header_network != network:
        raise ValueError(f"{raw_path.name}: filename/header identity mismatch")
    finite_tone = np.all(np.isfinite(i_data) & np.isfinite(q_data), axis=0)
    i_data = i_data[:, finite_tone]
    q_data = q_data[:, finite_tone]
    loading = loading[finite_tone]
    if len(loading) < 8:
        raise ValueError(
            f"{raw_path.name}: too few finite template tones remain"
        )
    phase = np.unwrap(np.arctan2(q_data, i_data), axis=0)
    phase -= phase[0, :]
    denominator = float(np.sum(loading**2))
    projected = (
        np.sum(phase * loading[None, :], axis=1) / denominator
    ).astype(float)
    del phase, i_data, q_data
    gc.collect()
    sample_frequency = float(1.0 / np.median(np.diff(time_unix)))
    step = _symmetric_step_filter(
        projected,
        sample_frequency_hz=sample_frequency,
        window_sec=step_window_sec,
        guard_sec=step_guard_sec,
    )
    finite_step = step[np.isfinite(step)]
    center = float(np.median(finite_step))
    sigma = _robust_sigma(finite_step)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"{raw_path.name}: invalid step-filter scale")
    score = (step - center) / sigma
    return Projection(
        obsnum=obsnum,
        network=network,
        raw_path=raw_path,
        apt_path=apt_path,
        template_source=template.source,
        template_training_obsnums=template.training_obsnums,
        template_tone_count=int(len(loading)),
        sample_frequency_hz=sample_frequency,
        time_unix_sec=time_unix,
        projected_phase_rad=projected,
        step_change_rad=step,
        step_score=score,
        step_center_rad=center,
        step_sigma_rad=sigma,
    )


def _network_candidate_rows(
    projection: Projection,
    *,
    threshold: float,
    prominence: float,
    minimum_separation_sec: float,
) -> list[dict[str, Any]]:
    distance = max(
        1,
        int(
            round(
                minimum_separation_sec * projection.sample_frequency_hz
            )
        ),
    )
    score = np.nan_to_num(
        np.abs(projection.step_score),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    peaks, properties = signal.find_peaks(
        score,
        height=float(threshold),
        prominence=float(prominence),
        distance=distance,
    )
    rows: list[dict[str, Any]] = []
    for ordinal, (sample, height, peak_prominence) in enumerate(
        zip(
            peaks,
            properties["peak_heights"],
            properties["prominences"],
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": (
                    f"c{projection.obsnum}_nw{projection.network}_"
                    f"{ordinal:05d}"
                ),
                "obsnum": projection.obsnum,
                "network": projection.network,
                "array": _array_name(projection.network),
                "rack": _rack(projection.network),
                "sample_index_zero_based": int(sample),
                "candidate_time_unix_sec": float(
                    projection.time_unix_sec[sample]
                ),
                "candidate_time_since_start_sec": float(
                    projection.time_unix_sec[sample]
                    - projection.time_unix_sec[0]
                ),
                "signed_step_change_rad_per_rms_loading": float(
                    projection.step_change_rad[sample]
                ),
                "signed_step_score": float(
                    projection.step_score[sample]
                ),
                "absolute_step_score": float(height),
                "score_prominence": float(peak_prominence),
                "template_tone_count": projection.template_tone_count,
                "template_source": projection.template_source,
                "template_training_obsnums": " ".join(
                    str(value)
                    for value in projection.template_training_obsnums
                ),
                "raw_path": str(projection.raw_path),
                "apt_path": str(projection.apt_path),
            }
        )
    return rows


def _cluster_arrays(
    time: np.ndarray,
    network: np.ndarray,
    score: np.ndarray,
    *,
    coincidence_sec: float,
    consume_sec: float,
) -> list[dict[str, Any]]:
    time = np.asarray(time, dtype=float)
    network = np.asarray(network, dtype=int)
    score = np.asarray(score, dtype=float)
    active = np.ones(len(time), dtype=bool)
    clusters: list[dict[str, Any]] = []
    while np.any(active):
        active_index = np.flatnonzero(active)
        seed = active_index[
            int(np.argmax(np.abs(score[active_index])))
        ]
        near = active_index[
            np.abs(time[active_index] - time[seed]) <= coincidence_sec
        ]
        chosen: list[int] = []
        for value in np.unique(network[near]):
            choices = near[network[near] == value]
            chosen.append(
                int(choices[int(np.argmax(np.abs(score[choices])))])
            )
        center = float(np.median(time[chosen]))
        near = active_index[
            np.abs(time[active_index] - center) <= coincidence_sec
        ]
        chosen = []
        for value in np.unique(network[near]):
            choices = near[network[near] == value]
            chosen.append(
                int(choices[int(np.argmax(np.abs(score[choices])))])
            )
        chosen_array = np.asarray(chosen, dtype=int)
        center = float(np.median(time[chosen_array]))
        clusters.append(
            {
                "center_time_unix_sec": center,
                "member_indices": chosen,
                "network_count": int(len(chosen)),
                "network_span_sec": float(
                    np.max(time[chosen_array])
                    - np.min(time[chosen_array])
                ),
                "maximum_absolute_score": float(
                    np.max(np.abs(score[chosen_array]))
                ),
            }
        )
        active[
            np.abs(time - center) <= consume_sec
        ] = False
    return clusters


def _quality_tier(networks: Iterable[int]) -> tuple[str, bool, bool]:
    values = sorted(set(int(value) for value in networks))
    cross_rack = any(value <= 6 for value in values) and any(
        value >= 7 for value in values
    )
    primary = cross_rack and len(values) >= 3
    if cross_rack and len(values) >= 5:
        return "A_cross_rack_5plus", primary, cross_rack
    if primary:
        return "B_cross_rack_3to4", primary, cross_rack
    if cross_rack:
        return "C_cross_rack_2", primary, cross_rack
    return "D_same_rack", primary, cross_rack


def _sign_sequence_metrics(
    event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        event_rows,
        key=lambda row: row.get(
            "refined_event_time_unix_sec",
            row["event_time_unix_sec"],
        ),
    )
    sign_map = {"positive": 1, "negative": -1}
    signs = np.asarray(
        [
            sign_map[row["dominant_projected_step_sign"]]
            for row in ordered
            if row["dominant_projected_step_sign"] in sign_map
        ],
        dtype=int,
    )
    if signs.size == 0:
        return {
            "primary_positive_step_fraction": None,
            "primary_adjacent_sign_alternation_fraction": None,
            "primary_iid_sign_alternation_expectation": None,
            "primary_sign_alternation_excess": None,
            "primary_maximum_same_sign_run_length": None,
        }
    positive_fraction = float(np.mean(signs > 0))
    iid_expectation = 2.0 * positive_fraction * (1.0 - positive_fraction)
    if signs.size >= 2:
        alternation = float(np.mean(signs[1:] != signs[:-1]))
        boundaries = np.concatenate(
            (
                np.asarray([0]),
                np.flatnonzero(signs[1:] != signs[:-1]) + 1,
                np.asarray([len(signs)]),
            )
        )
        maximum_run = int(np.max(np.diff(boundaries)))
    else:
        alternation = math.nan
        maximum_run = 1
    return {
        "primary_positive_step_fraction": positive_fraction,
        "primary_adjacent_sign_alternation_fraction": _finite_or_none(
            alternation
        ),
        "primary_iid_sign_alternation_expectation": iid_expectation,
        "primary_sign_alternation_excess": _finite_or_none(
            alternation - iid_expectation
        ),
        "primary_maximum_same_sign_run_length": maximum_run,
    }


def _cluster_candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    obsnum: int,
    coincidence_sec: float,
    consume_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not candidates:
        return [], []
    frame = pd.DataFrame(candidates)
    clusters = _cluster_arrays(
        frame["candidate_time_unix_sec"].to_numpy(dtype=float),
        frame["network"].to_numpy(dtype=int),
        frame["signed_step_score"].to_numpy(dtype=float),
        coincidence_sec=coincidence_sec,
        consume_sec=consume_sec,
    )
    rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    kept = [cluster for cluster in clusters if cluster["network_count"] >= 2]
    kept.sort(key=lambda value: value["center_time_unix_sec"])
    observation_start = float(
        np.median(
            frame["candidate_time_unix_sec"].to_numpy(dtype=float)
            - frame["candidate_time_since_start_sec"].to_numpy(dtype=float)
        )
    )
    for ordinal, cluster in enumerate(kept, start=1):
        event_id = f"e{obsnum}_{ordinal:05d}"
        selected = frame.iloc[cluster["member_indices"]].copy()
        networks = sorted(selected["network"].astype(int).unique())
        tier, primary, cross_rack = _quality_tier(networks)
        signs = np.sign(
            selected["signed_step_score"].to_numpy(dtype=float)
        )
        signs = signs[signs != 0.0]
        positive_count = int(np.count_nonzero(signs > 0.0))
        negative_count = int(np.count_nonzero(signs < 0.0))
        sign_agreement = (
            max(positive_count, negative_count) / len(signs)
            if len(signs)
            else math.nan
        )
        dominant_sign = (
            "positive"
            if positive_count > negative_count
            else "negative"
            if negative_count > positive_count
            else "tied"
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "obsnum": obsnum,
                "event_time_unix_sec": float(
                    cluster["center_time_unix_sec"]
                ),
                "event_time_utc": datetime.fromtimestamp(
                    float(cluster["center_time_unix_sec"]),
                    tz=UTC,
                ).isoformat(),
                "event_time_since_observation_start_sec": float(
                    cluster["center_time_unix_sec"] - observation_start
                ),
                "network_count": int(cluster["network_count"]),
                "networks": " ".join(str(value) for value in networks),
                "racks": " ".join(
                    sorted({_rack(value) for value in networks})
                ),
                "cross_rack": bool(cross_rack),
                "primary_event_candidate": bool(primary),
                "quality_tier": tier,
                "network_time_span_sec": float(
                    cluster["network_span_sec"]
                ),
                "maximum_absolute_step_score": float(
                    cluster["maximum_absolute_score"]
                ),
                "median_absolute_step_score": float(
                    selected["absolute_step_score"].median()
                ),
                "dominant_projected_step_sign": dominant_sign,
                "network_member_sign_agreement_fraction": _finite_or_none(
                    sign_agreement
                ),
                "network_member_sign_is_unanimous": bool(
                    len(signs) > 0 and sign_agreement == 1.0
                ),
            }
        )
        for candidate in selected.to_dict("records"):
            member_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event_id,
                    "obsnum": obsnum,
                    "network": int(candidate["network"]),
                    "candidate_id": candidate["candidate_id"],
                    "candidate_time_unix_sec": float(
                        candidate["candidate_time_unix_sec"]
                    ),
                    "candidate_lag_from_event_center_sec": float(
                        candidate["candidate_time_unix_sec"]
                        - cluster["center_time_unix_sec"]
                    ),
                    "signed_step_change_rad_per_rms_loading": float(
                        candidate[
                            "signed_step_change_rad_per_rms_loading"
                        ]
                    ),
                    "signed_step_score": float(
                        candidate["signed_step_score"]
                    ),
                    "absolute_step_score": float(
                        candidate["absolute_step_score"]
                    ),
                }
            )
    return rows, member_rows


def _shifted_cluster_counts(
    candidates: list[dict[str, Any]],
    *,
    observation_start_sec: float,
    observation_end_sec: float,
    coincidence_sec: float,
    consume_sec: float,
    n_permutations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates or n_permutations <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    frame = pd.DataFrame(candidates)
    duration = float(observation_end_sec - observation_start_sec)
    relative = (
        frame["candidate_time_unix_sec"].to_numpy(dtype=float)
        - observation_start_sec
    )
    network = frame["network"].to_numpy(dtype=int)
    score = frame["signed_step_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_seed))
    primary_count: list[int] = []
    tier_a_count: list[int] = []
    for _ in range(int(n_permutations)):
        shifted = relative.copy()
        for value in np.unique(network):
            offset = float(rng.uniform(0.0, duration))
            selected = network == value
            shifted[selected] = (shifted[selected] + offset) % duration
        clusters = _cluster_arrays(
            shifted,
            network,
            score,
            coincidence_sec=coincidence_sec,
            consume_sec=consume_sec,
        )
        primary = 0
        tier_a = 0
        for cluster in clusters:
            values = network[cluster["member_indices"]]
            tier, is_primary, _ = _quality_tier(values)
            primary += int(is_primary)
            tier_a += int(tier == "A_cross_rack_5plus")
        primary_count.append(primary)
        tier_a_count.append(tier_a)
    return (
        np.asarray(primary_count, dtype=int),
        np.asarray(tier_a_count, dtype=int),
    )


def _fit_recovery(
    time_sec: np.ndarray,
    value: np.ndarray,
) -> dict[str, Any]:
    time_sec = np.asarray(time_sec, dtype=float)
    value = np.asarray(value, dtype=float)
    finite = np.isfinite(time_sec) & np.isfinite(value)
    time_sec = time_sec[finite]
    value = value[finite]
    if len(value) < 20 or np.ptp(time_sec) < 0.5:
        return {
            "recovery_fit_status": "insufficient_samples",
            "recovery_tau_sec": None,
            "recovery_asymptote_fraction": None,
            "recovery_fit_r2": None,
        }

    def model(time: np.ndarray, asymptote: float, amplitude: float, tau: float):
        return asymptote + amplitude * np.exp(-time / tau)

    asymptote = float(np.median(value[-max(4, len(value) // 5) :]))
    amplitude = float(value[0] - asymptote)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            parameters, _ = optimize.curve_fit(
                model,
                time_sec,
                value,
                p0=(asymptote, amplitude, 1.0),
                bounds=((-5.0, -10.0, 0.02), (5.0, 10.0, 30.0)),
                maxfev=20_000,
            )
    except (RuntimeError, ValueError, FloatingPointError):
        return {
            "recovery_fit_status": "fit_failed",
            "recovery_tau_sec": None,
            "recovery_asymptote_fraction": None,
            "recovery_fit_r2": None,
        }
    prediction = model(time_sec, *parameters)
    denominator = float(np.sum((value - np.mean(value)) ** 2))
    r2 = (
        1.0 - float(np.sum((value - prediction) ** 2)) / denominator
        if denominator > 0.0
        else math.nan
    )
    return {
        "recovery_fit_status": "fit",
        "recovery_tau_sec": float(parameters[2]),
        "recovery_asymptote_fraction": float(parameters[0]),
        "recovery_fit_r2": _finite_or_none(r2),
    }


def _window_median(
    time_sec: np.ndarray,
    value: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> float:
    selected = (
        (time_sec >= float(start_sec))
        & (time_sec <= float(end_sec))
        & np.isfinite(value)
    )
    return float(np.median(value[selected])) if np.any(selected) else math.nan


def _rise_time(
    time_sec: np.ndarray,
    normalized: np.ndarray,
) -> float:
    selected = (
        (time_sec >= -0.15)
        & (time_sec <= 0.40)
        & np.isfinite(normalized)
    )
    time = time_sec[selected]
    value = normalized[selected]
    if len(value) < 5:
        return math.nan
    smooth_count = min(5, len(value) if len(value) % 2 == 1 else len(value) - 1)
    if smooth_count >= 3:
        value = signal.savgol_filter(
            value,
            window_length=smooth_count,
            polyorder=min(2, smooth_count - 1),
            mode="interp",
        )
    crossing_10 = np.flatnonzero(value >= 0.10)
    if crossing_10.size == 0:
        return math.nan
    first_10 = int(crossing_10[0])
    crossing_90 = np.flatnonzero(
        (np.arange(len(value)) >= first_10) & (value >= 0.90)
    )
    if crossing_90.size == 0:
        return math.nan
    return float(time[int(crossing_90[0])] - time[first_10])


def _refine_onset_time(
    projection: Projection,
    candidate_time_unix_sec: float,
    *,
    search_half_width_sec: float = 0.30,
    smooth_sec: float = 0.05,
) -> float:
    time = projection.time_unix_sec
    selected = np.flatnonzero(
        np.abs(time - float(candidate_time_unix_sec))
        <= float(search_half_width_sec)
    )
    if selected.size < 7:
        return float(candidate_time_unix_sec)
    value = projection.projected_phase_rad[selected]
    window = max(
        3,
        int(round(float(smooth_sec) * projection.sample_frequency_hz)),
    )
    if window % 2 == 0:
        window += 1
    window = min(window, len(value) if len(value) % 2 == 1 else len(value) - 1)
    if window >= 3:
        value = signal.savgol_filter(
            value,
            window_length=window,
            polyorder=min(2, window - 1),
            mode="interp",
        )
    derivative = np.gradient(value, time[selected])
    return float(time[selected[int(np.argmax(np.abs(derivative)))]])


def _event_morphology(
    *,
    event_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
    projections: dict[int, Projection],
    waveform_pre_sec: float,
    waveform_post_sec: float,
    waveform_step_sec: float,
    stack_minimum_snr: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    primary = [
        row for row in event_rows if row["primary_event_candidate"]
    ]
    primary.sort(key=lambda row: row["event_time_unix_sec"])
    member_frame = pd.DataFrame(member_rows)
    refined_members_by_event: dict[str, dict[int, float]] = {}
    refined_event_time: list[float] = []
    for event in primary:
        event_members = member_frame[
            member_frame["event_id"] == event["event_id"]
        ]
        refined = {
            int(row["network"]): _refine_onset_time(
                projections[int(row["network"])],
                float(row["candidate_time_unix_sec"]),
            )
            for row in event_members.to_dict("records")
        }
        center = float(np.median(list(refined.values())))
        event["refined_event_time_unix_sec"] = center
        event["refined_event_time_utc"] = datetime.fromtimestamp(
            center,
            tz=UTC,
        ).isoformat()
        event["refined_network_time_span_sec"] = float(
            max(refined.values()) - min(refined.values())
        )
        refined_members_by_event[event["event_id"]] = refined
        refined_event_time.append(center)
    morphology_rows: list[dict[str, Any]] = []
    waveform_records: list[dict[str, Any]] = []
    grid = np.arange(
        -float(waveform_pre_sec),
        float(waveform_post_sec) + 0.5 * waveform_step_sec,
        float(waveform_step_sec),
    )
    for event_index, event in enumerate(primary):
        previous_time = (
            refined_event_time[event_index - 1]
            if event_index > 0
            else -math.inf
        )
        next_time = (
            refined_event_time[event_index + 1]
            if event_index + 1 < len(primary)
            else math.inf
        )
        event_members = member_frame[
            member_frame["event_id"] == event["event_id"]
        ]
        member_by_network = {
            int(row["network"]): row
            for row in event_members.to_dict("records")
        }
        refined_by_network = refined_members_by_event[event["event_id"]]
        event_center = refined_event_time[event_index]
        for network, projection in sorted(projections.items()):
            member = member_by_network.get(network)
            participated = member is not None
            onset = (
                refined_by_network[network]
                if participated
                else event_center
            )
            available_pre = min(
                onset - float(projection.time_unix_sec[0]),
                onset - previous_time - 0.30,
            )
            available_post = min(
                float(projection.time_unix_sec[-1]) - onset,
                next_time - onset - 0.30,
            )
            absolute_grid = onset + grid
            waveform = np.interp(
                absolute_grid,
                projection.time_unix_sec,
                projection.projected_phase_rad,
                left=np.nan,
                right=np.nan,
            )
            waveform[grid < -available_pre] = np.nan
            waveform[grid > available_post] = np.nan
            baseline = (
                (grid >= -min(2.0, max(0.5, available_pre)))
                & (grid <= -0.35)
                & np.isfinite(waveform)
            )
            if np.count_nonzero(baseline) >= 8:
                coefficients = np.polyfit(
                    grid[baseline],
                    waveform[baseline],
                    deg=1,
                )
                detrended = waveform - np.polyval(coefficients, grid)
                pre_noise = _robust_sigma(detrended[baseline])
            else:
                detrended = np.full_like(waveform, np.nan)
                pre_noise = math.nan
            pre_level = _window_median(
                grid,
                detrended,
                -0.30,
                -0.08,
            )
            post_level = _window_median(
                grid,
                detrended,
                0.08,
                0.30,
            )
            step = post_level - pre_level
            step_snr = (
                abs(step) / pre_noise
                if np.isfinite(step)
                and np.isfinite(pre_noise)
                and pre_noise > 0.0
                else math.nan
            )
            normalized = (
                detrended / step
                if np.isfinite(step) and abs(step) > 0.0
                else np.full_like(detrended, np.nan)
            )
            peak_selected = (
                (grid >= 0.0)
                & (grid <= min(1.0, available_post))
                & np.isfinite(normalized)
            )
            if np.any(peak_selected):
                peak_local = np.flatnonzero(peak_selected)
                peak_index = int(
                    peak_local[
                        int(np.argmax(normalized[peak_selected]))
                    ]
                )
                peak_fraction = float(normalized[peak_index])
                peak_time = float(grid[peak_index])
            else:
                peak_fraction = math.nan
                peak_time = math.nan
            level_1 = _window_median(grid, normalized, 0.85, 1.15)
            level_3 = _window_median(grid, normalized, 2.80, 3.20)
            level_5 = _window_median(grid, normalized, 4.75, 5.25)
            fit_end = min(6.0, available_post)
            fit_selected = (
                (grid >= 0.20)
                & (grid <= fit_end)
                & np.isfinite(normalized)
            )
            recovery = _fit_recovery(
                grid[fit_selected],
                normalized[fit_selected],
            )
            status = (
                "valid"
                if np.isfinite(step_snr) and available_pre >= 0.5
                else "insufficient_baseline"
            )
            morphology_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event["event_id"],
                    "obsnum": event["obsnum"],
                    "network": network,
                    "array": _array_name(network),
                    "rack": _rack(network),
                    "network_participated_in_detection": participated,
                    "event_quality_tier": event["quality_tier"],
                    "network_detection_candidate_time_unix_sec": (
                        float(member["candidate_time_unix_sec"])
                        if participated
                        else None
                    ),
                    "network_onset_time_unix_sec": onset,
                    "network_onset_lag_from_event_center_sec": (
                        onset - event_center
                    ),
                    "available_pre_event_sec": _finite_or_none(available_pre),
                    "available_post_event_sec": _finite_or_none(
                        available_post
                    ),
                    "morphology_status": status,
                    "immediate_step_rad_per_rms_loading": _finite_or_none(step),
                    "pre_event_projected_phase_sigma_rad": _finite_or_none(
                        pre_noise
                    ),
                    "immediate_step_snr": _finite_or_none(step_snr),
                    "rise_time_10_to_90_sec": _finite_or_none(
                        _rise_time(grid, normalized)
                    ),
                    "peak_fraction_of_immediate_step": _finite_or_none(
                        peak_fraction
                    ),
                    "time_to_peak_sec": _finite_or_none(peak_time),
                    "level_fraction_at_1s": _finite_or_none(level_1),
                    "level_fraction_at_3s": _finite_or_none(level_3),
                    "level_fraction_at_5s": _finite_or_none(level_5),
                    **recovery,
                }
            )
            if (
                participated
                and np.isfinite(step_snr)
                and step_snr >= stack_minimum_snr
            ):
                waveform_records.append(
                    {
                        "event_id": event["event_id"],
                        "obsnum": event["obsnum"],
                        "network": network,
                        "quality_tier": event["quality_tier"],
                        "step_snr": float(step_snr),
                        "available_post_sec": float(available_post),
                        "time_sec": grid.copy(),
                        "normalized": normalized.copy(),
                    }
                )
    return morphology_rows, waveform_records


def _summarize_waveforms(
    waveform_records: list[dict[str, Any]],
    *,
    networks: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stack_rows: list[dict[str, Any]] = []
    for network in networks:
        records = [
            value
            for value in waveform_records
            if value["network"] == network
        ]
        if not records:
            continue
        grid = np.asarray(records[0]["time_sec"], dtype=float)
        if any(
            not np.array_equal(grid, np.asarray(value["time_sec"]))
            for value in records[1:]
        ):
            raise ValueError("waveform records do not share a common grid")
        matrix = np.vstack([value["normalized"] for value in records])
        for column, time_value in enumerate(grid):
            values = matrix[:, column]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            stack_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": network,
                    "time_from_network_onset_sec": float(time_value),
                    "contributing_event_count": int(len(finite)),
                    "median_normalized_projected_phase": float(
                        np.median(finite)
                    ),
                    "q16_normalized_projected_phase": float(
                        np.quantile(finite, 0.16)
                    ),
                    "q84_normalized_projected_phase": float(
                        np.quantile(finite, 0.84)
                    ),
                }
            )
    example_rows: list[dict[str, Any]] = []
    for network in networks:
        records = sorted(
            (
                value
                for value in waveform_records
                if value["network"] == network
                and value["available_post_sec"] >= 5.0
            ),
            key=lambda value: value["step_snr"],
            reverse=True,
        )[:3]
        for value in records:
            for time_value, normalized in zip(
                value["time_sec"],
                value["normalized"],
                strict=True,
            ):
                example_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "event_id": value["event_id"],
                        "obsnum": value["obsnum"],
                        "network": network,
                        "time_from_network_onset_sec": float(time_value),
                        "normalized_projected_phase": _finite_or_none(
                            normalized
                        ),
                        "immediate_step_snr": value["step_snr"],
                    }
                )
    return stack_rows, example_rows


def _telegraph_example_candidate(
    *,
    projection: Projection,
    event_rows: list[dict[str, Any]],
    window_sec: float,
    padding_sec: float = 2.0,
) -> dict[str, Any] | None:
    network_events = [
        row
        for row in event_rows
        if row["primary_event_candidate"]
        and projection.network
        in {int(value) for value in str(row["networks"]).split()}
    ]
    network_events.sort(key=lambda row: row["refined_event_time_unix_sec"])
    if not network_events:
        return None
    time = np.asarray(
        [
            row["refined_event_time_unix_sec"]
            for row in network_events
        ],
        dtype=float,
    )
    counts = np.asarray(
        [
            np.count_nonzero((time >= start) & (time <= start + window_sec))
            for start in time
        ],
        dtype=int,
    )
    start = float(time[int(np.argmax(counts))])
    selected_events = [
        row
        for row in network_events
        if start <= row["refined_event_time_unix_sec"] <= start + window_sec
    ]
    lower = max(float(projection.time_unix_sec[0]), start - padding_sec)
    upper = min(
        float(projection.time_unix_sec[-1]),
        start + window_sec + padding_sec,
    )
    selected_samples = (
        (projection.time_unix_sec >= lower)
        & (projection.time_unix_sec <= upper)
    )
    relative_time = projection.time_unix_sec[selected_samples] - start
    projected = projection.projected_phase_rad[selected_samples]
    coefficients = np.polyfit(relative_time, projected, deg=1)
    detrended = projected - np.polyval(coefficients, relative_time)
    sample_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "obsnum": projection.obsnum,
            "network": projection.network,
            "time_from_selected_window_start_sec": float(time_value),
            "time_since_observation_start_sec": float(
                absolute_time - projection.time_unix_sec[0]
            ),
            "projected_phase_rad_per_rms_loading": float(value),
            "linearly_detrended_projected_phase_rad_per_rms_loading": float(
                residual
            ),
        }
        for absolute_time, time_value, value, residual in zip(
            projection.time_unix_sec[selected_samples],
            relative_time,
            projected,
            detrended,
            strict=True,
        )
    ]
    marker_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "event_id": row["event_id"],
            "obsnum": projection.obsnum,
            "network": projection.network,
            "event_time_from_selected_window_start_sec": float(
                row["refined_event_time_unix_sec"] - start
            ),
            "dominant_projected_step_sign": row[
                "dominant_projected_step_sign"
            ],
            "quality_tier": row["quality_tier"],
            "network_count": int(row["network_count"]),
        }
        for row in selected_events
    ]
    return {
        "obsnum": projection.obsnum,
        "network": projection.network,
        "event_count": int(len(selected_events)),
        "window_sec": float(window_sec),
        "padding_sec": float(padding_sec),
        "window_start_unix_sec": start,
        "linear_detrend_slope_rad_per_sec": float(coefficients[0]),
        "sample_rows": sample_rows,
        "marker_rows": marker_rows,
    }


def _telegraph_example_figure(
    path: Path,
    example: dict[str, Any],
) -> None:
    samples = pd.DataFrame(example["sample_rows"])
    markers = pd.DataFrame(example["marker_rows"])
    figure, axis = plt.subplots(
        figsize=(13.0, 5.0),
        constrained_layout=True,
    )
    axis.plot(
        samples["time_from_selected_window_start_sec"],
        samples[
            "linearly_detrended_projected_phase_rad_per_rms_loading"
        ],
        linewidth=1.0,
        color="tab:blue",
    )
    colors = {"positive": "tab:red", "negative": "tab:blue"}
    seen: set[str] = set()
    for row in markers.to_dict("records"):
        sign = row["dominant_projected_step_sign"]
        axis.axvline(
            row["event_time_from_selected_window_start_sec"],
            color=colors[sign],
            alpha=0.55,
            linewidth=1.0,
            label=sign if sign not in seen else None,
        )
        seen.add(sign)
    axis.axvspan(
        0.0,
        float(example["window_sec"]),
        color="0.5",
        alpha=0.05,
    )
    axis.set_xlabel("time from selected window start (s)")
    axis.set_ylabel(
        "detrended projected phase\n(rad per RMS-normalized loading)"
    )
    axis.set_title(
        f"obs {example['obsnum']} nw{example['network']}: densest "
        f"{example['window_sec']:.0f} s window has "
        f"{example['event_count']} primary events"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _known_event_recall_rows(
    known: pd.DataFrame,
    catalog: list[dict[str, Any]],
    *,
    match_tolerance_sec: float,
) -> list[dict[str, Any]]:
    primary = pd.DataFrame(
        [row for row in catalog if row["primary_event_candidate"]]
    )
    rows: list[dict[str, Any]] = []
    for known_row in known.to_dict("records"):
        selected = (
            primary[primary["obsnum"] == int(known_row["obsnum"])]
            if not primary.empty
            else pd.DataFrame()
        )
        if selected.empty:
            nearest_id = None
            nearest_time = math.nan
            delta = math.inf
            tier = None
        else:
            time_field = (
                "refined_event_time_unix_sec"
                if "refined_event_time_unix_sec" in selected.columns
                else "event_time_unix_sec"
            )
            differences = (
                selected[time_field]
                - float(known_row["cluster_time_unix_sec"])
            ).abs()
            index = differences.idxmin()
            nearest = selected.loc[index]
            nearest_id = nearest["event_id"]
            nearest_time = float(nearest[time_field])
            delta = float(
                nearest_time - float(known_row["cluster_time_unix_sec"])
            )
            tier = nearest["quality_tier"]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "known_event_cluster_id": known_row["event_cluster_id"],
                "obsnum": int(known_row["obsnum"]),
                "known_event_time_unix_sec": float(
                    known_row["cluster_time_unix_sec"]
                ),
                "nearest_continuous_event_id": nearest_id,
                "nearest_continuous_event_time_unix_sec": _finite_or_none(
                    nearest_time
                ),
                "signed_time_difference_sec": _finite_or_none(delta),
                "absolute_time_difference_sec": _finite_or_none(abs(delta)),
                "matched_within_tolerance": bool(
                    np.isfinite(delta)
                    and abs(delta) <= match_tolerance_sec
                ),
                "match_tolerance_sec": float(match_tolerance_sec),
                "nearest_event_quality_tier": tier,
            }
        )
    return rows


def _morphology_summary_rows(
    rows: list[dict[str, Any]],
    *,
    group_by_observation: bool = False,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    selected = frame[
        frame["network_participated_in_detection"].astype(bool)
        & frame["morphology_status"].eq("valid")
    ]
    result: list[dict[str, Any]] = []
    grouped = (
        selected.groupby(["obsnum", "network"])
        if group_by_observation
        else selected.groupby("network")
    )
    for key, group in grouped:
        if group_by_observation:
            obsnum, network = key
        else:
            network = key
        good_fit = group[
            group["recovery_fit_status"].eq("fit")
            & (group["recovery_fit_r2"].fillna(-np.inf) >= 0.5)
        ]
        row = {
                "schema_version": SCHEMA_VERSION,
                "network": int(network),
                "participating_valid_event_count": int(len(group)),
                "median_immediate_step_snr": _finite_median(
                    group["immediate_step_snr"]
                ),
                "median_rise_time_10_to_90_sec": _finite_median(
                    group["rise_time_10_to_90_sec"]
                ),
                "median_peak_fraction_of_step": _finite_median(
                    group["peak_fraction_of_immediate_step"]
                ),
                "median_level_fraction_at_1s": _finite_median(
                    group["level_fraction_at_1s"]
                ),
                "median_level_fraction_at_3s": _finite_median(
                    group["level_fraction_at_3s"]
                ),
                "median_level_fraction_at_5s": _finite_median(
                    group["level_fraction_at_5s"]
                ),
                "recovery_fit_r2_ge_0p5_count": int(len(good_fit)),
                "median_recovery_tau_sec_for_r2_ge_0p5": (
                    _finite_median(good_fit["recovery_tau_sec"])
                    if not good_fit.empty
                    else None
                ),
                "median_recovery_asymptote_fraction_for_r2_ge_0p5": (
                    _finite_median(
                        good_fit["recovery_asymptote_fraction"]
                    )
                    if not good_fit.empty
                    else None
                ),
            }
        if group_by_observation:
            row = {
                "schema_version": SCHEMA_VERSION,
                "obsnum": int(obsnum),
                **{
                    field: value
                    for field, value in row.items()
                    if field != "schema_version"
                },
            }
        result.append(row)
    return result


def _chronology_figure(
    path: Path,
    catalog_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
) -> None:
    catalog = pd.DataFrame(catalog_rows)
    observations = pd.DataFrame(observation_rows).sort_values(
        "observation_start_unix_sec"
    )
    primary = catalog[catalog["primary_event_candidate"].astype(bool)]
    origin = float(observations["observation_start_unix_sec"].min())
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13.0, 7.0),
        constrained_layout=True,
        sharex=True,
        height_ratios=(2.0, 1.2),
    )
    axis = axes[0]
    for _, observation in observations.iterrows():
        start = (
            float(observation["observation_start_unix_sec"]) - origin
        ) / 3600.0
        end = (
            float(observation["observation_end_unix_sec"]) - origin
        ) / 3600.0
        axis.axvspan(start, end, color="0.94", zorder=0)
        axis.text(
            0.5 * (start + end),
            -0.45,
            str(int(observation["obsnum"])),
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
        )
    for tier, color, label in (
        ("B_cross_rack_3to4", "#f28e2b", "3–4 networks"),
        ("A_cross_rack_5plus", "#d62728", "5–6 networks"),
    ):
        selected = primary[primary["quality_tier"] == tier]
        time_field = (
            "refined_event_time_unix_sec"
            if "refined_event_time_unix_sec" in selected.columns
            else "event_time_unix_sec"
        )
        x = (
            selected[time_field].to_numpy(dtype=float) - origin
        ) / 3600.0
        axis.scatter(
            x,
            selected["network_count"],
            s=np.clip(
                selected["maximum_absolute_step_score"].to_numpy(dtype=float),
                8.0,
                80.0,
            ),
            alpha=0.65,
            color=color,
            label=label,
        )
    axis.set_ylabel("affected networks in event")
    axis.set_ylim(2.5, 6.6)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    axis.set_title(
        "Full-duration UID-template event catalog "
        "(marker size = maximum step score)"
    )
    rate_axis = axes[1]
    x = (
        observations["observation_start_unix_sec"].to_numpy(dtype=float)
        - origin
    ) / 3600.0
    rate_axis.plot(
        x,
        observations["primary_event_rate_per_min"],
        marker="o",
        label="observed cross-rack rate",
    )
    rate_axis.plot(
        x,
        observations["shift_null_primary_rate_p95_per_min"],
        marker="s",
        label="95% time-shift coincidence rate",
    )
    rate_axis.set_ylabel("events / min")
    rate_axis.set_xlabel("hours since first archived observation")
    rate_axis.grid(alpha=0.25)
    rate_axis.legend(loc="upper left")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _sign_sequence_figure(
    path: Path,
    catalog_rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(catalog_rows)
    selected = frame[frame["primary_event_candidate"].astype(bool)].copy()
    counts = selected.groupby("obsnum").size().sort_values(ascending=False)
    obsnums = sorted(int(value) for value in counts.head(3).index)
    if not obsnums:
        raise ValueError("no primary events are available for sign figure")
    figure, axes = plt.subplots(
        len(obsnums),
        1,
        figsize=(12.0, 2.6 * len(obsnums)),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = {
        "A_cross_rack_5plus": "tab:red",
        "B_cross_rack_3to4": "tab:orange",
    }
    for axis, obsnum in zip(axes, obsnums, strict=True):
        rows = (
            selected[selected["obsnum"] == obsnum]
            .sort_values("refined_event_time_unix_sec")
            .copy()
        )
        sign = rows["dominant_projected_step_sign"].map(
            {"positive": 1.0, "negative": -1.0}
        )
        time_min = (
            rows["event_time_since_observation_start_sec"].to_numpy()
            / 60.0
        )
        axis.plot(
            time_min,
            sign,
            color="0.55",
            linewidth=0.7,
            alpha=0.8,
        )
        for tier, group in rows.groupby("quality_tier"):
            group_sign = group["dominant_projected_step_sign"].map(
                {"positive": 1.0, "negative": -1.0}
            )
            axis.scatter(
                group["event_time_since_observation_start_sec"] / 60.0,
                group_sign,
                s=13,
                alpha=0.8,
                color=colors.get(tier, "0.4"),
                label=tier.split("_", 1)[0],
            )
        metrics = _sign_sequence_metrics(rows.to_dict("records"))
        axis.set_title(
            f"{obsnum}: {len(rows)} events; adjacent sign alternation "
            f"{metrics['primary_adjacent_sign_alternation_fraction']:.3f} "
            f"(independent-sign expectation "
            f"{metrics['primary_iid_sign_alternation_expectation']:.3f})"
        )
        axis.set_yticks([-1.0, 1.0], ["negative", "positive"])
        axis.set_ylim(-1.45, 1.45)
        axis.set_xlabel("time since observation start (min)")
        axis.grid(axis="x", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=True))
        axis.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            ncols=2,
        )
    figure.suptitle(
        "Primary cross-rack event directions: telegraph-like alternation",
        fontsize=16,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _waveform_figure(
    path: Path,
    stack_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(stack_rows)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.0, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    for axis, network in zip(axes.flat, networks, strict=True):
        selected = frame[frame["network"] == network].sort_values(
            "time_from_network_onset_sec"
        )
        time = selected["time_from_network_onset_sec"].to_numpy(dtype=float)
        median = selected[
            "median_normalized_projected_phase"
        ].to_numpy(dtype=float)
        lower = selected["q16_normalized_projected_phase"].to_numpy(
            dtype=float
        )
        upper = selected["q84_normalized_projected_phase"].to_numpy(
            dtype=float
        )
        axis.fill_between(time, lower, upper, alpha=0.22)
        axis.plot(time, median, linewidth=1.5)
        axis.axvline(0.0, color="0.25", linestyle="--", linewidth=0.9)
        axis.axhline(0.0, color="0.55", linewidth=0.7)
        axis.axhline(1.0, color="0.55", linewidth=0.7)
        axis.set_title(f"nw{network}")
        axis.grid(alpha=0.22)
        axis.set_xlim(-1.0, 6.0)
    axes[0, 0].set_ylabel("projected phase / immediate step")
    axes[1, 0].set_ylabel("projected phase / immediate step")
    for axis in axes[1, :]:
        axis.set_xlabel("time from network onset (s)")
    figure.suptitle(
        "Median event waveform; band is the event-to-event 16–84% range"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _morphology_figure(
    path: Path,
    morphology_rows: list[dict[str, Any]],
    *,
    networks: list[int],
) -> None:
    frame = pd.DataFrame(morphology_rows)
    selected = frame[
        frame["network_participated_in_detection"].astype(bool)
        & frame["morphology_status"].eq("valid")
    ]
    lag_data = [
        selected[selected["network"] == network][
            "network_onset_lag_from_event_center_sec"
        ].dropna()
        for network in networks
    ]
    rise_data = [
        selected[selected["network"] == network][
            "rise_time_10_to_90_sec"
        ].dropna()
        for network in networks
    ]
    tau_data = [
        selected[
            (selected["network"] == network)
            & selected["recovery_fit_status"].eq("fit")
            & (selected["recovery_fit_r2"].fillna(-np.inf) >= 0.5)
        ]["recovery_tau_sec"].dropna()
        for network in networks
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.5),
        constrained_layout=True,
    )
    for axis, data, ylabel, title in (
        (
            axes[0],
            lag_data,
            "lag from event center (s)",
            "Cross-network onset lag",
        ),
        (
            axes[1],
            rise_data,
            "10–90% rise time (s)",
            "Projected step rise time",
        ),
        (
            axes[2],
            tau_data,
            "recovery tau (s)",
            "Accepted exponential recovery",
        ),
    ):
        positions = np.arange(1, len(networks) + 1)
        nonempty = [
            (position, values)
            for position, values in zip(positions, data, strict=True)
            if len(values)
        ]
        if nonempty:
            axis.boxplot(
                [values for _, values in nonempty],
                positions=[position for position, _ in nonempty],
                showfliers=False,
            )
        axis.set_xticks(positions, [f"nw{value}" for value in networks])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--apt-root", type=Path, required=True)
    parser.add_argument("--event-vector-dir", type=Path, required=True)
    parser.add_argument("--tone-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--networks",
        nargs="+",
        type=int,
        default=list(DEFAULT_NETWORKS),
    )
    parser.add_argument(
        "--event-rich-obsnums",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVENT_RICH_OBSNUMS),
    )
    parser.add_argument("--obsnums", nargs="+", type=int)
    parser.add_argument(
        "--require-all-apts",
        action="store_true",
        help=(
            "fail if any selected raw observation lacks its exact matched APT"
        ),
    )
    parser.add_argument("--step-window-sec", type=float, default=0.20)
    parser.add_argument("--step-guard-sec", type=float, default=0.05)
    parser.add_argument("--single-network-threshold", type=float, default=8.0)
    parser.add_argument("--single-network-prominence", type=float, default=3.0)
    parser.add_argument(
        "--single-network-minimum-separation-sec",
        type=float,
        default=0.60,
    )
    parser.add_argument("--coincidence-sec", type=float, default=0.25)
    parser.add_argument("--cluster-consume-sec", type=float, default=0.50)
    parser.add_argument("--known-match-tolerance-sec", type=float, default=0.50)
    parser.add_argument("--time-shift-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument("--waveform-pre-sec", type=float, default=2.0)
    parser.add_argument("--waveform-post-sec", type=float, default=8.0)
    parser.add_argument("--waveform-step-sec", type=float, default=0.02)
    parser.add_argument("--stack-minimum-snr", type=float, default=8.0)
    parser.add_argument("--telegraph-example-network", type=int, default=8)
    parser.add_argument("--telegraph-example-window-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    networks = [int(value) for value in args.networks]
    event_rich_obsnums = [
        int(value) for value in args.event_rich_obsnums
    ]
    if len(networks) != 6:
        raise ValueError("the standard figure layout requires six networks")
    if args.time_shift_permutations < 1:
        raise ValueError("--time-shift-permutations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tone_path = args.event_vector_dir / "science_event_tone_vectors.csv"
    known_path = args.event_vector_dir / "science_raw_event_clusters.csv"
    template_path = (
        args.tone_analysis_dir / "science_tone_rank_one_modes.csv"
    )
    event_tones = pd.read_csv(tone_path)
    known_events = pd.read_csv(known_path)
    fixed_templates = pd.read_csv(template_path)
    if set(event_tones["schema_version"].astype(str)) != {
        EVENT_VECTOR_SCHEMA
    }:
        raise ValueError("event-tone input has the wrong schema")
    if set(fixed_templates["schema_version"].astype(str)) != {
        TEMPLATE_SCHEMA
    }:
        raise ValueError("fixed-template input has the wrong schema")

    raw_by_observation = _find_raw_files(
        args.data_root,
        networks=networks,
    )
    if args.obsnums is not None:
        requested = {int(value) for value in args.obsnums}
        missing = sorted(requested - set(raw_by_observation))
        if missing:
            raise FileNotFoundError(
                f"requested raw observations are unavailable: {missing}"
            )
        raw_by_observation = {
            obsnum: paths
            for obsnum, paths in raw_by_observation.items()
            if obsnum in requested
        }
    requested_raw_by_observation = raw_by_observation
    requested_obsnums = sorted(requested_raw_by_observation)
    raw_by_observation, observation_inventory_rows = (
        _partition_observations_by_apt(
            requested_raw_by_observation,
            apt_root=args.apt_root,
        )
    )
    excluded_obsnums = [
        int(row["obsnum"])
        for row in observation_inventory_rows
        if row["analysis_status"] != "analyzed"
    ]
    if excluded_obsnums and args.require_all_apts:
        raise FileNotFoundError(
            "selected raw observations lack exact matched APTs: "
            f"{excluded_obsnums}"
        )
    if not raw_by_observation:
        raise FileNotFoundError(
            "none of the selected raw observations has an exact matched APT"
        )
    for obsnum in excluded_obsnums:
        print(
            f"obs {obsnum}: excluded because its exact matched APT is absent"
        )

    all_candidate_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    all_member_rows: list[dict[str, Any]] = []
    all_morphology_rows: list[dict[str, Any]] = []
    all_waveform_records: list[dict[str, Any]] = []
    observation_network_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    best_telegraph_example: dict[str, Any] | None = None
    excluded_input_rows = [
        {
            "obsnum": obsnum,
            "network": network,
            "raw": _file_identity(raw_path),
            "apt": None,
            "analysis_status": "excluded_missing_exact_matched_apt",
        }
        for obsnum in excluded_obsnums
        for network, raw_path in sorted(
            requested_raw_by_observation[obsnum].items()
        )
    ]

    for obsnum, raw_paths in sorted(raw_by_observation.items()):
        apt_path = args.apt_root / f"apt_{obsnum}_matched.ecsv"
        projections: dict[int, Projection] = {}
        candidate_rows: list[dict[str, Any]] = []
        for network in networks:
            template = _load_template(
                obsnum=obsnum,
                network=network,
                event_tones=event_tones,
                fixed_templates=fixed_templates,
                event_rich_obsnums=event_rich_obsnums,
            )
            projection = _project_network(
                obsnum=obsnum,
                network=network,
                raw_path=raw_paths[network],
                apt_path=apt_path,
                template=template,
                step_window_sec=float(args.step_window_sec),
                step_guard_sec=float(args.step_guard_sec),
            )
            projections[network] = projection
            rows = _network_candidate_rows(
                projection,
                threshold=float(args.single_network_threshold),
                prominence=float(args.single_network_prominence),
                minimum_separation_sec=float(
                    args.single_network_minimum_separation_sec
                ),
            )
            candidate_rows.extend(rows)
            observation_network_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "obsnum": obsnum,
                    "network": network,
                    "observation_start_unix_sec": float(
                        projection.time_unix_sec[0]
                    ),
                    "observation_end_unix_sec": float(
                        projection.time_unix_sec[-1]
                    ),
                    "duration_sec": float(
                        projection.time_unix_sec[-1]
                        - projection.time_unix_sec[0]
                    ),
                    "sample_frequency_hz": projection.sample_frequency_hz,
                    "sample_count": int(
                        len(projection.time_unix_sec)
                    ),
                    "template_source": projection.template_source,
                    "template_training_obsnums": " ".join(
                        str(value)
                        for value in projection.template_training_obsnums
                    ),
                    "template_tone_count": projection.template_tone_count,
                    "step_center_rad": projection.step_center_rad,
                    "step_sigma_rad": projection.step_sigma_rad,
                    "single_network_candidate_count": int(len(rows)),
                    "raw_path": str(projection.raw_path),
                    "apt_path": str(projection.apt_path),
                }
            )
            input_rows.append(
                {
                    "obsnum": obsnum,
                    "network": network,
                    "raw": _file_identity(projection.raw_path),
                    "apt": _file_identity(apt_path),
                }
            )
        event_rows, member_rows = _cluster_candidate_rows(
            candidate_rows,
            obsnum=obsnum,
            coincidence_sec=float(args.coincidence_sec),
            consume_sec=float(args.cluster_consume_sec),
        )
        morphology, waveform_records = _event_morphology(
            event_rows=event_rows,
            member_rows=member_rows,
            projections=projections,
            waveform_pre_sec=float(args.waveform_pre_sec),
            waveform_post_sec=float(args.waveform_post_sec),
            waveform_step_sec=float(args.waveform_step_sec),
            stack_minimum_snr=float(args.stack_minimum_snr),
        )
        example_network = int(args.telegraph_example_network)
        if example_network in projections:
            example = _telegraph_example_candidate(
                projection=projections[example_network],
                event_rows=event_rows,
                window_sec=float(args.telegraph_example_window_sec),
            )
            if example is not None and (
                best_telegraph_example is None
                or example["event_count"]
                > best_telegraph_example["event_count"]
            ):
                best_telegraph_example = example
        start = float(
            min(
                projection.time_unix_sec[0]
                for projection in projections.values()
            )
        )
        end = float(
            max(
                projection.time_unix_sec[-1]
                for projection in projections.values()
            )
        )
        primary = [
            row for row in event_rows if row["primary_event_candidate"]
        ]
        tier_a = [
            row
            for row in event_rows
            if row["quality_tier"] == "A_cross_rack_5plus"
        ]
        shifted_primary, shifted_tier_a = _shifted_cluster_counts(
            candidate_rows,
            observation_start_sec=start,
            observation_end_sec=end,
            coincidence_sec=float(args.coincidence_sec),
            consume_sec=float(args.cluster_consume_sec),
            n_permutations=int(args.time_shift_permutations),
            random_seed=int(args.random_seed) + obsnum,
        )
        duration_min = (end - start) / 60.0
        observation_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "obsnum": obsnum,
                "observation_start_unix_sec": start,
                "observation_start_utc": datetime.fromtimestamp(
                    start,
                    tz=UTC,
                ).isoformat(),
                "observation_end_unix_sec": end,
                "duration_min": duration_min,
                "single_network_candidate_count": int(
                    len(candidate_rows)
                ),
                "two_plus_network_cluster_count": int(len(event_rows)),
                "primary_cross_rack_3plus_event_count": int(len(primary)),
                "tier_a_cross_rack_5plus_event_count": int(len(tier_a)),
                "primary_event_rate_per_min": float(
                    len(primary) / duration_min
                ),
                "tier_a_event_rate_per_min": float(
                    len(tier_a) / duration_min
                ),
                **_sign_sequence_metrics(primary),
                "primary_unanimous_network_sign_fraction": (
                    float(
                        np.mean(
                            [
                                row[
                                    "network_member_sign_is_unanimous"
                                ]
                                for row in primary
                            ]
                        )
                    )
                    if primary
                    else None
                ),
                "shift_null_primary_count_median": float(
                    np.median(shifted_primary)
                ),
                "shift_null_primary_count_p95": float(
                    np.quantile(shifted_primary, 0.95)
                ),
                "shift_null_primary_rate_p95_per_min": float(
                    np.quantile(shifted_primary, 0.95) / duration_min
                ),
                "shift_null_tier_a_count_median": float(
                    np.median(shifted_tier_a)
                ),
                "shift_null_tier_a_count_p95": float(
                    np.quantile(shifted_tier_a, 0.95)
                ),
                "primary_count_exceeds_all_shift_nulls": bool(
                    len(primary) > np.max(shifted_primary)
                ),
                "tier_a_count_exceeds_all_shift_nulls": bool(
                    len(tier_a) > np.max(shifted_tier_a)
                ),
            }
        )
        all_candidate_rows.extend(candidate_rows)
        all_event_rows.extend(event_rows)
        all_member_rows.extend(member_rows)
        all_morphology_rows.extend(morphology)
        all_waveform_records.extend(waveform_records)
        del projections
        gc.collect()
        print(
            f"obs {obsnum}: candidates={len(candidate_rows)} "
            f"primary={len(primary)} tier_a={len(tier_a)}"
        )

    recall_rows = _known_event_recall_rows(
        known_events,
        all_event_rows,
        match_tolerance_sec=float(args.known_match_tolerance_sec),
    )
    recall = pd.DataFrame(recall_rows)
    for row in observation_rows:
        selected = recall[recall["obsnum"] == row["obsnum"]]
        row["previously_known_event_count"] = int(len(selected))
        row["previously_known_event_recovered_count"] = int(
            selected["matched_within_tolerance"].sum()
        )
        row["previously_known_event_recall"] = (
            float(selected["matched_within_tolerance"].mean())
            if len(selected)
            else None
        )
    morphology_summary = _morphology_summary_rows(all_morphology_rows)
    morphology_observation_summary = _morphology_summary_rows(
        all_morphology_rows,
        group_by_observation=True,
    )
    all_stack_rows, all_example_rows = _summarize_waveforms(
        all_waveform_records,
        networks=networks,
    )
    if best_telegraph_example is None:
        raise ValueError("no telegraph example could be selected")

    output_names = {
        "network_candidates": "continuous_single_network_candidates.csv",
        "event_catalog": "continuous_event_catalog.csv",
        "event_members": "continuous_event_network_members.csv",
        "observation_network_summary": (
            "continuous_observation_network_summary.csv"
        ),
        "observation_inventory": "continuous_observation_inventory.csv",
        "observation_summary": "continuous_observation_summary.csv",
        "known_recall": "continuous_known_event_recall.csv",
        "morphology": "event_network_temporal_morphology.csv",
        "morphology_summary": "event_temporal_morphology_summary.csv",
        "morphology_observation_summary": (
            "event_observation_network_temporal_morphology_summary.csv"
        ),
        "waveform_stack": "event_projected_waveform_stack.csv",
        "waveform_examples": "event_projected_waveform_examples.csv",
        "telegraph_example_samples": "telegraph_example_projected_phase.csv",
        "telegraph_example_events": "telegraph_example_events.csv",
        "chronology_figure": "continuous_event_chronology.png",
        "sign_sequence_figure": "event_step_sign_sequences.png",
        "telegraph_example_figure": "telegraph_example_projected_phase.png",
        "waveform_figure": "event_projected_waveform_stacks.png",
        "morphology_figure": "event_temporal_morphology.png",
    }
    tables: dict[str, list[dict[str, Any]]] = {
        "network_candidates": all_candidate_rows,
        "event_catalog": all_event_rows,
        "event_members": all_member_rows,
        "observation_network_summary": observation_network_rows,
        "observation_inventory": observation_inventory_rows,
        "observation_summary": observation_rows,
        "known_recall": recall_rows,
        "morphology": all_morphology_rows,
        "morphology_summary": morphology_summary,
        "morphology_observation_summary": morphology_observation_summary,
        "waveform_stack": all_stack_rows,
        "waveform_examples": all_example_rows,
        "telegraph_example_samples": best_telegraph_example["sample_rows"],
        "telegraph_example_events": best_telegraph_example["marker_rows"],
    }
    for key, rows in tables.items():
        _write_csv(args.output_dir / output_names[key], rows)
    _chronology_figure(
        args.output_dir / output_names["chronology_figure"],
        all_event_rows,
        observation_rows,
    )
    _sign_sequence_figure(
        args.output_dir / output_names["sign_sequence_figure"],
        all_event_rows,
    )
    _telegraph_example_figure(
        args.output_dir / output_names["telegraph_example_figure"],
        best_telegraph_example,
    )
    _waveform_figure(
        args.output_dir / output_names["waveform_figure"],
        all_stack_rows,
        networks=networks,
    )
    _morphology_figure(
        args.output_dir / output_names["morphology_figure"],
        all_morphology_rows,
        networks=networks,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "description": (
            "Full-duration UID-template raw-I/Q event catalog and projected "
            "temporal morphology"
        ),
        "semantics": {
            "catalog_scope": (
                "complete durations for requested observations having exact "
                "matched APTs, but selected for the learned science UID mode; "
                "observations without exact matched APTs are inventoried and "
                "excluded, and the catalog is not complete for orthogonal "
                "pathologies"
            ),
            "apt_policy": (
                "require the exact observation-specific matched APT; never "
                "borrow or infer detector identity from another observation"
            ),
            "event_rich_template_policy": (
                "leave one observation out; known RTC-guided times are not "
                "used for continuous detection"
            ),
            "other_template_policy": (
                "fixed all-event science UID loading, without observation-"
                "local tone-shape refitting"
            ),
            "primary_event": (
                "step candidates within the coincidence window in at least "
                "three affected networks and both electronics racks"
            ),
            "tier_a": (
                "primary event with at least five affected networks"
            ),
            "shift_null": (
                "network candidate times circularly shifted independently; "
                "preserves per-network candidate count and score"
            ),
            "projected_phase": (
                "radians per RMS-normalized UID loading; an operational "
                "template coordinate, not calibrated detector phase"
            ),
            "normalized_waveform": (
                "detrended projected phase divided by the signed immediate "
                "post-minus-pre step"
            ),
            "telegraph_example": (
                "network chosen by parameter; observation and window chosen "
                "by maximum primary-event count, then linearly detrended for "
                "display without changing event detection"
            ),
        },
        "parameters": {
            "networks": networks,
            "requested_obsnums": requested_obsnums,
            "analyzed_obsnums": sorted(raw_by_observation),
            "excluded_missing_exact_matched_apt_obsnums": excluded_obsnums,
            "event_rich_obsnums": event_rich_obsnums,
            "step_window_sec": float(args.step_window_sec),
            "step_guard_sec": float(args.step_guard_sec),
            "single_network_threshold": float(
                args.single_network_threshold
            ),
            "single_network_prominence": float(
                args.single_network_prominence
            ),
            "single_network_minimum_separation_sec": float(
                args.single_network_minimum_separation_sec
            ),
            "coincidence_sec": float(args.coincidence_sec),
            "cluster_consume_sec": float(args.cluster_consume_sec),
            "known_match_tolerance_sec": float(
                args.known_match_tolerance_sec
            ),
            "time_shift_permutations": int(
                args.time_shift_permutations
            ),
            "random_seed": int(args.random_seed),
            "waveform_pre_sec": float(args.waveform_pre_sec),
            "waveform_post_sec": float(args.waveform_post_sec),
            "waveform_step_sec": float(args.waveform_step_sec),
            "stack_minimum_snr": float(args.stack_minimum_snr),
            "telegraph_example_network": int(
                args.telegraph_example_network
            ),
            "telegraph_example_window_sec": float(
                args.telegraph_example_window_sec
            ),
        },
        "inputs": {
            "event_tones": _file_identity(tone_path),
            "known_events": _file_identity(known_path),
            "fixed_templates": _file_identity(template_path),
            "raw_and_apt_files": input_rows,
            "excluded_raw_files": excluded_input_rows,
        },
        "counts": {key: len(rows) for key, rows in tables.items()},
        "outputs": output_names,
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
