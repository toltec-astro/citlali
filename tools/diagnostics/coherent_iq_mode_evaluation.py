#!/usr/bin/env python3
"""Evaluate the observe-only coherent-I/Q mode classifier.

Positive examples are the 52 synchronized event clusters from the Investigate
corpus.  Scores are cross-validated by fitting templates on alternating event
halves.  Negative examples are fixed, provenance-recorded epochs in selected
scans of the quiet observations 152390 and 152392.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from astropy.table import Table

from tools.diagnostics.coherent_iq_mode_observer import (
    attach_cross_network_coincidence,
    fit_rank_modes,
    make_template,
    score_event,
)


SCHEMA_VERSION = "citlali-coherent-iq-mode-evaluation-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty {path}")
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


def _network_groups(
    tone_rows: list[dict[str, str]], network: int
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tone_rows:
        if int(row["network"]) == int(network):
            result[row["event_cluster_id"]].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["uid"]))
    return dict(result)


def _template_from_training(
    groups: dict[str, list[dict[str, str]]],
    training_ids: list[str],
    *,
    network: int,
    fold: int | str,
    rank: int = 1,
) -> dict[str, Any]:
    common_uids = set(int(row["uid"]) for row in groups[training_ids[0]])
    for event_id in training_ids[1:]:
        common_uids &= {int(row["uid"]) for row in groups[event_id]}
    ordered_uids = np.asarray(sorted(common_uids), dtype=int)
    if ordered_uids.size < 16:
        raise ValueError(f"nw{network} fold {fold}: too few common UIDs")
    matrices = []
    coordinates: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for event_id in training_ids:
        by_uid = {int(row["uid"]): row for row in groups[event_id]}
        matrices.append(
            [float(by_uid[int(uid)]["phase_change_mrad"]) for uid in ordered_uids]
        )
        for uid in ordered_uids:
            row = by_uid[int(uid)]
            coordinates[int(uid)].append(
                (
                    int(row["tone_slot_zero_based"]),
                    float(row["tone_offset_frequency_hz"]),
                    float(row["probe_frequency_hz"]),
                )
            )
    matrix = np.asarray(matrices, dtype=float)
    modes, energy = fit_rank_modes(matrix, rank=rank)
    slots = []
    offsets = []
    probes = []
    for uid in ordered_uids:
        values = coordinates[int(uid)]
        slots.append(int(round(float(np.median([row[0] for row in values])))))
        offsets.append(float(np.median([row[1] for row in values])))
        probes.append(float(np.median([row[2] for row in values])))
    return make_template(
        template_id=f"ngc4449-nw{network}-phase-crossval-fold-{fold}",
        template_version="2026-07-30.1",
        network=network,
        uids=ordered_uids,
        tone_slots=slots,
        tone_offsets_hz=offsets,
        probe_frequencies_hz=probes,
        modes=modes,
        training={
            "dataset": "NGC4449 52-event raw-I/Q corpus",
            "event_ids": training_ids,
            "event_count": len(training_ids),
            "rank_energy_fractions": energy.tolist(),
            "method": "uncentered SVD; alternating-half cross-validation",
        },
        validation={"status": "cross_validation"},
        provenance={"investigate_commit": "422f25f5f"},
        tone_offset_tolerance_hz=500_000.0,
        minimum_compatible_tone_fraction=0.75,
    )


def _score_group(
    template: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    score = score_event(
        template,
        network=int(rows[0]["network"]),
        uids=[int(row["uid"]) for row in rows],
        tone_offsets_hz=[
            float(row["tone_offset_frequency_hz"]) for row in rows
        ],
        phase_change_mrad=[
            float(row["phase_change_mrad"]) for row in rows
        ],
    )
    return score.as_dict()


def _template_mode_by_uid(
    template: dict[str, Any], mode_id: str = "phase_mode_1"
) -> dict[int, float]:
    return {
        int(row["uid"]): float(row["loadings"][mode_id])
        for row in template["tone_coordinate"]["tones"]
    }


def _template_loading_cosine(
    left: dict[str, Any], right: dict[str, Any]
) -> float:
    left_mode = _template_mode_by_uid(left)
    right_mode = _template_mode_by_uid(right)
    uids = sorted(set(left_mode) & set(right_mode))
    if len(uids) < 3:
        return math.nan
    x = np.asarray([left_mode[uid] for uid in uids], dtype=float)
    y = np.asarray([right_mode[uid] for uid in uids], dtype=float)
    return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))


def _cluster_members(clusters: list[dict[str, str]]) -> dict[str, set[int]]:
    return {
        row["event_cluster_id"]: {
            int(value) for value in row["networks"].split()
        }
        for row in clusters
    }


def cross_validated_event_scores(
    tone_rows: list[dict[str, str]],
    clusters: list[dict[str, str]],
    networks: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    members = _cluster_members(clusters)
    results: list[dict[str, Any]] = []
    full_templates: dict[int, dict[str, Any]] = {}
    for network in networks:
        groups = _network_groups(tone_rows, network)
        event_ids = sorted(groups)
        folds = {
            0: event_ids[::2],
            1: event_ids[1::2],
        }
        templates = {
            fold: _template_from_training(
                groups,
                folds[1 - fold],
                network=network,
                fold=fold,
            )
            for fold in (0, 1)
        }
        full_templates[network] = _template_from_training(
            groups, event_ids, network=network, fold="all"
        )
        full_templates[network]["validation"].update(
            {
                "alternating_half_loading_cosine": _template_loading_cosine(
                    templates[0], templates[1]
                ),
                "cross_validation_event_count": len(event_ids),
            }
        )
        for fold, test_ids in folds.items():
            for event_id in test_ids:
                rows = groups[event_id]
                score = _score_group(templates[fold], rows)
                responsive = sum(
                    row["phase_responsive"] == "True" for row in rows
                )
                score.update(
                    {
                        "example_class": (
                            "cluster_member"
                            if network in members[event_id]
                            else "shared_epoch_nonmember"
                        ),
                        "event_cluster_id": event_id,
                        "obsnum": int(rows[0]["obsnum"]),
                        "citlali_scan_one_based": int(
                            rows[0]["citlali_scan_one_based"]
                        ),
                        "event_time_unix_sec": float(
                            next(
                                row["cluster_time_unix_sec"]
                                for row in clusters
                                if row["event_cluster_id"] == event_id
                            )
                        ),
                        "cross_validation_fold": fold,
                        "phase_responsive_tone_count": responsive,
                        "phase_responsive_tone_fraction": responsive / len(rows),
                    }
                )
                score["mode_selected_descriptive_0p6_5mrad"] = (
                    score["status"] == "scored"
                    and float(score["absolute_cosine_similarity"]) >= 0.6
                    and abs(float(score["projection_amplitude_mrad"])) >= 5.0
                )
                results.append(score)
    attach_cross_network_coincidence(
        results,
        selection_field="mode_selected_descriptive_0p6_5mrad",
    )
    return results, full_templates


def _apt_tone_identity(
    apt: Table,
    *,
    network: int,
    n_tones: int,
) -> tuple[np.ndarray, np.ndarray]:
    uids = np.full(n_tones, -1, dtype=int)
    usable = np.zeros(n_tones, dtype=bool)
    rows = apt[np.asarray(apt["nw"], dtype=int) == int(network)]
    for row in rows:
        tone_value = float(row["kids_tone"])
        if not math.isfinite(tone_value):
            continue
        tone = int(tone_value)
        if not 0 <= tone < n_tones:
            continue
        uid_value = float(row["uid"])
        kids_flag = float(row["kids_flag"]) if "kids_flag" in rows.colnames else 0.0
        map_flag = float(row["flag"]) if "flag" in rows.colnames else 0.0
        if math.isfinite(uid_value):
            uids[tone] = int(uid_value)
        usable[tone] = (
            math.isfinite(kids_flag)
            and math.isfinite(map_flag)
            and kids_flag == 0.0
            and map_flag == 0.0
        )
    return uids, usable


def _phase_vector(
    time: np.ndarray,
    iq: np.ndarray,
    event_time: float,
    *,
    pre_sec: float = 0.2,
    guard_sec: float = 0.05,
    post_sec: float = 0.2,
) -> np.ndarray:
    pre = (
        (time >= event_time - guard_sec - pre_sec)
        & (time < event_time - guard_sec)
    )
    post = (
        (time > event_time + guard_sec)
        & (time <= event_time + guard_sec + post_sec)
    )
    if np.count_nonzero(pre) < 4 or np.count_nonzero(post) < 4:
        raise ValueError("null epoch does not have a complete phase window")
    before = np.mean(iq[pre, :], axis=0)
    after = np.mean(iq[post, :], axis=0)
    return np.angle(after / before) * 1.0e3


def _scan_intervals_from_times(
    telescope_time_sec: np.ndarray,
    scan_duration_sec: np.ndarray,
) -> list[tuple[float, float]]:
    telescope_time_sec = np.asarray(telescope_time_sec, dtype=float)
    scan_duration_sec = np.asarray(scan_duration_sec, dtype=float)
    if telescope_time_sec.size < 4 or scan_duration_sec.size == 0:
        raise ValueError("telescope time or scan duration is empty")
    dt = float(np.median(np.diff(telescope_time_sec)))
    nominal_duration = float(np.median(scan_duration_sec))
    period_samples = int(round((nominal_duration + dt) / dt))
    required = scan_duration_sec.size * period_samples
    if period_samples < 2 or required > telescope_time_sec.size:
        raise ValueError("cannot reconstruct Citlali scan intervals")
    result = []
    for scan_row, duration in enumerate(scan_duration_sec):
        start_index = scan_row * period_samples
        end_index = start_index + period_samples - 1
        start = float(telescope_time_sec[start_index])
        end = float(telescope_time_sec[end_index])
        trim = max(0.0, end - start - float(duration))
        if scan_row == 0:
            start += trim
        elif scan_row == scan_duration_sec.size - 1:
            end -= trim
        result.append((start, end))
    return result


def quiet_scan_scores(
    manifest: dict[str, Any],
    scan_selection: list[dict[str, str]],
    templates: dict[int, dict[str, Any]],
    *,
    epochs_per_scan: int,
) -> list[dict[str, Any]]:
    inputs = {int(row["obsnum"]): row for row in manifest["inputs"]}
    selected = [
        row for row in scan_selection
        if int(row["obsnum"]) in {152390, 152392}
        and row["selected_for_raw_analysis"] == "True"
    ]
    results: list[dict[str, Any]] = []
    for selection in selected:
        obsnum = int(selection["obsnum"])
        scan_one = int(selection["citlali_scan_one_based"])
        entry = inputs[obsnum]
        with netCDF4.Dataset(entry["rtc_path"]) as rtc:
            output_scan = np.asarray(rtc.variables["output_scan_index"][:], dtype=int)
            durations = np.asarray(rtc.variables["scan_duration_s"][:], dtype=float)
            matches = np.flatnonzero(output_scan == scan_one)
            if matches.size != 1:
                raise ValueError(
                    f"obs {obsnum} scan {scan_one}: RTC row is not unique"
                )
            scan_row = int(matches[0])
        with netCDF4.Dataset(entry["telescope_path"]) as telescope:
            tel_time = np.asarray(
                telescope.variables["Data.TelescopeBackend.TelTime"][:],
                dtype=float,
            )
        intervals = _scan_intervals_from_times(tel_time, durations)
        scan_start, scan_stop = intervals[scan_row]
        apt = Table.read(entry["apt_path"], format="ascii.ecsv")
        for network, template in templates.items():
            raw_path = Path(entry["raw_paths"][str(network)])
            with netCDF4.Dataset(raw_path) as raw:
                recv = np.asarray(
                    raw.variables["Data.Toltec.RecvTime"][:], dtype=float
                )
                inside = np.flatnonzero(
                    (recv >= scan_start) & (recv <= scan_stop)
                )
                if inside.size < 32:
                    continue
                first, last = int(inside[0]), int(inside[-1]) + 1
                time = recv[first:last]
                i_data = np.asarray(
                    raw.variables["Data.Toltec.Is"][first:last, :], dtype=float
                )
                q_data = np.asarray(
                    raw.variables["Data.Toltec.Qs"][first:last, :], dtype=float
                )
                offsets = np.asarray(
                    raw.variables["Header.Toltec.ToneFreq"][0, :], dtype=float
                )
            uids, usable = _apt_tone_identity(
                apt, network=network, n_tones=i_data.shape[1]
            )
            lo = max(float(time[0]) + 0.3, scan_start + 0.3)
            hi = min(float(time[-1]) - 0.3, scan_stop - 0.3)
            if hi <= lo:
                continue
            epochs = np.linspace(lo, hi, epochs_per_scan + 2)[1:-1]
            for epoch_index, epoch in enumerate(epochs, start=1):
                phase = _phase_vector(time, i_data + 1j * q_data, float(epoch))
                valid = usable & (uids >= 0) & np.isfinite(phase)
                score = score_event(
                    template,
                    network=network,
                    uids=uids[valid],
                    tone_offsets_hz=offsets[valid],
                    phase_change_mrad=phase[valid],
                ).as_dict()
                score.update(
                    {
                        "example_class": "quiet_scan_epoch",
                        "event_cluster_id": "",
                        "obsnum": obsnum,
                        "citlali_scan_one_based": scan_one,
                        "event_time_unix_sec": float(epoch),
                        "null_epoch_index": epoch_index,
                        "phase_responsive_tone_count": "",
                        "phase_responsive_tone_fraction": "",
                    }
                )
                score["mode_selected_descriptive_0p6_5mrad"] = (
                    score["status"] == "scored"
                    and float(score["absolute_cosine_similarity"]) >= 0.6
                    and abs(float(score["projection_amplitude_mrad"])) >= 5.0
                )
                results.append(score)
    attach_cross_network_coincidence(
        results,
        selection_field="mode_selected_descriptive_0p6_5mrad",
    )
    return results


def _rtc_scan_record_metrics(
    manifest: dict[str, Any],
    event_scores: list[dict[str, Any]],
) -> None:
    inputs = {int(row["obsnum"]): row for row in manifest["inputs"]}
    events_per_key = Counter(
        (
            int(row["obsnum"]),
            int(row["citlali_scan_one_based"]),
            int(row["network"]),
        )
        for row in event_scores
    )
    cache: dict[tuple[int, int, int], tuple[int, int]] = {}
    for row in event_scores:
        key = (
            int(row["obsnum"]),
            int(row["citlali_scan_one_based"]),
            int(row["network"]),
        )
        if key not in cache:
            obsnum, scan_one, network = key
            with netCDF4.Dataset(inputs[obsnum]["rtc_path"]) as rtc:
                output_scan = np.asarray(
                    rtc.variables["output_scan_index"][:], dtype=int
                )
                scan_rows = np.flatnonzero(output_scan == scan_one)
                if scan_rows.size != 1:
                    cache[key] = (0, 0)
                    continue
                scan_row = int(scan_rows[0])
                nw = np.asarray(rtc.variables["apt_nw"][:], dtype=int)
                selected = nw == network
                raw_count = np.asarray(
                    rtc.variables[
                        "rtc_despike_local_raw_accepted_event_count"
                    ][scan_row, :],
                    dtype=int,
                )
                delta_count = np.asarray(
                    rtc.variables[
                        "rtc_despike_local_delta_accepted_event_count"
                    ][scan_row, :],
                    dtype=int,
                )
                flagged_count = np.asarray(
                    rtc.variables[
                        "rtc_despike_local_flagged_sample_count"
                    ][scan_row, :],
                    dtype=int,
                )
                records = int(
                    np.sum(np.maximum(raw_count[selected], 0))
                    + np.sum(np.maximum(delta_count[selected], 0))
                )
                flagged = int(np.sum(np.maximum(flagged_count[selected], 0)))
                cache[key] = (records, flagged)
        records, flagged = cache[key]
        divisor = max(1, events_per_key[key])
        row["rtc_accepted_local_event_records_in_scan"] = records
        row["rtc_final_flagged_detector_samples_in_scan"] = flagged
        row["rtc_records_per_coherent_event_estimate"] = records / divisor
        row["compact_record_reduction_factor_estimate"] = records / divisor


def _classified(
    row: dict[str, Any], *, cosine: float, amplitude: float
) -> bool:
    return (
        row["status"] == "scored"
        and float(row["absolute_cosine_similarity"]) >= cosine
        and abs(float(row["projection_amplitude_mrad"])) >= amplitude
    )


def threshold_rows(
    event_scores: list[dict[str, Any]],
    null_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positives = [
        row for row in event_scores if row["example_class"] == "cluster_member"
    ]
    low_amplitude = [
        row
        for row in event_scores
        if row["example_class"] == "shared_epoch_nonmember"
    ]
    rows = []
    for cosine in (0.4, 0.5, 0.6, 0.7, 0.8):
        for amplitude in (2.0, 5.0, 10.0, 20.0):
            tp = sum(
                _classified(row, cosine=cosine, amplitude=amplitude)
                for row in positives
            )
            fp = sum(
                _classified(row, cosine=cosine, amplitude=amplitude)
                for row in null_scores
            )
            low = sum(
                _classified(row, cosine=cosine, amplitude=amplitude)
                for row in low_amplitude
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "minimum_absolute_cosine": cosine,
                    "minimum_absolute_amplitude_mrad": amplitude,
                    "true_positive": tp,
                    "false_negative": len(positives) - tp,
                    "false_positive": fp,
                    "true_negative": len(null_scores) - fp,
                    "recall": tp / len(positives) if positives else math.nan,
                    "false_positive_rate": (
                        fp / len(null_scores) if null_scores else math.nan
                    ),
                    "shared_epoch_nonmember_detected": low,
                    "shared_epoch_nonmember_total": len(low_amplitude),
                }
            )
    return rows


def network_summary(
    scores: list[dict[str, Any]],
    rank_summary: list[dict[str, str]],
    templates: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rank_by_network = {int(row["network"]): row for row in rank_summary}
    result = []
    for network in sorted({int(row["network"]) for row in scores}):
        classes = sorted(
            {
                row["example_class"]
                for row in scores
                if int(row["network"]) == network
            }
        )
        for example_class in classes:
            rows = [
                row for row in scores
                if int(row["network"]) == network
                and row["example_class"] == example_class
            ]
            scored = [row for row in rows if row["status"] == "scored"]
            amplitudes = np.asarray(
                [abs(float(row["projection_amplitude_mrad"])) for row in scored]
            )
            cosines = np.asarray(
                [float(row["absolute_cosine_similarity"]) for row in scored]
            )
            rank = rank_by_network.get(network, {})
            template = templates[network]
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "network": network,
                    "example_class": example_class,
                    "example_count": len(rows),
                    "scored_count": len(scored),
                    "median_absolute_amplitude_mrad": (
                        float(np.median(amplitudes))
                        if amplitudes.size
                        else math.nan
                    ),
                    "median_absolute_cosine": (
                        float(np.median(cosines))
                        if cosines.size
                        else math.nan
                    ),
                    "fraction_abs_cosine_ge_0p6": (
                        float(np.mean(cosines >= 0.6))
                        if cosines.size
                        else math.nan
                    ),
                    "fraction_abs_cosine_ge_0p6_and_amp_ge_5mrad": (
                        float(np.mean((cosines >= 0.6) & (amplitudes >= 5.0)))
                        if cosines.size
                        else math.nan
                    ),
                    "training_rank1_energy_fraction": rank.get(
                        "phase_rank1_energy_fraction",
                        template["training"]["rank_energy_fractions"][0],
                    ),
                    "split_half_loading_cosine": rank.get(
                        "phase_rank1_split_half_loading_cosine",
                        template["validation"][
                            "alternating_half_loading_cosine"
                        ],
                    ),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-vector-dir", type=Path, required=True)
    parser.add_argument("--tone-susceptibility-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs-per-null-scan", type=int, default=5)
    parser.add_argument(
        "--networks",
        type=int,
        nargs="+",
        help=(
            "network IDs to evaluate; default is every network represented "
            "in the event-tone corpus"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tone_rows = read_csv(args.event_vector_dir / "science_event_tone_vectors.csv")
    clusters = read_csv(args.event_vector_dir / "science_raw_event_clusters.csv")
    scan_selection = read_csv(
        args.event_vector_dir / "science_event_scan_selection.csv"
    )
    rank_summary = read_csv(
        args.tone_susceptibility_dir / "science_tone_rank_one_summary.csv"
    )
    with (args.event_vector_dir / "manifest.json").open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    available_networks = tuple(
        sorted({int(row["network"]) for row in tone_rows})
    )
    networks = (
        tuple(dict.fromkeys(args.networks))
        if args.networks
        else available_networks
    )
    unavailable_networks = sorted(set(networks) - set(available_networks))
    if unavailable_networks:
        raise ValueError(
            "requested networks absent from tone-vector corpus: "
            + " ".join(map(str, unavailable_networks))
        )

    event_scores, templates = cross_validated_event_scores(
        tone_rows, clusters, networks
    )
    _rtc_scan_record_metrics(source_manifest, event_scores)
    null_scores = quiet_scan_scores(
        source_manifest,
        scan_selection,
        templates,
        epochs_per_scan=args.epochs_per_null_scan,
    )
    all_scores = event_scores + null_scores
    thresholds = threshold_rows(event_scores, null_scores)
    summaries = network_summary(all_scores, rank_summary, templates)

    write_csv(args.output_dir / "coherent_mode_scores.csv", all_scores)
    write_csv(args.output_dir / "coherent_mode_threshold_grid.csv", thresholds)
    write_csv(args.output_dir / "coherent_mode_network_summary.csv", summaries)
    for network, template in templates.items():
        with (
            args.output_dir / f"coherent_mode_template_nw{network}.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(template, handle, indent=2, sort_keys=True)
            handle.write("\n")

    selected = min(
        thresholds,
        key=lambda row: (
            abs(float(row["minimum_absolute_cosine"]) - 0.6),
            abs(float(row["minimum_absolute_amplitude_mrad"]) - 5.0),
        ),
    )
    status_counts = Counter(row["status"] for row in null_scores)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "classification_role": "observe_only; no flags, weights, data, or maps changed",
        "positive_definition": "network is a threshold-triggering member of a synchronized 52-event cluster",
        "ambiguous_definition": "network measured at a shared event epoch but did not independently trigger cluster membership",
        "negative_definition": "fixed epochs in selected scans of quiet observations 152390 and 152392",
        "cross_validation": "alternating event halves; each positive score uses a template fit without that event half",
        "descriptive_operating_point": {
            "minimum_absolute_cosine": 0.6,
            "minimum_absolute_amplitude_mrad": 5.0,
            "confusion": selected,
            "not_production_policy": True,
        },
        "counts": {
            "available_networks": list(available_networks),
            "evaluated_networks": list(networks),
            "instrument_networks_not_in_corpus": sorted(
                set(range(13)) - set(available_networks)
            ),
            "event_network_scores": len(event_scores),
            "cluster_member_positive_scores": sum(
                row["example_class"] == "cluster_member"
                for row in event_scores
            ),
            "shared_epoch_nonmember_scores": sum(
                row["example_class"] == "shared_epoch_nonmember"
                for row in event_scores
            ),
            "quiet_scan_epoch_scores": len(null_scores),
            "quiet_score_status": dict(status_counts),
        },
        "inputs": {
            "event_vector_dir": str(args.event_vector_dir.resolve()),
            "tone_susceptibility_dir": str(
                args.tone_susceptibility_dir.resolve()
            ),
            "investigate_commit": "422f25f5f",
        },
        "outputs": {
            "scores": "coherent_mode_scores.csv",
            "threshold_grid": "coherent_mode_threshold_grid.csv",
            "network_summary": "coherent_mode_network_summary.csv",
            "templates": "coherent_mode_template_nwN.json",
        },
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(
        json.dumps(
            manifest["descriptive_operating_point"]["confusion"],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
