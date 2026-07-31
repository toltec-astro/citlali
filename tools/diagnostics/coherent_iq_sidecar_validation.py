#!/usr/bin/env python3
"""Validate a production coherent-I/Q sidecar against offline event evidence.

The comparison keeps three questions separate:

* candidate-time recovery against the curated event-vector corpus;
* broader overlap with the independently detected continuous-event catalog;
* stability of network mode-shape and absolute-amplitude scores.

Unmatched sidecar candidates are deliberately not called false positives.  The
sidecar is RTC-seeded across every available network, whereas the continuous
catalog is selected for one learned six-network step-mode family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCHEMA_VERSION = "citlali-coherent-iq-sidecar-validation-v1"
SIDECAR_SCHEMA_VERSIONS = {
    "citlali-coherent-iq-mode-sidecar-v1",
    "citlali-coherent-iq-mode-sidecar-v2",
}


@dataclass(frozen=True)
class MatchSummary:
    reference_count: int
    matched_count: int
    recall: float
    median_signed_residual_sec: float | None
    median_absolute_residual_sec: float | None
    p95_absolute_residual_sec: float | None


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _network_set(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {int(token) for token in str(value).split() if token}


def _rack(network: int) -> str:
    return "RACKA" if int(network) <= 6 else "RACKO"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean_series(values: pd.Series) -> pd.Series:
    """Parse bool-like CSV values without treating nonempty "False" as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype("string").str.strip().str.lower()
    accepted = normalized.isin({"true", "1", "yes", "y"})
    rejected = normalized.isin({"false", "0", "no", "n", "", "<na>"})
    if not bool((accepted | rejected).all()):
        unknown = sorted(normalized[~(accepted | rejected)].dropna().unique())
        raise ValueError(f"unrecognized boolean values: {unknown}")
    return accepted


def load_sidecar(
    path: Path,
    *,
    score_source: str = "seed",
    minimum_absolute_cosine: float,
    minimum_absolute_amplitude_mrad: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
    if payload.get("schema_version") not in SIDECAR_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported sidecar schema in {path}")
    if score_source not in {"seed", "refined"}:
        raise ValueError(f"unsupported score source: {score_source}")
    if score_source == "refined" and payload.get("schema_version") == (
        "citlali-coherent-iq-mode-sidecar-v1"
    ):
        raise ValueError("refined scores require a v2 sidecar")
    obsnum = int(payload["observation"]["obsnum"])
    score_rows: list[dict[str, Any]] = []
    candidate_metadata: dict[tuple[int, float], dict[str, Any]] = {}
    for event in payload.get("events", []):
        scan = int(event["scan_one_based"])
        event_time = float(event["event_time_unix_sec"])
        key = (scan, event_time)
        metadata = {
            "scan_one_based": scan,
            "event_time_unix_sec": event_time,
            "candidate_kinds": str(event["candidate_kinds"]),
            "seed_network_count": int(event["seed_network_count"]),
            "seed_networks": str(event["seed_networks"]),
            "supporting_detector_events": int(
                event["supporting_detector_events"]
            ),
            "maximum_rtc_score": float(event["maximum_rtc_score"]),
        }
        previous = candidate_metadata.setdefault(key, metadata)
        if previous != metadata:
            raise ValueError(
                f"candidate metadata differs across networks for {key}"
            )
        score = event[
            "mode_score" if score_source == "seed" else "refined_mode_score"
        ]
        shared_refinement = event.get("shared_time_refinement", {})
        amplitude = score.get("projection_amplitude_mrad")
        absolute_cosine = score.get("absolute_cosine_similarity")
        selected = (
            score.get("status") == "scored"
            and amplitude is not None
            and absolute_cosine is not None
            and abs(float(amplitude)) >= minimum_absolute_amplitude_mrad
            and float(absolute_cosine) >= minimum_absolute_cosine
        )
        score_rows.append(
            {
                "obsnum": obsnum,
                "scan_one_based": scan,
                "event_time_unix_sec": event_time,
                "score_source": score_source,
                "scoring_time_unix_sec": (
                    event_time
                    if score_source == "seed"
                    else shared_refinement.get("refined_time_unix_sec")
                ),
                "shared_time_refinement_status": str(
                    shared_refinement.get("status", "not_available")
                ),
                "network": int(event["network"]),
                "status": str(score.get("status", "")),
                "template_id": str(score.get("template_id", "")),
                "template_version": str(score.get("template_version", "")),
                "projection_amplitude_mrad": amplitude,
                "sign": int(score.get("sign", 0)),
                "cosine_similarity": score.get("cosine_similarity"),
                "absolute_cosine_similarity": absolute_cosine,
                "explained_energy_fraction": score.get(
                    "explained_energy_fraction"
                ),
                "residual_energy_mrad2": score.get("residual_energy_mrad2"),
                "total_energy_mrad2": score.get("total_energy_mrad2"),
                "compatible_tone_count": int(
                    score.get("compatible_tone_count", 0)
                ),
                "template_tone_count": int(
                    score.get("template_tone_count", 0)
                ),
                "descriptive_mode_selected": bool(selected),
            }
        )
    scores = pd.DataFrame(score_rows)
    candidates = pd.DataFrame(candidate_metadata.values()).sort_values(
        ["event_time_unix_sec", "scan_one_based"], ignore_index=True
    )
    candidates.insert(
        0,
        "candidate_id",
        [f"s{obsnum}_candidate_{index:05d}" for index in range(len(candidates))],
    )
    candidates.insert(1, "obsnum", obsnum)
    if not scores.empty:
        selected = scores[scores["descriptive_mode_selected"]].copy()
        selected_counts = selected.groupby(
            ["scan_one_based", "event_time_unix_sec"]
        )["network"].agg(list)
        selected_map = {
            key: [int(value) for value in values]
            for key, values in selected_counts.items()
        }
        candidates["selected_networks"] = [
            " ".join(map(str, selected_map.get((row.scan_one_based, row.event_time_unix_sec), [])))
            for row in candidates.itertuples()
        ]
        candidates["selected_network_count"] = [
            len(selected_map.get((row.scan_one_based, row.event_time_unix_sec), []))
            for row in candidates.itertuples()
        ]
        candidates["selected_rack_count"] = [
            len({_rack(network) for network in selected_map.get(
                (row.scan_one_based, row.event_time_unix_sec), []
            )})
            for row in candidates.itertuples()
        ]
    else:
        candidates["selected_networks"] = ""
        candidates["selected_network_count"] = 0
        candidates["selected_rack_count"] = 0
    scores = scores.merge(
        candidates[
            ["candidate_id", "scan_one_based", "event_time_unix_sec"]
        ],
        on=["scan_one_based", "event_time_unix_sec"],
        how="left",
        validate="many_to_one",
    )
    return payload, candidates, scores


def closest_unique_matches(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    reference_time_column: str,
    tolerance_sec: float,
) -> pd.DataFrame:
    """Return deterministic closest-pair one-to-one time matches."""
    pairs: list[tuple[float, int, int]] = []
    candidate_times = candidates["event_time_unix_sec"].to_numpy(dtype=float)
    for reference_index, event_time in enumerate(
        reference[reference_time_column].to_numpy(dtype=float)
    ):
        compatible = np.flatnonzero(
            np.abs(candidate_times - event_time) <= tolerance_sec
        )
        pairs.extend(
            (abs(float(candidate_times[index] - event_time)), reference_index, int(index))
            for index in compatible
        )
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _, reference_index, candidate_index in sorted(pairs):
        if reference_index in used_reference or candidate_index in used_candidate:
            continue
        used_reference.add(reference_index)
        used_candidate.add(candidate_index)
        reference_row = reference.iloc[reference_index]
        candidate_row = candidates.iloc[candidate_index]
        matches.append(
            {
                "reference_row_zero_based": reference_index,
                "candidate_row_zero_based": candidate_index,
                "candidate_id": candidate_row["candidate_id"],
                "reference_time_unix_sec": float(
                    reference_row[reference_time_column]
                ),
                "candidate_time_unix_sec": float(
                    candidate_row["event_time_unix_sec"]
                ),
                "candidate_minus_reference_time_sec": float(
                    candidate_row["event_time_unix_sec"]
                    - reference_row[reference_time_column]
                ),
            }
        )
    return pd.DataFrame(matches)


def maximum_match_count(
    reference_times: np.ndarray,
    candidate_times: np.ndarray,
    tolerance_sec: float,
) -> int:
    """Maximum cardinality for sorted one-dimensional tolerance matches."""
    reference_times = np.sort(np.asarray(reference_times, dtype=float))
    candidate_times = np.sort(np.asarray(candidate_times, dtype=float))
    reference_index = 0
    candidate_index = 0
    count = 0
    while (
        reference_index < reference_times.size
        and candidate_index < candidate_times.size
    ):
        reference_time = reference_times[reference_index]
        candidate_time = candidate_times[candidate_index]
        if candidate_time < reference_time - tolerance_sec:
            candidate_index += 1
        elif candidate_time > reference_time + tolerance_sec:
            reference_index += 1
        else:
            count += 1
            reference_index += 1
            candidate_index += 1
    return count


def circular_shift_null(
    reference_times: np.ndarray,
    candidate_times: np.ndarray,
    *,
    tolerance_sec: float,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    reference_times = np.sort(np.asarray(reference_times, dtype=float))
    candidate_times = np.sort(np.asarray(candidate_times, dtype=float))
    observed = maximum_match_count(
        reference_times, candidate_times, tolerance_sec
    )
    if iterations <= 0 or reference_times.size == 0 or candidate_times.size == 0:
        return {
            "iterations": iterations,
            "observed_match_count": observed,
            "null_median_match_count": None,
            "null_p95_match_count": None,
            "null_maximum_match_count": None,
            "p_value_greater_or_equal": None,
        }
    start = float(min(reference_times.min(), candidate_times.min()))
    end = float(max(reference_times.max(), candidate_times.max()))
    duration = end - start
    if duration <= 2.0:
        raise ValueError("event-time span is too short for circular-shift null")
    generator = np.random.default_rng(random_seed)
    counts = np.empty(iterations, dtype=int)
    for iteration, shift in enumerate(
        generator.uniform(1.0, duration - 1.0, iterations)
    ):
        shifted = start + np.mod(candidate_times - start + shift, duration)
        counts[iteration] = maximum_match_count(
            reference_times, shifted, tolerance_sec
        )
    return {
        "iterations": iterations,
        "observed_match_count": observed,
        "null_median_match_count": float(np.median(counts)),
        "null_p95_match_count": float(np.quantile(counts, 0.95)),
        "null_maximum_match_count": int(counts.max()),
        "p_value_greater_or_equal": float(
            (1 + np.count_nonzero(counts >= observed)) / (iterations + 1)
        ),
    }


def match_summary(reference_count: int, matches: pd.DataFrame) -> MatchSummary:
    if matches.empty:
        return MatchSummary(reference_count, 0, 0.0, None, None, None)
    residual = matches["candidate_minus_reference_time_sec"].to_numpy(
        dtype=float
    )
    return MatchSummary(
        reference_count=reference_count,
        matched_count=len(matches),
        recall=float(len(matches) / reference_count) if reference_count else 0.0,
        median_signed_residual_sec=_finite_or_none(np.median(residual)),
        median_absolute_residual_sec=_finite_or_none(np.median(np.abs(residual))),
        p95_absolute_residual_sec=_finite_or_none(
            np.quantile(np.abs(residual), 0.95)
        ),
    )


def _correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    pair = pd.DataFrame(
        {"left": left.astype(float), "right": right.astype(float)}
    ).dropna()
    if (
        len(pair) < 2
        or pair["left"].nunique() < 2
        or pair["right"].nunique() < 2
    ):
        return None
    value = pair["left"].corr(pair["right"], method=method)
    return _finite_or_none(value)


def compare_scores(
    scores: pd.DataFrame,
    known_events: pd.DataFrame,
    known_matches: pd.DataFrame,
    offline_scores: pd.DataFrame,
    *,
    minimum_absolute_cosine: float,
    minimum_absolute_amplitude_mrad: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if known_matches.empty:
        return pd.DataFrame(), pd.DataFrame(), {"score_pair_count": 0}
    known_identity = known_events.reset_index(drop=True).reset_index().rename(
        columns={"index": "reference_row_zero_based"}
    )
    match_identity = known_matches.merge(
        known_identity[
            ["reference_row_zero_based", "event_cluster_id", "networks"]
        ],
        on="reference_row_zero_based",
        validate="one_to_one",
    )
    runtime = scores.merge(
        match_identity[
            [
                "candidate_id",
                "event_cluster_id",
                "networks",
                "candidate_minus_reference_time_sec",
            ]
        ],
        on="candidate_id",
        validate="many_to_one",
    )
    runtime = runtime[runtime["status"] == "scored"].copy()
    runtime = runtime.rename(
        columns={
            "status": "status_runtime",
            "projection_amplitude_mrad": "projection_amplitude_mrad_runtime",
            "absolute_cosine_similarity": "absolute_cosine_similarity_runtime",
            "explained_energy_fraction": "explained_energy_fraction_runtime",
            "descriptive_mode_selected": "descriptive_mode_selected_runtime",
        }
    )
    offline = offline_scores[
        (offline_scores["status"] == "scored")
        & offline_scores["event_cluster_id"].notna()
    ].copy()
    offline = offline.rename(
        columns={
            "status": "status_offline",
            "projection_amplitude_mrad": "projection_amplitude_mrad_offline",
            "absolute_cosine_similarity": "absolute_cosine_similarity_offline",
            "explained_energy_fraction": "explained_energy_fraction_offline",
        }
    )
    comparison = runtime.merge(
        offline,
        on=["event_cluster_id", "network"],
        suffixes=("_runtime_metadata", "_offline"),
        validate="one_to_one",
    )
    comparison["independently_participating_network"] = [
        int(network) in _network_set(networks)
        for network, networks in zip(
            comparison["network"], comparison["networks"]
        )
    ]
    comparison["offline_descriptive_mode_selected"] = (
        comparison["projection_amplitude_mrad_offline"].abs()
        >= minimum_absolute_amplitude_mrad
    ) & (
        comparison["absolute_cosine_similarity_offline"]
        >= minimum_absolute_cosine
    )
    comparison["selection_agrees"] = (
        comparison["descriptive_mode_selected_runtime"]
        == comparison["offline_descriptive_mode_selected"]
    )
    comparison["sign_agrees"] = (
        np.sign(comparison["projection_amplitude_mrad_runtime"])
        == np.sign(comparison["projection_amplitude_mrad_offline"])
    )
    comparison["absolute_amplitude_difference_mrad"] = (
        comparison["projection_amplitude_mrad_runtime"]
        - comparison["projection_amplitude_mrad_offline"]
    ).abs()
    comparison["absolute_cosine_difference"] = (
        comparison["absolute_cosine_similarity_runtime"]
        - comparison["absolute_cosine_similarity_offline"]
    ).abs()

    network_rows: list[dict[str, Any]] = []
    for network, group in comparison.groupby("network", sort=True):
        runtime_selected = group["descriptive_mode_selected_runtime"]
        offline_selected = group["offline_descriptive_mode_selected"]
        both = runtime_selected & offline_selected
        network_rows.append(
            {
                "network": int(network),
                "score_pair_count": len(group),
                "signed_amplitude_pearson": _correlation(
                    group["projection_amplitude_mrad_runtime"],
                    group["projection_amplitude_mrad_offline"],
                    "pearson",
                ),
                "signed_amplitude_spearman": _correlation(
                    group["projection_amplitude_mrad_runtime"],
                    group["projection_amplitude_mrad_offline"],
                    "spearman",
                ),
                "absolute_cosine_pearson": _correlation(
                    group["absolute_cosine_similarity_runtime"],
                    group["absolute_cosine_similarity_offline"],
                    "pearson",
                ),
                "sign_agreement_fraction": float(group["sign_agrees"].mean()),
                "selection_agreement_fraction": float(
                    group["selection_agrees"].mean()
                ),
                "runtime_selected_count": int(runtime_selected.sum()),
                "offline_selected_count": int(offline_selected.sum()),
                "both_selected_count": int(both.sum()),
            }
        )
    network_summary = pd.DataFrame(network_rows)
    runtime_selected = comparison["descriptive_mode_selected_runtime"]
    offline_selected = comparison["offline_descriptive_mode_selected"]
    both = runtime_selected & offline_selected
    summary = {
        "score_pair_count": len(comparison),
        "event_count": int(comparison["event_cluster_id"].nunique()),
        "network_count": int(comparison["network"].nunique()),
        "signed_amplitude_pearson": _correlation(
            comparison["projection_amplitude_mrad_runtime"],
            comparison["projection_amplitude_mrad_offline"],
            "pearson",
        ),
        "signed_amplitude_spearman": _correlation(
            comparison["projection_amplitude_mrad_runtime"],
            comparison["projection_amplitude_mrad_offline"],
            "spearman",
        ),
        "absolute_cosine_pearson": _correlation(
            comparison["absolute_cosine_similarity_runtime"],
            comparison["absolute_cosine_similarity_offline"],
            "pearson",
        ),
        "absolute_cosine_spearman": _correlation(
            comparison["absolute_cosine_similarity_runtime"],
            comparison["absolute_cosine_similarity_offline"],
            "spearman",
        ),
        "sign_agreement_fraction": float(comparison["sign_agrees"].mean()),
        "selection_agreement_fraction": float(
            comparison["selection_agrees"].mean()
        ),
        "runtime_selected_count": int(runtime_selected.sum()),
        "offline_selected_count": int(offline_selected.sum()),
        "both_selected_count": int(both.sum()),
        "runtime_recall_of_offline_selected": float(
            both.sum() / offline_selected.sum()
        ) if offline_selected.any() else None,
        "runtime_precision_against_offline_selected": float(
            both.sum() / runtime_selected.sum()
        ) if runtime_selected.any() else None,
        "median_absolute_amplitude_difference_mrad": float(
            comparison["absolute_amplitude_difference_mrad"].median()
        ),
        "median_absolute_cosine_difference": float(
            comparison["absolute_cosine_difference"].median()
        ),
        "absolute_timing_vs_absolute_cosine_error_spearman": _correlation(
            comparison["candidate_minus_reference_time_sec"].abs(),
            comparison["absolute_cosine_difference"],
            "spearman",
        ),
    }
    return comparison, network_summary, summary


def render_report(result: dict[str, Any]) -> str:
    known = result["curated_event_matching"]
    primary = result["continuous_primary_event_matching"]
    all_events = result["continuous_all_event_matching"]
    scores = result["network_score_comparison"]
    execution = result["observer_execution"]
    lines = [
        "# Coherent raw-I/Q production sidecar validation",
        "",
        f"- Observation: `{result['obsnum']}`",
        f"- Sidecar: `{result['inputs']['sidecar']['path']}`",
        f"- Candidate tolerance: `{result['parameters']['match_tolerance_sec']}` s",
        f"- Score source: `{result['parameters']['score_source']}`",
        f"- Runtime candidates: `{result['candidate_count']}`",
        f"- Network-event scores: `{result['network_event_score_count']}`",
        f"- Observer status: `{execution.get('status')}`",
        "",
        "## Event-time recovery",
        "",
        f"- Curated RTC/raw-IQ clusters: {known['matched_count']}/{known['reference_count']} "
        f"({known['recall']:.1%}); median signed residual "
        f"{known['median_signed_residual_sec']:.3f} s.",
        f"- Independent primary continuous events: {primary['matched_count']}/{primary['reference_count']} "
        f"({primary['recall']:.1%}); circular-shift p="
        f"{result['null_tests']['continuous_primary']['p_value_greater_or_equal']:.4g}.",
        f"- All continuous-catalog events: {all_events['matched_count']}/{all_events['reference_count']} "
        f"({all_events['recall']:.1%}).",
        "- Unmatched sidecar candidates are not false-positive labels: the sidecar and continuous catalog have different seed networks and pathology scope.",
        "",
        "## Network score transfer",
        "",
        f"- Compared score pairs: {scores['score_pair_count']} across "
        f"{scores['event_count']} matched events and {scores['network_count']} networks.",
        f"- Signed amplitude: Pearson {scores['signed_amplitude_pearson']:.3f}, "
        f"Spearman {scores['signed_amplitude_spearman']:.3f}; sign agreement "
        f"{scores['sign_agreement_fraction']:.1%}.",
        f"- Absolute cosine: Pearson {scores['absolute_cosine_pearson']:.3f}, "
        f"Spearman {scores['absolute_cosine_spearman']:.3f}.",
        f"- Descriptive operating point: runtime {scores['runtime_selected_count']}, "
        f"offline {scores['offline_selected_count']}, both {scores['both_selected_count']}; "
        f"runtime recall of offline-selected responses "
        f"{scores['runtime_recall_of_offline_selected']:.1%}.",
        f"- Absolute timing residual versus cosine-score error has Spearman "
        f"{scores['absolute_timing_vs_absolute_cosine_error_spearman']:.3f}.",
        "",
        "## Interpretation",
        "",
        "The production sidecar is operational and its candidate overlap with the independent catalog is well above circular-shift coincidence. It does not yet reproduce the offline score population. Candidate-time bias must be resolved before thresholds, masking, or subtraction are considered. Shape score, signed/absolute amplitude, displaced-state occupancy, and dwell duration remain separate quantities.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--known-events", type=Path, required=True)
    parser.add_argument("--offline-scores", type=Path, required=True)
    parser.add_argument("--continuous-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-source", choices=("seed", "refined"), default="seed"
    )
    parser.add_argument("--match-tolerance-sec", type=float, default=0.35)
    parser.add_argument("--minimum-absolute-cosine", type=float, default=0.6)
    parser.add_argument(
        "--minimum-absolute-amplitude-mrad", type=float, default=5.0
    )
    parser.add_argument("--null-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.match_tolerance_sec <= 0.0:
        raise ValueError("match tolerance must be positive")
    payload, candidates, scores = load_sidecar(
        args.sidecar,
        score_source=args.score_source,
        minimum_absolute_cosine=args.minimum_absolute_cosine,
        minimum_absolute_amplitude_mrad=args.minimum_absolute_amplitude_mrad,
    )
    obsnum = int(payload["observation"]["obsnum"])
    known_events = pd.read_csv(args.known_events)
    known_events = known_events[known_events["obsnum"] == obsnum].reset_index(
        drop=True
    )
    offline_scores = pd.read_csv(args.offline_scores)
    offline_scores = offline_scores[
        offline_scores["obsnum"] == obsnum
    ].reset_index(drop=True)
    continuous_events = pd.read_csv(args.continuous_events)
    continuous_events = continuous_events[
        continuous_events["obsnum"] == obsnum
    ].reset_index(drop=True)
    continuous_primary = continuous_events[
        _boolean_series(continuous_events["primary_event_candidate"])
    ].reset_index(drop=True)

    known_matches = closest_unique_matches(
        known_events,
        candidates,
        reference_time_column="cluster_time_unix_sec",
        tolerance_sec=args.match_tolerance_sec,
    )
    continuous_all_matches = closest_unique_matches(
        continuous_events,
        candidates,
        reference_time_column="event_time_unix_sec",
        tolerance_sec=args.match_tolerance_sec,
    )
    continuous_primary_matches = closest_unique_matches(
        continuous_primary,
        candidates,
        reference_time_column="event_time_unix_sec",
        tolerance_sec=args.match_tolerance_sec,
    )

    known_output = known_events.reset_index().rename(
        columns={"index": "reference_row_zero_based"}
    ).merge(known_matches, on="reference_row_zero_based", how="left")
    continuous_all_output = continuous_events.reset_index().rename(
        columns={"index": "reference_row_zero_based"}
    ).merge(
        continuous_all_matches, on="reference_row_zero_based", how="left"
    )
    continuous_primary_output = continuous_primary.reset_index().rename(
        columns={"index": "reference_row_zero_based"}
    ).merge(
        continuous_primary_matches, on="reference_row_zero_based", how="left"
    )
    score_comparison, network_summary, score_summary = compare_scores(
        scores,
        known_events,
        known_matches,
        offline_scores,
        minimum_absolute_cosine=args.minimum_absolute_cosine,
        minimum_absolute_amplitude_mrad=args.minimum_absolute_amplitude_mrad,
    )

    candidate_times = candidates["event_time_unix_sec"].to_numpy(dtype=float)
    nulls = {
        "curated": circular_shift_null(
            known_events["cluster_time_unix_sec"].to_numpy(dtype=float),
            candidate_times,
            tolerance_sec=args.match_tolerance_sec,
            iterations=args.null_iterations,
            random_seed=args.random_seed,
        ),
        "continuous_primary": circular_shift_null(
            continuous_primary["event_time_unix_sec"].to_numpy(dtype=float),
            candidate_times,
            tolerance_sec=args.match_tolerance_sec,
            iterations=args.null_iterations,
            random_seed=args.random_seed + 1,
        ),
        "continuous_all": circular_shift_null(
            continuous_events["event_time_unix_sec"].to_numpy(dtype=float),
            candidate_times,
            tolerance_sec=args.match_tolerance_sec,
            iterations=args.null_iterations,
            random_seed=args.random_seed + 2,
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "obsnum": obsnum,
        "inputs": {
            "sidecar": {
                "path": str(args.sidecar.resolve()),
                "sha256": _sha256(args.sidecar),
            },
            "known_events": {
                "path": str(args.known_events.resolve()),
                "sha256": _sha256(args.known_events),
            },
            "offline_scores": {
                "path": str(args.offline_scores.resolve()),
                "sha256": _sha256(args.offline_scores),
            },
            "continuous_events": {
                "path": str(args.continuous_events.resolve()),
                "sha256": _sha256(args.continuous_events),
            },
        },
        "parameters": {
            "match_tolerance_sec": args.match_tolerance_sec,
            "score_source": args.score_source,
            "minimum_absolute_cosine": args.minimum_absolute_cosine,
            "minimum_absolute_amplitude_mrad": args.minimum_absolute_amplitude_mrad,
            "null_iterations": args.null_iterations,
            "random_seed": args.random_seed,
        },
        "observer_execution": payload.get("observer_execution", {}),
        "candidate_count": len(candidates),
        "network_event_score_count": len(scores),
        "curated_event_matching": asdict(
            match_summary(len(known_events), known_matches)
        ),
        "continuous_primary_event_matching": asdict(
            match_summary(len(continuous_primary), continuous_primary_matches)
        ),
        "continuous_all_event_matching": asdict(
            match_summary(len(continuous_events), continuous_all_matches)
        ),
        "null_tests": nulls,
        "network_score_comparison": score_summary,
        "limitations": [
            "Unmatched sidecar candidates are not false-positive labels because the two catalogs have different seed networks and pathology scope.",
            "Offline scores use event-vector catalog times and alternating-half templates; runtime scores use RTC candidate times and full-corpus templates.",
            "The descriptive 0.6 cosine and 5 mrad amplitude point is not production policy.",
            "This analysis does not estimate displaced-state occupancy, settling boundaries, dwell duration, or repairable stable chunks.",
        ],
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "sidecar_candidates.csv", index=False)
    known_output.to_csv(output_dir / "curated_event_matches.csv", index=False)
    continuous_primary_output.to_csv(
        output_dir / "continuous_primary_event_matches.csv", index=False
    )
    continuous_all_output.to_csv(
        output_dir / "continuous_all_event_matches.csv", index=False
    )
    score_comparison.to_csv(
        output_dir / "network_score_comparison.csv", index=False
    )
    network_summary.to_csv(
        output_dir / "network_score_summary.csv", index=False
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(render_report(result), end="")


if __name__ == "__main__":
    main()
