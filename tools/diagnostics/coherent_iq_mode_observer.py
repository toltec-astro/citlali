#!/usr/bin/env python3
"""Observe-only classifier for coherent raw-I/Q network events.

The classifier consumes a pre/post phase-change vector at an already proposed
event time.  It never edits timestreams, flags, weights, or learning state.
Templates are joined by detector UID and checked against the signed digital
tone offset before any score is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "citlali-coherent-iq-mode-template-v1"
DIAGNOSTIC_SCHEMA_VERSION = "citlali-coherent-iq-mode-diagnostic-v1"


class TemplateCompatibilityError(ValueError):
    """Raised when an event cannot be safely compared with a template."""


@dataclass(frozen=True)
class EventScore:
    status: str
    template_id: str
    template_version: str
    network: int
    primary_mode_id: str | None
    projection_amplitude_mrad: float | None
    sign: int | None
    cosine_similarity: float | None
    absolute_cosine_similarity: float | None
    explained_energy_fraction: float | None
    residual_energy_mrad2: float | None
    total_energy_mrad2: float | None
    multi_mode_explained_energy_fraction: float | None
    common_phase_explained_energy_fraction: float | None
    delay_slope_explained_energy_fraction: float | None
    compatible_tone_count: int
    template_tone_count: int
    compatible_tone_fraction: float
    rejected_tone_count: int
    compatibility_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            **self.__dict__,
            "compatibility_notes": list(self.compatibility_notes),
        }


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_template(template: dict[str, Any]) -> None:
    if template.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported template schema {template.get('schema_version')!r}"
        )
    for name in (
        "template_id",
        "template_version",
        "lifecycle_state",
        "identity",
        "tone_coordinate",
        "normalization",
        "modes",
        "training",
        "compatibility",
        "validation",
        "provenance",
    ):
        if name not in template:
            raise ValueError(f"template is missing {name!r}")
    if template["lifecycle_state"] not in {"observe_only", "experimental"}:
        raise ValueError("only observe-only or experimental templates are accepted")
    identity = template["identity"]
    if not isinstance(identity.get("network"), int):
        raise ValueError("identity.network must be an integer network ID")
    tones = template["tone_coordinate"].get("tones")
    if not isinstance(tones, list) or not tones:
        raise ValueError("template must contain at least one tone")
    uids = [int(row["uid"]) for row in tones]
    if len(uids) != len(set(uids)):
        raise ValueError("template tone UIDs must be unique")
    mode_ids = [str(mode["mode_id"]) for mode in template["modes"]]
    if not mode_ids or len(mode_ids) != len(set(mode_ids)):
        raise ValueError("template mode IDs must be non-empty and unique")
    for tone in tones:
        loadings = tone.get("loadings", {})
        if set(loadings) != set(mode_ids):
            raise ValueError(
                f"uid {tone['uid']} does not carry every declared mode loading"
            )
        for mode_id in mode_ids:
            _finite_float(loadings[mode_id], f"loading {mode_id}")


def load_template(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        template = json.load(handle)
    _validate_template(template)
    return template


def _deterministic_mode_sign(mode: np.ndarray, uids: np.ndarray) -> tuple[np.ndarray, int]:
    anchor = int(np.argmax(np.abs(mode)))
    signed = np.asarray(mode, dtype=float).copy()
    if signed[anchor] < 0.0:
        signed *= -1.0
    return signed, int(uids[anchor])


def fit_rank_modes(
    event_vectors_mrad: np.ndarray,
    *,
    rank: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic RMS-normalized, uncentered phase modes."""
    matrix = np.asarray(event_vectors_mrad, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("event matrix must have shape [event, tone] with both axes >1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("event matrix must be finite")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("invalid requested mode rank")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    modes = vh[:rank, :].copy()
    for row in range(modes.shape[0]):
        rms = float(np.sqrt(np.mean(modes[row] ** 2)))
        if not math.isfinite(rms) or rms <= 0.0:
            raise ValueError("SVD returned a zero or non-finite mode")
        modes[row] /= rms
    energy = singular**2
    fractions = energy[:rank] / np.sum(energy)
    return modes, fractions


def make_template(
    *,
    template_id: str,
    template_version: str,
    network: int,
    uids: Sequence[int],
    tone_slots: Sequence[int],
    tone_offsets_hz: Sequence[float],
    probe_frequencies_hz: Sequence[float],
    modes: np.ndarray,
    training: dict[str, Any],
    validation: dict[str, Any],
    provenance: dict[str, Any],
    tone_offset_tolerance_hz: float = 500_000.0,
    minimum_compatible_tone_fraction: float = 0.8,
    required_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uids_array = np.asarray(uids, dtype=int)
    slots = np.asarray(tone_slots, dtype=int)
    offsets = np.asarray(tone_offsets_hz, dtype=float)
    probes = np.asarray(probe_frequencies_hz, dtype=float)
    mode_matrix = np.asarray(modes, dtype=float)
    n_tones = uids_array.size
    if (
        slots.shape != (n_tones,)
        or offsets.shape != (n_tones,)
        or probes.shape != (n_tones,)
        or mode_matrix.ndim != 2
        or mode_matrix.shape[1] != n_tones
    ):
        raise ValueError("template tone coordinates and modes have inconsistent shapes")
    if len(set(map(int, uids_array))) != n_tones:
        raise ValueError("template UIDs must be unique")
    order = np.argsort(uids_array)
    uids_array = uids_array[order]
    slots = slots[order]
    offsets = offsets[order]
    probes = probes[order]
    mode_matrix = mode_matrix[:, order]

    mode_rows: list[dict[str, Any]] = []
    anchor_uids: list[int] = []
    for index, mode in enumerate(mode_matrix):
        rms = float(np.sqrt(np.mean(mode**2)))
        if not math.isfinite(rms) or rms <= 0.0:
            raise ValueError("mode has zero or non-finite norm")
        normalized, anchor_uid = _deterministic_mode_sign(
            mode / rms, uids_array
        )
        mode_matrix[index] = normalized
        anchor_uids.append(anchor_uid)
        mode_rows.append(
            {
                "mode_id": f"phase_mode_{index + 1}",
                "rank": index + 1,
                "anchor_uid": anchor_uid,
            }
        )

    tones = []
    for tone_index in range(n_tones):
        tones.append(
            {
                "uid": int(uids_array[tone_index]),
                "tone_slot_zero_based": int(slots[tone_index]),
                "tone_offset_frequency_hz": float(offsets[tone_index]),
                "probe_frequency_hz": float(probes[tone_index]),
                "loadings": {
                    mode["mode_id"]: float(mode_matrix[mode_index, tone_index])
                    for mode_index, mode in enumerate(mode_rows)
                },
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "template_id": str(template_id),
        "template_version": str(template_version),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "lifecycle_state": "observe_only",
        "identity": {
            "network": int(network),
            "readout_id": f"toltec-nw{int(network)}",
            "readout_coordinate_system": "apt_uid+signed_digital_tone_offset_hz",
        },
        "tone_coordinate": {
            "identity_field": "uid",
            "ordering": "uid_ascending",
            "frequency_field": "tone_offset_frequency_hz",
            "frequency_meaning": "signed digital tone offset from network LO",
            "tones": tones,
        },
        "normalization": {
            "kind": "rms_unity_over_template_tones",
            "sign_rule": "largest_absolute_loading_is_positive",
            "anchor_uids": anchor_uids,
            "projection_amplitude_unit": "mrad RMS phase change",
        },
        "modes": mode_rows,
        "training": training,
        "compatibility": {
            "tone_offset_tolerance_hz": float(tone_offset_tolerance_hz),
            "minimum_compatible_tone_fraction": float(
                minimum_compatible_tone_fraction
            ),
            "required_metadata": dict(required_metadata or {}),
            "partial_match_policy": "explicit_coverage_or_fail_closed",
            "unresolved_metadata": [
                "firmware_state",
                "readout_software_version",
                "IF_configuration",
            ],
        },
        "validation": validation,
        "provenance": provenance,
    }
    _validate_template(result)
    return result


def _zero_baseline_r2(values: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return math.nan
    residual = values - prediction
    return 1.0 - float(np.dot(residual, residual)) / denominator


def _empty_score(
    template: dict[str, Any],
    status: str,
    *,
    compatible: int = 0,
    rejected: int = 0,
    notes: Iterable[str] = (),
) -> EventScore:
    total = len(template["tone_coordinate"]["tones"])
    return EventScore(
        status=status,
        template_id=str(template["template_id"]),
        template_version=str(template["template_version"]),
        network=int(template["identity"]["network"]),
        primary_mode_id=None,
        projection_amplitude_mrad=None,
        sign=None,
        cosine_similarity=None,
        absolute_cosine_similarity=None,
        explained_energy_fraction=None,
        residual_energy_mrad2=None,
        total_energy_mrad2=None,
        multi_mode_explained_energy_fraction=None,
        common_phase_explained_energy_fraction=None,
        delay_slope_explained_energy_fraction=None,
        compatible_tone_count=int(compatible),
        template_tone_count=int(total),
        compatible_tone_fraction=float(compatible / total if total else 0.0),
        rejected_tone_count=int(rejected),
        compatibility_notes=tuple(notes),
    )


def score_event(
    template: dict[str, Any],
    *,
    network: int,
    uids: Sequence[int],
    tone_offsets_hz: Sequence[float],
    phase_change_mrad: Sequence[float],
    metadata: dict[str, Any] | None = None,
) -> EventScore:
    """Score one candidate without mutating any caller-owned array."""
    _validate_template(template)
    if int(network) != int(template["identity"]["network"]):
        return _empty_score(
            template,
            "incompatible_network",
            notes=(f"candidate nw{network} != template nw{template['identity']['network']}",),
        )
    metadata = dict(metadata or {})
    required = template["compatibility"].get("required_metadata", {})
    mismatches = [
        key for key, value in required.items()
        if key not in metadata or metadata[key] != value
    ]
    if mismatches:
        return _empty_score(
            template,
            "incompatible_metadata",
            notes=tuple(f"required metadata mismatch: {key}" for key in mismatches),
        )

    uid_array = np.asarray(uids, dtype=int)
    offset_array = np.asarray(tone_offsets_hz, dtype=float)
    phase_array = np.asarray(phase_change_mrad, dtype=float)
    if (
        uid_array.ndim != 1
        or offset_array.shape != uid_array.shape
        or phase_array.shape != uid_array.shape
    ):
        raise ValueError("candidate UID, tone-offset, and phase arrays must align")
    if len(set(map(int, uid_array))) != uid_array.size:
        return _empty_score(
            template, "incompatible_tone_map", notes=("candidate UIDs are not unique",)
        )

    candidate_by_uid = {int(uid): index for index, uid in enumerate(uid_array)}
    tolerance = float(
        template["compatibility"]["tone_offset_tolerance_hz"]
    )
    mode_ids = [str(mode["mode_id"]) for mode in template["modes"]]
    observed: list[float] = []
    offsets: list[float] = []
    loading_rows: list[list[float]] = []
    rejected = 0
    for tone in template["tone_coordinate"]["tones"]:
        index = candidate_by_uid.get(int(tone["uid"]))
        if index is None:
            rejected += 1
            continue
        observed_offset = float(offset_array[index])
        observed_phase = float(phase_array[index])
        if (
            not math.isfinite(observed_offset)
            or not math.isfinite(observed_phase)
            or abs(observed_offset - float(tone["tone_offset_frequency_hz"]))
            > tolerance
        ):
            rejected += 1
            continue
        observed.append(observed_phase)
        offsets.append(observed_offset)
        loading_rows.append(
            [float(tone["loadings"][mode_id]) for mode_id in mode_ids]
        )

    compatible = len(observed)
    total = len(template["tone_coordinate"]["tones"])
    fraction = compatible / total if total else 0.0
    minimum_fraction = float(
        template["compatibility"]["minimum_compatible_tone_fraction"]
    )
    if compatible < max(3, len(mode_ids) + 1) or fraction < minimum_fraction:
        return _empty_score(
            template,
            "insufficient_compatible_tones",
            compatible=compatible,
            rejected=rejected,
            notes=(
                f"coverage {fraction:.4f} below required {minimum_fraction:.4f}",
            ),
        )

    y = np.asarray(observed, dtype=float)
    tone_offset = np.asarray(offsets, dtype=float)
    mode_matrix = np.asarray(loading_rows, dtype=float)
    total_energy = float(np.dot(y, y))
    if total_energy <= 0.0:
        return _empty_score(
            template,
            "zero_event_energy",
            compatible=compatible,
            rejected=rejected,
        )

    individual: list[tuple[float, float, float, np.ndarray]] = []
    for mode_index in range(mode_matrix.shape[1]):
        mode = mode_matrix[:, mode_index]
        denominator = float(np.dot(mode, mode))
        amplitude = float(np.dot(y, mode) / denominator)
        prediction = amplitude * mode
        cosine = float(
            np.dot(y, mode)
            / math.sqrt(total_energy * denominator)
        )
        individual.append(
            (cosine * cosine, amplitude, cosine, prediction)
        )
    primary_index = int(np.argmax([row[0] for row in individual]))
    explained, amplitude, cosine, prediction = individual[primary_index]
    residual_energy = float(np.dot(y - prediction, y - prediction))

    coefficients, _, _, _ = np.linalg.lstsq(mode_matrix, y, rcond=None)
    combined_prediction = mode_matrix @ coefficients
    multi_r2 = _zero_baseline_r2(y, combined_prediction)

    common = np.full(y.shape, np.mean(y))
    common_r2 = _zero_baseline_r2(y, common)
    offset_scale = float(np.std(tone_offset))
    if offset_scale > 0.0:
        x = (tone_offset - float(np.mean(tone_offset))) / offset_scale
        design = np.column_stack([np.ones(y.size), x])
        delay_coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        delay_prediction = design @ delay_coefficients
        delay_r2 = _zero_baseline_r2(y, delay_prediction)
    else:
        delay_r2 = math.nan

    return EventScore(
        status="scored",
        template_id=str(template["template_id"]),
        template_version=str(template["template_version"]),
        network=int(network),
        primary_mode_id=mode_ids[primary_index],
        projection_amplitude_mrad=amplitude,
        sign=1 if amplitude > 0.0 else (-1 if amplitude < 0.0 else 0),
        cosine_similarity=cosine,
        absolute_cosine_similarity=abs(cosine),
        explained_energy_fraction=explained,
        residual_energy_mrad2=residual_energy,
        total_energy_mrad2=total_energy,
        multi_mode_explained_energy_fraction=multi_r2,
        common_phase_explained_energy_fraction=common_r2,
        delay_slope_explained_energy_fraction=delay_r2,
        compatible_tone_count=compatible,
        template_tone_count=total,
        compatible_tone_fraction=fraction,
        rejected_tone_count=rejected,
        compatibility_notes=(),
    )


def attach_cross_network_coincidence(
    rows: list[dict[str, Any]],
    *,
    time_field: str = "event_time_unix_sec",
    tolerance_sec: float = 0.35,
    selection_field: str | None = None,
) -> None:
    """Annotate compact records with coincident distinct-network counts."""
    ordered = sorted(
        range(len(rows)),
        key=lambda index: float(rows[index][time_field]),
    )
    for index in ordered:
        epoch = float(rows[index][time_field])
        matches = [
            other
            for other in ordered
            if abs(float(rows[other][time_field]) - epoch) <= tolerance_sec
            and rows[other].get("status") == "scored"
            and (
                selection_field is None
                or bool(rows[other].get(selection_field, False))
            )
        ]
        networks = sorted({int(rows[other]["network"]) for other in matches})
        rows[index]["cross_network_coincident_count"] = len(networks)
        rows[index]["cross_network_coincident_networks"] = " ".join(
            map(str, networks)
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_template_from_rank_csv(args: argparse.Namespace) -> None:
    rows = []
    with args.rank_mode_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["network"]) == args.network:
                rows.append(row)
    if not rows:
        raise ValueError(f"rank-mode CSV has no rows for nw{args.network}")
    rows.sort(key=lambda row: int(row["uid"]))
    mode = np.asarray(
        [float(row["phase_rank1_loading_rms_normalized"]) for row in rows]
    )[None, :]
    template = make_template(
        template_id=args.template_id or f"ngc4449-20260219-nw{args.network}-phase",
        template_version=args.template_version,
        network=args.network,
        uids=[int(row["uid"]) for row in rows],
        tone_slots=[int(row["tone_slot_zero_based"]) for row in rows],
        tone_offsets_hz=[
            float(row["tone_offset_frequency_hz"]) for row in rows
        ],
        probe_frequencies_hz=[
            float(row["probe_frequency_hz"]) for row in rows
        ],
        modes=mode,
        training={
            "dataset": "NGC4449 late-night raw-I/Q event corpus",
            "event_count": 52,
            "method": "uncentered SVD on UIDs present in every event",
            "status": "same-night forensic training; not generalized",
        },
        validation={
            "status": "observe_only",
            "coverage_fraction": 1.0,
            "stability": "see source rank-one summary and split-half cosine",
            "uncertainty": "not yet quantified across nights or retunes",
        },
        provenance={
            "source_csv": str(args.rank_mode_csv.resolve()),
            "investigate_commit": "422f25f5f",
            "handoff": "handoff/SCIENCE_IQ_TONE_SUSCEPTIBILITY_ANALYSIS_2026-07-30.md",
        },
        tone_offset_tolerance_hz=args.tone_offset_tolerance_hz,
        minimum_compatible_tone_fraction=args.minimum_coverage,
    )
    _write_json(args.output, template)
    print(f"Wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-template", help="convert an Investigate rank-mode CSV to a template"
    )
    build.add_argument("--rank-mode-csv", type=Path, required=True)
    build.add_argument("--network", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--template-id")
    build.add_argument("--template-version", default="2026-07-30.1")
    build.add_argument("--tone-offset-tolerance-hz", type=float, default=500_000.0)
    build.add_argument("--minimum-coverage", type=float, default=0.8)
    args = parser.parse_args()
    if args.command == "build-template":
        _build_template_from_rank_csv(args)


if __name__ == "__main__":
    main()
