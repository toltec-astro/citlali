#!/usr/bin/env python3
"""Compare a restarted pointing trajectory with an uninterrupted reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    file_record,
    image,
    iteration_dirs,
    product_path,
    rms,
)


EXTENSIONS = ("signal_I", "kernel_I", "weight_I")
PENALTY_FIELDS = (
    "penalty_producer",
    "penalty_reason",
    "penalty_iteration",
    "penalty_scan",
    "penalty_uid",
    "penalty_network",
    "penalty_array",
    "penalty_factor",
    "penalty_score",
    "penalty_scan_local",
)


def array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def comparison_rows(
    reference_root: Path, replay_root: Path, obsnum: int,
) -> list[dict[str, int | str | bool | float]]:
    reference = iteration_dirs(reference_root, obsnum)
    replay = iteration_dirs(replay_root, obsnum)
    missing = sorted(set(replay) - set(reference))
    if missing:
        raise ValueError(
            f"replay iterations are absent from reference: {missing}"
        )

    rows: list[dict[str, int | str | bool | float]] = []
    for iteration in sorted(replay):
        for array in ARRAYS:
            for extension in EXTENSIONS:
                expected = image(
                    product_path(reference[iteration], obsnum, array),
                    extension,
                )
                actual = image(
                    product_path(replay[iteration], obsnum, array), extension,
                )
                if expected.shape != actual.shape:
                    raise ValueError(
                        "reference/replay image shapes differ: "
                        f"iteration={iteration} array={array} "
                        f"extension={extension} reference={expected.shape} "
                        f"replay={actual.shape}"
                    )
                difference = actual - expected
                expected_rms = rms(expected)
                difference_rms = rms(difference)
                finite = np.isfinite(difference)
                max_abs_difference = (
                    float(np.max(np.abs(difference[finite])))
                    if finite.any()
                    else math.nan
                )
                rows.append(
                    {
                        "iteration": iteration,
                        "array": array,
                        "extension": extension,
                        "exact": array_equal(expected, actual),
                        "difference_rms": difference_rms,
                        "reference_rms": expected_rms,
                        "relative_rms": (
                            difference_rms / expected_rms
                            if expected_rms > 0.0
                            else math.nan
                        ),
                        "max_abs_difference": max_abs_difference,
                    }
                )
    return rows


def _value(value: object) -> object:
    if isinstance(value, np.ma.MaskedArray):
        if value.size != 1 or bool(np.ma.is_masked(value)):
            raise ValueError("checkpoint penalty value is unavailable")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def penalty_records(path: Path) -> list[dict[str, object]]:
    with Dataset(path) as dataset:
        count = int(dataset["effective_detector_penalty_count"][0])
        observations = [str(value) for value in dataset["observation_id"][:]]
        observation_index = dataset["penalty_observation_index"][:]
        records = []
        for index in range(count):
            record: dict[str, object] = {
                "observation_id": observations[int(observation_index[index])]
            }
            for field in PENALTY_FIELDS:
                record[field.removeprefix("penalty_")] = _value(
                    dataset[field][index]
                )
            records.append(record)
        return records


def checkpoint_comparison(
    reference_path: Path, replay_path: Path,
) -> dict[str, object]:
    with Dataset(reference_path) as reference, Dataset(replay_path) as replay:
        reference_names = set(reference.variables)
        replay_names = set(replay.variables)
        common = sorted(reference_names & replay_names)
        differing = [
            name
            for name in common
            if not array_equal(
                np.asarray(reference[name][:]), np.asarray(replay[name][:])
            )
        ]
    return {
        "reference": file_record(reference_path.resolve()),
        "replay": file_record(replay_path.resolve()),
        "reference_only_variables": sorted(reference_names - replay_names),
        "replay_only_variables": sorted(replay_names - reference_names),
        "differing_variables": differing,
        "reference_penalties": penalty_records(reference_path),
        "replay_penalties": penalty_records(replay_path),
    }


def checkpoint_trajectory_comparisons(
    reference: dict[int, Path], replay: dict[int, Path],
) -> list[dict[str, object]]:
    results = []
    for iteration in sorted(replay):
        if iteration not in reference:
            raise ValueError(
                f"replay checkpoint iteration {iteration} is absent from reference"
            )
        comparison = checkpoint_comparison(
            reference[iteration] / "citlali_restart_checkpoint.nc",
            replay[iteration] / "citlali_restart_checkpoint.nc",
        )
        comparison["iteration"] = iteration
        comparison["exact"] = not (
            comparison["reference_only_variables"]
            or comparison["replay_only_variables"]
            or comparison["differing_variables"]
        )
        results.append(comparison)
    return results


def write_results(
    rows: list[dict[str, int | str | bool | float]],
    output: Path,
    *,
    reference_root: Path,
    replay_root: Path,
    obsnum: int,
    checkpoint_iteration: int,
    manifest_output: Path,
    test_id: str,
    evidence_paths: list[Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    reference = iteration_dirs(reference_root, obsnum)
    replay = iteration_dirs(replay_root, obsnum)
    if (
        checkpoint_iteration not in reference
        or checkpoint_iteration not in replay
    ):
        raise ValueError(
            f"checkpoint iteration {checkpoint_iteration} is unavailable"
        )
    checkpoint_comparisons = checkpoint_trajectory_comparisons(
        reference, replay
    )
    selected_checkpoint = next(
        comparison
        for comparison in checkpoint_comparisons
        if comparison["iteration"] == checkpoint_iteration
    )
    inputs = {Path(__file__).resolve()}
    inputs.update(path.resolve() for path in evidence_paths)
    for iteration in sorted(replay):
        for array in ARRAYS:
            inputs.add(
                product_path(reference[iteration], obsnum, array).resolve()
            )
            inputs.add(product_path(replay[iteration], obsnum, array).resolve())
        inputs.add(
            (reference[iteration] / "citlali_restart_checkpoint.nc").resolve()
        )
        inputs.add(
            (replay[iteration] / "citlali_restart_checkpoint.nc").resolve()
        )

    first_difference = next(
        (row for row in rows if not bool(row["exact"])), None
    )
    first_checkpoint_difference = next(
        (
            comparison
            for comparison in checkpoint_comparisons
            if not bool(comparison["exact"])
        ),
        None,
    )
    payload = {
        "schema_version": "sci-fruit-restart-replay-development-v2",
        "test_id": test_id,
        "role": "exploratory-development-only",
        "qualification_use_authorized": False,
        "obsnum": obsnum,
        "result": (
            "FAIL"
            if first_difference or first_checkpoint_difference
            else "PASS"
        ),
        "first_product_difference": first_difference,
        "first_checkpoint_difference": first_checkpoint_difference,
        "checkpoint_iteration_compared": checkpoint_iteration,
        # Retained for readers of the v1 manifest. The v2 trajectory field is
        # authoritative for a multi-iteration exact-restart claim.
        "checkpoint_comparison": selected_checkpoint,
        "checkpoint_trajectory_comparisons": checkpoint_comparisons,
        "inputs": [
            file_record(path) for path in sorted(inputs, key=str)
        ],
        "outputs": [
            file_record(
                output.resolve(),
                relative_to=manifest_output.parent.resolve(),
            )
        ],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--obsnum", required=True, type=int)
    parser.add_argument("--checkpoint-iteration", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--test-id", required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        type=Path,
        help="additional configuration, log, or timing input to hash",
    )
    args = parser.parse_args()

    rows = comparison_rows(args.reference, args.replay, args.obsnum)
    write_results(
        rows,
        args.output,
        reference_root=args.reference,
        replay_root=args.replay,
        obsnum=args.obsnum,
        checkpoint_iteration=args.checkpoint_iteration,
        manifest_output=args.manifest_output,
        test_id=args.test_id,
        evidence_paths=args.evidence,
    )
    exact = sum(bool(row["exact"]) for row in rows)
    print(f"wrote {len(rows)} rows to {args.output}: exact={exact}")
    print(f"wrote replay manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
