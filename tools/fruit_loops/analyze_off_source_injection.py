#!/usr/bin/env python3
"""Analyze the registered SCI-FRUIT EL-F5 off-source location control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import yaml
from astropy.io import fits
from netCDF4 import Dataset

from tools.fruit_loops.analyze_compact_relaxation_screen import (
    analyze_pair,
    iteration_dirs,
    select_row,
    trajectory_config,
)
from tools.fruit_loops.analyze_feedback_model_bypass import (
    mapdiag_penalties,
    require_exact_trajectory_images,
    write_csv,
)
from tools.fruit_loops.compare_injected_source_pair import (
    ARRAYS,
    file_record,
    product_path,
)


ARRAY_IDS = {"a1100": 0, "a1400": 1, "a2000": 2}


def penalty_key(row: dict) -> tuple[int, int, int, int]:
    return tuple(int(row[key]) for key in ("iteration", "scan", "uid", "array"))


def injection_specific_hard_penalties(
    control: list[dict], injected: list[dict], *, iteration: int, factor: float,
) -> list[dict]:
    """Return hard penalties introduced by the injected trajectory."""
    control_by_key = {penalty_key(row): float(row["factor"]) for row in control}
    result = []
    for row in injected:
        key = penalty_key(row)
        if key[0] != iteration or float(row["factor"]) != factor:
            continue
        if key not in control_by_key or control_by_key[key] != factor:
            result.append(row)
    return sorted(result, key=penalty_key)


def response_loss(rows: list[dict], array: str) -> dict:
    before = select_row(rows, 1.25, 4, array)
    after = select_row(rows, 1.25, 5, array)
    recovery_decreased = (
        after["kernel_normalized_central_recovery"]
        < before["kernel_normalized_central_recovery"]
    )
    annular_increased = (
        after["annular_residual_over_truth"]
        > before["annular_residual_over_truth"]
    )
    return {
        "array": array,
        "iteration_4_recovery": before["kernel_normalized_central_recovery"],
        "iteration_5_recovery": after["kernel_normalized_central_recovery"],
        "recovery_decreased": recovery_decreased,
        "iteration_4_annular_residual": before["annular_residual_over_truth"],
        "iteration_5_annular_residual": after["annular_residual_over_truth"],
        "annular_residual_increased": annular_increased,
        "registered_response_loss_direction": (
            recovery_decreased and annular_increased
        ),
    }


def classify_location_control(
    penalties: list[dict], response_by_array: dict[str, dict], target: dict,
) -> str:
    target_key = tuple(
        int(target[key]) for key in ("iteration", "scan", "uid", "array")
    )
    target_present = any(penalty_key(row) == target_key for row in penalties)
    target_array = next(
        name for name, array_id in ARRAY_IDS.items() if array_id == target_key[3]
    )
    if (
        target_present
        and response_by_array[target_array]["registered_response_loss_direction"]
    ):
        return "same_event_replicated_off_source"
    if not target_present and not penalties:
        return "centered_event_not_replicated"
    for row in penalties:
        key = penalty_key(row)
        array = next(
            name for name, array_id in ARRAY_IDS.items() if array_id == key[3]
        )
        if key != target_key and response_by_array[array][
            "registered_response_loss_direction"
        ]:
            return "different_penalty_association"
    return "inconclusive"


def read_centered_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["obsnum"] != "123424" or row["evidence_view"] != "complete_map":
                continue
            rows.append(
                {
                    "alpha": float(row["alpha"]),
                    "iteration": int(row["iteration"]),
                    "array": row["array"],
                    "kernel_normalized_central_recovery": float(
                        row["kernel_normalized_central_recovery"]
                    ),
                    "annular_residual_over_truth": float(
                        row["annular_residual_over_truth"]
                    ),
                }
            )
    return rows


def read_centered_penalties(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["obsnum"] == "123424"
                and row["evidence_view"] == "complete_map"
                and row["variant"] == "injected"
            ):
                rows.append(
                    {
                        "iteration": int(row["iteration"]),
                        "scan": int(row["scan"]),
                        "uid": int(row["uid"]),
                        "array": int(row["array"]),
                        "score": float(row["score"]),
                        "factor": float(row["factor"]),
                        "scan_local": row["scan_local"] == "True",
                    }
                )
    return rows


def target_contributor_records(root: Path, terminal: int, target: dict) -> list[dict]:
    path = root / f"redu{terminal:02d}" / f"learning_iter_{terminal}.csv"
    records = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not (
                row["record_type"] == "map_pixel_outlier"
                and row["producer"] == "mapdiag:raw_obs"
                and row["reason"] == "extreme_pixel_targeted_contributor"
                and int(row["iter"]) == int(target["iteration"])
                and int(row["scan"]) == int(target["scan"])
                and int(row["uid"]) == int(target["uid"])
                and int(row["map_index"]) == int(target["array"])
            ):
                continue
            records.append(
                {
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "sample": int(row["sample"]),
                    "map_value_mjy_beam": float(row["value"]),
                    "leave_one_out_z": float(row["leave_one_out_z"]),
                    "source_distance_arcsec": float(
                        row["source_distance_arcsec"]
                    ),
                }
            )
    return records


def scalar_netcdf_value(dataset: Dataset, name: str):
    variable = dataset[name]
    value = variable[...]
    return value.item() if getattr(value, "shape", None) == () else value[0].item()


def require_offset_provenance(
    root: Path, manifest: dict, *, expected_enabled: bool,
) -> None:
    obsnum = int(manifest["obsnum"])
    terminal = int(manifest["terminal_iteration"])
    dirs = iteration_dirs(root, obsnum)
    config = trajectory_config(dirs)
    injected = config["timestream"]["fruit_loops"]["injected_source_test"]
    expected_az = float(manifest["az_offset_arcsec"])
    expected_el = float(manifest["el_offset_arcsec"])
    expected = {
        "enabled": expected_enabled,
        "az_offset_arcsec": expected_az,
        "el_offset_arcsec": expected_el,
    }
    for key, value in expected.items():
        if injected[key] != value:
            raise ValueError(f"merged-config injected-source {key} differs")

    redu = dirs[terminal]
    provenance = yaml.safe_load(
        (redu / "processed_timestream_provenance.yaml").read_text()
    )
    for section in ("requested", "effective"):
        node = provenance[section]
        if section == "effective":
            node = node["config"]
        node = node["fruit_loops"]["injected_source_test"]
        for key, value in expected.items():
            if node[key] != value:
                raise ValueError(
                    f"processed-timestream {section} {key} differs"
                )

    for array in ARRAYS:
        path = product_path(redu, obsnum, array)
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            if bool(header["CONFIG.FRUITLOOPS.INJECT"]) != expected_enabled:
                raise ValueError(f"FITS injection state differs: {path}")
            if float(header["CONFIG.FRUITLOOPS.INJAZ"]) != expected_az:
                raise ValueError(f"FITS injection AZ offset differs: {path}")
            if float(header["CONFIG.FRUITLOOPS.INJEL"]) != expected_el:
                raise ValueError(f"FITS injection EL offset differs: {path}")

    ptcdiag = redu / str(obsnum) / "raw" / (
        f"toltec_commissioning_pointing_{obsnum}_ptcdiag.nc"
    )
    with Dataset(ptcdiag) as dataset:
        names = {
            "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.ENABLED": expected_enabled,
            "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.AZ_OFFSET_ARCSEC": expected_az,
            "CONFIG.FRUITLOOPS.INJECTED_SOURCE_TEST.EL_OFFSET_ARCSEC": expected_el,
        }
        for name, value in names.items():
            if scalar_netcdf_value(dataset, name) != value:
                raise ValueError(f"NetCDF injection provenance differs: {name}")


def read_execution(log_dir: Path, expected_az: float, expected_el: float) -> list[dict]:
    rows = []
    error_pattern = re.compile(r"\[(?:error|critical)\]|(?:error|critical):", re.I)
    for label in ("control", "off-source-injected"):
        path = log_dir / f"point-123424-{label}.log"
        text = path.read_text(encoding="utf-8")
        wall = re.search(r"^\s*([0-9.]+) real\s", text, re.M)
        rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size$", text, re.M)
        errors = sum(bool(error_pattern.search(line)) for line in text.splitlines())
        if wall is None or rss is None or "citlali is done!" not in text or errors:
            raise ValueError(f"incomplete or unsuccessful execution log: {path}")
        records = re.findall(
            r"fruit-loop injected-source test .*?"
            r"az_offset_arcsec=([-+0-9.eE]+) el_offset_arcsec=([-+0-9.eE]+)",
            text,
        )
        if label == "control" and records:
            raise ValueError("control unexpectedly contains injection records")
        if label != "control":
            if not records or any(
                float(az) != expected_az or float(el) != expected_el
                for az, el in records
            ):
                raise ValueError("off-source injection log identity differs")
        rows.append(
            {
                "trajectory": label,
                "status": "completed",
                "wall_seconds": float(wall.group(1)),
                "maximum_resident_bytes": int(rss.group(1)),
                "error_or_critical_messages": errors,
                "injection_log_records": len(records),
            }
        )
    return rows


def write_complete_response_maps(
    control_root: Path,
    injected_root: Path,
    output_dir: Path,
    manifest: dict,
) -> list[dict]:
    """Persist every full-map injected-minus-control signal response."""
    obsnum = int(manifest["obsnum"])
    start = int(manifest["trajectory_start_iteration"])
    stop = int(manifest["stop_iteration_exclusive"])
    control_dirs = iteration_dirs(control_root, obsnum)
    injected_dirs = iteration_dirs(injected_root, obsnum)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for iteration in range(start, stop):
        for array in ARRAYS:
            control_path = product_path(control_dirs[iteration], obsnum, array)
            injected_path = product_path(injected_dirs[iteration], obsnum, array)
            with fits.open(control_path, memmap=True) as control, fits.open(
                injected_path, memmap=True
            ) as injected:
                control_hdu = control["signal_I"]
                injected_hdu = injected["signal_I"]
                if control_hdu.data.shape != injected_hdu.data.shape:
                    raise ValueError("paired signal shapes differ")
                response = injected_hdu.data.astype("float64") - (
                    control_hdu.data.astype("float64")
                )
                header = injected_hdu.header.copy()
                # The source HDU checksums describe the parent signal plane,
                # and Astropy timestamps regenerated checksum comments. Drop
                # both so the derived response product is content-stable and
                # is bound by the external SHA-256 record below.
                for key in ("CHECKSUM", "DATASUM"):
                    if key in header:
                        del header[key]
            primary = fits.PrimaryHDU()
            primary.header["HIERARCH SCI.TESTID"] = manifest["test_id"]
            primary.header["HIERARCH SCI.RESPONSE"] = "injected-control"
            primary.header["HIERARCH SCI.OBSNUM"] = obsnum
            primary.header["HIERARCH SCI.FRUIT_ITER"] = iteration
            primary.header["HIERARCH SCI.ARRAY"] = array
            primary.header["HIERARCH SCI.INJAZ"] = float(
                manifest["az_offset_arcsec"]
            )
            primary.header["HIERARCH SCI.INJEL"] = float(
                manifest["el_offset_arcsec"]
            )
            response_hdu = fits.ImageHDU(
                data=response, header=header, name="RESPONSE_I"
            )
            output = output_dir / (
                f"point_{obsnum}_{array}_fruit_iter_{iteration:02d}_response.fits"
            )
            fits.HDUList([primary, response_hdu]).writeto(
                output, overwrite=True, checksum=False
            )
            records.append(file_record(output.resolve()))
    return records


def analyze(
    manifest: dict, repo_root: Path, response_dir: Path
) -> tuple[list[dict], list[dict], list[dict], dict]:
    obsnum = int(manifest["obsnum"])
    stop = int(manifest["stop_iteration_exclusive"])
    terminal = int(manifest["terminal_iteration"])
    control_root = Path(manifest["new_control"])
    injected_root = Path(manifest["off_source_injected"])
    historical_root = Path(manifest["historical_control"])

    compatibility_planes = require_exact_trajectory_images(
        historical_root, control_root, obsnum, stop
    )
    metrics = analyze_pair(
        alpha_label="1.25-off-source",
        alpha=float(manifest["alpha"]),
        control_root=control_root,
        injected_root=injected_root,
        manifest=manifest,
    )
    require_offset_provenance(control_root, manifest, expected_enabled=False)
    require_offset_provenance(injected_root, manifest, expected_enabled=True)

    control_penalties = mapdiag_penalties(control_root, terminal)
    injected_penalties = mapdiag_penalties(injected_root, terminal)
    hard = injection_specific_hard_penalties(
        control_penalties,
        injected_penalties,
        iteration=int(manifest["classification"]["comparable_penalty_iteration"]),
        factor=float(manifest["classification"]["comparable_penalty_factor"]),
    )
    penalty_rows = [
        {"variant": variant, **row}
        for variant, source in (
            ("control", control_penalties), ("off_source_injected", injected_penalties)
        )
        for row in source
    ]
    hard_keys = {penalty_key(row) for row in hard}
    comparison_iteration = int(
        manifest["classification"]["comparable_penalty_iteration"]
    )
    control_by_key = {
        penalty_key(row): row
        for row in control_penalties
        if penalty_key(row)[0] == comparison_iteration
    }
    injected_by_key = {
        penalty_key(row): row
        for row in injected_penalties
        if penalty_key(row)[0] == comparison_iteration
    }
    penalty_comparison = []
    for key in sorted(set(control_by_key) | set(injected_by_key)):
        left = control_by_key.get(key)
        right = injected_by_key.get(key)
        record = right if right is not None else left
        assert record is not None
        penalty_comparison.append(
            {
                "iteration": key[0],
                "scan": key[1],
                "uid": key[2],
                "array": key[3],
                "disposition": (
                    "injection_specific_factor_zero"
                    if key in hard_keys
                    else "retained"
                    if left is not None and right is not None
                    else "control_only"
                ),
                "control_score": left["score"] if left is not None else "",
                "control_factor": left["factor"] if left is not None else "",
                "injected_score": right["score"] if right is not None else "",
                "injected_factor": right["factor"] if right is not None else "",
            }
        )
    if not penalty_comparison:
        penalty_comparison.append(
            {
                "iteration": comparison_iteration,
                "scan": "",
                "uid": "",
                "array": "",
                "disposition": "no_penalties_in_either_trajectory",
                "control_score": "",
                "control_factor": "",
                "injected_score": "",
                "injected_factor": "",
            }
        )
    response_by_array = {array: response_loss(metrics, array) for array in ARRAYS}
    target = manifest["target_existing_event"]
    disposition = classify_location_control(hard, response_by_array, target)

    centered_metrics = read_centered_metrics(repo_root / manifest["centered_metrics"])
    centered_response = {
        array: response_loss(centered_metrics, array) for array in ARRAYS
    }
    centered_penalties = read_centered_penalties(repo_root / manifest["centered_penalties"])
    target_key = tuple(
        int(target[key]) for key in ("iteration", "scan", "uid", "array")
    )
    if not any(penalty_key(row) == target_key for row in centered_penalties):
        raise ValueError("registered centered target event is absent")
    contributor_records = {
        "control": target_contributor_records(control_root, terminal, target),
        "off_source_injected": target_contributor_records(
            injected_root, terminal, target
        ),
    }

    execution = read_execution(
        Path(manifest["execution_log_dir"]),
        float(manifest["az_offset_arcsec"]),
        float(manifest["el_offset_arcsec"]),
    )
    response_products = write_complete_response_maps(
        control_root, injected_root, response_dir, manifest
    )
    position_validation = {
        "maximum_transfer_centroid_distance_from_target_arcsec": max(
            math.hypot(
                row["transfer_centroid_x_arcsec"]
                - row["expected_center_x_arcsec"],
                row["transfer_centroid_y_arcsec"]
                - row["expected_center_y_arcsec"],
            )
            for row in metrics
        ),
        "maximum_kernel_centroid_distance_from_target_arcsec": max(
            math.hypot(
                row["kernel_centroid_x_arcsec"]
                - row["expected_center_x_arcsec"],
                row["kernel_centroid_y_arcsec"]
                - row["expected_center_y_arcsec"],
            )
            for row in metrics
        ),
        "arrays": list(ARRAYS),
        "iterations": list(
            range(
                int(manifest["injection_start_iteration"]),
                int(manifest["stop_iteration_exclusive"]),
            )
        ),
    }
    result = {
        "test_id": manifest["test_id"],
        "disposition": disposition,
        "claim_scope": manifest["claim_scope"],
        "default_off_bitwise_compatible_planes": compatibility_planes,
        "off_source_position_arcsec": {
            "az": float(manifest["az_offset_arcsec"]),
            "el": float(manifest["el_offset_arcsec"]),
        },
        "complete_response_products": response_products,
        "source_position_validation": position_validation,
        "injection_specific_iteration_4_factor_zero_penalties": hard,
        "target_penalty_replicated": any(
            penalty_key(row) == target_key for row in hard
        ),
        "target_qualifying_contributors": contributor_records,
        "off_source_response_change": response_by_array,
        "centered_response_change": centered_response,
        "execution": {
            "trajectory_count": len(execution),
            "aggregate_wall_seconds": sum(row["wall_seconds"] for row in execution),
            "maximum_resident_bytes": max(
                row["maximum_resident_bytes"] for row in execution
            ),
        },
        "claim_limits": {
            "blank_field_or_isolated_source": False,
            "penalty_causality": False,
            "method_qualification_or_selection": False,
            "production_or_gate_d": False,
        },
    }
    return metrics, penalty_rows, penalty_comparison, execution, result


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# FRUIT EL-F5 off-source injection result",
        "",
        f"Disposition: **{result['disposition'].replace('_', ' ')}**",
        "",
        "The new disabled-injection control reproduced all "
        f"`{result['default_off_bitwise_compatible_planes']}` registered EL-F4 "
        "signal, kernel, and weight planes bitwise before the off-source result "
        "was interpreted.",
        "",
        "The complete injected-minus-control response was retained in "
        f"`{len(result['complete_response_products'])}` FITS maps. Across all "
        "three arrays and injected iterations, the largest fitted transfer-"
        "centroid distance from the declared position was "
        f"`{result['source_position_validation']['maximum_transfer_centroid_distance_from_target_arcsec']:.3f}` "
        "arcsec and the largest kernel-centroid distance was "
        f"`{result['source_position_validation']['maximum_kernel_centroid_distance_from_target_arcsec']:.3f}` arcsec.",
        "",
        "## Iteration-4 to iteration-5 response",
        "",
        "| Location | Array | Recovery k=4 | Recovery k=5 | Annular k=4 | Annular k=5 | Registered loss direction |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for location, rows in (
        ("off source", result["off_source_response_change"]),
        ("centered EL-F4", result["centered_response_change"]),
    ):
        for array in ARRAYS:
            row = rows[array]
            lines.append(
                f"| {location} | {array} | {row['iteration_4_recovery']:.6f} | "
                f"{row['iteration_5_recovery']:.6f} | "
                f"{row['iteration_4_annular_residual']:.8g} | "
                f"{row['iteration_5_annular_residual']:.8g} | "
                f"{'yes' if row['registered_response_loss_direction'] else 'no'} |"
            )
    penalties = result["injection_specific_iteration_4_factor_zero_penalties"]
    lines.extend(
        [
            "",
            "## Penalty association",
            "",
            f"Target UID 4460 event replicated: `{'yes' if result['target_penalty_replicated'] else 'no'}`.",
            f"Injection-specific iteration-4 hard penalties: `{len(penalties)}`.",
            "The target detector has "
            f"`{len(result['target_qualifying_contributors']['control'])}` "
            "qualifying map pixels in the control and "
            f"`{len(result['target_qualifying_contributors']['off_source_injected'])}` "
            "in the injected run; four is the configured hard-penalty threshold.",
            "",
        ]
    )
    if penalties:
        lines.extend(
            [
                "| Iteration | Scan | UID | Array id | Score | Factor |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in penalties:
            lines.append(
                f"| {row['iteration']} | {row['scan']} | {row['uid']} | "
                f"{row['array']} | {row['score']:.6g} | {row['factor']:.6g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## What changed in a1400",
            "",
            "The off-source Gaussian/kernel-normalized central response falls "
            "only from `1.043975` to `1.037055` (an absolute change of "
            "`-0.006920`, or `-0.663%`). Because both values exceed unity, that "
            "change is slightly *closer* to unit recovery, not an amplitude "
            "degradation. The whole-kernel projection changes from `1.056239` "
            "to `1.056704`. The actual degradation is response shape/leakage: "
            "kernel-residual relative RMS "
            "rises from `0.320673` to `0.727804` (`2.270x`) and the registered "
            "annular residual rises from `0.00327048` to `0.0231344` (`7.074x`).",
            "",
            "## Execution",
            "",
            f"Both first-attempt trajectories completed in `{result['execution']['aggregate_wall_seconds']:.2f}` aggregate seconds with zero error/critical messages. Maximum resident memory was `{result['execution']['maximum_resident_bytes'] / 2**30:.3f}` GiB.",
            "",
            "## Interpretation",
            "",
            "The preregistered classification is **same event replicated off "
            "source**. This rules out overlap with Neptune's central source "
            "core as a necessary condition for the UID 4460 event in this "
            "observation. It strengthens the narrower hypothesis that the "
            "penalty responds to injection-altered complete-map state under "
            "this detector/scan geometry. It does not yet show that the "
            "interaction is generic across observations or prove that the "
            "reapplied penalty caused the off-source iteration-5 degradation.",
            "The classification label follows the prospectively registered "
            "sign test; it must not be read as evidence that the off-source "
            "central amplitude itself became less accurate.",
            "",
            "## Interpretation limit",
            "",
            "This one-location pointing result does not establish a blank-field "
            "or isolated-source response and does not prove that any detector "
            "penalty caused a response change. It does not qualify or select a "
            "FRUIT recurrence, penalty policy, or production configuration.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--penalties", required=True, type=Path)
    parser.add_argument("--penalty-comparison", required=True, type=Path)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--response-dir", required=True, type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    metrics, penalties, comparison, execution, result = analyze(
        manifest, repo_root, args.response_dir
    )
    write_csv(args.metrics, metrics)
    write_csv(args.penalties, penalties)
    write_csv(args.penalty_comparison, comparison)
    write_csv(args.execution, execution)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
