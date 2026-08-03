#!/usr/bin/env python3
"""Run the frozen SCI-ALIGN-001 sample-lineage/phase grid.

The signal-reading path refuses to run unless the signal-blind preregistration
committed at ``e1b29ab6...`` retains its exact aggregate digest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "validation/sci_align_001_sample_lineage_phase_2026-08-03"
LR_PACKAGE = REPO / "validation/sci_align_001_lr_beammap_2026-08-02"
BRANCH = "codex/sci-align-001-sample-lineage-phase"
PREREG_COMMIT = "e1b29ab6da16e26502bb3d26d96a2fa45b7247ef"
DT_SEC = 0.008192
HALF_CELL_SEC = DT_SEC / 2.0
EXTENT = 80
IMAGE_SIZE = 2 * EXTENT


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load_module(
    "sci_align_lineage_prepare",
    HERE / "prepare_sci_align_001_sample_lineage_phase.py",
)
lr = load_module(
    "sci_align_lr_beammap",
    HERE / "analyze_sci_align_001_lr_beammap.py",
)


@dataclass
class RawMapping:
    interface: str
    network: int
    times: np.ndarray
    slots: np.ndarray
    assigned: np.ndarray
    residual: np.ndarray
    row_for_slot: np.ndarray


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong branch: {branch}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_COMMIT, head],
        cwd=REPO,
        check=False,
    ).returncode:
        raise RuntimeError("preregistration commit is not an ancestor of HEAD")
    freeze = json.loads((PACKAGE / "preregistration_freeze.json").read_text())
    aggregate = hashlib.sha256()
    for name in freeze["ordered_files"]:
        path = PACKAGE / name
        aggregate.update(name.encode() + b"\0" + bytes.fromhex(sha256_file(path)))
    if aggregate.hexdigest() != freeze["aggregate_sha256"]:
        raise RuntimeError("preregistration aggregate changed")
    return {
        "branch": branch,
        "head_before_analysis": head,
        "preregistration_commit": PREREG_COMMIT,
        "preregistration_aggregate_sha256": aggregate.hexdigest(),
    }


def load_raw_mappings(count: int, phase: float) -> dict[int, RawMapping]:
    config = yaml.safe_load(prep.CONFIG.read_text())
    offsets = prep.offset_map(config)
    result = {}
    items = config["inputs"][0]["data_items"]
    for item in items:
        interface = str(item["meta"]["interface"])
        if not interface.startswith("toltec"):
            continue
        path = Path(item["filepath"])
        with Dataset(path) as dataset:
            network = int(prep.scalar(dataset, "Header.Toltec.RoachIndex"))
            fpga = float(prep.scalar(dataset, "Header.Toltec.FpgaFreq"))
            ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
        times = prep.reconstruct_legacy_timestamp(ts, fpga) + offsets.get(interface, 0.0)
        slots = np.floor((times - phase) / DT_SEC + 0.5).astype(np.int64)
        assigned = phase + slots.astype(float) * DT_SEC
        residual = times - assigned
        row_for_slot = np.full(count, -1, dtype=np.int64)
        inside = (slots >= 0) & (slots < count)
        row_for_slot[slots[inside]] = np.flatnonzero(inside)
        result[network] = RawMapping(
            interface, network, times, slots, assigned, residual, row_for_slot
        )
    return result


class TelescopeEvaluator:
    def __init__(self) -> None:
        names = (
            "TelAzAct",
            "TelElAct",
            "TelAzCor",
            "TelElCor",
            "SourceAz",
            "SourceEl",
        )
        with Dataset(lr.TELESCOPE) as dataset:
            self.native_time = np.asarray(
                dataset["Data.TelescopeBackend.TelTime"][:], dtype=float
            )
            self.fields = {
                name: lr.periodic_fix(
                    np.asarray(dataset[f"Data.TelescopeBackend.{name}"][:], dtype=float)
                )
                for name in names
            }
            self.hold = np.asarray(
                dataset["Data.TelescopeBackend.Hold"][:], dtype=np.uint64
            )
        config = yaml.safe_load(prep.CONFIG.read_text())
        astrometry = next(
            item
            for item in config["inputs"][0]["cal_items"]
            if item.get("type") == "astrometry"
        )["pointing_offsets"]
        values = {
            item.get("axes_name", "mjd"): item.get(
                "value_arcsec", item.get("modified_julian_date")
            )
            for item in astrometry
        }
        self.mjd_time = (
            (np.asarray(values["mjd"], dtype=float) - 40587.0) * 86400.0
        ).astype(np.int64).astype(float)
        self.pointing_az = np.asarray(values["az"], dtype=float)
        self.pointing_alt = np.asarray(values["alt"], dtype=float)

    def evaluate(self, target: np.ndarray) -> dict[str, np.ndarray]:
        target = np.asarray(target, dtype=float)
        finite = np.isfinite(target)
        safe = np.where(finite, target, self.native_time[0])
        aligned = {
            name: np.interp(safe, self.native_time, values)
            for name, values in self.fields.items()
        }
        tel_az = aligned["TelAzAct"].copy()
        wrap = tel_az - aligned["SourceAz"] > 0.9 * 2.0 * math.pi
        tel_az[wrap] -= 2.0 * math.pi
        y = (
            aligned["TelElAct"] - aligned["SourceEl"] - aligned["TelElCor"]
        ) * lr.RAD_TO_ARCSEC
        x = (
            np.cos(aligned["TelElAct"] - aligned["TelElCor"])
            * (tel_az - aligned["SourceAz"])
            - aligned["TelAzCor"]
        ) * lr.RAD_TO_ARCSEC
        x += np.interp(safe, self.mjd_time, self.pointing_az)
        y += np.interp(safe, self.mjd_time, self.pointing_alt)
        left = np.searchsorted(self.native_time, safe, side="right") - 1
        right = np.searchsorted(self.native_time, safe, side="left")
        bracket = finite & (left >= 0) & (right < self.native_time.size)
        left_safe = np.clip(left, 0, self.native_time.size - 1)
        right_safe = np.clip(right, 0, self.native_time.size - 1)
        valid = (
            bracket
            & (self.hold[left_safe] == 0)
            & (self.hold[right_safe] == 0)
            & (self.hold[left_safe] == self.hold[right_safe])
        )
        x[~finite] = np.nan
        y[~finite] = np.nan
        return {"x": x, "y": y, "valid": valid}


def model_id(basis: str, k: int, phi: float) -> str:
    return f"{basis}_k{k:+d}_phi{phi:+.1f}"


def model_coordinates(
    mapping: RawMapping,
    count: int,
    basis: str,
    k: int,
    phi: float,
    telescope: TelescopeEvaluator,
    extra_sec: float = 0.0,
) -> dict[str, np.ndarray]:
    source_row = mapping.row_for_slot
    shifted_row = source_row + k
    row_valid = (
        (source_row >= 0)
        & (shifted_row >= 0)
        & (shifted_row < mapping.times.size)
    )
    safe_row = np.clip(shifted_row, 0, mapping.times.size - 1)
    shifted_slot = np.where(row_valid, mapping.slots[safe_row], -2**62)
    if basis == "assigned_slot":
        target = np.where(
            row_valid,
            mapping.assigned[safe_row] + phi * DT_SEC + extra_sec,
            np.nan,
        )
    elif basis == "raw_detector_timestamp":
        target = np.where(
            row_valid,
            mapping.times[safe_row] + phi * DT_SEC + extra_sec,
            np.nan,
        )
    else:
        raise RuntimeError(f"unknown time basis {basis}")
    evaluated = telescope.evaluate(target)
    evaluated.update(
        {
            "source_row": source_row,
            "shifted_row": shifted_row,
            "shifted_slot": shifted_slot,
            "row_valid": row_valid,
            "target_time": target,
        }
    )
    return evaluated


def empty_map() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)


def empty_count() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)


def add_samples(
    sums: np.ndarray,
    counts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> None:
    px = np.floor(x + EXTENT).astype(int)
    py = np.floor(y + EXTENT).astype(int)
    inside = (px >= 0) & (px < IMAGE_SIZE) & (py >= 0) & (py < IMAGE_SIZE)
    np.add.at(sums, (py[inside], px[inside]), z[inside])
    np.add.at(counts, (py[inside], px[inside]), 1)


def fit_timing(
    left: dict[str, Any],
    right: dict[str, Any],
    axis: np.ndarray,
    cross_axis: np.ndarray,
    v_left: float,
    v_right: float,
) -> dict[str, Any]:
    if not left.get("quality") or not right.get("quality"):
        return {"quality": False, "reason": "left_or_right_fit_failed"}
    delta = np.array(
        [
            right["centroid_x_arcsec"] - left["centroid_x_arcsec"],
            right["centroid_y_arcsec"] - left["centroid_y_arcsec"],
        ]
    )
    parallel = float(delta @ axis)
    perpendicular = float(delta @ cross_axis)
    return {
        "quality": True,
        "parallel_arcsec": parallel,
        "perpendicular_arcsec": perpendicular,
        "v_left_arcsec_s": v_left,
        "v_right_arcsec_s": v_right,
        "timing_sec": parallel / (v_right - v_left),
        "left_centroid_x_arcsec": left["centroid_x_arcsec"],
        "left_centroid_y_arcsec": left["centroid_y_arcsec"],
        "right_centroid_x_arcsec": right["centroid_x_arcsec"],
        "right_centroid_y_arcsec": right["centroid_y_arcsec"],
        "left_major_fwhm_arcsec": left["major_fwhm_arcsec"],
        "right_major_fwhm_arcsec": right["major_fwhm_arcsec"],
        "left_minor_fwhm_arcsec": left["minor_fwhm_arcsec"],
        "right_minor_fwhm_arcsec": right["minor_fwhm_arcsec"],
        "left_ellipticity": left["ellipticity"],
        "right_ellipticity": right["ellipticity"],
        "left_amplitude": left["amplitude"],
        "right_amplitude": right["amplitude"],
    }


def map_total(
    scan_maps: dict[tuple[str, int], np.ndarray],
    scan_counts: dict[tuple[str, int], np.ndarray],
    direction: str,
    ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    sums = empty_map()
    counts = empty_count()
    for stable in ids:
        sums += scan_maps[(direction, stable)]
        counts += scan_counts[(direction, stable)]
    return sums, counts


def analyze_one_model(
    descriptor: dict[str, Any],
    state: dict[str, Any],
    registry: dict[int, dict[str, str]],
    ordinal_to_stable: dict[int, int],
    axis: np.ndarray,
    mappings: dict[int, RawMapping],
    telescope_evaluator: TelescopeEvaluator,
    base_telescope: dict[str, np.ndarray],
    detector: dict[str, np.ndarray],
    signal: np.ndarray,
    flags: np.ndarray,
    apt: Table,
    selected_detectors: np.ndarray,
    common_support: dict[int, np.ndarray],
    extra_sec: float = 0.0,
    pooled_only: bool = False,
) -> dict[str, Any]:
    basis = descriptor["time_basis"]
    k = int(descriptor["row_shift_k"])
    phi = float(descriptor["phase_phi_samples"])
    mid = str(descriptor["model_id"])
    count = base_telescope["time"].size
    cross_axis = np.array([-axis[1], axis[0]])
    coords = {
        network: model_coordinates(
            mapping, count, basis, k, phi, telescope_evaluator, extra_sec
        )
        for network, mapping in mappings.items()
    }
    for payload in coords.values():
        payload["vx"] = np.gradient(payload["x"], DT_SEC)
        payload["vy"] = np.gradient(payload["y"], DT_SEC)

    arrays = detector["arrays"]
    networks = detector["networks"]
    uid = detector["uid"]
    kind = detector["kind"]
    scan_index = detector["scan_index"]
    n_samples = detector["n_samples"]
    starts = detector["starts"]
    full_x = detector["full_x"]
    full_y = detector["full_y"]
    major_ref = detector["major_ref"]
    amp_ref = detector["amp_ref"]
    groups = ["all"]
    if not pooled_only:
        groups += ["array:a1100", "array:a1400", "array:a2000"]
        groups += [f"network:toltec{n}" for n in sorted(mappings)]
    supports = ("common",) if pooled_only else ("common", "native")
    sums = {
        (support, group, direction): empty_map()
        for support in supports
        for group in groups
        for direction in ("left", "right")
    }
    counts = {
        key: empty_count()
        for key in sums
    }
    stable_ids = sorted(registry)
    scan_sums = {
        (support, direction, stable): empty_map()
        for support in supports
        for direction in ("left", "right")
        for stable in stable_ids
    }
    scan_counts = {key: empty_count() for key in scan_sums}
    speed_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    support_counts: dict[tuple[str, str], int] = defaultdict(int)
    boundary_counts: dict[tuple[str, str], int] = defaultdict(int)

    apt_uid = np.asarray(apt["uid"], dtype=int)
    if not np.array_equal(uid, apt_uid):
        raise RuntimeError("retained detector-TOD and APT UID axes differ")

    for det in selected_detectors:
        network = int(networks[det])
        coord = coords[network]
        groups_det = ["all"]
        if not pooled_only:
            groups_det += [
                f"array:{lr.ARRAY_NAMES[int(arrays[det])]}",
                f"network:toltec{network}",
            ]
        for slot in np.flatnonzero(kind[det] == 2):
            ordinal = int(scan_index[det, slot])
            stable = ordinal_to_stable[ordinal]
            direction = registry[stable]["classification"]
            if direction not in ("left", "right") or registry[stable]["selected"] != "True":
                continue
            start = int(starts[det, slot])
            length = int(n_samples[det, slot])
            indices = np.arange(start, start + length, dtype=np.int64)
            z = signal[det, slot, :length].astype(float, copy=False)
            flag = flags[det, slot, :length]
            baseline_radial = np.hypot(
                base_telescope["x"][indices] - float(full_x[det]),
                base_telescope["y"][indices] - float(full_y[det]),
            )
            base_valid = (
                (flag == 0)
                & np.isfinite(z)
                & (baseline_radial <= 4.0 * float(major_ref[det]))
                & (base_telescope["hold_left"][indices] == 0)
                & (base_telescope["hold_right"][indices] == 0)
                & ~base_telescope["hold_transition"][indices]
            )
            shifted_slot = coord["shifted_slot"][indices]
            model_native = (
                coord["row_valid"][indices]
                & coord["valid"][indices]
                & (shifted_slot >= start)
                & (shifted_slot < start + length)
            )
            common = common_support[network][indices].copy()
            minus_slot = model_coordinates_cache[network][-1][indices]
            plus_slot = model_coordinates_cache[network][1][indices]
            common &= (
                (minus_slot >= start)
                & (minus_slot < start + length)
                & (plus_slot >= start)
                & (plus_slot < start + length)
                & coord["valid"][indices]
            )
            masks = {"common": base_valid & common}
            if not pooled_only:
                masks["native"] = base_valid & model_native
            for support, valid in masks.items():
                support_counts[(support, direction)] += int(np.sum(valid))
                if support == "native":
                    boundary_counts[(support, direction)] += int(
                        np.sum(valid & ~masks["common"])
                    )
                if not np.any(valid):
                    continue
                x = coord["x"][indices][valid] - float(full_x[det])
                y = coord["y"][indices][valid] - float(full_y[det])
                normalized = z[valid] / float(amp_ref[det])
                projected_speed = (
                    coord["vx"][indices][valid] * axis[0]
                    + coord["vy"][indices][valid] * axis[1]
                )
                for group in groups_det:
                    add_samples(
                        sums[(support, group, direction)],
                        counts[(support, group, direction)],
                        x,
                        y,
                        normalized,
                    )
                    speed_values[(support, group, direction)].append(
                        float(np.median(projected_speed))
                    )
                add_samples(
                    scan_sums[(support, direction, stable)],
                    scan_counts[(support, direction, stable)],
                    x,
                    y,
                    normalized,
                )

    group_rows = []
    pooled_results: dict[str, dict[str, Any]] = {}
    fit_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for support in supports:
        for group in groups:
            array_id = -1
            if group.startswith("array:"):
                array_id = next(
                    key for key, value in lr.ARRAY_NAMES.items() if group == f"array:{value}"
                )
            elif group.startswith("network:"):
                network = int(group.split("toltec", 1)[1])
                first = int(np.flatnonzero(networks == network)[0])
                array_id = int(arrays[first])
            fits = {}
            velocities = {}
            for direction in ("left", "right"):
                fit = lr.map_fit(
                    sums[(support, group, direction)],
                    counts[(support, group, direction)],
                    array_id,
                    EXTENT,
                )
                fits[direction] = fit
                fit_cache[(support, group, direction)] = fit
                velocities[direction] = float(
                    np.median(speed_values[(support, group, direction)])
                )
            result = fit_timing(
                fits["left"],
                fits["right"],
                axis,
                cross_axis,
                velocities["left"],
                velocities["right"],
            )
            row = {
                "model_id": mid,
                "time_basis": basis,
                "row_shift_k": k,
                "phase_phi_samples": phi,
                "extra_profile_sec": extra_sec,
                "support": support,
                "group": group,
                **result,
            }
            group_rows.append(row)
            if group == "all":
                pooled_results[support] = row

    controls = {}
    if not pooled_only:
        for support in supports:
            pooled = pooled_results[support]
            if not pooled.get("quality"):
                continue
            v_left = float(pooled["v_left_arcsec_s"])
            v_right = float(pooled["v_right_arcsec_s"])
            delete = []
            full_fit = {
                direction: fit_cache[(support, "all", direction)]
                for direction in ("left", "right")
            }
            for stable in stable_ids:
                direction = registry[stable]["classification"]
                modified_sum = (
                    sums[(support, "all", direction)]
                    - scan_sums[(support, direction, stable)]
                )
                modified_count = (
                    counts[(support, "all", direction)]
                    - scan_counts[(support, direction, stable)]
                )
                changed = lr.map_fit(modified_sum, modified_count, -1, EXTENT)
                pair = dict(full_fit)
                pair[direction] = changed
                value = fit_timing(
                    pair["left"], pair["right"], axis, cross_axis, v_left, v_right
                )
                if value.get("quality"):
                    delete.append(
                        {
                            "omitted_stable_scan_id": stable,
                            "timing_sec": value["timing_sec"],
                            "parallel_arcsec": value["parallel_arcsec"],
                            "perpendicular_arcsec": value["perpendicular_arcsec"],
                        }
                    )
            theta = np.asarray([row["timing_sec"] for row in delete], dtype=float)
            parallel = np.asarray([row["parallel_arcsec"] for row in delete], dtype=float)
            perpendicular = np.asarray(
                [row["perpendicular_arcsec"] for row in delete], dtype=float
            )

            def jackknife_se(values: np.ndarray) -> float:
                n = values.size
                return float(
                    np.sqrt((n - 1) / n * np.sum((values - np.mean(values)) ** 2))
                )

            se = jackknife_se(theta)
            pooled["timing_jackknife_se_sec"] = se
            pooled["timing_68_low_sec"] = pooled["timing_sec"] - se
            pooled["timing_68_high_sec"] = pooled["timing_sec"] + se
            pooled["timing_95_low_sec"] = pooled["timing_sec"] - 1.96 * se
            pooled["timing_95_high_sec"] = pooled["timing_sec"] + 1.96 * se
            pooled["parallel_jackknife_se_arcsec"] = jackknife_se(parallel)
            pooled["perpendicular_jackknife_se_arcsec"] = jackknife_se(perpendicular)
            pooled["jackknife_replicates"] = len(delete)

            ordered = sorted(
                stable_ids,
                key=lambda stable: int(registry[stable]["compatibility_ordinal_1based"]),
            )
            halves = {"first": ordered[: len(ordered) // 2], "second": ordered[len(ordered) // 2 :]}
            half_results = {}
            for label, ids in halves.items():
                fits = {}
                velocities = {}
                for direction in ("left", "right"):
                    chosen = [s for s in ids if registry[s]["classification"] == direction]
                    smap, cmap = map_total(
                        {(d, s): scan_sums[(support, d, s)] for d in ("left", "right") for s in stable_ids},
                        {(d, s): scan_counts[(support, d, s)] for d in ("left", "right") for s in stable_ids},
                        direction,
                        chosen,
                    )
                    fits[direction] = lr.map_fit(smap, cmap, -1, EXTENT)
                    velocities[direction] = (
                        v_left if direction == "left" else v_right
                    )
                half_results[label] = fit_timing(
                    fits["left"], fits["right"], axis, cross_axis,
                    velocities["left"], velocities["right"]
                )

            nulls = {}
            for direction in ("left", "right"):
                direction_ids = [
                    s for s in ordered if registry[s]["classification"] == direction
                ]
                partitions = (direction_ids[::2], direction_ids[1::2])
                fits = []
                for ids in partitions:
                    smap, cmap = map_total(
                        {(d, s): scan_sums[(support, d, s)] for d in ("left", "right") for s in stable_ids},
                        {(d, s): scan_counts[(support, d, s)] for d in ("left", "right") for s in stable_ids},
                        direction,
                        ids,
                    )
                    fits.append(lr.map_fit(smap, cmap, -1, EXTENT))
                if all(fit.get("quality") for fit in fits):
                    delta = np.array(
                        [
                            fits[1]["centroid_x_arcsec"] - fits[0]["centroid_x_arcsec"],
                            fits[1]["centroid_y_arcsec"] - fits[0]["centroid_y_arcsec"],
                        ]
                    )
                    nulls[direction] = {
                        "parallel_arcsec": float(delta @ axis),
                        "perpendicular_arcsec": float(delta @ cross_axis),
                        "partition_rule": "alternating within frozen direction-ordered set",
                    }
            controls[support] = {
                "jackknife": delete,
                "time_halves": half_results,
                "same_direction_null": nulls,
            }

    network_model = {}
    if not pooled_only:
        for support in supports:
            network_rows = [
                row
                for row in group_rows
                if row["support"] == support
                and row["group"].startswith("network:")
                and row.get("quality")
            ]
            residual = np.asarray(
                [
                    float(np.mean(mappings[int(row["group"].split("toltec", 1)[1])].residual))
                    for row in network_rows
                ]
            )
            timing = np.asarray([float(row["timing_sec"]) for row in network_rows])
            slope, intercept = np.polyfit(residual, timing, 1)
            network_model[support] = {
                "network_count": len(network_rows),
                "pearson": float(np.corrcoef(residual, timing)[0, 1]),
                "slope": float(slope),
                "intercept_sec": float(intercept),
                "residual_definition": json.loads(
                    (PACKAGE / "preregistered_protocol.json").read_text()
                )["assigned_slot_residual_definition"],
            }

    return {
        "descriptor": descriptor,
        "extra_profile_sec": extra_sec,
        "group_rows": group_rows,
        "pooled": pooled_results,
        "controls": controls,
        "network_residual_models": network_model,
        "support_counts": [
            {
                "support": support,
                "direction": direction,
                "detector_sample_count": count_value,
                "native_only_boundary_sample_count": boundary_counts[(support, direction)],
            }
            for (support, direction), count_value in sorted(support_counts.items())
        ],
    }


# Filled in by analyze() once from the exact raw mappings.  Keeping this map
# outside the inner loop avoids rebuilding k-neighbor slot identities for each
# detector while retaining an explicit genuine row shift.
model_coordinates_cache: dict[int, dict[int, np.ndarray]] = {}


def analyze() -> None:
    identity = verify_freeze()
    protocol = json.loads((PACKAGE / "preregistered_protocol.json").read_text())
    registry_rows = list(
        csv.DictReader((LR_PACKAGE / "scan_direction_registry.csv").open())
    )
    registry = {int(row["stable_scan_id"]): row for row in registry_rows}
    state = lr.read_state()
    count = int(state["alignment"]["governing_compatibility_axis"]["sample_count"])
    phase = float(state["alignment"]["grid"]["phase_sec"])
    base_telescope = lr.read_telescope(phase, count)
    lr_protocol = json.loads((LR_PACKAGE / "preregistered_protocol.json").read_text())
    axis = np.asarray(
        [
            lr_protocol["positive_scan_axis"]["x_az_tangent"],
            lr_protocol["positive_scan_axis"]["y_el_tangent"],
        ],
        dtype=float,
    )
    ordinal_to_stable = {
        int(row["compatibility_ordinal"] + 1): int(row["stable_id"])
        for row in state["records"]
    }
    mappings = load_raw_mappings(count, phase)
    telescope_evaluator = TelescopeEvaluator()

    global model_coordinates_cache
    common_support = {}
    for network, mapping in mappings.items():
        cache = {}
        for k in (-1, 1):
            source = mapping.row_for_slot
            shifted = source + k
            valid = (source >= 0) & (shifted >= 0) & (shifted < mapping.times.size)
            safe = np.clip(shifted, 0, mapping.times.size - 1)
            cache[k] = np.where(valid, mapping.slots[safe], -2**62)
        model_coordinates_cache[network] = cache
        common_support[network] = (
            (mapping.row_for_slot >= 0)
            & (cache[-1] == np.arange(count, dtype=np.int64) - 1)
            & (cache[1] == np.arange(count, dtype=np.int64) + 1)
        )

    apt = Table.read(lr.OUTPUT_APT, format="ascii.ecsv")
    cohort = np.loadtxt(PACKAGE / "frozen_confirmatory_uids.txt", dtype=int)
    with Dataset(lr.DETECTOR_TOD) as dataset:
        detector = {
            "uid": np.asarray(dataset["detector_tod_uid"][:], dtype=int),
            "arrays": np.asarray(dataset["detector_tod_array"][:], dtype=int),
            "networks": np.asarray(dataset["detector_tod_network"][:], dtype=int),
            "kind": np.asarray(dataset["detector_tod_slot_kind"][:], dtype=int),
            "scan_index": np.asarray(dataset["detector_tod_scan_index"][:], dtype=int),
            "n_samples": np.asarray(dataset["detector_tod_n_samples"][:], dtype=int),
            "starts": np.asarray(
                dataset["detector_tod_scan_inner_start_sample"][:], dtype=int
            ),
            "full_x": np.asarray(dataset["detector_tod_fit_x_t_arcsec"][:], dtype=float),
            "full_y": np.asarray(dataset["detector_tod_fit_y_t_arcsec"][:], dtype=float),
            "major_ref": np.maximum(
                np.asarray(apt["a_fwhm"], dtype=float),
                np.asarray(apt["b_fwhm"], dtype=float),
            ),
            "amp_ref": np.asarray(apt["amp"], dtype=float),
        }
        # First confirmatory signal read occurs only after verify_freeze().
        signal = np.asarray(dataset["signal"][:], dtype=np.float32)
        flags = np.asarray(dataset["flags"][:], dtype=np.int8)
    selected = np.flatnonzero(np.isin(detector["uid"], cohort))
    if selected.size != 4809:
        raise RuntimeError(f"frozen detector selection changed: {selected.size}")

    descriptors = list(csv.DictReader((PACKAGE / "model_registry.csv").open()))
    results = []
    checkpoint_dir = PACKAGE / "model_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for index, descriptor in enumerate(descriptors, 1):
        descriptor = dict(descriptor)
        descriptor["row_shift_k"] = int(descriptor["row_shift_k"])
        descriptor["phase_phi_samples"] = float(descriptor["phase_phi_samples"])
        path = checkpoint_dir / f"{descriptor['model_id']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            print(f"[{index}/{len(descriptors)}] reuse {descriptor['model_id']}", flush=True)
        else:
            print(f"[{index}/{len(descriptors)}] analyze {descriptor['model_id']}", flush=True)
            result = analyze_one_model(
                descriptor,
                state,
                registry,
                ordinal_to_stable,
                axis,
                mappings,
                telescope_evaluator,
                base_telescope,
                detector,
                signal,
                flags,
                apt,
                selected,
                common_support,
            )
            write_json(path, result)
        results.append(result)

    pooled_common = [
        result["pooled"]["common"]
        for result in results
        if result["pooled"].get("common", {}).get("quality")
    ]
    best = min(pooled_common, key=lambda row: abs(float(row["timing_sec"])))
    best_descriptor = next(
        result["descriptor"] for result in results if result["descriptor"]["model_id"] == best["model_id"]
    )
    profile = []
    for offset in np.linspace(-0.0015, 0.0015, 7):
        descriptor = dict(best_descriptor)
        descriptor["model_id"] = f"{best_descriptor['model_id']}_profile_{offset:+.7f}s"
        print(f"[profile] {offset:+.7f} s around {best_descriptor['model_id']}", flush=True)
        value = analyze_one_model(
            descriptor,
            state,
            registry,
            ordinal_to_stable,
            axis,
            mappings,
            telescope_evaluator,
            base_telescope,
            detector,
            signal,
            flags,
            apt,
            selected,
            common_support,
            extra_sec=float(offset),
            pooled_only=True,
        )["pooled"]["common"]
        profile.append(value)

    group_rows = [row for result in results for row in result["group_rows"]]
    support_rows = [
        {"model_id": result["descriptor"]["model_id"], **row}
        for result in results
        for row in result["support_counts"]
    ]
    controls = {
        result["descriptor"]["model_id"]: {
            "controls": result["controls"],
            "network_residual_models": result["network_residual_models"],
        }
        for result in results
    }
    write_csv(PACKAGE / "model_group_results.csv", group_rows)
    write_csv(PACKAGE / "sample_support_results.csv", support_rows)
    write_json(PACKAGE / "model_controls.json", controls)
    write_csv(PACKAGE / "continuous_profile.csv", profile)

    assigned_baseline = next(
        row
        for row in pooled_common
        if row["model_id"] == "assigned_slot_k+0_phi+0.0"
    )
    assigned_combined = next(
        row
        for row in pooled_common
        if row["model_id"] == "assigned_slot_k+1_phi+0.5"
    )
    raw_combined = next(
        row
        for row in pooled_common
        if row["model_id"] == "raw_detector_timestamp_k+1_phi+0.5"
    )
    network_model = next(
        result["network_residual_models"]["common"]
        for result in results
        if result["descriptor"]["model_id"] == best["model_id"]
    )
    first = next(
        result["controls"]["common"]["time_halves"]
        for result in results
        if result["descriptor"]["model_id"] == best["model_id"]
    )
    if not json.loads((PACKAGE / "stage_a_conclusion.json").read_text())[
        "direct_row_mismatch_found"
    ]:
        if abs(float(best["timing_sec"])) <= 1.96 * float(best["timing_jackknife_se_sec"]):
            classification = "evidence only for a common approximately 1.5-sample effective offset"
        elif abs(float(best["timing_sec"])) < 0.5 * abs(float(assigned_baseline["timing_sec"])):
            classification = "common component explained but interface/time-dependent residual remains"
        else:
            classification = "hypothesis not supported"
    else:
        classification = "demonstrated row-index error plus half-sample convention error"

    summary = {
        "identity": identity,
        "model_count": len(results),
        "frozen_detector_count": int(selected.size),
        "baseline_common_support": assigned_baseline,
        "assigned_combined_common_support": assigned_combined,
        "raw_combined_common_support": raw_combined,
        "best_discrete_common_support": best,
        "best_network_residual_model": network_model,
        "best_time_halves": first,
        "classification": classification,
        "direct_row_mismatch_found": False,
        "physical_timestamp_semantics": "unresolved",
        "production_correction_authorized": False,
        "unity_contacted": False,
        "new_citlali_reductions": 0,
    }
    write_json(PACKAGE / "result_summary.json", summary)

    plots = PACKAGE / "plots"
    plots.mkdir(exist_ok=True)
    assigned = [
        row
        for row in pooled_common
        if row["time_basis"] == "assigned_slot"
    ]
    raw = [
        row
        for row in pooled_common
        if row["time_basis"] == "raw_detector_timestamp"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, rows, title in zip(axes, (assigned, raw), ("Assigned-slot basis", "Raw-timestamp basis")):
        grid = np.full((3, 3), np.nan)
        for row in rows:
            i = {-1: 0, 0: 1, 1: 2}[int(row["row_shift_k"])]
            j = {-0.5: 0, 0.0: 1, 0.5: 2}[float(row["phase_phi_samples"])]
            grid[i, j] = 1000.0 * float(row["timing_sec"])
        image = ax.imshow(grid, origin="lower", cmap="coolwarm")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{grid[i,j]:+.3f}", ha="center", va="center")
        ax.set_xticks(range(3), ["-0.5", "0", "+0.5"])
        ax.set_yticks(range(3), ["-1", "0", "+1"])
        ax.set_xlabel("phase phi (samples)")
        ax.set_ylabel("row shift k")
        ax.set_title(title + " residual (ms)")
        fig.colorbar(image, ax=ax)
    fig.savefig(plots / "discrete_timing_grid.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(
        [1000.0 * float(row["extra_profile_sec"]) for row in profile],
        [1000.0 * float(row["timing_sec"]) for row in profile],
        marker="o",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("additional time shift around best discrete model (ms)")
    ax.set_ylabel("direction-reversal residual (ms)")
    ax.set_title("Secondary continuous timing profile")
    fig.savefig(plots / "continuous_profile.png", dpi=160)
    plt.close(fig)

    write_json(
        PACKAGE / "analysis_identity.json",
        {
            **identity,
            "confirmatory_signal_read_after_freeze_verification": True,
            "analysis_tool": str(Path(__file__).relative_to(REPO)),
            "analysis_tool_sha256": sha256_file(Path(__file__)),
            "protocol": protocol,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    analyze()


if __name__ == "__main__":
    main()
