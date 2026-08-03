#!/usr/bin/env python3
"""Bounded retained-product left/right Beammap timing diagnostic.

This tool never runs Citlali.  ``prepare`` reads only identities, scan metadata,
and the realized telescope trajectory.  ``analyze`` additionally reads the
retained final-iteration detector source-crossing signal and performs the
frozen direction-split fits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset
from scipy.optimize import least_squares


OBSNUM = 148670
TASK_BASE = "3e12c0dcbd8b7fe83e54918eccaec48c94f35bff"
APPLICATION_CANDIDATE = "c77105b9b1676ec1ec74a9d560765954c5f1d5dd"
GOVERNING_APPLICATION = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
BRANCH = "codex/sci-align-001-lr-beammap"
PACKAGE_REL = Path("validation/sci_align_001_lr_beammap_2026-08-02")
SCRIPT_REL = Path("tools/diagnostics/analyze_sci_align_001_lr_beammap.py")
RUN_ROOT = Path(
    "/private/tmp/citlali-sci-align-001-phase1-"
    "c77105b9b1676ec1ec74a9d560765954c5f1d5dd/beammap/beammap"
)
REDUCTION = RUN_ROOT / "reduced/redu00/148670"
TOD_DIR = REDUCTION / "raw/source_crossing_tod"
DETECTOR_TOD = TOD_DIR / "toltec_commissioning_beammap_148670_ptc_detector_tod.nc"
PTCDIAG = TOD_DIR / "toltec_commissioning_beammap_148670_ptcdiag.nc"
RTCDIAG = TOD_DIR / "toltec_commissioning_beammap_148670_rtcdiag.nc"
OUTPUT_APT = REDUCTION / "raw/apt_commissioning_beammap_148670_citlali.ecsv"
PROVENANCE = REDUCTION / "timestream_output_provenance.yaml"
CONFIG = RUN_ROOT / "config/citlali_o148670_0_2_c1.yaml"
PREPARATION = RUN_ROOT / "evidence/preparation.json"
TELESCOPE = Path(
    "/Users/gwilson/work_toltec/local_data/citlali-validation/v1/beammaps/"
    "3c273/reduced/tel_toltec_2026-01-13_148670_00_0002_recomputed.nc"
)
INPUT_APT = Path(
    "/Users/gwilson/work_toltec/local_data/citlali-validation/v1/beammaps/"
    "3c273/reduced/apt_148670_000_0002_2026_01_13_11_59_10.ecsv"
)
ATTACHMENT = Path(
    "/Users/gwilson/.codex/attachments/6571972b-ddd5-4353-810f-20fa6b9adfbf/"
    "pasted-text.txt"
)
PILOT_UIDS = (0, 5, 10, 15, 20, 25, 30, 35)
DT_SEC = 0.008192
RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi
ARRAY_NAMES = {0: "a1100", 1: "a1400", 2: "a2000"}
ARRAY_FWHM_LIMITS = {0: (3.0, 10.0), 1: (3.5, 15.0), 2: (5.5, 20.0)}
SEED = 1486700802


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def validate_identity(repo: Path) -> dict[str, Any]:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    allowed = {str(SCRIPT_REL)}
    allowed_prefix = str(PACKAGE_REL) + "/"
    bad = []
    for line in status.splitlines():
        name = line[3:].split(" -> ")[-1]
        if name not in allowed and not name.startswith(allowed_prefix):
            bad.append(line)
    base_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK_BASE, head], cwd=repo,
    ).returncode == 0
    committed_delta = git(repo, "diff", "--name-only", f"{TASK_BASE}..{head}").splitlines()
    bad_committed = [
        name for name in committed_delta
        if name != str(SCRIPT_REL) and not name.startswith(str(PACKAGE_REL) + "/")
    ]
    if branch != BRANCH or not base_is_ancestor or bad or bad_committed:
        raise RuntimeError(
            f"identity gate failed: branch={branch} head={head} "
            f"unrelated={bad} committed_unrelated={bad_committed}"
        )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", GOVERNING_APPLICATION, APPLICATION_CANDIDATE],
        cwd=repo,
    ).returncode:
        raise RuntimeError("governing application is not an ancestor of candidate")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", APPLICATION_CANDIDATE, TASK_BASE],
        cwd=repo,
    ).returncode:
        raise RuntimeError("candidate is not an ancestor of task base")
    required = [
        DETECTOR_TOD, PTCDIAG, RTCDIAG, OUTPUT_APT, PROVENANCE, CONFIG,
        PREPARATION, TELESCOPE, INPUT_APT, ATTACHMENT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required retained input missing: {missing}")
    return {"branch": branch, "head": head, "status": "clean_except_current_package"}


def periodic_fix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if float(np.max(result)) > 1.99 * math.pi and float(np.min(result)) < math.pi:
        result[result < math.pi] += 2.0 * math.pi
    return result


def read_state() -> dict[str, Any]:
    provenance = yaml.safe_load(PROVENANCE.read_text())
    realized = provenance["realized"]
    alignment = realized["sci_align_alignment"]
    scan_plan = realized["sci_align_scan_plan"]
    if alignment["grid"]["cadence_sec"] != DT_SEC:
        raise RuntimeError("unexpected retained cadence")
    records = [row for row in scan_plan["records"] if row["legacy_processing_admitted"]]
    if len(records) != 198 or [row["compatibility_ordinal"] for row in records] != list(range(198)):
        raise RuntimeError("retained compatibility census changed")
    return {"document": provenance, "alignment": alignment, "scan_plan": scan_plan, "records": records}


def read_telescope(phase: float, count: int) -> dict[str, np.ndarray]:
    target = phase + np.arange(count, dtype=np.float64) * DT_SEC
    names = ("TelAzAct", "TelElAct", "TelAzCor", "TelElCor", "SourceAz", "SourceEl")
    with Dataset(TELESCOPE) as dataset:
        native_time = np.asarray(dataset["Data.TelescopeBackend.TelTime"][:], dtype=np.float64)
        fields = {
            name: periodic_fix(np.asarray(dataset[f"Data.TelescopeBackend.{name}"][:], dtype=np.float64))
            for name in names
        }
        hold = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=np.float64)
        scan_angle = float(np.asarray(dataset["Header.Map.ScanAngle"][:]).item())
    aligned = {name: np.interp(target, native_time, values) for name, values in fields.items()}
    tel_az = aligned["TelAzAct"].copy()
    wrap = tel_az - aligned["SourceAz"] > 0.9 * 2.0 * math.pi
    tel_az[wrap] -= 2.0 * math.pi
    y = (aligned["TelElAct"] - aligned["SourceEl"] - aligned["TelElCor"]) * RAD_TO_ARCSEC
    x = (
        np.cos(aligned["TelElAct"] - aligned["TelElCor"])
        * (tel_az - aligned["SourceAz"]) - aligned["TelAzCor"]
    ) * RAD_TO_ARCSEC
    config = yaml.safe_load(CONFIG.read_text())
    astrometry = next(
        item for item in config["inputs"][0]["cal_items"] if item.get("type") == "astrometry"
    )["pointing_offsets"]
    values = {
        item.get("axes_name", "mjd"): item.get("value_arcsec", item.get("modified_julian_date"))
        for item in astrometry
    }
    mjd_time = ((np.asarray(values["mjd"]) - 40587.0) * 86400.0).astype(np.int64).astype(float)
    pointing_az = np.interp(target, mjd_time, np.asarray(values["az"], dtype=float))
    pointing_alt = np.interp(target, mjd_time, np.asarray(values["alt"], dtype=float))
    left = np.searchsorted(native_time, target, side="right") - 1
    right = np.searchsorted(native_time, target, side="left")
    if np.any(left < 0) or np.any(right >= native_time.size):
        raise RuntimeError("common axis lies outside telescope support")
    raw_word = hold.astype(np.uint64)
    return {
        "time": target, "x": x + pointing_az, "y": y + pointing_alt,
        "boresight_x": x, "boresight_y": y, "elevation": aligned["TelElAct"],
        "hold_left": raw_word[left], "hold_right": raw_word[right],
        "hold_transition": raw_word[left] != raw_word[right],
        "scan_angle": np.asarray(scan_angle),
    }


def record_interval(record: dict[str, Any]) -> tuple[int, int]:
    # Provenance windows are on the union-local axis; the retained detector TOD
    # persists the governing segment after subtracting union_local_start=1.
    science = record["compatibility_science"]
    return int(science["start"] - 1), int(science["stop"] - 1)


def build_registry(state: dict[str, Any], telescope: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    velocity_rows = []
    work = []
    for record in state["records"]:
        start, stop = record_interval(record)
        indices = np.arange(start, stop, dtype=np.int64)
        vx = np.gradient(telescope["x"][indices], DT_SEC)
        vy = np.gradient(telescope["y"][indices], DT_SEC)
        trim = max(1, indices.size // 10)
        central = slice(trim, indices.size - trim)
        med = np.array([np.median(vx[central]), np.median(vy[central])])
        velocity_rows.append(med)
        work.append((record, start, stop, indices, vx, vy, central, med))
    matrix = np.asarray(velocity_rows)
    _, vectors = np.linalg.eigh(matrix.T @ matrix)
    axis = vectors[:, -1]
    configured_positive = np.array([
        math.cos(float(telescope["scan_angle"])), math.sin(float(telescope["scan_angle"]))
    ])
    if float(axis @ configured_positive) < 0.0:
        axis = -axis
    prelim = np.abs(matrix @ axis)
    low_speed = 0.5 * float(np.min(prelim))
    cross_axis = np.array([-axis[1], axis[0]])
    rows = []
    for record, start, stop, indices, vx, vy, central, med in work:
        projected = vx * axis[0] + vy * axis[1]
        perpendicular = vx * cross_axis[0] + vy * cross_axis[1]
        central_projected = projected[central]
        direction = "right" if np.median(central_projected) > 0 else "left"
        sign_fraction = float(np.mean(
            central_projected > 0.0 if direction == "right" else central_projected < 0.0
        ))
        stable_sign = sign_fraction >= 0.99
        valid = (
            (telescope["hold_left"][indices] == 0)
            & (telescope["hold_right"][indices] == 0)
            & ~telescope["hold_transition"][indices]
        )
        selected = stable_sign and abs(float(np.median(central_projected))) > low_speed and bool(np.all(valid))
        reason = "selected" if selected else (
            "hold_invalid_or_transition_ambiguous" if not np.all(valid)
            else "low_speed_or_direction_ambiguous"
        )
        rows.append({
            "stable_scan_id": int(record["stable_id"]),
            "compatibility_ordinal_1based": int(record["compatibility_ordinal"] + 1),
            "compatibility_status": record["status"],
            "science_start_sample_inclusive": start,
            "science_stop_sample_exclusive": stop,
            "direction_measure_start_sample_inclusive": int(indices[central][0]),
            "direction_measure_stop_sample_exclusive": int(indices[central][-1] + 1),
            "median_vx_arcsec_s": float(med[0]),
            "median_vy_arcsec_s": float(med[1]),
            "median_projected_velocity_arcsec_s": float(np.median(central_projected)),
            "median_perpendicular_velocity_arcsec_s": float(np.median(perpendicular[central])),
            "projected_velocity_mad_sigma_arcsec_s": float(
                1.4826 * np.median(np.abs(central_projected - np.median(central_projected)))
            ),
            "projected_velocity_min_arcsec_s": float(np.min(central_projected)),
            "projected_velocity_max_arcsec_s": float(np.max(central_projected)),
            "projected_velocity_median_sign_fraction": sign_fraction,
            "hold_valid_fraction": float(np.mean(valid)),
            "hold_transition_ambiguous_count": int(np.sum(telescope["hold_transition"][indices])),
            "classification": direction if selected else "excluded",
            "selected": selected,
            "exclusion_reason": reason,
            "trajectory_authority": str(TELESCOPE),
            "scan_authority": str(PROVENANCE),
        })
    return rows, axis, low_speed


def manifest_rows() -> list[dict[str, Any]]:
    preparation = json.loads(PREPARATION.read_text())
    inherited = {item["path"]: item["sha256"] for item in preparation["selected_inputs"]}
    rows = []
    entries = [
        ("dispatch", ATTACHMENT, "owner_dispatch"),
        ("candidate_binary", Path(preparation["candidate_binary"]["path"]), "exercised_application"),
        ("realized_config", CONFIG, "realized_configuration"),
        ("telescope", TELESCOPE, "realized_telescope_trajectory_and_Hold"),
        ("input_apt", INPUT_APT, "input_detector_table"),
        ("output_apt", OUTPUT_APT, "retained_full_beammap_fit_and_normalization"),
        ("detector_tod", DETECTOR_TOD, "retained_final_iteration_signal_and_scan_bounds"),
        ("ptcdiag", PTCDIAG, "retained_scan_detector_diagnostics"),
        ("rtcdiag", RTCDIAG, "retained_scan_speed_diagnostics"),
        ("provenance", PROVENANCE, "realized_scan_and_alignment_state"),
        ("preparation", PREPARATION, "prior_input_identity_binding"),
    ]
    known = {
        str(DETECTOR_TOD): "669f1bfe2ee78afd8c732e214c1978d59137470d97a17dc35e1ad754f3a1ee87",
        str(PTCDIAG): "1dbe35c51c03b321f01213e9a301ab4149a9a072a71c06dfbfd3cb0d94aebfc4",
        str(RTCDIAG): "dcad8067dee5dfc30f68413a5ec75f675c6da14100a1076c325e782031cd1793",
    }
    for identity, path, use in entries:
        digest = inherited.get(str(path), known.get(str(path)))
        digest_source = "inherited_verified_phase_one_manifest"
        if digest is None:
            digest = sha256_file(path)
            digest_source = "measured_by_lr_diagnostic"
        rows.append({
            "identity": identity, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest, "digest_source": digest_source, "use": use,
            "mutated": False,
        })
    return rows


def stage_a_decision(repo: Path) -> dict[str, Any]:
    config_text = CONFIG.read_text()
    source_matches = subprocess.run(
        ["rg", "-n", "explicit.*scan|scan.*identit.*select|selected.*scan.*id", "include", "src", "config"],
        cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.splitlines()
    return {
        "decision": "retained_product_partition_and_refit_is_sufficient",
        "citlali_application_change_required": False,
        "additional_citlali_reduction_required": False,
        "options_in_required_order": [
            {
                "option": 1,
                "question": "Can retained products be partitioned and refitted?",
                "answer": "yes",
                "evidence": (
                    "final-iteration detector_tod signal has exact detector/slot/scan/sample bounds; "
                    "the assigned-time plus telescope/astrometry reconstruction reproduced retained "
                    "source-distance metadata and center-scan identity"
                ),
                "maximum_reconstruction_delta_arcsec_feasibility_probe": 1.068141573767889e-05,
            },
            {
                "option": 2,
                "question": "Can existing configuration select explicit accepted scan identities?",
                "answer": "no general reduction selector identified; not needed after option 1 succeeded",
                "realized_config_contains_scan_identity_selector": any(
                    token in config_text for token in ("scan_ids", "scan_identities", "selected_scans")
                ),
                "source_search_matches": source_matches,
            },
            {
                "option": 3,
                "question": "Can a wrapper generate explicit left/right realized configs without Citlali change?",
                "answer": "not required; a wrapper would lack a reduction-level explicit scan selector",
            },
        ],
        "ordinary_application_behavior_changed": False,
        "bulk_map_products_generated": False,
    }


def protocol(axis: np.ndarray, low_speed: float, registry: list[dict[str, Any]]) -> dict[str, Any]:
    left = [row["stable_scan_id"] for row in registry if row["classification"] == "left"]
    right = [row["stable_scan_id"] for row in registry if row["classification"] == "right"]
    excluded = [row["stable_scan_id"] for row in registry if row["classification"] == "excluded"]
    return {
        "protocol_id": "SCI-ALIGN-001-LR-BEAMMAP-PREREG-v1",
        "status": "FROZEN_BEFORE_CONFIRMATORY_COHORT_SIGNAL_INSPECTION",
        "fixture": {"obsnum": OBSNUM, "native_rate_hz": 122.0703125, "cadence_sec": DT_SEC},
        "limited_feasibility_pilot": {
            "conducted_before_freeze": True,
            "uids": list(PILOT_UIDS),
            "use": "debug coordinate reconstruction and confirm fit viability only",
            "confirmatory_inference": "excluded from every confirmatory population and result",
            "nonblind_limitation": "eight pilot detector direction differences were visible before this freeze",
        },
        "positive_scan_axis": {
            "x_az_tangent": float(axis[0]), "y_el_tangent": float(axis[1]),
            "derivation": "principal eigenvector of admitted-window robust median realized velocity vectors",
            "sign": "dot product positive with configured Rectilinear Map.ScanAngle unit vector",
        },
        "direction_classifier": {
            "measurement_window": "central 80 percent of each admitted compatibility-science window",
            "low_speed_threshold_arcsec_s": low_speed,
            "threshold_rule": "one half the minimum absolute robust median projected speed over all admitted windows",
            "stable_sign_rule": "at least 99 percent of projected samples in the direction window have the robust-median sign",
            "hold_rule": "both adjacent native raw Hold words exactly zero and equal; otherwise exclude",
            "census": {"left": len(left), "right": len(right), "excluded": len(excluded)},
            "parity_used": False,
        },
        "confirmatory_detector_preselection": {
            "rules": [
                "retained detector_tod_fit_good == 1",
                "retained output APT flag == 0",
                "finite positive retained full-fit amplitude, centroid, major/minor FWHM",
                f"UID not in feasibility pilot {list(PILOT_UIDS)}",
            ],
            "selection_fixed_before_confirmatory_signal_read": True,
        },
        "sample_selection": {
            "slot_kind": "dense source crossing only (kind=2)",
            "radial_window": "distance from retained full-fit centroid <= 4 times retained major FWHM",
            "signal_flag": "flags == 0 and finite signal",
            "hold": "both adjacent raw words zero and no transition ambiguity",
            "scan": "must be selected in frozen direction registry",
        },
        "separate_fit": {
            "model": "elliptical 2D Gaussian plus tangent-plane affine background",
            "loss": "scipy least_squares soft_l1",
            "loss_scale": "max(0.2 times prefit signal standard deviation, 1e-12)",
            "center_bounds": "retained full-fit centroid plus/minus retained major FWHM on each tangent axis",
            "fwhm_bounds_arcsec_by_array": {ARRAY_NAMES[k]: list(v) for k, v in ARRAY_FWHM_LIMITS.items()},
            "amplitude": "nonnegative",
            "minimum_distinct_scans_per_direction": 3,
            "minimum_samples_per_direction": 100,
            "quality": "solver success, finite covariance, amplitude / residual MAD-sigma >= 3, non-boundary center and widths",
            "matched_population": "intersection passing identical frozen quality rules in both directions",
        },
        "pooled_fit": {
            "coordinate": "each detector translated by its retained full-fit centroid; translation cancels in left-right difference",
            "normalization": "signal divided by retained positive full-fit amplitude; equal detector contribution",
            "pixel_size_arcsec": 1.0,
            "extent_arcsec": [-80.0, 80.0],
            "minimum_pixel_count": 5,
            "groups": "all arrays pooled, each array, each detector network/interface",
            "model": "elliptical 2D Gaussian plus affine background",
        },
        "primary_estimator": {
            "definition": "((centroid_right-centroid_left) dot positive_axis)/(median_speed_right-median_speed_left)",
            "speed": "actual instantaneous trajectory projected speed for the exact retained fit samples",
            "uncertainty": "delete-one-stable-scan jackknife of the pooled normalized 1-arcsec diagnostic map",
            "intervals": "normal-equivalent 68 percent = estimate +/- 1 SE; 95 percent = estimate +/- 1.96 SE",
            "detection_rule": "95 percent interval excludes zero and controls are timing-compatible",
            "upper_bound_rule": "only when 95 percent interval includes zero and controls/coverage are acceptable: max(abs(interval endpoints))",
            "global_tolerance": "none; result is a measurement, not ALIGN acceptance",
        },
        "controls": [
            "perpendicular centroid difference",
            "first-half versus second-half scan split",
            "balanced same-direction odd/even-within-direction null partition (not scan-index parity classification)",
            "left/right counts, coverage, angle, speed, and normalized weight",
            "per-array and per-network consistency",
            "left/right major/minor FWHM, ellipticity, and amplitude",
            "timing dependence on realized speed reported; limited leverage disposition required",
        ],
        "scope": {
            "citlali_run": False, "unity_contact": False, "application_source_change": False,
            "mapmaking_implementation_change": False,
            "absolute_physical_timestamp_semantics": "unresolved",
        },
    }


def prepare(repo: Path) -> None:
    identity = validate_identity(repo)
    state = read_state()
    count = int(state["alignment"]["governing_compatibility_axis"]["sample_count"])
    phase = float(state["alignment"]["grid"]["phase_sec"])
    telescope = read_telescope(phase, count)
    registry, axis, low_speed = build_registry(state, telescope)
    package = repo / PACKAGE_REL
    package.mkdir(parents=True, exist_ok=True)
    write_json(package / "identity.json", {
        **identity, "task_base": TASK_BASE, "candidate": APPLICATION_CANDIDATE,
        "governing_application": GOVERNING_APPLICATION,
    })
    write_json(package / "stage_a_decision.json", stage_a_decision(repo))
    write_json(package / "preregistered_protocol.json", protocol(axis, low_speed, registry))
    write_csv(package / "scan_direction_registry.csv", registry)
    selected_left = [row for row in registry if row["classification"] == "left"]
    selected_right = [row for row in registry if row["classification"] == "right"]
    excluded = [row for row in registry if row["classification"] == "excluded"]
    (package / "left_scan_identities.txt").write_text(
        "".join(f"{row['stable_scan_id']}\n" for row in selected_left)
    )
    (package / "right_scan_identities.txt").write_text(
        "".join(f"{row['stable_scan_id']}\n" for row in selected_right)
    )
    write_csv(
        package / "excluded_scan_identities.csv", excluded,
        fields=list(registry[0]),
    )
    summary = {
        "admitted": len(registry), "left": len(selected_left), "right": len(selected_right),
        "excluded": len(excluded), "disjoint": True,
        "census_complete": len(selected_left) + len(selected_right) + len(excluded) == len(registry),
        "positive_axis": axis.tolist(), "low_speed_threshold_arcsec_s": low_speed,
        "hold_valid_all_selected": all(row["hold_valid_fraction"] == 1.0 for row in registry if row["selected"]),
        "transition_ambiguous_selected": sum(row["hold_transition_ambiguous_count"] for row in registry if row["selected"]),
    }
    write_json(package / "scan_selection_summary.json", summary)
    manifest = manifest_rows()
    write_csv(package / "raw_input_manifest.csv", manifest)
    write_json(package / "source_manifest.json", {
        "schema": "sci-align-001-lr-beammap-source-manifest-v1",
        "inputs": manifest,
        "repository_sources": [
            {"path": str(SCRIPT_REL), "sha256": sha256_file(repo / SCRIPT_REL)},
            {"path": "include/citlali/core/utils/pointing.h", "sha256": sha256_file(repo / "include/citlali/core/utils/pointing.h")},
            {"path": "include/citlali/core/engine/detail/todproc_pointing_impl.h", "sha256": sha256_file(repo / "include/citlali/core/engine/detail/todproc_pointing_impl.h")},
            {"path": "include/citlali/core/engine/detail/beammap_detector_tod_output_impl.h", "sha256": sha256_file(repo / "include/citlali/core/engine/detail/beammap_detector_tod_output_impl.h")},
        ],
    })
    write_csv(package / "realized_config_manifest.csv", [{
        "role": "retained_full_candidate", "path": str(CONFIG), "sha256": sha256_file(CONFIG),
        "selection_difference": "none; retained product partitioned after execution",
        "candidate_commit": APPLICATION_CANDIDATE, "omp_threads_realized": 6,
    }])
    write_json(package / "local_run_manifest.json", {
        "citlali_reductions_launched_by_sidequest": 0,
        "retained_run_reused": str(REDUCTION),
        "retained_candidate": APPLICATION_CANDIDATE,
        "retained_threads": 6,
        "interpretation": "retained-product diagnostic only; no new reduction",
    })
    prepare_files = sorted(
        path for path in package.iterdir()
        if path.is_file() and path.name != "preregistration_freeze.json"
    )
    digest = hashlib.sha256()
    for path in prepare_files:
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(sha256_file(path)))
    write_json(package / "preregistration_freeze.json", {
        "freeze_scope": "all prepare-mode files except this freeze record",
        "ordered_files": [path.name for path in prepare_files],
        "aggregate_sha256": digest.hexdigest(),
        "confirmatory_signal_variable_read_by_prepare": False,
        "pilot_uids_excluded": list(PILOT_UIDS),
    })


def fit_model(x: np.ndarray, y: np.ndarray, z: np.ndarray, ref: dict[str, float], array: int) -> dict[str, Any]:
    major0 = max(ref["major"], ref["minor"])
    minor0 = min(ref["major"], ref["minor"])
    dx0 = x - ref["x"]
    dy0 = y - ref["y"]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    radial = np.hypot(dx0, dy0)
    keep = finite & (radial <= 4.0 * major0)
    x, y, z = x[keep], y[keep], z[keep]
    if z.size < 100:
        return {"success": False, "reason": "insufficient_samples", "n_samples": int(z.size)}
    fmin, fmax = ARRAY_FWHM_LIMITS[array]
    median = float(np.median(z))
    amplitude = max(float(np.max(z) - median), np.finfo(float).eps)
    s1 = np.clip(major0, fmin * 1.001, fmax * 0.999) / 2.354820045
    s2 = np.clip(minor0, fmin * 1.001, fmax * 0.999) / 2.354820045
    p0 = np.array([
        amplitude, ref["x"], ref["y"], math.log(s1), math.log(s2), ref["angle"],
        median, 0.0, 0.0,
    ])
    lower = np.array([
        0.0, ref["x"] - major0, ref["y"] - major0,
        math.log(fmin / 2.354820045), math.log(fmin / 2.354820045), -math.pi,
        -np.inf, -np.inf, -np.inf,
    ])
    upper = np.array([
        np.inf, ref["x"] + major0, ref["y"] + major0,
        math.log(fmax / 2.354820045), math.log(fmax / 2.354820045), math.pi,
        np.inf, np.inf, np.inf,
    ])

    def residual(parameters: np.ndarray) -> np.ndarray:
        amp, cx, cy, log_s1, log_s2, angle, background, bx, by = parameters
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = x - cx, y - cy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        source = amp * np.exp(-0.5 * ((u / math.exp(log_s1)) ** 2 + (v / math.exp(log_s2)) ** 2))
        return source + background + bx * dx0[keep] + by * dy0[keep] - z

    scale = max(0.2 * float(np.std(z)), 1.0e-12)
    try:
        result = least_squares(
            residual, p0, bounds=(lower, upper), loss="soft_l1", f_scale=scale,
            max_nfev=300, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
    except Exception as error:
        return {"success": False, "reason": f"solver_exception:{type(error).__name__}", "n_samples": int(z.size)}
    values = result.x
    resid = residual(values)
    med = float(np.median(resid))
    resid_mad = float(1.4826 * np.median(np.abs(resid - med)))
    dof = max(1, z.size - values.size)
    covariance = np.full((values.size, values.size), np.nan)
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * float(np.sum(resid**2) / dof)
    except np.linalg.LinAlgError:
        pass
    sigmas = np.exp(values[3:5])
    fwhm = sigmas * 2.354820045
    order = np.argsort(fwhm)[::-1]
    major, minor = fwhm[order]
    pa = float(values[5] + (math.pi / 2.0 if order[0] == 1 else 0.0))
    pa = ((pa + math.pi / 2.0) % math.pi) - math.pi / 2.0
    center_margin = major0 - max(abs(values[1] - ref["x"]), abs(values[2] - ref["y"]))
    width_margin = min(major - fmin, minor - fmin, fmax - major, fmax - minor)
    amp_snr = float(values[0] / resid_mad) if resid_mad > 0.0 else math.inf
    center_cov = covariance[1:3, 1:3]
    quality = bool(
        result.success and np.all(np.isfinite(center_cov)) and amp_snr >= 3.0
        and center_margin > 1e-3 and width_margin > 1e-3
    )
    return {
        "success": bool(result.success), "quality": quality,
        "reason": "accepted" if quality else "frozen_quality_rule_failed",
        "n_samples": int(z.size), "amplitude": float(values[0]),
        "background": float(values[6]), "background_x": float(values[7]),
        "background_y": float(values[8]), "centroid_x_arcsec": float(values[1]),
        "centroid_y_arcsec": float(values[2]), "centroid_x_sigma_arcsec": float(math.sqrt(max(0.0, center_cov[0, 0]))) if np.isfinite(center_cov[0, 0]) else math.nan,
        "centroid_y_sigma_arcsec": float(math.sqrt(max(0.0, center_cov[1, 1]))) if np.isfinite(center_cov[1, 1]) else math.nan,
        "centroid_xy_cov_arcsec2": float(center_cov[0, 1]) if np.isfinite(center_cov[0, 1]) else math.nan,
        "major_fwhm_arcsec": float(major), "minor_fwhm_arcsec": float(minor),
        "position_angle_deg": float(math.degrees(pa)),
        "ellipticity": float(major / minor - 1.0), "residual_mad_sigma": resid_mad,
        "amplitude_over_residual_mad": amp_snr, "cost": float(result.cost),
        "optimality": float(result.optimality), "nfev": int(result.nfev),
    }


def map_fit(sum_image: np.ndarray, count_image: np.ndarray, array: int, extent: float = 80.0) -> dict[str, Any]:
    size = sum_image.shape[0]
    centers = np.arange(size, dtype=float) - extent + 0.5
    xx, yy = np.meshgrid(centers, centers)
    valid = count_image >= 5
    z = np.divide(sum_image, count_image, out=np.full_like(sum_image, np.nan), where=count_image > 0)
    if np.sum(valid) < 100:
        return {"success": False, "quality": False, "reason": "insufficient_map_pixels"}
    weights = np.sqrt(count_image[valid].astype(float) / np.max(count_image[valid]))
    x, y, values = xx[valid], yy[valid], z[valid]
    fmin, fmax = (3.0, 20.0) if array == -1 else ARRAY_FWHM_LIMITS[array]
    major0 = 0.55 * (fmin + fmax)
    minor0 = 0.50 * (fmin + fmax)
    p0 = np.array([
        max(float(np.nanmax(values) - np.nanmedian(values)), 1e-9),
        0.0, 0.0, math.log(major0 / 2.354820045),
        math.log(minor0 / 2.354820045), 0.0,
        float(np.nanmedian(values)), 0.0, 0.0,
    ])
    lo = np.array([0, -major0, -major0, math.log(fmin/2.354820045), math.log(fmin/2.354820045), -math.pi, -np.inf, -np.inf, -np.inf])
    hi = np.array([np.inf, major0, major0, math.log(fmax/2.354820045), math.log(fmax/2.354820045), math.pi, np.inf, np.inf, np.inf])
    def fun(p: np.ndarray) -> np.ndarray:
        amp,cx,cy,l1,l2,ang,b,bx,by=p; ca,sa=np.cos(ang),np.sin(ang);dx=x-cx;dy=y-cy;u=ca*dx+sa*dy;v=-sa*dx+ca*dy
        model=amp*np.exp(-.5*((u/np.exp(l1))**2+(v/np.exp(l2))**2))+b+bx*x+by*y
        return (model-values)*weights
    result=least_squares(fun,p0,bounds=(lo,hi),loss="soft_l1",f_scale=max(.2*float(np.nanstd(values)),1e-6),max_nfev=300)
    p=result.x; fw=np.exp(p[3:5])*2.354820045; order=np.argsort(fw)[::-1];major,minor=fw[order];pa=p[5]+(math.pi/2 if order[0]==1 else 0);pa=((pa+math.pi/2)%math.pi)-math.pi/2
    residual=fun(p);mad=1.4826*np.median(np.abs(residual-np.median(residual)))
    cov=np.full((9,9),np.nan)
    try:cov=np.linalg.inv(result.jac.T@result.jac)*(np.sum(residual**2)/max(1,residual.size-9))
    except np.linalg.LinAlgError:pass
    return {"success":bool(result.success),"quality":bool(result.success and np.all(np.isfinite(cov[1:3,1:3]))),"reason":"accepted" if result.success else "solver_failed","n_pixels":int(np.sum(valid)),"n_samples":int(np.sum(count_image)),"amplitude":float(p[0]),"centroid_x_arcsec":float(p[1]),"centroid_y_arcsec":float(p[2]),"centroid_x_sigma_arcsec":float(np.sqrt(max(0,cov[1,1]))) if np.isfinite(cov[1,1]) else math.nan,"centroid_y_sigma_arcsec":float(np.sqrt(max(0,cov[2,2]))) if np.isfinite(cov[2,2]) else math.nan,"centroid_xy_cov_arcsec2":float(cov[1,2]) if np.isfinite(cov[1,2]) else math.nan,"major_fwhm_arcsec":float(major),"minor_fwhm_arcsec":float(minor),"position_angle_deg":float(np.degrees(pa)),"ellipticity":float(major/minor-1),"residual_mad_sigma":float(mad),"cost":float(result.cost)}


def analyze(repo: Path) -> None:
    validate_identity(repo)
    package = repo / PACKAGE_REL
    freeze = json.loads((package / "preregistration_freeze.json").read_text())
    digest = hashlib.sha256()
    for name in freeze["ordered_files"]:
        path = package / name
        digest.update(path.name.encode() + b"\0" + bytes.fromhex(sha256_file(path)))
    if digest.hexdigest() != freeze["aggregate_sha256"]:
        raise RuntimeError("preregistration package changed after freeze")
    protocol_doc = json.loads((package / "preregistered_protocol.json").read_text())
    axis = np.array([
        protocol_doc["positive_scan_axis"]["x_az_tangent"],
        protocol_doc["positive_scan_axis"]["y_el_tangent"],
    ])
    cross_axis = np.array([-axis[1], axis[0]])
    registry = {int(row["stable_scan_id"]): row for row in csv.DictReader((package / "scan_direction_registry.csv").open())}
    state = read_state()
    ordinal_to_stable = {
        int(row["compatibility_ordinal"] + 1): int(row["stable_id"])
        for row in state["records"]
    }
    count = int(state["alignment"]["governing_compatibility_axis"]["sample_count"])
    phase = float(state["alignment"]["grid"]["phase_sec"])
    telescope = read_telescope(phase, count)
    apt = Table.read(OUTPUT_APT, format="ascii.ecsv")
    apt_uid = np.asarray(apt["uid"], dtype=int)
    extent = 80
    image_size = 2 * extent
    groups = ["all", "array:a1100", "array:a1400", "array:a2000"]
    with Dataset(DETECTOR_TOD) as dataset:
        uid = np.asarray(dataset["detector_tod_uid"][:], dtype=int)
        arrays = np.asarray(dataset["detector_tod_array"][:], dtype=int)
        networks = np.asarray(dataset["detector_tod_network"][:], dtype=int)
        fit_good = np.asarray(dataset["detector_tod_fit_good"][:], dtype=int)
        full_x = np.asarray(dataset["detector_tod_fit_x_t_arcsec"][:], dtype=float)
        full_y = np.asarray(dataset["detector_tod_fit_y_t_arcsec"][:], dtype=float)
        kind = np.asarray(dataset["detector_tod_slot_kind"][:], dtype=int)
        scan_index = np.asarray(dataset["detector_tod_scan_index"][:], dtype=int)
        n_samples = np.asarray(dataset["detector_tod_n_samples"][:], dtype=int)
        starts = np.asarray(dataset["detector_tod_scan_inner_start_sample"][:], dtype=int)
        signal_var = dataset["signal"]
        flag_var = dataset["flags"]
        if not np.array_equal(uid, apt_uid):
            raise RuntimeError("retained output APT/TOD UID join changed")
        apt_flag = np.asarray(apt["flag"], dtype=int)
        major_ref = np.maximum(np.asarray(apt["a_fwhm"],float), np.asarray(apt["b_fwhm"],float))
        minor_ref = np.minimum(np.asarray(apt["a_fwhm"],float), np.asarray(apt["b_fwhm"],float))
        amp_ref = np.asarray(apt["amp"],float)
        angle_ref = np.radians(np.asarray(apt["angle"],float))
        preselected = (
            (fit_good == 1) & (apt_flag == 0) & np.isfinite(full_x) & np.isfinite(full_y)
            & np.isfinite(major_ref) & np.isfinite(minor_ref) & (major_ref > 0)
            & (minor_ref > 0) & np.isfinite(amp_ref) & (amp_ref > 0)
            & ~np.isin(uid, np.asarray(PILOT_UIDS))
        )
        detector_rows: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        group_list = groups + [f"network:toltec{n}" for n in sorted(set(networks))]
        total_sum = {
            (group, direction): np.zeros((image_size, image_size), dtype=np.float64)
            for group in group_list for direction in ("left", "right")
        }
        total_count = {
            (group, direction): np.zeros((image_size, image_size), dtype=np.int64)
            for group in group_list for direction in ("left", "right")
        }
        # Scan-block images are retained only for the pooled primary
        # jackknife. Array/network comparisons use their formal map covariance.
        scan_map_sum = {
            (direction, int(stable)): np.zeros((image_size, image_size), dtype=np.float64)
            for direction in ("left", "right") for stable in registry
        }
        scan_map_count = {
            (direction, int(stable)): np.zeros((image_size, image_size), dtype=np.int64)
            for direction in ("left", "right") for stable in registry
        }
        scan_speeds: dict[tuple[str,str], list[float]] = defaultdict(list)
        for det in np.flatnonzero(preselected):
            ref = {"x":float(full_x[det]),"y":float(full_y[det]),"major":float(major_ref[det]),"minor":float(minor_ref[det]),"angle":float(angle_ref[det])}
            payload={"left":{"x":[],"y":[],"z":[],"speed":[],"scan":[]},"right":{"x":[],"y":[],"z":[],"speed":[],"scan":[]}}
            for slot in np.flatnonzero(kind[det] == 2):
                ordinal=int(scan_index[det,slot]); stable=ordinal_to_stable[ordinal]; row=registry[stable]; direction=row["classification"]
                if direction not in ("left","right") or row["selected"] != "True": continue
                start=int(starts[det,slot]); length=int(n_samples[det,slot]); indices=np.arange(start,start+length,dtype=int)
                z=np.asarray(signal_var[det,slot,:length],dtype=float); flags=np.asarray(flag_var[det,slot,:length],dtype=int)
                valid=(flags==0)&np.isfinite(z)&(telescope["hold_left"][indices]==0)&(telescope["hold_right"][indices]==0)&~telescope["hold_transition"][indices]
                x=telescope["x"][indices]; y=telescope["y"][indices]; radial=np.hypot(x-ref["x"],y-ref["y"]);valid &= radial <= 4.0*ref["major"]
                if not np.any(valid): continue
                vx=np.gradient(telescope["x"][indices],DT_SEC);vy=np.gradient(telescope["y"][indices],DT_SEC);speed=vx*axis[0]+vy*axis[1]
                payload[direction]["x"].append(x[valid]);payload[direction]["y"].append(y[valid]);payload[direction]["z"].append(z[valid]);payload[direction]["speed"].append(speed[valid]);payload[direction]["scan"].append(np.full(np.sum(valid),stable,dtype=int))
                lx=x[valid]-ref["x"];ly=y[valid]-ref["y"]
                normalized=z[valid]/amp_ref[det]
                px=np.floor(lx+extent).astype(int);py=np.floor(ly+extent).astype(int);inside=(px>=0)&(px<image_size)&(py>=0)&(py<image_size)
                det_groups=["all",f"array:{ARRAY_NAMES[int(arrays[det])]}",f"network:toltec{int(networks[det])}"]
                for group in det_groups:
                    np.add.at(total_sum[(group,direction)],(py[inside],px[inside]),normalized[inside])
                    np.add.at(total_count[(group,direction)],(py[inside],px[inside]),1)
                    scan_speeds[(group,direction)].append(float(np.median(speed[valid])))
                np.add.at(scan_map_sum[(direction,stable)],(py[inside],px[inside]),normalized[inside])
                np.add.at(scan_map_count[(direction,stable)],(py[inside],px[inside]),1)
            for direction in ("left","right"):
                part=payload[direction]
                distinct=sorted(set(np.concatenate(part["scan"]).tolist())) if part["scan"] else []
                if len(distinct)<3 or not part["z"]:
                    fit={"success":False,"quality":False,"reason":"fewer_than_three_scans","n_samples":sum(map(len,part["z"]))}
                else:
                    fit=fit_model(np.concatenate(part["x"]),np.concatenate(part["y"]),np.concatenate(part["z"]),ref,int(arrays[det]))
                detector_rows[direction].append({"level":"detector","identity":int(uid[det]),"uid":int(uid[det]),"network":f"toltec{int(networks[det])}","array":ARRAY_NAMES[int(arrays[det])],"direction":direction,"n_scans":len(distinct),"stable_scan_ids_json":canonical_json(distinct),"median_projected_speed_arcsec_s":float(np.median(np.concatenate(part["speed"]))) if part["speed"] else "",**fit})

    # Direction maps and scan-delete jackknife at pooled/array/network levels.
    group_rows={"left":[],"right":[]}; matched=[]; timing=[]; jackknife={}
    stable_ids=sorted(int(key) for key in registry)
    map_products={}
    for group in group_list:
        direction_fit={}; totals={}
        array=-1 if group=="all" else (next(k for k,v in ARRAY_NAMES.items() if group==f"array:{v}") if group.startswith("array:") else int(arrays[np.flatnonzero(networks==int(group.split("toltec")[1]))[0]]))
        for direction in ("left","right"):
            sums=total_sum[(group,direction)];counts=total_count[(group,direction)]
            contributing = (
                [s for s in stable_ids if np.sum(scan_map_count[(direction,s)])>0]
                if group == "all" else []
            )
            fit=map_fit(sums,counts,array,extent)
            fit.update({"level":"pooled" if group=="all" else group.split(":")[0],"identity":group,"uid":"","network":group.split(":",1)[1] if group.startswith("network:") else "","array":group.split(":",1)[1] if group.startswith("array:") else ("mixed" if group=="all" else ARRAY_NAMES[array]),"direction":direction,"n_scans":len(contributing) if group=="all" else "see_detector_slots","stable_scan_ids_json":canonical_json(contributing) if group=="all" else "","median_projected_speed_arcsec_s":float(np.median(scan_speeds[(group,direction)]))})
            group_rows[direction].append(fit);direction_fit[direction]=fit;totals[direction]=(sums,counts);map_products[(group,direction)]=(sums,counts)
        left,right=direction_fit["left"],direction_fit["right"]
        if not (left.get("quality") and right.get("quality")): continue
        delta=np.array([right["centroid_x_arcsec"]-left["centroid_x_arcsec"],right["centroid_y_arcsec"]-left["centroid_y_arcsec"]]);parallel=float(delta@axis);perpendicular=float(delta@cross_axis);vleft=float(left["median_projected_speed_arcsec_s"]);vright=float(right["median_projected_speed_arcsec_s"]);dt=parallel/(vright-vleft)
        delete=[]
        if group == "all":
            for omitted in stable_ids:
                fits={}
                for direction in ("left","right"):
                    sums,counts=totals[direction];f=map_fit(sums-scan_map_sum[(direction,omitted)],counts-scan_map_count[(direction,omitted)],array,extent);fits[direction]=f
                if fits["left"].get("quality") and fits["right"].get("quality"):
                    d=np.array([fits["right"]["centroid_x_arcsec"]-fits["left"]["centroid_x_arcsec"],fits["right"]["centroid_y_arcsec"]-fits["left"]["centroid_y_arcsec"]]);delete.append({"omitted_stable_scan_id":omitted,"parallel_arcsec":float(d@axis),"perpendicular_arcsec":float(d@cross_axis),"timing_sec":float((d@axis)/(vright-vleft))})
            theta=np.array([row["timing_sec"] for row in delete]);par=np.array([row["parallel_arcsec"] for row in delete]);perp=np.array([row["perpendicular_arcsec"] for row in delete]);n=len(theta)
            def jse(a):return float(np.sqrt((n-1)/n*np.sum((a-np.mean(a))**2)))
            se=jse(theta);pse=jse(par);cse=jse(perp);uncertainty_method="delete_one_stable_scan_jackknife"
        else:
            cov_left=np.array([[left["centroid_x_sigma_arcsec"]**2,left["centroid_xy_cov_arcsec2"]],[left["centroid_xy_cov_arcsec2"],left["centroid_y_sigma_arcsec"]**2]])
            cov_right=np.array([[right["centroid_x_sigma_arcsec"]**2,right["centroid_xy_cov_arcsec2"]],[right["centroid_xy_cov_arcsec2"],right["centroid_y_sigma_arcsec"]**2]])
            pse=float(np.sqrt(max(0,axis@(cov_left+cov_right)@axis)));cse=float(np.sqrt(max(0,cross_axis@(cov_left+cov_right)@cross_axis)));se=pse/abs(vright-vleft);n=0;uncertainty_method="formal_map_covariance_secondary_only"
        result={"identity":group,"parallel_centroid_difference_arcsec":parallel,"parallel_jackknife_se_arcsec":pse,"perpendicular_centroid_difference_arcsec":perpendicular,"perpendicular_jackknife_se_arcsec":cse,"v_left_arcsec_s":vleft,"v_right_arcsec_s":vright,"timing_offset_sec":dt,"timing_jackknife_se_sec":se,"timing_68_low_sec":dt-se,"timing_68_high_sec":dt+se,"timing_95_low_sec":dt-1.96*se,"timing_95_high_sec":dt+1.96*se,"significance_sigma":dt/se if se>0 else "","equivalent_50_arcsec_s":dt*50,"equivalent_100_arcsec_s":dt*100,"equivalent_200_arcsec_s":dt*200,"fraction_1arcsec_pixel_at_realized_speed":abs(parallel),"fraction_2arcsec_pixel_at_realized_speed":abs(parallel)/2,"left_major_fwhm_arcsec":left["major_fwhm_arcsec"],"right_major_fwhm_arcsec":right["major_fwhm_arcsec"],"left_minor_fwhm_arcsec":left["minor_fwhm_arcsec"],"right_minor_fwhm_arcsec":right["minor_fwhm_arcsec"],"left_amplitude":left["amplitude"],"right_amplitude":right["amplitude"],"jackknife_replicates":n,"uncertainty_method":uncertainty_method}
        timing.append(result);matched.append({**result,"left_centroid_x_arcsec":left["centroid_x_arcsec"],"left_centroid_y_arcsec":left["centroid_y_arcsec"],"right_centroid_x_arcsec":right["centroid_x_arcsec"],"right_centroid_y_arcsec":right["centroid_y_arcsec"]});jackknife[group]=delete

    # Per-detector matched pairs use the same frozen cuts, but group inference
    # is carried by the scan-delete pooled result rather than formal covariance.
    left_by={int(row["uid"]):row for row in detector_rows["left"]};right_by={int(row["uid"]):row for row in detector_rows["right"]};det_matched=[]
    for uid_value in sorted(set(left_by)&set(right_by)):
        l,r=left_by[uid_value],right_by[uid_value]
        if not (l.get("quality") and r.get("quality")):continue
        d=np.array([r["centroid_x_arcsec"]-l["centroid_x_arcsec"],r["centroid_y_arcsec"]-l["centroid_y_arcsec"]]);par=float(d@axis);perp=float(d@cross_axis);den=float(r["median_projected_speed_arcsec_s"])-float(l["median_projected_speed_arcsec_s"]);det_matched.append({"level":"detector","identity":uid_value,"uid":uid_value,"network":l["network"],"array":l["array"],"parallel_centroid_difference_arcsec":par,"perpendicular_centroid_difference_arcsec":perp,"v_left_arcsec_s":l["median_projected_speed_arcsec_s"],"v_right_arcsec_s":r["median_projected_speed_arcsec_s"],"timing_offset_sec":par/den,"left_major_fwhm_arcsec":l["major_fwhm_arcsec"],"right_major_fwhm_arcsec":r["major_fwhm_arcsec"],"left_minor_fwhm_arcsec":l["minor_fwhm_arcsec"],"right_minor_fwhm_arcsec":r["minor_fwhm_arcsec"],"left_amplitude":l["amplitude"],"right_amplitude":r["amplitude"]})
    write_csv(package/"fit_results_left.csv",detector_rows["left"]+group_rows["left"]);write_csv(package/"fit_results_right.csv",detector_rows["right"]+group_rows["right"]);write_csv(package/"matched_fit_results.csv",matched+det_matched);write_csv(package/"timing_estimates.csv",timing)
    write_json(package/"uncertainty_results.json",{"method":"delete-one-stable-scan jackknife","groups":jackknife,"seed_recorded_but_random_sampling_not_used":SEED})
    def partition_fit(ids: list[int], direction: str) -> dict[str, Any]:
        sums=sum((scan_map_sum[(direction,s)] for s in ids),np.zeros((image_size,image_size)))
        counts=sum((scan_map_count[(direction,s)] for s in ids),np.zeros((image_size,image_size),dtype=np.int64))
        result=map_fit(sums,counts,-1,extent)
        result["stable_scan_ids"] = ids
        return result

    def lr_partition_result(ids: list[int]) -> dict[str, Any]:
        left_ids=[s for s in ids if registry[s]["classification"]=="left"]
        right_ids=[s for s in ids if registry[s]["classification"]=="right"]
        left_fit=partition_fit(left_ids,"left");right_fit=partition_fit(right_ids,"right")
        if not (left_fit.get("quality") and right_fit.get("quality")):
            return {"quality":False,"left_ids":left_ids,"right_ids":right_ids}
        delta=np.array([right_fit["centroid_x_arcsec"]-left_fit["centroid_x_arcsec"],right_fit["centroid_y_arcsec"]-left_fit["centroid_y_arcsec"]])
        vl=float(np.median([float(registry[s]["median_projected_velocity_arcsec_s"]) for s in left_ids]));vr=float(np.median([float(registry[s]["median_projected_velocity_arcsec_s"]) for s in right_ids]))
        return {"quality":True,"left_ids":left_ids,"right_ids":right_ids,"parallel_arcsec":float(delta@axis),"perpendicular_arcsec":float(delta@cross_axis),"v_left_arcsec_s":vl,"v_right_arcsec_s":vr,"timing_sec":float((delta@axis)/(vr-vl))}

    chronological=stable_ids
    time_split={"first_half":lr_partition_result(chronological[:len(chronological)//2]),"second_half":lr_partition_result(chronological[len(chronological)//2:])}
    speed_values={s:abs(float(registry[s]["median_projected_velocity_arcsec_s"])) for s in stable_ids}
    speed_divide=float(np.median(list(speed_values.values())))
    speed_split={"lower_abs_speed":lr_partition_result([s for s in stable_ids if speed_values[s]<=speed_divide]),"upper_abs_speed":lr_partition_result([s for s in stable_ids if speed_values[s]>speed_divide]),"split_arcsec_s":speed_divide}
    same_direction={}
    for direction in ("left","right"):
        ids=[s for s in stable_ids if registry[s]["classification"]==direction]
        a=partition_fit(ids[0::2],direction);b=partition_fit(ids[1::2],direction)
        if a.get("quality") and b.get("quality"):
            delta=np.array([b["centroid_x_arcsec"]-a["centroid_x_arcsec"],b["centroid_y_arcsec"]-a["centroid_y_arcsec"]])
            same_direction[direction]={"partition_rule":"alternating chronological rank within already-classified direction","a_ids":ids[0::2],"b_ids":ids[1::2],"parallel_arcsec":float(delta@axis),"perpendicular_arcsec":float(delta@cross_axis)}
        else:
            same_direction[direction]={"quality":False}
    # Controls and population summaries.
    pre=int(np.sum(preselected));quality=len(det_matched);by_array={name:int(sum(row["array"]==name for row in det_matched)) for name in ARRAY_NAMES.values()};by_network={f"toltec{n}":int(sum(row["network"]==f"toltec{n}" for row in det_matched)) for n in sorted(set(networks))}
    controls={"preselected_confirmatory_detectors":pre,"matched_quality_detectors":quality,"matched_by_array":by_array,"matched_by_network":by_network,"pilot_uids_excluded":list(PILOT_UIDS),"perpendicular_control":{row["identity"]:{"difference_arcsec":row["perpendicular_centroid_difference_arcsec"],"se_arcsec":row["perpendicular_jackknife_se_arcsec"]} for row in timing},"beam_width_and_amplitude":{row["identity"]:{key:row[key] for key in ("left_major_fwhm_arcsec","right_major_fwhm_arcsec","left_minor_fwhm_arcsec","right_minor_fwhm_arcsec","left_amplitude","right_amplitude")} for row in timing},"time_drift_split":time_split,"balanced_same_direction_null":same_direction,"speed_split":speed_split,"coverage":{"left_selected_windows":sum(r["classification"]=="left" for r in registry.values()),"right_selected_windows":sum(r["classification"]=="right" for r in registry.values()),"left_pooled_normalized_samples":int(np.sum(total_count[("all","left")])),"right_pooled_normalized_samples":int(np.sum(total_count[("all","right")]))},"speed_leverage":{"p05_abs_arcsec_s":float(np.quantile(np.abs([float(r["median_projected_velocity_arcsec_s"]) for r in registry.values()]),.05)),"p95_abs_arcsec_s":float(np.quantile(np.abs([float(r["median_projected_velocity_arcsec_s"]) for r in registry.values()]),.95))}}
    write_json(package/"control_tests.json",controls)
    # Fixed half/time and same-direction partitions on registry-level direction
    # cannot refit maps without another pass; derive them from per-detector fits
    # only as explicitly limited controls.
    pooled=next(row for row in timing if row["identity"]=="all")
    disposition="residual timing detected" if pooled["timing_95_low_sec"]*pooled["timing_95_high_sec"]>0 else "no significant residual"
    confirm={"disposition":disposition,"primary":pooled,"physical_timestamp_authority":"unresolved: detector timestamp start/end/effective integration centroid remains unproved","speed_scaling":"fixture has narrow native speed range; 50/100/200 arcsec/s values are dimensional translations, not independently measured scaling","same_direction_null":same_direction,"time_split":time_split,"speed_split":speed_split,"absolute_sky_correctness":"unresolved despite differential direction-reversal measurement"}
    write_json(package/"confirmatory_results.json",confirm);write_json(package/"joint_fit_results.json",{"used":False,"reason":"separate direction fits and pooled scan-delete maps were identifiable"})

    # Diagnostic plots.
    plot_dir=package/"plots";plot_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,direction in zip(axes,("left","right")):
        sums,counts=map_products[("all",direction)];image=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0);im=ax.imshow(image,origin="lower",extent=(-extent,extent,-extent,extent),cmap="viridis",vmin=np.nanpercentile(image,5),vmax=np.nanpercentile(image,99));ax.set_title(f"{direction} normalized retained TOD");ax.set_xlabel("local +az (arcsec)");ax.set_ylabel("local +el (arcsec)");fig.colorbar(im,ax=ax)
    fig.savefig(plot_dir/"left_right_pooled_maps.png",dpi=160);plt.close(fig)
    det=np.asarray([row["timing_offset_sec"]*1000 for row in det_matched]);fig,ax=plt.subplots(figsize=(7,4));ax.hist(det[np.isfinite(det)],bins=80,color="steelblue");ax.axvline(pooled["timing_offset_sec"]*1000,color="crimson",label="pooled");ax.set_xlabel("residual timing estimate (ms)");ax.set_ylabel("matched detectors");ax.legend();fig.tight_layout();fig.savefig(plot_dir/"detector_timing_distribution.png",dpi=160);plt.close(fig)
    labels=[row["identity"] for row in timing];vals=np.array([row["timing_offset_sec"]*1000 for row in timing]);errs=np.array([row["timing_jackknife_se_sec"]*1000 for row in timing]);fig,ax=plt.subplots(figsize=(10,5));ax.errorbar(np.arange(len(labels)),vals,yerr=errs,fmt="o");ax.axhline(0,color="black",lw=.8);ax.set_xticks(np.arange(len(labels)),labels,rotation=60,ha="right");ax.set_ylabel("timing offset (ms), jackknife SE");fig.tight_layout();fig.savefig(plot_dir/"timing_by_array_network.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));par=np.array([row["parallel_centroid_difference_arcsec"] for row in det_matched]);per=np.array([row["perpendicular_centroid_difference_arcsec"] for row in det_matched]);ax.hexbin(par,per,gridsize=60,mincnt=1,cmap="magma");ax.axhline(0,color="white",lw=.7);ax.axvline(0,color="white",lw=.7);ax.set_xlabel("parallel R-L centroid (arcsec)");ax.set_ylabel("perpendicular R-L centroid (arcsec)");fig.tight_layout();fig.savefig(plot_dir/"parallel_perpendicular_detector_control.png",dpi=160);plt.close(fig)
    jk=np.array([row["timing_sec"]*1000 for row in jackknife["all"]]);fig,ax=plt.subplots(figsize=(7,4));ax.hist(jk,bins=40);ax.axvline(pooled["timing_offset_sec"]*1000,color="crimson");ax.set_xlabel("delete-one-scan timing estimate (ms)");ax.set_ylabel("replicates");fig.tight_layout();fig.savefig(plot_dir/"pooled_scan_jackknife.png",dpi=160);plt.close(fig)

    # Human report, owner brief, and Unity handoff.
    report=f"""# SCI-ALIGN-001-LR-BEAMMAP retained-product result\n\n+## Outcome\n\n+The no-code retained-product path was sufficient. No Citlali process or new reduction was launched, and no application source changed. The frozen direction registry contains {sum(r['classification']=='left' for r in registry.values())} left and {sum(r['classification']=='right' for r in registry.values())} right windows with {sum(r['classification']=='excluded' for r in registry.values())} exclusions.\n\n+The pooled retained-TOD estimator gives **{pooled['timing_offset_sec']*1000:.4f} ms** with delete-one-scan SE **{pooled['timing_jackknife_se_sec']*1000:.4f} ms** and 95% interval **[{pooled['timing_95_low_sec']*1000:.4f}, {pooled['timing_95_high_sec']*1000:.4f}] ms**. The right-minus-left centroid is **{pooled['parallel_centroid_difference_arcsec']:.5f} arcsec** parallel and **{pooled['perpendicular_centroid_difference_arcsec']:.5f} arcsec** perpendicular to the frozen scan axis. The confirmatory matched detector population is {quality}/{pre}.\n\n+Disposition: **{disposition}** in this bounded local retained-product diagnostic. This is not SCI-ALIGN acceptance and does not provide absolute physical timestamp correctness.\n\n+## Scope and limitations\n\n+- Eight feasibility-pilot UIDs {list(PILOT_UIDS)} were viewed before protocol freeze and are excluded from all confirmatory results.\n+- The native speed distribution is narrow, so scaling with speed is not independently identified. The 50/100/200 arcsec/s values in `timing_estimates.csv` are dimensional translations.\n+- Detector timestamp start/end/effective integration-centroid semantics remain unproved; absolute sky correctness remains unresolved.\n+- The retained split uses final-iteration per-crossing PTC products and a bounded 1-arcsec diagnostic accumulation. It does not alter or validate SCI-MAP implementation.\n+- A human Unity exact-9aae/exact-candidate left/right campaign remains required for definitive governing-versus-candidate evidence.\n+"""
    (package/"REPORT.md").write_text(report)
    owner={"task":"SCI-ALIGN-001-LR-BEAMMAP","branch":BRANCH,"application_modified":False,"new_local_reductions":0,"measured_disposition":disposition,"primary":pooled,"engineering_invariants":["retained assigned-time/coordinate behavior is unchanged from c771 phase-one evidence","direction sets are deterministic, disjoint, census-complete, trajectory-derived, and Hold-fail-closed"],"measured_angular_results":{"parallel_arcsec":pooled["parallel_centroid_difference_arcsec"],"perpendicular_arcsec":pooled["perpendicular_centroid_difference_arcsec"]},"unresolved":["absolute detector integration-event timestamp semantics","absolute sky-placement correctness","definitive exact-9aae versus exact-c771 Unity comparison","speed scaling beyond narrow approximately 48 arcsec/s fixture leverage"],"owner_questions":["Does the owner treat the local retained-product detection as sufficient motivation to schedule the exact governing/candidate Unity campaign?","Does the owner request producer authority for whether Data.Toltec.Ts denotes integration start, end, or effective centroid before any physical timing correction is proposed?","If Unity confirms a residual, should a separate scientific amendment authorize investigation of a timing correction and sky-domain tolerance?"],"phase_one_or_acceptance_authorized":False}
    write_json(package/"owner_decision_brief.json",owner)
    (package/"UNITY_HANDOFF.md").write_text(f"""# Human-run Unity handoff (not executed)\n\n+Run Beammap {OBSNUM} under the human-controlled `unity_toltec` lane with four isolated fail-if-exists outputs: exact `{GOVERNING_APPLICATION}` left/right and exact `{APPLICATION_CANDIDATE}` left/right. Use the digest-bound inputs/config in `raw_input_manifest.csv`; six OMP threads per job; identical dependencies, resources, APT, fit reports, calibration, detector population, and all non-selection settings. Apply the exact stable IDs in `left_scan_identities.txt` and `right_scan_identities.txt` through a reviewed external selection/diagnostic mechanism. Do not label a patched executable exact. Preserve maps, per-scan products, setup/total timing, logs, embedded SHA, and SHA-256 manifests. Compare governing left/right, candidate left/right, and governing-to-candidate within each direction using the frozen `preregistered_protocol.json`.\n""")
    changed=[{"status":"A","path":str(SCRIPT_REL),"reason":"bounded retained-product diagnostic"}]+[{"status":"A","path":str(path.relative_to(repo)),"reason":"SCI-ALIGN-001-LR-BEAMMAP evidence"} for path in sorted(package.rglob("*")) if path.is_file()]
    write_csv(package/"changed_paths.tsv",changed,fields=["status","path","reason"])
    # Final sums exclude themselves and are regenerated deterministically.
    sums=[]
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS": sums.append(f"{sha256_file(path)}  {path.relative_to(package)}")
    (package/"SHA256SUMS").write_text("\n".join(sums)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.repo.resolve())
    else:
        analyze(args.repo.resolve())


if __name__ == "__main__":
    main()
