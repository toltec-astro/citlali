#!/usr/bin/env python3
"""Freeze the SCI-ALIGN-001 sample-lineage/phase diagnostic.

This preparation step reads source, compact provenance, NetCDF headers, raw
timestamp matrices, and already frozen cohort identities.  It deliberately
does not read the retained detector-TOD ``signal`` variable or evaluate any
counterfactual centroid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from netCDF4 import Dataset


BRANCH = "codex/sci-align-001-sample-lineage-phase"
TASK_BASE = "c468ffc58de95e8f1c55d6ac6382b6c452543f7d"
APPLICATION_CANDIDATE = "c77105b9b1676ec1ec74a9d560765954c5f1d5dd"
GOVERNING_APPLICATION = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
DT_SEC = 0.008192
HALF_CELL_SEC = DT_SEC / 2.0
PACKAGE_REL = Path("validation/sci_align_001_sample_lineage_phase_2026-08-03")
LR_PACKAGE_REL = Path("validation/sci_align_001_lr_beammap_2026-08-02")
REDUCTION = Path(
    "/private/tmp/citlali-sci-align-001-phase1-"
    "c77105b9b1676ec1ec74a9d560765954c5f1d5dd/beammap/beammap"
)
CONFIG = REDUCTION / "config/citlali_o148670_0_2_c1.yaml"
RESULT = REDUCTION / "reduced/redu00/148670"
PROVENANCE = RESULT / "timestream_output_provenance.yaml"
DETECTOR_TOD = (
    RESULT
    / "raw/source_crossing_tod/"
    "toltec_commissioning_beammap_148670_ptc_detector_tod.nc"
)
PREPARATION = REDUCTION / "evidence/preparation.json"
OWNER_TEXT = Path(
    "/Users/gwilson/.codex/attachments/"
    "24e95d50-b5a6-4256-b3e2-9273ea6cc506/pasted-text.txt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def validate_identity(repo: Path) -> dict[str, Any]:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    if branch != BRANCH or head != TASK_BASE:
        raise RuntimeError(
            f"identity gate failed: branch={branch} head={head}; "
            f"expected {BRANCH} at {TASK_BASE}"
        )
    for ancestor, descendant in (
        (GOVERNING_APPLICATION, APPLICATION_CANDIDATE),
        (APPLICATION_CANDIDATE, TASK_BASE),
    ):
        status = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            check=False,
        ).returncode
        if status:
            raise RuntimeError(f"required ancestry is absent: {ancestor} -> {descendant}")
    status = git(repo, "status", "--porcelain=v1")
    allowed = "tools/diagnostics/prepare_sci_align_001_sample_lineage_phase.py"
    unexpected = [line for line in status.splitlines() if allowed not in line]
    if unexpected:
        raise RuntimeError(f"unexpected dirty paths before preparation: {unexpected}")
    return {"branch": branch, "head": head, "status": "clean_except_prepare_tool"}


def scalar(dataset: Dataset, name: str) -> Any:
    return np.asarray(dataset[name][...]).item()


def reconstruct_legacy_timestamp(ts: np.ndarray, fpga_hz: float) -> np.ndarray:
    fields = np.asarray(ts, dtype=np.float64)
    anchor = int(fields[0, 0] + fields[0, 5] * 1.0e-9 - 0.5)
    delta = fields[:, 2] - fields[:, 4]
    delta[fields[:, 2] < fields[:, 4]] += 4294967295.0
    result = anchor + fields[:, 1] + delta / fpga_hz
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise RuntimeError("timestamp reconstruction is not finite and increasing")
    return result


def offset_map(config: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in config.get("interface_sync_offset", []):
        if len(entry) != 1:
            raise RuntimeError(f"malformed interface offset entry: {entry}")
        key, value = next(iter(entry.items()))
        result[str(key)] = float(value)
    return result


def source_span(path: Path, start_token: str, end_token: str) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    starts = [i + 1 for i, line in enumerate(lines) if start_token in line]
    if not starts:
        raise RuntimeError(f"source token not found in {path}: {start_token}")
    start = starts[0]
    ends = [i + 1 for i, line in enumerate(lines[start:], start) if end_token in line]
    return start, (ends[0] if ends else start)


def source_trace(repo: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "raw_schema_row_pair",
            Path("include/citlali/core/engine/detail/sci_align_netcdf_input_contract.h"),
            "require_legacy_toltec_timing_schema(",
            "Data.Toltec.Is/Data.Toltec.Qs 'time' dimension",
            "I, Q, and Ts share the same nonempty native time-row cardinality",
        ),
        (
            "timestamp_reconstruction",
            Path("include/citlali/core/pipeline/sci_align_contract.h"),
            "reconstruct_legacy_detector_timestamps(",
            "return result;",
            "Ts native row is reconstructed without row permutation; offset is later positive-add",
        ),
        (
            "raw_slice",
            Path("include/citlali/core/engine/detail/kidsproc_gaps_impl.h"),
            "load_rawobs_gaps(",
            "return result;",
            "one common source slice selects the row-aligned I/Q block",
        ),
        (
            "solver_row_geometry",
            Path("build/_deps/kidscpp-src/src/kids/timestream/solver.cpp"),
            "std::tie(ntimes, ntones)",
            "RMatrixXd qts(ntimes, ntones);",
            "solver outputs retain the input ntimes row axis",
        ),
        (
            "native_row_to_slot",
            Path("include/citlali/core/utils/utils.h"),
            "for (Eigen::Index native_row = 0; native_row < t_valid.size();",
            "result.row(slot) = data_valid.row(native_row);",
            "slot comes from t_valid(native_row), then the same data_valid native_row is copied",
        ),
        (
            "scan_population",
            Path("include/citlali/core/pipeline/timestream_scan_generation.h"),
            "populate_rtc_scan_samples(",
            "decltype(scan_rawobs)().swap(scan_rawobs);",
            "scan window and common-grid index bounds are passed unchanged into population",
        ),
        (
            "rtc_filter_and_inner_crop",
            Path("include/citlali/core/timestream/rtc/rtcproc.h"),
            "if (run_downsample)",
            "out.scans.data = in.scans.data.block(si, 0, sl, in.scans.data.cols());",
            "realized downsampling is disabled; the RTC output is the same ordered inner row block",
        ),
        (
            "ptc_row_geometry",
            Path("include/citlali/core/timestream/ptc/ptcproc.h"),
            "Eigen::Index n_pts = in.scans.data.rows();",
            "log_kernel_matrix_diag(logger, \"ptc run output\"",
            "PTC transforms values on the fixed n_pts row axis and does not relabel or permute rows",
        ),
        (
            "retained_output_copy",
            Path("include/citlali/core/engine/detail/beammap_detector_tod_netcdf_helpers.h"),
            "put_detector_tod_signal_flags(",
            "flags_block[data_idx] =",
            "retained sample index copies ptc.scans.data(sample, det) and its same-index flag",
        ),
    ]
    rows = []
    for stage, relative, begin, end, conclusion in specs:
        path = repo / relative
        first, last = source_span(path, begin, end)
        rows.append(
            {
                "stage": stage,
                "path": str(relative),
                "source_sha256": sha256_file(path),
                "line_start_1based": first,
                "line_stop_1based_inclusive": last,
                "lineage_conclusion": conclusion,
            }
        )
    return rows


def model_rows() -> list[dict[str, Any]]:
    rows = []
    for basis in ("assigned_slot", "raw_detector_timestamp"):
        for k in (-1, 0, 1):
            for phi in (-0.5, 0.0, 0.5):
                rows.append(
                    {
                        "model_id": f"{basis}_k{k:+d}_phi{phi:+.1f}",
                        "time_basis": basis,
                        "row_shift_k": k,
                        "phase_phi_samples": phi,
                        "phase_sec": phi * DT_SEC,
                        "effective_time_formula": "T[n+k] + phi * 0.008192 s",
                        "row_identity_changed": k != 0,
                        "primary_support": "all-model common interior support",
                        "secondary_support": "model-native within-scan support",
                    }
                )
    return rows


def prepare(repo: Path) -> None:
    identity = validate_identity(repo)
    required = [CONFIG, PROVENANCE, DETECTOR_TOD, PREPARATION, OWNER_TEXT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required retained evidence is missing: {missing}")

    package = repo / PACKAGE_REL
    package.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(CONFIG.read_text())
    provenance = yaml.safe_load(PROVENANCE.read_text())
    prior_inputs = {
        row["path"]: row["sha256"]
        for row in json.loads(PREPARATION.read_text())["selected_inputs"]
    }
    offsets = offset_map(config)
    raw_items = [
        row
        for row in config["inputs"][0]["data_items"]
        if str(row["meta"]["interface"]).startswith("toltec")
    ]
    raw_state = []
    for item in raw_items:
        interface = str(item["meta"]["interface"])
        path = Path(item["filepath"])
        with Dataset(path) as dataset:
            roach = int(scalar(dataset, "Header.Toltec.RoachIndex"))
            fpga_hz = float(scalar(dataset, "Header.Toltec.FpgaFreq"))
            accum = int(scalar(dataset, "Header.Toltec.AccumLen"))
            sample_hz = float(scalar(dataset, "Header.Toltec.SampleFreq"))
            ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
            iq_rows = int(dataset["Data.Toltec.Is"].shape[0])
            q_rows = int(dataset["Data.Toltec.Qs"].shape[0])
        raw_time = reconstruct_legacy_timestamp(ts, fpga_hz)
        corrected = raw_time + offsets.get(interface, 0.0)
        if roach != int(interface.removeprefix("toltec")):
            raise RuntimeError(f"interface/header mismatch for {interface}: {roach}")
        if ts.shape[0] != iq_rows or ts.shape[0] != q_rows:
            raise RuntimeError(f"raw row cardinality mismatch for {interface}")
        raw_state.append(
            {
                "interface": interface,
                "path": path,
                "roach": roach,
                "fpga_hz": fpga_hz,
                "accum": accum,
                "sample_hz": sample_hz,
                "ts": ts,
                "time": corrected,
                "offset": offsets.get(interface, 0.0),
            }
        )

    phase = max(float(row["time"][0]) for row in raw_state)
    alignment = provenance["realized"]["sci_align_alignment"]
    count = int(alignment["governing_compatibility_axis"]["sample_count"])
    if not math.isclose(float(alignment["grid"]["phase_sec"]), phase, abs_tol=0.0):
        raise RuntimeError("reconstructed phase is not exact provenance phase")
    if float(alignment["grid"]["cadence_sec"]) != DT_SEC:
        raise RuntimeError("retained cadence differs from frozen 8.192 ms")

    records = [
        row
        for row in provenance["realized"]["sci_align_scan_plan"]["records"]
        if row["legacy_processing_admitted"]
    ]
    science = np.zeros(count, dtype=bool)
    for record in records:
        start = int(record["compatibility_science"]["start"] - 1)
        stop = int(record["compatibility_science"]["stop"] - 1)
        science[start:stop] = True

    mapping_rows = []
    manifest_rows = []
    for row in raw_state:
        times = row["time"]
        slots = np.floor((times - phase) / DT_SEC + 0.5).astype(np.int64)
        assigned = phase + slots.astype(np.float64) * DT_SEC
        residual = times - assigned
        if np.any(np.abs(residual) >= HALF_CELL_SEC):
            raise RuntimeError(f"half-cell violation in {row['interface']}")
        if np.any(np.diff(slots) <= 0):
            raise RuntimeError(f"slot collision or reversal in {row['interface']}")
        slot_to_row = np.full(count, -1, dtype=np.int64)
        inside = (slots >= 0) & (slots < count)
        slot_to_row[slots[inside]] = np.flatnonzero(inside)
        science_missing = int(np.sum(science & (slot_to_row < 0)))
        packet_diff = np.diff(row["ts"][:, 3].astype(np.int64))
        mapping_rows.append(
            {
                "interface": row["interface"],
                "roach_index": row["roach"],
                "native_rows": int(times.size),
                "iq_ts_row_cardinality_exact": True,
                "sample_rate_hz": row["sample_hz"],
                "cadence_sec": row["accum"] / row["fpga_hz"],
                "interface_offset_sec_positive_add": row["offset"],
                "first_corrected_timestamp_sec": float(times[0]),
                "last_corrected_timestamp_sec": float(times[-1]),
                "first_global_slot": int(slots[0]),
                "last_global_slot": int(slots[-1]),
                "minimum_raw_minus_lattice_slot_sec": float(np.min(residual)),
                "maximum_raw_minus_lattice_slot_sec": float(np.max(residual)),
                "maximum_absolute_raw_minus_lattice_slot_sec": float(
                    np.max(np.abs(residual))
                ),
                "slot_collisions_or_reversals": int(np.sum(np.diff(slots) <= 0)),
                "packet_gap_events": int(np.sum(packet_diff > 1)),
                "science_slots_without_original_native_row": science_missing,
                "row_to_slot_mapping": "one_to_one_increasing",
            }
        )
        path = row["path"]
        digest = prior_inputs.get(str(path))
        if digest is None:
            raise RuntimeError(f"raw input lacks inherited digest: {path}")
        manifest_rows.append(
            {
                "identity": row["interface"],
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "digest_source": "phase_one_preparation_manifest",
                "use": "raw_I_Q_Ts_row_lineage_and_counterfactual_time_basis",
                "mutated": False,
            }
        )

    prior_matches = repo / LR_PACKAGE_REL / "matched_fit_results.csv"
    cohort = []
    with prior_matches.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["level"] == "detector":
                cohort.append(int(row["uid"]))
    cohort = sorted(set(cohort))
    if len(cohort) != 4809:
        raise RuntimeError(f"frozen detector cohort changed: {len(cohort)}")
    (package / "frozen_confirmatory_uids.txt").write_text(
        "".join(f"{uid}\n" for uid in cohort)
    )

    other_inputs = [
        ("owner_text", OWNER_TEXT, "owner_hypothesis_and_dispatch"),
        ("realized_config", CONFIG, "realized_input_and_offset_authority"),
        ("provenance", PROVENANCE, "realized_alignment_and_scan_authority"),
        ("detector_tod", DETECTOR_TOD, "retained_signal_and_scan_sample_identity"),
        ("prior_lr_checksums", repo / LR_PACKAGE_REL / "SHA256SUMS", "frozen_lr_package_identity"),
        ("prior_lr_protocol", repo / LR_PACKAGE_REL / "preregistered_protocol.json", "frozen_scan_and_fit_protocol"),
        ("prior_lr_registry", repo / LR_PACKAGE_REL / "scan_direction_registry.csv", "frozen_99_left_99_right_registry"),
        ("prior_lr_matches", prior_matches, "frozen_confirmatory_detector_cohort"),
    ]
    for name, path, use in other_inputs:
        manifest_rows.append(
            {
                "identity": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "digest_source": "measured_by_preregistration",
                "use": use,
                "mutated": False,
            }
        )

    protocol = {
        "task": "SCI-ALIGN-001-SAMPLE-LINEAGE-PHASE",
        "scientific_question": (
            "Does D[n] acquire a wrong Ts row, and does an independent edge-versus-centroid "
            "phase explain the retained Beammap direction-odd residual?"
        ),
        "not_authorized": [
            "application behavior change",
            "production timing correction",
            "SCI-MAP-001-UNITY-001 modification",
            "new Citlali reduction",
            "Unity access",
            "producer timestamp semantic claim",
        ],
        "cadence_sec": DT_SEC,
        "primary_models": {
            "formula": "t_eff(n;k,phi) = T[n+k] + phi * 0.008192 s",
            "k": [-1, 0, 1],
            "phi_samples": [-0.5, 0.0, 0.5],
            "time_bases": ["assigned_slot", "raw_detector_timestamp"],
            "row_shift_requirement": "k changes row association; it is not a constant timestamp addition",
            "phase_requirement": "phi preserves detector row identity",
        },
        "primary_population": {
            "detectors": "4809 frozen matched detector UIDs from LR-BEAMMAP",
            "scans": "frozen 99 left and 99 right trajectory-derived Hold-valid registry",
            "samples": (
                "intersection of original flags/finite/frozen baseline radial selection and "
                "within-scan valid support for k=-1,0,+1 across both bases and all phases"
            ),
            "purpose": "identical interior detector rows for every model",
        },
        "secondary_population": (
            "each model's native within-scan support, with added/removed boundary rows reported"
        ),
        "assigned_slot_residual_definition": {
            "formula": (
                "r[i,n] = (t_raw[i,n] + offset[i]) - "
                "(phase + round_half_up(((t_raw[i,n] + offset[i]) - phase)/dt) * dt)"
            ),
            "round_half_up": "floor(x + 0.5)",
            "sign": "raw-plus-offset minus lattice-slot time",
            "unit": "s",
            "offset_stage": "positive-add exactly once before lattice assignment",
            "warning": "half-cell distance is an engineering assignment residual, not a sky-error tolerance",
        },
        "frozen_discrete_predictions_from_existing_estimate_ms": {
            "baseline_magnitude": 12.138,
            "correcting_k_plus_one_only_remaining_magnitude": 3.946,
            "correcting_phi_plus_half_only_remaining_magnitude": 8.042,
            "correcting_k_plus_one_phi_plus_half_remaining_signed": 0.150,
            "sign_basis": (
                "existing signed estimate is -12.138 ms; evaluating telescope coordinates "
                "12.288 ms later predicts approximately +0.150 ms"
            ),
        },
        "metrics_every_model": [
            "pooled right-minus-left parallel and perpendicular centroid",
            "timing estimate and delete-one-stable-scan jackknife interval",
            "per-array and per-network estimates",
            "network correlation and slope versus exactly defined assigned-slot residual",
            "first-half versus second-half estimate",
            "balanced same-direction null",
            "major/minor FWHM, ellipticity, and amplitude",
            "common/native sample counts and exact boundary-row differences",
        ],
        "continuous_profile": (
            "secondary pooled common-support profile around the best discrete model; "
            "it cannot redefine the primary hypotheses"
        ),
        "classification": [
            "demonstrated row-index error plus half-sample convention error",
            "evidence only for a common approximately 1.5-sample effective offset",
            "common component explained but interface/time-dependent residual remains",
            "hypothesis not supported",
        ],
        "interpretation_rule": (
            "A numerical null cannot establish a row-index defect. Direct Stage-A lineage "
            "evidence is mandatory, and producer authority retains integration-event semantics."
        ),
        "confirmatory_signal_read_during_preregistration": False,
    }

    stage_a = {
        "status": "bounded_lineage_reconstruction_complete_before_counterfactual_signal_read",
        "direct_row_mismatch_found": False,
        "conclusion": (
            "The inspected implementation and actual raw cardinalities preserve D[n] with Ts[n] "
            "through native read, solver row geometry, slot insertion, scan indexing, and retained "
            "sample copy. No Citlali n<->n+/-1 defect is demonstrated."
        ),
        "qualification": (
            "Filters transform values across neighboring samples but do not relabel or permute the "
            "sample axis. Synthesized rows require separate lineage; the retained provenance reports "
            "zero synthesized interface slots in admitted science support."
        ),
        "producer_semantics": (
            "unresolved: Data.Toltec.Ts integration start/end/effective-centroid meaning is unavailable"
        ),
        "stage_b_row_shift_interpretation": (
            "k=+/-1 remains an explicit counterfactual. With no direct Stage-A mismatch, a null under "
            "k plus phi can support only an effective timing offset, not the proposed decomposition."
        ),
        "stage_b_technically_feasible_without_new_reduction": all(
            int(row["science_slots_without_original_native_row"]) == 0
            for row in mapping_rows
        ),
        "retained_dense_raw_to_output_mapping_persisted": False,
        "mapping_reconstruction": (
            "deterministic from raw Ts, exact offset/slot formula, compact scan origin, and same-index "
            "retained copy; expanded mappings are diagnostic-only"
        ),
    }

    write_json(package / "identity.json", identity)
    write_csv(package / "input_manifest.csv", manifest_rows)
    write_csv(package / "raw_row_mapping_summary.csv", mapping_rows)
    write_csv(package / "lineage_source_trace.csv", source_trace(repo))
    write_csv(package / "model_registry.csv", model_rows())
    write_json(package / "preregistered_protocol.json", protocol)
    write_json(package / "stage_a_conclusion.json", stage_a)

    ordered = [
        "frozen_confirmatory_uids.txt",
        "identity.json",
        "input_manifest.csv",
        "lineage_source_trace.csv",
        "model_registry.csv",
        "preregistered_protocol.json",
        "raw_row_mapping_summary.csv",
        "stage_a_conclusion.json",
    ]
    aggregate = hashlib.sha256()
    entries = []
    for name in ordered:
        digest = sha256_file(package / name)
        entries.append({"path": name, "sha256": digest})
        aggregate.update(name.encode() + b"\0" + bytes.fromhex(digest))
    write_json(
        package / "preregistration_freeze.json",
        {
            "ordered_files": ordered,
            "entries": entries,
            "aggregate_sha256": aggregate.hexdigest(),
            "confirmatory_signal_read": False,
            "freeze_rule": "analyze mode must verify this aggregate before reading signal",
        },
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    prepare(repo)


if __name__ == "__main__":
    main()
