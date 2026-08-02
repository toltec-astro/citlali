#!/usr/bin/env python3
"""Generate the additive SCI-ALIGN-001 D005 sky-domain decision package.

This evidence-only diagnostic propagates the frozen governing and proposed
detector slot assignments through the frozen telescope AltAz tangent-plane
construction.  It reads owner-local validation data and coordination records
without modifying them.  It does not execute Citlali, TolProj, mapmaking, or
Unity, and it does not edit application code or the frozen D005 package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from netCDF4 import Dataset


BRANCH = "codex/repair-sci-align-001"
TASK_BASE_COMMIT = "5a0d64b8f1b9b246b1b5d575c548269823203d22"
PHASE0_COMMIT = "53c7154a3633dfe19dc036cfb5a6250f729a897d"
GOVERNING_APPLICATION_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
COORDINATION_HEAD = "6785152c2a2d4113c9ba89073de00cb454aa70c4"
D001_COMMIT = "86434df2cfb5b85d0ccd306150cb428321abdbb9"
D002_COMMIT = "10981b29c1870e745b7f3c9cabed3c634a46427f"
D004_COMMIT = "a3775bf3039461a6435f07938572dd23b3f03d47"
D001_SHA256 = "0efe9d06bf02ceca473c92b00dce4c6d1ec9b6e564f226c9451f444ee5a6d66c"
D002_SHA256 = "7e4e2c02bf2e16035d6a2aceacaa4e07c7c528e08f09fdb2d0a9186d510465cd"
D004_SHA256 = "ea03c5b614c7ce64ab5ab071c48c07d2f4910b919941798e70668049e53faf78"
D005_PACKAGE = "sci_align_001_align_p0_d005_2026-08-01"
D005_SUMS_SHA256 = "149ef430af3223562d9e69b7224703b831f6f56629b2f3c513bf44c40a567bbb"
D005_REPORT_SHA256 = "6edf7a7bd79881c3e7f9809d1da36c56a763bb22cd9e7589e91950351536663a"
PACKAGE_NAME = "sci_align_001_align_p0_d005_sky_domain_2026-08-02"
GENERATOR_RELATIVE = "tools/diagnostics/generate_sci_align_001_d005_sky.py"
SELECTED_CONFIG_IDS = {
    "point_core", "beammap_core", "beammap_support", "science_support",
}
EXPECTED_INTERFACES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12)
EXPECTED_OBSERVATION_COUNTS = {
    148669: (85082, 85063, 19),
    148670: (4220705, 4220689, 16),
    148671: (83614, 83600, 14),
    152389: (84689, 84667, 22),
    152391: (85661, 85646, 15),
    152393: (85835, 85811, 24),
}
F_REF_HZ = 122.0703125
DT_SEC = 0.008192
HALF_DT_SEC = DT_SEC / 2.0
UINT32_MODULUS = 2**32
RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi
SPEED_LABELS = ("0_to_lt_50", "50_to_lt_100", "100_to_lt_200", "ge_200")
DIRECTION_LABELS = (
    "+az", "+az_plus_el", "+el", "-az_plus_el",
    "-az", "-az_minus_el", "-el", "+az_minus_el",
)
SHIFT_SPECS = (("minus_full", -1.0), ("minus_half", -0.5),
               ("plus_half", 0.5), ("plus_full", 1.0))


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=text,
    ).stdout


def git_blob(repo: Path, commit: str, relative: str) -> bytes:
    return git(repo, "show", f"{commit}:{relative}", text=False)  # type: ignore[return-value]


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalized_array_bytes(values: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False)).tobytes()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    return sha256_bytes(normalized_array_bytes(values, dtype))


def decode_chars(values: np.ndarray) -> str:
    return b"".join(bytes(value) for value in np.asarray(values).reshape(-1)).decode(
        "ascii", errors="strict",
    ).rstrip("\x00 ")


def periodic_fix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if float(np.max(result)) > 1.99 * math.pi and float(np.min(result)) < math.pi:
        result[result < math.pi] += 2.0 * math.pi
    return result


def cxx_round(values: np.ndarray) -> np.ndarray:
    return np.trunc(values + np.copysign(0.5, values)).astype(np.int64)


def array_name(interface: int) -> str:
    if 0 <= interface <= 6:
        return "a1100"
    if 7 <= interface <= 10:
        return "a1400"
    if 11 <= interface <= 12:
        return "a2000"
    raise RuntimeError(f"invalid TolTEC interface: {interface}")


def path_status_allowed(repo: Path) -> list[str]:
    lines = str(git(repo, "status", "--porcelain=v1", "--untracked-files=all")).splitlines()
    allowed_file = GENERATOR_RELATIVE
    allowed_prefix = f"validation/{PACKAGE_NAME}/"
    bad = []
    for line in lines:
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path != allowed_file and not path.startswith(allowed_prefix):
            bad.append(line)
    if bad:
        raise RuntimeError(f"unrelated or application worktree change present: {bad}")
    return lines


def validate_identity(repo: Path, coordination_repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch = str(git(repo, "symbolic-ref", "--short", "HEAD")).strip()
    head = str(git(repo, "rev-parse", "HEAD")).strip()
    if branch != BRANCH or not is_ancestor(repo, TASK_BASE_COMMIT, head):
        raise RuntimeError(f"repair identity mismatch: branch={branch} head={head}")
    path_status_allowed(repo)
    committed_delta = str(git(repo, "diff", "--name-only", f"{TASK_BASE_COMMIT}..{head}")).splitlines()
    allowed_committed = {GENERATOR_RELATIVE}
    if any(
        path not in allowed_committed and not path.startswith(f"validation/{PACKAGE_NAME}/")
        for path in committed_delta
    ):
        raise RuntimeError(f"unexpected committed delta after task base: {committed_delta}")

    # D005 already verified the coordination worktree clean at the immutable
    # snapshot below.  The coordinator may continue work in that checkout
    # while this additive package is generated, so never inspect its live
    # HEAD/status/files.  Bind only committed Git objects.
    try:
        git(coordination_repo, "cat-file", "-e", f"{COORDINATION_HEAD}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("frozen coordination commit is unavailable") from error

    d005 = repo / "validation" / D005_PACKAGE
    sums = d005 / "SHA256SUMS"
    if sha256_file(sums) != D005_SUMS_SHA256:
        raise RuntimeError("frozen D005 SHA256SUMS identity changed")
    for line in sums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        measured = sha256_file(d005 / name)
        if measured != expected:
            raise RuntimeError(f"frozen D005 artifact changed: {name}")
    if sha256_file(d005 / "REPORT.md") != D005_REPORT_SHA256:
        raise RuntimeError("frozen D005 report identity changed")

    decisions = (
        ("D001", D001_COMMIT, D001_SHA256,
         "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D001_DECISION_2026-08-01.md"),
        ("D002", D002_COMMIT, D002_SHA256,
         "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D002_DECISION_2026-08-01.md"),
        ("D004", D004_COMMIT, D004_SHA256,
         "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D004_DECISION_2026-08-01.md"),
    )
    authority_rows: list[dict[str, Any]] = []
    for decision_id, commit, expected, relative in decisions:
        blob = git_blob(coordination_repo, commit, relative)
        if sha256_bytes(blob) != expected or not is_ancestor(coordination_repo, commit, COORDINATION_HEAD):
            raise RuntimeError(f"{decision_id} authority identity changed")
        authority_rows.append({
            "authority_id": decision_id,
            "authority_class": "approved_owner_decision",
            "path": f"{coordination_repo}@{commit}:{relative}",
            "git_commit": commit, "sha256": expected,
            "use": "timestamp/slot/telescope scientific authority",
        })

    repository_authorities = (
        "AGENTS.md", "doc/ARCHITECTURE.md", "doc/SCIENTIFIC_CONVENTIONS.md",
        "doc/RETAINED_DEBT.md", "doc/REFACTOR_STATUS.md",
        "validation/validation_profiles.json", "validation/product_contracts.json",
        "validation/profiles/beammap_scientific_equivalence_v1.json",
    )
    for relative in repository_authorities:
        path = repo / relative
        authority_rows.append({
            "authority_id": f"repair_repo:{relative}",
            "authority_class": "repository_local_authority", "path": str(path),
            "git_commit": TASK_BASE_COMMIT, "sha256": sha256_file(path),
            "use": "science conventions, active comparison policy, and scope boundary",
        })

    source_paths = (
        "include/citlali/core/pipeline/timestream_alignment_helpers.h",
        "include/citlali/core/engine/detail/todproc_alignment_impl.h",
        "src/citlali/core/engine/telescope.cpp",
        "include/citlali/core/utils/pointing.h",
        "include/citlali/core/utils/constants.h",
        "include/citlali/core/utils/toltec_io.h",
    )
    source_rows = []
    for relative in source_paths:
        blob = git_blob(repo, GOVERNING_APPLICATION_SHA, relative)
        row = {
            "authority_id": f"governing_source:{relative}",
            "authority_class": "exact_governing_application_source",
            "path": relative, "git_commit": GOVERNING_APPLICATION_SHA,
            "sha256": sha256_bytes(blob),
            "use": "slot, telescope trajectory, detector geometry, unit, or array identity",
        }
        source_rows.append(row)
        authority_rows.append(row)

    authority_rows.append({
        "authority_id": "frozen_D005_SHA256SUMS",
        "authority_class": "frozen_parent_evidence_package",
        "path": str(sums), "git_commit": TASK_BASE_COMMIT,
        "sha256": D005_SUMS_SHA256,
        "use": "input selection, file SHA-256, Hold analysis, and preregistration continuity",
    })
    identity = {
        "protocol_id": "ALIGN-P0-D005-SKY-001",
        "repair_repository": str(repo), "branch": branch,
        "task_base_commit": TASK_BASE_COMMIT,
        "generation_head_is_base_or_package_only_descendant": True,
        "governing_application_sha": GOVERNING_APPLICATION_SHA,
        "phase_zero_evidence_commit": PHASE0_COMMIT,
        "coordination_repository": str(coordination_repo),
        "coordination_frozen_head": COORDINATION_HEAD,
        "coordination_frozen_snapshot_verified_clean_by_parent_D005": True,
        "coordination_live_head_status_inspected": False,
        "coordination_live_worktree_content_read": False,
        "repair_worktree_scope_validated": True,
        "allowed_status_scope": [GENERATOR_RELATIVE, f"validation/{PACKAGE_NAME}/"],
        "committed_delta_after_task_base_scope_validated": True,
        "application_source_changed": False,
        "frozen_D005_package": str(d005),
        "frozen_D005_sha256sums_sha256": D005_SUMS_SHA256,
        "frozen_D005_report_sha256": D005_REPORT_SHA256,
        "frozen_D005_rewritten": False,
        "owner_sky_domain_direction_received_in_task": "2026-08-02",
        "successor_output_inspected": False,
        "citlali_executed": False, "tolproj_executed": False,
        "unity_contacted": False, "mapmaking_executed": False,
        "phase_one_authorized": False,
        "generator_path": str((repo / GENERATOR_RELATIVE).resolve()),
        "generator_sha256": sha256_file(repo / GENERATOR_RELATIVE),
        "governing_source_blobs": source_rows,
    }
    return identity, authority_rows


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_selected_observations(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d005 = repo / "validation" / D005_PACKAGE
    inputs = [
        row for row in read_manifest(d005 / "selected_input_manifest.csv")
        if row["config_id"] in SELECTED_CONFIG_IDS
        and row["item_class"] == "application_stream_input"
    ]
    configs = {
        row["config_id"]: row for row in read_manifest(d005 / "selected_config_manifest.csv")
        if row["config_id"] in SELECTED_CONFIG_IDS
    }
    contexts: dict[tuple[str, int], dict[str, Any]] = {}
    for row in inputs:
        key = (row["config_id"], int(row["obsnum"]))
        context = contexts.setdefault(key, {
            "config_id": row["config_id"], "mode": row["mode"],
            "role": row["fixture_role"], "obsnum": int(row["obsnum"]),
            "detectors": {}, "telescope": None,
            "config_path": Path(configs[row["config_id"]]["path"]),
            "offsets": json.loads(configs[row["config_id"]]["requested_interface_offsets_sec_json"]),
        })
        path = Path(row["local_path"])
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"selected input missing or size changed: {path}")
        item = {
            "path": path, "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"], "config_id": row["config_id"],
            "role": row["fixture_role"], "mode": row["mode"],
        }
        if row["interface"] == "lmt":
            context["telescope"] = item
        elif row["interface"].startswith("toltec"):
            interface = int(row["interface"].removeprefix("toltec"))
            context["detectors"][interface] = item
    if len(contexts) != 7:
        raise RuntimeError(f"unexpected selected Pointing/Beammap context count: {len(contexts)}")

    by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts.values():
        if set(context["detectors"]) != set(EXPECTED_INTERFACES) or context["telescope"] is None:
            raise RuntimeError(f"incomplete selected context: {context['config_id']} {context['obsnum']}")
        if set(context["offsets"]) != {f"toltec{i}" for i in range(13)} | {"hwpr"}:
            raise RuntimeError(f"incomplete offset request: {context['config_path']}")
        if any(float(value) != 0.0 for value in context["offsets"].values()):
            raise RuntimeError(f"nonzero selected offset is outside D005 evidence: {context['config_path']}")
        by_obs[context["obsnum"]].append(context)

    rank = {"point_core": 0, "beammap_core": 0, "beammap_support": 1, "science_support": 2}
    observations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for obsnum, candidates in sorted(by_obs.items()):
        candidates.sort(key=lambda item: (rank[item["config_id"]], item["config_id"]))
        chosen = candidates[0]
        roles = sorted({item["role"] for item in candidates})
        config_ids = sorted({item["config_id"] for item in candidates})
        for interface in EXPECTED_INTERFACES:
            identities = {
                (item["detectors"][interface]["sha256"], item["detectors"][interface]["size_bytes"])
                for item in candidates
            }
            if len(identities) != 1:
                raise RuntimeError(f"conflicting duplicate detector identity: {obsnum} toltec{interface}")
        telescope_identities = {
            (item["telescope"]["sha256"], item["telescope"]["size_bytes"])
            for item in candidates
        }
        if len(telescope_identities) != 1:
            raise RuntimeError(f"conflicting duplicate telescope identity: {obsnum}")
        chosen = dict(chosen)
        chosen["selection_roles"] = roles
        chosen["selection_config_ids"] = config_ids
        observations.append(chosen)
        for interface in EXPECTED_INTERFACES:
            item = chosen["detectors"][interface]
            manifest_rows.append({
                "obsnum": obsnum, "stream": f"toltec{interface}",
                "array": array_name(interface), "selection_config_ids_json": canonical_json(config_ids),
                "selection_roles_json": canonical_json(roles), "canonical_path": str(item["path"]),
                "size_bytes": item["size_bytes"], "full_file_sha256_from_frozen_D005": item["sha256"],
                "full_digest_verification": "inherited_exactly_from_digest-verified_frozen_D005_manifest",
                "current_read_scope": "timing/header variables only; variable-level digest emitted",
            })
        tel = chosen["telescope"]
        manifest_rows.append({
            "obsnum": obsnum, "stream": "lmt", "array": "not_applicable",
            "selection_config_ids_json": canonical_json(config_ids),
            "selection_roles_json": canonical_json(roles), "canonical_path": str(tel["path"]),
            "size_bytes": tel["size_bytes"], "full_file_sha256_from_frozen_D005": tel["sha256"],
            "full_digest_verification": "inherited_exactly_from_digest-verified_frozen_D005_manifest",
            "current_read_scope": "trajectory/Hold/header variables only; variable-level digest emitted",
        })
    if [item["obsnum"] for item in observations] != sorted(EXPECTED_OBSERVATION_COUNTS):
        raise RuntimeError("selected Pointing/Beammap observation identity changed")
    return observations, manifest_rows


def read_detector(item: dict[str, Any], expected_obsnum: int, interface: int) -> dict[str, Any]:
    path = item["path"]
    with Dataset(path) as dataset:
        def scalar(name: str) -> Any:
            return np.asarray(dataset[name][:]).item()

        roach = int(scalar("Header.Toltec.RoachIndex"))
        obsnum = int(scalar("Header.Toltec.ObsNum"))
        fpga = float(scalar("Header.Toltec.FpgaFreq"))
        accum = int(scalar("Header.Toltec.AccumLen"))
        rate = float(scalar("Header.Toltec.SampleFreq"))
        compile_time = int(scalar("Header.Toltec.CompileTime"))
        ts_var = dataset["Data.Toltec.Ts"]
        recv_var = dataset["Data.Toltec.RecvTime"]
        ts = np.asarray(ts_var[:], dtype=np.int64)
        recv = np.asarray(recv_var[:], dtype=np.float64)
        global_attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        ts_attrs = {name: ts_var.getncattr(name) for name in ts_var.ncattrs()}
        recv_attrs = {name: recv_var.getncattr(name) for name in recv_var.ncattrs()}
    if roach != interface or obsnum != expected_obsnum or ts.shape != (recv.size, 6):
        raise RuntimeError(f"detector identity/shape mismatch: {path}")
    if not (
        fpga == 256000000.0 and accum == 2097152 and rate == F_REF_HZ
        and fpga / accum == rate and 1.0 / rate == DT_SEC
    ):
        raise RuntimeError(f"selected native profile changed: {path}")
    anchor = int(float(ts[0, 0]) + float(ts[0, 5]) * 1.0e-9 - 0.5)
    signed_delta = ts[:, 2].astype(np.float64) - ts[:, 4].astype(np.float64)
    ticks = np.where(signed_delta < 0, signed_delta + UINT32_MODULUS - 1, signed_delta)
    times = anchor + ts[:, 1].astype(np.float64) + ticks / fpga
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise RuntimeError(f"nonfinite/nonmonotonic reconstructed detector time: {path}")
    lag = recv - times
    if not np.all(np.isfinite(lag)):
        raise RuntimeError(f"nonfinite receive lag: {path}")
    semantics_text = canonical_json({
        "global": global_attrs, "Data.Toltec.Ts": ts_attrs,
        "Data.Toltec.RecvTime": recv_attrs,
    }).lower()
    semantic_terms = [
        term for term in ("integration start", "integration end", "integration centroid",
                          "effective time", "cell_methods", "time_bounds", "exposure")
        if term in semantics_text
    ]
    variable_digest = hashlib.sha256()
    variable_digest.update(normalized_array_bytes(ts, "<i8"))
    variable_digest.update(normalized_array_bytes(recv, "<f8"))
    variable_digest.update(canonical_json({
        "roach": roach, "obsnum": obsnum, "fpga": fpga, "accum": accum,
        "rate": rate, "compile_time": compile_time, "attrs": json.loads(semantics_text),
    }).encode())
    return {
        "path": path, "interface": interface, "array": array_name(interface),
        "times": times, "recv_lag": lag, "rows": int(times.size),
        "compile_time": compile_time, "global_attrs": global_attrs,
        "ts_attrs": ts_attrs, "recv_attrs": recv_attrs,
        "semantic_terms": semantic_terms,
        "relevant_variables_sha256": variable_digest.hexdigest(),
        "full_file_sha256": item["sha256"],
    }


def read_telescope(obs: dict[str, Any]) -> dict[str, Any]:
    item = obs["telescope"]
    path = item["path"]
    names = ("TelAzAct", "TelElAct", "TelAzCor", "TelElCor", "SourceAz", "SourceEl")
    with Dataset(path) as dataset:
        tel_time_var = dataset["Data.TelescopeBackend.TelTime"]
        times = np.asarray(tel_time_var[:], dtype=np.float64)
        fields = {
            name: periodic_fix(np.asarray(dataset[f"Data.TelescopeBackend.{name}"][:], dtype=np.float64))
            for name in names
        }
        hold = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=np.float64)
        tel_time_attrs = {name: tel_time_var.getncattr(name) for name in tel_time_var.ncattrs()}
        map_headers: dict[str, Any] = {}
        if obs["mode"] == "beammap":
            map_headers = {
                "x_length_rad": float(np.asarray(dataset["Header.Map.XLength"][:]).item()),
                "y_length_rad": float(np.asarray(dataset["Header.Map.YLength"][:]).item()),
                "scan_angle_rad": float(np.asarray(dataset["Header.Map.ScanAngle"][:]).item()),
                "map_path": decode_chars(dataset["Header.Map.MapPath"][:]),
                "map_coord": decode_chars(dataset["Header.Map.MapCoord"][:]),
                "map_motion": decode_chars(dataset["Header.Map.MapMotion"][:]),
                "exec_mode": int(np.asarray(dataset["Header.Map.ExecMode"][:]).item()),
            }
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise RuntimeError(f"invalid telescope time: {path}")
    if any(values.shape != times.shape or not np.all(np.isfinite(values)) for values in fields.values()):
        raise RuntimeError(f"invalid telescope trajectory field: {path}")
    if hold.shape != times.shape or not np.all(np.isfinite(hold)) or not np.all(hold == np.floor(hold)):
        raise RuntimeError(f"invalid telescope Hold field: {path}")
    raw_word = hold.astype(np.uint64)
    relevant = hashlib.sha256()
    relevant.update(normalized_array_bytes(times, "<f8"))
    for name in names:
        relevant.update(name.encode())
        relevant.update(normalized_array_bytes(fields[name], "<f8"))
    relevant.update(normalized_array_bytes(raw_word, "<u8"))
    relevant.update(canonical_json({"TelTime_attrs": tel_time_attrs, "map_headers": map_headers}).encode())
    config = yaml.safe_load(obs["config_path"].read_text())
    mapmaking = config.get("mapmaking", {})
    if mapmaking.get("pixel_axes") != "altaz":
        raise RuntimeError(f"selected sky frame is not AltAz: {obs['config_path']}")
    return {
        "path": path, "times": times, "fields": fields, "raw_word": raw_word,
        "raw_word_float": hold, "tel_time_attrs": tel_time_attrs,
        "map_headers": map_headers, "mapmaking": mapmaking,
        "relevant_variables_sha256": relevant.hexdigest(),
        "full_file_sha256": item["sha256"],
    }


def trajectory(telescope: dict[str, Any], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    native_time = telescope["times"]
    supported = (target >= native_time[0]) & (target <= native_time[-1])
    x = np.full(target.shape, np.nan, dtype=np.float64)
    y = np.full(target.shape, np.nan, dtype=np.float64)
    if not np.any(supported):
        return x, y
    aligned = {
        name: np.interp(target[supported], native_time, values)
        for name, values in telescope["fields"].items()
    }
    tel_az = aligned["TelAzAct"].copy()
    wrap = tel_az - aligned["SourceAz"] > 0.9 * 2.0 * math.pi
    tel_az[wrap] -= 2.0 * math.pi
    y_rad = aligned["TelElAct"] - aligned["SourceEl"] - aligned["TelElCor"]
    x_rad = (
        np.cos(aligned["TelElAct"] - aligned["TelElCor"])
        * (tel_az - aligned["SourceAz"]) - aligned["TelAzCor"]
    )
    x[supported] = x_rad * RAD_TO_ARCSEC
    y[supported] = y_rad * RAD_TO_ARCSEC
    return x, y


def hold_views(telescope: dict[str, Any], target: np.ndarray) -> dict[str, np.ndarray]:
    native_time = telescope["times"]
    raw_word = telescope["raw_word"]
    raw_float = telescope["raw_word_float"]
    left = np.searchsorted(native_time, target, side="right") - 1
    right = np.searchsorted(native_time, target, side="left")
    if np.any(left < 0) or np.any(right >= native_time.size):
        raise RuntimeError("selected grid lies outside Hold support")
    left_word = raw_word[left]
    right_word = raw_word[right]
    linear = np.interp(target, native_time, raw_float)
    return {
        "left_word": left_word, "right_word": right_word,
        "legacy_linear_any": linear != 0.0,
        "left_any": left_word != 0, "right_any": right_word != 0,
        "left_bit8": (left_word & np.uint64(0x08)) != 0,
        "right_bit8": (right_word & np.uint64(0x08)) != 0,
        "transition_bracket": left_word != right_word,
    }


def direction_codes(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    speed = np.hypot(vx, vy)
    code = np.full(speed.shape, 8, dtype=np.uint8)
    valid = np.isfinite(speed) & (speed > 0.0)
    angle = np.arctan2(vy[valid], vx[valid])
    code[valid] = (np.floor((angle + math.pi / 8.0) / (math.pi / 4.0)).astype(np.int64) % 8).astype(np.uint8)
    return code


def speed_codes(speed: np.ndarray) -> np.ndarray:
    code = np.full(speed.shape, 4, dtype=np.uint8)
    valid = np.isfinite(speed)
    code[valid] = np.searchsorted(np.array([50.0, 100.0, 200.0]), speed[valid], side="right").astype(np.uint8)
    return code


def acceleration_codes(accel: np.ndarray, edges: np.ndarray) -> np.ndarray:
    code = np.full(accel.shape, 4, dtype=np.uint8)
    valid = np.isfinite(accel)
    code[valid] = np.searchsorted(edges, accel[valid], side="right").astype(np.uint8)
    return code


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_n": 0, f"{prefix}_mean": "", f"{prefix}_median": "",
            f"{prefix}_std_population": "", f"{prefix}_mad_sigma": "",
            f"{prefix}_p05": "", f"{prefix}_p95": "", f"{prefix}_p99_abs": "",
            f"{prefix}_max_abs": "",
        }
    median = float(np.quantile(finite, 0.5))
    return {
        f"{prefix}_n": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": median,
        f"{prefix}_std_population": float(np.std(finite)),
        f"{prefix}_mad_sigma": float(1.4826 * np.median(np.abs(finite - median))),
        f"{prefix}_p05": float(np.quantile(finite, 0.05)),
        f"{prefix}_p95": float(np.quantile(finite, 0.95)),
        f"{prefix}_p99_abs": float(np.quantile(np.abs(finite), 0.99)),
        f"{prefix}_max_abs": float(np.max(np.abs(finite))),
    }


def weighted_stats(values: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.int64)
    valid = np.isfinite(values) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return {f"{prefix}_{name}": "" for name in ("n", "mean", "p50", "p90", "p95", "p99", "max")}
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = int(cumulative[-1])

    def quantile(q: float) -> float:
        # Deterministic inverse empirical CDF for integer row weights.
        index = int(np.searchsorted(cumulative, max(1, math.ceil(q * total)), side="left"))
        return float(values[index])

    return {
        f"{prefix}_n": total,
        f"{prefix}_mean": float(np.average(values, weights=weights)),
        f"{prefix}_p50": quantile(0.50), f"{prefix}_p90": quantile(0.90),
        f"{prefix}_p95": quantile(0.95), f"{prefix}_p99": quantile(0.99),
        f"{prefix}_max": float(values[-1]),
    }


def build_observation_trajectory(
    obs: dict[str, Any], telescope: dict[str, Any], phase: float,
    current_count: int, union_min: int, union_max: int,
) -> dict[str, Any]:
    slots = np.arange(union_min, union_max + 1, dtype=np.int64)
    grid = phase + slots.astype(np.float64) * DT_SEC
    x, y = trajectory(telescope, grid)
    minus_h_x, minus_h_y = trajectory(telescope, grid - HALF_DT_SEC)
    plus_h_x, plus_h_y = trajectory(telescope, grid + HALF_DT_SEC)
    vx = (plus_h_x - minus_h_x) / DT_SEC
    vy = (plus_h_y - minus_h_y) / DT_SEC
    ax = (plus_h_x - 2.0 * x + minus_h_x) / (HALF_DT_SEC**2)
    ay = (plus_h_y - 2.0 * y + minus_h_y) / (HALF_DT_SEC**2)
    speed = np.hypot(vx, vy)
    acceleration = np.hypot(ax, ay)
    direction = direction_codes(vx, vy)
    speed_bin = speed_codes(speed)
    current_indices = np.arange(0, current_count, dtype=np.int64) - union_min
    current_accel = acceleration[current_indices]
    current_accel = current_accel[np.isfinite(current_accel)]
    if current_accel.size == 0:
        raise RuntimeError(f"no trajectory acceleration support for {obs['obsnum']}")
    accel_edges = np.quantile(current_accel, [0.25, 0.50, 0.75])
    accel_bin = acceleration_codes(acceleration, accel_edges)

    holds = hold_views(telescope, grid)
    outside = np.full(grid.shape, 2, dtype=np.uint8)
    composite = np.full(grid.shape, 2, dtype=np.uint8)
    raster_direction = np.full(grid.shape, 3, dtype=np.uint8)
    if obs["mode"] == "beammap":
        headers = telescope["map_headers"]
        expected = {
            "map_path": "Rectilinear", "map_coord": "Az",
            "map_motion": "Continuous", "exec_mode": 0,
        }
        if any(headers[name] != value for name, value in expected.items()):
            raise RuntimeError(f"Beammap raster selector changed: {headers}")
        angle = headers["scan_angle_rad"]
        xp = (x / RAD_TO_ARCSEC) * math.cos(angle) + (y / RAD_TO_ARCSEC) * math.sin(angle)
        yp = -(x / RAD_TO_ARCSEC) * math.sin(angle) + (y / RAD_TO_ARCSEC) * math.cos(angle)
        inside = (
            (-headers["x_length_rad"] / 2.0 <= xp) & (xp <= headers["x_length_rad"] / 2.0)
            & (-headers["y_length_rad"] / 2.0 <= yp) & (yp <= headers["y_length_rad"] / 2.0)
        )
        outside = (~inside).astype(np.uint8)
        composite = (holds["legacy_linear_any"] | (~inside)).astype(np.uint8)
        projected = vx * math.cos(angle) + vy * math.sin(angle)
        raster_direction = np.where(
            ~np.isfinite(projected), 3, np.where(projected < 0, 0, np.where(projected > 0, 1, 2)),
        ).astype(np.uint8)

    shifts: dict[str, dict[str, np.ndarray]] = {}
    for label, fraction in SHIFT_SPECS:
        sx, sy = trajectory(telescope, grid + fraction * DT_SEC)
        dx = sx - x
        dy = sy - y
        radial = np.hypot(dx, dy)
        along = np.full(grid.shape, np.nan, dtype=np.float64)
        cross = np.full(grid.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(radial) & np.isfinite(speed) & (speed > 0.0)
        along[valid] = (dx[valid] * vx[valid] + dy[valid] * vy[valid]) / speed[valid]
        cross[valid] = (-dx[valid] * vy[valid] + dy[valid] * vx[valid]) / speed[valid]
        shifts[label] = {"dx": dx, "dy": dy, "radial": radial, "along": along, "cross": cross}

    digest = hashlib.sha256()
    for name, values, dtype in (
        ("grid", grid, "<f8"), ("x", x, "<f8"), ("y", y, "<f8"),
        ("vx", vx, "<f8"), ("vy", vy, "<f8"), ("ax", ax, "<f8"),
        ("ay", ay, "<f8"), ("direction", direction, "<u1"),
        ("speed_bin", speed_bin, "<u1"), ("accel_bin", accel_bin, "<u1"),
    ):
        digest.update(name.encode())
        digest.update(normalized_array_bytes(values, dtype))
    return {
        "slots": slots, "grid": grid, "x": x, "y": y,
        "vx": vx, "vy": vy, "ax": ax, "ay": ay,
        "speed": speed, "acceleration": acceleration,
        "direction": direction, "speed_bin": speed_bin, "accel_bin": accel_bin,
        "accel_edges": accel_edges, "holds": holds, "outside": outside,
        "composite": composite, "raster_direction": raster_direction,
        "shifts": shifts, "sha256": digest.hexdigest(),
    }


def key_arrays(
    support: np.ndarray, trajectory_info: dict[str, Any], indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    holds = trajectory_info["holds"]
    names = [
        "support", "direction", "speed_bin", "accel_bin",
        "left_word", "right_word", "legacy_linear_any", "left_any", "right_any",
        "left_bit8", "right_bit8", "transition_bracket", "outside", "composite",
        "raster_direction",
    ]
    arrays = [
        support.astype(np.uint8), trajectory_info["direction"][indices],
        trajectory_info["speed_bin"][indices], trajectory_info["accel_bin"][indices],
        holds["left_word"][indices].astype(np.uint8),
        holds["right_word"][indices].astype(np.uint8),
        holds["legacy_linear_any"][indices].astype(np.uint8),
        holds["left_any"][indices].astype(np.uint8), holds["right_any"][indices].astype(np.uint8),
        holds["left_bit8"][indices].astype(np.uint8), holds["right_bit8"][indices].astype(np.uint8),
        holds["transition_bracket"][indices].astype(np.uint8),
        trajectory_info["outside"][indices], trajectory_info["composite"][indices],
        trajectory_info["raster_direction"][indices],
    ]
    if any(np.max(values) > 255 for values in arrays):
        raise RuntimeError("group key exceeds uint8 domain")
    structured = np.rec.fromarrays(arrays, names=names, formats=["u1"] * len(names))
    return structured, names


def decode_key(record: Any) -> dict[str, Any]:
    direction = int(record["direction"])
    speed_bin = int(record["speed_bin"])
    accel_bin = int(record["accel_bin"])
    outside = int(record["outside"])
    composite = int(record["composite"])
    raster = int(record["raster_direction"])
    return {
        "support_class": "governing_supported_ordinary" if int(record["support"]) == 0 else "candidate_union_edge_baseline_unavailable",
        "scan_direction_8sector": DIRECTION_LABELS[direction] if direction < 8 else "unavailable_zero_or_unsupported_velocity",
        "scan_speed_bin_arcsec_per_sec": SPEED_LABELS[speed_bin] if speed_bin < 4 else "unavailable",
        "acceleration_quartile_from_governing_trajectory": f"q{accel_bin + 1}" if accel_bin < 4 else "unavailable",
        "hold_left_raw_word": int(record["left_word"]),
        "hold_right_raw_word": int(record["right_word"]),
        "hold_legacy_linear_any_nonzero": bool(record["legacy_linear_any"]),
        "hold_left_raw_any_nonzero": bool(record["left_any"]),
        "hold_right_raw_any_nonzero": bool(record["right_any"]),
        "hold_left_bit_0x08": bool(record["left_bit8"]),
        "hold_right_bit_0x08": bool(record["right_bit8"]),
        "hold_transition_bracket": bool(record["transition_bracket"]),
        "outside_map_box": ("false", "true", "not_applicable_non_raster")[outside],
        "composite_governing_compatibility_state": ("false", "true", "not_applicable_non_raster")[composite],
        "raster_scan_angle_direction": ("negative", "positive", "exact_zero", "not_applicable_or_unavailable")[raster],
    }


def aggregate_groups(
    obs: dict[str, Any], stream: dict[str, Any], keys: np.ndarray,
    raw_residual: np.ndarray, assigned_delta: np.ndarray,
    dx: np.ndarray, dy: np.ndarray, along: np.ndarray, cross: np.ndarray,
    radial: np.ndarray, scan_speed: np.ndarray, acceleration: np.ndarray,
) -> list[dict[str, Any]]:
    unique, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=unique.size)
    starts = np.concatenate(([0], np.cumsum(counts)))
    rows: list[dict[str, Any]] = []
    for group_index, record in enumerate(unique):
        selected = order[starts[group_index]:starts[group_index + 1]]
        row = {
            "obsnum": obs["obsnum"], "mode": obs["mode"],
            "selection_config_ids_json": canonical_json(obs["selection_config_ids"]),
            "selection_roles_json": canonical_json(obs["selection_roles"]),
            "interface": f"toltec{stream['interface']}", "array": stream["array"],
            **decode_key(record), "row_count": int(selected.size),
            "physical_integration_centroid_error_available_rows": 0,
            "physical_integration_centroid_error_unavailable_rows": int(selected.size),
        }
        row.update(finite_stats(raw_residual[selected], "raw_timestamp_minus_candidate_slot_sec"))
        row.update(finite_stats(raw_residual[selected] / DT_SEC, "raw_timestamp_minus_candidate_slot_cells"))
        row.update(finite_stats(scan_speed[selected], "actual_scan_speed_arcsec_per_sec"))
        row.update(finite_stats(acceleration[selected], "actual_acceleration_arcsec_per_sec2"))
        row.update(finite_stats(assigned_delta[selected], "candidate_minus_governing_assigned_time_sec"))
        row.update(finite_stats(dx[selected], "candidate_minus_governing_az_tangent_arcsec"))
        row.update(finite_stats(dy[selected], "candidate_minus_governing_el_tangent_arcsec"))
        row.update(finite_stats(along[selected], "candidate_minus_governing_along_scan_arcsec"))
        row.update(finite_stats(cross[selected], "candidate_minus_governing_cross_scan_arcsec"))
        row.update(finite_stats(radial[selected], "candidate_minus_governing_radial_arcsec"))
        row["one_arcsec_continuous_radial_pixel_delta_max"] = (
            float(np.nanmax(radial[selected])) if np.any(np.isfinite(radial[selected])) else ""
        )
        row["two_arcsec_continuous_radial_pixel_delta_max"] = (
            float(np.nanmax(radial[selected]) / 2.0) if np.any(np.isfinite(radial[selected])) else ""
        )
        comparable_radial = radial[selected][np.isfinite(radial[selected])]
        nearest_cell_result: int | str = (
            "" if comparable_radial.size == 0 else
            0 if np.all(comparable_radial == 0.0) else "requires_exact_baseline_WCS"
        )
        row["nearest_cell_identity_change_count_at_1_arcsec_any_fixed_baseline_WCS"] = nearest_cell_result
        row["nearest_cell_identity_change_count_at_2_arcsec_any_fixed_baseline_WCS"] = nearest_cell_result
        rows.append(row)
    return rows


def scenario_summary(obsnum: int, residuals: np.ndarray) -> list[dict[str, Any]]:
    formulas = {
        "timestamp_is_integration_start": -residuals - HALF_DT_SEC,
        "timestamp_is_effective_centroid": -residuals,
        "timestamp_is_integration_end": -residuals + HALF_DT_SEC,
    }
    rows = []
    for hypothesis, values in formulas.items():
        row = {
            "obsnum": obsnum, "hypothesis": hypothesis,
            "authority_status": "COUNTERFACTUAL_ONLY_PRODUCER_SEMANTIC_UNPROVED",
            "definition": "candidate_assigned_time_minus_hypothetical_physical_integration_centroid",
            "additional_unproved_assumption": "8.192_ms_cadence_is_a_contiguous_8.192_ms_integration_duration",
            "acceptance_use": "prohibited",
        }
        row.update(finite_stats(values, "time_error_sec"))
        rows.append(row)
    return rows


def build_reference_rows() -> list[dict[str, Any]]:
    rows = []
    for speed in (50.0, 100.0, 200.0):
        for label, fraction in (("half_slot", 0.5), ("full_slot", 1.0)):
            displacement = speed * DT_SEC * fraction
            rows.append({
                "native_rate_factor": "1x", "cadence_sec": DT_SEC,
                "reference_speed_arcsec_per_sec": speed, "slot_change": label,
                "absolute_time_change_sec": DT_SEC * fraction,
                "constant_speed_first_order_displacement_arcsec": displacement,
                "displacement_in_1_arcsec_pixels": displacement,
                "displacement_in_2_arcsec_pixels": displacement / 2.0,
                "interpretation": "dimensional sensitivity reference only; not an acceptance tolerance or actual curved trajectory result",
            })
    return rows


def protocol() -> dict[str, Any]:
    return {
        "protocol_id": "ALIGN-P0-D005-SKY-001",
        "status": "PREREGISTERED_OWNER_RETURN_REQUIRED_PHASE_ONE_UNAUTHORIZED",
        "successor_output_viewed": False,
        "parent_D005_owner_return_preserved": {
            "question_ids": [f"D005-Q{index}" for index in range(1, 9)],
            "status": "unchanged and incorporated by reference; this sky amendment does not answer or replace them",
        },
        "cohort": {
            "unique_observations": [148669, 148670, 148671, 152389, 152391, 152393],
            "mandatory_core": ["Pointing 152389", "Beammap 148670"],
            "deduplication": "Pointing 152389 point_core/science_support references are byte-identical and measured once; both roles retained",
            "native_rate": "genuine 1x only; 0.5x/2x/4x remain evidence-pending",
        },
        "three_distinct_time_quantities": [
            {
                "id": "raw_timestamp_to_slot_residual",
                "definition": "r=t_native_after_offset-t_candidate_slot",
                "role": "engineering placement diagnostic and strict unique-slot admission; not sky accuracy",
            },
            {
                "id": "governing_to_candidate_assigned_time_change",
                "definition": "delta_t=t_candidate_slot-t_governing_slot for governing-supported ordinary rows",
                "role": "differential sky-coordinate driver; union-edge rows are unpaired/unavailable",
            },
            {
                "id": "assigned_time_to_physical_integration_centroid_error",
                "definition": "epsilon=t_assigned-t_physical_integration_centroid",
                "role": "absolute physical placement; unavailable until producer event/window authority exists",
            },
        ],
        "trajectory_method": {
            "frame": "governing AltAz tangent plane; +x azimuth-like, +y elevation-like; arcsec",
            "ordering": "periodic-fix each native field; interpolate six fields at each assigned time; then apply exact governing tangent formula",
            "velocity": "v=[R(t+dt/2)-R(t-dt/2)]/dt on bracket-supported samples",
            "acceleration": "a=[R(t+dt/2)-2R(t)+R(t-dt/2)]/(dt/2)^2",
            "along_unit": "v/|v|",
            "cross_unit": "(-v_y,v_x)/|v|; positive is +90 degrees in tangent plane",
            "direction": "atan2(v_y,v_x) in eight fixed half-open 45-degree sectors",
            "speed_bins_arcsec_per_sec": ["[0,50)", "[50,100)", "[100,200)", "[200,infinity)"],
            "acceleration_bins": "per-observation governing-trajectory p25/p50/p75, digest-bound before successor",
            "estimator_window_is_tolerance": False,
        },
        "science_facing_metrics": [
            {
                "id": "sample_sky_coordinate_displacement",
                "definition": "paired delta x/y, radial, signed along/cross for every ordinary row",
                "aggregation": "record digest plus populated strata by observation, direction, interface, array, speed, acceleration, Hold views, support",
                "limit": "exact zero under active Point/Beammap successor policy; any nonzero result stops for owner review",
            },
            {
                "id": "systematic_offset_and_random_scatter",
                "definition": "signed mean/median plus population std and 1.4826*MAD of along/cross by interface and direction",
                "limit": "exact zero for this behavior-preserving repair; no nonzero angular tolerance authorized",
            },
            {
                "id": "one_and_two_arcsec_pixel_sensitivity",
                "definition": "continuous delta/p plus nearest-cell identity on exact fixed baseline WCS",
                "limit": "zero coordinate-induced change; 2-arcsec view is counterfactual sensitivity, not a new map configuration",
            },
            {
                "id": "gridding_weight_sentinel",
                "definition": "with exact baseline WCS/operator/signal/eligibility/weights held fixed, compare stencil keys and normalized per-sample stencil L1 delta",
                "limit": "exact unchanged under active map-product gate",
                "boundary": "SCI-MAP owns JINC/naive gridding and map weights; ALIGN may only supply coordinates and run a bounded downstream sentinel",
            },
            {
                "id": "source_crossing",
                "definition": "join UID/array/validity/stable scan; lexicographic argmin(distance^2,assigned_time,row); compare row/time/coordinate/distance",
                "limit": "exact unchanged; current historical artifact lacks closest row/time, so direct successor comparison remains pending",
            },
            {
                "id": "centroid",
                "definition": "Pointing per-array and Beammap UID/array/validity joined signed x/y and radial displacement",
                "limit": "exact unchanged under active policy; fit uncertainties are descriptive, not regression allowance",
            },
            {
                "id": "PSF",
                "definition": "major=max(a_fwhm,b_fwhm), minor=min(...), ellipticity=(major-minor)/(major+minor)",
                "limit": "exact unchanged under active policy; historical 0.0001-arcsec profile does not widen it",
            },
            {
                "id": "full_slot_reassignment_rate_and_wings",
                "definition": "R_full=N(abs(delta_slot)>=1)/N(governing-supported ordinary); union edges excluded",
                "limit": "exactly zero; any event stops",
                "required_stop_diagnostic": "fixed baseline centroid/orientation/FWHM ellipse; absolute map residual in rho<=1, 1<rho<=2, 2<rho<=4 normalized by baseline core absolute signal; no refit",
            },
        ],
        "engineering_invariants": [
            "strict abs(raw residual)<dt/2 is the unique nearest-slot admission rule only",
            "round-half-up is used once and identically for assignment/mask/identity",
            "zero ordinary slot changes, collisions, losses, or duplicates",
            "union-edge rows remain typed unavailable in the governing baseline and are not reassignment events",
        ],
        "explicit_prohibitions": [
            "do not use half-cell boundary distance or observed residual maximum as a sky tolerance",
            "do not infer start/end/centroid semantics from cadence, receive lag, or consumer behavior",
            "do not change AST frame/projection/detector geometry",
            "do not implement or tune SCI-MAP gridding/mapmaking",
            "do not select a Hold predicate or transition side",
            "do not authorize phase one, Unity, production expansion, re-audit, merge, rebase, or push",
        ],
        "stop_and_owner_return": [
            "any nonzero ordinary assigned-time or sky-coordinate change",
            "any changed active Point/Beammap product, crossing, centroid, FWHM, ellipticity, pixel identity, gridding key/weight, or PSF wing",
            "any full-slot reassignment",
            "any claim of absolute sky correctness without a versioned producer timestamp/window contract",
            "missing native 0.5x/2x/4x evidence remains pending and is not a pass",
        ],
    }


def owner_brief(measured: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "ALIGN-P0-D005-SKY-001",
        "recommendation": "RECORD_SKY_DOMAIN_REFRAME_AND_RETURN_TO_OWNER; PHASE_ONE_UNAUTHORIZED",
        "engineering_invariants": [
            "D001 legacy reconstruction and D002 round-half-up assignment are deterministic compatibility rules.",
            "The strict half cell admits a unique nearest slot; it is not an angular accuracy tolerance.",
            f"All {measured['ordinary_rows']} governing-supported ordinary rows retain exact slot/time identity; {measured['edge_rows']} union-edge rows are unpaired baseline-unavailable.",
            "The Pointing 152389 34.062668-us half-cell margin is only distance to an engineering decision boundary.",
        ],
        "measured_angular_non_degradation": [
            f"All {measured['ordinary_rows']} paired rows have exactly zero assigned-time, az-tangent, el-tangent, radial, along-scan, and cross-scan change.",
            "Systematic offsets and random scatter are exactly zero in every populated observation/interface/array/direction/speed/acceleration/Hold/support ordinary stratum.",
            "The rare full-slot reassignment rate is exactly zero; union-edge admission is not reassignment.",
            "Zero coordinate change implies zero coordinate-induced 1-arcsec/2-arcsec cell or gridding-key change when baseline geometry/operator/other inputs are fixed; no mapmaker was executed.",
            "Source-crossing/centroid/PSF successor products were not generated; their active exact downstream gates remain preregistered, not claimed as executed measurements.",
        ],
        "unresolved_physical_timestamp_authority": [
            "No selected detector header or local producer contract identifies the timestamp as integration start, end, effective centroid, FPGA capture, packet formation, or another event; cadence also does not prove a contiguous exposure duration.",
            "Consumer use without a +/-4.096-ms shift proves governing behavior only.",
            "Positive RecvTime lag is an undocumented end/completion hypothesis, not authority.",
            "Assigned-time-to-physical-integration-centroid error and absolute physical sky-placement correctness remain unavailable.",
            "Telescope time physical-event/absolute-precision authority also remains unproved under D004.",
        ],
        "ALIGN_AST_SCI_MAP_boundary": {
            "ALIGN": "native row identity, offsets, assigned time, aligned support/validity and per-sample aligned fields",
            "AST": "frozen conversion of aligned telescope fields and detector geometry into sky/tangent coordinates",
            "SCI-MAP": "WCS coordinate-to-pixel projection, JINC/naive coefficients, sample accumulation, weights, coverage and coaddition",
            "bounded_validation": "Pointing/Beammap map products may be exact downstream sentinels; this repair does not implement mapmaking",
        },
        "owner_questions_required_before_phase_one": [
            {
                "id": "SKY-Q1-PHYSICAL-TIME",
                "question": "Will the owner require a versioned producer ICD defining both observed compile identities' integration window, latch event, clock relation, pipeline latency, and RecvTime association before phase one, or explicitly proceed with relative compatibility while absolute sky correctness remains unresolved?",
            },
            {
                "id": "SKY-Q2-ANGULAR-GATE",
                "question": "Does the owner approve exact zero ordinary assigned-time/sky displacement and the existing exact Point/Beammap downstream policy as the phase-one gate, with any nonzero angular allowance requiring a separate pre-candidate authority amendment?",
            },
            {
                "id": "SKY-Q3-MAP-SENTINEL",
                "question": "Does the owner require the bounded fixed-WCS/fixed-JINC 1-arcsec map sentinel and counterfactual 2-arcsec coordinate sensitivity in phase-one validation, while keeping all gridding implementation under SCI-MAP?",
            },
            {
                "id": "SKY-Q4-MISSING-RATES",
                "question": "May any future phase one be restricted to native 1x while 0.5x/2x/4x sky-domain observational evidence remains pending, or are all four native strata prerequisites?",
            },
            {
                "id": "SKY-Q5-HOLD",
                "question": "Will the owner resolve the existing D005 Hold/scan predicate and transition-side return separately, without using the zero angular slot result to select either hypothesis?",
            },
        ],
        "parent_D005_owner_questions_still_required": {
            "question_ids": [f"D005-Q{index}" for index in range(1, 9)],
            "status": "unchanged; SKY questions refine timing/sky criteria and do not supersede the frozen parent return",
        },
        "phase_one_authorization": "NONE",
    }


def report_text(
    identity: dict[str, Any], observation_rows: list[dict[str, Any]],
    measured: dict[str, Any], brief: dict[str, Any], semantics: dict[str, Any],
) -> str:
    observation_lines = []
    for row in observation_rows:
        observation_lines.append(
            f"| {row['obsnum']} | {row['mode']} | {row['native_rows']:,} | "
            f"{row['ordinary_rows']:,} | {row['edge_rows']} | {row['speed_row_weighted_p50_arcsec_per_sec']:.6f} | "
            f"{row['speed_row_weighted_p95_arcsec_per_sec']:.6f} | {row['speed_row_weighted_max_arcsec_per_sec']:.6f} | "
            f"{row['plus_half_radial_row_weighted_p50_arcsec']:.6f} | {row['plus_half_radial_row_weighted_p99_arcsec']:.6f} | "
            f"{row['plus_full_radial_row_weighted_p50_arcsec']:.6f} | {row['plus_full_radial_row_weighted_p99_arcsec']:.6f} |"
        )
    return f"""# SCI-ALIGN-001 ALIGN-P0-D005 sky-domain amendment

Date: 2026-08-02

Verdict: **OWNER RETURN REQUIRED — PHASE ONE UNAUTHORIZED**

## Scope and identity

- Repair branch `{identity['branch']}`; additive task base `{identity['task_base_commit']}`.
- Governing application `{identity['governing_application_sha']}`; frozen phase-zero evidence `{identity['phase_zero_evidence_commit']}`.
- Frozen D005 `SHA256SUMS` `{identity['frozen_D005_sha256sums_sha256']}` was verified and not rewritten.
- Read-only coordination authority is the immutable `{identity['coordination_frozen_head']}` snapshot, verified clean by the frozen parent D005 package. The live coordination checkout was deliberately excluded; no live HEAD, status, or working-tree bytes enter this package.
- This package reads every canonical selected Pointing/Beammap timing row and records compact digests/strata. It does not execute Citlali, TolProj, AST changes, SCI-MAP, or Unity; inspect a successor; edit application code or sibling repositories; or authorize phase one.

## Three timing quantities are not interchangeable

1. `raw_timestamp_minus_candidate_slot` is the signed D001 reconstructed timestamp residual after the exactly-once requested offset. The strict half-cell rule is an engineering unique-slot invariant.
2. `candidate_minus_governing_assigned_time` is the differential time that can move a sample on the governing telescope trajectory.
3. `assigned_time_minus_physical_integration_centroid` is absolute physical timing. It is unavailable for every selected row because no producer authority identifies start, end, centroid, capture, or packet event.

The Pointing 152389 `34.062668 us` margin is only distance to the half-cell decision boundary. It is not sky-placement accuracy and is not converted into an acceptance tolerance.

## Measured differential sky result

The six canonical observations contain **{measured['native_rows']:,}** native detector rows. All **{measured['ordinary_rows']:,}** governing-supported ordinary rows retain the same integer slot and assigned time, so their candidate-minus-governing AltAz tangent coordinates and signed along/cross components are exactly **0 arcsec**. All **{measured['edge_rows']}** union-edge rows have no governing assigned-time baseline and are `unavailable`, not zero. The full-slot reassignment rate is exactly `0/{measured['ordinary_rows']}`.

For each requested time the diagnostic mirrors governing order: periodic-fix and interpolate `TelAzAct`, `TelElAct`, `TelAzCor`, `TelElCor`, `SourceAz`, and `SourceEl`, then form the AltAz tangent coordinate. Velocity and acceleration use the digest-bound symmetric half-cadence trajectory estimator. Hold raw words, left/right views, bit `0x08`, outside-map state, and composite compatibility state remain separate; none is called a physical turn.

| Obs | Mode | Native rows | Paired ordinary | Edge N/A | Speed p50 | Speed p95 | Speed max | +half radial p50 | +half radial p99 | +full radial p50 | +full radial p99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(observation_lines)}

Speeds are digest-bound symmetric finite-difference estimates from the actual tangent trajectory, and the shift columns are exact curved-trajectory sensitivity calculations, row-weighted over selected detector rows. They are descriptive, not producer-authoritative kinematics or limits.

## Half/full-slot dimensional references

At 1x, half slot is `4.096 ms` and full slot is `8.192 ms`:

| Reference speed | Half slot | Full slot | Half in 1/2 arcsec pixels | Full in 1/2 arcsec pixels |
| ---: | ---: | ---: | ---: | ---: |
| 50 arcsec/s | 0.2048 arcsec | 0.4096 arcsec | 0.2048 / 0.1024 | 0.4096 / 0.2048 |
| 100 arcsec/s | 0.4096 arcsec | 0.8192 arcsec | 0.4096 / 0.2048 | 0.8192 / 0.4096 |
| 200 arcsec/s | 0.8192 arcsec | 1.6384 arcsec | 0.8192 / 0.4096 | 1.6384 / 0.8192 |

These are constant-speed dimensional examples only. Actual comparisons use the trajectory table above and the full populated-stratum artifact.

## Physical timestamp authority

All selected detector files expose `FpgaFreq=256000000 Hz`, `AccumLen=2097152`, and `SampleFreq=122.0703125 Hz`, proving the 8.192-ms cadence relation but not the integration event or a contiguous 8.192-ms exposure duration. The timestamp/receive variables contain no start/end/centroid, bounds, exposure, or cell-method authority. There are {semantics['compile_identity_count']} producer compile identities and no local versioned ICD for either. `RecvTime - reconstructed_time` is nonnegative in the selected cohort (median `{semantics['receive_lag_median_sec'] * 1e6:.3f} us`), but its clock and row association are undocumented, so it cannot select end/completion semantics.

Counterfactual start/centroid/end formulas are recorded in `physical_timestamp_scenarios.csv`; they are not acceptance evidence. Absolute assigned-time-to-integration-centroid error and absolute physical sky-placement correctness remain unresolved even though the differential slot result is exact.

## Science-facing gates and ownership

- Sample coordinate, systematic/scatter, 1/2-arcsec pixel sensitivity, source crossing, centroid, major/minor FWHM, ellipticity, full-slot rate, and wing metrics are preregistered in `preregistration_protocol.json`.
- Current Point/Beammap policy remains exact complete-product/TOD equality for unaffected behavior. The historical `0.0001 arcsec` OG/refactor Beammap profile is not promoted into this successor gate.
- ALIGN owns row identity, offsets, assigned time, aligned support/validity, and aligned telescope fields. AST retains the frozen sky transform. SCI-MAP owns WCS projection, JINC/naive coefficients, map weights, coverage, and accumulation.
- Exact map-level Pointing/Beammap comparison may be a bounded downstream sentinel. No gridding or mapmaking implementation is in this repair.
- No successor was run, so source-crossing, centroid, PSF, ellipticity, and map products are preregistered downstream gates rather than claimed executed non-degradation results.

## Owner return

Engineering invariants, measured angular equivalence, and unresolved physical timestamp authority are separated in `owner_decision_brief.json`. The owner must answer SKY-Q1 through SKY-Q5, and the frozen parent D005-Q1 through D005-Q8 remain unchanged and incorporated by reference. In particular, no nonzero angular tolerance can be derived from cadence, half-cell margin, or the measured residual maximum.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--coordination-repo", type=Path,
        default=Path("/private/tmp/citlali-scientific-audit-framework"),
    )
    parser.add_argument(
        "--suite-root", type=Path,
        default=Path("/Users/gwilson/work_toltec/local_data/citlali-validation/v1"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    coordination_repo = args.coordination_repo.resolve()
    output = (args.output or repo / "validation" / PACKAGE_NAME).resolve()
    identity, authority_rows = validate_identity(repo, coordination_repo)
    observations, input_manifest = load_selected_observations(repo)
    output.mkdir(parents=True, exist_ok=True)

    strata_rows: list[dict[str, Any]] = []
    digest_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    semantics_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    total_native = total_ordinary = total_edge = total_full_slot = 0
    all_lags: list[np.ndarray] = []
    compile_identities: set[int] = set()

    for obs in observations:
        streams = [
            read_detector(obs["detectors"][interface], obs["obsnum"], interface)
            for interface in EXPECTED_INTERFACES
        ]
        telescope = read_telescope(obs)
        first = [float(stream["times"][0]) for stream in streams]
        last = [float(stream["times"][-1]) for stream in streams]
        phase = max(first)
        overlap_end = min(last)
        current_count = int((overlap_end - phase) / DT_SEC) + 1
        candidate_slots_by_stream = []
        union_min: int | None = None
        union_max: int | None = None
        for stream in streams:
            q = (stream["times"] - phase) / DT_SEC
            candidate = np.floor(q + 0.5).astype(np.int64)
            candidate_slots_by_stream.append(candidate)
            union_min = int(np.min(candidate)) if union_min is None else min(union_min, int(np.min(candidate)))
            union_max = int(np.max(candidate)) if union_max is None else max(union_max, int(np.max(candidate)))
        assert union_min is not None and union_max is not None
        trajectory_info = build_observation_trajectory(
            obs, telescope, phase, current_count, union_min, union_max,
        )
        slot_weights = np.zeros(union_max - union_min + 1, dtype=np.int64)
        observation_residuals: list[np.ndarray] = []
        obs_native = obs_ordinary = obs_edge = obs_full_slot = 0

        for stream, candidate in zip(streams, candidate_slots_by_stream, strict=True):
            times = stream["times"]
            q = (times - phase) / DT_SEC
            residual = times - (phase + candidate.astype(np.float64) * DT_SEC)
            admitted = np.abs(residual) < HALF_DT_SEC
            if not admitted.all():
                raise RuntimeError(f"strict half-cell violation in {stream['path']}")
            if np.unique(candidate).size != candidate.size:
                raise RuntimeError(f"slot collision in {stream['path']}")
            baseline = cxx_round(q)
            clipped = np.clip(baseline, 0, current_count - 1)
            baseline_residual = times - (phase + clipped.astype(np.float64) * DT_SEC)
            baseline_valid = (
                (baseline >= 0) & (baseline < current_count)
                & (np.abs(baseline_residual) <= HALF_DT_SEC)
            )
            edge = ~baseline_valid
            ordinary_change = baseline_valid & (baseline != candidate)
            grid_indices = candidate - union_min
            if np.any(grid_indices < 0) or np.any(grid_indices >= trajectory_info["grid"].size):
                raise RuntimeError("candidate slot outside union trajectory")
            slot_weights += np.bincount(grid_indices, minlength=slot_weights.size)

            assigned_delta = np.full(times.shape, np.nan, dtype=np.float64)
            assigned_delta[baseline_valid] = (candidate[baseline_valid] - baseline[baseline_valid]) * DT_SEC
            dx = np.full(times.shape, np.nan, dtype=np.float64)
            dy = np.full(times.shape, np.nan, dtype=np.float64)
            ordinary_candidate_idx = grid_indices[baseline_valid]
            ordinary_baseline_idx = baseline[baseline_valid] - union_min
            dx[baseline_valid] = (
                trajectory_info["x"][ordinary_candidate_idx]
                - trajectory_info["x"][ordinary_baseline_idx]
            )
            dy[baseline_valid] = (
                trajectory_info["y"][ordinary_candidate_idx]
                - trajectory_info["y"][ordinary_baseline_idx]
            )
            radial = np.hypot(dx, dy)
            along = np.full(times.shape, np.nan, dtype=np.float64)
            cross = np.full(times.shape, np.nan, dtype=np.float64)
            speed = trajectory_info["speed"][grid_indices]
            acceleration = trajectory_info["acceleration"][grid_indices]
            vx = trajectory_info["vx"][grid_indices]
            vy = trajectory_info["vy"][grid_indices]
            directional = baseline_valid & np.isfinite(speed) & (speed > 0.0)
            along[directional] = (dx[directional] * vx[directional] + dy[directional] * vy[directional]) / speed[directional]
            cross[directional] = (-dx[directional] * vy[directional] + dy[directional] * vx[directional]) / speed[directional]
            support_code = edge.astype(np.uint8)
            keys, _ = key_arrays(support_code, trajectory_info, grid_indices)
            strata_rows.extend(aggregate_groups(
                obs, stream, keys, residual, assigned_delta, dx, dy, along, cross,
                radial, speed, acceleration,
            ))

            baseline_for_digest = np.where(baseline_valid, baseline, np.iinfo(np.int64).min)
            row_digest = hashlib.sha256()
            row_digest.update(f"{obs['obsnum']}:toltec{stream['interface']}".encode())
            for name, values, dtype in (
                ("raw_time", times, "<f8"), ("candidate_slot", candidate, "<i8"),
                ("baseline_valid", baseline_valid, "<u1"),
                ("baseline_slot", baseline_for_digest, "<i8"),
                ("raw_residual", residual, "<f8"), ("assigned_delta", assigned_delta, "<f8"),
                ("dx", dx, "<f8"), ("dy", dy, "<f8"),
                ("along", along, "<f8"), ("cross", cross, "<f8"),
                ("group_key", keys, keys.dtype.str if keys.dtype.fields is None else "|V15"),
            ):
                row_digest.update(name.encode())
                if name == "group_key":
                    row_digest.update(np.ascontiguousarray(values).tobytes())
                else:
                    row_digest.update(normalized_array_bytes(values, dtype))
            digest_rows.append({
                "obsnum": obs["obsnum"], "interface": f"toltec{stream['interface']}",
                "array": stream["array"], "native_rows": stream["rows"],
                "governing_supported_ordinary_rows": int(baseline_valid.sum()),
                "candidate_union_edge_rows": int(edge.sum()),
                "ordinary_changed_slot_rows": int(ordinary_change.sum()),
                "full_slot_reassignment_rows": int(np.sum(baseline_valid & (np.abs(candidate - baseline) >= 1))),
                "physical_centroid_error_available_rows": 0,
                "physical_centroid_error_unavailable_rows": stream["rows"],
                "raw_full_file_sha256_from_frozen_D005": stream["full_file_sha256"],
                "raw_relevant_variables_sha256": stream["relevant_variables_sha256"],
                "trajectory_relevant_variables_sha256": telescope["relevant_variables_sha256"],
                "trajectory_grid_derived_sha256": trajectory_info["sha256"],
                "canonical_row_comparison_sha256": row_digest.hexdigest(),
                "row_digest_serialization": "identity UTF-8 then named little-endian arrays; NaN retained; structured group key packed uint8",
            })
            for native_index in np.flatnonzero(edge | ordinary_change):
                index = int(native_index)
                candidate_index = int(grid_indices[index])
                exception_rows.append({
                    "obsnum": obs["obsnum"], "interface": f"toltec{stream['interface']}",
                    "array": stream["array"], "native_row_0based": index,
                    "exception_class": (
                        "candidate_union_edge_baseline_unavailable" if edge[index]
                        else "ordinary_slot_change"
                    ),
                    "raw_time_after_offset_sec": float(times[index]),
                    "raw_timestamp_minus_candidate_slot_sec": float(residual[index]),
                    "candidate_slot_k": int(candidate[index]),
                    "candidate_assigned_time_sec": float(phase + candidate[index] * DT_SEC),
                    "candidate_az_tangent_arcsec": float(trajectory_info["x"][candidate_index]),
                    "candidate_el_tangent_arcsec": float(trajectory_info["y"][candidate_index]),
                    "governing_slot_k": int(baseline[index]) if baseline_valid[index] else "",
                    "governing_assigned_time_sec": (
                        float(phase + baseline[index] * DT_SEC) if baseline_valid[index] else ""
                    ),
                    "candidate_minus_governing_sky": "unavailable" if edge[index] else "nonzero_stop",
                })

            lag = stream["recv_lag"]
            all_lags.append(lag)
            compile_identities.add(stream["compile_time"])
            semantics_rows.append({
                "obsnum": obs["obsnum"], "interface": f"toltec{stream['interface']}",
                "array": stream["array"], "native_rows": stream["rows"],
                "producer_compile_time_unix": stream["compile_time"],
                "global_attribute_names_json": canonical_json(sorted(stream["global_attrs"])),
                "Data_Toltec_Ts_attributes_json": canonical_json(stream["ts_attrs"]),
                "Data_Toltec_RecvTime_attributes_json": canonical_json(stream["recv_attrs"]),
                "producer_semantic_terms_found_json": canonical_json(stream["semantic_terms"]),
                "integration_event_semantic": "UNPROVED_START_END_CENTROID_OR_OTHER",
                "receive_lag_min_sec": float(np.min(lag)),
                "receive_lag_p50_sec": float(np.quantile(lag, 0.50)),
                "receive_lag_p99_sec": float(np.quantile(lag, 0.99)),
                "receive_lag_max_sec": float(np.max(lag)),
                "negative_receive_lag_rows": int(np.sum(lag < 0.0)),
                "interpretation": "positive lag is an undocumented end/completion hypothesis only",
            })
            observation_residuals.append(residual)
            n_native = stream["rows"]
            n_ordinary = int(baseline_valid.sum())
            n_edge = int(edge.sum())
            n_full = int(np.sum(baseline_valid & (np.abs(candidate - baseline) >= 1)))
            obs_native += n_native
            obs_ordinary += n_ordinary
            obs_edge += n_edge
            obs_full_slot += n_full

        expected = EXPECTED_OBSERVATION_COUNTS[obs["obsnum"]]
        if (obs_native, obs_ordinary, obs_edge) != expected or obs_full_slot != 0:
            raise RuntimeError(
                f"observation row identity changed for {obs['obsnum']}: "
                f"{(obs_native, obs_ordinary, obs_edge, obs_full_slot)}"
            )
        residuals = np.concatenate(observation_residuals)
        scenario_rows.extend(scenario_summary(obs["obsnum"], residuals))
        trajectory_row: dict[str, Any] = {
            "obsnum": obs["obsnum"], "mode": obs["mode"],
            "selection_config_ids_json": canonical_json(obs["selection_config_ids"]),
            "selection_roles_json": canonical_json(obs["selection_roles"]),
            "native_rows": obs_native, "ordinary_rows": obs_ordinary, "edge_rows": obs_edge,
            "full_slot_reassignment_rows": obs_full_slot,
            "full_slot_reassignment_rate": 0.0,
            "phase_sec": phase, "cadence_sec": DT_SEC,
            "current_grid_count": current_count, "union_min_slot": union_min,
            "union_max_slot": union_max, "union_grid_count": int(slot_weights.size),
            "acceleration_quartile_edges_arcsec_per_sec2_json": canonical_json(
                [float(value) for value in trajectory_info["accel_edges"]]
            ),
            "trajectory_grid_sha256": trajectory_info["sha256"],
        }
        speed_stats = weighted_stats(trajectory_info["speed"], slot_weights, "speed_row_weighted")
        accel_stats = weighted_stats(trajectory_info["acceleration"], slot_weights, "acceleration_row_weighted")
        trajectory_row.update({f"{name}_arcsec_per_sec": value for name, value in speed_stats.items()})
        trajectory_row.update({f"{name}_arcsec_per_sec2": value for name, value in accel_stats.items()})
        for label, _ in SHIFT_SPECS:
            shift_stats = weighted_stats(
                trajectory_info["shifts"][label]["radial"], slot_weights,
                f"{label}_radial_row_weighted",
            )
            trajectory_row.update({f"{name}_arcsec": value for name, value in shift_stats.items()})
        observation_rows.append(trajectory_row)
        total_native += obs_native
        total_ordinary += obs_ordinary
        total_edge += obs_edge
        total_full_slot += obs_full_slot

    if (total_native, total_ordinary, total_edge, total_full_slot) != (4645586, 4645476, 110, 0):
        raise RuntimeError("selected total row identity changed")
    if any(row["producer_semantic_terms_found_json"] != "[]" for row in semantics_rows):
        raise RuntimeError("unexpected producer integration semantic appeared")
    if any(row["negative_receive_lag_rows"] != 0 for row in semantics_rows):
        raise RuntimeError("selected receive-lag sign evidence changed")
    all_lag = np.concatenate(all_lags)
    semantics = {
        "selected_detector_file_count": len(semantics_rows),
        "selected_detector_row_count": int(all_lag.size),
        "compile_identities": sorted(compile_identities),
        "compile_identity_count": len(compile_identities),
        "receive_lag_min_sec": float(np.min(all_lag)),
        "receive_lag_median_sec": float(np.quantile(all_lag, 0.50)),
        "receive_lag_p99_sec": float(np.quantile(all_lag, 0.99)),
        "receive_lag_max_sec": float(np.max(all_lag)),
        "negative_receive_lag_rows": int(np.sum(all_lag < 0.0)),
        "producer_timestamp_event": "UNPROVED",
        "cadence_equals_contiguous_integration_duration": "UNPROVED",
        "physical_integration_centroid_error": "UNAVAILABLE_FOR_ALL_ROWS",
        "absolute_physical_sky_placement_correctness": "UNRESOLVED",
        "governing_consumer_behavior": "uses reconstructed timestamp directly without plus/minus half-cadence adjustment",
        "receive_lag_interpretation": "end/completion hypothesis only; clock and row association undocumented",
    }
    measured = {
        "native_rows": total_native, "ordinary_rows": total_ordinary,
        "edge_rows": total_edge, "full_slot_reassignment_rows": total_full_slot,
        "ordinary_slot_change_rows": sum(row["ordinary_changed_slot_rows"] for row in digest_rows),
        "ordinary_assigned_time_change_max_abs_sec": max(
            float(row["candidate_minus_governing_assigned_time_sec_max_abs"] or 0.0)
            for row in strata_rows if row["support_class"] == "governing_supported_ordinary"
        ),
        "ordinary_sky_change_max_abs_arcsec": max(
            float(row["candidate_minus_governing_radial_arcsec_max_abs"] or 0.0)
            for row in strata_rows if row["support_class"] == "governing_supported_ordinary"
        ),
        "populated_stratum_count": len(strata_rows),
        "canonical_interface_row_digest_count": len(digest_rows),
        "exception_row_count": len(exception_rows),
    }
    if measured["ordinary_slot_change_rows"] != 0 or measured["ordinary_sky_change_max_abs_arcsec"] != 0.0:
        raise RuntimeError(f"ordinary differential non-degradation failed: {measured}")

    digest_rows.sort(key=lambda row: (row["obsnum"], int(row["interface"].removeprefix("toltec"))))
    global_row_digest = sha256_bytes((canonical_json([
        {"obsnum": row["obsnum"], "interface": row["interface"],
         "rows": row["native_rows"], "sha256": row["canonical_row_comparison_sha256"]}
        for row in digest_rows
    ]) + "\n").encode())
    identity["selected_unique_observations"] = [row["obsnum"] for row in observation_rows]
    identity["selected_canonical_detector_files"] = len(digest_rows)
    identity["selected_native_rows"] = total_native
    identity["canonical_all_row_comparison_sha256"] = global_row_digest
    identity["dense_ordinary_row_artifact_written"] = False
    identity["bounded_exception_rows_written"] = len(exception_rows)

    brief = owner_brief(measured)
    prereg = protocol()
    reference_rows = build_reference_rows()
    write_json(output / "authority_and_identity.json", identity)
    write_csv(output / "authority_manifest.csv", [
        "authority_id", "authority_class", "path", "git_commit", "sha256", "use",
    ], authority_rows)
    write_csv(output / "selected_sky_input_manifest.csv", list(input_manifest[0]), input_manifest)
    write_csv(output / "row_comparison_digest_registry.csv", list(digest_rows[0]), digest_rows)
    write_csv(output / "sky_assignment_strata.csv", list(strata_rows[0]), strata_rows)
    write_csv(output / "alignment_exceptions.csv", list(exception_rows[0]), exception_rows)
    write_csv(output / "trajectory_observation_summary.csv", list(observation_rows[0]), observation_rows)
    write_csv(output / "timestamp_semantics_inventory.csv", list(semantics_rows[0]), semantics_rows)
    write_json(output / "timestamp_semantics_summary.json", semantics)
    write_csv(output / "physical_timestamp_scenarios.csv", list(scenario_rows[0]), scenario_rows)
    write_csv(output / "reference_slot_displacements.csv", list(reference_rows[0]), reference_rows)
    write_json(output / "measured_conclusions.json", measured)
    write_json(output / "preregistration_protocol.json", prereg)
    write_json(output / "owner_decision_brief.json", brief)
    (output / "REPORT.md").write_text(report_text(identity, observation_rows, measured, brief, semantics))

    artifact_names = sorted({
        "authority_and_identity.json", "authority_manifest.csv", "selected_sky_input_manifest.csv",
        "row_comparison_digest_registry.csv", "sky_assignment_strata.csv", "alignment_exceptions.csv",
        "trajectory_observation_summary.csv", "timestamp_semantics_inventory.csv",
        "timestamp_semantics_summary.json", "physical_timestamp_scenarios.csv",
        "reference_slot_displacements.csv", "measured_conclusions.json",
        "preregistration_protocol.json", "owner_decision_brief.json", "REPORT.md",
    })
    expected = set(artifact_names) | {"SHA256SUMS"}
    unexpected = sorted(path.name for path in output.iterdir() if path.is_file() and path.name not in expected)
    if unexpected:
        raise RuntimeError(f"unexpected preexisting sky-domain artifacts: {unexpected}")
    (output / "SHA256SUMS").write_text("\n".join(
        f"{sha256_file(output / name)}  {name}" for name in artifact_names
    ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
