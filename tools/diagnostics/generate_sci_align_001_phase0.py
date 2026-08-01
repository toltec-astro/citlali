#!/usr/bin/env python3
"""Generate the bounded SCI-ALIGN-001 phase-0 evidence package.

This is evidence-only tooling.  It reads repository sources and owner-local
NetCDF data, writes deterministic CSV/JSON artifacts, and never imports or
modifies application state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset


REPAIR_BASE = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
FROZEN_COORDINATION_HEAD = "846128c8ee6dc27851bd6c71aeecbe4739e1d24a"
HANDOFF_RECORD_COMMIT = "0309fd48a973a6e7e136224906ac49c02f0171be"
OWNER_DECISION_COMMIT = "4f905f4f353e91847a303f4f3959654f3f03c302"
CORRECTION_COMMIT = "35cc8ce246e8e70c569e650be6c1eae2c91b80ef"
GATE_SNAPSHOT_SHA256 = "75b085b8f7bfea3af7dbdc579a1efb8ce17423080ea873d74647c945d0519481"
GATE_TIME_SCOPED_REFS = {
    "codex/audit-sci-align-001": "9e234eada67c88feacddfc8b7e1afb0e1cffd818",
    "codex/audit-sci-cal-001": "27b0916e725696597c3ba84fb6a82bf6cf0ea356",
    "codex/audit-sci-map-001": "a675c2a54a50ed0c67b077e9c5d420933fa11ab0",
    "codex/convolve-contract-audit": "800e8ae433f87d3fb7521fcb1a7fdf1d32532949",
    "codex/convolve-contract-implementation": "2d1fbb4897e1fa416a587847895266abecb43100",
    "codex/convolve-noise-correction": "02a198cbfb379eaf6ab279c5a3d44ee73ff90435",
    "codex/perf-map-accumulation-noise-lifecycle": "d01ffa5981551345eeb1c765c24125200b896847",
    "codex/refactor-mainline": REPAIR_BASE,
    "codex/repair-sci-cal-001": "ae99be1cef8c390d0e7490835ffca1f31da7ebc0",
    "codex/repair-sci-map-001": "ed28dafb37f9113c0d3c95297148157129a90886",
    "codex/scientific-audit-framework": FROZEN_COORDINATION_HEAD,
}
POINT_OBSERVATIONS = (
    "152389", "152391", "152393", "152418",
    "152420", "152430", "152432", "152434",
)
UINT32_MODULUS = 2**32
PROVISIONAL_SLOT_TOLERANCE_SEC = 0.004063


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_scalar(item) for item in value.tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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


def require_report_claims(report_text: str, claims: dict[str, str]) -> None:
    missing = {name: text for name, text in claims.items() if text not in report_text}
    if missing:
        raise RuntimeError(f"human report is stale or incomplete: {missing}")


def unique_path(paths: Iterable[Path], label: str) -> Path:
    matches = sorted(paths)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}: {matches}")
    return matches[0]


def require_sha256(path: Path, expected: str) -> str:
    measured = sha256_file(path)
    if measured != expected:
        raise RuntimeError(
            f"frozen evidence input identity changed for {path}: "
            f"expected {expected}, measured {measured}"
        )
    return measured


def require_gap_grid_enabled(path: Path) -> bool:
    matches = re.findall(
        r"(?m)^\s*interp_over_gaps:\s*(yes|true|no|false)\s*$",
        path.read_text(),
        flags=re.IGNORECASE,
    )
    if len(matches) != 1 or matches[0].lower() not in {"yes", "true"}:
        raise RuntimeError(f"accepted config does not uniquely enable gap-grid path: {path}")
    return True


def config_interface_offsets(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text())
    rows = payload.get("interface_sync_offset")
    if not isinstance(rows, list):
        raise RuntimeError(f"interface_sync_offset is not a list in {path}")
    offsets: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or len(row) != 1:
            raise RuntimeError(f"malformed interface_sync_offset row in {path}: {row}")
        name, value = next(iter(row.items()))
        if name in offsets:
            raise RuntimeError(f"duplicate interface_sync_offset {name} in {path}")
        offsets[str(name)] = float(value)
    expected = {f"toltec{index}" for index in range(13)} | {"hwpr"}
    if set(offsets) != expected or not all(value == 0.0 for value in offsets.values()):
        raise RuntimeError(f"phase-0 config is not exact 14-interface zero offset: {path}")
    return offsets


def application_input_identity_rows(
    config_specs: list[tuple[str, Path, str]], local_root: Path,
) -> list[dict[str, Any]]:
    digest_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    for config_role, config_path, config_sha in config_specs:
        payload = yaml.safe_load(config_path.read_text())
        if not isinstance(payload.get("inputs"), list) or len(payload["inputs"]) != 1:
            raise RuntimeError(f"unexpected inputs structure in {config_path}")
        items = payload["inputs"][0].get("data_items", [])
        requested_interfaces = [str(item.get("meta", {}).get("interface", "MISSING"))
                                for item in items]
        duplicates = {
            name for name, count in Counter(requested_interfaces).items() if count > 1
        }
        for index, item in enumerate(items):
            requested_interface = str(item.get("meta", {}).get("interface", "MISSING"))
            requested_path = str(item.get("filepath", "MISSING"))
            basename = Path(requested_path).name
            matches = sorted(path for path in local_root.rglob(basename) if path.is_file())
            remote_prefix = "/work/toltec/"
            mirrored_candidates: list[Path] = []
            if requested_path.startswith(remote_prefix):
                relative = Path(requested_path.removeprefix(remote_prefix))
                mirrored_candidates.append(local_root / relative)
                if relative.parts and relative.parts[0] == "commissioning2025-test":
                    mirrored_candidates.append(local_root.joinpath(*relative.parts[1:]))
            resolved_matches = sorted({
                path.resolve() for path in mirrored_candidates if path.is_file()
            })
            identities = []
            content_hashes = []
            for match in matches:
                if match not in digest_cache:
                    digest_cache[match] = sha256_file(match)
                content_hashes.append(digest_cache[match])
                if basename.startswith("toltec"):
                    with Dataset(match) as dataset:
                        roach = int(np.asarray(dataset["Header.Toltec.RoachIndex"][:]).item())
                    identities.append(f"toltec{roach}")
                elif re.search(r"hwp|hwpr", basename, flags=re.IGNORECASE):
                    with Dataset(match) as dataset:
                        roach = int(np.asarray(dataset["Header.Toltec.RoachIndex"][:]).item())
                    identities.append(f"hwpr (Header.Toltec.RoachIndex={roach})")
                elif basename.startswith("tel_toltec_"):
                    identities.append("lmt")
                else:
                    identities.append("UNPROVED")
            for resolved_match in resolved_matches:
                if resolved_match not in digest_cache:
                    digest_cache[resolved_match] = sha256_file(resolved_match)
            normalized_identities = [
                "hwpr" if identity.startswith("hwpr ") else identity
                for identity in identities
            ]
            reconciled = bool(matches) and all(
                identity == requested_interface for identity in normalized_identities
            )
            resolved_hashes = [digest_cache[path] for path in resolved_matches]
            if len(resolved_matches) == 1:
                content_resolution_status = "owner-local mirrored path resolved uniquely"
            elif resolved_matches and len(set(resolved_hashes)) == 1:
                content_resolution_status = "multiple owner-local mirrors with identical content"
            elif resolved_matches:
                content_resolution_status = "CONFLICT: mirrored paths have different content"
            elif len(matches) == 1:
                content_resolution_status = "basename fallback resolved uniquely"
            elif matches and len(set(content_hashes)) == 1:
                content_resolution_status = "basename-only multiple identical copies"
            elif matches:
                content_resolution_status = "AMBIGUOUS: basename matches multiple contents"
            else:
                content_resolution_status = "UNAVAILABLE_LOCAL"
            if not matches:
                reconciliation_status = "UNAVAILABLE_LOCAL"
            elif not reconciled:
                reconciliation_status = "CONFLICT: requested interface disagrees with raw identity"
            elif content_resolution_status.startswith(("CONFLICT", "AMBIGUOUS")):
                reconciliation_status = "INTERFACE_ONLY_CONTENT_AMBIGUOUS"
            else:
                reconciliation_status = "PROVED_LOCAL_REQUEST_TO_RAW_IDENTITY"
            rows.append({
                "config_role": config_role,
                "config_path": str(config_path),
                "config_sha256": config_sha,
                "item_index": index,
                "requested_interface": requested_interface,
                "requested_filepath": requested_path,
                "requested_basename": basename,
                "duplicate_requested_interface": requested_interface in duplicates,
                "local_match_count": len(matches),
                "local_unique_content_count": len(set(content_hashes)),
                "local_paths": canonical_json([str(path) for path in matches]),
                "local_sha256": canonical_json(content_hashes),
                "resolved_local_paths": canonical_json([
                    str(path) for path in resolved_matches
                ]),
                "resolved_local_sha256": canonical_json(resolved_hashes),
                "content_resolution_status": content_resolution_status,
                "raw_interface_identities": canonical_json(identities),
                "requested_interface_matches_raw": reconciled,
                "reconciliation_status": reconciliation_status,
                "source_authority": (
                    "parsed YAML request + owner-local mirror mapping + NetCDF header"
                ),
            })
    return rows


def accepted_record(path: Path, record_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    matches = [row for row in payload["records"] if row.get("record_id") == record_id]
    if len(matches) != 1:
        raise RuntimeError(f"accepted record {record_id} count is {len(matches)}")
    return matches[0]


def require_ecsv_meta_unit(table: Table, name: str, unit: str) -> None:
    metadata = table.meta.get(name, [])
    if not metadata or str(metadata[0]).strip() != f"units: {unit}":
        raise RuntimeError(f"ECSV metadata for {name} does not declare {unit}")


def pointing_product_rows(path: Path) -> list[dict[str, Any]]:
    table = Table.read(path, format="ascii.ecsv")
    if int(table.meta.get("obsnum", -1)) != 152389 or len(table) != 3:
        raise RuntimeError(f"unexpected Pointing identity/row count in {path}")
    if set(int(value) for value in table["array"]) != {0, 1, 2}:
        raise RuntimeError(f"unexpected Pointing array registry in {path}")
    metric_names = ("x_t", "y_t", "a_fwhm", "b_fwhm")
    for name in metric_names:
        require_ecsv_meta_unit(table, name, "arcsec")
    if not all(np.isfinite(np.asarray(table[name], dtype=np.float64)).all()
               for name in metric_names):
        raise RuntimeError(f"nonfinite Pointing compatibility metric in {path}")
    labels = {0: "a1100", 1: "a1400", 2: "a2000"}
    rows = []
    for row in table:
        array_index = int(row["array"])
        if array_index not in labels:
            raise RuntimeError(f"unexpected pointing array index {array_index}")
        rows.append({
            "array": labels[array_index],
            "x_t_arcsec": float(row["x_t"]),
            "y_t_arcsec": float(row["y_t"]),
            "a_fwhm_arcsec": float(row["a_fwhm"]),
            "b_fwhm_arcsec": float(row["b_fwhm"]),
        })
    return rows


def beammap_fwhm_rows(path: Path) -> tuple[int, list[dict[str, Any]]]:
    table = Table.read(path, format="ascii.ecsv")
    if int(table.meta.get("obsnum", -1)) != 148670 or len(table) != 5234:
        raise RuntimeError(f"unexpected Beammap identity/row count in {path}")
    for name in ("a_fwhm", "b_fwhm"):
        require_ecsv_meta_unit(table, name, "arcsec")
    if set(int(value) for value in np.unique(table["array"])) != {0, 1, 2}:
        raise RuntimeError(f"unexpected Beammap array registry in {path}")
    if set(int(value) for value in np.unique(table["flag"])) != {0, 1}:
        raise RuntimeError(f"unexpected Beammap flag registry in {path}")
    labels = {0: "a1100", 1: "a1400", 2: "a2000"}
    expected_counts = {0: 2899, 1: 1190, 2: 891}
    rows = []
    for array_index, label in labels.items():
        selected = table[(table["flag"] == 0) & (table["array"] == array_index)]
        if len(selected) != expected_counts[array_index]:
            raise RuntimeError(f"unexpected flag-zero Beammap count for {label}")
        a_fwhm = np.asarray(selected["a_fwhm"], dtype=np.float64)
        b_fwhm = np.asarray(selected["b_fwhm"], dtype=np.float64)
        if not np.isfinite(a_fwhm).all() or not np.isfinite(b_fwhm).all():
            raise RuntimeError(f"nonfinite Beammap FWHM for {label}")
        rows.append({
            "array": label,
            "n": len(selected),
            "a_median": float(np.median(a_fwhm)),
            "a_p16": float(np.quantile(a_fwhm, 0.16)),
            "a_p84": float(np.quantile(a_fwhm, 0.84)),
            "b_median": float(np.median(b_fwhm)),
            "b_p16": float(np.quantile(b_fwhm, 0.16)),
            "b_p84": float(np.quantile(b_fwhm, 0.84)),
        })
    return len(table), rows


def source_crossing_metrics(path: Path) -> dict[str, Any]:
    labels = {0: "a1100", 1: "a1400", 2: "a2000"}
    with Dataset(path) as dataset:
        if str(getattr(
            dataset["detector_tod_source_center_distance_arcsec"], "units", "",
        )) != "arcsec":
            raise RuntimeError("source-crossing distance unit is not arcsec")
        uids = np.asarray(dataset["detector_tod_uid"][:], dtype=np.int64)
        arrays = np.asarray(dataset["detector_tod_array"][:], dtype=np.int64)
        fit_good = np.asarray(dataset["detector_tod_fit_good"][:], dtype=np.int64)
        good = fit_good == 1
        distance = np.asarray(
            dataset["detector_tod_source_center_distance_arcsec"][:], dtype=np.float64,
        )
        sample_rate = float(np.asarray(dataset["PTC_SAMPRATE"][:]).item())
    if arrays.size != 5234 or uids.size != arrays.size:
        raise RuntimeError("unexpected source-crossing detector count")
    if set(int(value) for value in np.unique(fit_good)) - {0, 1}:
        raise RuntimeError("unexpected source-crossing fit-good domain")
    if int(good.sum()) != 5135 or sample_rate != 122.0703125:
        raise RuntimeError("unexpected source-crossing good-fit count or sample rate")
    expected_counts = {0: 3004, 1: 1209, 2: 922}
    rows = []
    for array_index, label in labels.items():
        selected = distance[good & (arrays == array_index)]
        if selected.size != expected_counts[array_index] or not np.isfinite(selected).all():
            raise RuntimeError(f"unexpected source-crossing selection for {label}")
        rows.append({
            "array": label,
            "n": int(selected.size),
            "median": float(np.median(selected)),
            "p95": float(np.quantile(selected, 0.95)),
            "max": float(np.max(selected)),
        })
    return {
        "detector_count": int(arrays.size),
        "good_fit_count": int(good.sum()),
        "sample_rate_hz": sample_rate,
        "closest_distance_arcsec": rows,
    }


def normalized_shape(variable: Any) -> str:
    shape = ["time" if dim == "time" else int(size)
             for dim, size in zip(variable.dimensions, variable.shape, strict=True)]
    return canonical_json(shape)


def variable_metadata(variable: Any) -> dict[str, Any]:
    return {
        "dtype": str(variable.dtype),
        "dimensions": list(variable.dimensions),
        "shape": normalized_shape(variable),
        "units": json_scalar(getattr(variable, "units", None)),
        "long_name": json_scalar(getattr(variable, "long_name", None)),
        "validity_attributes": {
            name: json_scalar(getattr(variable, name))
            for name in (
                "_FillValue", "missing_value", "valid_min", "valid_max",
                "valid_range", "calendar", "coordinates", "standard_name",
            )
            if name in variable.ncattrs()
        },
    }


def schema_rows(path: Path, prefix: str | None = None) -> list[dict[str, Any]]:
    with Dataset(path) as dataset:
        rows = []
        for name, variable in sorted(dataset.variables.items()):
            if prefix is not None and not name.startswith(prefix):
                continue
            row = {"name": name, **variable_metadata(variable)}
            rows.append(row)
        return rows


def schema_digest(path: Path, prefix: str | None = None) -> str:
    return sha256_text(canonical_json(schema_rows(path, prefix)) + "\n")


def timing_projection_digest(path: Path, variables: Iterable[str]) -> str:
    digest = hashlib.sha256()
    with Dataset(path) as dataset:
        for name in sorted(set(variables)):
            if name not in dataset.variables:
                digest.update(canonical_json({"name": name, "missing": True}).encode())
                digest.update(b"\n")
                continue
            variable = dataset[name]
            metadata = {"name": name, **variable_metadata(variable)}
            digest.update(canonical_json(metadata).encode())
            digest.update(b"\n")
            values = np.ascontiguousarray(np.asarray(variable[:]))
            digest.update(values.tobytes(order="C"))
            digest.update(b"\n")
    return digest.hexdigest()


def git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def git_text(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout


def git_name_lines(repo: Path, *arguments: str) -> list[str]:
    return [line for line in git_text(repo, *arguments).splitlines() if line]


def worktree_records(porcelain: str) -> list[dict[str, str]]:
    records = []
    for block in porcelain.strip().split("\n\n"):
        if not block:
            continue
        record: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            record[key] = value
        records.append(record)
    return records


def validate_git_and_coordination_gate(repo: Path, coordination_repo: Path) -> dict[str, Any]:
    snapshot_path = repo / "tools/diagnostics/sci_align_001_phase0_gate_snapshot.json"
    snapshot_sha = require_sha256(snapshot_path, GATE_SNAPSHOT_SHA256)
    snapshot = json.loads(snapshot_path.read_text())
    historical = snapshot["historical_prebranch_observation"]
    historical_coordination = snapshot["historical_coordination_observation"]
    if not (
        historical["head"] == REPAIR_BASE
        and historical["status_porcelain"] == ""
        and historical["codex_refactor_mainline"] == REPAIR_BASE
        and historical["repair_branch_lookup"]["exit_code"] == 1
        and historical["repair_branch_lookup"]["stdout"] == ""
        and historical["other_align_repair_worktrees"] == []
        and historical["created_branch_start"] == REPAIR_BASE
        and snapshot["gate_time_scoped_refs"] == GATE_TIME_SCOPED_REFS
        and historical_coordination["corrected_frozen_head"] == FROZEN_COORDINATION_HEAD
        and historical_coordination["status_porcelain"] == ""
        and historical_coordination["handoff_record_commit"] == HANDOFF_RECORD_COMMIT
        and historical_coordination["handoff_record_is_ancestor"] is True
    ):
        raise RuntimeError("historical gate snapshot does not match frozen repair identity")

    branch = git_text(repo, "symbolic-ref", "--short", "HEAD").strip()
    if branch != "codex/repair-sci-align-001":
        raise RuntimeError(f"unexpected repair branch: {branch}")
    head = git_text(repo, "rev-parse", "HEAD").strip()
    mainline = git_text(repo, "rev-parse", "refs/heads/codex/refactor-mainline").strip()
    repair_ref = git_text(
        repo, "rev-parse", "refs/heads/codex/repair-sci-align-001",
    ).strip()
    if mainline != REPAIR_BASE or repair_ref != head:
        raise RuntimeError(
            f"live branch identity mismatch: mainline={mainline} repair={repair_ref} head={head}"
        )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", REPAIR_BASE, "HEAD"], cwd=repo,
    ).returncode != 0:
        raise RuntimeError("repair base is not an ancestor of current HEAD")

    reflog_lines = git_name_lines(
        repo, "reflog", "show", "--date=iso-strict",
        "--format=%H%x09%gD%x09%gs", "refs/heads/codex/repair-sci-align-001",
    )
    creation_lines = [line for line in reflog_lines if "\tbranch: Created from " in line]
    expected_creation_suffix = f"\tbranch: Created from {REPAIR_BASE}"
    if len(creation_lines) != 1 or not (
        creation_lines[0].startswith(REPAIR_BASE + "\t")
        and creation_lines[0].endswith(expected_creation_suffix)
    ):
        raise RuntimeError(f"repair branch creation reflog is not unique/exact: {creation_lines}")

    worktrees = worktree_records(git_text(repo, "worktree", "list", "--porcelain"))
    repair_owners = [
        record for record in worktrees
        if record.get("branch") == "refs/heads/codex/repair-sci-align-001"
    ]
    if len(repair_owners) != 1 or Path(repair_owners[0]["worktree"]).resolve() != repo:
        raise RuntimeError(f"repair branch worktree ownership conflict: {repair_owners}")
    target_repo_path = Path("/Users/gwilson/GitHub/citlali-refactor")
    common_dir = Path(git_text(
        repo, "rev-parse", "--path-format=absolute", "--git-common-dir",
    ).strip()).resolve()
    target_common_dir = Path(git_text(
        target_repo_path, "rev-parse", "--path-format=absolute", "--git-common-dir",
    ).strip()).resolve()
    worktree_paths = {Path(record["worktree"]).resolve() for record in worktrees}
    if common_dir != target_common_dir or target_repo_path.resolve() not in worktree_paths:
        raise RuntimeError("supplied worktree does not share the requested repository identity")
    all_branch_names = git_name_lines(
        repo, "for-each-ref", "--format=%(refname:short)", "refs/heads",
    )
    other_repair_align_refs = sorted(
        name for name in all_branch_names
        if name != "codex/repair-sci-align-001"
        and "repair" in name.lower() and "align" in name.lower()
    )
    other_repair_align_worktrees = sorted(
        record.get("worktree", "") for record in worktrees
        if record.get("branch", "") != "refs/heads/codex/repair-sci-align-001"
        and "repair" in record.get("branch", "").lower()
        and "align" in record.get("branch", "").lower()
    )
    if other_repair_align_refs or other_repair_align_worktrees:
        raise RuntimeError(
            "another ALIGN repair state exists: "
            f"refs={other_repair_align_refs} worktrees={other_repair_align_worktrees}"
        )

    path_sets = [
        git_name_lines(repo, "diff", "--name-only", REPAIR_BASE, "HEAD"),
        git_name_lines(repo, "diff", "--cached", "--name-only"),
        git_name_lines(repo, "diff", "--name-only"),
        git_name_lines(repo, "ls-files", "--others", "--exclude-standard"),
    ]
    observed_paths = sorted({path for paths in path_sets for path in paths})
    exact_allowed = {
        "tools/diagnostics/generate_sci_align_001_phase0.py",
        "tools/diagnostics/sci_align_001_phase0_gate_snapshot.json",
        "tools/diagnostics/sci_align_001_phase0_report.md",
    }
    allowed_prefix = "validation/sci_align_001_phase0_2026-08-01/"
    disallowed_paths = [
        path for path in observed_paths
        if path not in exact_allowed and not path.startswith(allowed_prefix)
    ]
    if disallowed_paths:
        raise RuntimeError(f"non-evidence path delta on repair branch: {disallowed_paths}")

    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", HANDOFF_RECORD_COMMIT,
         FROZEN_COORDINATION_HEAD], cwd=coordination_repo,
    ).returncode != 0:
        raise RuntimeError("handoff record is not an ancestor of frozen coordination head")
    handoff_path = (
        "doc/audits/packages/"
        "SCI-ALIGN-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-08-01.md"
    )
    handoff_sha = hashlib.sha256(git_blob(
        coordination_repo, FROZEN_COORDINATION_HEAD, handoff_path,
    )).hexdigest()
    if handoff_sha != historical_coordination["handoff_blob_sha256"]:
        raise RuntimeError("frozen handoff blob hash changed")

    return {
        "historical_prebranch_observation": historical,
        "historical_coordination_observation": historical_coordination,
        "gate_snapshot_path": str(snapshot_path),
        "gate_snapshot_sha256": snapshot_sha,
        "live_reflog_and_scope_assertions": {
            "current_branch_is_repair_branch": True,
            "mainline_still_exact_repair_base": True,
            "repair_base_is_ancestor_of_head": True,
            "repair_ref_equals_head": True,
            "branch_creation_reflog_entry": creation_lines[0],
            "repair_branch_has_exactly_one_worktree_owner": True,
            "repair_branch_worktree": str(repo),
            "supplied_and_requested_paths_share_git_common_dir": True,
            "git_common_dir": str(common_dir),
            "other_repair_align_refs": [],
            "other_repair_align_worktrees": [],
            "evidence_only_path_allowlist_passed": True,
            "application_path_delta": [],
            "disallowed_path_delta": [],
            "frozen_handoff_commit_and_blob_checks_passed": True,
        },
        "allowed_path_scope": {
            "exact": sorted(exact_allowed),
            "prefix": allowed_prefix,
            "checked_sources": "base..HEAD, index, worktree, and untracked union",
        },
        "independence_conclusion": (
            "repair branch/worktree is distinct from audit, coordination, MAP, CAL, "
            "convolve, and noise refs/worktrees; only evidence paths are changed"
        ),
    }


def extract_cpp_map(text: str, map_name: str) -> list[tuple[str, str]]:
    match = re.search(
        rf"{re.escape(map_name)}\s*\{{(?P<body>.*?)\n\s*\}};",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"unable to parse {map_name}")
    return re.findall(r'\{"([^"]+)"\s*,\s*"([^"]+)"\}', match.group("body"))


def collect_schema_union(paths: list[Path]) -> dict[str, dict[str, Any]]:
    union: dict[str, dict[str, Any]] = {}
    for path in paths:
        with Dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                entry = union.setdefault(
                    name,
                    {
                        "dtypes": set(), "shapes": set(), "units": set(),
                        "long_names": set(), "validity_attrs": set(), "path_count": 0,
                    },
                )
                entry["dtypes"].add(str(variable.dtype))
                entry["shapes"].add(normalized_shape(variable))
                entry["units"].add(str(json_scalar(getattr(variable, "units", "UNPROVED"))))
                entry["long_names"].add(str(json_scalar(getattr(variable, "long_name", "UNPROVED"))))
                entry["validity_attrs"].update(
                    f"{name}={canonical_json(json_scalar(getattr(variable, name)))}"
                    for name in (
                        "_FillValue", "missing_value", "valid_min", "valid_max",
                        "valid_range", "calendar", "coordinates", "standard_name",
                    )
                    if name in variable.ncattrs()
                )
                entry["path_count"] += 1
    return union


def join_values(values: Iterable[Any]) -> str:
    return " | ".join(sorted(str(value) for value in values))


def stable_id(interface: str, raw_name: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", ".", raw_name.lower()).strip(".")
    return f"{interface}.{suffix}"


def candidate_topology(raw_name: str, container: str) -> str:
    leaf = raw_name.rsplit(".", 1)[-1]
    if container == "header":
        return "exact_only"
    if leaf in {"Hold", "BufPos", "ScanPos"}:
        return "declared_half_open_step_state"
    if leaf.lower().endswith("lst"):
        return "circular_angle"
    if re.search(r"(?:Time|Utc|Uts|Ts|Count|Chan|Pos|Date)$", leaf):
        return "exact_only"
    if leaf in {"HwprEncVal", "HwprZeroptVal", "HwprPpsVal"}:
        return "exact_only"
    if leaf == "HwprSensorVolt":
        return "continuous_scalar"
    if raw_name == "Data.Hwp.":
        return "circular_angle"
    if any(token in leaf for token in ("ParAng", "GalAng")):
        return "circular_angle"
    if any(token in leaf for token in ("Ra", "SourceL")):
        return "circular_angle"
    if "Az" in leaf and not any(token in leaf for token in ("Cor", "Map", "Off")):
        return "circular_angle"
    return "continuous_scalar"


def inferred_frame(raw_name: str) -> str:
    leaf = raw_name.rsplit(".", 1)[-1]
    if re.search(r"(?:Time|Utc|Uts|Ts|Count|Date|Lst)$", leaf, re.IGNORECASE):
        return "UNPROVED clock/epoch"
    if "Ra" in leaf or "Dec" in leaf:
        return "UNPROVED equatorial realization/epoch"
    if "SourceL" in leaf or "SourceB" in leaf or "GalAng" in leaf:
        return "UNPROVED galactic realization"
    if "Az" in leaf or "El" in leaf or "ParAng" in leaf:
        return "UNPROVED topocentric/horizontal realization"
    return "UNPROVED"


def registry_row(
    *, interface: str, raw_name: str, container: str,
    info: dict[str, Any] | None, availability: str,
    native_output: str, aligned_output: str, source_authority: str,
) -> dict[str, Any]:
    info = info or {
        "dtypes": {"UNAVAILABLE"}, "shapes": {"UNAVAILABLE"},
        "units": {"UNPROVED"}, "long_names": {"UNPROVED"},
        "validity_attrs": set(), "path_count": 0,
    }
    units = join_values(info["units"])
    long_names = join_values(info["long_names"])
    candidate = candidate_topology(raw_name, container)
    authority_complete = (
        availability.startswith("available")
        and units != "UNPROVED"
        and bool(info["validity_attrs"])
        and container == "header"
    )
    # Phase 0 is fail-closed: an unreviewed candidate topology is not activated.
    topology = "exact_only"
    conflicts = []
    if candidate != topology:
        conflicts.append(f"candidate_topology={candidate} requires producer/owner approval")
    if units == "UNPROVED":
        conflicts.append("unit unproved")
    if not info["validity_attrs"]:
        conflicts.append("missing/nonfinite/validity policy unproved")
    if candidate != "exact_only":
        conflicts.append("maximum support span unproved")
    if raw_name.endswith("TelUtc"):
        conflicts.append("metadata unit sec conflicts with radian-like UT1 values and code conversion")
    if raw_name.endswith("TelLst"):
        conflicts.append("metadata unit sec conflicts with periodic/radian-like local values")
    if raw_name.endswith("Hold"):
        conflicts.append("metadata boolean conflicts with observed bitmask values 0,2,8,10,64,66,72,74")
    if raw_name in {"Data.TelescopeBackend.TelRaAct", "Data.TelescopeBackend.TelDecAct"}:
        conflicts.append("absent in every surveyed local telescope schema; alias precedence unproved")
    if interface == "hwpr" and raw_name.startswith("Data.Hwp"):
        conflicts.append("application-required schema absent from every surveyed local HWPR file")
    source_dtypes = join_values(info["dtypes"])
    integer_storage = bool(re.search(r"\bu?int(?:8|16|32|64)\b", source_dtypes))
    hwpr_integer_value = bool(
        interface == "hwpr"
        and raw_name.startswith("Data.Toltec.Hwpr")
        and raw_name.endswith("Val")
    )
    hwpr_packed_value = bool(
        interface == "hwpr"
        and raw_name.startswith("Data.Toltec.Hwpr")
        and re.search(r"(?:Val|Volt|Chan)$", raw_name)
    )
    timing_field = is_timing_boundary_field(raw_name)
    if candidate == "circular_angle":
        candidate_parameters = "candidate only; period and wrap rule UNPROVED"
    elif candidate == "declared_half_open_step_state":
        candidate_parameters = "candidate only; state coding and half-open transition side UNPROVED"
    else:
        candidate_parameters = "N/A"
    return {
        "field_id": stable_id(interface, raw_name),
        "raw_name": raw_name,
        "raw_container": container,
        "interface_id": interface,
        "availability": availability,
        "source_dtype": source_dtypes,
        "source_shape": join_values(info["shapes"]),
        "scientific_identity": long_names,
        "unit": units,
        "unit_authority_status": "proved_by_raw_attribute" if units != "UNPROVED" else "unproved",
        "epoch_or_clock": (
            inferred_frame(raw_name)
            if re.search(
                r"Time|Utc|Uts|Ts|Count|Date|Lst", raw_name, re.IGNORECASE,
            ) else "N/A"
        ),
        "counter_width_bits": (
            f"storage={source_dtypes}; logical width/range UNPROVED"
            if integer_storage else "N/A"
        ),
        "rollover_policy": (
            "UNPROVED" if integer_storage and (timing_field or hwpr_integer_value) else "N/A"
        ),
        "coordinate_frame": inferred_frame(raw_name),
        "candidate_scientific_class": candidate,
        "topology": topology,
        "topology_parameters": candidate_parameters,
        "validity_missing_nonfinite_policy": (
            "native attrs: " + join_values(info["validity_attrs"])
            if info["validity_attrs"] else
            "UNPROVED; no native fill/missing/validity attributes"
        ),
        "maximum_support_span_sec": "0 (fail-closed pending owner review)",
        "native_cadence": (
            "N/A (file/observation header snapshot or vector)"
            if container == "header" else
            "UNPROVED; packed HWPR value/timestamp pairing and valid-count policy absent"
            if hwpr_packed_value else
            "per-file/per-field measurement in telescope_hwpr_timing_field_inventory.csv"
            if timing_field else
            "inherits interface coordinate in boundary_stream_inventory.csv; field-specific authority UNPROVED"
        ),
        "native_acquisition_bounds": (
            "file/observation scope; see boundary_stream_inventory.csv"
            if container == "header" else
            "UNPROVED; packed HWPR value/timestamp pairing and valid-count policy absent"
            if hwpr_packed_value else
            "per-file measurement in telescope_hwpr_timing_field_inventory.csv"
            if timing_field else
            "inherits interface coordinate bounds in boundary_stream_inventory.csv"
        ),
        "native_duration": (
            "N/A (header value persists at file/observation scope)"
            if container == "header" else
            "UNPROVED; packed HWPR value/timestamp pairing and valid-count policy absent"
            if hwpr_packed_value else
            "per-file measurement in telescope_hwpr_timing_field_inventory.csv"
            if timing_field else
            "inherits interface coordinate duration in boundary_stream_inventory.csv"
        ),
        "permitted_operator": "exact_coincidence_only",
        "native_output_identity": native_output,
        "aligned_output_identity": aligned_output,
        "source_authority": source_authority,
        "confidence": "proved" if authority_complete else "unproved_or_conflicting",
        "unresolved_conflicts": "; ".join(conflicts) if conflicts else "none",
    }


def boundary_stream_row(path: Path, kind: str, digest: str) -> dict[str, Any]:
    with Dataset(path) as dataset:
        if kind == "telescope":
            coordinate_name = "Data.TelescopeBackend.TelTime"
            values = np.asarray(dataset[coordinate_name][:], dtype=np.float64)
            units = str(json_scalar(getattr(dataset[coordinate_name], "units", "UNPROVED")))
            coordinate_status = "application boundary coordinate present"
            source_authority = "local raw schema; Telescope::get_tel_data"
            interface_id = "lmt"
            header_rate = "N/A"
            obs_start = "UNAVAILABLE"
            obs_end = "UNAVAILABLE"
            counter_width = "N/A"
            rollover = "N/A"
        else:
            detector = detector_times(path)
            coordinate_name = "Data.Toltec.Ts (raw candidate); Data.Hwp.Ts/Uts absent"
            values = detector["current_time"]
            units = "sec reconstructed provisionally; epoch UNPROVED"
            coordinate_status = "application-required HWPR coordinate schema absent"
            source_authority = "local raw schema; Calib::get_hwpr expects incompatible Data.Hwp.*"
            interface_id = "hwpr"
            header_rate = detector["sample_rate"]
            obs_start = detector["obs_start"]
            obs_end = detector["obs_end"]
            counter_width = "Data.Toltec.Ts storage int32; logical widths UNPROVED"
            rollover = "UNPROVED; diagnostic reconstruction mirrors current detector formula"
        steps = np.diff(values)
        return {
            "kind": kind,
            "interface_id": interface_id,
            "path": str(path),
            "sha256": digest,
            "coordinate_raw_name": coordinate_name,
            "coordinate_status": coordinate_status,
            "coordinate_unit": units,
            "coordinate_epoch": "UNPROVED",
            "counter_width_bits": counter_width,
            "rollover_policy": rollover,
            "header_sample_rate_hz": header_rate,
            "sample_count": int(values.size),
            "first_coordinate_sec": float(values[0]),
            "last_coordinate_sec": float(values[-1]),
            "duration_sec": float(values[-1] - values[0]),
            "cadence_min_sec": float(np.min(steps)) if steps.size else "N/A",
            "cadence_median_sec": float(np.median(steps)) if steps.size else "N/A",
            "cadence_max_sec": float(np.max(steps)) if steps.size else "N/A",
            "nonpositive_cadence_count": int(np.sum(steps <= 0)),
            "header_acquisition_start": obs_start,
            "header_acquisition_end": obs_end,
            "header_acquisition_unit_epoch": "UNPROVED",
            "all_coordinate_values_finite": bool(np.isfinite(values).all()),
            "missing_nonfinite_policy": "UNPROVED; no fill/missing/validity authority",
            "source_authority": source_authority,
        }


def is_timing_boundary_field(raw_name: str) -> bool:
    leaf = raw_name.rsplit(".", 1)[-1]
    if leaf in {
        "Ts", "UT1", "DUT1", "LST", "Clock", "TimeOfDay", "Timer",
        "TimeLock", "FpgaFreq", "SampleFreq", "AccumLen",
    }:
        return True
    if leaf.startswith("Hwpr") and leaf.endswith("Ts"):
        return True
    return bool(re.search(
        r"(?:Time|Utc|Uts|Count|Date|Lst)$", leaf, re.IGNORECASE,
    ))


def telescope_hwpr_timing_rows(
    path: Path, kind: str, digest: str,
) -> list[dict[str, Any]]:
    ts_identities = (
        ("ClockTime", "sec (long_name only)", "UNPROVED"),
        ("PpsCount", "pps ticks", "UNPROVED"),
        ("ClockCount", "clock ticks", "UNPROVED"),
        ("PacketCount", "packet ticks", "UNPROVED"),
        ("PpsTime", "clock ticks", "UNPROVED"),
        ("ClockTimeNanoSec", "nsec", "paired with ClockTime; epoch UNPROVED"),
    )
    rows: list[dict[str, Any]] = []
    with Dataset(path) as dataset:
        obs_start = (
            json_scalar(np.asarray(dataset["Header.Toltec.ObsStartTime"][:]).item())
            if "Header.Toltec.ObsStartTime" in dataset.variables else "UNAVAILABLE"
        )
        obs_end = (
            json_scalar(np.asarray(dataset["Header.Toltec.ObsEndTime"][:]).item())
            if "Header.Toltec.ObsEndTime" in dataset.variables else "UNAVAILABLE"
        )
        for raw_name, variable in sorted(dataset.variables.items()):
            if not is_timing_boundary_field(raw_name):
                continue
            raw_values = np.ma.asarray(variable[:])
            variants: list[tuple[str, np.ma.MaskedArray, str, str, str]]
            if raw_name == "Data.Toltec.Ts" and raw_values.ndim == 2 and raw_values.shape[1] == 6:
                variants = [
                    (f"Data.Toltec.Ts[:,{index}]", raw_values[:, index], identity, unit, epoch)
                    for index, (identity, unit, epoch) in enumerate(ts_identities)
                ]
            else:
                variants = [(
                    raw_name,
                    raw_values,
                    str(json_scalar(getattr(variable, "long_name", "UNPROVED"))),
                    str(json_scalar(getattr(variable, "units", "UNPROVED"))),
                    inferred_frame(raw_name) if re.search(
                        r"Time|Utc|Uts|Ts|Count|Date|Lst", raw_name, re.IGNORECASE,
                    ) else "N/A",
                )]
            for measured_name, values, identity, unit, epoch in variants:
                dtype = np.dtype(values.dtype)
                mask = np.ma.getmaskarray(values)
                masked_count = int(np.sum(mask))
                numeric = np.issubdtype(dtype, np.number)
                if numeric:
                    filled = np.asarray(
                        values.filled(np.nan)
                        if np.issubdtype(dtype, np.floating) else values.filled(0)
                    )
                else:
                    filled = np.asarray(values.data)
                sequence_1d = values.ndim == 1 and "time" in variable.dimensions
                if numeric:
                    numeric_values = filled.astype(np.float64, copy=False).reshape(-1)
                    finite = np.isfinite(numeric_values)
                    valid_values = numeric_values[finite & ~mask.reshape(-1)]
                    value_min = float(np.min(valid_values)) if valid_values.size else "N/A"
                    value_max = float(np.max(valid_values)) if valid_values.size else "N/A"
                    nonfinite_count: int | str = int(np.sum(~finite & ~mask.reshape(-1)))
                else:
                    valid_values = np.asarray([], dtype=np.float64)
                    value_min = "N/A"
                    value_max = "N/A"
                    nonfinite_count = "N/A"
                if sequence_1d and numeric and not masked_count and valid_values.size == values.size:
                    sequence = filled.astype(np.float64, copy=False)
                    differences = np.diff(sequence)
                    first_value: float | str = float(sequence[0]) if sequence.size else "N/A"
                    last_value: float | str = float(sequence[-1]) if sequence.size else "N/A"
                    duration: float | str = (
                        float(sequence[-1] - sequence[0]) if sequence.size else "N/A"
                    )
                    diff_min: float | str = float(np.min(differences)) if differences.size else "N/A"
                    diff_median: float | str = float(np.median(differences)) if differences.size else "N/A"
                    diff_max: float | str = float(np.max(differences)) if differences.size else "N/A"
                    nonpositive: int | str = int(np.sum(differences <= 0))
                    duplicates: int | str = int(np.sum(differences == 0))
                    sequence_status = "measured 1-D native sequence; scientific authority UNPROVED"
                else:
                    first_value = "N/A"
                    last_value = "N/A"
                    duration = "N/A"
                    diff_min = "N/A"
                    diff_median = "N/A"
                    diff_max = "N/A"
                    nonpositive = "N/A"
                    duplicates = "N/A"
                    sequence_status = (
                        "N/A: multidimensional packed records; ordering/valid-count policy UNPROVED"
                        if values.ndim > 1 else
                        "N/A: file/observation header snapshot or nonnumeric field"
                    )
                validity = variable_metadata(variable)["validity_attributes"]
                rows.append({
                    "kind": kind,
                    "interface_id": "lmt" if kind == "telescope" else "hwpr",
                    "path": str(path),
                    "sha256": digest,
                    "raw_name": measured_name,
                    "parent_raw_name": raw_name,
                    "scientific_identity": identity,
                    "source_dtype": str(dtype),
                    "source_dimensions": canonical_json(list(variable.dimensions)),
                    "source_shape": canonical_json(list(variable.shape)),
                    "unit": unit,
                    "epoch_or_clock": epoch,
                    "counter_width_bits": (
                        f"storage={dtype.itemsize * 8}; logical width/range UNPROVED"
                        if np.issubdtype(dtype, np.integer) else "N/A"
                    ),
                    "rollover_policy": (
                        "UNPROVED" if np.issubdtype(dtype, np.integer) else "N/A"
                    ),
                    "sequence_status": sequence_status,
                    "sample_or_element_count": int(values.size),
                    "outer_record_count": int(values.shape[0]) if values.ndim else 1,
                    "packed_width": int(np.prod(values.shape[1:])) if values.ndim > 1 else "N/A",
                    "first_value": first_value,
                    "last_value": last_value,
                    "value_min": value_min,
                    "value_max": value_max,
                    "duration_native_units": duration,
                    "step_min_native_units": diff_min,
                    "step_median_native_units": diff_median,
                    "step_max_native_units": diff_max,
                    "nonpositive_step_count": nonpositive,
                    "duplicate_step_count": duplicates,
                    "masked_value_count": masked_count,
                    "nonfinite_value_count": nonfinite_count,
                    "native_validity_attributes": canonical_json(validity),
                    "missing_nonfinite_policy": "UNPROVED beyond measured native metadata",
                    "header_acquisition_start": obs_start,
                    "header_acquisition_end": obs_end,
                    "header_acquisition_unit_epoch": "UNPROVED",
                    "source_authority": "local raw schema/value measurement; no producer contract",
                })
    return rows


def array_value_evidence(values: Any) -> dict[str, Any]:
    masked = np.ma.asarray(values)
    mask = np.ma.getmaskarray(masked)
    data = np.ascontiguousarray(np.asarray(masked.data))
    digest = hashlib.sha256()
    digest.update(canonical_json({
        "dtype": str(data.dtype), "shape": list(data.shape),
    }).encode())
    digest.update(b"\n")
    digest.update(data.tobytes(order="C"))
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C"))
    masked_count = int(np.sum(mask))
    if data.dtype.kind in "SU":
        decoded = data.tobytes(order="C").replace(b"\x00", b"").decode(
            "utf-8", errors="replace",
        )
        summary: Any = {"decoded": decoded, "masked_count": masked_count}
        nonfinite_count: int | str = "N/A"
    elif np.issubdtype(data.dtype, np.number):
        numeric = data.astype(np.float64, copy=False).reshape(-1)
        flat_mask = mask.reshape(-1)
        finite = np.isfinite(numeric)
        valid = numeric[finite & ~flat_mask]
        summary = {
            "finite_unmasked_count": int(valid.size),
            "masked_count": masked_count,
            "nonfinite_unmasked_count": int(np.sum(~finite & ~flat_mask)),
            "min": float(np.min(valid)) if valid.size else "N/A",
            "max": float(np.max(valid)) if valid.size else "N/A",
        }
        if numeric.size <= 32:
            summary["values"] = [json_scalar(value) for value in data.reshape(-1)]
        nonfinite_count = summary["nonfinite_unmasked_count"]
    else:
        summary = {"masked_count": masked_count, "representation": "digest only"}
        nonfinite_count = "N/A"
    return {
        "raw_value_sha256": digest.hexdigest(),
        "value_summary": canonical_json(summary),
        "masked_value_count": masked_count,
        "nonfinite_value_count": nonfinite_count,
    }


def detector_header_application_role(raw_name: str) -> str:
    if raw_name == "Header.Toltec.FpgaFreq":
        return "alignment_consumed + optional_hwpr_consumed"
    if raw_name in {"Header.Toltec.RoachIndex", "Header.Toltec.SampleFreq"}:
        return "alignment_consumed"
    if raw_name == "Header.Toltec.HwpInstalled":
        return "optional_hwpr_consumed"
    if raw_name == "Header.Toltec.AdcSnapData":
        return "diagnostic_inventory_consumed: rawobs_adc_snap"
    if raw_name in {"Header.Toltec.LoCenterFreq", "Header.Toltec.ToneFreq"}:
        return "diagnostic_inventory_consumed: tone-frequency/coherent-IQ"
    return "available_unconsumed_boundary_header"


def detector_timing_field_rows(timing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ts_specs = (
        (0, "ClockTime", "sec (long_name only)", "UNPROVED", "storage int32; logical width UNPROVED", "UNPROVED"),
        (1, "PpsCount", "pps ticks", "UNPROVED", "storage int32; logical width UNPROVED", "UNPROVED"),
        (2, "ClockCount", "clock ticks", "UNPROVED", "storage int32; code assumes uint32", "code adds 2^32-1 for negative differences; producer authority UNPROVED"),
        (3, "PacketCount", "packet ticks", "UNPROVED", "storage int32; logical width UNPROVED", "UNPROVED; current diff>1 test is not rollover-aware"),
        (4, "PpsTime", "clock ticks", "UNPROVED", "storage int32; code assumes uint32", "paired with ClockCount; producer authority UNPROVED"),
        (5, "ClockTimeNanoSec", "nsec", "paired with Ts[:,0]; epoch UNPROVED", "storage int32; logical width UNPROVED", "UNPROVED"),
    )
    result: list[dict[str, Any]] = []
    header_registries: set[tuple[str, ...]] = set()
    for timing in sorted(
        timing_rows, key=lambda row: (row["mode"], row["obsnum"], row["interface_id"]),
    ):
        raw_path = Path(timing["raw_path"])
        with Dataset(raw_path) as dataset:
            header_names = tuple(sorted(
                name for name in dataset.variables if name.startswith("Header.")
            ))
            header_registries.add(header_names)
            if "Data.Toltec.Ts" not in dataset.variables:
                raise RuntimeError(f"missing detector timing matrix in {raw_path}")
            ts_variable = dataset["Data.Toltec.Ts"]
            ts = np.ma.asarray(ts_variable[:])
            if ts.ndim != 2 or ts.shape[1] != 6:
                raise RuntimeError(f"unexpected Data.Toltec.Ts shape in {raw_path}: {ts.shape}")
            stream_context = {
                "detector_stream_context_sample_rate_hz": timing["sample_rate_hz"],
                "detector_stream_context_sample_count": timing["native_rows"],
                "detector_stream_context_first_time_sec": timing["native_first_time_sec"],
                "detector_stream_context_last_time_sec": timing["native_last_time_sec"],
                "detector_stream_context_duration_sec": timing["native_duration_sec"],
                "header_acquisition_start": timing["header_obs_start"],
                "header_acquisition_end": timing["header_obs_end"],
                "header_acquisition_duration_unproved_units": (
                    timing["header_obs_end"] - timing["header_obs_start"]
                ),
                "header_acquisition_unit_epoch": "UNPROVED",
                "common_grid_first_sec": timing["phase_sec_current_compatible_provisional"],
                "common_grid_last_sec": timing["current_grid_end_sec"],
                "common_grid_duration_sec": (
                    timing["current_grid_end_sec"]
                    - timing["phase_sec_current_compatible_provisional"]
                ),
            }

            def append_row(**field: Any) -> None:
                result.append({
                    "mode": timing["mode"], "obsnum": timing["obsnum"],
                    "interface_id": timing["interface_id"],
                    "raw_path": timing["raw_path"],
                    "timing_projection_sha256": timing["timing_projection_sha256"],
                    **field, **stream_context,
                })

            selected_config = str(timing["obsnum"]) in {"152389", "148670"}
            for raw_name, identity in (
                ("inputs[].data_items[].meta.interface", "requested stream/interface identity"),
                ("inputs[].data_items[].filepath", "requested source-file identity"),
            ):
                append_row(
                    field_id=stable_id(timing["interface_id"], raw_name),
                    raw_name=raw_name, raw_container="configuration",
                    availability=(
                        "available_selected_config" if selected_config else
                        "UNAVAILABLE: no observation-specific config supplied"
                    ),
                    application_role="requested_alignment_boundary",
                    source_dtype="YAML string", source_dimensions="[]",
                    source_shape="[]", scientific_identity=identity, unit="N/A",
                    unit_authority_status="N/A", epoch_or_clock="N/A",
                    counter_width_bits="N/A", rollover_policy="N/A",
                    topology="exact_only", field_scope="requested_configuration",
                    field_value_count=1 if selected_config else 0,
                    field_native_cadence_sec="N/A",
                    field_native_acquisition_bounds="N/A: requested configuration",
                    field_native_duration_sec="N/A",
                    native_validity_attributes="{}",
                    missing_nonfinite_policy=(
                        "selected request inventoried in application_input_identity_inventory.csv"
                        if selected_config else
                        "UNAVAILABLE; raw-zero-offset sensitivity only"
                    ),
                    masked_value_count="N/A", nonfinite_value_count="N/A",
                    raw_value_sha256="see application_input_identity_inventory.csv",
                    value_summary=(
                        "see application_input_identity_inventory.csv"
                        if selected_config else
                        "UNAVAILABLE: no observation-specific config supplied"
                    ),
                    source_authority="parsed selected YAML request or explicit absence",
                )

            for raw_name in header_names:
                variable = dataset[raw_name]
                values = np.ma.asarray(variable[:])
                evidence = array_value_evidence(values)
                dtype = np.dtype(variable.dtype)
                integer_storage = np.issubdtype(dtype, np.integer)
                timing_field = is_timing_boundary_field(raw_name)
                validity = variable_metadata(variable)["validity_attributes"]
                unit = str(json_scalar(getattr(variable, "units", "UNPROVED")))
                append_row(
                    field_id=stable_id(timing["interface_id"], raw_name),
                    raw_name=raw_name, raw_container="header", availability="available",
                    application_role=detector_header_application_role(raw_name),
                    source_dtype=str(variable.dtype),
                    source_dimensions=canonical_json(list(variable.dimensions)),
                    source_shape=normalized_shape(variable),
                    scientific_identity=str(json_scalar(
                        getattr(variable, "long_name", "UNPROVED"),
                    )),
                    unit=unit,
                    unit_authority_status=(
                        "proved_by_raw_attribute" if unit != "UNPROVED" else "unproved"
                    ),
                    epoch_or_clock=(
                        inferred_frame(raw_name) if timing_field else "N/A"
                    ),
                    counter_width_bits=(
                        f"storage={variable.dtype}; logical width/range UNPROVED"
                        if integer_storage else "N/A"
                    ),
                    rollover_policy=(
                        "UNPROVED" if integer_storage and timing_field else "N/A"
                    ),
                    topology="exact_only", field_scope="file_or_observation_header",
                    field_value_count=int(values.size), field_native_cadence_sec="N/A",
                    field_native_acquisition_bounds="N/A: file/observation snapshot",
                    field_native_duration_sec="N/A",
                    native_validity_attributes=canonical_json(validity),
                    missing_nonfinite_policy=(
                        "native attrs: " + canonical_json(validity) if validity else
                        "UNPROVED; no native fill/missing/validity attributes"
                    ),
                    **evidence,
                    source_authority=(
                        "local raw schema/value + io.h/rawobs_collection_impl.h; "
                        "application role resolved from named consumers"
                    ),
                )

            data_specs: list[tuple[str, Any, str, str, str, str, str]] = []
            sample_variable = dataset["Data.Toltec.SampleType"]
            data_specs.append((
                "Data.Toltec.SampleType", np.ma.asarray(sample_variable[:]),
                "sample-mode enum", str(json_scalar(getattr(sample_variable, "units", "N/A"))),
                "N/A", "storage int32; enum domain from long_name only", "N/A",
            ))
            for index, identity, unit, epoch, width, rollover in ts_specs:
                data_specs.append((
                    f"Data.Toltec.Ts[:,{index}]", ts[:, index],
                    identity, unit, epoch, width, rollover,
                ))
            recv_variable = dataset["Data.Toltec.RecvTime"]
            data_specs.append((
                "Data.Toltec.RecvTime", np.ma.asarray(recv_variable[:]),
                "receive timestamp",
                str(json_scalar(getattr(recv_variable, "units", "sec"))),
                "UNPROVED", "N/A", "N/A",
            ))
            for raw_name, values, identity, unit, epoch, width, rollover in data_specs:
                variable = (
                    ts_variable if raw_name.startswith("Data.Toltec.Ts[:,") else
                    dataset[raw_name]
                )
                validity = variable_metadata(variable)["validity_attributes"]
                append_row(
                    field_id=stable_id(timing["interface_id"], raw_name),
                    raw_name=raw_name, raw_container="data", availability="available",
                    application_role=(
                        "alignment_consumed" if raw_name.startswith("Data.Toltec.Ts[:,") else
                        "available_unconsumed_by_alignment"
                    ),
                    source_dtype=str(values.dtype), source_dimensions='["time"]',
                    source_shape=canonical_json(["time"]),
                    scientific_identity=identity, unit=unit,
                    unit_authority_status=(
                        "descriptive_long_name_only" if "long_name only" in unit else
                        "proved_by_raw_attribute" if unit not in {"N/A", "UNPROVED"} else
                        "unproved"
                    ),
                    epoch_or_clock=epoch, counter_width_bits=width,
                    rollover_policy=rollover, topology="exact_only",
                    field_scope="native_sample_stream",
                    field_value_count=int(values.size),
                    field_native_cadence_sec=timing["cadence_sec"],
                    field_native_acquisition_bounds=canonical_json({
                        "first_sec": timing["native_first_time_sec"],
                        "last_sec": timing["native_last_time_sec"],
                    }),
                    field_native_duration_sec=timing["native_duration_sec"],
                    native_validity_attributes=canonical_json(validity),
                    missing_nonfinite_policy=(
                        "native attrs: " + canonical_json(validity) if validity else
                        "UNPROVED; no native fill/missing/validity attributes"
                    ),
                    **array_value_evidence(values),
                    source_authority=(
                        "local raw schema/value projection + "
                        "timestream_alignment_helpers.h/todproc_alignment_impl.h"
                    ),
                )
    if len(header_registries) != 1:
        raise RuntimeError("selected detector header registries differ across files")
    header_registry = next(iter(header_registries))
    expected_header_digest = "4082642c7571af87cbcefbcfbe52cb64e3204e45d9d5ca78323f5ef010172c47"
    measured_header_digest = sha256_text("\n".join(header_registry) + "\n")
    if len(header_registry) != 37 or measured_header_digest != expected_header_digest:
        raise RuntimeError(
            "selected detector header registry changed: "
            f"count={len(header_registry)} sha256={measured_header_digest}"
        )
    return result


def cxx_round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.trunc(values + np.copysign(0.5, values)).astype(np.int64)


def numeric_nearest_grid_slot(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    right = np.searchsorted(grid, values, side="left")
    slots = np.clip(right, 0, grid.size - 1)
    interior = (right > 0) & (right < grid.size)
    interior_right = right[interior]
    interior_left = interior_right - 1
    choose_right = (
        np.abs(grid[interior_right] - values[interior])
        < np.abs(values[interior] - grid[interior_left])
    )
    slots[interior] = np.where(choose_right, interior_right, interior_left)
    return slots.astype(np.int64)


def detector_times(path: Path) -> dict[str, Any]:
    with Dataset(path) as dataset:
        roach = int(np.asarray(dataset["Header.Toltec.RoachIndex"][:]).item())
        fpga = float(np.asarray(dataset["Header.Toltec.FpgaFreq"][:]).item())
        sample_rate = float(np.asarray(dataset["Header.Toltec.SampleFreq"][:]).item())
        accum_len = int(np.asarray(dataset["Header.Toltec.AccumLen"][:]).item())
        obs_start = int(np.asarray(dataset["Header.Toltec.ObsStartTime"][:]).item())
        obs_end = int(np.asarray(dataset["Header.Toltec.ObsEndTime"][:]).item())
        ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
        recv = np.asarray(dataset["Data.Toltec.RecvTime"][:], dtype=np.float64)

    anchor = int(float(ts[0, 0]) + float(ts[0, 5]) * 1.0e-9 - 0.5)
    signed_delta = ts[:, 2].astype(np.float64) - ts[:, 4].astype(np.float64)
    current_ticks = np.where(
        signed_delta < 0,
        signed_delta + float(UINT32_MODULUS - 1),
        signed_delta,
    )
    current_time = anchor + ts[:, 1].astype(np.float64) + current_ticks / fpga

    clock_u32 = ts[:, 2] % UINT32_MODULUS
    pps_u32 = ts[:, 4] % UINT32_MODULUS
    modulo_phase_ticks = (clock_u32 - pps_u32) % UINT32_MODULUS
    logical_ticks = ts[:, 1] * int(fpga) + modulo_phase_ticks
    tick_steps = np.diff(logical_ticks)
    nominal_ticks = int(round(fpga / sample_rate))
    modeled_time = anchor + ts[:, 1].astype(np.float64) + modulo_phase_ticks / fpga
    return {
        "roach": roach,
        "fpga": fpga,
        "sample_rate": sample_rate,
        "accum_len": accum_len,
        "obs_start": obs_start,
        "obs_end": obs_end,
        "ts": ts,
        "recv": recv,
        "current_time": current_time,
        "modeled_time": modeled_time,
        "negative_delta": signed_delta < 0,
        "tick_steps": tick_steps,
        "nominal_ticks": nominal_ticks,
    }


def collect_timing_comparison(
    mode: str, obsnum: str, paths: list[Path], source_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    streams = [detector_times(path) | {"path": path} for path in paths]
    if not streams:
        raise RuntimeError(f"no detector files for {mode} {obsnum}")
    roaches = [stream["roach"] for stream in streams]
    if len(set(roaches)) != len(roaches):
        raise RuntimeError(
            f"duplicate Header.Toltec.RoachIndex for {mode} {obsnum}: {roaches}"
        )
    sample_rates = {stream["sample_rate"] for stream in streams}
    if len(sample_rates) != 1:
        raise RuntimeError(f"inconsistent sample rates for {mode} {obsnum}: {sample_rates}")
    cadence = 1.0 / next(iter(sample_rates))
    phase = max(float(stream["current_time"][0]) for stream in streams)
    overlap_end = min(float(stream["current_time"][-1]) for stream in streams)
    current_count = int((overlap_end - phase) / cadence) + 1
    current_grid_end = phase + cadence * (current_count - 1)
    common_grid = np.linspace(phase, current_grid_end, current_count, dtype=np.float64)
    realized_grid_step = (
        float(common_grid[1] - common_grid[0]) if current_count > 1 else "N/A"
    )
    rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for stream in sorted(streams, key=lambda item: item["roach"]):
        time = stream["current_time"]
        q = (time - phase) / cadence
        current_mask_slot = cxx_round_away_from_zero(q)
        current_numeric_slot = numeric_nearest_grid_slot(time, common_grid)
        proposed_slot = np.floor(q + 0.5).astype(np.int64)
        clipped_mask_slot = np.clip(current_mask_slot, 0, current_count - 1)
        current_residual = time - common_grid[clipped_mask_slot]
        current_valid = (
            (current_mask_slot >= 0) & (current_mask_slot < current_count)
            & (np.abs(current_residual) <= cadence / 2.0)
        )
        fractional_slot = q - np.floor(q)
        exact_half_ties = fractional_slot == 0.5
        near_half_ties = (
            np.isclose(fractional_slot, 0.5, rtol=0.0, atol=1.0e-12)
            & ~exact_half_ties
        )
        mask_proposed_diff = current_mask_slot != proposed_slot
        numeric_proposed_diff = current_numeric_slot != proposed_slot
        mask_numeric_diff = current_mask_slot != current_numeric_slot
        modeled_changed = stream["modeled_time"] != time
        modeled_q = (stream["modeled_time"] - phase) / cadence
        modeled_slot = np.floor(modeled_q + 0.5).astype(np.int64)
        clipped_proposed_slot = np.clip(proposed_slot, 0, current_count - 1)
        proposed_residual = time - common_grid[clipped_proposed_slot]
        proposed_in_range = (proposed_slot >= 0) & (proposed_slot < current_count)
        proposed_valid_current_tolerance = (
            proposed_in_range & (np.abs(proposed_residual) <= cadence / 2.0)
        )
        proposed_valid_strict_tolerance = (
            proposed_in_range
            & (np.abs(proposed_residual) < PROVISIONAL_SLOT_TOLERANCE_SEC)
        )
        current_invalid_proposed_valid = ~current_valid & proposed_valid_current_tolerance
        current_valid_proposed_invalid = current_valid & ~proposed_valid_current_tolerance
        current_invalid_proposed_strict_valid = (
            ~current_valid & proposed_valid_strict_tolerance
        )
        current_valid_proposed_strict_invalid = (
            current_valid & ~proposed_valid_strict_tolerance
        )
        finite_residual = current_residual[current_valid]
        unique_slots, slot_counts = np.unique(proposed_slot[current_valid], return_counts=True)
        collisions = int(np.sum(slot_counts > 1))
        mask_vector = np.zeros(current_count, dtype=bool)
        mask_vector[current_mask_slot[current_valid]] = True
        numeric_rejected = current_valid & ~mask_vector[current_numeric_slot]
        rows.append({
            "mode": mode,
            "obsnum": obsnum,
            "source_kind": source_kind,
            "interface_id": f"toltec{stream['roach']}",
            "raw_path": str(stream["path"]),
            "timing_projection_sha256": timing_projection_digest(
                stream["path"],
                (
                    "Header.Toltec.RoachIndex", "Header.Toltec.Master",
                    "Header.Toltec.CompileTime",
                    "Header.Toltec.ObsStartTime", "Header.Toltec.ObsEndTime",
                    "Header.Toltec.ObsNum", "Header.Toltec.SubObsNum",
                    "Header.Toltec.ScanNum", "Header.Toltec.FpgaFreq",
                    "Header.Toltec.AccumLen", "Header.Toltec.SampleFreq",
                    "Data.Toltec.SampleType", "Data.Toltec.Ts",
                    "Data.Toltec.RecvTime",
                ),
            ),
            "native_rows": int(time.size),
            "native_first_time_sec": float(time[0]),
            "native_last_time_sec": float(time[-1]),
            "native_duration_sec": float(time[-1] - time[0]),
            "header_obs_start": stream["obs_start"],
            "header_obs_end": stream["obs_end"],
            "sample_rate_hz": stream["sample_rate"],
            "fpga_freq_hz": stream["fpga"],
            "accum_len": stream["accum_len"],
            "cadence_sec": cadence,
            "phase_sec_current_compatible_provisional": phase,
            "current_grid_count": current_count,
            "current_grid_end_sec": current_grid_end,
            "realized_linspaced_grid_step_sec": realized_grid_step,
            "current_mapped_rows": int(current_valid.sum()),
            "proposed_mapped_rows_current_tolerance": int(
                proposed_valid_current_tolerance.sum()
            ),
            "proposed_mapped_rows_strict_candidate_tolerance": int(
                proposed_valid_strict_tolerance.sum()
            ),
            "current_invalid_proposed_valid_current_tolerance": int(
                current_invalid_proposed_valid.sum()
            ),
            "current_valid_proposed_invalid_current_tolerance": int(
                current_valid_proposed_invalid.sum()
            ),
            "current_invalid_proposed_valid_strict_candidate_tolerance": int(
                current_invalid_proposed_strict_valid.sum()
            ),
            "current_valid_proposed_invalid_strict_candidate_tolerance": int(
                current_valid_proposed_strict_invalid.sum()
            ),
            "both_current_and_proposed_invalid_current_tolerance": int(np.sum(
                ~current_valid & ~proposed_valid_current_tolerance
            )),
            "current_mask_numeric_slot_disagreements": int(np.sum(mask_numeric_diff & current_valid)),
            "round_half_up_vs_current_mask_changes": int(np.sum(mask_proposed_diff & current_valid)),
            "round_half_up_vs_current_numeric_changes": int(np.sum(numeric_proposed_diff & current_valid)),
            "round_half_up_union_changes": int(np.sum(
                current_valid & (mask_proposed_diff | numeric_proposed_diff)
            )),
            "round_half_up_vs_current_mask_changes_all_native": int(np.sum(mask_proposed_diff)),
            "round_half_up_vs_current_numeric_changes_all_native": int(np.sum(numeric_proposed_diff)),
            "numeric_rows_rejected_by_current_mask": int(np.sum(numeric_rejected)),
            "exact_half_ties": int(exact_half_ties.sum()),
            "near_half_ties_atol_1e_12": int(near_half_ties.sum()),
            "residual_min_sec": float(np.min(finite_residual)),
            "residual_p01_sec": float(np.quantile(finite_residual, 0.01)),
            "residual_median_sec": float(np.median(finite_residual)),
            "residual_p99_sec": float(np.quantile(finite_residual, 0.99)),
            "residual_max_sec": float(np.max(finite_residual)),
            "residual_abs_max_sec": float(np.max(np.abs(finite_residual))),
            "half_sample_margin_sec": float(cadence / 2.0 - np.max(np.abs(finite_residual))),
            "packet_gap_count_current_test": int(np.sum(np.diff(stream["ts"][:, 3]) > 1)),
            "tick_step_min_delta_from_nominal": int(np.min(stream["tick_steps"]) - stream["nominal_ticks"]),
            "tick_step_max_delta_from_nominal": int(np.max(stream["tick_steps"]) - stream["nominal_ticks"]),
            "rows_one_tick_low_if_modulo_2_32": int(stream["negative_delta"].sum()),
            "binary64_timestamps_changed_by_modeled_modulus": int(modeled_changed.sum()),
            "modeled_modulus_slot_changes": int(np.sum(modeled_slot != proposed_slot)),
            "per_interface_slot_collisions": collisions,
        })
        changed_union = (
            mask_proposed_diff | numeric_proposed_diff
            | (current_valid != proposed_valid_current_tolerance)
            | (current_valid != proposed_valid_strict_tolerance)
        )
        rows[-1]["all_native_comparison_changed_rows"] = int(changed_union.sum())
        for index in np.flatnonzero(changed_union):
            reasons = []
            if mask_proposed_diff[index]:
                reasons.append("round_half_up_slot_differs_from_current_mask_slot")
            if numeric_proposed_diff[index]:
                reasons.append("round_half_up_slot_differs_from_current_numeric_slot")
            if current_valid[index] != proposed_valid_current_tolerance[index]:
                reasons.append("current_half_sample_support_class_changes")
            if current_valid[index] != proposed_valid_strict_tolerance[index]:
                reasons.append("provisional_strict_tolerance_support_class_changes")
            changed.append({
                "mode": mode, "obsnum": obsnum,
                "interface_id": f"toltec{stream['roach']}",
                "native_row": int(index), "native_time_sec": float(time[index]),
                "current_mask_slot": int(current_mask_slot[index]),
                "current_numeric_slot": int(current_numeric_slot[index]),
                "round_half_up_slot": int(proposed_slot[index]),
                "current_valid": bool(current_valid[index]),
                "proposed_valid_current_tolerance": bool(
                    proposed_valid_current_tolerance[index]
                ),
                "proposed_valid_strict_candidate_tolerance": bool(
                    proposed_valid_strict_tolerance[index]
                ),
                "comparison_domain": (
                    "ordinary_current_support" if current_valid[index] else
                    "proposed_only_support" if proposed_valid_current_tolerance[index] else
                    "edge_outside_current_and_proposed_support"
                ),
                "residual_sec": float(proposed_residual[index]),
                "reason": "; ".join(reasons),
            })
    summary = {
        "mode": mode,
        "obsnum": obsnum,
        "interface_count": len(streams),
        "sample_rate_hz": next(iter(sample_rates)),
        "cadence_sec": cadence,
        "phase_sec_current_compatible_provisional": phase,
        "phase_authority_status": "UNPROVED; preserves current max-first-detector anchor only",
        "current_overlap_end_sec": overlap_end,
        "current_grid_count": current_count,
        "current_grid_end_sec": current_grid_end,
        "native_rows": sum(int(stream["current_time"].size) for stream in streams),
        "current_mapped_rows": sum(row["current_mapped_rows"] for row in rows),
        "proposed_mapped_rows_current_tolerance": sum(
            row["proposed_mapped_rows_current_tolerance"] for row in rows
        ),
        "proposed_mapped_rows_strict_candidate_tolerance": sum(
            row["proposed_mapped_rows_strict_candidate_tolerance"] for row in rows
        ),
        "current_invalid_proposed_valid_current_tolerance": sum(
            row["current_invalid_proposed_valid_current_tolerance"] for row in rows
        ),
        "current_valid_proposed_invalid_current_tolerance": sum(
            row["current_valid_proposed_invalid_current_tolerance"] for row in rows
        ),
        "current_invalid_proposed_valid_strict_candidate_tolerance": sum(
            row["current_invalid_proposed_valid_strict_candidate_tolerance"]
            for row in rows
        ),
        "current_valid_proposed_invalid_strict_candidate_tolerance": sum(
            row["current_valid_proposed_invalid_strict_candidate_tolerance"]
            for row in rows
        ),
        "both_current_and_proposed_invalid_current_tolerance": sum(
            row["both_current_and_proposed_invalid_current_tolerance"] for row in rows
        ),
        "edge_only_native_rows_outside_current_overlap": sum(
            int(stream["current_time"].size) for stream in streams
        ) - sum(row["current_mapped_rows"] for row in rows),
        "current_mask_numeric_slot_disagreements": sum(
            row["current_mask_numeric_slot_disagreements"] for row in rows
        ),
        "round_half_up_changed_rows": sum(row["round_half_up_union_changes"] for row in rows),
        "round_half_up_vs_current_mask_changes_all_native": sum(
            row["round_half_up_vs_current_mask_changes_all_native"] for row in rows
        ),
        "round_half_up_vs_current_numeric_changes_all_native": sum(
            row["round_half_up_vs_current_numeric_changes_all_native"] for row in rows
        ),
        "round_half_up_vs_current_mask_changes": sum(
            row["round_half_up_vs_current_mask_changes"] for row in rows
        ),
        "round_half_up_vs_current_numeric_changes": sum(
            row["round_half_up_vs_current_numeric_changes"] for row in rows
        ),
        "numeric_rows_rejected_by_current_mask": sum(
            row["numeric_rows_rejected_by_current_mask"] for row in rows
        ),
        "exact_half_ties": sum(row["exact_half_ties"] for row in rows),
        "near_half_ties_atol_1e_12": sum(
            row["near_half_ties_atol_1e_12"] for row in rows
        ),
        "all_native_comparison_changed_rows": sum(
            row["all_native_comparison_changed_rows"] for row in rows
        ),
        "max_abs_residual_sec": max(row["residual_abs_max_sec"] for row in rows),
        "minimum_half_sample_margin_sec": min(row["half_sample_margin_sec"] for row in rows),
        "packet_gap_count_current_test": sum(row["packet_gap_count_current_test"] for row in rows),
        "max_measured_native_jitter_ticks": max(
            max(abs(row["tick_step_min_delta_from_nominal"]),
                abs(row["tick_step_max_delta_from_nominal"]))
            for row in rows
        ),
        "rows_one_tick_low_if_modulo_2_32": sum(row["rows_one_tick_low_if_modulo_2_32"] for row in rows),
        "binary64_timestamps_changed_by_modeled_modulus": sum(
            row["binary64_timestamps_changed_by_modeled_modulus"] for row in rows
        ),
        "modeled_modulus_slot_changes": sum(row["modeled_modulus_slot_changes"] for row in rows),
    }
    return rows, summary, changed


def telescope_metrics(path: Path) -> dict[str, Any]:
    with Dataset(path) as dataset:
        tel_time = np.asarray(dataset["Data.TelescopeBackend.TelTime"][:], dtype=float)
        hold = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=float)
        pps_time = np.asarray(dataset["Data.TelescopeBackend.PpsTime"][:], dtype=float)
        finite = {}
        for name, variable in dataset.variables.items():
            if not name.startswith("Data.TelescopeBackend."):
                continue
            values = np.asarray(variable[:])
            finite[name] = bool(np.isfinite(values).all()) if np.issubdtype(values.dtype, np.number) else True
    steps = np.diff(tel_time)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": int(tel_time.size),
        "first_tel_time_sec": float(tel_time[0]),
        "last_tel_time_sec": float(tel_time[-1]),
        "duration_sec": float(tel_time[-1] - tel_time[0]),
        "cadence_min_sec": float(np.min(steps)),
        "cadence_median_sec": float(np.median(steps)),
        "cadence_max_sec": float(np.max(steps)),
        "nonpositive_cadence_count": int(np.sum(steps <= 0)),
        "all_telescope_data_finite": bool(all(finite.values())),
        "hold_unique_values": sorted(float(value) for value in np.unique(hold)),
        "hold_transition_count": int(np.sum(np.diff(hold) != 0)),
        "pps_time_median_step_sec": float(np.median(np.diff(pps_time))),
        "pps_time_max_step_sec": float(np.max(np.diff(pps_time))),
    }


def telescope_state_operator_comparison(
    path: Path, phase: float, count: int, cadence: float,
) -> dict[str, Any]:
    grid = phase + cadence * np.arange(count, dtype=np.float64)
    with Dataset(path) as dataset:
        tel_time = np.asarray(dataset["Data.TelescopeBackend.TelTime"][:], dtype=float)
        hold = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=float)
        pps_time = np.asarray(dataset["Data.TelescopeBackend.PpsTime"][:], dtype=float)
    if grid[0] < tel_time[0] or grid[-1] > tel_time[-1]:
        raise RuntimeError(f"telescope support does not enclose grid for {path}")
    current_hold = np.interp(grid, tel_time, hold)
    left_indices = np.searchsorted(tel_time, grid, side="right") - 1
    left_hold = hold[left_indices]
    hold_diff = current_hold != left_hold
    native_hold_values = set(float(value) for value in np.unique(hold))
    current_not_native = np.asarray(
        [float(value) not in native_hold_values for value in current_hold], dtype=bool,
    )
    current_pps = np.interp(grid, tel_time, pps_time)
    native_pps_values = set(float(value) for value in np.unique(pps_time))
    pps_not_native = sum(float(value) not in native_pps_values for value in current_pps)
    nearest_index = np.searchsorted(tel_time, grid)
    nearest_index = np.clip(nearest_index, 1, tel_time.size - 1)
    choose_left = (
        np.abs(grid - tel_time[nearest_index - 1])
        <= np.abs(tel_time[nearest_index] - grid)
    )
    nearest_index = np.where(choose_left, nearest_index - 1, nearest_index)
    nearest_residual = np.abs(grid - tel_time[nearest_index])
    return {
        "path": str(path),
        "grid_count": int(count),
        "grid_first_sec": float(grid[0]),
        "grid_last_sec": float(grid[-1]),
        "leading_telescope_support_sec": float(grid[0] - tel_time[0]),
        "trailing_telescope_support_sec": float(tel_time[-1] - grid[-1]),
        "nearest_telescope_residual_median_sec": float(np.median(nearest_residual)),
        "nearest_telescope_residual_p95_sec": float(np.quantile(nearest_residual, 0.95)),
        "nearest_telescope_residual_max_sec": float(np.max(nearest_residual)),
        "hold_native_transition_count": int(np.sum(np.diff(hold) != 0)),
        "hold_unique_native_values": sorted(native_hold_values),
        "hold_left_half_open_changed_rows": int(hold_diff.sum()),
        "hold_left_half_open_changed_fraction": float(hold_diff.mean()),
        "hold_current_interpolated_values_not_native_rows": int(current_not_native.sum()),
        "hold_max_abs_numeric_change": float(np.max(np.abs(current_hold - left_hold))),
        "pps_time_current_interpolated_values_not_native_rows": int(pps_not_native),
        "exact_grid_tel_time_coincidences": int(np.sum(nearest_residual == 0.0)),
        "authority_status": "measured operator sensitivity only; candidate registry not approved",
    }


def offset_rows(
    chain_id: str,
    config_path: Path,
    config_sha256_expected: str,
    provenance_path: Path | None,
    provenance_sha256_expected: str | None,
    present_roaches: set[int],
) -> list[dict[str, Any]]:
    config_sha256 = require_sha256(config_path, config_sha256_expected)
    values = config_interface_offsets(config_path)
    expected_interfaces = {f"toltec{index}" for index in range(13)} | {"hwpr"}
    provenance_sha256: str | None = None
    provenance_schema = "MISSING"
    provenance_has_offsets = False
    requested_offsets: dict[str, float] | None = None
    effective_offsets: dict[str, float] | None = None
    if provenance_path is not None:
        if provenance_sha256_expected is None:
            raise RuntimeError("provenance SHA is required with provenance path")
        provenance_sha256 = require_sha256(provenance_path, provenance_sha256_expected)
        provenance = yaml.safe_load(provenance_path.read_text())
        provenance_schema = str(provenance.get("schema_version", "MISSING"))
        requested_node = provenance.get("requested", {}).get("interface_sync_offset")
        effective_node = (
            provenance.get("effective", {}).get("config", {}).get("interface_sync_offset")
        )
        if (requested_node is None) != (effective_node is None):
            raise RuntimeError(f"partial offset provenance in {provenance_path}")
        if requested_node is not None:
            requested_offsets = {
                str(name): float(value) for name, value in requested_node["offsets"].items()
            }
            effective_offsets = {
                str(name): float(value) for name, value in effective_node["offsets"].items()
            }
            if not (
                requested_node["unit"] == "s"
                and effective_node["unit"] == "s"
                and set(requested_offsets) == expected_interfaces
                and set(effective_offsets) == expected_interfaces
                and requested_offsets == effective_offsets == values
            ):
                raise RuntimeError(
                    f"config/provenance offset state mismatch for {chain_id}"
                )
            provenance_has_offsets = True
    rows = [{
        "state_chain_id": chain_id,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "provenance_path": str(provenance_path) if provenance_path else "MISSING",
        "provenance_sha256": provenance_sha256 or "MISSING",
        "provenance_schema_version": provenance_schema,
        "interface_id": "lmt",
        "requested_value_sec": "MISSING: schema exposes no telescope offset",
        "requested_source": "MISSING",
        "effective_value_sec": "MISSING",
        "effective_resolution": "MISSING",
        "observation_resolved_value_sec": "MISSING",
        "observation_resolved_source": "MISSING",
        "realized_value_sec": "MISSING",
        "realized_applied": "no modeled telescope clock conversion/offset",
        "sign": "N/A until comparable epoch and offset authority exist",
        "unit": "s (required contract; not represented)",
        "reference": "detector clock required by ALIGN-OD2; runtime state absent",
        "application_stage": "MISSING",
        "exactly_once_status": "UNPROVED/unimplemented",
        "uncertainty_or_bound": "MISSING",
        "authority_status": "unavailable: telescope offset/epoch state is not modeled",
    }]
    for interface in [f"toltec{index}" for index in range(13)] + ["hwpr"]:
        value = values.get(interface, 0.0)
        is_hwpr = interface == "hwpr"
        roach = None if is_hwpr else int(interface.removeprefix("toltec"))
        present = is_hwpr or roach in present_roaches
        rows.append({
            "state_chain_id": chain_id,
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "provenance_path": str(provenance_path) if provenance_path else "MISSING",
            "provenance_sha256": provenance_sha256 or "MISSING",
            "provenance_schema_version": provenance_schema,
            "interface_id": interface,
            "requested_value_sec": value,
            "requested_source": (
                "matched config + provenance.requested" if provenance_has_offsets else
                "matched config only; provenance offset node unavailable"
            ),
            "effective_value_sec": (
                effective_offsets[interface] if effective_offsets is not None else "MISSING"
            ),
            "effective_resolution": (
                "matched provenance effective.config copies requested unchanged"
                if provenance_has_offsets else "MISSING from matched provenance"
            ),
            "observation_resolved_value_sec": "MISSING",
            "observation_resolved_source": "MISSING",
            "realized_value_sec": "MISSING",
            "realized_applied": (
                "yes, but not recorded" if (not is_hwpr and present)
                else "no; interface absent" if not present
                else "no runtime consumer"
            ),
            "sign": "positive value is added and places native coordinate later",
            "unit": "s",
            "reference": "detector clock (owner decision); runtime does not persist this fact",
            "application_stage": (
                "network_time_from_timestream_matrix before slotting"
                if not is_hwpr else "MISSING: HWPR offset has no consumer"
            ),
            "exactly_once_status": (
                "one runtime addition for present detector interface"
                if not is_hwpr and present else
                "not applicable: absent detector interface" if not present else
                "nonconformant: reported effective but ignored"
            ),
            "uncertainty_or_bound": "MISSING",
            "authority_status": (
                "conflicting/unimplemented" if is_hwpr else
                "zero-value config/runtime compatibility; provenance complete"
                if provenance_has_offsets else
                "zero requested; effective provenance unavailable; nonzero authority unproved"
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--coordination-repo", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    local_root = args.local_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_source = repo / "tools/diagnostics/sci_align_001_phase0_report.md"
    report_text = report_source.read_text()
    gate_evidence = validate_git_and_coordination_gate(
        repo, args.coordination_repo.resolve(),
    )

    telescope_source = repo / "include/citlali/core/engine/telescope.h"
    telescope_text = telescope_source.read_text()
    tel_data_map = extract_cpp_map(telescope_text, "tel_data_keys")
    tel_header_map = extract_cpp_map(telescope_text, "tel_header_keys")

    telescope_paths = sorted(
        path for path in local_root.rglob("tel_toltec_*.nc")
        if "_recomputed" not in path.name
    )
    hwpr_paths = sorted(
        path for path in local_root.rglob("*.nc")
        if re.search(r"hwp|hwpr", path.name, flags=re.IGNORECASE)
    )
    if not telescope_paths or not hwpr_paths:
        raise RuntimeError("local telescope/HWPR corpus is unavailable")

    raw_manifest = []
    for kind, paths in (("telescope", telescope_paths), ("hwpr", hwpr_paths)):
        for path in paths:
            with Dataset(path) as dataset:
                installed = (
                    int(np.asarray(dataset["Header.Toltec.HwpInstalled"][:]).item())
                    if "Header.Toltec.HwpInstalled" in dataset.variables else "N/A"
                )
                names = sorted(dataset.variables)
            raw_manifest.append({
                "kind": kind,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_sha256": schema_digest(path),
                "variable_name_registry_sha256": sha256_text("\n".join(names) + "\n"),
                "hwp_installed": installed,
            })
    write_csv(
        output / "raw_input_manifest.csv",
        ["kind", "path", "size_bytes", "sha256", "schema_sha256",
         "variable_name_registry_sha256", "hwp_installed"],
        raw_manifest,
    )
    raw_digest_by_path = {row["path"]: row["sha256"] for row in raw_manifest}
    boundary_streams = [
        boundary_stream_row(path, kind, raw_digest_by_path[str(path)])
        for kind, paths in (("telescope", telescope_paths), ("hwpr", hwpr_paths))
        for path in paths
    ]
    write_csv(
        output / "boundary_stream_inventory.csv",
        [
            "kind", "interface_id", "path", "sha256", "coordinate_raw_name",
            "coordinate_status", "coordinate_unit", "coordinate_epoch",
            "counter_width_bits", "rollover_policy", "header_sample_rate_hz",
            "sample_count", "first_coordinate_sec", "last_coordinate_sec",
            "duration_sec", "cadence_min_sec", "cadence_median_sec", "cadence_max_sec",
            "nonpositive_cadence_count", "header_acquisition_start",
            "header_acquisition_end", "header_acquisition_unit_epoch",
            "all_coordinate_values_finite", "missing_nonfinite_policy", "source_authority",
        ],
        boundary_streams,
    )
    telescope_hwpr_timing_inventory = [
        row
        for kind, paths in (("telescope", telescope_paths), ("hwpr", hwpr_paths))
        for path in paths
        for row in telescope_hwpr_timing_rows(
            path, kind, raw_digest_by_path[str(path)],
        )
    ]
    write_csv(
        output / "telescope_hwpr_timing_field_inventory.csv",
        [
            "kind", "interface_id", "path", "sha256", "raw_name",
            "parent_raw_name", "scientific_identity", "source_dtype",
            "source_dimensions", "source_shape", "unit", "epoch_or_clock",
            "counter_width_bits", "rollover_policy", "sequence_status",
            "sample_or_element_count", "outer_record_count", "packed_width",
            "first_value", "last_value", "value_min", "value_max",
            "duration_native_units", "step_min_native_units",
            "step_median_native_units", "step_max_native_units",
            "nonpositive_step_count", "duplicate_step_count", "masked_value_count",
            "nonfinite_value_count", "native_validity_attributes",
            "missing_nonfinite_policy", "header_acquisition_start",
            "header_acquisition_end", "header_acquisition_unit_epoch",
            "source_authority",
        ],
        telescope_hwpr_timing_inventory,
    )

    tel_union = collect_schema_union(telescope_paths)
    hwpr_union = collect_schema_union(hwpr_paths)
    data_alias = dict(tel_data_map)
    header_alias = dict(tel_header_map)
    direct_headers = {
        "Header.Dcs.ObsGoal": "Telescope::obs_goal",
        "Header.Dcs.ObsPgm": "Telescope::obs_pgm",
        "Header.Map.MapCoord": "Telescope::map_coord",
        "Header.Source.SourceName": "Telescope::source_name",
        "Header.Dcs.ProjectId": "Telescope::project_id",
        "Header.Sim.Jobkey": "Telescope::sim_job_key",
    }
    registry = []
    tel_names = sorted(set(tel_union) | set(data_alias) | set(header_alias) | set(direct_headers))
    for name in tel_names:
        info = tel_union.get(name)
        if name in data_alias:
            availability = "available_configured" if info else "unavailable_configured"
            native = f'telescope.tel_data["{data_alias[name]}"]'
            aligned = f'TOD::{data_alias[name]} (current metadata hard-coded rad)'
        elif name in header_alias:
            availability = "available_configured" if info else "unavailable_configured"
            native = f'telescope.tel_header["{header_alias[name]}"]'
            aligned = "TOD/FITS first element only (vector truncation where applicable)"
        elif name in direct_headers:
            availability = "available_direct_read" if info else "unavailable_direct_read"
            native = direct_headers[name]
            aligned = direct_headers[name]
        else:
            availability = "raw_available_unconsumed"
            native = "none"
            aligned = "none"
        container = "data" if name.startswith("Data.") else "header"
        registry.append(registry_row(
            interface="lmt", raw_name=name, container=container,
            info=info, availability=availability, native_output=native,
            aligned_output=aligned,
            source_authority="local raw schema + telescope.h/telescope.cpp boundary",
        ))

    expected_hwpr = {
        "Data.Hwp.": "calib.hwpr_angle",
        "Data.Hwp.Ts": "calib.hwpr_ts",
        "Data.Hwp.Uts": "calib.hwpr_recvt",
        "Header.Hwp.Installed": "calib.run_hwpr (simulation)",
    }
    hwpr_names = sorted(set(hwpr_union) | set(expected_hwpr))
    for name in hwpr_names:
        info = hwpr_union.get(name)
        if name in expected_hwpr:
            availability = "available_code_expected" if info else "unavailable_code_expected"
            native = expected_hwpr[name]
            aligned = expected_hwpr[name]
        elif name in {"Header.Toltec.HwpInstalled", "Header.Toltec.FpgaFreq"}:
            availability = "available_code_consumed"
            native = "Calib::get_hwpr"
            aligned = "calibration state"
        else:
            availability = "raw_available_unconsumed"
            native = "none"
            aligned = "none"
        container = "data" if name.startswith("Data.") else "header"
        registry.append(registry_row(
            interface="hwpr", raw_name=name, container=container,
            info=info, availability=availability, native_output=native,
            aligned_output=aligned,
            source_authority="local raw schema + Calib::get_hwpr boundary",
        ))
    registry_fields = [
        "field_id", "raw_name", "raw_container", "interface_id", "availability",
        "source_dtype", "source_shape", "scientific_identity", "unit",
        "unit_authority_status", "epoch_or_clock", "counter_width_bits",
        "rollover_policy", "coordinate_frame", "candidate_scientific_class",
        "topology", "topology_parameters", "validity_missing_nonfinite_policy",
        "maximum_support_span_sec", "native_cadence", "native_acquisition_bounds",
        "native_duration", "permitted_operator", "native_output_identity",
        "aligned_output_identity", "source_authority", "confidence",
        "unresolved_conflicts",
    ]
    write_csv(output / "field_registry.csv", registry_fields, registry)

    point_dir = local_root / "2025-C1-COM-01/data"
    beammap_dir = local_root / "beammaps/data"
    accepted_beammap_dir = local_root / "citlali-validation/v1/beammaps/data"
    timing_rows = []
    timing_summaries = []
    changed_rows = []
    for obsnum in POINT_OBSERVATIONS:
        paths = sorted(point_dir.glob(f"toltec[0-9]*_{obsnum}_000_0002_*.nc"))
        rows, summary, changed = collect_timing_comparison("point", obsnum, paths, "owner_local")
        timing_rows.extend(rows)
        timing_summaries.append(summary)
        changed_rows.extend(changed)
    beam_paths = sorted(beammap_dir.glob("toltec[0-9]*_152307_000_0002_*.nc"))
    rows, summary, changed = collect_timing_comparison(
        "beammap", "152307", beam_paths, "owner_local_timing_only",
    )
    timing_rows.extend(rows)
    timing_summaries.append(summary)
    changed_rows.extend(changed)
    accepted_beam_paths = sorted(
        accepted_beammap_dir.glob("toltec[0-9]*_148670_000_0002_*.nc")
    )
    rows, summary, changed = collect_timing_comparison(
        "beammap", "148670", accepted_beam_paths, "accepted_validation_corpus",
    )
    timing_rows.extend(rows)
    timing_summaries.append(summary)
    changed_rows.extend(changed)

    timing_fields = list(timing_rows[0])
    write_csv(output / "slot_mapping_comparison.csv", timing_fields, timing_rows)
    detector_inventory = detector_timing_field_rows(timing_rows)
    detector_field_counts = Counter(
        (row["mode"], row["obsnum"], row["interface_id"])
        for row in detector_inventory
    )
    if set(detector_field_counts.values()) != {47} or len(detector_inventory) != 5170:
        raise RuntimeError(
            "detector boundary inventory is incomplete: "
            f"per-interface={sorted(set(detector_field_counts.values()))} "
            f"total={len(detector_inventory)}"
        )
    detector_header_rows = sum(
        row["raw_container"] == "header" for row in detector_inventory
    )
    if detector_header_rows != 4070:
        raise RuntimeError(f"detector header inventory row count is {detector_header_rows}")
    write_csv(
        output / "detector_timing_inventory.csv",
        [
            "mode", "obsnum", "interface_id", "raw_path",
            "timing_projection_sha256", "field_id", "raw_name", "raw_container",
            "availability", "application_role", "source_dtype", "source_dimensions",
            "source_shape", "scientific_identity", "unit", "unit_authority_status",
            "epoch_or_clock", "counter_width_bits", "rollover_policy", "topology",
            "field_scope",
            "field_value_count", "field_native_cadence_sec",
            "field_native_acquisition_bounds", "field_native_duration_sec",
            "detector_stream_context_sample_rate_hz",
            "detector_stream_context_sample_count",
            "detector_stream_context_first_time_sec",
            "detector_stream_context_last_time_sec",
            "detector_stream_context_duration_sec", "header_acquisition_start",
            "header_acquisition_end", "header_acquisition_duration_unproved_units",
            "header_acquisition_unit_epoch",
            "common_grid_first_sec", "common_grid_last_sec", "common_grid_duration_sec",
            "native_validity_attributes", "missing_nonfinite_policy",
            "masked_value_count", "nonfinite_value_count", "raw_value_sha256",
            "value_summary", "source_authority",
        ],
        detector_inventory,
    )
    write_csv(
        output / "changed_rows.csv",
        ["mode", "obsnum", "interface_id", "native_row", "native_time_sec",
         "current_mask_slot", "current_numeric_slot", "round_half_up_slot",
         "current_valid", "proposed_valid_current_tolerance",
         "proposed_valid_strict_candidate_tolerance",
         "comparison_domain", "residual_sec", "reason"],
        changed_rows,
    )

    max_residual = max(item["max_abs_residual_sec"] for item in timing_summaries)
    cadence = timing_summaries[0]["cadence_sec"]
    fpga_rates = {row["fpga_freq_hz"] for row in timing_rows}
    accum_lengths = {row["accum_len"] for row in timing_rows}
    if len(fpga_rates) != 1 or len(accum_lengths) != 1:
        raise RuntimeError(
            f"inconsistent detector clock headers: fpga={fpga_rates}, accum={accum_lengths}"
        )
    fpga = next(iter(fpga_rates))
    accum_len = next(iter(accum_lengths))
    if not math.isclose(fpga / accum_len, 1.0 / cadence, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("FpgaFreq/AccumLen disagrees with SampleFreq/cadence")
    binary64_ulp = float(np.spacing(timing_summaries[0]["phase_sec_current_compatible_provisional"]))
    candidate_tolerance = math.ceil((max_residual + 1.0e-6) * 1.0e6) / 1.0e6
    if candidate_tolerance != PROVISIONAL_SLOT_TOLERANCE_SEC:
        raise RuntimeError(
            "measured provisional tolerance changed: "
            f"{candidate_tolerance} != {PROVISIONAL_SLOT_TOLERANCE_SEC}"
        )
    comparison_summary = {
        "scope": {
            "point_observations": list(POINT_OBSERVATIONS),
            "beammap_observations": ["152307", "148670"],
            "detector_interface_count_per_observation": 11,
            "offsets_sec": "all zero in representative owner-local config",
        },
        "convention": {
            "cadence_sec": cadence,
            "nominal_sample_rate_hz": 1.0 / cadence,
            "phase": "current-compatible max first detector timestamp; UNPROVED as frozen authority",
            "current_grid_constructor": "Eigen::VectorXd::LinSpaced over requested cadence-derived endpoints",
            "slot_operator_current_mask": "C++ std::round (half away from zero)",
            "slot_operator_current_numeric": "lower_bound nearest common-grid value; exact tie chooses left",
            "slot_operator_proposed": "floor(q + 0.5) round-half-up",
            "current_tolerance_sec": cadence / 2.0,
            "candidate_tolerance_sec": candidate_tolerance,
            "candidate_derivation": "ceil(max measured residual + 1 us guard) to 1 us",
            "candidate_authority_status": "PROVISIONAL ONLY; owner must approve/freeze",
        },
        "measured": {
            "native_rows": sum(item["native_rows"] for item in timing_summaries),
            "current_mapped_rows": sum(item["current_mapped_rows"] for item in timing_summaries),
            "proposed_mapped_rows_current_tolerance": sum(
                item["proposed_mapped_rows_current_tolerance"]
                for item in timing_summaries
            ),
            "proposed_mapped_rows_strict_candidate_tolerance": sum(
                item["proposed_mapped_rows_strict_candidate_tolerance"]
                for item in timing_summaries
            ),
            "current_invalid_proposed_valid_current_tolerance": sum(
                item["current_invalid_proposed_valid_current_tolerance"]
                for item in timing_summaries
            ),
            "current_valid_proposed_invalid_current_tolerance": sum(
                item["current_valid_proposed_invalid_current_tolerance"]
                for item in timing_summaries
            ),
            "current_invalid_proposed_valid_strict_candidate_tolerance": sum(
                item["current_invalid_proposed_valid_strict_candidate_tolerance"]
                for item in timing_summaries
            ),
            "current_valid_proposed_invalid_strict_candidate_tolerance": sum(
                item["current_valid_proposed_invalid_strict_candidate_tolerance"]
                for item in timing_summaries
            ),
            "both_current_and_proposed_invalid_current_tolerance": sum(
                item["both_current_and_proposed_invalid_current_tolerance"]
                for item in timing_summaries
            ),
            "edge_only_native_rows_outside_current_overlap": sum(
                item["edge_only_native_rows_outside_current_overlap"]
                for item in timing_summaries
            ),
            "round_half_up_changed_rows": sum(item["round_half_up_changed_rows"] for item in timing_summaries),
            "all_native_comparison_changed_rows": sum(
                item["all_native_comparison_changed_rows"] for item in timing_summaries
            ),
            "round_half_up_vs_current_mask_changes_all_native": sum(
                item["round_half_up_vs_current_mask_changes_all_native"]
                for item in timing_summaries
            ),
            "round_half_up_vs_current_numeric_changes_all_native": sum(
                item["round_half_up_vs_current_numeric_changes_all_native"]
                for item in timing_summaries
            ),
            "current_mask_numeric_slot_disagreements": sum(
                item["current_mask_numeric_slot_disagreements"] for item in timing_summaries
            ),
            "round_half_up_vs_current_mask_changes": sum(
                item["round_half_up_vs_current_mask_changes"] for item in timing_summaries
            ),
            "round_half_up_vs_current_numeric_changes": sum(
                item["round_half_up_vs_current_numeric_changes"] for item in timing_summaries
            ),
            "numeric_rows_rejected_by_current_mask": sum(
                item["numeric_rows_rejected_by_current_mask"] for item in timing_summaries
            ),
            "exact_half_ties": sum(item["exact_half_ties"] for item in timing_summaries),
            "near_half_ties_atol_1e_12": sum(
                item["near_half_ties_atol_1e_12"] for item in timing_summaries
            ),
            "max_abs_residual_sec": max_residual,
            "minimum_margin_to_half_sample_sec": min(
                item["minimum_half_sample_margin_sec"] for item in timing_summaries
            ),
            "max_native_jitter_ticks": max(
                item["max_measured_native_jitter_ticks"] for item in timing_summaries
            ),
            "native_tick_sec": 1.0 / fpga,
            "fpga_freq_hz": fpga,
            "accum_len": accum_len,
            "binary64_ulp_at_epoch_sec": binary64_ulp,
            "packet_gaps_under_current_test": sum(
                item["packet_gap_count_current_test"] for item in timing_summaries
            ),
            "rows_one_tick_low_if_modulo_2_32": sum(
                item["rows_one_tick_low_if_modulo_2_32"] for item in timing_summaries
            ),
            "binary64_timestamps_changed_by_modeled_modulus": sum(
                item["binary64_timestamps_changed_by_modeled_modulus"] for item in timing_summaries
            ),
            "modeled_modulus_slot_changes": sum(
                item["modeled_modulus_slot_changes"] for item in timing_summaries
            ),
        },
        "per_observation": timing_summaries,
        "stop": True,
        "stop_reasons": [
            "detector epoch/logical counter width/rollover authority is unproved",
            "phase authority is unproved",
            "candidate tolerance leaves a narrow measured margin and is not producer-authorized",
            "HWPR timestamp/angle schema is unavailable and HWPR offset is ignored",
            "telescope Hold/timestamp topology and maximum support spans are unproved/conflicting",
        ],
    }
    accepted_pair = [
        item for item in timing_summaries
        if item["obsnum"] in {"152389", "148670"}
    ]
    comparison_summary["accepted_pair"] = {
        "observations": ["point:152389", "beammap:148670"],
        "native_rows": sum(item["native_rows"] for item in accepted_pair),
        "ordinary_current_support_rows": sum(item["current_mapped_rows"] for item in accepted_pair),
        "proposed_support_rows_current_tolerance": sum(
            item["proposed_mapped_rows_current_tolerance"] for item in accepted_pair
        ),
        "proposed_support_rows_strict_candidate_tolerance": sum(
            item["proposed_mapped_rows_strict_candidate_tolerance"]
            for item in accepted_pair
        ),
        "edge_only_native_rows_outside_current_overlap": sum(
            item["edge_only_native_rows_outside_current_overlap"] for item in accepted_pair
        ),
        "round_half_up_changed_rows": sum(item["round_half_up_changed_rows"] for item in accepted_pair),
        "all_native_comparison_changed_rows": sum(
            item["all_native_comparison_changed_rows"] for item in accepted_pair
        ),
        "round_half_up_vs_current_mask_changes_all_native": sum(
            item["round_half_up_vs_current_mask_changes_all_native"]
            for item in accepted_pair
        ),
        "round_half_up_vs_current_numeric_changes_all_native": sum(
            item["round_half_up_vs_current_numeric_changes_all_native"]
            for item in accepted_pair
        ),
        "current_mask_numeric_slot_disagreements": sum(
            item["current_mask_numeric_slot_disagreements"] for item in accepted_pair
        ),
        "round_half_up_vs_current_mask_changes": sum(
            item["round_half_up_vs_current_mask_changes"] for item in accepted_pair
        ),
        "round_half_up_vs_current_numeric_changes": sum(
            item["round_half_up_vs_current_numeric_changes"] for item in accepted_pair
        ),
        "exact_half_ties": sum(item["exact_half_ties"] for item in accepted_pair),
        "near_half_ties_atol_1e_12": sum(
            item["near_half_ties_atol_1e_12"] for item in accepted_pair
        ),
        "current_invalid_proposed_valid_current_tolerance": sum(
            item["current_invalid_proposed_valid_current_tolerance"]
            for item in accepted_pair
        ),
        "current_valid_proposed_invalid_current_tolerance": sum(
            item["current_valid_proposed_invalid_current_tolerance"]
            for item in accepted_pair
        ),
        "current_invalid_proposed_valid_strict_candidate_tolerance": sum(
            item["current_invalid_proposed_valid_strict_candidate_tolerance"]
            for item in accepted_pair
        ),
        "current_valid_proposed_invalid_strict_candidate_tolerance": sum(
            item["current_valid_proposed_invalid_strict_candidate_tolerance"]
            for item in accepted_pair
        ),
        "packet_gaps_under_current_test": sum(item["packet_gap_count_current_test"] for item in accepted_pair),
        "conclusion": "no ordinary accepted detector row moves when phase and zero offsets are frozen",
    }
    write_json(output / "comparison_summary.json", comparison_summary)

    selected_detector_inputs = []
    for mode, obsnum, paths in (
        ("point", "152389", sorted(point_dir.glob("toltec[0-9]*_152389_000_0002_*.nc"))),
        ("beammap", "148670", accepted_beam_paths),
    ):
        for path in paths:
            with Dataset(path) as dataset:
                roach = int(np.asarray(dataset["Header.Toltec.RoachIndex"][:]).item())
            selected_detector_inputs.append({
                "mode": mode,
                "obsnum": obsnum,
                "interface_id": f"toltec{roach}",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "timing_projection_sha256": timing_projection_digest(
                    path,
                    (
                        "Header.Toltec.RoachIndex", "Header.Toltec.Master",
                        "Header.Toltec.CompileTime",
                        "Header.Toltec.ObsStartTime", "Header.Toltec.ObsEndTime",
                        "Header.Toltec.ObsNum", "Header.Toltec.SubObsNum",
                        "Header.Toltec.ScanNum", "Header.Toltec.FpgaFreq",
                        "Header.Toltec.AccumLen", "Header.Toltec.SampleFreq",
                        "Data.Toltec.SampleType", "Data.Toltec.Ts",
                        "Data.Toltec.RecvTime",
                    ),
                ),
            })
    write_csv(
        output / "selected_detector_input_manifest.csv",
        ["mode", "obsnum", "interface_id", "path", "size_bytes", "sha256",
         "timing_projection_sha256"],
        selected_detector_inputs,
    )

    representative_config = (
        local_root / "citlali-validation/v1/point/pointings_v22/reduced/"
        "citlali_o152389_0_2_c1.yaml"
    )
    accepted_config = (
        local_root / "2026-refactor/point/refactor/reduced/redu66/"
        "citlali_o152389_0_2_c1.yaml"
    )
    accepted_provenance = (
        local_root / "2026-refactor/point/refactor/reduced/redu66/152389/"
        "raw_timestream_provenance.yaml"
    )
    comparison_provenance = (
        local_root / "citlali-validation/v1/point/pointings_v22/reduced/"
        "redu00/152389/raw_timestream_provenance.yaml"
    )
    accepted_beam_root = local_root / "2026-refactor/beammap/refactor/reduced/redu18"
    accepted_beam_config = accepted_beam_root / "citlali_o148670_0_2_c1.yaml"
    accepted_beam_provenance = accepted_beam_root / "148670/raw_timestream_provenance.yaml"
    point_present_roaches = {
        int(row["interface_id"].removeprefix("toltec"))
        for row in timing_rows if row["obsnum"] == "152389"
    }
    beam_present_roaches = {
        int(row["interface_id"].removeprefix("toltec"))
        for row in timing_rows if row["obsnum"] == "148670"
    }
    offsets = []
    offsets.extend(offset_rows(
        "point_152389_comparison",
        representative_config,
        "340677ab1e873e735a44dcee84d7da9eba91a7c511f8d9229b044aa29d98f5ba",
        comparison_provenance,
        "c5c826f9ed9415a110ce2aff2c51d14dc16e6efab7b0cb8723b6487d7a6cdc44",
        point_present_roaches,
    ))
    offsets.extend(offset_rows(
        "point_152389_accepted",
        accepted_config,
        "67f6c6e6bade058fdee933fd043cbbba363187c9acbecc0f0a30d04877ce2cd7",
        accepted_provenance,
        "c5c826f9ed9415a110ce2aff2c51d14dc16e6efab7b0cb8723b6487d7a6cdc44",
        point_present_roaches,
    ))
    offsets.extend(offset_rows(
        "beammap_148670_accepted",
        accepted_beam_config,
        "aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d",
        accepted_beam_provenance,
        "ec6ad70a6583c44aba2a277ee40b93426eee39d67de4285dff318df3d845b610",
        beam_present_roaches,
    ))
    write_csv(output / "offset_state_trace.csv", list(offsets[0]), offsets)

    point_tel_raw = unique_path(
        point_dir.glob("tel_toltec_*152389*0002.nc"), "Pointing 152389 telescope input",
    )
    point_tel_recomputed = (
        local_root / "citlali-validation/v1/point/pointings_v22/reduced/"
        "tel_toltec_2026-02-19_152389_00_0002_recomputed.nc"
    )
    beam_tel = unique_path(
        beammap_dir.glob("tel_toltec_*152307*0002.nc"),
        "Beammap 152307 telescope input",
    )
    accepted_beam_tel = unique_path(
        accepted_beammap_dir.glob("tel_toltec_*148670*0002.nc"),
        "accepted Beammap 148670 telescope input",
    )
    accepted_beam_tel_configured = (
        local_root / "2026-refactor/beammap/refactor/reduced/"
        "tel_toltec_2026-01-13_148670_00_0002_recomputed.nc"
    )
    require_sha256(
        accepted_beam_tel_configured,
        "e39f5b9e3066fd20086105964dd915ff67709142d699e8a18bb58cfd9da6b7ae",
    )
    summary_by_obs = {item["obsnum"]: item for item in timing_summaries}
    telescope_comparison = {
        "point_raw": telescope_metrics(point_tel_raw),
        "point_configured_recomputed": telescope_metrics(point_tel_recomputed),
        "beammap_timing_support": telescope_metrics(beam_tel),
        "accepted_beammap_raw_timing_support": telescope_metrics(accepted_beam_tel),
        "accepted_beammap_configured_timing_support": telescope_metrics(
            accepted_beam_tel_configured
        ),
        "state_operator_sensitivity": {
            "point_152389": telescope_state_operator_comparison(
                point_tel_recomputed,
                summary_by_obs["152389"]["phase_sec_current_compatible_provisional"],
                summary_by_obs["152389"]["current_grid_count"],
                summary_by_obs["152389"]["cadence_sec"],
            ),
            "beammap_148670": telescope_state_operator_comparison(
                accepted_beam_tel_configured,
                summary_by_obs["148670"]["phase_sec_current_compatible_provisional"],
                summary_by_obs["148670"]["current_grid_count"],
                summary_by_obs["148670"]["cadence_sec"],
            ),
        },
        "point_configured_vs_raw_timing_arrays": {},
        "beammap_configured_vs_raw_timing_arrays": {},
        "operator_compatibility": {
            "ordinary_continuous_fields": "same linear operator would be bitwise/numerically unchanged; registry not approved",
            "circular_fields": "shortest-arc parameters unproved; no activation authorized",
            "Hold": "point 152389 constant zero; Beammap 152307 contains bitmask transitions and current linear interpolation is scientifically invalid",
            "PpsTime": "step-like one-Hz coordinate currently linearly interpolated; support semantics unproved",
            "source_crossing": "no slot movement observed, but Beammap scan-window impact cannot be cleared before Hold authority",
        },
    }
    selected_telescope_timing_names = (
        "Data.TelescopeBackend.AcuTime",
        "Data.TelescopeBackend.BackendTime",
        "Data.TelescopeBackend.PpsCount",
        "Data.TelescopeBackend.PpsTime",
        "Data.TelescopeBackend.TelLst",
        "Data.TelescopeBackend.TelTime",
        "Data.TelescopeBackend.TelUtDate",
        "Data.TelescopeBackend.TelUtc",
    )
    with Dataset(point_tel_raw) as raw_ds, Dataset(point_tel_recomputed) as recomputed_ds:
        for name in selected_telescope_timing_names + (
            "Data.TelescopeBackend.TelAzAct",
            "Data.TelescopeBackend.TelElAct", "Data.TelescopeBackend.SourceAz",
            "Data.TelescopeBackend.SourceEl", "Data.TelescopeBackend.Hold",
        ):
            left = np.asarray(raw_ds[name][:])
            right = np.asarray(recomputed_ds[name][:])
            comparison = {
                "shape_equal": left.shape == right.shape,
                "exact_equal": bool(np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))) if left.size else 0.0,
            }
            telescope_comparison["point_configured_vs_raw_timing_arrays"][name] = comparison
            if not comparison["shape_equal"] or not comparison["exact_equal"]:
                raise RuntimeError(f"Pointing configured/raw array mismatch for {name}")
    with Dataset(accepted_beam_tel) as raw_ds, Dataset(
        accepted_beam_tel_configured,
    ) as recomputed_ds:
        for name in selected_telescope_timing_names + (
            "Data.TelescopeBackend.TelAzAct",
            "Data.TelescopeBackend.TelElAct", "Data.TelescopeBackend.SourceAz",
            "Data.TelescopeBackend.SourceEl", "Data.TelescopeBackend.Hold",
        ):
            left = np.asarray(raw_ds[name][:])
            right = np.asarray(recomputed_ds[name][:])
            comparison = {
                "shape_equal": left.shape == right.shape,
                "exact_equal": bool(np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))) if left.size else 0.0,
            }
            telescope_comparison["beammap_configured_vs_raw_timing_arrays"][name] = comparison
            if not comparison["shape_equal"] or not comparison["exact_equal"]:
                raise RuntimeError(f"Beammap configured/raw array mismatch for {name}")
    timing_field_index = {
        (row["path"], row["raw_name"]): row
        for row in telescope_hwpr_timing_inventory
    }
    telescope_comparison["selected_timing_field_measurements"] = {
        "point_152389": [
            timing_field_index[(str(point_tel_raw), name)]
            for name in selected_telescope_timing_names
        ],
        "beammap_148670": [
            timing_field_index[(str(accepted_beam_tel), name)]
            for name in selected_telescope_timing_names
        ],
    }
    write_json(output / "telescope_compatibility.json", telescope_comparison)

    accepted_runs = repo / "validation/accepted_runs.json"
    accepted_point_ppt = (
        local_root / "2026-refactor/point/refactor/reduced/redu66/152389/raw/"
        "ppt_commissioning_pointing_152389_citlali.ecsv"
    )
    accepted_beam_apt = (
        accepted_beam_root / "148670/raw/apt_commissioning_beammap_148670_citlali.ecsv"
    )
    accepted_source_crossing = (
        accepted_beam_root / "148670/raw/source_crossing_tod/"
        "toltec_commissioning_beammap_148670_ptc_detector_tod.nc"
    )
    accepted_runs_sha = require_sha256(
        accepted_runs,
        "4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb",
    )
    accepted_provenance_sha = require_sha256(
        accepted_provenance,
        "c5c826f9ed9415a110ce2aff2c51d14dc16e6efab7b0cb8723b6487d7a6cdc44",
    )
    accepted_point_ppt_sha = require_sha256(
        accepted_point_ppt,
        "344c85500d367566b7a1b9463fc46a8cd4d8aef0671f9f7eb3891accfbb53763",
    )
    accepted_beam_apt_sha = require_sha256(
        accepted_beam_apt,
        "f1dcd7e7ea88eb47d1b494cdfac3d3b365d5a938d87b5393c97b5fcde9b5b25c",
    )
    accepted_source_crossing_sha = require_sha256(
        accepted_source_crossing,
        "948df213ac88ce516f85cf177ed33495f95ca5d26f85a5afedb5b68f548255ed",
    )
    accepted_config_sha = require_sha256(
        accepted_config,
        "67f6c6e6bade058fdee933fd043cbbba363187c9acbecc0f0a30d04877ce2cd7",
    )
    comparison_config_sha = require_sha256(
        representative_config,
        "340677ab1e873e735a44dcee84d7da9eba91a7c511f8d9229b044aa29d98f5ba",
    )
    accepted_beam_config_sha = require_sha256(
        accepted_beam_config,
        "aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d",
    )
    accepted_beam_provenance_sha = require_sha256(
        accepted_beam_provenance,
        "ec6ad70a6583c44aba2a277ee40b93426eee39d67de4285dff318df3d845b610",
    )
    point_offsets = config_interface_offsets(accepted_config)
    beam_offsets = config_interface_offsets(accepted_beam_config)
    comparison_offsets = config_interface_offsets(representative_config)
    comparison_summary["scope"]["accepted_pair_offsets"] = {
        "point_152389": point_offsets,
        "beammap_148670": beam_offsets,
        "authority": "parsed frozen accepted configs; all values exactly zero",
    }
    comparison_summary["scope"]["extended_corpus_offset_status"] = (
        "zero-offset sensitivity only; observation-specific config/realized offsets not supplied"
    )
    comparison_summary["scope"]["comparison_point_152389_offsets"] = comparison_offsets
    write_json(output / "comparison_summary.json", comparison_summary)

    application_inputs = application_input_identity_rows(
        [
            ("point_152389_comparison", representative_config, comparison_config_sha),
            ("point_152389_accepted", accepted_config, accepted_config_sha),
            ("beammap_148670_accepted", accepted_beam_config, accepted_beam_config_sha),
        ],
        local_root,
    )
    write_csv(
        output / "application_input_identity_inventory.csv",
        [
            "config_role", "config_path", "config_sha256", "item_index",
            "requested_interface", "requested_filepath", "requested_basename",
            "duplicate_requested_interface", "local_match_count",
            "local_unique_content_count", "local_paths", "local_sha256",
            "resolved_local_paths", "resolved_local_sha256",
            "content_resolution_status",
            "raw_interface_identities", "requested_interface_matches_raw",
            "reconciliation_status", "source_authority",
        ],
        application_inputs,
    )
    point_gap_grid_enabled = require_gap_grid_enabled(accepted_config)
    beam_gap_grid_enabled = require_gap_grid_enabled(accepted_beam_config)
    comparison_gap_grid_enabled = require_gap_grid_enabled(representative_config)
    point_record = accepted_record(
        accepted_runs, "point-152389-refactor-2a974e0d-redu66",
    )
    beam_record = accepted_record(
        accepted_runs, "beammap-148670-refactor-398d5127-redu18",
    )
    point_comparison = point_record["comparison"]
    if not (
        point_record["status"] == "accepted"
        and point_record["candidate"]["dirty"] is False
        and point_record["candidate"]["citlali_sha"]
        == "2a974e0dd3b76fca7e406c057026d6c6657b6159"
        and point_comparison["common_products"] == 19
        and point_comparison["skipped_records"] == 0
        and point_comparison["changed_records"] == 0
        and point_comparison["scientific_products_exact"] is True
        and point_comparison["tolerances"]["atol"] == 0
        and point_comparison["tolerances"]["rtol"] == 0
    ):
        raise RuntimeError("accepted Pointing authority no longer matches frozen phase-0 claim")
    beam_comparison = beam_record["comparison"]
    beam_differences = beam_record.get("accepted_differences", [])
    if not (
        beam_record["status"] == "accepted"
        and beam_record["candidate"]["dirty"] is False
        and beam_record["candidate"]["citlali_sha"] == "398d5127"
        and beam_comparison["common_products"] == 12
        and beam_comparison["skipped_records"] == 0
        and beam_comparison["changed_records"] == 6
        and beam_comparison["scientific_products_exact"] is True
        and beam_comparison["tolerances"]["atol"] == 2.0e-8
        and beam_comparison["tolerances"]["rtol"] == 1.0e-10
        and len(beam_differences) == 6
        and all(item.get("kind") == "inactive_config_metadata"
                for item in beam_differences)
    ):
        raise RuntimeError("accepted Beammap authority no longer matches frozen phase-0 claim")
    apt_row_count, beam_fwhm = beammap_fwhm_rows(accepted_beam_apt)
    crossing = source_crossing_metrics(accepted_source_crossing)
    science_evidence = {
        "accepted_pointing": {
            "record_id": point_record["record_id"],
            "status": "accepted historical baseline; not a phase-0 successor run",
            "ledger_status": point_record["status"],
            "ledger_comparison": point_record["comparison"],
            "candidate_sha": point_record["candidate"]["citlali_sha"],
            "accepted_runs_path": str(accepted_runs),
            "accepted_runs_sha256": accepted_runs_sha,
            "ppt_path": str(accepted_point_ppt),
            "ppt_sha256": accepted_point_ppt_sha,
            "pointing_rows": pointing_product_rows(accepted_point_ppt),
            "metric_extraction": "parsed from frozen ECSV columns; not hardcoded",
        },
        "accepted_raw_timestream_provenance": {
            "path": str(accepted_provenance),
            "sha256": accepted_provenance_sha,
            "offset_state_finding": "requested/effective present; observation/realized offsets absent",
        },
        "accepted_beammap": {
            "record_id": beam_record["record_id"],
            "status": "accepted historical baseline; not a phase-0 successor run",
            "ledger_status": beam_record["status"],
            "ledger_comparison": beam_record["comparison"],
            "candidate_sha": beam_record["candidate"]["citlali_sha"],
            "apt_path": str(accepted_beam_apt),
            "apt_sha256": accepted_beam_apt_sha,
            "apt_row_count": apt_row_count,
            "flag_zero_fwhm_arcsec": beam_fwhm,
            "metric_extraction": "parsed from frozen ECSV; flag==0 per array; NumPy quantiles",
            "config_path": str(accepted_beam_config),
            "config_sha256": accepted_beam_config_sha,
            "raw_provenance_path": str(accepted_beam_provenance),
            "raw_provenance_sha256": accepted_beam_provenance_sha,
            "accepted_gap_grid_path_enabled": beam_gap_grid_enabled,
            "config_interface_offsets_sec": beam_offsets,
            "offset_provenance_status": (
                "raw_timestream_provenance v1 has no interface_sync_offset nodes"
            ),
        },
        "accepted_source_crossing": {
            "path": str(accepted_source_crossing),
            "sha256": accepted_source_crossing_sha,
            **crossing,
            "metric_extraction": "parsed from frozen NetCDF; fit_good==1 per array; NumPy quantiles",
            "exact_crossing_time_comparison": "UNAVAILABLE: artifact does not retain per-sample time/pointing or closest-approach sample index",
        },
        "accepted_config": {
            "path": str(accepted_config),
            "sha256": accepted_config_sha,
            "role": "accepted redu66 authority",
            "accepted_gap_grid_path_enabled": point_gap_grid_enabled,
            "config_interface_offsets_sec": point_offsets,
        },
        "phase0_comparison_config": {
            "path": str(representative_config),
            "sha256": comparison_config_sha,
            "role": "newer owner-local ordinary-data comparison input; not accepted redu66",
            "gap_grid_path_enabled": comparison_gap_grid_enabled,
            "config_interface_offsets_sec": comparison_offsets,
        },
        "candidate_evidence": {
            "phase0_application_path_delta": gate_evidence[
                "live_reflog_and_scope_assertions"
            ]["application_path_delta"],
            "phase0_application_edits": bool(gate_evidence[
                "live_reflog_and_scope_assertions"
            ]["application_path_delta"]),
            "successor_reduction_available": False,
            "centroid_or_psf_degradation_tested": False,
            "reason": "phase 0 forbids application repair and Unity; direct successor products do not exist",
        },
    }
    write_json(output / "science_compatibility_evidence.json", science_evidence)

    source_paths = [
        "AGENTS.md", "doc/REFACTOR_STATUS.md", "doc/ARCHITECTURE.md",
        "doc/SCIENTIFIC_CONVENTIONS.md", "doc/RETAINED_DEBT.md",
        "doc/PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md",
        "doc/TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md",
        "doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md",
        "doc/PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md",
        "doc/PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md",
        "handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md",
        "doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md", "doc/adr/README.md",
        "include/citlali/core/engine/telescope.h",
        "include/citlali/core/engine/calib.h",
        "include/citlali/core/engine/io.h",
        "include/citlali/core/engine/detail/rawobs_collection_impl.h",
        "src/citlali/core/engine/telescope.cpp",
        "src/citlali/core/engine/calib.cpp",
        "include/citlali/core/engine/detail/todproc_alignment_impl.h",
        "include/citlali/core/engine/detail/kidsproc_gaps_impl.h",
        "include/citlali/core/pipeline/timestream_alignment_helpers.h",
        "include/citlali/core/pipeline/telescope_timestream_alignment.h",
        "include/citlali/core/config/interface_sync_config.h",
        "include/citlali/core/config/interface_sync_config_validation.h",
        "include/citlali/core/config/reduction_config_validation.h",
        "include/citlali/core/pipeline/citlali_config_read_sync_offsets.h",
        "include/citlali/core/pipeline/interface_sync_config_adapter.h",
        "include/citlali/core/pipeline/raw_timestream_execution_plan.h",
        "include/citlali/core/pipeline/raw_timestream_provenance.h",
        "include/citlali/core/pipeline/rawobs_adc_snap.h",
        "include/citlali/core/pipeline/rawobs_tone_frequency_inventory.h",
        "include/citlali/core/pipeline/coherent_iq_mode_sidecar.h",
        "include/citlali/core/pipeline/tod_data_static_metadata_vars.h",
        "include/citlali/core/pipeline/tod_data_optional_vars.h",
        "include/citlali/core/engine/detail/citlali_config_impl.h",
        "include/citlali/core/utils/utils.h",
        "tools/config/config_leaf_contract_resolved.json",
        "tools/diagnostics/generate_sci_align_001_phase0.py",
        "tools/diagnostics/sci_align_001_phase0_gate_snapshot.json",
        "tools/diagnostics/sci_align_001_phase0_report.md",
        "validation/accepted_runs.json", "validation/product_contracts.json",
        "validation/validation_profiles.json",
        "validation/intended_science_changes.json",
    ]
    source_manifest = [
        {"path": str(repo / relative), "sha256": sha256_file(repo / relative)}
        for relative in source_paths
    ]
    write_json(output / "source_manifest.json", source_manifest)

    coordination_sources = [
        (
            FROZEN_COORDINATION_HEAD,
            "doc/audits/packages/SCI-ALIGN-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-08-01.md",
        ),
        (
            OWNER_DECISION_COMMIT,
            "doc/audits/packages/SCI-ALIGN-001_COORDINATOR_DECISION_2026-08-01.md",
        ),
        (
            FROZEN_COORDINATION_HEAD,
            "doc/audits/packages/SCI-ALIGN-001_SCIENTIFIC_CONTRACT_AUDIT.tex",
        ),
        (
            FROZEN_COORDINATION_HEAD,
            "doc/audits/packages/SCI-ALIGN-001_INDEPENDENT_CORE.tex",
        ),
        (
            FROZEN_COORDINATION_HEAD,
            "doc/audits/audit-ledger.yaml",
        ),
    ]
    coordination_manifest = []
    for commit, relative in coordination_sources:
        contents = git_blob(args.coordination_repo.resolve(), commit, relative)
        coordination_manifest.append({
            "repository": str(args.coordination_repo.resolve()),
            "commit": commit,
            "path": relative,
            "size_bytes": len(contents),
            "line_count": contents.count(b"\n"),
            "sha256": hashlib.sha256(contents).hexdigest(),
        })
    write_json(output / "coordination_source_manifest.json", coordination_manifest)

    correction_parent = git_text(
        args.coordination_repo.resolve(), "show", "-s", "--format=%P", CORRECTION_COMMIT,
    ).strip()
    correction_paths = git_text(
        args.coordination_repo.resolve(), "diff-tree", "--no-commit-id", "--name-only",
        "-r", CORRECTION_COMMIT,
    ).splitlines()
    expected_correction_paths = {
        "doc/audits/audit-ledger.yaml",
        "doc/audits/packages/SCI-ALIGN-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-08-01.md",
        "doc/audits/prompts/SCI_AST_001_AUDIT_PROMPT.md",
    }
    owner_decision_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", OWNER_DECISION_COMMIT,
         CORRECTION_COMMIT], cwd=args.coordination_repo.resolve(),
    ).returncode == 0
    handoff_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", HANDOFF_RECORD_COMMIT,
         FROZEN_COORDINATION_HEAD], cwd=args.coordination_repo.resolve(),
    ).returncode == 0
    if (
        set(correction_paths) != expected_correction_paths
        or not owner_decision_is_ancestor or not handoff_is_ancestor
    ):
        raise RuntimeError("coordination correction identity/ancestry changed")
    coordination_correction = {
        "commit": CORRECTION_COMMIT,
        "parent": correction_parent,
        "subject": git_text(
            args.coordination_repo.resolve(), "show", "-s", "--format=%s", CORRECTION_COMMIT,
        ).strip(),
        "changed_paths": correction_paths,
        "correct_owner_decision_commit": OWNER_DECISION_COMMIT,
        "rejected_transcription": "4f905f4f39461c8f9a86b0bf589880362d0a49f7",
        "owner_decision_is_ancestor_of_correction": owner_decision_is_ancestor,
        "handoff_record_is_ancestor_of_frozen_dispatch": handoff_is_ancestor,
        "scientific_policy_change_status": (
            "none; authoritative coordinator correction states the scientific contract is unchanged"
        ),
        "scientific_policy_change_value_authority": (
            "explicit coordinator correction supplied in this task"
        ),
        "interpretation": (
            "owner-decision identity correction plus phase-zero coordination-state update; "
            "no SCI-ALIGN scientific-policy change; frozen dispatch remains historical evidence"
        ),
    }
    write_json(output / "coordination_correction.json", coordination_correction)

    timing_inventory = {
        "detector": {
            "interfaces_declared": [f"toltec{index}" for index in range(13)],
            "interfaces_present_point_152389": [
                f"toltec{index}" for index in sorted(point_present_roaches)
            ],
            "interfaces_absent_point_152389": [
                f"toltec{index}" for index in sorted(
                    set(range(13)) - point_present_roaches
                )
            ],
            "selected_corpus_file_count": len(timing_rows),
            "boundary_fields_per_selected_interface": 47,
            "complete_header_registry": {
                "field_count": 37,
                "per_file_rows": detector_header_rows,
                "sorted_name_registry_sha256": (
                    "4082642c7571af87cbcefbcfbe52cb64e3204e45d9d5ca78323f5ef010172c47"
                ),
                "scope": "every Header.* field exposed in each of 110 selected detector files",
                "operator": "exact_only; header snapshots do not inherit sample cadence",
            },
            "fields": [
                {"raw_name": "inputs[].data_items[].meta.interface", "identity": "requested interface", "policy": "selected configs reconciled by phase-0 evidence; runtime enforcement/persistence absent"},
                {"raw_name": "inputs[].data_items[].filepath", "identity": "source file", "policy": "required path"},
                {"raw_name": "Header.Toltec.RoachIndex", "identity": "realized interface selector", "storage": "int32", "policy": "range/duplicate/config consistency unproved"},
                {"raw_name": "Header.Toltec.Master", "identity": "declared clock/control master enum", "policy": "exact header; not consumed or validated by alignment"},
                {"raw_name": "Header.Toltec.CompileTime", "unit": "UNPROVED", "epoch": "UNPROVED", "policy": "exact header; not alignment-consumed"},
                {"raw_name": "Header.Toltec.ObsStartTime", "unit": "UNPROVED", "epoch": "UNPROVED", "policy": "exact header"},
                {"raw_name": "Header.Toltec.ObsEndTime", "unit": "UNPROVED", "epoch": "UNPROVED", "policy": "exact header"},
                {"raw_name": "Header.Toltec.ObsNum", "identity": "observation", "policy": "exact header"},
                {"raw_name": "Header.Toltec.SubObsNum", "identity": "sub-observation", "policy": "exact header"},
                {"raw_name": "Header.Toltec.ScanNum", "identity": "acquisition scan", "policy": "exact header"},
                {"raw_name": "Header.Toltec.FpgaFreq", "unit": "Hz", "policy": "exact header"},
                {"raw_name": "Header.Toltec.AccumLen", "unit": "UNPROVED", "policy": "exact header"},
                {"raw_name": "Header.Toltec.SampleFreq", "unit": "Hz", "policy": "exact header"},
                {"raw_name": "Data.Toltec.SampleType", "identity": "sample-mode enum", "policy": "not filtered by alignment"},
                {"raw_name": "Data.Toltec.Ts[:,0]", "identity": "ClockTime", "unit": "sec (long_name only)", "epoch": "UNPROVED", "storage": "int32"},
                {"raw_name": "Data.Toltec.Ts[:,1]", "identity": "PpsCount", "unit": "pps ticks", "width_rollover": "UNPROVED"},
                {"raw_name": "Data.Toltec.Ts[:,2]", "identity": "ClockCount", "unit": "clock ticks", "storage": "int32", "width_rollover": "code assumes 32-bit but uses 2^32-1; producer authority absent"},
                {"raw_name": "Data.Toltec.Ts[:,3]", "identity": "PacketCount", "unit": "packet ticks", "storage": "int32", "width_rollover": "UNPROVED; current diff>1 misses rollover"},
                {"raw_name": "Data.Toltec.Ts[:,4]", "identity": "PpsTime", "unit": "clock ticks", "storage": "int32", "width_rollover": "code assumes 32-bit but authority absent"},
                {"raw_name": "Data.Toltec.Ts[:,5]", "identity": "ClockTimeNanoSec", "unit": "nsec", "epoch": "paired with Ts[:,0], semantics UNPROVED"},
                {"raw_name": "Data.Toltec.RecvTime", "unit": "sec", "epoch": "UNPROVED", "policy": "not used for detector sample coordinate"},
            ],
            "cadence": comparison_summary["convention"],
            "measurements": timing_summaries,
            "missing_nonfinite_policy": "integer timing fields have no fill/validity attrs; ordering/duplicates/nonfinite are not comprehensively rejected",
        },
        "telescope": telescope_comparison,
        "hwpr": {
            "path_count": len(hwpr_paths),
            "unique_content_sha256_count": len({row["sha256"] for row in raw_manifest if row["kind"] == "hwpr"}),
            "all_hwp_installed_zero": all(row["hwp_installed"] == 0 for row in raw_manifest if row["kind"] == "hwpr"),
            "application_required_fields": ["Data.Hwp.", "Data.Hwp.Ts", "Data.Hwp.Uts"],
            "application_required_fields_present_count": 0,
            "raw_timing_fields": [
                "Data.Toltec.HwprEncTs", "Data.Toltec.HwprEncUts",
                "Data.Toltec.HwprZeroptTs", "Data.Toltec.HwprZeroptUts",
                "Data.Toltec.HwprPpsTs", "Data.Toltec.HwprPpsUts",
                "Data.Toltec.HwprSensorUts", "Data.Toltec.Ts", "Data.Toltec.RecvTime",
            ],
            "authority_status": "UNAVAILABLE/conflicting: raw unit/epoch/width/rollover/angle transformation absent",
        },
    }
    write_json(output / "timing_field_inventory.json", timing_inventory)

    registry_counts = Counter(row["availability"] for row in registry)
    stop_facts = {
        "phase0_verdict": "STOP_FOR_OWNER_AUTHORITY",
        "application_path_delta": gate_evidence[
            "live_reflog_and_scope_assertions"
        ]["application_path_delta"],
        "application_source_edited": bool(gate_evidence[
            "live_reflog_and_scope_assertions"
        ]["application_path_delta"]),
        "field_registry_rows": len(registry),
        "detector_timing_inventory_rows": len(detector_inventory),
        "detector_boundary_fields_per_interface": 47,
        "detector_header_fields_per_interface": 37,
        "detector_header_inventory_rows": detector_header_rows,
        "detector_header_registry_sha256": (
            "4082642c7571af87cbcefbcfbe52cb64e3204e45d9d5ca78323f5ef010172c47"
        ),
        "telescope_hwpr_timing_field_inventory_rows": len(
            telescope_hwpr_timing_inventory
        ),
        "application_input_identity_rows": len(application_inputs),
        "application_input_identity_conflicts": sum(
            row["reconciliation_status"].startswith("CONFLICT")
            for row in application_inputs
        ),
        "application_input_identity_content_ambiguous": sum(
            row["reconciliation_status"] == "INTERFACE_ONLY_CONTENT_AMBIGUOUS"
            for row in application_inputs
        ),
        "application_input_identity_unavailable_local": sum(
            row["reconciliation_status"] == "UNAVAILABLE_LOCAL"
            for row in application_inputs
        ),
        "registry_availability_counts": dict(sorted(registry_counts.items())),
        "telescope_path_count": len(telescope_paths),
        "telescope_unique_content_count": len({row["sha256"] for row in raw_manifest if row["kind"] == "telescope"}),
        "hwpr_path_count": len(hwpr_paths),
        "hwpr_unique_content_count": len({row["sha256"] for row in raw_manifest if row["kind"] == "hwpr"}),
        "unresolved_authority_facts": comparison_summary["stop_reasons"] + [
            "TelUtc metadata unit conflicts with code/value interpretation",
            "Hold metadata conflicts with observed bitmask values and transition semantics",
            "all field validity rules and nonzero interpolation spans await producer/owner authority",
            "RA/Dec alias precedence is unproved",
            "current telescope output mislabels timestamps/state as radians",
            "accepted centroid/PSF evidence is historical only; no successor run exists",
            "detector anchor epoch conversion and its subtract-0.5/truncation rule are unproved",
            "telescope clock conversion/offset requested, effective, and realized state is absent",
            "selected local configs reconcile requested interfaces to raw RoachIndex, but runtime does not enforce or persist that relation; malformed/duplicate stream failure policy is incomplete",
            "accepted Beammap AcuTime is nonmonotonic and PpsCount/PpsTime are duplicate-bearing; field-specific operator authority is absent",
            "TelLst and TelUtc values/metadata have periodic-or-epoch identity conflicts that prohibit guessed interpolation",
            "raw HWPR event arrays are multidimensional packed records with unproved ordering and valid-count rules",
        ],
    }
    write_json(output / "stop_facts.json", stop_facts)

    owner_questions = {
        "phase0_verdict": "STOP_FOR_OWNER_AUTHORITY",
        "questions": [
            {
                "id": "Q01",
                "question": (
                    "What producer document or responsible owner freezes the epoch, logical widths, "
                    "and rollover/modulus semantics of every Data.Toltec.Ts column and PacketCount?"
                ),
                "blocks": ["detector timestamp construction", "rollover correction"],
            },
            {
                "id": "Q02",
                "question": (
                    "May phase 1 freeze detector cadence at 0.008192 s from "
                    "FpgaFreq/AccumLen/SampleFreq, and freeze phase to the latest first "
                    "realized detector timestamp after offsets?"
                ),
                "blocks": ["shared slot grid"],
            },
            {
                "id": "Q03",
                "question": (
                    "May phase 1 use strict abs(residual) < 0.004063 s as the measured slot "
                    "tolerance, or what authoritative tolerance and guard must replace it?"
                ),
                "blocks": ["slot acceptance tolerance"],
            },
            {
                "id": "Q04",
                "question": (
                    "Given ALIGN-OD1 requires detector acquisition support to be retained, should "
                    "phase 1 use union support with per-interface unavailability for the 38 "
                    "accepted-pair edge-only rows, or what exact support rule should implement OD1?"
                ),
                "blocks": ["acquisition bounds", "duration", "output support"],
            },
            {
                "id": "Q05",
                "question": (
                    "For LMT, each toltec0..toltec12, and HWPR interface, who owns requested, "
                    "effective, observation-resolved, and realized offsets, including whether an "
                    "LMT offset is required and the absent-interface policy?"
                ),
                "blocks": ["offset provenance", "exactly-once application"],
            },
            {
                "id": "Q06",
                "question": (
                    "What is the authoritative HWPR input schema and angle transform, including "
                    "units, epoch, counter widths, rollover, cadence, support, and offset stage?"
                ),
                "blocks": ["HWPR alignment", "HWPR registry"],
            },
            {
                "id": "Q07",
                "question": (
                    "Is TelescopeBackend.Hold a bitmask state with left-continuous half-open "
                    "semantics, and what do observed values 0,2,8,10,64,66,72,74 mean?"
                ),
                "blocks": ["Hold topology", "state interpolation"],
            },
            {
                "id": "Q08",
                "question": (
                    "What are TelTime, TelUtc, TelLst, and PpsTime scientific identities and epochs, and "
                    "which are exact-only versus half-open step coordinates?"
                ),
                "blocks": ["telescope time registry", "TelUtc unit conflict"],
            },
            {
                "id": "Q09",
                "question": (
                    "Which telescope angles are circular, in which frames and periods, and may "
                    "shortest-arc interpolation be used for each?"
                ),
                "blocks": ["circular topology", "field operator"],
            },
            {
                "id": "Q10",
                "question": (
                    "What validity rules, missing/nonfinite policy, and maximum interpolation "
                    "support span apply to every telescope and HWPR field?"
                ),
                "blocks": ["typed field registry activation"],
            },
            {
                "id": "Q11",
                "question": (
                    "What is the authoritative precedence and scientific identity for configured "
                    "TelRaAct/TelDecAct versus observed SourceRaAct/SourceDecAct aliases?"
                ),
                "blocks": ["RA/Dec alias mapping"],
            },
            {
                "id": "Q12",
                "question": (
                    "What exact output units, identities, and vector-shape rules implement ALIGN-OD3 "
                    "for telescope timestamps, state fields, and vector-valued headers?"
                ),
                "blocks": ["output identity"],
            },
            {
                "id": "Q13",
                "question": (
                    "Which owner-approved local fixture or human-mediated Unity run will establish "
                    "direct timing, source-crossing, centroid, and PSF-width non-degradation?"
                ),
                "blocks": ["phase-1 compatibility gate"],
            },
            {
                "id": "Q14",
                "question": (
                    "Does the owner approve the fail-closed registry baseline (exact-only, zero "
                    "support span) until each field-specific authority is supplied?"
                ),
                "blocks": ["registry default policy"],
            },
            {
                "id": "Q15",
                "question": (
                    "What preregistered timing, source-crossing, centroid, and per-array major/minor "
                    "PSF-width tolerances—derived from existing repeatability or fit uncertainty—"
                    "must the successor satisfy before its candidate results are viewed?"
                ),
                "blocks": ["non-degradation thresholds", "phase-1 validation design"],
            },
            {
                "id": "Q16",
                "question": (
                    "Is the current detector anchor int(Ts[0,0] + Ts[0,5]*1e-9 - 0.5) "
                    "scientifically authoritative, or what checked epoch conversion and rounding "
                    "rule must replace it?"
                ),
                "blocks": ["detector epoch conversion", "timestamp anchor"],
            },
            {
                "id": "Q17",
                "question": (
                    "What fail-closed policy applies to interface/Roach mismatch, invalid or "
                    "duplicate Roach IDs, malformed Ts shape, nonpositive/nonfinite clock headers, "
                    "duplicate or rollover packet counters, and nonmonotonic reconstructed time?"
                ),
                "blocks": ["boundary validation", "stream identity"],
            },
        ],
    }
    write_json(output / "owner_questions.json", owner_questions)

    accepted_changed_rows = sum(
        row["obsnum"] in {"152389", "148670"} for row in changed_rows
    )
    measured_mapping = comparison_summary["measured"]
    accepted_mapping = comparison_summary["accepted_pair"]
    if not (
        len(changed_rows) == measured_mapping["all_native_comparison_changed_rows"] == 181
        and accepted_changed_rows == accepted_mapping["all_native_comparison_changed_rows"] == 38
        and all(
            row["comparison_domain"] == "edge_outside_current_and_proposed_support"
            and row["current_valid"] is False
            and row["proposed_valid_current_tolerance"] is False
            and row["proposed_valid_strict_candidate_tolerance"] is False
            for row in changed_rows
        )
        and measured_mapping["current_mapped_rows"]
        == measured_mapping["proposed_mapped_rows_current_tolerance"]
        == measured_mapping["proposed_mapped_rows_strict_candidate_tolerance"]
        == 9001641
        and measured_mapping["current_invalid_proposed_valid_current_tolerance"] == 0
        and measured_mapping["current_valid_proposed_invalid_current_tolerance"] == 0
        and measured_mapping[
            "current_invalid_proposed_valid_strict_candidate_tolerance"
        ] == 0
        and measured_mapping[
            "current_valid_proposed_invalid_strict_candidate_tolerance"
        ] == 0
        and measured_mapping["exact_half_ties"] == 0
        and measured_mapping["near_half_ties_atol_1e_12"] == 0
    ):
        raise RuntimeError("mapping/support evidence changed from the phase-0 report")
    if not (
        len(offsets) == 45
        and len(application_inputs) == 38
        and all(
            row["reconciliation_status"] == "PROVED_LOCAL_REQUEST_TO_RAW_IDENTITY"
            for row in application_inputs
        )
        and len(detector_inventory) == 5170
        and detector_header_rows == 4070
        and len(telescope_hwpr_timing_inventory) == 2029
        and len(registry) == 383
    ):
        raise RuntimeError("boundary inventory/report invariants changed")
    require_report_claims(report_text, {
        "verdict": "Verdict: **STOP_FOR_OWNER_AUTHORITY**",
        "repair base": REPAIR_BASE,
        "frozen coordination": FROZEN_COORDINATION_HEAD,
        "owner decision": OWNER_DECISION_COMMIT,
        "correction": CORRECTION_COMMIT,
        "gate snapshot": GATE_SNAPSHOT_SHA256,
        "telescope HWPR timing count": "has 2,029 rows",
        "detector inventory count": "has 5,170 rows: 47 exact boundary facts",
        "detector header rows": "(4,070 per-file header rows)",
        "detector header registry": (
            "4082642c7571af87cbcefbcfbe52cb64e3204e45d9d5ca78323f5ef010172c47"
        ),
        "application input count": "three selected configs contribute 38 application-input rows",
        "registry count": "registry has 383 stable rows",
        "accepted mapping row": (
            "| Accepted pair | 4,305,394 | 4,305,356 | 38 | 0 | 0 | 4.061937 ms |"
        ),
        "extended mapping row": (
            "| Extended local corpus | 9,001,822 | 9,001,641 | 181 | 0 | 0 | 4.061937 ms |"
        ),
        "changed row classification": (
            "contains 181 extended-corpus rows, including 38 accepted-pair"
        ),
        "offset count": "`offset_state_trace.csv` has 45 rows",
        "configured beam telescope": (
            "e39f5b9e3066fd20086105964dd915ff67709142d699e8a18bb58cfd9da6b7ae"
        ),
        "hold sensitivity": "changes 3,329/383,699 rows (0.867607%)",
        "pointing product": (
            "344c85500d367566b7a1b9463fc46a8cd4d8aef0671f9f7eb3891accfbb53763"
        ),
        "beammap product": (
            "f1dcd7e7ea88eb47d1b494cdfac3d3b365d5a938d87b5393c97b5fcde9b5b25c"
        ),
        "source crossing": (
            "948df213ac88ce516f85cf177ed33495f95ca5d26f85a5afedb5b68f548255ed"
        ),
        "owner TelLst question": "`TelTime`, `TelUtc`, `TelLst`, and `PpsTime`",
    })

    identity = {
        "repair_repository": str(repo),
        "repair_base": REPAIR_BASE,
        "branch": "codex/repair-sci-align-001",
        "frozen_coordination_head": FROZEN_COORDINATION_HEAD,
        "handoff_record_commit": HANDOFF_RECORD_COMMIT,
        "owner_decision_commit_corrected": OWNER_DECISION_COMMIT,
        "canonical_correction_commit": CORRECTION_COMMIT,
        "explicitly_rejected_transcription": "4f905f4f39461c8f9a86b0bf589880362d0a49f7",
        "scientific_policy_change_status": (
            "none per authoritative coordinator correction in this task"
        ),
        "gate_snapshot_sha256": GATE_SNAPSHOT_SHA256,
        "handoff_sha256_expected": "2231e09c4310e8ddf73b6e25cd52c3c10671234667607b88d3723571dfa7a5f8",
        "isolation": "dedicated supplied worktree; no merge/rebase/push/Unity; sibling repositories read-only",
    }
    write_json(output / "identity_and_isolation.json", identity)

    write_json(output / "git_isolation_evidence.json", gate_evidence)

    (output / "REPORT.md").write_text(report_text)
    artifact_names = [
        "application_input_identity_inventory.csv",
        "boundary_stream_inventory.csv",
        "changed_rows.csv",
        "comparison_summary.json",
        "coordination_correction.json",
        "coordination_source_manifest.json",
        "detector_timing_inventory.csv",
        "field_registry.csv",
        "git_isolation_evidence.json",
        "identity_and_isolation.json",
        "offset_state_trace.csv",
        "owner_questions.json",
        "raw_input_manifest.csv",
        "science_compatibility_evidence.json",
        "selected_detector_input_manifest.csv",
        "slot_mapping_comparison.csv",
        "source_manifest.json",
        "stop_facts.json",
        "telescope_compatibility.json",
        "telescope_hwpr_timing_field_inventory.csv",
        "timing_field_inventory.json",
        "REPORT.md",
    ]
    expected_files = set(artifact_names) | {"SHA256SUMS"}
    observed_files = {path.name for path in output.iterdir() if path.is_file()}
    unexpected_files = sorted(observed_files - expected_files)
    missing_files = sorted(set(artifact_names) - observed_files)
    if unexpected_files or missing_files:
        raise RuntimeError(
            f"noncanonical output contents: unexpected={unexpected_files}, missing={missing_files}"
        )
    digests = []
    for name in sorted(artifact_names):
        digests.append(f"{sha256_file(output / name)}  {name}")
    (output / "SHA256SUMS").write_text("\n".join(digests) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
