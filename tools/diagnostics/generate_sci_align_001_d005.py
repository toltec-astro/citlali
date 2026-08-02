#!/usr/bin/env python3
"""Generate the bounded SCI-ALIGN-001 ALIGN-P0-D005 preregistration package.

This evidence-only diagnostic reads frozen repository sources, the read-only
coordination checkout, and owner-local validation data.  It writes only the
requested deterministic D005 package.  It never executes Citlali, TolProj, or
Unity and never modifies application code, raw data, projects, reductions,
logs, APTs, or coordination records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset


PHASE0_COMMIT = "53c7154a3633dfe19dc036cfb5a6250f729a897d"
GOVERNING_APPLICATION_SHA = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
BRANCH = "codex/repair-sci-align-001"
COORDINATION_HEAD = "6785152c2a2d4113c9ba89073de00cb454aa70c4"
CORRECTED_OWNER_DECISION_COMMIT = "4f905f4f353e91847a303f4f3959654f3f03c302"
OWNER_DECISION_CORRECTION_COMMIT = "35cc8ce246e8e70c569e650be6c1eae2c91b80ef"
REJECTED_OWNER_DECISION_TYPO = "4f905f4f39461c8f9a86b0bf589880362d0a49f7"
SUITE_PRODUCT_SOURCE_COMMIT = "cfae989ce8eb828842cf775aa287b4f43d1e2385"
RELEASED_4X_COMMIT = "a398581f48200dcd0cf41e1e09d33b5b7922a06f"
F_REF_HZ = 122.0703125
UINT32_MODULUS = 2**32
PACKAGE_NAME = "sci_align_001_align_p0_d005_2026-08-01"
PHASE0_PACKAGE = "sci_align_001_phase0_2026-08-01"
PHASE0_SUMS_SHA256 = (
    "074aff9deddd062d13a055589714f5d1b52ee18753052286119a184d2dbc08a2"
)
PHASE0_REPORT_SHA256 = (
    "4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8"
)
PHASE0_FIELD_REGISTRY_SHA256 = (
    "5ac211f7f21e8a7547ceb4a3db8c37491711e06a5359e9b53aec01ed3115d6f3"
)
TELESCOPE_BRACKET_LIMIT_SEC = 0.021130561828613281

DECISION_SPECS = (
    (
        "D001",
        "86434df2cfb5b85d0ccd306150cb428321abdbb9",
        "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D001_DECISION_2026-08-01.md",
        "0efe9d06bf02ceca473c92b00dce4c6d1ec9b6e564f226c9451f444ee5a6d66c",
    ),
    (
        "D002",
        "10981b29c1870e745b7f3c9cabed3c634a46427f",
        "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D002_DECISION_2026-08-01.md",
        "7e4e2c02bf2e16035d6a2aceacaa4e07c7c528e08f09fdb2d0a9186d510465cd",
    ),
    (
        "D003",
        "d500e33da1869bc1e20383a49484daddca9e7ea7",
        "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D003_DECISION_2026-08-01.md",
        "a25b46ddec445a4164086aae248e3cbbfc70c7df8cb2c785c4c135d936e92d67",
    ),
    (
        "D004",
        "a3775bf3039461a6435f07938572dd23b3f03d47",
        "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_D004_DECISION_2026-08-01.md",
        "ea03c5b614c7ce64ab5ab071c48c07d2f4910b919941798e70668049e53faf78",
    ),
)

COORDINATION_AUTHORITIES = (
    (
        "current_ledger",
        "doc/audits/audit-ledger.yaml",
        "516f075d5739d50e12e774a9d66aae4140b104a0c03b29351d57325d9da304c9",
    ),
    (
        "current_coordination_status",
        "doc/REFACTOR_STATUS.md",
        "53a13e16526f53ae8414e2f44221ed50225a5f1284d7da161906ce2add36939b",
    ),
    (
        "current_coordinator_review",
        "doc/audits/packages/SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md",
        "e87f619452f4839241e78ced4b2c6b70efba7735c4f4ff50255bc79a716bd17b",
    ),
    (
        "owner_decision_contract",
        "doc/audits/packages/SCI-ALIGN-001_COORDINATOR_DECISION_2026-08-01.md",
        "14c8044fe7a26c7ee1af7b8dea3472e9cb604471c91ca58dc93b662a7d3d9895",
    ),
    (
        "owner_decision_brief",
        "doc/audits/packages/SCI-ALIGN-001_COORDINATOR_DECISION_BRIEF_2026-08-01.md",
        "19e2d001ca34a5b4e404a481b6a50bc315799719e295a9a96b4cf4697de8f0e6",
    ),
    (
        "bounded_repair_handoff",
        "doc/audits/packages/SCI-ALIGN-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-08-01.md",
        "71f76026cc95ae795f4e2a0cfabc7192d416ed3f8affc45367a0b4dae9807d9b",
    ),
    (
        "scientific_contract_audit",
        "doc/audits/packages/SCI-ALIGN-001_SCIENTIFIC_CONTRACT_AUDIT.tex",
        "6aaed0e6e16e4c37cd24d15b98346f84024ffd7920bd0524e7a170dbc728a393",
    ),
    (
        "independent_core",
        "doc/audits/packages/SCI-ALIGN-001_INDEPENDENT_CORE.tex",
        "4ee7b7e9cbe883ea626afe2e3d22756b20f556a2e06115d4a2832f2e78469785",
    ),
)

CONFIG_SPECS = (
    {
        "id": "point_core",
        "mode": "point",
        "role": "mandatory_core",
        "relative": "point/pointings_v22/reduced/redu00/citlali_o152389_0_2_c1.yaml",
    },
    {
        "id": "oof_context",
        "mode": "oof",
        "role": "lifecycle_timing_context",
        "relative": "oof/wilson/1146+399/reduced/redu00/citlali_o152385_0_1_c3.yaml",
    },
    {
        "id": "beammap_core",
        "mode": "beammap",
        "role": "mandatory_core",
        "relative": "beammaps/3c273/reduced/redu00/citlali_o148670_0_2_c1.yaml",
    },
    {
        "id": "beammap_support",
        "mode": "point",
        "role": "beammap_bracketing_pointing_context",
        "relative": "beammaps/pointings/reduced/redu00/citlali_o148669_0_2_c2.yaml",
    },
    {
        "id": "science_support",
        "mode": "point",
        "role": "science_pointing_context",
        "relative": "science/pointings/reduced/redu00/citlali_o152389_0_2_c3.yaml",
    },
    {
        "id": "science_core",
        "mode": "science",
        "role": "long_mode_sentinel",
        "relative": "science/reduction/NGC4449/reduced/redu03/citlali_o152390_0_2_c2.yaml",
    },
)

EXPECTED_RATE_PROFILES = {
    0.5: {"sample_rate_hz": 61.03515625, "accum_len": 4194304, "dt_sec": 0.016384},
    1.0: {"sample_rate_hz": 122.0703125, "accum_len": 2097152, "dt_sec": 0.008192},
    2.0: {"sample_rate_hz": 244.140625, "accum_len": 1048576, "dt_sec": 0.004096},
    4.0: {"sample_rate_hz": 488.28125, "accum_len": 524288, "dt_sec": 0.002048},
}

UNREALIZED_SCIENCE_SUPPORT = (152418, 152420, 152430, 152432, 152434)
EXPECTED_INTERFACES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12)
EXPECTED_EXTERNAL_MANIFESTS = {
    152418: "eaa1e53d9f1320a59254bbd784473e4be45f6454c82dcb6cd953026abb5b8aa1",
    152420: "f032b6e9096cc5780eac53f29a6717b4fa4d62a2693fbfa0f20c5d518621c688",
    152430: "0aec4f04c696bcc19b41d311b69d45563f61dc2bdad0c0121799f836bd56edf6",
    152432: "aa67137d4d1d689fc0cd2f78d0da903637c11e0b1273ab29317e8daa9d5f3e57",
    152434: "3064d0fefcfbdc143993df761305c8c41a71c3d6e0cc50ddef27e1e3ae392c6b",
}
EXPECTED_EXTERNAL_COMBINED_MANIFEST = (
    "1fb6bb026eb6a7e5e3c8398eb9fcd00470abf1810b37f1ee6873b8aac195f272"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE,
        text=text,
    ).stdout


def git_blob(repo: Path, commit: str, path: str) -> bytes:
    return git(repo, "show", f"{commit}:{path}", text=False)  # type: ignore[return-value]


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required read-only evidence file is absent: {path}")
    return path


def require_sha(path: Path, expected: str) -> str:
    measured = sha256_file(path)
    if measured != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {measured} != {expected}")
    return measured


def path_status_allowed(repo: Path) -> list[str]:
    lines = str(git(repo, "status", "--porcelain=v1", "--untracked-files=all")).splitlines()
    allowed = {
        "tools/diagnostics/generate_sci_align_001_d005.py",
    }
    package_prefix = f"validation/{PACKAGE_NAME}/"
    bad = []
    for line in lines:
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        if raw_path not in allowed and not raw_path.startswith(package_prefix):
            bad.append(line)
    if bad:
        raise RuntimeError(f"unrelated or application worktree change present: {bad}")
    return lines


def validate_identity(
    repo: Path, coordination_repo: Path, tolproj_repo: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch = str(git(repo, "symbolic-ref", "--short", "HEAD")).strip()
    head = str(git(repo, "rev-parse", "HEAD")).strip()
    if branch != BRANCH or head != PHASE0_COMMIT:
        raise RuntimeError(f"repair identity mismatch: branch={branch} head={head}")
    status = path_status_allowed(repo)
    if not is_ancestor(repo, GOVERNING_APPLICATION_SHA, PHASE0_COMMIT):
        raise RuntimeError("governing application SHA is not phase-zero commit parent/ancestor")
    parent = str(git(repo, "rev-parse", f"{PHASE0_COMMIT}^")) .strip()
    if parent != GOVERNING_APPLICATION_SHA:
        raise RuntimeError(f"phase-zero parent changed: {parent}")

    coord_head = str(git(coordination_repo, "rev-parse", "HEAD")).strip()
    coord_branch = str(git(coordination_repo, "symbolic-ref", "--short", "HEAD")).strip()
    coord_status = str(git(coordination_repo, "status", "--porcelain=v1"))
    if coord_head != COORDINATION_HEAD or coord_status != "":
        raise RuntimeError(
            f"coordination identity mismatch: head={coord_head} status={coord_status!r}"
        )
    for required_ancestor in (
        CORRECTED_OWNER_DECISION_COMMIT, OWNER_DECISION_CORRECTION_COMMIT,
    ):
        if not is_ancestor(coordination_repo, required_ancestor, COORDINATION_HEAD):
            raise RuntimeError(
                f"corrected owner-decision identity is not bound by coordination HEAD: "
                f"{required_ancestor}"
            )

    phase0 = repo / "validation" / PHASE0_PACKAGE
    sums = require_file(phase0 / "SHA256SUMS")
    require_sha(sums, PHASE0_SUMS_SHA256)
    for line in sums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        require_sha(phase0 / name, expected)
    require_sha(phase0 / "REPORT.md", PHASE0_REPORT_SHA256)
    require_sha(phase0 / "field_registry.csv", PHASE0_FIELD_REGISTRY_SHA256)

    authority_rows: list[dict[str, Any]] = []
    for decision_id, commit, relative, expected_sha in DECISION_SPECS:
        path = coordination_repo / relative
        require_sha(path, expected_sha)
        if not is_ancestor(coordination_repo, commit, COORDINATION_HEAD):
            raise RuntimeError(f"{decision_id} commit is not an ancestor of coordination HEAD")
        historical = git_blob(coordination_repo, commit, relative)
        if historical != path.read_bytes():
            raise RuntimeError(f"{decision_id} current bytes differ from content commit")
        authority_rows.append({
            "authority_id": decision_id,
            "authority_class": "approved_owner_decision",
            "path": str(path),
            "git_commit": commit,
            "sha256": expected_sha,
            "use": "binding SCI-ALIGN-001 phase-zero contract",
        })
    for authority_id, relative, expected_sha in COORDINATION_AUTHORITIES:
        path = coordination_repo / relative
        require_sha(path, expected_sha)
        authority_rows.append({
            "authority_id": authority_id,
            "authority_class": "current_coordination_authority",
            "path": str(path),
            "git_commit": COORDINATION_HEAD,
            "sha256": expected_sha,
            "use": "current SCI-ALIGN-001 ledger/status/dispatch state",
        })

    repository_authorities = (
        "AGENTS.md",
        "doc/ARCHITECTURE.md",
        "doc/SCIENTIFIC_CONVENTIONS.md",
        "doc/RETAINED_DEBT.md",
        "doc/REFACTOR_STATUS.md",
        "doc/PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md",
        "doc/TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md",
        "doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md",
        "doc/PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md",
        "validation/validation_profiles.json",
        "validation/product_contracts.json",
        "validation/profiles/science_refactor_snapshot_v1.json",
        "validation/profiles/science_scientific_equivalence_v2.json",
        "validation/profiles/beammap_scientific_equivalence_v1.json",
    )
    for relative in repository_authorities:
        path = require_file(repo / relative)
        authority_rows.append({
            "authority_id": f"repair_repo:{relative}",
            "authority_class": "repository_local_authority",
            "path": str(path),
            "git_commit": PHASE0_COMMIT,
            "sha256": sha256_file(path),
            "use": "architecture/science/status/integration/performance constraint",
        })

    tolproj_authorities = (
        "AGENTS.md", "README.md", "docs/STATUS.md", "docs/WORKFLOW_V0_2.md",
        "docs/CITLALI_REFACTOR_CONFIG.md", "docs/CITLALI_VALIDATION_SUITE.md",
    )
    tolproj_head = str(git(tolproj_repo, "rev-parse", "HEAD")).strip()
    for relative in tolproj_authorities:
        path = require_file(tolproj_repo / relative)
        authority_rows.append({
            "authority_id": f"tolproj:{relative}",
            "authority_class": "read_only_workflow_authority",
            "path": str(path),
            "git_commit": tolproj_head,
            "sha256": sha256_file(path),
            "use": "suite/project realization and nonmutating workflow interpretation",
        })

    source_paths = (
        "include/citlali/core/pipeline/timestream_alignment_helpers.h",
        "include/citlali/core/engine/detail/todproc_alignment_impl.h",
        "src/citlali/core/engine/telescope.cpp",
        "include/citlali/core/utils/utils.h",
        "include/citlali/core/pipeline/raw_tod_output_context.h",
    )
    source_blobs = []
    for relative in source_paths:
        blob = git_blob(repo, GOVERNING_APPLICATION_SHA, relative)
        product_blob = git_blob(repo, SUITE_PRODUCT_SOURCE_COMMIT, relative)
        if product_blob != blob:
            raise RuntimeError(
                f"suite product source differs from governing source for {relative}"
            )
        source_blobs.append({
            "identity": "governing_application_source",
            "path": relative,
            "git_commit": GOVERNING_APPLICATION_SHA,
            "sha256": sha256_bytes(blob),
            "use": "exact governing timestamp/alignment/interpolation/scan implementation",
        })
        source_blobs.append({
            "identity": "suite_product_source",
            "path": relative,
            "git_commit": SUITE_PRODUCT_SOURCE_COMMIT,
            "sha256": sha256_bytes(product_blob),
            "byte_identical_to_governing_application": True,
            "use": "historical Beammap 148670 suite source-path compatibility",
        })
    for relative in (
        "src/citlali/core/engine/telescope.cpp",
        "include/citlali/core/engine/todproc.h",
        "include/citlali/core/utils/utils.h",
    ):
        blob = git_blob(repo, RELEASED_4X_COMMIT, relative)
        source_blobs.append({
            "identity": "released_4x_source",
            "path": relative,
            "git_commit": RELEASED_4X_COMMIT,
            "sha256": sha256_bytes(blob),
            "use": "released-4.x whole-word-linear-then-nonzero compatibility authority",
        })

    identity = {
        "repair_repository": str(repo),
        "branch": branch,
        "phase_zero_evidence_commit": head,
        "governing_application_sha": parent,
        "repair_status_at_generation": status,
        "allowed_status_scope": [
            "tools/diagnostics/generate_sci_align_001_d005.py",
            f"validation/{PACKAGE_NAME}/",
        ],
        "application_path_delta": [],
        "coordination_repository": str(coordination_repo),
        "coordination_branch": coord_branch,
        "coordination_head": coord_head,
        "coordination_clean": coord_status == "",
        "owner_decision_identity_correction": {
            "authoritative_commit": CORRECTED_OWNER_DECISION_COMMIT,
            "canonical_correction_commit": OWNER_DECISION_CORRECTION_COMMIT,
            "rejected_transcription": REJECTED_OWNER_DECISION_TYPO,
            "scientific_policy_changed_by_correction": False,
        },
        "tolproj_repository": str(tolproj_repo),
        "tolproj_head_read_only": tolproj_head,
        "phase_zero_package": str(phase0),
        "phase_zero_sha256sums_sha256": PHASE0_SUMS_SHA256,
        "phase_zero_report_sha256": PHASE0_REPORT_SHA256,
        "phase_zero_field_registry_sha256": PHASE0_FIELD_REGISTRY_SHA256,
        "frozen_phase_zero_rewritten": False,
        "unity_contacted": False,
        "successor_output_inspected": False,
        "application_code_edited": False,
        "phase_one_authorized": False,
        "phase_zero_worktree_verified_clean_before_d005": True,
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    return identity | {"governing_source_blobs": source_blobs}, authority_rows


class DigestCache:
    def __init__(self) -> None:
        self.by_inode: dict[tuple[int, int, int], str] = {}

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size)
        if key not in self.by_inode:
            self.by_inode[key] = sha256_file(path)
        return self.by_inode[key]


def resolve_suite_path(requested: str, suite_root: Path) -> Path:
    prefix = "/work/toltec/citlali-validation/v1/"
    if not requested.startswith(prefix):
        raise RuntimeError(f"selected config references path outside frozen suite: {requested}")
    return require_file(suite_root / requested.removeprefix(prefix))


def obsnum_from_group(group: dict[str, Any]) -> int:
    name = str(group.get("meta", {}).get("name", ""))
    match = re.match(r"(\d+)_", name)
    if match:
        return int(match.group(1))
    for item in group.get("data_items", []):
        match = re.search(r"_(\d{6})_", Path(str(item.get("filepath", ""))).name)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"cannot resolve observation identity from input group: {name}")


def load_suite_inputs(
    suite_root: Path, digest_cache: DigestCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    input_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    observations: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in CONFIG_SPECS:
        config_path = require_file(suite_root / spec["relative"])
        payload = yaml.safe_load(config_path.read_text())
        config_sha = digest_cache.digest(config_path)
        offsets: dict[str, float] = {}
        for entry in payload.get("interface_sync_offset", []):
            if len(entry) != 1:
                raise RuntimeError(f"ambiguous interface_sync_offset entry in {config_path}")
            interface, value = next(iter(entry.items()))
            if interface in offsets or not isinstance(value, (int, float)):
                raise RuntimeError(f"invalid interface_sync_offset in {config_path}: {entry}")
            offsets[str(interface)] = float(value)
        expected_offset_interfaces = {f"toltec{index}" for index in range(13)} | {"hwpr"}
        if set(offsets) != expected_offset_interfaces:
            raise RuntimeError(f"incomplete interface_sync_offset set in {config_path}")
        config_rows.append({
            "config_id": spec["id"],
            "mode": spec["mode"],
            "role": spec["role"],
            "path": str(config_path),
            "size_bytes": config_path.stat().st_size,
            "sha256": config_sha,
            "evidence_class": "realized_historical_low_level_configuration",
            "requested_interface_offsets_sec_json": canonical_json(offsets),
            "requested_nonzero_offset_count": sum(value != 0.0 for value in offsets.values()),
        })
        for group in payload.get("inputs", []):
            obsnum = obsnum_from_group(group)
            key = (spec["id"], obsnum)
            if key in observations:
                raise RuntimeError(f"duplicate selected observation group: {key}")
            record = {
                "config_id": spec["id"], "mode": spec["mode"],
                "role": spec["role"], "obsnum": obsnum,
                "config_path": config_path, "config_sha256": config_sha,
                "requested_offsets_sec": offsets,
                "detectors": [], "telescope": None, "hwpr": None,
            }
            for collection, item_class in (
                (group.get("cal_items", []), "calibration_input"),
                (group.get("data_items", []), "application_stream_input"),
            ):
                for item in collection:
                    if "filepath" not in item:
                        # Inline photometry/astrometry calibration values are
                        # already digest-bound by the complete config file.
                        continue
                    interface = str(item.get("meta", {}).get("interface", "UNDECLARED"))
                    requested = str(item.get("filepath", ""))
                    local_path = resolve_suite_path(requested, suite_root)
                    sha = digest_cache.digest(local_path)
                    row = {
                        "config_id": spec["id"], "mode": spec["mode"],
                        "fixture_role": spec["role"], "obsnum": obsnum,
                        "item_class": item_class, "interface": interface,
                        "requested_unity_path": requested,
                        "local_path": str(local_path),
                        "size_bytes": local_path.stat().st_size,
                        "sha256": sha,
                        "selection_authority": str(suite_root / "suite.yaml"),
                        "realization_authority": str(config_path),
                        "availability": "realized_local_read_only",
                    }
                    input_rows.append(row)
                    if item_class == "application_stream_input":
                        if interface.startswith("toltec"):
                            record["detectors"].append({
                                "supplied_interface": interface,
                                "path": local_path,
                            })
                        elif interface == "lmt":
                            record["telescope"] = local_path
                        elif interface == "hwpr":
                            record["hwpr"] = local_path
            if record["telescope"] is None or len(record["detectors"]) != 11:
                raise RuntimeError(f"incomplete selected detector/telescope set for {key}")
            supplied_interfaces = [
                item["supplied_interface"] for item in record["detectors"]
            ]
            if (
                len(set(supplied_interfaces)) != len(supplied_interfaces)
                or set(supplied_interfaces) != {
                    f"toltec{index}" for index in EXPECTED_INTERFACES
                }
            ):
                raise RuntimeError(f"invalid supplied detector-interface identity set for {key}")
            observations[key] = record
    return input_rows, config_rows, observations


def detector_record(path: Path) -> dict[str, Any]:
    with Dataset(path) as dataset:
        def scalar(name: str) -> Any:
            return np.asarray(dataset[name][:]).item()

        roach = int(scalar("Header.Toltec.RoachIndex"))
        fpga = float(scalar("Header.Toltec.FpgaFreq"))
        accum = int(scalar("Header.Toltec.AccumLen"))
        sample_rate = float(scalar("Header.Toltec.SampleFreq"))
        header_obsnum = int(scalar("Header.Toltec.ObsNum"))
        subobs = int(scalar("Header.Toltec.SubObsNum"))
        scan = int(scalar("Header.Toltec.ScanNum"))
        obs_start = int(scalar("Header.Toltec.ObsStartTime"))
        obs_end = int(scalar("Header.Toltec.ObsEndTime"))
        ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
    if ts.ndim != 2 or ts.shape[1] != 6:
        raise RuntimeError(f"unexpected Data.Toltec.Ts shape in {path}: {ts.shape}")
    anchor = int(float(ts[0, 0]) + float(ts[0, 5]) * 1.0e-9 - 0.5)
    signed_delta = ts[:, 2].astype(np.float64) - ts[:, 4].astype(np.float64)
    ticks = np.where(signed_delta < 0, signed_delta + UINT32_MODULUS - 1, signed_delta)
    times = anchor + ts[:, 1].astype(np.float64) + ticks / fpga
    logical_phase = (ts[:, 2] % UINT32_MODULUS - ts[:, 4] % UINT32_MODULUS) % UINT32_MODULUS
    logical_ticks = ts[:, 1] * int(fpga) + logical_phase
    tick_steps = np.diff(logical_ticks)
    nominal_ticks = int(round(fpga / sample_rate))
    factors = [
        factor for factor, profile in EXPECTED_RATE_PROFILES.items()
        if math.isclose(sample_rate, profile["sample_rate_hz"], rel_tol=0, abs_tol=1e-12)
    ]
    if len(factors) != 1:
        factor: float | str = "OUTSIDE_APPROVED_FAMILY"
    else:
        factor = factors[0]
        profile = EXPECTED_RATE_PROFILES[factor]
        if accum != profile["accum_len"] or not math.isclose(
            fpga / accum, sample_rate, rel_tol=0, abs_tol=1e-12,
        ):
            raise RuntimeError(f"conflicting native cadence headers in {path}")
    return {
        "roach": roach, "fpga": fpga, "accum": accum,
        "sample_rate": sample_rate, "factor": factor,
        "header_obsnum": header_obsnum, "subobs": subobs, "scan": scan,
        "obs_start": obs_start, "obs_end": obs_end, "times": times,
        "native_rows": int(times.size),
        "native_first_sec": float(times[0]), "native_last_sec": float(times[-1]),
        "native_duration_sec": float(times[-1] - times[0]),
        "packet_gap_count_current_test": int(np.sum(np.diff(ts[:, 3]) > 1)),
        "clock_tick_step_min_delta": int(np.min(tick_steps) - nominal_ticks),
        "clock_tick_step_max_delta": int(np.max(tick_steps) - nominal_ticks),
    }


def rate_inventory_row(
    stream: dict[str, Any], path: Path, raw_sha256: str, *, config_id: str,
    mode: str, fixture_role: str, availability: str,
    requested_offset_sec: float | str, offset_evidence: str,
    supplied_interface: str,
) -> dict[str, Any]:
    header_interface = f"toltec{stream['roach']}"
    return {
        "config_id": config_id, "mode": mode,
        "fixture_role": fixture_role, "obsnum": stream["header_obsnum"],
        "interface": header_interface,
        "supplied_interface": supplied_interface,
        "header_interface": header_interface,
        "interface_identity_status": (
            "exact_match" if supplied_interface == header_interface
            else "not_applicable_no_realized_config"
        ),
        "raw_path": str(path),
        "raw_sha256": raw_sha256,
        "header_obsnum": stream["header_obsnum"], "subobsnum": stream["subobs"],
        "scannum": stream["scan"], "fpga_freq_hz": stream["fpga"],
        "accum_len": stream["accum"], "sample_freq_hz": stream["sample_rate"],
        "native_rate_factor": stream["factor"],
        "cadence_sec": 1.0 / stream["sample_rate"],
        "exclusive_half_cell_sec": 0.5 / stream["sample_rate"],
        "native_rows": stream["native_rows"],
        "native_first_sec": stream["native_first_sec"],
        "native_last_sec": stream["native_last_sec"],
        "native_duration_sec": stream["native_duration_sec"],
        "requested_interface_offset_sec": requested_offset_sec,
        "effective_realized_offset_evidence": offset_evidence,
        "header_acquisition_start": stream["obs_start"],
        "header_acquisition_end": stream["obs_end"],
        "packet_gap_count_current_test": stream["packet_gap_count_current_test"],
        "clock_tick_step_min_delta": stream["clock_tick_step_min_delta"],
        "clock_tick_step_max_delta": stream["clock_tick_step_max_delta"],
        "rate_evidence_class": "genuine_native_header_evidence",
        "availability": availability,
    }


def external_asset_class(path: Path, obsnum: int) -> str:
    name = path.name
    if re.match(rf"toltec\d+_{obsnum}_000_0002_.*\.nc$", name):
        return "native_detector_science_stream"
    if name.endswith("_tune_processed.nc"):
        return "detector_tune_processed"
    if name.endswith("_tune.nc"):
        return "detector_tune_raw"
    if name.endswith("_tune.txt"):
        return "detector_tune_report"
    if name.startswith("tel_") and "_0002.nc" in name:
        return "native_telescope_science_stream"
    if name.startswith("tel_") and "_0001.nc" in name:
        return "telescope_tune_context"
    if name.startswith("hwpr_"):
        return "optional_hwpr_stream_inactive_under_D004"
    if name.startswith("toltec_hk_"):
        return "housekeeping_context"
    raise RuntimeError(f"unclassified owner-local support asset: {path}")


def external_science_support_evidence(
    owner_data_root: Path, cache: DigestCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    all_paths: list[Path] = []
    observation_summaries = []
    for obsnum in UNREALIZED_SCIENCE_SUPPORT:
        paths = sorted(owner_data_root.glob(f"*_{obsnum}_*"))
        if len(paths) != 48 or not all(path.is_file() for path in paths):
            raise RuntimeError(
                f"expected 48 direct owner-local assets for {obsnum}, found {len(paths)}"
            )
        all_paths.extend(paths)
        payload_parts = []
        science_streams = []
        for path in paths:
            relative = path.relative_to(owner_data_root).as_posix()
            digest = cache.digest(path)
            payload_parts.append(f"{digest}  {relative}\n")
            asset_class = external_asset_class(path, obsnum)
            manifest_rows.append({
                "obsnum": obsnum,
                "selection_authority": "TolProj v1 suite.yaml",
                "realized_suite_availability": "not_realized_in_v1_project_or_config",
                "owner_local_availability": "complete_read_only_candidate_input_set",
                "asset_class": asset_class,
                "relative_path": relative,
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "D005_use": (
                    "native_rate_header_inventory_only"
                    if asset_class == "native_detector_science_stream"
                    else "input_identity_only"
                ),
                "future_human_run_status": "not_requested_and_not_authorized_by_D005",
            })
            if asset_class == "native_detector_science_stream":
                stream = detector_record(path)
                if stream["header_obsnum"] != obsnum or stream["scan"] != 2:
                    raise RuntimeError(f"external support header mismatch: {path}")
                science_streams.append(stream)
                rate_rows.append(rate_inventory_row(
                    stream, path, digest,
                    config_id="external_science_support_unrealized",
                    mode="pointing_support_candidate",
                    fixture_role="suite_selected_owner_local_unrealized_support",
                    availability="owner_local_raw_available_not_realized_in_suite_config",
                    requested_offset_sec="N/A_NOT_REALIZED",
                    offset_evidence="no realized config; raw header rate evidence only",
                    supplied_interface="N/A_NO_REALIZED_CONFIG",
                ))
        if sorted(stream["roach"] for stream in science_streams) != list(EXPECTED_INTERFACES):
            raise RuntimeError(f"external support interface set changed for {obsnum}")
        manifest_digest = sha256_bytes("".join(payload_parts).encode())
        if manifest_digest != EXPECTED_EXTERNAL_MANIFESTS[obsnum]:
            raise RuntimeError(
                f"external support manifest identity changed for {obsnum}: {manifest_digest}"
            )
        observation_summaries.append({
            "obsnum": obsnum, "file_count": len(paths),
            "total_bytes": sum(path.stat().st_size for path in paths),
            "manifest_sha256": manifest_digest,
            "native_detector_file_count": len(science_streams),
            "native_detector_rows": sum(stream["native_rows"] for stream in science_streams),
            "native_rate_factors": sorted({stream["factor"] for stream in science_streams}),
        })
    combined_payload = "".join(
        f"{cache.digest(path)}  {path.relative_to(owner_data_root).as_posix()}\n"
        for path in sorted(all_paths)
    ).encode()
    combined_digest = sha256_bytes(combined_payload)
    if combined_digest != EXPECTED_EXTERNAL_COMBINED_MANIFEST:
        raise RuntimeError(
            f"combined external support manifest identity changed: {combined_digest}"
        )
    summary = {
        "owner_data_root": str(owner_data_root),
        "selection_authority": "TolProj v1 suite.yaml",
        "realization_authority": "v1 project/config files; these five observations are not realized there",
        "owner_local_availability": "complete raw/tune/telescope/HWPR/housekeeping sets",
        "file_count": len(all_paths),
        "total_bytes": sum(path.stat().st_size for path in all_paths),
        "combined_manifest_serialization": "lowercase_sha256 + two spaces + root-relative POSIX path + LF; globally path-sorted",
        "combined_manifest_sha256": combined_digest,
        "observations": observation_summaries,
        "use_boundary": (
            "native detector headers contribute rate-stratum inventory only; no config realization, "
            "reduction, product, or acceptance evidence is inferred"
        ),
    }
    return manifest_rows, rate_rows, summary


def cxx_round(values: np.ndarray) -> np.ndarray:
    return np.trunc(values + np.copysign(0.5, values)).astype(np.int64)


def build_timing_evidence(
    observations: dict[tuple[str, int], dict[str, Any]],
    input_sha: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    rate_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    grids: dict[tuple[str, int], dict[str, Any]] = {}
    for key, obs in sorted(observations.items()):
        stream_records = []
        for detector_input in sorted(
            obs["detectors"], key=lambda item: item["supplied_interface"],
        ):
            path = detector_input["path"]
            supplied_interface = detector_input["supplied_interface"]
            stream = detector_record(path)
            if stream["header_obsnum"] != obs["obsnum"]:
                raise RuntimeError(f"raw/header observation mismatch for {path}")
            stream["path"] = path
            interface = f"toltec{stream['roach']}"
            if supplied_interface != interface:
                raise RuntimeError(
                    f"supplied/raw detector-interface identity mismatch for {path}: "
                    f"{supplied_interface} != {interface}"
                )
            requested_offset = obs["requested_offsets_sec"][interface]
            stream["requested_offset_sec"] = requested_offset
            stream["alignment_times"] = stream["times"] + requested_offset
            stream_records.append(stream)
            rate_rows.append(rate_inventory_row(
                stream, path, input_sha[str(path)],
                config_id=obs["config_id"], mode=obs["mode"],
                fixture_role=obs["role"], availability="realized_local_read_only",
                requested_offset_sec=requested_offset,
                offset_evidence=(
                    "realized low-level config value bound; applied once before slotting in this "
                    "diagnostic; governing runtime does not independently persist realized application"
                ),
                supplied_interface=supplied_interface,
            ))
        if sorted(stream["roach"] for stream in stream_records) != list(EXPECTED_INTERFACES):
            raise RuntimeError(f"unexpected interface identity set for {key}")
        sample_rates = {stream["sample_rate"] for stream in stream_records}
        factors = {stream["factor"] for stream in stream_records}
        if len(sample_rates) != 1 or len(factors) != 1:
            raise RuntimeError(f"mixed-rate selected observation: {key}")
        sample_rate = next(iter(sample_rates))
        dt = 1.0 / sample_rate
        phase = max(float(stream["alignment_times"][0]) for stream in stream_records)
        overlap_end = min(float(stream["alignment_times"][-1]) for stream in stream_records)
        current_count = int((overlap_end - phase) / dt) + 1
        union_min: int | None = None
        union_max: int | None = None
        obs_exact_ties = 0
        obs_collisions = 0
        obs_ordinary_changes = 0
        obs_edge_rows = 0
        obs_current_rows = 0
        obs_native_rows = 0
        obs_max_residual = 0.0
        obs_min_margin = math.inf
        for stream in sorted(stream_records, key=lambda item: item["roach"]):
            times = stream["alignment_times"]
            q = (times - phase) / dt
            proposed = np.floor(q + 0.5).astype(np.int64)
            residual = times - (phase + proposed.astype(np.float64) * dt)
            admitted = np.abs(residual) < dt / 2.0
            if not admitted.all():
                raise RuntimeError(f"selected ordinary row violates D002 half-cell: {stream['path']}")
            current = cxx_round(q)
            current_residual = times - (
                phase + np.clip(current, 0, current_count - 1).astype(np.float64) * dt
            )
            current_valid = (
                (current >= 0) & (current < current_count)
                & (np.abs(current_residual) <= dt / 2.0)
            )
            ordinary_changes = int(np.sum(current_valid & (current != proposed)))
            edge = admitted & ~current_valid
            unique_slots, counts = np.unique(proposed, return_counts=True)
            collisions = int(np.sum(counts > 1))
            fractional = q - np.floor(q)
            exact_ties = int(np.sum(fractional == 0.5))
            abs_residual = np.abs(residual)
            margin = dt / 2.0 - abs_residual
            union_min = int(np.min(proposed)) if union_min is None else min(union_min, int(np.min(proposed)))
            union_max = int(np.max(proposed)) if union_max is None else max(union_max, int(np.max(proposed)))
            obs_exact_ties += exact_ties
            obs_collisions += collisions
            obs_ordinary_changes += ordinary_changes
            obs_edge_rows += int(edge.sum())
            obs_current_rows += int(current_valid.sum())
            obs_native_rows += int(times.size)
            obs_max_residual = max(obs_max_residual, float(np.max(abs_residual)))
            obs_min_margin = min(obs_min_margin, float(np.min(margin)))
            slot_rows.append({
                "config_id": obs["config_id"], "mode": obs["mode"],
                "obsnum": obs["obsnum"], "interface": f"toltec{stream['roach']}",
                "native_rows": int(times.size), "current_supported_rows": int(current_valid.sum()),
                "union_admitted_rows": int(admitted.sum()), "edge_only_rows": int(edge.sum()),
                "ordinary_changed_slots": ordinary_changes, "exact_half_ties": exact_ties,
                "slot_collisions": collisions,
                "residual_abs_max_sec": float(np.max(abs_residual)),
                "residual_abs_p50_sec": float(np.quantile(abs_residual, 0.50)),
                "residual_abs_p95_sec": float(np.quantile(abs_residual, 0.95)),
                "residual_abs_p99_sec": float(np.quantile(abs_residual, 0.99)),
                "residual_abs_max_cells": float(np.max(abs_residual) / dt),
                "minimum_half_cell_margin_sec": float(np.min(margin)),
                "minimum_half_cell_margin_cells": float(np.min(margin) / dt),
                "distribution_identity_sha256": sha256_bytes(np.ascontiguousarray(residual).tobytes()),
            })
        assert union_min is not None and union_max is not None
        added_positions = max(0, -union_min) + max(0, union_max - (current_count - 1))
        observation_rows.append({
            "config_id": obs["config_id"], "mode": obs["mode"],
            "fixture_role": obs["role"], "obsnum": obs["obsnum"],
            "interface_count": len(stream_records), "native_rate_factor": next(iter(factors)),
            "sample_rate_hz": sample_rate, "cadence_sec": dt,
            "exclusive_half_cell_sec": dt / 2.0, "phase_sec": phase,
            "current_overlap_end_sec": overlap_end, "current_grid_count": current_count,
            "union_min_slot": union_min, "union_max_slot": union_max,
            "union_grid_count": union_max - union_min + 1,
            "union_added_grid_positions": added_positions,
            "native_rows": obs_native_rows, "current_supported_rows": obs_current_rows,
            "union_edge_only_native_rows": obs_edge_rows,
            "ordinary_changed_slots": obs_ordinary_changes,
            "exact_half_ties": obs_exact_ties, "slot_collisions": obs_collisions,
            "residual_abs_max_sec": obs_max_residual,
            "residual_abs_max_cells": obs_max_residual / dt,
            "minimum_half_cell_margin_sec": obs_min_margin,
            "minimum_half_cell_margin_cells": obs_min_margin / dt,
        })
        grids[key] = {
            "phase": phase, "dt": dt, "current_count": current_count,
            "union_min": union_min, "union_max": union_max,
            "current_grid": phase + dt * np.arange(current_count, dtype=np.float64),
            "union_grid": phase + dt * np.arange(union_min, union_max + 1, dtype=np.float64),
        }
    return rate_rows, slot_rows, observation_rows, grids


def telescope_bracket_row(obs: dict[str, Any], grid: np.ndarray) -> dict[str, Any]:
    path = obs["telescope"]
    with Dataset(path) as dataset:
        times = np.asarray(dataset["Data.TelescopeBackend.TelTime"][:], dtype=np.float64)
    finite = np.isfinite(times)
    steps = np.diff(times)
    if not finite.all() or not np.all(steps > 0):
        raise RuntimeError(f"telescope TelTime is nonfinite/nonmonotonic: {path}")
    right = np.searchsorted(times, grid, side="left")
    supported = (right >= 0) & (right < times.size)
    exact = np.zeros(grid.size, dtype=bool)
    valid_right = right < times.size
    exact[valid_right] = times[right[valid_right]] == grid[valid_right]
    supported &= exact | (right > 0)
    if not supported.all():
        raise RuntimeError(f"telescope support does not bracket selected union grid: {path}")
    spans = np.zeros(grid.size, dtype=np.float64)
    interpolated = ~exact
    spans[interpolated] = times[right[interpolated]] - times[right[interpolated] - 1]
    return {
        "config_id": obs["config_id"], "mode": obs["mode"],
        "fixture_role": obs["role"], "obsnum": obs["obsnum"],
        "telescope_path": str(path), "target_grid_count": int(grid.size),
        "target_support": "D002 union grid", "exact_native_matches": int(exact.sum()),
        "interpolated_brackets": int(interpolated.sum()),
        "invalid_or_extrapolated_targets": int((~supported).sum()),
        "native_nonpositive_step_count": int(np.sum(steps <= 0)),
        "native_cadence_median_sec": float(np.median(steps)),
        "native_cadence_max_sec": float(np.max(steps)),
        "used_bracket_p50_sec": float(np.quantile(spans, 0.50)),
        "used_bracket_p95_sec": float(np.quantile(spans, 0.95)),
        "used_bracket_max_sec": float(np.max(spans)),
        "preregistered_fixed_cohort_limit_sec": TELESCOPE_BRACKET_LIMIT_SEC,
        "limit_scope": "fixed D005 cohort only; not a general production cadence authority",
    }


def periodic_fix(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    if float(np.max(result)) > 1.99 * math.pi and float(np.min(result)) < math.pi:
        result[result < math.pi] += 2.0 * math.pi
    return result


def bool_digest(values: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(values.astype(np.uint8)).tobytes())


def scan_digest(windows: list[tuple[int, int, int, int]]) -> str:
    return sha256_bytes((canonical_json(windows) + "\n").encode())


def zero_runs(state: np.ndarray) -> list[tuple[int, int]]:
    zero = ~state
    padded = np.concatenate(([False], zero, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def governing_scan_windows(
    state: np.ndarray, sample_rate: float, context: int,
) -> tuple[int, list[tuple[int, int, int, int]], list[int]]:
    zero_run_candidates = zero_runs(state)
    adjusted = []
    first_post = []
    for start, end in zero_run_candidates:
        science_start = start if start == 0 else start + 1
        science_end = end
        if science_end < science_start:
            continue
        if science_end - science_start + 1 < 2.0 * sample_rate:
            continue
        adjusted.append([science_start, science_end])
        if start > 0:
            first_post.append(science_start)
    windows: list[tuple[int, int, int, int]] = []
    for start, end in adjusted:
        windows.append((
            start, end, max(0, start - context), min(state.size - 1, end + context),
        ))
    if context > 0 and windows:
        first = windows[0]
        windows[0] = (min(first[0] + context, first[1]), first[1], first[2], first[3])
        last = windows[-1]
        windows[-1] = (last[0], max(last[0], last[1] - context), last[2], last[3])
    return len(zero_run_candidates), windows, first_post


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / denominator if denominator else 0.0


def hold_analysis(
    beam_obs: dict[str, Any], grid_info: dict[str, Any],
    source_crossing_path: Path,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any],
]:
    path = beam_obs["telescope"]
    grid = grid_info["current_grid"]
    union_grid = grid_info["union_grid"]
    union_min = int(grid_info["union_min"])
    union_max = int(grid_info["union_max"])

    def header_text(dataset: Dataset, name: str) -> str:
        values = np.asarray(dataset[name][:]).reshape(-1)
        return b"".join(bytes(value) for value in values).decode("ascii").strip()

    with Dataset(path) as dataset:
        tel_time = np.asarray(dataset["Data.TelescopeBackend.TelTime"][:], dtype=np.float64)
        raw_word_float = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=np.float64)
        fields = {
            name: periodic_fix(np.asarray(dataset[f"Data.TelescopeBackend.{name}"][:], dtype=np.float64))
            for name in ("TelAzAct", "TelElAct", "TelAzCor", "TelElCor", "SourceAz", "SourceEl")
        }
        x_length = float(np.asarray(dataset["Header.Map.XLength"][:]).item())
        y_length = float(np.asarray(dataset["Header.Map.YLength"][:]).item())
        scan_angle = float(np.asarray(dataset["Header.Map.ScanAngle"][:]).item())
        selector_headers = {
            "obs_pgm": header_text(dataset, "Header.Dcs.ObsPgm"),
            "map_path": header_text(dataset, "Header.Map.MapPath"),
            "map_coord": header_text(dataset, "Header.Map.MapCoord"),
            "map_motion": header_text(dataset, "Header.Map.MapMotion"),
            "exec_mode": int(np.asarray(dataset["Header.Map.ExecMode"][:]).item()),
        }
    if not np.all(np.isfinite(raw_word_float)):
        raise RuntimeError("Hold contains nonfinite values on Beammap 148670")
    if np.any(raw_word_float < 0.0) or np.any(raw_word_float >= float(2**64)):
        raise RuntimeError("Hold is outside the uint64 domain on Beammap 148670")
    if not np.all(raw_word_float == np.floor(raw_word_float)):
        raise RuntimeError("Hold is not exactly integral on Beammap 148670")
    raw_word = raw_word_float.astype(np.uint64)
    if not np.array_equal(raw_word.astype(np.float64), raw_word_float):
        raise RuntimeError("Hold cannot be converted to uint64 losslessly on Beammap 148670")
    config_payload = yaml.safe_load(Path(beam_obs["config_path"]).read_text())
    chunking = config_payload.get("timestream", {}).get("chunking", {})
    expected_selectors = {
        "obs_pgm": "Map", "map_path": "Rectilinear", "map_coord": "Az",
        "map_motion": "Continuous", "exec_mode": 0,
    }
    if selector_headers != expected_selectors:
        raise RuntimeError(f"Beammap raster selector headers changed: {selector_headers}")
    if not (
        config_payload.get("runtime", {}).get("reduction_type") == "beammap"
        and chunking.get("chunk_mode") == "duration"
        and chunking.get("force_chunking") is False
    ):
        raise RuntimeError(f"Beammap raster selector config changed: {chunking}")

    def aligned_hold_views(target_grid: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        linear_word = np.interp(target_grid, tel_time, raw_word_float)
        left_index = np.searchsorted(tel_time, target_grid, side="right") - 1
        right_index = np.searchsorted(tel_time, target_grid, side="left")
        if np.any(left_index < 0) or np.any(right_index >= tel_time.size):
            raise RuntimeError("Hold grid is outside native telescope support")
        left_word = raw_word[left_index]
        right_word = raw_word[right_index]
        aligned = {
            name: np.interp(target_grid, tel_time, values) for name, values in fields.items()
        }
        tel_az = aligned["TelAzAct"].copy()
        wrap = tel_az - aligned["SourceAz"] > 0.9 * 2.0 * math.pi
        tel_az[wrap] -= 2.0 * math.pi
        alt_phys = aligned["TelElAct"] - aligned["SourceEl"] - aligned["TelElCor"]
        az_phys = (
            np.cos(aligned["TelElAct"] - aligned["TelElCor"])
            * (tel_az - aligned["SourceAz"]) - aligned["TelAzCor"]
        )
        xp = az_phys * math.cos(scan_angle) + alt_phys * math.sin(scan_angle)
        yp = -az_phys * math.sin(scan_angle) + alt_phys * math.cos(scan_angle)
        inside = (
            (-x_length / 2.0 <= xp) & (xp <= x_length / 2.0)
            & (-y_length / 2.0 <= yp) & (yp <= y_length / 2.0)
        )
        hypotheses_for_grid = {
            "governing_9aae_linear_word_nonzero": linear_word != 0.0,
            "released_4x_linear_word_nonzero": linear_word != 0.0,
            "left_native_raw_word_nonzero": left_word != 0,
            "right_native_raw_word_nonzero": right_word != 0,
            "left_native_bit_0x08": (left_word & 0x08) != 0,
            "right_native_bit_0x08": (right_word & 0x08) != 0,
        }
        return hypotheses_for_grid, ~inside

    hypotheses, outside = aligned_hold_views(grid)
    union_hypotheses, union_outside = aligned_hold_views(union_grid)
    context = 32
    governing_final = hypotheses["governing_9aae_linear_word_nonzero"] | outside
    union_governing_final = (
        union_hypotheses["governing_9aae_linear_word_nonzero"] | union_outside
    )
    zero_run_count, governing_windows, governing_first_post = governing_scan_windows(
        governing_final, F_REF_HZ, context,
    )
    candidate_runs = zero_runs(governing_final)
    candidate_half_open_windows = [(start, end + 1) for start, end in candidate_runs]
    union_candidate_runs = zero_runs(union_governing_final)
    union_candidate_lattice_windows = [
        (start + union_min, end + 1 + union_min) for start, end in union_candidate_runs
    ]
    if union_candidate_lattice_windows != candidate_half_open_windows:
        raise RuntimeError(
            "D002 union-grid support changed the conditional Hold zero-run lattice identity"
        )
    union_added_slots = [
        lattice_slot for lattice_slot in range(union_min, union_max + 1)
        if lattice_slot < 0 or lattice_slot >= grid.size
    ]
    if len(union_added_slots) != 3 or any(
        not union_governing_final[lattice_slot - union_min]
        for lattice_slot in union_added_slots
    ):
        raise RuntimeError("D002 union-grid added positions changed Hold candidate support")
    eligible_candidate_ids = []
    for candidate_id, (start, end) in enumerate(candidate_runs):
        legacy_start = start if start == 0 else start + 1
        legacy_end = end
        legacy_length = max(0, legacy_end - legacy_start + 1)
        if legacy_length >= 2.0 * F_REF_HZ:
            eligible_candidate_ids.append(candidate_id)
    if len(eligible_candidate_ids) != len(governing_windows):
        raise RuntimeError("legacy scan reconstruction lost zero-run/output mapping")
    governing_output_by_candidate_id = {
        candidate_id: (output_id, window)
        for output_id, (candidate_id, window) in enumerate(
            zip(eligible_candidate_ids, governing_windows, strict=True), 1,
        )
    }
    candidate_rows = []
    for candidate_id, (start, end) in enumerate(candidate_runs):
        corrected_stop = end + 1
        legacy_start = start if start == 0 else start + 1
        legacy_end = end
        legacy_stop = max(legacy_start, legacy_end + 1)
        legacy_length = max(0, legacy_end - legacy_start + 1)
        current = governing_output_by_candidate_id.get(candidate_id)
        current_output_id = current[0] if current else None
        current_window = current[1] if current else (None, None, None, None)
        candidate_rows.append({
            "candidate_zero_run_q_0based": candidate_id,
            "conditional_lattice_slot_k_start": start,
            "conditional_lattice_slot_k_stop_exclusive": corrected_stop,
            "current_grid_array_index_start": start,
            "current_grid_array_index_stop_exclusive": corrected_stop,
            "union_grid_array_index_start": start - union_min,
            "union_grid_array_index_stop_exclusive": corrected_stop - union_min,
            "candidate_science_length_samples": corrected_stop - start,
            "conditional_context_lattice_slot_k_start": max(union_min, start - context),
            "conditional_context_lattice_slot_k_stop_exclusive": min(
                union_max + 1, corrected_stop + context
            ),
            "candidate_status_if_authorized_under_OD5": (
                "short_under_legacy_two_second_threshold"
                if corrected_stop - start < 2.0 * F_REF_HZ else "ordinary"
            ),
            "legacy_precontext_start": legacy_start,
            "legacy_precontext_stop_exclusive": legacy_stop,
            "legacy_precontext_length_samples": legacy_length,
            "legacy_first_false_after_state_boundary_sample_omitted": start if start > 0 else "",
            "legacy_last_pre_hold_sample_omitted": "",
            "legacy_kept_as_output": current is not None,
            "legacy_output_scan_index_1based": current_output_id if current else "",
            "legacy_output_inner_start_inclusive": current_window[0] if current else "",
            "legacy_output_inner_end_inclusive": current_window[1] if current else "",
            "legacy_output_outer_start_inclusive": current_window[2] if current else "",
            "legacy_output_outer_end_inclusive": current_window[3] if current else "",
            "conditional_successor_requirement": (
                "only if the owner authorizes the legacy whole-word-linear-any compatibility view "
                "combined with the separately applied governing outside-map-box condition to control "
                "raster segmentation: retain this q and half-open candidate window, separate context, "
                "and do not let processor eligibility/status delete or renumber identity"
            ),
        })

    with Dataset(source_crossing_path) as dataset:
        arrays = {
            name: np.asarray(dataset[name][:], dtype=np.int64)
            for name in (
                "detector_tod_scan_index", "detector_tod_scan_inner_start_sample",
                "detector_tod_scan_inner_end_sample", "detector_tod_scan_outer_start_sample",
                "detector_tod_scan_outer_end_sample",
            )
        }
    artifact_windows: dict[int, tuple[int, int, int, int]] = {}
    for indices in np.ndindex(arrays["detector_tod_scan_index"].shape):
        scan_index = int(arrays["detector_tod_scan_index"][indices])
        if scan_index <= 0:
            continue
        value = (
            int(arrays["detector_tod_scan_inner_start_sample"][indices]),
            int(arrays["detector_tod_scan_inner_end_sample"][indices]),
            int(arrays["detector_tod_scan_outer_start_sample"][indices]),
            int(arrays["detector_tod_scan_outer_end_sample"][indices]),
        )
        if scan_index in artifact_windows and artifact_windows[scan_index] != value:
            raise RuntimeError("source-crossing artifact has conflicting scan-window identity")
        artifact_windows[scan_index] = value
    artifact_mismatch_rows = [
        (index, governing_windows[index - 1], value)
        for index, value in artifact_windows.items()
        if governing_windows[index - 1] != value
    ]
    artifact_mismatches = len(artifact_mismatch_rows)
    if artifact_mismatches:
        raise RuntimeError(
            "governing scan reconstruction disagrees with source-crossing artifact: "
            f"{artifact_mismatch_rows[:3]}"
        )

    comparison_rows = []
    window_rows = []
    all_final_digests = set()
    all_candidate_zero_run_digests = set()
    all_union_final_digests = set()
    all_union_candidate_lattice_digests = set()
    for name, pre_state in hypotheses.items():
        final_state = pre_state | outside
        union_final_state = union_hypotheses[name] | union_outside
        zero_run_count, windows, first_post = governing_scan_windows(final_state, F_REF_HZ, context)
        corrected_windows = [(start, end + 1) for start, end in zero_runs(final_state)]
        union_lattice_windows = [
            (start + union_min, end + 1 + union_min)
            for start, end in zero_runs(union_final_state)
        ]
        final_digest = bool_digest(final_state)
        all_final_digests.add(final_digest)
        union_final_digest = bool_digest(union_final_state)
        all_union_final_digests.add(union_final_digest)
        corrected_digest = sha256_bytes((canonical_json(corrected_windows) + "\n").encode())
        all_candidate_zero_run_digests.add(corrected_digest)
        union_lattice_digest = sha256_bytes(
            (canonical_json(union_lattice_windows) + "\n").encode()
        )
        all_union_candidate_lattice_digests.add(union_lattice_digest)
        comparison_rows.append({
            "hypothesis": name,
            "aligned_pre_outside_true_rows": int(pre_state.sum()),
            "aligned_pre_outside_transition_count": int(np.sum(np.diff(pre_state.astype(np.int8)) != 0)),
            "pre_outside_differing_rows_vs_governing": int(np.sum(pre_state != hypotheses["governing_9aae_linear_word_nonzero"])),
            "outside_map_box_true_rows": int(outside.sum()),
            "final_true_rows": int(final_state.sum()),
            "final_transition_count": int(np.sum(np.diff(final_state.astype(np.int8)) != 0)),
            "final_state_sha256_uint8": final_digest,
            "final_state_zero_run_count": zero_run_count,
            "conditional_OD5_candidate_half_open_window_count": len(corrected_windows),
            "conditional_OD5_candidate_windows_sha256": corrected_digest,
            "conditional_candidate_window_differences_vs_governing": sum(
                left != right for left, right in itertools.zip_longest(
                    corrected_windows, candidate_half_open_windows,
                )
            ),
            "union_final_true_rows": int(union_final_state.sum()),
            "union_final_transition_count": int(
                np.sum(np.diff(union_final_state.astype(np.int8)) != 0)
            ),
            "union_final_state_sha256_uint8": union_final_digest,
            "union_conditional_candidate_lattice_window_count": len(union_lattice_windows),
            "union_conditional_candidate_lattice_windows_sha256": union_lattice_digest,
            "union_lattice_window_differences_vs_current_lattice": sum(
                left != right for left, right in itertools.zip_longest(
                    union_lattice_windows, corrected_windows,
                )
            ),
            "kept_scan_count": len(windows),
            "scan_windows_sha256": scan_digest(windows),
            "scan_window_differences_vs_governing": sum(
                left != right for left, right in itertools.zip_longest(windows, governing_windows)
            ),
            "first_false_after_state_boundary_count": len(first_post),
            "first_false_after_state_boundary_sha256": sha256_bytes(
                (canonical_json(first_post) + "\n").encode()
            ),
            "first_false_after_state_boundary_differences_vs_governing": sum(
                left != right for left, right in itertools.zip_longest(first_post, governing_first_post)
            ),
            "first_post_turn_semantic_status": (
                "conditional candidate only; raw Hold transition is not producer-authoritative turn state"
            ),
            "source_crossing_centroid_psf_effect": (
                "zero difference among tested Hold hypotheses after outside-box masking; "
                "conditional OD5 boundary-repair output effect remains unmeasured because no successor ran"
            ),
        })
        for scan_index, (inner_start, inner_end, outer_start, outer_end) in enumerate(windows, 1):
            window_rows.append({
                "hypothesis": name, "legacy_output_scan_index_1based": scan_index,
                "legacy_inner_start_inclusive": inner_start,
                "legacy_inner_end_inclusive": inner_end,
                "legacy_outer_start_inclusive": outer_start,
                "legacy_outer_end_inclusive": outer_end,
                "legacy_inner_as_half_open_start": inner_start,
                "legacy_inner_as_half_open_stop_exclusive": inner_end + 1,
                "legacy_selected_first_sample": inner_start,
                "interpretation": (
                    "direct governing-output compatibility evidence; not an authoritative physical-scan identity"
                ),
            })

    bit_summary = []
    bit_transitions = []
    for bit_index in range(8):
        mask = np.uint64(1 << bit_index)
        active = (raw_word & mask) != 0
        changes = np.flatnonzero(np.diff(active.astype(np.int8)) != 0) + 1
        bit_summary.append({
            "bit_index": bit_index, "bit_hex": f"0x{int(mask):02x}",
            "native_active_rows": int(active.sum()),
            "native_transition_count": int(changes.size),
            "observed": bool(active.any()),
            "semantic_status": (
                "candidate turning predicate; producer meaning unproved" if bit_index == 3 else
                "unknown and explicitly retained" if bit_index in {1, 6} else
                "not observed; producer meaning unproved"
            ),
        })
        for index in changes:
            bit_transitions.append({
                "bit_index": bit_index, "bit_hex": f"0x{int(mask):02x}",
                "native_right_index": int(index),
                "native_left_time_sec": float(tel_time[index - 1]),
                "native_right_time_sec": float(tel_time[index]),
                "from_active": bool(active[index - 1]), "to_active": bool(active[index]),
                "raw_word_left": int(raw_word[index - 1]), "raw_word_right": int(raw_word[index]),
            })

    native_bit02 = (raw_word & 0x02) != 0
    native_bit08 = (raw_word & 0x08) != 0
    native_bit40 = (raw_word & 0x40) != 0
    findings = {
        "test_id": "ALIGN-D004-HOLD-VALIDATION-001",
        "fixture": "Beammap 148670",
        "governing_application_sha": GOVERNING_APPLICATION_SHA,
        "telescope_path": str(path),
        "grid_count": int(grid.size), "grid_first_sec": float(grid[0]),
        "grid_cadence_sec": grid_info["dt"],
        "current_grid_lattice_slot_k_min": 0,
        "current_grid_lattice_slot_k_max": int(grid.size - 1),
        "union_grid_count": int(union_grid.size),
        "union_grid_lattice_slot_k_min": union_min,
        "union_grid_lattice_slot_k_max": union_max,
        "union_added_lattice_slots": union_added_slots,
        "union_added_positions_all_final_true": True,
        "union_support_preserves_candidate_lattice_windows": True,
        "raster_selector_headers": selector_headers,
        "raster_selector_config": {
            "runtime_reduction_type": config_payload["runtime"]["reduction_type"],
            "chunk_mode": chunking["chunk_mode"],
            "force_chunking": chunking["force_chunking"],
        },
        "raw_hold_validation": {
            "finite": True, "nonnegative": True, "integral": True,
            "uint64_lossless": True,
        },
        "raw_unique_words": [int(value) for value in np.unique(raw_word)],
        "raw_word_counts": {
            str(int(value)): int(np.sum(raw_word == value)) for value in np.unique(raw_word)
        },
        "raw_word_transition_count": int(np.sum(np.diff(raw_word) != 0)),
        "outside_map_box_true_rows": int(outside.sum()),
        "outside_map_box_transition_count": int(np.sum(np.diff(outside.astype(np.int8)) != 0)),
        "every_tested_hold_true_row_is_outside_box": all(
            bool(np.all(~state | outside)) for state in hypotheses.values()
        ),
        "all_final_hypotheses_identical": len(all_final_digests) == 1,
        "all_candidate_zero_run_windows_identical": len(all_candidate_zero_run_digests) == 1,
        "all_union_final_hypotheses_identical": len(all_union_final_digests) == 1,
        "all_union_candidate_lattice_windows_identical": (
            len(all_union_candidate_lattice_digests) == 1
        ),
        "governing_final_state_zero_runs": zero_run_count,
        "governing_kept_scan_count": len(governing_windows),
        "conditional_OD5_candidate_zero_run_identity_count": len(candidate_runs),
        "conditional_lattice_slot_half_open_windows_sha256": sha256_bytes(
            (canonical_json(candidate_half_open_windows) + "\n").encode()
        ),
        "conditional_window_digest_serialization": (
            "canonical compact JSON list of [lattice_slot_k_start,"
            "lattice_slot_k_stop_exclusive] pairs plus LF"
        ),
        "legacy_deleted_candidate_zero_run_count": len(candidate_runs) - len(governing_windows),
        "legacy_first_false_after_state_boundary_omission_count": sum(
            start > 0 for start, _ in candidate_runs
        ),
        "legacy_kept_candidate_q_min": min(eligible_candidate_ids),
        "legacy_kept_candidate_q_max": max(eligible_candidate_ids),
        "legacy_first_output_candidate_q": eligible_candidate_ids[0],
        "legacy_last_output_candidate_q": eligible_candidate_ids[-1],
        "legacy_first_output_additional_context_trim_samples": context,
        "legacy_last_output_additional_context_trim_samples": context,
        "scan_context_samples": context,
        "source_crossing_distinct_recorded_scan_count": len(artifact_windows),
        "source_crossing_recorded_scan_min": min(artifact_windows),
        "source_crossing_recorded_scan_max": max(artifact_windows),
        "source_crossing_window_mismatch_count": artifact_mismatches,
        "historical_suite_product_application_commit": SUITE_PRODUCT_SOURCE_COMMIT,
        "historical_suite_alignment_scan_sources_byte_identical_to_governing": True,
        "historical_product_is_exact_whole_application_governing_sha_execution": False,
        "compatibility_vs_repair_identity": {
            "direct_legacy_baseline": "198 legacy outputs with inclusive inner/outer bounds",
            "conditional_OD5_candidate": (
                "if owner Q1 authorizes the legacy whole-word-linear-any compatibility view combined "
                "with the separately applied governing outside-map-box condition to control raster "
                "segmentation: 241 stable zero-run q candidates with half-open windows, first false "
                "sample after each composite final-state boundary included, and context/status separated"
            ),
            "allowed_differences": (
                "only named first-sample-after-authorized-state-boundary, remainder, short/empty "
                "retention, context separation, "
                "and invalid/gap rejection repairs"
            ),
        },
        "unknown_bit_relationships": {
            "bit02_active": int(native_bit02.sum()),
            "bit02_overlap_bit08": int(np.sum(native_bit02 & native_bit08)),
            "bit02_unknown_only": int(np.sum(native_bit02 & ~native_bit08)),
            "p_bit08_given_bit02": float(np.mean(native_bit08[native_bit02])),
            "phi_bit02_bit08": phi_coefficient(native_bit02, native_bit08),
            "bit40_active": int(native_bit40.sum()),
            "bit40_overlap_bit08": int(np.sum(native_bit40 & native_bit08)),
            "bit40_unknown_only": int(np.sum(native_bit40 & ~native_bit08)),
            "p_bit08_given_bit40": float(np.mean(native_bit08[native_bit40])),
            "phi_bit40_bit08": phi_coefficient(native_bit40, native_bit08),
        },
        "owner_return_triggers": {
            "unknown_bits_correlate_materially_with_turn_candidate": True,
            "hypotheses_differ_materially_before_outside_condition": True,
            "transition_side_supported": False,
            "controlling_predicate_supported": False,
            "reason": (
                "zero discrete-identity tolerance is exceeded before outside-box masking, "
                "while identical final state/windows make predicate and side non-identifiable"
            ),
        },
        "verdict": "OWNER_AMENDMENT_REQUIRED_DO_NOT_SELECT_PREDICATE_OR_SIDE",
        "authoritative_physical_raster_segmentation_status": (
            "UNAVAILABLE: no producer-authoritative Hold predicate or transition side; owner return required"
        ),
    }
    return comparison_rows, window_rows, candidate_rows, bit_summary, bit_transitions, findings


def pointing_rows(path: Path, cohort: str, use_scope: str) -> list[dict[str, Any]]:
    table = Table.read(path, format="ascii.ecsv")
    obsnum = int(table.meta["obsnum"])
    result = []
    labels = {0: "a1100", 1: "a1400", 2: "a2000"}
    for row in table:
        array = labels[int(row["array"])]
        a = float(row["a_fwhm"])
        b = float(row["b_fwhm"])
        a_err = float(row["a_fwhm_err"])
        b_err = float(row["b_fwhm_err"])
        result.append({
            "cohort": cohort, "use_scope": use_scope, "obsnum": obsnum,
            "array": array, "product_path": str(path), "x_t_arcsec": float(row["x_t"]),
            "x_t_err_arcsec": float(row["x_t_err"]), "y_t_arcsec": float(row["y_t"]),
            "y_t_err_arcsec": float(row["y_t_err"]), "a_fwhm_arcsec": a,
            "a_fwhm_err_arcsec": a_err, "b_fwhm_arcsec": b,
            "b_fwhm_err_arcsec": b_err, "major_fwhm_arcsec": max(a, b),
            "minor_fwhm_arcsec": min(a, b),
            "major_fwhm_err_arcsec": a_err if a >= b else b_err,
            "minor_fwhm_err_arcsec": b_err if a >= b else a_err,
        })
    if len(result) != 3:
        raise RuntimeError(f"unexpected pointing table row count: {path}")
    return result


def repeatability_rows(fit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cohort in ("beammap_bracketing_pointings", "science_realized_support_pointings"):
        for array in ("a1100", "a1400", "a2000"):
            rows = [row for row in fit_rows if row["cohort"] == cohort and row["array"] == array]
            if len(rows) < 2:
                continue
            x = np.asarray([row["x_t_arcsec"] for row in rows])
            y = np.asarray([row["y_t_arcsec"] for row in rows])
            major = np.asarray([row["major_fwhm_arcsec"] for row in rows])
            minor = np.asarray([row["minor_fwhm_arcsec"] for row in rows])
            pairwise = [
                math.hypot(x[i] - x[j], y[i] - y[j])
                for i, j in itertools.combinations(range(len(rows)), 2)
            ]
            result.append({
                "cohort": cohort, "array": array, "observation_count": len(rows),
                "obsnums": canonical_json(sorted(row["obsnum"] for row in rows)),
                "centroid_max_pairwise_arcsec": max(pairwise),
                "centroid_component_median_radial_deviation_arcsec": float(np.median(
                    np.hypot(x - np.median(x), y - np.median(y))
                )),
                "major_fwhm_range_arcsec": float(np.max(major) - np.min(major)),
                "minor_fwhm_range_arcsec": float(np.max(minor) - np.min(minor)),
                "median_x_fit_uncertainty_arcsec": float(np.median([row["x_t_err_arcsec"] for row in rows])),
                "median_y_fit_uncertainty_arcsec": float(np.median([row["y_t_err_arcsec"] for row in rows])),
                "median_major_fit_uncertainty_arcsec": float(np.median([row["major_fwhm_err_arcsec"] for row in rows])),
                "median_minor_fit_uncertainty_arcsec": float(np.median([row["minor_fwhm_err_arcsec"] for row in rows])),
                "threshold_use": "descriptive only; cannot widen active exact successor policy",
            })
    return result


def beammap_science_summary(apt_path: Path, source_crossing_path: Path) -> dict[str, Any]:
    table = Table.read(apt_path, format="ascii.ecsv")
    labels = {0: "a1100", 1: "a1400", 2: "a2000"}
    with Dataset(source_crossing_path) as dataset:
        arrays = np.asarray(dataset["detector_tod_array"][:], int)
        uids = np.asarray(dataset["detector_tod_uid"][:], int)
        good = np.asarray(dataset["detector_tod_fit_good"][:], int) == 1
        distance = np.asarray(dataset["detector_tod_source_center_distance_arcsec"][:], float)
        scan_index = np.asarray(dataset["detector_tod_source_center_scan_index"][:], int)
        sample_rate = float(np.asarray(dataset["PTC_SAMPRATE"][:]).item())
        variables = set(dataset.variables)
    apt_uids = np.asarray(table["uid"], int)
    if len(np.unique(apt_uids)) != len(apt_uids) or set(apt_uids) != set(uids):
        raise RuntimeError("Beammap APT/source-crossing UID sets are not one-to-one")
    apt_index_by_uid = {int(uid): index for index, uid in enumerate(apt_uids)}
    apt_index = np.asarray([apt_index_by_uid[int(uid)] for uid in uids], int)
    apt_valid = np.asarray(table["flag"], float)[apt_index] == 0.0
    apt_array = np.asarray(table["array"], int)[apt_index]
    if not np.array_equal(apt_array, arrays):
        raise RuntimeError("Beammap APT/source-crossing array identities disagree")
    valid = good & apt_valid
    fit_rows = []
    crossing_rows = []
    for index, label in labels.items():
        selected_table = table[np.asarray(table["array"], int) == index]
        selected_table = selected_table[np.asarray(selected_table["flag"], float) == 0.0]
        selected_uids = set(int(uid) for uid in uids[valid & (arrays == index)])
        selected_table = selected_table[
            np.asarray([int(uid) in selected_uids for uid in selected_table["uid"]], bool)
        ]
        major = np.maximum(
            np.asarray(selected_table["a_fwhm"], float),
            np.asarray(selected_table["b_fwhm"], float),
        )
        minor = np.minimum(
            np.asarray(selected_table["a_fwhm"], float),
            np.asarray(selected_table["b_fwhm"], float),
        )
        fit_rows.append({
            "array": label, "valid_intersection_count": len(selected_table),
            "centroid_x_median_arcsec": float(np.median(selected_table["x_t"])),
            "centroid_y_median_arcsec": float(np.median(selected_table["y_t"])),
            "major_fwhm_median_arcsec": float(np.median(major)),
            "major_fwhm_p16_arcsec": float(np.quantile(major, 0.16)),
            "major_fwhm_p84_arcsec": float(np.quantile(major, 0.84)),
            "minor_fwhm_median_arcsec": float(np.median(minor)),
            "minor_fwhm_p16_arcsec": float(np.quantile(minor, 0.16)),
            "minor_fwhm_p84_arcsec": float(np.quantile(minor, 0.84)),
            "centroid_x_uncertainty_median_arcsec": float(np.median(selected_table["x_t_err"])),
            "centroid_y_uncertainty_median_arcsec": float(np.median(selected_table["y_t_err"])),
        })
        selected = distance[valid & (arrays == index)]
        crossing_rows.append({
            "array": label, "valid_intersection_count": int(selected.size),
            "closest_distance_median_arcsec": float(np.median(selected)),
            "closest_distance_p95_arcsec": float(np.quantile(selected, 0.95)),
            "closest_distance_max_arcsec": float(np.max(selected)),
        })
    return {
        "beammap_obsnum": int(table.meta["obsnum"]),
        "apt_row_count": len(table), "apt_flag_values": sorted(int(value) for value in np.unique(table["flag"])),
        "fit_summary": fit_rows,
        "source_crossing": {
            "detector_count": int(uids.size), "unique_uid_count": int(np.unique(uids).size),
            "source_fit_good_count": int(good.sum()),
            "apt_flag_zero_count": int(apt_valid.sum()),
            "valid_uid_array_aptflag_sourcefit_intersection_count": int(valid.sum()),
            "ptc_sample_rate_hz": sample_rate,
            "source_center_scan_index_min": int(np.min(scan_index[good])),
            "source_center_scan_index_max": int(np.max(scan_index[good])),
            "per_array": crossing_rows,
            "direct_timestamp_variable_present": any("time" in name.lower() for name in variables if "source_center" in name),
            "direct_closest_approach_sample_identity_present": False,
            "absolute_crossing_time_status": "UNAVAILABLE; do not infer from distance-only record",
        },
        "comparison_policy": (
            "join by stable detector UID/array and intersect APT flag==0 with source fit_good==1; "
            "unaffected records retain exact policy; no nonzero tolerance for OD5-changed records is owner-approved"
        ),
        "aggregation_role": "diagnostic summaries only; per-detector product rows control comparison",
    }


def add_product(
    rows: list[dict[str, Any]], cache: DigestCache, path: Path,
    cohort: str, role: str, evidence_class: str,
) -> None:
    path = require_file(path)
    rows.append({
        "cohort": cohort, "role": role, "evidence_class": evidence_class,
        "path": str(path), "size_bytes": path.stat().st_size,
        "sha256": cache.digest(path),
    })


def one_glob(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


def baseline_products(suite_root: Path, cache: DigestCache, repo: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    # Frozen phase-zero evidence explicitly binds pointings_v22.  Neither the
    # suite nor project.yaml selects a reduction-version directory by itself.
    point_root = suite_root / "point/pointings_v22/reduced/redu00"
    oof_root = suite_root / "oof/wilson/1146+399/reduced/redu00"
    beam_root = suite_root / "beammaps/3c273/reduced/redu00"
    beam_point_root = suite_root / "beammaps/pointings/reduced/redu00"
    science_point_root = suite_root / "science/pointings/reduced/redu00"
    science_root = suite_root / "science/reduction/NGC4449/reduced"

    reduction_roots = (
        ("point_core", point_root), ("oof_context", oof_root),
        ("beammap_core", beam_root), ("beammap_support", beam_point_root),
        ("science_support", science_point_root),
    )
    for cohort, root in reduction_roots:
        for name, role in (
            ("citlali.log.gz", "historical_runtime_log"),
            ("citlali_profile.ecsv", "historical_stage_profile"),
            ("runtime_provenance.yaml", "historical_runtime_policy"),
        ):
            add_product(rows, cache, root / name, cohort, role, "suite_historical_product")
        add_product(rows, cache, one_glob(root, "citlali_o*.yaml"), cohort, "low_level_config", "suite_historical_product")

    pointing_sets = (
        ("point_core", point_root, (152389,)),
        ("oof_context", oof_root, (152385, 152386, 152387)),
        ("beammap_support", beam_point_root, (148669, 148671)),
        ("science_support", science_point_root, (152389, 152391, 152393)),
    )
    pointing_paths = []
    for cohort, root, obsnums in pointing_sets:
        for obsnum in obsnums:
            obs_root = root / str(obsnum)
            ppt = one_glob(obs_root / "raw", "ppt_*.ecsv")
            pointing_paths.append((cohort, ppt))
            add_product(rows, cache, ppt, cohort, "pointing_centroid_psf_table", "suite_historical_metric_product")
            add_product(rows, cache, one_glob(obs_root / "raw", "*rtcdiag.nc"), cohort, "scan_diagnostic", "suite_historical_metric_product")
            for name in ("raw_timestream_provenance.yaml", "timestream_output_provenance.yaml"):
                add_product(rows, cache, obs_root / name, cohort, name.removesuffix(".yaml"), "suite_historical_provenance")

    beam_obs = beam_root / "148670"
    apt = beam_obs / "raw/apt_commissioning_beammap_148670_citlali.ecsv"
    fit_qc = beam_obs / "raw/apt_commissioning_beammap_148670_citlali_fit_qc.ecsv"
    source_crossing = beam_obs / "raw/source_crossing_tod/toltec_commissioning_beammap_148670_ptc_detector_tod.nc"
    add_product(rows, cache, apt, "beammap_core", "detector_centroid_psf_table", "suite_historical_metric_product")
    add_product(rows, cache, fit_qc, "beammap_core", "fit_qc_table", "suite_historical_metric_product")
    add_product(rows, cache, source_crossing, "beammap_core", "source_crossing_and_scan_artifact", "suite_historical_metric_product")
    add_product(
        rows, cache, one_glob(beam_obs / "raw/source_crossing_tod", "*rtcdiag.nc"),
        "beammap_core", "scan_diagnostic", "suite_historical_metric_product",
    )
    for name in ("raw_timestream_provenance.yaml", "timestream_output_provenance.yaml"):
        add_product(rows, cache, beam_obs / name, "beammap_core", name.removesuffix(".yaml"), "suite_historical_provenance")

    for iteration in range(4):
        root = science_root / f"redu{iteration:02d}"
        for name, role in (
            ("citlali.log.gz", "historical_runtime_log"),
            ("runtime_provenance.yaml", "historical_runtime_policy"),
        ):
            add_product(rows, cache, root / name, f"science_iteration_{iteration}", role, "suite_historical_fruit_loop_iteration")
        profile = root / "citlali_profile.ecsv"
        if profile.is_file():
            add_product(rows, cache, profile, f"science_iteration_{iteration}", "historical_stage_profile", "suite_historical_fruit_loop_iteration")
        add_product(rows, cache, one_glob(root, "citlali_o*.yaml"), f"science_iteration_{iteration}", "low_level_config", "suite_historical_fruit_loop_iteration")
    final_science = science_root / "redu03"
    for path in sorted(final_science.glob("**/*_citlali.fits")):
        add_product(rows, cache, path, "science_final_iteration", "science_map_sentinel", "suite_historical_metric_product")
    for obsnum in (152390, 152392):
        obs_root = final_science / str(obsnum)
        add_product(
            rows, cache, one_glob(obs_root / "raw", "*rtcdiag.nc"),
            "science_final_iteration", "scan_diagnostic", "suite_historical_metric_product",
        )
        add_product(
            rows, cache, one_glob(obs_root / "raw", "*ptcdiag.nc"),
            "science_final_iteration", "ptc_diagnostic", "suite_historical_metric_product",
        )
        for name in ("raw_timestream_provenance.yaml", "timestream_output_provenance.yaml"):
            add_product(rows, cache, obs_root / name, "science_final_iteration", name.removesuffix(".yaml"), "suite_historical_provenance")

    phase0_science = json.loads((repo / "validation" / PHASE0_PACKAGE / "science_compatibility_evidence.json").read_text())
    accepted = (
        ("accepted_point", phase0_science["accepted_pointing"]["ppt_path"], "accepted_pointing_table"),
        ("accepted_beammap", phase0_science["accepted_beammap"]["apt_path"], "accepted_beammap_table"),
        ("accepted_beammap", phase0_science["accepted_source_crossing"]["path"], "accepted_source_crossing"),
    )
    for cohort, raw_path, role in accepted:
        add_product(rows, cache, Path(raw_path), cohort, role, "accepted_historical_refactor_snapshot_not_governing_sha")

    return rows, {
        "beam_apt": apt, "beam_fit_qc": fit_qc,
        "beam_source_crossing": source_crossing,
        "pointing_paths": pointing_paths,  # type: ignore[dict-item]
    }


def historical_scan_product_rows(product_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for product in product_rows:
        if product["role"] != "scan_diagnostic":
            continue
        path = Path(product["path"])
        with Dataset(path) as dataset:
            if "n_scans" not in dataset.dimensions or "output_scan_index" not in dataset.variables:
                raise RuntimeError(f"historical scan diagnostic lacks scan identity: {path}")
            scan_index = np.asarray(dataset["output_scan_index"][:], dtype=np.int64)
            duration = np.asarray(dataset["scan_duration_s"][:], dtype=np.float64)
        expected_index = np.arange(1, scan_index.size + 1, dtype=np.int64)
        rows.append({
            "cohort": product["cohort"], "path": str(path),
            "file_sha256": product["sha256"], "historical_output_scan_count": scan_index.size,
            "historical_output_scan_index_min": int(np.min(scan_index)),
            "historical_output_scan_index_max": int(np.max(scan_index)),
            "historical_output_scan_indices_contiguous_1based": bool(np.array_equal(scan_index, expected_index)),
            "historical_output_scan_index_sha256_int64": sha256_bytes(
                np.ascontiguousarray(scan_index).tobytes()
            ),
            "scan_duration_min_sec": float(np.min(duration)),
            "scan_duration_median_sec": float(np.median(duration)),
            "scan_duration_max_sec": float(np.max(duration)),
            "use": (
                "historical scan-count/index/duration sentinel only; it neither establishes "
                "authoritative physical raster segmentation nor proves a successor"
            ),
        })
    return rows


def log_identity(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", errors="replace") as handle:
        text = handle.read()
    timestamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]", text)
    version_match = re.search(r"citlali version:\s*(\S+)", text)
    serious = len(re.findall(r"\[(?:error|critical)\]", text, flags=re.IGNORECASE))
    if len(timestamps) < 2 or not version_match:
        raise RuntimeError(f"cannot parse historical log identity: {path}")
    first = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S.%f")
    last = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S.%f")
    return {
        "path": str(path), "version": version_match.group(1),
        "log_interval_sec": (last - first).total_seconds(),
        "serious_message_count": serious,
    }


def runtime_evidence(product_rows: list[dict[str, Any]]) -> dict[str, Any]:
    logs = [Path(row["path"]) for row in product_rows if row["role"] == "historical_runtime_log"]
    log_rows = [log_identity(path) for path in logs]
    profiles = []
    for row in product_rows:
        if row["role"] != "historical_stage_profile":
            continue
        table = Table.read(row["path"], format="ascii.ecsv")
        stage_column = "stage" if "stage" in table.colnames else table.colnames[0]
        profiles.append({
            "path": row["path"], "record_count": len(table),
            "unique_stage_count": len(set(str(value) for value in table[stage_column])),
            "alignment_setup_stage_present": any(
                "align" in str(value).lower() and "setup" in str(value).lower()
                for value in table[stage_column]
            ),
        })
    return {
        "historical_logs": log_rows,
        "historical_profiles": profiles,
        "interpretation": (
            "single historical reductions and science fruit-loop iterations are not paired repeatability trials"
        ),
        "preregistered_runtime_rule": {
            "authority": "doc/PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md plus D003 no-repeatable-regression rule",
            "design": (
                "same host/allocation/input/config/build/dependencies/storage/thread policy; one warmup per role; "
                "five retained alternating measured baseline/successor pairs"
            ),
            "primary_metric": "Citlali internal log interval; external wall retained as operational context",
            "aggregation": "paired successor/baseline ratios, median and IQR; retain every pair",
            "limits": [
                "fail if median successor/baseline wall ratio > 1.05 (existing accepted ceiling)",
                "owner return if successor/baseline ratio > 1.0 in all five measured pairs (repeatable same-sign slowdown)",
                "fail incomplete/mismatched runs or any unexpected serious log message",
            ],
            "setup_metric": (
                "paired elapsed alignment setup stage on the same five runs; no current dedicated stage or numerical equivalence margin exists"
            ),
        },
        "io_storage_rule": {
            "io_metrics": "paired filesystem counters and exact input/output inventory; counters are comparative indicators, not bytes",
            "storage_metrics": "exact scientific product inventory/bytes plus incremental ALIGN provenance bytes",
            "structural_limit": (
                "no routine dense sample-by-detector, per-sample, per-pixel provenance, response, or covariance products; "
                "new state is O(observations*interfaces + transitions/gaps/exceptions)"
            ),
            "numeric_byte_limit": "UNAVAILABLE until compact schema/serialization is owner-approved",
        },
    }


def suite_realization(suite_root: Path, cache: DigestCache) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suite_path = require_file(suite_root / "suite.yaml")
    marker_path = require_file(suite_root / ".tolproj-validation-suite.yaml")
    suite = yaml.safe_load(suite_path.read_text())
    marker = yaml.safe_load(marker_path.read_text())
    if marker.get("suite_sha256") != cache.digest(suite_path):
        raise RuntimeError("TolProj suite marker does not bind current suite.yaml")
    rows = []
    expected = {
        "point": ([152389], [152389], []),
        "oof": ([152385, 152386, 152387], [152385, 152386, 152387], []),
        "beammap": ([148670], [148669, 148670, 148671], []),
        "science": (
            [152389, 152390, 152391, 152392, 152393, 152418, 152420, 152430, 152432, 152434],
            [152389, 152390, 152391, 152392, 152393],
            list(UNREALIZED_SCIENCE_SUPPORT),
        ),
    }
    for project, (selected, realized, missing) in expected.items():
        selected_from_suite = [
            int(value) for value in suite["projects"][project]["obsnums"]
        ]
        if selected_from_suite != selected:
            raise RuntimeError(
                f"suite selection identity changed for {project}: {selected_from_suite}"
            )
        for obsnum in selected:
            is_mandatory = (
                (project == "point" and obsnum == 152389)
                or (project == "beammap" and obsnum == 148670)
            )
            role = (
                "mandatory_core" if is_mandatory else
                "science_primary" if obsnum in {152390, 152392} else
                "oof_context" if project == "oof" else
                "supporting_pointing"
            )
            rows.append({
                "project": project, "obsnum": obsnum, "fixture_role": role,
                "selected_by_suite": True, "realized_in_project_and_selected_config": obsnum in realized,
                "realization_status": (
                    "realized_local" if obsnum in realized else
                    "selected_and_owner_local_raw_available_but_absent_from_realized_suite_project_config"
                ),
                "D005_use": (
                    "mandatory" if is_mandatory else
                    "long_mode_timing_scan_product_sentinel" if role == "science_primary" else
                    "same_observation_timing_scan_product_sentinel" if project == "oof" else
                    "support_repeatability" if obsnum in realized else
                    "native_rate_inventory_only_product_evidence_pending"
                ),
            })
        if missing and missing != list(UNREALIZED_SCIENCE_SUPPORT):
            raise RuntimeError("unexpected missing science support identity")
    authority = [
        {
            "authority_id": "tolproj_suite_selection",
            "authority_class": "selection_authority",
            "path": str(suite_path), "git_commit": "N/A owner-local versioned suite",
            "sha256": cache.digest(suite_path), "use": "selected observation cohort",
        },
        {
            "authority_id": "tolproj_suite_state_marker",
            "authority_class": "selection_identity_binding",
            "path": str(marker_path), "git_commit": "N/A owner-local suite state",
            "sha256": cache.digest(marker_path), "use": "suite identity and requested observations",
        },
    ]
    for project in ("point", "oof", "beammaps", "science"):
        path = require_file(suite_root / project / "project.yaml")
        project_payload = yaml.safe_load(path.read_text())
        expected_key = "beammap" if project == "beammaps" else project
        expected_realized = expected[expected_key][1]
        if sorted(int(value) for value in project_payload.get("obsnums", [])) != sorted(expected_realized):
            raise RuntimeError(f"realized project membership changed for {project}")
        authority.append({
            "authority_id": f"realized_project:{project}",
            "authority_class": "realized_local_project_definition",
            "path": str(path), "git_commit": "N/A owner-local realized project",
            "sha256": cache.digest(path),
            "use": "realized project membership/status; status flags are not filesystem availability",
        })
    return rows, authority


def preregistration_protocol(observation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mandatory = {
        row["obsnum"]: row for row in observation_rows
        if row["config_id"] in {"point_core", "beammap_core"}
    }
    return {
        "protocol_id": "ALIGN-P0-D005",
        "status": "PREREGISTERED_OWNER_RETURN_REQUIRED",
        "successor_output_viewed": False,
        "phase_one_authorized": False,
        "cohort": {
            "mandatory": ["Pointing 152389", "Beammap 148670"],
            "beammap_support": [148669, 148671],
            "oof_context": [152385, 152386, 152387],
            "science_primary": [152390, 152392],
            "science_support_realized": [152389, 152391, 152393],
            "science_support_selected_but_unrealized": list(UNREALIZED_SCIENCE_SUPPORT),
            "supplemental_beammap_152307": "not selected; owner approval required because it is outside suite and heterogeneous",
        },
        "metrics": [
            {
                "id": "native_rate_identity",
                "definition": "per file/interface exact SampleFreq, FpgaFreq, AccumLen and common D002 factor",
                "aggregation": "all selected detector files, grouped by observation",
                "limit": "zero missing/conflicting headers; zero mixed-rate observations",
                "status": "frozen; local coverage is genuine native 1x only",
            },
            {
                "id": "native_row_slot_support_identity",
                "definition": "join observation/interface/native-row; compare round-half-up integer slot and typed support class",
                "aggregation": "exact row counts plus compact exception list only",
                "limit": (
                    "zero changed slots for every governing-supported ordinary row; zero lost/colliding/duplicate rows; "
                    f"Point 152389 preserves {mandatory[152389]['union_edge_only_native_rows']} edge rows and "
                    f"Beammap 148670 preserves {mandatory[148670]['union_edge_only_native_rows']} edge rows; "
                    "each mandatory observation adds exactly three union-grid positions"
                ),
                "status": "frozen",
            },
            {
                "id": "slot_residual_margin",
                "definition": "r=t_after_offset-(phase+k*dt); u=abs(r)/dt; margin=0.5-u",
                "aggregation": "per interface count/max/p50/p95/p99 and observation extrema",
                "limit": (
                    "strict u<0.5; at native 1x abs(r)<0.004096 s; unchanged inputs may not reduce frozen "
                    "minimum margin or change the residual distribution identity"
                ),
                "status": "frozen for 1x; empirical 0.5x/2x/4x margins unavailable",
            },
            {
                "id": "telescope_adjacent_bracket",
                "definition": "TelTime[right]-TelTime[left] for truly adjacent finite native rows bracketing each union-grid target; exact match span=0",
                "aggregation": "per observation count/p50/p95/max and fixed-cohort maximum",
                "limit": (
                    "inclusive <=0.021130561828613281 s; reject nonfinite/nonmonotonic endpoints, gaps, "
                    "cross-gap brackets, ambiguity, and extrapolation"
                ),
                "status": "frozen as fixed-cohort validation envelope only",
            },
            {
                "id": "scan_and_first_post_turn_identity",
                "definition": (
                    "exact governing legacy aligned state and 198 output windows; separately, if the "
                    "owner authorizes the legacy whole-word-linear-any compatibility view combined with "
                    "the separately applied governing outside-map-box condition to control raster segmentation, "
                    "stable candidate q plus ordered half-open science [start,stop), separate context, "
                    "status, and first false sample after every composite final-state true-to-false boundary"
                ),
                "aggregation": (
                    "exact 198 governing legacy outputs plus a conditional 241-row zero-run candidate "
                    "registry and tuple digest"
                ),
                "limit": (
                    "the 198 governing legacy outputs must remain exactly reconstructable; only if owner "
                    "Q1 authorizes that composite final state as raster segmentation, require exactly 241 "
                    "candidate identities q=0..240 with no deletion/renumbering and science start at the "
                    "first false sample, with only OD5-named boundary/context/status repairs"
                ),
                "status": (
                    "legacy baseline frozen; conditional candidate evidence frozen; authoritative physical "
                    "raster segmentation unavailable pending owner return"
                ),
            },
            {
                "id": "source_crossing",
                "definition": "stable detector UID/array/fit validity/scan tuple/closest-distance and any retained sample-time identity",
                "aggregation": "record-level exact comparison; per-array summaries diagnostic only",
                "limit": (
                    "zero difference for unaffected records; changed records must be attributable only "
                    "to named OD5 scan repairs and show no worsening, but a nonzero numeric tolerance "
                    "is unavailable pending owner decision"
                ),
                "status": (
                    "record/UID/distance/legacy-scan identity frozen; direct absolute crossing time and "
                    "changed-window tolerance unavailable"
                ),
            },
            {
                "id": "point_centroid_psf",
                "definition": "per-array validity then x_t/y_t and derived major/minor FWHM",
                "aggregation": "record-level comparison; summaries diagnostic",
                "limit": "zero difference under active Point policy",
                "status": "frozen",
            },
            {
                "id": "beammap_centroid_psf",
                "definition": "join valid detector fits by UID/array; compare flags, x_t/y_t and major/minor FWHM",
                "aggregation": "record-level exact comparison; per-array median/quantiles diagnostic",
                "limit": (
                    "zero difference for unaffected records under active Beammap policy; no nonzero "
                    "centroid/PSF tolerance is frozen for records affected by named OD5 repairs"
                ),
                "status": (
                    "unaffected exact gate frozen; one full Beammap cannot establish changed-window "
                    "repeatability or a nonzero threshold"
                ),
            },
            {
                "id": "oof_same_observation_sentinel",
                "definition": "each focus observation against its own exact historical baseline",
                "aggregation": "no cross-focus morphology aggregation",
                "limit": "zero difference under active OOF policy",
                "status": "frozen; focus sequence is not PSF repeatability evidence",
            },
            {
                "id": "science_long_mode_sentinel",
                "definition": "existing exact product/integer checks and numerical comparison policy only",
                "aggregation": "existing science comparator",
                "limit": "map RMS<=1e-8; PTC-weight RMS<=1e-9; detector median absolute/fractional<=5e-5/1e-3; other diagnostic RMS<=1e-7",
                "status": "frozen; not an ALIGN centroid/source-estimator/PSF policy",
            },
            {
                "id": "setup_runtime",
                "definition": "paired same-host/input/config/build/runtime-policy elapsed setup and total times",
                "aggregation": "five alternating retained pairs; paired ratios, median and IQR",
                "limit": "median total ratio<=1.05; owner return if all five ratios>1.0; incomplete/mismatched evidence fails",
                "status": "design frozen; no local paired distribution or setup-stage baseline exists",
            },
            {
                "id": "io_storage_proportionality",
                "definition": "paired filesystem indicators, exact product inventory/bytes, incremental ALIGN provenance bytes and scaling dimension",
                "aggregation": "per run and product class",
                "limit": "no routine dense outputs; only O(observations*interfaces + transitions/gaps/exceptions); exact scientific inventory preserved",
                "status": "structural rule frozen; numeric byte limit unavailable",
            },
        ],
        "missing_stratum_policy": {
            "0.5x": "evidence_pending: native Pointing or Beammap fixture with SampleFreq=61.03515625 Hz and AccumLen=4194304",
            "2x": "evidence_pending: native Pointing or Beammap fixture with SampleFreq=244.140625 Hz and AccumLen=1048576",
            "4x": "evidence_pending: native Pointing or Beammap fixture with SampleFreq=488.28125 Hz and AccumLen=524288",
            "prohibition": "do not resample 1x or scale its empirical residual distribution",
        },
        "stop_rules": [
            "any ordinary governing-supported row changes slot/support identity",
            "any strict half-cell, collision, gap, bracket, validity, or extrapolation failure",
            (
                "any governing legacy scan mismatch; if owner Q1 authorizes the composite-final-state "
                "candidate, any difference outside named first-post-state/remainder/short/context/invalid-gap "
                "repairs; or any source-crossing/centroid/PSF/active-profile regression"
            ),
            "any repeatable measurable runtime/I/O regression or dense/material storage burden",
            "unknown Hold bits correlate materially, Hold hypotheses differ, no side is supported, or accepted scan/astrometric behavior regresses",
            "missing native-rate evidence remains pending and cannot be counted as a pass",
        ],
        "CAL_AST_safe_scope": (
            "centroid, PSF, source-crossing, WCS, and map outputs are unchanged downstream sentinels only; "
            "do not tune flux/calibration/APT eligibility/OOF estimation/source estimation/mapmaking/inverse-TAN/frames/signs/handedness"
        ),
    }


def owner_brief(hold_findings: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "ALIGN-P0-D005",
        "recommendation": "RECORD_PREREGISTRATION_WITH_EXPLICIT_EVIDENCE_GAPS_AND_OWNER_RETURN_REQUIRED",
        "phase_one_authorization": "NONE",
        "decisive_stop": hold_findings["verdict"],
        "measured_conclusions": [
            "Pointing 152389 and Beammap 148670 are suitable mandatory 1x core fixtures.",
            (
                "All 187 inventoried detector-file references are genuine native 1x; after "
                "observation/interface canonicalization there are 176 identities across 16 observations. "
                "Native 0.5x, 2x, and 4x are absent."
            ),
            "The fixed cohort supports an inclusive telescope adjacent-bracket envelope of 0.021130561828613281 s only.",
            "OOF focus diversity and extended-source science cannot set ALIGN centroid/PSF repeatability tolerances.",
            "Historical suite products are not exact governing-9aae executions and single runtimes are not repeatability distributions.",
            (
                "Beammap 148670 composite Hold-plus-outside final state cannot discriminate raw predicate or transition "
                "side because outside-map masking subsumes all tested Hold states. Its 241 maximal false "
                "runs are a conditional segmentation candidate only, not authoritative physical scans; "
                "the 198 legacy outputs remain the exact governing compatibility baseline."
            ),
        ],
        "tolerances_frozen_now": [
            "native 1x strict slot support abs(residual)<0.004096 s and zero ordinary slot changes/collisions",
            "fixed-cohort telescope adjacent-bracket envelope <=0.021130561828613281 s",
            "Beammap 148670 exact 198-window governing legacy output identity",
            (
                "conditional Beammap 148670 241-zero-run candidate count and half-open digest as "
                "preregistration evidence only, not physical-scan authority"
            ),
            "exact unaffected Point/OOF/Beammap record/product comparison",
            "existing Science comparator limits only",
            "structural no-dense-output and O(observations*interfaces + exceptions) rule",
        ],
        "unresolved_authority_facts": [
            "producer-authoritative Hold bit meaning and left/right transition placement",
            "authoritative physical raster segmentation for Beammap 148670",
            "native 0.5x, 2x, and 4x observational residual/jitter profiles",
            "general telescope producer cadence/gap duration rather than fixed-cohort envelope",
            "exact whole-application governing-9aae product run for direct comparison",
            "direct source-crossing closest-sample/timestamp identity",
            "nonzero source-crossing/centroid/PSF tolerance for OD5-changed records",
            "paired setup/runtime distribution and setup-stage numerical limit",
            "numeric I/O/storage byte ceiling and peak-memory budget",
            "enabled HWPR/polarization schema, time, angle, support, and offset authority",
            "owner-approved second Beammap reduction for the D002 combined-Beammap study",
        ],
        "CAL_AST_safe_decision_boundary": (
            "ALIGN may use downstream astrometric/scientific products only as sentinels. No flux, "
            "calibration, APT eligibility, OOF estimator, source estimator, mapmaking, WCS/frame, "
            "projection, sign, or handedness policy may be changed or tuned."
        ),
        "owner_questions": [
            {
                "id": "D005-Q1-HOLD",
                "question": (
                    "Will the owner explicitly authorize the named legacy whole-word-linear-any "
                    "compatibility view combined with the separately applied governing outside-map-box "
                    "condition to control raster segmentation without a producer-turn claim, "
                    "leave stronger turnaround state unavailable, and apply only the OD5 scan-boundary "
                    "repairs; or keep physical raster segmentation unavailable pending a separate "
                    "scientific amendment with discriminating authority/evidence?"
                ),
            },
            {
                "id": "D005-Q2-RATES",
                "question": (
                    "May phase one be restricted to native 1x while 0.5x/2x/4x observational evidence remains pending, "
                    "or are all four native observational strata prerequisites?"
                ),
            },
            {
                "id": "D005-Q3-TELESCOPE",
                "question": (
                    "Is 0.021130561828613281 s approved only as this fixed-cohort validation envelope, "
                    "or as an admitted runtime limit? No general producer cadence/gap bound is proved."
                ),
            },
            {
                "id": "D005-Q4-BASELINE",
                "question": (
                    "Which direct comparison authority controls phase one: a future exact governing-SHA 9aae run, "
                    "or an explicit amendment naming historical accepted/suite snapshots?"
                ),
            },
            {
                "id": "D005-Q5-RUNTIME",
                "question": (
                    "Does the owner approve the five-pair runtime rule (existing 5% median ceiling plus owner return for five same-sign slowdowns) "
                    "and a new setup-stage measurement, while I/O/storage retain only the frozen "
                    "structural compactness rule until paired evidence exists; or what exact rule replaces it?"
                ),
            },
            {
                "id": "D005-Q6-SCIENCE-SUPPORT",
                "question": (
                    "Must the realized suite be completed with 152418/152420/152430/152432/152434, "
                    "whose complete owner-local raw sets are now digest-bound, or is their raw-header-only "
                    "1x evidence sufficient while product/repeatability evidence stays pending?"
                ),
            },
            {
                "id": "D005-Q7-COMBINED-BEAMMAP",
                "question": (
                    "D002 assigns a combined-Beammap study to D005, but 148670 is the only selected "
                    "suite Beammap. Will the owner admit and bind supplemental heterogeneous/out-of-suite "
                    "Beammap 152307, name another already-local accepted Beammap reduction, or explicitly "
                    "leave the combined-observation residual/source-crossing/centroid/PSF study evidence-pending?"
                ),
            },
            {
                "id": "D005-Q8-CHANGED-SCIENCE",
                "question": (
                    "For records changed solely by approved OD5 window repairs, must the current exact "
                    "Point/Beammap product policy remain the gate (so any numeric change stops), or will "
                    "the owner approve a separately specified uncertainty/repeatability-derived "
                    "non-degradation rule? D005 evidence cannot freeze a nonzero value."
                ),
            },
        ],
        "explicit_non_authorizations": [
            "phase one", "application edits", "new Hold predicate or transition side",
            "Unity evidence request", "CAL/AST/OOF/mapmaking/source-estimator repair",
            "enabled HWPR/polarization", "re-audit", "merge/rebase/push/production expansion",
        ],
    }


def report_text(
    identity: dict[str, Any], rate_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]], telescope_rows: list[dict[str, Any]],
    hold_findings: dict[str, Any], brief: dict[str, Any], product_rows: list[dict[str, Any]],
    rate_coverage: dict[str, Any], external_summary: dict[str, Any],
) -> str:
    mandatory = [
        row for row in observation_rows if row["config_id"] in {"point_core", "beammap_core"}
    ]
    max_bracket = max(row["used_bracket_max_sec"] for row in telescope_rows)
    versions = sorted({
        log_identity(Path(row["path"]))["version"]
        for row in product_rows if row["role"] == "historical_runtime_log"
    })
    mandatory_lines = "\n".join(
        f"| {row['mode']} {row['obsnum']} | {row['native_rows']:,} | "
        f"{row['current_supported_rows']:,} | {row['union_edge_only_native_rows']} | "
        f"{row['union_added_grid_positions']} | {row['residual_abs_max_sec'] * 1e3:.9f} | "
        f"{row['minimum_half_cell_margin_sec'] * 1e6:.6f} |"
        for row in mandatory
    )
    questions = "\n".join(
        f"{index}. **{item['id']}** — {item['question']}"
        for index, item in enumerate(brief["owner_questions"], 1)
    )
    return f"""# SCI-ALIGN-001 ALIGN-P0-D005 preregistration decision package

Date: 2026-08-01

Verdict: **OWNER_RETURN_REQUIRED — PHASE ONE UNAUTHORIZED**

## Frozen identity and scope

- Repair branch: `{identity['branch']}` at frozen phase-zero commit `{identity['phase_zero_evidence_commit']}`.
- Exact governing application parent: `{identity['governing_application_sha']}`.
- Read-only coordination authority: `{identity['coordination_head']}`; clean at generation.
- Corrected owner-decision identity: `{identity['owner_decision_identity_correction']['authoritative_commit']}`; the rejected expanded transcription `{identity['owner_decision_identity_correction']['rejected_transcription']}` is not used.
- Frozen phase-zero `SHA256SUMS`: `{identity['phase_zero_sha256sums_sha256']}`. The package was verified and not rewritten.
- This package contains evidence/protocol only. No Citlali execution, application edit, Unity access, TolProj/suite mutation, sibling-repository edit, phase-one fixture, re-audit, merge, rebase, push, or production expansion occurred.

## Cohort and evidence classification

Pointing 152389 and Beammap 148670 are mandatory core fixtures. Beammap-supporting Pointings 148669/148671 and science-supporting Pointings 152389/152391/152393 provide legitimate Pointing-context repeatability; OOF 152385/152386/152387 provides multi-observation lifecycle/timing but not cross-focus PSF repeatability; science 152390/152392 is a long-mode timing/scan/product sentinel, not an ALIGN centroid/source-estimator gate. The suite selects but has not realized science-support observations 152418, 152420, 152430, 152432, and 152434. Their complete owner-local 240-file input corpus is available read-only and bound by manifest `{external_summary['combined_manifest_sha256']}`; only its native detector headers are used in D005, not as realized product evidence. Beammap 152307 is not selected because it is heterogeneous and outside the suite; the D002 combined-Beammap study therefore remains evidence-pending and is an explicit owner question.

The suite products identify historical Citlali builds `{', '.join(versions)}`. None is an exact whole-application `{GOVERNING_APPLICATION_SHA}` execution. The Beammap 148670 product's `cfae989c` alignment/scan source files are byte-identical to the governing versions, so its 198-scan artifact is direct governing-code-path evidence but not an exact whole-build run. Frozen phase-zero continuity selects `pointings_v22`; suite/project metadata does not independently choose a versioned pointing directory. Accepted external Point/Beammap snapshots and suite products are separately classified in `baseline_product_manifest.csv`.

## Native timing and slot conclusions

All {len(rate_rows)} distinct detector-file paths/references expose `FpgaFreq=256000000 Hz`, `AccumLen=2097152`, and `SampleFreq=122.0703125 Hz`: genuine native **1x** evidence only. They comprise {rate_coverage['realized_config_detector_file_references']} realized-config references plus {rate_coverage['owner_local_unrealized_detector_file_references']} owner-local/unrealized support references; canonicalization yields {rate_coverage['canonical_unique_observation_interface_files']} observation/interface identities across {rate_coverage['distinct_observation_count']} observations and {rate_coverage['canonical_unique_native_rows']:,} native rows. The {rate_coverage['duplicate_observation_interface_group_count']} duplicated observation/interface groups are asserted byte- and header-equivalent before canonicalization. Downstream RTC rates were not counted. Native 0.5x, 2x, and 4x evidence is absent and remains evidence-pending; 1x will not be resampled or scaled to manufacture those strata.

| Mandatory fixture | Native rows | Current supported | Union edge rows | Added grid positions | Max abs residual (ms) | Min half-cell margin (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{mandatory_lines}

Every ordinary supported row retains its slot under D002 round-half-up, with no exact half ties or collisions. The controlling limit is the strict native half cell, `abs(residual) < 4.096 ms` at 1x; measured margins do not replace that contract boundary.

All six realized low-level configs request exactly `0 s` on all 14 declared detector/HWPR offset entries. D005 applies each present detector's requested zero exactly once before slotting. The config value is bound, but the governing runtime does not persist independent realized-application evidence; the frozen phase-zero offset trace remains authoritative for that limitation. The five owner-local/unrealized supports have no config offset claim and contribute rate headers only.

All {rate_coverage['realized_supplied_header_interface_exact_matches']} realized detector references join the config-supplied `toltecN` identity exactly to the raw `Header.Toltec.RoachIndex` identity before offset lookup and slot analysis; swapped or duplicate supplied identities fail generation.

## Telescope envelope

All selected union-grid targets have finite, strictly increasing adjacent `TelTime` support. The exact fixed-cohort maximum used bracket is `{max_bracket:.18f} s`, set as an inclusive D005 validation envelope. This is not promoted to a general production cadence/gap limit; no padding was guessed. Invalid endpoints, nonpositive time, gaps, ambiguity, cross-gap support, and extrapolation fail closed.

## Scientific compatibility limits

- Point, OOF, and Beammap retain the repository's active zero-tolerance complete-product policies for unaffected records (volatile profile sidecar excluded by the existing policy). No nonzero source-crossing/centroid/PSF limit for records changed only by approved OD5 boundary repairs can be derived from one Beammap; that owner choice remains open.
- The source-crossing artifact retains distance and scan-window identities but no direct closest-approach timestamp/sample identity; absolute crossing time is unavailable and is not inferred.
- Science retains only the existing successor policy: map RMS `<=1e-8`, PTC-weight RMS `<=1e-9`, detector-median absolute/fractional `<=5e-5`/`1e-3`, and other diagnostic RMS `<=1e-7`. It is not used to tune CAL, AST, mapmaking, or source estimation.
- Pointing repeatability and fit-uncertainty summaries are descriptive. A single full Beammap and deliberately defocused OOF sequence cannot justify looser scientific thresholds.
- Single historical runtimes are not repeatability trials. The package proposes the already-established five-pair controlled design and 5% median ceiling, with owner return for five same-sign slowdowns. A setup-specific numerical margin and a storage byte ceiling remain unavailable.

## ALIGN-D004-HOLD-VALIDATION-001

Beammap 148670 has {hold_findings['grid_count']:,} governing common-grid rows and raw words `{hold_findings['raw_unique_words']}`. The independent outside-map-box condition is true on {hold_findings['outside_map_box_true_rows']:,} rows. Released/current whole-word linear-to-nonzero, left/right raw-word nonzero, and left/right bit-`0x08` hypotheses differ materially before that condition; `0x02` and `0x40` both overlap `0x08` and occur alone.

Every tested Hold-true row is already outside the map box. Consequently all tested final states, all {hold_findings['governing_kept_scan_count']} legacy output windows, and all {hold_findings['conditional_OD5_candidate_zero_run_identity_count']} maximal-false-run candidate windows are hypothesis-invariant; {hold_findings['source_crossing_distinct_recorded_scan_count']} distinct source-crossing-recorded legacy windows reproduce with zero mismatch. The governing implementation drops the first false sample after a true-to-false composite-final-state boundary in all {hold_findings['legacy_first_false_after_state_boundary_omission_count']} candidate runs, deletes {hold_findings['legacy_deleted_candidate_zero_run_count']} short/partial candidates, renumbers candidate `q={hold_findings['legacy_kept_candidate_q_min']}..{hold_findings['legacy_kept_candidate_q_max']}` to outputs `1..{hold_findings['governing_kept_scan_count']}`, and applies extra edge context trimming.

These 241 zero runs are not authoritative physical scans. Only if owner Q1 authorizes the legacy whole-word-linear-any compatibility view combined with the separately applied governing outside-map-box condition to control raster segmentation do they become the preregistered candidate identities for applying OD5's half-open, first-sample, context, and retained-status repairs. Otherwise authoritative physical raster segmentation remains unavailable. The fixture proves compatibility of the current final state and reconstructs the exact 198 legacy outputs plus the conditional candidate identities, but it cannot identify a controlling predicate or transition side. The D004 owner-return condition is met. D005 selects neither and does not silently authorize a successor scan target.

The candidate registry is expressed in detector-reference lattice slots `k`, with separate current-grid and D002 union-grid array indices. The current grid spans `k=0..383698`; the union spans `k=-1..383700`. All three added union positions are final-state true, and the 241 candidate half-open lattice intervals and digest are unchanged by support expansion.

## Owner decision brief

Recommendation: record the D005 preregistration with its measured local evidence, explicit evidence gaps, and required owner return; do not treat it as phase-one authorization.

{questions}

Until these choices are recorded, phase one remains unauthorized.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--coordination-repo", type=Path, default=Path("/private/tmp/citlali-scientific-audit-framework"))
    parser.add_argument("--suite-root", type=Path, default=Path("/Users/gwilson/work_toltec/local_data/citlali-validation/v1"))
    parser.add_argument(
        "--owner-data-root", type=Path,
        default=Path("/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data"),
    )
    parser.add_argument("--tolproj-repo", type=Path, default=Path("/Users/gwilson/GitHub/tolproj"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    coordination_repo = args.coordination_repo.resolve()
    suite_root = args.suite_root.resolve()
    owner_data_root = args.owner_data_root.resolve()
    tolproj_repo = args.tolproj_repo.resolve()
    output = (args.output or (repo / "validation" / PACKAGE_NAME)).resolve()

    identity, authority_rows = validate_identity(repo, coordination_repo, tolproj_repo)
    output.mkdir(parents=True, exist_ok=True)
    cache = DigestCache()
    realization_rows, suite_authorities = suite_realization(suite_root, cache)
    authority_rows.extend(suite_authorities)
    input_rows, config_rows, observations = load_suite_inputs(suite_root, cache)
    input_sha = {row["local_path"]: row["sha256"] for row in input_rows}
    rate_rows, slot_rows, observation_rows, grids = build_timing_evidence(observations, input_sha)
    external_manifest, external_rate_rows, external_summary = external_science_support_evidence(
        owner_data_root, cache,
    )
    realized_rate_row_count = len(rate_rows)
    rate_rows.extend(external_rate_rows)
    telescope_rows = [
        telescope_bracket_row(obs, grids[key]["union_grid"])
        for key, obs in sorted(observations.items())
    ]
    products, product_paths = baseline_products(suite_root, cache, repo)
    historical_scan_rows = historical_scan_product_rows(products)
    beam_key = next(key for key in observations if key == ("beammap_core", 148670))
    (
        hold_comparison, hold_windows, hold_candidate_rows, bit_summary,
        bit_transitions, hold_findings,
    ) = hold_analysis(
        observations[beam_key], grids[beam_key], product_paths["beam_source_crossing"],
    )

    fit_rows: list[dict[str, Any]] = []
    scope_by_cohort = {
        "point_core": "mandatory exact successor sentinel; duplicated contexts are not repeat runs",
        "oof_context": "same-observation exact sentinel only; cross-focus repeatability prohibited",
        "beammap_support": "legitimate bracketing Pointing repeatability context",
        "science_support": "legitimate realized Pointing repeatability context",
    }
    cohort_names = {
        "point_core": "point_core_152389",
        "oof_context": "oof_deliberate_focus_sequence",
        "beammap_support": "beammap_bracketing_pointings",
        "science_support": "science_realized_support_pointings",
    }
    for cohort, path in product_paths["pointing_paths"]:  # type: ignore[index]
        fit_rows.extend(pointing_rows(path, cohort_names[cohort], scope_by_cohort[cohort]))
    repeat_rows = repeatability_rows(fit_rows)
    beam_summary = beammap_science_summary(product_paths["beam_apt"], product_paths["beam_source_crossing"])
    if [
        row["valid_intersection_count"] for row in beam_summary["fit_summary"]
    ] != [2901, 1186, 886]:
        raise RuntimeError(f"Beammap valid fit cohort identity changed: {beam_summary}")
    runtime = runtime_evidence(products)
    protocol = preregistration_protocol(observation_rows)
    brief = owner_brief(hold_findings)

    rate_counts = Counter(str(row["native_rate_factor"]) for row in rate_rows)
    rate_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rate_rows:
        rate_groups[(row["obsnum"], row["interface"])].append(row)
    duplicate_groups = {
        key: rows for key, rows in rate_groups.items() if len(rows) > 1
    }
    canonical_identity_fields = (
        "raw_sha256", "supplied_interface", "header_interface", "interface_identity_status",
        "header_obsnum", "subobsnum", "scannum", "fpga_freq_hz",
        "accum_len", "sample_freq_hz", "native_rate_factor", "cadence_sec",
        "exclusive_half_cell_sec", "native_rows", "native_first_sec", "native_last_sec",
        "native_duration_sec", "requested_interface_offset_sec", "header_acquisition_start",
        "header_acquisition_end", "packet_gap_count_current_test",
        "clock_tick_step_min_delta", "clock_tick_step_max_delta",
    )
    for key, rows in duplicate_groups.items():
        identities = {
            tuple(row[field] for field in canonical_identity_fields) for row in rows
        }
        if len(identities) != 1:
            raise RuntimeError(
                f"duplicate observation/interface inputs are not byte/header equivalent: {key}"
            )
    canonical_rate_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in sorted(
        rate_rows,
        key=lambda item: (
            item["obsnum"], item["interface"],
            0 if item["config_id"] == "point_core" else 1,
            item["config_id"], item["raw_path"],
        ),
    ):
        canonical_rate_rows.setdefault((row["obsnum"], row["interface"]), row)
    realized_canonical_rate_rows = {
        key: row for key, row in canonical_rate_rows.items()
        if row["config_id"] != "external_science_support_unrealized"
    }
    rate_coverage = {
        "native_detector_file_count": len(rate_rows),
        "realized_config_detector_file_references": realized_rate_row_count,
        "realized_supplied_header_interface_exact_matches": sum(
            row["interface_identity_status"] == "exact_match"
            for row in rate_rows
            if row["config_id"] != "external_science_support_unrealized"
        ),
        "owner_local_unrealized_detector_file_references": len(external_rate_rows),
        "distinct_file_path_count": len({row["raw_path"] for row in rate_rows}),
        "canonical_unique_observation_interface_files": len(canonical_rate_rows),
        "duplicate_observation_interface_group_count": len(duplicate_groups),
        "duplicate_observation_interface_reference_count": sum(
            len(rows) for rows in duplicate_groups.values()
        ),
        "duplicate_canonicalization_identity": (
            "byte-identical SHA-256 plus exact native header/row/timing/offset tuple"
        ),
        "distinct_observation_count": len({row["obsnum"] for row in rate_rows}),
        "canonical_unique_native_rows": sum(
            row["native_rows"] for row in canonical_rate_rows.values()
        ),
        "realized_canonical_unique_native_rows": sum(
            row["native_rows"] for row in realized_canonical_rate_rows.values()
        ),
        "owner_local_unrealized_native_rows": sum(
            row["native_rows"] for row in external_rate_rows
        ),
        "counts_by_native_factor": dict(sorted(rate_counts.items())),
        "profiles": [
            {
                "factor": factor, **profile,
                "exclusive_half_cell_sec": profile["dt_sec"] / 2.0,
                "local_observational_evidence": "available" if factor == 1.0 else "evidence_pending",
                "precise_missing_fixture": (
                    "none" if factor == 1.0 else
                    f"owner-approved native Pointing or Beammap raw fixture with SampleFreq={profile['sample_rate_hz']} Hz and AccumLen={profile['accum_len']}"
                ),
            }
            for factor, profile in EXPECTED_RATE_PROFILES.items()
        ],
        "downstream_RTC_rate_counts_as_native": False,
        "resampling_permitted_to_fill_stratum": False,
    }

    max_bracket = max(row["used_bracket_max_sec"] for row in telescope_rows)
    if (
        len(rate_rows) != 187 or rate_coverage["distinct_file_path_count"] != 187
        or realized_rate_row_count != 132
        or rate_coverage["realized_supplied_header_interface_exact_matches"] != 132
        or len(external_rate_rows) != 55 or len(canonical_rate_rows) != 176
        or len(duplicate_groups) != 11
        or rate_coverage["canonical_unique_native_rows"] != 8673204
        or rate_coverage["realized_canonical_unique_native_rows"] != 8243629
        or rate_coverage["owner_local_unrealized_native_rows"] != 429575
        or set(rate_counts) != {"1.0"}
    ):
        raise RuntimeError(f"unexpected native-rate coverage: rows={len(rate_rows)} {rate_counts}")
    if len(observation_rows) != 12:
        raise RuntimeError(f"unexpected selected observation/config count: {len(observation_rows)}")
    if any(row["requested_nonzero_offset_count"] != 0 for row in config_rows):
        raise RuntimeError("D005 realized suite unexpectedly requests a nonzero interface offset")
    if max_bracket != TELESCOPE_BRACKET_LIMIT_SEC:
        raise RuntimeError(f"fixed-cohort telescope maximum changed: {max_bracket!r}")
    mandatory = {row["obsnum"]: row for row in observation_rows if row["config_id"] in {"point_core", "beammap_core"}}
    if not (
        mandatory[152389]["union_edge_only_native_rows"] == 22
        and mandatory[148670]["union_edge_only_native_rows"] == 16
        and mandatory[152389]["union_added_grid_positions"] == 3
        and mandatory[148670]["union_added_grid_positions"] == 3
        and mandatory[152389]["ordinary_changed_slots"] == 0
        and mandatory[148670]["ordinary_changed_slots"] == 0
    ):
        raise RuntimeError(f"mandatory slot/support identity changed: {mandatory}")
    if not (
        hold_findings["grid_count"] == 383699
        and hold_findings["union_grid_count"] == 383702
        and hold_findings["union_grid_lattice_slot_k_min"] == -1
        and hold_findings["union_grid_lattice_slot_k_max"] == 383700
        and hold_findings["union_added_lattice_slots"] == [-1, 383699, 383700]
        and hold_findings["union_added_positions_all_final_true"] is True
        and hold_findings["union_support_preserves_candidate_lattice_windows"] is True
        and hold_findings["raw_unique_words"] == [0, 2, 8, 10, 64, 66, 72, 74]
        and hold_findings["governing_kept_scan_count"] == 198
        and hold_findings["conditional_OD5_candidate_zero_run_identity_count"] == 241
        and hold_findings["legacy_deleted_candidate_zero_run_count"] == 43
        and hold_findings["legacy_first_false_after_state_boundary_omission_count"] == 241
        and hold_findings["legacy_kept_candidate_q_min"] == 34
        and hold_findings["legacy_kept_candidate_q_max"] == 231
        and hold_findings["all_final_hypotheses_identical"] is True
        and hold_findings["all_union_final_hypotheses_identical"] is True
        and hold_findings["all_union_candidate_lattice_windows_identical"] is True
        and hold_findings["source_crossing_window_mismatch_count"] == 0
    ):
        raise RuntimeError(f"Hold validation identity changed: {hold_findings}")
    candidate_by_q = {
        row["candidate_zero_run_q_0based"]: row for row in hold_candidate_rows
    }
    expected_candidate_boundaries = {
        0: (767, 769),
        34: (4241, 4993),
        231: (380386, 381151),
        240: (383262, 383399),
    }
    if any(
        (
            candidate_by_q[q]["conditional_lattice_slot_k_start"],
            candidate_by_q[q]["conditional_lattice_slot_k_stop_exclusive"],
        ) != expected
        for q, expected in expected_candidate_boundaries.items()
    ):
        raise RuntimeError("conditional zero-run candidate boundary identity changed")
    if any(
        (
            candidate_by_q[q]["current_grid_array_index_start"],
            candidate_by_q[q]["current_grid_array_index_stop_exclusive"],
            candidate_by_q[q]["union_grid_array_index_start"],
            candidate_by_q[q]["union_grid_array_index_stop_exclusive"],
        ) != (expected[0], expected[1], expected[0] + 1, expected[1] + 1)
        for q, expected in expected_candidate_boundaries.items()
    ):
        raise RuntimeError("conditional current/union array-index identity changed")

    identity["application_path_delta"] = []
    identity["selected_suite_root"] = str(suite_root)
    identity["selected_config_count"] = len(config_rows)
    identity["selected_observation_context_count"] = len(observation_rows)
    identity["selected_detector_file_count"] = len(rate_rows)
    identity["realized_config_detector_file_references"] = realized_rate_row_count
    identity["owner_local_unrealized_detector_file_references"] = len(external_rate_rows)
    identity["owner_data_root_read_only"] = str(owner_data_root)

    write_json(output / "authority_and_identity.json", identity)
    write_csv(output / "authority_manifest.csv", [
        "authority_id", "authority_class", "path", "git_commit", "sha256", "use",
    ], authority_rows)
    write_csv(output / "suite_selection_realization.csv", [
        "project", "obsnum", "fixture_role", "selected_by_suite",
        "realized_in_project_and_selected_config", "realization_status", "D005_use",
    ], realization_rows)
    write_csv(output / "selected_config_manifest.csv", [
        "config_id", "mode", "role", "path", "size_bytes", "sha256", "evidence_class",
        "requested_interface_offsets_sec_json", "requested_nonzero_offset_count",
    ], config_rows)
    write_csv(output / "selected_input_manifest.csv", [
        "config_id", "mode", "fixture_role", "obsnum", "item_class", "interface",
        "requested_unity_path", "local_path", "size_bytes", "sha256",
        "selection_authority", "realization_authority", "availability",
    ], input_rows)
    write_csv(output / "external_science_support_manifest.csv", [
        "obsnum", "selection_authority", "realized_suite_availability",
        "owner_local_availability", "asset_class", "relative_path", "local_path",
        "size_bytes", "sha256", "D005_use", "future_human_run_status",
    ], external_manifest)
    write_json(output / "external_science_support_summary.json", external_summary)
    write_csv(output / "native_rate_inventory.csv", list(rate_rows[0]), rate_rows)
    write_json(output / "rate_coverage.json", rate_coverage)
    write_csv(output / "slot_residual_metrics.csv", list(slot_rows[0]), slot_rows)
    write_csv(output / "slot_observation_summary.csv", list(observation_rows[0]), observation_rows)
    write_csv(output / "telescope_bracket_metrics.csv", list(telescope_rows[0]), telescope_rows)
    write_csv(output / "baseline_product_manifest.csv", list(products[0]), products)
    write_csv(
        output / "historical_scan_product_summary.csv",
        list(historical_scan_rows[0]), historical_scan_rows,
    )
    write_csv(output / "pointing_fit_metrics.csv", list(fit_rows[0]), fit_rows)
    write_csv(output / "pointing_repeatability_summary.csv", list(repeat_rows[0]), repeat_rows)
    write_json(output / "beammap_source_fit_summary.json", beam_summary)
    write_json(output / "runtime_io_storage_evidence.json", runtime)
    write_csv(output / "hold_hypothesis_comparison.csv", list(hold_comparison[0]), hold_comparison)
    write_csv(output / "hold_scan_windows.csv", list(hold_windows[0]), hold_windows)
    write_csv(
        output / "hold_zero_run_candidate_registry.csv",
        list(hold_candidate_rows[0]), hold_candidate_rows,
    )
    write_csv(output / "hold_bit_summary.csv", list(bit_summary[0]), bit_summary)
    write_csv(output / "hold_raw_bit_transitions.csv", list(bit_transitions[0]), bit_transitions)
    write_json(output / "hold_findings.json", hold_findings)
    write_json(output / "preregistration_protocol.json", protocol)
    write_json(output / "owner_decision_brief.json", brief)
    (output / "REPORT.md").write_text(report_text(
        identity, rate_rows, observation_rows, telescope_rows, hold_findings, brief,
        products, rate_coverage, external_summary,
    ))

    artifact_names = sorted({
        "authority_and_identity.json", "authority_manifest.csv",
        "suite_selection_realization.csv", "selected_config_manifest.csv",
        "selected_input_manifest.csv", "external_science_support_manifest.csv",
        "external_science_support_summary.json", "native_rate_inventory.csv", "rate_coverage.json",
        "slot_residual_metrics.csv", "slot_observation_summary.csv",
        "telescope_bracket_metrics.csv", "baseline_product_manifest.csv",
        "historical_scan_product_summary.csv",
        "pointing_fit_metrics.csv", "pointing_repeatability_summary.csv",
        "beammap_source_fit_summary.json", "runtime_io_storage_evidence.json",
        "hold_hypothesis_comparison.csv", "hold_scan_windows.csv",
        "hold_zero_run_candidate_registry.csv", "hold_bit_summary.csv",
        "hold_raw_bit_transitions.csv", "hold_findings.json", "preregistration_protocol.json",
        "owner_decision_brief.json", "REPORT.md",
    })
    expected = set(artifact_names) | {"SHA256SUMS"}
    unexpected = sorted(path.name for path in output.iterdir() if path.is_file() and path.name not in expected)
    if unexpected:
        raise RuntimeError(f"unexpected preexisting output artifacts: {unexpected}")
    sums = [f"{sha256_file(output / name)}  {name}" for name in artifact_names]
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
