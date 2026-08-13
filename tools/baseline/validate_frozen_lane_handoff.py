#!/usr/bin/env python3
"""Validate an exact-SHA frozen-lane handoff packet without running its gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "citlali-frozen-lane-handoff-packet-v1"
FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REF_RE = re.compile(r"^refs/(?:heads|remotes|tags)/[A-Za-z0-9._/-]+$")
PLACEHOLDER_RE = re.compile(r"\b(?:TBD|TODO|FIXME)\b", re.IGNORECASE)

PACKET_KINDS = {"cal_lane", "align_lane", "combined"}
LIFECYCLE_STATES = {
    "preparing",
    "frozen",
    "returned_for_repair",
    "blocked",
    "authority_decision_required",
}
TARGET_STAGES = {"lane_handoff", "combined_acceptance"}
MODES = {"point", "oof", "science", "beammap"}
GOVERNED_CHANGE_KINDS = {
    "structural",
    "provenance",
    "algorithm",
    "numerical",
    "default",
    "schema",
    "scientific_product",
}
SCIENCE_DECLARATION_REQUIRED = {
    "algorithm",
    "numerical",
    "default",
    "schema",
    "scientific_product",
}
HISTORY_CATEGORIES = {
    "application",
    "test",
    "validation",
    "documentation",
    "generated_input",
    "generated_evidence",
    "audit",
    "failed_repair",
    "coordination",
    "contaminating_dependency",
}
FORBIDDEN_APPLICATION_CATEGORIES = {
    "audit",
    "failed_repair",
    "coordination",
    "contaminating_dependency",
}
IMPORT_DISPOSITIONS = {
    "include_application",
    "exclude_non_application",
    "reconstruct_required",
}
EXCLUDED_CATEGORIES = {
    "audit",
    "failed_repair",
    "abandoned",
    "generated_evidence",
    "coordination",
    "contaminating_dependency",
}
DEPENDENCY_CLASSIFICATIONS = {
    "base_present",
    "lane_local",
    "cross_lane",
    "separately_promoted",
    "external_blocked",
    "contaminating",
}
DEPENDENCY_DISPOSITIONS = {
    "imported",
    "reconstructed",
    "independently_required",
    "excluded",
    "deferred",
}
INTERFACE_CLASSIFICATIONS = {
    "additive",
    "textual_conflict",
    "semantic_conflict",
    "coupled",
}
FINDING_STATES = {"closed", "accepted", "conditioned", "open", "blocked"}
BLOCKING_STAGES = {
    "lane_handoff",
    "intermediate_freeze",
    "combined_freeze",
    "combined_acceptance",
    "mainline_promotion",
    "refactor_apt_generation",
    "refactor_apt_activation",
    "production_end_to_end",
}
STAGE_ORDER = {
    "lane_handoff": 0,
    "intermediate_freeze": 1,
    "combined_freeze": 2,
    "combined_acceptance": 3,
    "mainline_promotion": 4,
    "refactor_apt_generation": 5,
    "refactor_apt_activation": 6,
    "production_end_to_end": 7,
}
GATE_SCOPES = {"cal", "align", "overlap", "combined", "external"}
GATE_TIMINGS = {
    "lane_freeze",
    "order_selection",
    "post_first_import",
    "intermediate_freeze",
    "post_second_import",
    "combined_freeze",
    "human_unity",
    "pre_acceptance",
    "post_software_apt_generation",
    "preproduction_sample_generation",
    "human_unity_beam_campaign",
    "refactor_library_curation",
    "shadow_cross_generation_comparison",
    "refactor_generation_activation",
    "post_activation_regeneration",
}
GATE_RESULTS = {
    "not_run",
    "pass",
    "fail",
    "blocked",
    "conditioned",
    "not_applicable",
    "omitted",
}
ACTION_KINDS = {"local_command", "human_mediated_unity", "document_review"}
ARTIFACT_LOCATIONS = {
    "repository_blob",
    "local_artifact",
    "human_supplied_external",
}

CORE_GATES = {
    "AUTH-BASE-001",
    "AUTH-PREREQ-001",
    "AUTH-ANCESTRY-001",
    "AUTH-DISPOSITION-001",
    "IDENT-FREEZE-001",
    "ARCH-BOUNDARY-001",
    "BUILD-CLI-001",
    "TEST-CTEST-001",
    "CONFIG-PREFLIGHT-001",
    "VALIDATION-TOOLS-001",
    "PRODUCT-CONTRACT-001",
    "PROVENANCE-APT-LINEAGE-001",
    "REQUIRED-OUTPUTS-001",
    "FAILURE-LOG-001",
    "SCIENCE-EPOCH-001",
    "MODE-ROUTING-001",
    "LOCAL-SAME-SHA-001",
    "EXTERNAL-APT-001",
    "PACKET-CONFORMANCE-001",
}
CAL_GATES = {
    "CAL-FITS-CLOSE-001",
    "CAL-APT-LINEAGE-001",
    "CAL-FOCUS-001",
}
ALIGN_GATES = {
    "ALIGN-FOUNDATION-001",
    "ALIGN-CONSUMER-001",
    "ALIGN-PCA-001",
    "ALIGN-LIFECYCLE-001",
}
OVERLAP_GATES = {
    "OVERLAP-RTC-001",
    "OVERLAP-APT-001",
    "OVERLAP-PROVENANCE-001",
    "COMBINED-REGRESSION-001",
}
MODE_GATES = {
    "point": "MODE-POINT-001",
    "oof": "MODE-OOF-001",
    "science": "MODE-SCIENCE-001",
    "beammap": "MODE-BEAMMAP-001",
}
UNITY_GATE = "UNITY-SAME-SHA-001"
APT_INTERFACE_GATES = {
    "APT-A-RAW-KMP-CITLALI-AXIS-001": ("raw_kmp", "citlali"),
    "APT-B-CITLALI-BEAMMAP-EXPORT-001": ("citlali", "toltec_beammap"),
    "APT-C-BEAMMAP-MATCHING-001": ("toltec_beammap", "tolapt_or_tolproj"),
    "APT-D-TOLAPT-TOLPROJ-PACKAGE-001": ("tolapt", "tolproj"),
    "APT-E-TOLPROJ-TOLTECA-SELECTION-001": ("tolproj", "tolteca"),
    "APT-F-TOLTECA-CITLALI-TRANSPORT-001": ("tolteca", "citlali"),
    "APT-G-CITLALI-ADMISSION-001": ("selected_apt_artifact", "citlali"),
}
APT_ROUTE_ENDPOINTS = {
    "APT-A-RAW-KMP-CITLALI-AXIS-001": {("raw_kmp", "citlali")},
    "APT-B-CITLALI-BEAMMAP-EXPORT-001": {("citlali", "toltec_beammap")},
    "APT-C-BEAMMAP-MATCHING-001": {
        ("toltec_beammap", "tolapt"),
        ("toltec_beammap", "tolproj"),
    },
    "APT-D-TOLAPT-TOLPROJ-PACKAGE-001": {
        ("tolapt", "tolproj"),
        ("tolproj", "tolapt"),
    },
    "APT-E-TOLPROJ-TOLTECA-SELECTION-001": {("tolproj", "tolteca")},
    "APT-F-TOLTECA-CITLALI-TRANSPORT-001": {("tolteca", "citlali")},
    "APT-G-CITLALI-ADMISSION-001": {("selected_apt_artifact", "citlali")},
}
APT_LIBRARY_GENERATION_GATES = {
    "APT-LIB-SOFTWARE-FREEZE-001": (
        "post_software_apt_generation",
        "refactor_apt_generation",
    ),
    "APT-LIB-COHORT-MANIFEST-001": (
        "post_software_apt_generation",
        "refactor_apt_generation",
    ),
    "APT-SAMPLE-NEW-CONTRACT-FIXTURES-001": (
        "preproduction_sample_generation",
        "refactor_apt_generation",
    ),
    "APT-LIB-BEAM-CAMPAIGN-001": (
        "human_unity_beam_campaign",
        "refactor_apt_generation",
    ),
    "APT-LIB-CANDIDATE-CONFORMANCE-001": (
        "refactor_library_curation",
        "refactor_apt_generation",
    ),
    "APT-LIB-IMMUTABLE-GENERATION-001": (
        "refactor_library_curation",
        "refactor_apt_generation",
    ),
    "APT-LIB-COMPLETENESS-QUARANTINE-001": (
        "refactor_library_curation",
        "refactor_apt_generation",
    ),
    "APT-LIB-PROVENANCE-001": (
        "refactor_library_curation",
        "refactor_apt_generation",
    ),
    "APT-LIB-NO-MIXED-LINEAGE-001": (
        "refactor_library_curation",
        "refactor_apt_activation",
    ),
    "APT-LIB-HISTORICAL-IMMUTABILITY-001": (
        "refactor_library_curation",
        "refactor_apt_activation",
    ),
    "APT-LIB-SHADOW-COMPARISON-001": (
        "shadow_cross_generation_comparison",
        "refactor_apt_activation",
    ),
    "APT-LIB-ACTIVATION-ROLLBACK-001": (
        "refactor_generation_activation",
        "refactor_apt_activation",
    ),
    "APT-LIB-SELECTED-CONTRACT-001": (
        "refactor_generation_activation",
        "refactor_apt_activation",
    ),
    "APT-REFACTOR-REDUCTIONS-001": (
        "post_activation_regeneration",
        "production_end_to_end",
    ),
}
MANDATORY_RECORDED_GATES = set(APT_INTERFACE_GATES) | set(APT_LIBRARY_GENERATION_GATES)
APT_READINESS_STATES = {
    "not_run",
    "pass",
    "fail",
    "blocked",
    "conditioned",
    "not_applicable",
}
TOLAPT_ROLES = {
    "offline_downstream",
    "offline_package_exchange",
    "precomputed_input",
    "not_in_path",
}
APT_PHASE_READINESS_STATES = {
    "not_run",
    "pass",
    "fail",
    "blocked",
    "conditioned",
    "not_applicable",
}
APT_PHASE_DIGEST_FIELDS = {
    "software_revision_set_sha256",
    "config_manifest_sha256",
    "raw_data_manifest_sha256",
    "cohort_manifest_sha256",
    "artifact_manifest_sha256",
    "component_manifest_sha256",
    "membership_sha256",
    "mapping_sha256",
    "transformation_sha256",
    "application_sha256",
    "quarantine_manifest_sha256",
    "rollback_manifest_sha256",
}
ALLOWED_CURRENT_APT_EVIDENCE_CLASSES = {
    "source_static",
    "unit",
    "synthetic_counterexample",
    "schema_contract",
    "historical_cross_generation_comparison",
}


class PacketError(ValueError):
    pass


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PacketError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PacketError(f"non-finite JSON number {value!r} is forbidden")


def load_packet(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_duplicate_object,
                parse_constant=_reject_constant,
            )
    except PacketError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PacketError(str(error)) from error
    packet = _mapping(value, str(path))
    _reject_placeholders(packet, str(path))
    return packet


def _reject_placeholders(value: Any, context: str) -> None:
    if isinstance(value, str) and PLACEHOLDER_RE.search(value):
        raise PacketError(f"{context}: unresolved placeholder in {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_placeholders(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{context}[{index}]")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PacketError(f"{context}: expected object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PacketError(f"{context}: expected list")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise PacketError(f"{context}: missing required fields {missing}")
    if unknown:
        raise PacketError(f"{context}: unknown fields {unknown}")


def _text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "string" if allow_empty else "non-empty string"
        raise PacketError(f"{context}: expected {qualifier}")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise PacketError(f"{context}: expected boolean")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PacketError(f"{context}: expected integer")
    if minimum is not None and value < minimum:
        raise PacketError(f"{context}: expected integer >= {minimum}")
    return value


def _enum(value: Any, allowed: set[str], context: str) -> str:
    text = _text(value, context)
    if text not in allowed:
        raise PacketError(f"{context}: unsupported value {text!r}")
    return text


def _sha(value: Any, context: str) -> str:
    text = _text(value, context)
    if not FULL_GIT_SHA_RE.fullmatch(text):
        raise PacketError(f"{context}: expected lowercase full 40-character Git SHA")
    return text


def _nullable_sha(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _sha(value, context)


def _sha256(value: Any, context: str) -> str:
    text = _text(value, context)
    if not SHA256_RE.fullmatch(text):
        raise PacketError(f"{context}: expected lowercase SHA-256")
    return text


def _nullable_sha256(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


def _timestamp(value: Any, context: str) -> str:
    text = _text(value, context)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PacketError(f"{context}: invalid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise PacketError(f"{context}: timestamp must include timezone")
    return text


def _nullable_timestamp(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _timestamp(value, context)


def _string_list(
    value: Any,
    context: str,
    *,
    allowed: set[str] | None = None,
    nonempty: bool = False,
) -> list[str]:
    items = _list(value, context)
    if nonempty and not items:
        raise PacketError(f"{context}: expected non-empty list")
    result = [_text(item, f"{context}[{index}]") for index, item in enumerate(items)]
    if len(result) != len(set(result)):
        raise PacketError(f"{context}: duplicate value")
    if allowed is not None:
        unknown = sorted(set(result) - allowed)
        if unknown:
            raise PacketError(f"{context}: unsupported values {unknown}")
    return result


def _repo_path(value: Any, context: str) -> str:
    text = _text(value, context)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix() or text == ".":
        raise PacketError(f"{context}: expected normalized repository-relative path")
    return text


def _ref_name(value: Any, context: str) -> str:
    text = _text(value, context)
    if (
        not REF_RE.fullmatch(text)
        or ".." in text
        or "@{" in text
        or "//" in text
        or text.endswith(("/", ".lock"))
    ):
        raise PacketError(f"{context}: expected safe full refs/heads, refs/remotes, or refs/tags name")
    return text


def _git(
    repo_root: Path,
    arguments: list[str],
    *,
    allowed_returncodes: set[int] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    allowed = {0} if allowed_returncodes is None else allowed_returncodes
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    if result.returncode not in allowed:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise PacketError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def _git_text(repo_root: Path, arguments: list[str]) -> str:
    return _git(repo_root, arguments).stdout.decode().strip()


def _require_commit(repo_root: Path, commit: str, context: str) -> None:
    _sha(commit, context)
    _git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])


def _commit_facts(repo_root: Path, commit: str, context: str) -> tuple[list[str], str]:
    _require_commit(repo_root, commit, context)
    output = _git(
        repo_root,
        ["show", "-s", "--format=%P%x00%T%x00", commit],
    ).stdout
    fields = output.removesuffix(b"\n").split(b"\x00")
    if len(fields) != 3 or fields[2]:
        raise PacketError(f"{context}: unable to read parent/tree facts")
    try:
        parent_field = fields[0].decode("ascii")
        tree = fields[1].decode("ascii")
    except UnicodeDecodeError as error:
        raise PacketError(f"{context}: unable to decode parent/tree facts") from error
    parents = parent_field.split() if parent_field else []
    return parents, tree


def _resolve_ref(repo_root: Path, ref: str, context: str) -> str:
    _ref_name(ref, context)
    return _git_text(repo_root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _git(
        repo_root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        allowed_returncodes={0, 1},
    )
    return result.returncode == 0


def _tree_blob(repo_root: Path, commit: str, path: str) -> str | None:
    output = _git(repo_root, ["ls-tree", "-z", commit, "--", path]).stdout
    if not output:
        return None
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1:
        raise PacketError(f"{commit}:{path}: expected exactly one tree entry")
    metadata, actual_path = records[0].split(b"\t", 1)
    if actual_path.decode() != path:
        raise PacketError(f"{commit}:{path}: tree returned different path")
    fields = metadata.decode().split()
    if len(fields) != 3 or fields[1] not in {"blob", "commit"}:
        raise PacketError(f"{commit}:{path}: unsupported tree entry")
    return fields[2]


def _blob_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    return _git(repo_root, ["show", f"{commit}:{path}"]).stdout


def _blob_sha256(repo_root: Path, commit: str, path: str) -> str:
    return hashlib.sha256(_blob_bytes(repo_root, commit, path)).hexdigest()


def _diff_name_status(repo_root: Path, base: str, candidate: str) -> tuple[bytes, list[tuple[str, str]]]:
    output = _git(
        repo_root,
        ["diff", "--name-status", "--no-renames", "-z", base, candidate, "--"],
    ).stdout
    values = output.split(b"\0")
    if values and values[-1] == b"":
        values.pop()
    if len(values) % 2:
        raise PacketError("Git name-status output has incomplete record")
    records: list[tuple[str, str]] = []
    for index in range(0, len(values), 2):
        status = values[index].decode()
        path = values[index + 1].decode()
        if status not in {"A", "M", "D", "T"}:
            raise PacketError(f"unsupported Git name-status {status!r} for {path!r}")
        records.append((status, path))
    return output, records


def _validate_packet_identity(packet: dict[str, Any]) -> tuple[str, str, str]:
    identity = _mapping(packet["packet_identity"], "packet_identity")
    _exact_keys(
        identity,
        {
            "packet_id",
            "lane_id",
            "packet_kind",
            "recorded_at",
            "lifecycle_state",
            "target_stage",
        },
        "packet_identity",
    )
    _text(identity["packet_id"], "packet_identity.packet_id")
    _text(identity["lane_id"], "packet_identity.lane_id")
    packet_kind = _enum(identity["packet_kind"], PACKET_KINDS, "packet_identity.packet_kind")
    _timestamp(identity["recorded_at"], "packet_identity.recorded_at")
    lifecycle = _enum(
        identity["lifecycle_state"], LIFECYCLE_STATES, "packet_identity.lifecycle_state"
    )
    target = _enum(identity["target_stage"], TARGET_STAGES, "packet_identity.target_stage")
    if packet_kind == "combined" and target != "combined_acceptance":
        raise PacketError("packet_identity.target_stage: combined packet requires combined_acceptance")
    if packet_kind != "combined" and target != "lane_handoff":
        raise PacketError("packet_identity.target_stage: lane packet requires lane_handoff")
    return packet_kind, lifecycle, target


def _validate_candidate(
    packet: dict[str, Any], repo_root: Path
) -> tuple[dict[str, Any], str, str, str]:
    candidate = _mapping(packet["implementation_candidate"], "implementation_candidate")
    _exact_keys(
        candidate,
        {
            "source_ref",
            "snapshot_started_at",
            "snapshot_finished_at",
            "start_tip_sha",
            "end_tip_sha",
            "commit_sha",
            "parent_shas",
            "tree_sha",
            "authorized_base_sha",
            "authorized_base_tree",
            "merge_base_sha",
            "ahead_count",
            "behind_count",
            "standard_binary_patch_sha256",
            "name_status_sha256",
            "embedded_version",
            "implementation_frozen",
            "worktree_clean",
        },
        "implementation_candidate",
    )
    source_ref = _ref_name(candidate["source_ref"], "implementation_candidate.source_ref")
    _timestamp(candidate["snapshot_started_at"], "implementation_candidate.snapshot_started_at")
    _timestamp(candidate["snapshot_finished_at"], "implementation_candidate.snapshot_finished_at")
    start_tip = _sha(candidate["start_tip_sha"], "implementation_candidate.start_tip_sha")
    end_tip = _sha(candidate["end_tip_sha"], "implementation_candidate.end_tip_sha")
    commit = _sha(candidate["commit_sha"], "implementation_candidate.commit_sha")
    parents = [
        _sha(value, f"implementation_candidate.parent_shas[{index}]")
        for index, value in enumerate(_list(candidate["parent_shas"], "implementation_candidate.parent_shas"))
    ]
    if len(parents) != len(set(parents)):
        raise PacketError("implementation_candidate.parent_shas: duplicate parent")
    tree = _sha(candidate["tree_sha"], "implementation_candidate.tree_sha")
    base = _sha(candidate["authorized_base_sha"], "implementation_candidate.authorized_base_sha")
    base_tree = _sha(
        candidate["authorized_base_tree"], "implementation_candidate.authorized_base_tree"
    )
    merge_base = _sha(candidate["merge_base_sha"], "implementation_candidate.merge_base_sha")
    ahead = _integer(candidate["ahead_count"], "implementation_candidate.ahead_count", minimum=0)
    behind = _integer(candidate["behind_count"], "implementation_candidate.behind_count", minimum=0)
    patch_digest = _sha256(
        candidate["standard_binary_patch_sha256"],
        "implementation_candidate.standard_binary_patch_sha256",
    )
    name_digest = _sha256(
        candidate["name_status_sha256"], "implementation_candidate.name_status_sha256"
    )
    _text(candidate["embedded_version"], "implementation_candidate.embedded_version")
    frozen = _boolean(candidate["implementation_frozen"], "implementation_candidate.implementation_frozen")
    _boolean(candidate["worktree_clean"], "implementation_candidate.worktree_clean")

    actual_parents, actual_tree = _commit_facts(repo_root, commit, "implementation_candidate.commit_sha")
    if parents != actual_parents:
        raise PacketError(
            f"implementation_candidate.parent_shas: recorded {parents}, Git has {actual_parents}"
        )
    if tree != actual_tree:
        raise PacketError(f"implementation_candidate.tree_sha: expected {actual_tree}")
    _, actual_base_tree = _commit_facts(repo_root, base, "implementation_candidate.authorized_base_sha")
    if base_tree != actual_base_tree:
        raise PacketError(f"implementation_candidate.authorized_base_tree: expected {actual_base_tree}")
    actual_merge_base = _git_text(repo_root, ["merge-base", base, commit])
    if merge_base != actual_merge_base:
        raise PacketError(f"implementation_candidate.merge_base_sha: expected {actual_merge_base}")
    counts = _git_text(repo_root, ["rev-list", "--left-right", "--count", f"{base}...{commit}"]).split()
    if len(counts) != 2:
        raise PacketError("unable to read candidate divergence")
    actual_behind, actual_ahead = (int(value) for value in counts)
    if (behind, ahead) != (actual_behind, actual_ahead):
        raise PacketError(
            "implementation_candidate divergence: "
            f"recorded behind/ahead={(behind, ahead)}, Git has {(actual_behind, actual_ahead)}"
        )
    if commit != end_tip:
        raise PacketError("implementation_candidate.commit_sha must equal end_tip_sha")
    if frozen and start_tip != end_tip:
        raise PacketError("implementation_candidate: frozen tip moved during snapshot")
    resolved = _resolve_ref(repo_root, source_ref, "implementation_candidate.source_ref")
    if resolved != commit:
        raise PacketError(
            f"implementation_candidate.source_ref resolves to {resolved}, not candidate {commit}"
        )
    patch = _git(repo_root, ["diff", "--binary", base, commit, "--"]).stdout
    actual_patch_digest = hashlib.sha256(patch).hexdigest()
    if patch_digest != actual_patch_digest:
        raise PacketError(
            "implementation_candidate.standard_binary_patch_sha256: "
            f"expected {actual_patch_digest}"
        )
    name_status, _ = _diff_name_status(repo_root, base, commit)
    actual_name_digest = hashlib.sha256(name_status).hexdigest()
    if name_digest != actual_name_digest:
        raise PacketError(
            f"implementation_candidate.name_status_sha256: expected {actual_name_digest}"
        )
    return candidate, commit, tree, base


def _validate_packet_container(packet: dict[str, Any], repo_root: Path, candidate: str) -> None:
    container = _mapping(packet["packet_container"], "packet_container")
    _exact_keys(
        container,
        {"kind", "commit_sha", "tree_sha", "separate_from_implementation"},
        "packet_container",
    )
    kind = _enum(
        container["kind"], {"uncommitted_packet", "documentation_only_commit"}, "packet_container.kind"
    )
    separate = _boolean(
        container["separate_from_implementation"], "packet_container.separate_from_implementation"
    )
    if not separate:
        raise PacketError("packet_container.separate_from_implementation must be true")
    container_commit = _nullable_sha(container["commit_sha"], "packet_container.commit_sha")
    container_tree = _nullable_sha(container["tree_sha"], "packet_container.tree_sha")
    if kind == "uncommitted_packet":
        if container_commit is not None or container_tree is not None:
            raise PacketError("packet_container: uncommitted packet must use null commit/tree")
        return
    if container_commit is None or container_tree is None:
        raise PacketError("packet_container: documentation commit requires commit/tree")
    _, actual_tree = _commit_facts(repo_root, container_commit, "packet_container.commit_sha")
    if actual_tree != container_tree:
        raise PacketError(f"packet_container.tree_sha: expected {actual_tree}")
    if container_commit == candidate:
        raise PacketError("packet_container.commit_sha must differ from implementation candidate")
    if not _is_ancestor(repo_root, candidate, container_commit):
        raise PacketError("packet_container.commit_sha must descend from implementation candidate")


def _validate_freeze_snapshot(packet: dict[str, Any], repo_root: Path) -> None:
    snapshot = _mapping(packet["freeze_snapshot"], "freeze_snapshot")
    _exact_keys(snapshot, {"refs", "tip_moved"}, "freeze_snapshot")
    refs = _list(snapshot["refs"], "freeze_snapshot.refs")
    seen: set[str] = set()
    moved = False
    for index, value in enumerate(refs):
        context = f"freeze_snapshot.refs[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {"name", "availability", "start_sha", "end_sha", "verify_local"},
            context,
        )
        name = _text(item["name"], f"{context}.name")
        if name in seen:
            raise PacketError(f"{context}.name: duplicate ref")
        seen.add(name)
        availability = _enum(item["availability"], {"available", "unavailable"}, f"{context}.availability")
        start = _nullable_sha(item["start_sha"], f"{context}.start_sha")
        end = _nullable_sha(item["end_sha"], f"{context}.end_sha")
        verify_local = _boolean(item["verify_local"], f"{context}.verify_local")
        if availability == "unavailable":
            if start is not None or end is not None or verify_local:
                raise PacketError(f"{context}: unavailable ref must have null tips and verify_local=false")
            continue
        if start is None or end is None:
            raise PacketError(f"{context}: available ref requires start/end SHA")
        moved = moved or start != end
        if verify_local:
            resolved = _resolve_ref(repo_root, name, f"{context}.name")
            if resolved != end:
                raise PacketError(f"{context}: current local ref {resolved} differs from end tip {end}")
    if _boolean(snapshot["tip_moved"], "freeze_snapshot.tip_moved") != moved:
        raise PacketError(f"freeze_snapshot.tip_moved: expected {moved}")


def _validate_authority(packet: dict[str, Any], repo_root: Path, base: str) -> None:
    authority = _mapping(packet["authority"], "authority")
    _exact_keys(
        authority,
        {"convergence_base_decision", "owner_decision_refs", "authority_paths"},
        "authority",
    )
    _text(authority["convergence_base_decision"], "authority.convergence_base_decision")
    _string_list(authority["owner_decision_refs"], "authority.owner_decision_refs", nonempty=True)
    paths = _list(authority["authority_paths"], "authority.authority_paths")
    if not paths:
        raise PacketError("authority.authority_paths: expected non-empty list")
    seen: set[str] = set()
    for index, value in enumerate(paths):
        context = f"authority.authority_paths[{index}]"
        item = _mapping(value, context)
        _exact_keys(item, {"path", "blob_sha"}, context)
        path = _repo_path(item["path"], f"{context}.path")
        blob = _sha(item["blob_sha"], f"{context}.blob_sha")
        if path in seen:
            raise PacketError(f"{context}.path: duplicate path")
        seen.add(path)
        actual = _tree_blob(repo_root, base, path)
        if actual != blob:
            raise PacketError(f"{context}.blob_sha: expected {actual}")


def _validate_repository_scope(packet: dict[str, Any]) -> None:
    scope = _mapping(packet["repository_scope"], "repository_scope")
    _exact_keys(
        scope,
        {"citlali", "tolproj", "tolteca", "compensation_elsewhere_allowed"},
        "repository_scope",
    )
    expected = {
        "citlali": "repairable_in_current_authorized_repository_scope",
        "tolproj": "repairable_only_in_separately_reviewed_repository_lane",
        "tolteca": "blocked_deferred_read_only",
    }
    for key, value in expected.items():
        if scope[key] != value:
            raise PacketError(f"repository_scope.{key}: expected {value!r}")
    if _boolean(
        scope["compensation_elsewhere_allowed"], "repository_scope.compensation_elsewhere_allowed"
    ):
        raise PacketError("repository_scope.compensation_elsewhere_allowed must be false")


def _validate_ancestry(
    packet: dict[str, Any], repo_root: Path, base: str, candidate: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ancestry = _mapping(packet["ancestry"], "ancestry")
    _exact_keys(
        ancestry,
        {"application_history", "excluded_history", "source_dependencies"},
        "ancestry",
    )
    history_values = _list(ancestry["application_history"], "ancestry.application_history")
    expected_commits_text = _git_text(
        repo_root, ["rev-list", "--reverse", "--topo-order", f"{base}..{candidate}"]
    )
    expected_commits = expected_commits_text.splitlines() if expected_commits_text else []
    history: list[dict[str, Any]] = []
    observed_commits: list[str] = []
    for index, value in enumerate(history_values):
        context = f"ancestry.application_history[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {"commit_sha", "parent_shas", "tree_sha", "purpose", "categories", "import_disposition"},
            context,
        )
        commit = _sha(item["commit_sha"], f"{context}.commit_sha")
        parents = [
            _sha(parent, f"{context}.parent_shas[{parent_index}]")
            for parent_index, parent in enumerate(_list(item["parent_shas"], f"{context}.parent_shas"))
        ]
        tree = _sha(item["tree_sha"], f"{context}.tree_sha")
        _text(item["purpose"], f"{context}.purpose")
        categories = _string_list(
            item["categories"], f"{context}.categories", allowed=HISTORY_CATEGORIES, nonempty=True
        )
        disposition = _enum(
            item["import_disposition"], IMPORT_DISPOSITIONS, f"{context}.import_disposition"
        )
        actual_parents, actual_tree = _commit_facts(repo_root, commit, f"{context}.commit_sha")
        if parents != actual_parents:
            raise PacketError(f"{context}.parent_shas: expected {actual_parents}")
        if tree != actual_tree:
            raise PacketError(f"{context}.tree_sha: expected {actual_tree}")
        observed_commits.append(commit)
        history.append({"commit": commit, "categories": categories, "disposition": disposition})
    if observed_commits != expected_commits:
        raise PacketError(
            "ancestry.application_history: must exactly match ordered base..candidate commits; "
            f"recorded={observed_commits}, expected={expected_commits}"
        )

    excluded_values = _list(ancestry["excluded_history"], "ancestry.excluded_history")
    excluded: list[dict[str, Any]] = []
    excluded_commits: set[str] = set()
    for index, value in enumerate(excluded_values):
        context = f"ancestry.excluded_history[{index}]"
        item = _mapping(value, context)
        _exact_keys(item, {"commit_sha", "tree_sha", "category", "reason"}, context)
        commit = _sha(item["commit_sha"], f"{context}.commit_sha")
        tree = _sha(item["tree_sha"], f"{context}.tree_sha")
        category = _enum(item["category"], EXCLUDED_CATEGORIES, f"{context}.category")
        _text(item["reason"], f"{context}.reason")
        if commit in excluded_commits or commit in set(observed_commits):
            raise PacketError(f"{context}.commit_sha: duplicate or also application history")
        excluded_commits.add(commit)
        _, actual_tree = _commit_facts(repo_root, commit, f"{context}.commit_sha")
        if tree != actual_tree:
            raise PacketError(f"{context}.tree_sha: expected {actual_tree}")
        if _is_ancestor(repo_root, commit, candidate):
            raise PacketError(f"{context}.commit_sha: excluded history is candidate ancestor")
        excluded.append({"commit": commit, "category": category})

    dependencies_values = _list(ancestry["source_dependencies"], "ancestry.source_dependencies")
    dependencies: list[dict[str, Any]] = []
    dependency_ids: set[str] = set()
    for index, value in enumerate(dependencies_values):
        context = f"ancestry.source_dependencies[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {"dependency_id", "repository", "commit_sha", "classification", "disposition", "owner", "reason"},
            context,
        )
        dependency_id = _text(item["dependency_id"], f"{context}.dependency_id")
        if dependency_id in dependency_ids:
            raise PacketError(f"{context}.dependency_id: duplicate")
        dependency_ids.add(dependency_id)
        repository = _text(item["repository"], f"{context}.repository")
        commit = _nullable_sha(item["commit_sha"], f"{context}.commit_sha")
        classification = _enum(
            item["classification"], DEPENDENCY_CLASSIFICATIONS, f"{context}.classification"
        )
        disposition = _enum(
            item["disposition"], DEPENDENCY_DISPOSITIONS, f"{context}.disposition"
        )
        _text(item["owner"], f"{context}.owner")
        _text(item["reason"], f"{context}.reason")
        is_candidate_ancestor = False
        if commit is not None and repository == "citlali":
            _require_commit(repo_root, commit, f"{context}.commit_sha")
            is_candidate_ancestor = _is_ancestor(repo_root, commit, candidate)
        if disposition == "excluded" and is_candidate_ancestor:
            raise PacketError(f"{context}: excluded dependency remains in candidate ancestry")
        dependencies.append(
            {
                "classification": classification,
                "disposition": disposition,
                "is_candidate_ancestor": is_candidate_ancestor,
            }
        )
    return history, dependencies


def _validate_changed_scope(
    packet: dict[str, Any], repo_root: Path, base: str, candidate: str
) -> tuple[set[str], set[str]]:
    scope = _mapping(packet["changed_scope"], "changed_scope")
    _exact_keys(
        scope,
        {"paths", "interfaces", "affected_modes", "governed_change_kinds"},
        "changed_scope",
    )
    _, actual_records = _diff_name_status(repo_root, base, candidate)
    recorded: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(_list(scope["paths"], "changed_scope.paths")):
        context = f"changed_scope.paths[{index}]"
        item = _mapping(value, context)
        _exact_keys(item, {"status", "path", "blob_sha", "category", "owner"}, context)
        status = _enum(item["status"], {"A", "M", "D", "T"}, f"{context}.status")
        path = _repo_path(item["path"], f"{context}.path")
        blob = _nullable_sha(item["blob_sha"], f"{context}.blob_sha")
        _text(item["category"], f"{context}.category")
        _text(item["owner"], f"{context}.owner")
        if path in seen_paths:
            raise PacketError(f"{context}.path: duplicate")
        seen_paths.add(path)
        actual_blob = _tree_blob(repo_root, candidate, path)
        if status == "D":
            if blob is not None or actual_blob is not None:
                raise PacketError(f"{context}.blob_sha: deleted path requires null blob")
        elif blob != actual_blob:
            raise PacketError(f"{context}.blob_sha: expected {actual_blob}")
        recorded.append((status, path))
    if recorded != actual_records:
        raise PacketError(
            f"changed_scope.paths: recorded {recorded}, Git name-status has {actual_records}"
        )

    interfaces = _list(scope["interfaces"], "changed_scope.interfaces")
    if not interfaces:
        raise PacketError("changed_scope.interfaces: expected non-empty list")
    interface_ids: set[str] = set()
    for index, value in enumerate(interfaces):
        context = f"changed_scope.interfaces[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {
                "interface",
                "path",
                "architectural_owner",
                "scientific_owners",
                "lifecycle_owner",
                "classification",
                "required_evidence",
                "future_stage_owner",
            },
            context,
        )
        interface = _text(item["interface"], f"{context}.interface")
        if interface in interface_ids:
            raise PacketError(f"{context}.interface: duplicate")
        interface_ids.add(interface)
        if item["path"] is not None:
            _repo_path(item["path"], f"{context}.path")
        _text(item["architectural_owner"], f"{context}.architectural_owner")
        _string_list(item["scientific_owners"], f"{context}.scientific_owners", nonempty=True)
        _text(item["lifecycle_owner"], f"{context}.lifecycle_owner")
        _enum(item["classification"], INTERFACE_CLASSIFICATIONS, f"{context}.classification")
        _string_list(item["required_evidence"], f"{context}.required_evidence", nonempty=True)
        _text(item["future_stage_owner"], f"{context}.future_stage_owner")
    affected_modes = set(
        _string_list(scope["affected_modes"], "changed_scope.affected_modes", allowed=MODES, nonempty=True)
    )
    change_kinds = set(
        _string_list(
            scope["governed_change_kinds"],
            "changed_scope.governed_change_kinds",
            allowed=GOVERNED_CHANGE_KINDS,
            nonempty=True,
        )
    )
    return affected_modes, change_kinds


def _validate_disposition(
    packet: dict[str, Any], repo_root: Path
) -> tuple[dict[str, str], list[dict[str, str]]]:
    disposition = _mapping(packet["independent_disposition"], "independent_disposition")
    _exact_keys(
        disposition,
        {"review_commit_sha", "review_tree_sha", "report_path", "report_sha256", "axes", "findings"},
        "independent_disposition",
    )
    review = _sha(disposition["review_commit_sha"], "independent_disposition.review_commit_sha")
    review_tree = _sha(disposition["review_tree_sha"], "independent_disposition.review_tree_sha")
    report_path = _repo_path(disposition["report_path"], "independent_disposition.report_path")
    report_digest = _sha256(
        disposition["report_sha256"], "independent_disposition.report_sha256"
    )
    _, actual_tree = _commit_facts(repo_root, review, "independent_disposition.review_commit_sha")
    if review_tree != actual_tree:
        raise PacketError(f"independent_disposition.review_tree_sha: expected {actual_tree}")
    actual_report_digest = _blob_sha256(repo_root, review, report_path)
    if report_digest != actual_report_digest:
        raise PacketError(
            f"independent_disposition.report_sha256: expected {actual_report_digest}"
        )
    axes_value = _mapping(disposition["axes"], "independent_disposition.axes")
    _exact_keys(
        axes_value,
        {
            "scientific_contract",
            "implementation",
            "validation_readiness",
            "historical_fixture",
            "production",
            "verdict",
        },
        "independent_disposition.axes",
    )
    axes = {
        key: _text(value, f"independent_disposition.axes.{key}")
        for key, value in axes_value.items()
    }
    findings: list[dict[str, str]] = []
    finding_ids: set[str] = set()
    for index, value in enumerate(_list(disposition["findings"], "independent_disposition.findings")):
        context = f"independent_disposition.findings[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {"finding_id", "status", "owner", "blocking_stage", "rationale", "evidence_ids", "changed_at_candidate"},
            context,
        )
        finding_id = _text(item["finding_id"], f"{context}.finding_id")
        if finding_id in finding_ids:
            raise PacketError(f"{context}.finding_id: duplicate")
        finding_ids.add(finding_id)
        status = _enum(item["status"], FINDING_STATES, f"{context}.status")
        _text(item["owner"], f"{context}.owner")
        blocking_stage = _enum(item["blocking_stage"], BLOCKING_STAGES, f"{context}.blocking_stage")
        _text(item["rationale"], f"{context}.rationale")
        _string_list(item["evidence_ids"], f"{context}.evidence_ids")
        _boolean(item["changed_at_candidate"], f"{context}.changed_at_candidate")
        findings.append({"id": finding_id, "status": status, "blocking_stage": blocking_stage})
    return axes, findings


def _load_json_blob(repo_root: Path, commit: str, path: str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(_blob_bytes(repo_root, commit, path).decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PacketError(f"{context}: referenced blob is not valid JSON") from error
    return _mapping(value, context)


def _validate_scientific_change(
    packet: dict[str, Any], repo_root: Path, candidate: str, change_kinds: set[str]
) -> str:
    science = _mapping(packet["scientific_change"], "scientific_change")
    _exact_keys(
        science,
        {
            "state",
            "owner_basis",
            "ledger_path",
            "ledger_blob_sha",
            "change_ids",
            "predecessor_epoch_id",
            "successor_epoch_id",
            "successor_epoch_status",
            "profile_ids",
        },
        "scientific_change",
    )
    state = _enum(science["state"], {"none", "declared"}, "scientific_change.state")
    _text(science["owner_basis"], "scientific_change.owner_basis")
    ledger_path = _repo_path(science["ledger_path"], "scientific_change.ledger_path")
    ledger_blob = _sha(science["ledger_blob_sha"], "scientific_change.ledger_blob_sha")
    actual_blob = _tree_blob(repo_root, candidate, ledger_path)
    if ledger_blob != actual_blob:
        raise PacketError(f"scientific_change.ledger_blob_sha: expected {actual_blob}")
    change_ids = _string_list(science["change_ids"], "scientific_change.change_ids")
    predecessor = science["predecessor_epoch_id"]
    successor = science["successor_epoch_id"]
    successor_status = _enum(
        science["successor_epoch_status"], {"none", "preparing", "active"}, "scientific_change.successor_epoch_status"
    )
    profiles = _string_list(science["profile_ids"], "scientific_change.profile_ids")
    if state == "none":
        if change_ids or predecessor is not None or successor is not None or successor_status != "none" or profiles:
            raise PacketError("scientific_change: state none cannot claim change IDs or successor epoch")
        if change_kinds & SCIENCE_DECLARATION_REQUIRED:
            raise PacketError(
                "scientific_change.state: governed algorithm/numerical/default/schema/product change requires declared state"
            )
        return state
    _text(predecessor, "scientific_change.predecessor_epoch_id")
    _text(successor, "scientific_change.successor_epoch_id")
    if successor_status not in {"preparing", "active"}:
        raise PacketError("scientific_change.successor_epoch_status: declared change requires preparing or active")
    if not change_ids or not profiles:
        raise PacketError("scientific_change: declared change requires change_ids and profile_ids")
    ledger = _load_json_blob(repo_root, candidate, ledger_path, "scientific_change.ledger_path")
    changes = _list(ledger.get("changes"), "scientific_change ledger changes")
    accepted_ids = {
        item.get("change_id")
        for item in changes
        if isinstance(item, dict) and item.get("status") == "accepted"
    }
    unknown = sorted(set(change_ids) - accepted_ids)
    if unknown:
        raise PacketError(f"scientific_change.change_ids: unknown accepted ledger IDs {unknown}")
    return state


def _validate_artifact(
    value: Any,
    context: str,
    repo_root: Path,
    candidate: str,
) -> str:
    item = _mapping(value, context)
    _exact_keys(
        item,
        {
            "artifact_id",
            "location_kind",
            "path_or_uri",
            "source_commit_sha",
            "originating_candidate_sha",
            "sha256",
        },
        context,
    )
    artifact_id = _text(item["artifact_id"], f"{context}.artifact_id")
    location = _enum(item["location_kind"], ARTIFACT_LOCATIONS, f"{context}.location_kind")
    path_or_uri = _text(item["path_or_uri"], f"{context}.path_or_uri")
    source_commit = _nullable_sha(item["source_commit_sha"], f"{context}.source_commit_sha")
    origin = _sha(item["originating_candidate_sha"], f"{context}.originating_candidate_sha")
    digest = _sha256(item["sha256"], f"{context}.sha256")
    if origin != candidate:
        raise PacketError(f"{context}.originating_candidate_sha: differs from implementation candidate")
    if location == "repository_blob":
        path = _repo_path(path_or_uri, f"{context}.path_or_uri")
        if source_commit is None:
            raise PacketError(f"{context}.source_commit_sha: repository blob requires commit")
        _require_commit(repo_root, source_commit, f"{context}.source_commit_sha")
        actual = _blob_sha256(repo_root, source_commit, path)
        if actual != digest:
            raise PacketError(f"{context}.sha256: expected {actual}")
    return artifact_id


def _required_gate_ids(packet_kind: str, unity_required: bool, required_modes: set[str]) -> set[str]:
    result = set(CORE_GATES)
    if packet_kind in {"cal_lane", "combined"}:
        result.update(CAL_GATES)
    if packet_kind in {"align_lane", "combined"}:
        result.update(ALIGN_GATES)
    if packet_kind == "combined":
        result.update(OVERLAP_GATES)
    if unity_required:
        result.add(UNITY_GATE)
        result.update(MODE_GATES[mode] for mode in required_modes)
    return result


def _validate_interface_contract(
    value: Any,
    context: str,
    gate_id: str,
    gate_result: str,
    artifact_digests: set[str],
) -> list[str]:
    contract = _mapping(value, context)
    _exact_keys(
        contract,
        {
            "applicable",
            "interface_id",
            "producer_repository",
            "consumer_repository",
            "producer_commit_sha",
            "producer_tree_sha",
            "consumer_commit_sha",
            "consumer_tree_sha",
            "owner_repositories",
            "producer_artifact_schema",
            "consumer_preflight",
            "stable_scoped_keys",
            "exact_artifact_sha256",
            "mapping_sha256",
            "counterexamples",
            "readiness_status",
            "blocking_dependencies",
            "mode_routes",
        },
        context,
    )
    applicable = _boolean(contract["applicable"], f"{context}.applicable")
    interface_id = _text(contract["interface_id"], f"{context}.interface_id", allow_empty=True)
    producer = _text(
        contract["producer_repository"], f"{context}.producer_repository", allow_empty=True
    )
    consumer = _text(
        contract["consumer_repository"], f"{context}.consumer_repository", allow_empty=True
    )
    producer_commit = _nullable_sha(
        contract["producer_commit_sha"], f"{context}.producer_commit_sha"
    )
    producer_tree = _nullable_sha(
        contract["producer_tree_sha"], f"{context}.producer_tree_sha"
    )
    consumer_commit = _nullable_sha(
        contract["consumer_commit_sha"], f"{context}.consumer_commit_sha"
    )
    consumer_tree = _nullable_sha(
        contract["consumer_tree_sha"], f"{context}.consumer_tree_sha"
    )
    owners = _string_list(contract["owner_repositories"], f"{context}.owner_repositories")
    artifact_schema = _text(
        contract["producer_artifact_schema"],
        f"{context}.producer_artifact_schema",
        allow_empty=True,
    )
    preflight = _text(
        contract["consumer_preflight"], f"{context}.consumer_preflight", allow_empty=True
    )
    scoped_keys = _string_list(contract["stable_scoped_keys"], f"{context}.stable_scoped_keys")
    artifact_digest = _nullable_sha256(
        contract["exact_artifact_sha256"], f"{context}.exact_artifact_sha256"
    )
    mapping_digest = _nullable_sha256(
        contract["mapping_sha256"], f"{context}.mapping_sha256"
    )
    counterexamples = _string_list(contract["counterexamples"], f"{context}.counterexamples")
    readiness = _enum(
        contract["readiness_status"], APT_READINESS_STATES, f"{context}.readiness_status"
    )
    dependencies = _string_list(
        contract["blocking_dependencies"], f"{context}.blocking_dependencies"
    )
    routes = _list(contract["mode_routes"], f"{context}.mode_routes")

    expected_pair = APT_INTERFACE_GATES.get(gate_id)
    if expected_pair is None:
        if applicable:
            raise PacketError(f"{context}.applicable: only APT interface gates may be applicable")
        if any(
            (
                interface_id,
                producer,
                consumer,
                producer_commit,
                producer_tree,
                consumer_commit,
                consumer_tree,
                owners,
                artifact_schema,
                preflight,
                scoped_keys,
                artifact_digest,
                mapping_digest,
                counterexamples,
                dependencies,
                routes,
            )
        ) or readiness != "not_applicable":
            raise PacketError(f"{context}: non-APT gate requires an empty not_applicable contract")
        return []

    if not applicable or interface_id != gate_id:
        raise PacketError(f"{context}: mandatory APT row requires applicable=true and matching interface_id")
    if (producer, consumer) != expected_pair:
        raise PacketError(
            f"{context}: expected producer/consumer repositories {expected_pair}, got {(producer, consumer)}"
        )
    if not owners or not artifact_schema.strip() or not preflight.strip():
        raise PacketError(
            f"{context}: APT row requires owner repositories, producer schema, and consumer preflight"
        )
    if not scoped_keys or not counterexamples:
        raise PacketError(f"{context}: APT row requires scoped keys and counterexamples")
    if artifact_digest is not None and artifact_digest not in artifact_digests:
        raise PacketError(
            f"{context}.exact_artifact_sha256: must identify a gate input or output artifact"
        )
    if mapping_digest is not None and mapping_digest not in artifact_digests:
        raise PacketError(
            f"{context}.mapping_sha256: must identify a gate input or output artifact"
        )
    seen_modes: set[str] = set()
    applicable_routes = 0
    route_endpoints: set[str] = set()
    allowed_endpoints = APT_ROUTE_ENDPOINTS[gate_id]
    for index, route_value in enumerate(routes):
        route_context = f"{context}.mode_routes[{index}]"
        route = _mapping(route_value, route_context)
        _exact_keys(
            route,
            {
                "mode",
                "applicable",
                "actual_direction",
                "route_producer_repository",
                "route_consumer_repository",
                "tolapt_role",
                "nonapplicability_authority",
                "nonapplicability_reason",
            },
            route_context,
        )
        mode = _enum(route["mode"], MODES, f"{route_context}.mode")
        if mode in seen_modes:
            raise PacketError(f"{route_context}.mode: duplicate")
        seen_modes.add(mode)
        route_applicable = _boolean(route["applicable"], f"{route_context}.applicable")
        direction = _text(route["actual_direction"], f"{route_context}.actual_direction")
        route_producer = _text(
            route["route_producer_repository"],
            f"{route_context}.route_producer_repository",
            allow_empty=True,
        )
        route_consumer = _text(
            route["route_consumer_repository"],
            f"{route_context}.route_consumer_repository",
            allow_empty=True,
        )
        tolapt_role = _enum(route["tolapt_role"], TOLAPT_ROLES, f"{route_context}.tolapt_role")
        authority = _text(
            route["nonapplicability_authority"],
            f"{route_context}.nonapplicability_authority",
            allow_empty=True,
        )
        reason = _text(
            route["nonapplicability_reason"],
            f"{route_context}.nonapplicability_reason",
            allow_empty=True,
        )
        if route_applicable and (authority or reason):
            raise PacketError(f"{route_context}: applicable route requires empty nonapplicability fields")
        if not route_applicable and (not authority.strip() or not reason.strip()):
            raise PacketError(
                f"{route_context}: non-applicable route requires authority and explicit reason"
            )
        if route_applicable:
            applicable_routes += 1
            endpoint_pair = (route_producer, route_consumer)
            if endpoint_pair not in allowed_endpoints:
                raise PacketError(
                    f"{route_context}: endpoint pair {endpoint_pair} is invalid for {gate_id}"
                )
            if direction != f"{route_producer}->{route_consumer}":
                raise PacketError(
                    f"{route_context}.actual_direction: must equal the explicit endpoint direction"
                )
            route_endpoints.update(endpoint_pair)
            if gate_id in {
                "APT-A-RAW-KMP-CITLALI-AXIS-001",
                "APT-B-CITLALI-BEAMMAP-EXPORT-001",
            } and tolapt_role != "not_in_path":
                raise PacketError(f"{route_context}.tolapt_role: TolAPT is not in this interface")
            if gate_id == "APT-C-BEAMMAP-MATCHING-001":
                expected_role = (
                    "offline_downstream" if route_consumer == "tolapt" else "not_in_path"
                )
                if tolapt_role != expected_role:
                    raise PacketError(
                        f"{route_context}.tolapt_role: expected {expected_role!r} for this route"
                    )
            if gate_id == "APT-D-TOLAPT-TOLPROJ-PACKAGE-001" and (
                tolapt_role != "offline_package_exchange"
            ):
                raise PacketError(
                    f"{route_context}.tolapt_role: TolAPT/TolProj exchange is offline"
                )
            if gate_id in {
                "APT-E-TOLPROJ-TOLTECA-SELECTION-001",
                "APT-F-TOLTECA-CITLALI-TRANSPORT-001",
                "APT-G-CITLALI-ADMISSION-001",
            } and tolapt_role not in {"precomputed_input", "not_in_path"}:
                raise PacketError(
                    f"{route_context}.tolapt_role: expected precomputed_input or not_in_path"
                )
        else:
            if route_producer or route_consumer or direction != "not_applicable":
                raise PacketError(
                    f"{route_context}: non-applicable route requires empty endpoints and "
                    "actual_direction='not_applicable'"
                )
            if tolapt_role != "not_in_path":
                raise PacketError(f"{route_context}.tolapt_role: non-applicable route is not_in_path")
    if seen_modes != MODES:
        raise PacketError(
            f"{context}.mode_routes: must state actual direction for all modes {sorted(MODES)}"
        )
    if applicable_routes == 0:
        raise PacketError(f"{context}.mode_routes: at least one production mode must be applicable")
    if gate_id in {
        "APT-A-RAW-KMP-CITLALI-AXIS-001",
        "APT-B-CITLALI-BEAMMAP-EXPORT-001",
        "APT-C-BEAMMAP-MATCHING-001",
    }:
        beammap_route = next(route for route in routes if route["mode"] == "beammap")
        if not beammap_route["applicable"]:
            raise PacketError(f"{context}.mode_routes: Beammap route must be applicable")
    if not route_endpoints.issubset(set(owners)):
        raise PacketError(
            f"{context}.owner_repositories: must include every applicable route endpoint"
        )
    if gate_result == "pass":
        if (
            readiness != "pass"
            or artifact_digest is None
            or mapping_digest is None
            or producer_commit is None
            or producer_tree is None
            or consumer_commit is None
            or consumer_tree is None
            or dependencies
        ):
            raise PacketError(
                f"{context}: passing interface gate requires exact endpoint SHA/trees, "
                "passing readiness, artifact/mapping digests, and no blockers"
            )
    elif readiness == "pass":
        raise PacketError(f"{context}.readiness_status: cannot pass when gate result is {gate_result}")
    elif not dependencies:
        raise PacketError(
            f"{context}.blocking_dependencies: non-passing APT interface requires blockers"
        )
    return dependencies


def _validate_apt_phase_contract(
    value: Any,
    context: str,
    gate_id: str,
    gate_result: str,
    artifact_digests: set[str],
) -> list[str]:
    contract = _mapping(value, context)
    keys = {
        "applicable",
        "phase_id",
        "readiness_status",
        "software_revisions",
        "generation_id",
        "generation_root",
        "network_count",
        "artifact_scope_count",
        "complete_case_count",
        "permutation_case_count",
        "rejection_case_count",
        "legacy_input_count",
        "mixed_generation_count",
        "selected_artifacts_all_contract_generated",
        "immutable_generation",
        "historical_evidence_only",
        "blocking_dependencies",
    } | APT_PHASE_DIGEST_FIELDS
    _exact_keys(contract, keys, context)
    applicable = _boolean(contract["applicable"], f"{context}.applicable")
    phase_id = _text(contract["phase_id"], f"{context}.phase_id", allow_empty=True)
    readiness = _enum(
        contract["readiness_status"],
        APT_PHASE_READINESS_STATES,
        f"{context}.readiness_status",
    )
    revisions = _list(contract["software_revisions"], f"{context}.software_revisions")
    parsed_revisions: list[dict[str, Any]] = []
    seen_repositories: set[str] = set()
    for index, value_revision in enumerate(revisions):
        revision_context = f"{context}.software_revisions[{index}]"
        revision = _mapping(value_revision, revision_context)
        _exact_keys(
            revision,
            {
                "repository",
                "role",
                "commit_sha",
                "tree_sha",
                "dirty",
                "acceptance_evidence_sha256",
            },
            revision_context,
        )
        repository = _text(revision["repository"], f"{revision_context}.repository")
        if repository in seen_repositories:
            raise PacketError(f"{revision_context}.repository: duplicate")
        seen_repositories.add(repository)
        parsed_revisions.append(
            {
                "repository": repository,
                "role": _text(revision["role"], f"{revision_context}.role"),
                "commit_sha": _sha(revision["commit_sha"], f"{revision_context}.commit_sha"),
                "tree_sha": _sha(revision["tree_sha"], f"{revision_context}.tree_sha"),
                "dirty": _boolean(revision["dirty"], f"{revision_context}.dirty"),
                "acceptance_evidence_sha256": _sha256(
                    revision["acceptance_evidence_sha256"],
                    f"{revision_context}.acceptance_evidence_sha256",
                ),
            }
        )
    generation_id = _text(
        contract["generation_id"], f"{context}.generation_id", allow_empty=True
    )
    generation_root = _text(
        contract["generation_root"], f"{context}.generation_root", allow_empty=True
    )
    digests = {
        field: _nullable_sha256(contract[field], f"{context}.{field}")
        for field in APT_PHASE_DIGEST_FIELDS
    }
    for field, digest in digests.items():
        if digest is not None and digest not in artifact_digests:
            raise PacketError(
                f"{context}.{field}: must identify a gate input or output artifact"
            )
    counts = {
        field: _integer(contract[field], f"{context}.{field}", minimum=0)
        for field in (
            "network_count",
            "artifact_scope_count",
            "complete_case_count",
            "permutation_case_count",
            "rejection_case_count",
            "legacy_input_count",
            "mixed_generation_count",
        )
    }
    selected_contract = _boolean(
        contract["selected_artifacts_all_contract_generated"],
        f"{context}.selected_artifacts_all_contract_generated",
    )
    immutable_generation = _boolean(
        contract["immutable_generation"], f"{context}.immutable_generation"
    )
    historical_only = _boolean(
        contract["historical_evidence_only"], f"{context}.historical_evidence_only"
    )
    dependencies = _string_list(
        contract["blocking_dependencies"], f"{context}.blocking_dependencies"
    )

    if gate_id not in APT_LIBRARY_GENERATION_GATES:
        empty_scalars = any((phase_id, generation_id, generation_root, revisions, dependencies))
        empty_digests = any(value_digest is not None for value_digest in digests.values())
        empty_counts = any(counts.values())
        empty_flags = selected_contract or immutable_generation or historical_only
        if (
            applicable
            or readiness != "not_applicable"
            or empty_scalars
            or empty_digests
            or empty_counts
            or empty_flags
        ):
            raise PacketError(
                f"{context}: non-APT-library gate requires an empty not_applicable contract"
            )
        return []

    if not applicable or phase_id != gate_id:
        raise PacketError(
            f"{context}: mandatory APT-library row requires applicable=true and matching phase_id"
        )
    if not historical_only:
        raise PacketError(
            f"{context}.historical_evidence_only: legacy libraries/runs are comparison evidence only"
        )
    if counts["legacy_input_count"] != 0 or counts["mixed_generation_count"] != 0:
        raise PacketError(
            f"{context}: refactor generation and reductions prohibit legacy inputs and mixed generations"
        )
    if gate_result == "pass":
        if readiness != "pass" or dependencies:
            raise PacketError(
                f"{context}: pass requires readiness_status=pass and zero blockers"
            )
        if any(revision["dirty"] for revision in parsed_revisions):
            raise PacketError(f"{context}.software_revisions: passing revisions must be clean")
        if gate_id == "APT-LIB-SOFTWARE-FREEZE-001":
            if not parsed_revisions or digests["software_revision_set_sha256"] is None:
                raise PacketError(
                    f"{context}: software freeze pass requires accepted exact revisions and set digest"
                )
        elif gate_id == "APT-LIB-COHORT-MANIFEST-001":
            if any(
                digests[field] is None
                for field in (
                    "software_revision_set_sha256",
                    "config_manifest_sha256",
                    "raw_data_manifest_sha256",
                    "cohort_manifest_sha256",
                )
            ):
                raise PacketError(
                    f"{context}: cohort pass requires exact software/config/raw/cohort manifests"
                )
        elif gate_id == "APT-SAMPLE-NEW-CONTRACT-FIXTURES-001":
            required_sample_digests = (
                "software_revision_set_sha256",
                "config_manifest_sha256",
                "raw_data_manifest_sha256",
                "artifact_manifest_sha256",
                "component_manifest_sha256",
                "membership_sha256",
                "mapping_sha256",
                "transformation_sha256",
                "application_sha256",
            )
            if not parsed_revisions or any(
                digests[field] is None for field in required_sample_digests
            ):
                raise PacketError(f"{context}: sample milestone lacks an exact required digest")
            if (
                counts["network_count"] < 2
                or counts["artifact_scope_count"] < 2
                or counts["complete_case_count"] < 1
                or counts["permutation_case_count"] < 1
                or counts["rejection_case_count"] < 1
            ):
                raise PacketError(
                    f"{context}: sample milestone requires multiple networks/scopes and "
                    "complete, permutation, and rejection cases"
                )
            if not selected_contract:
                raise PacketError(
                    f"{context}.selected_artifacts_all_contract_generated must be true"
                )
        elif gate_id in {
            "APT-LIB-BEAM-CAMPAIGN-001",
            "APT-LIB-CANDIDATE-CONFORMANCE-001",
            "APT-LIB-PROVENANCE-001",
            "APT-LIB-SELECTED-CONTRACT-001",
            "APT-REFACTOR-REDUCTIONS-001",
        }:
            required_lineage_digests = (
                "software_revision_set_sha256",
                "config_manifest_sha256",
                "raw_data_manifest_sha256",
                "artifact_manifest_sha256",
                "component_manifest_sha256",
                "membership_sha256",
                "mapping_sha256",
                "transformation_sha256",
                "application_sha256",
            )
            if not parsed_revisions or any(
                digests[field] is None for field in required_lineage_digests
            ):
                raise PacketError(f"{context}: pass requires a complete exact refactor lineage")
            if not selected_contract:
                raise PacketError(
                    f"{context}.selected_artifacts_all_contract_generated must be true"
                )
        elif gate_id == "APT-LIB-IMMUTABLE-GENERATION-001":
            if (
                not generation_id
                or not generation_root
                or not immutable_generation
                or digests["artifact_manifest_sha256"] is None
                or digests["membership_sha256"] is None
                or digests["mapping_sha256"] is None
            ):
                raise PacketError(
                    f"{context}: pass requires a named immutable generation and exact manifests"
                )
        elif gate_id == "APT-LIB-COMPLETENESS-QUARANTINE-001":
            if (
                digests["artifact_manifest_sha256"] is None
                or digests["quarantine_manifest_sha256"] is None
                or not selected_contract
            ):
                raise PacketError(
                    f"{context}: pass requires complete candidate/quarantine accounting"
                )
        elif gate_id == "APT-LIB-NO-MIXED-LINEAGE-001":
            if not generation_id or not generation_root or not selected_contract:
                raise PacketError(
                    f"{context}: pass requires one selected refactor generation"
                )
        elif gate_id == "APT-LIB-HISTORICAL-IMMUTABILITY-001":
            if digests["artifact_manifest_sha256"] is None:
                raise PacketError(
                    f"{context}: pass requires a historical immutability inventory"
                )
        elif gate_id == "APT-LIB-SHADOW-COMPARISON-001":
            if digests["transformation_sha256"] is None:
                raise PacketError(
                    f"{context}: comparison evidence requires an exact cross-generation report"
                )
        elif gate_id == "APT-LIB-ACTIVATION-ROLLBACK-001":
            if (
                not generation_id
                or not generation_root
                or not immutable_generation
                or digests["rollback_manifest_sha256"] is None
            ):
                raise PacketError(
                    f"{context}: activation pass requires immutable generation and exact rollback"
                )
    elif readiness == "pass":
        raise PacketError(f"{context}.readiness_status: cannot pass when gate result is {gate_result}")
    elif not dependencies:
        raise PacketError(
            f"{context}.blocking_dependencies: non-passing APT-library phase requires blockers"
        )
    return dependencies


def _validate_gate_policy(
    packet: dict[str, Any], packet_kind: str, affected_modes: set[str]
) -> tuple[set[str], set[str], bool]:
    policy = _mapping(packet["gate_policy"], "gate_policy")
    _exact_keys(
        policy,
        {"required_gate_ids", "required_modes", "unity_required", "unity_omission"},
        "gate_policy",
    )
    declared = set(
        _string_list(policy["required_gate_ids"], "gate_policy.required_gate_ids", nonempty=True)
    )
    required_modes = set(
        _string_list(policy["required_modes"], "gate_policy.required_modes", allowed=MODES, nonempty=True)
    )
    if not affected_modes.issubset(required_modes):
        raise PacketError("gate_policy.required_modes must include every changed_scope.affected_mode")
    unity_required = _boolean(policy["unity_required"], "gate_policy.unity_required")
    omission = _mapping(policy["unity_omission"], "gate_policy.unity_omission")
    _exact_keys(omission, {"authority", "reason"}, "gate_policy.unity_omission")
    omission_authority = _text(
        omission["authority"], "gate_policy.unity_omission.authority", allow_empty=True
    )
    omission_reason = _text(omission["reason"], "gate_policy.unity_omission.reason", allow_empty=True)
    if unity_required and (omission_authority or omission_reason):
        raise PacketError("gate_policy.unity_omission must be empty when Unity is required")
    if not unity_required and (not omission_authority.strip() or not omission_reason.strip()):
        raise PacketError("gate_policy.unity_omission requires authority and reason")
    if packet_kind == "combined":
        if not unity_required:
            raise PacketError("gate_policy.unity_required: combined packet requires Unity matrix")
        if required_modes != MODES:
            raise PacketError("gate_policy.required_modes: combined packet requires all four modes")
    expected = _required_gate_ids(packet_kind, unity_required, required_modes)
    missing = sorted(expected - declared)
    if missing:
        raise PacketError(f"gate_policy.required_gate_ids: missing mandatory gates {missing}")
    return declared, required_modes, unity_required


def _validate_gate_results(
    packet: dict[str, Any],
    repo_root: Path,
    candidate: str,
    tree: str,
    base: str,
    required_gate_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = _list(packet["gate_results"], "gate_results")
    by_id: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for index, value in enumerate(rows):
        context = f"gate_results[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {
                "gate_id",
                "gate_version",
                "domain",
                "scope",
                "required",
                "timing",
                "blocking_stage",
                "candidate",
                "inputs",
                "action",
                "outputs",
                "criteria",
                "result",
                "omission",
                "owners",
                "evidence_reference",
                "claim_constraints",
                "started_at",
                "finished_at",
                "metrics",
                "interface_contract",
                "apt_phase_contract",
            },
            context,
        )
        gate_id = _text(item["gate_id"], f"{context}.gate_id")
        if gate_id in by_id:
            raise PacketError(f"{context}.gate_id: duplicate {gate_id!r}")
        _text(item["gate_version"], f"{context}.gate_version")
        _text(item["domain"], f"{context}.domain")
        _enum(item["scope"], GATE_SCOPES, f"{context}.scope")
        required = _boolean(item["required"], f"{context}.required")
        if required != (gate_id in required_gate_ids):
            raise PacketError(
                f"{context}.required: must equal membership in gate_policy.required_gate_ids"
            )
        timings = set(
            _string_list(
                item["timing"], f"{context}.timing", allowed=GATE_TIMINGS, nonempty=True
            )
        )
        blocking_stage = _enum(
            item["blocking_stage"], BLOCKING_STAGES, f"{context}.blocking_stage"
        )
        generation_policy = APT_LIBRARY_GENERATION_GATES.get(gate_id)
        if generation_policy is not None:
            expected_timing, expected_stage = generation_policy
            if expected_timing not in timings:
                raise PacketError(
                    f"{context}.timing: {gate_id} requires {expected_timing!r}"
                )
            if blocking_stage != expected_stage:
                raise PacketError(
                    f"{context}.blocking_stage: {gate_id} requires {expected_stage!r}"
                )
            if gate_id == "APT-LIB-SHADOW-COMPARISON-001" and required:
                raise PacketError(
                    f"{context}.required: cross-generation comparison is evidence, "
                    "not a backward-compatibility promotion requirement"
                )
        gate_candidate = _mapping(item["candidate"], f"{context}.candidate")
        _exact_keys(gate_candidate, {"sha", "tree", "base_sha"}, f"{context}.candidate")
        if _sha(gate_candidate["sha"], f"{context}.candidate.sha") != candidate:
            raise PacketError(f"{context}.candidate.sha: differs from implementation candidate")
        if _sha(gate_candidate["tree"], f"{context}.candidate.tree") != tree:
            raise PacketError(f"{context}.candidate.tree: differs from implementation candidate")
        if _sha(gate_candidate["base_sha"], f"{context}.candidate.base_sha") != base:
            raise PacketError(f"{context}.candidate.base_sha: differs from authorized base")

        input_values = _list(item["inputs"], f"{context}.inputs")
        input_ids = [
            _validate_artifact(artifact, f"{context}.inputs[{artifact_index}]", repo_root, candidate)
            for artifact_index, artifact in enumerate(input_values)
        ]
        if not input_ids or len(input_ids) != len(set(input_ids)):
            raise PacketError(f"{context}.inputs: expected non-empty unique artifact IDs")
        action = _mapping(item["action"], f"{context}.action")
        _exact_keys(action, {"kind", "command_argv", "procedure"}, f"{context}.action")
        action_kind = _enum(action["kind"], ACTION_KINDS, f"{context}.action.kind")
        command = _string_list(action["command_argv"], f"{context}.action.command_argv")
        procedure = _text(action["procedure"], f"{context}.action.procedure", allow_empty=True)
        if action_kind == "local_command" and not command:
            raise PacketError(f"{context}.action.command_argv: local command requires argv")
        if action_kind != "local_command" and not procedure.strip():
            raise PacketError(f"{context}.action.procedure: non-command action requires procedure")

        output_values = _list(item["outputs"], f"{context}.outputs")
        output_ids = [
            _validate_artifact(artifact, f"{context}.outputs[{artifact_index}]", repo_root, candidate)
            for artifact_index, artifact in enumerate(output_values)
        ]
        if len(output_ids) != len(set(output_ids)):
            raise PacketError(f"{context}.outputs: duplicate artifact ID")
        _string_list(item["criteria"], f"{context}.criteria", nonempty=True)
        result = _enum(item["result"], GATE_RESULTS, f"{context}.result")
        artifact_digests = {
            _mapping(artifact, f"{context}.artifact")["sha256"]
            for artifact in input_values + output_values
        }
        interface_dependencies = _validate_interface_contract(
            item["interface_contract"],
            f"{context}.interface_contract",
            gate_id,
            result,
            artifact_digests,
        )
        apt_phase_dependencies = _validate_apt_phase_contract(
            item["apt_phase_contract"],
            f"{context}.apt_phase_contract",
            gate_id,
            result,
            artifact_digests,
        )
        omission = _mapping(item["omission"], f"{context}.omission")
        _exact_keys(omission, {"authority", "reason"}, f"{context}.omission")
        omission_authority = _text(
            omission["authority"], f"{context}.omission.authority", allow_empty=True
        )
        omission_reason = _text(omission["reason"], f"{context}.omission.reason", allow_empty=True)
        if result in {"omitted", "not_applicable", "conditioned"}:
            if not omission_authority.strip() or not omission_reason.strip():
                raise PacketError(f"{context}.omission: result {result} requires authority and reason")
        elif omission_authority or omission_reason:
            raise PacketError(f"{context}.omission: must be empty for result {result}")

        owners = _mapping(item["owners"], f"{context}.owners")
        _exact_keys(
            owners,
            {"execution", "architectural", "scientific", "evidence"},
            f"{context}.owners",
        )
        _text(owners["execution"], f"{context}.owners.execution")
        _text(owners["architectural"], f"{context}.owners.architectural")
        _string_list(owners["scientific"], f"{context}.owners.scientific", nonempty=True)
        _text(owners["evidence"], f"{context}.owners.evidence")
        evidence_reference = _text(
            item["evidence_reference"], f"{context}.evidence_reference", allow_empty=True
        )
        _string_list(item["claim_constraints"], f"{context}.claim_constraints")
        started = _nullable_timestamp(item["started_at"], f"{context}.started_at")
        finished = _nullable_timestamp(item["finished_at"], f"{context}.finished_at")
        metrics = _mapping(item["metrics"], f"{context}.metrics")
        _exact_keys(
            metrics,
            {
                "exit_status",
                "unexpected_error_count",
                "unexplained_required_output_failure_count",
                "missing_required_output_count",
                "skipped_required_comparison_count",
            },
            f"{context}.metrics",
        )
        metric_values: dict[str, int | None] = {}
        for key, value_metric in metrics.items():
            metric_values[key] = (
                None
                if value_metric is None
                else _integer(value_metric, f"{context}.metrics.{key}", minimum=0)
            )
        if result == "pass":
            if not output_ids or not evidence_reference.strip() or started is None or finished is None:
                raise PacketError(
                    f"{context}: pass requires outputs, evidence_reference, and start/finish timestamps"
                )
            if metric_values["exit_status"] != 0:
                raise PacketError(f"{context}.metrics.exit_status: pass requires zero")
            for key in (
                "unexpected_error_count",
                "unexplained_required_output_failure_count",
                "missing_required_output_count",
                "skipped_required_comparison_count",
            ):
                if metric_values[key] != 0:
                    raise PacketError(f"{context}.metrics.{key}: pass requires zero")
        if required and result != "pass":
            blockers.append(f"gate:{gate_id}:{result}")
        by_id[gate_id] = {
            "required": required,
            "result": result,
            "action_kind": action_kind,
            "interface_dependencies": interface_dependencies,
            "apt_phase_dependencies": apt_phase_dependencies,
        }
    missing_rows = sorted((required_gate_ids | MANDATORY_RECORDED_GATES) - set(by_id))
    if missing_rows:
        raise PacketError(f"gate_results: missing mandatory gate rows {missing_rows}")
    return by_id, blockers


def _validate_generated_evidence(
    packet: dict[str, Any], repo_root: Path, candidate: str
) -> None:
    records = _list(packet["generated_evidence"], "generated_evidence")
    seen: set[str] = set()
    for index, value in enumerate(records):
        context = f"generated_evidence[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {
                "evidence_id",
                "source_commit_sha",
                "originating_candidate_sha",
                "command_argv",
                "input_artifacts",
                "output_artifacts",
                "environment",
                "retention_location",
            },
            context,
        )
        evidence_id = _text(item["evidence_id"], f"{context}.evidence_id")
        if evidence_id in seen:
            raise PacketError(f"{context}.evidence_id: duplicate")
        seen.add(evidence_id)
        source_commit = _sha(item["source_commit_sha"], f"{context}.source_commit_sha")
        _require_commit(repo_root, source_commit, f"{context}.source_commit_sha")
        if _sha(item["originating_candidate_sha"], f"{context}.originating_candidate_sha") != candidate:
            raise PacketError(f"{context}.originating_candidate_sha: differs from candidate")
        _string_list(item["command_argv"], f"{context}.command_argv", nonempty=True)
        for field in ("input_artifacts", "output_artifacts"):
            artifacts = _list(item[field], f"{context}.{field}")
            if not artifacts:
                raise PacketError(f"{context}.{field}: expected non-empty list")
            for artifact_index, artifact in enumerate(artifacts):
                _validate_artifact(
                    artifact, f"{context}.{field}[{artifact_index}]", repo_root, candidate
                )
        _string_list(item["environment"], f"{context}.environment", nonempty=True)
        _text(item["retention_location"], f"{context}.retention_location")


def _validate_local_evidence(
    packet: dict[str, Any], candidate: str, tree: str, gates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    evidence = _mapping(packet["local_evidence"], "local_evidence")
    _exact_keys(
        evidence,
        {"candidate_sha", "candidate_tree", "gate_ids", "clean_after_gates", "evidence_references"},
        "local_evidence",
    )
    if _sha(evidence["candidate_sha"], "local_evidence.candidate_sha") != candidate:
        raise PacketError("local_evidence.candidate_sha: differs from candidate")
    if _sha(evidence["candidate_tree"], "local_evidence.candidate_tree") != tree:
        raise PacketError("local_evidence.candidate_tree: differs from candidate")
    gate_ids = _string_list(evidence["gate_ids"], "local_evidence.gate_ids")
    unknown = sorted(set(gate_ids) - set(gates))
    if unknown:
        raise PacketError(f"local_evidence.gate_ids: unknown gates {unknown}")
    clean = _boolean(evidence["clean_after_gates"], "local_evidence.clean_after_gates")
    _string_list(evidence["evidence_references"], "local_evidence.evidence_references")
    return {"gate_ids": set(gate_ids), "clean": clean}


def _validate_unity_evidence(
    packet: dict[str, Any],
    candidate: str,
    tree: str,
    required_modes: set[str],
    unity_required: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence = _mapping(packet["unity_evidence"], "unity_evidence")
    _exact_keys(
        evidence,
        {
            "required",
            "human_mediated_only",
            "codex_accessed_unity",
            "dependency_environment_sha256",
            "omission",
            "rows",
        },
        "unity_evidence",
    )
    required = _boolean(evidence["required"], "unity_evidence.required")
    if required != unity_required:
        raise PacketError("unity_evidence.required: differs from gate_policy.unity_required")
    if not _boolean(evidence["human_mediated_only"], "unity_evidence.human_mediated_only"):
        raise PacketError("unity_evidence.human_mediated_only must be true")
    if _boolean(evidence["codex_accessed_unity"], "unity_evidence.codex_accessed_unity"):
        raise PacketError("unity_evidence.codex_accessed_unity must be false")
    environment_digest = _nullable_sha256(
        evidence["dependency_environment_sha256"], "unity_evidence.dependency_environment_sha256"
    )
    omission = _mapping(evidence["omission"], "unity_evidence.omission")
    _exact_keys(omission, {"authority", "reason"}, "unity_evidence.omission")
    omission_authority = _text(
        omission["authority"], "unity_evidence.omission.authority", allow_empty=True
    )
    omission_reason = _text(omission["reason"], "unity_evidence.omission.reason", allow_empty=True)
    if required:
        if environment_digest is None or omission_authority or omission_reason:
            raise PacketError("unity_evidence: required matrix needs environment digest and empty omission")
    elif not omission_authority.strip() or not omission_reason.strip():
        raise PacketError("unity_evidence.omission: omitted Unity matrix requires authority and reason")

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    seen_modes: set[str] = set()
    for index, value in enumerate(_list(evidence["rows"], "unity_evidence.rows")):
        context = f"unity_evidence.rows[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {
                "mode",
                "profile_id",
                "run_id",
                "candidate_sha",
                "candidate_tree",
                "embedded_version",
                "binary_sha256",
                "config_sha256",
                "input_manifest_sha256",
                "output_manifest_sha256",
                "report_sha256",
                "log_sha256",
                "retrieved_at",
                "retrieved_by",
                "retrieval_source",
                "provided_by_authorized_human",
                "result",
                "unexpected_error_count",
                "unexplained_required_output_failure_count",
                "missing_required_output_count",
                "skipped_required_comparison_count",
            },
            context,
        )
        mode = _enum(item["mode"], MODES, f"{context}.mode")
        if mode in seen_modes:
            raise PacketError(f"{context}.mode: duplicate")
        seen_modes.add(mode)
        _text(item["profile_id"], f"{context}.profile_id")
        _text(item["run_id"], f"{context}.run_id")
        if _sha(item["candidate_sha"], f"{context}.candidate_sha") != candidate:
            raise PacketError(f"{context}.candidate_sha: mixed-SHA Unity evidence")
        if _sha(item["candidate_tree"], f"{context}.candidate_tree") != tree:
            raise PacketError(f"{context}.candidate_tree: differs from candidate")
        _text(item["embedded_version"], f"{context}.embedded_version")
        for key in (
            "binary_sha256",
            "config_sha256",
            "input_manifest_sha256",
            "output_manifest_sha256",
            "report_sha256",
            "log_sha256",
        ):
            _sha256(item[key], f"{context}.{key}")
        _timestamp(item["retrieved_at"], f"{context}.retrieved_at")
        _text(item["retrieved_by"], f"{context}.retrieved_by")
        _text(item["retrieval_source"], f"{context}.retrieval_source")
        if not _boolean(
            item["provided_by_authorized_human"], f"{context}.provided_by_authorized_human"
        ):
            raise PacketError(f"{context}.provided_by_authorized_human must be true")
        result = _enum(item["result"], {"pass", "fail", "blocked", "omitted"}, f"{context}.result")
        counts = {}
        for key in (
            "unexpected_error_count",
            "unexplained_required_output_failure_count",
            "missing_required_output_count",
            "skipped_required_comparison_count",
        ):
            counts[key] = _integer(item[key], f"{context}.{key}", minimum=0)
        if result != "pass" or any(counts.values()):
            blockers.append(f"unity:{mode}:{result}")
        rows.append({"mode": mode, "result": result, "counts": counts})
    if required and seen_modes != required_modes:
        raise PacketError(
            f"unity_evidence.rows: required modes {sorted(required_modes)}, got {sorted(seen_modes)}"
        )
    return rows, blockers


def _validate_external_dependencies(packet: dict[str, Any]) -> list[dict[str, Any]]:
    values = _list(packet["external_dependencies"], "external_dependencies")
    if not values:
        raise PacketError("external_dependencies: expected TolProj and TolTECA records")
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    repositories: set[str] = set()
    for index, value in enumerate(values):
        context = f"external_dependencies[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {
                "dependency_id",
                "repository",
                "classification",
                "status",
                "owner",
                "boundary",
                "evidence_authority",
                "finding_ids",
                "exit_condition",
                "blocking_stage",
                "read_only",
                "compensation_elsewhere_allowed",
                "resolved_commit_sha",
                "resolved_tree_sha",
                "closure_evidence_sha256",
            },
            context,
        )
        dependency_id = _text(item["dependency_id"], f"{context}.dependency_id")
        if dependency_id in ids:
            raise PacketError(f"{context}.dependency_id: duplicate")
        ids.add(dependency_id)
        repository = _text(item["repository"], f"{context}.repository")
        repositories.add(repository)
        classification = _enum(
            item["classification"],
            {
                "repairable_in_current_authorized_repository_scope",
                "repairable_only_in_separately_reviewed_repository_lane",
                "blocked_deferred_at_tolteca",
                "external_owner_dependency",
            },
            f"{context}.classification",
        )
        status = _enum(item["status"], {"open", "blocked", "deferred", "closed"}, f"{context}.status")
        _text(item["owner"], f"{context}.owner")
        _text(item["boundary"], f"{context}.boundary")
        _text(item["evidence_authority"], f"{context}.evidence_authority")
        _string_list(item["finding_ids"], f"{context}.finding_ids", nonempty=True)
        _text(item["exit_condition"], f"{context}.exit_condition")
        blocking_stage = _enum(item["blocking_stage"], BLOCKING_STAGES, f"{context}.blocking_stage")
        read_only = _boolean(item["read_only"], f"{context}.read_only")
        compensation = _boolean(
            item["compensation_elsewhere_allowed"], f"{context}.compensation_elsewhere_allowed"
        )
        resolved_commit = _nullable_sha(
            item["resolved_commit_sha"], f"{context}.resolved_commit_sha"
        )
        resolved_tree = _nullable_sha(item["resolved_tree_sha"], f"{context}.resolved_tree_sha")
        closure_evidence = _nullable_sha256(
            item["closure_evidence_sha256"], f"{context}.closure_evidence_sha256"
        )
        if compensation:
            raise PacketError(f"{context}.compensation_elsewhere_allowed must be false")
        if status == "closed":
            if resolved_commit is None or resolved_tree is None or closure_evidence is None:
                raise PacketError(
                    f"{context}: closed dependency requires exact commit/tree and closure evidence"
                )
        elif any(value is not None for value in (resolved_commit, resolved_tree, closure_evidence)):
            raise PacketError(f"{context}: unresolved dependency cannot claim closure identities")
        if repository == "tolteca":
            if classification != "blocked_deferred_at_tolteca":
                raise PacketError(f"{context}.classification: TolTECA must remain blocked_deferred_at_tolteca")
            if status == "closed" or not read_only or blocking_stage != "production_end_to_end":
                raise PacketError(
                    f"{context}: TolTECA v1 policy requires open/deferred, read-only, production-end-to-end blocker"
                )
        if repository == "tolproj" and classification != "repairable_only_in_separately_reviewed_repository_lane":
            raise PacketError(f"{context}.classification: TolProj must remain a separate reviewed lane")
        if repository == "tolapt" and classification != "external_owner_dependency":
            raise PacketError(
                f"{context}.classification: TolAPT must remain an external owner dependency"
            )
        records.append(
            {
                "dependency_id": dependency_id,
                "repository": repository,
                "classification": classification,
                "status": status,
                "blocking_stage": blocking_stage,
            }
        )
    missing = {"tolapt", "tolproj", "tolteca", "toltec_beammap"} - repositories
    if missing:
        raise PacketError(f"external_dependencies: missing required repository records {sorted(missing)}")
    for record, original in zip(records, values):
        if record["repository"] == "toltec_beammap":
            item = _mapping(original, "external_dependencies[toltec_beammap]")
            if record["classification"] != "external_owner_dependency":
                raise PacketError(
                    "external_dependencies[toltec_beammap].classification: "
                    "expected external_owner_dependency"
                )
            if "BM-R1" not in item["finding_ids"]:
                raise PacketError(
                    "external_dependencies[toltec_beammap].finding_ids: BM-R1 is required"
                )
    return records


def _validate_claims_and_attestations(packet: dict[str, Any]) -> dict[str, bool]:
    claims = _mapping(packet["claims"], "claims")
    _exact_keys(
        claims,
        {
            "supported",
            "conditioned",
            "prohibited",
            "cross_repository_apt_conformance",
            "production_end_to_end_apt_contract",
            "refactor_apt_generation_selected",
            "refactor_reductions_regenerated",
            "legacy_lineage_used_as_refactor_input",
            "legacy_selection_equivalence_required",
            "new_contract_sample_artifact_milestone_met",
            "real_end_to_end_apt_chain_conformance",
            "scientific_readiness",
            "production_readiness",
            "refactor_apt_library_validated",
        },
        "claims",
    )
    _string_list(
        claims["supported"],
        "claims.supported",
        allowed=ALLOWED_CURRENT_APT_EVIDENCE_CLASSES,
        nonempty=True,
    )
    _string_list(claims["conditioned"], "claims.conditioned")
    _string_list(claims["prohibited"], "claims.prohibited", nonempty=True)
    if _boolean(
        claims["cross_repository_apt_conformance"],
        "claims.cross_repository_apt_conformance",
    ):
        raise PacketError(
            "claims.cross_repository_apt_conformance must be false under the v1 external-boundary policy"
        )
    if _boolean(
        claims["production_end_to_end_apt_contract"], "claims.production_end_to_end_apt_contract"
    ):
        raise PacketError(
            "claims.production_end_to_end_apt_contract must be false while TolTECA blocker is policy-bound"
        )
    for key in (
        "refactor_apt_generation_selected",
        "refactor_reductions_regenerated",
        "legacy_lineage_used_as_refactor_input",
        "legacy_selection_equivalence_required",
        "new_contract_sample_artifact_milestone_met",
        "real_end_to_end_apt_chain_conformance",
        "scientific_readiness",
        "production_readiness",
        "refactor_apt_library_validated",
    ):
        if _boolean(claims[key], f"claims.{key}"):
            raise PacketError(
                f"claims.{key} must be false in the currently blocked v1 preparation policy"
            )
    attestations = _mapping(packet["attestations"], "attestations")
    keys = {
        "application_history_separated",
        "zero_unexplained_required_output_failures",
        "zero_unexpected_error_logs",
        "no_skipped_required_comparisons",
        "requested_effective_observation_realized_checked",
        "product_inventory_checked",
        "scientific_conventions_checked",
        "same_sha_local",
        "same_sha_local_unity",
        "compensating_identity_or_admission_weakening",
    }
    _exact_keys(attestations, keys, "attestations")
    result = {key: _boolean(value, f"attestations.{key}") for key, value in attestations.items()}
    if result["compensating_identity_or_admission_weakening"]:
        raise PacketError("attestations.compensating_identity_or_admission_weakening must be false")
    return result


def _validate_approvals(
    packet: dict[str, Any], candidate: str, tree: str
) -> list[dict[str, str]]:
    values = _list(packet["approvals"], "approvals")
    approvals: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, value in enumerate(values):
        context = f"approvals[{index}]"
        item = _mapping(value, context)
        _exact_keys(
            item,
            {"role", "owner", "status", "candidate_sha", "candidate_tree", "recorded_at", "conditions"},
            context,
        )
        role = _enum(
            item["role"],
            {"lane_owner", "scientific_owner", "independent_auditor", "coordinator"},
            f"{context}.role",
        )
        if role in roles:
            raise PacketError(f"{context}.role: duplicate")
        roles.add(role)
        _text(item["owner"], f"{context}.owner")
        status = _enum(item["status"], {"pending", "approved", "conditioned", "rejected"}, f"{context}.status")
        if _sha(item["candidate_sha"], f"{context}.candidate_sha") != candidate:
            raise PacketError(f"{context}.candidate_sha: differs from candidate")
        if _sha(item["candidate_tree"], f"{context}.candidate_tree") != tree:
            raise PacketError(f"{context}.candidate_tree: differs from candidate")
        _timestamp(item["recorded_at"], f"{context}.recorded_at")
        conditions = _string_list(item["conditions"], f"{context}.conditions")
        if status == "approved" and conditions:
            raise PacketError(f"{context}.conditions: approved record must be unconditional")
        approvals.append({"role": role, "status": status})
    return approvals


def _verify_candidate_worktree(candidate_worktree: Path, candidate: str) -> None:
    head = _git_text(candidate_worktree, ["rev-parse", "HEAD"])
    if head != candidate:
        raise PacketError(f"candidate worktree HEAD {head} differs from packet candidate {candidate}")
    status = _git(candidate_worktree, ["status", "--porcelain=v2"]).stdout
    if status:
        raise PacketError("candidate worktree is not clean")


def validate_packet(
    packet: dict[str, Any],
    *,
    repo_root: Path,
    expected_sha: str | None = None,
    require_ready: bool = False,
    candidate_worktree: Path | None = None,
) -> dict[str, Any]:
    top_keys = {
        "schema_version",
        "packet_identity",
        "implementation_candidate",
        "packet_container",
        "freeze_snapshot",
        "authority",
        "repository_scope",
        "ancestry",
        "changed_scope",
        "independent_disposition",
        "scientific_change",
        "gate_policy",
        "gate_results",
        "generated_evidence",
        "local_evidence",
        "unity_evidence",
        "external_dependencies",
        "claims",
        "attestations",
        "approvals",
    }
    _exact_keys(packet, top_keys, "packet")
    if packet["schema_version"] != SCHEMA_VERSION:
        raise PacketError(f"schema_version: expected {SCHEMA_VERSION!r}")
    _reject_placeholders(packet, "packet")
    packet_kind, lifecycle, target = _validate_packet_identity(packet)
    candidate_record, candidate, tree, base = _validate_candidate(packet, repo_root)
    if expected_sha is not None:
        expected = _sha(expected_sha, "expected_sha")
        if candidate != expected:
            raise PacketError(f"candidate {candidate} differs from expected SHA {expected}")
    _validate_packet_container(packet, repo_root, candidate)
    _validate_freeze_snapshot(packet, repo_root)
    _validate_authority(packet, repo_root, base)
    _validate_repository_scope(packet)
    history, dependencies = _validate_ancestry(packet, repo_root, base, candidate)
    affected_modes, change_kinds = _validate_changed_scope(packet, repo_root, base, candidate)
    axes, findings = _validate_disposition(packet, repo_root)
    _validate_scientific_change(packet, repo_root, candidate, change_kinds)
    required_gate_ids, required_modes, unity_required = _validate_gate_policy(
        packet, packet_kind, affected_modes
    )
    gates, gate_blockers = _validate_gate_results(
        packet, repo_root, candidate, tree, base, required_gate_ids
    )
    _validate_generated_evidence(packet, repo_root, candidate)
    local = _validate_local_evidence(packet, candidate, tree, gates)
    _, unity_blockers = _validate_unity_evidence(
        packet, candidate, tree, required_modes, unity_required
    )
    external_dependencies = _validate_external_dependencies(packet)
    declared_dependency_ids = {
        item["dependency_id"] for item in external_dependencies
    }
    declared_gate_or_dependency_ids = set(gates) | declared_dependency_ids
    for gate_id, gate in gates.items():
        unknown_dependencies = sorted(
            (
                set(gate["interface_dependencies"])
                | set(gate["apt_phase_dependencies"])
            )
            - declared_gate_or_dependency_ids
        )
        if unknown_dependencies:
            raise PacketError(
                f"gate_results[{gate_id}].blocking_dependencies: "
                f"unknown IDs {unknown_dependencies}"
            )
    attestations = _validate_claims_and_attestations(packet)
    approvals = _validate_approvals(packet, candidate, tree)

    blockers: list[str] = list(gate_blockers) + list(unity_blockers)
    if lifecycle != "frozen":
        blockers.append(f"lifecycle:{lifecycle}")
    if not candidate_record["implementation_frozen"]:
        blockers.append("candidate:not_frozen")
    if not candidate_record["worktree_clean"]:
        blockers.append("candidate:worktree_not_clean")
    for entry in history:
        if entry["disposition"] != "include_application":
            blockers.append(f"ancestry:{entry['commit']}:{entry['disposition']}")
        if "application" not in entry["categories"]:
            blockers.append(f"ancestry:{entry['commit']}:not_application")
        forbidden = set(entry["categories"]) & FORBIDDEN_APPLICATION_CATEGORIES
        if forbidden:
            blockers.append(f"ancestry:{entry['commit']}:forbidden_categories")
    for dependency in dependencies:
        if dependency["disposition"] in {"independently_required", "deferred"}:
            blockers.append(f"dependency:{dependency['disposition']}")
        if dependency["classification"] == "contaminating" and dependency["is_candidate_ancestor"]:
            blockers.append("dependency:contaminating_ancestor")
    if axes["scientific_contract"] != "approved":
        blockers.append("disposition:scientific_contract")
    if axes["implementation"] != "conformant":
        blockers.append("disposition:implementation")
    if axes["validation_readiness"] != "complete":
        blockers.append("disposition:validation_readiness")
    if axes["verdict"] not in {"accept", "retain"}:
        blockers.append("disposition:verdict")
    target_rank = STAGE_ORDER[target]
    for finding in findings:
        if (
            finding["status"] in {"open", "blocked", "conditioned"}
            and STAGE_ORDER[finding["blocking_stage"]] <= target_rank
        ):
            blockers.append(f"finding:{finding['id']}:{finding['status']}")
    if not local["clean"]:
        blockers.append("local_evidence:not_clean")
    required_local_ids = {
        gate_id
        for gate_id in required_gate_ids
        if gates[gate_id]["action_kind"] != "human_mediated_unity"
    }
    if not required_local_ids.issubset(local["gate_ids"]):
        blockers.append("local_evidence:missing_required_gate")
    required_attestations = {
        "application_history_separated",
        "zero_unexplained_required_output_failures",
        "zero_unexpected_error_logs",
        "no_skipped_required_comparisons",
        "requested_effective_observation_realized_checked",
        "product_inventory_checked",
        "scientific_conventions_checked",
        "same_sha_local",
    }
    if unity_required:
        required_attestations.add("same_sha_local_unity")
    for key in sorted(required_attestations):
        if not attestations[key]:
            blockers.append(f"attestation:{key}")
    approval_index = {item["role"]: item["status"] for item in approvals}
    for role in ("lane_owner", "scientific_owner", "independent_auditor", "coordinator"):
        if approval_index.get(role) != "approved":
            blockers.append(f"approval:{role}")

    blockers = sorted(set(blockers))
    ready = not blockers
    if require_ready:
        if candidate_worktree is None:
            raise PacketError("candidate_worktree is required with require_ready")
        _verify_candidate_worktree(candidate_worktree, candidate)
    identity = packet["packet_identity"]
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": identity["packet_id"],
        "lane_id": identity["lane_id"],
        "packet_kind": packet_kind,
        "candidate_sha": candidate,
        "candidate_tree": tree,
        "target_stage": target,
        "gate_count": len(gates),
        "required_gate_count": len(required_gate_ids),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "ready": ready,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--candidate-worktree", type=Path)
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return nonzero unless every derived handoff requirement passes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        packet = load_packet(args.packet.expanduser().resolve())
        result = validate_packet(
            packet,
            repo_root=args.repo_root.expanduser().resolve(),
            expected_sha=args.expected_sha,
            require_ready=args.require_ready,
            candidate_worktree=(
                args.candidate_worktree.expanduser().resolve()
                if args.candidate_worktree is not None
                else args.repo_root.expanduser().resolve()
            ),
        )
    except PacketError as error:
        print(f"frozen lane handoff packet invalid: {error}", file=sys.stderr)
        return 2
    summary = (
        "frozen lane handoff packet valid: "
        f"packet_id={result['packet_id']} candidate={result['candidate_sha']} "
        f"ready={str(result['ready']).lower()} gates={result['gate_count']} "
        f"blockers={result['blocker_count']}"
    )
    print(summary)
    if args.require_ready and not result["ready"]:
        print(
            "frozen lane handoff packet not ready: " + ", ".join(result["blockers"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
