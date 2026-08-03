#!/usr/bin/env python3
"""Aggregate compact SCI-ALIGN-001 3C273 retained-product diagnostics.

The command has two deliberately separate stages.  ``freeze`` derives the
independent validation grouping from the selected inventory without opening a
timing result.  ``run`` verifies that freeze and consumes only compact per-map
outputs.  It never opens a reduction or raw detector product.

The statistical models are diagnostic prediction models, not physical clock
corrections.  In particular, native frame phase and native-to-assigned-slot
residual are retained as two distinct predictors.  A predictive result can
motivate a later structural native-time/fractional-slot investigation; this
tool never authorizes a fixed physical clock correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


SCHEMA_VERSION = "sci-align-001-3c273-aggregate-v2"
PROTOCOL_VERSION = "sci-align-001-3c273-frozen-analysis-v2"
SELECTED_MANIFEST_SCHEMA = "sci-align-001-3c273-selected-manifest-v2"
INVENTORY_SCHEMA = "sci-align-001-3c273-candidate-inventory-v2"
SELECTION_SCHEMA = "sci-align-001-3c273-selection-v2"
DEFAULT_ALPHA = 0.05
NETWORK_TIMING_ALIASES = (
    "timing_residual_sec",
    "timing_sec",
    "timing_offset_sec",
)
NETWORK_SE_ALIASES = (
    "timing_se_sec",
    "timing_jackknife_se_sec",
    "timing_uncertainty_sec",
)
SLOT_ALIASES = (
    "native_to_assigned_slot_residual_sec",
    "slot_residual_sec",
    "assigned_slot_residual_sec",
)
PHASE_ALIASES = (
    "native_frame_phase_mean_sec",
    "native_frame_phase_sec",
)
MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {"id": "M0_GLOBAL", "family": "M0", "network": False, "session": False},
    {"id": "M1_NETWORK", "family": "M1", "network": True, "session": False},
    {"id": "M2_SESSION", "family": "M2", "network": False, "session": True},
    {"id": "M3_SLOT_NEG1", "family": "M3", "predictor": "slot", "fixed_beta": -1.0},
    {"id": "M3_SLOT_FREE", "family": "M3", "predictor": "slot"},
    {"id": "M3_PHASE_NEG1", "family": "M3", "predictor": "phase", "fixed_beta": -1.0},
    {"id": "M3_PHASE_FREE", "family": "M3", "predictor": "phase"},
    {"id": "M4_NETWORK_SLOT_NEG1", "family": "M4", "network": True, "predictor": "slot", "fixed_beta": -1.0},
    {"id": "M4_NETWORK_SLOT_FREE", "family": "M4", "network": True, "predictor": "slot"},
    {"id": "M4_NETWORK_PHASE_NEG1", "family": "M4", "network": True, "predictor": "phase", "fixed_beta": -1.0},
    {"id": "M4_NETWORK_PHASE_FREE", "family": "M4", "network": True, "predictor": "phase"},
    {"id": "M4_SESSION_SLOT_NEG1", "family": "M4", "session": True, "predictor": "slot", "fixed_beta": -1.0},
    {"id": "M4_SESSION_SLOT_FREE", "family": "M4", "session": True, "predictor": "slot"},
    {"id": "M4_SESSION_PHASE_NEG1", "family": "M4", "session": True, "predictor": "phase", "fixed_beta": -1.0},
    {"id": "M4_SESSION_PHASE_FREE", "family": "M4", "session": True, "predictor": "phase"},
    {"id": "M4_NETWORK_SESSION", "family": "M4", "network": True, "session": True},
    {"id": "M4_NETWORK_SESSION_SLOT_FREE", "family": "M4", "network": True, "session": True, "predictor": "slot"},
    {"id": "M4_NETWORK_SESSION_PHASE_FREE", "family": "M4", "network": True, "session": True, "predictor": "phase"},
    {"id": "M5_ABSTAIN", "family": "M5", "abstain": True},
)
AGGREGATOR_SCRIPT_PATH = Path(__file__).resolve()


class AggregateError(RuntimeError):
    """A fail-closed input, freeze, or analysis error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _semantic_digest(value: Mapping[str, Any], excluded: Sequence[str] = ()) -> str:
    clean = {key: val for key, val in value.items() if key not in set(excluded)}
    return hashlib.sha256(_canonical_json(clean).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_registry_digest(
    registry: Sequence[Mapping[str, Any]] = MODEL_REGISTRY,
) -> str:
    return hashlib.sha256(
        _canonical_json([dict(item) for item in registry]).encode("ascii")
    ).hexdigest()


def _aggregation_tool_identity() -> dict[str, Any]:
    return {
        "script_name": AGGREGATOR_SCRIPT_PATH.name,
        "script_sha256": sha256_file(AGGREGATOR_SCRIPT_PATH),
        "candidate_model_registry_sha256": _candidate_registry_digest(),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _csv_value(row.get(name)) for name in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_checksums(directory: Path, included: Sequence[Path] | None = None) -> Path:
    checksum = directory / "SHA256SUMS"
    paths = (
        sorted((path for path in directory.rglob("*") if path.is_file() and path != checksum), key=str)
        if included is None
        else sorted((path for path in included if path.is_file() and path != checksum), key=str)
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(directory)}" for path in paths]
    _atomic_text(checksum, "\n".join(lines) + "\n")
    return checksum


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AggregateError("non-finite value cannot be serialized")
        return format(value, ".17g")
    if isinstance(value, (list, dict)):
        return _canonical_json(value)
    return value


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "selected", "eligible", "primary", "canonical"}:
        return True
    if token in {"0", "false", "no", "n", "unselected", "ineligible"}:
        return False
    return default


def _float(value: Any, *, field_name: str, optional: bool = False) -> float | None:
    if value in (None, ""):
        if optional:
            return None
        raise AggregateError(f"missing required numeric field {field_name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AggregateError(f"invalid numeric field {field_name}: {value!r}") from error
    if not math.isfinite(result):
        raise AggregateError(f"non-finite numeric field {field_name}")
    return result


def _int(value: Any, *, field_name: str) -> int:
    result = _float(value, field_name=field_name)
    assert result is not None
    integer = int(result)
    if float(integer) != result:
        raise AggregateError(f"non-integral field {field_name}: {value!r}")
    return integer


def _read_table(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _require_sha256(value: Any, field_name: str) -> str:
    token = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise AggregateError(f"selected manifest has invalid {field_name}")
    return token


def _selected_manifest_document(path: Path) -> dict[str, Any]:
    if path.name != "selected_manifest.json":
        raise AggregateError("aggregation requires the exact selected_manifest.json product")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"cannot read selected manifest JSON: {error}") from error
    if not isinstance(document, dict):
        raise AggregateError("selected manifest must be a JSON object")
    if document.get("schema_version") != SELECTED_MANIFEST_SCHEMA:
        raise AggregateError("unsupported selected-manifest schema")
    expected_keys = {
        "schema_version", "source_inventory_sha256", "owner_selection_sha256",
        "owner_selection_format", "obsnum_allowlist_sha256",
        "obsnum_allowlist_schema_version", "obsnum_allowlist_filename",
        "rows", "manifest_sha256",
    }
    if set(document) != expected_keys:
        raise AggregateError("selected manifest has unexpected or missing top-level fields")
    source_inventory_sha = _require_sha256(
        document.get("source_inventory_sha256"), "source_inventory_sha256",
    )
    inventory_candidates = (
        path.parent / "candidate_inventory.json",
        path.parent.parent / "candidate_inventory.json",
    )
    matching_inventory: dict[str, Any] | None = None
    for inventory_path in dict.fromkeys(inventory_candidates):
        if not inventory_path.is_file():
            continue
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AggregateError(f"cannot read source inventory authority: {error}") from error
        if not isinstance(inventory, dict) or inventory.get("schema_version") != INVENTORY_SCHEMA:
            raise AggregateError("source inventory authority has unsupported schema")
        recorded_inventory_sha = _require_sha256(
            inventory.get("inventory_sha256"), "source inventory inventory_sha256",
        )
        measured_inventory_sha = _semantic_digest(inventory, ("inventory_sha256",))
        if recorded_inventory_sha != measured_inventory_sha:
            raise AggregateError("source inventory internal digest mismatch")
        if recorded_inventory_sha == source_inventory_sha:
            matching_inventory = dict(inventory)
            break
    if matching_inventory is None:
        raise AggregateError(
            "selected manifest source_inventory_sha256 has no matching source inventory authority"
        )
    owner_selection_sha = _require_sha256(
        document.get("owner_selection_sha256"), "owner_selection_sha256",
    )
    owner_format = str(document.get("owner_selection_format") or "")
    if owner_format not in {"csv", "json"}:
        raise AggregateError("selected manifest has invalid owner_selection_format")
    allowlist_sha = _require_sha256(
        document.get("obsnum_allowlist_sha256"), "obsnum_allowlist_sha256",
    )
    if document.get("obsnum_allowlist_schema_version") != "sci-align-001-3c273-obsnum-allowlist-v1":
        raise AggregateError("selected manifest has unsupported obsnum allowlist schema")
    inventory_allowlist = matching_inventory.get("obsnum_allowlist")
    if not isinstance(inventory_allowlist, Mapping) or inventory_allowlist.get("sha256") != allowlist_sha:
        raise AggregateError("selected manifest obsnum allowlist does not match source inventory")
    allowlist_name = str(document.get("obsnum_allowlist_filename") or "")
    if allowlist_name != str(inventory_allowlist.get("filename") or ""):
        raise AggregateError("selected manifest obsnum allowlist filename does not match source inventory")
    allowlist_copy = path.parent / allowlist_name
    if not allowlist_name or not allowlist_copy.is_file() or sha256_file(allowlist_copy) != allowlist_sha:
        raise AggregateError("selected manifest obsnum allowlist copy/digest is invalid")
    owner_selection = path.parent / f"owner_selection.{owner_format}"
    if not owner_selection.is_file() or sha256_file(owner_selection) != owner_selection_sha:
        raise AggregateError("owner-selection file/digest does not match selected manifest")
    if owner_format == "json":
        try:
            owner_document = json.loads(owner_selection.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AggregateError(f"cannot read owner-selection JSON: {error}") from error
        if (
            not isinstance(owner_document, dict)
            or owner_document.get("schema_version") != SELECTION_SCHEMA
            or owner_document.get("source_inventory_sha256") != source_inventory_sha
            or not isinstance(owner_document.get("rows"), list)
        ):
            raise AggregateError("owner-selection JSON authority is invalid")
        owner_rows = owner_document["rows"]
    else:
        owner_rows = _read_table(owner_selection)
        if not owner_rows or {
            str(row.get("source_inventory_sha256") or "") for row in owner_rows
        } != {source_inventory_sha}:
            raise AggregateError("owner-selection CSV authority is invalid")
    source_candidate_ids = {
        str(row.get("candidate_id"))
        for row in matching_inventory.get("rows", [])
        if isinstance(row, Mapping)
    }
    owner_candidate_ids = {
        str(row.get("candidate_id"))
        for row in owner_rows
        if isinstance(row, Mapping)
    }
    if owner_candidate_ids != source_candidate_ids:
        raise AggregateError("owner selection does not preserve every inventory candidate")
    recorded = _require_sha256(document.get("manifest_sha256"), "manifest_sha256")
    measured = _semantic_digest(document, ("manifest_sha256",))
    if recorded != measured:
        raise AggregateError(
            f"selected manifest internal digest mismatch: recorded={recorded} measured={measured}"
        )
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AggregateError("selected manifest rows must be a nonempty array")
    required_row_fields = {
        "candidate_id", "map_id", "observation_number", "obsnum",
        "analysis_role", "reduction_id", "duplicate_group_id",
        "core_eligible", "enhanced_eligible",
    }
    by_observation: dict[int, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not required_row_fields.issubset(row):
            raise AggregateError(
                f"selected manifest row {index} does not match selected-manifest-v1"
            )
        role = str(row.get("analysis_role") or "").lower()
        if role not in {"primary", "sensitivity"}:
            raise AggregateError(f"selected manifest row {index} has invalid analysis_role")
        if not _bool(row.get("core_eligible"), False):
            raise AggregateError(f"selected manifest row {index} is not core eligible")
        obsnum = _int(row.get("obsnum"), field_name=f"row {index} obsnum")
        observation_number = _int(
            row.get("observation_number"),
            field_name=f"row {index} observation_number",
        )
        if obsnum != observation_number:
            raise AggregateError(f"selected manifest row {index} observation identity mismatch")
        if str(row.get("map_id")) != str(row.get("candidate_id")):
            raise AggregateError(f"selected manifest row {index} map/candidate identity mismatch")
        by_observation.setdefault(obsnum, []).append(row)
    for obsnum, observation_rows in sorted(by_observation.items()):
        primary_count = sum(
            str(row["analysis_role"]).lower() == "primary"
            for row in observation_rows
        )
        if primary_count != 1:
            raise AggregateError(
                f"eligible observation {obsnum} requires exactly one primary; "
                f"found {primary_count}"
            )
    source_rows = matching_inventory.get("rows")
    if not isinstance(source_rows, list):
        raise AggregateError("source inventory authority lacks candidate rows")
    eligible_source_rows = [
        row for row in source_rows
        if isinstance(row, Mapping) and _bool(row.get("core_eligible"), False)
    ]
    eligible_observations = {
        _int(row.get("observation_number"), field_name="source inventory observation_number")
        for row in eligible_source_rows
    }
    if set(by_observation) != eligible_observations:
        raise AggregateError(
            "selected manifest does not contain exactly one primary for every "
            "core-eligible source-inventory observation"
        )
    eligible_candidate_ids = {
        str(row.get("candidate_id")) for row in eligible_source_rows
    }
    manifest_candidate_ids = {str(row["candidate_id"]) for row in rows}
    if manifest_candidate_ids != eligible_candidate_ids:
        raise AggregateError(
            "selected manifest does not preserve every core-eligible source-inventory candidate"
        )
    # Bind the validated owner-selection digest explicitly; it is otherwise
    # used only as opaque provenance by aggregation.
    assert owner_selection_sha
    return dict(document)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    document = _selected_manifest_document(path)
    return [dict(row) for row in document["rows"]]


def _validate_manifest_internal_digest(path: Path) -> None:
    _selected_manifest_document(path)


def _inventory_for_selected_manifest(path: Path, source_inventory_sha256: str) -> dict[str, Any]:
    for candidate in (path.parent / "candidate_inventory.json", path.parent.parent / "candidate_inventory.json"):
        if not candidate.is_file():
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == INVENTORY_SCHEMA
            and document.get("inventory_sha256") == source_inventory_sha256
            and _semantic_digest(document, ("inventory_sha256",)) == source_inventory_sha256
        ):
            return document
    raise AggregateError("cannot locate checksum-bound source inventory for known omissions")


def known_omissions(
    inventory: Mapping[str, Any],
    bundles: Sequence[MapBundle],
    combined_network_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authoritative = inventory.get("authoritative_obsnum_status", [])
    status_rows = [dict(row) for row in authoritative if isinstance(row, Mapping)]
    deficiencies = [
        row for row in status_rows
        if row.get("status") != "eligible_canonical_candidate_found"
    ]
    missing_networks = [
        dict(row) for row in combined_network_rows
        if str(row.get("status")) == "missing_network"
    ]
    incomplete = [
        {
            "map_id": bundle.map_id,
            "observation_number": bundle.obsnum,
            "status": bundle.summary.get("status"),
        }
        for bundle in bundles
        if str(bundle.summary.get("status")) != "success"
    ]
    raw_unavailable = [
        {
            "map_id": bundle.map_id,
            "observation_number": bundle.obsnum,
            "network_id": row.network_id,
            "reason": "raw_phase_or_counter_metadata_unavailable",
        }
        for bundle in bundles
        for row in bundle.rows
        if row.raw_linkage_status not in {"proved_original_row_one_to_one", "config_proven"}
    ]
    return {
        "schema_version": "sci-align-001-3c273-known-omissions-v1",
        "authoritative_obsnum_deficiencies": deficiencies,
        "ambiguous_duplicates_awaiting_owner_selection": [
            row for row in deficiencies
            if row.get("status") == "ambiguous_duplicate_requires_owner_review"
        ],
        "missing_network_rows": missing_networks,
        "unavailable_raw_metadata": raw_unavailable,
        "failed_or_incomplete_per_map_tasks": incomplete,
        "deliberately_skipped_sensitivity_duplicates": [],
        "intentional_compact_archive_exclusions": [
            "raw timestreams are not included",
            "retained beammap reduction products are not included",
        ],
        "timing_result_used_as_cut": False,
    }


def _normalize_timestamp(value: Any) -> tuple[str | None, str | None]:
    if value in (None, ""):
        return None, None
    token = str(value).strip()
    date = None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        token = parsed.isoformat().replace("+00:00", "Z")
        date = parsed.date().isoformat()
    except ValueError:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})", token)
        if match:
            date = match.group(1)
    return token, date


def _complete_t0_key(row: Mapping[str, Any]) -> str | None:
    """Return a session key only for a complete ordered integer T0 vector.

    ClockTimeNanoSec/column 5 is intentionally not accepted as phase authority.
    """
    value = _first(row, (
        "network_t0_vector", "network_t0_vector_json",
        "t0_clocktime_vector", "clocktime_col0_vector", "t0_vector",
    ))
    status = _first(row, ("network_t0_status",))
    complete = (
        str(status) == "complete_unambiguous"
        if status not in (None, "")
        else _bool(_first(row, ("t0_vector_complete", "clocktime_vector_complete")), False)
    )
    if value in (None, "") or not complete:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list) or not value:
        return None
    normalized: list[dict[str, int]] = []
    for position, item in enumerate(value):
        if isinstance(item, Mapping):
            network_value = _first(item, ("network", "network_id"))
            t0_value = _first(item, ("t0", "clock_time", "clocktime_col0"))
        else:
            network_value = position
            t0_value = item
        if isinstance(network_value, bool) or isinstance(t0_value, bool):
            return None
        try:
            network = int(network_value)
            integer = int(t0_value)
        except (TypeError, ValueError):
            return None
        if not isinstance(t0_value, int):
            try:
                if float(t0_value) != integer:
                    return None
            except (TypeError, ValueError):
                return None
        normalized.append({"network": network, "t0": integer})
    networks = [item["network"] for item in normalized]
    if networks != sorted(networks) or len(networks) != len(set(networks)):
        return None
    digest = hashlib.sha256(_canonical_json(normalized).encode("ascii")).hexdigest()
    supplied = _first(row, ("network_t0_vector_sha256",))
    if supplied not in (None, "") and str(supplied).lower() != digest:
        return None
    return "roach-t0:" + digest[:20]


@dataclass(frozen=True)
class ManifestRow:
    map_id: str
    obsnum: int
    reduction_id: str
    observation_start_utc: str | None
    observing_date: str | None
    session_id: str | None
    session_status: str
    t0_session_key: str | None
    duplicate_group_id: str
    analysis_role: str
    core_eligible: bool
    enhanced_eligible: bool


def normalize_manifest(rows: Sequence[Mapping[str, Any]]) -> list[ManifestRow]:
    result: list[ManifestRow] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        obsnum = _int(_first(row, ("obsnum", "observation_number", "observation")), field_name="obsnum")
        reduction_id = str(_first(row, ("reduction_id", "candidate_id", "reduction_path", "project_path"), f"row-{index:06d}"))
        map_id = str(_first(row, ("map_id", "candidate_id"), f"o{obsnum}-{hashlib.sha256(reduction_id.encode()).hexdigest()[:12]}"))
        if map_id in seen:
            raise AggregateError(f"duplicate map_id in selected manifest: {map_id}")
        seen.add(map_id)
        start, derived_date = _normalize_timestamp(_first(row, ("observation_start_utc", "start_utc", "date_time", "date")))
        date = str(_first(row, ("observing_date", "observation_date"), derived_date or "")) or None
        session_id_raw = _first(row, ("session_id", "initialization_session_id", "roach_session_id"))
        session_id = str(session_id_raw) if session_id_raw not in (None, "") else None
        session_status = str(_first(row, ("session_status", "initialization_session_status"), "unavailable")).lower()
        if session_status in {"ambiguous", "unknown", "unavailable", "missing", "incomplete", "conflict"}:
            session_id = None
        # Inventory date fallback is not an independent initialization session.
        # A complete network-T0 vector is handled by t0_session_key above; do
        # not relabel either representation as generic provenance-session data.
        if session_status in {"date_group_fallback", "network_t0_vector"}:
            session_id = None
        selection = str(_first(row, ("analysis_role", "selection_status", "canonical_status"), "")).lower()
        sensitivity_roles = {"duplicate", "duplicate_sensitivity", "sensitivity"}
        selected = _bool(
            _first(row, ("selected", "canonical_selected", "is_canonical")),
            selection not in sensitivity_roles | {"excluded"},
        )
        role = (
            "primary" if selected
            else "duplicate_sensitivity" if selection in sensitivity_roles
            else "excluded"
        )
        core = _bool(_first(row, ("core_eligible", "reduction_only_eligible", "core")), False)
        enhanced = _bool(_first(row, ("enhanced_eligible", "raw_timestamp_eligible", "enhanced")), False)
        if role == "primary" and not core:
            role = "excluded"
        result.append(ManifestRow(
            map_id=map_id,
            obsnum=obsnum,
            reduction_id=reduction_id,
            observation_start_utc=start,
            observing_date=date,
            session_id=session_id,
            session_status=session_status,
            t0_session_key=_complete_t0_key(row),
            duplicate_group_id=str(_first(row, ("duplicate_group_id",), f"obs:{obsnum}")),
            analysis_role=role,
            core_eligible=core,
            enhanced_eligible=enhanced,
        ))
    primary = [row for row in result if row.analysis_role == "primary"]
    if not primary:
        raise AggregateError("selected manifest has no primary core-eligible maps")
    primary_by_obs: dict[int, int] = {}
    for row in primary:
        primary_by_obs[row.obsnum] = primary_by_obs.get(row.obsnum, 0) + 1
    duplicates = [obs for obs, count in primary_by_obs.items() if count > 1]
    if duplicates:
        raise AggregateError(f"multiple primary reductions remain for observations {duplicates}")
    return sorted(result, key=lambda row: (row.obsnum, row.map_id, row.reduction_id))


def freeze_partition(
    manifest_path: Path,
    *,
    protocol_template: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    _validate_manifest_internal_digest(manifest_path)
    rows = normalize_manifest(_read_manifest(manifest_path))
    primary = [row for row in rows if row.analysis_role == "primary"]

    t0_values = [row.t0_session_key for row in primary]
    ordinary_sessions = [row.session_id for row in primary]
    dates = [row.observing_date for row in primary]
    if all(t0_values) and len(set(t0_values)) >= 3:
        grouping = "t0_clocktime_vector"
        primary_keys = {row.obsnum: str(row.t0_session_key) for row in primary}
    elif all(ordinary_sessions) and len(set(ordinary_sessions)) >= 3:
        grouping = "provenance_session"
        primary_keys = {row.obsnum: f"session:{row.session_id}" for row in primary}
    elif all(dates) and len(set(dates)) >= 3:
        grouping = "observing_date"
        primary_keys = {row.obsnum: f"date:{row.observing_date}" for row in primary}
    else:
        grouping = "observation_number"
        primary_keys = {row.obsnum: f"obsnum:{row.obsnum}" for row in primary}

    registry: list[dict[str, Any]] = []
    for row in rows:
        group_id = primary_keys.get(row.obsnum, f"obsnum:{row.obsnum}")
        registry.append({
            "map_id": row.map_id,
            "obsnum": row.obsnum,
            "reduction_id": row.reduction_id,
            "analysis_role": row.analysis_role,
            "duplicate_group_id": row.duplicate_group_id,
            "observation_start_utc": row.observation_start_utc,
            "observing_date": row.observing_date,
            "session_id": row.session_id,
            "session_status": row.session_status,
            "t0_session_key": row.t0_session_key,
            "grouping_kind": grouping,
            "validation_group_id": group_id,
            "fold_id": group_id,
            "core_eligible": row.core_eligible,
            "enhanced_eligible": row.enhanced_eligible,
        })
    registry.sort(key=lambda row: (str(row["validation_group_id"]), int(row["obsnum"]), str(row["map_id"])))
    primary_groups = sorted({str(row["validation_group_id"]) for row in registry if row["analysis_role"] == "primary"})
    template = dict(protocol_template or {})
    protocol: dict[str, Any] = {
        **template,
        # Preserve the runner's corpus-protocol schema when a protocol template
        # is augmented in place.  The separate freeze schema records the
        # aggregation extension without making the retained-product runner
        # reject its own authority document.
        "schema_version": str(template.get("schema_version", PROTOCOL_VERSION)),
        "aggregation_freeze_schema_version": PROTOCOL_VERSION,
        "selected_manifest_path_recorded": str(manifest_path),
        "selected_manifest_sha256": sha256_file(manifest_path),
        "timing_results_inspected_during_freeze": False,
        "grouping_kind": grouping,
        "grouping_priority": [
            "complete ordered integer ClockTime(col0) T0 vector with >=3 groups",
            "unambiguous provenance session with >=3 groups",
            "trusted UTC observing date with >=3 groups",
            "observation number",
        ],
        "clocktime_nanosecond_column_is_phase_authority": False,
        "independent_group_count": len(primary_groups),
        "independent_groups": primary_groups,
        "validation_strategy": "leave_one_frozen_group_out",
        "session_anchor_strategy": "target excluded; only earlier maps in target session; other sessions may train the global component",
        "alpha": float(template.get("alpha", DEFAULT_ALPHA)),
        "candidate_models": [dict(item) for item in MODEL_REGISTRY],
        "candidate_model_registry_sha256": _candidate_registry_digest(),
        "aggregation_tool": _aggregation_tool_identity(),
        "classification_precedence": ["G", "A", "B", "D", "C", "E", "F", "G"],
        "terminology": {
            "half_change": "within-observation timing variation",
            "clock_drift_claim": "forbidden unless raw counter evidence contradicts the shared-clock account",
        },
        "producer_authority": {
            "t0": "integer-second ROACH-initialization label",
            "clock": "shared Octo 10 MHz/PPS; PPS sampled at detector cadence and does not reset sample cadence",
            "phase": "per-network integration phase may differ and may be stable",
            "t0_vector": "ordered integer ClockTime column 0 only",
            "native_frame_phase_formula": "((uint32 ClockCount[j] - uint32 PpsTime[j]) mod 2**32) / Header.Toltec.FpgaFreq at the paired PpsTime-transition row j when ordered one-to-one pairing is unambiguous",
            "pps_count_row_geometry": "recorded separately; it is not native-frame phase authority",
            "ambiguous_pps_pairing": "native frame phase unavailable",
        },
        "production_correction_authorized": False,
        "science_tolerance_assessed": False,
    }
    protocol["partition_rows"] = registry
    protocol["protocol_sha256"] = _semantic_digest(protocol, ("protocol_sha256",))
    return protocol, registry


def _validate_frozen_tooling(protocol: Mapping[str, Any]) -> dict[str, Any]:
    frozen_registry = protocol.get("candidate_models")
    if not isinstance(frozen_registry, list) or not frozen_registry:
        raise AggregateError("frozen protocol lacks the candidate-model registry")
    frozen_registry_digest = str(protocol.get("candidate_model_registry_sha256", ""))
    calculated_frozen_digest = _candidate_registry_digest(
        [item for item in frozen_registry if isinstance(item, Mapping)]
    )
    if len(frozen_registry) != len([
        item for item in frozen_registry if isinstance(item, Mapping)
    ]) or frozen_registry_digest != calculated_frozen_digest:
        raise AggregateError("frozen candidate-model registry digest mismatch")
    current_registry_digest = _candidate_registry_digest()
    if frozen_registry_digest != current_registry_digest:
        raise AggregateError("current candidate-model registry differs from frozen protocol")
    if _canonical_json(frozen_registry) != _canonical_json([
        dict(item) for item in MODEL_REGISTRY
    ]):
        raise AggregateError("current candidate-model registry content differs from frozen protocol")

    frozen_tool = protocol.get("aggregation_tool")
    if not isinstance(frozen_tool, Mapping):
        raise AggregateError("frozen protocol lacks aggregation-tool identity")
    current_tool = _aggregation_tool_identity()
    if frozen_tool.get("script_name") != current_tool["script_name"]:
        raise AggregateError("aggregation-tool filename differs from frozen protocol")
    if frozen_tool.get("script_sha256") != current_tool["script_sha256"]:
        raise AggregateError("current aggregation tool differs from frozen protocol")
    if frozen_tool.get("candidate_model_registry_sha256") != current_registry_digest:
        raise AggregateError("frozen aggregation-tool registry binding mismatch")
    return current_tool


def _freeze_output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        return output, output.with_name("session_registry.csv")
    return output / "frozen_analysis_protocol.json", output / "session_registry.csv"


def command_freeze(args: argparse.Namespace) -> int:
    _verify_checksum_file(args.selected_manifest.resolve().parent)
    template = json.loads(args.protocol_template.read_text()) if args.protocol_template else None
    if template is not None:
        template["runner_protocol_template_sha256"] = sha256_file(args.protocol_template.resolve())
    protocol, registry = freeze_partition(args.selected_manifest, protocol_template=template)
    protocol_path, registry_path = _freeze_output_paths(args.output)
    if args.dry_run:
        print(json.dumps({
            "action": "freeze",
            "would_write": [str(protocol_path), str(registry_path)],
            "grouping_kind": protocol["grouping_kind"],
            "independent_group_count": protocol["independent_group_count"],
            "protocol_sha256": protocol["protocol_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    write_json(protocol_path, protocol)
    write_csv(registry_path, registry, (
        "map_id", "obsnum", "reduction_id", "analysis_role", "duplicate_group_id",
        "observation_start_utc", "observing_date", "session_id", "session_status",
        "t0_session_key", "grouping_kind", "validation_group_id", "fold_id",
        "core_eligible", "enhanced_eligible",
    ))
    write_checksums(protocol_path.parent, (protocol_path, registry_path))
    print(str(protocol_path))
    return 0


def _verify_checksum_file(directory: Path) -> None:
    checksum = directory / "SHA256SUMS"
    if not checksum.is_file():
        raise AggregateError(f"checksum-bound package lacks SHA256SUMS: {directory}")
    for line in checksum.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise AggregateError(f"malformed checksum line in {checksum}: {line!r}") from error
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AggregateError(f"package checksum mismatch: {path}")


def _discover_map_directories(roots: Sequence[Path]) -> list[Path]:
    directories: set[Path] = set()
    for original in roots:
        root = original.resolve()
        if not root.exists():
            raise AggregateError(f"map output root does not exist: {root}")
        if root.is_file():
            if root.name not in {"map_result.json", "map_summary.json", "map_summary.csv"}:
                raise AggregateError(f"unsupported compact map file: {root}")
            directories.add(root.parent)
            continue
        if any((root / name).is_file() for name in ("map_result.json", "map_summary.json", "map_summary.csv")):
            directories.add(root)
        for name in ("map_result.json", "map_summary.json", "map_summary.csv"):
            for path in root.rglob(name):
                directories.add(path.parent.resolve())
    if not directories:
        raise AggregateError("no compact Stage-2 map outputs found")
    return sorted(directories, key=str)


def _read_optional_csv(directory: Path, name: str) -> list[dict[str, Any]]:
    path = directory / name
    return _read_table(path) if path.is_file() else []


def _load_summary(directory: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("summary"), dict):
        return dict(result["summary"])
    json_path = directory / "map_summary.json"
    if json_path.is_file():
        value = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AggregateError(f"map_summary.json is not an object: {json_path}")
        return dict(value)
    csv_path = directory / "map_summary.csv"
    if csv_path.is_file():
        rows = _read_table(csv_path)
        if len(rows) != 1:
            raise AggregateError(f"map_summary.csv must have exactly one row: {csv_path}")
        return rows[0]
    raise AggregateError(f"compact map output lacks a summary: {directory}")


def _result_document(directory: Path) -> dict[str, Any]:
    path = directory / "map_result.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregateError(f"map_result.json is not an object: {path}")
    return dict(value)


def _extract_digest(document: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for container in (
        document,
        document.get("identity", {}),
        document.get("provenance", {}),
        document.get("binding", {}),
    ):
        if not isinstance(container, Mapping):
            continue
        value = _first(container, names)
        if value not in (None, ""):
            return str(value)
    return None


def _protocol_authority_digest(directory: Path, result: Mapping[str, Any]) -> str | None:
    direct = _extract_digest(result, (
        "frozen_protocol_sha256", "protocol_sha256", "analysis_protocol_sha256",
    ))
    protocol = result.get("protocol", {})
    if isinstance(protocol, Mapping):
        direct = str(_first(protocol, (
            "authority_document_sha256", "frozen_protocol_sha256", "protocol_sha256",
        ), direct or "")) or direct
    binding_path = directory / "resume_binding.json"
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        nested = binding.get("protocol", {}) if isinstance(binding, dict) else {}
        if isinstance(nested, Mapping):
            direct = str(_first(nested, ("authority_document_sha256", "protocol_sha256"), direct or "")) or direct
            analysis_protocol = nested.get("analysis_protocol", {})
            if isinstance(analysis_protocol, Mapping):
                direct = str(_first(
                    analysis_protocol,
                    ("authority_document_sha256", "protocol_sha256"),
                    direct or "",
                )) or direct
    return direct


@dataclass
class NetworkDatum:
    map_id: str
    obsnum: int
    group_id: str
    session_id: str | None
    start_utc: str | None
    network_id: int
    array_name: str | None
    timing_sec: float
    timing_se_sec: float
    slot_sec: float | None
    slot_se_sec: float
    phase_sec: float | None
    phase_se_sec: float
    speed_arcsec_s: float | None
    parallel_fwhm_arcsec: float | None
    counter_anomaly: bool
    raw_linkage_status: str
    source_row: dict[str, Any] = field(repr=False)


@dataclass
class MapBundle:
    directory: Path
    map_id: str
    obsnum: int
    role: str
    group_id: str
    session_id: str | None
    start_utc: str | None
    summary: dict[str, Any]
    rows: list[NetworkDatum]
    covariance: np.ndarray
    covariance_source: str
    phase_rows: list[dict[str, Any]]
    timing_rows: list[dict[str, Any]]
    input_files: list[dict[str, str]]


def _raw_phase_lookup(directory: Path, result: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    values = result.get("raw_phase_summary")
    if not isinstance(values, list):
        values = _read_optional_csv(directory, "raw_phase_summary.csv")
    lookup: dict[int, dict[str, Any]] = {}
    for item in values or []:
        if not isinstance(item, Mapping):
            continue
        network = _int(_first(item, ("network_id", "network")), field_name="raw phase network_id")
        if network in lookup:
            raise AggregateError(f"duplicate raw phase row for network {network} in {directory}")
        lookup[network] = dict(item)
    return lookup


def _counter_anomaly(row: Mapping[str, Any]) -> bool:
    positive_counts = (
        "pps_spacing_other_count",
        "repeat_128_interval_mismatch_count",
        "clock_increment_mismatch_count",
        "packet_increment_mismatch_count",
        "pps_time_increment_mismatch_count",
        "pps_time_transition_offset_other_count",
    )
    for name in positive_counts:
        value = _float(row.get(name), field_name=name, optional=True)
        if value is not None and value > 0:
            return True
    return _bool(row.get("variable_metadata_capture_or_isr_latency_observed"), False)


def _network_rows(directory: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = result.get("network_results")
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    return _read_optional_csv(directory, "network_map_results.csv")


def _timing_rows(directory: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = result.get("timing_models")
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    return _read_optional_csv(directory, "timing_phase_results.csv")


def _parallel_fwhm(row: Mapping[str, Any]) -> float | None:
    direct = _float(_first(row, ("parallel_fwhm_arcsec",)), field_name="parallel_fwhm_arcsec", optional=True)
    if direct is not None and direct > 0:
        return direct
    widths = [
        _float(row.get(name), field_name=name, optional=True)
        for name in (
            "left_major_fwhm_arcsec", "left_minor_fwhm_arcsec",
            "right_major_fwhm_arcsec", "right_minor_fwhm_arcsec",
        )
    ]
    finite = [value for value in widths if value is not None and value > 0]
    return float(np.mean(finite)) if finite else None


def _load_covariance(
    directory: Path,
    rows: Sequence[NetworkDatum],
) -> tuple[np.ndarray, str]:
    size = len(rows)
    covariance = np.diag([row.timing_se_sec**2 for row in rows])
    table = _read_optional_csv(directory, "measurement_covariance.csv")
    if not table:
        return covariance, "diagonal_from_timing_se"
    index = {row.network_id: position for position, row in enumerate(rows)}
    filled: dict[tuple[int, int], float] = {}
    for item in table:
        left = _int(_first(item, ("network_i", "network_id_i")), field_name="network_i")
        right = _int(_first(item, ("network_j", "network_id_j")), field_name="network_j")
        if left not in index or right not in index:
            continue
        value = _float(_first(item, ("covariance_sec2", "covariance")), field_name="covariance_sec2")
        assert value is not None
        key = tuple(sorted((left, right)))
        if key in filled and not math.isclose(filled[key], value, rel_tol=1e-10, abs_tol=1e-20):
            raise AggregateError(f"conflicting covariance entry {key} in {directory}")
        filled[key] = value
        covariance[index[left], index[right]] = value
        covariance[index[right], index[left]] = value
    missing_diagonal = [row.network_id for row in rows if (row.network_id, row.network_id) not in filled]
    if missing_diagonal:
        raise AggregateError(f"covariance table lacks diagonals for networks {missing_diagonal} in {directory}")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-20):
        raise AggregateError(f"measurement covariance is not symmetric in {directory}")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = max(1e-20, 1e-10 * float(np.max(np.diag(covariance))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise AggregateError(f"measurement covariance is not positive semidefinite in {directory}")
    if float(np.min(eigenvalues)) <= 0:
        covariance += np.eye(size) * (tolerance - float(np.min(eigenvalues)))
        return covariance, "jackknife_psd_roundoff_floor"
    return covariance, "jackknife_full"


def _effective_session_id(registry_row: Mapping[str, Any]) -> str | None:
    t0 = registry_row.get("t0_session_key")
    if t0 not in (None, ""):
        return str(t0)
    session = registry_row.get("session_id")
    return str(session) if session not in (None, "") else None


def load_map_bundle(
    directory: Path,
    registry_row: Mapping[str, Any],
    *,
    manifest_sha256: str,
    protocol_file_sha256: str,
    protocol_semantic_sha256: str,
    runner_protocol_sha256: str | None = None,
) -> MapBundle:
    _verify_checksum_file(directory)
    result = _result_document(directory)
    summary = _load_summary(directory, result)
    expected_map_schema = "sci-align-001-3c273-map-result-v1"
    if result and result.get("schema") != expected_map_schema:
        raise AggregateError(
            f"unsupported compact map result schema in {directory}: {result.get('schema')!r}"
        )
    summary_schema = _first(summary, ("schema", "schema_version"))
    if summary_schema not in (None, "", expected_map_schema):
        raise AggregateError(
            f"unsupported compact map summary schema in {directory}: {summary_schema!r}"
        )
    map_id = str(_first(summary, ("map_id", "candidate_id"), _first(result.get("identity", {}) if isinstance(result.get("identity"), Mapping) else {}, ("candidate_id",), "")))
    expected_map_id = str(registry_row["map_id"])
    if map_id != expected_map_id:
        raise AggregateError(f"map identity mismatch: expected {expected_map_id}, found {map_id} in {directory}")
    obsnum = _int(_first(summary, ("obsnum", "observation_number", "observation")), field_name="observation_number")
    if obsnum != int(registry_row["obsnum"]):
        raise AggregateError(f"observation identity mismatch for {map_id}")
    accepted_status = {
        "success", "available", "accepted",
        "partial_core_success_enhanced_failed",
    }
    if not _bool(_first(summary, ("quality", "available")), True) or str(summary.get("status", "success")).lower() not in accepted_status:
        raise AggregateError(f"primary map result is not quality-accepted: {map_id}")

    recorded_protocol = _protocol_authority_digest(directory, result)
    accepted_protocol = {protocol_file_sha256, protocol_semantic_sha256}
    if runner_protocol_sha256:
        accepted_protocol.add(runner_protocol_sha256)
    if recorded_protocol not in accepted_protocol:
        raise AggregateError(
            f"frozen protocol digest missing/mismatched for {map_id}: {recorded_protocol!r}"
        )
    recorded_manifest = _extract_digest(result, ("selected_manifest_sha256", "manifest_sha256"))
    binding_path = directory / "resume_binding.json"
    if recorded_manifest is None and binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        recorded_manifest = _extract_digest(binding, ("selected_manifest_sha256", "manifest_sha256"))
    if recorded_manifest != manifest_sha256:
        raise AggregateError(
            f"selected-manifest digest missing/mismatched for {map_id}: {recorded_manifest!r}"
        )

    phase_lookup = _raw_phase_lookup(directory, result)
    network_values = _network_rows(directory, result)
    normalized: list[NetworkDatum] = []
    seen_networks: set[int] = set()
    for item in network_values:
        if not _bool(_first(item, ("available", "quality")), False):
            continue
        network = _int(_first(item, ("network_id", "network")), field_name="network_id")
        if network in seen_networks:
            raise AggregateError(f"duplicate network {network} for map {map_id}")
        seen_networks.add(network)
        timing = _float(_first(item, NETWORK_TIMING_ALIASES), field_name="timing_residual_sec")
        timing_se = _float(_first(item, NETWORK_SE_ALIASES), field_name="timing_se_sec")
        assert timing is not None and timing_se is not None
        if timing_se <= 0:
            raise AggregateError(f"timing_se_sec must be positive for {map_id}/network {network}")
        phase = dict(phase_lookup.get(network, {}))
        merged = {**phase, **item}
        slot = _float(_first(merged, SLOT_ALIASES), field_name="native_to_assigned_slot_residual_sec", optional=True)
        slot_se = _float(_first(merged, ("native_to_assigned_slot_residual_se_sec", "slot_residual_se_sec")), field_name="slot_residual_se_sec", optional=True) or 0.0
        native_phase = _float(_first(merged, PHASE_ALIASES), field_name="native_frame_phase_mean_sec", optional=True)
        phase_se = _float(_first(merged, ("native_frame_phase_se_sec",)), field_name="native_frame_phase_se_sec", optional=True)
        if phase_se is None:
            phase_std = _float(merged.get("native_frame_phase_std_sec"), field_name="native_frame_phase_std_sec", optional=True)
            count = _float(merged.get("pps_transition_count"), field_name="pps_transition_count", optional=True)
            phase_se = (phase_std / math.sqrt(count)) if phase_std is not None and count and count > 0 else 0.0
        speed = _float(_first(item, ("scan_speed_abs_arcsec_s", "scan_speed_abs_median_arcsec_s", "scan_speed_p50_arcsec_s")), field_name="scan_speed_abs_arcsec_s", optional=True)
        linkage = str(_first(merged, ("raw_linkage_status",), "unavailable"))
        normalized.append(NetworkDatum(
            map_id=map_id,
            obsnum=obsnum,
            group_id=str(registry_row["validation_group_id"]),
            session_id=_effective_session_id(registry_row),
            start_utc=str(registry_row["observation_start_utc"]) if registry_row.get("observation_start_utc") not in (None, "") else None,
            network_id=network,
            array_name=str(_first(item, ("array_name", "array"), "")) or None,
            timing_sec=timing,
            timing_se_sec=timing_se,
            slot_sec=slot,
            slot_se_sec=slot_se,
            phase_sec=native_phase,
            phase_se_sec=phase_se,
            speed_arcsec_s=speed,
            parallel_fwhm_arcsec=_parallel_fwhm(item),
            counter_anomaly=_counter_anomaly(merged),
            raw_linkage_status=linkage,
            source_row=merged,
        ))
    normalized.sort(key=lambda row: row.network_id)
    if not normalized:
        raise AggregateError(f"map has no accepted network timing results: {map_id}")
    covariance, covariance_source = _load_covariance(directory, normalized)
    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file():
            files.append({"map_id": map_id, "path": str(path), "sha256": sha256_file(path)})
    return MapBundle(
        directory=directory,
        map_id=map_id,
        obsnum=obsnum,
        role=str(registry_row["analysis_role"]),
        group_id=str(registry_row["validation_group_id"]),
        session_id=_effective_session_id(registry_row),
        start_utc=str(registry_row["observation_start_utc"]) if registry_row.get("observation_start_utc") not in (None, "") else None,
        summary=summary,
        rows=normalized,
        covariance=covariance,
        covariance_source=covariance_source,
        phase_rows=[dict(row) for row in phase_lookup.values()],
        timing_rows=_timing_rows(directory, result),
        input_files=files,
    )


@dataclass
class FittedCandidate:
    model_id: str
    family: str
    specification: dict[str, Any]
    columns: list[str]
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    network_levels: list[int]
    session_levels: list[str]
    reference_network: int | None
    reference_session: str | None
    beta: float | None
    beta_se: float | None
    training_rows: int
    training_groups: int
    fit_status: str


def _specification(model_id: str) -> dict[str, Any]:
    for item in MODEL_REGISTRY:
        if item["id"] == model_id:
            return dict(item)
    raise AggregateError(f"unknown model {model_id}")


def _predictor_value(row: NetworkDatum, predictor: str | None) -> float | None:
    if predictor == "slot":
        return row.slot_sec
    if predictor == "phase":
        return row.phase_sec
    return None


def _predictor_se(row: NetworkDatum, predictor: str | None) -> float:
    if predictor == "slot":
        return row.slot_se_sec
    if predictor == "phase":
        return row.phase_se_sec
    return 0.0


def _applicable(row: NetworkDatum, spec: Mapping[str, Any]) -> bool:
    if spec.get("abstain"):
        return True
    if spec.get("session") and row.session_id is None:
        return False
    predictor = spec.get("predictor")
    return predictor is None or _predictor_value(row, str(predictor)) is not None


def _design_encoding(
    rows: Sequence[NetworkDatum],
    spec: Mapping[str, Any],
) -> tuple[list[str], list[int], list[str], int | None, str | None]:
    columns = ["intercept"]
    networks = sorted({row.network_id for row in rows}) if spec.get("network") else []
    sessions = sorted({str(row.session_id) for row in rows if row.session_id is not None}) if spec.get("session") else []
    reference_network = networks[0] if networks else None
    reference_session = sessions[0] if sessions else None
    columns.extend(f"network:{network}" for network in networks[1:])
    columns.extend(f"session:{session}" for session in sessions[1:])
    if spec.get("predictor") and "fixed_beta" not in spec:
        columns.append(f"predictor:{spec['predictor']}")
    return columns, networks, sessions, reference_network, reference_session


def _design_row(
    row: NetworkDatum,
    spec: Mapping[str, Any],
    columns: Sequence[str],
    networks: Sequence[int],
    sessions: Sequence[str],
) -> tuple[np.ndarray | None, str]:
    if not _applicable(row, spec):
        return None, "not_applicable"
    if spec.get("network") and row.network_id not in networks:
        return None, "unsupported_network_level"
    if spec.get("session") and str(row.session_id) not in sessions:
        return None, "unsupported_session_level"
    values = [1.0]
    values.extend(1.0 if row.network_id == network else 0.0 for network in networks[1:])
    values.extend(1.0 if str(row.session_id) == session else 0.0 for session in sessions[1:])
    if spec.get("predictor") and "fixed_beta" not in spec:
        predictor = _predictor_value(row, str(spec["predictor"]))
        if predictor is None:
            return None, "not_applicable"
        values.append(float(predictor))
    vector = np.asarray(values, dtype=float)
    if vector.size != len(columns):
        raise AggregateError("internal design-column mismatch")
    return vector, "supported"


def _measurement_matrix(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    predictor: str | None,
    beta: float,
) -> np.ndarray:
    size = len(rows)
    result = np.zeros((size, size), dtype=float)
    by_map: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_map.setdefault(row.map_id, []).append(index)
    for map_id, positions in by_map.items():
        bundle = bundles[map_id]
        local = {row.network_id: index for index, row in enumerate(bundle.rows)}
        for outer in positions:
            left = rows[outer]
            for inner in positions:
                right = rows[inner]
                result[outer, inner] = bundle.covariance[
                    local[left.network_id], local[right.network_id]
                ]
            result[outer, outer] += (beta * _predictor_se(left, predictor)) ** 2
    if np.any(np.diag(result) <= 0):
        raise AggregateError("measurement matrix has nonpositive diagonal")
    return result


def _gls(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    try:
        inverse_y = np.linalg.solve(covariance, y)
        inverse_x = np.linalg.solve(covariance, x)
    except np.linalg.LinAlgError as error:
        raise AggregateError("singular measurement covariance") from error
    normal = x.T @ inverse_x
    condition = float(np.linalg.cond(normal))
    if (
        np.linalg.matrix_rank(normal) != normal.shape[0]
        or not math.isfinite(condition)
        or condition > 1.0e12
    ):
        raise AggregateError("candidate design is rank deficient")
    coefficient_covariance = np.linalg.inv(normal)
    coefficients = coefficient_covariance @ (x.T @ inverse_y)
    if not (
        np.all(np.isfinite(coefficient_covariance))
        and np.all(np.isfinite(coefficients))
    ):
        raise AggregateError("candidate coefficient solution is non-finite")
    residual = y - x @ coefficients
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    return coefficients, coefficient_covariance, chi2, int(y.size - x.shape[1])


def fit_candidate(
    model_id: str,
    input_rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> FittedCandidate:
    spec = _specification(model_id)
    if spec.get("abstain"):
        return FittedCandidate(
            model_id, str(spec["family"]), spec, [], np.asarray([]), np.empty((0, 0)),
            [], [], None, None, None, None, 0, 0, "abstain",
        )
    rows = [row for row in input_rows if _applicable(row, spec)]
    columns, networks, sessions, reference_network, reference_session = _design_encoding(rows, spec)
    minimum = len(columns) + 1
    if len(rows) < minimum or len({row.group_id for row in rows}) < 2:
        return FittedCandidate(
            model_id, str(spec["family"]), spec, columns, np.asarray([]), np.empty((0, 0)),
            networks, sessions, reference_network, reference_session, None, None,
            len(rows), len({row.group_id for row in rows}), "insufficient_training_support",
        )
    vectors = []
    for row in rows:
        vector, status = _design_row(row, spec, columns, networks, sessions)
        if vector is None or status != "supported":
            raise AggregateError("training row unexpectedly unsupported")
        vectors.append(vector)
    x = np.vstack(vectors)
    fixed_beta = float(spec.get("fixed_beta", 0.0))
    predictor_name = str(spec["predictor"]) if spec.get("predictor") else None
    predictor_values = np.asarray([
        _predictor_value(row, predictor_name) or 0.0 for row in rows
    ])
    y = np.asarray([row.timing_sec for row in rows]) - fixed_beta * predictor_values
    beta_for_error = fixed_beta
    coefficients = np.asarray([])
    coefficient_covariance = np.empty((0, 0))
    try:
        for _ in range(4):
            covariance = _measurement_matrix(rows, bundles, predictor_name, beta_for_error)
            coefficients, coefficient_covariance, _, _ = _gls(x, y, covariance)
            if predictor_name and "fixed_beta" not in spec:
                beta_index = columns.index(f"predictor:{predictor_name}")
                new_beta = float(coefficients[beta_index])
                if math.isclose(new_beta, beta_for_error, rel_tol=1e-12, abs_tol=1e-15):
                    break
                beta_for_error = new_beta
        beta = fixed_beta if "fixed_beta" in spec else (
            float(coefficients[columns.index(f"predictor:{predictor_name}")])
            if predictor_name else None
        )
        beta_se = (
            0.0 if "fixed_beta" in spec and predictor_name
            else math.sqrt(max(0.0, coefficient_covariance[
                columns.index(f"predictor:{predictor_name}"),
                columns.index(f"predictor:{predictor_name}"),
            ])) if predictor_name else None
        )
        status = "fit"
    except AggregateError:
        return FittedCandidate(
            model_id, str(spec["family"]), spec, columns, np.asarray([]), np.empty((0, 0)),
            networks, sessions, reference_network, reference_session, None, None,
            len(rows), len({row.group_id for row in rows}), "rank_or_covariance_failure",
        )
    return FittedCandidate(
        model_id=model_id,
        family=str(spec["family"]),
        specification=spec,
        columns=columns,
        coefficients=coefficients,
        coefficient_covariance=coefficient_covariance,
        network_levels=networks,
        session_levels=sessions,
        reference_network=reference_network,
        reference_session=reference_session,
        beta=beta,
        beta_se=beta_se,
        training_rows=len(rows),
        training_groups=len({row.group_id for row in rows}),
        fit_status=status,
    )


def predict_candidate(
    fitted: FittedCandidate,
    row: NetworkDatum,
    *,
    regime: str,
    fold_id: str,
) -> dict[str, Any]:
    base = {
        "model_id": fitted.model_id,
        "model_family": fitted.family,
        "validation_regime": regime,
        "fold_id": fold_id,
        "map_id": row.map_id,
        "obsnum": row.obsnum,
        "group_id": row.group_id,
        "session_id": row.session_id,
        "network_id": row.network_id,
        "timing_observed_sec": row.timing_sec,
        "timing_measurement_se_sec": row.timing_se_sec,
        "predictor_kind": fitted.specification.get("predictor"),
        "predictor_sec": _predictor_value(row, fitted.specification.get("predictor")),
        "training_rows": fitted.training_rows,
        "training_groups": fitted.training_groups,
        "fitted_parameter_count": len(fitted.columns),
        "fitted_parameter_names": list(fitted.columns),
    }
    if fitted.fit_status != "fit":
        return {**base, "prediction_status": fitted.fit_status, "applicable": _applicable(row, fitted.specification), "supported": False}
    vector, status = _design_row(
        row, fitted.specification, fitted.columns,
        fitted.network_levels, fitted.session_levels,
    )
    if vector is None:
        return {**base, "prediction_status": status, "applicable": status != "not_applicable", "supported": False}
    fixed = 0.0
    predictor_name = fitted.specification.get("predictor")
    if "fixed_beta" in fitted.specification and predictor_name:
        predictor = _predictor_value(row, str(predictor_name))
        assert predictor is not None
        fixed = float(fitted.specification["fixed_beta"]) * predictor
    prediction = float(vector @ fitted.coefficients + fixed)
    parameter_variance = float(vector @ fitted.coefficient_covariance @ vector)
    beta = float(fitted.beta or 0.0)
    predictive_variance = (
        row.timing_se_sec**2 + parameter_variance
        + (beta * _predictor_se(row, str(predictor_name) if predictor_name else None)) ** 2
    )
    if predictive_variance <= 0 or not math.isfinite(predictive_variance):
        raise AggregateError("invalid predictive variance")
    residual = row.timing_sec - prediction
    return {
        **base,
        "prediction_status": "supported",
        "applicable": True,
        "supported": True,
        "timing_predicted_sec": prediction,
        "diagnostic_predicted_offset_sec": prediction,
        "timing_residual_after_prediction_sec": residual,
        "predictive_se_sec": math.sqrt(predictive_variance),
        "prediction_parameter_variance_sec2": parameter_variance,
        "standardized_residual": residual / math.sqrt(predictive_variance),
        "beta": fitted.beta,
        "beta_se": fitted.beta_se,
    }


def _predictive_block_metrics(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float | int]:
    if residual.ndim != 1 or covariance.shape != (residual.size, residual.size):
        raise AggregateError("predictive block has inconsistent shape")
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not math.isfinite(float(log_determinant)):
        raise AggregateError("predictive block covariance is not positive definite")
    try:
        mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    except np.linalg.LinAlgError as error:
        raise AggregateError("predictive block covariance is singular") from error
    count = int(residual.size)
    negative_log_predictive_density = 0.5 * (
        mahalanobis + float(log_determinant) + count * math.log(2.0 * math.pi)
    )
    return {
        "observation_count": count,
        "mahalanobis": mahalanobis,
        "dof": count,
        "pvalue": float(stats.chi2.sf(mahalanobis, count)),
        "log_determinant_sec2": float(log_determinant),
        "negative_log_predictive_density": negative_log_predictive_density,
        "negative_log_predictive_density_per_observation": (
            negative_log_predictive_density / count
        ),
    }


def predict_candidate_block(
    fitted: FittedCandidate,
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    *,
    regime: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    """Predict one held-out block and retain its joint covariance score.

    A block is the complete set predicted from one fitted model into one held-
    out fold or rolling target. Available within-map covariance is retained,
    maps are block-diagonal where no cross-map covariance exists, and the
    shared fitted-parameter covariance supplies cross-row prediction terms.
    """
    ordered = sorted(rows, key=lambda row: (row.obsnum, row.map_id, row.network_id))
    predictions = [
        predict_candidate(fitted, row, regime=regime, fold_id=fold_id)
        for row in ordered
    ]
    block_id = f"{regime}|{fold_id}|{fitted.model_id}"
    for prediction in predictions:
        prediction["predictive_block_id"] = block_id
    if not predictions or not all(_bool(row.get("supported"), False) for row in predictions):
        return predictions

    design_rows = []
    for row in ordered:
        vector, status = _design_row(
            row, fitted.specification, fitted.columns,
            fitted.network_levels, fitted.session_levels,
        )
        if vector is None or status != "supported":
            raise AggregateError("supported prediction lacks a design row")
        design_rows.append(vector)
    design = np.vstack(design_rows)
    predictor_name = (
        str(fitted.specification["predictor"])
        if fitted.specification.get("predictor") else None
    )
    beta = float(fitted.beta or 0.0)
    measurement_covariance = _measurement_matrix(
        ordered, bundles, predictor_name, beta,
    )
    maximum_design = max(1.0, float(np.max(np.abs(design))))
    maximum_coefficient_covariance = float(
        np.max(np.abs(fitted.coefficient_covariance))
    )
    parameter_count = max(1, fitted.coefficient_covariance.shape[0])
    log_product_bound = (
        2.0 * math.log(maximum_design)
        + math.log(maximum_coefficient_covariance)
        + math.log(parameter_count)
        if maximum_coefficient_covariance > 0 else -math.inf
    )
    if log_product_bound > math.log(np.finfo(float).max) - math.log(16.0):
        for prediction in predictions:
            prediction["supported"] = False
            prediction["prediction_status"] = (
                "prediction_parameter_covariance_not_numerically_representable"
            )
        return predictions
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            parameter_covariance = design @ fitted.coefficient_covariance @ design.T
    except FloatingPointError:
        for prediction in predictions:
            prediction["supported"] = False
            prediction["prediction_status"] = (
                "prediction_parameter_covariance_not_numerically_representable"
            )
        return predictions
    predictive_covariance = measurement_covariance + parameter_covariance
    residual = np.asarray([
        float(row["timing_residual_after_prediction_sec"])
        for row in predictions
    ])
    metrics = _predictive_block_metrics(residual, predictive_covariance)
    covariance_sources = sorted({bundles[row.map_id].covariance_source for row in ordered})
    annotation = {
        "predictive_block_observation_count": metrics["observation_count"],
        "predictive_block_mahalanobis": metrics["mahalanobis"],
        "predictive_block_dof": metrics["dof"],
        "predictive_block_pvalue": metrics["pvalue"],
        "predictive_block_log_determinant_sec2": metrics["log_determinant_sec2"],
        "predictive_block_negative_log_predictive_density": (
            metrics["negative_log_predictive_density"]
        ),
        "predictive_block_nlpd_per_observation": (
            metrics["negative_log_predictive_density_per_observation"]
        ),
        "predictive_block_parameter_covariance_trace_sec2": float(
            np.trace(parameter_covariance)
        ),
        "predictive_block_measurement_covariance_sources": covariance_sources,
        "predictive_covariance_accounting": (
            "full available within-map covariance plus predictor uncertainty "
            "and shared fitted-parameter prediction covariance; cross-map "
            "measurement covariance unavailable"
        ),
    }
    for prediction in predictions:
        prediction.update(annotation)
    return predictions


def heldout_predictions(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    groups = sorted({row.group_id for row in rows})
    for model in MODEL_REGISTRY:
        model_id = str(model["id"])
        if model.get("abstain"):
            for row in rows:
                predictions.append({
                    "model_id": model_id, "model_family": "M5",
                    "validation_regime": "outer_logo", "fold_id": row.group_id,
                    "map_id": row.map_id, "obsnum": row.obsnum,
                    "group_id": row.group_id, "session_id": row.session_id,
                    "network_id": row.network_id, "applicable": True,
                    "supported": False, "prediction_status": "abstain",
                    "timing_observed_sec": row.timing_sec,
                    "timing_measurement_se_sec": row.timing_se_sec,
                })
            continue
        for group in groups:
            train = [row for row in rows if row.group_id != group]
            test = [row for row in rows if row.group_id == group]
            fitted = fit_candidate(model_id, train, bundles)
            predictions.extend(predict_candidate_block(
                fitted, test, bundles, regime="outer_logo", fold_id=group,
            ))

    session_specs = [
        str(item["id"]) for item in MODEL_REGISTRY
        if item.get("session") and not item.get("abstain")
    ]
    maps_by_session: dict[str, list[str]] = {}
    for row in rows:
        if row.session_id is not None:
            maps_by_session.setdefault(row.session_id, []).append(row.map_id)
    map_order = {
        bundle.map_id: (bundle.start_utc or "", bundle.obsnum, bundle.map_id)
        for bundle in bundles.values()
    }
    for session, map_ids in sorted(maps_by_session.items()):
        ordered = sorted(set(map_ids), key=lambda map_id: map_order[map_id])
        for position, target_map in enumerate(ordered[1:], start=1):
            earlier = set(ordered[:position])
            target_rows = [row for row in rows if row.map_id == target_map]
            train = [
                row for row in rows
                if row.map_id != target_map
                and (row.session_id != session or row.map_id in earlier)
            ]
            for model_id in session_specs:
                fitted = fit_candidate(model_id, train, bundles)
                predictions.extend(predict_candidate_block(
                    fitted, target_rows, bundles,
                    regime="rolling_session_anchor",
                    fold_id=f"session-anchor:{session}:{target_map}",
                ))
    predictions.extend(nested_outer_predictions(rows, bundles))
    predictions.extend(nested_session_anchor_predictions(rows, bundles))
    return sorted(predictions, key=lambda row: (
        str(row["validation_regime"]), str(row["model_id"]),
        str(row["fold_id"]), int(row["obsnum"]), int(row["network_id"]),
    ))


def _supported_prediction_blocks(
    predictions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    blocks: dict[str, Mapping[str, Any]] = {}
    for row in predictions:
        if not _bool(row.get("supported"), False):
            continue
        block_id = str(row.get("predictive_block_id", ""))
        if not block_id or row.get("predictive_block_nlpd_per_observation") is None:
            raise AggregateError("supported held-out prediction lacks joint block metrics")
        previous = blocks.get(block_id)
        if previous is not None:
            for field_name in (
                "group_id", "predictive_block_observation_count",
                "predictive_block_mahalanobis", "predictive_block_dof",
                "predictive_block_pvalue",
                "predictive_block_nlpd_per_observation",
                "fitted_parameter_count",
            ):
                if previous.get(field_name) != row.get(field_name):
                    raise AggregateError(
                        f"inconsistent predictive block annotation for {block_id}"
                    )
        else:
            blocks[block_id] = row
    return [blocks[key] for key in sorted(blocks)]


def _prediction_parameter_complexity(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[int, float]:
    blocks = _supported_prediction_blocks(predictions)
    counts = [int(row["fitted_parameter_count"]) for row in blocks]
    if not counts:
        raise AggregateError("cannot determine fitted-parameter complexity")
    return max(counts), float(np.mean(counts))


def _prediction_score(
    predictions: Sequence[Mapping[str, Any]],
    expected_count: int,
) -> float | None:
    supported = [row for row in predictions if row.get("supported")]
    if len(supported) != expected_count:
        return None
    blocks = _supported_prediction_blocks(supported)
    by_group: dict[str, list[float]] = {}
    for row in blocks:
        by_group.setdefault(str(row["group_id"]), []).append(
            float(row["predictive_block_nlpd_per_observation"])
        )
    if not by_group:
        return None
    # Each frozen independent group has equal influence. Multiple rolling
    # targets inside one group are averaged before the corpus-level mean.
    return float(np.mean([np.mean(values) for _, values in sorted(by_group.items())]))


def _logo_for_model(
    model_id: str,
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    *,
    regime: str,
) -> list[dict[str, Any]]:
    output = []
    for group in sorted({row.group_id for row in rows}):
        train = [row for row in rows if row.group_id != group]
        test = [row for row in rows if row.group_id == group]
        fitted = fit_candidate(model_id, train, bundles)
        output.extend(predict_candidate_block(
            fitted, test, bundles, regime=regime, fold_id=group,
        ))
    return output


def nested_outer_predictions(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> list[dict[str, Any]]:
    """Select inside each outer training set, then score its untouched group."""
    groups = sorted({row.group_id for row in rows})
    if len(groups) < 4:
        return []
    candidates = [
        dict(item) for item in MODEL_REGISTRY
        if not item.get("abstain")
    ]
    output: list[dict[str, Any]] = []
    for outer_group in groups:
        outer_train = [row for row in rows if row.group_id != outer_group]
        outer_test = [row for row in rows if row.group_id == outer_group]
        ranking = []
        for spec in candidates:
            inner = _logo_for_model(
                str(spec["id"]), outer_train, bundles,
                regime=f"inner_for_outer:{outer_group}",
            )
            score = _prediction_score(inner, len(outer_train))
            if score is not None:
                parameter_max, parameter_mean = _prediction_parameter_complexity(inner)
                ranking.append((score, parameter_max, parameter_mean, str(spec["id"])))
        if not ranking:
            for row in outer_test:
                output.append({
                    "model_id": "NESTED_SELECTED", "model_family": "NESTED",
                    "validation_regime": "nested_outer_logo", "fold_id": outer_group,
                    "map_id": row.map_id, "obsnum": row.obsnum,
                    "group_id": row.group_id, "session_id": row.session_id,
                    "network_id": row.network_id, "applicable": True,
                    "supported": False, "prediction_status": "no_inner_supported_model",
                    "timing_observed_sec": row.timing_sec,
                    "timing_measurement_se_sec": row.timing_se_sec,
                })
            continue
        ranking.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        selected = ranking[0][3]
        fitted = fit_candidate(selected, outer_train, bundles)
        for prediction in predict_candidate_block(
            fitted, outer_test, bundles,
            regime="nested_outer_logo", fold_id=outer_group,
        ):
            prediction["selected_model_id"] = selected
            prediction["selected_model_family"] = fitted.family
            prediction["inner_selection_score"] = ranking[0][0]
            prediction["inner_selection_parameter_count_max"] = ranking[0][1]
            prediction["inner_selection_parameter_count_mean"] = ranking[0][2]
            prediction["model_id"] = "NESTED_SELECTED"
            prediction["model_family"] = "NESTED"
            output.append(prediction)
    return output


def _rolling_targets(rows: Sequence[NetworkDatum]) -> list[tuple[str, str, set[str], list[NetworkDatum]]]:
    map_meta: dict[str, tuple[str | None, int, str, str | None]] = {}
    for row in rows:
        map_meta[row.map_id] = (row.start_utc, row.obsnum, row.map_id, row.session_id)
    output = []
    sessions = sorted({row.session_id for row in rows if row.session_id is not None})
    for session in sessions:
        maps = sorted(
            {row.map_id for row in rows if row.session_id == session},
            key=lambda map_id: (map_meta[map_id][0] or "", map_meta[map_id][1], map_id),
        )
        for position, target in enumerate(maps[1:], start=1):
            output.append((str(session), target, set(maps[:position]), [row for row in rows if row.map_id == target]))
    return output


def _rolling_predictions_for_model(
    model_id: str,
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    *,
    excluded_target: str | None,
    regime: str,
) -> list[dict[str, Any]]:
    output = []
    for session, target, earlier, target_rows in _rolling_targets(rows):
        if target == excluded_target:
            continue
        train = [
            row for row in rows
            if row.map_id != target
            and (row.session_id != session or row.map_id in earlier)
        ]
        fitted = fit_candidate(model_id, train, bundles)
        output.extend(predict_candidate_block(
            fitted, target_rows, bundles, regime=regime,
            fold_id=f"session-anchor:{session}:{target}",
        ))
    return output


def nested_session_anchor_predictions(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> list[dict[str, Any]]:
    targets = _rolling_targets(rows)
    if len({item[0] for item in targets}) < 4:
        return []
    candidates = [
        dict(item) for item in MODEL_REGISTRY
        if item.get("session") and not item.get("abstain")
    ]
    output = []
    for session, target, earlier, target_rows in targets:
        outer_train = [
            row for row in rows
            if row.map_id != target
            and (row.session_id != session or row.map_id in earlier)
        ]
        ranking = []
        for spec in candidates:
            inner = _rolling_predictions_for_model(
                str(spec["id"]), outer_train, bundles,
                excluded_target=None,
                regime=f"inner_session_for:{target}",
            )
            expected = sum(len(item[3]) for item in _rolling_targets(outer_train))
            score = _prediction_score(inner, expected) if expected else None
            if score is not None:
                parameter_max, parameter_mean = _prediction_parameter_complexity(inner)
                ranking.append((score, parameter_max, parameter_mean, str(spec["id"])))
        if not ranking:
            continue
        ranking.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        selected = ranking[0][3]
        fitted = fit_candidate(selected, outer_train, bundles)
        for prediction in predict_candidate_block(
            fitted, target_rows, bundles,
            regime="nested_rolling_session_anchor",
            fold_id=f"session-anchor:{session}:{target}",
        ):
            prediction["selected_model_id"] = selected
            prediction["selected_model_family"] = fitted.family
            prediction["inner_selection_score"] = ranking[0][0]
            prediction["inner_selection_parameter_count_max"] = ranking[0][1]
            prediction["inner_selection_parameter_count_mean"] = ranking[0][2]
            prediction["model_id"] = "NESTED_SESSION_SELECTED"
            prediction["model_family"] = "NESTED_SESSION"
            output.append(prediction)
    return output


def _finite_values(rows: Sequence[Mapping[str, Any]], field_name: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(field_name)
        if value not in (None, ""):
            number = float(value)
            if math.isfinite(number):
                result.append(number)
    return result


def summarize_candidates(
    predictions: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault((str(row["model_id"]), str(row["validation_regime"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model_id, regime), values in sorted(grouped.items()):
        applicable = [row for row in values if _bool(row.get("applicable"), False)]
        supported = [row for row in applicable if _bool(row.get("supported"), False)]
        residual = np.asarray(_finite_values(supported, "timing_residual_after_prediction_sec"), dtype=float)
        groups = {str(row["group_id"]) for row in supported}
        blocks = _supported_prediction_blocks(supported) if supported else []
        if blocks:
            chi2 = float(sum(float(row["predictive_block_mahalanobis"]) for row in blocks))
            dof = int(sum(int(row["predictive_block_dof"]) for row in blocks))
            minimum_block_pvalue = min(
                float(row["predictive_block_pvalue"]) for row in blocks
            )
            pvalue = min(1.0, minimum_block_pvalue * len(blocks))
            score_by_group: dict[str, list[float]] = {}
            for row in blocks:
                score_by_group.setdefault(str(row["group_id"]), []).append(
                    float(row["predictive_block_nlpd_per_observation"])
                )
            group_aware_score = float(np.mean([
                np.mean(group_scores)
                for _, group_scores in sorted(score_by_group.items())
            ]))
            parameter_counts = [int(row["fitted_parameter_count"]) for row in blocks]
        else:
            chi2, dof, pvalue = None, 0, None
            minimum_block_pvalue = None
            group_aware_score = None
            parameter_counts = []
        complete_support = bool(applicable and len(supported) == len(applicable))
        passes = bool(
            model_id != "M5_ABSTAIN"
            and complete_support and len(groups) >= 3
            and pvalue is not None and pvalue >= alpha
        )
        beta_values = sorted({
            (str(row["fold_id"]), float(row["beta"]), float(row.get("beta_se") or 0.0))
            for row in supported if row.get("beta") not in (None, "")
        })
        beta_mean = float(np.mean([item[1] for item in beta_values])) if beta_values else None
        beta_fold_rms = float(np.std([item[1] for item in beta_values])) if beta_values else None
        summaries.append({
            "model_id": model_id,
            "model_family": str(values[0].get("model_family", "")),
            "validation_regime": regime,
            "inference_role": (
                "nested_selection_then_untouched_outer_validation"
                if model_id in {"NESTED_SELECTED", "NESTED_SESSION_SELECTED"}
                else "descriptive_preregistered_fixed_candidate"
            ),
            "applicable_prediction_count": len(applicable),
            "supported_prediction_count": len(supported),
            "unsupported_prediction_count": len(applicable) - len(supported),
            "heldout_group_count": len(groups),
            "complete_support": complete_support,
            "timing_rmse_sec": float(math.sqrt(np.mean(residual**2))) if residual.size else None,
            "timing_median_abs_error_sec": float(np.median(np.abs(residual))) if residual.size else None,
            "predictive_chi2": chi2,
            "predictive_dof": dof,
            "predictive_pvalue": pvalue,
            "predictive_pvalue_method": (
                "Bonferroni-adjusted minimum joint predictive-block chi-square p-value; "
                "valid without assuming cross-fold independence"
            ),
            "minimum_predictive_block_pvalue": minimum_block_pvalue,
            "predictive_block_count": len(blocks),
            "group_aware_mean_nlpd_per_observation": group_aware_score,
            "selection_score_definition": (
                "equal-frozen-group mean joint Gaussian negative log predictive density "
                "per observation, including log determinant"
            ),
            "fitted_parameter_count_min": min(parameter_counts) if parameter_counts else None,
            "fitted_parameter_count_mean": (
                float(np.mean(parameter_counts)) if parameter_counts else None
            ),
            "fitted_parameter_count_max": max(parameter_counts) if parameter_counts else None,
            "beta_fold_mean": beta_mean,
            "beta_fold_rms": beta_fold_rms,
            "passes_predictive_gate": passes,
            "gate_definition": (
                "complete applicable support, >=3 frozen independent heldout groups, "
                "and no full-covariance predictive block rejected under Bonferroni "
                "familywise alpha"
            ),
        })
    return summaries


def _in_sample_residuals(
    model_id: str,
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> tuple[FittedCandidate, dict[tuple[str, int], float]]:
    fitted = fit_candidate(model_id, rows, bundles)
    if fitted.fit_status != "fit":
        raise AggregateError(f"cannot fit corpus diagnostic model {model_id}: {fitted.fit_status}")
    residuals: dict[tuple[str, int], float] = {}
    for row in rows:
        result = predict_candidate(fitted, row, regime="descriptive_full_corpus", fold_id="all")
        if result.get("supported"):
            residuals[(row.map_id, row.network_id)] = float(result["timing_residual_after_prediction_sec"])
    return fitted, residuals


def variance_components(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int], float], dict[str, float]]:
    fitted, residuals = _in_sample_residuals("M1_NETWORK", rows, bundles)
    map_effects: dict[str, float] = {}
    map_variances: dict[str, float] = {}
    between_q = 0.0
    for map_id in sorted({row.map_id for row in rows}):
        local_rows = [row for row in rows if row.map_id == map_id]
        bundle = bundles[map_id]
        local_index = {item.network_id: index for index, item in enumerate(bundle.rows)}
        indices = [local_index[row.network_id] for row in local_rows]
        covariance = bundle.covariance[np.ix_(indices, indices)]
        one = np.ones(len(local_rows))
        inverse_one = np.linalg.solve(covariance, one)
        denominator = float(one @ inverse_one)
        weights = inverse_one / denominator
        vector = np.asarray([residuals[(map_id, row.network_id)] for row in local_rows])
        effect = float(weights @ vector)
        variance = 1.0 / denominator
        map_effects[map_id] = effect
        map_variances[map_id] = variance
        between_q += effect**2 / variance
    map_count = len(map_effects)
    between_df = max(1, map_count - 1)
    between_p = float(stats.chi2.sf(between_q, between_df)) if map_count > 1 else 1.0
    effects = np.asarray(list(map_effects.values()))
    tau_b2 = max(0.0, float(np.var(effects, ddof=1)) - float(np.mean(list(map_variances.values())))) if map_count > 1 else 0.0

    interaction_values = []
    interaction_variances = []
    interaction_q = 0.0
    for row in rows:
        value = residuals[(row.map_id, row.network_id)] - map_effects[row.map_id]
        interaction_values.append(value)
        interaction_variances.append(row.timing_se_sec**2)
        interaction_q += value**2 / row.timing_se_sec**2
    interaction_df = max(1, len(rows) - map_count - max(1, len(fitted.columns)))
    interaction_p = float(stats.chi2.sf(interaction_q, interaction_df))
    tau_i2 = max(
        0.0,
        float(np.mean(np.square(interaction_values)))
        - float(np.mean(interaction_variances)),
    )
    records = [
        {
            "component": "between_beammap_common",
            "method": "explicit measurement-error-subtracted method of moments after full-corpus fixed-network GLS",
            "intrinsic_sd_sec": math.sqrt(tau_b2),
            "q_statistic": between_q,
            "q_dof": between_df,
            "pvalue": between_p,
            "resolved_beyond_measurement_error": between_p < alpha,
            "profile_interval_available": False,
        },
        {
            "component": "network_by_beammap_interaction",
            "method": "map-centered measurement-error-subtracted method of moments after full-corpus fixed-network GLS",
            "intrinsic_sd_sec": math.sqrt(tau_i2),
            "q_statistic": interaction_q,
            "q_dof": interaction_df,
            "pvalue": interaction_p,
            "resolved_beyond_measurement_error": interaction_p < alpha,
            "profile_interval_available": False,
        },
    ]
    summary = {
        "method": "deterministic GLS plus explicit method-of-moments approximation; not monolithic mixed-model REML",
        "between_beammap_intrinsic_sd_sec": math.sqrt(tau_b2),
        "between_beammap_pvalue": between_p,
        "between_beammap_resolved": between_p < alpha,
        "network_by_beammap_interaction_sd_sec": math.sqrt(tau_i2),
        "network_by_beammap_pvalue": interaction_p,
        "network_by_beammap_resolved": interaction_p < alpha,
        "overclaim_warning": "variance-component estimates are method-of-moments diagnostics; boundary profile intervals are unavailable",
    }
    return records, summary, residuals, map_effects


def network_repeatability(
    rows: Sequence[NetworkDatum],
    residuals: Mapping[tuple[str, int], float],
    map_effects: Mapping[str, float],
) -> list[dict[str, Any]]:
    output = []
    for network in sorted({row.network_id for row in rows}):
        selected = [row for row in rows if row.network_id == network]
        values = np.asarray([
            residuals[(row.map_id, row.network_id)] - map_effects[row.map_id]
            for row in selected
        ])
        variances = np.asarray([row.timing_se_sec**2 for row in selected])
        weights = 1.0 / variances
        mean = float(np.sum(weights * values) / np.sum(weights))
        observed = float(math.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))
        intrinsic = math.sqrt(max(0.0, observed**2 - float(np.sum(weights * variances) / np.sum(weights))))
        output.append({
            "network_id": network,
            "map_count": len(selected),
            "independent_group_count": len({row.group_id for row in selected}),
            "map_centered_mean_sec": mean,
            "observed_repeatability_sd_sec": observed,
            "measurement_error_subtracted_repeatability_sd_sec": intrinsic,
            "missing_map_count": len({row.map_id for row in rows}) - len(selected),
        })
    return output


def duplicate_sensitivity_comparisons(
    bundles: Sequence[MapBundle],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_by_observation = {
        bundle.obsnum: bundle for bundle in bundles if bundle.role == "primary"
    }
    sensitivity = sorted(
        (bundle for bundle in bundles if bundle.role == "duplicate_sensitivity"),
        key=lambda bundle: (bundle.obsnum, bundle.map_id),
    )
    output: list[dict[str, Any]] = []
    for alternate in sensitivity:
        primary = primary_by_observation.get(alternate.obsnum)
        if primary is None:
            raise AggregateError(
                f"sensitivity reduction lacks primary observation authority: {alternate.map_id}"
            )
        primary_rows = {row.network_id: row for row in primary.rows}
        alternate_rows = {row.network_id: row for row in alternate.rows}
        for network in sorted(set(primary_rows) | set(alternate_rows)):
            left = primary_rows.get(network)
            right = alternate_rows.get(network)
            paired = left is not None and right is not None
            output.append({
                "observation_number": alternate.obsnum,
                "primary_map_id": primary.map_id,
                "sensitivity_map_id": alternate.map_id,
                "network_id": network,
                "comparison_status": (
                    "paired" if paired
                    else "missing_sensitivity_network" if left is not None
                    else "missing_primary_network"
                ),
                "primary_timing_sec": left.timing_sec if left else None,
                "sensitivity_timing_sec": right.timing_sec if right else None,
                "sensitivity_minus_primary_timing_sec": (
                    right.timing_sec - left.timing_sec if paired else None
                ),
                "absolute_timing_difference_sec": (
                    abs(right.timing_sec - left.timing_sec) if paired else None
                ),
                "primary_slot_sec": left.slot_sec if left else None,
                "sensitivity_slot_sec": right.slot_sec if right else None,
                "sensitivity_minus_primary_slot_sec": (
                    right.slot_sec - left.slot_sec
                    if paired and left.slot_sec is not None and right.slot_sec is not None
                    else None
                ),
                "primary_phase_sec": left.phase_sec if left else None,
                "sensitivity_phase_sec": right.phase_sec if right else None,
                "sensitivity_minus_primary_phase_sec": (
                    right.phase_sec - left.phase_sec
                    if paired and left.phase_sec is not None and right.phase_sec is not None
                    else None
                ),
                "cross_reduction_covariance_available": False,
                "inferential_test_performed": False,
                "used_for_model_fitting_or_classification": False,
            })
    differences = np.asarray([
        float(row["sensitivity_minus_primary_timing_sec"])
        for row in output
        if row["sensitivity_minus_primary_timing_sec"] is not None
    ])
    summary = {
        "available": bool(sensitivity),
        "sensitivity_reduction_count": len(sensitivity),
        "observation_count": len({bundle.obsnum for bundle in sensitivity}),
        "network_comparison_count": len(output),
        "paired_network_comparison_count": int(differences.size),
        "median_absolute_timing_difference_sec": (
            float(np.median(np.abs(differences))) if differences.size else None
        ),
        "maximum_absolute_timing_difference_sec": (
            float(np.max(np.abs(differences))) if differences.size else None
        ),
        "cross_reduction_covariance_available": False,
        "inferential_test_performed": False,
        "used_for_model_fitting_or_classification": False,
        "scope": (
            "deterministic primary-vs-sensitivity reduction comparison only; "
            "sensitivity reductions never add independent observations"
        ),
    }
    return output, summary


def _linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    if x.size < 3 or float(np.ptp(x)) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(x)))):
        return {"available": False, "reason": "insufficient_predictor_leverage", "row_count": int(x.size)}
    design = np.column_stack((np.ones(x.size), x))
    try:
        coefficient, coefficient_covariance, chi2, dof = _gls(design, y, covariance)
    except AggregateError as error:
        return {"available": False, "reason": str(error), "row_count": int(x.size)}
    beta = float(coefficient[1])
    beta_se = math.sqrt(max(0.0, float(coefficient_covariance[1, 1])))
    z_minus_one = (beta + 1.0) / beta_se if beta_se > 0 else math.inf
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0
        else math.nan
    )
    return {
        "available": True,
        "row_count": int(x.size),
        "intercept_sec": float(coefficient[0]),
        "beta": beta,
        "beta_se": beta_se,
        "beta_95_low": beta - 1.96 * beta_se,
        "beta_95_high": beta + 1.96 * beta_se,
        "beta_consistent_with_minus_one_95": beta_se > 0 and abs(z_minus_one) <= 1.96,
        "beta_minus_one_comparison_z": z_minus_one,
        "beta_minus_one_comparison_pvalue": float(2.0 * stats.norm.sf(abs(z_minus_one))) if beta_se > 0 else 0.0,
        "fit_chi2": chi2,
        "fit_dof": dof,
        "pearson": pearson if math.isfinite(pearson) else None,
    }


def slot_regressions(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for predictor, label in (("slot", "native_to_assigned_slot_residual"), ("phase", "native_frame_phase")):
        usable = [row for row in rows if _predictor_value(row, predictor) is not None]
        if usable:
            x = np.asarray([float(_predictor_value(row, predictor)) for row in usable])
            y = np.asarray([row.timing_sec for row in usable])
            covariance = _measurement_matrix(usable, bundles, predictor, -1.0)
            corpus = _linear_fit(x, y, covariance)
        else:
            corpus = {"available": False, "reason": "predictor_unavailable", "row_count": 0}
        corpus_row = {
            "predictor": label,
            "scope": "corpus_unadjusted",
            "map_id": None,
            "session_id": None,
            **corpus,
        }
        output.append(corpus_row)
        summary[label] = corpus
        for map_id in sorted({row.map_id for row in usable}):
            local = [row for row in usable if row.map_id == map_id]
            if not local:
                continue
            x = np.asarray([float(_predictor_value(row, predictor)) for row in local])
            y = np.asarray([row.timing_sec for row in local])
            bundle = bundles[map_id]
            indices = {row.network_id: i for i, row in enumerate(bundle.rows)}
            positions = [indices[row.network_id] for row in local]
            covariance = bundle.covariance[np.ix_(positions, positions)].copy()
            for index, row in enumerate(local):
                covariance[index, index] += _predictor_se(row, predictor) ** 2
            output.append({
                "predictor": label, "scope": "within_map",
                "map_id": map_id, "session_id": local[0].session_id,
                **_linear_fit(x, y, covariance),
            })
        sessions = sorted({row.session_id for row in usable if row.session_id is not None})
        for session in sessions:
            for network in sorted({row.network_id for row in usable if row.session_id == session}):
                local = [
                    row for row in usable
                    if row.session_id == session and row.network_id == network
                ]
                predictor_values = np.asarray([
                    float(_predictor_value(row, predictor)) for row in local
                ])
                output.append({
                    "predictor": label,
                    "scope": "within_exact_session_network_stability",
                    "map_id": None,
                    "session_id": session,
                    "network_id": network,
                    "available": True,
                    "row_count": len(local),
                    "predictor_mean_sec": float(np.mean(predictor_values)),
                    "predictor_sd_sec": float(np.std(predictor_values)),
                    "map_count": len({row.map_id for row in local}),
                })
        for network in sorted({row.network_id for row in usable}):
            session_means = []
            for session in sessions:
                local = [
                    row for row in usable
                    if row.session_id == session and row.network_id == network
                ]
                if local:
                    session_means.append(float(np.mean([
                        float(_predictor_value(row, predictor)) for row in local
                    ])))
            if session_means:
                output.append({
                    "predictor": label,
                    "scope": "across_exact_sessions_per_network",
                    "map_id": None,
                    "session_id": None,
                    "network_id": network,
                    "available": len(session_means) >= 2,
                    "session_count": len(session_means),
                    "session_mean_predictor_mean_sec": float(np.mean(session_means)),
                    "session_mean_predictor_sd_sec": float(np.std(session_means)),
                    "session_mean_predictor_range_sec": float(np.ptp(session_means)),
                })
    return output, summary


def _conservative_group_scalar(
    values: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Collapse correlated repeats without gaining within-group precision."""
    estimates = np.asarray([item[0] for item in values], dtype=float)
    standard_errors = np.asarray([item[1] for item in values], dtype=float)
    mean = float(np.mean(estimates))
    repeatability = float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0
    standard_error = max(float(np.max(standard_errors)), repeatability)
    if standard_error <= 0 or not math.isfinite(standard_error):
        raise AggregateError("independent-group scalar summary has invalid uncertainty")
    return mean, standard_error


def drift_statistics(
    bundles: Sequence[MapBundle],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    quantified_by_group: dict[str, list[tuple[float, float]]] = {}
    counter_anomaly_maps = 0
    counter_anomaly_groups: set[str] = set()
    for bundle in sorted(bundles, key=lambda item: (item.obsnum, item.map_id)):
        difference = _float(_first(bundle.summary, (
            "first_second_half_difference_sec", "first_half_minus_second_half_sec",
        )), field_name="first_second_half_difference_sec", optional=True)
        difference_se = _float(_first(bundle.summary, (
            "first_second_half_difference_se_sec", "half_difference_se_sec",
        )), field_name="first_second_half_difference_se_sec", optional=True)
        anomalies = sum(row.counter_anomaly for row in bundle.rows)
        counter_anomaly_maps += int(anomalies > 0)
        if anomalies > 0:
            counter_anomaly_groups.add(bundle.group_id)
        pvalue = None
        resolved = False
        if difference is not None and difference_se is not None and difference_se > 0:
            z = difference / difference_se
            pvalue = float(2.0 * stats.norm.sf(abs(z)))
            resolved = pvalue < alpha
            quantified_by_group.setdefault(bundle.group_id, []).append(
                (difference, difference_se)
            )
        output.append({
            "record_type": "map_descriptive",
            "map_id": bundle.map_id,
            "obsnum": bundle.obsnum,
            "validation_group_id": bundle.group_id,
            "session_id": bundle.session_id,
            "metric": "first_half_minus_second_half",
            "value_sec": difference,
            "se_sec": difference_se,
            "pvalue": pvalue,
            "resolved_within_observation_timing_variation": resolved,
            "terminology": "within-observation timing variation; not clock drift absent contradictory raw-counter evidence",
            "counter_anomaly_network_count": anomalies,
            "clock_drift_claimed": False,
        })
    group_summaries = []
    for group_id, group_values in sorted(quantified_by_group.items()):
        value, standard_error = _conservative_group_scalar(group_values)
        group_pvalue = float(2.0 * stats.norm.sf(abs(value / standard_error)))
        group_summaries.append((group_id, value, standard_error))
        output.append({
            "record_type": "frozen_independent_group_summary",
            "map_id": None,
            "obsnum": None,
            "validation_group_id": group_id,
            "session_id": None,
            "metric": "first_half_minus_second_half",
            "value_sec": value,
            "se_sec": standard_error,
            "pvalue": group_pvalue,
            "resolved_within_observation_timing_variation": group_pvalue < alpha,
            "map_count": len(group_values),
            "aggregation_rule": (
                "equal-map mean with no within-group precision gain; SE is the "
                "maximum map SE or between-map repeatability SD"
            ),
            "clock_drift_claimed": False,
        })
    if group_summaries:
        values = np.asarray([item[1] for item in group_summaries])
        variance = np.square([item[2] for item in group_summaries])
        weights = 1.0 / variance
        mean = float(np.sum(weights * values) / np.sum(weights))
        se = math.sqrt(1.0 / float(np.sum(weights)))
        pvalue = float(2.0 * stats.norm.sf(abs(mean / se)))
        q = float(np.sum(weights * (values - mean) ** 2))
        q_p = float(stats.chi2.sf(q, max(1, len(values) - 1)))
        detected = len(group_summaries) >= 3 and (pvalue < alpha or q_p < alpha)
    else:
        mean = se = pvalue = q = q_p = None
        detected = False
    summary = {
        "quantified_map_count": sum(len(values) for values in quantified_by_group.values()),
        "quantified_independent_group_count": len(group_summaries),
        "meta_mean_half_difference_sec": mean,
        "meta_mean_se_sec": se,
        "meta_mean_pvalue": pvalue,
        "heterogeneity_q": q,
        "heterogeneity_pvalue": q_p,
        "within_observation_timing_variation_resolved": detected,
        "counter_anomaly_map_count": counter_anomaly_maps,
        "counter_anomaly_independent_group_count": len(counter_anomaly_groups),
        "clock_drift_claimed": False,
        "gate_unit": "frozen independent validation group",
        "limitation": (
            "unquantified half changes are descriptive and cannot trigger TIME-VARIABLE; "
            "correlated maps inside one frozen group do not add independent evidence"
        ),
    }
    return output, summary


def _bundle_gls_mean(
    bundle: MapBundle, network_ids: set[int]
) -> tuple[float, float, np.ndarray] | None:
    indices = [index for index, row in enumerate(bundle.rows) if row.network_id in network_ids]
    if not indices:
        return None
    covariance = bundle.covariance[np.ix_(indices, indices)]
    inverse = np.linalg.pinv(covariance, rcond=1.0e-12)
    ones = np.ones(len(indices), dtype=float)
    denominator = float(ones @ inverse @ ones)
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    local_weights = (inverse @ ones) / denominator
    weights = np.zeros(len(bundle.rows), dtype=float)
    weights[indices] = local_weights
    timing = float(weights @ np.asarray([row.timing_sec for row in bundle.rows]))
    variance = float(weights @ bundle.covariance @ weights)
    return timing, math.sqrt(max(0.0, variance)), weights


def nw9_anomaly_diagnostics(
    bundles: Sequence[MapBundle],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Summarize the preregistered PpsTime increment anomaly without causal claims."""

    occurrence: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    for bundle in sorted(bundles, key=lambda item: (item.obsnum, item.map_id)):
        raw_by_network = {
            _int(row.get("network_id"), field_name="raw phase network_id"): row
            for row in bundle.phase_rows
            if isinstance(row, Mapping) and row.get("network_id") not in (None, "")
        }
        for network, raw in sorted(raw_by_network.items()):
            denominator = _float(raw.get("pps_time_increment_eligible_count"), field_name="pps_time_increment_eligible_count", optional=True)
            count = _float(raw.get("pps_time_increment_mismatch_count"), field_name="pps_time_increment_mismatch_count", optional=True)
            rate = _float(raw.get("pps_time_increment_mismatch_rate"), field_name="pps_time_increment_mismatch_rate", optional=True)
            occurrence.append({
                "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
                "analysis_role": bundle.role,
                "t0_session_id": bundle.session_id,
                "network_id": network,
                "metadata_status": "available",
                "eligible_increment_count": int(denominator) if denominator is not None else None,
                "mismatch_count": int(count) if count is not None else None,
                "mismatch_rate": rate,
                "first_transition_row_zero_based": raw.get("pps_time_increment_anomaly_first_transition_row_zero_based"),
                "last_transition_row_zero_based": raw.get("pps_time_increment_anomaly_last_transition_row_zero_based"),
                "isolated_count": raw.get("pps_time_increment_anomaly_isolated_count"),
                "consecutive_count": raw.get("pps_time_increment_anomaly_consecutive_count"),
                "tick_definition": "PpsTime transition increment minus authenticated Header.Toltec.FpgaFreq",
                "timing_result_used_as_cut": False,
            })
        for network in sorted({row.network_id for row in bundle.rows} - set(raw_by_network)):
            occurrence.append({
                "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
                "analysis_role": bundle.role,
                "t0_session_id": bundle.session_id,
                "network_id": network,
                "metadata_status": "unavailable_or_unreadable_not_zero_anomalies",
                "eligible_increment_count": None,
                "mismatch_count": None,
                "mismatch_rate": None,
                "first_transition_row_zero_based": None,
                "last_transition_row_zero_based": None,
                "isolated_count": None,
                "consecutive_count": None,
                "tick_definition": None,
                "timing_result_used_as_cut": False,
            })
        if 9 not in raw_by_network and not any(row.network_id == 9 for row in bundle.rows):
            occurrence.append({
                "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
                "analysis_role": bundle.role,
                "t0_session_id": bundle.session_id,
                "network_id": 9,
                "metadata_status": "nw9_not_present_or_metadata_unavailable_not_zero_anomalies",
                "eligible_increment_count": None,
                "mismatch_count": None,
                "mismatch_rate": None,
                "first_transition_row_zero_based": None,
                "last_transition_row_zero_based": None,
                "isolated_count": None,
                "consecutive_count": None,
                "tick_definition": None,
                "timing_result_used_as_cut": False,
            })
        for raw in _read_optional_csv(bundle.directory, "raw_pps_time_increment_anomalies.csv"):
            item = dict(raw)
            item.update({
                "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
                "analysis_role": bundle.role,
                "t0_session_id": bundle.session_id,
            })
            details.append(item)
        nw9 = next((row for row in bundle.rows if row.network_id == 9), None)
        if nw9 is None:
            continue
        all_mean = _bundle_gls_mean(bundle, {row.network_id for row in bundle.rows})
        other_mean = _bundle_gls_mean(bundle, {row.network_id for row in bundle.rows if row.network_id != 9})
        leave_mean = other_mean
        nw9_index = next(index for index, row in enumerate(bundle.rows) if row.network_id == 9)
        nw9_weights = np.zeros(len(bundle.rows), dtype=float)
        nw9_weights[nw9_index] = 1.0
        if other_mean is not None:
            contrast = nw9_weights - other_mean[2]
            relative = float(contrast @ np.asarray([row.timing_sec for row in bundle.rows]))
            relative_se = math.sqrt(max(0.0, float(contrast @ bundle.covariance @ contrast)))
        else:
            relative, relative_se = None, None
        if all_mean is not None and leave_mean is not None:
            difference_weights = all_mean[2] - leave_mean[2]
            leave_difference = float(difference_weights @ np.asarray([row.timing_sec for row in bundle.rows]))
            leave_difference_se = math.sqrt(max(0.0, float(difference_weights @ bundle.covariance @ difference_weights)))
        else:
            leave_difference, leave_difference_se = None, None
        anomaly = raw_by_network.get(9, {})
        timing.append({
            "map_id": bundle.map_id,
            "observation_number": bundle.obsnum,
            "t0_session_id": bundle.session_id,
            "nw9_timing_sec": nw9.timing_sec,
            "nw9_timing_se_sec": nw9.timing_se_sec,
            "nw9_relative_to_other_networks_sec": relative,
            "nw9_relative_to_other_networks_se_sec": relative_se,
            "nw9_anomaly_count": anomaly.get("pps_time_increment_mismatch_count"),
            "nw9_eligible_increment_count": anomaly.get("pps_time_increment_eligible_count"),
            "nw9_anomaly_rate": anomaly.get("pps_time_increment_mismatch_rate"),
            "all_network_pooled_timing_sec": all_mean[0] if all_mean else None,
            "all_network_pooled_timing_se_sec": all_mean[1] if all_mean else None,
            "leave_nw9_out_pooled_timing_sec": leave_mean[0] if leave_mean else None,
            "leave_nw9_out_pooled_timing_se_sec": leave_mean[1] if leave_mean else None,
            "all_minus_leave_nw9_out_sec": leave_difference,
            "all_minus_leave_nw9_out_se_sec": leave_difference_se,
            "affected_row_mask_or_repair": "not_authorized_metadata_semantics_ambiguous",
            "association_is_causal_claim": False,
        })
    association_rows = [
        row for row in timing
        if row.get("nw9_anomaly_rate") not in (None, "")
        and row.get("nw9_relative_to_other_networks_sec") not in (None, "")
        and row.get("nw9_relative_to_other_networks_se_sec") not in (None, "")
    ]
    if association_rows:
        x = np.asarray([float(row["nw9_anomaly_rate"]) for row in association_rows])
        y = np.asarray([float(row["nw9_relative_to_other_networks_sec"]) for row in association_rows])
        covariance = np.diag([
            max(float(row["nw9_relative_to_other_networks_se_sec"]) ** 2, 1.0e-24)
            for row in association_rows
        ])
        association = _linear_fit(x, y, covariance)
    else:
        association = {"available": False, "reason": "no_nw9_anomaly_rate_and_timing_pairs", "row_count": 0}
    nw9_occurrence = [row for row in occurrence if row["network_id"] == 9 and row["analysis_role"] == "primary"]
    total_denominator = sum(int(row["eligible_increment_count"] or 0) for row in nw9_occurrence)
    total_count = sum(int(row["mismatch_count"] or 0) for row in nw9_occurrence)
    summary = {
        "nw9_observation_count_with_raw_metadata": len(nw9_occurrence),
        "nw9_observation_count_with_at_least_one_anomaly": sum(int(row["mismatch_count"] or 0) > 0 for row in nw9_occurrence),
        "nw9_t0_session_count_with_at_least_one_anomaly": len({
            row["t0_session_id"] for row in nw9_occurrence
            if int(row["mismatch_count"] or 0) > 0 and row.get("t0_session_id") not in (None, "")
        }),
        "nw9_total_mismatch_count": total_count,
        "nw9_total_eligible_increment_count": total_denominator,
        "nw9_total_mismatch_rate": total_count / total_denominator if total_denominator else None,
        "all_network_control_occurrence_row_count": len(occurrence),
        "nw9_timing_association": association,
        "affected_row_mask_or_repair_authorized": False,
        "leave_nw9_out_sensitivity": "reported per observation; no timing-based exclusion",
        "interpretation": "metadata occurrence and association measurement only; not clock-drift or causal evidence",
    }
    return occurrence, details, timing, summary


def session_statistics(
    rows: Sequence[NetworkDatum],
    bundles: Mapping[str, MapBundle],
    *,
    alpha: float,
) -> dict[str, Any]:
    fitted = fit_candidate("M1_NETWORK", rows, bundles)
    if fitted.fit_status != "fit":
        return {
            "available": False,
            "reason": "persistent_network_adjustment_unavailable",
            "network_adjustment_fit_status": fitted.fit_status,
            "session_count": 0,
        }

    map_summaries: dict[str, tuple[float, float]] = {}
    for map_id in sorted({row.map_id for row in rows}):
        local = sorted(
            (row for row in rows if row.map_id == map_id),
            key=lambda row: row.network_id,
        )
        design_rows = []
        for row in local:
            vector, status = _design_row(
                row, fitted.specification, fitted.columns,
                fitted.network_levels, fitted.session_levels,
            )
            if vector is None or status != "supported":
                raise AggregateError("network-adjusted session row is unsupported")
            design_rows.append(vector)
        design = np.vstack(design_rows)
        observed = np.asarray([row.timing_sec for row in local])
        residual = observed - design @ fitted.coefficients
        bundle = bundles[map_id]
        bundle_index = {row.network_id: index for index, row in enumerate(bundle.rows)}
        positions = [bundle_index[row.network_id] for row in local]
        covariance = bundle.covariance[np.ix_(positions, positions)].copy()
        covariance += design @ fitted.coefficient_covariance @ design.T
        one = np.ones(len(local))
        inverse_one = np.linalg.solve(covariance, one)
        denominator = float(one @ inverse_one)
        map_summaries[map_id] = (
            float((inverse_one / denominator) @ residual),
            math.sqrt(1.0 / denominator),
        )

    group_rows: list[dict[str, Any]] = []
    for group_id in sorted({row.group_id for row in rows}):
        selected = [row for row in rows if row.group_id == group_id]
        sessions = sorted({str(row.session_id) for row in selected if row.session_id is not None})
        if len(sessions) != 1:
            continue
        map_ids = sorted({row.map_id for row in selected})
        mean, standard_error = _conservative_group_scalar([
            map_summaries[map_id] for map_id in map_ids
        ])
        group_rows.append({
            "validation_group_id": group_id,
            "session_id": sessions[0],
            "map_count": len(map_ids),
            "network_map_count": len(selected),
            "network_adjusted_mean_timing_sec": mean,
            "network_adjusted_mean_se_sec": standard_error,
            "aggregation_rule": (
                "full within-map covariance after persistent-network GLS; "
                "correlated maps within one frozen group receive no precision gain"
            ),
        })
    session_ids = sorted({str(row["session_id"]) for row in group_rows})
    if len(session_ids) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_unambiguous_sessions_after_group_collapse",
            "session_count": len(session_ids),
            "independent_group_count": len(group_rows),
            "independent_group_records": group_rows,
        }

    session_records = []
    means = []
    variances = []
    for session in session_ids:
        selected_groups = [row for row in group_rows if row["session_id"] == session]
        values = np.asarray([
            float(row["network_adjusted_mean_timing_sec"])
            for row in selected_groups
        ])
        variance = np.square([
            float(row["network_adjusted_mean_se_sec"])
            for row in selected_groups
        ])
        weights = 1.0 / variance
        mean = float(np.sum(weights * values) / np.sum(weights))
        session_variance = 1.0 / float(np.sum(weights))
        means.append(mean)
        variances.append(session_variance)
        session_records.append({
            "session_id": session,
            "independent_group_count": len(selected_groups),
            "map_count": sum(int(row["map_count"]) for row in selected_groups),
            "network_map_count": sum(
                int(row["network_map_count"]) for row in selected_groups
            ),
            "network_adjusted_weighted_mean_timing_sec": mean,
            "network_adjusted_weighted_mean_se_sec": math.sqrt(session_variance),
        })
    inverse = 1.0 / np.asarray(variances)
    grand = float(np.sum(inverse * means) / np.sum(inverse))
    q = float(np.sum(inverse * np.square(np.asarray(means) - grand)))
    pvalue = float(stats.chi2.sf(q, len(session_ids) - 1))
    return {
        "available": True,
        "method": (
            "persistent-network fixed effects removed by full-covariance GLS; "
            "one conservative summary per frozen independent group; session "
            "heterogeneity tested only across independent session summaries"
        ),
        "session_count": len(session_ids),
        "independent_group_count": len(group_rows),
        "independent_group_records": group_rows,
        "session_records": session_records,
        "session_network_adjusted_mean_range_sec": float(np.ptp(means)),
        "session_heterogeneity_q": q,
        "session_heterogeneity_pvalue": pvalue,
        "session_effect_resolved": pvalue < alpha,
        "session_identity_rule": "complete exact T0 vector preferred; otherwise unambiguous provenance session only",
    }


def _candidate_lookup(
    summaries: Sequence[Mapping[str, Any]],
    model_id: str,
    regime: str = "outer_logo",
) -> Mapping[str, Any] | None:
    return next((
        row for row in summaries
        if row.get("model_id") == model_id and row.get("validation_regime") == regime
    ), None)


def _passes_any(
    summaries: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    regime: str,
) -> list[str]:
    return [
        model_id for model_id in model_ids
        if (row := _candidate_lookup(summaries, model_id, regime)) is not None
        and _bool(row.get("passes_predictive_gate"), False)
    ]


def classify_result(
    *,
    group_count: int,
    candidate_summaries: Sequence[Mapping[str, Any]],
    variance_summary: Mapping[str, Any],
    drift_summary: Mapping[str, Any],
    session_summary: Mapping[str, Any],
    slot_summary: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    residual_structure = bool(
        variance_summary.get("between_beammap_resolved")
        or variance_summary.get("network_by_beammap_resolved")
    )
    timing_variation = bool(drift_summary.get("within_observation_timing_variation_resolved"))
    nested_outer = _candidate_lookup(candidate_summaries, "NESTED_SELECTED", "nested_outer_logo")
    nested_outer_pass = bool(nested_outer and nested_outer.get("passes_predictive_gate"))
    nested_session = _candidate_lookup(
        candidate_summaries, "NESTED_SESSION_SELECTED", "nested_rolling_session_anchor"
    )
    nested_session_pass = bool(nested_session and nested_session.get("passes_predictive_gate"))
    selected_outer = sorted({
        str(row["selected_model_id"])
        for row in predictions
        if row.get("validation_regime") == "nested_outer_logo"
        and row.get("supported") and row.get("selected_model_id")
    })
    selected_session = sorted({
        str(row["selected_model_id"])
        for row in predictions
        if row.get("validation_regime") == "nested_rolling_session_anchor"
        and row.get("supported") and row.get("selected_model_id")
    })
    selected_outer_specs = [_specification(model_id) for model_id in selected_outer]
    selected_session_specs = [_specification(model_id) for model_id in selected_session]
    m0 = nested_outer_pass and selected_outer == ["M0_GLOBAL"]
    m1 = (
        nested_outer_pass and bool(selected_outer_specs)
        and all(spec.get("family") == "M1" for spec in selected_outer_specs)
    )
    predictive = (
        selected_outer if nested_outer_pass and bool(selected_outer_specs)
        and all(spec.get("predictor") in {"slot", "phase"} for spec in selected_outer_specs)
        else []
    )
    session = (
        selected_session if nested_session_pass and bool(selected_session_specs)
        and all(spec.get("session") for spec in selected_session_specs)
        else []
    )
    session_predictor_evidence = bool(session) and all(
        spec.get("predictor") in {"slot", "phase"}
        for spec in selected_session_specs
    )

    reasons: list[str] = []
    if group_count < 4:
        code, category = "G", "INSUFFICIENT"
        reasons.append(
            "fewer than four frozen independent groups; exactly three permits fixed-model reporting but not data-driven category selection"
        )
    elif m0 and not residual_structure and not timing_variation:
        code, category = "A", "GLOBAL-STABLE"
        reasons.append("M0 passed held-out prediction and no residual variance/variation component was resolved")
    elif m1 and not residual_structure and not timing_variation:
        code, category = "B", "NETWORK-STABLE"
        reasons.append("M1 passed held-out prediction with no resolved residual map/interaction/variation component")
    elif predictive and not timing_variation:
        code, category = "D", "SLOT-PREDICTABLE"
        reasons.append(f"dynamic native-phase/slot predictor models passed held-out prediction: {predictive}")
    elif (
        session and int(session_summary.get("session_count", 0)) >= 3
        and _bool(session_summary.get("session_effect_resolved"), False)
        and not timing_variation
    ):
        code, category = "C", "SESSION-STABLE"
        reasons.append(f"rolling session-anchor models passed: {session}")
    elif timing_variation:
        code, category = "E", "TIME-VARIABLE"
        reasons.append("quantified within-observation timing variation remains statistically resolved")
    elif residual_structure:
        code, category = "F", "UNPREDICTABLE"
        reasons.append("variability exceeds measurement uncertainty and no preregistered predictive model passed")
    else:
        code, category = "G", "INSUFFICIENT"
        reasons.append("no category met its complete preregistered evidence gate")
    beta_status = {
        name: {
            "available": value.get("available", False),
            "beta": value.get("beta"),
            "beta_se": value.get("beta_se"),
            "beta_consistent_with_minus_one_95": value.get("beta_consistent_with_minus_one_95"),
        }
        for name, value in slot_summary.items()
    }
    structural_followup_supported = bool(
        category == "SLOT-PREDICTABLE"
        or (category == "SESSION-STABLE" and session_predictor_evidence)
    )
    structural_followup = (
        "Stable or T0-session-predictable native phase/slot evidence supports "
        "a later bounded native-time/fractional-slot investigation; do not "
        "hard-code a physical clock correction."
        if structural_followup_supported else None
    )
    if category == "SESSION-STABLE" and not session_predictor_evidence:
        producer_followup_interpretation = (
            "Session timing offsets were predictive, but native phase/slot "
            "evidence was not itself selected as stable or T0-session-predictable; "
            "session timing alone does not support native-time/fractional-slot "
            "follow-up."
        )
    elif structural_followup is not None:
        producer_followup_interpretation = structural_followup
    else:
        producer_followup_interpretation = (
            "No stable or T0-session-predictable native phase/slot evidence was "
            "established for structural follow-up."
        )
    return {
        "code": code,
        "category": category,
        "reasons": reasons,
        "nested_outer_prediction_passed": nested_outer_pass,
        "nested_outer_selected_models": selected_outer,
        "nested_session_prediction_passed": nested_session_pass,
        "nested_session_selected_models": selected_session,
        "passing_global_model": m0,
        "passing_network_model": m1,
        "passing_dynamic_predictor_models": predictive,
        "passing_rolling_session_models": session,
        "session_native_phase_or_slot_evidence": session_predictor_evidence,
        "predictor_beta_status": beta_status,
        "production_correction_authorized": False,
        "science_tolerance_assessed": False,
        "physical_clock_correction_recommended": False,
        "structural_followup_supported_by_native_phase_or_slot": (
            structural_followup_supported
        ),
        "structural_followup": structural_followup,
        "producer_followup_interpretation": producer_followup_interpretation,
    }


def _fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> list[str]:
    result = list(preferred)
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _save_plot(path: Path, draw: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.2, 4.5))
    try:
        draw(figure)
        figure.tight_layout()
        fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
        os.close(fd)
        try:
            figure.savefig(temporary, dpi=150, metadata={"Date": None, "Software": "SCI-ALIGN-001"})
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    finally:
        plt.close(figure)


def write_plots(
    output: Path,
    bundles: Sequence[MapBundle],
    candidate_rows: Sequence[Mapping[str, Any]],
    slot_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output / "plots"

    def timing_plot(figure: Any) -> None:
        axis = figure.subplots()
        ordered = sorted(bundles, key=lambda item: (item.obsnum, item.map_id))
        x = np.arange(len(ordered))
        y = [float(bundle.summary["timing_residual_sec"]) * 1000.0 for bundle in ordered]
        se = [float(_first(bundle.summary, NETWORK_SE_ALIASES)) * 1000.0 for bundle in ordered]
        axis.errorbar(x, y, yerr=se, marker="o", linestyle="none")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, [str(bundle.obsnum) for bundle in ordered], rotation=45, ha="right")
        axis.set_xlabel("observation number")
        axis.set_ylabel("pooled left/right timing residual (ms)")
    _save_plot(plot_dir / "pooled_timing_vs_beammap.png", timing_plot)

    def model_plot(figure: Any) -> None:
        axis = figure.subplots()
        selected = [row for row in candidate_rows if row.get("validation_regime") == "outer_logo" and row.get("timing_rmse_sec") is not None]
        selected.sort(key=lambda row: str(row["model_id"]))
        axis.bar(np.arange(len(selected)), [float(row["timing_rmse_sec"]) * 1000.0 for row in selected])
        axis.set_xticks(np.arange(len(selected)), [str(row["model_id"]) for row in selected], rotation=70, ha="right")
        axis.set_ylabel("held-out timing RMSE (ms)")
    _save_plot(plot_dir / "heldout_model_rmse.png", model_plot)

    def predictor_plot(figure: Any) -> None:
        axis = figure.subplots()
        styles = {"native_to_assigned_slot_residual": "o", "native_frame_phase": "s"}
        for predictor, marker in styles.items():
            values = [row for row in slot_rows if row.get("scope") == "within_map" and row.get("predictor") == predictor and row.get("available")]
            if values:
                axis.errorbar(
                    np.arange(len(values)), [float(row["beta"]) for row in values],
                    yerr=[float(row["beta_se"]) for row in values], marker=marker,
                    linestyle="none", label=predictor,
                )
        axis.axhline(-1.0, color="black", linestyle="--", linewidth=0.9, label="beta=-1")
        axis.set_ylabel("within-map predictor slope beta")
        axis.set_xlabel("deterministically ordered map fit")
        axis.legend(fontsize="small")
    _save_plot(plot_dir / "predictor_slopes.png", predictor_plot)


def _report_text(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    variance = summary["variance_components"]
    drift = summary["within_observation_timing_variation"]
    sensitivity = summary["duplicate_reduction_sensitivity"]
    nw9 = summary.get("nw9_pps_time_anomaly", {})
    heldout = summary.get("nested_selected_heldout_performance", {})
    timing_rmse = heldout.get("timing_rmse_sec")
    timing_text = f"{float(timing_rmse):.9g} s" if timing_rmse is not None else "unavailable"
    sensitivity_maximum = sensitivity.get("maximum_absolute_timing_difference_sec")
    sensitivity_maximum_text = (
        f"{float(sensitivity_maximum):.9g} s"
        if sensitivity_maximum is not None else "unavailable"
    )
    followup_interpretation = str(decision["producer_followup_interpretation"])
    return f"""# SCI-ALIGN-001 3C273 corpus aggregate

## Result

Classification: **{decision['code']}. {decision['category']}**.

This is held-out predictive diagnostic evidence only. It authorizes neither a
production correction nor a physical clock interpretation. No scientific
acceptability threshold is imposed or assessed in this run.

Reasons: {'; '.join(decision['reasons'])}.

Classification uses deterministic inner-fold model selection followed by
untouched outer-group prediction. Candidate-specific outer-LOGO rows are
retained as descriptive fixed-model comparisons and do not select their own
category.

Nested-selected held-out timing RMSE: **{timing_text}**. Translation into
on-sky scientific impact is deferred to a separate downstream analysis.

## Variability

- Between-beammap intrinsic SD (method-of-moments): {variance['between_beammap_intrinsic_sd_sec']:.9g} s.
- Network-by-beammap interaction SD (method-of-moments): {variance['network_by_beammap_interaction_sd_sec']:.9g} s.
- Quantified within-observation timing variation resolved: {str(drift['within_observation_timing_variation_resolved']).lower()}.
- Raw-counter anomaly maps: {drift['counter_anomaly_map_count']}.

First-half/second-half changes are called *within-observation timing variation*,
not clock drift, unless raw counters contradict the shared-Octo clock account.

## nw9 PpsTime increment anomaly

- nw9 observations with raw metadata: {nw9.get('nw9_observation_count_with_raw_metadata', 0)}.
- nw9 observations with one or more mismatches: {nw9.get('nw9_observation_count_with_at_least_one_anomaly', 0)}.
- Corpus numerator/denominator: {nw9.get('nw9_total_mismatch_count', 0)}/{nw9.get('nw9_total_eligible_increment_count', 0)}.
- Corpus nw9 mismatch rate: {nw9.get('nw9_total_mismatch_rate', None)}.

`pps_time_increment_occurrence.csv` retains every available-network control,
`raw_pps_time_increment_anomalies.csv` preserves each delivered anomaly and
its row/counter geometry, and `nw9_timing_sensitivity.csv` reports nw9 versus
other-network and leave-nw9-out effects with uncertainty. These are association
measurements only; no affected-row mask or repair is authorized because
metadata-to-integration semantics remain unresolved.

## Duplicate-reduction sensitivity

- Sensitivity reductions compared: {sensitivity['sensitivity_reduction_count']}.
- Paired network comparisons: {sensitivity['paired_network_comparison_count']}.
- Maximum absolute timing difference: {sensitivity_maximum_text}.

These comparisons are sensitivity-only. They do not add independent
observations, enter model fitting, or affect classification; no inferential
test is made because cross-reduction covariance is unavailable.

## Producer-account interpretation

T0 is an integer-second ROACH-initialization label. The shared Octo 10 MHz/PPS
does not reset detector sample cadence, so stable per-network integration phase
is possible. Native frame phase and native-to-assigned-slot residual were
analyzed separately. {followup_interpretation}

## Scope

- Production correction authorized: false
- Science acceptability threshold assessed: false
- Citlali reductions launched by aggregation: 0
- Source products modified: false
- Unity contacted by aggregation: false
- Raw-row reassociation claimed: false
- Physical timestamp-event semantics: unresolved
- Arbitrary millisecond NTP error: strongly disfavored
- Differential oscillator drift: strongly disfavored
- Distinct stable per-network integration phase: possible

The verified compact concatenations are `map_summary.csv`,
`network_map_results.csv`, `timing_phase_results.csv`, and
`fit_controls.csv/json`. Canonical/duplicate role and frozen grouping are in
`session_registry.csv`; inventory-owned exclusion and duplicate registries
remain authoritative in the transferred inventory package.
"""


def command_run(args: argparse.Namespace) -> int:
    manifest = args.selected_manifest.resolve()
    protocol_path = args.frozen_protocol.resolve()
    _verify_checksum_file(protocol_path.parent)
    _verify_checksum_file(manifest.parent)
    selected_document = _selected_manifest_document(manifest)
    manifest_sha = sha256_file(manifest)
    source_inventory = _inventory_for_selected_manifest(
        manifest, str(selected_document["source_inventory_sha256"])
    )
    protocol_file_sha = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise AggregateError("frozen protocol must be an object")
    if protocol.get("aggregation_freeze_schema_version") != PROTOCOL_VERSION:
        raise AggregateError("protocol lacks a SCI-ALIGN-001 aggregation freeze")
    recorded_semantic = str(protocol.get("protocol_sha256", ""))
    calculated_semantic = _semantic_digest(protocol, ("protocol_sha256",))
    if recorded_semantic != calculated_semantic:
        raise AggregateError("frozen protocol semantic digest mismatch")
    tool_identity = _validate_frozen_tooling(protocol)
    if protocol.get("selected_manifest_sha256") != manifest_sha:
        raise AggregateError("selected manifest differs from frozen protocol")
    runner_protocol_sha = str(protocol.get("runner_protocol_template_sha256", "")) or None
    registry = protocol.get("partition_rows")
    if not isinstance(registry, list) or not registry:
        raise AggregateError("frozen protocol lacks partition rows")
    registry_by_map = {str(row["map_id"]): dict(row) for row in registry}
    if len(registry_by_map) != len(registry):
        raise AggregateError("frozen partition has duplicate map identities")
    current = normalize_manifest(_read_manifest(manifest))
    if {row.map_id for row in current} != set(registry_by_map):
        raise AggregateError("selected manifest identities differ from frozen partition")

    directories = _discover_map_directories(args.map_output_root)
    directory_identity: dict[str, Path] = {}
    for directory in directories:
        result = _result_document(directory)
        summary = _load_summary(directory, result)
        map_id = str(_first(summary, ("map_id", "candidate_id"), ""))
        if map_id not in registry_by_map:
            raise AggregateError(f"compact output map is absent from frozen partition: {map_id}")
        if map_id in directory_identity:
            raise AggregateError(f"multiple compact output directories for map {map_id}")
        directory_identity[map_id] = directory
    required = sorted(
        map_id for map_id, row in registry_by_map.items()
        if row.get("analysis_role") in {"primary", "duplicate_sensitivity"}
    )
    missing = [map_id for map_id in required if map_id not in directory_identity]
    if missing:
        raise AggregateError(f"missing compact primary map outputs: {missing}")
    bundles = [
        load_map_bundle(
            directory_identity[map_id], registry_by_map[map_id],
            manifest_sha256=manifest_sha,
            protocol_file_sha256=protocol_file_sha,
            protocol_semantic_sha256=recorded_semantic,
            runner_protocol_sha256=runner_protocol_sha,
        )
        for map_id in sorted(directory_identity)
    ]
    primary = [bundle for bundle in bundles if bundle.role == "primary"]
    rows = [row for bundle in primary for row in bundle.rows]
    bundle_lookup = {bundle.map_id: bundle for bundle in primary}
    group_count = len({bundle.group_id for bundle in primary})
    if args.dry_run:
        print(json.dumps({
            "action": "aggregate_run", "writes_performed": False,
            "primary_map_count": len(primary), "network_map_count": len(rows),
            "independent_group_count": group_count,
            "verified_selected_manifest_sha256": manifest_sha,
            "verified_protocol_sha256": recorded_semantic,
            "verified_aggregation_tool_sha256": tool_identity["script_sha256"],
            "verified_candidate_model_registry_sha256": (
                tool_identity["candidate_model_registry_sha256"]
            ),
            "would_write": str(args.output.resolve()),
        }, indent=2, sort_keys=True))
        return 0

    alpha = float(protocol.get("alpha", protocol.get("aggregation", {}).get("alpha", DEFAULT_ALPHA)))
    predictions = heldout_predictions(rows, bundle_lookup)
    candidate_rows = summarize_candidates(predictions, alpha=alpha)
    variance_rows, variance_summary, residuals, map_effects = variance_components(
        rows, bundle_lookup, alpha=alpha,
    )
    repeatability_rows = network_repeatability(rows, residuals, map_effects)
    sensitivity_rows, sensitivity_summary = duplicate_sensitivity_comparisons(bundles)
    regression_rows, regression_summary = slot_regressions(rows, bundle_lookup)
    drift_rows, drift_summary = drift_statistics(primary, alpha=alpha)
    session_summary = session_statistics(rows, bundle_lookup, alpha=alpha)
    nw9_occurrence_rows, nw9_detail_rows, nw9_timing_rows, nw9_summary = (
        nw9_anomaly_diagnostics(primary)
    )
    decision = classify_result(
        group_count=group_count,
        candidate_summaries=candidate_rows,
        variance_summary=variance_summary,
        drift_summary=drift_summary,
        session_summary=session_summary,
        slot_summary=regression_summary,
        predictions=predictions,
    )
    preferred_nested_model = (
        "NESTED_SESSION_SELECTED"
        if decision["category"] == "SESSION-STABLE" else "NESTED_SELECTED"
    )
    nested_performance = next((
        dict(row) for row in candidate_rows
        if row.get("model_id") == preferred_nested_model
    ), {})
    limitations = []
    for bundle in primary:
        if str(bundle.summary.get("status")) == "partial_core_success_enhanced_failed":
            limitations.append({
                "map_id": bundle.map_id,
                "limitation": "enhanced raw-timestamp analysis failed; retained core result included",
            })
        if bundle.covariance_source == "diagonal_from_timing_se":
            limitations.append({
                "map_id": bundle.map_id,
                "limitation": "paired network covariance unavailable; diagonal measurement covariance used",
            })
    corpus_summary = {
        "schema_version": SCHEMA_VERSION,
        "selected_manifest_sha256": manifest_sha,
        "frozen_protocol_file_sha256": protocol_file_sha,
        "frozen_protocol_semantic_sha256": recorded_semantic,
        "runner_protocol_template_sha256": runner_protocol_sha,
        "aggregation_tool": tool_identity,
        "map_count": len(primary),
        "network_map_count": len(rows),
        "independent_group_count": group_count,
        "grouping_kind": protocol.get("grouping_kind"),
        "variance_components": variance_summary,
        "within_observation_timing_variation": drift_summary,
        "session_effects": session_summary,
        "predictor_regressions": regression_summary,
        "duplicate_reduction_sensitivity": sensitivity_summary,
        "nw9_pps_time_anomaly": nw9_summary,
        "decision": decision,
        "nested_selected_heldout_performance": nested_performance,
        "classification_uses": (
            "nested inner selection followed by untouched outer-group predictions; "
            "candidate-specific outer LOGO rows are descriptive"
        ),
        "limitations": limitations,
        "production_correction_authorized": False,
        "science_tolerance_assessed": False,
        "physical_clock_correction_recommended": False,
        "citlali_reductions_launched": 0,
        "unity_contacted": False,
        "raw_row_reassociation_claimed": False,
        "physical_timestamp_event_semantics": "unresolved",
        "upstream_fpga_metadata_to_integration_association": "unresolved",
        "arbitrary_millisecond_ntp_error": "strongly_disfavored",
        "differential_oscillator_drift": "strongly_disfavored",
        "distinct_stable_per_network_integration_phase": "possible",
        "large_products_consumed": False,
        "source_products_modified": False,
    }

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise AggregateError(
            f"aggregate output directory is not empty; use a fresh directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    combined_map_rows = []
    combined_network_rows = []
    combined_timing_rows = []
    combined_fit_controls = []
    for bundle in sorted(bundles, key=lambda item: (item.obsnum, item.map_id)):
        combined_map_rows.append({
            **bundle.summary,
            "map_id": bundle.map_id,
            "observation_number": bundle.obsnum,
            "analysis_role": bundle.role,
            "validation_group_id": bundle.group_id,
            "effective_session_id": bundle.session_id,
            "measurement_covariance_source": bundle.covariance_source,
        })
        for row in bundle.rows:
            combined_network_rows.append({
                **row.source_row,
                "map_id": row.map_id,
                "observation_number": row.obsnum,
                "validation_group_id": row.group_id,
                "effective_session_id": row.session_id,
                "network_id": row.network_id,
                "timing_residual_sec": row.timing_sec,
                "timing_se_sec": row.timing_se_sec,
                "native_to_assigned_slot_residual_sec": row.slot_sec,
                "native_frame_phase_mean_sec": row.phase_sec,
                "counter_anomaly": row.counter_anomaly,
            })
        for row in bundle.timing_rows:
            combined_timing_rows.append({
                **row, "record_type": "timing_model", "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
            })
        for row in bundle.phase_rows:
            combined_timing_rows.append({
                **row, "record_type": "raw_phase_summary", "map_id": bundle.map_id,
                "observation_number": bundle.obsnum,
            })
        controls_path = bundle.directory / "fit_controls.json"
        controls = json.loads(controls_path.read_text()) if controls_path.is_file() else {}
        cohort = controls.get("cohort", {}) if isinstance(controls, Mapping) else {}
        combined_fit_controls.append({
            "map_id": bundle.map_id,
            "observation_number": bundle.obsnum,
            "analysis_role": bundle.role,
            "preselected_detector_count": _first(cohort, ("preselected_detector_count",), bundle.summary.get("preselected_detector_count")),
            "matched_detector_count": _first(cohort, ("matched_detector_count",), bundle.summary.get("matched_detector_count")),
            "selection_depends_on_timing_estimate": _bool(cohort.get("selection_depends_on_timing_estimate"), False),
            "quality": _bool(bundle.summary.get("quality"), True),
            "status": bundle.summary.get("status"),
        })
    write_csv(output / "map_summary.csv", combined_map_rows, _fields(combined_map_rows, (
        "map_id", "observation_number", "analysis_role", "validation_group_id",
        "effective_session_id", "status", "quality", "timing_residual_sec", "timing_se_sec",
    )))
    write_csv(output / "network_map_results.csv", combined_network_rows, _fields(combined_network_rows, (
        "map_id", "observation_number", "validation_group_id", "effective_session_id",
        "network_id", "timing_residual_sec", "timing_se_sec",
        "native_to_assigned_slot_residual_sec", "native_frame_phase_mean_sec",
    )))
    write_csv(output / "timing_phase_results.csv", combined_timing_rows, _fields(combined_timing_rows, (
        "record_type", "map_id", "observation_number", "network_id", "model_id",
        "native_to_assigned_slot_residual_sec", "native_frame_phase_mean_sec",
    )))
    write_csv(output / "fit_controls.csv", combined_fit_controls, _fields(combined_fit_controls))
    write_json(output / "fit_controls.json", {
        "schema_version": "sci-align-001-3c273-aggregate-fit-controls-v1",
        "maps": combined_fit_controls,
        "timing_result_used_as_quality_cut": False,
    })
    write_csv(output / "session_registry.csv", registry, _fields(registry, (
        "map_id", "obsnum", "analysis_role", "session_id", "t0_session_key",
        "validation_group_id", "fold_id",
    )))
    duplicate_rows = [row for row in registry if row.get("analysis_role") == "duplicate_sensitivity"]
    write_csv(
        output / "duplicate_reduction_registry.csv",
        duplicate_rows,
        _fields(duplicate_rows, (
            "map_id", "obsnum", "reduction_id", "duplicate_group_id",
            "analysis_role", "validation_group_id",
        )),
    )
    write_csv(
        output / "duplicate_reduction_sensitivity.csv",
        sensitivity_rows,
        _fields(sensitivity_rows, (
            "observation_number", "primary_map_id", "sensitivity_map_id",
            "network_id", "comparison_status",
            "sensitivity_minus_primary_timing_sec",
            "absolute_timing_difference_sec",
            "used_for_model_fitting_or_classification",
        )),
    )
    exclusion_rows = [{
        "stage": "frozen_selection",
        "map_id": row.get("map_id"),
        "reason": "frozen_manifest_role_excluded",
        "timing_result_used_as_cut": False,
    } for row in registry if row.get("analysis_role") == "excluded"]
    write_csv(
        output / "exclusion_registry.csv", exclusion_rows,
        _fields(exclusion_rows, ("stage", "map_id", "reason", "timing_result_used_as_cut")),
    )
    write_csv(output / "candidate_model_results.csv", candidate_rows, _fields(candidate_rows, (
        "model_id", "model_family", "validation_regime", "heldout_group_count",
        "supported_prediction_count", "timing_rmse_sec", "predictive_pvalue",
        "passes_predictive_gate",
    )))
    write_csv(output / "heldout_predictions.csv", predictions, _fields(predictions, (
        "model_id", "validation_regime", "fold_id", "map_id", "obsnum",
        "network_id", "prediction_status", "supported", "timing_observed_sec",
        "diagnostic_predicted_offset_sec", "timing_residual_after_prediction_sec",
    )))
    write_csv(output / "variance_components.csv", variance_rows, _fields(variance_rows))
    write_csv(output / "network_repeatability.csv", repeatability_rows, _fields(repeatability_rows))
    write_csv(output / "slot_regression_results.csv", regression_rows, _fields(regression_rows, (
        "predictor", "scope", "map_id", "session_id", "available", "beta", "beta_se",
        "beta_95_low", "beta_95_high", "beta_consistent_with_minus_one_95",
    )))
    write_csv(output / "drift_results.csv", drift_rows, _fields(drift_rows))
    write_csv(
        output / "pps_time_increment_occurrence.csv",
        nw9_occurrence_rows,
        _fields(nw9_occurrence_rows),
    )
    write_csv(
        output / "raw_pps_time_increment_anomalies.csv",
        nw9_detail_rows,
        _fields(nw9_detail_rows),
    )
    write_csv(
        output / "nw9_timing_sensitivity.csv",
        nw9_timing_rows,
        _fields(nw9_timing_rows),
    )
    write_json(
        output / "known_omissions.json",
        known_omissions(source_inventory, bundles, combined_network_rows),
    )
    write_json(output / "corpus_summary.json", corpus_summary)
    input_rows = [row for bundle in bundles for row in bundle.input_files]
    write_csv(output / "input_digests.csv", input_rows, ("map_id", "path", "sha256"))
    write_json(output / "input_digests.json", {
        "schema_version": "sci-align-001-3c273-aggregate-input-digests-v1",
        "selected_manifest": {"path": str(manifest), "sha256": manifest_sha},
        "frozen_protocol": {"path": str(protocol_path), "sha256": protocol_file_sha},
        "aggregation_tool": tool_identity,
        "compact_inputs": input_rows,
    })
    _atomic_text(output / "REPORT.md", _report_text(corpus_summary))
    write_plots(output, primary, candidate_rows, regression_rows)
    write_checksums(output)
    print(str(output / "corpus_summary.json"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze provenance-only grouping")
    freeze.add_argument("--selected-manifest", type=Path, required=True)
    freeze.add_argument("--protocol-template", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--dry-run", action="store_true")
    freeze.set_defaults(function=command_freeze)
    run = subparsers.add_parser("run", help="aggregate verified compact map outputs")
    run.add_argument("--selected-manifest", type=Path, required=True)
    run.add_argument("--frozen-protocol", type=Path, required=True)
    run.add_argument("--map-output-root", type=Path, action="append", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(function=lambda args: command_run(args))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (AggregateError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
