#!/usr/bin/env python3
"""Shared read-only contracts for the SCI-ALIGN-001 3C273 corpus runner.

This module is intentionally independent of the frozen single-observation
diagnostics.  It preserves their numerical definitions while replacing their
branch, path, observation, and package globals with explicit inputs.

The module never launches Citlali and never writes an input product.  File
publication helpers accept only an owner-selected output directory and use
atomic replacement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset
from scipy.optimize import least_squares


RUNNER_SCHEMA = "sci-align-001-3c273-map-result-v2"
INPUT_SCHEMA = "sci-align-001-3c273-map-input-v1"
PROTOCOL_SCHEMA = "sci-align-001-3c273-fit-protocol-v1"
SELECTED_MANIFEST_SCHEMA = "sci-align-001-3c273-selected-manifest-v2"
ARRAY_NAMES = {0: "a1100", 1: "a1400", 2: "a2000"}
ARRAY_FWHM_LIMITS = {0: (3.0, 10.0), 1: (3.5, 15.0), 2: (5.5, 20.0)}
DEFAULT_EXPECTED_NETWORKS = tuple(network for network in range(13) if network != 10)
FROZEN_PILOT_UIDS = (0, 5, 10, 15, 20, 25, 30, 35)
RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi


class ContractError(RuntimeError):
    """Raised when a scientific or input identity contract is not satisfied."""


class RawLinkageError(ContractError):
    """Raised when enhanced raw linkage fails but core analysis may remain valid."""


def _json_value(value: Any, location: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item(), location)
    if isinstance(value, np.ndarray):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(
            "non-finite value is prohibited in deterministic output "
            f"at {location}"
        )
    return value


def _finite_or_none(value: float | np.floating[Any]) -> float | None:
    """Represent an unavailable optional diagnostic as JSON null, never NaN.

    A rejected fit can have no invertible covariance or no admitted samples in
    one direction.  Those are retained diagnostic facts, not a reason to
    serialize a non-finite pseudo-measurement or abort a whole map analysis.
    """
    result = float(value)
    return result if math.isfinite(result) else None


def canonical_json(value: Any) -> str:
    """Return the byte-stable JSON representation used in identity digests."""

    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically publish text beneath an already authorized output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")


def csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _json_value(row.get(field, "")) for field in fields})
    return stream.getvalue()


def atomic_write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    atomic_write_text(path, csv_text(rows, fields))


def checksum_lines(directory: Path, excluded: Sequence[str] = ("SHA256SUMS",)) -> str:
    excluded_set = set(excluded)
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded_set:
            lines.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    return "\n".join(lines) + "\n"


def verify_checksums(directory: Path) -> bool:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        return False
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            return False
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def resume_binding_digest(
    protocol: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tool_digests: Mapping[str, str],
) -> str:
    payload = {
        "schema": RUNNER_SCHEMA,
        "protocol": protocol,
        "inputs": sorted(
            (_json_value(row) for row in inputs),
            key=lambda row: (str(row.get("role", "")), str(row.get("path", ""))),
        ),
        "tools": dict(sorted(tool_digests.items())),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def resume_is_valid(directory: Path, binding_sha256: str) -> bool:
    binding_path = directory / "resume_binding.json"
    if not binding_path.is_file() or not (directory / "map_result.json").is_file():
        return False
    try:
        recorded = json.loads(binding_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        recorded.get("binding_sha256") == binding_sha256
        and verify_checksums(directory)
    )


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ContractError(f"invalid boolean value: {value!r}")


def _path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser().resolve()


def _parse_raw_files(value: Any) -> tuple[dict[int, Path], dict[str, str]]:
    if value in (None, ""):
        return {}, {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ContractError(f"raw_files_json is not valid JSON: {error}") from error
    result: dict[int, Path] = {}
    supplied: dict[str, str] = {}
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                payload = dict(item)
                payload.setdefault("network_id", key)
                items.append(payload)
            else:
                items.append({"network_id": key, "path": item})
    elif isinstance(value, list):
        items = value
    else:
        raise ContractError("raw_files_json must be an object or array")
    for item in items:
        if isinstance(item, str):
            match = re.search(r"toltec(\d+)", Path(item).name)
            if not match:
                raise ContractError(f"cannot infer raw network from {item}")
            network = int(match.group(1))
            path = _path(item)
        elif isinstance(item, Mapping):
            identity = _first(item, "network_id", "network", "roach_index", "interface")
            if identity is None:
                raise ContractError(f"raw file entry lacks network identity: {item}")
            text = str(identity)
            network = int(text.removeprefix("toltec"))
            path = _path(_first(item, "path", "filepath", "raw_path"))
        else:
            raise ContractError(f"malformed raw file entry: {item!r}")
        if path is None:
            raise ContractError(f"raw file entry lacks a path: {item}")
        if network in result and result[network] != path:
            raise ContractError(f"duplicate raw paths for network {network}")
        digest = item.get("sha256") if isinstance(item, Mapping) else None
        digest_status = (
            str(item.get("digest_status", "")).strip().lower()
            if isinstance(item, Mapping)
            else ""
        )
        if digest not in (None, ""):
            normalized_digest = str(digest).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", normalized_digest):
                raise ContractError(
                    f"raw toltec{network} has malformed embedded SHA-256"
                )
            prior = supplied.get(str(path))
            if prior is not None and prior != normalized_digest:
                raise ContractError(
                    f"raw toltec{network} has conflicting embedded SHA-256 values"
                )
            supplied[str(path)] = normalized_digest
        elif digest_status == "sha256":
            raise ContractError(
                f"raw toltec{network} claims SHA-256 status without a digest"
            )
        result[network] = path
    return dict(sorted(result.items())), dict(sorted(supplied.items()))


def canonical_network_t0_vector(
    values: Sequence[tuple[int, int]],
) -> list[dict[str, int]]:
    """Return the one canonical document shape used for T0 session identity."""

    return [
        {"network": int(network), "t0": int(t0)}
        for network, t0 in values
    ]


def _parse_network_t0_authority(
    row: Mapping[str, Any], candidate: str
) -> tuple[tuple[tuple[int, int], ...], str | None, str | None]:
    """Parse and authenticate the inventory's canonical network/T0 vector."""

    value = _first(row, "network_t0_vector", "network_t0_vector_json")
    digest_value = row.get("network_t0_vector_sha256")
    status_value = row.get("network_t0_status")
    status = (
        str(status_value).strip()
        if status_value not in (None, "")
        else None
    )
    if value in (None, ""):
        if digest_value not in (None, ""):
            raise ContractError(
                f"candidate {candidate} supplies network_t0_vector_sha256 without "
                "network_t0_vector"
            )
        if status == "complete_unambiguous":
            raise ContractError(
                f"candidate {candidate} claims complete_unambiguous network T0 "
                "status without a vector"
            )
        return (), None, status
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"candidate {candidate} network_t0_vector_json is not valid JSON: {error}"
            ) from error
    if not isinstance(value, list) or not value:
        raise ContractError(
            f"candidate {candidate} network_t0_vector must be a nonempty array"
        )
    parsed: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractError(
                f"candidate {candidate} network_t0_vector[{index}] must be an object"
            )
        network = item.get("network")
        t0 = item.get("t0")
        if not isinstance(network, (int, np.integer)) or isinstance(network, bool):
            raise ContractError(
                f"candidate {candidate} network_t0_vector[{index}].network "
                "must be an integer"
            )
        if not isinstance(t0, (int, np.integer)) or isinstance(t0, bool):
            raise ContractError(
                f"candidate {candidate} network_t0_vector[{index}].t0 "
                "must be an integer"
            )
        if int(network) < 0:
            raise ContractError(
                f"candidate {candidate} network_t0_vector[{index}].network "
                "must be nonnegative"
            )
        parsed.append((int(network), int(t0)))
    networks = [network for network, _ in parsed]
    if networks != sorted(networks) or len(networks) != len(set(networks)):
        raise ContractError(
            f"candidate {candidate} network_t0_vector networks must be unique "
            "and strictly increasing"
        )
    digest = str(digest_value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ContractError(
            f"candidate {candidate} network_t0_vector lacks a valid SHA-256 digest"
        )
    measured = sha256_bytes(
        canonical_json(canonical_network_t0_vector(parsed)).encode("utf-8")
    )
    if digest != measured:
        raise ContractError(
            f"candidate {candidate} network_t0_vector digest mismatch: "
            f"recorded {digest}, measured {measured}"
        )
    if status != "complete_unambiguous":
        raise ContractError(
            f"candidate {candidate} authenticated network_t0_vector requires "
            "network_t0_status='complete_unambiguous'"
        )
    return tuple(parsed), digest, status


@dataclass
class ReductionInputs:
    """Explicit identity and products for one retained Beammap reduction."""

    candidate_id: str
    observation_number: int
    reduction_path: Path
    config_path: Path
    detector_tod_path: Path | None = None
    output_apt_path: Path | None = None
    provenance_path: Path | None = None
    telescope_path: Path | None = None
    project_path: Path | None = None
    raw_by_network: dict[int, Path] = field(default_factory=dict)
    supplied_sha256: dict[str, str] = field(default_factory=dict)
    network_t0_vector: tuple[tuple[int, int], ...] = ()
    network_t0_vector_sha256: str | None = None
    network_t0_status: str | None = None
    analysis_role: str = "primary"
    core_eligible: bool = True
    enhanced_eligible: bool = False

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReductionInputs":
        candidate = str(
            _first(row, "candidate_id", "map_id", "reduction_id", default="")
        ).strip()
        if not candidate:
            raise ContractError("manifest row lacks candidate_id")
        obs_value = _first(row, "observation_number", "obsnum", "observation")
        try:
            obsnum = int(obs_value)
        except (TypeError, ValueError) as error:
            raise ContractError(f"candidate {candidate} has invalid observation number") from error
        reduction = _path(_first(row, "reduction_path", "reduction_root"))
        config = _path(_first(row, "config_path", "realized_config_path"))
        if reduction is None or config is None:
            raise ContractError(
                f"candidate {candidate} requires reduction_path and config_path"
            )
        raw, raw_supplied = _parse_raw_files(
            _first(
                row,
                "raw_files",
                "raw_files_json",
                "raw_by_network",
                default={},
            )
        )
        supplied: dict[str, str] = dict(raw_supplied)
        digest_pairs = (
            (config, _first(row, "config_sha256")),
            (_path(_first(row, "detector_tod_path")), _first(row, "detector_tod_sha256")),
            (_path(_first(row, "telescope_path")), _first(row, "telescope_sha256")),
            (_path(_first(row, "provenance_path")), _first(row, "provenance_sha256")),
            (_path(_first(row, "output_apt_path", "beammap_apt_path")), _first(row, "output_apt_sha256", "beammap_apt_sha256")),
        )
        for path, digest in digest_pairs:
            if path is not None and digest not in (None, ""):
                supplied[str(path)] = str(digest)
        raw_digest_value = _first(row, "raw_sha256_json")
        if raw_digest_value not in (None, ""):
            decoded = json.loads(str(raw_digest_value)) if isinstance(raw_digest_value, str) else raw_digest_value
            if isinstance(decoded, Mapping):
                for key, digest in decoded.items():
                    network = int(str(key).removeprefix("toltec"))
                    if network in raw:
                        supplied[str(raw[network])] = str(digest)
        analysis_role = str(row.get("analysis_role", "primary")).strip().lower()
        if analysis_role not in {"primary", "sensitivity"}:
            raise ContractError(
                f"candidate {candidate} has invalid analysis_role {analysis_role!r}"
            )
        network_t0_vector, network_t0_digest, network_t0_status = (
            _parse_network_t0_authority(row, candidate)
        )
        instance = cls(
            candidate_id=candidate,
            observation_number=obsnum,
            reduction_path=reduction,
            config_path=config,
            detector_tod_path=_path(_first(row, "detector_tod_path")),
            output_apt_path=_path(_first(row, "output_apt_path", "beammap_apt_path", "apt_path")),
            provenance_path=_path(_first(row, "provenance_path")),
            telescope_path=_path(_first(row, "telescope_path")),
            project_path=_path(_first(row, "project_path")),
            raw_by_network=raw,
            supplied_sha256=supplied,
            network_t0_vector=network_t0_vector,
            network_t0_vector_sha256=network_t0_digest,
            network_t0_status=network_t0_status,
            analysis_role=analysis_role,
            core_eligible=parse_bool(row.get("core_eligible"), True),
            enhanced_eligible=parse_bool(row.get("enhanced_eligible"), bool(raw)),
        )
        instance.resolve_products()
        if instance.network_t0_vector:
            authority_networks = [
                network for network, _ in instance.network_t0_vector
            ]
            raw_networks = sorted(instance.raw_by_network)
            if authority_networks != raw_networks:
                raise ContractError(
                    f"candidate {candidate} network_t0_vector networks "
                    f"{authority_networks} differ from raw-file networks {raw_networks}"
                )
        return instance

    def resolve_products(self) -> None:
        """Resolve only uniquely discoverable products; ambiguity is fatal."""

        self.reduction_path = self.reduction_path.resolve()
        self.config_path = self.config_path.resolve()
        raw_dir = self.reduction_path / "raw"
        tod_dir = raw_dir / "source_crossing_tod"
        if self.detector_tod_path is None:
            self.detector_tod_path = unique_file(tod_dir, "*_ptc_detector_tod.nc", "detector TOD")
        if self.output_apt_path is None:
            candidates = [
                path
                for path in sorted(raw_dir.glob("apt_*_citlali.ecsv"))
                if "fit_qc" not in path.name and "prior" not in path.name
            ]
            self.output_apt_path = require_unique(candidates, "output APT")
        if self.provenance_path is None:
            self.provenance_path = self.reduction_path / "timestream_output_provenance.yaml"
        config = load_yaml(self.config_path)
        if self.telescope_path is None:
            for item in config_input_items(config):
                if str(item.get("meta", {}).get("interface", "")).lower() == "lmt":
                    self.telescope_path = _path(item.get("filepath"))
                    break
        if not self.raw_by_network:
            raw = {}
            for item in config_input_items(config):
                interface = str(item.get("meta", {}).get("interface", ""))
                if interface.startswith("toltec"):
                    raw[int(interface.removeprefix("toltec"))] = Path(
                        str(item["filepath"])
                    ).expanduser().resolve()
            self.raw_by_network = dict(sorted(raw.items()))

    def required_paths(self, enhanced: bool) -> list[tuple[str, Path]]:
        values = [
            ("reduction", self.reduction_path),
            ("config", self.config_path),
            ("detector_tod", self.detector_tod_path),
            ("output_apt", self.output_apt_path),
            ("provenance", self.provenance_path),
            ("telescope", self.telescope_path),
        ]
        result = []
        for role, path in values:
            if path is None:
                raise ContractError(f"{self.candidate_id} lacks {role} path")
            result.append((role, path))
        if enhanced:
            result.extend(
                (f"raw_toltec{network}", path)
                for network, path in sorted(self.raw_by_network.items())
            )
        return result

    def validate(self, enhanced: bool) -> None:
        if not self.core_eligible:
            raise ContractError(f"candidate {self.candidate_id} is not core eligible")
        if enhanced and not self.enhanced_eligible:
            raise ContractError(f"candidate {self.candidate_id} is not enhanced eligible")
        for role, path in self.required_paths(enhanced):
            if role == "reduction":
                if not path.is_dir():
                    raise ContractError(f"missing reduction directory: {path}")
            elif not path.is_file():
                raise ContractError(f"missing {role}: {path}")

    def identity(self) -> dict[str, Any]:
        return {
            "schema": INPUT_SCHEMA,
            "candidate_id": self.candidate_id,
            "observation_number": self.observation_number,
            "reduction_path": str(self.reduction_path),
            "project_path": str(self.project_path) if self.project_path else None,
            "config_path": str(self.config_path),
            "detector_tod_path": str(self.detector_tod_path),
            "output_apt_path": str(self.output_apt_path),
            "provenance_path": str(self.provenance_path),
            "telescope_path": str(self.telescope_path),
            "raw_by_network": {
                str(network): str(path)
                for network, path in sorted(self.raw_by_network.items())
            },
            "network_t0_vector": canonical_network_t0_vector(
                self.network_t0_vector
            ),
            "network_t0_vector_sha256": self.network_t0_vector_sha256,
            "network_t0_status": self.network_t0_status,
            "analysis_role": self.analysis_role,
        }


def require_unique(paths: Sequence[Path], identity: str) -> Path:
    if len(paths) != 1:
        raise ContractError(
            f"expected exactly one {identity}, found {len(paths)}: "
            f"{[str(path) for path in paths]}"
        )
    return paths[0].resolve()


def unique_file(directory: Path, pattern: str, identity: str) -> Path:
    return require_unique(sorted(directory.glob(pattern)), identity)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"expected YAML object in {path}")
    return value


def config_input_items(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
        raise ContractError("realized config must contain exactly one input observation")
    items = inputs[0].get("data_items")
    if not isinstance(items, list):
        raise ContractError("realized config lacks inputs[0].data_items")
    return [dict(item) for item in items]


def parse_manifest(path: Path) -> list[ReductionInputs]:
    """Parse and authenticate a frozen selected-manifest JSON document."""

    if path.suffix.lower() == ".csv":
        raise ContractError(
            "per-map execution requires checksum-bound selected_manifest.json, not CSV"
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError("selected manifest must be a JSON object")
    if value.get("schema_version") != SELECTED_MANIFEST_SCHEMA:
        raise ContractError(
            f"unsupported selected-manifest schema: {value.get('schema_version')!r}"
        )
    recorded_digest = str(value.get("manifest_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest):
        raise ContractError("selected manifest lacks a valid internal manifest_sha256")
    source_inventory_digest = str(value.get("source_inventory_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_inventory_digest):
        raise ContractError("selected manifest lacks a valid source_inventory_sha256")
    owner_selection_digest = str(value.get("owner_selection_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", owner_selection_digest):
        raise ContractError("selected manifest lacks a valid owner_selection_sha256")
    if value.get("owner_selection_format") not in {"csv", "json"}:
        raise ContractError("selected manifest owner_selection_format must be csv or json")
    allowlist_digest = str(value.get("obsnum_allowlist_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", allowlist_digest):
        raise ContractError("selected manifest lacks a valid obsnum_allowlist_sha256")
    if value.get("obsnum_allowlist_schema_version") != "sci-align-001-3c273-obsnum-allowlist-v1":
        raise ContractError("selected manifest has unsupported obsnum allowlist schema")
    allowlist_name = str(value.get("obsnum_allowlist_filename", ""))
    if Path(allowlist_name).name != allowlist_name or not allowlist_name.endswith(".json"):
        raise ContractError("selected manifest has invalid obsnum allowlist filename")
    allowlist_path = path.parent / allowlist_name
    if not allowlist_path.is_file() or sha256_file(allowlist_path) != allowlist_digest:
        raise ContractError("selected manifest obsnum allowlist copy/digest is invalid")
    digest_payload = {
        key: item for key, item in value.items() if key != "manifest_sha256"
    }
    measured_digest = sha256_bytes(
        canonical_json(digest_payload).encode("utf-8")
    )
    if recorded_digest != measured_digest:
        raise ContractError(
            "selected manifest internal digest mismatch: "
            f"recorded {recorded_digest}, measured {measured_digest}"
        )
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ContractError("selected manifest must contain a rows array")
    result = [ReductionInputs.from_mapping(row) for row in rows]
    identities = [item.candidate_id for item in result]
    if len(identities) != len(set(identities)):
        raise ContractError("manifest contains duplicate candidate_id values")
    primaries_by_observation: dict[int, int] = defaultdict(int)
    for item in result:
        if item.analysis_role == "primary":
            primaries_by_observation[item.observation_number] += 1
    duplicate_primary_observations = sorted(
        observation
        for observation, count in primaries_by_observation.items()
        if count > 1
    )
    if duplicate_primary_observations:
        raise ContractError(
            "selected manifest contains more than one primary candidate for observations "
            f"{duplicate_primary_observations}"
        )
    observations_without_primary = sorted(
        {item.observation_number for item in result} - set(primaries_by_observation)
    )
    if observations_without_primary:
        raise ContractError(
            "selected manifest lacks exactly one primary candidate for observations "
            f"{observations_without_primary}"
        )
    return sorted(
        result,
        key=lambda item: (
            item.observation_number,
            item.candidate_id,
            str(item.reduction_path),
        ),
    )


@dataclass(frozen=True)
class AnalysisProtocol:
    schema: str = PROTOCOL_SCHEMA
    central_fraction: float = 0.8
    stable_sign_fraction: float = 0.99
    low_speed_fraction_of_minimum: float = 0.5
    radial_fwhm_multiplier: float = 4.0
    minimum_amplitude_snr: float = 3.0
    fit_boundary_margin_minimum: float = 1.0e-3
    minimum_detector_samples: int = 100
    minimum_distinct_scans_per_direction: int = 3
    minimum_matched_detectors: int = 100
    map_extent_arcsec: int = 80
    map_pixel_arcsec: float = 1.0
    minimum_map_pixel_count: int = 5
    minimum_fitted_map_pixels: int = 100
    expected_networks: tuple[int, ...] = DEFAULT_EXPECTED_NETWORKS
    excluded_uids: tuple[int, ...] = ()
    fixture_exclusion_observation: int = 148670
    fixture_excluded_uids: tuple[int, ...] = FROZEN_PILOT_UIDS
    common_row_shifts: tuple[int, ...] = (-1, 0, 1)
    enhanced_models: tuple[tuple[str, int, float], ...] = (
        ("assigned_slot", 0, 0.0),
        ("raw_detector_timestamp", 0, 0.0),
        ("assigned_slot", 1, 0.5),
        ("raw_detector_timestamp", 1, 0.5),
    )
    authority_document_sha256: str = ""
    authority_schema_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @classmethod
    def from_json(cls, path: Path | None) -> "AnalysisProtocol":
        if path is None:
            return cls()
        value = json.loads(path.read_text())
        if value.get("schema_version") == "sci-align-001-3c273-corpus-protocol-v2":
            required_sections = {
                "aggregation",
                "decision",
                "eligibility",
                "fit_quality",
                "identity",
                "outputs",
                "producer_authority",
                "prohibited",
                "raw_linkage",
                "raw_phase_and_counter_diagnostics",
                "scan_classifier",
                "timing_models",
                "uncertainty_and_controls",
            }
            missing_sections = sorted(required_sections - set(value))
            if missing_sections:
                raise ContractError(
                    f"frozen corpus protocol lacks sections: {missing_sections}"
                )
            if value.get("status") != "FROZEN_BEFORE_CORPUS_TIMING_RESULTS":
                raise ContractError("corpus protocol does not have frozen-before-results status")
            fit = value["fit_quality"]
            scan = value["scan_classifier"]
            timing = value["timing_models"]
            if not all(isinstance(item, Mapping) for item in (fit, scan, timing)):
                raise ContractError("frozen fit, scan, and timing sections must be objects")
            raw_diagnostics = value["raw_phase_and_counter_diagnostics"]
            if not isinstance(raw_diagnostics, Mapping) or not isinstance(
                raw_diagnostics.get("nw9_pps_time_increment_anomaly"), Mapping
            ):
                raise ContractError("frozen raw-counter protocol lacks nw9 anomaly diagnostics")
            required_fit_keys = {
                "fwhm_bounds_arcsec_by_array",
                "fit_boundary_margin_minimum",
                "expected_networks",
                "legacy_feasibility_exclusions_by_observation",
                "map_extent_arcsec",
                "map_minimum_pixel_count",
                "map_pixel_size_arcsec",
                "minimum_distinct_scans_per_direction",
                "minimum_fitted_map_pixels",
                "minimum_amplitude_snr",
                "minimum_matched_detectors",
                "minimum_samples_per_direction",
                "radial_fwhm_multiplier",
                "radial_support",
                "timing_result_not_a_cut",
            }
            missing_fit_keys = sorted(required_fit_keys - set(fit))
            if missing_fit_keys:
                raise ContractError(
                    f"frozen fit-quality protocol lacks keys: {missing_fit_keys}"
                )
            expected_fwhm = {
                ARRAY_NAMES[array_id]: list(bounds)
                for array_id, bounds in ARRAY_FWHM_LIMITS.items()
            }
            if fit["fwhm_bounds_arcsec_by_array"] != expected_fwhm:
                raise ContractError("frozen FWHM bounds differ from implemented fit kernel")
            exclusions = fit["legacy_feasibility_exclusions_by_observation"]
            if not isinstance(exclusions, Mapping) or set(exclusions) != {"148670"}:
                raise ContractError("unsupported observation-specific exclusion registry")
            fixture_excluded = tuple(int(item) for item in exclusions["148670"])
            if fixture_excluded != FROZEN_PILOT_UIDS:
                raise ContractError("148670 exclusions differ from frozen predecessor")
            if fit["timing_result_not_a_cut"] is not True:
                raise ContractError("timing-result blinding cut is not frozen true")
            required_scan_keys = {
                "central_fraction",
                "direction_measurement",
                "low_speed_fraction_of_minimum",
                "low_speed_threshold",
                "stable_sign_fraction_minimum",
            }
            missing_scan_keys = sorted(required_scan_keys - set(scan))
            if missing_scan_keys:
                raise ContractError(
                    f"frozen scan-classifier protocol lacks keys: {missing_scan_keys}"
                )
            models = tuple(
                (
                    str(item["basis"]),
                    int(item["k"]),
                    float(item["phi_samples"]),
                )
                for item in timing.get("comparison_models", [])
            )
            required_models = {
                ("assigned_slot", 0, 0.0),
                ("raw_detector_timestamp", 0, 0.0),
                ("assigned_slot", 1, 0.5),
                ("raw_detector_timestamp", 1, 0.5),
            }
            if len(models) != 4 or set(models) != required_models:
                raise ContractError("frozen timing-model registry differs from implemented four models")
            row_shifts = tuple(int(item) for item in timing.get("guard_rows_k", []))
            if row_shifts != (-1, 0, 1):
                raise ContractError("frozen timing-model row guards must be [-1,0,1]")
            kwargs: dict[str, Any] = {
                "central_fraction": float(scan["central_fraction"]),
                "stable_sign_fraction": float(scan["stable_sign_fraction_minimum"]),
                "low_speed_fraction_of_minimum": float(
                    scan["low_speed_fraction_of_minimum"]
                ),
                "radial_fwhm_multiplier": float(fit["radial_fwhm_multiplier"]),
                "minimum_amplitude_snr": float(fit["minimum_amplitude_snr"]),
                "fit_boundary_margin_minimum": float(
                    fit["fit_boundary_margin_minimum"]
                ),
                "minimum_detector_samples": int(fit["minimum_samples_per_direction"]),
                "minimum_distinct_scans_per_direction": int(
                    fit["minimum_distinct_scans_per_direction"]
                ),
                "minimum_matched_detectors": int(fit["minimum_matched_detectors"]),
                "map_extent_arcsec": int(fit["map_extent_arcsec"]),
                "map_pixel_arcsec": float(fit["map_pixel_size_arcsec"]),
                "minimum_map_pixel_count": int(fit["map_minimum_pixel_count"]),
                "minimum_fitted_map_pixels": int(fit["minimum_fitted_map_pixels"]),
                "expected_networks": tuple(int(item) for item in fit["expected_networks"]),
                "fixture_excluded_uids": fixture_excluded,
                "common_row_shifts": row_shifts,
                "enhanced_models": models,
                "authority_document_sha256": sha256_file(path),
                "authority_schema_version": str(value["schema_version"]),
            }
            expected_runtime_values = {
                "minimum_detector_samples": 100,
                "minimum_distinct_scans_per_direction": 3,
                "minimum_matched_detectors": 100,
                "central_fraction": 0.8,
                "stable_sign_fraction": 0.99,
                "low_speed_fraction_of_minimum": 0.5,
                "radial_fwhm_multiplier": 4.0,
                "minimum_amplitude_snr": 3.0,
                "fit_boundary_margin_minimum": 1.0e-3,
                "map_extent_arcsec": 80,
                "map_pixel_arcsec": 1.0,
                "minimum_map_pixel_count": 5,
                "minimum_fitted_map_pixels": 100,
                "expected_networks": DEFAULT_EXPECTED_NETWORKS,
            }
            for name, expected in expected_runtime_values.items():
                if kwargs[name] != expected:
                    raise ContractError(
                        f"frozen {name}={kwargs[name]!r} differs from implemented {expected!r}"
                    )
            return cls(**kwargs)
        if value.get("schema") not in (None, PROTOCOL_SCHEMA):
            raise ContractError(f"unsupported protocol schema: {value.get('schema')}")
        accepted = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - accepted)
        if unknown:
            raise ContractError(f"unknown protocol fields: {unknown}")
        for name in (
            "expected_networks",
            "excluded_uids",
            "fixture_excluded_uids",
            "common_row_shifts",
        ):
            if name in value:
                value[name] = tuple(int(item) for item in value[name])
        if "enhanced_models" in value:
            value["enhanced_models"] = tuple(
                (str(item[0]), int(item[1]), float(item[2]))
                for item in value["enhanced_models"]
            )
        return cls(**value)

    def excluded_uids_for_observation(self, observation_number: int) -> tuple[int, ...]:
        values = set(int(value) for value in self.excluded_uids)
        if int(observation_number) == self.fixture_exclusion_observation:
            values.update(int(value) for value in self.fixture_excluded_uids)
        return tuple(sorted(values))


@dataclass
class AlignmentState:
    cadence_sec: float
    phase_sec: float
    sample_count: int
    union_local_start: int
    records: list[dict[str, Any]]
    ordinal_to_stable: dict[int, int]
    interface_residuals: dict[int, float]
    provenance: dict[str, Any]


def load_alignment_state(path: Path) -> AlignmentState:
    provenance = load_yaml(path)
    try:
        realized = provenance["realized"]
        alignment = realized["sci_align_alignment"]
        scan_plan = realized["sci_align_scan_plan"]
        grid = alignment["grid"]
        axis = alignment["governing_compatibility_axis"]
    except (KeyError, TypeError) as error:
        raise ContractError(f"missing SCI-ALIGN provenance contract in {path}") from error
    cadence = float(grid["cadence_sec"])
    phase = float(grid["phase_sec"])
    count = int(axis["sample_count"])
    union_start = int(axis.get("union_local_start", 0))
    if not (math.isfinite(cadence) and cadence > 0 and math.isfinite(phase) and count > 0):
        raise ContractError("invalid realized alignment grid")
    records = [
        dict(row)
        for row in scan_plan.get("records", [])
        if row.get("legacy_processing_admitted")
    ]
    records.sort(key=lambda row: int(row["compatibility_ordinal"]))
    ordinals = [int(row["compatibility_ordinal"]) for row in records]
    if ordinals != list(range(len(records))):
        raise ContractError("admitted compatibility ordinals are not dense from zero")
    stable = [int(row["stable_id"]) for row in records]
    if len(stable) != len(set(stable)):
        raise ContractError("admitted stable scan identities are not unique")
    for row in records:
        science = row.get("compatibility_science")
        if not isinstance(science, dict):
            raise ContractError("admitted scan lacks compatibility_science")
        start = int(science["start"]) - union_start
        stop = int(science["stop"]) - union_start
        if start < 0 or stop <= start or stop > count:
            raise ContractError(f"scan {row['stable_id']} is outside retained sample support")
    interface_residuals: dict[int, float] = {}
    for item in alignment.get("interfaces", []):
        name = str(item.get("interface_id", ""))
        if not name.startswith("toltec"):
            continue
        network = int(name.removeprefix("toltec"))
        interface_residuals[network] = 0.5 * (
            float(item["minimum_residual_sec"]) + float(item["maximum_residual_sec"])
        )
    return AlignmentState(
        cadence,
        phase,
        count,
        union_start,
        records,
        {int(row["compatibility_ordinal"]) + 1: int(row["stable_id"]) for row in records},
        interface_residuals,
        provenance,
    )


def periodic_fix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.size and float(np.max(result)) > 1.99 * math.pi and float(np.min(result)) < math.pi:
        result[result < math.pi] += 2.0 * math.pi
    return result


def astrometry_offsets(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ContractError("expected one realized config input for astrometry")
    cal_items = inputs[0].get("cal_items", [])
    matches = [item for item in cal_items if item.get("type") == "astrometry"]
    if len(matches) != 1:
        raise ContractError(f"expected one astrometry calibration item, found {len(matches)}")
    offsets = matches[0].get("pointing_offsets")
    if not isinstance(offsets, list):
        raise ContractError("astrometry calibration lacks pointing_offsets")
    values = {
        item.get("axes_name", "mjd"): item.get(
            "value_arcsec", item.get("modified_julian_date")
        )
        for item in offsets
    }
    if not {"mjd", "az", "alt"}.issubset(values):
        raise ContractError("astrometry pointing offsets require mjd, az, and alt")
    times = ((np.asarray(values["mjd"], dtype=float) - 40587.0) * 86400.0).astype(
        np.int64
    ).astype(float)
    az = np.asarray(values["az"], dtype=float)
    alt = np.asarray(values["alt"], dtype=float)
    if times.ndim != 1 or times.size == 0 or az.shape != times.shape or alt.shape != times.shape:
        raise ContractError("astrometry offset axes have inconsistent shapes")
    return times, az, alt


class TelescopeEvaluator:
    """Evaluate the retained AltAz tangent-plane trajectory at explicit times."""

    def __init__(self, telescope_path: Path, config: Mapping[str, Any]) -> None:
        names = ("TelAzAct", "TelElAct", "TelAzCor", "TelElCor", "SourceAz", "SourceEl")
        with Dataset(telescope_path) as dataset:
            required = [f"Data.TelescopeBackend.{name}" for name in names]
            required += ["Data.TelescopeBackend.TelTime", "Data.TelescopeBackend.Hold"]
            missing = [name for name in required if name not in dataset.variables]
            if missing:
                raise ContractError(f"telescope product lacks variables: {missing}")
            self.native_time = np.asarray(
                dataset["Data.TelescopeBackend.TelTime"][:], dtype=float
            )
            self.fields = {
                name: periodic_fix(
                    np.asarray(dataset[f"Data.TelescopeBackend.{name}"][:], dtype=float)
                )
                for name in names
            }
            raw_hold = np.asarray(dataset["Data.TelescopeBackend.Hold"][:], dtype=float)
            if "Header.Map.ScanAngle" not in dataset.variables:
                raise ContractError("telescope product lacks Header.Map.ScanAngle")
            self.scan_angle = float(np.asarray(dataset["Header.Map.ScanAngle"][:]).item())
        if (
            self.native_time.ndim != 1
            or self.native_time.size < 2
            or not np.all(np.isfinite(self.native_time))
            or np.any(np.diff(self.native_time) <= 0)
        ):
            raise ContractError("telescope time axis is not finite and increasing")
        if any(values.shape != self.native_time.shape for values in self.fields.values()):
            raise ContractError("telescope field cardinalities differ")
        if (
            raw_hold.shape != self.native_time.shape
            or not np.all(np.isfinite(raw_hold))
            or np.any(raw_hold < 0)
            or np.any(raw_hold != np.floor(raw_hold))
        ):
            raise ContractError("Hold words must be finite nonnegative integers")
        self.hold = raw_hold.astype(np.uint64)
        self.mjd_time, self.pointing_az, self.pointing_alt = astrometry_offsets(config)

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
        ) * RAD_TO_ARCSEC
        x = (
            np.cos(aligned["TelElAct"] - aligned["TelElCor"])
            * (tel_az - aligned["SourceAz"])
            - aligned["TelAzCor"]
        ) * RAD_TO_ARCSEC
        x += np.interp(safe, self.mjd_time, self.pointing_az)
        y += np.interp(safe, self.mjd_time, self.pointing_alt)
        left = np.searchsorted(self.native_time, safe, side="right") - 1
        right = np.searchsorted(self.native_time, safe, side="left")
        bracket = finite & (left >= 0) & (right < self.native_time.size)
        left_safe = np.clip(left, 0, self.native_time.size - 1)
        right_safe = np.clip(right, 0, self.native_time.size - 1)
        hold_left = self.hold[left_safe]
        hold_right = self.hold[right_safe]
        valid = bracket & (hold_left == 0) & (hold_right == 0) & (hold_left == hold_right)
        x[~finite] = np.nan
        y[~finite] = np.nan
        return {
            "time": target,
            "x": x,
            "y": y,
            "elevation": aligned["TelElAct"],
            "bracket": bracket,
            "valid": valid,
            "hold_left": hold_left,
            "hold_right": hold_right,
            "hold_transition": hold_left != hold_right,
        }


def classify_scan_direction(
    projected_velocity: np.ndarray,
    hold_valid: np.ndarray,
    low_speed_threshold: float,
    stable_sign_fraction: float = 0.99,
) -> tuple[str, str, float]:
    """Classify one already-centralized scan without inspecting signal data."""

    projected = np.asarray(projected_velocity, dtype=float)
    hold_valid = np.asarray(hold_valid, dtype=bool)
    if projected.ndim != 1 or projected.size == 0 or hold_valid.shape != projected.shape:
        raise ContractError("direction-classifier arrays have incompatible shapes")
    if not np.all(np.isfinite(projected)):
        return "excluded", "nonfinite_velocity", 0.0
    median = float(np.median(projected))
    direction = "right" if median > 0 else "left"
    sign = projected > 0 if direction == "right" else projected < 0
    fraction = float(np.mean(sign))
    if not bool(np.all(hold_valid)):
        return "excluded", "hold_invalid_or_transition_ambiguous", fraction
    if abs(median) <= low_speed_threshold or fraction < stable_sign_fraction:
        return "excluded", "low_speed_or_direction_ambiguous", fraction
    return direction, "selected", fraction


def build_scan_registry(
    state: AlignmentState,
    telescope: Mapping[str, np.ndarray],
    scan_angle: float,
    protocol: AnalysisProtocol,
    trajectory_authority: Path,
    scan_authority: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    work = []
    median_velocities = []
    dt = state.cadence_sec
    for record in state.records:
        science = record["compatibility_science"]
        start = int(science["start"]) - state.union_local_start
        stop = int(science["stop"]) - state.union_local_start
        indices = np.arange(start, stop, dtype=np.int64)
        vx = np.gradient(telescope["x"][indices], dt)
        vy = np.gradient(telescope["y"][indices], dt)
        trim = max(1, int(math.floor(indices.size * (1.0 - protocol.central_fraction) / 2.0)))
        if indices.size <= 2 * trim:
            raise ContractError(f"scan {record['stable_id']} is too short for direction cut")
        central = slice(trim, indices.size - trim)
        median = np.array([np.median(vx[central]), np.median(vy[central])])
        median_velocities.append(median)
        work.append((record, start, stop, indices, vx, vy, central, median))
    if not work:
        raise ContractError("no admitted scans are available")
    matrix = np.asarray(median_velocities)
    _, vectors = np.linalg.eigh(matrix.T @ matrix)
    axis = vectors[:, -1]
    configured = np.array([math.cos(scan_angle), math.sin(scan_angle)])
    if float(axis @ configured) < 0:
        axis = -axis
    preliminary = np.abs(matrix @ axis)
    if not np.all(np.isfinite(preliminary)) or float(np.min(preliminary)) <= 0:
        raise ContractError("scan-direction speed scale is not identifiable")
    low_speed = protocol.low_speed_fraction_of_minimum * float(np.min(preliminary))
    cross_axis = np.array([-axis[1], axis[0]])
    rows = []
    for record, start, stop, indices, vx, vy, central, median in work:
        projected = vx * axis[0] + vy * axis[1]
        perpendicular = vx * cross_axis[0] + vy * cross_axis[1]
        hold_valid = (
            (telescope["hold_left"][indices] == 0)
            & (telescope["hold_right"][indices] == 0)
            & ~telescope["hold_transition"][indices]
        )
        classification, reason, sign_fraction = classify_scan_direction(
            projected[central],
            hold_valid[central],
            low_speed,
            protocol.stable_sign_fraction,
        )
        rows.append(
            {
                "stable_scan_id": int(record["stable_id"]),
                "compatibility_ordinal_1based": int(record["compatibility_ordinal"]) + 1,
                "compatibility_status": str(record.get("status", "")),
                "science_start_sample_inclusive": start,
                "science_stop_sample_exclusive": stop,
                "direction_measure_start_sample_inclusive": int(indices[central][0]),
                "direction_measure_stop_sample_exclusive": int(indices[central][-1] + 1),
                "median_vx_arcsec_s": float(median[0]),
                "median_vy_arcsec_s": float(median[1]),
                "median_projected_velocity_arcsec_s": float(np.median(projected[central])),
                "median_perpendicular_velocity_arcsec_s": float(np.median(perpendicular[central])),
                "projected_velocity_mad_sigma_arcsec_s": float(
                    1.4826
                    * np.median(
                        np.abs(projected[central] - np.median(projected[central]))
                    )
                ),
                "projected_velocity_min_arcsec_s": float(np.min(projected[central])),
                "projected_velocity_max_arcsec_s": float(np.max(projected[central])),
                "projected_velocity_median_sign_fraction": sign_fraction,
                "hold_valid_fraction": float(np.mean(hold_valid)),
                "hold_transition_ambiguous_count": int(
                    np.sum(telescope["hold_transition"][indices])
                ),
                "classification": classification,
                "selected": classification in {"left", "right"},
                "exclusion_reason": reason,
                "trajectory_authority": str(trajectory_authority),
                "scan_authority": str(scan_authority),
            }
        )
    return rows, axis, low_speed


def group_selected_scans(
    registry: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    groups = {"left": [], "right": [], "excluded": []}
    ordered = sorted(registry, key=lambda row: int(row["compatibility_ordinal_1based"]))
    for row in ordered:
        classification = str(row["classification"])
        key = classification if classification in {"left", "right"} else "excluded"
        groups[key].append(int(row["stable_scan_id"]))
    return groups


def reconstruct_legacy_timestamp(
    timestamp_fields: np.ndarray,
    fpga_hz: float,
    packet_counts: np.ndarray | None = None,
) -> np.ndarray:
    raw_fields = np.asarray(timestamp_fields)
    fields = np.asarray(raw_fields, dtype=np.float64)
    if fields.ndim != 2 or fields.shape[0] == 0 or fields.shape[1] != 6:
        raise ContractError("legacy detector timestamp must have nonzero [row,6] shape")
    if not math.isfinite(fpga_hz) or fpga_hz <= 0 or not np.all(np.isfinite(fields)):
        raise ContractError("legacy timestamp inputs are not finite with positive FPGA rate")
    anchor_expression = fields[0, 0] + fields[0, 5] * 1.0e-9 - 0.5
    if anchor_expression < np.iinfo(np.int32).min or anchor_expression > np.iinfo(np.int32).max:
        raise ContractError("legacy timestamp anchor exceeds C++ int")
    anchor = int(anchor_expression)
    delta = fields[:, 2] - fields[:, 4]
    delta[fields[:, 2] < fields[:, 4]] += 4294967295.0
    result = anchor + fields[:, 1] + delta / fpga_hz
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise ContractError("reconstructed detector timestamps are not strictly increasing")
    if packet_counts is None:
        packet_counts = raw_fields[:, 3]
    packet_counts = np.asarray(packet_counts)
    if packet_counts.shape != (fields.shape[0],):
        raise ContractError("PacketCount cardinality differs from timestamp rows")
    # PacketCount is delivered transport metadata.  Its non-unit increments
    # are counted and preserved by raw_counter_diagnostics(), but do not alter
    # the delivered D[n]/Ts[n] pairing.  The strict reconstructed-time and
    # native-to-slot checks below remain the Stage-A row-lineage boundary and
    # still fail if an admitted science row is genuinely absent or ambiguous.
    _u32(packet_counts)
    return result


@dataclass
class RawMapping:
    interface: str
    network: int
    path: Path
    times: np.ndarray
    slots: np.ndarray
    assigned: np.ndarray
    residual: np.ndarray
    row_for_slot: np.ndarray
    packet_gap_events: int
    sample_rate_hz: float
    cadence_sec: float
    fpga_hz: float
    accumulation_ticks: int
    timestamp_fields: np.ndarray
    counter_transitions: list[dict[str, Any]]
    phase_summary: dict[str, Any]
    pps_time_increment_anomalies: list[dict[str, Any]] = field(default_factory=list)


def build_row_mapping(
    times: np.ndarray,
    phase_sec: float,
    cadence_sec: float,
    sample_count: int,
    *,
    iq_rows: int | None = None,
    q_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ContractError("native timestamp axis is empty or non-finite")
    if np.any(np.diff(times) <= 0):
        raise ContractError("native timestamp row order is not strictly increasing")
    if iq_rows is not None and (times.size != iq_rows or times.size != q_rows):
        raise ContractError("I/Q/Ts raw row cardinality mismatch")
    slots = np.floor((times - phase_sec) / cadence_sec + 0.5).astype(np.int64)
    if np.any(np.diff(slots) == 0):
        raise ContractError("multiple native detector rows occupy one assigned slot")
    if np.any(np.diff(slots) < 0):
        raise ContractError("native-to-slot mapping reverses")
    assigned = phase_sec + slots.astype(float) * cadence_sec
    residual = times - assigned
    half_cell = 0.5 * cadence_sec
    if np.any(np.abs(residual) >= half_cell) or np.any(
        np.isclose(np.abs(residual), half_cell, rtol=0.0, atol=np.spacing(half_cell))
    ):
        raise ContractError("native timestamp is on or outside the exclusive half cell")
    row_for_slot = np.full(sample_count, -1, dtype=np.int64)
    inside = (slots >= 0) & (slots < sample_count)
    row_for_slot[slots[inside]] = np.flatnonzero(inside)
    return slots, assigned, residual, row_for_slot


def interface_offset_map(config: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in config.get("interface_sync_offset", []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ContractError(f"malformed interface_sync_offset entry: {entry}")
        key, value = next(iter(entry.items()))
        result[str(key)] = float(value)
    return result


def prove_raw_mapping(
    path: Path,
    network: int,
    observation_number: int,
    state: AlignmentState,
    offset_sec: float,
    science_mask: np.ndarray,
) -> RawMapping:
    with Dataset(path) as dataset:
        required = [
            "Header.Toltec.RoachIndex",
            "Header.Toltec.ObsNum",
            "Header.Toltec.FpgaFreq",
            "Header.Toltec.AccumLen",
            "Header.Toltec.SampleFreq",
            "Data.Toltec.Ts",
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
        ]
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ContractError(f"raw toltec{network} lacks variables: {missing}")
        roach = int(np.asarray(dataset["Header.Toltec.RoachIndex"][...]).item())
        raw_observation = int(
            np.asarray(dataset["Header.Toltec.ObsNum"][...]).item()
        )
        fpga = float(np.asarray(dataset["Header.Toltec.FpgaFreq"][...]).item())
        accum = int(np.asarray(dataset["Header.Toltec.AccumLen"][...]).item())
        sample_rate = float(np.asarray(dataset["Header.Toltec.SampleFreq"][...]).item())
        ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
        i_rows = int(dataset["Data.Toltec.Is"].shape[0])
        q_rows = int(dataset["Data.Toltec.Qs"].shape[0])
    if roach != network:
        raise ContractError(f"raw interface toltec{network} has RoachIndex {roach}")
    if raw_observation != observation_number:
        raise ContractError(
            f"raw toltec{network} observation {raw_observation} != {observation_number}"
        )
    if accum <= 0 or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ContractError(f"raw toltec{network} has invalid timing headers")
    transition_rows, transition_summary, anomaly_rows = raw_counter_diagnostics(
        ts, network, fpga, accum
    )
    reconstructed = reconstruct_legacy_timestamp(ts, fpga, ts[:, 3]) + offset_sec
    cadence = accum / fpga
    if not math.isclose(cadence, state.cadence_sec, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            f"raw toltec{network} cadence {cadence} != provenance {state.cadence_sec}"
        )
    slots, assigned, residual, row_for_slot = build_row_mapping(
        reconstructed,
        state.phase_sec,
        state.cadence_sec,
        state.sample_count,
        iq_rows=i_rows,
        q_rows=q_rows,
    )
    if science_mask.shape != (state.sample_count,):
        raise ContractError("science support mask has wrong cardinality")
    missing_science = int(np.sum(science_mask & (row_for_slot < 0)))
    if missing_science:
        raise ContractError(
            f"raw toltec{network} lacks {missing_science} original admitted science rows"
        )
    return RawMapping(
        f"toltec{network}",
        network,
        path,
        reconstructed,
        slots,
        assigned,
        residual,
        row_for_slot,
        int(transition_summary["packet_increment_mismatch_count"]),
        sample_rate,
        cadence,
        fpga,
        accum,
        ts,
        transition_rows,
        transition_summary,
        anomaly_rows,
    )


def build_common_support(
    mapping: RawMapping, sample_count: int, shifts: Sequence[int] = (-1, 0, 1)
) -> np.ndarray:
    source = mapping.row_for_slot
    if source.shape != (sample_count,):
        raise ContractError("row_for_slot cardinality differs from common axis")
    common = source >= 0
    slots = np.arange(sample_count, dtype=np.int64)
    for shift in shifts:
        shifted = source + int(shift)
        valid = common & (shifted >= 0) & (shifted < mapping.times.size)
        safe = np.clip(shifted, 0, mapping.times.size - 1)
        common &= valid & (mapping.slots[safe] == slots + int(shift))
    return common


def raw_residual_summary(mapping: RawMapping) -> dict[str, Any]:
    inside = (mapping.slots >= 0) & (mapping.slots < mapping.row_for_slot.size)
    times = mapping.times[inside]
    residual = mapping.residual[inside]
    if residual.size < 3:
        raise ContractError(f"toltec{mapping.network} has insufficient raw residual samples")
    thirds = np.array_split(np.arange(residual.size), 3)
    elapsed = times - times[0]
    centered = elapsed - np.mean(elapsed)
    scale = float(np.max(np.abs(centered)))
    if math.isfinite(scale) and scale > 0:
        normalized_time = centered / scale
        denominator = float(np.sum(normalized_time * normalized_time, dtype=np.float64))
        slope = float(
            np.sum(
                normalized_time * (residual - np.mean(residual)),
                dtype=np.float64,
            )
            / (scale * denominator)
        )
    else:
        slope = None
    result = {
        "network_id": mapping.network,
        "raw_linkage_status": "proved_original_row_one_to_one",
        "native_row_count": int(mapping.times.size),
        "packet_gap_events": mapping.packet_gap_events,
        "native_to_assigned_begin_sec": float(np.mean(residual[thirds[0]])),
        "native_to_assigned_middle_sec": float(np.mean(residual[thirds[1]])),
        "native_to_assigned_end_sec": float(np.mean(residual[thirds[2]])),
        "native_to_assigned_mean_sec": float(np.mean(residual)),
        "native_to_assigned_min_sec": float(np.min(residual)),
        "native_to_assigned_max_sec": float(np.max(residual)),
        "native_to_assigned_within_observation_variation_slope_sec_per_sec": slope,
        "raw_timestamp_physical_semantics": "unresolved",
        "clock_drift_claimed": False,
        "stage_a_boundary": (
            "proved from delivered D[n]/Ts[n] pair through retained sample; "
            "upstream FPGA metadata-to-integration association remains unresolved"
        ),
    }
    result.update(mapping.phase_summary)
    return result


def resolve_network_t0_session(
    inputs: ReductionInputs,
    raw_rows: Sequence[Mapping[str, Any]],
    analyzed_detector_networks: Sequence[int],
    enhanced: bool,
) -> dict[str, Any]:
    """Bind raw-recomputed T0 values to the selected-manifest authority."""

    analyzed_networks = sorted(set(int(value) for value in analyzed_detector_networks))
    raw_by_network: dict[int, Mapping[str, Any]] = {}
    for row in raw_rows:
        if "network_id" not in row:
            raise RawLinkageError("raw T0 summary row lacks network_id")
        network = int(row["network_id"])
        if network in raw_by_network:
            raise RawLinkageError(
                f"raw T0 summary contains duplicate network toltec{network}"
            )
        raw_by_network[network] = row

    missing_t0_networks = sorted(set(analyzed_networks) - set(raw_by_network))
    ambiguous_t0_networks = sorted(
        network
        for network in analyzed_networks
        if network in raw_by_network
        and (
            int(raw_by_network[network].get("t0_integer_value_count", 0)) != 1
            or not isinstance(
                raw_by_network[network].get("t0_integer_sec"),
                (int, np.integer),
            )
            or isinstance(raw_by_network[network].get("t0_integer_sec"), bool)
        )
    )
    expected_raw_networks = sorted(inputs.raw_by_network) if enhanced else []
    missing_raw_summary_networks = sorted(
        set(expected_raw_networks) - set(raw_by_network)
    )
    extra_raw_summary_networks = sorted(
        set(raw_by_network) - set(expected_raw_networks)
    )
    ambiguous_raw_summary_networks = sorted(
        network
        for network in expected_raw_networks
        if network in raw_by_network
        and (
            int(raw_by_network[network].get("t0_integer_value_count", 0)) != 1
            or not isinstance(
                raw_by_network[network].get("t0_integer_sec"),
                (int, np.integer),
            )
            or isinstance(raw_by_network[network].get("t0_integer_sec"), bool)
        )
    )
    raw_vector_complete = bool(enhanced) and not (
        missing_raw_summary_networks
        or extra_raw_summary_networks
        or ambiguous_raw_summary_networks
    )
    raw_vector = (
        [
            {
                "network": network,
                "t0": int(raw_by_network[network]["t0_integer_sec"]),
            }
            for network in expected_raw_networks
        ]
        if raw_vector_complete
        else []
    )
    raw_digest = (
        sha256_bytes(canonical_json(raw_vector).encode("utf-8"))
        if raw_vector_complete
        else None
    )
    authority_vector = canonical_network_t0_vector(inputs.network_t0_vector)
    authority_digest = inputs.network_t0_vector_sha256
    authority_validated = False
    if authority_digest is not None and enhanced:
        if not raw_vector_complete:
            raise RawLinkageError(
                "cannot validate selected-manifest network_t0_vector against raw "
                "T0 values: "
                f"missing={missing_raw_summary_networks}, "
                f"extra={extra_raw_summary_networks}, "
                f"ambiguous={ambiguous_raw_summary_networks}"
            )
        if raw_vector != authority_vector or raw_digest != authority_digest:
            raise RawLinkageError(
                "raw-recomputed network T0 vector differs from selected-manifest "
                "authority: "
                f"manifest={canonical_json(authority_vector)}, "
                f"raw={canonical_json(raw_vector)}, "
                f"manifest_sha256={authority_digest}, raw_sha256={raw_digest}"
            )
        authority_validated = True

    if authority_validated:
        status = "manifest_authority_validated_against_raw"
    elif not enhanced and authority_digest is not None:
        status = "manifest_authority_present_not_raw_validated_in_core_mode"
    elif not enhanced:
        status = "unavailable_core_analysis_without_raw_counter_proof"
    elif raw_vector_complete:
        status = "raw_recomputed_without_manifest_authority"
    else:
        status = "incomplete_or_ambiguous_raw_t0_vector"

    return {
        "status": status,
        "network_t0_vector_authority": (
            "selected_manifest" if authority_digest is not None else None
        ),
        "network_t0_status": inputs.network_t0_status,
        "network_t0_vector": authority_vector,
        "network_t0_vector_sha256": authority_digest,
        "manifest_authority_validated_against_raw": authority_validated,
        "raw_recomputed_network_t0_vector": raw_vector,
        "raw_recomputed_network_t0_vector_sha256": raw_digest,
        "analyzed_detector_networks": analyzed_networks,
        "missing_t0_networks": missing_t0_networks,
        "ambiguous_t0_networks": ambiguous_t0_networks,
        "extra_raw_networks_not_in_analyzed_detector_population": sorted(
            set(raw_by_network) - set(analyzed_networks)
        ),
        "missing_raw_t0_summary_networks": missing_raw_summary_networks,
        "extra_raw_t0_summary_networks": extra_raw_summary_networks,
        "ambiguous_raw_t0_summary_networks": ambiguous_raw_summary_networks,
        "clock_time_nanosecond_fields_preserved_separately": True,
    }


def _u32(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
            raise ContractError("counter values must be finite integers")
    integers = array.astype(np.int64, copy=False)
    return np.mod(integers, 2**32).astype(np.uint64)


def _modular_difference(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    modulus = np.uint64(2**32)
    return (after + modulus - before) % modulus


def raw_counter_diagnostics(
    timestamp_fields: np.ndarray,
    network: int,
    fpga_hz: float,
    accumulation_ticks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Inventory delivered PPS/internal-counter relationships without semantics claims."""

    fields = np.asarray(timestamp_fields, dtype=np.int64)
    if fields.ndim != 2 or fields.shape[1] != 6 or fields.shape[0] < 2:
        raise ContractError("counter diagnostics require nonempty Data.Toltec.Ts[row,6]")
    if not math.isfinite(fpga_hz) or fpga_hz <= 0 or accumulation_ticks <= 0:
        raise ContractError("counter diagnostics require positive FPGA rate and accumulation")
    t0 = fields[:, 0]
    pps_count = _u32(fields[:, 1])
    clock_count = _u32(fields[:, 2])
    packet_count = _u32(fields[:, 3])
    pps_time = _u32(fields[:, 4])
    t0_nanosec = fields[:, 5]
    pps_diff = _modular_difference(pps_count[1:], pps_count[:-1])
    transition_indices = np.flatnonzero(pps_diff != 0) + 1
    if transition_indices.size == 0:
        raise ContractError(f"toltec{network} has no delivered PPS-counter transitions")
    if np.any(pps_diff[transition_indices - 1] != 1):
        bad = transition_indices[pps_diff[transition_indices - 1] != 1]
        raise ContractError(
            f"toltec{network} PPS counter has non-unit transitions at rows {bad[:10].tolist()}"
        )
    clock_step = _modular_difference(clock_count[1:], clock_count[:-1])
    packet_step = _modular_difference(packet_count[1:], packet_count[:-1])
    pps_time_change = np.flatnonzero(pps_time[1:] != pps_time[:-1]) + 1
    pps_time_change_set = set(int(value) for value in pps_time_change)
    paired_pps_time_rows: list[int] = []
    transition_pairing_unambiguous = pps_time_change.size == transition_indices.size
    if transition_pairing_unambiguous:
        used_rows: set[int] = set()
        for count_row in transition_indices:
            nearby = pps_time_change[
                np.abs(pps_time_change.astype(np.int64) - int(count_row)) <= 1
            ]
            if nearby.size != 1 or int(nearby[0]) in used_rows:
                transition_pairing_unambiguous = False
                paired_pps_time_rows = []
                break
            paired_row = int(nearby[0])
            paired_pps_time_rows.append(paired_row)
            used_rows.add(paired_row)
        if transition_pairing_unambiguous:
            transition_pairing_unambiguous = (
                len(used_rows) == pps_time_change.size
                and bool(np.all(np.diff(paired_pps_time_rows) > 0))
            )
    paired_pps_time = np.asarray(paired_pps_time_rows, dtype=np.int64)
    paired_offsets = (
        paired_pps_time - transition_indices.astype(np.int64)
        if transition_pairing_unambiguous
        else np.asarray([], dtype=np.int64)
    )
    rows = []
    for ordinal, row in enumerate(transition_indices):
        row = int(row)
        count_row_geometry_ticks = int(
            _modular_difference(clock_count[row : row + 1], pps_time[row : row + 1])[0]
        )
        paired_pps_time_row = (
            int(paired_pps_time[ordinal]) if transition_pairing_unambiguous else None
        )
        native_frame_phase_ticks = (
            int(
                _modular_difference(
                    clock_count[paired_pps_time_row : paired_pps_time_row + 1],
                    pps_time[paired_pps_time_row : paired_pps_time_row + 1],
                )[0]
            )
            if paired_pps_time_row is not None
            else None
        )
        # Signed location of the preceding delivered row relative to the newly
        # retained PPS-time counter.  This is metadata geometry, not an event-
        # time or integration-centroid claim.
        before_mod = int(
            _modular_difference(
                clock_count[row - 1 : row], pps_time[row : row + 1]
            )[0]
        )
        signed_before_ticks = before_mod - 2**32 if before_mod > 2**31 else before_mod
        rows.append(
            {
                "network_id": network,
                "transition_ordinal": ordinal,
                "transition_row_zero_based": row,
                "rows_since_previous_transition": row - int(transition_indices[ordinal - 1])
                if ordinal > 0
                else "",
                "t0_integer_sec": int(t0[row]),
                "clock_time_nanosec_retained": int(t0_nanosec[row]),
                "pps_count_before": int(pps_count[row - 1]),
                "pps_count_after": int(pps_count[row]),
                "clock_count_before_u32": int(clock_count[row - 1]),
                "clock_count_after_u32": int(clock_count[row]),
                "pps_time_before_u32": int(pps_time[row - 1]),
                "pps_time_after_u32": int(pps_time[row]),
                "packet_count_before": int(packet_count[row - 1]),
                "packet_count_after": int(packet_count[row]),
                "pps_time_changed_on_pps_count_transition_row": row
                in pps_time_change_set,
                "pps_time_transition_pairing_status": (
                    "unique_ordered_same_or_adjacent_row_bijection"
                )
                if transition_pairing_unambiguous
                else "ambiguous_transition_geometry",
                "paired_pps_time_transition_row_zero_based": int(
                    paired_pps_time[ordinal]
                )
                if transition_pairing_unambiguous
                else "",
                "pps_time_minus_pps_count_transition_rows": int(
                    paired_offsets[ordinal]
                )
                if transition_pairing_unambiguous
                else "",
                "count_row_clock_minus_pps_time_ticks": count_row_geometry_ticks,
                "count_row_clock_minus_pps_time_sec": count_row_geometry_ticks
                / fpga_hz,
                "count_row_geometry_is_native_frame_phase": False,
                "native_frame_phase_row_zero_based": paired_pps_time_row
                if paired_pps_time_row is not None
                else "",
                "native_frame_phase_available": native_frame_phase_ticks is not None,
                "phase_after_ticks": native_frame_phase_ticks
                if native_frame_phase_ticks is not None
                else "",
                "phase_after_sec": native_frame_phase_ticks / fpga_hz
                if native_frame_phase_ticks is not None
                else "",
                "signed_before_relative_to_new_pps_ticks": signed_before_ticks,
                "signed_before_relative_to_new_pps_sec": signed_before_ticks / fpga_hz,
                "metadata_to_integration_association_proved": False,
            }
        )
    spacing = np.diff(transition_indices)
    repeat_delta = (
        transition_indices[128:] - transition_indices[:-128]
        if transition_indices.size > 128
        else np.asarray([], dtype=int)
    )
    transition_phase = (
        np.asarray([row["phase_after_sec"] for row in rows], dtype=float)
        if transition_pairing_unambiguous
        else np.asarray([], dtype=float)
    )
    phase_parts = (
        np.array_split(np.arange(transition_phase.size), 3)
        if transition_phase.size
        else []
    )
    pps_time_transition_step = (
        _modular_difference(
            pps_time[pps_time_change[1:]], pps_time[pps_time_change[:-1]]
        )
        if pps_time_change.size > 1
        else np.asarray([], dtype=np.uint64)
    )
    expected_pps_ticks = int(round(fpga_hz))
    anomaly_ordinals = np.flatnonzero(pps_time_transition_step != expected_pps_ticks)
    anomaly_rows: list[dict[str, Any]] = []
    phase_geometry = _modular_difference(clock_count, pps_time)
    count_transition_set = {int(value) for value in transition_indices}
    for anomaly_index, ordinal_value in enumerate(anomaly_ordinals):
        ordinal = int(ordinal_value)
        previous_row = int(pps_time_change[ordinal])
        row = int(pps_time_change[ordinal + 1])
        actual_increment = int(pps_time_transition_step[ordinal])
        residual_mod = (actual_increment - expected_pps_ticks) % 2**32
        signed_residual = residual_mod - 2**32 if residual_mod > 2**31 else residual_mod
        nearest_index = int(np.argmin(np.abs(transition_indices.astype(np.int64) - row)))
        nearest_count_row = int(transition_indices[nearest_index])
        following_residual = None
        if ordinal + 1 < pps_time_transition_step.size:
            following_actual = int(pps_time_transition_step[ordinal + 1])
            following_mod = (following_actual - expected_pps_ticks) % 2**32
            following_residual = following_mod - 2**32 if following_mod > 2**31 else following_mod
        adjacent_anomaly = (
            (anomaly_index > 0 and int(anomaly_ordinals[anomaly_index - 1]) == ordinal - 1)
            or (
                anomaly_index + 1 < anomaly_ordinals.size
                and int(anomaly_ordinals[anomaly_index + 1]) == ordinal + 1
            )
        )
        phase_step_mod = int(
            _modular_difference(phase_geometry[row:row + 1], phase_geometry[row - 1:row])[0]
        )
        signed_phase_step = phase_step_mod - 2**32 if phase_step_mod > 2**31 else phase_step_mod
        sample_pps_time_step = int(_modular_difference(pps_time[row:row + 1], pps_time[row - 1:row])[0])
        delivered_timestamp_step_residual = int(
            int(pps_diff[row - 1]) * expected_pps_ticks
            + int(clock_step[row - 1])
            - sample_pps_time_step
            - accumulation_ticks
        )
        anomaly_rows.append({
            "network_id": network,
            "anomaly_ordinal": anomaly_index,
            "pps_time_increment_ordinal": ordinal,
            "pps_time_previous_transition_row_zero_based": previous_row,
            "pps_time_transition_row_zero_based": row,
            "t0_integer_sec": int(t0[row]),
            "actual_pps_time_increment_ticks_u32": actual_increment,
            "expected_pps_time_increment_ticks": expected_pps_ticks,
            "signed_tick_residual": signed_residual,
            "absolute_tick_residual": abs(signed_residual),
            "signed_time_residual_sec": signed_residual / fpga_hz,
            "absolute_time_residual_sec": abs(signed_residual) / fpga_hz,
            "pps_count_transition_nearest_row_zero_based": nearest_count_row,
            "pps_time_minus_nearest_pps_count_rows": row - nearest_count_row,
            "on_pps_count_transition_row": row in count_transition_set,
            "clock_step_before_ticks": int(clock_step[row - 1]),
            "packet_step_before": int(packet_step[row - 1]),
            "pps_count_step_before": int(pps_diff[row - 1]),
            "sample_pps_time_step_ticks": sample_pps_time_step,
            "delivered_reconstructed_timestamp_step_residual_ticks": delivered_timestamp_step_residual,
            "delivered_reconstructed_timestamp_step_residual_sec": delivered_timestamp_step_residual / fpga_hz,
            "phase_geometry_before_ticks": int(phase_geometry[row - 1]),
            "phase_geometry_after_ticks": int(phase_geometry[row]),
            "phase_geometry_step_signed_ticks": signed_phase_step,
            "following_pps_time_increment_signed_tick_residual": following_residual,
            "cluster_class": "consecutive" if adjacent_anomaly else "isolated",
            "persistence_class": (
                "subsequent_increment_returns_nominal_counter_offset_persists_in_delivered_field"
                if following_residual == 0 else
                "subsequent_increment_also_anomalous" if following_residual is not None else
                "last_transition_no_following_increment"
            ),
            "delivered_data_timestamp_row_association": "D[n]/Ts[n] row lineage proved downstream; upstream FPGA association unresolved",
            "metadata_to_integration_association_proved": False,
        })
    unique_t0 = sorted(set(int(value) for value in t0))
    unique_nanosec = sorted(set(int(value) for value in t0_nanosec))
    summary = {
        "t0_integer_sec": unique_t0[0] if len(unique_t0) == 1 else "",
        "t0_integer_values_json": canonical_json(unique_t0),
        "t0_integer_value_count": len(unique_t0),
        "clock_time_nanosec_values_json": canonical_json(unique_nanosec),
        "clock_time_nanosec_value_count": len(unique_nanosec),
        "clock_time_nanosec_interpreted_as_phase": False,
        "pps_transition_count": int(transition_indices.size),
        "pps_spacing_122_count": int(np.sum(spacing == 122)),
        "pps_spacing_123_count": int(np.sum(spacing == 123)),
        "pps_spacing_other_count": int(np.sum(~np.isin(spacing, (122, 123)))),
        "pps_spacing_all_122_or_123": bool(np.all(np.isin(spacing, (122, 123)))),
        "repeat_128_interval_test_count": int(repeat_delta.size),
        "repeat_128_interval_15625_rows_count": int(np.sum(repeat_delta == 15625)),
        "repeat_128_interval_mismatch_count": int(np.sum(repeat_delta != 15625)),
        "clock_increment_expected_ticks": accumulation_ticks,
        "clock_increment_mismatch_count": int(np.sum(clock_step != accumulation_ticks)),
        "clock_increment_min_ticks": int(np.min(clock_step)),
        "clock_increment_max_ticks": int(np.max(clock_step)),
        "packet_increment_mismatch_count": int(np.sum(packet_step != 1)),
        "pps_time_transition_same_row_count": int(
            sum(bool(row["pps_time_changed_on_pps_count_transition_row"]) for row in rows)
        ),
        "pps_time_transition_different_row_count": int(
            sum(not bool(row["pps_time_changed_on_pps_count_transition_row"]) for row in rows)
        ),
        "pps_time_increment_expected_ticks": expected_pps_ticks,
        "pps_time_increment_eligible_count": int(pps_time_transition_step.size),
        "pps_time_increment_mismatch_count": int(
            np.sum(pps_time_transition_step != expected_pps_ticks)
        ),
        "pps_time_increment_mismatch_rate": (
            float(anomaly_ordinals.size / pps_time_transition_step.size)
            if pps_time_transition_step.size else None
        ),
        "pps_time_increment_anomaly_first_transition_row_zero_based": (
            int(anomaly_rows[0]["pps_time_transition_row_zero_based"])
            if anomaly_rows else None
        ),
        "pps_time_increment_anomaly_last_transition_row_zero_based": (
            int(anomaly_rows[-1]["pps_time_transition_row_zero_based"])
            if anomaly_rows else None
        ),
        "pps_time_increment_anomaly_isolated_count": int(
            sum(row["cluster_class"] == "isolated" for row in anomaly_rows)
        ),
        "pps_time_increment_anomaly_consecutive_count": int(
            sum(row["cluster_class"] == "consecutive" for row in anomaly_rows)
        ),
        "pps_time_increment_anomaly_periodicity": (
            "reported_from_transition_ordinals_only; no periodicity cut applied"
        ),
        "pps_time_transition_pairing_status": "unique_ordered_same_or_adjacent_row_bijection"
        if transition_pairing_unambiguous
        else "ambiguous_transition_geometry",
        "pps_time_transition_pairing_rule": (
            "Every PpsCount transition must have exactly one unused PpsTime transition "
            "on the same or immediately adjacent delivered row, all PpsTime transitions "
            "must be consumed, and paired rows must increase strictly."
        ),
        "pps_time_transition_pairing_physical_event_simultaneity_claimed": False,
        "pps_time_minus_pps_count_transition_offset_min_rows": int(
            np.min(paired_offsets)
        )
        if paired_offsets.size
        else "",
        "pps_time_minus_pps_count_transition_offset_max_rows": int(
            np.max(paired_offsets)
        )
        if paired_offsets.size
        else "",
        "pps_time_minus_pps_count_transition_offsets_json": canonical_json(
            sorted(set(int(value) for value in paired_offsets))
        )
        if paired_offsets.size
        else "[]",
        "pps_time_transition_offset_minus_one_count": int(np.sum(paired_offsets == -1)),
        "pps_time_transition_offset_zero_count": int(np.sum(paired_offsets == 0)),
        "pps_time_transition_offset_plus_one_count": int(np.sum(paired_offsets == 1)),
        "pps_time_transition_offset_other_count": int(
            np.sum(~np.isin(paired_offsets, (-1, 0, 1)))
        ),
        "variable_metadata_capture_or_isr_latency_observed": bool(
            paired_offsets.size and np.unique(paired_offsets).size > 1
        ),
        "variable_latency_inference_authorized": transition_pairing_unambiguous,
        "native_frame_phase_status": "available_at_ordered_paired_pps_time_transition_rows"
        if transition_pairing_unambiguous
        else "unavailable_ambiguous_pps_transition_pairing",
        "native_frame_phase_begin_sec": float(
            np.mean(transition_phase[phase_parts[0]])
        )
        if transition_phase.size
        else None,
        "native_frame_phase_middle_sec": float(
            np.mean(transition_phase[phase_parts[1]])
        )
        if transition_phase.size
        else None,
        "native_frame_phase_end_sec": float(
            np.mean(transition_phase[phase_parts[2]])
        )
        if transition_phase.size
        else None,
        "native_frame_phase_mean_sec": float(np.mean(transition_phase))
        if transition_phase.size
        else None,
        "native_frame_phase_std_sec": float(np.std(transition_phase))
        if transition_phase.size
        else None,
        "native_frame_phase_min_sec": float(np.min(transition_phase))
        if transition_phase.size
        else None,
        "native_frame_phase_max_sec": float(np.max(transition_phase))
        if transition_phase.size
        else None,
        "native_frame_phase_definition": (
            "(ClockCount[row]-PpsTime[row]) mod 2^32 / Header.Toltec.FpgaFreq "
            "on the ordered paired delivered PpsTime transition row; unavailable "
            "when PpsCount/PpsTime transition pairing is ambiguous"
        ),
        "shared_reference_architecture": "Octo 10MHz and PPS; PPS does not reset sample cadence",
        "arbitrary_millisecond_ntp_error": "strongly_disfavored",
        "differential_oscillator_drift": "strongly_disfavored",
        "distinct_stable_network_integration_phase": "allowed",
        "metadata_to_integration_association": "unresolved_without_fpga_source",
    }
    return rows, summary, anomaly_rows


def model_id(basis: str, k: int, phi: float) -> str:
    return f"{basis}_k{k:+d}_phi{phi:+.1f}"


def model_coordinates(
    mapping: RawMapping,
    sample_count: int,
    basis: str,
    k: int,
    phi: float,
    telescope: TelescopeEvaluator,
) -> dict[str, np.ndarray]:
    source = mapping.row_for_slot
    shifted = source + k
    valid = (source >= 0) & (shifted >= 0) & (shifted < mapping.times.size)
    safe = np.clip(shifted, 0, mapping.times.size - 1)
    shifted_slot = np.where(valid, mapping.slots[safe], np.iinfo(np.int64).min)
    if basis == "assigned_slot":
        target = np.where(valid, mapping.assigned[safe] + phi * mapping.cadence_sec, np.nan)
    elif basis == "raw_detector_timestamp":
        target = np.where(valid, mapping.times[safe] + phi * mapping.cadence_sec, np.nan)
    else:
        raise ContractError(f"unknown timing basis {basis}")
    evaluated = telescope.evaluate(target)
    evaluated.update(
        {
            "source_row": source,
            "shifted_row": shifted,
            "shifted_slot": shifted_slot,
            "row_valid": valid,
            "target_time": target,
            "vx": np.gradient(evaluated["x"], mapping.cadence_sec),
            "vy": np.gradient(evaluated["y"], mapping.cadence_sec),
        }
    )
    return evaluated


def fit_detector_model(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    reference: Mapping[str, float],
    array_id: int,
    protocol: AnalysisProtocol,
) -> dict[str, Any]:
    major0 = max(float(reference["major"]), float(reference["minor"]))
    minor0 = min(float(reference["major"]), float(reference["minor"]))
    dx0 = x - float(reference["x"])
    dy0 = y - float(reference["y"])
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    keep = finite & (np.hypot(dx0, dy0) <= protocol.radial_fwhm_multiplier * major0)
    x, y, z = x[keep], y[keep], z[keep]
    if z.size < protocol.minimum_detector_samples:
        return {
            "success": False,
            "quality": False,
            "reason": "insufficient_samples",
            "n_samples": int(z.size),
        }
    if array_id not in ARRAY_FWHM_LIMITS:
        raise ContractError(f"unknown detector array id {array_id}")
    fmin, fmax = ARRAY_FWHM_LIMITS[array_id]
    median = float(np.median(z))
    amplitude = max(float(np.max(z) - median), np.finfo(float).eps)
    s1 = np.clip(major0, fmin * 1.001, fmax * 0.999) / 2.354820045
    s2 = np.clip(minor0, fmin * 1.001, fmax * 0.999) / 2.354820045
    p0 = np.array(
        [
            amplitude,
            reference["x"],
            reference["y"],
            math.log(s1),
            math.log(s2),
            reference["angle"],
            median,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    lower = np.array(
        [
            0.0,
            reference["x"] - major0,
            reference["y"] - major0,
            math.log(fmin / 2.354820045),
            math.log(fmin / 2.354820045),
            -math.pi,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    upper = np.array(
        [
            np.inf,
            reference["x"] + major0,
            reference["y"] + major0,
            math.log(fmax / 2.354820045),
            math.log(fmax / 2.354820045),
            math.pi,
            np.inf,
            np.inf,
            np.inf,
        ]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amp, cx, cy, log_s1, log_s2, angle, background, bx, by = parameters
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = x - cx, y - cy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        source = amp * np.exp(
            -0.5 * ((u / math.exp(log_s1)) ** 2 + (v / math.exp(log_s2)) ** 2)
        )
        return source + background + bx * dx0[keep] + by * dy0[keep] - z

    scale = max(0.2 * float(np.std(z)), 1.0e-12)
    try:
        result = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=scale,
            max_nfev=300,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
    except Exception as error:  # scipy reports several numeric exception types
        return {
            "success": False,
            "quality": False,
            "reason": f"solver_exception:{type(error).__name__}",
            "n_samples": int(z.size),
        }
    values = result.x
    residual_values = residual(values)
    residual_median = float(np.median(residual_values))
    residual_mad = float(
        1.4826 * np.median(np.abs(residual_values - residual_median))
    )
    covariance = np.full((values.size, values.size), np.nan)
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * float(
            np.sum(residual_values**2) / max(1, z.size - values.size)
        )
    except np.linalg.LinAlgError:
        pass
    fwhm = np.exp(values[3:5]) * 2.354820045
    order = np.argsort(fwhm)[::-1]
    major, minor = fwhm[order]
    position_angle = float(values[5] + (math.pi / 2.0 if order[0] == 1 else 0.0))
    position_angle = ((position_angle + math.pi / 2.0) % math.pi) - math.pi / 2.0
    center_margin = major0 - max(
        abs(values[1] - reference["x"]), abs(values[2] - reference["y"])
    )
    width_margin = min(major - fmin, minor - fmin, fmax - major, fmax - minor)
    amplitude_snr = float(values[0] / residual_mad) if residual_mad > 0 else math.inf
    center_covariance = covariance[1:3, 1:3]
    quality = bool(
        result.success
        and np.all(np.isfinite(center_covariance))
        and amplitude_snr >= protocol.minimum_amplitude_snr
        and center_margin > protocol.fit_boundary_margin_minimum
        and width_margin > protocol.fit_boundary_margin_minimum
    )
    return {
        "success": bool(result.success),
        "quality": quality,
        "reason": "accepted" if quality else "frozen_quality_rule_failed",
        "n_samples": int(z.size),
        "amplitude": _finite_or_none(values[0]),
        "background": _finite_or_none(values[6]),
        "background_x": _finite_or_none(values[7]),
        "background_y": _finite_or_none(values[8]),
        "centroid_x_arcsec": _finite_or_none(values[1]),
        "centroid_y_arcsec": _finite_or_none(values[2]),
        "centroid_x_sigma_arcsec": _finite_or_none(
            math.sqrt(max(0.0, center_covariance[0, 0]))
            if np.isfinite(center_covariance[0, 0]) else math.nan
        ),
        "centroid_y_sigma_arcsec": _finite_or_none(
            math.sqrt(max(0.0, center_covariance[1, 1]))
            if np.isfinite(center_covariance[1, 1]) else math.nan
        ),
        "centroid_xy_cov_arcsec2": _finite_or_none(center_covariance[0, 1]),
        "major_fwhm_arcsec": _finite_or_none(major),
        "minor_fwhm_arcsec": _finite_or_none(minor),
        "position_angle_deg": _finite_or_none(math.degrees(position_angle)),
        "ellipticity": _finite_or_none(major / minor - 1.0),
        "residual_mad_sigma": _finite_or_none(residual_mad),
        "amplitude_over_residual_mad": _finite_or_none(amplitude_snr),
        "cost": _finite_or_none(result.cost),
        "optimality": _finite_or_none(result.optimality),
        "nfev": int(result.nfev),
    }


def empty_map(protocol: AnalysisProtocol) -> np.ndarray:
    size = int(round(2 * protocol.map_extent_arcsec / protocol.map_pixel_arcsec))
    return np.zeros((size, size), dtype=np.float64)


def empty_count(protocol: AnalysisProtocol) -> np.ndarray:
    return np.zeros(empty_map(protocol).shape, dtype=np.int64)


def add_samples(
    sums: np.ndarray,
    counts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    protocol: AnalysisProtocol,
) -> None:
    extent = float(protocol.map_extent_arcsec)
    pixel = float(protocol.map_pixel_arcsec)
    px = np.floor((x + extent) / pixel).astype(int)
    py = np.floor((y + extent) / pixel).astype(int)
    inside = (
        (px >= 0)
        & (px < sums.shape[1])
        & (py >= 0)
        & (py < sums.shape[0])
        & np.isfinite(z)
    )
    np.add.at(sums, (py[inside], px[inside]), z[inside])
    np.add.at(counts, (py[inside], px[inside]), 1)


def map_fit(
    sums: np.ndarray,
    counts: np.ndarray,
    array_id: int,
    protocol: AnalysisProtocol,
) -> dict[str, Any]:
    centers = (
        np.arange(sums.shape[0], dtype=float) * protocol.map_pixel_arcsec
        - protocol.map_extent_arcsec
        + 0.5 * protocol.map_pixel_arcsec
    )
    xx, yy = np.meshgrid(centers, centers)
    valid = counts >= protocol.minimum_map_pixel_count
    values_image = np.divide(
        sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
    )
    if int(np.sum(valid)) < protocol.minimum_fitted_map_pixels:
        return {
            "success": False,
            "quality": False,
            "reason": "insufficient_map_pixels",
            "n_pixels": int(np.sum(valid)),
            "n_samples": int(np.sum(counts)),
        }
    weights = np.sqrt(counts[valid].astype(float) / np.max(counts[valid]))
    x, y, values = xx[valid], yy[valid], values_image[valid]
    fmin, fmax = (3.0, 20.0) if array_id == -1 else ARRAY_FWHM_LIMITS[array_id]
    major0 = 0.55 * (fmin + fmax)
    minor0 = 0.50 * (fmin + fmax)
    p0 = np.array(
        [
            max(float(np.nanmax(values) - np.nanmedian(values)), 1.0e-9),
            0.0,
            0.0,
            math.log(major0 / 2.354820045),
            math.log(minor0 / 2.354820045),
            0.0,
            float(np.nanmedian(values)),
            0.0,
            0.0,
        ]
    )
    lower = np.array(
        [
            0.0,
            -major0,
            -major0,
            math.log(fmin / 2.354820045),
            math.log(fmin / 2.354820045),
            -math.pi,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    upper = np.array(
        [
            np.inf,
            major0,
            major0,
            math.log(fmax / 2.354820045),
            math.log(fmax / 2.354820045),
            math.pi,
            np.inf,
            np.inf,
            np.inf,
        ]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amp, cx, cy, log_s1, log_s2, angle, background, bx, by = parameters
        ca, sa = np.cos(angle), np.sin(angle)
        dx, dy = x - cx, y - cy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        model = (
            amp
            * np.exp(
                -0.5
                * ((u / np.exp(log_s1)) ** 2 + (v / np.exp(log_s2)) ** 2)
            )
            + background
            + bx * x
            + by * y
        )
        return (model - values) * weights

    result = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(0.2 * float(np.nanstd(values)), 1.0e-6),
        max_nfev=300,
    )
    fitted = result.x
    fwhm = np.exp(fitted[3:5]) * 2.354820045
    order = np.argsort(fwhm)[::-1]
    major, minor = fwhm[order]
    angle = fitted[5] + (math.pi / 2 if order[0] == 1 else 0)
    angle = ((angle + math.pi / 2) % math.pi) - math.pi / 2
    residual_values = residual(fitted)
    residual_mad = 1.4826 * np.median(
        np.abs(residual_values - np.median(residual_values))
    )
    covariance = np.full((9, 9), np.nan)
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac) * (
            np.sum(residual_values**2) / max(1, residual_values.size - 9)
        )
    except np.linalg.LinAlgError:
        pass
    quality = bool(result.success and np.all(np.isfinite(covariance[1:3, 1:3])))
    return {
        "success": bool(result.success),
        "quality": quality,
        "reason": "accepted" if quality else "fit_or_covariance_failed",
        "n_pixels": int(np.sum(valid)),
        "n_samples": int(np.sum(counts)),
        "amplitude": _finite_or_none(fitted[0]),
        "centroid_x_arcsec": _finite_or_none(fitted[1]),
        "centroid_y_arcsec": _finite_or_none(fitted[2]),
        "centroid_x_sigma_arcsec": _finite_or_none(
            math.sqrt(max(0.0, covariance[1, 1]))
            if np.isfinite(covariance[1, 1]) else math.nan
        ),
        "centroid_y_sigma_arcsec": _finite_or_none(
            math.sqrt(max(0.0, covariance[2, 2]))
            if np.isfinite(covariance[2, 2]) else math.nan
        ),
        "centroid_xy_cov_arcsec2": _finite_or_none(covariance[1, 2]),
        "major_fwhm_arcsec": _finite_or_none(major),
        "minor_fwhm_arcsec": _finite_or_none(minor),
        "position_angle_deg": _finite_or_none(np.degrees(angle)),
        "ellipticity": _finite_or_none(major / minor - 1.0),
        "residual_mad_sigma": _finite_or_none(residual_mad),
        "cost": _finite_or_none(result.cost),
    }


def fit_timing(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    axis: np.ndarray,
    v_left: float | None,
    v_right: float | None,
) -> dict[str, Any]:
    if not left.get("quality") or not right.get("quality"):
        return {"quality": False, "reason": "left_or_right_fit_failed"}
    if v_left is None or v_right is None:
        return {"quality": False, "reason": "direction_speed_unavailable"}
    denominator = v_right - v_left
    if not math.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
        return {"quality": False, "reason": "direction_speed_denominator_invalid"}
    cross_axis = np.array([-axis[1], axis[0]])
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
        "timing_residual_sec": parallel / denominator,
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


def jackknife_se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ContractError("at least two finite jackknife replicates are required")
    return float(
        np.sqrt((array.size - 1) / array.size * np.sum((array - np.mean(array)) ** 2))
    )


def formal_timing_se(
    left: Mapping[str, Any], right: Mapping[str, Any], axis: np.ndarray, speed_span: float
) -> tuple[float, float]:
    left_cov = np.array(
        [
            [left["centroid_x_sigma_arcsec"] ** 2, left["centroid_xy_cov_arcsec2"]],
            [left["centroid_xy_cov_arcsec2"], left["centroid_y_sigma_arcsec"] ** 2],
        ]
    )
    right_cov = np.array(
        [
            [right["centroid_x_sigma_arcsec"] ** 2, right["centroid_xy_cov_arcsec2"]],
            [right["centroid_xy_cov_arcsec2"], right["centroid_y_sigma_arcsec"] ** 2],
        ]
    )
    cross = np.array([-axis[1], axis[0]])
    parallel = float(np.sqrt(max(0.0, axis @ (left_cov + right_cov) @ axis)))
    perpendicular = float(np.sqrt(max(0.0, cross @ (left_cov + right_cov) @ cross)))
    return parallel / abs(speed_span), perpendicular


def explicit_missing_network_rows(
    expected_networks: Sequence[int],
    present_rows: Mapping[int, Mapping[str, Any]],
    map_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for network in sorted(set(int(item) for item in expected_networks) | set(present_rows)):
        if network in present_rows:
            row = dict(present_rows[network])
            row.setdefault("available", True)
            row.setdefault("status", "available")
        else:
            row = {
                "map_id": map_id,
                "network_id": network,
                "available": False,
                "status": "missing_network",
                "timing_residual_sec": "",
                "timing_se_sec": "",
            }
        rows.append(row)
    return rows


def science_support_mask(state: AlignmentState) -> np.ndarray:
    mask = np.zeros(state.sample_count, dtype=bool)
    for record in state.records:
        science = record["compatibility_science"]
        start = int(science["start"]) - state.union_local_start
        stop = int(science["stop"]) - state.union_local_start
        mask[start:stop] = True
    return mask


def source_write_guard(inputs: ReductionInputs, output_root: Path) -> None:
    """Reject output placement in an individual retained product or raw directory.

    An owner versioned run directory may live below the broad Beammap project
    directory; it must still be outside every individual reduction and raw
    source retained by the selected manifest.
    """

    output = output_root.expanduser().resolve()
    forbidden = {inputs.reduction_path.resolve()}
    for path in (
        inputs.config_path,
        inputs.telescope_path,
        inputs.provenance_path,
        inputs.detector_tod_path,
        inputs.output_apt_path,
    ):
        if path is not None:
            forbidden.add(path.parent.resolve())
    forbidden.update(path.parent.resolve() for path in inputs.raw_by_network.values())
    for source in sorted(forbidden):
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise ContractError(f"output root {output} is inside source directory {source}")


def safe_candidate_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not normalized or normalized in {".", ".."}:
        raise ContractError(f"candidate_id is not a safe output component: {value!r}")
    return normalized


def table_fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


@dataclass
class DetectorContract:
    uid: np.ndarray
    arrays: np.ndarray
    networks: np.ndarray
    fit_good: np.ndarray
    full_x: np.ndarray
    full_y: np.ndarray
    kind: np.ndarray
    scan_index: np.ndarray
    n_samples: np.ndarray
    starts: np.ndarray
    apt_flag: np.ndarray
    major_ref: np.ndarray
    minor_ref: np.ndarray
    amplitude_ref: np.ndarray
    angle_ref: np.ndarray


def load_detector_contract(
    dataset: Dataset,
    apt: Table,
    protocol: AnalysisProtocol,
    observation_number: int,
) -> tuple[DetectorContract, np.ndarray]:
    required_variables = [
        "detector_tod_uid",
        "detector_tod_array",
        "detector_tod_network",
        "detector_tod_fit_good",
        "detector_tod_fit_x_t_arcsec",
        "detector_tod_fit_y_t_arcsec",
        "detector_tod_slot_kind",
        "detector_tod_scan_index",
        "detector_tod_n_samples",
        "detector_tod_scan_inner_start_sample",
        "signal",
        "flags",
    ]
    missing_variables = [name for name in required_variables if name not in dataset.variables]
    if missing_variables:
        raise ContractError(f"detector TOD lacks variables: {missing_variables}")
    required_columns = {"uid", "flag", "amp", "a_fwhm", "b_fwhm", "angle"}
    missing_columns = sorted(required_columns - set(apt.colnames))
    if missing_columns:
        raise ContractError(f"output APT lacks columns: {missing_columns}")
    uid = np.asarray(dataset["detector_tod_uid"][:], dtype=int)
    if uid.ndim != 1 or uid.size == 0 or len(set(uid.tolist())) != uid.size:
        raise ContractError("detector TOD UID axis is empty or non-unique")
    apt_uid = np.asarray(apt["uid"], dtype=int)
    if len(set(apt_uid.tolist())) != apt_uid.size:
        raise ContractError("output APT UID axis is non-unique")
    apt_lookup = {int(value): index for index, value in enumerate(apt_uid)}
    if any(int(value) not in apt_lookup for value in uid):
        raise ContractError("output APT lacks one or more detector TOD UIDs")
    apt_order = np.asarray([apt_lookup[int(value)] for value in uid], dtype=int)
    arrays = np.asarray(dataset["detector_tod_array"][:], dtype=int)
    networks = np.asarray(dataset["detector_tod_network"][:], dtype=int)
    fit_good = np.asarray(dataset["detector_tod_fit_good"][:], dtype=int)
    full_x = np.asarray(dataset["detector_tod_fit_x_t_arcsec"][:], dtype=float)
    full_y = np.asarray(dataset["detector_tod_fit_y_t_arcsec"][:], dtype=float)
    kind = np.asarray(dataset["detector_tod_slot_kind"][:], dtype=int)
    scan_index = np.asarray(dataset["detector_tod_scan_index"][:], dtype=int)
    n_samples = np.asarray(dataset["detector_tod_n_samples"][:], dtype=int)
    starts = np.asarray(
        dataset["detector_tod_scan_inner_start_sample"][:], dtype=int
    )
    vectors = (arrays, networks, fit_good, full_x, full_y)
    matrices = (kind, scan_index, n_samples, starts)
    if any(value.shape != uid.shape for value in vectors):
        raise ContractError("detector TOD detector-vector shapes differ")
    if any(value.ndim != 2 or value.shape[0] != uid.size for value in matrices):
        raise ContractError("detector TOD slot-matrix shapes differ")
    if any(value.shape != kind.shape for value in matrices):
        raise ContractError("detector TOD slot matrices have inconsistent shapes")
    if any(int(value) not in ARRAY_NAMES for value in np.unique(arrays)):
        raise ContractError(f"detector TOD contains unknown array IDs: {np.unique(arrays)}")
    apt_flag = np.asarray(apt["flag"], dtype=int)[apt_order]
    major_ref = np.maximum(
        np.asarray(apt["a_fwhm"], dtype=float)[apt_order],
        np.asarray(apt["b_fwhm"], dtype=float)[apt_order],
    )
    minor_ref = np.minimum(
        np.asarray(apt["a_fwhm"], dtype=float)[apt_order],
        np.asarray(apt["b_fwhm"], dtype=float)[apt_order],
    )
    amplitude_ref = np.asarray(apt["amp"], dtype=float)[apt_order]
    angle_ref = np.radians(np.asarray(apt["angle"], dtype=float)[apt_order])
    preselected = (
        (fit_good == 1)
        & (apt_flag == 0)
        & np.isfinite(full_x)
        & np.isfinite(full_y)
        & np.isfinite(major_ref)
        & np.isfinite(minor_ref)
        & (major_ref > 0)
        & (minor_ref > 0)
        & np.isfinite(amplitude_ref)
        & (amplitude_ref > 0)
        & ~np.isin(
            uid,
            np.asarray(
                protocol.excluded_uids_for_observation(observation_number), dtype=int
            ),
        )
    )
    return (
        DetectorContract(
            uid,
            arrays,
            networks,
            fit_good,
            full_x,
            full_y,
            kind,
            scan_index,
            n_samples,
            starts,
            apt_flag,
            major_ref,
            minor_ref,
            amplitude_ref,
            angle_ref,
        ),
        preselected,
    )


def _slot_indices(
    detector: DetectorContract,
    det: int,
    slot: int,
    signal_width: int,
    sample_count: int,
) -> tuple[np.ndarray, int]:
    start = int(detector.starts[det, slot])
    length = int(detector.n_samples[det, slot])
    if length < 0 or length > signal_width:
        raise ContractError(
            f"UID {detector.uid[det]} slot {slot} has invalid n_samples={length}"
        )
    stop = start + length
    if start < 0 or stop > sample_count:
        raise ContractError(
            f"UID {detector.uid[det]} slot {slot} lies outside common sample axis"
        )
    return np.arange(start, stop, dtype=np.int64), length


def _baseline_sample_mask(
    detector: DetectorContract,
    det: int,
    indices: np.ndarray,
    signal: np.ndarray,
    flags: np.ndarray,
    telescope: Mapping[str, np.ndarray],
    protocol: AnalysisProtocol,
) -> np.ndarray:
    radial = np.hypot(
        telescope["x"][indices] - float(detector.full_x[det]),
        telescope["y"][indices] - float(detector.full_y[det]),
    )
    return (
        (flags == 0)
        & np.isfinite(signal)
        & (radial <= protocol.radial_fwhm_multiplier * float(detector.major_ref[det]))
        & (telescope["hold_left"][indices] == 0)
        & (telescope["hold_right"][indices] == 0)
        & ~telescope["hold_transition"][indices]
    )


def select_matched_detectors(
    detector: DetectorContract,
    preselected: np.ndarray,
    signal_variable: Any,
    flag_variable: Any,
    signal_width: int,
    state: AlignmentState,
    registry: Mapping[int, Mapping[str, Any]],
    telescope: Mapping[str, np.ndarray],
    axis: np.ndarray,
    protocol: AnalysisProtocol,
    map_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[int, int]]:
    """Apply the frozen signal-independent preselection and two-direction fit cut."""

    controls: list[dict[str, Any]] = []
    matched: list[int] = []
    matched_by_network: dict[int, int] = defaultdict(int)
    for det in np.flatnonzero(preselected):
        payload = {
            "left": {"x": [], "y": [], "z": [], "scan": []},
            "right": {"x": [], "y": [], "z": [], "scan": []},
        }
        for slot in np.flatnonzero(detector.kind[det] == 2):
            ordinal = int(detector.scan_index[det, slot])
            if ordinal not in state.ordinal_to_stable:
                raise ContractError(
                    f"UID {detector.uid[det]} references unknown output scan index {ordinal}"
                )
            stable = state.ordinal_to_stable[ordinal]
            row = registry[stable]
            direction = str(row["classification"])
            if direction not in {"left", "right"} or not bool(row["selected"]):
                continue
            indices, length = _slot_indices(
                detector, det, int(slot), signal_width, state.sample_count
            )
            if length == 0:
                continue
            signal = np.asarray(signal_variable[det, slot, :length], dtype=float)
            flags = np.asarray(flag_variable[det, slot, :length], dtype=int)
            valid = _baseline_sample_mask(
                detector, det, indices, signal, flags, telescope, protocol
            )
            if not np.any(valid):
                continue
            payload[direction]["x"].append(telescope["x"][indices][valid])
            payload[direction]["y"].append(telescope["y"][indices][valid])
            payload[direction]["z"].append(signal[valid])
            payload[direction]["scan"].append(
                np.full(int(np.sum(valid)), stable, dtype=int)
            )
        accepted = {}
        for direction in ("left", "right"):
            part = payload[direction]
            distinct = (
                sorted(set(np.concatenate(part["scan"]).tolist()))
                if part["scan"]
                else []
            )
            if len(distinct) < protocol.minimum_distinct_scans_per_direction:
                fit = {
                    "success": False,
                    "quality": False,
                    "reason": "fewer_than_minimum_distinct_scans",
                    "n_samples": int(sum(map(len, part["z"]))),
                }
            else:
                reference = {
                    "x": float(detector.full_x[det]),
                    "y": float(detector.full_y[det]),
                    "major": float(detector.major_ref[det]),
                    "minor": float(detector.minor_ref[det]),
                    "angle": float(detector.angle_ref[det]),
                }
                fit = fit_detector_model(
                    np.concatenate(part["x"]),
                    np.concatenate(part["y"]),
                    np.concatenate(part["z"]),
                    reference,
                    int(detector.arrays[det]),
                    protocol,
                )
            controls.append(
                {
                    "map_id": map_id,
                    "model_id": "assigned_slot_k+0_phi+0.0",
                    "support": "independent_cohort_selection",
                    "group": f"detector:{int(detector.uid[det])}",
                    "uid": int(detector.uid[det]),
                    "network_id": int(detector.networks[det]),
                    "array": ARRAY_NAMES[int(detector.arrays[det])],
                    "direction": direction,
                    "n_scans": len(distinct),
                    "stable_scan_ids_json": canonical_json(distinct),
                    **fit,
                }
            )
            accepted[direction] = bool(fit.get("quality"))
        if accepted.get("left") and accepted.get("right"):
            matched.append(int(det))
            matched_by_network[int(detector.networks[det])] += 1
    if len(matched) < protocol.minimum_matched_detectors:
        raise ContractError(
            f"matched detector cohort has {len(matched)} detectors, below "
            f"minimum {protocol.minimum_matched_detectors}"
        )
    return np.asarray(matched, dtype=int), controls, dict(matched_by_network)


def _group_array_id(
    group: str, detector: DetectorContract, matched: np.ndarray
) -> int:
    if group == "all":
        return -1
    if group.startswith("array:"):
        return next(key for key, value in ARRAY_NAMES.items() if group == f"array:{value}")
    if group.startswith("network:toltec"):
        network = int(group.split("toltec", 1)[1])
        indices = matched[detector.networks[matched] == network]
        if indices.size == 0:
            raise ContractError(f"cannot identify array for absent network {network}")
        arrays = np.unique(detector.arrays[indices])
        if arrays.size != 1:
            raise ContractError(f"network {network} spans multiple arrays: {arrays}")
        return int(arrays[0])
    raise ContractError(f"unknown analysis group {group}")


def _timing_row(
    descriptor: Mapping[str, Any],
    group: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    speeds: Mapping[str, float | None],
    axis: np.ndarray,
) -> dict[str, Any]:
    return {
        "model_id": descriptor["model_id"],
        "time_basis": descriptor["time_basis"],
        "row_shift_k": descriptor["row_shift_k"],
        "phase_phi_samples": descriptor["phase_phi_samples"],
        "support": "all_model_common_interior",
        "group": group,
        **fit_timing(left, right, axis, speeds["left"], speeds["right"]),
    }


def _map_total(
    maps: Mapping[tuple[str, int], np.ndarray],
    direction: str,
    stable_ids: Sequence[int],
    protocol: AnalysisProtocol,
    counts: bool = False,
) -> np.ndarray:
    result = empty_count(protocol) if counts else empty_map(protocol)
    for stable in stable_ids:
        result += maps[(direction, int(stable))]
    return result


@dataclass
class AnalysisProducts:
    map_summary: dict[str, Any]
    map_result: dict[str, Any]
    network_rows: list[dict[str, Any]]
    timing_rows: list[dict[str, Any]]
    fit_control_rows: list[dict[str, Any]]
    fit_controls: dict[str, Any]
    scan_registry: list[dict[str, Any]]
    raw_linkage_rows: list[dict[str, Any]]
    raw_counter_rows: list[dict[str, Any]]
    raw_phase_rows: list[dict[str, Any]]
    raw_pps_time_increment_anomaly_rows: list[dict[str, Any]] = field(default_factory=list)


def linear_predictor_diagnostic(
    rows: Sequence[Mapping[str, Any]], predictor: str
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("available")
        and row.get("timing_residual_sec") not in (None, "")
        and row.get(predictor) not in (None, "")
    ]
    if len(usable) < 3:
        return {
            "available": False,
            "reason": "fewer_than_three_networks",
            "predictor": predictor,
            "network_count": len(usable),
        }
    x = np.asarray([float(row[predictor]) for row in usable])
    y = np.asarray([float(row["timing_residual_sec"]) for row in usable])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {
            "available": False,
            "reason": "non_finite_predictor_or_response",
            "predictor": predictor,
            "network_count": len(usable),
        }
    x_scale = max(1.0, float(np.max(np.abs(x))))
    if float(np.ptp(x)) <= np.finfo(float).eps * x_scale:
        return {
            "available": False,
            "reason": "predictor_has_no_network_leverage",
            "predictor": predictor,
            "network_count": len(usable),
        }
    y_scale = max(1.0, float(np.max(np.abs(y))))
    if float(np.ptp(y)) <= np.finfo(float).eps * y_scale:
        return {
            "available": False,
            "reason": "response_has_no_network_leverage",
            "predictor": predictor,
            "network_count": len(usable),
        }
    slope, intercept = np.polyfit(x, y, 1)
    pearson = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(pearson):
        return {
            "available": False,
            "reason": "correlation_undefined",
            "predictor": predictor,
            "network_count": len(usable),
        }
    return {
        "available": True,
        "predictor": predictor,
        "network_count": len(usable),
        "slope": float(slope),
        "intercept_sec": float(intercept),
        "pearson": pearson,
        "preregistered_slot_relation_slope": -1.0
        if predictor == "native_to_assigned_slot_residual_sec"
        else None,
        "slope_minus_preregistered_minus_one": float(slope + 1.0)
        if predictor == "native_to_assigned_slot_residual_sec"
        else None,
        "physical_correction_authorized": False,
    }


def analyze_reduction(
    inputs: ReductionInputs,
    protocol: AnalysisProtocol,
    mode: str,
    log: list[str] | None = None,
) -> AnalysisProducts:
    """Analyze one retained Beammap without launching or modifying Citlali."""

    if mode not in {"core", "enhanced"}:
        raise ContractError(f"unsupported analysis mode {mode}")
    enhanced = mode == "enhanced"
    inputs.validate(enhanced)
    messages = log if log is not None else []
    messages.append(f"input candidate_id={inputs.candidate_id} mode={mode}")
    config = load_yaml(inputs.config_path)
    state = load_alignment_state(inputs.provenance_path)
    evaluator = TelescopeEvaluator(inputs.telescope_path, config)
    assigned_time = state.phase_sec + np.arange(state.sample_count) * state.cadence_sec
    baseline = evaluator.evaluate(assigned_time)
    if not np.all(baseline["bracket"]):
        raise ContractError("common detector axis lies outside telescope support")
    registry_rows, axis, low_speed = build_scan_registry(
        state,
        baseline,
        evaluator.scan_angle,
        protocol,
        inputs.telescope_path,
        inputs.provenance_path,
    )
    registry = {int(row["stable_scan_id"]): row for row in registry_rows}
    scan_groups = group_selected_scans(registry_rows)
    if (
        len(scan_groups["left"]) < protocol.minimum_distinct_scans_per_direction
        or len(scan_groups["right"]) < protocol.minimum_distinct_scans_per_direction
    ):
        raise ContractError("insufficient independently selected left/right scans")
    messages.append(
        "scan_registry "
        f"left={len(scan_groups['left'])} right={len(scan_groups['right'])} "
        f"excluded={len(scan_groups['excluded'])}"
    )

    mappings: dict[int, RawMapping] = {}
    raw_rows: list[dict[str, Any]] = []
    raw_counter_rows: list[dict[str, Any]] = []
    raw_pps_time_increment_anomaly_rows: list[dict[str, Any]] = []
    model_descriptors: list[dict[str, Any]] = []
    model_coordinate: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    common_by_network: dict[int, np.ndarray] = {}
    science_mask = science_support_mask(state)
    if enhanced:
        offsets = interface_offset_map(config)
        try:
            for network, path in sorted(inputs.raw_by_network.items()):
                mapping = prove_raw_mapping(
                    path,
                    network,
                    inputs.observation_number,
                    state,
                    offsets.get(f"toltec{network}", 0.0),
                    science_mask,
                )
                mappings[network] = mapping
                raw_rows.append(raw_residual_summary(mapping))
                raw_counter_rows.extend(mapping.counter_transitions)
                raw_pps_time_increment_anomaly_rows.extend(
                    mapping.pps_time_increment_anomalies
                )
        except ContractError as error:
            raise RawLinkageError(str(error)) from error
        for basis, k, phi in protocol.enhanced_models:
            descriptor = {
                "model_id": model_id(basis, k, phi),
                "time_basis": basis,
                "row_shift_k": k,
                "phase_phi_samples": phi,
            }
            model_descriptors.append(descriptor)
            model_coordinate[descriptor["model_id"]] = {
                network: model_coordinates(
                    mapping,
                    state.sample_count,
                    basis,
                    k,
                    phi,
                    evaluator,
                )
                for network, mapping in sorted(mappings.items())
            }
        for network, mapping in sorted(mappings.items()):
            common = build_common_support(
                mapping, state.sample_count, protocol.common_row_shifts
            )
            for descriptor in model_descriptors:
                coordinates = model_coordinate[descriptor["model_id"]][network]
                common &= coordinates["row_valid"] & coordinates["valid"]
            common_by_network[network] = common
        messages.append(f"raw_linkage proved_networks={sorted(mappings)}")
    else:
        descriptor = {
            "model_id": "assigned_slot_k+0_phi+0.0",
            "time_basis": "assigned_slot",
            "row_shift_k": 0,
            "phase_phi_samples": 0.0,
        }
        model_descriptors = [descriptor]

    apt = Table.read(inputs.output_apt_path, format="ascii.ecsv")
    with Dataset(inputs.detector_tod_path) as dataset:
        detector, preselected = load_detector_contract(
            dataset, apt, protocol, inputs.observation_number
        )
        present_networks = sorted(set(int(value) for value in detector.networks))
        if enhanced:
            missing_raw = sorted(set(present_networks) - set(mappings))
            if missing_raw:
                raise RawLinkageError(
                    f"enhanced mode lacks raw linkage for detector networks {missing_raw}"
                )
        else:
            baseline_with_speed = dict(baseline)
            baseline_with_speed["vx"] = np.gradient(baseline["x"], state.cadence_sec)
            baseline_with_speed["vy"] = np.gradient(baseline["y"], state.cadence_sec)
            baseline_with_speed["shifted_slot"] = np.arange(
                state.sample_count, dtype=np.int64
            )
            baseline_with_speed["row_valid"] = np.ones(state.sample_count, dtype=bool)
            model_coordinate[descriptor["model_id"]] = {
                network: baseline_with_speed for network in present_networks
            }
            common_by_network = {
                network: np.ones(state.sample_count, dtype=bool)
                for network in present_networks
            }
        t0_session = resolve_network_t0_session(
            inputs,
            raw_rows,
            present_networks,
            enhanced,
        )
        signal_variable = dataset["signal"]
        flag_variable = dataset["flags"]
        signal_width = int(signal_variable.shape[2])
        matched, detector_controls, matched_by_network = select_matched_detectors(
            detector,
            preselected,
            signal_variable,
            flag_variable,
            signal_width,
            state,
            registry,
            baseline,
            axis,
            protocol,
            inputs.candidate_id,
        )
        messages.append(
            f"detector_cohort preselected={int(np.sum(preselected))} matched={matched.size}"
        )
        groups = ["all"]
        groups += [f"array:{name}" for name in ARRAY_NAMES.values()]
        groups += [
            f"network:toltec{network}"
            for network in sorted(set(int(detector.networks[index]) for index in matched))
        ]
        sums = {
            (descriptor["model_id"], group, direction): empty_map(protocol)
            for descriptor in model_descriptors
            for group in groups
            for direction in ("left", "right")
        }
        counts = {
            key: empty_count(protocol)
            for key in sums
        }
        selected_stable = scan_groups["left"] + scan_groups["right"]
        scan_sums = {
            (descriptor["model_id"], direction, stable): empty_map(protocol)
            for descriptor in model_descriptors
            for direction in ("left", "right")
            for stable in selected_stable
        }
        scan_counts = {key: empty_count(protocol) for key in scan_sums}
        speeds: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        sample_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for det in matched:
            network = int(detector.networks[det])
            detector_groups = [
                "all",
                f"array:{ARRAY_NAMES[int(detector.arrays[det])]}",
                f"network:toltec{network}",
            ]
            for slot in np.flatnonzero(detector.kind[det] == 2):
                ordinal = int(detector.scan_index[det, slot])
                if ordinal not in state.ordinal_to_stable:
                    raise ContractError(
                        f"UID {detector.uid[det]} references unknown output scan index {ordinal}"
                    )
                stable = state.ordinal_to_stable[ordinal]
                direction = str(registry[stable]["classification"])
                if direction not in {"left", "right"} or not bool(registry[stable]["selected"]):
                    continue
                indices, length = _slot_indices(
                    detector, int(det), int(slot), signal_width, state.sample_count
                )
                if length == 0:
                    continue
                signal = np.asarray(signal_variable[det, slot, :length], dtype=float)
                flags = np.asarray(flag_variable[det, slot, :length], dtype=int)
                base_valid = _baseline_sample_mask(
                    detector, int(det), indices, signal, flags, baseline, protocol
                )
                common = common_by_network[network][indices].copy()
                for shift in protocol.common_row_shifts:
                    common &= (indices + shift >= indices[0]) & (
                        indices + shift < indices[-1] + 1
                    )
                for descriptor_item in model_descriptors:
                    coordinate = model_coordinate[descriptor_item["model_id"]][network]
                    shifted_slot = coordinate["shifted_slot"][indices]
                    common &= (shifted_slot >= indices[0]) & (
                        shifted_slot < indices[-1] + 1
                    )
                common &= base_valid
                if not np.any(common):
                    continue
                normalized = signal[common] / float(detector.amplitude_ref[det])
                for descriptor_item in model_descriptors:
                    mid = descriptor_item["model_id"]
                    coordinate = model_coordinate[mid][network]
                    x = coordinate["x"][indices][common] - float(detector.full_x[det])
                    y = coordinate["y"][indices][common] - float(detector.full_y[det])
                    projected_speed = (
                        coordinate["vx"][indices][common] * axis[0]
                        + coordinate["vy"][indices][common] * axis[1]
                    )
                    for group in detector_groups:
                        add_samples(
                            sums[(mid, group, direction)],
                            counts[(mid, group, direction)],
                            x,
                            y,
                            normalized,
                            protocol,
                        )
                        speeds[(mid, group, direction)].append(
                            float(np.median(projected_speed))
                        )
                        sample_counts[(mid, group, direction)] += int(np.sum(common))
                    add_samples(
                        scan_sums[(mid, direction, stable)],
                        scan_counts[(mid, direction, stable)],
                        x,
                        y,
                        normalized,
                        protocol,
                    )

    fit_controls = list(detector_controls)
    timing_rows: list[dict[str, Any]] = []
    pooled_controls: dict[str, Any] = {}
    fit_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for descriptor in model_descriptors:
        mid = descriptor["model_id"]
        for group in groups:
            array_id = _group_array_id(group, detector, matched)
            direction_fits = {}
            direction_speeds = {}
            for direction in ("left", "right"):
                fit = map_fit(
                    sums[(mid, group, direction)],
                    counts[(mid, group, direction)],
                    array_id,
                    protocol,
                )
                fit_cache[(mid, group, direction)] = fit
                speed_values = speeds[(mid, group, direction)]
                if not speed_values:
                    fit = {
                        **fit,
                        "quality": False,
                        "reason": "no_selected_direction_samples",
                    }
                    direction_speeds[direction] = None
                else:
                    direction_speeds[direction] = float(np.median(speed_values))
                direction_fits[direction] = fit
                fit_controls.append(
                    {
                        "map_id": inputs.candidate_id,
                        "model_id": mid,
                        "support": "all_model_common_interior",
                        "group": group,
                        "uid": "",
                        "network_id": int(group.split("toltec", 1)[1])
                        if group.startswith("network:toltec")
                        else "",
                        "array": group.split(":", 1)[1]
                        if group.startswith("array:")
                        else "mixed" if group == "all" else ARRAY_NAMES[array_id],
                        "direction": direction,
                        "n_scans": len(scan_groups[direction]),
                        "stable_scan_ids_json": canonical_json(scan_groups[direction]),
                        "median_projected_speed_arcsec_s": direction_speeds[direction],
                        "selected_sample_count": sample_counts[(mid, group, direction)],
                        **fit,
                    }
                )
            row = _timing_row(
                descriptor,
                group,
                direction_fits["left"],
                direction_fits["right"],
                direction_speeds,
                axis,
            )
            row.update(
                {
                    "map_id": inputs.candidate_id,
                    "observation_number": inputs.observation_number,
                }
            )
            if row.get("quality"):
                if group == "all":
                    replicates = []
                    full_fits = direction_fits
                    for stable in selected_stable:
                        direction = str(registry[stable]["classification"])
                        modified_sum = (
                            sums[(mid, "all", direction)]
                            - scan_sums[(mid, direction, stable)]
                        )
                        modified_count = (
                            counts[(mid, "all", direction)]
                            - scan_counts[(mid, direction, stable)]
                        )
                        changed = map_fit(modified_sum, modified_count, -1, protocol)
                        pair = dict(full_fits)
                        pair[direction] = changed
                        value = fit_timing(
                            pair["left"],
                            pair["right"],
                            axis,
                            direction_speeds["left"],
                            direction_speeds["right"],
                        )
                        if value.get("quality"):
                            replicates.append(
                                {
                                    "omitted_stable_scan_id": stable,
                                    "timing_residual_sec": value["timing_residual_sec"],
                                    "parallel_arcsec": value["parallel_arcsec"],
                                    "perpendicular_arcsec": value["perpendicular_arcsec"],
                                }
                            )
                    row["timing_se_sec"] = jackknife_se(
                        [item["timing_residual_sec"] for item in replicates]
                    )
                    row["parallel_se_arcsec"] = jackknife_se(
                        [item["parallel_arcsec"] for item in replicates]
                    )
                    row["perpendicular_se_arcsec"] = jackknife_se(
                        [item["perpendicular_arcsec"] for item in replicates]
                    )
                    row["uncertainty_method"] = "delete_one_selected_stable_scan_jackknife"
                    row["jackknife_replicates"] = len(replicates)
                    pooled_controls[mid] = {"jackknife": replicates}
                else:
                    speed_span = (
                        direction_speeds["right"] - direction_speeds["left"]
                    )
                    timing_se, perpendicular_se = formal_timing_se(
                        direction_fits["left"],
                        direction_fits["right"],
                        axis,
                        speed_span,
                    )
                    row["timing_se_sec"] = timing_se
                    row["parallel_se_arcsec"] = timing_se * abs(speed_span)
                    row["perpendicular_se_arcsec"] = perpendicular_se
                    row["uncertainty_method"] = "formal_map_covariance_secondary_only"
                    row["jackknife_replicates"] = 0
                se = float(row["timing_se_sec"])
                estimate = float(row["timing_residual_sec"])
                row["timing_68_low_sec"] = estimate - se
                row["timing_68_high_sec"] = estimate + se
                row["timing_95_low_sec"] = estimate - 1.96 * se
                row["timing_95_high_sec"] = estimate + 1.96 * se
            timing_rows.append(row)

    baseline_id = "assigned_slot_k+0_phi+0.0"
    baseline_pooled = next(
        row
        for row in timing_rows
        if row["model_id"] == baseline_id and row["group"] == "all"
    )
    if not baseline_pooled.get("quality"):
        raise ContractError(f"primary pooled fit failed: {baseline_pooled.get('reason')}")

    ordered_selected = [
        int(row["stable_scan_id"])
        for row in sorted(
            (row for row in registry_rows if row["selected"]),
            key=lambda row: int(row["compatibility_ordinal_1based"]),
        )
    ]
    halves = {
        "first": ordered_selected[: len(ordered_selected) // 2],
        "second": ordered_selected[len(ordered_selected) // 2 :],
    }
    half_results = {}
    half_jackknife: dict[str, list[dict[str, Any]]] = {}
    baseline_speeds = {
        direction: float(
            np.median(speeds[(baseline_id, "all", direction)])
        )
        for direction in ("left", "right")
    }
    for label, stable_ids in halves.items():
        fits = {}
        partition_maps: dict[str, np.ndarray] = {}
        partition_counts: dict[str, np.ndarray] = {}
        for direction in ("left", "right"):
            chosen = [
                stable
                for stable in stable_ids
                if registry[stable]["classification"] == direction
            ]
            partition_maps[direction] = _map_total(
                {
                    (d, stable): scan_sums[(baseline_id, d, stable)]
                    for d in ("left", "right")
                    for stable in selected_stable
                },
                direction,
                chosen,
                protocol,
            )
            partition_counts[direction] = _map_total(
                {
                    (d, stable): scan_counts[(baseline_id, d, stable)]
                    for d in ("left", "right")
                    for stable in selected_stable
                },
                direction,
                chosen,
                protocol,
                counts=True,
            )
            fits[direction] = map_fit(
                partition_maps[direction],
                partition_counts[direction],
                -1,
                protocol,
            )
        half_result = fit_timing(
            fits["left"],
            fits["right"],
            axis,
            baseline_speeds["left"],
            baseline_speeds["right"],
        )
        replicates = []
        for omitted in stable_ids:
            direction = str(registry[omitted]["classification"])
            changed = map_fit(
                partition_maps[direction]
                - scan_sums[(baseline_id, direction, omitted)],
                partition_counts[direction]
                - scan_counts[(baseline_id, direction, omitted)],
                -1,
                protocol,
            )
            pair = dict(fits)
            pair[direction] = changed
            value = fit_timing(
                pair["left"],
                pair["right"],
                axis,
                baseline_speeds["left"],
                baseline_speeds["right"],
            )
            if value.get("quality"):
                replicates.append(
                    {
                        "omitted_stable_scan_id": int(omitted),
                        "timing_residual_sec": value["timing_residual_sec"],
                        "parallel_arcsec": value["parallel_arcsec"],
                        "perpendicular_arcsec": value["perpendicular_arcsec"],
                    }
                )
        half_jackknife[label] = replicates
        complete_jackknife = len(replicates) == len(stable_ids) and len(replicates) >= 2
        if half_result.get("quality") and complete_jackknife:
            half_result["timing_se_sec"] = jackknife_se(
                [item["timing_residual_sec"] for item in replicates]
            )
            half_result["uncertainty_method"] = (
                "fixed_chronological_half_delete_one_selected_scan_jackknife"
            )
            half_result["jackknife_replicates"] = len(replicates)
        else:
            half_result["timing_se_sec"] = None
            half_result["uncertainty_method"] = "unavailable_incomplete_half_jackknife"
            half_result["jackknife_replicates"] = len(replicates)
        half_results[label] = half_result

    first_second_difference_se = None
    first_second_difference_uncertainty_method = "unavailable"
    if all(
        half_results[label].get("quality")
        and half_results[label].get("timing_se_sec") is not None
        for label in ("first", "second")
    ):
        first_second_difference_se = float(
            math.hypot(
                float(half_results["first"]["timing_se_sec"]),
                float(half_results["second"]["timing_se_sec"]),
            )
        )
        first_second_difference_uncertainty_method = (
            "quadrature_of_independent_fixed_half_delete_one_selected_scan_jackknifes"
        )

    cross_axis = np.array([-axis[1], axis[0]])
    same_direction_null: dict[str, dict[str, Any]] = {}
    for direction in ("left", "right"):
        direction_ids = [
            stable
            for stable in ordered_selected
            if registry[stable]["classification"] == direction
        ]
        partition_ids = (direction_ids[::2], direction_ids[1::2])
        partition_fits = []
        for stable_ids in partition_ids:
            partition_fits.append(
                map_fit(
                    _map_total(
                        {
                            (d, stable): scan_sums[(baseline_id, d, stable)]
                            for d in ("left", "right")
                            for stable in selected_stable
                        },
                        direction,
                        stable_ids,
                        protocol,
                    ),
                    _map_total(
                        {
                            (d, stable): scan_counts[(baseline_id, d, stable)]
                            for d in ("left", "right")
                            for stable in selected_stable
                        },
                        direction,
                        stable_ids,
                        protocol,
                        counts=True,
                    ),
                    -1,
                    protocol,
                )
            )
        null_row: dict[str, Any] = {
            "quality": bool(all(fit.get("quality") for fit in partition_fits)),
            "partition_rule": "alternating chronological rank within selected direction",
            "partition_a_stable_scan_ids": list(partition_ids[0]),
            "partition_b_stable_scan_ids": list(partition_ids[1]),
        }
        if null_row["quality"]:
            delta = np.array(
                [
                    partition_fits[1]["centroid_x_arcsec"]
                    - partition_fits[0]["centroid_x_arcsec"],
                    partition_fits[1]["centroid_y_arcsec"]
                    - partition_fits[0]["centroid_y_arcsec"],
                ]
            )
            null_row.update(
                {
                    "parallel_arcsec": float(delta @ axis),
                    "perpendicular_arcsec": float(delta @ cross_axis),
                }
            )
        else:
            null_row["reason"] = "one_or_both_partition_fits_failed"
        same_direction_null[direction] = null_row

    pooled_controls[baseline_id]["time_halves"] = half_results
    pooled_controls[baseline_id]["time_half_jackknife"] = half_jackknife
    pooled_controls[baseline_id]["first_second_half_difference_se_sec"] = (
        first_second_difference_se
    )
    pooled_controls[baseline_id]["first_second_half_difference_uncertainty_method"] = (
        first_second_difference_uncertainty_method
    )
    pooled_controls[baseline_id]["same_direction_null"] = same_direction_null

    raw_by_network = {int(row["network_id"]): row for row in raw_rows}
    timing_lookup = {(row["model_id"], row["group"]): row for row in timing_rows}
    network_present: dict[int, dict[str, Any]] = {}
    for network, detector_count in sorted(matched_by_network.items()):
        group = f"network:toltec{network}"
        baseline_row = timing_lookup.get((baseline_id, group), {})
        if not baseline_row.get("quality"):
            network_present[network] = {
                "map_id": inputs.candidate_id,
                "observation_number": inputs.observation_number,
                "network_id": network,
                "array": ARRAY_NAMES[
                    int(detector.arrays[matched[detector.networks[matched] == network][0]])
                ],
                "detector_count": detector_count,
                "available": False,
                "status": f"fit_failed:{baseline_row.get('reason', 'unknown')}",
                "timing_residual_sec": "",
                "timing_se_sec": "",
            }
            continue
        row = {
            "map_id": inputs.candidate_id,
            "observation_number": inputs.observation_number,
            "network_id": network,
            "array": ARRAY_NAMES[
                int(detector.arrays[matched[detector.networks[matched] == network][0]])
            ],
            "detector_count": detector_count,
            "available": True,
            "status": "available",
            "left_scan_count": len(scan_groups["left"]),
            "right_scan_count": len(scan_groups["right"]),
            "timing_residual_sec": baseline_row["timing_residual_sec"],
            "timing_se_sec": baseline_row["timing_se_sec"],
            "uncertainty_method": baseline_row["uncertainty_method"],
            "parallel_arcsec": baseline_row["parallel_arcsec"],
            "perpendicular_arcsec": baseline_row["perpendicular_arcsec"],
            "v_left_arcsec_s": baseline_row["v_left_arcsec_s"],
            "v_right_arcsec_s": baseline_row["v_right_arcsec_s"],
            "scan_speed_abs_arcsec_s": 0.5
            * (
                abs(float(baseline_row["v_left_arcsec_s"]))
                + abs(float(baseline_row["v_right_arcsec_s"]))
            ),
            "left_major_fwhm_arcsec": baseline_row["left_major_fwhm_arcsec"],
            "right_major_fwhm_arcsec": baseline_row["right_major_fwhm_arcsec"],
            "left_minor_fwhm_arcsec": baseline_row["left_minor_fwhm_arcsec"],
            "right_minor_fwhm_arcsec": baseline_row["right_minor_fwhm_arcsec"],
            "left_amplitude": baseline_row["left_amplitude"],
            "right_amplitude": baseline_row["right_amplitude"],
        }
        if network in raw_by_network:
            row.update(raw_by_network[network])
            row["native_to_assigned_slot_residual_sec"] = row[
                "native_to_assigned_mean_sec"
            ]
        for descriptor in model_descriptors:
            mid = descriptor["model_id"]
            model_row = timing_lookup.get((mid, group), {})
            prefix = (
                "assigned_baseline"
                if mid == baseline_id
                else "raw_baseline"
                if mid == "raw_detector_timestamp_k+0_phi+0.0"
                else "assigned_counterfactual_k1_phi0p5"
                if mid == "assigned_slot_k+1_phi+0.5"
                else "raw_counterfactual_k1_phi0p5"
            )
            row[f"{prefix}_timing_sec"] = model_row.get("timing_residual_sec", "")
            row[f"{prefix}_timing_se_sec"] = model_row.get("timing_se_sec", "")
        network_present[network] = row
    network_rows = explicit_missing_network_rows(
        protocol.expected_networks, network_present, inputs.candidate_id
    )
    for row in network_rows:
        row.setdefault("observation_number", inputs.observation_number)
        row.setdefault("analysis_role", inputs.analysis_role)
    within_map_predictors = {
        "native_to_assigned_slot": linear_predictor_diagnostic(
            network_rows, "native_to_assigned_slot_residual_sec"
        ),
        "native_frame_phase": linear_predictor_diagnostic(
            network_rows, "native_frame_phase_mean_sec"
        ),
    }

    selected_speed = np.asarray(
        [
            abs(float(row["median_projected_velocity_arcsec_s"]))
            for row in registry_rows
            if row["selected"]
        ]
    )
    mean_major = 0.25 * (
        baseline_pooled["left_major_fwhm_arcsec"]
        + baseline_pooled["right_major_fwhm_arcsec"]
        + baseline_pooled["left_minor_fwhm_arcsec"]
        + baseline_pooled["right_minor_fwhm_arcsec"]
    )
    map_summary = {
        "schema": RUNNER_SCHEMA,
        "map_id": inputs.candidate_id,
        "candidate_id": inputs.candidate_id,
        "observation_number": inputs.observation_number,
        "analysis_role": inputs.analysis_role,
        "analysis_mode": mode,
        "status": "success",
        "quality": True,
        "cadence_sec": state.cadence_sec,
        "left_scan_count": len(scan_groups["left"]),
        "right_scan_count": len(scan_groups["right"]),
        "excluded_scan_count": len(scan_groups["excluded"]),
        "preselected_detector_count": int(np.sum(preselected)),
        "matched_detector_count": int(matched.size),
        "network_count": len(matched_by_network),
        "timing_residual_sec": baseline_pooled["timing_residual_sec"],
        "timing_se_sec": baseline_pooled["timing_se_sec"],
        "timing_95_low_sec": baseline_pooled["timing_95_low_sec"],
        "timing_95_high_sec": baseline_pooled["timing_95_high_sec"],
        "uncertainty_method": baseline_pooled["uncertainty_method"],
        "parallel_arcsec": baseline_pooled["parallel_arcsec"],
        "perpendicular_arcsec": baseline_pooled["perpendicular_arcsec"],
        "parallel_fwhm_fraction": abs(float(baseline_pooled["parallel_arcsec"]))
        / float(mean_major),
        "v_left_arcsec_s": baseline_pooled["v_left_arcsec_s"],
        "v_right_arcsec_s": baseline_pooled["v_right_arcsec_s"],
        "scan_speed_abs_p05_arcsec_s": float(np.quantile(selected_speed, 0.05)),
        "scan_speed_abs_median_arcsec_s": float(np.median(selected_speed)),
        "scan_speed_abs_p95_arcsec_s": float(np.quantile(selected_speed, 0.95)),
        "left_major_fwhm_arcsec": baseline_pooled["left_major_fwhm_arcsec"],
        "right_major_fwhm_arcsec": baseline_pooled["right_major_fwhm_arcsec"],
        "left_minor_fwhm_arcsec": baseline_pooled["left_minor_fwhm_arcsec"],
        "right_minor_fwhm_arcsec": baseline_pooled["right_minor_fwhm_arcsec"],
        "left_amplitude": baseline_pooled["left_amplitude"],
        "right_amplitude": baseline_pooled["right_amplitude"],
        "first_half_timing_sec": half_results["first"].get("timing_residual_sec"),
        "first_half_timing_se_sec": half_results["first"].get("timing_se_sec"),
        "second_half_timing_sec": half_results["second"].get("timing_residual_sec"),
        "second_half_timing_se_sec": half_results["second"].get("timing_se_sec"),
        "first_second_half_difference_sec": (
            float(half_results["first"]["timing_residual_sec"])
            - float(half_results["second"]["timing_residual_sec"])
        )
        if half_results["first"].get("quality") and half_results["second"].get("quality")
        else None,
        "first_second_half_difference_se_sec": first_second_difference_se,
        "first_second_half_difference_uncertainty_method": (
            first_second_difference_uncertainty_method
        ),
        "same_direction_null_left_parallel_arcsec": same_direction_null["left"].get(
            "parallel_arcsec"
        ),
        "same_direction_null_left_perpendicular_arcsec": same_direction_null[
            "left"
        ].get("perpendicular_arcsec"),
        "same_direction_null_right_parallel_arcsec": same_direction_null["right"].get(
            "parallel_arcsec"
        ),
        "same_direction_null_right_perpendicular_arcsec": same_direction_null[
            "right"
        ].get("perpendicular_arcsec"),
        "physical_timestamp_semantics": "unresolved",
        "raw_row_reassociation_claimed": False,
        "production_correction_authorized": False,
        "network_t0_vector_sha256": t0_session["network_t0_vector_sha256"],
        "network_t0_vector_authority": t0_session[
            "network_t0_vector_authority"
        ],
        "manifest_t0_authority_validated_against_raw": t0_session[
            "manifest_authority_validated_against_raw"
        ],
        "raw_recomputed_network_t0_vector_sha256": t0_session[
            "raw_recomputed_network_t0_vector_sha256"
        ],
        "t0_session_status": t0_session["status"],
        "within_map_slot_predictor_slope": within_map_predictors[
            "native_to_assigned_slot"
        ].get("slope"),
        "within_map_slot_predictor_pearson": within_map_predictors[
            "native_to_assigned_slot"
        ].get("pearson"),
        "within_map_native_phase_predictor_slope": within_map_predictors[
            "native_frame_phase"
        ].get("slope"),
        "first_second_half_interpretation": (
            "within-observation timing variation; not clock drift absent raw counter contradiction"
        ),
    }
    for mid, prefix in (
        ("raw_detector_timestamp_k+0_phi+0.0", "raw_baseline"),
        ("assigned_slot_k+1_phi+0.5", "assigned_counterfactual_k1_phi0p5"),
        ("raw_detector_timestamp_k+1_phi+0.5", "raw_counterfactual_k1_phi0p5"),
    ):
        value = timing_lookup.get((mid, "all"))
        if value and value.get("quality"):
            map_summary[f"{prefix}_timing_sec"] = value["timing_residual_sec"]
            map_summary[f"{prefix}_timing_se_sec"] = value["timing_se_sec"]
    fit_controls_json = {
        "schema": "sci-align-001-3c273-fit-controls-v1",
        "cohort": {
            "preselected_detector_count": int(np.sum(preselected)),
            "matched_detector_count": int(matched.size),
            "matched_by_network": {
                str(key): value for key, value in sorted(matched_by_network.items())
            },
            "excluded_uids": list(
                protocol.excluded_uids_for_observation(inputs.observation_number)
            ),
            "fixture_specific_exclusion": inputs.observation_number
            == protocol.fixture_exclusion_observation,
            "selection_depends_on_timing_estimate": False,
        },
        "scan_axis": {
            "x_az_tangent": float(axis[0]),
            "y_el_tangent": float(axis[1]),
            "low_speed_threshold_arcsec_s": low_speed,
        },
        "pooled_controls": pooled_controls,
    }
    map_result = {
        "schema": RUNNER_SCHEMA,
        "identity": inputs.identity(),
        "protocol": protocol.to_dict(),
        "summary": map_summary,
        "primary": baseline_pooled,
        "timing_models": [
            row for row in timing_rows if row["group"] == "all"
        ],
        "network_results": network_rows,
        "raw_linkage": raw_rows,
        "raw_counter_transition_product": {
            "path": "raw_counter_transitions.csv",
            "row_count": len(raw_counter_rows),
            "grain": "one delivered PpsCount transition per analyzed raw network",
            "checksum_authority": "candidate SHA256SUMS",
        },
        "raw_pps_time_increment_anomaly_product": {
            "path": "raw_pps_time_increment_anomalies.csv",
            "row_count": len(raw_pps_time_increment_anomaly_rows),
            "grain": "one delivered PpsTime increment mismatch per raw network",
            "metadata_to_integration_association_proved": False,
            "row_mask_or_repair_authorized": False,
        },
        "raw_phase_summary": raw_rows,
        "t0_session": t0_session,
        "within_map_predictors": within_map_predictors,
        "controls": fit_controls_json,
        "scope": {
            "citlali_reduction_launched": False,
            "source_products_modified": False,
            "unity_contacted": False,
            "physical_timestamp_semantics": "unresolved",
            "raw_row_reassociation_claimed": False,
            "production_correction_authorized": False,
            "producer_clock_architecture": (
                "NTP integer-second T0; shared Octo 10MHz and PPS; PPS does not reset cadence"
            ),
            "arbitrary_millisecond_ntp_error": "strongly_disfavored",
            "differential_oscillator_drift": "strongly_disfavored",
            "distinct_stable_network_integration_phase": "allowed",
            "fpga_metadata_to_integration_association": "unresolved",
        },
    }
    messages.append(
        f"primary timing_sec={baseline_pooled['timing_residual_sec']:.12g} "
        f"se_sec={baseline_pooled['timing_se_sec']:.12g}"
    )
    return AnalysisProducts(
        map_summary,
        map_result,
        network_rows,
        timing_rows,
        fit_controls,
        fit_controls_json,
        registry_rows,
        raw_rows,
        raw_counter_rows,
        raw_rows,
        raw_pps_time_increment_anomaly_rows,
    )
