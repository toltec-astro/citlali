#!/usr/bin/env python3
"""Inventory retained 3C273 Beammap products for SCI-ALIGN-001.

The inventory is deliberately read-only with respect to every reduction and
raw-data root.  It records only provenance and input capability; timing fit
results are neither opened nor used for eligibility or duplicate selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from netCDF4 import Dataset


INVENTORY_SCHEMA = "sci-align-001-3c273-candidate-inventory-v2"
SELECTION_SCHEMA = "sci-align-001-3c273-selection-v2"
SELECTED_MANIFEST_SCHEMA = "sci-align-001-3c273-selected-manifest-v2"
ALLOWLIST_SCHEMA = "sci-align-001-3c273-obsnum-allowlist-v1"
DIGEST_CACHE_SCHEMA = "sci-align-001-file-digest-cache-v1"
DEFAULT_SOURCE_REGEX = r"(?i)^3c[ _-]?273$"
DEFAULT_LARGE_FILE_THRESHOLD = 64 * 1024 * 1024
REDU_RE = re.compile(r"^redu\d+$", re.IGNORECASE)
RAW_NAME_RE = re.compile(r"^toltec(?P<network>\d+)_(?P<obsnum>\d+)_.*\.nc$")
OBS_TIME_RE = re.compile(
    r"(?P<year>20\d{2})[_-](?P<month>\d{2})[_-](?P<day>\d{2})"
    r"[_T-](?P<hour>\d{2})[_:-](?P<minute>\d{2})[_:-](?P<second>\d{2})"
)

INVENTORY_FIELDS = [
    "candidate_id",
    "map_id",
    "observation_number",
    "obsnum",
    "observation_start_utc",
    "observation_date",
    "session_id",
    "session_status",
    "reduction_id",
    "duplicate_group_id",
    "provenance_signature_id",
    "reduction_path",
    "reduction_run_path",
    "project_path",
    "source_status",
    "source_normalized",
    "source_identities_json",
    "product_identity_resolutions_json",
    "config_path",
    "config_sha256",
    "config_digest_status",
    "config_size_bytes",
    "software_version",
    "software_sha",
    "software_identities_json",
    "provenance_path",
    "provenance_sha256",
    "provenance_digest_status",
    "provenance_size_bytes",
    "detector_tod_path",
    "detector_tod_sha256",
    "detector_tod_digest_status",
    "detector_tod_size_bytes",
    "telescope_path",
    "telescope_sha256",
    "telescope_digest_status",
    "telescope_size_bytes",
    "output_apt_path",
    "output_apt_sha256",
    "output_apt_digest_status",
    "output_apt_size_bytes",
    "scan_registry_available",
    "detector_networks_json",
    "detector_count_by_network_json",
    "raw_files_json",
    "raw_networks_json",
    "missing_raw_networks_json",
    "raw_timestamp_available",
    "raw_linkage_status",
    "raw_linkage_reasons_json",
    "network_t0_vector_json",
    "network_t0_vector_sha256",
    "network_t0_status",
    "timestamp_counter_fields_json",
    "timestamp_semantics_json",
    "core_eligible",
    "enhanced_eligible",
    "eligibility",
    "exclusion_reasons_json",
    "canonical_quality_score",
    "canonical_proposal",
    "canonical_proposal_rule",
    "owner_selection_required",
    "in_authoritative_corpus",
]

SELECTED_FIELDS = [
    "candidate_id",
    "map_id",
    "observation_number",
    "obsnum",
    "analysis_role",
    "observation_start_utc",
    "observation_date",
    "session_id",
    "session_status",
    "reduction_id",
    "duplicate_group_id",
    "provenance_signature_id",
    "reduction_path",
    "reduction_run_path",
    "project_path",
    "config_path",
    "config_sha256",
    "config_digest_status",
    "config_size_bytes",
    "software_version",
    "software_sha",
    "detector_tod_path",
    "detector_tod_sha256",
    "detector_tod_digest_status",
    "detector_tod_size_bytes",
    "telescope_path",
    "telescope_sha256",
    "telescope_digest_status",
    "telescope_size_bytes",
    "output_apt_path",
    "output_apt_sha256",
    "output_apt_digest_status",
    "output_apt_size_bytes",
    "provenance_path",
    "provenance_sha256",
    "provenance_digest_status",
    "provenance_size_bytes",
    "detector_networks_json",
    "raw_files_json",
    "raw_networks_json",
    "missing_raw_networks_json",
    "raw_timestamp_available",
    "raw_linkage_status",
    "network_t0_vector_json",
    "network_t0_vector_sha256",
    "network_t0_status",
    "timestamp_counter_fields_json",
    "timestamp_semantics_json",
    "core_eligible",
    "enhanced_eligible",
]


class InventoryError(ValueError):
    """Fail-closed inventory or selection error."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_object(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256sums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS"
    files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path != checksum_path
        ),
        key=lambda path: str(path.relative_to(directory)),
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "selected"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    raise InventoryError(f"{label}: expected a boolean selection value")


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def paths_overlap(first: Path, second: Path) -> bool:
    first = resolved(first)
    second = resolved(second)
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def validate_roots(
    reduction_roots: Sequence[Path], raw_roots: Sequence[Path], output: Path,
    excluded_paths: Sequence[Path] = (),
) -> tuple[list[Path], list[Path], Path, list[Path]]:
    reductions = sorted({resolved(path) for path in reduction_roots}, key=str)
    raws = sorted({resolved(path) for path in raw_roots}, key=str)
    destination = resolved(output)
    exclusions = sorted({resolved(path) for path in excluded_paths}, key=str)
    if not reductions:
        raise InventoryError("at least one --reduction-root is required")
    for label, roots in (("reduction", reductions), ("raw", raws)):
        missing = [str(path) for path in roots if not path.is_dir()]
        if missing:
            raise InventoryError(f"{label} root is not a directory: {missing}")
        for path in roots:
            if paths_overlap(path, destination) and not (
                label == "reduction"
                and any(destination.is_relative_to(exclusion) for exclusion in exclusions)
                and any(exclusion.is_relative_to(path) for exclusion in exclusions)
            ):
                raise InventoryError(
                    f"output directory overlaps {label} source root: {destination} and {path}"
                )
    for exclusion in exclusions:
        if not any(exclusion.is_relative_to(root) for root in reductions):
            raise InventoryError(
                f"--exclude-path must be below a reduction root: {exclusion}"
            )
    if any(destination.is_relative_to(root) for root in reductions) and not any(
        destination.is_relative_to(exclusion) for exclusion in exclusions
    ):
        raise InventoryError(
            "an output below --reduction-root requires an enclosing --exclude-path"
        )
    return reductions, raws, destination, exclusions


@dataclass(frozen=True)
class DigestResult:
    size_bytes: int
    status: str
    sha256: str | None


@dataclass(frozen=True)
class ProductResolution:
    """Deterministic resolution of one required retained product identity."""

    path: Path | None
    candidates: tuple[Path, ...]

    @property
    def status(self) -> str:
        if len(self.candidates) == 1:
            return "unique"
        return "ambiguous" if self.candidates else "missing"


def product_resolution(paths: Iterable[Path]) -> ProductResolution:
    candidates = tuple(find_existing(paths))
    return ProductResolution(
        path=candidates[0] if len(candidates) == 1 else None,
        candidates=candidates,
    )


class DigestCache:
    """Cache SHA-256 by physical-file identity and hash each inode once per run."""

    def __init__(self, path: Path, *, threshold: int, hash_large: bool) -> None:
        self.path = path
        self.threshold = threshold
        self.hash_large = hash_large
        self.entries: dict[str, str] = {}
        self._memo: dict[str, DigestResult] = {}
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("schema_version") == DIGEST_CACHE_SCHEMA:
                    raw_entries = value.get("entries", {})
                    if isinstance(raw_entries, dict):
                        self.entries = {
                            str(key): str(digest)
                            for key, digest in raw_entries.items()
                            if re.fullmatch(r"[0-9a-f]{64}", str(digest))
                        }
            except (OSError, json.JSONDecodeError):
                self.entries = {}

    @staticmethod
    def physical_key(path: Path) -> tuple[str, int]:
        stat = path.stat()
        key = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
        return key, stat.st_size

    def digest(self, path: Path) -> DigestResult:
        path = resolved(path)
        key, size = self.physical_key(path)
        if key in self._memo:
            return self._memo[key]
        if size > self.threshold and not self.hash_large:
            result = DigestResult(size, "not_hashed_large", None)
            self._memo[key] = result
            return result
        digest = self.entries.get(key)
        if digest is None:
            state = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    state.update(block)
            digest = state.hexdigest()
            self.entries[key] = digest
        result = DigestResult(size, "sha256", digest)
        self._memo[key] = result
        return result

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": DIGEST_CACHE_SCHEMA,
            "physical_identity": "device:inode:size:mtime_ns",
            "entries": dict(sorted(self.entries.items())),
        }


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": json_safe(value.tolist()),
        }
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def decode_text(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "S":
        if array.dtype.itemsize == 1:
            raw = b"".join(bytes(item) for item in array.ravel())
            return raw.decode("utf-8", errors="replace").rstrip("\x00 ")
        return "".join(
            bytes(item).decode("utf-8", errors="replace") for item in array.ravel()
        ).rstrip("\x00 ")
    if array.dtype.kind == "U":
        return "".join(str(item) for item in array.ravel()).rstrip("\x00 ")
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").rstrip("\x00 ")
        return str(item).rstrip("\x00 ")
    if array.size == 1:
        return str(array.ravel()[0]).rstrip("\x00 ")
    return "".join(str(item) for item in array.ravel()).rstrip("\x00 ")


def normalized_source(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", "").strip())


def canonical_source(value: str) -> str:
    return re.sub(r"[\s_-]+", "", normalized_source(value).casefold())


def source_record(path: Path, field: str, value: Any) -> dict[str, Any]:
    decoded = decode_text(value)
    return {
        "path": str(resolved(path)),
        "field": field,
        "raw": json_safe(value),
        "decoded": decoded,
        "normalized": normalized_source(decoded),
        "canonical": canonical_source(decoded),
    }


def yaml_load(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def recursive_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if str(name) == key:
                result.append(child)
            result.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            result.extend(recursive_values(child, key))
    return result


def first_scalar(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def product_markers(path: Path) -> bool:
    return any(
        marker.exists()
        for marker in (
            path / "index.yaml",
            path / "timestream_output_provenance.yaml",
            path / "raw/source_crossing_tod",
            path / "raw",
        )
    )


def discover_candidate_dirs(
    roots: Sequence[Path], excluded_paths: Sequence[Path] = ()
) -> list[Path]:
    candidates: set[Path] = set()
    for root in roots:
        if root.name.isdigit() and product_markers(root) and not any(
            root.is_relative_to(exclusion) for exclusion in excluded_paths
        ):
            candidates.add(root)
        for path in root.rglob("*"):
            if any(path.is_relative_to(exclusion) for exclusion in excluded_paths):
                continue
            if not path.is_dir() or not path.name.isdigit():
                continue
            if REDU_RE.fullmatch(path.parent.name) or product_markers(path):
                candidates.add(path)
    return sorted((resolved(path) for path in candidates), key=str)


def load_obsnum_allowlist(path: Path) -> tuple[dict[str, Any], list[int], str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read --obsnum-allowlist {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != ALLOWLIST_SCHEMA:
        raise InventoryError("unsupported --obsnum-allowlist schema_version")
    if not isinstance(document.get("corpus_id"), str) or not document["corpus_id"].strip():
        raise InventoryError("--obsnum-allowlist lacks corpus_id")
    values = document.get("obsnums")
    if not isinstance(values, list) or not values:
        raise InventoryError("--obsnum-allowlist must contain a nonempty obsnums array")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise InventoryError("--obsnum-allowlist obsnums must be positive integers")
    if values != sorted(values) or len(values) != len(set(values)):
        raise InventoryError("--obsnum-allowlist obsnums must be sorted and unique")
    return document, list(values), sha256_file(path)


def project_path_for(candidate: Path) -> Path:
    for ancestor in candidate.parents:
        if ancestor.name == "reduced":
            return ancestor.parent
    return candidate.parent


def reduction_run_for(candidate: Path) -> Path:
    return candidate.parent if REDU_RE.fullmatch(candidate.parent.name) else candidate


def find_existing(paths: Iterable[Path]) -> list[Path]:
    return sorted({resolved(path) for path in paths if path.is_file()}, key=str)


def find_config(candidate: Path, obsnum: int) -> ProductResolution:
    run = reduction_run_for(candidate)
    project = project_path_for(candidate)
    # A completed reduction's config-source manifest is the provenance
    # authority for the copied input config.  Prefer that checksum-verified
    # immutable copy over path-nearby duplicates of the same YAML.
    manifests = find_existing(
        [run / "config_source_manifest.yaml", candidate / "config_source_manifest.yaml"]
    )
    authoritative_copies: list[Path] = []
    for manifest in manifests:
        value = yaml_load(manifest)
        sources = value.get("sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            copied_name = source.get("copied_filename")
            recorded_sha = str(source.get("sha256") or "").lower()
            if not copied_name or not re.fullmatch(r"[0-9a-f]{64}", recorded_sha):
                continue
            copied = run / str(copied_name)
            if (
                copied.is_file()
                and f"{obsnum}" in copied.name
                and sha256_file(copied) == recorded_sha
            ):
                authoritative_copies.append(copied)
    authoritative_copies = find_existing(authoritative_copies)
    if authoritative_copies:
        return product_resolution(authoritative_copies)

    search_dirs = [candidate, run, run.parent, project, project / "config"]
    for ancestor in candidate.parents:
        if ancestor.name == "reduced":
            search_dirs.append(ancestor)
            break
    choices: list[Path] = []
    for directory in search_dirs:
        choices.extend(directory.glob(f"citlali_o{obsnum}_*.yaml"))
    choices = find_existing(choices)
    if choices:
        return product_resolution(choices)
    for manifest in manifests:
        value = yaml_load(manifest)
        for raw in recursive_values(value, "source_path"):
            path = Path(str(raw)).expanduser()
            if path.is_file() and f"{obsnum}" in path.name:
                choices.append(resolved(path))
    return product_resolution(choices)


def config_items(config: dict[str, Any]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    inputs = config.get("inputs")
    if not isinstance(inputs, list):
        return result
    for input_group in inputs:
        if not isinstance(input_group, dict):
            continue
        for item in input_group.get("data_items", []) or []:
            if not isinstance(item, dict) or not item.get("filepath"):
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            result.append(
                (str(meta.get("interface") or ""), resolved(Path(str(item["filepath"]))))
            )
    return result


def nearby_telescope(candidate: Path, obsnum: int) -> list[Path]:
    choices: list[Path] = []
    for ancestor in candidate.parents:
        if ancestor.name == "reduced":
            choices.extend(ancestor.glob(f"tel*{obsnum}*.nc"))
            break
    return find_existing(choices)


def find_detector_tod(candidate: Path) -> ProductResolution:
    choices = find_existing(
        list((candidate / "raw/source_crossing_tod").glob("*_ptc_detector_tod.nc"))
        + list((candidate / "raw").glob("**/*_ptc_detector_tod.nc"))
    )
    return product_resolution(choices)


def find_output_apt(candidate: Path) -> ProductResolution:
    choices = find_existing(
        path
        for path in (candidate / "raw").glob("*apt*_citlali.ecsv")
        if "fit_qc" not in path.name and "prior" not in path.name
    )
    return product_resolution(choices)


def find_provenance(candidate: Path) -> ProductResolution:
    primary = candidate / "timestream_output_provenance.yaml"
    if primary.is_file():
        return product_resolution([primary])
    choices = find_existing(candidate.glob("*timestream_output*provenance*.yaml"))
    return product_resolution(choices)


def read_detector_metadata(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "obsnum": None,
        "source_records": [],
        "networks": [],
        "detector_count_by_network": {},
        "schema_ok": False,
    }
    if path is None:
        return result
    try:
        with Dataset(path) as dataset:
            if "obsnum" in dataset.variables:
                result["obsnum"] = int(np.asarray(dataset["obsnum"][...]).item())
            if "SOURCE" in dataset.variables:
                raw = dataset["SOURCE"][...]
                result["source_records"].append(source_record(path, "SOURCE", raw))
            required = {
                "detector_tod_uid",
                "detector_tod_network",
                "detector_tod_slot_kind",
                "detector_tod_n_samples",
                "signal",
                "flags",
            }
            if required.issubset(dataset.variables):
                networks = np.asarray(dataset["detector_tod_network"][:], dtype=int)
                unique, counts = np.unique(networks, return_counts=True)
                result["networks"] = [int(item) for item in unique]
                result["detector_count_by_network"] = {
                    str(int(network)): int(count)
                    for network, count in zip(unique, counts)
                }
                result["schema_ok"] = True
    except (OSError, KeyError, TypeError, ValueError):
        result["schema_ok"] = False
    return result


def read_telescope_metadata(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "obsnum": None,
        "source_records": [],
        "trajectory_ok": False,
    }
    if path is None:
        return result
    try:
        with Dataset(path) as dataset:
            for name in ("Header.Dcs.ObsNum", "obsnum"):
                if name in dataset.variables:
                    result["obsnum"] = int(np.asarray(dataset[name][...]).item())
                    break
            for name in ("Header.Source.SourceName", "SOURCE", "SourceName"):
                if name in dataset.variables:
                    result["source_records"].append(
                        source_record(path, name, dataset[name][...])
                    )
            result["trajectory_ok"] = all(
                name in dataset.variables
                for name in (
                    "Data.TelescopeBackend.TelTime",
                    "Data.TelescopeBackend.TelAzAct",
                    "Data.TelescopeBackend.TelElAct",
                    "Data.TelescopeBackend.Hold",
                )
            )
    except (OSError, KeyError, TypeError, ValueError):
        result["trajectory_ok"] = False
    return result


def raw_file_index(raw_roots: Sequence[Path]) -> dict[int, list[Path]]:
    result: dict[int, list[Path]] = {}
    for root in raw_roots:
        for path in root.rglob("*.nc"):
            match = RAW_NAME_RE.fullmatch(path.name)
            if match:
                result.setdefault(int(match.group("obsnum")), []).append(resolved(path))
    return {
        obsnum: sorted(set(paths), key=str)
        for obsnum, paths in sorted(result.items())
    }


def raw_capability(path: Path, configured_interface: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(resolved(path)),
        "configured_interface": configured_interface,
        "network": None,
        "timestamp_available": False,
        "fpga_association_status": "unproved",
        "t0": None,
        "fields": {},
        "source_records": [],
    }
    try:
        with Dataset(path) as dataset:
            network_name = "Header.Toltec.RoachIndex"
            if network_name in dataset.variables:
                record["network"] = int(np.asarray(dataset[network_name][...]).item())
            for source_name in ("Header.Source.SourceName", "SOURCE"):
                if source_name in dataset.variables:
                    record["source_records"].append(
                        source_record(path, source_name, dataset[source_name][...])
                    )
            ts_name = "Data.Toltec.Ts"
            fields = {
                "clock_time_integer_t0": {"field": ts_name, "column": 0},
                "pps_count": {"field": ts_name, "column": 1},
                "clock_count": {"field": ts_name, "column": 2},
                "packet_count": {"field": ts_name, "column": 3},
                "pps_time": {"field": ts_name, "column": 4},
                "clock_time_nanosec": {"field": ts_name, "column": 5},
            }
            if ts_name in dataset.variables:
                variable = dataset[ts_name]
                shape = tuple(variable.shape)
                available = len(shape) == 2 and shape[0] > 0 and shape[1] >= 6
                record["timestamp_available"] = available
                first = np.asarray(variable[0, :6], dtype=np.int64) if available else None
                for name, identity in fields.items():
                    column = int(identity["column"])
                    record["fields"][name] = {
                        **identity,
                        "available": available,
                        "first_row_value": int(first[column]) if first is not None else None,
                    }
                if first is not None:
                    # Producer authority defines T0 as integer ClockTime column
                    # 0.  Read exactly that single timestamp column over all
                    # rows; detector I/Q and timing-fit products remain closed.
                    distinct_t0 = np.unique(
                        np.asarray(variable[:, 0], dtype=np.int64)
                    )
                    distinct_values = [int(value) for value in distinct_t0]
                    record["fields"]["clock_time_integer_t0"].update(
                        {
                            "distinct_value_count": len(distinct_values),
                            "distinct_values": distinct_values,
                            "constant_over_file": len(distinct_values) == 1,
                        }
                    )
                    if len(distinct_values) == 1:
                        record["t0"] = distinct_values[0]
            else:
                record["fields"] = {
                    name: {**identity, "available": False, "first_row_value": None}
                    for name, identity in fields.items()
                }
            for header in ("Header.Toltec.FpgaFreq", "Header.Toltec.AccumLen"):
                key = "fpga_freq" if header.endswith("FpgaFreq") else "accum_len"
                available = header in dataset.variables
                record["fields"][key] = {
                    "field": header,
                    "available": available,
                    "value": (
                        json_safe(np.asarray(dataset[header][...]).item())
                        if available
                        else None
                    ),
                }
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        pass
    return record


def observation_time(paths: Iterable[Path]) -> tuple[str | None, str | None]:
    values: list[datetime] = []
    for path in paths:
        match = OBS_TIME_RE.search(path.name)
        if not match:
            continue
        try:
            values.append(
                datetime(
                    *(int(match.group(name)) for name in (
                        "year", "month", "day", "hour", "minute", "second"
                    )),
                    tzinfo=timezone.utc,
                )
            )
        except ValueError:
            continue
    if not values:
        return None, None
    value = min(values)
    return value.isoformat().replace("+00:00", "Z"), value.date().isoformat()


def source_disposition(
    records: list[dict[str, Any]], pattern: re.Pattern[str]
) -> tuple[str, str | None, list[str]]:
    nonempty = [record for record in records if record["normalized"]]
    if not nonempty:
        return "missing", None, ["source_identity_missing"]
    canonical = {record["canonical"] for record in nonempty}
    if len(canonical) != 1:
        return "ambiguous", None, ["source_identity_ambiguous"]
    matches = [bool(pattern.fullmatch(record["normalized"])) for record in nonempty]
    if not all(matches):
        return "not_target", nonempty[0]["normalized"], ["source_regex_fullmatch_failed"]
    return "target", nonempty[0]["normalized"], []


def scan_registry_available(provenance: dict[str, Any]) -> bool:
    plans = recursive_values(provenance, "sci_align_scan_plan")
    return any(
        isinstance(plan, dict)
        and isinstance(plan.get("records"), list)
        and len(plan["records"]) > 0
        for plan in plans
    )


def index_software(candidate: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    index = yaml_load(candidate / "index.yaml")
    raw_version = first_scalar(index.get("citlali_version"))
    if raw_version:
        version = str(raw_version)
        match = re.search(r"-g([0-9a-f]{7,40})(?:$|-)", version)
        identities.append(
            {
                "authority": "result_index",
                "field": "citlali_version",
                "value": version,
                "sha": match.group(1) if match else None,
            }
        )
    for value in recursive_values(provenance, "source_application_sha"):
        sha = str(first_scalar(value))
        identities.append(
            {
                "authority": "timestream_provenance",
                "field": "source_application_sha",
                "value": sha,
                "sha": sha if re.fullmatch(r"[0-9a-f]{7,40}", sha) else None,
            }
        )
    identities.sort(key=lambda row: (row["authority"], row["field"], row["value"]))
    version_identity = next(
        (row for row in identities if row["authority"] == "result_index"), None
    )
    return {
        "version": version_identity["value"] if version_identity else None,
        "sha": version_identity["sha"] if version_identity else None,
        "identities": identities,
    }


def configured_source_records(config_path: Path | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    if config_path is None:
        return []
    source = config.get("source")
    values: list[tuple[str, Any]] = []
    if isinstance(source, str):
        values.append(("source", source))
    elif isinstance(source, dict):
        for key in ("name", "source_name", "source"):
            if source.get(key) not in (None, ""):
                values.append((f"source.{key}", source[key]))
    return [source_record(config_path, field, value) for field, value in values]


def session_identity(
    observation_date: str | None,
    t0_vector: list[dict[str, int]],
    t0_status: str,
    raw_linkage_status: str,
) -> tuple[str | None, str]:
    if t0_status == "complete_unambiguous" and raw_linkage_status == "config_proven":
        return "roach-t0:" + digest_object(t0_vector)[:20], "network_t0_vector"
    if observation_date:
        return f"date:{observation_date}", "date_group_fallback"
    return None, "unavailable"


def candidate_inventory(
    candidate: Path,
    *,
    source_pattern: re.Pattern[str],
    raw_by_obsnum: dict[int, list[Path]],
    digest_cache: DigestCache,
) -> dict[str, Any]:
    obsnum = int(candidate.name)
    run = reduction_run_for(candidate)
    project = project_path_for(candidate)
    config_resolution = find_config(candidate, obsnum)
    config_path = config_resolution.path
    config = yaml_load(config_path) if config_path else {}
    items = config_items(config)
    detector_resolution = find_detector_tod(candidate)
    detector_tod = detector_resolution.path
    output_apt_resolution = find_output_apt(candidate)
    output_apt = output_apt_resolution.path
    provenance_resolution = find_provenance(candidate)
    provenance_path = provenance_resolution.path
    provenance = yaml_load(provenance_path) if provenance_path else {}

    configured_telescope = [path for interface, path in items if interface == "lmt" and path.is_file()]
    telescope_resolution = product_resolution(
        configured_telescope or nearby_telescope(candidate, obsnum)
    )
    telescope_path = telescope_resolution.path
    product_resolutions = {
        "config": {
            "status": config_resolution.status,
            "candidates": [str(path) for path in config_resolution.candidates],
        },
        "detector_tod": {
            "status": detector_resolution.status,
            "candidates": [str(path) for path in detector_resolution.candidates],
        },
        "output_apt": {
            "status": output_apt_resolution.status,
            "candidates": [str(path) for path in output_apt_resolution.candidates],
        },
        "provenance": {
            "status": provenance_resolution.status,
            "candidates": [str(path) for path in provenance_resolution.candidates],
        },
        "telescope": {
            "status": telescope_resolution.status,
            "candidates": [str(path) for path in telescope_resolution.candidates],
        },
    }

    detector = read_detector_metadata(detector_tod)
    telescope = read_telescope_metadata(telescope_path)
    source_records = list(detector["source_records"]) + list(telescope["source_records"])
    source_records += configured_source_records(config_path, config)

    configured_raw: dict[Path, str] = {}
    configured_raw_conflicts: list[str] = []
    configured_raw_missing: list[tuple[str, Path]] = []
    for interface, path in items:
        if not re.fullmatch(r"toltec\d+", interface):
            continue
        if not path.is_file():
            configured_raw_missing.append((interface, path))
            continue
        prior_interface = configured_raw.get(path)
        if prior_interface is not None and prior_interface != interface:
            configured_raw_conflicts.append(
                f"configured_raw_path_has_multiple_interfaces_{prior_interface}_{interface}"
            )
        configured_raw[path] = interface
    # Config-linked files are the only raw association that can support the
    # enhanced eligibility gate.  Filename/obsnum discoveries are retained as
    # a fail-closed fallback, but must not introduce tune files beside a
    # complete configured set or make that set appear ambiguous.
    discovered_raw = {path: None for path in raw_by_obsnum.get(obsnum, [])}
    combined_raw = dict(configured_raw) if configured_raw else discovered_raw
    raw_records = [
        raw_capability(path, interface)
        for path, interface in sorted(combined_raw.items(), key=lambda item: str(item[0]))
    ]
    for record in raw_records:
        raw_digest = digest_cache.digest(Path(record["path"]))
        record.update(
            {
                "sha256": raw_digest.sha256,
                "digest_status": raw_digest.status,
                "size_bytes": raw_digest.size_bytes,
            }
        )
    for record in raw_records:
        source_records.extend(record["source_records"])
    source_records.sort(key=lambda row: (row["path"], row["field"], row["decoded"]))
    source_status, source_value, source_reasons = source_disposition(
        source_records, source_pattern
    )

    raw_network_records: dict[int, list[dict[str, Any]]] = {}
    for record in raw_records:
        network = record.get("network")
        if network is not None:
            raw_network_records.setdefault(int(network), []).append(record)
    raw_networks = sorted(raw_network_records)
    detector_networks = [int(value) for value in detector["networks"]]
    missing_raw = sorted(set(detector_networks) - set(raw_networks))

    linkage_reasons: list[str] = list(configured_raw_conflicts)
    linkage_reasons.extend(
        f"configured_raw_path_missing_{interface}" for interface, _ in configured_raw_missing
    )
    for network, records in raw_network_records.items():
        paths = {record["path"] for record in records}
        if len(paths) != 1:
            linkage_reasons.append(f"raw_network_{network}_ambiguous")
        configured = [record for record in records if record.get("configured_interface")]
        if not configured:
            linkage_reasons.append(f"raw_network_{network}_obsnum_only_unproved")
        for record in configured:
            expected_network = int(str(record["configured_interface"]).removeprefix("toltec"))
            if expected_network != network:
                linkage_reasons.append(
                    f"configured_interface_header_conflict_toltec{expected_network}_toltec{network}"
                )
    for record in raw_records:
        if record.get("configured_interface") and record.get("network") is None:
            linkage_reasons.append(
                f"configured_raw_network_header_missing_{record['configured_interface']}"
            )
    if missing_raw:
        linkage_reasons.append("raw_network_coverage_incomplete")
    raw_linkage_status = "config_proven" if raw_records and not linkage_reasons else (
        "unproved_or_incomplete" if raw_records else "unavailable"
    )

    t0_vector: list[dict[str, int]] = []
    t0_status = "unavailable"
    t0_ambiguous = False
    for network in raw_networks:
        values = {record.get("t0") for record in raw_network_records[network]}
        if None in values or len(values) != 1:
            t0_ambiguous = True
            continue
        t0_vector.append({"network": network, "t0": int(next(iter(values)))})
    if t0_ambiguous:
        t0_status = "ambiguous"
    elif raw_networks and len(t0_vector) == len(raw_networks) and not missing_raw:
        t0_status = "complete_unambiguous"
    elif t0_vector:
        t0_status = "incomplete"
    t0_digest = digest_object(t0_vector) if t0_status == "complete_unambiguous" else None

    timestamp_fields = [
        {
            "network": record.get("network"),
            "path": record["path"],
            "fpga_association_status": "unproved",
            "fields": record["fields"],
        }
        for record in raw_records
    ]
    timestamp_fields.sort(key=lambda row: (row["network"] is None, row["network"], row["path"]))

    config_digest = digest_cache.digest(config_path) if config_path else None
    provenance_digest = digest_cache.digest(provenance_path) if provenance_path else None
    detector_digest = digest_cache.digest(detector_tod) if detector_tod else None
    telescope_digest = digest_cache.digest(telescope_path) if telescope_path else None
    output_apt_digest = digest_cache.digest(output_apt) if output_apt else None
    software = index_software(candidate, provenance)
    observed_times = [Path(record["path"]) for record in raw_records]
    if telescope_path:
        observed_times.append(telescope_path)
    start_utc, observation_date = observation_time(observed_times)
    session_id, session_status = session_identity(
        observation_date, t0_vector, t0_status, raw_linkage_status
    )

    reasons = list(source_reasons)
    if detector.get("obsnum") not in (None, obsnum):
        reasons.append("detector_tod_observation_identity_conflict")
    if telescope.get("obsnum") not in (None, obsnum):
        reasons.append("telescope_observation_identity_conflict")
    if config_resolution.status != "unique":
        reasons.append(f"config_identity_{config_resolution.status}")
    if detector_resolution.status != "unique":
        reasons.append(f"detector_tod_identity_{detector_resolution.status}")
    elif not detector["schema_ok"]:
        reasons.append("detector_tod_schema_incomplete")
    elif not detector_networks:
        reasons.append("detector_tod_networks_empty")
    if telescope_resolution.status != "unique":
        reasons.append(f"telescope_trajectory_identity_{telescope_resolution.status}")
    elif not telescope["trajectory_ok"]:
        reasons.append("telescope_trajectory_schema_incomplete")
    if output_apt_resolution.status != "unique":
        reasons.append(f"output_apt_identity_{output_apt_resolution.status}")
    if provenance_resolution.status != "unique":
        reasons.append(f"scan_provenance_identity_{provenance_resolution.status}")
    elif not scan_registry_available(provenance):
        reasons.append("scan_registry_missing")
    reasons = sorted(set(reasons))
    core_eligible = not reasons
    timestamps_complete = bool(raw_records) and all(
        record["timestamp_available"] for record in raw_records
    )
    enhanced_eligible = bool(
        core_eligible
        and timestamps_complete
        and raw_linkage_status == "config_proven"
        and t0_status == "complete_unambiguous"
    )
    eligibility = "enhanced" if enhanced_eligible else ("core" if core_eligible else "ineligible")
    quality_score = sum(
        int(value)
        for value in (
            core_eligible,
            enhanced_eligible,
            config_path is not None,
            software["version"] is not None,
            provenance_path is not None,
            detector.get("schema_ok", False),
            telescope.get("trajectory_ok", False),
            bool(detector_networks),
            raw_linkage_status == "config_proven",
            t0_status == "complete_unambiguous",
        )
    )

    provenance_signature = {
        "obsnum": obsnum,
        "config_sha256": config_digest.sha256 if config_digest else None,
        "provenance_sha256": provenance_digest.sha256 if provenance_digest else None,
        "software_identities": software["identities"],
    }
    provenance_signature_id = "prov:" + digest_object(provenance_signature)[:20]
    # Every reduction of one observation belongs to one independence group,
    # regardless of configuration or software differences.  Those differences
    # remain visible through provenance_signature_id and must never create
    # extra statistical folds.
    duplicate_group_id = f"obs:{obsnum}"
    candidate_identity = {
        "observation_number": obsnum,
        "reduction_path": str(candidate),
        "provenance_signature_id": provenance_signature_id,
    }
    candidate_id = "map:" + digest_object(candidate_identity)[:20]
    raw_file_rows = [
        {
            "path": record["path"],
            "network": record["network"],
            "configured_interface": record["configured_interface"],
            "timestamp_available": record["timestamp_available"],
            "sha256": record["sha256"],
            "digest_status": record["digest_status"],
            "size_bytes": record["size_bytes"],
        }
        for record in raw_records
    ]
    raw_file_rows.sort(
        key=lambda row: (
            row["network"] is None,
            row["network"] if row["network"] is not None else -1,
            row["path"],
        )
    )
    return {
        "candidate_id": candidate_id,
        "map_id": candidate_id,
        "observation_number": obsnum,
        "obsnum": obsnum,
        "observation_start_utc": start_utc,
        "observation_date": observation_date,
        "session_id": session_id,
        "session_status": session_status,
        "reduction_id": run.name if REDU_RE.fullmatch(run.name) else "direct",
        "duplicate_group_id": duplicate_group_id,
        "provenance_signature_id": provenance_signature_id,
        "reduction_path": str(candidate),
        "reduction_run_path": str(run),
        "project_path": str(project),
        "source_status": source_status,
        "source_normalized": source_value,
        "source_identities": source_records,
        "product_identity_resolutions": product_resolutions,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_digest.sha256 if config_digest else None,
        "config_digest_status": config_digest.status if config_digest else "missing",
        "config_size_bytes": config_digest.size_bytes if config_digest else None,
        "software_version": software["version"],
        "software_sha": software["sha"],
        "software_identities": software["identities"],
        "provenance_path": str(provenance_path) if provenance_path else None,
        "provenance_sha256": provenance_digest.sha256 if provenance_digest else None,
        "provenance_digest_status": provenance_digest.status if provenance_digest else "missing",
        "provenance_size_bytes": provenance_digest.size_bytes if provenance_digest else None,
        "detector_tod_path": str(detector_tod) if detector_tod else None,
        "detector_tod_sha256": detector_digest.sha256 if detector_digest else None,
        "detector_tod_digest_status": detector_digest.status if detector_digest else "missing",
        "detector_tod_size_bytes": detector_digest.size_bytes if detector_digest else None,
        "telescope_path": str(telescope_path) if telescope_path else None,
        "telescope_sha256": telescope_digest.sha256 if telescope_digest else None,
        "telescope_digest_status": telescope_digest.status if telescope_digest else "missing",
        "telescope_size_bytes": telescope_digest.size_bytes if telescope_digest else None,
        "output_apt_path": str(output_apt) if output_apt else None,
        "output_apt_sha256": output_apt_digest.sha256 if output_apt_digest else None,
        "output_apt_digest_status": output_apt_digest.status if output_apt_digest else "missing",
        "output_apt_size_bytes": output_apt_digest.size_bytes if output_apt_digest else None,
        "scan_registry_available": scan_registry_available(provenance),
        "detector_networks": detector_networks,
        "detector_count_by_network": detector["detector_count_by_network"],
        "raw_files": raw_file_rows,
        "raw_networks": raw_networks,
        "missing_raw_networks": missing_raw,
        "raw_timestamp_available": timestamps_complete,
        "raw_linkage_status": raw_linkage_status,
        "raw_linkage_reasons": sorted(set(linkage_reasons)),
        "network_t0_vector": t0_vector,
        "network_t0_vector_sha256": t0_digest,
        "network_t0_status": t0_status,
        "timestamp_counter_fields": timestamp_fields,
        "timestamp_semantics": {
            "t0_definition": "Data.Toltec.Ts column 0 ClockTime integer only",
            "clock_time_nanosec_definition": "Data.Toltec.Ts column 5 retained separately",
            "common_phase_inferred": False,
            "fpga_association_status": "unproved",
            "stage_a_entry": "delivered raw D[n]/Ts[n] pair",
            "ntp_millisecond_error_hypothesis": "strongly_disfavored",
            "differential_drift_hypothesis": "strongly_disfavored",
        },
        "core_eligible": core_eligible,
        "enhanced_eligible": enhanced_eligible,
        "eligibility": eligibility,
        "exclusion_reasons": reasons,
        "canonical_quality_score": quality_score,
        "canonical_proposal": False,
        "canonical_proposal_rule": None,
        "owner_selection_required": False,
        "in_authoritative_corpus": False,
    }


def apply_duplicate_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_observation: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_observation.setdefault(int(row["observation_number"]), []).append(row)
    for candidates in by_observation.values():
        eligible = [row for row in candidates if row["core_eligible"]]
        if len(eligible) == 1:
            eligible[0]["canonical_proposal"] = True
            eligible[0]["canonical_proposal_rule"] = "sole_core_eligible_reduction"
            continue
        if not eligible:
            continue
        by_reduction: dict[str, list[dict[str, Any]]] = {}
        for row in eligible:
            by_reduction.setdefault(str(row.get("reduction_id") or ""), []).append(row)
        supported_locations = {"redu00", "redu01"}
        if (
            set(by_reduction) != supported_locations
            or any(len(values) != 1 for values in by_reduction.values())
        ):
            for row in candidates:
                row["owner_selection_required"] = True
                row["canonical_proposal_rule"] = "ambiguous_duplicate_requires_owner_review"
            continue
        redu01 = by_reduction["redu01"][0]
        redu01["canonical_proposal"] = True
        redu01["canonical_proposal_rule"] = "one_redu00_one_redu01_select_later_redu01"
        by_reduction["redu00"][0]["canonical_proposal_rule"] = (
            "one_redu00_one_redu01_retain_redu00_as_sensitivity"
        )
    return rows


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "source_identities",
        "product_identity_resolutions",
        "software_identities",
        "detector_networks",
        "detector_count_by_network",
        "raw_files",
        "raw_networks",
        "missing_raw_networks",
        "raw_linkage_reasons",
        "network_t0_vector",
        "timestamp_counter_fields",
        "timestamp_semantics",
        "exclusion_reasons",
    ):
        result[f"{key}_json"] = canonical_json(row.get(key))
    for key in (
        "scan_registry_available",
        "raw_timestamp_available",
        "core_eligible",
        "enhanced_eligible",
        "canonical_proposal",
        "owner_selection_required",
        "in_authoritative_corpus",
    ):
        result[key] = bool_text(bool(row.get(key)))
    return result


def duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_observation: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_observation.setdefault(int(row["observation_number"]), []).append(row)
    result = []
    for obsnum, candidates in sorted(by_observation.items()):
        if len(candidates) < 2:
            continue
        for row in sorted(candidates, key=lambda item: item["candidate_id"]):
            result.append(
                {
                    "observation_number": obsnum,
                    "candidate_count": len(candidates),
                    "candidate_id": row["candidate_id"],
                    "duplicate_group_id": row["duplicate_group_id"],
                    "provenance_signature_id": row["provenance_signature_id"],
                    "config_sha256": row["config_sha256"],
                    "provenance_sha256": row["provenance_sha256"],
                    "software_version": row["software_version"],
                    "eligibility": row["eligibility"],
                    "canonical_quality_score": row["canonical_quality_score"],
                    "canonical_proposal": bool_text(row["canonical_proposal"]),
                    "canonical_proposal_rule": row["canonical_proposal_rule"],
                    "owner_selection_required": bool_text(row["owner_selection_required"]),
                    "reduction_path": row["reduction_path"],
                }
            )
    return result


def exclusion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "observation_number": row["observation_number"],
            "eligibility": row["eligibility"],
            "reason": reason,
            "reduction_path": row["reduction_path"],
        }
        for row in rows
        for reason in row["exclusion_reasons"]
    ]


def authoritative_obsnum_status(
    obsnums: Sequence[int], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_obsnum: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_obsnum.setdefault(int(row["observation_number"]), []).append(row)
    result: list[dict[str, Any]] = []
    for obsnum in obsnums:
        candidates = sorted(by_obsnum.get(int(obsnum), []), key=lambda item: str(item["candidate_id"]))
        eligible = [row for row in candidates if bool(row.get("core_eligible"))]
        canonical = [row for row in candidates if bool(row.get("canonical_proposal"))]
        if not candidates:
            status = "no_retained_reduction_found"
        elif not eligible:
            status = "retained_reduction_found_no_eligible_candidate"
        elif any(bool(row.get("owner_selection_required")) for row in candidates):
            status = "ambiguous_duplicate_requires_owner_review"
        elif len(canonical) == 1:
            status = "eligible_canonical_candidate_found"
        else:
            status = "eligible_candidate_without_canonical_resolution"
        result.append({
            "observation_number": int(obsnum),
            "candidate_count": len(candidates),
            "core_eligible_candidate_count": len(eligible),
            "canonical_candidate_id": canonical[0]["candidate_id"] if len(canonical) == 1 else "",
            "status": status,
            "candidate_ids_json": canonical_json([row["candidate_id"] for row in candidates]),
            "eligibility_reasons_json": canonical_json([
                {"candidate_id": row["candidate_id"], "reasons": row.get("exclusion_reasons", [])}
                for row in candidates
            ]),
        })
    return result


def network_availability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = tuple(range(13))
    result: list[dict[str, Any]] = []
    for row in rows:
        present = {int(value) for value in row.get("detector_networks", [])}
        metadata_available = bool(row.get("detector_tod_path"))
        for network in expected:
            if network == 10:
                kind = "structural_absence_nw10"
                status = "not_a_network"
            elif not metadata_available:
                kind = "metadata_unavailable"
                status = "detector_network_metadata_unreadable_or_absent"
            elif network in present:
                kind = "present"
                status = "present"
            elif network == 6:
                kind = "known_intermittent_absence_nw6"
                status = "not_exclusionary"
            else:
                kind = "unexpected_per_observation_absence"
                status = "record_only_not_automatic_observation_exclusion"
            result.append({
                "observation_number": int(row["observation_number"]),
                "candidate_id": row["candidate_id"],
                "reduction_id": row["reduction_id"],
                "core_eligible": bool_text(bool(row.get("core_eligible"))),
                "network_id": network,
                "availability_kind": kind,
                "availability_status": status,
                "detector_metadata_available": bool_text(metadata_available),
                "timing_result_used_as_cut": "false",
            })
    return result


def candidate_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SCI-ALIGN-001 3C273 candidate inventory",
        "",
        "| Obs | Candidate | Reduction | Source | Core | Enhanced | Networks | Canonical | Owner choice |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {obs} | `{cid}` | `{redu}` | {source} | {core} | {enhanced} | "
            "{networks} | {canonical} | {owner} |".format(
                obs=row["observation_number"],
                cid=row["candidate_id"],
                redu=row["reduction_id"],
                source=row["source_status"],
                core=bool_text(row["core_eligible"]),
                enhanced=bool_text(row["enhanced_eligible"]),
                networks=",".join(str(value) for value in row["detector_networks"]),
                canonical=bool_text(row["canonical_proposal"]),
                owner=bool_text(row["owner_selection_required"]),
            )
        )
    return "\n".join(lines) + "\n"


def selection_template(rows: list[dict[str, Any]], inventory_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_SCHEMA,
        "source_inventory_sha256": inventory_sha256,
        "selection_rule": (
            "The provenance-only canonical rule selects the sole eligible reduction, "
            "or redu01 when exactly one redu00 and one redu01 are eligible. Other "
            "eligible duplicates remain sensitivity rows. Ambiguous duplicates cannot "
            "be frozen; timing outcomes must not be consulted."
        ),
        "rows": [
            {
                "candidate_id": row["candidate_id"],
                "observation_number": row["observation_number"],
                "selected": bool(row["canonical_proposal"] and not row["owner_selection_required"]),
                "owner_note": "",
            }
            for row in rows
        ],
    }


def read_selection(path: Path) -> tuple[str, list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise InventoryError("owner selection must use a .csv or .json suffix")
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise InventoryError("selection JSON must be an object")
        if value.get("schema_version") != SELECTION_SCHEMA:
            raise InventoryError("unsupported selection JSON schema")
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise InventoryError("selection JSON rows must be a list")
        digest = value.get("source_inventory_sha256")
        if not isinstance(digest, str) or not digest:
            raise InventoryError("selection JSON has no source_inventory_sha256")
        return digest, rows
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise InventoryError("selection CSV has no candidate rows")
    digests = {
        str(row.get("source_inventory_sha256") or "").strip() for row in rows
    }
    if "" in digests:
        raise InventoryError("every selection CSV row must preserve source_inventory_sha256")
    if len(digests) != 1:
        raise InventoryError("selection CSV contains conflicting inventory digests")
    return next(iter(digests)), rows


def freeze_selection(
    inventory_rows: list[dict[str, Any]],
    inventory_sha256: str,
    selection_path: Path,
    *,
    obsnum_allowlist_sha256: str,
    obsnum_allowlist_schema_version: str,
    obsnum_allowlist_filename: str,
) -> dict[str, Any]:
    selected_digest, selection_rows = read_selection(selection_path)
    if selected_digest != inventory_sha256:
        raise InventoryError("selection was prepared from a different candidate inventory")
    by_id = {row["candidate_id"]: row for row in inventory_rows}
    unresolved = sorted(
        int(row["observation_number"])
        for row in inventory_rows
        if row.get("owner_selection_required")
    )
    if unresolved:
        raise InventoryError(
            "ambiguous duplicate provenance prevents selection freeze; review observation(s): "
            + ", ".join(str(value) for value in sorted(set(unresolved)))
        )
    selected: list[dict[str, Any]] = []
    seen_selection_ids: set[str] = set()
    for index, selection in enumerate(selection_rows):
        candidate_id = str(selection.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_selection_ids:
            raise InventoryError(f"selection row {index}: missing or duplicate candidate_id")
        seen_selection_ids.add(candidate_id)
        if candidate_id not in by_id:
            raise InventoryError(f"selection row {index}: unknown candidate_id {candidate_id}")
        try:
            selected_obsnum = int(selection.get("observation_number"))
        except (TypeError, ValueError) as error:
            raise InventoryError(
                f"selection row {index}: invalid observation_number"
            ) from error
        if selected_obsnum != int(by_id[candidate_id]["observation_number"]):
            raise InventoryError(
                f"selection row {index}: candidate/observation identity mismatch"
            )
        if parse_bool(selection.get("selected", False), f"selection row {index}"):
            candidate = by_id[candidate_id]
            if not candidate["core_eligible"]:
                raise InventoryError(f"selected candidate is not core eligible: {candidate_id}")
            selected.append(candidate)
    missing_selection_ids = sorted(set(by_id) - seen_selection_ids)
    if missing_selection_ids:
        raise InventoryError(
            "selection must preserve every inventory candidate; missing candidate_id(s): "
            + ", ".join(missing_selection_ids)
        )
    primary_obsnums = [int(row["observation_number"]) for row in selected]
    if len(primary_obsnums) != len(set(primary_obsnums)):
        raise InventoryError("selection contains more than one reduction for an observation")
    eligible_observations = {
        int(row["observation_number"])
        for row in inventory_rows
        if bool(row.get("core_eligible"))
    }
    missing_primary = sorted(eligible_observations - set(primary_obsnums))
    if missing_primary:
        raise InventoryError(
            "every observation with a core-eligible candidate requires exactly one "
            "primary owner selection; missing observation(s): "
            + ", ".join(str(value) for value in missing_primary)
        )
    primary_ids = {str(row["candidate_id"]) for row in selected}
    canonical_primary_ids = {
        str(row["candidate_id"])
        for row in inventory_rows
        if bool(row.get("core_eligible")) and bool(row.get("canonical_proposal"))
    }
    if primary_ids != canonical_primary_ids:
        raise InventoryError(
            "selection must preserve the frozen canonical-reduction rule; "
            "do not substitute timing-informed primary candidates"
        )
    analysis_rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        if not bool(row.get("core_eligible")):
            continue
        if int(row["observation_number"]) not in set(primary_obsnums):
            continue
        realized = dict(row)
        realized["analysis_role"] = (
            "primary" if str(row["candidate_id"]) in primary_ids else "sensitivity"
        )
        analysis_rows.append(realized)
    analysis_rows.sort(
        key=lambda row: (
            int(row["observation_number"]),
            0 if row["analysis_role"] == "primary" else 1,
            str(row["candidate_id"]),
        )
    )
    manifest_rows = [
        {key: row.get(key) for key in SELECTED_FIELDS if not key.endswith("_json")}
        for row in analysis_rows
    ]
    for output, original in zip(manifest_rows, analysis_rows):
        for key in (
            "detector_networks",
            "raw_files",
            "raw_networks",
            "missing_raw_networks",
            "network_t0_vector",
            "timestamp_counter_fields",
            "timestamp_semantics",
        ):
            output[key] = original[key]
    base = {
        "schema_version": SELECTED_MANIFEST_SCHEMA,
        "source_inventory_sha256": inventory_sha256,
        "owner_selection_sha256": sha256_file(selection_path),
        "owner_selection_format": selection_path.suffix.lower().removeprefix("."),
        "obsnum_allowlist_sha256": obsnum_allowlist_sha256,
        "obsnum_allowlist_schema_version": obsnum_allowlist_schema_version,
        "obsnum_allowlist_filename": obsnum_allowlist_filename,
        "rows": manifest_rows,
    }
    return {**base, "manifest_sha256": digest_object(base)}


def load_frozen_inventory(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read source inventory {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != INVENTORY_SCHEMA:
        raise InventoryError("unsupported source inventory schema")
    recorded = document.get("inventory_sha256")
    if not isinstance(recorded, str):
        raise InventoryError("source inventory has no inventory_sha256")
    base = {key: value for key, value in document.items() if key != "inventory_sha256"}
    measured = digest_object(base)
    if measured != recorded:
        raise InventoryError(
            f"source inventory digest mismatch: recorded={recorded} measured={measured}"
        )
    if not isinstance(document.get("rows"), list):
        raise InventoryError("source inventory rows must be a list")
    return document


def selected_csv_rows(selected: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in selected["rows"]:
        serialized = dict(row)
        for key in (
            "detector_networks", "raw_files", "raw_networks",
            "missing_raw_networks", "network_t0_vector",
            "timestamp_counter_fields", "timestamp_semantics",
        ):
            serialized[f"{key}_json"] = canonical_json(row.get(key))
        for key in (
            "raw_timestamp_available", "core_eligible", "enhanced_eligible"
        ):
            serialized[key] = bool_text(bool(row.get(key)))
        result.append(serialized)
    return result


def emit_selected_manifest(
    output: Path, selected: dict[str, Any], selection_path: Path,
    allowlist_path: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selected_manifest.json", selected)
    write_csv(
        output / "selected_manifest.csv",
        selected_csv_rows(selected),
        SELECTED_FIELDS,
    )
    suffix = selection_path.suffix.lower()
    selection_copy = output / f"owner_selection{suffix}"
    if selection_copy.resolve() != selection_path.resolve():
        shutil.copyfile(selection_path, selection_copy)
    allowlist_copy = output / allowlist_path.name
    if allowlist_copy.resolve() != allowlist_path.resolve():
        shutil.copyfile(allowlist_path, allowlist_copy)
    write_sha256sums(output)


def inventory(
    reduction_roots: Sequence[Path],
    raw_roots: Sequence[Path],
    *,
    output: Path,
    source_regex: str,
    obsnum_allowlist: Path,
    excluded_paths: Sequence[Path] = (),
    threshold: int = DEFAULT_LARGE_FILE_THRESHOLD,
    hash_large: bool = False,
) -> tuple[dict[str, Any], DigestCache]:
    reductions, raws, destination, exclusions = validate_roots(
        reduction_roots, raw_roots, output, excluded_paths
    )
    allowlist_document, allowed_obsnums, allowlist_sha256 = load_obsnum_allowlist(
        resolved(obsnum_allowlist)
    )
    try:
        pattern = re.compile(source_regex)
    except re.error as error:
        raise InventoryError(f"invalid --source-regex: {error}") from error
    cache = DigestCache(
        destination / "digest_cache.json", threshold=threshold, hash_large=hash_large
    )
    raw_index = raw_file_index(raws)
    discovered = [
        candidate_inventory(
            candidate,
            source_pattern=pattern,
            raw_by_obsnum=raw_index,
            digest_cache=cache,
        )
        for candidate in discover_candidate_dirs(reductions, exclusions)
    ]
    allowed = set(allowed_obsnums)
    rows = [row for row in discovered if int(row["observation_number"]) in allowed]
    for row in rows:
        row["in_authoritative_corpus"] = True
    rows = apply_duplicate_policy(
        sorted(rows, key=lambda row: (row["observation_number"], row["reduction_path"]))
    )
    out_of_scope = sorted(
        [
            row for row in discovered
            if int(row["observation_number"]) not in allowed
            and row.get("source_status") in {"accepted", "target"}
        ],
        key=lambda row: (int(row["observation_number"]), str(row["reduction_path"])),
    )
    status_rows = authoritative_obsnum_status(allowed_obsnums, rows)
    inventory_base = {
        "schema_version": INVENTORY_SCHEMA,
        "source_regex": source_regex,
        "reduction_roots": [str(path) for path in reductions],
        "raw_roots": [str(path) for path in raws],
        "excluded_paths": [str(path) for path in exclusions],
        "obsnum_allowlist": {
            "path": str(resolved(obsnum_allowlist)),
            "filename": resolved(obsnum_allowlist).name,
            "sha256": allowlist_sha256,
            "schema_version": allowlist_document["schema_version"],
            "corpus_id": allowlist_document["corpus_id"],
            "obsnums": allowed_obsnums,
        },
        "eligibility_policy": {
            "timing_results_used": False,
            "core": [
                "unambiguous target source identity",
                "consistent observation identity",
                "identified config",
                "detector TOD required variables and network identities",
                "telescope trajectory and Hold",
                "unique retained output APT",
                "nonempty scan registry provenance",
            ],
            "enhanced": [
                "core eligible",
                "configured raw file linkage for every detector network",
                "Data.Toltec.Ts has six producer-defined columns",
                "complete unambiguous per-network integer T0 vector",
            ],
            "t0_session_rule": "Data.Toltec.Ts column 0 only; no common phase inferred",
        },
        "rows": rows,
        "authoritative_obsnum_status": status_rows,
        "out_of_scope_3c273_rows": out_of_scope,
    }
    return {
        **inventory_base,
        "inventory_sha256": digest_object(inventory_base),
    }, cache


def next_commands(args: argparse.Namespace, output: Path) -> list[str]:
    base = [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
    ]
    for root in args.reduction_root:
        base += ["--reduction-root", str(resolved(root))]
    for root in args.raw_root:
        base += ["--raw-root", str(resolved(root))]
    for path in args.exclude_path:
        base += ["--exclude-path", str(resolved(path))]
    base += [
        "--obsnum-allowlist", str(resolved(args.obsnum_allowlist)),
        "--output", str(output), "--source-regex", args.source_regex,
    ]
    inventory_command = " ".join(json.dumps(part) for part in base)
    owner_selection = output / "owner_selection.csv"
    copy_selection_command = " ".join(
        json.dumps(part)
        for part in (
            "cp",
            str(output / "selection_template.csv"),
            str(owner_selection),
        )
    )
    freeze_command = " ".join(
        json.dumps(part)
        for part in (
            str(Path(sys.executable)),
            str(Path(__file__).resolve()),
            "--inventory",
            str(output / "candidate_inventory.json"),
            "--freeze-selection",
            str(owner_selection),
            "--output",
            str(output / "frozen"),
        )
    )
    return [
        inventory_command,
        f"Review {output / 'candidate_table.md'} and {output / 'duplicate_reduction_registry.csv'}",
        copy_selection_command,
        f"emacs -nw {owner_selection}  # edit only owner_note; selection is canonical and timing-blind",
        freeze_command,
    ]


def emit(
    document: dict[str, Any],
    cache: DigestCache,
    output: Path,
    *,
    commands: list[str],
    obsnum_allowlist: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = document["rows"]
    write_json(output / "candidate_inventory.json", document)
    write_csv(output / "candidate_inventory.csv", [csv_row(row) for row in rows], INVENTORY_FIELDS)
    duplicates = duplicate_rows(rows)
    write_csv(
        output / "duplicate_reduction_registry.csv",
        duplicates,
        [
            "observation_number", "candidate_count", "candidate_id",
            "duplicate_group_id", "provenance_signature_id", "config_sha256", "provenance_sha256",
            "software_version", "eligibility", "canonical_quality_score",
            "canonical_proposal", "canonical_proposal_rule", "owner_selection_required", "reduction_path",
        ],
    )
    write_csv(
        output / "exclusion_registry.csv",
        exclusion_rows(rows),
        ["candidate_id", "observation_number", "eligibility", "reason", "reduction_path"],
    )
    write_csv(
        output / "authoritative_obsnum_status.csv",
        document["authoritative_obsnum_status"],
        [
            "observation_number", "candidate_count", "core_eligible_candidate_count",
            "canonical_candidate_id", "status", "candidate_ids_json",
            "eligibility_reasons_json",
        ],
    )
    write_csv(
        output / "network_availability.csv",
        network_availability_rows(rows),
        [
            "observation_number", "candidate_id", "reduction_id", "core_eligible",
            "network_id", "availability_kind", "availability_status",
            "detector_metadata_available", "timing_result_used_as_cut",
        ],
    )
    write_csv(
        output / "out_of_scope_3c273_discovery.csv",
        [csv_row(row) for row in document["out_of_scope_3c273_rows"]],
        INVENTORY_FIELDS,
    )
    (output / "candidate_table.md").write_text(candidate_table(rows), encoding="utf-8")
    template = selection_template(rows, document["inventory_sha256"])
    write_json(output / "selection_template.json", template)
    write_csv(
        output / "selection_template.csv",
        [
            {
                "source_inventory_sha256": document["inventory_sha256"],
                **row,
                "selected": bool_text(row["selected"]),
            }
            for row in template["rows"]
        ],
        ["source_inventory_sha256", "candidate_id", "observation_number", "selected", "owner_note"],
    )
    write_json(output / "next_commands.json", {"commands": commands})
    (output / "next_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_json(output / "digest_cache.json", cache.document())
    shutil.copyfile(resolved(obsnum_allowlist), output / resolved(obsnum_allowlist).name)
    write_sha256sums(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction-root", action="append", type=Path, default=[])
    parser.add_argument("--raw-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--inventory", type=Path,
        help="Freeze-only mode: previously emitted candidate_inventory.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-regex", default=DEFAULT_SOURCE_REGEX)
    parser.add_argument("--obsnum-allowlist", type=Path)
    parser.add_argument(
        "--exclude-path", action="append", type=Path, default=[],
        help="A discovery subtree, such as the owner run directory, excluded from candidates.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze-selection", type=Path)
    parser.add_argument("--hash-large", action="store_true")
    parser.add_argument(
        "--large-file-threshold-mib", type=float, default=64.0,
        help="Files larger than this are recorded but not hashed unless --hash-large.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.large_file_threshold_mib <= 0:
        raise InventoryError("--large-file-threshold-mib must be positive")
    output = resolved(args.output)
    if args.inventory is not None:
        if args.reduction_root or args.raw_root:
            raise InventoryError(
                "--inventory freeze-only mode cannot be combined with source roots"
            )
        if args.freeze_selection is None:
            raise InventoryError(
                "--inventory requires --freeze-selection in freeze-only mode"
            )
        document = load_frozen_inventory(resolved(args.inventory))
        recorded_exclusions = [Path(str(path)) for path in document.get("excluded_paths", [])]
        for raw_root in document.get("raw_roots", []):
            if paths_overlap(Path(str(raw_root)), output):
                raise InventoryError(
                    f"output directory overlaps source root recorded by inventory: {raw_root}"
                )
        for reduction_root in document.get("reduction_roots", []):
            root = Path(str(reduction_root))
            if paths_overlap(root, output) and not any(
                output.is_relative_to(path) for path in recorded_exclusions
            ):
                raise InventoryError(
                    f"output directory overlaps source root recorded by inventory: {reduction_root}"
                )
        allowlist = document.get("obsnum_allowlist")
        if not isinstance(allowlist, dict):
            raise InventoryError("inventory lacks checksum-bound obsnum allowlist")
        allowlist_path = resolved(args.inventory).parent / str(allowlist.get("filename") or "")
        if not allowlist_path.is_file() or sha256_file(allowlist_path) != allowlist.get("sha256"):
            raise InventoryError("inventory obsnum allowlist copy is missing or has a digest mismatch")
        selected = freeze_selection(
            document["rows"],
            document["inventory_sha256"],
            resolved(args.freeze_selection),
            obsnum_allowlist_sha256=str(allowlist["sha256"]),
            obsnum_allowlist_schema_version=str(allowlist["schema_version"]),
            obsnum_allowlist_filename=str(allowlist["filename"]),
        )
        if args.dry_run:
            print(
                f"selection valid: rows={len(selected['rows'])} "
                f"manifest_sha256={selected['manifest_sha256']}"
            )
            return 0
        emit_selected_manifest(
            output, selected, resolved(args.freeze_selection), allowlist_path
        )
        print(
            f"selection frozen: rows={len(selected['rows'])} "
            f"output={output} sha256={selected['manifest_sha256']}"
        )
        return 0
    if not args.reduction_root:
        raise InventoryError(
            "discovery mode requires at least one --reduction-root; "
            "freeze mode uses --inventory and --freeze-selection"
        )
    if args.freeze_selection is not None:
        raise InventoryError(
            "--freeze-selection requires freeze-only --inventory mode"
        )
    if args.obsnum_allowlist is None:
        raise InventoryError("discovery mode requires --obsnum-allowlist")
    document, cache = inventory(
        args.reduction_root,
        args.raw_root,
        output=output,
        source_regex=args.source_regex,
        obsnum_allowlist=args.obsnum_allowlist,
        excluded_paths=args.exclude_path,
        threshold=int(args.large_file_threshold_mib * 1024 * 1024),
        hash_large=args.hash_large,
    )
    commands = next_commands(args, output)
    if args.dry_run:
        print(candidate_table(document["rows"]), end="")
        print("Next commands:")
        for command in commands:
            print(f"  {command}")
        return 0
    emit(
        document,
        cache,
        output,
        commands=commands,
        obsnum_allowlist=args.obsnum_allowlist,
    )
    print(
        f"inventory complete: candidates={len(document['rows'])} "
        f"output={output} sha256={document['inventory_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        raise SystemExit(2)
