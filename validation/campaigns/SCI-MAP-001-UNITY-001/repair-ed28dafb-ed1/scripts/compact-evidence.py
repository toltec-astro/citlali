#!/usr/bin/env python3
"""Bounded compact-evidence producer for SCI-MAP-001-UNITY-001.

The tool is deliberately local and file-only.  It cannot contact Unity, run
Citlali, submit work, or remove retained capture products.  It streams the
unchanged candidate's full/all processed-TOD NetCDF directly; a normalized NPZ
adapter exists only for bounded local contract fixtures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np


CANDIDATE_SHA = "ed28dafb37f9113c0d3c95297148157129a90886"
CAMPAIGN_REVISION = "repair-sha-ed28dafb-ed1-2026-08-02"
REQUEST_ID = "SCI-MAP-001-UNITY-001"
RESOURCE_SCHEMA = "sci-map-001-resource-record-v1"
RESOURCE_INVENTORY_SCHEMA = "sci-map-001-resource-inventory-v1"
RESOURCE_CEILING_BYTES = 214748364800
RESOURCE_MAX_AGE_SECONDS = 900
RESOURCE_MAX_FUTURE_SKEW_SECONDS = 60
STREAM_SCHEMA = "sci-map-001-producer-stream-v1"
GROUP_SCHEMA = "sci-map-001-compact-evidence-group-v1"
TRACE_SCHEMA = "sci-map-001-deterministic-trace-selection-v1"
REQUEST_SCHEMA = "sci-map-001-discrepancy-request-v1"
EXPANSION_PLAN_SCHEMA = "sci-map-001-discrepancy-expansion-plan-v1"
EXPANSION_SCHEMA = "sci-map-001-discrepancy-expansion-v1"
ADAPTER = "full-ptc-normalized-npz-v1"
NETCDF_ADAPTER = "candidate-full-ptc-netcdf-v1"
TERM_ORDER = "scan-major-detector-major-sample-minor-cartesian-v1"
TRACE_POLICY = "first-lower-middle-last-every-network-sha256-min-valid-flagged-v1"
REALIZATIONS = 64
MAX_EXPANSION_TERMS = 1_000_000
ARRAY_NETWORKS = {
    "a1100": set(range(0, 7)),
    "a1400": set(range(7, 11)),
    "a2000": set(range(11, 13)),
}
REQUIRED_GROUP_KEYS = tuple(
    f"{obs}:{array}"
    for obs in (152389, 152390, 152392)
    for array in ("a1100", "a1400", "a2000")
)
TERM_INT64 = (
    "row", "col", "scan_index", "detector_index", "sample_index", "network",
)
TERM_UINT8 = (
    "geometric_in_bounds", "detector_apt_flagged", "sample_flagged",
    "upstream_eligible",
)
TERM_FLOAT64 = ("coefficient", "sample_signal", "sample_kernel")
SOURCE_MEMBERS = (
    "metadata_json", *TERM_INT64, *TERM_UINT8, *TERM_FLOAT64,
    "realization_signs",
)
STATS_MEMBERS = {
    "signal_numerator": np.dtype("float64"),
    "weight": np.dtype("float64"),
    "kernel_numerator": np.dtype("float64"),
    "upstream_eligible_exposure": np.dtype("float64"),
    "retained_exposure": np.dtype("float64"),
    "geometric_hits": np.dtype("int64"),
    "contributing_hits": np.dtype("int64"),
    "realization_signs": np.dtype("int8"),
}
TRACE_MEMBERS = {
    "sequence_offsets": np.dtype("int64"),
    "row": np.dtype("int64"),
    "col": np.dtype("int64"),
    "geometric_in_bounds": np.dtype("uint8"),
    "detector_apt_flagged": np.dtype("uint8"),
    "sample_flagged": np.dtype("uint8"),
    "upstream_eligible": np.dtype("uint8"),
    "coefficient": np.dtype("float64"),
    "sample_signal": np.dtype("float64"),
    "sample_kernel": np.dtype("float64"),
    "sample_interval_s": np.dtype("float64"),
}
GROUP_KEYS = {
    "schema_version", "candidate_sha", "campaign_revision",
    "raw_input_manifest_sha256", "producer_identity", "source_stream_sha256",
    "realized_raw_timestream_provenance_sha256",
    "realized_mapmaking_provenance_sha256",
    "mapmaking_bundle_identity_digest", "obsnum", "array", "map_shape",
    "rates", "scan_order", "active_networks", "population",
    "stream_digests", "artifacts", "trace_selection",
}


class EvidenceError(RuntimeError):
    """A fail-closed evidence-contract violation."""


def die(message: str) -> NoReturn:
    raise EvidenceError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceError(f"cannot parse JSON {path}: {exc}") from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write deterministic, uncompressed NPZ bytes with fixed ZIP metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            value = np.ascontiguousarray(arrays[name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, buffer.getvalue())


def exact_float_node(value: float, authority: str) -> dict[str, str]:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        die(f"{authority} must be finite and positive")
    return {
        "decimal": format(number, ".17g"),
        "hex": number.hex(),
        "authority": authority,
    }


def parse_exact_float(node: Any, authority: str) -> float:
    if not isinstance(node, Mapping) or set(node) != {"decimal", "hex", "authority"}:
        die(f"{authority} rate node is incomplete or has extra fields")
    if node["authority"] != authority:
        die(f"rate authority differs: {node['authority']!r} != {authority!r}")
    try:
        decimal = float(node["decimal"])
        hexadecimal = float.fromhex(node["hex"])
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{authority} rate encoding is invalid: {exc}") from exc
    if not math.isfinite(decimal) or decimal <= 0.0 or \
            np.float64(decimal).view(np.uint64) != np.float64(hexadecimal).view(np.uint64):
        die(f"{authority} decimal/hex values are not the same positive binary64")
    return decimal


def boost_mt19937_scan_signs(scan_count: int) -> np.ndarray:
    if scan_count <= 0:
        die("realization signs require at least one scan")
    engine = np.random.RandomState(5489)
    words = engine.randint(0, 2 ** 32, size=scan_count * REALIZATIONS,
                           dtype=np.uint32)
    bits = (words >> np.uint32(31)).astype(np.int8)
    return (2 * bits - 1).reshape(scan_count, REALIZATIONS)


def _require_hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or \
            any(c not in "0123456789abcdef" for c in value):
        die(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        die(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        die(f"{label} keys differ; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}")
    return value


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    keys = {
        "schema_version", "adapter", "capture_output_mode",
        "capture_detector_selection", "candidate_sha", "campaign_revision",
        "raw_input_manifest_sha256", "producer_identity", "capture_ptc_sha256",
        "realized_raw_timestream_provenance_sha256",
        "realized_mapmaking_provenance_sha256",
        "mapmaking_bundle_identity_digest", "obsnum", "array", "map_shape",
        "map_pixel_size_rad",
        "native_fsmp_hz", "effective_d_fsmp_hz", "scan_order",
        "detector_order", "primitive_term_count", "term_order",
    }
    node = dict(_require_exact_keys(metadata, keys, "producer stream metadata"))
    exact = {
        "schema_version": STREAM_SCHEMA,
        "capture_output_mode": "full",
        "capture_detector_selection": "all",
        "candidate_sha": CANDIDATE_SHA,
        "campaign_revision": CAMPAIGN_REVISION,
        "term_order": TERM_ORDER,
    }
    for name, wanted in exact.items():
        if node[name] != wanted:
            die(f"producer stream {name} differs: {node[name]!r} != {wanted!r}")
    if node["adapter"] not in (ADAPTER, NETCDF_ADAPTER):
        die("producer stream adapter is not an approved full-PTC adapter")
    _require_hex_digest(node["raw_input_manifest_sha256"],
                        "raw_input_manifest_sha256")
    _require_hex_digest(node["capture_ptc_sha256"], "capture_ptc_sha256")
    _require_hex_digest(node["realized_raw_timestream_provenance_sha256"],
                        "realized_raw_timestream_provenance_sha256")
    _require_hex_digest(node["realized_mapmaking_provenance_sha256"],
                        "realized_mapmaking_provenance_sha256")
    if not isinstance(node["producer_identity"], str) or \
            not (1 <= len(node["producer_identity"]) <= 512):
        die("producer_identity is empty or too long")
    bundle = node["mapmaking_bundle_identity_digest"]
    prefix = "canonical-hexfloat-sha256-v1:"
    if not isinstance(bundle, str) or not bundle.startswith(prefix):
        die("mapmaking_bundle_identity_digest has the wrong domain")
    _require_hex_digest(bundle[len(prefix):], "mapmaking bundle digest")
    if node["obsnum"] not in (152389, 152390, 152392):
        die("producer stream observation is outside the campaign")
    if node["array"] not in ARRAY_NETWORKS:
        die("producer stream array is outside the campaign")
    shape = _require_exact_keys(node["map_shape"], {"rows", "cols"}, "map_shape")
    if any(not isinstance(shape[name], int) or isinstance(shape[name], bool) or
           shape[name] <= 0 for name in ("rows", "cols")):
        die("map_shape rows and cols must be positive integers")
    native = parse_exact_float(node["native_fsmp_hz"], "telescope.fsmp")
    effective = parse_exact_float(node["effective_d_fsmp_hz"], "telescope.d_fsmp")
    pixel_size = parse_exact_float(
        node["map_pixel_size_rad"],
        "realized_mapmaking.effective_pixel_size_rad")
    node["_native_fsmp"] = native
    node["_effective_d_fsmp"] = effective
    node["_map_pixel_size_rad"] = pixel_size
    node["_sample_interval"] = float(np.float64(1.0) / np.float64(effective))
    if not math.isfinite(node["_sample_interval"]) or node["_sample_interval"] <= 0.0:
        die("effective sample interval is not finite and positive")

    scans = node["scan_order"]
    if not isinstance(scans, list) or not scans:
        die("scan_order must be a nonempty array")
    scan_ids: set[str] = set()
    for index, scan in enumerate(scans):
        scan = _require_exact_keys(
            scan, {"scan_index", "scan_identity", "output_scan_index", "sample_count"},
            f"scan_order[{index}]")
        if scan["scan_index"] != index:
            die("scan_order indices must be contiguous and zero-based")
        if not isinstance(scan["scan_identity"], str) or not scan["scan_identity"] \
                or scan["scan_identity"] in scan_ids:
            die("scan identities must be nonempty and unique")
        scan_ids.add(scan["scan_identity"])
        if not isinstance(scan["output_scan_index"], int) or \
                isinstance(scan["output_scan_index"], bool) or \
                scan["output_scan_index"] != index + 1:
            die("all-scan output_scan_index must be contiguous and one-based")
        if not isinstance(scan["sample_count"], int) or \
                isinstance(scan["sample_count"], bool) or scan["sample_count"] <= 0:
            die("scan sample_count must be a positive integer")

    detectors = node["detector_order"]
    if not isinstance(detectors, list) or not detectors:
        die("detector_order must be a nonempty array")
    detector_identities: set[str] = set()
    for index, detector in enumerate(detectors):
        detector = _require_exact_keys(
            detector,
            {"detector_index", "apt_row_index", "network", "kids_tone",
             "detector_uid", "detector_identity", "apt_flagged"},
            f"detector_order[{index}]")
        if detector["detector_index"] != index:
            die("detector_order indices must be contiguous and zero-based")
        uid = detector["detector_uid"]
        if not isinstance(uid, str) or not uid or \
                (uid.startswith("-") and not uid[1:].isdigit()) or \
                (not uid.startswith("-") and not uid.isdigit()) or \
                str(int(uid)) != uid:
            die("detector_uid must be a canonical base-10 integral string")
        if not isinstance(detector["apt_row_index"], int) or \
                isinstance(detector["apt_row_index"], bool) or \
                detector["apt_row_index"] < 0:
            die("apt_row_index must be a nonnegative integer")
        if not isinstance(detector["kids_tone"], int) or \
                isinstance(detector["kids_tone"], bool) or \
                detector["kids_tone"] < 0:
            die("kids_tone must be a nonnegative integer")
        if detector["network"] not in ARRAY_NETWORKS[node["array"]]:
            die("detector network does not belong to the declared array")
        expected_identity = (
            f"nw={detector['network']};kids_tone={detector['kids_tone']};"
            f"uid={uid};apt_row_index={detector['apt_row_index']}")
        if detector["detector_identity"] != expected_identity or \
                expected_identity in detector_identities:
            die("detector composite identity is malformed or repeated")
        detector_identities.add(expected_identity)
        if not isinstance(detector["apt_flagged"], bool):
            die("detector apt_flagged must be Boolean")
    expected_terms = sum(int(s["sample_count"]) for s in scans) * len(detectors)
    if not isinstance(node["primitive_term_count"], int) or \
            isinstance(node["primitive_term_count"], bool) or \
            node["primitive_term_count"] != expected_terms:
        die(f"primitive_term_count differs from Cartesian membership: "
            f"{node['primitive_term_count']!r} != {expected_terms}")
    return node


@dataclass
class PrimitiveChunk:
    start: int
    stop: int
    arrays: dict[str, np.ndarray]


class FullPtcFixtureAdapter:
    """Normalized full/all PTC primitive adapter for local contract fixtures."""

    def __init__(self, path: Path):
        raw_path = Path(path)
        if raw_path.is_symlink() or not raw_path.is_file():
            die(f"normalized full PTC source is not a regular file: {path}")
        self.path = raw_path.resolve()
        try:
            self.archive = np.load(self.path, allow_pickle=False)
        except Exception as exc:
            raise EvidenceError(f"cannot load normalized full PTC source {path}: {exc}") from exc
        if set(self.archive.files) != set(SOURCE_MEMBERS):
            die(f"normalized full PTC members differ; "
                f"missing={sorted(set(SOURCE_MEMBERS)-set(self.archive.files))}, "
                f"extra={sorted(set(self.archive.files)-set(SOURCE_MEMBERS))}")
        raw_metadata = np.asarray(self.archive["metadata_json"])
        if raw_metadata.size != 1 or raw_metadata.dtype.kind not in "SU":
            die("metadata_json must be one string scalar")
        try:
            metadata = json.loads(str(raw_metadata.reshape(-1)[0]))
        except Exception as exc:
            raise EvidenceError(f"metadata_json is invalid: {exc}") from exc
        self.metadata = _validate_metadata(metadata)
        self.term_count = int(self.metadata["primitive_term_count"])
        for name in TERM_INT64:
            self._validate_term_array(name, np.dtype("int64"))
        for name in TERM_UINT8:
            self._validate_term_array(name, np.dtype("uint8"))
        for name in TERM_FLOAT64:
            self._validate_term_array(name, np.dtype("float64"))
        signs = np.asarray(self.archive["realization_signs"])
        expected_shape = (len(self.metadata["scan_order"]), REALIZATIONS)
        if signs.dtype != np.dtype("int8") or signs.shape != expected_shape or \
                not np.all(np.isin(signs, (-1, 1))):
            die(f"realization_signs must be int8{expected_shape} Rademacher values")
        pinned = boost_mt19937_scan_signs(expected_shape[0])
        if not np.array_equal(signs, pinned):
            die("realization signs differ from the pinned Boost MT19937 stream")
        self.realization_signs = signs
        self.source_sha256 = sha256_file(self.path)

    def _validate_term_array(self, name: str, dtype: np.dtype) -> None:
        value = np.asarray(self.archive[name])
        if value.dtype != dtype or value.ndim != 1 or value.size != self.term_count:
            die(f"{name} must be {dtype}[primitive_term_count]")

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "FullPtcFixtureAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_chunks(self, chunk_size: int) -> Iterable[PrimitiveChunk]:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            die("chunk_size must be a positive integer")
        for start in range(0, self.term_count, chunk_size):
            stop = min(self.term_count, start + chunk_size)
            yield PrimitiveChunk(
                start=start,
                stop=stop,
                arrays={name: np.asarray(self.archive[name][start:stop])
                        for name in (*TERM_INT64, *TERM_UINT8, *TERM_FLOAT64)},
            )


def _nonsymlink_regular(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        die(f"{label} must be a nonsymlink regular file: {path}")
    return candidate.resolve()


def _new_output_path(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or candidate.exists():
        die(f"{label} already exists: {path}")
    return candidate.resolve()


def _canonical_governed_roots(roots: Sequence[Path]) -> list[str]:
    if len(roots) != 5:
        die("resource gate requires exactly five roots in point-source-project, "
            "science-source-project, CAP-POINT, CAP-SCIENCE, compact order")
    canonical: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            die(f"governed root must be an existing nonsymlink absolute directory: {root}")
        resolved = root.resolve()
        if str(root) != str(resolved):
            die(f"governed root is not a canonical resolved path: {root}")
        canonical.append(str(resolved))
    if len(set(canonical)) != len(canonical):
        die("governed root list contains a duplicate")
    return canonical


def _resource_inventory(roots: Sequence[Path]) -> dict[str, Any]:
    """Inventory generated roots without following symlink targets."""
    canonical_roots = _canonical_governed_roots(roots)
    entries: list[dict[str, Any]] = []
    for root_index, root_text in enumerate(canonical_roots):
        root = Path(root_text)
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                directory_before = directory.lstat()
            except OSError as exc:
                raise EvidenceError(
                    f"cannot stat governed directory {directory}: {exc}") from exc
            directory_relative = directory.relative_to(root).as_posix() or "."
            entries.append({
                "root_index": root_index,
                "relative_path": directory_relative,
                "kind": "directory",
                "logical_bytes": int(directory_before.st_size),
                "allocated_bytes": int(directory_before.st_blocks) * 512,
                "sha256": hashlib.sha256(
                    f"directory-v1\0{root_index}\0{directory_relative}".encode()
                ).hexdigest(),
                "symlink_target": None,
            })
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise EvidenceError(f"cannot inventory governed directory {directory}: {exc}") from exc
            directories: list[Path] = []
            for path in children:
                try:
                    before = path.lstat()
                except OSError as exc:
                    raise EvidenceError(f"cannot stat governed path {path}: {exc}") from exc
                relative = path.relative_to(root).as_posix()
                if stat.S_ISLNK(before.st_mode):
                    try:
                        link_text = os.readlink(path)
                        link_bytes = link_text.encode("utf-8")
                    except (OSError, UnicodeEncodeError) as exc:
                        raise EvidenceError(
                            f"cannot encode governed symlink text {path}: {exc}") from exc
                    target = Path(link_text)
                    if not target.is_absolute():
                        target = path.parent / target
                    entries.append({
                        "root_index": root_index,
                        "relative_path": relative,
                        "kind": "symlink",
                        "logical_bytes": int(before.st_size),
                        "allocated_bytes": int(before.st_blocks) * 512,
                        "sha256": hashlib.sha256(link_bytes).hexdigest(),
                        "symlink_target": str(target.resolve(strict=False)),
                    })
                elif stat.S_ISDIR(before.st_mode):
                    directories.append(path)
                elif stat.S_ISREG(before.st_mode):
                    digest = sha256_file(path)
                    try:
                        after = path.lstat()
                    except OSError as exc:
                        raise EvidenceError(
                            f"cannot restat governed file {path}: {exc}") from exc
                    stable = (
                        before.st_dev, before.st_ino, before.st_size,
                        before.st_mtime_ns, before.st_blocks,
                    ) == (
                        after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_blocks,
                    )
                    if not stable or not stat.S_ISREG(after.st_mode):
                        die(f"governed file changed during inventory: {path}")
                    entries.append({
                        "root_index": root_index,
                        "relative_path": relative,
                        "kind": "regular-file",
                        "logical_bytes": int(after.st_size),
                        "allocated_bytes": int(after.st_blocks) * 512,
                        "sha256": digest,
                        "symlink_target": None,
                    })
                else:
                    die(f"governed root contains unsupported special file: {path}")
            pending.extend(reversed(directories))
            try:
                directory_after = directory.lstat()
            except OSError as exc:
                raise EvidenceError(
                    f"cannot restat governed directory {directory}: {exc}") from exc
            if (directory_before.st_dev, directory_before.st_ino,
                    directory_before.st_size, directory_before.st_mtime_ns,
                    directory_before.st_blocks) != (
                    directory_after.st_dev, directory_after.st_ino,
                    directory_after.st_size, directory_after.st_mtime_ns,
                    directory_after.st_blocks):
                die(f"governed directory changed during inventory: {directory}")
    entries.sort(key=lambda item: (item["root_index"], item["relative_path"]))
    return {
        "schema_version": RESOURCE_INVENTORY_SCHEMA,
        "governed_roots": canonical_roots,
        "entries": entries,
    }


def _nonnegative_integer(value: Any, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        die(f"{label} must be a {qualifier} integer")
    return value


def validate_resource_gate(record_path: Path, inventory_path: Path,
                           governed_roots: Sequence[Path], expected_stage: str,
                           stage_target: Path, *,
                           now: datetime | None = None) -> dict[str, Any]:
    """Validate a fresh ED2 pre-stage record and its live root inventory."""
    record_path = _nonsymlink_regular(record_path, "resource pre-stage record")
    inventory_path = _nonsymlink_regular(
        inventory_path, "resource pre-stage inventory")
    roots = _canonical_governed_roots(governed_roots)
    target = Path(stage_target).resolve()
    if not any(target == Path(root) or Path(root) in target.parents for root in roots):
        die("stage destination is outside the exact governed roots")

    record_keys = {
        "schema_version", "request_id", "revision", "candidate_sha", "stage",
        "recorded_at_utc", "governed_roots", "ceiling_bytes",
        "filesystem_root", "filesystem_device",
        "current_logical_bytes", "current_allocated_bytes",
        "projected_incremental_bytes", "filesystem_available_bytes",
        "logical_plus_projected_bytes", "allocated_plus_projected_bytes",
        "projection_authority", "inventory", "passed", "retention",
    }
    record = dict(_require_exact_keys(
        read_json(record_path), record_keys, "resource pre-stage record"))
    exact = {
        "schema_version": RESOURCE_SCHEMA,
        "request_id": REQUEST_ID,
        "revision": CAMPAIGN_REVISION,
        "candidate_sha": CANDIDATE_SHA,
        "stage": expected_stage,
        "ceiling_bytes": RESOURCE_CEILING_BYTES,
        "passed": True,
    }
    for name, expected in exact.items():
        if record[name] != expected:
            die(f"resource pre-stage record {name} differs")
    if record["governed_roots"] != roots:
        die("resource pre-stage governed roots differ from the exact CLI roots")
    filesystem_root = Path(record["filesystem_root"])
    filesystem_device = record["filesystem_device"]
    if not filesystem_root.is_absolute() or filesystem_root.is_symlink() \
            or not filesystem_root.is_dir() \
            or str(filesystem_root.resolve()) != str(filesystem_root) \
            or not isinstance(filesystem_device, int) \
            or isinstance(filesystem_device, bool) or filesystem_device < 0 \
            or int(filesystem_root.stat().st_dev) != filesystem_device \
            or any(int(Path(root).stat().st_dev) != filesystem_device for root in roots):
        die("resource pre-stage filesystem binding differs from governed roots")
    retention = _require_exact_keys(record["retention"], {
        "automatic_cleanup", "capture_point_retained", "capture_science_retained",
    }, "resource retention")
    if retention != {
        "automatic_cleanup": False,
        "capture_point_retained": True,
        "capture_science_retained": True,
    }:
        die("resource retention proof does not retain both full captures")
    projection_binding = _require_exact_keys(record["projection_authority"], {
        "path", "sha256", "method"}, "resource projection binding")
    projection_path = _nonsymlink_regular(
        Path(projection_binding["path"]), "resource projection authority")
    if projection_binding["sha256"] != sha256_file(projection_path):
        die("resource projection authority digest differs")
    projection = _require_exact_keys(read_json(projection_path), {
        "schema_version", "request_id", "revision", "candidate_sha", "stage",
        "method", "source", "fixed_overhead_bytes", "unit_count",
        "bytes_per_unit", "projected_incremental_bytes",
    }, "resource projection authority")
    source_binding = _require_exact_keys(projection["source"], {
        "path", "size_bytes", "sha256", "schema_version",
    }, "resource projection source")
    source_path = _nonsymlink_regular(
        Path(source_binding["path"]), "resource projection source")
    source_node = read_json(source_path)
    if expected_stage.startswith("compact-production:"):
        expected_method = "primitive-count-two-bytes-plus-64mib-v1"
        expected_unit_count = source_node.get("primitive_term_count")
        expected_bytes_per_unit = 2
        group_id = expected_stage.removeprefix("compact-production:")
        source_identity_ok = source_node.get("schema_version") == \
            STREAM_SCHEMA and source_node.get("candidate_sha") == \
            CANDIDATE_SHA and source_node.get("campaign_revision") == \
            CAMPAIGN_REVISION and \
            f"{source_node.get('obsnum')}:{source_node.get('array')}" == group_id
    elif expected_stage.startswith("focused-expansion-plan:") \
            or expected_stage.startswith("focused-expansion:"):
        expected_method = "bounded-request-max-terms-v1"
        expected_unit_count = source_node.get("max_terms")
        expected_bytes_per_unit = 256 if expected_stage.startswith(
            "focused-expansion-plan:") else 2048
        request_id = expected_stage.split(":", 1)[1]
        source_identity_ok = source_node.get("schema_version") == \
            REQUEST_SCHEMA and source_node.get("candidate_sha") == \
            CANDIDATE_SHA and source_node.get("campaign_revision") == \
            CAMPAIGN_REVISION and source_node.get("request_id") == request_id
    else:
        die("compact producer received an unsupported resource stage")
    if projection["schema_version"] != "sci-map-001-resource-projection-v1" \
            or projection["request_id"] != REQUEST_ID \
            or projection["revision"] != CAMPAIGN_REVISION \
            or projection["candidate_sha"] != CANDIDATE_SHA \
            or projection["stage"] != expected_stage \
            or projection["method"] != expected_method \
            or projection["method"] != projection_binding["method"] \
            or projection["projected_incremental_bytes"] != \
            record["projected_incremental_bytes"] \
            or projection["projected_incremental_bytes"] != \
            projection["fixed_overhead_bytes"] + projection["unit_count"] * \
            projection["bytes_per_unit"] \
            or projection["fixed_overhead_bytes"] != 64 * 1024 * 1024 \
            or projection["bytes_per_unit"] != expected_bytes_per_unit \
            or projection["unit_count"] != expected_unit_count \
            or not source_identity_ok \
            or source_binding["size_bytes"] != source_path.stat().st_size \
            or source_binding["sha256"] != sha256_file(source_path) \
            or source_binding["schema_version"] != \
            source_node.get("schema_version"):
        die("compact resource projection derivation differs")

    timestamp_text = record["recorded_at_utc"]
    if not isinstance(timestamp_text, str):
        die("resource pre-stage timestamp is not a string")
    try:
        recorded_at = datetime.strptime(
            timestamp_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise EvidenceError(
            "resource pre-stage timestamp is not RFC3339 UTC whole seconds") from exc
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        die("resource validation clock must be timezone-aware")
    age = (current_time.astimezone(timezone.utc) - recorded_at).total_seconds()
    if age > RESOURCE_MAX_AGE_SECONDS or age < -RESOURCE_MAX_FUTURE_SKEW_SECONDS:
        die("resource pre-stage record is stale or implausibly future-dated")

    numeric_names = (
        "current_logical_bytes", "current_allocated_bytes",
        "projected_incremental_bytes", "filesystem_available_bytes",
        "logical_plus_projected_bytes", "allocated_plus_projected_bytes",
    )
    numbers = {
        name: _nonnegative_integer(
            record[name], f"resource {name}",
            positive=name == "projected_incremental_bytes")
        for name in numeric_names
    }
    if numbers["logical_plus_projected_bytes"] != \
            numbers["current_logical_bytes"] + numbers["projected_incremental_bytes"]:
        die("resource logical-plus-projected arithmetic differs")
    if numbers["allocated_plus_projected_bytes"] != \
            numbers["current_allocated_bytes"] + numbers["projected_incremental_bytes"]:
        die("resource allocated-plus-projected arithmetic differs")
    if numbers["logical_plus_projected_bytes"] > RESOURCE_CEILING_BYTES or \
            numbers["allocated_plus_projected_bytes"] > RESOURCE_CEILING_BYTES:
        die("resource pre-stage cumulative usage exceeds the 200-GiB ceiling")
    if numbers["filesystem_available_bytes"] < numbers["projected_incremental_bytes"]:
        die("resource filesystem capacity does not cover projected incremental output")

    inventory_node = dict(_require_exact_keys(record["inventory"], {
        "path_count", "total_logical_bytes", "total_allocated_bytes", "sha256",
    }, "resource inventory summary"))
    path_count = _nonnegative_integer(
        inventory_node["path_count"], "resource inventory path_count")
    total_logical = _nonnegative_integer(
        inventory_node["total_logical_bytes"],
        "resource inventory total_logical_bytes")
    total_allocated = _nonnegative_integer(
        inventory_node["total_allocated_bytes"],
        "resource inventory total_allocated_bytes")
    inventory_digest = _require_hex_digest(
        inventory_node["sha256"], "resource inventory sha256")
    if total_logical != numbers["current_logical_bytes"] or \
            total_allocated != numbers["current_allocated_bytes"]:
        die("resource inventory totals differ from current governed usage")

    recorded_inventory = read_json(inventory_path)
    recorded_inventory = dict(_require_exact_keys(
        recorded_inventory, {"schema_version", "governed_roots", "entries"},
        "resource inventory document"))
    if recorded_inventory["schema_version"] != RESOURCE_INVENTORY_SCHEMA or \
            recorded_inventory["governed_roots"] != roots or \
            not isinstance(recorded_inventory["entries"], list):
        die("resource inventory document identity or roots differ")
    if hashlib.sha256(canonical_json_bytes(recorded_inventory)).hexdigest() != \
            inventory_digest:
        die("resource inventory canonical digest differs")
    live_inventory = _resource_inventory([Path(root) for root in roots])
    if recorded_inventory != live_inventory:
        die("governed roots changed after the resource pre-stage inventory")
    entries = live_inventory["entries"]
    if path_count != len(entries) or total_logical != sum(
            int(entry["logical_bytes"]) for entry in entries) or \
            total_allocated != sum(int(entry["allocated_bytes"]) for entry in entries):
        die("resource inventory count or totals differ from inventoried entries")
    return record


def _adapter_context(source: Path, authority_path: Path | None) -> Any:
    return (
        CandidatePtcNetCDFAdapter(source, authority_path)
        if authority_path is not None else FullPtcFixtureAdapter(source)
    )


def _plain_netcdf(value: Any, label: str) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        die(f"candidate PTC {label} contains masked/fill values")
    return np.asarray(np.ma.getdata(value))


def _exact_integral(value: Any, label: str) -> np.ndarray:
    array = _plain_netcdf(value, label).astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)) or \
            np.any(array < np.iinfo(np.int64).min) or \
            np.any(array > np.iinfo(np.int64).max):
        die(f"candidate PTC {label} is not finite exact integral authority")
    return array.astype(np.int64)


def _netcdf_scalar(dataset: Any, name: str) -> Any:
    if name not in dataset.variables:
        die(f"candidate PTC lacks scalar {name}")
    value = _plain_netcdf(dataset.variables[name][...], name)
    if value.size != 1:
        die(f"candidate PTC scalar {name} has shape {value.shape}")
    scalar = value.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _cpp_llround(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.full(values.shape, -1, dtype=np.int64)
    if np.any(finite):
        selected = values[finite]
        limit = float(np.iinfo(np.int64).max) - 1.0
        if np.any(np.abs(selected) > limit):
            die(f"candidate PTC {label} cannot be represented by int64 llround")
        rounded = np.where(selected >= 0.0,
                           np.floor(selected + 0.5),
                           np.ceil(selected - 0.5))
        result[finite] = rounded.astype(np.int64)
    return result


class CandidatePtcNetCDFAdapter:
    """Direct, bounded-memory adapter for the unchanged candidate full PTC."""

    ARRAY_IDS = {"a1100": 0, "a1400": 1, "a2000": 2}
    REQUIRED_DOUBLE_VARS = (
        "signal", "flags", "kernel", "det_lat", "det_lon", "weights",
        "apt_flag", "apt_array", "apt_nw", "apt_kids_tone", "apt_uid",
    )

    def __init__(self, ptc_path: Path, authority_path: Path):
        self.path = _nonsymlink_regular(ptc_path, "candidate full PTC")
        self.authority_path = _nonsymlink_regular(
            authority_path, "candidate PTC producer authority")
        self.source_sha256 = sha256_file(self.path)
        authority = read_json(self.authority_path)
        self.metadata = _validate_metadata(authority)
        if self.metadata["adapter"] != NETCDF_ADAPTER:
            die("direct candidate PTC requires candidate-full-ptc-netcdf-v1 authority")
        if self.metadata["capture_ptc_sha256"] != self.source_sha256:
            die("candidate PTC digest differs from producer authority")
        try:
            import netCDF4  # type: ignore
            self.dataset = netCDF4.Dataset(self.path, mode="r")
        except Exception as exc:
            raise EvidenceError(f"cannot open candidate full PTC NetCDF {self.path}: {exc}") from exc
        try:
            self._validate_dataset()
        except Exception:
            self.dataset.close()
            raise

    def _require_var(self, name: str) -> Any:
        variable = self.dataset.variables.get(name)
        if variable is None:
            die(f"candidate full PTC lacks required primitive variable {name}")
        return variable

    def _validate_dataset(self) -> None:
        for name in self.REQUIRED_DOUBLE_VARS:
            variable = self._require_var(name)
            if np.dtype(variable.dtype) != np.dtype("float64"):
                die(f"candidate full PTC {name} is not full binary64")
        for name in ("scan_indices", "output_scan_index", "tod_output_type",
                     "SAMPRATE", "obsnum"):
            self._require_var(name)
        output_type = str(_netcdf_scalar(self.dataset, "tod_output_type"))
        if output_type != "ptc":
            die("candidate capture is not a processed-TOD stream")
        obsnum = int(_netcdf_scalar(self.dataset, "obsnum"))
        if obsnum != self.metadata["obsnum"]:
            die("candidate PTC observation differs from producer authority")
        native = float(_netcdf_scalar(self.dataset, "SAMPRATE"))
        if not math.isfinite(native) or native <= 0.0 or \
                np.float64(native).view(np.uint64) != \
                np.float64(self.metadata["_native_fsmp"]).view(np.uint64):
            die("candidate PTC SAMPRATE differs from native telescope.fsmp authority")

        signal = self.dataset.variables["signal"]
        if len(signal.shape) != 2:
            die("candidate PTC signal must have dimensions [n_pts,n_dets]")
        self.n_pts, self.n_dets = map(int, signal.shape)
        if self.n_pts <= 0 or self.n_dets <= 0:
            die("candidate PTC signal dimensions are empty")
        for name in ("flags", "kernel", "det_lat", "det_lon"):
            if tuple(map(int, self.dataset.variables[name].shape)) != \
                    (self.n_pts, self.n_dets):
                die(f"candidate PTC {name} shape differs from signal")
        scan_indices = _exact_integral(self.dataset.variables["scan_indices"][:],
                                       "scan_indices")
        output_scan = _exact_integral(self.dataset.variables["output_scan_index"][:],
                                      "output_scan_index")
        weights = self.dataset.variables["weights"]
        nscan = len(self.metadata["scan_order"])
        if scan_indices.shape != (nscan, 2) or output_scan.shape != (nscan,) or \
                tuple(map(int, weights.shape)) != (nscan, self.n_dets):
            die("candidate PTC scan/weight dimensions differ from manifest authority")
        if scan_indices[0, 0] != 0 or scan_indices[-1, 1] != self.n_pts - 1 or \
                np.any(scan_indices[:, 0] < 0) or \
                np.any(scan_indices[:, 1] < scan_indices[:, 0]) or \
                np.any(scan_indices[1:, 0] != scan_indices[:-1, 1] + 1):
            die("candidate PTC scan_indices do not cover a contiguous full/all timebase")
        for index, scan in enumerate(self.metadata["scan_order"]):
            count = int(scan_indices[index, 1] - scan_indices[index, 0] + 1)
            if count != scan["sample_count"] or \
                    int(output_scan[index]) != scan["output_scan_index"]:
                die("candidate PTC scan cardinality/identity differs from manifest")
        self.scan_indices = scan_indices

        apt_values = {
            name: _exact_integral(self.dataset.variables[name][:], name)
            for name in ("apt_flag", "apt_array", "apt_nw", "apt_kids_tone", "apt_uid")
        }
        if any(value.shape != (self.n_dets,) for value in apt_values.values()):
            die("candidate PTC APT primitive shape differs from n_dets")
        array_id = self.ARRAY_IDS[self.metadata["array"]]
        apt_rows = np.flatnonzero(apt_values["apt_array"] == array_id).astype(np.int64)
        if apt_rows.size <= 0:
            die("candidate PTC contains no detector for requested array")
        detectors = self.metadata["detector_order"]
        if len(detectors) != int(apt_rows.size):
            die("candidate PTC array detector count differs from manifest")
        for group_index, (detector, apt_row) in enumerate(zip(detectors, apt_rows.tolist())):
            network = int(apt_values["apt_nw"][apt_row])
            kids_tone = int(apt_values["apt_kids_tone"][apt_row])
            uid = str(int(apt_values["apt_uid"][apt_row]))
            flagged = bool(apt_values["apt_flag"][apt_row] != 0)
            expected_identity = (
                f"nw={network};kids_tone={kids_tone};uid={uid};"
                f"apt_row_index={apt_row}")
            actual = {
                "detector_index": group_index,
                "apt_row_index": apt_row,
                "network": network,
                "kids_tone": kids_tone,
                "detector_uid": uid,
                "detector_identity": expected_identity,
                "apt_flagged": flagged,
            }
            if detector != actual:
                die("candidate PTC composite detector identity differs from manifest")
        self.apt_rows = apt_rows
        self.detector_network = np.asarray(
            [d["network"] for d in detectors], dtype=np.int64)
        self.detector_flagged = np.asarray(
            [d["apt_flagged"] for d in detectors], dtype=np.uint8)
        self.term_count = int(self.metadata["primitive_term_count"])
        expected_terms = sum(s["sample_count"] for s in self.metadata["scan_order"]) * \
            len(detectors)
        if self.term_count != expected_terms:
            die("candidate PTC term population differs from manifest")
        self.realization_signs = boost_mt19937_scan_signs(nscan)

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "CandidatePtcNetCDFAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_chunks(self, chunk_size: int) -> Iterable[PrimitiveChunk]:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            die("chunk_size must be a positive integer")
        rows = int(self.metadata["map_shape"]["rows"])
        cols = int(self.metadata["map_shape"]["cols"])
        pixel_size = float(self.metadata["_map_pixel_size_rad"])
        row_center = (rows - 1) / 2.0
        col_center = (cols - 1) / 2.0
        ordinal = 0
        variables = self.dataset.variables
        for scan_index, scan in enumerate(self.metadata["scan_order"]):
            global_start = int(self.scan_indices[scan_index, 0])
            sample_count = int(scan["sample_count"])
            for detector_index, apt_row in enumerate(self.apt_rows.tolist()):
                network = int(self.detector_network[detector_index])
                apt_flagged = int(self.detector_flagged[detector_index])
                coefficient = float(_plain_netcdf(
                    variables["weights"][scan_index, apt_row], "weights").reshape(-1)[0])
                for local_start in range(0, sample_count, chunk_size):
                    local_stop = min(sample_count, local_start + chunk_size)
                    time_slice = slice(global_start + local_start,
                                       global_start + local_stop)
                    signal = _plain_netcdf(
                        variables["signal"][time_slice, apt_row], "signal").astype(
                            np.float64, copy=False)
                    flags_raw = _plain_netcdf(
                        variables["flags"][time_slice, apt_row], "flags").astype(
                            np.float64, copy=False)
                    kernel = _plain_netcdf(
                        variables["kernel"][time_slice, apt_row], "kernel").astype(
                            np.float64, copy=False)
                    lat = _plain_netcdf(
                        variables["det_lat"][time_slice, apt_row], "det_lat").astype(
                            np.float64, copy=False)
                    lon = _plain_netcdf(
                        variables["det_lon"][time_slice, apt_row], "det_lon").astype(
                            np.float64, copy=False)
                    if not np.all(np.isfinite(flags_raw)) or \
                            not np.all(flags_raw == np.rint(flags_raw)):
                        die("candidate PTC flags are not finite exact integral states")
                    sample_flagged = (flags_raw != 0.0).astype(np.uint8)
                    projected_row = lat / pixel_size + row_center
                    projected_col = lon / pixel_size + col_center
                    projection_finite = np.isfinite(projected_row) & \
                        np.isfinite(projected_col)
                    detector_sample_eligible = (apt_flagged == 0) & \
                        ~sample_flagged.astype(bool)
                    if np.any(detector_sample_eligible & ~projection_finite):
                        die("eligible candidate PTC sample has non-finite projection")
                    row = _cpp_llround(projected_row, "det_lat projection")
                    col = _cpp_llround(projected_col, "det_lon projection")
                    geometric = projection_finite & (row >= 0) & (row < rows) & \
                        (col >= 0) & (col < cols)
                    upstream = geometric & detector_sample_eligible
                    count = local_stop - local_start
                    arrays = {
                        "row": row,
                        "col": col,
                        "scan_index": np.full(count, scan_index, dtype=np.int64),
                        "detector_index": np.full(count, detector_index, dtype=np.int64),
                        "sample_index": np.arange(local_start, local_stop, dtype=np.int64),
                        "network": np.full(count, network, dtype=np.int64),
                        "geometric_in_bounds": geometric.astype(np.uint8),
                        "detector_apt_flagged": np.full(count, apt_flagged, dtype=np.uint8),
                        "sample_flagged": sample_flagged,
                        "upstream_eligible": upstream.astype(np.uint8),
                        "coefficient": np.full(count, coefficient, dtype=np.float64),
                        "sample_signal": signal,
                        "sample_kernel": kernel,
                    }
                    yield PrimitiveChunk(ordinal, ordinal + count, arrays)
                    ordinal += count
        if ordinal != self.term_count:
            die("candidate PTC adapter emitted the wrong primitive population")


def _expected_identities(metadata: Mapping[str, Any], start: int,
                         stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordinals = np.arange(start, stop, dtype=np.int64)
    detector_count = len(metadata["detector_order"])
    scan_block_sizes = np.asarray(
        [s["sample_count"] * detector_count for s in metadata["scan_order"]],
        dtype=np.int64,
    )
    block_ends = np.cumsum(scan_block_sizes, dtype=np.int64)
    scan = np.searchsorted(block_ends, ordinals, side="right").astype(np.int64)
    block_starts = np.concatenate((np.array([0], dtype=np.int64), block_ends[:-1]))
    within = ordinals - block_starts[scan]
    sample_counts = np.asarray([s["sample_count"] for s in metadata["scan_order"]],
                               dtype=np.int64)
    detector = within // sample_counts[scan]
    sample = within % sample_counts[scan]
    return scan, detector, sample


def _domain_hash(domain: str, header: Mapping[str, Any]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(b"SCI-MAP-001-COMPACT-V1\x00")
    digest.update(domain.encode("ascii") + b"\x00")
    payload = canonical_json_bytes(header)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)
    return digest


def _digest_header(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_sha": metadata["candidate_sha"],
        "campaign_revision": metadata["campaign_revision"],
        "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
        "mapmaking_bundle_identity_digest": metadata["mapmaking_bundle_identity_digest"],
        "obsnum": metadata["obsnum"],
        "array": metadata["array"],
        "map_shape": metadata["map_shape"],
        "map_pixel_size_rad": metadata["map_pixel_size_rad"],
        "capture_ptc_sha256": metadata["capture_ptc_sha256"],
        "realized_raw_timestream_provenance_sha256":
            metadata["realized_raw_timestream_provenance_sha256"],
        "realized_mapmaking_provenance_sha256":
            metadata["realized_mapmaking_provenance_sha256"],
        "native_fsmp_hz": metadata["native_fsmp_hz"],
        "effective_d_fsmp_hz": metadata["effective_d_fsmp_hz"],
        "scan_order": metadata["scan_order"],
        "detector_order": metadata["detector_order"],
        "primitive_term_count": metadata["primitive_term_count"],
        "term_order": metadata["term_order"],
    }


def _selection_digest(metadata: Mapping[str, Any], scan: Mapping[str, Any],
                      roles: Sequence[str], network: int, state: str,
                      detector: Mapping[str, Any] | None) -> str:
    node = {
        "domain": "deterministic-trace-detector-selection-v1",
        "candidate_sha": metadata["candidate_sha"],
        "campaign_revision": metadata["campaign_revision"],
        "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
        "obsnum": metadata["obsnum"],
        "array": metadata["array"],
        "network": network,
        "scan_index": scan["scan_index"],
        "scan_identity": scan["scan_identity"],
        "scan_roles": list(roles),
        "detector_state": state,
        "detector_identity": None if detector is None else detector["detector_identity"],
        "detector_uid": None if detector is None else detector["detector_uid"],
        "kids_tone": None if detector is None else detector["kids_tone"],
        "apt_row_index": None if detector is None else detector["apt_row_index"],
    }
    return hashlib.sha256(canonical_json_bytes(node)).hexdigest()


def select_trace(metadata: Mapping[str, Any], signs: np.ndarray
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scans = metadata["scan_order"]
    nscan = len(scans)
    role_indices = (("first", 0), ("middle", (nscan - 1) // 2),
                    ("last", nscan - 1))
    selected: dict[int, list[str]] = {}
    for role, index in role_indices:
        selected.setdefault(index, []).append(role)
    networks = sorted({int(d["network"]) for d in metadata["detector_order"]})
    entries: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    selected_signs: list[dict[str, Any]] = []
    absence_count = 0
    budget = 0
    for scan_index in sorted(selected):
        scan = scans[scan_index]
        roles = selected[scan_index]
        selected_signs.append({
            "scan_index": scan_index,
            "scan_identity": scan["scan_identity"],
            "scan_roles": roles,
            "realization_signs": signs[scan_index].astype(int).tolist(),
        })
        for network in networks:
            members = [d for d in metadata["detector_order"]
                       if int(d["network"]) == network]
            for state, flagged in (("valid", False), ("flagged", True)):
                candidates: list[tuple[str, Mapping[str, Any]]] = []
                for detector in members:
                    if bool(detector["apt_flagged"]) != flagged:
                        continue
                    selection_hash = _selection_digest(
                        metadata, scan, roles, network, state, detector,
                    )
                    candidates.append((selection_hash, detector))
                base = {
                    "scan_index": scan_index,
                    "scan_identity": scan["scan_identity"],
                    "scan_roles": roles,
                    "network": network,
                    "detector_state": state,
                }
                if not candidates:
                    absence_count += 1
                    entries.append({
                        **base,
                        "present": False,
                        "absence_reason": "no detector in frozen APT class",
                        "selection_domain_sha256": _selection_digest(
                            metadata, scan, roles, network, state, None),
                    })
                    continue
                selection_hash, detector = min(candidates, key=lambda item: item[0])
                sequence_index = len(sequences)
                sample_count = int(scan["sample_count"])
                sequence = {
                    **base,
                    "detector_index": int(detector["detector_index"]),
                    "apt_row_index": int(detector["apt_row_index"]),
                    "kids_tone": int(detector["kids_tone"]),
                    "detector_identity": str(detector["detector_identity"]),
                    "detector_uid": str(detector["detector_uid"]),
                    "selection_hash": selection_hash,
                    "sequence_index": sequence_index,
                    "sample_count": sample_count,
                    "offset_start": budget,
                    "offset_stop": budget + sample_count,
                }
                entries.append({**sequence, "present": True})
                sequences.append(sequence)
                budget += sample_count
    selection = {
        "schema_version": TRACE_SCHEMA,
        "candidate_sha": metadata["candidate_sha"],
        "campaign_revision": metadata["campaign_revision"],
        "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
        "obsnum": metadata["obsnum"],
        "array": metadata["array"],
        "policy_id": TRACE_POLICY,
        "policy_preregistered_before_capture_output_read": True,
        "active_networks": networks,
        "selected_scans": selected_signs,
        "entries": entries,
        "fixed_budget_terms": budget,
        "sequence_count": len(sequences),
        "absence_fact_count": absence_count,
        "sample_identity_encoding": "implicit-zero-based-index-within-sequence-offsets",
        "coverage_claim": "bounded-engineering-falsification-surface-not-exhaustive",
    }
    if budget <= 0 or not sequences:
        die("deterministic trace selection produced no present sequence")
    return selection, sequences


def _new_trace_arrays(sequences: Sequence[Mapping[str, Any]],
                      sample_interval: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    offsets = [0]
    for sequence in sequences:
        offsets.append(offsets[-1] + int(sequence["sample_count"]))
    budget = offsets[-1]
    arrays: dict[str, np.ndarray] = {
        "sequence_offsets": np.asarray(offsets, dtype=np.int64),
        "row": np.empty(budget, dtype=np.int64),
        "col": np.empty(budget, dtype=np.int64),
        "geometric_in_bounds": np.empty(budget, dtype=np.uint8),
        "detector_apt_flagged": np.empty(budget, dtype=np.uint8),
        "sample_flagged": np.empty(budget, dtype=np.uint8),
        "upstream_eligible": np.empty(budget, dtype=np.uint8),
        "coefficient": np.empty(budget, dtype=np.float64),
        "sample_signal": np.empty(budget, dtype=np.float64),
        "sample_kernel": np.empty(budget, dtype=np.float64),
        "sample_interval_s": np.full(budget, sample_interval, dtype=np.float64),
    }
    return arrays, np.zeros(budget, dtype=bool)


def _hash_term_records(digests: Mapping[str, Any], ordinal: np.ndarray,
                       arrays: Mapping[str, np.ndarray], interval: float) -> None:
    n = ordinal.size
    complete_dtype = np.dtype([
        ("ordinal", "<u8"), ("scan", "<i8"), ("detector", "<i8"),
        ("sample", "<i8"), ("network", "<i8"), ("row", "<i8"),
        ("col", "<i8"), ("geometric", "u1"), ("apt_flagged", "u1"),
        ("sample_flagged", "u1"), ("upstream", "u1"),
        ("coefficient", "<f8"), ("signal", "<f8"), ("kernel", "<f8"),
        ("interval", "<f8"),
    ], align=False)
    complete = np.empty(n, dtype=complete_dtype)
    assignments = {
        "ordinal": ordinal.astype(np.uint64, copy=False),
        "scan": arrays["scan_index"], "detector": arrays["detector_index"],
        "sample": arrays["sample_index"], "network": arrays["network"],
        "row": arrays["row"], "col": arrays["col"],
        "geometric": arrays["geometric_in_bounds"],
        "apt_flagged": arrays["detector_apt_flagged"],
        "sample_flagged": arrays["sample_flagged"],
        "upstream": arrays["upstream_eligible"],
        "coefficient": arrays["coefficient"],
        "signal": arrays["sample_signal"], "kernel": arrays["sample_kernel"],
        "interval": np.full(n, interval, dtype=np.float64),
    }
    for name, value in assignments.items():
        complete[name] = value
    digests["complete_primitive_order_sha256"].update(complete.tobytes(order="C"))

    identity_dtype = np.dtype([
        ("ordinal", "<u8"), ("scan", "<i8"), ("detector", "<i8"),
        ("sample", "<i8"), ("network", "<i8"), ("row", "<i8"),
        ("col", "<i8"),
    ], align=False)
    identity = np.empty(n, dtype=identity_dtype)
    for name in identity.dtype.names or ():
        identity[name] = complete[name]
    digests["identity_order_sha256"].update(identity.tobytes(order="C"))

    eligibility_dtype = np.dtype([
        ("ordinal", "<u8"), ("geometric", "u1"), ("apt_flagged", "u1"),
        ("sample_flagged", "u1"), ("upstream", "u1"),
    ], align=False)
    eligibility = np.empty(n, dtype=eligibility_dtype)
    for name in eligibility.dtype.names or ():
        eligibility[name] = complete[name]
    digests["eligibility_sha256"].update(eligibility.tobytes(order="C"))

    contribution_dtype = np.dtype([
        ("ordinal", "<u8"), ("coefficient", "<f8"), ("signal", "<f8"),
        ("kernel", "<f8"), ("interval", "<f8"),
    ], align=False)
    contribution = np.empty(n, dtype=contribution_dtype)
    for name in contribution.dtype.names or ():
        contribution[name] = complete[name]
    digests["contribution_sha256"].update(contribution.tobytes(order="C"))


def produce_compact_group(source: Path, output_dir: Path, chunk_size: int = 262144,
                          *, authority_path: Path | None = None,
                          resource_record_path: Path | None = None,
                          resource_inventory_path: Path | None = None,
                          governed_roots: Sequence[Path] = (),
                          resource_now: datetime | None = None) -> Path:
    """Stream one fixture or direct candidate full/all PTC into one group."""
    output_dir = _new_output_path(output_dir, "compact group destination")
    gate_values = (resource_record_path, resource_inventory_path)
    if authority_path is None:
        if any(value is not None for value in gate_values) or governed_roots:
            die("local fixture production does not accept operational resource records")
    else:
        if any(value is None for value in gate_values) or not governed_roots:
            die("direct candidate PTC production requires a fresh resource record, "
                "inventory, and exact governed roots")
        authority = _validate_metadata(read_json(_nonsymlink_regular(
            authority_path, "candidate PTC producer authority")))
        stage = f"compact-production:{authority['obsnum']}:{authority['array']}"
        validate_resource_gate(
            resource_record_path, resource_inventory_path, governed_roots,
            stage, output_dir, now=resource_now)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.",
                                      dir=output_dir.parent))
    try:
        adapter_context = _adapter_context(source, authority_path)
        with adapter_context as adapter:
            metadata = adapter.metadata
            nscan = len(metadata["scan_order"])
            ndet = len(metadata["detector_order"])
            rows = int(metadata["map_shape"]["rows"])
            cols = int(metadata["map_shape"]["cols"])
            npixel = rows * cols
            interval = float(metadata["_sample_interval"])
            scan_shape = (nscan, rows, cols)
            stats = {
                "signal_numerator": np.zeros(scan_shape, dtype=np.float64),
                "weight": np.zeros(scan_shape, dtype=np.float64),
                "kernel_numerator": np.zeros(scan_shape, dtype=np.float64),
                "upstream_eligible_exposure": np.zeros(scan_shape, dtype=np.float64),
                "retained_exposure": np.zeros(scan_shape, dtype=np.float64),
                "geometric_hits": np.zeros((rows, cols), dtype=np.int64),
                "contributing_hits": np.zeros((rows, cols), dtype=np.int64),
                "realization_signs": adapter.realization_signs.copy(),
            }
            selection, sequences = select_trace(metadata, adapter.realization_signs)
            trace, trace_seen = _new_trace_arrays(sequences, interval)
            sequence_by_pair = {
                (int(sequence["scan_index"]), int(sequence["detector_index"])):
                    sequence
                for sequence in sequences
            }

            header = _digest_header(metadata)
            digest_states = {
                "complete_primitive_order_sha256":
                    _domain_hash("complete-primitive-order", header),
                "population_membership_sha256":
                    _domain_hash("population-membership", header),
                "identity_order_sha256": _domain_hash("identity-order", header),
                "eligibility_sha256": _domain_hash("eligibility", header),
                "contribution_sha256": _domain_hash("contribution", header),
                "realization_signs_sha256":
                    _domain_hash("realization-signs", header),
            }
            digest_states["realization_signs_sha256"].update(
                np.ascontiguousarray(adapter.realization_signs).tobytes(order="C"))

            detector_network = np.asarray(
                [d["network"] for d in metadata["detector_order"]], dtype=np.int64)
            detector_flagged = np.asarray(
                [d["apt_flagged"] for d in metadata["detector_order"]],
                dtype=np.uint8)
            for chunk in adapter.iter_chunks(chunk_size):
                values = chunk.arrays
                ordinal = np.arange(chunk.start, chunk.stop, dtype=np.uint64)
                expected_scan, expected_detector, expected_sample = \
                    _expected_identities(metadata, chunk.start, chunk.stop)
                if not np.array_equal(values["scan_index"], expected_scan) or \
                        not np.array_equal(values["detector_index"], expected_detector) or \
                        not np.array_equal(values["sample_index"], expected_sample):
                    die(f"primitive stream omits, repeats, or reorders Cartesian identity "
                        f"at ordinal range [{chunk.start},{chunk.stop})")
                if not np.array_equal(values["network"],
                                      detector_network[expected_detector]):
                    die("primitive network differs from frozen detector authority")
                if not np.array_equal(values["detector_apt_flagged"],
                                      detector_flagged[expected_detector]):
                    die("primitive detector state differs from frozen APT authority")
                for name in TERM_UINT8:
                    if not np.all(np.isin(values[name], (0, 1))):
                        die(f"primitive {name} is not binary")
                geometric = values["geometric_in_bounds"].astype(bool, copy=False)
                apt_flagged = values["detector_apt_flagged"].astype(bool, copy=False)
                sample_flagged = values["sample_flagged"].astype(bool, copy=False)
                upstream = values["upstream_eligible"].astype(bool, copy=False)
                expected_upstream = geometric & ~apt_flagged & ~sample_flagged
                if not np.array_equal(upstream, expected_upstream):
                    die("upstream_eligible differs from geometry/APT/sample flags")
                row = values["row"]
                col = values["col"]
                if np.any(geometric & ((row < 0) | (row >= rows) |
                                       (col < 0) | (col >= cols))):
                    die("geometrically in-bounds primitive has invalid pixel index")
                coefficient = values["coefficient"]
                signal = values["sample_signal"]
                kernel = values["sample_kernel"]
                if np.any(upstream & ~np.isfinite(coefficient)):
                    die("upstream-eligible coefficient is non-finite")
                contributing = upstream & np.isfinite(coefficient) & (coefficient > 0.0)
                if np.any(contributing & ~np.isfinite(signal)):
                    die("contributing sample signal is non-finite")
                if np.any(contributing & ~np.isfinite(kernel)):
                    die("contributing sample kernel is non-finite")
                weighted_signal = coefficient[contributing] * signal[contributing]
                weighted_kernel = coefficient[contributing] * kernel[contributing]
                if not np.all(np.isfinite(weighted_signal)) or \
                        not np.all(np.isfinite(weighted_kernel)):
                    die("weighted contribution overflowed binary64")

                _hash_term_records(digest_states, ordinal, values, interval)

                geom_pixel = row[geometric] * cols + col[geometric]
                np.add.at(stats["geometric_hits"].reshape(-1), geom_pixel, 1)
                upstream_flat = expected_scan[upstream] * npixel + \
                    row[upstream] * cols + col[upstream]
                np.add.at(stats["upstream_eligible_exposure"].reshape(-1),
                          upstream_flat, interval)
                contrib_flat = expected_scan[contributing] * npixel + \
                    row[contributing] * cols + col[contributing]
                aggregate_pixel = row[contributing] * cols + col[contributing]
                np.add.at(stats["contributing_hits"].reshape(-1), aggregate_pixel, 1)
                np.add.at(stats["signal_numerator"].reshape(-1), contrib_flat,
                          weighted_signal)
                np.add.at(stats["weight"].reshape(-1), contrib_flat,
                          coefficient[contributing])
                np.add.at(stats["kernel_numerator"].reshape(-1), contrib_flat,
                          weighted_kernel)
                np.add.at(stats["retained_exposure"].reshape(-1), contrib_flat,
                          interval)

                keys = expected_scan * ndet + expected_detector
                for (scan_index, detector_index), sequence in sequence_by_pair.items():
                    selected = keys == scan_index * ndet + detector_index
                    if not np.any(selected):
                        continue
                    samples = expected_sample[selected]
                    destination = int(sequence["offset_start"]) + samples
                    if np.any(trace_seen[destination]):
                        die("deterministic trace contains a repeated sample identity")
                    trace_seen[destination] = True
                    for name in ("row", "col", *TERM_UINT8, *TERM_FLOAT64):
                        trace[name][destination] = values[name][selected]

            if not np.all(trace_seen):
                die("deterministic trace is missing selected primitive samples")
            if any(np.any(value < 0) for name, value in stats.items()
                   if name in ("weight", "upstream_eligible_exposure",
                               "retained_exposure", "geometric_hits",
                               "contributing_hits")):
                die("compact nonnegative statistic overflowed or became negative")
            if any(not np.all(np.isfinite(value)) for name, value in stats.items()
                   if value.dtype == np.float64):
                die("compact sufficient statistic is non-finite")

            stats_path = temporary / "sufficient-statistics.npz"
            trace_path = temporary / "deterministic-trace.npz"
            selection_path = temporary / "trace-selection.json"
            deterministic_npz(stats_path, stats)
            deterministic_npz(trace_path, trace)
            write_json(selection_path, selection)
            artifacts = {
                "sufficient_statistics": {
                    "path": stats_path.name, "sha256": sha256_file(stats_path),
                    "stored_bytes": stats_path.stat().st_size,
                },
                "deterministic_trace": {
                    "path": trace_path.name, "sha256": sha256_file(trace_path),
                    "stored_bytes": trace_path.stat().st_size,
                },
                "trace_selection": {
                    "path": selection_path.name, "sha256": sha256_file(selection_path),
                    "stored_bytes": selection_path.stat().st_size,
                },
            }
            networks = selection["active_networks"]
            group = {
                "schema_version": GROUP_SCHEMA,
                "candidate_sha": metadata["candidate_sha"],
                "campaign_revision": metadata["campaign_revision"],
                "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
                "producer_identity": metadata["producer_identity"],
                "source_stream_sha256": adapter.source_sha256,
                "realized_raw_timestream_provenance_sha256":
                    metadata["realized_raw_timestream_provenance_sha256"],
                "realized_mapmaking_provenance_sha256":
                    metadata["realized_mapmaking_provenance_sha256"],
                "mapmaking_bundle_identity_digest":
                    metadata["mapmaking_bundle_identity_digest"],
                "obsnum": metadata["obsnum"],
                "array": metadata["array"],
                "map_shape": {
                    **dict(metadata["map_shape"]),
                    "pixel_size_rad": dict(metadata["map_pixel_size_rad"]),
                },
                "rates": {
                    "native_fsmp_hz": dict(metadata["native_fsmp_hz"]),
                    "effective_d_fsmp_hz": dict(metadata["effective_d_fsmp_hz"]),
                    "sample_interval_s": exact_float_node(
                        interval, "binary64(1/telescope.d_fsmp)"),
                },
                "scan_order": [
                    {name: scan[name]
                     for name in ("scan_index", "scan_identity", "sample_count")}
                    for scan in metadata["scan_order"]
                ],
                "active_networks": networks,
                "population": {
                    "scan_count": nscan,
                    "detector_count": ndet,
                    "pixel_count": npixel,
                    "primitive_term_count": adapter.term_count,
                    "selected_scan_count": len(selection["selected_scans"]),
                    "active_network_count": len(networks),
                    "trace_sequence_count": len(sequences),
                    "trace_term_count": int(selection["fixed_budget_terms"]),
                },
                "stream_digests": {
                    name: digest.hexdigest() for name, digest in digest_states.items()
                },
                "artifacts": artifacts,
                "trace_selection": {
                    "policy_id": TRACE_POLICY,
                    "fixed_budget_terms": int(selection["fixed_budget_terms"]),
                    "measured_terms": int(trace_seen.sum()),
                    "sequence_count": len(sequences),
                    "absence_fact_count": int(selection["absence_fact_count"]),
                },
            }
            write_json(temporary / "group.json", group)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "group.json"


@dataclass
class LoadedCompactGroup:
    path: Path
    group: dict[str, Any]
    stats: dict[str, np.ndarray]
    trace_selection: dict[str, Any]
    trace: dict[str, np.ndarray]


def _load_npz_exact(path: Path, members: Mapping[str, np.dtype]
                    ) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        die(f"compact artifact is not a nonsymlink regular file: {path}")
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise EvidenceError(f"cannot load compact artifact {path}: {exc}") from exc
    with archive:
        if set(archive.files) != set(members):
            die(f"{path.name} members differ; missing={sorted(set(members)-set(archive.files))}, "
                f"extra={sorted(set(archive.files)-set(members))}")
        arrays = {name: np.asarray(archive[name]).copy() for name in members}
    for name, dtype in members.items():
        if arrays[name].dtype != dtype:
            die(f"{path.name}:{name} dtype differs: {arrays[name].dtype} != {dtype}")
    return arrays


def _validate_group_json(group: Any) -> dict[str, Any]:
    node = dict(_require_exact_keys(group, GROUP_KEYS, "compact group"))
    exact = {
        "schema_version": GROUP_SCHEMA,
        "candidate_sha": CANDIDATE_SHA,
        "campaign_revision": CAMPAIGN_REVISION,
    }
    for name, expected in exact.items():
        if node[name] != expected:
            die(f"compact group {name} differs")
    _require_hex_digest(node["raw_input_manifest_sha256"],
                        "group raw manifest digest")
    _require_hex_digest(node["source_stream_sha256"], "group source stream digest")
    _require_hex_digest(node["realized_raw_timestream_provenance_sha256"],
                        "group realized raw-timestream provenance digest")
    _require_hex_digest(node["realized_mapmaking_provenance_sha256"],
                        "group realized mapmaking provenance digest")
    if not isinstance(node["producer_identity"], str) or not node["producer_identity"]:
        die("compact group producer_identity is empty")
    prefix = "canonical-hexfloat-sha256-v1:"
    identity = node["mapmaking_bundle_identity_digest"]
    if not isinstance(identity, str) or not identity.startswith(prefix):
        die("compact group mapmaking identity domain differs")
    _require_hex_digest(identity[len(prefix):], "group mapmaking identity")
    if node["obsnum"] not in (152389, 152390, 152392) or \
            node["array"] not in ARRAY_NETWORKS:
        die("compact group observation/array is outside the campaign")
    shape = _require_exact_keys(node["map_shape"], {"rows", "cols", "pixel_size_rad"},
                                "compact map_shape")
    rows, cols = shape["rows"], shape["cols"]
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
           for x in (rows, cols)):
        die("compact map shape is invalid")
    parse_exact_float(shape["pixel_size_rad"],
                      "realized_mapmaking.effective_pixel_size_rad")
    rates = _require_exact_keys(node["rates"],
                               {"native_fsmp_hz", "effective_d_fsmp_hz",
                                "sample_interval_s"}, "compact rates")
    parse_exact_float(rates["native_fsmp_hz"], "telescope.fsmp")
    effective = parse_exact_float(rates["effective_d_fsmp_hz"],
                                  "telescope.d_fsmp")
    interval = parse_exact_float(rates["sample_interval_s"],
                                 "binary64(1/telescope.d_fsmp)")
    expected_interval = float(np.float64(1.0) / np.float64(effective))
    if np.float64(interval).view(np.uint64) != np.float64(expected_interval).view(np.uint64):
        die("compact sample interval is not bit-equal to 1/effective d_fsmp")
    scans = node["scan_order"]
    if not isinstance(scans, list) or not scans:
        die("compact scan_order is empty")
    for index, scan in enumerate(scans):
        scan = _require_exact_keys(scan, {"scan_index", "scan_identity", "sample_count"},
                                   f"compact scan_order[{index}]")
        if scan["scan_index"] != index or not isinstance(scan["scan_identity"], str) or \
                not scan["scan_identity"] or not isinstance(scan["sample_count"], int) or \
                isinstance(scan["sample_count"], bool) or scan["sample_count"] <= 0:
            die("compact scan_order entry is invalid")
    if len({s["scan_identity"] for s in scans}) != len(scans):
        die("compact scan identities are repeated")
    networks = node["active_networks"]
    if not isinstance(networks, list) or not networks or \
            networks != sorted(set(networks)) or \
            any(n not in ARRAY_NETWORKS[node["array"]] for n in networks):
        die("compact active_networks are invalid, repeated, or out of array")
    population_keys = {
        "scan_count", "detector_count", "pixel_count", "primitive_term_count",
        "selected_scan_count", "active_network_count", "trace_sequence_count",
        "trace_term_count",
    }
    population = _require_exact_keys(node["population"], population_keys,
                                     "compact population")
    if any(not isinstance(population[name], int) or isinstance(population[name], bool) or
           population[name] <= 0 for name in population_keys):
        die("compact population counts must be positive integers")
    if population["scan_count"] != len(scans) or \
            population["pixel_count"] != rows * cols or \
            population["active_network_count"] != len(networks) or \
            population["selected_scan_count"] > 3:
        die("compact population counts differ from group dimensions")
    digest_keys = {
        "complete_primitive_order_sha256", "population_membership_sha256",
        "identity_order_sha256", "eligibility_sha256", "contribution_sha256",
        "realization_signs_sha256",
    }
    stream_digests = _require_exact_keys(node["stream_digests"], digest_keys,
                                         "compact stream_digests")
    for name in digest_keys:
        _require_hex_digest(stream_digests[name], f"stream digest {name}")
    artifact_keys = {"sufficient_statistics", "deterministic_trace", "trace_selection"}
    artifacts = _require_exact_keys(node["artifacts"], artifact_keys,
                                    "compact artifacts")
    expected_names = {
        "sufficient_statistics": "sufficient-statistics.npz",
        "deterministic_trace": "deterministic-trace.npz",
        "trace_selection": "trace-selection.json",
    }
    for name, filename in expected_names.items():
        artifact = _require_exact_keys(artifacts[name], {"path", "sha256", "stored_bytes"},
                                       f"artifact {name}")
        if artifact["path"] != filename:
            die(f"artifact {name} path differs")
        _require_hex_digest(artifact["sha256"], f"artifact {name} digest")
        if not isinstance(artifact["stored_bytes"], int) or \
                isinstance(artifact["stored_bytes"], bool) or artifact["stored_bytes"] <= 0:
            die(f"artifact {name} stored_bytes is invalid")
    trace_summary = _require_exact_keys(
        node["trace_selection"],
        {"policy_id", "fixed_budget_terms", "measured_terms", "sequence_count",
         "absence_fact_count"}, "compact trace_selection summary")
    if trace_summary["policy_id"] != TRACE_POLICY:
        die("compact trace selection policy differs")
    for name in ("fixed_budget_terms", "measured_terms", "sequence_count"):
        if not isinstance(trace_summary[name], int) or isinstance(trace_summary[name], bool) or \
                trace_summary[name] <= 0:
            die(f"compact trace summary {name} is invalid")
    if not isinstance(trace_summary["absence_fact_count"], int) or \
            isinstance(trace_summary["absence_fact_count"], bool) or \
            trace_summary["absence_fact_count"] < 0:
        die("compact absence_fact_count is invalid")
    if trace_summary["fixed_budget_terms"] != trace_summary["measured_terms"] or \
            trace_summary["fixed_budget_terms"] != population["trace_term_count"] or \
            trace_summary["sequence_count"] != population["trace_sequence_count"]:
        die("compact trace summary/population counts differ")
    return node


def _validate_trace_selection(selection: Any, group: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "candidate_sha", "campaign_revision",
        "raw_input_manifest_sha256", "obsnum", "array", "policy_id",
        "policy_preregistered_before_capture_output_read", "active_networks",
        "selected_scans", "entries", "fixed_budget_terms", "sequence_count",
        "absence_fact_count", "sample_identity_encoding", "coverage_claim",
    }
    node = dict(_require_exact_keys(selection, expected_keys, "trace selection"))
    comparisons = {
        "schema_version": TRACE_SCHEMA,
        "candidate_sha": group["candidate_sha"],
        "campaign_revision": group["campaign_revision"],
        "raw_input_manifest_sha256": group["raw_input_manifest_sha256"],
        "obsnum": group["obsnum"], "array": group["array"],
        "policy_id": TRACE_POLICY,
        "active_networks": group["active_networks"],
        "fixed_budget_terms": group["trace_selection"]["fixed_budget_terms"],
        "sequence_count": group["trace_selection"]["sequence_count"],
        "absence_fact_count": group["trace_selection"]["absence_fact_count"],
        "sample_identity_encoding":
            "implicit-zero-based-index-within-sequence-offsets",
        "coverage_claim": "bounded-engineering-falsification-surface-not-exhaustive",
    }
    for name, expected in comparisons.items():
        if node[name] != expected:
            die(f"trace selection {name} differs from compact group")
    if node["policy_preregistered_before_capture_output_read"] is not True:
        die("trace policy was not recorded as preregistered")
    selected_scans = node["selected_scans"]
    if not isinstance(selected_scans, list) or \
            len(selected_scans) != group["population"]["selected_scan_count"]:
        die("trace selected-scan count differs")
    selected_indices: list[int] = []
    for item in selected_scans:
        item = _require_exact_keys(
            item, {"scan_index", "scan_identity", "scan_roles", "realization_signs"},
            "trace selected scan")
        index = item["scan_index"]
        if not isinstance(index, int) or index < 0 or index >= len(group["scan_order"]) or \
                item["scan_identity"] != group["scan_order"][index]["scan_identity"]:
            die("trace selected scan identity differs")
        if not isinstance(item["scan_roles"], list) or not item["scan_roles"] or \
                any(role not in ("first", "middle", "last") for role in item["scan_roles"]):
            die("trace scan roles are invalid")
        signs = item["realization_signs"]
        if not isinstance(signs, list) or len(signs) != REALIZATIONS or \
                any(sign not in (-1, 1) for sign in signs):
            die("trace selected realization signs are invalid")
        selected_indices.append(index)
    expected_indices = sorted({0, (len(group["scan_order"]) - 1) // 2,
                               len(group["scan_order"]) - 1})
    if selected_indices != expected_indices:
        die("trace selected scans differ from first/lower-middle/last")
    entries = node["entries"]
    expected_entry_count = len(selected_indices) * len(group["active_networks"]) * 2
    if not isinstance(entries, list) or len(entries) != expected_entry_count:
        die("trace entries do not cover every selected scan/network/class")
    present = 0
    absent = 0
    sequence_indices: list[int] = []
    seen_domains: set[tuple[int, int, str]] = set()
    for entry in entries:
        common = {"scan_index", "scan_identity", "scan_roles", "network",
                  "detector_state", "present"}
        if not isinstance(entry, Mapping) or not common.issubset(entry):
            die("trace entry is incomplete")
        key = (entry["scan_index"], entry["network"], entry["detector_state"])
        if key in seen_domains or entry["scan_index"] not in selected_indices or \
                entry["network"] not in group["active_networks"] or \
                entry["detector_state"] not in ("valid", "flagged"):
            die("trace entry domain is repeated or invalid")
        seen_domains.add(key)
        if entry["present"] is True:
            expected = common | {"detector_index", "detector_uid", "selection_hash",
                                 "sequence_index", "sample_count", "offset_start",
                                 "offset_stop", "apt_row_index", "kids_tone",
                                 "detector_identity"}
            if set(entry) != expected:
                die("present trace entry fields differ")
            _require_hex_digest(entry["selection_hash"], "trace selection hash")
            if not isinstance(entry["detector_uid"], str) or not entry["detector_uid"]:
                die("present trace detector UID is empty")
            expected_identity = (
                f"nw={entry['network']};kids_tone={entry['kids_tone']};"
                f"uid={entry['detector_uid']};apt_row_index={entry['apt_row_index']}")
            if entry["detector_identity"] != expected_identity:
                die("present trace composite detector identity differs")
            if entry["sample_count"] != group["scan_order"][entry["scan_index"]]["sample_count"]:
                die("present trace sample_count differs from scan authority")
            if entry["offset_stop"] - entry["offset_start"] != entry["sample_count"]:
                die("present trace offset interval differs from sample_count")
            sequence_indices.append(entry["sequence_index"])
            present += 1
        elif entry["present"] is False:
            expected = common | {"absence_reason", "selection_domain_sha256"}
            if set(entry) != expected or entry["absence_reason"] != \
                    "no detector in frozen APT class":
                die("absent trace entry fields differ")
            _require_hex_digest(entry["selection_domain_sha256"],
                                "trace absence domain digest")
            absent += 1
        else:
            die("trace entry present must be Boolean")
    if sequence_indices != list(range(present)) or present != node["sequence_count"] or \
            absent != node["absence_fact_count"]:
        die("trace present/absence sequence counts differ")
    return node


def load_compact_group(path: Path | str) -> LoadedCompactGroup:
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        die(f"compact group path must be a nonsymlink group.json: {path}")
    group_path = raw_path.resolve()
    if group_path.name != "group.json":
        die(f"compact group path must be a nonsymlink group.json: {path}")
    group = _validate_group_json(read_json(group_path))
    directory = group_path.parent
    for name, artifact in group["artifacts"].items():
        artifact_path = directory / artifact["path"]
        if artifact_path.parent != directory or not artifact_path.is_file() or \
                artifact_path.is_symlink():
            die(f"compact artifact {name} is absent, unsafe, or escapes group directory")
        if artifact_path.stat().st_size != artifact["stored_bytes"] or \
                sha256_file(artifact_path) != artifact["sha256"]:
            die(f"compact artifact {name} size or digest differs")
    stats = _load_npz_exact(directory / "sufficient-statistics.npz", STATS_MEMBERS)
    trace = _load_npz_exact(directory / "deterministic-trace.npz", TRACE_MEMBERS)
    selection = _validate_trace_selection(read_json(directory / "trace-selection.json"),
                                          group)
    rows = int(group["map_shape"]["rows"])
    cols = int(group["map_shape"]["cols"])
    nscan = int(group["population"]["scan_count"])
    scan_shape = (nscan, rows, cols)
    for name in ("signal_numerator", "weight", "kernel_numerator",
                 "upstream_eligible_exposure", "retained_exposure"):
        if stats[name].shape != scan_shape or not np.all(np.isfinite(stats[name])):
            die(f"compact statistic {name} has wrong shape or non-finite values")
    for name in ("geometric_hits", "contributing_hits"):
        if stats[name].shape != (rows, cols) or np.any(stats[name] < 0):
            die(f"compact statistic {name} has wrong shape or negative values")
    for name in ("weight", "upstream_eligible_exposure", "retained_exposure"):
        if np.any(stats[name] < 0):
            die(f"compact statistic {name} is negative")
    signs = stats["realization_signs"]
    if signs.shape != (nscan, REALIZATIONS) or not np.all(np.isin(signs, (-1, 1))) or \
            not np.array_equal(signs, boost_mt19937_scan_signs(nscan)):
        die("compact realization_signs differ from pinned stream")
    expected_stats_elements = 5 * nscan * rows * cols + 2 * rows * cols + \
        REALIZATIONS * nscan
    if sum(value.size for value in stats.values()) != expected_stats_elements:
        die("compact sufficient statistics exceed the ordinary-array bound")

    budget = int(group["trace_selection"]["fixed_budget_terms"])
    sequences = int(group["trace_selection"]["sequence_count"])
    offsets = trace["sequence_offsets"]
    if offsets.shape != (sequences + 1,) or offsets[0] != 0 or offsets[-1] != budget or \
            np.any(np.diff(offsets) <= 0):
        die("trace sequence_offsets differ from fixed positive sequence budget")
    for name, value in trace.items():
        if name != "sequence_offsets" and value.shape != (budget,):
            die(f"trace {name} shape differs from fixed trace budget")
    for name in ("geometric_in_bounds", "detector_apt_flagged", "sample_flagged",
                 "upstream_eligible"):
        if not np.all(np.isin(trace[name], (0, 1))):
            die(f"trace {name} is not binary")
    expected_upstream = trace["geometric_in_bounds"].astype(bool) & \
        ~trace["detector_apt_flagged"].astype(bool) & \
        ~trace["sample_flagged"].astype(bool)
    if not np.array_equal(trace["upstream_eligible"].astype(bool), expected_upstream):
        die("trace upstream eligibility differs from primitive flags")
    interval = parse_exact_float(group["rates"]["sample_interval_s"],
                                 "binary64(1/telescope.d_fsmp)")
    if not np.array_equal(
            np.ascontiguousarray(trace["sample_interval_s"]).view(np.uint64),
            np.full(budget, interval, dtype=np.float64).view(np.uint64)):
        die("trace sample intervals differ from effective-rate authority")
    return LoadedCompactGroup(group_path, group, stats, selection, trace)


def threshold_selection(weight: np.ndarray, cut: float) -> dict[str, Any]:
    if not math.isfinite(cut) or cut < 0.0:
        die("coverage cut must be finite and nonnegative")
    flat = np.asarray(weight, dtype=np.float64).ravel()
    values = np.sort(flat[np.isfinite(flat) & (flat > 0.0)])
    count = int(values.size)
    if count == 0:
        return {"threshold": 0.0, "selected": 0.0, "count": 0, "index": None}
    lower = int(math.floor(0.75 * count))
    index = (lower + count) // 2
    selected = float(values[index])
    return {"threshold": float(cut) * selected, "selected": selected,
            "count": count, "index": index}


def reconstruct_compact_group(path: Path | str, coverage_cut: float
                              ) -> dict[str, Any]:
    loaded = load_compact_group(path)
    stats = loaded.stats
    nscan, rows, cols = stats["weight"].shape
    aggregate_names = (
        "signal_numerator", "weight", "kernel_numerator",
        "upstream_eligible_exposure", "retained_exposure",
    )
    aggregate = {name: np.zeros((rows, cols), dtype=np.float64)
                 for name in aggregate_names}
    noise_numerator = np.zeros((rows, cols, REALIZATIONS), dtype=np.float64)
    for scan in range(nscan):
        for name in aggregate_names:
            aggregate[name] += stats[name][scan]
        noise_numerator += stats["realization_signs"][scan][None, None, :] * \
            stats["signal_numerator"][scan][..., None]
    weight = aggregate["weight"]
    normalization = threshold_selection(weight, coverage_cut / 10.0)
    norm = np.isfinite(weight) & (weight > 0.0) & \
        (weight >= normalization["threshold"])
    denominator = np.where(weight > 0.0, weight, 1.0)
    final_weight = np.where(norm, weight, 0.0)
    signal = np.where(norm, aggregate["signal_numerator"] / denominator, 0.0)
    kernel = np.where(norm, aggregate["kernel_numerator"] / denominator, 0.0)
    noise = np.where(norm[..., None], noise_numerator / denominator[..., None], 0.0)
    retained = np.where(norm, aggregate["retained_exposure"], 0.0)
    science_policy = threshold_selection(final_weight, coverage_cut)
    policy = np.isfinite(final_weight) & (final_weight > 0.0) & \
        (final_weight >= science_policy["threshold"])
    finite_companions = np.isfinite(signal) & np.isfinite(final_weight) & \
        (final_weight > 0.0) & np.isfinite(kernel) & \
        np.all(np.isfinite(noise), axis=-1)
    valid = norm & policy & finite_companions
    planes = {
        "signal_I": signal,
        "weight_I": final_weight,
        "kernel_I": kernel,
        "geometric_hits_I": stats["geometric_hits"].copy(),
        "contributing_hits_I": stats["contributing_hits"].copy(),
        "upstream_eligible_exposure_I": aggregate["upstream_eligible_exposure"],
        "retained_exposure_I": retained,
        "normalization_support_I": norm.astype(np.uint8),
        "science_policy_support_I": policy.astype(np.uint8),
        "science_valid_I": valid.astype(np.uint8),
        "coverage_I": retained.copy(),
        "coverage_bool_I": policy.astype(np.uint8),
    }
    return {
        "group": loaded.group,
        "planes": planes,
        "noise": noise,
        "normalization": normalization,
        "science_policy": science_policy,
        "raw_numerators": aggregate,
        "per_scan": {name: stats[name].copy() for name in aggregate_names},
        "realization_signs": stats["realization_signs"].copy(),
    }


def verify_nine_group_mapping(collection_path: Path | str) -> dict[str, str]:
    raw_collection_path = Path(collection_path)
    if raw_collection_path.is_symlink() or not raw_collection_path.is_file():
        die("result collection must be a nonsymlink regular file")
    path = raw_collection_path.resolve()
    collection = read_json(path)
    if not isinstance(collection, Mapping) or not isinstance(collection.get("compact_groups"),
                                                              Mapping):
        die("result collection lacks compact_groups object")
    request_root_value = collection.get("request_root")
    if not isinstance(request_root_value, str) or \
            not Path(request_root_value).is_absolute():
        die("result collection lacks an absolute declared request_root")
    request_root_path = Path(request_root_value)
    if request_root_path.is_symlink() or not request_root_path.is_dir():
        die("result collection request_root is absent, symlinked, or not a directory")
    request_root = request_root_path.resolve()
    mapping = collection["compact_groups"]
    if set(mapping) != set(REQUIRED_GROUP_KEYS):
        die("result collection compact_groups keys differ from the exact nine groups")
    identities: dict[str, str] = {}
    resolved: set[Path] = set()
    for key in REQUIRED_GROUP_KEYS:
        relative = mapping[key]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            die(f"compact group {key} path must be a nonempty relative string")
        raw_group_path = request_root / relative
        if raw_group_path.is_symlink():
            die(f"compact group {key} path is symlinked")
        group_path = raw_group_path.resolve()
        if request_root not in group_path.parents:
            die(f"compact group {key} escapes declared request_root")
        if group_path in resolved:
            die("two compact group keys resolve to one group.json")
        resolved.add(group_path)
        loaded = load_compact_group(group_path)
        actual = f"{loaded.group['obsnum']}:{loaded.group['array']}"
        if actual != key:
            die(f"compact group mapping identity differs: {key} -> {actual}")
        identities[key] = sha256_file(group_path)
    return identities


def _validate_expansion_request(request: Any, metadata: Mapping[str, Any]
                                ) -> dict[str, Any]:
    keys = {
        "schema_version", "request_id", "candidate_sha", "campaign_revision",
        "raw_input_manifest_sha256", "trigger", "target", "max_terms",
        "full_population",
    }
    node = dict(_require_exact_keys(request, keys, "discrepancy request"))
    exact = {
        "schema_version": REQUEST_SCHEMA,
        "candidate_sha": CANDIDATE_SHA,
        "campaign_revision": CAMPAIGN_REVISION,
        "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
        "full_population": False,
    }
    for name, expected in exact.items():
        if node[name] != expected:
            die(f"discrepancy request {name} differs")
    request_id = node["request_id"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128 or \
            any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
                for c in request_id):
        die("discrepancy request_id is invalid")
    trigger = _require_exact_keys(node["trigger"], {"kind", "name"},
                                  "discrepancy trigger")
    if trigger["kind"] not in ("named_discrepancy", "named_reauditor") or \
            not isinstance(trigger["name"], str) or not trigger["name"].strip() or \
            len(trigger["name"]) > 256:
        die("focused expansion requires one named discrepancy or re-auditor")
    max_terms = node["max_terms"]
    if not isinstance(max_terms, int) or isinstance(max_terms, bool) or \
            not (1 <= max_terms <= MAX_EXPANSION_TERMS):
        die(f"max_terms must be within 1..{MAX_EXPANSION_TERMS}")
    target = node["target"]
    if not isinstance(target, Mapping):
        die("discrepancy target must be an object")
    common = {"kind", "obsnum", "array", "network"}
    if not common.issubset(target):
        die("discrepancy target lacks observation/array/network specificity")
    if target["obsnum"] != metadata["obsnum"] or target["array"] != metadata["array"]:
        die("discrepancy target observation/array differs from source")
    active_networks = {int(d["network"]) for d in metadata["detector_order"]}
    if target["network"] not in active_networks:
        die("discrepancy target network is not active in the source")
    if target["kind"] == "detector_sequence":
        expected = common | {"scan_identity", "detector_identity"}
        if set(target) != expected:
            die("detector-sequence target fields differ")
        scans = [s for s in metadata["scan_order"]
                 if s["scan_identity"] == target["scan_identity"]]
        detectors = [d for d in metadata["detector_order"]
                     if d["detector_identity"] == target["detector_identity"] and
                     d["network"] == target["network"]]
        if len(scans) != 1 or len(detectors) != 1:
            die("detector-sequence target is absent or ambiguous")
    elif target["kind"] == "pixel":
        expected = common | {"row", "col"}
        if set(target) != expected:
            die("pixel target fields differ")
        rows = int(metadata["map_shape"]["rows"])
        cols = int(metadata["map_shape"]["cols"])
        if any(not isinstance(target[name], int) or isinstance(target[name], bool)
               for name in ("row", "col")) or not (0 <= target["row"] < rows) or \
                not (0 <= target["col"] < cols):
            die("pixel discrepancy target is outside the map")
    else:
        die("focused expansion target must be detector_sequence or pixel")
    return node


def _expansion_mask(metadata: Mapping[str, Any], target: Mapping[str, Any],
                    arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if target["kind"] == "detector_sequence":
        scan = next(s for s in metadata["scan_order"]
                    if s["scan_identity"] == target["scan_identity"])
        detector = next(d for d in metadata["detector_order"]
                        if d["detector_identity"] == target["detector_identity"] and
                        d["network"] == target["network"])
        return (arrays["scan_index"] == int(scan["scan_index"])) & \
            (arrays["detector_index"] == int(detector["detector_index"]))
    return (arrays["network"] == int(target["network"])) & \
        arrays["geometric_in_bounds"].astype(bool) & \
        (arrays["row"] == int(target["row"])) & \
        (arrays["col"] == int(target["col"]))


def _expansion_record_bytes(ordinals: np.ndarray, arrays: Mapping[str, np.ndarray],
                            interval: float) -> bytes:
    dtype = np.dtype([
        ("ordinal", "<i8"), ("row", "<i8"), ("col", "<i8"),
        ("scan_index", "<i8"), ("detector_index", "<i8"),
        ("sample_index", "<i8"), ("network", "<i8"),
        ("geometric_in_bounds", "u1"), ("detector_apt_flagged", "u1"),
        ("sample_flagged", "u1"), ("upstream_eligible", "u1"),
        ("coefficient", "<f8"), ("sample_signal", "<f8"),
        ("sample_kernel", "<f8"), ("sample_interval_s", "<f8"),
    ], align=False)
    records = np.empty(ordinals.size, dtype=dtype)
    records["ordinal"] = ordinals
    for name in (*TERM_INT64, *TERM_UINT8, *TERM_FLOAT64):
        records[name] = arrays[name]
    records["sample_interval_s"] = interval
    return records.tobytes(order="C")


def plan_expansion(source: Path, request_path: Path, output_path: Path,
                   chunk_size: int = 262144, *,
                   authority_path: Path | None = None,
                   resource_record_path: Path | None = None,
                   resource_inventory_path: Path | None = None,
                   governed_roots: Sequence[Path] = (),
                   resource_now: datetime | None = None) -> Path:
    output_path = _new_output_path(output_path, "expansion plan destination")
    request_path = _nonsymlink_regular(request_path, "discrepancy request")
    with _adapter_context(source, authority_path) as adapter:
        request = _validate_expansion_request(read_json(request_path), adapter.metadata)
        gate_values = (resource_record_path, resource_inventory_path)
        if authority_path is None:
            if any(value is not None for value in gate_values) or governed_roots:
                die("local fixture expansion does not accept operational resource records")
        else:
            if any(value is None for value in gate_values) or not governed_roots:
                die("direct candidate expansion planning requires a fresh resource "
                    "record, inventory, and exact governed roots")
            validate_resource_gate(
                resource_record_path, resource_inventory_path, governed_roots,
                f"focused-expansion-plan:{request['request_id']}", output_path,
                now=resource_now)
        request_sha = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        digest = _domain_hash("focused-expansion-selection", {
            **_digest_header(adapter.metadata),
            "request_sha256": request_sha,
            "target": request["target"],
        })
        count = 0
        interval = float(adapter.metadata["_sample_interval"])
        for chunk in adapter.iter_chunks(chunk_size):
            expected = _expected_identities(adapter.metadata, chunk.start, chunk.stop)
            if not all(np.array_equal(chunk.arrays[name], wanted)
                       for name, wanted in zip(
                           ("scan_index", "detector_index", "sample_index"), expected)):
                die("focused-expansion source is not in canonical term order")
            mask = _expansion_mask(adapter.metadata, request["target"], chunk.arrays)
            selected_count = int(mask.sum())
            if selected_count:
                ordinals = np.arange(chunk.start, chunk.stop, dtype=np.int64)[mask]
                selected = {name: value[mask] for name, value in chunk.arrays.items()}
                digest.update(_expansion_record_bytes(ordinals, selected, interval))
                count += selected_count
                if count > request["max_terms"]:
                    die("focused expansion exceeds the request's explicit max_terms")
        if count <= 0:
            die("focused expansion target selected no primitive term")
        plan = {
            "schema_version": EXPANSION_PLAN_SCHEMA,
            "candidate_sha": CANDIDATE_SHA,
            "campaign_revision": CAMPAIGN_REVISION,
            "raw_input_manifest_sha256": adapter.metadata["raw_input_manifest_sha256"],
            "source_stream_sha256": adapter.source_sha256,
            "producer_authority_sha256": (
                sha256_file(adapter.authority_path)
                if isinstance(adapter, CandidatePtcNetCDFAdapter)
                else adapter.source_sha256),
            "request_sha256": request_sha,
            "request": request,
            "planned_terms": count,
            "maximum_terms": request["max_terms"],
            "selection_sha256": digest.hexdigest(),
            "uncompressed_payload_bytes": count * 92,
            "pass": "first-of-two-count-and-digest-only",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, plan)
    return output_path


def _validate_expansion_plan(plan: Any) -> dict[str, Any]:
    keys = {
        "schema_version", "candidate_sha", "campaign_revision",
        "raw_input_manifest_sha256", "source_stream_sha256",
        "producer_authority_sha256", "request_sha256",
        "request", "planned_terms", "maximum_terms", "selection_sha256",
        "uncompressed_payload_bytes", "pass",
    }
    node = dict(_require_exact_keys(plan, keys, "focused expansion plan"))
    if node["schema_version"] != EXPANSION_PLAN_SCHEMA or \
            node["candidate_sha"] != CANDIDATE_SHA or \
            node["campaign_revision"] != CAMPAIGN_REVISION or \
            node["pass"] != "first-of-two-count-and-digest-only":
        die("focused expansion plan identity differs")
    for name in ("raw_input_manifest_sha256", "source_stream_sha256",
                 "producer_authority_sha256",
                 "request_sha256", "selection_sha256"):
        _require_hex_digest(node[name], f"expansion plan {name}")
    if hashlib.sha256(canonical_json_bytes(node["request"])).hexdigest() != \
            node["request_sha256"]:
        die("focused expansion plan request digest differs")
    for name in ("planned_terms", "maximum_terms", "uncompressed_payload_bytes"):
        if not isinstance(node[name], int) or isinstance(node[name], bool) or node[name] <= 0:
            die(f"focused expansion plan {name} is invalid")
    if node["planned_terms"] > node["maximum_terms"] or \
            node["maximum_terms"] > MAX_EXPANSION_TERMS or \
            node["uncompressed_payload_bytes"] != node["planned_terms"] * 92:
        die("focused expansion plan is unbounded or byte count differs")
    return node


def emit_expansion(source: Path, plan_path: Path, output_path: Path,
                   chunk_size: int = 262144, *,
                   authority_path: Path | None = None,
                   resource_record_path: Path | None = None,
                   resource_inventory_path: Path | None = None,
                   governed_roots: Sequence[Path] = (),
                   resource_now: datetime | None = None) -> Path:
    output_path = _new_output_path(output_path, "focused expansion destination")
    plan_path = _nonsymlink_regular(plan_path, "focused expansion plan")
    plan = _validate_expansion_plan(read_json(plan_path))
    collected: dict[str, list[np.ndarray]] = {
        "ordinal": [], **{name: [] for name in (*TERM_INT64, *TERM_UINT8,
                                                  *TERM_FLOAT64)}
    }
    with _adapter_context(source, authority_path) as adapter:
        authority_sha256 = (
            sha256_file(adapter.authority_path)
            if isinstance(adapter, CandidatePtcNetCDFAdapter)
            else adapter.source_sha256)
        if adapter.source_sha256 != plan["source_stream_sha256"] or \
                authority_sha256 != plan["producer_authority_sha256"] or \
                adapter.metadata["raw_input_manifest_sha256"] != \
                plan["raw_input_manifest_sha256"]:
            die("focused expansion source changed between pass one and pass two")
        request = _validate_expansion_request(plan["request"], adapter.metadata)
        gate_values = (resource_record_path, resource_inventory_path)
        if authority_path is None:
            if any(value is not None for value in gate_values) or governed_roots:
                die("local fixture expansion does not accept operational resource records")
        else:
            if any(value is None for value in gate_values) or not governed_roots:
                die("direct candidate expansion emission requires a fresh resource "
                    "record, inventory, and exact governed roots")
            validate_resource_gate(
                resource_record_path, resource_inventory_path, governed_roots,
                f"focused-expansion:{request['request_id']}", output_path,
                now=resource_now)
        digest = _domain_hash("focused-expansion-selection", {
            **_digest_header(adapter.metadata),
            "request_sha256": plan["request_sha256"],
            "target": request["target"],
        })
        count = 0
        interval = float(adapter.metadata["_sample_interval"])
        for chunk in adapter.iter_chunks(chunk_size):
            expected = _expected_identities(adapter.metadata, chunk.start, chunk.stop)
            if not all(np.array_equal(chunk.arrays[name], wanted)
                       for name, wanted in zip(
                           ("scan_index", "detector_index", "sample_index"), expected)):
                die("focused-expansion source changed order between passes")
            mask = _expansion_mask(adapter.metadata, request["target"], chunk.arrays)
            if not np.any(mask):
                continue
            ordinals = np.arange(chunk.start, chunk.stop, dtype=np.int64)[mask]
            selected = {name: value[mask] for name, value in chunk.arrays.items()}
            digest.update(_expansion_record_bytes(ordinals, selected, interval))
            collected["ordinal"].append(ordinals)
            for name in (*TERM_INT64, *TERM_UINT8, *TERM_FLOAT64):
                collected[name].append(selected[name])
            count += int(mask.sum())
            if count > plan["maximum_terms"]:
                die("focused expansion exceeded its first-pass bound")
        if count != plan["planned_terms"] or digest.hexdigest() != plan["selection_sha256"]:
            die("focused expansion second pass differs in count or digest")
        arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
        arrays["sample_interval_s"] = np.full(count, interval, dtype=np.float64)
        metadata = {
            "schema_version": EXPANSION_SCHEMA,
            "candidate_sha": CANDIDATE_SHA,
            "campaign_revision": CAMPAIGN_REVISION,
            "raw_input_manifest_sha256": adapter.metadata["raw_input_manifest_sha256"],
            "source_stream_sha256": adapter.source_sha256,
            "producer_authority_sha256": plan["producer_authority_sha256"],
            "request_sha256": plan["request_sha256"],
            "selection_sha256": plan["selection_sha256"],
            "term_count": count,
            "bounded": True,
            "full_population": False,
            "pass": "second-of-two-emit-exact-selection",
        }
        arrays["metadata_json"] = np.array(canonical_json_bytes(metadata).decode("ascii"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_npz(output_path, arrays)
    return output_path


def write_self_check_fixture(path: Path, obsnum: int = 152389,
                             array: str = "a1100") -> dict[str, Any]:
    """Create a deterministic, complete full-PTC-like primitive fixture."""
    if obsnum not in (152389, 152390, 152392) or array not in ARRAY_NETWORKS:
        die("self-check fixture identity is outside the nine groups")
    networks = sorted(ARRAY_NETWORKS[array])
    if len(networks) >= 3:
        detector_states = [
            (networks[0], False), (networks[0], True), (networks[0], False),
            (networks[1], False), (networks[2], True),
        ]
    else:
        detector_states = [
            (networks[0], False), (networks[0], True), (networks[0], False),
            (networks[1], False), (networks[1], True),
        ]
    scans = [
        {"scan_index": 0, "scan_identity": f"{obsnum}:scan:1",
         "output_scan_index": 1, "sample_count": 2},
        {"scan_index": 1, "scan_identity": f"{obsnum}:scan:2",
         "output_scan_index": 2, "sample_count": 3},
        {"scan_index": 2, "scan_identity": f"{obsnum}:scan:3",
         "output_scan_index": 3, "sample_count": 2},
    ]
    detectors = []
    for index, (network, flagged) in enumerate(detector_states):
        uid = str(1000 + index % 3)
        kids_tone = index
        detectors.append({
            "detector_index": index,
            "apt_row_index": index,
            "network": network,
            "kids_tone": kids_tone,
            "detector_uid": uid,
            "detector_identity":
                f"nw={network};kids_tone={kids_tone};uid={uid};apt_row_index={index}",
            "apt_flagged": flagged,
        })
    count = sum(s["sample_count"] for s in scans) * len(detectors)
    metadata = {
        "schema_version": STREAM_SCHEMA,
        "adapter": ADAPTER,
        "capture_output_mode": "full",
        "capture_detector_selection": "all",
        "candidate_sha": CANDIDATE_SHA,
        "campaign_revision": CAMPAIGN_REVISION,
        "raw_input_manifest_sha256": hashlib.sha256(
            f"raw:{obsnum}:{array}".encode("ascii")).hexdigest(),
        "producer_identity": "compact-evidence-self-check-fixture-v1",
        "capture_ptc_sha256": hashlib.sha256(
            f"ptc:{obsnum}:{array}".encode("ascii")).hexdigest(),
        "realized_raw_timestream_provenance_sha256": hashlib.sha256(
            f"raw-provenance:{obsnum}:{array}".encode("ascii")).hexdigest(),
        "realized_mapmaking_provenance_sha256": hashlib.sha256(
            f"map-provenance:{obsnum}:{array}".encode("ascii")).hexdigest(),
        "mapmaking_bundle_identity_digest":
            "canonical-hexfloat-sha256-v1:" + hashlib.sha256(
                f"map:{obsnum}:{array}".encode("ascii")).hexdigest(),
        "obsnum": obsnum,
        "array": array,
        "map_shape": {"rows": 4, "cols": 5},
        "map_pixel_size_rad": exact_float_node(
            2.0e-5, "realized_mapmaking.effective_pixel_size_rad"),
        "native_fsmp_hz": exact_float_node(488.0, "telescope.fsmp"),
        "effective_d_fsmp_hz": exact_float_node(122.0, "telescope.d_fsmp"),
        "scan_order": scans,
        "detector_order": detectors,
        "primitive_term_count": count,
        "term_order": TERM_ORDER,
    }
    values: dict[str, list[Any]] = {name: [] for name in
                                    (*TERM_INT64, *TERM_UINT8, *TERM_FLOAT64)}
    ordinal = 0
    for scan in scans:
        for detector in detectors:
            for sample in range(scan["sample_count"]):
                geometric = ordinal % 11 != 0
                row = (scan["scan_index"] + detector["detector_index"] + sample) % 4
                col = (2 * detector["detector_index"] + sample) % 5
                if not geometric:
                    row = -1
                    col = -1
                sample_flagged = ordinal % 13 == 0
                apt_flagged = bool(detector["apt_flagged"])
                upstream = geometric and not apt_flagged and not sample_flagged
                coefficient = 1.0 + 0.25 * ((detector["detector_index"] + sample) % 4)
                if ordinal % 9 == 0:
                    coefficient = 0.0
                values["row"].append(row)
                values["col"].append(col)
                values["scan_index"].append(scan["scan_index"])
                values["detector_index"].append(detector["detector_index"])
                values["sample_index"].append(sample)
                values["network"].append(detector["network"])
                values["geometric_in_bounds"].append(int(geometric))
                values["detector_apt_flagged"].append(int(apt_flagged))
                values["sample_flagged"].append(int(sample_flagged))
                values["upstream_eligible"].append(int(upstream))
                values["coefficient"].append(coefficient)
                values["sample_signal"].append(-2.0 + 0.125 * ordinal)
                values["sample_kernel"].append(0.25 + 0.03125 * ordinal)
                ordinal += 1
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.array(canonical_json_bytes(metadata).decode("ascii")),
        **{name: np.asarray(values[name], dtype=np.int64) for name in TERM_INT64},
        **{name: np.asarray(values[name], dtype=np.uint8) for name in TERM_UINT8},
        **{name: np.asarray(values[name], dtype=np.float64) for name in TERM_FLOAT64},
        "realization_signs": boost_mt19937_scan_signs(len(scans)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_npz(path, arrays)
    return metadata


def full_fixture_reference(source: Path, coverage_cut: float) -> dict[str, Any]:
    """Independent full-term reference used only for small-fixture parity."""
    with FullPtcFixtureAdapter(source) as adapter:
        metadata = adapter.metadata
        rows = int(metadata["map_shape"]["rows"])
        cols = int(metadata["map_shape"]["cols"])
        interval = float(metadata["_sample_interval"])
        shape = (rows, cols)
        geometric_hits = np.zeros(shape, dtype=np.int64)
        contributing_hits = np.zeros(shape, dtype=np.int64)
        upstream_exposure = np.zeros(shape, dtype=np.float64)
        retained = np.zeros(shape, dtype=np.float64)
        numerator = np.zeros(shape, dtype=np.float64)
        weight = np.zeros(shape, dtype=np.float64)
        kernel_numerator = np.zeros(shape, dtype=np.float64)
        noise_numerator = np.zeros((*shape, REALIZATIONS), dtype=np.float64)
        for chunk in adapter.iter_chunks(7):
            a = chunk.arrays
            for offset in range(chunk.stop - chunk.start):
                if not bool(a["geometric_in_bounds"][offset]):
                    continue
                location = (int(a["row"][offset]), int(a["col"][offset]))
                geometric_hits[location] += 1
                if not bool(a["upstream_eligible"][offset]):
                    continue
                upstream_exposure[location] += interval
                coefficient = float(a["coefficient"][offset])
                if not math.isfinite(coefficient) or coefficient <= 0.0:
                    continue
                signal = coefficient * float(a["sample_signal"][offset])
                kernel = coefficient * float(a["sample_kernel"][offset])
                contributing_hits[location] += 1
                retained[location] += interval
                numerator[location] += signal
                weight[location] += coefficient
                kernel_numerator[location] += kernel
                scan = int(a["scan_index"][offset])
                noise_numerator[location] += adapter.realization_signs[scan] * signal
    normalization = threshold_selection(weight, coverage_cut / 10.0)
    norm = np.isfinite(weight) & (weight > 0.0) & \
        (weight >= normalization["threshold"])
    denominator = np.where(weight > 0.0, weight, 1.0)
    final_weight = np.where(norm, weight, 0.0)
    signal = np.where(norm, numerator / denominator, 0.0)
    kernel = np.where(norm, kernel_numerator / denominator, 0.0)
    noise = np.where(norm[..., None], noise_numerator / denominator[..., None], 0.0)
    final_retained = np.where(norm, retained, 0.0)
    science_policy = threshold_selection(final_weight, coverage_cut)
    policy = np.isfinite(final_weight) & (final_weight > 0.0) & \
        (final_weight >= science_policy["threshold"])
    valid = norm & policy & np.isfinite(signal) & np.isfinite(final_weight) & \
        (final_weight > 0.0) & np.isfinite(kernel) & np.all(np.isfinite(noise), axis=-1)
    return {
        "planes": {
            "signal_I": signal, "weight_I": final_weight, "kernel_I": kernel,
            "geometric_hits_I": geometric_hits,
            "contributing_hits_I": contributing_hits,
            "upstream_eligible_exposure_I": upstream_exposure,
            "retained_exposure_I": final_retained,
            "normalization_support_I": norm.astype(np.uint8),
            "science_policy_support_I": policy.astype(np.uint8),
            "science_valid_I": valid.astype(np.uint8),
            "coverage_I": final_retained.copy(),
            "coverage_bool_I": policy.astype(np.uint8),
        },
        "noise": noise,
    }


def assert_compact_parity(compact: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    for name, expected in reference["planes"].items():
        actual = compact["planes"][name]
        if expected.dtype.kind in "iub":
            if not np.array_equal(actual, expected):
                die(f"compact/full fixture integer parity failed for {name}")
        elif not np.allclose(actual, expected, atol=2.0e-8, rtol=1.0e-10,
                             equal_nan=True):
            die(f"compact/full fixture numerical parity failed for {name}")
    if not np.allclose(compact["noise"], reference["noise"],
                       atol=2.0e-8, rtol=1.0e-10, equal_nan=True):
        die("compact/full fixture numerical parity failed for realizations")


def self_check() -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sci-map-compact-self-check-") as raw:
        root = Path(raw)
        source = root / "source-152389-a1100.npz"
        metadata = write_self_check_fixture(source)
        group_one = produce_compact_group(source, root / "group-one", chunk_size=1)
        group_two = produce_compact_group(source, root / "group-two", chunk_size=11)
        for filename in ("group.json", "sufficient-statistics.npz",
                         "deterministic-trace.npz", "trace-selection.json"):
            if (group_one.parent / filename).read_bytes() != \
                    (group_two.parent / filename).read_bytes():
                die(f"chunk-size invariance failed for {filename}")
        compact = reconstruct_compact_group(group_one, 0.1)
        reference = full_fixture_reference(source, 0.1)
        assert_compact_parity(compact, reference)

        request = {
            "schema_version": REQUEST_SCHEMA,
            "request_id": "self-check-named-discrepancy",
            "candidate_sha": CANDIDATE_SHA,
            "campaign_revision": CAMPAIGN_REVISION,
            "raw_input_manifest_sha256": metadata["raw_input_manifest_sha256"],
            "trigger": {"kind": "named_discrepancy", "name": "F010-self-check"},
            "target": {
                "kind": "detector_sequence", "obsnum": 152389,
                "array": "a1100", "network": metadata["detector_order"][0]["network"],
                "scan_identity": metadata["scan_order"][1]["scan_identity"],
                "detector_identity":
                    metadata["detector_order"][0]["detector_identity"],
            },
            "max_terms": 16,
            "full_population": False,
        }
        request_path = root / "request.json"
        plan_path = root / "plan.json"
        expansion_path = root / "expansion.npz"
        write_json(request_path, request)
        plan_expansion(source, request_path, plan_path, chunk_size=2)
        emit_expansion(source, plan_path, expansion_path, chunk_size=9)

        groups_root = root / "nine"
        mapping: dict[str, str] = {}
        for key in REQUIRED_GROUP_KEYS:
            obs_text, array = key.split(":", 1)
            obsnum = int(obs_text)
            fixture = groups_root / "sources" / f"{obsnum}-{array}.npz"
            write_self_check_fixture(fixture, obsnum, array)
            destination = groups_root / "groups" / f"{obsnum}-{array}"
            produce_compact_group(fixture, destination, chunk_size=13)
            mapping[key] = str(destination.relative_to(groups_root) / "group.json")
        collection = groups_root / "collection.json"
        write_json(collection, {"request_root": str(groups_root.resolve()),
                                "compact_groups": mapping})
        identities = verify_nine_group_mapping(collection)
        stored_bytes = sum(
            path.stat().st_size for path in group_one.parent.iterdir() if path.is_file())
        return {
            "status": "pass",
            "schema_version": GROUP_SCHEMA,
            "fixture_primitive_terms": metadata["primitive_term_count"],
            "fixture_compact_stored_bytes": stored_bytes,
            "nine_group_count": len(identities),
            "chunk_sizes_compared": [1, 11],
            "focused_expansion_terms": read_json(plan_path)["planned_terms"],
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "local synthetic evidence only; not Unity evidence",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="produce one compact group")
    produce.add_argument("--source", type=Path, required=True)
    produce.add_argument(
        "--authority", type=Path,
        help="candidate PTC producer authority (required for direct NetCDF input)")
    produce.add_argument("--resource-record", type=Path)
    produce.add_argument("--resource-inventory", type=Path)
    produce.add_argument("--governed-root", type=Path, action="append", default=[])
    produce.add_argument("--output-dir", type=Path, required=True)
    produce.add_argument("--chunk-size", type=int, default=262144)
    verify = commands.add_parser("verify", help="verify one compact group")
    verify.add_argument("--group", type=Path, required=True)
    reconstruct = commands.add_parser("reconstruct", help="verify and reconstruct one group")
    reconstruct.add_argument("--group", type=Path, required=True)
    reconstruct.add_argument("--coverage-cut", type=float, required=True)
    verify_nine = commands.add_parser("verify-nine", help="verify exact nine-group mapping")
    verify_nine.add_argument("--collection", type=Path, required=True)
    plan = commands.add_parser("plan-expansion", help="first pass for named expansion")
    plan.add_argument("--source", type=Path, required=True)
    plan.add_argument(
        "--authority", type=Path,
        help="candidate PTC producer authority (required for direct NetCDF input)")
    plan.add_argument("--resource-record", type=Path)
    plan.add_argument("--resource-inventory", type=Path)
    plan.add_argument("--governed-root", type=Path, action="append", default=[])
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--chunk-size", type=int, default=262144)
    emit = commands.add_parser("emit-expansion", help="second pass for named expansion")
    emit.add_argument("--source", type=Path, required=True)
    emit.add_argument(
        "--authority", type=Path,
        help="candidate PTC producer authority (required for direct NetCDF input)")
    emit.add_argument("--resource-record", type=Path)
    emit.add_argument("--resource-inventory", type=Path)
    emit.add_argument("--governed-root", type=Path, action="append", default=[])
    emit.add_argument("--plan", type=Path, required=True)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--chunk-size", type=int, default=262144)
    commands.add_parser("self-check", help="run local synthetic self-check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "produce":
            result: Any = {"group": str(produce_compact_group(
                args.source, args.output_dir, args.chunk_size,
                authority_path=args.authority,
                resource_record_path=args.resource_record,
                resource_inventory_path=args.resource_inventory,
                governed_roots=args.governed_root))}
        elif args.command == "verify":
            loaded = load_compact_group(args.group)
            result = {"status": "pass", "group": str(loaded.path),
                      "identity": f"{loaded.group['obsnum']}:{loaded.group['array']}"}
        elif args.command == "reconstruct":
            reconstruction = reconstruct_compact_group(args.group, args.coverage_cut)
            result = {
                "status": "pass",
                "identity": f"{reconstruction['group']['obsnum']}:"
                            f"{reconstruction['group']['array']}",
                "plane_count": len(reconstruction["planes"]),
                "realization_count": reconstruction["noise"].shape[-1],
            }
        elif args.command == "verify-nine":
            result = {"status": "pass",
                      "compact_groups": verify_nine_group_mapping(args.collection)}
        elif args.command == "plan-expansion":
            result = {"plan": str(plan_expansion(
                args.source, args.request, args.output, args.chunk_size,
                authority_path=args.authority,
                resource_record_path=args.resource_record,
                resource_inventory_path=args.resource_inventory,
                governed_roots=args.governed_root))}
        elif args.command == "emit-expansion":
            result = {"expansion": str(emit_expansion(
                args.source, args.plan, args.output, args.chunk_size,
                authority_path=args.authority,
                resource_record_path=args.resource_record,
                resource_inventory_path=args.resource_inventory,
                governed_roots=args.governed_root))}
        elif args.command == "self-check":
            result = self_check()
        else:  # pragma: no cover - argparse prevents this path
            die("unknown command")
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
