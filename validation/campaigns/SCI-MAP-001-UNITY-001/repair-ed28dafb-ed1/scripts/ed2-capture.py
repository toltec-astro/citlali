#!/usr/bin/env python3
"""Fail-closed, file-only preparation for the MAP-UNITY-ED2 capture lane.

This tool never contacts Unity, invokes Citlali or TolProj, submits work, or
deletes retained evidence.  It is run explicitly by a human on Unity after
the package and owner values have been accepted.  Every output is single-shot.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import yaml


REQUEST_ID = "SCI-MAP-001-UNITY-001"
REVISION = "repair-sha-ed28dafb-ed1-2026-08-02"
CANDIDATE = "ed28dafb37f9113c0d3c95297148157129a90886"
CANDIDATE_TREE = "cf75c36557178f351fb62781108a6f4b41b19225"
CEILING = 214748364800
ARRAYS = ("a1100", "a1400", "a2000")
ARRAY_IDS = {"a1100": 0, "a1400": 1, "a2000": 2}
ARRAY_NETWORKS = {
    "a1100": tuple(range(0, 7)),
    "a1400": tuple(range(7, 11)),
    "a2000": tuple(range(11, 13)),
}
SOURCE_ROLES = (
    "raw_timestream", "kids_fit_report", "apt", "calibration",
    "pointing_support", "projection_authority", "sample_rate_authority",
    "fwhm_authority", "target_authority",
)
CAPTURES = {
    "CAP-POINT": {
        "mode": "point", "observations": [152389], "support": [],
        "manifest_support": [152389], "raw_observations": [152389],
    },
    "CAP-SCIENCE": {
        "mode": "science", "observations": [152390, 152392],
        "support": [152389, 152391, 152393],
        "manifest_support": [152389, 152391, 152393],
        "raw_observations": [152389, 152390, 152391, 152392, 152393],
    },
}
ALLOWLIST = {
    "timestream.processed_time_chunk.output.enabled": True,
    "timestream.processed_time_chunk.output.mode": "full",
    "timestream.processed_time_chunk.output.indices": "all",
}
NUMBERED = {
    "point": [
        "40_setup.yaml", "60_pointing_internal_policy.yaml",
        "71_pointing_runtime.yaml", "72_pointing_observation.yaml",
        "81_pointing_defaults.yaml", "82_pointing_products.yaml",
        "90_pointing_advanced_overrides.yaml",
        "99_pointing_expert_overrides.yaml",
        "99_zz_tolproj_submission_runtime.yaml",
    ],
    "science": [
        "40_setup.yaml", "60_science_internal_policy.yaml",
        "71_science_runtime.yaml", "72_science_observation.yaml",
        "81_science_defaults.yaml", "82_science_products.yaml",
        "90_science_advanced_overrides.yaml",
        "99_science_expert_overrides.yaml",
        "99_zz_tolproj_submission_runtime.yaml",
    ],
}
CAPTURE_PROJECTION_BYTES = {
    # This is an owner-reviewed planning estimate for the initial lightweight
    # project/staging/configuration step.  It is deliberately not represented
    # as a serialized full-PTC upper bound.
    "PREPARE-STAGING": 144244000824,
    "CAP-POINT": 1079515834,
    "CAP-SCIENCE": 125984615806,
}
MIB = 1024 * 1024
GIB = 1024 * MIB


class CaptureError(RuntimeError):
    pass


def die(message: str) -> NoReturn:
    raise CaptureError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CaptureError(f"cannot read JSON {path}: {exc}") from exc


def read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CaptureError(f"cannot read YAML {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        die(f"{label} must be a nonsymlink regular file: {path}")
    return path.resolve()


def directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        die(f"{label} must be an existing nonsymlink absolute directory: {path}")
    resolved = path.resolve()
    if str(path) != str(resolved):
        die(f"{label} must be a canonical resolved path: {path}")
    return resolved


def new_path(path: Path, label: str) -> Path:
    if path.exists() or path.is_symlink():
        die(f"{label} already exists: {path}")
    if not path.is_absolute():
        die(f"{label} must be absolute: {path}")
    return path.resolve(strict=False)


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> Path:
    destination = new_path(path, "output")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        die(f"output parent must be an existing nonsymlink directory: {destination.parent}")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    destination.chmod(mode)
    return destination


def exact_float(value: Any, authority: str) -> dict[str, str]:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        die(f"{authority} must be finite and positive")
    return {"decimal": format(number, ".17g"), "hex": number.hex(),
            "authority": authority}


def raw_exact_float(value: Any) -> dict[str, str]:
    number = float(value)
    if not math.isfinite(number):
        die("raw-manifest exact float must be finite")
    return {"numeric": format(number, ".17g"), "hex": number.hex(),
            "encoding": "binary64-max-digits10-and-c99-hexfloat"}


def parse_exact(node: Any, label: str, *, authority: str | None = None) -> float:
    if not isinstance(node, Mapping):
        die(f"{label} must be an exact-float object")
    if set(node) == {"decimal", "hex", "authority"}:
        if authority is not None and node["authority"] != authority:
            die(f"{label} authority differs")
        decimal_key = "decimal"
    elif set(node) == {"numeric", "hex", "encoding"}:
        if node["encoding"] != "binary64-max-digits10-and-c99-hexfloat":
            die(f"{label} encoding differs")
        decimal_key = "numeric"
    else:
        die(f"{label} exact-float fields differ")
    try:
        decimal = float(node[decimal_key])
        hexadecimal = float.fromhex(str(node["hex"]))
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{label} exact-float encoding is invalid") from exc
    if not math.isfinite(decimal) or struct.pack("=d", decimal) != struct.pack("=d", hexadecimal):
        die(f"{label} decimal/hex values differ or are nonfinite")
    return decimal


def require_capture(capture_id: str) -> dict[str, Any]:
    if capture_id not in CAPTURES:
        die(f"unknown capture identity: {capture_id}")
    return dict(CAPTURES[capture_id])


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _raw_selection(path: Path, capture_id: str) -> list[dict[str, Any]]:
    value = read_json(regular(path, "raw selection"))
    if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "capture_id", "records"} \
            or value["schema_version"] != "sci-map-001-raw-selection-v1" \
            or value["capture_id"] != capture_id or not isinstance(value["records"], list):
        die("raw selection identity/shape differs")
    observations = require_capture(capture_id)["raw_observations"]
    rows: list[dict[str, Any]] = []
    basenames: set[str] = set()
    for index, item in enumerate(value["records"]):
        if not isinstance(item, Mapping) or set(item) != {"observation", "basename"}:
            die(f"raw selection record {index} fields differ")
        obsnum, basename = item["observation"], item["basename"]
        if obsnum not in observations or not isinstance(basename, str) \
                or Path(basename).name != basename or basename in ("", ".", "..") \
                or basename in basenames or not re.search(
                    rf"(?<![0-9]){obsnum}(?![0-9])", basename):
            die(f"raw selection record {index} identity/basename differs")
        basenames.add(basename)
        rows.append({"observation": obsnum, "basename": basename})
    if not rows or sorted({row["observation"] for row in rows}) != observations:
        die("raw selection does not cover the exact target observations")
    return rows


def command_raw_manifest(args: argparse.Namespace) -> dict[str, Any]:
    capture = require_capture(args.capture_id)
    root = directory(args.canonical_root, "canonical raw root")
    if root != Path("/work/toltec"):
        die("canonical raw root must be the owner-verified /work/toltec")
    records = []
    resolved_seen: set[Path] = set()
    for selected in _raw_selection(args.selection, args.capture_id):
        matches = sorted(root.rglob(selected["basename"]))
        if len(matches) != 1:
            die(f"raw basename {selected['basename']} has {len(matches)} matches")
        source = regular(matches[0], f"canonical raw {selected['basename']}")
        if not _under(source, root) or source in resolved_seen:
            die(f"raw source escapes/repeats canonical root: {source}")
        before = source.stat()
        digest = sha256(source)
        after = source.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            die(f"raw source changed while hashing: {source}")
        resolved_seen.add(source)
        records.append({
            **selected, "resolved_target": str(source),
            "size_bytes": before.st_size, "device": before.st_dev,
            "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
            "sha256": digest,
        })
    manifest = {
        "schema_version": "sci-map-001-raw-link-manifest-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "capture_id": args.capture_id,
        "mode": capture["mode"], "canonical_raw_root": "/work/toltec",
        "records": records, "staging_policy": "individual-file-symlinks-only",
        "tolproj_copy_raw_after_staging": "prohibited",
    }
    write_new(args.output, pretty_json(manifest))
    return manifest


def _validate_raw_link_manifest(
        path: Path, expected_capture_id: str | None = None) -> dict[str, Any]:
    manifest = read_json(regular(path, "raw-link manifest"))
    required = {
        "schema_version", "request_id", "revision", "candidate_sha",
        "capture_id", "mode", "canonical_raw_root", "records",
        "staging_policy", "tolproj_copy_raw_after_staging",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required \
            or manifest["schema_version"] != "sci-map-001-raw-link-manifest-v1" \
            or manifest["request_id"] != REQUEST_ID or manifest["revision"] != REVISION \
            or manifest["candidate_sha"] != CANDIDATE \
            or manifest["staging_policy"] != "individual-file-symlinks-only" \
            or manifest["tolproj_copy_raw_after_staging"] != "prohibited":
        die("raw-link manifest identity differs")
    capture_id = str(manifest["capture_id"])
    capture = require_capture(capture_id)
    if expected_capture_id is not None and capture_id != expected_capture_id:
        die("raw-link manifest capture identity differs")
    if manifest["mode"] != capture["mode"] or manifest["canonical_raw_root"] != "/work/toltec":
        die("raw-link manifest capture/root differs")
    rows = manifest["records"]
    required_row = {
        "observation", "basename", "resolved_target", "size_bytes", "device",
        "inode", "mtime_ns", "sha256",
    }
    if not isinstance(rows, list) or not rows:
        die("raw-link manifest records are absent")
    basenames: set[str] = set()
    targets: set[Path] = set()
    root = Path("/work/toltec").resolve(strict=False)
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != required_row:
            die(f"raw-link manifest record {index} fields differ")
        obsnum, basename = raw["observation"], raw["basename"]
        if obsnum not in capture["raw_observations"] \
                or not isinstance(basename, str) \
                or Path(basename).name != basename \
                or basename in ("", ".", "..") \
                or basename in basenames \
                or re.search(rf"(?<![0-9]){obsnum}(?![0-9])", basename) is None:
            die(f"raw-link manifest record {index} observation/basename differs")
        source = regular(Path(raw["resolved_target"]), "manifest raw target")
        if not _under(source, root) or source.name != basename or source in targets:
            die(f"raw-link manifest target escapes/repeats canonical root: {source}")
        current = source.stat()
        integral = ("size_bytes", "device", "inode", "mtime_ns")
        if any(not isinstance(raw[name], int) or isinstance(raw[name], bool)
               for name in integral) \
                or raw["size_bytes"] <= 0 \
                or (current.st_dev, current.st_ino, current.st_size,
                    current.st_mtime_ns) != (
                        raw["device"], raw["inode"], raw["size_bytes"],
                        raw["mtime_ns"]) \
                or not isinstance(raw["sha256"], str) \
                or re.fullmatch(r"[0-9a-f]{64}", raw["sha256"]) is None \
                or sha256(source) != raw["sha256"]:
            die(f"raw-link manifest target identity/digest differs: {source}")
        basenames.add(basename)
        targets.add(source)
    if sorted({int(row["observation"]) for row in rows}) != \
            capture["raw_observations"]:
        die("raw-link manifest observation coverage differs")
    return dict(manifest)


def _validate_raw_link_staging(
        path: Path, manifest_path: Path,
        expected_capture_id: str) -> dict[str, Any]:
    source_manifest_path = regular(manifest_path, "raw-link source manifest")
    source_manifest = _validate_raw_link_manifest(
        source_manifest_path, expected_capture_id)
    node = read_json(regular(path, "raw-link staging manifest"))
    required = {
        "schema_version", "request_id", "revision", "candidate_sha",
        "capture_id", "source_manifest", "source_manifest_sha256",
        "destination", "records", "directory_symlinks", "copied_raw_files",
        "tolproj_copy_raw_after_staging",
    }
    recorded_source_manifest = regular(
        Path(str(node.get("source_manifest", ""))),
        "recorded raw-link source manifest") \
        if isinstance(node, Mapping) else source_manifest_path
    if not isinstance(node, Mapping) or set(node) != required \
            or node["schema_version"] != "sci-map-001-raw-link-staging-v1" \
            or node["request_id"] != REQUEST_ID or node["revision"] != REVISION \
            or node["candidate_sha"] != CANDIDATE \
            or node["capture_id"] != expected_capture_id \
            or recorded_source_manifest != source_manifest_path \
            or node["source_manifest_sha256"] != sha256(source_manifest_path) \
            or node["directory_symlinks"] is not False \
            or node["copied_raw_files"] is not False \
            or node["tolproj_copy_raw_after_staging"] != "prohibited":
        die("raw-link staging manifest identity/policy differs")
    destination = directory(Path(node["destination"]), "raw-link staging destination")
    rows = node["records"]
    source_rows = source_manifest["records"]
    required_row = {
        "observation", "basename", "link_path", "resolved_target",
        "size_bytes", "sha256",
    }
    if not isinstance(rows, list) or len(rows) != len(source_rows):
        die("raw-link staging record cardinality differs")
    for index, (row, source_row) in enumerate(zip(rows, source_rows)):
        if not isinstance(row, Mapping) or set(row) != required_row \
                or {key: row[key] for key in (
                    "observation", "basename", "resolved_target", "size_bytes",
                    "sha256")} != {key: source_row[key] for key in (
                        "observation", "basename", "resolved_target", "size_bytes",
                        "sha256")}:
            die(f"raw-link staging record {index} differs from source manifest")
        link = Path(row["link_path"])
        source = Path(row["resolved_target"]).resolve(strict=True)
        if not link.is_absolute() or link.parent.resolve(strict=True) != destination \
                or link.name != row["basename"] or not link.is_symlink() \
                or not Path(os.readlink(link)).is_absolute() \
                or link.resolve(strict=True) != source \
                or sha256(source) != row["sha256"]:
            die(f"raw-link staging record {index} live link/digest differs")
    return dict(node)


def _validate_authority_staging(
        path: Path, expected_capture_id: str) -> dict[str, Any]:
    node = read_json(regular(path, "authority staging manifest"))
    required = {
        "schema_version", "request_id", "revision", "candidate_sha",
        "capture_id", "records",
        "wholesale_legacy_reduction_used",
    }
    if not isinstance(node, Mapping) or set(node) != required \
            or node["schema_version"] != "sci-map-001-authority-staging-v1" \
            or node["request_id"] != REQUEST_ID or node["revision"] != REVISION \
            or node["candidate_sha"] != CANDIDATE \
            or node["capture_id"] != expected_capture_id \
            or node["wholesale_legacy_reduction_used"] is not False:
        die("authority staging manifest identity/policy differs")
    expected = ([('apt', 152389, 'apt_152389_matched.ecsv')]
                if expected_capture_id == "CAP-POINT" else [
                    ('apt', 152390, 'apt_152390_matched.ecsv'),
                    ('apt', 152392, 'apt_152392_matched.ecsv'),
                    ('ppt', 152389, None), ('ppt', 152391, None),
                    ('ppt', 152393, None)])
    rows = node["records"]
    required_row = {
        "role", "observation", "basename", "source_path", "destination_path",
        "size_bytes", "sha256",
    }
    if not isinstance(rows, list) or len(rows) != len(expected):
        die("authority staging record cardinality differs")
    sources: set[Path] = set()
    destinations: set[Path] = set()
    for index, (row, policy) in enumerate(zip(rows, expected)):
        if not isinstance(row, Mapping) or set(row) != required_row:
            die(f"authority staging record {index} fields differ")
        role, obsnum, exact_name = policy
        basename = row["basename"]
        ppt_name = exact_name is None and isinstance(basename, str) \
            and basename.startswith("ppt_") and basename.endswith(".ecsv") \
            and re.search(rf"(?<![0-9]){obsnum}(?![0-9])", basename)
        if row["role"] != role or row["observation"] != obsnum \
                or (exact_name is not None and basename != exact_name) \
                or (exact_name is None and not ppt_name):
            die(f"authority staging record {index} fixed identity differs")
        source = regular(Path(row["source_path"]), "staged authority source")
        destination = regular(
            Path(row["destination_path"]), "staged authority destination")
        if "citlali-validation/v1" in source.as_posix() \
                or source.name != basename or destination.name != basename \
                or source in sources or destination in destinations \
                or not isinstance(row["size_bytes"], int) \
                or isinstance(row["size_bytes"], bool) \
                or row["size_bytes"] <= 0 \
                or source.stat().st_size != row["size_bytes"] \
                or destination.stat().st_size != row["size_bytes"] \
                or not isinstance(row["sha256"], str) \
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None \
                or sha256(source) != row["sha256"] \
                or sha256(destination) != row["sha256"]:
            die(f"authority staging record {index} path/digest differs")
        sources.add(source)
        destinations.add(destination)
    return dict(node)


def command_stage_raw(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = regular(args.manifest, "raw-link manifest")
    manifest = _validate_raw_link_manifest(manifest_path)
    destination = directory(args.destination, "fresh project data directory")
    if any(destination.iterdir()):
        die("raw-link destination must be empty")
    prepared: list[tuple[Path, Path, Mapping[str, Any]]] = []
    for row in manifest["records"]:
        source = regular(Path(row["resolved_target"]), "manifest raw target")
        current = source.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != \
                (row["device"], row["inode"], row["size_bytes"], row["mtime_ns"]) \
                or sha256(source) != row["sha256"]:
            die(f"raw target identity changed: {source}")
        link = destination / row["basename"]
        if link.exists() or link.is_symlink():
            die(f"raw-link destination already exists: {link}")
        prepared.append((source, link, row))
    for source, link, _ in prepared:
        os.symlink(str(source), link)
    records = []
    for source, link, row in prepared:
        if not link.is_symlink() or link.resolve(strict=True) != source:
            die(f"staged raw symlink does not resolve exactly: {link}")
        records.append({"observation": row["observation"], "basename": row["basename"],
                        "link_path": str(link), "resolved_target": str(source),
                        "size_bytes": row["size_bytes"], "sha256": row["sha256"]})
    output = {
        "schema_version": "sci-map-001-raw-link-staging-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "capture_id": manifest["capture_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "destination": str(destination), "records": records,
        "directory_symlinks": False, "copied_raw_files": False,
        "tolproj_copy_raw_after_staging": "prohibited",
    }
    write_new(args.output, pretty_json(output))
    return output


def _authority_selection(path: Path, capture_id: str) -> list[dict[str, Any]]:
    value = read_json(regular(path, "authority selection"))
    if isinstance(value, Mapping) \
            and value.get("schema_version") == \
            "sci-map-001-authority-selection-set-v1" \
            and set(value) == {"schema_version", "captures"} \
            and isinstance(value.get("captures"), Mapping) \
            and set(value["captures"]) == set(CAPTURES):
        selected_capture = value["captures"].get(capture_id)
        value = {"schema_version": "sci-map-001-authority-selection-v1",
                 "capture_id": capture_id,
                 "records": selected_capture.get("records")
                 if isinstance(selected_capture, Mapping) \
                 and set(selected_capture) == {"records"} else None}
    if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "capture_id", "records"} \
            or value["schema_version"] != "sci-map-001-authority-selection-v1" \
            or value["capture_id"] != capture_id or not isinstance(value["records"], list):
        die("authority selection identity/shape differs")
    expected = ([('apt', 152389, 'apt_152389_matched.ecsv')]
                if capture_id == "CAP-POINT" else [
                    ('apt', 152390, 'apt_152390_matched.ecsv'),
                    ('apt', 152392, 'apt_152392_matched.ecsv'),
                    ('ppt', 152389, None), ('ppt', 152391, None),
                    ('ppt', 152393, None)])
    rows: list[dict[str, Any]] = []
    for index, (item, expected_row) in enumerate(zip(value["records"], expected)):
        if not isinstance(item, Mapping) or set(item) != {
                "role", "observation", "basename", "source_path"}:
            die(f"authority selection record {index} fields differ")
        role, obsnum, basename, source_path = (
            item["role"], item["observation"], item["basename"], item["source_path"])
        expected_role, expected_obs, expected_name = expected_row
        ppt_ok = expected_name is None and isinstance(basename, str) \
            and basename.startswith("ppt_") and basename.endswith(".ecsv") \
            and re.search(rf"(?<![0-9]){obsnum}(?![0-9])", basename)
        if not isinstance(source_path, str) or not source_path.startswith("/") \
                or Path(source_path).name != basename \
                or role != expected_role or obsnum != expected_obs \
                or (expected_name is not None and basename != expected_name) \
                or (expected_name is None and not ppt_ok):
            die(f"authority selection record {index} differs from fixed policy")
        rows.append(dict(item))
    if len(value["records"]) != len(expected):
        die("authority selection cardinality differs")
    if len({row["basename"] for row in rows}) != len(rows):
        die("authority selection repeats a basename")
    return rows


def command_stage_authorities(args: argparse.Namespace) -> dict[str, Any]:
    apt_destination = directory(args.apt_destination, "APT destination")
    ppt_destination = directory(args.ppt_destination, "PPT destination") \
        if args.capture_id == "CAP-SCIENCE" else None
    selected = _authority_selection(args.selection, args.capture_id)
    prepared: list[tuple[Path, Path, Mapping[str, Any], str]] = []
    for row in selected:
        source = regular(Path(row["source_path"]), f"{row['role']} authority")
        if source.name != row["basename"]:
            die(f"authority source basename differs from selection: {source}")
        if "citlali-validation/v1" in source.as_posix():
            die("legacy citlali-validation/v1 is reference-only, not authority")
        target_root = apt_destination if row["role"] == "apt" else ppt_destination
        if target_root is None:
            die("Point capture cannot stage PPT authority")
        target = target_root / row["basename"]
        if target.exists() or target.is_symlink():
            die(f"authority destination already exists: {target}")
        prepared.append((source, target, row, sha256(source)))
    records = []
    for source, target, row, digest in prepared:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        target.chmod(0o444)
        if sha256(target) != digest:
            die(f"copied authority digest differs: {target}")
        records.append({**row, "source_path": str(source),
                        "destination_path": str(target),
                        "size_bytes": target.stat().st_size, "sha256": digest})
    output = {
        "schema_version": "sci-map-001-authority-staging-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "capture_id": args.capture_id,
        "records": records,
        "wholesale_legacy_reduction_used": False,
    }
    write_new(args.output, pretty_json(output))
    return output


def flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            result.update(flatten(value[key], (*prefix, str(key))))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(flatten(child, (*prefix, str(index))))
        if not value:
            result[".".join(prefix)] = []
        return result
    return {".".join(prefix): value}


def leaf_differences(fixed: Mapping[str, Any],
                     capture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return membership-aware differences; explicit null is not absence."""
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(fixed) | set(capture)):
        fixed_present = key in fixed
        capture_present = key in capture
        if fixed_present == capture_present \
                and (not fixed_present or fixed[key] == capture[key]):
            continue
        differences[key] = {
            "fixed_present": fixed_present,
            "fixed": fixed[key] if fixed_present else None,
            "capture_present": capture_present,
            "capture": capture[key] if capture_present else None,
        }
    return differences


def low_level(value: Any) -> Any:
    try:
        return value["reduce"]["steps"][0]["config"]["low_level"]
    except (KeyError, IndexError, TypeError):
        return value


def merged_config_differences(fixed_node: Any,
                              capture_node: Any) -> dict[str, dict[str, Any]]:
    """Compare every realized leaf while reporting allowlist-relative names."""
    wrapped = False
    try:
        fixed_node["reduce"]["steps"][0]["config"]["low_level"]
        capture_node["reduce"]["steps"][0]["config"]["low_level"]
        wrapped = True
    except (KeyError, IndexError, TypeError):
        pass
    fixed = flatten(fixed_node)
    realized = flatten(capture_node)
    raw = leaf_differences(fixed, realized)
    prefix = "reduce.steps.0.config.low_level." if wrapped else ""
    expected = {prefix + key: value for key, value in ALLOWLIST.items()}
    if set(raw) != set(expected) \
            or any(raw[key]["capture"] != value
                   for key, value in expected.items()):
        die(f"fully merged capture config diff escapes allowlist: {raw}")
    return {key: raw[prefix + key] for key in ALLOWLIST}


def command_config_inventory(args: argparse.Namespace) -> dict[str, Any]:
    expected = NUMBERED[args.mode]
    root = directory(args.numbered_dir, "numbered config directory")
    actual = sorted(path.name for path in root.iterdir()
                    if path.is_file() and re.fullmatch(r"[0-9]{2}_.+\.ya?ml", path.name))
    if actual != sorted(expected):
        die(f"numbered config inventory differs; actual={actual}")
    records = []
    for order, name in enumerate(expected):
        path = regular(root / name, f"numbered config {name}")
        records.append({"order": order, "name": name, "path": str(path),
                        "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    fragments = []
    for raw in args.included_fragment:
        path = regular(raw, "included config fragment")
        fragments.append({"name": path.name, "path": str(path),
                          "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    fragments.sort(key=lambda row: (row["name"], row["path"]))
    if len({row["name"] for row in fragments}) != len(fragments):
        die("included config fragments repeat a basename")
    output = {
        "schema_version": "sci-map-001-realized-config-inventory-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "mode": args.mode,
        "ordered_numbered_sources": records, "included_fragments": fragments,
    }
    write_new(args.output, pretty_json(output))
    return output


def command_capture_overlay(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = regular(args.reference_overlay, "fixed-case reference overlay")
    binary = regular(args.candidate_binary, "ordinary exact-candidate binary")
    reference = read_yaml(reference_path)
    if not isinstance(reference, Mapping):
        die("fixed-case reference overlay is not a mapping")
    capture = copy.deepcopy(reference)
    try:
        step = capture["reduce"]["steps"][0]
        configured_binary = regular(Path(step["path"]),
                                    "reference-overlay candidate binary")
        low = step["config"]["low_level"]
        if not isinstance(low, Mapping):
            die("reference overlay low_level is not a mapping")
    except (KeyError, IndexError, TypeError) as exc:
        raise CaptureError("fixed-case reference overlay structure differs") from exc
    if configured_binary != binary:
        die("reference overlay does not bind the one ordinary candidate binary")
    output_node = low.setdefault("timestream", {}).setdefault(
        "processed_time_chunk", {}).setdefault("output", {})
    output_node.update({"enabled": True, "mode": "full", "indices": "all"})
    before = flatten(low_level(reference))
    after = flatten(low_level(capture))
    differences = leaf_differences(before, after)
    if set(differences) != set(ALLOWLIST) \
            or any(differences[key]["capture"] != value
                   for key, value in ALLOWLIST.items()):
        die(f"mechanical capture overlay diff escapes allowlist: {differences}")
    rendered = yaml.safe_dump(capture, sort_keys=False).encode("utf-8")
    write_new(args.output, rendered)
    return {"schema_version": "sci-map-001-capture-overlay-v1",
            "candidate_sha": CANDIDATE,
            "reference_overlay": str(reference_path),
            "reference_overlay_sha256": sha256(reference_path),
            "capture_overlay": str(args.output.resolve(strict=False)),
            "candidate_binary": str(binary), "candidate_binary_sha256": sha256(binary),
            "differences": differences}


def _config_inventory(path: Path, mode: str) -> dict[str, Any]:
    value = read_json(regular(path, "realized config inventory"))
    required = {
        "schema_version", "request_id", "revision", "candidate_sha", "mode",
        "ordered_numbered_sources", "included_fragments",
    }
    if not isinstance(value, Mapping) or set(value) != required \
            or value.get("schema_version") != \
            "sci-map-001-realized-config-inventory-v1" \
            or value.get("request_id") != REQUEST_ID \
            or value.get("revision") != REVISION \
            or value.get("candidate_sha") != CANDIDATE \
            or value.get("mode") != mode:
        die("realized config inventory identity differs")
    expected_names = NUMBERED[mode]
    rows = value["ordered_numbered_sources"]
    if not isinstance(rows, list) or len(rows) != len(expected_names):
        die("realized config inventory numbered cardinality differs")
    numbered_root: Path | None = None
    for order, (row, expected_name) in enumerate(zip(rows, expected_names)):
        if not isinstance(row, Mapping) or set(row) != {
                "order", "name", "path", "size_bytes", "sha256"} \
                or row["order"] != order or row["name"] != expected_name:
            die("realized config inventory numbered order/fields differ")
        source = regular(Path(row["path"]), f"numbered config {expected_name}")
        if source.name != expected_name:
            die("realized config inventory numbered path differs")
        if numbered_root is None:
            numbered_root = source.parent
        elif source.parent != numbered_root:
            die("realized config inventory spans multiple numbered directories")
        if row["size_bytes"] != source.stat().st_size \
                or row["sha256"] != sha256(source):
            die(f"realized config inventory digest differs: {expected_name}")
    assert numbered_root is not None
    actual = sorted(item.name for item in numbered_root.iterdir()
                    if item.is_file() and re.fullmatch(
                        r"[0-9]{2}_.+\.ya?ml", item.name))
    if actual != sorted(expected_names):
        die("realized config inventory live numbered directory differs")
    fragments = value["included_fragments"]
    if not isinstance(fragments, list):
        die("realized config fragment inventory differs")
    normalized_fragments = []
    for row in fragments:
        if not isinstance(row, Mapping) or set(row) != {
                "name", "path", "size_bytes", "sha256"}:
            die("realized config fragment inventory fields differ")
        source = regular(Path(row["path"]), "included config fragment")
        if row["name"] != source.name \
                or row["size_bytes"] != source.stat().st_size \
                or row["sha256"] != sha256(source):
            die("realized config fragment identity/digest differs")
        normalized_fragments.append((row["name"], row["path"]))
    if normalized_fragments != sorted(normalized_fragments) \
            or len({name for name, _ in normalized_fragments}) != len(fragments):
        die("realized config fragment order/basename differs")
    return dict(value)


def _config_proof_material(
        capture_id: str, fixed_config: Path, capture_config: Path,
        fixed_inventory_path: Path,
        capture_inventory_path: Path) -> dict[str, Any]:
    capture = require_capture(capture_id)
    differences = merged_config_differences(
        read_yaml(fixed_config), read_yaml(capture_config))
    fixed_inventory = _config_inventory(fixed_inventory_path, capture["mode"])
    capture_inventory = _config_inventory(capture_inventory_path, capture["mode"])
    fixed_rows = fixed_inventory["ordered_numbered_sources"]
    capture_rows = capture_inventory["ordered_numbered_sources"]
    if [row["name"] for row in fixed_rows] != NUMBERED[capture["mode"]] \
            or [row["name"] for row in capture_rows] != NUMBERED[capture["mode"]]:
        die("capture/fixed numbered source order differs")
    expert = f"99_{'pointing' if capture['mode'] == 'point' else 'science'}_expert_overrides.yaml"
    for left, right in zip(fixed_rows, capture_rows):
        if left["name"] != right["name"]:
            die("numbered source identity differs")
        if left["name"] != expert and left["sha256"] != right["sha256"]:
            die(f"numbered source differs outside expert overlay: {left['name']}")
        if left["name"] == expert:
            source_diff = flatten(low_level(read_yaml(Path(right["path"]))))
            source_fixed = flatten(low_level(read_yaml(Path(left["path"]))))
            actual_source_diff = leaf_differences(source_fixed, source_diff)
            if set(actual_source_diff) != set(ALLOWLIST) \
                    or any(actual_source_diff[key]["capture"] != value
                           for key, value in ALLOWLIST.items()):
                die("capture expert source does not contain exactly the allowlisted leaves")
    fixed_fragments = [(row["name"], row["sha256"])
                       for row in fixed_inventory["included_fragments"]]
    capture_fragments = [(row["name"], row["sha256"])
                         for row in capture_inventory["included_fragments"]]
    if fixed_fragments != capture_fragments:
        die("included config fragment inventory/digests differ")
    return {
        "differences": differences,
        "numbered_order": NUMBERED[capture["mode"]],
        "included_fragments": [list(row) for row in fixed_fragments],
    }


def _validate_config_proof(path: Path, capture_id: str) -> dict[str, Any]:
    proof_path = regular(path, "capture config proof")
    proof = read_json(proof_path)
    capture = require_capture(capture_id)
    required = {
        "schema_version", "request_id", "revision", "candidate_sha",
        "capture_id", "mode", "reference_case", "fixed_merged_config",
        "capture_merged_config", "fixed_inventory", "capture_inventory",
        "differences", "allowlist", "numbered_order", "included_fragments",
        "passed",
    }
    if not isinstance(proof, Mapping) or set(proof) != required \
            or proof["schema_version"] != "sci-map-001-capture-config-proof-v1" \
            or proof["request_id"] != REQUEST_ID \
            or proof["revision"] != REVISION \
            or proof["candidate_sha"] != CANDIDATE \
            or proof["capture_id"] != capture_id \
            or proof["mode"] != capture["mode"] \
            or proof["reference_case"] != (
                "P-SEQ" if capture_id == "CAP-POINT" else "S-E-SEQ") \
            or proof["allowlist"] != ALLOWLIST or proof["passed"] is not True:
        die("capture config proof identity/outcome differs")

    bound: dict[str, Path] = {}
    for name in ("fixed_merged_config", "capture_merged_config",
                 "fixed_inventory", "capture_inventory"):
        binding = proof[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            die(f"capture config proof {name} binding differs")
        source = regular(Path(binding["path"]), f"capture config proof {name}")
        if binding["sha256"] != sha256(source):
            die(f"capture config proof {name} digest differs")
        bound[name] = source
    material = _config_proof_material(
        capture_id, bound["fixed_merged_config"], bound["capture_merged_config"],
        bound["fixed_inventory"], bound["capture_inventory"])
    if proof["differences"] != material["differences"] \
            or proof["numbered_order"] != material["numbered_order"] \
            or proof["included_fragments"] != material["included_fragments"]:
        die("capture config proof live reconstruction differs")
    return dict(proof)


def command_config_proof(args: argparse.Namespace) -> dict[str, Any]:
    capture = require_capture(args.capture_id)
    fixed_config = regular(args.fixed_config, "fixed merged config")
    capture_config = regular(args.capture_config, "capture merged config")
    fixed_inventory_path = regular(args.fixed_inventory, "fixed config inventory")
    capture_inventory_path = regular(args.capture_inventory, "capture config inventory")
    material = _config_proof_material(
        args.capture_id, fixed_config, capture_config, fixed_inventory_path,
        capture_inventory_path)
    output = {
        "schema_version": "sci-map-001-capture-config-proof-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "capture_id": args.capture_id,
        "mode": capture["mode"], "reference_case": (
            "P-SEQ" if args.capture_id == "CAP-POINT" else "S-E-SEQ"),
        "fixed_merged_config": {"path": str(fixed_config), "sha256": sha256(fixed_config)},
        "capture_merged_config": {"path": str(capture_config), "sha256": sha256(capture_config)},
        "fixed_inventory": {"path": str(fixed_inventory_path),
                            "sha256": sha256(fixed_inventory_path)},
        "capture_inventory": {"path": str(capture_inventory_path),
                              "sha256": sha256(capture_inventory_path)},
        "differences": material["differences"], "allowlist": ALLOWLIST,
        "numbered_order": material["numbered_order"],
        "included_fragments": material["included_fragments"], "passed": True,
    }
    write_new(args.output, pretty_json(output))
    _validate_config_proof(args.output, args.capture_id)
    return output


def canonical_roots(roots: Sequence[Path]) -> list[str]:
    if len(roots) != 5:
        die("resource gate requires five roots in Point-project, Science-project, "
            "CAP-POINT, CAP-SCIENCE, compact order")
    values = [str(directory(root, "governed root")) for root in roots]
    if len(set(values)) != 5:
        die("governed roots repeat")
    return values


def resource_inventory(roots: Sequence[Path]) -> dict[str, Any]:
    canonical = canonical_roots(roots)
    entries: list[dict[str, Any]] = []
    for root_index, text in enumerate(canonical):
        root = Path(text)
        pending = [root]
        while pending:
            current = pending.pop()
            current_before = current.lstat()
            current_relative = current.relative_to(root).as_posix() or "."
            entries.append({
                "root_index": root_index,
                "relative_path": current_relative,
                "kind": "directory",
                "logical_bytes": current_before.st_size,
                "allocated_bytes": current_before.st_blocks * 512,
                "sha256": hashlib.sha256(
                    f"directory-v1\0{root_index}\0{current_relative}".encode()
                ).hexdigest(),
                "symlink_target": None,
            })
            directories = []
            for path in sorted(current.iterdir(), key=lambda item: item.name):
                before = path.lstat()
                relative = path.relative_to(root).as_posix()
                if stat.S_ISDIR(before.st_mode):
                    directories.append(path)
                elif stat.S_ISLNK(before.st_mode):
                    link_text = os.readlink(path)
                    target = Path(link_text)
                    if not target.is_absolute():
                        target = path.parent / target
                    entries.append({
                        "root_index": root_index, "relative_path": relative,
                        "kind": "symlink", "logical_bytes": before.st_size,
                        "allocated_bytes": before.st_blocks * 512,
                        "sha256": hashlib.sha256(link_text.encode()).hexdigest(),
                        "symlink_target": str(target.resolve(strict=False)),
                    })
                elif stat.S_ISREG(before.st_mode):
                    digest = sha256(path)
                    after = path.lstat()
                    if (before.st_dev, before.st_ino, before.st_size,
                            before.st_mtime_ns, before.st_blocks) != \
                            (after.st_dev, after.st_ino, after.st_size,
                             after.st_mtime_ns, after.st_blocks):
                        die(f"governed file changed while hashing: {path}")
                    entries.append({
                        "root_index": root_index, "relative_path": relative,
                        "kind": "regular-file", "logical_bytes": after.st_size,
                        "allocated_bytes": after.st_blocks * 512,
                        "sha256": digest, "symlink_target": None,
                    })
                else:
                    die(f"governed root contains unsupported special file: {path}")
            pending.extend(reversed(directories))
            current_after = current.lstat()
            if (current_before.st_dev, current_before.st_ino,
                    current_before.st_size, current_before.st_mtime_ns,
                    current_before.st_blocks) != (
                    current_after.st_dev, current_after.st_ino,
                    current_after.st_size, current_after.st_mtime_ns,
                    current_after.st_blocks):
                die(f"governed directory changed while inventorying: {current}")
    entries.sort(key=lambda row: (row["root_index"], row["relative_path"]))
    return {"schema_version": "sci-map-001-resource-inventory-v1",
            "governed_roots": canonical, "entries": entries}


def _projection_payload(stage: str, source_path: Path) -> dict[str, Any]:
    source = regular(source_path, "resource projection source")
    source_node = read_json(source)
    source_schema = source_node.get("schema_version") \
        if isinstance(source_node, Mapping) else None
    method: str
    fixed: int
    unit_count: int
    bytes_per_unit: int
    if stage in CAPTURE_PROJECTION_BYTES:
        expected_report = Path(__file__).resolve().parent.parent / "resource-report.json"
        if source != expected_report.resolve(strict=True) \
                or source_schema != "sci-map-001-ed2-resource-report-v1" \
                or source_node.get("request_id") != REQUEST_ID \
                or source_node.get("revision") != REVISION \
                or source_node.get("candidate_sha") != CANDIDATE:
            die("capture projection source is not the frozen resource report")
        envelope = source_node.get("projected_unity_envelope", {})
        rows = envelope.get("incremental_stage_bytes", [])
        values = {row.get("stage"): row.get("bytes") for row in rows
                  if isinstance(row, Mapping)}
        if stage == "PREPARE-STAGING":
            if envelope.get("projected_total_bytes") != CAPTURE_PROJECTION_BYTES[stage]:
                die("frozen preparation planning estimate differs from resource report")
        elif values.get(stage) != CAPTURE_PROJECTION_BYTES[stage]:
            die("frozen capture planning estimate differs from resource report")
        method = "owner-reviewed-local-planning-estimate-v2"
        fixed, unit_count, bytes_per_unit = 0, 1, CAPTURE_PROJECTION_BYTES[stage]
    elif stage.startswith("compact-production:"):
        group_id = stage.removeprefix("compact-production:")
        if source_schema != "sci-map-001-producer-stream-v1" \
                or source_node.get("candidate_sha") != CANDIDATE \
                or source_node.get("campaign_revision") != REVISION \
                or f"{source_node.get('obsnum')}:{source_node.get('array')}" != group_id:
            die("compact projection source identity differs")
        _validate_json_schema(source_node, "producer-stream.schema.json")
        method = "primitive-count-two-bytes-plus-64mib-v1"
        fixed, unit_count, bytes_per_unit = (
            64 * MIB, int(source_node["primitive_term_count"]), 2)
    elif stage.startswith("focused-expansion-plan:") \
            or stage.startswith("focused-expansion:"):
        if source_schema != "sci-map-001-discrepancy-request-v1" \
                or source_node.get("candidate_sha") != CANDIDATE \
                or source_node.get("campaign_revision") != REVISION:
            die("focused-expansion projection source identity differs")
        _validate_json_schema(source_node, "discrepancy-request.schema.json")
        method = "bounded-request-max-terms-v1"
        fixed = 64 * MIB
        unit_count = int(source_node["max_terms"])
        bytes_per_unit = 256 if stage.startswith("focused-expansion-plan:") else 2048
    elif stage == "ANALYSIS":
        if source_schema != "sci-map-001-result-collection-v2" \
                or source_node.get("request_id") != REQUEST_ID \
                or source_node.get("revision") != REVISION \
                or source_node.get("candidate_sha") != CANDIDATE:
            die("analysis projection source identity differs")
        method = "result-collection-size-plus-4gib-v1"
        fixed, unit_count, bytes_per_unit = 4 * GIB, source.stat().st_size, 1
    elif stage == "FINAL-BUNDLE":
        if source_schema != "sci-map-unity-evidence-inventory-v1" \
                or source_node.get("request_id") != REQUEST_ID \
                or source_node.get("candidate_sha") != CANDIDATE \
                or source_node.get("scope") != "final" \
                or not isinstance(source_node.get("records"), list):
            die("final-bundle projection source identity differs")
        request_root = directory(Path(source_node["root"]), "final inventory root")
        excluded: set[str] = set()
        for capture_id in ("CAP-POINT", "CAP-SCIENCE"):
            capture_record = read_json(regular(
                request_root / "captures" / capture_id / "capture-record.json",
                "final projection capture record"))
            for row in capture_record.get("ptc_outputs", []):
                ptc = regular(Path(row["path"]), "retained full PTC")
                if row["sha256"] != sha256(ptc) or not _under(ptc, request_root):
                    die("final projection retained PTC identity differs")
                excluded.add(ptc.relative_to(request_root).as_posix())
        total = 0
        for row in source_node["records"]:
            if not isinstance(row, Mapping) or row.get("type") != "file" \
                    or row.get("path") in excluded:
                continue
            size = row.get("size")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                die("final inventory file size differs")
            total += size
        if total <= 0:
            die("final-bundle projection has no bounded return members")
        method = "three-times-return-members-plus-64mib-v1"
        fixed, unit_count, bytes_per_unit = 64 * MIB, total, 3
    else:
        die(f"resource projection stage is not recognized: {stage}")
    projected = fixed + unit_count * bytes_per_unit
    if projected <= 0 or projected > CEILING:
        die("metadata-derived resource projection is outside the campaign ceiling")
    return {
        "schema_version": "sci-map-001-resource-projection-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "stage": stage, "method": method,
        "source": {"path": str(source), "size_bytes": source.stat().st_size,
                   "sha256": sha256(source), "schema_version": source_schema},
        "fixed_overhead_bytes": fixed, "unit_count": unit_count,
        "bytes_per_unit": bytes_per_unit,
        "projected_incremental_bytes": projected,
    }


def command_resource_projection(args: argparse.Namespace) -> dict[str, Any]:
    expected_name = f"{args.stage.replace(':', '-')}.projection.json"
    if args.output.name != expected_name:
        die("resource projection filename is not canonical for the stage")
    payload = _projection_payload(args.stage, args.source)
    _validate_json_schema(payload, "resource-projection.schema.json")
    write_new(args.output, pretty_json(payload))
    return payload


def _validate_resource_projection(path: Path, stage: str) -> dict[str, Any]:
    projection_path = regular(path, "resource projection authority")
    node = read_json(projection_path)
    _validate_json_schema(node, "resource-projection.schema.json")
    # Projection source records bind path, size, schema, and digest; validate
    # through the deterministic stage-specific reconstruction rather than
    # trusting any recorded arithmetic.
    source = regular(Path(node["source"]["path"]), "resource projection source")
    expected = _projection_payload(stage, source)
    if node != expected:
        die("resource projection authority reconstruction differs")
    return dict(node)


def command_resource_record(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args.stage, str) or not args.stage.strip() \
            or args.stage != args.stage.strip() \
            or any(token in args.stage for token in ("\n", "\r", "\\")):
        die("resource stage identity is invalid")
    phase = getattr(args, "phase", None)
    if phase not in ("pre", "post"):
        die("resource phase must be pre or post")
    expected_record_name = f"{args.stage.replace(':', '-')}.{phase}.json"
    expected_inventory_name = f"{args.stage.replace(':', '-')}.{phase}.inventory.json"
    if args.record.name != expected_record_name \
            or args.inventory.name != expected_inventory_name \
            or args.record.parent.resolve() != args.inventory.parent.resolve():
        die("resource record/inventory names must be canonical stage.phase pairs")
    projection_binding: dict[str, Any] | None = None
    if phase == "pre":
        if args.projection_authority is None:
            die("pre-stage resource projection authority is required")
        projection_path = regular(
            args.projection_authority, "resource projection authority")
        projection = _validate_resource_projection(projection_path, args.stage)
        projected = int(projection["projected_incremental_bytes"])
        projection_binding = {
            "path": str(projection_path), "sha256": sha256(projection_path),
            "method": projection["method"],
        }
    else:
        if args.projection_authority is not None:
            die("post-stage resource record cannot accept a projection authority")
        projected = 0
    roots = canonical_roots(args.governed_root)
    records_root = Path(roots[-1]) / "_campaign" / "resource-records"
    if args.record.parent.resolve(strict=False) != records_root.resolve(strict=False) \
            or args.inventory.parent.resolve(strict=False) != records_root.resolve(strict=False):
        die("resource records and inventories must remain under the governed compact root")
    filesystem_root = directory(args.filesystem_root, "resource filesystem root")
    filesystem_device = int(filesystem_root.stat().st_dev)
    if any(int(Path(root).stat().st_dev) != filesystem_device for root in roots):
        die("governed roots and resource filesystem root are not on one filesystem")
    inventory = resource_inventory(args.governed_root)
    logical = sum(row["logical_bytes"] for row in inventory["entries"])
    allocated = sum(row["allocated_bytes"] for row in inventory["entries"])
    available = shutil.disk_usage(filesystem_root).free
    logical_plus = logical + projected
    allocated_plus = allocated + projected
    if logical > CEILING or allocated > CEILING or logical_plus > CEILING \
            or allocated_plus > CEILING or available < projected:
        die("resource gate fails cumulative ceiling or filesystem capacity")
    inventory_path = write_new(args.inventory, canonical_json(inventory) + b"\n")
    record = {
        "schema_version": "sci-map-001-resource-record-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "stage": args.stage,
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "governed_roots": roots, "ceiling_bytes": CEILING,
        "filesystem_root": str(filesystem_root),
        "filesystem_device": filesystem_device,
        "current_logical_bytes": logical, "current_allocated_bytes": allocated,
        "projected_incremental_bytes": projected,
        "projection_authority": projection_binding,
        "filesystem_available_bytes": available,
        "logical_plus_projected_bytes": logical_plus,
        "allocated_plus_projected_bytes": allocated_plus,
        "inventory": {"path_count": len(inventory["entries"]),
                      "total_logical_bytes": logical,
                      "total_allocated_bytes": allocated,
                      "sha256": hashlib.sha256(canonical_json(inventory)).hexdigest()},
        "passed": True,
        "retention": {"automatic_cleanup": False,
                      "capture_point_retained": True,
                      "capture_science_retained": True},
    }
    write_new(args.record, pretty_json(record))
    # The record and its inventory are campaign artifacts inside the compact
    # governed root.  They are measured on the following record rather than
    # recursively attempting to include their own bytes in their snapshot.
    return record


def _plain_netcdf(value: Any, label: str) -> np.ndarray:
    array = np.asanyarray(value)
    if np.ma.isMaskedArray(array):
        if np.any(np.ma.getmaskarray(array)):
            die(f"candidate PTC {label} contains masked values")
        array = np.asarray(array.data)
    return np.asarray(array)


def _netcdf_scalar(dataset: Any, name: str) -> Any:
    variable = dataset.variables.get(name)
    if variable is None:
        die(f"candidate PTC lacks scalar {name}")
    value = _plain_netcdf(variable[...], name)
    if value.size != 1:
        die(f"candidate PTC scalar {name} is not scalar")
    scalar = value.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _exact_integral(value: Any, label: str) -> np.ndarray:
    array = _plain_netcdf(value, label)
    if array.dtype.kind in "fc":
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            die(f"candidate PTC {label} is not exact integral data")
    try:
        converted = array.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CaptureError(f"candidate PTC {label} cannot be represented as int64") from exc
    if array.dtype.kind in "iu" and not np.all(converted.astype(array.dtype) == array):
        die(f"candidate PTC {label} integer conversion loses information")
    return converted


def inspect_ptc(path: Path, expected_obsnum: int) -> dict[str, Any]:
    source = regular(path, "candidate full/all PTC")
    try:
        import netCDF4  # type: ignore
        dataset = netCDF4.Dataset(source, mode="r")
    except Exception as exc:
        raise CaptureError(f"cannot open candidate PTC {source}: {exc}") from exc
    try:
        required_double = (
            "signal", "flags", "kernel", "det_lat", "det_lon", "weights",
            "apt_flag", "apt_array", "apt_nw", "apt_kids_tone", "apt_uid",
        )
        for name in required_double:
            variable = dataset.variables.get(name)
            if variable is None or np.dtype(variable.dtype) != np.dtype("float64"):
                die(f"candidate PTC {name} is absent or not full binary64")
        for name in ("scan_indices", "output_scan_index", "tod_output_type",
                     "SAMPRATE", "obsnum"):
            if name not in dataset.variables:
                die(f"candidate PTC lacks required variable {name}")
        if str(_netcdf_scalar(dataset, "tod_output_type")) != "ptc":
            die("candidate PTC tod_output_type is not ptc")
        obsnum = int(_netcdf_scalar(dataset, "obsnum"))
        if obsnum != expected_obsnum:
            die(f"candidate PTC observation differs: {obsnum} != {expected_obsnum}")
        native = float(_netcdf_scalar(dataset, "SAMPRATE"))
        if not math.isfinite(native) or native <= 0.0:
            die("candidate PTC SAMPRATE is not finite positive telescope.fsmp")
        signal = dataset.variables["signal"]
        if len(signal.shape) != 2:
            die("candidate PTC signal shape is not [n_pts,n_dets]")
        n_pts, n_dets = map(int, signal.shape)
        if n_pts <= 0 or n_dets <= 0:
            die("candidate PTC has an empty sample or detector dimension")
        for name in ("flags", "kernel", "det_lat", "det_lon"):
            if tuple(map(int, dataset.variables[name].shape)) != (n_pts, n_dets):
                die(f"candidate PTC {name} shape differs from signal")
        scan_indices = _exact_integral(dataset.variables["scan_indices"][:],
                                       "scan_indices")
        output_scan = _exact_integral(dataset.variables["output_scan_index"][:],
                                      "output_scan_index")
        if scan_indices.ndim != 2 or scan_indices.shape[1] != 2:
            die("candidate PTC scan_indices shape is not [n_scans,2]")
        n_scans = int(scan_indices.shape[0])
        if n_scans <= 0 or output_scan.shape != (n_scans,) \
                or tuple(map(int, dataset.variables["weights"].shape)) != (n_scans, n_dets):
            die("candidate PTC scan/weight cardinality differs")
        if scan_indices[0, 0] != 0 or scan_indices[-1, 1] != n_pts - 1 \
                or np.any(scan_indices[:, 0] < 0) \
                or np.any(scan_indices[:, 1] < scan_indices[:, 0]) \
                or np.any(scan_indices[1:, 0] != scan_indices[:-1, 1] + 1):
            die("candidate PTC scans do not cover one contiguous full timebase")
        if not np.array_equal(output_scan, np.arange(1, n_scans + 1, dtype=np.int64)):
            die("candidate PTC full/all output scan order is not contiguous one-based")
        scan_order = []
        for index in range(n_scans):
            start = int(scan_indices[index, 0])
            stop = int(scan_indices[index, 1])
            scan_order.append({
                "scan_index": index,
                "scan_identity": f"obs={obsnum};output_scan={int(output_scan[index])}",
                "output_scan_index": int(output_scan[index]),
                "sample_start": start,
                "sample_stop_inclusive": stop,
                "sample_count": stop - start + 1,
            })
        apt = {
            name: _exact_integral(dataset.variables[name][:], name)
            for name in ("apt_flag", "apt_array", "apt_nw", "apt_kids_tone", "apt_uid")
        }
        if any(value.shape != (n_dets,) for value in apt.values()):
            die("candidate PTC APT vectors differ from detector cardinality")
        detectors: dict[str, list[dict[str, Any]]] = {}
        fwhm: dict[str, float] = {}
        for array in ARRAYS:
            rows = np.flatnonzero(apt["apt_array"] == ARRAY_IDS[array]).tolist()
            if not rows:
                die(f"candidate PTC contains no detector for {array}")
            members = []
            identities: set[str] = set()
            networks: set[int] = set()
            for detector_index, apt_row in enumerate(rows):
                network = int(apt["apt_nw"][apt_row])
                kids_tone = int(apt["apt_kids_tone"][apt_row])
                uid = str(int(apt["apt_uid"][apt_row]))
                if network not in ARRAY_NETWORKS[array] or kids_tone < 0:
                    die(f"candidate PTC detector membership is invalid for {array}")
                identity = (f"nw={network};kids_tone={kids_tone};uid={uid};"
                            f"apt_row_index={apt_row}")
                if identity in identities:
                    die(f"candidate PTC repeats composite detector identity for {array}")
                identities.add(identity)
                networks.add(network)
                members.append({
                    "detector_index": detector_index, "apt_row_index": int(apt_row),
                    "network": network, "kids_tone": kids_tone,
                    "detector_uid": uid, "detector_identity": identity,
                    "apt_flagged": bool(apt["apt_flag"][apt_row] != 0),
                })
            detectors[array] = members
            major = float(_netcdf_scalar(dataset, f"BMAJ_{array}"))
            minor = float(_netcdf_scalar(dataset, f"BMIN_{array}"))
            mean_fwhm = (major + minor) / 2.0
            if not all(math.isfinite(value) and value > 0.0
                       for value in (major, minor, mean_fwhm)):
                die(f"candidate PTC FWHM authority is invalid for {array}")
            fwhm[array] = mean_fwhm
        return {
            "path": str(source), "size_bytes": source.stat().st_size,
            "sha256": sha256(source), "obsnum": obsnum,
            "native_fsmp_hz": native, "sample_count": n_pts,
            "detector_count": n_dets, "scan_count": n_scans,
            "scan_order": scan_order, "detectors": detectors, "fwhm_arcsec": fwhm,
        }
    finally:
        dataset.close()


def realized_raw_provenance(path: Path, expected_obsnum: int,
                            ptc: Mapping[str, Any]) -> dict[str, Any]:
    source = regular(path, "realized raw-timestream provenance")
    node = read_yaml(source)
    if not isinstance(node, Mapping) \
            or node.get("schema_version") != "citlali-raw-timestream-provenance-v2" \
            or node.get("initialized") is not True:
        die("raw-timestream provenance identity/initialization differs")
    observation = node.get("observation")
    if not isinstance(observation, Mapping) or observation.get("available") is not True \
            or not isinstance(observation.get("value"), Mapping):
        die("raw-timestream observation authority is unavailable")
    value = observation["value"]
    rates = []
    for key in ("native_sample_rate_hz", "effective_sample_rate_hz"):
        rate = value.get(key)
        if not isinstance(rate, Mapping) or rate.get("available") is not True:
            die(f"raw-timestream provenance lacks {key}")
        numeric = float(rate.get("value"))
        if not math.isfinite(numeric) or numeric <= 0.0:
            die(f"raw-timestream provenance {key} is not finite positive")
        rates.append(numeric)
    native, effective = rates
    factor_node = value.get("downsample_factor")
    if not isinstance(factor_node, Mapping) \
            or factor_node.get("available") is not True \
            or not isinstance(factor_node.get("value"), int) \
            or isinstance(factor_node.get("value"), bool) \
            or factor_node["value"] <= 0:
        die("raw-timestream provenance lacks finite positive integral "
            "downsample_factor")
    downsample_factor = int(factor_node["value"])
    derived_effective = float(
        np.float64(native) / np.float64(downsample_factor))
    if struct.pack("=d", derived_effective) != struct.pack("=d", effective):
        die("raw-timestream telescope.d_fsmp is not bit-equal to "
            "telescope.fsmp/downsample_factor")
    if struct.pack("=d", native) != struct.pack("=d", float(ptc["native_fsmp_hz"])):
        die("PTC SAMPRATE is not bit-equal to raw provenance telescope.fsmp")
    realized = node.get("realized")
    if not isinstance(realized, Mapping) or realized.get("execution_completed") is not True:
        die("raw-timestream provenance does not record completed execution")
    counts = []
    for key in ("completed_scan_count", "required_timestream_write_count"):
        count = realized.get(key)
        if not isinstance(count, Mapping) or count.get("available") is not True \
                or not isinstance(count.get("value"), int) \
                or isinstance(count.get("value"), bool) or count["value"] <= 0:
            die(f"raw-timestream provenance lacks realized {key}")
        counts.append(int(count["value"]))
    completed, required_writes = counts
    if completed != ptc["scan_count"] or required_writes != ptc["scan_count"]:
        die("realized scan/write cardinality differs from full/all PTC")
    return {
        "obsnum": expected_obsnum, "path": str(source),
        "size_bytes": source.stat().st_size, "sha256": sha256(source),
        "schema_version": node["schema_version"], "native_fsmp_hz": native,
        "effective_d_fsmp_hz": effective,
        "downsample_factor": downsample_factor,
        "completed_scan_count": completed,
        "required_timestream_write_count": required_writes,
        "ptc_scan_count": int(ptc["scan_count"]),
        "ptc_sample_count": int(ptc["sample_count"]),
        "scan_cardinality_matches": True,
    }


def _raw_exact_value(node: Any, label: str) -> float:
    return parse_exact(node, label)


def realized_mapmaking(path: Path, expected_obsnum: int) -> dict[str, Any]:
    source = regular(path, "realized mapmaking provenance")
    node = read_yaml(source)
    if not isinstance(node, Mapping) \
            or node.get("schema_version") != "citlali-mapmaking-provenance-v3" \
            or node.get("initialized") is not True:
        die("mapmaking provenance identity/initialization differs")
    realized = node.get("realized")
    if not isinstance(realized, Mapping) \
            or realized.get("reduction_completed") is not True \
            or realized.get("mapmaking_executed") is not True:
        die("mapmaking provenance does not record completed mapmaking")
    observations = node.get("observations")
    matches = [row for row in observations if isinstance(row, Mapping)
               and row.get("obsnum") == expected_obsnum] \
        if isinstance(observations, list) else []
    if len(matches) != 1:
        die(f"mapmaking provenance observation {expected_obsnum} is absent/ambiguous")
    observation = matches[0]
    if observation.get("outputs_completed") is not True:
        die("mapmaking observation outputs are incomplete")
    pixel_size = float(observation.get("effective_pixel_size_rad"))
    if not math.isfinite(pixel_size) or pixel_size <= 0.0:
        die("mapmaking effective pixel-size authority is invalid")
    science_state = observation.get("science_state")
    bundle_node = science_state.get("bundle_identity") \
        if isinstance(science_state, Mapping) else None
    if not isinstance(science_state, Mapping) or science_state.get("available") is not True \
            or not isinstance(bundle_node, Mapping) \
            or bundle_node.get("available") is not True \
            or not isinstance(bundle_node.get("value"), Mapping):
        die("mapmaking science bundle identity authority is unavailable")
    bundle = bundle_node["value"]
    digest = bundle.get("identity_digest")
    if not isinstance(digest, str) \
            or re.fullmatch(r"canonical-hexfloat-sha256-v1:[0-9a-f]{64}", digest) is None \
            or bundle.get("grouping") != "array":
        die("mapmaking bundle identity/grouping differs")
    shape = bundle.get("shape")
    if not isinstance(shape, Mapping) \
            or any(not isinstance(shape.get(key), int) or isinstance(shape.get(key), bool)
                   or shape[key] <= 0 for key in ("rows", "cols")):
        die("mapmaking bundle shape authority is invalid")
    wcs = bundle.get("wcs")
    if not isinstance(wcs, Mapping):
        die("mapmaking WCS identity is absent")
    coordinate_frame = str(wcs.get("coordinate_frame", ""))
    if coordinate_frame in ("altaz", "azel", "az_el"):
        frame = "altaz"
    elif coordinate_frame in ("radec", "ra_dec", "fk5", "equatorial"):
        frame = "fk5"
    else:
        die(f"mapmaking coordinate frame is unsupported: {coordinate_frame!r}")
    reference = wcs.get("reference_world")
    units = wcs.get("axis_units")
    if not isinstance(reference, list) or len(reference) != 2 \
            or not isinstance(units, list) or len(units) != 2:
        die("mapmaking WCS target reference is incomplete")
    target_values = []
    for index, (exact, unit) in enumerate(zip(reference, units)):
        numeric = _raw_exact_value(exact, f"reference_world[{index}]")
        if unit == "deg":
            converted = numeric
        elif unit == "arcsec":
            converted = numeric / 3600.0
        else:
            die(f"mapmaking WCS target unit is unsupported: {unit!r}")
        target_values.append(raw_exact_float(converted))
    slots = bundle.get("ordered_slots")
    if not isinstance(slots, list):
        die("mapmaking ordered map slots are absent")
    slot_by_array: dict[str, Mapping[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping) or slot.get("grouping") != "array" \
                or slot.get("stokes_identity") != "I":
            continue
        array = slot.get("array_identity")
        if array in ARRAYS:
            if array in slot_by_array:
                die(f"mapmaking bundle repeats array slot {array}")
            slot_by_array[str(array)] = slot
    if set(slot_by_array) != set(ARRAYS):
        die("mapmaking bundle does not contain exactly one I slot per array")
    return {
        "path": str(source), "size_bytes": source.stat().st_size,
        "sha256": sha256(source), "obsnum": expected_obsnum,
        "identity_digest": digest, "frame": frame,
        "map_rows": int(shape["rows"]), "map_cols": int(shape["cols"]),
        "pixel_size_rad": pixel_size,
        "target": {"frame": frame, "axis1": target_values[0],
                   "axis2": target_values[1], "unit": "deg"},
        "slot_by_array": slot_by_array,
    }


def parse_obs_paths(values: Sequence[str], expected: Sequence[int],
                    label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values:
        match = re.fullmatch(r"([0-9]+)=(/.+)", raw)
        if match is None:
            die(f"{label} must use OBSNUM=/absolute/path")
        obsnum = int(match.group(1))
        if obsnum in result:
            die(f"{label} repeats observation {obsnum}")
        result[obsnum] = regular(Path(match.group(2)), label)
    if list(result) != list(expected):
        die(f"{label} order/identity differs: {list(result)} != {list(expected)}")
    return result


def _source_selection(
        path: Path, capture_id: str, capture_root: Path,
        raw_link_manifest: Mapping[str, Any],
        authority_staging: Mapping[str, Any]) -> list[dict[str, Any]]:
    node = read_json(regular(path, "raw-input source selection"))
    _validate_json_schema(node, "source-selection.schema.json")
    if not isinstance(node, Mapping) or set(node) != {
            "schema_version", "capture_id", "records"} \
            or node.get("schema_version") != "sci-map-001-source-selection-v1" \
            or node.get("capture_id") != capture_id \
            or not isinstance(node.get("records"), list):
        die("raw-input source selection identity/shape differs")
    capture = require_capture(capture_id)
    manual_roles = {"raw_timestream", "kids_fit_report", "apt", "calibration",
                    "pointing_support"}
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    matched_raw: set[int] = set()
    matched_apt: set[int] = set()
    matched_pointing: set[int] = set()
    raw_rows = raw_link_manifest["records"]
    authority_rows = authority_staging["records"]
    for index, raw in enumerate(node["records"]):
        expected_keys = {"id", "role", "path", "obsnums", "arrays", "networks"}
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            die(f"source selection record {index} fields differ")
        row = dict(raw)
        identifier = row["id"]
        if not isinstance(identifier, str) or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identifier) is None \
                or identifier in identities or row["role"] not in manual_roles:
            die(f"source selection record {index} identity/role differs")
        identities.add(identifier)
        allowed_obs = capture["manifest_support"] if row["role"] == "pointing_support" \
            else capture["observations"]
        if not isinstance(row["obsnums"], list) or row["obsnums"] != \
                sorted(set(row["obsnums"])) or not row["obsnums"] \
                or not set(row["obsnums"]).issubset(allowed_obs):
            die(f"source selection {identifier} observation membership differs")
        if not isinstance(row["arrays"], list) or row["arrays"] != \
                [array for array in ARRAYS if array in row["arrays"]] or not row["arrays"]:
            die(f"source selection {identifier} array membership differs")
        allowed_networks = sorted({nw for array in row["arrays"]
                                   for nw in ARRAY_NETWORKS[array]})
        if not isinstance(row["networks"], list) or not row["networks"] \
                or row["networks"] != sorted(set(row["networks"])) \
                or not set(row["networks"]).issubset(allowed_networks):
            die(f"source selection {identifier} network membership differs")
        source = regular(Path(row["path"]), f"source selection {identifier}")
        row.update({"path": str(source), "size_bytes": source.stat().st_size,
                    "sha256": sha256(source)})
        if row["size_bytes"] <= 0:
            die(f"source selection {identifier} is empty")
        role = row["role"]
        if role == "raw_timestream":
            matches = [
                raw_index for raw_index, authority in enumerate(raw_rows)
                if Path(authority["resolved_target"]).resolve(strict=True) == source
                and authority["size_bytes"] == row["size_bytes"]
                and authority["sha256"] == row["sha256"]
                and row["obsnums"] == [authority["observation"]]
            ]
            if len(matches) != 1 or matches[0] in matched_raw:
                die(f"source selection {identifier} does not uniquely bind staged raw authority")
            matched_raw.add(matches[0])
        elif role == "apt":
            matches = [
                authority_index for authority_index, authority
                in enumerate(authority_rows)
                if authority["role"] == "apt"
                and Path(authority["destination_path"]).resolve(strict=True) == source
                and authority["size_bytes"] == row["size_bytes"]
                and authority["sha256"] == row["sha256"]
                and row["obsnums"] == [authority["observation"]]
            ]
            if len(matches) != 1 or matches[0] in matched_apt:
                die(f"source selection {identifier} does not uniquely bind staged APT authority")
            matched_apt.add(matches[0])
        elif role == "pointing_support":
            authority_role = "apt" if capture_id == "CAP-POINT" else "ppt"
            matches = [
                authority_index for authority_index, authority
                in enumerate(authority_rows)
                if authority["role"] == authority_role
                and Path(authority["destination_path"]).resolve(strict=True) == source
                and authority["size_bytes"] == row["size_bytes"]
                and authority["sha256"] == row["sha256"]
                and row["obsnums"] == [authority["observation"]]
            ]
            if len(matches) != 1 or matches[0] in matched_pointing:
                die(f"source selection {identifier} does not uniquely bind staged pointing authority")
            matched_pointing.add(matches[0])
        elif not _under(source, capture_root):
            die(f"source selection {identifier} escapes the capture root")
        records.append(row)
    if {row["role"] for row in records} != manual_roles:
        die("source selection does not cover the five staged/manual roles")
    support = sorted({obs for row in records if row["role"] == "pointing_support"
                      for obs in row["obsnums"]})
    if support != capture["manifest_support"]:
        die("source selection pointing-support observation set differs")
    expected_raw = {
        index for index, row in enumerate(raw_rows)
        if row["observation"] in capture["observations"]}
    expected_apt = {index for index, row in enumerate(authority_rows)
                    if row["role"] == "apt"}
    expected_pointing = {
        index for index, row in enumerate(authority_rows)
        if row["role"] == ("apt" if capture_id == "CAP-POINT" else "ppt")
    }
    if matched_raw != expected_raw or matched_apt != expected_apt \
            or matched_pointing != expected_pointing:
        die("source selection does not exactly cover staged raw/APT/PPT authorities")
    return records


def _validate_json_schema(instance: Any, schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parent.parent / schema_name
    try:
        import jsonschema  # type: ignore
        jsonschema.Draft202012Validator(read_json(schema_path)).validate(instance)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"{schema_name} validation failed: {exc}") from exc


def command_build_raw_input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    capture = require_capture(args.capture_id)
    capture_root = directory(args.capture_root, "capture root")
    output = capture_root / "raw-input-manifest.json"
    new_path(output, "raw-input manifest")
    ptc_paths = parse_obs_paths(args.ptc, capture["observations"], "PTC input")
    raw_paths = parse_obs_paths(args.raw_provenance, capture["observations"],
                                "raw provenance input")
    map_paths = parse_obs_paths(args.map_provenance, capture["observations"],
                                "mapmaking provenance input")
    for label, paths in (("PTC input", ptc_paths),
                         ("raw provenance input", raw_paths),
                         ("mapmaking provenance input", map_paths)):
        for source in paths.values():
            if not _under(source, capture_root):
                die(f"{label} escapes capture root")
    raw_link_path = regular(args.raw_link_manifest, "raw-link manifest")
    raw_link = _validate_raw_link_manifest(raw_link_path, args.capture_id)
    raw_staging_path = regular(args.raw_link_staging, "raw-link staging manifest")
    _validate_raw_link_staging(
        raw_staging_path, raw_link_path, args.capture_id)
    authority_path = regular(args.authority_manifest, "authority staging manifest")
    authority_staging = _validate_authority_staging(
        authority_path, args.capture_id)
    source_selection_path = regular(
        args.source_selection, "raw-input source selection")
    ptcs: dict[int, dict[str, Any]] = {}
    raws: dict[int, dict[str, Any]] = {}
    maps: dict[int, dict[str, Any]] = {}
    for obsnum in capture["observations"]:
        ptcs[obsnum] = inspect_ptc(ptc_paths[obsnum], obsnum)
        raws[obsnum] = realized_raw_provenance(raw_paths[obsnum], obsnum, ptcs[obsnum])
        maps[obsnum] = realized_mapmaking(map_paths[obsnum], obsnum)
        expected_frame = "altaz" if capture["mode"] == "point" else "fk5"
        if maps[obsnum]["frame"] != expected_frame:
            die(f"mapmaking frame differs for capture mode {capture['mode']}")
    manual = _source_selection(
        source_selection_path, args.capture_id, capture_root, raw_link,
        authority_staging)
    source_records = list(manual)
    for obsnum in capture["observations"]:
        all_networks = sorted({detector["network"] for array in ARRAYS
                               for detector in ptcs[obsnum]["detectors"][array]})
        automatic = (
            (f"projection-{obsnum}", "projection_authority", maps[obsnum]),
            (f"sample-rate-{obsnum}", "sample_rate_authority", raws[obsnum]),
            (f"fwhm-{obsnum}", "fwhm_authority", ptcs[obsnum]),
            (f"target-{obsnum}", "target_authority", maps[obsnum]),
        )
        for identifier, role, source in automatic:
            source_records.append({
                "id": identifier, "role": role, "path": source["path"],
                "size_bytes": source["size_bytes"], "sha256": source["sha256"],
                "obsnums": [obsnum], "arrays": list(ARRAYS),
                "networks": all_networks,
            })
    memberships = []
    used: set[str] = set()
    for obsnum in capture["observations"]:
        ptc = ptcs[obsnum]
        raw = raws[obsnum]
        mapping = maps[obsnum]
        interval = float(np.float64(1.0) / np.float64(raw["effective_d_fsmp_hz"]))
        if not math.isfinite(interval) or interval <= 0.0:
            die("binary64 sample interval authority is invalid")
        for array in ARRAYS:
            detectors = ptc["detectors"][array]
            networks = sorted({row["network"] for row in detectors})
            refs: dict[str, list[str]] = {}
            for role in SOURCE_ROLES:
                matches = []
                coverage: set[int] = set()
                for source in source_records:
                    observation_applies = role == "pointing_support" or \
                        obsnum in source["obsnums"]
                    if source["role"] == role and observation_applies \
                            and array in source["arrays"]:
                        matches.append(source["id"])
                        coverage.update(source["networks"])
                if not matches or not set(networks).issubset(coverage):
                    die(f"{obsnum}:{array} lacks {role} coverage for every network")
                if role in ("raw_timestream", "kids_fit_report") \
                        and coverage != set(networks):
                    die(f"{obsnum}:{array} {role} network coverage is not exact")
                refs[role] = matches
                used.update(matches)
            scan_order = [{"scan_index": row["scan_index"],
                           "identity": row["scan_identity"],
                           "sample_count": row["sample_count"]}
                          for row in ptc["scan_order"]]
            detector_order = [{key: row[key] for key in (
                "detector_index", "apt_row_index", "network", "kids_tone",
                "detector_uid", "detector_identity", "apt_flagged")}
                for row in detectors]
            memberships.append({
                "obsnum": obsnum, "array": array, "networks": networks,
                "record_order": "scan-major-detector-major-sample-minor-cartesian-v1",
                "projection_record_count": sum(row["sample_count"] for row in scan_order)
                * len(detector_order), "scan_order": scan_order,
                "detector_order": detector_order, "source_refs": refs,
                "projection": {
                    "identity_digest": mapping["identity_digest"],
                    "grouping": "array", "stokes": "I", "frame": mapping["frame"],
                    "map_rows": mapping["map_rows"], "map_cols": mapping["map_cols"],
                    "native_fsmp_hz": raw_exact_float(raw["native_fsmp_hz"]),
                    "effective_d_fsmp_hz": raw_exact_float(raw["effective_d_fsmp_hz"]),
                    "sample_interval_s": raw_exact_float(interval),
                    "pixel_size_rad": raw_exact_float(mapping["pixel_size_rad"]),
                    "fwhm_arcsec": raw_exact_float(ptc["fwhm_arcsec"][array]),
                    "target": mapping["target"],
                },
            })
    if used != {row["id"] for row in source_records}:
        die("raw-input manifest source selection contains unused records")
    program = regular(Path(__file__).resolve(), "raw-input manifest producer")
    manifest = {
        "schema_version": "sci-map-001-raw-input-manifest-v2",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "capture_id": args.capture_id,
        "mode": capture["mode"], "observations": capture["observations"],
        "arrays": list(ARRAYS),
        "staging": {
            "raw_link_manifest": {
                "path": str(raw_link_path), "sha256": sha256(raw_link_path)},
            "raw_link_staging": {
                "path": str(raw_staging_path), "sha256": sha256(raw_staging_path)},
            "authority_staging": {
                "path": str(authority_path), "sha256": sha256(authority_path)},
            "source_selection": {
                "path": str(source_selection_path),
                "sha256": sha256(source_selection_path)},
        },
        "producer": {"identity": "sci-map-001-ed2-capture-authority-v1",
                     "program_path": str(program), "program_sha256": sha256(program),
                     "invocation": list(sys.argv)},
        "source_records": source_records, "memberships": memberships,
    }
    _validate_json_schema(manifest, "raw-input-manifest.schema.json")
    write_new(output, canonical_json(manifest) + b"\n")
    return manifest


def _bound_file(binding: Any, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        die(f"{label} binding fields differ")
    path = regular(Path(binding["path"]), label)
    if binding["sha256"] != sha256(path):
        die(f"{label} binding digest differs")
    return path


def _validate_raw_input_manifest(
        path: Path, capture_root: Path, expected_capture_id: str,
        *, expected_raw_link: Path | None = None,
        expected_raw_staging: Path | None = None,
        expected_authority: Path | None = None) -> dict[str, Any]:
    manifest_path = _require_capture_artifact(
        path, capture_root, "automatic raw-input manifest")
    manifest = read_json(manifest_path)
    _validate_json_schema(manifest, "raw-input-manifest.schema.json")
    capture = require_capture(expected_capture_id)
    required = {
        "schema_version", "request_id", "revision", "candidate_sha",
        "capture_id", "mode", "observations", "arrays", "staging",
        "producer", "source_records", "memberships",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required \
            or manifest["schema_version"] != "sci-map-001-raw-input-manifest-v2" \
            or manifest["request_id"] != REQUEST_ID \
            or manifest["revision"] != REVISION \
            or manifest["candidate_sha"] != CANDIDATE \
            or manifest["capture_id"] != expected_capture_id \
            or manifest["mode"] != capture["mode"] \
            or manifest["observations"] != capture["observations"] \
            or manifest["arrays"] != list(ARRAYS):
        die("raw-input manifest identity differs")
    staging = manifest["staging"]
    if not isinstance(staging, Mapping) or set(staging) != {
            "raw_link_manifest", "raw_link_staging", "authority_staging",
            "source_selection"}:
        die("raw-input manifest staging bindings differ")
    raw_link_path = _bound_file(
        staging["raw_link_manifest"], "raw-input raw-link manifest")
    raw_staging_path = _bound_file(
        staging["raw_link_staging"], "raw-input raw-link staging")
    authority_path = _bound_file(
        staging["authority_staging"], "raw-input authority staging")
    source_selection_path = _bound_file(
        staging["source_selection"], "raw-input source selection")
    for expected, actual, label in (
            (expected_raw_link, raw_link_path, "raw-link manifest"),
            (expected_raw_staging, raw_staging_path, "raw-link staging"),
            (expected_authority, authority_path, "authority staging")):
        if expected is not None and regular(expected, label) != actual:
            die(f"raw-input manifest {label} path differs")
    raw_link = _validate_raw_link_manifest(raw_link_path, expected_capture_id)
    _validate_raw_link_staging(
        raw_staging_path, raw_link_path, expected_capture_id)
    authorities = _validate_authority_staging(
        authority_path, expected_capture_id)
    manual = _source_selection(
        source_selection_path, expected_capture_id, capture_root, raw_link,
        authorities)
    source_records = manifest["source_records"]
    if not isinstance(source_records, list) \
            or source_records[:len(manual)] != manual:
        die("raw-input manifest manual source reconstruction differs")
    identifiers: set[str] = set()
    for row in source_records:
        if not isinstance(row, Mapping) or row["id"] in identifiers:
            die("raw-input manifest source identity repeats")
        identifiers.add(row["id"])
        source = regular(Path(row["path"]), f"raw-input source {row['id']}")
        if source.stat().st_size != row["size_bytes"] \
                or sha256(source) != row["sha256"]:
            die(f"raw-input source {row['id']} live digest differs")
    expected_automatic: list[dict[str, Any]] = []
    ptcs: dict[int, dict[str, Any]] = {}
    raws: dict[int, dict[str, Any]] = {}
    maps: dict[int, dict[str, Any]] = {}
    automatic_rows = source_records[len(manual):]
    automatic_by_id = {row["id"]: row for row in automatic_rows}
    expected_ids = {
        f"{prefix}-{obsnum}"
        for obsnum in capture["observations"]
        for prefix in ("projection", "sample-rate", "fwhm", "target")
    }
    if set(automatic_by_id) != expected_ids \
            or len(automatic_by_id) != len(automatic_rows):
        die("raw-input manifest automatic source identities differ")
    for obsnum in capture["observations"]:
        fwhm_row = automatic_by_id[f"fwhm-{obsnum}"]
        rate_row = automatic_by_id[f"sample-rate-{obsnum}"]
        projection_row = automatic_by_id[f"projection-{obsnum}"]
        target_row = automatic_by_id[f"target-{obsnum}"]
        for row in (fwhm_row, rate_row, projection_row, target_row):
            if not _under(Path(row["path"]).resolve(strict=True), capture_root):
                die("raw-input automatic authority escapes capture root")
        ptcs[obsnum] = inspect_ptc(Path(fwhm_row["path"]), obsnum)
        raws[obsnum] = realized_raw_provenance(
            Path(rate_row["path"]), obsnum, ptcs[obsnum])
        maps[obsnum] = realized_mapmaking(Path(projection_row["path"]), obsnum)
        if target_row["path"] != projection_row["path"] \
                or target_row["sha256"] != projection_row["sha256"]:
            die("raw-input target/projection authority identity differs")
        all_networks = sorted({detector["network"] for array in ARRAYS
                               for detector in ptcs[obsnum]["detectors"][array]})
        expected_automatic.extend((
            {"id": f"projection-{obsnum}", "role": "projection_authority",
             "path": maps[obsnum]["path"], "size_bytes": maps[obsnum]["size_bytes"],
             "sha256": maps[obsnum]["sha256"], "obsnums": [obsnum],
             "arrays": list(ARRAYS), "networks": all_networks},
            {"id": f"sample-rate-{obsnum}", "role": "sample_rate_authority",
             "path": raws[obsnum]["path"], "size_bytes": raws[obsnum]["size_bytes"],
             "sha256": raws[obsnum]["sha256"], "obsnums": [obsnum],
             "arrays": list(ARRAYS), "networks": all_networks},
            {"id": f"fwhm-{obsnum}", "role": "fwhm_authority",
             "path": ptcs[obsnum]["path"], "size_bytes": ptcs[obsnum]["size_bytes"],
             "sha256": ptcs[obsnum]["sha256"], "obsnums": [obsnum],
             "arrays": list(ARRAYS), "networks": all_networks},
            {"id": f"target-{obsnum}", "role": "target_authority",
             "path": maps[obsnum]["path"], "size_bytes": maps[obsnum]["size_bytes"],
             "sha256": maps[obsnum]["sha256"], "obsnums": [obsnum],
             "arrays": list(ARRAYS), "networks": all_networks},
        ))
    if automatic_rows != expected_automatic:
        die("raw-input manifest automatic source reconstruction differs")
    producer = manifest["producer"]
    program = regular(Path(producer["program_path"]), "raw-input producer program")
    if program != Path(__file__).resolve() \
            or producer["identity"] != "sci-map-001-ed2-capture-authority-v1" \
            or producer["program_sha256"] != sha256(program):
        die("raw-input manifest producer identity differs")
    expected_memberships = [
        (obsnum, array) for obsnum in capture["observations"] for array in ARRAYS]
    memberships = manifest["memberships"]
    if [(row["obsnum"], row["array"]) for row in memberships] != expected_memberships:
        die("raw-input manifest membership order differs")
    for membership in memberships:
        obsnum, array = membership["obsnum"], membership["array"]
        ptc, raw, mapping = ptcs[obsnum], raws[obsnum], maps[obsnum]
        detectors = ptc["detectors"][array]
        networks = sorted({row["network"] for row in detectors})
        scans = [{"scan_index": row["scan_index"],
                  "identity": row["scan_identity"],
                  "sample_count": row["sample_count"]}
                 for row in ptc["scan_order"]]
        expected_detector_order = [
            {key: row[key] for key in (
                "detector_index", "apt_row_index", "network", "kids_tone",
                "detector_uid", "detector_identity", "apt_flagged")}
            for row in detectors]
        if membership["networks"] != networks \
                or membership["scan_order"] != scans \
                or membership["detector_order"] != expected_detector_order \
                or membership["projection_record_count"] != \
                sum(row["sample_count"] for row in scans) * len(detectors):
            die("raw-input manifest PTC membership reconstruction differs")
        expected_refs: dict[str, list[str]] = {}
        for role in SOURCE_ROLES:
            expected_refs[role] = [
                source["id"] for source in source_records
                if source["role"] == role
                and (role == "pointing_support" or obsnum in source["obsnums"])
                and array in source["arrays"]]
        if membership["source_refs"] != expected_refs:
            die("raw-input manifest source references differ")
        projection = membership["projection"]
        expected_interval = float(
            np.float64(1.0) / np.float64(raw["effective_d_fsmp_hz"]))
        exact_checks = (
            ("native_fsmp_hz", raw["native_fsmp_hz"]),
            ("effective_d_fsmp_hz", raw["effective_d_fsmp_hz"]),
            ("sample_interval_s", expected_interval),
            ("pixel_size_rad", mapping["pixel_size_rad"]),
            ("fwhm_arcsec", ptc["fwhm_arcsec"][array]),
        )
        if projection["identity_digest"] != mapping["identity_digest"] \
                or projection["frame"] != mapping["frame"] \
                or projection["map_rows"] != mapping["map_rows"] \
                or projection["map_cols"] != mapping["map_cols"] \
                or projection["target"] != mapping["target"]:
            die("raw-input manifest projection identity differs")
        for name, expected in exact_checks:
            actual = parse_exact(projection[name], f"raw-input {name}")
            if struct.pack("=d", actual) != struct.pack("=d", expected):
                die(f"raw-input manifest projection {name} differs")
    return dict(manifest)


def command_producer_authority(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = regular(args.raw_input_manifest, "raw-input manifest")
    preliminary = read_json(raw_path)
    observations = preliminary.get("observations") \
        if isinstance(preliminary, Mapping) else None
    capture_id = "CAP-POINT" if observations == [152389] \
        else "CAP-SCIENCE" if observations == [152390, 152392] \
        else ""
    manifest = _validate_raw_input_manifest(
        raw_path, directory(raw_path.parent, "capture root"), capture_id)
    matches = [row for row in manifest["memberships"]
               if row["obsnum"] == args.obsnum and row["array"] == args.array]
    if len(matches) != 1:
        die("raw-input membership is absent/ambiguous")
    membership = matches[0]
    sources = {row["id"]: row for row in manifest["source_records"]}
    ptc_ids = membership["source_refs"]["fwhm_authority"]
    map_ids = membership["source_refs"]["projection_authority"]
    rate_ids = membership["source_refs"]["sample_rate_authority"]
    if len(ptc_ids) != 1 or len(map_ids) != 1 or len(rate_ids) != 1:
        die("automatic PTC/raw/mapmaking authority reference is ambiguous")
    ptc_source = sources[ptc_ids[0]]
    map_source = sources[map_ids[0]]
    rate_source = sources[rate_ids[0]]
    ptc = inspect_ptc(Path(ptc_source["path"]), args.obsnum)
    if ptc["sha256"] != ptc_source["sha256"]:
        die("PTC authority digest differs from raw-input manifest")
    detectors = ptc["detectors"][args.array]
    raw_detectors = membership["detector_order"]
    if [(row["detector_index"], row["detector_uid"], row["network"])
            for row in detectors] != [(row["detector_index"], row["detector_uid"],
                                       row["network"]) for row in raw_detectors]:
        die("PTC detector identity differs from raw-input membership")
    scan_order = [{"scan_index": row["scan_index"],
                   "scan_identity": row["scan_identity"],
                   "output_scan_index": row["output_scan_index"],
                   "sample_count": row["sample_count"]} for row in ptc["scan_order"]]
    raw_scans = membership["scan_order"]
    if [(row["scan_index"], row["scan_identity"], row["sample_count"])
            for row in scan_order] != [(row["scan_index"], row["identity"],
                                        row["sample_count"]) for row in raw_scans]:
        die("PTC scan identity differs from raw-input membership")
    projection = membership["projection"]
    native = parse_exact(projection["native_fsmp_hz"], "native_fsmp_hz")
    if struct.pack("=d", native) != struct.pack("=d", ptc["native_fsmp_hz"]):
        die("PTC SAMPRATE differs from raw-input native authority")
    authority = {
        "schema_version": "sci-map-001-producer-stream-v1",
        "adapter": "candidate-full-ptc-netcdf-v1",
        "capture_output_mode": "full", "capture_detector_selection": "all",
        "candidate_sha": CANDIDATE, "campaign_revision": REVISION,
        "raw_input_manifest_sha256": sha256(raw_path),
        "producer_identity": manifest["producer"]["identity"],
        "capture_ptc_sha256": ptc["sha256"],
        "realized_raw_timestream_provenance_sha256": rate_source["sha256"],
        "realized_mapmaking_provenance_sha256": map_source["sha256"],
        "mapmaking_bundle_identity_digest": projection["identity_digest"],
        "obsnum": args.obsnum, "array": args.array,
        "map_shape": {"rows": projection["map_rows"],
                      "cols": projection["map_cols"]},
        "map_pixel_size_rad": exact_float(
            parse_exact(projection["pixel_size_rad"], "pixel_size_rad"),
            "realized_mapmaking.effective_pixel_size_rad"),
        "native_fsmp_hz": exact_float(native, "telescope.fsmp"),
        "effective_d_fsmp_hz": exact_float(
            parse_exact(projection["effective_d_fsmp_hz"], "effective_d_fsmp_hz"),
            "telescope.d_fsmp"),
        "scan_order": scan_order, "detector_order": detectors,
        "primitive_term_count": sum(row["sample_count"] for row in scan_order)
        * len(detectors),
        "term_order": "scan-major-detector-major-sample-minor-cartesian-v1",
    }
    _validate_json_schema(authority, "producer-stream.schema.json")
    write_new(args.output, canonical_json(authority) + b"\n")
    return authority


def _require_capture_artifact(path: Path, capture_root: Path, label: str) -> Path:
    source = regular(path, label)
    if not _under(source, capture_root):
        die(f"{label} escapes capture root")
    return source


def _full_ptc_inventory(capture_root: Path) -> set[Path]:
    try:
        import netCDF4  # type: ignore
    except Exception as exc:
        raise CaptureError(f"cannot inventory capture NetCDF outputs: {exc}") from exc
    outputs: set[Path] = set()
    for path in capture_root.rglob("*.nc"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with netCDF4.Dataset(path, mode="r") as dataset:
                if "tod_output_type" in dataset.variables \
                        and str(_netcdf_scalar(dataset, "tod_output_type")) == "ptc":
                    outputs.add(path.resolve(strict=True))
        except CaptureError:
            raise
        except Exception:
            # Other capture NetCDF products are not PTC authority and remain
            # governed by the resource inventory.
            continue
    return outputs


def _validate_binary_evidence(
        binary_path: Path, build_manifest_path: Path,
        version_output_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    binary = regular(binary_path, "ordinary candidate binary")
    build_path = regular(build_manifest_path, "candidate build manifest")
    version = regular(version_output_path, "candidate version output")
    if version.stat().st_size <= 0:
        die("candidate version output is empty")
    binary_digest = sha256(binary)
    build = read_json(build_path)
    required = {
        "schema_version", "started_at", "completed_at", "candidate_sha",
        "candidate_tree", "build_preset", "build_target", "binary",
        "binary_sha256",
        "built_binary", "built_binary_sha256", "cmake_cache_sha256",
        "compile_commands_sha256", "compiler", "compiler_sha256",
        "version_output", "version_output_sha256", "binary_count", "ordinary",
        "instrumented", "dependencies",
    }
    if not isinstance(build, Mapping) or set(build) != required \
            or build["schema_version"] != "sci-map-unity-build-state-v1" \
            or build["candidate_sha"] != CANDIDATE \
            or build["candidate_tree"] != CANDIDATE_TREE \
            or build["build_preset"] != "unity_release" \
            or build["build_target"] != "citlali_cli" \
            or Path(str(build["binary"])).resolve(strict=True) != binary \
            or build["binary_sha256"] != binary_digest \
            or Path(str(build["version_output"])).resolve(strict=True) != version \
            or build["version_output_sha256"] != sha256(version) \
            or build["binary_count"] != 1 \
            or build["ordinary"] is not True \
            or build["instrumented"] is not False \
            or not isinstance(build["started_at"], str) \
            or not isinstance(build["completed_at"], str) \
            or not isinstance(build["dependencies"], Mapping):
        die("candidate build manifest does not bind the exact ordinary binary")
    built = regular(Path(build["built_binary"]), "built candidate binary")
    compiler = regular(Path(build["compiler"]), "candidate build compiler")
    if build["built_binary_sha256"] != binary_digest \
            or sha256(built) != binary_digest \
            or build["compiler_sha256"] != sha256(compiler) \
            or re.fullmatch(r"[0-9a-f]{64}", str(build["cmake_cache_sha256"])) is None \
            or re.fullmatch(r"[0-9a-f]{64}", str(build["compile_commands_sha256"])) is None:
        die("candidate build binary/compiler digest differs")
    return binary, build_path, version, dict(build)


def command_capture_record(args: argparse.Namespace) -> dict[str, Any]:
    capture = require_capture(args.capture_id)
    root = directory(args.capture_root, "capture root")
    output = root / "capture-record.json"
    new_path(output, "capture record")
    binary, build_manifest, version_output, _ = _validate_binary_evidence(
        args.binary, args.build_manifest, args.version_output)
    binary_digest = sha256(binary)
    raw_link = regular(args.raw_link_manifest, "raw-link manifest")
    raw_staging = regular(args.raw_link_staging, "raw-link staging manifest")
    authority_manifest = regular(args.authority_manifest, "authority manifest")
    config_proof_path = _require_capture_artifact(
        args.config_proof, root, "capture config proof")
    _validate_config_proof(config_proof_path, args.capture_id)
    _validate_raw_link_manifest(raw_link, args.capture_id)
    _validate_raw_link_staging(raw_staging, raw_link, args.capture_id)
    _validate_authority_staging(authority_manifest, args.capture_id)
    raw_manifest_path = _require_capture_artifact(
        root / "raw-input-manifest.json", root, "raw-input manifest")
    _validate_raw_input_manifest(
        raw_manifest_path, root, args.capture_id,
        expected_raw_link=raw_link, expected_raw_staging=raw_staging,
        expected_authority=authority_manifest)
    ptc_paths = parse_obs_paths(args.ptc, capture["observations"], "PTC output")
    raw_paths = parse_obs_paths(args.raw_provenance, capture["observations"],
                                "raw provenance output")
    if _full_ptc_inventory(root) != set(ptc_paths.values()):
        die("capture root full-PTC inventory differs from the explicit observation set")
    ptc_rows = []
    provenance_rows = []
    common_native: float | None = None
    common_effective: float | None = None
    for obsnum in capture["observations"]:
        _require_capture_artifact(ptc_paths[obsnum], root, "PTC output")
        _require_capture_artifact(raw_paths[obsnum], root, "raw provenance output")
        ptc = inspect_ptc(ptc_paths[obsnum], obsnum)
        provenance = realized_raw_provenance(raw_paths[obsnum], obsnum, ptc)
        native = provenance["native_fsmp_hz"]
        effective = provenance["effective_d_fsmp_hz"]
        if common_native is None:
            common_native, common_effective = native, effective
        elif struct.pack("=d", common_native) != struct.pack("=d", native) \
                or struct.pack("=d", common_effective) != struct.pack("=d", effective):
            die("capture observations do not share exact native/effective rates")
        scan_rows = []
        for scan in ptc["scan_order"]:
            exposure = float(np.float64(scan["sample_count"]) / np.float64(effective))
            if not math.isfinite(exposure) or exposure <= 0.0:
                die("mapmaking exposure is not finite positive")
            scan_rows.append({
                "scan_index": scan["scan_index"],
                "output_scan_index": scan["output_scan_index"],
                "sample_start": scan["sample_start"],
                "sample_stop_inclusive": scan["sample_stop_inclusive"],
                "sample_count": scan["sample_count"],
                "exposure_s": exact_float(
                    exposure, "binary64(sample_count/telescope.d_fsmp)"),
            })
        ptc_rows.append({
            "obsnum": obsnum, "path": ptc["path"],
            "size_bytes": ptc["size_bytes"], "sha256": ptc["sha256"],
            "tod_output_type": "ptc", "mode": "full", "indices": "all",
            "native_samprate_hz": exact_float(native, "telescope.fsmp"),
            "sample_count": ptc["sample_count"],
            "detector_count": ptc["detector_count"],
            "scan_count": ptc["scan_count"], "scan_order": scan_rows,
        })
        provenance_rows.append({
            **{key: provenance[key] for key in (
                "obsnum", "path", "size_bytes", "sha256", "schema_version",
                "completed_scan_count", "required_timestream_write_count",
                "ptc_scan_count", "ptc_sample_count", "scan_cardinality_matches")},
            "native_fsmp_hz": exact_float(native, "telescope.fsmp"),
            "effective_d_fsmp_hz": exact_float(effective, "telescope.d_fsmp"),
        })
    assert common_native is not None and common_effective is not None
    interval = float(np.float64(1.0) / np.float64(common_effective))
    record = {
        "schema_version": "sci-map-001-ed2-capture-record-v1",
        "request_id": REQUEST_ID, "revision": REVISION,
        "candidate_sha": CANDIDATE, "candidate_tree": CANDIDATE_TREE,
        "capture_id": args.capture_id, "mode": capture["mode"],
        "target_observations": capture["observations"],
        "pointing_support_observations": capture["support"],
        "binary_sha256": binary_digest,
        "raw_input_manifest": {"path": str(raw_manifest_path),
                               "sha256": sha256(raw_manifest_path)},
        "retained": True,
        "binary": {"path": str(binary), "sha256": binary_digest,
                   "build_manifest_path": str(build_manifest),
                   "build_manifest_sha256": sha256(build_manifest),
                   "version_output_path": str(version_output),
                   "version_output_sha256": sha256(version_output),
                   "ordinary": True, "instrumented": False},
        "staging": {"raw_link_manifest_path": str(raw_link),
                    "raw_link_manifest_sha256": sha256(raw_link),
                    "raw_link_staging_path": str(raw_staging),
                    "raw_link_staging_sha256": sha256(raw_staging),
                    "authority_manifest_path": str(authority_manifest),
                    "authority_manifest_sha256": sha256(authority_manifest)},
        "config_proof": {"path": str(config_proof_path),
                         "sha256": sha256(config_proof_path), "passed": True},
        "ptc_outputs": ptc_rows,
        "rates": {"native_fsmp_hz": exact_float(common_native, "telescope.fsmp"),
                  "effective_d_fsmp_hz": exact_float(common_effective,
                                                      "telescope.d_fsmp"),
                  "sample_interval_s": exact_float(
                      interval, "binary64(1/telescope.d_fsmp)")},
        "realized_provenance": provenance_rows,
        "retention": {"retained": True, "automatic_cleanup": False,
                      "retain_through": ["fresh-map-reaudit", "focused-expansion"]},
    }
    _validate_json_schema(record, "capture-record.schema.json")
    write_new(output, canonical_json(record) + b"\n")
    return record


def command_verify_capture_record(args: argparse.Namespace) -> dict[str, Any]:
    record_path = regular(args.capture_record, "capture record")
    if record_path.name != "capture-record.json":
        die("capture record must use the canonical basename")
    root = directory(record_path.parent, "capture root")
    record = read_json(record_path)
    _validate_json_schema(record, "capture-record.schema.json")
    capture_id = record.get("capture_id") if isinstance(record, Mapping) else None
    capture = require_capture(str(capture_id))
    if record.get("request_id") != REQUEST_ID \
            or record.get("revision") != REVISION \
            or record.get("candidate_sha") != CANDIDATE \
            or record.get("candidate_tree") != CANDIDATE_TREE \
            or record.get("target_observations") != capture["observations"] \
            or record.get("pointing_support_observations") != capture["support"] \
            or record.get("mode") != capture["mode"] \
            or record.get("retained") is not True:
        die("capture record identity or retention differs")
    binary_binding = record["binary"]
    binary, build_manifest, version_output, _ = _validate_binary_evidence(
        Path(binary_binding["path"]), Path(binary_binding["build_manifest_path"]),
        Path(binary_binding["version_output_path"]))
    binary_digest = sha256(binary)
    if record["binary_sha256"] != binary_digest \
            or binary_binding["sha256"] != binary_digest \
            or binary_binding["build_manifest_sha256"] != sha256(build_manifest) \
            or binary_binding["version_output_sha256"] != sha256(version_output) \
            or binary_binding["ordinary"] is not True \
            or binary_binding["instrumented"] is not False:
        die("capture record binary evidence differs")
    staging = record["staging"]
    raw_link = regular(
        Path(staging["raw_link_manifest_path"]), "recorded raw-link manifest")
    raw_staging = regular(
        Path(staging["raw_link_staging_path"]), "recorded raw-link staging")
    authority = regular(
        Path(staging["authority_manifest_path"]), "recorded authority staging")
    if staging["raw_link_manifest_sha256"] != sha256(raw_link) \
            or staging["raw_link_staging_sha256"] != sha256(raw_staging) \
            or staging["authority_manifest_sha256"] != sha256(authority):
        die("capture record staging digest differs")
    _validate_raw_link_manifest(raw_link, str(capture_id))
    _validate_raw_link_staging(raw_staging, raw_link, str(capture_id))
    _validate_authority_staging(authority, str(capture_id))
    raw_binding = record["raw_input_manifest"]
    raw_manifest_path = _require_capture_artifact(
        Path(raw_binding["path"]), root, "automatic raw-input manifest")
    if raw_binding["sha256"] != sha256(raw_manifest_path):
        die("capture record raw-input manifest digest differs")
    _validate_raw_input_manifest(
        raw_manifest_path, root, str(capture_id),
        expected_raw_link=raw_link, expected_raw_staging=raw_staging,
        expected_authority=authority)
    config_binding = record["config_proof"]
    config_path = _require_capture_artifact(
        Path(config_binding["path"]), root, "capture config proof")
    if config_binding["passed"] is not True \
            or config_binding["sha256"] != sha256(config_path):
        die("capture record config proof digest/outcome differs")
    _validate_config_proof(config_path, str(capture_id))

    rates = record["rates"]
    record_native = parse_exact(
        rates["native_fsmp_hz"], "capture native rate",
        authority="telescope.fsmp")
    record_effective = parse_exact(
        rates["effective_d_fsmp_hz"], "capture effective rate",
        authority="telescope.d_fsmp")
    record_interval = parse_exact(
        rates["sample_interval_s"], "capture sample interval",
        authority="binary64(1/telescope.d_fsmp)")
    expected_interval = float(
        np.float64(1.0) / np.float64(record_effective))
    if record_native <= 0.0 or record_effective <= 0.0 \
            or record_interval <= 0.0 \
            or struct.pack("=d", record_interval) != \
            struct.pack("=d", expected_interval):
        die("capture record native/effective rate or sample interval differs")

    ptcs: dict[int, dict[str, Any]] = {}
    expected_paths: set[Path] = set()
    for expected_obsnum, row in zip(capture["observations"], record["ptc_outputs"]):
        if row["obsnum"] != expected_obsnum:
            die("capture record PTC observation order differs")
        path = _require_capture_artifact(
            Path(row["path"]), root, f"PTC observation {expected_obsnum}")
        inspected = inspect_ptc(path, expected_obsnum)
        if row["path"] != inspected["path"] \
                or row["size_bytes"] != inspected["size_bytes"] \
                or row["sha256"] != inspected["sha256"] \
                or row["sample_count"] != inspected["sample_count"] \
                or row["detector_count"] != inspected["detector_count"] \
                or row["scan_count"] != inspected["scan_count"]:
            die("capture record PTC path/digest/cardinality differs")
        row_native = parse_exact(
            row["native_samprate_hz"], "PTC native SAMPRATE",
            authority="telescope.fsmp")
        if struct.pack("=d", row_native) != \
                struct.pack("=d", inspected["native_fsmp_hz"]) \
                or struct.pack("=d", row_native) != \
                struct.pack("=d", record_native):
            die("capture record PTC native SAMPRATE differs")
        recorded_scans = [{key: scan[key] for key in (
            "scan_index", "output_scan_index", "sample_start",
            "sample_stop_inclusive", "sample_count")}
                          for scan in row["scan_order"]]
        inspected_scans = [{key: scan[key] for key in (
            "scan_index", "output_scan_index", "sample_start",
            "sample_stop_inclusive", "sample_count")}
                           for scan in inspected["scan_order"]]
        if recorded_scans != inspected_scans:
            die("capture record PTC scan order differs")
        for scan in row["scan_order"]:
            exposure = parse_exact(
                scan["exposure_s"], "capture scan exposure",
                authority="binary64(sample_count/telescope.d_fsmp)")
            expected_exposure = float(
                np.float64(scan["sample_count"]) /
                np.float64(record_effective))
            if exposure <= 0.0 or struct.pack("=d", exposure) != \
                    struct.pack("=d", expected_exposure):
                die("capture record scan exposure differs from telescope.d_fsmp")
        ptcs[expected_obsnum] = inspected
        expected_paths.add(path)
    if _full_ptc_inventory(root) != expected_paths:
        die("live capture root full-PTC inventory differs from capture record")

    for expected_obsnum, row in zip(
            capture["observations"], record["realized_provenance"]):
        if row["obsnum"] != expected_obsnum:
            die("capture record raw-provenance observation order differs")
        path = _require_capture_artifact(
            Path(row["path"]), root,
            f"raw-timestream provenance observation {expected_obsnum}")
        realized = realized_raw_provenance(path, expected_obsnum, ptcs[expected_obsnum])
        for name in (
                "path", "size_bytes", "sha256", "schema_version",
                "completed_scan_count", "required_timestream_write_count",
                "ptc_scan_count", "ptc_sample_count", "scan_cardinality_matches"):
            if row[name] != realized[name]:
                die(f"capture record raw provenance {name} differs")
        for name, authority in (("native_fsmp_hz", "telescope.fsmp"),
                                ("effective_d_fsmp_hz", "telescope.d_fsmp")):
            recorded = parse_exact(row[name], name, authority=authority)
            expected = realized[
                "native_fsmp_hz" if name == "native_fsmp_hz"
                else "effective_d_fsmp_hz"]
            capture_rate = record_native if name == "native_fsmp_hz" \
                else record_effective
            if struct.pack("=d", recorded) != struct.pack("=d", expected) \
                    or struct.pack("=d", recorded) != \
                    struct.pack("=d", capture_rate):
                die(f"capture record raw provenance {name} differs")
    result = {
        "schema_version": "sci-map-001-ed2-capture-record-verification-v1",
        "capture_id": capture_id,
        "capture_record_sha256": sha256(record_path),
        "full_ptc_count": len(expected_paths),
        "status": "pass",
    }
    print(json.dumps(result, sort_keys=True))
    return result


def command_self_check(_: argparse.Namespace) -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sci-map-ed2-capture-") as text:
        root = Path(text).resolve()
        governed = []
        for index in range(5):
            path = root / f"root-{index}"
            path.mkdir()
            (path / "fixture.txt").write_text(f"fixture-{index}\n", encoding="utf-8")
            governed.append(path)
        inventory = resource_inventory(governed)
        if len(inventory["entries"]) != 10:
            die("resource inventory self-check cardinality differs")
        value = 122.0703125
        node = exact_float(value, "telescope.d_fsmp")
        if struct.pack("=d", parse_exact(node, "self-check")) != struct.pack("=d", value):
            die("exact-float self-check differs")
        if set(flatten({"a": {"b": [1, 2]}})) != {"a.b.0", "a.b.1"}:
            die("merged-config flatten self-check differs")
    result = {"schema_version": "sci-map-001-ed2-capture-self-check-v1",
              "candidate_sha": CANDIDATE, "status": "pass",
              "resource_inventory_entries": 10}
    print(json.dumps(result, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    raw = commands.add_parser("raw-manifest")
    raw.add_argument("--capture-id", choices=tuple(CAPTURES), required=True)
    raw.add_argument("--canonical-root", type=Path, required=True)
    raw.add_argument("--selection", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    raw.set_defaults(function=command_raw_manifest)

    stage_raw = commands.add_parser("stage-raw")
    stage_raw.add_argument("--manifest", type=Path, required=True)
    stage_raw.add_argument("--destination", type=Path, required=True)
    stage_raw.add_argument("--output", type=Path, required=True)
    stage_raw.set_defaults(function=command_stage_raw)

    authorities = commands.add_parser("stage-authorities")
    authorities.add_argument("--capture-id", choices=tuple(CAPTURES), required=True)
    authorities.add_argument("--selection", type=Path, required=True)
    authorities.add_argument("--apt-destination", type=Path, required=True)
    authorities.add_argument("--ppt-destination", type=Path)
    authorities.add_argument("--output", type=Path, required=True)
    authorities.set_defaults(function=command_stage_authorities)

    inventory = commands.add_parser("config-inventory")
    inventory.add_argument("--mode", choices=("point", "science"), required=True)
    inventory.add_argument("--numbered-dir", type=Path, required=True)
    inventory.add_argument("--included-fragment", type=Path, action="append", default=[])
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(function=command_config_inventory)

    overlay = commands.add_parser("capture-overlay")
    overlay.add_argument("--reference-overlay", type=Path, required=True)
    overlay.add_argument("--candidate-binary", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    overlay.set_defaults(function=command_capture_overlay)

    proof = commands.add_parser("config-proof")
    proof.add_argument("--capture-id", choices=tuple(CAPTURES), required=True)
    proof.add_argument("--fixed-config", type=Path, required=True)
    proof.add_argument("--capture-config", type=Path, required=True)
    proof.add_argument("--fixed-inventory", type=Path, required=True)
    proof.add_argument("--capture-inventory", type=Path, required=True)
    proof.add_argument("--output", type=Path, required=True)
    proof.set_defaults(function=command_config_proof)

    projection = commands.add_parser("resource-projection")
    projection.add_argument("--stage", required=True)
    projection.add_argument("--source", type=Path, required=True)
    projection.add_argument("--output", type=Path, required=True)
    projection.set_defaults(function=command_resource_projection)

    resource = commands.add_parser("resource-record")
    resource.add_argument("--stage", required=True)
    resource.add_argument("--phase", choices=("pre", "post"), required=True)
    resource.add_argument("--projection-authority", type=Path)
    resource.add_argument("--filesystem-root", type=Path, required=True)
    resource.add_argument("--governed-root", type=Path, action="append", required=True)
    resource.add_argument("--inventory", type=Path, required=True)
    resource.add_argument("--record", type=Path, required=True)
    resource.set_defaults(function=command_resource_record)

    manifest = commands.add_parser("build-raw-input-manifest")
    manifest.add_argument("--capture-id", choices=tuple(CAPTURES), required=True)
    manifest.add_argument("--capture-root", type=Path, required=True)
    manifest.add_argument("--ptc", action="append", required=True)
    manifest.add_argument("--raw-provenance", action="append", required=True)
    manifest.add_argument("--map-provenance", action="append", required=True)
    manifest.add_argument("--source-selection", type=Path, required=True)
    manifest.add_argument("--raw-link-manifest", type=Path, required=True)
    manifest.add_argument("--raw-link-staging", type=Path, required=True)
    manifest.add_argument("--authority-manifest", type=Path, required=True)
    manifest.set_defaults(function=command_build_raw_input_manifest)

    authority = commands.add_parser("producer-authority")
    authority.add_argument("--raw-input-manifest", type=Path, required=True)
    authority.add_argument("--obsnum", type=int, choices=(152389, 152390, 152392),
                           required=True)
    authority.add_argument("--array", choices=ARRAYS, required=True)
    authority.add_argument("--output", type=Path, required=True)
    authority.set_defaults(function=command_producer_authority)

    record = commands.add_parser("capture-record")
    record.add_argument("--capture-id", choices=tuple(CAPTURES), required=True)
    record.add_argument("--capture-root", type=Path, required=True)
    record.add_argument("--binary", type=Path, required=True)
    record.add_argument("--build-manifest", type=Path, required=True)
    record.add_argument("--version-output", type=Path, required=True)
    record.add_argument("--raw-link-manifest", type=Path, required=True)
    record.add_argument("--raw-link-staging", type=Path, required=True)
    record.add_argument("--authority-manifest", type=Path, required=True)
    record.add_argument("--config-proof", type=Path, required=True)
    record.add_argument("--ptc", action="append", required=True)
    record.add_argument("--raw-provenance", action="append", required=True)
    record.set_defaults(function=command_capture_record)

    verify_record = commands.add_parser("verify-capture-record")
    verify_record.add_argument("--capture-record", type=Path, required=True)
    verify_record.set_defaults(function=command_verify_capture_record)

    check = commands.add_parser("self-check")
    check.set_defaults(function=command_self_check)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.function(args)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
