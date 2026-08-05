#!/usr/bin/env python3
"""Frozen, fail-closed verifier for SCI-MAP-001-UNITY-001.

This program never contacts Unity and never executes Citlali.  It materializes
the seven authorized expert overlays, freezes result paths and hashes, and
checks returned FITS products against the candidate product registry and an
independent per-sample contribution ledger.  A successful invocation means
only that this program's checks passed; it is not a conformance decision and
does not close any audit finding.
"""

from __future__ import annotations

import argparse
import csv
import copy
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage


PROGRAM_SCHEMA = "sci-map-001-analysis-program-v1"
CAMPAIGN_SCHEMA = "sci-map-unity-campaign-v1"
OWNER_SCHEMA = "sci-map-unity-owner-values-v1"
INPUT_SCHEMA = "sci-map-unity-analysis-inputs-v1"
RAW_MANIFEST_SCHEMA = "sci-map-001-raw-input-manifest-v1"
COLLECTION_SCHEMAS = {
    "sci-map-unity-result-collection-v1",
    "sci-map-001-result-collection-v1",
}
COLLECTION_CASE_FILE_FIELDS = (
    "preflight_manifest", "submit_record", "stdout", "stderr",
    "exit_record", "slurm_accounting",
)
SLURM_FIELDS = (
    "JobIDRaw", "JobName", "Partition", "AllocCPUS", "NodeList", "State",
    "ExitCode", "Elapsed", "MaxRSS", "ReqMem", "Submit", "Start", "End",
)
RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$")
SLURM_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
LEDGER_SCHEMA = "sci-map-001-independent-sample-ledger-v1"
REQUEST_ID = "SCI-MAP-001-UNITY-001"
CANDIDATE_SHA = "ed28dafb37f9113c0d3c95297148157129a90886"
ARRAYS = ("a1100", "a1400", "a2000")
REALIZATIONS = 64
REGISTERED_ATOL = 2.0e-8
REGISTERED_RTOL = 1.0e-10
SOURCE_ROLES = (
    "raw_timestream", "kids_fit_report", "apt", "calibration",
    "pointing_support", "projection_authority", "sample_rate_authority",
    "fwhm_authority", "target_authority",
)
MODE_POLICY_SHA256 = {
    "point": "86e4ffa03f37aae32d614e76245ca9710efe27a2c28d56cb53e3bf4753f531ad",
    "science": "50793ea4cdbc44a3e8d21afb3bb3be4ed553c7d25e97df12837e4419a29a9672",
}
TOLPROJ_VENDOR_MANIFEST_SHA256 = (
    "040c8ab8daabddf9b3194d21847a3cd63c1cfef1b574f49fe22ff051f60e4b5b"
)
TOLPROJ_FROZEN_NUMBERED_SHA256 = {
    "point": {
        "60_pointing_internal_policy.yaml":
            "cca1a45572f26e81d9325c9ebd20ce1a60fd4d691485be37c20439aaf4bf2b99",
        "71_pointing_runtime.yaml":
            "00707eb203531c80c66d91ea743ef475a2a44e4a581710cdf75073f46151faa5",
        "81_pointing_defaults.yaml":
            "f7cc1e861aa0340bf178e6c9a221202990a17b2c667dd190acdbae4378028a3d",
        "82_pointing_products.yaml":
            "1fbb71a63866e93b181c4d3152e450d4393188af1b83474d1e5e4e047eb7908d",
        "90_pointing_advanced_overrides.yaml":
            "910e92df1125ffad013bd474c9c017268a27bd37f15717422082dd5287ed7f1b",
    },
    "science": {
        "60_science_internal_policy.yaml":
            "1dedc01449f2d40f3c57a451c146c74274196a756c849da6b10e86bec34d6ee5",
        "71_science_runtime.yaml":
            "6b5ad9d8decc45f0ef52cca7496b56e99dcfbe532bf7ce3487c48b1fc296c4e2",
        "81_science_defaults.yaml":
            "803d5b0203725b40d8abafbb4b21f32a79f54804df696dab0e0fe3c9b41f080f",
        "82_science_products.yaml":
            "52b2508f51af080259797a3fbe8c5d2a7c45842cdc223eae12444d8c87af8a5a",
        "90_science_advanced_overrides.yaml":
            "910e92df1125ffad013bd474c9c017268a27bd37f15717422082dd5287ed7f1b",
    },
}
EDGE_BINS = (
    (0.0, 2.0), (2.0, 5.0), (5.0, 10.0),
    (10.0, 20.0), (20.0, 40.0), (40.0, math.inf),
)

F010_OBSERVATION = (
    "geometric_hits_I",
    "contributing_hits_I",
    "upstream_eligible_exposure_I",
    "retained_exposure_I",
    "normalization_support_I",
    "science_policy_support_I",
    "science_valid_I",
    "coverage_I",
    "coverage_bool_I",
)
F010_COADD = (
    "geometric_hits_I",
    "contributing_hits_I",
    "coadd_observation_count_I",
    "upstream_eligible_exposure_I",
    "retained_exposure_I",
    "normalization_support_I",
    "science_policy_support_I",
    "science_valid_I",
    "coverage_I",
    "coverage_bool_I",
)
COMMON_PLANES = ("signal_I", "weight_I", "kernel_I")
INTEGER_PLANES = (
    "geometric_hits_I",
    "contributing_hits_I",
    "coadd_observation_count_I",
    "normalization_support_I",
    "science_policy_support_I",
    "science_valid_I",
    "coverage_bool_I",
)
FLOAT_PLANES = (
    "signal_I",
    "weight_I",
    "kernel_I",
    "upstream_eligible_exposure_I",
    "retained_exposure_I",
    "coverage_I",
)
EMPIRICAL_PLANES = (
    "weight_formal_I",
    "noise_variance_I",
    "sig2noise_I",
    "sig2noise_pixel_I",
)
COADD_FORBIDDEN = (
    "weight_formal_I",
    "noise_variance_I",
    "formal_standardized_signal_I",
    "sig2noise_I",
    "sig2noise_pixel_I",
    "point_source_flux_I",
    "point_source_uncertainty_I",
    "sig2noise_point_source_I",
)
OBSERVATION_PRODUCTS_OFF_FORBIDDEN = tuple(
    name for name in COADD_FORBIDDEN
    if name != "formal_standardized_signal_I"
)

EXPECTED_CASES: dict[str, dict[str, Any]] = {
    "P-SEQ": dict(mode="point", jobkey="sci_map_001_point_seq", coadd=False,
                  coverage_cut=0.1, products_enabled=True,
                  parallel_policy="seq", threads=1, cpus=1,
                  expected_observations=[152389]),
    "P-OMP": dict(mode="point", jobkey="sci_map_001_point_omp", coadd=False,
                  coverage_cut=0.1, products_enabled=True,
                  parallel_policy="omp", threads=6, cpus=6,
                  expected_observations=[152389]),
    "S-C-SEQ": dict(mode="science", jobkey="sci_map_001_science_coadd_seq",
                    coadd=True, coverage_cut=0.5, products_enabled=False,
                    parallel_policy="seq", threads=1, cpus=1,
                    expected_observations=[152390, 152392]),
    "S-C-OMP": dict(mode="science", jobkey="sci_map_001_science_coadd_omp",
                    coadd=True, coverage_cut=0.5, products_enabled=False,
                    parallel_policy="omp", threads=16, cpus=16,
                    expected_observations=[152390, 152392]),
    "S-E-SEQ": dict(mode="science", jobkey="sci_map_001_science_empirical_seq",
                    coadd=False, coverage_cut=0.5, products_enabled=True,
                    parallel_policy="seq", threads=1, cpus=1,
                    expected_observations=[152390, 152392]),
    "S-E-OMP": dict(mode="science", jobkey="sci_map_001_science_empirical_omp",
                    coadd=False, coverage_cut=0.5, products_enabled=True,
                    parallel_policy="omp", threads=16, cpus=16,
                    expected_observations=[152390, 152392]),
    # Historical jobkey retained intentionally; repaired expected result is 0.
    "S-X-SEQ": dict(mode="science", jobkey="sci_map_001_science_expected_failure",
                    coadd=True, coverage_cut=0.5, products_enabled=True,
                    parallel_policy="seq", threads=1, cpus=1,
                    expected_observations=[152390, 152392]),
}

SCIENCE_ALGORITHMS = {
    "contract_version": "citlali-science-map-contract-v1",
    "order_statistic_algorithm": "positive-order-statistic-floor-075-midpoint-v1",
    "normalization_support_algorithm": "finite-positive-ge-threshold-coverage-cut-div10-v1",
    "science_policy_support_algorithm": "finite-positive-ge-threshold-coverage-cut-v1",
    "validity_algorithm": "normalization-and-policy-and-finite-companions-and-identity-v1",
    "contribution_algorithm": "ordinary-naive-finite-positive-coefficient-v1",
    "coadd_estimator": "centered-integer-normalized-weighted-mean-L-identity-v1",
    "nonfinite_policy": "explicit-invalid-skip-valid-nonfinite-fail-v1",
}
ESTIMATOR_IDENTITY = "ordinary-naive-normalized-gridding-v1"
COEFFICIENT_POLICY = "nonprecision-normalization-coefficient-v1"
PARALLEL_POLICY = "within-scan-exact-scan-farm-2gamma-n-sumabs-v1"
NORMALIZATION_STAGE_OBS = "pre-observation-normalization-accumulated-coefficient"
NORMALIZATION_STAGE_COADD = "pre-coadd-normalization-sum-of-admitted-observation-coefficients"


class EvidenceError(RuntimeError):
    """A fail-closed package, input, or evidence error."""


def die(message: str) -> "NoReturn":
    raise EvidenceError(message)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"cannot read YAML {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.chmod(path, mode)
    except FileExistsError as exc:
        raise EvidenceError(f"refusing to overwrite existing path: {path}") from exc
    except OSError as exc:
        with contextlib.suppress(OSError):
            path.unlink()
        raise EvidenceError(f"cannot write {path}: {exc}") from exc


def exact_float_equal(left: float, right: float) -> bool:
    return struct.pack("=d", float(left)) == struct.pack("=d", float(right))


def exact_float_node(node: Mapping[str, Any], label: str) -> float:
    if not isinstance(node, Mapping):
        die(f"{label}: exact binary64 record is not a mapping")
    if node.get("encoding") != "binary64-max-digits10-and-c99-hexfloat":
        die(f"{label}: unsupported exact binary64 encoding")
    try:
        numeric = float(node["numeric"])
        hexadecimal = float.fromhex(str(node["hex"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"{label}: malformed exact binary64 record") from exc
    if not exact_float_equal(numeric, hexadecimal):
        die(f"{label}: decimal and hexadecimal encodings differ")
    return hexadecimal


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        die(f"{label}: expected an object")
    missing = sorted(set(keys) - set(value))
    extra = sorted(set(value) - set(keys))
    if missing or extra:
        die(f"{label}: keys differ; missing={missing}, extra={extra}")
    return value


def ordered_unique_integers(value: Any, label: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, list) or not value or any(
            not isinstance(item, int) or isinstance(item, bool) for item in value):
        die(f"{label}: expected a nonempty integer array")
    if value != sorted(set(value)):
        die(f"{label}: values must be strictly ordered and unique")
    minimum = 1 if positive else 0
    if any(item < minimum for item in value):
        die(f"{label}: contains an out-of-domain integer")
    return value


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def verified_regular_source(path_value: Any, digest_value: Any, size_value: Any,
                            label: str, forbidden_roots: Sequence[Path]) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        die(f"{label}: source path must be absolute")
    lexical = Path(path_value)
    if lexical.is_symlink():
        die(f"{label}: source path must not be a symlink")
    try:
        path = lexical.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"{label}: cannot resolve source path: {exc}") from exc
    if not path.is_file():
        die(f"{label}: source is not a regular file: {path}")
    if any(is_within(path, root.resolve()) for root in forbidden_roots):
        die(f"{label}: source is inside a reduction-output root: {path}")
    if not isinstance(size_value, int) or isinstance(size_value, bool) or \
            size_value <= 0 or path.stat().st_size != size_value:
        die(f"{label}: source size differs from the manifest")
    if not isinstance(digest_value, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest_value) is None or sha256(path) != digest_value:
        die(f"{label}: source digest differs from the manifest")
    return path


@dataclass(frozen=True)
class RawManifestAuthority:
    path: Path
    digest: str
    mode: str
    producer_identity: str
    producer_program: Path
    producer_program_sha256: str
    sources: Mapping[str, Mapping[str, Any]]
    memberships: Mapping[tuple[int, str], Mapping[str, Any]]


def validate_raw_input_manifest(
        path: Path, mode: str, expected_observations: Sequence[int],
        *, forbidden_roots: Sequence[Path] = ()) -> RawManifestAuthority:
    manifest_path = path.resolve()
    raw = require_exact_keys(read_json(manifest_path), (
        "schema_version", "request_id", "candidate_sha", "mode",
        "observations", "arrays", "producer", "source_records", "memberships",
    ), f"{mode} raw-input manifest")
    expected_frame = "altaz" if mode == "point" else "fk5"
    expected_obs = list(expected_observations)
    if raw["schema_version"] != RAW_MANIFEST_SCHEMA or \
            raw["request_id"] != REQUEST_ID or raw["candidate_sha"] != CANDIDATE_SHA:
        die(f"{mode} raw-input manifest identity differs")
    if raw["mode"] != mode or raw["observations"] != expected_obs or \
            raw["arrays"] != list(ARRAYS):
        die(f"{mode} raw-input manifest observation/array identity differs")

    producer = require_exact_keys(raw["producer"], (
        "identity", "program_path", "program_sha256", "invocation",
    ), f"{mode} raw-input producer")
    producer_identity = producer["identity"]
    invocation = producer["invocation"]
    if not isinstance(producer_identity, str) or not producer_identity.strip() or \
            producer_identity != producer_identity.strip() or len(producer_identity) > 256:
        die(f"{mode} raw-input producer identity is unresolved")
    if not isinstance(invocation, list) or not invocation or any(
            not isinstance(item, str) or not item for item in invocation):
        die(f"{mode} raw-input producer invocation is unresolved")
    producer_program = verified_regular_source(
        producer["program_path"], producer["program_sha256"],
        Path(str(producer["program_path"])).stat().st_size
        if Path(str(producer["program_path"])).is_file() else None,
        f"{mode} raw-input producer program", forbidden_roots)

    source_rows = raw["source_records"]
    if not isinstance(source_rows, list) or len(source_rows) < len(SOURCE_ROLES):
        die(f"{mode} raw-input source inventory is incomplete")
    sources: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(source_rows):
        row = dict(require_exact_keys(value, (
            "id", "role", "path", "size_bytes", "sha256",
            "obsnums", "arrays", "networks",
        ), f"{mode} source_records[{index}]"))
        identifier = row["id"]
        if not isinstance(identifier, str) or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identifier) is None or \
                identifier in sources or row["role"] not in SOURCE_ROLES:
            die(f"{mode} source record identity/role is invalid: {identifier!r}")
        obsnums = ordered_unique_integers(
            row["obsnums"], f"{mode} {identifier} obsnums", positive=True)
        if not set(obsnums).issubset(expected_obs):
            die(f"{mode} source record observation membership differs: {identifier}")
        arrays = row["arrays"]
        if not isinstance(arrays, list) or not arrays or arrays != [
                array for array in ARRAYS if array in arrays] or any(
                    not isinstance(array, str) or array not in ARRAYS for array in arrays):
            die(f"{mode} source record array membership differs: {identifier}")
        networks = ordered_unique_integers(
            row["networks"], f"{mode} {identifier} networks")
        source_path = verified_regular_source(
            row["path"], row["sha256"], row["size_bytes"],
            f"{mode} source record {identifier}", forbidden_roots)
        row["resolved_path"] = str(source_path)
        row["obsnums"] = obsnums
        row["networks"] = networks
        sources[identifier] = row

    expected_pairs = [(obsnum, array) for obsnum in expected_obs for array in ARRAYS]
    membership_rows = raw["memberships"]
    if not isinstance(membership_rows, list) or len(membership_rows) != len(expected_pairs):
        die(f"{mode} raw-input membership cardinality differs")
    memberships: dict[tuple[int, str], dict[str, Any]] = {}
    used_sources: set[str] = set()
    for index, (value, expected_pair) in enumerate(zip(membership_rows, expected_pairs)):
        row = dict(require_exact_keys(value, (
            "obsnum", "array", "networks", "source_refs", "projection",
            "record_order", "projection_record_count", "scan_order",
            "detector_order",
        ), f"{mode} memberships[{index}]"))
        pair = (row["obsnum"], row["array"])
        if pair != expected_pair:
            die(f"{mode} raw-input membership order differs: {pair} != {expected_pair}")
        networks = ordered_unique_integers(row["networks"], f"{mode} {pair} networks")
        refs = require_exact_keys(row["source_refs"], SOURCE_ROLES,
                                  f"{mode} {pair} source references")
        for role in SOURCE_ROLES:
            identifiers = refs[role]
            if not isinstance(identifiers, list) or not identifiers or \
                    len(identifiers) != len(set(identifiers)) or any(
                        not isinstance(item, str) for item in identifiers):
                die(f"{mode} {pair} {role} references are incomplete/repeated")
            covered_networks: set[int] = set()
            for identifier in identifiers:
                source = sources.get(identifier)
                if source is None or source["role"] != role or \
                        pair[0] not in source["obsnums"] or pair[1] not in source["arrays"]:
                    die(f"{mode} {pair} {role} references an inapplicable source: {identifier}")
                covered_networks.update(source["networks"])
                used_sources.add(identifier)
            if not set(networks).issubset(covered_networks) or \
                    (role in ("raw_timestream", "kids_fit_report") and
                     covered_networks != set(networks)):
                die(f"{mode} {pair} {role} network coverage differs")
        projection = dict(require_exact_keys(row["projection"], (
            "identity_digest", "grouping", "stokes", "frame", "map_rows",
            "map_cols", "sample_rate_hz", "fwhm_arcsec", "target",
        ), f"{mode} {pair} projection"))
        if not canonical_digest_string(projection["identity_digest"]) or \
                projection["grouping"] != "array" or projection["stokes"] != "I" or \
                projection["frame"] != expected_frame:
            die(f"{mode} {pair} projection identity differs")
        for dimension in ("map_rows", "map_cols"):
            if not isinstance(projection[dimension], int) or \
                    isinstance(projection[dimension], bool) or projection[dimension] <= 0:
                die(f"{mode} {pair} {dimension} is invalid")
        sample_rate = exact_float_node(
            require_exact_keys(projection["sample_rate_hz"],
                               ("numeric", "hex", "encoding"),
                               f"{mode} {pair} sample_rate_hz"),
            f"{mode} {pair} sample_rate_hz")
        fwhm = exact_float_node(
            require_exact_keys(projection["fwhm_arcsec"],
                               ("numeric", "hex", "encoding"),
                               f"{mode} {pair} fwhm_arcsec"),
            f"{mode} {pair} fwhm_arcsec")
        target = require_exact_keys(projection["target"],
                                    ("frame", "axis1", "axis2", "unit"),
                                    f"{mode} {pair} target")
        axis1 = exact_float_node(require_exact_keys(
            target["axis1"], ("numeric", "hex", "encoding"),
            f"{mode} {pair} target axis1"), f"{mode} {pair} target axis1")
        axis2 = exact_float_node(require_exact_keys(
            target["axis2"], ("numeric", "hex", "encoding"),
            f"{mode} {pair} target axis2"), f"{mode} {pair} target axis2")
        if not math.isfinite(sample_rate) or sample_rate <= 0.0 or \
                not math.isfinite(fwhm) or fwhm <= 0.0 or \
                target["frame"] != expected_frame or target["unit"] != "deg" or \
                not math.isfinite(axis1) or not math.isfinite(axis2):
            die(f"{mode} {pair} projection numerical authority is invalid")
        projection["_sample_rate_hz"] = sample_rate
        projection["_fwhm_arcsec"] = fwhm
        projection["_target_axis1"] = axis1
        projection["_target_axis2"] = axis2
        if row["record_order"] != \
                "scan-major-detector-major-sample-minor-cartesian-v1":
            die(f"{mode} {pair} processed-term record order differs")
        declared_count = row["projection_record_count"]
        scans = row["scan_order"]
        if not isinstance(declared_count, int) or isinstance(declared_count, bool) or \
                declared_count <= 0 or not isinstance(scans, list) or not scans:
            die(f"{mode} {pair} processed-term cardinality is invalid")
        scan_identities: set[str] = set()
        normalized_scans: list[dict[str, Any]] = []
        for scan_position, scan_value in enumerate(scans):
            scan_record = dict(require_exact_keys(scan_value, (
                "scan_index", "identity", "sample_count",
            ), f"{mode} {pair} processed scan {scan_position}"))
            scan_identity = scan_record["identity"]
            sample_count = scan_record["sample_count"]
            if scan_record["scan_index"] != scan_position or \
                    not isinstance(scan_identity, str) or not scan_identity.strip() or \
                    scan_identity != scan_identity.strip() or len(scan_identity) > 256 or \
                    scan_identity in scan_identities or \
                    not isinstance(sample_count, int) or isinstance(sample_count, bool) or \
                    sample_count <= 0:
                die(f"{mode} {pair} processed scan identity/cardinality differs")
            scan_identities.add(scan_identity)
            normalized_scans.append(scan_record)
        detectors = row["detector_order"]
        if not isinstance(detectors, list) or not detectors:
            die(f"{mode} {pair} processed detector order is absent")
        detector_uids: set[str] = set()
        normalized_detectors: list[dict[str, Any]] = []
        for detector_position, detector_value in enumerate(detectors):
            detector_record = dict(require_exact_keys(detector_value, (
                "detector_index", "detector_uid", "network",
            ), f"{mode} {pair} processed detector {detector_position}"))
            detector_uid = detector_record["detector_uid"]
            network = detector_record["network"]
            if detector_record["detector_index"] != detector_position or \
                    not isinstance(detector_uid, str) or not detector_uid.strip() or \
                    detector_uid != detector_uid.strip() or len(detector_uid) > 256 or \
                    detector_uid in detector_uids or not isinstance(network, int) or \
                    isinstance(network, bool) or network not in networks:
                die(f"{mode} {pair} processed detector identity differs")
            detector_uids.add(detector_uid)
            normalized_detectors.append(detector_record)
        if {record["network"] for record in normalized_detectors} != set(networks):
            die(f"{mode} {pair} processed detector networks differ")
        expanded_count = sum(record["sample_count"] for record in normalized_scans) * \
            len(normalized_detectors)
        if expanded_count != declared_count:
            die(f"{mode} {pair} processed-term expanded cardinality differs: "
                f"{expanded_count} != {declared_count}")
        row["scan_order"] = normalized_scans
        row["detector_order"] = normalized_detectors
        row["networks"] = networks
        row["projection"] = projection
        memberships[pair] = row
    if used_sources != set(sources) or \
            {row["role"] for row in sources.values()} != set(SOURCE_ROLES):
        die(f"{mode} raw-input manifest contains unused or missing source roles")
    for obsnum in expected_obs:
        scan_orders = [memberships[(obsnum, array)]["scan_order"] for array in ARRAYS]
        if any(order != scan_orders[0] for order in scan_orders[1:]):
            die(f"{mode} observation {obsnum} scan order differs across arrays")
    return RawManifestAuthority(
        path=manifest_path, digest=sha256(manifest_path), mode=mode,
        producer_identity=producer_identity, producer_program=producer_program,
        producer_program_sha256=str(producer["program_sha256"]),
        sources=sources, memberships=memberships)


def raw_authority_record(authority: RawManifestAuthority) -> dict[str, Any]:
    source_records = []
    for source in authority.sources.values():
        source_records.append({key: value for key, value in source.items()
                               if key != "resolved_path"})
    membership_records = []
    for membership in authority.memberships.values():
        record = copy.deepcopy(dict(membership))
        record["projection"] = {
            key: value for key, value in record["projection"].items()
            if not key.startswith("_")
        }
        membership_records.append(record)
    return {
        "schema_version": RAW_MANIFEST_SCHEMA,
        "path": str(authority.path),
        "sha256": authority.digest,
        "mode": authority.mode,
        "producer": {
            "identity": authority.producer_identity,
            "program_path": str(authority.producer_program),
            "program_sha256": authority.producer_program_sha256,
        },
        "source_records": source_records,
        "memberships": membership_records,
    }


def cxx_hexfloat(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    mantissa, exponent = value.hex().split("p")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return mantissa + "p" + exponent


class CanonicalDigest:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def add_string(self, value: Any) -> None:
        encoded = str(value).encode("utf-8")
        self._digest.update(str(len(encoded)).encode("ascii"))
        self._digest.update(b":")
        self._digest.update(encoded)
        self._digest.update(b";")

    def add_integer(self, value: Any) -> None:
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        self.add_string(str(int(value)))

    def add_double(self, value: float) -> None:
        self.add_string(cxx_hexfloat(float(value)))

    def add_matrix(self, matrix: np.ndarray) -> None:
        value = np.asarray(matrix)
        if value.ndim != 2:
            die(f"canonical digest matrix must be two-dimensional, got {value.shape}")
        self.add_integer(value.shape[0])
        self.add_integer(value.shape[1])
        integral = value.dtype.kind in ("i", "u", "b")
        for col in range(value.shape[1]):
            for row in range(value.shape[0]):
                if integral:
                    self.add_integer(value[row, col])
                else:
                    self.add_double(value[row, col])

    def finish(self) -> str:
        return "canonical-hexfloat-sha256-v1:" + self._digest.hexdigest()


def digest_exact_sequence(digest: CanonicalDigest, values: Any,
                          label: str) -> None:
    if not isinstance(values, list):
        die(f"{label}: expected exact-double sequence")
    digest.add_integer(len(values))
    for index, node in enumerate(values):
        digest.add_double(exact_float_node(node, f"{label}[{index}]"))


def recompute_bundle_identity_digest(identity: Mapping[str, Any]) -> str:
    digest = CanonicalDigest()
    for key in ("contract_version", "grouping", "signal_unit",
                "estimator_identity", "response_identity",
                "parallel_equivalence_policy"):
        digest.add_string(identity.get(key, ""))
    companions = identity.get("required_companions")
    if not isinstance(companions, list):
        die("bundle identity required_companions is malformed")
    digest.add_integer(len(companions))
    for companion in companions:
        digest.add_string(companion)
    policies = identity.get("policies")
    if not isinstance(policies, Mapping):
        die("bundle identity policies are malformed")
    for key in ("validity", "coefficient", "normalization_support",
                "science_policy_support", "nonfinite"):
        digest.add_string(policies.get(key, ""))
    wcs = identity.get("wcs")
    if not isinstance(wcs, Mapping):
        die("bundle identity WCS is malformed")
    digest.add_string(wcs.get("coordinate_frame", ""))
    digest.add_string(wcs.get("projection", ""))
    for key in ("axis_types", "axis_units"):
        values = wcs.get(key)
        if not isinstance(values, list):
            die(f"bundle identity WCS {key} is malformed")
        digest.add_integer(len(values))
        for value in values:
            digest.add_string(value)
    for key in ("pixel_scale", "reference_world", "reference_pixel"):
        digest_exact_sequence(digest, wcs.get(key), f"identity.wcs.{key}")
    digest.add_double(exact_float_node(wcs.get("source_epoch"),
                                       "identity.wcs.source_epoch"))
    digest.add_double(exact_float_node(wcs.get("orientation_rad"),
                                       "identity.wcs.orientation_rad"))
    shape = identity.get("shape")
    if not isinstance(shape, Mapping):
        die("bundle identity shape is malformed")
    digest.add_integer(shape.get("rows"))
    digest.add_integer(shape.get("cols"))
    slots = identity.get("ordered_slots")
    if not isinstance(slots, list):
        die("bundle identity ordered slots are malformed")
    digest.add_integer(len(slots))
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            die("bundle identity slot is malformed")
        digest.add_integer(slot.get("ordered_slot"))
        digest.add_string(slot.get("grouping", ""))
        digest.add_string(slot.get("group_identity", ""))
        digest.add_integer(slot.get("array_identity"))
        digest.add_integer(slot.get("stokes_identity"))
        digest.add_double(exact_float_node(slot.get("frequency_hz"),
                                           f"identity.slot[{index}].frequency_hz"))
    return digest.finish()


def canonical_digest_string(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"canonical-hexfloat-sha256-v1:[0-9a-f]{64}", value) is not None


def hash_exact_threshold(digest: CanonicalDigest,
                         record: Mapping[str, Any], label: str) -> None:
    for key in ("order_statistic_algorithm", "support_algorithm",
                "coefficient_product", "coefficient_stage"):
        digest.add_string(record.get(key, ""))
    for key in ("requested_cut", "realized_cut", "realized_threshold",
                "selected_positive_value"):
        digest.add_double(exact_float_node(record.get(key), f"{label}.{key}"))
    digest.add_integer(record.get("positive_value_count"))
    index = record.get("selected_zero_based_index")
    if not isinstance(index, Mapping):
        die(f"{label}.selected_zero_based_index is malformed")
    available = bool(index.get("available"))
    digest.add_integer(index.get("value", 0) if available else 0)
    digest.add_integer(available)
    for key in ("finite_convention", "positivity_convention",
                "comparison_convention"):
        digest.add_string(record.get(key, ""))


def hash_coadd_admissions(digest: CanonicalDigest,
                          provenance: Mapping[str, Any]) -> None:
    state = provenance.get("observation_resolved")
    if not isinstance(state, Mapping):
        die("coadd provenance observation_resolved state is malformed")
    admissions = state.get("admissions")
    if not isinstance(admissions, list):
        die("coadd provenance admissions are malformed")
    digest.add_integer(len(admissions))
    for index, admission in enumerate(admissions):
        if not isinstance(admission, Mapping):
            die(f"coadd admission {index} is malformed")
        embedding = admission.get("embedding")
        observation_shape = admission.get("observation_shape")
        coadd_shape = admission.get("coadd_shape")
        policies = admission.get("policies")
        if not all(isinstance(item, Mapping) for item in
                   (embedding, observation_shape, coadd_shape, policies)):
            die(f"coadd admission {index} has malformed nested state")
        digest.add_string(admission.get("observation_id", ""))
        digest.add_integer(embedding.get("delta_row"))
        digest.add_integer(embedding.get("delta_col"))
        digest.add_integer(observation_shape.get("rows"))
        digest.add_integer(observation_shape.get("cols"))
        digest.add_integer(coadd_shape.get("rows"))
        digest.add_integer(coadd_shape.get("cols"))
        digest.add_integer(admission.get("ordered_map_count"))
        digest.add_string(admission.get("admitted_bundle_identity", ""))
        digest.add_string(admission.get("response_identity", ""))
        digest.add_string(embedding.get("registration_identity", ""))
        digest.add_string(embedding.get("centering_identity", ""))
        digest.add_string(admission.get("coefficient_stage", ""))
        digest.add_string(policies.get("normalization_support", ""))
        digest.add_string(policies.get("science_policy_support", ""))
        digest.add_string(policies.get("validity", ""))
        digest.add_string(policies.get("nonfinite", ""))
        digest.add_double(exact_float_node(
            admission.get("observation_exposure_seconds"),
            f"coadd.admission[{index}].observation_exposure_seconds"))
        contributing = admission.get("numerically_contributing_pixel_count")
        parents = admission.get("observation_raw_parent_digests")
        if not isinstance(contributing, list) or not isinstance(parents, list):
            die(f"coadd admission {index} has malformed vector facts")
        digest.add_integer(len(contributing))
        for value in contributing:
            digest.add_integer(value)
        digest.add_integer(len(parents))
        for value in parents:
            digest.add_string(value)


def recompute_raw_parent_digest(identity: Mapping[str, Any],
                                realized: Mapping[str, Any],
                                product: "FitsProduct",
                                noise: np.ndarray,
                                scope: str,
                                provenance: Mapping[str, Any]) -> str:
    """Reproduce science_map_raw_parent_digest from serialized product bytes."""
    digest = CanonicalDigest()
    digest.add_string(recompute_bundle_identity_digest(identity))
    if scope == "coadd":
        hash_coadd_admissions(digest, provenance)
    for name in COMMON_PLANES:
        digest.add_matrix(internal_spatial(product.array(name)))
    if noise.shape[-1] != REALIZATIONS:
        die("raw-parent digest requires exactly 64 realization planes")
    digest.add_integer(REALIZATIONS)
    for realization in range(REALIZATIONS):
        digest.add_matrix(noise[..., realization])
    shape = image_spatial_shape(product.array("signal_I"))
    for name in F010_COADD[:-2]:
        if scope == "observation" and name == "coadd_observation_count_I":
            matrix = np.zeros(shape, dtype=np.int64)
        else:
            if name not in product.hdus:
                die(f"raw-parent digest requires product plane {name}")
            matrix = internal_spatial(product.array(name))
        digest.add_matrix(matrix)
    digest.add_integer(bool(realized.get("initialized")))
    products = realized.get("products")
    if not isinstance(products, list) or len(products) != 8:
        die("raw-parent digest requires eight realized product records")
    by_name = {entry.get("identity"): entry for entry in products
               if isinstance(entry, Mapping)}
    ordered_products = F010_COADD[:-2]
    if set(by_name) != set(ordered_products):
        die("raw-parent digest product identities are incomplete or repeated")
    for name in ordered_products:
        entry = by_name[name]
        digest.add_integer(bool(entry.get("available")))
        digest.add_string(entry.get("absence_reason", ""))
        digest.add_integer(entry.get("nonzero_count"))
        digest.add_string(entry.get("value_sum", ""))
    thresholds = realized.get("thresholds")
    if not isinstance(thresholds, Mapping):
        die("raw-parent digest threshold state is malformed")
    for name in ("normalization", "science_policy"):
        value = thresholds.get(name)
        if not isinstance(value, Mapping):
            die(f"raw-parent digest threshold {name} is malformed")
        hash_exact_threshold(digest, value, f"raw_parent.thresholds.{name}")
    companions = realized.get("required_companions")
    if not isinstance(companions, list):
        die("raw-parent digest companions are malformed")
    digest.add_integer(len(companions))
    for companion in companions:
        digest.add_string(companion)
    digest.add_string(realized.get("admitted_bundle_identity", ""))
    return digest.finish()


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def repository_root() -> Path:
    for candidate in (package_dir(), *package_dir().parents):
        if (candidate / "validation" / "product_contracts.json").is_file():
            return candidate
    die("cannot locate candidate repository root from frozen program path")


def resolve_relative(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_campaign(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    campaign_path = (path or package_dir() / "campaign.json").resolve()
    campaign = read_json(campaign_path)
    if not isinstance(campaign, dict):
        die("campaign.json is not an object")
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        die("campaign.json schema_version is not pinned")
    if campaign.get("request_id") != REQUEST_ID:
        die("campaign request_id differs from SCI-MAP-001-UNITY-001")
    if campaign.get("candidate_sha") != CANDIDATE_SHA:
        die("campaign candidate SHA differs from the frozen repair candidate")
    raw_cases = campaign.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 7:
        die("campaign must contain exactly seven case records")
    by_id: dict[str, dict[str, Any]] = {}
    for record in raw_cases:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            die("campaign contains a malformed case record")
        case_id = record["id"]
        if case_id in by_id:
            die(f"campaign repeats case {case_id}")
        by_id[case_id] = record
    if list(by_id) != list(EXPECTED_CASES):
        die("campaign case order/identity differs from the pinned seven-case protocol")
    for case_id, expected in EXPECTED_CASES.items():
        record = by_id[case_id]
        for key, value in expected.items():
            if record.get(key) != value:
                die(f"campaign {case_id}.{key} differs from pinned value {value!r}")
        if record.get("expected_arrays") != list(ARRAYS):
            die(f"campaign {case_id} does not require all three arrays in order")
    fixed = campaign.get("fixed_execution", {})
    fixed_expected = {
        "ssh_alias": "unity_toltec",
        "noise_seed": 5489,
        "noise_realizations": REALIZATIONS,
        "atol": REGISTERED_ATOL,
        "rtol": REGISTERED_RTOL,
        "arrays": list(ARRAYS),
        "science_observations": [152390, 152392],
        "point_observations": [152389],
        "scan_farm_bound": PARALLEL_POLICY,
    }
    for key, value in fixed_expected.items():
        if fixed.get(key) != value:
            die(f"campaign fixed_execution.{key} differs from {value!r}")
    return campaign, campaign_path


def product_contract_path(campaign: Mapping[str, Any], campaign_path: Path,
                          override: Path | None = None) -> Path:
    path = override.resolve() if override else resolve_relative(
        str(campaign.get("product_contracts", "")), campaign_path.parent)
    if not path.is_file():
        die(f"candidate product-contract registry is absent: {path}")
    expected = campaign.get("pinned_source_sha256", {}).get(
        "validation/product_contracts.json")
    if not isinstance(expected, str) or sha256(path) != expected:
        die("candidate product-contract registry digest differs from campaign pin")
    return path


def load_contracts(campaign: Mapping[str, Any], campaign_path: Path,
                   override: Path | None = None) -> dict[str, Any]:
    contracts = read_json(product_contract_path(campaign, campaign_path, override))
    if not isinstance(contracts, dict) or contracts.get("schema_version") != \
            "citlali-product-contract-registry-v2":
        die("candidate product-contract registry schema is not v2")
    science = contracts.get("science_map_contracts", {}).get("sci-map-001-f010-v1")
    if not isinstance(science, dict):
        die("candidate registry lacks sci-map-001-f010-v1")
    return contracts


def validate_successor_registries(
        campaign: Mapping[str, Any], campaign_path: Path, source_root: Path,
        profile_override: Path | None, accepted_override: Path | None,
        point_contract: str | None, science_contract: str | None) -> None:
    authority = campaign.get("authority")
    if not isinstance(authority, Mapping):
        die("campaign authority is malformed")
    expected_point_contract = authority.get("point_product_contract_id")
    expected_science_contract = authority.get("science_product_contract_id")
    effective_point_contract = point_contract or str(expected_point_contract)
    effective_science_contract = science_contract or str(expected_science_contract)
    if effective_point_contract != expected_point_contract or \
            effective_science_contract != expected_science_contract:
        die("run product-contract IDs differ from the pinned SCI-MAP-001 successors")
    pins = campaign["pinned_source_sha256"]
    profile_path = (profile_override or
                    source_root / "validation/validation_profiles.json").resolve()
    accepted_path = (accepted_override or
                     source_root / "validation/accepted_runs.json").resolve()
    for path, relative, label in (
            (profile_path, "validation/validation_profiles.json", "profile registry"),
            (accepted_path, "validation/accepted_runs.json", "accepted-runs registry")):
        if not path.is_file() or sha256(path) != pins.get(relative):
            die(f"run {label} differs from the exact candidate pin")
    profiles = read_json(profile_path)
    profile_rows = profiles.get("profiles") if isinstance(profiles, Mapping) else None
    if not isinstance(profile_rows, list):
        die("validation profile registry is malformed")
    by_id = {record.get("profile_id"): record for record in profile_rows
             if isinstance(record, Mapping)}
    expected_profiles = {
        authority.get("point_profile_id"): ("point", expected_point_contract),
        authority.get("science_profile_id"): ("science", expected_science_contract),
    }
    for profile_id, (mode, contract_id) in expected_profiles.items():
        record = by_id.get(profile_id)
        if not isinstance(record, Mapping) or record.get("mode") != mode or \
                record.get("status") != "preparing" or \
                record.get("product_contract_id") != contract_id or \
                record.get("baseline_record_id") is not None:
            die(f"successor validation profile differs: {profile_id}")
    contracts = load_contracts(campaign, campaign_path)
    contract_rows = contracts.get("contracts")
    if not isinstance(contract_rows, list):
        die("product-contract registry contract list is malformed")
    contract_by_id = {record.get("contract_id"): record for record in contract_rows
                      if isinstance(record, Mapping)}
    for profile_id, (_, contract_id) in expected_profiles.items():
        record = contract_by_id.get(contract_id)
        if not isinstance(record, Mapping) or record.get("profile_id") != profile_id:
            die(f"successor product contract differs: {contract_id}")
    accepted = read_json(accepted_path)
    accepted_rows = accepted.get("records") if isinstance(accepted, Mapping) else None
    if not isinstance(accepted_rows, list):
        die("accepted-runs registry is malformed")
    accepted_by_id = {record.get("record_id"): record for record in accepted_rows
                      if isinstance(record, Mapping)}
    for key, mode in (("tolproj_point_record_id", "point"),
                      ("tolproj_science_record_id", "science")):
        record_id = authority.get(key)
        record = accepted_by_id.get(record_id)
        if not isinstance(record, Mapping) or record.get("mode") != mode or \
                not str(record.get("status", "")).startswith("accepted"):
            die(f"accepted base record differs: {record_id}")


def case_by_id(campaign: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    for record in campaign["cases"]:
        if record["id"] == case_id:
            return record
    die(f"unknown campaign case: {case_id}")


OWNER_REQUIRED = (
    "schema_version", "unity_host_alias", "unity_source_checkout",
    "request_root", "deployed_campaign_path", "unity_python",
    "tolproj_executable", "tolproj_site_config", "point_project",
    "point_source_filter", "point_apt_dir", "science_project",
    "science_source_basename", "science_pointing_reduction",
    "evidence_operator", "slurm_account", "slurm_qos", "slurm_constraint",
    "slurm_reservation", "kidscpp_source_dir", "tula_source_dir",
    "local_retrieval_destination",
)
OWNER_PATHS = (
    "unity_source_checkout", "request_root", "deployed_campaign_path",
    "unity_python", "tolproj_executable", "tolproj_site_config",
    "point_project", "point_apt_dir", "science_project",
    "kidscpp_source_dir", "tula_source_dir", "local_retrieval_destination",
)


def validate_owner_path_string(value: Any, key: str) -> Path:
    if not isinstance(value, str) or any(token in value for token in ("\r", "\n", "\\")):
        die(f"owner-values {key} path has invalid characters")
    if value != os.path.normpath(value):
        die(f"owner-values {key} path must be lexically canonical without a trailing slash")
    path = Path(value)
    if not path.is_absolute() or value in ("/", "/tmp"):
        die(f"owner-values {key} must be a specific absolute path")
    return path


def validate_owner_values(path: Path, require_existing: bool = False) -> dict[str, Any]:
    values = read_json(path)
    if not isinstance(values, dict):
        die("owner-values input is not an object")
    missing = [key for key in OWNER_REQUIRED if key not in values]
    extra = sorted(set(values) - set(OWNER_REQUIRED))
    if missing or extra:
        die(f"owner-values keys differ; missing={missing}, extra={extra}")
    if values["schema_version"] != OWNER_SCHEMA:
        die("owner-values schema_version is not pinned")
    if values["unity_host_alias"] != "unity_toltec":
        die("Unity host alias must be exactly unity_toltec")
    placeholders = ("TODO", "CHANGEME", "UNKNOWN", "<", ">")
    for key, value in values.items():
        if key in ("slurm_qos", "slurm_constraint", "slurm_reservation"):
            if not isinstance(value, str):
                die(f"owner-values {key} must be a string (empty is explicit)")
            continue
        if not isinstance(value, str) or not value.strip():
            die(f"owner-values {key} is unresolved")
        if any(token.lower() in value.lower() for token in placeholders):
            die(f"owner-values {key} contains a placeholder")
    for key in OWNER_PATHS:
        validate_owner_path_string(values[key], key)
    if "/" in values["science_source_basename"]:
        die("science_source_basename must be one basename, not a path")
    if re.fullmatch(r"redu[0-9]{2}", values["science_pointing_reduction"]) is None:
        die("science_pointing_reduction must match reduNN")
    if require_existing:
        for key in (
            "unity_source_checkout", "deployed_campaign_path", "unity_python",
            "tolproj_executable", "tolproj_site_config", "point_project",
            "point_apt_dir", "science_project", "kidscpp_source_dir",
            "tula_source_dir",
        ):
            if not Path(values[key]).exists():
                die(f"owner-values {key} does not exist on this host")
    return values


def candidate_executable(values: Mapping[str, Any] | None,
                         explicit: str | None) -> str:
    if explicit:
        result = Path(explicit)
    elif values:
        result = Path(str(values["request_root"])) / "bin" / "citlali"
    else:
        die("materialization requires --owner-values or --citlali-executable")
    if not result.is_absolute() or str(result) in ("/", "/tmp"):
        die("candidate executable must be a specific absolute path")
    return str(result)


def overlay_bytes(case: Mapping[str, Any], executable: str) -> bytes:
    def boolean(value: Any) -> str:
        return "true" if value else "false"

    text = f"""reduce:
  jobkey: {case['jobkey']}
  steps:
    0:
      path: {executable}
      config:
        low_level:
          coadd: {{enabled: {boolean(case['coadd'])}}}
          mapmaking:
            method: naive
            coverage_cut: {case['coverage_cut']}
          noise_maps:
            enabled: true
            n_noise_maps: 64
            randomize_dets: false
            write_realizations: true
            products:
              enabled: {boolean(case['products_enabled'])}
              apply_empirical_weights: false
          post_processing:
            map_filtering: {{enabled: false}}
            source_finding: {{enabled: false}}
          runtime:
            n_threads: {case['threads']}
            parallel_policy: {case['parallel_policy']}
            verbose: true
          timestream:
            fruit_loops: {{enabled: false}}
"""
    # Round-trip here catches accidental YAML scalar or indentation defects.
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        die("internal overlay renderer produced invalid YAML")
    return text.encode("utf-8")


def materialize_case(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    case = case_by_id(campaign, args.case_id)
    values = validate_owner_values(args.owner_values) if args.owner_values else None
    executable = candidate_executable(values, args.citlali_executable)
    write_new(args.output.resolve(), overlay_bytes(case, executable))
    print(json.dumps({"case_id": args.case_id, "output": str(args.output.resolve()),
                      "sha256": sha256(args.output.resolve())}, sort_keys=True))
    return 0


def materialize_all(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner_values(args.owner_values) if args.owner_values else None
    executable = candidate_executable(values, args.citlali_executable)
    output = args.output.resolve()
    if output.exists():
        die(f"refusing to reuse materialization directory: {output}")
    output.mkdir(parents=True)
    records = []
    for case in campaign["cases"]:
        path = output / f"{case['id']}-expert-materialized.yaml"
        write_new(path, overlay_bytes(case, executable))
        records.append({"case_id": case["id"], "path": path.name,
                        "sha256": sha256(path)})
    write_new(output / "manifest.json", json_bytes({
        "schema_version": "sci-map-unity-materialized-overlays-v1",
        "candidate_sha": CANDIDATE_SHA,
        "records": records,
    }))
    print(json.dumps({"output": str(output), "records": records}, sort_keys=True))
    return 0


def nested(mapping: Mapping[str, Any], path: Sequence[Any]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            die("merged config lacks " + ".".join(map(str, path)))
        value = value[key]
    return value


def run_git(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"git identity check failed for {source}: {exc}") from exc
    return result.stdout.strip()


def validate_installed_kit_marker(
        marker_value: Any, vendor_value: Any, bundle_value: Any,
        case: Mapping[str, Any], campaign: Mapping[str, Any],
        expected_numbered_order: Sequence[str]) -> dict[str, Any]:
    marker = dict(require_exact_keys(marker_value, (
        "schema_version", "kit_version", "bundle", "mode",
        "observation_filename", "installed_filenames", "policy_sha256",
        "record_id", "source_repository", "source_commit",
    ), "TolProj installed kit marker"))
    vendor = require_exact_keys(vendor_value, (
        "schema_version", "source_repository", "site_normalizations",
        "mode_kits", "files",
    ), "TolProj vendor manifest")
    if vendor["schema_version"] != "tolproj-citlali-refactor-vendor-v2" or \
            vendor["source_repository"] != "toltec-astro/citlali":
        die("TolProj vendor schema/repository differs from the pinned authority")
    mode_kits = vendor["mode_kits"]
    if not isinstance(mode_kits, Mapping):
        die("TolProj vendor mode_kits is malformed")
    mode = str(case["mode"])
    vendor_mode = require_exact_keys(mode_kits.get(mode), (
        "bundle", "kit_version", "source_commit", "observation_filename",
        "repository_policy_filenames", "operator_filenames",
    ), f"TolProj vendor {mode} kit")
    bundle = require_exact_keys(bundle_value, (
        "schema_version", "kit_version", "scope", "modes",
    ), "TolProj active bundle manifest")
    if bundle["schema_version"] != "citlali-tolteca-mode-kit-manifest-v2" or \
            bundle["kit_version"] != campaign["authority"]["tolproj_bundle"] or \
            bundle["scope"] != "all-supported-modes":
        die("TolProj active bundle top-level identity differs")
    bundle_modes = bundle["modes"]
    if not isinstance(bundle_modes, Mapping):
        die("TolProj active bundle modes are malformed")
    mode_manifest = bundle_modes.get(mode)
    if not isinstance(mode_manifest, Mapping):
        die(f"TolProj active bundle lacks {mode} mode")

    repository_files = vendor_mode["repository_policy_filenames"]
    operator_files = vendor_mode["operator_filenames"]
    observation_filename = vendor_mode["observation_filename"]
    if not isinstance(repository_files, list) or not repository_files or \
            not isinstance(operator_files, list) or not operator_files or \
            any(not isinstance(name, str) or not name for name in
                [*repository_files, *operator_files]) or \
            not isinstance(observation_filename, str) or not observation_filename:
        die(f"TolProj vendor {mode} filename roles are malformed")
    selected_filenames = sorted(
        [*repository_files, *operator_files, observation_filename])
    if len(selected_filenames) != len(set(selected_filenames)):
        die(f"TolProj vendor {mode} filename roles overlap")
    expected_installed_from_numbered = sorted(
        name for name in expected_numbered_order
        if name not in ("40_setup.yaml", "99_zz_tolproj_submission_runtime.yaml"))
    required_files = mode_manifest.get("required_files")
    if required_files != [name for name in expected_numbered_order
                          if name not in ("40_setup.yaml",
                                          "99_zz_tolproj_submission_runtime.yaml")] or \
            selected_filenames != expected_installed_from_numbered:
        die(f"TolProj {mode} installed filenames differ across numbered/vendor/bundle authorities")

    record_key = ("tolproj_point_record_id" if mode == "point"
                  else "tolproj_science_record_id")
    expected = {
        "schema_version": "tolproj-installed-citlali-refactor-kit-v2",
        "kit_version": campaign["authority"]["tolproj_bundle"],
        "bundle": "phase4_1_v2_1",
        "mode": mode,
        "observation_filename": observation_filename,
        "installed_filenames": selected_filenames,
        "policy_sha256": MODE_POLICY_SHA256[mode],
        "record_id": campaign["authority"][record_key],
        "source_repository": "toltec-astro/citlali",
        "source_commit": campaign["authority"]["tolproj_bundle_source_commit"],
    }
    if vendor_mode["bundle"] != expected["bundle"] or \
            vendor_mode["kit_version"] != expected["kit_version"] or \
            vendor_mode["source_commit"] != expected["source_commit"] or \
            mode_manifest.get("record_id") != expected["record_id"] or \
            mode_manifest.get("policy_sha256") != expected["policy_sha256"]:
        die(f"TolProj {mode} vendor/bundle/campaign identity differs")
    if marker != expected:
        differences = {
            key: {"actual": marker.get(key), "expected": expected.get(key)}
            for key in expected if marker.get(key) != expected.get(key)
        }
        die(f"TolProj installed kit marker differs: {differences}")
    return expected


def validate_installed_numbered_bytes(
        case_dir: Path, case: Mapping[str, Any], vendor: Mapping[str, Any],
        executable: str) -> dict[str, str]:
    """Bind installed policy bytes to the frozen TolProj kit and generator."""
    mode = str(case["mode"])
    mode_record = vendor.get("mode_kits", {}).get(mode)
    files = vendor.get("files")
    if not isinstance(mode_record, Mapping) or not isinstance(files, Mapping):
        die(f"TolProj {mode} vendor byte authority is malformed")
    expert = ("99_pointing_expert_overrides.yaml" if mode == "point"
              else "99_science_expert_overrides.yaml")
    expected_expert = overlay_bytes(case, executable)
    expert_path = case_dir / expert
    if not expert_path.is_file() or expert_path.read_bytes() != expected_expert:
        die(f"{case['id']}: installed expert overlay differs from frozen generator")

    expected_frozen = TOLPROJ_FROZEN_NUMBERED_SHA256[mode]
    repository = mode_record.get("repository_policy_filenames")
    operators = mode_record.get("operator_filenames")
    if not isinstance(repository, list) or not isinstance(operators, list):
        die(f"TolProj {mode} vendor numbered roles are malformed")
    unchanged = sorted([*repository, *(name for name in operators
                                       if name != expert)])
    if unchanged != sorted(expected_frozen):
        die(f"TolProj {mode} unchanged numbered-policy inventory differs")
    verified = {expert: hashlib.sha256(expected_expert).hexdigest()}
    for name in unchanged:
        key = f"phase4_1_v2_1/{mode}/{name}"
        expected = expected_frozen[name]
        if files.get(key) != expected:
            die(f"TolProj vendor digest differs from frozen authority: {key}")
        path = case_dir / name
        if not path.is_file() or sha256(path) != expected:
            die(f"{case['id']}: installed numbered policy differs: {name}")
        verified[name] = expected
    return dict(sorted(verified.items()))


def validate_preflight_marker_binding(
        preflight: Mapping[str, Any], case: Mapping[str, Any],
        campaign: Mapping[str, Any]) -> dict[str, Any]:
    authority = dict(require_exact_keys(
        preflight.get("installed_kit_marker_authority"), (
            "schema_version", "kit_version", "bundle", "mode",
            "observation_filename", "installed_filenames", "policy_sha256",
            "record_id", "source_repository", "source_commit",
        ), f"{case['id']} preflight installed marker authority"))
    mode = str(case["mode"])
    record_key = ("tolproj_point_record_id" if mode == "point"
                  else "tolproj_science_record_id")
    numbered_key = "point_order" if mode == "point" else "science_order"
    installed = sorted(
        name for name in campaign["numbered_config_contract"][numbered_key]
        if name not in ("40_setup.yaml", "99_zz_tolproj_submission_runtime.yaml"))
    expected = {
        "schema_version": "tolproj-installed-citlali-refactor-kit-v2",
        "kit_version": campaign["authority"]["tolproj_bundle"],
        "bundle": "phase4_1_v2_1", "mode": mode,
        "observation_filename": ("72_pointing_observation.yaml"
                                 if mode == "point" else
                                 "72_science_observation.yaml"),
        "installed_filenames": installed,
        "policy_sha256": MODE_POLICY_SHA256[mode],
        "record_id": campaign["authority"][record_key],
        "source_repository": "toltec-astro/citlali",
        "source_commit": campaign["authority"]["tolproj_bundle_source_commit"],
    }
    if authority != expected:
        die(f"{case['id']}: preflight installed marker authority differs")
    paths = preflight.get("paths")
    digests = preflight.get("sha256")
    if not isinstance(paths, Mapping) or not isinstance(digests, Mapping):
        die(f"{case['id']}: preflight path/digest bindings are malformed")
    marker_path_value = paths.get("marker")
    marker_digest = digests.get("marker")
    if not isinstance(marker_path_value, str) or not isinstance(marker_digest, str):
        die(f"{case['id']}: preflight marker path/digest binding is absent")
    marker_path = Path(marker_path_value)
    if not marker_path.is_absolute() or not marker_path.is_file() or \
            sha256(marker_path) != marker_digest:
        die(f"{case['id']}: installed marker changed after preflight")
    marker = require_exact_keys(read_yaml(marker_path), tuple(expected),
                                f"{case['id']} frozen installed marker")
    if dict(marker) != authority:
        die(f"{case['id']}: frozen installed marker differs from preflight authority")
    return authority


def validate_preflight_file_binding(
        preflight: Mapping[str, Any], case: Mapping[str, Any],
        campaign: Mapping[str, Any], reduction_root: Path,
        merged_config: Path, raw_manifest: Path,
        product_contracts: Path) -> None:
    paths = require_exact_keys(preflight.get("paths"), (
        "case_dir", "merged", "marker", "source_root", "raw_input_manifest",
        "candidate_executable", "launcher", "launcher_source",
    ), f"{case['id']} preflight paths")
    digests = require_exact_keys(preflight.get("sha256"), (
        "merged", "marker", "vendor_manifest", "bundle_manifest",
        "canonical_manifest", "product_contracts", "raw_input_manifest",
        "candidate_executable", "launcher", "launcher_source", "numbered",
    ), f"{case['id']} preflight digests")
    raw_launcher = reduction_root / ".tolproj/citlali-launcher"
    raw_launcher_source = reduction_root / ".tolproj/citlali-source"
    if raw_launcher.is_symlink() or raw_launcher_source.is_symlink() or \
            not raw_launcher.is_file() or not raw_launcher_source.is_file():
        die(f"{case['id']}: returned TolProj launcher/source is not regular")
    expected_path_pairs = {
        "case_dir": reduction_root.resolve(),
        "merged": merged_config.resolve(),
        "marker": (reduction_root / ".citlali_refactor_kit.yaml").resolve(),
        "raw_input_manifest": raw_manifest.resolve(),
        "launcher": raw_launcher.resolve(),
        "launcher_source": raw_launcher_source.resolve(),
    }
    for key, expected in expected_path_pairs.items():
        value = paths[key]
        if not isinstance(value, str) or not Path(value).is_absolute() or \
                Path(value).resolve() != expected:
            die(f"{case['id']}: preflight {key} path differs from returned evidence")
    source_root = paths["source_root"]
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        die(f"{case['id']}: preflight source root is not absolute")
    marker_path = expected_path_pairs["marker"]
    if not marker_path.is_file():
        die(f"{case['id']}: installed marker is absent from returned reduction root")
    expected_digest_pairs = {
        "merged": sha256(merged_config), "marker": sha256(marker_path),
        "product_contracts": sha256(product_contracts),
        "raw_input_manifest": sha256(raw_manifest),
        "launcher": sha256(expected_path_pairs["launcher"]),
        "launcher_source": sha256(expected_path_pairs["launcher_source"]),
    }
    for key, expected in expected_digest_pairs.items():
        if digests[key] != expected:
            die(f"{case['id']}: preflight {key} digest differs from returned evidence")
    numbered_key = "point_order" if case["mode"] == "point" else "science_order"
    expected_names = campaign["numbered_config_contract"][numbered_key]
    numbered = digests["numbered"]
    if not isinstance(numbered, Mapping) or list(numbered) != expected_names:
        die(f"{case['id']}: preflight numbered-file identity/order differs")
    for name in expected_names:
        path = reduction_root / name
        if not path.is_file() or numbered[name] != sha256(path):
            die(f"{case['id']}: numbered config changed after preflight: {name}")
    merged = read_yaml(merged_config)
    if not isinstance(merged, Mapping):
        die(f"{case['id']}: frozen merged config is malformed")
    merged_executable = nested(merged, ("reduce", "steps", 0, "path"))
    if merged_executable != str(expected_path_pairs["launcher"]):
        die(f"{case['id']}: frozen merged config does not select case launcher; "
            f"actual={merged_executable!r} "
            f"expected={str(expected_path_pairs['launcher'])!r}")
    candidate_value = paths["candidate_executable"]
    if not isinstance(candidate_value, str) or not Path(candidate_value).is_absolute():
        die(f"{case['id']}: frozen candidate executable path is unresolved")
    executable = Path(candidate_value)
    if executable.is_symlink() or not executable.is_file() or \
            digests["candidate_executable"] != sha256(executable):
        die(f"{case['id']}: frozen candidate executable changed")
    source_lines = expected_path_pairs["launcher_source"].read_text(
        encoding="utf-8").splitlines()
    if source_lines != [str(executable)]:
        die(f"{case['id']}: TolProj launcher source does not select master binary")
    mode = str(case["mode"])
    expert = ("99_pointing_expert_overrides.yaml" if mode == "point"
              else "99_science_expert_overrides.yaml")
    expected_authority = {
        **TOLPROJ_FROZEN_NUMBERED_SHA256[mode],
        expert: hashlib.sha256(overlay_bytes(case, str(executable))).hexdigest(),
    }
    if preflight.get("installed_numbered_authority") != \
            dict(sorted(expected_authority.items())):
        die(f"{case['id']}: installed numbered byte authority differs")
    for name, expected in expected_authority.items():
        if sha256(reduction_root / name) != expected:
            die(f"{case['id']}: installed policy/generator bytes changed: {name}")
    validate_preflight_marker_binding(preflight, case, campaign)


def preflight_case(args: argparse.Namespace) -> int:
    campaign, campaign_path = load_campaign(args.campaign)
    contracts_path = product_contract_path(campaign, campaign_path,
                                           args.product_contracts)
    case = case_by_id(campaign, args.case_id)
    if args.mode != case["mode"]:
        die(f"preflight mode {args.mode} differs from case mode {case['mode']}")
    case_dir = args.case_dir.resolve()
    if not case_dir.is_dir():
        die(f"case directory is absent: {case_dir}")
    raw_authority = validate_raw_input_manifest(
        args.raw_input_manifest.resolve(), case["mode"],
        case["expected_observations"], forbidden_roots=(case_dir,))
    merged = read_yaml(args.merged.resolve())
    if not isinstance(merged, dict):
        die("merged configuration is not a mapping")
    values = validate_owner_values(args.owner_values) if args.owner_values else None
    executable = candidate_executable(values, args.citlali_executable)
    executable_path = Path(executable)
    if executable_path.is_symlink() or not executable_path.is_file():
        die("candidate executable must be an existing regular file")
    launcher_input = case_dir / ".tolproj/citlali-launcher"
    launcher_source_input = case_dir / ".tolproj/citlali-source"
    if launcher_input.is_symlink() or launcher_source_input.is_symlink():
        die("TolProj launcher/source must not be symlinks")
    launcher = launcher_input.resolve()
    launcher_source = launcher_source_input.resolve()
    if not launcher.is_file() or \
            not os.access(launcher, os.X_OK) or \
            not launcher_source.is_file() or launcher_source.read_text(
                encoding="utf-8").splitlines() != [executable]:
        die("TolProj launcher/source does not bind the candidate master executable")
    low = nested(merged, ("reduce", "steps", 0, "config", "low_level"))
    if not isinstance(low, Mapping):
        die("merged low_level config is not a mapping")
    expected_leaves = {
        ("coadd", "enabled"): case["coadd"],
        ("mapmaking", "method"): "naive",
        ("mapmaking", "coverage_cut"): case["coverage_cut"],
        ("noise_maps", "enabled"): True,
        ("noise_maps", "n_noise_maps"): REALIZATIONS,
        ("noise_maps", "randomize_dets"): False,
        ("noise_maps", "write_realizations"): True,
        ("noise_maps", "products", "enabled"): case["products_enabled"],
        ("noise_maps", "products", "apply_empirical_weights"): False,
        ("post_processing", "map_filtering", "enabled"): False,
        ("post_processing", "source_finding", "enabled"): False,
        ("runtime", "n_threads"): case["threads"],
        ("runtime", "parallel_policy"): case["parallel_policy"],
        ("runtime", "verbose"): True,
        ("timestream", "fruit_loops", "enabled"): False,
    }
    for path, expected in expected_leaves.items():
        actual = nested(low, path)
        if isinstance(expected, float):
            if not exact_float_equal(float(actual), expected):
                die(f"merged config {'.'.join(path)} differs from pinned value")
        elif actual != expected:
            die(f"merged config {'.'.join(path)} differs: {actual!r} != {expected!r}")
    if nested(merged, ("reduce", "jobkey")) != case["jobkey"]:
        die("merged jobkey differs from pinned case")
    if nested(merged, ("reduce", "steps", 0, "path")) != str(launcher):
        die("merged executable path differs from the case-local TolProj launcher")

    marker = read_yaml(args.marker.resolve())

    numbered = sorted(path.name for path in case_dir.iterdir()
                      if path.is_file() and re.fullmatch(r"[0-9]{2}_.+\.ya?ml", path.name))
    expected_order = campaign["numbered_config_contract"][
        "point_order" if case["mode"] == "point" else "science_order"]
    if numbered != sorted(expected_order):
        die(f"numbered config inventory differs; actual={numbered}, expected={sorted(expected_order)}")
    other_token = "science" if case["mode"] == "point" else "pointing"
    if any(other_token in name for name in numbered):
        die("numbered config inventory mixes point and science families")

    source_root = args.source_root.resolve()
    if run_git(source_root, "rev-parse", "HEAD") != CANDIDATE_SHA:
        die("source root is not at the exact repair candidate")
    pinned = campaign.get("pinned_source_sha256", {})
    for relative, expected_digest in pinned.items():
        source_path = source_root / relative
        if not source_path.is_file() or sha256(source_path) != expected_digest:
            die(f"pinned candidate source differs: {relative}")

    if sha256(args.vendor_manifest.resolve()) != TOLPROJ_VENDOR_MANIFEST_SHA256:
        die("TolProj vendor manifest bytes differ from the frozen bundle authority")
    vendor = read_yaml(args.vendor_manifest.resolve())
    bundle = read_yaml(args.bundle_manifest.resolve())
    canonical = read_yaml(args.canonical_manifest.resolve())
    if not isinstance(vendor, Mapping) or not isinstance(bundle, Mapping) or \
            not isinstance(canonical, Mapping):
        die("TolProj vendor/bundle/canonical manifest is malformed")
    if bundle != canonical:
        die("TolProj bundle manifest differs structurally from candidate canonical manifest")
    if bundle.get("kit_version") != campaign["authority"]["tolproj_bundle"]:
        die("bundle manifest kit version differs")
    vendor_mode = vendor.get("mode_kits", {}).get(case["mode"])
    if not isinstance(vendor_mode, dict):
        die("vendor manifest lacks selected mode")
    if vendor_mode.get("source_commit") != campaign["authority"]["tolproj_bundle_source_commit"]:
        die("vendor manifest source commit differs from campaign")
    mode_manifest = bundle.get("modes", {}).get(case["mode"])
    if not isinstance(mode_manifest, dict):
        die("bundle manifest lacks selected mode")
    expected_post_setup = [name for name in expected_order if name != "40_setup.yaml"]
    if sorted(mode_manifest.get("required_files", [])) != sorted(expected_post_setup[:-1]):
        die("bundle required_files differs from ordered TolProj inputs")
    marker_authority = validate_installed_kit_marker(
        marker, vendor, bundle, case, campaign, expected_order)
    installed_numbered_authority = validate_installed_numbered_bytes(
        case_dir, case, vendor, executable)

    record = {
        "schema_version": "sci-map-unity-case-preflight-v1",
        "request_id": REQUEST_ID,
        "candidate_sha": CANDIDATE_SHA,
        "case": case,
        "paths": {
            "case_dir": str(case_dir), "merged": str(args.merged.resolve()),
            "marker": str(args.marker.resolve()), "source_root": str(source_root),
            "raw_input_manifest": str(raw_authority.path),
            "candidate_executable": executable, "launcher": str(launcher),
            "launcher_source": str(launcher_source),
        },
        "sha256": {
            "merged": sha256(args.merged.resolve()),
            "marker": sha256(args.marker.resolve()),
            "vendor_manifest": sha256(args.vendor_manifest.resolve()),
            "bundle_manifest": sha256(args.bundle_manifest.resolve()),
            "canonical_manifest": sha256(args.canonical_manifest.resolve()),
            "product_contracts": sha256(contracts_path),
            "raw_input_manifest": raw_authority.digest,
            "candidate_executable": sha256(executable_path),
            "launcher": sha256(launcher),
            "launcher_source": sha256(launcher_source),
            "numbered": {name: sha256(case_dir / name) for name in expected_order},
        },
        "raw_input_authority": raw_authority_record(raw_authority),
        "installed_kit_marker_authority": marker_authority,
        "installed_numbered_authority": installed_numbered_authority,
        "result": "pass",
        "interpretation": "preflight_only_not_external_evidence",
    }
    write_new(args.output.resolve(), json_bytes(record))
    print(json.dumps({"case_id": args.case_id, "result": "pass",
                      "output": str(args.output.resolve())}, sort_keys=True))
    return 0


def collection_case_records(collection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = collection.get("cases")
    records: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        iterator = []
        for case_id, value in raw.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("case_id", case_id)
                iterator.append(record)
            else:
                die(f"collection case {case_id} is not an object")
    elif isinstance(raw, list):
        iterator = raw
    else:
        die("result collection cases must be an object or array")
    for value in iterator:
        if not isinstance(value, Mapping):
            die("result collection contains a malformed case")
        case_id = value.get("case_id", value.get("id"))
        if not isinstance(case_id, str) or case_id in records:
            die("result collection case identities are missing or repeated")
        records[case_id] = dict(value)
    if set(records) != set(EXPECTED_CASES):
        die(f"result collection must contain exactly seven cases; got {sorted(records)}")
    return records


def path_from_record(value: Any, base: Path, label: str,
                     *, required: bool = True) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        die(f"{label} path is unresolved")
    path = resolve_relative(value, base)
    if not path.is_file():
        die(f"{label} file is absent: {path}")
    return path


def unique_glob(root: Path, pattern: str, label: str,
                *, required: bool = True) -> Path | None:
    matches = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not matches and not required:
        return None
    if len(matches) != 1:
        die(f"{label}: expected exactly one match for {pattern}, got {len(matches)}")
    return matches[0]


def discover_map(root: Path, mode: str, scope: str, obsnum: int | None,
                 array: str) -> Path:
    if scope == "coadd":
        pattern = f"coadded/raw/*_{array}_citlali.fits"
    elif mode == "point":
        pattern = f"{obsnum}/raw/*_{array}_pointing_{obsnum}_citlali.fits"
    else:
        pattern = f"{obsnum}/raw/*_{array}_science_{obsnum}_citlali.fits"
    candidates = [path for path in root.glob(pattern)
                  if "_noise_citlali.fits" not in path.name]
    if len(candidates) != 1:
        die(f"{root}: expected one {scope} map for obs={obsnum} array={array}; "
            f"found {len(candidates)}")
    return candidates[0].resolve()


def discover_noise(root: Path, mode: str, scope: str, obsnum: int | None,
                   array: str, required: bool) -> Path | None:
    if scope == "coadd":
        pattern = f"coadded/raw/*_{array}_noise_citlali.fits"
    elif mode == "point":
        pattern = f"{obsnum}/raw/*_{array}_pointing_{obsnum}_noise_citlali.fits"
    else:
        pattern = f"{obsnum}/raw/*_{array}_science_{obsnum}_noise_citlali.fits"
    candidates = list(root.glob(pattern))
    if not candidates:
        # Tolerate the established placement of the noise token before mode,
        # but never tolerate ambiguity.
        parent = "coadded/raw" if scope == "coadd" else f"{obsnum}/raw"
        candidates = list(root.glob(f"{parent}/*_{array}_*noise*_citlali.fits"))
    if not candidates and not required:
        return None
    if len(candidates) != 1:
        die(f"{root}: expected {'one' if required else 'zero or one'} {scope} noise "
            f"file for obs={obsnum} array={array}; found {len(candidates)}")
    return candidates[0].resolve()


def ledger_lookup(case_record: Mapping[str, Any], root: Path, obsnum: int,
                  array: str, collection_base: Path) -> Path | None:
    ledgers = case_record.get("sample_ledgers", {})
    key = f"{obsnum}:{array}"
    value: Any = None
    if isinstance(ledgers, Mapping):
        value = ledgers.get(key)
        if value is None:
            obs_record = ledgers.get(str(obsnum), ledgers.get(obsnum))
            if isinstance(obs_record, Mapping):
                value = obs_record.get(array)
    elif isinstance(ledgers, list):
        matches = [item for item in ledgers if isinstance(item, Mapping)
                   and int(item.get("obsnum", -1)) == obsnum
                   and item.get("array") == array]
        if len(matches) > 1:
            die(f"result collection repeats sample ledger {key}")
        value = matches[0].get("path") if matches else None
    if isinstance(value, Mapping):
        value = value.get("path")
    if isinstance(value, str) and value:
        path = resolve_relative(value, collection_base)
        return path if path.is_file() else None
    conventional = root / "sample-ledgers" / f"obs{obsnum}_{array}.npz"
    return conventional.resolve() if conventional.is_file() else None


def find_provenance(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct.resolve()
    matches = [path.resolve() for path in root.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        die(f"{root}: expected exactly one {filename}, found {len(matches)}")
    return matches[0]


def normalized_exit_status(record: Mapping[str, Any], base: Path,
                           case_id: str) -> tuple[int, Path]:
    value = record.get("exit_status")
    status_path_value = record.get("exit_status_path")
    if status_path_value is not None:
        path = path_from_record(status_path_value, base,
                                f"{case_id} exit status")
        try:
            parsed = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"{case_id}: invalid exit status file") from exc
        if value is not None and int(value) != parsed:
            die(f"{case_id}: inline and file exit statuses disagree")
        return parsed, path
    if value is None:
        die(f"{case_id}: result collection omits exit status")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{case_id}: exit status is not an integer") from exc
    # Freeze inline status into the analysis input without manufacturing a file.
    return parsed, Path()


def validate_runtime_config_authority(
        reduction_root: Path, pre_run_merged: Path,
        case_id: str) -> dict[str, Any]:
    runtime_path = (reduction_root / "citlali_merged_config.yaml").resolve()
    manifest_path = (reduction_root / "config_source_manifest.yaml").resolve()
    runtime_candidates = sorted(
        path.resolve() for path in reduction_root.rglob("citlali_merged_config.yaml")
        if path.is_file())
    manifest_candidates = sorted(
        path.resolve() for path in reduction_root.rglob("config_source_manifest.yaml")
        if path.is_file())
    if runtime_candidates != [runtime_path] or manifest_candidates != [manifest_path]:
        die(f"{case_id}: runtime merged-config/manifest inventory is not exact and unique")
    runtime = read_yaml(runtime_path)
    pre_run = read_yaml(pre_run_merged)
    if not isinstance(runtime, Mapping) or not isinstance(pre_run, Mapping):
        die(f"{case_id}: runtime or pre-run merged config is malformed")
    low = nested(pre_run, ("reduce", "steps", 0, "config", "low_level"))
    if not isinstance(low, Mapping):
        die(f"{case_id}: pre-run low-level config is malformed")
    missing = sorted(set(low) - set(runtime))
    changed = sorted(key for key in set(low) & set(runtime)
                     if runtime[key] != low[key])
    runtime_only = sorted(set(runtime) - set(low))
    inputs = runtime.get("inputs")
    if missing or changed or runtime_only != ["inputs"] or \
            not isinstance(inputs, list) or not inputs:
        die(f"{case_id}: runtime config differs from pre-run effective low-level authority; "
            f"missing={missing}, changed={changed}, runtime_only={runtime_only}, "
            f"inputs_nonempty={isinstance(inputs, list) and bool(inputs)}")

    manifest = require_exact_keys(read_yaml(manifest_path), (
        "schema_version", "merge_authority", "merge_semantics", "upstream",
        "sources", "merged",
    ), f"{case_id} config-source manifest")
    if manifest["schema_version"] != "citlali-config-source-manifest-v1" or \
            manifest["merge_authority"] != "citlali_cli" or \
            manifest["merge_semantics"] != "ordered_later_sources_override" or \
            manifest["upstream"] != {
                "authority": "tolteca", "ordered_sources_provided": False}:
        die(f"{case_id}: config-source manifest identity differs")
    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        die(f"{case_id}: config-source manifest has no CLI sources")
    copied_paths: list[str] = []
    for index, value in enumerate(sources):
        source = require_exact_keys(value, (
            "precedence", "role", "source_path", "copied_filename",
            "size_bytes", "sha256",
        ), f"{case_id} config-source[{index}]")
        copied_name = source["copied_filename"]
        if source["precedence"] != index or source["role"] != "citlali_cli_config" or \
                not isinstance(source["source_path"], str) or \
                not source["source_path"] or not isinstance(copied_name, str) or \
                Path(copied_name).name != copied_name or copied_name in copied_paths:
            die(f"{case_id}: config-source record {index} identity differs")
        copied_path = reduction_root / copied_name
        if not copied_path.is_file() or copied_path.stat().st_size != \
                source["size_bytes"] or sha256(copied_path) != source["sha256"]:
            die(f"{case_id}: config-source copy differs: {copied_name}")
        copied_paths.append(copied_name)
    merged = require_exact_keys(manifest["merged"], (
        "snapshot_filename", "serialization", "size_bytes", "sha256",
    ), f"{case_id} config-source merged snapshot")
    if merged["snapshot_filename"] != runtime_path.name or \
            merged["serialization"] != "yaml_cpp_dump" or \
            merged["size_bytes"] != runtime_path.stat().st_size or \
            merged["sha256"] != sha256(runtime_path):
        die(f"{case_id}: runtime merged config is not manifest-authorized")
    return {
        "runtime_path": runtime_path, "manifest_path": manifest_path,
        "runtime_only_top_level_keys": runtime_only,
        "pre_run_bound_top_level_keys": sorted(low),
        "source_copy_filenames": copied_paths,
    }


def build_analysis_inputs(args: argparse.Namespace) -> int:
    campaign, campaign_path = load_campaign(args.campaign)
    contracts = product_contract_path(campaign, campaign_path, args.product_contracts)
    collection_path = (args.collection or args.request_root / "result-collection.json").resolve()
    collection = read_json(collection_path)
    if not isinstance(collection, dict):
        die("result collection is not an object")
    schema = collection.get("schema_version")
    if schema not in COLLECTION_SCHEMAS:
        die(f"unsupported result collection schema: {schema!r}")
    if collection.get("candidate_sha") != CANDIDATE_SHA:
        die("result collection candidate SHA differs from campaign")
    if collection.get("request_id") != REQUEST_ID:
        die("result collection request identity differs from campaign")
    declared_request_root = collection.get("request_root")
    if declared_request_root is not None:
        if not isinstance(declared_request_root, str) or \
                Path(declared_request_root).resolve() != args.request_root.resolve():
            die("result collection request_root differs from the requested root")
    records = collection_case_records(collection)
    output_cases: list[dict[str, Any]] = []
    evidence_gaps: list[dict[str, str]] = []
    reduction_roots: dict[str, Path] = {}
    for case_id, supplied in records.items():
        root_value = supplied.get("reduction_root", supplied.get("case_dir"))
        if not isinstance(root_value, str):
            die(f"{case_id}: reduction_root is unresolved")
        root = resolve_relative(root_value, collection_path.parent)
        if not root.is_dir():
            die(f"{case_id}: reduction root is absent: {root}")
        reduction_roots[case_id] = root
    if len({root.resolve() for root in reduction_roots.values()}) != len(reduction_roots):
        die("result collection reuses a reduction root across cases")

    for case in campaign["cases"]:
        case_id = case["id"]
        supplied = records[case_id]
        root = reduction_roots[case_id]
        status, status_path = normalized_exit_status(supplied, collection_path.parent,
                                                     case_id)
        if status != 0:
            die(f"{case_id}: repaired campaign requires exit status 0, got {status}")
        result_artifacts = {
            name: path_from_record(supplied.get(name), collection_path.parent,
                                   f"{case_id} {name.replace('_', ' ')}")
            for name in COLLECTION_CASE_FILE_FIELDS
        }
        if status_path and result_artifacts["exit_record"] != status_path:
            die(f"{case_id}: exit_status_path and exit_record differ")
        logs_raw = supplied.get("logs")
        if not isinstance(logs_raw, list) or not logs_raw:
            die(f"{case_id}: complete log inventory is required")
        logs = [path_from_record(item, collection_path.parent,
                                 f"{case_id} log") for item in logs_raw]
        required_log_paths = {
            result_artifacts[name] for name in
            ("submit_record", "stdout", "stderr", "slurm_accounting")
        }
        if not required_log_paths.issubset(set(logs)):
            die(f"{case_id}: complete logs omit a required job artifact")
        merged = path_from_record(supplied.get("merged_config"),
                                  collection_path.parent,
                                  f"{case_id} merged config")
        raw_manifest = path_from_record(supplied.get("raw_input_manifest"),
                                        collection_path.parent,
                                        f"{case_id} raw-input manifest")
        raw_manifest_digest = sha256(raw_manifest)
        declared_raw_digest = supplied.get("raw_input_manifest_sha256")
        if declared_raw_digest is not None and declared_raw_digest != raw_manifest_digest:
            die(f"{case_id}: declared raw-input manifest digest differs from file")
        raw_authority = validate_raw_input_manifest(
            raw_manifest, case["mode"], case["expected_observations"],
            forbidden_roots=tuple(reduction_roots.values()))
        preflight_record = read_json(result_artifacts["preflight_manifest"])
        if not isinstance(preflight_record, Mapping) or \
                preflight_record.get("schema_version") != \
                "sci-map-unity-case-preflight-v1" or \
                preflight_record.get("request_id") != REQUEST_ID or \
                preflight_record.get("candidate_sha") != CANDIDATE_SHA or \
                preflight_record.get("case", {}).get("id") != case_id or \
                preflight_record.get("sha256", {}).get("raw_input_manifest") != \
                raw_authority.digest or \
                preflight_record.get("raw_input_authority") != \
                raw_authority_record(raw_authority):
            die(f"{case_id}: pre-output preflight does not bind the exact raw authority")
        validate_preflight_file_binding(
            preflight_record, case, campaign, root, merged,
            raw_manifest, contracts)
        mapmaking = path_from_record(
            supplied.get("mapmaking_provenance") or str(find_provenance(root, "mapmaking_provenance.yaml")),
            collection_path.parent, f"{case_id} mapmaking provenance")
        # The coadd lifecycle sidecar is required even when coadd is disabled;
        # it is the explicit requested/effective/realized absence authority.
        coadd = path_from_record(
            supplied.get("coadd_provenance") or str(find_provenance(root, "coadd_provenance.yaml")),
            collection_path.parent, f"{case_id} coadd provenance")
        noise_provenance = path_from_record(
            supplied.get("noise_products_provenance") or str(find_provenance(
                root, "noise_products_provenance.yaml")),
            collection_path.parent, f"{case_id} noise-products provenance")
        runtime_authority = validate_runtime_config_authority(
            root, merged, case_id)

        map_records: list[dict[str, Any]] = []
        for obsnum in case["expected_observations"]:
            for array in ARRAYS:
                map_path = discover_map(root, case["mode"], "observation", obsnum, array)
                noise_required = bool(case["products_enabled"])
                noise_path = discover_noise(root, case["mode"], "observation",
                                            obsnum, array, noise_required)
                ledger_path = ledger_lookup(supplied, root, obsnum, array,
                                            collection_path.parent)
                if ledger_path is None:
                    evidence_gaps.append({
                        "id": "SCI-MAP-001-UNITY-001-EG-INDEPENDENT-SAMPLE-LEDGER",
                        "case_id": case_id,
                        "map": f"{obsnum}:{array}",
                        "detail": "independent observation F010 reconstruction ledger absent",
                    })
                map_records.append({
                    "scope": "observation", "obsnum": obsnum, "array": array,
                    "map": str(map_path), "map_sha256": sha256(map_path),
                    "noise": str(noise_path) if noise_path else None,
                    "noise_sha256": sha256(noise_path) if noise_path else None,
                    "sample_ledger": str(ledger_path) if ledger_path else None,
                    "sample_ledger_sha256": sha256(ledger_path) if ledger_path else None,
                    "raw_input_manifest_sha256": raw_manifest_digest,
                })
        if case["coadd"]:
            for array in ARRAYS:
                map_path = discover_map(root, case["mode"], "coadd", None, array)
                noise_path = discover_noise(root, case["mode"], "coadd", None,
                                            array, True)
                map_records.append({
                    "scope": "coadd", "obsnum": None, "array": array,
                    "map": str(map_path), "map_sha256": sha256(map_path),
                    "noise": str(noise_path), "noise_sha256": sha256(noise_path),
                    "sample_ledger": None, "sample_ledger_sha256": None,
                })

        expected = case.get("expected_counts", {})
        actual = {
            "observation_maps": sum(m["scope"] == "observation" for m in map_records),
            "observation_noise_files": sum(m["scope"] == "observation" and m["noise"]
                                           is not None for m in map_records),
            "coadd_maps": sum(m["scope"] == "coadd" for m in map_records),
            "coadd_noise_files": sum(m["scope"] == "coadd" and m["noise"]
                                     is not None for m in map_records),
        }
        if actual != expected:
            die(f"{case_id}: discovered map/noise counts differ: {actual} != {expected}")
        all_files = sorted(path.resolve() for path in root.rglob("*") if path.is_file())
        if not all_files:
            die(f"{case_id}: reduction root is empty")
        output_cases.append({
            "case_id": case_id,
            "reduction_root": str(root),
            "exit_status": status,
            "exit_status_path": str(status_path) if status_path else None,
            "merged_config": str(merged), "merged_config_sha256": sha256(merged),
            "raw_input_manifest": str(raw_manifest),
            "raw_input_manifest_sha256": raw_manifest_digest,
            "raw_input_authority": raw_authority_record(raw_authority),
            "mapmaking_provenance": str(mapmaking),
            "mapmaking_provenance_sha256": sha256(mapmaking),
            "coadd_provenance": str(coadd) if coadd else None,
            "coadd_provenance_sha256": sha256(coadd) if coadd else None,
            "noise_products_provenance": str(noise_provenance),
            "noise_products_provenance_sha256": sha256(noise_provenance),
            "runtime_merged_config": {
                "path": str(runtime_authority["runtime_path"]),
                "sha256": sha256(runtime_authority["runtime_path"]),
                "size": runtime_authority["runtime_path"].stat().st_size,
                "mtime_ns": runtime_authority["runtime_path"].stat().st_mtime_ns,
            },
            "config_source_manifest": {
                "path": str(runtime_authority["manifest_path"]),
                "sha256": sha256(runtime_authority["manifest_path"]),
                "size": runtime_authority["manifest_path"].stat().st_size,
                "mtime_ns": runtime_authority["manifest_path"].stat().st_mtime_ns,
            },
            "runtime_config_binding": {
                key: value for key, value in runtime_authority.items()
                if key not in ("runtime_path", "manifest_path")
            },
            "result_artifacts": {
                name: {"path": str(path), "sha256": sha256(path),
                       "size": path.stat().st_size,
                       "mtime_ns": path.stat().st_mtime_ns}
                for name, path in result_artifacts.items()
            },
            "logs": [{"path": str(path), "sha256": sha256(path),
                      "size": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns} for path in logs],
            "maps": map_records,
            "physical_inventory": [
                {"path": str(path), "relative_path": str(path.relative_to(root)),
                 "size": path.stat().st_size,
                 "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)}
                for path in all_files
            ],
        })

    point_digests = {record["raw_input_manifest_sha256"] for record in output_cases
                     if EXPECTED_CASES[record["case_id"]]["mode"] == "point"}
    science_digests = {record["raw_input_manifest_sha256"] for record in output_cases
                       if EXPECTED_CASES[record["case_id"]]["mode"] == "science"}
    if len(point_digests) != 1 or len(science_digests) != 1:
        die("point or science cases do not share byte-identical raw-input authority")

    inputs = {
        "schema_version": INPUT_SCHEMA,
        "program_schema": PROGRAM_SCHEMA,
        "request_id": REQUEST_ID,
        "candidate_sha": CANDIDATE_SHA,
        "campaign": str(campaign_path), "campaign_sha256": sha256(campaign_path),
        "product_contracts": str(contracts),
        "product_contracts_sha256": sha256(contracts),
        "collection": str(collection_path),
        "collection_sha256": sha256(collection_path),
        "request_root": str(args.request_root.resolve()),
        "cases": output_cases,
        "evidence_gaps": evidence_gaps,
    }
    write_new(args.output.resolve(), json_bytes(inputs))
    print(json.dumps({"output": str(args.output.resolve()),
                      "sha256": sha256(args.output.resolve()),
                      "evidence_gap_count": len(evidence_gaps)}, sort_keys=True))
    return 0


def verify_frozen_path(path_value: Any, digest_value: Any, label: str,
                       *, optional: bool = False) -> Path | None:
    if path_value is None and optional:
        if digest_value is not None:
            die(f"{label}: digest exists for absent path")
        return None
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        die(f"{label}: path/digest pair is incomplete")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file():
        die(f"{label}: frozen file is absent: {path}")
    actual = sha256(path)
    if actual != digest_value:
        die(f"{label}: SHA-256 changed ({actual} != {digest_value})")
    return path


def load_analysis_inputs(path: Path) -> dict[str, Any]:
    inputs = read_json(path)
    if not isinstance(inputs, dict) or inputs.get("schema_version") != INPUT_SCHEMA:
        die("analysis inputs schema is not pinned")
    if inputs.get("request_id") != REQUEST_ID or inputs.get("candidate_sha") != CANDIDATE_SHA:
        die("analysis inputs identity differs from campaign")
    verify_frozen_path(inputs.get("campaign"), inputs.get("campaign_sha256"),
                       "campaign")
    verify_frozen_path(inputs.get("product_contracts"),
                       inputs.get("product_contracts_sha256"),
                       "product contracts")
    verify_frozen_path(inputs.get("collection"), inputs.get("collection_sha256"),
                       "result collection")
    cases = inputs.get("cases")
    if not isinstance(cases, list) or [case.get("case_id") for case in cases] != \
            list(EXPECTED_CASES):
        die("analysis inputs do not contain the ordered seven cases")
    return inputs


def extname(hdu: fits.hdu.base.ExtensionHDU) -> str:
    value = hdu.header.get("EXTNAME")
    return str(value) if value is not None else "PRIMARY"


@dataclass
class FitsProduct:
    path: Path
    hdul: fits.HDUList
    hdus: dict[str, Any]

    @classmethod
    def open(cls, path: Path) -> "FitsProduct":
        try:
            hdul = fits.open(path, mode="readonly", memmap=True,
                             checksum=True, ignore_missing_end=False)
        except Exception as exc:
            raise EvidenceError(f"cannot open FITS {path}: {exc}") from exc
        if not hdul or not isinstance(hdul[0], fits.PrimaryHDU) or \
                hdul[0].data is not None:
            hdul.close()
            die(f"{path}: PRIMARY must be the metadata-only primary HDU")
        names: dict[str, Any] = {}
        for hdu in hdul:
            name = extname(hdu)
            if name != "PRIMARY":
                if name in names:
                    hdul.close()
                    die(f"{path}: repeated EXTNAME {name}")
                names[name] = hdu
        return cls(path, hdul, names)

    def close(self) -> None:
        self.hdul.close()

    def array(self, name: str) -> np.ndarray:
        if name not in self.hdus or self.hdus[name].data is None:
            die(f"{self.path}: missing image data for {name}")
        return np.asarray(self.hdus[name].data)


def dtype_name(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype)
    if dtype.kind == "i" and dtype.itemsize == 8:
        return "int64"
    if dtype.kind == "u" and dtype.itemsize == 1:
        return "uint8"
    if dtype.kind == "f" and dtype.itemsize == 8:
        return "float64"
    return dtype.name


def wcs_cards(header: fits.Header) -> dict[str, Any]:
    patterns = (
        r"WCSAXES", r"NAXIS[0-9]+", r"CTYPE[0-9]+", r"CUNIT[0-9]+",
        r"CRPIX[0-9]+", r"CRVAL[0-9]+", r"CDELT[0-9]+",
        r"CROTA[0-9]+", r"CD[0-9]+_[0-9]+", r"PC[0-9]+_[0-9]+",
        r"PV[0-9]+_[0-9]+", r"PS[0-9]+_[0-9]+", r"LONPOLE",
        r"LATPOLE", r"RADESYS", r"EQUINOX", r"RESTFRQ", r"SPECSYS",
    )
    matcher = re.compile("^(?:" + "|".join(patterns) + ")$")
    return {card.keyword: card.value for card in header.cards
            if matcher.fullmatch(card.keyword)}


def image_spatial_shape(array: np.ndarray) -> tuple[int, int]:
    if array.ndim < 2:
        die(f"map plane has fewer than two axes: shape={array.shape}")
    return int(array.shape[-2]), int(array.shape[-1])


def squeeze_spatial(array: np.ndarray) -> np.ndarray:
    result = np.squeeze(np.asarray(array))
    if result.ndim != 2:
        die(f"expected singleton frequency/Stokes axes around a 2-D map, got {array.shape}")
    return result


def internal_spatial(array: np.ndarray) -> np.ndarray:
    """Undo fitsIO::add_typed_hdu's explicit x-axis reversal."""
    return squeeze_spatial(array)[:, ::-1]


def threshold_selection(weight: np.ndarray, cut: float) -> dict[str, Any]:
    flat = np.asarray(weight, dtype=np.float64).ravel()
    values = np.sort(flat[np.isfinite(flat) & (flat > 0.0)])
    count = int(values.size)
    if count == 0:
        return {"threshold": 0.0, "selected": 0.0, "count": 0,
                "index": None}
    lower = int(math.floor(0.75 * count))
    index = (lower + count) // 2
    selected = float(values[index])
    return {"threshold": float(cut) * selected, "selected": selected,
            "count": count, "index": index}


def residual_key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")


def record_lossless_residual(
        arrays: dict[str, np.ndarray], label: str,
        actual: np.ndarray, expected: np.ndarray, *, bitwise: bool = False) -> None:
    """Store every element residual plus non-finite/bitwise topology."""
    key = residual_key(label)
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    arrays[f"{key}__actual_shape"] = np.asarray(actual_array.shape, dtype=np.int64)
    arrays[f"{key}__expected_shape"] = np.asarray(expected_array.shape, dtype=np.int64)
    arrays[f"{key}__actual_dtype"] = np.array(str(actual_array.dtype))
    arrays[f"{key}__expected_dtype"] = np.array(str(expected_array.dtype))
    if actual_array.shape != expected_array.shape:
        arrays[f"{key}__actual_values"] = np.ascontiguousarray(actual_array)
        arrays[f"{key}__expected_values"] = np.ascontiguousarray(expected_array)
        return
    if actual_array.dtype.kind in "iub" and expected_array.dtype.kind in "iub":
        actual_min = int(actual_array.min()) if actual_array.size else 0
        actual_max = int(actual_array.max()) if actual_array.size else 0
        expected_min = int(expected_array.min()) if expected_array.size else 0
        expected_max = int(expected_array.max()) if expected_array.size else 0
        limit = np.iinfo(np.int64)
        if actual_max - expected_min > limit.max or \
                actual_min - expected_max < limit.min:
            die(f"integer residual cannot be represented exactly for {label}")
        arrays[f"{key}__integer_delta"] = (
            actual_array.astype(np.int64) - expected_array.astype(np.int64))
    elif actual_array.dtype.kind == "f" and expected_array.dtype.kind == "f":
        actual64 = actual_array.astype(np.float64)
        expected64 = expected_array.astype(np.float64)
        finite = np.isfinite(actual64) & np.isfinite(expected64)
        residual = np.full(actual64.shape, np.nan, dtype=np.float64)
        residual[finite] = actual64[finite] - expected64[finite]
        arrays[f"{key}__finite_delta"] = residual
        for side, value in (("actual", actual64), ("expected", expected64)):
            arrays[f"{key}__{side}_finite"] = np.isfinite(value).astype(np.uint8)
            arrays[f"{key}__{side}_nan"] = np.isnan(value).astype(np.uint8)
            arrays[f"{key}__{side}_posinf"] = np.isposinf(value).astype(np.uint8)
            arrays[f"{key}__{side}_neginf"] = np.isneginf(value).astype(np.uint8)
    else:
        arrays[f"{key}__actual_values"] = np.ascontiguousarray(actual_array)
        arrays[f"{key}__expected_values"] = np.ascontiguousarray(expected_array)
    if bitwise and actual_array.dtype == expected_array.dtype:
        left = np.ascontiguousarray(actual_array).view(np.uint8).reshape(-1)
        right = np.ascontiguousarray(expected_array).view(np.uint8).reshape(-1)
        arrays[f"{key}__bitwise_xor"] = np.bitwise_xor(left, right)


def numeric_close(actual: np.ndarray, expected: np.ndarray,
                  atol: float = REGISTERED_ATOL,
                  rtol: float = REGISTERED_RTOL) -> tuple[bool, dict[str, Any]]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return False, {"shape_actual": list(actual.shape),
                       "shape_expected": list(expected.shape)}
    finite_match = np.array_equal(np.isfinite(actual), np.isfinite(expected))
    nan_match = np.array_equal(np.isnan(actual), np.isnan(expected))
    posinf_match = np.array_equal(np.isposinf(actual), np.isposinf(expected))
    neginf_match = np.array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    if finite.any():
        delta = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
        bound = atol + rtol * np.abs(expected[finite].astype(np.float64))
        within = bool(np.all(delta <= bound))
        max_abs = float(delta.max())
        max_ratio = float(np.max(delta / np.maximum(bound, np.finfo(float).tiny)))
    else:
        within, max_abs, max_ratio = True, 0.0, 0.0
    return bool(finite_match and nan_match and posinf_match and neginf_match and within), {
        "finite_topology_equal": bool(finite_match and nan_match and posinf_match and neginf_match),
        "finite_count": int(finite.sum()), "max_abs_delta": max_abs,
        "max_registered_bound_ratio": max_ratio, "atol": atol, "rtol": rtol,
    }


def integer_equal(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, dict[str, Any]]:
    same = actual.shape == expected.shape and np.array_equal(actual, expected)
    mismatch = int(np.count_nonzero(actual != expected)) if actual.shape == expected.shape else None
    return bool(same), {"mismatch_count": mismatch}


@dataclass
class CheckBook:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, check_id: str, passed: bool, detail: Any = None,
            *, evidence_gap: bool = False) -> bool:
        record: dict[str, Any] = {
            "id": check_id,
            "result": "pass" if passed else ("evidence_gap" if evidence_gap else "fail"),
        }
        if detail is not None:
            record["detail"] = detail
        self.checks.append(record)
        return passed

    def add_not_applicable(self, check_id: str, detail: Any) -> None:
        self.checks.append({"id": check_id, "result": "not_applicable",
                            "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["result"] in ("pass", "not_applicable")
                   for item in self.checks)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.checks
                if item["result"] not in ("pass", "not_applicable")]

    @property
    def evidence_gaps(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if item["result"] == "evidence_gap"]

    @property
    def hard_failures(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if item["result"] == "fail"]


@dataclass
class Reconstruction:
    planes: dict[str, np.ndarray]
    noise: np.ndarray
    normalization: dict[str, Any]
    science_policy: dict[str, Any]
    per_scan_sum_abs: dict[str, np.ndarray]
    ledger_identity: dict[str, Any]
    raw_numerators: dict[str, np.ndarray]
    verified_raw_parent_digest: str | None = None


LEDGER_ARRAYS = (
    "row", "col", "detector_index", "sample_index", "scan_index",
    "geometric_in_bounds", "upstream_eligible", "coefficient",
    "sample_signal", "sample_kernel", "sample_interval_s",
    "realization_signs",
)
LEDGER_SCALARS = (
    "schema_version", "candidate_sha", "raw_input_manifest_sha256",
    "producer_identity", "bundle_identity_digest", "obsnum", "array",
    "map_rows", "map_cols", "sample_rate_hz_numeric",
    "sample_rate_hz_hex", "sample_rate_hz_encoding",
)


def load_npz_scalar(bundle: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in bundle:
        die(f"sample ledger lacks scalar {name}")
    value = np.asarray(bundle[name])
    if value.size != 1:
        die(f"sample ledger scalar {name} has shape {value.shape}")
    return value.reshape(-1)[0].item()


def boost_mt19937_scan_signs(scan_count: int) -> np.ndarray:
    """Reproduce boost::mt19937 + uniform_int_distribution<int>(0,1)."""
    if scan_count <= 0:
        die("scan-sign reproduction requires a positive scan count")
    # Boost's mt19937 seed sequence is the standard 32-bit MT19937 sequence.
    # For destination range [0,1], Boost uniform_int_distribution divides the
    # full uint32 range into two 2^31 buckets, i.e. consumes the high bit.
    engine = np.random.RandomState(5489)
    words = engine.randint(
        0, 2 ** 32, size=scan_count * REALIZATIONS, dtype=np.uint32)
    bits = (words >> np.uint32(31)).astype(np.int8)
    return (2 * bits - 1).reshape(scan_count, REALIZATIONS)


def reconstruct_observation_from_ledger(
        path: Path, obsnum: int, array: str, shape: tuple[int, int], cut: float,
        authority: RawManifestAuthority) -> Reconstruction:
    membership = authority.memberships.get((obsnum, array))
    if not isinstance(membership, Mapping):
        die(f"{path}: raw-input authority lacks membership for {obsnum}:{array}")
    projection = membership["projection"]
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise EvidenceError(f"cannot load independent sample ledger {path}: {exc}") from exc
    with archive:
        expected_members = set(LEDGER_ARRAYS) | set(LEDGER_SCALARS)
        if set(archive.files) != expected_members:
            die(f"{path}: sample ledger members differ; "
                f"missing={sorted(expected_members - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - expected_members)}")
        missing = [name for name in LEDGER_ARRAYS if name not in archive]
        if missing:
            die(f"{path}: sample ledger arrays absent: {missing}")
        absent_scalars = [name for name in LEDGER_SCALARS if name not in archive]
        if absent_scalars:
            die(f"{path}: sample ledger binding scalars absent: {absent_scalars}")
        schema = str(load_npz_scalar(archive, "schema_version"))
        candidate = str(load_npz_scalar(archive, "candidate_sha"))
        raw_manifest_digest = str(load_npz_scalar(archive, "raw_input_manifest_sha256"))
        producer = str(load_npz_scalar(archive, "producer_identity"))
        bundle_digest = str(load_npz_scalar(archive, "bundle_identity_digest"))
        ledger_obs = int(load_npz_scalar(archive, "obsnum"))
        ledger_array = str(load_npz_scalar(archive, "array"))
        rows = int(load_npz_scalar(archive, "map_rows"))
        cols = int(load_npz_scalar(archive, "map_cols"))
        sample_rate_node = {
            "numeric": str(load_npz_scalar(archive, "sample_rate_hz_numeric")),
            "hex": str(load_npz_scalar(archive, "sample_rate_hz_hex")),
            "encoding": str(load_npz_scalar(archive, "sample_rate_hz_encoding")),
        }
        sample_rate_hz = exact_float_node(sample_rate_node,
                                          f"{path}: sample_rate_hz")
        if schema != LEDGER_SCHEMA:
            die(f"{path}: sample ledger schema differs from {LEDGER_SCHEMA}")
        if candidate != CANDIDATE_SHA:
            die(f"{path}: sample ledger candidate SHA differs")
        if raw_manifest_digest != authority.digest:
            die(f"{path}: sample ledger raw-input manifest binding differs")
        if producer != authority.producer_identity:
            die(f"{path}: sample ledger producer identity differs from raw authority")
        if re.fullmatch(r"canonical-hexfloat-sha256-v1:[0-9a-f]{64}",
                        bundle_digest) is None:
            die(f"{path}: sample ledger bundle identity digest is invalid")
        if bundle_digest != projection["identity_digest"]:
            die(f"{path}: sample ledger projection identity differs from raw authority")
        authority_shape = (projection["map_rows"], projection["map_cols"])
        if (ledger_obs, ledger_array, (rows, cols)) != \
                (obsnum, array, shape) or (rows, cols) != authority_shape:
            die(f"{path}: ledger identity/shape differs from map")
        vectors = {name: np.asarray(archive[name]) for name in LEDGER_ARRAYS[:-1]}
        signs = np.asarray(archive["realization_signs"])

    lengths = {name: int(value.size) for name, value in vectors.items()}
    if len(set(lengths.values())) != 1:
        die(f"{path}: ledger term arrays have inconsistent cardinality {lengths}")
    count = next(iter(lengths.values()), 0)
    if count <= 0:
        die(f"{path}: sample ledger has no projection records")
    for name, value in vectors.items():
        if value.ndim != 1:
            die(f"{path}: ledger array {name} must be one-dimensional")
    expected_dtypes = {
        "row": np.dtype("int64"), "col": np.dtype("int64"),
        "detector_index": np.dtype("int64"), "sample_index": np.dtype("int64"),
        "scan_index": np.dtype("int64"), "geometric_in_bounds": np.dtype("uint8"),
        "upstream_eligible": np.dtype("uint8"), "coefficient": np.dtype("float64"),
        "sample_signal": np.dtype("float64"), "sample_kernel": np.dtype("float64"),
        "sample_interval_s": np.dtype("float64"),
    }
    for name, dtype in expected_dtypes.items():
        if vectors[name].dtype != dtype:
            die(f"{path}: ledger {name} dtype differs: {vectors[name].dtype} != {dtype}")
    if signs.dtype != np.dtype("int8"):
        die(f"{path}: realization_signs dtype differs: {signs.dtype}")
    declared_count = int(membership["projection_record_count"])
    if count != declared_count:
        die(f"{path}: ledger term count differs from raw authority: "
            f"{count} != {declared_count}")
    if signs.ndim != 2 or signs.shape[1] != REALIZATIONS:
        die(f"{path}: realization_signs must have shape (nscan,{REALIZATIONS})")
    if signs.shape[0] <= 0 or not np.all(np.isfinite(signs)) or \
            not np.all(np.isin(signs, (-1, 1))):
        die(f"{path}: realization signs must be finite Rademacher values")
    scan_order = membership["scan_order"]
    detector_order = membership["detector_order"]
    if signs.shape[0] != len(scan_order):
        die(f"{path}: realization sign scan count differs from raw authority")
    expected_signs = boost_mt19937_scan_signs(len(scan_order))
    if not np.array_equal(signs.astype(np.int8, copy=False), expected_signs):
        die(f"{path}: realization signs differ from the pinned Boost MT19937 stream")

    row = vectors["row"].astype(np.int64, copy=False)
    col = vectors["col"].astype(np.int64, copy=False)
    detector = vectors["detector_index"].astype(np.int64, copy=False)
    sample = vectors["sample_index"].astype(np.int64, copy=False)
    scan = vectors["scan_index"].astype(np.int64, copy=False)
    geometric = vectors["geometric_in_bounds"]
    eligible = vectors["upstream_eligible"]
    coefficient = vectors["coefficient"].astype(np.float64, copy=False)
    signal = vectors["sample_signal"].astype(np.float64, copy=False)
    kernel = vectors["sample_kernel"].astype(np.float64, copy=False)
    interval = vectors["sample_interval_s"].astype(np.float64, copy=False)
    if not np.all(np.isin(geometric, (0, 1))) or not np.all(np.isin(eligible, (0, 1))):
        die(f"{path}: geometric/upstream states must be binary")
    geometric = geometric.astype(bool, copy=False)
    eligible = eligible.astype(bool, copy=False)
    if np.any(eligible & ~geometric):
        die(f"{path}: upstream-eligible record lies outside geometric projection")
    if np.any(geometric & ((row < 0) | (row >= rows) | (col < 0) | (col >= cols))):
        die(f"{path}: geometric record has an out-of-bounds pixel")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0 or \
            not exact_float_equal(sample_rate_hz, projection["_sample_rate_hz"]):
        die(f"{path}: sample rate must be finite and positive")
    expected_interval = np.float64(1.0 / sample_rate_hz)
    if not math.isfinite(float(expected_interval)) or expected_interval <= 0.0:
        die(f"{path}: reciprocal sample interval is invalid")
    interval_bits = np.ascontiguousarray(interval, dtype=np.float64).view(np.uint64)
    expected_bits = np.full(interval.shape, expected_interval,
                            dtype=np.float64).view(np.uint64)
    if not np.array_equal(interval_bits, expected_bits):
        die(f"{path}: sample_interval_s is not bit-equal to 1/sample_rate_hz")
    if np.any(scan < 0) or np.any(scan >= signs.shape[0]):
        die(f"{path}: scan_index is outside realization_signs")
    identities = np.stack((scan, detector, sample), axis=1)
    if np.unique(identities, axis=0).shape[0] != count:
        die(f"{path}: (scan,detector,sample) projection identity is repeated")
    order = np.lexsort((sample, detector, scan))
    if not np.array_equal(order, np.arange(count)):
        die(f"{path}: records are not scan-major/detector-major/sample-minor")
    cursor = 0
    for scan_record in scan_order:
        scan_index = int(scan_record["scan_index"])
        sample_count = int(scan_record["sample_count"])
        expected_samples = np.arange(sample_count, dtype=np.int64)
        for detector_record in detector_order:
            stop = cursor + sample_count
            detector_index = int(detector_record["detector_index"])
            if stop > count or not np.all(scan[cursor:stop] == scan_index) or \
                    not np.all(detector[cursor:stop] == detector_index) or \
                    not np.array_equal(sample[cursor:stop], expected_samples):
                die(f"{path}: ledger omits, fabricates, or reorders the exact "
                    f"Cartesian term block at scan={scan_index}, detector={detector_index}")
            cursor = stop
    if cursor != count:
        die(f"{path}: ledger contains terms outside raw-authorized membership")

    contribution = eligible & np.isfinite(coefficient) & (coefficient > 0.0)
    if np.any(eligible & ~np.isfinite(coefficient)):
        die(f"{path}: eligible coefficient is non-finite")
    if np.any(contribution & ~np.isfinite(signal)):
        die(f"{path}: contributing signal is non-finite")
    if np.any(contribution & ~np.isfinite(kernel)):
        die(f"{path}: contributing kernel is non-finite")

    def zeros_float() -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    def zeros_int() -> np.ndarray:
        return np.zeros(shape, dtype=np.int64)

    geometric_hits = zeros_int()
    contributing_hits = zeros_int()
    eligible_exposure = zeros_float()
    retained = zeros_float()
    numerator = zeros_float()
    weight = zeros_float()
    kernel_num = zeros_float()
    noise_num = np.zeros((*shape, REALIZATIONS), dtype=np.float64)
    nscan = signs.shape[0]
    scan_planes = {
        "signal_I": np.zeros((*shape, nscan), dtype=np.float64),
        "weight_I": np.zeros((*shape, nscan), dtype=np.float64),
        "kernel_I": np.zeros((*shape, nscan), dtype=np.float64),
        "retained_exposure_I": np.zeros((*shape, nscan), dtype=np.float64),
        "upstream_eligible_exposure_I": np.zeros((*shape, nscan), dtype=np.float64),
    }
    for index in range(count):
        if not geometric[index]:
            continue
        location = (row[index], col[index])
        geometric_hits[location] += 1
        if not eligible[index]:
            continue
        dt = float(interval[index])
        eligible_exposure[location] += dt
        scan_planes["upstream_eligible_exposure_I"][location + (scan[index],)] += dt
        if not contribution[index]:
            continue
        coeff = float(coefficient[index])
        weighted_signal = coeff * float(signal[index])
        weighted_kernel = coeff * float(kernel[index])
        if not np.isfinite(weighted_signal) or not np.isfinite(weighted_kernel):
            die(f"{path}: weighted ledger contribution overflowed")
        contributing_hits[location] += 1
        retained[location] += dt
        numerator[location] += weighted_signal
        weight[location] += coeff
        kernel_num[location] += weighted_kernel
        noise_num[location] += signs[scan[index]] * weighted_signal
        scan_planes["signal_I"][location + (scan[index],)] += weighted_signal
        scan_planes["weight_I"][location + (scan[index],)] += coeff
        scan_planes["kernel_I"][location + (scan[index],)] += weighted_kernel
        scan_planes["retained_exposure_I"][location + (scan[index],)] += dt

    normalization = threshold_selection(weight, cut / 10.0)
    norm = np.isfinite(weight) & (weight > 0.0) & \
        (weight >= normalization["threshold"])
    final_weight = np.where(norm, weight, 0.0)
    final_signal = np.where(norm, numerator / np.where(weight > 0.0, weight, 1.0), 0.0)
    final_kernel = np.where(norm, kernel_num / np.where(weight > 0.0, weight, 1.0), 0.0)
    final_noise = np.where(norm[..., None],
                           noise_num / np.where(weight > 0.0, weight, 1.0)[..., None],
                           0.0)
    final_retained = np.where(norm, retained, 0.0)
    science_policy = threshold_selection(final_weight, cut)
    policy = np.isfinite(final_weight) & (final_weight > 0.0) & \
        (final_weight >= science_policy["threshold"])
    companions_finite = np.isfinite(final_signal) & np.isfinite(final_weight) & \
        (final_weight > 0.0) & np.isfinite(final_kernel) & \
        np.all(np.isfinite(final_noise), axis=-1)
    valid = norm & policy & companions_finite
    planes = {
        "signal_I": final_signal,
        "weight_I": final_weight,
        "kernel_I": final_kernel,
        "geometric_hits_I": geometric_hits,
        "contributing_hits_I": contributing_hits,
        "upstream_eligible_exposure_I": eligible_exposure,
        "retained_exposure_I": final_retained,
        "normalization_support_I": norm.astype(np.uint8),
        "science_policy_support_I": policy.astype(np.uint8),
        "science_valid_I": valid.astype(np.uint8),
        "coverage_I": final_retained.copy(),
        "coverage_bool_I": policy.astype(np.uint8),
    }
    # This is an independent sequential reconstruction of per-scan planes.  It
    # is useful for F010 reconstruction but is not the run-produced binary64
    # pre-final-reduction authority required by the scan-farm gamma_n claim.
    collapsed_abs = {
        name: np.sum(np.abs(value.astype(np.longdouble)), axis=-1,
                     dtype=np.longdouble)
        for name, value in scan_planes.items()
    }
    return Reconstruction(
        planes=planes, noise=final_noise,
        normalization=normalization, science_policy=science_policy,
        per_scan_sum_abs=collapsed_abs,
        ledger_identity={
            "schema_version": schema, "obsnum": ledger_obs, "array": ledger_array,
            "shape": [rows, cols], "projection_record_count": count,
            "scan_count": int(nscan), "sha256": sha256(path),
            "candidate_sha": candidate,
            "raw_input_manifest_sha256": raw_manifest_digest,
            "producer_identity": producer,
            "bundle_identity_digest": bundle_digest,
            "sample_rate_hz": sample_rate_node,
            "raw_membership_projection_record_count": declared_count,
            "raw_membership_scan_count": len(scan_order),
            "raw_membership_detector_count": len(detector_order),
            "raw_membership_complete": True,
        },
        raw_numerators={
            "signal_numerator": numerator.copy(),
            "weight_I": weight.copy(),
            "kernel_numerator": kernel_num.copy(),
            "retained_exposure_I": retained.copy(),
            "upstream_eligible_exposure_I": eligible_exposure.copy(),
        },
    )


def threshold_record_check(record: Mapping[str, Any], expected: Mapping[str, Any],
                           cut: float, algorithm: str, stage: str,
                           label: str, book: CheckBook) -> None:
    try:
        actual_cut = exact_float_node(record["requested_cut"], label + ".requested_cut")
        realized_cut = exact_float_node(record["realized_cut"], label + ".realized_cut")
        threshold = exact_float_node(record["realized_threshold"], label + ".threshold")
        selected = exact_float_node(record["selected_positive_value"], label + ".selected")
    except (KeyError, TypeError) as exc:
        book.add(label + ".record_complete", False, str(exc))
        return
    book.add(label + ".algorithm",
             record.get("order_statistic_algorithm") == SCIENCE_ALGORITHMS["order_statistic_algorithm"]
             and record.get("support_algorithm") == algorithm
             and record.get("coefficient_product") == "weight_I"
             and record.get("coefficient_stage") == stage,
             {"record": dict(record)})
    book.add(label + ".conventions",
             record.get("finite_convention") == "coefficient must be finite"
             and record.get("positivity_convention") == "coefficient > 0"
             and record.get("comparison_convention") == ">=",
             {"finite": record.get("finite_convention"),
              "positive": record.get("positivity_convention"),
              "comparison": record.get("comparison_convention")})
    index_record = record.get("selected_zero_based_index", {})
    actual_index = index_record.get("value") if index_record.get("available") else None
    book.add(label + ".exact_values",
             exact_float_equal(actual_cut, cut)
             and exact_float_equal(realized_cut, cut)
             and exact_float_equal(threshold, float(expected["threshold"]))
             and exact_float_equal(selected, float(expected["selected"]))
             and int(record.get("positive_value_count", -1)) == int(expected["count"])
             and actual_index == expected["index"],
             {"actual": {"cut": actual_cut, "realized_cut": realized_cut,
                          "threshold": threshold, "selected": selected,
                          "count": record.get("positive_value_count"),
                          "index": actual_index},
              "expected": dict(expected)})


def realized_map_record(provenance: Mapping[str, Any], scope: str, obsnum: int | None,
                        array: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if scope == "observation":
        observations = provenance.get("observations")
        if not isinstance(observations, list):
            die("mapmaking provenance observations are absent")
        matches = [item for item in observations if isinstance(item, Mapping)
                   and int(item.get("obsnum", -1)) == int(obsnum)]
        if len(matches) != 1:
            die(f"mapmaking provenance does not uniquely identify observation {obsnum}")
        state = matches[0].get("science_state", {})
        identity_node = state.get("bundle_identity", {})
        if not state.get("available") or not identity_node.get("available"):
            die(f"observation {obsnum} science-state identity is unavailable")
        identity = identity_node.get("value", {})
        realized = state.get("realized_maps")
    else:
        state = provenance.get("observation_resolved", {})
        identity_node = state.get("common_identity", {})
        if not state.get("available") or not identity_node.get("available"):
            die("coadd science-state identity is unavailable")
        identity = identity_node.get("value", {})
        realized = state.get("realized_maps")
    slots = identity.get("ordered_slots") if isinstance(identity, Mapping) else None
    if not isinstance(slots, list) or not isinstance(realized, list) or len(slots) != len(realized):
        die("science-state slot/realized cardinality is inconsistent")
    slot_index = ARRAYS.index(array)
    matches = [item for item in slots if isinstance(item, Mapping)
               and int(item.get("ordered_slot", -1)) == slot_index]
    if len(matches) != 1:
        die(f"science-state identity has no unique ordered slot for {array}")
    record_matches = [item for item in realized if isinstance(item, Mapping)
                      and int(item.get("ordered_slot", -1)) == slot_index]
    if len(record_matches) != 1:
        die(f"science-state realized maps have no unique slot for {array}")
    return identity, record_matches[0]


def product_inventory_check(record: Mapping[str, Any], product: FitsProduct,
                            scope: str, book: CheckBook, prefix: str) -> None:
    entries = record.get("products")
    if not isinstance(entries, list) or len(entries) != 8:
        book.add(prefix + ".provenance_product_inventory", False,
                 "realized product inventory must contain eight canonical entries")
        return
    by_name = {entry.get("identity"): entry for entry in entries
               if isinstance(entry, Mapping)}
    canonical = F010_COADD[:-2] if scope == "coadd" else (
        "geometric_hits_I", "contributing_hits_I", "coadd_observation_count_I",
        "upstream_eligible_exposure_I", "retained_exposure_I",
        "normalization_support_I", "science_policy_support_I", "science_valid_I")
    book.add(prefix + ".provenance_product_names", set(by_name) == set(canonical),
             {"actual": sorted(str(name) for name in by_name),
              "expected": sorted(canonical)})
    for name in canonical:
        entry = by_name.get(name)
        should_exist = not (scope == "observation" and name == "coadd_observation_count_I")
        if not isinstance(entry, Mapping):
            continue
        expected_absence_reason = (
            "" if should_exist else "not applicable to observation maps")
        book.add(prefix + f".{name}.availability",
                 bool(entry.get("available")) == should_exist
                 and entry.get("absence_reason") == expected_absence_reason,
                 {"available": entry.get("available"),
                  "absence_reason": entry.get("absence_reason"),
                  "expected_absence_reason": expected_absence_reason})
        if not should_exist or name not in product.hdus:
            continue
        array = internal_spatial(product.array(name))
        nonzero = int(np.count_nonzero(array))
        if name in ("upstream_eligible_exposure_I", "retained_exposure_I"):
            try:
                expected_sum = float.fromhex(str(entry.get("value_sum")))
            except ValueError:
                expected_sum = math.nan
            # Eigen matrices are column-major. Reproduce Plane::sum()'s
            # binary64 association instead of accepting NumPy's row-major
            # reduction as an exact provenance check.
            actual_sum = 0.0
            for col in range(array.shape[1]):
                for row in range(array.shape[0]):
                    actual_sum += float(array[row, col])
            sum_ok = exact_float_equal(actual_sum, expected_sum)
        else:
            try:
                expected_sum = int(str(entry.get("value_sum")))
            except ValueError:
                expected_sum = -1
            actual_sum = int(np.sum(array, dtype=np.int64))
            sum_ok = actual_sum == expected_sum
        book.add(prefix + f".{name}.realized_facts",
                 nonzero == int(entry.get("nonzero_count", -1)) and sum_ok,
                 {"nonzero_actual": nonzero,
                  "nonzero_recorded": entry.get("nonzero_count"),
                  "sum_actual": actual_sum, "sum_recorded": entry.get("value_sum")})


def apply_contract_check(product: FitsProduct, check: Mapping[str, Any],
                         book: CheckBook, prefix: str,
                         residuals: dict[str, np.ndarray] | None = None) -> None:
    names = set(product.hdus)
    minimum = check.get("min_hdus")
    if minimum is not None:
        book.add(prefix + ".minimum_hdus", len(product.hdul) >= int(minimum),
                 {"actual": len(product.hdul), "minimum": minimum})
    if "primary_bunit" in check:
        actual_primary_unit = product.hdul[0].header.get("BUNIT")
        book.add(prefix + ".primary_bunit",
                 actual_primary_unit == check["primary_bunit"],
                 {"actual": actual_primary_unit,
                  "expected": check["primary_bunit"]})
    required = tuple(check.get("required_extnames", ()))
    forbidden = tuple(check.get("forbidden_extnames", ()))
    book.add(prefix + ".required_extensions",
             all(name in names for name in required),
             {"missing": sorted(set(required) - names)})
    book.add(prefix + ".forbidden_extensions",
             all(name not in names for name in forbidden),
             {"present": sorted(set(forbidden) & names)})
    for name, expected in check.get("required_ext_bitpix", {}).items():
        actual = product.hdus[name].header.get("BITPIX") if name in product.hdus else None
        book.add(prefix + f".{name}.bitpix", actual == expected,
                 {"actual": actual, "expected": expected})
    for name, expected in check.get("required_ext_dtypes", {}).items():
        actual = dtype_name(product.array(name)) if name in product.hdus else None
        book.add(prefix + f".{name}.dtype", actual == expected,
                 {"actual": actual, "expected": expected})
    for name, expected in check.get("ext_bunits", {}).items():
        actual = product.hdus[name].header.get("BUNIT") if name in product.hdus else None
        book.add(prefix + f".{name}.bunit", actual == expected,
                 {"actual": actual, "expected": expected})
    for name, headers in check.get("required_ext_headers", {}).items():
        if name not in product.hdus:
            continue
        for key, expected in headers.items():
            actual = product.hdus[name].header.get(key)
            book.add(prefix + f".{name}.header.{key}", str(actual) == str(expected),
                     {"actual": actual, "expected": expected})
    for name, choices in check.get("required_ext_header_one_of", {}).items():
        if name not in product.hdus:
            continue
        for key, allowed in choices.items():
            actual = product.hdus[name].header.get(key)
            book.add(prefix + f".{name}.header_one_of.{key}", actual in allowed,
                     {"actual": actual, "allowed": allowed})
    for name in check.get("binary_extnames", ()):
        if name in product.hdus:
            values = product.array(name)
            book.add(prefix + f".{name}.binary",
                     bool(np.all(np.isin(values, (0, 1)))),
                     {"unique": np.unique(values).tolist()})
    for canonical, alias in check.get("exact_aliases", {}).items():
        if canonical in product.hdus and alias in product.hdus:
            left, right = product.array(canonical), product.array(alias)
            equal = left.dtype == right.dtype and left.shape == right.shape and \
                left.tobytes(order="C") == right.tobytes(order="C")
            book.add(prefix + f".alias.{alias}", equal,
                     {"canonical": canonical, "relationship": "bitwise_equal"})
            if residuals is not None:
                record_lossless_residual(
                    residuals, f"alias_{canonical}_to_{alias}", left, right,
                    bitwise=True)
    shapes = check.get("same_shape_extnames", ())
    available = [name for name in shapes if name in product.hdus]
    shape_values = {name: list(product.array(name).shape) for name in available}
    book.add(prefix + ".same_shape",
             len(available) == len(shapes) and len({tuple(v) for v in shape_values.values()}) <= 1,
             shape_values)
    wcs_names = check.get("same_wcs_extnames", ())
    available = [name for name in wcs_names if name in product.hdus]
    wcs_values = {name: wcs_cards(product.hdus[name].header) for name in available}
    book.add(prefix + ".same_wcs",
             len(available) == len(wcs_names) and
             len({json.dumps(v, sort_keys=True, default=str) for v in wcs_values.values()}) <= 1,
             {"extension_count": len(available), "required_count": len(wcs_names)})
    axis_types = check.get("axis_types")
    axis_units = check.get("axis_units")
    if axis_types and "signal_I" in product.hdus:
        header = product.hdus["signal_I"].header
        actual_types = [header.get(f"CTYPE{index}") for index in range(1, len(axis_types) + 1)]
        actual_units = [header.get(f"CUNIT{index}") for index in range(1, len(axis_units) + 1)]
        unit_match = all((actual == expected or (expected is None and actual in (None, "")))
                         for actual, expected in zip(actual_units, axis_units))
        book.add(prefix + ".axis_types", actual_types == list(axis_types),
                 {"actual": actual_types, "expected": axis_types})
        book.add(prefix + ".axis_units", unit_match,
                 {"actual": actual_units, "expected": axis_units})


def verify_noise_file(product: FitsProduct, expected_shape: tuple[int, ...],
                      reference_header: fits.Header, expected_unit: str,
                      book: CheckBook, prefix: str) -> np.ndarray | None:
    pattern = re.compile(r"signal_([0-9]+)_I$")
    indexed: dict[int, str] = {}
    for name in product.hdus:
        match = pattern.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index in indexed:
                book.add(prefix + ".unique_realization_indices", False,
                         {"repeated": index})
            indexed[index] = name
    expected_indices = set(range(REALIZATIONS))
    expected_names = {f"signal_{index}_I" for index in expected_indices}
    inventory_exact = set(indexed) == expected_indices and \
        set(product.hdus) == expected_names
    book.add(prefix + ".realization_inventory", inventory_exact,
             {"missing": sorted(expected_indices - set(indexed)),
              "extra_indices": sorted(set(indexed) - expected_indices),
              "extra_hdus": sorted(set(product.hdus) - expected_names)})
    if not inventory_exact:
        return None
    arrays = []
    medrms_values: list[float] = []
    reference_wcs = wcs_cards(reference_header)
    rows, cols = int(expected_shape[-2]), int(expected_shape[-1])
    sample_pixels = np.asarray([
        [0.0, 0.0], [float(cols - 1), 0.0],
        [0.0, float(rows - 1)], [float(cols - 1), float(rows - 1)],
        [(float(cols) - 1.0) / 2.0, (float(rows) - 1.0) / 2.0],
    ])

    def sampled_world(header: fits.Header) -> np.ndarray:
        wcs = WCS(header)
        points = np.empty((sample_pixels.shape[0], wcs.pixel_n_dim), dtype=float)
        points[:, 0:2] = sample_pixels
        for axis in range(2, wcs.pixel_n_dim):
            points[:, axis] = float(header.get(f"CRPIX{axis + 1}", 1.0)) - 1.0
        return np.asarray(wcs.all_pix2world(points, 0), dtype=np.float64)

    reference_world = sampled_world(reference_header)
    for index in range(REALIZATIONS):
        array = product.array(indexed[index])
        header = product.hdus[indexed[index]].header
        book.add(prefix + f".realization_{index}.shape", array.shape == expected_shape,
                 {"actual": list(array.shape), "expected": list(expected_shape)})
        book.add(prefix + f".realization_{index}.dtype", dtype_name(array) == "float64",
                 dtype_name(array))
        book.add(prefix + f".realization_{index}.wcs_cards",
                 wcs_cards(header) == reference_wcs, None)
        try:
            world = sampled_world(header)
            residual = np.abs(world - reference_world)
            if residual.shape[1] and str(header.get("CTYPE1", "")).startswith("RA"):
                residual[:, 0] = np.abs(
                    (world[:, 0] - reference_world[:, 0] + 180.0) % 360.0 - 180.0)
            transform_ok = np.array_equal(np.isfinite(world),
                                          np.isfinite(reference_world)) and \
                (not np.isfinite(residual).any() or
                 float(np.max(residual[np.isfinite(residual)])) <= 1.0e-12)
            transform_detail: Any = {
                "sample_count": int(sample_pixels.shape[0]),
                "max_world_residual": float(np.max(residual[np.isfinite(residual)]))
                if np.isfinite(residual).any() else None,
            }
        except Exception as exc:
            transform_ok, transform_detail = False, str(exc)
        book.add(prefix + f".realization_{index}.wcs_transform",
                 transform_ok, transform_detail)
        medrms_value = header.get("MEDRMS")
        try:
            medrms = float(medrms_value)
        except (TypeError, ValueError):
            medrms = math.nan
        medrms_values.append(medrms)
        metadata_ok = header.get("UNIT") == expected_unit and \
            math.isfinite(medrms) and medrms > 0.0 and all(
                key not in header for key in ("BUNIT", "DESCRIP", "TYPE", "ESTTYPE"))
        book.add(prefix + f".realization_{index}.metadata", metadata_ok,
                 {"UNIT": header.get("UNIT"), "expected_UNIT": expected_unit,
                  "MEDRMS": medrms_value,
                  "forbidden_present": [key for key in
                      ("BUNIT", "DESCRIP", "TYPE", "ESTTYPE") if key in header]})
        arrays.append(internal_spatial(array))
    book.add(prefix + ".MEDRMS_exact_across_realizations",
             all(exact_float_equal(value, medrms_values[0])
                 for value in medrms_values[1:]),
             {"count": len(medrms_values), "first": medrms_values[0]})
    return np.stack(arrays, axis=-1)


def binary32_value(value: float) -> float:
    return float(np.float32(float(value)))


def verify_identity_wcs_adapter(identity: Mapping[str, Any], array: str,
                                product: FitsProduct, book: CheckBook,
                                prefix: str) -> None:
    wcs = identity.get("wcs")
    slots = identity.get("ordered_slots")
    if not isinstance(wcs, Mapping) or not isinstance(slots, list):
        book.add(prefix + ".identity_wcs_adapter", False,
                 "identity WCS or ordered slots are malformed")
        return
    slot_index = ARRAYS.index(array)
    slot_matches = [slot for slot in slots if isinstance(slot, Mapping)
                    and int(slot.get("ordered_slot", -1)) == slot_index]
    if len(slot_matches) != 1:
        book.add(prefix + ".identity_wcs_adapter.slot", False,
                 {"slot_index": slot_index, "matches": len(slot_matches)})
        return
    slot = slot_matches[0]
    header = product.hdus["signal_I"].header
    try:
        axis_types = list(wcs["axis_types"])
        axis_units = list(wcs["axis_units"])
        pixel_scale = [exact_float_node(node, f"{prefix}.pixel_scale[{index}]")
                       for index, node in enumerate(wcs["pixel_scale"])]
        reference_world = [
            exact_float_node(node, f"{prefix}.reference_world[{index}]")
            for index, node in enumerate(wcs["reference_world"])]
        reference_pixel = [
            exact_float_node(node, f"{prefix}.reference_pixel[{index}]")
            for index, node in enumerate(wcs["reference_pixel"])]
        source_epoch = exact_float_node(wcs["source_epoch"],
                                        f"{prefix}.source_epoch")
        orientation = exact_float_node(wcs["orientation_rad"],
                                        f"{prefix}.orientation_rad")
        frequency = exact_float_node(slot["frequency_hz"],
                                     f"{prefix}.slot.frequency_hz")
    except (KeyError, TypeError) as exc:
        book.add(prefix + ".identity_wcs_adapter.exact_nodes", False, str(exc))
        return
    book.add(prefix + ".identity_wcs_adapter.cardinality",
             len(axis_types) == len(axis_units) == len(pixel_scale) ==
             len(reference_world) == len(reference_pixel) == 2,
             {"axis_types": len(axis_types), "axis_units": len(axis_units),
              "pixel_scale": len(pixel_scale),
              "reference_world": len(reference_world),
              "reference_pixel": len(reference_pixel)})
    if not all(len(values) == 2 for values in
               (axis_types, axis_units, pixel_scale,
                reference_world, reference_pixel)):
        return
    card_pairs = {
        "CTYPE1": axis_types[0], "CTYPE2": axis_types[1],
        "CUNIT1": axis_units[0], "CUNIT2": axis_units[1],
        "CTYPE3": "FREQ", "CTYPE4": "STOKES", "CUNIT3": "Hz",
    }
    book.add(prefix + ".identity_wcs_adapter.string_cards",
             all(header.get(key) == value for key, value in card_pairs.items())
             and header.get("CUNIT4", "") in (None, ""),
             {key: header.get(key) for key in (*card_pairs, "CUNIT4")})
    numeric_expected = {
        "CRVAL1": binary32_value(reference_world[0]),
        "CRVAL2": binary32_value(reference_world[1]),
        "CDELT1": binary32_value(pixel_scale[0]),
        "CDELT2": binary32_value(pixel_scale[1]),
        "CRPIX1": binary32_value(reference_pixel[0]) + 1.0,
        "CRPIX2": binary32_value(reference_pixel[1]) + 1.0,
        "CRVAL3": binary32_value(frequency),
        "CRVAL4": binary32_value(float(slot.get("stokes_identity"))),
        "CRPIX3": 1.0, "CRPIX4": 1.0,
        "CDELT3": 1.0, "CDELT4": 1.0,
    }
    mismatches = {
        key: {"actual": header.get(key), "expected": expected}
        for key, expected in numeric_expected.items()
        if header.get(key) is None or not exact_float_equal(float(header[key]), expected)
    }
    book.add(prefix + ".identity_wcs_adapter.numeric_cards", not mismatches,
             mismatches)
    shape = identity.get("shape", {})
    book.add(prefix + ".identity_wcs_adapter.shape",
             header.get("NAXIS1") == int(shape.get("cols", -1))
             and header.get("NAXIS2") == int(shape.get("rows", -1)),
             {"fits": [header.get("NAXIS2"), header.get("NAXIS1")],
              "identity": [shape.get("rows"), shape.get("cols")]})
    book.add(prefix + ".identity_wcs_adapter.epoch_orientation",
             header.get("EQUINOX") is not None
             and exact_float_equal(float(header["EQUINOX"]), source_epoch)
             and exact_float_equal(orientation, 0.0),
             {"fits_epoch": header.get("EQUINOX"),
              "identity_epoch": source_epoch, "orientation_rad": orientation})
    book.add(prefix + ".identity_wcs_adapter.slot",
             slot.get("grouping") == "array"
             and slot.get("group_identity") == f"array:{slot_index}"
             and int(slot.get("array_identity", -1)) == slot_index
             and int(slot.get("stokes_identity", -1)) == 0
             and int(slot.get("ordered_slot", -1)) == slot_index,
             dict(slot))


def verify_science_identity(identity: Mapping[str, Any], record: Mapping[str, Any],
                            map_record: Mapping[str, Any], product: FitsProduct,
                            mode: str, book: CheckBook, prefix: str) -> str:
    policies = identity.get("policies")
    identity_wcs = identity.get("wcs")
    expected_frame = "altaz" if mode == "point" else "radec"
    expected_projection = "offset-plane" if mode == "point" else "TAN"
    wcs_vocabulary_exact = isinstance(identity_wcs, Mapping) and \
        identity_wcs.get("coordinate_frame") == expected_frame and \
        identity_wcs.get("projection") == expected_projection
    policy_exact = isinstance(policies, Mapping) and \
        policies.get("validity") == SCIENCE_ALGORITHMS["validity_algorithm"] and \
        policies.get("coefficient") == COEFFICIENT_POLICY and \
        policies.get("normalization_support") == \
        SCIENCE_ALGORITHMS["normalization_support_algorithm"] and \
        policies.get("science_policy_support") == \
        SCIENCE_ALGORITHMS["science_policy_support_algorithm"] and \
        policies.get("nonfinite") == SCIENCE_ALGORITHMS["nonfinite_policy"]
    book.add(prefix + ".identity_contract",
             identity.get("contract_version") == SCIENCE_ALGORITHMS["contract_version"]
             and identity.get("grouping") == "array"
             and identity.get("signal_unit") == "mJy/beam"
             and identity.get("estimator_identity") == ESTIMATOR_IDENTITY
             and canonical_digest_string(identity.get("response_identity"))
             and identity.get("parallel_equivalence_policy") == PARALLEL_POLICY
             and policy_exact and wcs_vocabulary_exact,
             {"contract_version": identity.get("contract_version"),
              "grouping": identity.get("grouping"),
              "signal_unit": identity.get("signal_unit"),
              "estimator_identity": identity.get("estimator_identity"),
              "response_identity": identity.get("response_identity"),
              "wcs_coordinate_frame": (
                  identity_wcs.get("coordinate_frame")
                  if isinstance(identity_wcs, Mapping) else None),
              "expected_wcs_coordinate_frame": expected_frame,
              "wcs_projection": (identity_wcs.get("projection")
                                 if isinstance(identity_wcs, Mapping) else None),
              "expected_wcs_projection": expected_projection,
              "parallel_equivalence_policy": identity.get("parallel_equivalence_policy"),
              "policies": policies})
    slots = identity.get("ordered_slots")
    slots_exact = isinstance(slots, list) and len(slots) == len(ARRAYS) and all(
        isinstance(slot, Mapping)
        and slot.get("ordered_slot") == index
        and slot.get("grouping") == "array"
        and slot.get("group_identity") == f"array:{index}"
        and slot.get("array_identity") == index
        and slot.get("stokes_identity") == 0
        for index, slot in enumerate(slots))
    book.add(prefix + ".identity_ordered_slots_exact", slots_exact,
             {"expected_slot_count": len(ARRAYS), "actual": slots})
    shape = identity.get("shape", {})
    actual_shape = image_spatial_shape(product.array("signal_I"))
    book.add(prefix + ".identity_shape",
             (shape.get("rows"), shape.get("cols")) == actual_shape,
             {"identity": shape, "fits": actual_shape})
    digest = identity.get("identity_digest")
    recomputed_digest = recompute_bundle_identity_digest(identity)
    book.add(prefix + ".identity_digest_recomputed",
             canonical_digest_string(digest) and digest == recomputed_digest,
             {"recorded": digest, "recomputed": recomputed_digest})
    book.add(prefix + ".admitted_identity",
             bool(record.get("admitted_bundle_identity"))
             and record.get("admitted_bundle_identity") == digest,
             {"identity_digest": digest,
              "admitted": record.get("admitted_bundle_identity")})
    book.add(prefix + ".realized_initialized", record.get("initialized") is True,
             record.get("initialized"))
    book.add(prefix + ".raw_parent_digest",
             canonical_digest_string(record.get("raw_parent_digest")),
             record.get("raw_parent_digest"))
    companions = record.get("required_companions")
    identity_companions = identity.get("required_companions")
    expected_companions = ["kernel_I"] + [f"noise_realization_{i}_I"
                                           for i in range(REALIZATIONS)]
    book.add(prefix + ".required_companions",
             companions == identity_companions == expected_companions,
             {"actual": companions, "expected_count": len(expected_companions)})
    return recomputed_digest


def verify_thresholds(record: Mapping[str, Any], reconstruction: Reconstruction,
                      cut: float, scope: str, product: FitsProduct,
                      book: CheckBook, prefix: str) -> None:
    thresholds = record.get("thresholds", {})
    normal_record = thresholds.get("normalization", {})
    policy_record = thresholds.get("science_policy", {})
    normal_stage = NORMALIZATION_STAGE_COADD if scope == "coadd" else NORMALIZATION_STAGE_OBS
    threshold_record_check(
        normal_record, reconstruction.normalization, cut / 10.0,
        SCIENCE_ALGORITHMS["normalization_support_algorithm"], normal_stage,
        prefix + ".normalization_threshold", book)
    policy_stage = policy_record.get("coefficient_stage")
    expected_policy_stage = (
        "post-observation-normalization-no-empirical-rescale"
        if scope == "observation" else
        "post-coadd-normalization-no-empirical-rescale")
    book.add(prefix + ".science_policy_stage",
             policy_stage == expected_policy_stage,
             {"actual": policy_stage, "expected": expected_policy_stage})
    threshold_record_check(
        policy_record, reconstruction.science_policy, cut,
        SCIENCE_ALGORITHMS["science_policy_support_algorithm"],
        str(policy_stage), prefix + ".science_policy_threshold", book)
    normal_header = product.hdus["normalization_support_I"].header.get("WTTHRESH")
    book.add(prefix + ".normalization_support_I.WTTHRESH",
             normal_header is not None and exact_float_equal(
                 float(normal_header), reconstruction.normalization["threshold"]),
             {"actual": normal_header,
              "expected": reconstruction.normalization["threshold"]})
    header_thresholds = []
    for name in ("science_policy_support_I", "coverage_bool_I"):
        value = product.hdus[name].header.get("WTTHRESH")
        header_thresholds.append(value)
        book.add(prefix + f".{name}.WTTHRESH",
                 value is not None and exact_float_equal(float(value),
                                                         reconstruction.science_policy["threshold"]),
                 {"actual": value,
                  "expected": reconstruction.science_policy["threshold"]})
    book.add(prefix + ".policy_threshold_headers_equal",
             len(header_thresholds) == 2
             and exact_float_equal(float(header_thresholds[0]), float(header_thresholds[1])),
             header_thresholds)


def compare_reconstruction(product: FitsProduct, noise: np.ndarray | None,
                           reconstruction: Reconstruction, book: CheckBook,
                           prefix: str,
                           residuals: dict[str, np.ndarray]) -> None:
    for name, expected in reconstruction.planes.items():
        actual = internal_spatial(product.array(name))
        if name in INTEGER_PLANES:
            passed, detail = integer_equal(actual, expected)
        else:
            passed, detail = numeric_close(actual, expected)
        book.add(prefix + f".reconstruct.{name}", passed, detail)
        record_lossless_residual(
            residuals, f"reconstruct_{name}", actual, expected)
    if noise is None:
        book.add(prefix + ".reconstruct.realizations", False,
                 "realization FITS unavailable; sample ledger alone cannot prove serialized companions",
                 evidence_gap=True)
    else:
        passed, detail = numeric_close(noise, reconstruction.noise)
        book.add(prefix + ".reconstruct.realizations", passed, detail)
        record_lossless_residual(
            residuals, "reconstruct_noise_realizations", noise,
            reconstruction.noise)


def validate_empirical_products(product: FitsProduct, noise: np.ndarray | None,
                                products_enabled: bool, scope: str,
                                book: CheckBook, prefix: str,
                                noise_product: FitsProduct | None = None,
                                residuals: dict[str, np.ndarray] | None = None) -> None:
    names = set(product.hdus)
    if scope == "coadd":
        book.add(prefix + ".coadd_forbidden_statistical_products",
                 not (names & set(COADD_FORBIDDEN)),
                 {"present": sorted(names & set(COADD_FORBIDDEN))})
        return
    if products_enabled:
        book.add(prefix + ".empirical_inventory",
                 set(EMPIRICAL_PLANES).issubset(names)
                 and "formal_standardized_signal_I" not in names,
                 {"missing": sorted(set(EMPIRICAL_PLANES) - names),
                  "formal_standardized_present": "formal_standardized_signal_I" in names})
        if not set(EMPIRICAL_PLANES).issubset(names) or noise is None:
            return
        weight = internal_spatial(product.array("weight_I"))
        formal = internal_spatial(product.array("weight_formal_I"))
        book.add(prefix + ".formal_weight_snapshot_exact",
                 weight.dtype == formal.dtype and weight.shape == formal.shape
                 and weight.tobytes() == formal.tobytes(), None)
        if residuals is not None:
            record_lossless_residual(
                residuals, "formal_weight_snapshot", weight, formal,
                bitwise=True)
        _, variance_expected, _ = population_noise_statistics(noise)
        variance_actual = internal_spatial(product.array("noise_variance_I"))
        passed, detail = numeric_close(variance_actual, variance_expected)
        book.add(prefix + ".noise_variance_identity", passed, detail)
        if residuals is not None:
            record_lossless_residual(
                residuals, "noise_variance_identity", variance_actual,
                variance_expected)
        signal = internal_spatial(product.array("signal_I"))
        weight_header = product.hdus["weight_I"].header
        scale_value = weight_header.get("EMP_SCALE")
        ratio_value = weight_header.get("WVARMED")
        header_values_ok = scale_value is not None and ratio_value is not None
        try:
            scale = float(scale_value)
            ratio = float(ratio_value)
            header_values_ok = header_values_ok and math.isfinite(scale) and \
                scale > 0.0 and math.isfinite(ratio) and ratio > 0.0
        except (TypeError, ValueError):
            scale, ratio = math.nan, math.nan
            header_values_ok = False
        book.add(prefix + ".empirical_scale_headers", header_values_ok,
                 {"EMP_SCALE": scale_value, "WVARMED": ratio_value})
        book.add(prefix + ".empirical_scale_reciprocal",
                 header_values_ok and exact_float_equal(scale, 1.0 / ratio),
                 {"EMP_SCALE": scale, "reciprocal_WVARMED":
                  1.0 / ratio if header_values_ok else None})
        threshold_value = product.hdus[
            "science_policy_support_I"].header.get("WTTHRESH")
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            threshold = math.nan
        ratio_mask = np.isfinite(formal) & (formal > 0.0) & \
            np.isfinite(variance_expected) & (variance_expected > 0.0) & \
            (formal >= threshold)
        recomputed_ratio = float(np.median(
            (formal * variance_expected)[ratio_mask])) \
            if np.count_nonzero(ratio_mask) else math.nan
        ratio_passed, ratio_detail = numeric_close(
            np.asarray([ratio]), np.asarray([recomputed_ratio]))
        book.add(prefix + ".WVARMED_recomputed",
                 math.isfinite(threshold) and ratio_passed,
                 {**ratio_detail, "WTTHRESH": threshold_value,
                  "valid_pixel_count": int(np.count_nonzero(ratio_mask)),
                  "recorded": ratio, "recomputed": recomputed_ratio})
        if residuals is not None:
            record_lossless_residual(
                residuals, "WVARMED_recomputed", np.asarray([ratio]),
                np.asarray([recomputed_ratio]))
        expected = signal * np.sqrt(np.maximum(formal * scale, 0.0))
        snr_left = product.array("sig2noise_I")
        snr_right = product.array("sig2noise_pixel_I")
        book.add(prefix + ".sig2noise_aliases_bitwise_equal",
                 snr_left.dtype == snr_right.dtype and
                 snr_left.shape == snr_right.shape and
                 snr_left.tobytes() == snr_right.tobytes(), None)
        if residuals is not None:
            record_lossless_residual(
                residuals, "sig2noise_alias", snr_left, snr_right,
                bitwise=True)
        for name in ("sig2noise_I", "sig2noise_pixel_I"):
            actual_standardized = internal_spatial(product.array(name))
            passed, detail = numeric_close(actual_standardized, expected)
            book.add(prefix + f".{name}.identity", passed, detail)
            if residuals is not None:
                record_lossless_residual(
                    residuals, f"{name}_identity", actual_standardized,
                    expected)
        median_rms_mask = np.isfinite(formal) & (formal >= threshold)
        realization_rms = np.asarray([
            math.sqrt(float(np.mean(noise[..., index][median_rms_mask] ** 2)))
            for index in range(REALIZATIONS)
        ]) if np.count_nonzero(median_rms_mask) else np.asarray([], dtype=float)
        recomputed_medrms = float(np.median(realization_rms)) \
            if realization_rms.size else math.nan
        variance_medrms = product.hdus["noise_variance_I"].header.get("MEDRMS")
        noise_medrms = []
        if noise_product is not None:
            noise_medrms = [noise_product.hdus[f"signal_{index}_I"].header.get(
                "MEDRMS") for index in range(REALIZATIONS)]
        try:
            medrms_values = [float(variance_medrms),
                             *(float(value) for value in noise_medrms)]
        except (TypeError, ValueError):
            medrms_values = []
        header_medrms_ok = len(medrms_values) == REALIZATIONS + 1 and \
            all(math.isfinite(value) and value > 0.0 for value in medrms_values) and \
            all(exact_float_equal(value, medrms_values[0])
                for value in medrms_values[1:])
        medrms_close, medrms_detail = numeric_close(
            np.asarray([medrms_values[0] if medrms_values else math.nan]),
            np.asarray([recomputed_medrms]))
        book.add(prefix + ".MEDRMS_headers_and_reconstruction",
                 header_medrms_ok and medrms_close,
                 {**medrms_detail, "variance_header": variance_medrms,
                  "noise_header_count": len(noise_medrms),
                  "recomputed": recomputed_medrms,
                  "valid_pixel_count": int(np.count_nonzero(median_rms_mask))})
        if residuals is not None:
            record_lossless_residual(
                residuals, "MEDRMS_recomputed",
                np.asarray(medrms_values, dtype=np.float64),
                np.full((len(medrms_values),), recomputed_medrms,
                        dtype=np.float64))
    else:
        book.add(prefix + ".formal_standardized_inventory",
                 "formal_standardized_signal_I" in names
                 and not (names & set(EMPIRICAL_PLANES)),
                 {"empirical_present": sorted(names & set(EMPIRICAL_PLANES)),
                  "formal_present": "formal_standardized_signal_I" in names})
        if "formal_standardized_signal_I" in names:
            signal = internal_spatial(product.array("signal_I"))
            weight = internal_spatial(product.array("weight_I"))
            expected = signal * np.sqrt(np.maximum(weight, 0.0))
            passed, detail = numeric_close(
                internal_spatial(product.array("formal_standardized_signal_I")), expected)
            book.add(prefix + ".formal_standardized_identity", passed, detail)
            if residuals is not None:
                record_lossless_residual(
                    residuals, "formal_standardized_identity",
                    internal_spatial(product.array(
                        "formal_standardized_signal_I")), expected)


def verify_map(record: Mapping[str, Any], case: Mapping[str, Any],
               contracts: Mapping[str, Any], mapmaking: Mapping[str, Any],
               coadd_provenance: Mapping[str, Any] | None,
               raw_authority: RawManifestAuthority,
               output: Path,
               book: CheckBook) -> tuple[dict[str, Any], Reconstruction | None]:
    scope, obsnum, array = record["scope"], record["obsnum"], record["array"]
    prefix = f"{case['id']}.{scope}.{obsnum if obsnum else 'coadd'}.{array}"
    residuals: dict[str, np.ndarray] = {
        "schema_version": np.array(
            "sci-map-001-persisted-identity-residuals-v1"),
        "case_id": np.array(case["id"]), "scope": np.array(scope),
        "obsnum": np.array(-1 if obsnum is None else int(obsnum),
                           dtype=np.int64),
        "array": np.array(array),
    }
    map_path = verify_frozen_path(record.get("map"), record.get("map_sha256"),
                                  prefix + ".map")
    noise_path = verify_frozen_path(record.get("noise"), record.get("noise_sha256"),
                                    prefix + ".noise", optional=True)
    ledger_path = verify_frozen_path(record.get("sample_ledger"),
                                     record.get("sample_ledger_sha256"),
                                     prefix + ".sample_ledger", optional=True)
    product = FitsProduct.open(map_path)
    noise_product = FitsProduct.open(noise_path) if noise_path else None
    try:
        common_id = "sci_map_common_altaz_v1" if case["mode"] == "point" \
            else "sci_map_common_celestial_v1"
        f010_id = "sci_map_naive_coadd_v1" if scope == "coadd" \
            else "sci_map_naive_observation_v1"
        checks = contracts.get("checks", {})
        if common_id not in checks or f010_id not in checks:
            die("candidate product registry lacks required SCI-MAP checks")
        apply_contract_check(product, checks[common_id], book,
                             prefix + ".common_contract", residuals)
        apply_contract_check(product, checks[f010_id], book,
                             prefix + ".f010_contract", residuals)
        statistical_wcs: list[str] = []
        if scope == "observation":
            statistical_id = ("sci_map_empirical_products_v1" if
                              case["products_enabled"] else
                              "sci_map_formal_standardized_only_v1")
            if statistical_id not in checks:
                die(f"candidate product registry lacks {statistical_id}")
            apply_contract_check(product, checks[statistical_id], book,
                                 prefix + ".statistical_contract", residuals)
            statistical_wcs = list(checks[statistical_id].get(
                "required_extnames", ()))
        required_wcs = list(COMMON_PLANES) + list(F010_COADD if scope == "coadd"
                                                  else F010_OBSERVATION) + \
            statistical_wcs
        all_wcs = {name: wcs_cards(product.hdus[name].header) for name in required_wcs
                   if name in product.hdus}
        book.add(prefix + ".whole_bundle_wcs",
                 len(all_wcs) == len(required_wcs)
                 and len({json.dumps(value, sort_keys=True, default=str)
                          for value in all_wcs.values()}) == 1,
                 {"available": sorted(all_wcs), "required": required_wcs})
        shape = product.array("signal_I").shape
        signal_header = product.hdus["signal_I"].header
        signal_unit = str(signal_header.get("UNIT", ""))
        book.add(prefix + ".signal_unit_exact",
                 signal_unit == "mJy/beam" and
                 signal_header.get("BUNIT") == "mJy/beam",
                 {"UNIT": signal_header.get("UNIT"),
                  "BUNIT": signal_header.get("BUNIT"),
                  "expected": "mJy/beam"})
        noise_cube = verify_noise_file(
            noise_product, shape, signal_header, "mJy/beam",
            book, prefix + ".noise") if noise_product else None
        validate_empirical_products(product, noise_cube,
                                    bool(case["products_enabled"]), scope,
                                    book, prefix, noise_product, residuals)

        provenance = coadd_provenance if scope == "coadd" else mapmaking
        if provenance is None:
            die(prefix + ": required provenance is absent")
        science_contract = provenance.get("science_contract", {})
        book.add(prefix + ".provenance_algorithms",
                 all(science_contract.get(key) == value
                     for key, value in SCIENCE_ALGORITHMS.items()),
                 {key: science_contract.get(key) for key in SCIENCE_ALGORITHMS})
        identity, realized = realized_map_record(provenance, scope, obsnum, array)
        identity_digest = verify_science_identity(
            identity, realized, record, product, case["mode"], book, prefix)
        verify_identity_wcs_adapter(identity, array, product, book, prefix)
        product_inventory_check(realized, product, scope, book, prefix)

        reconstruction: Reconstruction | None = None
        if scope == "observation":
            if ledger_path is None:
                book.add(prefix + ".independent_f010_reconstruction", False,
                         "SCI-MAP-001-UNITY-001-EG-INDEPENDENT-SAMPLE-LEDGER: "
                         "required sample ledger is absent",
                         evidence_gap=True)
            else:
                reconstruction = reconstruct_observation_from_ledger(
                    ledger_path, int(obsnum), array,
                    image_spatial_shape(product.array("signal_I")),
                    float(case["coverage_cut"]), raw_authority)
                ledger_identity = reconstruction.ledger_identity
                book.add(prefix + ".ledger.bundle_identity_digest",
                         ledger_identity.get("bundle_identity_digest") == identity_digest,
                         {"ledger": ledger_identity.get("bundle_identity_digest"),
                          "provenance_recomputed": identity_digest})
                book.add(prefix + ".ledger.raw_input_manifest_sha256",
                         ledger_identity.get("raw_input_manifest_sha256") ==
                         record.get("raw_input_manifest_sha256"),
                         {"ledger": ledger_identity.get("raw_input_manifest_sha256"),
                          "collection": record.get("raw_input_manifest_sha256")})
                book.add(prefix + ".ledger.raw_membership_complete",
                         ledger_identity.get("raw_membership_complete") is True,
                         {key: value for key, value in ledger_identity.items()
                          if key.startswith("raw_membership_")})
                # Some products-off observation lanes intentionally do not
                # serialize realization FITS; the ledger still reconstructs
                # the required companions used by science_valid.
                compare_reconstruction(product, noise_cube, reconstruction,
                                       book, prefix, residuals)
                if noise_cube is None and not case["products_enabled"]:
                    # Replace only the serialization evidence-gap check with a
                    # positive in-memory-companion reconstruction statement.
                    book.checks[-1] = {
                        "id": prefix + ".reconstruct.realizations",
                        "result": "pass",
                        "detail": "products-off lane: 64 required companions reconstructed "
                                  "from the independent ledger; observation realization FITS "
                                  "are intentionally absent",
                    }
                verify_thresholds(realized, reconstruction,
                                  float(case["coverage_cut"]), scope,
                                  product, book, prefix)
        digest_noise = noise_cube
        if digest_noise is None and reconstruction is not None:
            # Products-off observation FITS intentionally omit realizations,
            # but the approved processed-input ledger is an exact authority
            # for the same 64 in-memory planes included in the raw digest.
            digest_noise = reconstruction.noise
        if digest_noise is None:
            book.add(prefix + ".raw_parent_digest_recomputed", False,
                     "SCI-MAP-001-UNITY-001-EG-RAW-PARENT-REALIZATIONS: "
                     "neither serialized realizations nor an approved exact "
                     "processed-input ledger is available",
                     evidence_gap=True)
        else:
            recomputed_raw_parent = recompute_raw_parent_digest(
                identity, realized, product, digest_noise, scope, provenance)
            raw_parent_passed = book.add(
                prefix + ".raw_parent_digest_recomputed",
                realized.get("raw_parent_digest") == recomputed_raw_parent,
                {"recorded": realized.get("raw_parent_digest"),
                 "recomputed": recomputed_raw_parent,
                 "realization_authority": ("serialized_noise_fits" if
                     noise_cube is not None else
                     "approved_independent_sample_ledger")})
            if raw_parent_passed and reconstruction is not None:
                reconstruction.verified_raw_parent_digest = recomputed_raw_parent
        residual_path = output / (
            f"{case['id']}-{scope}-"
            f"{obsnum if obsnum is not None else 'coadd'}-{array}-"
            "persisted-identity-residuals.npz")
        write_npz_new(residual_path, **residuals)
        residual_detail = {
            "path": str(residual_path), "sha256": sha256(residual_path),
            "array_keys": sorted(residuals),
        }
        book.add(prefix + ".persisted_identity.lossless_residuals",
                 len(residuals) > 5, residual_detail)
        summary = {
            "case_id": case["id"], "scope": scope, "obsnum": obsnum,
            "array": array, "map": str(map_path), "map_sha256": sha256(map_path),
            "noise": str(noise_path) if noise_path else None,
            "noise_sha256": sha256(noise_path) if noise_path else None,
            "extensions": sorted(product.hdus),
            "shape": list(shape),
            "wcs": wcs_cards(product.hdus["signal_I"].header),
            "ledger_identity": reconstruction.ledger_identity if reconstruction else None,
            "persisted_identity_residuals": residual_detail,
        }
        return summary, reconstruction
    finally:
        product.close()
        if noise_product:
            noise_product.close()


def load_map_arrays(record: Mapping[str, Any]) -> tuple[dict[str, np.ndarray],
                                                        np.ndarray | None,
                                                        fits.Header]:
    product = FitsProduct.open(Path(record["map"]))
    noise_product = FitsProduct.open(Path(record["noise"])) if record.get("noise") else None
    try:
        plane_names = dict.fromkeys((
            *COMMON_PLANES, *F010_COADD, *EMPIRICAL_PLANES,
            "formal_standardized_signal_I",
        ))
        arrays = {name: internal_spatial(product.array(name)).copy()
                  for name in plane_names
                  if name in product.hdus}
        noise = None
        if noise_product:
            names = {int(match.group(1)): name for name in noise_product.hdus
                     if (match := re.fullmatch(r"signal_([0-9]+)_I", name))}
            if set(names) == set(range(REALIZATIONS)):
                noise = np.stack([internal_spatial(noise_product.array(names[index])).copy()
                                  for index in range(REALIZATIONS)], axis=-1)
        return arrays, noise, product.hdus["signal_I"].header.copy()
    finally:
        product.close()
        if noise_product:
            noise_product.close()


def load_primary_header(record: Mapping[str, Any]) -> fits.Header:
    product = FitsProduct.open(Path(record["map"]))
    try:
        return product.hdul[0].header.copy()
    finally:
        product.close()


def centered_wcs_check(observation_header: fits.Header, coadd_header: fits.Header,
                       observation_shape: tuple[int, int], coadd_shape: tuple[int, int],
                       delta_row: int, delta_col: int) -> tuple[bool, dict[str, Any]]:
    rows, cols = observation_shape
    coadd_rows, coadd_cols = coadd_shape
    even_centered = coadd_rows >= rows and coadd_cols >= cols and \
        (coadd_rows - rows) % 2 == 0 and (coadd_cols - cols) % 2 == 0 and \
        delta_row == (coadd_rows - rows) // 2 and \
        delta_col == (coadd_cols - cols) // 2
    crpix = exact_float_equal(float(coadd_header.get("CRPIX1", math.nan)),
                              float(observation_header.get("CRPIX1", math.nan)) + delta_col) and \
        exact_float_equal(float(coadd_header.get("CRPIX2", math.nan)),
                          float(observation_header.get("CRPIX2", math.nan)) + delta_row)
    excluded = {"CRPIX1", "CRPIX2", "NAXIS1", "NAXIS2"}
    obs_cards = {key: value for key, value in wcs_cards(observation_header).items()
                 if key not in excluded}
    coadd_cards = {key: value for key, value in wcs_cards(coadd_header).items()
                   if key not in excluded}
    card_match = obs_cards == coadd_cards
    transform_match = False
    transform_detail: Any = None
    checked_pixels = 0
    try:
        obs_wcs = WCS(observation_header)
        coadd_wcs = WCS(coadd_header)
        # FITS pixel axis 1 is column, axis 2 is row. Fill the singleton
        # frequency/Stokes coordinates at their reference pixels.
        max_delta = 0.0
        transform_match = True
        chunk_rows = max(1, min(rows, 262144 // max(cols, 1)))
        for row_start in range(0, rows, chunk_rows):
            row_stop = min(rows, row_start + chunk_rows)
            row_grid, col_grid = np.mgrid[row_start:row_stop, 0:cols]
            obs_points_array = np.empty((row_grid.size, obs_wcs.pixel_n_dim), dtype=float)
            obs_points_array[:, 0] = col_grid.ravel()
            obs_points_array[:, 1] = row_grid.ravel()
            for axis in range(3, obs_wcs.pixel_n_dim + 1):
                obs_points_array[:, axis - 1] = float(
                    observation_header.get(f"CRPIX{axis}", 1.0)) - 1.0
            coadd_points = obs_points_array.copy()
            coadd_points[:, 0] += delta_col
            coadd_points[:, 1] += delta_row
            obs_world = obs_wcs.all_pix2world(obs_points_array, 0)
            coadd_world = coadd_wcs.all_pix2world(coadd_points, 0)
            difference = obs_world - coadd_world
            if difference.shape[1] >= 1 and str(observation_header.get("CTYPE1", "")).startswith("RA"):
                difference[:, 0] = (difference[:, 0] + 180.0) % 360.0 - 180.0
            finite = np.isfinite(difference)
            if not np.array_equal(np.isfinite(obs_world), np.isfinite(coadd_world)):
                transform_match = False
            if finite.any():
                chunk_max = float(np.max(np.abs(difference[finite])))
                max_delta = max(max_delta, chunk_max)
                if chunk_max > 1.0e-12:
                    transform_match = False
            checked_pixels += row_grid.size
        transform_detail = max_delta
    except Exception as exc:
        transform_detail = str(exc)
    return bool(even_centered and crpix and card_match and transform_match), {
        "even_centered_embedding": even_centered,
        "crpix_shift_exact": crpix,
        "other_wcs_cards_exact": card_match,
        "world_transform_match": transform_match,
        "max_world_delta": transform_detail,
        "checked_observation_pixels": checked_pixels,
        "delta_row": delta_row, "delta_col": delta_col,
        "observation_shape": list(observation_shape),
        "coadd_shape": list(coadd_shape),
    }


def observation_identity_from_coadd(
        common_identity: Mapping[str, Any], admission: Mapping[str, Any],
        prefix: str) -> dict[str, Any]:
    identity = copy.deepcopy(dict(common_identity))
    observation_shape = admission.get("observation_shape")
    embedding = admission.get("embedding")
    if not isinstance(observation_shape, Mapping) or not isinstance(embedding, Mapping):
        die(f"{prefix}: admission shape/embedding is malformed")
    identity["shape"] = {
        "rows": int(observation_shape.get("rows")),
        "cols": int(observation_shape.get("cols")),
    }
    wcs = identity.get("wcs")
    if not isinstance(wcs, dict):
        die(f"{prefix}: common identity WCS is malformed")
    reference = wcs.get("reference_pixel")
    if not isinstance(reference, list) or len(reference) != 2:
        die(f"{prefix}: common identity reference pixel is malformed")
    coadd_col = exact_float_node(reference[0], prefix + ".coadd_reference_col")
    coadd_row = exact_float_node(reference[1], prefix + ".coadd_reference_row")
    wcs["reference_pixel"] = [
        exact_node(coadd_col - int(embedding.get("delta_col"))),
        exact_node(coadd_row - int(embedding.get("delta_row"))),
    ]
    identity.pop("identity_digest", None)
    identity["identity_digest"] = recompute_bundle_identity_digest(identity)
    return identity


def verify_complete_admission(
        common_identity: Mapping[str, Any], admission: Mapping[str, Any],
        reconstruction: Reconstruction, coadd_shape: tuple[int, int],
        expected_raw_parents: Sequence[str], observation_exposure_seconds: float,
        prefix: str, book: CheckBook) -> bool:
    expected_identity = observation_identity_from_coadd(
        common_identity, admission, prefix)
    expected_digest = expected_identity["identity_digest"]
    embedding = admission["embedding"]
    observation_shape = admission["observation_shape"]
    recorded_coadd_shape = admission.get("coadd_shape")
    policies = admission.get("policies")
    identity_policies = expected_identity.get("policies")
    slots = expected_identity.get("ordered_slots")
    raw_parents = admission.get("observation_raw_parent_digests")
    counts = admission.get("numerically_contributing_pixel_count")
    if not isinstance(recorded_coadd_shape, Mapping) or \
            not isinstance(policies, Mapping) or \
            not isinstance(identity_policies, Mapping) or \
            not isinstance(slots, list) or not isinstance(raw_parents, list) or \
            not isinstance(counts, list):
        book.add(prefix + ".complete_identity", False,
                 "admission nested identity facts are malformed")
        return False
    checks = []
    checks.append(book.add(
        prefix + ".bundle_digest_before_arithmetic",
        admission.get("admitted_bundle_identity") == expected_digest ==
        reconstruction.ledger_identity.get("bundle_identity_digest"),
        {"admission": admission.get("admitted_bundle_identity"),
         "derived_observation_identity": expected_digest,
         "ledger": reconstruction.ledger_identity.get("bundle_identity_digest")}))
    checks.append(book.add(
        prefix + ".shape_before_arithmetic",
        (int(observation_shape.get("rows", -1)),
         int(observation_shape.get("cols", -1))) ==
        tuple(reconstruction.planes["signal_I"].shape)
        and (int(recorded_coadd_shape.get("rows", -1)),
             int(recorded_coadd_shape.get("cols", -1))) == coadd_shape,
        {"admission_observation_shape": dict(observation_shape),
         "ledger_shape": list(reconstruction.planes["signal_I"].shape),
         "admission_coadd_shape": dict(recorded_coadd_shape),
         "actual_coadd_shape": list(coadd_shape)}))
    checks.append(book.add(
        prefix + ".identity_facts_before_arithmetic",
        admission.get("response_identity") ==
        expected_identity.get("response_identity")
        and admission.get("ordered_map_count") == len(slots) == len(ARRAYS)
        and admission.get("coefficient_stage") ==
        "post-observation-normalization-no-empirical-rescale"
        and embedding.get("registration_identity") ==
        "centered-integer-common-grid-embedding-v1"
        and embedding.get("centering_identity") == "L-identity-v1",
        {"response_identity": admission.get("response_identity"),
         "expected_response_identity": expected_identity.get("response_identity"),
         "ordered_map_count": admission.get("ordered_map_count"),
         "coefficient_stage": admission.get("coefficient_stage"),
         "registration_identity": embedding.get("registration_identity"),
         "centering_identity": embedding.get("centering_identity")}))
    checks.append(book.add(
        prefix + ".policies_before_arithmetic",
        policies.get("normalization_support") ==
        identity_policies.get("normalization_support")
        and policies.get("science_policy_support") ==
        identity_policies.get("science_policy_support")
        and policies.get("validity") == identity_policies.get("validity")
        and policies.get("nonfinite") == identity_policies.get("nonfinite"),
        {"admission": dict(policies), "identity": dict(identity_policies)}))
    checks.append(book.add(
        prefix + ".slot_parent_vectors_before_arithmetic",
        len(counts) == len(raw_parents) == len(slots) == len(ARRAYS)
        and all(canonical_digest_string(value) for value in raw_parents)
        and list(raw_parents) == list(expected_raw_parents),
        {"count_vector_length": len(counts),
         "raw_parent_vector": raw_parents,
         "independently_verified_observation_raw_parents":
             list(expected_raw_parents),
         "slot_count": len(slots)}))
    recorded_exposure = exact_float_node(
        admission.get("observation_exposure_seconds"),
        prefix + ".observation_exposure_seconds")
    checks.append(book.add(
        prefix + ".observation_exposure_before_arithmetic",
        math.isfinite(observation_exposure_seconds)
        and observation_exposure_seconds > 0.0
        and exact_float_equal(recorded_exposure, observation_exposure_seconds),
        {"admission_seconds": recorded_exposure,
         "observation_primary_EXPTIME": observation_exposure_seconds}))
    return all(checks)


def reconstruct_coadd(output: Path, case: Mapping[str, Any],
                      map_records: Sequence[Mapping[str, Any]],
                      observation_reconstructions: Mapping[tuple[int, str], Reconstruction],
                      coadd_provenance: Mapping[str, Any], array: str,
                      book: CheckBook) -> Reconstruction | None:
    case_id = case["id"]
    prefix = f"{case_id}.coadd.coadd.{array}"
    coadd_record = next((record for record in map_records
                         if record["scope"] == "coadd" and record["array"] == array), None)
    if coadd_record is None:
        book.add(prefix + ".reconstruction_inventory", False,
                 "coadd map record absent")
        return None
    actual, actual_noise, coadd_header = load_map_arrays(coadd_record)
    if actual_noise is None:
        book.add(prefix + ".reconstruction_noise_inventory", False,
                 "coadd noise realizations absent")
        return None
    coadd_shape = actual["signal_I"].shape
    coadd_primary = load_primary_header(coadd_record)
    try:
        coadd_exposure_seconds = float(coadd_primary["EXPTIME"])
    except (KeyError, TypeError, ValueError):
        coadd_exposure_seconds = math.nan
    state = coadd_provenance.get("observation_resolved", {})
    common_identity_node = state.get("common_identity", {})
    common_identity = common_identity_node.get("value") \
        if isinstance(common_identity_node, Mapping) else None
    if not isinstance(common_identity, Mapping):
        book.add(prefix + ".common_identity", False,
                 "coadd common typed identity is absent")
        return None
    admissions = state.get("admissions")
    if not isinstance(admissions, list):
        book.add(prefix + ".admissions", False, "coadd admissions absent")
        return None
    expected_obs = case["expected_observations"]
    if [int(item.get("observation_id", -1)) for item in admissions
        if isinstance(item, Mapping)] != expected_obs:
        book.add(prefix + ".admission_order", False,
                 {"actual": [item.get("observation_id") for item in admissions],
                  "expected": expected_obs})
        return None
    book.add(prefix + ".admission_order", True, expected_obs)

    q = np.zeros(coadd_shape, dtype=np.float64)
    n = np.zeros(coadd_shape, dtype=np.float64)
    k = np.zeros(coadd_shape, dtype=np.float64)
    noise_num = np.zeros((*coadd_shape, REALIZATIONS), dtype=np.float64)
    geometric = np.zeros(coadd_shape, dtype=np.int64)
    contributing = np.zeros(coadd_shape, dtype=np.int64)
    observation_count = np.zeros(coadd_shape, dtype=np.int64)
    eligible = np.zeros(coadd_shape, dtype=np.float64)
    retained = np.zeros(coadd_shape, dtype=np.float64)
    per_scan_abs = {name: np.zeros(coadd_shape, dtype=np.longdouble)
                    for name in ("signal_I", "weight_I", "kernel_I",
                                 "retained_exposure_I",
                                 "upstream_eligible_exposure_I")}
    observation_exposure_sum = 0.0

    for admission, obsnum in zip(admissions, expected_obs):
        reconstruction = observation_reconstructions.get((obsnum, array))
        if reconstruction is None:
            book.add(prefix + f".observation_{obsnum}_ledger", False,
                     "independent observation reconstruction unavailable",
                     evidence_gap=True)
            return None
        expected_raw_parents = [
            observation_reconstructions[(obsnum, sibling)].verified_raw_parent_digest
            if (obsnum, sibling) in observation_reconstructions else None
            for sibling in ARRAYS
        ]
        if any(value is None for value in expected_raw_parents):
            book.add(prefix + f".observation_{obsnum}.raw_parent_authority",
                     False,
                     "SCI-MAP-001-UNITY-001-EG-OBSERVATION-RAW-PARENT: "
                     "an observation slot lacks an independently verified raw-parent digest",
                     evidence_gap=True)
            return None
        obs_record = next(record for record in map_records
                          if record["scope"] == "observation"
                          and record["obsnum"] == obsnum
                          and record["array"] == array)
        obs_primary = load_primary_header(obs_record)
        try:
            observation_exposure_seconds = float(obs_primary["EXPTIME"])
        except (KeyError, TypeError, ValueError):
            observation_exposure_seconds = math.nan
        if not verify_complete_admission(
                common_identity, admission, reconstruction, coadd_shape,
                [str(value) for value in expected_raw_parents],
                observation_exposure_seconds,
                prefix + f".observation_{obsnum}.admission", book):
            book.add(prefix + f".observation_{obsnum}.pre_arithmetic_gate",
                     False,
                     "complete typed admission identity failed before arithmetic")
            return None
        observation_exposure_sum += observation_exposure_seconds
        delta = admission.get("embedding", {})
        try:
            delta_row = int(delta["delta_row"])
            delta_col = int(delta["delta_col"])
        except (KeyError, TypeError, ValueError):
            book.add(prefix + f".observation_{obsnum}_embedding", False,
                     "integer embedding offsets absent")
            return None
        obs_arrays, _, obs_header = load_map_arrays(obs_record)
        obs_shape = obs_arrays["signal_I"].shape
        passed, detail = centered_wcs_check(obs_header, coadd_header, obs_shape,
                                            coadd_shape, delta_row, delta_col)
        book.add(prefix + f".observation_{obsnum}.centered_wcs", passed, detail)
        book.add(prefix + f".observation_{obsnum}.admission_identity",
                 delta.get("registration_identity") ==
                 "centered-integer-common-grid-embedding-v1"
                 and delta.get("centering_identity") == "L-identity-v1"
                 and int(admission.get("ordered_map_count", -1)) == len(ARRAYS),
                 {"registration_identity": delta.get("registration_identity"),
                  "centering_identity": delta.get("centering_identity"),
                  "ordered_map_count": admission.get("ordered_map_count")})
        row_slice = slice(delta_row, delta_row + obs_shape[0])
        col_slice = slice(delta_col, delta_col + obs_shape[1])
        planes = reconstruction.planes
        norm = planes["normalization_support_I"].astype(bool)
        geometric[row_slice, col_slice] += planes["geometric_hits_I"]
        eligible[row_slice, col_slice] += planes["upstream_eligible_exposure_I"]
        supported_weight = np.where(norm, planes["weight_I"], 0.0)
        q[row_slice, col_slice] += supported_weight
        n[row_slice, col_slice] += np.where(norm,
            planes["weight_I"] * planes["signal_I"], 0.0)
        k[row_slice, col_slice] += np.where(norm,
            planes["weight_I"] * planes["kernel_I"], 0.0)
        noise_num[row_slice, col_slice] += np.where(
            norm[..., None], planes["weight_I"][..., None] * reconstruction.noise,
            0.0)
        contributing[row_slice, col_slice] += np.where(
            norm, planes["contributing_hits_I"], 0)
        observation_count[row_slice, col_slice] += norm.astype(np.int64)
        retained[row_slice, col_slice] += np.where(
            norm, planes["retained_exposure_I"], 0.0)
        for name, values in reconstruction.per_scan_sum_abs.items():
            per_scan_abs[name][row_slice, col_slice] += values

        counts = admission.get("numerically_contributing_pixel_count")
        if isinstance(counts, list) and len(counts) == len(ARRAYS):
            actual_count = int(np.count_nonzero(norm))
            book.add(prefix + f".observation_{obsnum}.admitted_pixel_count",
                     int(counts[ARRAYS.index(array)]) == actual_count,
                     {"recorded": counts[ARRAYS.index(array)],
                      "reconstructed": actual_count})
        else:
            book.add(prefix + f".observation_{obsnum}.admitted_pixel_count", False,
                     "admission count vector malformed")

    book.add(prefix + ".coadd_primary_exposure_sum",
             math.isfinite(coadd_exposure_seconds)
             and coadd_exposure_seconds > 0.0
             and exact_float_equal(coadd_exposure_seconds,
                                   observation_exposure_sum),
             {"coadd_primary_EXPTIME": coadd_exposure_seconds,
              "ordered_observation_EXPTIME_sum": observation_exposure_sum,
              "observation_order": expected_obs})

    normalization = threshold_selection(q, float(case["coverage_cut"]) / 10.0)
    norm = np.isfinite(q) & (q > 0.0) & (q >= normalization["threshold"])
    final_weight = np.where(norm, q, 0.0)
    safe_q = np.where(q > 0.0, q, 1.0)
    final_signal = np.where(norm, n / safe_q, 0.0)
    final_kernel = np.where(norm, k / safe_q, 0.0)
    final_noise = np.where(norm[..., None], noise_num / safe_q[..., None], 0.0)
    retained = np.where(norm, retained, 0.0)
    policy_selection = threshold_selection(final_weight, float(case["coverage_cut"]))
    policy = np.isfinite(final_weight) & (final_weight > 0.0) & \
        (final_weight >= policy_selection["threshold"])
    finite = np.isfinite(final_signal) & np.isfinite(final_weight) & \
        (final_weight > 0.0) & np.isfinite(final_kernel) & \
        np.all(np.isfinite(final_noise), axis=-1)
    valid = norm & policy & finite
    planes = {
        "signal_I": final_signal, "weight_I": final_weight,
        "kernel_I": final_kernel, "geometric_hits_I": geometric,
        "contributing_hits_I": contributing,
        "coadd_observation_count_I": observation_count,
        "upstream_eligible_exposure_I": eligible,
        "retained_exposure_I": retained,
        "normalization_support_I": norm.astype(np.uint8),
        "science_policy_support_I": policy.astype(np.uint8),
        "science_valid_I": valid.astype(np.uint8),
        "coverage_I": retained.copy(),
        "coverage_bool_I": policy.astype(np.uint8),
    }
    result = Reconstruction(
        planes=planes, noise=final_noise, normalization=normalization,
        science_policy=policy_selection, per_scan_sum_abs=per_scan_abs,
        ledger_identity={"source": "independently reconstructed observation products",
                         "observations": expected_obs, "array": array},
        raw_numerators={
            "signal_numerator": n.copy(), "weight_I": q.copy(),
            "kernel_numerator": k.copy(), "retained_exposure_I": retained.copy(),
            "upstream_eligible_exposure_I": eligible.copy(),
        })
    for name, expected in planes.items():
        if name in INTEGER_PLANES:
            passed, detail = integer_equal(actual[name], expected)
        else:
            passed, detail = numeric_close(actual[name], expected)
        book.add(prefix + f".recombine.{name}", passed, detail)
    passed, detail = numeric_close(actual_noise, final_noise)
    book.add(prefix + ".recombine.realizations", passed, detail)
    residual_arrays: dict[str, np.ndarray] = {
        "schema_version": np.array("sci-map-001-coadd-recombination-residuals-v1"),
        "case_id": np.array(case_id), "array": np.array(array),
    }
    for name, expected in planes.items():
        record_lossless_residual(
            residual_arrays, f"recombine_{name}", actual[name], expected,
            bitwise=name in INTEGER_PLANES)
    record_lossless_residual(
        residual_arrays, "recombine_noise_realizations", actual_noise,
        final_noise)
    residual_path = output / f"{case_id}-coadd-{array}-recombination-residuals.npz"
    write_npz_new(residual_path, **residual_arrays)
    book.add(prefix + ".lossless_residuals", True,
             {"path": str(residual_path), "sha256": sha256(residual_path),
              "array_keys": sorted(residual_arrays)})

    identity, realized = realized_map_record(coadd_provenance, "coadd", None, array)
    del identity
    # Use a lightweight wrapper so the same threshold checker consumes the
    # independently reconstructed coadd coefficient stages.
    map_product = FitsProduct.open(Path(coadd_record["map"]))
    try:
        verify_thresholds(realized, result, float(case["coverage_cut"]),
                          "coadd", map_product, book, prefix)
    finally:
        map_product.close()
    return result


def map_record_index(case_record: Mapping[str, Any]) -> dict[tuple[str, Any, str], Mapping[str, Any]]:
    records = case_record.get("maps")
    if not isinstance(records, list):
        die(f"{case_record.get('case_id')}: map record list is absent")
    result = {}
    for record in records:
        if not isinstance(record, Mapping):
            die("analysis map record is malformed")
        key = (record.get("scope"), record.get("obsnum"), record.get("array"))
        if key in result:
            die(f"analysis map record is repeated: {key}")
        result[key] = record
    return result


def successor_diagnostic_entries(
        contracts: Mapping[str, Any], case: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = contracts.get("contracts")
    if not isinstance(rows, list):
        die("product-contract registry contract list is malformed")
    by_id = {row.get("contract_id"): row for row in rows
             if isinstance(row, Mapping)}
    successor_id = ("sci-map-001-point-products-v1" if case["mode"] == "point"
                    else "sci-map-001-science-products-v1")
    successor = by_id.get(successor_id)
    if not isinstance(successor, Mapping):
        die(f"candidate registry lacks {successor_id}")
    base = by_id.get(successor.get("extends_contract_id"))
    entries = base.get("entries") if isinstance(base, Mapping) else None
    if not isinstance(entries, list):
        die(f"{successor_id} base contract entries are malformed")
    observation_ids = (
        {"raw-histogram", "raw-psd", "raw-map-diagnostics"}
        if case["mode"] == "point" else
        {"observation-histogram", "observation-psd",
         "observation-map-diagnostics"})
    coadd_ids = ({"coadded-histogram", "coadded-psd",
                  "coadded-map-diagnostics"} if case["coadd"] else set())
    selected_ids = observation_ids | coadd_ids
    selected = [dict(row) for row in entries
                if isinstance(row, Mapping) and row.get("entry_id") in selected_ids]
    if {row.get("entry_id") for row in selected} != selected_ids or \
            len(selected) != len(selected_ids):
        die(f"{successor_id} diagnostic entry authority is incomplete")
    expected_families = {"map_histogram", "map_psd", "map_diagnostics"}
    for row in selected:
        pattern = row.get("pattern")
        if row.get("classification") not in ("required", "config_conditional") or \
                row.get("family_id") not in expected_families or \
                not isinstance(pattern, str) or Path(pattern).is_absolute() or \
                ".." in Path(pattern).parts:
            die(f"{successor_id} diagnostic entry is not fail-closed")
    return sorted(selected, key=lambda row: str(row["entry_id"]))


def verify_diagnostic_families(
        reduction_root: Path, case: Mapping[str, Any],
        contracts: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
        book: CheckBook) -> dict[Path, str]:
    entries = successor_diagnostic_entries(contracts, case)
    classified: dict[Path, str] = {}
    for row in entries:
        scopes = case["expected_observations"] \
            if row["scope"] == "per_observation" else (None,)
        for obsnum in scopes:
            try:
                pattern = str(row["pattern"]).format(obs=obsnum)
            except (KeyError, ValueError) as exc:
                raise EvidenceError(
                    f"{case['id']}: diagnostic pattern has unresolved fields") from exc
            if "{" in pattern or "}" in pattern:
                die(f"{case['id']}: diagnostic pattern remains unresolved")
            matches = sorted(path.resolve() for path in reduction_root.glob(pattern)
                             if path.is_file() and not path.is_symlink())
            check_id = (f"{case['id']}.diagnostic.{row['entry_id']}."
                        f"{obsnum if obsnum is not None else 'coadd'}")
            book.add(check_id, len(matches) == 1,
                     {"authority_pattern": pattern,
                      "family_id": row["family_id"],
                      "matches": [str(path) for path in matches]})
            if len(matches) == 1:
                if matches[0] in classified:
                    die(f"{case['id']}: one diagnostic satisfies multiple entries")
                classified[matches[0]] = str(row["family_id"])
    physical_diagnostics = {
        Path(str(item["path"])).resolve()
        for item in inventory
        if re.fullmatch(r".*_(?:hist|psd|mapdiag)(?:_filtered)?\.nc",
                        Path(str(item["path"])).name)
    }
    book.add(f"{case['id']}.diagnostic.closed_world_inventory",
             physical_diagnostics == set(classified),
             {"unclassified": sorted(str(path) for path in
                                     physical_diagnostics - set(classified)),
              "expected_missing": sorted(str(path) for path in
                                         set(classified) - physical_diagnostics),
              "classified": {str(path): family
                             for path, family in sorted(
                                 classified.items(), key=lambda item: str(item[0]))}})
    return classified


def compare_map_records(left_record: Mapping[str, Any], right_record: Mapping[str, Any],
                        plane_names: Sequence[str], compare_noise: bool,
                        output: Path, book: CheckBook, prefix: str) -> None:
    residuals: dict[str, np.ndarray] = {
        "schema_version": np.array("sci-map-001-map-comparison-residuals-v1"),
        "comparison_id": np.array(prefix),
        "left_map_sha256": np.array(str(left_record.get("map_sha256"))),
        "right_map_sha256": np.array(str(right_record.get("map_sha256"))),
    }
    left_product = FitsProduct.open(Path(left_record["map"]))
    right_product = FitsProduct.open(Path(right_record["map"]))
    try:
        left_names = set(left_product.hdus)
        right_names = set(right_product.hdus)
        book.add(prefix + ".hdu_inventory_exact",
                 left_names == right_names,
                 {"left_only": sorted(left_names - right_names),
                  "right_only": sorted(right_names - left_names)})
        required = set(plane_names)
        book.add(prefix + ".required_numeric_plane_inventory",
                 required.issubset(left_names) and required.issubset(right_names),
                 {"left_missing": sorted(required - left_names),
                  "right_missing": sorted(required - right_names)})
        for name in sorted(left_names & right_names):
            left_hdu = left_product.hdus[name]
            right_hdu = right_product.hdus[name]
            left_image = isinstance(left_hdu, (fits.ImageHDU, fits.CompImageHDU))
            right_image = isinstance(right_hdu, (fits.ImageHDU, fits.CompImageHDU))
            classification_detail = {
                "left_hdu_class": type(left_hdu).__name__,
                "right_hdu_class": type(right_hdu).__name__,
                "left_is_image": left_image, "right_is_image": right_image,
            }
            if left_image != right_image:
                book.add(prefix + f".{name}.hdu_classification", False,
                         classification_detail)
                continue
            if not left_image:
                book.add_not_applicable(
                    prefix + f".{name}.non_image_hdu",
                    {**classification_detail,
                     "classification": "common_non_image_hdu_no_numeric_plane_claim"})
                continue
            left_data = left_hdu.data
            right_data = right_hdu.data
            if left_data is None or right_data is None:
                book.add(prefix + f".{name}.image_data_present", False,
                         {"left": left_data is not None,
                          "right": right_data is not None})
                continue
            left_array = np.asarray(left_data)
            right_array = np.asarray(right_data)
            dtype_shape_exact = left_array.dtype == right_array.dtype and \
                left_array.shape == right_array.shape
            book.add(prefix + f".{name}.dtype_shape_exact", dtype_shape_exact,
                     {"left_dtype": str(left_array.dtype),
                      "right_dtype": str(right_array.dtype),
                      "left_shape": list(left_array.shape),
                      "right_shape": list(right_array.shape)})
            book.add(prefix + f".{name}.wcs_exact",
                     wcs_cards(left_hdu.header) == wcs_cards(right_hdu.header),
                     {"left": wcs_cards(left_hdu.header),
                      "right": wcs_cards(right_hdu.header)})
            if left_array.dtype.kind in "iub" and right_array.dtype.kind in "iub":
                passed, detail = integer_equal(left_array, right_array)
                book.add(prefix + f".{name}.all_numeric_pixels_integer_exact",
                         dtype_shape_exact and passed, detail)
                record_lossless_residual(
                    residuals, f"map_hdu_{name}", left_array, right_array,
                    bitwise=True)
            elif left_array.dtype.kind == "f" and right_array.dtype.kind == "f":
                passed, detail = numeric_close(left_array, right_array)
                book.add(prefix + f".{name}.all_numeric_pixels_registered",
                         dtype_shape_exact and passed, detail)
                record_lossless_residual(
                    residuals, f"map_hdu_{name}", left_array, right_array)
            else:
                book.add_not_applicable(
                    prefix + f".{name}.nonnumeric_image_hdu",
                    {**classification_detail,
                     "left_dtype": str(left_array.dtype),
                     "right_dtype": str(right_array.dtype),
                     "classification": "common_image_hdu_non_numeric"})
        left_header = left_product.hdus["signal_I"].header.copy()
        right_header = right_product.hdus["signal_I"].header.copy()
    finally:
        left_product.close()
        right_product.close()
    _, left_noise, _ = load_map_arrays(left_record)
    _, right_noise, _ = load_map_arrays(right_record)
    book.add(prefix + ".wcs", wcs_cards(left_header) == wcs_cards(right_header), None)
    if compare_noise:
        if left_noise is None or right_noise is None:
            book.add(prefix + ".realizations", False,
                     {"left": left_noise is not None, "right": right_noise is not None})
        else:
            passed, detail = numeric_close(left_noise, right_noise)
            book.add(prefix + ".realizations", passed, detail)
            left_noise_product = FitsProduct.open(Path(left_record["noise"]))
            right_noise_product = FitsProduct.open(Path(right_record["noise"]))
            try:
                book.add(prefix + ".noise_hdu_inventory_exact",
                         set(left_noise_product.hdus) == set(right_noise_product.hdus),
                         {"left_only": sorted(set(left_noise_product.hdus) -
                                              set(right_noise_product.hdus)),
                          "right_only": sorted(set(right_noise_product.hdus) -
                                               set(left_noise_product.hdus))})
                for name in sorted(set(left_noise_product.hdus) &
                                   set(right_noise_product.hdus)):
                    left_hdu = left_noise_product.hdus[name]
                    right_hdu = right_noise_product.hdus[name]
                    if not isinstance(left_hdu, (fits.ImageHDU, fits.CompImageHDU)) or \
                            not isinstance(right_hdu, (fits.ImageHDU,
                                                       fits.CompImageHDU)) or \
                            left_hdu.data is None or right_hdu.data is None:
                        continue
                    left_array = np.asarray(left_hdu.data)
                    right_array = np.asarray(right_hdu.data)
                    dtype_shape_exact = left_array.dtype == right_array.dtype and \
                        left_array.shape == right_array.shape
                    if left_array.dtype.kind in "iub" and \
                            right_array.dtype.kind in "iub":
                        passed, detail = integer_equal(left_array, right_array)
                        record_lossless_residual(
                            residuals, f"noise_hdu_{name}", left_array,
                            right_array, bitwise=True)
                    elif left_array.dtype.kind == "f" and \
                            right_array.dtype.kind == "f":
                        passed, detail = numeric_close(left_array, right_array)
                        record_lossless_residual(
                            residuals, f"noise_hdu_{name}", left_array,
                            right_array)
                    else:
                        continue
                    book.add(prefix + f".noise.{name}.all_numeric_pixels",
                             dtype_shape_exact and passed, detail)
                    book.add(prefix + f".noise.{name}.wcs_exact",
                             wcs_cards(left_hdu.header) ==
                             wcs_cards(right_hdu.header), None)
            finally:
                left_noise_product.close()
                right_noise_product.close()
    residual_path = output / (residual_key(prefix) + "-comparison-residuals.npz")
    write_npz_new(residual_path, **residuals)
    book.add(prefix + ".lossless_residuals", len(residuals) > 4,
             {"path": str(residual_path), "sha256": sha256(residual_path),
              "array_keys": sorted(residuals)})


def compare_cases(inputs_cases: Mapping[str, Mapping[str, Any]], output: Path,
                  book: CheckBook) -> None:
    pairs = (("P-SEQ", "P-OMP", "P"),
             ("S-C-SEQ", "S-C-OMP", "S-C"),
             ("S-E-SEQ", "S-E-OMP", "S-E"))
    for left_id, right_id, label in pairs:
        left, right = map_record_index(inputs_cases[left_id]), map_record_index(inputs_cases[right_id])
        book.add(f"compare.{label}.inventory", set(left) == set(right),
                 {"left_only": [str(key) for key in set(left) - set(right)],
                  "right_only": [str(key) for key in set(right) - set(left)]})
        for key in sorted(set(left) & set(right), key=str):
            scope = key[0]
            statistical = (EMPIRICAL_PLANES if scope == "observation" and
                           EXPECTED_CASES[left_id]["products_enabled"] else
                           (("formal_standardized_signal_I",)
                            if scope == "observation" else ()))
            planes = (*COMMON_PLANES,
                      *(F010_COADD if scope == "coadd" else F010_OBSERVATION),
                      *statistical)
            compare_map_records(left[key], right[key], planes,
                                bool(left[key].get("noise")), output, book,
                                f"compare.{label}.{scope}.{key[1]}.{key[2]}")
            if scope == "observation":
                book.add(f"compare.{label}.{scope}.{key[1]}.{key[2]}.ledger_digest",
                         left[key].get("sample_ledger_sha256") is not None
                         and left[key].get("sample_ledger_sha256") ==
                         right[key].get("sample_ledger_sha256"),
                         {"left": left[key].get("sample_ledger_sha256"),
                          "right": right[key].get("sample_ledger_sha256")})

    # Repaired S-X is an ordinary successful products+coadd composition. Its
    # observation products must match S-E-SEQ and its coadd must match S-C-SEQ.
    sx = map_record_index(inputs_cases["S-X-SEQ"])
    se = map_record_index(inputs_cases["S-E-SEQ"])
    sc = map_record_index(inputs_cases["S-C-SEQ"])
    for key, record in sx.items():
        peer = se.get(key) if key[0] == "observation" else sc.get(key)
        if peer is None:
            book.add(f"compare.S-X.{key}.inventory", False, "peer record absent")
            continue
        statistical = EMPIRICAL_PLANES if key[0] == "observation" else ()
        planes = (*COMMON_PLANES,
                  *(F010_COADD if key[0] == "coadd" else F010_OBSERVATION),
                  *statistical)
        compare_map_records(record, peer, planes, bool(record.get("noise")),
                            output, book,
                            f"compare.S-X.{key[0]}.{key[1]}.{key[2]}")
        if key[0] == "observation":
            book.add(f"compare.S-X.observation.{key[1]}.{key[2]}.ledger_digest",
                     record.get("sample_ledger_sha256") is not None
                     and record.get("sample_ledger_sha256") == peer.get("sample_ledger_sha256"),
                     {"sx": record.get("sample_ledger_sha256"),
                      "peer": peer.get("sample_ledger_sha256")})


def verify_sc_se_recombination_preconditions(
        output: Path, cases: Mapping[str, Mapping[str, Any]],
        reconstructions: Mapping[str, Mapping[tuple[int, str], Reconstruction]],
        book: CheckBook) -> dict[str, bool]:
    pair_gates: dict[str, bool] = {}
    support_masks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for sc_id, se_id, policy in (
            ("S-C-SEQ", "S-E-SEQ", "seq"),
            ("S-C-OMP", "S-E-OMP", "omp")):
        sc_records = map_record_index(cases[sc_id])
        se_records = map_record_index(cases[se_id])
        pair_checks: list[bool] = []
        for obsnum in (152390, 152392):
            for array in ARRAYS:
                prefix = f"recombination_precondition.{policy}.{obsnum}.{array}"
                sc_record = sc_records[("observation", obsnum, array)]
                se_record = se_records[("observation", obsnum, array)]
                sc_arrays, _, sc_header = load_map_arrays(sc_record)
                se_arrays, _, se_header = load_map_arrays(se_record)
                named_planes = (*COMMON_PLANES, "coverage_I")
                exact_planes: dict[str, bool] = {}
                for name in named_planes:
                    left = sc_arrays[name]
                    right = se_arrays[name]
                    exact_planes[name] = left.dtype == right.dtype and \
                        left.shape == right.shape and left.tobytes() == right.tobytes()
                pair_checks.append(book.add(
                    prefix + ".exact_observation_products",
                    all(exact_planes.values()),
                    {"obsnum": obsnum, "array": array,
                     "plane_bitwise_equal": exact_planes,
                     "shape_S_C": list(sc_arrays["signal_I"].shape),
                     "shape_S_E": list(se_arrays["signal_I"].shape)}))
                pair_checks.append(book.add(
                    prefix + ".exact_wcs",
                    wcs_cards(sc_header) == wcs_cards(se_header), None))
                sc_reconstruction = reconstructions[sc_id].get((obsnum, array))
                se_reconstruction = reconstructions[se_id].get((obsnum, array))
                if sc_reconstruction is None or se_reconstruction is None:
                    pair_checks.append(book.add(
                        prefix + ".realization_alignment", False,
                        "SCI-MAP-001-UNITY-001-EG-SC-SE-REALIZATION-AUTHORITY: "
                        "one independently reconstructed observation is absent",
                        evidence_gap=True))
                    continue
                sc_identity = sc_reconstruction.ledger_identity
                se_identity = se_reconstruction.ledger_identity
                alignment = sc_identity.get("sha256") == se_identity.get("sha256") and \
                    sc_identity.get("raw_membership_projection_record_count") == \
                    se_identity.get("raw_membership_projection_record_count") and \
                    sc_identity.get("scan_count") == se_identity.get("scan_count") and \
                    sc_identity.get("raw_membership_complete") is True and \
                    se_identity.get("raw_membership_complete") is True and \
                    np.array_equal(sc_reconstruction.noise, se_reconstruction.noise)
                pair_checks.append(book.add(
                    prefix + ".realization_alignment", alignment,
                    {"S_C_ledger": sc_identity, "S_E_ledger": se_identity,
                     "realization_count": REALIZATIONS,
                     "randomize_dets": False, "seed": 5489}))
                pair_checks.append(book.add(
                    prefix + ".raw_parent_identity",
                    sc_reconstruction.verified_raw_parent_digest is not None and
                    sc_reconstruction.verified_raw_parent_digest ==
                    se_reconstruction.verified_raw_parent_digest,
                    {"S_C": sc_reconstruction.verified_raw_parent_digest,
                     "S_E": se_reconstruction.verified_raw_parent_digest}))

        for array in ARRAYS:
            prefix = f"recombination_precondition.{policy}.A_floor.{array}"
            first = reconstructions[sc_id].get((152390, array))
            second = reconstructions[sc_id].get((152392, array))
            if first is None or second is None or \
                    first.planes["weight_I"].shape != second.planes["weight_I"].shape:
                pair_checks.append(book.add(
                    prefix + ".authority", False,
                    "SCI-MAP-001-UNITY-001-EG-A-FLOOR-AUTHORITY: "
                    "matched independent observation supports are unavailable",
                    evidence_gap=True))
                continue
            g1 = np.isfinite(first.planes["upstream_eligible_exposure_I"]) & \
                (first.planes["upstream_eligible_exposure_I"] > 0.0)
            g2 = np.isfinite(second.planes["upstream_eligible_exposure_I"]) & \
                (second.planes["upstream_eligible_exposure_I"] > 0.0)
            s1 = np.isfinite(first.planes["weight_I"]) & \
                (first.planes["weight_I"] > 0.0)
            s2 = np.isfinite(second.planes["weight_I"]) & \
                (second.planes["weight_I"] > 0.0)
            a_floor = g1 & g2 & np.logical_xor(s1, s2)
            unobserved = ~(g1 & g2)
            domain_path = output / f"S-C-{policy}-{array}-support-domains.npz"
            write_npz_new(
                domain_path,
                schema_version=np.array("sci-map-001-asymmetric-support-domain-v1"),
                observation_ids=np.asarray([152390, 152392], dtype=np.int64),
                pre_support_positive_exposure_152390=g1.astype(np.uint8),
                pre_support_positive_exposure_152392=g2.astype(np.uint8),
                post_support_positive_weight_152390=s1.astype(np.uint8),
                post_support_positive_weight_152392=s2.astype(np.uint8),
                asymmetric_support_floor=a_floor.astype(np.uint8),
                fewer_than_two_pre_support_exposures=unobserved.astype(np.uint8),
            )
            support_masks[(policy, array)] = (a_floor, unobserved)
            nonempty = bool(np.any(a_floor))
            pair_checks.append(book.add(
                prefix + ".nonempty", nonempty,
                {"A_floor_pixel_count": int(np.count_nonzero(a_floor)),
                 "fewer_than_two_pre_support_exposures":
                     int(np.count_nonzero(unobserved)),
                 "lossless_domain_file": str(domain_path),
                 "lossless_domain_sha256": sha256(domain_path)},
                evidence_gap=not nonempty))
        pair_gates[sc_id] = all(pair_checks)

    for array in ARRAYS:
        seq = support_masks.get(("seq", array))
        omp = support_masks.get(("omp", array))
        identical = seq is not None and omp is not None and \
            np.array_equal(seq[0], omp[0]) and np.array_equal(seq[1], omp[1])
        book.add(f"recombination_precondition.A_floor.seq_omp.{array}",
                 identical, None, evidence_gap=not identical)
        if not identical:
            pair_gates["S-C-SEQ"] = False
            pair_gates["S-C-OMP"] = False
    return pair_gates


def scan_farm_bound(actual_left: np.ndarray, actual_right: np.ndarray,
                    sum_abs: np.ndarray, scan_count: int) -> tuple[bool, dict[str, Any]]:
    epsilon = np.finfo(np.float64).eps
    product = float(scan_count) * epsilon
    if scan_count <= 0 or product >= 1.0:
        return False, {"scan_count": scan_count, "reason": "invalid gamma_n"}
    gamma = product / (1.0 - product)
    left = np.asarray(actual_left, dtype=np.float64)
    right = np.asarray(actual_right, dtype=np.float64)
    absolute = np.asarray(sum_abs, dtype=np.longdouble)
    if left.shape != right.shape or left.shape != absolute.shape:
        return False, {"left_shape": list(left.shape),
                       "right_shape": list(right.shape),
                       "sum_abs_shape": list(absolute.shape)}
    topology = np.array_equal(np.isfinite(left), np.isfinite(right))
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.zeros(left.shape, dtype=np.longdouble)
    delta[finite] = np.abs(left[finite].astype(np.longdouble) -
                           right[finite].astype(np.longdouble))
    bound = np.longdouble(2.0 * gamma) * absolute
    within = bool(np.all(delta[finite] <= bound[finite])) if finite.any() else True
    excess = delta - bound
    return bool(topology and within), {
        "policy": PARALLEL_POLICY, "scan_count": scan_count,
        "binary64_epsilon": epsilon, "gamma_n": gamma,
        "finite_count": int(np.count_nonzero(finite)),
        "max_abs_delta": float(np.max(delta[finite])) if finite.any() else 0.0,
        "max_bound": float(np.max(bound[finite])) if finite.any() else 0.0,
        "max_excess": float(np.max(excess[finite])) if finite.any() else 0.0,
        "finite_topology_equal": topology,
    }


def verify_scan_farm_pairs(cases: Mapping[str, Mapping[str, Any]],
                           reconstructions: Mapping[
                               str, Mapping[tuple[int, str], Reconstruction]],
                           book: CheckBook) -> None:
    for left_id, right_id, label in (
            ("P-SEQ", "P-OMP", "P"),
            ("S-C-SEQ", "S-C-OMP", "S-C"),
            ("S-E-SEQ", "S-E-OMP", "S-E")):
        for key, left_reconstruction in reconstructions[left_id].items():
            obsnum, array = key
            right_reconstruction = reconstructions[right_id].get(key)
            prefix = f"scan_farm.{label}.{obsnum}.{array}"
            if right_reconstruction is None:
                book.add(prefix + ".ledger", False,
                         "paired independent ledger reconstruction is absent",
                         evidence_gap=True)
                continue
            left_identity = left_reconstruction.ledger_identity
            right_identity = right_reconstruction.ledger_identity
            scan_count = int(left_identity.get("scan_count", 0))
            book.add(prefix + ".ordered_ledger_identity",
                     left_identity.get("sha256") == right_identity.get("sha256")
                     and scan_count == int(right_identity.get("scan_count", -1)),
                     {"left": left_identity, "right": right_identity})
            book.add_not_applicable(
                prefix + ".external_scan_farm_gamma_n",
                {
                    "reason": "The v1 independent sample ledger proves ordered input "
                              "membership but does not contain the exact run-produced "
                              "binary64 per-scan accumulator planes and final commit order. "
                              "Normalized signal_I*weight_I is not an authority for the "
                              "pre-normalization signal numerator.",
                    "required_schema_additions": {
                        "scan_commit_order": "int64[n_scans], an exact permutation of "
                                             "0..n_scans-1 from the run",
                        "per_scan_signal_numerator":
                            "float64[n_scans,map_rows,map_cols]",
                        "per_scan_weight": "float64[n_scans,map_rows,map_cols]",
                        "per_scan_kernel_numerator":
                            "float64[n_scans,map_rows,map_cols]",
                        "per_scan_retained_exposure":
                            "float64[n_scans,map_rows,map_cols]",
                        "per_scan_upstream_eligible_exposure":
                            "float64[n_scans,map_rows,map_cols]",
                    },
                    "bound": "2*gamma_n*sum(abs(binary64_per_scan_plane))",
                    "scope": "candidate/local F011 policy; not an external F012 "
                             "campaign pass/fail gate",
                    "claim": "no_external_gamma_n_verification",
                })


def run_tool(command: Sequence[str], stdout_path: Path,
             stderr_path: Path) -> int:
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(list(command), stdout=stdout, stderr=stderr,
                                       check=False)
    except (OSError, FileExistsError) as exc:
        raise EvidenceError(f"cannot execute frozen baseline tool: {exc}") from exc
    os.chmod(stdout_path, 0o444)
    os.chmod(stderr_path, 0o444)
    return int(completed.returncode)


def run_baseline_tools(source_root: Path, campaign: Mapping[str, Any],
                       cases: Mapping[str, Mapping[str, Any]], output: Path,
                       python: str, audit_override: Path | None,
                       compare_override: Path | None,
                       book: CheckBook) -> list[dict[str, Any]]:
    audit_tool = (audit_override or source_root / "tools/baseline/audit_reduction_run.py").resolve()
    compare_tool = (compare_override or source_root / "tools/baseline/compare_reduction_products.py").resolve()
    pins = campaign["pinned_source_sha256"]
    for path, relative in ((audit_tool, "tools/baseline/audit_reduction_run.py"),
                           (compare_tool, "tools/baseline/compare_reduction_products.py")):
        if not path.is_file() or sha256(path) != pins[relative]:
            die(f"frozen baseline tool differs from candidate pin: {path}")
    # The original protocol names these artifacts at the analysis-output root.
    tool_output = output
    records: list[dict[str, Any]] = []
    common_flags = [
        "--top", "2147483647",
        "--require-runtime-provenance",
        "--require-processed-provenance",
        "--require-raw-provenance",
        "--require-mapmaking-provenance",
        "--require-coadd-provenance",
        "--require-noise-products-provenance",
        "--require-post-processing-provenance",
        "--require-kids-external-provenance",
        "--require-polarimetry-provenance",
        "--require-astrometry-provenance",
        "--require-config-source-manifest",
    ]
    for case_id in EXPECTED_CASES:
        case = cases[case_id]
        json_out = tool_output / f"baseline-audit-{case_id}.json"
        report_out = tool_output / f"baseline-audit-{case_id}.md"
        stdout = tool_output / f"baseline-audit-{case_id}.stdout.txt"
        stderr = tool_output / f"baseline-audit-{case_id}.stderr.txt"
        command = [python, str(audit_tool), str(case["reduction_root"]),
                   *common_flags]
        if EXPECTED_CASES[case_id]["mode"] == "point":
            command.append("--require-pointing-provenance")
        command.extend(["--json-out", str(json_out), "--report-out", str(report_out)])
        exit_code = run_tool(command, stdout, stderr)
        exit_path = tool_output / f"baseline-audit-{case_id}.exit.txt"
        write_new(exit_path, f"{exit_code}\n".encode())
        outputs_exist = json_out.is_file() and json_out.stat().st_size > 0 and \
            report_out.is_file() and report_out.stat().st_size > 0
        for artifact in (json_out, report_out):
            if artifact.exists():
                os.chmod(artifact, 0o444)
        book.add(f"baseline.audit.{case_id}", exit_code == 0 and outputs_exist,
                 {"exit_code": exit_code, "outputs_exist": outputs_exist,
                  "command": command})
        records.append({"kind": "audit", "case_id": case_id,
                        "command": command, "exit_code": exit_code,
                        "json": str(json_out), "report": str(report_out),
                        "stdout": str(stdout), "stderr": str(stderr)})

    for left_id, right_id, label in (
        ("P-SEQ", "P-OMP", "P"),
        ("S-C-SEQ", "S-C-OMP", "S-C"),
        ("S-E-SEQ", "S-E-OMP", "S-E"),
    ):
        json_out = tool_output / f"baseline-compare-{label}.json"
        report_out = tool_output / f"baseline-compare-{label}.md"
        stdout = tool_output / f"baseline-compare-{label}.stdout.txt"
        stderr = tool_output / f"baseline-compare-{label}.stderr.txt"
        command = [
            python, str(compare_tool),
            str(cases[left_id]["reduction_root"]),
            str(cases[right_id]["reduction_root"]),
            "--mode", EXPECTED_CASES[left_id]["mode"],
            "--baseline-label", "seq", "--candidate-label", "omp",
            "--strict", "--include-timestream", "--exclude", "citlali_profile.ecsv",
            "--max-array-elements", "0", "--max-records", "2147483647",
            "--top", "2147483647", "--atol", "2e-8", "--rtol", "1e-10",
            "--json-out", str(json_out), "--report-out", str(report_out),
        ]
        exit_code = run_tool(command, stdout, stderr)
        exit_path = tool_output / f"baseline-compare-{label}.exit.txt"
        write_new(exit_path, f"{exit_code}\n".encode())
        outputs_exist = json_out.is_file() and json_out.stat().st_size > 0 and \
            report_out.is_file() and report_out.stat().st_size > 0
        for artifact in (json_out, report_out):
            if artifact.exists():
                os.chmod(artifact, 0o444)
        book.add(f"baseline.compare.{label}", exit_code == 0 and outputs_exist,
                 {"exit_code": exit_code, "outputs_exist": outputs_exist,
                  "command": command})
        records.append({"kind": "compare", "pair": label,
                        "command": command, "exit_code": exit_code,
                        "json": str(json_out), "report": str(report_out),
                        "stdout": str(stdout), "stderr": str(stderr)})
    return records


def serious_log_lines(path: Path) -> list[str]:
    patterns = re.compile(
        r"(?:\[(?:error|critical|fatal)\]|\b(?:ERROR|CRITICAL|FATAL)\b|"
        r"terminate called|uncaught exception)", re.IGNORECASE)
    results: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if patterns.search(line):
                    results.append(f"{number}:{line.rstrip()}")
    except OSError as exc:
        raise EvidenceError(f"cannot scan log {path}: {exc}") from exc
    return results


def required_data_skip_lines(path: Path) -> list[str]:
    """Return every log line that indicates required scientific data was skipped."""
    skip_then_data = re.compile(
        r"\b(?:skip(?:ped|ping)?|omit(?:ted|ting)?|missing|absent|unavailable|"
        r"not\s+found|could\s+not\s+(?:open|read|load))\b[^\n]{0,240}"
        r"\b(?:required(?:-data)?|mandatory|raw(?:\s+input)?|timestream|"
        r"calibration|apt|pointing|observation|scan|detector|input\s+data|"
        r"science\s+data|product)\b", re.IGNORECASE)
    data_then_skip = re.compile(
        r"\b(?:required(?:-data)?|mandatory|raw(?:\s+input)?|timestream|"
        r"calibration|apt|pointing|observation|scan|detector|input\s+data|"
        r"science\s+data|product)\b[^\n]{0,240}"
        r"\b(?:skip(?:ped|ping)?|omit(?:ted|ting)?|missing|absent|unavailable|"
        r"not\s+found|could\s+not\s+(?:open|read|load))\b", re.IGNORECASE)
    results: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if skip_then_data.search(line) or data_then_skip.search(line):
                    results.append(f"{number}:{line.rstrip()}")
    except OSError as exc:
        raise EvidenceError(f"cannot scan required-data skips in {path}: {exc}") from exc
    return results


def exact_utc_timestamp(path: Path, label: str) -> tuple[str, dt.datetime]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {path}: {exc}") from exc
    if len(lines) != 1 or RFC3339_UTC_RE.fullmatch(lines[0]) is None:
        die(f"{label} is not exactly one RFC3339 UTC timestamp: {path}")
    try:
        parsed = dt.datetime.strptime(lines[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise EvidenceError(f"{label} is not a calendar-valid UTC timestamp: {path}") from exc
    return lines[0], parsed


def parse_sha256_record(path: Path, label: str) -> list[tuple[Path, str]]:
    """Parse and verify a GNU sha256sum manifest without trusting its bytes."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {path}: {exc}") from exc
    if not lines:
        die(f"{label} is empty: {path}")
    records: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (/[^\n\r]+)", line)
        if match is None:
            die(f"{label} line {index} is not a strict SHA-256 record")
        expected, raw_path = match.groups()
        target = Path(raw_path)
        if target != Path(os.path.normpath(raw_path)) or target in seen or \
                target.is_symlink() or not target.is_file():
            die(f"{label} line {index} has a noncanonical/repeated/missing target")
        if sha256(target) != expected:
            die(f"{label} line {index} target digest differs: {target}")
        seen.add(target)
        records.append((target, expected))
    return records


def parse_runtime_environment(path: Path, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {path}: {exc}") from exc
    if not lines or lines != sorted(lines):
        die(f"{label} is empty or not bytewise sorted")
    values: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"((?:OMP|SLURM|TOLPROJ)_[A-Z0-9_]+)=(.*)", line)
        if match is None or match.group(1) in values:
            die(f"{label} contains an invalid or repeated environment record")
        values[match.group(1)] = match.group(2)
    return values


def parse_cpu_affinity(path: Path, label: str) -> set[int] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read {label}: {path}: {exc}") from exc
    if lines == ["taskset unavailable"]:
        return None
    if len(lines) != 1:
        die(f"{label} must contain one taskset affinity record")
    match = re.fullmatch(r"pid [0-9]+'s current affinity list: ([0-9,-]+)", lines[0])
    if match is None:
        die(f"{label} is not a usable taskset affinity record")
    cpus: set[int] = set()
    for token in match.group(1).split(","):
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                die(f"{label} contains a reversed CPU range")
            values = set(range(first, last + 1))
        else:
            values = {int(token)}
        if cpus & values:
            die(f"{label} contains overlapping CPU identities")
        cpus.update(values)
    return cpus


def slurm_memory_bytes(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value)
    if match is None:
        return None
    scale = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3,
             "T": 1024 ** 4, "P": 1024 ** 5}[match.group(2)]
    result = float(match.group(1)) * scale
    return int(result) if math.isfinite(result) and result > 0.0 else None


def verify_frozen_stat(path: Path, record: Mapping[str, Any], label: str) -> None:
    if set(record) != {"path", "sha256", "size", "mtime_ns"}:
        die(f"{label}: frozen file record fields differ")
    stat = path.stat()
    if not isinstance(record.get("size"), int) or record["size"] < 0 or \
            stat.st_size != record["size"]:
        die(f"{label}: frozen size changed for {path}")
    if not isinstance(record.get("mtime_ns"), int) or record["mtime_ns"] < 0 or \
            stat.st_mtime_ns != record["mtime_ns"]:
        die(f"{label}: frozen mtime changed for {path}")


def verify_job_evidence(
        case_id: str, case: Mapping[str, Any], execution: Mapping[str, Any],
        artifact_paths: Mapping[str, Path], log_records: Sequence[Mapping[str, Any]],
        inventory: Sequence[Mapping[str, Any]], reduction_root: Path,
        merged_config: Path, expected_numbered_order: Sequence[str],
        book: CheckBook) -> None:
    log_paths = [Path(str(record["path"])) for record in log_records]
    if len(set(log_paths)) != len(log_paths):
        die(f"{case_id}: complete log inventory repeats a path")
    by_name: dict[str, Path] = {}
    for path in log_paths:
        if path.name in by_name:
            die(f"{case_id}: complete log inventory repeats basename {path.name}")
        by_name[path.name] = path

    fixed_names = {
        "stdout.txt", "stderr.txt", "submit-record.txt", "submit-stderr.txt",
        "slurm-accounting.txt", "started-at-utc.txt", "completed-at-utc.txt",
        "pre-run-sha256.txt", "post-run-sha256.txt",
        "integrity-manifest.sha256", "reconstruction-manifest.sha256",
        "hostname.txt", "runtime-environment.txt", "affinity.txt",
    }
    wrapper_stdout = sorted(name for name in by_name if re.fullmatch(
        r"slurm-wrapper-[0-9]+\.out", name))
    wrapper_stderr = sorted(name for name in by_name if re.fullmatch(
        r"slurm-wrapper-[0-9]+\.err", name))
    expected_names = fixed_names | set(wrapper_stdout) | set(wrapper_stderr)
    complete_inventory = set(by_name) == expected_names and \
        len(wrapper_stdout) == 1 and len(wrapper_stderr) == 1
    book.add(f"{case_id}.log.complete_role_inventory", complete_inventory,
             {"actual": sorted(by_name), "expected_fixed": sorted(fixed_names),
              "wrapper_stdout": wrapper_stdout,
              "wrapper_stderr": wrapper_stderr})
    if not complete_inventory:
        return

    artifact_log_names = {
        "submit_record": "submit-record.txt", "stdout": "stdout.txt",
        "stderr": "stderr.txt", "slurm_accounting": "slurm-accounting.txt",
    }
    book.add(f"{case_id}.log.result_artifact_bindings",
             all(artifact_paths[key] == by_name[name]
                 for key, name in artifact_log_names.items()),
             {key: str(artifact_paths[key]) for key in artifact_log_names})

    try:
        submit_lines = by_name["submit-record.txt"].read_text(
            encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"{case_id}: cannot read submit record: {exc}") from exc
    submit_match = re.fullmatch(
        r"job_ref=([0-9]+)(?:;[A-Za-z0-9._-]+)?\nsubmit_rc=0",
        "\n".join(submit_lines))
    book.add(f"{case_id}.log.submit_success_record", submit_match is not None,
             submit_lines)
    if submit_match is None:
        return
    job_id = submit_match.group(1)
    expected_wrapper_names = {
        f"slurm-wrapper-{job_id}.out", f"slurm-wrapper-{job_id}.err"}
    book.add(f"{case_id}.log.wrapper_job_identity",
             set(wrapper_stdout + wrapper_stderr) == expected_wrapper_names,
             {"job_id": job_id,
              "wrapper_records": wrapper_stdout + wrapper_stderr})

    accounting_path = by_name["slurm-accounting.txt"]
    try:
        accounting_lines = accounting_path.read_text(
            encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"{case_id}: cannot read Slurm accounting: {exc}") from exc
    header_ok = bool(accounting_lines) and \
        accounting_lines[0].split("|") == list(SLURM_FIELDS)
    rows: list[dict[str, str]] = []
    rows_ok = header_ok and len(accounting_lines) >= 2
    if rows_ok:
        for line in accounting_lines[1:]:
            values = line.split("|")
            if len(values) != len(SLURM_FIELDS):
                rows_ok = False
                break
            rows.append(dict(zip(SLURM_FIELDS, values)))
        rows_ok = rows_ok and len({row["JobIDRaw"] for row in rows}) == len(rows) \
            and all(row["JobIDRaw"] for row in rows)
    book.add(f"{case_id}.log.slurm_accounting_shape", rows_ok,
             {"header": accounting_lines[0] if accounting_lines else None,
              "row_count": max(0, len(accounting_lines) - 1)})
    if not rows_ok:
        return
    top_rows = [row for row in rows if row["JobIDRaw"] == job_id]
    top = top_rows[0] if len(top_rows) == 1 else {}
    expected_top = {
        "JobName": f"sci-map-001-{case_id}",
        "Partition": execution.get("partition"),
        "AllocCPUS": str(case.get("cpus")),
        "State": "COMPLETED", "ExitCode": "0:0",
    }
    top_ok = len(top_rows) == 1 and all(
        top.get(key) == value for key, value in expected_top.items()) and all(
            top.get(key) for key in ("NodeList", "Elapsed", "ReqMem")) and all(
                SLURM_TIMESTAMP_RE.fullmatch(str(top.get(key, ""))) is not None
                for key in ("Submit", "Start", "End"))
    book.add(f"{case_id}.log.slurm_completed_identity", top_ok,
             {"job_id": job_id, "actual": top, "expected": expected_top})
    if top_ok:
        book.add(f"{case_id}.log.slurm_timestamp_order",
                 top["Submit"] <= top["Start"] <= top["End"],
                 {key: top[key] for key in ("Submit", "Start", "End")})
    memory_rows = [
        {"job": row["JobIDRaw"], "raw": row["MaxRSS"],
         "bytes": slurm_memory_bytes(row["MaxRSS"])}
        for row in rows if row["JobIDRaw"] == job_id or
        row["JobIDRaw"].startswith(job_id + ".")
    ]
    usable_memory = [row for row in memory_rows if row["bytes"] is not None]
    book.add(f"{case_id}.log.slurm_peak_memory",
             bool(usable_memory),
             {"records": memory_rows,
              "peak_bytes": max((int(row["bytes"]) for row in usable_memory),
                                default=None)},
             evidence_gap=not usable_memory)

    try:
        exit_lines = artifact_paths["exit_record"].read_text(
            encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"{case_id}: cannot read exit record: {exc}") from exc
    book.add(f"{case_id}.log.exact_zero_exit_record", exit_lines == ["0"],
             exit_lines)
    started_text, started = exact_utc_timestamp(
        by_name["started-at-utc.txt"], f"{case_id} start marker")
    completed_text, completed = exact_utc_timestamp(
        by_name["completed-at-utc.txt"], f"{case_id} completion marker")
    book.add(f"{case_id}.log.execution_timestamp_order",
             started <= completed,
             {"started": started_text, "completed": completed_text})
    hostname_lines = by_name["hostname.txt"].read_text(
        encoding="utf-8").splitlines()
    hostname_ok = len(hostname_lines) == 1 and bool(re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", hostname_lines[0]))
    book.add(f"{case_id}.log.hostname_usable", hostname_ok, hostname_lines)

    runtime_environment = parse_runtime_environment(
        by_name["runtime-environment.txt"], f"{case_id} runtime environment")
    expected_runtime = {
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_NAME": f"sci-map-001-{case_id}",
        "SLURM_JOB_PARTITION": str(execution.get("partition")),
        "SLURM_CPUS_PER_TASK": str(case.get("cpus")),
        "OMP_NUM_THREADS": str(case.get("threads")),
    }
    book.add(f"{case_id}.log.runtime_allocation_identity",
             all(runtime_environment.get(key) == value
                 for key, value in expected_runtime.items()),
             {"expected": expected_runtime,
              "actual": {key: runtime_environment.get(key)
                         for key in expected_runtime}},
             evidence_gap=any(runtime_environment.get(key) != value
                              for key, value in expected_runtime.items()))
    snapshot_value = runtime_environment.get("TOLPROJ_CITLALI_SNAPSHOT")
    snapshot_digest = runtime_environment.get("TOLPROJ_CITLALI_SHA256")
    snapshot = Path(snapshot_value) if isinstance(snapshot_value, str) else Path()
    snapshot_ok = isinstance(snapshot_value, str) and snapshot.is_absolute() and \
        snapshot == Path(os.path.normpath(snapshot_value)) and \
        not snapshot.is_symlink() and snapshot.is_file() and \
        isinstance(snapshot_digest, str) and \
        re.fullmatch(r"[0-9a-f]{64}", snapshot_digest) is not None and \
        sha256(snapshot) == snapshot_digest
    book.add(f"{case_id}.log.snapshot_runtime_identity", snapshot_ok,
             {"path": snapshot_value, "sha256": snapshot_digest})

    affinity = parse_cpu_affinity(
        by_name["affinity.txt"], f"{case_id} CPU affinity")
    affinity_ok = affinity is not None and len(affinity) >= int(case["cpus"])
    book.add(f"{case_id}.log.affinity_usable", affinity_ok,
             {"cpu_count": len(affinity) if affinity is not None else None,
              "cpus": sorted(affinity) if affinity is not None else None,
              "required_allocation": case["cpus"]},
             evidence_gap=not affinity_ok)

    merged = read_yaml(merged_config)
    if not isinstance(merged, Mapping):
        die(f"{case_id}: pre-run merged config is malformed")
    preflight = read_json(artifact_paths["preflight_manifest"])
    preflight_paths = preflight.get("paths") if isinstance(preflight, Mapping) else None
    executable_value = (preflight_paths.get("candidate_executable")
                        if isinstance(preflight_paths, Mapping) else None)
    executable = Path(executable_value) if isinstance(executable_value, str) else Path()
    expected_integrity_paths = (
        executable, snapshot, reduction_root / "02_redu.sh")
    pre_records = parse_sha256_record(
        by_name["pre-run-sha256.txt"], f"{case_id} pre-run authority")
    post_records = parse_sha256_record(
        by_name["post-run-sha256.txt"], f"{case_id} post-run authority")
    pre_paths = tuple(path for path, _ in pre_records)
    pre_post_ok = pre_records == post_records and \
        snapshot_ok and pre_paths == expected_integrity_paths and \
        pre_records[0][1] == snapshot_digest == pre_records[1][1]
    book.add(f"{case_id}.log.pre_post_authority_identity", pre_post_ok,
             {"expected_paths": [str(path) for path in expected_integrity_paths],
              "actual_paths": [str(path) for path in pre_paths],
              "pre_sha256": sha256(by_name["pre-run-sha256.txt"]),
              "post_sha256": sha256(by_name["post-run-sha256.txt"])})

    request_root = artifact_paths["preflight_manifest"].parent.parent.parent
    expected_bound_manifests = {
        "integrity-manifest.sha256":
            artifact_paths["preflight_manifest"].parent / "pre-submit-sha256.txt",
        "reconstruction-manifest.sha256":
            request_root / "manifests" /
            "SCI-MAP-001-RECONSTRUCTION-AUTHORITY.sha256",
    }
    manifest_records: dict[str, list[tuple[Path, str]]] = {}
    for binding_name, expected_target in expected_bound_manifests.items():
        binding_records = parse_sha256_record(
            by_name[binding_name], f"{case_id} {binding_name} binding")
        binding_ok = len(binding_records) == 1 and \
            binding_records[0][0] == expected_target
        book.add(f"{case_id}.log.{binding_name}.exact_target",
                 binding_ok,
                 {"expected": str(expected_target),
                  "actual": [str(path) for path, _ in binding_records]})
        if binding_ok:
            manifest_records[binding_name] = parse_sha256_record(
                expected_target, f"{case_id} bound {binding_name}")
    integrity_targets = {
        path for path, _ in manifest_records.get(
            "integrity-manifest.sha256", ())}
    required_integrity_targets = {
        artifact_paths["preflight_manifest"], merged_config,
        reduction_root / "02_redu.sh", snapshot,
        Path(str(preflight_paths["launcher"])),
        Path(str(preflight_paths["launcher_source"])),
        *(reduction_root / name for name in expected_numbered_order),
    }
    book.add(f"{case_id}.log.pre_submit_integrity_coverage",
             required_integrity_targets.issubset(integrity_targets),
             {"missing": sorted(str(path) for path in
                                required_integrity_targets - integrity_targets)})

    gnu_time_matches = []
    for line in by_name["stderr.txt"].read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(
            r"\s*Maximum resident set size \(kbytes\): ([0-9]+)\s*", line)
        if match:
            gnu_time_matches.append(int(match.group(1)))
    book.add(f"{case_id}.log.gnu_time_peak_memory",
             len(gnu_time_matches) == 1 and gnu_time_matches[0] > 0,
             {"maximum_resident_set_size_kbytes": gnu_time_matches},
             evidence_gap=not (len(gnu_time_matches) == 1 and
                               gnu_time_matches[0] > 0))

    completion_record = next(record for record in log_records
                             if Path(str(record["path"])).name ==
                             "completed-at-utc.txt")
    completion_mtime = int(completion_record["mtime_ns"])
    late_files = [
        {"relative_path": item.get("relative_path"),
         "mtime_ns": item.get("mtime_ns")}
        for item in inventory if not isinstance(item.get("mtime_ns"), int)
        or int(item["mtime_ns"]) > completion_mtime
    ]
    book.add(f"{case_id}.log.completion_after_physical_inventory",
             not late_files,
             {"completion_marker_mtime_ns": completion_mtime,
              "physical_file_count": len(inventory), "late_files": late_files})

    required_skips = [
        {"path": str(path), "matches": matches}
        for path in log_paths
        if (matches := required_data_skip_lines(path))
    ]
    book.add(f"{case_id}.log.zero_required_data_skips", not required_skips,
             required_skips)


def optional_realized_value(node: Any) -> Any:
    if not isinstance(node, Mapping) or node.get("available") is not True:
        return None
    return node.get("value")


def verify_noise_provenance(provenance: Mapping[str, Any],
                            case: Mapping[str, Any], book: CheckBook) -> None:
    prefix = f"{case['id']}.noise_products_provenance"
    requested = provenance.get("requested")
    effective_node = provenance.get("effective")
    effective = effective_node.get("config") \
        if isinstance(effective_node, Mapping) else None
    resolution = effective_node.get("resolution") \
        if isinstance(effective_node, Mapping) else None
    realized = provenance.get("realized")
    if not all(isinstance(value, Mapping) for value in
               (requested, effective, resolution, realized)):
        book.add(prefix + ".shape", False, "provenance state is malformed")
        return
    expected_config = {
        "enabled": True, "n_noise_maps": REALIZATIONS,
        "randomize_dets": False, "write_realizations": True,
        "products": {"enabled": bool(case["products_enabled"]),
                     "apply_empirical_weights": False},
    }
    book.add(prefix + ".requested_effective_config",
             provenance.get("schema_version") ==
             "citlali-noise-products-provenance-v1"
             and provenance.get("initialized") is True
             and requested == expected_config and effective == expected_config,
             {"requested": requested, "effective": effective,
              "expected": expected_config})
    randomization = resolution.get("randomization")
    expected_randomization = {
        "engine": "boost::random::mt19937", "seed": 5489,
        "seed_policy": "fixed_internal_default",
        "generator_scope": "reduction_pipeline_invocation",
    }
    book.add(prefix + ".randomization_identity",
             randomization == expected_randomization,
             {"actual": randomization, "expected": expected_randomization})
    book.add(prefix + ".activation_resolution",
             resolution.get("mapmaking_enabled") is True
             and resolution.get("requested_enabled") is True
             and resolution.get("effective_enabled") is True
             and resolution.get("disabled_by_mapmaking") is False
             and resolution.get("requested_n_noise_maps") == REALIZATIONS
             and resolution.get("effective_n_noise_maps") == REALIZATIONS
             and resolution.get("count_zeroed_while_disabled") is False,
             dict(resolution))
    observation_maps = len(case["expected_observations"]) * len(ARRAYS)
    coadd_maps = len(ARRAYS) if case["coadd"] else 0
    realization_writes = (
        case["expected_counts"]["observation_noise_files"] +
        case["expected_counts"]["coadd_noise_files"]) * REALIZATIONS
    expected_counts = {
        "noise_maps_per_scientific_map": REALIZATIONS,
        "observation_scientific_map_count": observation_maps,
        "observation_noise_realization_count": observation_maps * REALIZATIONS,
        "coadd_scientific_map_count": coadd_maps,
        "coadd_noise_realization_count": coadd_maps * REALIZATIONS,
        "total_noise_realization_count":
            (observation_maps + coadd_maps) * REALIZATIONS,
        "empirical_product_map_count":
            observation_maps if case["products_enabled"] else 0,
        "realization_image_write_count": realization_writes,
    }
    actual_counts = {key: optional_realized_value(realized.get(key))
                     for key in expected_counts}
    book.add(prefix + ".realized_counts",
             actual_counts == expected_counts,
             {"actual": actual_counts, "expected": expected_counts})
    book.add(prefix + ".completion",
             realized.get("reduction_completed") is True
             and realized.get("generation_executed") is True
             and realized.get("outputs_completed") is True,
             dict(realized))


def verify_case_frozen_inputs(
        case_record: Mapping[str, Any], case: Mapping[str, Any], book: CheckBook,
        forbidden_roots: Sequence[Path],
        campaign: Mapping[str, Any],
        product_contracts: Path,
        contracts: Mapping[str, Any]) -> RawManifestAuthority:
    case_id = case["id"]
    book.add(f"{case_id}.exit_status", case_record.get("exit_status") == 0,
             case_record.get("exit_status"))
    verify_frozen_path(case_record.get("merged_config"),
                       case_record.get("merged_config_sha256"),
                       f"{case_id} merged config")
    raw_path = verify_frozen_path(case_record.get("raw_input_manifest"),
                                  case_record.get("raw_input_manifest_sha256"),
                                  f"{case_id} raw-input manifest")
    raw_authority = validate_raw_input_manifest(
        raw_path, case["mode"], case["expected_observations"],
        forbidden_roots=forbidden_roots)
    if case_record.get("raw_input_authority") != raw_authority_record(raw_authority):
        die(f"{case_id}: frozen raw-input authority summary differs")
    mapmaking_path = verify_frozen_path(case_record.get("mapmaking_provenance"),
                                        case_record.get("mapmaking_provenance_sha256"),
                                        f"{case_id} mapmaking provenance")
    coadd_path = verify_frozen_path(case_record.get("coadd_provenance"),
                                    case_record.get("coadd_provenance_sha256"),
                                    f"{case_id} coadd provenance")
    noise_provenance_path = verify_frozen_path(
        case_record.get("noise_products_provenance"),
        case_record.get("noise_products_provenance_sha256"),
        f"{case_id} noise-products provenance")
    book.add(f"{case_id}.coadd_provenance_inventory",
             coadd_path is not None,
             {"present": coadd_path is not None, "coadd_enabled": case["coadd"]})
    result_artifacts = case_record.get("result_artifacts")
    if not isinstance(result_artifacts, Mapping) or \
            set(result_artifacts) != set(COLLECTION_CASE_FILE_FIELDS):
        die(f"{case_id}: frozen result-artifact inventory is incomplete")
    artifact_paths: dict[str, Path] = {}
    for name in COLLECTION_CASE_FILE_FIELDS:
        artifact = result_artifacts[name]
        if not isinstance(artifact, Mapping):
            die(f"{case_id}: frozen result artifact {name} is malformed")
        artifact_path = verify_frozen_path(
            artifact.get("path"), artifact.get("sha256"),
            f"{case_id} {name.replace('_', ' ')}")
        verify_frozen_stat(artifact_path, artifact,
                           f"{case_id} {name.replace('_', ' ')}")
        artifact_paths[name] = artifact_path
    preflight = read_json(artifact_paths["preflight_manifest"])
    preflight_case = preflight.get("case") if isinstance(preflight, Mapping) else None
    if not isinstance(preflight, Mapping) or not isinstance(preflight_case, Mapping) or \
            preflight.get("schema_version") != "sci-map-unity-case-preflight-v1" or \
            preflight.get("request_id") != REQUEST_ID or \
            preflight.get("candidate_sha") != CANDIDATE_SHA or \
            preflight_case.get("id") != case_id or \
            preflight.get("sha256", {}).get("raw_input_manifest") != \
            raw_authority.digest or preflight.get("raw_input_authority") != \
            raw_authority_record(raw_authority):
        die(f"{case_id}: frozen preflight/raw-input authority binding differs")
    validate_preflight_file_binding(
        preflight, case, campaign,
        Path(str(case_record["reduction_root"])),
        Path(str(case_record["merged_config"])), raw_path,
        product_contracts)
    log_records = case_record.get("logs")
    if not isinstance(log_records, list) or not log_records:
        die(f"{case_id}: frozen complete log inventory is absent")
    for index, log in enumerate(log_records):
        if not isinstance(log, Mapping):
            die(f"{case_id}: frozen log {index} is malformed")
        path = verify_frozen_path(log.get("path"), log.get("sha256"),
                                  f"{case_id} log {index}")
        verify_frozen_stat(path, log, f"{case_id} log {index}")
        lines = serious_log_lines(path)
        book.add(f"{case_id}.log.{index}.no_unexpected_serious_records",
                 not lines, lines)
    inventory = case_record.get("physical_inventory")
    if not isinstance(inventory, list) or not inventory:
        die(f"{case_id}: physical inventory is absent")
    reduction_root = Path(str(case_record.get("reduction_root", ""))).resolve()
    if not reduction_root.is_dir():
        die(f"{case_id}: frozen reduction root is absent")
    runtime_records: dict[str, Path] = {}
    for key in ("runtime_merged_config", "config_source_manifest"):
        value = case_record.get(key)
        if not isinstance(value, Mapping):
            die(f"{case_id}: frozen {key} record is malformed")
        frozen_path = verify_frozen_path(
            value.get("path"), value.get("sha256"),
            f"{case_id} {key.replace('_', ' ')}")
        verify_frozen_stat(frozen_path, value,
                           f"{case_id} {key.replace('_', ' ')}")
        runtime_records[key] = frozen_path
    runtime_authority = validate_runtime_config_authority(
        reduction_root, Path(str(case_record["merged_config"])), case_id)
    runtime_binding = {
        key: value for key, value in runtime_authority.items()
        if key not in ("runtime_path", "manifest_path")
    }
    if runtime_records["runtime_merged_config"] != \
            runtime_authority["runtime_path"] or \
            runtime_records["config_source_manifest"] != \
            runtime_authority["manifest_path"] or \
            case_record.get("runtime_config_binding") != runtime_binding:
        die(f"{case_id}: frozen runtime config authority differs")
    frozen_inventory_paths: set[Path] = set()
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping) or set(item) != {
                "path", "relative_path", "size", "mtime_ns", "sha256"}:
            die(f"{case_id}: physical inventory record {index} is malformed")
        path = verify_frozen_path(item.get("path"), item.get("sha256"),
                                  f"{case_id} physical inventory {index}")
        stat = path.stat()
        if stat.st_size != item.get("size"):
            die(f"{case_id}: physical inventory size changed for {path}")
        if stat.st_mtime_ns != item.get("mtime_ns"):
            die(f"{case_id}: physical inventory mtime changed for {path}")
        try:
            relative = path.relative_to(reduction_root).as_posix()
        except ValueError as exc:
            raise EvidenceError(
                f"{case_id}: physical inventory escapes reduction root: {path}") from exc
        if item.get("relative_path") != relative or path in frozen_inventory_paths:
            die(f"{case_id}: physical inventory path identity differs for {path}")
        frozen_inventory_paths.add(path)
    current_inventory_paths = {
        path.resolve() for path in reduction_root.rglob("*") if path.is_file()
    }
    if current_inventory_paths != frozen_inventory_paths:
        die(f"{case_id}: reduction-root inventory changed after freezing; "
            f"added={sorted(str(path) for path in current_inventory_paths - frozen_inventory_paths)}, "
            f"removed={sorted(str(path) for path in frozen_inventory_paths - current_inventory_paths)}")
    verify_diagnostic_families(
        reduction_root, case, contracts, inventory, book)
    numbered_key = "point_order" if case["mode"] == "point" else "science_order"
    verify_job_evidence(
        case_id, case, campaign["fixed_execution"], artifact_paths,
        log_records, inventory, reduction_root,
        Path(str(case_record["merged_config"])),
        campaign["numbered_config_contract"][numbered_key], book)
    mapmaking = read_yaml(mapmaking_path)
    if not isinstance(mapmaking, Mapping):
        die(f"{case_id}: mapmaking provenance is malformed")
    book.add(f"{case_id}.mapmaking_completion",
             mapmaking.get("schema_version") == "citlali-mapmaking-provenance-v3"
             and mapmaking.get("initialized") is True
             and mapmaking.get("realized", {}).get("reduction_completed") is True
             and mapmaking.get("realized", {}).get("mapmaking_executed") is True,
             mapmaking.get("realized"))
    coadd = read_yaml(coadd_path)
    if not isinstance(coadd, Mapping):
        die(f"{case_id}: coadd provenance is malformed")
    realized = coadd.get("realized", {})
    book.add(f"{case_id}.coadd_completion",
             coadd.get("schema_version") == "citlali-coadd-provenance-v2"
             and coadd.get("initialized") is True
             and realized.get("reduction_completed") is True
             and realized.get("coadd_executed") is bool(case["coadd"])
             and realized.get("outputs_completed") is bool(case["coadd"]),
             realized)
    noise_provenance = read_yaml(noise_provenance_path)
    if not isinstance(noise_provenance, Mapping):
        die(f"{case_id}: noise-products provenance is malformed")
    verify_noise_provenance(noise_provenance, case, book)
    return raw_authority


def write_npz_new(path: Path, **arrays: np.ndarray) -> None:
    try:
        with path.open("xb") as handle, zipfile.ZipFile(
                handle, mode="w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=9) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    die(f"invalid deterministic NPZ array name: {name!r}")
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload, np.asanyarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(
                    filename=name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compress_type=
                                 zipfile.ZIP_DEFLATED, compresslevel=9)
        os.chmod(path, 0o444)
    except (OSError, FileExistsError, ValueError, zipfile.BadZipFile) as exc:
        with contextlib.suppress(OSError):
            path.unlink()
        raise EvidenceError(f"cannot write lossless NPZ {path}: {exc}") from exc


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "rms": None, "median": None,
                "minimum": None, "maximum": None}
    return {
        "count": int(finite.size), "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)), "maximum": float(np.max(finite)),
    }


def map_identifier(record: Mapping[str, Any]) -> str:
    obs = "coadd" if record["scope"] == "coadd" else str(record["obsnum"])
    return f"{record['scope']}-{obs}-{record['array']}"


def fits_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else {
            "classification": "nonfinite_fits_header_value",
            "text": str(numeric),
        }
    if isinstance(value, np.integer):
        return int(value)
    return str(value)


def selected_header_metadata(header: fits.Header) -> dict[str, Any]:
    threshold_keys = {
        card.keyword: fits_json_value(card.value) for card in header.cards
        if "THRESH" in card.keyword or card.keyword.endswith("CUT")
    }
    response_keys = (
        "FWHM", "RESPNORM", "CALTYPE", "PRECSTAT", "COVSTAT",
        "RAWSTATE", "RAWPDGST", "VALAUTH", "ALIASOF", "DEPRCATD",
        "DATTYP", "MEDRMS", "EMP_SCALE", "WVARMED",
    )
    return {
        "threshold_keys": threshold_keys,
        "response_and_identity_keys": {
            key: fits_json_value(header[key]) for key in response_keys
            if key in header
        },
    }


def primary_observation_membership(header: fits.Header) -> dict[str, Any]:
    membership = {
        card.keyword: fits_json_value(card.value) for card in header.cards
        if re.fullmatch(r"(?:NOBS|OBSNUM[0-9]*|OBSID[0-9]*)", card.keyword)
    }
    geometry = {
        key: fits_json_value(header[key])
        for key in ("EXPTIME", "SRC_RA", "SRC_DEC", "TAN_RA", "TAN_DEC")
        if key in header
    }
    return {
        "observation_keys": membership,
        "array_key": fits_json_value(header.get("WAV")),
        "geometry_keys": geometry,
    }


def primary_membership_matches_record(
        membership: Mapping[str, Any], record: Mapping[str, Any],
        case: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    expected_observations = (
        [int(record["obsnum"])] if record["scope"] == "observation"
        else [int(value) for value in case["expected_observations"]])
    expected_keys = {
        f"OBSNUM{index}": obsnum
        for index, obsnum in enumerate(expected_observations)
    }
    actual_keys = membership.get("observation_keys")
    actual_array = membership.get("array_key")
    passed = actual_keys == expected_keys and actual_array == record["array"]
    return passed, {
        "actual_observation_keys": actual_keys,
        "expected_observation_keys": expected_keys,
        "actual_WAV": actual_array,
        "expected_WAV": record["array"],
        "scope": record["scope"],
    }


def hdu_inventory_record(
        name: str, hdu: Any, classification: str,
        reference_wcs: Mapping[str, Any], membership: Mapping[str, Any],
        parent_file_sha256: str) -> dict[str, Any]:
    is_image = isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU))
    record: dict[str, Any] = {
        "extension": name, "classification": classification,
        "hdu_class": type(hdu).__name__, "is_image_hdu": is_image,
        "parent_file_sha256": parent_file_sha256,
        "bitpix": fits_json_value(hdu.header.get("BITPIX")),
        "UNIT": fits_json_value(hdu.header.get("UNIT")),
        "BUNIT": fits_json_value(hdu.header.get("BUNIT")),
        "TYPE": fits_json_value(hdu.header.get("TYPE")),
        "ESTTYPE": fits_json_value(hdu.header.get("ESTTYPE")),
        "wcs": {key: fits_json_value(value)
                for key, value in wcs_cards(hdu.header).items()},
        "wcs_relationship_to_signal_I": (
            "exact_same_cards" if wcs_cards(hdu.header) == reference_wcs
            else "different_cards"),
        "observation_membership": dict(membership),
        **selected_header_metadata(hdu.header),
    }
    if not is_image or hdu.data is None:
        record.update({
            "data_present": hdu.data is not None,
            "numeric_image": False,
            "data_shape": None,
            "array_element_count": None,
            "data_sha256_c_order": None,
        })
        return record
    array = np.asarray(hdu.data)
    numeric = array.dtype.kind in "iufcb"
    record.update({
        "data_present": True,
        "numeric_image": numeric,
        "dtype": str(array.dtype),
        "data_shape": list(array.shape),
        "spatial_shape": list(array.shape[-2:]) if array.ndim >= 2 else None,
        "array_element_count": int(array.size),
        "map_plane_count": 1,
        "finite_element_count": (int(np.count_nonzero(np.isfinite(array)))
                                 if numeric else None),
        "nonfinite_element_count": (int(np.count_nonzero(~np.isfinite(array)))
                                    if numeric else None),
        "data_sha256_c_order": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")).hexdigest(),
        "data_digest_definition":
            "sha256-of-astropy-decoded-array-C-order-bytes; dtype-and-shape-separate",
    })
    return record


def request_union_plane_contract(
        case: Mapping[str, Any], scope: str) -> tuple[set[str], set[str]]:
    if scope == "observation":
        required = {
            *COMMON_PLANES, *F010_OBSERVATION,
            *(EMPIRICAL_PLANES if case["products_enabled"] else
              ("formal_standardized_signal_I",)),
        }
        forbidden = {
            *(OBSERVATION_PRODUCTS_OFF_FORBIDDEN
              if not case["products_enabled"] else
              ("formal_standardized_signal_I",)),
            "coadd_observation_count_I",
        }
    elif scope == "coadd":
        required = ({*COMMON_PLANES, *F010_COADD}
                    if case["coadd"] else set())
        forbidden = set(COADD_FORBIDDEN)
    else:
        die(f"unknown request-union scope: {scope}")
    if required & forbidden:
        die(f"{case['id']} {scope} required/forbidden plane sets overlap")
    return required, forbidden


def write_request_union(output: Path, campaign: Mapping[str, Any],
                        frozen_cases: Mapping[str, Mapping[str, Any]],
                        book: CheckBook) -> None:
    case_outputs = []
    for case in campaign["cases"]:
        observation_expected, observation_forbidden_set = \
            request_union_plane_contract(case, "observation")
        coadd_expected, coadd_forbidden_set = \
            request_union_plane_contract(case, "coadd")
        observation_required = sorted(observation_expected)
        observation_forbidden = sorted(observation_forbidden_set)
        coadd_required = sorted(coadd_expected)
        expected_by_scope = {
            "observation": observation_expected,
            "coadd": coadd_expected,
        }
        forbidden_by_scope = {
            "observation": observation_forbidden_set,
            "coadd": coadd_forbidden_set,
        }
        returned_maps = []
        frozen_case = frozen_cases[case["id"]]
        for record in frozen_case["maps"]:
            product = FitsProduct.open(Path(record["map"]))
            try:
                actual = set(product.hdus)
                expected = expected_by_scope[record["scope"]]
                forbidden = forbidden_by_scope[record["scope"]]
                missing = sorted(expected - actual)
                forbidden_present = sorted(actual & forbidden)
                reference_wcs = wcs_cards(product.hdus["signal_I"].header) \
                    if "signal_I" in product.hdus else {}
                membership = primary_observation_membership(
                    product.hdul[0].header)
                membership_passed, membership_detail = \
                    primary_membership_matches_record(membership, record, case)
                book.add(
                    f"request_union.{case['id']}.{map_identifier(record)}."
                    "map_primary_membership",
                    membership_passed, membership_detail)
                plane_inventory = [hdu_inventory_record(
                    "PRIMARY", product.hdul[0], "required_metadata_primary",
                    reference_wcs, membership, record["map_sha256"]), *[
                    hdu_inventory_record(
                        name, product.hdus[name],
                        ("required_returned_map_product" if name in expected else
                         "additional_returned_map_product"),
                        reference_wcs, membership, record["map_sha256"])
                    for name in sorted(actual)
                ]]
                additional = [item for item in plane_inventory
                              if item["classification"] ==
                              "additional_returned_map_product"]
                book.add(
                    f"request_union.{case['id']}.{map_identifier(record)}.required",
                    not missing and not forbidden_present,
                    {"missing": missing,
                     "forbidden_present": forbidden_present,
                                  "additional_classified": additional})
                noise_plane_inventory: list[dict[str, Any]] = []
                noise_membership: dict[str, Any] | None = None
                if record.get("noise"):
                    noise_product = FitsProduct.open(Path(record["noise"]))
                    try:
                        noise_expected = {
                            f"signal_{index}_I" for index in range(REALIZATIONS)}
                        noise_actual = set(noise_product.hdus)
                        noise_missing = sorted(noise_expected - noise_actual)
                        noise_additional = sorted(noise_actual - noise_expected)
                        noise_membership = primary_observation_membership(
                            noise_product.hdul[0].header)
                        noise_membership_expected, noise_membership_detail = \
                            primary_membership_matches_record(
                                noise_membership, record, case)
                        book.add(
                            f"request_union.{case['id']}."
                            f"{map_identifier(record)}.noise_primary_membership",
                            noise_membership_expected
                            and noise_membership == membership,
                            {**noise_membership_detail,
                             "noise_primary_membership": noise_membership,
                             "map_primary_membership": membership,
                             "map_noise_membership_exact":
                                 noise_membership == membership})
                        noise_plane_inventory = [hdu_inventory_record(
                            "PRIMARY", noise_product.hdul[0],
                            "required_metadata_primary", reference_wcs,
                            noise_membership, str(record["noise_sha256"])), *[
                            hdu_inventory_record(
                                name, noise_product.hdus[name],
                                ("required_noise_realization_product"
                                 if name in noise_expected else
                                 "additional_returned_noise_product"),
                                reference_wcs, noise_membership,
                                str(record["noise_sha256"]))
                            for name in sorted(noise_actual)
                        ]]
                        book.add(
                            f"request_union.{case['id']}."
                            f"{map_identifier(record)}.noise_required",
                            not noise_missing and not noise_additional,
                            {"missing": noise_missing,
                             "additional": noise_additional})
                    finally:
                        noise_product.close()
                returned_maps.append({
                    "map_identity": map_identifier(record),
                    "map": record["map"], "map_sha256": record["map_sha256"],
                    "required_extensions": sorted(expected),
                    "actual_extensions": sorted(actual), "missing": missing,
                    "forbidden_extensions": sorted(forbidden),
                    "forbidden_present": forbidden_present,
                    "primary_observation_membership": membership,
                    "map_plane_inventory": plane_inventory,
                    "additional_classified": additional,
                    "noise": record.get("noise"),
                    "noise_sha256": record.get("noise_sha256"),
                    "noise_primary_observation_membership": (
                        noise_membership),
                    "noise_plane_inventory": noise_plane_inventory,
                })
            finally:
                product.close()
        known_paths = {
            str(Path(record[key]).resolve()): classification
            for record in frozen_case["maps"]
            for key, classification in (("map", "map_product"),
                                        ("noise", "noise_realization_product"))
            if record.get(key)
        }
        known_paths.update({
            str(Path(frozen_case[key]).resolve()): classification
            for key, classification in (
                ("merged_config", "merged_config"),
                ("raw_input_manifest", "raw_input_manifest"),
                ("mapmaking_provenance", "mapmaking_provenance"),
                ("coadd_provenance", "coadd_provenance"),
                ("noise_products_provenance", "noise_products_provenance"))
        })
        known_paths.update({
            str(Path(frozen_case[key]["path"]).resolve()): classification
            for key, classification in (
                ("runtime_merged_config", "runtime_merged_config"),
                ("config_source_manifest", "config_source_manifest"))
        })
        known_paths.update({str(Path(item["path"]).resolve()): "complete_log"
                            for item in frozen_case["logs"]})
        diagnostic_suffixes = {
            "hist": "map_histogram", "psd": "map_psd",
            "mapdiag": "map_diagnostics",
        }
        for item in frozen_case["physical_inventory"]:
            name = Path(str(item["path"])).name
            match = re.fullmatch(
                r".*_(hist|psd|mapdiag)(?:_filtered)?\.nc", name)
            if match:
                known_paths[str(Path(str(item["path"])).resolve())] = \
                    diagnostic_suffixes[match.group(1)]
        returned_files = [
            {**item, "classification": known_paths.get(
                str(Path(item["path"]).resolve()), "additional_returned_file")}
            for item in frozen_case["physical_inventory"]
        ]
        case_outputs.append({
            "case_id": case["id"], "mode": case["mode"],
            "expected_observations": case["expected_observations"],
            "arrays": list(ARRAYS), "expected_counts": case["expected_counts"],
            "observation_required_extensions": observation_required,
            "observation_forbidden_extensions": sorted(set(observation_forbidden)),
            "observation_realization_indices": (
                list(range(REALIZATIONS)) if case["products_enabled"] else []),
            "coadd_required_extensions": coadd_required,
            "coadd_forbidden_extensions": list(COADD_FORBIDDEN),
            "coadd_realization_indices": (
                list(range(REALIZATIONS)) if case["coadd"] else []),
            "returned_maps": returned_maps,
            "returned_files": returned_files,
        })
    write_new(output / "request-union.json", json_bytes({
        "schema_version": "sci-map-001-request-specific-union-v1",
        "request_id": REQUEST_ID, "candidate_sha": CANDIDATE_SHA,
        "cases": case_outputs,
        "interpretation": "request-specific required/additional/absent product union",
    }))


def population_noise_statistics(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                np.ndarray]:
    if noise.ndim != 3 or noise.shape[-1] != REALIZATIONS:
        die(f"population noise statistics require (...,{REALIZATIONS}), got {noise.shape}")
    realization_finite = np.isfinite(noise)
    all_finite = np.all(realization_finite, axis=-1)
    mean = np.full(noise.shape[:2], np.nan, dtype=np.float64)
    variance = np.full(noise.shape[:2], np.nan, dtype=np.float64)
    values = noise[all_finite]
    if values.size:
        means = np.mean(values, axis=1)
        centered = values - means[:, None]
        mean[all_finite] = means
        variance[all_finite] = np.mean(centered * centered, axis=1)
    return mean, variance, realization_finite


def finite_blank_domain(signal: np.ndarray, weight: np.ndarray,
                        exposure: np.ndarray) -> np.ndarray:
    return np.isfinite(signal) & np.isfinite(weight) & np.isfinite(exposure) & \
        (weight > 0.0) & (exposure > 0.0)


def angular_unit_arcsec_scale(unit: str, label: str) -> float:
    normalized = unit.strip().lower()
    scales = {
        "arcsec": 1.0, "arcsecond": 1.0, "arcseconds": 1.0,
        "arcmin": 60.0, "deg": 3600.0, "degree": 3600.0,
        "degrees": 3600.0, "rad": 180.0 * 3600.0 / math.pi,
        "radian": 180.0 * 3600.0 / math.pi,
        "mas": 1.0e-3,
    }
    if normalized not in scales:
        die(f"{label}: unsupported angular unit {unit!r}")
    return scales[normalized]


def projection_for_diagnostic(
        authority: RawManifestAuthority,
        record: Mapping[str, Any]) -> Mapping[str, Any]:
    array = str(record["array"])
    if record["scope"] == "observation":
        return authority.memberships[(int(record["obsnum"]), array)]["projection"]
    projections = [authority.memberships[(obsnum, array)]["projection"]
                   for obsnum in EXPECTED_CASES[str(record["case_id"])]
                   ["expected_observations"]]
    first = projections[0]
    for projection in projections[1:]:
        if not exact_float_equal(projection["_fwhm_arcsec"],
                                 first["_fwhm_arcsec"]) or \
                not exact_float_equal(projection["_target_axis1"],
                                      first["_target_axis1"]) or \
                not exact_float_equal(projection["_target_axis2"],
                                      first["_target_axis2"]) or \
                projection["frame"] != first["frame"] or \
                projection["target"]["unit"] != first["target"]["unit"]:
            die(f"{record['case_id']} coadd {array}: observation target/FWHM authorities differ")
    return first


def target_distance_arcsec(
        header: fits.Header, shape: tuple[int, int], mode: str,
        projection: Mapping[str, Any],
        primary_header: fits.Header | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    rows, cols = shape
    row_grid, internal_col_grid = np.mgrid[0:rows, 0:cols]
    fits_col = (cols - 1 - internal_col_grid).astype(np.float64)
    fits_row = row_grid.astype(np.float64)
    if mode == "science":
        celestial = WCS(header).celestial
        longitude, latitude = celestial.all_pix2world(fits_col, fits_row, 0)
        target_longitude = float(projection["_target_axis1"])
        target_latitude = float(projection["_target_axis2"])
        longitude_rad = np.deg2rad(longitude)
        latitude_rad = np.deg2rad(latitude)
        target_longitude_rad = math.radians(target_longitude)
        target_latitude_rad = math.radians(target_latitude)
        delta_longitude = (longitude_rad - target_longitude_rad + math.pi) % \
            (2.0 * math.pi) - math.pi
        haversine = np.sin((latitude_rad - target_latitude_rad) / 2.0) ** 2 + \
            np.cos(latitude_rad) * math.cos(target_latitude_rad) * \
            np.sin(delta_longitude / 2.0) ** 2
        distance = np.rad2deg(2.0 * np.arcsin(
            np.sqrt(np.clip(haversine, 0.0, 1.0)))) * 3600.0
        center_longitude, center_latitude = celestial.all_pix2world(
            float(header.get("CRPIX1", 1.0)) - 1.0,
            float(header.get("CRPIX2", 1.0)) - 1.0, 0)
        center_longitude_residual = abs(
            (float(center_longitude) - binary32_value(target_longitude) + 180.0) %
            360.0 - 180.0)
        center_latitude_residual = abs(
            float(center_latitude) - binary32_value(target_latitude))
        return distance, {
            "method": "great_circle_celestial_wcs",
            "coordinate_frame": "fk5", "input_unit": "deg",
            "output_unit": "arcsec", "pixel_origin": 0,
            "configured_target": {
                "axis1": target_longitude, "axis2": target_latitude,
                "unit": projection["target"]["unit"],
            },
            "wcs_reference_world": [float(center_longitude), float(center_latitude)],
            "binary32_adapter_target": [binary32_value(target_longitude),
                                         binary32_value(target_latitude)],
            "configured_target_center_bound":
                center_longitude_residual <= 1.0e-12 and
                center_latitude_residual <= 1.0e-12,
        }

    wcs = WCS(header)
    points = np.empty((rows * cols, wcs.pixel_n_dim), dtype=np.float64)
    points[:, 0] = fits_col.ravel()
    points[:, 1] = fits_row.ravel()
    center = np.empty((1, wcs.pixel_n_dim), dtype=np.float64)
    for axis in range(wcs.pixel_n_dim):
        reference = float(header.get(f"CRPIX{axis + 1}", 1.0)) - 1.0
        if axis >= 2:
            points[:, axis] = reference
        center[:, axis] = reference
    world = np.asarray(wcs.all_pix2world(points, 0), dtype=np.float64)
    center_world = np.asarray(wcs.all_pix2world(center, 0), dtype=np.float64)[0]
    unit1 = str(header.get("CUNIT1", ""))
    unit2 = str(header.get("CUNIT2", ""))
    delta_axis1 = (world[:, 0] - center_world[0]) * \
        angular_unit_arcsec_scale(unit1, "Point WCS axis 1")
    delta_axis2 = (world[:, 1] - center_world[1]) * \
        angular_unit_arcsec_scale(unit2, "Point WCS axis 2")
    distance = np.hypot(delta_axis1, delta_axis2).reshape(shape)
    expected_source_radians = (
        math.radians(float(projection["_target_axis1"])),
        math.radians(float(projection["_target_axis2"])),
    )
    primary_values: dict[str, float] = {}
    if primary_header is not None:
        try:
            primary_values = {
                key: float(primary_header[key])
                for key in ("SRC_RA", "SRC_DEC", "TAN_RA", "TAN_DEC")
            }
        except (KeyError, TypeError, ValueError):
            primary_values = {}
    absolute_target_bound = len(primary_values) == 4 and all(
        math.isfinite(value) for value in primary_values.values()) and \
        abs(primary_values["SRC_RA"] - expected_source_radians[0]) <= 1.0e-12 and \
        abs(primary_values["SRC_DEC"] - expected_source_radians[1]) <= 1.0e-12 and \
        abs(primary_values["TAN_RA"] - expected_source_radians[0]) <= 1.0e-12 and \
        abs(primary_values["TAN_DEC"] - expected_source_radians[1]) <= 1.0e-12 and \
        exact_float_equal(primary_values["SRC_RA"], primary_values["TAN_RA"]) and \
        exact_float_equal(primary_values["SRC_DEC"], primary_values["TAN_DEC"])
    return distance, {
        "method": "euclidean_declared_tangent_plane_axes",
        "coordinate_frame": "altaz", "axis_units": [unit1, unit2],
        "output_unit": "arcsec", "pixel_origin": 0,
        "configured_target_absolute_authority": {
            "axis1": float(projection["_target_axis1"]),
            "axis2": float(projection["_target_axis2"]),
            "unit": projection["target"]["unit"],
        },
        "tangent_plane_center_world": [float(center_world[0]),
                                         float(center_world[1])],
        "primary_header_absolute_target_radians": primary_values,
        "raw_configured_target_radians": list(expected_source_radians),
        "primary_source_tangent_target_bound": absolute_target_bound,
        "configured_target_center_bound":
            abs(float(center_world[0])) <= 1.0e-12 and
            abs(float(center_world[1])) <= 1.0e-12 and absolute_target_bound,
    }


def connected_region_records(mask: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    records: list[dict[str, Any]] = []
    for label_index, region_slice in enumerate(ndimage.find_objects(labels), 1):
        if region_slice is None:
            continue
        local_mask = labels[region_slice] == label_index
        local_values = values[region_slice][local_mask]
        peak_position = int(np.argmax(np.abs(local_values)))
        records.append({
            "label": label_index,
            "pixel_count": int(np.count_nonzero(local_mask)),
            "peak_z": float(local_values[peak_position]),
            "max_abs_z": float(abs(local_values[peak_position])),
            "bounding_box": {
                "row_start": int(region_slice[0].start),
                "row_stop_exclusive": int(region_slice[0].stop),
                "col_start": int(region_slice[1].start),
                "col_stop_exclusive": int(region_slice[1].stop),
            },
        })
    return records


def compute_statistical_diagnostics(
        output: Path, cases: Mapping[str, Mapping[str, Any]],
        reconstructions: Mapping[str, Mapping[tuple[int, str], Reconstruction]],
        raw_authorities: Mapping[str, RawManifestAuthority],
        book: CheckBook) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                  list[dict[str, Any]]]:
    blank_records: list[dict[str, Any]] = []
    false_z_records: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        mode = EXPECTED_CASES[case_id]["mode"]
        case_record = cases[case_id]
        for original_record in case_record["maps"]:
            selected = (case_id.startswith("P-") and
                        original_record["scope"] == "observation") or \
                (case_id.startswith("S-E-") and
                 original_record["scope"] == "observation") or \
                (case_id.startswith("S-C-") and
                 original_record["scope"] == "coadd")
            if not selected:
                continue
            record = dict(original_record)
            record["case_id"] = case_id
            arrays, noise, header = load_map_arrays(record)
            identity = map_identifier(record)
            prefix = f"diagnostics.{case_id}.{identity}"
            if noise is None:
                book.add(prefix + ".realizations", False,
                         "SCI-MAP-001-UNITY-001-EG-DIAGNOSTIC-REALIZATIONS: "
                         "the protocol-required serialized realization file is absent",
                         evidence_gap=True)
                continue
            signal = np.asarray(arrays["signal_I"], dtype=np.float64)
            formal_weight = arrays.get("weight_formal_I")
            weight = np.asarray(formal_weight if formal_weight is not None
                                else arrays["weight_I"], dtype=np.float64)
            exposure = np.asarray(arrays["coverage_I"], dtype=np.float64)
            if formal_weight is not None:
                bitwise_weights = formal_weight.dtype == arrays["weight_I"].dtype and \
                    formal_weight.shape == arrays["weight_I"].shape and \
                    formal_weight.tobytes() == arrays["weight_I"].tobytes()
                book.add(prefix + ".formal_weight_equals_weight_bitwise",
                         bitwise_weights, None)
            finite_domain = finite_blank_domain(signal, weight, exposure)
            edge_distance = ndimage.distance_transform_edt(
                np.pad(finite_domain, 1, mode="constant", constant_values=False)
            )[1:-1, 1:-1]
            projection = projection_for_diagnostic(raw_authorities[case_id], record)
            product = FitsProduct.open(Path(record["map"]))
            try:
                primary_header = product.hdul[0].header.copy()
                persisted_fwhm = product.hdus["kernel_I"].header.get("FWHM")
            finally:
                product.close()
            target_distance, target_metadata = target_distance_arcsec(
                header, signal.shape, mode, projection, primary_header)
            if record["scope"] == "observation":
                reconstruction = reconstructions[case_id].get(
                    (int(record["obsnum"]), str(record["array"])))
                projection_bound = reconstruction is not None and \
                    reconstruction.ledger_identity.get("bundle_identity_digest") == \
                    projection["identity_digest"] and \
                    reconstruction.ledger_identity.get("raw_membership_complete") is True
            else:
                # Coadd admission checks independently bind each observation
                # projection identity before centered-grid arithmetic.
                projection_bound = True
            book.add(prefix + ".configured_target_center_binding",
                     bool(target_metadata["configured_target_center_bound"])
                     and projection_bound,
                     {"target_distance_authority": target_metadata,
                      "raw_projection_identity": projection["identity_digest"],
                      "projection_chain_bound": projection_bound})
            fwhm_arcsec = float(projection["_fwhm_arcsec"])
            try:
                persisted_fwhm_value = float(persisted_fwhm)
            except (TypeError, ValueError):
                persisted_fwhm_value = math.nan
            book.add(prefix + ".pre_run_fwhm_authority",
                     exact_float_equal(persisted_fwhm_value, fwhm_arcsec),
                     {"pre_run_fwhm_arcsec": fwhm_arcsec,
                      "kernel_header_FWHM": persisted_fwhm})
            target_exclusion = target_distance >= 5.0 * fwhm_arcsec
            background = finite_domain & (edge_distance >= 5.0) & target_exclusion
            background_count = int(np.count_nonzero(background))
            book.add(prefix + ".background_nonempty", background_count > 0,
                     {"background_pixel_count": background_count,
                      "F_pixel_count": int(np.count_nonzero(finite_domain)),
                      "edge_guard_pixels": 5.0,
                      "target_exclusion_radius_fwhm": 5.0,
                      "target_exclusion_radius_arcsec": 5.0 * fwhm_arcsec,
                      "F_definition": "finite(signal,q,coverage) and q>0 and coverage>0; "
                                      "no persisted validity/policy mask"},
                     evidence_gap=background_count == 0)

            population_mean, variance, realization_finite = \
                population_noise_statistics(noise)
            if "noise_variance_I" in arrays:
                passed, detail = numeric_close(arrays["noise_variance_I"], variance)
                book.add(prefix + ".persisted_population_variance_full_image",
                         passed, detail)
            q_times_variance = np.full(signal.shape, np.nan, dtype=np.float64)
            qv_defined = np.isfinite(weight) & np.isfinite(variance)
            q_times_variance[qv_defined] = weight[qv_defined] * variance[qv_defined]
            z_formal = np.full(signal.shape, np.nan, dtype=np.float64)
            formal_defined = np.isfinite(signal) & np.isfinite(weight) & (weight > 0.0)
            z_formal[formal_defined] = signal[formal_defined] * \
                np.sqrt(weight[formal_defined])
            z_empirical = np.full(signal.shape, np.nan, dtype=np.float64)
            empirical_defined = np.isfinite(signal) & np.isfinite(variance) & \
                (variance > 0.0)
            z_empirical[empirical_defined] = signal[empirical_defined] / \
                np.sqrt(variance[empirical_defined])

            flat_indices = np.flatnonzero(background)
            pixel_rows, pixel_cols = np.unravel_index(flat_indices, signal.shape)
            background_realizations = noise.reshape(-1, REALIZATIONS)[flat_indices].T
            background_entries_finite = bool(np.all(np.isfinite(background_realizations)))
            book.add(prefix + ".background_realizations_finite",
                     background_entries_finite,
                     {"nonfinite_entry_count": int(np.count_nonzero(
                         ~np.isfinite(background_realizations)))})
            if background_count and background_entries_finite:
                realization_means = np.mean(background_realizations, axis=0)
                centered = background_realizations - realization_means[None, :]
                factor = centered / math.sqrt(float(REALIZATIONS))
                covariance_diagonal = np.mean(centered * centered, axis=0)
                variance_on_background = variance[background]
                diagonal_equal = np.array_equal(covariance_diagonal,
                                                variance_on_background)
                book.add(prefix + ".covariance_diagonal_equals_population_variance",
                         diagonal_equal,
                         {"column_count": background_count,
                          "bitwise_equal": diagonal_equal})
                zero_variance = covariance_diagonal == 0.0
                correlation_defined = np.isfinite(covariance_diagonal) & \
                    (covariance_diagonal > 0.0)
                correlation_factor = np.full(factor.shape, np.nan, dtype=np.float64)
                correlation_factor[:, correlation_defined] = \
                    factor[:, correlation_defined] / \
                    np.sqrt(covariance_diagonal[correlation_defined])[None, :]
                gram = factor @ factor.T
                eigenvalues = np.linalg.eigvalsh(gram)
                trace_c = float(np.sum(factor * factor, dtype=np.float64))
                trace_c2 = float(np.sum(gram * gram, dtype=np.float64))
                participation = (trace_c * trace_c / trace_c2
                                 if trace_c2 > 0.0 else None)
                rank_threshold = np.finfo(float).eps * max(
                    float(np.max(np.abs(eigenvalues))), 1.0)
                covariance_rank = int(np.count_nonzero(eigenvalues > rank_threshold))
            else:
                realization_means = np.full((background_count,), np.nan,
                                            dtype=np.float64)
                centered = np.full((REALIZATIONS, background_count), np.nan,
                                   dtype=np.float64)
                factor = np.full((REALIZATIONS, background_count), np.nan,
                                 dtype=np.float64)
                covariance_diagonal = np.full((background_count,), np.nan,
                                              dtype=np.float64)
                correlation_factor = np.full((REALIZATIONS, background_count),
                                             np.nan, dtype=np.float64)
                correlation_defined = np.zeros((background_count,), dtype=bool)
                zero_variance = np.zeros((background_count,), dtype=bool)
                gram = np.empty((0, 0), dtype=np.float64)
                eigenvalues = np.empty((0,), dtype=np.float64)
                participation = None
                covariance_rank = 0

            covariance_path = output / f"{case_id}-{identity}-covariance-factor.npz"
            write_npz_new(
                covariance_path,
                schema_version=np.array("sci-map-001-background-covariance-factor-v2"),
                case_id=np.array(case_id), map_identity=np.array(identity),
                realization_count=np.array(REALIZATIONS, dtype=np.int64),
                population_normalization=np.array(REALIZATIONS, dtype=np.int64),
                signal=np.asarray(signal, dtype=np.float64),
                formal_weight=np.asarray(weight, dtype=np.float64),
                coverage=np.asarray(exposure, dtype=np.float64),
                realization_population_mean=population_mean,
                realization_population_variance=variance,
                realization_finite_mask=realization_finite.astype(np.uint8),
                q_times_population_variance=q_times_variance,
                z_formal=z_formal, z_empirical=z_empirical,
                finite_domain_mask=finite_domain.astype(np.uint8),
                target_distance_arcsec=target_distance,
                edge_distance_pixels=edge_distance,
                background_mask=background.astype(np.uint8),
                background_linear_indices=flat_indices.astype(np.int64),
                background_rows=np.asarray(pixel_rows, dtype=np.int64),
                background_cols=np.asarray(pixel_cols, dtype=np.int64),
                background_realizations=np.asarray(background_realizations,
                                                   dtype=np.float64),
                background_realization_means=realization_means,
                centered_realizations=np.asarray(centered, dtype=np.float64),
                centered_covariance_factor=np.asarray(factor, dtype=np.float64),
                covariance_diagonal=np.asarray(covariance_diagonal, dtype=np.float64),
                correlation_factor=np.asarray(correlation_factor, dtype=np.float64),
                correlation_defined_columns=correlation_defined.astype(np.uint8),
                zero_variance_column_indices=np.flatnonzero(zero_variance).astype(np.int64),
                realization_gram=np.asarray(gram, dtype=np.float64),
                gram_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            )
            blank_records.append({
                "case_id": case_id, "map_identity": identity,
                "scope": record["scope"], "obsnum": record["obsnum"],
                "array": record["array"],
                "finite_domain_pixel_count": int(np.count_nonzero(finite_domain)),
                "background_pixel_count": background_count,
                "background_state": "defined" if background_count else
                    "SCI-MAP-001-UNITY-001-EG-BACKGROUND-NONEMPTY",
                "edge_guard_pixels": 5.0,
                "target_exclusion_radius_fwhm": 5.0,
                "target_exclusion_radius_arcsec": 5.0 * fwhm_arcsec,
                "target_distance": target_metadata,
                "signal": finite_summary(signal[background]),
                "q_times_population_variance":
                    finite_summary(q_times_variance[background]),
                "noise_population_variance": finite_summary(variance[background]),
                "z_formal": finite_summary(z_formal[background]),
                "z_empirical": finite_summary(z_empirical[background]),
                "covariance_factor": str(covariance_path),
                "covariance_factor_sha256": sha256(covariance_path),
                "covariance_rank": covariance_rank,
                "zero_variance_correlation_column_count":
                    int(np.count_nonzero(zero_variance)),
                "effective_eigenvalue_participation": participation,
                "distributional_acceptance": "not_authorized_without_SCI-NOI_contract",
            })

            for z_identity, z_values in (
                    ("z_formal", z_formal), ("z_empirical", z_empirical)):
                eligible = background & np.isfinite(z_values)
                false_mask = eligible & (np.abs(z_values) > 5.0)
                regions = connected_region_records(false_mask, z_values)
                eligible_count = int(np.count_nonzero(eligible))
                exceedance_count = int(np.count_nonzero(false_mask))
                false_z_records.append({
                    "case_id": case_id, "map_identity": identity,
                    "z_identity": z_identity,
                    "background_pixel_count": background_count,
                    "eligible_pixel_count": eligible_count,
                    "threshold_abs_z": 5.0,
                    "exceedance_pixel_count": exceedance_count,
                    "exceedance_rate": (exceedance_count / eligible_count
                                        if eligible_count else None),
                    "component_count": len(regions), "regions": regions,
                    "connectivity": "eight-neighbour",
                    "interpretation": "characterization_only_no_null_tail_acceptance",
                })

            edge_domain = finite_domain & target_exclusion
            for lower, upper in EDGE_BINS:
                bin_domain = edge_domain & (edge_distance >= lower) & \
                    (edge_distance < upper)
                for z_identity, z_values in (
                        ("z_formal", z_formal), ("z_empirical", z_empirical)):
                    eligible = bin_domain & np.isfinite(z_values)
                    exceedance = eligible & (np.abs(z_values) > 5.0)
                    eligible_count = int(np.count_nonzero(eligible))
                    exceedance_count = int(np.count_nonzero(exceedance))
                    edge_rows.append({
                        "case_id": case_id, "scope": record["scope"],
                        "obsnum": record["obsnum"] if record["obsnum"] is not None
                        else "coadd", "array": record["array"],
                        "z_identity": z_identity,
                        "edge_lower_inclusive_pixels": lower,
                        "edge_upper_exclusive_pixels": upper if math.isfinite(upper)
                        else "inf",
                        "target_excluded_domain_count": int(np.count_nonzero(bin_domain)),
                        "eligible_count": eligible_count,
                        "exceedance_count": exceedance_count,
                        "exceedance_rate": (exceedance_count / eligible_count
                                            if eligible_count else None),
                        "component_count": len(connected_region_records(
                            exceedance, z_values)),
                    })
    return blank_records, false_z_records, edge_rows


def write_characterization_outputs(
        output: Path, inputs: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]],
        blank_records: Sequence[Mapping[str, Any]],
        false_z_records: Sequence[Mapping[str, Any]],
        edge_rows: Sequence[Mapping[str, Any]],
        book: CheckBook, baseline_records: Sequence[Mapping[str, Any]]) -> None:
    write_new(output / "wcs.json", json_bytes({
        "schema_version": "sci-map-001-wcs-verification-v1",
        "maps": [{"case_id": item["case_id"], "scope": item["scope"],
                  "obsnum": item["obsnum"], "array": item["array"],
                  "shape": item["shape"], "wcs": item["wcs"]}
                 for item in summaries],
        "checks": [item for item in book.checks if "wcs" in item["id"].lower()],
    }))
    pixel_tokens = ("reconstruct", "recombine", "alias", "valid",
                    "threshold", "digest", "identity", "empirical",
                    "formal_standardized", "scan_farm")
    write_new(output / "pixel-identities.json", json_bytes({
        "schema_version": "sci-map-001-pixel-identities-v1",
        "checks": [item for item in book.checks
                   if any(token in item["id"].lower() for token in pixel_tokens)],
    }))
    write_new(output / "blank-summary.json", json_bytes({
        "schema_version": "sci-map-001-blank-characterization-v1",
        "maps": list(blank_records),
        "interpretation": "characterization_only_no_distributional_acceptance",
    }))
    write_new(output / "false-z-regions.json", json_bytes({
        "schema_version": "sci-map-001-false-z-regions-v1",
        "maps": list(false_z_records),
    }))
    edge_path = output / "edge-tables.csv"
    try:
        with edge_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            fields = (
                "case_id", "scope", "obsnum", "array", "z_identity",
                "edge_lower_inclusive_pixels", "edge_upper_exclusive_pixels",
                "target_excluded_domain_count", "eligible_count",
                "exceedance_count", "exceedance_rate", "component_count",
            )
            writer.writerow(fields)
            writer.writerows([[row[field] for field in fields] for row in edge_rows])
        os.chmod(edge_path, 0o444)
    except (OSError, FileExistsError) as exc:
        raise EvidenceError(f"cannot write edge table: {exc}") from exc
    write_new(output / "coadd-recombination.json", json_bytes({
        "schema_version": "sci-map-001-coadd-recombination-v1",
        "checks": [item for item in book.checks
                   if ".coadd." in item["id"] or ".recombine." in item["id"]],
    }))
    log_checks = [item for item in book.checks if any(
        token in item["id"] for token in
        (".log.", ".exit_status", "_completion", "baseline."))]
    inventories = [{
        "case_id": case["case_id"],
        "ordered_by_mtime_ns_then_path": sorted(
            case["physical_inventory"],
            key=lambda item: (item["mtime_ns"], item["relative_path"])),
    } for case in inputs["cases"]]
    write_new(output / "log-completion.json", json_bytes({
        "schema_version": "sci-map-001-log-completion-v1",
        "checks": log_checks, "baseline_tool_runs": list(baseline_records),
        "physical_inventories": inventories,
    }))


def write_digest_manifest(output: Path) -> None:
    digest_path = output / "SHA256SUMS"
    if digest_path.exists():
        die(f"refusing to replace digest manifest: {digest_path}")
    files = sorted(path for path in output.rglob("*")
                   if path.is_file() and path != digest_path)
    payload = "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    write_new(digest_path, payload.encode())


def write_residual_manifest(output: Path, book: CheckBook) -> None:
    records = []
    seen: set[Path] = set()
    for check in book.checks:
        if not str(check.get("id", "")).endswith("lossless_residuals"):
            continue
        detail = check.get("detail")
        if not isinstance(detail, Mapping):
            die("lossless residual check lacks a manifested path/digest")
        path = Path(str(detail.get("path", "")))
        digest = detail.get("sha256")
        array_keys = detail.get("array_keys")
        if not path.is_absolute() or path in seen or not path.is_file() or \
                digest != sha256(path):
            die("lossless residual artifact binding is absent or repeated")
        if not isinstance(array_keys, list) or not array_keys or \
                any(not isinstance(key, str) or not key for key in array_keys) or \
                len(array_keys) != len(set(array_keys)):
            die("lossless residual artifact has invalid array-key authority")
        seen.add(path)
        records.append({
            "check_id": check["id"], "path": str(path),
            "sha256": digest, "size_bytes": path.stat().st_size,
            "array_keys": array_keys,
        })
    if not records:
        die("lossless residual manifest has no artifacts")
    write_new(output / "residual-manifest.json", json_bytes({
        "schema_version": "sci-map-001-lossless-residual-manifest-v1",
        "request_id": REQUEST_ID, "candidate_sha": CANDIDATE_SHA,
        "records": sorted(records, key=lambda item: item["check_id"]),
    }))


def run_analysis(args: argparse.Namespace) -> int:
    inputs_path = args.inputs.resolve()
    inputs = load_analysis_inputs(inputs_path)
    campaign_path = Path(inputs["campaign"])
    campaign, _ = load_campaign(campaign_path)
    if args.request_root is not None and \
            args.request_root.resolve() != Path(inputs.get("request_root", "")).resolve():
        die("run --request-root differs from the frozen analysis inputs")
    if args.product_contracts is not None and \
            args.product_contracts.resolve() != Path(inputs["product_contracts"]).resolve():
        die("run --product-contracts differs from the frozen analysis inputs")
    source_root = (args.source_root or repository_root()).resolve()
    if run_git(source_root, "rev-parse", "HEAD") != CANDIDATE_SHA:
        die("analysis source root is not the exact repair candidate")
    validate_successor_registries(
        campaign, campaign_path, source_root, args.profile_registry,
        args.accepted_runs, args.point_contract, args.science_contract)
    python_path = Path(args.python)
    if not python_path.is_absolute() or not python_path.is_file():
        die("run Python interpreter must be an existing absolute file")
    contracts = load_contracts(campaign, campaign_path,
                               Path(inputs["product_contracts"]))
    output = args.output.resolve()
    if output.exists():
        die(f"refusing to reuse analysis output directory: {output}")
    output.mkdir(parents=True)
    book = CheckBook()
    cases = {record["case_id"]: record for record in inputs["cases"]}
    summaries: list[dict[str, Any]] = []
    reconstructions: dict[str, dict[tuple[int, str], Reconstruction]] = {}
    raw_authorities: dict[str, RawManifestAuthority] = {}
    coadd_provenances: dict[str, Mapping[str, Any]] = {}
    reduction_roots = tuple(Path(record["reduction_root"]).resolve()
                            for record in inputs["cases"])

    for case in campaign["cases"]:
        case_id = case["id"]
        case_record = cases[case_id]
        raw_authority = verify_case_frozen_inputs(
            case_record, case, book, reduction_roots,
            campaign, Path(inputs["product_contracts"]), contracts)
        raw_authorities[case_id] = raw_authority
        mapmaking = read_yaml(Path(case_record["mapmaking_provenance"]))
        coadd = read_yaml(Path(case_record["coadd_provenance"]))
        coadd_provenances[case_id] = coadd
        case_reconstruction: dict[tuple[int, str], Reconstruction] = {}
        # Observations first so the coadd lane can use their independently
        # reconstructed, already-checked bundles atomically.
        ordered_records = sorted(case_record["maps"],
                                 key=lambda item: (item["scope"] == "coadd",
                                                   item.get("obsnum") or 0,
                                                   ARRAYS.index(item["array"])))
        for record in ordered_records:
            summary, reconstruction = verify_map(record, case, contracts,
                                                 mapmaking, coadd,
                                                 raw_authority, output, book)
            summaries.append(summary)
            if record["scope"] == "observation" and reconstruction:
                case_reconstruction[(int(record["obsnum"]), record["array"])] = reconstruction
        reconstructions[case_id] = case_reconstruction

    recombination_gates = verify_sc_se_recombination_preconditions(
        output, cases, reconstructions, book)
    for case in campaign["cases"]:
        case_id = case["id"]
        if not case["coadd"]:
            continue
        if case_id.startswith("S-C-") and not recombination_gates.get(case_id, False):
            book.add(f"{case_id}.coadd.recombination_precondition", False,
                     "SCI-MAP-001-UNITY-001-EG-SC-SE-PRECONDITION: exact "
                     "observation/realization/A_floor precondition failed",
                     evidence_gap=True)
            continue
        for array in ARRAYS:
            reconstruct_coadd(
                output, case, cases[case_id]["maps"], reconstructions[case_id],
                coadd_provenances[case_id], array, book)

    # All science cases are the same ordered raw inputs. Exact ledger digests
    # prove the comparison is not confounded by a different extraction.
    science_case_ids = ("S-C-SEQ", "S-C-OMP", "S-E-SEQ", "S-E-OMP", "S-X-SEQ")
    for obsnum in (152390, 152392):
        for array in ARRAYS:
            digests = []
            for case_id in science_case_ids:
                record = map_record_index(cases[case_id])[("observation", obsnum, array)]
                digests.append(record.get("sample_ledger_sha256"))
            book.add(f"science_inputs.{obsnum}.{array}.ledger_digest_exact",
                     digests[0] is not None and len(set(digests)) == 1,
                     dict(zip(science_case_ids, digests)))
    compare_cases(cases, output, book)
    verify_scan_farm_pairs(cases, reconstructions, book)

    write_request_union(output, campaign, cases, book)
    blank_records, false_z_records, edge_rows = compute_statistical_diagnostics(
        output, cases, reconstructions, raw_authorities, book)

    baseline_records = run_baseline_tools(
        source_root, campaign, cases, output, args.python,
        args.audit_tool, args.compare_tool, book)
    write_characterization_outputs(
        output, inputs, summaries, blank_records, false_z_records,
        edge_rows, book, baseline_records)
    write_residual_manifest(output, book)

    if book.evidence_gaps:
        analysis_result = ("local_analysis_failed_with_evidence_gaps"
                           if book.hard_failures else
                           "local_analysis_evidence_gap")
    elif book.hard_failures:
        analysis_result = "local_analysis_failed"
    else:
        analysis_result = "local_analysis_pass"
    runtime_versions = {
        "python": sys.version,
        **{package: importlib.metadata.version(package)
           for package in ("numpy", "scipy", "astropy", "PyYAML")},
    }
    report = {
        "schema_version": "sci-map-unity-verification-v1",
        "program_schema": PROGRAM_SCHEMA,
        "request_id": REQUEST_ID,
        "candidate_sha": CANDIDATE_SHA,
        "analysis_inputs": str(inputs_path),
        "analysis_inputs_sha256": sha256(inputs_path),
        "result": analysis_result,
        "runtime_versions": runtime_versions,
        "conformance_claim": "none_independent_reaudit_required",
        "external_evidence_claim": "returned_files_unreviewed_until_owner_retrieval_and_independent_audit",
        "finding_state": {
            "F009": "addressed_pending_reaudit",
            "F010": "addressed_pending_reaudit",
            "F012": "outstanding_until_human_run_bundle_is_independently_audited",
            "F013": "conditioned_on_named_upstream_audits",
        },
        "dependency_nonclosure": campaign["coordination_state"]["not_closed_by_this_campaign"],
        "conditioned_limitations": [{
            "id": "SCI-MAP-001-F013-ASTROMETRIC-EPOCH-AUTHORITY",
            "fact": "source_epoch and RADESYS are checked for internal identity/FITS consistency but are not fixed here to J2000 or independently bound to raw astrometric authority",
            "disposition": "absolute astrometric acceptance remains conditioned on SCI-AST-001 and F013",
        }],
        "checks": book.checks,
        "failures": book.failures,
        "hard_failures": book.hard_failures,
        "evidence_gaps": book.evidence_gaps,
    }
    write_new(output / "verification.json", json_bytes(report))
    write_new(output / "inventory.json", json_bytes({
        "schema_version": "sci-map-unity-verified-map-inventory-v1",
        "maps": summaries,
        "physical_inventories": [
            {"case_id": case["case_id"],
             "records": case["physical_inventory"]}
            for case in inputs["cases"]
        ],
    }))
    write_new(output / "baseline-tool-runs.json", json_bytes(baseline_records))
    write_new(output / "VERDICT.txt", (
        (analysis_result.replace("_", " ").upper() + "\n")
        + "No conformance or finding-closure claim is made. Independent re-audit is required.\n"
    ).encode())
    write_digest_manifest(output)
    print(json.dumps({"result": report["result"], "checks": len(book.checks),
                      "failures": len(book.failures), "output": str(output)},
                     sort_keys=True))
    if book.evidence_gaps:
        return 3
    return 0 if book.passed else 2


def exact_node(value: float) -> dict[str, str]:
    return {
        "numeric": format(float(value), ".17g"),
        "hex": float(value).hex(),
        "encoding": "binary64-max-digits10-and-c99-hexfloat",
    }


def synthetic_wcs_header(rows: int, cols: int, *, celestial: bool = True,
                         delta_row: int = 0, delta_col: int = 0) -> fits.Header:
    header = fits.Header()
    header["WCSAXES"] = 4
    header["CTYPE1"] = "RA---TAN" if celestial else "AZOFFSET"
    header["CTYPE2"] = "DEC--TAN" if celestial else "ELOFFSET"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "STOKES"
    header["CUNIT1"] = "deg" if celestial else "arcsec"
    header["CUNIT2"] = "deg" if celestial else "arcsec"
    header["CUNIT3"] = "Hz"
    header["CRPIX1"] = (cols + 1) / 2.0 + delta_col
    header["CRPIX2"] = (rows + 1) / 2.0 + delta_row
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["CRVAL1"] = 150.0 if celestial else 0.0
    header["CRVAL2"] = 2.0 if celestial else 0.0
    header["CRVAL3"] = 273.0e9
    header["CRVAL4"] = 1.0
    header["CDELT1"] = -1.0 / 3600.0 if celestial else -1.0
    header["CDELT2"] = 1.0 / 3600.0 if celestial else 1.0
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1.0
    if celestial:
        header["RADESYS"] = "FK5"
        header["EQUINOX"] = 2000.0
    return header


def write_synthetic_map(path: Path, reconstruction: Reconstruction,
                        contracts: Mapping[str, Any], *, coadd: bool = False,
                        celestial: bool = True,
                        delta_row: int = 0, delta_col: int = 0,
                        empirical: bool = False) -> None:
    rows, cols = reconstruction.planes["signal_I"].shape
    common = contracts["checks"]["sci_map_common_celestial_v1" if celestial
                                  else "sci_map_common_altaz_v1"]
    f010 = contracts["checks"]["sci_map_naive_coadd_v1" if coadd
                                else "sci_map_naive_observation_v1"]
    primary = fits.PrimaryHDU()
    primary.header["BUNIT"] = "mJy/beam"
    hdus: list[Any] = [primary]
    names = [*COMMON_PLANES, *(F010_COADD if coadd else F010_OBSERVATION)]
    for name in names:
        data2d = reconstruction.planes[name][:, ::-1]
        data = data2d[np.newaxis, np.newaxis, :, :]
        header = synthetic_wcs_header(rows, cols, celestial=celestial,
                                      delta_row=delta_row, delta_col=delta_col)
        unit = common.get("ext_bunits", {}).get(name,
               f010.get("ext_bunits", {}).get(name))
        if unit is not None:
            header["BUNIT"] = unit
            header["UNIT"] = unit
        metadata = {}
        metadata.update(common.get("required_ext_headers", {}).get(name, {}))
        metadata.update(f010.get("required_ext_headers", {}).get(name, {}))
        for key, value in metadata.items():
            header[key] = value
        if name == "weight_I":
            header["CALTYPE"] = "formal"
        if name in ("science_policy_support_I", "coverage_bool_I"):
            header["WTTHRESH"] = reconstruction.science_policy["threshold"]
        header["EXTNAME"] = name
        hdus.append(fits.ImageHDU(data=data, header=header))
    if empirical:
        signal = reconstruction.planes["signal_I"]
        weight = reconstruction.planes["weight_I"]
        variance = np.mean((reconstruction.noise - np.mean(
            reconstruction.noise, axis=-1, keepdims=True)) ** 2, axis=-1)
        extras = {
            "weight_formal_I": weight,
            "noise_variance_I": variance,
            "sig2noise_I": signal * np.sqrt(weight),
            "sig2noise_pixel_I": signal * np.sqrt(weight),
        }
        for name, values in extras.items():
            header = synthetic_wcs_header(rows, cols, celestial=celestial,
                                          delta_row=delta_row, delta_col=delta_col)
            if name == "weight_formal_I":
                header["ESTTYPE"] = "formal_normalization_coefficient_snapshot"
                header["TYPE"] = "formal_normalization_coefficient_snapshot"
                header["PRECSTAT"] = "conditional_SCI-PTC-001"
                header["COVSTAT"] = "unavailable"
            header["EXTNAME"] = name
            hdus.append(fits.ImageHDU(values[:, ::-1][np.newaxis, np.newaxis],
                                      header=header))
    fits.HDUList(hdus).writeto(path, checksum=True)


def write_synthetic_noise(path: Path, reconstruction: Reconstruction,
                          *, celestial: bool = True,
                          delta_row: int = 0, delta_col: int = 0) -> None:
    rows, cols, count = reconstruction.noise.shape
    if count != REALIZATIONS:
        die("synthetic realization count differs from campaign")
    hdus: list[Any] = [fits.PrimaryHDU()]
    for index in range(count):
        header = synthetic_wcs_header(rows, cols, celestial=celestial,
                                      delta_row=delta_row, delta_col=delta_col)
        header["EXTNAME"] = f"signal_{index}_I"
        header["UNIT"] = "mJy/beam"
        header["MEDRMS"] = 1.0
        hdus.append(fits.ImageHDU(
            reconstruction.noise[..., index][:, ::-1][np.newaxis, np.newaxis],
            header=header))
    fits.HDUList(hdus).writeto(path, checksum=True)


def self_check(args: argparse.Namespace) -> int:
    campaign, campaign_path = load_campaign(args.campaign)
    source_root = args.source_root.resolve(strict=True)
    if not (source_root / "validation/product_contracts.json").is_file() or \
            not (source_root / "config/tolteca/v2/manifest.yaml").is_file():
        die("self-check source root lacks the required candidate authorities")
    contracts_file = product_contract_path(
        campaign, campaign_path, args.product_contracts)
    contracts = load_contracts(campaign, campaign_path, contracts_file)
    book = CheckBook()
    owner_path_positive = validate_owner_path_string(
        "/work/toltec/sci-map-001", "self_check")
    book.add("self.owner_path.lexically_canonical_positive",
             str(owner_path_positive) == "/work/toltec/sci-map-001", None)
    for label, invalid_path in (
            ("trailing_slash", "/work/toltec/sci-map-001/"),
            ("dot_segment", "/work/toltec/../sci-map-001"),
            ("backslash", "/work/toltec\\sci-map-001"),
            ("newline", "/work/toltec/sci-map-001\nother")):
        rejected = False
        try:
            validate_owner_path_string(invalid_path, "self_check")
        except EvidenceError:
            rejected = True
        book.add(f"self.owner_path.rejects_{label}", rejected,
                 {"classification": label})
    point_case = campaign["cases"][0]
    plane_contracts_ok = True
    plane_contract_detail = []
    for campaign_case in campaign["cases"]:
        for scope in ("observation", "coadd"):
            required, forbidden = request_union_plane_contract(
                campaign_case, scope)
            overlap = sorted(required & forbidden)
            plane_contracts_ok = plane_contracts_ok and not overlap
            plane_contract_detail.append({
                "case_id": campaign_case["id"], "scope": scope,
                "overlap": overlap,
            })
    book.add("self.request_union.required_forbidden_disjoint_all_scopes",
             plane_contracts_ok, plane_contract_detail)
    sc_case = case_by_id(campaign, "S-C-SEQ")
    sc_required, sc_forbidden = request_union_plane_contract(
        sc_case, "observation")
    sc_correct = set(sc_required)
    book.add("self.request_union.S_C_products_off_correct_inventory",
             "formal_standardized_signal_I" in sc_required
             and "formal_standardized_signal_I" not in sc_forbidden
             and not (sc_correct & sc_forbidden),
             {"required": sorted(sc_required),
              "forbidden": sorted(sc_forbidden)})
    sc_with_extra = sc_correct | {"weight_formal_I"}
    book.add("self.request_union.S_C_rejects_forbidden_extra",
             bool(sc_with_extra & sc_forbidden),
             {"forbidden_present": sorted(sc_with_extra & sc_forbidden)})
    diagnostic_entries = successor_diagnostic_entries(contracts, sc_case)
    book.add("self.diagnostic.successor_contract_patterns_exact",
             len(diagnostic_entries) == 6
             and {row["family_id"] for row in diagnostic_entries} == {
                 "map_histogram", "map_psd", "map_diagnostics"},
             diagnostic_entries)
    damaged_contracts = copy.deepcopy(contracts)
    science_successor = next(row for row in damaged_contracts["contracts"]
                             if row.get("contract_id") ==
                             "sci-map-001-science-products-v1")
    science_base = next(row for row in damaged_contracts["contracts"]
                        if row.get("contract_id") ==
                        science_successor["extends_contract_id"])
    science_base["entries"] = [
        row for row in science_base["entries"]
        if row.get("entry_id") != "observation-psd"]
    diagnostic_contract_rejected = False
    try:
        successor_diagnostic_entries(damaged_contracts, sc_case)
    except EvidenceError:
        diagnostic_contract_rejected = True
    book.add("self.diagnostic.rejects_incomplete_contract_authority",
             diagnostic_contract_rejected, None)
    point_order = campaign["numbered_config_contract"]["point_order"]
    bundle_fixture = read_yaml(source_root / "config/tolteca/v2/manifest.yaml")
    vendor_fixture = {
        "schema_version": "tolproj-citlali-refactor-vendor-v2",
        "source_repository": "toltec-astro/citlali",
        "site_normalizations": [], "files": {},
        "mode_kits": {"point": {
            "bundle": "phase4_1_v2_1", "kit_version": "phase4.1-v2.1",
            "source_commit": campaign["authority"][
                "tolproj_bundle_source_commit"],
            "observation_filename": "72_pointing_observation.yaml",
            "repository_policy_filenames": [
                "60_pointing_internal_policy.yaml"],
            "operator_filenames": [
                "71_pointing_runtime.yaml", "81_pointing_defaults.yaml",
                "82_pointing_products.yaml", "90_pointing_advanced_overrides.yaml",
                "99_pointing_expert_overrides.yaml"],
        }},
    }
    installed_fixture = sorted(
        name for name in point_order
        if name not in ("40_setup.yaml", "99_zz_tolproj_submission_runtime.yaml"))
    marker_fixture = {
        "schema_version": "tolproj-installed-citlali-refactor-kit-v2",
        "kit_version": "phase4.1-v2.1", "bundle": "phase4_1_v2_1",
        "mode": "point",
        "observation_filename": "72_pointing_observation.yaml",
        "installed_filenames": installed_fixture,
        "policy_sha256": MODE_POLICY_SHA256["point"],
        "record_id": campaign["authority"]["tolproj_point_record_id"],
        "source_repository": "toltec-astro/citlali",
        "source_commit": campaign["authority"]["tolproj_bundle_source_commit"],
    }
    marker_validated = validate_installed_kit_marker(
        marker_fixture, vendor_fixture, bundle_fixture,
        point_case, campaign, point_order)
    book.add("self.preflight.installed_marker_exact_positive",
             marker_validated == marker_fixture, marker_validated)
    marker_mutations = {
        "schema_version": "wrong-schema", "kit_version": "wrong-version",
        "bundle": "wrong-bundle", "mode": "science",
        "observation_filename": "wrong-observation.yaml",
        "installed_filenames": list(reversed(installed_fixture)),
        "policy_sha256": "0" * 64, "record_id": "wrong-record",
        "source_repository": "wrong/repository", "source_commit": "0" * 40,
    }
    for field_name, replacement in marker_mutations.items():
        tampered_marker = copy.deepcopy(marker_fixture)
        tampered_marker[field_name] = replacement
        rejected = False
        try:
            validate_installed_kit_marker(
                tampered_marker, vendor_fixture, bundle_fixture,
                point_case, campaign, point_order)
        except EvidenceError:
            rejected = True
        book.add(f"self.preflight.installed_marker_rejects_{field_name}",
                 rejected, {"tampered_field": field_name})
    marker_with_extra = {**marker_fixture, "unregistered": "value"}
    extra_rejected = False
    try:
        validate_installed_kit_marker(
            marker_with_extra, vendor_fixture, bundle_fixture,
            point_case, campaign, point_order)
    except EvidenceError:
        extra_rejected = True
    book.add("self.preflight.installed_marker_rejects_extra_field",
             extra_rejected, None)

    membership_header = fits.Header()
    membership_header["OBSNUM0"] = 152389
    membership_header["WAV"] = "a1100"
    membership = primary_observation_membership(membership_header)
    membership_ok, membership_detail = primary_membership_matches_record(
        membership,
        {"scope": "observation", "obsnum": 152389, "array": "a1100"},
        point_case)
    book.add("self.inventory.primary_membership_positive",
             membership_ok, membership_detail)
    wrong_membership = copy.deepcopy(membership)
    wrong_membership["observation_keys"]["OBSNUM0"] = 152390
    wrong_ok, wrong_detail = primary_membership_matches_record(
        wrong_membership,
        {"scope": "observation", "obsnum": 152389, "array": "a1100"},
        point_case)
    book.add("self.inventory.primary_membership_rejects_wrong_observation",
             not wrong_ok, wrong_detail)
    # Edge policy: non-finite, negative, and zero coefficients are never
    # positive inputs. The selected order statistic and >= boundary are exact.
    edge = np.array([[np.nan, np.inf, -np.inf, -1.0, 0.0,
                      1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    selected = threshold_selection(edge, 0.5)
    book.add("self.threshold.edge_selection",
             selected == {"threshold": 2.5, "selected": 5.0,
                          "count": 5, "index": 4}, selected)
    edge_mask = np.isfinite(edge) & (edge > 0.0) & (edge >= selected["threshold"])
    book.add("self.threshold.finite_positive_ge",
             np.array_equal(edge_mask,
                            np.array([[False, False, False, False, False,
                                       False, False, True, True, True]])),
             edge_mask.astype(int).tolist())

    with tempfile.TemporaryDirectory(prefix="sci-map-001-self-check-") as tmp:
        root = Path(tmp)
        authority_target = root / "authority.bin"
        authority_target.write_bytes(b"self-check-authority\n")
        authority_manifest = root / "authority.sha256"
        authority_manifest.write_text(
            f"{sha256(authority_target)}  {authority_target}\n",
            encoding="utf-8")
        parsed_authority = parse_sha256_record(
            authority_manifest, "self-check authority")
        book.add("self.runtime.sha256_manifest_semantic_positive",
                 parsed_authority == [(authority_target,
                                       sha256(authority_target))],
                 [(str(path), digest)
                  for path, digest in parsed_authority])
        empty_authority = root / "empty-authority.sha256"
        empty_authority.write_text("", encoding="utf-8")
        empty_rejected = False
        try:
            parse_sha256_record(empty_authority, "self-check empty authority")
        except EvidenceError:
            empty_rejected = True
        book.add("self.runtime.sha256_manifest_rejects_empty",
                 empty_rejected, None)
        environment_path = root / "runtime-environment.txt"
        environment_lines = sorted([
            "OMP_NUM_THREADS=16", "SLURM_CPUS_PER_TASK=16",
            "SLURM_JOB_ID=123", "SLURM_JOB_NAME=sci-map-001-S-C-OMP",
            "SLURM_JOB_PARTITION=toltec-cpu",
            f"TOLPROJ_CITLALI_SHA256={sha256(authority_target)}",
            f"TOLPROJ_CITLALI_SNAPSHOT={authority_target}",
        ])
        environment_path.write_text(
            "\n".join(environment_lines) + "\n", encoding="utf-8")
        parsed_environment = parse_runtime_environment(
            environment_path, "self-check runtime environment")
        book.add("self.runtime.environment_semantic_positive",
                 parsed_environment["OMP_NUM_THREADS"] == "16"
                 and parsed_environment["SLURM_JOB_ID"] == "123"
                 and parsed_environment["TOLPROJ_CITLALI_SNAPSHOT"] ==
                 str(authority_target), parsed_environment)
        environment_path.write_text(
            "\n".join(environment_lines + [environment_lines[0]]) + "\n",
            encoding="utf-8")
        environment_rejected = False
        try:
            parse_runtime_environment(
                environment_path, "self-check repeated runtime environment")
        except EvidenceError:
            environment_rejected = True
        book.add("self.runtime.environment_rejects_repeat_or_unsorted",
                 environment_rejected, None)
        affinity_path = root / "affinity.txt"
        affinity_path.write_text(
            "pid 123's current affinity list: 0-3,8,10-11\n",
            encoding="utf-8")
        affinity = parse_cpu_affinity(affinity_path, "self-check affinity")
        book.add("self.runtime.affinity_semantic_positive",
                 affinity == {0, 1, 2, 3, 8, 10, 11}, sorted(affinity or ()))
        affinity_path.write_text("taskset unavailable\n", encoding="utf-8")
        book.add("self.runtime.affinity_unavailable_fail_closed_classification",
                 parse_cpu_affinity(affinity_path, "self-check unavailable")
                 is None, None)
        book.add("self.runtime.slurm_peak_memory_parser",
                 slurm_memory_bytes("1.5G") == int(1.5 * 1024 ** 3)
                 and slurm_memory_bytes("") is None
                 and slurm_memory_bytes("0K") is None, None)
        deterministic_npz_a = root / "deterministic-a.npz"
        deterministic_npz_b = root / "deterministic-b.npz"
        deterministic_arrays = {
            "integer_delta": np.asarray([0, -1, 2], dtype=np.int64),
            "finite_delta": np.asarray([0.0, np.nan, np.inf]),
        }
        write_npz_new(deterministic_npz_a, **deterministic_arrays)
        write_npz_new(deterministic_npz_b, **deterministic_arrays)
        book.add("self.residuals.deterministic_npz_bytes",
                 sha256(deterministic_npz_a) == sha256(deterministic_npz_b),
                 {"sha256": sha256(deterministic_npz_a)})

        config_root = root / "runtime-config"
        config_root.mkdir()
        master_executable = root / "bin/citlali"
        master_executable.parent.mkdir()
        master_executable.write_bytes(b"self-check Citlali executable\n")
        master_executable.chmod(0o555)
        launcher = config_root / ".tolproj/citlali-launcher"
        launcher.parent.mkdir()
        launcher.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o555)
        launcher_source = config_root / ".tolproj/citlali-source"
        launcher_source.write_text(
            f"{master_executable}\n", encoding="utf-8")
        pre_run_config = config_root.parent / "pre-run-merged.yaml"
        low_fixture = {
            "coadd": {"enabled": False},
            "runtime": {"n_threads": 1, "parallel_policy": "seq"},
        }
        self_executable = str(master_executable)
        pre_run_fixture = {
            "reduce": {"steps": {0: {
                "path": str(launcher.resolve()),
                "config": {"low_level": low_fixture}}}}}
        pre_run_config.write_text(yaml.safe_dump(pre_run_fixture), encoding="utf-8")
        runtime_fixture = {**copy.deepcopy(low_fixture),
                           "inputs": [{"data_items": [{"filepath": "/raw/input"}]}]}
        runtime_path = config_root / "citlali_merged_config.yaml"
        runtime_path.write_text(yaml.safe_dump(runtime_fixture), encoding="utf-8")
        source_copy = config_root / "source_000_input.yaml"
        source_copy.write_text(yaml.safe_dump(runtime_fixture), encoding="utf-8")
        source_manifest = {
            "schema_version": "citlali-config-source-manifest-v1",
            "merge_authority": "citlali_cli",
            "merge_semantics": "ordered_later_sources_override",
            "upstream": {"authority": "tolteca",
                         "ordered_sources_provided": False},
            "sources": [{
                "precedence": 0, "role": "citlali_cli_config",
                "source_path": "/upstream/input.yaml",
                "copied_filename": source_copy.name,
                "size_bytes": source_copy.stat().st_size,
                "sha256": sha256(source_copy),
            }],
            "merged": {
                "snapshot_filename": runtime_path.name,
                "serialization": "yaml_cpp_dump",
                "size_bytes": runtime_path.stat().st_size,
                "sha256": sha256(runtime_path),
            },
        }
        manifest_path = config_root / "config_source_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(source_manifest), encoding="utf-8")
        runtime_authority = validate_runtime_config_authority(
            config_root, pre_run_config, "self-runtime")
        book.add("self.runtime_config.manifest_and_pre_run_binding_positive",
                 runtime_authority["runtime_path"] == runtime_path.resolve()
                 and runtime_authority["runtime_only_top_level_keys"] == ["inputs"],
                 {key: str(value) if isinstance(value, Path) else value
                  for key, value in runtime_authority.items()})
        runtime_path.write_text(yaml.safe_dump({
            **runtime_fixture, "coadd": {"enabled": True}}), encoding="utf-8")
        runtime_tamper_rejected = False
        try:
            validate_runtime_config_authority(
                config_root, pre_run_config, "self-runtime-tampered")
        except EvidenceError:
            runtime_tamper_rejected = True
        book.add("self.runtime_config.rejects_effective_config_tamper",
                 runtime_tamper_rejected, None)
        runtime_path.write_text(yaml.safe_dump(runtime_fixture), encoding="utf-8")

        for name in point_order:
            path = config_root / name
            if name in TOLPROJ_FROZEN_NUMBERED_SHA256["point"]:
                source = (source_root / "config/tolteca/v2/point" /
                          name).read_bytes()
                source = source.replace(
                    b"path: /work/toltec/citlali_dev/citlali_refactor/build/bin/citlali",
                    b"path: citlali")
                path.write_bytes(source)
                if sha256(path) != TOLPROJ_FROZEN_NUMBERED_SHA256["point"][name]:
                    die(f"self-check cannot reconstruct frozen TolProj bytes: {name}")
            elif name == "99_pointing_expert_overrides.yaml":
                path.write_bytes(overlay_bytes(point_case, self_executable))
            else:
                path.write_text(
                    f"self-check numbered source {name}\n", encoding="utf-8")
        marker_path = config_root / ".citlali_refactor_kit.yaml"
        marker_path.write_text(yaml.safe_dump(marker_fixture), encoding="utf-8")
        raw_path = root / "raw-input.json"
        raw_path.write_text("{}\n", encoding="utf-8")
        preflight_fixture = {
            "paths": {
                "case_dir": str(config_root.resolve()),
                "merged": str(pre_run_config.resolve()),
                "marker": str(marker_path.resolve()),
                "source_root": str(source_root),
                "raw_input_manifest": str(raw_path.resolve()),
                "candidate_executable": self_executable,
                "launcher": str(launcher.resolve()),
                "launcher_source": str(launcher_source.resolve()),
            },
            "sha256": {
                "merged": sha256(pre_run_config), "marker": sha256(marker_path),
                "vendor_manifest": "1" * 64, "bundle_manifest": "2" * 64,
                "canonical_manifest": "3" * 64,
                "product_contracts": sha256(contracts_file),
                "raw_input_manifest": sha256(raw_path),
                "candidate_executable": sha256(master_executable),
                "launcher": sha256(launcher),
                "launcher_source": sha256(launcher_source),
                "numbered": {name: sha256(config_root / name)
                             for name in point_order},
            },
            "installed_kit_marker_authority": marker_fixture,
            "installed_numbered_authority": dict(sorted({
                **TOLPROJ_FROZEN_NUMBERED_SHA256["point"],
                "99_pointing_expert_overrides.yaml": hashlib.sha256(
                    overlay_bytes(point_case, self_executable)).hexdigest(),
            }.items())),
        }
        validate_preflight_file_binding(
            preflight_fixture, point_case, campaign, config_root,
            pre_run_config, raw_path, contracts_file)
        book.add("self.preflight.returned_file_bindings_positive", True, None)
        swapped_roles = copy.deepcopy(preflight_fixture)
        swapped_roles["paths"]["candidate_executable"] = str(launcher)
        swapped_roles["sha256"]["candidate_executable"] = sha256(launcher)
        swapped_roles_rejected = False
        try:
            validate_preflight_file_binding(
                swapped_roles, point_case, campaign, config_root,
                pre_run_config, raw_path, contracts_file)
        except EvidenceError:
            swapped_roles_rejected = True
        book.add("self.preflight.rejects_master_launcher_role_swap",
                 swapped_roles_rejected, None)
        wrong_merged = copy.deepcopy(pre_run_fixture)
        wrong_merged["reduce"]["steps"][0]["path"] = self_executable
        pre_run_config.write_text(
            yaml.safe_dump(wrong_merged), encoding="utf-8")
        wrong_merged_preflight = copy.deepcopy(preflight_fixture)
        wrong_merged_preflight["sha256"]["merged"] = sha256(pre_run_config)
        wrong_merged_rejected = False
        try:
            validate_preflight_file_binding(
                wrong_merged_preflight, point_case, campaign, config_root,
                pre_run_config, raw_path, contracts_file)
        except EvidenceError:
            wrong_merged_rejected = True
        book.add("self.preflight.rejects_master_as_final_merged_path",
                 wrong_merged_rejected, None)
        pre_run_config.write_text(
            yaml.safe_dump(pre_run_fixture), encoding="utf-8")
        bad_preflight = copy.deepcopy(preflight_fixture)
        bad_preflight["sha256"]["merged"] = "0" * 64
        preflight_tamper_rejected = False
        try:
            validate_preflight_file_binding(
                bad_preflight, point_case, campaign, config_root,
                pre_run_config, raw_path, contracts_file)
        except EvidenceError:
            preflight_tamper_rejected = True
        book.add("self.preflight.rejects_merged_digest_tamper",
                 preflight_tamper_rejected, None)
        bad_numbered_authority = copy.deepcopy(preflight_fixture)
        bad_numbered_authority["installed_numbered_authority"][
            "99_pointing_expert_overrides.yaml"] = "0" * 64
        numbered_authority_rejected = False
        try:
            validate_preflight_file_binding(
                bad_numbered_authority, point_case, campaign, config_root,
                pre_run_config, raw_path, contracts_file)
        except EvidenceError:
            numbered_authority_rejected = True
        book.add("self.preflight.rejects_expert_generator_authority_tamper",
                 numbered_authority_rejected, None)

        diagnostic_root = root / "diagnostic-root"
        diagnostic_root.mkdir()
        diagnostic_paths = []
        for entry in successor_diagnostic_entries(contracts, point_case):
            relative = str(entry["pattern"]).format(obs=152389).replace(
                "*", "self")
            path = diagnostic_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"self-check diagnostic\n")
            diagnostic_paths.append(path)
        diagnostic_inventory = [{"path": str(path)}
                                for path in diagnostic_paths]
        diagnostic_book = CheckBook()
        diagnostic_classified = verify_diagnostic_families(
            diagnostic_root, point_case, contracts, diagnostic_inventory,
            diagnostic_book)
        book.add("self.diagnostic.exact_pattern_inventory_positive",
                 diagnostic_book.passed
                 and len(diagnostic_classified) == len(diagnostic_paths),
                 diagnostic_book.checks)
        extra_diagnostic = (diagnostic_root / "152389/raw" /
                            "unexpected_pointing_152389_hist.nc")
        extra_diagnostic.write_bytes(b"unexpected\n")
        negative_inventory = [*diagnostic_inventory,
                              {"path": str(extra_diagnostic)}]
        negative_diagnostic_book = CheckBook()
        verify_diagnostic_families(
            diagnostic_root, point_case, contracts, negative_inventory,
            negative_diagnostic_book)
        book.add("self.diagnostic.rejects_extra_or_ambiguous_match",
                 bool(negative_diagnostic_book.hard_failures),
                 negative_diagnostic_book.hard_failures)

        rows = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        cols = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        count = rows.size
        signs = boost_mt19937_scan_signs(2)
        projection_digest = "canonical-hexfloat-sha256-v1:" + "1" * 64
        authority = RawManifestAuthority(
            path=root / "raw-input.json", digest="0" * 64, mode="science",
            producer_identity="frozen-self-check-independent-ledger",
            producer_program=root / "producer", producer_program_sha256="2" * 64,
            sources={}, memberships={(152390, "a1100"): {
                "projection_record_count": count,
                "scan_order": [
                    {"scan_index": 0, "identity": "scan-0", "sample_count": 2},
                    {"scan_index": 1, "identity": "scan-1", "sample_count": 2},
                ],
                "detector_order": [
                    {"detector_index": 0, "detector_uid": "det-0", "network": 0},
                    {"detector_index": 1, "detector_uid": "det-1", "network": 0},
                ],
                "projection": {
                    "identity_digest": projection_digest,
                    "map_rows": 4, "map_cols": 4, "_sample_rate_hz": 10.0,
                },
            }})
        ledger = root / "ledger.npz"
        np.savez(
            ledger,
            schema_version=np.array(LEDGER_SCHEMA), obsnum=np.array(152390),
            candidate_sha=np.array(CANDIDATE_SHA),
            raw_input_manifest_sha256=np.array("0" * 64),
            producer_identity=np.array("frozen-self-check-independent-ledger"),
            bundle_identity_digest=np.array(projection_digest),
            array=np.array("a1100"), map_rows=np.array(4), map_cols=np.array(4),
            sample_rate_hz_numeric=np.array("10"),
            sample_rate_hz_hex=np.array(float(10.0).hex()),
            sample_rate_hz_encoding=np.array(
                "binary64-max-digits10-and-c99-hexfloat"),
            row=rows, col=cols,
            detector_index=np.tile(np.repeat(np.arange(2), 2), 2),
            sample_index=np.tile(np.arange(2), 4),
            scan_index=np.repeat(np.arange(2), count // 2),
            geometric_in_bounds=np.ones(count, dtype=np.uint8),
            upstream_eligible=np.ones(count, dtype=np.uint8),
            coefficient=np.array([0.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            sample_signal=np.arange(1, count + 1, dtype=np.float64),
            sample_kernel=np.full(count, 2.0),
            sample_interval_s=np.full(count, 0.1), realization_signs=signs,
        )
        reconstruction = reconstruct_observation_from_ledger(
            ledger, 152390, "a1100", (4, 4), 0.5, authority)
        book.add("self.ledger.nonpositive_coefficients_skip",
                 int(np.sum(reconstruction.planes["contributing_hits_I"])) == 6,
                 int(np.sum(reconstruction.planes["contributing_hits_I"])))
        with np.load(ledger, allow_pickle=False) as archive:
            bad_payload = {name: np.asarray(archive[name]).copy()
                           for name in archive.files}
        bad_payload["sample_signal"][-1] = np.nan
        bad_ledger = root / "ledger-valid-nonfinite.npz"
        np.savez(bad_ledger, **bad_payload)
        nonfinite_failed = False
        try:
            reconstruct_observation_from_ledger(
                bad_ledger, 152390, "a1100", (4, 4), 0.5, authority)
        except EvidenceError:
            nonfinite_failed = True
        book.add("self.ledger.valid_nonfinite_fails_closed",
                 nonfinite_failed, None)
        book.add("self.ledger.realizations_64",
                 reconstruction.noise.shape == (4, 4, REALIZATIONS),
                 list(reconstruction.noise.shape))
        book.add("self.ledger.f010_typed",
                 reconstruction.planes["geometric_hits_I"].dtype == np.int64
                 and reconstruction.planes["normalization_support_I"].dtype == np.uint8
                 and reconstruction.planes["retained_exposure_I"].dtype == np.float64,
                 {name: str(value.dtype) for name, value in reconstruction.planes.items()})
        book.add("self.alias.coverage_bitwise",
                 reconstruction.planes["coverage_I"].tobytes() ==
                 reconstruction.planes["retained_exposure_I"].tobytes(), None)
        book.add("self.alias.coverage_bool_bitwise",
                 reconstruction.planes["coverage_bool_I"].tobytes() ==
                 reconstruction.planes["science_policy_support_I"].tobytes(), None)
        expected_valid = reconstruction.planes["normalization_support_I"].astype(bool) & \
            reconstruction.planes["science_policy_support_I"].astype(bool) & \
            np.isfinite(reconstruction.planes["signal_I"]) & \
            np.isfinite(reconstruction.planes["kernel_I"]) & \
            np.all(np.isfinite(reconstruction.noise), axis=-1)
        book.add("self.validity.conjunction",
                 np.array_equal(reconstruction.planes["science_valid_I"].astype(bool),
                                expected_valid), None)

        # The blank domain is contract-derived and deliberately ignores every
        # persisted policy/validity mask and realization finite state.
        blank_signal = np.array([[1.0, np.nan], [2.0, 3.0]])
        blank_weight = np.array([[2.0, 2.0], [0.0, 4.0]])
        blank_exposure = np.array([[1.0, 1.0], [1.0, -1.0]])
        blank_domain = finite_blank_domain(blank_signal, blank_weight,
                                           blank_exposure)
        book.add("self.diagnostics.F_independent_of_persisted_masks",
                 np.array_equal(blank_domain,
                                np.array([[True, False], [False, False]])),
                 blank_domain.astype(int).tolist())

        population_cube = np.zeros((1, 2, REALIZATIONS), dtype=np.float64)
        population_cube[0, 0] = np.arange(REALIZATIONS, dtype=np.float64)
        population_cube[0, 1, 7] = np.nan
        population_mean, population_variance, population_finite = \
            population_noise_statistics(population_cube)
        expected_population_variance = float(np.var(
            np.arange(REALIZATIONS, dtype=np.float64), ddof=0))
        book.add("self.diagnostics.population_64_nonfinite_undefined",
                 exact_float_equal(population_variance[0, 0],
                                   expected_population_variance)
                 and math.isnan(population_mean[0, 1])
                 and math.isnan(population_variance[0, 1])
                 and not population_finite[0, 1, 7],
                 {"finite_variance": population_variance[0, 0],
                  "nonfinite_variance": None})

        point_projection = {
            "_target_axis1": 100.0, "_target_axis2": 45.0,
            "target": {"unit": "deg"},
        }
        point_primary = fits.Header()
        point_primary["SRC_RA"] = math.radians(100.0)
        point_primary["SRC_DEC"] = math.radians(45.0)
        point_primary["TAN_RA"] = math.radians(100.0)
        point_primary["TAN_DEC"] = math.radians(45.0)
        point_distance, point_metadata = target_distance_arcsec(
            synthetic_wcs_header(5, 5, celestial=False), (5, 5),
            "point", point_projection, point_primary)
        science_projection = {
            "_target_axis1": 150.0, "_target_axis2": 2.0,
            "target": {"unit": "deg"},
        }
        science_distance, science_metadata = target_distance_arcsec(
            synthetic_wcs_header(5, 5, celestial=True), (5, 5),
            "science", science_projection)
        book.add("self.diagnostics.target_distance_units_and_frames",
                 point_metadata["method"] ==
                 "euclidean_declared_tangent_plane_axes"
                 and point_metadata["output_unit"] == "arcsec"
                 and point_metadata["primary_source_tangent_target_bound"] is True
                 and abs(point_distance[2, 2]) <= 1.0e-12
                 and abs(point_distance[2, 3] - 1.0) <= 1.0e-10
                 and science_metadata["method"] == "great_circle_celestial_wcs"
                 and science_metadata["output_unit"] == "arcsec"
                 and abs(science_distance[2, 2]) <= 1.0e-8,
                 {"point_center": point_distance[2, 2],
                  "point_adjacent": point_distance[2, 3],
                  "science_center": science_distance[2, 2]})

        diagonal_mask = np.zeros((3, 3), dtype=bool)
        diagonal_mask[0, 0] = True
        diagonal_mask[1, 1] = True
        diagonal_values = np.zeros((3, 3), dtype=np.float64)
        diagonal_values[0, 0], diagonal_values[1, 1] = 6.0, -7.0
        diagonal_regions = connected_region_records(diagonal_mask,
                                                     diagonal_values)
        book.add("self.diagnostics.eight_neighbour_components",
                 len(diagonal_regions) == 1
                 and diagonal_regions[0]["pixel_count"] == 2
                 and diagonal_regions[0]["max_abs_z"] == 7.0,
                 diagonal_regions)
        book.add("self.diagnostics.fixed_edge_bins",
                 EDGE_BINS == ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0),
                               (10.0, 20.0), (20.0, 40.0),
                               (40.0, math.inf)),
                 [{"lower_inclusive": lower,
                   "upper_exclusive": (upper if math.isfinite(upper)
                                         else "infinity")}
                  for lower, upper in EDGE_BINS])

        covariance_y = np.column_stack((
            np.arange(REALIZATIONS, dtype=np.float64),
            np.zeros(REALIZATIONS, dtype=np.float64),
        ))
        covariance_centered = covariance_y - np.mean(covariance_y, axis=0)
        covariance_factor = covariance_centered / math.sqrt(REALIZATIONS)
        covariance_diagonal = np.mean(covariance_centered ** 2, axis=0)
        book.add("self.diagnostics.covariance_diagonal_and_zero_correlation",
                 np.array_equal(np.sum(covariance_factor ** 2, axis=0),
                                covariance_diagonal)
                 and covariance_diagonal[0] > 0.0
                 and covariance_diagonal[1] == 0.0,
                 {"diagonal": covariance_diagonal.tolist(),
                  "zero_variance_columns": [1]})
        cancelled_per_scan_plane = np.float64(0.0)
        cancelled_per_scan_plane += np.float64(1.0)
        cancelled_per_scan_plane += np.float64(-1.0)
        exact_sum_abs = np.longdouble(abs(cancelled_per_scan_plane))
        incorrect_termwise_sum_abs = np.longdouble(abs(1.0) + abs(-1.0))
        rejected, _ = scan_farm_bound(
            np.array([0.0]), np.array([np.nextafter(0.0, 1.0)]),
            np.array([exact_sum_abs], dtype=np.longdouble), 1)
        book.add("self.scan_farm.per_scan_cancellation_zero_bound",
                 exact_sum_abs == 0.0 and incorrect_termwise_sum_abs == 2.0
                 and not rejected,
                 {"sum_abs_binary64_per_scan_plane": float(exact_sum_abs),
                  "incorrect_sum_abs_sample_terms":
                      float(incorrect_termwise_sum_abs)})
        gap_book = CheckBook()
        gap_book.add("self.synthetic.K_nonempty", False,
                     "SCI-MAP-001-UNITY-001-EG-BACKGROUND-NONEMPTY",
                     evidence_gap=True)
        book.add("self.diagnostics.K_zero_named_evidence_gap",
                 len(gap_book.evidence_gaps) == 1
                 and gap_book.evidence_gaps[0]["result"] == "evidence_gap",
                 gap_book.evidence_gaps)

        map_path = root / "map.fits"
        noise_path = root / "noise.fits"
        write_synthetic_map(map_path, reconstruction, contracts,
                            celestial=True, empirical=True)
        write_synthetic_noise(noise_path, reconstruction, celestial=True)
        product = FitsProduct.open(map_path)
        noise_product = FitsProduct.open(noise_path)
        self_residuals: dict[str, np.ndarray] = {}
        try:
            apply_contract_check(product, contracts["checks"]["sci_map_common_celestial_v1"],
                                 book, "self.fits.common", self_residuals)
            apply_contract_check(product, contracts["checks"]["sci_map_naive_observation_v1"],
                                 book, "self.fits.f010", self_residuals)
            cube = verify_noise_file(
                noise_product, product.array("signal_I").shape,
                product.hdus["signal_I"].header, "mJy/beam",
                book, "self.fits.noise")
            compare_reconstruction(product, cube, reconstruction, book,
                                   "self.fits", self_residuals)
        finally:
            product.close()
            noise_product.close()
        book.add("self.residuals.reconstruction_and_alias_coverage",
                 any(key.endswith("__integer_delta") for key in self_residuals)
                 and any(key.endswith("__finite_delta") for key in self_residuals)
                 and any(key.endswith("__bitwise_xor") for key in self_residuals),
                 sorted(self_residuals))
        tampered = np.array([1.0, np.nan, np.inf], dtype=np.float64)
        expected_tampered = np.array([0.0, np.nan, -np.inf], dtype=np.float64)
        residual_fixture: dict[str, np.ndarray] = {}
        record_lossless_residual(
            residual_fixture, "tampered", tampered, expected_tampered)
        book.add("self.residuals.nonfinite_topology_and_delta_negative",
                 residual_fixture["tampered__finite_delta"][0] == 1.0
                 and residual_fixture["tampered__actual_nan"][1] == 1
                 and residual_fixture["tampered__actual_posinf"][2] == 1
                 and residual_fixture["tampered__expected_neginf"][2] == 1,
                 sorted(residual_fixture))
        integer_residual_fixture: dict[str, np.ndarray] = {}
        record_lossless_residual(
            integer_residual_fixture, "uint8_mask",
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([1, 0], dtype=np.uint8), bitwise=True)
        book.add("self.residuals.uint8_negative_delta_is_signed_exact",
                 np.array_equal(
                     integer_residual_fixture[
                         "uint8_mask__integer_delta"],
                     np.asarray([-1, 1], dtype=np.int64))
                 and integer_residual_fixture[
                     "uint8_mask__integer_delta"].dtype == np.int64
                 and np.array_equal(
                     integer_residual_fixture["uint8_mask__bitwise_xor"],
                     np.asarray([1, 1], dtype=np.uint8)),
                 {key: value.tolist()
                  for key, value in integer_residual_fixture.items()
                  if key.endswith(("__integer_delta", "__bitwise_xor"))})
        bad_primary_path = root / "bad-primary.fits"
        fits.HDUList([fits.PrimaryHDU(
            data=np.ones((2, 2), dtype=np.float64))]).writeto(
                bad_primary_path, checksum=True)
        primary_rejected = False
        try:
            bad_primary_product = FitsProduct.open(bad_primary_path)
            bad_primary_product.close()
        except EvidenceError:
            primary_rejected = True
        book.add("self.inventory.rejects_numeric_PRIMARY",
                 primary_rejected, None)

        comparison_record = {
            "map": str(map_path), "map_sha256": sha256(map_path),
            "noise": str(noise_path), "noise_sha256": sha256(noise_path),
        }
        comparison_book = CheckBook()
        compare_map_records(
            comparison_record, comparison_record,
            (*COMMON_PLANES, *F010_OBSERVATION, *EMPIRICAL_PLANES),
            True, root, comparison_book, "self.compare.seq_omp")
        comparison_residual = root / (
            "self_compare_seq_omp-comparison-residuals.npz")
        with np.load(comparison_residual, allow_pickle=False) as archive:
            comparison_keys = set(archive.files)
        book.add("self.residuals.seq_omp_all_common_numeric_hdus",
                 comparison_book.passed
                 and "map_hdu_signal_I__finite_delta" in comparison_keys
                 and "map_hdu_geometric_hits_I__integer_delta" in
                 comparison_keys
                 and "noise_hdu_signal_0_I__finite_delta" in
                 comparison_keys
                 and "noise_hdu_signal_63_I__finite_delta" in
                 comparison_keys,
                 {"comparison_check_count": len(comparison_book.checks),
                  "archive_key_count": len(comparison_keys)})

        obs_header = synthetic_wcs_header(4, 4, celestial=True)
        coadd_header = synthetic_wcs_header(4, 4, celestial=True,
                                            delta_row=1, delta_col=2)
        # NAXIS is carried by HDUs in production; centered_wcs_check consumes
        # explicit shapes and only needs the WCS cards here.
        passed, detail = centered_wcs_check(obs_header, coadd_header,
                                            (4, 4), (6, 8), 1, 2)
        book.add("self.wcs.centered_even_embedding", passed, detail)

        # Independent two-observation coadd identities including all F010
        # integer/exposure terms and every realization.
        norm = reconstruction.planes["normalization_support_I"].astype(bool)
        q = 2.0 * reconstruction.planes["weight_I"]
        safe = np.where(q > 0.0, q, 1.0)
        n = 2.0 * reconstruction.planes["weight_I"] * reconstruction.planes["signal_I"]
        coadd_signal = np.where(norm, n / safe, 0.0)
        coadd_noise = np.where(norm[..., None],
            (2.0 * reconstruction.planes["weight_I"][..., None] * reconstruction.noise)
            / safe[..., None], 0.0)
        book.add("self.coadd.Q_N_identity",
                 np.allclose(coadd_signal, reconstruction.planes["signal_I"],
                             atol=0.0, rtol=0.0), None)
        book.add("self.coadd.f010_sums",
                 np.array_equal(2 * reconstruction.planes["geometric_hits_I"],
                                reconstruction.planes["geometric_hits_I"]
                                + reconstruction.planes["geometric_hits_I"])
                 and np.array_equal(2 * norm.astype(np.int64),
                                    norm.astype(np.int64) + norm.astype(np.int64))
                 and np.allclose(2 * reconstruction.planes["retained_exposure_I"],
                                 reconstruction.planes["retained_exposure_I"] * 2,
                                 atol=0.0, rtol=0.0), None)
        book.add("self.coadd.realization_identity",
                 np.allclose(coadd_noise, reconstruction.noise,
                             atol=0.0, rtol=0.0), None)

    float_ok, _ = numeric_close(np.array([1.0 + 1.0e-11]), np.array([1.0]))
    integer_bad, _ = integer_equal(np.array([1], dtype=np.int64),
                                   np.array([2], dtype=np.int64))
    book.add("self.comparison.registered_float_tolerance", float_ok, None)
    book.add("self.comparison.integer_exact_rejects_delta", not integer_bad, None)
    book.add("self.collection.exact_driver_case_shape",
             COLLECTION_CASE_FILE_FIELDS == (
                 "preflight_manifest", "submit_record", "stdout", "stderr",
                 "exit_record", "slurm_accounting"),
             list(COLLECTION_CASE_FILE_FIELDS))
    book.add("self.digest.cxx_hexfloat",
             cxx_hexfloat(1.0) == "0x1p+0"
             and cxx_hexfloat(-0.0) == "-0x0p+0"
             and cxx_hexfloat(1.5) == "0x1.8p+0",
             {"one": cxx_hexfloat(1.0), "negative_zero": cxx_hexfloat(-0.0),
              "one_point_five": cxx_hexfloat(1.5)})
    book.add("self.noise.boost_mt19937_conformance_vector",
             boost_mt19937_scan_signs(1)[0, :10].tolist() ==
             [1, -1, 1, 1, -1, 1, 1, -1, 1, -1],
             boost_mt19937_scan_signs(1)[0, :10].tolist())
    if not book.passed:
        raise EvidenceError("self-check failed: " + json.dumps(book.failures,
                                                               sort_keys=True, default=str))
    result = {
        "schema_version": "sci-map-001-analysis-self-check-v1",
        "program_schema": PROGRAM_SCHEMA,
        "candidate_sha": CANDIDATE_SHA,
        "result": "pass",
        "check_count": len(book.checks),
        "checks": book.checks,
        "claim": "synthetic_program_self_check_only_not_external_evidence",
    }
    if args.output:
        write_new(args.output.resolve(), json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def validate_owner_values_command(args: argparse.Namespace) -> int:
    values = validate_owner_values(args.input.resolve(), args.require_existing)
    result = {
        "schema_version": "sci-map-unity-owner-values-validation-v1",
        "result": "pass", "unity_host_alias": values["unity_host_alias"],
        "candidate_sha": CANDIDATE_SHA,
        "path_checks": "required" if args.require_existing else "structural_only",
    }
    if args.output:
        write_new(args.output.resolve(), json_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_parser = subparsers.add_parser("self-check",
                                        help="run contract-derived synthetic checks")
    self_parser.add_argument("--campaign", type=Path)
    self_parser.add_argument("--product-contracts", type=Path)
    self_parser.add_argument("--source-root", type=Path, required=True)
    self_parser.add_argument("--output", type=Path)
    self_parser.set_defaults(function=self_check)

    owner = subparsers.add_parser("validate-owner-values",
                                  help="validate explicit owner deployment facts")
    owner.add_argument("--input", type=Path, required=True)
    owner.add_argument("--require-existing", action="store_true")
    owner.add_argument("--output", type=Path)
    owner.set_defaults(function=validate_owner_values_command)

    materialize = subparsers.add_parser("materialize-case",
                                        help="materialize one exact expert overlay")
    materialize.add_argument("--campaign", type=Path)
    materialize.add_argument("--case-id", choices=tuple(EXPECTED_CASES), required=True)
    materialize.add_argument("--owner-values", type=Path)
    materialize.add_argument("--citlali-executable")
    materialize.add_argument("--output", type=Path, required=True)
    materialize.set_defaults(function=materialize_case)

    materialize_all_parser = subparsers.add_parser(
        "materialize-all", help="materialize all seven exact expert overlays")
    materialize_all_parser.add_argument("--campaign", type=Path)
    materialize_all_parser.add_argument("--owner-values", type=Path)
    materialize_all_parser.add_argument("--citlali-executable")
    materialize_all_parser.add_argument("--output", type=Path, required=True)
    materialize_all_parser.set_defaults(function=materialize_all)

    preflight = subparsers.add_parser("preflight-case",
                                      help="verify one merged case and config authority")
    preflight.add_argument("--campaign", type=Path)
    preflight.add_argument("--case-id", choices=tuple(EXPECTED_CASES), required=True)
    preflight.add_argument("--mode", choices=("point", "science"), required=True)
    preflight.add_argument("--case-dir", type=Path, required=True)
    preflight.add_argument("--merged", type=Path, required=True)
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--vendor-manifest", type=Path, required=True)
    preflight.add_argument("--bundle-manifest", type=Path, required=True)
    preflight.add_argument("--canonical-manifest", type=Path, required=True)
    preflight.add_argument("--product-contracts", type=Path)
    preflight.add_argument("--marker", type=Path, required=True)
    preflight.add_argument("--raw-input-manifest", type=Path, required=True)
    preflight.add_argument("--owner-values", type=Path)
    preflight.add_argument("--citlali-executable")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(function=preflight_case)

    build = subparsers.add_parser("build-analysis-inputs",
                                  help="freeze returned result paths and hashes")
    build.add_argument("--campaign", type=Path)
    build.add_argument("--request-root", type=Path, required=True)
    build.add_argument("--collection", type=Path)
    build.add_argument("--product-contracts", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=build_analysis_inputs)

    run = subparsers.add_parser("run", help="run the frozen evidence verifier")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--source-root", type=Path)
    run.add_argument("--audit-tool", type=Path)
    run.add_argument("--compare-tool", type=Path)
    run.add_argument("--python", default=sys.executable)
    # Retain accepted protocol spelling; the values are cross-checked through
    # the frozen campaign/inputs rather than used as mutable policy switches.
    run.add_argument("--request-root", type=Path)
    run.add_argument("--product-contracts", type=Path)
    run.add_argument("--profile-registry", type=Path)
    run.add_argument("--accepted-runs", type=Path)
    run.add_argument("--point-contract")
    run.add_argument("--science-contract")
    run.add_argument("--realizations", type=int, default=REALIZATIONS)
    run.add_argument("--atol", type=float, default=REGISTERED_ATOL)
    run.add_argument("--rtol", type=float, default=REGISTERED_RTOL)
    run.add_argument("--max-array-elements", type=int, default=0)
    run.add_argument("--max-records", type=int, default=2147483647)
    run.set_defaults(function=run_analysis)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw == ["--self-check"]:
        raw = ["self-check"]
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.command == "run":
        if args.realizations != REALIZATIONS or not exact_float_equal(args.atol, REGISTERED_ATOL) \
                or not exact_float_equal(args.rtol, REGISTERED_RTOL):
            parser.error("run tolerances/realization count are frozen by campaign.json")
        if args.max_array_elements != 0 or args.max_records != 2147483647:
            parser.error("run comparison bounds are frozen by the campaign protocol")
    try:
        return int(args.function(args))
    except EvidenceError as exc:
        print(f"SCI-MAP-001 evidence package error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("SCI-MAP-001 evidence package interrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
