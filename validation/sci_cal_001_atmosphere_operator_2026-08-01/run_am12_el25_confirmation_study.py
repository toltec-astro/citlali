#!/usr/bin/env python3
"""Run the preregistered SCI-CAL-001 AM12 EL25 confirmation study.

This is a standalone evidence driver, not Citlali application code.  AM may
run only with the explicit ``--run-confirmation`` mode, from a clean Git
preregistration commit, into the exact fresh external-cache basename frozen in
the machine preregistration.  The other modes are cache-only and cannot launch
AM.  No FTS or Beammap input is accepted by this driver.

The inclusive one-percent gate is numerical representation fidelity for the
declared AM calculation.  It is not observational or physical photometric
accuracy and does not authorize an operator or an operational domain.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from jsonschema import Draft202012Validator

import probe_am12_h2o_scale_hypotheses as p1_driver
import run_am12_successor_adoption_study as adoption_driver


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

SCHEMA_VERSION = "sci-cal-001-am12-el25-confirmation-study-v1"
RESULT_SCHEMA_VERSION = "sci-cal-001-am12-el25-confirmation-result-v1"
STUDY_ID = "SCI-CAL-001-AM12-EL25-CONFIRMATION-001"
PACKAGE_ID = "SCI-CAL-001"

PROTOCOL_NAME = "AM12_EL25_CONFIRMATION_PROTOCOL.md"
PREREGISTRATION_NAME = "am12_el25_confirmation_preregistration.json"
RESULT_SCHEMA_NAME = "am12_el25_confirmation_result.schema.json"
RUNNER_NAME = "run_am12_el25_confirmation_study.py"
RESULT_SCHEMA_SHA256 = (
    "a28e738970b2a462fd1fb68c78aad552e32cbd396f8f60956f8615e4be2a3965"
)

DEFAULT_P1_CACHE = Path(
    "/private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root"
)
DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
DEFAULT_AM_EXECUTABLE = Path(
    "/private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am"
)
DEFAULT_TOLTECA_REPO = Path("/Users/gwilson/GitHub/tolteca")
DEFAULT_COORDINATION_REPO = Path(
    "/private/tmp/citlali-scientific-audit-framework"
)

EXPECTED_AM_EXECUTABLE_SHA256 = (
    "78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb"
)
EXPECTED_AM_SOURCE_SHA256 = (
    "0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8"
)
EXPECTED_V2_RUNNER_SHA256 = (
    "ace8e08a037535260b6b1d889f83dbf722ffc932e05bc1f7f83f0565ef0ff47c"
)
EXPECTED_P1_RUNNER_SHA256 = (
    "caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c"
)
EXPECTED_NODE_SHA256 = (
    "8005c8ae1d4ab1c8de39f06a632d76d3e8f248939dc63c616dd176bcbd2f6fe2"
)
EXPECTED_HOLDOUT_ROWS_SHA256 = (
    "ad74d19ef0bc915255b9cc7a507e8977f96435fb37ce0d0bd7cb385991c1802c"
)
EXPECTED_V2_MANIFEST_SHA256 = (
    "c9f6aea80851fb7726b8845d4697af1cb270cb7ff7ce51d3d5fc63828f793b3a"
)
EXPECTED_V2_DECISION_SHA256 = (
    "976c6c6a269a1b5dabde2b5eba89cb6176b02b837ea2b7b0e26a64307fe9ee59"
)
EXPECTED_F401_SHA256SUMS_SHA256 = (
    "bafd34e4a3d5bffb95b3af1fdbcfb7c993146248b2bccd1d0333bae91fd3caad"
)

F401_HEAD = "f4014d3669b94b1eceb8158da7993737efc908f2"
REPAIR_BASE = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
COORDINATION_BINDING_COMMIT = "8fc9263a2f502656b51d32cb60655481f83509f1"
IMMUTABLE_DECISION_COMMIT = "f513f410b88d147be6bd016d4c79ac1d3a5b2a8e"

TAU_MIN = 0.0
TAU_MAX = 0.158313198574890929
ELEVATION_MIN_DEG = 25.0
ELEVATION_MAX_DEG = 80.0
ELEVATIONS_DEG = tuple(range(25, 81))
ALPHAS = (-1, 0, 2, 4)
ARRAYS = ("a1100", "a1400", "a2000")
FIDELITY_GATE = 0.01
PHYSICAL_TOLERANCE = 1.0e-12
ANCHOR_TOLERANCE = 1.0e-12
LOW_SEGMENT_TOLERANCE = 1.0e-12
CONTINUITY_TOLERANCE = 1.0e-10
JOBS = 8
OMP_THREADS = 1
CACHE_SHARDS = 8
CACHE_BASENAME = "sci_cal_001_am12_el25_confirmation_v1_20260802_root"
CACHE_LOCK_NAME = ".am12_el25_confirmation.lock"
PINNED_LOCALE = {"LANG": "C", "LC_ALL": "C"}
PASSBAND_SET_ID = (
    "toltec-passband-set-v1:"
    "sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433"
)
PASSBAND_SET_DIGEST = PASSBAND_SET_ID.rsplit(":", 1)[1]
PASSBAND_INDEX_SHA256 = (
    "74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5"
)
PASSBAND_TOTAL_BYTES = 1_297_803

CANDIDATES = (
    {
        "candidate_id": (
            "fixed_djf25_v1+am12_piecewise_linear_los_tau_eval_v0"
        ),
        "lane": "fixed_djf25_v1",
        "operator": "am12_piecewise_linear_los_tau_eval_v0",
        "role": "primary_confirmatory",
    },
    {
        "candidate_id": "fixed_djf25_v1+am12_pchip_los_tau_eval_v0",
        "lane": "fixed_djf25_v1",
        "operator": "am12_pchip_los_tau_eval_v0",
        "role": "secondary_descriptive",
    },
    {
        "candidate_id": (
            "conditioned_djf_v1+am12_piecewise_linear_los_tau_eval_v0"
        ),
        "lane": "conditioned_djf_v1",
        "operator": "am12_piecewise_linear_los_tau_eval_v0",
        "role": "secondary_descriptive",
    },
    {
        "candidate_id": "conditioned_djf_v1+am12_pchip_los_tau_eval_v0",
        "lane": "conditioned_djf_v1",
        "operator": "am12_pchip_los_tau_eval_v0",
        "role": "secondary_descriptive",
    },
)

OUTPUT_NAMES = (
    "am12_el25_confirmation_execution_context.json",
    "am12_el25_confirmation_scales.csv",
    "am12_el25_confirmation_run_inventory.csv",
    "am12_el25_confirmation_rows.csv",
    "am12_el25_confirmation_metrics.csv",
    "am12_el25_confirmation_physical_metrics.csv",
    "am12_el25_confirmation_coverage.json",
    "am12_el25_confirmation_decision.json",
    "AM12_EL25_CONFIRMATION_REPORT.md",
    "am12_el25_confirmation_manifest.json",
)

CACHE_MUTATION_DIAGNOSTIC = re.compile(
    rb"^! Warning: Unable to rename file in "
    rb"(?:insert_as_mru|promote_to_mru)\(\)\.$",
    re.MULTILINE,
)
RAW_WARNING_PREFIX = b"! Warning:"
RAW_ERROR_PREFIX = b"! Error:"
RAW_UNRESOLVED_SUMMARY_HEADER = (
    b"! Warning: Encountered in-band lines narrower than the frequency"
)
RAW_UNRESOLVED_COLUMN_WARNING = re.compile(
    rb"^! Warning: Column included [0-9]+ unresolved lines\.$"
)


@dataclass(frozen=True)
class Controls:
    protocol_path: Path
    preregistration_path: Path
    schema_path: Path
    protocol_sha256: str
    preregistration_sha256: str
    schema_sha256: str
    preregistration: dict[str, Any]
    result_schema: dict[str, Any]


@dataclass(frozen=True)
class Coordinate:
    coordinate_id: str
    interval: str
    fraction_numerator: int
    fraction_denominator: int
    requested_tau: Decimal
    analytic_transmission: Decimal
    target_literal: str
    achieved_tau: Decimal
    residual: Decimal
    negative_bound: Decimal
    positive_bound: Decimal


@dataclass(frozen=True)
class ScaleCase:
    coordinate: Coordinate
    profile: str


@dataclass(frozen=True)
class TruthRecord:
    coordinate: Coordinate
    profile: str
    elevation_deg: int
    scale_decimal: str
    scale_hex: str
    achieved_transmission: float
    parsed: adoption_driver.ParsedAM
    raw_sha256: str
    sidecar_sha256: str
    cache_id: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_control_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def f17(value: float) -> str:
    return f"{float(value):.17e}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def decimal_e6(value: Decimal) -> str:
    mantissa, exponent = format(value, ".6e").split("e")
    return f"{mantissa}e{int(exponent):+03d}"


def csv_bytes(fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output.getvalue().encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return is_relative_to(left, right) or is_relative_to(right, left)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_output(repo: Path, *args: str) -> bytes:
    return adoption_driver.git_output(repo, *args)


def git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    return git_output(repo, "show", f"{commit}:{relative_path}")


def git_relative(path: Path) -> str:
    resolved = path.resolve()
    require(is_relative_to(resolved, REPO_ROOT), f"control escapes repository: {path}")
    return resolved.relative_to(REPO_ROOT).as_posix()


def execution_host() -> dict[str, str]:
    import scipy

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def load_controls(
    protocol_path: Path,
    preregistration_path: Path,
    schema_path: Path,
    *,
    execution_mode: bool,
) -> Controls:
    paths = (protocol_path.resolve(), preregistration_path.resolve(), schema_path.resolve())
    names = (PROTOCOL_NAME, PREREGISTRATION_NAME, RESULT_SCHEMA_NAME)
    for path, name in zip(paths, names, strict=True):
        require(path.parent == PACKAGE_DIR, f"control must be in task package: {path}")
        require(path.name == name, f"unexpected control filename: {path.name}")
        require(path.is_file(), f"missing control file: {path}")

    protocol, preregistration_path, schema_path = paths
    preregistration = canonical_control_json(preregistration_path)
    result_schema = canonical_control_json(schema_path)
    protocol_sha = sha256_path(protocol)
    prereg_sha = sha256_path(preregistration_path)
    schema_sha = sha256_path(schema_path)
    require(schema_sha == RESULT_SCHEMA_SHA256, "result schema digest mismatch")

    protocol_text = protocol.read_text(encoding="utf-8")
    require(prereg_sha in protocol_text, "protocol does not bind current preregistration")
    require(schema_sha in protocol_text, "protocol does not bind current result schema")
    runner_sha = sha256_path(Path(__file__).resolve())
    require(
        "RUNNER_SHA256_TO_BE_FROZEN_BEFORE_COMMIT" not in protocol_text,
        "protocol runner digest placeholder is not frozen",
    )
    require(runner_sha in protocol_text, "protocol does not bind current runner")
    require(
        preregistration.get("evidence_driver")
        == {"filename": RUNNER_NAME, "sha256": runner_sha},
        "preregistration does not bind current runner",
    )

    require(
        preregistration.get("schema_version")
        == "sci-cal-001-am12-el25-confirmation-preregistration-v1",
        "preregistration schema identity mismatch",
    )
    require(preregistration.get("package_id") == PACKAGE_ID, "package mismatch")
    require(
        preregistration.get("registration_status")
        == "frozen_before_any_confirmation_am_execution_or_result_inspection",
        "preregistration status is not frozen",
    )
    require(
        result_schema.get("$id")
        == preregistration["expected_artifacts"]["decision_schema"]["schema_id"],
        "result schema ID mismatch",
    )
    require(
        preregistration["expected_artifacts"]["decision_schema"]["sha256"]
        == schema_sha,
        "preregistration result-schema digest mismatch",
    )
    Draft202012Validator.check_schema(result_schema)
    validate_preregistration_semantics(preregistration)
    return Controls(
        protocol_path=protocol,
        preregistration_path=preregistration_path,
        schema_path=schema_path,
        protocol_sha256=protocol_sha,
        preregistration_sha256=prereg_sha,
        schema_sha256=schema_sha,
        preregistration=preregistration,
        result_schema=result_schema,
    )


def validate_preregistration_semantics(prereg: dict[str, Any]) -> None:
    support = prereg["support"]
    require(
        support
        == {
            "tau225_min": "0",
            "tau225_max": "0.158313198574890929",
            "tau225_endpoints_inclusive": True,
            "elevation_min_deg": "25",
            "elevation_max_deg": "80",
            "elevation_endpoints_inclusive": True,
            "q95_included": False,
            "outside_support_policy": "fail_closed",
            "zenith_opacity_identity": "tau225",
            "sample_airmass": "full_modified_secant_airmass_for_each_eligible_sample",
            "airmass_reference": "top_of_atmosphere_X_ref_0",
        },
        "support contract mismatch",
    )
    execution = prereg["atmosphere_execution"]
    require(execution["model"] == "AM 12.2", "AM model mismatch")
    require(
        execution["executable_sha256"] == EXPECTED_AM_EXECUTABLE_SHA256,
        "AM executable preregistration mismatch",
    )
    require(
        execution["source_payload_aggregate_sha256"] == EXPECTED_AM_SOURCE_SHA256,
        "AM source preregistration mismatch",
    )
    require(
        execution["frequency_grid"]
        == {
            "minimum_ghz": "0",
            "maximum_ghz": "500",
            "step_ghz": "0.01",
            "row_count": 50001,
        },
        "AM frequency-grid contract mismatch",
    )
    require(execution["root_iterations"] == 48, "root iteration mismatch")
    require(
        execution["maximum_bracket_expansions"] == 64,
        "bracket expansion mismatch",
    )
    require(execution["jobs"] == JOBS, "jobs mismatch")
    require(execution["omp_threads_per_process"] == OMP_THREADS, "OMP mismatch")
    require(execution["cache_shard_count"] == CACHE_SHARDS, "shard mismatch")
    require(
        execution["external_cache_basename"] == CACHE_BASENAME,
        "cache basename mismatch",
    )
    require(execution["locale"] == PINNED_LOCALE, "locale mismatch")
    require(
        p1_driver.ROOT_ITERATIONS == 48
        and p1_driver.MAX_BRACKET_EXPANSIONS == 64,
        "imported P1 solver constants changed",
    )
    require(tuple(prereg["integration"]["alpha_values"]) == ALPHAS, "alpha mismatch")
    require(
        prereg["passband_set"]["passband_set_id"] == PASSBAND_SET_ID,
        "passband-set mismatch",
    )
    require(
        prereg["passband_set"]["fts_challenger_included"] is False,
        "FTS must be excluded",
    )
    require(
        tuple(prereg["elevation_selection"]["elevations_deg"]) == ELEVATIONS_DEG,
        "elevation lattice mismatch",
    )
    require(
        prereg["profile_allocation"]["scale_case_count"] == 16,
        "scale-case count mismatch",
    )
    require(
        prereg["expected_coverage"]
        == {
            "opacity_coordinate_count": 6,
            "profile_scale_case_count": 16,
            "elevation_count": 56,
            "full_direct_am_grid_count": 896,
            "direct_band_alpha_truth_count": 10752,
            "candidate_count": 4,
            "array_count": 3,
            "alpha_count": 4,
            "expanded_candidate_row_count": 43008,
            "requested_predecessor_overlap_count": 0,
            "achieved_predecessor_overlap_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "duplicate_key_count": 0,
        },
        "coverage constants mismatch",
    )
    registered_candidates = [
        {
            **prereg["candidates"]["primary_confirmatory_candidate"],
            "role": "primary_confirmatory",
        },
        *[
            {**item, "role": "secondary_descriptive"}
            for item in prereg["candidates"]["secondary_descriptive_candidates"]
        ],
    ]
    require(tuple(registered_candidates) == CANDIDATES, "candidate order mismatch")
    require(
        prereg["candidates"]["secondary_may_rescue_primary_failure"] is False,
        "secondary rescue policy mismatch",
    )
    require(
        tuple(prereg["expected_artifacts"]["deterministic_package_artifacts"])
        == OUTPUT_NAMES,
        "artifact inventory mismatch",
    )


def coordinate_plan(prereg: dict[str, Any]) -> list[Coordinate]:
    selection = prereg["opacity_selection"]
    context_spec = selection["decimal_context"]
    require(
        context_spec["precision"] == 80
        and context_spec["rounding"] == "ROUND_HALF_EVEN",
        "Decimal context mismatch",
    )
    anchors = selection["anchor_coordinates"]
    anchor_ids = ("am_q0", "am_q25", "am_q50", "am_q75")
    require(tuple(item["anchor_id"] for item in anchors) == anchor_ids, "anchor order")
    anchor_tau = {item["anchor_id"]: Decimal(item["tau225_exact"]) for item in anchors}
    interval_anchors = {
        "q0_q25": (anchor_tau["am_q0"], anchor_tau["am_q25"]),
        "q25_q50": (anchor_tau["am_q25"], anchor_tau["am_q50"]),
        "q50_q75": (anchor_tau["am_q50"], anchor_tau["am_q75"]),
    }
    tolerance = Decimal(context_spec["stored_value_recomparison_tolerance"])
    result: list[Coordinate] = []
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        x80 = Decimal(selection["x80_modified_secant"])
        half_step = Decimal(selection["display_half_step"])
        for frozen in selection["coordinates"]:
            lo, hi = interval_anchors[frozen["interval"]]
            numerator = int(frozen["fraction_numerator"])
            denominator = int(frozen["fraction_denominator"])
            requested = ((denominator - numerator) * lo + numerator * hi) / denominator
            if "requested_tau225_rational" in frozen:
                rational_n, rational_d = frozen["requested_tau225_rational"].split("/")
                require(
                    requested == Decimal(rational_n) / Decimal(rational_d),
                    f"registered rational mismatch: {frozen['coordinate_id']}",
                )
            analytic = (-requested * x80).exp()
            literal = decimal_e6(analytic)
            represented = Decimal(literal)
            achieved = -(represented.ln()) / x80
            residual = achieved - requested
            negative = ((represented + half_step) / represented).ln() / x80
            positive = (represented / (represented - half_step)).ln() / x80
            comparisons = {
                "requested_tau225_exact": requested,
                "analytic_transmission_el80": analytic,
                "achieved_tau225_exact": achieved,
                "coordinate_residual_exact": residual,
                "negative_lower_tau_half_step_exact": negative,
                "positive_upper_tau_half_step_exact": positive,
            }
            require(
                literal == frozen["target_transmission_literal"],
                f"target literal mismatch: {frozen['coordinate_id']}",
            )
            for name, actual in comparisons.items():
                require(
                    abs(actual - Decimal(frozen[name])) <= tolerance,
                    f"stored Decimal mismatch: {frozen['coordinate_id']}/{name}",
                )
            require(
                -negative <= residual <= positive,
                f"asymmetric coordinate bound failed: {frozen['coordinate_id']}",
            )
            result.append(
                Coordinate(
                    coordinate_id=frozen["coordinate_id"],
                    interval=frozen["interval"],
                    fraction_numerator=numerator,
                    fraction_denominator=denominator,
                    requested_tau=requested,
                    analytic_transmission=analytic,
                    target_literal=literal,
                    achieved_tau=achieved,
                    residual=residual,
                    negative_bound=negative,
                    positive_bound=positive,
                )
            )
    require(len(result) == 6, "coordinate count mismatch")
    return result


def scale_cases(prereg: dict[str, Any], coordinates: Sequence[Coordinate]) -> list[ScaleCase]:
    by_interval = prereg["profile_allocation"]["by_interval"]
    cases = [
        ScaleCase(coordinate, profile)
        for coordinate in coordinates
        for profile in by_interval[coordinate.interval]
    ]
    require(len(cases) == 16, "scale-case expansion mismatch")
    require(
        len({(item.coordinate.coordinate_id, item.profile) for item in cases}) == 16,
        "duplicate scale case",
    )
    return cases


def validate_authority(prereg: dict[str, Any], coordination_repo: Path) -> dict[str, Any]:
    require(coordination_repo.is_dir(), f"missing coordination repository: {coordination_repo}")
    require(
        git_output(coordination_repo, "rev-parse", COORDINATION_BINDING_COMMIT)
        .decode()
        .strip()
        == COORDINATION_BINDING_COMMIT,
        "coordination binding commit did not resolve exactly",
    )
    authority = prereg["authority"]
    bindings = (
        ("decision", authority["decision_path"], authority["decision_sha256"]),
        (
            "passband_authority",
            authority["passband_record_path"],
            authority["passband_record_sha256"],
        ),
        (
            "align_dependency",
            authority["align_dependency_path"],
            authority["align_dependency_sha256"],
        ),
    )
    result: dict[str, Any] = {}
    for identity, path, expected_sha in bindings:
        data = git_blob(coordination_repo, COORDINATION_BINDING_COMMIT, path)
        actual_sha = sha256_bytes(data)
        require(actual_sha == expected_sha, f"authority digest mismatch: {identity}")
        result[identity] = {
            "path": path,
            "sha256": actual_sha,
            "coordination_commit": COORDINATION_BINDING_COMMIT,
        }
    return result


def validate_predecessors(
    prereg: dict[str, Any], p1: adoption_driver.P1Cache
) -> dict[str, Any]:
    bindings = prereg["predecessor_bindings"]
    expected = {
        "f401_package_sha256s_sha256": EXPECTED_F401_SHA256SUMS_SHA256,
        "completed_v2_manifest_sha256": EXPECTED_V2_MANIFEST_SHA256,
        "completed_v2_decision_sha256": EXPECTED_V2_DECISION_SHA256,
        "completed_v2_operator_nodes_sha256": EXPECTED_NODE_SHA256,
        "completed_v2_holdout_rows_sha256": EXPECTED_HOLDOUT_ROWS_SHA256,
        "completed_v2_runner_sha256": EXPECTED_V2_RUNNER_SHA256,
        "canonical_p1_runner_sha256": EXPECTED_P1_RUNNER_SHA256,
        "canonical_p1_execution_context_sha256": p1.context_sha256,
        "legacy_anchor_manifest_sha256": adoption_driver.LEGACY_ANCHOR_MANIFEST_SHA256,
    }
    require(bindings == expected, "predecessor binding constants mismatch")

    package_relative = PACKAGE_DIR.relative_to(REPO_ROOT).as_posix()
    frozen_sums = git_blob(REPO_ROOT, F401_HEAD, f"{package_relative}/SHA256SUMS")
    require(
        sha256_bytes(frozen_sums) == EXPECTED_F401_SHA256SUMS_SHA256,
        "f401 package SHA256SUMS digest mismatch",
    )
    frozen_lines = frozen_sums.decode("ascii").splitlines()
    require(len(frozen_lines) == 75, "f401 SHA256SUMS cardinality mismatch")
    require(
        all(re.fullmatch(r"[0-9a-f]{64}  [^/]+", line) for line in frozen_lines),
        "f401 SHA256SUMS grammar mismatch",
    )
    require(
        [line[66:] for line in frozen_lines]
        == sorted(line[66:] for line in frozen_lines),
        "f401 SHA256SUMS filenames are not sorted",
    )
    frozen_index = {line[66:]: line[:64] for line in frozen_lines}

    paths = {
        "manifest": PACKAGE_DIR / "am12_successor_adoption_manifest.json",
        "decision": PACKAGE_DIR / "am12_successor_decision.json",
        "nodes": PACKAGE_DIR / "am12_successor_operator_nodes.csv",
        "holdout_rows": PACKAGE_DIR / "am12_successor_holdout_rows.csv",
        "holdout_context": PACKAGE_DIR / "am12_successor_holdout_execution_context.json",
        "legacy_anchors": PACKAGE_DIR / "legacy_anchor_manifest.json",
        "v2_runner": Path(adoption_driver.__file__).resolve(),
        "p1_runner": Path(p1_driver.__file__).resolve(),
    }
    expected_paths = {
        "manifest": EXPECTED_V2_MANIFEST_SHA256,
        "decision": EXPECTED_V2_DECISION_SHA256,
        "nodes": EXPECTED_NODE_SHA256,
        "holdout_rows": EXPECTED_HOLDOUT_ROWS_SHA256,
        "legacy_anchors": adoption_driver.LEGACY_ANCHOR_MANIFEST_SHA256,
        "v2_runner": EXPECTED_V2_RUNNER_SHA256,
        "p1_runner": EXPECTED_P1_RUNNER_SHA256,
    }
    identities: dict[str, Any] = {}
    for identity, path in paths.items():
        require(path.is_file(), f"missing predecessor file: {path}")
        actual = sha256_path(path)
        if identity in expected_paths:
            require(actual == expected_paths[identity], f"predecessor digest: {identity}")
        identities[identity] = {
            "path_relative_to_package": (
                path.name if path.parent == PACKAGE_DIR else None
            ),
            "size_bytes": path.stat().st_size,
            "sha256": actual,
        }
    for name in (
        "am12_successor_adoption_manifest.json",
        "am12_successor_decision.json",
        "am12_successor_operator_nodes.csv",
        "am12_successor_holdout_rows.csv",
    ):
        require(
            frozen_index.get(name) == sha256_path(PACKAGE_DIR / name),
            f"f401 index/current predecessor mismatch: {name}",
        )

    manifest = json.loads(paths["manifest"].read_bytes())
    decision = json.loads(paths["decision"].read_bytes())
    require(manifest["decision"] == decision, "v2 manifest/decision mismatch")
    artifact_bindings = manifest["artifacts"]
    require(
        artifact_bindings[paths["nodes"].name]["sha256"] == EXPECTED_NODE_SHA256,
        "v2 manifest node binding mismatch",
    )
    require(
        artifact_bindings[paths["holdout_rows"].name]["sha256"]
        == EXPECTED_HOLDOUT_ROWS_SHA256,
        "v2 manifest holdout binding mismatch",
    )
    require(
        artifact_bindings[paths["holdout_context"].name]["sha256"]
        == identities["holdout_context"]["sha256"],
        "v2 manifest holdout-context binding mismatch",
    )
    require(
        frozen_index.get(paths["holdout_context"].name)
        == identities["holdout_context"]["sha256"],
        "f401 index/current holdout-context mismatch",
    )
    tau, t225 = adoption_driver.target_coordinates()
    return {
        "f401_evidence_head": F401_HEAD,
        "f401_package_sha256s": {
            "sha256": sha256_bytes(frozen_sums),
            "entry_count": len(frozen_lines),
        },
        "files": identities,
        "canonical_p1_execution_context_sha256": p1.context_sha256,
        "q0_q75_coordinate_source": {
            target: {
                "tau225": f17(tau[target]),
                "reference_t225_literal": t225[target],
            }
            for target in ("am_q0", "am_q25", "am_q50", "am_q75")
        },
    }


def predecessor_coordinate_sets() -> dict[str, set[tuple[Decimal, int]]]:
    manifest = json.loads(
        (PACKAGE_DIR / "am12_successor_adoption_manifest.json").read_bytes()
    )
    coordinate_source = manifest["coordinate_source"]["q0_q75_targets"]
    fit = {
        (Decimal(item["tau225_selector_anchor_binary64"]), elevation)
        for item in coordinate_source.values()
        for elevation in range(20, 81, 2)
    }
    require(len(fit) == 124, "predecessor fit-coordinate cardinality mismatch")

    context = json.loads(
        (PACKAGE_DIR / "am12_successor_holdout_execution_context.json").read_bytes()
    )
    requested: set[tuple[Decimal, int]] = set()
    achieved: set[tuple[Decimal, int]] = set()
    for item in context["holdout_plan"]:
        for elevation in item["elevations_deg"]:
            requested.add((Decimal(item["tau_mid"]), int(elevation)))
            achieved.add((Decimal(item["tau_achieved"]), int(elevation)))
    require(len(requested) == 90, "predecessor requested-coordinate count mismatch")
    require(len(achieved) == 90, "predecessor achieved-coordinate count mismatch")
    return {"fit": fit, "requested": requested, "achieved": achieved}


def independence_check(coordinates: Sequence[Coordinate]) -> dict[str, Any]:
    requested = {
        (coordinate.requested_tau, elevation)
        for coordinate in coordinates
        for elevation in ELEVATIONS_DEG
    }
    achieved = {
        (coordinate.achieved_tau, elevation)
        for coordinate in coordinates
        for elevation in ELEVATIONS_DEG
    }
    require(len(requested) == 336, "new requested-coordinate count mismatch")
    require(len(achieved) == 336, "new achieved-coordinate count mismatch")
    predecessor = predecessor_coordinate_sets()
    requested_overlaps = {
        name: requested & values for name, values in predecessor.items()
    }
    achieved_overlaps = {name: achieved & values for name, values in predecessor.items()}
    requested_count = sum(len(values) for values in requested_overlaps.values())
    achieved_count = sum(len(values) for values in achieved_overlaps.values())
    require(requested_count == 0, "requested-coordinate predecessor overlap")
    require(achieved_count == 0, "achieved-coordinate predecessor overlap")
    require(not requested & achieved, "requested and achieved confirmation keys overlap")
    return {
        "joint_key": ["exact_decimal_tau225", "integer_elevation_deg"],
        "profile_retained_as_additional_identity": True,
        "requested_confirmation_key_count": len(requested),
        "achieved_confirmation_key_count": len(achieved),
        "predecessor_fit_key_count": len(predecessor["fit"]),
        "predecessor_requested_tune_key_count": len(predecessor["requested"]),
        "predecessor_achieved_tune_key_count": len(predecessor["achieved"]),
        "requested_predecessor_overlap_count": requested_count,
        "achieved_predecessor_overlap_count": achieved_count,
        "pass": True,
    }


def load_primary_bandpasses(tolteca_repo: Path, prereg: dict[str, Any]) -> list[adoption_driver.Bandpass]:
    require(tolteca_repo.is_dir(), f"missing TolTECA repository: {tolteca_repo}")
    commit = prereg["passband_set"]["source_commit"]
    require(commit == adoption_driver.TOLTECA_COMMIT, "TolTECA commit mismatch")
    require(
        git_output(tolteca_repo, "rev-parse", commit).decode().strip() == commit,
        "TolTECA commit did not resolve exactly",
    )
    base = "tolteca/data/cal/toltec_passband"
    member_paths = {
        "index.yaml": f"{base}/index.yaml",
        **{
            f"data/{array}_passband.ecsv": adoption_driver.PRIMARY_BLOBS[array]
            for array in ARRAYS
        },
    }
    member_bytes = {
        relative: git_blob(tolteca_repo, commit, repository_path)
        for relative, repository_path in member_paths.items()
    }
    expected_sha = {
        "index.yaml": PASSBAND_INDEX_SHA256,
        **{
            f"data/{array}_passband.ecsv": adoption_driver.PRIMARY_SHA256[array]
            for array in ARRAYS
        },
    }
    for relative, data in member_bytes.items():
        require(
            sha256_bytes(data) == expected_sha[relative],
            f"passband member digest mismatch: {relative}",
        )
    require(
        sum(len(data) for data in member_bytes.values()) == PASSBAND_TOTAL_BYTES,
        "passband-set byte count mismatch",
    )
    aggregate = hashlib.sha256()
    for relative in sorted(member_bytes):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(sha256_bytes(member_bytes[relative])))
        aggregate.update(b"\0")
    require(aggregate.hexdigest() == PASSBAND_SET_DIGEST, "passband-set digest mismatch")

    bandpasses: list[adoption_driver.Bandpass] = []
    for array in ARRAYS:
        relative = f"data/{array}_passband.ecsv"
        frequency, response = adoption_driver.parse_primary_ecsv(
            member_bytes[relative], array
        )
        require(
            frequency[0] >= 0.0 and frequency[-1] <= 500.0,
            f"passband requires forbidden spectral extrapolation: {array}",
        )
        bandpasses.append(
            adoption_driver.Bandpass(
                identity=f"tolteca_v1_{array}",
                array=array,
                family="primary_tolteca_ecsv",
                frequency_ghz=frequency,
                response=response,
                source_path=adoption_driver.PRIMARY_BLOBS[array],
                source_sha256=adoption_driver.PRIMARY_SHA256[array],
                source_commit=commit,
                convention="ECSV throughput used as supplied",
            )
        )
    require(
        tuple(item.identity for item in bandpasses)
        == tuple(prereg["passband_set"]["selected_candidate_node_ids"][a] for a in ARRAYS),
        "selected passband node identity mismatch",
    )
    return bandpasses


def load_candidate_nodes(
    bandpasses: Sequence[adoption_driver.Bandpass],
) -> tuple[dict[tuple[str, str, int], np.ndarray], np.ndarray]:
    path = PACKAGE_DIR / "am12_successor_operator_nodes.csv"
    require(sha256_path(path) == EXPECTED_NODE_SHA256, "candidate node digest mismatch")
    tau_by_target, t225_by_target = adoption_driver.target_coordinates()
    tau_nodes = np.asarray(
        [tau_by_target[target] for target in ("am_q0", "am_q25", "am_q50", "am_q75")],
        dtype=np.float64,
    )
    selected = {item.identity: item.array for item in bandpasses}
    target_index = {"am_q25": 1, "am_q50": 2, "am_q75": 3}
    elevation_index = {
        int(value): index
        for index, value in enumerate(adoption_driver.ELEVATIONS_EVEN_DEG)
    }
    arrays = {
        (lane, passband.identity, alpha): np.zeros(
            (4, adoption_driver.ELEVATIONS_EVEN_DEG.size), dtype=np.float64
        )
        for lane in adoption_driver.LANES
        for passband in bandpasses
        for alpha in ALPHAS
    }
    seen: set[tuple[str, str, int, str, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            passband_id = row["passband_id"]
            if passband_id not in selected:
                continue
            lane = row["lane"]
            target = row["target"]
            elevation = int(row["elevation_deg"])
            alpha = int(row["alpha"])
            key = (lane, target, elevation, passband_id, alpha)
            require(key not in seen, f"duplicate selected candidate node: {key}")
            seen.add(key)
            require(lane in adoption_driver.LANES, f"unknown node lane: {lane}")
            require(target in target_index, f"unknown node target: {target}")
            require(elevation in elevation_index, f"unknown node elevation: {elevation}")
            require(alpha in ALPHAS, f"unknown node alpha: {alpha}")
            require(row["array"] == selected[passband_id], "node array mismatch")
            require(
                row["source_profile"] == adoption_driver.LANES[lane][target],
                "node source-profile mismatch",
            )
            require(float(row["tau225"]) == tau_by_target[target], "node tau mismatch")
            require(
                row["reference_t225_literal"] == t225_by_target[target],
                "node T225 mismatch",
            )
            los_tau = float(row["line_of_sight_optical_depth"])
            require(
                math.isfinite(los_tau)
                and abs(math.exp(los_tau) - float(row["extinction_correction"]))
                <= 5.0e-16 * math.exp(los_tau),
                "node correction mismatch",
            )
            arrays[(lane, passband_id, alpha)][
                target_index[target], elevation_index[elevation]
            ] = los_tau
    require(len(seen) == 2232, f"selected node coverage mismatch: {len(seen)}")
    require(
        all(np.count_nonzero(grid[1:]) == 93 for grid in arrays.values()),
        "selected node tensor is incomplete",
    )
    return arrays, tau_nodes


def run_git_binding(controls: Controls) -> dict[str, Any]:
    status = git_output(REPO_ROOT, "status", "--porcelain", "--untracked-files=all")
    require(status == b"", "--run-confirmation requires a clean repository")
    head = git_output(REPO_ROOT, "rev-parse", "HEAD").decode().strip()
    parent = git_output(REPO_ROOT, "rev-parse", "HEAD^").decode().strip()
    require(parent == F401_HEAD, "preregistration commit parent is not frozen f401 head")
    files = (
        Path(__file__).resolve(),
        controls.protocol_path,
        controls.preregistration_path,
        controls.schema_path,
    )
    identities: dict[str, Any] = {}
    for path in files:
        relative = git_relative(path)
        committed = git_blob(REPO_ROOT, head, relative)
        local = path.read_bytes()
        require(committed == local, f"HEAD does not contain exact control bytes: {relative}")
        identities[path.name] = {
            "path_relative_to_repository": relative,
            "size_bytes": len(local),
            "sha256": sha256_bytes(local),
        }
    return {
        "preregistration_commit": head,
        "preregistration_parent": parent,
        "control_files": identities,
    }


def replay_git_binding(context: dict[str, Any]) -> dict[str, Any]:
    binding = context["preregistration_git_binding"]
    commit = binding["preregistration_commit"]
    parent = binding["preregistration_parent"]
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid recorded commit")
    require(parent == F401_HEAD, "recorded preregistration parent mismatch")
    require(
        git_output(REPO_ROOT, "rev-parse", f"{commit}^").decode().strip() == parent,
        "recorded preregistration commit is inaccessible or has wrong parent",
    )
    for name, item in binding["control_files"].items():
        relative = item["path_relative_to_repository"]
        committed = git_blob(REPO_ROOT, commit, relative)
        require(len(committed) == item["size_bytes"], f"recorded size mismatch: {name}")
        require(sha256_bytes(committed) == item["sha256"], f"recorded digest mismatch: {name}")
        local = REPO_ROOT / relative
        require(local.is_file(), f"missing local control: {local}")
        require(local.read_bytes() == committed, f"local control differs from prereg commit: {name}")
    return binding


def validate_am_inputs(
    p1: adoption_driver.P1Cache,
    am_root: Path,
    executable: Path,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    require(am_root.is_dir(), f"missing copied AM root: {am_root}")
    require(executable.is_file(), f"missing native AM executable: {executable}")
    build = p1_driver.build_identity(executable)
    require(build.sha256 == EXPECTED_AM_EXECUTABLE_SHA256, "AM executable digest mismatch")
    require(build.binary_format == "mach-o", "AM executable is not the frozen Mach-O build")

    source_root = am_root / "am-12.2/src"
    require(source_root.is_dir(), f"missing AM source root: {source_root}")
    source_inventory = p1_driver.inventory_files(
        source_root, p1_driver.source_files(source_root)
    )
    require(source_inventory["file_count"] == 135, "AM source file count mismatch")
    require(
        source_inventory["aggregate_sha256"] == EXPECTED_AM_SOURCE_SHA256,
        "AM source payload aggregate mismatch",
    )
    require(
        p1.context["inputs"]["am_source_inventory"] == source_inventory,
        "actual AM source inventory differs from canonical P1 context",
    )
    contract_files = p1_driver.validate_am_contract_files(am_root)

    profile_root = am_root / "Big_Atmosphere/LMT_am_inputs"
    profile_rows = []
    for profile, expected_sha in sorted(prereg["profile_allocation"]["profile_sha256"].items()):
        path = profile_root / f"{profile}.amc"
        require(path.is_file(), f"missing AMC profile: {path}")
        actual_sha = sha256_path(path)
        require(actual_sha == expected_sha, f"AMC profile digest mismatch: {profile}")
        profile_rows.append(
            {
                "profile": profile,
                "path_relative_to_am_root": path.relative_to(am_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha,
            }
        )
    copied_inputs = p1.copied_npz_inventory([item["profile"] for item in profile_rows])
    require(copied_inputs["cross_inventory_identity_pass"], "copied NPZ provenance failed")
    return {
        "model": "AM 12.2",
        "executable": p1_driver.build_identity_payload(build),
        "source_inventory": source_inventory,
        "source_contract_files": contract_files,
        "profiles": profile_rows,
        "copied_scale1_npz_inputs": copied_inputs,
        "only_varied_profile_parameter": (
            "Nscale troposphere h2o through AMC argv %9"
        ),
    }


def build_execution_context(
    *,
    controls: Controls,
    git_binding: dict[str, Any],
    authority: dict[str, Any],
    predecessors: dict[str, Any],
    independence: dict[str, Any],
    am_inputs: dict[str, Any],
    p1: adoption_driver.P1Cache,
    bandpasses: Sequence[adoption_driver.Bandpass],
    coordinates: Sequence[Coordinate],
    cases: Sequence[ScaleCase],
    cache_dir: Path,
) -> dict[str, Any]:
    control_files = git_binding["control_files"]
    require(
        control_files[RUNNER_NAME]["sha256"] == sha256_path(Path(__file__).resolve()),
        "Git binding runner digest mismatch",
    )
    require(
        control_files[PROTOCOL_NAME]["sha256"] == controls.protocol_sha256,
        "Git binding protocol digest mismatch",
    )
    require(
        control_files[PREREGISTRATION_NAME]["sha256"]
        == controls.preregistration_sha256,
        "Git binding preregistration digest mismatch",
    )
    require(
        control_files[RESULT_SCHEMA_NAME]["sha256"] == controls.schema_sha256,
        "Git binding schema digest mismatch",
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}-execution-context-v1",
        "package_id": PACKAGE_ID,
        "study_id": STUDY_ID,
        "authorization": (
            "one_bounded_numerical_confirmation_no_operator_or_domain_authorization"
        ),
        "preregistration_git_binding": git_binding,
        "controls": {
            "protocol": {
                "filename": controls.protocol_path.name,
                "sha256": controls.protocol_sha256,
            },
            "preregistration": {
                "filename": controls.preregistration_path.name,
                "sha256": controls.preregistration_sha256,
            },
            "result_schema": {
                "filename": controls.schema_path.name,
                "sha256": controls.schema_sha256,
                "schema_id": controls.result_schema["$id"],
            },
            "runner": {
                "filename": RUNNER_NAME,
                "sha256": control_files[RUNNER_NAME]["sha256"],
            },
        },
        "authority": authority,
        "predecessors": predecessors,
        "imported_drivers": {
            "canonical_p1_runner": {
                "filename": Path(p1_driver.__file__).name,
                "sha256": sha256_path(Path(p1_driver.__file__).resolve()),
            },
            "frozen_v2_runner": {
                "filename": Path(adoption_driver.__file__).name,
                "sha256": sha256_path(Path(adoption_driver.__file__).resolve()),
            },
        },
        "canonical_p1_cache": {
            "external_cache_basename": p1.cache_dir.name,
            "execution_context_sha256": p1.context_sha256,
            "manifest_sha256": p1.manifest_sha256,
            "scale_table_sha256": p1.scales_sha256,
            "access": "shared_lock_read_only",
        },
        "atmosphere": am_inputs,
        "passband_set": {
            "passband_set_id": PASSBAND_SET_ID,
            "source_commit": adoption_driver.TOLTECA_COMMIT,
            "member_count": 4,
            "total_bytes": PASSBAND_TOTAL_BYTES,
            "members": [
                {
                    "passband_id": item.identity,
                    "array": item.array,
                    "source_path": item.source_path,
                    "sha256": item.source_sha256,
                }
                for item in bandpasses
            ],
            "fts_or_beammap_access": False,
        },
        "coordinate_plan": [
            {
                "coordinate_id": item.coordinate_id,
                "interval": item.interval,
                "fraction_numerator": item.fraction_numerator,
                "fraction_denominator": item.fraction_denominator,
                "requested_tau225_exact": str(item.requested_tau),
                "analytic_transmission_el80": str(item.analytic_transmission),
                "target_transmission_literal": item.target_literal,
                "achieved_tau225_exact": str(item.achieved_tau),
                "coordinate_residual_exact": str(item.residual),
                "negative_lower_tau_half_step_exact": str(item.negative_bound),
                "positive_upper_tau_half_step_exact": str(item.positive_bound),
            }
            for item in coordinates
        ],
        "scale_cases": [
            {
                "coordinate_id": item.coordinate.coordinate_id,
                "truth_profile": item.profile,
            }
            for item in cases
        ],
        "independence": independence,
        "execution_host": execution_host(),
        "execution_parameters": {
            "jobs": JOBS,
            "omp_threads_per_process": OMP_THREADS,
            "locale": PINNED_LOCALE,
            "cache_shard_count": CACHE_SHARDS,
            "root_iterations": 48,
            "maximum_bracket_expansions": 64,
            "frequency_grid": "0--500 GHz inclusive at 10 MHz",
            "elevations_deg": list(ELEVATIONS_DEG),
            "external_cache_basename": CACHE_BASENAME,
            "external_cache_resolved_path": str(cache_dir),
            "initial_state_for_run_confirmation": "path_absent",
        },
        "security": {
            "network_access": False,
            "unity_access": False,
            "beammap_or_fts_access": False,
            "citlali_application_code_modified": False,
            "sibling_repository_writes": False,
        },
    }


@contextmanager
def confirmation_cache_lock(cache_dir: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = cache_dir / CACHE_LOCK_NAME
    if not exclusive:
        require(lock_path.is_file(), f"missing confirmation cache lock: {lock_path}")
    mode = "a+b" if exclusive else "rb"
    with lock_path.open(mode) as handle:
        lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(handle.fileno(), lock_mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("confirmation cache is already locked") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_fresh_cache_root(cache_dir: Path) -> None:
    require(not cache_dir.exists(), "--run-confirmation requires an absent cache path")
    cache_dir.mkdir(parents=True, exist_ok=False)


def initialize_fresh_cache(cache_dir: Path, context_bytes: bytes) -> None:
    require(
        set(path.name for path in cache_dir.iterdir()) == {CACHE_LOCK_NAME},
        "fresh confirmation cache root changed before initialization",
    )
    for directory in (
        "raw_outputs",
        "execution_records",
        "scale_traces",
        "failed_attempts",
    ):
        (cache_dir / directory).mkdir()
    atomic_write(cache_dir / "execution_context.json", context_bytes)


def load_cached_context(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "execution_context.json"
    require(path.is_file(), f"missing confirmation execution context: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    require(raw == json_bytes(payload), "noncanonical confirmation execution context")
    return payload


def cache_evidence_inventory(cache_dir: Path) -> dict[str, Any]:
    relative_files: list[Path] = [Path("execution_context.json")]
    for directory, pattern in (
        ("raw_outputs", "*.txt"),
        ("execution_records", "*.run.json"),
        ("scale_traces", "*.json"),
        ("failed_attempts", "*"),
    ):
        relative_files.extend(
            path.relative_to(cache_dir)
            for path in sorted((cache_dir / directory).glob(pattern))
            if path.is_file()
        )
    aggregate = hashlib.sha256()
    total_bytes = 0
    entries = []
    for relative in sorted(relative_files, key=lambda item: item.as_posix()):
        path = cache_dir / relative
        digest = sha256_path(path)
        size = path.stat().st_size
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
        total_bytes += size
        entries.append({"path": relative.as_posix(), "size_bytes": size, "sha256": digest})
    return {
        "algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "failed_attempt_file_count": sum(
            item["path"].startswith("failed_attempts/") for item in entries
        ),
    }


def strict_warning_diagnostics(
    result: p1_driver.RunResult, raw: bytes
) -> dict[str, int | None]:
    parsed = result.parsed
    raw_lines = raw.split(b"\n")
    raw_warning_headers = [
        line for line in raw_lines if line.startswith(RAW_WARNING_PREFIX)
    ]
    raw_error_headers = [
        line for line in raw_lines if line.startswith(RAW_ERROR_PREFIX)
    ]
    raw_summary_header_count = sum(
        line == RAW_UNRESOLVED_SUMMARY_HEADER for line in raw_warning_headers
    )
    raw_column_warning_count = sum(
        RAW_UNRESOLVED_COLUMN_WARNING.fullmatch(line) is not None
        for line in raw_warning_headers
    )
    raw_unknown_warning_count = sum(
        line != RAW_UNRESOLVED_SUMMARY_HEADER
        and RAW_UNRESOLVED_COLUMN_WARNING.fullmatch(line) is None
        for line in raw_warning_headers
    )
    mutation_count = len(CACHE_MUTATION_DIAGNOSTIC.findall(raw))
    diagnostics: dict[str, int | None] = {
        "return_code": result.return_code,
        "numeric_row_count": int(parsed.samples.shape[0]),
        "unresolved_line_warning_count": parsed.warning_count,
        "unresolved_column_warning_line_count": raw_column_warning_count,
        "unresolved_summary_warning_line_count": raw_summary_header_count,
        "other_warning_line_count": raw_unknown_warning_count,
        "error_line_count": len(raw_error_headers),
        "cache_mutation_warning_count": mutation_count,
    }
    if result.return_code == 0:
        admitted = (
            parsed.warning_count is None
            and parsed.unresolved_column_warning_line_count == 0
            and parsed.unresolved_summary_warning_line_count == 0
            and parsed.other_warning_line_count == 0
            and parsed.error_line_count == 0
            and not raw_warning_headers
            and not raw_error_headers
            and mutation_count == 0
        )
    elif result.return_code == 1:
        admitted = (
            result.spec.expected_rows == 50001
            and parsed.samples.shape[0] == 50001
            and parsed.warning_count in {86, 87, 88}
            and parsed.unresolved_summary_warning_line_count == 1
            and parsed.other_warning_line_count == 0
            and parsed.error_line_count == 0
            and raw_summary_header_count == 1
            and raw_column_warning_count
            == parsed.unresolved_column_warning_line_count
            and raw_column_warning_count > 0
            and raw_unknown_warning_count == 0
            and not raw_error_headers
            and mutation_count == 0
        )
    else:
        admitted = False
    require(admitted, f"strict WARN-001 rejection: {result.cache_id}/{diagnostics}")
    return diagnostics


class StrictRunner(p1_driver.Runner):
    """Canonical P1 runner with the narrower confirmation status contract."""

    def run_or_load(self, spec: p1_driver.RunSpec) -> p1_driver.RunResult:
        result = super().run_or_load(spec)
        raw = self.raw_path(result.cache_id).read_bytes()
        strict_warning_diagnostics(result, raw)
        require(
            set(result.sidecar) == adoption_driver.CANONICAL_RUN_SIDECAR_KEYS,
            f"confirmation sidecar key-set mismatch: {result.cache_id}",
        )
        return result


def valid_align_state(elevation_deg: float, status: str = "original_eligible") -> dict[str, Any]:
    return {
        "sample_identity": f"confirmation-probe-EL{elevation_deg:g}",
        "aligned_elevation_deg": elevation_deg,
        "timing_gap_or_interpolation_origin": "direct_confirmation_coordinate",
        "duration_s": 1.0,
        "original_or_synthesized_eligibility": status,
        "eligible": True,
    }


def validate_align_state(state: dict[str, Any], elevation_deg: float) -> None:
    required = {
        "sample_identity",
        "aligned_elevation_deg",
        "timing_gap_or_interpolation_origin",
        "duration_s",
        "original_or_synthesized_eligibility",
        "eligible",
    }
    if not isinstance(state, dict) or not required <= set(state):
        raise ValueError("incomplete ALIGN state")
    if not isinstance(state["sample_identity"], str) or not state[
        "sample_identity"
    ].strip():
        raise ValueError("invalid sample identity")
    if not isinstance(state["timing_gap_or_interpolation_origin"], str) or not state[
        "timing_gap_or_interpolation_origin"
    ].strip():
        raise ValueError("invalid timing/interpolation origin")
    if state["original_or_synthesized_eligibility"] not in {
        "original_eligible",
        "synthesized_eligible",
    }:
        raise ValueError("invalid original/synthesized eligibility")
    if state["eligible"] is not True:
        raise ValueError("sample is not explicitly eligible")
    aligned = state["aligned_elevation_deg"]
    duration = state["duration_s"]
    if isinstance(aligned, bool) or not isinstance(aligned, (int, float)):
        raise ValueError("invalid aligned elevation type")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("invalid duration type")
    if not math.isfinite(float(aligned)) or float(aligned) != float(elevation_deg):
        raise ValueError("aligned elevation does not match operator elevation")
    if not math.isfinite(float(duration)) or float(duration) <= 0.0:
        raise ValueError("duration is not finite and positive")


def evaluate_confirmation_scalar(
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    tau_nodes: np.ndarray,
    *,
    lane: str,
    passband_id: str,
    alpha: int,
    operator: str,
    tau225: float,
    elevation_deg: float,
    align_state: dict[str, Any],
) -> float:
    if (
        not math.isfinite(float(tau225))
        or not math.isfinite(float(elevation_deg))
        or float(tau225) < TAU_MIN
        or float(tau225) > TAU_MAX
        or float(elevation_deg) < ELEVATION_MIN_DEG
        or float(elevation_deg) > ELEVATION_MAX_DEG
    ):
        raise ValueError("coordinate outside closed confirmation support")
    validate_align_state(align_state, float(elevation_deg))
    values = adoption_driver.evaluate_named_operator(
        node_arrays,
        lane=lane,
        passband_id=passband_id,
        alpha=alpha,
        operator=operator,
        tau_nodes=tau_nodes,
        tau_query=np.asarray([tau225], dtype=np.float64),
        elevation_query_deg=np.asarray([elevation_deg], dtype=np.float64),
    )
    result = float(values[0, 0])
    if not math.isfinite(result):
        raise ValueError("nonfinite operator result")
    return result


def fail_closed_probe_pass(
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    tau_nodes: np.ndarray,
    candidate: dict[str, str],
    passband_id: str,
    alpha: int,
) -> bool:
    def evaluate(
        tau: float,
        elevation: float,
        state: dict[str, Any],
        *,
        lane: str = candidate["lane"],
        selected_passband: str = passband_id,
        selected_alpha: int = alpha,
        operator: str = candidate["operator"],
    ) -> float:
        return evaluate_confirmation_scalar(
            node_arrays,
            tau_nodes,
            lane=lane,
            passband_id=selected_passband,
            alpha=selected_alpha,
            operator=operator,
            tau225=tau,
            elevation_deg=elevation,
            align_state=state,
        )

    try:
        for status in ("original_eligible", "synthesized_eligible"):
            for elevation in (25.0, 50.0, 80.0):
                evaluate(float(tau_nodes[1]), elevation, valid_align_state(elevation, status))
        for tau in (TAU_MIN, TAU_MAX):
            for elevation in (ELEVATION_MIN_DEG, ELEVATION_MAX_DEG):
                evaluate(tau, elevation, valid_align_state(elevation))
    except (ValueError, KeyError):
        return False

    base_state = valid_align_state(50.0)
    rejected: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for field in tuple(base_state):
        state = dict(base_state)
        del state[field]
        rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    for field in (
        "sample_identity",
        "timing_gap_or_interpolation_origin",
        "original_or_synthesized_eligibility",
    ):
        state = dict(base_state)
        state[field] = ""
        rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    state = dict(base_state)
    state["original_or_synthesized_eligibility"] = "unknown"
    rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    for eligibility in (False, 1, "true", None):
        state = dict(base_state)
        state["eligible"] = eligibility
        rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    state = dict(base_state)
    state["aligned_elevation_deg"] = np.nextafter(50.0, math.inf)
    rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    for duration in (0.0, -1.0, math.nan, math.inf, -math.inf):
        state = dict(base_state)
        state["duration_s"] = duration
        rejected.append((float(tau_nodes[1]), 50.0, state, {}))
    for tau in (
        np.nextafter(TAU_MIN, -math.inf),
        np.nextafter(TAU_MAX, math.inf),
        math.nan,
        math.inf,
        -math.inf,
    ):
        rejected.append((float(tau), 50.0, dict(base_state), {}))
    for elevation in (
        np.nextafter(ELEVATION_MIN_DEG, -math.inf),
        np.nextafter(ELEVATION_MAX_DEG, math.inf),
        math.nan,
        math.inf,
        -math.inf,
    ):
        rejected.append(
            (
                float(tau_nodes[1]),
                float(elevation),
                valid_align_state(float(elevation)),
                {},
            )
        )
    rejected.extend(
        [
            (
                float(tau_nodes[1]),
                50.0,
                dict(base_state),
                {"lane": "unknown_lane"},
            ),
            (
                float(tau_nodes[1]),
                50.0,
                dict(base_state),
                {"selected_passband": "unknown_passband"},
            ),
            (
                float(tau_nodes[1]),
                50.0,
                dict(base_state),
                {"selected_alpha": 999},
            ),
            (
                float(tau_nodes[1]),
                50.0,
                dict(base_state),
                {"operator": "unknown_operator"},
            ),
        ]
    )
    for tau, elevation, state, overrides in rejected:
        try:
            evaluate(tau, elevation, state, **overrides)
        except (ValueError, KeyError, TypeError):
            continue
        return False

    nodes = node_arrays[(candidate["lane"], passband_id, alpha)]
    incomplete = (
        (
            nodes[:3],
            tau_nodes[:3],
            adoption_driver.ELEVATIONS_EVEN_DEG,
            tau_nodes[-1],
            50.0,
        ),
        (
            nodes[:, :-1],
            tau_nodes,
            adoption_driver.ELEVATIONS_EVEN_DEG[:-1],
            tau_nodes[1],
            ELEVATION_MAX_DEG,
        ),
    )
    for (
        selected_nodes,
        selected_tau,
        selected_elevation,
        query_tau,
        query_elevation,
    ) in incomplete:
        try:
            adoption_driver.evaluate_operator_grid(
                selected_nodes,
                selected_tau,
                selected_elevation,
                np.asarray([query_tau]),
                np.asarray([query_elevation]),
                candidate["operator"],
            )
        except ValueError:
            continue
        return False
    return True


def physical_metrics(
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    tau_nodes: np.ndarray,
    bandpasses: Sequence[adoption_driver.Bandpass],
    coordinates: Sequence[Coordinate],
) -> list[dict[str, str]]:
    achieved = np.asarray([float(item.achieved_tau) for item in coordinates])
    dense_tau = np.unique(
        np.concatenate([np.linspace(TAU_MIN, TAU_MAX, 1001), tau_nodes, achieved])
    )
    even_inside = adoption_driver.ELEVATIONS_EVEN_DEG[
        adoption_driver.ELEVATIONS_EVEN_DEG >= ELEVATION_MIN_DEG
    ]
    dense_elevation = np.unique(
        np.concatenate(
            [
                np.linspace(ELEVATION_MIN_DEG, ELEVATION_MAX_DEG, 551),
                np.asarray([ELEVATION_MIN_DEG, ELEVATION_MAX_DEG]),
                even_inside,
            ]
        )
    )
    rows: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        for bandpass in bandpasses:
            for alpha in ALPHAS:
                nodes = node_arrays[(candidate["lane"], bandpass.identity, alpha)]
                evaluated = adoption_driver.evaluate_operator_grid(
                    nodes,
                    tau_nodes,
                    adoption_driver.ELEVATIONS_EVEN_DEG,
                    dense_tau,
                    dense_elevation,
                    candidate["operator"],
                )
                finite = bool(np.all(np.isfinite(evaluated)))
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    transmission = np.exp(-evaluated)
                    correction = np.exp(evaluated)
                domain_pass = bool(
                    finite
                    and np.all(transmission > 0.0)
                    and np.all(transmission <= math.exp(PHYSICAL_TOLERANCE))
                    and np.all(correction > 0.0)
                    and np.all(correction >= math.exp(-PHYSICAL_TOLERANCE))
                )
                minimum = float(np.min(evaluated))
                minimum_index = np.unravel_index(int(np.argmin(evaluated)), evaluated.shape)
                tau_deltas = np.diff(evaluated, axis=0)
                elevation_deltas = np.diff(evaluated, axis=1)
                tau_min_delta = float(np.min(tau_deltas))
                elevation_max_delta = float(np.max(elevation_deltas))
                tau_wrong = tau_deltas < -PHYSICAL_TOLERANCE
                elevation_wrong = elevation_deltas > PHYSICAL_TOLERANCE
                tau_wrong_count = int(np.count_nonzero(tau_wrong))
                elevation_wrong_count = int(np.count_nonzero(elevation_wrong))
                tau_excursion = (
                    float(np.max(np.abs(np.expm1(tau_deltas[tau_wrong]))))
                    if tau_wrong_count
                    else 0.0
                )
                elevation_excursion = (
                    float(np.max(np.abs(np.expm1(elevation_deltas[elevation_wrong]))))
                    if elevation_wrong_count
                    else 0.0
                )

                exact = adoption_driver.evaluate_operator_grid(
                    nodes,
                    tau_nodes,
                    adoption_driver.ELEVATIONS_EVEN_DEG,
                    tau_nodes,
                    adoption_driver.ELEVATIONS_EVEN_DEG,
                    candidate["operator"],
                )
                internal_anchor_residual = float(np.max(np.abs(exact - nodes)))
                low_tau = dense_tau[dense_tau <= tau_nodes[1]]
                low_values = adoption_driver.evaluate_operator_grid(
                    nodes,
                    tau_nodes,
                    adoption_driver.ELEVATIONS_EVEN_DEG,
                    low_tau,
                    dense_elevation,
                    candidate["operator"],
                )
                q25_values = adoption_driver.evaluate_operator_grid(
                    nodes,
                    tau_nodes,
                    adoption_driver.ELEVATIONS_EVEN_DEG,
                    np.asarray([tau_nodes[1]]),
                    dense_elevation,
                    candidate["operator"],
                )[0]
                low_expected = (low_tau[:, None] / tau_nodes[1]) * q25_values[None, :]
                low_residual = float(np.max(np.abs(low_values - low_expected)))

                continuity_residuals: list[float] = []
                span = tau_nodes[-1] - tau_nodes[0]
                for knot in tau_nodes[1:-1]:
                    queries = np.asarray(
                        [
                            np.nextafter(knot, -math.inf),
                            np.nextafter(knot, math.inf),
                            knot - 1.0e-12 * span,
                            knot + 1.0e-12 * span,
                        ]
                    )
                    values = adoption_driver.evaluate_operator_grid(
                        nodes,
                        tau_nodes,
                        adoption_driver.ELEVATIONS_EVEN_DEG,
                        queries,
                        dense_elevation,
                        candidate["operator"],
                    )
                    continuity_residuals.extend(
                        np.abs(np.expm1(values[0] - values[1])).tolist()
                    )
                    continuity_residuals.extend(
                        np.abs(np.expm1(values[2] - values[3])).tolist()
                    )
                max_continuity = max(continuity_residuals)
                fail_closed = fail_closed_probe_pass(
                    node_arrays, tau_nodes, candidate, bandpass.identity, alpha
                )
                positivity_pass = minimum >= -PHYSICAL_TOLERANCE
                tau_pass = tau_min_delta >= -PHYSICAL_TOLERANCE
                elevation_pass = elevation_max_delta <= PHYSICAL_TOLERANCE
                continuity_pass = max_continuity <= CONTINUITY_TOLERANCE
                anchor_pass = internal_anchor_residual <= ANCHOR_TOLERANCE
                low_pass = low_residual <= LOW_SEGMENT_TOLERANCE
                contract_pass = all(
                    (
                        finite,
                        domain_pass,
                        positivity_pass,
                        tau_pass,
                        elevation_pass,
                        continuity_pass,
                        anchor_pass,
                        low_pass,
                    )
                )
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_role": candidate["role"],
                        "lane": candidate["lane"],
                        "operator": candidate["operator"],
                        "passband_id": bandpass.identity,
                        "array": bandpass.array,
                        "alpha": str(alpha),
                        "all_evaluated_quantities_finite": bool_text(finite),
                        "minimum_line_of_sight_optical_depth": f17(minimum),
                        "minimum_lambda_tau225": f17(float(dense_tau[minimum_index[0]])),
                        "minimum_lambda_elevation_deg": f17(
                            float(dense_elevation[minimum_index[1]])
                        ),
                        "maximum_effective_transmission": f17(float(np.max(transmission))),
                        "minimum_extinction_correction": f17(float(np.min(correction))),
                        "minimum_tau_direction_delta": f17(tau_min_delta),
                        "maximum_elevation_direction_delta": f17(elevation_max_delta),
                        "tau_wrong_way_step_count": str(tau_wrong_count),
                        "elevation_wrong_way_step_count": str(elevation_wrong_count),
                        "maximum_tau_wrong_way_fractional_correction_excursion": f17(
                            tau_excursion
                        ),
                        "maximum_elevation_wrong_way_fractional_correction_excursion": f17(
                            elevation_excursion
                        ),
                        "maximum_internal_anchor_absolute_residual": f17(
                            internal_anchor_residual
                        ),
                        "maximum_low_segment_absolute_residual": f17(low_residual),
                        "maximum_relative_correction_continuity_residual": f17(
                            max_continuity
                        ),
                        "positivity_pass": bool_text(positivity_pass),
                        "domain_pass": bool_text(domain_pass),
                        "tau_monotonicity_pass": bool_text(tau_pass),
                        "elevation_monotonicity_pass": bool_text(elevation_pass),
                        "continuity_pass": bool_text(continuity_pass),
                        "fail_closed_pass": bool_text(fail_closed),
                        "internal_anchor_pass": bool_text(anchor_pass),
                        "exact_low_segment_pass": bool_text(low_pass),
                        "physical_contract_pass": bool_text(contract_pass),
                    }
                )
    require(len(rows) == 48, "physical metric coverage mismatch")
    return sorted(
        rows,
        key=lambda row: (row["candidate_id"], row["array"], int(row["alpha"])),
    )


def parsed_am(result: p1_driver.RunResult) -> adoption_driver.ParsedAM:
    samples = result.parsed.samples
    return adoption_driver.ParsedAM(
        frequency_ghz=np.asarray(samples[:, 0], dtype=np.float64),
        tau_los=np.asarray(samples[:, 1], dtype=np.float64),
        transmission=np.asarray(samples[:, 2], dtype=np.float64),
        numeric_sha256=result.parsed.numeric_text_sha256,
        normalized_sha256=result.parsed.normalized_output_sha256,
        version=result.parsed.version_identity,
        unresolved_lines=result.parsed.warning_count,
    )


def optional_f17(value: float | None) -> str:
    return "" if value is None else f17(value)


def solve_and_evaluate(
    *,
    runner: StrictRunner,
    p1: adoption_driver.P1Cache,
    cases: Sequence[ScaleCase],
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    tau_nodes: np.ndarray,
    bandpasses: Sequence[adoption_driver.Bandpass],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[p1_driver.ScaleSolution],
    dict[str, dict[str, Any]],
    set[tuple[str, str, int, str, int]],
]:
    scale_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    solutions: list[p1_driver.ScaleSolution] = []
    full_run_meta: dict[str, dict[str, Any]] = {}
    direct_truth_keys: set[tuple[str, str, int, str, int]] = set()

    for case in cases:
        coordinate = case.coordinate
        target = coordinate.coordinate_id
        p1_driver.EXPECTED_TARGET_TRANSMISSIONS[target] = coordinate.target_literal
        scale0 = runner.run_or_load(
            p1_driver.anchor_spec(case.profile, target, f17(0.0))
        )
        scale1 = runner.run_or_load(
            p1_driver.anchor_spec(case.profile, target, f17(1.0))
        )
        copied_tau, copied_transmission = p1_driver.copied_anchor(
            p1.am_root, case.profile
        )
        solution = p1_driver.solve_scale_hypothesis(
            runner=runner,
            profile=case.profile,
            target=target,
            scale0=scale0,
            scale1=scale1,
            copied_scale1_tau=copied_tau,
            copied_scale1_transmission=copied_transmission,
        )
        require(solution.exact_parsed_transmission_match, f"no exact scale plateau: {target}/{case.profile}")
        achieved_transmission = p1_driver.anchor_values(solution.fitted)[1]
        require(
            achieved_transmission == float(coordinate.target_literal),
            f"parsed T225 mismatch: {target}/{case.profile}",
        )
        achieved_from_literal = -math.log(achieved_transmission) / float(
            Decimal("1.01538872688246729")
        )
        require(
            abs(achieved_from_literal - float(coordinate.achieved_tau)) <= 5.0e-17,
            f"achieved coordinate mismatch: {target}/{case.profile}",
        )
        solutions.append(solution)
        scale_rows.append(
            {
                "coordinate_id": target,
                "interval": coordinate.interval,
                "fraction_numerator": str(coordinate.fraction_numerator),
                "fraction_denominator": str(coordinate.fraction_denominator),
                "truth_profile": case.profile,
                "requested_tau225": f17(float(coordinate.requested_tau)),
                "achieved_tau225": f17(float(coordinate.achieved_tau)),
                "coordinate_residual": f17(float(coordinate.residual)),
                "analytic_transmission_decimal": str(
                    coordinate.analytic_transmission
                ),
                "target_transmission_literal": coordinate.target_literal,
                "achieved_transmission_el80": f17(achieved_transmission),
                "h2o_scale_decimal": solution.scale_decimal,
                "h2o_scale_hex": float(solution.scale_decimal).hex(),
                "negative_lower_tau_half_step": str(coordinate.negative_bound),
                "positive_upper_tau_half_step": str(coordinate.positive_bound),
                "plateau_lower_outside_scale": optional_f17(
                    solution.plateau_lower_outside_scale
                ),
                "plateau_lower_inside_scale": optional_f17(
                    solution.plateau_lower_inside_scale
                ),
                "plateau_upper_inside_scale": optional_f17(
                    solution.plateau_upper_inside_scale
                ),
                "plateau_upper_outside_scale": optional_f17(
                    solution.plateau_upper_outside_scale
                ),
                "trace_path_relative_to_cache": solution.trace_relative_path,
                "trace_sha256": solution.trace_sha256,
            }
        )

        specs = [
            p1_driver.full_grid_spec(
                "confirmation_trisection_integer_elevation_full_grid",
                case.profile,
                target,
                90 - elevation,
                solution.scale_decimal,
            )
            for elevation in ELEVATIONS_DEG
        ]
        runs = runner.run_many(specs, JOBS)
        for run in runs:
            elevation = run.spec.elevation_deg
            if elevation == 80:
                require(
                    run.parsed.samples[22500, 0] == 225.0
                    and run.parsed.samples[22500, 2]
                    == float(coordinate.target_literal),
                    f"full-grid EL80 T225 mismatch: {target}/{case.profile}",
                )
            sidecar_path = runner.sidecar_path(run.cache_id)
            truth = TruthRecord(
                coordinate=coordinate,
                profile=case.profile,
                elevation_deg=elevation,
                scale_decimal=solution.scale_decimal,
                scale_hex=float(solution.scale_decimal).hex(),
                achieved_transmission=achieved_transmission,
                parsed=parsed_am(run),
                raw_sha256=run.raw_sha256,
                sidecar_sha256=sha256_path(sidecar_path),
                cache_id=run.cache_id,
            )
            integrated = adoption_driver.integrate_record(truth, bandpasses)
            full_run_meta[run.cache_id] = {
                "coordinate_id": target,
                "truth_profile": case.profile,
                "elevation_deg": elevation,
                "trace_path": solution.trace_relative_path,
                "trace_sha256": solution.trace_sha256,
            }
            for bandpass in bandpasses:
                for alpha in ALPHAS:
                    direct_key = (
                        target,
                        case.profile,
                        elevation,
                        bandpass.array,
                        alpha,
                    )
                    require(direct_key not in direct_truth_keys, f"duplicate direct truth: {direct_key}")
                    direct_truth_keys.add(direct_key)
                    effective, direct_tau, direct_correction = integrated[
                        (bandpass.identity, alpha)
                    ]
                    for candidate in CANDIDATES:
                        operator_tau = evaluate_confirmation_scalar(
                            node_arrays,
                            tau_nodes,
                            lane=candidate["lane"],
                            passband_id=bandpass.identity,
                            alpha=alpha,
                            operator=candidate["operator"],
                            tau225=float(coordinate.achieved_tau),
                            elevation_deg=float(elevation),
                            align_state=valid_align_state(float(elevation)),
                        )
                        operator_correction = math.exp(operator_tau)
                        error = operator_correction / direct_correction - 1.0
                        comparison_rows.append(
                            {
                                "coordinate_id": target,
                                "interval": coordinate.interval,
                                "fraction_numerator": str(
                                    coordinate.fraction_numerator
                                ),
                                "fraction_denominator": str(
                                    coordinate.fraction_denominator
                                ),
                                "truth_profile": case.profile,
                                "requested_tau225": f17(
                                    float(coordinate.requested_tau)
                                ),
                                "achieved_tau225": f17(
                                    float(coordinate.achieved_tau)
                                ),
                                "coordinate_residual": f17(
                                    float(coordinate.residual)
                                ),
                                "h2o_scale_decimal": solution.scale_decimal,
                                "h2o_scale_hex": truth.scale_hex,
                                "target_transmission_literal": (
                                    coordinate.target_literal
                                ),
                                "achieved_transmission_el80": f17(
                                    achieved_transmission
                                ),
                                "elevation_deg": str(elevation),
                                "airmass": f17(
                                    float(adoption_driver.modified_airmass(elevation))
                                ),
                                "candidate_id": candidate["candidate_id"],
                                "candidate_role": candidate["role"],
                                "lane": candidate["lane"],
                                "operator": candidate["operator"],
                                "passband_set_id": PASSBAND_SET_ID,
                                "passband_id": bandpass.identity,
                                "array": bandpass.array,
                                "alpha": str(alpha),
                                "direct_effective_transmission": f17(effective),
                                "direct_line_of_sight_optical_depth": f17(
                                    direct_tau
                                ),
                                "direct_extinction_correction": f17(
                                    direct_correction
                                ),
                                "operator_line_of_sight_optical_depth": f17(
                                    operator_tau
                                ),
                                "operator_extinction_correction": f17(
                                    operator_correction
                                ),
                                "signed_fractional_correction_error": f17(error),
                                "absolute_fractional_correction_error": f17(
                                    abs(error)
                                ),
                                "raw_sha256": run.raw_sha256,
                                "sidecar_sha256": truth.sidecar_sha256,
                                "_error": error,
                            }
                        )
        del runs
    require(len(scale_rows) == 16, "scale table coverage mismatch")
    require(len(full_run_meta) == 896, "full-grid cache identity coverage mismatch")
    require(len(direct_truth_keys) == 10752, "direct truth coverage mismatch")
    require(len(comparison_rows) == 43008, "comparison row coverage mismatch")
    return scale_rows, comparison_rows, solutions, full_run_meta, direct_truth_keys


def build_run_inventory(
    runner: StrictRunner,
    solutions: Sequence[p1_driver.ScaleSolution],
    full_run_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_runs: dict[str, dict[str, Any]] = {}
    trace_by_case: dict[tuple[str, str], dict[str, str]] = {}
    for solution in solutions:
        trace_path = runner.cache_dir / solution.trace_relative_path
        require(sha256_path(trace_path) == solution.trace_sha256, "scale trace digest mismatch")
        raw_trace = trace_path.read_bytes()
        trace = json.loads(raw_trace)
        require(raw_trace == p1_driver.json_bytes(trace), "noncanonical P1 scale trace")
        require(
            trace["execution_context_sha256"] == runner.execution_context_sha256
            and trace["target"] == solution.target
            and trace["profile"] == solution.profile
            and trace["root_iterations"] == 48
            and trace["maximum_bracket_expansions"] == 64
            and len(trace["evaluations"]) == solution.trace_evaluation_count,
            "scale trace identity mismatch",
        )
        case_key = (solution.target, solution.profile)
        require(case_key not in trace_by_case, f"duplicate solution trace: {case_key}")
        trace_by_case[case_key] = {
            "scale_trace_path_relative_to_cache": solution.trace_relative_path,
            "scale_trace_sha256": solution.trace_sha256,
        }
        for index, evaluation in enumerate(trace["evaluations"]):
            require(evaluation["evaluation_index"] == index, "nonsequential scale trace")
            spec = p1_driver.anchor_spec(
                solution.profile, solution.target, evaluation["scale_decimal"]
            )
            cache_id = runner.cache_id(spec)
            require(cache_id not in trace_runs, f"duplicate scale-run identity: {cache_id}")
            trace_runs[cache_id] = {
                "coordinate_id": solution.target,
                "truth_profile": solution.profile,
                "scale_trace_path_relative_to_cache": solution.trace_relative_path,
                "scale_trace_sha256": solution.trace_sha256,
                "scale_trace_evaluation_index": str(index),
                "scale_trace_role": evaluation["role"],
            }

    require(not set(trace_runs) & set(full_run_meta), "scale/full run identity overlap")
    expected_ids = set(trace_runs) | set(full_run_meta)
    observed = {item.cache_id: item for item in runner.observed_runs()}
    require(set(observed) == expected_ids, "Runner observation anti-join failed")
    raw_ids = {
        path.name.removesuffix(".txt")
        for path in (runner.cache_dir / "raw_outputs").glob("*.txt")
    }
    sidecar_ids = {
        path.name.removesuffix(".run.json")
        for path in (runner.cache_dir / "execution_records").glob("*.run.json")
    }
    failed = [
        path
        for path in (runner.cache_dir / "failed_attempts").iterdir()
        if path.is_file()
    ]
    require(raw_ids == expected_ids, "raw-output cache anti-join failed")
    require(sidecar_ids == expected_ids, "sidecar cache anti-join failed")
    require(not failed, "failed_attempts is not empty")

    rows: list[dict[str, Any]] = []
    warning = {
        "status_0_run_count": 0,
        "status_1_warning_bearing_run_count": 0,
        "return_code_counts": {"0": 0, "1": 0},
        "unresolved_summary_count_histogram": {"86": 0, "87": 0, "88": 0},
        "status_1_wrong_row_count": 0,
        "scale_search_status_1_count": 0,
        "unknown_warning_line_count": 0,
        "error_line_count": 0,
        "cache_mutation_warning_count": 0,
        "admission_pass": True,
    }
    for cache_id in sorted(expected_ids):
        observation = observed[cache_id]
        result = runner.run_or_load(observation.spec)
        raw_path = runner.raw_path(cache_id)
        sidecar_path = runner.sidecar_path(cache_id)
        raw = raw_path.read_bytes()
        diagnostics = strict_warning_diagnostics(result, raw)
        if cache_id in trace_runs:
            run_class = "confirmation_scale_search_anchor"
            meta = trace_runs[cache_id]
        else:
            run_class = "confirmation_integer_elevation_full_grid"
            full_meta = full_run_meta[cache_id]
            meta = {
                "coordinate_id": full_meta["coordinate_id"],
                "truth_profile": full_meta["truth_profile"],
                "scale_trace_path_relative_to_cache": full_meta["trace_path"],
                "scale_trace_sha256": full_meta["trace_sha256"],
                "scale_trace_evaluation_index": "",
                "scale_trace_role": "fitted_scale_full_grid_truth",
            }
        return_code = result.return_code
        warning["return_code_counts"][str(return_code)] += 1
        if return_code == 0:
            warning["status_0_run_count"] += 1
        else:
            warning["status_1_warning_bearing_run_count"] += 1
            warning["unresolved_summary_count_histogram"][
                str(result.parsed.warning_count)
            ] += 1
            if result.spec.expected_rows != 50001:
                warning["status_1_wrong_row_count"] += 1
            if run_class == "confirmation_scale_search_anchor":
                warning["scale_search_status_1_count"] += 1
        warning["unknown_warning_line_count"] += result.parsed.other_warning_line_count
        warning["error_line_count"] += result.parsed.error_line_count
        warning["cache_mutation_warning_count"] += int(
            diagnostics["cache_mutation_warning_count"] or 0
        )
        sidecar = result.sidecar
        rows.append(
            {
                "run_class": run_class,
                "coordinate_id": meta["coordinate_id"],
                "truth_profile": meta["truth_profile"],
                "cache_id": cache_id,
                "stage": result.spec.stage,
                "scale_decimal": result.spec.scale_decimal,
                "elevation_deg": str(result.spec.elevation_deg),
                "zenith_angle_deg": str(result.spec.zenith_angle_deg),
                "frequency_min_centi_ghz": str(result.spec.f_min_centi_ghz),
                "frequency_max_centi_ghz": str(result.spec.f_max_centi_ghz),
                "argv_json": json.dumps(sidecar["argv"], sort_keys=True, separators=(",", ":")),
                "working_directory_role": sidecar["working_directory_role"],
                "profile_sha256": sidecar["profile_sha256"],
                "am_executable_sha256": sidecar["am_executable_sha256"],
                "omp_threads": str(sidecar["omp_threads"]),
                "locale_json": json.dumps(sidecar["locale"], sort_keys=True, separators=(",", ":")),
                "execution_host_json": json.dumps(
                    sidecar["execution_host"], sort_keys=True, separators=(",", ":")
                ),
                "execution_context_sha256": sidecar["execution_context_sha256"],
                "am_cache_shard_index": str(sidecar["am_cache_shard_index"]),
                "am_cache_shard_count": str(sidecar["am_cache_shard_count"]),
                "raw_path_relative_to_cache": sidecar[
                    "combined_output_path_relative_to_cache"
                ],
                "raw_sha256": result.raw_sha256,
                "sidecar_path_relative_to_cache": sidecar_path.relative_to(
                    runner.cache_dir
                ).as_posix(),
                "sidecar_sha256": sha256_path(sidecar_path),
                "return_code": str(return_code),
                "am_version_identity": result.parsed.version_identity,
                "numeric_row_count": str(result.parsed.samples.shape[0]),
                "unresolved_line_warning_count": (
                    "" if result.parsed.warning_count is None else str(result.parsed.warning_count)
                ),
                "unresolved_column_warning_line_count": str(
                    result.parsed.unresolved_column_warning_line_count
                ),
                "unresolved_summary_warning_line_count": str(
                    result.parsed.unresolved_summary_warning_line_count
                ),
                "other_warning_line_count": str(result.parsed.other_warning_line_count),
                "error_line_count": str(result.parsed.error_line_count),
                "scale_trace_path_relative_to_cache": meta[
                    "scale_trace_path_relative_to_cache"
                ],
                "scale_trace_sha256": meta["scale_trace_sha256"],
                "scale_trace_evaluation_index": meta[
                    "scale_trace_evaluation_index"
                ],
                "scale_trace_role": meta["scale_trace_role"],
            }
        )
    warning["admission_pass"] = all(
        (
            warning["status_1_wrong_row_count"] == 0,
            warning["scale_search_status_1_count"] == 0,
            warning["unknown_warning_line_count"] == 0,
            warning["error_line_count"] == 0,
            warning["cache_mutation_warning_count"] == 0,
        )
    )
    require(warning["admission_pass"], "warning evidence admission failed")
    return rows, warning


def location_key(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        row["coordinate_id"],
        row["truth_profile"],
        int(row["elevation_deg"]),
        row["array"],
        int(row["alpha"]),
    )


def worst_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(abs(float(row["_error"])) for row in rows)
    tied = [row for row in rows if abs(float(row["_error"])) == maximum]
    return min(tied, key=location_key)


def build_metrics(comparison_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in comparison_rows:
        grouped.setdefault(
            (row["candidate_id"], row["passband_id"], int(row["alpha"])), []
        ).append(row)
    require(len(grouped) == 48, "metric group coverage mismatch")
    metrics: list[dict[str, Any]] = []
    for (candidate_id, passband_id, alpha), rows in sorted(grouped.items()):
        require(len(rows) == 896, f"metric sample coverage mismatch: {candidate_id}/{passband_id}/{alpha}")
        values = np.asarray([float(row["_error"]) for row in rows], dtype=np.float64)
        worst = worst_row(rows)
        maximum = float(np.max(np.abs(values)))
        require(abs(float(worst["_error"])) == maximum, "worst-location selection mismatch")
        first = rows[0]
        metrics.append(
            {
                "candidate_id": candidate_id,
                "candidate_role": first["candidate_role"],
                "lane": first["lane"],
                "operator": first["operator"],
                "passband_id": passband_id,
                "array": first["array"],
                "alpha": str(alpha),
                "n": str(values.size),
                "signed_min_fractional_correction_error": f17(float(np.min(values))),
                "signed_max_fractional_correction_error": f17(float(np.max(values))),
                "signed_bias_fractional_correction_error": f17(float(np.mean(values))),
                "rms_fractional_correction_error": f17(
                    float(np.sqrt(np.mean(values**2)))
                ),
                "p95_absolute_fractional_correction_error": f17(
                    float(np.quantile(np.abs(values), 0.95, method="linear"))
                ),
                "median_absolute_fractional_correction_error": f17(
                    float(np.median(np.abs(values)))
                ),
                "max_absolute_fractional_correction_error": f17(maximum),
                "gate_threshold": f17(FIDELITY_GATE),
                "gate_pass": bool_text(maximum <= FIDELITY_GATE),
                "worst_coordinate_id": worst["coordinate_id"],
                "worst_truth_profile": worst["truth_profile"],
                "worst_requested_tau225": worst["requested_tau225"],
                "worst_achieved_tau225": worst["achieved_tau225"],
                "worst_elevation_deg": worst["elevation_deg"],
                "worst_signed_fractional_correction_error": f17(
                    float(worst["_error"])
                ),
                "_maximum": maximum,
                "_worst": worst,
            }
        )
    return metrics


def coverage_result(
    *,
    cases: Sequence[ScaleCase],
    scale_rows: Sequence[dict[str, Any]],
    full_run_meta: dict[str, dict[str, Any]],
    direct_truth_keys: set[tuple[str, str, int, str, int]],
    comparison_rows: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
    physical_rows: Sequence[dict[str, Any]],
    independence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_scale = {
        (case.coordinate.coordinate_id, case.profile) for case in cases
    }
    actual_scale_list = [
        (row["coordinate_id"], row["truth_profile"]) for row in scale_rows
    ]
    expected_full = {
        (case.coordinate.coordinate_id, case.profile, elevation)
        for case in cases
        for elevation in ELEVATIONS_DEG
    }
    actual_full_list = [
        (item["coordinate_id"], item["truth_profile"], int(item["elevation_deg"]))
        for item in full_run_meta.values()
    ]
    expected_truth = {
        (
            case.coordinate.coordinate_id,
            case.profile,
            elevation,
            array,
            alpha,
        )
        for case in cases
        for elevation in ELEVATIONS_DEG
        for array in ARRAYS
        for alpha in ALPHAS
    }
    expected_rows = {
        (
            candidate["candidate_id"],
            array,
            alpha,
            case.coordinate.coordinate_id,
            case.profile,
            elevation,
        )
        for candidate in CANDIDATES
        for array in ARRAYS
        for alpha in ALPHAS
        for case in cases
        for elevation in ELEVATIONS_DEG
    }
    actual_row_list = [
        (
            row["candidate_id"],
            row["array"],
            int(row["alpha"]),
            row["coordinate_id"],
            row["truth_profile"],
            int(row["elevation_deg"]),
        )
        for row in comparison_rows
    ]
    expected_metric = {
        (candidate["candidate_id"], array, alpha)
        for candidate in CANDIDATES
        for array in ARRAYS
        for alpha in ALPHAS
    }
    actual_metric_list = [
        (row["candidate_id"], row["array"], int(row["alpha"])) for row in metrics
    ]
    actual_physical_list = [
        (row["candidate_id"], row["array"], int(row["alpha"]))
        for row in physical_rows
    ]
    checks = (
        ("scale_cases", expected_scale, actual_scale_list),
        ("full_grids", expected_full, actual_full_list),
        ("direct_truth", expected_truth, list(direct_truth_keys)),
        ("comparison_rows", expected_rows, actual_row_list),
        ("metrics", expected_metric, actual_metric_list),
        ("physical_metrics", expected_metric, actual_physical_list),
    )
    sections: dict[str, Any] = {}
    missing_count = unexpected_count = duplicate_count = 0
    for name, expected, actual_list in checks:
        actual = set(actual_list)
        missing = expected - actual
        unexpected = actual - expected
        duplicates = len(actual_list) - len(actual)
        missing_count += len(missing)
        unexpected_count += len(unexpected)
        duplicate_count += duplicates
        sections[name] = {
            "expected_count": len(expected),
            "actual_row_count": len(actual_list),
            "actual_unique_count": len(actual),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "duplicate_count": duplicates,
            "pass": not missing and not unexpected and duplicates == 0,
        }
    decision_coverage = {
        "opacity_coordinate_count": 6,
        "profile_scale_case_count": len(scale_rows),
        "elevation_count": len(ELEVATIONS_DEG),
        "full_direct_am_grid_count": len(full_run_meta),
        "direct_band_alpha_truth_count": len(direct_truth_keys),
        "candidate_count": len(CANDIDATES),
        "array_count": len(ARRAYS),
        "alpha_count": len(ALPHAS),
        "expanded_candidate_row_count": len(comparison_rows),
        "requested_predecessor_overlap_count": independence[
            "requested_predecessor_overlap_count"
        ],
        "achieved_predecessor_overlap_count": independence[
            "achieved_predecessor_overlap_count"
        ],
        "missing_key_count": missing_count,
        "unexpected_key_count": unexpected_count,
        "duplicate_key_count": duplicate_count,
        "pass": all(item["pass"] for item in sections.values()),
    }
    return decision_coverage, {
        "schema_version": f"{SCHEMA_VERSION}-coverage-v1",
        "decision_coverage": decision_coverage,
        "sections": sections,
        "independence": independence,
    }


def maximum_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_fractional_correction_error": f17(abs(float(row["_error"]))),
        "signed_fractional_correction_error": f17(float(row["_error"])),
        "coordinate_id": row["coordinate_id"],
        "truth_profile": row["truth_profile"],
        "requested_tau225": row["requested_tau225"],
        "achieved_tau225": row["achieved_tau225"],
        "elevation_deg": int(row["elevation_deg"]),
        "array": row["array"],
        "alpha": int(row["alpha"]),
    }


def build_decision(
    *,
    context: dict[str, Any],
    context_sha256: str,
    comparison_rows: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
    physical_rows: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    warning: dict[str, Any],
    independence: dict[str, Any],
    result_schema: dict[str, Any],
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        selected_rows = [row for row in comparison_rows if row["candidate_id"] == candidate_id]
        selected_metrics = [row for row in metrics if row["candidate_id"] == candidate_id]
        selected_physical = [row for row in physical_rows if row["candidate_id"] == candidate_id]
        require(
            len(selected_rows) == 10752
            and len(selected_metrics) == 12
            and len(selected_physical) == 12,
            f"candidate decision coverage mismatch: {candidate_id}",
        )
        maximum_row = worst_row(selected_rows)
        gates = {
            "G0": bool(warning["admission_pass"]),
            "G1": bool(independence["pass"]),
            "G2": True,
            "G3": all(row["physical_contract_pass"] == "true" for row in selected_physical),
            "G4": all(row["fail_closed_pass"] == "true" for row in selected_physical),
            "G5": all(row["gate_pass"] == "true" for row in selected_metrics),
            "G6": bool(coverage["pass"]),
        }
        gates["all_pass"] = all(gates.values())
        candidate_results.append(
            {
                "candidate": dict(candidate),
                "gate_results": gates,
                "maximum": maximum_payload(maximum_row),
                "can_rescue_primary_failure": False,
            }
        )

    primary = candidate_results[0]
    primary_gates = primary["gate_results"]
    validity = all(primary_gates[name] for name in ("G0", "G1", "G2", "G4", "G6"))
    if not validity:
        status = "confirmation_invalid"
        software_status = "invalid"
        numerical_status = "invalid"
        primary_maximum = None
    elif primary_gates["all_pass"]:
        status = "primary_confirmation_gate_pass"
        software_status = (
            "pass_warning_bearing_evidence"
            if warning["status_1_warning_bearing_run_count"]
            else "pass_clean"
        )
        numerical_status = "primary_pass"
        primary_maximum = primary["maximum"]
    else:
        status = "primary_confirmation_gate_fail"
        software_status = (
            "pass_warning_bearing_evidence"
            if warning["status_1_warning_bearing_run_count"]
            else "pass_clean"
        )
        numerical_status = "primary_fail"
        primary_maximum = primary["maximum"]

    git_binding = context["preregistration_git_binding"]
    controls = context["controls"]
    decision = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "package_id": PACKAGE_ID,
        "study_id": STUDY_ID,
        "status": status,
        "authorization": (
            "numerical_confirmation_evidence_only_no_adoption_or_operational_authorization"
        ),
        "software_status": software_status,
        "numerical_representation_status": numerical_status,
        "observational_status": "not_evaluated_required_before_production",
        "operator_adoption": None,
        "operational_domain_authorization": None,
        "provenance": {
            "preregistration_commit": git_binding["preregistration_commit"],
            "preregistration_parent": git_binding["preregistration_parent"],
            "protocol_sha256": controls["protocol"]["sha256"],
            "preregistration_sha256": controls["preregistration"]["sha256"],
            "result_schema_sha256": controls["result_schema"]["sha256"],
            "runner_sha256": controls["runner"]["sha256"],
            "execution_context_sha256": context_sha256,
            "application_repair_base": REPAIR_BASE,
            "coordination_binding_commit": COORDINATION_BINDING_COMMIT,
            "immutable_decision_commit": IMMUTABLE_DECISION_COMMIT,
            "am_executable_sha256": EXPECTED_AM_EXECUTABLE_SHA256,
            "am_source_payload_sha256": EXPECTED_AM_SOURCE_SHA256,
            "candidate_nodes_sha256": EXPECTED_NODE_SHA256,
            "passband_set_id": PASSBAND_SET_ID,
        },
        "primary_candidate": dict(CANDIDATES[0]),
        "primary_gate_results": primary_gates,
        "primary_maximum": primary_maximum,
        "secondary_candidates": candidate_results[1:],
        "coverage": coverage,
        "warning_evidence": warning,
        "limitations": [
            "The one-percent threshold is numerical representation fidelity, not physical photometric accuracy.",
            "Absolute-flux accuracy and observation-to-observation repeatability were not evaluated.",
            "q95 conditions are excluded and no q95 operator is authorized.",
            "SCI-ALIGN-001 remains an open application-interface dependency.",
            "This result cannot adopt an operator or authorize an operational domain.",
        ],
    }
    Draft202012Validator(result_schema).validate(decision)
    return decision


def build_report(
    decision: dict[str, Any],
    run_inventory_count: int,
    scale_search_count: int,
) -> bytes:
    maximum = decision["primary_maximum"]
    if maximum is None:
        maximum_line = "Primary maximum: unavailable because the confirmation is invalid."
    else:
        maximum_line = (
            "Primary maximum absolute fractional correction error: "
            f"{maximum['absolute_fractional_correction_error']} at "
            f"{maximum['coordinate_id']}/{maximum['truth_profile']}/"
            f"EL{maximum['elevation_deg']}/{maximum['array']}/alpha={maximum['alpha']}."
        )
    lines = [
        "# SCI-CAL-001 AM 12.2 EL25 numerical confirmation",
        "",
        f"Status: **{decision['status']}**.",
        "",
        maximum_line,
        "",
        "## Fixed question and evidence",
        "",
        "- Primary: fixed_djf25_v1 plus am12_piecewise_linear_los_tau_eval_v0.",
        "- Support evaluated: closed q0--q75 and aligned EL25--80; q95 excluded.",
        "- Independent truth: 16 scale cases and 896 full AM grids.",
        f"- Digest-bound run inventory: {run_inventory_count} runs, including {scale_search_count} scale-search anchors.",
        "- TolTECA v1 ECSV passbands only, with alpha -1, 0, 2, and 4.",
        "- Secondary candidates are descriptive and cannot rescue a primary failure.",
        "",
        "## Gate separation",
        "",
        f"- Software status: {decision['software_status']}.",
        f"- Numerical representation status: {decision['numerical_representation_status']}.",
        f"- Observational status: {decision['observational_status']}.",
        "- The inclusive one-percent criterion is provisional numerical representation fidelity only.",
        "- The observational 5--10% absolute-flux and approximately 5% repeatability objectives remain unevaluated.",
        "",
        "## Authorization boundary",
        "",
        "This result does not adopt an operator, authorize an operational domain, modify Citlali, resolve SCI-ALIGN-001, or launch CAL repair or re-audit.",
        "",
        "## Machine result",
        "",
        "```json",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_manifest(
    *,
    artifacts: dict[str, bytes],
    decision: dict[str, Any],
    context: dict[str, Any],
    context_sha256: str,
    external_cache_inventory: dict[str, Any],
    coverage_artifact: dict[str, Any],
    cache_dir: Path,
) -> bytes:
    require(len(artifacts) == 9, "manifest must bind exactly nine predecessor artifacts")
    payload = {
        "schema_version": f"{SCHEMA_VERSION}-manifest-v1",
        "package_id": PACKAGE_ID,
        "study_id": STUDY_ID,
        "status": decision["status"],
        "authorization": (
            "numerical_confirmation_evidence_only_no_adoption_or_operational_authorization"
        ),
        "preregistration_git_binding": context["preregistration_git_binding"],
        "controls": context["controls"],
        "execution_context_sha256": context_sha256,
        "external_cache": {
            "basename": cache_dir.name,
            "cache_policy": (
                "fresh exclusive execution; shared-lock cache-only replay"
            ),
            "evidence_inventory": external_cache_inventory,
        },
        "provenance": decision["provenance"],
        "candidate_roles": [dict(item) for item in CANDIDATES],
        "domain": {
            "tau225_min_inclusive": f17(TAU_MIN),
            "tau225_max_inclusive_q75": f17(TAU_MAX),
            "elevation_min_deg_inclusive": f17(ELEVATION_MIN_DEG),
            "elevation_max_deg_inclusive": f17(ELEVATION_MAX_DEG),
            "q95_included": False,
            "outside_domain_policy": "fail_closed",
        },
        "coverage": coverage_artifact,
        "decision": decision,
        "artifacts": {
            name: {"size_bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(artifacts.items())
        },
        "security": context["security"],
        "limitations": decision["limitations"],
    }
    return json_bytes(payload)


def build_outputs(
    *,
    runner: StrictRunner,
    p1: adoption_driver.P1Cache,
    controls: Controls,
    context: dict[str, Any],
    context_bytes: bytes,
    cases: Sequence[ScaleCase],
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    tau_nodes: np.ndarray,
    bandpasses: Sequence[adoption_driver.Bandpass],
    physical_rows: Sequence[dict[str, Any]],
    independence: dict[str, Any],
    cache_dir: Path,
) -> dict[str, bytes]:
    (
        scale_rows,
        comparison_rows,
        solutions,
        full_run_meta,
        direct_truth_keys,
    ) = solve_and_evaluate(
        runner=runner,
        p1=p1,
        cases=cases,
        node_arrays=node_arrays,
        tau_nodes=tau_nodes,
        bandpasses=bandpasses,
    )
    run_rows, warning = build_run_inventory(runner, solutions, full_run_meta)
    metrics = build_metrics(comparison_rows)
    decision_coverage, coverage_artifact = coverage_result(
        cases=cases,
        scale_rows=scale_rows,
        full_run_meta=full_run_meta,
        direct_truth_keys=direct_truth_keys,
        comparison_rows=comparison_rows,
        metrics=metrics,
        physical_rows=physical_rows,
        independence=independence,
    )
    context_sha = sha256_bytes(context_bytes)
    decision = build_decision(
        context=context,
        context_sha256=context_sha,
        comparison_rows=comparison_rows,
        metrics=metrics,
        physical_rows=physical_rows,
        coverage=decision_coverage,
        warning=warning,
        independence=independence,
        result_schema=controls.result_schema,
    )
    scale_search_count = sum(
        row["run_class"] == "confirmation_scale_search_anchor" for row in run_rows
    )
    report = build_report(decision, len(run_rows), scale_search_count)

    schemas = controls.preregistration["output_table_schemas"]
    scale_rows = sorted(
        scale_rows, key=lambda row: (row["coordinate_id"], row["truth_profile"])
    )
    run_rows = sorted(
        run_rows,
        key=lambda row: (
            row["run_class"],
            row["coordinate_id"],
            row["truth_profile"],
            int(row["elevation_deg"]),
            row["cache_id"],
        ),
    )
    comparison_rows = sorted(
        comparison_rows,
        key=lambda row: (
            row["candidate_id"],
            row["array"],
            int(row["alpha"]),
            row["coordinate_id"],
            row["truth_profile"],
            int(row["elevation_deg"]),
        ),
    )
    metrics = sorted(
        metrics,
        key=lambda row: (row["candidate_id"], row["array"], int(row["alpha"])),
    )
    physical_rows = sorted(
        physical_rows,
        key=lambda row: (row["candidate_id"], row["array"], int(row["alpha"])),
    )
    artifacts = {
        OUTPUT_NAMES[0]: context_bytes,
        OUTPUT_NAMES[1]: csv_bytes(
            schemas[OUTPUT_NAMES[1]]["fieldnames"], scale_rows
        ),
        OUTPUT_NAMES[2]: csv_bytes(
            schemas[OUTPUT_NAMES[2]]["fieldnames"], run_rows
        ),
        OUTPUT_NAMES[3]: csv_bytes(
            schemas[OUTPUT_NAMES[3]]["fieldnames"], comparison_rows
        ),
        OUTPUT_NAMES[4]: csv_bytes(
            schemas[OUTPUT_NAMES[4]]["fieldnames"], metrics
        ),
        OUTPUT_NAMES[5]: csv_bytes(
            schemas[OUTPUT_NAMES[5]]["fieldnames"], physical_rows
        ),
        OUTPUT_NAMES[6]: json_bytes(coverage_artifact),
        OUTPUT_NAMES[7]: json_bytes(decision),
        OUTPUT_NAMES[8]: report,
    }
    external_inventory = cache_evidence_inventory(cache_dir)
    require(
        external_inventory["failed_attempt_file_count"] == 0,
        "failed attempt entered final evidence inventory",
    )
    artifacts[OUTPUT_NAMES[9]] = build_manifest(
        artifacts=artifacts,
        decision=decision,
        context=context,
        context_sha256=context_sha,
        external_cache_inventory=external_inventory,
        coverage_artifact=coverage_artifact,
        cache_dir=cache_dir,
    )
    require(set(artifacts) == set(OUTPUT_NAMES), "deterministic artifact set mismatch")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run-confirmation",
        action="store_true",
        help=(
            "execute the preregistered confirmation into the exact absent fresh cache"
        ),
    )
    mode.add_argument(
        "--regenerate-from-cache",
        action="store_true",
        help="cache-only deterministic artifact regeneration; never launch AM",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="cache-only byte-for-byte artifact check; write nothing and never launch AM",
    )
    parser.add_argument("--confirmation-cache-dir", type=Path, required=True)
    parser.add_argument("--p1-cache-dir", type=Path, default=DEFAULT_P1_CACHE)
    parser.add_argument("--am-root", type=Path, default=DEFAULT_AM_ROOT)
    parser.add_argument("--am-executable", type=Path, default=DEFAULT_AM_EXECUTABLE)
    parser.add_argument("--tolteca-repo", type=Path, default=DEFAULT_TOLTECA_REPO)
    parser.add_argument(
        "--coordination-repo", type=Path, default=DEFAULT_COORDINATION_REPO
    )
    parser.add_argument("--protocol", type=Path, default=PACKAGE_DIR / PROTOCOL_NAME)
    parser.add_argument(
        "--preregistration", type=Path, default=PACKAGE_DIR / PREREGISTRATION_NAME
    )
    parser.add_argument(
        "--result-schema", type=Path, default=PACKAGE_DIR / RESULT_SCHEMA_NAME
    )
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.confirmation_cache_dir.expanduser().resolve()
    p1_cache_dir = args.p1_cache_dir.expanduser().resolve()
    am_root = args.am_root.expanduser().resolve()
    executable = args.am_executable.expanduser().resolve()
    tolteca_repo = args.tolteca_repo.expanduser().resolve()
    coordination_repo = args.coordination_repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    result_schema = args.result_schema.expanduser().resolve()

    require(cache_dir.name == CACHE_BASENAME, f"cache basename must be exactly {CACHE_BASENAME}")
    require(output_dir == PACKAGE_DIR, f"output directory must be task package: {PACKAGE_DIR}")
    protected = {
        "repository": REPO_ROOT,
        "task package": PACKAGE_DIR,
        "canonical P1 cache": p1_cache_dir,
        "copied AM root": am_root,
        "AM executable build": executable.parent,
        "TolTECA input repository": tolteca_repo,
        "coordination repository": coordination_repo,
    }
    for label, path in protected.items():
        require(not paths_overlap(cache_dir, path), f"confirmation cache overlaps {label}: {path}")
    if args.run_confirmation:
        require(not cache_dir.exists(), "--run-confirmation cache path already exists")
    else:
        require(cache_dir.is_dir(), f"missing confirmation cache: {cache_dir}")

    controls = load_controls(
        protocol,
        preregistration,
        result_schema,
        execution_mode=args.run_confirmation,
    )
    coordinates = coordinate_plan(controls.preregistration)
    cases = scale_cases(controls.preregistration, coordinates)
    p1 = adoption_driver.P1Cache(p1_cache_dir, am_root)
    with p1.shared_lock():
        predecessors = validate_predecessors(controls.preregistration, p1)
        am_inputs = validate_am_inputs(
            p1, am_root, executable, controls.preregistration
        )
    authority = validate_authority(controls.preregistration, coordination_repo)
    independence = independence_check(coordinates)
    bandpasses = load_primary_bandpasses(tolteca_repo, controls.preregistration)
    node_arrays, tau_nodes = load_candidate_nodes(bandpasses)
    physical_rows = physical_metrics(
        node_arrays, tau_nodes, bandpasses, coordinates
    )

    if args.run_confirmation:
        git_binding = run_git_binding(controls)
        context = build_execution_context(
            controls=controls,
            git_binding=git_binding,
            authority=authority,
            predecessors=predecessors,
            independence=independence,
            am_inputs=am_inputs,
            p1=p1,
            bandpasses=bandpasses,
            coordinates=coordinates,
            cases=cases,
            cache_dir=cache_dir,
        )
        context_bytes = json_bytes(context)
        prepare_fresh_cache_root(cache_dir)
    else:
        cached_context = load_cached_context(cache_dir)
        git_binding = replay_git_binding(cached_context)
        context = build_execution_context(
            controls=controls,
            git_binding=git_binding,
            authority=authority,
            predecessors=predecessors,
            independence=independence,
            am_inputs=am_inputs,
            p1=p1,
            bandpasses=bandpasses,
            coordinates=coordinates,
            cases=cases,
            cache_dir=cache_dir,
        )
        context["execution_host"] = cached_context["execution_host"]
        context_bytes = json_bytes(context)
        require(
            context_bytes == (cache_dir / "execution_context.json").read_bytes(),
            "confirmation cache execution context mismatch",
        )

    build = p1_driver.build_identity(executable)
    context_sha = sha256_bytes(context_bytes)
    before_inventory = None if args.run_confirmation else cache_evidence_inventory(cache_dir)
    with confirmation_cache_lock(cache_dir, exclusive=args.run_confirmation):
        if args.run_confirmation:
            initialize_fresh_cache(cache_dir, context_bytes)
        runner = StrictRunner(
            executable=build,
            am_root=am_root,
            cache_dir=cache_dir,
            omp_threads=OMP_THREADS,
            cache_shard_count=CACHE_SHARDS,
            execution_host=context["execution_host"],
            execution_context_sha256=context_sha,
            execute=args.run_confirmation,
        )
        artifacts = build_outputs(
            runner=runner,
            p1=p1,
            controls=controls,
            context=context,
            context_bytes=context_bytes,
            cases=cases,
            node_arrays=node_arrays,
            tau_nodes=tau_nodes,
            bandpasses=bandpasses,
            physical_rows=physical_rows,
            independence=independence,
            cache_dir=cache_dir,
        )
        after_inventory = cache_evidence_inventory(cache_dir)
    if before_inventory is not None:
        require(
            before_inventory == after_inventory,
            "cache-only replay changed the evidence-file aggregate",
        )

    if args.check:
        for name, expected in sorted(artifacts.items()):
            path = output_dir / name
            require(path.is_file(), f"missing checked confirmation artifact: {path}")
            require(
                path.read_bytes() == expected,
                f"confirmation artifact differs from cache-only replay: {path}",
            )
        print(
            f"Validated {len(artifacts)} deterministic confirmation artifacts "
            "cache-only; no AM process executed."
        )
        return 0

    for name, data in sorted(artifacts.items()):
        atomic_write(output_dir / name, data)
    action = "new direct AM confirmation" if args.run_confirmation else "cache-only"
    print(f"Wrote {len(artifacts)} deterministic artifacts from {action} evidence.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
