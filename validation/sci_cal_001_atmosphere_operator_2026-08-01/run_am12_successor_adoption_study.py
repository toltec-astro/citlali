#!/usr/bin/env python3
"""Evaluate the bounded AM-12.2 successor atmosphere operators.

This is an evidence driver, not Citlali application code.  It consumes the
canonical P1 direct-AM cache, integrates the spectra through frozen TolTECA
passbands and representative FTS challengers, and compares two profile lanes
and two continuous operators over q0--q75.  AM execution is impossible unless
``--run-holdouts`` is supplied explicitly.  Cache-only replay is the normal
artifact-generation path.

Run with the project venv, for example::

    $HOME/tolteca/bin/python run_am12_successor_adoption_study.py \
        --regenerate-from-cache --p1-cache-dir /private/tmp/<canonical-p1> \
        --holdout-cache-dir /private/tmp/<adoption-holdout-cache>

The provisional one-percent criterion used here is numerical representation
fidelity.  It is not a statement of per-sample or absolute photometric
accuracy.
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
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

import probe_am12_h2o_scale_hypotheses as p1_driver


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

SCHEMA_VERSION = "sci-cal-001-am12-successor-adoption-study-v2"
P1_SCHEMA_VERSION = "sci-cal-001-am12-h2o-scale-hypothesis-v2"
TOLTECA_COMMIT = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
BEAMMAP_COMMIT = "958a2a15f43189846a24556a63ef908da789c7b8"
TARGETS = ("am_q25", "am_q50", "am_q75")
TARGET_ORDER = {"am_q0": 0, "am_q25": 1, "am_q50": 2, "am_q75": 3}
LEGACY_ANCHOR_MANIFEST_SHA256 = (
    "e481a0053e4262d7db884e34f184463c2a17a546996ad9ba1d5065c6eb97bb74"
)
STUDY_PROTOCOL_SHA256 = (
    "bbf73c25e2b6d3c3d4315ae6b18e39d327abb51c435683b057039c505c2cfc96"
)
PREEXECUTION_CLARIFICATION_SHA256 = (
    "01957dab95e37a4b87e6224c713ec96ec645b37ae39c5dd95b6b49af493b9a66"
)
EXECUTION_ERRATUM_SHA256 = (
    "590f49007065e604aced97fc391067e981a94d7336db1cec81512dd0de893e4e"
)
FROZEN_TARGET_COORDINATES = {
    "am_q0": ("0.00000000000000000e+00", "1.0"),
    "am_q25": ("5.04874104674104401e-02", "0.9500275"),
    "am_q50": ("8.83393725904400573e-02", "0.9142065"),
    "am_q75": ("1.58313198574890929e-01", "0.8515054"),
}
SEASONS = ("annual", "DJF", "MAM", "JJA", "SON")
PROFILE_PERCENTILES = (5, 25, 50, 75)
PROFILES = tuple(
    f"LMT_{season}_{percentile}"
    for season in SEASONS
    for percentile in PROFILE_PERCENTILES
)
ELEVATIONS_EVEN_DEG = np.arange(20, 82, 2, dtype=np.float64)
ELEVATIONS_ODD_DEG = np.arange(21, 80, 2, dtype=np.float64)
ALPHAS = (-1, 0, 2, 4)
FIDELITY_GATE = 0.01
PHYSICAL_TOLERANCE = 1.0e-12
ANCHOR_TOLERANCE = 1.0e-12
PINNED_LOCALE = {"LANG": "C", "LC_ALL": "C"}
CANONICAL_RUN_SIDECAR_KEYS = {
    "schema_version",
    "cache_id",
    "request",
    "argv",
    "working_directory_role",
    "profile_path_relative_to_working_directory",
    "profile_sha256",
    "am_executable_sha256",
    "omp_threads",
    "locale",
    "execution_host",
    "execution_context_sha256",
    "am_cache_shard_index",
    "am_cache_shard_count",
    "am_cache_path_relative_to_cache",
    "combined_output_path_relative_to_cache",
    "combined_output_sha256",
    "numeric_text_sha256",
    "normalized_output_sha256",
    "numeric_row_count",
    "unresolved_line_warning_count",
    "unresolved_column_warning_line_count",
    "unresolved_summary_warning_line_count",
    "other_warning_line_count",
    "error_line_count",
    "am_version_identity",
    "return_code",
}

DEFAULT_P1_CACHE = Path(
    "/private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root"
)
DEFAULT_AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
DEFAULT_AM_EXECUTABLE = Path(
    "/private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am"
)
DEFAULT_TOLTECA_REPO = Path("/Users/gwilson/GitHub/tolteca")
DEFAULT_BEAMMAP_REPO = Path("/Users/gwilson/GitHub/toltec_beammap")
EXPECTED_AM_EXECUTABLE_SHA256 = (
    "78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb"
)
HOLDOUT_PROFILE_SHA256 = {
    "LMT_DJF_5": "fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474",
    "LMT_DJF_25": "aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866",
    "LMT_DJF_50": "d7c256d04d922beb51c9f8ab715e5be1a962252580eff2d08ba1be4d206eb5b0",
    "LMT_DJF_75": "b63503c7f4170404d18f3797735b64fb947ce73eed35f0315155d0a29d499721",
    "LMT_annual_25": "a9524553a5808a549eb18046a9ed6f8bd67ca1e29ccd1c91e05b351b64ea23e6",
    "LMT_MAM_25": "82ac1e2a49a528244c1571daadcc8d42bd6d13c0ba8a7b5d2f81d10ebc13caee",
}
HOLDOUT_CASES = (
    ("q0_q25_midpoint", "LMT_DJF_5", "am_q0", "am_q25"),
    ("q0_q25_midpoint", "LMT_DJF_25", "am_q0", "am_q25"),
    ("q25_q50_midpoint", "LMT_DJF_25", "am_q25", "am_q50"),
    ("q25_q50_midpoint", "LMT_DJF_50", "am_q25", "am_q50"),
    ("q50_q75_midpoint", "LMT_DJF_50", "am_q50", "am_q75"),
    ("q50_q75_midpoint", "LMT_DJF_75", "am_q50", "am_q75"),
    ("q50_q75_midpoint", "LMT_annual_25", "am_q50", "am_q75"),
    ("q50_q75_midpoint", "LMT_MAM_25", "am_q50", "am_q75"),
)
C1_X80 = "1.01538872688246729"
C1_INTERVALS = {
    "q0_q25_midpoint": {
        "tau_mid": "2.524370523370522005e-2",
        "t_analytic": "9.74693541581147455535088318156280150e-1",
        "literal": "9.746935e-01",
        "tau_achieved": "2.52437472479008390850963986147116523e-2",
        "residual": "4.20141956190350963986147116523499926e-8",
        "lower_bound": "5.05207263490210769585533250450814916e-8",
        "upper_bound": "5.05207289406423222439544156868184469e-8",
    },
    "q25_q50_midpoint": {
        "tau_mid": "6.941339152892524870e-2",
        "t_analytic": "9.31944910216666213788824485393985953e-1",
        "literal": "9.319449e-01",
        "tau_achieved": "6.94134023255157373905797836863371040e-2",
        "residual": "1.07965904886905797836863371039666709e-8",
        "lower_bound": "5.28381275864426589515504053295119258e-8",
        "upper_bound": "5.28381304212738278252825721731728213e-8",
    },
    "q50_q75_midpoint": {
        "tau_mid": "1.2332628558266549315e-1",
        "t_analytic": "8.82299139444837037445844988932535555e-1",
        "literal": "8.822991e-01",
        "tau_achieved": "1.23326329611985962328974442523710106e-1",
        "residual": "4.40293204691789744425237101058733478e-8",
        "lower_bound": "5.58112588524756115790882535541648871e-8",
        "upper_bound": "5.58112620153066987858634307860799623e-8",
    },
}

PRIMARY_BLOBS = {
    array: (f"tolteca/data/cal/toltec_passband/data/{array}_passband.ecsv")
    for array in ("a1100", "a1400", "a2000")
}
PRIMARY_SHA256 = {
    "a1100": "13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72",
    "a1400": "a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e",
    "a2000": "77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff",
}
FTS_FILES = {
    "a1100": "data/beammap/spectra/FTS_5934_N0.npz",
    "a1400": "data/beammap/spectra/FTS_9932_N9.npz",
    "a2000": "data/beammap/spectra/FTS_9903_N11.npz",
}
FTS_SHA256 = {
    "a1100": "b72e9be7a4637adbfb5f2a6e131741a4a7b151effc03dea49410d0e56a5df74c",
    "a1400": "da440896f537545871ac0d026b5149aeb5ba356e2f613c934567ef20fde0fd36",
    "a2000": "b3d5a0b1a332d40b4cdc436cb4327e849afbce42cfd766108aacf6016e775e65",
}
PIVOT_GHZ = {"a1100": 272.73, "a1400": 214.29, "a2000": 150.0}

LANES = {
    "fixed_djf25_v1": {
        "am_q25": "LMT_DJF_25",
        "am_q50": "LMT_DJF_25",
        "am_q75": "LMT_DJF_25",
    },
    "conditioned_djf_v1": {
        "am_q25": "LMT_DJF_25",
        "am_q50": "LMT_DJF_50",
        "am_q75": "LMT_DJF_75",
    },
}
TRAINING_KEYS = tuple(
    sorted(
        {
            (target, profile)
            for profiles in LANES.values()
            for target, profile in profiles.items()
        },
        key=lambda item: (TARGET_ORDER[item[0]], item[1]),
    )
)
P1_SELECTED_STAGE_TRAINING_KEYS = {
    ("am_q50", "LMT_DJF_25"),
    ("am_q75", "LMT_DJF_75"),
}
OPERATORS = (
    "am12_piecewise_linear_los_tau_eval_v0",
    "am12_pchip_los_tau_eval_v0",
)

OUTPUT_NAMES = {
    "bandpasses": "am12_successor_bandpass_inventory.csv",
    "nodes": "am12_successor_operator_nodes.csv",
    "metrics": "am12_successor_operator_metrics.csv",
    "physical": "am12_successor_physical_metrics.csv",
    "p1_runs": "am12_successor_p1_run_inventory.csv",
    "holdout_runs": "am12_successor_holdout_run_inventory.csv",
    "execution_context": "am12_successor_holdout_execution_context.json",
    "holdouts": "am12_successor_holdout_rows.csv",
    "scales": "am12_successor_holdout_scales.csv",
    "coverage": "am12_successor_coverage.json",
    "report": "AM12_SUCCESSOR_ADOPTION_STUDY_REPORT.md",
    "decision": "am12_successor_decision.json",
    "manifest": "am12_successor_adoption_manifest.json",
}

FLOAT_TOKEN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?")
VERSION_LINE = re.compile(r"^# (am version[^\r\n]+)$", re.MULTILINE)
UNRESOLVED_WARNING = re.compile(
    r"^! Warning: Encountered in-band lines narrower than the frequency\r?\n"
    r"^!          grid spacing\.  The output configuration data includes\r?\n"
    r"^!          the unresolved line count after each column definition\r?\n"
    r"^!          for which this occurred\.  Count: (\d+)\s*$",
    re.MULTILINE,
)
WARNING_HEADER = re.compile(r"^! Warning: (.+)$", re.MULTILINE)
ERROR_LINE = re.compile(r"^! Error:", re.MULTILINE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def f17(value: float) -> str:
    return f"{float(value):.17e}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != json_bytes(payload):
        raise RuntimeError(f"noncanonical JSON: {path}")
    return payload


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return is_relative_to(left_resolved, right_resolved) or is_relative_to(
        right_resolved, left_resolved
    )


def modified_airmass(elevation_deg: np.ndarray | float) -> np.ndarray:
    elevation = np.asarray(elevation_deg, dtype=np.float64)
    secant = 1.0 / np.sin(np.deg2rad(elevation))
    return secant * (1.0 - 0.0012 * (secant * secant - 1.0))


def git_output(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


@dataclass(frozen=True)
class Bandpass:
    identity: str
    array: str
    family: str
    frequency_ghz: np.ndarray
    response: np.ndarray
    source_path: str
    source_sha256: str
    source_commit: str
    convention: str
    clipped_node_count: int = 0
    clipped_negative_integral_fraction: float = 0.0

    def weights(self, alpha: int) -> np.ndarray:
        frequency = self.frequency_ghz
        delta = np.diff(frequency)
        quadrature = np.empty_like(frequency)
        quadrature[0] = delta[0] / 2.0
        quadrature[-1] = delta[-1] / 2.0
        quadrature[1:-1] = (delta[:-1] + delta[1:]) / 2.0
        weight = (
            quadrature * self.response * (frequency / PIVOT_GHZ[self.array]) ** alpha
        )
        denominator = float(np.sum(weight))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(f"nonpositive passband normalization: {self.identity}")
        return weight / denominator


def parse_primary_ecsv(data: bytes, array: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    header_seen = False
    for line in data.decode("utf-8").splitlines():
        if line == "f wl throughput":
            header_seen = True
            continue
        if header_seen and line and not line.startswith("#"):
            tokens = line.split()
            if len(tokens) != 3:
                raise RuntimeError(f"malformed {array} ECSV data row")
            rows.append((float(tokens[0]), float(tokens[2])))
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise RuntimeError(f"empty or malformed primary passband for {array}")
    frequency, response = values.T
    if not (
        np.all(np.isfinite(values))
        and np.all(np.diff(frequency) > 0.0)
        and np.all(response >= 0.0)
        and np.any(response > 0.0)
    ):
        raise RuntimeError(f"invalid primary passband for {array}")
    return frequency, response


def load_bandpasses(tolteca_repo: Path, beammap_repo: Path) -> list[Bandpass]:
    resolved_tolteca = (
        git_output(tolteca_repo, "rev-parse", TOLTECA_COMMIT).decode().strip()
    )
    if resolved_tolteca != TOLTECA_COMMIT:
        raise RuntimeError("TolTECA frozen commit did not resolve exactly")
    resolved_beammap = (
        git_output(beammap_repo, "rev-parse", BEAMMAP_COMMIT).decode().strip()
    )
    if resolved_beammap != BEAMMAP_COMMIT:
        raise RuntimeError("toltec_beammap frozen commit did not resolve exactly")

    result: list[Bandpass] = []
    primary_support: dict[str, tuple[float, float]] = {}
    for array in ("a1100", "a1400", "a2000"):
        blob_path = PRIMARY_BLOBS[array]
        data = git_output(tolteca_repo, "show", f"{TOLTECA_COMMIT}:{blob_path}")
        if sha256_bytes(data) != PRIMARY_SHA256[array]:
            raise RuntimeError(f"TolTECA passband digest mismatch: {array}")
        frequency, response = parse_primary_ecsv(data, array)
        primary_support[array] = (float(frequency[0]), float(frequency[-1]))
        result.append(
            Bandpass(
                identity=f"tolteca_v1_{array}",
                array=array,
                family="primary_tolteca_ecsv",
                frequency_ghz=frequency,
                response=response,
                source_path=blob_path,
                source_sha256=PRIMARY_SHA256[array],
                source_commit=TOLTECA_COMMIT,
                convention="ECSV throughput used as supplied",
                clipped_node_count=0,
                clipped_negative_integral_fraction=0.0,
            )
        )

    for array in ("a1100", "a1400", "a2000"):
        relative = FTS_FILES[array]
        path = beammap_repo / relative
        if sha256_path(path) != FTS_SHA256[array]:
            raise RuntimeError(f"FTS challenger digest mismatch: {path}")
        tracked = git_output(beammap_repo, "show", f"{BEAMMAP_COMMIT}:{relative}")
        if sha256_bytes(tracked) != FTS_SHA256[array]:
            raise RuntimeError(f"FTS blob differs from frozen commit: {relative}")
        with np.load(io.BytesIO(tracked), allow_pickle=False) as archive:
            if set(archive.files) != {"fc", "sc", "shigherr", "slowerr"}:
                raise RuntimeError(f"unexpected FTS members: {relative}")
            frequency = np.asarray(archive["fc"], dtype=np.float64)
            response = np.asarray(archive["sc"], dtype=np.float64)
        lower, upper = primary_support[array]
        selected = (frequency >= lower) & (frequency <= upper)
        frequency = frequency[selected]
        signed_response = response[selected]
        negative = np.clip(-signed_response, 0.0, None)
        response = np.clip(signed_response, 0.0, None)
        positive_integral = float(np.trapezoid(response, frequency))
        negative_integral = float(np.trapezoid(negative, frequency))
        if not (
            frequency.size >= 2
            and np.all(np.isfinite(frequency))
            and np.all(np.isfinite(response))
            and np.all(np.diff(frequency) > 0.0)
            and np.any(response > 0.0)
        ):
            raise RuntimeError(f"invalid FTS challenger: {relative}")
        device = Path(relative).stem.lower()
        result.append(
            Bandpass(
                identity=f"{device}_{array}",
                array=array,
                family="representative_fts_challenger",
                frequency_ghz=frequency,
                response=response,
                source_path=relative,
                source_sha256=FTS_SHA256[array],
                source_commit=BEAMMAP_COMMIT,
                convention=(
                    "sc clipped at zero and restricted to corresponding frozen "
                    "TolTECA ECSV support; challenger only"
                ),
                clipped_node_count=int(np.count_nonzero(signed_response < 0.0)),
                clipped_negative_integral_fraction=(
                    negative_integral / positive_integral
                ),
            )
        )
    return sorted(result, key=lambda item: (item.array, item.family, item.identity))


@dataclass(frozen=True)
class ParsedAM:
    frequency_ghz: np.ndarray
    tau_los: np.ndarray
    transmission: np.ndarray
    numeric_sha256: str
    normalized_sha256: str
    version: str
    unresolved_lines: int | None


def normalize_am_output(data: bytes) -> bytes:
    normalized: list[str] = []
    for line in data.decode("utf-8").splitlines():
        if line.startswith("# run time "):
            normalized.append("# run time <volatile>")
        elif line.startswith("# dcache hit: "):
            normalized.append("# dcache counters <volatile>")
        else:
            normalized.append(line)
    return ("\n".join(normalized) + "\n").encode("utf-8")


def parse_am_output(data: bytes, label: str) -> ParsedAM:
    rows: list[list[float]] = []
    numeric = hashlib.sha256()
    for raw_line in data.splitlines(keepends=True):
        try:
            tokens = raw_line.decode("ascii").strip().split()
        except UnicodeDecodeError as error:
            raise RuntimeError(f"non-ASCII AM output: {label}") from error
        if len(tokens) == 5 and all(FLOAT_TOKEN.fullmatch(token) for token in tokens):
            rows.append([float(token) for token in tokens])
            numeric.update(raw_line)
    samples = np.asarray(rows, dtype=np.float64)
    if samples.shape != (50001, 5):
        raise RuntimeError(f"unexpected AM grid {samples.shape}: {label}")
    expected_frequency = np.arange(50001, dtype=np.float64) / 100.0
    if not np.array_equal(samples[:, 0], expected_frequency):
        raise RuntimeError(f"unexpected AM frequency grid: {label}")
    if not (
        np.all(np.isfinite(samples))
        and np.all(samples[:, 1] >= 0.0)
        and np.all(samples[:, 2] >= 0.0)
        and np.all(samples[:, 2] <= 1.0)
    ):
        raise RuntimeError(f"invalid AM physical domain: {label}")
    text = data.decode("utf-8")
    versions = VERSION_LINE.findall(text)
    if len(versions) != 1 or not versions[0].startswith("am version 12.2"):
        raise RuntimeError(f"unexpected AM version: {label}")
    if ERROR_LINE.search(text):
        raise RuntimeError(f"AM error diagnostic: {label}")
    warnings = [int(value) for value in UNRESOLVED_WARNING.findall(text)]
    if len(warnings) > 1:
        raise RuntimeError(f"multiple unresolved-line summaries: {label}")
    warning_headers = WARNING_HEADER.findall(text)
    allowed_summary = "Encountered in-band lines narrower than the frequency"
    unknown = [
        value
        for value in warning_headers
        if value != allowed_summary
        and re.fullmatch(r"Column included \d+ unresolved lines\.", value) is None
    ]
    if unknown:
        raise RuntimeError(f"unknown or cache-mutation AM warning {unknown!r}: {label}")
    if warning_headers and not warnings:
        raise RuntimeError(f"incomplete unresolved-line warning: {label}")
    return ParsedAM(
        frequency_ghz=samples[:, 0],
        tau_los=samples[:, 1],
        transmission=samples[:, 2],
        numeric_sha256=numeric.hexdigest(),
        normalized_sha256=sha256_bytes(normalize_am_output(data)),
        version=versions[0],
        unresolved_lines=warnings[0] if warnings else None,
    )


@dataclass(frozen=True)
class P1Record:
    target: str
    profile: str
    elevation_deg: int
    scale_decimal: str
    parsed: ParsedAM
    raw_relative_path: str
    raw_sha256: str
    sidecar_relative_path: str
    sidecar_sha256: str
    return_code: int


@dataclass(frozen=True)
class HoldoutRecord:
    kind: str
    profile: str
    requested_tau225: float
    achieved_tau225: float
    elevation_deg: int
    scale_decimal: str
    scale_hex: str
    analytic_transmission_decimal: str
    target_transmission_literal: str
    achieved_transmission_el80: float
    plateau_lower_outside_scale: float | None
    plateau_lower_inside_scale: float | None
    plateau_upper_inside_scale: float | None
    plateau_upper_outside_scale: float | None
    lower_tau_half_step: str
    upper_tau_half_step: str
    coordinate_acceptance_bound: str
    trace_relative_path: str
    trace_sha256: str
    parsed: ParsedAM
    raw_relative_path: str
    raw_sha256: str
    sidecar_relative_path: str
    sidecar_sha256: str
    return_code: int
    cache_id: str


class P1Cache:
    def __init__(self, cache_dir: Path, am_root: Path):
        self.cache_dir = cache_dir.resolve()
        self.am_root = am_root.resolve()
        self.context_path = self.cache_dir / "execution_context.json"
        self.context = canonical_json(self.context_path)
        self.context_sha256 = sha256_path(self.context_path)
        self.manifest_path = PACKAGE_DIR / "h2o_scale_hypothesis_manifest.json"
        manifest = canonical_json(self.manifest_path)
        self.manifest_sha256 = sha256_path(self.manifest_path)
        recorded = manifest["cache_execution_context"]
        if (
            recorded["sha256"] != self.context_sha256
            or recorded["content"] != self.context
        ):
            raise RuntimeError(
                "P1 cache execution context is not the committed canonical context"
            )
        imported_runner = Path(p1_driver.__file__).resolve()
        imported_runner_sha = sha256_path(imported_runner)
        if imported_runner_sha != self.context["runner"]["sha256"]:
            raise RuntimeError("imported P1 runner differs from canonical P1 context")
        if p1_driver.ROOT_ITERATIONS != 48 or p1_driver.MAX_BRACKET_EXPANSIONS != 64:
            raise RuntimeError("canonical P1 scale-solver iteration contract mismatch")
        build_payload = self.context["builds"]["regeneration"]
        executable_path = Path(build_payload["resolved_path"]).resolve()
        if (
            not executable_path.is_file()
            or executable_path.stat().st_size != build_payload["size_bytes"]
            or sha256_path(executable_path) != build_payload["sha256"]
        ):
            raise RuntimeError("canonical P1 executable identity mismatch")
        build = p1_driver.BuildIdentity(
            supplied_path=build_payload["supplied_path"],
            resolved_path=build_payload["resolved_path"],
            size_bytes=build_payload["size_bytes"],
            sha256=build_payload["sha256"],
            binary_format=build_payload["binary_format"],
        )
        parameters = self.context["execution_parameters"]
        self.runner = p1_driver.Runner(
            executable=build,
            am_root=self.am_root,
            cache_dir=self.cache_dir,
            omp_threads=parameters["omp_threads_per_process"],
            cache_shard_count=parameters["am_cache_sharding"]["shard_count"],
            execution_host=self.context["execution_host"],
            execution_context_sha256=self.context_sha256,
            execute=False,
        )
        self.scales_path = PACKAGE_DIR / "h2o_scale_hypothesis_scales.csv"
        recorded_scale = manifest["artifacts"]["h2o_scale_hypothesis_scales.csv"]
        self.scales_sha256 = sha256_path(self.scales_path)
        if recorded_scale["sha256"] != self.scales_sha256:
            raise RuntimeError("committed P1 scale table digest mismatch")
        self.scales: dict[tuple[str, str], dict[str, str]] = {}
        with self.scales_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target = row["target_model"]
                profile = row["source_profile"]
                if (target, profile) in TRAINING_KEYS:
                    self.scales[(target, profile)] = row
        expected = set(TRAINING_KEYS)
        if set(self.scales) != expected:
            raise RuntimeError("P1 frozen training scale table is incomplete")
        stage_flags = {
            row["ancillary_screening_transmission_rank1"]
            for row in self.scales.values()
        }
        if not stage_flags <= {"true", "false"}:
            raise RuntimeError("P1 frozen training stage flag is invalid")
        selected_stage_keys = {
            key
            for key, row in self.scales.items()
            if row["ancillary_screening_transmission_rank1"] == "true"
        }
        if selected_stage_keys != P1_SELECTED_STAGE_TRAINING_KEYS:
            raise RuntimeError(
                "P1 frozen selected-stage training identity is not the "
                "preregistered five-construction coverage"
            )
        self._records: dict[tuple[str, str, int], P1Record] = {}

    @contextmanager
    def shared_lock(self) -> Iterator[None]:
        lock_path = self.cache_dir / ".h2o_scale_hypothesis.lock"
        if not lock_path.is_file():
            raise RuntimeError(f"missing P1 cache lock: {lock_path}")
        with lock_path.open("rb") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "canonical P1 cache is currently writer-locked"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def scale(self, target: str, profile: str) -> str:
        return self.scales[(target, profile)]["fitted_h2o_scale_decimal"]

    def copied_npz_inventory(self, profiles: Sequence[str]) -> dict[str, Any]:
        manifest_path = PACKAGE_DIR / "copied_am_manifest.json"
        manifest = canonical_json(manifest_path)
        current_manifest_sha = sha256_path(manifest_path)
        p1_manifest = self.context["inputs"]["copied_suite_manifest"]
        current_product_manifest_sha = manifest["copied_suite"][
            "canonical_manifest_sha256"
        ]
        if (
            current_product_manifest_sha
            != p1_manifest["canonical_product_manifest_sha256"]
            or manifest["copied_suite"]["product_count"] != 25
            or len(manifest["copied_suite"]["products"]) != 25
            or len(self.context["inputs"]["copied_scale1_npz_products"]) != 25
        ):
            raise RuntimeError(
                "copied-suite canonical product manifest differs from P1 input"
            )
        copied_products = {
            Path(item["filename"]).stem: item
            for item in manifest["copied_suite"]["products"]
        }
        p1_products = {
            item["profile"]: item
            for item in self.context["inputs"]["copied_scale1_npz_products"]
        }
        if len(copied_products) != len(manifest["copied_suite"]["products"]):
            raise RuntimeError("duplicate copied-product identity in copied manifest")
        if len(p1_products) != len(
            self.context["inputs"]["copied_scale1_npz_products"]
        ):
            raise RuntimeError("duplicate copied-product identity in P1 inventory")
        products = []
        for profile in sorted(set(profiles)):
            if profile not in copied_products or profile not in p1_products:
                raise RuntimeError(f"missing copied scale-1 NPZ provenance: {profile}")
            copied = copied_products[profile]
            p1_item = p1_products[profile]
            path = (
                self.am_root / "Big_Atmosphere/LMT_am_npz" / f"{profile}.npz"
            ).resolve()
            actual_sha = sha256_path(path)
            actual_size = path.stat().st_size
            if (
                copied["filename"] != p1_item["filename"]
                or copied["sha256"] != p1_item["sha256"]
                or copied["md5"] != p1_item["md5"]
                or copied["bytes"] != p1_item["size_bytes"]
                or actual_sha != copied["sha256"]
                or actual_size != copied["bytes"]
            ):
                raise RuntimeError(
                    f"copied-manifest/P1/actual NPZ identity mismatch: {profile}"
                )
            copied_tau_los, copied_transmission = p1_driver.copied_anchor(
                self.am_root, profile
            )
            products.append(
                {
                    "profile": profile,
                    "path_relative_to_am_root": (
                        f"Big_Atmosphere/LMT_am_npz/{profile}.npz"
                    ),
                    "filename": copied["filename"],
                    "size_bytes": actual_size,
                    "sha256": actual_sha,
                    "md5_from_both_inventories": copied["md5"],
                    "copied_scale1_tau_los_el80_225ghz": f17(copied_tau_los),
                    "copied_scale1_transmission_el80_225ghz": f17(copied_transmission),
                }
            )
        return {
            "copied_am_manifest": {
                "path_relative_to_package": manifest_path.name,
                "size_bytes": manifest_path.stat().st_size,
                "sha256": current_manifest_sha,
                "canonical_product_manifest_sha256": (current_product_manifest_sha),
            },
            "p1_inventory": {
                "execution_context_sha256": self.context_sha256,
                "inventory_path": "inputs.copied_scale1_npz_products",
                "recorded_copied_suite_manifest": p1_manifest,
            },
            "whole_copied_manifest_identity_match": (
                current_manifest_sha == p1_manifest["sha256"]
                and manifest_path.stat().st_size == p1_manifest["size_bytes"]
            ),
            "whole_manifest_lineage_disposition": (
                "current_package_manifest_wrapper_revision_differs_from_"
                "P1_frozen_whole_file_but_canonical_product_manifest_and_"
                "each_used_NPZ_identity_match_exactly"
            ),
            "cross_inventory_identity_pass": True,
            "products": products,
        }

    def load(self, target: str, profile: str, elevation_deg: int) -> P1Record:
        key = (target, profile, elevation_deg)
        if key in self._records:
            return self._records[key]
        za = 90 - elevation_deg
        scale_row = self.scales[(target, profile)]
        scale = scale_row["fitted_h2o_scale_decimal"]
        stage = (
            "direct_full_grid_selected_transmission_rank1"
            if scale_row["ancillary_screening_transmission_rank1"] == "true"
            else "direct_full_grid_all_hypotheses"
        )
        spec = p1_driver.full_grid_spec(
            stage,
            profile,
            target,
            za,
            scale,
        )
        result = self.runner.run_or_load(spec)
        if set(result.sidecar) != CANONICAL_RUN_SIDECAR_KEYS:
            raise RuntimeError(f"canonical P1 sidecar key-set mismatch: {key}")
        cache_id = self.runner.cache_id(spec)
        if result.cache_id != cache_id:
            raise RuntimeError(f"canonical P1 cache-id mismatch: {key}")
        sidecar_path = self.runner.sidecar_path(cache_id)
        trace_row = self.scales[(target, profile)]
        trace_path = self.cache_dir / trace_row["scale_trace_path_relative_to_cache"]
        if sha256_path(trace_path) != trace_row["scale_trace_sha256"]:
            raise RuntimeError(f"P1 scale trace digest mismatch: {trace_path}")
        raw_path = self.runner.raw_path(cache_id).resolve()
        if not is_relative_to(raw_path, self.cache_dir):
            raise RuntimeError(f"P1 raw path escapes cache: {raw_path}")
        samples = result.parsed.samples
        parsed = ParsedAM(
            frequency_ghz=np.asarray(samples[:, 0], dtype=np.float64),
            tau_los=np.asarray(samples[:, 1], dtype=np.float64),
            transmission=np.asarray(samples[:, 2], dtype=np.float64),
            numeric_sha256=result.parsed.numeric_text_sha256,
            normalized_sha256=result.parsed.normalized_output_sha256,
            version=result.parsed.version_identity,
            unresolved_lines=result.parsed.warning_count,
        )
        record = P1Record(
            target=target,
            profile=profile,
            elevation_deg=elevation_deg,
            scale_decimal=scale,
            parsed=parsed,
            raw_relative_path=raw_path.relative_to(self.cache_dir).as_posix(),
            raw_sha256=result.raw_sha256,
            sidecar_relative_path=sidecar_path.relative_to(self.cache_dir).as_posix(),
            sidecar_sha256=sha256_path(sidecar_path),
            return_code=result.return_code,
        )
        self._records[key] = record
        return record


def integrate_record(
    record: P1Record, bandpasses: Sequence[Bandpass]
) -> dict[tuple[str, int], tuple[float, float, float]]:
    result: dict[tuple[str, int], tuple[float, float, float]] = {}
    for bandpass in bandpasses:
        transmission = np.interp(
            bandpass.frequency_ghz,
            record.parsed.frequency_ghz,
            record.parsed.transmission,
        )
        for alpha in ALPHAS:
            effective = float(np.dot(bandpass.weights(alpha), transmission))
            if not 0.0 < effective <= 1.0:
                raise RuntimeError(
                    f"invalid band transmission {effective}: {bandpass.identity}"
                )
            los_tau = -math.log(effective)
            correction = math.exp(los_tau)
            result[(bandpass.identity, alpha)] = (effective, los_tau, correction)
    return result


def target_coordinates() -> tuple[dict[str, float], dict[str, str]]:
    manifest_path = PACKAGE_DIR / "legacy_anchor_manifest.json"
    if sha256_path(manifest_path) != LEGACY_ANCHOR_MANIFEST_SHA256:
        raise RuntimeError("frozen legacy coordinate-manifest digest mismatch")
    manifest = canonical_json(manifest_path)
    tau: dict[str, float] = {}
    t225: dict[str, str] = {}
    for anchor in manifest["anchors"]:
        model = anchor["model"]
        if model in ("am_q0", *TARGETS):
            actual = (
                anchor["tau225_selector_anchor_binary64"],
                anchor["reference_225ghz_transmission_literal"],
            )
            if actual != FROZEN_TARGET_COORDINATES[model]:
                raise RuntimeError(f"frozen target coordinate mismatch: {model}")
            tau[model] = float(anchor["tau225_selector_anchor_binary64"])
            t225[model] = anchor["reference_225ghz_transmission_literal"]
    if tuple(sorted(tau, key=TARGET_ORDER.get)) != ("am_q0", *TARGETS):
        raise RuntimeError("legacy q0--q75 coordinate inventory mismatch")
    return tau, t225


def evaluate_operator_grid(
    nodes: np.ndarray,
    tau_nodes: np.ndarray,
    elevation_nodes_deg: np.ndarray,
    tau_query: np.ndarray,
    elevation_query_deg: np.ndarray,
    operator: str,
) -> np.ndarray:
    tau_query = np.asarray(tau_query, dtype=np.float64)
    elevation_query_deg = np.asarray(elevation_query_deg, dtype=np.float64)
    if (
        nodes.shape != (4, elevation_nodes_deg.size)
        or tau_nodes.shape != (4,)
        or not np.all(np.diff(tau_nodes) > 0.0)
        or not np.all(np.diff(elevation_nodes_deg) > 0.0)
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(tau_nodes))
        or not np.all(np.isfinite(elevation_nodes_deg))
        or tau_query.ndim != 1
        or elevation_query_deg.ndim != 1
        or not np.all(np.isfinite(tau_query))
        or not np.all(np.isfinite(elevation_query_deg))
        or np.any(tau_query < tau_nodes[0])
        or np.any(tau_query > tau_nodes[-1])
        or np.any(elevation_query_deg < elevation_nodes_deg[0])
        or np.any(elevation_query_deg > elevation_nodes_deg[-1])
        or operator not in OPERATORS
    ):
        raise ValueError("unsupported opacity/elevation/operator identity")

    # Frozen common elevation representation: PCHIP(lambda versus elevation)
    # at each nonzero anchor.  The clear row remains analytic zero.
    elevation_state = np.zeros((4, elevation_query_deg.size), dtype=np.float64)
    elevation_state[1:] = np.asarray(
        PchipInterpolator(
            elevation_nodes_deg,
            nodes[1:],
            axis=1,
            extrapolate=False,
        )(elevation_query_deg),
        dtype=np.float64,
    )
    output = np.empty((tau_query.size, elevation_query_deg.size), dtype=np.float64)
    low = tau_query <= tau_nodes[1]
    output[low] = (tau_query[low, None] / tau_nodes[1]) * elevation_state[1][None, :]
    high = ~low
    if np.any(high):
        if operator == "am12_piecewise_linear_los_tau_eval_v0":
            for column in range(elevation_query_deg.size):
                output[high, column] = np.interp(
                    tau_query[high], tau_nodes[1:], elevation_state[1:, column]
                )
        else:
            output[high] = np.asarray(
                PchipInterpolator(
                    tau_nodes[1:],
                    elevation_state[1:],
                    axis=0,
                    extrapolate=False,
                )(tau_query[high]),
                dtype=np.float64,
            )
    return output


def evaluate_named_operator(
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    *,
    lane: str,
    passband_id: str,
    alpha: int,
    operator: str,
    tau_nodes: np.ndarray,
    tau_query: np.ndarray,
    elevation_query_deg: np.ndarray,
) -> np.ndarray:
    if lane not in LANES or alpha not in ALPHAS:
        raise ValueError("unsupported lane or spectral identity")
    try:
        nodes = node_arrays[(lane, passband_id, alpha)]
    except KeyError as error:
        raise ValueError("unsupported passband identity") from error
    return evaluate_operator_grid(
        nodes,
        tau_nodes,
        ELEVATIONS_EVEN_DEG,
        tau_query,
        elevation_query_deg,
        operator,
    )


def fail_closed_contract_pass(
    node_arrays: dict[tuple[str, str, int], np.ndarray],
    *,
    lane: str,
    passband_id: str,
    alpha: int,
    tau_nodes: np.ndarray,
    operator: str,
) -> bool:
    nodes = node_arrays[(lane, passband_id, alpha)]
    probes = (
        (np.asarray([-np.finfo(float).tiny]), np.asarray([50.0]), operator),
        (np.asarray([np.nan]), np.asarray([50.0]), operator),
        (
            np.asarray([np.nextafter(tau_nodes[-1], math.inf)]),
            np.asarray([50.0]),
            operator,
        ),
        (
            np.asarray([tau_nodes[1]]),
            np.asarray([np.nextafter(20.0, -math.inf)]),
            operator,
        ),
        (
            np.asarray([tau_nodes[1]]),
            np.asarray([np.nextafter(80.0, math.inf)]),
            operator,
        ),
        (np.asarray([tau_nodes[1]]), np.asarray([np.nan]), operator),
        (np.asarray([tau_nodes[1]]), np.asarray([50.0]), "invalid_operator"),
    )
    for tau, elevation, identity in probes:
        try:
            evaluate_operator_grid(
                nodes,
                tau_nodes,
                ELEVATIONS_EVEN_DEG,
                tau,
                elevation,
                identity,
            )
        except ValueError:
            continue
        return False

    named_identity_probes = (
        {"lane": lane, "passband_id": "invalid_passband", "alpha": alpha},
        {"lane": lane, "passband_id": passband_id, "alpha": 999},
        {"lane": "invalid_lane", "passband_id": passband_id, "alpha": alpha},
    )
    for identity in named_identity_probes:
        try:
            evaluate_named_operator(
                node_arrays,
                lane=identity["lane"],
                passband_id=identity["passband_id"],
                alpha=identity["alpha"],
                operator=operator,
                tau_nodes=tau_nodes,
                tau_query=np.asarray([tau_nodes[1]]),
                elevation_query_deg=np.asarray([50.0]),
            )
        except ValueError:
            continue
        return False

    # Missing either a required opacity anchor or an elevation bracket must
    # fail before interpolation; neither reduced support nor extrapolation is
    # an accepted substitute.
    bracket_probes = (
        (nodes[:3], tau_nodes[:3], ELEVATIONS_EVEN_DEG, 50.0),
        (nodes[:, :-1], tau_nodes, ELEVATIONS_EVEN_DEG[:-1], 80.0),
    )
    for (
        incomplete_nodes,
        incomplete_tau,
        incomplete_elevation,
        query_elevation,
    ) in bracket_probes:
        try:
            evaluate_operator_grid(
                incomplete_nodes,
                incomplete_tau,
                incomplete_elevation,
                np.asarray([tau_nodes[1]]),
                np.asarray([query_elevation]),
                operator,
            )
        except ValueError:
            continue
        return False

    zero_normalization = Bandpass(
        identity="g5_nonpositive_normalization_probe",
        array="a1100",
        family="contract_probe",
        frequency_ghz=np.asarray([200.0, 201.0]),
        response=np.asarray([0.0, 0.0]),
        source_path="generated_contract_probe",
        source_sha256=hashlib.sha256(b"g5_nonpositive_normalization_probe").hexdigest(),
        source_commit="none",
        convention="deliberately zero for G5",
    )
    try:
        zero_normalization.weights(alpha)
    except RuntimeError:
        return True
    return False


def summarize_errors(
    errors: Sequence[float],
    *,
    metric_group: str,
    lane: str,
    operator: str,
    bandpass: Bandpass,
    alpha: int,
    evidence_slice: str,
    gated: bool,
) -> dict[str, str]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        raise RuntimeError(f"cannot summarize empty error set: {metric_group}")
    maximum = float(np.max(np.abs(values)))
    return {
        "metric_group": metric_group,
        "lane": lane,
        "operator": operator,
        "passband_id": bandpass.identity,
        "array": bandpass.array,
        "alpha": str(alpha),
        "evidence_slice": evidence_slice,
        "n": str(values.size),
        "signed_min_fractional_correction_error": f17(float(np.min(values))),
        "signed_max_fractional_correction_error": f17(float(np.max(values))),
        "rms_fractional_correction_error": f17(float(np.sqrt(np.mean(values**2)))),
        "p95_absolute_fractional_correction_error": f17(
            float(np.quantile(np.abs(values), 0.95, method="linear"))
        ),
        "median_absolute_fractional_correction_error": f17(
            float(np.median(np.abs(values)))
        ),
        "max_absolute_fractional_correction_error": f17(maximum),
        "gate_threshold": f17(FIDELITY_GATE) if gated else "",
        "gate_pass": bool_text(maximum <= FIDELITY_GATE) if gated else "not_gated",
    }


def coverage_section(
    expected_keys: set[tuple[str, ...]],
    actual_keys: Sequence[tuple[str, ...]],
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    counts: dict[tuple[str, ...], int] = {}
    for key in actual_keys:
        counts[key] = counts.get(key, 0) + 1
    actual_unique = set(counts)

    def serialize(key: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(fieldnames, key, strict=True))

    missing = sorted(expected_keys - actual_unique)
    unexpected = sorted(actual_unique - expected_keys)
    duplicates = sorted((key, count) for key, count in counts.items() if count != 1)
    passed = not missing and not unexpected and not duplicates
    return {
        "expected_row_count": len(expected_keys),
        "actual_row_count": len(actual_keys),
        "actual_unique_key_count": len(actual_unique),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "duplicate_key_count": len(duplicates),
        "missing_key_anti_join": [serialize(key) for key in missing],
        "unexpected_key_anti_join": [serialize(key) for key in unexpected],
        "duplicate_keys": [
            {**serialize(key), "multiplicity": count} for key, count in duplicates
        ],
        "pass": passed,
    }


def expanded_holdout_coverage(
    holdout_records: Sequence[HoldoutRecord],
    holdout_rows: Sequence[dict[str, str]],
    bandpasses: Sequence[Bandpass],
) -> dict[str, Any]:
    raw_expected = {
        (kind, profile, str(int(elevation)))
        for kind, profile, _lower, _upper in HOLDOUT_CASES
        for elevation in ELEVATIONS_ODD_DEG
    }
    raw_actual = [
        (item.kind, item.profile, str(item.elevation_deg)) for item in holdout_records
    ]
    expanded_expected = {
        (
            kind,
            profile,
            str(int(elevation)),
            lane,
            operator,
            bandpass.identity,
            str(alpha),
        )
        for kind, profile, _lower, _upper in HOLDOUT_CASES
        for elevation in ELEVATIONS_ODD_DEG
        for lane in LANES
        for operator in OPERATORS
        for bandpass in bandpasses
        for alpha in ALPHAS
    }
    expanded_actual = [
        (
            row["holdout_kind"],
            row["truth_profile"],
            row["elevation_deg"],
            row["lane"],
            row["operator"],
            row["passband_id"],
            row["alpha"],
        )
        for row in holdout_rows
    ]
    raw = coverage_section(
        raw_expected,
        raw_actual,
        ("holdout_kind", "truth_profile", "elevation_deg"),
    )
    expanded = coverage_section(
        expanded_expected,
        expanded_actual,
        (
            "holdout_kind",
            "truth_profile",
            "elevation_deg",
            "lane",
            "operator",
            "passband_id",
            "alpha",
        ),
    )
    if len(raw_expected) != 240 or len(expanded_expected) != 23040:
        raise RuntimeError("internal frozen G8 cardinality mismatch")
    report = {
        "schema_version": f"{SCHEMA_VERSION}-coverage-v1",
        "required_dimensions": {
            "opacity_interval_count": 3,
            "registered_midpoint_profile_case_count": 8,
            "odd_elevation_count": 30,
            "lane_count": 2,
            "operator_count": 2,
            "passband_count": 6,
            "spectral_index_count": 4,
        },
        "raw_direct_grid_coverage": raw,
        "expanded_holdout_row_coverage": expanded,
        "pass": bool(raw["pass"] and expanded["pass"]),
    }
    if not report["pass"]:
        raise RuntimeError(
            "G8 holdout anti-join failed: "
            f"raw={raw['actual_row_count']}/240, "
            f"expanded={expanded['actual_row_count']}/23040, "
            f"missing={expanded['missing_key_count']}, "
            f"unexpected={expanded['unexpected_key_count']}, "
            f"duplicates={expanded['duplicate_key_count']}"
        )
    return report


def build_study(
    p1: P1Cache,
    bandpasses: Sequence[Bandpass],
    holdout_records: Sequence["HoldoutRecord"],
    holdout_run_rows: Sequence[dict[str, str]],
    holdout_execution_context: dict[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if len(holdout_records) != 240:
        raise RuntimeError(
            f"G8 requires exactly 240 direct holdout grids, found {len(holdout_records)}"
        )
    tau_by_target, t225_by_target = target_coordinates()
    tau_nodes = np.asarray([tau_by_target[name] for name in ("am_q0", *TARGETS)])
    even_airmass_desc = modified_airmass(ELEVATIONS_EVEN_DEG)

    direct: dict[tuple[str, str, int, str, int], tuple[float, float, float]] = {}
    run_inventory: list[dict[str, str]] = []
    for target, profile in TRAINING_KEYS:
        for elevation_value in ELEVATIONS_EVEN_DEG.astype(int):
            record = p1.load(target, profile, int(elevation_value))
            if int(elevation_value) == 80:
                index_225 = 22500
                if record.parsed.frequency_ghz[
                    index_225
                ] != 225.0 or record.parsed.transmission[index_225] != float(
                    t225_by_target[target]
                ):
                    raise RuntimeError(
                        f"G1 training EL80 T225 mismatch: {target}/{profile}"
                    )
            for (passband_id, alpha), values in integrate_record(
                record, bandpasses
            ).items():
                direct[(target, profile, int(elevation_value), passband_id, alpha)] = (
                    values
                )
            run_inventory.append(
                {
                    "target": target,
                    "profile": profile,
                    "elevation_deg": str(int(elevation_value)),
                    "zenith_angle_deg": str(90 - int(elevation_value)),
                    "h2o_scale_decimal": record.scale_decimal,
                    "raw_path_relative_to_p1_cache": record.raw_relative_path,
                    "raw_sha256": record.raw_sha256,
                    "sidecar_path_relative_to_p1_cache": record.sidecar_relative_path,
                    "sidecar_sha256": record.sidecar_sha256,
                    "return_code": str(record.return_code),
                    "unresolved_line_warning_count": (
                        ""
                        if record.parsed.unresolved_lines is None
                        else str(record.parsed.unresolved_lines)
                    ),
                }
            )
    expected_training_count = len(TRAINING_KEYS) * ELEVATIONS_EVEN_DEG.size
    if len(run_inventory) != expected_training_count or expected_training_count != 155:
        raise RuntimeError(
            "G8 P1 training-grid coverage mismatch: "
            f"expected 155, found {len(run_inventory)}"
        )

    node_arrays: dict[tuple[str, str, int], np.ndarray] = {}
    node_rows: list[dict[str, str]] = []
    for lane, lane_profiles in LANES.items():
        for bandpass in bandpasses:
            for alpha in ALPHAS:
                grid = np.zeros((4, ELEVATIONS_EVEN_DEG.size), dtype=np.float64)
                for target_index, target in enumerate(TARGETS, start=1):
                    profile = lane_profiles[target]
                    for elevation_index, elevation_value in enumerate(
                        ELEVATIONS_EVEN_DEG.astype(int)
                    ):
                        _, los_tau, correction = direct[
                            (
                                target,
                                profile,
                                int(elevation_value),
                                bandpass.identity,
                                alpha,
                            )
                        ]
                        grid[target_index, elevation_index] = los_tau
                        node_rows.append(
                            {
                                "lane": lane,
                                "target": target,
                                "source_profile": profile,
                                "tau225": f17(tau_by_target[target]),
                                "reference_t225_literal": t225_by_target[target],
                                "elevation_deg": str(int(elevation_value)),
                                "airmass": f17(
                                    float(even_airmass_desc[elevation_index])
                                ),
                                "passband_id": bandpass.identity,
                                "array": bandpass.array,
                                "alpha": str(alpha),
                                "line_of_sight_optical_depth": f17(los_tau),
                                "extinction_correction": f17(correction),
                            }
                        )
                node_arrays[(lane, bandpass.identity, alpha)] = grid

    metrics: list[dict[str, str]] = []
    physical_rows: list[dict[str, str]] = []
    dense_tau = np.unique(
        np.concatenate(
            [
                np.linspace(tau_nodes[0], tau_nodes[-1], 1001),
                tau_nodes,
                (tau_nodes[:-1] + tau_nodes[1:]) / 2.0,
            ]
        )
    )
    dense_elevation = np.unique(
        np.concatenate([np.linspace(20.0, 80.0, 601), ELEVATIONS_EVEN_DEG])
    )
    for lane in LANES:
        for operator in OPERATORS:
            for bandpass in bandpasses:
                for alpha in ALPHAS:
                    nodes = node_arrays[(lane, bandpass.identity, alpha)]
                    evaluated = evaluate_operator_grid(
                        nodes,
                        tau_nodes,
                        ELEVATIONS_EVEN_DEG,
                        dense_tau,
                        dense_elevation,
                        operator,
                    )
                    finite_pass = bool(np.all(np.isfinite(evaluated)))
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        effective_transmission = np.exp(-evaluated)
                        extinction_correction = np.exp(evaluated)
                    g2_domain_pass = bool(
                        finite_pass
                        and np.all(effective_transmission > 0.0)
                        and np.all(
                            effective_transmission <= math.exp(PHYSICAL_TOLERANCE)
                        )
                        and np.all(extinction_correction > 0.0)
                        and np.all(
                            extinction_correction >= math.exp(-PHYSICAL_TOLERANCE)
                        )
                    )
                    tau_deltas = np.diff(evaluated, axis=0)
                    elevation_deltas = np.diff(evaluated, axis=1)
                    tau_min_delta = float(np.min(tau_deltas))
                    elevation_max_delta = float(np.max(elevation_deltas))
                    tau_wrong = tau_deltas < -PHYSICAL_TOLERANCE
                    elevation_wrong = elevation_deltas > PHYSICAL_TOLERANCE
                    tau_wrong_count = int(np.count_nonzero(tau_wrong))
                    elevation_wrong_count = int(np.count_nonzero(elevation_wrong))
                    tau_wrong_correction_excursion = (
                        float(np.max(np.abs(np.expm1(tau_deltas[tau_wrong]))))
                        if tau_wrong_count
                        else 0.0
                    )
                    elevation_wrong_correction_excursion = (
                        float(
                            np.max(np.abs(np.expm1(elevation_deltas[elevation_wrong])))
                        )
                        if elevation_wrong_count
                        else 0.0
                    )
                    exact = evaluate_operator_grid(
                        nodes,
                        tau_nodes,
                        ELEVATIONS_EVEN_DEG,
                        tau_nodes,
                        ELEVATIONS_EVEN_DEG,
                        operator,
                    )
                    anchor_residual = float(np.max(np.abs(exact - nodes)))
                    low_tau_probe = np.asarray(
                        [0.0, tau_nodes[1] / 3.0, tau_nodes[1] / 2.0, tau_nodes[1]]
                    )
                    low_values = evaluate_operator_grid(
                        nodes,
                        tau_nodes,
                        ELEVATIONS_EVEN_DEG,
                        low_tau_probe,
                        dense_elevation,
                        operator,
                    )
                    low_expected = (low_tau_probe[:, None] / tau_nodes[1]) * low_values[
                        -1
                    ][None, :]
                    low_segment_residual = float(
                        np.max(np.abs(low_values - low_expected))
                    )
                    minimum = float(np.min(evaluated))
                    minimum_index = np.unravel_index(
                        int(np.argmin(evaluated)), evaluated.shape
                    )
                    physical_pass = (
                        g2_domain_pass
                        and minimum >= -PHYSICAL_TOLERANCE
                        and tau_min_delta >= -PHYSICAL_TOLERANCE
                        and elevation_max_delta <= PHYSICAL_TOLERANCE
                        and anchor_residual <= ANCHOR_TOLERANCE
                        and low_segment_residual <= 1.0e-12
                    )
                    span = tau_nodes[-1] - tau_nodes[0]
                    continuity_errors: list[float] = []
                    for knot in tau_nodes[1:-1]:
                        queries = (
                            np.nextafter(knot, -math.inf),
                            np.nextafter(knot, math.inf),
                            knot - 1.0e-12 * span,
                            knot + 1.0e-12 * span,
                        )
                        values = evaluate_operator_grid(
                            nodes,
                            tau_nodes,
                            ELEVATIONS_EVEN_DEG,
                            np.asarray(queries),
                            dense_elevation,
                            operator,
                        )
                        continuity_errors.extend(
                            np.abs(np.exp(values[0] - values[1]) - 1.0).tolist()
                        )
                        continuity_errors.extend(
                            np.abs(np.exp(values[2] - values[3]) - 1.0).tolist()
                        )
                    max_continuity = max(continuity_errors)
                    fail_closed = fail_closed_contract_pass(
                        node_arrays,
                        lane=lane,
                        passband_id=bandpass.identity,
                        alpha=alpha,
                        tau_nodes=tau_nodes,
                        operator=operator,
                    )
                    physical_pass = (
                        physical_pass and max_continuity <= 1.0e-10 and fail_closed
                    )
                    physical_rows.append(
                        {
                            "lane": lane,
                            "operator": operator,
                            "passband_id": bandpass.identity,
                            "array": bandpass.array,
                            "alpha": str(alpha),
                            "all_evaluated_quantities_finite": bool_text(finite_pass),
                            "minimum_line_of_sight_optical_depth": f17(minimum),
                            "minimum_lambda_tau225": f17(
                                float(dense_tau[minimum_index[0]])
                            ),
                            "minimum_lambda_elevation_deg": f17(
                                float(dense_elevation[minimum_index[1]])
                            ),
                            "maximum_effective_transmission": f17(
                                float(np.max(effective_transmission))
                            ),
                            "minimum_extinction_correction": f17(
                                float(np.min(extinction_correction))
                            ),
                            "minimum_tau_direction_delta": f17(tau_min_delta),
                            "maximum_elevation_direction_delta": f17(
                                elevation_max_delta
                            ),
                            "tau_wrong_way_step_count": str(tau_wrong_count),
                            "elevation_wrong_way_step_count": str(
                                elevation_wrong_count
                            ),
                            "maximum_tau_wrong_way_fractional_correction_excursion": (
                                f17(tau_wrong_correction_excursion)
                            ),
                            "maximum_elevation_wrong_way_fractional_correction_excursion": (
                                f17(elevation_wrong_correction_excursion)
                            ),
                            "maximum_anchor_absolute_residual": f17(anchor_residual),
                            "maximum_low_segment_absolute_residual": f17(
                                low_segment_residual
                            ),
                            "maximum_relative_correction_continuity_residual": f17(
                                max_continuity
                            ),
                            "positivity_pass": bool_text(
                                minimum >= -PHYSICAL_TOLERANCE
                            ),
                            "g2_domain_pass": bool_text(g2_domain_pass),
                            "tau_monotonicity_pass": bool_text(
                                tau_min_delta >= -PHYSICAL_TOLERANCE
                            ),
                            "elevation_monotonicity_pass": bool_text(
                                elevation_max_delta <= PHYSICAL_TOLERANCE
                            ),
                            "continuity_pass": bool_text(max_continuity <= 1.0e-10),
                            "fail_closed_pass": bool_text(fail_closed),
                            "exact_anchor_pass": bool_text(
                                anchor_residual <= ANCHOR_TOLERANCE
                            ),
                            "exact_low_segment_pass": bool_text(
                                low_segment_residual <= 1.0e-12
                            ),
                            "physical_contract_pass": bool_text(physical_pass),
                        }
                    )

    for operator in OPERATORS:
        for bandpass in bandpasses:
            for alpha in ALPHAS:
                first = evaluate_operator_grid(
                    node_arrays[("fixed_djf25_v1", bandpass.identity, alpha)],
                    tau_nodes,
                    ELEVATIONS_EVEN_DEG,
                    dense_tau,
                    dense_elevation,
                    operator,
                )
                second = evaluate_operator_grid(
                    node_arrays[("conditioned_djf_v1", bandpass.identity, alpha)],
                    tau_nodes,
                    ELEVATIONS_EVEN_DEG,
                    dense_tau,
                    dense_elevation,
                    operator,
                )
                errors = np.exp(first - second).ravel() - 1.0
                metrics.append(
                    summarize_errors(
                        errors,
                        metric_group="lane_disagreement_dense_domain",
                        lane="fixed_djf25_v1_vs_conditioned_djf_v1",
                        operator=operator,
                        bandpass=bandpass,
                        alpha=alpha,
                        evidence_slice="q0_q75_el20_el80",
                        gated=False,
                    )
                )

    holdout_rows: list[dict[str, str]] = []
    scale_rows: list[dict[str, str]] = []
    if holdout_records:
        for record in holdout_records:
            if record.elevation_deg != int(ELEVATIONS_ODD_DEG[0]):
                continue
            scale_rows.append(
                {
                    "holdout_kind": record.kind,
                    "profile": record.profile,
                    "requested_tau225": f17(record.requested_tau225),
                    "achieved_tau225": f17(record.achieved_tau225),
                    "coordinate_residual": f17(
                        record.achieved_tau225 - record.requested_tau225
                    ),
                    "analytic_transmission_decimal": (
                        record.analytic_transmission_decimal
                    ),
                    "target_transmission_literal": record.target_transmission_literal,
                    "achieved_transmission_el80": f17(
                        record.achieved_transmission_el80
                    ),
                    "h2o_scale_decimal": record.scale_decimal,
                    "h2o_scale_hex": record.scale_hex,
                    "lower_tau_half_step": record.lower_tau_half_step,
                    "upper_tau_half_step": record.upper_tau_half_step,
                    "coordinate_acceptance_bound": (record.coordinate_acceptance_bound),
                    "plateau_lower_outside_scale": (
                        ""
                        if record.plateau_lower_outside_scale is None
                        else f17(record.plateau_lower_outside_scale)
                    ),
                    "plateau_lower_inside_scale": (
                        ""
                        if record.plateau_lower_inside_scale is None
                        else f17(record.plateau_lower_inside_scale)
                    ),
                    "plateau_upper_inside_scale": (
                        ""
                        if record.plateau_upper_inside_scale is None
                        else f17(record.plateau_upper_inside_scale)
                    ),
                    "plateau_upper_outside_scale": (
                        ""
                        if record.plateau_upper_outside_scale is None
                        else f17(record.plateau_upper_outside_scale)
                    ),
                    "trace_path_relative_to_holdout_cache": record.trace_relative_path,
                    "trace_sha256": record.trace_sha256,
                }
            )
        for record in holdout_records:
            integrated = integrate_record(record, bandpasses)
            target_tau = record.achieved_tau225
            for lane in LANES:
                for operator in OPERATORS:
                    query_elevation = np.asarray([float(record.elevation_deg)])
                    for bandpass in bandpasses:
                        for alpha in ALPHAS:
                            predicted_tau = float(
                                evaluate_operator_grid(
                                    node_arrays[(lane, bandpass.identity, alpha)],
                                    tau_nodes,
                                    ELEVATIONS_EVEN_DEG,
                                    np.asarray([target_tau]),
                                    query_elevation,
                                    operator,
                                )[0, 0]
                            )
                            direct_correction = integrated[(bandpass.identity, alpha)][
                                2
                            ]
                            error = math.exp(predicted_tau) / direct_correction - 1.0
                            holdout_rows.append(
                                {
                                    "holdout_kind": record.kind,
                                    "truth_profile": record.profile,
                                    "requested_tau225": f17(record.requested_tau225),
                                    "achieved_tau225": f17(record.achieved_tau225),
                                    "coordinate_residual": f17(
                                        record.achieved_tau225 - record.requested_tau225
                                    ),
                                    "h2o_scale_decimal": record.scale_decimal,
                                    "h2o_scale_hex": record.scale_hex,
                                    "analytic_transmission_decimal": (
                                        record.analytic_transmission_decimal
                                    ),
                                    "target_transmission_literal": (
                                        record.target_transmission_literal
                                    ),
                                    "achieved_transmission_el80": f17(
                                        record.achieved_transmission_el80
                                    ),
                                    "elevation_deg": str(record.elevation_deg),
                                    "airmass": f17(
                                        float(modified_airmass(record.elevation_deg))
                                    ),
                                    "lane": lane,
                                    "operator": operator,
                                    "passband_id": bandpass.identity,
                                    "array": bandpass.array,
                                    "alpha": str(alpha),
                                    "direct_extinction_correction": f17(
                                        direct_correction
                                    ),
                                    "operator_extinction_correction": f17(
                                        math.exp(predicted_tau)
                                    ),
                                    "fractional_correction_error": f17(error),
                                    "raw_sha256": record.raw_sha256,
                                    "sidecar_sha256": record.sidecar_sha256,
                                }
                            )
        coverage = expanded_holdout_coverage(
            holdout_records,
            holdout_rows,
            bandpasses,
        )
        full_grid_inventory_count = sum(
            row["run_class"] == "midpoint_odd_elevation_full_grid"
            for row in holdout_run_rows
        )
        scale_search_inventory_count = sum(
            row["run_class"] == "midpoint_scale_search_anchor"
            for row in holdout_run_rows
        )
        if (
            len(scale_rows) != 8
            or full_grid_inventory_count != 240
            or scale_search_inventory_count <= 0
            or len(holdout_run_rows)
            != full_grid_inventory_count + scale_search_inventory_count
        ):
            raise RuntimeError(
                "G8 scale/raw-inventory coverage mismatch: "
                f"scales={len(scale_rows)}, full={full_grid_inventory_count}, "
                f"scale_search={scale_search_inventory_count}, "
                f"total={len(holdout_run_rows)}"
            )
        grouped: dict[tuple[str, str, str, int, str], list[float]] = {}
        bandpass_by_id = {item.identity: item for item in bandpasses}
        for row in holdout_rows:
            key = (
                row["lane"],
                row["operator"],
                row["passband_id"],
                int(row["alpha"]),
                row["holdout_kind"],
            )
            grouped.setdefault(key, []).append(
                float(row["fractional_correction_error"])
            )
        for (lane, operator, passband_id, alpha, kind), errors in sorted(
            grouped.items()
        ):
            bandpass = bandpass_by_id[passband_id]
            metrics.append(
                summarize_errors(
                    errors,
                    metric_group="direct_am_holdout_representation_fidelity",
                    lane=lane,
                    operator=operator,
                    bandpass=bandpass,
                    alpha=alpha,
                    evidence_slice=kind,
                    gated=True,
                )
            )
        combined: dict[tuple[str, str, str, int], list[float]] = {}
        for row in holdout_rows:
            key = (
                row["lane"],
                row["operator"],
                row["passband_id"],
                int(row["alpha"]),
            )
            combined.setdefault(key, []).append(
                float(row["fractional_correction_error"])
            )
        for (lane, operator, passband_id, alpha), errors in sorted(combined.items()):
            source_rows = [
                row
                for row in holdout_rows
                if row["lane"] == lane
                and row["operator"] == operator
                and row["passband_id"] == passband_id
                and int(row["alpha"]) == alpha
            ]
            worst = max(
                source_rows,
                key=lambda row: abs(float(row["fractional_correction_error"])),
            )
            summary_row = summarize_errors(
                errors,
                metric_group="direct_am_holdout_representation_fidelity",
                lane=lane,
                operator=operator,
                bandpass=bandpass_by_id[passband_id],
                alpha=alpha,
                evidence_slice="combined_required_holdouts",
                gated=True,
            )
            summary_row.update(
                {
                    "worst_holdout_kind": worst["holdout_kind"],
                    "worst_truth_profile": worst["truth_profile"],
                    "worst_h2o_scale_decimal": worst["h2o_scale_decimal"],
                    "worst_elevation_deg": worst["elevation_deg"],
                }
            )
            metrics.append(summary_row)

        primary_id = {
            item.array: item.identity
            for item in bandpasses
            if item.family == "primary_tolteca_ecsv"
        }
        fts_id = {
            item.array: item.identity
            for item in bandpasses
            if item.family == "representative_fts_challenger"
        }
        truth_index: dict[tuple[str, str, str, str, str], float] = {}
        for row in holdout_rows:
            key = (
                row["holdout_kind"],
                row["truth_profile"],
                row["elevation_deg"],
                row["passband_id"],
                row["alpha"],
            )
            value = float(row["direct_extinction_correction"])
            if key in truth_index and truth_index[key] != value:
                raise RuntimeError(f"inconsistent repeated direct truth row: {key}")
            truth_index[key] = value
        for array in ("a1100", "a1400", "a2000"):
            for alpha in ALPHAS:
                errors = []
                for record in holdout_records:
                    common = (record.kind, record.profile, str(record.elevation_deg))
                    primary = truth_index[(*common, primary_id[array], str(alpha))]
                    challenger = truth_index[(*common, fts_id[array], str(alpha))]
                    errors.append(challenger / primary - 1.0)
                primary_bandpass = next(
                    item for item in bandpasses if item.identity == primary_id[array]
                )
                worst_index = int(np.argmax(np.abs(np.asarray(errors))))
                worst_record = holdout_records[worst_index]
                summary_row = summarize_errors(
                    errors,
                    metric_group="fts_truth_vs_primary_truth",
                    lane="direct_am_truth",
                    operator="none",
                    bandpass=primary_bandpass,
                    alpha=alpha,
                    evidence_slice="all_8_profiles_30_odd_elevations",
                    gated=False,
                )
                summary_row.update(
                    {
                        "worst_holdout_kind": worst_record.kind,
                        "worst_truth_profile": worst_record.profile,
                        "worst_h2o_scale_decimal": worst_record.scale_decimal,
                        "worst_elevation_deg": str(worst_record.elevation_deg),
                    }
                )
                metrics.append(summary_row)

    bandpass_rows: list[dict[str, str]] = []
    for bandpass in bandpasses:
        for alpha in ALPHAS:
            normalized = bandpass.weights(alpha)
            bandpass_rows.append(
                {
                    "passband_id": bandpass.identity,
                    "array": bandpass.array,
                    "family": bandpass.family,
                    "alpha": str(alpha),
                    "source_path": bandpass.source_path,
                    "source_sha256": bandpass.source_sha256,
                    "source_commit": bandpass.source_commit,
                    "sample_count": str(bandpass.frequency_ghz.size),
                    "frequency_min_ghz": f17(float(bandpass.frequency_ghz[0])),
                    "frequency_max_ghz": f17(float(bandpass.frequency_ghz[-1])),
                    "effective_frequency_ghz": f17(
                        float(np.dot(normalized, bandpass.frequency_ghz))
                    ),
                    "clipped_negative_node_count": str(bandpass.clipped_node_count),
                    "clipped_negative_integral_fraction": f17(
                        bandpass.clipped_negative_integral_fraction
                    ),
                    "response_convention": bandpass.convention,
                    "spectral_convention": (
                        f"top-of-atmosphere S_nu proportional to nu^{alpha}; "
                        "energy-weighted throughput"
                    ),
                }
            )

    metric_fields = [
        "metric_group",
        "lane",
        "operator",
        "passband_id",
        "array",
        "alpha",
        "evidence_slice",
        "n",
        "signed_min_fractional_correction_error",
        "signed_max_fractional_correction_error",
        "rms_fractional_correction_error",
        "p95_absolute_fractional_correction_error",
        "median_absolute_fractional_correction_error",
        "max_absolute_fractional_correction_error",
        "worst_holdout_kind",
        "worst_truth_profile",
        "worst_h2o_scale_decimal",
        "worst_elevation_deg",
        "gate_threshold",
        "gate_pass",
    ]
    metrics = sorted(
        metrics,
        key=lambda row: (
            row["metric_group"],
            row["lane"],
            row["operator"],
            row["passband_id"],
            int(row["alpha"]),
            row["evidence_slice"],
        ),
    )

    combined_expected = {
        (lane, operator, bandpass.identity, str(alpha))
        for lane in LANES
        for operator in OPERATORS
        for bandpass in bandpasses
        for alpha in ALPHAS
    }
    combined_actual = [
        (row["lane"], row["operator"], row["passband_id"], row["alpha"])
        for row in metrics
        if row["metric_group"] == "direct_am_holdout_representation_fidelity"
        and row["evidence_slice"] == "combined_required_holdouts"
    ]
    combined_coverage = coverage_section(
        combined_expected,
        combined_actual,
        ("lane", "operator", "passband_id", "alpha"),
    )
    if len(combined_expected) != 96 or not combined_coverage["pass"]:
        raise RuntimeError("G8 combined-summary anti-join failed")
    coverage["combined_required_metric_coverage"] = combined_coverage
    coverage["pass"] = bool(coverage["pass"] and combined_coverage["pass"])

    if len(physical_rows) != 96:
        raise RuntimeError(
            f"G8 physical-metric coverage mismatch: expected 96, found {len(physical_rows)}"
        )
    physical_pass = all(
        row["physical_contract_pass"] == "true" for row in physical_rows
    )
    primary_ids = {
        item.identity for item in bandpasses if item.family == "primary_tolteca_ecsv"
    }
    challenger_ids = {
        item.identity
        for item in bandpasses
        if item.family == "representative_fts_challenger"
    }
    truth_difference_rows = [
        row for row in metrics if row["metric_group"] == "fts_truth_vs_primary_truth"
    ]
    truth_difference_max = max(
        float(row["max_absolute_fractional_correction_error"])
        for row in truth_difference_rows
    )
    truth_difference_location = max(
        truth_difference_rows,
        key=lambda row: float(row["max_absolute_fractional_correction_error"]),
    )
    g0_provenance_pass = bool(
        holdout_execution_context["scale_solver_contract"]["identity_match_pass"]
        and holdout_execution_context["copied_scale1_npz_inputs"][
            "cross_inventory_identity_pass"
        ]
        and len(run_inventory) == 155
        and len(holdout_run_rows) == (240 + scale_search_inventory_count)
    )
    if not g0_provenance_pass:
        raise RuntimeError("G0 provenance/execution-context gate failed")
    candidate_decisions: list[dict[str, Any]] = []
    for lane in LANES:
        for operator in OPERATORS:
            combined_rows = [
                row
                for row in metrics
                if row["metric_group"] == "direct_am_holdout_representation_fidelity"
                and row["evidence_slice"] == "combined_required_holdouts"
                and row["lane"] == lane
                and row["operator"] == operator
            ]
            primary_rows = [
                row for row in combined_rows if row["passband_id"] in primary_ids
            ]
            challenger_rows = [
                row for row in combined_rows if row["passband_id"] in challenger_ids
            ]
            if len(primary_rows) != 12 or len(challenger_rows) != 12:
                raise RuntimeError(
                    f"G8 combined metric coverage mismatch: {lane}/{operator}"
                )
            primary_pass = len(primary_rows) == 12 and all(
                row["gate_pass"] == "true" for row in primary_rows
            )
            primary_physical_pass = all(
                row["physical_contract_pass"] == "true"
                for row in physical_rows
                if row["lane"] == lane
                and row["operator"] == operator
                and row["passband_id"] in primary_ids
            )
            challenger_physical_pass = all(
                row["physical_contract_pass"] == "true"
                for row in physical_rows
                if row["lane"] == lane
                and row["operator"] == operator
                and row["passband_id"] in challenger_ids
            )
            candidate_physical_rows = [
                row
                for row in physical_rows
                if row["lane"] == lane and row["operator"] == operator
            ]
            if (
                len(candidate_physical_rows) != 24
                or sum(
                    row["passband_id"] in primary_ids for row in candidate_physical_rows
                )
                != 12
                or sum(
                    row["passband_id"] in challenger_ids
                    for row in candidate_physical_rows
                )
                != 12
            ):
                raise RuntimeError(
                    f"G8 physical candidate coverage mismatch: {lane}/{operator}"
                )
            challenger_representation_pass = challenger_physical_pass and all(
                row["gate_pass"] == "true" for row in challenger_rows
            )
            candidate_physical_pass = primary_physical_pass and challenger_physical_pass
            if not challenger_representation_pass:
                challenger_status = "fail"
            elif truth_difference_max > FIDELITY_GATE:
                challenger_status = "owner_choice_required"
            else:
                challenger_status = "pass"
            eligible = (
                primary_pass
                and primary_physical_pass
                and challenger_representation_pass
                and challenger_status == "pass"
            )
            primary_worst = max(
                primary_rows,
                key=lambda row: float(row["max_absolute_fractional_correction_error"]),
            )
            primary_max = max(
                float(row["max_absolute_fractional_correction_error"])
                for row in primary_rows
            )
            challenger_gate_results: list[dict[str, Any]] = []
            for row in sorted(
                challenger_rows,
                key=lambda item: (item["array"], int(item["alpha"])),
            ):
                physical = next(
                    item
                    for item in candidate_physical_rows
                    if item["passband_id"] == row["passband_id"]
                    and item["alpha"] == row["alpha"]
                )
                gate_results = {
                    "G0_provenance_and_execution_context": g0_provenance_pass,
                    "G1_anchor_and_clear_segment_identity": (
                        physical["exact_anchor_pass"] == "true"
                        and physical["exact_low_segment_pass"] == "true"
                    ),
                    "G2_finite_domain_and_positivity": (
                        physical["g2_domain_pass"] == "true"
                    ),
                    "G3_continuity": physical["continuity_pass"] == "true",
                    "G4_physical_monotonicity": (
                        physical["tau_monotonicity_pass"] == "true"
                        and physical["elevation_monotonicity_pass"] == "true"
                    ),
                    "G5_fail_closed_support": (physical["fail_closed_pass"] == "true"),
                    "G6_representation_fidelity": row["gate_pass"] == "true",
                    "G8_evidence_coverage": bool(coverage["pass"]),
                }
                challenger_gate_results.append(
                    {
                        "passband_id": row["passband_id"],
                        "array": row["array"],
                        "alpha": row["alpha"],
                        "max_absolute_fractional_correction_error": row[
                            "max_absolute_fractional_correction_error"
                        ],
                        "gate_results": gate_results,
                        "all_G0_through_G6_and_G8_pass": all(gate_results.values()),
                    }
                )
            candidate_decisions.append(
                {
                    "lane": lane,
                    "operator": operator,
                    "primary_representation_pass": primary_pass,
                    "primary_physical_pass": primary_physical_pass,
                    "challenger_representation_pass": challenger_representation_pass,
                    "challenger_physical_pass": challenger_physical_pass,
                    "physical_contract_pass": candidate_physical_pass,
                    "challenger_status": challenger_status,
                    "challenger_gate_results": challenger_gate_results,
                    "fts_vs_ecsv_truth_max_absolute_fractional_difference": f17(
                        truth_difference_max
                    ),
                    "primary_max_absolute_fractional_correction_error": f17(
                        primary_max
                    ),
                    "primary_worst_location": {
                        "band": primary_worst["array"],
                        "alpha": primary_worst["alpha"],
                        "interval": primary_worst["worst_holdout_kind"],
                        "profile": primary_worst["worst_truth_profile"],
                        "h2o_scale_decimal": primary_worst["worst_h2o_scale_decimal"],
                        "elevation_deg": primary_worst["worst_elevation_deg"],
                    },
                    "eligible": eligible,
                }
            )

    def ranked_candidates(
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        remaining = list(candidates)
        ranked = []
        while remaining:
            best_value = min(
                float(item["primary_max_absolute_fractional_correction_error"])
                for item in remaining
            )
            tied = [
                item
                for item in remaining
                if float(item["primary_max_absolute_fractional_correction_error"])
                <= best_value + 1.0e-4
            ]
            tied.sort(
                key=lambda item: (
                    item["lane"] != "fixed_djf25_v1",
                    item["operator"] != "am12_piecewise_linear_los_tau_eval_v0",
                    float(item["primary_max_absolute_fractional_correction_error"]),
                )
            )
            ranked.extend(tied)
            selected_ids = {(item["lane"], item["operator"]) for item in tied}
            remaining = [
                item
                for item in remaining
                if (item["lane"], item["operator"]) not in selected_ids
            ]
        return ranked

    conditionally_rankable = [
        item
        for item in candidate_decisions
        if item["primary_representation_pass"]
        and item["primary_physical_pass"]
        and item["challenger_representation_pass"]
    ]
    conditional_ranking = [
        {
            "rank": index,
            "lane": item["lane"],
            "operator": item["operator"],
            "primary_max_absolute_fractional_correction_error": item[
                "primary_max_absolute_fractional_correction_error"
            ],
            "challenger_status": item["challenger_status"],
        }
        for index, item in enumerate(ranked_candidates(conditionally_rankable), start=1)
    ]
    eligible = [item for item in candidate_decisions if item["eligible"]]
    recommendation: dict[str, Any] | None = None
    if eligible:
        winner = ranked_candidates(eligible)[0]
        lane = winner["lane"]
        operator = winner["operator"]
        primary_passbands = sorted(
            (item for item in bandpasses if item.family == "primary_tolteca_ecsv"),
            key=lambda item: item.array,
        )
        recommendation = {
            "model_recipe": {
                "lane_identity": lane,
                "clear_anchor": "analytic_q0_zero_los_optical_depth",
                "nonzero_anchors": {
                    target: {
                        "source_profile": LANES[lane][target],
                        "h2o_scale_decimal": p1.scale(target, LANES[lane][target]),
                        "tau225_selector_anchor_binary64": (
                            FROZEN_TARGET_COORDINATES[target][0]
                        ),
                        "reference_225ghz_transmission_literal": (
                            FROZEN_TARGET_COORDINATES[target][1]
                        ),
                    }
                    for target in TARGETS
                },
            },
            "operator": {
                "identity": operator,
                "elevation_representation": (
                    "shape_preserving_PCHIP_in_line_of_sight_optical_depth_"
                    "at_each_nonzero_anchor"
                ),
                "q0_q25_representation": (
                    "analytic_linear_in_line_of_sight_optical_depth"
                ),
                "q25_q75_representation": (
                    "piecewise_linear_in_tau225"
                    if operator == "am12_piecewise_linear_los_tau_eval_v0"
                    else "shape_preserving_PCHIP_in_tau225"
                ),
            },
            "primary_passband_contract": {
                "family": "primary_tolteca_ecsv",
                "integration_convention": (
                    "energy-weighted supplied ECSV throughput at frozen TolTECA commit"
                ),
                "passbands": [
                    {
                        "id": item.identity,
                        "array": item.array,
                        "source_path": item.source_path,
                        "source_sha256": item.source_sha256,
                        "source_commit": item.source_commit,
                        "response_convention": item.convention,
                    }
                    for item in primary_passbands
                ],
            },
            "spectral_contract": {
                "alpha_values": list(ALPHAS),
                "source_spectrum": (
                    "top-of-atmosphere S_nu proportional to (nu/pivot_frequency)^alpha"
                ),
            },
            "operational_domain": {
                "tau225_min_inclusive": FROZEN_TARGET_COORDINATES["am_q0"][0],
                "tau225_max_inclusive_q75": FROZEN_TARGET_COORDINATES["am_q75"][0],
                "elevation_min_deg_inclusive": f17(20.0),
                "elevation_max_deg_inclusive": f17(80.0),
                "outside_domain_policy": "fail_closed_explicit_invalid",
            },
        }
        status = "provisional_numerical_adoption_evidence_pass"
    elif any(
        item["primary_representation_pass"]
        and item["primary_physical_pass"]
        and item["challenger_representation_pass"]
        and item["challenger_status"] == "owner_choice_required"
        for item in candidate_decisions
    ):
        status = "owner_passband_choice_required"
    else:
        status = "numerical_adoption_evidence_fail"
    decision = {
        "schema_version": f"{SCHEMA_VERSION}-decision-v1",
        "status": status,
        "G0_provenance_and_execution_context_pass": g0_provenance_pass,
        "candidate_decisions": candidate_decisions,
        "truth_challenger": {
            "maximum_absolute_fractional_difference": f17(truth_difference_max),
            "individual_band_alpha_results": [
                {
                    "array": row["array"],
                    "alpha": row["alpha"],
                    "maximum_absolute_fractional_difference": row[
                        "max_absolute_fractional_correction_error"
                    ],
                    "within_one_percent": (
                        float(row["max_absolute_fractional_correction_error"])
                        <= FIDELITY_GATE
                    ),
                    "location": {
                        "interval": row["worst_holdout_kind"],
                        "profile": row["worst_truth_profile"],
                        "h2o_scale_decimal": row["worst_h2o_scale_decimal"],
                        "elevation_deg": row["worst_elevation_deg"],
                    },
                }
                for row in sorted(
                    truth_difference_rows,
                    key=lambda item: (item["array"], int(item["alpha"])),
                )
            ],
            "location": {
                "array": truth_difference_location["array"],
                "alpha": truth_difference_location["alpha"],
                "interval": truth_difference_location["worst_holdout_kind"],
                "profile": truth_difference_location["worst_truth_profile"],
                "h2o_scale_decimal": truth_difference_location[
                    "worst_h2o_scale_decimal"
                ],
                "elevation_deg": truth_difference_location["worst_elevation_deg"],
            },
        },
        "tie_tolerance_fractional_correction": f17(1.0e-4),
        "conditional_primary_ranking": conditional_ranking,
        "recommendation": recommendation,
        "authorization": "none_owner_selection_required",
    }

    report_lines = [
        "# SCI-CAL-001 AM-12.2 successor adoption study",
        "",
        f"Status: **{status}**.",
        "",
        "This study is bounded to q0--q75 and elevations 20--80 degrees. It does not evaluate q95.",
        "The one-percent threshold is numerical representation fidelity, not observational photometric accuracy.",
        "",
        "## Frozen candidates",
        "",
        "- `fixed_djf25_v1`: `LMT_DJF_25` independently H2O-scaled to q25, q50, and q75.",
        "- `conditioned_djf_v1`: `LMT_DJF_25@q25`, `LMT_DJF_50@q50`, and `LMT_DJF_75@q75`.",
        "- Shape-preserving PCHIP in line-of-sight optical depth versus elevation for every nonzero anchor; analytic linear q0--q25 followed by either piecewise-linear or PCHIP opacity interpolation.",
        "- TolTECA v1 ECSV passbands are primary. FTS spectra are challengers, not replacements.",
        "- Source spectra use `S_nu` proportional to `nu^alpha` for alpha -1, 0, 2, and 4.",
        "",
        "## Evidence",
        "",
        f"- Canonical P1 direct grids validated and integrated: {len(run_inventory)}.",
        "- Frozen training grids: five unique target/profile constructions at 31 even elevations.",
        f"- Independent band-integrated candidate holdout rows: {len(holdout_rows)} from all 240 direct AM truth grids.",
        f"- Digest-bound holdout run inventory: {len(holdout_run_rows)} total ({scale_search_inventory_count} scale-search anchors plus 240 full grids).",
        "- G8 expanded-row key coverage: 23,040/23,040 with zero missing, unexpected, or duplicate keys.",
        f"- Positivity, opacity/elevation monotonicity, continuity, fail-closed support, low-segment identity, and exact-anchor contract: {'PASS' if physical_pass else 'FAIL'}.",
        f"- G7 challenger disposition maximum direct-truth difference: {truth_difference_max:.6%}.",
        f"- Machine decision status: `{status}`.",
        "",
        "## Interpretation boundary",
        "",
        "FTS-versus-primary sensitivity follows the frozen three-state G7 disposition and is not charged to interpolation error. A numerical recommendation is not owner selection or observational authorization; calibrator repeatability and absolute-flux gates remain separate.",
        "",
    ]
    report_lines.extend(
        [
            "## Machine recommendation and conditional primary ranking",
            "",
            "```json",
            json.dumps(
                {
                    "conditional_primary_ranking": conditional_ranking,
                    "recommendation": recommendation,
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    report = "\n".join(report_lines).encode("utf-8")

    artifacts: dict[str, bytes] = {
        OUTPUT_NAMES["bandpasses"]: csv_bytes(
            list(bandpass_rows[0]),
            sorted(
                bandpass_rows,
                key=lambda row: (row["array"], row["family"], int(row["alpha"])),
            ),
        ),
        OUTPUT_NAMES["nodes"]: csv_bytes(
            list(node_rows[0]),
            sorted(
                node_rows,
                key=lambda row: (
                    row["lane"],
                    row["passband_id"],
                    int(row["alpha"]),
                    TARGET_ORDER[row["target"]],
                    int(row["elevation_deg"]),
                ),
            ),
        ),
        OUTPUT_NAMES["metrics"]: csv_bytes(metric_fields, metrics),
        OUTPUT_NAMES["physical"]: csv_bytes(
            list(physical_rows[0]),
            sorted(
                physical_rows,
                key=lambda row: (
                    row["lane"],
                    row["operator"],
                    row["passband_id"],
                    int(row["alpha"]),
                ),
            ),
        ),
        OUTPUT_NAMES["p1_runs"]: csv_bytes(
            list(run_inventory[0]),
            sorted(
                run_inventory,
                key=lambda row: (
                    TARGET_ORDER[row["target"]],
                    row["profile"],
                    int(row["elevation_deg"]),
                ),
            ),
        ),
        OUTPUT_NAMES["holdout_runs"]: csv_bytes(
            list(holdout_run_rows[0]),
            sorted(
                holdout_run_rows,
                key=lambda row: (
                    row["holdout_kind"],
                    row["truth_profile"],
                    int(row["elevation_deg"]),
                ),
            ),
        ),
        OUTPUT_NAMES["execution_context"]: json_bytes(holdout_execution_context),
        OUTPUT_NAMES["holdouts"]: csv_bytes(
            [
                "holdout_kind",
                "truth_profile",
                "requested_tau225",
                "achieved_tau225",
                "coordinate_residual",
                "h2o_scale_decimal",
                "h2o_scale_hex",
                "analytic_transmission_decimal",
                "target_transmission_literal",
                "achieved_transmission_el80",
                "elevation_deg",
                "airmass",
                "lane",
                "operator",
                "passband_id",
                "array",
                "alpha",
                "direct_extinction_correction",
                "operator_extinction_correction",
                "fractional_correction_error",
                "raw_sha256",
                "sidecar_sha256",
            ],
            sorted(
                holdout_rows,
                key=lambda row: (
                    row["holdout_kind"],
                    row["truth_profile"],
                    int(row["elevation_deg"]),
                    row["lane"],
                    row["operator"],
                    row["passband_id"],
                    int(row["alpha"]),
                ),
            ),
        ),
        OUTPUT_NAMES["scales"]: csv_bytes(
            list(scale_rows[0]),
            sorted(scale_rows, key=lambda row: (row["holdout_kind"], row["profile"])),
        ),
        OUTPUT_NAMES["coverage"]: json_bytes(coverage),
        OUTPUT_NAMES["report"]: report,
        OUTPUT_NAMES["decision"]: json_bytes(decision),
    }
    summary = {
        "status": status,
        "physical_contract_pass": physical_pass,
        "primary_holdout_fidelity_pass": all(
            item["primary_representation_pass"] for item in candidate_decisions
        ),
        "challenger_statuses": sorted(
            {item["challenger_status"] for item in candidate_decisions}
        ),
        "decision": decision,
        "p1_direct_grid_count": len(run_inventory),
        "holdout_direct_grid_count": len(holdout_records),
        "holdout_scale_search_run_count": scale_search_inventory_count,
        "holdout_total_run_count": len(holdout_run_rows),
        "holdout_metric_row_count": len(holdout_rows),
    }
    return artifacts, summary


def c1_midpoint_quantization() -> dict[str, dict[str, Decimal | str]]:
    result: dict[str, dict[str, Decimal | str]] = {}
    with localcontext() as context:
        context.prec = 80
        x80 = Decimal(C1_X80)
        half_step = Decimal("5e-8")
        comparison_tolerance = Decimal("5e-36")
        for kind, frozen in C1_INTERVALS.items():
            tau_mid = Decimal(frozen["tau_mid"])
            analytic = (-tau_mid * x80).exp()
            literal = format(analytic, ".6e")
            if Decimal(literal) != Decimal(frozen["literal"]):
                raise RuntimeError(f"C1 display literal mismatch: {kind}/{literal}")
            represented = Decimal(frozen["literal"])
            achieved = -(represented.ln()) / x80
            residual = achieved - tau_mid
            lower = ((represented + half_step) / represented).ln() / x80
            upper = (represented / (represented - half_step)).ln() / x80
            checks = {
                "t_analytic": analytic,
                "tau_achieved": achieved,
                "residual": residual,
                "lower_bound": lower,
                "upper_bound": upper,
            }
            for name, actual in checks.items():
                if abs(actual - Decimal(frozen[name])) > comparison_tolerance:
                    raise RuntimeError(f"C1 frozen {name} mismatch: {kind}")
            bound = max(lower, upper)
            if abs(residual) > bound:
                raise RuntimeError(f"C1 asymmetric propagated bound failed: {kind}")
            result[kind] = {
                "tau_mid": tau_mid,
                "t_analytic": analytic,
                "literal": frozen["literal"],
                "tau_achieved": achieved,
                "residual": residual,
                "lower_bound": lower,
                "upper_bound": upper,
                "acceptance_bound": bound,
            }
    return result


def holdout_plan(tau: dict[str, float]) -> list[dict[str, Any]]:
    del tau  # coordinates are frozen and independently checked by C1
    c1 = c1_midpoint_quantization()
    return [
        {
            "kind": kind,
            "profile": profile,
            "tau_mid": str(c1[kind]["tau_mid"]),
            "tau_achieved": str(c1[kind]["tau_achieved"]),
            "t_analytic": str(c1[kind]["t_analytic"]),
            "parsed_target_transmission_literal": c1[kind]["literal"],
            "acceptance_bound": str(c1[kind]["acceptance_bound"]),
            "elevations_deg": [int(value) for value in ELEVATIONS_ODD_DEG],
        }
        for kind, profile, lower, upper in HOLDOUT_CASES
    ]


def execution_host() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }


def holdout_execution_context(
    *,
    p1: P1Cache,
    bandpasses: Sequence[Bandpass],
    executable: Path,
    jobs: int,
    omp_threads: int,
) -> dict[str, Any]:
    protocol = PACKAGE_DIR / "AM12_SUCCESSOR_ADOPTION_STUDY_PROTOCOL.md"
    clarification = (
        PACKAGE_DIR / "AM12_SUCCESSOR_ADOPTION_STUDY_PREEXECUTION_CLARIFICATIONS.md"
    )
    erratum = (
        PACKAGE_DIR / "AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_ERRATUM_2026-08-01.md"
    )
    if sha256_path(protocol) != STUDY_PROTOCOL_SHA256:
        raise RuntimeError("frozen adoption-study protocol digest mismatch")
    if sha256_path(clarification) != PREEXECUTION_CLARIFICATION_SHA256:
        raise RuntimeError("frozen pre-execution clarification digest mismatch")
    if sha256_path(erratum) != EXECUTION_ERRATUM_SHA256:
        raise RuntimeError("frozen execution erratum digest mismatch")
    imported_runner = Path(p1_driver.__file__).resolve()
    imported_runner_sha = sha256_path(imported_runner)
    if imported_runner_sha != p1.context["runner"]["sha256"]:
        raise RuntimeError("holdout solver does not match canonical P1 runner")
    if p1_driver.ROOT_ITERATIONS != 48 or p1_driver.MAX_BRACKET_EXPANSIONS != 64:
        raise RuntimeError("holdout solver iteration constants are not frozen")
    profile_root = p1.am_root / "Big_Atmosphere/LMT_am_inputs"
    profiles = []
    for profile, expected in sorted(HOLDOUT_PROFILE_SHA256.items()):
        path = profile_root / f"{profile}.amc"
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"frozen holdout AMC digest mismatch: {path}")
        profiles.append({"profile": profile, "sha256": actual})
    executable_sha = sha256_path(executable)
    if executable_sha != EXPECTED_AM_EXECUTABLE_SHA256:
        raise RuntimeError("frozen native AM executable digest mismatch")
    tau, t225 = target_coordinates()
    plan = holdout_plan(tau)
    copied_npz_inputs = p1.copied_npz_inventory(
        [profile for _kind, profile, _lower, _upper in HOLDOUT_CASES]
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}-holdout-execution-context-v1",
        "runner": {
            "filename": Path(__file__).name,
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "protocol": {"filename": protocol.name, "sha256": sha256_path(protocol)},
        "preexecution_clarification": {
            "filename": clarification.name,
            "sha256": sha256_path(clarification),
        },
        "execution_erratum": {
            "filename": erratum.name,
            "sha256": sha256_path(erratum),
            "predecessor_cache_disposition": "excluded_not_reused",
            "predecessor_execution_context_sha256": (
                "f0acb32cd43fd0bd128a06ab8d7e354bc6a6c1389d6d0794db716753d03f85c8"
            ),
            "correction": (
                "derive_P1_cache_stage_from_frozen_"
                "ancillary_screening_transmission_rank1"
            ),
        },
        "imported_canonical_p1_runner": {
            "filename": imported_runner.name,
            "sha256": imported_runner_sha,
            "canonical_p1_context_sha256": p1.context["runner"]["sha256"],
        },
        "scale_solver_contract": {
            "root_iterations": p1_driver.ROOT_ITERATIONS,
            "maximum_bracket_expansions": (p1_driver.MAX_BRACKET_EXPANSIONS),
            "identity_match_pass": True,
        },
        "p1_execution_context_sha256": p1.context_sha256,
        "p1_committed_evidence": {
            "manifest": {
                "path_relative_to_package": p1.manifest_path.name,
                "sha256": p1.manifest_sha256,
            },
            "scale_table": {
                "path_relative_to_package": p1.scales_path.name,
                "sha256": p1.scales_sha256,
            },
        },
        "copied_scale1_npz_inputs": copied_npz_inputs,
        "coordinate_source": {
            "path_relative_to_package": "legacy_anchor_manifest.json",
            "sha256": LEGACY_ANCHOR_MANIFEST_SHA256,
            "q0_q75_targets": {
                model: {
                    "tau225_selector_anchor_binary64": f17(tau[model]),
                    "reference_225ghz_transmission_literal": t225[model],
                }
                for model in ("am_q0", *TARGETS)
            },
        },
        "execution_host": execution_host(),
        "am_executable": {
            "resolved_path": str(executable),
            "size_bytes": executable.stat().st_size,
            "sha256": executable_sha,
            "binary_format": "mach-o",
        },
        "profiles": profiles,
        "passbands": [
            {
                "id": item.identity,
                "sha256": item.source_sha256,
                "source_commit": item.source_commit,
            }
            for item in bandpasses
        ],
        "execution_parameters": {
            "jobs": jobs,
            "omp_threads_per_process": omp_threads,
            "locale": PINNED_LOCALE,
            "cache_shard_count": jobs,
            "root_iterations": 48,
            "maximum_bracket_expansions": 64,
            "grid": "0--500 GHz inclusive at 10 MHz",
            "elevations_deg": [int(value) for value in ELEVATIONS_ODD_DEG],
        },
        "holdout_plan": plan,
        "security": {"network_access": False, "unity_access": False},
    }


@contextmanager
def holdout_cache_lock(cache_dir: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = cache_dir / ".am12_successor_adoption.lock"
    if not exclusive and not lock_path.is_file():
        raise RuntimeError(f"missing holdout cache lock: {lock_path}")
    open_mode = "a+b" if exclusive else "rb"
    with lock_path.open(open_mode) as handle:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("holdout cache is already locked") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_holdout_run_inventory(
    *,
    runner: p1_driver.Runner,
    records: Sequence[HoldoutRecord],
    solutions: Sequence[p1_driver.ScaleSolution],
) -> list[dict[str, str]]:
    trace_runs: dict[str, dict[str, str]] = {}
    trace_by_case: dict[tuple[str, str], dict[str, str]] = {}
    for solution in solutions:
        trace_path = runner.cache_dir / solution.trace_relative_path
        if sha256_path(trace_path) != solution.trace_sha256:
            raise RuntimeError(f"scale-trace digest mismatch: {trace_path}")
        trace = canonical_json(trace_path)
        if (
            trace.get("execution_context_sha256") != runner.execution_context_sha256
            or trace.get("target") != solution.target
            or trace.get("profile") != solution.profile
            or trace.get("root_iterations") != 48
            or trace.get("maximum_bracket_expansions") != 64
            or len(trace.get("evaluations", [])) != solution.trace_evaluation_count
        ):
            raise RuntimeError(
                f"scale-trace identity mismatch: {solution.target}/{solution.profile}"
            )
        trace_meta = {
            "scale_trace_path_relative_to_holdout_cache": (
                solution.trace_relative_path
            ),
            "scale_trace_sha256": solution.trace_sha256,
        }
        case = (solution.target, solution.profile)
        if case in trace_by_case:
            raise RuntimeError(f"duplicate scale solution: {case}")
        trace_by_case[case] = trace_meta
        for expected_index, evaluation in enumerate(trace["evaluations"]):
            if evaluation.get("evaluation_index") != expected_index:
                raise RuntimeError(f"nonsequential scale trace: {trace_path}")
            spec = p1_driver.anchor_spec(
                solution.profile,
                solution.target,
                evaluation["scale_decimal"],
            )
            cache_id = runner.cache_id(spec)
            if cache_id in trace_runs:
                raise RuntimeError(f"duplicate trace cache identity: {cache_id}")
            trace_runs[cache_id] = {
                **trace_meta,
                "scale_trace_evaluation_index": str(expected_index),
                "scale_trace_role": evaluation["role"],
            }

    full_run_ids = {item.cache_id for item in records}
    if len(full_run_ids) != 240:
        raise RuntimeError(
            f"full-grid inventory identity mismatch: {len(full_run_ids)}/240"
        )
    if full_run_ids & set(trace_runs):
        raise RuntimeError("anchor and full-grid cache identities overlap")
    expected_ids = set(trace_runs) | full_run_ids
    observed = {item.cache_id: item for item in runner.observed_runs()}
    if set(observed) != expected_ids:
        missing = sorted(expected_ids - set(observed))
        unexpected = sorted(set(observed) - expected_ids)
        raise RuntimeError(
            "holdout Runner observation/trace anti-join failed: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    actual_raw_ids = {
        path.name.removesuffix(".txt")
        for path in (runner.cache_dir / "raw_outputs").glob("*.txt")
    }
    actual_sidecar_ids = {
        path.name.removesuffix(".run.json")
        for path in (runner.cache_dir / "execution_records").glob("*.run.json")
    }
    failed_attempt_files = sorted(
        path.relative_to(runner.cache_dir).as_posix()
        for path in (runner.cache_dir / "failed_attempts").iterdir()
        if path.is_file()
    )
    if (
        actual_raw_ids != expected_ids
        or actual_sidecar_ids != expected_ids
        or failed_attempt_files
    ):
        raise RuntimeError(
            "holdout cache file anti-join failed: "
            f"raw_missing={len(expected_ids - actual_raw_ids)}, "
            f"raw_unexpected={len(actual_raw_ids - expected_ids)}, "
            f"sidecar_missing={len(expected_ids - actual_sidecar_ids)}, "
            f"sidecar_unexpected={len(actual_sidecar_ids - expected_ids)}, "
            f"failed_attempt_files={failed_attempt_files[:3]}"
        )

    inventory: list[dict[str, str]] = []
    for cache_id in sorted(expected_ids):
        observation = observed[cache_id]
        # Re-entering cache-only validates the complete canonical sidecar core:
        # exact cache-id/filename, request, argv, executable, profile, cwd role,
        # locale, OMP, host, shard, output path/digest, diagnostics, and context.
        result = runner.run_or_load(observation.spec)
        raw_path = runner.raw_path(cache_id)
        sidecar_path = runner.sidecar_path(cache_id)
        if result.cache_id != cache_id:
            raise RuntimeError(f"revalidated cache identity changed: {cache_id}")
        if set(result.sidecar) != CANONICAL_RUN_SIDECAR_KEYS:
            raise RuntimeError(f"holdout sidecar key-set mismatch: {cache_id}")
        if cache_id in trace_runs:
            run_class = "midpoint_scale_search_anchor"
            trace_meta = trace_runs[cache_id]
        else:
            run_class = "midpoint_odd_elevation_full_grid"
            case = (observation.spec.target, observation.spec.profile)
            if case not in trace_by_case:
                raise RuntimeError(f"full grid has no scale trace: {cache_id}")
            trace_meta = {
                **trace_by_case[case],
                "scale_trace_evaluation_index": "",
                "scale_trace_role": "fitted_scale_full_grid_truth",
            }
        sidecar = result.sidecar
        inventory.append(
            {
                "run_class": run_class,
                "holdout_kind": observation.spec.target,
                "truth_profile": observation.spec.profile,
                "cache_id": cache_id,
                "stage": observation.spec.stage,
                "scale_decimal": observation.spec.scale_decimal,
                "elevation_deg": str(observation.spec.elevation_deg),
                "zenith_angle_deg": str(observation.spec.zenith_angle_deg),
                "frequency_min_centi_ghz": str(observation.spec.f_min_centi_ghz),
                "frequency_max_centi_ghz": str(observation.spec.f_max_centi_ghz),
                "argv_json": json.dumps(
                    sidecar["argv"], sort_keys=True, separators=(",", ":")
                ),
                "working_directory_role": sidecar["working_directory_role"],
                "profile_sha256": sidecar["profile_sha256"],
                "am_executable_sha256": sidecar["am_executable_sha256"],
                "omp_threads": str(sidecar["omp_threads"]),
                "locale_json": json.dumps(
                    sidecar["locale"], sort_keys=True, separators=(",", ":")
                ),
                "execution_host_json": json.dumps(
                    sidecar["execution_host"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "execution_context_sha256": sidecar["execution_context_sha256"],
                "am_cache_shard_index": str(sidecar["am_cache_shard_index"]),
                "am_cache_shard_count": str(sidecar["am_cache_shard_count"]),
                "raw_path_relative_to_holdout_cache": sidecar[
                    "combined_output_path_relative_to_cache"
                ],
                "raw_sha256": result.raw_sha256,
                "sidecar_path_relative_to_holdout_cache": (
                    sidecar_path.relative_to(runner.cache_dir).as_posix()
                ),
                "sidecar_sha256": sha256_path(sidecar_path),
                "return_code": str(result.return_code),
                "am_version_identity": result.parsed.version_identity,
                "numeric_row_count": str(result.parsed.samples.shape[0]),
                "unresolved_line_warning_count": (
                    ""
                    if result.parsed.warning_count is None
                    else str(result.parsed.warning_count)
                ),
                **trace_meta,
            }
        )
        if sha256_path(raw_path) != result.raw_sha256:
            raise RuntimeError(f"raw inventory digest changed: {raw_path}")
    if len(inventory) != len(expected_ids):
        raise RuntimeError("holdout run inventory cardinality mismatch")
    return inventory


def load_or_run_holdouts(
    *,
    cache_dir: Path,
    p1: P1Cache,
    bandpasses: Sequence[Bandpass],
    executable: Path,
    jobs: int,
    omp_threads: int,
    execute: bool,
) -> tuple[list[HoldoutRecord], list[dict[str, str]], dict[str, Any]]:
    context = holdout_execution_context(
        p1=p1,
        bandpasses=bandpasses,
        executable=executable,
        jobs=jobs,
        omp_threads=omp_threads,
    )
    context_bytes = json_bytes(context)
    context_digest = sha256_bytes(context_bytes)
    context_path = cache_dir / "execution_context.json"
    if execute:
        cache_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            path.name
            for path in cache_dir.iterdir()
            if path.name != ".am12_successor_adoption.lock"
        ]
        if existing:
            raise RuntimeError(
                f"--run-holdouts requires a fresh cache; found {sorted(existing)}"
            )
        for directory in (
            "raw_outputs",
            "execution_records",
            "scale_traces",
            "failed_attempts",
        ):
            (cache_dir / directory).mkdir(parents=True, exist_ok=True)
        atomic_write(context_path, context_bytes)
    elif not context_path.is_file() or context_path.read_bytes() != context_bytes:
        raise RuntimeError("holdout cache execution context mismatch")

    build = p1_driver.BuildIdentity(
        supplied_path=str(executable),
        resolved_path=str(executable),
        size_bytes=executable.stat().st_size,
        sha256=EXPECTED_AM_EXECUTABLE_SHA256,
        binary_format="mach-o",
    )
    runner = p1_driver.Runner(
        executable=build,
        am_root=p1.am_root,
        cache_dir=cache_dir,
        omp_threads=omp_threads,
        cache_shard_count=jobs,
        execution_host=context["execution_host"],
        execution_context_sha256=context_digest,
        execute=execute,
    )

    c1 = c1_midpoint_quantization()
    records: list[HoldoutRecord] = []
    solutions: list[p1_driver.ScaleSolution] = []
    for kind, profile, lower, upper in HOLDOUT_CASES:
        del lower, upper
        requested_tau = float(c1[kind]["tau_mid"])
        achieved_tau = float(c1[kind]["tau_achieved"])
        target_literal = str(c1[kind]["literal"])
        target_key = kind
        p1_driver.EXPECTED_TARGET_TRANSMISSIONS[target_key] = target_literal
        scale0 = runner.run_or_load(
            p1_driver.anchor_spec(profile, target_key, f17(0.0))
        )
        scale1 = runner.run_or_load(
            p1_driver.anchor_spec(profile, target_key, f17(1.0))
        )
        copied_tau, copied_tx = p1_driver.copied_anchor(p1.am_root, profile)
        solution = p1_driver.solve_scale_hypothesis(
            runner=runner,
            profile=profile,
            target=target_key,
            scale0=scale0,
            scale1=scale1,
            copied_scale1_tau=copied_tau,
            copied_scale1_transmission=copied_tx,
        )
        solutions.append(solution)
        achieved_tx = p1_driver.anchor_values(solution.fitted)[1]
        if achieved_tx != float(target_literal):
            raise RuntimeError(f"midpoint parsed T225 mismatch: {kind}/{profile}")
        achieved_from_run = -math.log(achieved_tx) / float(Decimal(C1_X80))
        if abs(achieved_from_run - achieved_tau) > 5.0e-17:
            raise RuntimeError(f"C1 achieved coordinate mismatch: {kind}/{profile}")
        coordinate_tolerance = float(c1[kind]["acceptance_bound"])
        if abs(achieved_tau - requested_tau) > coordinate_tolerance:
            raise RuntimeError(
                f"midpoint coordinate residual too large: {kind}/{profile}"
            )
        specs = [
            p1_driver.full_grid_spec(
                "adoption_midpoint_odd_elevation_holdout",
                profile,
                target_key,
                90 - int(elevation),
                solution.scale_decimal,
            )
            for elevation in ELEVATIONS_ODD_DEG.astype(int)
        ]
        runs = runner.run_many(specs, jobs)
        for run in runs:
            raw_path = runner.raw_path(run.cache_id)
            sidecar_path = runner.sidecar_path(run.cache_id)
            raw = raw_path.read_bytes()
            parsed = parse_am_output(
                raw, f"{kind}/{profile}/EL{run.spec.elevation_deg}"
            )
            records.append(
                HoldoutRecord(
                    kind=kind,
                    profile=profile,
                    requested_tau225=requested_tau,
                    achieved_tau225=achieved_tau,
                    elevation_deg=run.spec.elevation_deg,
                    scale_decimal=solution.scale_decimal,
                    scale_hex=float(solution.scale_decimal).hex(),
                    analytic_transmission_decimal=str(c1[kind]["t_analytic"]),
                    target_transmission_literal=target_literal,
                    achieved_transmission_el80=achieved_tx,
                    plateau_lower_outside_scale=solution.plateau_lower_outside_scale,
                    plateau_lower_inside_scale=solution.plateau_lower_inside_scale,
                    plateau_upper_inside_scale=solution.plateau_upper_inside_scale,
                    plateau_upper_outside_scale=solution.plateau_upper_outside_scale,
                    lower_tau_half_step=str(c1[kind]["lower_bound"]),
                    upper_tau_half_step=str(c1[kind]["upper_bound"]),
                    coordinate_acceptance_bound=str(c1[kind]["acceptance_bound"]),
                    trace_relative_path=solution.trace_relative_path,
                    trace_sha256=solution.trace_sha256,
                    parsed=parsed,
                    raw_relative_path=raw_path.relative_to(cache_dir).as_posix(),
                    raw_sha256=sha256_path(raw_path),
                    sidecar_relative_path=sidecar_path.relative_to(
                        cache_dir
                    ).as_posix(),
                    sidecar_sha256=sha256_path(sidecar_path),
                    return_code=run.return_code,
                    cache_id=run.cache_id,
                )
            )
    expected_keys = {
        (kind, profile, int(elevation))
        for kind, profile, _, _ in HOLDOUT_CASES
        for elevation in ELEVATIONS_ODD_DEG
    }
    actual_keys = {(item.kind, item.profile, item.elevation_deg) for item in records}
    if actual_keys != expected_keys or len(records) != 240:
        raise RuntimeError("G8 holdout coverage mismatch")
    ordered_records = sorted(
        records, key=lambda item: (item.kind, item.profile, item.elevation_deg)
    )
    inventory = build_holdout_run_inventory(
        runner=runner,
        records=ordered_records,
        solutions=solutions,
    )
    return ordered_records, inventory, context


def build_manifest(
    artifacts: dict[str, bytes],
    summary: dict[str, Any],
    p1: P1Cache,
    bandpasses: Sequence[Bandpass],
    holdout_cache: Path | None,
) -> bytes:
    coordinate_tau, coordinate_t225 = target_coordinates()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "package": "SCI-CAL-001",
            "scope": "q0_q75_only_no_q95",
            "evidence_status": summary["status"],
            "operator_authorization": "none_evidence_for_owner_adoption_decision",
        },
        "execution_erratum": {
            "filename": (
                "AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_ERRATUM_2026-08-01.md"
            ),
            "sha256": EXECUTION_ERRATUM_SHA256,
            "predecessor_cache_disposition": "excluded_not_reused",
            "predecessor_execution_context_sha256": (
                "f0acb32cd43fd0bd128a06ab8d7e354bc6a6c1389d6d0794db716753d03f85c8"
            ),
        },
        "domain": {
            "tau225_min": f17(0.0),
            "tau225_max_q75_selector_anchor": f17(target_coordinates()[0]["am_q75"]),
            "elevation_min_deg": f17(20.0),
            "elevation_max_deg": f17(80.0),
            "outside_domain_policy": "fail_closed",
        },
        "lanes": LANES,
        "operators": list(OPERATORS),
        "spectral_indices": list(ALPHAS),
        "coordinate_source": {
            "path_relative_to_package": "legacy_anchor_manifest.json",
            "sha256": LEGACY_ANCHOR_MANIFEST_SHA256,
            "q0_q75_targets": {
                model: {
                    "tau225_selector_anchor_binary64": f17(coordinate_tau[model]),
                    "reference_225ghz_transmission_literal": (coordinate_t225[model]),
                }
                for model in ("am_q0", *TARGETS)
            },
        },
        "bandpass_provenance": [
            {
                "id": item.identity,
                "array": item.array,
                "family": item.family,
                "source_path": item.source_path,
                "source_sha256": item.source_sha256,
                "source_commit": item.source_commit,
                "convention": item.convention,
                "clipped_negative_node_count": item.clipped_node_count,
                "clipped_negative_integral_fraction": f17(
                    item.clipped_negative_integral_fraction
                ),
            }
            for item in bandpasses
        ],
        "p1_cache": {
            "execution_context_sha256": p1.context_sha256,
            "direct_grid_count": summary["p1_direct_grid_count"],
            "external_cache_basename": p1.cache_dir.name,
            "cache_policy": "shared-lock cache-only validation; no process execution",
        },
        "holdouts": {
            "external_cache_basename": None
            if holdout_cache is None
            else holdout_cache.name,
            "metric_row_count": summary["holdout_metric_row_count"],
            "direct_grid_count": summary["holdout_direct_grid_count"],
            "scale_search_run_count": summary["holdout_scale_search_run_count"],
            "total_run_inventory_count": summary["holdout_total_run_count"],
            "execution_context_sha256": (
                None
                if holdout_cache is None
                else sha256_path(holdout_cache / "execution_context.json")
            ),
            "status": (
                "not_supplied"
                if summary["holdout_metric_row_count"] == 0
                else "cache_validated"
            ),
        },
        "gates": {
            "fractional_extinction_correction_representation_fidelity_max": f17(
                FIDELITY_GATE
            ),
            "criterion_interpretation": (
                "provisional numerical representation fidelity only; not "
                "per-sample or absolute physical photometric accuracy"
            ),
            "physical_contract_pass": summary["physical_contract_pass"],
            "primary_holdout_fidelity_pass": summary["primary_holdout_fidelity_pass"],
            "challenger_statuses": summary["challenger_statuses"],
        },
        "decision": summary["decision"],
        "artifacts": {
            name: {"size_bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(artifacts.items())
        },
        "security": {
            "network_access": False,
            "unity_access": False,
            "citlali_application_code_modified": False,
            "sibling_repositories_modified": False,
        },
        "limitations": [
            "P1 H2O-scale fits are post-hoc candidate recipes, not historical custody proof.",
            "FTS spectra are representative challengers and are not promoted over the TolTECA ECSV passbands.",
            "The eight frozen midpoint profile cases are independent representation holdouts, not observational validation.",
            "No q95 model, profile, target, or operational condition is evaluated.",
            "Observational absolute-flux and repeatability gates remain separate.",
        ],
    }
    return json_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="cache-only recomputation and byte-for-byte artifact check; write nothing",
    )
    mode.add_argument(
        "--regenerate-from-cache",
        action="store_true",
        help="cache-only validation and deterministic artifact rewrite; run no AM process",
    )
    mode.add_argument(
        "--run-holdouts",
        action="store_true",
        help=(
            "solve all eight midpoint scales, execute 240 odd-elevation AM "
            "holdouts in a fresh external cache, and write study artifacts"
        ),
    )
    parser.add_argument("--p1-cache-dir", type=Path, default=DEFAULT_P1_CACHE)
    parser.add_argument("--holdout-cache-dir", type=Path)
    parser.add_argument("--am-root", type=Path, default=DEFAULT_AM_ROOT)
    parser.add_argument("--am-executable", type=Path, default=DEFAULT_AM_EXECUTABLE)
    parser.add_argument("--tolteca-repo", type=Path, default=DEFAULT_TOLTECA_REPO)
    parser.add_argument("--beammap-repo", type=Path, default=DEFAULT_BEAMMAP_REPO)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p1_cache_dir = args.p1_cache_dir.expanduser().resolve()
    am_root = args.am_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tolteca_repo = args.tolteca_repo.expanduser().resolve()
    beammap_repo = args.beammap_repo.expanduser().resolve()
    executable = args.am_executable.expanduser().resolve()
    holdout_cache = (
        None
        if args.holdout_cache_dir is None
        else args.holdout_cache_dir.expanduser().resolve()
    )
    if holdout_cache is None:
        raise RuntimeError("all modes require --holdout-cache-dir")
    protected_roots = {
        "canonical P1 cache": p1_cache_dir,
        "copied AM root": am_root,
        "task package": PACKAGE_DIR,
        "repository": REPO_ROOT,
        "TolTECA input repository": tolteca_repo,
        "beammap input repository": beammap_repo,
        "native AM build root": executable.parent,
    }
    for label, protected in protected_roots.items():
        if paths_overlap(holdout_cache, protected):
            raise RuntimeError(
                f"holdout cache overlaps protected {label}: "
                f"{holdout_cache} / {protected}"
            )
    if not args.check and output_dir != PACKAGE_DIR:
        raise RuntimeError(
            "--run-holdouts and --regenerate-from-cache may write only to "
            f"the task package: {PACKAGE_DIR}"
        )
    if not p1_cache_dir.is_dir():
        raise RuntimeError(f"missing canonical P1 cache: {p1_cache_dir}")
    if not am_root.is_dir():
        raise RuntimeError(f"missing copied AM root: {am_root}")
    if args.jobs < 1 or args.omp_threads < 1:
        raise RuntimeError("--jobs and --omp-threads must be positive")
    p1 = P1Cache(p1_cache_dir, am_root)
    bandpasses = load_bandpasses(tolteca_repo, beammap_repo)
    if not executable.is_file():
        raise RuntimeError(f"missing native AM executable: {executable}")
    if args.run_holdouts:
        holdout_cache.mkdir(parents=True, exist_ok=True)
    elif not holdout_cache.is_dir():
        raise RuntimeError(f"missing holdout cache: {holdout_cache}")
    with holdout_cache_lock(holdout_cache, exclusive=args.run_holdouts):
        (
            holdout_records,
            holdout_run_rows,
            holdout_execution_context,
        ) = load_or_run_holdouts(
            cache_dir=holdout_cache,
            p1=p1,
            bandpasses=bandpasses,
            executable=executable,
            jobs=args.jobs,
            omp_threads=args.omp_threads,
            execute=args.run_holdouts,
        )
    with p1.shared_lock():
        artifacts, summary = build_study(
            p1,
            bandpasses,
            holdout_records,
            holdout_run_rows,
            holdout_execution_context,
        )
    manifest_bytes = build_manifest(artifacts, summary, p1, bandpasses, holdout_cache)
    artifacts[OUTPUT_NAMES["manifest"]] = manifest_bytes

    if args.check:
        for name, expected in sorted(artifacts.items()):
            path = output_dir / name
            if not path.is_file():
                raise RuntimeError(f"missing checked-in adoption artifact: {path}")
            actual = path.read_bytes()
            if actual != expected:
                raise RuntimeError(
                    f"adoption artifact differs from deterministic replay: {path}"
                )
        print(
            f"Validated {len(artifacts)} deterministic artifacts cache-only; "
            "no AM process executed."
        )
        return 0

    for name, data in sorted(artifacts.items()):
        atomic_write(output_dir / name, data)
    action = "with 240 newly executed holdouts" if args.run_holdouts else "cache-only"
    print(f"Wrote {len(artifacts)} deterministic artifacts {action}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
