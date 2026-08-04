#!/usr/bin/env python3
"""Execute the frozen SCI-CAL-001 TAU025 direct-AM evidence request.

``--dry-run`` is deliberately read-only: it verifies every input, provenance
annotation, inventory, and the absent selected cache root, but never invokes
AM or creates that root.  ``--run`` is the only execution mode.  It repeats
preflight, atomically admits the exact fresh root, and stops at the first
input, cache, AM, or WARN-001 failure while preserving that failed attempt.

This is task-local validation evidence code, not Citlali or TolTECA code.  It
does not fit, evaluate, select, or implement an atmosphere operator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable

import probe_am12_h2o_scale_hypotheses as p1


PACKAGE_DIR = Path(__file__).resolve().parent
REQUEST_PATH = PACKAGE_DIR / "SCI_CAL_001_TAU025_ENGINEERING_EXTENSION_EXECUTION_REQUEST.md"
PROTOCOL_PATH = PACKAGE_DIR / "SCI_CAL_001_TAU025_ENGINEERING_EXTENSION_PROTOCOL.md"
COPIED_MANIFEST_PATH = PACKAGE_DIR / "copied_am_manifest.json"
AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
AM_EXECUTABLE = Path("/private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am")
TOLTECA_REPO = Path("/Users/gwilson/GitHub/tolteca")
TOLTECA_COMMIT = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
CACHE_ROOT = Path("/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_002_root")
CACHE_BASENAME = "sci_cal_001_tau025_engineering_extension_002_root"
FORENSIC_CACHE_ROOT = Path("/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_001_root")
RETRY_ROOT_AUTHORIZATION_ID = "CAL-ATM-D007-RETRY-ROOT-001"
RETRY_ROOT_AUTHORIZATION_SHA256 = "295b6ce5fdfae2d204364085bb808bfbf3bbb1ec50be0d28e73771d3e62525d8"
CACHE_LOCK_NAME = ".tau025-engineering.lock"
AM_SOURCE_SHA256 = "0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8"
AM_EXECUTABLE_SHA256 = "78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb"
AMC_AGGREGATE_SHA256 = "b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c"
PASSBAND_SET_SHA256 = "5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433"
PASSBAND_TOTAL_BYTES = 1_297_803
JOBS = 8
OMP_THREADS = 1
SHARDS = 8
MIN_FREE_BYTES = 12 * 1024**3
X80 = Decimal("1.01538872688246729")
ROOT_ITERATIONS = 48
MAX_BRACKET_EXPANSIONS = 64

# node id, role, requested tau, exact AM parsed T225 literal, printed derived tau
NODES = (
    ("tau015", "construction", ".15", "8.587235e-01", "0.1499999859125433062628881602402745"),
    ("tau01625", "heldout", ".1625", "8.478931e-01", "0.1625000436670042842458733986011408"),
    ("tau0175", "heldout", ".175", "8.371994e-01", "0.1749999782159755418032064132046966"),
    ("tau01875", "heldout", ".1875", "8.266405e-01", "0.1874999959892568741794020809655989"),
    ("tau020", "construction", ".20", "8.162148e-01", "0.1999999783213567867059666712638576"),
    ("tau02125", "heldout", ".2125", "8.059206e-01", "0.2124999488193856859985890648134455"),
    ("tau0225", "heldout", ".225", "7.957562e-01", "0.2249999585478593390938136819948858"),
    ("tau02375", "heldout", ".2375", "7.857200e-01", "0.2374999620652431274454965339427345"),
    ("tau025", "construction", ".25", "7.758104e-01", "0.2499999377860148032413478624431719"),
)
CONSTRUCTION_ELEVATIONS = (25, 35, 45, 55, 65, 75, 80)
HOLDOUT_ELEVATIONS = (29, 41, 53, 67, 79)


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class Node:
    node_id: str
    role: str
    requested_tau: str
    target_literal: str
    printed_achieved_tau: str


@dataclass(frozen=True)
class FullRun:
    run_id: str
    profile: str
    node: Node
    elevation_deg: int


def nodes() -> tuple[Node, ...]:
    return tuple(Node(*row) for row in NODES)


def profile_inventory() -> list[dict[str, Any]]:
    manifest = json.loads(COPIED_MANIFEST_PATH.read_text())["amc_inputs"]
    require(sha256_path(COPIED_MANIFEST_PATH) == "714ad24329e625625da281d3b31ac2d28d04ab3c516980d516cff5ddadb027a9", "copied AMC manifest digest mismatch")
    require(manifest["file_count"] == 25, "unexpected AMC profile count")
    require(manifest["canonical_nul_records"]["sha256"] == AMC_AGGREGATE_SHA256, "AMC aggregate mismatch")
    rows = sorted(manifest["files"], key=lambda item: item["filename"])
    root = AM_ROOT / "Big_Atmosphere/LMT_am_inputs"
    for row in rows:
        path = root / row["filename"]
        require(path.is_file(), f"missing AMC profile: {path}")
        require(path.stat().st_size == row["bytes"], f"AMC size mismatch: {path.name}")
        require(sha256_path(path) == row["sha256"], f"AMC digest mismatch: {path.name}")
    return rows


def full_inventory(profiles: Iterable[dict[str, Any]]) -> list[FullRun]:
    result: list[FullRun] = []
    for profile_row in profiles:
        stem = Path(profile_row["filename"]).stem
        for node in nodes():
            elevations = CONSTRUCTION_ELEVATIONS if node.role == "construction" else HOLDOUT_ELEVATIONS
            for elevation in elevations:
                result.append(FullRun(f"tau025e001/{node.role}/{stem}/{node.node_id}/el{elevation:02d}", stem, node, elevation))
    result.sort(key=lambda item: item.run_id)
    require(len(result) == 1275, f"inventory count mismatch: {len(result)}")
    require(len({item.run_id for item in result}) == 1275, "duplicate full-grid run identifier")
    require(sum(item.node.role == "construction" for item in result) == 525, "construction inventory mismatch")
    require(sum(item.node.role == "heldout" for item in result) == 750, "held-out inventory mismatch")
    return result


def scale_inventory(profiles: Iterable[dict[str, Any]]) -> list[tuple[str, Node]]:
    result = [(Path(profile["filename"]).stem, node) for profile in profiles for node in nodes()]
    require(len(result) == 225 and len(set((a, b.node_id) for a, b in result)) == 225, "scale trace inventory mismatch")
    return result


def passband_binding() -> dict[str, Any]:
    paths = sorted(("index.yaml", "data/a1100_passband.ecsv", "data/a1400_passband.ecsv", "data/a2000_passband.ecsv"))
    aggregate = hashlib.sha256(); total = 0; members = []
    for relative in paths:
        data = subprocess.check_output(["git", "-C", str(TOLTECA_REPO), "show", f"{TOLTECA_COMMIT}:tolteca/data/cal/toltec_passband/{relative}"])
        digest = sha256_bytes(data)
        aggregate.update(relative.encode("utf-8")); aggregate.update(b"\0"); aggregate.update(bytes.fromhex(digest)); aggregate.update(b"\0")
        total += len(data); members.append({"relative_path": relative, "bytes": len(data), "sha256": digest})
    require(aggregate.hexdigest() == PASSBAND_SET_SHA256 and total == PASSBAND_TOTAL_BYTES, "passband byte binding mismatch")
    return {"set_sha256": aggregate.hexdigest(), "total_bytes": total, "members": members}


def input_binding(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    executable = p1.build_identity(AM_EXECUTABLE)
    require(executable.sha256 == AM_EXECUTABLE_SHA256 and executable.binary_format == "mach-o", "AM executable identity mismatch")
    source_root = AM_ROOT / "am-12.2/src"
    source = p1.inventory_files(source_root, p1.source_files(source_root))
    require(source["file_count"] == 135 and source["aggregate_sha256"] == AM_SOURCE_SHA256, "AM source payload mismatch")
    contracts = p1.validate_am_contract_files(AM_ROOT)
    return {"am_executable": p1.build_identity_payload(executable), "am_source": source, "am_contract_files": contracts, "profiles": profiles, "passbands": passband_binding()}


def derived_provenance() -> list[dict[str, str]]:
    getcontext().prec = 100
    rows = []
    for node in nodes():
        calculated_literal = format(float((-Decimal(node.requested_tau) * X80).exp()), ".6e")
        require(calculated_literal == node.target_literal, f"target literal mismatch: {node.node_id}")
        recomputed = -Decimal(node.target_literal).ln() / X80
        difference = abs(recomputed - Decimal(node.printed_achieved_tau))
        require(difference <= Decimal("1e-12"), f"derived provenance mismatch: {node.node_id}")
        rows.append({"node_id": node.node_id, "requested_tau": node.requested_tau, "target_literal": node.target_literal, "recomputed_achieved_tau": format(recomputed, ".70f"), "absolute_difference": format(difference, ".70E")})
    return rows


def cache_admission(cache_root: Path, *, dry_run_sentinel: bool = False) -> dict[str, Any]:
    """Verify an execution root, or a deliberately uncreatable dry-run leaf.

    The sentinel option exists solely for regression/preflight evidence after
    the formerly selected execution root became preserved forensic evidence.
    It is never accepted by ``--run`` and this function does not create it.
    """
    approved_execution_root = cache_root == CACHE_ROOT
    approved_sentinel = (
        dry_run_sentinel
        and cache_root.is_absolute()
        and cache_root.name == CACHE_BASENAME
        and cache_root != CACHE_ROOT
    )
    require(cache_root.is_absolute() and cache_root.name == CACHE_BASENAME and (approved_execution_root or approved_sentinel), "unapproved cache root")
    require(cache_root != FORENSIC_CACHE_ROOT, "forensic cache root is ineligible")
    parent = cache_root.parent
    require(parent.is_dir() and os.access(parent, os.W_OK | os.X_OK), "cache parent is not writable")
    require(not cache_root.exists() and not cache_root.is_symlink(), "fresh cache root already exists")
    admission_lock = parent / f".{cache_root.name}.admission.lock"
    require(not admission_lock.exists() and not admission_lock.is_symlink(), "cache admission lock already exists")
    free = shutil.disk_usage(parent).free
    require(free >= MIN_FREE_BYTES, f"insufficient cache storage: {free}")
    return {"cache_root": str(cache_root), "target_absent": True, "parent_writable": True, "admission_lock_absent": True, "free_bytes": free, "minimum_free_bytes": MIN_FREE_BYTES, "dry_run_sentinel": dry_run_sentinel, "retry_root_authorization": {"decision_id": RETRY_ROOT_AUTHORIZATION_ID, "sha256": RETRY_ROOT_AUTHORIZATION_SHA256}}


def preflight(cache_root: Path, *, dry_run_sentinel: bool = False) -> dict[str, Any]:
    profiles = profile_inventory()
    inventory = full_inventory(profiles)
    scales = scale_inventory(profiles)
    return {"runner_id": "SCI-CAL-001-TAU025-RUNNER-001", "runner_sha256": sha256_path(Path(__file__)), "request_sha256": sha256_path(REQUEST_PATH), "protocol_sha256": sha256_path(PROTOCOL_PATH), "inputs": input_binding(profiles), "derived_provenance": derived_provenance(), "full_run_count": len(inventory), "scale_trace_count": len(scales), "full_inventory": [{"run_id": item.run_id, "profile": item.profile, "node_id": item.node.node_id, "role": item.node.role, "elevation_deg": item.elevation_deg, "zenith_angle_deg": 90 - item.elevation_deg} for item in inventory], "cache_admission": cache_admission(cache_root, dry_run_sentinel=dry_run_sentinel)}


def deserialize_full_inventory(rows: Iterable[dict[str, Any]]) -> list[FullRun]:
    """Restore the committed JSON inventory before any execution expansion."""
    node_by_id = {node.node_id: node for node in nodes()}
    result: list[FullRun] = []
    for row in rows:
        node_id = str(row["node_id"])
        require(node_id in node_by_id, f"unknown serialized node: {node_id}")
        node = node_by_id[node_id]
        item = FullRun(str(row["run_id"]), str(row["profile"]), node, int(row["elevation_deg"]))
        require(row["role"] == node.role, f"serialized role mismatch: {item.run_id}")
        require(int(row["zenith_angle_deg"]) == 90 - item.elevation_deg, f"serialized geometry mismatch: {item.run_id}")
        expected_id = f"tau025e001/{node.role}/{item.profile}/{node.node_id}/el{item.elevation_deg:02d}"
        require(item.run_id == expected_id, f"serialized run identifier mismatch: {item.run_id}")
        result.append(item)
    result.sort(key=lambda item: item.run_id)
    require(len(result) == 1275 and len({item.run_id for item in result}) == 1275, "serialized full-grid inventory mismatch")
    require(sum(item.node.role == "construction" for item in result) == 525, "serialized construction inventory mismatch")
    require(sum(item.node.role == "heldout" for item in result) == 750, "serialized held-out inventory mismatch")
    return result


def full_grid_specification(item: FullRun, scale_decimal: str) -> p1.RunSpec:
    """Build the first would-be direct-AM command request from restored data."""
    return p1.full_grid_spec(
        "tau025_direct_full_grid", item.profile, item.node.node_id,
        90 - item.elevation_deg, scale_decimal,
    )


def cache_id(spec: p1.RunSpec, profile_sha256: str, context_sha256: str) -> str:
    identity = {"request": spec.request_payload(), "am_executable_sha256": AM_EXECUTABLE_SHA256, "profile_sha256": profile_sha256, "omp_threads": OMP_THREADS, "cache_shard_count": SHARDS, "execution_context_sha256": context_sha256}
    return sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())[:24]


class CacheRunner:
    """Minimal run/sidecar surface with the frozen AM and WARN-001 contract."""

    def __init__(self, root: Path, context: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> None:
        self.root, self.context, self.context_sha256, self.profiles = root, context, sha256_bytes(canonical_json(context)), profiles
        self.executable = p1.build_identity(AM_EXECUTABLE)
        self.observations: list[dict[str, Any]] = []
        self.shard_locks = [threading.Lock() for _ in range(SHARDS)]

    def _paths(self, identifier: str, spec: p1.RunSpec) -> tuple[Path, Path, int, str]:
        profile_sha = self.profiles[f"{spec.profile}.amc"]["sha256"]
        digest = cache_id(spec, profile_sha, self.context_sha256)
        shard = int.from_bytes(hashlib.sha256(digest.encode()).digest()[:8], "big") % SHARDS
        safe = identifier.replace("/", "__") + "__" + digest
        return self.root / "raw_outputs" / f"{safe}.txt", self.root / "sidecars" / f"{safe}.json", shard, digest

    def argv(self, spec: p1.RunSpec) -> list[str]:
        return [self.executable.resolved_path, f"LMT_am_inputs/{spec.profile}.amc", p1.f64(spec.f_min_centi_ghz / 100.0), "GHz", p1.f64(spec.f_max_centi_ghz / 100.0), "GHz", "10", "MHz", str(spec.zenith_angle_deg), "deg", spec.scale_decimal]

    def _warn_ok(self, status: int, parsed: p1.ParsedOutput, expected_rows: int) -> None:
        if status == 0:
            require(parsed.warning_count is None and parsed.other_warning_line_count == 0 and parsed.error_line_count == 0, "status-0 warning/error evidence")
            return
        require(status == 1, f"WARN-001 rejected AM status: {status}")
        require(parsed.samples.shape[0] == expected_rows == 50001, "WARN-001 row-count failure")
        require(parsed.warning_count in {86, 87, 88} and parsed.unresolved_summary_warning_line_count == 1 and parsed.unresolved_column_warning_line_count >= 1 and parsed.other_warning_line_count == 0 and parsed.error_line_count == 0, "WARN-001 warning-class failure")

    def run(self, identifier: str, spec: p1.RunSpec) -> dict[str, Any]:
        raw_path, sidecar_path, shard, digest = self._paths(identifier, spec)
        if raw_path.exists() or sidecar_path.exists():
            raise RuntimeError(f"unexpected cache reuse: {identifier}")
        environment = os.environ.copy(); environment.update({"LANG": "C", "LC_ALL": "C", "OMP_NUM_THREADS": "1", "AM_CACHE_PATH": str(self.root / "am_spectral_cache" / f"shard_{shard:02d}")})
        with self.shard_locks[shard]:
            completed = subprocess.run(self.argv(spec), cwd=AM_ROOT / "Big_Atmosphere", env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        raw = completed.stdout
        try:
            parsed = p1.parse_output(raw, spec)
            self._warn_ok(completed.returncode, parsed, spec.expected_rows)
        except Exception as error:
            failed = self.root / "failed_attempts" / f"{identifier.replace('/', '__')}__{sha256_bytes(raw)[:16]}"
            write_atomic(failed.with_suffix(".txt"), raw)
            write_atomic(failed.with_suffix(".failure.json"), canonical_json({"run_id": identifier, "argv": self.argv(spec), "return_code": completed.returncode, "raw_sha256": sha256_bytes(raw), "rejection": f"{type(error).__name__}: {error}"}))
            raise
        sidecar = {"schema_version": "sci-cal-001-tau025-am-run-v1", "run_id": identifier, "cache_id": digest, "request": spec.request_payload(), "argv": self.argv(spec), "profile_sha256": self.profiles[f"{spec.profile}.amc"]["sha256"], "am_executable_sha256": AM_EXECUTABLE_SHA256, "execution_context_sha256": self.context_sha256, "return_code": completed.returncode, "raw_sha256": sha256_bytes(raw), "numeric_text_sha256": parsed.numeric_text_sha256, "normalized_output_sha256": parsed.normalized_output_sha256, "numeric_row_count": int(parsed.samples.shape[0]), "unresolved_line_warning_count": parsed.warning_count, "unresolved_column_warning_line_count": parsed.unresolved_column_warning_line_count, "unresolved_summary_warning_line_count": parsed.unresolved_summary_warning_line_count, "other_warning_line_count": parsed.other_warning_line_count, "error_line_count": parsed.error_line_count, "am_version_identity": parsed.version_identity, "am_cache_shard": shard}
        write_atomic(raw_path, raw); write_atomic(sidecar_path, canonical_json(sidecar)); self.observations.append(sidecar)
        return {"parsed": parsed, "sidecar": sidecar, "spec": spec}


class ScaleAdapter:
    """The frozen P1 plateau solver over this runner's cache/provenance surface."""

    def __init__(self, runner: CacheRunner) -> None:
        self.runner = runner
        self.cache_dir = runner.root
        self.execution_context_sha256 = runner.context_sha256
        self.execute = True
        self._results: dict[tuple[tuple[str, Any], ...], p1.RunResult] = {}

    def run_or_load(self, spec: p1.RunSpec) -> p1.RunResult:
        key = tuple(sorted(spec.request_payload().items()))
        if key not in self._results:
            identifier = "/".join(("tau025_scale", spec.profile, spec.target, spec.stage, spec.scale_decimal))
            item = self.runner.run(identifier, spec)
            parsed, sidecar = item["parsed"], item["sidecar"]
            self._results[key] = p1.RunResult(spec, parsed, int(sidecar["return_code"]), sidecar["raw_sha256"], sidecar, sidecar["cache_id"])
        return self._results[key]


def create_cache(root: Path, context: dict[str, Any]) -> tuple[int, Any]:
    """Acquire a sibling admission lock before the one permitted root mkdir."""
    admission = root.parent / f".{root.name}.admission.lock"
    descriptor = os.open(admission, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        require(not root.exists(), "fresh root raced into existence")
        root.mkdir(mode=0o700)
        for relative in ("raw_outputs", "sidecars", "scale_traces", "failed_attempts", "manifests", *(f"am_spectral_cache/shard_{i:02d}" for i in range(SHARDS))):
            (root / relative).mkdir(parents=True, exist_ok=False)
        write_atomic(root / "execution_context.json", canonical_json(context))
        write_atomic(root / "inputs_manifest.json", canonical_json(context["inputs"]))
        lock = (root / CACHE_LOCK_NAME).open("a+b")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor, lock
    except Exception:
        if admission.exists():
            os.unlink(admission)
        os.close(descriptor)
        raise


def run_study(cache_root: Path) -> None:
    context = preflight(cache_root)
    profiles = {row["filename"]: row for row in context["inputs"]["profiles"]}
    inventory = deserialize_full_inventory(context["full_inventory"])
    descriptor, lock = create_cache(cache_root, context)
    admission = cache_root.parent / f".{cache_root.name}.admission.lock"
    try:
        os.unlink(admission)
        runner = CacheRunner(cache_root, context, profiles)
        scales: dict[tuple[str, str], str] = {}
        adapter = ScaleAdapter(runner)
        # Exact plateau search is delegated to the frozen P1 routine with the
        # new literals; it uses only Nscale through argv %9.
        saved = dict(p1.EXPECTED_TARGET_TRANSMISSIONS)
        try:
            p1.EXPECTED_TARGET_TRANSMISSIONS.update({node.node_id: node.target_literal for node in nodes()})
            for profile in sorted({item.profile for item in inventory}):
                scale0 = adapter.run_or_load(p1.anchor_spec(profile, "shared_scale0", p1.f64(0.0)))
                scale1 = adapter.run_or_load(p1.anchor_spec(profile, "shared_scale1", p1.f64(1.0)))
                copied_tau, copied_tx = p1.copied_anchor(AM_ROOT, profile)
                for node in nodes():
                    solution = p1.solve_scale_hypothesis(runner=adapter, profile=profile, target=node.node_id, scale0=scale0, scale1=scale1, copied_scale1_tau=copied_tau, copied_scale1_transmission=copied_tx)
                    require(solution.exact_parsed_transmission_match, f"exact parsed literal not reached: {profile}/{node.node_id}")
                    source = cache_root / solution.trace_relative_path
                    destination = cache_root / "scale_traces" / f"{profile}__{node.node_id}.json"
                    require(source.is_file() and not destination.exists(), f"scale trace collision: {profile}/{node.node_id}")
                    os.replace(source, destination)
                    scales[(profile, node.node_id)] = solution.scale_decimal
            require(len(scales) == 225, f"scale inventory incomplete: {len(scales)}")
            def full_grid(item: FullRun) -> dict[str, Any]:
                spec = full_grid_specification(item, scales[(item.profile, item.node.node_id)])
                return runner.run(item.run_id, spec)
            with concurrent.futures.ThreadPoolExecutor(max_workers=JOBS) as pool:
                list(pool.map(full_grid, inventory))
            require(len(runner.observations) >= 1275, "full-grid evidence count incomplete")
            require(not list((cache_root / "failed_attempts").glob("*.failure.json")), "rejected AM attempt present")
            raw_files = sorted((cache_root / "raw_outputs").glob("*.txt"))
            sidecars = sorted((cache_root / "sidecars").glob("*.json"))
            require(len(raw_files) == len(sidecars), "raw/sidecar pairing mismatch")
            manifest = {"study_id": "SCI-CAL-001-TAU025-ENGINEERING-EXTENSION-001", "execution_context_sha256": sha256_bytes(canonical_json(context)), "full_grid_count": 1275, "scale_trace_count": 225, "raw_sidecar_pair_count": len(raw_files), "raw_outputs": [{"path": path.relative_to(cache_root).as_posix(), "sha256": sha256_path(path)} for path in raw_files], "sidecars": [{"path": path.relative_to(cache_root).as_posix(), "sha256": sha256_path(path)} for path in sidecars]}
            write_atomic(cache_root / "manifests" / "execution_manifest.json", canonical_json(manifest))
        finally:
            p1.EXPECTED_TARGET_TRANSMISSIONS.clear(); p1.EXPECTED_TARGET_TRANSMISSIONS.update(saved)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close(); os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--test-sentinel-cache-root", type=Path,
                        help="nonexistent test-only leaf for --dry-run; rejected by --run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        cache_root = args.test_sentinel_cache_root or args.cache_root
        print(json.dumps(preflight(cache_root, dry_run_sentinel=args.test_sentinel_cache_root is not None), indent=2, sort_keys=True))
        return 0
    require(args.test_sentinel_cache_root is None, "--test-sentinel-cache-root is dry-run only")
    run_study(args.cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
