#!/usr/bin/env python3
"""Fail-closed, human-run Unity driver for SCI-MAP-001-UNITY-001.

This program is an owner tool.  It never contacts Unity, calls ``sbatch``, or
runs the numerical analysis.  When an authorized owner runs it *on* Unity it
can verify identities, initialize the isolated request root, build the pinned
candidate, create the seven native TolProj reductions, and emit (but never
execute) explicit Slurm and retrieval plans.

Every mutating command refuses to overwrite an existing campaign artifact.
An interrupted command therefore leaves state for owner inspection rather
than trying to repair or reuse an uncertain partial campaign.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, NoReturn, Sequence


CAMPAIGN_SCHEMA = "sci-map-unity-campaign-v1"
OWNER_SCHEMA = "sci-map-unity-owner-values-v1"
REQUEST_ID = "SCI-MAP-001-UNITY-001"
EXPECTED_CANDIDATE = "ed28dafb37f9113c0d3c95297148157129a90886"
EXPECTED_TREE = "cf75c36557178f351fb62781108a6f4b41b19225"
EXPECTED_PARENT = "9aae0e669384c5c0c0dda93debc194d6b8dac787"
FORBIDDEN_ANCESTOR = "02a198cbfb379eaf6ab279c5a3d44ee73ff90435"
CASE_IDS = (
    "P-SEQ", "P-OMP", "S-C-SEQ", "S-C-OMP", "S-E-SEQ", "S-E-OMP",
    "S-X-SEQ",
)
CASE_CPUS = {
    "P-SEQ": 1, "P-OMP": 6, "S-C-SEQ": 1, "S-C-OMP": 16,
    "S-E-SEQ": 1, "S-E-OMP": 16, "S-X-SEQ": 1,
}
FROZEN_PACKAGE_RELATIVE = Path(
    "frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
)
PREPARED_DIRECTORIES = (
    "bin", "governing", "tolproj-kit", "records", "state", "evidence",
    "manifests", "raw-input-manifests", "sample-ledgers", "analysis", "plans",
    "frozen-package-tree",
)
OWNER_KEYS = (
    "schema_version", "unity_host_alias", "unity_source_checkout",
    "request_root", "deployed_campaign_path", "unity_python",
    "tolproj_executable", "tolproj_site_config", "point_project",
    "point_source_filter", "point_apt_dir", "science_project",
    "science_source_basename", "science_pointing_reduction",
    "evidence_operator", "slurm_account", "slurm_qos",
    "slurm_constraint", "slurm_reservation", "kidscpp_source_dir",
    "tula_source_dir", "local_retrieval_destination",
)
UNITY_EXISTING_PATHS = (
    "unity_source_checkout", "deployed_campaign_path", "unity_python",
    "tolproj_executable", "tolproj_site_config", "point_project",
    "point_apt_dir", "science_project", "kidscpp_source_dir",
    "tula_source_dir",
)
ABSOLUTE_PATH_KEYS = (
    "unity_source_checkout", "request_root", "deployed_campaign_path",
    "unity_python", "tolproj_executable", "tolproj_site_config",
    "point_project", "point_apt_dir", "science_project",
    "kidscpp_source_dir", "tula_source_dir", "local_retrieval_destination",
)
OPTIONAL_OWNER_STRINGS = ("slurm_qos", "slurm_constraint", "slurm_reservation")
NUMBERED_RE = re.compile(r"^[0-9]{2}_.+\.ya?ml$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$")
SLURM_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
SLURM_FIELDS = (
    "JobIDRaw", "JobName", "Partition", "AllocCPUS", "NodeList", "State",
    "ExitCode", "Elapsed", "MaxRSS", "ReqMem", "Submit", "Start", "End",
)
RAW_MANIFEST_SCHEMA = "sci-map-001-raw-input-manifest-v1"
LEDGER_SCHEMA = "sci-map-001-independent-sample-ledger-v1"
ARRAYS = ("a1100", "a1400", "a2000")
SOURCE_ROLES = (
    "raw_timestream", "kids_fit_report", "apt", "calibration",
    "pointing_support", "projection_authority", "sample_rate_authority",
    "fwhm_authority", "target_authority",
)
LEDGER_ARRAYS = (
    "row", "col", "detector_index", "sample_index", "scan_index",
    "geometric_in_bounds", "upstream_eligible", "coefficient",
    "sample_signal", "sample_kernel", "sample_interval_s", "realization_signs",
)
LEDGER_SCALARS = (
    "schema_version", "candidate_sha", "raw_input_manifest_sha256",
    "producer_identity", "bundle_identity_digest", "obsnum", "array",
    "map_rows", "map_cols", "sample_rate_hz_numeric",
    "sample_rate_hz_hex", "sample_rate_hz_encoding",
)


class CampaignError(RuntimeError):
    """A fail-closed campaign preparation error."""


def fail(message: str) -> NoReturn:
    raise CampaignError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def expected_align_state() -> dict[str, Any]:
    return {
        "owner_decision_commit": "4f905f4f353e91847a303f4f3959654f3f03c302",
        "coordination_identity_correction_commit": (
            "35cc8ce246e8e70c569e650be6c1eae2c91b80ef"),
        "handoff_coordination_commit": "0309fd48a973a6e7e136224906ac49c02f0171be",
        "coordination_ledger_head": "846128c8ee6dc27851bd6c71aeecbe4739e1d24a",
        "repair_base": EXPECTED_PARENT,
        "coordination_state": "bounded_repair_reaudit_handoff_integrated",
        "implementation_conformance": "nonconformant",
        "validation_state": "in_progress",
        "production_state": "existing_use_only",
        "repair_task_state": "dedicated_phase_0_repair_active",
        "application_repair_commit": None,
        "re_audit_state": "not_started",
        "approved_outcomes": [*[f"ALIGN-OD{index}" for index in range(1, 9)],
                              "ALIGN-C001"],
        "map_execution_condition": (
            "F013 remains conditioned until the ALIGN repair, exact-repair-SHA "
            "evidence, and fresh re-audit succeed; this MAP campaign supplies no "
            "ALIGN closure evidence"),
    }


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read JSON {path}: {exc}") from exc


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CampaignError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CampaignError(f"refusing to overwrite existing artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(mode)
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def copy_new(source: Path, destination: Path, mode: int = 0o444) -> None:
    if destination.exists() or destination.is_symlink():
        fail(f"refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as src:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
        destination.chmod(mode)
    except Exception:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise


def run(command: Sequence[str], *, cwd: Path | None = None,
        check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [str(item) for item in command], cwd=str(cwd) if cwd else None,
            check=False, text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except OSError as exc:
        raise CampaignError(f"cannot execute {shlex.join(command)}: {exc}") from exc
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        fail(f"command failed ({result.returncode}): {shlex.join(command)}"
             + (f"\n{stderr}" if stderr else ""))
    return result


def run_logged(command: Sequence[str], *, cwd: Path, transcript: Path) -> None:
    if transcript.exists() or transcript.is_symlink():
        fail(f"refusing to overwrite transcript: {transcript}")
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("x", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join([str(item) for item in command]) + "\n")
        stream.flush()
        try:
            result = subprocess.run(
                [str(item) for item in command], cwd=str(cwd), check=False,
                text=True, stdout=stream, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            stream.write(f"driver execution error: {exc}\n")
            raise CampaignError(f"cannot execute {command[0]}: {exc}") from exc
        stream.write(f"\n[driver exit status: {result.returncode}]\n")
        stream.flush()
        os.fsync(stream.fileno())
    transcript.chmod(0o444)
    if result.returncode != 0:
        fail(f"command failed ({result.returncode}); inspect {transcript}")


def git(source: Path, *arguments: str, check: bool = True) -> str:
    result = run(("git", "-C", str(source), *arguments), check=check)
    return (result.stdout or "").strip()


def load_campaign(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve(strict=True)
    data = read_json(resolved)
    if not isinstance(data, dict) or data.get("schema_version") != CAMPAIGN_SCHEMA:
        fail("campaign.json has an unsupported schema")
    if data.get("request_id") != REQUEST_ID:
        fail("campaign request identity differs")
    if data.get("candidate_sha") != EXPECTED_CANDIDATE:
        fail("campaign candidate SHA differs")
    if data.get("candidate_tree") != EXPECTED_TREE or data.get("candidate_parent") != EXPECTED_PARENT:
        fail("campaign candidate tree or parent differs")
    cases = data.get("cases")
    if not isinstance(cases, list) or [case.get("id") for case in cases] != list(CASE_IDS):
        fail("campaign must contain the seven cases in pinned order")
    if data.get("fixed_execution", {}).get("ssh_alias") != "unity_toltec":
        fail("campaign SSH alias differs from unity_toltec")
    if data.get("coordination_state", {}).get("F013") != "conditioned_on_named_upstream_audits":
        fail("campaign no longer records the F013 upstream condition")
    align = data.get("coordination_state", {}).get("sci_align_001", {})
    if align != expected_align_state():
        fail("campaign SCI-ALIGN-001 owner-approved active-repair state differs")
    if data.get("coordination_state", {}).get("not_closed_by_this_campaign") != [
            "SCI-ALIGN-001", "SCI-CAL-001", "SCI-AST-001", "SCI-PTC-001",
            "SCI-VAL-001"]:
        fail("campaign upstream nonclosure boundary differs")
    numbered = data.get("numbered_config_contract", {})
    if numbered.get("runtime_recognized_total") != 9 or numbered.get(
            "post_setup_source_count") != 8:
        fail("campaign numbered-config count is not the corrected 40+8 contract")
    return data, resolved


def package_root(campaign_path: Path) -> Path:
    return campaign_path.parent


def verify_package(root: Path, *, require_inventory: bool = True) -> dict[str, str]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        fail(f"campaign package is not a directory: {root}")
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        if require_inventory:
            fail(f"campaign package lacks SHA256SUMS: {sums_path}")
        return {}
    records: dict[str, str] = {}
    for line_number, raw in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", raw)
        if match is None:
            fail(f"invalid SHA256SUMS line {line_number}")
        digest, relative = match.groups()
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or relative == "SHA256SUMS":
            fail(f"unsafe or self-referential SHA256SUMS path: {relative}")
        if relative in records:
            fail(f"duplicate SHA256SUMS path: {relative}")
        target = root / rel
        if not target.is_file() or target.is_symlink():
            fail(f"SHA256SUMS target is absent or not a regular file: {relative}")
        actual = sha256(target)
        if actual != digest:
            fail(f"campaign package digest differs: {relative}")
        records[relative] = digest
    required = {
        "campaign.json", "owner-values.schema.json", "owner-values.template.json",
        "raw-input-manifest.schema.json",
        "raw-input-manifest.point.template.json",
        "raw-input-manifest.science.template.json",
        "result-collection.schema.json", "result-collection.template.json",
        "sample-ledger-contract.json", "SCI-MAP-001-analysis.py",
        "scripts/analysis-job-wrapper.sh", "scripts/case-job-wrapper.sh",
        "scripts/hash-tree.py", "scripts/unity-campaign.py",
        "scripts/verify-package.sh",
    }
    if not required.issubset(records):
        fail(f"SHA256SUMS omits required package files: {sorted(required - set(records))}")
    entries = list(root.rglob("*"))
    symlinks = sorted(path.relative_to(root).as_posix() for path in entries
                      if path.is_symlink())
    if symlinks:
        fail(f"campaign package contains symlinks: {symlinks}")
    actual_files = {
        path.relative_to(root).as_posix() for path in entries
        if path.is_file() and path != sums_path
    }
    if actual_files != set(records):
        fail("SHA256SUMS is not an exhaustive package inventory; "
             f"missing={sorted(actual_files - set(records))}, "
             f"stale={sorted(set(records) - actual_files)}")
    executable_files = {
        "SCI-MAP-001-analysis.py", "scripts/analysis-job-wrapper.sh",
        "scripts/case-job-wrapper.sh", "scripts/hash-tree.py",
        "scripts/unity-campaign.py", "scripts/verify-package.sh",
    }
    not_executable = sorted(relative for relative in executable_files
                            if not os.access(root / relative, os.X_OK))
    if not_executable:
        fail(f"campaign executable files lack execute permission: {not_executable}")
    return records


def validate_owner(path: Path, campaign: Mapping[str, Any],
                   *, require_existing: bool = True) -> dict[str, str]:
    raw = read_json(path.expanduser().resolve(strict=True))
    if not isinstance(raw, dict):
        fail("owner values are not a JSON object")
    missing = sorted(set(OWNER_KEYS) - set(raw))
    extra = sorted(set(raw) - set(OWNER_KEYS))
    if missing or extra:
        fail(f"owner-value keys differ; missing={missing}, extra={extra}")
    if raw.get("schema_version") != OWNER_SCHEMA:
        fail("owner-value schema_version differs")
    if raw.get("unity_host_alias") != "unity_toltec":
        fail("unity_host_alias must be exactly unity_toltec")
    values: dict[str, str] = {}
    placeholders = ("todo", "change_me", "changeme", "unknown", "<", ">")
    for key in OWNER_KEYS:
        value = raw[key]
        if key in OPTIONAL_OWNER_STRINGS:
            if not isinstance(value, str):
                fail(f"{key} must be a string; empty is the explicit unused value")
        elif not isinstance(value, str) or not value.strip():
            fail(f"owner value {key} is unresolved")
        if isinstance(value, str) and ("\n" in value or "\r" in value):
            fail(f"owner value {key} contains a line break")
        if isinstance(value, str) and any(token in value.lower() for token in placeholders):
            fail(f"owner value {key} contains a placeholder")
        values[key] = value
    for key in ABSOLUTE_PATH_KEYS:
        raw_path = values[key]
        value = Path(raw_path)
        if not value.is_absolute() or value in (Path("/"), Path("/tmp"), Path("/work")):
            fail(f"owner value {key} must be a specific absolute path")
        if raw_path != os.path.normpath(raw_path):
            fail(f"owner path {key} must be lexically normalized with no trailing slash")
        if "\\" in raw_path:
            fail(f"owner path {key} contains an unsupported backslash")
    if "/" in values["science_source_basename"] or values["science_source_basename"] in (".", ".."):
        fail("science_source_basename must be one safe basename")
    if re.fullmatch(r"redu[0-9]{2}", values["science_pointing_reduction"]) is None:
        fail("science_pointing_reduction must match reduNN")
    if require_existing:
        for key in UNITY_EXISTING_PATHS:
            if not Path(values[key]).exists():
                fail(f"owner Unity path does not exist: {key}={values[key]}")
        for key in ("unity_python", "tolproj_executable"):
            if not Path(values[key]).is_file() or not os.access(values[key], os.X_OK):
                fail(f"owner executable is not an executable file: {key}")
        for key in ("tolproj_site_config",):
            if not Path(values[key]).is_file():
                fail(f"owner file is absent: {key}")
        for key in ("unity_source_checkout", "deployed_campaign_path", "point_project",
                    "point_apt_dir", "science_project", "kidscpp_source_dir",
                    "tula_source_dir"):
            if not Path(values[key]).is_dir():
                fail(f"owner directory is absent: {key}")
    request_root = Path(values["request_root"]).resolve(strict=False)
    source = Path(values["unity_source_checkout"]).resolve(strict=require_existing)
    deployed = Path(values["deployed_campaign_path"]).resolve(strict=require_existing)
    if request_root == source or source in request_root.parents or request_root in source.parents:
        fail("request_root and candidate checkout must be disjoint")
    if (request_root == deployed or request_root in deployed.parents or
            deployed in request_root.parents):
        fail("request_root and deployed_campaign_path must be disjoint")
    if require_existing:
        deployed_campaign, deployed_campaign_path = load_campaign(deployed / "campaign.json")
        if deployed_campaign != campaign:
            fail("deployed campaign.json differs from the selected campaign")
        # The staging transfer contains only this versioned package. Before
        # request-root preparation, executable authorities therefore come from
        # the already-proven exact candidate checkout rather than an invented
        # sibling path outside the standalone package.
        source_root = Path(values["unity_source_checkout"]).resolve(strict=True)
        for pinned_relative in (
            "validation/product_contracts.json",
            "validation/validation_profiles.json",
        ):
            authority_path = source_root / pinned_relative
            expected_digest = campaign["pinned_source_sha256"][pinned_relative]
            if not authority_path.is_file() or sha256(authority_path) != expected_digest:
                fail(f"candidate checkout cannot resolve pinned {pinned_relative}")
    return values


def check_request_root(values: Mapping[str, str], expectation: str) -> Path:
    root = Path(values["request_root"]).resolve(strict=False)
    exists = os.path.lexists(root)
    if expectation == "absent" and exists:
        fail(f"request_root already exists; stop for owner inspection: {root}")
    if expectation == "prepared":
        if not exists or not root.is_dir() or root.is_symlink():
            fail(f"request_root is not a prepared ordinary directory: {root}")
        state_path = root / "state" / "prepared.json"
        state = read_json(state_path)
        expected = {
            "schema_version": "sci-map-unity-prepared-state-v1",
            "request_id": REQUEST_ID,
            "candidate_sha": EXPECTED_CANDIDATE,
        }
        if not isinstance(state, dict) or any(state.get(k) != v for k, v in expected.items()):
            fail(f"request_root has no matching prepared state: {state_path}")
        if state.get("sci_align_001") != expected_align_state() or \
                state.get("not_closed_by_this_campaign") != [
                    "SCI-ALIGN-001", "SCI-CAL-001", "SCI-AST-001",
                    "SCI-PTC-001", "SCI-VAL-001"]:
            fail(f"request_root has stale upstream coordination state: {state_path}")
    return root


def require_frozen_owner_values(root: Path, values: Mapping[str, str]) -> None:
    frozen_path = root / "owner-values.json"
    frozen = read_json(frozen_path)
    if not isinstance(frozen, dict) or frozen != dict(values):
        fail(
            "owner values differ from the immutable preparation record; stop "
            f"for owner inspection instead of reusing changed deployment facts: {frozen_path}"
        )


def case_by_id(campaign: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    for case in campaign["cases"]:
        if case["id"] == case_id:
            return dict(case)
    fail(f"unknown case: {case_id}")


def tolproj_identity(values: Mapping[str, str], campaign: Mapping[str, Any]) -> dict[str, Any]:
    executable = Path(values["tolproj_executable"]).resolve(strict=True)
    shebang = executable.open("rb").readline().decode("utf-8", "replace").strip()
    expected_python = Path(values["unity_python"]).resolve(strict=True)
    if shebang.startswith("#!"):
        interpreter = Path(shebang[2:].split()[0]).resolve(strict=True)
        if interpreter != expected_python:
            fail(f"TolProj executable uses {interpreter}, not unity_python {expected_python}")
    else:
        fail("TolProj executable has no pinned Python shebang")
    if Path(sys.executable).resolve(strict=True) != expected_python:
        fail(f"run this driver with the explicit unity_python: {expected_python}")

    try:
        import tolproj
        from tolproj.config import config_to_plain_data, load_config, redact_config
        from tolproj.refactor_config import verified_vendor_metadata
        from tolteca.utils.runtime_context import RuntimeContext
        import tolteca
    except Exception as exc:
        raise CampaignError(f"cannot import TolProj/TolTECA with unity_python: {exc}") from exc

    package_dir = Path(tolproj.__file__).resolve(strict=True).parent
    vendor_path = package_dir / "templates/citlali_refactor/vendor.yaml"
    bundle_path = package_dir / "templates/citlali_refactor/phase4_1_v2_1/manifest.yaml"
    expected_manifest = campaign["pinned_source_sha256"]["config/tolteca/v2/manifest.yaml"]
    if sha256(bundle_path) != expected_manifest:
        fail("installed TolProj phase4_1_v2_1 manifest differs from candidate authority")
    vendor = verified_vendor_metadata()
    for mode in ("point", "science"):
        row = vendor["mode_kits"][mode]
        if row.get("bundle") != "phase4_1_v2_1" or row.get("source_commit") != campaign[
                "authority"]["tolproj_bundle_source_commit"]:
            fail(f"installed TolProj {mode} bundle identity differs")
    site = load_config(Path(values["tolproj_site_config"]))
    rendered = config_to_plain_data(redact_config(site))
    if not isinstance(site, Mapping):
        fail("TolProj site config did not resolve to a mapping")
    help_result = run((str(executable), "--help"))
    if "setup-pointing-reductions" not in (help_result.stdout or ""):
        fail("TolProj executable does not expose the required setup surface")
    runtime_module = Path(sys.modules[RuntimeContext.__module__].__file__).resolve(strict=True)
    tolteca_module = Path(tolteca.__file__).resolve(strict=True)
    python_files = sorted(path for path in package_dir.rglob("*.py") if path.is_file())
    package_digest = hashlib.sha256()
    for path in python_files:
        package_digest.update(path.relative_to(package_dir).as_posix().encode() + b"\0")
        package_digest.update(bytes.fromhex(sha256(path)))
    return {
        "tolproj_executable": str(executable),
        "tolproj_executable_sha256": sha256(executable),
        "tolproj_version": importlib.metadata.version("tolproj"),
        "tolproj_package_dir": str(package_dir),
        "tolproj_python_tree_sha256": package_digest.hexdigest(),
        "tolproj_python_file_count": len(python_files),
        "vendor_manifest": str(vendor_path),
        "vendor_manifest_sha256": sha256(vendor_path),
        "bundle_manifest": str(bundle_path),
        "bundle_manifest_sha256": sha256(bundle_path),
        "tolproj_site_config": str(Path(values["tolproj_site_config"]).resolve()),
        "tolproj_site_config_sha256": sha256(Path(values["tolproj_site_config"])),
        "tolproj_site_config_redacted": rendered,
        "tolteca_version": importlib.metadata.version("tolteca"),
        "tolteca_module": str(tolteca_module),
        "runtime_context_module": str(runtime_module),
        "runtime_context_module_sha256": sha256(runtime_module),
        "runtime_context_api": "tolteca.utils.runtime_context.RuntimeContext.config_backend.config_files",
    }


def dependency_identity(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if git(path, "rev-parse", "--is-inside-work-tree") != "true":
        fail(f"{label} is not a Git worktree: {path}")
    status_text = git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if status_text:
        fail(f"{label} worktree is dirty: {path}")
    return {
        "path": str(path), "head": git(path, "rev-parse", "HEAD"),
        "tree": git(path, "rev-parse", "HEAD^{tree}"), "status": "clean",
    }


def identity(values: Mapping[str, str], campaign: Mapping[str, Any],
             root_expectation: str) -> dict[str, Any]:
    root = check_request_root(values, root_expectation)
    if root_expectation == "prepared":
        require_frozen_owner_values(root, values)
        for name in PREPARED_DIRECTORIES:
            path = root / name
            if path.is_symlink() or not path.is_dir():
                fail(f"prepared request directory is absent or a symlink: {path}")
    source = Path(values["unity_source_checkout"]).resolve(strict=True)
    if git(source, "rev-parse", "--show-toplevel") != str(source):
        fail("unity_source_checkout is not the Git worktree root")
    head = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    parents = git(source, "show", "-s", "--format=%P", "HEAD").split()
    status_text = git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if head != EXPECTED_CANDIDATE or tree != EXPECTED_TREE or parents != [EXPECTED_PARENT]:
        fail("candidate checkout does not have the exact pinned commit/tree/parent")
    if status_text:
        fail("candidate checkout is dirty")
    forbidden = run(("git", "-C", str(source), "merge-base", "--is-ancestor",
                     FORBIDDEN_ANCESTOR, "HEAD"), check=False)
    if forbidden.returncode == 0:
        fail(f"forbidden commit is an ancestor of the candidate: {FORBIDDEN_ANCESTOR}")
    if forbidden.returncode not in (0, 1):
        fail("could not prove forbidden-commit ancestry")

    pinned_actual: dict[str, str] = {}
    for relative, expected in campaign["pinned_source_sha256"].items():
        path = source / relative
        if not path.is_file():
            fail(f"pinned source is absent: {relative}")
        actual = sha256(path)
        if actual != expected:
            fail(f"pinned source digest differs: {relative}")
        pinned_actual[relative] = actual
    build_dir = source / "build_unity_release"
    if Path(values["kidscpp_source_dir"]).resolve() != (build_dir / "_deps/kidscpp-src").resolve():
        fail("kidscpp_source_dir must name build_unity_release/_deps/kidscpp-src")
    if Path(values["tula_source_dir"]).resolve() != (source.parent / "tula").resolve():
        fail("tula_source_dir must name the unity_release preset's sourceParentDir/tula")
    package_records = verify_package(Path(values["deployed_campaign_path"]))
    if (source / "conanfile.py").exists() or (source / "conan.lock").exists():
        fail("unexpected Conan authority appeared at the pinned candidate")
    return {
        "schema_version": "sci-map-unity-identity-v1",
        "recorded_at": utc_now(),
        "request_id": REQUEST_ID,
        "candidate_sha": head,
        "candidate_tree": tree,
        "candidate_parent": parents[0],
        "candidate_status": "clean",
        "forbidden_commit_is_ancestor": False,
        "source": str(source),
        "request_root": str(root),
        "pinned_source_sha256": pinned_actual,
        "campaign_package_sha256": package_records,
        "tolproj_tolteca": tolproj_identity(values, campaign),
        "dependencies": {
            "kidscpp": dependency_identity(Path(values["kidscpp_source_dir"]), "kidscpp"),
            "tula": dependency_identity(Path(values["tula_source_dir"]), "tula"),
        },
        "conan": "not_applicable_no_conanfile_or_lock",
    }


def frozen_package(root: Path) -> Path:
    path = root / FROZEN_PACKAGE_RELATIVE
    verify_package(path)
    return path


def require_prepared(values: Mapping[str, str]) -> Path:
    root = check_request_root(values, "prepared")
    require_frozen_owner_values(root, values)
    for name in PREPARED_DIRECTORIES:
        path = root / name
        if path.is_symlink() or not path.is_dir():
            fail(f"prepared request directory is absent or a symlink: {path}")
    return root


def require_build(root: Path) -> dict[str, Any]:
    state = read_json(root / "state" / "build.json")
    binary = root / "bin" / "citlali"
    if not isinstance(state, dict) or state.get("candidate_sha") != EXPECTED_CANDIDATE:
        fail("build state is absent or does not match the candidate")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        fail("immutable candidate binary is absent")
    if sha256(binary) != state.get("binary_sha256"):
        fail("immutable candidate binary digest differs from build state")
    return state


def command_self_check(args: argparse.Namespace) -> int:
    campaign, campaign_path = load_campaign(args.campaign)
    root = package_root(campaign_path)
    verify_package(root, require_inventory=args.require_inventory)
    raw_schema = read_json(root / "raw-input-manifest.schema.json")
    if not isinstance(raw_schema, Mapping) or raw_schema.get("$id") != RAW_MANIFEST_SCHEMA:
        fail("raw-input-manifest.schema.json identity differs")
    for mode, observations in (("point", [152389]), ("science", [152390, 152392])):
        template = read_json(root / f"raw-input-manifest.{mode}.template.json")
        expected_pairs = [(obsnum, array) for obsnum in observations for array in ARRAYS]
        if not isinstance(template, Mapping) \
                or template.get("schema_version") != RAW_MANIFEST_SCHEMA \
                or template.get("request_id") != REQUEST_ID \
                or template.get("candidate_sha") != EXPECTED_CANDIDATE \
                or template.get("mode") != mode \
                or template.get("observations") != observations \
                or template.get("arrays") != list(ARRAYS):
            fail(f"{mode} raw-input manifest template identity differs")
        memberships = template.get("memberships")
        if not isinstance(memberships, list) or [
                (row.get("obsnum"), row.get("array"))
                for row in memberships if isinstance(row, Mapping)
        ] != expected_pairs:
            fail(f"{mode} raw-input manifest template membership order differs")
    for case in campaign["cases"]:
        if case["expected_arrays"] != campaign["fixed_execution"]["arrays"]:
            fail(f"case array inventory differs: {case['id']}")
        if case["id"] == "S-X-SEQ" and case.get("historical_jobkey_note") is None:
            fail("S-X-SEQ lacks its repaired-success identity note")
        test_values = {
            "slurm_account": "self-check-account", "slurm_qos": "",
            "slurm_constraint": "", "slurm_reservation": "",
        }
        test_options = slurm_options(
            test_values, campaign, cpus=int(case["cpus"]), memory="1G",
            time="00:01:00", job_name=f"self-check-{case['id']}",
            chdir=Path("/self-check"), stdout=Path("/self-check.out"),
            stderr=Path("/self-check.err"),
        )
        expected_export = f"--export=ALL,OMP_NUM_THREADS={case['cpus']}"
        if expected_export not in test_options or "--wait" not in test_options:
            fail(f"case Slurm/OMP export policy differs: {case['id']}")
    print("SCI-MAP-001 Unity campaign driver self-check passed")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign, require_existing=args.require_existing)
    if args.expect_request_root != "either":
        if args.expect_request_root == "prepared":
            require_prepared(values)
        else:
            check_request_root(values, args.expect_request_root)
    print(json.dumps({"result": "pass", "owner_values": str(args.owner_values.resolve()),
                      "request_root_expectation": args.expect_request_root}, sort_keys=True))
    return 0


def command_identity(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    record = identity(values, campaign, args.expect_request_root)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    record = identity(values, campaign, "absent")
    root = Path(values["request_root"]).resolve(strict=False)
    root.mkdir(mode=0o755, parents=False)
    for name in PREPARED_DIRECTORIES:
        (root / name).mkdir(mode=0o755)
    deployed = Path(values["deployed_campaign_path"]).resolve(strict=True)
    destination = root / FROZEN_PACKAGE_RELATIVE
    destination.parent.mkdir(parents=True, mode=0o755)
    shutil.copytree(deployed, destination, symlinks=True, copy_function=shutil.copy2)
    verify_package(destination)
    write_new(root / "owner-values.json", json_bytes(values))
    write_new(root / "records" / "identity.json", json_bytes(record))
    source = Path(values["unity_source_checkout"]).resolve(strict=True)
    # Preserve the package's repository-relative executable-authority paths.
    copy_new(source / "validation/product_contracts.json",
             root / "frozen-package-tree/validation/product_contracts.json")
    copy_new(source / "validation/validation_profiles.json",
             root / "frozen-package-tree/validation/validation_profiles.json")
    for relative in campaign["pinned_source_sha256"]:
        copy_new(source / relative, root / "governing" / relative)
    tolproj_record = record["tolproj_tolteca"]
    copy_new(Path(tolproj_record["vendor_manifest"]), root / "tolproj-kit" / "vendor.yaml")
    copy_new(Path(tolproj_record["bundle_manifest"]),
             root / "tolproj-kit" / "phase4_1_v2_1-manifest.yaml")
    state = {
        "schema_version": "sci-map-unity-prepared-state-v1",
        "prepared_at": utc_now(),
        "request_id": REQUEST_ID,
        "revision": campaign["revision"],
        "candidate_sha": EXPECTED_CANDIDATE,
        "implementation_conformance": "nonconformant",
        "F009": "addressed_pending_reaudit",
        "F010": "addressed_pending_reaudit",
        "F012": "outstanding_human_run_exact_repair_sha_evidence_gate",
        "F013": "conditioned_on_named_upstream_audits",
        "sci_align_001": campaign["coordination_state"]["sci_align_001"],
        "not_closed_by_this_campaign": campaign[
            "coordination_state"]["not_closed_by_this_campaign"],
        "external_evidence": "not_created_by_preparation",
    }
    write_new(root / "state" / "prepared.json", json_bytes(state))
    print(f"prepared isolated request root: {root}")
    return 0


def command_build(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    identity(values, campaign, "prepared")
    binary = root / "bin" / "citlali"
    state_path = root / "state" / "build.json"
    if os.path.lexists(binary) or os.path.lexists(state_path):
        fail("build output already exists; stop for owner inspection")
    source = Path(values["unity_source_checkout"]).resolve(strict=True)
    build_dir = source / "build_unity_release"
    build_started_at = utc_now()
    configure = (
        "cmake", "--preset", campaign["fixed_execution"]["build_preset"],
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    )
    run_logged(configure, cwd=source, transcript=root / "records" / "build-configure.txt")
    build = (
        "cmake", "--build", str(build_dir), "--target",
        campaign["fixed_execution"]["build_target"], "-j",
        str(campaign["fixed_execution"]["build_jobs"]),
    )
    run_logged(build, cwd=source, transcript=root / "records" / "build-compile.txt")
    built = build_dir / "bin" / "citlali"
    if not built.is_file() or not os.access(built, os.X_OK):
        fail(f"build did not produce the expected executable: {built}")
    copy_new(built, binary, mode=0o555)
    if built.read_bytes() != binary.read_bytes():
        fail("installed immutable binary differs from build output")
    cache = build_dir / "CMakeCache.txt"
    compile_commands = build_dir / "compile_commands.json"
    copy_new(cache, root / "records" / "CMakeCache.txt")
    copy_new(compile_commands, root / "records" / "compile_commands.json")
    rendered = run(("cmake", "-N", "-LA", str(build_dir)), cwd=source)
    write_new(root / "records" / "cmake-options-rendered.txt",
              (rendered.stdout or "").encode())
    version = run((str(binary), "--version"))
    write_new(root / "records" / "citlali-version.txt", (version.stdout or "").encode())
    compiler_match = re.search(r"^CMAKE_CXX_COMPILER:[^=]*=(.+)$",
                               cache.read_text(encoding="utf-8"), re.MULTILINE)
    if compiler_match is None:
        fail("CMakeCache does not identify CMAKE_CXX_COMPILER")
    compiler = Path(compiler_match.group(1).strip()).resolve(strict=True)
    compiler_version = run((str(compiler), "--version"))
    write_new(root / "records" / "compiler-version.txt",
              (compiler_version.stdout or "").encode())
    post = identity(values, campaign, "prepared")
    build_state = {
        "schema_version": "sci-map-unity-build-state-v1",
        "started_at": build_started_at,
        "completed_at": utc_now(),
        "candidate_sha": EXPECTED_CANDIDATE,
        "build_preset": campaign["fixed_execution"]["build_preset"],
        "build_target": campaign["fixed_execution"]["build_target"],
        "binary": str(binary), "binary_sha256": sha256(binary),
        "built_binary": str(built), "built_binary_sha256": sha256(built),
        "cmake_cache_sha256": sha256(root / "records" / "CMakeCache.txt"),
        "compile_commands_sha256": sha256(root / "records" / "compile_commands.json"),
        "compiler": str(compiler), "compiler_sha256": sha256(compiler),
        "dependencies": post["dependencies"],
    }
    write_new(state_path, json_bytes(build_state))
    print(json.dumps(build_state, indent=2, sort_keys=True))
    return 0


def expected_numbered(campaign: Mapping[str, Any], case: Mapping[str, Any]) -> list[str]:
    key = "point_order" if case["mode"] == "point" else "science_order"
    return list(campaign["numbered_config_contract"][key])


def verify_overlay(path: Path, case: Mapping[str, Any], binary: Path) -> None:
    try:
        import yaml
        actual = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignError(f"cannot parse materialized overlay {path}: {exc}") from exc
    expected = {
        "reduce": {
            "jobkey": case["jobkey"],
            "steps": {0: {"path": str(binary), "config": {"low_level": {
                "coadd": {"enabled": case["coadd"]},
                "mapmaking": {"method": "naive", "coverage_cut": case["coverage_cut"]},
                "noise_maps": {"enabled": True, "n_noise_maps": 64,
                               "randomize_dets": False, "write_realizations": True,
                               "products": {"enabled": case["products_enabled"],
                                            "apply_empirical_weights": False}},
                "post_processing": {"map_filtering": {"enabled": False},
                                    "source_finding": {"enabled": False}},
                "runtime": {"n_threads": case["threads"],
                            "parallel_policy": case["parallel_policy"], "verbose": True},
                "timestream": {"fruit_loops": {"enabled": False}},
            }}}},
        }
    }
    if actual != expected:
        fail(f"materialized expert overlay differs semantically for {case['id']}")


def verify_empty_expert(path: Path) -> None:
    """Require the native kit's expert role to contribute no effective leaf."""
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignError(f"cannot parse original expert overlay {path}: {exc}") from exc
    expected = {"reduce": {"steps": {0: {"config": {"low_level": {}}}}}}
    if value != expected:
        fail(f"installed original expert overlay is not the exact empty role: {path}")


def verify_marker(path: Path, case: Mapping[str, Any], campaign: Mapping[str, Any]) -> None:
    try:
        import yaml
        marker = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise CampaignError(f"cannot parse TolProj marker {path}: {exc}") from exc
    expected_mode = "point" if case["mode"] == "point" else "science"
    record_key = "tolproj_point_record_id" if expected_mode == "point" else "tolproj_science_record_id"
    checks = {
        "schema_version": "tolproj-installed-citlali-refactor-kit-v2",
        "kit_version": campaign["authority"]["tolproj_bundle"],
        "bundle": "phase4_1_v2_1", "mode": expected_mode,
        "record_id": campaign["authority"][record_key],
        "source_commit": campaign["authority"]["tolproj_bundle_source_commit"],
    }
    differences = {key: (marker.get(key), value) for key, value in checks.items()
                   if marker.get(key) != value}
    if differences:
        fail(f"TolProj marker differs for {case['id']}: {differences}")


def runtime_context_merge(case_dir: Path, expected_files: Sequence[str]) -> dict[str, Any]:
    try:
        from tolteca.utils.runtime_context import RuntimeContext
    except Exception as exc:
        raise CampaignError(f"cannot import deployed TolTECA RuntimeContext: {exc}") from exc
    context = RuntimeContext(case_dir)
    actual_files = [Path(path).name for path in context.config_backend.config_files]
    if actual_files != list(expected_files):
        fail(f"RuntimeContext source order differs in {case_dir}: "
             f"actual={actual_files}, expected={list(expected_files)}")
    config = dict(context.config)
    config.pop("runtime_info", None)
    for key in list(config):
        if str(key).startswith("__"):
            config.pop(key)
    return config


def normalized_merge(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): normalized_merge(child) for key, child in value.items()
                if str(key) != "runtime_info" and not str(key).startswith("__")}
    if isinstance(value, list):
        return [normalized_merge(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def contained(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def regular_path(path: Path, *, allowed_roots: Sequence[Path], label: str) -> Path:
    if path.is_symlink():
        fail(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CampaignError(f"{label} is absent: {path}") from exc
    roots = [root.resolve(strict=True) for root in allowed_roots]
    if not contained(resolved, roots):
        fail(f"{label} escapes its authorized roots: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        fail(f"{label} is not a regular file: {resolved}")
    return resolved


def explicit_external_authority(path_value: Any, *, request_root: Path,
                                label: str) -> Path:
    if not isinstance(path_value, str) or "\n" in path_value or "\r" in path_value \
            or "\\" in path_value or not Path(path_value).is_absolute():
        fail(f"{label} must be an explicit absolute path")
    resolved = regular_path(Path(path_value), allowed_roots=(Path("/"),), label=label)
    resolved_request = request_root.resolve(strict=True)
    if resolved == resolved_request or resolved_request in resolved.parents:
        fail(f"{label} must not be derived from the campaign request tree: {resolved}")
    return resolved


def write_sha_manifest(paths: Iterable[Path], output: Path,
                       *, allowed_roots: Sequence[Path]) -> None:
    records = []
    resolved_paths = {
        regular_path(item, allowed_roots=allowed_roots,
                     label="integrity manifest target") for item in paths
    }
    for path in sorted(resolved_paths, key=str):
        if any(token in str(path) for token in ("\n", "\r", "\\")):
            fail(f"integrity manifest target has an unsupported path: {path}")
        records.append(f"{sha256(path)}  {path}\n")
    write_new(output, "".join(records).encode())


def command_prepare_cases(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    build_state = require_build(root)
    identity(values, campaign, "prepared")
    projects = root / "projects"
    complete = root / "state" / "cases.json"
    case_state_paths = [
        projects, complete, root / "records" / "analysis-self-check.txt",
        *(root / "records" / case_id for case_id in CASE_IDS),
    ]
    if any(os.path.lexists(path) for path in case_state_paths):
        fail("case project state already exists; stop for owner inspection")
    raw_paths, _, _ = validate_raw_authority(root, values)
    for raw_manifest in raw_paths.values():
        raw_manifest.chmod(0o444)
    projects.mkdir(mode=0o755)
    package = frozen_package(root)
    analysis = package / "SCI-MAP-001-analysis.py"
    frozen_owner = root / "owner-values.json"
    python = Path(values["unity_python"]).resolve(strict=True)
    source = Path(values["unity_source_checkout"]).resolve(strict=True)
    tolproj = Path(values["tolproj_executable"]).resolve(strict=True)
    site = Path(values["tolproj_site_config"]).resolve(strict=True)
    binary = root / "bin" / "citlali"
    self_check = (
        str(python), str(analysis), "self-check",
        "--campaign", str(package / "campaign.json"),
        "--product-contracts", str(source / "validation/product_contracts.json"),
        "--source-root", str(source),
    )
    run_logged(self_check, cwd=package, transcript=root / "records" / "analysis-self-check.txt")

    from tolproj.reduction_runtime import freeze_reduction_executable
    case_records = []
    for case_id in CASE_IDS:
        case = case_by_id(campaign, case_id)
        raw_manifest = raw_paths[case_id]
        project_source = values["point_project"] if case["mode"] == "point" \
            else values["science_project"]
        record_dir = root / "records" / case_id
        record_dir.mkdir(mode=0o755)
        duplicate = (
            str(tolproj), "duplicate", project_source, case_id,
            "--destination-root", str(projects), "--config", str(site), "--refactor",
        )
        run_logged(duplicate, cwd=projects,
                   transcript=record_dir / "01-duplicate-transcript.txt")
        project = projects / case_id
        if not project.is_dir():
            fail(f"TolProj did not create case project: {project}")
        if case["mode"] == "point":
            case_dir = project / "audit-pointings"
            setup = (
                str(tolproj), "setup-pointing-reductions", str(project),
                "--config", str(site), "--refactor", "--source",
                values["point_source_filter"], "--pointings-dir", "audit-pointings",
                "--apt-dir", values["point_apt_dir"], "--cpus", str(case["cpus"]),
                "--time", campaign["fixed_execution"]["case_time"],
                "--mem", campaign["fixed_execution"]["case_memory"],
            )
        else:
            user_root = project / "sci_map_001"
            case_dir = user_root / values["science_source_basename"]
            setup = (
                str(tolproj), "setup-science-reductions", str(project),
                "--config", str(site), "--refactor", "--user", "sci_map_001",
                "--pointing-reduction", values["science_pointing_reduction"],
                "--apt-product", "matched", "--cpus", str(case["cpus"]),
                "--time", campaign["fixed_execution"]["case_time"],
                "--mem", campaign["fixed_execution"]["case_memory"],
            )
        run_logged(setup, cwd=project, transcript=record_dir / "02-setup-transcript.txt")
        if not case_dir.is_dir() or not (case_dir / "02_redu.sh").is_file():
            fail(f"TolProj did not create the expected case reduction: {case_dir}")
        if case["mode"] == "science":
            actual_sources = sorted(path.name for path in (project / "sci_map_001").iterdir()
                                    if path.is_dir())
            if actual_sources != [values["science_source_basename"]]:
                fail(f"science setup produced an unpinned source set: {actual_sources}")

        expert_name = "99_pointing_expert_overrides.yaml" if case["mode"] == "point" \
            else "99_science_expert_overrides.yaml"
        expert = case_dir / expert_name
        marker = case_dir / ".citlali_refactor_kit.yaml"
        verify_marker(marker, case, campaign)
        verify_empty_expert(expert)
        copy_new(expert, record_dir / "expert-original.yaml")
        materialized = record_dir / "expert-materialized.yaml"
        materialize = (
            str(python), str(analysis), "materialize-case",
            "--campaign", str(package / "campaign.json"), "--case-id", case_id,
            "--owner-values", str(frozen_owner), "--output", str(materialized),
        )
        run_logged(materialize, cwd=package,
                   transcript=record_dir / "03-materialize-transcript.txt")
        verify_overlay(materialized, case, binary)
        expert.chmod(0o644)
        shutil.copyfile(materialized, expert)
        if expert.read_bytes() != materialized.read_bytes():
            fail(f"installed expert overlay differs for {case_id}")
        expert.chmod(0o444)

        frozen = freeze_reduction_executable(case_dir)
        if frozen.source.resolve() != binary.resolve() or frozen.sha256 != build_state["binary_sha256"]:
            fail(f"TolProj froze the wrong candidate executable for {case_id}")
        if sha256(frozen.snapshot) != build_state["binary_sha256"]:
            fail(f"TolProj snapshot digest differs for {case_id}")
        expected_files = expected_numbered(campaign, case)
        runtime_merged = runtime_context_merge(case_dir, expected_files)

        merged_yaml = record_dir / "merged.yaml"
        merge_report = record_dir / "merge-report.json"
        merge_command = (
            str(python), str(source / "tools/config/tolteca_mode_kit.py"), "merge",
            "--mode", case["mode"], "--mode-dir", str(case_dir),
            "--manifest", str(source / "config/tolteca/v2/manifest.yaml"),
            "--leaf-contract", str(source / "tools/config/config_leaf_contract_resolved.json"),
            "--yaml-out", str(merged_yaml), "--json-out", str(merge_report),
        )
        run_logged(merge_command, cwd=source,
                   transcript=record_dir / "04-merge-transcript.txt")
        import yaml
        tool_merged = yaml.safe_load(merged_yaml.read_text(encoding="utf-8")) or {}
        if normalized_merge(runtime_merged) != normalized_merge(tool_merged):
            fail(f"TolTECA RuntimeContext and deterministic merge disagree for {case_id}")
        write_new(record_dir / "runtime-context-merged.json",
                  json_bytes(normalized_merge(runtime_merged)))

        preflight = record_dir / f"{case_id}-case-inputs.json"
        preflight_command = (
            str(python), str(analysis), "preflight-case",
            "--campaign", str(package / "campaign.json"),
            "--case-id", case_id, "--mode", case["mode"],
            "--case-dir", str(case_dir), "--merged", str(merged_yaml),
            "--source-root", str(source),
            "--vendor-manifest", str(root / "tolproj-kit/vendor.yaml"),
            "--bundle-manifest", str(root / "tolproj-kit/phase4_1_v2_1-manifest.yaml"),
            "--canonical-manifest", str(root / "governing/config/tolteca/v2/manifest.yaml"),
            "--product-contracts", str(root / "governing/validation/product_contracts.json"),
            "--marker", str(marker), "--owner-values", str(frozen_owner),
            "--raw-input-manifest", str(raw_manifest),
            "--output", str(preflight),
        )
        run_logged(preflight_command, cwd=package,
                   transcript=record_dir / "05-preflight-transcript.txt")
        validate_command = (str(tolproj), "validate-reduction", str(case_dir))
        run_logged(validate_command, cwd=case_dir,
                   transcript=record_dir / "06-validate-reduction.txt")
        numbered = [case_dir / name for name in expected_files]
        integrity_paths = [*numbered, marker, case_dir / "02_redu.sh", preflight,
                           merged_yaml, merge_report, materialized,
                           record_dir / "expert-original.yaml", raw_manifest]
        integrity_paths.extend(path for path in (case_dir / ".tolproj").rglob("*")
                               if path.is_file() and not path.is_symlink())
        integrity = record_dir / "pre-submit-sha256.txt"
        write_sha_manifest(integrity_paths, integrity,
                           allowed_roots=(case_dir, record_dir,
                                          root / "raw-input-manifests"))
        case_record = {
            "schema_version": "sci-map-unity-prepared-case-v1",
            "candidate_sha": EXPECTED_CANDIDATE,
            "recorded_at": utc_now(),
            "case_id": case_id, "mode": case["mode"], "case_dir": str(case_dir),
            "numbered_sources": expected_files,
            "snapshot": str(frozen.snapshot), "snapshot_sha256": frozen.sha256,
            "master_binary": str(binary), "master_binary_sha256": build_state["binary_sha256"],
            "raw_input_manifest": str(raw_manifest),
            "raw_input_manifest_sha256": sha256(raw_manifest),
            "preflight_manifest": str(preflight), "integrity_manifest": str(integrity),
        }
        write_new(record_dir / "case.json", json_bytes(case_record))
        case_records.append(case_record)
    write_new(complete, json_bytes({
        "schema_version": "sci-map-unity-cases-state-v1",
        "candidate_sha": EXPECTED_CANDIDATE, "recorded_at": utc_now(),
        "cases": case_records,
    }))
    print(f"prepared and preflighted {len(case_records)} isolated cases")
    return 0


def slurm_options(values: Mapping[str, str], campaign: Mapping[str, Any],
                  *, cpus: int, memory: str, time: str, job_name: str,
                  chdir: Path, stdout: Path, stderr: Path) -> list[str]:
    options = [
        "sbatch", "--wait", "--parsable", "--partition=" + campaign["fixed_execution"]["partition"],
        "--account=" + values["slurm_account"],
        f"--export=ALL,OMP_NUM_THREADS={cpus}",
        f"--cpus-per-task={cpus}", "--mem=" + memory, "--time=" + time,
        "--job-name=" + job_name, "--chdir=" + str(chdir),
        "--output=" + str(stdout), "--error=" + str(stderr),
    ]
    for key, option in (("slurm_qos", "--qos="), ("slurm_constraint", "--constraint="),
                        ("slurm_reservation", "--reservation=")):
        if values[key]:
            options.append(option + values[key])
    return options


def load_cases_state(root: Path) -> list[dict[str, Any]]:
    value = read_json(root / "state" / "cases.json")
    records = value.get("cases") if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping) \
            or value.get("schema_version") != "sci-map-unity-cases-state-v1" \
            or value.get("candidate_sha") != EXPECTED_CANDIDATE \
            or not isinstance(value.get("recorded_at"), str):
        fail("prepared case state identity is absent or differs")
    if not isinstance(records, list) or [row.get("case_id") for row in records] != list(CASE_IDS):
        fail("prepared case state is absent or incomplete")
    build = require_build(root)
    output: list[dict[str, Any]] = []
    for row in records:
        case_id = row["case_id"]
        immutable_path = root / "records" / case_id / "case.json"
        immutable = read_json(immutable_path)
        if immutable != row:
            fail(f"case state differs from immutable record: {case_id}")
        if row.get("schema_version") != "sci-map-unity-prepared-case-v1" \
                or row.get("candidate_sha") != EXPECTED_CANDIDATE \
                or not isinstance(row.get("recorded_at"), str):
            fail(f"case record identity differs: {case_id}")
        expected_root = (root / "projects" / case_id).resolve(strict=True)
        case_dir = Path(row["case_dir"])
        if case_dir.is_symlink():
            fail(f"case reduction root is a symlink: {case_id}")
        resolved_case = case_dir.resolve(strict=True)
        if expected_root not in resolved_case.parents:
            fail(f"case reduction root escapes its case-owned project: {case_id}")
        if Path(row["master_binary"]).resolve(strict=True) != (root / "bin/citlali").resolve() \
                or row.get("master_binary_sha256") != build["binary_sha256"]:
            fail(f"case master binary identity differs: {case_id}")
        snapshot = regular_path(Path(row["snapshot"]), allowed_roots=(resolved_case,),
                                label=f"{case_id} executable snapshot")
        if sha256(snapshot) != row.get("snapshot_sha256") \
                or row.get("snapshot_sha256") != build["binary_sha256"]:
            fail(f"case executable snapshot differs: {case_id}")
        preflight = regular_path(
            Path(row["preflight_manifest"]),
            allowed_roots=(root / "records" / case_id,),
            label=f"{case_id} preflight manifest")
        raw_manifest = regular_path(
            Path(row.get("raw_input_manifest", "")),
            allowed_roots=(root / "raw-input-manifests",),
            label=f"{case_id} raw-input manifest")
        if raw_manifest != (root / "raw-input-manifests" / f"{case_id}.json").resolve() \
                or row.get("raw_input_manifest_sha256") != sha256(raw_manifest):
            fail(f"case raw-input authority differs: {case_id}")
        preflight_record = read_json(preflight)
        preflight_paths = preflight_record.get("paths") \
            if isinstance(preflight_record, Mapping) else None
        preflight_hashes = preflight_record.get("sha256") \
            if isinstance(preflight_record, Mapping) else None
        if not isinstance(preflight_record, Mapping) \
                or preflight_record.get("schema_version") != \
                "sci-map-unity-case-preflight-v1" \
                or preflight_record.get("candidate_sha") != EXPECTED_CANDIDATE \
                or preflight_record.get("result") != "pass" \
                or not isinstance(preflight_paths, Mapping) \
                or preflight_paths.get("raw_input_manifest") != str(raw_manifest) \
                or not isinstance(preflight_hashes, Mapping) \
                or preflight_hashes.get("raw_input_manifest") != sha256(raw_manifest):
            fail(f"case preflight/raw-input binding differs: {case_id}")
        integrity = regular_path(Path(row["integrity_manifest"]),
                                 allowed_roots=(root / "records" / case_id,),
                                 label=f"{case_id} integrity manifest")
        verify_sha_manifest(integrity, allowed_roots=(
            resolved_case, root / "records" / case_id,
            root / "raw-input-manifests"))
        verify_case_cpu_script(resolved_case, CASE_CPUS[case_id])
        output.append(dict(row))
    return output


def verify_sha_manifest(path: Path, *, allowed_roots: Sequence[Path]) -> None:
    seen: set[Path] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (/[^\n\r]+)", line)
        if match is None:
            fail(f"invalid SHA manifest line {line_number}: {path}")
        expected, raw_path = match.groups()
        target = regular_path(Path(raw_path), allowed_roots=allowed_roots,
                              label=f"SHA manifest target line {line_number}")
        if target in seen or sha256(target) != expected:
            fail(f"SHA manifest target differs/repeats: {target}")
        seen.add(target)
    if not seen:
        fail(f"SHA manifest is empty: {path}")


def verify_relative_sha_inventory(path: Path, *, root: Path) -> None:
    root = root.resolve(strict=True)
    manifest = regular_path(path, allowed_roots=(root,), label="relative SHA inventory")
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n\r]+)", line)
        if match is None:
            fail(f"invalid relative SHA inventory line {line_number}: {manifest}")
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts \
                or "\\" in relative \
                or relative in seen or relative_path == manifest.relative_to(root):
            fail(f"unsafe/repeated relative SHA inventory path: {relative}")
        target = regular_path(root / relative_path, allowed_roots=(root,),
                              label=f"relative SHA inventory target {relative}")
        if sha256(target) != expected:
            fail(f"relative SHA inventory target differs: {target}")
        seen.add(relative)
    entries = list(root.rglob("*"))
    symlinks = [item for item in entries if item.is_symlink()]
    if symlinks:
        fail(f"relative SHA inventory tree contains symlinks: {symlinks}")
    actual = {
        item.relative_to(root).as_posix() for item in entries
        if item.is_file() and not item.is_symlink() and item != manifest
    }
    if not seen or seen != actual:
        fail("relative SHA inventory is not exhaustive; "
             f"missing={sorted(actual - seen)}, stale={sorted(seen - actual)}")


def verify_case_cpu_script(case_dir: Path, cpus: int) -> None:
    script = regular_path(case_dir / "02_redu.sh", allowed_roots=(case_dir,),
                          label="case reduction script")
    pattern = re.compile(
        rf"^\s*#SBATCH\s+--cpus-per-task(?:=|\s+){cpus}\s*$", re.MULTILINE)
    matches = pattern.findall(script.read_text(encoding="utf-8"))
    all_cpu = re.findall(r"^\s*#SBATCH\s+--cpus-per-task(?:=|\s+)\d+\s*$",
                         script.read_text(encoding="utf-8"), re.MULTILINE)
    if len(matches) != 1 or len(all_cpu) != 1:
        fail(f"02_redu.sh does not contain one pinned CPU directive: {case_dir}")


def parse_submit_record(path: Path, *, allowed_root: Path,
                        expected_rc: int) -> tuple[str, int]:
    record = regular_path(path, allowed_roots=(allowed_root,),
                          label="Slurm submit record")
    lines = record.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or not lines[0].startswith("job_ref=") \
            or not lines[1].startswith("submit_rc="):
        fail(f"Slurm submit record shape differs: {record}")
    job_ref = lines[0].removeprefix("job_ref=")
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9._-]+)?", job_ref)
    if match is None:
        fail(f"Slurm submit record has no numeric job identity: {record}")
    try:
        submit_rc = int(lines[1].removeprefix("submit_rc="))
    except ValueError as exc:
        raise CampaignError(f"Slurm submit record has an invalid return code: {record}") from exc
    if submit_rc != expected_rc:
        fail(f"Slurm submit status differs in {record}: {submit_rc} != {expected_rc}")
    return match.group(1), submit_rc


def parse_slurm_accounting(path: Path, *, allowed_root: Path, job_id: str,
                           job_name: str, partition: str, cpus: int,
                           state: str, exit_code: str) -> list[dict[str, str]]:
    accounting = regular_path(path, allowed_roots=(allowed_root,),
                              label="Slurm accounting record")
    lines = accounting.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0].split("|") != list(SLURM_FIELDS):
        fail(f"Slurm accounting header/data differ: {accounting}")
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(lines[1:], 2):
        values = line.split("|")
        if len(values) != len(SLURM_FIELDS):
            fail(f"Slurm accounting row {line_number} has the wrong field count")
        row = dict(zip(SLURM_FIELDS, values))
        if not row["JobIDRaw"] or any(
                existing["JobIDRaw"] == row["JobIDRaw"] for existing in rows):
            fail(f"Slurm accounting has an empty/repeated JobIDRaw at row {line_number}")
        rows.append(row)
    top_rows = [row for row in rows if row["JobIDRaw"] == job_id]
    if len(top_rows) != 1:
        fail(f"Slurm accounting does not contain exactly one top job {job_id}")
    top = top_rows[0]
    expected = {
        "JobName": job_name, "Partition": partition, "AllocCPUS": str(cpus),
        "State": state, "ExitCode": exit_code,
    }
    differences = {key: (top[key], value) for key, value in expected.items()
                   if top[key] != value}
    if differences:
        fail(f"Slurm accounting top-job identity differs: {differences}")
    if not top["NodeList"] or not top["Elapsed"] or not top["ReqMem"]:
        fail("Slurm accounting lacks allocation/runtime fields for the top job")
    for name in ("Submit", "Start", "End"):
        if SLURM_TIMESTAMP_RE.fullmatch(top[name]) is None:
            fail(f"Slurm accounting top-job {name} timestamp differs: {top[name]!r}")
    return rows


def require_utc_timestamp(path: Path, *, allowed_root: Path, label: str) -> Path:
    timestamp = regular_path(path, allowed_roots=(allowed_root,), label=label)
    value = timestamp.read_text(encoding="utf-8").strip()
    if RFC3339_UTC_RE.fullmatch(value) is None:
        fail(f"{label} is not one RFC3339 UTC timestamp: {timestamp}")
    return timestamp


def require_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    missing = sorted(set(keys) - set(value))
    extra = sorted(set(value) - set(keys))
    if missing or extra:
        fail(f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def exact_float(record: Any, label: str, *, positive: bool = False) -> float:
    node = require_keys(record, ("numeric", "hex", "encoding"), label)
    if node["encoding"] != "binary64-max-digits10-and-c99-hexfloat":
        fail(f"{label} encoding differs")
    if not isinstance(node["numeric"], str) or not isinstance(node["hex"], str):
        fail(f"{label} numeric and hex renderings must be strings")
    try:
        numeric = float(node["numeric"])
        hexadecimal = float.fromhex(node["hex"])
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{label} is not an exact binary64 record") from exc
    if struct.pack("=d", numeric) != struct.pack("=d", hexadecimal):
        fail(f"{label} decimal and C99 hex values differ")
    if not __import__("math").isfinite(numeric) or (positive and numeric <= 0.0):
        fail(f"{label} must be {'finite positive' if positive else 'finite'}")
    return numeric


def ordered_unique_integers(value: Any, label: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, list) or not value or any(
            not isinstance(item, int) or isinstance(item, bool) for item in value):
        fail(f"{label} must be a nonempty integer array")
    if value != sorted(set(value)):
        fail(f"{label} must be strictly ordered and unique")
    minimum = 1 if positive else 0
    if any(item < minimum for item in value):
        fail(f"{label} contains an out-of-domain integer")
    return value


def validate_raw_input_manifest(path: Path, mode: str,
                                values: Mapping[str, str]) -> dict[str, Any]:
    expected_obs = [152389] if mode == "point" else [152390, 152392]
    expected_frame = "altaz" if mode == "point" else "fk5"
    request_root = Path(values["request_root"])
    manifest_path = regular_path(path, allowed_roots=(path.parent,),
                                 label=f"{mode} raw-input manifest")
    raw = require_keys(read_json(manifest_path), (
        "schema_version", "request_id", "candidate_sha", "mode", "observations",
        "arrays", "producer", "source_records", "memberships",
    ), f"{mode} raw-input manifest")
    if raw["schema_version"] != RAW_MANIFEST_SCHEMA or raw["request_id"] != REQUEST_ID:
        fail(f"{mode} raw-input manifest identity differs")
    # Candidate is mandatory: never use get(..., expected) here.
    if raw["candidate_sha"] != EXPECTED_CANDIDATE:
        fail(f"{mode} raw-input manifest candidate SHA differs")
    if raw["mode"] != mode or raw["observations"] != expected_obs:
        fail(f"{mode} raw-input manifest observation identity differs")
    if raw["arrays"] != list(ARRAYS):
        fail(f"{mode} raw-input manifest array order differs")

    producer = require_keys(raw["producer"],
                            ("identity", "program_path", "program_sha256", "invocation"),
                            f"{mode} producer")
    if not isinstance(producer["identity"], str) or not producer["identity"].strip() \
            or producer["identity"] != producer["identity"].strip() \
            or len(producer["identity"]) > 256:
        fail(f"{mode} producer identity is unresolved")
    if not isinstance(producer["invocation"], list) or not producer["invocation"] \
            or any(not isinstance(item, str) or not item for item in producer["invocation"]):
        fail(f"{mode} producer invocation is unresolved")
    producer_path = explicit_external_authority(
        producer["program_path"], request_root=request_root,
        label=f"{mode} producer program")
    if not SHA256_RE.fullmatch(str(producer["program_sha256"])) \
            or sha256(producer_path) != producer["program_sha256"]:
        fail(f"{mode} producer program digest differs")

    source_rows = raw["source_records"]
    if not isinstance(source_rows, list) or len(source_rows) < len(SOURCE_ROLES):
        fail(f"{mode} source_records are incomplete")
    sources: dict[str, dict[str, Any]] = {}
    source_paths: list[Path] = []
    for index, value in enumerate(source_rows):
        row = dict(require_keys(value, (
            "id", "role", "path", "size_bytes", "sha256", "obsnums", "arrays",
            "networks",
        ), f"{mode} source_records[{index}]"))
        identifier = row["id"]
        if not isinstance(identifier, str) or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identifier) is None:
            fail(f"{mode} source record id is invalid")
        if identifier in sources or row["role"] not in SOURCE_ROLES:
            fail(f"{mode} source record identity/role is invalid: {identifier}")
        obsnums = ordered_unique_integers(
            row["obsnums"], f"{mode} {identifier} obsnums", positive=True)
        if not set(obsnums).issubset(expected_obs):
            fail(f"{mode} source record obsnums differ: {identifier}")
        if not isinstance(row["arrays"], list) or not row["arrays"] \
                or any(not isinstance(item, str) or item not in ARRAYS
                       for item in row["arrays"]) \
                or row["arrays"] != [array for array in ARRAYS if array in row["arrays"]]:
            fail(f"{mode} source record arrays differ: {identifier}")
        ordered_unique_integers(row["networks"], f"{mode} {identifier} networks")
        source_path = explicit_external_authority(
            row["path"], request_root=request_root,
            label=f"{mode} source record {identifier}")
        if not isinstance(row["size_bytes"], int) or isinstance(row["size_bytes"], bool) \
                or row["size_bytes"] <= 0 \
                or source_path.stat().st_size != row["size_bytes"]:
            fail(f"{mode} source record size differs: {identifier}")
        if not SHA256_RE.fullmatch(str(row["sha256"])) or sha256(source_path) != row["sha256"]:
            fail(f"{mode} source record digest differs: {identifier}")
        row["resolved_path"] = str(source_path)
        sources[identifier] = row
        source_paths.append(source_path)

    memberships = raw["memberships"]
    expected_pairs = [(obsnum, array) for obsnum in expected_obs for array in ARRAYS]
    if not isinstance(memberships, list) or len(memberships) != len(expected_pairs):
        fail(f"{mode} membership cardinality differs")
    membership_index: dict[tuple[int, str], dict[str, Any]] = {}
    scan_orders_by_observation: dict[int, list[dict[str, Any]]] = {}
    used_sources: set[str] = set()
    for index, (value, expected_pair) in enumerate(zip(memberships, expected_pairs)):
        row = dict(require_keys(value, (
            "obsnum", "array", "networks", "record_order",
            "projection_record_count", "scan_order", "detector_order", "source_refs",
            "projection",
        ), f"{mode} memberships[{index}]"))
        pair = (row["obsnum"], row["array"])
        if pair != expected_pair:
            fail(f"{mode} membership order differs at {index}: {pair} != {expected_pair}")
        networks = ordered_unique_integers(row["networks"], f"{mode} {pair} networks")
        if row["record_order"] != \
                "scan-major-detector-major-sample-minor-cartesian-v1":
            fail(f"{mode} {pair} processed-term record order differs")
        if not isinstance(row["projection_record_count"], int) \
                or isinstance(row["projection_record_count"], bool) \
                or row["projection_record_count"] <= 0:
            fail(f"{mode} {pair} projection_record_count is invalid")
        scan_order = row["scan_order"]
        if not isinstance(scan_order, list) or not scan_order:
            fail(f"{mode} {pair} scan_order is empty")
        parsed_scans: list[dict[str, Any]] = []
        seen_scan_identities: set[str] = set()
        for scan_index, scan_value in enumerate(scan_order):
            scan = dict(require_keys(
                scan_value, ("scan_index", "identity", "sample_count"),
                f"{mode} {pair} scan_order[{scan_index}]"))
            if not isinstance(scan["scan_index"], int) \
                    or isinstance(scan["scan_index"], bool) \
                    or scan["scan_index"] != scan_index:
                fail(f"{mode} {pair} scan indices are not zero-based and contiguous")
            scan_identity = scan["identity"]
            if not isinstance(scan_identity, str) or not scan_identity.strip() \
                    or scan_identity != scan_identity.strip() or len(scan_identity) > 256 \
                    or "\n" in scan_identity or "\r" in scan_identity \
                    or scan_identity in seen_scan_identities:
                fail(f"{mode} {pair} scan identity is invalid/repeated")
            if not isinstance(scan["sample_count"], int) \
                    or isinstance(scan["sample_count"], bool) \
                    or scan["sample_count"] <= 0:
                fail(f"{mode} {pair} scan sample_count is invalid")
            seen_scan_identities.add(scan_identity)
            parsed_scans.append(scan)
        prior_scans = scan_orders_by_observation.setdefault(pair[0], parsed_scans)
        if parsed_scans != prior_scans:
            fail(f"{mode} observation {pair[0]} scan order differs across arrays")
        detector_order = row["detector_order"]
        if not isinstance(detector_order, list) or not detector_order:
            fail(f"{mode} {pair} detector_order is empty")
        parsed_detectors: list[dict[str, Any]] = []
        seen_detector_uids: set[str] = set()
        for detector_index, detector_value in enumerate(detector_order):
            detector = dict(require_keys(
                detector_value, ("detector_index", "detector_uid", "network"),
                f"{mode} {pair} detector_order[{detector_index}]"))
            if not isinstance(detector["detector_index"], int) \
                    or isinstance(detector["detector_index"], bool) \
                    or detector["detector_index"] != detector_index:
                fail(f"{mode} {pair} detector indices are not zero-based and contiguous")
            detector_uid = detector["detector_uid"]
            if not isinstance(detector_uid, str) or not detector_uid.strip() \
                    or detector_uid != detector_uid.strip() or len(detector_uid) > 256 \
                    or "\n" in detector_uid or "\r" in detector_uid \
                    or detector_uid in seen_detector_uids:
                fail(f"{mode} {pair} detector UID is invalid/repeated")
            if not isinstance(detector["network"], int) \
                    or isinstance(detector["network"], bool) \
                    or detector["network"] not in networks:
                fail(f"{mode} {pair} detector network is outside membership")
            seen_detector_uids.add(detector_uid)
            parsed_detectors.append(detector)
        if {detector["network"] for detector in parsed_detectors} != set(networks):
            fail(f"{mode} {pair} detector order does not cover every network")
        calculated_record_count = len(parsed_detectors) * sum(
            scan["sample_count"] for scan in parsed_scans)
        if row["projection_record_count"] != calculated_record_count:
            fail(f"{mode} {pair} projection_record_count differs from membership")
        row["scan_order"] = parsed_scans
        row["detector_order"] = parsed_detectors
        refs = require_keys(row["source_refs"], SOURCE_ROLES, f"{mode} {pair} source_refs")
        for role in SOURCE_ROLES:
            identifiers = refs[role]
            if not isinstance(identifiers, list) or not identifiers \
                    or any(not isinstance(item, str) or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", item) is None
                           for item in identifiers) \
                    or len(identifiers) != len(set(identifiers)):
                fail(f"{mode} {pair} {role} references are incomplete/repeated")
            referenced_networks: set[int] = set()
            for identifier in identifiers:
                source = sources.get(identifier)
                if source is None or source["role"] != role:
                    fail(f"{mode} {pair} {role} references the wrong source: {identifier}")
                if pair[0] not in source["obsnums"] or pair[1] not in source["arrays"]:
                    fail(f"{mode} {pair} source does not declare membership: {identifier}")
                referenced_networks.update(source["networks"])
                used_sources.add(identifier)
            if not set(networks).issubset(referenced_networks):
                fail(f"{mode} {pair} {role} does not cover every declared network")
            if role in ("raw_timestream", "kids_fit_report") \
                    and referenced_networks != set(networks):
                fail(f"{mode} {pair} {role} network membership is not exact")
        projection = require_keys(row["projection"], (
            "identity_digest", "grouping", "stokes", "frame", "map_rows", "map_cols",
            "sample_rate_hz", "fwhm_arcsec", "target",
        ), f"{mode} {pair} projection")
        if re.fullmatch(r"canonical-hexfloat-sha256-v1:[0-9a-f]{64}",
                        str(projection["identity_digest"])) is None:
            fail(f"{mode} {pair} projection identity digest differs")
        if projection["grouping"] != "array" or projection["stokes"] != "I" \
                or projection["frame"] != expected_frame:
            fail(f"{mode} {pair} projection identity differs")
        for dimension in ("map_rows", "map_cols"):
            if not isinstance(projection[dimension], int) \
                    or isinstance(projection[dimension], bool) \
                    or projection[dimension] <= 0:
                fail(f"{mode} {pair} {dimension} is invalid")
        sample_rate = exact_float(projection["sample_rate_hz"],
                                  f"{mode} {pair} sample_rate_hz", positive=True)
        exact_float(projection["fwhm_arcsec"], f"{mode} {pair} fwhm_arcsec", positive=True)
        target = require_keys(projection["target"], ("frame", "axis1", "axis2", "unit"),
                              f"{mode} {pair} target")
        if target["frame"] != expected_frame or target["unit"] != "deg":
            fail(f"{mode} {pair} target frame/unit differs")
        exact_float(target["axis1"], f"{mode} {pair} target axis1")
        exact_float(target["axis2"], f"{mode} {pair} target axis2")
        row["sample_rate"] = sample_rate
        membership_index[pair] = row
    if used_sources != set(sources):
        fail(f"{mode} raw-input manifest has unused source records: "
             f"{sorted(set(sources) - used_sources)}")
    if {row["role"] for row in sources.values()} != set(SOURCE_ROLES):
        fail(f"{mode} raw-input manifest does not cover every required source role")
    return {
        "path": manifest_path, "sha256": sha256(manifest_path), "mode": mode,
        "sources": sources, "source_paths": source_paths,
        "producer_path": producer_path, "producer_identity": producer["identity"],
        "memberships": membership_index,
    }


def npz_scalar(archive: Mapping[str, Any], name: str) -> Any:
    import numpy as np
    if name not in archive:
        fail(f"sample ledger lacks scalar {name}")
    value = np.asarray(archive[name])
    if value.size != 1:
        fail(f"sample ledger scalar {name} is not scalar")
    return value.reshape(-1)[0].item()


def npz_integer_scalar(archive: Mapping[str, Any], name: str) -> int:
    import numpy as np
    if name not in archive:
        fail(f"sample ledger lacks scalar {name}")
    value = np.asarray(archive[name])
    if value.size != 1 or value.dtype.kind not in ("i", "u"):
        fail(f"sample ledger scalar {name} is not an integer scalar")
    return int(value.reshape(-1)[0].item())


def validate_sample_ledger(path: Path, obsnum: int, array: str,
                           raw_manifest: Mapping[str, Any], root: Path) -> Path:
    try:
        import numpy as np
    except Exception as exc:
        raise CampaignError(f"cannot validate sample ledger without NumPy: {exc}") from exc
    ledger = regular_path(path, allowed_roots=(root / "sample-ledgers",),
                          label=f"sample ledger {obsnum}:{array}")
    membership = raw_manifest["memberships"][(obsnum, array)]
    try:
        archive_context = np.load(ledger, allow_pickle=False)
    except Exception as exc:
        raise CampaignError(f"cannot load independent sample ledger {ledger}: {exc}") from exc
    with archive_context as archive:
        expected_names = set(LEDGER_ARRAYS) | set(LEDGER_SCALARS)
        if set(archive.files) != expected_names:
            fail(f"sample ledger fields differ for {obsnum}:{array}; "
                 f"missing={sorted(expected_names - set(archive.files))}, "
                 f"extra={sorted(set(archive.files) - expected_names)}")
        if str(npz_scalar(archive, "schema_version")) != LEDGER_SCHEMA:
            fail(f"sample ledger schema differs for {obsnum}:{array}")
        if str(npz_scalar(archive, "candidate_sha")) != EXPECTED_CANDIDATE:
            fail(f"sample ledger candidate differs for {obsnum}:{array}")
        if str(npz_scalar(archive, "raw_input_manifest_sha256")) != raw_manifest["sha256"]:
            fail(f"sample ledger raw-manifest binding differs for {obsnum}:{array}")
        producer = str(npz_scalar(archive, "producer_identity"))
        if not producer.strip() or producer != producer.strip() or len(producer) > 256 \
                or producer != raw_manifest["producer_identity"]:
            fail(f"sample ledger producer differs for {obsnum}:{array}")
        bundle_identity = str(npz_scalar(archive, "bundle_identity_digest"))
        if re.fullmatch(r"canonical-hexfloat-sha256-v1:[0-9a-f]{64}",
                        bundle_identity) is None:
            fail(f"sample ledger bundle identity differs for {obsnum}:{array}")
        rows = npz_integer_scalar(archive, "map_rows")
        cols = npz_integer_scalar(archive, "map_cols")
        if (npz_integer_scalar(archive, "obsnum"),
                str(npz_scalar(archive, "array"))) != \
                (obsnum, array):
            fail(f"sample ledger observation/array differs for {obsnum}:{array}")
        projection = membership["projection"]
        if bundle_identity != projection["identity_digest"]:
            fail(f"sample ledger/raw projection identity differs for {obsnum}:{array}")
        ledger_sample_rate = exact_float({
            "numeric": str(npz_scalar(archive, "sample_rate_hz_numeric")),
            "hex": str(npz_scalar(archive, "sample_rate_hz_hex")),
            "encoding": str(npz_scalar(archive, "sample_rate_hz_encoding")),
        }, f"sample ledger {obsnum}:{array} sample_rate_hz", positive=True)
        if struct.pack("=d", ledger_sample_rate) != \
                struct.pack("=d", membership["sample_rate"]):
            fail(f"sample ledger/raw sample rate differs for {obsnum}:{array}")
        if (rows, cols) != (projection["map_rows"], projection["map_cols"]):
            fail(f"sample ledger projection shape differs for {obsnum}:{array}")
        vectors = {name: np.asarray(archive[name]) for name in LEDGER_ARRAYS[:-1]}
        signs = np.asarray(archive["realization_signs"])
        lengths = {name: value.size for name, value in vectors.items()}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values()), 0) <= 0:
            fail(f"sample ledger term cardinality differs for {obsnum}:{array}")
        count = next(iter(lengths.values()))
        if any(value.ndim != 1 for value in vectors.values()):
            fail(f"sample ledger term arrays are not vectors for {obsnum}:{array}")
        dtype_expectations = {
            "row": np.dtype("int64"), "col": np.dtype("int64"),
            "detector_index": np.dtype("int64"), "sample_index": np.dtype("int64"),
            "scan_index": np.dtype("int64"), "geometric_in_bounds": np.dtype("uint8"),
            "upstream_eligible": np.dtype("uint8"), "coefficient": np.dtype("float64"),
            "sample_signal": np.dtype("float64"), "sample_kernel": np.dtype("float64"),
            "sample_interval_s": np.dtype("float64"),
        }
        for name, expected_dtype in dtype_expectations.items():
            if vectors[name].dtype != expected_dtype:
                fail(f"sample ledger {name} dtype differs for {obsnum}:{array}")
        if signs.dtype != np.dtype("int8") or signs.ndim != 2 or signs.shape[1] != 64:
            fail(f"sample ledger realization signs differ for {obsnum}:{array}")
        if not np.all(np.isin(vectors["geometric_in_bounds"], (0, 1))) \
                or not np.all(np.isin(vectors["upstream_eligible"], (0, 1))) \
                or not np.all(np.isin(signs, (-1, 1))):
            fail(f"sample ledger binary/sign values differ for {obsnum}:{array}")
        geometric = vectors["geometric_in_bounds"].astype(bool)
        if np.any(vectors["upstream_eligible"].astype(bool) & ~geometric):
            fail(f"sample ledger upstream eligibility escapes geometry for {obsnum}:{array}")
        if np.any(vectors["row"][geometric] < 0) or np.any(vectors["row"][geometric] >= rows) \
                or np.any(vectors["col"][geometric] < 0) \
                or np.any(vectors["col"][geometric] >= cols):
            fail(f"sample ledger projected coordinates differ for {obsnum}:{array}")
        for name in ("detector_index", "sample_index", "scan_index"):
            if np.any(vectors[name] < 0):
                fail(f"sample ledger {name} contains negative values for {obsnum}:{array}")
        if np.max(vectors["scan_index"], initial=-1) >= signs.shape[0]:
            fail(f"sample ledger scan/sign membership differs for {obsnum}:{array}")
        scans = membership["scan_order"]
        detector_count = len(membership["detector_order"])
        expected_count = detector_count * sum(
            scan["sample_count"] for scan in scans)
        if count != expected_count or count != membership["projection_record_count"]:
            fail(f"sample ledger does not cover the exact raw term set for {obsnum}:{array}")
        cursor = 0
        for scan in scans:
            sample_count = scan["sample_count"]
            for detector_index in range(detector_count):
                end = cursor + sample_count
                if not np.all(vectors["scan_index"][cursor:end] == scan["scan_index"]) \
                        or not np.all(
                            vectors["detector_index"][cursor:end] == detector_index) \
                        or not np.array_equal(
                            vectors["sample_index"][cursor:end],
                            np.arange(sample_count, dtype=np.int64)):
                    fail(f"sample ledger term identity/order differs for {obsnum}:{array}")
                cursor = end
        if cursor != count:
            fail(f"sample ledger term cursor differs for {obsnum}:{array}")
        engine = np.random.RandomState(5489)
        words = engine.randint(
            0, 2 ** 32, size=signs.shape[0] * 64, dtype=np.uint32)
        expected_signs = (2 * (words >> np.uint32(31)).astype(np.int8) - 1).reshape(
            signs.shape[0], 64)
        if not np.array_equal(signs, expected_signs):
            fail(f"sample ledger signs differ from pinned Boost MT19937 for {obsnum}:{array}")
        interval = vectors["sample_interval_s"]
        expected_interval = np.float64(1.0 / ledger_sample_rate)
        if not np.all(np.isfinite(interval)) or not np.all(interval > 0.0) \
                or not np.all(interval == expected_interval):
            fail(f"sample ledger sample interval/rate identity differs for {obsnum}:{array}")
        if count != vectors["row"].size:
            fail(f"sample ledger internal cardinality differs for {obsnum}:{array}")
        eligible = vectors["upstream_eligible"].astype(bool)
        coefficient = vectors["coefficient"]
        contributing = eligible & np.isfinite(coefficient) & (coefficient > 0.0)
        if np.any(eligible & ~np.isfinite(coefficient)) \
                or np.any(contributing & ~np.isfinite(vectors["sample_signal"])) \
                or np.any(contributing & ~np.isfinite(vectors["sample_kernel"])):
            fail(f"sample ledger has a nonfinite successful-run contribution for {obsnum}:{array}")
        if np.any(~np.isfinite(
                coefficient[contributing] * vectors["sample_signal"][contributing])) \
                or np.any(~np.isfinite(
                    coefficient[contributing] * vectors["sample_kernel"][contributing])):
            fail(f"sample ledger weighted contribution overflows for {obsnum}:{array}")
    return ledger


def reconstruction_manifest_bytes(paths: Iterable[Path]) -> bytes:
    unique = sorted(set(paths), key=str)
    if any(any(token in str(path) for token in ("\n", "\r", "\\"))
           for path in unique):
        fail("reconstruction authority contains an unsupported path")
    return "".join(f"{sha256(path)}  {path}\n" for path in unique).encode()


def validate_raw_authority(root: Path,
                           values: Mapping[str, str]) -> tuple[
                               dict[str, Path], dict[str, Any], dict[str, Any]]:
    raw_root = root / "raw-input-manifests"
    raw_paths = {
        case_id: regular_path(
            raw_root / f"{case_id}.json", allowed_roots=(raw_root,),
            label=f"{case_id} raw-input manifest")
        for case_id in CASE_IDS
    }
    point_bytes = [raw_paths[case_id].read_bytes() for case_id in ("P-SEQ", "P-OMP")]
    if point_bytes[0] != point_bytes[1]:
        fail("P-SEQ and P-OMP require byte-identical raw-input manifests")
    science_ids = ("S-C-SEQ", "S-C-OMP", "S-E-SEQ", "S-E-OMP", "S-X-SEQ")
    science_bytes = [raw_paths[case_id].read_bytes() for case_id in science_ids]
    if len(set(science_bytes)) != 1:
        fail("all five science cases require byte-identical raw-input manifests")
    point = validate_raw_input_manifest(raw_paths["P-SEQ"], "point", values)
    science = validate_raw_input_manifest(raw_paths["S-C-SEQ"], "science", values)
    # Validate every copy independently for candidate/mode/path discipline even
    # after byte equality has been proven.
    for case_id in ("P-OMP",):
        validate_raw_input_manifest(raw_paths[case_id], "point", values)
    for case_id in science_ids[1:]:
        validate_raw_input_manifest(raw_paths[case_id], "science", values)
    return raw_paths, point, science


def validate_reconstruction_authority(root: Path, campaign: Mapping[str, Any],
                                      values: Mapping[str, str], *, create: bool) -> Path:
    raw_paths, point, science = validate_raw_authority(root, values)

    ledger_paths: list[Path] = []
    for obsnum, mode_record in ((152389, point), (152390, science), (152392, science)):
        mode = "point" if obsnum == 152389 else "science"
        for array in ARRAYS:
            ledger_paths.append(validate_sample_ledger(
                root / "sample-ledgers" / f"{mode}-{obsnum}-{array}.npz",
                obsnum, array, mode_record, root))
    package = frozen_package(root)
    authority_paths = [
        *raw_paths.values(), *ledger_paths,
        package / "raw-input-manifest.schema.json",
        package / "sample-ledger-contract.json",
        point["producer_path"], science["producer_path"],
        *point["source_paths"], *science["source_paths"],
    ]
    regular = [regular_path(path, allowed_roots=(Path("/"),),
                            label="reconstruction authority input")
               for path in authority_paths]
    payload = reconstruction_manifest_bytes(regular)
    output = root / "manifests" / "SCI-MAP-001-RECONSTRUCTION-AUTHORITY.sha256"
    if create:
        for path in (*raw_paths.values(), *ledger_paths):
            path.chmod(0o444)
        write_new(output, payload)
    else:
        if not output.is_file() or output.read_bytes() != payload:
            fail("immutable reconstruction-authority SHA manifest differs or is absent")
    return output


def command_emit_submit_plan(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    build_state = require_build(root)
    identity(values, campaign, "prepared")
    records = load_cases_state(root)
    plan = root / "plans" / "submit-seven-cases.sh"
    reconstruction_path = root / "manifests" / \
        "SCI-MAP-001-RECONSTRUCTION-AUTHORITY.sha256"
    execution_state_paths = [
        root / "evidence" / case_id for case_id in CASE_IDS
    ] + [
        root / "records" / case_id / name
        for case_id in CASE_IDS
        for name in ("submit-record.txt", "submit-stderr.txt", "slurm-accounting.txt")
    ]
    if os.path.lexists(plan) or os.path.lexists(reconstruction_path) \
            or any(os.path.lexists(path) for path in execution_state_paths):
        fail("submission-plan state already exists; stop for owner inspection")
    reconstruction_authority = validate_reconstruction_authority(
        root, campaign, values, create=True)
    wrapper = frozen_package(root) / "scripts" / "case-job-wrapper.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "",
             "# HUMAN ACTION: this file submits all seven repaired-success cases.",
             "# It was emitted by the campaign driver; the driver did not execute it.", ""]
    for record in records:
        case = case_by_id(campaign, record["case_id"])
        case_id = case["id"]
        case_dir = Path(record["case_dir"])
        evidence = root / "evidence" / case_id
        submission = root / "records" / case_id / "submit-record.txt"
        submission_err = root / "records" / case_id / "submit-stderr.txt"
        sacct = root / "records" / case_id / "slurm-accounting.txt"
        wrapper_command = shlex.join((
            "bash", str(wrapper), str(case_dir), str(evidence),
            record["master_binary"], build_state["binary_sha256"],
            record["snapshot"], record["snapshot_sha256"],
            record["integrity_manifest"], str(reconstruction_authority),
        ))
        options = slurm_options(
            values, campaign, cpus=int(case["cpus"]),
            memory=campaign["fixed_execution"]["case_memory"],
            time=campaign["fixed_execution"]["case_time"],
            job_name=f"sci-map-001-{case_id}", chdir=case_dir,
            stdout=evidence / "slurm-wrapper-%j.out",
            stderr=evidence / "slurm-wrapper-%j.err",
        )
        options.extend(("--wrap", wrapper_command))
        lines.extend([
            f"# {case_id}: repaired expected outcome is exit 0.",
            f"sha256sum -c {shlex.quote(record['integrity_manifest'])}",
            f"sha256sum -c {shlex.quote(str(reconstruction_authority))}",
            f"test ! -e {shlex.quote(str(evidence))}",
            f"test ! -L {shlex.quote(str(evidence))}",
            f"test ! -e {shlex.quote(str(submission))}",
            f"test ! -L {shlex.quote(str(submission))}",
            f"test ! -e {shlex.quote(str(submission_err))}",
            f"test ! -L {shlex.quote(str(submission_err))}",
            f"test ! -e {shlex.quote(str(sacct))}",
            f"test ! -L {shlex.quote(str(sacct))}",
            f"mkdir {shlex.quote(str(evidence))}",
            "set +e",
            f"job_ref=$({shlex.join(options)} 2> {shlex.quote(str(submission_err))})",
            "submit_rc=$?", "set -e",
            f"printf 'job_ref=%s\\nsubmit_rc=%s\\n' \"$job_ref\" \"$submit_rc\" > {shlex.quote(str(submission))}",
            "job_id=${job_ref%%;*}",
            "case \"$job_id\" in ''|*[!0-9]*) exit 65 ;; esac",
            f"sacct -j \"$job_id\" --format={','.join(SLURM_FIELDS)} -P > {shlex.quote(str(sacct))}",
            "test \"$submit_rc\" -eq 0", "",
        ])
    write_new(plan, ("\n".join(lines) + "\n").encode())
    print(f"wrote submission plan without executing it: {plan}")
    return 0


def completed_case_collection_record(
        root: Path, campaign: Mapping[str, Any], record: Mapping[str, Any],
        ledger_by_observation: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    case_id = str(record["case_id"])
    case = case_by_id(campaign, case_id)
    evidence = root / "evidence" / case_id
    paths = {
        "preflight_manifest": Path(str(record["preflight_manifest"])),
        "submit_record": root / "records" / case_id / "submit-record.txt",
        "stdout": evidence / "stdout.txt", "stderr": evidence / "stderr.txt",
        "exit_record": evidence / "exit-status.txt",
        "slurm_accounting": root / "records" / case_id / "slurm-accounting.txt",
    }
    for label, path in paths.items():
        regular_path(path, allowed_roots=(root,),
                     label=f"{case_id} result artifact {label}")
    job_id, _ = parse_submit_record(
        paths["submit_record"], allowed_root=root / "records" / case_id,
        expected_rc=0)
    parse_slurm_accounting(
        paths["slurm_accounting"], allowed_root=root / "records" / case_id,
        job_id=job_id, job_name=f"sci-map-001-{case_id}",
        partition=campaign["fixed_execution"]["partition"],
        cpus=int(case["cpus"]), state="COMPLETED", exit_code="0:0")
    try:
        exit_status = int(paths["exit_record"].read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise CampaignError(f"{case_id} exit record is invalid") from exc
    if exit_status != 0:
        fail(f"{case_id} did not have the repaired expected-success outcome: {exit_status}")
    wrapper_stdout = sorted(evidence.glob("slurm-wrapper-*.out"))
    wrapper_stderr = sorted(evidence.glob("slurm-wrapper-*.err"))
    expected_wrapper_stdout = evidence / f"slurm-wrapper-{job_id}.out"
    expected_wrapper_stderr = evidence / f"slurm-wrapper-{job_id}.err"
    if wrapper_stdout != [expected_wrapper_stdout] \
            or wrapper_stderr != [expected_wrapper_stderr]:
        fail(f"{case_id} requires exactly one outer Slurm stdout/stderr pair")
    started = require_utc_timestamp(
        evidence / "started-at-utc.txt", allowed_root=evidence,
        label=f"{case_id} execution-start timestamp")
    completed = require_utc_timestamp(
        evidence / "completed-at-utc.txt", allowed_root=evidence,
        label=f"{case_id} execution-complete timestamp")
    if started.read_text(encoding="utf-8").strip() > \
            completed.read_text(encoding="utf-8").strip():
        fail(f"{case_id} execution timestamps are reversed")
    pre_run = regular_path(evidence / "pre-run-sha256.txt",
                           allowed_roots=(evidence,),
                           label=f"{case_id} pre-run integrity")
    post_run = regular_path(evidence / "post-run-sha256.txt",
                            allowed_roots=(evidence,),
                            label=f"{case_id} post-run integrity")
    if pre_run.read_bytes() != post_run.read_bytes():
        fail(f"{case_id} executable/script authorities changed during execution")
    integrity_binding = regular_path(
        evidence / "integrity-manifest.sha256", allowed_roots=(evidence,),
        label=f"{case_id} integrity-manifest binding")
    reconstruction_binding = regular_path(
        evidence / "reconstruction-manifest.sha256", allowed_roots=(evidence,),
        label=f"{case_id} reconstruction-manifest binding")
    verify_sha_manifest(integrity_binding, allowed_roots=(root,))
    verify_sha_manifest(reconstruction_binding, allowed_roots=(root,))
    log_paths = [
        paths["stdout"], paths["stderr"], paths["submit_record"],
        root / "records" / case_id / "submit-stderr.txt",
        paths["slurm_accounting"], wrapper_stdout[0], wrapper_stderr[0],
        started, completed, pre_run, post_run, integrity_binding,
        reconstruction_binding, evidence / "hostname.txt",
        evidence / "runtime-environment.txt", evidence / "affinity.txt",
    ]
    for path in log_paths:
        regular_path(path, allowed_roots=(root,), label=f"{case_id} complete log")
    if len(set(path.resolve() for path in log_paths)) != len(log_paths):
        fail(f"{case_id} complete log set contains duplicate paths")
    merged_config = regular_path(
        root / "records" / case_id / "merged.yaml", allowed_roots=(root,),
        label=f"{case_id} pre-run merged config")
    raw_input_manifest = regular_path(
        root / "raw-input-manifests" / f"{case_id}.json",
        allowed_roots=(root / "raw-input-manifests",),
        label=f"{case_id} raw-input manifest")
    raw_record = read_json(raw_input_manifest)
    if not isinstance(raw_record, Mapping) \
            or raw_record.get("candidate_sha") != EXPECTED_CANDIDATE:
        fail(f"{case_id} raw-input manifest identity differs")
    raw_manifest_digest = sha256(raw_input_manifest)
    return {
        "case_id": case_id, "reduction_root": str(record["case_dir"]),
        "exit_status_path": str(paths["exit_record"].resolve()),
        "merged_config": str(merged_config),
        "raw_input_manifest": str(raw_input_manifest),
        "raw_input_manifest_sha256": raw_manifest_digest,
        "logs": [str(path.resolve()) for path in log_paths],
        "sample_ledgers": ledger_by_observation,
        **{label: str(path.resolve()) for label, path in paths.items()},
    }


def command_build_result_collection(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    require_build(root)
    identity(values, campaign, "prepared")
    records = load_cases_state(root)
    regular_path(root / "plans" / "submit-seven-cases.sh",
                 allowed_roots=(root / "plans",), label="seven-case submission plan")
    validate_reconstruction_authority(root, campaign, values, create=False)
    ledgers: dict[str, str] = {}
    ledger_by_observation: dict[str, dict[str, str]] = {}
    for obsnum in (152389, 152390, 152392):
        mode = "point" if obsnum == 152389 else "science"
        for array in campaign["fixed_execution"]["arrays"]:
            key = f"{mode}-{obsnum}-{array}"
            path = root / "sample-ledgers" / f"{key}.npz"
            if not path.is_file():
                fail(f"required independent reconstruction ledger is absent: {path}")
            ledgers[key] = str(path.resolve())
            ledger_by_observation.setdefault(str(obsnum), {})[array] = str(path.resolve())
    cases = []
    raw_manifest_digests: dict[str, str] = {}
    for record in records:
        case_id = str(record["case_id"])
        case_record = completed_case_collection_record(
            root, campaign, record, ledger_by_observation)
        raw_manifest_digest = str(case_record["raw_input_manifest_sha256"])
        raw_manifest_digests[case_id] = raw_manifest_digest
        cases.append(case_record)
    if len({raw_manifest_digests[case_id] for case_id in ("P-SEQ", "P-OMP")}) != 1:
        fail("Point seq/OMP raw-input manifests are not byte-identical")
    science_ids = ("S-C-SEQ", "S-C-OMP", "S-E-SEQ", "S-E-OMP", "S-X-SEQ")
    if len({raw_manifest_digests[case_id] for case_id in science_ids}) != 1:
        fail("Science cases do not share one byte-identical raw-input manifest")
    point_digest = raw_manifest_digests["P-SEQ"]
    science_digest = raw_manifest_digests["S-C-SEQ"]
    try:
        import numpy as np
    except Exception as exc:
        raise CampaignError(f"cannot inspect reconstruction-ledger bindings: {exc}") from exc
    for key, path_text in ledgers.items():
        with np.load(path_text, allow_pickle=False) as archive:
            if "raw_input_manifest_sha256" not in archive:
                fail(f"independent ledger lacks raw-input-manifest binding: {path_text}")
            value = np.asarray(archive["raw_input_manifest_sha256"])
            if value.size != 1:
                fail(f"independent ledger has non-scalar raw-input binding: {path_text}")
            actual_digest = str(value.reshape(-1)[0].item())
        expected_digest = point_digest if key.startswith("point-") else science_digest
        if actual_digest != expected_digest:
            fail(f"independent ledger raw-input binding differs: {key}")
    collection = {
        "schema_version": "sci-map-001-result-collection-v1",
        "request_id": REQUEST_ID, "revision": campaign["revision"],
        "candidate_sha": EXPECTED_CANDIDATE, "request_root": str(root),
        "independent_reconstruction_inputs": ledgers, "cases": cases,
    }
    canonical_output = root / "analysis" / "result-collection.json"
    output = args.output.resolve() if args.output else canonical_output
    if output != canonical_output.resolve(strict=False):
        fail(f"result collection output must remain at {canonical_output}")
    write_new(output, json_bytes(collection))
    print(f"wrote fail-closed result collection: {output}")
    return 0


def validate_collection_identity(path: Path, root: Path,
                                 campaign: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical = (root / "analysis" / "result-collection.json").resolve(strict=False)
    if path.resolve(strict=False) != canonical:
        fail(f"result collection must be the canonical request artifact: {canonical}")
    collection_path = regular_path(path, allowed_roots=(root / "analysis",),
                                   label="result collection")
    value = read_json(collection_path)
    if not isinstance(value, Mapping) \
            or set(value) != {
                "schema_version", "request_id", "revision", "candidate_sha",
                "request_root", "independent_reconstruction_inputs", "cases"} \
            or value.get("schema_version") != "sci-map-001-result-collection-v1" \
            or value.get("request_id") != REQUEST_ID \
            or value.get("revision") != campaign["revision"] \
            or value.get("candidate_sha") != EXPECTED_CANDIDATE \
            or value.get("request_root") != str(root):
        fail("result collection identity differs")
    cases = value.get("cases")
    if not isinstance(cases, list) or [
            row.get("case_id") for row in cases if isinstance(row, Mapping)
    ] != list(CASE_IDS):
        fail("result collection case order/identity differs")
    expected_ledgers = {
        f"{'point' if obsnum == 152389 else 'science'}-{obsnum}-{array}"
        for obsnum in (152389, 152390, 152392) for array in ARRAYS
    }
    ledgers = value.get("independent_reconstruction_inputs")
    if not isinstance(ledgers, Mapping) or set(ledgers) != expected_ledgers:
        fail("result collection reconstruction-ledger identity differs")
    expected_ledger_paths: dict[str, str] = {}
    ledger_by_observation: dict[str, dict[str, str]] = {}
    for obsnum in (152389, 152390, 152392):
        mode = "point" if obsnum == 152389 else "science"
        for array in ARRAYS:
            key = f"{mode}-{obsnum}-{array}"
            ledger_path = regular_path(
                root / "sample-ledgers" / f"{key}.npz",
                allowed_roots=(root / "sample-ledgers",),
                label=f"result collection ledger {key}")
            expected_ledger_paths[key] = str(ledger_path)
            ledger_by_observation.setdefault(str(obsnum), {})[array] = str(ledger_path)
    if dict(ledgers) != expected_ledger_paths:
        fail("result collection reconstruction-ledger paths differ")
    expected_cases = [
        completed_case_collection_record(root, campaign, record,
                                         ledger_by_observation)
        for record in load_cases_state(root)
    ]
    if cases != expected_cases:
        fail("result collection case evidence differs from current immutable artifacts")
    return value


def command_validate_analysis_accounting(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    require_build(root)
    identity(values, campaign, "prepared")
    load_cases_state(root)
    validate_reconstruction_authority(root, campaign, values, create=False)
    validate_collection_identity(root / "analysis/result-collection.json", root, campaign)
    regular_path(root / "analysis/analysis-inputs.json",
                 allowed_roots=(root / "analysis",), label="analysis inputs")
    evidence = root / "evidence" / "ANALYSIS"
    exit_path = regular_path(evidence / "exit-status.txt", allowed_roots=(evidence,),
                             label="analysis exit record")
    try:
        analysis_rc = int(exit_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise CampaignError("analysis exit record is invalid") from exc
    if analysis_rc not in (0, 2):
        fail(f"analysis exit is neither complete pass nor complete nonconformance: {analysis_rc}")
    job_id, _ = parse_submit_record(
        root / "analysis/analysis-submit.txt", allowed_root=root / "analysis",
        expected_rc=analysis_rc)
    parse_slurm_accounting(
        root / "analysis/analysis-sacct.txt", allowed_root=root / "analysis",
        job_id=job_id, job_name="sci-map-001-analysis",
        partition=campaign["fixed_execution"]["partition"],
        cpus=int(campaign["fixed_execution"]["analysis_cpus"]),
        state="COMPLETED" if analysis_rc == 0 else "FAILED",
        exit_code=f"{analysis_rc}:0")
    exact_outer = {
        root / f"analysis/analysis-slurm-{job_id}.out",
        root / f"analysis/analysis-slurm-{job_id}.err",
    }
    actual_outer = set((root / "analysis").glob("analysis-slurm-*.out")) \
        | set((root / "analysis").glob("analysis-slurm-*.err"))
    if actual_outer != exact_outer:
        fail("analysis requires exactly one job-ID-bound outer stdout/stderr pair")
    for path in exact_outer:
        regular_path(path, allowed_roots=(root,), label="analysis outer Slurm log")
    for path, label in (
            (root / "analysis/analysis-submit.err", "analysis submit stderr"),
            (evidence / "stdout.txt", "analysis stdout"),
            (evidence / "stderr.txt", "analysis stderr"),
            (evidence / "hostname.txt", "analysis hostname"),
            (evidence / "runtime-environment.txt", "analysis runtime environment"),
            (evidence / "affinity.txt", "analysis CPU affinity"),
            (evidence / "python-version.txt", "analysis Python version"),
            (evidence / "module-versions.txt", "analysis module versions"),
            (evidence / "pre-run-sha256.txt", "analysis pre-run integrity"),
            (evidence / "post-run-sha256.txt", "analysis post-run integrity")):
        regular_path(path, allowed_roots=(root,), label=label)
    if (evidence / "pre-run-sha256.txt").read_bytes() != \
            (evidence / "post-run-sha256.txt").read_bytes():
        fail("analysis executable/input authorities changed during execution")
    verify_sha_manifest(evidence / "pre-run-sha256.txt", allowed_roots=(root,))
    verify_sha_manifest(evidence / "post-run-sha256.txt", allowed_roots=(root,))
    started = require_utc_timestamp(
        evidence / "started-at-utc.txt", allowed_root=evidence,
        label="analysis execution-start timestamp")
    completed = require_utc_timestamp(
        evidence / "completed-at-utc.txt", allowed_root=evidence,
        label="analysis execution-complete timestamp")
    if started.read_text(encoding="utf-8").strip() > \
            completed.read_text(encoding="utf-8").strip():
        fail("analysis execution timestamps are reversed")
    results = root / "analysis" / "results"
    if results.is_symlink() or not results.is_dir():
        fail("analysis result directory is absent or a symlink")
    required_results = {
        "request-union.json", "wcs.json", "pixel-identities.json",
        "blank-summary.json", "false-z-regions.json", "edge-tables.csv",
        "coadd-recombination.json", "log-completion.json", "verification.json",
        "inventory.json", "baseline-tool-runs.json", "residual-manifest.json",
        "VERDICT.txt", "SHA256SUMS",
    }
    missing_results = sorted(
        name for name in required_results if not (results / name).is_file())
    if missing_results:
        fail(f"analysis result set lacks mandatory outputs: {missing_results}")
    verify_relative_sha_inventory(results / "SHA256SUMS", root=results)
    residual_manifest = read_json(results / "residual-manifest.json")
    if not isinstance(residual_manifest, Mapping) or set(residual_manifest) != {
            "schema_version", "request_id", "candidate_sha", "records"} \
            or residual_manifest.get("schema_version") != \
            "sci-map-001-lossless-residual-manifest-v1" \
            or residual_manifest.get("request_id") != REQUEST_ID \
            or residual_manifest.get("candidate_sha") != EXPECTED_CANDIDATE:
        fail("lossless residual manifest identity or schema differs")
    residual_records = residual_manifest.get("records")
    if not isinstance(residual_records, list) or not residual_records:
        fail("lossless residual manifest has no records")
    residual_paths: set[Path] = set()
    residual_check_ids: set[str] = set()
    try:
        import numpy as np
    except Exception as exc:
        raise CampaignError(f"cannot inspect lossless residual archives: {exc}") from exc
    for index, record in enumerate(residual_records):
        if not isinstance(record, Mapping) or set(record) != {
                "check_id", "path", "sha256", "size_bytes", "array_keys"}:
            fail(f"lossless residual record {index} has an invalid shape")
        check_id = record.get("check_id")
        path_value = record.get("path")
        array_keys = record.get("array_keys")
        if not isinstance(check_id, str) or not check_id.endswith(
                ".lossless_residuals") or check_id in residual_check_ids:
            fail(f"lossless residual record {index} has an invalid/repeated check ID")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute() \
                or any(token in path_value for token in ("\n", "\r", "\\")):
            fail(f"lossless residual record {index} has an unsafe path")
        raw_path = Path(path_value)
        path = regular_path(raw_path, allowed_roots=(results,),
                            label=f"lossless residual record {index}")
        if str(path) != path_value or path.suffix != ".npz" or path in residual_paths:
            fail(f"lossless residual record {index} has a noncanonical/repeated path")
        if record.get("sha256") != sha256(path) or record.get(
                "size_bytes") != path.stat().st_size:
            fail(f"lossless residual record {index} differs from its artifact")
        if not isinstance(array_keys, list) or not array_keys \
                or any(not isinstance(key, str) or not key for key in array_keys) \
                or len(array_keys) != len(set(array_keys)):
            fail(f"lossless residual record {index} has invalid array keys")
        try:
            with np.load(path, allow_pickle=False) as archive:
                archive_keys = list(archive.files)
        except Exception as exc:
            raise CampaignError(
                f"cannot inspect lossless residual record {index}: {exc}") from exc
        if len(archive_keys) != len(set(archive_keys)) or \
                sorted(array_keys) != sorted(archive_keys):
            fail(f"lossless residual record {index} array keys differ from its archive")
        residual_paths.add(path)
        residual_check_ids.add(check_id)
    actual_residual_paths = {
        path.resolve(strict=True) for path in results.rglob("*-residuals.npz")
        if path.is_file() and not path.is_symlink()
    }
    if residual_paths != actual_residual_paths:
        fail("lossless residual manifest does not exhaust the NPZ result artifacts")
    verification = read_json(results / "verification.json")
    expected_result = "local_analysis_pass" if analysis_rc == 0 \
        else "local_analysis_failed"
    finding_state = verification.get("finding_state") \
        if isinstance(verification, Mapping) else None
    if not isinstance(verification, Mapping) \
            or verification.get("request_id") != REQUEST_ID \
            or verification.get("candidate_sha") != EXPECTED_CANDIDATE \
            or verification.get("result") != expected_result \
            or verification.get("conformance_claim") != \
            "none_independent_reaudit_required" \
            or verification.get("external_evidence_claim") != \
            "returned_files_unreviewed_until_owner_retrieval_and_independent_audit" \
            or verification.get("dependency_nonclosure") != [
                "SCI-ALIGN-001", "SCI-CAL-001", "SCI-AST-001",
                "SCI-PTC-001", "SCI-VAL-001"] \
            or not isinstance(finding_state, Mapping) \
            or finding_state.get("F009") != "addressed_pending_reaudit" \
            or finding_state.get("F010") != "addressed_pending_reaudit" \
            or finding_state.get("F012") != \
            "outstanding_until_human_run_bundle_is_independently_audited" \
            or finding_state.get("F013") != "conditioned_on_named_upstream_audits":
        fail("analysis verification identity/status boundaries differ")
    print(json.dumps({"result": "pass", "analysis_exit_status": analysis_rc,
                      "slurm_job_id": job_id}, sort_keys=True))
    return 0


def inventory_entry(path: Path, root: Path) -> dict[str, Any]:
    info = path.lstat()
    common = {
        "path": path.relative_to(root).as_posix(), "mtime_ns": info.st_mtime_ns,
        "mode": oct(stat.S_IMODE(info.st_mode)),
    }
    if path.is_symlink():
        fail(f"evidence tree contains a forbidden symlink: {path}")
    if path.is_file():
        return {**common, "type": "file", "size": info.st_size, "sha256": sha256(path)}
    if path.is_dir():
        return {**common, "type": "directory"}
    fail(f"unsupported filesystem entry in evidence: {path}")


def command_hash_evidence(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    canonical = root / "manifests" / (
        "final-inventory.json" if args.final else "pre-analysis-inventory.json")
    output = args.output.resolve(strict=False) if args.output else canonical
    if output != canonical.resolve(strict=False):
        fail(f"evidence inventory output must remain at {canonical}")
    digest_output = output.with_suffix(output.suffix + ".sha256")
    if os.path.lexists(output) or os.path.lexists(digest_output):
        fail(f"refusing to overwrite existing evidence inventory: {output}")
    entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    records = [inventory_entry(path, root) for path in entries]
    payload = {
        "schema_version": "sci-map-unity-evidence-inventory-v1",
        "request_id": REQUEST_ID, "candidate_sha": EXPECTED_CANDIDATE,
        "scope": "final" if args.final else "pre-analysis", "root": str(root),
        "records": records,
    }
    write_new(output, json_bytes(payload))
    write_new(digest_output, f"{sha256(output)}  {output.name}\n".encode())
    print(f"wrote {payload['scope']} evidence inventory: {output}")
    return 0


def command_emit_final_plan(args: argparse.Namespace) -> int:
    campaign, _ = load_campaign(args.campaign)
    values = validate_owner(args.owner_values, campaign)
    root = require_prepared(values)
    require_build(root)
    identity(values, campaign, "prepared")
    load_cases_state(root)
    reconstruction_authority = validate_reconstruction_authority(
        root, campaign, values, create=False)
    regular_path(root / "plans" / "submit-seven-cases.sh",
                 allowed_roots=(root / "plans",), label="seven-case submission plan")
    package = frozen_package(root)
    canonical_collection = root / "analysis/result-collection.json"
    collection = args.collection.resolve(strict=False) if args.collection \
        else canonical_collection
    validate_collection_identity(collection, root, campaign)
    collection_digest = sha256(collection)
    analysis = package / "SCI-MAP-001-analysis.py"
    python = Path(values["unity_python"]).resolve(strict=True)
    source = Path(values["unity_source_checkout"]).resolve(strict=True)
    inputs = root / "analysis" / "analysis-inputs.json"
    results = root / "analysis" / "results"
    job_stdout = root / "analysis" / "analysis-slurm-%j.out"
    job_stderr = root / "analysis" / "analysis-slurm-%j.err"
    analysis_wrapper = package / "scripts" / "analysis-job-wrapper.sh"
    analysis_evidence = root / "evidence" / "ANALYSIS"
    governing_validation = root / "governing" / "validation"
    analysis_command = shlex.join((
        "bash", str(analysis_wrapper), str(python), str(analysis), sha256(analysis),
        str(package / "campaign.json"), str(inputs), str(results),
        str(analysis_evidence), str(source),
        str(governing_validation / "product_contracts.json"),
        str(governing_validation / "validation_profiles.json"),
        str(governing_validation / "accepted_runs.json"),
        campaign["authority"]["point_product_contract_id"],
        campaign["authority"]["science_product_contract_id"],
    ))
    options = slurm_options(
        values, campaign, cpus=campaign["fixed_execution"]["analysis_cpus"],
        memory=campaign["fixed_execution"]["analysis_memory"],
        time=campaign["fixed_execution"]["analysis_time"],
        job_name="sci-map-001-analysis", chdir=package,
        stdout=job_stdout, stderr=job_stderr,
    )
    options.extend(("--wrap", analysis_command))
    build_inputs = shlex.join((
        str(python), str(analysis), "build-analysis-inputs",
        "--campaign", str(package / "campaign.json"),
        "--request-root", str(root), "--collection", str(collection),
        "--product-contracts", str(root / "governing/validation/product_contracts.json"),
        "--output", str(inputs),
    ))
    driver = package / "scripts" / "unity-campaign.py"
    owner_path = root / "owner-values.json"
    plan = root / "plans" / "analyze-freeze-and-retrieve.sh"
    pre_analysis_identity = root / "analysis" / "pre-analysis-identity.json"
    inputs_frozen_at = root / "analysis" / "analysis-inputs-frozen-at-utc.txt"
    final_frozen_at = root / "records" / "final-freeze-at-utc.txt"
    manifest = root / "SCI-MAP-001-UNITY-001-MANIFEST.sha256"
    bundle = root.parent / "SCI-MAP-001-UNITY-001.tar.gz"
    bundle_digest = Path(str(bundle) + ".sha256")
    manifest_tmp = root.parent / ".SCI-MAP-001-UNITY-001-MANIFEST.sha256.tmp"
    tar_tmp = root.parent / ".SCI-MAP-001-UNITY-001.tar.tmp"
    gzip_tmp = root.parent / ".SCI-MAP-001-UNITY-001.tar.gz.tmp"
    should_be_absent = (
        inputs, results, analysis_evidence,
        root / "analysis/analysis-submit.err", root / "analysis/analysis-submit.txt",
        root / "analysis/analysis-sacct.txt", pre_analysis_identity,
        inputs_frozen_at, final_frozen_at,
        root / "manifests/pre-analysis-inventory.json",
        root / "manifests/pre-analysis-inventory.json.sha256",
        root / "manifests/final-inventory.json",
        root / "manifests/final-inventory.json.sha256",
        manifest, bundle, bundle_digest, manifest_tmp, tar_tmp, gzip_tmp,
    )
    existing = [str(path) for path in should_be_absent if os.path.lexists(path)]
    if existing:
        fail(f"analysis/freeze state already exists; stop for owner inspection: {existing}")
    identity_command = shlex.join((
        str(python), str(driver), "--campaign", str(package / "campaign.json"),
        "identity", "--owner-values", str(owner_path),
        "--expect-request-root", "prepared",
    ))
    pre_hash_command = shlex.join((
        str(python), str(driver), "--campaign", str(package / "campaign.json"),
        "hash-evidence", "--owner-values", str(owner_path),
    ))
    analysis_accounting_command = shlex.join((
        str(python), str(driver), "--campaign", str(package / "campaign.json"),
        "validate-analysis-accounting", "--owner-values", str(owner_path),
    ))
    final_hash_command = shlex.join((
        str(python), str(driver), "--campaign", str(package / "campaign.json"),
        "hash-evidence", "--owner-values", str(owner_path), "--final",
    ))
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail", "",
        "# HUMAN ACTION: metadata freeze, bounded Slurm analysis, and final inventory.",
        f"test \"$(sha256sum {shlex.quote(str(collection))} | awk '{{print $1}}')\" = {collection_digest}",
        f"sha256sum -c {shlex.quote(str(reconstruction_authority))}",
        f"test ! -e {shlex.quote(str(pre_analysis_identity))}",
        f"test ! -L {shlex.quote(str(pre_analysis_identity))}",
        f"{identity_command} > {shlex.quote(str(pre_analysis_identity))}",
        build_inputs,
        f"test ! -e {shlex.quote(str(inputs_frozen_at))}",
        f"test ! -L {shlex.quote(str(inputs_frozen_at))}",
        f"date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(str(inputs_frozen_at))}",
        pre_hash_command,
        f"test ! -e {shlex.quote(str(results))}",
        f"test ! -L {shlex.quote(str(results))}",
        f"test ! -e {shlex.quote(str(analysis_evidence))}",
        f"test ! -L {shlex.quote(str(analysis_evidence))}",
        f"test ! -e {shlex.quote(str(root / 'analysis/analysis-submit.err'))}",
        f"test ! -L {shlex.quote(str(root / 'analysis/analysis-submit.err'))}",
        f"test ! -e {shlex.quote(str(root / 'analysis/analysis-submit.txt'))}",
        f"test ! -L {shlex.quote(str(root / 'analysis/analysis-submit.txt'))}",
        f"test ! -e {shlex.quote(str(root / 'analysis/analysis-sacct.txt'))}",
        f"test ! -L {shlex.quote(str(root / 'analysis/analysis-sacct.txt'))}",
        "set +e",
        f"analysis_job_ref=$({shlex.join(options)} 2> {shlex.quote(str(root / 'analysis/analysis-submit.err'))})",
        "analysis_submit_rc=$?", "set -e",
        f"printf 'job_ref=%s\\nsubmit_rc=%s\\n' \"$analysis_job_ref\" \"$analysis_submit_rc\" > {shlex.quote(str(root / 'analysis/analysis-submit.txt'))}",
        "analysis_job_id=${analysis_job_ref%%;*}",
        "case \"$analysis_job_id\" in ''|*[!0-9]*) exit 65 ;; esac",
        f"sacct -j \"$analysis_job_id\" --format={','.join(SLURM_FIELDS)} -P > {shlex.quote(str(root / 'analysis/analysis-sacct.txt'))}",
        "# Analysis exit 0 means checks passed; exit 2 means complete nonconformance.",
        "# Any status remains evidence for independent review, never a finding closure.",
        f"analysis_rc=$(cat {shlex.quote(str(analysis_evidence / 'exit-status.txt'))})",
        "case \"$analysis_rc\" in 0|2) : ;; *) exit \"$analysis_rc\" ;; esac",
        analysis_accounting_command,
        f"test \"$(git -C {shlex.quote(str(source))} rev-parse HEAD)\" = {EXPECTED_CANDIDATE}",
        f"test -z \"$(git -C {shlex.quote(str(source))} status --porcelain=v1 --untracked-files=all)\"",
        f"test ! -e {shlex.quote(str(final_frozen_at))}",
        f"test ! -L {shlex.quote(str(final_frozen_at))}",
        f"date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(str(final_frozen_at))}",
        final_hash_command,
        f"test ! -e {shlex.quote(str(manifest))}",
        f"test ! -L {shlex.quote(str(manifest))}",
        f"test ! -e {shlex.quote(str(manifest_tmp))}",
        f"test ! -L {shlex.quote(str(manifest_tmp))}",
        f"(cd {shlex.quote(str(root))} && find . -type f ! -path './SCI-MAP-001-UNITY-001-MANIFEST.sha256' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > {shlex.quote(str(manifest_tmp))}",
        f"mv {shlex.quote(str(manifest_tmp))} {shlex.quote(str(manifest))}",
        f"(cd {shlex.quote(str(root))} && sha256sum -c ./SCI-MAP-001-UNITY-001-MANIFEST.sha256)",
        f"test ! -e {shlex.quote(str(bundle))}",
        f"test ! -L {shlex.quote(str(bundle))}",
        f"test ! -e {shlex.quote(str(bundle_digest))}",
        f"test ! -L {shlex.quote(str(bundle_digest))}",
        f"test ! -e {shlex.quote(str(tar_tmp))}",
        f"test ! -L {shlex.quote(str(tar_tmp))}",
        f"test ! -e {shlex.quote(str(gzip_tmp))}",
        f"test ! -L {shlex.quote(str(gzip_tmp))}",
        "tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=posix "
        "--pax-option=delete=atime,delete=ctime -cf " + shlex.quote(str(tar_tmp))
        + " -C " + shlex.quote(str(root.parent)) + " -- " + shlex.quote("./" + root.name),
        f"gzip -n -c {shlex.quote(str(tar_tmp))} > {shlex.quote(str(gzip_tmp))}",
        f"mv {shlex.quote(str(gzip_tmp))} {shlex.quote(str(bundle))}",
        f"rm -f {shlex.quote(str(tar_tmp))}",
        f"(cd {shlex.quote(str(root.parent))} && sha256sum {shlex.quote(bundle.name)} > {shlex.quote(bundle_digest.name)})",
        "", "# Run the following retrieval command on the owner's local machine, not Unity:",
        "# " + shlex.join(("rsync", "-a", "--checksum", "--protect-args",
                            f"unity_toltec:{bundle}",
                            f"unity_toltec:{bundle_digest}",
                            str(Path(values['local_retrieval_destination'])) + "/")),
    ]
    write_new(plan, ("\n".join(lines) + "\n").encode())
    print(f"wrote analysis/final-bundle plan without submitting: {plan}")
    return 0


def parser() -> argparse.ArgumentParser:
    default_campaign = Path(__file__).resolve().parent.parent / "campaign.json"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign", type=Path, default=default_campaign)
    commands = result.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("self-check", help="check immutable driver/campaign invariants")
    sub.add_argument("--require-inventory", action="store_true")
    sub.set_defaults(handler=command_self_check)

    sub = commands.add_parser("validate", help="validate explicit owner values without mutation")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.add_argument("--require-existing", action="store_true")
    sub.add_argument("--expect-request-root", choices=("absent", "prepared", "either"),
                     default="either")
    sub.set_defaults(handler=command_validate)

    sub = commands.add_parser("identity", help="prove candidate, package, TolProj, and TolTECA identity")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.add_argument("--expect-request-root", choices=("absent", "prepared"), default="absent")
    sub.set_defaults(handler=command_identity)

    sub = commands.add_parser("prepare", help="explicitly initialize an absent request root")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.set_defaults(handler=command_prepare)

    sub = commands.add_parser("build", help="perform the pinned disconnected candidate build")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.set_defaults(handler=command_build)

    sub = commands.add_parser("prepare-cases", help="create and preflight all seven native cases")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.set_defaults(handler=command_prepare_cases)

    sub = commands.add_parser("emit-submit-plan", help="emit, never execute, seven sbatch --wait commands")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.set_defaults(handler=command_emit_submit_plan)

    sub = commands.add_parser("build-result-collection",
                              help="freeze successful returned paths and independent ledgers")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.add_argument("--output", type=Path)
    sub.set_defaults(handler=command_build_result_collection)

    sub = commands.add_parser(
        "validate-analysis-accounting",
        help="reconcile the bounded analysis outcome with Slurm and frozen authorities")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.set_defaults(handler=command_validate_analysis_accounting)

    sub = commands.add_parser("hash-evidence", help="write a non-self-referential evidence inventory")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.add_argument("--output", type=Path)
    sub.add_argument("--final", action="store_true")
    sub.set_defaults(handler=command_hash_evidence)

    sub = commands.add_parser("emit-final-plan",
                              help="emit bounded analysis and unity_toltec retrieval commands")
    sub.add_argument("--owner-values", type=Path, required=True)
    sub.add_argument("--collection", type=Path)
    sub.set_defaults(handler=command_emit_final_plan)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"SCI-MAP-001 campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
