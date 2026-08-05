#!/usr/bin/env python3
"""Prepare the owner-authorized 16-map SCI-ALIGN-001 diagnostic campaign.

The campaign is a controlled new reduction, not an attempt to invent missing
per-observation Citlali YAML files.  It derives each direct Citlali request from
the historical numbered-config contract: the shared Beammap low-level policy in
``70_reduce.yaml`` and the selected observation's calibration in
``72_reduce.yaml``.  It writes only to a new owner-specified campaign root and
does not invoke Citlali, Slurm, or Unity services.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "sci-align-001-3c273-numbered-config-replay-campaign-v1"
RAW_RE = re.compile(r"^toltec(?P<network>\d+)_(?P<obsnum>\d+)_000_0002_.+\.nc$")
PHOTOMETRY_SELECT_RE = re.compile(r"^\s*obsnum\s*==\s*(?P<obsnum>\d+)\s*$")
PRIOR_RELATIVE = Path("data/beammap_priors/beammap_slot_priors_soft_v1.ecsv")
BASE_70_SHA256 = "71273b3199bc762e406f51d34f9e31c8cf51d42ac0d30889d2bf4cace1ebdaff"
CALIBRATION_72_SHA256 = "7fec4b95842ecdaa0414592dac116120ee88fae52ff0e881368072876fe6ddfa"

# 148670 is already checksum-verified and is the sixteenth independent map.
# The fifteen new maps are stratified over March 2024--February 2026.  Each
# batch is intentionally capped at four owner-submitted jobs.
CAMPAIGN = (
    (113862, 1), (131925, 1), (136279, 1), (152882, 1),
    (128588, 2), (133543, 2), (150819, 2), (152451, 2),
    (129687, 3), (134643, 3), (151126, 3), (151950, 3),
    (130922, 4), (135397, 4), (151600, 4),
)
EXISTING_REPLAY_OBSNUM = 148670


class CampaignError(RuntimeError):
    """Raised when the campaign cannot preserve its numbered-config contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CampaignError(f"{label} is missing or is not a regular file: {resolved}")
    return resolved


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise CampaignError(f"{label} is missing or is not a directory: {resolved}")
    return resolved


def require_executable(path: Path) -> Path:
    executable = require_file(path, "Citlali executable")
    if not os.access(executable, os.X_OK):
        raise CampaignError(f"Citlali executable is not executable: {executable}")
    return executable


def checksum_lines(directory: Path) -> str:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "PREPARATION_SHA256SUMS"}:
            rows.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    return "\n".join(rows) + "\n"


def _yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(require_file(path, label).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CampaignError(f"cannot parse {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} is not a mapping: {path}")
    return value


def _verified_authority(path: Path, label: str, expected_sha256: str) -> Path:
    resolved = require_file(path, label)
    measured = sha256_file(resolved)
    if measured != expected_sha256:
        raise CampaignError(
            f"{label} digest mismatch: expected {expected_sha256}, measured {measured}"
        )
    return resolved


def _step(document: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        reduce = document["reduce"]
        steps = reduce["steps"]
        step = steps[0] if isinstance(steps, list) else steps[0]
    except (KeyError, IndexError, TypeError) as error:
        raise CampaignError(f"{label} lacks reduce.steps[0]") from error
    if not isinstance(step, dict):
        raise CampaignError(f"{label} reduce.steps[0] is not a mapping")
    return step


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} is not a mapping")
    return value


def _photometry_by_obsnum(calibration: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    config = _mapping(_step(_yaml(calibration, "72 calibration overlay"), "72 calibration overlay").get("config"), "72 config")
    cal_items = config.get("cal_items")
    if not isinstance(cal_items, list):
        raise CampaignError("72 calibration overlay lacks config.cal_items")
    astrometry = [item for item in cal_items if isinstance(item, dict) and item.get("type") == "astrometry"]
    if len(astrometry) != 1:
        raise CampaignError("72 calibration overlay must contain exactly one astrometry item")
    by_obsnum: dict[int, dict[str, Any]] = {}
    for item in cal_items:
        if not isinstance(item, dict) or item.get("type") != "photometry":
            continue
        selector = item.get("select")
        if not isinstance(selector, str) or not (match := PHOTOMETRY_SELECT_RE.fullmatch(selector)):
            raise CampaignError(f"unsupported 72 photometry selector: {selector!r}")
        obsnum = int(match.group("obsnum"))
        if obsnum in by_obsnum:
            raise CampaignError(f"72 calibration overlay duplicates photometry for ObsNum {obsnum}")
        value = copy.deepcopy(item)
        value.pop("select", None)
        by_obsnum[obsnum] = value
    return copy.deepcopy(astrometry[0]), by_obsnum


def authority(base: Path, calibration: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    base = _verified_authority(base, "70 policy", BASE_70_SHA256)
    calibration = _verified_authority(calibration, "72 calibration overlay", CALIBRATION_72_SHA256)
    base_config = _mapping(_step(_yaml(base, "70 policy"), "70 policy").get("config"), "70 config")
    low_level = _mapping(base_config.get("low_level"), "70 config.low_level")
    astrometry, photometry = _photometry_by_obsnum(calibration)
    missing = [obsnum for obsnum, _batch in CAMPAIGN if obsnum not in photometry]
    if missing:
        raise CampaignError(f"72 calibration overlay lacks campaign photometry for ObsNums {missing}")
    return copy.deepcopy(low_level), astrometry, photometry


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        rendered = ", ".join(str(path) for path in paths) or "none"
        raise CampaignError(f"{label} must resolve to exactly one file; found {rendered}")
    return paths[0]


def observation_inputs(analysis_root: Path, raw_root: Path, obsnum: int) -> dict[str, Path]:
    analysis = require_directory(analysis_root, "analysis root")
    raw = require_directory(raw_root, "raw root")
    reduced = require_directory(analysis / "reduced", "analysis reduced directory")
    telescope = _one(sorted(reduced.glob(f"tel_toltec_*_{obsnum}_00_0002_recomputed.nc")), f"ObsNum {obsnum} telescope")
    apt = _one(sorted(reduced.glob(f"apt_{obsnum}_000_0002_*.ecsv")), f"ObsNum {obsnum} matched-input APT")
    raw_by_network: dict[int, Path] = {}
    for path in sorted(raw.glob(f"toltec*_{obsnum}_000_0002_*.nc")):
        match = RAW_RE.fullmatch(path.name)
        if match is None or int(match.group("obsnum")) != obsnum:
            continue
        network = int(match.group("network"))
        if network in raw_by_network:
            raise CampaignError(f"ObsNum {obsnum} has ambiguous raw network {network}: {raw_by_network[network]}, {path}")
        raw_by_network[network] = path
    if not raw_by_network:
        raise CampaignError(f"ObsNum {obsnum} has no exact scannum-2 TolTEC raw files")
    values: dict[str, Path] = {"matched_input_apt": apt, "telescope": telescope}
    values.update({f"toltec{network}": path for network, path in sorted(raw_by_network.items())})
    return values


def direct_config(
    *, low_level: dict[str, Any], astrometry: dict[str, Any], photometry: dict[str, Any],
    inputs: dict[str, Path], prior: Path, fitreport_root: Path, output: Path, obsnum: int, threads: int,
) -> dict[str, Any]:
    if threads < 1:
        raise CampaignError("--threads must be positive")
    config = copy.deepcopy(low_level)
    beammap = _mapping(config.get("beammap"), "70 low_level.beammap")
    kids = _mapping(config.get("kids"), "70 low_level.kids")
    solver = _mapping(kids.get("solver"), "70 low_level.kids.solver")
    runtime = _mapping(config.get("runtime"), "70 low_level.runtime")
    mapmaking = _mapping(config.get("mapmaking"), "70 low_level.mapmaking")
    prior_config = _mapping(beammap.get("priors"), "70 low_level.beammap.priors")
    sidecar = beammap.get("detector_tod_output")
    if not isinstance(sidecar, dict):
        sidecar = {}
        beammap["detector_tod_output"] = sidecar
    raw_items = []
    for role, path in sorted(inputs.items()):
        if role.startswith("toltec"):
            raw_items.append({"filepath": str(path), "meta": {"interface": role}})
    config["inputs"] = [{
        "cal_items": [copy.deepcopy(astrometry), copy.deepcopy(photometry),
                      {"filepath": str(inputs["matched_input_apt"]), "meta": {"interface": "apt"}, "type": "array_prop_table"}],
        "data_items": [{"filepath": str(inputs["telescope"]), "meta": {"interface": "lmt"}}, *raw_items],
        "meta": {"name": f"{obsnum}_0_2"},
    }]
    prior_config["filepath"] = str(prior)
    solver["fitreportdir"] = str(fitreport_root)
    runtime["n_threads"] = threads
    runtime["output_dir"] = str(output)
    runtime["parallel_policy"] = "omp"
    runtime["use_subdir"] = True
    # The campaign preserves raw timestamps and row identity, but admits only
    # detector rows with native telescope bracketing. This keeps early legacy
    # observations fail-closed against extrapolation while recording any
    # excluded endpoint rows in the alignment provenance.
    runtime["crop_detector_to_telescope_support"] = True
    # Detector-resolved PTC TOD is a distinct diagnostic product. Citlali's
    # typed contract requires detector map grouping whenever it is enabled.
    mapmaking["grouping"] = "detector"
    sidecar["enabled"] = True
    sidecar["subdir_name"] = "source_crossing_tod"
    return config


def _manifest(rows: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for role, path in sorted(rows.items())
    ]


def _write_replay(
    *, campaign_root: Path, obsnum: int, batch: int, low_level: dict[str, Any], astrometry: dict[str, Any],
    photometry: dict[str, Any], analysis_root: Path, raw_root: Path, repo_root: Path, prior: Path,
    fitreport_root: Path, executable: Path, source_authority: dict[str, Any], threads: int,
) -> dict[str, Any]:
    inputs = observation_inputs(analysis_root, raw_root, obsnum)
    output = campaign_root / f"replay_o{obsnum}"
    if output.exists():
        raise CampaignError(f"replay output already exists: {output}")
    config = direct_config(
        low_level=low_level, astrometry=astrometry, photometry=photometry, inputs=inputs, prior=prior,
        fitreport_root=fitreport_root, output=output / "reduced", obsnum=obsnum, threads=threads,
    )
    rendered = yaml.safe_dump(config, sort_keys=False)
    config_name = f"citlali_o{obsnum}_sci_align_reproduction.yaml"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config").mkdir()
    (output / "evidence").mkdir()
    config_path = output / "config" / config_name
    atomic_write(config_path, rendered)
    preparation = {
        "schema": SCHEMA,
        "purpose": "owner-authorized source-isolated diagnostic reduction; no production correction authorization",
        "observation_number": obsnum,
        "batch": batch,
        "numbered_config_authority": source_authority,
        "citlali": {"path": str(executable), "sha256": sha256_file(executable)},
        "inputs": _manifest({**inputs, "beammap_prior": prior}),
        "generated_config": {"relative_path": f"config/{config_name}", "sha256": hashlib.sha256(rendered.encode()).hexdigest()},
        "fitreport_directory": str(fitreport_root),
        "source_products_modified": False,
        "detector_tod_requested": True,
    }
    atomic_write(output / "evidence" / "preparation.json", canonical_json(preparation))
    atomic_write(output / "evidence" / "input_manifest.json", canonical_json(preparation["inputs"]))
    run = output / f"run_o{obsnum}_reproduction.sh"
    atomic_write(run, "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"{json.dumps(str(executable))} {json.dumps(str(config_path))} --grppiex omp",
        f"replay_root={json.dumps(str(output))}",
        "find \"$replay_root\" -type f ! -name SHA256SUMS -print0 \\",
        "  | LC_ALL=C sort -z | xargs -0 shasum -a 256 > \"$replay_root/SHA256SUMS\"", "",
    ]))
    run.chmod(run.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    submit = output / f"submit_o{obsnum}_reproduction.sbatch"
    atomic_write(submit, "\n".join([
        "#!/usr/bin/env bash", f"#SBATCH --job-name=sci-align-{obsnum}", "#SBATCH --time=48:00:00",
        "#SBATCH --mem=64G", f"#SBATCH --cpus-per-task={threads}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --partition=toltec-cpu", "#SBATCH --parsable", "set -euo pipefail", json.dumps(str(run)), "",
    ]))
    atomic_write(output / "SHA256SUMS", checksum_lines(output))
    return {"observation_number": obsnum, "batch": batch, "replay_root": str(output)}


def describe(base: Path, calibration: Path) -> dict[str, Any]:
    _low_level, _astrometry, photometry = authority(base, calibration)
    return {
        "schema": SCHEMA,
        "base_70_reduce": {"path": str(require_file(base, "70 policy")), "sha256": sha256_file(require_file(base, "70 policy"))},
        "calibration_72_reduce": {"path": str(require_file(calibration, "72 calibration overlay")), "sha256": sha256_file(require_file(calibration, "72 calibration overlay"))},
        "existing_verified_replay_observation_number": EXISTING_REPLAY_OBSNUM,
        "new_replays": [{"observation_number": obsnum, "batch": batch, "photometry_present": obsnum in photometry} for obsnum, batch in CAMPAIGN],
    }


def selected_campaign(obsnums: list[int] | None) -> tuple[tuple[int, int], ...]:
    if not obsnums:
        return CAMPAIGN
    requested = set(obsnums)
    if len(requested) != len(obsnums):
        raise CampaignError("--obsnum may be supplied at most once per observation")
    available = {obsnum for obsnum, _batch in CAMPAIGN}
    unknown = sorted(requested - available)
    if unknown:
        raise CampaignError(f"--obsnum is not a campaign member: {unknown}")
    return tuple(row for row in CAMPAIGN if row[0] in requested)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    campaign_root = args.output_root.expanduser().resolve()
    if campaign_root.exists():
        raise CampaignError(f"campaign output root already exists: {campaign_root}")
    base = require_file(args.base_config, "70 policy")
    calibration = require_file(args.calibration_config, "72 calibration overlay")
    executable = require_executable(args.citlali_bin)
    analysis_root = require_directory(args.analysis_root, "analysis root")
    raw_root = require_directory(args.raw_root, "raw root")
    repo_root = require_directory(args.repo_root, "repository root")
    prior = require_file(args.beammap_prior or repo_root / PRIOR_RELATIVE, "beammap prior")
    low_level, astrometry, photometry = authority(base, calibration)
    campaign = selected_campaign(args.obsnum)
    source_authority = {
        "70_reduce": {"path": str(base), "sha256": sha256_file(base), "used_for": "shared low_level policy"},
        "72_reduce": {"path": str(calibration), "sha256": sha256_file(calibration), "used_for": "per-ObsNum photometry and astrometry"},
        "merge_interpretation": "direct Citlali input is generated from the numbered requested configuration; output/runtime, absolute fitreport directory, prior path, and detector-TOD sidecar are explicit diagnostic bindings",
    }
    campaign_root.mkdir(parents=True, exist_ok=False)
    results = []
    for obsnum, batch in campaign:
        results.append(_write_replay(
            campaign_root=campaign_root, obsnum=obsnum, batch=batch, low_level=low_level, astrometry=astrometry,
            photometry=photometry[obsnum], analysis_root=analysis_root, raw_root=raw_root, repo_root=repo_root,
            prior=prior, fitreport_root=raw_root, executable=executable, source_authority=source_authority, threads=args.threads,
        ))
    index = {
        "schema": SCHEMA,
        "source_products_modified": False,
        "citlali_reductions_launched": 0,
        "selection": "full_campaign" if campaign == CAMPAIGN else "explicit_subset",
        "replays": results,
    }
    atomic_write(campaign_root / "campaign_preparation.json", canonical_json(index))
    for batch in range(1, 5):
        rows = [row for row in results if row["batch"] == batch]
        if not rows:
            continue
        script = ["#!/usr/bin/env bash", "set -euo pipefail"]
        for row in rows:
            submit = Path(row["replay_root"]) / f"submit_o{row['observation_number']}_reproduction.sbatch"
            script += [f"job_id=$(sbatch {json.dumps(str(submit))})", f"printf 'obsnum={row['observation_number']} job_id=%s\\n' \"$job_id\""]
        path = campaign_root / f"submit_batch_{batch:02d}.sh"
        atomic_write(path, "\n".join(script) + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    atomic_write(campaign_root / "PREPARATION_SHA256SUMS", checksum_lines(campaign_root))
    return {"campaign_root": str(campaign_root), "replays": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (commands.add_parser("describe"), commands.add_parser("prepare")):
        command.add_argument("--base-config", required=True, type=Path)
        command.add_argument("--calibration-config", required=True, type=Path)
    prepare_parser = commands.choices["prepare"]
    prepare_parser.add_argument("--analysis-root", required=True, type=Path)
    prepare_parser.add_argument("--raw-root", required=True, type=Path)
    prepare_parser.add_argument("--repo-root", required=True, type=Path)
    prepare_parser.add_argument("--citlali-bin", required=True, type=Path)
    prepare_parser.add_argument("--output-root", required=True, type=Path)
    prepare_parser.add_argument("--beammap-prior", type=Path)
    prepare_parser.add_argument("--threads", type=int, default=6)
    prepare_parser.add_argument(
        "--obsnum", type=int, action="append",
        help="Prepare only this declared campaign ObsNum; repeat for a reviewed subset.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = describe(args.base_config, args.calibration_config) if args.command == "describe" else prepare(args)
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
