#!/usr/bin/env python3
"""Prepare a checksum-bound, source-isolated 3C273/148670 Beammap replay.

This tool does not invoke Citlali.  It derives one low-level configuration from
the accepted Beammap policy, binds the exact 148670 raw/telescope/APT inputs,
and writes only to an owner-specified reproduction directory.  The resulting
configuration requests the existing detector-resolved PTC TOD sidecar required
by SCI-ALIGN-001's left/right diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


OBSNUM = 148670
NETWORKS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12)
RAW_STEM = "148670_000_0002_2026_01_13_11_59_10"
APT_NAME = "apt_148670_000_0002_2026_01_13_11_59_10.ecsv"
TELESCOPE_NAME = "tel_toltec_2026-01-13_148670_00_0002_recomputed.nc"
PRIOR_RELATIVE = Path("data/beammap_priors/beammap_slot_priors_soft_v1.ecsv")
ARCHIVED_CONFIG_SHA256 = "d81ac8b1aa52c06c0ef7d69158c802850499695aa9d614ebaf996147ba736788"
ARCHIVED_APT_SHA256 = "19182ba57ff8a94b036cd35f9d4a14c6a9f9201ceb08ba6a0202d12b15e80a39"
ARCHIVED_TELESCOPE_SHA256 = "e39f5b9e3066fd20086105964dd915ff67709142d699e8a18bb58cfd9da6b7ae"
ARCHIVED_POLICY_SHA256 = "fbb27e4beaf82abf970a1f0d1dd9e1bfafd432d9ea4c0334aa7d775f3e3467cf"
ARCHIVED_RAW_SHA256 = {
    0: "6795e1c342371a232f2d62d635616eb485206935ead36a95fbc11343e39bb260",
    1: "03c95bce5feecbc6b69213462382b1707c14c45c555eeacdcd329e708c132cf9",
    2: "930bd5b7994f3881d1cd852206de44dd64aed884b18b2a41c630a44a549db7e8",
    3: "0438224d19713ace44a76b9056c7147f3ade1652f7d5143a298946e2dbc8e767",
    4: "a5b9e6455e4fd404fdf7b9c616d5dbc91b519ecc144359d03afc28f8e465e8a2",
    5: "927ad049a15382da14234044d43fad1c12df27eb72601d93250fe8fd9076216c",
    7: "161c23a6c941246b46537aea8c4607c7374687d70286ce96a8784c671d57f312",
    8: "e94974b7ce8265e07d184574e9f3f5f09117c6bcdc7ebb2942b353e4dcf3647b",
    9: "8de1ffa74b94256ff67cbb5be9213210635b81cd14dfa1d0ed1dc1a872a9b908",
    11: "5db856c114b4efebf2c976c18a920214a5cb112469c4f82d7b8596aee327d924",
    12: "b4b968f36658c5c0014b2e678110c8aaa12ce81fbe6ee43b95567342c4edf7fb",
}
CONFIG_NAME = "citlali_o148670_0_2_c1_sci_align_reproduction.yaml"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICY = REPO / "config/tolteca/v2/beammap/60_beammap_internal_policy.yaml"


class PreparationError(RuntimeError):
    """Raised when a reproducibility input or output contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


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
        raise PreparationError(f"{label} is missing or is not a regular file: {resolved}")
    return resolved


def require_executable(path: Path) -> Path:
    resolved = require_file(path, "Citlali executable")
    if not os.access(resolved, os.X_OK):
        raise PreparationError(f"Citlali executable is not executable: {resolved}")
    return resolved


def input_paths(analysis_root: Path, raw_root: Path, repo_root: Path) -> dict[str, Path]:
    analysis = analysis_root.expanduser().resolve()
    raw = raw_root.expanduser().resolve()
    repo = repo_root.expanduser().resolve()
    values: dict[str, Path] = {
        "matched_input_apt": analysis / "reduced" / APT_NAME,
        "telescope": analysis / "reduced" / TELESCOPE_NAME,
        "beammap_prior": repo / PRIOR_RELATIVE,
    }
    values.update(
        {
            f"toltec{network}": raw / f"toltec{network}_{RAW_STEM}.nc"
            for network in NETWORKS
        }
    )
    return values


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    resolved = require_file(path, "Beammap policy")
    measured = sha256_file(resolved)
    if measured != ARCHIVED_POLICY_SHA256:
        raise PreparationError(
            "accepted Beammap policy differs from the archived 148670 replay authority: "
            f"expected {ARCHIVED_POLICY_SHA256}, measured {measured}"
        )
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    try:
        policy = document["reduce"]["steps"][0]["config"]["low_level"]
    except (KeyError, IndexError, TypeError) as error:
        raise PreparationError(f"Beammap policy lacks reduce.steps[0].config.low_level: {path}") from error
    if not isinstance(policy, dict):
        raise PreparationError(f"Beammap policy low_level value is not a mapping: {path}")
    return copy.deepcopy(policy)


def reproduction_config(
    *,
    analysis_root: Path,
    raw_root: Path,
    repo_root: Path,
    output_root: Path,
    threads: int,
) -> dict[str, Any]:
    if threads < 1:
        raise PreparationError("--threads must be positive")
    paths = input_paths(analysis_root, raw_root, repo_root)
    config = load_policy()
    config["inputs"] = [
        {
            "cal_items": [
                {
                    "pointing_offsets": [
                        {
                            "axes_name": "az",
                            "value_arcsec": [-9.114653534540448, -10.152703285217285],
                        },
                        {
                            "axes_name": "alt",
                            "value_arcsec": [0.4350562268725868, -1.4902329444885254],
                        },
                        {
                            "modified_julian_date": [61053.49491131658, 61053.53868002251],
                        },
                    ],
                    "type": "astrometry",
                },
                {
                    "beammap_source": {
                        "fluxes": [
                            {"array_name": "a1100", "uncertainty_mJy": 0.05, "value_mJy": 3127.216072341717},
                            {"array_name": "a1400", "uncertainty_mJy": 0.05, "value_mJy": 3774.111995796148},
                            {"array_name": "a2000", "uncertainty_mJy": 0.05, "value_mJy": 4986.577299339246},
                        ]
                    },
                    "type": "photometry",
                },
                {"filepath": str(paths["matched_input_apt"]), "meta": {"interface": "apt"}, "type": "array_prop_table"},
            ],
            "data_items": [
                {"filepath": str(paths["telescope"]), "meta": {"interface": "lmt"}},
                *[
                    {"filepath": str(paths[f"toltec{network}"]), "meta": {"interface": f"toltec{network}"}}
                    for network in NETWORKS
                ],
            ],
            "meta": {"name": "148670_0_2"},
        }
    ]
    config["beammap"]["priors"]["filepath"] = str(paths["beammap_prior"])
    config["runtime"]["n_threads"] = int(threads)
    config["runtime"]["output_dir"] = str(output_root)
    config["runtime"]["parallel_policy"] = "omp"
    config["runtime"]["use_subdir"] = True
    config["beammap"]["detector_tod_output"]["enabled"] = True
    config["beammap"]["detector_tod_output"]["subdir_name"] = "source_crossing_tod"
    return config


def input_manifest(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for role, path in sorted(paths.items()):
        resolved = require_file(path, role)
        rows.append(
            {
                "role": role,
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return rows


def verify_archived_inputs(rows: Iterable[dict[str, Any]]) -> None:
    digests = {row["role"]: row["sha256"] for row in rows}
    expected = {
        "matched_input_apt": ARCHIVED_APT_SHA256,
        "telescope": ARCHIVED_TELESCOPE_SHA256,
        **{f"toltec{network}": digest for network, digest in ARCHIVED_RAW_SHA256.items()},
    }
    mismatches = [
        f"{role}: expected {digest}, measured {digests.get(role)}"
        for role, digest in expected.items()
        if digests.get(role) != digest
    ]
    if mismatches:
        raise PreparationError("archived 148670 input identity mismatch: " + "; ".join(mismatches))


def output_is_safe(output: Path, sources: Iterable[Path]) -> Path:
    resolved = output.expanduser().resolve()
    if resolved.exists():
        raise PreparationError(f"output root must not already exist: {resolved}")
    for source in sources:
        source_root = source.parent.resolve()
        if resolved.is_relative_to(source_root):
            raise PreparationError(f"output root overlaps source directory {source_root}: {resolved}")
    return resolved


def checksum_lines(directory: Path) -> str:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    return "\n".join(rows) + "\n"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    analysis_root = args.analysis_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    executable = require_executable(args.citlali_bin)
    paths = input_paths(analysis_root, raw_root, repo_root)
    rows = input_manifest(paths)
    verify_archived_inputs(rows)
    output = output_is_safe(output_root, paths.values())
    config = reproduction_config(
        analysis_root=analysis_root,
        raw_root=raw_root,
        repo_root=repo_root,
        output_root=output / "reduced",
        threads=args.threads,
    )
    rendered = yaml.safe_dump(config, sort_keys=False)
    preparation = {
        "schema": "sci-align-001-148670-reproduction-preparation-v1",
        "purpose": "isolated diagnostic reproduction before SCI-ALIGN-001 corpus analysis",
        "observation_number": OBSNUM,
        "archived_reference": {
            "config_sha256": ARCHIVED_CONFIG_SHA256,
            "matched_input_apt_sha256": ARCHIVED_APT_SHA256,
            "telescope_sha256": ARCHIVED_TELESCOPE_SHA256,
            "policy_path": str(POLICY.resolve()),
            "policy_sha256": sha256_file(POLICY),
            "policy_equivalence": "matches archived 148670 low-level policy except inputs, runtime, and prior path",
        },
        "citlali": {"path": str(executable), "sha256": sha256_file(executable)},
        "inputs": rows,
        "generated_config": {"relative_path": f"config/{CONFIG_NAME}", "sha256": hashlib.sha256(rendered.encode()).hexdigest()},
        "output_root": str(output),
        "source_products_modified": False,
        "detector_tod_requested": True,
    }
    if args.dry_run:
        return preparation
    output.mkdir(parents=True, exist_ok=False)
    config_directory = output / "config"
    evidence_directory = output / "evidence"
    config_directory.mkdir()
    evidence_directory.mkdir()
    config_path = config_directory / CONFIG_NAME
    atomic_write(config_path, rendered)
    atomic_write(evidence_directory / "preparation.json", canonical_json(preparation))
    atomic_write(evidence_directory / "input_manifest.json", canonical_json(rows))
    command = " ".join(
        [
            json.dumps(str(executable)),
            json.dumps(str(config_path)),
            "--grppiex",
            "omp",
        ]
    )
    run_script = output / "run_148670_reproduction.sh"
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            command,
            f"replay_root={json.dumps(str(output))}",
            "find \"$replay_root\" -type f ! -name SHA256SUMS -print0 \\",
            "  | LC_ALL=C sort -z | xargs -0 shasum -a 256 > \"$replay_root/SHA256SUMS\"",
            "",
        ]
    )
    atomic_write(run_script, script)
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    slurm_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=sci-align-148670",
            "#SBATCH --time=48:00:00",
            "#SBATCH --mem=64G",
            f"#SBATCH --cpus-per-task={args.threads}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --partition=toltec-cpu",
            "#SBATCH --parsable",
            "set -euo pipefail",
            json.dumps(str(run_script)),
            "",
        ]
    )
    submit_script = output / "submit_148670_reproduction.sbatch"
    atomic_write(submit_script, slurm_script)
    atomic_write(output / "SHA256SUMS", checksum_lines(output))
    return preparation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--citlali-bin", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = prepare(parse_args(argv))
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
