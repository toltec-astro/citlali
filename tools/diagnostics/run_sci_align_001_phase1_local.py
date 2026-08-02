#!/usr/bin/env python3
"""Prepare and execute the two bounded SCI-ALIGN-001 local fixtures.

The tool is intentionally fail-if-exists and never edits the owner-supplied
validation suite.  It localizes only path bindings in a copied realized YAML,
records the exact input/binary/config identities, and writes logs and run
metadata below an explicitly supplied scratch root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path


BRANCH = "codex/repair-sci-align-001"
SUITE_ROOT = Path("/Users/gwilson/work_toltec/local_data/citlali-validation/v1")
REMOTE_SUITE_ROOT = "/work/toltec/citlali-validation/v1/"
CONFIGS = {
    "point": {
        "observation": "152389",
        "source": SUITE_ROOT
        / "point/pointings_v22/reduced/redu00/citlali_o152389_0_2_c1.yaml",
        "sha256": "340677ab1e873e735a44dcee84d7da9eba91a7c511f8d9229b044aa29d98f5ba",
        # Fourteen selected input paths plus the realized output_dir binding.
        "suite_path_binding_count": 15,
    },
    "beammap": {
        "observation": "148670",
        "source": SUITE_ROOT
        / "beammaps/3c273/reduced/redu00/citlali_o148670_0_2_c1.yaml",
        "sha256": "d81ac8b1aa52c06c0ef7d69158c802850499695aa9d614ebaf996147ba736788",
        # Thirteen selected input paths plus the realized output_dir binding.
        "suite_path_binding_count": 14,
    },
}
REMOTE_PRIOR = (
    "/work/toltec/citlali_dev/citlali/data/beammap_priors/"
    "beammap_slot_priors_soft_v1.ecsv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def require_clean_identity(repo: Path, expected_sha: str) -> dict[str, str]:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1")
    if branch != BRANCH or head != expected_sha or status:
        raise RuntimeError(
            "repair worktree identity is not the requested clean branch/SHA"
        )
    return {"branch": branch, "head": head, "status": "clean"}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new)


def selected_paths(config_text: str) -> list[Path]:
    values = re.findall(
        r"^\s*(?:-\s+)?filepath:\s+([^#\n]+?)\s*$",
        config_text,
        flags=re.MULTILINE,
    )
    paths = [Path(value.strip("'\"")) for value in values]
    if not paths or len(paths) != len(set(paths)):
        raise RuntimeError("localized config filepath inventory is empty or duplicated")
    return sorted(paths)


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare(args: argparse.Namespace, repo: Path, mode_root: Path) -> None:
    identity = require_clean_identity(repo, args.expected_sha)
    if mode_root.exists():
        raise FileExistsError(f"scratch mode root already exists: {mode_root}")
    config_dir = mode_root / "config"
    data_dir = mode_root / "data"
    reduced_dir = mode_root / "reduced"
    evidence_dir = mode_root / "evidence"
    for path in (config_dir, data_dir, reduced_dir, evidence_dir):
        path.mkdir(parents=True, exist_ok=False)

    spec = CONFIGS[args.mode]
    source = Path(spec["source"])
    if sha256(source) != spec["sha256"]:
        raise RuntimeError("owner-supplied realized config digest changed")
    text = source.read_text(encoding="utf-8")
    suite_replacements = text.count(REMOTE_SUITE_ROOT)
    if suite_replacements != spec["suite_path_binding_count"]:
        raise RuntimeError(
            "realized config suite-binding count changed: "
            f"expected {spec['suite_path_binding_count']}, got {suite_replacements}"
        )
    text = text.replace(REMOTE_SUITE_ROOT, f"{SUITE_ROOT}/")
    if args.mode == "beammap":
        text = replace_once(
            text,
            REMOTE_PRIOR,
            str(repo / "data/beammap_priors/beammap_slot_priors_soft_v1.ecsv"),
            "Beammap prior binding",
        )
    text = replace_once(
        text,
        "    fitreportdir: ../data",
        f"    fitreportdir: {data_dir}",
        "fitreportdir binding",
    )
    text, output_count = re.subn(
        r"(?m)^  output_dir: .*?$",
        f"  output_dir: {reduced_dir}",
        text,
    )
    if output_count != 1 or "/work/toltec/" in text:
        raise RuntimeError("localized config retained an unresolved runtime binding")

    localized = config_dir / source.name
    localized.write_text(text, encoding="utf-8")
    paths = selected_paths(text)
    if len(paths) != 14:
        raise RuntimeError(
            f"expected 14 selected input/prior files; found {len(paths)}"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("localized inputs are missing: " + ", ".join(missing))
    manifest = []
    for path in paths:
        manifest.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"candidate binary is not executable: {binary}")
    write_json_new(
        evidence_dir / "preparation.json",
        {
            "schema_version": "sci-align-001-phase1-local-preparation-v1",
            "mode": args.mode,
            "observation": spec["observation"],
            "repair_identity": identity,
            "source_config": {
                "path": str(source),
                "sha256": spec["sha256"],
            },
            "localized_config": {
                "path": str(localized),
                "sha256": sha256(localized),
                "suite_path_binding_count": suite_replacements,
            },
            "candidate_binary": {
                "path": str(binary),
                "size_bytes": binary.stat().st_size,
                "sha256": sha256(binary),
            },
            "selected_inputs": manifest,
        },
    )
    print(localized)


def execute(args: argparse.Namespace, repo: Path, mode_root: Path) -> None:
    identity = require_clean_identity(repo, args.expected_sha)
    evidence_dir = mode_root / "evidence"
    preparation_path = evidence_dir / "preparation.json"
    if not preparation_path.is_file():
        raise RuntimeError("fixture has not been prepared")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation["repair_identity"] != identity:
        raise RuntimeError("prepared and executing repair identities differ")
    binary = Path(preparation["candidate_binary"]["path"])
    if sha256(binary) != preparation["candidate_binary"]["sha256"]:
        raise RuntimeError("candidate binary changed after preparation")
    config = Path(preparation["localized_config"]["path"])
    if sha256(config) != preparation["localized_config"]["sha256"]:
        raise RuntimeError("localized config changed after preparation")

    log_path = evidence_dir / "citlali.log"
    summary_path = evidence_dir / "run_summary.json"
    if log_path.exists() or summary_path.exists():
        raise FileExistsError("fixture execution evidence already exists")
    start_utc = dt.datetime.now(dt.timezone.utc)
    start_ns = time.monotonic_ns()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(binary), str(config), "--grppiex", "omp"],
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    stop_ns = time.monotonic_ns()
    stop_utc = dt.datetime.now(dt.timezone.utc)
    output_dir = mode_root / "reduced" / "redu00"
    summary = {
        "schema_version": "sci-align-001-phase1-local-run-v1",
        "mode": args.mode,
        "observation": preparation["observation"],
        "repair_identity": identity,
        "command": [str(binary), str(config), "--grppiex", "omp"],
        "start_utc": start_utc.isoformat(),
        "stop_utc": stop_utc.isoformat(),
        "elapsed_monotonic_sec": (stop_ns - start_ns) / 1.0e9,
        "exit_code": completed.returncode,
        "rusage_children": {
            "platform_ru_maxrss_units": (
                "bytes" if platform.system() == "Darwin" else "KiB"
            ),
            "ru_maxrss_after": after.ru_maxrss,
            "ru_utime_delta_sec": after.ru_utime - before.ru_utime,
            "ru_stime_delta_sec": after.ru_stime - before.ru_stime,
        },
        "log": {"path": str(log_path), "sha256": sha256(log_path)},
        "expected_reduction_root": str(output_dir),
        "expected_reduction_root_exists": output_dir.is_dir(),
    }
    write_json_new(summary_path, summary)
    print(summary_path)
    if completed.returncode != 0 or not output_dir.is_dir():
        return_code = completed.returncode if completed.returncode != 0 else 2
        raise SystemExit(return_code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--mode", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/citlali"))
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    args.binary = (
        (repo / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    )
    mode_root = args.scratch_root.resolve() / args.mode
    if args.action == "prepare":
        prepare(args, repo, mode_root)
    else:
        execute(args, repo, mode_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
