#!/usr/bin/env python3
"""Run the compact-config validation preflight used by the refactor."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd).returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        default="/private/tmp/citlali_config_preflight",
        help="Directory for generated JSON/CSV/Markdown reports.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail when any local compatibility baseline listed in the case suite is missing.",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Do not fail when user-facing compact coverage gaps remain.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(repo_root),
        help="Repository root. Defaults to the parent of tools/config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    scripts = [
        "tools/config/expand_compact_config.py",
        "tools/config/lowlevel_to_compact_config.py",
        "tools/config/run_compact_compatibility.py",
        "tools/config/audit_compact_surface_coverage.py",
        "tools/config/classify_lowlevel_config.py",
        "tools/config/compare_lowlevel_yaml.py",
        "tools/config/validate_config_authority_inventory.py",
    ]
    commands: list[list[str]] = [
        [sys.executable, "-m", "py_compile", *scripts],
        [
            sys.executable,
            "tools/config/run_compact_compatibility.py",
            "--fail-on-warnings",
            "--work-dir",
            str(work_dir / "compact_compat"),
            "--json-out",
            str(work_dir / "compact_compat/results.json"),
            "--markdown-out",
            str(work_dir / "compact_compat/results.md"),
        ],
        [
            sys.executable,
            "tools/config/audit_compact_surface_coverage.py",
            "--json-out",
            str(work_dir / "surface_coverage.json"),
            "--csv-out",
            str(work_dir / "surface_coverage.csv"),
            "--markdown-out",
            str(work_dir / "surface_coverage.md"),
        ],
        [
            sys.executable,
            "tools/config/validate_config_authority_inventory.py",
        ],
    ]
    if args.require_all:
        commands[1].append("--require-all")
        commands[2].append("--require-all")
    if not args.allow_gaps:
        commands[2].append("--fail-on-gaps")

    for command in commands:
        rc = run(command, cwd=repo_root)
        if rc != 0:
            return rc
    print(f"config preflight reports written under {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
