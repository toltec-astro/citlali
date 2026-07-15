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
        "tools/config/audit_config_authority_reads.py",
        "tools/config/audit_config_leaf_contract.py",
        "tools/config/test_audit_config_leaf_contract.py",
        "tools/config/audit_mapmaking_boundary.py",
        "tools/config/test_audit_mapmaking_boundary.py",
        "tools/config/audit_coadd_boundary.py",
        "tools/config/test_audit_coadd_boundary.py",
        "tools/config/audit_noise_products_boundary.py",
        "tools/config/test_audit_noise_products_boundary.py",
        "tools/config/audit_pointing_boundary.py",
        "tools/config/test_audit_pointing_boundary.py",
        "tools/config/audit_polarimetry_boundary.py",
        "tools/config/test_audit_polarimetry_boundary.py",
        "tools/config/audit_astrometry_boundary.py",
        "tools/config/test_audit_astrometry_boundary.py",
        "tools/config/audit_post_processing_boundary.py",
        "tools/config/test_audit_post_processing_boundary.py",
        "tools/config/audit_beammap_boundary.py",
        "tools/config/test_audit_beammap_boundary.py",
        "tools/config/audit_kids_external_boundary.py",
        "tools/config/test_audit_kids_external_boundary.py",
        "tools/config/audit_config_source_manifest.py",
        "tools/config/test_audit_config_source_manifest.py",
        "tools/config/audit_processed_timestream_boundary.py",
        "tools/config/test_audit_processed_timestream_boundary.py",
        "tools/config/audit_raw_timestream_boundary.py",
        "tools/config/test_audit_raw_timestream_boundary.py",
        "tools/config/audit_raw_timestream_execution_reads.py",
        "tools/config/test_audit_raw_timestream_execution_reads.py",
        "tools/config/test_validate_config_authority_inventory.py",
        "tools/config/classify_lowlevel_config.py",
        "tools/config/compare_lowlevel_yaml.py",
        "tools/config/validate_config_authority_inventory.py",
    ]
    commands: list[list[str]] = [
        [sys.executable, "-m", "py_compile", *scripts],
        [
            sys.executable,
            "-m",
            "unittest",
            "tools.config.test_audit_processed_timestream_boundary",
            "tools.config.test_audit_mapmaking_boundary",
            "tools.config.test_audit_coadd_boundary",
            "tools.config.test_audit_noise_products_boundary",
            "tools.config.test_audit_pointing_boundary",
            "tools.config.test_audit_polarimetry_boundary",
            "tools.config.test_audit_astrometry_boundary",
            "tools.config.test_audit_post_processing_boundary",
            "tools.config.test_audit_beammap_boundary",
            "tools.config.test_audit_kids_external_boundary",
            "tools.config.test_audit_config_source_manifest",
            "tools.config.test_audit_raw_timestream_boundary",
            "tools.config.test_audit_raw_timestream_execution_reads",
            "tools.config.test_validate_config_authority_inventory",
            "tools.config.test_audit_config_leaf_contract",
        ],
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
        [
            sys.executable,
            "tools/config/audit_config_leaf_contract.py",
            "--json-out",
            str(work_dir / "config_leaf_contract.json"),
        ],
        [
            sys.executable,
            "tools/config/audit_config_authority_reads.py",
            "--json-out",
            str(work_dir / "config_read_census.json"),
            "--markdown-out",
            str(work_dir / "config_read_census.md"),
            "--fail-on-review",
        ],
        [
            sys.executable,
            "tools/config/audit_processed_timestream_boundary.py",
            "--json-out",
            str(work_dir / "processed_timestream_boundary.json"),
            "--markdown-out",
            str(work_dir / "processed_timestream_boundary.md"),
            "--fail-on-drift",
            "--fail-on-uncovered",
        ],
        [
            sys.executable,
            "tools/config/audit_mapmaking_boundary.py",
            "--json-out",
            str(work_dir / "mapmaking_boundary.json"),
            "--markdown-out",
            str(work_dir / "mapmaking_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_coadd_boundary.py",
            "--json-out",
            str(work_dir / "coadd_boundary.json"),
            "--markdown-out",
            str(work_dir / "coadd_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_noise_products_boundary.py",
            "--json-out",
            str(work_dir / "noise_products_boundary.json"),
            "--markdown-out",
            str(work_dir / "noise_products_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_pointing_boundary.py",
            "--json-out",
            str(work_dir / "pointing_boundary.json"),
            "--markdown-out",
            str(work_dir / "pointing_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_polarimetry_boundary.py",
            "--json-out",
            str(work_dir / "polarimetry_boundary.json"),
            "--markdown-out",
            str(work_dir / "polarimetry_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_astrometry_boundary.py",
            "--json-out",
            str(work_dir / "astrometry_boundary.json"),
            "--markdown-out",
            str(work_dir / "astrometry_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_post_processing_boundary.py",
            "--json-out",
            str(work_dir / "post_processing_boundary.json"),
            "--markdown-out",
            str(work_dir / "post_processing_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_beammap_boundary.py",
            "--json-out",
            str(work_dir / "beammap_boundary.json"),
            "--markdown-out",
            str(work_dir / "beammap_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_kids_external_boundary.py",
            "--json-out",
            str(work_dir / "kids_external_boundary.json"),
            "--markdown-out",
            str(work_dir / "kids_external_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_config_source_manifest.py",
            "--json-out",
            str(work_dir / "config_source_manifest.json"),
            "--markdown-out",
            str(work_dir / "config_source_manifest.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_raw_timestream_boundary.py",
            "--json-out",
            str(work_dir / "raw_timestream_boundary.json"),
            "--markdown-out",
            str(work_dir / "raw_timestream_boundary.md"),
            "--fail-on-drift",
        ],
        [
            sys.executable,
            "tools/config/audit_raw_timestream_execution_reads.py",
            "--json-out",
            str(work_dir / "raw_timestream_execution_reads.json"),
            "--markdown-out",
            str(work_dir / "raw_timestream_execution_reads.md"),
            "--fail-on-drift",
            "--fail-on-review",
        ],
    ]
    if args.require_all:
        commands[2].append("--require-all")
        commands[3].append("--require-all")
        commands[5].append("--require-all")
    if not args.allow_gaps:
        commands[3].append("--fail-on-gaps")

    for command in commands:
        rc = run(command, cwd=repo_root)
        if rc != 0:
            return rc
    print(f"config preflight reports written under {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
