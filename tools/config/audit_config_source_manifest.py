#!/usr/bin/env python3
"""Audit the ordered Citlali CLI config-source provenance contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MANIFEST_SOURCE = "include/citlali/core/pipeline/config_source_manifest.h"
COPY_SOURCE = "include/citlali/core/pipeline/output_config_copy.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"


def count(text: str, token: str) -> int:
    return text.count(token)


def audit(repo_root: Path) -> dict[str, object]:
    manifest = (repo_root / MANIFEST_SOURCE).read_text(encoding="utf-8")
    copy = (repo_root / COPY_SOURCE).read_text(encoding="utf-8")
    cli = (repo_root / CLI_SOURCE).read_text(encoding="utf-8")
    checks = {
        "schema_v1": "citlali-config-source-manifest-v1" in manifest,
        "ordered_override_semantics": (
            "ordered_later_sources_override" in manifest
        ),
        "tolteca_upstream_authority": (
            'root["upstream"]["authority"]' in manifest
        ),
        "merged_snapshot": "citlali_merged_config.yaml" in manifest,
        "source_sha256": count(manifest, "sha256_file(copied_path)") == 1,
        "merged_sha256": count(manifest, "sha256_file(merged_path)") == 1,
        "collision_safe_copy": "source_" in copy and "matching_basenames" in copy,
        "required_cli_write": count(cli, "write_config_source_manifest(") == 1,
    }
    return {"checks": checks, "drift": not all(checks.values())}


def markdown(result: dict[str, object]) -> str:
    lines = ["# Config Source Manifest Audit", ""]
    lines.append(f"- Drift: `{result['drift']}`")
    for name, passed in result["checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(Path(args.repo_root).resolve())
    report = markdown(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(report)
    print(f"config source manifest: drift={result['drift']}")
    return int(args.fail_on_drift and result["drift"])


if __name__ == "__main__":
    raise SystemExit(main())
