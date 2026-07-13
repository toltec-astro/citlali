#!/usr/bin/env python3
"""Audit the frozen coadd config surface and typed authority boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/coadd_legacy_paths.json"
READER_SOURCE = "include/citlali/core/pipeline/coadd_config_read.h"
CONFIG_SOURCE = "include/citlali/core/config/coadd_config.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/mapmaking_config_impl.h"
ACCESSOR_SOURCE = "include/citlali/core/pipeline/reduction_config_accessors.h"
ACTIVATION_SOURCE = "include/citlali/core/pipeline/mapmaking_activation_policy.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/coadd_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
EXPECTED_MANIFEST_SCHEMA = "citlali-frozen-coadd-config-paths-v1"
EXPECTED_PATHS = ["coadd.enabled"]
EXPECTED_PATH_SHA256 = (
    "b4b87923304067219ffbc2da502fded0b867f850fa9f5254204fd9f7b537e957"
)
EXPECTED_PROVENANCE_SCHEMA = "citlali-coadd-provenance-v1"
RETIRED_SYMBOLS = ("read_coadd_enabled_config", "set_coadd_enabled")


def call_count(source_text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source_text))


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError("coadd paths must be a string sequence")
    if paths != sorted(set(paths)):
        raise ValueError("coadd paths must be sorted and unique")
    digest = path_digest(paths)
    exact = bool(
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("path_count") == len(paths) == 1
        and manifest.get("path_sha256")
        == digest
        == EXPECTED_PATH_SHA256
        and paths == EXPECTED_PATHS
    )
    return {
        "source": MANIFEST_SOURCE,
        "path_count": len(paths),
        "path_sha256": digest,
        "paths": paths,
        "exact": exact,
    }


def reader_state(source_text: str) -> dict[str, object]:
    direct_read_count = call_count(source_text, "read_config_value")
    exact = bool(
        direct_read_count == 1
        and '"coadd", "enabled"' in source_text
        and "read_mirrored_config_value" not in source_text
    )
    return {
        "source": READER_SOURCE,
        "direct_read_count": direct_read_count,
        "exact": exact,
    }


def authority_state(
    config: str, boundary: str, accessor: str, activation: str
) -> dict[str, object]:
    read_count = call_count(boundary, "read_coadd_request_config")
    reset_count = len(
        re.findall(r"\bcoadd_plan\s*\.\s*reset_from_request\s*\(", boundary)
    )
    read_position = boundary.find("read_coadd_request_config(")
    reset_position = boundary.find("coadd_plan.reset_from_request(")
    effective_accessor = "engine.coadd_plan.effective" in accessor
    retired_counts = {
        symbol: (
            config.count(symbol)
            + activation.count(symbol)
            + boundary.count(symbol)
        )
        for symbol in RETIRED_SYMBOLS
    }
    exact = bool(
        read_count == 1
        and reset_count == 1
        and 0 <= read_position < reset_position
        and effective_accessor
        and not any(retired_counts.values())
    )
    return {
        "source": BOUNDARY_SOURCE,
        "read_count": read_count,
        "plan_reset_count": reset_count,
        "order_exact": 0 <= read_position < reset_position,
        "effective_accessor": effective_accessor,
        "retired_symbol_counts": retired_counts,
        "exact": exact,
    }


def provenance_state(provenance: str, cli: str) -> dict[str, object]:
    schema_count = provenance.count(EXPECTED_PROVENANCE_SCHEMA)
    write_count = call_count(cli, "write_coadd_provenance_file")
    completion_count = call_count(cli, "record_coadd_run_completed")
    exact = schema_count == write_count == completion_count == 1
    return {
        "schema_version": EXPECTED_PROVENANCE_SCHEMA,
        "schema_count": schema_count,
        "cli_write_count": write_count,
        "cli_completion_count": completion_count,
        "exact": exact,
    }


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(
        json.loads((repo_root / MANIFEST_SOURCE).read_text())
    )
    reader = reader_state((repo_root / READER_SOURCE).read_text())
    authority = authority_state(
        (repo_root / CONFIG_SOURCE).read_text(),
        (repo_root / BOUNDARY_SOURCE).read_text(),
        (repo_root / ACCESSOR_SOURCE).read_text(),
        (repo_root / ACTIVATION_SOURCE).read_text(),
    )
    provenance = provenance_state(
        (repo_root / PROVENANCE_SOURCE).read_text(),
        (repo_root / CLI_SOURCE).read_text(),
    )
    drift = not (
        manifest["exact"]
        and reader["exact"]
        and authority["exact"]
        and provenance["exact"]
    )
    return {
        "manifest": manifest,
        "typed_reader": reader,
        "authority_boundary": authority,
        "provenance": provenance,
        "drift": drift,
    }


def markdown_report(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Coadd Config Boundary Audit",
            "",
            f"- Drift: `{result['drift']}`",
            f"- Frozen paths: `{result['manifest']['path_count']}`",
            f"- Direct typed reader exact: `{result['typed_reader']['exact']}`",
            f"- Authority boundary exact: `{result['authority_boundary']['exact']}`",
            f"- Versioned provenance exact: `{result['provenance']['exact']}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    result = audit(repo_root)
    report = markdown_report(result)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    if args.markdown_out:
        Path(args.markdown_out).write_text(report)
    print(
        "coadd config boundary: "
        f"paths={result['manifest']['path_count']} "
        f"reader={result['typed_reader']['exact']} "
        f"authority={result['authority_boundary']['exact']} "
        f"provenance={result['provenance']['exact']} "
        f"drift={result['drift']}"
    )
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
