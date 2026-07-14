#!/usr/bin/env python3
"""Audit the frozen pointing config and execution boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/pointing_legacy_paths.json"
READER_SOURCE = "include/citlali/core/pipeline/pointing_config_read.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/pointing_config_impl.h"
ACCESSOR_SOURCE = "include/citlali/core/pipeline/reduction_config_accessors.h"
ADAPTER_SOURCE = "include/citlali/core/pipeline/pointing_config_adapter.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/pointing_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
EXPECTED_MANIFEST_SCHEMA = "citlali-frozen-pointing-config-paths-v1"
EXPECTED_PATHS = [
    "pointing.source_strategy.fit_gaussian",
    "pointing.source_strategy.fruitloops_center_mode",
    "pointing.source_strategy.header_max_radius_arcsec",
    "pointing.source_strategy.header_require_coverage",
    "pointing.source_strategy.mode",
]
EXPECTED_PATH_SHA256 = (
    "fdda04ef21cdea0c36f9f8b9766fac9b5ad4b7d2b673bbb4ed74cd627f688f13"
)
EXPECTED_PROVENANCE_SCHEMA = "citlali-pointing-provenance-v1"
RETIRED_SYMBOLS = ("read_pointing_source_strategy_config",)


def call_count(source_text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source_text))


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError("pointing paths must be a string sequence")
    if paths != sorted(set(paths)):
        raise ValueError("pointing paths must be sorted and unique")
    digest = path_digest(paths)
    exact = bool(
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("path_count") == len(paths) == 5
        and manifest.get("path_sha256") == digest == EXPECTED_PATH_SHA256
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
    enum_reads = call_count(source_text, "read_optional_pointing_enum") - 1
    scalar_reads = call_count(source_text, "read_optional_config_value")
    covered_paths = []
    for match in re.finditer(
        r"const\s+auto\s+\w+_key\s*=\s*std::tuple\s*\{(.*?)\};",
        source_text,
        flags=re.DOTALL,
    ):
        parts = re.findall(r'"([^"]+)"', match.group(1))
        if parts:
            covered_paths.append(".".join(parts))
    covered_paths.sort()
    exact = bool(
        enum_reads == 2
        and scalar_reads == 3
        and covered_paths == EXPECTED_PATHS
        and "read_mirrored_config_value" not in source_text
        and "read_optional_mirrored_config_value" not in source_text
        and "read_optional_parsed_mirrored_config_value" not in source_text
    )
    return {
        "source": READER_SOURCE,
        "enum_read_count": enum_reads,
        "scalar_read_count": scalar_reads,
        "covered_paths": covered_paths,
        "exact": exact,
    }


def authority_state(
    boundary: str, accessor: str, adapter: str, reader: str
) -> dict[str, object]:
    read_count = call_count(boundary, "read_pointing_request_config")
    reset_count = len(
        re.findall(r"\bpointing_plan\s*\.\s*reset_from_request\s*\(", boundary)
    )
    adapter_count = call_count(boundary, "adapt_pointing_config_one_way")
    read_position = boundary.find("read_pointing_request_config(")
    reset_position = boundary.find("pointing_plan.reset_from_request(")
    adapter_position = boundary.find("adapt_pointing_config_one_way(")
    effective_accessor = "engine.pointing_plan.effective" in accessor
    one_way_adapter = (
        "const citlali::config::PointingConfig &effective" in adapter
    )
    combined = "\n".join((boundary, reader))
    retired_counts = {
        symbol: combined.count(symbol) for symbol in RETIRED_SYMBOLS
    }
    ordered = 0 <= read_position < reset_position < adapter_position
    exact = bool(
        read_count == reset_count == adapter_count == 1
        and ordered
        and effective_accessor
        and one_way_adapter
        and not any(retired_counts.values())
    )
    return {
        "source": BOUNDARY_SOURCE,
        "read_count": read_count,
        "plan_reset_count": reset_count,
        "adapter_count": adapter_count,
        "order_exact": ordered,
        "effective_accessor": effective_accessor,
        "one_way_adapter": one_way_adapter,
        "retired_symbol_counts": retired_counts,
        "exact": exact,
    }


def provenance_state(provenance: str, cli: str) -> dict[str, object]:
    schema_count = provenance.count(EXPECTED_PROVENANCE_SCHEMA)
    write_count = call_count(cli, "write_pointing_provenance_file")
    completion_count = call_count(cli, "record_pointing_run_completed")
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
    reader_text = (repo_root / READER_SOURCE).read_text()
    reader = reader_state(reader_text)
    authority = authority_state(
        (repo_root / BOUNDARY_SOURCE).read_text(),
        (repo_root / ACCESSOR_SOURCE).read_text(),
        (repo_root / ADAPTER_SOURCE).read_text(),
        reader_text,
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
            "# Pointing Config Boundary Audit",
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
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        path = Path(args.markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(result))
    print(markdown_report(result), end="")
    return 2 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
