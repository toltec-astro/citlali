#!/usr/bin/env python3
"""Audit the frozen mapmaking config surface and typed authority boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/mapmaking_legacy_paths.json"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/mapmaking_config_impl.h"
MODEL_SOURCE = "include/citlali/core/config/mapmaking_config.h"
CORE_READER_SOURCE = (
    "include/citlali/core/pipeline/mapmaking_config_read_core.h"
)
METHOD_SOURCE = "include/citlali/core/pipeline/mapmaking_method_config.h"
OUTPUT_SOURCE = "include/citlali/core/pipeline/mapmaking_output_config.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/mapmaking_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
MAP_HEADER_SOURCE = "include/citlali/core/mapmaking/map.h"
MAP_IMPLEMENTATION_SOURCE = "src/citlali/core/mapmaking/map.cpp"
EXPECTED_MANIFEST_SCHEMA = "citlali-frozen-mapmaking-config-paths-v1"
EXPECTED_PATH_COUNT = 22
EXPECTED_PATH_SHA256 = (
    "fe72da6e9e0b6af1d63ad0cc465a0d00950332899d7938559276c0d873a54460"
)
EXPECTED_PROVENANCE_SCHEMA = "citlali-mapmaking-provenance-v2"

CORE_PATHS = {
    "mapmaking.enabled",
    "mapmaking.grouping",
    "mapmaking.method",
    "mapmaking.pixel_axes",
}
METHOD_PATHS = {
    "mapmaking.jinc_filter.r_max",
    "mapmaking.jinc_filter.shape_params.a1100",
    "mapmaking.jinc_filter.shape_params.a1400",
    "mapmaking.jinc_filter.shape_params.a2000",
    "mapmaking.jinc_filter.subpixel_n",
    "mapmaking.maximum_likelihood.max_iterations",
    "mapmaking.maximum_likelihood.tolerance",
}
OUTPUT_PATHS = {
    "mapmaking.coverage_cut",
    "mapmaking.crpix1",
    "mapmaking.crpix2",
    "mapmaking.crval1_J2000",
    "mapmaking.crval2_J2000",
    "mapmaking.cunit",
    "mapmaking.pixel_size_arcsec",
    "mapmaking.tan_dec",
    "mapmaking.tan_ra",
    "mapmaking.x_size_pix",
    "mapmaking.y_size_pix",
}

RETIRED_SYMBOLS = (
    "MapBuffer::get_config",
    "read_method_specific_mapmaker_config",
    "read_jinc_filter_config",
    "read_maximum_likelihood_mapmaker_config",
    "read_output_map_block_config",
    "read_coadd_map_block_config",
)


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def load_frozen_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError(f"invalid paths in {path}")
    if paths != sorted(set(paths)):
        raise ValueError(f"paths must be sorted and unique in {path}")
    return manifest


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = list(manifest["paths"])
    digest = path_digest(paths)
    expected_families = CORE_PATHS | METHOD_PATHS | OUTPUT_PATHS
    exact = (
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("path_count") == len(paths) == EXPECTED_PATH_COUNT
        and manifest.get("path_sha256") == digest == EXPECTED_PATH_SHA256
        and set(paths) == expected_families
    )
    return {
        "source": MANIFEST_SOURCE,
        "schema_version": manifest.get("schema_version"),
        "path_count": len(paths),
        "path_sha256": digest,
        "family_counts": {
            "core": len(CORE_PATHS),
            "method": len(METHOD_PATHS),
            "output": len(OUTPUT_PATHS),
        },
        "exact": exact,
        "paths": paths,
    }


def source_mentions_path(path: str, source_text: str) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    return f'"{leaf}"' in source_text


def reader_coverage(repo_root: Path) -> dict[str, object]:
    sources = {
        "core": (CORE_PATHS, (repo_root / CORE_READER_SOURCE).read_text()),
        "method": (
            METHOD_PATHS,
            (repo_root / METHOD_SOURCE).read_text()
            + (repo_root / MODEL_SOURCE).read_text(),
        ),
        "output": (
            OUTPUT_PATHS,
            (repo_root / OUTPUT_SOURCE).read_text(),
        ),
    }
    uncovered = sorted(
        path
        for paths, source_text in sources.values()
        for path in paths
        if not source_mentions_path(path, source_text)
    )
    return {
        "covered_count": EXPECTED_PATH_COUNT - len(uncovered),
        "uncovered": uncovered,
        "complete": not uncovered,
    }


def retired_parser_state(repo_root: Path) -> dict[str, object]:
    source_text = "\n".join(
        path.read_text()
        for root in (repo_root / "include", repo_root / "src")
        for path in root.rglob("*.h")
    )
    source_text += "\n" + "\n".join(
        path.read_text() for path in (repo_root / "src").rglob("*.cpp")
    )
    symbol_counts = {
        symbol: source_text.count(symbol) for symbol in RETIRED_SYMBOLS
    }
    map_sources = (
        (repo_root / MAP_HEADER_SOURCE).read_text()
        + (repo_root / MAP_IMPLEMENTATION_SOURCE).read_text()
    )
    forbidden_map_dependencies = sorted(
        token
        for token in ("YamlConfig", "get_config_value", "tula/config")
        if token in map_sources
    )
    return {
        "retired_symbol_counts": symbol_counts,
        "forbidden_map_dependencies": forbidden_map_dependencies,
        "retired": not any(symbol_counts.values())
        and not forbidden_map_dependencies,
    }


def call_count(source_text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source_text))


def authority_boundary(source_text: str) -> dict[str, object]:
    expected_counts = {
        "read_mapmaking_enabled_config": 1,
        "read_map_grouping_config": 1,
        "read_map_method_config": 1,
        "read_map_pixel_axes_config": 1,
        "read_mapmaking_output_request_config": 1,
        "read_mapmaking_method_request_config": 1,
        "adapt_mapmaking_output_config_one_way": 2,
        "adapt_jinc_filter_config_one_way": 1,
        "adapt_maximum_likelihood_config_one_way": 1,
        "mapmaking_plan": 1,
    }
    actual_counts = {
        name: call_count(source_text, name) for name in expected_counts
    }
    request_pos = source_text.find("read_mapmaking_output_request_config(")
    output_adapter_pos = source_text.find(
        "adapt_mapmaking_output_config_one_way("
    )
    method_pos = source_text.find("read_mapmaking_method_request_config(")
    method_adapter_positions = [
        source_text.find("adapt_jinc_filter_config_one_way("),
        source_text.find("adapt_maximum_likelihood_config_one_way("),
    ]
    plan_pos = source_text.find("mapmaking_plan.reset_from_request(")
    order_exact = (
        0 <= request_pos < output_adapter_pos < method_pos < plan_pos
        and all(method_pos < position < plan_pos
                for position in method_adapter_positions)
    )
    return {
        "source": BOUNDARY_SOURCE,
        "call_counts": actual_counts,
        "expected_call_counts": expected_counts,
        "order_exact": order_exact,
        "exact": actual_counts == expected_counts and order_exact,
    }


def provenance_state(repo_root: Path) -> dict[str, object]:
    provenance = (repo_root / PROVENANCE_SOURCE).read_text()
    cli = (repo_root / CLI_SOURCE).read_text()
    schema_count = provenance.count(EXPECTED_PROVENANCE_SCHEMA)
    write_count = call_count(cli, "write_mapmaking_provenance_file")
    completion_count = call_count(cli, "record_mapmaking_run_completed")
    exact = schema_count == 1 and write_count == 1 and completion_count == 1
    return {
        "schema_version": EXPECTED_PROVENANCE_SCHEMA,
        "schema_count": schema_count,
        "cli_write_count": write_count,
        "cli_completion_count": completion_count,
        "exact": exact,
    }


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(
        load_frozen_manifest(repo_root / MANIFEST_SOURCE)
    )
    readers = reader_coverage(repo_root)
    retired = retired_parser_state(repo_root)
    boundary = authority_boundary((repo_root / BOUNDARY_SOURCE).read_text())
    provenance = provenance_state(repo_root)
    drift = not (
        manifest["exact"]
        and readers["complete"]
        and retired["retired"]
        and boundary["exact"]
        and provenance["exact"]
    )
    return {
        "manifest": manifest,
        "typed_reader_coverage": readers,
        "retired_parser": retired,
        "authority_boundary": boundary,
        "provenance": provenance,
        "drift": drift,
    }


def markdown_report(result: dict[str, object]) -> str:
    coverage = result["typed_reader_coverage"]
    return "\n".join(
        [
            "# Mapmaking Config Boundary Audit",
            "",
            f"- Drift: `{result['drift']}`",
            f"- Frozen paths: `{result['manifest']['path_count']}`",
            f"- Typed reader coverage: `{coverage['covered_count']}/"
            f"{EXPECTED_PATH_COUNT}`",
            f"- Retired parser absent: `"
            f"{result['retired_parser']['retired']}`",
            f"- Authority boundary exact: `"
            f"{result['authority_boundary']['exact']}`",
            f"- Versioned provenance exact: `"
            f"{result['provenance']['exact']}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    result = audit(repo_root)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown_report(result))
    print(
        "mapmaking config boundary: "
        f"paths={result['manifest']['path_count']} "
        f"typed_coverage={result['typed_reader_coverage']['covered_count']}/"
        f"{EXPECTED_PATH_COUNT} "
        f"parser_retired={result['retired_parser']['retired']} "
        f"provenance={result['provenance']['exact']} "
        f"drift={result['drift']}"
    )
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
