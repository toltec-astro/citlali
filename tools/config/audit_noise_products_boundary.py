#!/usr/bin/env python3
"""Audit the frozen noise-products config and execution boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/noise_products_legacy_paths.json"
READER_SOURCE = "include/citlali/core/pipeline/noise_config_read.h"
CONFIG_SOURCE = "include/citlali/core/config/noise_config.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/mapmaking_config_impl.h"
ACCESSOR_SOURCE = "include/citlali/core/pipeline/reduction_config_accessors.h"
ADAPTER_SOURCE = "include/citlali/core/pipeline/noise_config_adapter.h"
MAP_FILTER_POLICY_SOURCE = (
    "include/citlali/core/pipeline/map_filter_config_policy.h"
)
MAP_FILTER_BOUNDARY_SOURCE = (
    "include/citlali/core/engine/detail/map_filter_config_impl.h"
)
PROVENANCE_SOURCE = "include/citlali/core/pipeline/noise_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
RNG_SOURCES = (
    "include/citlali/core/engine/detail/pointing_pipeline_impl.h",
    "include/citlali/core/engine/detail/lali_setup_pipeline_impl.h",
    "include/citlali/core/engine/detail/beammap_run_loop_impl.h",
)
EXPECTED_MANIFEST_SCHEMA = (
    "citlali-frozen-noise-products-config-paths-v1"
)
EXPECTED_PATHS = [
    "noise_maps.enabled",
    "noise_maps.n_noise_maps",
    "noise_maps.products.apply_empirical_weights",
    "noise_maps.products.enabled",
    "noise_maps.randomize_dets",
    "noise_maps.write_realizations",
]
EXPECTED_REQUIRED_PATHS = {
    "noise_maps.enabled",
    "noise_maps.n_noise_maps",
    "noise_maps.randomize_dets",
}
EXPECTED_OPTIONAL_PATHS = set(EXPECTED_PATHS) - EXPECTED_REQUIRED_PATHS
EXPECTED_PATH_SHA256 = (
    "e5e23109f40bb155047bef945faf8e0ae28e08783ee5a0a7a49abd56eb82023b"
)
EXPECTED_PROVENANCE_SCHEMA = "citlali-noise-products-provenance-v1"
RETIRED_SYMBOLS = (
    "read_noise_map_config",
    "read_noise_product_config",
    "read_noise_maps_enabled_config",
    "read_noise_map_count_config",
    "read_noise_randomize_dets_config",
    "read_noise_write_realizations_config",
    "read_noise_products_enabled_config",
    "read_noise_empirical_weights_config",
    "mirror_noise_map_settings_to_coadd",
    "disable_noise_map_settings",
    "set_noise_maps_enabled",
)


def call_count(source_text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source_text))


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError("noise-products paths must be a string sequence")
    if paths != sorted(set(paths)):
        raise ValueError("noise-products paths must be sorted and unique")
    digest = path_digest(paths)
    exact = bool(
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("path_count") == len(paths) == 6
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


def config_read_paths(source_text: str, function_name: str) -> list[str]:
    paths = []
    for call in re.findall(
        rf"\b{re.escape(function_name)}\s*\((.*?)\);",
        source_text,
        flags=re.DOTALL,
    ):
        tuple_match = re.search(r"std::tuple\s*\{(.*?)\}", call, re.DOTALL)
        if tuple_match is None:
            continue
        keys = re.findall(r'"([^"]+)"', tuple_match.group(1))
        if keys:
            paths.append(".".join(keys))
    return paths


def reader_state(source_text: str) -> dict[str, object]:
    required_paths = config_read_paths(source_text, "read_config_value")
    optional_paths = config_read_paths(
        source_text, "read_optional_config_value"
    )
    covered = sorted(set(required_paths) | set(optional_paths))
    exact = bool(
        len(required_paths) == 3
        and len(optional_paths) == 3
        and set(required_paths) == EXPECTED_REQUIRED_PATHS
        and set(optional_paths) == EXPECTED_OPTIONAL_PATHS
        and "read_mirrored_config_value" not in source_text
        and "read_optional_mirrored_config_value" not in source_text
    )
    return {
        "source": READER_SOURCE,
        "required_read_count": len(required_paths),
        "optional_read_count": len(optional_paths),
        "required_paths": required_paths,
        "optional_paths": optional_paths,
        "covered_path_count": len(covered),
        "exact": exact,
    }


def authority_state(
    config: str, boundary: str, accessor: str, adapter: str,
) -> dict[str, object]:
    read_count = call_count(boundary, "read_noise_request_config")
    reset_count = len(re.findall(
        r"\bnoise_plan\s*\.\s*reset_from_request\s*\(", boundary
    ))
    adapter_count = call_count(boundary, "adapt_noise_config_one_way")
    read_position = boundary.find("read_noise_request_config(")
    reset_position = boundary.find("noise_plan.reset_from_request(")
    adapter_position = boundary.find("adapt_noise_config_one_way(")
    effective_accessor = "engine.noise_plan.effective" in accessor
    one_way_adapter = "const citlali::config::NoiseConfig &effective" in adapter
    combined = "\n".join((config, boundary, adapter))
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


def rng_state(source_texts: list[str]) -> dict[str, object]:
    explicit_seed_counts = [
        text.count("noise_random_seed") for text in source_texts
    ]
    unseeded_counts = [
        len(re.findall(r"boost::random::mt19937\s+eng\s*;", text))
        for text in source_texts
    ]
    exact = explicit_seed_counts == [1, 1, 1] and not any(unseeded_counts)
    return {
        "sources": list(RNG_SOURCES),
        "explicit_seed_counts": explicit_seed_counts,
        "unseeded_counts": unseeded_counts,
        "exact": exact,
    }


def execution_policy_state(
    policy_source: str, boundary_source: str
) -> dict[str, object]:
    direct_requested_reads = policy_source.count("reduction_config.noise")
    effective_accessor_count = call_count(boundary_source, "noise_config")
    exact = direct_requested_reads == 0 and effective_accessor_count == 1
    return {
        "policy_source": MAP_FILTER_POLICY_SOURCE,
        "boundary_source": MAP_FILTER_BOUNDARY_SOURCE,
        "direct_requested_read_count": direct_requested_reads,
        "effective_accessor_count": effective_accessor_count,
        "exact": exact,
    }


def provenance_state(provenance: str, cli: str) -> dict[str, object]:
    schema_count = provenance.count(EXPECTED_PROVENANCE_SCHEMA)
    write_count = call_count(cli, "write_noise_provenance_file")
    completion_count = call_count(cli, "record_noise_run_completed")
    exact = schema_count == write_count == completion_count == 1
    return {
        "schema_version": EXPECTED_PROVENANCE_SCHEMA,
        "schema_count": schema_count,
        "cli_write_count": write_count,
        "cli_completion_count": completion_count,
        "exact": exact,
    }


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(json.loads(
        (repo_root / MANIFEST_SOURCE).read_text()
    ))
    reader = reader_state((repo_root / READER_SOURCE).read_text())
    authority = authority_state(
        (repo_root / CONFIG_SOURCE).read_text(),
        (repo_root / BOUNDARY_SOURCE).read_text(),
        (repo_root / ACCESSOR_SOURCE).read_text(),
        (repo_root / ADAPTER_SOURCE).read_text(),
    )
    rng = rng_state([
        (repo_root / source).read_text() for source in RNG_SOURCES
    ])
    execution_policy = execution_policy_state(
        (repo_root / MAP_FILTER_POLICY_SOURCE).read_text(),
        (repo_root / MAP_FILTER_BOUNDARY_SOURCE).read_text(),
    )
    provenance = provenance_state(
        (repo_root / PROVENANCE_SOURCE).read_text(),
        (repo_root / CLI_SOURCE).read_text(),
    )
    drift = not all(
        state["exact"]
        for state in (
            manifest, reader, authority, rng, execution_policy, provenance
        )
    )
    return {
        "manifest": manifest,
        "typed_reader": reader,
        "authority_boundary": authority,
        "randomization": rng,
        "execution_policy": execution_policy,
        "provenance": provenance,
        "drift": drift,
    }


def markdown_report(result: dict[str, object]) -> str:
    return "\n".join([
        "# Noise-Products Config Boundary Audit",
        "",
        f"- Drift: `{result['drift']}`",
        f"- Frozen paths: `{result['manifest']['path_count']}`",
        f"- Direct typed reader exact: `{result['typed_reader']['exact']}`",
        f"- Authority boundary exact: `{result['authority_boundary']['exact']}`",
        f"- RNG identity exact: `{result['randomization']['exact']}`",
        f"- Effective execution policy exact: `{result['execution_policy']['exact']}`",
        f"- Versioned provenance exact: `{result['provenance']['exact']}`",
        "",
    ])


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
        "noise-products config boundary: "
        f"paths={result['manifest']['path_count']} "
        f"reader={result['typed_reader']['exact']} "
        f"authority={result['authority_boundary']['exact']} "
        f"rng={result['randomization']['exact']} "
        f"execution_policy={result['execution_policy']['exact']} "
        f"provenance={result['provenance']['exact']} "
        f"drift={result['drift']}"
    )
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
