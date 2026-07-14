#!/usr/bin/env python3
"""Audit the frozen mixed post-processing config boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


MANIFEST_SOURCE = "tools/config/post_processing_legacy_paths.json"
DEFAULT_CONFIG_SOURCE = "data/config.yaml"
AUTHORITY_SOURCE = "tools/config/config_authority_inventory.json"
CONFIG_SOURCE = "include/citlali/core/config/post_processing_config.h"
DIRECT_READER_SOURCE = (
    "include/citlali/core/pipeline/post_processing_config_read.h"
)
SHADOW_SOURCE = (
    "include/citlali/core/pipeline/post_processing_config_shadow.h"
)
ENGINE_BOUNDARY_SOURCE = "include/citlali/core/engine/detail/citlali_config_impl.h"
FILTER_BOUNDARY_SOURCE = "include/citlali/core/engine/detail/map_filter_config_impl.h"
LEGACY_FILTER_SOURCE = "include/citlali/core/mapmaking/wiener_filter.h"
MIRROR_SOURCE = "include/citlali/core/pipeline/map_filter_config_policy.h"
POST_READER_SOURCE = (
    "include/citlali/core/pipeline/citlali_config_read_post_processing.h"
)
FINDING_READER_SOURCE = (
    "include/citlali/core/pipeline/citlali_config_read_source_finding.h"
)
ACCESSOR_SOURCE = "include/citlali/core/pipeline/reduction_config_accessors.h"
ACTIVATION_SOURCE = "include/citlali/core/pipeline/mapmaking_activation_policy.h"

EXPECTED_SCHEMA = "citlali-frozen-post-processing-config-paths-v1"
EXPECTED_PREFIXES = ["post_processing", "wiener_filter"]
EXPECTED_PATH_COUNT = 35
EXPECTED_PATH_SHA256 = (
    "cb39f13ee971f1e079a91ee595f8c0b15b9364326c479bedbf21f2d0ef6ae2a7"
)
EXPECTED_TYPED_GAPS: list[str] = []


def call_count(source: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source))


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def flatten_paths(value: object, prefix: list[str]) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(flatten_paths(child, [*prefix, str(key)]))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(flatten_paths(child, [*prefix, str(index)]))
        return paths
    return [".".join(prefix)]


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(path, str) for path in paths
    ):
        raise ValueError("post-processing paths must be a string sequence")
    if paths != sorted(set(paths)):
        raise ValueError("post-processing paths must be sorted and unique")
    digest = path_digest(paths)
    exact = bool(
        manifest.get("schema_version") == EXPECTED_SCHEMA
        and manifest.get("path_count") == len(paths) == EXPECTED_PATH_COUNT
        and manifest.get("path_sha256")
        == digest
        == EXPECTED_PATH_SHA256
        and manifest.get("config_prefixes") == EXPECTED_PREFIXES
        and manifest.get("known_typed_gaps") == EXPECTED_TYPED_GAPS
    )
    return {
        "source": MANIFEST_SOURCE,
        "path_count": len(paths),
        "path_sha256": digest,
        "paths": paths,
        "known_typed_gaps": manifest.get("known_typed_gaps"),
        "exact": exact,
    }


def default_surface_state(
    config: dict[str, object], manifest_paths: list[str]
) -> dict[str, object]:
    paths: list[str] = []
    for prefix in EXPECTED_PREFIXES:
        if prefix not in config:
            raise ValueError(f"default config is missing {prefix}")
        paths.extend(flatten_paths(config[prefix], [prefix]))
    paths = sorted(paths)
    return {
        "source": DEFAULT_CONFIG_SOURCE,
        "path_count": len(paths),
        "path_sha256": path_digest(paths),
        "missing_paths": sorted(set(manifest_paths) - set(paths)),
        "extra_paths": sorted(set(paths) - set(manifest_paths)),
        "exact": paths == manifest_paths,
    }


def boundary_state(repo_root: Path) -> dict[str, object]:
    authority = json.loads((repo_root / AUTHORITY_SOURCE).read_text())
    domain = next(
        item for item in authority["domains"]
        if item["id"] == "post-processing"
    )
    config_source = (repo_root / CONFIG_SOURCE).read_text()
    direct_reader = (repo_root / DIRECT_READER_SOURCE).read_text()
    shadow_source = (repo_root / SHADOW_SOURCE).read_text()
    engine = (repo_root / ENGINE_BOUNDARY_SOURCE).read_text()
    filter_boundary = (repo_root / FILTER_BOUNDARY_SOURCE).read_text()
    legacy_filter = (repo_root / LEGACY_FILTER_SOURCE).read_text()
    mirror = (repo_root / MIRROR_SOURCE).read_text()
    post_reader = (repo_root / POST_READER_SOURCE).read_text()
    finding_reader = (repo_root / FINDING_READER_SOURCE).read_text()
    accessor = (repo_root / ACCESSOR_SOURCE).read_text()
    activation = (repo_root / ACTIVATION_SOURCE).read_text()

    source_model_pattern = re.compile(
        r'std::tuple\s*\{\s*"post_processing"\s*,\s*"source_fitting"'
        r'\s*,\s*"model"'
    )
    checks = {
        "authority_prefixes_exact": domain["config_prefixes"] == EXPECTED_PREFIXES,
        "complete_request_reader_present": (
            "void read_post_processing_request_config" in direct_reader
            and "void read_map_filter_request_config" in direct_reader
            and "void read_source_finding_request_config" in direct_reader
            and "void read_source_fitting_request_config" in direct_reader
        ),
        "direct_request_reader_call_count": call_count(
            engine, "read_post_processing_request_config"
        ),
        "shadow_comparison_call_count": call_count(
            engine, "compare_post_processing_config_shadow"
        ),
        "shadow_report_present": (
            "struct PostProcessingConfigShadowReport" in shadow_source
            and "compared_map_filter_details" in shadow_source
            and "compared_source_finding_details" in shadow_source
            and "compared_source_fitting_details" in shadow_source
        ),
        "legacy_filter_parser_present": (
            legacy_filter.count("void WienerFilter::get_config") == 1
        ),
        "legacy_filter_boundary_call_count": call_count(
            filter_boundary, "read_processor_config"
        ),
        "reverse_filter_mirror_call_count": call_count(
            filter_boundary, "mirror_wiener_filter_config"
        ),
        "activation_reader_call_count": call_count(
            engine, "read_post_processing_activation_config"
        ),
        "source_fitting_reader_call_count": call_count(
            engine, "read_source_fitting_config"
        ),
        "source_finding_reader_call_count": call_count(
            engine, "read_source_finding_config"
        ),
        "post_load_request_mutation_count": call_count(
            engine, "disable_map_products_if_mapmaking_disabled"
        ),
        "request_accessor_count": accessor.count(
            "return reduction_config(engine).post_processing;"
        ),
        "source_model_typed": (
            "SourceFitModel model" in config_source
            and bool(source_model_pattern.search(direct_reader))
        ),
        "kernel_tail_legacy": "kernel_template_tail_mode" in legacy_filter,
        "kernel_tail_typed": (
            "MapFilterKernelTailMode kernel_template_tail_mode" in config_source
            and '"kernel_template_tail_mode"' in direct_reader
        ),
        "reverse_filter_mirror_present": (
            "void mirror_wiener_filter_config" in mirror
        ),
        "kernel_tail_reverse_mirrored": (
            "parse_map_filter_kernel_tail_mode" in mirror
        ),
        "source_finding_reverse_mirror_present": (
            "mirror_source_finding_config_to_coadd" in finding_reader
        ),
        "request_mutation_present": (
            "set_map_filtering_enabled" in activation
            and "set_source_finding_enabled" in activation
            and "set_source_fitting_active" in activation
        ),
    }
    exact = bool(
        checks["authority_prefixes_exact"]
        and checks["complete_request_reader_present"]
        and checks["direct_request_reader_call_count"] == 1
        and checks["shadow_comparison_call_count"] == 1
        and checks["shadow_report_present"]
        and checks["legacy_filter_parser_present"]
        and checks["legacy_filter_boundary_call_count"] == 1
        and checks["reverse_filter_mirror_call_count"] == 1
        and checks["activation_reader_call_count"] == 1
        and checks["source_fitting_reader_call_count"] == 1
        and checks["source_finding_reader_call_count"] == 1
        and checks["post_load_request_mutation_count"] == 1
        and checks["request_accessor_count"] == 2
        and checks["source_model_typed"]
        and checks["kernel_tail_legacy"]
        and checks["kernel_tail_typed"]
        and checks["reverse_filter_mirror_present"]
        and checks["kernel_tail_reverse_mirrored"]
        and checks["source_finding_reverse_mirror_present"]
        and checks["request_mutation_present"]
    )
    return {"checks": checks, "exact": exact}


def audit(repo_root: Path) -> dict[str, object]:
    manifest_document = json.loads(
        (repo_root / MANIFEST_SOURCE).read_text()
    )
    manifest = manifest_state(manifest_document)
    default_config = yaml.safe_load(
        (repo_root / DEFAULT_CONFIG_SOURCE).read_text()
    )
    default_surface = default_surface_state(
        default_config, manifest["paths"]
    )
    boundary = boundary_state(repo_root)
    return {
        "manifest": manifest,
        "default_surface": default_surface,
        "mixed_boundary": boundary,
        "drift": not (
            manifest["exact"]
            and default_surface["exact"]
            and boundary["exact"]
        ),
    }


def markdown_report(result: dict[str, object]) -> str:
    typed_gaps = result["manifest"]["known_typed_gaps"] or ["none"]
    return "\n".join([
        "# Post-Processing Config Boundary Audit",
        "",
        f"- Drift: `{result['drift']}`",
        f"- Frozen paths: `{result['manifest']['path_count']}`",
        f"- Default surface exact: `{result['default_surface']['exact']}`",
        f"- Mixed boundary exact: `{result['mixed_boundary']['exact']}`",
        "- Known typed gaps: `" + ", ".join(
            typed_gaps
        ) + "`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    result = audit(args.repo_root.resolve())
    report = markdown_report(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report)
    print(report, end="")
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
