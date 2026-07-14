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
VALIDATION_SOURCE = (
    "include/citlali/core/config/post_processing_config_validation.h"
)
DIRECT_READER_SOURCE = (
    "include/citlali/core/pipeline/post_processing_config_read.h"
)
SHADOW_SOURCE = (
    "include/citlali/core/pipeline/post_processing_config_shadow.h"
)
PLAN_SOURCE = (
    "include/citlali/core/pipeline/post_processing_execution_plan.h"
)
LIFECYCLE_SOURCE = (
    "include/citlali/core/pipeline/post_processing_provenance_lifecycle.h"
)
SERIALIZATION_SOURCE = (
    "include/citlali/core/pipeline/post_processing_config_serialization.h"
)
PROVENANCE_SOURCE = (
    "include/citlali/core/pipeline/post_processing_provenance.h"
)
CLI_EXECUTION_SOURCE = "include/citlali/core/cli/reduction_execution.h"
ENGINE_BOUNDARY_SOURCE = "include/citlali/core/engine/detail/citlali_config_impl.h"
FILTER_BOUNDARY_SOURCE = "include/citlali/core/engine/detail/map_filter_config_impl.h"
LEGACY_FILTER_SOURCE = "include/citlali/core/mapmaking/wiener_filter.h"
LEGACY_FILTER_OMP_SOURCE = (
    "include/citlali/core/mapmaking/wiener_filter_omp.h"
)
MIRROR_SOURCE = "include/citlali/core/pipeline/map_filter_config_policy.h"
ACTIVATION_READER_SOURCE = (
    "include/citlali/core/pipeline/post_processing_activation_config_read.h"
)
MAPMAKING_OUTPUT_SOURCE = (
    "include/citlali/core/pipeline/mapmaking_output_config.h"
)
LEGACY_FITTING_READER_SOURCE = (
    "include/citlali/core/pipeline/citlali_config_read_post_processing.h"
)
FITTING_POLICY_SOURCE = (
    "include/citlali/core/pipeline/source_fitting_config_policy.h"
)
LEGACY_FINDING_READER_SOURCE = (
    "include/citlali/core/pipeline/citlali_config_read_source_finding.h"
)
FINDING_POLICY_SOURCE = (
    "include/citlali/core/pipeline/source_finding_config_policy.h"
)
SOURCE_CALLBACK_SOURCE = (
    "include/citlali/core/pipeline/map_source_config_callbacks.h"
)
ACCESSOR_SOURCE = "include/citlali/core/pipeline/reduction_config_accessors.h"
BEAMMAP_PLAN_SOURCE = (
    "include/citlali/core/pipeline/beammap_execution_plan.h"
)
OUTPUT_POLICY_SOURCE = "include/citlali/core/pipeline/output_policy.h"
MAPDIAG_OUTPUT_SOURCE = (
    "include/citlali/core/engine/detail/mapdiag_output_impl.h"
)

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
    validation_source = (repo_root / VALIDATION_SOURCE).read_text()
    direct_reader = (repo_root / DIRECT_READER_SOURCE).read_text()
    shadow_path = repo_root / SHADOW_SOURCE
    shadow_source = shadow_path.read_text() if shadow_path.exists() else ""
    plan_source = (repo_root / PLAN_SOURCE).read_text()
    lifecycle_source = (repo_root / LIFECYCLE_SOURCE).read_text()
    serialization_source = (repo_root / SERIALIZATION_SOURCE).read_text()
    provenance_source = (repo_root / PROVENANCE_SOURCE).read_text()
    cli_execution_source = (repo_root / CLI_EXECUTION_SOURCE).read_text()
    engine = (repo_root / ENGINE_BOUNDARY_SOURCE).read_text()
    filter_boundary = (repo_root / FILTER_BOUNDARY_SOURCE).read_text()
    legacy_filter = (repo_root / LEGACY_FILTER_SOURCE).read_text()
    legacy_filter_omp = (repo_root / LEGACY_FILTER_OMP_SOURCE).read_text()
    mirror = (repo_root / MIRROR_SOURCE).read_text()
    activation_reader_path = repo_root / ACTIVATION_READER_SOURCE
    activation_reader = (
        activation_reader_path.read_text()
        if activation_reader_path.exists() else ""
    )
    mapmaking_output = (repo_root / MAPMAKING_OUTPUT_SOURCE).read_text()
    legacy_fitting_reader_path = repo_root / LEGACY_FITTING_READER_SOURCE
    fitting_policy = (repo_root / FITTING_POLICY_SOURCE).read_text()
    legacy_finding_reader_path = repo_root / LEGACY_FINDING_READER_SOURCE
    legacy_finding_reader = (
        legacy_finding_reader_path.read_text()
        if legacy_finding_reader_path.exists() else ""
    )
    finding_policy = (repo_root / FINDING_POLICY_SOURCE).read_text()
    source_callbacks = (repo_root / SOURCE_CALLBACK_SOURCE).read_text()
    accessor = (repo_root / ACCESSOR_SOURCE).read_text()
    beammap_plan = (repo_root / BEAMMAP_PLAN_SOURCE).read_text()
    output_policy = (repo_root / OUTPUT_POLICY_SOURCE).read_text()
    mapdiag_output = (repo_root / MAPDIAG_OUTPUT_SOURCE).read_text()

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
        "legacy_shadow_retired": (
            not shadow_path.exists()
            and "compare_post_processing_config_shadow" not in engine
        ),
        "typed_request_precedes_mapmaking_setup": (
            engine.index("read_post_processing_request_config")
            < engine.index("get_mapmaking_config")
        ),
        "duplicate_histogram_reader_retired": (
            'std::tuple{"post_processing", "map_histogram_n_bins"}'
            not in mapmaking_output
        ),
        "execution_plan_present": (
            "struct PostProcessingExecutionPlan" in plan_source
            and "PostProcessingEffectiveResolutionRecord" in plan_source
            and "PostProcessingRealizedState" in plan_source
            and "requested" in plan_source
            and "effective" in plan_source
            and "realized" in plan_source
        ),
        "realized_cardinality_present": all(
            field in plan_source
            for field in (
                "filter_context_count",
                "filtered_map_count",
                "source_finding_context_count",
                "detected_source_count",
                "source_table_write_count",
                "source_table_row_count",
                "pointing_raw_fits",
                "pointing_filtered_fits",
                "beammap_fits",
                "attempt_count",
                "valid_count",
            )
        ),
        "source_finding_requires_filtering": (
            "config.source_finding.enabled &&"
            " !config.map_filtering.enabled" in validation_source
            and "requires post_processing.map_filtering.enabled=true"
            in validation_source
        ),
        "realized_lifecycle_present": all(
            name in lifecycle_source
            for name in (
                "record_post_processing_filter_completed",
                "record_post_processing_catalog_fits_completed",
                "record_post_processing_source_table_written",
                "record_post_processing_pointing_fits_completed",
                "record_post_processing_beammap_fits_completed",
                "record_post_processing_run_completed",
            )
        ),
        "realized_lifecycle_checks_cardinality": all(
            text in lifecycle_source
            for text in (
                "source row cardinality is inconsistent",
                "pointing fit cardinality is incomplete",
                "beammap reduction recorded no fitting contexts",
            )
        ),
        "provenance_schema_present": (
            "citlali-post-processing-provenance-v1" in provenance_source
            and "post_processing_provenance.yaml" in provenance_source
            and "post_processing_config_node(plan.requested)"
            in provenance_source
            and "post_processing_config_node(plan.effective)"
            in provenance_source
            and "post_processing_realized_state_node(plan.realized)"
            in provenance_source
            and "write_yaml_file_atomic" in provenance_source
            and "post_processing_fit_cardinality_node"
            in serialization_source
        ),
        "cli_completion_call_count": call_count(
            cli_execution_source, "record_post_processing_run_completed"
        ),
        "cli_provenance_write_call_count": call_count(
            cli_execution_source, "write_post_processing_provenance_file"
        ),
        "execution_plan_reset_call_count": call_count(
            engine, "post_processing_plan.reset_from_request"
        ),
        "execution_plan_accessor_count": accessor.count(
            "return engine.post_processing_plan;"
        ),
        "serial_filter_parser_retired": (
            "WienerFilter::get_config" not in legacy_filter
            and "void get_config" not in legacy_filter
        ),
        "omp_filter_parser_retired": (
            "WienerFilter::get_config" not in legacy_filter_omp
            and "void get_config" not in legacy_filter_omp
        ),
        "legacy_filter_boundary_call_count": call_count(
            filter_boundary, "read_processor_config"
        ),
        "reverse_filter_mirror_call_count": call_count(
            filter_boundary, "mirror_wiener_filter_config"
        ),
        "typed_filter_adapter_present": (
            "void adapt_map_filter_config_one_way" in mirror
            and "template_fwhm_rad.clear" in mirror
            and "map_filter_template_uses_fwhm" in mirror
        ),
        "typed_filter_adapter_call_count": call_count(
            filter_boundary, "adapt_map_filter_config_one_way"
        ),
        "effective_filter_accessor_call_count": call_count(
            filter_boundary, "effective_post_processing_config"
        ),
        "filter_output_policy_is_effective": (
            "effective_post_processing_config(engine)" in output_policy
            and "effective_post_processing_config(*this)" in mapdiag_output
        ),
        "activation_reader_call_count": call_count(
            engine, "read_post_processing_activation_config"
        ),
        "activation_reader_retired": (
            not activation_reader_path.exists()
            and "read_post_processing_activation_config" not in engine
        ),
        "source_fitting_parser_retired": (
            not legacy_fitting_reader_path.exists()
            and "read_source_fitting_config" not in engine
        ),
        "source_fitting_parser_call_count": call_count(
            engine, "read_source_fitting_config"
        ),
        "typed_source_fitting_adapter_present": (
            "void adapt_source_fitting_config_one_way" in fitting_policy
            and "config.bounding_box_arcsec" in fitting_policy
            and "config.fitting_radius_arcsec" in fitting_policy
            and "config.fit_rotation_angle" in fitting_policy
            and "config.amp_limit_factors" in fitting_policy
            and "config.fwhm_limit_factors" in fitting_policy
        ),
        "typed_source_fitting_adapter_call_count": call_count(
            engine, "adapt_source_fitting_config_one_way"
        ),
        "effective_source_fitting_policy_used": (
            "source_fitting_active(" in engine
            and "post_processing_plan.effective.source_fitting" in engine
        ),
        "source_fitting_shadow_details_retired": (
            not shadow_source
        ),
        "source_finding_parser_retired": (
            not legacy_finding_reader_path.exists()
            and "read_source_finding_config" not in engine
        ),
        "source_finding_parser_call_count": call_count(
            engine, "read_source_finding_config"
        ),
        "typed_source_finding_adapter_present": (
            "void adapt_source_finding_config_one_way" in finding_policy
            and "config.source_sigma" in finding_policy
            and "config.source_window_arcsec * arcsec_to_rad"
            in finding_policy
            and "config.mode" in finding_policy
        ),
        "typed_source_finding_adapter_call_count": call_count(
            engine, "adapt_source_finding_config_one_way"
        ),
        "effective_source_finding_policy_used": (
            "source_finding_active(\n            post_processing_plan.effective)"
            in engine
            and "post_processing_plan.effective.source_finding" in engine
        ),
        "source_finding_output_policy_is_effective": (
            "bool source_finding_enabled" in output_policy
            and "source_finding_active(\n        effective_post_processing_config(engine))"
            in output_policy
        ),
        "source_finding_shadow_details_retired": (
            not shadow_source
        ),
        "post_load_request_mutation_count": call_count(
            engine, "disable_map_products_if_mapmaking_disabled"
        ),
        "beammap_disabled_iteration_call_count": call_count(
            engine, "normalize_beammap_iterations_if_mapmaking_disabled"
        ),
        "request_accessor_count": accessor.count(
            "return reduction_config(engine).post_processing;"
        ),
        "source_model_typed": (
            "SourceFitModel model" in config_source
            and bool(source_model_pattern.search(direct_reader))
        ),
        "kernel_tail_numerical_target": (
            "kernel_template_tail_mode" in legacy_filter
            and "kernel_template_tail_mode" in legacy_filter_omp
        ),
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
            "mirror_source_finding_config_to_coadd" in legacy_finding_reader
            or "mirror_source_finding_config_to_coadd" in source_callbacks
        ),
        "post_processing_request_mutation_retired": (
            "disable_map_products_if_mapmaking_disabled" not in engine
            and "set_map_filtering_enabled" not in engine
            and "set_source_finding_enabled" not in engine
            and "set_source_fitting_active" not in engine
        ),
        "beammap_disabled_iteration_policy_moved_to_plan": (
            "if (!mapmaking_enabled)" in beammap_plan
            and "effective_.iteration.max_iterations = 1" in beammap_plan
            and "max_iterations_forced_without_mapmaking" in beammap_plan
        ),
    }
    exact = bool(
        checks["authority_prefixes_exact"]
        and checks["complete_request_reader_present"]
        and checks["direct_request_reader_call_count"] == 1
        and checks["shadow_comparison_call_count"] == 0
        and checks["legacy_shadow_retired"]
        and checks["typed_request_precedes_mapmaking_setup"]
        and checks["duplicate_histogram_reader_retired"]
        and checks["execution_plan_present"]
        and checks["realized_cardinality_present"]
        and checks["source_finding_requires_filtering"]
        and checks["realized_lifecycle_present"]
        and checks["realized_lifecycle_checks_cardinality"]
        and checks["provenance_schema_present"]
        and checks["cli_completion_call_count"] == 1
        and checks["cli_provenance_write_call_count"] == 1
        and checks["execution_plan_reset_call_count"] == 1
        and checks["execution_plan_accessor_count"] == 2
        and checks["serial_filter_parser_retired"]
        and checks["omp_filter_parser_retired"]
        and checks["legacy_filter_boundary_call_count"] == 0
        and checks["reverse_filter_mirror_call_count"] == 0
        and checks["typed_filter_adapter_present"]
        and checks["typed_filter_adapter_call_count"] == 1
        and checks["effective_filter_accessor_call_count"] == 1
        and checks["filter_output_policy_is_effective"]
        and checks["activation_reader_retired"]
        and checks["activation_reader_call_count"] == 0
        and checks["source_fitting_parser_retired"]
        and checks["source_fitting_parser_call_count"] == 0
        and checks["typed_source_fitting_adapter_present"]
        and checks["typed_source_fitting_adapter_call_count"] == 1
        and checks["effective_source_fitting_policy_used"]
        and checks["source_fitting_shadow_details_retired"]
        and checks["source_finding_parser_retired"]
        and checks["source_finding_parser_call_count"] == 0
        and checks["typed_source_finding_adapter_present"]
        and checks["typed_source_finding_adapter_call_count"] == 1
        and checks["effective_source_finding_policy_used"]
        and checks["source_finding_output_policy_is_effective"]
        and checks["source_finding_shadow_details_retired"]
        and checks["post_load_request_mutation_count"] == 0
        and checks["beammap_disabled_iteration_call_count"] == 0
        and checks["request_accessor_count"] == 2
        and checks["source_model_typed"]
        and checks["kernel_tail_numerical_target"]
        and checks["kernel_tail_typed"]
        and not checks["reverse_filter_mirror_present"]
        and not checks["kernel_tail_reverse_mirrored"]
        and not checks["source_finding_reverse_mirror_present"]
        and checks["post_processing_request_mutation_retired"]
        and checks["beammap_disabled_iteration_policy_moved_to_plan"]
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
        "authority_boundary": boundary,
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
        f"- Authority boundary exact: `{result['authority_boundary']['exact']}`",
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
