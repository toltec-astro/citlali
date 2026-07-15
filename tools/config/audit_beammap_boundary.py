#!/usr/bin/env python3
"""Audit the frozen Beammap config surface and current typed boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


MANIFEST_SOURCE = "tools/config/beammap_legacy_paths.json"
DEFAULT_CONFIG_SOURCE = "data/config.yaml"
AUTHORITY_SOURCE = "tools/config/config_authority_inventory.json"
MODEL_SOURCE = "include/citlali/core/config/beammap_config.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/beammap_config_impl.h"
ADAPTER_SOURCE = "include/citlali/core/pipeline/beammap_config_tod_mirror.h"
SERIALIZER_SOURCE = (
    "include/citlali/core/pipeline/beammap_config_serialization.h"
)
PLAN_SOURCE = "include/citlali/core/pipeline/beammap_execution_plan.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/beammap_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
LIFECYCLE_SOURCE = (
    "include/citlali/core/pipeline/beammap_provenance_lifecycle.h"
)
EXPECTED_LIFECYCLE_CALLS = {
    "include/citlali/core/pipeline/reduction_iteration_setup.h": {
        "begin_beammap_run_if_available": 1,
    },
    "include/citlali/core/engine/detail/beammap_pipeline_entry_impl.h": {
        "begin_beammap_observation_if_available": 1,
    },
    "include/citlali/core/engine/detail/beammap_run_loop_impl.h": {
        "begin_beammap_internal_iteration_if_available": 1,
        "record_beammap_source_aware_rtc_if_available": 1,
        "complete_beammap_internal_iteration_if_available": 1,
    },
    "include/citlali/core/engine/detail/beammap_mapmaking_pass_impl.h": {
        "record_beammap_mapmaking_pass_completed_if_available": 1,
    },
    "include/citlali/core/engine/detail/beammap_fit_stage_impl.h": {
        "record_beammap_fitting_completed_if_available": 1,
    },
    "include/citlali/core/engine/detail/beammap_detector_tod_output_impl.h": {
        "record_beammap_detector_tod_written_if_available": 1,
    },
    "include/citlali/core/pipeline/observation_output_execution.h": {
        "complete_beammap_observation_if_available": 1,
    },
    CLI_SOURCE: {
        "record_beammap_run_completed": 1,
        "write_beammap_provenance_file": 1,
    },
}
READER_SOURCES = (
    "include/citlali/core/pipeline/beammap_config_loading.h",
    "include/citlali/core/pipeline/beammap_config_core_loading.h",
    "include/citlali/core/pipeline/beammap_config_fitting_flagging.h",
    "include/citlali/core/pipeline/beammap_config_priors_loading.h",
    "include/citlali/core/pipeline/beammap_config_split_outputs.h",
    "include/citlali/core/pipeline/beammap_config_tod_mirror.h",
)

EXPECTED_SCHEMA = "citlali-frozen-beammap-config-paths-v1"
EXPECTED_PATH_COUNT = 74
EXPECTED_PATH_SHA256 = (
    "23b89ad61e2c318eb2f2a37369968e3079101ff3f52e814ac4c00e786e542f97"
)
EXPECTED_LITERAL_FILES = {
    "include/citlali/core/config/beammap_config_validation.h",
    "include/citlali/core/config/reduction_config_validation.h",
    "include/citlali/core/pipeline/beammap_config_loading.h",
    "include/citlali/core/pipeline/beammap_config_core_loading.h",
    "include/citlali/core/pipeline/beammap_config_fitting_flagging.h",
    "include/citlali/core/pipeline/beammap_config_priors_loading.h",
    "include/citlali/core/pipeline/beammap_config_split_outputs.h",
    "include/citlali/core/pipeline/beammap_config_tod_mirror.h",
}
LITERAL_SCAN_EXCLUDED_FILES = {
    "include/citlali/core/pipeline/config_leaf_schema_generated.h",
}
EXPECTED_STRUCTS = {
    "BeammapConfig",
    "BeammapDetectorTodOutputConfig",
    "BeammapFittingConfig",
    "BeammapFlaggingConfig",
    "BeammapIterationConfig",
    "BeammapPhaseStrategyConfig",
    "BeammapPriorsConfig",
    "BeammapReferenceConfig",
    "BeammapRfiMaskConfig",
    "BeammapScanBandMaskConfig",
    "BeammapSplitFitsByFlagConfig",
}
EXPECTED_BOUNDARY_CALLS = {
    "read_beammap_request_config": 1,
    "reset_from_request": 1,
    "log_beammap_effective_resolution": 1,
    "install_beammap_effective_compatibility_config": 1,
    "sync_beammap_map_fitter": 1,
}
RETIRED_READER_MUTATION_HELPERS = (
    "normalize_beammap_phase_strategy",
    "set_beammap_priors_iteration_defaults",
    "disable_missing_beammap_priors",
    "normalized_beammap_split_flag_values",
)


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


def load_manifest(path: Path) -> dict[str, object]:
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
    exact = bool(
        manifest.get("schema_version") == EXPECTED_SCHEMA
        and manifest.get("path_count") == len(paths) == EXPECTED_PATH_COUNT
        and manifest.get("path_sha256")
        == digest
        == EXPECTED_PATH_SHA256
        and manifest.get("config_prefixes") == ["beammap"]
        and manifest.get("known_typed_gaps") == []
    )
    return {
        "schema_version": manifest.get("schema_version"),
        "path_count": len(paths),
        "path_sha256": digest,
        "known_typed_gaps": manifest.get("known_typed_gaps"),
        "paths": paths,
        "exact": exact,
    }


def default_surface_state(repo_root: Path, paths: list[str]) -> dict[str, object]:
    config = yaml.safe_load(
        (repo_root / DEFAULT_CONFIG_SOURCE).read_text(encoding="utf-8")
    )
    actual = sorted(flatten_paths(config["beammap"], ["beammap"]))
    return {
        "source": DEFAULT_CONFIG_SOURCE,
        "path_count": len(actual),
        "path_sha256": path_digest(actual),
        "missing_paths": sorted(set(paths) - set(actual)),
        "extra_paths": sorted(set(actual) - set(paths)),
        "exact": actual == paths,
    }


def beammap_literal_files(repo_root: Path) -> set[str]:
    pattern = re.compile(r'(?:std::tuple\s*)?\{\s*"beammap"\s*,')
    found: set[str] = set()
    for source_root in (repo_root / "include", repo_root / "src"):
        for suffix in ("*.h", "*.cpp"):
            for path in source_root.rglob(suffix):
                relative = path.relative_to(repo_root).as_posix()
                if relative in LITERAL_SCAN_EXCLUDED_FILES:
                    continue
                if pattern.search(path.read_text(encoding="utf-8")):
                    found.add(relative)
    return found


def literal_boundary_state(repo_root: Path) -> dict[str, object]:
    actual = beammap_literal_files(repo_root)
    return {
        "files": sorted(actual),
        "expected_files": sorted(EXPECTED_LITERAL_FILES),
        "unexpected_files": sorted(actual - EXPECTED_LITERAL_FILES),
        "missing_files": sorted(EXPECTED_LITERAL_FILES - actual),
        "exact": actual == EXPECTED_LITERAL_FILES,
    }


def expand_path_roots(
    manifest_paths: list[str], roots: set[str]
) -> dict[str, object]:
    covered = sorted(
        path
        for path in manifest_paths
        if any(path == root or path.startswith(root + ".") for root in roots)
    )
    extra_roots = sorted(
        root
        for root in roots
        if not any(
            path == root or path.startswith(root + ".")
            for path in manifest_paths
        )
    )
    missing = sorted(set(manifest_paths) - set(covered))
    return {
        "root_count": len(roots),
        "roots": sorted(roots),
        "covered_path_count": len(covered),
        "covered_paths": covered,
        "missing_paths": missing,
        "extra_roots": extra_roots,
        "exact": not missing and not extra_roots,
    }


def reader_coverage_state(
    repo_root: Path, manifest_paths: list[str]
) -> dict[str, object]:
    pattern = re.compile(
        r'(?:std::tuple\s*)?\{\s*"beammap"'
        r'(?P<body>(?:\s*,\s*"[^"]+")+?)\s*\}'
    )
    roots: set[str] = set()
    for source in READER_SOURCES:
        text = (repo_root / source).read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            roots.add(".".join(re.findall(r'"([^"]+)"', match.group(0))))
    state = expand_path_roots(manifest_paths, roots)
    state["sources"] = list(READER_SOURCES)
    state["retired_mutation_helper_counts"] = {
        helper: sum(
            (repo_root / source).read_text(encoding="utf-8").count(helper)
            for source in READER_SOURCES
        )
        for helper in RETIRED_READER_MUTATION_HELPERS
    }
    state["mutation_helpers_retired"] = not any(
        state["retired_mutation_helper_counts"].values()
    )
    state["exact"] = state["exact"] and state["mutation_helpers_retired"]
    return state


def serializer_coverage_state(
    repo_root: Path, manifest_paths: list[str]
) -> dict[str, object]:
    source = (repo_root / SERIALIZER_SOURCE).read_text(encoding="utf-8")
    pattern = re.compile(
        r'\bnode(?P<body>(?:\s*\[\s*"[^"]+"\s*\])+)'
    )
    roots = {
        "beammap."
        + ".".join(re.findall(r'"([^"]+)"', match.group("body")))
        for match in pattern.finditer(source)
    }
    state = expand_path_roots(manifest_paths, roots)
    state["source"] = SERIALIZER_SOURCE
    return state


def production_references(
    repo_root: Path, excluded_source: str, needles: tuple[str, ...]
) -> list[str]:
    references: list[str] = []
    for source_root in (repo_root / "include", repo_root / "src"):
        for suffix in ("*.h", "*.cpp"):
            for path in source_root.rglob(suffix):
                relative = path.relative_to(repo_root).as_posix()
                if relative == excluded_source:
                    continue
                text = path.read_text(encoding="utf-8")
                if any(needle in text for needle in needles):
                    references.append(relative)
    return sorted(set(references))


def execution_plan_state(repo_root: Path) -> dict[str, object]:
    path = repo_root / PLAN_SOURCE
    source_exists = path.exists()
    source = path.read_text(encoding="utf-8") if source_exists else ""
    plan_references = production_references(
        repo_root,
        PLAN_SOURCE,
        ("beammap_execution_plan.h", "BeammapExecutionPlan"),
    )
    serializer_references = production_references(
        repo_root,
        SERIALIZER_SOURCE,
        ("beammap_config_serialization.h", "beammap_config_node("),
    )
    contract_present = all(
        token in source
        for token in (
            "class BeammapExecutionPlan",
            "BeammapConfig requested_",
            "BeammapConfig effective_",
            "BeammapEffectiveResolutionRecord resolution_",
            "reset_from_request",
        )
    )
    expected_plan_references = [
        "include/citlali/core/pipeline/beammap_config_loading.h",
        "include/citlali/core/pipeline/beammap_provenance.h",
        LIFECYCLE_SOURCE,
        "include/citlali/core/pipeline/beammap_provenance_serialization.h",
        "include/citlali/core/pipeline/reduction_config_state.h",
    ]
    expected_serializer_references = [PROVENANCE_SOURCE]
    boundary = (repo_root / BOUNDARY_SOURCE).read_text(encoding="utf-8")
    wired = all(
        token in boundary
        for token in (
            "beammap_plan.reset_from_request(",
            "install_beammap_effective_compatibility_config(",
            "beammap_plan.effective().fitting",
        )
    )
    return {
        "source": PLAN_SOURCE,
        "source_exists": source_exists,
        "contract_present": contract_present,
        "production_references": plan_references,
        "expected_production_references": expected_plan_references,
        "serializer_production_references": serializer_references,
        "wired_at_boundary": wired,
        "status": "wired-realized-provenance"
        if source_exists
        and contract_present
        and plan_references == expected_plan_references
        and wired
        and serializer_references == expected_serializer_references
        else "unexpected",
        "exact": source_exists
        and contract_present
        and plan_references == expected_plan_references
        and wired
        and serializer_references == expected_serializer_references,
    }


def typed_model_state(source: str) -> dict[str, object]:
    present = sorted(
        name for name in EXPECTED_STRUCTS if f"struct {name}" in source
    )
    missing = sorted(EXPECTED_STRUCTS - set(present))
    return {
        "source": MODEL_SOURCE,
        "present_structs": present,
        "missing_structs": missing,
        "exact": not missing,
    }


def call_count(source: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source))


def authority_boundary_state(boundary: str, adapter: str) -> dict[str, object]:
    calls = {
        name: call_count(boundary, name) for name in EXPECTED_BOUNDARY_CALLS
    }
    ordered_calls = [
        "read_beammap_request_config",
        "reset_from_request",
        "log_beammap_effective_resolution",
        "install_beammap_effective_compatibility_config",
        "sync_beammap_map_fitter",
    ]
    positions = [boundary.find(f"{name}(") for name in ordered_calls]
    order_exact = (
        all(position >= 0 for position in positions)
        and positions == sorted(positions)
    )
    adapter_exact = (
        call_count(adapter, "sync_beammap_map_fitter") == 1
        and "map_fitter.beammap_fit_radius_fwhm =" in adapter
        and "fitting.fit_radius_fwhm" in adapter
    )
    return {
        "source": BOUNDARY_SOURCE,
        "call_counts": calls,
        "expected_call_counts": EXPECTED_BOUNDARY_CALLS,
        "order_exact": order_exact,
        "adapter_source": ADAPTER_SOURCE,
        "adapter_exact": adapter_exact,
        "exact": calls == EXPECTED_BOUNDARY_CALLS
        and order_exact
        and adapter_exact,
    }


def inventory_state(repo_root: Path) -> dict[str, object]:
    inventory = json.loads((repo_root / AUTHORITY_SOURCE).read_text())
    domain = next(item for item in inventory["domains"] if item["id"] == "beammap")
    exact = bool(
        domain["config_prefixes"] == ["beammap"]
        and domain["execution_authority"] == "typed"
        and domain["adapter_direction"] == "typed-to-legacy"
        and domain["migration_status"] == "typed-authoritative-with-adapter"
        and domain["provenance_status"] == "complete"
    )
    return {"source": AUTHORITY_SOURCE, "domain": domain, "exact": exact}


def provenance_state(repo_root: Path) -> dict[str, object]:
    source_exists = (repo_root / PROVENANCE_SOURCE).exists()
    source = (
        (repo_root / PROVENANCE_SOURCE).read_text(encoding="utf-8")
        if source_exists
        else ""
    )
    cli = (repo_root / CLI_SOURCE).read_text(encoding="utf-8")
    write_count = call_count(cli, "write_beammap_provenance_file")
    completion_count = call_count(cli, "record_beammap_run_completed")
    contract_exact = all(
        token in source
        for token in (
            '"citlali-beammap-provenance-v2"',
            '"beammap_provenance.yaml"',
            "plan.realized().reduction_completed",
            "plan.realized().outputs_completed",
            "write_yaml_file_atomic(",
        )
    )
    completion_position = cli.find("record_beammap_run_completed(")
    write_position = cli.find("write_beammap_provenance_file(")
    ordered = (
        completion_position >= 0
        and write_position > completion_position
    )
    exact = bool(
        source_exists
        and write_count == 1
        and completion_count == 1
        and contract_exact
        and ordered
    )
    return {
        "source": PROVENANCE_SOURCE,
        "source_exists": source_exists,
        "cli_write_count": write_count,
        "cli_completion_count": completion_count,
        "contract_exact": contract_exact,
        "completion_before_write": ordered,
        "status": "required-atomic-v2" if exact else "unexpected",
        "exact": exact,
    }


def lifecycle_state(repo_root: Path) -> dict[str, object]:
    source_exists = (repo_root / LIFECYCLE_SOURCE).exists()
    call_counts: dict[str, dict[str, int]] = {}
    for source, expected in EXPECTED_LIFECYCLE_CALLS.items():
        text = (repo_root / source).read_text(encoding="utf-8")
        call_counts[source] = {
            name: call_count(text, name) for name in expected
        }
    exact = bool(
        source_exists
        and call_counts == EXPECTED_LIFECYCLE_CALLS
    )
    return {
        "source": LIFECYCLE_SOURCE,
        "source_exists": source_exists,
        "call_counts": call_counts,
        "expected_call_counts": EXPECTED_LIFECYCLE_CALLS,
        "exact": exact,
    }


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(load_manifest(repo_root / MANIFEST_SOURCE))
    defaults = default_surface_state(repo_root, manifest["paths"])
    literals = literal_boundary_state(repo_root)
    readers = reader_coverage_state(repo_root, manifest["paths"])
    serializer = serializer_coverage_state(repo_root, manifest["paths"])
    plan = execution_plan_state(repo_root)
    model = typed_model_state((repo_root / MODEL_SOURCE).read_text())
    boundary = authority_boundary_state(
        (repo_root / BOUNDARY_SOURCE).read_text(),
        (repo_root / ADAPTER_SOURCE).read_text(),
    )
    inventory = inventory_state(repo_root)
    provenance = provenance_state(repo_root)
    lifecycle = lifecycle_state(repo_root)
    drift = not all(
        (
            manifest["exact"],
            defaults["exact"],
            literals["exact"],
            readers["exact"],
            serializer["exact"],
            plan["exact"],
            model["exact"],
            boundary["exact"],
            inventory["exact"],
            provenance["exact"],
            lifecycle["exact"],
        )
    )
    return {
        "manifest": manifest,
        "default_surface": defaults,
        "config_literal_boundary": literals,
        "reader_coverage": readers,
        "serializer_coverage": serializer,
        "execution_plan": plan,
        "typed_model": model,
        "authority_boundary": boundary,
        "inventory": inventory,
        "provenance": provenance,
        "lifecycle": lifecycle,
        "drift": drift,
    }


def markdown_report(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Beammap Config Boundary Audit",
            "",
            f"- Drift: `{result['drift']}`",
            f"- Frozen paths: `{result['manifest']['path_count']}`",
            f"- Default surface exact: `{result['default_surface']['exact']}`",
            f"- Typed model exact: `{result['typed_model']['exact']}`",
            f"- Reader coverage: `"
            f"{result['reader_coverage']['covered_path_count']}/"
            f"{result['manifest']['path_count']}`",
            f"- Serializer coverage: `"
            f"{result['serializer_coverage']['covered_path_count']}/"
            f"{result['manifest']['path_count']}`",
            f"- Execution plan status: `{result['execution_plan']['status']}`",
            f"- Config literal boundary exact: `"
            f"{result['config_literal_boundary']['exact']}`",
            f"- Authority boundary exact: `"
            f"{result['authority_boundary']['exact']}`",
            f"- Provenance status: `{result['provenance']['status']}`",
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
        "beammap config boundary: "
        f"paths={result['manifest']['path_count']} "
        f"defaults={result['default_surface']['exact']} "
        f"typed_model={result['typed_model']['exact']} "
        f"readers={result['reader_coverage']['covered_path_count']}/"
        f"{result['manifest']['path_count']} "
        f"serializer={result['serializer_coverage']['covered_path_count']}/"
        f"{result['manifest']['path_count']} "
        f"plan={result['execution_plan']['status']} "
        f"literal_boundary={result['config_literal_boundary']['exact']} "
        f"authority={result['authority_boundary']['exact']} "
        f"provenance={result['provenance']['status']} "
        f"drift={result['drift']}"
    )
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
