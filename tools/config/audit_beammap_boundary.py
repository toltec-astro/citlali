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
PROVENANCE_SOURCE = "include/citlali/core/pipeline/beammap_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"

EXPECTED_SCHEMA = "citlali-frozen-beammap-config-paths-v1"
EXPECTED_PATH_COUNT = 74
EXPECTED_PATH_SHA256 = (
    "23b89ad61e2c318eb2f2a37369968e3079101ff3f52e814ac4c00e786e542f97"
)
EXPECTED_LITERAL_FILES = {
    "include/citlali/core/config/beammap_config_validation.h",
    "include/citlali/core/config/reduction_config_validation.h",
    "include/citlali/core/pipeline/beammap_config_core_loading.h",
    "include/citlali/core/pipeline/beammap_config_fitting_flagging.h",
    "include/citlali/core/pipeline/beammap_config_priors_loading.h",
    "include/citlali/core/pipeline/beammap_config_split_outputs.h",
    "include/citlali/core/pipeline/beammap_config_tod_mirror.h",
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
    "read_beammap_core_config": 1,
    "read_beammap_fitting_config": 1,
    "read_beammap_scan_band_mask_config": 1,
    "read_beammap_split_fits_config": 1,
    "sync_beammap_map_fitter": 1,
    "read_beammap_priors_config": 1,
    "read_beammap_flagging_config": 1,
    "read_beammap_sensitivity_config": 1,
    "read_beammap_detector_tod_output_config": 1,
    "apply_beammap_typed_config": 1,
}


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
                if pattern.search(path.read_text(encoding="utf-8")):
                    found.add(path.relative_to(repo_root).as_posix())
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
    reader_positions = [
        boundary.find(f"{name}(")
        for name in EXPECTED_BOUNDARY_CALLS
        if name not in {"sync_beammap_map_fitter", "apply_beammap_typed_config"}
    ]
    adapter_position = boundary.find("sync_beammap_map_fitter(")
    apply_position = boundary.find("apply_beammap_typed_config(")
    order_exact = (
        all(position >= 0 for position in reader_positions)
        and adapter_position >= 0
        and apply_position > adapter_position
        and apply_position > max(reader_positions)
    )
    adapter_exact = (
        call_count(adapter, "sync_beammap_map_fitter") == 1
        and "map_fitter.beammap_fit_radius_fwhm =" in adapter
        and "fitting_values.fitting.fit_radius_fwhm" in adapter
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
        and domain["provenance_status"] == "partial"
    )
    return {"source": AUTHORITY_SOURCE, "domain": domain, "exact": exact}


def provenance_state(repo_root: Path) -> dict[str, object]:
    source_exists = (repo_root / PROVENANCE_SOURCE).exists()
    cli = (repo_root / CLI_SOURCE).read_text(encoding="utf-8")
    write_count = call_count(cli, "write_beammap_provenance_file")
    return {
        "source": PROVENANCE_SOURCE,
        "source_exists": source_exists,
        "cli_write_count": write_count,
        "status": "missing" if not source_exists and write_count == 0 else "present",
        "expected_missing": not source_exists and write_count == 0,
    }


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(load_manifest(repo_root / MANIFEST_SOURCE))
    defaults = default_surface_state(repo_root, manifest["paths"])
    literals = literal_boundary_state(repo_root)
    model = typed_model_state((repo_root / MODEL_SOURCE).read_text())
    boundary = authority_boundary_state(
        (repo_root / BOUNDARY_SOURCE).read_text(),
        (repo_root / ADAPTER_SOURCE).read_text(),
    )
    inventory = inventory_state(repo_root)
    provenance = provenance_state(repo_root)
    drift = not all(
        (
            manifest["exact"],
            defaults["exact"],
            literals["exact"],
            model["exact"],
            boundary["exact"],
            inventory["exact"],
            provenance["expected_missing"],
        )
    )
    return {
        "manifest": manifest,
        "default_surface": defaults,
        "config_literal_boundary": literals,
        "typed_model": model,
        "authority_boundary": boundary,
        "inventory": inventory,
        "provenance": provenance,
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
        f"literal_boundary={result['config_literal_boundary']['exact']} "
        f"authority={result['authority_boundary']['exact']} "
        f"provenance={result['provenance']['status']} "
        f"drift={result['drift']}"
    )
    return 1 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
