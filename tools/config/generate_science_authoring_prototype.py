#!/usr/bin/env python3
"""Generate the human-facing Phase 4.1 V2 science configuration prototype."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tolteca_mode_kit import (
    extract_low_level,
    merge_files,
    numbered_yaml_files,
    policy_sha256,
)


SCHEMA_VERSION = "citlali-tolteca-mode-kit-manifest-v2"
KIT_VERSION = "phase4.1-v2-science-prototype"
FILES = {
    "internal_policy": "60_science_internal_policy.yaml",
    "runtime": "71_science_runtime.yaml",
    "observation": "72_science_observation.yaml",
    "defaults": "81_science_defaults.yaml",
    "products": "82_science_products.yaml",
    "advanced": "90_science_advanced_overrides.yaml",
    "expert": "99_science_expert_overrides.yaml",
}

RUNTIME_PATHS = (
    "runtime.n_threads",
    "runtime.output_dir",
    "runtime.use_subdir",
    "runtime.verbose",
    "kids.solver.fitreportdir",
)

DEFAULT_PATHS = (
    "mapmaking.cunit",
    "mapmaking.method",
    "mapmaking.pixel_axes",
    "mapmaking.pixel_size_arcsec",
    "timestream.raw_time_chunk.despike.enabled",
    "timestream.raw_time_chunk.filter.enabled",
    "timestream.raw_time_chunk.extinction_correction.enabled",
    "timestream.raw_time_chunk.flux_calibration.enabled",
    "timestream.processed_time_chunk.clean.enabled",
    "timestream.processed_time_chunk.clean.grouping",
    "timestream.processed_time_chunk.clean.standard_pca.enabled",
    "timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut",
    "timestream.processed_time_chunk.clean.null_model.enabled",
    "timestream.processed_time_chunk.clean.marchenko_pastur.enabled",
    "timestream.processed_time_chunk.clean.adaptive_selector.enabled",
    "timestream.processed_time_chunk.flagging.second_pass_local.enabled",
    "timestream.processed_time_chunk.weighting.type",
    "timestream.fruit_loops.enabled",
    "timestream.fruit_loops.max_iters",
    "timestream.fruit_loops.sig2noise_limit",
    "timestream.fruit_loops.array_flux_limit",
    "timestream.fruit_loops.save_all_iters",
    "timestream.learning.enabled",
)

PRODUCT_PATHS = (
    "coadd.enabled",
    "noise_maps.enabled",
    "noise_maps.n_noise_maps",
    "noise_maps.products.enabled",
    "post_processing.map_filtering.enabled",
    "post_processing.map_filtering.type",
    "post_processing.map_filtering.normalize_errors",
    "post_processing.source_finding.enabled",
    "post_processing.source_fitting.model",
    "post_processing.source_fitting.bounding_box_arcsec",
    "post_processing.source_fitting.fitting_radius_arcsec",
    "wiener_filter.template_type",
    "wiener_filter.template_fwhm_arcsec",
    "wiener_filter.lowpass_only",
    "timestream.raw_time_chunk.output.enabled",
    "timestream.raw_time_chunk.output.indices",
    "timestream.processed_time_chunk.output.enabled",
    "timestream.processed_time_chunk.output.indices",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def write_yaml(
    path: Path,
    value: Any,
    comments: tuple[str, ...],
    section_headers: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(f"# {comment}\n" for comment in comments)
    body = yaml.safe_dump(value, sort_keys=False, width=100)
    for marker, section_comments in section_headers:
        if body.count(marker) != 1:
            raise ValueError(f"expected one generated YAML marker {marker!r}")
        indent = marker[: len(marker) - len(marker.lstrip())]
        section = "\n" + "".join(
            f"{indent}# {comment}\n" for comment in section_comments
        )
        body = body.replace(marker, section + marker, 1)
    path.write_text(
        header + body,
        encoding="utf-8",
    )


def get_path(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"accepted science policy has no path {dotted_path!r}")
        current = current[part]
    return copy.deepcopy(current)


def set_path(value: dict[str, Any], dotted_path: str, selected: Any) -> None:
    parts = dotted_path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"path collision while selecting {dotted_path!r}")
        current = child
    current[parts[-1]] = selected


def select_paths(policy: Mapping[str, Any], paths: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for path in paths:
        set_path(selected, path, get_path(policy, path))
    return selected


def low_level_patch(policy: Mapping[str, Any], paths: tuple[str, ...]) -> dict[str, Any]:
    return {
        "reduce": {
            "steps": {
                0: {
                    "config": {
                        "low_level": select_paths(policy, paths),
                    }
                }
            }
        }
    }


def empty_override_patch() -> dict[str, Any]:
    return {
        "reduce": {
            "steps": {
                0: {
                    "config": {
                        "low_level": {},
                    }
                }
            }
        }
    }


def runtime_patch(merged: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    reduce_section = merged["reduce"]
    inputs = reduce_section["inputs"]
    steps = reduce_section["steps"]
    return {
        "reduce": {
            "jobkey": reduce_section["jobkey"],
            "inputs": {0: {"path": inputs[0]["path"]}},
            "steps": {
                0: {
                    "path": steps[0]["path"],
                    "config": {
                        "low_level": select_paths(policy, RUNTIME_PATHS),
                    },
                }
            },
        }
    }


def generate(source_root: Path, output_root: Path) -> None:
    source_mode_dir = source_root / "science"
    source_manifest = load_yaml(source_root / "manifest.yaml")
    source_entry = source_manifest["modes"]["science"]
    merged, _, _ = merge_files(numbered_yaml_files(source_mode_dir))
    policy = extract_low_level(merged)

    science_dir = output_root / "science"
    write_yaml(
        science_dir / FILES["internal_policy"],
        merged,
        (
            "Complete validated Citlali science policy; generated and maintainer-owned.",
            "Normal reducers should use 81_science_defaults.yaml and 82_science_products.yaml.",
        ),
    )
    write_yaml(
        science_dir / FILES["runtime"],
        runtime_patch(merged, policy),
        (
            "Workspace paths and ordinary runtime resources.",
            "TolPROJ supplies this file; site operators may adjust it.",
            "Set n_threads to the CPU allocation available to this reduction.",
        ),
        (
            ("  inputs:\n", ("Input data location",)),
            ("  steps:\n", ("Citlali executable and runtime settings",)),
            ("          runtime:\n", ("CPU use, output location, and log verbosity",)),
            ("          kids:\n", ("KIDs fit-report location",)),
        ),
    )
    write_yaml(
        science_dir / FILES["observation"],
        load_yaml(source_mode_dir / "72_observation.yaml"),
        (
            "Observation selection, APTs, fluxes, and pointing support.",
            "TolPROJ generates this file from project metadata.",
        ),
    )
    write_yaml(
        science_dir / FILES["defaults"],
        low_level_patch(policy, DEFAULT_PATHS),
        (
            "Primary user-facing science analysis defaults.",
            "Edit routine mapmaking, calibration, cleaning, weighting, and iteration choices here.",
            "Enable one primary cleaner; n_eig_to_cut has one value per active cleaning group.",
            "Learning and second-pass flagging are explicit because they can materially affect analysis.",
        ),
        (
            ("          mapmaking:\n", ("Map geometry and mapmaking method",)),
            (
                "          timestream:\n",
                ("Timestream calibration, cleaning, weighting, and iterations",),
            ),
            (
                "            raw_time_chunk:\n",
                ("High-level raw-data corrections; thresholds remain expert policy",),
            ),
            (
                "            processed_time_chunk:\n",
                ("Processed-data cleaner selection, flagging, and map weighting",),
            ),
            ("                standard_pca:\n", ("Primary cleaner and per-array PCA depth",)),
            (
                "                null_model:\n",
                ("Alternative cleaners; normally leave all disabled",),
            ),
            (
                "            fruit_loops:\n",
                ("Iterative source-model subtraction and retained iteration products",),
            ),
            (
                "              sig2noise_limit:",
                (
                    "Source-model mask cuts; a pixel is retained when either enabled cut passes",
                    "Set a cut to 0 to disable it; the S/N cut also requires noise maps",
                ),
            ),
            (
                "              array_flux_limit:\n",
                ("Flux thresholds in map units, ordered [a1100, a1400, a2000]",),
            ),
            (
                "              save_all_iters:",
                ("Keep every fruit-loop iteration instead of only the final products",),
            ),
            ("            learning:\n", ("Cross-iteration learned masks and detector state",)),
        ),
    )
    write_yaml(
        science_dir / FILES["products"],
        low_level_patch(policy, PRODUCT_PATHS),
        (
            "Primary user-facing science product choices.",
            "Edit coadds, noise products, filtering, fitting, and retained TOD products here.",
            "noise_maps.enabled creates realizations; noise_maps.products.enabled writes summaries.",
            "TOD indices are used only when the corresponding output.enabled value is true.",
            "Wiener settings apply when map_filtering.type is wiener_filter.",
        ),
        (
            ("          coadd:\n", ("Coadded maps and noise-map products",)),
            ("          post_processing:\n", ("Filtered maps and source products",)),
            ("          wiener_filter:\n", ("User-facing Wiener template choices",)),
            (
                "          timestream:\n",
                ("Optional retained raw and processed TOD products",),
            ),
            ("            raw_time_chunk:\n", ("Raw calibrated TOD output",)),
            ("            processed_time_chunk:\n", ("Cleaned and weighted TOD output",)),
        ),
    )
    write_yaml(
        science_dir / FILES["advanced"],
        empty_override_patch(),
        (
            "Optional advanced overrides for supported user-facing controls not shown in 81/82.",
            "Keep this empty unless a documented analysis need requires an additional control.",
        ),
        (
            (
                "        low_level: {}\n",
                ("Replace {} with additional supported user-facing low-level settings",),
            ),
        ),
    )
    write_yaml(
        science_dir / FILES["expert"],
        empty_override_patch(),
        (
            "Optional expert-only low-level overrides.",
            "Changes here require an explicit rationale and successor validation evidence.",
        ),
        (
            (
                "        low_level: {}\n",
                ("Replace {} with deliberate expert low-level settings",),
            ),
        ),
    )

    entry = copy.deepcopy(source_entry)
    entry.update(
        {
            "policy_sha256": policy_sha256(policy),
            "required_files": list(FILES.values()),
            "internal_policy_file": FILES["internal_policy"],
            "user_facing_files": [
                FILES["runtime"],
                FILES["defaults"],
                FILES["products"],
                FILES["advanced"],
            ],
            "expert_override_file": FILES["expert"],
        }
    )
    write_yaml(
        output_root / "manifest.yaml",
        {
            "schema_version": SCHEMA_VERSION,
            "kit_version": KIT_VERSION,
            "prototype_scope": "science-only",
            "modes": {"science": entry},
        },
        ("Science-only human-facing authoring prototype; not yet deployed by TolPROJ.",),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(repo_root / "config/tolteca"))
    parser.add_argument("--output-root", default=str(repo_root / "config/tolteca/v2"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    generate(
        Path(args.source_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
