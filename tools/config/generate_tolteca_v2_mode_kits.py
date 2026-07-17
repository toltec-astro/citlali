#!/usr/bin/env python3
"""Generate the human-facing Phase 4.1 V2 configuration for all modes."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from dataclasses import dataclass
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
KIT_VERSION = "phase4.1-v2"

RUNTIME_PATHS = (
    "runtime.n_threads",
    "runtime.output_dir",
    "runtime.use_subdir",
    "runtime.verbose",
)

SCIENCE_DEFAULT_PATHS = (
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

SCIENCE_PRODUCT_PATHS = (
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

POINT_SOURCE_PATHS = (
    "source.map_regime",
    "pointing.source_strategy",
)

MAP_PATHS = (
    "mapmaking.cunit",
    "mapmaking.method",
    "mapmaking.pixel_axes",
    "mapmaking.pixel_size_arcsec",
)

CHUNKING_PATHS = (
    "timestream.chunking.chunk_mode",
    "timestream.chunking.force_chunking",
    "timestream.chunking.value",
)

COMMON_TIMESTREAM_PATHS = (
    "timestream.raw_time_chunk.despike.enabled",
    "timestream.raw_time_chunk.filter.enabled",
    "timestream.raw_time_chunk.IIR_filter.enabled",
    "timestream.raw_time_chunk.downsample.enabled",
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
)

POINT_SOURCE_PROTECTION_PATHS = (
    "timestream.raw_time_chunk.despike.source_protection.radius_arcsec",
    "timestream.processed_time_chunk.flagging.second_pass_local.source_protection.radius_arcsec",
    "timestream.processed_time_chunk.weighting.source_mask_radius_arcsec",
)

POINT_FRUIT_LOOP_PATHS = (
    "timestream.fruit_loops.enabled",
    "timestream.fruit_loops.max_iters",
    "timestream.fruit_loops.sig2noise_limit",
    "timestream.fruit_loops.array_flux_limit",
    "timestream.fruit_loops.center_keep_radius_arcsec",
    "timestream.fruit_loops.adaptive_support_radius_arcsec",
    "timestream.fruit_loops.adaptive_support_radius_fwhm",
    "timestream.fruit_loops.save_all_iters",
    "timestream.learning.enabled",
)

POINT_DEFAULT_PATHS = (
    *POINT_SOURCE_PATHS,
    *MAP_PATHS,
    *CHUNKING_PATHS,
    *COMMON_TIMESTREAM_PATHS,
    *POINT_SOURCE_PROTECTION_PATHS,
    *POINT_FRUIT_LOOP_PATHS,
)

POINT_PRODUCT_PATHS = (
    "noise_maps.enabled",
    "noise_maps.n_noise_maps",
    "noise_maps.randomize_dets",
    "noise_maps.write_realizations",
    "noise_maps.products.enabled",
    "noise_maps.products.apply_empirical_weights",
    "post_processing.map_filtering.enabled",
    "post_processing.map_filtering.type",
    "post_processing.map_filtering.normalize_errors",
    "post_processing.map_histogram_n_bins",
    "post_processing.source_finding.enabled",
    "post_processing.source_finding.mode",
    "post_processing.source_finding.source_sigma",
    "post_processing.source_finding.source_window_arcsec",
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

BEAMMAP_PATHS = (
    "source.map_regime",
    "beammap.iter_max",
    "beammap.iter_tolerance",
    "beammap.convergence_radius_arcsec",
    "beammap.derotate",
    "beammap.subtract_reference_det",
    "beammap.reference_det",
    "beammap.detector_weighting.mode",
    "beammap.priors.enabled",
    "beammap.priors.filepath",
    "beammap.rfi_mask.enabled",
    "beammap.scan_band_mask.enabled",
)

BEAMMAP_FRUIT_LOOP_PATHS = (
    "timestream.fruit_loops.enabled",
    "timestream.fruit_loops.max_iters",
    "timestream.fruit_loops.sig2noise_limit",
    "timestream.fruit_loops.array_flux_limit",
    "timestream.fruit_loops.save_all_iters",
)

BEAMMAP_DEFAULT_PATHS = (
    *BEAMMAP_PATHS,
    *MAP_PATHS,
    *CHUNKING_PATHS,
    *COMMON_TIMESTREAM_PATHS,
    *BEAMMAP_FRUIT_LOOP_PATHS,
)

BEAMMAP_PRODUCT_PATHS = (
    "beammap.detector_tod_output.enabled",
    "beammap.split_fits_by_flag.enabled",
    "timestream.raw_time_chunk.line_audit.enabled",
    "timestream.raw_time_chunk.output.enabled",
    "timestream.processed_time_chunk.output.enabled",
)


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    slug: str
    label: str
    default_paths: tuple[str, ...]
    product_paths: tuple[str, ...]

    @property
    def files(self) -> dict[str, str]:
        return {
            "internal_policy": f"60_{self.slug}_internal_policy.yaml",
            "runtime": f"71_{self.slug}_runtime.yaml",
            "observation": f"72_{self.slug}_observation.yaml",
            "defaults": f"81_{self.slug}_defaults.yaml",
            "products": f"82_{self.slug}_products.yaml",
            "advanced": f"90_{self.slug}_advanced_overrides.yaml",
            "expert": f"99_{self.slug}_expert_overrides.yaml",
        }


MODE_SPECS = {
    "point": ModeSpec(
        mode="point",
        slug="pointing",
        label="pointing",
        default_paths=POINT_DEFAULT_PATHS,
        product_paths=POINT_PRODUCT_PATHS,
    ),
    "oof": ModeSpec(
        mode="oof",
        slug="oof",
        label="OOF",
        default_paths=POINT_DEFAULT_PATHS,
        product_paths=POINT_PRODUCT_PATHS,
    ),
    "beammap": ModeSpec(
        mode="beammap",
        slug="beammap",
        label="Beammap",
        default_paths=BEAMMAP_DEFAULT_PATHS,
        product_paths=BEAMMAP_PRODUCT_PATHS,
    ),
    "science": ModeSpec(
        mode="science",
        slug="science",
        label="science",
        default_paths=SCIENCE_DEFAULT_PATHS,
        product_paths=SCIENCE_PRODUCT_PATHS,
    ),
}


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
            raise KeyError(f"accepted mode policy has no path {dotted_path!r}")
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
    steps = reduce_section["steps"]
    return {
        "reduce": {
            "jobkey": reduce_section["jobkey"],
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


def observation_patch(
    source_mode_dir: Path,
    merged: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    observation = load_yaml(source_mode_dir / "72_observation.yaml")
    observation["reduce"]["inputs"][0]["path"] = merged["reduce"]["inputs"][0][
        "path"
    ]
    step_config = observation["reduce"]["steps"][0]["config"]
    low_level = step_config.setdefault("low_level", {})
    set_path(
        low_level,
        "kids.solver.fitreportdir",
        get_path(policy, "kids.solver.fitreportdir"),
    )
    return observation


def default_comments(spec: ModeSpec) -> tuple[str, ...]:
    comments = [
        f"Primary user-facing {spec.label} analysis defaults.",
        "Edit routine geometry, calibration, cleaning, weighting, and iteration choices here.",
        "Enable one primary cleaner; n_eig_to_cut has one value per active cleaning group.",
    ]
    if spec.mode != "beammap":
        comments.append(
            "Learning and second-pass flagging are explicit when they affect this mode."
        )
    return tuple(comments)


def default_sections(spec: ModeSpec) -> tuple[tuple[str, tuple[str, ...]], ...]:
    common = (
        ("          mapmaking:\n", ("Map geometry and mapmaking method",)),
        (
            "          timestream:\n",
            ("Timestream chunking, calibration, cleaning, weighting, and iterations",),
        ),
        ("            chunking:\n", ("Scan chunk duration and subdivision policy",)),
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
    )
    if spec.mode == "science":
        return tuple(section for section in common if section[0] != "            chunking:\n") + (
            ("            learning:\n", ("Cross-iteration learned masks and detector state",)),
        )
    if spec.mode in {"point", "oof"}:
        return (
            ("          source:\n", ("Source context used by pointing diagnostics",)),
            ("          pointing:\n", ("Pointing and OOF source strategy",)),
            *common,
            (
                "              center_keep_radius_arcsec:",
                ("Central source region retained during pointing/OOF feedback",),
            ),
            ("            learning:\n", ("Cross-iteration learned masks and detector state",)),
        )
    return (
        ("          source:\n", ("Source-dominant Beammap context",)),
        (
            "          beammap:\n",
            ("Beammap iterations, convergence, reference detector, masks, and priors",),
        ),
        ("            detector_weighting:\n", ("Detector-map weighting after iteration zero",)),
        ("            priors:\n", ("Optional detector-position prior policy and file",)),
        *common,
    )


def product_comments(spec: ModeSpec) -> tuple[str, ...]:
    if spec.mode == "beammap":
        return (
            "Primary user-facing Beammap product choices.",
            "Edit detector TOD, split FITS, line-audit, and retained TOD products here.",
            "Detailed detector selection and split-flag policy remain expert-only.",
        )
    return (
        f"Primary user-facing {spec.label} product choices.",
        "Edit noise products, filtering, fitting, source finding, and retained TOD products here.",
        "Source finding remains experimental and disabled in the accepted policy.",
        "TOD indices are used only when the corresponding output.enabled value is true.",
        "Wiener settings apply when map_filtering.type is wiener_filter.",
    )


def product_sections(spec: ModeSpec) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if spec.mode == "beammap":
        return (
            ("          beammap:\n", ("Detector-resolved Beammap product families",)),
            ("            detector_tod_output:\n", ("Per-detector TOD sidecar",)),
            ("            split_fits_by_flag:\n", ("Per-flag split FITS maps",)),
            ("          timestream:\n", ("Line diagnostics and optional retained TOD",)),
            ("            raw_time_chunk:\n", ("Raw calibrated TOD and line-audit products",)),
            ("            processed_time_chunk:\n", ("Cleaned and weighted TOD output",)),
        )
    return (
        ("          noise_maps:\n", ("Noise realizations and empirical summaries",)),
        ("          post_processing:\n", ("Filtered maps, source catalogs, and source fits",)),
        (
            "            source_finding:\n",
            ("Experimental source catalog; leave disabled for validated production",),
        ),
        ("          wiener_filter:\n", ("User-facing Wiener template choices",)),
        ("          timestream:\n", ("Optional retained raw and processed TOD products",)),
        ("            raw_time_chunk:\n", ("Raw calibrated TOD output",)),
        ("            processed_time_chunk:\n", ("Cleaned and weighted TOD output",)),
    )


def generate_mode(
    spec: ModeSpec,
    source_root: Path,
    output_root: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_mode_dir = source_root / spec.mode
    source_entry = source_manifest["modes"][spec.mode]
    merged, _, _ = merge_files(numbered_yaml_files(source_mode_dir))
    policy = extract_low_level(merged)
    mode_dir = output_root / spec.mode
    files = spec.files

    write_yaml(
        mode_dir / files["internal_policy"],
        merged,
        (
            f"Complete validated Citlali {spec.label} policy; generated and maintainer-owned.",
            f"Normal reducers should use {files['defaults']} and {files['products']}.",
        ),
    )
    write_yaml(
        mode_dir / files["runtime"],
        runtime_patch(merged, policy),
        (
            "Citlali executable and ordinary runtime resources.",
            "TolPROJ supplies this file; site operators may adjust it.",
            "Set n_threads to the CPU allocation available to this reduction.",
        ),
        (
            ("  steps:\n", ("Citlali executable and runtime settings",)),
            ("          runtime:\n", ("CPU use, output location, and log verbosity",)),
        ),
    )
    write_yaml(
        mode_dir / files["observation"],
        observation_patch(source_mode_dir, merged, policy),
        (
            "Data location, observation selection, APTs, fluxes, and pointing support.",
            "TolPROJ generates this file from project metadata and directory layout.",
        ),
        (
            ("  inputs:\n", ("Shared data path and observation selection",)),
            ("        low_level:\n", ("Generated paths used while loading observation data",)),
        ),
    )
    write_yaml(
        mode_dir / files["defaults"],
        low_level_patch(policy, spec.default_paths),
        default_comments(spec),
        default_sections(spec),
    )
    write_yaml(
        mode_dir / files["products"],
        low_level_patch(policy, spec.product_paths),
        product_comments(spec),
        product_sections(spec),
    )
    write_yaml(
        mode_dir / files["advanced"],
        empty_override_patch(),
        (
            "Optional advanced overrides for supported user-facing controls not shown in 81/82.",
            "Keep this empty unless a documented analysis need requires an additional control.",
        ),
        (("        low_level: {}\n", ("Replace {} with additional supported user-facing low-level settings",)),),
    )
    write_yaml(
        mode_dir / files["expert"],
        empty_override_patch(),
        (
            "Optional expert-only low-level overrides.",
            "Changes here require an explicit rationale and successor validation evidence.",
        ),
        (("        low_level: {}\n", ("Replace {} with deliberate expert low-level settings",)),),
    )

    entry = copy.deepcopy(source_entry)
    entry.update(
        {
            "policy_sha256": policy_sha256(policy),
            "required_files": list(files.values()),
            "internal_policy_file": files["internal_policy"],
            "user_facing_files": [
                files["runtime"],
                files["defaults"],
                files["products"],
                files["advanced"],
            ],
            "expert_override_file": files["expert"],
        }
    )
    return entry


def generate(source_root: Path, output_root: Path) -> None:
    source_manifest = load_yaml(source_root / "manifest.yaml")
    entries = {
        mode: generate_mode(spec, source_root, output_root, source_manifest)
        for mode, spec in MODE_SPECS.items()
    }
    write_yaml(
        output_root / "manifest.yaml",
        {
            "schema_version": SCHEMA_VERSION,
            "kit_version": KIT_VERSION,
            "scope": "all-supported-modes",
            "modes": entries,
        },
        ("Human-facing four-mode authoring kit; canonical source for TolPROJ vendoring.",),
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
