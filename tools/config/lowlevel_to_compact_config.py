#!/usr/bin/env python3
"""Generate a compact compatibility config from a Citlali low-level YAML file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import compare_lowlevel_yaml
import expand_compact_config


COMPACT_SCHEMA = "citlali-reduction-v2"
SUMMARY_SCHEMA = "citlali-lowlevel-to-compact-summary-v1"
PASSTHROUGH_PROFILE_BY_MODE = {
    "pointing": "pointing_compat_passthrough",
    "oof": "oof_compat_passthrough",
    "beammap": "beammap_compat_passthrough",
    "science": "science_compat_passthrough",
}


class ConvertError(RuntimeError):
    """Raised for user-correctable conversion errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def path_get(data: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    cursor = data
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return False, None
        cursor = cursor[key]
    return True, cursor


def set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConvertError(f"cannot set compact path {'.'.join(path)}")
        cursor = child
    cursor[path[-1]] = value


def copy_path(
    low_level: dict[str, Any],
    compact: dict[str, Any],
    summary: list[dict[str, str]],
    source_path: tuple[str, ...],
    compact_path: tuple[str, ...],
) -> None:
    found, value = path_get(low_level, source_path)
    if not found:
        return
    set_path(compact, compact_path, value)
    summary.append({"low_level": ".".join(source_path), "compact": ".".join(compact_path)})


def infer_mode(low_level: dict[str, Any], mode_override: str | None) -> str:
    if mode_override:
        return mode_override
    found, reduction_type = path_get(low_level, ("runtime", "reduction_type"))
    if not found:
        raise ConvertError("cannot infer mode because runtime.reduction_type is missing; pass --mode")
    if reduction_type in {"science", "beammap"}:
        return str(reduction_type)
    if reduction_type == "pointing":
        return "pointing"
    raise ConvertError(f"cannot infer compact mode from runtime.reduction_type={reduction_type!r}; pass --mode")


def tod_mode(low_level: dict[str, Any]) -> str | None:
    rtc_found, rtc_enabled = path_get(low_level, ("timestream", "raw_time_chunk", "output", "enabled"))
    ptc_found, ptc_enabled = path_get(low_level, ("timestream", "processed_time_chunk", "output", "enabled"))
    if not rtc_found and not ptc_found:
        return None
    rtc = bool(rtc_enabled) if rtc_found else False
    ptc = bool(ptc_enabled) if ptc_found else False
    if rtc and ptc:
        return "both"
    if rtc:
        return "rtc"
    if ptc:
        return "ptc"
    return "none"


def clean_mode(low_level: dict[str, Any]) -> str | None:
    clean_found, clean_enabled = path_get(low_level, ("timestream", "processed_time_chunk", "clean", "enabled"))
    if not clean_found:
        return None
    if not clean_enabled:
        return "off"

    flags = {
        "standard": ("timestream", "processed_time_chunk", "clean", "standard_pca", "enabled"),
        "null_model": ("timestream", "processed_time_chunk", "clean", "null_model", "enabled"),
        "marchenko_pastur": ("timestream", "processed_time_chunk", "clean", "marchenko_pastur", "enabled"),
        "adaptive_selector": ("timestream", "processed_time_chunk", "clean", "adaptive_selector", "enabled"),
    }
    values: dict[str, bool] = {}
    for mode, path in flags.items():
        found, value = path_get(low_level, path)
        if not found:
            return None
        values[mode] = bool(value)

    enabled_modes = [mode for mode, enabled in values.items() if enabled]
    if len(enabled_modes) == 1:
        return enabled_modes[0]
    return None


def copy_common_sections(
    low_level: dict[str, Any],
    compact: dict[str, Any],
    summary: list[dict[str, str]],
    include_output_dir: bool,
) -> None:
    if include_output_dir:
        copy_path(low_level, compact, summary, ("runtime", "output_dir"), ("output", "dir"))
    else:
        found, output_dir = path_get(low_level, ("runtime", "output_dir"))
        if found and output_dir == ".":
            copy_path(low_level, compact, summary, ("runtime", "output_dir"), ("output", "dir"))

    common_paths = {
        ("runtime", "use_subdir"): ("output", "subdir"),
        ("runtime", "verbose"): ("output", "verbose"),
        ("runtime", "n_threads"): ("runtime", "threads"),
        ("runtime", "parallel_policy"): ("runtime", "parallel"),
        ("mapmaking", "cunit"): ("map", "unit"),
        ("mapmaking", "method"): ("map", "method"),
        ("mapmaking", "grouping"): ("map", "grouping"),
        ("mapmaking", "pixel_axes"): ("map", "pixel_axes"),
        ("mapmaking", "pixel_size_arcsec"): ("map", "pixel_size_arcsec"),
        ("coadd", "enabled"): ("map", "coadd"),
        ("mapmaking", "enabled"): ("products", "maps"),
        ("noise_maps", "enabled"): ("products", "noise"),
        ("noise_maps", "n_noise_maps"): ("products", "noise_count"),
        ("noise_maps", "products", "enabled"): ("products", "noise_products"),
        ("noise_maps", "write_realizations"): ("products", "noise_realizations"),
        ("timestream", "enabled"): ("processing", "tod"),
        ("timestream", "processed_time_chunk", "clean", "grouping"): ("processing", "clean_grouping"),
        ("timestream", "processed_time_chunk", "weighting", "type"): ("processing", "weighting"),
        ("timestream", "fruit_loops", "enabled"): ("processing", "fruitloops"),
        ("timestream", "fruit_loops", "max_iters"): ("processing", "fruitloops_iters"),
    }
    for source_path, compact_path in common_paths.items():
        copy_path(low_level, compact, summary, source_path, compact_path)

    mode = tod_mode(low_level)
    if mode is not None:
        set_path(compact, ("products", "tod"), mode)
        summary.append({"low_level": "timestream.*_time_chunk.output.enabled", "compact": "products.tod"})

    cleaner = clean_mode(low_level)
    if cleaner is not None:
        set_path(compact, ("processing", "clean"), cleaner)
        summary.append({"low_level": "timestream.processed_time_chunk.clean.*", "compact": "processing.clean"})

    found, verbose = path_get(low_level, ("runtime", "verbose"))
    if found:
        set_path(compact, ("products", "diagnostics"), "verbose" if verbose else "normal")
        summary.append({"low_level": "runtime.verbose", "compact": "products.diagnostics"})


def copy_pointing_section(
    low_level: dict[str, Any],
    compact: dict[str, Any],
    summary: list[dict[str, str]],
    section_name: str,
) -> None:
    strategy_fields = {
        ("pointing", "source_strategy", "mode"): (section_name, "source_strategy"),
        ("pointing", "source_strategy", "fit_gaussian"): (section_name, "fit_gaussian"),
        ("pointing", "source_strategy", "fruitloops_center_mode"): (section_name, "fruitloops_center_mode"),
        ("pointing", "source_strategy", "header_max_radius_arcsec"): (section_name, "header_max_radius_arcsec"),
        ("pointing", "source_strategy", "header_require_coverage"): (section_name, "header_require_coverage"),
        ("post_processing", "source_fitting", "fitting_radius_arcsec"): (section_name, "fit_radius_arcsec"),
        ("post_processing", "source_fitting", "bounding_box_arcsec"): (section_name, "fit_box_arcsec"),
    }
    if section_name == "pointing":
        strategy_fields = {
            ("pointing", "source_strategy"): ("pointing", "source_strategy"),
            ("post_processing", "source_fitting", "fitting_radius_arcsec"): ("pointing", "fit_radius_arcsec"),
            ("post_processing", "source_fitting", "bounding_box_arcsec"): ("pointing", "fit_box_arcsec"),
        }
    for source_path, compact_path in strategy_fields.items():
        copy_path(low_level, compact, summary, source_path, compact_path)

    raw_found, raw_radius = path_get(
        low_level,
        ("timestream", "raw_time_chunk", "despike", "source_protection", "radius_arcsec"),
    )
    ptc_found, ptc_radius = path_get(
        low_level,
        ("timestream", "processed_time_chunk", "flagging", "second_pass_local", "source_protection", "radius_arcsec"),
    )
    if raw_found and ptc_found and raw_radius == ptc_radius:
        set_path(compact, (section_name, "source_protection_radius_arcsec"), raw_radius)
        summary.append(
            {
                "low_level": "timestream.*.source_protection.radius_arcsec",
                "compact": f"{section_name}.source_protection_radius_arcsec",
            }
        )

    if section_name == "oof":
        copy_path(
            low_level,
            compact,
            summary,
            ("timestream", "fruit_loops", "center_keep_radius_arcsec"),
            ("oof", "center_keep_radius_arcsec"),
        )
        copy_path(
            low_level,
            compact,
            summary,
            ("timestream", "fruit_loops", "adaptive_support_radius_arcsec"),
            ("oof", "adaptive_support_radius_arcsec"),
        )


def copy_beammap_section(low_level: dict[str, Any], compact: dict[str, Any], summary: list[dict[str, str]]) -> None:
    beammap_paths = {
        ("beammap", "iter_max"): ("beammap", "iterations"),
        ("beammap", "iter_tolerance"): ("beammap", "convergence_tolerance"),
        ("beammap", "convergence_radius_arcsec"): ("beammap", "convergence_radius_arcsec"),
        ("beammap", "derotate"): ("beammap", "derotate"),
        ("beammap", "subtract_reference_det"): ("beammap", "subtract_reference_det"),
        ("beammap", "reference_det"): ("beammap", "reference_det"),
        ("beammap", "detector_weighting", "mode"): ("beammap", "detector_weighting"),
        ("beammap", "detector_tod_output", "enabled"): ("beammap", "detector_tod"),
    }
    for source_path, compact_path in beammap_paths.items():
        copy_path(low_level, compact, summary, source_path, compact_path)

    priors: dict[str, Any] = {}
    found, enabled = path_get(low_level, ("beammap", "priors", "enabled"))
    if found:
        priors["enabled"] = enabled
    found, filepath = path_get(low_level, ("beammap", "priors", "filepath"))
    if found:
        priors["filepath"] = filepath
    if priors:
        set_path(compact, ("beammap", "priors"), priors)
        summary.append({"low_level": "beammap.priors", "compact": "beammap.priors"})


def build_compact(
    low_level: dict[str, Any],
    *,
    mode: str,
    profile: str,
    include_output_dir: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    compact: dict[str, Any] = {
        "schema": COMPACT_SCHEMA,
        "mode": mode,
        "profile": profile,
    }
    summary: list[dict[str, str]] = []

    copy_common_sections(low_level, compact, summary, include_output_dir)
    if mode == "pointing":
        copy_pointing_section(low_level, compact, summary, "pointing")
    elif mode == "oof":
        copy_pointing_section(low_level, compact, summary, "oof")
    elif mode == "beammap":
        copy_beammap_section(low_level, compact, summary)
    elif mode != "science":
        raise ConvertError(f"unsupported compact mode {mode!r}")

    compact["expert"] = {}
    return compact, summary


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_config", help="Full low-level YAML or TolTECA YAML containing reduce.steps.*.config.low_level.")
    parser.add_argument("--mode", choices=("pointing", "oof", "beammap", "science"), default=None)
    parser.add_argument("--profile", default=None, help="Compact profile name. Defaults to the mode compatibility profile.")
    parser.add_argument("--include-output-dir", action="store_true", help="Emit output.dir even when it is site-specific.")
    parser.add_argument("--output", "-o", default="-", help="Generated compact YAML path, or '-' for stdout.")
    parser.add_argument("--summary-out", default="", help="Optional YAML summary output path.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    base_path = Path(os.path.expandvars(os.path.expanduser(args.base_config))).resolve()
    try:
        low_level = compare_lowlevel_yaml.extract_low_level(load_yaml(base_path))
        if not isinstance(low_level, dict):
            raise ConvertError("low-level config must be a mapping")
        mode = infer_mode(low_level, args.mode)
        profile = args.profile or PASSTHROUGH_PROFILE_BY_MODE[mode]
        profiles_dir = Path(__file__).resolve().with_name("profiles")
        expand_compact_config.load_profile(profiles_dir, profile)
        compact, mappings = build_compact(
            low_level,
            mode=mode,
            profile=profile,
            include_output_dir=args.include_output_dir,
        )
    except (OSError, yaml.YAMLError, ConvertError, expand_compact_config.ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = dump_yaml(compact)
    if args.output == "-":
        print(text, end="")
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    if args.summary_out:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "base_config": str(base_path),
            "mode": mode,
            "profile": profile,
            "include_output_dir": args.include_output_dir,
            "mapping_count": len(mappings),
            "mappings": mappings,
        }
        summary_path = Path(args.summary_out).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(dump_yaml(summary), encoding="utf-8")
    else:
        print(f"generated compact {mode} config with {len(mappings)} mapped paths", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
