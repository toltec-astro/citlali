#!/usr/bin/env python3
"""Expand a compact Citlali config into the current full YAML shape."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


COMPACT_SCHEMA = "citlali-reduction-v2"
PROFILE_SCHEMA = "citlali-compact-profile-v1"
DEFAULT_PROFILE_BY_MODE = {
    "science": "science_standard",
    "pointing": "pointing_standard",
    "oof": "oof_standard",
    "beammap": "beammap_detector",
}
LEGACY_REDUCTION_TYPE_BY_MODE = {
    "oof": "pointing",
}
TOP_LEVEL_KEYS = {
    "schema",
    "mode",
    "profile",
    "inputs",
    "output",
    "runtime",
    "map",
    "products",
    "processing",
    "filter",
    "source",
    "pointing",
    "oof",
    "beammap",
    "expert",
}
TOD_MODES = {
    "none": (False, False),
    "rtc": (True, False),
    "ptc": (False, True),
    "both": (True, True),
}
LOW_LEVEL_EXCLUDED_TOP_LEVEL_KEYS = {"inputs"}


class ConfigError(RuntimeError):
    """Raised for user-correctable compact config errors."""


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fo:
        return yaml.safe_load(fo)


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)


def deep_merge(base: Any, patch: Any) -> Any:
    """Return base with patch recursively applied.

    Dictionaries merge by key. Lists and scalar values are replaced.
    """
    if isinstance(base, dict) and isinstance(patch, dict):
        result = copy.deepcopy(base)
        for key, value in patch.items():
            if key in result:
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    return copy.deepcopy(patch)


def step_values(steps: Any) -> list[Any]:
    if isinstance(steps, list):
        return steps
    if isinstance(steps, dict):
        def step_sort_key(item: Any) -> tuple[int, Any]:
            text = str(item)
            if text.isdigit():
                return (0, int(text))
            return (1, text)

        return [steps[key] for key in sorted(steps, key=step_sort_key)]
    return []


def extract_low_level(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    reduce_section = data.get("reduce")
    if not isinstance(reduce_section, dict):
        return data
    for step in step_values(reduce_section.get("steps", [])):
        if not isinstance(step, dict):
            continue
        config = step.get("config", {})
        if isinstance(config, dict) and "low_level" in config:
            return config["low_level"] or {}
    return data


def to_low_level_config(expanded: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(expanded)
    for key in LOW_LEVEL_EXCLUDED_TOP_LEVEL_KEYS:
        result.pop(key, None)
    return result


def format_expanded_output(expanded: dict[str, Any], output_format: str) -> dict[str, Any]:
    if output_format == "full":
        return expanded
    if output_format == "low_level":
        return to_low_level_config(expanded)
    if output_format == "tolteca":
        return {
            "reduce": {
                "steps": {
                    0: {
                        "config": {
                            "low_level": to_low_level_config(expanded),
                        }
                    }
                }
            }
        }
    raise ConfigError(f"unsupported output format {output_format!r}")


def set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {'.'.join(path)} because {key} is not a mapping")
        cursor = child
    cursor[path[-1]] = copy.deepcopy(value)


def summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return f"mapping({len(value)})"
    if isinstance(value, list):
        return f"list({len(value)})"
    return value


def record_set(
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    path: tuple[str, ...],
    value: Any,
) -> None:
    set_path(patch, path, value)
    applied.append({"path": ".".join(path), "value": summarize_value(value)})


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"`{name}` must be a mapping")
    return value


def warn_unknown_keys(
    warnings: list[str],
    location: str,
    value: dict[str, Any],
    allowed: set[str],
) -> None:
    for key in sorted(set(value) - allowed):
        warnings.append(f"unknown compact key `{location}.{key}` was ignored")


def normalize_boolish(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "enabled", "standard"}:
            return True
        if lowered in {"false", "no", "off", "none", "disabled"}:
            return False
    raise ConfigError(f"`{field}` must be boolean-like, got {value!r}")


def normalize_string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ConfigError(f"`{field}` must be a string or a list of strings")


def load_legacy_inputs(value: Any, compact_path: Path) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        raise ConfigError("`inputs.legacy` must be a YAML path, list, or mapping")
    input_path = Path(value).expanduser()
    if not input_path.is_absolute():
        input_path = compact_path.parent / input_path
    input_data = load_yaml(input_path)
    if isinstance(input_data, dict) and "inputs" in input_data:
        return input_data["inputs"]
    return input_data


def apply_inputs(
    compact: dict[str, Any],
    compact_path: Path,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    if "inputs" not in compact:
        return
    value = compact["inputs"]
    if value is None:
        return
    if isinstance(value, list):
        record_set(patch, applied, ("inputs",), value)
        return
    if isinstance(value, str):
        record_set(patch, applied, ("inputs",), load_legacy_inputs(value, compact_path))
        return
    if not isinstance(value, dict):
        raise ConfigError("`inputs` must be a mapping, list, YAML path, or null")
    allowed = {"legacy", "full", "file", "manifest"}
    if "legacy" in value:
        record_set(patch, applied, ("inputs",), load_legacy_inputs(value["legacy"], compact_path))
    elif "full" in value:
        record_set(patch, applied, ("inputs",), value["full"])
    elif "file" in value:
        record_set(patch, applied, ("inputs",), load_legacy_inputs(value["file"], compact_path))
    elif "manifest" in value:
        record_set(patch, applied, ("inputs",), load_legacy_inputs(value["manifest"], compact_path))
    for key in sorted(set(value) - allowed):
        raise ConfigError(f"unsupported `inputs.{key}`; use legacy, full, file, or manifest")


def apply_direct_mappings(
    src: dict[str, Any],
    mapping: dict[str, tuple[str, ...]],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    for src_key, dst_path in mapping.items():
        if src_key in src:
            record_set(patch, applied, dst_path, src[src_key])


def apply_map_center(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    if value is None or value == "auto":
        return
    if isinstance(value, list) and len(value) == 2:
        record_set(patch, applied, ("mapmaking", "crval1_J2000"), value[0])
        record_set(patch, applied, ("mapmaking", "crval2_J2000"), value[1])
        return
    if isinstance(value, dict):
        ra = value.get("ra_J2000", value.get("ra"))
        dec = value.get("dec_J2000", value.get("dec"))
        if ra is not None and dec is not None:
            record_set(patch, applied, ("mapmaking", "crval1_J2000"), ra)
            record_set(patch, applied, ("mapmaking", "crval2_J2000"), dec)
            return
    raise ConfigError("`map.center` must be auto, [ra, dec], or {ra, dec}")


def apply_map_size(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    if value is None or value == "auto":
        return
    if isinstance(value, int):
        record_set(patch, applied, ("mapmaking", "x_size_pix"), value)
        record_set(patch, applied, ("mapmaking", "y_size_pix"), value)
        return
    if isinstance(value, list) and len(value) == 2:
        record_set(patch, applied, ("mapmaking", "x_size_pix"), value[0])
        record_set(patch, applied, ("mapmaking", "y_size_pix"), value[1])
        return
    if isinstance(value, dict):
        x_size = value.get("x_size_pix", value.get("x_pix"))
        y_size = value.get("y_size_pix", value.get("y_pix"))
        if x_size is not None and y_size is not None:
            record_set(patch, applied, ("mapmaking", "x_size_pix"), x_size)
            record_set(patch, applied, ("mapmaking", "y_size_pix"), y_size)
            return
    raise ConfigError("`map.size` must be auto, an integer, [x, y], or {x_size_pix, y_size_pix}")


def apply_tod_mode(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    if isinstance(value, bool):
        mode = "both" if value else "none"
    elif isinstance(value, str):
        mode = value.strip().lower()
    else:
        raise ConfigError("`products.tod` must be one of none, rtc, ptc, both, true, or false")
    if mode not in TOD_MODES:
        raise ConfigError("`products.tod` must be one of none, rtc, ptc, both, true, or false")
    rtc_enabled, ptc_enabled = TOD_MODES[mode]
    record_set(patch, applied, ("timestream", "raw_time_chunk", "output", "enabled"), rtc_enabled)
    record_set(patch, applied, ("timestream", "processed_time_chunk", "output", "enabled"), ptc_enabled)


def apply_diagnostics(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError("`products.diagnostics` must be a string")
    mode = value.strip().lower()
    if mode in {"none", "off", "minimal", "normal"}:
        return
    if mode in {"verbose", "detailed"}:
        record_set(patch, applied, ("runtime", "verbose"), True)
        return
    if mode == "line_audit":
        record_set(patch, applied, ("runtime", "verbose"), True)
        record_set(patch, applied, ("timestream", "raw_time_chunk", "line_audit", "enabled"), True)
        warnings.append("`products.diagnostics: line_audit` can be expensive and should be validated on Unity")
        return
    raise ConfigError("`products.diagnostics` must be none, minimal, normal, verbose, detailed, or line_audit")


def apply_map_filtering(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if isinstance(value, bool):
        record_set(patch, applied, ("post_processing", "map_filtering", "enabled"), value)
        return
    if not isinstance(value, dict):
        raise ConfigError("`products.map_filtering` must be boolean or a mapping")
    warn_unknown_keys(
        warnings,
        "products.map_filtering",
        value,
        {"enabled", "type", "normalize_errors"},
    )
    apply_direct_mappings(
        value,
        {
            "enabled": ("post_processing", "map_filtering", "enabled"),
            "type": ("post_processing", "map_filtering", "type"),
            "normalize_errors": ("post_processing", "map_filtering", "normalize_errors"),
        },
        patch,
        applied,
    )


def apply_source_finding(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if isinstance(value, bool):
        record_set(patch, applied, ("post_processing", "source_finding", "enabled"), value)
        return
    if not isinstance(value, dict):
        raise ConfigError("`products.source_finding` must be boolean or a mapping")
    warn_unknown_keys(
        warnings,
        "products.source_finding",
        value,
        {"enabled", "mode", "source_sigma", "source_window_arcsec"},
    )
    apply_direct_mappings(
        value,
        {
            "enabled": ("post_processing", "source_finding", "enabled"),
            "mode": ("post_processing", "source_finding", "mode"),
            "source_sigma": ("post_processing", "source_finding", "source_sigma"),
            "source_window_arcsec": ("post_processing", "source_finding", "source_window_arcsec"),
        },
        patch,
        applied,
    )


def apply_tod_indices(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        raise ConfigError("`products.tod_indices` must be a mapping")
    warn_unknown_keys(warnings, "products.tod_indices", value, {"rtc", "ptc"})
    if "rtc" in value:
        record_set(patch, applied, ("timestream", "raw_time_chunk", "output", "indices"), value["rtc"])
    if "ptc" in value:
        record_set(patch, applied, ("timestream", "processed_time_chunk", "output", "indices"), value["ptc"])


def apply_clean_mode(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
) -> None:
    if isinstance(value, bool):
        mode = "standard" if value else "off"
    elif isinstance(value, str):
        mode = value.strip().lower()
    else:
        raise ConfigError("`processing.clean` must be boolean-like or a cleaner name")

    if mode in {"off", "none", "false", "disabled"}:
        record_set(patch, applied, ("timestream", "processed_time_chunk", "clean", "enabled"), False)
        return

    cleaner_flags = {
        "standard_pca": False,
        "null_model": False,
        "marchenko_pastur": False,
        "adaptive_selector": False,
    }
    if mode in {"on", "true", "standard", "pca", "standard_pca"}:
        cleaner_flags["standard_pca"] = True
    elif mode in {"null", "null_model"}:
        cleaner_flags["null_model"] = True
    elif mode in {"mp", "marchenko_pastur"}:
        cleaner_flags["marchenko_pastur"] = True
    elif mode in {"adaptive", "adaptive_selector"}:
        cleaner_flags["adaptive_selector"] = True
    else:
        raise ConfigError(
            "`processing.clean` must be off, standard, null_model, marchenko_pastur, or adaptive_selector"
        )

    record_set(patch, applied, ("timestream", "processed_time_chunk", "clean", "enabled"), True)
    for cleaner, enabled in cleaner_flags.items():
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "clean", cleaner, "enabled"),
            enabled,
        )


def apply_chunking(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        raise ConfigError("`processing.chunking` must be a mapping")
    warn_unknown_keys(warnings, "processing.chunking", value, {"mode", "force", "value"})
    apply_direct_mappings(
        value,
        {
            "mode": ("timestream", "chunking", "chunk_mode"),
            "force": ("timestream", "chunking", "force_chunking"),
            "value": ("timestream", "chunking", "value"),
        },
        patch,
        applied,
    )


def apply_raw_processing(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        raise ConfigError("`processing.raw` must be a mapping")
    warn_unknown_keys(
        warnings,
        "processing.raw",
        value,
        {
            "despike",
            "filter",
            "iir_filter",
            "downsample",
            "flux_calibration",
            "extinction_correction",
            "line_audit",
        },
    )
    apply_direct_mappings(
        value,
        {
            "despike": ("timestream", "raw_time_chunk", "despike", "enabled"),
            "filter": ("timestream", "raw_time_chunk", "filter", "enabled"),
            "iir_filter": ("timestream", "raw_time_chunk", "IIR_filter", "enabled"),
            "downsample": ("timestream", "raw_time_chunk", "downsample", "enabled"),
            "flux_calibration": ("timestream", "raw_time_chunk", "flux_calibration", "enabled"),
            "extinction_correction": ("timestream", "raw_time_chunk", "extinction_correction", "enabled"),
            "line_audit": ("timestream", "raw_time_chunk", "line_audit", "enabled"),
        },
        patch,
        applied,
    )


def apply_standard_pca(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        raise ConfigError("`processing.standard_pca` must be a mapping")
    warn_unknown_keys(warnings, "processing.standard_pca", value, {"n_eig_to_cut", "stddev_limit", "n_calc"})
    apply_direct_mappings(
        value,
        {
            "n_eig_to_cut": ("timestream", "processed_time_chunk", "clean", "standard_pca", "n_eig_to_cut"),
            "stddev_limit": ("timestream", "processed_time_chunk", "clean", "standard_pca", "stddev_limit"),
            "n_calc": ("timestream", "processed_time_chunk", "clean", "standard_pca", "n_calc"),
        },
        patch,
        applied,
    )


def apply_polarimetry(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if isinstance(value, bool):
        record_set(patch, applied, ("timestream", "polarimetry", "enabled"), value)
        return
    if not isinstance(value, dict):
        raise ConfigError("`processing.polarimetry` must be boolean or a mapping")
    warn_unknown_keys(warnings, "processing.polarimetry", value, {"enabled", "grouping", "ignore_hwpr"})
    apply_direct_mappings(
        value,
        {
            "enabled": ("timestream", "polarimetry", "enabled"),
            "grouping": ("timestream", "polarimetry", "grouping"),
            "ignore_hwpr": ("timestream", "polarimetry", "ignore_hwpr"),
        },
        patch,
        applied,
    )


def apply_processing(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    processing = section(compact, "processing")
    warn_unknown_keys(
        warnings,
        "processing",
        processing,
        {
            "tod",
            "chunking",
            "raw",
            "clean",
            "clean_enabled",
            "clean_grouping",
            "standard_pca",
            "weighting",
            "source_mask_radius_arcsec",
            "second_pass_local",
            "learning",
            "polarimetry",
            "fruitloops",
            "fruitloops_iters",
            "fruitloops_source",
            "fruitloops_type",
            "fruitloops_save_all_iters",
            "fruitloops_support_radius_arcsec",
            "fruitloops_support_radius_fwhm",
            "fruitloops_center_keep_radius_arcsec",
        },
    )
    if "tod" in processing:
        record_set(patch, applied, ("timestream", "enabled"), normalize_boolish(processing["tod"], field="processing.tod"))
    if "chunking" in processing:
        apply_chunking(processing["chunking"], patch, applied, warnings)
    if "raw" in processing:
        apply_raw_processing(processing["raw"], patch, applied, warnings)
    if "clean" in processing:
        apply_clean_mode(processing["clean"], patch, applied)
    if "clean_enabled" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "clean", "enabled"),
            normalize_boolish(processing["clean_enabled"], field="processing.clean_enabled"),
        )
    if "clean_grouping" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "clean", "grouping"),
            normalize_string_list(processing["clean_grouping"], field="processing.clean_grouping"),
        )
    if "standard_pca" in processing:
        apply_standard_pca(processing["standard_pca"], patch, applied, warnings)
    if "weighting" in processing:
        record_set(patch, applied, ("timestream", "processed_time_chunk", "weighting", "type"), processing["weighting"])
    if "source_mask_radius_arcsec" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "weighting", "source_mask_radius_arcsec"),
            processing["source_mask_radius_arcsec"],
        )
    if "second_pass_local" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "flagging", "second_pass_local", "enabled"),
            normalize_boolish(processing["second_pass_local"], field="processing.second_pass_local"),
        )
    if "learning" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "learning", "enabled"),
            normalize_boolish(processing["learning"], field="processing.learning"),
        )
    if "polarimetry" in processing:
        apply_polarimetry(processing["polarimetry"], patch, applied, warnings)
    if "fruitloops" in processing:
        record_set(
            patch,
            applied,
            ("timestream", "fruit_loops", "enabled"),
            normalize_boolish(processing["fruitloops"], field="processing.fruitloops"),
        )
    if "fruitloops_iters" in processing:
        record_set(patch, applied, ("timestream", "fruit_loops", "max_iters"), processing["fruitloops_iters"])
    direct_fruitloops = {
        "fruitloops_source": ("timestream", "fruit_loops", "path"),
        "fruitloops_type": ("timestream", "fruit_loops", "type"),
        "fruitloops_save_all_iters": ("timestream", "fruit_loops", "save_all_iters"),
        "fruitloops_support_radius_arcsec": ("timestream", "fruit_loops", "adaptive_support_radius_arcsec"),
        "fruitloops_support_radius_fwhm": ("timestream", "fruit_loops", "adaptive_support_radius_fwhm"),
        "fruitloops_center_keep_radius_arcsec": ("timestream", "fruit_loops", "center_keep_radius_arcsec"),
    }
    apply_direct_mappings(processing, direct_fruitloops, patch, applied)


def apply_source(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    source = section(compact, "source")
    warn_unknown_keys(
        warnings,
        "source",
        source,
        {"map_regime", "fit_model", "fit_radius_arcsec", "fit_box_arcsec"},
    )
    apply_direct_mappings(
        source,
        {
            "map_regime": ("source", "map_regime"),
            "fit_model": ("post_processing", "source_fitting", "model"),
            "fit_radius_arcsec": ("post_processing", "source_fitting", "fitting_radius_arcsec"),
            "fit_box_arcsec": ("post_processing", "source_fitting", "bounding_box_arcsec"),
        },
        patch,
        applied,
    )


def apply_filter(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    filter_config = section(compact, "filter")
    warn_unknown_keys(warnings, "filter", filter_config, {"wiener"})
    wiener = filter_config.get("wiener")
    if wiener is None:
        return
    if not isinstance(wiener, dict):
        raise ConfigError("`filter.wiener` must be a mapping")
    warn_unknown_keys(warnings, "filter.wiener", wiener, {"template_type", "template_fwhm_arcsec", "lowpass_only"})
    apply_direct_mappings(
        wiener,
        {
            "template_type": ("wiener_filter", "template_type"),
            "template_fwhm_arcsec": ("wiener_filter", "template_fwhm_arcsec"),
            "lowpass_only": ("wiener_filter", "lowpass_only"),
        },
        patch,
        applied,
    )


def apply_pointing(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    pointing = section(compact, "pointing")
    warn_unknown_keys(
        warnings,
        "pointing",
        pointing,
        {"source_strategy", "source_protection_radius_arcsec", "fit_radius_arcsec", "fit_box_arcsec"},
    )
    if "source_strategy" in pointing:
        value = pointing["source_strategy"]
        if isinstance(value, str):
            record_set(patch, applied, ("pointing", "source_strategy", "mode"), value)
        elif isinstance(value, dict):
            for key, child in value.items():
                record_set(patch, applied, ("pointing", "source_strategy", key), child)
        else:
            raise ConfigError("`pointing.source_strategy` must be a string or mapping")
    if "source_protection_radius_arcsec" in pointing:
        radius = pointing["source_protection_radius_arcsec"]
        record_set(
            patch,
            applied,
            ("timestream", "raw_time_chunk", "despike", "source_protection", "radius_arcsec"),
            radius,
        )
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "flagging", "second_pass_local", "source_protection", "radius_arcsec"),
            radius,
        )
    if "fit_radius_arcsec" in pointing:
        record_set(patch, applied, ("post_processing", "source_fitting", "fitting_radius_arcsec"), pointing["fit_radius_arcsec"])
    if "fit_box_arcsec" in pointing:
        record_set(patch, applied, ("post_processing", "source_fitting", "bounding_box_arcsec"), pointing["fit_box_arcsec"])


def apply_oof(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    oof = section(compact, "oof")
    warn_unknown_keys(
        warnings,
        "oof",
        oof,
        {
            "source_strategy",
            "source_protection_radius_arcsec",
            "fit_gaussian",
            "fit_radius_arcsec",
            "fit_box_arcsec",
            "fruitloops_center_mode",
            "header_max_radius_arcsec",
            "header_require_coverage",
            "center_keep_radius_arcsec",
            "adaptive_support_radius_arcsec",
        },
    )
    if "source_strategy" in oof:
        record_set(patch, applied, ("pointing", "source_strategy", "mode"), oof["source_strategy"])
    if "fit_gaussian" in oof:
        record_set(patch, applied, ("pointing", "source_strategy", "fit_gaussian"), oof["fit_gaussian"])
    if "fruitloops_center_mode" in oof:
        record_set(
            patch,
            applied,
            ("pointing", "source_strategy", "fruitloops_center_mode"),
            oof["fruitloops_center_mode"],
        )
    if "header_max_radius_arcsec" in oof:
        record_set(
            patch,
            applied,
            ("pointing", "source_strategy", "header_max_radius_arcsec"),
            oof["header_max_radius_arcsec"],
        )
    if "header_require_coverage" in oof:
        record_set(
            patch,
            applied,
            ("pointing", "source_strategy", "header_require_coverage"),
            oof["header_require_coverage"],
        )
    if "source_protection_radius_arcsec" in oof:
        radius = oof["source_protection_radius_arcsec"]
        record_set(
            patch,
            applied,
            ("timestream", "raw_time_chunk", "despike", "source_protection", "radius_arcsec"),
            radius,
        )
        record_set(
            patch,
            applied,
            ("timestream", "processed_time_chunk", "flagging", "second_pass_local", "source_protection", "radius_arcsec"),
            radius,
        )
    if "fit_radius_arcsec" in oof:
        record_set(patch, applied, ("post_processing", "source_fitting", "fitting_radius_arcsec"), oof["fit_radius_arcsec"])
    if "fit_box_arcsec" in oof:
        record_set(patch, applied, ("post_processing", "source_fitting", "bounding_box_arcsec"), oof["fit_box_arcsec"])
    if "center_keep_radius_arcsec" in oof:
        record_set(patch, applied, ("timestream", "fruit_loops", "center_keep_radius_arcsec"), oof["center_keep_radius_arcsec"])
    if "adaptive_support_radius_arcsec" in oof:
        record_set(patch, applied, ("timestream", "fruit_loops", "adaptive_support_radius_arcsec"), oof["adaptive_support_radius_arcsec"])


def apply_beammap_priors(
    value: Any,
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if isinstance(value, bool):
        record_set(patch, applied, ("beammap", "priors", "enabled"), value)
        return
    if not isinstance(value, dict):
        raise ConfigError("`beammap.priors` must be boolean or a mapping")
    warn_unknown_keys(warnings, "beammap.priors", value, {"enabled", "filepath", "file"})
    if "enabled" in value:
        record_set(patch, applied, ("beammap", "priors", "enabled"), normalize_boolish(value["enabled"], field="beammap.priors.enabled"))
    if "filepath" in value or "file" in value:
        filepath = value.get("filepath", value.get("file"))
        record_set(patch, applied, ("beammap", "priors", "filepath"), filepath)
        if "enabled" not in value:
            record_set(patch, applied, ("beammap", "priors", "enabled"), True)


def apply_beammap(
    compact: dict[str, Any],
    patch: dict[str, Any],
    applied: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    beammap = section(compact, "beammap")
    warn_unknown_keys(
        warnings,
        "beammap",
        beammap,
        {
            "iterations",
            "convergence_tolerance",
            "convergence_radius_arcsec",
            "derotate",
            "subtract_reference_det",
            "reference_det",
            "detector_weighting",
            "detector_tod",
            "rfi_mask",
            "scan_band_mask",
            "split_fits",
            "priors",
        },
    )
    direct = {
        "iterations": ("beammap", "iter_max"),
        "convergence_tolerance": ("beammap", "iter_tolerance"),
        "convergence_radius_arcsec": ("beammap", "convergence_radius_arcsec"),
        "derotate": ("beammap", "derotate"),
        "subtract_reference_det": ("beammap", "subtract_reference_det"),
        "reference_det": ("beammap", "reference_det"),
        "detector_weighting": ("beammap", "detector_weighting", "mode"),
        "detector_tod": ("beammap", "detector_tod_output", "enabled"),
        "rfi_mask": ("beammap", "rfi_mask", "enabled"),
        "scan_band_mask": ("beammap", "scan_band_mask", "enabled"),
        "split_fits": ("beammap", "split_fits_by_flag", "enabled"),
    }
    apply_direct_mappings(beammap, direct, patch, applied)
    if "priors" in beammap:
        apply_beammap_priors(beammap["priors"], patch, applied, warnings)


def build_compact_patch(compact: dict[str, Any], compact_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    patch: dict[str, Any] = {}
    applied: list[dict[str, Any]] = []
    warnings: list[str] = []

    for key in sorted(set(compact) - TOP_LEVEL_KEYS):
        warnings.append(f"unknown compact top-level key `{key}` was ignored")

    mode = compact.get("mode")
    if mode is not None:
        record_set(patch, applied, ("runtime", "reduction_type"), LEGACY_REDUCTION_TYPE_BY_MODE.get(mode, mode))

    apply_inputs(compact, compact_path, patch, applied)

    output = section(compact, "output")
    warn_unknown_keys(warnings, "output", output, {"dir", "subdir", "verbose"})
    apply_direct_mappings(
        output,
        {
            "dir": ("runtime", "output_dir"),
            "subdir": ("runtime", "use_subdir"),
            "verbose": ("runtime", "verbose"),
        },
        patch,
        applied,
    )

    runtime = section(compact, "runtime")
    warn_unknown_keys(warnings, "runtime", runtime, {"threads", "parallel", "fitreport_dir"})
    apply_direct_mappings(
        runtime,
        {
            "threads": ("runtime", "n_threads"),
            "parallel": ("runtime", "parallel_policy"),
            "fitreport_dir": ("kids", "solver", "fitreportdir"),
        },
        patch,
        applied,
    )

    map_config = section(compact, "map")
    warn_unknown_keys(
        warnings,
        "map",
        map_config,
        {"unit", "method", "grouping", "pixel_axes", "pixel_size_arcsec", "center", "size", "coadd"},
    )
    apply_direct_mappings(
        map_config,
        {
            "unit": ("mapmaking", "cunit"),
            "method": ("mapmaking", "method"),
            "grouping": ("mapmaking", "grouping"),
            "pixel_axes": ("mapmaking", "pixel_axes"),
            "pixel_size_arcsec": ("mapmaking", "pixel_size_arcsec"),
            "coadd": ("coadd", "enabled"),
        },
        patch,
        applied,
    )
    if "center" in map_config:
        apply_map_center(map_config["center"], patch, applied)
    if "size" in map_config:
        apply_map_size(map_config["size"], patch, applied)

    products = section(compact, "products")
    warn_unknown_keys(
        warnings,
        "products",
        products,
        {
            "maps",
            "noise",
            "noise_count",
            "noise_products",
            "noise_realizations",
            "noise_randomize_dets",
            "noise_apply_empirical_weights",
            "tod",
            "tod_indices",
            "diagnostics",
            "map_filtering",
            "map_histogram_bins",
            "source_finding",
        },
    )
    apply_direct_mappings(
        products,
        {
            "maps": ("mapmaking", "enabled"),
            "noise": ("noise_maps", "enabled"),
            "noise_count": ("noise_maps", "n_noise_maps"),
            "noise_products": ("noise_maps", "products", "enabled"),
            "noise_realizations": ("noise_maps", "write_realizations"),
            "noise_randomize_dets": ("noise_maps", "randomize_dets"),
            "noise_apply_empirical_weights": ("noise_maps", "products", "apply_empirical_weights"),
            "map_histogram_bins": ("post_processing", "map_histogram_n_bins"),
        },
        patch,
        applied,
    )
    if "tod" in products:
        apply_tod_mode(products["tod"], patch, applied)
    if "tod_indices" in products:
        apply_tod_indices(products["tod_indices"], patch, applied, warnings)
    if "diagnostics" in products:
        apply_diagnostics(products["diagnostics"], patch, applied, warnings)
    if "map_filtering" in products:
        apply_map_filtering(products["map_filtering"], patch, applied, warnings)
    if "source_finding" in products:
        apply_source_finding(products["source_finding"], patch, applied, warnings)

    apply_processing(compact, patch, applied, warnings)
    apply_filter(compact, patch, applied, warnings)
    apply_source(compact, patch, applied, warnings)
    apply_pointing(compact, patch, applied, warnings)
    apply_oof(compact, patch, applied, warnings)
    apply_beammap(compact, patch, applied, warnings)
    return patch, applied, warnings


def leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(leaf_paths(child, prefix + (str(key),)))
        return paths
    return [".".join(prefix)]


def load_profile(profiles_dir: Path, name: str) -> tuple[Path, dict[str, Any]]:
    profile_path = profiles_dir / f"{name}.yaml"
    if not profile_path.exists():
        raise ConfigError(f"profile `{name}` not found at {profile_path}")
    profile = load_yaml(profile_path)
    if not isinstance(profile, dict):
        raise ConfigError(f"profile `{name}` must be a mapping")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ConfigError(f"profile `{name}` has unsupported schema {profile.get('schema')!r}")
    if profile.get("name") != name:
        raise ConfigError(f"profile file {profile_path} declares name {profile.get('name')!r}")
    if "full" not in profile or not isinstance(profile["full"], dict):
        raise ConfigError(f"profile `{name}` must contain a `full` mapping")
    return profile_path, profile


def expand_config(
    compact_path: Path,
    base_config_path: Path | None,
    profiles_dir: Path,
    profile_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {} if base_config_path is None else extract_low_level(load_yaml(base_config_path))
    compact = load_yaml(compact_path)
    if not isinstance(base, dict):
        raise ConfigError("base config must be a mapping")
    if not isinstance(compact, dict):
        raise ConfigError("compact config must be a mapping")
    if compact.get("schema") != COMPACT_SCHEMA:
        raise ConfigError(f"compact config must declare `schema: {COMPACT_SCHEMA}`")

    mode = compact.get("mode")
    profile_name = profile_override or compact.get("profile")
    if profile_name is None:
        if mode not in DEFAULT_PROFILE_BY_MODE:
            raise ConfigError("compact config must set `profile` when `mode` has no default profile")
        profile_name = DEFAULT_PROFILE_BY_MODE[mode]

    profile_path, profile = load_profile(profiles_dir, str(profile_name))
    profile_mode = profile.get("mode")
    if mode is None:
        mode = profile_mode
    elif profile_mode is not None and mode != profile_mode:
        raise ConfigError(f"compact mode `{mode}` does not match profile `{profile_name}` mode `{profile_mode}`")

    compact_patch, applied, warnings = build_compact_patch(compact, compact_path)
    expert_patch = compact.get("expert", {})
    if expert_patch is None:
        expert_patch = {}
    if not isinstance(expert_patch, dict):
        raise ConfigError("`expert` must be a mapping when present")

    expanded = deep_merge(base, profile["full"])
    expanded = deep_merge(expanded, compact_patch)
    expanded = deep_merge(expanded, expert_patch)

    summary = {
        "schema": "citlali-compact-expansion-summary-v1",
        "compact_schema": COMPACT_SCHEMA,
        "compact_config": str(compact_path),
        "base_config": "none" if base_config_path is None else str(base_config_path),
        "profile": str(profile_name),
        "profile_path": str(profile_path),
        "mode": mode,
        "legacy_reduction_type": LEGACY_REDUCTION_TYPE_BY_MODE.get(str(mode), mode),
        "applied_compact_paths": applied,
        "expert_override_paths": sorted(leaf_paths(expert_patch)),
        "warnings": warnings,
    }
    return expanded, summary


def list_profiles(profiles_dir: Path) -> None:
    for path in sorted(profiles_dir.glob("*.yaml")):
        try:
            profile = load_yaml(path)
        except Exception as exc:  # pragma: no cover - CLI diagnostic path
            print(f"{path.name}: failed to load: {exc}", file=sys.stderr)
            continue
        if not isinstance(profile, dict):
            continue
        name = profile.get("name", path.stem)
        mode = profile.get("mode", "unknown")
        description = profile.get("description", "")
        print(f"{name}\t{mode}\t{description}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact_config", nargs="?", help="Compact config YAML to expand.")
    parser.add_argument(
        "--base-config",
        default="data/config.yaml",
        help=(
            "Current full config baseline, a TolTECA YAML file containing "
            "reduce.steps.*.config.low_level, or `none`/`empty` to expand from "
            "an empty low-level base."
        ),
    )
    parser.add_argument("--profiles-dir", default="tools/config/profiles", help="Directory of compact profile YAML files.")
    parser.add_argument("--profile", default=None, help="Override profile name from the compact config.")
    parser.add_argument("--output", "-o", default="-", help="Expanded full config output path, or '-' for stdout.")
    parser.add_argument(
        "--output-format",
        choices=("full", "low_level", "tolteca"),
        default="full",
        help="Output full Citlali YAML, a bare low_level block, or a TolTECA reduce.steps.0.config.low_level wrapper.",
    )
    parser.add_argument("--summary-out", default="", help="Optional YAML expansion summary output path.")
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return non-zero when expansion emits warnings such as unknown compact keys.",
    )
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    profiles_dir = Path(args.profiles_dir).expanduser().resolve()

    if args.list_profiles:
        list_profiles(profiles_dir)
        return 0

    if not args.compact_config:
        print("error: compact_config is required unless --list-profiles is used", file=sys.stderr)
        return 2

    compact_path = Path(args.compact_config).expanduser().resolve()
    base_config_path = None
    if str(args.base_config).strip().lower() not in {"none", "empty"}:
        base_config_path = Path(args.base_config).expanduser().resolve()
    try:
        expanded, summary = expand_config(compact_path, base_config_path, profiles_dir, args.profile)
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.fail_on_warnings and summary["warnings"]:
        for warning in summary["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        print("error: compact config emitted warnings", file=sys.stderr)
        return 1

    output_data = format_expanded_output(expanded, args.output_format)
    expanded_text = dump_yaml(output_data)
    if args.output == "-":
        print(expanded_text, end="")
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(expanded_text, encoding="utf-8")

    if args.summary_out:
        summary_path = Path(args.summary_out).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(dump_yaml(summary), encoding="utf-8")
    else:
        print(
            f"expanded {compact_path.name} with profile {summary['profile']} "
            f"as {args.output_format} "
            f"({len(summary['applied_compact_paths'])} compact overrides, "
            f"{len(summary['expert_override_paths'])} expert overrides)",
            file=sys.stderr,
        )
        for warning in summary["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
