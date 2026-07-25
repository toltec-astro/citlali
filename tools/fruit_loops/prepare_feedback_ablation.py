#!/usr/bin/env python3
"""Prepare controlled fruit-loop feedback ablations from one low-level config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


VARIANTS = {
    "full_policy_diagnostic": {},
    "learning_disabled": {
        ("timestream", "learning", "enabled"): False,
    },
    "weight_feedback_disabled": {
        (
            "timestream",
            "fruit_loops",
            "weight_feedback",
            "enabled",
        ): False,
    },
    "recompute_weights_after_addback": {
        (
            "timestream",
            "fruit_loops",
            "recompute_weights_after_addback",
        ): True,
    },
    "all_three": {
        ("timestream", "learning", "enabled"): False,
        (
            "timestream",
            "fruit_loops",
            "weight_feedback",
            "enabled",
        ): False,
        (
            "timestream",
            "fruit_loops",
            "recompute_weights_after_addback",
        ): True,
    },
    "snr_only_model": {
        ("timestream", "fruit_loops", "array_flux_limit"): [0, 0, 0],
    },
    "snr_only_s50": {
        ("timestream", "fruit_loops", "array_flux_limit"): [0, 0, 0],
        ("timestream", "fruit_loops", "sig2noise_limit"): 50,
    },
    "snr_only_s200": {
        ("timestream", "fruit_loops", "array_flux_limit"): [0, 0, 0],
        ("timestream", "fruit_loops", "sig2noise_limit"): 200,
    },
    "adaptive_peak_5pct": {
        ("timestream", "fruit_loops", "array_flux_limit"): [0, 0, 0],
        ("timestream", "fruit_loops", "sig2noise_limit"): 0,
        ("timestream", "fruit_loops", "peak_fraction_limit"): 0.05,
        ("timestream", "fruit_loops", "local_snr_floor"): 0,
    },
    "adaptive_local_snr5": {
        ("timestream", "fruit_loops", "array_flux_limit"): [0, 0, 0],
        ("timestream", "fruit_loops", "sig2noise_limit"): 0,
        ("timestream", "fruit_loops", "peak_fraction_limit"): 0,
        ("timestream", "fruit_loops", "local_snr_floor"): 5,
    },
    "ptc_cleaning_disabled": {
        ("timestream", "processed_time_chunk", "clean", "enabled"): False,
    },
    "ptc_pca_one_mode": {
        (
            "timestream",
            "processed_time_chunk",
            "clean",
            "standard_pca",
            "n_eig_to_cut",
        ): {"a1100": [1], "a1400": [1], "a2000": [1]},
    },
    "ptc_pca_ten_modes": {
        (
            "timestream",
            "processed_time_chunk",
            "clean",
            "standard_pca",
            "n_eig_to_cut",
        ): {"a1100": [10], "a1400": [10], "a2000": [10]},
    },
    "ptc_source_mask_30arcsec": {
        (
            "timestream",
            "processed_time_chunk",
            "clean",
            "mask_radius_arcsec",
        ): 30,
    },
    "projection_bilinear": {
        ("timestream", "fruit_loops", "interp_mode_override"): "bilinear",
    },
    "projection_legacy_trunc": {
        ("timestream", "fruit_loops", "interp_mode_override"): "trunc",
        ("timestream", "fruit_loops", "legacy_center"): True,
    },
    "naive_mapmaking": {
        ("mapmaking", "method"): "naive",
    },
    "full_policy_10_iters": {
        ("timestream", "fruit_loops", "max_iters"): 10,
    },
}

INITIAL_VARIANTS = (
    "full_policy_diagnostic",
    "learning_disabled",
    "weight_feedback_disabled",
    "recompute_weights_after_addback",
    "all_three",
)

FOLLOWUP_VARIANTS = tuple(
    name for name in VARIANTS if name not in INITIAL_VARIANTS
)


def nested_get(config: dict, path: tuple[str, ...]) -> object:
    node: object = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"required config path is absent: {'.'.join(path)}")
        node = node[key]
    return node


def nested_set(config: dict, path: tuple[str, ...], value: object) -> None:
    node = config
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"required config path is absent: {'.'.join(path)}")
        node = child
    if path[-1] not in node:
        raise ValueError(f"required config path is absent: {'.'.join(path)}")
    node[path[-1]] = value


def require_seed_policy(config: dict) -> None:
    expected = {
        ("runtime", "reduction_type"): "pointing",
        ("timestream", "fruit_loops", "enabled"): True,
        ("timestream", "fruit_loops", "max_iters"): 5,
        ("timestream", "fruit_loops", "save_all_iters"): True,
        ("timestream", "learning", "enabled"): True,
        (
            "timestream",
            "fruit_loops",
            "weight_feedback",
            "enabled",
        ): True,
        (
            "timestream",
            "fruit_loops",
            "recompute_weights_after_addback",
        ): False,
    }
    mismatches = []
    for path, wanted in expected.items():
        actual = nested_get(config, path)
        if actual != wanted:
            mismatches.append(
                f"{'.'.join(path)} expected={wanted!r} actual={actual!r}"
            )
    inputs = config.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        mismatches.append("inputs must contain exactly one observation")
    else:
        name = inputs[0].get("meta", {}).get("name")
        if not str(name).startswith("133410_"):
            mismatches.append(f"expected observation 133410, found {name!r}")
    if mismatches:
        raise ValueError("input is not the frozen 133410 policy:\n  " + "\n  ".join(mismatches))


def write_manifest(
    output_dir: Path,
    source: Path,
    output_root: str,
    variant_names: tuple[str, ...],
) -> None:
    rows = []
    for name in variant_names:
        changes = VARIANTS[name]
        rows.append(
            {
                "name": name,
                "config": f"citlali_fruitloop_ablation_{name}_o133410.yaml",
                "runtime_output_dir": f"{output_root.rstrip('/')}/{name}/reduced/",
                "changes": [
                    {"path": ".".join(path), "value": value}
                    for path, value in changes.items()
                ],
            }
        )
    manifest = {
        "source_config": str(source),
        "observation": 133410,
        "common_controls": {
            "fruit_loops_enabled": True,
            "max_iters": 5,
            "save_all_iters": True,
            "fruit_loop_diagnostics_enabled": True,
        },
        "variants": rows,
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-output-root",
        required=True,
        help="Unity output root; each variant receives its own reduced directory",
    )
    parser.add_argument(
        "--matrix",
        choices=("initial", "followup", "all"),
        default="all",
        help="Variant matrix to generate (default: all)",
    )
    args = parser.parse_args()

    source = yaml.safe_load(args.input.read_text())
    if not isinstance(source, dict):
        raise ValueError("input must contain one YAML mapping")
    require_seed_policy(source)

    variant_names = {
        "initial": INITIAL_VARIANTS,
        "followup": FOLLOWUP_VARIANTS,
        "all": tuple(VARIANTS),
    }[args.matrix]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in variant_names:
        changes = VARIANTS[name]
        config = copy.deepcopy(source)
        for path, value in changes.items():
            nested_set(config, path, value)
        nested_set(
            config,
            ("runtime", "output_dir"),
            f"{args.runtime_output_root.rstrip('/')}/{name}/reduced/",
        )
        fruit_loops = nested_get(config, ("timestream", "fruit_loops"))
        assert isinstance(fruit_loops, dict)
        fruit_loops["diagnostics_enabled"] = True
        destination = (
            args.output_dir
            / f"citlali_fruitloop_ablation_{name}_o133410.yaml"
        )
        destination.write_text(yaml.safe_dump(config, sort_keys=False))

    write_manifest(
        args.output_dir,
        args.input.resolve(),
        args.runtime_output_root,
        variant_names,
    )
    print(
        f"wrote {len(variant_names)} {args.matrix} configs to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
