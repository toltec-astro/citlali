#!/usr/bin/env python3
"""Generate four checked TolTECA mode kits from accepted low-level baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from tolteca_mode_kit import MODE_REDUCTION_TYPES, canonical_policy, policy_sha256


SCHEMA_VERSION = "citlali-tolteca-mode-kit-manifest-v1"
DEFAULT_EXECUTABLE = "/work/toltec/citlali_dev/citlali_refactor/build/bin/citlali"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"baseline must contain a mapping: {path}")
    return value


def write_yaml(path: Path, value: Any, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {comment}\n" + yaml.safe_dump(value, sort_keys=False, width=100),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cal_items() -> list[dict[str, Any]]:
    return [
        {
            "beammap_source": {
                "fluxes": [
                    {"array_name": "a1100", "uncertainty_mJy": 0.05, "value_mJy": 0.0},
                    {"array_name": "a1400", "uncertainty_mJy": 0.05, "value_mJy": 0.0},
                    {"array_name": "a2000", "uncertainty_mJy": 0.05, "value_mJy": 0.0},
                ]
            },
            "type": "photometry",
        },
        {
            "pointing_offsets": [
                {"axes_name": "az", "value_arcsec": [0.0]},
                {"axes_name": "alt", "value_arcsec": [0.0]},
            ],
            "type": "astrometry",
        },
        {
            "filepath": "../apts/apt_OBSNUM_matched.ecsv",
            "meta": {"interface": "apt"},
            "select": "obsnum == OBSNUM",
            "type": "array_prop_table",
        },
    ]


def observation_patch(mode: str) -> dict[str, Any]:
    scannum = 1 if mode == "oof" else 2
    config: dict[str, Any] = {"cal_items": _cal_items()}
    if mode in {"science", "beammap"}:
        config["cal_objs"] = []
    return {
        "reduce": {
            "inputs": {0: {"select": f"scannum == {scannum} & (obsnum in [OBSNUM])"}},
            "steps": {0: {"config": config}},
        }
    }


def runtime_patch(policy: dict[str, Any]) -> dict[str, Any]:
    low_level: dict[str, Any] = {"runtime": copy.deepcopy(policy["runtime"])}
    kids = policy.get("kids", {})
    solver = kids.get("solver", {}) if isinstance(kids, dict) else {}
    if isinstance(solver, dict) and "fitreportdir" in solver:
        low_level["kids"] = {"solver": {"fitreportdir": solver["fitreportdir"]}}
    return {
        "reduce": {
            "jobkey": "reduced",
            "inputs": {0: {"path": "../data"}},
            "steps": {
                0: {
                    "path": DEFAULT_EXECUTABLE,
                    "config": {"low_level": low_level},
                }
            },
        }
    }


def product_patch(policy: dict[str, Any]) -> dict[str, Any]:
    low_level: dict[str, Any] = {}
    for key in ("coadd", "noise_maps", "post_processing", "wiener_filter"):
        if key in policy:
            low_level[key] = copy.deepcopy(policy[key])

    timestream = policy.get("timestream", {})
    if isinstance(timestream, dict):
        output: dict[str, Any] = {}
        if "output" in timestream:
            output["output"] = copy.deepcopy(timestream["output"])
        for chunk_name in ("raw_time_chunk", "processed_time_chunk"):
            chunk = timestream.get(chunk_name, {})
            if isinstance(chunk, dict) and "output" in chunk:
                output[chunk_name] = {"output": copy.deepcopy(chunk["output"])}
        fruit_loops = timestream.get("fruit_loops", {})
        if isinstance(fruit_loops, dict) and "save_all_iters" in fruit_loops:
            output["fruit_loops"] = {
                "save_all_iters": copy.deepcopy(fruit_loops["save_all_iters"])
            }
        if output:
            low_level["timestream"] = output

    beammap = policy.get("beammap", {})
    if isinstance(beammap, dict):
        products = {
            key: copy.deepcopy(beammap[key])
            for key in ("detector_tod_output", "split_fits_by_flag")
            if key in beammap
        }
        if products:
            low_level["beammap"] = products

    return {"reduce": {"steps": {0: {"config": {"low_level": low_level}}}}}


def pipeline_config(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "reduce": {
            "jobkey": "reduced",
            "inputs": [{"path": "../data", "select": "obsnum == OBSNUM"}],
            "steps": [
                {
                    "name": "citlali",
                    "path": DEFAULT_EXECUTABLE,
                    "config": {
                        "cal_items": _cal_items(),
                        "cal_objs": [],
                        "low_level": policy,
                    },
                }
            ],
        }
    }


def generate_mode(
    mode: str,
    baseline: Path,
    record_id: str,
    output_root: Path,
) -> dict[str, Any]:
    baseline_data = load_yaml(baseline)
    policy = canonical_policy(baseline_data)
    mode_dir = output_root / mode
    write_yaml(
        mode_dir / "70_pipeline.yaml",
        pipeline_config(policy),
        "Repository-managed canonical policy; normal users do not edit this file.",
    )
    write_yaml(
        mode_dir / "71_runtime.yaml",
        runtime_patch(policy),
        "Edit deployment paths and runtime resources for this workspace.",
    )
    write_yaml(
        mode_dir / "72_observation.yaml",
        observation_patch(mode),
        "Replace OBSNUM, APT, calibration, and pointing records for this dataset.",
    )
    write_yaml(
        mode_dir / "80_products.yaml",
        product_patch(policy),
        "Standard product policy for this mode; values intentionally reassert the baseline.",
    )
    (mode_dir / "90_user_overrides.yaml").write_text(
        "# Optional deliberate expert overrides go here and win by numeric precedence.\n{}\n",
        encoding="utf-8",
    )
    return {
        "record_id": record_id,
        "baseline_filename": baseline.name,
        "baseline_sha256": file_sha256(baseline),
        "policy_sha256": policy_sha256(policy),
        "reduction_type": MODE_REDUCTION_TYPES[mode],
        "normalizations": ["remove inputs", "set runtime.output_dir to ."],
    }


def _mode_mapping(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use MODE=VALUE syntax: {value!r}")
        mode, item = value.split("=", 1)
        if mode not in MODE_REDUCTION_TYPES:
            raise ValueError(f"unsupported mode {mode!r}")
        result[mode] = item
    missing = set(MODE_REDUCTION_TYPES) - set(result)
    if missing:
        raise ValueError(f"missing {label} entries for: {', '.join(sorted(missing))}")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", default=[], metavar="MODE=PATH")
    parser.add_argument("--record", action="append", default=[], metavar="MODE=ID")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--kit-version", default="phase4.1-v1")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    baselines = _mode_mapping(args.baseline, "baseline")
    records = _mode_mapping(args.record, "record")
    output_root = Path(args.output_root).expanduser().resolve()
    modes = {}
    for mode in MODE_REDUCTION_TYPES:
        modes[mode] = generate_mode(
            mode,
            Path(baselines[mode]).expanduser().resolve(),
            records[mode],
            output_root,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kit_version": args.kit_version,
        "modes": modes,
    }
    write_yaml(
        output_root / "manifest.yaml",
        manifest,
        "Accepted low-level baselines and canonical policy identities for these kits.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
