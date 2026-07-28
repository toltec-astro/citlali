#!/usr/bin/env python3
"""Configure and verify the generated NGC4449 pointing reduction.

Run this only after ``tolproj setup-pointing-reductions --refactor`` has
created a fresh pointing directory.  The script deliberately edits only the
two reducer-owned files documented by TolProj:

* 81_pointing_defaults.yaml
* 82_pointing_products.yaml

Without ``--write`` it performs a dry run.  In both modes it verifies the
installed kit, selected observations, and explicit hero APT bindings.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml


EXPECTED_OBSNUMS = (
    152389,
    152391,
    152393,
    152418,
    152420,
    152430,
    152432,
    152434,
)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected a YAML mapping")
    return data


def _replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one editable occurrence, found {count}"
        )
    return updated


def _configured_text(defaults_path: Path, products_path: Path) -> tuple[str, str]:
    defaults = defaults_path.read_text()
    defaults = _replace_one(
        defaults,
        (
            r"^([ \t]+fruit_loops:\n"
            r"[ \t]+enabled:[ \t]*)(?:true|false)([ \t]*(?:#.*)?)$"
        ),
        r"\1true\2",
        "fruit_loops.enabled",
    )
    defaults = _replace_one(
        defaults,
        r"^([ \t]+max_iters:[ \t]*)[0-9]+([ \t]*(?:#.*)?)$",
        r"\g<1>10\2",
        "fruit_loops.max_iters",
    )
    defaults = _replace_one(
        defaults,
        r"^([ \t]+save_all_iters:[ \t]*)(?:true|false)([ \t]*(?:#.*)?)$",
        r"\1true\2",
        "fruit_loops.save_all_iters",
    )

    products = products_path.read_text()
    products = _replace_one(
        products,
        (
            r"^([ \t]+raw_time_chunk:\n"
            r"[ \t]+output:\n"
            r"[ \t]+enabled:[ \t]*)(?:true|false)([ \t]*(?:#.*)?)$"
        ),
        r"\1false\2",
        "raw_time_chunk.output.enabled",
    )
    products = _replace_one(
        products,
        (
            r"^([ \t]+processed_time_chunk:\n"
            r"[ \t]+output:\n"
            r"[ \t]+enabled:[ \t]*)(?:true|false)([ \t]*(?:#.*)?)$"
        ),
        r"\1false\2",
        "processed_time_chunk.output.enabled",
    )
    return defaults, products


def _steps_zero(data: dict) -> dict:
    steps = data["reduce"]["steps"]
    return steps[0] if 0 in steps else steps["0"]


def _verify_scientific_settings(defaults: dict, products: dict) -> None:
    defaults_ll = _steps_zero(defaults)["config"]["low_level"]
    products_ll = _steps_zero(products)["config"]["low_level"]
    fruit = defaults_ll["timestream"]["fruit_loops"]
    if fruit["enabled"] is not True:
        raise RuntimeError("fruit_loops.enabled is not true")
    if fruit["max_iters"] != 10:
        raise RuntimeError(f"fruit_loops.max_iters is {fruit['max_iters']!r}, not 10")
    if fruit["save_all_iters"] is not True:
        raise RuntimeError("fruit_loops.save_all_iters is not true")
    if defaults_ll["timestream"]["learning"]["enabled"] is not True:
        raise RuntimeError("timestream.learning.enabled is not true")
    if products_ll["noise_maps"]["enabled"] is not True:
        raise RuntimeError("noise_maps.enabled is not true")
    if products_ll["noise_maps"]["products"]["enabled"] is not True:
        raise RuntimeError("noise-map empirical products are not enabled")
    if products_ll["timestream"]["raw_time_chunk"]["output"]["enabled"] is not False:
        raise RuntimeError("raw timestream output is not disabled")
    processed = products_ll["timestream"]["processed_time_chunk"]["output"]
    if processed["enabled"] is not False:
        raise RuntimeError("processed timestream output is not disabled")


def _verify_observation_binding(path: Path, apt_dir: Path) -> None:
    data = _load_yaml(path)
    reduce = data["reduce"]
    inputs = reduce["inputs"]
    input_zero = inputs[0] if 0 in inputs else inputs["0"]
    selector = input_zero["select"]
    match = re.search(r"obsnum\s+in\s+\[([^\]]+)\]", selector)
    if match is None:
        raise RuntimeError(f"{path}: cannot parse obsnum selection: {selector!r}")
    selected = tuple(sorted(int(value) for value in re.findall(r"\d+", match.group(1))))
    expected = tuple(sorted(EXPECTED_OBSNUMS))
    if selected != expected:
        raise RuntimeError(
            f"{path}: selected obsnums {selected!r}; expected {expected!r}"
        )

    cal_items = _steps_zero(data)["config"]["cal_items"]
    apt_items = [
        item
        for item in cal_items
        if item.get("type") == "array_prop_table"
    ]
    if len(apt_items) != len(expected):
        raise RuntimeError(
            f"{path}: found {len(apt_items)} APT items; expected {len(expected)}"
        )
    actual_paths = {Path(item["filepath"]) for item in apt_items}
    expected_paths = {
        apt_dir / f"apt_{obsnum}_matched.ecsv" for obsnum in EXPECTED_OBSNUMS
    }
    if actual_paths != expected_paths:
        missing = sorted(str(p) for p in expected_paths - actual_paths)
        unexpected = sorted(str(p) for p in actual_paths - expected_paths)
        raise RuntimeError(
            f"{path}: hero APT binding mismatch; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _verify_kit(path: Path) -> None:
    marker = _load_yaml(path)
    expected = {
        "schema_version": "tolproj-installed-citlali-refactor-kit-v2",
        "kit_version": "phase4.1-v2.1",
        "bundle": "phase4_1_v2_1",
        "mode": "pointing",
        "observation_filename": "72_pointing_observation.yaml",
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{path}: unexpected TolProj kit identity: {mismatches!r}")


def _write_with_backup(path: Path, text: str) -> None:
    backup = path.with_name(f"{path.name}.before_ngc4449")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reduction_dir",
        type=Path,
        help="Fresh TolProj-generated pointing reduction directory",
    )
    parser.add_argument(
        "--apt-dir",
        type=Path,
        help="Expected absolute hero APT directory; defaults to PROJECT/apts/hero",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the requested settings; otherwise perform a dry run",
    )
    args = parser.parse_args()

    reduction_dir = args.reduction_dir.expanduser().resolve()
    project_dir = reduction_dir.parent
    apt_dir = (
        args.apt_dir.expanduser().resolve()
        if args.apt_dir is not None
        else (project_dir / "apts" / "hero").resolve()
    )
    defaults_path = reduction_dir / "81_pointing_defaults.yaml"
    products_path = reduction_dir / "82_pointing_products.yaml"
    observation_path = reduction_dir / "72_pointing_observation.yaml"
    marker_path = reduction_dir / ".citlali_refactor_kit.yaml"

    required = (defaults_path, products_path, observation_path, marker_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing generated file(s): {missing!r}")

    _verify_kit(marker_path)
    _verify_observation_binding(observation_path, apt_dir)
    defaults_text, products_text = _configured_text(defaults_path, products_path)

    if args.write:
        _write_with_backup(defaults_path, defaults_text)
        _write_with_backup(products_path, products_text)
        action = "configured"
    else:
        action = "dry-run verified"

    defaults = yaml.safe_load(defaults_text)
    products = yaml.safe_load(products_text)
    _verify_scientific_settings(defaults, products)

    print(f"{action}: {reduction_dir}")
    print(f"pointings: {', '.join(str(v) for v in EXPECTED_OBSNUMS)}")
    print(f"APT directory: {apt_dir}")
    print("fruit loops: enabled, max_iters=10, save_all_iters=true")
    print("learning/noise products: enabled")
    print("raw and processed timestream products: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
