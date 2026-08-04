#!/usr/bin/env python3
"""Prove a MAP naive-versus-JINC comparator is configuration-equivalent.

This is a file-only owner tool.  It reads either two already merged Citlali
YAML files or the ordered numbered YAML fragments of two reduction directories.
It never contacts Unity, invokes TolProj/Citlali, writes a reduction, or
submits a job.  A passing result means that the two semantic low-level
configurations differ solely at ``mapmaking.method`` (naive -> jinc), and that
the science comparator has the empirical-noise and disabled-fruit-loop
configuration needed to avoid the CAP-SCIENCE admission failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import yaml


NUMBERED = re.compile(r"^[0-9]{2}_.+\\.ya?ml$")
LOW_LEVEL_PATH = ("reduce", "steps", 0, "config", "low_level")
METHOD_PATH = ("mapmaking", "method")
SCIENCE_REQUIREMENTS = {
    ("noise_maps", "enabled"): True,
    ("noise_maps", "n_noise_maps"): 64,
    ("noise_maps", "products", "enabled"): True,
    ("timestream", "fruit_loops", "enabled"): False,
}


class CheckError(RuntimeError):
    """A comparator is absent, malformed, or scientifically non-equivalent."""


def fail(message: str) -> None:
    raise CheckError(message)


def load_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as exc:
        fail(f"cannot read YAML {path}: {exc}")


def indexed(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if index < len(value):
            return value[index]
    elif isinstance(value, Mapping):
        if index in value:
            return value[index]
        if str(index) in value:
            return value[str(index)]
    fail(f"missing indexed configuration step {index}")


def nested(value: Any, path: tuple[str | int, ...]) -> Any:
    current = value
    for item in path:
        if isinstance(item, int):
            current = indexed(current, item)
        elif isinstance(current, Mapping) and item in current:
            current = current[item]
        else:
            fail("missing configuration leaf " + ".".join(map(str, path)))
    return current


def merge(base: Any, update: Any) -> Any:
    """Apply the mapping merge used by the ordered numbered configuration kit."""
    if isinstance(base, Mapping) and isinstance(update, Mapping):
        result = dict(base)
        for key, value in update.items():
            result[key] = merge(result[key], value) if key in result else value
        return result
    return update


def merged_directory(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_dir():
        fail(f"reduction directory is absent: {path}")
    fragments = sorted(
        item for item in path.iterdir()
        if item.is_file() and NUMBERED.fullmatch(item.name)
    )
    if not fragments:
        fail(f"no numbered YAML fragments in {path}")
    result: Any = {}
    for fragment in fragments:
        loaded = load_yaml(fragment)
        if not isinstance(loaded, Mapping):
            fail(f"numbered YAML fragment is not a mapping: {fragment}")
        result = merge(result, loaded)
    if not isinstance(result, dict):
        fail(f"merged numbered configuration is not a mapping: {path}")
    return result, [item.name for item in fragments]


def flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        result: dict[tuple[str, ...], Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            result.update(flatten(value[key], prefix + (str(key),)))
        return result
    return {prefix: value}


def equal(left: Any, right: Any) -> bool:
    return (
        isinstance(left, float) and isinstance(right, float)
        and math.isnan(left) and math.isnan(right)
    ) or left == right


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_from_args(args: argparse.Namespace, prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_arg = getattr(args, f"{prefix}_merged")
    directory_arg = getattr(args, f"{prefix}_directory")
    if bool(merged_arg) == bool(directory_arg):
        fail(f"supply exactly one --{prefix.replace('_', '-')}-(merged|directory)")
    if merged_arg:
        path = Path(merged_arg).expanduser().resolve(strict=True)
        loaded = load_yaml(path)
        if not isinstance(loaded, dict):
            fail(f"merged configuration is not a mapping: {path}")
        return loaded, {"kind": "merged-file", "path": str(path), "sha256": sha256(path)}
    directory = Path(directory_arg).expanduser().resolve(strict=True)
    merged, fragments = merged_directory(directory)
    return merged, {"kind": "numbered-directory", "path": str(directory), "fragments": fragments}


def comparison(baseline: dict[str, Any], candidate: dict[str, Any], *, science: bool,
               strict: bool = True) -> dict[str, Any]:
    baseline_low = nested(baseline, LOW_LEVEL_PATH)
    candidate_low = nested(candidate, LOW_LEVEL_PATH)
    if not isinstance(baseline_low, Mapping) or not isinstance(candidate_low, Mapping):
        fail("reduce.steps.0.config.low_level must be a mapping")
    baseline_leaves = flatten(baseline_low)
    candidate_leaves = flatten(candidate_low)
    keys = sorted(set(baseline_leaves) | set(candidate_leaves))
    differences = [
        {
            "path": ".".join(key),
            "baseline": baseline_leaves.get(key),
            "candidate": candidate_leaves.get(key),
        }
        for key in keys
        if key not in baseline_leaves or key not in candidate_leaves
        or not equal(baseline_leaves[key], candidate_leaves[key])
    ]
    allowed = ".".join(METHOD_PATH)
    errors: list[str] = []
    if [item["path"] for item in differences] != [allowed]:
        errors.append("semantic low-level diff is not method-only")
    baseline_method = nested(baseline_low, METHOD_PATH)
    candidate_method = nested(candidate_low, METHOD_PATH)
    if baseline_method != "naive" or candidate_method != "jinc":
        errors.append("mapmaking method transition is not exactly naive -> jinc")
    requirements: dict[str, Any] = {}
    if science:
        for path, expected in SCIENCE_REQUIREMENTS.items():
            baseline_value = nested(baseline_low, path)
            candidate_value = nested(candidate_low, path)
            if baseline_value != expected or candidate_value != expected:
                errors.append(
                    "science admission requirement differs at " + ".".join(path)
                    + f": baseline={baseline_value!r}, candidate={candidate_value!r}, expected={expected!r}"
                )
            requirements[".".join(path)] = expected
    if errors and strict:
        fail("; ".join(errors) + ": " + json.dumps(differences, sort_keys=True))
    return {
        "status": "pass" if not errors else "nonconformant",
        "errors": errors,
        "semantic_diff": differences,
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "science_admission_requirements": requirements,
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    baseline, baseline_source = source_from_args(args, "baseline")
    candidate, candidate_source = source_from_args(args, "candidate")
    result = comparison(
        baseline, candidate, science=args.science,
        strict=not args.report_nonconformant,
    )
    return {
        "schema_version": "sci-map-001-jinc-comparator-check-v1",
        "comparison_kind": "science" if args.science else "point",
        "baseline": baseline_source,
        "candidate": candidate_source,
        **result,
    }


def self_check() -> None:
    baseline = {
        "reduce": {
            "steps": {
                0: {
                    "config": {
                        "low_level": {
                            "mapmaking": {"method": "naive", "pixel_size_arcsec": 2.0},
                            "noise_maps": {"enabled": True, "n_noise_maps": 64,
                                           "products": {"enabled": True}},
                            "runtime": {"n_threads": 1, "parallel_policy": "seq"},
                            "timestream": {"fruit_loops": {
                                "enabled": False, "sig2noise_limit": 2.5,
                            }},
                        },
                    },
                },
            },
        },
    }
    candidate = merge(baseline, {
        "reduce": {"steps": {0: {"config": {"low_level": {
            "mapmaking": {"method": "jinc"},
        }}}}},
    })
    checked = comparison(baseline, candidate, science=True)
    if checked["semantic_diff"] != [{"path": "mapmaking.method", "baseline": "naive", "candidate": "jinc"}]:
        fail("self-check did not produce the expected method-only diff")
    bad = merge(candidate, {
        "reduce": {"steps": {0: {"config": {"low_level": {
            "runtime": {"n_threads": 8},
        }}}}},
    })
    try:
        comparison(baseline, bad, science=True)
    except CheckError:
        return
    fail("self-check accepted a non-method-only change")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-check", action="store_true")
    for prefix in ("baseline", "candidate"):
        group = result.add_mutually_exclusive_group()
        group.add_argument(f"--{prefix}-merged")
        group.add_argument(f"--{prefix}-directory")
    result.add_argument("--science", action="store_true", help="enforce the 64-realization empirical-noise contract")
    result.add_argument(
        "--report-nonconformant", action="store_true",
        help="write a full structural diff for inspection, then return status 1 if it is not method-only",
    )
    result.add_argument("--output", type=Path, help="new JSON proof path; omitted writes JSON to stdout")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.self_check:
            if any((args.baseline_merged, args.baseline_directory, args.candidate_merged, args.candidate_directory, args.output)) or args.science or args.report_nonconformant:
                fail("--self-check cannot be combined with comparison arguments")
            self_check()
            print("jinc comparator self-check passed")
            return 0
        record = run_check(args)
        payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            path = args.output.expanduser().resolve(strict=False)
            if path.exists() or path.is_symlink():
                fail(f"refusing to overwrite proof: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            print(path)
        else:
            print(payload, end="")
        return 0 if record["status"] == "pass" else 1
    except CheckError as exc:
        print(f"jinc comparator check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
