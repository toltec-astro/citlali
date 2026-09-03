#!/usr/bin/env python3
"""Change one exact learning-policy field in a copied restart checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from netCDF4 import Dataset

from tools.fruit_loops.edit_restart_checkpoint_penalty import (
    create_variable_like,
    sha256,
    values_equal,
    write_variable_values,
)


POLICY_VARIABLE = "learning_policy_yaml"
LEGACY_DEFAULTS = {
    "map_pixel_outlier_detector_exclusion_application": "pre_cleaning",
}


def scalar_text(value: object) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def read_policy(value: object) -> dict[str, object]:
    policy = yaml.safe_load(scalar_text(value))
    if not isinstance(policy, dict):
        raise ValueError("checkpoint learning policy is not a mapping")
    return policy


def normalized_policy(policy: dict[str, object]) -> dict[str, object]:
    result = dict(policy)
    for key, value in LEGACY_DEFAULTS.items():
        result.setdefault(key, value)
    return result


def audit_transformation(
    source_path: Path,
    output_path: Path,
    key: str,
    expected_value: str,
    replacement_value: str,
) -> None:
    with Dataset(source_path) as source, Dataset(output_path) as output:
        if source.ncattrs() != output.ncattrs():
            raise ValueError("global attribute names changed")
        for attribute in source.ncattrs():
            if source.getncattr(attribute) != output.getncattr(attribute):
                raise ValueError(f"global attribute changed: {attribute}")
        if set(source.dimensions) != set(output.dimensions):
            raise ValueError("dimension names changed")
        for name, dimension in source.dimensions.items():
            if len(dimension) != len(output.dimensions[name]):
                raise ValueError(f"dimension changed: {name}")
        if set(source.variables) != set(output.variables):
            raise ValueError("variable names changed")

        for name, source_variable in source.variables.items():
            output_variable = output.variables[name]
            if source_variable.dimensions != output_variable.dimensions:
                raise ValueError(f"variable dimensions changed: {name}")
            if source_variable.dtype != output_variable.dtype:
                raise ValueError(f"variable type changed: {name}")
            if source_variable.ncattrs() != output_variable.ncattrs():
                raise ValueError(f"variable attribute names changed: {name}")
            for attribute in source_variable.ncattrs():
                if source_variable.getncattr(
                    attribute
                ) != output_variable.getncattr(attribute):
                    raise ValueError(
                        f"variable attribute changed: {name}:{attribute}"
                    )
            if name == POLICY_VARIABLE:
                continue
            if not values_equal(source_variable[...], output_variable[...]):
                raise ValueError(f"unregistered variable changed: {name}")

        source_policy = normalized_policy(
            read_policy(source.variables[POLICY_VARIABLE][...])
        )
        output_policy = normalized_policy(
            read_policy(output.variables[POLICY_VARIABLE][...])
        )
        if source_policy.get(key) != expected_value:
            raise ValueError("source policy no longer has the expected value")
        expected_output = dict(source_policy)
        expected_output[key] = replacement_value
        if output_policy != expected_output:
            raise ValueError("learning policy changed beyond the registered field")


def transform_checkpoint(
    source_path: Path,
    output_path: Path,
    expected_source_sha256: str,
    key: str,
    expected_value: str,
    replacement_value: str,
) -> dict:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("source and output checkpoint paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    source_digest = sha256(source_path)
    if source_digest != expected_source_sha256:
        raise ValueError(
            "source checkpoint SHA-256 mismatch: "
            f"expected={expected_source_sha256} actual={source_digest}"
        )
    if key not in LEGACY_DEFAULTS:
        raise ValueError(f"field is not approved for legacy normalization: {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(source_path) as source:
        if POLICY_VARIABLE not in source.variables:
            raise ValueError(f"checkpoint is missing {POLICY_VARIABLE}")
        source_policy = read_policy(source.variables[POLICY_VARIABLE][...])
        actual_value = source_policy.get(key, LEGACY_DEFAULTS[key])
        if actual_value != expected_value:
            raise ValueError(
                f"source policy {key} is {actual_value!r}, "
                f"expected {expected_value!r}"
            )
        source_policy[key] = replacement_value
        replacement_yaml = yaml.safe_dump(
            source_policy, sort_keys=False
        ).rstrip()

        with Dataset(output_path, "w", format=source.file_format) as output:
            if source.ncattrs():
                output.setncatts(
                    {
                        attribute: source.getncattr(attribute)
                        for attribute in source.ncattrs()
                    }
                )
            for name, dimension in source.dimensions.items():
                size = None if dimension.isunlimited() else len(dimension)
                output.createDimension(name, size)
            for name, source_variable in source.variables.items():
                output_variable = create_variable_like(
                    output, name, source_variable
                )
                values = source_variable[...]
                if name == POLICY_VARIABLE:
                    values = np.asarray(values, dtype=object).copy()
                    values.reshape(-1)[0] = replacement_yaml
                write_variable_values(output_variable, values)

    audit_transformation(
        source_path,
        output_path,
        key,
        expected_value,
        replacement_value,
    )
    return {
        "schema_version": "sci-fruit-checkpoint-learning-policy-intervention-v1",
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": source_digest,
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "transformation": {
            "variable": POLICY_VARIABLE,
            "field": key,
            "source_value": expected_value,
            "output_value": replacement_value,
            "legacy_default_was_allowed": key in LEGACY_DEFAULTS,
            "all_other_values_verified_equal": True,
            "all_types_dimensions_and_attributes_verified": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--from-value", required=True)
    parser.add_argument("--to-value", required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    audit = transform_checkpoint(
        args.source,
        args.output,
        args.source_sha256,
        args.field,
        args.from_value,
        args.to_value,
    )
    if args.audit_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.audit_json}")
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(args.audit_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
