#!/usr/bin/env python3
"""Remove one exact effective detector penalty from a copied checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


PENALTY_DIMENSION = "effective_detector_penalty"
COUNT_VARIABLE = "effective_detector_penalty_count"
SELECTOR_VARIABLES = {
    "producer": "penalty_producer",
    "reason": "penalty_reason",
    "iteration": "penalty_iteration",
    "scan": "penalty_scan",
    "uid": "penalty_uid",
    "network": "penalty_network",
    "array": "penalty_array",
    "factor": "penalty_factor",
    "score": "penalty_score",
    "scan_local": "penalty_scan_local",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_value(value):
    return value.item() if hasattr(value, "item") else value


def values_equal(left, right) -> bool:
    left_array = np.ma.asarray(left)
    right_array = np.ma.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if not np.array_equal(
        np.ma.getmaskarray(left_array),
        np.ma.getmaskarray(right_array),
    ):
        return False
    left_data = np.ma.getdata(left_array)
    right_data = np.ma.getdata(right_array)
    if np.issubdtype(left_data.dtype, np.number):
        return bool(np.array_equal(left_data, right_data, equal_nan=True))
    return bool(np.array_equal(left_data, right_data))


def selector_matches(dataset: Dataset, index: int, selector: dict) -> bool:
    for key, variable_name in SELECTOR_VARIABLES.items():
        actual = scalar_value(dataset.variables[variable_name][index])
        expected = selector[key]
        if key in {"factor", "score"}:
            if not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                return False
        elif key == "scan_local":
            if bool(actual) is not bool(expected):
                return False
        elif actual != expected:
            return False
    return True


def penalty_record(dataset: Dataset, index: int) -> dict:
    result = {
        key: scalar_value(dataset.variables[name][index])
        for key, name in SELECTOR_VARIABLES.items()
    }
    event_time = float(
        dataset.variables["penalty_event_time_unix_sec"][index]
    )
    result["event_time_unix_sec"] = event_time if math.isfinite(event_time) else None
    result["scan_local"] = bool(result["scan_local"])
    return result


def create_variable_like(destination: Dataset, name: str, source_variable):
    datatype = str if source_variable.dtype is str else source_variable.datatype
    attributes = {
        attribute: source_variable.getncattr(attribute)
        for attribute in source_variable.ncattrs()
        if attribute != "_FillValue"
    }
    fill_value = (
        source_variable.getncattr("_FillValue")
        if "_FillValue" in source_variable.ncattrs()
        else None
    )
    kwargs = {}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    if datatype is not str and source_variable.endian() in {"little", "big"}:
        kwargs["endian"] = source_variable.endian()
    variable = destination.createVariable(
        name,
        datatype,
        source_variable.dimensions,
        **kwargs,
    )
    if attributes:
        variable.setncatts(attributes)
    return variable


def write_variable_values(variable, values) -> None:
    if variable.dtype is str:
        array = np.asarray(values, dtype=object)
        for index in np.ndindex(array.shape):
            variable[index] = str(array[index])
        return
    variable[...] = values


def transform_checkpoint(
    source_path: Path,
    output_path: Path,
    selector: dict,
    expected_source_sha256: str,
) -> dict:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    source_digest = sha256(source_path)
    if source_digest != expected_source_sha256:
        raise ValueError(
            "source checkpoint SHA-256 mismatch: "
            f"expected={expected_source_sha256} actual={source_digest}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(source_path) as source:
        if PENALTY_DIMENSION not in source.dimensions:
            raise ValueError(f"missing {PENALTY_DIMENSION} dimension")
        penalty_count = len(source.dimensions[PENALTY_DIMENSION])
        recorded_count = int(np.asarray(source.variables[COUNT_VARIABLE][:]).item())
        if recorded_count != penalty_count:
            raise ValueError(
                f"penalty count {recorded_count} differs from dimension "
                f"length {penalty_count}"
            )
        missing = [
            name
            for name in (*SELECTOR_VARIABLES.values(), "penalty_event_time_unix_sec")
            if name not in source.variables
        ]
        if missing:
            raise ValueError(f"checkpoint is missing penalty variables: {missing}")
        matches = [
            index
            for index in range(penalty_count)
            if selector_matches(source, index, selector)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one matching penalty, found {len(matches)}"
            )
        removed_index = matches[0]
        removed_record = penalty_record(source, removed_index)

        with Dataset(output_path, "w", format=source.file_format) as destination:
            if source.ncattrs():
                destination.setncatts(
                    {
                        attribute: source.getncattr(attribute)
                        for attribute in source.ncattrs()
                    }
                )
            for name, dimension in source.dimensions.items():
                if name == PENALTY_DIMENSION:
                    size = penalty_count - 1
                else:
                    size = None if dimension.isunlimited() else len(dimension)
                destination.createDimension(name, size)

            for name, source_variable in source.variables.items():
                destination_variable = create_variable_like(
                    destination, name, source_variable
                )
                values = source_variable[...]
                if PENALTY_DIMENSION in source_variable.dimensions:
                    axis = source_variable.dimensions.index(PENALTY_DIMENSION)
                    values = np.delete(values, removed_index, axis=axis)
                elif name == COUNT_VARIABLE:
                    values = np.ma.asarray(values).copy()
                    values.reshape(-1)[0] = penalty_count - 1
                write_variable_values(destination_variable, values)

    audit_transformation(source_path, output_path, removed_index)
    return {
        "schema_version": "sci-fruit-checkpoint-penalty-intervention-v1",
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
            "dimension": PENALTY_DIMENSION,
            "source_count": penalty_count,
            "output_count": penalty_count - 1,
            "removed_index": removed_index,
            "removed_record": removed_record,
            "all_other_values_verified_equal": True,
            "all_types_dimensions_and_attributes_verified": True,
        },
    }


def audit_transformation(
    source_path: Path,
    output_path: Path,
    removed_index: int,
) -> None:
    with Dataset(source_path) as source, Dataset(output_path) as output:
        if source.ncattrs() != output.ncattrs():
            raise ValueError("global attribute names changed")
        for attribute in source.ncattrs():
            if source.getncattr(attribute) != output.getncattr(attribute):
                raise ValueError(f"global attribute changed: {attribute}")
        if set(source.dimensions) != set(output.dimensions):
            raise ValueError("dimension names changed")
        for name, source_dimension in source.dimensions.items():
            expected = len(source_dimension) - (name == PENALTY_DIMENSION)
            if len(output.dimensions[name]) != expected:
                raise ValueError(f"dimension changed unexpectedly: {name}")
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
                if source_variable.getncattr(attribute) != output_variable.getncattr(
                    attribute
                ):
                    raise ValueError(
                        f"variable attribute changed: {name}:{attribute}"
                    )

            expected_values = source_variable[...]
            if PENALTY_DIMENSION in source_variable.dimensions:
                axis = source_variable.dimensions.index(PENALTY_DIMENSION)
                expected_values = np.delete(
                    expected_values, removed_index, axis=axis
                )
            elif name == COUNT_VARIABLE:
                expected_values = np.ma.asarray(expected_values).copy()
                expected_values.reshape(-1)[0] -= 1
            if not values_equal(expected_values, output_variable[...]):
                raise ValueError(f"variable values changed unexpectedly: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--scan", required=True, type=int)
    parser.add_argument("--uid", required=True, type=int)
    parser.add_argument("--network", required=True, type=int)
    parser.add_argument("--array", required=True, type=int)
    parser.add_argument("--factor", required=True, type=float)
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--scan-local", required=True, type=int, choices=(0, 1))
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()

    selector = {
        "producer": args.producer,
        "reason": args.reason,
        "iteration": args.iteration,
        "scan": args.scan,
        "uid": args.uid,
        "network": args.network,
        "array": args.array,
        "factor": args.factor,
        "score": args.score,
        "scan_local": bool(args.scan_local),
    }
    audit = transform_checkpoint(
        args.source,
        args.output,
        selector,
        args.expected_source_sha256,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "removed exactly one effective detector penalty; "
        f"wrote {args.output} and {args.audit_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
