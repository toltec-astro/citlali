from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from tools.fruit_loops import edit_restart_checkpoint_penalty as edit


def make_checkpoint(path: Path) -> None:
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("effective_detector_penalty_count_dim", 1)
        dataset.createDimension("effective_detector_penalty", 3)
        dataset.createDimension("other", 2)
        count = dataset.createVariable(
            "effective_detector_penalty_count",
            "i8",
            ("effective_detector_penalty_count_dim",),
        )
        count[:] = [3]
        records = {
            "penalty_producer": (str, ["p0", "target", "p2"]),
            "penalty_reason": (str, ["r0", "reason", "r2"]),
            "penalty_iteration": ("i4", [1, 4, 3]),
            "penalty_scan": ("i4", [0, 5, 2]),
            "penalty_uid": ("i4", [10, 4460, 30]),
            "penalty_network": ("i4", [1, -1, 3]),
            "penalty_array": ("i4", [0, 1, 2]),
            "penalty_factor": ("f8", [0.5, 0.0, 0.25]),
            "penalty_score": ("f8", [2.0, 4.0, 6.0]),
            "penalty_event_time_unix_sec": ("f8", [10.0, 20.0, 30.0]),
            "penalty_scan_local": ("i4", [0, 1, 1]),
        }
        for name, (datatype, values) in records.items():
            variable = dataset.createVariable(
                name, datatype, ("effective_detector_penalty",)
            )
            if datatype is str:
                for index, value in enumerate(values):
                    variable[index] = value
            else:
                variable[:] = values
        other = dataset.createVariable("unrelated", "f8", ("other",))
        other[:] = [np.nan, 42.0]


def selector() -> dict:
    return {
        "producer": "target",
        "reason": "reason",
        "iteration": 4,
        "scan": 5,
        "uid": 4460,
        "network": -1,
        "array": 1,
        "factor": 0.0,
        "score": 4.0,
        "scan_local": True,
    }


def test_transform_removes_only_exact_penalty(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    output = tmp_path / "output.nc"
    make_checkpoint(source)

    audit = edit.transform_checkpoint(
        source,
        output,
        selector(),
        edit.sha256(source),
    )

    assert audit["transformation"]["removed_index"] == 1
    assert audit["transformation"]["removed_record"]["uid"] == 4460
    with Dataset(output) as dataset:
        assert len(dataset.dimensions[edit.PENALTY_DIMENSION]) == 2
        assert int(dataset.variables[edit.COUNT_VARIABLE][:].item()) == 2
        assert dataset.variables["penalty_uid"][:].tolist() == [10, 30]
        assert np.array_equal(
            dataset.variables["unrelated"][:],
            np.array([np.nan, 42.0]),
            equal_nan=True,
        )


def test_transform_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    make_checkpoint(source)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        edit.transform_checkpoint(
            source,
            tmp_path / "output.nc",
            selector(),
            "0" * 64,
        )


def test_transform_requires_one_match(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    make_checkpoint(source)
    missing = selector()
    missing["uid"] = 9999

    with pytest.raises(ValueError, match="expected one matching penalty"):
        edit.transform_checkpoint(
            source,
            tmp_path / "output.nc",
            missing,
            edit.sha256(source),
        )
