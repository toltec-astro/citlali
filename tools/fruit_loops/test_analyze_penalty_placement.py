from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from netCDF4 import Dataset

from tools.fruit_loops import analyze_penalty_placement as analysis


def test_learning_policy_normalization_adds_legacy_default() -> None:
    old = np.asarray(["enabled: true\n"], dtype=object)
    new = np.asarray(
        [
            "enabled: true\n"
            "map_pixel_outlier_detector_exclusion_application: pre_cleaning\n"
        ],
        dtype=object,
    )

    assert analysis.normalized_learning_policy(old) == (
        analysis.normalized_learning_policy(new)
    )


def test_checkpoint_compatibility_allows_only_normalized_provenance(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.nc"
    actual = tmp_path / "actual.nc"
    for path, creator, policy in (
        (expected, "old", "enabled: true\n"),
        (
            actual,
            "new",
            "enabled: true\n"
            "map_pixel_outlier_detector_exclusion_application: pre_cleaning\n",
        ),
    ):
        with Dataset(path, "w") as dataset:
            dataset.createDimension("one", 1)
            dataset.createVariable("scientific", "f8", ("one",))[:] = [2.0]
            dataset.createVariable("creator_version", str, ("one",))[0] = creator
            dataset.createVariable("learning_policy_yaml", str, ("one",))[0] = (
                policy
            )

    result = analysis.require_compatible_checkpoint(
        expected,
        actual,
        {"creator_version", "learning_policy_yaml"},
    )

    assert result["scientific_values_identical"] is True
    assert result["observed_allowed_differences"] == [
        "creator_version",
        "learning_policy_yaml",
    ]


def test_checkpoint_compatibility_rejects_scientific_change(tmp_path: Path) -> None:
    expected = tmp_path / "expected.nc"
    actual = tmp_path / "actual.nc"
    for path, value in ((expected, 2.0), (actual, 3.0)):
        with Dataset(path, "w") as dataset:
            dataset.createDimension("one", 1)
            dataset.createVariable("scientific", "f8", ("one",))[:] = [value]

    try:
        analysis.require_compatible_checkpoint(expected, actual, set())
    except ValueError as error:
        assert "scientific checkpoint value differs" in str(error)
    else:
        raise AssertionError("scientific checkpoint change was accepted")


def test_normalized_placement_pair_removes_only_registered_fields() -> None:
    current = {
        "runtime": {"output_dir": "/current"},
        "timestream": {
            "fruit_loops": {"restart_path": "/current/restart", "max_iters": 6},
            "learning": {
                "map_pixel_outlier_detector_exclusion_application": (
                    "pre_cleaning"
                ),
                "enabled": True,
            },
        },
    }
    moved = yaml.safe_load(yaml.safe_dump(current))
    moved["runtime"]["output_dir"] = "/map"
    moved["timestream"]["fruit_loops"]["restart_path"] = "/map/restart"
    moved["timestream"]["learning"][
        "map_pixel_outlier_detector_exclusion_application"
    ] = "pre_mapmaking"

    assert analysis.normalized_placement_pair(current) == (
        analysis.normalized_placement_pair(moved)
    )


def test_q_identity_closes_exactly() -> None:
    current = np.arange(9, dtype=float).reshape(3, 3)
    moved = current + 0.25
    no_uid = current - 1.5
    d_current = current - no_uid
    d_map = moved - no_uid
    q = current - moved

    assert np.array_equal(q, d_current - d_map)
