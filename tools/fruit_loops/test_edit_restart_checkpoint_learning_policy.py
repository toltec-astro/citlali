from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from netCDF4 import Dataset

from tools.fruit_loops import edit_restart_checkpoint_learning_policy as edit


FIELD = "map_pixel_outlier_detector_exclusion_application"


def make_checkpoint(path: Path, placement: str | None = None) -> None:
    policy = {"enabled": True, "learn_iters": 2}
    if placement is not None:
        policy[FIELD] = placement
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncattr("identity", "test")
        dataset.createDimension("one", 1)
        dataset.createDimension("other", 2)
        learning = dataset.createVariable("learning_policy_yaml", str, ("one",))
        learning[0] = yaml.safe_dump(policy, sort_keys=False).rstrip()
        unrelated = dataset.createVariable("unrelated", "f8", ("other",))
        unrelated[:] = [np.nan, 42.0]


def transform(source: Path, output: Path) -> dict:
    return edit.transform_checkpoint(
        source,
        output,
        edit.sha256(source),
        FIELD,
        "pre_cleaning",
        "pre_mapmaking",
    )


def test_transform_changes_only_registered_legacy_default(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    output = tmp_path / "output.nc"
    make_checkpoint(source)

    audit = transform(source, output)

    assert audit["transformation"]["source_value"] == "pre_cleaning"
    assert audit["transformation"]["output_value"] == "pre_mapmaking"
    with Dataset(output) as dataset:
        policy = edit.read_policy(dataset.variables[edit.POLICY_VARIABLE][...])
        assert policy[FIELD] == "pre_mapmaking"
        assert policy["enabled"] is True
        assert np.array_equal(
            dataset.variables["unrelated"][:],
            np.array([np.nan, 42.0]),
            equal_nan=True,
        )


def test_transform_accepts_explicit_legacy_value(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    make_checkpoint(source, "pre_cleaning")
    audit = transform(source, tmp_path / "output.nc")
    assert audit["transformation"]["all_other_values_verified_equal"]


def test_transform_rejects_wrong_value_hash_and_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    make_checkpoint(source, "pre_mapmaking")
    with pytest.raises(ValueError, match="expected 'pre_cleaning'"):
        transform(source, tmp_path / "wrong.nc")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        edit.transform_checkpoint(
            source,
            tmp_path / "hash.nc",
            "0" * 64,
            FIELD,
            "pre_cleaning",
            "pre_mapmaking",
        )
    existing = tmp_path / "existing.nc"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transform(source, existing)
