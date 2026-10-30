"""Synthetic contract checks for the POINT-to-TolTECA measurement adapter."""

from copy import deepcopy
import json

import numpy as np
import pytest
from astropy.table import Table

import configured_point_products as products
from configured_point_table import ARRAYS, POINT_MODEL_KEYS, publish_pointing_table
from inspect_forward_map import digest
from test_configured_point_products import native_map


def metadata(observation=152389):
    return dict(
        status="available",
        cause=None,
        meta=dict(
            obsnum=observation,
            source="synthetic POINT",
            date="2026-02-19T01:02:03.456",
            mjd=61090.123456789,
            **dict(zip(POINT_MODEL_KEYS, (1e-5, -2e-5, 3e-5, -4e-5))),
        ),
        applied_pointing_offsets=dict(az=[0.0], alt=[0.0]),
        telescope=dict(path="/synthetic/telescope.nc", sha256="a" * 64),
    )


def measurements():
    return {
        array: dict(
            accepted=True,
            status="accepted",
            complete=True,
            cause=None,
            accepted_parameters=[1000.0 + index, 2.0 + index, -3.0 - index, 4.0, 5.0, 0.2, 6.0],
            accepted_centroid_tangent_arcsec=[2.0 + index, -3.0 - index],
            sigma_A=17.0 + index,
        )
        for index, array in enumerate(ARRAYS)
    }


def records():
    return [
        dict(array=array, map=array + ".fits", sha256=str(index) * 64,
             native_product_identity="synthetic-native:" + array)
        for index, array in enumerate(ARRAYS)
    ]


def test_table_is_complete_signed_measurement_with_explicit_unavailable_errors(tmp_path):
    headers, guards = metadata(), measurements()
    before = deepcopy((headers, guards))
    result = publish_pointing_table(tmp_path, 152389, guards, records()[::-1], headers)
    assert result["status"] == "published"
    assert result["pointing_eligible"] and not result["calibration_eligible"]
    path = tmp_path / result["table"]
    assert path.name == "ppt_successor_pointing_152389_citlali.ecsv"
    assert result["sha256"] == digest(path)
    table = Table.read(path, format="ascii.ecsv")
    assert table["array"].dtype.kind == "i"
    assert list(table["array"]) == [0, 1, 2]
    assert list(table["amp"]) == [1000.0, 1001.0, 1002.0]
    assert list(table["x_t"]) == [2.0, 3.0, 4.0]
    assert list(table["y_t"]) == [-3.0, -4.0, -5.0]
    assert str(table["amp"].unit) == str(table["amp_err"].unit) == "mJy / beam"
    assert str(table["x_t"].unit) == str(table["y_t"].unit) == "arcsec"
    for name in ("amp_err", "x_t_err", "y_t_err"):
        assert np.isnan(table[name]).all()
    for key, value in headers["meta"].items():
        assert table.meta[key] == value
    assert table.meta["array_order"] == ["0: a1100", "1: a1400", "2: a2000"]
    assert table.meta["uncertainty_status"] == "unavailable"
    assert table.meta["measurements_are_corrections"] is False
    assert table.meta["pointing_eligible"] is True
    assert table.meta["calibration_eligible"] is False
    assert table.meta["telescope_input"] == headers["telescope"]
    assert json.loads((tmp_path / "point-table.json").read_text()) == result
    assert (headers, guards) == before


@pytest.mark.parametrize("which", ["missing", "unavailable", "obsnum", "source", "date", "mjd",
                                   "header", "binding", "offset-context", "az", "alt"])
def test_metadata_or_applied_correction_withholds_table(tmp_path, which):
    headers = metadata()
    if which == "missing":
        headers = None
    elif which == "unavailable":
        headers.update(status="unavailable", cause="missing telescope header")
    elif which == "obsnum":
        headers["meta"]["obsnum"] += 1
    elif which in ("source", "date"):
        headers["meta"][which] = ""
    elif which == "mjd":
        headers["meta"]["mjd"] = float("nan")
    elif which == "header":
        del headers["meta"][POINT_MODEL_KEYS[0]]
    elif which == "binding":
        headers["telescope"]["sha256"] = "not a digest"
    elif which == "offset-context":
        del headers["applied_pointing_offsets"]
    else:
        headers["applied_pointing_offsets"][which] = [0.0, -0.01]
    result = publish_pointing_table(tmp_path, 152389, measurements(), records(), headers)
    assert result["status"] == "withheld"
    assert result["cause"]
    assert not result["pointing_eligible"] and not result["calibration_eligible"]
    assert not list(tmp_path.glob("ppt_*.ecsv"))
    assert json.loads((tmp_path / "point-table.json").read_text()) == result


@pytest.mark.parametrize("which", ["missing", "failed", "unavailable", "incomplete", "not-accepted",
                                   "duplicate-array"])
def test_no_partial_array_or_unqualified_measurement_can_be_discovered(tmp_path, which):
    guards, rows = measurements(), records()
    if which == "missing":
        del guards["a2000"]
    elif which == "duplicate-array":
        rows[2] = rows[1]
    else:
        guard = guards["a2000"]
        if which == "incomplete":
            guard["complete"] = False
        elif which == "not-accepted":
            guard["accepted"] = False
        else:
            guard.update(status=which, accepted=False, cause="synthetic unavailable source")
    result = publish_pointing_table(tmp_path, 152389, guards, rows, metadata())
    assert result["status"] == "withheld"
    assert not list(tmp_path.glob("ppt_*.ecsv"))


@pytest.mark.parametrize("which", ["nonfinite", "shape", "centroid"])
def test_malformed_accepted_measurement_is_a_propagated_contract_failure(tmp_path, which):
    guards = measurements()
    if which == "nonfinite":
        guards["a1400"]["accepted_parameters"][0] = float("nan")
    elif which == "shape":
        guards["a1400"]["accepted_parameters"] = [1.0]
    else:
        guards["a1400"]["accepted_centroid_tangent_arcsec"] = [99.0, 0.0]
    with pytest.raises(ValueError, match="malformed or inconsistent"):
        publish_pointing_table(tmp_path, 152389, guards, records(), metadata())
    assert not list(tmp_path.glob("ppt_*.ecsv"))


def test_preexisting_table_is_never_overwritten(tmp_path):
    path = tmp_path / "ppt_successor_pointing_152389_citlali.ecsv"
    path.write_text("preserved result")
    with pytest.raises(OSError):
        publish_pointing_table(tmp_path, 152389, measurements(), records(), metadata())
    assert path.read_text() == "preserved result"


def test_publication_reuses_its_final_guards_and_links_the_handoff(tmp_path, monkeypatch):
    import point_reference

    inputs, destination = tmp_path / "native", tmp_path / "published"
    originals = {}
    for array in ARRAYS:
        originals[array] = native_map(inputs / array, array)
    guards, calls = measurements(), []

    def measured(path, **kwargs):
        array = next(array for array in ARRAYS if array in path.name)
        calls.append(array)
        return guards[array]

    monkeypatch.setattr(point_reference, "measure_forward", measured)
    rows = products.publish_maps(inputs, destination, 152389, ARRAYS,
                                 fit_diagnostics=True, verbose=False,
                                 pointing_metadata=metadata())
    assert calls == list(ARRAYS)
    table = Table.read(destination / rows[0]["pointing_table"], format="ascii.ecsv")
    assert list(table["x_t"]) == [2.0, 3.0, 4.0]
    for row in rows:
        assert row["pointing_eligible"] is True
        assert row["calibration_eligible"] is False
        assert row["legacy_analysis_compatible"] is False
        assert row["pointing_table_diagnostic"] == "point-table.json"
        assert (destination / row["map"]).read_bytes() == originals[row["array"]].read_bytes()
        assert json.loads((destination / f"point-product-{row['array']}.json").read_text()) == row


def test_fit_disabled_preserves_maps_and_explicitly_withholds_table(tmp_path, monkeypatch):
    import point_reference

    for array in ARRAYS:
        native_map(tmp_path / "native" / array, array)
    monkeypatch.setattr(point_reference, "measure_forward", lambda *a, **k: pytest.fail("new fit"))
    rows = products.publish_maps(tmp_path / "native", tmp_path / "published", 152389, ARRAYS,
                                 fit_diagnostics=False, verbose=False,
                                 pointing_metadata=metadata())
    assert all(row["pointing_table"] is None for row in rows)
    status = json.loads((tmp_path / "published/point-table.json").read_text())
    assert status["status"] == "withheld"
    assert "all three arrays" in status["cause"]
    assert not list((tmp_path / "published").glob("ppt_*"))
