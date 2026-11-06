"""Facts-only POINT handoff; Grant owns execution of these regression checks."""

from copy import deepcopy
import json

import pytest

from configured_point_assessment import build_assessment, publish_assessment
from inspect_forward_map import digest
from test_configured_point_table import measurements, records


def inputs():
    rows, guards = records(), measurements()
    for row in rows:
        row.update(observation=152389, frame="delta-source-AltAz", axis_units="arcsec",
                   fit_status="accepted", fit=f"point-fit-{row['array']}.json")
    table = dict(observation=152389, status="published", pointing_eligible=True,
                 table="ppt_successor_pointing_152389_citlali.ecsv", sha256="e" * 64)
    return rows, guards, table


def test_assessment_preserves_measurements_without_selecting_or_mutating():
    rows, guards, table = inputs()
    before = deepcopy((rows, guards, table))
    result = build_assessment(152389, rows[::-1], guards, table)
    assert result["frame"] == "delta-source-AltAz"
    assert result["measurement_axes"] == ["tangent_azimuth", "elevation"]
    assert result["measurement_units"] == "arcsec"
    assert result["measurements_are_corrections"] is False
    assert result["parameter_names"] == ["amplitude", "cx", "cy", "sigma_x", "sigma_y", "angle", "constant"]
    for item, row in zip(result["arrays"], rows, strict=True):
        array = row["array"]
        assert item["array"] == array
        assert item["source"]["sha256"] == row["sha256"]
        assert item["fit_guard"]["accepted_parameters"] == guards[array]["accepted_parameters"]
        assert item["fit_guard"]["accepted_centroid_tangent_arcsec"] == guards[array]["accepted_parameters"][1:3]
        assert item["fit_guard"]["sigma_A"] == guards[array]["sigma_A"]
    assert result["rule_evaluation"] == dict(
        status="not_evaluated", cause="rule implementation pending", policy_identity=None,
        selected_arrays=[], adopted_displacement_arcsec=None,
    )
    assert result["existing_table"]["pointing_eligible"] is True
    assert result["positional_uncertainty_status"] == "unavailable"
    assert result["calibration_eligible"] is False
    result["arrays"][0]["fit_guard"]["accepted_parameters"][1] = 999
    result["existing_table"]["pointing_eligible"] = False
    assert (rows, guards, table) == before


def test_assessment_keeps_absent_maps_and_unrequested_fits_distinct():
    rows, _, table = inputs()
    row = rows[-1]
    row.update(fit_status="not_requested")
    del row["fit"]
    table.update(status="withheld", pointing_eligible=False, table=None, sha256=None)
    result = build_assessment(152389, [row], {}, table)
    assert [item["map_status"] for item in result["arrays"]] == ["not_produced", "not_produced", "published"]
    assert [item["fit_status"] for item in result["arrays"]] == ["not_produced", "not_produced", "not_requested"]
    assert all(item["fit_guard"] is None for item in result["arrays"])


@pytest.mark.parametrize("status,complete", [("failed", True), ("unavailable", False)])
def test_assessment_preserves_failed_and_unavailable_without_resurrecting_fit(status, complete):
    rows, guards, table = inputs()
    guard = guards["a2000"]
    guard.update(status=status, complete=complete, accepted=False, cause="original fit cause",
                 accepted_parameters=None, accepted_centroid_tangent_arcsec=None,
                 diagnostic=dict(parameters=[1, 2, 3, 4, 5, 6, 7]), SNR_A=float("nan"))
    rows[-1]["fit_status"] = status
    result = build_assessment(152389, rows, guards, table)
    actual = result["arrays"][-1]["fit_guard"]
    assert actual["status"] == status and actual["complete"] is complete
    assert actual["cause"] == "original fit cause"
    assert actual["accepted_parameters"] is None
    assert actual["accepted_centroid_tangent_arcsec"] is None
    assert actual["SNR_A"] is None
    assert "diagnostic" not in actual


@pytest.mark.parametrize("defect", ["nan", "bool", "shape", "centroid", "state", "method", "missing", "foreign", "frame", "duplicate"])
def test_assessment_contract_failure_propagates_before_output(tmp_path, defect):
    rows, guards, table = inputs()
    guard = guards["a1100"]
    if defect == "nan":
        guard["accepted_parameters"][0] = float("nan")
    elif defect == "bool":
        guard["accepted_parameters"][0] = True
    elif defect == "shape":
        guard["accepted_parameters"] = [1.0]
    elif defect == "centroid":
        guard["accepted_centroid_tangent_arcsec"][0] = 99.0
    elif defect == "state":
        guard["complete"] = False
    elif defect == "method":
        del guard["method"]
    elif defect == "missing":
        del guards["a1100"]
    elif defect == "foreign":
        rows[0]["observation"] += 1
    elif defect == "frame":
        rows[0]["frame"] = "ICRS"
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        publish_assessment(tmp_path, 152389, rows, guards, table)
    assert not list(tmp_path.iterdir())


def test_assessment_binding_covers_written_bytes_and_never_overwrites(tmp_path):
    rows, guards, table = inputs()
    binding = publish_assessment(tmp_path, 152389, rows, guards, table)
    path = tmp_path / binding["file"]
    before = path.read_bytes()
    assert binding["sha256"] == digest(path)
    assert json.loads(before) == build_assessment(152389, rows, guards, table)
    with pytest.raises(FileExistsError):
        publish_assessment(tmp_path, 152389, rows, guards, table)
    assert path.read_bytes() == before
