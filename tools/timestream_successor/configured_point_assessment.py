"""Observation-local facts for future POINT rule evaluation.

The publication owner supplies its already inspected maps and final guards.
This adapter performs no fitting, map reading, selection or correction. Accepted
parameters are the existing Gaussian-plus-constant vector; cx/cy are measured
tangent-azimuth/elevation displacements in the source-offset AltAz frame.
Failed/unavailable fits have no accepted parameters. Conditional sigma_A/SNR_A
retain their existing guard meaning and are not positional/calibration errors.
"""

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from inspect_forward_map import require
from point_additive_diagnostics import clean


ARRAYS = ("a1100", "a1400", "a2000")
ASSESSMENT_FILE = "point-assessment.json"
SCHEMA = "citlali-configured-point-assessment-v1"
PARAMETER_NAMES = ("amplitude", "cx", "cy", "sigma_x", "sigma_y", "angle", "constant")
PARAMETER_UNITS = (
    "mJy/nominal beam", "arcsec", "arcsec", "arcsec", "arcsec", "rad", "mJy/nominal beam",
)
SOURCE_KEYS = (
    "map", "sha256", "native_bundle_sha256", "native_product_identity",
    "nominal_beam_fwhm_arcsec", "coefficient_family", "native_cycle_index",
    "frozen_health_identity", "native_plan_sha256",
)
GUARD_KEYS = (
    "method", "status", "accepted", "complete", "cause", "accepted_parameters",
    "accepted_centroid_tangent_arcsec", "sigma_A", "SNR_A",
)


def _finite_vector(values, size):
    return (
        isinstance(values, (list, tuple)) and len(values) == size
        and all(isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value) for value in values)
    )


def _fit_facts(guard):
    status = guard.get("status")
    require(status in ("accepted", "failed", "unavailable"), "unknown POINT fit status")
    require(
        guard.get("accepted") is (status == "accepted")
        and guard.get("complete") is (status != "unavailable")
        and isinstance(guard.get("method"), str) and bool(guard["method"]),
        "inconsistent POINT fit state or missing method identity",
    )
    values = guard.get("accepted_parameters")
    centroid = guard.get("accepted_centroid_tangent_arcsec")
    if status == "accepted":
        require(
            _finite_vector(values, 7) and _finite_vector(centroid, 2)
            and list(centroid) == list(values[1:3]) and guard.get("cause") is None,
            "accepted POINT fit has malformed or inconsistent measurements",
        )
    else:
        require(
            values is None and centroid is None
            and isinstance(guard.get("cause"), str) and bool(guard["cause"]),
            "unaccepted POINT fit has accepted measurements or missing cause",
        )
    # Match existing JSON diagnostics: nonfinite diagnostic scalars become null.
    # Accepted measurement vectors were checked above, before this conversion.
    return clean({key: deepcopy(guard.get(key)) for key in GUARD_KEYS})


def build_assessment(observation: int, records: list[dict], guards: dict[str, dict],
                     table: dict) -> dict:
    """Collect facts in canonical array order; inputs remain owned by the caller.

    A record absent from this publication is 'not_produced', not declared offline.
    A produced map without a guard is 'not_requested' only when its product says
    so. Unexpected missing/mismatched guards fail the required publication.
    The existing table's eligibility is separate from future rule evaluation.
    """
    require(type(observation) is int and observation > 0, "invalid POINT observation")
    by_array = {record["array"]: record for record in records}
    require(
        bool(records) and len(by_array) == len(records) and set(by_array) <= set(ARRAYS)
        and set(guards) <= set(by_array), "inconsistent POINT assessment array domain",
    )
    require(table.get("observation") == observation, "foreign POINT table assessment")
    arrays = []
    for array in ARRAYS:
        record = by_array.get(array)
        item = dict(array=array, map_status="not_produced", source=None,
                    fit_status="not_produced", fit_diagnostic=None, fit_guard=None)
        if record is not None:
            require(
                record.get("observation") == observation
                and record.get("frame") == "delta-source-AltAz"
                and record.get("axis_units") == "arcsec",
                "POINT assessment observation or measurement frame differs",
            )
            item.update(map_status="published", source={
                key: deepcopy(record[key]) for key in SOURCE_KEYS if key in record
            })
            guard = guards.get(array)
            if guard is None:
                require(record.get("fit_status") == "not_requested" and not record.get("fit"),
                        "missing required POINT fit assessment")
                item["fit_status"] = "not_requested"
            else:
                fit = _fit_facts(guard)
                require(record.get("fit_status") == fit["status"] and bool(record.get("fit")),
                        "POINT fit/product assessment differs")
                item.update(fit_status=fit["status"], fit_diagnostic=record["fit"], fit_guard=fit)
        arrays.append(item)
    return dict(
        schema=SCHEMA, observation=observation,
        frame="delta-source-AltAz", measurement_axes=["tangent_azimuth", "elevation"],
        measurement_units="arcsec", measurements_are_corrections=False,
        parameter_names=list(PARAMETER_NAMES), parameter_units=list(PARAMETER_UNITS),
        positional_uncertainty_status="unavailable",
        conditional_amplitude_uncertainty_scope=(
            "full frozen C0 reference-scale sigma_A in mJy/nominal beam and dimensionless SNR_A; "
            "not endpoint empirical uncertainty or full adaptive error"
        ),
        arrays=arrays,
        rule_evaluation=dict(status="not_evaluated", cause="rule implementation pending",
                             policy_identity=None, selected_arrays=[],
                             adopted_displacement_arcsec=None),
        existing_table=deepcopy(table), calibration_eligible=False,
    )


def publish_assessment(destination: Path, observation: int, records: list[dict],
                       guards: dict[str, dict], table: dict) -> dict:
    """Write one required staged report and bind exactly its serialized bytes.

    The caller owns final observation publication. I/O errors propagate and the
    existing staged rename cannot run after this write fails. No input rereads.
    """
    report = build_assessment(observation, records, guards, table)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with (Path(destination) / ASSESSMENT_FILE).open("xb") as stream:
        stream.write(payload)
    return dict(file=ASSESSMENT_FILE, schema=SCHEMA, sha256=hashlib.sha256(payload).hexdigest())
