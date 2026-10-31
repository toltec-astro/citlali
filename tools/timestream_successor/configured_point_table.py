"""One-way POINT measurement table for the established TolTECA resolver.

The table contains signed tangent-azimuth/elevation displacements, not pointing
corrections. TolTECA composes User/Paddle offsets and negates the measurement.
Only the existing all-array POINT guard can admit measurements here. Unavailable
uncertainties stay NaN; conditional source-selection sigma_A is never amp_err.
"""

from pathlib import Path
import re

import numpy as np
from astropy.table import Table

from inspect_forward_map import digest, require
from point_additive_diagnostics import save_json

ARRAYS = ("a1100", "a1400", "a2000")
POINT_MODEL_KEYS = tuple(
    "Header.PointModel." + name for name in ("AzUserOff", "AzPaddleOff", "ElUserOff", "ElPaddleOff")
)


def _finite_scalar(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _metadata_cause(metadata, observation):
    if metadata is None:
        return "POINT observation metadata was not supplied"
    if metadata.get("status") != "available":
        return "POINT observation metadata unavailable: " + str(metadata.get("cause"))
    meta = metadata.get("meta", {})
    if type(meta.get("obsnum")) is not int or meta["obsnum"] != observation:
        return "POINT metadata observation differs from the published maps"
    if any(
        not isinstance(meta.get(key), str) or not meta[key].strip() for key in ("source", "date")
    ):
        return "POINT source or observation date is unavailable"
    if not _finite_scalar(meta.get("mjd")) or meta["mjd"] <= 0:
        return "POINT mean observation MJD is unavailable"
    if any(not _finite_scalar(meta.get(key)) for key in POINT_MODEL_KEYS):
        return "POINT User/Paddle metadata is unavailable"
    telescope = metadata.get("telescope", {})
    if (
        not isinstance(telescope.get("path"), str)
        or not telescope["path"]
        or not isinstance(telescope.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", telescope["sha256"]) is None
    ):
        return "POINT telescope input identity is unavailable"
    offsets = metadata.get("applied_pointing_offsets", {})
    for axis in ("az", "alt"):
        values = offsets.get(axis)
        if (
            not isinstance(values, (list, tuple))
            or len(values) not in (1, 2)
            or any(not _finite_scalar(value) for value in values)
        ):
            return "POINT applied correction context is unavailable"
        if any(value != 0 for value in values):
            return "POINT table withheld: nonzero applied corrections require a qualified residual-to-measurement adapter"
    return None


def publish_pointing_table(
    destination: Path,
    observation: int,
    guards: dict[str, dict],
    records: list[dict],
    pointing_metadata: dict | None,
) -> dict:
    """Publish one complete astrometry table or an explicit withholding record.

    ``guards`` are the final-map measurements already made by the product owner;
    this adapter performs no fit, correction, new scientific admission or error
    estimation. File/software failures propagate to the staged publication owner.
    A nonzero correction applied during POINT is unsupported for this handoff:
    publishing that residual as an uncorrected measurement would double-correct.
    """
    destination = Path(destination)
    require(type(observation) is int and observation > 0, "invalid POINT table observation")
    require(destination.is_dir(), "POINT table destination must already exist")
    result = dict(
        schema="citlali-configured-point-table-v1",
        observation=observation,
        status="withheld",
        cause=None,
        table=None,
        sha256=None,
        pointing_eligible=False,
        calibration_eligible=False,
        uncertainty_status="unavailable",
        uncertainty_scope="conditional source-selection sigma_A is not a calibration uncertainty",
        frame="delta-source-AltAz",
        measurement_axes=["tangent_azimuth", "elevation"],
        measurement_units="arcsec",
        measurements_are_corrections=False,
        source_products=[
            {key: row[key] for key in ("array", "map", "sha256", "native_product_identity")}
            for row in records
        ],
    )
    cause = _metadata_cause(pointing_metadata, observation)
    if cause is None and (
        len(records) != len(ARRAYS)
        or {row["array"] for row in records} != set(ARRAYS)
        or set(guards) != set(ARRAYS)
    ):
        cause = "POINT table requires final fit measurements for all three arrays"
    parameters = {}
    if cause is None:
        for array in ARRAYS:
            guard = guards[array]
            if (
                guard.get("accepted") is not True
                or guard.get("status") != "accepted"
                or guard.get("complete") is not True
            ):
                cause = "POINT fit is not qualified for " + array + ": " + str(guard.get("cause"))
                break
            values = np.asarray(guard.get("accepted_parameters"), dtype=float)
            centroid = np.asarray(guard.get("accepted_centroid_tangent_arcsec"), dtype=float)
            require(
                values.shape == (7,)
                and np.isfinite(values).all()
                and centroid.shape == (2,)
                and np.array_equal(centroid, values[1:3]),
                "accepted POINT fit has malformed or inconsistent measurements: " + array,
            )
            parameters[array] = values
    if cause is not None:
        result["cause"] = cause
        save_json(destination / "point-table.json", result)
        return result

    # _metadata_cause checked the complete observation/header association above.
    meta = dict(pointing_metadata["meta"])
    meta.update(
        schema="citlali-configured-point-astrometry-table-v1",
        Radesys="altaz",
        array_order=[f"{index}: {array}" for index, array in enumerate(ARRAYS)],
        pointing_eligible=True,
        calibration_eligible=False,
        uncertainty_status="unavailable",
        uncertainty_scope=result["uncertainty_scope"],
        signal_meaning="mJy per nominal beam; not a measured beam or full response",
        measurements_are_corrections=False,
        applied_pointing_offsets=pointing_metadata["applied_pointing_offsets"],
        telescope_input=pointing_metadata["telescope"],
        source_products=result["source_products"],
    )
    columns = dict(
        array=np.arange(len(ARRAYS), dtype=np.int64),
        amp=[parameters[array][0] for array in ARRAYS],
        amp_err=np.full(len(ARRAYS), np.nan),
        x_t=[parameters[array][1] for array in ARRAYS],
        x_t_err=np.full(len(ARRAYS), np.nan),
        y_t=[parameters[array][2] for array in ARRAYS],
        y_t_err=np.full(len(ARRAYS), np.nan),
    )
    table = Table(columns, meta=meta)
    for name in ("amp", "amp_err"):
        table[name].unit = "mJy/beam"
    for name in ("x_t", "x_t_err", "y_t", "y_t_err"):
        table[name].unit = "arcsec"
    name = f"ppt_successor_pointing_{observation}_citlali.ecsv"
    path = destination / name
    table.write(path, format="ascii.ecsv", overwrite=False)
    result.update(status="published", table=name, sha256=digest(path), pointing_eligible=True)
    save_json(destination / "point-table.json", result)
    return result
