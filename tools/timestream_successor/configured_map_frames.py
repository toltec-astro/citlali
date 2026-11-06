"""Explicit persisted frames and the existing tangent-learner unit adapter.

AST computes sample coordinates. These helpers only describe/validate the
published WCS and convert POINT arcsecond increments for the existing learner.
Celestial coordinates remain FK5/J2000 TAN in degrees, without reprojection.
"""

import math

POINT_FRAME = dict(ASTFRAME="delta-source-AltAz", CTYPE1="AZOFFSET", CTYPE2="ELOFFSET",
                   CUNIT1="arcsec", CUNIT2="arcsec")
SCIENCE_FRAME = dict(ASTFRAME="J2000-radec-gnomonic-tangent", CTYPE1="RA---TAN",
                     CTYPE2="DEC--TAN", CUNIT1="deg", CUNIT2="deg",
                     RADESYS="FK5", EQUINOX=2000.0)
NUMERIC_GRID_KEYS = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "NOMBFWHM")


def is_celestial(header):
    return header.get("ASTFRAME") == SCIENCE_FRAME["ASTFRAME"]


def frame_spec(header):
    return SCIENCE_FRAME if is_celestial(header) else POINT_FRAME


def grid_keys(header, *, array=False):
    return tuple(frame_spec(header)) + NUMERIC_GRID_KEYS + (("ARRAY",) if array else ())


def validate_frame(header, *, celestial=None):
    science = is_celestial(header)
    if celestial is not None and science != celestial:
        raise ValueError("configured MAP frame does not match requested reduction mode")
    if any(header.get(k) != v for k, v in frame_spec(header).items()):
        raise ValueError("configured MAP frame, axis units or reference system differs")
    if not all(math.isfinite(header[k]) for k in NUMERIC_GRID_KEYS):
        raise ValueError("configured MAP grid and nominal beam must be finite")
    if not (header["CDELT1"] < 0 < header["CDELT2"] and header["NOMBFWHM"] > 0):
        raise ValueError("configured MAP axes or nominal beam differs")
    if science:
        if not -90 <= header["CRVAL2"] <= 90:
            raise ValueError("configured SCIENCE declination is outside the celestial domain")
    elif header["CRVAL1"] != 0 or header["CRVAL2"] != 0:
        raise ValueError("configured POINT offset origin differs")


def learner_header(header):
    result = header.copy()
    if not is_celestial(header):
        for axis in (1, 2):
            result[f"CDELT{axis}"] /= 3600.0
            result[f"CRVAL{axis}"] /= 3600.0
            result[f"CUNIT{axis}"] = "deg"
    return result


def guard_frame(guard, header):
    if is_celestial(header):
        guard.update(frame=SCIENCE_FRAME["ASTFRAME"], centroid_axes=["tangent_ra", "tangent_dec"])
    else:
        guard.pop("expected_origin_sky", None)
        guard.update(frame=POINT_FRAME["ASTFRAME"], centroid_axes=["azimuth_offset", "elevation_offset"])
    return guard
