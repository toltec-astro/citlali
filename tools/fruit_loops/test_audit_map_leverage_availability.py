from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from tools.fruit_loops import audit_map_leverage_availability as audit


def write_test_map(path: Path, formal: np.ndarray, empirical: np.ndarray) -> None:
    primary = fits.PrimaryHDU()
    primary.header["METHOD"] = "jinc"
    hdus = [primary]
    planes = {
        "SIGNAL_I": np.ones_like(formal),
        "WEIGHT_I": empirical,
        "WEIGHT_FORMAL_I": formal,
    }
    for name, values in planes.items():
        hdu = fits.ImageHDU(values, name=name)
        hdu.header["BUNIT"] = (
            "mJy/beam" if name == "SIGNAL_I" else "1/(mJy/beam)^2"
        )
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CRPIX2"] = 1.0
        hdu.header["CRVAL1"] = 0.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CDELT1"] = -1.0
        hdu.header["CDELT2"] = 1.0
        hdu.header["CUNIT1"] = "arcsec"
        hdu.header["CUNIT2"] = "arcsec"
        hdu.header["CTYPE1"] = "AZOFFSET"
        hdu.header["CTYPE2"] = "ELOFFSET"
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)


def test_midpoint_percentile_handles_ties() -> None:
    values = np.asarray([0.0, 0.0, 0.0, 1.0])

    assert audit.midpoint_percentile(values, 0.0) == 37.5
    assert audit.midpoint_percentile(values, 1.0) == 87.5


def test_jinc_final_coefficient_is_not_additive() -> None:
    examples = audit.jinc_nonadditivity_examples()

    assert examples["all_minus_without_negative"] < 0
    assert examples["all_minus_without_positive"] > 0
    assert examples["demonstrates_signed_final_coefficient_difference"] is True


def test_signed_formal_difference_stops_leverage_interpretation(
    tmp_path: Path,
) -> None:
    all_path = tmp_path / "all.fits"
    without_path = tmp_path / "without.fits"
    write_test_map(
        all_path,
        np.asarray([[0.8, 1.2], [0.0, 1.0]]),
        np.asarray([[1.6, 2.4], [0.0, 2.0]]),
    )
    write_test_map(
        without_path,
        np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        np.asarray([[3.0, 3.0], [3.0, 3.0]]),
    )

    compatibility = audit.require_compatible_maps(all_path, without_path)
    result, _ = audit.analyze_final_coefficients(all_path, without_path)

    assert compatibility["method"] == "jinc"
    assert result["materially_negative_difference_pixels"] == 2
    assert result["materially_positive_difference_pixels"] == 1
    assert result["exact_uid_leverage_available"] is False
    assert result["n5_empirical_to_formal_ratio_median"] == 2.0
    assert result["a5_map_empirical_to_formal_ratio_median"] == 3.0


def test_learning_evidence_tracks_four_trigger_pixels_and_application(
    tmp_path: Path,
) -> None:
    header = [
        "record_type",
        "iter",
        "reason",
        "scan",
        "uid",
        "factor",
        "score",
        "scan_local",
        "row",
        "col",
        "sample",
        "value",
        "n_eff",
        "leave_one_out_z",
        "source_distance_arcsec",
        "application_stage",
        "proposed_samples",
        "newly_flagged_samples",
        "already_flagged_samples",
        "source_protected_samples",
        "applied",
    ]
    source = tmp_path / "source.csv"
    application = tmp_path / "application.csv"
    import csv

    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerow(
            {
                "record_type": "detector_penalty",
                "iter": 4,
                "reason": "map_pixel_outlier_detector_dominance",
                "scan": 5,
                "uid": 4460,
                "factor": 0,
                "score": 4,
                "scan_local": 1,
            }
        )
        for index in range(4):
            writer.writerow(
                {
                    "record_type": "map_pixel_outlier",
                    "iter": 4,
                    "reason": "extreme_pixel_targeted_contributor",
                    "scan": 5,
                    "uid": 4460,
                    "row": 10 + index,
                    "col": 20,
                    "sample": 30,
                    "value": 100,
                    "n_eff": 200,
                    "leave_one_out_z": 2,
                    "source_distance_arcsec": 100,
                }
            )
    with application.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerow(
            {
                "record_type": "detector_penalty_application",
                "iter": 5,
                "scan": 5,
                "uid": -1,
                "application_stage": "pre_mapmaking_detector_exclusion",
                "proposed_samples": 305,
                "newly_flagged_samples": 271,
                "already_flagged_samples": 34,
                "source_protected_samples": 0,
                "applied": 1,
            }
        )

    result, triggers = audit.learning_evidence(
        source, application, uid=4460, scan=5
    )

    assert result["penalty_score"] == 4
    assert result["proposed_samples"] == 305
    assert result["already_flagged_samples"] == 34
    assert len(triggers) == 4
