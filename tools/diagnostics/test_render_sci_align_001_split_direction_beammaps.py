#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from pypdf import PdfReader


SCRIPT = Path(__file__).with_name(
    "render_sci_align_001_split_direction_beammaps.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def make_apt(path: Path, mode: str, offsets: list[float]) -> None:
    rows = []
    for uid, offset in enumerate(offsets):
        rows.append({
            "uid": uid,
            "array": 0,
            "nw": uid % 2,
            "flag": 0,
            "flag2": 0,
            "amp": 1.0,
            "amp_err": 0.01,
            # Deliberately keep processed focal-plane coordinates independent
            # of direction.  The renderer must use raw map-frame positions.
            "x_t": 30.0 + uid,
            "x_t_raw": 10.0 * uid + offset,
            "x_t_err": 0.05,
            "y_t": 20.0 + uid,
            "y_t_raw": -5.0 * uid,
            "y_t_err": 0.05,
            "a_fwhm": 5.0,
            "b_fwhm": 5.5,
            "angle": 0.0,
            "sig2noise": 100.0 - uid,
        })
    table = Table(rows=rows)
    table.meta["obsnum"] = 42
    table.meta["beammap_direction_mode"] = mode
    table.write(path, format="ascii.ecsv")
    qc = path.with_name(path.stem + "_fit_qc.ecsv")
    Table(rows=[{"uid": row["uid"], "good_fit": 1} for row in rows]).write(
        qc, format="ascii.ecsv"
    )


def image_header() -> fits.Header:
    header = fits.Header()
    header["BUNIT"] = "mJy/beam"
    header["CTYPE1"] = "AZOFFSET"
    header["CTYPE2"] = "ELOFFSET"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "STOKES"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 51.0
    header["CRPIX2"] = 51.0
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = 270.0e9
    header["CRVAL4"] = 0.0
    header["CDELT1"] = -1.0
    header["CDELT2"] = 1.0
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1.0
    return header


def make_fits(path: Path, offsets: list[float]) -> None:
    yy, xx = np.mgrid[:101, :101]
    world_x = -(xx - 50.0)
    world_y = yy - 50.0
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU()
    ]
    for uid, offset in enumerate(offsets):
        cx, cy = 10.0 * uid + offset, -5.0 * uid
        signal = np.exp(-0.5 * (((world_x - cx) / 2.2) ** 2 + ((world_y - cy) / 2.4) ** 2))
        weight = np.ones_like(signal)
        kernel = signal.copy()
        for kind, data, unit in (
            ("signal", signal, "mJy/beam"),
            ("weight", weight, "1/(mJy/beam)^2"),
            ("kernel", kernel, "mJy/beam"),
        ):
            header = image_header()
            header["BUNIT"] = unit
            hdus.append(fits.ImageHDU(
                data=data[np.newaxis, np.newaxis, :, :],
                header=header,
                name=f"{kind}_det_{uid}_I",
            ))
    fits.HDUList(hdus).writeto(path)


def make_registry(path: Path) -> None:
    path.write_text(
        "scan_index,science_start,science_stop_exclusive,sample_count,"
        "start_time_sec,stop_time_sec,duration_sec,coordinate_x_key,"
        "coordinate_y_key,scan_angle_rad,fast_axis_displacement_rad,"
        "signed_fast_axis_rate_rad_per_sec,same_sign_step_fraction,"
        "direction,selected,mode\n"
        "0,0,100,100,0,1,1,az_phys,alt_phys,0,-0.001,-0.0005,1,left,true,all\n"
        "1,100,200,100,1,2,1,az_phys,alt_phys,0,0.001,0.0005,1,right,true,all\n"
    )


class SplitDirectionVisualizationTest(unittest.TestCase):
    def test_end_to_end_one_detector_per_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "reduced" / "redu00" / "42" / "raw"
            raw.mkdir(parents=True)
            base = raw / "apt_commissioning_beammap_42_citlali.ecsv"
            make_apt(base, "standard", [0.0, 0.0, 0.0])
            make_apt(base.with_name(base.stem + "_left.ecsv"), "left", [-2.0, -2.0, -2.0])
            make_apt(base.with_name(base.stem + "_right.ecsv"), "right", [2.0, 2.0, 2.0])
            make_fits(raw / "toltec_commissioning_a1100_beammap_42_citlali.fits", [0.0, 0.0, 0.0])
            make_fits(
                raw / "toltec_commissioning_a1100_beammap_42_citlali_left.fits",
                [-2.0, -2.0, -2.0],
            )
            make_fits(
                raw / "toltec_commissioning_a1100_beammap_42_citlali_right.fits",
                [2.0, 2.0, 2.0],
            )
            make_registry(raw / "beammap_direction_scan_registry_all.csv")
            output = root / "review"
            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"
            env["MPLCONFIGDIR"] = str(root / "mpl")
            env["XDG_CACHE_HOME"] = str(root / "xdg")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--reduction-root", str(root),
                    "--output", str(output),
                    "--max-detectors", "2",
                    "--detectors-per-page", "1",
                    "--half-width-arcsec", "12",
                ],
                env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            pdf = output / "split_direction_beammaps_o42_a1100.pdf"
            self.assertEqual(len(PdfReader(pdf).pages), 2)
            metrics = Table.read(output / "detector_metrics.ecsv")
            self.assertEqual(len(metrics), 2)
            np.testing.assert_allclose(
                np.asarray(metrics["delta_parallel_right_minus_left_arcsec"]),
                4.0,
            )
            self.assertEqual(set(metrics["position_frame"]), {"raw_altaz_detector_map"})
            np.testing.assert_allclose(
                np.asarray(metrics["standard_x_t_raw"]), [0.0, 10.0],
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["position_authority"]["centroid_columns"],
                ["x_t_raw", "y_t_raw"],
            )
            self.assertFalse(manifest["selection"]["uses_directional_displacement"])
            self.assertEqual(manifest["layout"]["maximum_supported_detectors_per_page"], 2)
            self.assertEqual(manifest["tool"]["sha256"], sha256_file(SCRIPT))
            self.assertTrue((output / "SHA256SUMS").is_file())

            two_up = root / "review_two_up"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--reduction-root", str(root),
                    "--output", str(two_up),
                    "--max-detectors", "2",
                    "--detectors-per-page", "2",
                    "--half-width-arcsec", "12",
                ],
                env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                len(PdfReader(two_up / "split_direction_beammaps_o42_a1100.pdf").pages),
                1,
            )

    def test_parser_rejects_more_than_two_detectors_per_page(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--reduction-root", "/does/not/matter",
                "--output", "/does/not/matter",
                "--detectors-per-page", "3",
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
