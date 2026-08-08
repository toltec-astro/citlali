#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np
from astropy.io import fits
from astropy.table import Table
from pypdf import PdfReader


SCRIPT = Path(__file__).with_name("analyze_sci_align_001_ptc_sampling.py")
RAD_TO_ARCSEC = 180.0 * 3600.0 / math.pi
MODES = ("standard", "left", "right")


def image_header() -> fits.Header:
    header = fits.Header()
    header["BUNIT"] = "mJy/beam"
    header["CTYPE1"] = "AZOFFSET"
    header["CTYPE2"] = "ELOFFSET"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "STOKES"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 11.0
    header["CRPIX2"] = 11.0
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = 270.0e9
    header["CRVAL4"] = 0.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1.0
    return header


def scan_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows = [
        [4.0, 2.0, 0.0, -2.0, -4.0],
        [-4.0, -2.0, 0.0, 2.0, 4.0],
        [4.0, 2.0, 0.0, -2.0, -4.0],
        [-4.0, -2.0, 0.0, 2.0, 4.0],
    ]
    y_rows = [-2.0, -1.0, 1.0, 2.0]
    x = np.full(25, 9.0, dtype=float)
    y = np.full(25, 9.0, dtype=float)
    direction = np.full(25, "outside", dtype="U8")
    for start, x_row, y_row, scan_direction in zip(
        (1, 7, 13, 19), x_rows, y_rows,
        ("left", "right", "left", "right"), strict=True,
    ):
        x[start:start + 5] = x_row
        y[start:start + 5] = y_row
        direction[start:start + 5] = scan_direction
    time = np.arange(x.size, dtype=float) * 0.01 + 1000.0
    return x, y, direction, time


def make_ptc(path: Path) -> None:
    x, y, _, time = scan_samples()
    n_pts, n_dets, n_scans = x.size, 2, 4
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("n_pts", None)
        dataset.createDimension("n_dets", n_dets)
        dataset.createDimension("n_scans", n_scans)
        dataset.createDimension("n_scan_indices", 2)
        dataset.createDimension("fruitloops_iter_dim", 1)
        two_d = ("n_pts", "n_dets")
        for name in ("signal", "flags", "det_lat", "det_lon"):
            dataset.createVariable(name, "f8", two_d)
        for name in ("TelTime", "TelUTC", "alt_phys", "az_phys"):
            dataset.createVariable(name, "f8", ("n_pts",))
        dataset.createVariable("scan_indices", "i4", ("n_scans", "n_scan_indices"))
        dataset.createVariable("output_scan_index", "i4", ("n_scans",))
        dataset.createVariable("weights", "f8", ("n_scans", "n_dets"))
        for name in ("apt_uid", "apt_array", "apt_nw", "apt_flag"):
            dataset.createVariable(name, "f8", ("n_dets",))
        dataset.createVariable("FRUITLOOPS_ITER", "i4", ("fruitloops_iter_dim",))
        dataset["signal"][:] = np.column_stack([np.linspace(1.0, 2.0, n_pts), np.ones(n_pts)])
        flags = np.zeros((n_pts, n_dets), dtype=float)
        flags[3, 0] = 1.0
        dataset["flags"][:] = flags
        dataset["det_lon"][:] = np.column_stack([x, x + 0.25]) / RAD_TO_ARCSEC
        dataset["det_lat"][:] = np.column_stack([y, y + 0.25]) / RAD_TO_ARCSEC
        dataset["az_phys"][:] = (x - 12.0) / RAD_TO_ARCSEC
        dataset["alt_phys"][:] = (y + 3.0) / RAD_TO_ARCSEC
        dataset["TelTime"][:] = time
        dataset["TelUTC"][:] = time / 86400.0
        dataset["scan_indices"][:] = np.asarray([[1, 5], [7, 11], [13, 17], [19, 23]])
        dataset["output_scan_index"][:] = np.arange(1, n_scans + 1)
        dataset["weights"][:] = 2.0
        dataset["apt_uid"][:] = [199.0, 200.0]
        dataset["apt_array"][:] = [0.0, 0.0]
        dataset["apt_nw"][:] = [0.0, 0.0]
        dataset["apt_flag"][:] = [0.0, 0.0]
        dataset["FRUITLOOPS_ITER"][:] = [0]


def support_for_mode(mode: str) -> np.ndarray:
    x, y, direction, _ = scan_samples()
    accepted = direction != "outside"
    accepted[3] = False
    if mode != "standard":
        accepted &= direction == mode
    result = np.zeros((21, 21), dtype=bool)
    rows = np.floor(y[accepted] + 10.0 + 0.5).astype(int)
    columns = np.floor(x[accepted] + 10.0 + 0.5).astype(int)
    result[rows, columns] = True
    return result


def make_apt(path: Path, mode: str) -> None:
    table = Table(rows=[
        {"uid": 199, "array": 0, "nw": 0, "flag": 0,
         "x_t_raw": 0.0, "y_t_raw": 0.0},
        {"uid": 200, "array": 0, "nw": 0, "flag": 0,
         "x_t_raw": 1.0, "y_t_raw": 1.0},
    ])
    table.meta["obsnum"] = 150819
    table.meta["beammap_direction_mode"] = mode
    table.write(path, format="ascii.ecsv")


def make_map(path: Path, mode: str) -> None:
    support = support_for_mode(mode)
    signal = np.zeros((21, 21), dtype=float)
    signal[support] = np.linspace(1.0, 2.0, np.sum(support))
    weight = support.astype(float) * 2.0
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [fits.PrimaryHDU()]
    for detector in (0, 1):
        for kind, data, unit in (
            ("signal", signal, "mJy/beam"),
            ("weight", weight, "1/(mJy/beam)^2"),
        ):
            header = image_header()
            header["BUNIT"] = unit
            hdus.append(fits.ImageHDU(
                data=data[np.newaxis, np.newaxis, :, :], header=header,
                name=f"{kind}_det_{detector}_I",
            ))
    fits.HDUList(hdus).writeto(path)


class PtcSamplingAuditTest(unittest.TestCase):
    def test_exact_naive_hit_support_and_self_contained_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ptc = root / "full_ptc.nc"
            make_ptc(ptc)
            raw = root / "maps" / "redu00" / "150819" / "raw"
            raw.mkdir(parents=True)
            base = raw / "apt_commissioning_beammap_150819_citlali.ecsv"
            for mode in MODES:
                apt = base if mode == "standard" else base.with_name(base.stem + f"_{mode}.ecsv")
                make_apt(apt, mode)
                suffix = "" if mode == "standard" else f"_{mode}"
                make_map(
                    raw / f"toltec_commissioning_a1100_beammap_150819_citlali{suffix}.fits",
                    mode,
                )
            output = root / "audit"
            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"
            env["MPLCONFIGDIR"] = str(root / "mpl")
            env["XDG_CACHE_HOME"] = str(root / "xdg")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--ptc-tod", str(ptc),
                    "--map-reduction-root", str(root / "maps"),
                    "--output", str(output),
                    "--uid", "199",
                    "--half-width-arcsec", "8",
                ],
                env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("left=2 right=2", result.stdout)
            metrics = Table.read(output / "mode_support_metrics.ecsv")
            self.assertEqual(list(metrics["mode"]), list(MODES))
            np.testing.assert_allclose(np.asarray(metrics["raw_jaccard"]), 1.0)
            np.testing.assert_array_equal(np.asarray(metrics["raw_hit_only_pixels"]), 0)
            np.testing.assert_array_equal(np.asarray(metrics["raw_map_only_pixels"]), 0)
            np.testing.assert_array_equal(np.asarray(metrics["registered_row_shift_pixels"]), 0)
            np.testing.assert_array_equal(np.asarray(metrics["registered_column_shift_pixels"]), 0)
            scans = Table.read(output / "scan_classification.ecsv")
            self.assertEqual(list(scans["direction"]), ["left", "right", "left", "right"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["fruitloops_iter"], 0)
            self.assertEqual(manifest["left_scan_count"], 2)
            self.assertEqual(manifest["right_scan_count"], 2)
            self.assertEqual(manifest["unclassified_sample_count"], 5)
            self.assertLess(
                manifest["trajectory_metrics"][
                    "detector_minus_telescope_step_residual_max_arcsec"
                ],
                1.0e-8,
            )
            self.assertEqual(len(PdfReader(output / "ptc_sampling_audit_o150819_uid199.pdf").pages), 2)
            self.assertTrue((output / "hit_counts.npz").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())


if __name__ == "__main__":
    unittest.main()
