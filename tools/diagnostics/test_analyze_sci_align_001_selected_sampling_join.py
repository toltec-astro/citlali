#!/usr/bin/env python3

from __future__ import annotations

import csv
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


SCRIPT = Path(__file__).with_name(
    "analyze_sci_align_001_selected_sampling_join.py"
)
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
        (1, 7, 13, 19),
        x_rows,
        y_rows,
        ("left", "right", "left", "right"),
        strict=True,
    ):
        x[start:start + 5] = x_row
        y[start:start + 5] = y_row
        direction[start:start + 5] = scan_direction
    time = np.arange(x.size, dtype=float) * 0.01 + 1000.0
    return x, y, direction, time


def make_full_ptc(path: Path) -> None:
    x, y, _, time = scan_samples()
    n_pts, n_dets, n_scans = x.size, 2, 4
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("n_pts", None)
        dataset.createDimension("n_dets", n_dets)
        dataset.createDimension("n_scans", n_scans)
        dataset.createDimension("n_scan_indices", 2)
        dataset.createDimension("n_raw_scan_indices", 4)
        dataset.createDimension("fruitloops_iter_dim", 1)
        two_d = ("n_pts", "n_dets")
        for name in ("signal", "flags", "det_lat", "det_lon"):
            dataset.createVariable(name, "f8", two_d)
        for name in (
            "TelTime",
            "TelUTC",
            "alt_phys",
            "az_phys",
            "pointing_offset_alt",
            "pointing_offset_az",
        ):
            variable = dataset.createVariable(name, "f8", ("n_pts",))
            if name.startswith("pointing_offset"):
                variable.units = "arcsec"
        dataset.createVariable(
            "scan_indices", "i4", ("n_scans", "n_scan_indices")
        )
        raw_scan_indices = dataset.createVariable(
            "raw_scan_indices", "i4", ("n_scans", "n_raw_scan_indices")
        )
        raw_scan_indices.comment = (
            "indices in output timebase: inner_start, inner_end, "
            "outer_start, outer_end"
        )
        dataset.createVariable("output_scan_index", "i4", ("n_scans",))
        dataset.createVariable("weights", "f8", ("n_scans", "n_dets"))
        for name in ("apt_uid", "apt_array", "apt_nw", "apt_flag"):
            dataset.createVariable(name, "f8", ("n_dets",))
        dataset.createVariable("FRUITLOOPS_ITER", "i4", ("fruitloops_iter_dim",))
        dataset["signal"][:] = np.column_stack(
            [np.linspace(1.0, 2.0, n_pts), np.ones(n_pts)]
        )
        dataset["flags"][:] = 0.0
        dataset["det_lon"][:] = np.column_stack([x, x + 0.25]) / RAD_TO_ARCSEC
        dataset["det_lat"][:] = np.column_stack([y, y + 0.25]) / RAD_TO_ARCSEC
        dataset["az_phys"][:] = (x - 12.0) / RAD_TO_ARCSEC
        dataset["alt_phys"][:] = (y + 3.0) / RAD_TO_ARCSEC
        dataset["pointing_offset_az"][:] = 12.0
        dataset["pointing_offset_alt"][:] = -3.0
        dataset["TelTime"][:] = time
        dataset["TelUTC"][:] = time / 86400.0
        dataset["scan_indices"][:] = np.asarray(
            [[2, 4], [8, 10], [14, 16], [20, 22]]
        )
        dataset["raw_scan_indices"][:] = np.asarray([
            [2, 4, 1, 5],
            [8, 10, 7, 11],
            [14, 16, 13, 17],
            [20, 22, 19, 23],
        ])
        dataset["output_scan_index"][:] = np.arange(1, n_scans + 1)
        dataset["weights"][:] = 2.0
        dataset["apt_uid"][:] = [199.0, 200.0]
        dataset["apt_array"][:] = [0.0, 0.0]
        dataset["apt_nw"][:] = [0.0, 0.0]
        dataset["apt_flag"][:] = [0.0, 0.0]
        dataset["FRUITLOOPS_ITER"][:] = [0]


def retained_scan_signal(scan_id: int) -> np.ndarray:
    return np.linspace(float(scan_id), float(scan_id) + 0.4, 5)


def retained_scan_flags(scan_id: int) -> np.ndarray:
    flags = np.zeros(5, dtype=np.int8)
    if scan_id == 1:
        flags[2] = 1
    return flags


def make_selected_tod(path: Path) -> None:
    scan_ids = np.asarray([1, 2, 3, 4, 2], dtype=np.int32)
    slot_kinds = np.asarray([1, 1, 1, 1, 2], dtype=np.int32)
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("n_dets", 2)
        dataset.createDimension("n_slots", scan_ids.size)
        dataset.createDimension("n_samples", 5)
        for name in (
            "detector_tod_uid",
            "detector_tod_array",
            "detector_tod_network",
            "detector_tod_fit_good",
        ):
            dataset.createVariable(name, "i4", ("n_dets",))
        for name in (
            "detector_tod_fit_x_t_arcsec",
            "detector_tod_fit_y_t_arcsec",
        ):
            dataset.createVariable(name, "f8", ("n_dets",))
        for name in (
            "detector_tod_slot_kind",
            "detector_tod_scan_index",
            "detector_tod_n_samples",
            "detector_tod_scan_inner_start_sample",
            "detector_tod_scan_inner_end_sample",
        ):
            dataset.createVariable(name, "i4", ("n_dets", "n_slots"))
        dataset.createVariable(
            "signal", "f8", ("n_dets", "n_slots", "n_samples")
        )
        dataset.createVariable(
            "flags", "i1", ("n_dets", "n_slots", "n_samples")
        )
        dataset["detector_tod_uid"][:] = [199, 200]
        dataset["detector_tod_array"][:] = [0, 0]
        dataset["detector_tod_network"][:] = [0, 0]
        dataset["detector_tod_fit_good"][:] = [1, 1]
        dataset["detector_tod_fit_x_t_arcsec"][:] = [0.0, 1.0]
        dataset["detector_tod_fit_y_t_arcsec"][:] = [0.0, 1.0]
        dataset["detector_tod_slot_kind"][:] = np.vstack(
            [slot_kinds, slot_kinds]
        )
        dataset["detector_tod_scan_index"][:] = np.vstack(
            [scan_ids, scan_ids]
        )
        dataset["detector_tod_n_samples"][:] = 5
        dataset["detector_tod_scan_inner_start_sample"][:] = 10
        dataset["detector_tod_scan_inner_end_sample"][:] = 14
        signal = np.zeros((2, scan_ids.size, 5), dtype=float)
        flags = np.zeros((2, scan_ids.size, 5), dtype=np.int8)
        for slot, scan_id in enumerate(scan_ids):
            signal[:, slot, :] = retained_scan_signal(int(scan_id))
            flags[:, slot, :] = retained_scan_flags(int(scan_id))
        dataset["signal"][:] = signal
        dataset["flags"][:] = flags


def make_ptcdiag(path: Path) -> None:
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("n_scans", 4)
        dataset.createDimension("n_dets", 2)
        dataset.createVariable("output_scan_index", "i4", ("n_scans",))
        dataset.createVariable("ptc_diag_uid", "i4", ("n_dets",))
        dataset.createVariable(
            "ptc_detector_weight", "f8", ("n_scans", "n_dets")
        )
        dataset["output_scan_index"][:] = [1, 2, 3, 4]
        dataset["ptc_diag_uid"][:] = [199, 200]
        dataset["ptc_detector_weight"][:] = np.asarray(
            [[1.0, 2.0], [1.5, 2.0], [2.0, 2.0], [2.5, 2.0]]
        )


def make_registry(path: Path) -> None:
    columns = [
        "scan_index", "sample_count", "direction", "selected", "mode"
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for scan_index, direction in enumerate(
            ("left", "right", "left", "right")
        ):
            writer.writerow({
                "scan_index": scan_index,
                "sample_count": 5,
                "direction": direction,
                "selected": "true",
                "mode": "all",
            })


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
        {
            "uid": 199,
            "array": 0,
            "nw": 0,
            "flag": 0,
            "x_t_raw": 0.0,
            "y_t_raw": 0.0,
        },
        {
            "uid": 200,
            "array": 0,
            "nw": 0,
            "flag": 0,
            "x_t_raw": 1.0,
            "y_t_raw": 1.0,
        },
    ])
    table.meta["obsnum"] = 150819
    table.meta["beammap_direction_mode"] = mode
    table.write(path, format="ascii.ecsv")


def make_map(path: Path, mode: str) -> None:
    support = support_for_mode(mode)
    signal = np.zeros((21, 21), dtype=float)
    signal[support] = np.linspace(1.0, 2.0, np.sum(support))
    weight = support.astype(float) * 2.0
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU()
    ]
    for detector in (0, 1):
        for kind, data, unit in (
            ("signal", signal, "mJy/beam"),
            ("weight", weight, "1/(mJy/beam)^2"),
        ):
            header = image_header()
            header["BUNIT"] = unit
            hdus.append(fits.ImageHDU(
                data=data[np.newaxis, np.newaxis, :, :],
                header=header,
                name=f"{kind}_det_{detector}_I",
            ))
    fits.HDUList(hdus).writeto(path)


class SelectedSamplingJoinTest(unittest.TestCase):
    def test_explicit_scan_join_and_exact_selected_support(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ptc = root / "full_ptc.nc"
            make_full_ptc(ptc)
            raw = root / "maps" / "redu00" / "150819" / "raw"
            tod = raw / "source_crossing_tod"
            tod.mkdir(parents=True)
            make_selected_tod(tod / "synthetic_ptc_detector_tod.nc")
            make_ptcdiag(tod / "synthetic_ptcdiag.nc")
            make_registry(raw / "beammap_direction_scan_registry_all.csv")
            base = raw / "apt_commissioning_beammap_150819_citlali.ecsv"
            for mode in MODES:
                apt = (
                    base
                    if mode == "standard"
                    else base.with_name(base.stem + f"_{mode}.ecsv")
                )
                make_apt(apt, mode)
                suffix = "" if mode == "standard" else f"_{mode}"
                make_map(
                    raw
                    / f"toltec_commissioning_a1100_beammap_150819_citlali{suffix}.fits",
                    mode,
                )
            output = root / "audit"
            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"
            env["MPLCONFIGDIR"] = str(root / "mpl")
            env["XDG_CACHE_HOME"] = str(root / "xdg")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--full-ptc-tod",
                    str(ptc),
                    "--map-reduction-root",
                    str(root / "maps"),
                    "--output",
                    str(output),
                    "--uid",
                    "199",
                    "--half-width-arcsec",
                    "8",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("distinct_scans=4 duplicates=1 left=2 right=2", result.stdout)
            metrics = Table.read(output / "mode_selected_support.ecsv")
            self.assertEqual(list(metrics["mode"]), list(MODES))
            np.testing.assert_allclose(
                np.asarray(metrics["selected_hit_supported_fraction"]), 1.0
            )
            np.testing.assert_array_equal(
                np.asarray(metrics["selected_hit_only_pixels"]), 0
            )
            scans = Table.read(output / "selected_scan_join.ecsv")
            self.assertEqual(list(scans["scan_id_one_based"]), [1, 2, 3, 4])
            self.assertEqual(list(scans["direction"]), ["left", "right", "left", "right"])
            self.assertEqual(list(scans["duplicate_slot_count"]), [1, 2, 1, 1])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["selected_slot_count"], 5)
            self.assertEqual(manifest["distinct_joined_scan_count"], 4)
            self.assertEqual(manifest["duplicate_selected_slot_count"], 1)
            self.assertEqual(manifest["joined_left_scan_count"], 2)
            self.assertEqual(manifest["joined_right_scan_count"], 2)
            self.assertEqual(
                len(
                    PdfReader(
                        output / "selected_sampling_join_o150819_uid199.pdf"
                    ).pages
                ),
                2,
            )
            self.assertTrue((output / "selected_hit_counts.npz").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())


if __name__ == "__main__":
    unittest.main()
