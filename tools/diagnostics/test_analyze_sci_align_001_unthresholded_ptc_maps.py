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


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_unthresholded_ptc_maps as target  # noqa: E402
from analyze_sci_align_001_ptc_sampling import PtcDetector  # noqa: E402


SCRIPT = Path(__file__).with_name(
    "analyze_sci_align_001_unthresholded_ptc_maps.py"
)
RAD_TO_ARCSEC = 206264.80624709636


def synthetic_ptc() -> PtcDetector:
    signal = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    flags = np.zeros(signal.size, dtype=float)
    flags[1] = 1.0
    tel_time = np.arange(signal.size, dtype=float)
    az = np.asarray([2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0])
    lon = np.asarray([-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0])
    lat = np.zeros(signal.size, dtype=float)
    return PtcDetector(
        path=Path("synthetic.nc"),
        detector_index=0,
        uid=199,
        array=0,
        nw=0,
        apt_flag=0,
        fruitloops_iter=0,
        signal=signal,
        flags=flags,
        det_lat=lat,
        det_lon=lon,
        tel_time=tel_time,
        tel_utc=tel_time,
        alt_phys=lat,
        az_phys=az,
        scan_indices=np.asarray([[0, 3], [4, 7]], dtype=np.int64),
        output_scan_index=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([2.0, 3.0]),
    )


class UnthresholdedPtcMapTest(unittest.TestCase):
    def test_exact_naive_accumulation_and_direction_split(self) -> None:
        ptc = synthetic_ptc()
        registry = {
            1: target.RegistryScan(1, 4, "left"),
            2: target.RegistryScan(2, 4, "right"),
        }
        maps = target.reconstruct_unthresholded_maps(
            ptc,
            ptc.scan_indices,
            registry,
            ptc.det_lat,
            ptc.det_lon,
            (7, 7),
            1.0,
        )

        self.assertEqual(maps["standard"].scan_count, 2)
        self.assertEqual(maps["left"].scan_count, 1)
        self.assertEqual(maps["right"].scan_count, 1)
        self.assertEqual(maps["standard"].accepted_sample_count, 7)
        self.assertEqual(maps["left"].accepted_sample_count, 3)
        self.assertEqual(maps["right"].accepted_sample_count, 4)

        row = 3
        # lon=0 receives left signal 3 at weight 2 and right signal 30 at weight 3.
        col = 3
        self.assertEqual(maps["standard"].hit_count[row, col], 2)
        self.assertAlmostEqual(maps["standard"].weight[row, col], 5.0)
        self.assertAlmostEqual(maps["standard"].signal[row, col], 19.2)
        self.assertAlmostEqual(maps["left"].signal[row, col], 3.0)
        self.assertAlmostEqual(maps["right"].signal[row, col], 30.0)

        # The flagged left sample at lon=-1 contributes nowhere.
        flagged_col = 2
        self.assertEqual(maps["left"].hit_count[row, flagged_col], 0)
        self.assertEqual(maps["standard"].hit_count[row, flagged_col], 1)
        self.assertAlmostEqual(maps["standard"].signal[row, flagged_col], 20.0)

    def test_gaussian_fit_recovers_directional_centers(self) -> None:
        x = np.arange(-40.0, 41.0)
        y = np.arange(-40.0, 41.0)
        xx, yy = np.meshgrid(x, y)
        centers = {
            "standard": (0.0, -0.25),
            "left": (1.25, -0.5),
            "right": (-1.35, -0.45),
        }
        maps: dict[str, target.ReconstructedMap] = {}
        for mode, (cx, cy) in centers.items():
            image = (
                8.0 * np.exp(-0.5 * (((xx - cx) / 3.0) ** 2 + ((yy - cy) / 4.0) ** 2))
                + 0.01 * xx - 0.005 * yy + 0.2
            )
            item = target.empty_map(mode, image.shape)
            item.signal[:] = image
            item.weight[:] = 1.0
            item.scan_count = 1
            maps[mode] = item

        fits = target.fit_reconstructions(maps, x, y, 0.0, 0.0, 12.0)
        for mode, expected in centers.items():
            self.assertAlmostEqual(fits[mode]["x_arcsec"], expected[0], places=3)
            self.assertAlmostEqual(fits[mode]["y_arcsec"], expected[1], places=3)

        class Scan:
            axis_x = 1.0
            axis_y = 0.0
            cross_x = 0.0
            cross_y = 1.0
            left_rate_arcsec_s = -100.0
            right_rate_arcsec_s = 100.0

        row = target.displacement_row(
            "synthetic",
            {mode: centers[mode] for mode in target.MODES},
            Scan(),
        )
        self.assertAlmostEqual(
            row["delta_parallel_right_minus_left_arcsec"], -2.6
        )
        self.assertAlmostEqual(row["timing_equivalent_ms"], -13.0)

    def test_end_to_end_products_preserve_unthresholded_shift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ptc_path = root / "full_ptc.nc"
            map_root = root / "maps"
            raw = map_root / "redu00" / "150819" / "raw"
            raw.mkdir(parents=True)
            make_end_to_end_ptc(ptc_path)
            make_end_to_end_registry(raw / "beammap_direction_scan_registry_all.csv")
            make_end_to_end_retained_products(raw)
            output = root / "output"
            environment = os.environ.copy()
            environment.update({
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(root / "mpl"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "--full-ptc-tod",
                    str(ptc_path),
                    "--map-reduction-root",
                    str(map_root),
                    "--output",
                    str(output),
                    "--uid",
                    "199",
                    "--fit-half-width-arcsec",
                    "10",
                    "--plot-half-width-arcsec",
                    "18",
                ],
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            row = manifest["displacement_comparison"][0]
            self.assertEqual(row["family"], "unthresholded_full_ptc_reconstruction")
            self.assertAlmostEqual(
                row["delta_parallel_right_minus_left_arcsec"], -2.4, delta=0.08
            )
            pointing = manifest["pointing_contract"]
            self.assertAlmostEqual(
                pointing["observed_offset_median_arcsec"], math.sqrt(13.0),
                places=6,
            )
            self.assertLess(pointing["offset_model_max_residual_arcsec"], 1.0e-8)
            self.assertEqual(len(PdfReader(
                output / "unthresholded_ptc_maps_o150819_uid199.pdf"
            ).pages), 2)
            checksums = (output / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(checksums), 5)


def synthetic_scan_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scan_count = 40
    samples_per_scan = 81
    x_template = np.linspace(-20.0, 20.0, samples_per_scan)
    y_rows = np.linspace(-9.75, 9.75, scan_count // 2)
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    directions: list[str] = []
    signal_values: list[np.ndarray] = []
    for scan_index in range(scan_count):
        direction = "left" if scan_index % 2 == 0 else "right"
        pair = scan_index // 2
        x = x_template[::-1] if direction == "left" else x_template
        y = np.full(samples_per_scan, y_rows[pair])
        center_x = 1.2 if direction == "left" else -1.2
        signal = (
            10.0 * np.exp(-0.5 * (((x - center_x) / 3.0) ** 2 + (y / 4.0) ** 2))
            + 0.1 + 0.002 * x
        )
        x_values.append(x)
        y_values.append(y)
        directions.append(direction)
        signal_values.append(signal)
    return (
        np.concatenate(x_values),
        np.concatenate(y_values),
        np.asarray(directions),
        np.concatenate(signal_values),
    )


def make_end_to_end_ptc(path: Path) -> None:
    x, y, directions, signal = synthetic_scan_arrays()
    scan_count = directions.size
    samples_per_scan = signal.size // scan_count
    elevation = 0.75
    apt_x, apt_y = 3.0, -2.0
    physical_lon_offset = math.cos(elevation) * apt_x - math.sin(elevation) * apt_y
    physical_lat_offset = math.cos(elevation) * apt_y + math.sin(elevation) * apt_x
    with netCDF4.Dataset(path, mode="w") as dataset:
        dataset.createDimension("n_pts", None)
        dataset.createDimension("n_dets", 1)
        dataset.createDimension("n_scans", scan_count)
        dataset.createDimension("n_scan_indices", 2)
        dataset.createDimension("n_raw_scan_indices", 4)
        dataset.createDimension("fruitloops_iter_dim", 1)
        for name in ("signal", "flags", "det_lat", "det_lon"):
            dataset.createVariable(name, "f8", ("n_pts", "n_dets"))
        for name in (
            "TelTime", "TelUTC", "alt_phys", "az_phys",
            "pointing_offset_alt", "pointing_offset_az",
        ):
            variable = dataset.createVariable(name, "f8", ("n_pts",))
            if name.startswith("pointing_offset"):
                variable.units = "arcsec"
        dataset.createVariable("scan_indices", "i4", ("n_scans", "n_scan_indices"))
        raw_bounds = dataset.createVariable(
            "raw_scan_indices", "i4", ("n_scans", "n_raw_scan_indices")
        )
        raw_bounds.comment = "indices in output timebase: inner_start, inner_end, outer_start, outer_end"
        dataset.createVariable("output_scan_index", "i4", ("n_scans",))
        dataset.createVariable("weights", "f8", ("n_scans", "n_dets"))
        for name in ("apt_uid", "apt_array", "apt_nw", "apt_flag"):
            dataset.createVariable(name, "f8", ("n_dets",))
        for name in ("apt_x_t", "apt_y_t"):
            variable = dataset.createVariable(name, "f8", ("n_dets",))
            variable.units = "arcsec"
        tel_el = dataset.createVariable("TelElAct", "f8", ("n_pts",))
        tel_el.units = "rad"
        dataset.createVariable("FRUITLOOPS_ITER", "i4", ("fruitloops_iter_dim",))
        dataset["signal"][:] = signal[:, None]
        dataset["flags"][:] = 0.0
        dataset["det_lon"][:] = ((x + physical_lon_offset) / RAD_TO_ARCSEC)[:, None]
        dataset["det_lat"][:] = ((y + physical_lat_offset) / RAD_TO_ARCSEC)[:, None]
        dataset["az_phys"][:] = x / RAD_TO_ARCSEC
        dataset["alt_phys"][:] = y / RAD_TO_ARCSEC
        dataset["TelElAct"][:] = elevation
        dataset["pointing_offset_az"][:] = 0.0
        dataset["pointing_offset_alt"][:] = 0.0
        dataset["TelTime"][:] = np.arange(signal.size, dtype=float) * 0.02 + 1000.0
        dataset["TelUTC"][:] = dataset["TelTime"][:] / 86400.0
        bounds = np.asarray([
            [row * samples_per_scan, (row + 1) * samples_per_scan - 1]
            for row in range(scan_count)
        ], dtype=np.int32)
        dataset["scan_indices"][:] = bounds
        dataset["raw_scan_indices"][:] = np.column_stack([bounds, bounds])
        dataset["output_scan_index"][:] = np.arange(1, scan_count + 1)
        dataset["weights"][:] = np.linspace(1.0, 2.0, scan_count)[:, None]
        dataset["apt_uid"][:] = [199.0]
        dataset["apt_array"][:] = [0.0]
        dataset["apt_nw"][:] = [0.0]
        dataset["apt_flag"][:] = [0.0]
        dataset["apt_x_t"][:] = [apt_x]
        dataset["apt_y_t"][:] = [apt_y]
        dataset["FRUITLOOPS_ITER"][:] = [0]


def make_end_to_end_registry(path: Path) -> None:
    _, _, directions, _ = synthetic_scan_arrays()
    columns = [
        "scan_index", "sample_count", "start_time_sec", "stop_time_sec",
        "duration_sec", "scan_angle_rad", "signed_fast_axis_rate_rad_per_sec",
        "direction", "selected", "mode",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for scan_index, direction in enumerate(directions):
            signed_rate = -25.0 if direction == "left" else 25.0
            writer.writerow({
                "scan_index": scan_index,
                "sample_count": 81,
                "start_time_sec": 1000.0 + scan_index * 1.62,
                "stop_time_sec": 1001.6 + scan_index * 1.62,
                "duration_sec": 1.6,
                "scan_angle_rad": 0.0,
                "signed_fast_axis_rate_rad_per_sec": signed_rate / RAD_TO_ARCSEC,
                "direction": direction,
                "selected": "true",
                "mode": "all",
            })


def synthetic_image_header() -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "AZOFFSET"
    header["CTYPE2"] = "ELOFFSET"
    header["CTYPE3"] = "FREQ"
    header["CTYPE4"] = "STOKES"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 41.0
    header["CRPIX2"] = 41.0
    header["CRPIX3"] = 1.0
    header["CRPIX4"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRVAL3"] = 270.0e9
    header["CRVAL4"] = 0.0
    header["CDELT1"] = -0.5
    header["CDELT2"] = 0.5
    header["CDELT3"] = 1.0
    header["CDELT4"] = 1.0
    return header


def make_end_to_end_retained_products(raw: Path) -> None:
    x = np.linspace(-20.0, 20.0, 81)
    y = np.linspace(-20.0, 20.0, 81)
    xx, yy = np.meshgrid(x, y)
    centers = {"standard": (0.0, 0.0), "left": (1.2, 0.0), "right": (-1.2, 0.0)}
    base = raw / "apt_commissioning_beammap_150819_citlali.ecsv"
    for mode in target.MODES:
        center_x, center_y = centers[mode]
        apt_path = base if mode == "standard" else base.with_name(base.stem + f"_{mode}.ecsv")
        table = Table(rows=[{
            "uid": 199, "array": 0, "nw": 0, "flag": 0,
            "x_t_raw": center_x, "y_t_raw": center_y,
        }])
        table.meta["obsnum"] = 150819
        table.meta["beammap_direction_mode"] = mode
        table.write(apt_path, format="ascii.ecsv")
        image = 10.0 * np.exp(-0.5 * (((xx - center_x) / 3.0) ** 2 + (yy / 4.0) ** 2))
        support = image > 0.03
        image = np.where(support, image, 0.0)
        suffix = "" if mode == "standard" else f"_{mode}"
        fits_path = raw / f"toltec_commissioning_a1100_beammap_150819_citlali{suffix}_flag0_good.fits"
        hdus = [fits.PrimaryHDU()]
        for kind, values in (
            ("signal", image),
            ("weight", support.astype(float)),
        ):
            hdus.append(fits.ImageHDU(
                data=values[np.newaxis, np.newaxis, :, :],
                header=synthetic_image_header(),
                name=f"{kind}_det_0_I",
            ))
        fits.HDUList(hdus).writeto(fits_path)


if __name__ == "__main__":
    unittest.main()
