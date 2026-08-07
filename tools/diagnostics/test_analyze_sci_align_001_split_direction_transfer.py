#!/usr/bin/env python3

from __future__ import annotations

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
    "analyze_sci_align_001_split_direction_transfer.py"
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
    header["CRPIX1"] = 81.0
    header["CRPIX2"] = 81.0
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


def make_apt(path: Path, mode: str, signal_offset: float, count: int = 4) -> None:
    rows = []
    for uid in range(count):
        x0, y0 = 12.0 * uid - 18.0, 8.0 * (uid % 2) - 4.0
        rows.append({
            "uid": uid,
            "array": 0,
            "nw": uid % 2,
            "flag": 0,
            "flag2": 0,
            "amp": 1.0,
            "amp_err": 0.01,
            "x_t": x0 + signal_offset,
            "x_t_raw": x0 + signal_offset,
            "x_t_err": 0.04,
            "y_t": y0,
            "y_t_raw": y0,
            "y_t_err": 0.04,
            "a_fwhm": 5.0,
            "b_fwhm": 5.5,
            "angle": 0.0,
            "sig2noise": 100.0 - uid,
        })
    table = Table(rows=rows)
    table.meta["obsnum"] = 42
    table.meta["beammap_direction_mode"] = mode
    table.write(path, format="ascii.ecsv")


def make_fits(
    path: Path,
    signal_offset: float,
    kernel_offset: float,
    count: int = 4,
    include_kernel: bool = True,
) -> None:
    yy, xx = np.mgrid[:161, :161]
    world_x = -(xx - 80.0)
    world_y = yy - 80.0
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [fits.PrimaryHDU()]
    for uid in range(count):
        x0, y0 = 12.0 * uid - 18.0, 8.0 * (uid % 2) - 4.0
        sx = x0 + signal_offset
        # The source has a compact nucleus plus a mostly vertical jet.  The
        # entire morphology receives the same imposed directional translation.
        nucleus = np.exp(-0.5 * (((world_x - sx) / 2.2) ** 2 + ((world_y - y0) / 2.4) ** 2))
        jet = 0.20 * np.exp(
            -0.5 * (((world_x - sx) / 2.8) ** 2 + ((world_y - (y0 + 9.0)) / 5.5) ** 2)
        )
        signal = nucleus + jet
        kx = x0 + kernel_offset
        kernel = np.exp(
            -0.5 * (((world_x - kx) / 2.0) ** 2 + ((world_y - y0) / 2.1) ** 2)
        )
        weight = np.ones_like(signal)
        planes = [
            ("signal", signal, "mJy/beam"),
            ("weight", weight, "1/(mJy/beam)^2"),
        ]
        if include_kernel:
            planes.append(("kernel", kernel, "mJy/beam"))
        for kind, data, unit in planes:
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


def prepare_case(
    root: Path,
    kernel_moves: bool,
    missing_kernel_mode: str | None = None,
) -> tuple[Path, Path]:
    raw = root / "reduced" / "redu00" / "42" / "raw"
    raw.mkdir(parents=True)
    apt = raw / "apt_commissioning_beammap_42_citlali.ecsv"
    offsets = {"standard": 0.0, "left": -2.0, "right": 2.0}
    suffixes = {"standard": "", "left": "_left", "right": "_right"}
    for mode, offset in offsets.items():
        make_apt(apt.with_name(apt.stem + suffixes[mode] + ".ecsv"), mode, offset)
        make_fits(
            raw / f"toltec_commissioning_a1100_beammap_42_citlali{suffixes[mode]}.fits",
            offset,
            offset if kernel_moves else 0.0,
            include_kernel=mode != missing_kernel_mode,
        )
    make_registry(raw / "beammap_direction_scan_registry_all.csv")
    selection = root / "selected_detectors.ecsv"
    Table(rows=[{"uid": uid} for uid in range(4)]).write(
        selection, format="ascii.ecsv"
    )
    return raw, selection


class SplitDirectionTransferTest(unittest.TestCase):
    def run_case(self, kernel_moves: bool) -> tuple[Table, dict[str, object], Path]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        _, selection = prepare_case(root, kernel_moves)
        output = root / "transfer"
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(root / "mpl")
        env["XDG_CACHE_HOME"] = str(root / "xdg")
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--reduction-root", str(root),
                "--selection", str(selection),
                "--output", str(output),
                "--minimum-clean-detectors", "2",
                "--half-width-arcsec", "20",
            ],
            env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        registration = Table.read(output / "stack_registration.ecsv")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(
            len(PdfReader(output / "split_direction_transfer_o42_a1100.pdf").pages),
            2,
        )
        self.assertTrue((output / "SHA256SUMS").is_file())
        return registration, manifest, output

    @staticmethod
    def row(table: Table, family: str, region: str):
        matches = table[(table["family"] == family) & (table["region"] == region)]
        if len(matches) != 1:
            raise AssertionError(f"expected one row for {family}/{region}")
        return matches[0]

    def test_signal_moves_while_retained_kernel_stays_centered(self) -> None:
        registration, manifest, output = self.run_case(kernel_moves=False)
        nuclear = self.row(registration, "signal", "nuclear_core")
        combined = self.row(registration, "signal", "core_plus_vertical_jet")
        kernel = self.row(registration, "kernel", "kernel_core")
        self.assertAlmostEqual(nuclear["delta_parallel_right_minus_left_arcsec"], 4.0, delta=0.2)
        self.assertAlmostEqual(combined["delta_parallel_right_minus_left_arcsec"], 4.0, delta=0.3)
        self.assertAlmostEqual(kernel["delta_parallel_right_minus_left_arcsec"], 0.0, delta=0.1)
        self.assertFalse(manifest["selection"]["uses_kernel_or_directional_result_for_membership"])
        self.assertEqual(manifest["position_frame"], "raw_altaz_detector_map")
        self.assertEqual(len(Table.read(output / "detector_kernel_metrics.ecsv")), 4)
        decision = json.loads((output / "diagnostic_decision.json").read_text())
        self.assertEqual(
            decision["classification"],
            "signal_shift_with_centered_downstream_transfer_kernel",
        )
        self.assertEqual(
            decision["downstream_filtering_artifact_disposition"],
            "strongly_disfavored_within_kernel_scope",
        )

    def test_signal_and_retained_kernel_move_together(self) -> None:
        registration, _, output = self.run_case(kernel_moves=True)
        nuclear = self.row(registration, "signal", "nuclear_core")
        kernel = self.row(registration, "kernel", "kernel_core")
        self.assertAlmostEqual(nuclear["delta_parallel_right_minus_left_arcsec"], 4.0, delta=0.2)
        self.assertAlmostEqual(kernel["delta_parallel_right_minus_left_arcsec"], 4.0, delta=0.2)
        decision = json.loads((output / "diagnostic_decision.json").read_text())
        self.assertEqual(
            decision["classification"],
            "signal_and_downstream_transfer_kernel_comove",
        )
        self.assertEqual(
            decision["downstream_filtering_artifact_disposition"],
            "favored_within_kernel_scope",
        )

    def test_missing_retained_kernel_fails_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, selection = prepare_case(root, False, missing_kernel_mode="left")
            output = root / "transfer"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--reduction-root", str(root),
                    "--selection", str(selection),
                    "--output", str(output),
                    "--minimum-clean-detectors", "2",
                ],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required retained kernel extension is missing", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
