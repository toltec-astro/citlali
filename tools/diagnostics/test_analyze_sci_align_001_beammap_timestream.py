#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
from astropy.table import Table


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_beammap_timestream as target  # noqa: E402


PROTOCOL = Path(__file__).resolve().parents[2] / (
    "validation/sci_align_001_beammap_timestream_2026-08-11/"
    "frozen_protocol.json"
)


def synthetic_observation(tau_sec: float = 0.013) -> target.PreparedObservation:
    protocol = target.load_protocol(PROTOCOL)
    rng = np.random.default_rng(20260811)
    detector_count = 12
    uid = 2000 + np.arange(detector_count)
    network = np.arange(detector_count) % 6
    center_x = np.linspace(-18.0, 18.0, detector_count)
    center_y = 2.0 * np.sin(np.linspace(0.0, 2.0 * math.pi, detector_count))
    geometry = target.DetectorGeometry(
        uid=uid,
        network=network,
        center_x_arcsec=center_x,
        center_y_arcsec=center_y,
        major_fwhm_arcsec=np.linspace(8.0, 10.0, detector_count),
        minor_fwhm_arcsec=np.linspace(5.0, 6.0, detector_count),
        angle_rad=np.linspace(-0.2, 0.2, detector_count),
    )
    scans = []
    scored = 0
    support = 0
    for scan_row in range(12):
        full_time = np.arange(0.0, 5.001, 0.02)
        direction = -1.0 if scan_row % 2 == 0 else 1.0
        speed = 42.0 + 2.5 * scan_row
        x = direction * speed * (full_time - 2.5)
        y = np.full(full_time.shape, 0.2 * math.sin(scan_row))
        az = x / target.RAD_TO_ARCSEC
        alt = y / target.RAD_TO_ARCSEC
        vx, vy = target.pointing.scan_velocity(full_time, az, alt)
        common = (
            (full_time - 0.05 >= full_time[0])
            & (full_time + 0.05 <= full_time[-1])
        )
        recorded = full_time[common]
        x_ref = x[common]
        y_ref = y[common]
        x_signal = np.interp(recorded + tau_sec, full_time, x)
        y_signal = np.interp(recorded + tau_sec, full_time, y)
        template = target.detector_template(
            x_signal,
            y_signal,
            center_x[None, :],
            center_y[None, :],
            geometry,
        )
        u = target.pointing.normalized_scan_time(recorded)
        baseline = 0.02 * np.cos(0.1 * uid + scan_row)
        signal = (
            template * np.linspace(2.0, 3.0, detector_count)[None, :]
            + baseline[None, :]
            + 0.0005 * rng.normal(size=template.shape)
        )
        valid = np.ones(signal.shape, dtype=bool)
        radius = np.hypot(
            x_ref[:, None] - center_x[None, :],
            y_ref[:, None] - center_y[None, :],
        )
        score = valid & (radius <= 25.0)
        offsource = valid & (radius >= 40.0)
        residual, _ = target.pointing.fit_offsource_baseline(
            signal, valid, offsource, u, "constant"
        )
        scans.append(target.PreparedScan(
            scan_row=scan_row,
            output_scan_index=scan_row + 1,
            full_time=full_time,
            full_az=az,
            full_alt=alt,
            full_pointing_az=np.zeros(full_time.shape),
            full_pointing_alt=np.zeros(full_time.shape),
            full_velocity_x=vx,
            full_velocity_y=vy,
            recorded_time=recorded,
            geometry=geometry,
            ptc_weight=np.ones(detector_count),
            score_mask=score,
            residual_signal=residual,
            reference_x=x_ref,
            reference_y=y_ref,
        ))
        scored += int(np.count_nonzero(score))
        support += recorded.size
    return target.PreparedObservation(
        obsnum=999999,
        ptc_path=Path("synthetic.nc"),
        standard_apt_path=Path("synthetic.ecsv"),
        scans=scans,
        scan_axis_x=1.0,
        scan_axis_y=0.0,
        selected_detector_count=detector_count,
        used_detector_count=detector_count,
        eligible_networks=tuple(range(6)),
        common_support_sample_count=support,
        scored_value_count=scored,
        objective_normalization=float(np.mean([
            np.nanmean(np.where(scan.score_mask, scan.residual_signal ** 2, np.nan))
            for scan in scans
        ])),
        protocol=protocol,
    )


class BeammapTimestreamTest(unittest.TestCase):
    def test_freeze_and_prepare_small_full_ptc_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apt_path = root / "apt_commissioning_beammap_42_citlali.ecsv"
            rows = []
            for uid, center, nw in ((10, -8.0, 0), (11, 8.0, 1)):
                rows.append({
                    "uid": uid,
                    "array": 0,
                    "nw": nw,
                    "flag": 0,
                    "flag2": 0,
                    "amp": 2.0,
                    "sig2noise": 50.0 - uid,
                    "x_t": center,
                    "x_t_raw": center,
                    "x_t_err": 0.1,
                    "y_t": 0.0,
                    "y_t_raw": 0.0,
                    "y_t_err": 0.1,
                    "a_fwhm": 9.0,
                    "b_fwhm": 6.0,
                    "angle": 0.0,
                })
            apt = Table(rows=rows, meta={"obsnum": 42})
            apt.write(apt_path, format="ascii.ecsv")

            ptc_path = root / "beammap_42_ptc_timestream.nc"
            scan_size = 251
            scan_count = 8
            sample_count = scan_size * scan_count
            with netCDF4.Dataset(ptc_path, "w") as dataset:
                dataset.createDimension("n_pts", None)
                dataset.createDimension("n_scans", scan_count)
                dataset.createDimension("n_dets", 2)
                dataset.createDimension("n_scan_indices", 2)
                dataset.createDimension("n_raw_scan_indices", 4)
                variables = {
                    "signal": ("f8", ("n_pts", "n_dets")),
                    "flags": ("f8", ("n_pts", "n_dets")),
                    "weights": ("f8", ("n_scans", "n_dets")),
                    "scan_indices": ("i8", ("n_scans", "n_scan_indices")),
                    "raw_scan_indices": ("i8", ("n_scans", "n_raw_scan_indices")),
                    "output_scan_index": ("i8", ("n_scans",)),
                    "apt_array": ("i8", ("n_dets",)),
                    "apt_uid": ("i8", ("n_dets",)),
                    "apt_nw": ("i8", ("n_dets",)),
                    "TelTime": ("f8", ("n_pts",)),
                    "az_phys": ("f8", ("n_pts",)),
                    "alt_phys": ("f8", ("n_pts",)),
                    "pointing_offset_az": ("f8", ("n_pts",)),
                    "pointing_offset_alt": ("f8", ("n_pts",)),
                }
                created = {
                    name: dataset.createVariable(name, dtype, dimensions)
                    for name, (dtype, dimensions) in variables.items()
                }
                bounds = []
                signal = np.zeros((sample_count, 2))
                time = np.zeros(sample_count)
                az = np.zeros(sample_count)
                for scan in range(scan_count):
                    start = scan * scan_size
                    stop = start + scan_size - 1
                    bounds.append((start, stop))
                    local_time = np.arange(scan_size) * 0.02 + 10.0 * scan
                    x = (-1.0 if scan % 2 == 0 else 1.0) * 45.0 * (
                        local_time - np.mean(local_time)
                    )
                    time[start:stop + 1] = local_time
                    az[start:stop + 1] = x / target.RAD_TO_ARCSEC
                    for detector, center in enumerate((-8.0, 8.0)):
                        signal[start:stop + 1, detector] = (
                            2.0 * np.exp(-0.5 * ((x - center) / 3.0) ** 2) + 0.1
                        )
                bounds_array = np.asarray(bounds, dtype=np.int64)
                created["signal"][:] = signal
                created["flags"][:] = 0.0
                created["weights"][:] = 1.0
                created["scan_indices"][:] = bounds_array
                created["raw_scan_indices"][:] = np.column_stack([bounds_array, bounds_array])
                created["output_scan_index"][:] = np.arange(1, scan_count + 1)
                created["apt_array"][:] = 0
                created["apt_uid"][:] = [10, 11]
                created["apt_nw"][:] = [0, 1]
                created["TelTime"][:] = time
                created["az_phys"][:] = az
                created["alt_phys"][:] = 0.0
                created["pointing_offset_az"][:] = 0.0
                created["pointing_offset_alt"][:] = 0.0

            frozen_root = root / "frozen"
            target.freeze(SimpleNamespace(
                protocol=PROTOCOL,
                ptc=ptc_path,
                standard_apt=apt_path,
                obsnum=42,
                output=frozen_root,
            ))
            frozen, protocol = target.load_frozen(frozen_root / "frozen_input.json")
            observation = target.prepare(frozen, protocol)
            self.assertEqual(observation.selected_detector_count, 2)
            self.assertEqual(observation.used_detector_count, 2)
            self.assertEqual(len(observation.scans), scan_count)
            self.assertEqual(target.coordinate_gate(observation)["status"], "pass")
            result_root = root / "result"
            target.run(SimpleNamespace(
                frozen_input=frozen_root / "frozen_input.json",
                output=result_root,
            ))
            target.pointing.verify_sha256s(result_root)
            result = target.json.loads((result_root / "result.json").read_text())
            self.assertEqual(
                result["coordinate_gate"]["semantics"],
                "telescope tangent pointing plus pointing offsets; detector APT offsets suppressed",
            )

    def test_coordinate_contract_suppresses_physical_detector_offsets(self) -> None:
        observation = synthetic_observation()
        scan = observation.scans[0]
        x, y, _, _ = scan.coordinates(0.0)
        np.testing.assert_allclose(x, scan.reference_x, atol=1.0e-12)
        np.testing.assert_allclose(y, scan.reference_y, atol=1.0e-12)
        self.assertEqual(target.coordinate_gate(observation)["status"], "pass")

    def test_common_lag_recovers_injected_complete_coordinate_shift(self) -> None:
        observation = synthetic_observation(tau_sec=0.013)
        fit = target.fit_model(observation, "lag")
        self.assertEqual(fit["status"], "success")
        self.assertGreater(fit["optimizer_iterations"], 0)
        self.assertAlmostEqual(fit["tau_ms"], 13.0, delta=0.25)
        self.assertAlmostEqual(
            fit["parameters"]["delta_x_arcsec"], 0.0, delta=0.05
        )

    def test_map_equivalent_has_opposite_sign_and_no_factor_of_two(self) -> None:
        direct_tau_sec = 0.013
        left_rate = -50.0
        right_rate = 50.0
        left_centroid = -left_rate * direct_tau_sec
        right_centroid = -right_rate * direct_tau_sec
        map_equivalent_sec = (
            (right_centroid - left_centroid) / (right_rate - left_rate)
        )
        self.assertAlmostEqual(map_equivalent_sec, -direct_tau_sec, places=15)
        self.assertAlmostEqual(
            0.5 * (right_centroid - left_centroid) / right_rate,
            map_equivalent_sec,
            places=15,
        )

    def test_optimizer_uses_millisecond_specific_finite_difference_step(self) -> None:
        np.testing.assert_allclose(
            target.optimizer_steps("joint"),
            [1.0e-4, 1.0e-4, 0.01, 1.0e-4],
        )

    def test_varying_scan_speed_distinguishes_lag_from_fixed_hysteresis(self) -> None:
        observation = synthetic_observation(tau_sec=0.013)
        lag = target.fit_model(observation, "lag")
        hysteresis = target.fit_model(observation, "scan_hysteresis")
        self.assertLess(lag["objective"], 0.25 * hysteresis["objective"])


if __name__ == "__main__":
    unittest.main()
