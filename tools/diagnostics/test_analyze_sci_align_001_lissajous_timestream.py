#!/usr/bin/env python3

from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sci_align_001_lissajous_timestream as target  # noqa: E402


PROTOCOL = Path(__file__).resolve().parents[2] / (
    "validation/sci_align_001_lissajous_timestream_2026-08-10/"
    "frozen_protocol.json"
)


def synthetic_observation(
    *,
    tau_sec: float = 0.0,
    x0: float = 2.5,
    y0: float = -1.75,
    h_az: float = 0.0,
    h_el: float = 0.0,
    seed: int = 42,
) -> target.PreparedObservation:
    protocol = target.load_protocol(PROTOCOL)
    beam = target.BeamGeometry(10.0, 7.5, 0.2)
    rng = np.random.default_rng(seed)
    scans: list[target.PreparedScan] = []
    all_uids: set[int] = set()
    support_count = 0
    scored_count = 0
    detector_count = 14
    apt_x = np.linspace(-4.0, 4.0, detector_count)
    apt_y = 2.0 * np.sin(np.linspace(0.0, 2.0 * math.pi, detector_count))
    uid = 1000 + np.arange(detector_count)
    network = np.arange(detector_count) % 7
    amplitudes = np.linspace(2.0, 4.0, detector_count)

    for scan_row in range(12):
        increments = 0.0195 + 0.0015 * rng.random(321)
        full_time = np.cumsum(increments)
        full_time -= full_time[0]
        phase = 0.37 * scan_row
        normalized = full_time / full_time[-1]
        x_tel = 62.0 * np.sin(2.0 * math.pi * normalized + phase)
        # Vary the second-axis phase independently across scans so the source
        # is crossed with both signs of each velocity component.
        y_tel = (1.0 if scan_row % 2 == 0 else -1.0) * 52.0 * np.sin(
            4.0 * math.pi * normalized + 0.31 + 0.73 * scan_row
        )
        elevation = np.full(full_time.shape, 0.65 + 0.01 * scan_row)
        az = x_tel / target.RAD_TO_ARCSEC
        alt = y_tel / target.RAD_TO_ARCSEC
        vx, vy = target.scan_velocity(full_time, az, alt)
        common = (
            (full_time - 0.05 >= full_time[0])
            & (full_time + 0.05 <= full_time[-1])
        )
        recorded = full_time[common]
        query = recorded + tau_sec
        x_shift = np.interp(query, full_time, x_tel)
        y_shift = np.interp(query, full_time, y_tel)
        elevation_shift = np.interp(query, full_time, elevation)
        vx_shift = np.interp(query, full_time, vx)
        vy_shift = np.interp(query, full_time, vy)
        ct = np.cos(elevation_shift)[:, None]
        st = np.sin(elevation_shift)[:, None]
        detector_x = (
            x_shift[:, None] + ct * apt_x[None, :] - st * apt_y[None, :]
        )
        detector_y = (
            y_shift[:, None] + ct * apt_y[None, :] + st * apt_x[None, :]
        )
        center_x = x0 + h_az * np.sign(vx_shift)
        center_y = y0 + h_el * np.sign(vy_shift)
        template = target.gaussian_beam(
            detector_x, detector_y, center_x[:, None], center_y[:, None], beam
        )
        u = target.normalized_scan_time(recorded)
        baseline_intercept = 0.1 * np.cos(0.2 * uid + scan_row)
        baseline_slope = 0.015 * np.sin(0.3 * uid + scan_row)
        common_noise = 0.002 * rng.normal(size=recorded.size)
        signal = (
            template * amplitudes[None, :]
            + baseline_intercept[None, :]
            + u[:, None] * baseline_slope[None, :]
            + common_noise[:, None]
            + 0.0005 * rng.normal(size=template.shape)
        )
        valid = np.ones(template.shape, dtype=bool)
        valid[rng.random(template.shape) < 0.01] = False
        # Fixed support is defined at recorded time, not injected tau.
        elevation_ref = elevation[common]
        ct_ref = np.cos(elevation_ref)[:, None]
        st_ref = np.sin(elevation_ref)[:, None]
        x_ref = x_tel[common, None] + ct_ref * apt_x[None, :] - st_ref * apt_y[None, :]
        y_ref = y_tel[common, None] + ct_ref * apt_y[None, :] + st_ref * apt_x[None, :]
        radius = np.hypot(x_ref, y_ref)
        score_mask = valid & (radius <= 35.0)
        offsource_mask = valid & (radius >= 45.0)
        residuals = {}
        coefficients = {}
        for mode in target.BASELINE_NAMES:
            residual, coefficient = target.fit_offsource_baseline(
                signal, valid, offsource_mask, u, mode
            )
            residuals[mode] = residual
            coefficients[mode] = coefficient
        scans.append(target.PreparedScan(
            scan_row=scan_row,
            output_scan_index=scan_row + 1,
            full_time=full_time,
            full_az=az,
            full_alt=alt,
            full_elevation=elevation,
            full_pointing_az=np.zeros(full_time.size),
            full_pointing_alt=np.zeros(full_time.size),
            full_velocity_x=vx,
            full_velocity_y=vy,
            recorded_time=recorded,
            apt_x=apt_x,
            apt_y=apt_y,
            detector_uid=uid,
            detector_network=network,
            ptc_weight=np.linspace(0.8, 1.2, detector_count),
            valid=valid,
            score_mask=score_mask,
            offsource_mask=offsource_mask,
            residual_by_baseline=residuals,
            baseline_coefficients=coefficients,
            reference_x=x_ref,
            reference_y=y_ref,
        ))
        all_uids.update(map(int, uid))
        support_count += recorded.size
        scored_count += int(np.count_nonzero(score_mask))
    return target.PreparedObservation(
        obsnum=999999,
        ptc_path=Path("synthetic.nc"),
        ppt_path=Path("synthetic.ecsv"),
        ppt_x_arcsec=0.0,
        ppt_y_arcsec=0.0,
        beam=beam,
        scans=scans,
        eligible_uid_count=len(all_uids),
        eligible_networks=tuple(range(7)),
        common_support_sample_count=support_count,
        scored_value_count=scored_count,
        protocol=protocol,
    )


class LissajousTimestreamTest(unittest.TestCase):
    def test_run_monitor_appends_with_monotonic_event_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = target.RunMonitor(root)
            first.emit("first")
            second = target.RunMonitor(root)
            second.emit("second")
            rows = [
                json.loads(line)
                for line in (root / "progress.jsonl").read_text().splitlines()
            ]
        self.assertEqual([row["event_index"] for row in rows], [0, 1])

    def test_run_monitor_records_progress_and_enforces_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monitor = target.RunMonitor(root, maximum_wall_seconds=0.001)
            monitor.emit("run_start", stage="test")
            monitor.started_monotonic -= 1.0
            with self.assertRaisesRegex(target.ContractError, "maximum wall time"):
                monitor.check_deadline("synthetic_objective")
            rows = [
                json.loads(line)
                for line in (root / "progress.jsonl").read_text().splitlines()
            ]
        self.assertEqual(
            [row["event"] for row in rows],
            ["run_start", "runtime_limit_exceeded"],
        )
        self.assertEqual(rows[-1]["stage"], "synthetic_objective")

    def test_runtime_audit_groups_stages_attempts_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "event": "optimizer_attempt_end",
                    "elapsed_seconds": 3.0,
                    "fit_label": "full.lag",
                    "model": "lag",
                    "attempt_index": 0,
                    "status": "converged",
                    "duration_seconds": 2.5,
                    "objective": 1.0,
                    "optimizer_iterations": 4,
                    "optimizer_function_evaluations": 20,
                    "optimizer_gradient_evaluations": 5,
                    "optimizer_message": "ok",
                },
                {
                    "event": "optimizer_fallback",
                    "elapsed_seconds": 3.1,
                    "fit_label": "bootstrap.timestream[0]",
                },
                {
                    "event": "stage_end",
                    "elapsed_seconds": 7.0,
                    "stage": "full_model_fits",
                    "status": "success",
                    "duration_seconds": 6.5,
                },
            ]
            progress = root / "progress.jsonl"
            progress.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n"
            )
            audit = target.runtime_audit(progress)
        self.assertEqual(audit["event_count"], 3)
        self.assertEqual(audit["optimizer_attempt_count"], 1)
        self.assertEqual(audit["optimizer_fallback_count"], 1)
        self.assertEqual(audit["family_rows"][0]["family"], "full")
        self.assertEqual(
            audit["family_rows"][0]["total_function_evaluations"], 20
        )
        self.assertEqual(audit["stage_rows"][0]["stage"], "full_model_fits")

    def test_failed_finite_optimizer_is_rejected_and_initial_fit_retries(self) -> None:
        observation = synthetic_observation()
        initial = np.asarray([0.0, 0.0, 11.728])
        failed = SimpleNamespace(
            success=False,
            fun=10.0,
            x=initial,
            message="ABNORMAL",
            nit=0,
        )
        converged = [
            SimpleNamespace(
                success=True,
                fun=objective,
                x=np.asarray([0.0, 0.0, tau_ms]),
                message="CONVERGENCE",
                nit=5,
            )
            for objective, tau_ms in ((9.0, -3.0), (8.0, 7.5), (8.5, 18.0))
        ]
        with tempfile.TemporaryDirectory() as directory:
            monitor = target.RunMonitor(Path(directory), maximum_wall_seconds=60.0)
            with patch.object(target, "minimize", side_effect=[failed, *converged]):
                fit = target.fit_observation_model(
                    observation, "lag", initial=initial, monitor=monitor,
                    fit_label="synthetic_fallback",
                )
        self.assertEqual(fit["status"], "success", fit)
        self.assertTrue(fit["optimizer_initial_fallback_used"])
        self.assertEqual(fit["optimizer_initial_failure_messages"], ["ABNORMAL"])
        self.assertTrue(fit["optimizer_success"])
        self.assertAlmostEqual(fit["tau_ms"], 7.5)
        self.assertEqual(monitor.optimizer_attempt_count, 4)
        self.assertEqual(monitor.optimizer_fallback_count, 1)

    def test_stage_checkpoint_roundtrip_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target.write_json(root / "fit_gate.json", {"identity": "test"})
            target.write_checksums(
                root, ["fit_gate.json"], "FIT_GATE_SHA256SUMS"
            )
            state = target.load_stage_checkpoints(root)
            value = [{"tau_ms": -3.25, "objective": 10.5}]
            target.save_stage_checkpoint(
                root, state, "objective_profile", value
            )
            restored = target.load_stage_checkpoints(root)
            self.assertEqual(restored["values"]["objective_profile"], value)
            self.assertEqual(
                restored["completed_stages"], ["objective_profile"]
            )
            target.write_json(
                root / target.STAGE_CHECKPOINT_FILES["objective_profile"],
                [{"tau_ms": 99.0}],
            )
            with self.assertRaisesRegex(target.ContractError, "checksum mismatch"):
                target.load_stage_checkpoints(root)

    def test_fit_gate_structural_status_never_depends_on_tau(self) -> None:
        scan_rows = [{"best_weighted_mse": 1.0}]

        def fits(tau_ms: float) -> dict[str, dict[str, object]]:
            return {
                name: {
                    "status": "success",
                    "objective": 10.0 + index,
                    "boundary": False,
                    "tau_ms": tau_ms if name in {"lag", "joint"} else 0.0,
                }
                for index, name in enumerate(target.MODEL_NAMES)
            }

        negative = target.fit_gate_quality_summary(
            {"status": "pass"}, fits(-49.0), scan_rows
        )
        positive = target.fit_gate_quality_summary(
            {"status": "pass"}, fits(49.0), scan_rows
        )
        self.assertEqual(negative, positive)
        self.assertFalse(negative["tau_used_as_gate"])
        self.assertEqual(negative["automatic_structural_status"], "pass")

    def test_fit_gate_rejects_missing_scan_diagnostics(self) -> None:
        fits = {
            name: {
                "status": "success",
                "objective": 10.0,
                "boundary": False,
                "tau_ms": 0.0,
            }
            for name in target.MODEL_NAMES
        }
        quality = target.fit_gate_quality_summary(
            {"status": "pass"}, fits, []
        )
        self.assertEqual(quality["automatic_structural_status"], "fail")
        self.assertFalse(
            quality["structural_checks"]["scan_residual_metrics_available"]
        )

    def test_fit_gate_support_identity_survives_json_roundtrip(self) -> None:
        support = target.observation_support_summary(synthetic_observation())
        self.assertEqual(support, json.loads(json.dumps(support)))

    def test_direct_full_analysis_is_disabled(self) -> None:
        with self.assertRaisesRegex(target.ContractError, "run fit-gate"):
            target.analyze_observation(SimpleNamespace())

    def test_checkpointed_fit_is_numerically_identical_after_json_roundtrip(
        self,
    ) -> None:
        observation = synthetic_observation(tau_sec=-0.006)
        fit = target.fit_observation_model(
            observation, "lag", baseline_mode="linear"
        )
        restored = json.loads(json.dumps(fit))
        before = target.fit_to_optimizer_vector(fit, "lag", "fixed")
        after = target.fit_to_optimizer_vector(restored, "lag", "fixed")
        self.assertTrue(np.array_equal(before, after))
        before_objective = target.observation_objective(
            before, observation, "lag", "fixed", "linear"
        )
        after_objective = target.observation_objective(
            after, observation, "lag", "fixed", "linear"
        )
        self.assertEqual(before_objective, after_objective)

    def test_multimodal_bootstrap_cannot_converge_at_500(self) -> None:
        protocol = target.load_protocol(PROTOCOL)
        rng = np.random.default_rng(1234)
        values = np.concatenate([
            rng.normal(-8.0, 0.35, 250), rng.normal(8.0, 0.35, 250)
        ])
        converged, diagnostic = target.bootstrap_is_converged(values, protocol)
        self.assertFalse(converged)
        self.assertEqual(diagnostic["status"], "extend")
        self.assertTrue(diagnostic["multimodal"])

    def test_map_centroid_slope_has_opposite_coordinate_shift_sign(self) -> None:
        injected_coordinate_shift_ms = 12.0
        velocity = np.asarray([-40.0, -10.0, 20.0, 50.0])
        centroid = 3.0 - velocity * injected_coordinate_shift_ms / 1000.0
        slope = np.linalg.lstsq(
            np.column_stack([np.ones(velocity.size), velocity]),
            centroid,
            rcond=None,
        )[0][1]
        recovered = target.map_centroid_tau_to_coordinate_shift_ms(
            1000.0 * slope
        )
        self.assertAlmostEqual(recovered, injected_coordinate_shift_ms)

    def test_optimizer_tau_coordinate_is_milliseconds(self) -> None:
        parameters = target.parameter_dict(
            np.asarray([1.0, -2.0, 12.5]), "lag", "fixed"
        )
        self.assertEqual(parameters["x0_arcsec"], 1.0)
        self.assertEqual(parameters["y0_arcsec"], -2.0)
        self.assertEqual(parameters["tau_sec"], 0.0125)
        steps = target.optimizer_finite_difference_steps("lag", "fixed")
        self.assertTrue(np.array_equal(steps, np.asarray([1.0e-4, 1.0e-4, 0.01])))

    def test_wrap_safe_interpolation(self) -> None:
        time = np.asarray([0.0, 1.0, 2.0])
        angle = np.radians([179.0, -179.0, -177.0])
        result = target.interpolate_unwrapped(
            np.asarray([0.5, 1.5]), time, angle
        )
        self.assertTrue(np.allclose(np.degrees(result), [180.0, 182.0]))

    def test_common_support_is_invariant_across_lag_bound(self) -> None:
        observation = synthetic_observation()
        for scan in observation.scans:
            expected_shape = scan.reference_x.shape
            for tau in (-0.05, 0.0, 0.05):
                x, y, _, _ = scan.coordinates(tau)
                self.assertEqual(x.shape, expected_shape)
                self.assertEqual(y.shape, expected_shape)

    def test_injected_zero_negative_and_positive_lag(self) -> None:
        for injected in (0.0, -0.012, 0.009):
            with self.subTest(injected=injected):
                observation = synthetic_observation(tau_sec=injected)
                fit = target.fit_observation_model(
                    observation, "lag", baseline_mode="linear"
                )
                self.assertEqual(fit["status"], "success", fit)
                self.assertAlmostEqual(fit["tau_ms"], 1000.0 * injected, delta=0.8)

    def test_lag_is_distinct_from_static_source_offset(self) -> None:
        observation = synthetic_observation(
            tau_sec=-0.012, x0=5.25, y0=-4.5
        )
        fit = target.fit_observation_model(
            observation, "lag", baseline_mode="linear"
        )
        self.assertEqual(fit["status"], "success", fit)
        self.assertAlmostEqual(fit["tau_ms"], -12.0, delta=0.8)
        self.assertAlmostEqual(fit["parameters"]["x0_arcsec"], 5.25, delta=0.2)
        self.assertAlmostEqual(fit["parameters"]["y0_arcsec"], -4.5, delta=0.2)

    def test_pure_hysteresis(self) -> None:
        observation = synthetic_observation(h_az=-1.1, h_el=0.55)
        fit = target.fit_observation_model(
            observation, "hysteresis", baseline_mode="linear"
        )
        self.assertEqual(fit["status"], "success", fit)
        self.assertAlmostEqual(fit["parameters"]["h_az_arcsec"], -1.1, delta=0.15)
        self.assertAlmostEqual(fit["parameters"]["h_el_arcsec"], 0.55, delta=0.15)

    def test_joint_lag_and_hysteresis(self) -> None:
        observation = synthetic_observation(
            tau_sec=-0.012, h_az=-0.8, h_el=0.4
        )
        fit = target.fit_observation_model(
            observation, "joint", baseline_mode="linear"
        )
        self.assertEqual(fit["status"], "success", fit)
        self.assertAlmostEqual(fit["tau_ms"], -12.0, delta=1.5)
        self.assertAlmostEqual(fit["parameters"]["h_az_arcsec"], -0.8, delta=0.2)
        self.assertAlmostEqual(fit["parameters"]["h_el_arcsec"], 0.4, delta=0.2)

    def test_derivative_crosscheck_agrees_for_small_injected_lag(self) -> None:
        observation = synthetic_observation(tau_sec=-0.004)
        no_lag = target.fit_observation_model(
            observation, "constant", baseline_mode="linear"
        )
        derivative = target.derivative_tau_estimate(
            observation, no_lag, baseline_mode="linear"
        )
        self.assertEqual(derivative["status"], "success")
        self.assertAlmostEqual(float(derivative["tau_ms"]), -4.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
