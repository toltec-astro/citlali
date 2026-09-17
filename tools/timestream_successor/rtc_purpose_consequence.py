#!/usr/bin/env python3
"""Bounded purpose-aware diagnostics of existing RTC filters; no qualification."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from run_rtc_notch_recovery import binding, write_json
from rtc_speed_support_replay import ASEC, runs
from rtc_wide_notch_experiment import IDS

FREQUENCY = 11.006255844577186
PHASES = np.arange(4) * np.pi / 2


def crossing(t, xy, center, when, radius, near):
    distance = np.linalg.norm(xy - center, axis=1)
    nearby = np.flatnonzero(abs(t - when) <= 10)
    index = nearby[np.argmin(distance[nearby])]
    if distance[index] > near:
        lo, hi = np.searchsorted(t, [when - 0.5, when + 0.5])
        return [int(lo), int(hi)], "no principal crossing at this detector pointing"
    lo, hi = next((lo, hi) for lo, hi in runs(distance <= radius) if lo <= index < hi)
    return [int(lo), int(hi)], "" if lo > 0 and hi < len(
        t
    ) else "crossing truncated by observation boundary"


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.input.read_text())
    speed = json.loads(args.design.read_text())
    prior = Path(speed["original_prior"])
    unix = np.fromfile(prior / "native-unix.f64", "<f8")
    t = unix - unix[0]
    n = len(t)
    channels = [d["channel"] for d in baseline["detectors"]]
    knot_path = Path(speed["knot_times"]["path"])
    assert binding(knot_path)["sha256"] == speed["knot_times"]["sha256"]
    knots = np.fromfile(knot_path, "<f8") - unix[0]
    apt = np.fromfile(prior / "geometry-02/apt.f64", "<f8").reshape(491, 6)
    line = np.fromfile(prior / "real-01/known-added-line-selected.f64", "<f8").reshape(
        n, 12, 2
    )
    quiet = next(c for c in speed["components"] if c["identity"] == "quiet")
    fastest = next(c for c in speed["components"] if c["identity"] == "maximum-speed")
    models = []
    for c in (quiet, fastest):
        for scale in (1, 20):
            models.append(
                dict(
                    c,
                    identity=c["identity"] + ("-faint" if scale == 1 else "-bright"),
                    scale=scale,
                )
            )
    for width in (25.451808, 50.903616):
        models.append(
            dict(
                quiet,
                identity="extended-" + str(round(width)),
                kind="extended",
                FWHM_arcsec=width,
                scale=1,
            )
        )
    positions = []
    sources = {m["identity"]: [] for m in models}
    line_bindings = {}
    independent = np.random.default_rng(15239012).uniform(-np.pi, np.pi, len(channels))
    line_levels = []
    convergence = []
    for di, d in enumerate(channels):
        kp = next(
            x
            for x in speed["knot_pointings"]
            if Path(x["path"]).name == f"pointing-{d}.f64"
        )
        assert binding(Path(kp["path"]))["sha256"] == kp["sha256"]
        xy_knots = np.fromfile(kp["path"], "<f8").reshape(-1, 2) * ASEC
        xy = np.column_stack([np.interp(t, knots, xy_knots[:, k]) for k in range(2)])
        path = args.output / f"positions-{d}.f64"
        xy.astype("<f8").tofile(path)
        positions.append(dict(channel=d, **binding(path)))
        for m in models:
            if m["kind"] == "compact":
                original_probe = next(
                    p
                    for p in baseline["fixed_plan_injections"]
                    if p["identity"] == m["identity"].rsplit("-", 1)[0]
                )
                src = original_probe["detectors"][di]
                assert (
                    src["channel"] == d
                    and binding(Path(src["path"]))["sha256"] == src["sha256"]
                )
                values = np.fromfile(src["path"], "<f8").reshape(n, 2) * m["scale"]
                radius = 3.8317059702075125 / np.pi * (299792458 / 150e9) / 50 * ASEC
                near = radius
            else:
                sigma = m["FWHM_arcsec"] / np.sqrt(8 * np.log(2))
                result = {}
                for sub in (64, 128):
                    v = np.zeros(n)
                    for k in range(sub):
                        shifted = t + ((k + 0.5) / sub - 0.5) * 0.008192
                        if shifted[0] < knots[0] or shifted[-1] > knots[-1]:
                            raise ValueError("unknown pointing outside knot coverage")
                        xx = (
                            np.interp(shifted, knots, xy_knots[:, 0])
                            - m["center_arcsec"][0]
                        )
                        yy = (
                            np.interp(shifted, knots, xy_knots[:, 1])
                            - m["center_arcsec"][1]
                        )
                        v += np.exp(-0.5 * (xx * xx + yy * yy) / sigma**2) / sub
                    result[sub] = v * 5 / apt[d, 2]
                values = np.column_stack([result[128], 0.2 * result[128]])
                convergence.append(
                    dict(
                        channel=d,
                        model=m["identity"],
                        max_fractional_64_128=float(
                            np.max(abs(result[64] - result[128]))
                            / np.max(abs(result[128]))
                        ),
                    )
                )
                radius, near = 5 * sigma, m["FWHM_arcsec"] / 2
            path = args.output / f"source-{m['identity']}-{d}.f64"
            values.astype("<f8").tofile(path)
            win, reason = crossing(
                t, xy, np.array(m["center_arcsec"]), m["time"], radius, near
            )
            center_time = float(np.mean(t[win[0] : win[1]]))
            ring = np.searchsorted(t, [center_time - 3, center_time + 3])
            ring = [int(min(win[0], ring[0])), int(max(win[1], ring[1]))]
            sources[m["identity"]].append(
                dict(
                    channel=d,
                    source=binding(path),
                    window=win,
                    ringing_window=ring,
                    window_unavailable=reason,
                )
            )
        rms = np.sqrt(np.mean(line[:, IDS.index(d), :] ** 2, axis=0))
        line_levels.append(dict(channel=d, native_rms_xr=rms.tolist()))
        for kind in ("shared", "independent"):
            for pi, phase in enumerate(PHASES):
                relative = phase + (independent[di] if kind == "independent" else 0)
                wave = (
                    np.sqrt(2)
                    * np.sin(2 * np.pi * FREQUENCY * t + relative)[:, None]
                    * rms
                )
                path = args.output / f"line-{kind}-{pi}-{d}.f64"
                wave.astype("<f8").tofile(path)
                line_bindings[kind, pi, d] = binding(path)
    cases = []
    missing = [
        "numerical purpose-specific acceptance requirement",
        "population statistical uncertainty from one noise realization",
        "physical optical r response",
        "POINT two-dimensional fitted offset and repeatability",
        "BEAM fitted width, wings and calibrated normalization",
        "OOF defocused template and focus/wavefront evaluator",
    ]
    for m in models:
        basic = dict(
            identity=m["identity"],
            controlled=False,
            source_model=(
                "50m150GHz Airy"
                if m["kind"] == "compact"
                else f"apparent circular Gaussian FWHM={m['FWHM_arcsec']}arcsec"
            ),
            regime=f"nominal {5 * m['scale']} scale / fixed prior factor; fractional native-unit probe, not calibrated flux",
            purposes=["science", "pointing", "beammap"]
            if m["kind"] == "compact"
            else ["science"],
            required_unavailable=missing,
            detectors=sources[m["identity"]],
        )
        cases.append(basic)
        if m["identity"] not in ("quiet-faint", "quiet-bright", "extended-25"):
            continue
        noise_probe = next(
            p
            for p in baseline["fixed_plan_injections"]
            if p["identity"] == "white-noise"
        )
        for kind in ("shared", "independent"):
            for pi in range(4):
                case = copy.deepcopy(basic)
                case.update(
                    identity=m["identity"] + f"-{kind}-{pi}",
                    controlled=True,
                    phase_rad=float(PHASES[pi]),
                    line_population=kind,
                )
                for di, entry in enumerate(case["detectors"]):
                    noise = noise_probe["detectors"][di]
                    assert binding(Path(noise["path"]))["sha256"] == noise["sha256"]
                    entry.update(
                        noise=noise, line=line_bindings[kind, pi, entry["channel"]]
                    )
                cases.append(case)
    study = dict(
        schema="rtc-purpose-consequence-study-v1",
        models=models,
        cases=cases,
        positions=positions,
        geometry_identity="actual-telescope-knots:sha256:"
        + speed["knot_times"]["sha256"],
        geometry_inputs=speed["knot_pointings"],
        baseline_input=binding(args.input),
        cadence_seconds=0.008192,
        factor=2,
        phase_native_rows=0,
        line_frequency_hz=FREQUENCY,
        line_levels=line_levels,
        independent_phase_seed=15239012,
        independent_phase_radians=independent.tolist(),
        relative_phases=PHASES.tolist(),
        noise="existing white-noise probe: independent seed152390+channel; native sigma1e-6; one realization; no calibrated sensitivity claim",
        source_readout=speed["readout"],
        r_semantics=speed["r_ratio_scope"],
        prior_factor_identity=binding(prior / "geometry-02/apt.f64"),
        measured_line_identity=binding(prior / "real-01/known-added-line-selected.f64"),
        extended_quadrature_convergence=convergence,
        source_qualification=False,
        selected_coefficients_sha256=hashlib.sha256(
            np.array(baseline["finite_notch"], "<f8").tobytes()
        ).hexdigest(),
        lowpass_coefficients_sha256=hashlib.sha256(
            np.array(baseline["fir"], "<f8").tobytes()
        ).hexdigest(),
    )
    write_json(args.output / "study.json", study)
    baseline["consequence_study"] = binding(args.output / "study.json")
    write_json(args.output / "input.json", baseline)
    print(
        f"Prepared {len(models)} transfer cases and {len(cases) - len(models)} controlled phase cases; 2 frozen treatments; no thresholds"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
