#!/usr/bin/env python3
"""Fixed four-arm 152390 support comparison; no policy search or production route."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import j1

from run_rtc_notch_recovery import binding, write_json
from rtc_wide_notch_experiment import IDS

ARMS = ("control", "low-only", "high-only", "both")
ASEC = 206264.80624709636


def runs(mask):
    change = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def airy(xy, center):
    z = np.pi * 50 / (299792458 / 150e9) / ASEC * np.linalg.norm(xy - center, axis=-1)
    a = np.ones_like(z)
    np.divide(2 * j1(z), z, out=a, where=z != 0)
    return 5 * a * a  # linear diagnostic scale: 5 mJy/beam compact source


def prepare(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.input.read_text())
    channels = [d["channel"] for d in cfg["detectors"]]
    n = len(np.fromfile(a.prior / "native-full/native-time.f64", "<f8"))
    unix = np.fromfile(a.prior / "native-unix.f64", "<f8")
    t = unix - unix[0]
    dt = 0.008192
    knots = np.fromfile(a.knots / "telescope-knots.f64", "<f8") - unix[0]
    geometry = np.fromfile(a.geometry, "<f8").reshape(n, 4)
    if not np.allclose(geometry[:, 0], unix, rtol=0, atol=2e-7):
        raise ValueError("AST geometry/native registration differs")
    causes = np.fromfile(a.control / "269-input-causes.u8", "u1")
    old_keep = np.fromfile(a.control / "269-state.u8", "u1") & 3 == 3

    # Selection uses only original exclusions, motion and support, never results.
    def gap(kind):
        choices = [
            (lo, hi)
            for lo, hi in runs(causes == kind)
            if 3000 < lo < n - 3000
            and np.isin(causes[lo - 400 : hi + 400], [0, 7, 8]).all()
        ]
        if not choices:
            raise ValueError("missing predeclared speed-boundary source case")
        return choices[len(choices) // 2]

    low, high = gap(7), gap(8)
    quiet = np.flatnonzero(old_keep & (geometry[:, 1] > 40) & (geometry[:, 1] < 90))
    chosen = [
        ("low-boundary", low[0] - 10),
        ("high-boundary", high[0] - 10),
        ("maximum-speed", int(np.nanargmax(geometry[:, 1]))),
        ("quiet", int(quiet[len(quiet) // 2])),
    ]
    apt = np.fromfile(a.prior / "geometry-02/apt.f64", "<f8").reshape(491, 6)
    point269 = (
        np.fromfile(a.knots / "geometry-knots/pointing-269.f64", "<f8").reshape(-1, 2)
        * ASEC
    )
    components = []
    for name, row in chosen:
        row = int(row)
        center = [float(np.interp(t[row], knots, point269[:, k])) for k in range(2)]
        components.append(
            dict(
                identity=name,
                native_row=row,
                time=float(t[row]),
                center_arcsec=center,
                speed_arcsec_per_second=float(geometry[row, 1]),
                kind="compact",
                peak_mJy_per_beam=5,
            )
        )
    probes = [dict(identity=c["identity"], detectors=[]) for c in components]
    probes += [
        dict(identity="known-line", detectors=[]),
        dict(identity="white-noise", detectors=[]),
    ]
    line = np.fromfile(
        a.prior / "real-01/known-added-line-selected.f64", "<f8"
    ).reshape(n, 12, 2)
    convergence = []
    for d in channels:
        xy = (
            np.fromfile(a.knots / f"geometry-knots/pointing-{d}.f64", "<f8").reshape(
                -1, 2
            )
            * ASEC
        )
        for comp, probe in zip(components, probes):
            # Continuous piecewise-linear sky trajectory is defined on actual
            # ~50Hz telescope knots, not on native detector sample centers.
            # All native occurrences, including excluded support-only times,
            # receive the astronomical signal before any Apply mask is used.
            values = {}
            for sub in (64, 128):
                y = np.empty(n)
                offsets = ((np.arange(sub) + 0.5) / sub - 0.5) * dt
                for first in range(0, n, 4096):
                    stop = min(n, first + 4096)
                    times = t[first:stop, None] + offsets
                    if times.min() < knots[0] or times.max() > knots[-1]:
                        raise ValueError("integration extends beyond known trajectory")
                    p = np.stack(
                        [np.interp(times, knots, xy[:, k]) for k in range(2)], axis=-1
                    )
                    y[first:stop] = np.mean(airy(p, comp["center_arcsec"]), axis=1)
                values[sub] = y
            delta = np.column_stack([values[128], 0.2 * values[128]]) / apt[d, 2]
            path = out / f"{d}-{comp['identity']}.f64"
            delta.astype("<f8").tofile(path)
            probe["detectors"].append(dict(channel=d, **binding(path)))
            convergence.append(
                dict(
                    channel=d,
                    identity=comp["identity"],
                    maximum_64_128_difference_mJy=float(
                        np.max(abs(values[64] - values[128]))
                    ),
                    rms_64_128_difference_mJy=float(
                        np.sqrt(np.mean((values[64] - values[128]) ** 2))
                    ),
                )
            )
        for probe, delta in [
            (probes[-2], line[:, IDS.index(d)]),
            (probes[-1], np.random.default_rng(152390 + d).normal(0, 1e-6, (n, 2))),
        ]:
            path = out / f"{d}-{probe['identity']}.f64"
            delta.astype("<f8").tofile(path)
            probe["detectors"].append(dict(channel=d, **binding(path)))
    # Fine sampled time truth around the fastest crossing: quantify power
    # above native and F2 Nyquist before/after the provisional uniform exposure.
    # Windowed local diagnostic; it cannot qualify unknown hardware averaging.
    comp = next(c for c in components if c["identity"] == "maximum-speed")
    from scipy import signal

    sub = 128
    fine_dt = dt / sub
    fine_t = np.arange(comp["time"] - 8, comp["time"] + 8, fine_dt)
    p = np.column_stack([np.interp(fine_t, knots, point269[:, k]) for k in range(2)])
    y = airy(p, comp["center_arcsec"])
    averaged = signal.fftconvolve(y, np.ones(sub) / sub, mode="same")
    fractions = {}
    f = np.fft.rfftfreq(len(y), fine_dt)
    for name, v in [("instantaneous", y), ("provisional_uniform_exposure", averaged)]:
        power = abs(np.fft.rfft(v * signal.windows.tukey(len(v), 0.25))) ** 2
        fractions[name] = {
            label: float(power[f > hz].sum() / power.sum())
            for label, hz in [
                ("above_native_nyquist", 1 / (2 * dt)),
                ("above_F2_nyquist", 1 / (4 * dt)),
            ]
        }
    np.column_stack([fine_t, y, averaged]).astype("<f8").tofile(
        out / "maximum-speed-fine-truth.f64"
    )
    for arm in ARMS:
        write_json(
            out / f"{arm}.json",
            dict(cfg, speed_support_comparison=arm, fixed_plan_injections=probes),
        )
    write_json(
        out / "design.json",
        dict(
            components=components,
            probes=probes,
            interval_seconds=dt,
            nominal_interval=dt,
            channels=channels,
            input=binding(a.input),
            original_control=str(a.control),
            original_prior=str(a.prior),
            knot_times=binding(a.knots / "telescope-knots.f64"),
            knot_pointings=[
                binding(a.knots / f"geometry-knots/pointing-{d}.f64") for d in channels
            ],
            ast_geometry=binding(a.geometry),
            convergence=convergence,
            fine_power_fractions=fractions,
            low_excursion=list(map(int, low)),
            high_excursion=list(map(int, high)),
            readout="provisional centered uniform .008192s exposure;128 midpoint samples;64 convergence comparison;hardware response unknown",
            trajectory="continuous piecewise-linear own-detector sky positions at actual telescope knots;no detector-native interpolation used as truth",
            r_ratio=0.2,
            r_ratio_scope="diagnostic paired operator probe, not a measured optical r response or donor policy",
            source_retained_during_speed_exclusions=True,
            no_PCA_population_change=True,
            production=False,
            FRUIT=False,
        ),
    )
    print("Prepared fixed comparison, source, known-line and seeded noise probes")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--knots", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
