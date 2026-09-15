#!/usr/bin/env python3
"""Bounded, explicitly selected RTC notch experiment; no runtime selection policy.

prepare freezes three cases/coefficient arrays from sealed audit evidence.
inject uses the first run's accepted AST mapping to prepare paired source deltas.
run executes each exact manifest into a NEW directory, preserving prior trials.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy import signal, special

CASES = [
    ("inband", 11, 455, 10.743655413271245),
    ("high", 11, 105, 53.71082654249127),
    ("quiet", 12, 193, 10.743655413271245),
]
# Centers measured once from original x with a 32768-sample Hann spectrum in
# the preselected audit family; not a candidate selector or a width search.


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def binding(p):
    p = Path(p).resolve(strict=True)
    return dict(path=str(p), sha256=digest(p))


def write_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    invocations = [
        json.loads(s)
        for s in (a.audit / "campaign-01/invocations.jsonl").read_text().splitlines()
    ]
    records = [
        json.loads(s)
        for s in gzip.open(a.audit / "analysis-final/detectors.jsonl.gz", "rt")
    ]
    expected = json.loads((a.audit / "analysis-final/summary.json").read_text())[
        "bindings"
    ]["burden_accounting_sha256"]
    assert digest(a.burden / "detector-accounting.jsonl.gz") == expected
    accounting = {
        (r["observation"], r["network"], r["detector"]): r
        for r in map(
            json.loads, gzip.open(a.burden / "detector-accounting.jsonl.gz", "rt")
        )
    }
    # Verify all chosen exported originals against the already sealed audit.
    sealed = {
        s[66:]: s[:64] for s in (a.audit / "AUDIT_SHA256SUMS").read_text().splitlines()
    }

    def sealed_binding(p):
        b = binding(p)
        rel = str(p.relative_to(a.audit))
        assert sealed.get(rel, sealed.get("./" + rel)) == b["sha256"], rel
        return b

    tel = a.data / "v1/science/data/tel_toltec_2026-02-19_152390_00_0002.nc"
    config = a.data / "v2/science/refactor/NGC4449/reduced/citlali_o152390_0_2_c2.yaml"
    ast = (
        a.data
        / "wp7/ast-route-family-motion/adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7/ast-scan-motion-152390-v2.json"
    )
    assert (
        digest(tel)
        == "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b"
    )
    assert (
        digest(config)
        == "69e93f75b71209772ff080b074c4f3f91f99b2d6397b2f3f3ed24852f084b1e2"
    )
    summary = []
    for name, nw, d, center in CASES:
        record = next(
            r
            for r in records
            if (r["observation"], r["network"], r["detector"]) == (152390, nw, d)
        )
        assert record["array"] == "a2000" and record["apt_good"]
        folder = a.audit / f"campaign-01/152390-{nw:02d}"
        receipt = json.loads((folder / "receipt.json").read_text())
        invocation = next(
            r for r in invocations if (r["observation"], r["network"]) == (152390, nw)
        )
        fs = 1 / receipt["measured_interval"]
        optical = 235 * np.pi / (180 * 3600) * 50 / (299792458 / 150e9)
        stop = fs / 4 * (1 - 0.0001)
        taps, beta = signal.kaiserord(80, (stop - optical) / (fs / 2))
        taps |= 1
        fir = signal.firwin(
            taps, (optical + stop) / 2, window=("kaiser", beta), fs=fs, scale=True
        )
        fir = (fir + fir[::-1]) / 2
        fir /= sum(fir)
        freq, response = signal.freqz(fir, worN=131072, fs=fs)
        pass_error = float(np.max(np.abs(abs(response[freq <= optical]) - 1)))
        stop_max = float(np.max(abs(response[freq >= stop])))
        assert pass_error < 0.01 and abs(sum(fir) - 1) < 1e-12 and stop_max < 0.001
        cfg = dict(
            case=name,
            channel=d,
            network=nw,
            observation=152390,
            array="a2000",
            raw=binding(invocation["command"][1]),
            tune=binding(invocation["command"][2]),
            manifest=binding(invocation["command"][3]),
            samples=sealed_binding(folder / f"samples-{d}.f64"),
            audit_receipt=sealed_binding(folder / "receipt.json"),
            telescope=binding(tel),
            effective_config=binding(config),
            ast_acceptance=binding(ast),
            notch_hz=center,
            lowpass_identity="offline-explicit-a2000-235arcsec-per-sec-80dB-Kaiser-F2",
            fir=fir.tolist(),
            nominal_interval=receipt["nominal_interval"],
            measured_interval=receipt["measured_interval"],
            science=dict(
                beam="50m-unobscured-Airy-150GHz",
                speed_ceiling_arcsec_per_sec=235,
                measured_AST_maximum_arcsec_per_sec=221.40490828695155,
                inclusive_minimum_speed_arcsec_per_sec=1.0,
                raw_four_sample_speed_ceiling_arcsec_per_sec=(
                    1.028993969962188
                    * (299792458 / 150e9)
                    / 50
                    / (np.pi / (180 * 3600))
                )
                * (0.9999 / receipt["nominal_interval"])
                / (4 * 2 * 1.05),
                design_ceiling_is_not_final_occurrence_admission=True,
                notch_has_finite_five_second_footprint=False,
                speed_margin=0.05,
                cadence_margin=0.0001,
                optical_support_hz=optical,
                stop_hz=stop,
                taps=taps,
                kaiser_beta=beta,
                sampled_passband_magnitude_error=pass_error,
                sampled_stopband_amplitude_maximum=stop_max,
                not_bank_qualification=True,
            ),
            trials=[
                dict(id="lowpass", notch=False, reject=False),
                dict(id="notch", notch=True, reject=False),
                dict(id="reject", notch=False, reject=True),
            ],
            injections=[],
            audit_record=record,
            conditional_transient_accounting=accounting[152390, nw, d],
            exclusion_scope="existing screening Apply; source unknown; no accepted jump without actual scan binding; measured direct support only a separate conditional comparison",
        )
        write_json(a.output / f"{name}.json", cfg)
        summary.append(
            dict(
                case=name,
                network=nw,
                channel=d,
                notch_hz=center,
                science=cfg["science"],
            )
        )
    write_json(a.output / "selection.json", summary)


def inject(a):
    from netCDF4 import Dataset

    a.output.mkdir(parents=True, exist_ok=False)
    for name, _, _, _ in CASES:
        cfg = json.loads((a.plans / f"{name}.json").read_text())
        run = a.geometry / name
        g = np.fromfile(run / "geometry.f64", "<f8").reshape(-1, 4)
        t = g[:, 0]
        dt = cfg["nominal_interval"]
        n = len(t)
        # Telescope axes are the exact AST source, not a map or a common grid.
        with Dataset(cfg["telescope"]["path"]) as nc:
            tt = np.asarray(nc["Data.TelescopeBackend.TelTime"][:])
            ra = np.asarray(nc["Data.TelescopeBackend.SourceRaAct"][:])
            dec = np.asarray(nc["Data.TelescopeBackend.SourceDecAct"][:])
        cause = np.fromfile(run / "notch-causes.u8", "u1")
        eligible = np.flatnonzero(cause == 0)
        # Three actual speeds, fixed once before injection outcomes: select
        # quantiles of eligible measured AST speed, then first matching row.
        interior = eligible[(eligible > 2000) & (eligible < n - 2000)]
        speed = g[interior, 1]
        centers = []
        for q in (0.1, 0.5, 0.9):
            target = np.quantile(speed, q)
            idx = int(interior[np.argmin(abs(speed - target))])
            centers += [
                (f"q{int(q * 100)}p0", idx, 0.0),
                (f"q{int(q * 100)}p5", idx, 0.5),
            ]
        idx = int(interior[np.argmax(speed)])
        centers += [("maxp0", idx, 0.0), ("maxp5", idx, 0.5)]
        # The original fastest motion is an explicit excluded-domain control,
        # not silently dropped from the experiment after support admission.
        idx = int(np.nanargmax(g[:, 1]))
        centers.append(("max-original-excluded", idx, 0.0))
        # A boundary challenge uses the first *input-admitted* row. Its lost
        # source support is reported, never hidden by selecting retained rows.
        guard = json.loads((run / "receipt.json").read_text())["trials"][1][
            "guard_samples_each_end"
        ]
        centers.append(("boundary", int(eligible[0] - guard + 2), 0.0))
        injections = []
        for label, index, phase in centers:
            center = t[index] + phase * dt
            r0 = np.interp(center, tt, ra)
            d0 = np.interp(center, tt, dec)
            use = np.arange(n)
            wave = np.zeros(n)
            wave_fine = np.zeros(n)
            for quad, dest in [(32, wave), (64, wave_fine)]:
                for chunk in np.array_split(use, max(1, (n + 9999) // 10000)):
                    qt = t[chunk, None] + ((np.arange(quad) + 0.5) / quad - 0.5) * dt
                    rr = np.interp(qt.ravel(), tt, ra).reshape(qt.shape)
                    dd = np.interp(qt.ravel(), tt, dec).reshape(qt.shape)
                    # Stable spherical angular separation, exact J2000 radians.
                    angle = 2 * np.arcsin(
                        np.sqrt(
                            np.clip(
                                np.sin((dd - d0) / 2) ** 2
                                + np.cos(dd) * np.cos(d0) * np.sin((rr - r0) / 2) ** 2,
                                0,
                                1,
                            )
                        )
                    )
                    u = np.pi * 50 / (299792458 / 150e9) * angle
                    beam = np.ones_like(u)
                    np.divide(2 * special.j1(u), u, out=beam, where=u != 0)
                    beam **= 2
                    dest[chunk] = beam.mean(axis=1)
            assert np.max(abs(wave - wave_fine)) < 1e-5
            delta = np.column_stack([1e-5 * wave_fine, 2e-6 * wave_fine])
            path = a.output / f"{name}-{label}.f64"
            delta.astype("<f8").tofile(path)
            injections.append(
                dict(
                    **binding(path),
                    identity=label,
                    center_unix_sec=center,
                    native_row=index,
                    subsample_phase=phase,
                    measured_speed_arcsec_per_sec=float(g[index, 1]),
                    source_radec_rad=[float(r0), float(d0)],
                    beam="50m-unobscured-Airy-150GHz",
                    integration="owner-provisional-uniform-average-center;64-midpoint-quadrature",
                    interval_seconds=dt,
                    support="full original observation; fixed sky point; no temporal truncation",
                    quadrature_max_unit_peak_error=float(max(abs(wave - wave_fine))),
                    geometry_binding=binding(run / "geometry.f64"),
                    boundary_challenge=label == "boundary",
                )
            )
        cfg["injections"] = injections
        write_json(a.output / f"{name}.json", cfg)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    exe = a.executable.resolve(strict=True)
    records = []
    for name, _, _, _ in CASES:
        plan = a.plans / f"{name}.json"
        cmd = [str(exe), str(plan.resolve()), str((a.output / name).resolve())]
        start = time.monotonic()
        with (a.output / f"{name}.log").open("wb") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            _, status, usage = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
        record = dict(
            case=name,
            command=cmd,
            executable=binding(exe),
            plan=binding(plan),
            wall_seconds=time.monotonic() - start,
            max_rss_bytes=usage.ru_maxrss,
            exit_status=p.returncode,
        )
        records.append(record)
        write_json(a.output / "invocations.json", records)
        print(name, p.returncode, round(record["wall_seconds"], 2), flush=True)
        if p.returncode:
            raise RuntimeError(f"failed {name}; inspect preserved log")


def shorter(a):
    """Three fixed-duration finite notches; no depth fitting or width search.

    Reuse the prior exact cases, source overlays, AST and exclusion authorities.
    Requested band width stays 0.5 Hz; actual attenuation depends on length.
    """
    a.output.mkdir(parents=True, exist_ok=False)
    for name, _, _, _ in CASES:
        prior = a.prior / f"{name}.json"
        cfg = json.loads(prior.read_text())
        for value in cfg.values():
            if isinstance(value, dict) and "path" in value and "sha256" in value:
                assert digest(value["path"]) == value["sha256"]
        for value in cfg["injections"]:
            assert digest(value["path"]) == value["sha256"]
            assert (
                digest(value["geometry_binding"]["path"])
                == value["geometry_binding"]["sha256"]
            )
        cfg["prior_exact_trial_manifest"] = binding(prior)
        cfg["injection_selection"] = (
            "unchanged ten full-sky paired fixtures from prior experiment; includes retained113arcsec/s and excluded221arcsec/s crossings"
        )
        cfg["science"]["notch_has_finite_five_second_footprint"] = True
        cfg["trials"] = [dict(id="lowpass", notch=False, reject=False)]
        dt = cfg["measured_interval"]
        fs = 1 / dt
        center = cfg["notch_hz"]
        for seconds in (1, 3, 6):
            half = int(np.floor(seconds / (2 * dt)))
            taps = 2 * half + 1
            h = signal.firwin(
                taps,
                [center - 0.25, center + 0.25],
                pass_zero="bandstop",
                window="hann",
                fs=fs,
                scale=True,
            )
            h = (h + h[::-1]) / 2
            h /= sum(h)
            total_half = (half + len(cfg["fir"]) // 2) * dt
            assert total_half <= 5 and abs(sum(h) - 1) < 1e-12
            k = np.arange(-half, half + 1)
            response = float(np.dot(h, np.cos(2 * np.pi * center * dt * k)))
            cfg["trials"].append(
                dict(
                    id=f"finite-{seconds}s",
                    notch=False,
                    reject=False,
                    finite_notch_identity=f"explicit-Hann-bandstop-width0.5Hz-span{seconds}s",
                    finite_notch=h.tolist(),
                    requested_span_seconds=seconds,
                    actual_span_seconds=2 * half * dt,
                    requested_width_hz=0.5,
                    actual_center_amplitude=response,
                    cumulative_half_seconds=total_half,
                    operation="centered finite notch then unchanged centered low-pass; complete support; no padding",
                )
            )
        cfg["trials"].append(dict(id="reject", notch=False, reject=True))
        write_json(a.output / f"{name}.json", cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--audit", type=Path, required=True)
    q.add_argument("--burden", type=Path, required=True)
    q.add_argument("--data", type=Path, required=True)
    q = sub.add_parser("inject")
    q.add_argument("--plans", type=Path, required=True)
    q.add_argument("--geometry", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--plans", type=Path, required=True)
    q.add_argument("--executable", type=Path, required=True)
    q = sub.add_parser("shorter")
    q.add_argument("--prior", type=Path, required=True)
    for q in sub.choices.values():
        q.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a)
