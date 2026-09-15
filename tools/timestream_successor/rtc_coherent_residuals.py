#!/usr/bin/env python3
"""Matched complete-window residuals before/after the unchanged PCA diagnostic."""

import argparse
import json
from pathlib import Path
import numpy as np
from run_rtc_coherent_science import IDS
from run_rtc_notch_recovery import write_json, binding


def analyze(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepare / "science-design.json").read_text())
    t = np.fromfile(a.evidence / "native-full/native-time.f64", "<f8")
    nr = len(t)
    receipt = json.loads((a.evidence / "native-full/receipt.json").read_text())
    dt = 2 * receipt["measured_interval"]
    nfft = int(round(4 / dt))
    freq = np.fft.rfftfreq(nfft, dt)
    h = np.hanning(nfft)
    df = 1 / (nfft * dt)
    masks = {
        arm: np.memmap(a.prepare / (arm + "-good.u8"), np.bool_, "r", shape=(nr, 491))
        for arm in ("lowpass", "finite", "coherent")
    }
    windows = []
    spectra = []
    for chunk in design["chunks"]:
        rr = np.flatnonzero(
            (t >= chunk * 10) & (t < (chunk + 1) * 10) & (np.arange(nr) % 2 == 0)
        )
        for d in IDS:
            good = np.logical_and.reduce([m[rr, d] for m in masks.values()])
            for lo in range(0, len(rr) - nfft + 1, nfft // 2):
                hi = lo + nfft
                if not good[lo:hi].all():
                    continue
                if np.max(np.diff(t[rr[lo:hi]])) > dt * 1.001:
                    raise ValueError("window crosses physical time gap")
                windows.append(
                    dict(
                        chunk=chunk,
                        detector=d,
                        output_rows=[lo, hi],
                        native_rows=rr[lo:hi].tolist(),
                    )
                )
                for coord in (0, 1):
                    for arm in masks:
                        for stage in ("rtc", "pca"):
                            path = (
                                a.real / "real" / f"{arm}-none-{chunk}-{coord}.f64"
                                if stage == "rtc"
                                else a.real
                                / "real/downstream"
                                / f"{arm}-none-{chunk}-{coord}"
                                / "cleaned.f64"
                            )
                            m = np.memmap(path, "<f8", "r", shape=(len(rr), 491))
                            v = np.array(m[lo:hi, d])
                            v -= np.median(v)
                            fft = np.fft.rfft(v * h)
                            psd = dt * abs(fft) ** 2 / (h * h).sum()
                            psd[1:-1] *= 2
                            spectra.append(
                                dict(
                                    window=len(windows) - 1,
                                    coordinate="x" if coord == 0 else "r_proxy",
                                    arm=arm,
                                    stage=stage,
                                    power_9p75_12p25_mJy2_per_beam2=float(
                                        psd[(freq >= 9.75) & (freq <= 12.25)].sum() * df
                                    ),
                                    psd=psd.tolist(),
                                )
                            )
    write_json(
        a.output / "window-residuals.json",
        dict(
            profile="conditioned/reference diagnostic4sHann; median detrend; one-sided density; exact real complete common-recovery support; no sideband noise authority",
            native_original_receipt=binding(a.evidence / "native-full/receipt.json"),
            windows=windows,
            frequency_hz=freq.tolist(),
            spectra=spectra,
        ),
    )
    summary = []
    for coord in ("x", "r_proxy"):
        for d in IDS:
            for arm in masks:
                r = dict(coordinate=coord, detector=d, arm=arm)
                for stage in ("rtc", "pca"):
                    vals = [
                        v["power_9p75_12p25_mJy2_per_beam2"]
                        for v in spectra
                        if v["coordinate"] == coord
                        and v["arm"] == arm
                        and v["stage"] == stage
                        and windows[v["window"]]["detector"] == d
                    ]
                    r[stage] = dict(
                        windows=len(vals),
                        mean_band_power=float(np.mean(vals)) if vals else None,
                    )
                summary.append(r)
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("evidence", "prepare", "real", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    analyze(p.parse_args())
