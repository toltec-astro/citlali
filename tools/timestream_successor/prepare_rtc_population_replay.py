#!/usr/bin/env python3
"""One contribution-ranked replay and one targeted same-network coherence check.

Selection is a frozen audit decision, not runtime automatic treatment policy.
The existing quiet/in-band/high-frequency replays are reused separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rtc_disturbance_burden import records
from run_rtc_notch_recovery import binding, digest, finite_trial, write_json


def cross_spectrum(left, right):
    """Matched-window descriptive coherence; no significance/causal test."""
    if (
        left.shape != right.shape
        or left.ndim != 2
        or len(left) < 3
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("coherence requires finite matched independent windows")
    xx, yy = np.mean(abs(left) ** 2, axis=0), np.mean(abs(right) ** 2, axis=0)
    xy = np.mean(left * np.conj(right), axis=0)
    if np.any(xx * yy <= 0):
        raise ValueError("coherence unavailable for zero-power bins")
    return abs(xy) ** 2 / (xx * yy), np.angle(xy)


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((a.short / "plans/quiet.json").read_text())
    prior = json.loads(Path(cfg["audit_receipt"]["path"]).read_text())
    receipt = json.loads((a.native / "receipt.json").read_text())
    for key in (
        "raw_sha256",
        "tune_sha256",
        "manifest_sha256",
        "rows",
        "channels",
        "measured_interval",
        "original_pair_fingerprint_before",
        "original_pair_fingerprint_after",
    ):
        if receipt[key] != prior[key]:
            raise ValueError("native replay differs from preserved audit: " + key)
    for name in ("spectra.f64", "spectra.jsonl", "windows.f64", "native-time.f64"):
        if digest(a.native / name) != digest(
            Path(cfg["audit_receipt"]["path"]).parent / name
        ):
            raise ValueError("accepted Learn changed: " + name)
    ranking = json.loads((a.population / "ranking.json").read_text())
    chosen = next(
        r
        for r in ranking
        if r["observation"] == 152390 and r["network"] == 12 and r["detector"] == 269
    )
    if chosen["family"] != "around11" or not chosen["weighted"] or not chosen["bound"]:
        raise ValueError("frozen contribution selection no longer applies")
    cfg["case"] = "valuable"
    cfg["channel"] = 269
    cfg["samples"] = binding(a.native / "samples-269.f64")
    cfg["audit_receipt"] = binding(a.native / "receipt.json")
    cfg["notch_hz"] = chosen["spectral"]["narrow_frequency_hz"]
    cfg["trials"] = [
        dict(id="lowpass", notch=False, reject=False),
        finite_trial(cfg["measured_interval"], cfg["notch_hz"], 6, cfg["fir"]),
        dict(id="reject", notch=False, reject=True),
    ]
    cfg["audit_record"] = next(
        r
        for r in records(a.population / "detectors.jsonl.gz")
        if (r["observation"], r["network"], r["detector"]) == (152390, 12, 269)
    )
    cfg["conditional_transient_accounting"] = next(
        r
        for r in records(a.burden / "detector-accounting.jsonl.gz")
        if (r["observation"], r["network"], r["detector"]) == (152390, 12, 269)
    )
    cfg["selection"] = dict(
        reason="highest static-APT-weighted six-second footprint potential in the exactly bound population",
        ranking=binding(a.population / "ranking.json"),
        source="prior accepted four-second original-native D2 peak bin",
        center_bin_uncertainty_hz=1
        / (2 * receipt["fft_samples"] * receipt["measured_interval"]),
        intrinsic_width_and_drift_qualified=False,
    )
    cfg["prior_exact_trial_manifest"] = binding(a.short / "plans/quiet.json")
    cfg["injection_selection"] = (
        "unchanged ten original a2000/network12 fixed-sky fixtures; same exact native geometry; no newly retained speed/phase qualification"
    )
    for value in cfg["injections"]:
        if digest(value["path"]) != value["sha256"]:
            raise ValueError("changed prior source overlay")
    write_json(a.output / "valuable.json", cfg)


def coherence(a):
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.native / "receipt.json").read_text())
    meta = list(records(a.native / "spectra.jsonl"))
    windows = np.memmap(
        a.native / "windows.f64", "<f8", "r", shape=(receipt["window_records"], 14)
    )
    n = receipt["fft_samples"]
    dt = receipt["measured_interval"]
    h = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    samples = {
        d: np.fromfile(a.native / f"samples-{d}.f64", "<f8").reshape(-1, 4)
        for d in (269, 402)
    }
    results = []
    for c in (0, 1):
        row_sets = []
        for d in (269, 402):
            m = meta[2 * d + c]
            w = windows[m["window_offset"] : m["window_offset"] + m["window_count"]]
            row_sets.append({(int(r[2]), int(r[3])) for r in w if r[3] - r[2] == n})
        common = sorted(row_sets[0] & row_sets[1])
        selected, end = [], -1
        for first, last in common:
            if first >= end:
                selected.append((first, last))
                end = last
        if len(selected) < 3:
            raise ValueError("targeted coherence has insufficient independent windows")
        transforms = []
        for d in (269, 402):
            s = samples[d]
            if any(not s[first:last, 2:4].all() for first, last in selected):
                raise ValueError("coherence crosses declared invalid pair support")
            centered = s[:, c] - meta[2 * d + c]["population_median"]
            transforms.append(
                np.array(
                    [
                        np.fft.rfft(
                            (centered[first:last] - np.median(centered[first:last])) * h
                        )
                        for first, last in selected
                    ]
                )
            )
        left, right = transforms
        coh, phase = cross_spectrum(left, right)
        frequency = np.arange(len(coh)) / (n * dt)
        i = int(np.argmin(abs(frequency - 10.756113666291341)))
        groups = np.array_split(np.arange(len(selected)), 4)
        quarter = []
        for group in groups:
            x, y = left[group, i], right[group, i]
            quarter.append(
                float(
                    abs(np.mean(x * np.conj(y))) ** 2
                    / (np.mean(abs(x) ** 2) * np.mean(abs(y) ** 2))
                )
            )
        results.append(
            dict(
                coordinate="x" if c == 0 else "r",
                windows=len(selected),
                native_row_support=selected,
                frequency_hz=frequency.tolist(),
                magnitude_squared_coherence=coh.tolist(),
                cross_phase_rad=phase.tolist(),
                target_bin=i,
                target_frequency_hz=float(frequency[i]),
                target_coherence=float(coh[i]),
                target_quarter_coherence=quarter,
                quarter_is_not_confidence_interval=True,
                direct_origin_or_interference_classification=False,
            )
        )
    write_json(
        a.output / "coherence.json",
        dict(
            observation=152390,
            network=12,
            detectors=[269, 402],
            source_bindings={
                name: binding(a.native / name)
                for name in (
                    "receipt.json",
                    "spectra.jsonl",
                    "windows.f64",
                    "samples-269.f64",
                    "samples-402.f64",
                )
            },
            estimator="matched original accepted full four-second symmetric Hann windows; inherited two-median centering; native-disjoint subset; mean cross/auto spectra; no common grid",
            selection="two highest contribution-ranked detectors on the exactly bound network; one targeted pair only",
            source_protection="unknown and retained",
            covariance_authority="diagnostic only; no independent-array averaging or common-mode treatment authorized",
            results=results,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "coherence"])
    for name in ("native", "population", "short", "burden", "output"):
        p.add_argument("--" + name, type=Path, required=name in ("native", "output"))
    args = p.parse_args()
    globals()[args.action](args)
