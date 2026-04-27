#!/usr/bin/env python
"""Audit beammap FWHM flags against simple non-parametric map widths."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


BITS = (
    (1, "BadFit"),
    (2, "AzFWHM"),
    (4, "ElFWHM"),
    (8, "Sig2Noise"),
    (16, "Sens"),
    (32, "Position"),
    (64, "PriorDist"),
    (128, "NetworkPos"),
)


def flag_names(flag2: int) -> str:
    names = [name for bit, name in BITS if int(flag2) & bit]
    return "+".join(names) if names else "Good"


def load_rows(run_dir: Path) -> dict[int, dict]:
    table = Table.read(
        run_dir
        / "152307"
        / "raw"
        / "apt_commissioning_beammap_152307_citlali_fit_qc.ecsv",
        format="ascii.ecsv",
    )
    return {int(row["uid"]): {name: row[name] for name in table.colnames} for row in table}


def fits_index(hdul) -> dict[int, tuple[int, int]]:
    names = {h.name: i for i, h in enumerate(hdul)}
    index = {}
    for name, i in names.items():
        match = re.fullmatch(r"signal_det_(\d+)_I", name)
        if match is None:
            continue
        uid = int(match.group(1))
        weight_name = f"weight_det_{uid}_I"
        if weight_name in names:
            index[uid] = (i, names[weight_name])
    return index


def get_map(run_dir: Path, uid: int, flag: int, fits_cache):
    raw = run_dir / "152307" / "raw"
    suffixes = ["flag0_good", "flag1_bad"] if flag == 0 else ["flag1_bad", "flag0_good"]
    for suffix in suffixes:
        path = raw / f"toltec_commissioning_a1100_beammap_152307_citlali_{suffix}.fits"
        if path not in fits_cache:
            hdul = fits.open(path, memmap=True)
            fits_cache[path] = (hdul, fits_index(hdul))
        hdul, index = fits_cache[path]
        if uid in index:
            sig_i, wt_i = index[uid]
            signal = np.asarray(hdul[sig_i].data[0, 0], dtype=float)
            weight = np.asarray(hdul[wt_i].data[0, 0], dtype=float)
            cdelt1 = float(hdul[sig_i].header.get("CDELT1", -1.0))
            cdelt2 = float(hdul[sig_i].header.get("CDELT2", 1.0))
            return signal, weight, cdelt1, cdelt2
    raise KeyError(f"uid {uid} not found in a1100 split FITS for {run_dir}")


def robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad > 0:
        return 1.4826 * mad
    return float(np.std(values))


def morphology(signal, weight, x_arcsec, y_arcsec, cdelt1, cdelt2, radius=12.0):
    n_rows, n_cols = signal.shape
    col0 = (n_cols - 1) / 2.0
    row0 = (n_rows - 1) / 2.0
    col_c = col0 + x_arcsec / cdelt1
    row_c = row0 + y_arcsec / cdelt2
    yy, xx = np.indices(signal.shape, dtype=float)
    x = (xx - col0) * cdelt1
    y = (yy - row0) * cdelt2
    rr_fit = np.hypot(x - x_arcsec, y - y_arcsec)
    support = np.isfinite(signal) & np.isfinite(weight) & (weight > 0)
    core = support & (rr_fit <= radius)
    ann = support & (rr_fit > radius) & (rr_fit <= 2.0 * radius)
    if core.sum() < 10:
        return None
    bkg = np.nanmedian(signal[ann]) if ann.sum() >= 10 else np.nanmedian(signal[support])
    noise = robust_sigma(signal[ann]) if ann.sum() >= 10 else robust_sigma(signal[support])
    excess = signal - bkg
    pos = core & (excess > 0)
    if pos.sum() < 6 or not np.isfinite(noise) or noise <= 0:
        return None
    weights = excess[pos] * np.sqrt(weight[pos] / np.nanmedian(weight[pos]))
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        return None
    xp = x[pos]
    yp = y[pos]
    xbar = float(np.sum(weights * xp) / np.sum(weights))
    ybar = float(np.sum(weights * yp) / np.sum(weights))
    dx = xp - xbar
    dy = yp - ybar
    cov = np.array(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=float,
    ) / np.sum(weights)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    minor, major = np.sqrt(evals)
    fwhm_minor = 2.354820045 * float(minor)
    fwhm_major = 2.354820045 * float(major)
    peak = float(np.nanmax(signal[core]))
    center_offset = float(math.hypot(xbar - x_arcsec, ybar - y_arcsec))
    snr_peak = (peak - bkg) / noise
    theta = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
    return {
        "mom_fwhm_major": fwhm_major,
        "mom_fwhm_minor": fwhm_minor,
        "mom_x": xbar,
        "mom_y": ybar,
        "mom_center_offset": center_offset,
        "mom_angle_deg": theta,
        "local_peak_snr": snr_peak,
        "support_pixels": int(core.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/Users/gwilson/work_toltec/local_data/beammaps/3c273"))
    parser.add_argument("--old", default="redu02")
    parser.add_argument("--new", default="redu04")
    parser.add_argument("--out", type=Path, default=Path("/tmp/beammap_fwhm_morph_audit.csv"))
    args = parser.parse_args()

    old_dir = args.base / args.old
    new_dir = args.base / args.new
    old = load_rows(old_dir)
    new = load_rows(new_dir)
    ids = sorted(set(old) & set(new))
    rows = []
    transition_counts = Counter()
    reason_counts = Counter()
    network_counts = defaultdict(Counter)

    selected = []
    for uid in ids:
        if int(old[uid]["array"]) != 0:
            continue
        old_good = int(old[uid]["flag"]) == 0
        new_good = int(new[uid]["flag"]) == 0
        transition = (
            "both_good"
            if old_good and new_good
            else "old_good_new_bad"
            if old_good
            else "old_bad_new_good"
            if new_good
            else "both_bad"
        )
        transition_counts[transition] += 1
        network_counts[int(new[uid]["nw"])][transition] += 1
        if transition in {"old_good_new_bad", "old_bad_new_good"}:
            selected.append(uid)
        elif not new_good and (int(new[uid]["flag2"]) & (2 | 4)):
            selected.append(uid)
        elif new_good and len([u for u in selected if int(new[u]["nw"]) == int(new[uid]["nw"])]) < 5:
            selected.append(uid)

    fits_cache = {}
    for uid in sorted(set(selected)):
        row = new[uid]
        try:
            signal, weight, cdelt1, cdelt2 = get_map(new_dir, uid, int(row["flag"]), fits_cache)
            morph = morphology(signal, weight, float(row["x_t_raw"]), float(row["y_t_raw"]), cdelt1, cdelt2)
        except Exception as exc:
            morph = {"error": str(exc)}
        old_good = int(old[uid]["flag"]) == 0
        new_good = int(row["flag"]) == 0
        transition = (
            "both_good"
            if old_good and new_good
            else "old_good_new_bad"
            if old_good
            else "old_bad_new_good"
            if new_good
            else "both_bad"
        )
        reason = flag_names(int(row["flag2"]))
        reason_counts[(transition, reason)] += 1
        out = {
            "uid": uid,
            "nw": int(row["nw"]),
            "transition": transition,
            "new_reason": reason,
            "old_reason": flag_names(int(old[uid]["flag2"])),
            "new_flag": int(row["flag"]),
            "old_flag": int(old[uid]["flag"]),
            "new_a_fwhm": float(row["a_fwhm"]),
            "new_b_fwhm": float(row["b_fwhm"]),
            "old_a_fwhm": float(old[uid]["a_fwhm"]),
            "old_b_fwhm": float(old[uid]["b_fwhm"]),
            "new_fit_snr": float(row["fit_sig2noise"]),
            "new_map_snr": float(row["map_sig2noise"]),
            "new_prior_d2": float(row["final_prior_d2"]),
            "x_t_raw": float(row["x_t_raw"]),
            "y_t_raw": float(row["y_t_raw"]),
        }
        if morph:
            out.update(morph)
        rows.append(out)

    for hdul, _ in fits_cache.values():
        hdul.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.out.open("w", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    print("a1100 transitions", dict(transition_counts))
    print("a1100 network transitions")
    for nw in sorted(network_counts):
        print(nw, dict(network_counts[nw]))
    print("selected transition/reason counts")
    for (transition, reason), count in reason_counts.most_common(20):
        print(transition, reason, count)

    def summarize(label, filt):
        vals = [row for row in rows if filt(row) and "mom_fwhm_major" in row]
        print(f"\n{label} n={len(vals)}")
        for key in ["new_a_fwhm", "new_b_fwhm", "mom_fwhm_major", "mom_fwhm_minor", "mom_center_offset", "local_peak_snr"]:
            arr = np.array([float(row[key]) for row in vals if key in row and np.isfinite(float(row[key]))])
            if arr.size:
                print(key, "median", round(float(np.median(arr)), 3), "p90", round(float(np.percentile(arr, 90)), 3))

    summarize("new FWHM-flagged", lambda r: "FWHM" in r["new_reason"])
    summarize("old_good_new_bad", lambda r: r["transition"] == "old_good_new_bad")
    summarize("old_bad_new_good", lambda r: r["transition"] == "old_bad_new_good")
    summarize("both_good sample", lambda r: r["transition"] == "both_good")


if __name__ == "__main__":
    main()
