#!/usr/bin/env python3
"""Build targeted RTC/PTC diagnostic galleries with matched control rows."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to generate the network residual control gallery") from exc


ARRAY_TO_ID = {
    "a1100": 0,
    "a1400": 1,
    "a2000": 2,
}

DEFAULT_TARGETS = [
    "151928:4",
    "151930:4",
    "152524:4",
    "152526:1",
]


@dataclass(frozen=True)
class RowChoice:
    kind: str
    obsnum: int
    network: int
    scan: int
    output_scan_index: int
    corr_z: float
    topmode_z: float
    lowmid: float
    line_prom: float
    tail4_z: float


def _parse_target(spec: str) -> tuple[int, int]:
    obs_s, nw_s = spec.split(":")
    return int(obs_s), int(nw_s)


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        std = np.std(x)
        return float(std) if std > 0 else 1.0
    return float(1.4826 * mad)


def _robust_standardize(signal: np.ndarray, flags: np.ndarray) -> np.ndarray:
    good = np.isfinite(signal) & (flags == 0)
    if good.sum() < 8:
        return np.full_like(signal, np.nan, dtype=float)
    mu = float(np.median(signal[good]))
    sig = _mad_sigma(signal[good])
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    out = (signal.astype(float) - mu) / sig
    out[~np.isfinite(out)] = np.nan
    return out


def _nanmedian_trace(mat: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.nanmedian(mat, axis=0)
    return np.asarray(out, dtype=float)


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.zeros_like(x)
    if valid.sum() == 1:
        return np.full_like(x, float(x[valid][0]))
    idx = np.arange(x.size, dtype=float)
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out


def _simple_psd(trace: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = _fill_nan_linear(trace)
    x = x - np.mean(x)
    n = x.size
    if n < 16:
        return np.asarray([]), np.asarray([])
    window = np.hanning(n)
    xw = x * window
    spec = np.fft.rfft(xw)
    psd = (np.abs(spec) ** 2) / max((window**2).sum() * fs_hz, 1e-12)
    freq = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freq, psd


def _load_detail_csv(path: Path, obsnum: int, network: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["obsnum"] == obsnum) & (df["network"] == network)].copy()
    if df.empty:
        raise RuntimeError(f"no null-audit rows found for obs={obsnum} nw={network} in {path}")
    df.sort_values(["med_abs_corr_surr_z", "tail4_binom_z"], ascending=[False, False], inplace=True)
    return df


def _choose_rows(df: pd.DataFrame) -> tuple[RowChoice, RowChoice]:
    bad = df.iloc[0]
    ctrl_df = df.sort_values(
        ["med_abs_corr_surr_z", "tail4_binom_z", "cm_low_mid_ratio", "top_mode_surr_z"],
        ascending=[True, True, True, True],
    )
    ctrl = ctrl_df.iloc[0]

    def to_choice(kind: str, row: pd.Series) -> RowChoice:
        return RowChoice(
            kind=kind,
            obsnum=int(row["obsnum"]),
            network=int(row["network"]),
            scan=int(row["scan"]),
            output_scan_index=int(row["output_scan_index"]),
            corr_z=float(row["med_abs_corr_surr_z"]),
            topmode_z=float(row["top_mode_surr_z"]),
            lowmid=float(row["cm_low_mid_ratio"]),
            line_prom=float(row["cm_peak_prominence"]),
            tail4_z=float(row["tail4_binom_z"]),
        )

    return to_choice("problem", bad), to_choice("control", ctrl)


def _find_scan_row_index(ds: Dataset, output_scan_index: int) -> int:
    out = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
    hits = np.where(out == int(output_scan_index))[0]
    if hits.size == 0:
        raise RuntimeError(f"output_scan_index={output_scan_index} not present in {ds.filepath()}")
    return int(hits[0])


def _extract_case(
    rtc_path: Path,
    ptc_path: Path,
    array_name: str,
    row: RowChoice,
    max_det: int,
) -> dict[str, object]:
    array_id = ARRAY_TO_ID[array_name]
    with Dataset(rtc_path) as rtc_ds, Dataset(ptc_path) as ptc_ds:
        scan_i = _find_scan_row_index(ptc_ds, row.output_scan_index)
        start, stop = [int(v) for v in ptc_ds.variables["scan_indices"][scan_i]]
        apt_array = np.asarray(ptc_ds.variables["apt_array"][:], dtype=int)
        apt_nw = np.asarray(ptc_ds.variables["apt_nw"][:], dtype=int)
        apt_flag = np.asarray(ptc_ds.variables["apt_flag"][:], dtype=int)
        apt_uid = np.asarray(ptc_ds.variables["apt_uid"][:], dtype=int)
        weights = np.asarray(ptc_ds.variables["weights"][scan_i], dtype=float)

        det_mask = (apt_array == array_id) & (apt_nw == row.network) & (apt_flag == 0) & np.isfinite(weights) & (weights > 0)
        det_idx = np.where(det_mask)[0]
        if det_idx.size == 0:
            raise RuntimeError(f"no detectors left for obs={row.obsnum} nw={row.network} array={array_name}")
        det_order = det_idx[np.argsort(weights[det_idx])[::-1]]
        det_order = det_order[: min(max_det, det_order.size)]

        rtc_signal = np.asarray(rtc_ds.variables["signal"][start:stop, det_order], dtype=float).T
        rtc_flags = np.asarray(rtc_ds.variables["flags"][start:stop, det_order], dtype=int).T
        ptc_signal = np.asarray(ptc_ds.variables["signal"][start:stop, det_order], dtype=float).T
        ptc_flags = np.asarray(ptc_ds.variables["flags"][start:stop, det_order], dtype=int).T

        rtc_z = np.vstack([_robust_standardize(rtc_signal[i], rtc_flags[i]) for i in range(rtc_signal.shape[0])])
        ptc_z = np.vstack([_robust_standardize(ptc_signal[i], ptc_flags[i]) for i in range(ptc_signal.shape[0])])

        rtc_z_masked = rtc_z.copy()
        rtc_z_masked[rtc_flags > 0] = np.nan
        ptc_z_masked = ptc_z.copy()
        ptc_z_masked[ptc_flags > 0] = np.nan

        ptc_cm = _nanmedian_trace(ptc_z_masked)
        rtc_cm = _nanmedian_trace(rtc_z_masked)

        corr = []
        valid_cm = np.isfinite(ptc_cm)
        for i in range(ptc_z.shape[0]):
            good = np.isfinite(ptc_z_masked[i]) & valid_cm
            if good.sum() < 8:
                corr.append(-np.inf)
            else:
                corr.append(float(np.corrcoef(ptc_z_masked[i, good], ptc_cm[good])[0, 1]))
        det_sort = np.argsort(np.nan_to_num(corr, nan=-np.inf))[::-1]
        rtc_z = rtc_z[det_sort]
        rtc_flags = rtc_flags[det_sort]
        ptc_z = ptc_z[det_sort]
        ptc_flags = ptc_flags[det_sort]
        det_uids = apt_uid[det_order][det_sort]

        rtc_z_masked = rtc_z.copy()
        rtc_z_masked[rtc_flags > 0] = np.nan
        ptc_z_masked = ptc_z.copy()
        ptc_z_masked[ptc_flags > 0] = np.nan
        rtc_cm = _nanmedian_trace(rtc_z_masked)
        ptc_cm = _nanmedian_trace(ptc_z_masked)

        fs_hz = float(ptc_ds.variables["signal"].fsmp if hasattr(ptc_ds.variables["signal"], "fsmp") else 61.03558)
        f_rtc, p_rtc = _simple_psd(rtc_cm, fs_hz)
        f_ptc, p_ptc = _simple_psd(ptc_cm, fs_hz)

        sp_nw = np.asarray(ptc_ds.variables["ptc_second_pass_network_ids"][:], dtype=int)
        sp_idx = np.where(sp_nw == row.network)[0]
        sp_summary = None
        if sp_idx.size > 0:
            j = int(sp_idx[0])
            sp_summary = {
                "accepted_clusters": int(ptc_ds.variables["ptc_second_pass_n_accepted_clusters"][scan_i, j]),
                "candidate_clusters": int(ptc_ds.variables["ptc_second_pass_n_candidate_clusters"][scan_i, j]),
                "busy_veto": int(ptc_ds.variables["ptc_second_pass_busy_network_vetoed"][scan_i, j]),
                "newfrac": float(ptc_ds.variables["ptc_second_pass_newly_flagged_fraction"][scan_i, j]),
                "top_score": float(ptc_ds.variables["ptc_second_pass_top_candidate_cluster_peak_score"][scan_i, j]),
                "top_ndet": int(ptc_ds.variables["ptc_second_pass_top_candidate_cluster_n_detectors"][scan_i, j]),
                "top_uid": int(ptc_ds.variables["ptc_second_pass_max_unflagged_residual_uid"][scan_i, j]),
                "top_z": float(ptc_ds.variables["ptc_second_pass_max_unflagged_residual_z"][scan_i, j]),
            }

        t = np.arange(stop - start, dtype=float) / fs_hz
        return {
            "time_s": t,
            "rtc_z": rtc_z,
            "ptc_z": ptc_z,
            "ptc_flags": ptc_flags,
            "rtc_cm": rtc_cm,
            "ptc_cm": ptc_cm,
            "f_rtc": f_rtc,
            "p_rtc": p_rtc,
            "f_ptc": f_ptc,
            "p_ptc": p_ptc,
            "uids": det_uids,
            "scan_i": scan_i,
            "sp": sp_summary,
        }


def _panel_title(row: RowChoice, sp: dict[str, object] | None) -> str:
    line = (
        f"{row.kind.title()} row: scan {row.output_scan_index} nw {row.network}  "
        f"corr_z={row.corr_z:.2f} topmode_z={row.topmode_z:.1f} "
        f"low/mid={row.lowmid:.2f} tail4_z={row.tail4_z:.2f}"
    )
    if sp is None:
        return line
    return (
        line
        + f"\nsecond-pass: cand={sp['candidate_clusters']} acc={sp['accepted_clusters']} "
        + f"busy={sp['busy_veto']} addfrac={sp['newfrac']:.2e}"
    )


def _plot_pair(outpath: Path, rtc_path: Path, ptc_path: Path, array_name: str, bad: RowChoice, ctrl: RowChoice, max_det: int) -> None:
    bad_case = _extract_case(rtc_path, ptc_path, array_name, bad, max_det=max_det)
    ctrl_case = _extract_case(rtc_path, ptc_path, array_name, ctrl, max_det=max_det)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)

    for col, (row, case) in enumerate([(bad, bad_case), (ctrl, ctrl_case)]):
        t = np.asarray(case["time_s"])
        ptc_z = np.asarray(case["ptc_z"])
        ptc_flags = np.asarray(case["ptc_flags"]) > 0
        rtc_cm = np.asarray(case["rtc_cm"])
        ptc_cm = np.asarray(case["ptc_cm"])
        f_rtc = np.asarray(case["f_rtc"])
        p_rtc = np.asarray(case["p_rtc"])
        f_ptc = np.asarray(case["f_ptc"])
        p_ptc = np.asarray(case["p_ptc"])

        ax = axes[0, col]
        im = ax.imshow(
            ptc_z,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], 0, ptc_z.shape[0]],
            vmin=-6,
            vmax=6,
            cmap="coolwarm",
        )
        if np.any(ptc_flags):
            x = np.linspace(t[0], t[-1], ptc_flags.shape[1])
            y = np.arange(ptc_flags.shape[0])
            ax.contourf(
                x,
                y,
                ptc_flags.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
                alpha=0.0,
            )
        ax.set_title(_panel_title(row, case["sp"]), fontsize=10)
        ax.set_ylabel("PTC detectors")
        if col == 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.047)
            cbar.set_label("PTC z")

        ax = axes[1, col]
        ax.plot(t, rtc_cm, label="RTC common median", lw=1.5)
        ax.plot(t, ptc_cm, label="PTC common median", lw=1.5)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylabel("common z")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[2, col]
        if f_rtc.size:
            mask = (f_rtc > 0.05) & (f_rtc <= 20.0)
            ax.semilogy(f_rtc[mask], p_rtc[mask], label="RTC PSD", lw=1.5)
        if f_ptc.size:
            mask = (f_ptc > 0.05) & (f_ptc <= 20.0)
            ax.semilogy(f_ptc[mask], p_ptc[mask], label="PTC PSD", lw=1.5)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("common PSD")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[2, 0].set_xlim(0.05, 20.0)
    axes[2, 1].set_xlim(0.05, 20.0)
    axes[1, 0].set_xlabel("time in row [s]")
    axes[1, 1].set_xlabel("time in row [s]")
    axes[0, 0].set_xlabel("time in row [s]")
    axes[0, 1].set_xlabel("time in row [s]")

    fig.suptitle(
        f"{array_name} residual/control diagnostic\n"
        f"obs {bad.obsnum} nw {bad.network}  |  left=worst null-audit row, right=matched low-scoring control row",
        fontsize=14,
    )
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _write_report(outpath: Path, entries: list[dict[str, object]], array_name: str) -> None:
    lines = [
        f"# {array_name} Network Residual/Control Gallery",
        "",
        "- Left column in each PNG: worst null-audit row for the target obs/network.",
        "- Right column in each PNG: lowest-scoring control row from the same obs/network.",
        "- Heatmap: standardized PTC detector signals in that row.",
        "- Hatched heatmap samples are flagged in the PTC timestream and should not propagate into downstream mapmaking.",
        "- Middle panel: RTC vs PTC robust common trace using only unflagged samples.",
        "- Bottom panel: qualitative common-trace PSD from that unflagged common trace.",
        "",
        "## Cases",
        "",
    ]
    for e in entries:
        bad: RowChoice = e["bad"]  # type: ignore[assignment]
        ctrl: RowChoice = e["ctrl"]  # type: ignore[assignment]
        png: Path = e["png"]  # type: ignore[assignment]
        lines.extend(
            [
                f"### obs {bad.obsnum} nw {bad.network}",
                "",
                f"- PNG: `{png.name}`",
                f"- Problem row: output scan `{bad.output_scan_index}` corr_z `{bad.corr_z:.2f}` topmode_z `{bad.topmode_z:.1f}` low/mid `{bad.lowmid:.2f}` tail4_z `{bad.tail4_z:.2f}`",
                f"- Control row: output scan `{ctrl.output_scan_index}` corr_z `{ctrl.corr_z:.2f}` topmode_z `{ctrl.topmode_z:.1f}` low/mid `{ctrl.lowmid:.2f}` tail4_z `{ctrl.tail4_z:.2f}`",
                "",
            ]
        )
    outpath.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Reduction directory, e.g. .../reduced/redu12")
    ap.add_argument("--array", default="a1100", choices=sorted(ARRAY_TO_ID))
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="Target obs:nw pairs")
    ap.add_argument("--max-det", type=int, default=48, help="Maximum detectors to show in each panel")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory")
    args = ap.parse_args()

    outdir = args.outdir or (args.run_dir / f"{args.array}_network_diagnostic_gallery")
    outdir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for spec in args.targets:
        obsnum, network = _parse_target(spec)
        null_csv = args.run_dir / "pca_audit_focus" / f"null_obs{obsnum}_{args.array}" / "blank_sky_null_audit_detailed.csv"
        if not null_csv.exists():
            raise RuntimeError(f"missing null-audit detailed CSV: {null_csv}")
        df = _load_detail_csv(null_csv, obsnum=obsnum, network=network)
        bad, ctrl = _choose_rows(df)
        rtc_path = args.run_dir / str(obsnum) / "raw" / f"toltec_commissioning_science_{obsnum}_rtc_timestream.nc"
        ptc_path = args.run_dir / str(obsnum) / "raw" / f"toltec_commissioning_science_{obsnum}_ptc_timestream.nc"
        png = outdir / f"{args.array}_obs{obsnum}_nw{network}_problem_vs_control.png"
        _plot_pair(png, rtc_path, ptc_path, args.array, bad, ctrl, max_det=args.max_det)
        entries.append({"bad": bad, "ctrl": ctrl, "png": png})

    _write_report(outdir / "NETWORK_RESIDUAL_CONTROL_GALLERY.md", entries, args.array)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
