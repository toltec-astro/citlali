#!/usr/bin/env python3
"""Quantify common-mode/coherence behavior for targeted residual rows."""

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
    raise RuntimeError("matplotlib is required to generate the common-mode focus report") from exc


ARRAY_TO_ID = {
    "a1100": 0,
    "a1400": 1,
    "a2000": 2,
}

COMMON_MODE_BANDS = (
    ("low", 0.05, 0.5),
    ("mid", 0.5, 2.0),
    ("high", 2.0, 10.0),
)

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


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


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


def _corr_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 8:
        return float("nan")
    xx = x[good] - np.mean(x[good])
    yy = y[good] - np.mean(y[good])
    sx = float(np.std(xx, ddof=1))
    sy = float(np.std(yy, ddof=1))
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(abs(np.corrcoef(xx, yy)[0, 1]))


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


def _common_mode_spectrum_metrics(common_mode: np.ndarray, fs: float) -> dict[str, float]:
    common_mode = np.asarray(common_mode, dtype=float)
    common_mode = common_mode[np.isfinite(common_mode)]
    if common_mode.size < 16 or not np.isfinite(fs) or fs <= 0:
        return {
            "cm_bp_low": float("nan"),
            "cm_bp_mid": float("nan"),
            "cm_bp_high": float("nan"),
            "cm_low_mid_ratio": float("nan"),
            "cm_high_mid_ratio": float("nan"),
            "cm_peak_freq_hz": float("nan"),
            "cm_peak_prominence": float("nan"),
        }
    x = common_mode - np.mean(common_mode)
    win = np.hanning(x.size)
    spec = np.fft.rfft(x * win)
    power = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    band_power: dict[str, float] = {}
    for name, f0, f1 in COMMON_MODE_BANDS:
        mask = (freq >= f0) & (freq < f1)
        band_power[name] = float(np.median(power[mask])) if np.any(mask) else float("nan")
    peak_mask = (freq >= 0.05) & (freq <= min(16.0, float(np.max(freq))))
    if np.any(peak_mask):
        local_power = power[peak_mask]
        local_freq = freq[peak_mask]
        idx = int(np.argmax(local_power))
        peak_power = float(local_power[idx])
        peak_freq = float(local_freq[idx])
        peak_prominence = _safe_ratio(peak_power, float(np.median(local_power)))
    else:
        peak_freq = float("nan")
        peak_prominence = float("nan")
    return {
        "cm_bp_low": band_power["low"],
        "cm_bp_mid": band_power["mid"],
        "cm_bp_high": band_power["high"],
        "cm_low_mid_ratio": _safe_ratio(band_power["low"], band_power["mid"]),
        "cm_high_mid_ratio": _safe_ratio(band_power["high"], band_power["mid"]),
        "cm_peak_freq_hz": peak_freq,
        "cm_peak_prominence": peak_prominence,
    }


def _masked_mode_metrics(z_masked: np.ndarray, min_good_det_frac: float = 0.8) -> dict[str, float]:
    # Input: detector x time standardized matrix with NaN where flagged.
    valid = np.isfinite(z_masked)
    if z_masked.ndim != 2 or z_masked.shape[0] < 2 or z_masked.shape[1] < 8:
        return {"top_mode_frac_masked": float("nan"), "k90_masked": float("nan"), "mode_time_frac": float("nan")}
    keep_cols = np.where(np.mean(valid, axis=0) >= min_good_det_frac)[0]
    if keep_cols.size < 8:
        keep_cols = np.where(np.mean(valid, axis=0) >= 0.5)[0]
    if keep_cols.size < 8:
        return {"top_mode_frac_masked": float("nan"), "k90_masked": float("nan"), "mode_time_frac": float("nan")}
    z = np.asarray(z_masked[:, keep_cols].T, dtype=float)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = z - np.mean(z, axis=0, keepdims=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    evals = evals[np.isfinite(evals) & (evals > 0)]
    if evals.size == 0:
        return {"top_mode_frac_masked": float("nan"), "k90_masked": float("nan"), "mode_time_frac": float("nan")}
    frac = evals / np.sum(evals)
    return {
        "top_mode_frac_masked": float(frac[0]),
        "k90_masked": float(np.searchsorted(np.cumsum(frac), 0.90) + 1),
        "mode_time_frac": float(keep_cols.size / z_masked.shape[1]),
    }


def _detector_common_coupling(z_masked: np.ndarray, common: np.ndarray) -> dict[str, float]:
    corrs: list[float] = []
    for i in range(z_masked.shape[0]):
        c = _corr_abs(z_masked[i], common)
        if np.isfinite(c):
            corrs.append(c)
    if not corrs:
        return {
            "median_det_common_corr_abs": float("nan"),
            "p95_det_common_corr_abs": float("nan"),
            "frac_det_common_corr_gt05": float("nan"),
        }
    arr = np.asarray(corrs, dtype=float)
    return {
        "median_det_common_corr_abs": float(np.median(arr)),
        "p95_det_common_corr_abs": float(np.percentile(arr, 95.0)),
        "frac_det_common_corr_gt05": float(np.mean(arr > 0.5)),
    }


def _extract_case_metrics(rtc_path: Path, ptc_path: Path, array_name: str, row: RowChoice) -> dict[str, float]:
    array_id = ARRAY_TO_ID[array_name]
    with Dataset(rtc_path) as rtc_ds, Dataset(ptc_path) as ptc_ds:
        scan_i = _find_scan_row_index(ptc_ds, row.output_scan_index)
        start, stop = [int(v) for v in ptc_ds.variables["scan_indices"][scan_i]]
        apt_array = np.asarray(ptc_ds.variables["apt_array"][:], dtype=int)
        apt_nw = np.asarray(ptc_ds.variables["apt_nw"][:], dtype=int)
        apt_flag = np.asarray(ptc_ds.variables["apt_flag"][:], dtype=int)
        weights = np.asarray(ptc_ds.variables["weights"][scan_i], dtype=float)

        det_mask = (apt_array == array_id) & (apt_nw == row.network) & (apt_flag == 0) & np.isfinite(weights) & (weights > 0)
        det_idx = np.where(det_mask)[0]
        if det_idx.size == 0:
            raise RuntimeError(f"no detectors left for obs={row.obsnum} nw={row.network} array={array_name}")

        rtc_signal = np.asarray(rtc_ds.variables["signal"][start:stop, det_idx], dtype=float).T
        rtc_flags = np.asarray(rtc_ds.variables["flags"][start:stop, det_idx], dtype=int).T
        ptc_signal = np.asarray(ptc_ds.variables["signal"][start:stop, det_idx], dtype=float).T
        ptc_flags = np.asarray(ptc_ds.variables["flags"][start:stop, det_idx], dtype=int).T

        rtc_z = np.vstack([_robust_standardize(rtc_signal[i], rtc_flags[i]) for i in range(rtc_signal.shape[0])])
        ptc_z = np.vstack([_robust_standardize(ptc_signal[i], ptc_flags[i]) for i in range(ptc_signal.shape[0])])
        rtc_z[rtc_flags > 0] = np.nan
        ptc_z[ptc_flags > 0] = np.nan

        rtc_cm = _nanmedian_trace(rtc_z)
        ptc_cm = _nanmedian_trace(ptc_z)
        fs_hz = float(ptc_ds.variables["signal"].fsmp if hasattr(ptc_ds.variables["signal"], "fsmp") else 61.03558)

        mode = _masked_mode_metrics(ptc_z)
        coupling = _detector_common_coupling(ptc_z, ptc_cm)
        spec = _common_mode_spectrum_metrics(ptc_cm, fs_hz)

        return {
            "n_det": float(det_idx.size),
            "n_time": float(stop - start),
            "ptc_flagged_frac": float(np.mean(ptc_flags > 0)),
            "rtc_ptc_common_corr_abs": _corr_abs(rtc_cm, ptc_cm),
            "ptc_common_rms": float(np.nanstd(ptc_cm, ddof=1)) if np.sum(np.isfinite(ptc_cm)) >= 8 else float("nan"),
            "rtc_common_rms": float(np.nanstd(rtc_cm, ddof=1)) if np.sum(np.isfinite(rtc_cm)) >= 8 else float("nan"),
            **mode,
            **coupling,
            **spec,
        }


def _write_case_report(outpath: Path, rows: list[dict[str, object]], array_name: str) -> None:
    lines = [
        f"# {array_name} Common-Mode Focus Report",
        "",
        "This note compares the targeted problem/control rows using only unflagged samples for the common-trace and spectral metrics.",
        "",
        "Definitions:",
        "- `top_mode_frac_masked`: fraction of masked PTC detector variance in the first covariance eigenmode. Larger means a stronger surviving shared mode.",
        "- `median_det_common_corr_abs`: median absolute detector correlation with the masked PTC common trace. Larger means more row-wide coupling.",
        "- `rtc_ptc_common_corr_abs`: absolute correlation between masked RTC and PTC common traces. Larger means the common structure leaked through PCA more directly.",
        "- `cm_low_mid_ratio`: ratio of low-band (`0.05-0.5 Hz`) to mid-band (`0.5-2 Hz`) power in the masked PTC common trace.",
        "",
        "## Case Table",
        "",
        "| case | row | scan | corr_z | topmode_z | low/mid | tail4_z | flagged frac | ptc common rms | rtc/ptc common corr | masked top-mode frac | med det/common corr | p95 det/common corr | masked cm low/mid | peak freq [Hz] | peak prom |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        choice: RowChoice = row["choice"]  # type: ignore[assignment]
        metrics: dict[str, float] = row["metrics"]  # type: ignore[assignment]
        lines.append(
            "| {case} | {kind} | {scan} | {corr:.2f} | {topz:.1f} | {lowmid:.2f} | {tail4:.2f} | "
            "{flagfrac:.3f} | {ptcrms:.3f} | {rtccorr:.3f} | {topfrac:.3f} | {medcorr:.3f} | {p95corr:.3f} | "
            "{cmratio:.2f} | {peakf:.2f} | {peakp:.2f} |".format(
                case=f"{choice.obsnum}/nw{choice.network}",
                kind=choice.kind,
                scan=choice.output_scan_index,
                corr=choice.corr_z,
                topz=choice.topmode_z,
                lowmid=choice.lowmid,
                tail4=choice.tail4_z,
                flagfrac=metrics["ptc_flagged_frac"],
                ptcrms=metrics["ptc_common_rms"],
                rtccorr=metrics["rtc_ptc_common_corr_abs"],
                topfrac=metrics["top_mode_frac_masked"],
                medcorr=metrics["median_det_common_corr_abs"],
                p95corr=metrics["p95_det_common_corr_abs"],
                cmratio=metrics["cm_low_mid_ratio"],
                peakf=metrics["cm_peak_freq_hz"],
                peakp=metrics["cm_peak_prominence"],
            )
        )

    lines.extend(["", "## Case Readout", ""])
    grouped: dict[tuple[int, int], dict[str, dict[str, float] | RowChoice]] = {}
    for row in rows:
        choice: RowChoice = row["choice"]  # type: ignore[assignment]
        grouped.setdefault((choice.obsnum, choice.network), {})[choice.kind] = row  # type: ignore[index]

    for (obsnum, network), pair in grouped.items():
        bad = pair["problem"]
        ctrl = pair["control"]
        bad_choice: RowChoice = bad["choice"]  # type: ignore[index,assignment]
        ctrl_choice: RowChoice = ctrl["choice"]  # type: ignore[index,assignment]
        bad_m: dict[str, float] = bad["metrics"]  # type: ignore[index,assignment]
        ctrl_m: dict[str, float] = ctrl["metrics"]  # type: ignore[index,assignment]
        top_ratio = _safe_ratio(bad_m["top_mode_frac_masked"], ctrl_m["top_mode_frac_masked"])
        med_ratio = _safe_ratio(bad_m["median_det_common_corr_abs"], ctrl_m["median_det_common_corr_abs"])
        rtcptc_ratio = _safe_ratio(bad_m["rtc_ptc_common_corr_abs"], ctrl_m["rtc_ptc_common_corr_abs"])
        low_ratio = _safe_ratio(bad_m["cm_low_mid_ratio"], ctrl_m["cm_low_mid_ratio"])
        if np.isfinite(top_ratio) and np.isfinite(med_ratio) and top_ratio > 1.2 and med_ratio > 1.2:
            takeaway = "Problem row carries measurably stronger shared-mode structure than its control."
        elif np.isfinite(low_ratio) and low_ratio > 1.5:
            takeaway = "Problem row is distinguished mainly by excess low-frequency common-mode power."
        elif bad_choice.tail4_z > ctrl_choice.tail4_z:
            takeaway = "Problem row remains mixed: tail/non-Gaussian behavior matters as much as common-mode leakage."
        else:
            takeaway = "Problem/control separation is weak in shared-mode metrics; residual behavior is mixed."
        lines.extend(
            [
                f"### obs {obsnum} nw {network}",
                "",
                f"- Problem row scan `{bad_choice.output_scan_index}` vs control scan `{ctrl_choice.output_scan_index}`.",
                f"- `masked top_mode_frac`: `{bad_m['top_mode_frac_masked']:.3f}` vs `{ctrl_m['top_mode_frac_masked']:.3f}`.",
                f"- `median_det_common_corr_abs`: `{bad_m['median_det_common_corr_abs']:.3f}` vs `{ctrl_m['median_det_common_corr_abs']:.3f}`.",
                f"- `rtc_ptc_common_corr_abs`: `{bad_m['rtc_ptc_common_corr_abs']:.3f}` vs `{ctrl_m['rtc_ptc_common_corr_abs']:.3f}`.",
                f"- `masked cm_low_mid_ratio`: `{bad_m['cm_low_mid_ratio']:.2f}` vs `{ctrl_m['cm_low_mid_ratio']:.2f}`.",
                f"- Takeaway: {takeaway}",
                "",
            ]
        )

    outpath.write_text("\n".join(lines))


def _plot_summary(outpath: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, int], dict[str, dict[str, float] | RowChoice]] = {}
    for row in rows:
        choice: RowChoice = row["choice"]  # type: ignore[assignment]
        grouped.setdefault((choice.obsnum, choice.network), {})[choice.kind] = row  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("top_mode_frac_masked", "masked top-mode frac"),
        ("median_det_common_corr_abs", "median det/common corr"),
        ("rtc_ptc_common_corr_abs", "RTC/PTC common corr"),
        ("cm_low_mid_ratio", "masked low/mid ratio"),
    ]

    for ax, ((obsnum, network), pair) in zip(axes.flat, sorted(grouped.items())):
        bad = pair["problem"]["metrics"]  # type: ignore[index]
        ctrl = pair["control"]["metrics"]  # type: ignore[index]
        x = np.arange(len(metrics))
        width = 0.36
        bad_vals = [float(bad[key]) for key, _ in metrics]
        ctrl_vals = [float(ctrl[key]) for key, _ in metrics]
        ax.bar(x - width / 2, bad_vals, width=width, label="problem")
        ax.bar(x + width / 2, ctrl_vals, width=width, label="control")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
        ax.set_title(f"obs {obsnum} nw {network}")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Common-mode/coherence summary (unflagged samples only)")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="Reduction directory, e.g. .../reduced/redu12")
    ap.add_argument("--array", default="a1100", choices=sorted(ARRAY_TO_ID))
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="Target obs:nw pairs")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory")
    args = ap.parse_args()

    outdir = args.outdir or (args.run_dir / f"{args.array}_common_mode_focus")
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in args.targets:
        obsnum, network = _parse_target(spec)
        null_csv = args.run_dir / "pca_audit_focus" / f"null_obs{obsnum}_{args.array}" / "blank_sky_null_audit_detailed.csv"
        if not null_csv.exists():
            raise RuntimeError(f"missing null-audit detailed CSV: {null_csv}")
        df = _load_detail_csv(null_csv, obsnum=obsnum, network=network)
        bad, ctrl = _choose_rows(df)
        rtc_path = args.run_dir / str(obsnum) / "raw" / f"toltec_commissioning_science_{obsnum}_rtc_timestream.nc"
        ptc_path = args.run_dir / str(obsnum) / "raw" / f"toltec_commissioning_science_{obsnum}_ptc_timestream.nc"
        rows.append({"choice": bad, "metrics": _extract_case_metrics(rtc_path, ptc_path, args.array, bad)})
        rows.append({"choice": ctrl, "metrics": _extract_case_metrics(rtc_path, ptc_path, args.array, ctrl)})

    _write_case_report(outdir / "COMMON_MODE_FOCUS_REPORT.md", rows, args.array)
    _plot_summary(outdir / "common_mode_focus_summary.png", rows)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
