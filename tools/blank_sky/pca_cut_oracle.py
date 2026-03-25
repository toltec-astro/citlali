#!/usr/bin/env python3
"""Sweep bounded PCA cuts and score residuals with blank-sky audit metrics.

This is an offline "oracle" study for adaptive PCA depth selection. It uses
RTC mini timestreams as the pre-clean input, applies a Citlali-like masked PCA
subtraction for a small set of candidate `k` values, and scores each cleaned
scan/network row with the same residual metrics used by the blank-sky null
audits.

The intent is not to declare a final runtime policy directly. The intent is to
answer:

- for a given baseline cut, which nearby `k` values actually improve the
  residual coherence/common-mode behavior we care about?
- how often would an adaptive rule move away from the baseline?
- does the "best" `k` vary in a way that looks structured enough to justify an
  inline bounded adaptive selector?
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from blank_sky_null_audit import (
    _common_mode_spectrum_metrics,
    _eigen_metrics,
    _get_scan_templates,
    _nanmedian,
    _nanstd,
    _parse_networks,
    _parse_scans,
    _prepare_detector_matrix,
    _quartile_metrics,
    _resolve_obsnum,
    _row_value,
    _safe_ratio,
    _sample_pair_corr_metrics,
    _shape_metrics,
    _surrogate_metrics,
    _tail_metrics,
    _top_rows,
    _write_csv,
)


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("expected at least one integer")
    return out


def _rtc_files(redu_dir: Path) -> list[Path]:
    files = sorted(redu_dir.glob("*/raw/*_rtc_timestream.nc"))
    if not files:
        raise FileNotFoundError(f"no *_rtc_timestream.nc files found under {redu_dir}")
    return files


def _calc_cov_with_mask(scans: np.ndarray, flags: np.ndarray) -> np.ndarray:
    scans = np.asarray(scans, dtype=float)
    good = (np.asarray(flags) == 0).astype(float)
    det = scans * good
    denom = good.T @ good - 1.0
    numer = det.T @ det
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(denom > 0.0, numer / denom, 0.0)
    cov = np.asarray(cov, dtype=float)
    cov[~np.isfinite(cov)] = 0.0
    return cov


def _apply_masked_pca_cut(signal: np.ndarray, flags: np.ndarray, k: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    flags = np.asarray(flags)
    if signal.ndim != 2:
        raise ValueError("signal must be time x detector")
    if k <= 0 or signal.shape[1] < 2:
        return signal.copy()

    cov = _calc_cov_with_mask(signal, flags)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    k_use = max(0, min(int(k), evecs.shape[1]))
    if k_use == 0:
        return signal.copy()
    evecs_cut = evecs[:, :k_use]
    good = (flags == 0).astype(float)
    proj = (signal * good) @ evecs_cut
    model = proj @ evecs_cut.T
    cleaned = signal - model * good
    return np.asarray(cleaned, dtype=float)


def _score_terms(row: dict[str, object]) -> dict[str, float]:
    corr_z = _row_value(row, "med_abs_corr_surr_z")
    low_mid = _row_value(row, "cm_low_mid_ratio")
    tail4_z = _row_value(row, "tail4_binom_z")
    topmode_z = _row_value(row, "top_mode_surr_z")

    low_term = float(max(math.log2(low_mid), 0.0)) if np.isfinite(low_mid) and low_mid > 0 else float("nan")
    tail_term = float(max(tail4_z, 0.0)) if np.isfinite(tail4_z) else float("nan")
    corr_term = float(max(corr_z, 0.0)) if np.isfinite(corr_z) else float("nan")
    topmode_term = float(max(topmode_z, 0.0)) if np.isfinite(topmode_z) else float("nan")
    return {
        "oracle_corr_term": corr_term,
        "oracle_low_term": low_term,
        "oracle_tail_term": tail_term,
        "oracle_topmode_term": topmode_term,
    }


def _normalize_for_row(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        norm = np.full(vals.shape, np.nan, dtype=float)
        if np.any(finite):
            vmin = float(np.min(vals[finite]))
            vmax = float(np.max(vals[finite]))
            if vmax > vmin:
                norm[finite] = (vals[finite] - vmin) / (vmax - vmin)
            else:
                norm[finite] = 0.0
        out[f"{col}_norm"] = norm
    return out


def _score_candidates(
    df: pd.DataFrame,
    baseline_k: int,
    k_step: int,
    w_corr: float,
    w_low: float,
    w_tail: float,
    w_topmode: float,
    w_reg: float,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    term_cols = [
        "oracle_corr_term",
        "oracle_low_term",
        "oracle_tail_term",
        "oracle_topmode_term",
    ]
    for _, g in df.groupby(["obsnum", "scan", "network"], sort=False):
        gg = _normalize_for_row(g, term_cols)
        reg = np.abs(pd.to_numeric(gg["k"], errors="coerce") - float(baseline_k)) / max(float(k_step), 1.0)
        gg["oracle_reg_term"] = reg
        gg["oracle_score"] = (
            w_corr * gg["oracle_corr_term_norm"].fillna(1.0)
            + w_low * gg["oracle_low_term_norm"].fillna(1.0)
            + w_tail * gg["oracle_tail_term_norm"].fillna(1.0)
            + w_topmode * gg["oracle_topmode_term_norm"].fillna(1.0)
            + w_reg * gg["oracle_reg_term"].fillna(0.0)
        )
        groups.append(gg)
    return pd.concat(groups, ignore_index=True) if groups else df


def _make_metric_row(
    *,
    obsnum: str,
    scan: int,
    output_scan_index: int,
    network: int,
    k: int,
    prepared: dict[str, np.ndarray | float | int],
    fs_native: float,
    tel_el: np.ndarray,
    tel_az: np.ndarray,
    d_el: np.ndarray,
    d_az: np.ndarray,
    t: np.ndarray,
    n_pairs: int,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    x_centered = np.asarray(prepared["x_centered"], dtype=float)
    z = np.asarray(prepared["z"], dtype=float)
    valid = np.asarray(prepared["valid"], dtype=bool)
    z_valid = z[valid]
    if z_valid.size < 32:
        return None

    med_abs_corr, p95_abs_corr = _sample_pair_corr_metrics(z, int(n_pairs), rng)
    top_mode_frac, k90 = _eigen_metrics(z)
    surr_corr, surr_top = _surrogate_metrics(z, int(n_pairs), int(n_surrogates), rng)

    common_mode = np.median(x_centered, axis=1)
    fs_eff = fs_native / max(int(prepared["time_stride"]), 1)
    time_stride = max(int(prepared["time_stride"]), 1)
    n_time = int(prepared["n_time"])

    def _match_template(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)[::time_stride]
        if arr.size >= n_time:
            return arr[:n_time]
        out = np.full(n_time, np.nan, dtype=float)
        out[: arr.size] = arr
        return out

    tel_el = _match_template(tel_el)
    tel_az = _match_template(tel_az)
    d_el = _match_template(d_el)
    d_az = _match_template(d_az)
    t = _match_template(t)

    spec = _common_mode_spectrum_metrics(common_mode, fs_eff)
    quartile_sigma_ratio, quartile_tail4_range = _quartile_metrics(z, valid)
    skew, excess_kurtosis = _shape_metrics(z_valid)
    tail3 = _tail_metrics(z_valid, 3.0)
    tail4 = _tail_metrics(z_valid, 4.0)
    tail5 = _tail_metrics(z_valid, 5.0)
    pos4 = int(np.sum(z_valid > 4.0))
    neg4 = int(np.sum(z_valid < -4.0))
    pn_den = pos4 + neg4
    tail4_pos_neg_asym = float((pos4 - neg4) / pn_den) if pn_den > 0 else float("nan")

    row = {
        "obsnum": str(obsnum),
        "scan": int(scan),
        "output_scan_index": int(output_scan_index),
        "network": int(network),
        "k": int(k),
        "n_det_used": int(prepared["n_det"]),
        "n_time_used": int(prepared["n_time"]),
        "valid_frac": float(prepared["valid_frac"]),
        "fs_hz": float(fs_eff),
        "z_skew": skew,
        "z_excess_kurtosis": excess_kurtosis,
        "tail3_frac": tail3["frac"],
        "tail3_ratio": tail3["ratio"],
        "tail3_binom_z": tail3["binom_z"],
        "tail4_frac": tail4["frac"],
        "tail4_ratio": tail4["ratio"],
        "tail4_binom_z": tail4["binom_z"],
        "tail5_frac": tail5["frac"],
        "tail5_ratio": tail5["ratio"],
        "tail5_binom_z": tail5["binom_z"],
        "tail4_pos_neg_asym": tail4_pos_neg_asym,
        "med_abs_corr": med_abs_corr,
        "p95_abs_corr": p95_abs_corr,
        "top_mode_frac": top_mode_frac,
        "k90_corr_modes": k90,
        "med_abs_corr_surr_median": _nanmedian(surr_corr),
        "med_abs_corr_surr_std": _nanstd(surr_corr),
        "med_abs_corr_over_surr_median": _safe_ratio(med_abs_corr, _nanmedian(surr_corr)),
        "med_abs_corr_surr_z": _safe_ratio(med_abs_corr - _nanmedian(surr_corr), _nanstd(surr_corr)),
        "top_mode_surr_median": _nanmedian(surr_top),
        "top_mode_surr_std": _nanstd(surr_top),
        "top_mode_over_surr_median": _safe_ratio(top_mode_frac, _nanmedian(surr_top)),
        "top_mode_surr_z": _safe_ratio(top_mode_frac - _nanmedian(surr_top), _nanstd(surr_top)),
        "cm_corr_el": _safe_ratio(0.0, 0.0),  # filled below
        "cm_corr_del": _safe_ratio(0.0, 0.0),
        "cm_corr_az": _safe_ratio(0.0, 0.0),
        "cm_corr_daz": _safe_ratio(0.0, 0.0),
        "cm_corr_t": _safe_ratio(0.0, 0.0),
        "cm_bp_low": spec["cm_bp_low"],
        "cm_bp_mid": spec["cm_bp_mid"],
        "cm_bp_high": spec["cm_bp_high"],
        "cm_low_mid_ratio": spec["cm_low_mid_ratio"],
        "cm_high_mid_ratio": spec["cm_high_mid_ratio"],
        "cm_peak_freq_hz": spec["cm_peak_freq_hz"],
        "cm_peak_prominence": spec["cm_peak_prominence"],
        "quartile_sigma_ratio": quartile_sigma_ratio,
        "quartile_tail4_range": quartile_tail4_range,
    }
    from blank_sky_null_audit import _corr_abs  # local import to keep namespace small

    row["cm_corr_el"] = _corr_abs(common_mode, tel_el)
    row["cm_corr_del"] = _corr_abs(common_mode, d_el)
    row["cm_corr_az"] = _corr_abs(common_mode, tel_az)
    row["cm_corr_daz"] = _corr_abs(common_mode, d_az)
    row["cm_corr_t"] = _corr_abs(common_mode, t)
    row.update(_score_terms(row))
    return row


def _make_summary_by_k(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k, g in df.groupby("k"):
        rows.append(
            {
                "k": int(k),
                "n_rows": int(len(g)),
                "median_oracle_score": float(g["oracle_score"].median()),
                "median_corr_z": float(g["med_abs_corr_surr_z"].median()),
                "max_corr_z": float(g["med_abs_corr_surr_z"].max()),
                "median_low_mid": float(g["cm_low_mid_ratio"].median()),
                "median_tail4_z": float(g["tail4_binom_z"].median()),
                "median_topmode_z": float(g["top_mode_surr_z"].median()),
            }
        )
    rows.sort(key=lambda r: int(r["k"]))
    return rows


def _make_best_rows(df: pd.DataFrame, baseline_k: int) -> pd.DataFrame:
    idx = df.groupby(["obsnum", "scan", "network"])["oracle_score"].idxmin()
    best = df.loc[idx].copy()
    base = (
        df[df["k"] == int(baseline_k)][["obsnum", "scan", "network", "oracle_score", "med_abs_corr_surr_z", "cm_low_mid_ratio", "tail4_binom_z"]]
        .rename(
            columns={
                "oracle_score": "baseline_oracle_score",
                "med_abs_corr_surr_z": "baseline_corr_z",
                "cm_low_mid_ratio": "baseline_low_mid",
                "tail4_binom_z": "baseline_tail4_z",
            }
        )
        .copy()
    )
    merged = best.merge(base, on=["obsnum", "scan", "network"], how="left")
    merged["delta_oracle_score"] = merged["oracle_score"] - merged["baseline_oracle_score"]
    merged["delta_corr_z"] = merged["med_abs_corr_surr_z"] - merged["baseline_corr_z"]
    merged["delta_low_mid"] = merged["cm_low_mid_ratio"] - merged["baseline_low_mid"]
    merged["delta_tail4_z"] = merged["tail4_binom_z"] - merged["baseline_tail4_z"]
    return merged.sort_values(["oracle_score", "obsnum", "scan", "network"], ascending=[True, True, True, True])


def _write_report(
    outpath: Path,
    redu_dir: Path,
    array: str,
    source_run: str,
    cuts: list[int],
    baseline_k: int,
    detailed_df: pd.DataFrame,
    best_df: pd.DataFrame,
    summary_rows: list[dict[str, object]],
) -> None:
    best_counts = best_df["k"].value_counts().sort_index()
    top_changed = best_df[best_df["k"] != baseline_k].sort_values("delta_oracle_score").head(12)
    top_corr = detailed_df[detailed_df["k"] == baseline_k].nlargest(8, "med_abs_corr_surr_z")

    lines = [
        f"# PCA Cut Oracle: {array}",
        "",
        f"- Reduction: `{redu_dir}`",
        f"- Source run: `{source_run}`",
        f"- Candidate cuts: `{cuts}`",
        f"- Baseline cut: `{baseline_k}`",
        "",
        "This oracle uses RTC mini timestreams as pre-clean input, applies the",
        "same masked PCA subtraction model Citlali uses for `[nw]` cleaning,",
        "then scores each candidate cut with blank-sky residual metrics.",
        "",
        "Oracle score terms:",
        "- strongest weight: residual coherence-null (`med_abs_corr_surr_z`)",
        "- secondary weight: low-frequency common-mode leakage (`cm_low_mid_ratio`)",
        "- penalty: tail excess (`tail4_binom_z`)",
        "- regularization: distance from the baseline cut",
        "",
        "## Summary By Cut",
        "",
        "| k | n_rows | median oracle score | median corr z | max corr z | median low/mid | median tail4 z | median topmode z |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {k} | {n_rows} | {median_oracle_score:.3f} | {median_corr_z:.2f} | {max_corr_z:.2f} | "
            "{median_low_mid:.2f} | {median_tail4_z:.2f} | {median_topmode_z:.2f} |".format(**row)
        )

    lines.extend(["", "## Oracle Best-k Distribution", ""])
    for k, count in best_counts.items():
        lines.append(f"- `k={int(k)}`: {int(count)} row(s)")

    lines.extend(["", "## Best Rows That Move Away From Baseline", ""])
    if top_changed.empty:
        lines.append("- none")
    else:
        for _, row in top_changed.iterrows():
            lines.append(
                "- obs={obs} scan={scan} nw={nw}: best `k={k}` vs baseline `{base}` "
                "score `{score:.3f}` (`Δ={delta:.3f}`), corr_z `{corr:.2f}` (`Δ={dcorr:.2f}`), "
                "low/mid `{low:.2f}`, tail4_z `{tail:.2f}`".format(
                    obs=row["obsnum"],
                    scan=int(row["scan"]),
                    nw=int(row["network"]),
                    k=int(row["k"]),
                    base=int(baseline_k),
                    score=float(row["oracle_score"]),
                    delta=float(row["delta_oracle_score"]),
                    corr=float(row["med_abs_corr_surr_z"]),
                    dcorr=float(row["delta_corr_z"]),
                    low=float(row["cm_low_mid_ratio"]),
                    tail=float(row["tail4_binom_z"]),
                )
            )

    lines.extend(["", f"## Baseline `k={baseline_k}` Top Coherence Rows", ""])
    for _, row in top_corr.iterrows():
        lines.append(
            "- obs={obs} scan={scan} nw={nw} corr_z={corr:.2f} low/mid={low:.2f} tail4_z={tail:.2f}".format(
                obs=row["obsnum"],
                scan=int(row["scan"]),
                nw=int(row["network"]),
                corr=float(row["med_abs_corr_surr_z"]),
                low=float(row["cm_low_mid_ratio"]),
                tail=float(row["tail4_binom_z"]),
            )
        )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redu-dir", required=True, help="Reduction directory, e.g. .../reduced/redu11")
    ap.add_argument("--array", default="a1100", choices=["a1100", "a1400", "a2000"])
    ap.add_argument("--cuts", default="16,18,20,22,24", help="Comma list of candidate PCA cuts")
    ap.add_argument("--baseline-k", type=int, default=20)
    ap.add_argument("--scans", default="all")
    ap.add_argument("--networks", default="all")
    ap.add_argument("--utils-root", default="~/GitHub/toltec-data-product-utilities")
    ap.add_argument("--min-good-frac", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=180)
    ap.add_argument("--max-time", type=int, default=2048)
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--n-surrogates", type=int, default=6)
    ap.add_argument("--clip-z", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--w-corr", type=float, default=1.0)
    ap.add_argument("--w-low", type=float, default=0.4)
    ap.add_argument("--w-tail", type=float, default=0.2)
    ap.add_argument("--w-topmode", type=float, default=0.0)
    ap.add_argument("--w-reg", type=float, default=0.15)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    redu_dir = Path(os.path.expanduser(args.redu_dir)).resolve()
    outdir = Path(os.path.expanduser(args.outdir)).resolve() if args.outdir else (redu_dir / f"{args.array}_pca_cut_oracle")
    outdir.mkdir(parents=True, exist_ok=True)

    cuts = sorted(set(_parse_int_list(args.cuts)))
    if int(args.baseline_k) not in cuts:
        cuts.append(int(args.baseline_k))
        cuts = sorted(set(cuts))
    if len(cuts) < 2:
        raise ValueError("need at least two candidate cuts for an oracle sweep")
    k_step = min(abs(b - a) for a, b in zip(cuts[:-1], cuts[1:])) if len(cuts) > 1 else 1

    utils_root = Path(os.path.expanduser(args.utils_root)).resolve()
    if str(utils_root) not in sys.path:
        sys.path.insert(0, str(utils_root))
    from toltec_dp_utils.ToltecCitlaliTimestream import ToltecCitlaliTimestream  # pylint: disable=import-error

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for rtc_file in _rtc_files(redu_dir):
        obsnum = _resolve_obsnum(rtc_file, None)
        tcs = ToltecCitlaliTimestream(ncFile=str(rtc_file), array=args.array, load_data=False, interactive=False)
        if tcs.scan_indices is None:
            continue

        with netCDF4.Dataset(str(rtc_file)) as ds:
            output_scan_index = (
                np.asarray(ds.variables["output_scan_index"][:], dtype=int)
                if "output_scan_index" in ds.variables
                else np.arange(1, int(tcs.scan_indices.shape[0]) + 1, dtype=int)
            )
            apt_nw = np.asarray(ds.variables["apt_nw"][:], dtype=int)
            selected_global = np.asarray(tcs._det_index, dtype=int)  # noqa: SLF001
            selected_networks = apt_nw[selected_global]
            scans = _parse_scans(args.scans, int(tcs.scan_indices.shape[0]))
            networks = _parse_networks(args.networks, selected_networks)
            fs_native = float(tcs.sampleRate)

            for scan in scans:
                scan_idx = tcs.getScanIndices(scan)
                tel_el, tel_az, d_el, d_az, t = _get_scan_templates(ds, scan_idx)
                out_scan_idx = int(output_scan_index[scan]) if scan < output_scan_index.size else -1

                for nw in networks:
                    signal, _ = tcs._get_signal(scan_idx, network=nw)  # noqa: SLF001
                    flags, _ = tcs._get_flags(scan_idx, network=nw)  # noqa: SLF001
                    signal = np.asarray(signal, dtype=float)
                    flags = np.asarray(flags)
                    if signal.ndim != 2 or signal.shape[1] < 2:
                        continue

                    for k in cuts:
                        cleaned = _apply_masked_pca_cut(signal, flags, int(k))
                        prepared = _prepare_detector_matrix(
                            signal=cleaned,
                            flags=flags,
                            min_good_frac=float(args.min_good_frac),
                            max_det=args.max_det,
                            max_time=args.max_time,
                            clip_z=float(args.clip_z),
                        )
                        if prepared is None:
                            continue
                        row = _make_metric_row(
                            obsnum=obsnum,
                            scan=int(scan),
                            output_scan_index=out_scan_idx,
                            network=int(nw),
                            k=int(k),
                            prepared=prepared,
                            fs_native=fs_native,
                            tel_el=tel_el,
                            tel_az=tel_az,
                            d_el=d_el,
                            d_az=d_az,
                            t=t,
                            n_pairs=int(args.n_pairs),
                            n_surrogates=int(args.n_surrogates),
                            rng=rng,
                        )
                        if row is not None:
                            rows.append(row)

    if not rows:
        raise RuntimeError(f"no oracle rows generated for {redu_dir}")

    detailed_df = pd.DataFrame(rows)
    detailed_df = _score_candidates(
        detailed_df,
        baseline_k=int(args.baseline_k),
        k_step=int(k_step),
        w_corr=float(args.w_corr),
        w_low=float(args.w_low),
        w_tail=float(args.w_tail),
        w_topmode=float(args.w_topmode),
        w_reg=float(args.w_reg),
    )
    best_df = _make_best_rows(detailed_df, baseline_k=int(args.baseline_k))
    summary_rows = _make_summary_by_k(detailed_df)

    detailed_csv = outdir / "pca_cut_oracle_detailed.csv"
    best_csv = outdir / "pca_cut_oracle_best_by_row.csv"
    summary_csv = outdir / "pca_cut_oracle_summary_by_k.csv"
    report_md = outdir / "PCA_CUT_ORACLE.md"

    detailed_df.to_csv(detailed_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    _write_csv(summary_csv, summary_rows)
    _write_report(
        report_md,
        redu_dir=redu_dir,
        array=args.array,
        source_run=redu_dir.name,
        cuts=cuts,
        baseline_k=int(args.baseline_k),
        detailed_df=detailed_df,
        best_df=best_df,
        summary_rows=summary_rows,
    )

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {best_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
