#!/usr/bin/env python3
"""Summarize RTC line-audit families and plot representative PSDs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

try:
    from .blank_sky_null_audit import _write_csv
    from .rtc_line_audit import (
        _common_mode_from_centered,
        _masked_welch_psd,
        _network_signal_valid,
        _rolling_median,
    )
    from .mp_mode_estimator import _infer_dt_sec
except ImportError:
    from blank_sky_null_audit import _write_csv
    from rtc_line_audit import (
        _common_mode_from_centered,
        _masked_welch_psd,
        _network_signal_valid,
        _rolling_median,
    )
    from mp_mode_estimator import _infer_dt_sec


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _to_int(value: str) -> int:
    return int(float(value))


def _cluster_rows(rows: list[dict[str, str]], freq_key: str, tol_hz: float) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: _to_float(row[freq_key]))
    clusters: list[list[dict[str, str]]] = [[rows[0]]]
    for row in rows[1:]:
        prev = clusters[-1][-1]
        if abs(_to_float(row[freq_key]) - _to_float(prev[freq_key])) <= tol_hz:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _obsnum_to_rtc_files(redu_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(redu_dir.glob("*/raw/*_rtc_timestream.nc")):
        obsnum = path.name.split("_science_")[-1].split("_rtc_timestream.nc")[0]
        mapping[str(obsnum)] = path
    return mapping


def _find_scan_idx(ds: netCDF4.Dataset, output_scan_index: int) -> int | None:
    output_scan = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
    matches = np.where(output_scan == int(output_scan_index))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _extract_common_mode_psd(
    nc_file: Path,
    *,
    network: int,
    output_scan_index: int,
    min_good_frac: float,
    plot_max_det: int,
    segment_sec: float,
    min_segment_sec: float,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    with netCDF4.Dataset(nc_file) as ds:
        scan = _find_scan_idx(ds, output_scan_index)
        if scan is None:
            return None
        scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int)
        i0 = int(scan_indices[scan, 0])
        i1 = int(scan_indices[scan, 1])
        dt_sec = _infer_dt_sec(ds, i0, i1)
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            return None
        nw_all = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        det_sel = np.where(nw_all == int(network))[0]
        if det_sel.size == 0:
            return None
        signal = np.asarray(ds.variables["signal"][i0 : i1 + 1, det_sel], dtype=float)
        flags = np.asarray(ds.variables["flags"][i0 : i1 + 1, det_sel], dtype=np.int8)
        x_centered, valid, _, _ = _network_signal_valid(
            signal,
            flags,
            min_good_frac=float(min_good_frac),
            max_det=int(plot_max_det),
        )
        if x_centered.shape[1] == 0:
            return None
        cm = _common_mode_from_centered(x_centered, valid)
        cm_valid = np.sum(valid, axis=1) >= max(4, int(0.25 * x_centered.shape[1]))
        freq, psd, n_win = _masked_welch_psd(
            cm,
            cm_valid,
            dt_sec,
            segment_sec=float(segment_sec),
            min_segment_sec=float(min_segment_sec),
            overlap_frac=float(overlap_frac),
        )
        if freq is None or psd is None:
            return None
        continuum = _rolling_median(np.asarray(psd, dtype=float), 6)
        meta = {
            "n_det_used": float(x_centered.shape[1]),
            "n_psd_windows": float(n_win),
            "fs_hz": float(1.0 / dt_sec),
        }
        return np.asarray(freq), np.asarray(psd), np.asarray(continuum), meta


def _extract_detector_psd(
    nc_file: Path,
    *,
    network: int,
    uid: int,
    output_scan_index: int,
    segment_sec: float,
    min_segment_sec: float,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    with netCDF4.Dataset(nc_file) as ds:
        scan = _find_scan_idx(ds, output_scan_index)
        if scan is None:
            return None
        scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int)
        i0 = int(scan_indices[scan, 0])
        i1 = int(scan_indices[scan, 1])
        dt_sec = _infer_dt_sec(ds, i0, i1)
        if not np.isfinite(dt_sec) or dt_sec <= 0:
            return None
        nw_all = np.asarray(ds.variables["apt_nw"][:], dtype=int)
        uid_all = np.asarray(ds.variables["apt_uid"][:], dtype=int)
        det_sel = np.where((nw_all == int(network)) & (uid_all == int(uid)))[0]
        if det_sel.size == 0:
            return None
        j = int(det_sel[0])
        signal = np.asarray(ds.variables["signal"][i0 : i1 + 1, j], dtype=float)
        flags = np.asarray(ds.variables["flags"][i0 : i1 + 1, j], dtype=np.int8)
        valid = np.isfinite(signal) & (flags == 0)
        vals = signal[valid]
        if vals.size < 16:
            return None
        med = float(np.median(vals))
        centered = signal - med
        freq, psd, n_win = _masked_welch_psd(
            centered,
            valid,
            dt_sec,
            segment_sec=float(segment_sec),
            min_segment_sec=float(min_segment_sec),
            overlap_frac=float(overlap_frac),
        )
        if freq is None or psd is None:
            return None
        continuum = _rolling_median(np.asarray(psd, dtype=float), 6)
        meta = {
            "n_psd_windows": float(n_win),
            "fs_hz": float(1.0 / dt_sec),
        }
        return np.asarray(freq), np.asarray(psd), np.asarray(continuum), meta


def _make_shared_family_rows(rows: list[dict[str, str]], tol_hz: float) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    notch_rows = [row for row in rows if _to_int(row["recommend_notch"]) == 1]
    out: list[dict[str, object]] = []
    reps: list[dict[str, str]] = []
    for cluster in _cluster_rows(notch_rows, "cluster_freq_hz", tol_hz):
        freqs = [_to_float(row["cluster_freq_hz"]) for row in cluster]
        det_fracs = [_to_float(row["detector_frac"]) for row in cluster]
        med_proms = [_to_float(row["median_prominence"]) for row in cluster]
        cm_proms = [_to_float(row["common_mode_prominence"]) for row in cluster if np.isfinite(_to_float(row["common_mode_prominence"]))]
        scores = [_to_float(row["notch_score"]) for row in cluster]
        rep = max(
            cluster,
            key=lambda row: (
                np.isfinite(_to_float(row["common_mode_prominence"])),
                _to_float(row["common_mode_prominence"]) if np.isfinite(_to_float(row["common_mode_prominence"])) else -1.0,
                _to_float(row["notch_score"]),
            ),
        )
        reps.append(rep)
        out.append(
            {
                "family_freq_hz": float(np.median(freqs)),
                "n_rows": int(len(cluster)),
                "n_obsnums": int(len({str(row["obsnum"]) for row in cluster})),
                "n_networks": int(len({int(row["network"]) for row in cluster})),
                "median_detector_frac": float(np.median(det_fracs)),
                "median_prominence": float(np.median(med_proms)),
                "max_common_mode_prominence": float(max(cm_proms) if cm_proms else float("nan")),
                "score_sum": float(np.sum(scores)),
                "representative_obsnum": str(rep["obsnum"]),
                "representative_output_scan_index": int(_to_int(rep["output_scan_index"])),
                "representative_network": int(_to_int(rep["network"])),
            }
        )
    out.sort(key=lambda row: (-float(row["score_sum"]), -int(row["n_rows"]), float(row["family_freq_hz"])))
    return out, reps


def _pick_bad_detector_representatives(
    bad_rows: list[dict[str, str]],
    det_peak_rows: list[dict[str, str]],
    tol_hz: float,
) -> list[dict[str, object]]:
    peak_index: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in det_peak_rows:
        key = (str(row["obsnum"]), _to_int(row["network"]), _to_int(row["uid"]))
        peak_index[key].append(row)

    out: list[dict[str, object]] = []
    for row in bad_rows:
        if _to_int(row["recommend_bad_detector"]) != 1:
            continue
        obsnum = str(row["obsnum"])
        nw = _to_int(row["network"])
        uid = _to_int(row["uid"])
        dom_freq = _to_float(row["dominant_freq_hz"])
        matches = [
            r for r in peak_index.get((obsnum, nw, uid), [])
            if abs(_to_float(r["freq_hz"]) - dom_freq) <= tol_hz
        ]
        if matches:
            rep = max(matches, key=lambda r: _to_float(r["prominence"]))
            out.append(
                {
                    "obsnum": obsnum,
                    "network": nw,
                    "uid": uid,
                    "dominant_freq_hz": dom_freq,
                    "n_scan_hits": _to_int(row["n_scan_hits"]),
                    "median_prominence": _to_float(row["median_prominence"]),
                    "median_line_power_frac": _to_float(row["median_line_power_frac"]),
                    "median_cluster_detector_frac": _to_float(row["median_cluster_detector_frac"]),
                    "representative_output_scan_index": _to_int(rep["output_scan_index"]),
                    "representative_prominence": _to_float(rep["prominence"]),
                }
            )
    out.sort(key=lambda r: (-int(r["n_scan_hits"]), -float(r["median_prominence"]), str(r["obsnum"]), int(r["uid"])))
    return out


def _plot_shared_psds(
    rows: list[dict[str, object]],
    rtc_files: dict[str, Path],
    outpath: Path,
    args: argparse.Namespace,
) -> None:
    n_plot = min(len(rows), int(args.n_shared_plots))
    if n_plot <= 0:
        return
    ncols = 2
    nrows = int(math.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n_plot:]:
        ax.axis("off")

    for ax, row in zip(axes_flat, rows[:n_plot]):
        obsnum = str(row["representative_obsnum"])
        nc_file = rtc_files.get(obsnum)
        if nc_file is None:
            ax.text(0.5, 0.5, f"missing RTC file for obs {obsnum}", ha="center", va="center")
            ax.axis("off")
            continue
        result = _extract_common_mode_psd(
            nc_file,
            network=int(row["representative_network"]),
            output_scan_index=int(row["representative_output_scan_index"]),
            min_good_frac=float(args.min_good_frac),
            plot_max_det=int(args.plot_max_det),
            segment_sec=float(args.segment_sec),
            min_segment_sec=float(args.min_segment_sec),
            overlap_frac=float(args.overlap_frac),
        )
        if result is None:
            ax.text(0.5, 0.5, "PSD unavailable", ha="center", va="center")
            ax.axis("off")
            continue
        freq, psd, continuum, meta = result
        f0 = float(row["family_freq_hz"])
        mask = (freq >= max(0.0, f0 - float(args.line_window_hz))) & (freq <= f0 + float(args.line_window_hz))
        ax.semilogy(freq[mask], psd[mask], color="C0", label="common mode PSD")
        ax.semilogy(freq[mask], continuum[mask], color="0.5", linestyle="--", label="local continuum")
        ax.axvline(f0, color="C3", linestyle=":", linewidth=1.5)
        ax.set_title(
            f"obs {obsnum} scan {int(row['representative_output_scan_index'])} nw {int(row['representative_network'])}\n"
            f"family {f0:.3f} Hz rows={int(row['n_rows'])} score_sum={float(row['score_sum']):.1f}"
        )
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("RTC Shared-Line Representative PSDs", fontsize=18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_bad_detector_psds(
    rows: list[dict[str, object]],
    rtc_files: dict[str, Path],
    outpath: Path,
    args: argparse.Namespace,
) -> None:
    n_plot = min(len(rows), int(args.n_bad_detector_plots))
    if n_plot <= 0:
        return
    ncols = 2
    nrows = int(math.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n_plot:]:
        ax.axis("off")

    for ax, row in zip(axes_flat, rows[:n_plot]):
        obsnum = str(row["obsnum"])
        nc_file = rtc_files.get(obsnum)
        if nc_file is None:
            ax.text(0.5, 0.5, f"missing RTC file for obs {obsnum}", ha="center", va="center")
            ax.axis("off")
            continue
        result = _extract_detector_psd(
            nc_file,
            network=int(row["network"]),
            uid=int(row["uid"]),
            output_scan_index=int(row["representative_output_scan_index"]),
            segment_sec=float(args.segment_sec),
            min_segment_sec=float(args.min_segment_sec),
            overlap_frac=float(args.overlap_frac),
        )
        if result is None:
            ax.text(0.5, 0.5, "PSD unavailable", ha="center", va="center")
            ax.axis("off")
            continue
        freq, psd, continuum, _meta = result
        f0 = float(row["dominant_freq_hz"])
        mask = (freq >= max(0.0, f0 - float(args.line_window_hz))) & (freq <= f0 + float(args.line_window_hz))
        ax.semilogy(freq[mask], psd[mask], color="C0", label="detector PSD")
        ax.semilogy(freq[mask], continuum[mask], color="0.5", linestyle="--", label="local continuum")
        ax.axvline(f0, color="C3", linestyle=":", linewidth=1.5)
        ax.set_title(
            f"obs {obsnum} uid {int(row['uid'])} nw {int(row['network'])} scan {int(row['representative_output_scan_index'])}\n"
            f"{f0:.3f} Hz hits={int(row['n_scan_hits'])} med prom={float(row['median_prominence']):.1f}"
        )
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("RTC Bad-Detector Representative PSDs", fontsize=18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _write_markdown(
    outpath: Path,
    *,
    shared_rows: list[dict[str, object]],
    bad_rows: list[dict[str, object]],
    shared_png: str,
    bad_png: str,
) -> None:
    lines = [
        "# RTC Line Family Report",
        "",
        "This report aggregates RTC line-audit results across a whole reduction and",
        "provides representative PSD plots for:",
        "",
        "- broad shared line families that may justify notch filtering",
        "- recurrent detector-local lines that may justify bad-detector flagging",
        "",
        f"- shared PSD gallery: `{shared_png}`",
        f"- bad-detector PSD gallery: `{bad_png}`",
        "",
        "## Top Shared Line Families",
        "",
        "| family freq [Hz] | rows | obsnums | networks | median det frac | median prom | max cm prom | score sum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shared_rows[:12]:
        lines.append(
            f"| {float(row['family_freq_hz']):.3f} | {int(row['n_rows'])} | {int(row['n_obsnums'])} | "
            f"{int(row['n_networks'])} | {float(row['median_detector_frac']):.3f} | "
            f"{float(row['median_prominence']):.1f} | {float(row['max_common_mode_prominence']):.1f} | "
            f"{float(row['score_sum']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Top Bad-Detector Candidates",
            "",
            "| obsnum | nw | uid | freq [Hz] | scan hits | median prom | line power frac | median cluster det frac |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bad_rows[:20]:
        lines.append(
            f"| {row['obsnum']} | {int(row['network'])} | {int(row['uid'])} | {float(row['dominant_freq_hz']):.3f} | "
            f"{int(row['n_scan_hits'])} | {float(row['median_prominence']):.1f} | "
            f"{float(row['median_line_power_frac']):.3f} | {float(row['median_cluster_detector_frac']):.3f} |"
        )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redu-dir", type=Path, required=True, help="Reduction directory containing rtc_line_audit outputs.")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Override audit directory. Defaults to <redu-dir>/rtc_line_audit.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory. Defaults to <redu-dir>/rtc_line_family_report.")
    parser.add_argument("--family-tol-hz", type=float, default=0.5)
    parser.add_argument("--line-window-hz", type=float, default=4.0)
    parser.add_argument("--n-shared-plots", type=int, default=6)
    parser.add_argument("--n-bad-detector-plots", type=int, default=6)
    parser.add_argument("--min-good-frac", type=float, default=0.8)
    parser.add_argument("--plot-max-det", type=int, default=256)
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--overlap-frac", type=float, default=0.5)
    args = parser.parse_args()

    redu_dir = args.redu_dir.expanduser().resolve()
    audit_dir = (args.audit_dir.expanduser().resolve() if args.audit_dir is not None else (redu_dir / "rtc_line_audit"))
    outdir = (args.outdir.expanduser().resolve() if args.outdir is not None else (redu_dir / "rtc_line_family_report"))
    outdir.mkdir(parents=True, exist_ok=True)

    scan_rows = _read_csv(audit_dir / "rtc_line_audit_scan_network.csv")
    det_peak_rows = _read_csv(audit_dir / "rtc_line_audit_detector_peaks.csv")
    bad_rows = _read_csv(audit_dir / "rtc_line_audit_bad_detectors.csv")

    shared_rows, _shared_reps = _make_shared_family_rows(scan_rows, float(args.family_tol_hz))
    bad_rep_rows = _pick_bad_detector_representatives(bad_rows, det_peak_rows, float(args.family_tol_hz))

    _write_csv(outdir / "rtc_line_family_summary.csv", shared_rows)
    _write_csv(outdir / "rtc_line_bad_detector_representatives.csv", bad_rep_rows)

    rtc_files = _obsnum_to_rtc_files(redu_dir)
    shared_png = "rtc_shared_line_psd_gallery.png"
    bad_png = "rtc_bad_detector_psd_gallery.png"
    _plot_shared_psds(shared_rows, rtc_files, outdir / shared_png, args)
    _plot_bad_detector_psds(bad_rep_rows, rtc_files, outdir / bad_png, args)
    _write_markdown(outdir / "RTC_LINE_FAMILY_REPORT.md", shared_rows=shared_rows, bad_rows=bad_rep_rows, shared_png=shared_png, bad_png=bad_png)

    print(f"Wrote {outdir / 'rtc_line_family_summary.csv'}")
    print(f"Wrote {outdir / 'rtc_line_bad_detector_representatives.csv'}")
    print(f"Wrote {outdir / shared_png}")
    print(f"Wrote {outdir / bad_png}")
    print(f"Wrote {outdir / 'RTC_LINE_FAMILY_REPORT.md'}")


if __name__ == "__main__":
    main()
