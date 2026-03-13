#!/usr/bin/env python3
"""Summarize and plot detector clusters in focal-plane and sky coordinates."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np


RAD_TO_ARCSEC = 206264.80624709636


def _parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _infer_scan_from_name(path: Path) -> int | None:
    match = re.search(r"_scan(\d+)_", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _offset_arcsec(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    dx = (ra - ra0) * np.cos(dec0) * RAD_TO_ARCSEC
    dy = (dec - dec0) * RAD_TO_ARCSEC
    return dx, dy


def _choose_clusters(
    rows: list[dict[str, object]],
    cluster_ids: list[int],
    min_cluster_size: int,
    top_k: int,
    include_flagged: bool,
) -> list[int]:
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        good = int(row["good_samples"]) > 0
        if good or include_flagged:
            counts[int(row["cluster_id"])] += 1
    if cluster_ids:
        return sorted([cid for cid in cluster_ids if cid in counts])
    selected = [cid for cid, n in counts.items() if n >= min_cluster_size]
    if selected:
        return sorted(selected, key=lambda cid: (-counts[cid], cid))[:top_k]
    return [cid for cid, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def _read_detector_csv(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "local_det_index": int(row["local_det_index"]),
                    "global_det_index": int(row["global_det_index"]),
                    "cluster_id": int(row["cluster_id"]),
                    "order_rank": int(row["order_rank"]),
                    "apt_uid": float(row["apt_uid"]),
                    "apt_array": int(float(row["apt_array"])),
                    "apt_nw": int(float(row["apt_nw"])),
                    "apt_flag": int(float(row["apt_flag"])),
                    "apt_tone_freq_hz": float(row["apt_tone_freq_hz"]),
                    "good_samples": int(float(row["good_samples"])),
                }
            )
    if not rows:
        raise RuntimeError(f"no rows in detector csv: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols = sorted(set().union(*[row.keys() for row in rows]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_focal_plane(
    outpath: Path,
    apt_x: np.ndarray,
    apt_y: np.ndarray,
    selected_idx: np.ndarray,
    selected_cluster_ids: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    ax.scatter(apt_x, apt_y, s=5, color="0.80", alpha=0.8, linewidths=0, label="all dets")

    uniq = sorted(int(v) for v in np.unique(selected_cluster_ids))
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, cid in enumerate(uniq):
        mask = selected_cluster_ids == cid
        idx = selected_idx[mask]
        ax.scatter(apt_x[idx], apt_y[idx], s=18, color=cmap(i), linewidths=0, label=f"cluster {cid}")

    ax.set_xlabel("apt_x_t [arcsec]")
    ax.set_ylabel("apt_y_t [arcsec]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if uniq:
        ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_sky_tracks(
    outpath: Path,
    cluster_tracks: list[dict[str, object]],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    cmap = plt.get_cmap("tab20", max(len(cluster_tracks), 1))
    for i, track in enumerate(cluster_tracks):
        x = np.asarray(track["x_track_arcsec"], dtype=float)
        y = np.asarray(track["y_track_arcsec"], dtype=float)
        ax.plot(x, y, lw=1.2, alpha=0.9, color=cmap(i), label=f"cluster {track['cluster_id']}")
        ax.scatter([x[0]], [y[0]], s=12, color=cmap(i))
    ax.set_xlabel("dRA*cos(dec) [arcsec]")
    ax.set_ylabel("dDec [arcsec]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if cluster_tracks:
        ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _write_note(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ptc", required=True, help="Path to *_ptc_timestream.nc")
    ap.add_argument("--detector-csv", required=True, help="Detector cluster CSV from analyze_timestream_correlations.py")
    ap.add_argument("--scan", type=int, default=None, help="Internal scan index. Default: infer from detector CSV filename.")
    ap.add_argument("--cluster-ids", default=None, help="Optional comma-separated cluster IDs to include")
    ap.add_argument("--min-cluster-size", type=int, default=2, help="Default cluster-size cutoff when auto-selecting")
    ap.add_argument("--top-k", type=int, default=12, help="Max clusters to plot when auto-selecting")
    ap.add_argument("--include-flagged", action="store_true", help="Include zero-good-sample detectors in cluster selection")
    ap.add_argument("--outdir", default=None, help="Default: <detector csv stem>_localized")
    args = ap.parse_args()

    ptc_path = Path(args.ptc).expanduser().resolve()
    detector_csv = Path(args.detector_csv).expanduser().resolve()
    if not ptc_path.exists():
        raise FileNotFoundError(f"missing ptc file: {ptc_path}")
    if not detector_csv.exists():
        raise FileNotFoundError(f"missing detector csv: {detector_csv}")

    scan = args.scan
    if scan is None:
        scan = _infer_scan_from_name(detector_csv)
    if scan is None:
        raise ValueError("unable to infer scan from detector CSV name; pass --scan explicitly")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else detector_csv.with_name(detector_csv.stem + "_localized")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_detector_csv(detector_csv)
    selected_clusters = _choose_clusters(
        rows=rows,
        cluster_ids=_parse_int_list(args.cluster_ids),
        min_cluster_size=max(1, int(args.min_cluster_size)),
        top_k=max(1, int(args.top_k)),
        include_flagged=bool(args.include_flagged),
    )
    if not selected_clusters:
        raise RuntimeError("no clusters selected")

    with netCDF4.Dataset(str(ptc_path), "r") as ds:
        scan_indices = np.asarray(ds.variables["scan_indices"][:], dtype=int)
        if scan < 0 or scan >= scan_indices.shape[0]:
            raise ValueError(f"scan {scan} out of range [0, {scan_indices.shape[0]})")
        i0, i1 = [int(v) for v in scan_indices[scan, :2]]
        scan_slice = slice(i0, i1 + 1)

        apt_x = np.asarray(ds.variables["apt_x_t"][:], dtype=float)
        apt_y = np.asarray(ds.variables["apt_y_t"][:], dtype=float)
        apt_uid = np.asarray(ds.variables["apt_uid"][:], dtype=float)
        apt_tone = np.asarray(ds.variables["apt_tone_freq"][:], dtype=float)
        have_sky = "det_ra" in ds.variables and "det_dec" in ds.variables and "SourceRa" in ds.variables and "SourceDec" in ds.variables
        if have_sky:
            det_ra = np.asarray(ds.variables["det_ra"][scan_slice, :], dtype=float)
            det_dec = np.asarray(ds.variables["det_dec"][scan_slice, :], dtype=float)
            source_ra = float(np.asarray(ds.variables["SourceRa"][:]).reshape(-1)[0])
            source_dec = float(np.asarray(ds.variables["SourceDec"][:]).reshape(-1)[0])
        else:
            det_ra = None
            det_dec = None
            source_ra = float("nan")
            source_dec = float("nan")

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cid = int(row["cluster_id"])
        if cid in selected_clusters:
            grouped[cid].append(row)

    summary_rows: list[dict[str, object]] = []
    cluster_tracks: list[dict[str, object]] = []
    plot_indices: list[int] = []
    plot_cluster_ids: list[int] = []

    for cid in selected_clusters:
        rr = grouped.get(cid, [])
        if not rr:
            continue
        idx_all = np.asarray([int(row["global_det_index"]) for row in rr], dtype=int)
        idx_good = np.asarray([int(row["global_det_index"]) for row in rr if int(row["good_samples"]) > 0], dtype=int)
        idx_use = idx_good if idx_good.size > 0 else idx_all

        plot_indices.extend(idx_use.tolist())
        plot_cluster_ids.extend([cid] * idx_use.size)

        x_med = float(np.median(apt_x[idx_use]))
        y_med = float(np.median(apt_y[idx_use]))
        tone_vals = apt_tone[idx_use]
        uid_vals = apt_uid[idx_use]

        if have_sky:
            ra_cluster = det_ra[:, idx_use]
            dec_cluster = det_dec[:, idx_use]
            ra_track = np.median(ra_cluster, axis=1)
            dec_track = np.median(dec_cluster, axis=1)
            dx_track, dy_track = _offset_arcsec(ra_track, dec_track, source_ra, source_dec)

            cluster_tracks.append(
                {
                    "cluster_id": cid,
                    "x_track_arcsec": dx_track,
                    "y_track_arcsec": dy_track,
                }
            )
            sky_x_med = float(np.median(dx_track))
            sky_y_med = float(np.median(dy_track))
            sky_x_span = float(np.max(dx_track) - np.min(dx_track))
            sky_y_span = float(np.max(dy_track) - np.min(dy_track))
            sky_track_rms = float(np.sqrt(np.mean((dx_track - np.median(dx_track)) ** 2 + (dy_track - np.median(dy_track)) ** 2)))
        else:
            sky_x_med = float("nan")
            sky_y_med = float("nan")
            sky_x_span = float("nan")
            sky_y_span = float("nan")
            sky_track_rms = float("nan")

        summary_rows.append(
            {
                "cluster_id": cid,
                "n_det_total": int(idx_all.size),
                "n_det_good": int(idx_good.size),
                "n_det_used": int(idx_use.size),
                "good_frac": float(idx_good.size / idx_all.size) if idx_all.size > 0 else float("nan"),
                "apt_x_med_arcsec": x_med,
                "apt_y_med_arcsec": y_med,
                "apt_x_span_arcsec": float(np.max(apt_x[idx_use]) - np.min(apt_x[idx_use])) if idx_use.size else float("nan"),
                "apt_y_span_arcsec": float(np.max(apt_y[idx_use]) - np.min(apt_y[idx_use])) if idx_use.size else float("nan"),
                "tone_med_hz": float(np.median(tone_vals)),
                "tone_span_hz": float(np.max(tone_vals) - np.min(tone_vals)) if tone_vals.size else float("nan"),
                "uid_min": float(np.min(uid_vals)) if uid_vals.size else float("nan"),
                "uid_max": float(np.max(uid_vals)) if uid_vals.size else float("nan"),
                "sky_x_med_arcsec": sky_x_med,
                "sky_y_med_arcsec": sky_y_med,
                "sky_x_span_arcsec": sky_x_span,
                "sky_y_span_arcsec": sky_y_span,
                "sky_track_rms_arcsec": sky_track_rms,
            }
        )

    summary_rows.sort(key=lambda row: (-int(row["n_det_used"]), int(row["cluster_id"])))
    _write_csv(outdir / "cluster_localization_summary.csv", summary_rows)

    _plot_focal_plane(
        outpath=outdir / "cluster_focal_plane.png",
        apt_x=apt_x,
        apt_y=apt_y,
        selected_idx=np.asarray(plot_indices, dtype=int),
        selected_cluster_ids=np.asarray(plot_cluster_ids, dtype=int),
        title=f"{detector_csv.stem} focal-plane localization",
    )
    note_lines = [
        f"PTC file: {ptc_path}",
        f"detector csv: {detector_csv}",
        f"scan: {scan}",
        f"selected clusters: {','.join(str(cid) for cid in selected_clusters)}",
    ]
    if cluster_tracks:
        _plot_sky_tracks(
            outpath=outdir / "cluster_sky_tracks.png",
            cluster_tracks=cluster_tracks,
            title=f"{detector_csv.stem} sky-track localization",
        )
        note_lines.append("sky localization: enabled")
    else:
        note_lines.append("sky localization: unavailable (PTC file missing det_ra/det_dec or SourceRa/SourceDec)")
    _write_note(outdir / "README.txt", note_lines)

    print(f"Wrote {outdir / 'cluster_localization_summary.csv'}")
    print(f"Wrote {outdir / 'cluster_focal_plane.png'}")
    if cluster_tracks:
        print(f"Wrote {outdir / 'cluster_sky_tracks.png'}")
    print(f"Wrote {outdir / 'README.txt'}")


if __name__ == "__main__":
    main()
