#!/usr/bin/env python3
"""Plot per-chunk scan-speed histograms from TolTEC telescope files.

The speed is computed in Citlali's delta-source altaz frame:

    alt_phys = TelElAct - SourceEl - TelElCor
    az_phys  = cos(TelElAct - TelElCor) * (TelAzAct - SourceAz) - TelAzCor

For each telescope file, the script splits the valid timeline into fixed
duration chunks and records a robust peak speed per chunk.
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


DEFAULT_ROOT = Path(
    "/Users/gwilson/work_toltec/local_data/2025-C1-COM-21/AzTEC-C1/reduced"
)


@dataclass(frozen=True)
class ChunkSpeed:
    obsnum: str
    chunk_index: int
    t_start: float
    t_stop: float
    speed_arcsec_s: float
    median_speed_arcsec_s: float
    n_samples: int


def _read_var(ds: Dataset, name: str) -> np.ndarray:
    if name not in ds.variables:
        raise KeyError(f"{name!r} not found in {ds.filepath()}")
    return np.asarray(ds.variables[name][:], dtype=float)


def _obsnum_from_path(path: Path) -> str:
    with Dataset(path) as ds:
        name = "Header.TelescopeBackend.ObsNum"
        if name in ds.variables:
            return str(int(np.asarray(ds.variables[name][()]).item()))
    parts = path.name.split("_")
    for part in parts:
        if part.isdigit() and len(part) >= 5:
            return part
    return path.stem


def _delta_source_altaz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        time = _read_var(ds, "Data.TelescopeBackend.TelTime")
        tel_az = np.unwrap(_read_var(ds, "Data.TelescopeBackend.TelAzAct"))
        tel_el = _read_var(ds, "Data.TelescopeBackend.TelElAct")
        src_az = np.unwrap(_read_var(ds, "Data.TelescopeBackend.SourceAz"))
        src_el = _read_var(ds, "Data.TelescopeBackend.SourceEl")
        az_cor = _read_var(ds, "Data.TelescopeBackend.TelAzCor")
        el_cor = _read_var(ds, "Data.TelescopeBackend.TelElCor")

    az_diff = tel_az - src_az
    az_phys = np.cos(tel_el - el_cor) * az_diff - az_cor
    alt_phys = tel_el - src_el - el_cor
    return time, az_phys, alt_phys


def _instantaneous_speed_arcsec_s(
    time: np.ndarray,
    az_phys: np.ndarray,
    alt_phys: np.ndarray,
    *,
    max_dt_s: float,
    max_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(time)
    daz = np.diff(az_phys)
    dalt = np.diff(alt_phys)
    mid_time = 0.5 * (time[:-1] + time[1:])

    valid = (
        np.isfinite(dt)
        & np.isfinite(daz)
        & np.isfinite(dalt)
        & (dt > 0)
        & (dt <= max_dt_s)
        & (np.abs(daz) <= max_step_rad)
        & (np.abs(dalt) <= max_step_rad)
    )
    speed = np.full_like(dt, np.nan, dtype=float)
    speed[valid] = np.hypot(daz[valid], dalt[valid]) / dt[valid] * 206264.80624709636
    return mid_time, speed


def collect_chunk_speeds(
    paths: list[Path],
    *,
    chunk_sec: float,
    peak_percentile: float,
    min_samples: int,
    max_dt_s: float,
    max_step_rad: float,
) -> list[ChunkSpeed]:
    rows: list[ChunkSpeed] = []
    for path in paths:
        obsnum = _obsnum_from_path(path)
        time, az_phys, alt_phys = _delta_source_altaz(path)
        mid_time, speed = _instantaneous_speed_arcsec_s(
            time, az_phys, alt_phys, max_dt_s=max_dt_s, max_step_rad=max_step_rad
        )

        t0 = float(time[0])
        t1 = float(time[-1])
        n_chunks = int(np.floor((t1 - t0) / chunk_sec))
        for chunk_index in range(n_chunks):
            lo = t0 + chunk_index * chunk_sec
            hi = lo + chunk_sec
            mask = (mid_time >= lo) & (mid_time < hi) & np.isfinite(speed)
            if int(mask.sum()) < min_samples:
                continue
            chunk_speed = speed[mask]
            rows.append(
                ChunkSpeed(
                    obsnum=obsnum,
                    chunk_index=chunk_index,
                    t_start=lo,
                    t_stop=hi,
                    speed_arcsec_s=float(np.nanpercentile(chunk_speed, peak_percentile)),
                    median_speed_arcsec_s=float(np.nanmedian(chunk_speed)),
                    n_samples=int(mask.sum()),
                )
            )
    return rows


def write_csv(path: Path, rows: list[ChunkSpeed]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fo:
        writer = csv.DictWriter(
            fo,
            fieldnames=[
                "obsnum",
                "chunk_index",
                "t_start",
                "t_stop",
                "speed_arcsec_s",
                "median_speed_arcsec_s",
                "n_samples",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_histogram(
    path: Path,
    rows: list[ChunkSpeed],
    *,
    bins: int,
    chunk_sec: float,
    peak_percentile: float,
) -> None:
    speeds = np.asarray([row.speed_arcsec_s for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.hist(speeds, bins=bins, color="#277da1", edgecolor="white", linewidth=0.7)
    percentiles = [5, 16, 50, 84, 95, 99]
    vals = np.nanpercentile(speeds, percentiles)
    for pct, val in zip(percentiles, vals):
        style = "-" if pct == 50 else "--"
        alpha = 0.9 if pct in (50, 95) else 0.45
        ax.axvline(val, color="#222222", linestyle=style, linewidth=1.2, alpha=alpha)
        if pct in (5, 50, 95, 99):
            ax.text(
                val,
                ax.get_ylim()[1] * 0.94,
                f"p{pct}={val:.1f}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
            )
    ax.set_title(
        f"Per-chunk delta-source altaz scan speeds ({len(rows)} chunks, {chunk_sec:g} s chunks)"
    )
    ax.set_xlabel(f"Chunk p{peak_percentile:g} speed [arcsec/s]")
    ax.set_ylabel("Chunk count")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(rows: list[ChunkSpeed]) -> None:
    speeds = np.asarray([row.speed_arcsec_s for row in rows], dtype=float)
    med_speeds = np.asarray([row.median_speed_arcsec_s for row in rows], dtype=float)
    print(f"chunks: {len(rows)}")
    print(
        "peak speed arcsec/s: "
        f"min={np.nanmin(speeds):.2f} "
        f"p05={np.nanpercentile(speeds, 5):.2f} "
        f"p16={np.nanpercentile(speeds, 16):.2f} "
        f"median={np.nanmedian(speeds):.2f} "
        f"p84={np.nanpercentile(speeds, 84):.2f} "
        f"p95={np.nanpercentile(speeds, 95):.2f} "
        f"p99={np.nanpercentile(speeds, 99):.2f} "
        f"max={np.nanmax(speeds):.2f}"
    )
    print(
        "median speed arcsec/s: "
        f"min={np.nanmin(med_speeds):.2f} "
        f"median={np.nanmedian(med_speeds):.2f} "
        f"max={np.nanmax(med_speeds):.2f}"
    )
    print("per obs peak-speed medians:")
    for obsnum in sorted({row.obsnum for row in rows}):
        obs_speeds = np.asarray(
            [row.speed_arcsec_s for row in rows if row.obsnum == obsnum], dtype=float
        )
        print(
            f"  {obsnum}: n={obs_speeds.size:3d} "
            f"median={np.nanmedian(obs_speeds):7.2f} "
            f"p05={np.nanpercentile(obs_speeds, 5):7.2f} "
            f"p95={np.nanpercentile(obs_speeds, 95):7.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tel-glob",
        default=str(DEFAULT_ROOT / "tel_*_recomputed.nc"),
        help="Glob selecting telescope NetCDF files.",
    )
    parser.add_argument("--chunk-sec", type=float, default=10.0)
    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=99.5,
        help="Per-chunk robust peak speed percentile.",
    )
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument(
        "--max-dt-s",
        type=float,
        default=0.1,
        help="Reject derivative samples with larger time steps.",
    )
    parser.add_argument(
        "--max-step-rad",
        type=float,
        default=0.01,
        help="Reject derivative samples with larger coordinate steps.",
    )
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ROOT / "scan_speed_histogram_altaz.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_ROOT / "scan_speed_histogram_altaz.csv",
        help="Output per-chunk CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.tel_glob))
    if not paths:
        raise SystemExit(f"no telescope files matched {args.tel_glob!r}")

    rows = collect_chunk_speeds(
        paths,
        chunk_sec=args.chunk_sec,
        peak_percentile=args.peak_percentile,
        min_samples=args.min_samples,
        max_dt_s=args.max_dt_s,
        max_step_rad=args.max_step_rad,
    )
    if not rows:
        raise SystemExit("no valid chunks found")

    write_csv(args.csv, rows)
    plot_histogram(
        args.out,
        rows,
        bins=args.bins,
        chunk_sec=args.chunk_sec,
        peak_percentile=args.peak_percentile,
    )
    print_summary(rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
