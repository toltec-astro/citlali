#!/usr/bin/env python3
"""Build a small PNG gallery of representative RTC glitch classes.

This is meant as a quick engineering artifact before deleting large historical
RTC timestream products. It uses:

- recent `rtcdiag` files for compact impulsive snippets and cross-network burst
  examples
- older `*_rtc_timestream.nc` files for a true network step/coherent segment

Outputs:
- `glitch_raw_like.png`
- `glitch_delta_like.png`
- `glitch_cross_network_impulsive.png`
- `glitch_step_like.png`
- `GLITCH_GALLERY.md`
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("matplotlib is required to generate the glitch gallery") from exc


DEFAULT_REDUCED_ROOT = Path(
    "/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced"
)


def _filled(var: netCDF4.Variable, fill: float | int | None = None) -> np.ndarray:
    data = var[:]
    if np.ma.isMaskedArray(data):
        if fill is None:
            dtype = np.asarray(data).dtype
            if np.issubdtype(dtype, np.floating):
                fill = float("nan")
            else:
                fill = np.iinfo(np.int32).min
        data = np.ma.filled(data, fill_value=fill)
    return np.asarray(data)


def _redu_num(path: Path) -> int:
    for part in path.parts:
        m = re.fullmatch(r"redu(\d+)", part)
        if m:
            return int(m.group(1))
    return -1


def _obsnum(path: Path) -> int:
    m = re.search(r"_science_(\d+)_", path.name)
    if not m:
        raise ValueError(f"could not parse obsnum from {path}")
    return int(m.group(1))


def _dt_from_dataset(ds: netCDF4.Dataset) -> float:
    if "TelTime" in ds.variables:
        t = np.asarray(ds.variables["TelTime"][:], dtype=float).reshape(-1)
        if t.size >= 2:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                return dt
    for name in ("RTC_SAMPRATE", "SAMPRATE"):
        if name in ds.variables:
            fs = float(np.asarray(ds.variables[name][:], dtype=float).reshape(-1)[0])
            if np.isfinite(fs) and fs > 0:
                return 1.0 / fs
    return 1.0


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sigma


@dataclass
class SnippetExample:
    nc_file: Path
    obsnum: int
    redu: int
    scan_idx: int
    output_scan: int
    network: int
    slot: int
    kind_label: str
    event_score: float
    event_sample: int
    apt_uid: int
    peak_abs_z: float
    peak_delta_abs_z: float
    snippet_z: np.ndarray
    snippet_flag: np.ndarray
    offset_sec: np.ndarray


@dataclass
class CrossNetworkExample:
    nc_file: Path
    obsnum: int
    redu: int
    scan_idx: int
    output_scan: int
    center_sample: int
    cluster_network_count: int
    override_score: float
    network_rows: list[dict[str, object]]
    dt_sec: float
    sample_tol: int


@dataclass
class StepExample:
    nc_file: Path
    obsnum: int
    redu: int
    scan_idx: int
    network: int
    output_scan: int
    dominant_sample: int
    score_max: float
    align_frac: float
    window_start: int
    window_end: int
    dt_sec: float
    traces: list[dict[str, object]]


def _preferred_rtcdiag_files(root: Path) -> list[Path]:
    preferred = [
        root / "redu59/151928/raw/toltec_commissioning_science_151928_rtcdiag.nc",
        root / "redu59/152524/raw/toltec_commissioning_science_152524_rtcdiag.nc",
        root / "redu59/152526/raw/toltec_commissioning_science_152526_rtcdiag.nc",
    ]
    keep = [p for p in preferred if p.exists()]
    if keep:
        return keep
    return sorted(root.glob("redu*/**/*_rtcdiag.nc"))


def _rtc_timestream_files(root: Path) -> list[Path]:
    files = sorted(root.glob("redu*/**/*_rtc_timestream.nc"))
    if not files:
        raise FileNotFoundError(f"no *_rtc_timestream.nc files under {root}")
    return files


def _best_snippet(files: list[Path], kind_value: int) -> SnippetExample:
    best: SnippetExample | None = None
    for nc_file in files:
        with netCDF4.Dataset(nc_file) as ds:
            required = [
                "rtc_impulsive_slot_event_kind",
                "rtc_impulsive_slot_event_score",
                "rtc_impulsive_slot_event_sample",
                "rtc_impulsive_slot_det_index",
                "rtc_impulsive_slot_peak_abs_z",
                "rtc_impulsive_slot_peak_delta_abs_z",
                "rtc_impulsive_slot_snippet_z",
                "rtc_impulsive_slot_snippet_flag",
                "rtc_impulsive_snippet_offset_samples",
                "rtc_diag_network_ids",
                "output_scan_index",
                "apt_uid",
            ]
            if any(name not in ds.variables for name in required):
                continue
            kind = _filled(ds.variables["rtc_impulsive_slot_event_kind"], fill=-1).astype(int)
            score = _filled(ds.variables["rtc_impulsive_slot_event_score"], fill=np.nan).astype(float)
            mask = np.isfinite(score) & (kind == kind_value)
            if not np.any(mask):
                continue
            idx = np.unravel_index(np.nanargmax(np.where(mask, score, np.nan)), score.shape)
            s, nwi, slot = [int(v) for v in idx]
            det_index = int(_filled(ds.variables["rtc_impulsive_slot_det_index"], fill=-1)[idx])
            if det_index < 0:
                continue
            dt_sec = _dt_from_dataset(ds)
            offsets = _filled(ds.variables["rtc_impulsive_snippet_offset_samples"], fill=0).astype(float) * dt_sec
            example = SnippetExample(
                nc_file=nc_file,
                obsnum=_obsnum(nc_file),
                redu=_redu_num(nc_file),
                scan_idx=s,
                output_scan=int(_filled(ds.variables["output_scan_index"], fill=-1)[s]),
                network=int(_filled(ds.variables["rtc_diag_network_ids"], fill=-1)[nwi]),
                slot=slot,
                kind_label="raw_like" if kind_value == 0 else "delta_like",
                event_score=float(score[idx]),
                event_sample=int(_filled(ds.variables["rtc_impulsive_slot_event_sample"], fill=-1)[idx]),
                apt_uid=int(_filled(ds.variables["apt_uid"], fill=-1)[det_index]),
                peak_abs_z=float(_filled(ds.variables["rtc_impulsive_slot_peak_abs_z"], fill=np.nan)[idx]),
                peak_delta_abs_z=float(_filled(ds.variables["rtc_impulsive_slot_peak_delta_abs_z"], fill=np.nan)[idx]),
                snippet_z=_filled(ds.variables["rtc_impulsive_slot_snippet_z"], fill=np.nan)[idx].astype(float),
                snippet_flag=_filled(ds.variables["rtc_impulsive_slot_snippet_flag"], fill=0)[idx].astype(int) != 0,
                offset_sec=offsets,
            )
            if best is None or example.event_score > best.event_score:
                best = example
    if best is None:
        raise RuntimeError(f"no kind={kind_value} impulsive snippet found")
    return best


def _best_cross_network(files: list[Path], sample_tol: int = 5) -> CrossNetworkExample:
    best: CrossNetworkExample | None = None
    best_key: tuple[int, int, float, float] | None = None
    for nc_file in files:
        with netCDF4.Dataset(nc_file) as ds:
            required = [
                "rtc_network_impulsive_mask_applied",
                "rtc_network_impulsive_mask_cross_network_trigger",
                "rtc_network_impulsive_mask_high_score_override_trigger",
                "rtc_network_impulsive_mask_cluster_center_sample",
                "rtc_network_impulsive_mask_cluster_network_count",
                "rtc_network_impulsive_mask_override_score",
                "rtc_impulsive_slot_event_score",
                "rtc_impulsive_slot_event_kind",
                "rtc_impulsive_slot_event_sample",
                "rtc_impulsive_slot_det_index",
                "rtc_impulsive_slot_snippet_z",
                "rtc_impulsive_slot_snippet_flag",
                "rtc_impulsive_snippet_offset_samples",
                "rtc_diag_network_ids",
                "output_scan_index",
                "apt_uid",
            ]
            if any(name not in ds.variables for name in required):
                continue

            applied = _filled(ds.variables["rtc_network_impulsive_mask_applied"], fill=0).astype(int)
            cross = _filled(ds.variables["rtc_network_impulsive_mask_cross_network_trigger"], fill=0).astype(int)
            override = _filled(ds.variables["rtc_network_impulsive_mask_high_score_override_trigger"], fill=0).astype(int)
            center = _filled(ds.variables["rtc_network_impulsive_mask_cluster_center_sample"], fill=-1).astype(int)
            cluster_n = _filled(ds.variables["rtc_network_impulsive_mask_cluster_network_count"], fill=0).astype(int)
            override_score = _filled(ds.variables["rtc_network_impulsive_mask_override_score"], fill=np.nan).astype(float)

            candidate_rows = np.argwhere((applied > 0) & ((cross > 0) | (override > 0)) & np.isfinite(override_score))
            if candidate_rows.size == 0:
                continue
            nws = _filled(ds.variables["rtc_diag_network_ids"], fill=-1).astype(int)
            scans = _filled(ds.variables["output_scan_index"], fill=-1).astype(int)
            slot_score = _filled(ds.variables["rtc_impulsive_slot_event_score"], fill=np.nan).astype(float)
            slot_kind = _filled(ds.variables["rtc_impulsive_slot_event_kind"], fill=-1).astype(int)
            slot_sample = _filled(ds.variables["rtc_impulsive_slot_event_sample"], fill=-1).astype(int)
            slot_det = _filled(ds.variables["rtc_impulsive_slot_det_index"], fill=-1).astype(int)
            slot_z = _filled(ds.variables["rtc_impulsive_slot_snippet_z"], fill=np.nan).astype(float)
            slot_flag = _filled(ds.variables["rtc_impulsive_slot_snippet_flag"], fill=0).astype(int) != 0
            offsets = _filled(ds.variables["rtc_impulsive_snippet_offset_samples"], fill=0).astype(float)
            dt_sec = _dt_from_dataset(ds)
            apt_uid = _filled(ds.variables["apt_uid"], fill=-1).astype(int)
            for s, nwi in candidate_rows:
                s = int(s)
                nwi = int(nwi)
                row_center = int(center[s, nwi])
                network_rows: list[dict[str, object]] = []
                for other_nwi, nw in enumerate(nws):
                    if int(applied[s, other_nwi]) <= 0:
                        continue
                    if int(center[s, other_nwi]) != row_center:
                        continue
                    row = slot_score[s, other_nwi, :]
                    row_samples = slot_sample[s, other_nwi, :].astype(int)
                    finite = np.isfinite(row) & (row_samples >= 0)
                    if not np.any(finite):
                        continue
                    proximity = np.where(finite, np.abs(row_samples - row_center), np.iinfo(np.int32).max)
                    score_key = np.where(finite, -row, np.inf)
                    slot = int(np.lexsort((score_key, proximity))[0])
                    event_sample = int(slot_sample[s, other_nwi, slot])
                    absdiff = abs(event_sample - row_center)
                    if absdiff > sample_tol:
                        continue
                    det_index = int(slot_det[s, other_nwi, slot])
                    network_rows.append(
                        {
                            "network": int(nw),
                            "slot": slot,
                            "event_score": float(row[slot]),
                            "event_kind": int(slot_kind[s, other_nwi, slot]),
                            "event_sample": event_sample,
                            "event_sample_absdiff": absdiff,
                            "apt_uid": int(apt_uid[det_index]) if det_index >= 0 else -1,
                            "snippet_z": slot_z[s, other_nwi, slot, :].astype(float),
                            "snippet_flag": slot_flag[s, other_nwi, slot, :].astype(bool),
                            "offset_sec": (offsets + (event_sample - row_center)) * dt_sec,
                        }
                    )
                network_rows.sort(key=lambda row: int(row["network"]))
                if len(network_rows) < 2:
                    continue
                n_delta = sum(int(row["event_kind"]) == 1 for row in network_rows)
                score_sum = float(sum(float(row["event_score"]) for row in network_rows))
                override_flag = int(override[s, nwi])
                key = (n_delta, len(network_rows), score_sum, float(override_score[s, nwi]) + 0.1 * override_flag)
                if best is None or best_key is None or key > best_key:
                    best_key = key
                    best = CrossNetworkExample(
                        nc_file=nc_file,
                        obsnum=_obsnum(nc_file),
                        redu=_redu_num(nc_file),
                        scan_idx=s,
                        output_scan=int(scans[s]),
                        center_sample=row_center,
                        cluster_network_count=int(cluster_n[s, nwi]),
                        override_score=float(override_score[s, nwi]),
                        network_rows=network_rows,
                        dt_sec=dt_sec,
                        sample_tol=sample_tol,
                    )
    if best is None:
        raise RuntimeError("no cross-network impulsive coincidence example found")
    return best


def _best_step(files: list[Path]) -> StepExample:
    best: StepExample | None = None
    for nc_file in files:
        with netCDF4.Dataset(nc_file) as ds:
            required = [
                "signal",
                "flags",
                "scan_indices",
                "TelTime",
                "apt_nw",
                "apt_uid",
                "rtc_network_step_mask_applied",
                "rtc_network_step_score_max",
                "rtc_network_step_alignment_frac",
                "rtc_network_step_dominant_sample",
                "rtc_network_step_mask_start_sample",
                "rtc_network_step_mask_end_sample",
                "rtc_diag_network_ids",
                "rtc_step_score",
                "rtc_step_sample",
            ]
            if any(name not in ds.variables for name in required):
                continue
            applied = _filled(ds.variables["rtc_network_step_mask_applied"], fill=0).astype(int)
            score = _filled(ds.variables["rtc_network_step_score_max"], fill=np.nan).astype(float)
            align = _filled(ds.variables["rtc_network_step_alignment_frac"], fill=np.nan).astype(float)
            candidate_mask = (applied > 0) & np.isfinite(score)
            if not np.any(candidate_mask):
                continue
            idx = np.unravel_index(np.nanargmax(np.where(candidate_mask, score, np.nan)), score.shape)
            scan_idx, nwi = [int(v) for v in idx]
            network = int(_filled(ds.variables["rtc_diag_network_ids"], fill=-1)[nwi])
            scan_indices = _filled(ds.variables["scan_indices"], fill=-1).astype(int)
            i0, i1 = [int(v) for v in scan_indices[scan_idx, :2]]
            dominant = int(_filled(ds.variables["rtc_network_step_dominant_sample"], fill=-1)[idx])
            start = int(_filled(ds.variables["rtc_network_step_mask_start_sample"], fill=-1)[idx])
            end = int(_filled(ds.variables["rtc_network_step_mask_end_sample"], fill=-1)[idx])
            apt_nw = _filled(ds.variables["apt_nw"], fill=-1).astype(int)
            apt_uid = _filled(ds.variables["apt_uid"], fill=-1).astype(int)
            det_keep = np.where(apt_nw == network)[0]
            det_scores = _filled(ds.variables["rtc_step_score"], fill=np.nan).astype(float)[scan_idx, det_keep]
            det_samples = _filled(ds.variables["rtc_step_sample"], fill=-1).astype(int)[scan_idx, det_keep]
            good = np.isfinite(det_scores) & (det_scores > 0)
            if np.count_nonzero(good) < 3:
                continue
            top = np.argsort(np.where(good, det_scores, -np.inf))[::-1][:6]
            signal = _filled(ds.variables["signal"], fill=np.nan).astype(float)
            flags = _filled(ds.variables["flags"], fill=0).astype(int)
            tel_time = _filled(ds.variables["TelTime"], fill=np.nan).astype(float)
            center_global = i0 + dominant
            pad = max(120, (end - start) * 4 if end > start else 120)
            g0 = max(i0, center_global - pad)
            g1 = min(i1, center_global + pad)
            if g1 <= g0:
                continue
            traces: list[dict[str, object]] = []
            t_rel = tel_time[g0 : g1 + 1] - tel_time[center_global]
            for rel_rank, j in enumerate(top):
                det = int(det_keep[j])
                y = signal[g0 : g1 + 1, det].astype(float)
                sigma = _robust_sigma(y[np.isfinite(y)])
                med = float(np.nanmedian(y))
                z = (y - med) / sigma
                traces.append(
                    {
                        "det_index": det,
                        "apt_uid": int(apt_uid[det]),
                        "step_score": float(det_scores[j]),
                        "step_sample": int(det_samples[j]),
                        "t_rel": t_rel,
                        "z": z,
                        "flag": flags[g0 : g1 + 1, det].astype(int) != 0,
                    }
                )
            example = StepExample(
                nc_file=nc_file,
                obsnum=_obsnum(nc_file),
                redu=_redu_num(nc_file),
                scan_idx=scan_idx,
                network=network,
                output_scan=scan_idx + 1,
                dominant_sample=dominant,
                score_max=float(score[idx]),
                align_frac=float(align[idx]),
                window_start=start,
                window_end=end,
                dt_sec=_dt_from_dataset(ds),
                traces=traces,
            )
            key = (example.score_max, example.align_frac)
            best_key = (best.score_max, best.align_frac) if best is not None else None
            if best is None or key > best_key:
                best = example
    if best is None:
        raise RuntimeError("no masked network step example found")
    return best


def _plot_snippet(example: SnippetExample, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(example.offset_sec, example.snippet_z, color="tab:blue", lw=1.8)
    if np.any(example.snippet_flag):
        ax.scatter(
            example.offset_sec[example.snippet_flag],
            example.snippet_z[example.snippet_flag],
            color="tab:red",
            s=24,
            zorder=3,
            label="flagged samples",
        )
    ax.axvline(0.0, color="0.35", lw=1.0, ls="--", label="event sample")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.grid(alpha=0.18)
    ax.set_xlabel("time from event [s]")
    ax.set_ylabel("standardized RTC signal [z]")
    ax.set_title(
        f"{example.kind_label.replace('_', '-')} impulsive example\n"
        f"obs {example.obsnum} redu{example.redu} scan {example.output_scan} nw {example.network} "
        f"slot {example.slot} uid {example.apt_uid} score {example.event_score:.1f}"
    )
    subtitle = (
        f"peak |z|={example.peak_abs_z:.1f}, peak |Δz|={example.peak_delta_abs_z:.1f}. "
        f"Red points are samples Citlali flagged in the captured snippet."
    )
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_cross_network(example: CrossNetworkExample, outpath: Path) -> None:
    n = len(example.network_rows)
    fig, axes = plt.subplots(n, 1, figsize=(9.2, 2.2 * n), sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, row in zip(axes, example.network_rows):
        x = np.asarray(row["offset_sec"], dtype=float)
        y = np.asarray(row["snippet_z"], dtype=float)
        flag = np.asarray(row["snippet_flag"], dtype=bool)
        ax.plot(x, y, color="tab:blue", lw=1.4)
        if np.any(flag):
            ax.scatter(x[flag], y[flag], color="tab:red", s=12, zorder=3)
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--")
        ax.axhline(0.0, color="0.8", lw=0.7)
        ax.grid(alpha=0.15)
        kind_label = "raw_like" if int(row["event_kind"]) == 0 else "delta_like"
        ax.set_ylabel(f"nw{row['network']}\nz", rotation=0, labelpad=16, va="center")
        ax.set_title(
            f"uid {row['apt_uid']} slot {row['slot']} {kind_label} "
            f"score {row['event_score']:.1f} sample {row['event_sample']} "
            f"(Δ={row['event_sample_absdiff']} samp)",
            fontsize=9,
        )
    axes[-1].set_xlabel("time from shared coincidence center [s]")
    fig.suptitle(
        f"Cross-network impulsive coincidence example\n"
        f"obs {example.obsnum} redu{example.redu} scan {example.output_scan} "
        f"cluster_networks={example.cluster_network_count} override_score={example.override_score:.1f}",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        f"Each panel is the nearest stored slot snippet to the recovered coincidence center. "
        f"Only networks with a stored slot within {example.sample_tol} samples of the center are shown.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_step(example: StepExample, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    gap = 8.0
    start_t = example.window_start * example.dt_sec - example.dominant_sample * example.dt_sec
    end_t = example.window_end * example.dt_sec - example.dominant_sample * example.dt_sec
    ax.axvspan(start_t, end_t, color="tab:red", alpha=0.12, label="network step mask window")
    for i, row in enumerate(example.traces):
        y = np.asarray(row["z"], dtype=float) + i * gap
        x = np.asarray(row["t_rel"], dtype=float)
        ax.plot(x, y, lw=1.15, label=f"uid {row['apt_uid']} score {row['step_score']:.1f}")
        flag = np.asarray(row["flag"], dtype=bool)
        if np.any(flag):
            ax.scatter(x[flag], y[flag], color="tab:red", s=8, zorder=3)
    ax.axvline(0.0, color="0.25", lw=1.0, ls="--", label="dominant step sample")
    ax.set_xlabel("time from dominant step sample [s]")
    ax.set_ylabel("standardized RTC signal + offset")
    ax.set_title(
        f"Network step / coherent example\n"
        f"obs {example.obsnum} redu{example.redu} scan {example.output_scan} nw {example.network} "
        f"score {example.score_max:.1f} alignment {example.align_frac:.2f}"
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.text(
        0.5,
        0.01,
        "Top stepped detectors in the selected network, standardized within the displayed window. "
        "The red shaded region is the network-level step mask window.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_summary(
    outpath: Path,
    raw_like: SnippetExample,
    delta_like: SnippetExample,
    cross: CrossNetworkExample,
    step: StepExample,
) -> None:
    lines = [
        "# Blank-Sky Glitch Gallery",
        "",
        "This gallery is intended as a quick visual reference for the main glitch/contamination classes we are currently hunting in Citlali blank-sky reductions.",
        "",
        "## Files",
        "",
        f"- `glitch_raw_like.png`: strongest compact raw-like impulsive snippet found in recent `rtcdiag` products.",
        f"- `glitch_delta_like.png`: strongest compact delta-like impulsive snippet found in recent `rtcdiag` products.",
        f"- `glitch_cross_network_impulsive.png`: recovered multi-network impulsive coincidence example.",
        f"- `glitch_step_like.png`: masked network step/coherent segment from an older full RTC timestream product.",
        "",
        "## Selected Examples",
        "",
        f"- Raw-like: obs `{raw_like.obsnum}` `redu{raw_like.redu}` scan `{raw_like.output_scan}` nw `{raw_like.network}` uid `{raw_like.apt_uid}` score `{raw_like.event_score:.2f}` from `{raw_like.nc_file}`",
        f"- Delta-like: obs `{delta_like.obsnum}` `redu{delta_like.redu}` scan `{delta_like.output_scan}` nw `{delta_like.network}` uid `{delta_like.apt_uid}` score `{delta_like.event_score:.2f}` from `{delta_like.nc_file}`",
        f"- Cross-network coincidence: obs `{cross.obsnum}` `redu{cross.redu}` scan `{cross.output_scan}` cluster networks `{cross.cluster_network_count}` override score `{cross.override_score:.2f}` from `{cross.nc_file}`",
        f"- Step-like: obs `{step.obsnum}` `redu{step.redu}` scan `{step.output_scan}` nw `{step.network}` step score `{step.score_max:.2f}` alignment `{step.align_frac:.2f}` from `{step.nc_file}`",
        "",
    ]
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reduced-root", default=str(DEFAULT_REDUCED_ROOT), help="Root reduced directory")
    ap.add_argument("--outdir", default=None, help="Output directory for PNG gallery")
    args = ap.parse_args()

    reduced_root = Path(args.reduced_root).expanduser().resolve()
    if not reduced_root.exists():
        raise FileNotFoundError(reduced_root)
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else reduced_root / "glitch_gallery_blank_sky_2026-03-20"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rtcdiag_files = _preferred_rtcdiag_files(reduced_root)
    rtc_files = _rtc_timestream_files(reduced_root)

    raw_like = _best_snippet(rtcdiag_files, kind_value=0)
    delta_like = _best_snippet(rtcdiag_files, kind_value=1)
    cross = _best_cross_network(rtcdiag_files)
    step = _best_step(rtc_files)

    _plot_snippet(raw_like, outdir / "glitch_raw_like.png")
    _plot_snippet(delta_like, outdir / "glitch_delta_like.png")
    _plot_cross_network(cross, outdir / "glitch_cross_network_impulsive.png")
    _plot_step(step, outdir / "glitch_step_like.png")
    _write_summary(outdir / "GLITCH_GALLERY.md", raw_like, delta_like, cross, step)

    print(f"Wrote {outdir / 'glitch_raw_like.png'}")
    print(f"Wrote {outdir / 'glitch_delta_like.png'}")
    print(f"Wrote {outdir / 'glitch_cross_network_impulsive.png'}")
    print(f"Wrote {outdir / 'glitch_step_like.png'}")
    print(f"Wrote {outdir / 'GLITCH_GALLERY.md'}")


if __name__ == "__main__":
    main()
