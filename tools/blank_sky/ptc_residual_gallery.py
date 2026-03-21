#!/usr/bin/env python3
"""Build RTC-vs-PTC residual comparison plots for selected impulsive scans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("matplotlib is required to generate the PTC residual gallery") from exc


DEFAULT_REDU_ROOT = Path(
    "/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu64"
)
DEFAULT_OUTDIR = DEFAULT_REDU_ROOT / "ptc_residual_gallery"
DEFAULT_CASES = (
    ("151928", 82),
    ("151930", 36),
    ("152524", 19),
    ("152526", 39),
)
DEFAULT_MATCH_TOL_SAMPLES = 8


@dataclass
class ResidualCase:
    obsnum: str
    scan: int


@dataclass
class ResidualExample:
    obsnum: str
    scan: int
    network: int
    center_sample: int
    top_score: float
    overall_flagged_frac: float
    det_uid: int
    det_local_sample: int
    det_local_z: float
    dets_gt5_unflag: int
    rtc_slot: int
    rtc_slot_uid: int
    rtc_slot_kind: str
    rtc_slot_score: float
    rtc_slot_sample: int
    rtc_ptc_sample_delta: int
    rtc_ptc_same_uid: bool
    rtc_offset_sec: np.ndarray
    rtc_snippet_z: np.ndarray
    rtc_snippet_flag: np.ndarray
    ptc_t_rel: np.ndarray
    ptc_z: np.ndarray
    ptc_flag: np.ndarray


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


def _dt_from_dataset(ds: netCDF4.Dataset) -> float:
    for name in ("TelTime", "PpsTime"):
        if name in ds.variables:
            t = np.asarray(ds.variables[name][:], dtype=float).reshape(-1)
            if t.size >= 2:
                dt = float(np.nanmedian(np.diff(t)))
                if np.isfinite(dt) and dt > 0:
                    return dt
    return 1.0


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(x))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sigma


def _detector_z_matrix(signal: np.ndarray, flags: np.ndarray) -> np.ndarray:
    z = np.full_like(signal, np.nan, dtype=float)
    for j in range(signal.shape[1]):
        good = np.isfinite(signal[:, j]) & (flags[:, j] == 0)
        if good.sum() < 32:
            continue
        med = float(np.nanmedian(signal[good, j]))
        sigma = _robust_sigma(signal[good, j] - med)
        if np.isfinite(sigma) and sigma > 0:
            z[:, j] = (signal[:, j] - med) / sigma
    return z


def _parse_cases(case_args: list[str]) -> list[ResidualCase]:
    cases: list[ResidualCase] = []
    for item in case_args:
        obs, scan = item.split(":")
        cases.append(ResidualCase(obsnum=obs, scan=int(scan)))
    return cases


def _choose_example(redu_dir: Path, case: ResidualCase, match_tol_samples: int) -> ResidualExample:
    ptc_path = redu_dir / case.obsnum / "raw" / f"toltec_commissioning_science_{case.obsnum}_ptc_timestream.nc"
    rtc_path = redu_dir / case.obsnum / "raw" / f"toltec_commissioning_science_{case.obsnum}_rtcdiag.nc"
    if not ptc_path.exists():
        raise FileNotFoundError(ptc_path)
    if not rtc_path.exists():
        raise FileNotFoundError(rtc_path)

    with netCDF4.Dataset(ptc_path) as ds_ptc, netCDF4.Dataset(rtc_path) as ds_rtc:
        out = _filled(ds_ptc.variables["output_scan_index"], fill=-1).astype(int)
        idx = int(np.where(out == case.scan)[0][0])
        scan_indices = _filled(ds_ptc.variables["scan_indices"], fill=-1).astype(int)
        start, end = [int(v) for v in scan_indices[idx, :2]]
        signal = _filled(ds_ptc.variables["signal"], fill=np.nan).astype(float)[start : end + 1, :]
        flags = _filled(ds_ptc.variables["flags"], fill=0).astype(int)[start : end + 1, :]
        apt_nw = _filled(ds_ptc.variables["apt_nw"], fill=-1).astype(int)
        apt_uid = _filled(ds_ptc.variables["apt_uid"], fill=-1).astype(int)
        nw_order = sorted(set(int(v) for v in apt_nw.tolist()))
        dt_sec = _dt_from_dataset(ds_ptc)
        zfull = _detector_z_matrix(signal, flags)

        row = case.scan - 1
        cand = _filled(ds_rtc.variables["rtc_network_impulsive_mask_candidate_available"], fill=0).astype(int)[row, :]
        applied = _filled(ds_rtc.variables["rtc_network_impulsive_mask_applied"], fill=0).astype(int)[row, :]
        centers = _filled(ds_rtc.variables["rtc_network_impulsive_mask_cluster_center_sample"], fill=-1).astype(int)[row, :]
        scores = _filled(ds_rtc.variables["rtc_network_impulsive_mask_override_score"], fill=np.nan).astype(float)[row, :]

        slot_score = _filled(ds_rtc.variables["rtc_impulsive_slot_event_score"], fill=np.nan).astype(float)[row, :, :]
        slot_kind = _filled(ds_rtc.variables["rtc_impulsive_slot_event_kind"], fill=-1).astype(int)[row, :, :]
        slot_sample = _filled(ds_rtc.variables["rtc_impulsive_slot_event_sample"], fill=-1).astype(int)[row, :, :]
        slot_det = _filled(ds_rtc.variables["rtc_impulsive_slot_det_index"], fill=-1).astype(int)[row, :, :]
        slot_z = _filled(ds_rtc.variables["rtc_impulsive_slot_snippet_z"], fill=np.nan).astype(float)[row, :, :, :]
        slot_flag = _filled(ds_rtc.variables["rtc_impulsive_slot_snippet_flag"], fill=0).astype(int)[row, :, :, :] != 0
        snippet_offsets = _filled(ds_rtc.variables["rtc_impulsive_snippet_offset_samples"], fill=0).astype(float)

        best: dict[str, object] | None = None
        for i, nw in enumerate(nw_order):
            if not (cand[i] or applied[i]):
                continue
            center = int(centers[i])
            if center < 0:
                continue
            dets = np.where(apt_nw == nw)[0]
            local_lo = max(0, center - 3)
            local_hi = min(zfull.shape[0], center + 4)
            local = zfull[local_lo:local_hi, :][:, dets]
            local_f = flags[local_lo:local_hi, :][:, dets]
            if not np.isfinite(local).any():
                continue
            local_unflag = np.where(local_f == 0, local, np.nan)
            if not np.isfinite(local_unflag).any():
                max_unflag = float("nan")
                dets_gt5 = 0
                det_local_j = 0
                det_local_row = 0
                det_local_val = float("nan")
            else:
                loc = int(np.nanargmax(np.abs(local_unflag)))
                det_local_row, det_local_j = np.unravel_index(loc, local_unflag.shape)
                det_local_val = float(local_unflag[det_local_row, det_local_j])
                max_unflag = abs(det_local_val)
                col_max = np.max(
                    np.where(np.isfinite(local_unflag), np.abs(local_unflag), -np.inf),
                    axis=0,
                )
                dets_gt5 = int(np.sum(col_max > 5))
            max_all = float(np.nanmax(np.abs(local)))
            det_col = int(dets[det_local_j])
            det_local_sample = int(local_lo + det_local_row)
            det_uid = int(apt_uid[det_col])

            row_scores = slot_score[i, :]
            row_samples = slot_sample[i, :].astype(int)
            row_det = slot_det[i, :].astype(int)
            finite = np.isfinite(row_scores) & (row_samples >= 0) & (row_det >= 0)
            if not np.any(finite):
                continue
            sample_delta = np.where(
                finite,
                np.abs(row_samples - det_local_sample),
                np.iinfo(np.int32).max,
            )
            matched = finite & (sample_delta <= match_tol_samples)
            if not np.any(matched):
                continue
            slot_uid = np.full(row_det.shape, -1, dtype=int)
            valid_det = row_det >= 0
            slot_uid[valid_det] = apt_uid[row_det[valid_det]]
            same_uid = slot_uid == det_uid
            same_uid_key = np.where(matched & same_uid, 0, np.where(matched, 1, 2))
            score_key = np.where(matched, -row_scores, np.inf)
            slot_idx = int(np.lexsort((score_key, sample_delta, same_uid_key))[0])
            event_sample = int(slot_sample[i, slot_idx])
            rtc_offset_sec = (snippet_offsets + (event_sample - center)) * dt_sec
            candidate = {
                "network": nw,
                "center": center,
                "score": float(scores[i]),
                "max_all": max_all,
                "max_unflag": max_unflag,
                "dets_gt5": dets_gt5,
                "det_col": det_col,
                "det_local_sample": det_local_sample,
                "det_local_val": det_local_val,
                "slot_idx": slot_idx,
                "slot_uid": int(slot_uid[slot_idx]),
                "slot_kind": "raw_like" if int(slot_kind[i, slot_idx]) == 0 else "delta_like",
                "slot_score": float(slot_score[i, slot_idx]),
                "slot_sample": event_sample,
                "slot_sample_delta": int(sample_delta[slot_idx]),
                "slot_same_uid": bool(same_uid[slot_idx]),
                "rtc_offset_sec": rtc_offset_sec,
                "rtc_snippet_z": slot_z[i, slot_idx, :].astype(float),
                "rtc_snippet_flag": slot_flag[i, slot_idx, :].astype(bool),
            }
            key = (
                -np.nan_to_num(candidate["max_unflag"], nan=-1.0),
                -int(candidate["dets_gt5"]),
                -float(candidate["score"]),
                -float(candidate["max_all"]),
            )
            if best is None or key < best["key"]:
                candidate["key"] = key
                best = candidate

        if best is None:
            raise RuntimeError(f"no impulsive residual candidate found for obs {case.obsnum} scan {case.scan}")

        center = int(best["center"])
        det_col = int(best["det_col"])
        n_pre = int(round(0.25 / dt_sec))
        n_post = int(round(0.75 / dt_sec))
        lo = max(0, center - n_pre)
        hi = min(signal.shape[0], center + n_post + 1)
        t_rel = (np.arange(lo, hi, dtype=float) - center) * dt_sec
        return ResidualExample(
            obsnum=case.obsnum,
            scan=case.scan,
            network=int(best["network"]),
            center_sample=center,
            top_score=float(best["score"]),
            overall_flagged_frac=float(np.mean(flags != 0)),
            det_uid=int(apt_uid[det_col]),
            det_local_sample=int(best["det_local_sample"]),
            det_local_z=float(best["det_local_val"]),
            dets_gt5_unflag=int(best["dets_gt5"]),
            rtc_slot=int(best["slot_idx"]),
            rtc_slot_uid=int(best["slot_uid"]),
            rtc_slot_kind=str(best["slot_kind"]),
            rtc_slot_score=float(best["slot_score"]),
            rtc_slot_sample=int(best["slot_sample"]),
            rtc_ptc_sample_delta=int(best["slot_sample_delta"]),
            rtc_ptc_same_uid=bool(best["slot_same_uid"]),
            rtc_offset_sec=np.asarray(best["rtc_offset_sec"], dtype=float),
            rtc_snippet_z=np.asarray(best["rtc_snippet_z"], dtype=float),
            rtc_snippet_flag=np.asarray(best["rtc_snippet_flag"], dtype=bool),
            ptc_t_rel=t_rel,
            ptc_z=zfull[lo:hi, det_col].astype(float),
            ptc_flag=(flags[lo:hi, det_col] != 0),
        )


def _plot_example(example: ResidualExample, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=False, squeeze=False)
    ax_rtc, ax_ptc = axes.ravel()

    ax_rtc.plot(example.rtc_offset_sec, example.rtc_snippet_z, color="tab:blue", lw=1.6)
    if np.any(example.rtc_snippet_flag):
        ax_rtc.scatter(
            example.rtc_offset_sec[example.rtc_snippet_flag],
            example.rtc_snippet_z[example.rtc_snippet_flag],
            color="tab:red",
            s=18,
            zorder=3,
        )
    ax_rtc.axvline(0.0, color="0.35", lw=1.0, ls="--")
    ax_rtc.axhline(0.0, color="0.8", lw=0.8)
    ax_rtc.grid(alpha=0.18)
    ax_rtc.set_ylabel("RTC snippet z")
    ax_rtc.set_title(
        f"RTC saved slot: nw{example.network} slot {example.rtc_slot} "
        f"{example.rtc_slot_kind} uid {example.rtc_slot_uid} "
        f"score {example.rtc_slot_score:.1f} sample {example.rtc_slot_sample} "
        f"(Δ={example.rtc_ptc_sample_delta} samp"
        f"{' same uid' if example.rtc_ptc_same_uid else ''})",
        fontsize=10,
    )

    ax_ptc.plot(example.ptc_t_rel, example.ptc_z, color="0.65", lw=1.2, label="PTC detector")
    unflag = ~example.ptc_flag
    if np.any(unflag):
        ax_ptc.plot(example.ptc_t_rel[unflag], example.ptc_z[unflag], color="tab:blue", lw=1.6, label="unflagged")
    if np.any(example.ptc_flag):
        ax_ptc.scatter(
            example.ptc_t_rel[example.ptc_flag],
            example.ptc_z[example.ptc_flag],
            color="tab:red",
            s=18,
            zorder=3,
            label="flagged",
        )
    ax_ptc.axvline(0.0, color="0.35", lw=1.0, ls="--")
    ax_ptc.axhline(0.0, color="0.8", lw=0.8)
    ax_ptc.grid(alpha=0.18)
    ax_ptc.set_xlabel("time from RTC coincidence center [s]")
    ax_ptc.set_ylabel("PTC detector z")
    ax_ptc.set_title(
        f"PTC residual detector: uid {example.det_uid} local sample {example.det_local_sample} "
        f"local z {example.det_local_z:.2f}",
        fontsize=10,
    )
    ax_ptc.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"RTC vs PTC residual comparison\n"
        f"obs {example.obsnum} scan {example.scan} nw {example.network} "
        f"top_score={example.top_score:.1f} dets_gt5_unflag={example.dets_gt5_unflag} "
        f"overall_flagged={example.overall_flagged_frac:.3f}",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Top: RTC impulsive slot time-matched to the chosen PTC residual sample. "
        "Bottom: the strongest surviving unflagged PTC detector residual in that same network.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _write_summary(outpath: Path, examples: list[ResidualExample]) -> None:
    lines = [
        "# PTC Residual Gallery",
        "",
        "RTC-vs-PTC comparison plots for the first blank-sky post-PCA residual audit.",
        "",
    ]
    for ex in examples:
        lines.extend(
            [
                f"## obs {ex.obsnum} scan {ex.scan}",
                "",
                f"- network: `{ex.network}`",
                f"- RTC slot: `{ex.rtc_slot}` `{ex.rtc_slot_kind}` uid `{ex.rtc_slot_uid}` score `{ex.rtc_slot_score:.2f}` sample `{ex.rtc_slot_sample}`",
                f"- RTC/PTC sample delta: `{ex.rtc_ptc_sample_delta}`{' (same uid)' if ex.rtc_ptc_same_uid else ''}",
                f"- PTC residual detector: uid `{ex.det_uid}` local sample `{ex.det_local_sample}` local z `{ex.det_local_z:.2f}`",
                f"- detectors with unflagged local `|z|>5` in this network cluster: `{ex.dets_gt5_unflag}`",
                f"- overall PTC flagged fraction in selected scan block: `{ex.overall_flagged_frac:.4f}`",
                "",
            ]
        )
    outpath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", type=Path, default=DEFAULT_REDU_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as obsnum:scan. Can be repeated. Defaults to the four strongest residual scans.",
    )
    parser.add_argument(
        "--match-tol-samples",
        type=int,
        default=DEFAULT_MATCH_TOL_SAMPLES,
        help="Maximum RTC/PTC sample separation allowed when pairing a saved RTC slot to the chosen PTC residual.",
    )
    args = parser.parse_args()

    cases = _parse_cases(args.case) if args.case else [ResidualCase(*item) for item in DEFAULT_CASES]
    args.outdir.mkdir(parents=True, exist_ok=True)

    examples: list[ResidualExample] = []
    for case in cases:
        ex = _choose_example(args.redu_dir, case, match_tol_samples=args.match_tol_samples)
        examples.append(ex)
        outpath = args.outdir / f"ptc_residual_obs{ex.obsnum}_scan{ex.scan}.png"
        _plot_example(ex, outpath)

    _write_summary(args.outdir / "PTC_RESIDUAL_GALLERY.md", examples)


if __name__ == "__main__":
    main()
