#!/usr/bin/env python3
"""Build a human-readable RTC/PTC despiking report for one reduction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("matplotlib is required to generate despike diagnostics") from exc

try:
    from .blank_sky_null_audit import _write_csv
    from .rtcdiag_data import load_reduction_tables
except ImportError:
    from blank_sky_null_audit import _write_csv
    from rtcdiag_data import load_reduction_tables


@dataclass
class CaseExample:
    label: str
    obsnum: str
    output_scan_index: int
    network: int
    candidate_clusters: int
    accepted_clusters: int
    busy_vetoed: int
    newly_flagged_fraction: float
    rtc_imp_mask: int
    rtc_cross: int
    rtc_override: int
    rtc_row_severity: float
    center_sample: int
    det_uid: int
    det_local_z: float
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
    ptc_added_flag: np.ndarray


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


def _dt_from_dataset(ds: netCDF4.Dataset) -> float:
    for name in ("TelTime", "PpsTime"):
        if name in ds.variables:
            t = np.asarray(ds.variables[name][:], dtype=float).reshape(-1)
            if t.size >= 2:
                dt = float(np.nanmedian(np.diff(t)))
                if np.isfinite(dt) and dt > 0:
                    return dt
    return 1.0


def _parse_obsnum(path: Path) -> str:
    return path.name.split("_")[3]


def _collect_ptc_rows(redu_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ptc_path in sorted(redu_dir.glob("*/raw/*_ptc_timestream.nc")):
        obsnum = _parse_obsnum(ptc_path)
        with netCDF4.Dataset(ptc_path) as ds:
            required = [
                "output_scan_index",
                "ptc_second_pass_network_ids",
                "ptc_second_pass_n_candidate_clusters",
                "ptc_second_pass_n_accepted_clusters",
                "ptc_second_pass_busy_network_vetoed",
                "ptc_second_pass_newly_flagged_fraction",
                "ptc_second_pass_added_flag",
            ]
            if any(name not in ds.variables for name in required):
                continue
            scans = _filled(ds.variables["output_scan_index"], fill=-1).astype(int)
            networks = _filled(ds.variables["ptc_second_pass_network_ids"], fill=-1).astype(int)
            cand = _filled(ds.variables["ptc_second_pass_n_candidate_clusters"], fill=0).astype(int)
            acc = _filled(ds.variables["ptc_second_pass_n_accepted_clusters"], fill=0).astype(int)
            veto = _filled(ds.variables["ptc_second_pass_busy_network_vetoed"], fill=0).astype(int)
            frac = _filled(ds.variables["ptc_second_pass_newly_flagged_fraction"], fill=np.nan).astype(float)
            added_flag = _filled(ds.variables["ptc_second_pass_added_flag"], fill=0).astype(int)
            added_fraction_total = float(np.mean(added_flag != 0))
            for i, scan in enumerate(scans):
                for j, nw in enumerate(networks):
                    rows.append(
                        {
                            "obsnum": obsnum,
                            "output_scan_index": int(scan),
                            "network": int(nw),
                            "candidate_clusters": int(cand[i, j]),
                            "accepted_clusters": int(acc[i, j]),
                            "busy_vetoed": int(veto[i, j]),
                            "newly_flagged_fraction": float(frac[i, j]),
                            "file_added_fraction_total": added_fraction_total,
                            "ptc_path": str(ptc_path),
                        }
                    )
    return rows


def _aggregate_ptc_obs(ptc_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_obs: dict[str, list[dict[str, object]]] = {}
    for row in ptc_rows:
        by_obs.setdefault(str(row["obsnum"]), []).append(row)
    out: list[dict[str, object]] = []
    for obsnum, rows in sorted(by_obs.items()):
        out.append(
            {
                "obsnum": obsnum,
                "n_rows": len(rows),
                "candidate_clusters_total": int(sum(int(r["candidate_clusters"]) for r in rows)),
                "accepted_clusters_total": int(sum(int(r["accepted_clusters"]) for r in rows)),
                "busy_vetoed_rows": int(sum(int(r["busy_vetoed"]) for r in rows)),
                "newly_flagged_fraction_sum": float(sum(float(r["newly_flagged_fraction"]) for r in rows)),
                "max_newly_flagged_fraction": float(max(float(r["newly_flagged_fraction"]) for r in rows)),
                "file_added_fraction_total": float(rows[0]["file_added_fraction_total"]),
            }
        )
    return out


def _choose_cases(
    ptc_rows: list[dict[str, object]],
    rtc_lookup: dict[tuple[str, int, int], dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    enriched: list[dict[str, object]] = []
    for row in ptc_rows:
        rtc = rtc_lookup.get((str(row["obsnum"]), int(row["output_scan_index"]), int(row["network"])))
        merged = dict(row)
        if rtc is not None:
            merged["rtc_imp_mask"] = int(rtc.get("impulsive_mask_applied", 0))
            merged["rtc_cross"] = int(rtc.get("impulsive_mask_cross_network_trigger", 0))
            merged["rtc_override"] = int(rtc.get("impulsive_mask_high_score_override_trigger", 0))
            merged["rtc_row_severity"] = float(rtc.get("row_severity", float("nan")))
            merged["rtc_cluster_center_sample"] = int(rtc.get("impulsive_mask_cluster_center_sample", -1))
        else:
            merged["rtc_imp_mask"] = 0
            merged["rtc_cross"] = 0
            merged["rtc_override"] = 0
            merged["rtc_row_severity"] = float("nan")
            merged["rtc_cluster_center_sample"] = -1
        enriched.append(merged)

    case_pools: dict[str, list[dict[str, object]]] = {}
    accepted_survivors = [
        row for row in enriched if int(row["accepted_clusters"]) > 0 and int(row["rtc_imp_mask"]) == 0
    ]
    if accepted_survivors:
        pool = sorted(
            accepted_survivors,
            key=lambda rr: (
                -float(rr["newly_flagged_fraction"]),
                -int(rr["accepted_clusters"]),
                -(float(rr["rtc_row_severity"]) if np.isfinite(float(rr["rtc_row_severity"])) else -1.0),
            ),
        )
        case_pools["Accepted Survivor"] = [dict(row) for row in pool]

    accepted_reinforced = [
        row for row in enriched if int(row["accepted_clusters"]) > 0 and int(row["rtc_imp_mask"]) != 0
    ]
    if accepted_reinforced:
        pool = sorted(
            accepted_reinforced,
            key=lambda rr: (
                -float(rr["newly_flagged_fraction"]),
                -int(rr["accepted_clusters"]),
                -int(rr["rtc_override"]),
                -int(rr["rtc_cross"]),
            ),
        )
        case_pools["Accepted Reinforcement"] = [dict(row) for row in pool]

    vetoed = [row for row in enriched if int(row["busy_vetoed"]) != 0 and int(row["candidate_clusters"]) > 0]
    if vetoed:
        pool = sorted(
            vetoed,
            key=lambda rr: (
                -int(rr["candidate_clusters"]),
                -(float(rr["rtc_row_severity"]) if np.isfinite(float(rr["rtc_row_severity"])) else -1.0),
            ),
        )
        case_pools["Busy Veto"] = [dict(row) for row in pool]
    return case_pools


def _load_case_example(redu_dir: Path, row: dict[str, object], match_tol_samples: int = 8) -> CaseExample:
    obsnum = str(row["obsnum"])
    scan_want = int(row["output_scan_index"])
    nw_want = int(row["network"])
    ptc_path = redu_dir / obsnum / "raw" / f"toltec_commissioning_science_{obsnum}_ptc_timestream.nc"
    rtc_path = redu_dir / obsnum / "raw" / f"toltec_commissioning_science_{obsnum}_rtcdiag.nc"

    with netCDF4.Dataset(ptc_path) as ds_ptc, netCDF4.Dataset(rtc_path) as ds_rtc:
        out = _filled(ds_ptc.variables["output_scan_index"], fill=-1).astype(int)
        idx = int(np.where(out == scan_want)[0][0])
        scan_indices = _filled(ds_ptc.variables["scan_indices"], fill=-1).astype(int)
        start, end = [int(v) for v in scan_indices[idx, :2]]
        signal = _filled(ds_ptc.variables["signal"], fill=np.nan).astype(float)[start : end + 1, :]
        flags = _filled(ds_ptc.variables["flags"], fill=0).astype(int)[start : end + 1, :]
        added_flags = _filled(ds_ptc.variables["ptc_second_pass_added_flag"], fill=0).astype(int)[start : end + 1, :]
        apt_nw = _filled(ds_ptc.variables["apt_nw"], fill=-1).astype(int)
        apt_uid = _filled(ds_ptc.variables["apt_uid"], fill=-1).astype(int)
        dt_sec = _dt_from_dataset(ds_ptc)
        zfull = _detector_z_matrix(signal, flags)

        det_cols = np.where(apt_nw == nw_want)[0]
        if det_cols.size == 0:
            raise RuntimeError(f"no detectors for obs {obsnum} nw {nw_want}")
        local_added = added_flags[:, det_cols] != 0
        if np.any(local_added):
            local_z_for_pick = np.where(local_added & np.isfinite(zfull[:, det_cols]), np.abs(zfull[:, det_cols]), -np.inf)
            loc = int(np.nanargmax(local_z_for_pick))
            det_row, det_local_j = np.unravel_index(loc, local_z_for_pick.shape)
        else:
            local_good = np.where((flags[:, det_cols] == 0) & np.isfinite(zfull[:, det_cols]), np.abs(zfull[:, det_cols]), -np.inf)
            loc = int(np.nanargmax(local_good))
            det_row, det_local_j = np.unravel_index(loc, local_good.shape)
        det_col = int(det_cols[det_local_j])
        center_sample = int(det_row)

        rtc_networks = _filled(ds_rtc.variables["rtc_diag_network_ids"], fill=-1).astype(int)
        rtc_nw_idx = int(np.where(rtc_networks == nw_want)[0][0])
        rtc_row = scan_want - 1
        slot_score = _filled(ds_rtc.variables["rtc_impulsive_slot_event_score"], fill=np.nan).astype(float)[rtc_row, rtc_nw_idx, :]
        slot_kind = _filled(ds_rtc.variables["rtc_impulsive_slot_event_kind"], fill=-1).astype(int)[rtc_row, rtc_nw_idx, :]
        slot_sample = _filled(ds_rtc.variables["rtc_impulsive_slot_event_sample"], fill=-1).astype(int)[rtc_row, rtc_nw_idx, :]
        slot_det = _filled(ds_rtc.variables["rtc_impulsive_slot_det_index"], fill=-1).astype(int)[rtc_row, rtc_nw_idx, :]
        slot_z = _filled(ds_rtc.variables["rtc_impulsive_slot_snippet_z"], fill=np.nan).astype(float)[rtc_row, rtc_nw_idx, :, :]
        slot_flag = _filled(ds_rtc.variables["rtc_impulsive_slot_snippet_flag"], fill=0).astype(int)[rtc_row, rtc_nw_idx, :, :] != 0
        offsets = _filled(ds_rtc.variables["rtc_impulsive_snippet_offset_samples"], fill=0).astype(float)
        finite = np.isfinite(slot_score) & (slot_sample >= 0) & (slot_det >= 0)
        if not np.any(finite):
            raise RuntimeError(f"no RTC slots for obs {obsnum} scan {scan_want} nw {nw_want}")
        sample_delta = np.where(finite, np.abs(slot_sample - center_sample), np.iinfo(np.int32).max)
        slot_uid = np.full(slot_det.shape, -1, dtype=int)
        valid_det = slot_det >= 0
        slot_uid[valid_det] = apt_uid[slot_det[valid_det]]
        same_uid = slot_uid == int(apt_uid[det_col])
        same_uid_key = np.where(same_uid & finite, 0, np.where(finite, 1, 2))
        score_key = np.where(finite, -slot_score, np.inf)
        slot_idx = int(np.lexsort((score_key, sample_delta, same_uid_key))[0])
        if int(sample_delta[slot_idx]) > match_tol_samples:
            raise RuntimeError(
                f"no RTC slot within {match_tol_samples} samples for obs {obsnum} "
                f"scan {scan_want} nw {nw_want}"
            )
        rtc_sample = int(slot_sample[slot_idx])
        rtc_offset_sec = (offsets + (rtc_sample - center_sample)) * dt_sec

        n_pre = int(round(0.25 / dt_sec))
        n_post = int(round(0.75 / dt_sec))
        lo = max(0, center_sample - n_pre)
        hi = min(signal.shape[0], center_sample + n_post + 1)
        t_rel = (np.arange(lo, hi, dtype=float) - center_sample) * dt_sec

        return CaseExample(
            label=str(row["case_label"]),
            obsnum=obsnum,
            output_scan_index=scan_want,
            network=nw_want,
            candidate_clusters=int(row["candidate_clusters"]),
            accepted_clusters=int(row["accepted_clusters"]),
            busy_vetoed=int(row["busy_vetoed"]),
            newly_flagged_fraction=float(row["newly_flagged_fraction"]),
            rtc_imp_mask=int(row["rtc_imp_mask"]),
            rtc_cross=int(row["rtc_cross"]),
            rtc_override=int(row["rtc_override"]),
            rtc_row_severity=float(row["rtc_row_severity"]),
            center_sample=center_sample,
            det_uid=int(apt_uid[det_col]),
            det_local_z=float(zfull[center_sample, det_col]),
            rtc_slot_uid=int(slot_uid[slot_idx]),
            rtc_slot_kind="raw_like" if int(slot_kind[slot_idx]) == 0 else "delta_like",
            rtc_slot_score=float(slot_score[slot_idx]),
            rtc_slot_sample=rtc_sample,
            rtc_ptc_sample_delta=int(sample_delta[slot_idx]),
            rtc_ptc_same_uid=bool(same_uid[slot_idx]),
            rtc_offset_sec=np.asarray(rtc_offset_sec, dtype=float),
            rtc_snippet_z=np.asarray(slot_z[slot_idx, :], dtype=float),
            rtc_snippet_flag=np.asarray(slot_flag[slot_idx, :], dtype=bool),
            ptc_t_rel=t_rel,
            ptc_z=np.asarray(zfull[lo:hi, det_col], dtype=float),
            ptc_flag=np.asarray(flags[lo:hi, det_col] != 0, dtype=bool),
            ptc_added_flag=np.asarray(added_flags[lo:hi, det_col] != 0, dtype=bool),
        )


def _plot_case(example: CaseExample, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=False, constrained_layout=True)
    fig.suptitle(
        f"{example.label}: obs {example.obsnum} scan {example.output_scan_index} nw {example.network}\n"
        f"cand={example.candidate_clusters} acc={example.accepted_clusters} veto={example.busy_vetoed} "
        f"new_frac={example.newly_flagged_fraction:.5f} rtc_imp_mask={example.rtc_imp_mask}",
        fontsize=18,
    )

    ax = axes[0]
    ax.plot(example.rtc_offset_sec, example.rtc_snippet_z, color="#1b6ca8", lw=2)
    if np.any(example.rtc_snippet_flag):
        ax.scatter(
            example.rtc_offset_sec[example.rtc_snippet_flag],
            example.rtc_snippet_z[example.rtc_snippet_flag],
            s=44,
            color="#cf3f3f",
            zorder=3,
            label="RTC flagged",
        )
    ax.axvline(0.0, color="#666666", ls="--", lw=1.2)
    ax.axhline(0.0, color="#bbbbbb", lw=1.0)
    ax.set_title(
        f"RTC slot: uid {example.rtc_slot_uid} {example.rtc_slot_kind} "
        f"score {example.rtc_slot_score:.1f} sample {example.rtc_slot_sample} "
        f"(Δ={example.rtc_ptc_sample_delta} samp"
        + (" same uid)" if example.rtc_ptc_same_uid else ")"),
        fontsize=13,
    )
    ax.set_ylabel("RTC snippet z")
    ax.grid(True, alpha=0.25)
    if np.any(example.rtc_snippet_flag):
        ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(example.ptc_t_rel, example.ptc_z, color="#355070", lw=2, label="PTC detector")
    if np.any(example.ptc_flag):
        ax.scatter(
            example.ptc_t_rel[example.ptc_flag],
            example.ptc_z[example.ptc_flag],
            s=40,
            color="#b0b0b0",
            zorder=3,
            label="existing flag",
        )
    if np.any(example.ptc_added_flag):
        ax.scatter(
            example.ptc_t_rel[example.ptc_added_flag],
            example.ptc_z[example.ptc_added_flag],
            s=54,
            color="#d76a03",
            zorder=4,
            label="PTC second pass",
        )
    ax.axvline(0.0, color="#666666", ls="--", lw=1.2)
    ax.axhline(0.0, color="#bbbbbb", lw=1.0)
    ax.set_title(
        f"PTC detector: uid {example.det_uid} local z {example.det_local_z:.2f} "
        f"rtc_cross={example.rtc_cross} rtc_override={example.rtc_override} "
        f"rtc_severity={example.rtc_row_severity:.2f}",
        fontsize=13,
    )
    ax.set_xlabel("time from chosen PTC sample [s]")
    ax.set_ylabel("PTC detector z")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _write_report(
    outpath: Path,
    redu_dir: Path,
    array: str,
    rtc_obs_rows: list[dict[str, object]],
    rtc_top_rows: list[dict[str, object]],
    ptc_obs_rows: list[dict[str, object]],
    case_examples: list[CaseExample],
) -> None:
    row_counts = sorted({int(row.get("n_scan_network_rows", 0)) for row in rtc_obs_rows if int(row.get("n_scan_network_rows", 0)) > 0})
    if len(row_counts) == 1:
        row10 = max(1, int(round(0.10 * row_counts[0])))
        row40 = max(1, int(round(0.40 * row_counts[0])))
        row70 = max(1, int(round(0.70 * row_counts[0])))
        row_count_note = (
            f"In this report each obsnum has `{row_counts[0]}` scan/network rows, "
            f"so roughly `{row10}`, `{row40}`, and `{row70}` masked rows correspond to about "
            "`10%`, `40%`, and `70%`."
        )
    elif row_counts:
        row_count_note = (
            f"In this report obsnums span `{row_counts[0]}` to `{row_counts[-1]}` scan/network rows. "
            "Interpret `imp masked rows` as a fraction of each obsnum's total rows, not as a universal count."
        )
    else:
        row_count_note = (
            "Interpret `imp masked rows` as a fraction of each obsnum's total scan/network rows, "
            "not as a universal count."
        )

    lines = [
        f"# Despike Diagnostic Report: {redu_dir.name}",
        "",
        f"- Reduction directory: `{redu_dir}`",
        f"- Array selection: `{array}`",
        f"- RTC obsnums summarized: {len(rtc_obs_rows)}",
        f"- PTC obsnums with second-pass output: {len(ptc_obs_rows)}",
        "",
        "## What The Current Stack Does",
        "",
        "- `RTC local impulsive capture`: records compact raw-like and delta-like candidates for later inspection.",
        "- `RTC impulsive coincidence mask`: pre-PCA masking path for sparse but aligned multi-network bursts.",
        "- `RTC step mask`: pre-PCA masking path for coherent step-like behavior.",
        "- `PTC second pass local`: post-PCA detector-local residual cleanup. It merges repeated hits, clusters them within each network, auto-flags only conservative clusters, and busy-vetoes stormy scan/network rows.",
        "",
        "## Key Column Definitions",
        "",
        "These are engineering triage heuristics for blank-sky reductions. They are not universal pass/fail cuts, and none of the three metrics below should be used alone.",
        "",
        f"- `max severity`: the worst single RTC `row_severity` seen anywhere in the obsnum. `row_severity` is a ranking score, not a calibrated physical unit.",
        "  - `< 5`: quiet / usually acceptable",
        "  - `5 - 15`: moderate contamination; usually acceptable if localized",
        "  - `15 - 30`: strong contamination in at least one row; should be explainable by masking or a known bad chunk",
        "  - `> 30`: extreme row; failure-candidate unless clearly contained",
        "",
        "- `imp masked rows`: number of scan/network rows where the RTC impulsive mask fired.",
        f"  - {row_count_note}",
        "  - `< 10%` of rows: selective / comfortable",
        "  - `10% - 40%`: active but often acceptable on hard obsnums",
        "  - `40% - 70%`: aggressive; review carefully",
        "  - `> 70%`: likely overfiring / failure-candidate unless the obs is obviously burst-dominated",
        "",
        "- `top slot score`: strongest captured RTC compact-event score in the obsnum.",
        "  - `< 10`: mild compact event activity",
        "  - `10 - 30`: clear impulsive event",
        "  - `30 - 100`: very strong event; should be localized or masked",
        "  - `> 100`: extreme outlier; acceptable only if it is clearly contained and does not survive downstream",
        "",
        "## PTC Summary Column Definitions",
        "",
        "These columns describe what the post-PCA second pass saw in the PTC residuals. They are useful only together.",
        "",
        "- `candidate clusters`: total number of second-pass network clusters that met the candidate rule before any busy-veto logic.",
        "  - `< 20`: sparse residual activity / comfortable",
        "  - `20 - 60`: active but still plausible on hard blank-sky obs",
        "  - `60 - 120`: very busy; should usually be accompanied by vetoes or very small added fractions",
        "  - `> 120`: failure-candidate unless this was an intentionally pathological debug case",
        "",
        "- `accepted clusters`: total number of candidate clusters that survived the conservative auto-flag rules and actually added flags.",
        "  - `< 20`: very conservative",
        "  - `20 - 50`: active but still reasonable on hard obs",
        "  - `> 50`: aggressive; check added-flag fraction and whether storm cases are being vetoed",
        "  - close to `candidate clusters` on a visibly messy obs is a failure sign",
        "",
        "- `busy-veto rows`: scan/network rows where the second pass saw many candidate clusters and deliberately refused to auto-flag that network.",
        "  - `0 - 5`: little need for storm protection",
        "  - `5 - 20`: active but healthy veto usage on hard obs",
        "  - `> 20`: many storm-like rows; review the obs even if no extra flags were added",
        "  - not a failure by itself; high veto counts are often better than blindly flagging everything",
        "",
        "- `added-flag fraction`: total fraction of all PTC detector-samples in the file that were newly flagged by the second pass.",
        "  - `< 1e-4`: very conservative / comfortable",
        "  - `1e-4 - 1e-3`: noticeable but usually still acceptable",
        "  - `1e-3 - 1e-2`: aggressive",
        "  - `> 1e-2`: likely over-flagging / failure-candidate",
        "",
        "## RTC Summary",
        "",
        "| obsnum | max severity | imp masked rows | top slot score |",
        "|---:|---:|---:|---:|",
    ]
    for row in sorted(rtc_obs_rows, key=lambda rr: float(rr["max_row_severity"]), reverse=True):
        lines.append(
            f"| {row['obsnum']} | {float(row['max_row_severity']):.2f} | "
            f"{int(row['impulsive_masked_network_scans'])} | {float(row['top_slot_event_score']):.2f} |"
        )

    lines.extend(["", "## PTC Second-Pass Summary", "", "| obsnum | candidate clusters | accepted clusters | busy-veto rows | added-flag fraction |", "|---:|---:|---:|---:|---:|"])
    for row in ptc_obs_rows:
        lines.append(
            f"| {row['obsnum']} | {int(row['candidate_clusters_total'])} | {int(row['accepted_clusters_total'])} | "
            f"{int(row['busy_vetoed_rows'])} | {float(row['file_added_fraction_total']):.6f} |"
        )

    lines.extend(["", "## Top RTC Scan/Network Rows", ""])
    for row in rtc_top_rows[:8]:
        lines.append(
            f"- obs `{row['obsnum']}` scan `{int(row['output_scan_index'])}` nw `{int(row['network'])}` "
            f"severity `{float(row['row_severity']):.2f}` step_mask `{int(row['step_mask_applied'])}` "
            f"imp_mask `{int(row['impulsive_mask_applied'])}` cross `{int(row['impulsive_mask_cross_network_trigger'])}` "
            f"override `{int(row['impulsive_mask_high_score_override_trigger'])}` "
            f"slot `{float(row['max_slot_event_score']):.2f}`"
        )

    lines.extend(["", "## Representative PTC Second-Pass Cases", ""])
    if not case_examples:
        lines.append("- none")
    for example in case_examples:
        png = (
            f"despike_case_{example.label.lower().replace(' ', '_')}"
            f"_obs{example.obsnum}_scan{example.output_scan_index}_nw{example.network}.png"
        )
        if example.busy_vetoed:
            behavior = "PTC saw many candidate clusters here and deliberately refused to auto-flag the network."
        elif example.accepted_clusters > 0 and example.rtc_imp_mask == 0:
            behavior = "RTC left this row alone, but PTC second pass found a compact residual survivor and added flags."
        else:
            behavior = "RTC already treated this row as impulsive, and PTC second pass still found a small residual worth cleaning."
        lines.extend(
            [
                f"### {example.label}",
                "",
                f"- obs `{example.obsnum}` scan `{example.output_scan_index}` nw `{example.network}`",
                f"- candidate clusters `{example.candidate_clusters}` accepted clusters `{example.accepted_clusters}` busy_veto `{example.busy_vetoed}`",
                f"- newly flagged fraction `{example.newly_flagged_fraction:.6f}`",
                f"- RTC mask state: imp_mask `{example.rtc_imp_mask}` cross `{example.rtc_cross}` override `{example.rtc_override}` severity `{example.rtc_row_severity:.2f}`",
                f"- detector pairing: PTC uid `{example.det_uid}` vs RTC uid `{example.rtc_slot_uid}` sample delta `{example.rtc_ptc_sample_delta}` same_uid `{int(example.rtc_ptc_same_uid)}`",
                f"- interpretation: {behavior}",
                f"- figure: [{png}]({png})",
                "",
            ]
        )
    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redu-dir", required=True, help="Reduction directory, e.g. .../reduced/redu65")
    parser.add_argument("--array", default="a1100", choices=["a1100", "a1400", "a2000"])
    parser.add_argument("--networks", default="all")
    parser.add_argument("--obsnums", default="all")
    parser.add_argument("--outdir", default=None, help="Default: <redu-dir>/despike_diagnostic_report")
    args = parser.parse_args()

    redu_dir = Path(args.redu_dir).expanduser().resolve()
    if not redu_dir.is_dir():
        raise NotADirectoryError(redu_dir)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (redu_dir / "despike_diagnostic_report")
    outdir.mkdir(parents=True, exist_ok=True)

    rtc_dataset = load_reduction_tables(
        redu_dir=redu_dir,
        array=args.array,
        networks_spec=args.networks,
        obsnums_spec=args.obsnums,
    )
    rtc_obs_rows = rtc_dataset["obs_rows"]
    rtc_top_rows = rtc_dataset["top_scan_rows"]
    rtc_lookup = {
        (str(row["obsnum"]), int(row["output_scan_index"]), int(row["network"])): row
        for row in rtc_dataset["scan_network_rows"]
    }

    ptc_rows = _collect_ptc_rows(redu_dir)
    ptc_obs_rows = _aggregate_ptc_obs(ptc_rows)
    _write_csv(outdir / "ptc_second_pass_by_scan_network.csv", ptc_rows if ptc_rows else [{"obsnum": ""}])
    _write_csv(outdir / "ptc_second_pass_by_obsnum.csv", ptc_obs_rows if ptc_obs_rows else [{"obsnum": ""}])

    case_pools = _choose_cases(ptc_rows, rtc_lookup)
    case_examples: list[CaseExample] = []
    for label, pool in case_pools.items():
        example = None
        for row in pool:
            row["case_label"] = label
            try:
                example = _load_case_example(redu_dir, row)
                break
            except RuntimeError:
                continue
        if example is None:
            continue
        case_examples.append(example)
        png_name = (
            f"despike_case_{example.label.lower().replace(' ', '_')}"
            f"_obs{example.obsnum}_scan{example.output_scan_index}_nw{example.network}.png"
        )
        _plot_case(example, outdir / png_name)

    _write_report(
        outdir / "DESPIKE_DIAGNOSTIC_REPORT.md",
        redu_dir,
        args.array,
        rtc_obs_rows,
        rtc_top_rows,
        ptc_obs_rows,
        case_examples,
    )

    print(f"Wrote {outdir / 'ptc_second_pass_by_scan_network.csv'}")
    print(f"Wrote {outdir / 'ptc_second_pass_by_obsnum.csv'}")
    print(f"Wrote {outdir / 'DESPIKE_DIAGNOSTIC_REPORT.md'}")


if __name__ == "__main__":
    main()
