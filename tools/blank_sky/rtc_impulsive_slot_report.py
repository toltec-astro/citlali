#!/usr/bin/env python3
"""Summarize and plot captured RTC impulsive-event slots.

This reads the compact `rtc_impulsive_slot_*` products written into RTC
timestream netCDF files and turns them into:

- a detailed per-event CSV
- a per-network summary CSV
- a short markdown report
- a gallery plot of the captured standardized snippets

The intended use is quick triage of the new impulsive instrumentation so we can
see whether the surviving bad networks are dominated by raw-like excursions,
delta-like excursions, or events that were not actually touched by despiking.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import netCDF4
import numpy as np

from blank_sky_null_audit import _parse_networks, _parse_scans, _resolve_obsnum, _write_csv
from mp_mode_estimator import _infer_dt_sec

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional runtime dependency
    plt = None


ARRAY_NAME_TO_ID = {"a1100": 0, "a1400": 1, "a2000": 2}
ARRAY_ID_TO_NAME = {value: key for key, value in ARRAY_NAME_TO_ID.items()}
EVENT_KIND_LABELS = {0: "raw_like", 1: "delta_like"}


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


def _array_selector(spec: str) -> set[int]:
    if spec == "all":
        return set(ARRAY_ID_TO_NAME)
    return {ARRAY_NAME_TO_ID[spec]}


def _scan_dt_sec(ds: netCDF4.Dataset, scan_idx: int) -> float:
    for name in ("scan_indices", "raw_scan_indices"):
        if name not in ds.variables:
            continue
        idx = np.asarray(ds.variables[name][scan_idx], dtype=int).reshape(-1)
        idx = idx[idx >= 0]
        if idx.size >= 2:
            return float(_infer_dt_sec(ds, int(idx[0]), int(idx[-1])))
    return 1.0


def _kind_label(kind: int) -> str:
    return EVENT_KIND_LABELS.get(int(kind), "unknown")


def _fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _top_rows(rows: list[dict[str, object]], key: str, n: int = 10) -> list[dict[str, object]]:
    ranked: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        value = float(row.get(key, float("nan")))
        if np.isfinite(value):
            ranked.append((value, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked[:n]]


def _make_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    networks = sorted(set(int(row["network"]) for row in rows))
    summary: list[dict[str, object]] = []
    for nw in networks:
        rr = [row for row in rows if int(row["network"]) == nw]
        scores = np.asarray([float(row["event_score"]) for row in rr], dtype=float)
        raw_like = np.asarray([int(row["event_kind"]) == 0 for row in rr], dtype=float)
        delta_like = np.asarray([int(row["event_kind"]) == 1 for row in rr], dtype=float)
        untouched = np.asarray(
            [
                int(row["raw_exceed_count"]) <= 0
                and int(row["delta_spike_count"]) <= 0
                and int(row["local_raw_accepted_event_count"]) <= 0
                and int(row["local_flagged_sample_count"]) <= 0
                and int(row["local_delta_accepted_event_count"]) <= 0
                and float(row["added_flagged_frac"]) <= 0
                for row in rr
            ],
            dtype=float,
        )
        touched_by_local = np.asarray(
            [
                int(row["local_raw_accepted_event_count"]) > 0
                or int(row["local_flagged_sample_count"]) > 0
                or int(row["local_delta_accepted_event_count"]) > 0
                for row in rr
            ],
            dtype=float,
        )
        local_raw_candidates = np.asarray(
            [int(row["local_raw_candidate_count"]) > 0 for row in rr],
            dtype=float,
        )
        local_raw_rejected = np.asarray(
            [int(row["local_raw_reject_count"]) > 0 for row in rr],
            dtype=float,
        )
        local_delta_candidates = np.asarray(
            [int(row["local_delta_candidate_count"]) > 0 for row in rr],
            dtype=float,
        )
        local_delta_rejected = np.asarray(
            [int(row["local_delta_reject_count"]) > 0 for row in rr],
            dtype=float,
        )
        summary.append(
            {
                "network": nw,
                "n_events": len(rr),
                "n_scans_with_events": len(set(int(row["scan"]) for row in rr)),
                "median_event_score": float(np.median(scores)),
                "max_event_score": float(np.max(scores)),
                "median_peak_abs_z": float(np.median([float(row["peak_abs_z"]) for row in rr])),
                "median_peak_delta_abs_z": float(np.median([float(row["peak_delta_abs_z"]) for row in rr])),
                "median_added_flagged_frac": float(np.median([float(row["added_flagged_frac"]) for row in rr])),
                "median_raw_exceed_count": float(np.median([float(row["raw_exceed_count"]) for row in rr])),
                "median_local_raw_candidate_count": float(np.median([float(row["local_raw_candidate_count"]) for row in rr])),
                "median_local_raw_accepted_event_count": float(np.median([float(row["local_raw_accepted_event_count"]) for row in rr])),
                "median_local_flagged_sample_count": float(np.median([float(row["local_flagged_sample_count"]) for row in rr])),
                "median_local_raw_reject_count": float(np.median([float(row["local_raw_reject_count"]) for row in rr])),
                "median_delta_spike_count": float(np.median([float(row["delta_spike_count"]) for row in rr])),
                "median_local_delta_candidate_count": float(np.median([float(row["local_delta_candidate_count"]) for row in rr])),
                "median_local_delta_accepted_event_count": float(np.median([float(row["local_delta_accepted_event_count"]) for row in rr])),
                "median_local_delta_reject_count": float(np.median([float(row["local_delta_reject_count"]) for row in rr])),
                "frac_raw_like": float(np.mean(raw_like)),
                "frac_delta_like": float(np.mean(delta_like)),
                "frac_with_local_raw_candidates": float(np.mean(local_raw_candidates)),
                "frac_with_local_raw_rejects": float(np.mean(local_raw_rejected)),
                "frac_with_local_delta_candidates": float(np.mean(local_delta_candidates)),
                "frac_with_local_delta_rejects": float(np.mean(local_delta_rejected)),
                "frac_touched_by_local": float(np.mean(touched_by_local)),
                "frac_untouched_by_despike": float(np.mean(untouched)),
            }
        )
    return summary


def _write_report(
    outpath: Path,
    nc_file: Path,
    array: str,
    networks: list[int],
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    gallery_path: Path | None,
) -> None:
    top_events = _top_rows(rows, "event_score", n=12)
    untouched = [
        row
        for row in rows
        if int(row["raw_exceed_count"]) <= 0
        and int(row["delta_spike_count"]) <= 0
        and int(row["local_raw_accepted_event_count"]) <= 0
        and int(row["local_flagged_sample_count"]) <= 0
        and int(row["local_delta_accepted_event_count"]) <= 0
        and float(row["added_flagged_frac"]) <= 0
    ]
    touched_by_local = [
        row
        for row in rows
        if int(row["local_raw_accepted_event_count"]) > 0
        or int(row["local_flagged_sample_count"]) > 0
        or int(row["local_delta_accepted_event_count"]) > 0
    ]
    with_local_delta_candidates = [
        row for row in rows if int(row["local_delta_candidate_count"]) > 0
    ]
    with_local_delta_rejects = [
        row for row in rows if int(row["local_delta_reject_count"]) > 0
    ]
    with_local_raw_candidates = [
        row for row in rows if int(row["local_raw_candidate_count"]) > 0
    ]
    with_local_raw_rejects = [
        row for row in rows if int(row["local_raw_reject_count"]) > 0
    ]
    lines = [
        f"# RTC Impulsive Slot Report: {nc_file.name}",
        "",
        f"- Input file: `{nc_file}`",
        f"- Array selection: `{array}`",
        f"- Networks: `{','.join(str(nw) for nw in networks)}`",
        f"- Captured events: {len(rows)}",
        f"- Events touched by local-residual despike: {len(touched_by_local)}",
        f"- Events with compact-gate local-raw candidates: {len(with_local_raw_candidates)}",
        f"- Events with compact-gate local-raw rejects: {len(with_local_raw_rejects)}",
        f"- Events with compact-gate local-delta candidates: {len(with_local_delta_candidates)}",
        f"- Events with compact-gate local-delta rejects: {len(with_local_delta_rejects)}",
        f"- Events untouched by despike counters: {len(untouched)}",
    ]
    if gallery_path is not None:
        lines.append(f"- Gallery plot: `{gallery_path.name}`")
    lines.extend(
        [
            "",
            "## Network Summary",
            "",
            "| nw | n_events | scans | med score | max score | med peak z | med delta z | med add-flag frac | raw frac | delta frac | raw cand | raw rej | delta cand | delta rej | local-touch | untouched |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {network} | {n_events} | {n_scans_with_events} | {median_event_score:.2f} | "
            "{max_event_score:.2f} | {median_peak_abs_z:.2f} | {median_peak_delta_abs_z:.2f} | "
            "{median_added_flagged_frac:.4f} | {frac_raw_like:.2f} | {frac_delta_like:.2f} | "
            "{frac_with_local_raw_candidates:.2f} | {frac_with_local_raw_rejects:.2f} | "
            "{frac_with_local_delta_candidates:.2f} | {frac_with_local_delta_rejects:.2f} | {frac_touched_by_local:.2f} | "
            "{frac_untouched_by_despike:.2f} |".format(**row)
        )
    lines.extend(["", "## Top Events", ""])
    if not top_events:
        lines.append("- none")
    else:
        for row in top_events:
            lines.append(
                "- scan={scan} output_scan={output_scan_index} nw={network} slot={slot} det_uid={apt_uid} "
                "kind={event_kind_label} score={event_score:.2f} sample={event_sample} "
                "peak_z={peak_abs_z:.2f} delta_z={peak_delta_abs_z:.2f} raw_count={raw_exceed_count} "
                "local_raw_cand={local_raw_candidate_count} local_raw_accept={local_raw_accepted_event_count} local_flag_samples={local_flagged_sample_count} "
                "local_raw_reject={local_raw_reject_count} delta_count={delta_spike_count} "
                "local_delta_cand={local_delta_candidate_count} local_delta_accept={local_delta_accepted_event_count} "
                "local_delta_reject={local_delta_reject_count} add_flag_frac={added_flagged_frac:.4f}".format(**row)
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `raw_like` means the strongest captured excursion came from absolute sample amplitude.",
            "- `delta_like` means the strongest captured excursion came from adjacent-sample differences.",
            "- `raw cand` / `raw rej` and `delta cand` / `delta rej` show how often the compact morphology gates saw a candidate but rejected it as too broad or too step-like.",
            "- High `untouched` fractions mean the most suspicious events are not being counted by either the native or local-residual despike counters.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n")


def _plot_gallery(
    outpath: Path,
    rows: list[dict[str, object]],
    offsets: np.ndarray,
    max_plots: int,
) -> Path | None:
    if plt is None or not rows:
        return None
    ranked = sorted(rows, key=lambda row: float(row["event_score"]), reverse=True)
    ranked = ranked[: max(1, int(max_plots))]
    n = len(ranked)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), squeeze=False, sharex=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, row in zip(axes.ravel(), ranked):
        snippet = np.asarray(row["snippet_z"], dtype=float)
        flag = np.asarray(row["snippet_flag"], dtype=bool)
        ax.plot(offsets, snippet, color="tab:blue", lw=1.3)
        if np.any(flag):
            ax.scatter(offsets[flag], snippet[flag], color="tab:red", s=12, zorder=3)
        ax.axvline(0.0, color="0.35", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.75", lw=0.6)
        ax.set_title(
            "scan {output_scan_index} nw{network} slot{slot} uid {apt_uid}\n"
            "{event_kind_label} score {event_score:.1f} peak {peak_abs_z:.1f} dpeak {peak_delta_abs_z:.1f}".format(**row),
            fontsize=9,
        )
        ax.set_ylabel("z")
        ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("time from event [s]")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc-file", required=True, help="Path to RTC timestream netCDF with impulsive slot products")
    ap.add_argument("--obsnum", default=None, help="Obsnum label override")
    ap.add_argument("--array", default="all", choices=["all", "a1100", "a1400", "a2000"])
    ap.add_argument("--networks", default="all", help="Comma list or 'all'")
    ap.add_argument("--scans", default="all", help="Internal scan indices or 'all'")
    ap.add_argument("--outdir", default=None, help="Default: <nc parent>/rtc_impulsive_slot_report")
    ap.add_argument("--max-plots", type=int, default=12, help="Maximum events to include in gallery plot")
    args = ap.parse_args()

    nc_file = Path(args.nc_file).expanduser().resolve()
    if not nc_file.exists():
        raise FileNotFoundError(nc_file)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (nc_file.parent / "rtc_impulsive_slot_report")
    outdir.mkdir(parents=True, exist_ok=True)

    required = [
        "apt_array",
        "apt_nw",
        "apt_uid",
        "rtc_diag_network_ids",
        "output_scan_index",
        "rtc_impulsive_slot_det_index",
        "rtc_impulsive_slot_event_sample",
        "rtc_impulsive_slot_event_kind",
        "rtc_impulsive_slot_event_score",
        "rtc_impulsive_slot_peak_abs_z",
        "rtc_impulsive_slot_peak_delta_abs_z",
        "rtc_impulsive_slot_added_flagged_frac",
        "rtc_impulsive_slot_raw_exceed_count",
        "rtc_impulsive_slot_delta_spike_count",
        "rtc_impulsive_slot_snippet_z",
        "rtc_impulsive_slot_snippet_flag",
        "rtc_impulsive_snippet_offset_samples",
    ]

    with netCDF4.Dataset(nc_file) as ds:
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise KeyError(f"missing RTC impulsive slot variable(s): {missing}")

        apt_array = _filled(ds.variables["apt_array"], fill=np.nan)
        apt_nw = _filled(ds.variables["apt_nw"], fill=np.nan)
        apt_uid = _filled(ds.variables["apt_uid"], fill=np.nan)
        apt_x = _filled(ds.variables["apt_x_t"], fill=np.nan) if "apt_x_t" in ds.variables else np.full_like(apt_uid, np.nan)
        apt_y = _filled(ds.variables["apt_y_t"], fill=np.nan) if "apt_y_t" in ds.variables else np.full_like(apt_uid, np.nan)

        diag_networks = _filled(ds.variables["rtc_diag_network_ids"], fill=np.iinfo(np.int32).min).astype(int)
        output_scans = _filled(ds.variables["output_scan_index"], fill=np.iinfo(np.int32).min).astype(int)
        n_scans = int(output_scans.size)
        scan_indices = _parse_scans(args.scans, n_scans)

        array_ids = _array_selector(args.array)
        det_array_ids = np.rint(apt_array).astype(int)
        det_network_ids = np.rint(apt_nw).astype(int)
        det_keep = np.isfinite(apt_array) & np.isfinite(apt_nw) & np.isin(det_array_ids, list(array_ids))
        available_networks = det_network_ids[det_keep]
        networks = _parse_networks(args.networks, available_networks)

        diag_keep = [i for i, nw in enumerate(diag_networks) if int(nw) in networks]
        if not diag_keep:
            raise ValueError(f"no selected RTC diagnostic networks for array={args.array} networks={networks}")

        slot_det = _filled(ds.variables["rtc_impulsive_slot_det_index"], fill=np.iinfo(np.int32).min).astype(int)
        slot_sample = _filled(ds.variables["rtc_impulsive_slot_event_sample"], fill=np.iinfo(np.int32).min).astype(int)
        slot_kind = _filled(ds.variables["rtc_impulsive_slot_event_kind"], fill=np.iinfo(np.int32).min).astype(int)
        slot_score = _filled(ds.variables["rtc_impulsive_slot_event_score"], fill=np.nan)
        slot_peak = _filled(ds.variables["rtc_impulsive_slot_peak_abs_z"], fill=np.nan)
        slot_dpeak = _filled(ds.variables["rtc_impulsive_slot_peak_delta_abs_z"], fill=np.nan)
        slot_raw = _filled(ds.variables["rtc_impulsive_slot_raw_exceed_count"], fill=np.iinfo(np.int32).min).astype(int)
        if "rtc_impulsive_slot_local_raw_candidate_count" in ds.variables:
            slot_local_raw_candidate = _filled(ds.variables["rtc_impulsive_slot_local_raw_candidate_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_raw_candidate = np.zeros_like(slot_raw)
        if "rtc_impulsive_slot_local_raw_accepted_event_count" in ds.variables:
            slot_local_raw_accept = _filled(ds.variables["rtc_impulsive_slot_local_raw_accepted_event_count"], fill=np.iinfo(np.int32).min).astype(int)
        elif "rtc_impulsive_slot_local_exceed_count" in ds.variables:
            slot_local_raw_accept = _filled(ds.variables["rtc_impulsive_slot_local_exceed_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_raw_accept = np.zeros_like(slot_raw)
        if "rtc_impulsive_slot_local_flagged_sample_count" in ds.variables:
            slot_local_flagged = _filled(ds.variables["rtc_impulsive_slot_local_flagged_sample_count"], fill=np.iinfo(np.int32).min).astype(int)
        elif "rtc_impulsive_slot_local_exceed_count" in ds.variables:
            slot_local_flagged = _filled(ds.variables["rtc_impulsive_slot_local_exceed_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_flagged = np.zeros_like(slot_raw)
        if "rtc_impulsive_slot_local_raw_reject_count" in ds.variables:
            slot_local_raw_reject = _filled(ds.variables["rtc_impulsive_slot_local_raw_reject_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_raw_reject = np.zeros_like(slot_raw)
        slot_delta = _filled(ds.variables["rtc_impulsive_slot_delta_spike_count"], fill=np.iinfo(np.int32).min).astype(int)
        if "rtc_impulsive_slot_local_delta_accepted_event_count" in ds.variables:
            slot_local_delta_accept = _filled(ds.variables["rtc_impulsive_slot_local_delta_accepted_event_count"], fill=np.iinfo(np.int32).min).astype(int)
        elif "rtc_impulsive_slot_local_delta_exceed_count" in ds.variables:
            slot_local_delta_accept = _filled(ds.variables["rtc_impulsive_slot_local_delta_exceed_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_delta_accept = np.zeros_like(slot_delta)
        if "rtc_impulsive_slot_local_delta_candidate_count" in ds.variables:
            slot_local_delta_candidate = _filled(ds.variables["rtc_impulsive_slot_local_delta_candidate_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_delta_candidate = np.zeros_like(slot_delta)
        if "rtc_impulsive_slot_local_delta_reject_count" in ds.variables:
            slot_local_delta_reject = _filled(ds.variables["rtc_impulsive_slot_local_delta_reject_count"], fill=np.iinfo(np.int32).min).astype(int)
        else:
            slot_local_delta_reject = np.zeros_like(slot_delta)
        slot_add_flag = _filled(ds.variables["rtc_impulsive_slot_added_flagged_frac"], fill=np.nan)
        snippet_z = _filled(ds.variables["rtc_impulsive_slot_snippet_z"], fill=np.nan)
        snippet_flag = _filled(ds.variables["rtc_impulsive_slot_snippet_flag"], fill=0).astype(int) != 0
        snippet_offset_samples = _filled(ds.variables["rtc_impulsive_snippet_offset_samples"], fill=0).astype(int)

        rows: list[dict[str, object]] = []
        for scan in scan_indices:
            dt_sec = _scan_dt_sec(ds, scan)
            snippet_offset_sec = snippet_offset_samples.astype(float) * dt_sec
            output_scan = int(output_scans[scan])
            for diag_idx in diag_keep:
                nw = int(diag_networks[diag_idx])
                for slot in range(slot_det.shape[2]):
                    det_index = int(slot_det[scan, diag_idx, slot])
                    if det_index < 0 or det_index >= apt_uid.size:
                        continue
                    det_array = int(round(float(apt_array[det_index])))
                    det_network = int(round(float(apt_nw[det_index])))
                    if det_array not in array_ids or det_network != nw:
                        continue
                    row = {
                        "obsnum": _resolve_obsnum(nc_file, args.obsnum),
                        "array": ARRAY_ID_TO_NAME.get(det_array, str(det_array)),
                        "scan": int(scan),
                        "output_scan_index": output_scan,
                        "network": nw,
                        "slot": int(slot),
                        "det_index": det_index,
                        "apt_uid": int(round(float(apt_uid[det_index]))),
                        "apt_x_t": float(apt_x[det_index]),
                        "apt_y_t": float(apt_y[det_index]),
                        "event_sample": int(slot_sample[scan, diag_idx, slot]),
                        "event_time_sec": float(slot_sample[scan, diag_idx, slot]) * dt_sec,
                        "event_kind": int(slot_kind[scan, diag_idx, slot]),
                        "event_kind_label": _kind_label(int(slot_kind[scan, diag_idx, slot])),
                        "event_score": float(slot_score[scan, diag_idx, slot]),
                        "peak_abs_z": float(slot_peak[scan, diag_idx, slot]),
                        "peak_delta_abs_z": float(slot_dpeak[scan, diag_idx, slot]),
                        "raw_exceed_count": int(slot_raw[scan, diag_idx, slot]),
                        "local_raw_candidate_count": int(slot_local_raw_candidate[scan, diag_idx, slot]),
                        "local_raw_accepted_event_count": int(slot_local_raw_accept[scan, diag_idx, slot]),
                        "local_flagged_sample_count": int(slot_local_flagged[scan, diag_idx, slot]),
                        "local_raw_reject_count": int(slot_local_raw_reject[scan, diag_idx, slot]),
                        "delta_spike_count": int(slot_delta[scan, diag_idx, slot]),
                        "local_delta_candidate_count": int(slot_local_delta_candidate[scan, diag_idx, slot]),
                        "local_delta_accepted_event_count": int(slot_local_delta_accept[scan, diag_idx, slot]),
                        "local_delta_reject_count": int(slot_local_delta_reject[scan, diag_idx, slot]),
                        "added_flagged_frac": float(slot_add_flag[scan, diag_idx, slot]),
                        "snippet_peak_abs_z": float(np.nanmax(np.abs(snippet_z[scan, diag_idx, slot, :]))),
                        "snippet_flagged_frac": float(np.mean(snippet_flag[scan, diag_idx, slot, :])),
                        "snippet_z": np.asarray(snippet_z[scan, diag_idx, slot, :], dtype=float),
                        "snippet_flag": np.asarray(snippet_flag[scan, diag_idx, slot, :], dtype=bool),
                        "snippet_offset_sec": snippet_offset_sec.copy(),
                    }
                    rows.append(row)

    summary_rows = _make_summary_rows(rows) if rows else []
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        out = {k: v for k, v in row.items() if k not in {"snippet_z", "snippet_flag", "snippet_offset_sec"}}
        csv_rows.append(out)

    detailed_csv = outdir / "rtc_impulsive_slot_report_detailed.csv"
    summary_csv = outdir / "rtc_impulsive_slot_report_summary_by_network.csv"
    report_md = outdir / "RTC_IMPULSIVE_SLOT_REPORT.md"
    gallery_png = outdir / "rtc_impulsive_slot_gallery.png"

    if csv_rows:
        _write_csv(detailed_csv, csv_rows)
        _write_csv(summary_csv, summary_rows)
    else:
        _write_csv(detailed_csv, [{"note": "no impulsive slots found for selection"}])
        _write_csv(summary_csv, [{"note": "no impulsive slots found for selection"}])

    offsets = rows[0]["snippet_offset_sec"] if rows else np.asarray([], dtype=float)
    gallery_path = _plot_gallery(gallery_png, rows, np.asarray(offsets, dtype=float), args.max_plots)
    _write_report(report_md, nc_file, args.array, networks, rows, summary_rows, gallery_path)

    print(f"Wrote {detailed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")
    if gallery_path is not None:
        print(f"Wrote {gallery_path}")


if __name__ == "__main__":
    main()
