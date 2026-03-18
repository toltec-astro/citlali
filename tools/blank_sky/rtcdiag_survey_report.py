#!/usr/bin/env python3
"""Summarize RTC diagnostic products across a reduction tree.

This prefers compact `*_rtcdiag.nc` products and falls back to
`*_rtc_timestream.nc` when needed. It writes:

- per-obsnum summary CSV
- per-obsnum, per-network summary CSV
- per-network summary CSV aggregated across obsnums
- ranked scan/network row CSV
- ranked impulsive-slot CSV
- markdown survey report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np

from blank_sky_null_audit import _parse_int_list, _parse_networks, _resolve_obsnum, _write_csv


ARRAY_NAME_TO_ID = {"a1100": 0, "a1400": 1, "a2000": 2}
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


def _fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmax(values))


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmedian(values))


def _severity_score(
    step_det_frac: float,
    step_alignment_frac: float,
    cm_lowmid: float,
    max_impulsive_score: float,
    max_slot_score: float,
    step_frac_ref: float,
    alignment_ref: float,
    cm_lowmid_ref: float,
    impulsive_threshold: float,
) -> float:
    terms = []
    if np.isfinite(step_det_frac) and np.isfinite(step_alignment_frac):
        terms.append((step_det_frac / max(step_frac_ref, 1e-12)) *
                     (step_alignment_frac / max(alignment_ref, 1e-12)))
    if np.isfinite(cm_lowmid):
        terms.append(cm_lowmid / max(cm_lowmid_ref, 1e-12))
    if np.isfinite(max_impulsive_score):
        terms.append(max_impulsive_score / max(impulsive_threshold, 1e-12))
    if np.isfinite(max_slot_score):
        terms.append(max_slot_score / max(impulsive_threshold, 1e-12))
    if not terms:
        return float("nan")
    return float(max(terms))


def _find_preferred_product(raw_dir: Path) -> tuple[str, Path] | None:
    rtcdiag = sorted(raw_dir.rglob("*_rtcdiag.nc"))
    if rtcdiag:
        return "rtcdiag", rtcdiag[0]
    rtc = sorted(raw_dir.rglob("*_rtc_timestream.nc"))
    if rtc:
        return "rtc", rtc[0]
    return None


def _collect_products(redu_dir: Path) -> list[tuple[str, str, Path]]:
    products: list[tuple[str, str, Path]] = []
    for obsdir in sorted(redu_dir.iterdir()):
        if not obsdir.is_dir() or not obsdir.name.isdigit():
            continue
        raw_dir = obsdir / "raw"
        if not raw_dir.is_dir():
            continue
        product = _find_preferred_product(raw_dir)
        if product is None:
            continue
        kind, path = product
        products.append((obsdir.name, kind, path))
    return products


def _build_obs_summary(
    obsnum: str,
    product_kind: str,
    nc_file: Path,
    array: str,
    network_rows: list[dict[str, object]],
    scan_network_rows: list[dict[str, object]],
    slot_rows: list[dict[str, object]],
    impulsive_scores: np.ndarray,
    impulsive_threshold: float,
) -> dict[str, object]:
    mask_flags = np.asarray([float(row["step_mask_applied"]) for row in scan_network_rows], dtype=float)
    mask_fracs = np.asarray([float(row["step_mask_flagged_fraction"]) for row in scan_network_rows], dtype=float)
    impulsive_mask_flags = np.asarray([float(row.get("impulsive_mask_applied", 0.0)) for row in scan_network_rows], dtype=float)
    impulsive_mask_fracs = np.asarray([float(row.get("impulsive_mask_flagged_fraction", float("nan"))) for row in scan_network_rows], dtype=float)
    step_det = np.asarray([float(row["step_det_frac"]) for row in scan_network_rows], dtype=float)
    step_align = np.asarray([float(row["step_alignment_frac"]) for row in scan_network_rows], dtype=float)
    cm_lowmid = np.asarray([float(row["cm_lowmid"]) for row in scan_network_rows], dtype=float)
    sev = np.asarray([float(row["row_severity"]) for row in scan_network_rows], dtype=float)
    scan_ids = [int(row["output_scan_index"]) for row in scan_network_rows
                if float(row["step_det_frac"]) >= 0.1 and float(row["step_alignment_frac"]) >= 0.5]
    top_slot = max(slot_rows, key=lambda row: float(row["event_score"])) if slot_rows else None

    valid_imp = np.isfinite(impulsive_scores)
    impulsive_frac_ge = float(np.mean(impulsive_scores[valid_imp] >= impulsive_threshold)) if valid_imp.any() else float("nan")
    return {
        "obsnum": obsnum,
        "array": array,
        "product_kind": product_kind,
        "source_file": str(nc_file),
        "n_network_rows": len(network_rows),
        "n_scan_network_rows": len(scan_network_rows),
        "n_slot_rows": len(slot_rows),
        "max_step_det_frac": _nanmax(step_det),
        "max_step_alignment_frac": _nanmax(step_align),
        "max_cm_lowmid": _nanmax(cm_lowmid),
        "mean_cm_lowmid": _nanmean(cm_lowmid),
        "max_row_severity": _nanmax(sev),
        "mean_row_severity": _nanmean(sev),
        "impulsive_frac_ge_threshold": impulsive_frac_ge,
        "max_impulsive_event_score": _nanmax(impulsive_scores),
        "masked_network_scans": int(np.count_nonzero(mask_flags != 0)),
        "masked_fraction_sum": float(np.nansum(mask_fracs)),
        "masked_fraction_mean": _nanmean(mask_fracs[mask_flags != 0]) if np.count_nonzero(mask_flags != 0) else float("nan"),
        "impulsive_masked_network_scans": int(np.count_nonzero(impulsive_mask_flags != 0)),
        "impulsive_masked_fraction_sum": float(np.nansum(impulsive_mask_fracs)),
        "impulsive_masked_fraction_mean": _nanmean(impulsive_mask_fracs[impulsive_mask_flags != 0]) if np.count_nonzero(impulsive_mask_flags != 0) else float("nan"),
        "n_scans_step_active": len(set(scan_ids)),
        "top_slot_event_score": float(top_slot["event_score"]) if top_slot else float("nan"),
        "top_slot_network": int(top_slot["network"]) if top_slot else -2147483647,
        "top_slot_output_scan": int(top_slot["output_scan_index"]) if top_slot else -2147483647,
    }


def _make_network_summary(obs_network_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    networks = sorted(set(int(row["network"]) for row in obs_network_rows))
    summary: list[dict[str, object]] = []
    for nw in networks:
        rr = [row for row in obs_network_rows if int(row["network"]) == nw]
        top = max(rr, key=lambda row: float(row["max_row_severity"]))
        summary.append(
            {
                "network": nw,
                "n_obsnums": len(rr),
                "median_max_step_det_frac": _nanmedian([float(row["max_step_det_frac"]) for row in rr]),
                "max_max_step_det_frac": _nanmax([float(row["max_step_det_frac"]) for row in rr]),
                "median_mean_cm_lowmid": _nanmedian([float(row["mean_cm_lowmid"]) for row in rr]),
                "max_max_cm_lowmid": _nanmax([float(row["max_cm_lowmid"]) for row in rr]),
                "median_impulsive_frac_ge_threshold": _nanmedian([float(row["impulsive_frac_ge_threshold"]) for row in rr]),
                "max_impulsive_frac_ge_threshold": _nanmax([float(row["impulsive_frac_ge_threshold"]) for row in rr]),
                "total_masked_network_scans": int(sum(int(row["masked_scans"]) for row in rr)),
                "obsnums_with_masked_scans": int(sum(int(row["masked_scans"]) > 0 for row in rr)),
                "total_impulsive_masked_network_scans": int(sum(int(row.get("impulsive_masked_scans", 0)) for row in rr)),
                "obsnums_with_impulsive_masked_scans": int(sum(int(row.get("impulsive_masked_scans", 0)) > 0 for row in rr)),
                "max_slot_event_score": _nanmax([float(row["top_slot_event_score"]) for row in rr]),
                "max_row_severity": _nanmax([float(row["max_row_severity"]) for row in rr]),
                "worst_obsnum": top["obsnum"],
            }
        )
    return summary


def _write_report(
    outpath: Path,
    redu_dir: Path,
    array: str,
    networks: list[int],
    obs_rows: list[dict[str, object]],
    top_scan_rows: list[dict[str, object]],
    top_slot_rows: list[dict[str, object]],
    n_rtcdiag: int,
    n_rtc_fallback: int,
) -> None:
    worst_obs = sorted(obs_rows, key=lambda row: float(row["max_row_severity"]), reverse=True)[:10]
    lines = [
        f"# RTCDiag Survey Report: {redu_dir.name}",
        "",
        f"- Reduction directory: `{redu_dir}`",
        f"- Array selection: `{array}`",
        f"- Networks: `{','.join(str(nw) for nw in networks)}`",
        f"- Obsnums summarized: {len(obs_rows)}",
        f"- Products used: `{n_rtcdiag}` rtcdiag, `{n_rtc_fallback}` rtc_timestream fallback",
        "",
        "## Worst Obsnums",
        "",
        "| obsnum | product | max severity | max step frac | max align | max low/mid | impulsive frac>=thr | step-mask nw-scans | imp-mask nw-scans | top slot |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in worst_obs:
        lines.append(
            "| {obsnum} | {product_kind} | {max_row_severity:.2f} | {max_step_det_frac:.3f} | "
            "{max_step_alignment_frac:.3f} | {max_cm_lowmid:.2f} | {impulsive_frac_ge_threshold:.3f} | "
            "{masked_network_scans} | {impulsive_masked_network_scans} | {top_slot_event_score:.2f} |".format(**row)
        )

    lines.extend(["", "## Top Scan-Network Rows", ""])
    if not top_scan_rows:
        lines.append("- none")
    else:
        for row in top_scan_rows[:12]:
            lines.append(
                "- obsnum={obsnum} scan={output_scan_index} nw={network} severity={row_severity:.2f} "
                "step_frac={step_det_frac:.3f} align={step_alignment_frac:.3f} "
                "imp_frac={network_impulsive_det_frac:.3f} imp_align={network_impulsive_alignment_frac:.3f} "
                "cm_lowmid={cm_lowmid:.2f} max_imp={max_impulsive_event_score:.2f} "
                "top_slot={max_slot_event_score:.2f} step_mask={step_mask_applied} "
                "imp_mask={impulsive_mask_applied}".format(**row)
            )

    lines.extend(["", "## Top Impulsive Slots", ""])
    if not top_slot_rows:
        lines.append("- none")
    else:
        for row in top_slot_rows[:12]:
            lines.append(
                "- obsnum={obsnum} scan={output_scan_index} nw={network} slot={slot} uid={apt_uid} "
                "kind={event_kind_label} score={event_score:.2f} peak={peak_abs_z:.2f} "
                "dpeak={peak_delta_abs_z:.2f}".format(**row)
            )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redu-dir", required=True, help="Reduction directory, e.g. .../reduced/redu37")
    ap.add_argument("--array", default="a1100", choices=["a1100", "a1400", "a2000"])
    ap.add_argument("--networks", default="all", help="Comma list or 'all'")
    ap.add_argument("--obsnums", default="all", help="Comma list or 'all'")
    ap.add_argument("--impulsive-threshold", type=float, default=6.0)
    ap.add_argument("--step-frac-ref", type=float, default=0.10)
    ap.add_argument("--alignment-ref", type=float, default=0.50)
    ap.add_argument("--cm-lowmid-ref", type=float, default=5.0)
    ap.add_argument("--top-n", type=int, default=100, help="Number of ranked rows/events to keep in CSV outputs")
    ap.add_argument("--outdir", default=None, help="Default: <redu-dir>/rtcdiag_survey_report")
    args = ap.parse_args()

    redu_dir = Path(args.redu_dir).expanduser().resolve()
    if not redu_dir.is_dir():
        raise NotADirectoryError(redu_dir)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (redu_dir / "rtcdiag_survey_report")
    outdir.mkdir(parents=True, exist_ok=True)

    obsnum_filter = None if args.obsnums == "all" else set(_parse_int_list(args.obsnums))
    products = _collect_products(redu_dir)
    if obsnum_filter is not None:
        products = [(obs, kind, path) for obs, kind, path in products if int(obs) in obsnum_filter]
    if not products:
        raise FileNotFoundError(f"no rtcdiag or rtc_timestream products found under {redu_dir}")

    array_id = ARRAY_NAME_TO_ID[args.array]
    obs_rows: list[dict[str, object]] = []
    obs_network_rows: list[dict[str, object]] = []
    scan_network_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    n_rtcdiag = 0
    n_rtc_fallback = 0
    selected_networks_global: set[int] = set()

    for obsnum, product_kind, nc_file in products:
        if product_kind == "rtcdiag":
            n_rtcdiag += 1
        else:
            n_rtc_fallback += 1

        with netCDF4.Dataset(nc_file) as ds:
            apt_array = _filled(ds.variables["apt_array"], fill=np.nan)
            apt_nw = _filled(ds.variables["apt_nw"], fill=np.nan)
            apt_uid = _filled(ds.variables["apt_uid"], fill=np.nan)
            det_array = np.rint(apt_array).astype(int)
            det_network = np.rint(apt_nw).astype(int)
            det_keep = np.isfinite(apt_array) & np.isfinite(apt_nw) & (det_array == array_id)
            available_networks = det_network[det_keep]
            if available_networks.size == 0:
                continue
            networks = _parse_networks(args.networks, available_networks)
            selected_networks_global.update(networks)

            diag_networks = _filled(ds.variables["rtc_diag_network_ids"], fill=np.iinfo(np.int32).min).astype(int)
            diag_indices = [i for i, nw in enumerate(diag_networks) if int(nw) in networks]
            if not diag_indices:
                continue

            output_scans = _filled(ds.variables["output_scan_index"], fill=np.iinfo(np.int32).min).astype(int)
            step_det = _filled(ds.variables["rtc_network_step_det_frac"], fill=np.nan)
            step_align = _filled(ds.variables["rtc_network_step_alignment_frac"], fill=np.nan)
            imp_score_max_nw = (
                _filled(ds.variables["rtc_network_impulsive_score_max"], fill=np.nan)
                if "rtc_network_impulsive_score_max" in ds.variables
                else np.full_like(step_det, np.nan, dtype=float)
            )
            imp_det_nw = (
                _filled(ds.variables["rtc_network_impulsive_det_frac"], fill=np.nan)
                if "rtc_network_impulsive_det_frac" in ds.variables
                else np.full_like(step_det, np.nan, dtype=float)
            )
            imp_align_nw = (
                _filled(ds.variables["rtc_network_impulsive_alignment_frac"], fill=np.nan)
                if "rtc_network_impulsive_alignment_frac" in ds.variables
                else np.full_like(step_det, np.nan, dtype=float)
            )
            imp_sample_nw = (
                _filled(ds.variables["rtc_network_impulsive_dominant_sample"], fill=np.iinfo(np.int32).min).astype(int)
                if "rtc_network_impulsive_dominant_sample" in ds.variables
                else np.full(step_det.shape, np.iinfo(np.int32).min, dtype=int)
            )
            cm_lowmid = _filled(ds.variables["rtc_network_cm_low_mid_ratio"], fill=np.nan)
            step_mask_applied = _filled(ds.variables["rtc_network_step_mask_applied"], fill=0).astype(int)
            step_mask_flagfrac = _filled(ds.variables["rtc_network_step_mask_flagged_fraction"], fill=np.nan)
            impulsive_mask_applied = (
                _filled(ds.variables["rtc_network_impulsive_mask_applied"], fill=0).astype(int)
                if "rtc_network_impulsive_mask_applied" in ds.variables
                else np.zeros_like(step_mask_applied, dtype=int)
            )
            impulsive_mask_flagfrac = (
                _filled(ds.variables["rtc_network_impulsive_mask_flagged_fraction"], fill=np.nan)
                if "rtc_network_impulsive_mask_flagged_fraction" in ds.variables
                else np.full_like(step_mask_flagfrac, np.nan, dtype=float)
            )
            impulsive = _filled(ds.variables["rtc_impulsive_event_score"], fill=np.nan)

            slot_score = _filled(ds.variables["rtc_impulsive_slot_event_score"], fill=np.nan) if "rtc_impulsive_slot_event_score" in ds.variables else None
            slot_kind = _filled(ds.variables["rtc_impulsive_slot_event_kind"], fill=np.iinfo(np.int32).min).astype(int) if "rtc_impulsive_slot_event_kind" in ds.variables else None
            slot_det = _filled(ds.variables["rtc_impulsive_slot_det_index"], fill=np.iinfo(np.int32).min).astype(int) if "rtc_impulsive_slot_det_index" in ds.variables else None
            slot_peak = _filled(ds.variables["rtc_impulsive_slot_peak_abs_z"], fill=np.nan) if "rtc_impulsive_slot_peak_abs_z" in ds.variables else None
            slot_dpeak = _filled(ds.variables["rtc_impulsive_slot_peak_delta_abs_z"], fill=np.nan) if "rtc_impulsive_slot_peak_delta_abs_z" in ds.variables else None

            n_scans = int(output_scans.size)
            for diag_idx in diag_indices:
                nw = int(diag_networks[diag_idx])
                nw_det_idx = np.where(det_keep & (det_network == nw))[0]
                nw_imp = impulsive[:, nw_det_idx] if nw_det_idx.size else np.empty((n_scans, 0))
                scan_max_imp = np.nanmax(nw_imp, axis=1) if nw_imp.size else np.full(n_scans, np.nan)
                scan_imp_frac_ge = np.mean(nw_imp >= args.impulsive_threshold, axis=1) if nw_imp.size else np.full(n_scans, np.nan)

                max_slot_score_by_scan = np.full(n_scans, np.nan)
                if slot_score is not None:
                    slot_plane = np.asarray(slot_score[:, diag_idx, :], dtype=float)
                    have_finite = np.isfinite(slot_plane).any(axis=1)
                    if np.any(have_finite):
                        max_slot_score_by_scan[have_finite] = np.nanmax(slot_plane[have_finite], axis=1)

                for scan in range(n_scans):
                    row = {
                        "obsnum": obsnum,
                        "array": args.array,
                        "product_kind": product_kind,
                        "source_file": str(nc_file),
                        "scan": int(scan),
                        "output_scan_index": int(output_scans[scan]),
                        "network": nw,
                        "step_det_frac": float(step_det[scan, diag_idx]),
                        "step_alignment_frac": float(step_align[scan, diag_idx]),
                        "network_impulsive_score_max": float(imp_score_max_nw[scan, diag_idx]),
                        "network_impulsive_det_frac": float(imp_det_nw[scan, diag_idx]),
                        "network_impulsive_alignment_frac": float(imp_align_nw[scan, diag_idx]),
                        "network_impulsive_dominant_sample": int(imp_sample_nw[scan, diag_idx]),
                        "cm_lowmid": float(cm_lowmid[scan, diag_idx]),
                        "step_mask_applied": int(step_mask_applied[scan, diag_idx]),
                        "step_mask_flagged_fraction": float(step_mask_flagfrac[scan, diag_idx]),
                        "impulsive_mask_applied": int(impulsive_mask_applied[scan, diag_idx]),
                        "impulsive_mask_flagged_fraction": float(impulsive_mask_flagfrac[scan, diag_idx]),
                        "max_impulsive_event_score": float(scan_max_imp[scan]),
                        "impulsive_frac_ge_threshold": float(scan_imp_frac_ge[scan]),
                        "max_slot_event_score": float(max_slot_score_by_scan[scan]),
                    }
                    row["row_severity"] = _severity_score(
                        row["step_det_frac"],
                        row["step_alignment_frac"],
                        row["cm_lowmid"],
                        row["max_impulsive_event_score"],
                        row["max_slot_event_score"],
                        args.step_frac_ref,
                        args.alignment_ref,
                        args.cm_lowmid_ref,
                        args.impulsive_threshold,
                    )
                    scan_network_rows.append(row)

                nw_rows = [row for row in scan_network_rows if row["obsnum"] == obsnum and int(row["network"]) == nw]
                top_slot_nw = max((row for row in nw_rows if np.isfinite(float(row["max_slot_event_score"]))),
                                  key=lambda row: float(row["max_slot_event_score"]),
                                  default=None)
                obs_network_rows.append(
                    {
                        "obsnum": obsnum,
                        "array": args.array,
                        "product_kind": product_kind,
                        "network": nw,
                        "n_scans": n_scans,
                        "max_step_det_frac": _nanmax([float(row["step_det_frac"]) for row in nw_rows]),
                        "mean_step_det_frac": _nanmean([float(row["step_det_frac"]) for row in nw_rows]),
                        "max_step_alignment_frac": _nanmax([float(row["step_alignment_frac"]) for row in nw_rows]),
                        "max_network_impulsive_score": _nanmax([float(row["network_impulsive_score_max"]) for row in nw_rows]),
                        "max_network_impulsive_det_frac": _nanmax([float(row["network_impulsive_det_frac"]) for row in nw_rows]),
                        "max_network_impulsive_alignment_frac": _nanmax([float(row["network_impulsive_alignment_frac"]) for row in nw_rows]),
                        "max_cm_lowmid": _nanmax([float(row["cm_lowmid"]) for row in nw_rows]),
                        "mean_cm_lowmid": _nanmean([float(row["cm_lowmid"]) for row in nw_rows]),
                        "max_impulsive_event_score": _nanmax(scan_max_imp),
                        "impulsive_frac_ge_threshold": float(np.mean(nw_imp[np.isfinite(nw_imp)] >= args.impulsive_threshold))
                            if nw_imp.size and np.isfinite(nw_imp).any() else float("nan"),
                        "masked_scans": int(np.count_nonzero(step_mask_applied[:, diag_idx] != 0)),
                        "masked_fraction_sum": float(np.nansum(step_mask_flagfrac[:, diag_idx])),
                        "impulsive_masked_scans": int(np.count_nonzero(impulsive_mask_applied[:, diag_idx] != 0)),
                        "impulsive_masked_fraction_sum": float(np.nansum(impulsive_mask_flagfrac[:, diag_idx])),
                        "max_row_severity": _nanmax([float(row["row_severity"]) for row in nw_rows]),
                        "top_slot_event_score": float(top_slot_nw["max_slot_event_score"]) if top_slot_nw else float("nan"),
                    }
                )

                if slot_score is not None and slot_det is not None and slot_kind is not None:
                    n_slots = slot_score.shape[2]
                    for scan in range(n_scans):
                        for slot in range(n_slots):
                            det_index = int(slot_det[scan, diag_idx, slot])
                            if det_index < 0 or det_index >= apt_uid.size:
                                continue
                            if int(round(float(apt_array[det_index]))) != array_id:
                                continue
                            slot_rows.append(
                                {
                                    "obsnum": obsnum,
                                    "array": args.array,
                                    "product_kind": product_kind,
                                    "source_file": str(nc_file),
                                    "scan": int(scan),
                                    "output_scan_index": int(output_scans[scan]),
                                    "network": nw,
                                    "slot": int(slot),
                                    "det_index": det_index,
                                    "apt_uid": int(round(float(apt_uid[det_index]))),
                                    "event_kind": int(slot_kind[scan, diag_idx, slot]),
                                    "event_kind_label": EVENT_KIND_LABELS.get(int(slot_kind[scan, diag_idx, slot]), "unknown"),
                                    "event_score": float(slot_score[scan, diag_idx, slot]),
                                    "peak_abs_z": float(slot_peak[scan, diag_idx, slot]) if slot_peak is not None else float("nan"),
                                    "peak_delta_abs_z": float(slot_dpeak[scan, diag_idx, slot]) if slot_dpeak is not None else float("nan"),
                                }
                            )

            obs_rows.append(
                _build_obs_summary(
                    obsnum,
                    product_kind,
                    nc_file,
                    args.array,
                    [row for row in obs_network_rows if row["obsnum"] == obsnum],
                    [row for row in scan_network_rows if row["obsnum"] == obsnum],
                    [row for row in slot_rows if row["obsnum"] == obsnum],
                    impulsive[:, det_keep],
                    args.impulsive_threshold,
                )
            )

    if not obs_rows:
        raise RuntimeError(f"no usable RTC diagnostic products found for array={args.array}")

    selected_networks = sorted(selected_networks_global)
    by_network_rows = _make_network_summary(obs_network_rows)
    top_scan_rows = sorted(
        scan_network_rows,
        key=lambda row: float(row["row_severity"]) if np.isfinite(float(row["row_severity"])) else -np.inf,
        reverse=True,
    )[: max(1, int(args.top_n))]
    top_slot_rows = sorted(
        [row for row in slot_rows if np.isfinite(float(row["event_score"]))],
        key=lambda row: float(row["event_score"]),
        reverse=True,
    )[: max(1, int(args.top_n))]

    _write_csv(outdir / "rtcdiag_survey_by_obsnum.csv", obs_rows)
    _write_csv(outdir / "rtcdiag_survey_by_obsnum_network.csv", obs_network_rows)
    _write_csv(outdir / "rtcdiag_survey_by_network.csv", by_network_rows)
    _write_csv(outdir / "rtcdiag_survey_top_scan_network_rows.csv", top_scan_rows)
    _write_csv(outdir / "rtcdiag_survey_top_impulsive_slots.csv", top_slot_rows if top_slot_rows else [{"note": "no slot rows found"}])
    _write_report(
        outdir / "RTCDIAG_SURVEY_REPORT.md",
        redu_dir,
        args.array,
        selected_networks,
        obs_rows,
        top_scan_rows,
        top_slot_rows,
        n_rtcdiag,
        n_rtc_fallback,
    )

    print(f"Wrote {outdir / 'rtcdiag_survey_by_obsnum.csv'}")
    print(f"Wrote {outdir / 'rtcdiag_survey_by_obsnum_network.csv'}")
    print(f"Wrote {outdir / 'rtcdiag_survey_by_network.csv'}")
    print(f"Wrote {outdir / 'rtcdiag_survey_top_scan_network_rows.csv'}")
    print(f"Wrote {outdir / 'rtcdiag_survey_top_impulsive_slots.csv'}")
    print(f"Wrote {outdir / 'RTCDIAG_SURVEY_REPORT.md'}")


if __name__ == "__main__":
    main()
