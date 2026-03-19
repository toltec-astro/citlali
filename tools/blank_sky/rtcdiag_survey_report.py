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

import numpy as np

from blank_sky_null_audit import _write_csv
from rtcdiag_data import load_reduction_tables


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

    dataset = load_reduction_tables(
        redu_dir=redu_dir,
        array=args.array,
        networks_spec=args.networks,
        obsnums_spec=args.obsnums,
        impulsive_threshold=args.impulsive_threshold,
        step_frac_ref=args.step_frac_ref,
        alignment_ref=args.alignment_ref,
        cm_lowmid_ref=args.cm_lowmid_ref,
    )

    obs_rows = dataset["obs_rows"]
    obs_network_rows = dataset["obs_network_rows"]
    selected_networks = dataset["selected_networks"]
    by_network_rows = dataset["by_network_rows"]
    top_scan_rows = dataset["top_scan_rows"][: max(1, int(args.top_n))]
    top_slot_rows = dataset["top_slot_rows"][: max(1, int(args.top_n))]
    n_rtcdiag = int(dataset["n_rtcdiag"])
    n_rtc_fallback = int(dataset["n_rtc_fallback"])

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
