#!/usr/bin/env python3
"""Summarize live RTC line-audit outputs persisted in rtcdiag files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np


FILL_INT = -2147483647


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cluster_rows(rows: list[dict[str, object]], key: str, tol_hz: float) -> list[list[dict[str, object]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: float(row[key]))
    clusters: list[list[dict[str, object]]] = [[rows[0]]]
    for row in rows[1:]:
        if abs(float(row[key]) - float(clusters[-1][-1][key])) <= tol_hz:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _obsnum_to_rtcdiag_files(redu_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(redu_dir.glob("*/raw/*_rtcdiag.nc")):
        obsnum = path.name.split("_science_")[-1].split("_rtcdiag.nc")[0]
        mapping[str(obsnum)] = path
    return mapping


def _collect_runtime_rows(redu_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shared_rows: list[dict[str, object]] = []
    bad_rows: list[dict[str, object]] = []
    for obsnum, path in _obsnum_to_rtcdiag_files(redu_dir).items():
        with netCDF4.Dataset(path) as ds:
            output_scan_index = np.asarray(ds.variables["output_scan_index"][:], dtype=int)
            nw_ids = np.asarray(ds.variables["rtc_diag_network_ids"][:], dtype=int)

            shared_freq = np.asarray(ds.variables["rtc_network_line_audit_shared_freq_hz"][:], dtype=float)
            shared_det_count = np.asarray(ds.variables["rtc_network_line_audit_shared_detector_count"][:], dtype=int)
            shared_det_frac = np.asarray(ds.variables["rtc_network_line_audit_shared_detector_frac"][:], dtype=float)
            shared_med_prom = np.asarray(ds.variables["rtc_network_line_audit_shared_median_prominence"][:], dtype=float)
            shared_max_prom = np.asarray(ds.variables["rtc_network_line_audit_shared_max_prominence"][:], dtype=float)
            shared_width = np.asarray(ds.variables["rtc_network_line_audit_shared_width_hz"][:], dtype=float)
            shared_power_frac = np.asarray(ds.variables["rtc_network_line_audit_shared_line_power_frac"][:], dtype=float)
            shared_cm_freq = np.asarray(ds.variables["rtc_network_line_audit_shared_common_mode_freq_hz"][:], dtype=float)
            shared_cm_prom = np.asarray(ds.variables["rtc_network_line_audit_shared_common_mode_prominence"][:], dtype=float)
            shared_score = np.asarray(ds.variables["rtc_network_line_audit_shared_notch_score"][:], dtype=float)
            shared_rec = np.asarray(ds.variables["rtc_network_line_audit_shared_recommend_notch"][:], dtype=int)

            det_uid = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_uid"][:], dtype=int)
            det_freq = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_freq_hz"][:], dtype=float)
            det_prom = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_prominence"][:], dtype=float)
            det_power_frac = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_line_power_frac"][:], dtype=float)
            det_cluster_frac = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_cluster_detector_frac"][:], dtype=float)
            det_rec = np.asarray(ds.variables["rtc_network_line_audit_detector_candidate_recommend_flag"][:], dtype=int)

        for iscan, scan_idx in enumerate(output_scan_index):
            for inw, nw in enumerate(nw_ids):
                if shared_rec[iscan, inw] == 1 and np.isfinite(shared_freq[iscan, inw]):
                    shared_rows.append(
                        {
                            "obsnum": str(obsnum),
                            "output_scan_index": int(scan_idx),
                            "network": int(nw),
                            "freq_hz": float(shared_freq[iscan, inw]),
                            "detector_count": int(shared_det_count[iscan, inw]),
                            "detector_frac": float(shared_det_frac[iscan, inw]),
                            "median_prominence": float(shared_med_prom[iscan, inw]),
                            "max_prominence": float(shared_max_prom[iscan, inw]),
                            "width_hz": float(shared_width[iscan, inw]),
                            "line_power_frac": float(shared_power_frac[iscan, inw]),
                            "common_mode_freq_hz": float(shared_cm_freq[iscan, inw]),
                            "common_mode_prominence": float(shared_cm_prom[iscan, inw]),
                            "notch_score": float(shared_score[iscan, inw]),
                        }
                    )
                if (
                    det_rec[iscan, inw] == 1
                    and det_uid[iscan, inw] != FILL_INT
                    and np.isfinite(det_freq[iscan, inw])
                ):
                    bad_rows.append(
                        {
                            "obsnum": str(obsnum),
                            "output_scan_index": int(scan_idx),
                            "network": int(nw),
                            "uid": int(det_uid[iscan, inw]),
                            "freq_hz": float(det_freq[iscan, inw]),
                            "prominence": float(det_prom[iscan, inw]),
                            "line_power_frac": float(det_power_frac[iscan, inw]),
                            "cluster_detector_frac": float(det_cluster_frac[iscan, inw]),
                        }
                    )
    return shared_rows, bad_rows


def _summarize_shared(rows: list[dict[str, object]], tol_hz: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cluster in _cluster_rows(rows, "freq_hz", tol_hz):
        freqs = np.asarray([float(r["freq_hz"]) for r in cluster], dtype=float)
        det_frac = np.asarray([float(r["detector_frac"]) for r in cluster], dtype=float)
        med_prom = np.asarray([float(r["median_prominence"]) for r in cluster], dtype=float)
        cm_prom = np.asarray([float(r["common_mode_prominence"]) for r in cluster], dtype=float)
        scores = np.asarray([float(r["notch_score"]) for r in cluster], dtype=float)
        rep = max(
            cluster,
            key=lambda r: (
                float(r["notch_score"]),
                float(r["common_mode_prominence"]) if np.isfinite(float(r["common_mode_prominence"])) else -1.0,
                float(r["detector_frac"]),
            ),
        )
        out.append(
            {
                "family_freq_hz": float(np.median(freqs)),
                "n_rows": int(len(cluster)),
                "n_obsnums": int(len({str(r["obsnum"]) for r in cluster})),
                "n_networks": int(len({int(r["network"]) for r in cluster})),
                "obsnums": ",".join(sorted({str(r["obsnum"]) for r in cluster})),
                "median_detector_frac": float(np.median(det_frac)),
                "max_detector_frac": float(np.max(det_frac)),
                "median_prominence": float(np.median(med_prom)),
                "max_common_mode_prominence": float(np.nanmax(cm_prom)),
                "score_sum": float(np.sum(scores)),
                "representative_obsnum": str(rep["obsnum"]),
                "representative_output_scan_index": int(rep["output_scan_index"]),
                "representative_network": int(rep["network"]),
            }
        )
    out.sort(key=lambda row: (-int(row["n_rows"]), -float(row["score_sum"]), float(row["family_freq_hz"])))
    return out


def _summarize_bad_detectors(rows: list[dict[str, object]], tol_hz: float) -> list[dict[str, object]]:
    by_uid: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_uid[int(row["uid"])].append(row)

    out: list[dict[str, object]] = []
    for uid, uid_rows in by_uid.items():
        for cluster in _cluster_rows(uid_rows, "freq_hz", tol_hz):
            freqs = np.asarray([float(r["freq_hz"]) for r in cluster], dtype=float)
            prom = np.asarray([float(r["prominence"]) for r in cluster], dtype=float)
            power = np.asarray([float(r["line_power_frac"]) for r in cluster], dtype=float)
            clfrac = np.asarray([float(r["cluster_detector_frac"]) for r in cluster], dtype=float)
            rep = max(cluster, key=lambda r: float(r["prominence"]))
            out.append(
                {
                    "uid": int(uid),
                    "network": int(rep["network"]),
                    "family_freq_hz": float(np.median(freqs)),
                    "n_rows": int(len(cluster)),
                    "n_obsnums": int(len({str(r["obsnum"]) for r in cluster})),
                    "obsnums": ",".join(sorted({str(r["obsnum"]) for r in cluster})),
                    "median_prominence": float(np.median(prom)),
                    "max_prominence": float(np.max(prom)),
                    "median_line_power_frac": float(np.median(power)),
                    "median_cluster_detector_frac": float(np.median(clfrac)),
                    "representative_obsnum": str(rep["obsnum"]),
                    "representative_output_scan_index": int(rep["output_scan_index"]),
                }
            )
    out.sort(key=lambda row: (-int(row["n_rows"]), -float(row["median_prominence"]), int(row["uid"])))
    return out


def _write_markdown(
    outpath: Path,
    *,
    redu_dir: Path,
    family_rows: list[dict[str, object]],
    bad_rows: list[dict[str, object]],
    tol_hz: float,
) -> None:
    lines = [
        "# RTC Runtime Line Summary",
        "",
        f"Source reduction: `{redu_dir}`",
        f"Frequency-family clustering tolerance: `{tol_hz:.3f} Hz`",
        "",
        "This report summarizes the live RTC `line_audit` results persisted in the",
        "`rtcdiag` files. It is based on the runtime recommendation flags, not the",
        "offline Python PSD audit.",
        "",
        "## Top Shared Line Families",
        "",
        "| family freq [Hz] | rows | obsnums | networks | median det frac | max det frac | median prom | max cm prom | score sum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in family_rows[:15]:
        lines.append(
            f"| {float(row['family_freq_hz']):.3f} | {int(row['n_rows'])} | {int(row['n_obsnums'])} | "
            f"{int(row['n_networks'])} | {float(row['median_detector_frac']):.3f} | "
            f"{float(row['max_detector_frac']):.3f} | {float(row['median_prominence']):.1f} | "
            f"{float(row['max_common_mode_prominence']):.1f} | {float(row['score_sum']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Top Bad-Detector Candidates",
            "",
            "| uid | nw | freq [Hz] | rows | obsnums | median prom | max prom | median line power frac | median cluster det frac |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bad_rows[:20]:
        lines.append(
            f"| {int(row['uid'])} | {int(row['network'])} | {float(row['family_freq_hz']):.3f} | "
            f"{int(row['n_rows'])} | {int(row['n_obsnums'])} | {float(row['median_prominence']):.1f} | "
            f"{float(row['max_prominence']):.1f} | {float(row['median_line_power_frac']):.3f} | "
            f"{float(row['median_cluster_detector_frac']):.3f} |"
        )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redu-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--family-tol-hz", type=float, default=0.5)
    args = parser.parse_args()

    redu_dir = args.redu_dir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve() if args.outdir is not None else (redu_dir / "rtc_runtime_line_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    shared_rows, bad_rows = _collect_runtime_rows(redu_dir)
    family_rows = _summarize_shared(shared_rows, float(args.family_tol_hz))
    bad_summary = _summarize_bad_detectors(bad_rows, float(args.family_tol_hz))

    _write_csv(outdir / "rtc_runtime_shared_families.csv", family_rows)
    _write_csv(outdir / "rtc_runtime_bad_detectors.csv", bad_summary)
    _write_markdown(
        outdir / "RTC_RUNTIME_LINE_SUMMARY.md",
        redu_dir=redu_dir,
        family_rows=family_rows,
        bad_rows=bad_summary,
        tol_hz=float(args.family_tol_hz),
    )

    print(f"Wrote {outdir / 'rtc_runtime_shared_families.csv'}")
    print(f"Wrote {outdir / 'rtc_runtime_bad_detectors.csv'}")
    print(f"Wrote {outdir / 'RTC_RUNTIME_LINE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
