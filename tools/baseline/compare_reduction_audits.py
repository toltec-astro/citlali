#!/usr/bin/env python3
"""Compare two Citlali reduction run audits without reading large arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__:
    from . import audit_reduction_run
else:
    import audit_reduction_run


def audit_for(
    path: str, expected_mode: str, expected_label: str, top: int,
    require_processed_provenance: bool = False,
    require_raw_provenance: bool = False,
    require_mapmaking_provenance: bool = False,
    require_coadd_provenance: bool = False,
    require_noise_products_provenance: bool = False,
    require_pointing_provenance: bool = False,
) -> dict[str, Any]:
    args = argparse.Namespace(
        reduction=path,
        expected_mode=expected_mode,
        expected_label=expected_label,
        top=top,
        require_processed_provenance=require_processed_provenance,
        require_raw_provenance=require_raw_provenance,
        require_mapmaking_provenance=require_mapmaking_provenance,
        require_coadd_provenance=require_coadd_provenance,
        require_noise_products_provenance=(
            require_noise_products_provenance
        ),
        require_pointing_provenance=require_pointing_provenance,
    )
    return audit_reduction_run.build_audit(args)


def log_finished(audit: dict[str, Any]) -> bool:
    return "done" in audit.get("log", {}).get("markers", {})


def serious_issue_counts(audit: dict[str, Any]) -> dict[str, int]:
    """Return every log severity that invalidates a successful reduction."""
    counts = audit.get("log", {}).get("issue_counts", {})
    return {
        key: int(counts.get(key, 0))
        for key in ("error", "fatal", "critical", "traceback")
        if int(counts.get(key, 0))
    }


def status_ok(audit: dict[str, Any]) -> bool:
    checks = [
        audit.get("mode_ok") is not False,
        audit.get("label_ok") is not False,
        log_finished(audit),
        not serious_issue_counts(audit),
        audit_reduction_run.provenance_ok(audit),
    ]
    return all(checks)


def seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt_seconds(value: Any) -> str:
    number = seconds(value)
    return "" if number is None else f"{number:.3f}"


def fmt_delta(value: Any) -> str:
    number = seconds(value)
    if number is None:
        return ""
    return f"{number:+.3f}"


def fmt_ratio(candidate: Any, baseline: Any) -> str:
    cand = seconds(candidate)
    base = seconds(baseline)
    if cand is None or base is None or base == 0:
        return ""
    return f"{cand / base:.4f}"


def markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    out = []
    for index, row in enumerate(rows):
        out.append("| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) + " |")
        if index == 0:
            out.append("| " + " | ".join("-" * widths[col] for col in range(len(row))) + " |")
    return "\n".join(out)


def compare_audits(args: argparse.Namespace) -> dict[str, Any]:
    baseline = audit_for(
        args.baseline, args.expected_mode, args.baseline_label, args.top
    )
    candidate = audit_for(
        args.candidate,
        args.expected_mode,
        args.candidate_label,
        args.top,
        require_processed_provenance=getattr(
            args, "require_candidate_processed_provenance", False
        ),
        require_raw_provenance=getattr(
            args, "require_candidate_raw_provenance", False
        ),
        require_mapmaking_provenance=getattr(
            args, "require_candidate_mapmaking_provenance", False
        ),
        require_coadd_provenance=getattr(
            args, "require_candidate_coadd_provenance", False
        ),
        require_noise_products_provenance=getattr(
            args, "require_candidate_noise_products_provenance", False
        ),
        require_pointing_provenance=getattr(
            args, "require_candidate_pointing_provenance", False
        ),
    )
    base_intervals = baseline.get("log", {}).get("interval_seconds", {})
    cand_intervals = candidate.get("log", {}).get("interval_seconds", {})
    timing = []
    for name in sorted(set(base_intervals) | set(cand_intervals)):
        base = seconds(base_intervals.get(name))
        cand = seconds(cand_intervals.get(name))
        timing.append(
            {
                "interval": name,
                "baseline_seconds": base,
                "candidate_seconds": cand,
                "delta_seconds": None if base is None or cand is None else cand - base,
                "ratio": None if base in (None, 0) or cand is None else cand / base,
            }
        )
    return {
        "expected_mode": args.expected_mode,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_ok": status_ok(baseline),
        "candidate_ok": status_ok(candidate),
        "product_counts_match": baseline.get("products", {}).get("stable_counts_by_kind")
        == candidate.get("products", {}).get("stable_counts_by_kind"),
        "comparable_count_match": baseline.get("products", {}).get("stable_comparable_count")
        == candidate.get("products", {}).get("stable_comparable_count"),
        "timing": timing,
    }


def render_identity(audit: dict[str, Any]) -> str:
    labels = audit.get("labels", [])
    label_text = ", ".join(f"{item.get('mode')}/{item.get('label')}" for item in labels) or "none"
    return (
        f"`{audit.get('reduction')}`<br>"
        f"labels: `{label_text}`<br>"
        f"mode OK: `{audit.get('mode_ok')}` label OK: `{audit.get('label_ok')}`<br>"
        f"finished: `{log_finished(audit)}` serious issues: `{serious_issue_counts(audit)}`"
    )


def render_provenance(record: dict[str, Any]) -> str:
    if not record.get("present"):
        return f"absent; required={record.get('required', False)}"
    digest = audit_reduction_run.provenance_hash_summary(record)
    return (
        f"valid={record.get('valid')} schema={record.get('schema_version', '')} "
        f"sha256={digest}"
    )


def render_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    candidate = result["candidate"]
    lines = [
        "# Citlali Reduction Audit Comparison",
        "",
        f"- Expected mode: `{result['expected_mode']}`",
        f"- Baseline label: `{result['baseline_label']}`",
        f"- Candidate label: `{result['candidate_label']}`",
        f"- Baseline OK: `{result['baseline_ok']}`",
        f"- Candidate OK: `{result['candidate_ok']}`",
        f"- Stable product kind counts match: `{result['product_counts_match']}`",
        f"- Stable comparable product counts match: `{result['comparable_count_match']}`",
        "",
        "## Identity",
        "",
        markdown_table(
            [
                ["side", "audit"],
                ["baseline", render_identity(baseline)],
                ["candidate", render_identity(candidate)],
            ]
        ),
        "",
        "## Products",
        "",
        markdown_table(
            [
                ["metric", "baseline", "candidate"],
                [
                    "file_count",
                    str(baseline.get("products", {}).get("file_count", "")),
                    str(candidate.get("products", {}).get("file_count", "")),
                ],
                [
                    "comparable_count",
                    str(baseline.get("products", {}).get("comparable_count", "")),
                    str(candidate.get("products", {}).get("comparable_count", "")),
                ],
                [
                    "stable_comparable_count",
                    str(baseline.get("products", {}).get("stable_comparable_count", "")),
                    str(candidate.get("products", {}).get("stable_comparable_count", "")),
                ],
                [
                    "counts_by_kind",
                    str(baseline.get("products", {}).get("counts_by_kind", "")),
                    str(candidate.get("products", {}).get("counts_by_kind", "")),
                ],
                [
                    "stable_counts_by_kind",
                    str(baseline.get("products", {}).get("stable_counts_by_kind", "")),
                    str(candidate.get("products", {}).get("stable_counts_by_kind", "")),
                ],
                [
                    "profile_sidecars",
                    str(baseline.get("products", {}).get("profile_sidecars", "")),
                    str(candidate.get("products", {}).get("profile_sidecars", "")),
                ],
            ]
        ),
        "",
        "## Provenance",
        "",
        markdown_table(
            [
                ["sidecar", "baseline", "candidate"],
                *[
                    [
                        name,
                        render_provenance(
                            baseline.get("provenance", {}).get(name, {})
                        ),
                        render_provenance(
                            candidate.get("provenance", {}).get(name, {})
                        ),
                    ]
                    for name in sorted(
                        set(baseline.get("provenance", {}))
                        | set(candidate.get("provenance", {}))
                    )
                ],
            ]
        ),
        "",
        "## Timing",
        "",
    ]
    timing_rows = [["interval", "baseline s", "candidate s", "delta s", "ratio"]]
    for row in result["timing"]:
        timing_rows.append(
            [
                str(row["interval"]),
                fmt_seconds(row["baseline_seconds"]),
                fmt_seconds(row["candidate_seconds"]),
                fmt_delta(row["delta_seconds"]),
                fmt_ratio(row["candidate_seconds"], row["baseline_seconds"]),
            ]
        )
    lines.append(markdown_table(timing_rows))
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline reduNN directory or reduced root.")
    parser.add_argument("candidate", help="Candidate reduNN directory or reduced root.")
    parser.add_argument("--expected-mode", default="", help="Expected validation mode, e.g. beammap.")
    parser.add_argument("--baseline-label", default="citlali")
    parser.add_argument("--candidate-label", default="refactor")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--require-candidate-processed-provenance",
        action="store_true",
        help="Require a valid processed provenance sidecar only for the candidate.",
    )
    parser.add_argument(
        "--require-candidate-raw-provenance",
        action="store_true",
        help="Require valid raw provenance for every candidate observation.",
    )
    parser.add_argument(
        "--require-candidate-mapmaking-provenance",
        action="store_true",
        help="Require valid mapmaking provenance only for the candidate.",
    )
    parser.add_argument(
        "--require-candidate-coadd-provenance",
        action="store_true",
        help="Require valid coadd provenance only for the candidate.",
    )
    parser.add_argument(
        "--require-candidate-noise-products-provenance",
        action="store_true",
        help="Require valid noise-products provenance only for the candidate.",
    )
    parser.add_argument(
        "--require-candidate-pointing-provenance",
        action="store_true",
        help="Require valid pointing provenance only for the candidate.",
    )
    parser.add_argument("--json-out", default="", help="Optional path for machine-readable JSON.")
    parser.add_argument("--report-out", default="", help="Optional path for Markdown output.")
    return parser.parse_args(argv)


def write_text(path: str, text: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = compare_audits(args)
    report = render_markdown(result)
    if args.json_out:
        write_text(args.json_out, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.report_out:
        write_text(args.report_out, report)
    print(report, end="")
    if not result["baseline_ok"] or not result["candidate_ok"]:
        return 2
    if not result["product_counts_match"] or not result["comparable_count_match"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
