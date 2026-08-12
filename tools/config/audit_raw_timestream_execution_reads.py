#!/usr/bin/env python3
"""Classify external RTCProc accesses during raw-timestream migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


SOURCE_ROOTS = (
    "include/citlali/core/engine",
    "include/citlali/core/pipeline",
    "src/citlali/core",
)
EXCLUDED_FILES = {
    "include/citlali/core/engine/detail/rtc_config_impl.h",
    "include/citlali/core/pipeline/raw_tod_output_context.h",
    "include/citlali/core/pipeline/timestream_config_adapter_polarimetry.h",
    "include/citlali/core/pipeline/raw_timestream_authority.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_filtering.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_flagging.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_line_audit.h",
}
EXPECTED_RECORD_COUNT = 87
EXPECTED_RECORD_SHA256 = (
    "6f850cbdf06458db3c408080a933149a87d97874e79d2a6161f4ba1c5757e488"
)

EXECUTOR_OPERATIONS = {
    "append_diag_to_netcdf",
    "append_to_netcdf",
    "apply_filter_edge_guard",
    "apply_rtc_line_audit_detector_notches",
    "apply_rtc_line_audit_fixed_notches",
    "apply_rtc_line_audit_shared_notches",
    "begin_observation_applied_response_history",
    "begin_ptc_response_iteration",
    "calibration.admit_product",
    "calibration.calc_tau",
    "calibration.disable_extinction",
    "calibration.reset_product_admission",
    "calibration.setup",
    "capture_rtc_line_audit",
    "clear_cached_diagnostics",
    "consume_applied_response_notches",
    "count_rtc_line_audit_fixed_notches",
    "filter.make_filter",
    "filter.make_notch_filter",
    "kernel.clear_source_centers",
    "kernel.set_source_centers",
    "kernel.setup",
    "homogeneous_calibration_join",
    "record_finalized_calibration_join",
    "remove_bad_dets",
    "remove_flagged_dets",
    "remove_nearby_tones",
    "run",
    "reset_coherent_iq_mode_candidates",
    "snapshot_coherent_iq_mode_candidates",
    "snapshot_detector_diag_summary",
    "snapshot_source_protection_diag_summary",
}
OBSERVATION_STATE = {
    "calibration",
    "calibration.extinction_model",
    "calibration.tx_225_zenith",
    "despiker.fsmp",
    "despiker.source_protection_enabled",
    "despiker.source_protection_radius_arcsec",
    "downsampler.factor",
    "kernel.map_grouping",
}
OUTPUT_OR_REALIZED_STATE = {
    "calibration.calibration_quality_regime",
    "calibration.calibration_valid",
    "calibration.calibration_validity_reason",
    "calibration.effective_reference_spectral_index_alpha",
    "calibration.operator_contract_sha256",
    "calibration.operator_id",
    "calibration.operator_nodes_sha256",
    "calibration.passband_set_id",
    "calibration.product",
    "calibration.product.valid",
    "calibration.realized_tau225",
    "calibration.reduction_calibration_quality_regime",
    "calibration.reduction_maximum_tau225",
    "calibration.reference_profile_id",
    "calibration.reference_spectral_index_default_applied",
    "calibration.requested_reference_spectral_index_alpha",
    "despiker.local_residual",
    "filter_edge_guard",
    "filter_edge_guard.context_samples",
    "filter_edge_guard.guard_samples",
    "filter.a_gibbs",
    "filter.freq_high_Hz",
    "filter.freq_low_Hz",
    "filter.iir_highpass_freq_Hz",
    "filter.iir_highpass_order",
    "filter.iir_highpass_zero_phase",
    "filter.n_terms",
    "filter.notch_zero_phase",
    "filter.qs",
    "filter.qs.size",
    "filter.w0s",
    "filter.w0s.size",
    "kernel",
    "kernel.has_source_centers",
    "line_audit",
    "remove_bad_dets_window_sec",
    "run_tod_filter",
    "run_tod_iir_highpass",
    "run_tod_notch",
    "snapshot_applied_response_notches",
    "applied_response_history_available",
}
RAW_POLICY_READS = {"run_downsample", "run_extinction", "run_kernel"}


def strip_non_code(text: str) -> str:
    pattern = re.compile(
        r"^[ \t]*\#[^\n]*|//[^\n]*|/\*.*?\*/|"
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.MULTILINE | re.DOTALL,
    )
    return pattern.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def classify_access(chain: str) -> str:
    if chain == "run_polarization" or chain.startswith("polarization"):
        return "separate_polarimetry_domain"
    if chain in EXECUTOR_OPERATIONS:
        return "executor_operation"
    if chain in OBSERVATION_STATE:
        return "observation_state"
    if chain in OUTPUT_OR_REALIZED_STATE:
        return "output_or_realized_state"
    if chain in RAW_POLICY_READS:
        return "raw_policy_read"
    return "review_required"


def scan_accesses(repo_root: Path) -> list[dict[str, object]]:
    pattern = re.compile(r"\brtcproc\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
    occurrences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for source_root in SOURCE_ROOTS:
        root = repo_root / source_root
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".h", ".cpp"}:
                continue
            relative = str(path.relative_to(repo_root))
            if relative in EXCLUDED_FILES:
                continue
            text = strip_non_code(path.read_text())
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                occurrences[match.group(1)].append((relative, line))
    records = []
    for chain, sites in sorted(occurrences.items()):
        records.append(
            {
                "access": chain,
                "classification": classify_access(chain),
                "occurrence_count": len(sites),
                "files": sorted({path for path, _ in sites}),
            }
        )
    return records


def record_digest(records: list[dict[str, object]]) -> str:
    stable = [
        {
            "access": record["access"],
            "classification": record["classification"],
            "occurrence_count": record["occurrence_count"],
            "files": record["files"],
        }
        for record in records
    ]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--fail-on-review", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    records = scan_accesses(repo_root)
    digest = record_digest(records)
    review = [
        record for record in records
        if record["classification"] == "review_required"
    ]
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["classification"])] += 1
    drift = (
        len(records) != EXPECTED_RECORD_COUNT
        or digest != EXPECTED_RECORD_SHA256
    )
    result = {
        "schema_version": "citlali-raw-execution-read-census-v1",
        "record_count": len(records),
        "record_sha256": digest,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "expected_record_sha256": EXPECTED_RECORD_SHA256,
        "classification_counts": dict(sorted(counts.items())),
        "review_required_count": len(review),
        "drift": drift,
        "excluded_compatibility_boundaries": sorted(EXCLUDED_FILES),
        "records": records,
        "note": (
            "The named one-way adapter and legacy mirrors are compatibility "
            "boundaries, not external consumers. Numerical method calls may "
            "remain on RTCProc. Temporary raw policy reads are limited to "
            "typed/legacy shadow comparisons. Other raw policy reads, "
            "observation-state mutation, and output/realized state must move "
            "to explicit typed plans or metadata contracts. Polarimetry is a "
            "separate authority domain."
        ),
    }
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"| `{record['access']}` | {record['classification']} | "
            f"{record['occurrence_count']} |"
            for record in records
        )
        output.write_text(
            "# Raw Timestream External Execution Read Census\n\n"
            f"- Records: `{len(records)}`\n"
            f"- Digest: `{digest}`\n"
            f"- Review required: `{len(review)}`\n"
            f"- Drift: `{drift}`\n\n"
            "| Access | Classification | Occurrences |\n"
            "| --- | --- | ---: |\n"
            f"{rows}\n"
        )
    print(
        "raw execution read census: "
        f"records={len(records)} digest={digest} "
        f"review_required={len(review)} drift={drift} "
        f"classifications={dict(sorted(counts.items()))}"
    )
    if args.fail_on_review and review:
        return 1
    if args.fail_on_drift and drift:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
