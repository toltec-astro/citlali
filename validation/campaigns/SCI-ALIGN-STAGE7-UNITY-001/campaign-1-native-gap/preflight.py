#!/usr/bin/env python3
"""Validate and freeze the merged config for Stage 7 native-gap Campaign 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.config.tolteca_mode_kit import (
        ModeKitError,
        dump_yaml,
        extract_low_level,
        flatten_leaves,
        load_yaml,
        merge_files,
        numbered_yaml_files,
    )
except ModuleNotFoundError as error:  # pragma: no cover - command-line aid
    raise SystemExit(
        "run this preflight from the Citlali repository root"
    ) from error


SCHEMA_VERSION = "sci-align-stage7-native-gap-preflight-v1"
OVERLAY_NAME = "99_zzz_sci_align_stage7_native_gap.yaml"
LOW_LEVEL_PREFIX = "reduce.steps[0].config.low_level."

# These are the complete leaves that the campaign overlay is allowed to own.
# Values are checked both in the overlay and after the complete TolTECA merge.
EXPECTED_OVERLAY_LEAVES: dict[str, Any] = {
    "coadd.enabled": False,
    "mapmaking.method": "naive",
    "noise_maps.enabled": False,
    "post_processing.map_filtering.enabled": False,
    "post_processing.source_finding.enabled": False,
    "timestream.fruit_loops.enabled": False,
    "timestream.learning.enabled": False,
    "timestream.polarimetry.enabled": False,
    "timestream.raw_time_chunk.altaz_destripe.enabled": False,
    "timestream.raw_time_chunk.extinction_correction.enabled": False,
    "timestream.raw_time_chunk.filter.enabled": False,
    "timestream.raw_time_chunk.coherent_iq_mode_observer.enabled": False,
    "timestream.raw_time_chunk.flagging.impulsive_coincidence.enabled": False,
    "timestream.raw_time_chunk.flagging.lower_tod_inv_var_factor": 0.0,
    "timestream.raw_time_chunk.flagging.upper_tod_inv_var_factor": 0.0,
    "timestream.raw_time_chunk.kernel.enabled": False,
    "timestream.raw_time_chunk.line_audit.enabled": False,
    "timestream.raw_time_chunk.output.enabled": False,
    "timestream.processed_time_chunk.clean.mask_radius_arcsec": 0.0,
    "timestream.processed_time_chunk.flagging.lower_tod_inv_var_factor": 0.0,
    "timestream.processed_time_chunk.flagging.upper_tod_inv_var_factor": 0.0,
    "timestream.processed_time_chunk.flagging.second_pass_local.enabled": False,
    "timestream.processed_time_chunk.output.enabled": False,
    "timestream.processed_time_chunk.weighting.busy_row_suppression.enabled": False,
    "timestream.processed_time_chunk.weighting.source_mask_radius_arcsec": 0.0,
    "timestream.processed_time_chunk.weighting.validation.enabled": False,
}

# Supported operations deliberately retained so Campaign 1 exercises the
# native numerical route.  A later operator file is not allowed to hollow the
# campaign out or silently alter its scope.
EXPECTED_RETAINED_LEAVES: dict[str, Any] = {
    "runtime.reduction_type": "science",
    "runtime.interp_over_gaps": True,
    "mapmaking.enabled": True,
    "timestream.enabled": True,
    "timestream.raw_time_chunk.despike.enabled": True,
    "timestream.raw_time_chunk.flagging.impulsive_capture.enabled": True,
    "timestream.raw_time_chunk.flagging.network_step_mask.enabled": True,
    "timestream.raw_time_chunk.flux_calibration.enabled": True,
    "timestream.processed_time_chunk.clean.enabled": True,
    "timestream.processed_time_chunk.clean.grouping[0]": "nw",
    "timestream.processed_time_chunk.clean.standard_pca.enabled": True,
    "timestream.processed_time_chunk.weighting.type": "full",
}


class PreflightError(RuntimeError):
    """Raised when the deployed campaign does not match its frozen scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_campaign_files(mode_dir: Path, overlay: Path) -> list[Path]:
    files = numbered_yaml_files(mode_dir)
    resolved = {path.resolve(): path for path in files}
    if overlay.resolve() not in resolved:
        files.append(overlay)
    files.sort(key=lambda path: (int(path.name.split("_", 1)[0]), path.name))
    return files


def overlay_low_level_leaves(overlay: Path) -> dict[str, Any]:
    document = load_yaml(overlay) or {}
    if not isinstance(document, Mapping):
        raise PreflightError("campaign overlay is not a YAML mapping")
    try:
        low_level = extract_low_level(document)
    except ModeKitError as error:
        raise PreflightError(str(error)) from error
    return flatten_leaves(low_level)


def compare_exact(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> list[str]:
    errors: list[str] = []
    for path, value in expected.items():
        if path not in actual:
            errors.append(f"{label} missing {path}")
        elif actual[path] != value:
            errors.append(
                f"{label} {path}={actual[path]!r}; expected {value!r}"
            )
    return errors


def validate(mode_dir: Path, overlay: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mode_dir = mode_dir.expanduser().resolve(strict=True)
    overlay = overlay.expanduser().resolve(strict=True)
    if overlay.name != OVERLAY_NAME:
        raise PreflightError(
            f"overlay must retain the frozen filename {OVERLAY_NAME}"
        )

    files = ordered_campaign_files(mode_dir, overlay)
    if not files or files[-1].resolve() != overlay:
        final = files[-1].name if files else "<none>"
        raise PreflightError(
            f"campaign overlay is not the final numbered source; final source is {final}"
        )
    if sum(path.name == OVERLAY_NAME for path in files) != 1:
        raise PreflightError("campaign overlay must occur exactly once")

    overlay_leaves = overlay_low_level_leaves(overlay)
    errors = compare_exact(
        overlay_leaves, EXPECTED_OVERLAY_LEAVES, "campaign overlay"
    )
    unexpected = sorted(set(overlay_leaves) - set(EXPECTED_OVERLAY_LEAVES))
    if unexpected:
        errors.append("campaign overlay owns unexpected leaves: " + ", ".join(unexpected))

    try:
        merged, origins, changes = merge_files(files)
        low_level = extract_low_level(merged)
    except ModeKitError as error:
        raise PreflightError(str(error)) from error
    merged_leaves = flatten_leaves(low_level)
    errors.extend(
        compare_exact(merged_leaves, EXPECTED_OVERLAY_LEAVES, "merged config")
    )
    errors.extend(
        compare_exact(merged_leaves, EXPECTED_RETAINED_LEAVES, "merged config")
    )

    for path in EXPECTED_OVERLAY_LEAVES:
        origin = origins.get(LOW_LEVEL_PREFIX + path)
        if origin != OVERLAY_NAME:
            errors.append(
                f"merged config {path} origin is {origin!r}; expected {OVERLAY_NAME}"
            )

    if errors:
        raise PreflightError("\n".join(errors))

    file_records = [
        {
            "order": index,
            "name": path.name,
            "path": str(path),
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for index, path in enumerate(files)
    ]
    overlay_changes = [
        change for change in changes if change["source"] == OVERLAY_NAME
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "config_ready_observation_checks_pending",
        "campaign": "SCI-ALIGN-STAGE7-UNITY-001/campaign-1-native-gap",
        "mode_dir": str(mode_dir),
        "overlay": str(overlay),
        "numbered_sources": file_records,
        "merged_low_level_sha256": canonical_sha256(low_level),
        "merged_low_level_leaf_count": len(merged_leaves),
        "campaign_overlay_leaf_count": len(overlay_leaves),
        "campaign_overlay_changes": overlay_changes,
        "verified_config_conditions": {
            **EXPECTED_OVERLAY_LEAVES,
            **EXPECTED_RETAINED_LEAVES,
        },
        "pending_observation_conditions": [
            "matched-v2 root manifest is fresh and verifies against exact raw-source bytes",
            "native alignment and pointing carriers are complete and match the relation",
            "runtime duplicate_tone detector vector is exact and entirely zero",
            "realized scan science bounds equal loaded outer-context bounds for every scan",
        ],
    }
    return merged, report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode-dir",
        required=True,
        type=Path,
        help="TolTECA reduction directory containing its numbered YAML sources.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        help=(
            "Campaign overlay path. Defaults to MODE_DIR/"
            + OVERLAY_NAME
            + "."
        ),
    )
    parser.add_argument("--merged-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    overlay = args.overlay or args.mode_dir / OVERLAY_NAME
    try:
        merged, report = validate(args.mode_dir, overlay)
    except (OSError, PreflightError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2

    args.merged_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.merged_out.write_text(dump_yaml(merged), encoding="utf-8")
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "config preflight passed; observation-dependent admission checks remain pending"
    )
    print(f"merged config: {args.merged_out}")
    print(f"report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
