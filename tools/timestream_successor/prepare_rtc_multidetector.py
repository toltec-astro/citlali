#!/usr/bin/env python3
"""Prepare the fixed 152390 simultaneous RTC input; make no event selections."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from run_rtc_pipeline_replay import AFFECTED, BASELINE_MANIFEST, CONTROLS, digest

NO_MASK = "owner-152390-initial-no-spatial-mask-2026-09-15"
COMMON = ("raw", "tune", "manifest", "audit_receipt", "telescope",
          "effective_config", "ast_acceptance", "lowpass_identity", "fir")


def combine(configs):
    channels = sorted(AFFECTED + CONTROLS)
    if sorted(configs) != channels:
        raise ValueError("the fixed twelve-detector population must be complete")
    base = configs[AFFECTED[0]]
    out = {key: copy.deepcopy(base[key]) for key in COMMON}
    notch = next(t for t in base["trials"] if t["id"] == "w1-t3")
    if notch["requested_width_hz"] != 1 or notch["requested_span_seconds"] != 3:
        raise ValueError("working notch baseline changed")
    out.update(schema="rtc-multidetector-experiment-v1", observation=152390,
               network=12, source_protection_authority=NO_MASK,
               finite_notch_identity=notch["finite_notch_identity"],
               finite_notch=copy.deepcopy(notch["finite_notch"]), detectors=[])
    for channel in channels:
        c = configs[channel]
        if c["channel"] != channel or c["observation"] != 152390 or c["network"] != 12:
            raise ValueError("foreign detector/observation/network")
        if any(c[key] != base[key] for key in COMMON):
            raise ValueError("simultaneous projection requires one exact shared input/domain")
        if channel in AFFECTED:
            selected = next(t for t in c["trials"] if t["id"] == "w1-t3")
            if selected != notch:
                raise ValueError("affected detectors disagree on the saved notch design")
        out["detectors"].append(dict(channel=channel, samples=copy.deepcopy(c["samples"]),
                                     filter="w1-t3" if channel in AFFECTED else "lowpass"))
    return out


def prepare(prior, output):
    manifest = prior / "EVIDENCE_SHA256SUMS"
    if digest(manifest) != BASELINE_MANIFEST:
        raise ValueError("prior experiment manifest changed")
    sealed = {line[66:]: line[:64] for line in manifest.read_text().splitlines()}
    configs = {}
    for d in AFFECTED + CONTROLS:
        path = prior / "prepared-01" / f"{d}.json"
        if digest(path) != sealed[str(path.relative_to(prior))]:
            raise ValueError("prior configuration changed")
        configs[d] = json.loads(path.read_text())
    combined = combine(configs)
    combined["predecessor_manifest_sha256"] = BASELINE_MANIFEST
    for entry in [combined[k] for k in COMMON if isinstance(combined[k], dict)] + [d["samples"] for d in combined["detectors"]]:
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"input changed: {entry['path']}")
    with output.open("x") as stream:
        stream.write(json.dumps(combined, indent=2) + "\n")


def review_template(receipt):
    if receipt["schema"] != "rtc-multidetector-learning-v1" or receipt["Apply_performed"]:
        raise ValueError("review template requires an original learning receipt")
    # Empty support is explicitly unavailable. Never turn the full observation
    # or lack of detected boundaries into an asserted stable interval.
    return dict(schema="rtc-reviewed-selection-v1", approved=False, authority="",
                learning_binding=receipt["learning_binding"], VAL_generation=receipt["VAL_generation"],
                stable_support_authority="", contamination_authority="",
                detectors=[dict(channel=d["channel"], occurrence=d["occurrence"],
                                stable_segments=[], contaminated=[]) for d in receipt["detectors"]],
                events=[], existing_scans=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prior", type=Path)
    group.add_argument("--learning-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prior:
        prepare(args.prior, args.output)
    else:
        receipt = yaml.safe_load(args.learning_receipt.read_text())
        template = review_template(receipt)
        template["receipt_sha256"] = hashlib.sha256(args.learning_receipt.read_bytes()).hexdigest()
        with args.output.open("x") as stream:
            yaml.safe_dump(template, stream, sort_keys=False)
