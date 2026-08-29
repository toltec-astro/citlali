#!/usr/bin/env python3
"""Verify the exact SCI-NOI Stage A author packet and closure bindings."""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "AUTHOR_PACKET_MANIFEST.md"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(spec: str) -> bytes:
    return subprocess.check_output(["git", "show", spec], cwd=ROOT)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def verify_external(name: str) -> None:
    binding_name = f"{pathlib.Path(name).stem}.sha256"
    record = (ROOT / binding_name).read_text(encoding="utf-8").strip()
    expected, recorded_name = record.split(maxsplit=1)
    if recorded_name != name:
        fail(f"external binding names {recorded_name!r}, expected {name!r}")
    actual = digest((ROOT / name).read_bytes())
    if actual != expected:
        fail(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| \d+ \| `([^`]+)`(?:, `[^`]+`)? \| `([0-9a-f]{64})` \|",
        text,
        flags=re.MULTILINE,
    )
    if len(rows) != 17:
        fail(f"manifest has {len(rows)} parsed objects, expected 17")

    for source, expected in rows:
        if ":" in source and not (ROOT / source).exists():
            data = git_bytes(source)
        else:
            path = ROOT / source
            if not path.is_file():
                fail(f"missing manifest object: {source}")
            data = path.read_bytes()
        actual = digest(data)
        if actual != expected:
            fail(f"{source}: expected {expected}, got {actual}")

    verify_external("AUTHOR_PACKET_MANIFEST.md")
    verify_external("BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.md")

    package_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.glob("*.md")
        if path.name != "STAGE_A_CHANGE_LOG.md"
    )
    required = [
        "NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1",
        "NOI-GEN/PTC-TO-FROZEN-JINC-CONDITIONAL-SIGN@1",
        "NOI-GEN/REALIZED-MAP-CONDITIONAL-SIGN@1",
        "NOI-GEN/REALIZED-JINC-CONDITIONAL-SIGN@1",
        "NOI-GEN/FIXED-FLT-CONDITIONAL-SIGN@1",
        "SCI-NOI:uncertainty_member_admission@1",
        "unit(empirical_scale_standardized_signal) = 1",
        "SCI-NOI-ODQ-102A",
        "SCI-NOI-ODQ-102B",
        "SCI-NOI-ODQ-102C",
        "SCI-NOI-ODQ-102D",
        "materialized randomized timestream",
        "not an ordinary MAP science product",
        "throughout the observation",
        "network-stratified",
        "cross-network",
        "marginal sign probability `1/2`",
        "approved only as a suggestion",
        "implementation-blind scientific-contract author",
        "scientifically consequential adjacent reduction state",
        "does not require exhaustive implementation provenance",
        "randomization is intended to suppress source signal",
        "does not, by construction alone, establish",
        "nonbinding terminology suggestion",
        "Candidate assignments rejected during finite-design construction",
        "fails the complete GEN ensemble",
        "survivor or partial design",
        "V_hat_cond(p) = sum_b omega_b M_b(p)^2",
        "finite ensemble mean is not subtracted",
        "no `B-1` correction",
        "common all-member domain",
        "not covariance merely because it is pointwise or diagonal-like",
        "A retained ensemble is a representation",
        "Unreported off-diagonal entries",
        "not zero or independent",
        "No representation implies invertibility",
        "NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE",
        "W_hat_cond(p) = 1 / V_hat_cond(p)",
        "It shall not substitute a numerical zero",
        "not inverse variance or precision",
        "cross-boundary use requires explicit scientific authority",
    ]
    for token in required:
        if token not in package_text:
            fail(f"required closure token absent: {token}")
    if "SCI-NOI:realization_member_completion@1" in package_text:
        fail("superseded completion-profile identity remains outside change log")

    for path in ROOT.glob("*.md"):
        body = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)", body):
            if "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                fail(f"broken local link in {path.name}: {target}")

    print(
        "PASS: 17/17 author objects, external bindings, closure invariants, "
        "and local links"
    )


if __name__ == "__main__":
    main()
