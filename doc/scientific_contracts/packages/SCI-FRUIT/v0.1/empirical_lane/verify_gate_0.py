#!/usr/bin/env python3
"""Verify the manifest-bounded SCI-FRUIT empirical-lane Gate-0 record."""

from __future__ import annotations

import hashlib
import pathlib
import subprocess


HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = HERE.parents[5]
ACCEPTED_PACKET_COMMIT = "90e55fa7d04fcab5cc716f7d258032275d6f6d7d"
ACCEPTED_PACKET_TREE = "9ed88eb5cb87dad955870e65f436878b0713482d"
EXPECTED_BRANCH = "codex/sci-fruit-v0.1-empirical-lane"
MANIFEST = HERE / "EMPIRICAL_LANE_BUNDLE_MANIFEST_R0.1.md"
ARCHIVE = REPO / "SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz"
APPROVED_BYTES = {
    MANIFEST: (
        2738,
        "4072f41eaa876c2e143de93d0d7292fddccc1f25de2c11763eafd24c473a28ce",
    ),
    ARCHIVE: (
        11602,
        "761ba278a53e32ad1d5d3977230cc4f1f90f257056b547508342797f584167ed",
    ),
}
REQUIRED_RECORDS = {
    "SCIENTIFIC_OWNER_GATE_0_APPROVAL_2026-09-01.md": (
        "SCI-FRUIT-EL-G0-REGISTRATION-PREPARATION-R0.1",
        "Gate 0 approved against the exact r0.1 manifest",
        "Every later gate remains subject to a separate exact owner decision",
    ),
    "EL_G0_EXECUTION_PREFLIGHT_2026-09-01.md": (
        EXPECTED_BRANCH,
        ACCEPTED_PACKET_COMMIT,
        "no Unity action authorized",
    ),
    "EL_G0_ACCESS_INCIDENT_001_2026-09-01.md": (
        "SCI-FRUIT-EL-G0-ACCESS-INCIDENT-001-2026-09-01",
        "validation/fruit_loop*",
        "ineligible for untouched",
        "qualification evidence",
    ),
    "EL_G0_HISTORICAL_CONTROL_FEASIBILITY_R0.1.md": (
        "automatically the historical executable",
        "`v1.x`",
        "Pinned Reconstruction",
    ),
    "EL_G0_PROFILE_AND_METRIC_REGISTRATION_R0.1.md": (
        "compact_high_snr_response_recovery",
        "extended_low_snr_mode_recovery",
        "no profile, metric",
    ),
    "EL_G0_POPULATION_ACCESS_AND_LINEAGE_PLAN_R0.1.md": (
        "opaque",
        "qualification",
        "lineage",
    ),
    "EL_G0_CANDIDATE_FAMILY_AND_RESOURCE_FRAME_R0.1.md": (
        "Complete-product replacement family",
        "Relaxed complete-product transition family",
        "Response-aware residual-correction family",
    ),
    "EL_G0_DATA_AND_CONTROL_INVENTORY_REQUEST_R0.1.md": (
        "non-outcome",
        "human",
        "Unity",
    ),
    "EL_G0_GATE_D_READINESS_R0.1.md": (
        "Gate-D launch not ready",
        "Gate 0 remains active",
        "no prototype",
    ),
}
ALLOWED_CHANGED_PATHS = {
    "SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz",
    "SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz",
    "doc/REFACTOR_STATUS.md",
    "doc/scientific_contracts/INDEX.md",
    "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/README.md",
    "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/DECISION_LOG.md",
    "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/CROSSWALK.md",
    "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/verify_stage_a.py",
}
EMPIRICAL_PREFIX = (
    "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/"
)


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_paths() -> set[str]:
    paths = set(
        filter(
            None,
            run(
                "git", "diff", "--name-only", ACCEPTED_PACKET_COMMIT
            ).splitlines(),
        )
    )
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO, text=True
    )
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def verify_identity() -> None:
    tree = run("git", "rev-parse", f"{ACCEPTED_PACKET_COMMIT}^{{tree}}")
    if tree != ACCEPTED_PACKET_TREE:
        fail(f"accepted packet tree changed: {tree}")
    if subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ACCEPTED_PACKET_COMMIT,
            "HEAD",
        ],
        cwd=REPO,
    ):
        fail("HEAD does not descend from the accepted packet commit")
    branch = run("git", "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        fail(f"Gate 0 must run on {EXPECTED_BRANCH}, not {branch}")


def verify_approved_bytes() -> None:
    for path, (expected_size, expected_hash) in APPROVED_BYTES.items():
        if not path.is_file():
            fail(f"approved object missing: {path.relative_to(REPO)}")
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != expected_size or actual_hash != expected_hash:
            fail(
                "approved bytes changed: "
                f"{path.relative_to(REPO)} size={actual_size} sha256={actual_hash}"
            )
    committed_manifest = subprocess.check_output(
        [
            "git",
            "show",
            f"{ACCEPTED_PACKET_COMMIT}:"
            "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/"
            "EMPIRICAL_LANE_BUNDLE_MANIFEST_R0.1.md",
        ],
        cwd=REPO,
    )
    if committed_manifest != MANIFEST.read_bytes():
        fail("working manifest differs from the accepted packet commit")


def verify_records() -> None:
    for relative, required_tokens in REQUIRED_RECORDS.items():
        path = HERE / relative
        if not path.is_file():
            fail(f"missing Gate-0 record: {relative}")
        text = path.read_text(encoding="utf-8")
        for token in required_tokens:
            if token not in text:
                fail(f"{relative} is missing required text: {token}")


def verify_edit_scope() -> None:
    unexpected = sorted(
        path
        for path in changed_paths()
        if not path.startswith(EMPIRICAL_PREFIX)
        and path not in ALLOWED_CHANGED_PATHS
    )
    if unexpected:
        fail("Gate-0 scope includes protected changes: " + ", ".join(unexpected))
    validation_changes = sorted(
        path for path in changed_paths() if path.startswith("validation/")
    )
    if validation_changes:
        fail("Gate 0 changed quarantined validation paths: " + ", ".join(validation_changes))


def main() -> None:
    verify_identity()
    verify_approved_bytes()
    verify_records()
    verify_edit_scope()
    subprocess.check_call(["git", "diff", "--check"], cwd=REPO)
    print("PASS: SCI-FRUIT empirical-lane Gate-0 record verified")


if __name__ == "__main__":
    main()
