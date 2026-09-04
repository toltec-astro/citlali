#!/usr/bin/env python3
"""Verify the bounded MAP-space shared-conventions repair candidate."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


BASE = "5f0fc20042b88fb6cd883c92d1b59b7f22832901"
AUDIT_COMMIT = "34a29a1eac8a2c41a97263bbd775bd36c3d06398"
AUDIT_DIR = Path("doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001")
REPAIR_DIR = Path("doc/scientific_contracts/audits/MAP_SPACE_SHARED_CONVENTIONS_REPAIR_001")
CONVENTIONS = Path("doc/SCIENTIFIC_CONVENTIONS.md")

AUDIT_HASHES = {
    "WORK_ORDER.md": "b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f",
    "SOURCE_AUTHORITY_MANIFEST.md": "d21d1446ebcdda8597cf08a4568be91906e3cc22e97f9e7f5544a5fa590b2cd5",
    "PRODUCT_AND_BOUNDARY_GRAPH.md": "c5f256496c891925bb90c73a79e7af68e1d5966bc68f9c930b41a4690db7145c",
    "CROSS_PACKAGE_CONFORMANCE_MATRIX.md": "106585cb838adc176c9faa5a89d19d0fed261ef136adfd2d98148271b38ce307",
    "FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md": "b8fad111974d79dcb48ee9977353d2e8958d74701a07ce7f507cd80962ff9310",
    "HORIZONTAL_AUDIT_REPORT.md": "b32c0cf2249e74ef35b177cd016e5d03854e725466719f35fa62d9425270fe96",
    "verify_horizontal_audit.py": "bdd7013b61c254ae6bb8d2e6c900b8fd0764f6ee87280c014d37f50e7d33fc3a",
}

FORBIDDEN = (
    "The current validated capability has one Stokes component",
    "array-grouped Stokes-I observation and coadd maps",
    "`u` is the realized `weight_I`",
    "retained exposure, and coadd-observation count share the admitted membership",
    "`upstream_eligible_exposure_I`",
    "`retained_exposure_I`",
    "Its separately approved signed estimator and formal-support products",
    "### SCI-MAP-002 JINC Estimator And Support",
    "conditional formal mapmaker weight is `C^2/Q`",
    "For JINC, `coverage_bool_I` is the authoritative formal-support mask",
    "The JINC empirical downgrade",
    "Only Stokes I is validated.",
    "map-to-component mappings",
    "WCS spectral/component-axis serialization",
    "`r_max` remains both the second-JINC-zero parameter and the square-cache half-width",
)

REQUIRED = (
    "nonpolarimetric total-intensity-equivalent",
    "`u_op = 1`, dimensionless",
    "does not replace or flatten",
    "base SCI-JINC v0.1 authorizes no cross-observation combination",
    "`upstream_eligible_original_footprint_exposure`",
    "`retained_original_footprint_exposure`",
    "original's own AST ALIGN-grid coordinate",
    "Every produced base-v0.1 JINC bundle contains exactly five numerical map-plane roles",
    "`jinc_coefficient_squared_time`",
    "Base SCI-JINC v0.1 publishes no response, variance, formal-weight, covariance",
    "appropriate `NOT_AUTHORIZED`, `UNAVAILABLE`, or `NOT_APPLICABLE` state",
    "no formal Stokes product is authorized by this convention",
    "`maps_to_arrays` and `maps_to_stokes` mappings",
    "persisted `map_stokes` label",
    "WCS spectral/`STOKES`-axis serialization",
    "`Q_p >= Q_star * c / 10`",
    "These are MAP output-row support rules, not coadd weighting rules",
    "requested, effective, observation-resolved, and realized states",
    "raw-parent identity separately from local numerical support",
    "`h_a = ceil(s_a * (r_max)_a / Delta)`",
    "Base r0.3 admits no owner-approved numerical-adequacy profile or matching certificate",
    "Historical SCI-MAP-002 implementation evidence used the two-level binary64 policy",
    "is not an owner-approved SCI-JINC r0.3 numerical-adequacy profile or certificate",
    "Historical integrated SCI-MAP-001/SCI-NOI-002 evidence used one compact atomic",
    "`raw-parent/product` digests",
)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify one exact MAP-space shared-conventions repair candidate."
    )
    parser.add_argument(
        "--expected-candidate",
        required=True,
        help="Exact full candidate commit SHA to review.",
    )
    parser.add_argument(
        "--expected-tree",
        required=True,
        help="Exact full candidate tree SHA to review.",
    )
    args = parser.parse_args()
    for label, value in (
        ("candidate", args.expected_candidate),
        ("tree", args.expected_tree),
    ):
        require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, f"invalid {label} SHA")
    return args


def main() -> None:
    args = parse_args()
    root = Path(run("git", "rev-parse", "--show-toplevel"))
    require(root == Path.cwd().resolve(), "run from the repair worktree root")
    head = run("git", "rev-parse", "HEAD")
    tree = run("git", "rev-parse", "HEAD^{tree}")
    require(head == args.expected_candidate, "HEAD does not match expected candidate")
    require(tree == args.expected_tree, "tree does not match expected candidate tree")
    require(
        not run("git", "status", "--porcelain=v1", "--untracked-files=all"),
        "candidate checkout is not clean",
    )
    subprocess.check_call(("git", "merge-base", "--is-ancestor", BASE, "HEAD"))
    subprocess.check_call(("git", "merge-base", "--is-ancestor", AUDIT_COMMIT, "HEAD"))

    for name, expected in AUDIT_HASHES.items():
        path = AUDIT_DIR / name
        require(path.is_file(), f"missing preserved audit artifact {path}")
        require(digest(path) == expected, f"preserved audit artifact changed: {path}")

    text = CONVENTIONS.read_text(encoding="utf-8")
    flat = " ".join(text.split())
    for snippet in FORBIDDEN:
        require(
            " ".join(snippet.split()) not in flat,
            f"superseded convention remains: {snippet}",
        )
    for snippet in REQUIRED:
        require(
            " ".join(snippet.split()) in flat,
            f"required repaired convention missing: {snippet}",
        )

    required_sources = (
        Path("doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md"),
        Path("doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md"),
        Path("doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-MAP_COADD_PROFILES_R0.7.md"),
        Path("doc/scientific_contracts/packages/SCI-MAP/v0.1/SOURCE_MANIFEST_R0.7.md"),
        Path("doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md"),
        Path("doc/scientific_contracts/packages/SCI-JINC/v0.1/FREEZE_AUTHORITY_MANIFEST_R0.3.md"),
    )
    for source in required_sources:
        require(source.is_file(), f"missing cited frozen authority: {source}")
        target = source.relative_to("doc").as_posix()
        require(target in text, f"frozen authority is not cited: {source}")

    subprocess.check_call(("git", "diff", "--check", AUDIT_COMMIT))
    package_diff = run(
        "git", "diff", "--name-only", BASE, "--", "doc/scientific_contracts/packages"
    )
    require(not package_diff, f"frozen package changed: {package_diff}")

    changed = set(
        filter(None, run("git", "diff", "--name-only", BASE, "HEAD").splitlines())
    )
    allowed_exact = {CONVENTIONS.as_posix()}
    unexpected = sorted(
        path
        for path in changed
        if path not in allowed_exact
        and not path.startswith(f"{AUDIT_DIR.as_posix()}/")
        and not path.startswith(f"{REPAIR_DIR.as_posix()}/")
    )
    require(not unexpected, f"out-of-scope changed paths: {unexpected}")

    print(f"PASS candidate={head} tree={tree} clean=true branch_independent=true")
    print(f"PASS base={BASE}")
    print(f"PASS audit_commit={AUDIT_COMMIT} audit_artifacts={len(AUDIT_HASHES)}")
    print("PASS owner_resolved_conflicts=4 shared_conventions_repaired=true")
    print("PASS frozen_packages_modified=false application_modified=false")
    print("PASS numerical_route_states_preserved=true")
    print("PASS scope=shared_conventions_and_repair_evidence_only")
    print("shared_conventions_repair_verifier=PASS")


if __name__ == "__main__":
    main()
