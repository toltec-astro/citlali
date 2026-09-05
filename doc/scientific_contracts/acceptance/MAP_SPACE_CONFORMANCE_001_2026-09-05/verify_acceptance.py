#!/usr/bin/env python3
"""Read-only verification of the exact MAP-space documentation integration.

The accepted packet and its historical branch-specific verifier remain
unchanged. This adapter verifies integration ancestry/scope, then reuses the
accepted verifier's content checks. It makes no scientific or approval decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = "ae953ed4d87d1f693d2bbf42aebbc25ef730c771"
BASE_TREE = "37ee17cf001ceb2c193fbea5e2b5ae3d147ba4a1"
SUBJECT = "f36cb788dc1ce99dad9c30bcd4442dc8bfbf681e"
SUBJECT_TREE = "c12a3cb915cd900331661fc1f9e3fdf2899a197c"
SOURCE_BASE = "9f42d348298d76c5d5145aaf0c3eace1f3e154c1"
PACKET = "doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/"
RECORD = "doc/scientific_contracts/acceptance/MAP_SPACE_CONFORMANCE_001_2026-09-05/"
BRANCH = "refs/heads/codex/integrate-map-space-conformance-001-2026-09-05"
MANAGER_PATHS = {
    "doc/REFACTOR_STATUS.md",
    "doc/INTEGRATION_LEDGER.md",
    "doc/scientific_contracts/INDEX.md",
    "doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md",
    *(RECORD + name for name in (
        "README.md", "SOURCE_STUDY_REVIEW.md", "REVIEW_SUBJECT_MANIFEST.json",
        "verify_acceptance.py",
    )),
}
EVIDENCE_DIGESTS = {
    "SOURCE_STUDY_REVIEW.md": "94640880cd91fd183313cd37648840b2b92a183d9413b62d5921640bab71dc0f",
    "REVIEW_SUBJECT_MANIFEST.json": "bccd998efe7ef32a67c7e6a96994c9eb1945a4fafd3f53bd42463e2802d3b235",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=REPO,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


def git(*args: str) -> str:
    return git_bytes(*args).decode().strip()


def verify_packet_bytes() -> set[str]:
    for name, digest in EVIDENCE_DIGESTS.items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest,
                f"preserved review evidence changed: {name}")
    report_lines = (HERE / "SOURCE_STUDY_REVIEW.md").read_bytes().splitlines(keepends=True)
    original_report = b"".join(
        line[:-1] + b"  \n" if number in {41, 50, 53} else line
        for number, line in enumerate(report_lines, 1)
    )
    require(hashlib.sha256(original_report).hexdigest() ==
            "8030650c0944695210c40505bc888db185787588ec877c81d26ac2666d23017a",
            "original review report cannot be recovered exactly")
    manifest = json.loads((HERE / "REVIEW_SUBJECT_MANIFEST.json").read_text())
    require(manifest["candidate"] == SUBJECT and manifest["tree"] == SUBJECT_TREE,
            "review subject identity mismatch")
    paths = set(git("ls-tree", "-r", "--name-only", SUBJECT, "--", PACKET).splitlines())
    require(len(paths) == 13 and paths == set(manifest["packet_sha256"]),
            "accepted packet manifest must contain exactly 13 artifacts")
    actual = {str(p.relative_to(REPO)) for p in (REPO / PACKET).rglob("*") if p.is_file()}
    require(actual == paths, "integrated packet artifact set changed")
    for path in paths:
        current = REPO / path
        require(current.is_file() and not current.is_symlink(), f"nonregular packet path: {path}")
        data = current.read_bytes()
        require(data == git_bytes("show", f"{SUBJECT}:{path}"),
                f"accepted packet bytes changed: {path}")
        require(hashlib.sha256(data).hexdigest() == manifest["packet_sha256"][path],
                f"accepted packet digest mismatch: {path}")
    return paths


def verify_scope(packet_paths: set[str]) -> str:
    require(git("rev-parse", f"{BASE}^{{tree}}") == BASE_TREE, "canonical base tree mismatch")
    require(git("rev-parse", f"{SUBJECT}^{{tree}}") == SUBJECT_TREE, "accepted study tree mismatch")
    require(git("merge-base", BASE, SUBJECT) == SOURCE_BASE, "unexpected merge base")
    for branch, commit in (
        ("codex/map-space-contract-to-implementation-conformance-001", "93c2b4591bb5d0cf8efe4491975c31e5f8fb5903"),
        ("codex/map-space-conformance-001-doc-repair-2026-09-04", "402b82bc7c38d8a3739d7803f46ccf3f1bbd90f8"),
        ("codex/map-space-conformance-001-review-repair-2026-09-04", SUBJECT),
    ):
        require(git("rev-parse", "refs/heads/" + branch) == commit,
                f"preserved study ref moved: {branch}")
    head = git("rev-parse", "HEAD")
    expected = packet_paths | MANAGER_PATHS
    require(len(expected) == 21, "expected integration scope is not 21 paths")
    status = git_bytes("status", "--porcelain=v1", "--untracked-files=all").decode().splitlines()
    if head == BASE:
        require(git("symbolic-ref", "HEAD") == BRANCH, "wrong integration preparation branch")
        require(git("rev-parse", "MERGE_HEAD") == SUBJECT, "wrong pending merge subject")
        require(all(line[:2] in {"A ", " M", "M ", "MM", "AM", "??"} for line in status),
                "unexpected pre-commit status or merge conflict")
        require({line[3:] for line in status} == expected, "unexpected pre-commit path set")
        untracked = {line[3:] for line in status if line.startswith("?? ")}
        changed = set(git("diff", "--name-only", BASE).splitlines()) | untracked
        mode = "pre_commit"
    else:
        require(git("show", "-s", "--format=%P", "HEAD").split() == [BASE, SUBJECT],
                "integration must have exact ordered canonical/study parents")
        require(not status, "committed integration checkout is dirty")
        changed = set(git("diff", "--name-only", f"{BASE}..HEAD").splitlines())
        mode = "post_commit"
    require(changed == expected, "integration changed paths differ from the exact scope")
    require(git("rev-parse", "refs/heads/codex/refactor-mainline") in {BASE, head},
            "canonical moved outside the bounded integration route")
    git("diff", "--check", BASE)
    git("diff", "--cached", "--check")
    return mode


def verify_decisions() -> None:
    text = (HERE / "README.md").read_text()
    rows = re.findall(r"^\| (CTI-OD-\d{3}) \| ([^|]+) \|$", text, re.M)
    require([row[0] for row in rows] == [f"CTI-OD-{i:03d}" for i in range(1, 8)],
            "manager decision IDs/order changed")
    require(rows[0][1].startswith("ACCEPTED / CLOSED"), "CTI-OD-001 acceptance missing")
    require(all(value.startswith("OPEN:") for _, value in rows[1:6]),
            "an unresolved owner decision was closed")
    require(rows[6][1].startswith("INHERITED_CLOSED:"), "CTI-OD-007 inherited status changed")
    for phrase in (SUBJECT, SUBJECT_TREE, "Program adherence and prior-work recovery",
                   "no active\napplication implementation unit", "Grant Wilson", "2026-09-05"):
        require(phrase in text, f"acceptance/preflight statement missing: {phrase}")
    for path in MANAGER_PATHS - {RECORD + name for name in (
            "README.md", "SOURCE_STUDY_REVIEW.md", "REVIEW_SUBJECT_MANIFEST.json", "verify_acceptance.py")}:
        require(SUBJECT in (REPO / path).read_text(), f"manager link lacks exact subject: {path}")


def main() -> int:
    try:
        paths = verify_packet_bytes()
        mode = verify_scope(paths)
        verify_decisions()
        # Reuse only the unchanged accepted content checks. Its historical
        # branch/single-parent scope gate is replaced by verify_scope above.
        study = runpy.run_path(str(REPO / PACKET / "verify_packet.py"))
        study["verify_artifacts"]()
        sources = study["verify_source_manifest"]()
        products, edges, traces = study["verify_stable_ids_and_states"]()
        study["verify_required_statements"]()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS mode={mode} head={git('rev-parse', 'HEAD')} tree={git('rev-parse', 'HEAD^{tree}')}")
    print("PASS exact_two_parent_route=true accepted_packet_files=13 byte_identical=true manager_paths=8")
    print(f"PASS admitted_sources={sources} products={products} edges={edges} traces={traces}")
    print("PASS CTI-OD-001=accepted_closed CTI-OD-002--006=open CTI-OD-007=inherited_closed")
    print("PASS application_science_config_tests_registry_validation=unchanged")
    print("PASS study_refs=preserved source_study_limitations=retained")
    print("map_space_source_study_acceptance=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
