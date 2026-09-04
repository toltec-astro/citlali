#!/usr/bin/env python3
"""Verify the bounded MAP-space contract-to-implementation packet.

The verifier is read-only.  It validates the exact base identity, accepted
oracle and inspected-source bytes, artifact closure, stable product/edge/trace
IDs, closed state vocabulary, and packet-only mutation both before and after
the single candidate commit.  It performs no scientific interpretation.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


PACKET_DIR = Path(__file__).resolve().parent
REPO = PACKET_DIR.parents[3]
PACKET_PREFIX = (
    "doc/scientific_contracts/audits/"
    "MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/"
)
EXPECTED_BASE = "9f42d348298d76c5d5145aaf0c3eace1f3e154c1"
EXPECTED_BASE_TREE = "e51f22760c64454ce7233c45dd740aa710777bae"
EXPECTED_BRANCH = "refs/heads/codex/map-space-contract-to-implementation-conformance-001"
EXPECTED_MAINLINE = EXPECTED_BASE

REQUIRED_FILES = {
    "WORK_ORDER.md",
    "SOURCE_AUTHORITY_MANIFEST.md",
    "PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md",
    "ROUTE_AVAILABILITY_CLASSIFICATION.md",
    "FAILURE_MODE_SOURCE_AUDIT.md",
    "REPRESENTATIVE_TRACE_VALIDATION_PLAN.md",
    "PRIORITIZED_REPAIR_BACKLOG.md",
    "FRUIT_ATTACHMENT_ENVELOPE.md",
    "OOF_ATTACHMENT_ENVELOPE.md",
    "OWNER_DECISION_LEDGER.md",
    "FINAL_REPORT.md",
    "verify_packet.py",
}

PRODUCT_IDS = tuple(f"MSP-P{number:03d}" for number in range(1, 17)) + (
    "MSP-PX01",
)
EDGE_IDS = tuple(f"MSP-E{number:03d}" for number in range(1, 33))
TRACE_IDS = tuple(f"MSP-T{number:03d}" for number in range(1, 17))
STATES = {
    "IMPLEMENTED_CONFORMANT_AT_SOURCE_LEVEL",
    "IMPLEMENTED_LEGACY_SEMANTICS",
    "DECLARED_NOT_IMPLEMENTED",
    "UNAVAILABLE_BY_DESIGN",
    "MISSING_AUTHORITY",
    "MISSING_IMPLEMENTATION",
    "CONTRADICTORY",
    "NOT_APPLICABLE",
    "INDETERMINATE",
}
SOURCE_CLASSES = {"ORACLE", "IMPLEMENTATION", "CONFIG", "TEST", "VALIDATION"}


class VerificationFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise VerificationFailure(
            f"git {' '.join(args)} failed: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result


def git_value(*args: str) -> str:
    return git(*args).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def between(text: str, start: str, end: str) -> str:
    require(start in text and end in text, f"missing marker pair {start}, {end}")
    return text.split(start, 1)[1].split(end, 1)[0]


def markdown_rows(section: str, first_cell: str) -> list[list[str]]:
    pattern = re.compile(first_cell)
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and pattern.fullmatch(cells[0]):
            rows.append(cells)
    return rows


def status_paths() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in git("status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines():
        require(len(line) >= 4, f"malformed status line: {line!r}")
        path = line[3:].split(" -> ", 1)[-1]
        rows.append((line[:2], path))
    return rows


def verify_repository_scope() -> str:
    require(git_value("rev-parse", f"{EXPECTED_BASE}^{{tree}}") == EXPECTED_BASE_TREE,
            "base tree mismatch")
    require(git_value("rev-parse", "refs/heads/codex/refactor-mainline") == EXPECTED_MAINLINE,
            "codex/refactor-mainline moved")
    require(git_value("symbolic-ref", "-q", "HEAD") == EXPECTED_BRANCH,
            "not on the dedicated task branch")
    head = git_value("rev-parse", "HEAD")
    expected_paths = {PACKET_PREFIX + name for name in REQUIRED_FILES}
    status = status_paths()

    if head == EXPECTED_BASE:
        require({path for _, path in status} == expected_paths,
                "pre-commit status is not the exact packet artifact set")
        require(all(code == "??" for code, _ in status),
                "pre-commit packet files are not all untracked")
        require(not git("diff", "--name-only").stdout.strip(),
                "tracked working-tree changes exist before packet commit")
        require(not git("diff", "--cached", "--name-only").stdout.strip(),
                "staged changes exist before packet commit")
        mode = "pre_commit"
    else:
        require(git_value("rev-parse", "HEAD^") == EXPECTED_BASE,
                "candidate is not a single child commit of the exact base")
        changed = set(git("diff", "--name-only", f"{EXPECTED_BASE}..HEAD").stdout.splitlines())
        require(changed == expected_paths,
                "candidate commit does not contain exactly the packet artifacts")
        require(not status, "post-commit worktree is not clean")
        require(not git("diff", "--check", f"{EXPECTED_BASE}..HEAD").stdout.strip(),
                "candidate diff check failed")
        mode = "post_commit"
    return mode


def verify_artifacts() -> None:
    actual = {path.name for path in PACKET_DIR.iterdir() if path.is_file()}
    require(actual == REQUIRED_FILES, f"packet artifact set mismatch: {sorted(actual)}")
    require(not list(PACKET_DIR.rglob("*.pdf")), "PDF found in packet")
    for path in PACKET_DIR.iterdir():
        require(path.is_file() and not path.is_symlink(), f"non-regular artifact: {path.name}")


def verify_source_manifest() -> int:
    text = (PACKET_DIR / "SOURCE_AUTHORITY_MANIFEST.md").read_text(encoding="utf-8")
    section = between(text, "<!-- BEGIN-ADMITTED-SOURCES -->",
                      "<!-- END-ADMITTED-SOURCES -->")
    rows = markdown_rows(section, r"CTI-S\d{3}")
    require([row[0] for row in rows] ==
            [f"CTI-S{number:03d}" for number in range(1, 49)],
            "admitted source IDs are not exactly CTI-S001..CTI-S048")
    for row in rows:
        require(len(row) == 5, f"malformed source row: {row[0]}")
        source_id, source_class, quoted_path, quoted_digest, _ = row
        require(source_class in SOURCE_CLASSES, f"invalid source class: {source_id}")
        require(quoted_path.startswith("`") and quoted_path.endswith("`"),
                f"unquoted source path: {source_id}")
        require(quoted_digest.startswith("`") and quoted_digest.endswith("`"),
                f"unquoted source digest: {source_id}")
        relative = quoted_path[1:-1]
        expected_digest = quoted_digest[1:-1]
        require(re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None,
                f"invalid digest form: {source_id}")
        path = REPO / relative
        require(path.is_file(), f"missing admitted source: {source_id} {relative}")
        require(sha256_file(path) == expected_digest,
                f"admitted source digest mismatch: {source_id} {relative}")
    return len(rows)


def verify_stable_ids_and_states() -> tuple[int, int, int]:
    product_text = (PACKET_DIR / "PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md").read_text(encoding="utf-8")
    route_text = (PACKET_DIR / "ROUTE_AVAILABILITY_CLASSIFICATION.md").read_text(encoding="utf-8")
    trace_text = (PACKET_DIR / "REPRESENTATIVE_TRACE_VALIDATION_PLAN.md").read_text(encoding="utf-8")

    product_rows = markdown_rows(
        between(product_text, "<!-- BEGIN-PRODUCT-TRACEABILITY -->",
                "<!-- END-PRODUCT-TRACEABILITY -->"),
        r"MSP-P(?:\d{3}|X01)",
    )
    route_rows = markdown_rows(
        between(route_text, "<!-- BEGIN-ROUTE-CLASSIFICATION -->",
                "<!-- END-ROUTE-CLASSIFICATION -->"),
        r"MSP-E\d{3}",
    )
    trace_rows = markdown_rows(
        between(trace_text, "<!-- BEGIN-TRACE-PLAN -->",
                "<!-- END-TRACE-PLAN -->"),
        r"MSP-T\d{3}",
    )
    require(tuple(row[0] for row in product_rows) == PRODUCT_IDS,
            "product IDs/order differ from the accepted graph")
    require(tuple(row[0] for row in route_rows) == EDGE_IDS,
            "edge IDs/order differ from the accepted graph")
    require(tuple(row[0] for row in trace_rows) == TRACE_IDS,
            "trace IDs/order differ from the accepted audit")
    for row in product_rows:
        require(len(row) == 6, f"product column count: {row[0]}")
        require(row[2].strip("`") in STATES, f"invalid product state: {row[0]}")
    for row in route_rows:
        require(len(row) == 5, f"route column count: {row[0]}")
        require(row[2].strip("`") in STATES, f"invalid route state: {row[0]}")
    for row in trace_rows:
        require(len(row) == 6, f"trace column count: {row[0]}")

    packet_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKET_DIR.glob("*.md")
    )
    for state in STATES:
        require(state in packet_text, f"state vocabulary missing: {state}")
    seen_products = set(re.findall(r"MSP-P(?:\d{3}|X01)", packet_text))
    seen_edges = set(re.findall(r"MSP-E\d{3}", packet_text))
    seen_traces = set(re.findall(r"MSP-T\d{3}", packet_text))
    require(seen_products == set(PRODUCT_IDS), "unknown or missing product IDs in packet")
    require(seen_edges == set(EDGE_IDS), "unknown or missing edge IDs in packet")
    require(seen_traces == set(TRACE_IDS), "unknown or missing trace IDs in packet")
    return len(product_rows), len(route_rows), len(trace_rows)


def verify_required_statements() -> None:
    work_order = (PACKET_DIR / "WORK_ORDER.md").read_text(encoding="utf-8")
    report = (PACKET_DIR / "FINAL_REPORT.md").read_text(encoding="utf-8")
    fruit = (PACKET_DIR / "FRUIT_ATTACHMENT_ENVELOPE.md").read_text(encoding="utf-8")
    oof = (PACKET_DIR / "OOF_ATTACHMENT_ENVELOPE.md").read_text(encoding="utf-8")
    for phrase in (
        "Program adherence and prior-work recovery",
        EXPECTED_BASE,
        EXPECTED_BASE_TREE,
        "source-level",
        "No implementation-derived material is promoted to scientific authority",
    ):
        require(phrase in work_order, f"work-order statement missing: {phrase}")
    for phrase in (
        "zero complete conformant end-to-end routes",
        "No configured local `build/` directory existed",
        "scientific-owner review",
        "No active FRUIT branch or historical ALIGN worktree was inspected",
    ):
        require(phrase in report, f"report statement missing: {phrase}")
    require("No route-realization" in fruit and "no OOF implementation claim" in oof,
            "attachment-envelope nonclaim missing")
    forbidden_claims = (
        "Status: production-ready",
        "is a validated frozen implementation",
        "Unity conformance established",
        "all map-space routes conform",
    )
    all_text = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKET_DIR.glob("*.md")
    )
    for phrase in forbidden_claims:
        require(phrase not in all_text, f"forbidden overclaim found: {phrase}")


def main() -> int:
    try:
        mode = verify_repository_scope()
        verify_artifacts()
        source_count = verify_source_manifest()
        product_count, edge_count, trace_count = verify_stable_ids_and_states()
        verify_required_statements()
    except (VerificationFailure, OSError, UnicodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    head = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", "HEAD^{tree}")
    print(f"PASS mode={mode} head={head} tree={tree}")
    print(f"PASS exact_base={EXPECTED_BASE} base_tree={EXPECTED_BASE_TREE}")
    print(f"PASS admitted_sources={source_count} digests_match=true")
    print(f"PASS products={product_count} edges={edge_count} traces={trace_count}")
    print("PASS state_vocabulary=closed stable_ids=exact")
    print("PASS packet_only=true frozen_sources_unchanged=true application_unchanged=true")
    print("PASS fruit_not_inspected=true align_worktree_not_inspected=true oof_envelope_only=true")
    print("PASS claims=source_level_only owner_review_required=true")
    print("map_space_contract_to_implementation_packet=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
