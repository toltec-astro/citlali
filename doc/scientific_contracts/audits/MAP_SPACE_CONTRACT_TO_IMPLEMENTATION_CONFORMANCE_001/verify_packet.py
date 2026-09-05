#!/usr/bin/env python3
"""Verify the bounded MAP-space contract-to-implementation packet.

The verifier is read-only.  It validates the exact base identity, accepted
oracle and inspected-source bytes, artifact closure, stable product/edge/trace
IDs, closed state vocabulary, derived summary counts, and the six-file repair
boundary before and after one review-repair commit.  Only the owner-authorized
coordinate rows and optional citation may change; predecessor commits remain
fixed.  It performs no scientific interpretation.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


PACKET_DIR = Path(__file__).resolve().parent
REPO = PACKET_DIR.parents[3]
PACKET_PREFIX = (
    "doc/scientific_contracts/audits/"
    "MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/"
)
EXPECTED_BASE = "9f42d348298d76c5d5145aaf0c3eace1f3e154c1"
EXPECTED_BASE_TREE = "e51f22760c64454ce7233c45dd740aa710777bae"
ORIGINAL_CANDIDATE = "93c2b4591bb5d0cf8efe4491975c31e5f8fb5903"
ORIGINAL_TREE = "e0b51383cdeb4ad318d3548b05ad803dd9ef1cf4"
ORIGINAL_BRANCH = "refs/heads/codex/map-space-contract-to-implementation-conformance-001"
PREVIOUS_CANDIDATE = "402b82bc7c38d8a3739d7803f46ccf3f1bbd90f8"
PREVIOUS_TREE = "163b8136066cf56d320cfb24488350118540510f"
PREVIOUS_BRANCH = "refs/heads/codex/map-space-conformance-001-doc-repair-2026-09-04"
EXPECTED_BRANCH = "refs/heads/codex/map-space-conformance-001-review-repair-2026-09-04"
HANDOFF_COMMIT = "ae953ed4d87d1f693d2bbf42aebbc25ef730c771"
REPAIR_RECORD = "REVIEW_REPAIR_RECORD_2026-09-04.md"
REPAIR_FILES = {
    "FINAL_REPORT.md",
    "PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md",
    "ROUTE_AVAILABILITY_CLASSIFICATION.md",
    "OOF_ATTACHMENT_ENVELOPE.md",
    "verify_packet.py",
    REPAIR_RECORD,
}

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
    REPAIR_RECORD,
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
    require(git_value("rev-parse", f"{ORIGINAL_CANDIDATE}^{{tree}}") == ORIGINAL_TREE,
            "original candidate tree mismatch")
    require(git_value("rev-parse", f"{ORIGINAL_CANDIDATE}^") == EXPECTED_BASE,
            "original candidate parent mismatch")
    require(git_value("rev-parse", ORIGINAL_BRANCH) == ORIGINAL_CANDIDATE,
            "original candidate branch moved")
    require(git_value("rev-parse", f"{PREVIOUS_CANDIDATE}^{{tree}}") == PREVIOUS_TREE,
            "previous candidate tree mismatch")
    require(git_value("rev-parse", f"{PREVIOUS_CANDIDATE}^") == ORIGINAL_CANDIDATE,
            "previous candidate parent mismatch")
    require(git_value("rev-parse", PREVIOUS_BRANCH) == PREVIOUS_CANDIDATE,
            "previous candidate branch moved")
    require(git_value("symbolic-ref", "-q", "HEAD") == EXPECTED_BRANCH,
            "not on the dedicated task branch")
    head = git_value("rev-parse", "HEAD")
    expected_paths = {PACKET_PREFIX + name for name in REQUIRED_FILES - {REPAIR_RECORD}}
    original_paths = set(git_value(
        "diff", "--name-only", f"{EXPECTED_BASE}..{ORIGINAL_CANDIDATE}"
    ).splitlines())
    require(original_paths == expected_paths, "original packet path set mismatch")
    repair_paths = {PACKET_PREFIX + name for name in REPAIR_FILES}
    status = status_paths()

    if head == PREVIOUS_CANDIDATE:
        require({path for _, path in status} == repair_paths,
                "pre-commit status is not the exact six-file repair set")
        for code, path in status:
            allowed = {"??", "A ", "AM"} if path == PACKET_PREFIX + REPAIR_RECORD else {" M", "M ", "MM"}
            require(code in allowed, f"unexpected repair path status: {code} {path}")
        untracked = {path for code, path in status if code == "??"}
        require(set(git_value("diff", "--name-only", PREVIOUS_CANDIDATE).splitlines())
                == repair_paths - untracked, "working repair path set mismatch")
        git("diff", "--check", PREVIOUS_CANDIDATE)
        git("diff", "--cached", "--check")
        mode = "pre_commit"
    else:
        require(git_value("rev-parse", "HEAD^") == PREVIOUS_CANDIDATE,
                "successor is not a single child commit of the reviewed predecessor")
        changed = set(git_value("diff", "--name-only", f"{PREVIOUS_CANDIDATE}..HEAD").splitlines())
        require(changed == repair_paths,
                "successor commit does not contain exactly the six repair files")
        require(not status, "post-commit worktree is not clean")
        git("diff", "--check", f"{EXPECTED_BASE}..HEAD")
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

    verify_bounded_table_changes(product_rows, route_rows)
    verify_summary_counts(product_rows, route_rows, route_text)

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


def verify_bounded_table_changes(product_rows: list[list[str]],
                                 route_rows: list[list[str]]) -> None:
    """Permit only the three coordinate rows and the optional test-citation repair."""
    tables = (
        (product_rows, "PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md",
         r"MSP-P(?:\d{3}|X01)", {"MSP-P002"}),
        (route_rows, "ROUTE_AVAILABILITY_CLASSIFICATION.md",
         r"MSP-E\d{3}", {"MSP-E002", "MSP-E006"}),
    )
    for current_rows, filename, pattern, coordinate_ids in tables:
        previous_text = git("show", f"{PREVIOUS_CANDIDATE}:{PACKET_PREFIX}{filename}").stdout
        previous_rows = markdown_rows(previous_text, pattern)
        require([row[0] for row in current_rows] == [row[0] for row in previous_rows],
                f"predecessor row IDs changed: {filename}")
        for current, previous in zip(current_rows, previous_rows):
            row_id = current[0]
            if row_id in coordinate_ids:
                require(current[:2] == previous[:2] and
                        current[2] == "`IMPLEMENTED_LEGACY_SEMANTICS`",
                        f"coordinate repair identity/state mismatch: {row_id}")
                if row_id == "MSP-P002":
                    require(current[3] == previous[3], "MSP-P002 evidence grade changed")
            elif row_id == "MSP-P009":
                expected = previous.copy()
                expected[3] = "`A`"
                expected[4] = expected[4].replace(
                    "; `tests/test_session_failure_boundaries.cpp:280-342`", ""
                )
                require(current == expected, "MSP-P009 optional citation repair mismatch")
            else:
                require(current == previous, f"unrelated predecessor row changed: {row_id}")


def verify_summary_counts(product_rows: list[list[str]],
                          route_rows: list[list[str]], route_text: str) -> None:
    """Derive counts from the classified rows, then check both report views."""
    report = (PACKET_DIR / "FINAL_REPORT.md").read_text(encoding="utf-8")
    product_counts = Counter(row[2].strip("`") for row in product_rows)
    route_counts = Counter(row[2].strip("`") for row in route_rows)

    def check_table(section: str, counts: Counter[str], label: str) -> None:
        rows = markdown_rows(section, r"`[A-Z_]+`")
        require(len(rows) == len(STATES) and
                {row[0].strip('`') for row in rows} == STATES,
                f"{label} summary state set is not exact")
        require(all(len(row) >= 2 and row[1].isdigit() for row in rows),
                f"{label} summary count is malformed")
        for row in rows:
            state = row[0].strip("`")
            require(int(row[1]) == counts[state],
                    f"{label} summary count mismatch: {state}; "
                    f"reported={row[1]} derived={counts[state]}")
        require(sum(int(row[1]) for row in rows) == sum(counts.values()),
                f"{label} summary total mismatch")

    check_table(between(report, "## Product results", "## Route results"),
                product_counts, "product")
    summary = route_text.split("## Availability summary", 1)[-1]
    check_table(summary, route_counts, "route")
    totals = markdown_rows(summary, r"Total")
    require(totals == [["Total", "32"]] and sum(route_counts.values()) == 32,
            "route summary must total exactly 32")

    prose = " ".join(between(report, "## Route results", "## Principal blockers").split())
    require(re.findall(r"All (\d+) original edges", prose) == ["32"],
            "final report route total must be 32")
    labels = {
        "IMPLEMENTED_CONFORMANT_AT_SOURCE_LEVEL": "source-level conformant coordinate fragments",
        "IMPLEMENTED_LEGACY_SEMANTICS": "legacy routes",
        "UNAVAILABLE_BY_DESIGN": "intentionally unavailable routes",
        "MISSING_AUTHORITY": "authority gaps",
        "MISSING_IMPLEMENTATION": "implementation gap",
        "CONTRADICTORY": "contradictions",
        "NOT_APPLICABLE": "excluded FRUIT envelope",
    }
    for state, label in labels.items():
        require(re.findall(r"\b(\d+) " + re.escape(label) + r"\b", prose)
                == [str(route_counts[state])],
                f"final report route count mismatch: {state}")
    require(sum(route_counts[state] for state in labels) == 32,
            "final report omits a nonzero route state")


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
        "The original source study did not inspect the active FRUIT branch",
        "CTI-OD-001--CTI-OD-006 remain open",
        "CTI-OD-007 records inherited/closed",
    ):
        require(phrase in report, f"report statement missing: {phrase}")
    require("No route-realization" in fruit and "no OOF implementation claim" in oof,
            "attachment-envelope nonclaim missing")
    require("non-exhaustive" in oof and "may consider only" not in oof,
            "OOF envelope must not impose an exhaustive parent list")
    repair = (PACKET_DIR / REPAIR_RECORD).read_text(encoding="utf-8")
    for phrase in (
        "Program adherence and prior-work recovery",
        PREVIOUS_CANDIDATE,
        PREVIOUS_TREE,
        "The owner",
        "accepted that recommendation",
        "not an executed numerical test",
        "MSP-P002, MSP-E002 and MSP-E006",
    ):
        require(phrase in repair, f"review-repair record statement missing: {phrase}")
    ledger = (PACKET_DIR / "OWNER_DECISION_LEDGER.md").read_text(encoding="utf-8")
    decision_rows = markdown_rows(
        between(ledger, "<!-- BEGIN-OWNER-DECISIONS -->", "<!-- END-OWNER-DECISIONS -->"),
        r"CTI-OD-\d{3}",
    )
    original_ledger = git(
        "show", f"{ORIGINAL_CANDIDATE}:{PACKET_PREFIX}OWNER_DECISION_LEDGER.md"
    ).stdout
    original_decisions = markdown_rows(original_ledger, r"CTI-OD-\d{3}")
    require([row[0] for row in decision_rows] == [row[0] for row in original_decisions]
            and decision_rows[:6] == original_decisions[:6],
            "owner-decision IDs or open decisions changed")
    require(len(decision_rows[-1]) == 5 and decision_rows[-1][-1] == "`INHERITED_CLOSED`",
            "CTI-OD-007 must record inherited/closed sequencing")
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
    print("PASS state_vocabulary=closed stable_ids=exact coordinate_repairs=three_named_rows")
    print("PASS optional_citation=removed all_other_table_rows=unchanged")
    print("PASS product_summary_counts=derived route_summary_counts=derived route_total=32")
    print("PASS packet_only=true frozen_sources_unchanged=true application_unchanged=true")
    print(f"PASS original_candidate={ORIGINAL_CANDIDATE} original_branch=preserved")
    print(f"PASS previous_candidate={PREVIOUS_CANDIDATE} previous_branch=preserved")
    print("PASS successor_scope=six_packet_files oof_envelope_only=true")
    mainline = git_value("rev-parse", "refs/heads/codex/refactor-mainline")
    print(f"OBSERVED canonical={mainline} handoff_snapshot={HANDOFF_COMMIT} "
          f"drift={str(mainline != HANDOFF_COMMIT).lower()} (not an authority decision)")
    print("PASS claims=source_level_only owner_review_required=true")
    print("map_space_contract_to_implementation_packet=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
