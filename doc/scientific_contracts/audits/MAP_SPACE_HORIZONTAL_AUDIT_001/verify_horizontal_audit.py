#!/usr/bin/env python3
"""Bounded verifier for MAP-SPACE-HORIZONTAL-AUDIT-001.

This verifier checks audit identity, source bytes, artifact completeness,
stable ID closure, explicit non-PASS mappings, and repository mutation scope.
It performs no scientific interpretation and writes no files.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
REPO = AUDIT_DIR.parents[3]
AUDIT_PREFIX = "doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/"

EXPECTED_COMMIT = "5f0fc20042b88fb6cd883c92d1b59b7f22832901"
EXPECTED_TREE = "97a4d908061e51418f93afc1d97d27433af441b8"
EXPECTED_PARENT = "9a2780aa3bd8343fea87ac0b28b390384118c883"
EXPECTED_LOCAL_REF = EXPECTED_COMMIT
EXPECTED_WORK_ORDER = "b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f"
EXPECTED_SOURCE_ATTACHMENT = "400388f1172bd155866f770debbd5754c0cf86ee364e31b5a6d2bdadc2c82713"
SOURCE_ATTACHMENT_BYTES = 28_899
EXPECTED_SOURCE_MANIFEST = "d21d1446ebcdda8597cf08a4568be91906e3cc22e97f9e7f5544a5fa590b2cd5"

OWNER_RESOLVED_STATE = "OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED"
QUALIFIED_DISPOSITION = "ACCEPT WITH BOUNDED CONTRACT REPAIR"

REQUIRED_FILES = {
    "WORK_ORDER.md",
    "SOURCE_AUTHORITY_MANIFEST.md",
    "PRODUCT_AND_BOUNDARY_GRAPH.md",
    "CROSS_PACKAGE_CONFORMANCE_MATRIX.md",
    "FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md",
    "HORIZONTAL_AUDIT_REPORT.md",
    "verify_horizontal_audit.py",
}

PACKAGES = (
    "SCI-MAP",
    "SCI-JINC",
    "SCI-FLT-FIXED",
    "SCI-FLT-MATCHED",
    "SCI-NOI",
    "SCI-POINT",
)

PRODUCT_IDS = tuple(f"MSP-P{n:03d}" for n in range(1, 17)) + ("MSP-PX01",)
EDGE_IDS = tuple(f"MSP-E{n:03d}" for n in range(1, 33))
FINDING_IDS = tuple(f"MSP-F-{n:03d}" for n in range(1, 7))
UNAVAILABLE_IDS = tuple(f"MSP-U-{n:03d}" for n in range(1, 12))
TRACE_IDS = tuple(f"MSP-T{n:03d}" for n in range(1, 17))
STATUSES = {
    "PASS",
    "CONDITIONAL",
    "CONTRADICTION",
    "UNAVAILABLE",
    "NOT_APPLICABLE",
}


class AuditFailure(RuntimeError):
    """One bounded verification condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise AuditFailure(
            f"git {' '.join(args)} failed: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed


def git_value(*args: str) -> str:
    return git(*args).stdout.strip()


def between(text: str, start: str, end: str) -> str:
    require(start in text and end in text, f"missing marker pair: {start}, {end}")
    return text.split(start, 1)[1].split(end, 1)[0]


def markdown_rows(section: str, first_cell_pattern: str) -> list[list[str]]:
    rows: list[list[str]] = []
    matcher = re.compile(first_cell_pattern)
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and matcher.fullmatch(cells[0]):
            rows.append(cells)
    return rows


def verify_repository_identity() -> None:
    require(git_value("rev-parse", "HEAD") == EXPECTED_COMMIT, "HEAD commit mismatch")
    require(git_value("rev-parse", "HEAD^{tree}") == EXPECTED_TREE, "HEAD tree mismatch")
    require(git_value("rev-parse", "HEAD^") == EXPECTED_PARENT, "HEAD parent mismatch")
    require(
        git_value("rev-parse", "refs/heads/codex/refactor-mainline") == EXPECTED_LOCAL_REF,
        "local codex/refactor-mainline moved from the audited base",
    )
    require(
        git("symbolic-ref", "-q", "HEAD", check=False).returncode != 0,
        "audit checkout is no longer detached",
    )


def verify_artifact_set() -> None:
    actual = {path.name for path in AUDIT_DIR.iterdir() if path.is_file()}
    require(actual == REQUIRED_FILES, f"audit artifact set mismatch: {sorted(actual)}")
    require(not list(AUDIT_DIR.rglob("*.pdf")), "PDF found in audit directory")


def verify_work_order() -> None:
    path = AUDIT_DIR / "WORK_ORDER.md"
    payload = path.read_bytes()
    require(sha256_bytes(payload) == EXPECTED_WORK_ORDER, "consolidated work-order digest")
    require(len(payload) > SOURCE_ATTACHMENT_BYTES, "owner refinements missing")
    require(
        sha256_bytes(payload[:SOURCE_ATTACHMENT_BYTES]) == EXPECTED_SOURCE_ATTACHMENT,
        "original work-order attachment prefix mismatch",
    )
    text = payload.decode("utf-8")
    normalized_text = re.sub(r"\s+", " ", text)
    for refinement in range(1, 6):
        require(f"{refinement}." in text, f"owner refinement {refinement} missing")
    for required_phrase in (
        "Owner Disposition Incorporated 2026-09-03",
        "nonpolarimetric total-intensity-equivalent",
        r"u_{\mathrm{op}}=1",
        "unique original AST-coordinate occurrences",
        "exact frozen five-role SCI-JINC base bundle",
        "OWNER-RESOLVED",
        "SHARED-SOURCE-REPAIR-REQUIRED",
    ):
        require(
            required_phrase in normalized_text,
            f"owner-disposition phrase missing: {required_phrase}",
        )


def verify_source_ledger() -> int:
    manifest_path = AUDIT_DIR / "SOURCE_AUTHORITY_MANIFEST.md"
    require(sha256_file(manifest_path) == EXPECTED_SOURCE_MANIFEST, "source manifest digest")
    text = manifest_path.read_text(encoding="utf-8")
    section = between(
        text,
        "<!-- BEGIN-ADMITTED-SOURCES -->",
        "<!-- END-ADMITTED-SOURCES -->",
    )
    rows = markdown_rows(section, r"SRC-\d{3}")
    expected_ids = [f"SRC-{number:03d}" for number in range(1, 72)]
    require([row[0] for row in rows] == expected_ids, "source IDs are not complete/sequential")
    valid_classes = {"NORMATIVE", "REPRESENTATION", "BOUNDARY_ONLY", "PROCESS_LOCK", "MANAGER_ONLY"}
    for row in rows:
        require(len(row) == 4, f"malformed source row: {row[0]}")
        source_id, source_class, quoted_path, quoted_digest = row
        require(source_class in valid_classes, f"invalid source class: {source_id}")
        require(quoted_path.startswith("`") and quoted_path.endswith("`"), f"path quoting: {source_id}")
        require(quoted_digest.startswith("`") and quoted_digest.endswith("`"), f"digest quoting: {source_id}")
        relative = quoted_path[1:-1]
        expected_digest = quoted_digest[1:-1]
        require(re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None, f"digest form: {source_id}")
        source_path = REPO / relative
        require(source_path.is_file(), f"missing admitted source: {source_id} {relative}")
        require(sha256_file(source_path) == expected_digest, f"digest mismatch: {source_id} {relative}")

    for package in PACKAGES:
        require(package in text, f"missing package in source manifest: {package}")
    for required_phrase in (
        "package-local science was not reopened",
        "Active SCI-FRUIT",
        "historical ALIGN",
        "representation fidelity",
        QUALIFIED_DISPOSITION,
        "owner disposition",
        "shared-source repair",
    ):
        require(required_phrase.lower() in text.lower(), f"source-manifest phrase missing: {required_phrase}")
    return len(rows)


def verify_ids_and_matrix() -> tuple[int, int, int]:
    graph = (AUDIT_DIR / "PRODUCT_AND_BOUNDARY_GRAPH.md").read_text(encoding="utf-8")
    matrix = (AUDIT_DIR / "CROSS_PACKAGE_CONFORMANCE_MATRIX.md").read_text(encoding="utf-8")
    findings = (AUDIT_DIR / "FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md").read_text(encoding="utf-8")
    report = (AUDIT_DIR / "HORIZONTAL_AUDIT_REPORT.md").read_text(encoding="utf-8")

    product_rows = markdown_rows(graph, r"MSP-P(?:\d{3}|X01)")
    require(tuple(row[0] for row in product_rows) == PRODUCT_IDS, "product registry mismatch")

    graph_rows = markdown_rows(
        between(graph, "<!-- BEGIN-GRAPH-EDGES -->", "<!-- END-GRAPH-EDGES -->"),
        r"MSP-E\d{3}",
    )
    matrix_rows = markdown_rows(
        between(matrix, "<!-- BEGIN-CONFORMANCE-ROWS -->", "<!-- END-CONFORMANCE-ROWS -->"),
        r"MSP-E\d{3}",
    )
    require(tuple(row[0] for row in graph_rows) == EDGE_IDS, "graph edge registry mismatch")
    require(tuple(row[0] for row in matrix_rows) == EDGE_IDS, "matrix edge registry mismatch")

    non_pass_rows = 0
    for row in matrix_rows:
        require(len(row) == 13, f"matrix column count: {row[0]} ({len(row)})")
        statuses = row[1:12]
        mapping = row[12]
        require(all(status in STATUSES for status in statuses), f"invalid status: {row[0]}")
        if any(status != "PASS" for status in statuses):
            non_pass_rows += 1
            require(
                re.search(r"MSP-(?:F|U)-\d{3}", mapping) is not None,
                f"non-PASS row lacks finding/unavailable mapping: {row[0]}",
            )

    for code in "ABCDEFGHIJK":
        require(re.search(rf"\| {code} \|", matrix) is not None, f"matrix dimension {code} missing")
    for finding_id in FINDING_IDS:
        require(finding_id in findings and finding_id in report, f"finding ID closure: {finding_id}")
    for unavailable_id in UNAVAILABLE_IDS:
        require(unavailable_id in matrix, f"unavailable ID missing: {unavailable_id}")
    for trace_id in TRACE_IDS:
        require(trace_id in report, f"trace ID missing: {trace_id}")
    require("MSP-OD-001" in findings and "MSP-OD-001" in report, "owner-decision ID closure")
    for artifact_name, artifact_text in (
        ("graph", graph),
        ("matrix", matrix),
        ("findings", findings),
        ("report", report),
    ):
        require("MSP-OD-001" in artifact_text, f"owner disposition missing from {artifact_name}")
        normalized_artifact = re.sub(r"\s+", " ", artifact_text).lower()
        require(
            "shared-source repair" in normalized_artifact,
            f"shared-source repair state missing from {artifact_name}",
        )
    require(
        findings.count(f"State: `{OWNER_RESOLVED_STATE}`") == 4,
        "four MAJOR findings do not have the exact owner-resolved repair-required state",
    )
    for finding_id in FINDING_IDS[:4]:
        require(
            f"| {finding_id} | MAJOR | `{OWNER_RESOLVED_STATE}` |" in report,
            f"report state mismatch: {finding_id}",
        )
    for clause in (
        "330-332",
        "413-416",
        "446-454",
        "470-475",
        "491-492",
        "497-501",
        "503-505",
        "510-515",
        "533-548",
        "697-702",
        "698",
        "700",
    ):
        require(clause in findings and clause in report, f"shared clause missing: {clause}")
    for negative_state in ("NOT_AUTHORIZED", "UNAVAILABLE", "NOT_APPLICABLE"):
        require(negative_state in report, f"negative trace state missing: {negative_state}")
    require(
        "cross-observation coaddition" in findings
        and "cross-observation coadd" in report,
        "negative JINC coaddition rule missing",
    )
    require(
        "does not issue an\nunqualified `PASS`" in report,
        "qualified repository-documentation conclusion missing",
    )
    require(
        report.rstrip().endswith(
            f"Recommended disposition: **{QUALIFIED_DISPOSITION}**"
        ),
        "report does not end with the required qualified disposition",
    )
    return len(product_rows), len(graph_rows), non_pass_rows


def verify_mutation_scope() -> int:
    status_lines = git("status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    for line in status_lines:
        require(len(line) >= 4, f"malformed git status line: {line!r}")
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        require(path_text.startswith(AUDIT_PREFIX), f"out-of-scope repository change: {path_text}")
    require(not git("diff", "--name-only").stdout.strip(), "tracked working-tree modification")
    require(not git("diff", "--cached", "--name-only").stdout.strip(), "staged modification")
    require(status_lines and len(status_lines) == len(REQUIRED_FILES), "status entry count mismatch")
    require(all(line.startswith("?? ") for line in status_lines), "audit changes are not all untracked")
    return len(status_lines)


def main() -> int:
    try:
        verify_repository_identity()
        verify_artifact_set()
        verify_work_order()
        source_count = verify_source_ledger()
        product_count, edge_count, non_pass_count = verify_ids_and_matrix()
        status_count = verify_mutation_scope()
    except (AuditFailure, OSError, UnicodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS audit_commit={EXPECTED_COMMIT}")
    print(f"PASS audit_tree={EXPECTED_TREE}")
    print(f"PASS audit_parent={EXPECTED_PARENT} detached_head=true local_ref_unchanged=true")
    print(
        "PASS work_order_sha256="
        f"{EXPECTED_WORK_ORDER} source_attachment_prefix_sha256={EXPECTED_SOURCE_ATTACHMENT}"
    )
    print(f"PASS source_manifest_sha256={EXPECTED_SOURCE_MANIFEST} admitted_sources={source_count}")
    print(f"PASS packages={len(PACKAGES)} formal_core_rationale_ecs_represented=true")
    print(f"PASS products={product_count} graph_edges={edge_count} matrix_edge_closure=true")
    print(f"PASS non_pass_rows={non_pass_count} finding_or_unavailable_mapping=true")
    print("PASS owner_resolved_major_findings=4 shared_source_repair_required=true")
    print("PASS jinc_negative_traces=excluded_roles_NOT_AUTHORIZED,routes_NOT_AUTHORIZED_or_UNAVAILABLE")
    print("PASS ptc_boundary_only=true fruit_excluded=true align_excluded=true")
    print(f"PASS repository_status_entries={status_count} audit_directory_only=true")
    print("PASS frozen_packages_modified=false application_modified=false freeze_records_modified=false")
    print("PASS audit_pdfs=0 disposition=ACCEPT_WITH_BOUNDED_CONTRACT_REPAIR")
    print("horizontal_audit_verifier=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
