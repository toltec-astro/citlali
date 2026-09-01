#!/usr/bin/env python3
"""Generate the r0.6 authority-only source/link report and manifest."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent

# path, version, authority role, approval state, compatibility/supersession,
# generated-view relation
OBJECTS = [
    ("SCIENTIFIC_OWNER_R0.6_DIRECTIVE_2026-09-01.md", "r0.6", "exact scientific-owner micro-repair directive", "binding repair authority; not title/AO disposition", "accepts r0.5 as repair basis", "authority record"),
    ("SCIENTIFIC_OWNER_DECISION_LEDGER.md", "r0.6", "stable owner-decision ledger", "all 17 SODL IDs disposed", "retains decided/deferred/superseded states", "shared traceability"),
    ("STOCHASTIC_MODEL_AND_OBSERVED_PAYLOAD_AMENDMENT_R0.6.md", "r0.6", "stochastic/observed type amendment", "frozen scientific authority", "supersedes conflated random/observed notation", "derived from exact directive"),
    ("LIFECYCLE_STATE_GRAPH_R0.6.md", "r0.6", "lifecycle-order amendment", "frozen scientific authority", "supersedes incorrect publication order", "derived from exact directive"),
    ("AO001_AUTHORIZATION_MULTIPLICITY_AMENDMENT_R0.6.md", "r0.6", "authorization multiplicity amendment", "frozen scientific authority", "clarifies package versus realization authority", "derived from exact directive"),
    ("AO_OWNER_DISPOSITION_PACKET_R0.6.md", "r0.6", "AO disposition packet", "owner approved", "A/C eligible parameterized methods; B/D successor triggers", "identical two-view routing basis"),
    ("OPTIMALITY_TITLE_OWNER_DISPOSITION_R0.6.md", "r0.6", "title disposition packet", "owner selected option 1", "supersedes provisional historical rendering", "identical two-view routing basis"),
    ("CONDITIONAL_SCIENTIFIC_FREEZE_PROPOSAL_R0.6.md", "r0.6", "scientific-freeze gate", "adopted", "enacted by exact external manifest binding", "owner-facing gate"),
    ("SCIENTIFIC_OWNER_FINAL_DISPOSITIONS_R0.6.md", "r0.6", "final owner dispositions", "binding frozen science", "closes all stable SODL entries", "normative routing basis"),
    ("ROUTE_STATUS_R0.6.md", "r0.6", "route availability record", "frozen scientific status", "supersedes active r0.5 route status", "shared status view"),
    ("SEMANTIC_CHANGE_MAP_R0.6.md", "r0.6", "micro-repair map", "frozen traceability", "maps r0.5 to r0.6 only", "source/two-view parity"),
    ("OWNER_DECISION_PARITY_R0.6.md", "r0.6", "owner-direction parity record", "frozen traceability", "supersedes active r0.5 parity", "source/two-view parity"),
    ("SCI-MAP_TO_SCI-FLT-MATCHED-v0.1-r0.6.md", "v0.1 r0.6", "MAP producer boundary", "frozen scientific authority", "supersedes r0.5 boundary draft", "derived current boundary"),
    ("SCI-TEMPLATE_TO_SCI-FLT-MATCHED-v0.1-r0.6.md", "v0.1 r0.6", "template producer boundary", "frozen scientific authority", "supersedes r0.5 boundary draft", "derived current boundary"),
    ("SCI-FLT-MATCHED_TO_SCI-NOI-v0.1-r0.6.md", "v0.1 r0.6", "NOI consumer boundary", "frozen scientific authority", "supersedes r0.5 boundary draft", "derived current boundary"),
    ("SCI-FLT-MATCHED_TO_SCI-FRUIT-v0.1-r0.6.md", "v0.1 r0.6", "FRUIT producer-envelope boundary", "frozen scientific authority", "supersedes r0.5 boundary draft", "derived current boundary"),
    *[(f"role_semantics/{code}_R0.6.md", "r0.6", f"{code} role-semantics definition", "owner approved; Registry-unregistered and unevaluable", "not an executable policy", "shared normative semantics") for code in ["PA", "SA", "SP", "CU", "NU", "RU", "FH"]],
    ("CROSSWALK.md", "r0.6", "96-ID stable crosswalk", "authored traceability", "counts unchanged", "shared source and both views"),
    ("src/common/README.md", "r0.6", "shared-source module index", "authored support", "active r0.6 module set", "describes shared source"),
    *[(f"src/common/{name}", "r0.6", f"shared normative {name.removesuffix('.tex')}", "authored micro-repair", "supersedes active r0.5 source", "imported byte-identically by both views") for name in ["notation.tex", "definitions.tex", "equations.tex", "assumptions.tex", "requirements.tex", "edge_cases.tex"]],
    ("src/scientific-rationale.tex", "v0.1 r0.6", "scientific-view source", "frozen scientific authority", "supersedes active r0.5 view source", "generated view over shared core"),
    ("src/engineering-conformance.tex", "v0.1 r0.6", "engineering-view source", "frozen scientific authority", "supersedes active r0.5 view source", "generated view over shared core"),
    ("pdf/README.md", "r0.6", "rendered-artifact index", "authored support", "active r0.6 PDF set", "describes generated views"),
    ("pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf", "v0.1 r0.6", "canonical scientific rendered view", "frozen scientific authority", "supersedes active r0.5 PDF", "generated from scientific source/shared core"),
    ("pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf", "v0.1 r0.6", "canonical engineering rendered view", "frozen scientific authority", "supersedes active r0.5 PDF", "generated from engineering source/shared core"),
    ("PDF_QA_R0.6.md", "r0.6", "all-page PDF QA evidence", "mechanical and visual QA passed", "supersedes active r0.5 QA", "evidence for generated views"),
    ("build/BUILD_VERIFICATION.md", "r0.6", "build verification record", "mechanical verification passed", "supersedes active r0.5 build record", "source/PDF evidence"),
    ("build/consistency-report.json", "r0.6", "machine consistency evidence", "mechanical verification passed", "active r0.6 report", "generated from source/PDFs"),
    ("build/verify_consistency.py", "r0.6", "source/PDF verifier", "verification tooling", "active r0.6 checks", "tests source and views"),
    ("build/audit_bundle_links.py", "r0.6", "standalone link auditor", "verification tooling", "new r0.6 bundle-policy check", "tests extracted Markdown tree"),
    ("build/scientific-rationale.log", "r0.6", "scientific build log", "clean build evidence", "supersedes active r0.5 log", "render evidence"),
    ("build/engineering-conformance.log", "r0.6", "engineering build log", "clean build evidence", "supersedes active r0.5 log", "render evidence"),
    ("build/generate_r0_6_authority.py", "r0.6", "authority inventory generator", "verification tooling", "active r0.6 generator", "generates report/manifest"),
    ("verify_stage_b_draft.py", "r0.6", "standalone authority verifier", "verification tooling", "supersedes r0.5 verifier behavior", "tests manifest-bound objects"),
]

HISTORICAL_ANCHORS = [
    ("Stage A approved author-packet manifest", "AUTHOR_PACKET_MANIFEST.md", "255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8"),
    ("r0.2 scientific-owner directive", "SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md", "ef03af840c00b9934b04ceab0635be4f7273afe929e65f84eaa1b31985daa155"),
    ("r0.3 repair authority", "SCIENTIFIC_OWNER_R0.3_REPAIR_AUTHORITY_2026-09-01.md", "354a0647709aa98f331b9a50606b6c08d4e26d61e161d796c62e4e2e819d5e4a"),
    ("r0.4 scientific-owner directive", "SCIENTIFIC_OWNER_R0.4_DIRECTIVE_2026-09-01.md", "ccaca43228f4ac9f719aa93201cc13c37abdd8b2ddc01c65f2733785d12cf408"),
    ("r0.5 scientific-owner directive", "SCIENTIFIC_OWNER_R0.5_DIRECTIVE_2026-09-01.md", "e27c04e62f01fa2997b229cf50d4a5b1c12cfd3d74805f4806bef6dde1261274"),
]

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def details(relative: str) -> tuple[int, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path.stat().st_size, digest(path)


def object_link_failures() -> list[str]:
    active = {relative for relative, *_ in OBJECTS}
    failures: list[str] = []
    for relative in sorted(active):
        if not relative.endswith(".md"):
            continue
        source = ROOT / relative
        for raw in LINK_RE.findall(source.read_text(encoding="utf-8")):
            target = raw.strip().strip("<>").split(maxsplit=1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            resolved = (source.parent / target).resolve()
            try:
                resolved_relative = str(resolved.relative_to(ROOT))
            except ValueError:
                failures.append(f"{relative} -> {raw} (escapes package)")
                continue
            if resolved_relative not in active:
                failures.append(f"{relative} -> {raw} (not an active bundle object)")
    return failures


def generate_source_report() -> None:
    failures = object_link_failures()
    if failures:
        raise RuntimeError("active-object link closure failed: " + "; ".join(failures))
    lines = [
        "# SCI-FLT-MATCHED v0.1 r0.6 Source-Byte and Link Closure",
        "",
        "Date: `2026-09-01`",
        "",
        "Status: exact frozen-authority byte inventory and standalone bundle-link",
        "closure; no implementation, conformity, validation, or production claim",
        "",
        "Every digest below was computed from the actual package-local bytes. The",
        "active object set contains zero unresolved local Markdown links and no link",
        "to a repository-context-only historical object. The manifest, sidecar, and",
        "bundle note are added after this report and contain no Markdown links.",
        "`verify_stage_a.py` is intentionally not a bundle object; its successful",
        "repository-context execution is recorded in the build verification record.",
        "",
        "| Object | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for relative, *_ in OBJECTS:
        size, hexdigest = details(relative)
        lines.append(f"| `{relative}` | {size} | `{hexdigest}` |")
    lines.extend([
        "",
        f"Active objects audited: `{len(OBJECTS)}`.",
        "Unresolved active-object local Markdown links: `0`.",
        "",
        "This report is itself bound by the active authority manifest. The manifest",
        "is externally bound by `STAGE_B_DRAFT_MANIFEST.sha256`, avoiding self-hash",
        "cycles.",
        "",
    ])
    (ROOT / "SOURCE_BYTE_AND_LINK_CLOSURE_R0.6.md").write_text("\n".join(lines), encoding="utf-8")


def generate_manifest() -> None:
    rows = [*OBJECTS, ("SOURCE_BYTE_AND_LINK_CLOSURE_R0.6.md", "r0.6", "exact source/link closure", "mechanical inventory passed", "current authority-only object set", "inventory of all other active objects")]
    lines = [
        "# SCI-FLT-MATCHED v0.1 — Frozen Scientific Authority Manifest",
        "",
        "Manifest identity: `SCI-FLT-MATCHED_SCIENTIFIC_AUTHORITY v0.1/r0.6`",
        "",
        "Status: exact object binding for frozen scientific authority; external",
        "freeze record binds the SHA-256 of this manifest",
        "",
        "The object rows are the complete standalone authority-only bundle set.",
        "Historical anchors below are repository-context provenance, not bundle",
        "objects and not unresolved bundle-local links.",
        "",
        "| # | Object | Version | Bytes | SHA-256 | Authority role | Approval state | Compatibility / supersession | Generated-view relation |",
        "| ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for number, row in enumerate(rows, start=1):
        relative, version, authority, approval, compatibility, relation = row
        size, hexdigest = details(relative)
        lines.append(f"| {number} | `{relative}` | `{version}` | {size} | `{hexdigest}` | {authority} | {approval} | {compatibility} | {relation} |")
    lines.extend([
        "",
        "## Historical authority anchors — repository context only",
        "",
        "| Role | Repository object | SHA-256 |",
        "| --- | --- | --- |",
    ])
    for role, relative, hexdigest in HISTORICAL_ANCHORS:
        if digest(ROOT / relative) != hexdigest:
            raise RuntimeError(f"historical anchor changed: {relative}")
        lines.append(f"| {role} | `{relative}` | `{hexdigest}` |")
    lines.extend([
        "",
        "The r0.6 directive is an active object row and has SHA-256",
        "`5758640064918b2d3021afc7ea63ffba063ba7b1abbb66dc6d43d945ed73ebd3`.",
        "The external digest in `STAGE_B_DRAFT_MANIFEST.sha256` binds this",
        "manifest.",
        "",
        "The selected human title is Matched-template map amplitude estimation.",
        "A and C are separately authorized eligible parameterized methods; B and D",
        "remain successor triggers. This set establishes no available concrete",
        "numerical route, registered SCI-VAL profile, implementation",
        "conformity, response/covariance fidelity, observational validation, achieved",
        "performance, readiness, production authorization, or",
        "Unity claim.",
        "",
    ])
    manifest = ROOT / "STAGE_B_DRAFT_MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "STAGE_B_DRAFT_MANIFEST.sha256").write_text(
        f"{digest(manifest)}  STAGE_B_DRAFT_MANIFEST.md\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate_source_report()
    generate_manifest()
