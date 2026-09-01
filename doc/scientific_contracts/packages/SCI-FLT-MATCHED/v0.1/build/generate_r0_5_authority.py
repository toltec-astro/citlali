#!/usr/bin/env python3
"""Generate the SCI-FLT-MATCHED r0.5 source report and authority manifest."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


# path, version, authority role, approval state, compatibility/supersession,
# generated-view relation
OBJECTS = [
    ("AUTHOR_PACKET_MANIFEST.md", "Stage A", "approved packet identity", "owner approved exact bytes", "unchanged from approved packet", "author input"),
    ("SCOPE_BRIEF.md", "Stage A", "admitted scientific scope", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("AUTHOR_SUPERSESSION_COVER.md", "Stage A", "supersession authority", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("AUTHOR_CONVENTIONS_AND_OWNERSHIP.md", "Stage A", "conventions and ownership authority", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md", "Stage A", "scientific-owner direction", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md", "Stage A", "operator and product authority", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("AUTHOR_BOUNDARIES.md", "Stage A", "scientific boundary authority", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("REQUIRED_AUTHORED_OPTION_SETS.md", "Stage A", "option-authorship assignment", "owner approved exact bytes", "active Stage B basis", "author input"),
    ("AUTHOR_PACKET_MANIFEST.sha256", "Stage A", "packet-manifest digest", "owner approved binding", "unchanged from approved packet", "external digest of packet manifest"),
    ("SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md", "Stage A", "launch authority", "binding owner approval", "authorizes exact eight-object packet", "authority record"),
    ("SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md", "r0.2", "targeted closure authority", "binding owner directive", "supplements Stage A scope", "authority record"),
    ("CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md", "r0.2 review", "process provenance", "review result only", "superseded as repair basis by owner r0.3 authority", "independent review record"),
    ("SCIENTIFIC_OWNER_R0.3_REPAIR_AUTHORITY_2026-09-01.md", "r0.3", "repair authority", "binding owner authorization", "authorizes directed r0.3 repair", "authority record"),
    ("CHATGPT_PRO_INDEPENDENT_REVIEW_R0.3_2026-09-01.md", "r0.3 review", "process provenance", "review result only", "superseded as policy authority by owner r0.4 directive", "independent review record"),
    ("SCIENTIFIC_OWNER_R0.4_DIRECTIVE_2026-09-01.md", "r0.4", "covariance-scope and AO-001-C policy authority", "binding owner directive", "supplements r0.3 repair", "authority record"),
    ("SCIENTIFIC_OWNER_R0.5_DIRECTIVE_2026-09-01.md", "r0.5", "final targeted closure authority", "binding owner directive", "supersedes r0.4 where explicitly amended", "authority record"),
    ("AO_OWNER_DISPOSITION_PACKET_R0.5.md", "r0.5", "author recommendation packet", "owner decision pending", "no AO selection", "presented identically in both views"),
    ("OPTIMALITY_TITLE_OWNER_DISPOSITION_R0.5.md", "r0.5", "author title recommendation packet", "owner decision pending", "provisional title is not a selection", "presented identically in both views"),
    ("SCIENTIFIC_OWNER_DECISION_LEDGER.md", "r0.5", "owner-decision ledger", "mixed decided open and superseded states", "retains all 17 stable SODL IDs", "shared traceability record"),
    ("NUMERICAL_APPLICATION_DOMAIN_AMENDMENT_R0.5.md", "r0.5", "directed type amendment", "authored closure draft", "supersedes r0.4 application-domain wording", "derived from r0.5 directive and shared core"),
    ("VALIDITY_DOMAIN_CROSSWALK_R0.5.md", "r0.5", "directed validity amendment", "authored closure draft", "supersedes undifferentiated validity wording", "derived from r0.5 directive and shared core"),
    ("LIFECYCLE_AMENDMENT_R0.5.md", "r0.5", "directed lifecycle amendment", "authored closure draft", "supersedes incomplete r0.4 lifecycle", "derived from r0.5 directive and shared core"),
    ("PRE_DRAW_CONDITIONING_AMENDMENT_R0.5.md", "r0.5", "directed conditioning amendment", "authored closure draft", "supersedes outcome-dependent conditioning ambiguity", "derived from r0.5 directive and shared core"),
    ("GLS_REFERENCE_AND_OPERATIONAL_COVARIANCE_R0.5.md", "r0.5", "directed covariance-role amendment", "authored closure draft", "clarifies r0.4 covariance roles", "derived from r0.5 directive and shared core"),
    ("TEMPLATE_SOURCE_BOUNDARY_AMENDMENT_R0.5.md", "r0.5", "directed template-authority amendment", "authored closure draft", "supersedes overbroad outside-source wording", "derived from r0.5 directive and shared core"),
    ("DOWNSTREAM_BOUNDARY_REQUEST_AMENDMENT_R0.5.md", "r0.5", "directed downstream-state amendment", "authored closure draft", "supersedes unconditional companion implications", "derived from r0.5 directive and boundaries"),
    ("SEMANTIC_CHANGE_MAP_R0.5.md", "r0.5", "semantic repair map", "authored traceability draft", "maps r0.4 to r0.5", "shared source and two-view crosswalk"),
    ("NOTATION_CROSSWALK_R0.5.md", "r0.5", "notation traceability", "authored traceability draft", "supersedes r0.4 where amended", "shared source crosswalk"),
    ("REPRESENTATION_CROSSWALK_R0.5.md", "r0.5", "science-engineering representation boundary", "authored traceability draft", "supersedes r0.4 where amended", "shared source crosswalk"),
    ("ROUTE_STATUS_R0.5.md", "r0.5", "route availability record", "authored status draft", "supersedes active r0.4 route status", "shared source status view"),
    ("OWNER_DECISION_PARITY_R0.5.md", "r0.5", "owner-direction parity record", "authored traceability draft", "supersedes active r0.4 parity", "source and two-view parity"),
    ("CROSSWALK.md", "r0.5", "stable-ID crosswalk", "authored traceability draft", "96 active IDs", "shared source and both rendered views"),
    ("SCI-MAP_TO_SCI-FLT-MATCHED-v0.1-r0.5.md", "v0.1 r0.5", "MAP producer boundary draft", "owner review pending", "supersedes r0.4 boundary draft", "derived from authority and shared core"),
    ("SCI-TEMPLATE_TO_SCI-FLT-MATCHED-v0.1-r0.5.md", "v0.1 r0.5", "template producer boundary draft", "owner review pending", "supersedes r0.4 boundary draft", "derived from authority and shared core"),
    ("SCI-FLT-MATCHED_TO_SCI-NOI-v0.1-r0.5.md", "v0.1 r0.5", "NOI consumer boundary draft", "owner review pending", "supersedes r0.4 boundary draft", "derived from authority and shared core"),
    ("SCI-FLT-MATCHED_TO_SCI-FRUIT-v0.1-r0.5.md", "v0.1 r0.5", "FRUIT producer-envelope boundary draft", "owner review pending", "supersedes r0.4 boundary draft", "derived from authority and shared core"),
    ("role_profiles/PA_R0.5.md", "r0.5", "PA role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/SA_R0.5.md", "r0.5", "SA role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/SP_R0.5.md", "r0.5", "SP role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/CU_R0.5.md", "r0.5", "CU role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/NU_R0.5.md", "r0.5", "NU role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/RU_R0.5.md", "r0.5", "RU role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("role_profiles/FH_R0.5.md", "r0.5", "FH role-policy draft", "owner review pending", "active r0.5 role profile", "derived from shared normative semantics"),
    ("src/common/README.md", "r0.5", "shared-source module index", "authored draft support", "active r0.5 module set", "describes normative source"),
    ("src/common/notation.tex", "r0.5", "shared normative notation", "authored closure draft", "supersedes active r0.4 source", "imported unchanged by both views"),
    ("src/common/definitions.tex", "r0.5", "shared normative definitions", "authored closure draft", "supersedes active r0.4 source", "imported unchanged by both views"),
    ("src/common/equations.tex", "r0.5", "shared normative equations", "authored closure draft", "supersedes active r0.4 source", "imported unchanged by both views"),
    ("src/common/assumptions.tex", "r0.5", "shared normative assumptions", "authored closure draft", "supersedes active r0.4 source", "imported unchanged by both views"),
    ("src/common/requirements.tex", "r0.5", "shared normative requirements and predictions", "authored closure draft", "REQ-001 through 050 stable and PRED-025 appended", "imported unchanged by both views"),
    ("src/common/edge_cases.tex", "r0.5", "shared normative edge and failure cases", "authored closure draft", "supersedes active r0.4 source", "imported unchanged by both views"),
    ("src/scientific-rationale.tex", "v0.1 r0.5", "scientific view source", "owner review pending", "supersedes active r0.4 view source", "generated view over exact shared core"),
    ("src/engineering-conformance.tex", "v0.1 r0.5", "engineering view source", "owner review pending", "supersedes active r0.4 view source", "generated view over exact shared core"),
    ("pdf/README.md", "r0.5", "rendered-artifact index", "authored draft support", "active r0.5 PDF set", "describes generated views"),
    ("pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf", "v0.1 r0.5", "canonical scientific rendered view", "owner review pending", "supersedes active r0.4 rendered view", "generated from scientific source and exact shared core"),
    ("pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf", "v0.1 r0.5", "canonical engineering rendered view", "owner review pending", "supersedes active r0.4 rendered view", "generated from engineering source and exact shared core"),
    ("PDF_QA_R0.5.md", "r0.5", "PDF visual-QA and metadata evidence", "mechanical and human QA passed", "supersedes active r0.4 QA", "evidence for both generated views"),
    ("build/BUILD_VERIFICATION.md", "r0.5", "build verification record", "mechanical verification passed", "supersedes active r0.4 build record", "evidence for source and rendered views"),
    ("build/consistency-report.json", "r0.5", "machine consistency evidence", "mechanical verification passed", "active r0.5 report", "generated from source and PDFs"),
    ("build/verify_consistency.py", "r0.5", "consistency verifier", "verification tooling", "active r0.5 checks", "tests source and generated views"),
    ("build/scientific-rationale.log", "r0.5", "scientific build log", "clean build evidence", "supersedes active r0.4 log", "generated while rendering scientific PDF"),
    ("build/engineering-conformance.log", "r0.5", "engineering build log", "clean build evidence", "supersedes active r0.4 log", "generated while rendering engineering PDF"),
    ("build/generate_r0_5_authority.py", "r0.5", "authority inventory generator", "verification tooling", "active r0.5 generator", "generates report and manifest"),
    ("verify_stage_a.py", "Stage A", "approved-packet verifier", "verification tooling", "unchanged Stage A gate", "tests author inputs"),
    ("verify_stage_b_draft.py", "r0.5", "authority-manifest verifier", "verification tooling", "supersedes r0.4 verifier behavior", "tests manifest-bound objects"),
    ("README.md", "r0.5", "package gate and index", "authored status draft", "supersedes active r0.4 package status", "indexes authority and generated views"),
    ("DECISION_LOG.md", "r0.5", "process decision record", "authored provenance record", "appends r0.5 closure", "process provenance"),
]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def details(relative: str) -> tuple[int, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path.stat().st_size, digest(path)


def generate_source_report() -> None:
    lines = [
        "# SCI-FLT-MATCHED v0.1 r0.5 Source-Byte Report",
        "",
        "Date: `2026-09-01`",
        "",
        "Status: exact returned-source and rendered-byte inventory; no scientific",
        "approval, option selection, implementation claim, or freeze",
        "",
        "Every quoted digest below was computed from the actual package-local bytes.",
        "The independent reviews are process provenance; the owner directives and",
        "dispositions carry the scientific authority stated in the manifest.",
        "",
        "| Object | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for relative, *_ in OBJECTS:
        size, hexdigest = details(relative)
        lines.append(f"| `{relative}` | {size} | `{hexdigest}` |")
    lines.extend(
        [
            "",
            "The active r0.5 authority manifest additionally binds this report.",
            "Because the manifest is externally bound by",
            "`STAGE_B_DRAFT_MANIFEST.sha256`, neither this report nor the manifest",
            "self-lists its own digest.",
            "",
        ]
    )
    (ROOT / "SOURCE_BYTE_REPORT_R0.5.md").write_text("\n".join(lines))


def generate_manifest() -> None:
    rows = list(OBJECTS)
    rows.insert(
        -2,
        (
            "SOURCE_BYTE_REPORT_R0.5.md",
            "r0.5",
            "exact source-byte report",
            "mechanical inventory",
            "supersedes active r0.4 source report",
            "inventory of all other bound object bytes",
        ),
    )
    lines = [
        "# SCI-FLT-MATCHED v0.1 — Stage B r0.5 Authority Manifest",
        "",
        "Manifest identity: `SCI-FLT-MATCHED_STAGE_B_AUTHORITY v0.1/r0.5`",
        "",
        "Status: owner-directed final targeted closure binding for explicit owner",
        "review; not scientific-owner approval of the r0.5 draft and not scientific",
        "freeze",
        "",
        "The independent-review records below are process provenance only. Scientific",
        "authority is carried by the exact owner approval, repair authority, and",
        "directives identified as such. Pending title and AO packets are recommendations,",
        "not owner dispositions. The manifest binds the exact bytes, not merely names.",
        "",
        "| # | Object | Version | Bytes | SHA-256 | Authority role | Approval state | Compatibility / supersession | Generated-view relation |",
        "| ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for number, row in enumerate(rows, start=1):
        relative, version, authority, approval, compatibility, relation = row
        size, hexdigest = details(relative)
        lines.append(
            f"| {number} | `{relative}` | `{version}` | {size} | `{hexdigest}` | "
            f"{authority} | {approval} | {compatibility} | {relation} |"
        )
    lines.extend(
        [
            "",
            "The external digest in `STAGE_B_DRAFT_MANIFEST.sha256` binds this",
            "manifest. The exact approved eight-object author packet is rows 1--8;",
            "row 1 reproduces the owner-approved manifest SHA-256",
            "`255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.",
            "",
            "This authority set selects no title, weighting, concrete covariance",
            "role/scope, representation, named-use profile, or numerical route. It",
            "establishes no implementation conformity, response/covariance fidelity,",
            "observational validation, detection performance, readiness, scientific",
            "freeze, production suitability, production authorization, or Unity claim.",
            "",
        ]
    )
    manifest = ROOT / "STAGE_B_DRAFT_MANIFEST.md"
    manifest.write_text("\n".join(lines))
    (ROOT / "STAGE_B_DRAFT_MANIFEST.sha256").write_text(
        f"{digest(manifest)}  STAGE_B_DRAFT_MANIFEST.md\n"
    )


if __name__ == "__main__":
    generate_source_report()
    generate_manifest()
