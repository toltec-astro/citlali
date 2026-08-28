#!/usr/bin/env python3
"""Mechanical consistency checks for the SCI-MAP v0.1 author artifacts."""

from __future__ import annotations

import re
import hashlib
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PDF = ROOT / "pdf"

SHARED = SRC / "SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex"
COMMON = SRC / "common"
COMMON_MODULES = (
    COMMON / "notation.tex",
    COMMON / "definitions.tex",
    COMMON / "equations.tex",
    COMMON / "assumptions.tex",
    COMMON / "requirements.tex",
    COMMON / "edge_cases.tex",
)
FORMAL_TEX = SRC / "formal-scientific-engineering-contract.tex"
RATIONALE_TEX = SRC / "scientific-rationale.tex"
ENG_TEX = SRC / "engineering-conformance.tex"
FORMAL_PDF = PDF / "SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.1.pdf"
RATIONALE_PDF = PDF / "SCI-MAP-SCIENTIFIC-RATIONALE-v0.1.pdf"
ENG_PDF = PDF / "SCI-MAP-ENGINEERING-CONFORMANCE-v0.1.pdf"
FORMAL_REV_PDF = PDF / "SCI-MAP-v0.1_FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT_r0.7.1-DRAFT.pdf"
RATIONALE_REV_PDF = PDF / "SCI-MAP-v0.1_SCIENCE-TEAM-RATIONALE_r0.7.1-DRAFT.pdf"
ENG_REV_PDF = PDF / "SCI-MAP-v0.1_ENGINEERING-CONFORMANCE_r0.7.1-DRAFT.pdf"
CROSSWALK = ROOT / "CROSSWALK.md"
SCIENTIST_CROSSWALK = ROOT / "SCIENTIST_CROSSWALK_R0.3.md"
OWNER_LEDGER = ROOT / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"
AUTHOR_DECISIONS = ROOT / "AUTHOR_DRAFT_DECISIONS.md"
INCONSISTENCY = ROOT / "CONTRACT_INCONSISTENCY_AND_PROPOSED_AMENDMENT_R0.2.md"
CONSISTENCY = ROOT / "SCIENTIFIC_FORMAL_CONSISTENCY_R0.3.md"
OWNER_REGISTER = SRC / "SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex"
OWNER_REGISTER_GENERATOR = SRC / "generate_owner_decision_register.py"
MANIFEST = ROOT / "SOURCE_MANIFEST_R0.7.md"
MANIFEST_BINDING = ROOT / "SOURCE_MANIFEST_R0.7.sha256"
VAL_ROOT = ROOT.parent.parent / "SCI-VAL" / "v0.1"
VAL_REGISTRY = VAL_ROOT / "PROFILE_REGISTRY.md"
VAL_SOURCE_BINDING = VAL_ROOT / "SOURCE_BINDING_REGISTER.md"
PTC_BOUNDARY = ROOT.parent.parent / "SCI-PTC" / "v0.1" / "SCI-PTC_TO_SCI-MAP_BOUNDARY.md"
MAP_BOUNDARY = ROOT / "SCI-PTC_TO_SCI-MAP_BOUNDARY.md"
AST_FOOTPRINT_BOUNDARY = ROOT / "SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md"
MAP_ADMISSION_PROFILE = ROOT / "SCI-MAP_UPSTREAM_ADMISSION_PROFILE.md"
MAP_COADD_PROFILE = ROOT / "SCI-MAP_COADD_PROFILES_R0.7.md"
PARITY_REPORTS = (
    ROOT / "RATIONALE_FORMAL_ECS_PARITY_R0.7.1.md",
    ROOT / "OWNER_DECISION_PARITY_R0.7.1.md",
)
BYTE_REPORT = ROOT / "BYTE_EQUALITY_AND_SHARED_AUTHORITY_REPORT_R0.7.1.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def markdown_section_bytes(path: Path, heading: str) -> bytes:
    payload = path.read_bytes()
    marker = (heading + "\n").encode()
    start = payload.index(marker)
    level = len(heading) - len(heading.lstrip("#"))
    candidates = []
    for next_level in range(1, level + 1):
        pos = payload.find(b"\n" + (b"#" * next_level) + b" ", start + len(marker))
        if pos >= 0:
            candidates.append(pos + 1)
    end = min(candidates) if candidates else len(payload)
    return payload[start:end]


def markdown_row_bytes(path: Path, row_label: str) -> bytes:
    for line in path.read_bytes().splitlines(keepends=True):
        if line.startswith(("| " + row_label + " |").encode()):
            return line
    raise AssertionError(f"missing source-binding row: {row_label}")


def expect_sequence(found: list[str], prefix: str, count: int) -> None:
    expected = [f"{prefix}{number:03d}" for number in range(1, count + 1)]
    assert found == expected, f"{prefix} sequence mismatch: {found!r}"


def pdf_text(path: Path) -> tuple[PdfReader, list[str]]:
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    assert pages and all(page.strip() for page in pages), f"empty PDF page in {path.name}"
    return reader, pages


shared_wrapper = SHARED.read_text(encoding="utf-8")
shared = "\n".join(path.read_text(encoding="utf-8") for path in COMMON_MODULES)
formal_tex = FORMAL_TEX.read_text(encoding="utf-8")
rationale_tex = RATIONALE_TEX.read_text(encoding="utf-8")
eng_tex = ENG_TEX.read_text(encoding="utf-8")
crosswalk = CROSSWALK.read_text(encoding="utf-8")
scientist_crosswalk = SCIENTIST_CROSSWALK.read_text(encoding="utf-8")
ledger = OWNER_LEDGER.read_text(encoding="utf-8")
author_decisions = AUTHOR_DECISIONS.read_text(encoding="utf-8")
inconsistency = INCONSISTENCY.read_text(encoding="utf-8")
consistency = CONSISTENCY.read_text(encoding="utf-8")
owner_register = OWNER_REGISTER.read_text(encoding="utf-8")

requirements = re.findall(r"\\SCIMapRequirement\{(SCI-MAP-REQ-\d{3})\}", shared)
predictions = re.findall(r"\\SCIMapPrediction\{(SCI-MAP-PRED-\d{3})\}", shared)
expect_sequence(requirements, "SCI-MAP-REQ-", 52)
expect_sequence(predictions, "SCI-MAP-PRED-", 25)
assert len(requirements) == len(set(requirements))
assert len(predictions) == len(set(predictions))

for module in COMMON_MODULES:
    assert module.is_file(), f"missing canonical common module: {module.name}"
    assert f"common/{module.name}" in shared_wrapper

for source in (formal_tex, rationale_tex, eng_tex):
    assert source.count(r"\input{SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex}") == 1

for source in (formal_tex, eng_tex):
    assert source.count(r"\SCIMapRequirements") == 1
    assert source.count(r"\SCIMapPredictions") == 1

assert r"\SCIMapRequirements" not in rationale_tex
assert r"\SCIMapPredictions" not in rationale_tex
assert formal_tex.count(r"\input{SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex}") == 1
register_check = subprocess.run(
    [sys.executable, str(OWNER_REGISTER_GENERATOR), "--check"],
    check=False,
    capture_output=True,
    text=True,
)
assert register_check.returncode == 0, register_check.stderr or register_check.stdout

for forbidden in (
    r"\SCIMapObservationEquations",
    r"\SCIMapOperatorEquations",
    r"\SCIMapFormalWeightEquations",
    r"\SCIMapThresholdEquations",
    r"\SCIMapCoaddEquations",
    r"\begin{align}",
):
    assert forbidden not in eng_tex, f"independent science in engineering source: {forbidden}"

assert r"D_Q^{+}" not in shared, "full-grid pseudoinverse remains in map operator"
for required_support_token in (
    r"\mathcal S_{\rm out}",
    r"J_{\rm out}",
    r"A_{\rm out}",
    r"C_{m,{\rm out}}",
    r"\mathcal S^c_{\rm out}",
    r"J^c_{\rm out}",
    r"B_{\rm out}",
    r"\operatorname{admit}_{\Pi_{\rm eff}}",
    r"\mathcal G",
    r"\sigma_g(i,p)",
    r"\operatorname{pass}_g",
    r"\operatorname{pass}_{\gamma,{\rm QC}}",
    r"H_{{\rm fixed},\Theta}",
    r"\Delta\boldsymbol z_{{\rm PTC}\text{-}{\rm FP}}",
    r"E^{{\rm up},{\rm footprint}}_p",
):
    assert required_support_token in shared, f"missing support-domain token: {required_support_token}"

crosswalk_requirements = re.findall(r"^\| (SCI-MAP-REQ-\d{3}) \|", crosswalk, re.MULTILINE)
crosswalk_predictions = re.findall(r"^\| (SCI-MAP-PRED-\d{3}) \|", crosswalk, re.MULTILINE)
expect_sequence(crosswalk_requirements, "SCI-MAP-REQ-", 52)
expect_sequence(crosswalk_predictions, "SCI-MAP-PRED-", 25)

owner_decisions = list(dict.fromkeys(re.findall(
    r"^\| (SCI-MAP-OD-\d{3}) \| \*\*(?:OPEN|RESOLVED)\*\* \|",
    ledger,
    re.MULTILINE,
)))
open_owner_decisions = re.findall(
    r"^\| (SCI-MAP-OD-\d{3}) \| \*\*OPEN\*\* \|", ledger, re.MULTILINE
)
expect_sequence(owner_decisions, "SCI-MAP-OD-", 9)
assert len(open_owner_decisions) == 8
assert "SCI-MAP-OD-008" not in open_owner_decisions
for decision in owner_decisions:
    assert decision.replace("SCI-MAP-OD", "OD") in author_decisions
    assert owner_register.count(decision) == 1, f"{decision} register coverage"

assert "SCI-MAP-CI-001" in inconsistency
assert "dimensionless" in inconsistency
assert "RESOLVED" in inconsistency
assert "RESOLVED" in ledger
assert "SCI-MAP-CI-001" in consistency
assert "dimensionless" in shared
assert r"\operatorname{unit\_status}(c)" not in shared
assert "SCI-MAP-CI-001" in rationale_tex
assert r"\providecommand{\SCIMapUpstreamProfile}" in shared
assert r"\providecommand{\SCIMapCoaddProfile}" in shared
assert r"\SCIMapPTCMapBoundaryId" in formal_tex
assert r"\providecommand{\SCIMapASTFootprintBoundaryId}" in shared
assert r"A_{{\rm MAP},\Pi}\equiv A_{\rm out}" in shared
assert "PTC+MAP re-resolved" in shared
assert "whole-chain RTC-to-CAL-to-PTC-to-MAP" in shared
assert "eight map-local decisions remain open" in formal_tex.lower()
assert "exact owner-admitted numerical \\code{coverage_cut} state/value" in rationale_tex
assert "operatorname{request}_{\\gamma,{\\rm QC}}" in shared
assert "operatorname{applicability}_{\\gamma,{\\rm QC}}" in shared
assert "operatorname{eligibility}_{\\gamma,{\\rm QC}}" in shared
assert "operatorname{realization}_{\\gamma,{\\rm QC}}" in shared
assert "(r,a,e" not in shared and "(r, a, e" not in shared
assert r"upstream\_eligible\_original\_footprint\_exposure" in shared
assert r"retained\_original\_footprint\_exposure" in shared
assert "ordinary nonpassing field value is not thereby a structural failure" in shared
assert "structural source- or profile-binding failure is recorded only" in shared.lower()
for stale in (
    "raw-invalid", "raw validity", "raw-valid", "raw parentage",
    "raw bundle", "immutable raw parent", "raw accumulators", "C_x",
    r"Q_\gamma", "Any The PTC", "nine MAP decisions remain unresolved",
    r"\Delta\Sigma", "full-chain re-resolved", "SCI-PTC_TO_SCI-MAPv0.1/r0.1",
):
    assert stale.lower() not in (shared + formal_tex + rationale_tex + eng_tex + crosswalk + ledger).lower(), (
        f"stale r0.6 term remains: {stale}"
    )
for decision in ("OD-001", "OD-007", "OD-008", "OD-009"):
    assert decision in rationale_tex, f"compact decision coverage missing: {decision}"
for authority_range in (
    "REQ-001--010",
    "REQ-011--018",
    "REQ-019--024",
    "REQ-025--035",
    "REQ-036--042",
    "REQ-043--048",
    "REQ-049--052",
    "PRED-001--004",
    "PRED-005--009",
    "PRED-010--013",
    "PRED-014--019",
    "PRED-020--025",
):
    assert authority_range in scientist_crosswalk, f"scientist crosswalk gap: {authority_range}"

assert MAP_BOUNDARY.read_bytes() == PTC_BOUNDARY.read_bytes(), "PTC/MAP boundary byte mismatch"
for exact_path in (
    MAP_BOUNDARY,
    PTC_BOUNDARY,
    AST_FOOTPRINT_BOUNDARY,
    MAP_ADMISSION_PROFILE,
    MAP_COADD_PROFILE,
    VAL_REGISTRY,
    VAL_SOURCE_BINDING,
    OWNER_LEDGER,
    *PARITY_REPORTS,
    BYTE_REPORT,
):
    assert exact_path.is_file(), f"missing manifest-bound artifact: {exact_path}"

manifest_text = MANIFEST.read_text(encoding="utf-8")
manifest_digest = sha256_file(MANIFEST)
binding_fields = MANIFEST_BINDING.read_text(encoding="utf-8").strip().split()
assert binding_fields[0] == manifest_digest, "source-manifest companion digest mismatch"
assert binding_fields[-1] == "SOURCE_MANIFEST_R0.7.md", "source-manifest companion path mismatch"

for label, exact_path in (
    ("MAP PTC-to-MAP boundary copy", MAP_BOUNDARY),
    ("PTC PTC-to-MAP boundary copy", PTC_BOUNDARY),
    ("Original-footprint-coordinate boundary", AST_FOOTPRINT_BOUNDARY),
    ("MAP upstream-admission profile", MAP_ADMISSION_PROFILE),
    ("MAP coadd profiles", MAP_COADD_PROFILE),
    ("SCI-VAL Profile Registry", VAL_REGISTRY),
    ("SCI-VAL Source-Binding Register", VAL_SOURCE_BINDING),
    ("Owner-decision ledger", OWNER_LEDGER),
    ("Rationale/formal/ECS parity report", PARITY_REPORTS[0]),
    ("Owner-decision parity report", PARITY_REPORTS[1]),
    ("Byte-equality/shared-authority report", BYTE_REPORT),
):
    matching_lines = [line for line in manifest_text.splitlines() if line.startswith(f"| {label}")]
    assert len(matching_lines) == 1 and f"`{sha256_file(exact_path)}`" in matching_lines[0], (
        f"manifest digest missing or stale: {label}"
    )

shared_aggregate = b"".join(path.read_bytes() for path in (SHARED, *COMMON_MODULES))
assert f"| Shared r0.7.1 authority aggregate | `{sha256_bytes(shared_aggregate)}` |" in manifest_text

for label, heading in (
    ("Registry record `SCI-MAP:map_upstream_admission@2`", "### `SCI-MAP:map_upstream_admission@2`"),
    ("Registry record `SCI-MAP:observation_coadd_admission@1`", "### `SCI-MAP:observation_coadd_admission@1`"),
):
    digest = sha256_bytes(markdown_section_bytes(VAL_REGISTRY, heading))
    assert f"| {label} | `{digest}` |" in manifest_text, f"stale {label} digest"

for row_label in (
    "SCI-ALIGN",
    "SCI-AST",
    "SCI-RTC",
    "SCI-CAL",
    "SCI-PTC",
    "Tune/readout and telescope inputs",
    "SCI-MAP",
):
    digest = sha256_bytes(markdown_row_bytes(VAL_SOURCE_BINDING, row_label))
    assert f"| SCI-VAL source-binding row `{row_label}` | `{digest}` |" in manifest_text

for source_text in (shared, formal_tex, rationale_tex, eng_tex):
    assert "SCI-PTC_TO_SCI-MAPv0.1/r0.1" not in source_text
    assert "SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATEv0.1/r0.1" not in source_text

formal_reader, formal_pages = pdf_text(FORMAL_PDF)
rationale_reader, rationale_pages = pdf_text(RATIONALE_PDF)
eng_reader, eng_pages = pdf_text(ENG_PDF)
for canonical, revisioned in (
    (FORMAL_PDF, FORMAL_REV_PDF),
    (RATIONALE_PDF, RATIONALE_REV_PDF),
    (ENG_PDF, ENG_REV_PDF),
):
    assert revisioned.is_file(), f"missing revision-bearing PDF: {revisioned.name}"
    assert canonical.read_bytes() == revisioned.read_bytes(), (
        f"stable and revision-bearing PDFs differ: {canonical.name}"
    )
for reader, pages, label in (
    (formal_reader, formal_pages, "formal"),
    (eng_reader, eng_pages, "engineering"),
):
    joined = "\n".join(pages)
    assert "v0.1" in joined, f"contract version missing in {label} PDF"
    for requirement in requirements:
        assert joined.count(requirement) == 1, f"{requirement} coverage in {label} PDF"
    for prediction in predictions:
        assert joined.count(prediction) == 1, f"{prediction} coverage in {label} PDF"

formal_joined = "\n".join(formal_pages)
for decision in owner_decisions:
    assert decision in formal_joined, f"{decision} missing from formal PDF"
assert "Owner-decision register" in formal_joined
assert "Requirement and crosswalk summary" in formal_joined

appendix_page = next(
    index + 1
    for index, text in enumerate(rationale_pages)
    if any(
        line.strip() == "A Compact rationale-to-contract crosswalk"
        for line in text.splitlines()
    )
)
narrative_pages = appendix_page - 1
assert 8 <= narrative_pages <= 12, f"main narrative is {narrative_pages} pages"
rationale_joined = "\n".join(rationale_pages)
for required_text in (
    "SCI-MAP-CI-001",
    "dimensionless",
    "one_hot_containing_pixel",
    "map_upstream_admission",
    "OD-008",
    "OD-009",
    "0.1 arcsec",
    "Not assessed",
):
    assert required_text in rationale_joined, f"rationale content missing: {required_text}"
for requirement in requirements:
    assert requirement not in rationale_joined, f"full requirement leaked into rationale: {requirement}"
for prediction in predictions:
    assert prediction not in rationale_joined, f"full prediction leaked into rationale: {prediction}"

warning_pattern = re.compile(
    r"LaTeX Warning|Package .* Warning|Overfull \\hbox|Underfull \\hbox|"
    r"undefined references|multiply defined",
    re.IGNORECASE,
)
for log in PDF.glob("*.log"):
    matches = warning_pattern.findall(log.read_text(encoding="utf-8", errors="replace"))
    assert not matches, f"LaTeX warnings in {log.name}: {matches}"

print(f"requirements={len(requirements)} sequential_unique=PASS")
print(f"predictions={len(predictions)} sequential_unique=PASS")
print("crosswalk_requirement_coverage=PASS")
print("crosswalk_prediction_coverage=PASS")
print(f"owner_decision_ids={len(owner_decisions)}")
print(f"open_owner_decisions={len(open_owner_decisions)}")
print("owner_decision_register_exact_check=PASS")
print("canonical_common_module_layout=PASS")
print("support_authorized_operator_domain_check=PASS")
print("engineering_independent_science_check=PASS")
print(f"formal_pdf_pages={len(formal_reader.pages)}")
print(f"rationale_pdf_pages={len(rationale_reader.pages)}")
print(f"rationale_main_narrative_pages={narrative_pages}")
print(f"engineering_pdf_pages={len(eng_reader.pages)}")
print("formal_engineering_shared_requirement_prediction_coverage=PASS")
print("rationale_inventory_separation=PASS")
print("dimensional_inconsistency_record=PASS")
print("ci_001_owner_resolution_and_amendment=PASS")
print("revision_bearing_pdf_aliases=PASS")
print("manifest_external_sha256_binding=PASS")
print("registry_record_and_source_row_binding=PASS")
print("latex_warning_check=PASS")
