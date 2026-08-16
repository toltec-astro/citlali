#!/usr/bin/env python3
"""Mechanical consistency checks for the SCI-MAP v0.1 author artifacts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PDF = ROOT / "pdf"

SHARED = SRC / "SCI-MAP-v0.1_SHARED_AUTHORITY_r0.1.tex"
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
CROSSWALK = ROOT / "CROSSWALK.md"
SCIENTIST_CROSSWALK = ROOT / "SCIENTIST_CROSSWALK_R0.2.md"
OWNER_LEDGER = ROOT / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"
AUTHOR_DECISIONS = ROOT / "AUTHOR_DRAFT_DECISIONS.md"
INCONSISTENCY = ROOT / "CONTRACT_INCONSISTENCY_AND_PROPOSED_AMENDMENT_R0.2.md"
CONSISTENCY = ROOT / "SCIENTIFIC_FORMAL_CONSISTENCY_R0.2.md"
OWNER_REGISTER = SRC / "SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex"
OWNER_REGISTER_GENERATOR = SRC / "generate_owner_decision_register.py"


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
    assert source.count(r"\input{SCI-MAP-v0.1_SHARED_AUTHORITY_r0.1.tex}") == 1

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
    r"C_{x,{\rm out}}",
    r"\mathcal S^c_{\rm out}",
    r"J^c_{\rm out}",
    r"B_{\rm out}",
    r"\operatorname{admit}_{\Pi_{\rm eff}}",
):
    assert required_support_token in shared, f"missing support-domain token: {required_support_token}"

crosswalk_requirements = re.findall(r"^\| (SCI-MAP-REQ-\d{3}) \|", crosswalk, re.MULTILINE)
crosswalk_predictions = re.findall(r"^\| (SCI-MAP-PRED-\d{3}) \|", crosswalk, re.MULTILINE)
expect_sequence(crosswalk_requirements, "SCI-MAP-REQ-", 52)
expect_sequence(crosswalk_predictions, "SCI-MAP-PRED-", 25)

owner_decisions = re.findall(
    r"^\| (SCI-MAP-OD-\d{3}) \| \*\*OPEN\*\* \|", ledger, re.MULTILINE
)
expect_sequence(owner_decisions, "SCI-MAP-OD-", 9)
for decision in owner_decisions:
    assert decision.replace("SCI-MAP-OD", "OD") in author_decisions
    assert owner_register.count(decision) == 1, f"{decision} register coverage"

assert "SCI-MAP-CI-001" in inconsistency
assert "dimensionless" in inconsistency
assert "owner approval required" in inconsistency.lower()
assert "normative clauses remain unchanged" in consistency
assert "SCI-MAP-CI-001" in rationale_tex
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

formal_reader, formal_pages = pdf_text(FORMAL_PDF)
rationale_reader, rationale_pages = pdf_text(RATIONALE_PDF)
eng_reader, eng_pages = pdf_text(ENG_PDF)
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
assert 8 <= narrative_pages <= 10, f"main narrative is {narrative_pages} pages"
rationale_joined = "\n".join(rationale_pages)
for required_text in (
    "SCI-MAP-CI-001",
    "Q = 1",
    "variance",
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
print(f"open_owner_decisions={len(owner_decisions)}")
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
print("latex_warning_check=PASS")
