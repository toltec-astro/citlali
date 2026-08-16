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
SCI_TEX = SRC / "SCI-MAP-v0.1_SCIENTIFIC_RATIONALE_AND_CONTRACT_r0.1.tex"
ENG_TEX = SRC / "SCI-MAP-v0.1_ENGINEERING_CONFORMANCE_SPECIFICATION_r0.1.tex"
SCI_PDF = PDF / "SCI-MAP-v0.1_SCIENTIFIC_RATIONALE_AND_CONTRACT_r0.1.pdf"
ENG_PDF = PDF / "SCI-MAP-v0.1_ENGINEERING_CONFORMANCE_SPECIFICATION_r0.1.pdf"
CROSSWALK = ROOT / "CROSSWALK.md"
OWNER_LEDGER = ROOT / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"
AUTHOR_DECISIONS = ROOT / "AUTHOR_DRAFT_DECISIONS.md"
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


shared = SHARED.read_text(encoding="utf-8")
sci_tex = SCI_TEX.read_text(encoding="utf-8")
eng_tex = ENG_TEX.read_text(encoding="utf-8")
crosswalk = CROSSWALK.read_text(encoding="utf-8")
ledger = OWNER_LEDGER.read_text(encoding="utf-8")
author_decisions = AUTHOR_DECISIONS.read_text(encoding="utf-8")
owner_register = OWNER_REGISTER.read_text(encoding="utf-8")

requirements = re.findall(r"\\SCIMapRequirement\{(SCI-MAP-REQ-\d{3})\}", shared)
predictions = re.findall(r"\\SCIMapPrediction\{(SCI-MAP-PRED-\d{3})\}", shared)
expect_sequence(requirements, "SCI-MAP-REQ-", 52)
expect_sequence(predictions, "SCI-MAP-PRED-", 25)
assert len(requirements) == len(set(requirements))
assert len(predictions) == len(set(predictions))

for source in (sci_tex, eng_tex):
    assert source.count(r"\input{SCI-MAP-v0.1_SHARED_AUTHORITY_r0.1.tex}") == 1
    assert source.count(r"\SCIMapRequirements") == 1
    assert source.count(r"\SCIMapPredictions") == 1

assert sci_tex.count(r"\input{SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex}") == 1
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
expect_sequence(owner_decisions, "SCI-MAP-OD-", 7)
for decision in owner_decisions:
    assert decision.replace("SCI-MAP-OD", "OD") in author_decisions
    assert owner_register.count(decision) == 1, f"{decision} register coverage"

sci_reader, sci_pages = pdf_text(SCI_PDF)
eng_reader, eng_pages = pdf_text(ENG_PDF)
for reader, pages, label in (
    (sci_reader, sci_pages, "scientific"),
    (eng_reader, eng_pages, "engineering"),
):
    joined = "\n".join(pages)
    assert "v0.1" in joined and "r0.1" in joined, f"version metadata missing in {label} PDF"
    for requirement in requirements:
        assert joined.count(requirement) == 1, f"{requirement} coverage in {label} PDF"
    for prediction in predictions:
        assert joined.count(prediction) == 1, f"{prediction} coverage in {label} PDF"

sci_joined = "\n".join(sci_pages)
for decision in owner_decisions:
    assert decision in sci_joined, f"{decision} missing from scientific PDF"
assert "Owner-decision register" in sci_joined
assert "Requirement and crosswalk summary" in sci_joined

appendix_page = next(
    index + 1
    for index, text in enumerate(sci_pages)
    if any(line.strip() == "A Owner-decision register" for line in text.splitlines())
)
narrative_pages = appendix_page - 1
assert 8 <= narrative_pages <= 12, f"main narrative is {narrative_pages} pages"

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
print("support_authorized_operator_domain_check=PASS")
print("engineering_independent_science_check=PASS")
print(f"scientific_pdf_pages={len(sci_reader.pages)}")
print(f"scientific_main_narrative_pages={narrative_pages}")
print(f"engineering_pdf_pages={len(eng_reader.pages)}")
print("pdf_shared_requirement_prediction_coverage=PASS")
print("latex_warning_check=PASS")
