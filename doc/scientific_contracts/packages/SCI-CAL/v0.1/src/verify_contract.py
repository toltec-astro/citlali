#!/usr/bin/env python3
"""Mechanical checks for the SCI-CAL v0.1 r0.4/r0.3 document pair."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader


SRC = Path(__file__).resolve().parent
PKG = SRC.parent
COMMON = SRC / "common"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sequence(found: list[str], count: int, label: str) -> None:
    expected = [f"{number:03d}" for number in range(1, count + 1)]
    require(found == expected, f"{label} IDs differ: {found}")


def main() -> None:
    require(
        digest(SRC / "scientific-rationale.tex")
        == "423f1794b0d6395ee48dc930aa357e68b2a21d086d5d0884215d1ded0499625e",
        "science rationale source differs from reviewed r0.4 source",
    )
    require(
        digest(SRC / "engineering-conformance.tex")
        == "8bbef22647c091b5005322ef7842372168680275c97601ba53fc8fd1e413effa",
        "engineering source differs from reviewed r0.3 source",
    )
    science_pdf = PKG / "pdf" / "SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf"
    require(
        digest(science_pdf)
        == "0ff32bbd63b42fca3cd8273ba9f8297213402ec6fa920ccd63afdddb9aaa09b7",
        "canonical science PDF differs from reviewed r0.4 artifact",
    )
    engineering_pdf = PKG / "pdf" / "SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf"
    require(
        digest(engineering_pdf)
        == "3795bcf13c18f707b4935d07eab465d1c65ed241d7227a4f036f80c8e9a3af70",
        "canonical engineering PDF differs from reviewed r0.3 artifact",
    )

    assumptions = re.findall(
        r"\\item\[\\Assump\{(\d{3})\}", text(COMMON / "assumptions.tex")
    )
    requirements = re.findall(
        r"\\item\[\\Req\{(\d{3})\}", text(COMMON / "requirements.tex")
    )
    edges = re.findall(r"\\Edge\{(\d{3})\}", text(COMMON / "edge_cases.tex"))
    require_sequence(assumptions, 11, "assumption")
    require_sequence(requirements, 50, "requirement")
    require_sequence(edges, 30, "edge")

    engineering = text(SRC / "engineering-conformance.tex")
    require(
        r"\newcommand{\DocRevision}{Engineering revision r0.3}" in engineering,
        "engineering revision stamp",
    )
    science = text(SRC / "scientific-rationale.tex")
    require("Science-Team Rationale r0.4" in science, "science revision stamp")
    require(
        r"\input{scientific-crosswalk-r0.4.tex}" in science,
        "science crosswalk revision",
    )
    expected_inputs = [
        rf"\input{{common/{name}.tex}}"
        for name in (
            "preamble",
            "notation",
            "definitions",
            "assumptions",
            "equations",
            "requirements",
            "edge_cases",
        )
    ]
    require(
        re.findall(r"\\input\{common/[^}]+\}", engineering) == expected_inputs,
        "engineering shared-authority inclusion order",
    )

    for question in range(1, 10):
        decision = f"SCI-CAL-OWNER-Q{question:02d}"
        require(decision in engineering, f"{decision} missing from engineering wrapper")
        require(
            decision in text(PKG / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"),
            f"{decision} missing from owner ledger",
        )

    combined = "\n".join(
        text(path)
        for path in (
            SRC / "engineering-conformance.tex",
            COMMON / "definitions.tex",
            COMMON / "assumptions.tex",
            COMMON / "requirements.tex",
            COMMON / "edge_cases.tex",
        )
    ).lower()
    for marker in (
        "producer--transformer--delivery--consumer",
        "source apt",
        "observation-specific child apt",
        "photometric-convention",
        "pipeline-order",
        "resolves q06 only",
    ):
        require(marker in combined, f"alignment marker missing: {marker}")

    crosswalk = text(PKG / "CROSSWALK.md")
    crosswalk_ids = re.findall(r"^\| SCI-CAL-REQ-(\d{3}) \|", crosswalk, re.MULTILINE)
    require_sequence(crosswalk_ids, 50, "crosswalk requirement")

    pdfs = (
        ("science", science_pdf, 14),
        ("engineering", engineering_pdf, 25),
    )
    extracted: dict[str, str] = {}
    for label, path, page_count in pdfs:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        require(
            len(pages) == page_count and all(page.strip() for page in pages),
            f"{label} PDF page count or empty-page check",
        )
        extracted[label] = re.sub(r"\s+", "", "\n".join(pages))

    engineering_pdf_text = extracted["engineering"]
    require(
        "Science-TeamRationaler0.4" in extracted["science"],
        "science PDF revision",
    )
    require(
        "Engineeringrevisionr0.3" in engineering_pdf_text,
        "engineering PDF revision",
    )
    require(
        "TolTECAdelivery" in extracted["science"],
        "science PDF delivery boundary",
    )
    require(
        "pointing-correctionpremiseshowntobeabsent" in engineering_pdf_text,
        "engineering PDF not-applicable example",
    )
    require(
        "CitlaliScientificConventions" not in extracted["science"],
        "science PDF contains out-of-packet direct citation",
    )
    require(
        "RTC" not in engineering_pdf_text,
        "engineering PDF contains undefined RTC acronym",
    )

    for stem, count in (
        ("SCI-CAL-ASM-", 11),
        ("SCI-CAL-REQ-", 50),
        ("SCI-CAL-EDGE-", 30),
    ):
        for number in range(1, count + 1):
            identifier = f"{stem}{number:03d}"
            require(identifier in engineering_pdf_text, f"PDF missing {identifier}")
    for number in range(1, 10):
        identifier = f"SCI-CAL-OWNER-Q{number:02d}"
        require(identifier in engineering_pdf_text, f"PDF missing {identifier}")

    print("PASS: reviewed r0.4/r0.3 sources and canonical PDF hashes")
    print("PASS: assumptions=11 requirements=50 edge_predictions=30")
    print("PASS: crosswalk requirements=50")
    print("PASS: science revision=r0.4, engineering revision=r0.3, and Q01--Q09 carried")
    print("PASS: ownership, lineage, broadband, order, and Q06-only markers")
    print("PASS: science PDF=14 pages, engineering PDF=25 pages, corrected terminology, and ID coverage")


if __name__ == "__main__":
    main()
