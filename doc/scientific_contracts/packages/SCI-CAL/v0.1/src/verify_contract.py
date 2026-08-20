#!/usr/bin/env python3
"""Mechanical checks for the SCI-CAL v0.1 r0.3/r0.2 document pair."""

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
        == "ea63e1260af3f897c9572639ed0c418561d35e14781e42019a8cea71b58f5374",
        "science rationale source changed during engineering repair",
    )
    require(
        digest(PKG / "pdf" / "SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf")
        == "075efafcbe4f0f3897be3bb88604e00a575d5d623a2eaf78a11d25ed7c3284d3",
        "canonical science rationale PDF changed during engineering repair",
    )
    engineering_pdf = PKG / "pdf" / "SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf"
    require(
        digest(engineering_pdf)
        == "7caa69eb4ca3e0da99ddf23959e8c9ccbaae9e607cdd5eeaff30a4cd1097c30d",
        "canonical engineering PDF differs from reviewed r0.2 artifact",
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
    require(r"\newcommand{\DocRevision}{Engineering revision r0.2}" in engineering,
            "engineering revision stamp")
    expected_inputs = [
        rf"\input{{common/{name}.tex}}"
        for name in (
            "preamble", "notation", "definitions", "assumptions", "equations",
            "requirements", "edge_cases",
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

    consistency = text(PKG / "SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.3.md")
    require("corrected consistency candidate" in consistency, "consistency status")
    require("fresh implementation-blind" in consistency,
            "remaining review gate")

    reader = PdfReader(str(engineering_pdf))
    pages = [page.extract_text() or "" for page in reader.pages]
    require(len(pages) == 25 and all(page.strip() for page in pages),
            "engineering PDF page count or empty-page check")
    pdf_text = "\n".join(pages)
    compact_pdf_text = re.sub(r"\s+", "", pdf_text)
    for stem, count in (
        ("SCI-CAL-ASM-", 11),
        ("SCI-CAL-REQ-", 50),
        ("SCI-CAL-EDGE-", 30),
    ):
        for number in range(1, count + 1):
            identifier = f"{stem}{number:03d}"
            require(identifier in compact_pdf_text, f"PDF missing {identifier}")
    for number in range(1, 10):
        identifier = f"SCI-CAL-OWNER-Q{number:02d}"
        require(identifier in compact_pdf_text, f"PDF missing {identifier}")

    print("PASS: science rationale source and canonical PDF unchanged")
    print("PASS: assumptions=11 requirements=50 edge_predictions=30")
    print("PASS: crosswalk requirements=50")
    print("PASS: engineering revision=r0.2 and Q01--Q09 carried")
    print("PASS: ownership, lineage, broadband, order, and Q06-only markers")
    print("PASS: reviewed engineering PDF hash, 25 nonempty pages, and ID coverage")


if __name__ == "__main__":
    main()
