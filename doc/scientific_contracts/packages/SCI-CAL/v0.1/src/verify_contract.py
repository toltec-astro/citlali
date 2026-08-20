#!/usr/bin/env python3
"""Mechanical checks for the SCI-CAL v0.1 r0.5/r0.4 document pair."""

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
        == "edeed1dbdd2706b9a518abc95f3369db7c1c83002fb7cb34ba02b8054020ca24",
        "science rationale source differs from reviewed r0.5 source",
    )
    require(
        digest(SRC / "engineering-conformance.tex")
        == "3dc656da692bfaa5802f1578d5ae022c45576c22bc4d13c0f5419257449b1ef3",
        "engineering source differs from reviewed r0.4 source",
    )
    common_hashes = {
        "notation.tex": "c5d4ec103d6a01eaec15bcb816d019d78b6aaf8700998e563e0849122421f4db",
        "definitions.tex": "2a9c91f485ea7d41ba6d5b13c77f77b8314612da4bc4b59eb1228235374b71b5",
        "assumptions.tex": "6da85c4a44d5b20b222f5796dae8922594f1b1d043a9ac993f5fb6f12059eea9",
        "equations.tex": "b8027f5e0b787a95708be6cc51018bb993d32f4863c34c4b1b55dd71bd2d3322",
        "requirements.tex": "ff4b4f924ecd0c21e7a131ca823396b578d80ffde6bd91ab9e7e63bf946e6218",
        "edge_cases.tex": "a0c6b6be73cd0ef8e3b0655dc842eef83b2ca3037c48b8b2651101570f645556",
    }
    for name, expected in common_hashes.items():
        require(digest(COMMON / name) == expected, f"shared authority differs: {name}")
    science_pdf = PKG / "pdf" / "SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf"
    require(
        digest(science_pdf)
        == "d4024db374f361854060ef4939796ae8c2fec910a33935852f832384f7d692a3",
        "canonical science PDF differs from reviewed r0.5 artifact",
    )
    engineering_pdf = PKG / "pdf" / "SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf"
    require(
        digest(engineering_pdf)
        == "994a641b21c0f4af0701c3eb5c09d86669bb7943b0be02d2080024d00331ac0d",
        "canonical engineering PDF differs from reviewed r0.4 artifact",
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
        r"\newcommand{\DocRevision}{Engineering revision r0.4}" in engineering,
        "engineering revision stamp",
    )
    science = text(SRC / "scientific-rationale.tex")
    require("Science-Team Rationale r0.5" in science, "science revision stamp")
    require(
        r"\input{scientific-crosswalk-r0.5.tex}" in science,
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
        "pipeline boundary/order",
        "cal-before-ptc",
        "closure",
        "associated-pointing",
    ):
        require(marker in combined, f"alignment marker missing: {marker}")

    crosswalk = text(PKG / "CROSSWALK.md")
    crosswalk_ids = re.findall(r"^\| SCI-CAL-REQ-(\d{3}) \|", crosswalk, re.MULTILINE)
    require_sequence(crosswalk_ids, 50, "crosswalk requirement")

    pdfs = (
        ("science", science_pdf, 14),
        ("engineering", engineering_pdf, 26),
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
        "Science-TeamRationaler0.5" in extracted["science"],
        "science PDF revision",
    )
    require(
        "Engineeringrevisionr0.4" in engineering_pdf_text,
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

    print("PASS: reviewed r0.5/r0.4 sources, shared authority, and canonical PDF hashes")
    print("PASS: assumptions=11 requirements=50 edge_predictions=30")
    print("PASS: crosswalk requirements=50")
    print("PASS: science revision=r0.5, engineering revision=r0.4, and Q01--Q09 decided")
    print("PASS: ownership, lineage, broadband, order, closure, and transfer markers")
    print("PASS: science PDF=14 pages, engineering PDF=26 pages, corrected terminology, and ID coverage")


if __name__ == "__main__":
    main()
