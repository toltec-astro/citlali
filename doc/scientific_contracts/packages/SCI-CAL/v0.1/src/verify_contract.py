#!/usr/bin/env python3
"""Mechanical checks for the frozen SCI-CAL v0.1 r0.5/r0.4 authority."""

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
    package_hashes = {
        "README.md": "c6db96f6011d8c6f3c8f8ac0b217e650fac39df6b6ee2422c8a681e5eb3be067",
        "CROSSWALK.md": "d8e99a105f832d21f536137b931ec6ae84e41a10f66df74926bfbcb7814e432a",
        "DECISION_LOG.md": "1aafd3989123a015e23893856efde37745013743fbb1f5573db95887a8eda9a5",
        "SCIENTIFIC_OWNER_DECISIONS_R0.5.md": "32810d617c0df166415c1fe172054dc32ea89e3489137ff0ee1ed29280410506",
        "SCIENTIFIC_OWNER_DECISION_LEDGER.md": "39b0f3b495e61699090d30eb497d09dcb003ab5c6a40a82d0c46178b79386ca1",
        "SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.5.md": "e649cb6a7d84b8ff1e098a7e1b42b49722381755d49e70dc1eea961d7dd2a37f",
        "SCIENTIFIC_ENGINEERING_R0.5_R0.4_BUILD_REVIEW.md": "e47599858e1b0e95b09fa78b695013bf9c9be79887ac7dda62a784bbf32548fd",
        "SCIENTIFIC_OWNER_FREEZE_R0.5.md": "413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22",
        "pdf/README.md": "24245914000a9bae674a079cece8445defb94097a0e053b29e78d446a766f800",
    }
    for name, expected in package_hashes.items():
        require(digest(PKG / name) == expected, f"frozen package artifact differs: {name}")

    require(
        digest(SRC / "scientific-rationale.tex")
        == "f780cef579cb39ac1ed748f021a0024d9f1576d7960fe3b4363557c10bbff318",
        "science rationale source differs from frozen r0.5 source",
    )
    require(
        digest(SRC / "engineering-conformance.tex")
        == "4d807192adabf0dc0fc8dc20505c528eca865d484dd5891e37cb5913bf138f7a",
        "engineering source differs from frozen r0.4 source",
    )
    common_hashes = {
        "preamble.tex": "beb69a3e05260db7561be87aca04f6bf3eee4ea36f80411d1b0baddb5e0ac7a9",
        "notation.tex": "c5d4ec103d6a01eaec15bcb816d019d78b6aaf8700998e563e0849122421f4db",
        "definitions.tex": "2a9c91f485ea7d41ba6d5b13c77f77b8314612da4bc4b59eb1228235374b71b5",
        "assumptions.tex": "6da85c4a44d5b20b222f5796dae8922594f1b1d043a9ac993f5fb6f12059eea9",
        "equations.tex": "b8027f5e0b787a95708be6cc51018bb993d32f4863c34c4b1b55dd71bd2d3322",
        "requirements.tex": "80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74",
        "edge_cases.tex": "45ff8dc1befe04216f9e93cb8f2713f2bfdb3799459714e71187d7202dc084c0",
    }
    for name, expected in common_hashes.items():
        require(digest(COMMON / name) == expected, f"shared authority differs: {name}")
    science_pdf = PKG / "pdf" / "SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf"
    require(
        digest(science_pdf)
        == "fa6a11a359bcc4f54cb75ab5057ba56cf76fa8eef8d28ac3bcb7954963a12034",
        "canonical science PDF differs from frozen r0.5 artifact",
    )
    engineering_pdf = PKG / "pdf" / "SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf"
    require(
        digest(engineering_pdf)
        == "07dd88895b21ee02eca611bba8d1adcf90f9c37439f791437c4f46e351700101",
        "canonical engineering PDF differs from frozen r0.4 artifact",
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
    require("FROZEN" in science and "DRAFT" not in science, "science frozen status")
    require(
        "FROZEN" in engineering and "DRAFT" not in engineering,
        "engineering frozen status",
    )
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
        require(not reader.is_encrypted, f"{label} PDF is encrypted")
        require("/AcroForm" not in reader.trailer["/Root"], f"{label} PDF has forms")
        names = reader.trailer["/Root"].get("/Names", {})
        require("/JavaScript" not in names, f"{label} PDF has JavaScript")
        for page in reader.pages:
            box = page.mediabox
            require(
                float(box.width) == 612.0 and float(box.height) == 792.0,
                f"{label} PDF has a non-letter page",
            )
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
    require("FROZEN" in extracted["science"], "science PDF frozen status")
    require("FROZEN" in engineering_pdf_text, "engineering PDF frozen status")
    require("DRAFT" not in extracted["science"], "science PDF retains draft status")
    require("DRAFT" not in engineering_pdf_text, "engineering PDF retains draft status")
    require(
        "Achieved-performanceacceptance" in engineering_pdf_text,
        "engineering PDF achieved-performance acceptance boundary",
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

    print("PASS: frozen r0.5/r0.4 sources, package artifacts, shared authority, and PDF hashes")
    print("PASS: assumptions=11 requirements=50 edge_predictions=30")
    print("PASS: crosswalk requirements=50")
    print("PASS: science revision=r0.5, engineering revision=r0.4, and Q01--Q09 decided")
    print("PASS: ownership, lineage, broadband, order, closure, and transfer markers")
    print("PASS: science PDF=14 pages, engineering PDF=26 pages, corrected terminology, and ID coverage")


if __name__ == "__main__":
    main()
