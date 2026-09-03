#!/usr/bin/env python3
"""Mechanical checks for frozen SCI-PTC v0.1/r0.5 authority.

This helper reads only the approved author packet and package deliverables.
It does not inspect implementation or claim conformity, validation,
performance, production readiness, MAP availability, or finding closure.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader


SRC = Path(__file__).resolve().parent
PKG = SRC.parent
COMMON = SRC / "common"
PDF = PKG / "pdf"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence(found: list[str], stem: str, count: int) -> None:
    expected = [f"{stem}{number:03d}" for number in range(1, count + 1)]
    require(found == expected, f"{stem} sequence or count differs")


def pdf_text(path: Path) -> tuple[PdfReader, str]:
    reader = PdfReader(path)
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    extracted = extracted.translate(str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff"}))
    return reader, extracted


def main() -> None:
    expected_hashes = {
        PKG / "SCOPE_BRIEF.md":
            "8aa05920589b67cb7634003f466161769101e3013cf82573260e94b257532bed",
        PKG / "AUTHOR_SUPERSESSION_COVER.md":
            "2a13d3984c2334ccd1886021d2d869bb71363abd3a06bb7f9fbf536614d9ee3e",
        PKG / "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md":
            "568b35ff3da16c8ed6902d3bb0d845e01eec38e5374c6e89e75823f1f8ecabe6",
        PKG / "AUTHOR_METHOD_REFERENCE_BOUNDARY.md":
            "d5d33180c9e40958237916ec6dd98ba655d161bc984a3b694197a1a90d78be61",
    }
    for path, expected in expected_hashes.items():
        require(digest(path) == expected, f"approved packet hash changed: {path.name}")

    retained_core = subprocess.run(
        [
            "git", "show",
            "01ee247461d6c19bc4db81ccac4fec21af162c88:"
            "doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex",
        ],
        cwd=PKG,
        check=True,
        capture_output=True,
    ).stdout
    require(
        hashlib.sha256(retained_core).hexdigest()
        == "82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2",
        "retained-core hash changed",
    )

    definitions = re.findall(
        r"\\PTCDefinition\{(\d{3})\}", text(COMMON / "definitions.tex")
    )
    assumptions = re.findall(
        r"\\PTCAssumption\{(\d{3})\}", text(COMMON / "assumptions.tex")
    )
    requirements = re.findall(
        r"\\PTCRequirement\{(\d{3})\}", text(COMMON / "requirements.tex")
    )
    predictions = re.findall(
        r"\\PTCPrediction\{(\d{3})\}", text(COMMON / "edge_cases.tex")
    )
    sequence([f"SCI-PTC-DEF-{item}" for item in definitions], "SCI-PTC-DEF-", 45)
    sequence([f"SCI-PTC-ASM-{item}" for item in assumptions], "SCI-PTC-ASM-", 35)
    sequence([f"SCI-PTC-REQ-{item}" for item in requirements], "SCI-PTC-REQ-", 99)
    sequence([f"SCI-PTC-PRED-{item}" for item in predictions], "SCI-PTC-PRED-", 60)

    metadata_pattern = re.compile(
        r"\\(PTCRequirement|PTCPrediction)\{(\d{3})\}\{([^{}]*)\}"
        r"\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}\{"
    )
    metadata = []
    for path in (COMMON / "requirements.tex", COMMON / "edge_cases.tex"):
        metadata.extend(metadata_pattern.findall(text(path)))
    require(len(metadata) == 159, "normative metadata row count differs")
    for macro, number, _title, locator, _decision, _dependency in metadata:
        stem = "REQ" if macro == "PTCRequirement" else "PRED"
        require(locator.strip() != "", f"SCI-PTC-{stem}-{number} has blank rationale locator")
        require(
            locator.startswith("Rationale ") and re.search(r"\d", locator),
            f"SCI-PTC-{stem}-{number} has unresolved rationale locator: {locator!r}",
        )
        require(
            not re.search(r"\b(?:TBD|UNRESOLVED|PENDING|UNKNOWN)\b", locator, re.I),
            f"SCI-PTC-{stem}-{number} has unresolved rationale locator: {locator!r}",
        )
    equation_source = text(COMMON / "equations.tex")
    require(
        len(re.findall(r"\\begin\{(?:equation|align)\}", equation_source)) == 25,
        "numbered-equation count differs",
    )

    common_files = [
        "notation.tex", "definitions.tex", "equations.tex",
        "assumptions.tex", "requirements.tex", "edge_cases.tex",
    ]
    expected_inputs = [rf"\input{{common/{name}}}" for name in common_files]
    engineering = text(SRC / "engineering-conformance.tex")
    rationale = text(SRC / "scientific-rationale.tex")
    require(
        re.findall(r"\\input\{common/[^}]+\}", engineering) == expected_inputs,
        "engineering view does not import the six shared modules exactly once",
    )
    require(
        not re.findall(r"\\input\{common/[^}]+\}", rationale),
        "standalone rationale still imports the full formal core",
    )
    require(
        rationale.index(r"\section{What PTC estimates -- and what it cannot identify}")
        < rationale.index(r"\section{Provenance and program adherence}"),
        "science does not precede program provenance",
    )
    require(
        rationale.count(r"\begin{figure}") == 2,
        "standalone rationale does not contain exactly two explanatory figures",
    )
    require(
        "standalone science-team rationale" in rationale
        and "companion" in rationale.lower(),
        "rationale does not declare the split-view architecture",
    )

    forbidden_display = (
        r"\\begin\{equation\*?\}|\\begin\{align\*?\}|"
        r"\\begin\{gather\*?\}|^\s*\\\[|\$\$"
    )
    require(
        not re.search(forbidden_display, engineering, flags=re.MULTILINE),
        "engineering wrapper independently restates displayed mathematics",
    )

    subprocess.run(
        [str(Path("/Users/gwilson/tolteca/bin/python")),
         str(SRC / "generate_crosswalk.py"), "--check"],
        cwd=PKG,
        check=True,
    )
    crosswalk = text(PKG / "CROSSWALK.md")
    crosswalk_ids = re.findall(
        r"^\| .(SCI-PTC-(?:REQ|PRED)-\d{3}). \|",
        crosswalk,
        flags=re.MULTILINE,
    )
    sequence(crosswalk_ids[:99], "SCI-PTC-REQ-", 99)
    sequence(crosswalk_ids[99:], "SCI-PTC-PRED-", 60)
    require(len(crosswalk_ids) == 159, "crosswalk row count differs")

    decisions = text(PKG / "AUTHOR_DRAFT_DECISIONS.md")
    author_ids = re.findall(
        r"^\| .(PTC-AUTH-D\d{3}). \|", decisions, flags=re.MULTILINE
    )
    sequence(author_ids, "PTC-AUTH-D", 38)
    owner_rows = re.findall(
        r"^\| .(PTC-(?:OWNER-Q|OD-)\d{3}). \| [^|]+ \| ([^|]+) \|",
        decisions,
        flags=re.MULTILINE,
    )
    require(len(owner_rows) == 15, "detailed owner-ledger row count differs")
    owner_states = [state.strip() for _identifier, state in owner_rows]
    require(
        {
            "decided": sum(state.startswith("decided") for state in owner_states),
            "open": sum(state == "open" for state in owner_states),
            "known": sum(state == "known but not supplied" for state in owner_states),
            "deferred": sum(state.startswith("deferred") for state in owner_states),
        }
        == {"decided": 5, "open": 3, "known": 4, "deferred": 3},
        "detailed owner-ledger state counts differ",
    )
    require("PTC-OWNER-Q002" in decisions and "decided" in decisions,
            "owner-approved projection decision is missing")
    review = text(PKG / "SCIENTIFIC_OWNER_REVIEW_R0.2.md")
    require(
        "bd4aa11330b477f628b62118c13f2274d625dff2a873c13f025f9d932d01aac8"
        in review
        and "High effort is sufficient" in review,
        "r0.2 scientific-owner review record or effort disposition is missing",
    )
    freeze_r04 = PKG / "SCIENTIFIC_OWNER_FREEZE_R0.4.md"
    require(
        digest(freeze_r04)
        == "90334ea7853e1ab274f6858fad66078356c06326438625c7fe294e41c07fbcc4",
        "r0.4 scientific-owner freeze digest",
    )
    freeze_status = (
        "Scientific authority frozen; implementation conformity not yet assessed "
        "under this contract."
    )
    require(
        freeze_status in text(freeze_r04).replace("\n", " "),
        "exact r0.4 frozen status in owner record",
    )
    freeze_r05 = PKG / "SCIENTIFIC_OWNER_FREEZE_R0.5.md"
    require(
        digest(freeze_r05) == "8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66",
        "r0.5 scientific-owner freeze digest",
    )
    require("Freeze SCI-PTC v0.1/r0.5." in text(freeze_r05),
            "exact r0.5 owner freeze statement")
    require("8f0ecccfacbdce0543141c4289ec06c702065f5e" in text(freeze_r05),
            "verified r0.5 candidate commit")
    require("Scientific authority frozen by owner" in rationale,
            "frozen status in rationale")
    require("Scientific authority frozen by owner" in engineering,
            "frozen status in engineering")

    required_markers = (
        r"\mathcal A_{\Theta}",
        r"U_{{\rm total},\Theta}(Y^{\rm CAL})=Y^{\rm CAL}-Z",
        r"D\mathcal A^{\rm sub}_{\widehat U_{\rm total}}[Y](H)=H",
        r"\Delta_{\rm state}",
        r"\mathcal P_Z(Y^{\rm CAL}+\epsilon h^{\rm CAL})",
        r"U_{\star,td}=\sum_{k=1}^{K}M_{\star,tk}A_{\star,dk}",
        r"k_{\rm src}^{\rm CAL}=K_{\rm up\to CAL}\tau_{\rm src}",
        r"\mathsf{route}=\mathsf{RTC\_terminal\_export}",
        r"C_{\lambda_g}(Y')=Y'-\lambda_g",
        r"\mathcal C&=\bigcup_i\mathcal C_i",
        r"\operatorname{Decision}_U",
        r"k_{{\rm req},g}\in\mathbb Z",
        r"\lambda_{g,d}",
        r"\mathcal T^{\rm ctr}_{g,d}",
        r"w^{\rm ctr}_{gtd}&=1",
        r"\operatorname{rank}_{\tau_g}(G^{\rm app}_{g,t})",
        r"\texttt{decision\_unavailable}",
    )
    for marker in required_markers:
        require(marker in equation_source, f"missing r0.5 formal marker: {marker}")
    require(r"C^{-1}" not in equation_source, "centering inverse/restoration remains")
    require(
        r"$y=P(x-\lambda)$" in rationale
        and r"not $\lambda+P(x-\lambda)$" in rationale,
        "scientist-facing nonrestoring-centering statement is missing",
    )

    candidate_pdfs = {
        PDF / "r0.5-candidate"
        / "SCI-PTC-SCIENTIFIC-RATIONALE-v0.1-r0.5-CANDIDATE.pdf":
            "eb881ba6d85193d01b3c5f3cc387e5e59d70d498f5210914cfdf0041a4671703",
        PDF / "r0.5-candidate"
        / "SCI-PTC-ENGINEERING-CONFORMANCE-v0.1-r0.5-CANDIDATE.pdf":
            "23941bb70fb24a0e46f24a41409d0f94ba5de9c4b14b35126b239f6452ff4dfc",
    }
    for path, expected in candidate_pdfs.items():
        require(digest(path) == expected, f"candidate history changed: {path.name}")

    rationale_path = PDF / "SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf"
    engineering_path = PDF / "SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf"
    frozen_pdfs = {
        rationale_path:
            "3c927dbcb631b2033f04d933fa4d69911698b840b7b4642e759ad8c715c16ab6",
        engineering_path:
            "f0b3bdc28c0997e4960b8231d6113339fe391d331e84dc7c0638912b0d3f7adb",
    }
    for path, expected in frozen_pdfs.items():
        require(digest(path) == expected, f"frozen PDF hash changed: {path.name}")
    rationale_pdf, rationale_text = pdf_text(rationale_path)
    engineering_pdf, engineering_text = pdf_text(engineering_path)
    require(10 <= len(rationale_pdf.pages) <= 18,
            "standalone rationale page count is outside the review range")
    require(15 <= len(engineering_pdf.pages) <= 26,
            "engineering page count is outside the review range")
    for name, reader, extracted in (
        ("scientific rationale", rationale_pdf, rationale_text),
        ("engineering conformance", engineering_pdf, engineering_text),
    ):
        require(not reader.is_encrypted, f"{name} is encrypted")
        require(reader.get_fields() is None, f"{name} contains a form")
        root = reader.trailer["/Root"]
        names = root.get("/Names", {})
        require("/JavaScript" not in names, f"{name} contains document JavaScript")
        for page in reader.pages:
            require(
                float(page.mediabox.width) == 612.0
                and float(page.mediabox.height) == 792.0,
                f"{name} contains a non-letter page",
            )
        require("v0.1" in extracted and "r0.5" in extracted,
                f"{name} version stamps")
        require("Scientific authority frozen by owner" in extracted,
                f"{name} frozen status")
        require("diagnostic-only" in extracted.lower(),
                f"{name} diagnostic-r disposition")
        require("frozen" in extracted.lower() and "projection" in extracted.lower(),
                f"{name} projection decision")

    for number in range(1, 100):
        require(
            f"SCI-PTC-REQ-{number:03d}" in engineering_text,
            f"engineering PDF missing requirement {number:03d}",
        )
    for number in range(1, 61):
        require(
            f"SCI-PTC-PRED-{number:03d}" in engineering_text,
            f"engineering PDF missing prediction {number:03d}",
        )
    require("SCI-PTC-REQ-099" not in rationale_text,
            "rationale still contains the full normative requirement register")
    for marker in (
        "What PTC estimates",
        "Validity is use-specific support",
        "The exact group-local frozen application map",
        "Scientific validation program",
        "Compact rationale-to-contract crosswalk",
        "Provenance and program adherence",
    ):
        require(marker in rationale_text, f"rationale PDF missing section: {marker}")

    active_source = "\n".join(text(path) for path in sorted(SRC.rglob("*.tex")))
    for forbidden in (
        "science-qualified", "production-ready", "validated implementation", "inputenc",
    ):
        require(forbidden not in active_source,
                f"forbidden active-source claim: {forbidden}")

    print("PASS: approved packet hashes (5 items including retained core)")
    print("PASS: definitions=45 equations=25 assumptions=35 requirements=99 predictions=60")
    print("PASS: standalone rationale plus six-file engineering/formal view")
    print("PASS: crosswalk rows=159, exact and sequential; all rationale locators resolved")
    print("PASS: author decisions=38, owner entries=15 (5 decided, 3 open, 4 known-but-not-supplied, 3 deferred)")
    print("PASS: r0.5 operator, knowledge, grouping, rank, kernel, and RTC-terminal markers")
    print(
        f"PASS: PDFs={len(rationale_pdf.pages)}/{len(engineering_pdf.pages)} "
        "letter pages, unencrypted, no forms/JavaScript"
    )
    print("PASS: exact r0.5 scientific-owner freeze bound; r0.4 freeze preserved")
    print("PASS: canonical frozen r0.5 PDF hashes (2); candidate history preserved")
    print("PASS: engineering PDF contains all 159 normative IDs")
    print("PASS: engineering wrapper has no independent displayed mathematics")


if __name__ == "__main__":
    main()
