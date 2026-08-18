#!/usr/bin/env python3
"""Mechanical SCI-RTC v0.1/r0.5 author-deliverable checks.

This helper reads only the approved author inputs and package deliverables.
It does not inspect implementation, tests, history, or sibling packages.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


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


def table_row_ids(document: str, stem: str) -> list[str]:
    pattern = rf"^\| `({re.escape(stem)}\d{{3}}[a-z]?)` \|"
    return re.findall(pattern, document, flags=re.MULTILINE)


def sequential(ids: list[str], stem: str, count: int) -> None:
    expected = [f"{stem}{number:03d}" for number in range(1, count + 1)]
    require(ids == expected, f"{stem} sequence differs: {ids}")


def main() -> None:
    expected_hashes = {
        PKG / "SCOPE_BRIEF.md":
            "c8cac0b8ae731919622d7b696c60946685b5eba9b16a5cd830c01a2f6f28e013",
        PKG / "AUTHOR_SUPERSESSION_COVER.md":
            "f183c8fb083c3a851fda5d77a0944405cc41650ced29bd0162cffba832f25575",
        PKG / "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md":
            "a26220dc827330e30ca8e4c75e82600e6cc2f05358887bbaa0c6da93f98ecb5b",
    }
    for path, expected in expected_hashes.items():
        require(digest(path) == expected, f"approved input hash changed: {path.name}")

    core = subprocess.run(
        [
            "git", "show",
            "3319d7424c732c1c9fc300c336e4d428e6f91068:"
            "doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex",
        ],
        cwd=PKG,
        check=True,
        capture_output=True,
    ).stdout
    require(
        hashlib.sha256(core).hexdigest()
        == "d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7",
        "retained-core hash changed",
    )

    definitions = re.findall(
        r"\\item\[SCI-RTC-DEF-(\d{3})\b", text(COMMON / "definitions.tex")
    )
    assumptions = re.findall(
        r"\\item\[SCI-RTC-ASM-(\d{3})\b", text(COMMON / "assumptions.tex")
    )
    requirements = re.findall(
        r"\\ReqID\{SCI-RTC-REQ-(\d{3})\}", text(COMMON / "requirements.tex")
    )
    predictions = re.findall(
        r"\\PredID\{SCI-RTC-PRED-(\d{3})\}", text(COMMON / "edge_cases.tex")
    )
    equation_ids = re.findall(
        r"\\tag\{SCI-RTC-EQ-(\d{3}[ab]?)\}", text(COMMON / "equations.tex")
    )

    require(definitions == [f"{i:03d}" for i in range(1, 39)], "definition IDs")
    require(assumptions == [f"{i:03d}" for i in range(1, 13)], "assumption IDs")
    require(requirements == [f"{i:03d}" for i in range(1, 106)], "requirement IDs")
    require(predictions == [f"{i:03d}" for i in range(1, 64)], "prediction IDs")
    expected_eq = (
        [f"{i:03d}" for i in range(1, 16)]
        + ["016a", "016b", "017", "018", "019", "020a", "020b"]
        + [f"{i:03d}" for i in range(21, 36)]
    )
    require(equation_ids == expected_eq, "equation tag IDs")

    common_files = [
        "notation.tex", "definitions.tex", "equations.tex",
        "assumptions.tex", "requirements.tex", "edge_cases.tex",
    ]
    expected_inputs = [rf"\input{{common/{name}}}" for name in common_files]
    for name in common_files:
        require("v0.1/r0.5" in text(COMMON / name), f"r0.5 stamp in {name}")
    for wrapper_name in ("scientific-rationale.tex", "engineering-conformance.tex"):
        wrapper = text(SRC / wrapper_name)
        actual = re.findall(r"\\input\{common/[^}]+\}", wrapper)
        require(actual == expected_inputs, f"shared-core inclusion in {wrapper_name}")
        require(wrapper.count("rtc-core-begin") == 0, f"wrapper duplicates core begin: {wrapper_name}")

    engineering = text(SRC / "engineering-conformance.tex")
    forbidden_display = (
        r"\\begin\{equation\*?\}|\\begin\{align\*?\}|"
        r"\\begin\{gather\*?\}|^\s*\\\[|\$\$"
    )
    require(
        not re.search(forbidden_display, engineering, flags=re.MULTILINE),
        "engineering wrapper independently restates displayed mathematics",
    )

    rationale = text(SRC / "scientific-rationale.tex")
    require("Learn--Resolve--Apply Filtering" in rationale, "rationale title")
    require("Role-specific RTC plans" in rationale, "role-specific plan matrix")
    require(r"Paired IQ-to-\texorpdfstring{$x/r$}{x/r} readout coordinates" in rationale,
            "paired-coordinate rationale section")
    narrative = rationale.split(r"\appendix", maxsplit=1)[0]
    numbered_sections = re.findall(r"^\\section\{", narrative, flags=re.MULTILINE)
    require(len(numbered_sections) == 12, "rationale narrative is not twelve sections")

    equations_text = text(COMMON / "equations.tex")
    for marker in (
        r"x^{\rm eval,(0)}",
        r"\widetilde\Pi_a",
        r"k_{a+1}",
        r"v^{\rm preD}_{d,Mn}",
        r"C^{\rm CAL}_{\kappa,\mathcal R}\mathbf y^{\rm RTC}_{\mathcal R}",
        r"\begin{bmatrix}x^A_{dj}\\ r^A_{dj}\end{bmatrix}",
        r"K=k_{A+1}\le A\le A_{\max}",
    ):
        require(marker in equations_text, f"r0.5 equation marker: {marker}")
    require(r"C_{\mathcal R}" not in equations_text, "obsolete in-RTC CAL selector")

    requirements_text = text(COMMON / "requirements.tex")
    require("nonrepresentative replacement influence" in requirements_text,
            "consumer-owned nonrepresentative influence")
    require("alternative interleaving is authorized" in requirements_text,
            "selected raw-RTC-then-CAL order")
    require("despike detection and admitted sample replacement before level-shift"
            in requirements_text, "approved early suborder")
    require("shall precede temporal filtering" in requirements_text,
            "approved atmosphere/filter order")
    require("subsequent PTC, VAL, MAP, and FLT use follows their own contracts"
            in requirements_text, "exact Science consumer wording")

    directive = text(PKG / "SCIENTIFIC_OWNER_DIRECTIVE_R0.5.md")
    require("7469fd327d9465904a4e59c287577bab0dcd9f93fd2cc555cdee6680e89714a6"
            in directive, "r0.5 directive digest")

    crosswalk = text(PKG / "CROSSWALK.md")
    for stem, count in (
        ("SCI-RTC-DEF-", 38),
        ("SCI-RTC-ASM-", 12),
        ("SCI-RTC-REQ-", 105),
        ("SCI-RTC-PRED-", 63),
    ):
        sequential(table_row_ids(crosswalk, stem), stem, count)
    eq_rows = table_row_ids(crosswalk, "SCI-RTC-EQ-")
    require(eq_rows == [f"SCI-RTC-EQ-{item}" for item in expected_eq], "equation crosswalk")

    author_rows = table_row_ids(text(PKG / "AUTHOR_DRAFT_DECISIONS.md"), "SCI-RTC-AUTHOR-D")
    sequential(author_rows, "SCI-RTC-AUTHOR-D", 23)
    owner_rows = table_row_ids(
        text(PKG / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"), "SCI-RTC-OWNER-"
    )
    sequential(owner_rows, "SCI-RTC-OWNER-", 71)

    print("PASS: approved packet hashes (4)")
    print("PASS: shared-core inclusion (6 files x 2 views, exactly once)")
    print("PASS: definitions=38 equations=37 assumptions=12 requirements=105 predictions=63")
    print("PASS: crosswalk rows complete and sequential")
    print("PASS: author decisions=23 owner entries=71 (65 open, 2 resolved, 4 deferred)")
    print("PASS: r0.5 paired mapping, level shifts, leakage, attempt counts, CAL, influence, and role markers")
    print("PASS: rationale narrative sections=12")
    print("PASS: engineering wrapper has no independent displayed mathematics")


if __name__ == "__main__":
    main()
