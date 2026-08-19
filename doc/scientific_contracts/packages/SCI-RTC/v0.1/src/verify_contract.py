#!/usr/bin/env python3
"""Mechanical SCI-RTC v0.1/r0.7 author-deliverable checks.

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
    require(requirements == [f"{i:03d}" for i in range(1, 109)], "requirement IDs")
    require(predictions == [f"{i:03d}" for i in range(1, 72)], "prediction IDs")
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
        require("v0.1/r0.7" in text(COMMON / name), f"r0.7 stamp in {name}")
    engineering = text(SRC / "engineering-conformance.tex")
    rationale = text(SRC / "scientific-rationale.tex")
    require(
        re.findall(r"\\input\{common/[^}]+\}", engineering) == expected_inputs,
        "shared-core inclusion in engineering formal view",
    )
    require(
        not re.findall(r"\\input\{common/[^}]+\}", rationale),
        "science-team rationale duplicates formal core",
    )
    require(engineering.count("rtc-core-begin") == 0,
            "engineering wrapper duplicates core begin")

    forbidden_display = (
        r"\\begin\{equation\*?\}|\\begin\{align\*?\}|"
        r"\\begin\{gather\*?\}|^\s*\\\[|\$\$"
    )
    require(
        not re.search(forbidden_display, engineering, flags=re.MULTILINE),
        "engineering wrapper independently restates displayed mathematics",
    )

    require("Learn--Resolve--Apply Filtering" in rationale, "rationale title")
    require("Role-specific RTC plans" in rationale, "role-specific plan matrix")
    require(r"Paired IQ-to-\texorpdfstring{$x/r$}{x/r} readout coordinates" in rationale,
            "paired-coordinate rationale section")
    numbered_sections = re.findall(r"^\\section\{", rationale, flags=re.MULTILINE)
    require(len(numbered_sections) == 12, "rationale narrative is not twelve sections")
    require(r"\appendix" not in rationale, "rationale is still the hybrid authoring view")
    require("companion Engineering Conformance PDF" in rationale,
            "rationale does not identify the separate formal authority")
    require(
        not re.search(forbidden_display, rationale, flags=re.MULTILINE),
        "rationale independently restates displayed normative mathematics",
    )

    equations_text = text(COMMON / "equations.tex")
    for marker in (
        r"x^{\rm eval,(0)}",
        r"\widetilde\Pi_a",
        r"k_{a+1}",
        r"v^{x,\rm preD}_{d,Mn}",
        r"C^{\rm CAL}_{\kappa,\mathcal R}\mathbf y^{x,\rm RTC}_{\mathcal R}",
        r"\begin{bmatrix}x^{\rm acq}_{dm}\\ r^{\rm acq}_{dm}\end{bmatrix}",
        r"\mathcal T_{d,\zeta}",
        r"\epsilon^{(c)}_d&=\frac{\alpha^{r,(c)}_d}{\alpha^{x,(c)}_d}",
        r"K^{x\leftarrow r,\rm RTC}_\Omega",
        r"J_{\rm num,\Omega}=\begin{bmatrix}L^x_\Omega&0\end{bmatrix}",
        r"\tau_{de}=\tau_e+\delta\tau_{de}",
        r"\operatorname{atan2}",
        r"(\mathbf y^x,\mathbf r^{A,\rm parent},\mathcal J^{xr},\mathcal K^{x}",
        r"K=k_{A+1}\le A\le A_{\max}",
    ):
        require(marker in equations_text, f"r0.7 equation marker: {marker}")
    complete_operator = equations_text.split(
        r"\tag{SCI-RTC-EQ-005}", maxsplit=1
    )[0].rsplit(r"\begin{equation}", maxsplit=1)[1]
    require(r"A_\alpha" not in complete_operator,
            "ALIGN remains in the RTC-local complete operator")
    require("acquired input $(q,j)$" not in equations_text,
            "acquired-grid residue remains in RTC-local response")
    require(r"\partial x^{A}_{qj}" in equations_text,
            "RTC-local response is not differentiated from aligned input")
    require(r"C_{\mathcal R}" not in equations_text, "obsolete in-RTC CAL selector")

    requirements_text = text(COMMON / "requirements.tex")
    require("nonrepresentative replacement influence" in requirements_text,
            "consumer-owned nonrepresentative influence")
    require("no calibrated RTC branch or second ALIGN application is authorized"
            in requirements_text,
            "conditioned-x-then-CAL order")
    require("without donor substitution" in requirements_text,
            "original-pair shift-learning boundary")
    require("shall not be subtracted from science $x$" in requirements_text,
            "diagnostic-only atmospheric-template boundary")
    require("Any separately requested conditioned $r$ product" in requirements_text,
            "separately authorized conditioned-r boundary")
    require(r"$J_{\rm num,\Omega}=[L^x_\Omega\ 0]$" in requirements_text,
            "fixed-state numerical covariance boundary")
    require("coordinate-comparison compatibility" in requirements_text,
            "leakage coordinate compatibility")
    require("Carry across the boundary is permitted only" in requirements_text,
            "carry continuity exception")
    require("subsequent PTC, VAL, MAP, and FLT use follows their own contracts"
            in requirements_text, "exact Science consumer wording")

    directive = text(PKG / "SCIENTIFIC_OWNER_DIRECTIVE_R0.5.md")
    require("7469fd327d9465904a4e59c287577bab0dcd9f93fd2cc555cdee6680e89714a6"
            in directive, "r0.5 directive digest")
    review = text(PKG / "SCIENTIFIC_OWNER_REVIEW_R0.6.md")
    require("2a4163d1ed0775e83ef981573d1a3a1f65fe2d89860bd92b0ad456e61fa8e266"
            in review, "r0.6 review digest")
    review_r07 = text(PKG / "SCIENTIFIC_OWNER_REVIEW_R0.7.md")
    require("01ec886e6d1dad89835463a1cee39dd0da067cf7532608698f90262cb41a9937"
            in review_r07, "r0.7 review digest")

    active_source = "\n".join(text(path) for path in sorted(SRC.rglob("*.tex")))
    for forbidden in (
        "atmospheric-template removal",
        "atmosphere-template removal",
        "despike-before-shift",
        "IQ-map parent",
    ):
        require(forbidden not in active_source, f"superseded active-source phrase: {forbidden}")

    crosswalk = text(PKG / "CROSSWALK.md")
    for stem, count in (
        ("SCI-RTC-DEF-", 38),
        ("SCI-RTC-ASM-", 12),
        ("SCI-RTC-REQ-", 108),
        ("SCI-RTC-PRED-", 71),
    ):
        sequential(table_row_ids(crosswalk, stem), stem, count)
    eq_rows = table_row_ids(crosswalk, "SCI-RTC-EQ-")
    require(eq_rows == [f"SCI-RTC-EQ-{item}" for item in expected_eq], "equation crosswalk")

    author_rows = table_row_ids(text(PKG / "AUTHOR_DRAFT_DECISIONS.md"), "SCI-RTC-AUTHOR-D")
    sequential(author_rows, "SCI-RTC-AUTHOR-D", 24)
    owner_rows = table_row_ids(
        text(PKG / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"), "SCI-RTC-OWNER-"
    )
    sequential(owner_rows, "SCI-RTC-OWNER-", 74)

    owner_ledger = text(PKG / "SCIENTIFIC_OWNER_DECISION_LEDGER.md")
    owner_states = re.findall(
        r"^\| `SCI-RTC-OWNER-\d{3}` \| ([A-Z]+) \|", owner_ledger,
        flags=re.MULTILINE,
    )
    require(
        {state: owner_states.count(state) for state in set(owner_states)}
        == {"OPEN": 63, "CONDITIONAL": 1, "RESOLVED": 5, "DEFERRED": 5},
        "owner-ledger state counts",
    )

    print("PASS: approved packet hashes (4)")
    print("PASS: focused rationale plus complete six-file engineering/formal view")
    print("PASS: definitions=38 equations=37 assumptions=12 requirements=108 predictions=71")
    print("PASS: crosswalk rows complete and sequential")
    print("PASS: author decisions=24 owner entries=74 (63 open, 1 conditional, 5 resolved, 5 deferred)")
    print("PASS: r0.7 ALIGN, fixed-state covariance, leakage metric, event-time, carry, inventory, and atmosphere markers")
    print("PASS: rationale narrative sections=12")
    print("PASS: engineering wrapper has no independent displayed mathematics")


if __name__ == "__main__":
    main()
