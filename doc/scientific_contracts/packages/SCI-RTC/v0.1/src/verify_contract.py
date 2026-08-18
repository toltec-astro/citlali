#!/usr/bin/env python3
"""Mechanical SCI-RTC v0.1/r0.2 author-deliverable checks.

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

    require(definitions == [f"{i:03d}" for i in range(1, 27)], "definition IDs")
    require(assumptions == [f"{i:03d}" for i in range(1, 13)], "assumption IDs")
    require(requirements == [f"{i:03d}" for i in range(1, 71)], "requirement IDs")
    require(predictions == [f"{i:03d}" for i in range(1, 39)], "prediction IDs")
    expected_eq = (
        [f"{i:03d}" for i in range(1, 16)]
        + ["016a", "016b", "017", "018", "019", "020a", "020b"]
        + [f"{i:03d}" for i in range(21, 29)]
    )
    require(equation_ids == expected_eq, "equation tag IDs")

    common_files = [
        "notation.tex", "definitions.tex", "equations.tex",
        "assumptions.tex", "requirements.tex", "edge_cases.tex",
    ]
    expected_inputs = [rf"\input{{common/{name}}}" for name in common_files]
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
    narrative = rationale.split(r"\appendix", maxsplit=1)[0]
    numbered_sections = re.findall(r"^\\section\{", narrative, flags=re.MULTILINE)
    require(len(numbered_sections) == 12, "rationale narrative is not twelve sections")

    crosswalk = text(PKG / "CROSSWALK.md")
    for stem, count in (
        ("SCI-RTC-DEF-", 26),
        ("SCI-RTC-ASM-", 12),
        ("SCI-RTC-REQ-", 70),
        ("SCI-RTC-PRED-", 38),
    ):
        sequential(table_row_ids(crosswalk, stem), stem, count)
    eq_rows = table_row_ids(crosswalk, "SCI-RTC-EQ-")
    require(eq_rows == [f"SCI-RTC-EQ-{item}" for item in expected_eq], "equation crosswalk")

    author_rows = table_row_ids(text(PKG / "AUTHOR_DRAFT_DECISIONS.md"), "SCI-RTC-AUTHOR-D")
    sequential(author_rows, "SCI-RTC-AUTHOR-D", 18)
    owner_rows = table_row_ids(
        text(PKG / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"), "SCI-RTC-OWNER-"
    )
    sequential(owner_rows, "SCI-RTC-OWNER-", 36)

    print("PASS: approved packet hashes (4)")
    print("PASS: shared-core inclusion (6 files x 2 views, exactly once)")
    print("PASS: definitions=26 equations=30 assumptions=12 requirements=70 predictions=38")
    print("PASS: crosswalk rows complete and sequential")
    print("PASS: author decisions=18 owner entries=36")
    print("PASS: rationale narrative sections=12")
    print("PASS: engineering wrapper has no independent displayed mathematics")


if __name__ == "__main__":
    main()
