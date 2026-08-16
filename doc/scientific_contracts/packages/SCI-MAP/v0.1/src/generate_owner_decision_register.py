#!/usr/bin/env python3
"""Generate the scientist-PDF owner-decision register from the Markdown ledger."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "SCIENTIFIC_OWNER_DECISION_LEDGER.md"
OUTPUT = Path(__file__).resolve().parent / "SCI-MAP-v0.1_OWNER_DECISION_REGISTER_r0.1.tex"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_inline(markdown: str) -> str:
    parts = markdown.split("`")
    assert len(parts) % 2 == 1, f"unbalanced inline-code delimiter: {markdown!r}"
    rendered: list[str] = []
    for index, part in enumerate(parts):
        escaped = latex_escape(part)
        rendered.append(rf"\texttt{{{escaped}}}" if index % 2 else escaped)
    return "".join(rendered)


def ledger_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| SCI-MAP-OD-"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        assert len(cells) == 6, f"unexpected owner-ledger row: {line}"
        decision_id, status, question, _, interim, _ = cells
        rows.append((decision_id, status.replace("**", ""), question, interim))
    expected = [f"SCI-MAP-OD-{number:03d}" for number in range(1, 8)]
    assert [row[0] for row in rows] == expected, "owner-decision sequence mismatch"
    assert all(row[1] == "OPEN" for row in rows), "non-OPEN owner decision present"
    return rows


def render() -> str:
    lines = [
        "% GENERATED from SCIENTIFIC_OWNER_DECISION_LEDGER.md; do not hand edit.",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{L{0.14\linewidth}L{0.37\linewidth}L{0.39\linewidth}}",
        r"\toprule",
        r"Decision/status & Exact decision requested & Conservative draft behavior pending decision\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Decision/status & Exact decision requested & Conservative draft behavior pending decision\\",
        r"\midrule",
        r"\endhead",
    ]
    for decision_id, status, question, interim in ledger_rows():
        lines.extend(
            [
                rf"\texttt{{{latex_escape(decision_id)}}}\\\textbf{{{latex_escape(status)}}} &",
                rf"{latex_inline(question)} &",
                rf"{latex_inline(interim)}\\",
                r"\addlinespace[0.35em]",
            ]
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8")
        assert actual == expected, f"generated owner register is stale: {OUTPUT}"
        return
    OUTPUT.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
