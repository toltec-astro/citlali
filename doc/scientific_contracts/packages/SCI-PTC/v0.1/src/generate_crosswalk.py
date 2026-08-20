#!/usr/bin/env python3
"""Generate or check the exact SCI-PTC v0.1/r0.3 crosswalk."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SRC = Path(__file__).resolve().parent
PKG = SRC.parent
COMMON = SRC / "common"
OUTPUT = PKG / "CROSSWALK.md"


def parse_rows(path: Path, macro: str, stem: str, audience: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"\\{macro}\{{(\d{{3}})\}}\{{([^{{}}]*)\}}"
        rf"\{{([^{{}}]*)\}}\{{([^{{}}]*)\}}\{{([^{{}}]*)\}}\{{"
    )
    rows = []
    for number, title, rationale, decision, dependency in pattern.findall(source):
        identifier = f"SCI-PTC-{stem}-{number}"
        rows.append(
            f"| `{identifier}` | `src/common/{path.name}` ({title}) | "
            f"{rationale}; compact crosswalk | {audience} | "
            f"{decision} | {dependency} |"
        )
    return rows


def render() -> str:
    requirements = parse_rows(
        COMMON / "requirements.tex",
        "PTCRequirement",
        "REQ",
        "Engineering normative requirements",
    )
    predictions = parse_rows(
        COMMON / "edge_cases.tex",
        "PTCPrediction",
        "PRED",
        "Engineering falsifiable predictions",
    )
    lines = [
        "# SCI-PTC v0.1 -- Requirement And Prediction Crosswalk",
        "",
        "Status: Stage B bounded freeze-candidate draft `r0.3`; generated from the shared "
        "normative macro metadata",
        "",
        f"Coverage: {len(requirements)} requirements and {len(predictions)} predictions.",
        "",
        "| Identifier | Shared canonical source | Scientist-facing source | "
        "Engineering-facing source | Owner decision | Dependency |",
        "| --- | --- | --- | --- | --- | --- |",
        *requirements,
        *predictions,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if CROSSWALK.md is not exact"
    )
    args = parser.parse_args()
    expected = render()
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit("FAIL: CROSSWALK.md is not generated from current metadata")
        print("PASS: CROSSWALK.md matches current normative metadata")
        return
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"WROTE: {OUTPUT}")


if __name__ == "__main__":
    main()
