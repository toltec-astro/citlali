#!/usr/bin/env python3
"""Verify the minimal uniform scientific-contract package layout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMON = (
    "notation.tex",
    "definitions.tex",
    "equations.tex",
    "assumptions.tex",
    "requirements.tex",
    "edge_cases.tex",
)
PACKAGES = {
    "SCI-CAL": "v0.1",
    "SCI-MAP": "v0.1",
}


assert (ROOT / "INDEX.md").is_file()
assert (ROOT / "templates" / "SCOPE_BRIEF.md").is_file()

for package, version in PACKAGES.items():
    base = ROOT / "packages" / package / version
    for name in ("README.md", "SCOPE_BRIEF.md", "DECISION_LOG.md", "CROSSWALK.md"):
        assert (base / name).is_file(), f"{package}: missing {name}"
    for name in COMMON:
        assert (base / "src" / "common" / name).is_file(), f"{package}: missing common/{name}"
    for name in ("scientific-rationale.tex", "engineering-conformance.tex"):
        assert (base / "src" / name).is_file(), f"{package}: missing src/{name}"
    for view in ("SCIENTIFIC-RATIONALE", "ENGINEERING-CONFORMANCE"):
        output = base / "pdf" / f"{package}-{view}-{version}.pdf"
        assert output.is_file() and output.stat().st_size > 0, f"{package}: missing {output.name}"

print("scientific_contract_layout=PASS")
print("packages=" + ",".join(f"{name}/{version}" for name, version in PACKAGES.items()))
