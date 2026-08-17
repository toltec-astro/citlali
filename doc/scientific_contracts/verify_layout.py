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
COMPLETE_PACKAGES = {
    "SCI-CAL": "v0.1",
    "SCI-MAP": "v0.1",
}
STAGE_B_PACKAGES = {
    "SCI-BEAM": "v0.1",
}


assert (ROOT / "INDEX.md").is_file()
assert (ROOT / "templates" / "SCOPE_BRIEF.md").is_file()

for package, version in {**COMPLETE_PACKAGES, **STAGE_B_PACKAGES}.items():
    base = ROOT / "packages" / package / version
    for name in (
        "README.md",
        "PRIOR_WORK.md",
        "SCOPE_BRIEF.md",
        "DECISION_LOG.md",
        "CROSSWALK.md",
        "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    ):
        assert (base / name).is_file(), f"{package}: missing {name}"
    for name in COMMON:
        assert (base / "src" / "common" / name).is_file(), f"{package}: missing common/{name}"
    for name in ("scientific-rationale.tex", "engineering-conformance.tex"):
        assert (base / "src" / name).is_file(), f"{package}: missing src/{name}"
    if package in COMPLETE_PACKAGES:
        for view in ("SCIENTIFIC-RATIONALE", "ENGINEERING-CONFORMANCE"):
            output = base / "pdf" / f"{package}-{view}-{version}.pdf"
            assert output.is_file() and output.stat().st_size > 0, f"{package}: missing {output.name}"
    else:
        assert (base / "INTERNAL_DOSSIER.md").is_file(), f"{package}: missing INTERNAL_DOSSIER.md"
        for name in (
            "AUTHOR_PACKET_MANIFEST.md",
            "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
            "AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md",
        ):
            assert (base / name).is_file(), f"{package}: missing {name}"
        assert (base / "pdf" / "README.md").is_file(), f"{package}: missing pdf/README.md"

print("scientific_contract_layout=PASS")
print("complete_packages=" + ",".join(f"{name}/{version}" for name, version in COMPLETE_PACKAGES.items()))
print("stage_b_authorized_packages=" + ",".join(f"{name}/{version}" for name, version in STAGE_B_PACKAGES.items()))
