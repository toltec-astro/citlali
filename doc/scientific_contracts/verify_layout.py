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
    "SCI-BEAM": "v0.1",
    "SCI-RTC": "v0.1",
    "SCI-PTC": "v0.1",
    "SCI-VAL": "v0.1",
    "SCI-ALIGN": "v0.1",
    "SCI-AST": "v0.1",
}
STAGE_A_PACKAGES = {}
FROZEN_R03_PACKAGES = {
    "SCI-BEAM": "v0.1",
}
FROZEN_MANIFEST_PACKAGES = {
    "SCI-ALIGN": "v0.1",
    "SCI-AST": "v0.1",
}
RENDERED_DRAFT_PACKAGES = {}


assert (ROOT / "INDEX.md").is_file()
assert (ROOT / "templates" / "SCOPE_BRIEF.md").is_file()

BOUNDARY_ROOT = ROOT / "boundaries" / "v0.1"
for name in (
    "README.md",
    "SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md",
    "DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md",
    "TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY.md",
    "WP2_BOUNDARY_CANDIDATE_CHANGE_MAP.md",
    "SCIENTIFIC_OWNER_APPROVAL_2026-08-23.md",
    "SOURCE_MANIFEST.md",
):
    assert (BOUNDARY_ROOT / name).is_file(), f"timestream boundary: missing {name}"

for package, version in {
    **COMPLETE_PACKAGES,
    **RENDERED_DRAFT_PACKAGES,
    **STAGE_A_PACKAGES,
}.items():
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
    if package in {**COMPLETE_PACKAGES, **RENDERED_DRAFT_PACKAGES}:
        for view in ("SCIENTIFIC-RATIONALE", "ENGINEERING-CONFORMANCE"):
            output = base / "pdf" / f"{package}-{view}-{version}.pdf"
            assert output.is_file() and output.stat().st_size > 0, f"{package}: missing {output.name}"
    if package in STAGE_A_PACKAGES:
        assert (base / "INTERNAL_DOSSIER.md").is_file(), f"{package}: missing INTERNAL_DOSSIER.md"
        assert (base / "AUTHOR_PACKET_MANIFEST.md").is_file(), f"{package}: missing AUTHOR_PACKET_MANIFEST.md"
        assert (base / "AUTHOR_SUPERSESSION_COVER.md").is_file(), f"{package}: missing AUTHOR_SUPERSESSION_COVER.md"
        assert (base / "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md").is_file(), f"{package}: missing AUTHOR_CONVENTIONS_AND_OWNERSHIP.md"
        assert (base / "pdf" / "README.md").is_file(), f"{package}: missing pdf/README.md"
        for name in COMMON:
            text = (base / "src" / "common" / name).read_text()
            assert "Reserved for implementation-blind Stage B" in text, f"{package}: non-placeholder common/{name}"
        for name in ("scientific-rationale.tex", "engineering-conformance.tex"):
            text = (base / "src" / name).read_text()
            assert "contains no normative science" in text, f"{package}: non-placeholder src/{name}"
    if package in RENDERED_DRAFT_PACKAGES:
        assert (base / "INTERNAL_DOSSIER.md").is_file(), f"{package}: missing INTERNAL_DOSSIER.md"
        for name in (
            "AUTHOR_PACKET_MANIFEST.md",
            "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
            "AUTHOR_SUPERSESSION_COVER.md",
        ):
            assert (base / name).is_file(), f"{package}: missing {name}"
        assert (base / "pdf" / "README.md").is_file(), f"{package}: missing pdf/README.md"
        assert (base / "AUTHOR_DRAFT_DECISIONS.md").is_file(), f"{package}: missing AUTHOR_DRAFT_DECISIONS.md"
        assert (base / "MANAGER_REVIEW_R0.1.md").is_file(), f"{package}: missing MANAGER_REVIEW_R0.1.md"
        rationale = (base / "src" / "scientific-rationale.tex").read_text()
        formal = (base / "src" / "engineering-conformance.tex").read_text()
        assert "r0.1" in rationale and "r0.1" in formal, f"{package}: stale document revision"
        for name in COMMON:
            token = "\\input{common/" + name + "}"
            assert formal.count(token) == 1, f"{package}: formal view must import {name} exactly once"
            assert rationale.count(token) == 1, f"{package}: rationale must import {name} exactly once"
    if package in FROZEN_R03_PACKAGES:
        for name in (
            "SCIENTIFIC_OWNER_REVIEW_R0.3.md",
            "CHANGE_LOG_R0.3.md",
            "CROSS_DOCUMENT_FOLLOWUP_R0.3.md",
            "CONSISTENCY_REPORT_R0.3.md",
        ):
            assert (base / name).is_file(), f"{package}: missing {name}"
        for name in (
            "SCI-BEAM-v0.1_SCIENCE-TEAM-RATIONALE_r0.3.pdf",
            "SCI-BEAM-v0.1_FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT_r0.3.pdf",
        ):
            output = base / "pdf" / name
            assert output.is_file() and output.stat().st_size > 0, f"{package}: missing {name}"
        rationale = (base / "src" / "scientific-rationale.tex").read_text()
        formal = (base / "src" / "engineering-conformance.tex").read_text()
        assert "r0.3" in rationale and "r0.3" in formal, f"{package}: stale final document revision"
        assert "DRAFT" not in rationale and "DRAFT" not in formal, f"{package}: frozen source labeled draft"
        assert "SCI-BEAM-REQ-" not in rationale and "SCI-BEAM-PRED-" not in rationale, f"{package}: rationale contains formal inventory"
        for name in COMMON:
            token = "\\input{common/" + name.removesuffix(".tex") + "}"
            assert formal.count(token) == 1, f"{package}: formal view must import {name} exactly once"
            assert token not in rationale, f"{package}: rationale must remain scientist-facing"

for package, version in FROZEN_MANIFEST_PACKAGES.items():
    base = ROOT / "packages" / package / version
    readme = (base / "README.md").read_text()
    assert "Scientific authority frozen; implementation conformity not yet" in readme
    assert (base / "SOURCE_MANIFEST.md").is_file(), f"{package}: missing SOURCE_MANIFEST.md"
    assert (base / "history" / "r0.2" / "SOURCE_MANIFEST.md").is_file(), (
        f"{package}: missing retained r0.2 manifest"
    )

align_boundary = ROOT / "packages" / "SCI-ALIGN" / "v0.1" / "SCI-ALIGN_TO_SCI-AST_BOUNDARY.md"
ast_boundary = ROOT / "packages" / "SCI-AST" / "v0.1" / "SCI-ALIGN_TO_SCI-AST_BOUNDARY.md"
assert align_boundary.read_bytes() == ast_boundary.read_bytes(), "ALIGN/AST boundary copies differ"

print("scientific_contract_layout=PASS")
print("complete_packages=" + ",".join(f"{name}/{version}" for name, version in COMPLETE_PACKAGES.items()))
print("rendered_draft_packages=" + ",".join(f"{name}/{version}" for name, version in RENDERED_DRAFT_PACKAGES.items()))
print("frozen_r03_packages=" + ",".join(f"{name}/{version}" for name, version in FROZEN_R03_PACKAGES.items()))
print("frozen_manifest_packages=" + ",".join(f"{name}/{version}" for name, version in FROZEN_MANIFEST_PACKAGES.items()))
print("stage_a_packages=" + ",".join(f"{name}/{version}" for name, version in STAGE_A_PACKAGES.items()))
