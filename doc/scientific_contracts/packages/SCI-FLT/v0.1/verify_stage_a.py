#!/usr/bin/env python3
"""Verify the SCI-FLT v0.1 recovery-first Stage A package."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

REQUIRED = (
    "README.md",
    "PRIOR_WORK.md",
    "INTERNAL_DOSSIER.md",
    "SCOPE_BRIEF.md",
    "OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md",
    "AUTHOR_BOUNDARY_INPUTS.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "AUTHOR_SUPERSESSION_COVER.md",
    "AUTHOR_PACKET_MANIFEST.md",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "CROSSWALK.md",
    "DECISION_LOG.md",
    "pdf/README.md",
    "src/scientific-rationale.tex",
    "src/engineering-conformance.tex",
)

COMMON = (
    "notation.tex",
    "definitions.tex",
    "equations.tex",
    "assumptions.tex",
    "requirements.tex",
    "edge_cases.tex",
)

AUTHOR_OBJECTS = (
    "SCOPE_BRIEF.md",
    "AUTHOR_SUPERSESSION_COVER.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "AUTHOR_BOUNDARY_INPUTS.md",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
)


for relative in REQUIRED:
    path = ROOT / relative
    assert path.is_file() and path.stat().st_size > 0, f"missing {relative}"

for name in COMMON:
    path = ROOT / "src" / "common" / name
    assert path.is_file(), f"missing common/{name}"
    assert "Reserved for implementation-blind Stage B" in path.read_text(), (
        f"non-placeholder common/{name}"
    )

for name in ("scientific-rationale.tex", "engineering-conformance.tex"):
    text = (ROOT / "src" / name).read_text()
    assert "contains no normative science" in text, f"non-placeholder src/{name}"

readme = (ROOT / "README.md").read_text()
assert "Stage B not authorized" in readme
assert "implementation-conformity" in readme

decisions = (ROOT / "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md").read_text()
assert "Status: unavailable" in decisions
assert "contains no approved" in decisions

manifest = (ROOT / "AUTHOR_PACKET_MANIFEST.md").read_text()
for name in AUTHOR_OBJECTS:
    assert f"`{name}`" in manifest, f"manifest omits {name}"
assert manifest.count("| 1 |") == 1
assert "UNAVAILABLE_PENDING_WALKTHROUGH" in manifest

for name in AUTHOR_OBJECTS:
    text = (ROOT / name).read_text()
    for forbidden in (
        "include/citlali/",
        "src/citlali/",
        "validation/",
        "tests/",
        "test/",
    ):
        assert forbidden not in text, f"author object {name} leaks {forbidden}"

assert not (ROOT / "stage_b").exists(), "Stage B directory must not exist"

for markdown in ROOT.rglob("*.md"):
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        local_target = target.split("#", 1)[0]
        assert (markdown.parent / local_target).resolve().exists(), (
            f"broken local link in {markdown.relative_to(ROOT)}: {target}"
        )

print("sci_flt_stage_a=PASS")
print("author_packet_objects=7")
print("stage_b_authorized=false")
