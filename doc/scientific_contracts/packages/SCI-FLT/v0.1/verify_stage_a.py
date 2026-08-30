#!/usr/bin/env python3
"""Verify the repaired SCI-FLT-FIXED v0.1 Stage A package."""

from hashlib import sha256
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
    "AUTHOR_PACKET_MANIFEST.sha256",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "SCIENTIFIC_OWNER_FINAL_STAGE_A_SCOPE_DIRECTION_2026-08-30.md",
    "SCI-MAP_TO_SCI-FLT-FIXED_BOUNDARY.md",
    "SCI-JINC_TO_SCI-FLT-FIXED_BOUNDARY.md",
    "SCI-FLT-FIXED_TO_SCI-NOI_BOUNDARY.md",
    "FIXED_LINEAR_OPERATOR_AND_CONVOLUTION_SPECIFICATION.md",
    "WCS_KERNEL_DISCRETIZATION_DECISION_TABLE.md",
    "EDGE_MISSING_NONFINITE_METHOD_DECISION_TABLE.md",
    "NORMALIZATION_UNIT_BEAM_DECISION_TABLE.md",
    "RESPONSE_NULLSPACE_COVARIANCE_PRODUCT_TABLE.md",
    "OBSERVATION_COADD_NONCOMMUTATION_TABLE.md",
    "SCI_FLT_VAL_PROFILE_DRAFTS.md",
    "FIXED_PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md",
    "CROSSWALK.md",
    "DECISION_LOG.md",
    "STAGE_A_CHANGE_LOG.md",
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
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md",
    "SCI-MAP_TO_SCI-FLT-FIXED_BOUNDARY.md",
    "SCI-JINC_TO_SCI-FLT-FIXED_BOUNDARY.md",
    "SCI-FLT-FIXED_TO_SCI-NOI_BOUNDARY.md",
    "FIXED_LINEAR_OPERATOR_AND_CONVOLUTION_SPECIFICATION.md",
    "WCS_KERNEL_DISCRETIZATION_DECISION_TABLE.md",
    "EDGE_MISSING_NONFINITE_METHOD_DECISION_TABLE.md",
    "NORMALIZATION_UNIT_BEAM_DECISION_TABLE.md",
    "RESPONSE_NULLSPACE_COVARIANCE_PRODUCT_TABLE.md",
    "OBSERVATION_COADD_NONCOMMUTATION_TABLE.md",
    "SCI_FLT_VAL_PROFILE_DRAFTS.md",
    "FIXED_PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
assert "SCI-FLT-FIXED" in readme

decisions = (ROOT / "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md").read_text()
assert "SCI-FLT-FIXED_OWNER_DECISIONS v0.1/r0.1" in decisions
for decision in ("ODQ-101", "ODQ-102A", "ODQ-102B", "ODQ-103", "ODQ-104",
                 "ODQ-105", "ODQ-106", "ODQ-107", "ODQ-108"):
    assert decision in decisions, f"owner record omits {decision}"

manifest_path = ROOT / "AUTHOR_PACKET_MANIFEST.md"
manifest = manifest_path.read_text()
assert "SCI-FLT-FIXED_AUTHOR_PACKET v0.1/r0.1" in manifest
assert "HASH_PENDING" not in manifest

row_pattern = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$",
    re.MULTILINE,
)
rows = row_pattern.findall(manifest)
assert len(rows) == len(AUTHOR_OBJECTS), f"manifest rows={len(rows)}"
for expected_index, expected_name in enumerate(AUTHOR_OBJECTS, start=1):
    row_index, row_name, row_digest = rows[expected_index - 1]
    assert int(row_index) == expected_index, f"manifest index mismatch {row_index}"
    assert row_name == expected_name, f"manifest object mismatch {row_name}"
    assert digest(ROOT / row_name) == row_digest, f"hash mismatch {row_name}"

manifest_digest_file = (ROOT / "AUTHOR_PACKET_MANIFEST.sha256").read_text().strip()
expected_manifest_digest = f"{digest(manifest_path)}  AUTHOR_PACKET_MANIFEST.md"
assert manifest_digest_file == expected_manifest_digest, "manifest digest mismatch"

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

print("sci_flt_fixed_stage_a=PASS")
print(f"author_packet_objects={len(AUTHOR_OBJECTS)}")
print(f"author_packet_manifest_sha256={digest(manifest_path)}")
print("stage_b_authorized=false")
