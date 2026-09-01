#!/usr/bin/env python3
"""Verify the SCI-FLT-MATCHED v0.1 Stage A author packet."""

from hashlib import sha256
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

REQUIRED = (
    "README.md", "PRIOR_WORK.md", "INTERNAL_DOSSIER.md", "SCOPE_BRIEF.md",
    "AUTHOR_SUPERSESSION_COVER.md", "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md", "AUTHOR_BOUNDARIES.md",
    "REQUIRED_AUTHORED_OPTION_SETS.md", "AUTHOR_PACKET_MANIFEST.md",
    "AUTHOR_PACKET_MANIFEST.sha256", "STAGE_A_CHANGE_LOG.md", "CROSSWALK.md",
    "SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md",
    "pdf/README.md", "src/scientific-rationale.tex",
    "src/engineering-conformance.tex", "src/common/README.md",
)

AUTHOR_OBJECTS = (
    "SCOPE_BRIEF.md", "AUTHOR_SUPERSESSION_COVER.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md", "AUTHOR_BOUNDARIES.md",
    "REQUIRED_AUTHORED_OPTION_SETS.md",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


for relative in REQUIRED:
    path = ROOT / relative
    assert path.is_file() and path.stat().st_size > 0, f"missing {relative}"

readme = (ROOT / "README.md").read_text()
assert "Stage B r0.2" in readme or "implementation-blind Stage B r0.1 draft returned" in readme
assert "SCI-FLT-MATCHED" in readme

scope = (ROOT / "SCOPE_BRIEF.md").read_text()
assert "Program Adherence And Prior-Work Recovery" in scope
assert "A_hat(x) = N(x) / D(x)" in scope

decisions = (ROOT / "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md").read_text()
for decision in range(1, 14):
    assert f"ODQ-{decision:03d}" in decisions, f"owner record omits ODQ-{decision:03d}"
assert "SCI-FLT-MATCHED_OWNER_DECISIONS v0.1/r0.1" in decisions
assert "first-class FLT→FRUIT" in decisions

options = (ROOT / "REQUIRED_AUTHORED_OPTION_SETS.md").read_text()
for option in range(1, 7):
    assert f"SCI-FLT-MATCHED-AO-{option:03d}" in options, f"missing AO-{option:03d}"
assert re.search(r"radially\s+symmetrized average map noise PSD", options)

manifest_path = ROOT / "AUTHOR_PACKET_MANIFEST.md"
manifest = manifest_path.read_text()
assert "SCI-FLT-MATCHED_AUTHOR_PACKET v0.1/r0.1" in manifest
assert "HASH_PENDING" not in manifest
row_pattern = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$",
    re.MULTILINE,
)
rows = row_pattern.findall(manifest)
assert len(rows) == len(AUTHOR_OBJECTS), f"manifest hashed rows={len(rows)}"
for expected_index, expected_name in enumerate(AUTHOR_OBJECTS, start=1):
    row_index, row_name, row_digest = rows[expected_index - 1]
    assert int(row_index) == expected_index
    assert row_name == expected_name
    assert digest(ROOT / row_name) == row_digest, f"hash mismatch {row_name}"
assert "SELF_HASH_IN_SHA256_SIDECAR" in manifest

sidecar = (ROOT / "AUTHOR_PACKET_MANIFEST.sha256").read_text().strip()
assert sidecar == f"{digest(manifest_path)}  AUTHOR_PACKET_MANIFEST.md"

approval = (ROOT / "SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md").read_text()
assert digest(manifest_path) in approval
assert "authorize a fresh implementation-blind Stage B author" in approval

for name in AUTHOR_OBJECTS:
    text = (ROOT / name).read_text()
    for forbidden in ("include/citlali/", "src/citlali/", "validation/", "tests/"):
        assert forbidden not in text, f"author object {name} leaks {forbidden}"

assert not (ROOT / "stage_b").exists(), "Stage B directory must not exist"

for markdown in ROOT.rglob("*.md"):
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        local = target.split("#", 1)[0]
        assert (markdown.parent / local).resolve().exists(), (
            f"broken local link in {markdown.relative_to(ROOT)}: {target}"
        )

print("sci_flt_matched_stage_a=PASS")
print(f"author_packet_objects={len(AUTHOR_OBJECTS) + 1}")
print(f"author_packet_manifest_sha256={digest(manifest_path)}")
print("stage_b_authorized=true")
