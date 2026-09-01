#!/usr/bin/env python3
"""Verify the content-bound SCI-FLT-MATCHED Stage B r0.5 authority set."""

from hashlib import sha256
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STAGE_B_DRAFT_MANIFEST.md"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


text = MANIFEST.read_text()
assert "SCI-FLT-MATCHED_STAGE_B_AUTHORITY v0.1/r0.5" in text
rows = re.findall(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*`[^`]+`\s*\|\s*(\d+)\s*\|"
    r"\s*`([0-9a-f]{64})`\s*\|.*\|$",
    text,
    re.MULTILINE,
)
assert rows, "manifest has no object rows"
for expected, (number, relative, expected_bytes, expected_hash) in enumerate(rows, start=1):
    assert int(number) == expected, f"row order mismatch {number}"
    path = ROOT / relative
    assert path.is_file(), f"missing {relative}"
    assert path.stat().st_size == int(expected_bytes), f"byte-count mismatch {relative}"
    assert digest(path) == expected_hash, f"hash mismatch {relative}"

sidecar = (ROOT / "STAGE_B_DRAFT_MANIFEST.sha256").read_text().strip()
assert sidecar == f"{digest(MANIFEST)}  STAGE_B_DRAFT_MANIFEST.md"
assert digest(ROOT / "AUTHOR_PACKET_MANIFEST.md") == (
    "255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8"
)

print("sci_flt_matched_stage_b_draft=PASS")
print(f"draft_objects={len(rows)}")
print(f"draft_manifest_sha256={digest(MANIFEST)}")
print("draft_revision=r0.5")
print("scientific_authority_frozen=false")
