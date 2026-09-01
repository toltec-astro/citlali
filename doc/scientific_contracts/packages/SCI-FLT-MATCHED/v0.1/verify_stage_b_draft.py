#!/usr/bin/env python3
"""Verify the content-bound SCI-FLT-MATCHED Stage B r0.6 preflight set."""

from hashlib import sha256
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STAGE_B_DRAFT_MANIFEST.md"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


text = MANIFEST.read_text()
assert "SCI-FLT-MATCHED_STAGE_B_CONDITIONAL_FREEZE_PREFLIGHT v0.1/r0.6" in text
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
assert "255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8" in text
assert digest(ROOT / "SCIENTIFIC_OWNER_R0.6_DIRECTIVE_2026-09-01.md") == (
    "5758640064918b2d3021afc7ea63ffba063ba7b1abbb66dc6d43d945ed73ebd3"
)
closure = (ROOT / "SOURCE_BYTE_AND_LINK_CLOSURE_R0.6.md").read_text()
assert "Unresolved active-object local Markdown links: `0`." in closure

# The bundle note exists only in the standalone extracted archive root. In that
# context, run the bundled auditor over every Markdown file, including the
# manifest and note. Repository-context verification stays authority-object
# focused because the package README intentionally links to program history.
if (ROOT / "BUNDLE_README_R0.6.md").is_file():
    import importlib.util

    auditor_path = ROOT / "build" / "audit_bundle_links.py"
    spec = importlib.util.spec_from_file_location("audit_bundle_links", auditor_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    link_result = module.audit(ROOT)
    assert link_result["status"] == "PASS", link_result

print("sci_flt_matched_stage_b_draft=PASS")
print(f"draft_objects={len(rows)}")
print(f"draft_manifest_sha256={digest(MANIFEST)}")
print("draft_revision=r0.6")
print("owner_dispositions_complete=false")
print("scientific_authority_frozen=false")
