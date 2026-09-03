#!/usr/bin/env python3
"""Verify the bounded SCI-POINT v0.1 Stage A packet."""

from __future__ import annotations

from hashlib import sha256
import re
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parent

REQUIRED = {
    "README.md",
    "SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_003A_003B_DIRECTION_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_004_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-09-02.md",
    "SCIENTIFIC_OWNER_METHOD_AUTHORITY_RESPONSE_2026-09-02.md",
    "SCIENTIFIC_OWNER_R0_3_CLOSURE_DIRECTIVE_2026-09-02.md",
    "POINT_COMPATIBILITY_METHOD_RECOVERY_BRIEF.md",
    "POINT_FORMAL_ERROR_METHOD_RECOVERY_BRIEF.md",
    "POINT_FULL_MAP_RMS_METHOD_RECOVERY_BRIEF.md",
    "PRIOR_WORK.md",
    "WORKING_WHEEL_ADOPTION_REGISTER.md",
    "INTERNAL_DOSSIER.md",
    "SCOPE_BRIEF.md",
    "OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md",
    "OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "PARENT_ROUTE_AND_CLAIM_MATRIX.md",
    "CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md",
    "AUTHOR_SCOPE_BRIEF.md",
    "AUTHOR_SUPERSESSION_COVER.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "AUTHOR_WORKING_METHOD_CONSTRAINTS.md",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "AUTHOR_PARENT_ROUTE_AND_CLAIM_MATRIX.md",
    "AUTHOR_BOUNDARIES_AND_UNAVAILABLE_STATES.md",
    "AUTHOR_PREDECESSOR_BOUNDARY_INPUTS.md",
    "AUTHOR_COMPATIBILITY_METHOD_EXTRACT.md",
    "AUTHOR_FORMAL_ERROR_METHOD_EXTRACT.md",
    "AUTHOR_FULL_MAP_RMS_METHOD_EXTRACT.md",
    "AUTHOR_SOURCE_REFERENCE_AND_DISPLACEMENT_BOUNDARY.md",
    "AUTHOR_ALTAZ_TANGENT_BASIS_BOUNDARY.md",
    "AUTHOR_GAUSSIAN_MODEL_AND_PARAMETER_CONVENTIONS.md",
    "AUTHOR_OBJECTIVE_WEIGHTING_AND_FORMAL_ERROR_DECISION.md",
    "AUTHOR_SEARCH_ASSOCIATION_AND_FALLBACK_SPECIFICATION.md",
    "AUTHOR_APPLICABILITY_FACT_TABLE.md",
    "AUTHOR_ROUTE_SPECIFIC_COMPATIBILITY_TABLE.md",
    "AUTHOR_PARENT_SIGNAL_ROLE_AND_BOUNDARY_STATUS.md",
    "AUTHOR_PRODUCT_AND_CLAIM_DEPENDENCY_MATRIX.md",
    "AUTHOR_ZERO_BACKGROUND_RESIDUAL_IDENTIFIABILITY.md",
    "AUTHOR_LIFECYCLE_AND_NAMED_USE_DISPOSITIONS.md",
    "AUTHOR_DIAGNOSTIC_FORMULA_TABLE.md",
    "AUTHOR_RESPONSE_AND_BIAS_STATES.md",
    "AUTHOR_UNCERTAINTY_BUDGET.md",
    "AUTHOR_MAP_TO_POINT_BOUNDARY.md",
    "AUTHOR_JINC_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_FIXED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_MATCHED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FRUIT_TERMINAL_ANCESTRY_ENVELOPE.md",
    "AUTHOR_POINT_TO_POINTING_SUPPORT_ENVELOPE.md",
    "AUTHOR_POINT_TO_TELESCOPE_QC_ENVELOPE.md",
    "AUTHOR_POINT_TO_CAL_TOLPROJ_ENVELOPE.md",
    "AUTHOR_POLICY_PROFILE_RECORDS.md",
    "AUTHOR_FALSIFIABLE_PREDICTIONS.md",
    "AUTHOR_STAGE_A_SEMANTIC_CHANGE_REPORT.md",
    "AUTHOR_PACKET_MANIFEST.md",
    "AUTHOR_PACKET_MANIFEST.sha256",
    "PROPOSED_SANITIZED_AUTHOR_INPUTS.md",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SOURCE_IDENTITY_MANIFEST.md",
    "DECISION_LOG.md",
    "CROSSWALK.md",
    "STAGE_A_CHANGE_LOG.md",
    "src/common/notation.tex",
    "src/common/definitions.tex",
    "src/common/equations.tex",
    "src/common/assumptions.tex",
    "src/common/requirements.tex",
    "src/common/edge_cases.tex",
    "src/scientific-rationale.tex",
    "src/engineering-conformance.tex",
    "pdf/README.md",
    "author_packet/README.md",
    "author_packet/SCI-POINT-v0.1-r0.1-stage-b-author-packet.tar.gz",
    "author_packet/SCI-POINT-v0.1-r0.1-stage-b-author-packet.tar.gz.sha256",
    "author_packet/SCI-POINT-v0.1-r0.2-stage-b-author-packet.tar.gz",
    "author_packet/SCI-POINT-v0.1-r0.2-stage-b-author-packet.tar.gz.sha256",
    "author_packet/SCI-POINT-v0.1-r0.2-stage-b-author-packet.tar.gz.bytes",
    "author_packet/create_author_packet_r0_3.py",
    "author_packet/verify_author_packet_r0_3.py",
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz",
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz.sha256",
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz.bytes",
}

PLACEHOLDERS = {
    "src/common/notation.tex",
    "src/common/definitions.tex",
    "src/common/equations.tex",
    "src/common/assumptions.tex",
    "src/common/requirements.tex",
    "src/common/edge_cases.tex",
    "src/scientific-rationale.tex",
    "src/engineering-conformance.tex",
}

AUTHOR_OBJECTS = [
    "AUTHOR_SCOPE_BRIEF.md",
    "AUTHOR_SUPERSESSION_COVER.md",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "AUTHOR_WORKING_METHOD_CONSTRAINTS.md",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "AUTHOR_PARENT_ROUTE_AND_CLAIM_MATRIX.md",
    "AUTHOR_BOUNDARIES_AND_UNAVAILABLE_STATES.md",
    "AUTHOR_PREDECESSOR_BOUNDARY_INPUTS.md",
    "AUTHOR_COMPATIBILITY_METHOD_EXTRACT.md",
    "AUTHOR_FORMAL_ERROR_METHOD_EXTRACT.md",
    "AUTHOR_FULL_MAP_RMS_METHOD_EXTRACT.md",
    "AUTHOR_SOURCE_REFERENCE_AND_DISPLACEMENT_BOUNDARY.md",
    "AUTHOR_ALTAZ_TANGENT_BASIS_BOUNDARY.md",
    "AUTHOR_GAUSSIAN_MODEL_AND_PARAMETER_CONVENTIONS.md",
    "AUTHOR_OBJECTIVE_WEIGHTING_AND_FORMAL_ERROR_DECISION.md",
    "AUTHOR_SEARCH_ASSOCIATION_AND_FALLBACK_SPECIFICATION.md",
    "AUTHOR_APPLICABILITY_FACT_TABLE.md",
    "AUTHOR_ROUTE_SPECIFIC_COMPATIBILITY_TABLE.md",
    "AUTHOR_PARENT_SIGNAL_ROLE_AND_BOUNDARY_STATUS.md",
    "AUTHOR_PRODUCT_AND_CLAIM_DEPENDENCY_MATRIX.md",
    "AUTHOR_ZERO_BACKGROUND_RESIDUAL_IDENTIFIABILITY.md",
    "AUTHOR_LIFECYCLE_AND_NAMED_USE_DISPOSITIONS.md",
    "AUTHOR_DIAGNOSTIC_FORMULA_TABLE.md",
    "AUTHOR_RESPONSE_AND_BIAS_STATES.md",
    "AUTHOR_UNCERTAINTY_BUDGET.md",
    "AUTHOR_MAP_TO_POINT_BOUNDARY.md",
    "AUTHOR_JINC_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_FIXED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_MATCHED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FRUIT_TERMINAL_ANCESTRY_ENVELOPE.md",
    "AUTHOR_POINT_TO_POINTING_SUPPORT_ENVELOPE.md",
    "AUTHOR_POINT_TO_TELESCOPE_QC_ENVELOPE.md",
    "AUTHOR_POINT_TO_CAL_TOLPROJ_ENVELOPE.md",
    "AUTHOR_POLICY_PROFILE_RECORDS.md",
    "AUTHOR_FALSIFIABLE_PREDICTIONS.md",
    "AUTHOR_STAGE_A_SEMANTIC_CHANGE_REPORT.md",
]
AUTHOR_MANIFEST = ROOT / "AUTHOR_PACKET_MANIFEST.md"
AUTHOR_MANIFEST_SIDECAR = ROOT / "AUTHOR_PACKET_MANIFEST.sha256"
AUTHOR_ARCHIVE = (
    ROOT
    / "author_packet"
    / "SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz"
)
AUTHOR_ARCHIVE_SIDECAR = Path(f"{AUTHOR_ARCHIVE}.sha256")
AUTHOR_ARCHIVE_BYTES_SIDECAR = Path(f"{AUTHOR_ARCHIVE}.bytes")
AUTHOR_ARCHIVE_PREFIX = "SCI-POINT-v0.1-r0.3-stage-b-author-packet"
HISTORICAL_R0_1_ARCHIVE = (
    ROOT
    / "author_packet"
    / "SCI-POINT-v0.1-r0.1-stage-b-author-packet.tar.gz"
)
HISTORICAL_R0_1_ARCHIVE_SIDECAR = Path(f"{HISTORICAL_R0_1_ARCHIVE}.sha256")
HISTORICAL_R0_2_ARCHIVE = (
    ROOT
    / "author_packet"
    / "SCI-POINT-v0.1-r0.2-stage-b-author-packet.tar.gz"
)
HISTORICAL_R0_2_ARCHIVE_SIDECAR = Path(f"{HISTORICAL_R0_2_ARCHIVE}.sha256")
HISTORICAL_R0_2_BYTES_SIDECAR = Path(f"{HISTORICAL_R0_2_ARCHIVE}.bytes")


def fail(message: str) -> None:
    raise SystemExit(f"SCI-POINT Stage A verification failed: {message}")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
if missing:
    fail(f"missing required files: {', '.join(missing)}")

for relative in PLACEHOLDERS:
    text = (ROOT / relative).read_text(encoding="utf-8")
    if "placeholder" not in text.lower():
        fail(f"Stage B source is not explicitly a placeholder: {relative}")
    if re.search(r"SCI-POINT-(?:REQ|PRED)-\d+", text):
        fail(f"normative IDs appeared during Stage A: {relative}")

pdfs = sorted(ROOT.joinpath("pdf").glob("*.pdf"))
if pdfs:
    fail(f"Stage A unexpectedly contains rendered PDFs: {', '.join(p.name for p in pdfs)}")

author_manifest = AUTHOR_MANIFEST.read_text(encoding="utf-8")
if "SCI-POINT_AUTHOR_PACKET_MANIFEST v0.1/r0.3" not in author_manifest:
    fail("wrong author-packet manifest identity")
if "HASH_PENDING" in author_manifest:
    fail("author-packet manifest contains pending hashes")
if "Stage B not launched" not in author_manifest:
    fail("author-packet manifest lacks the Stage B dispatch gate")

author_rows = re.findall(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$",
    author_manifest,
    re.MULTILINE,
)
if [name for _, name, _ in author_rows] != AUTHOR_OBJECTS:
    fail("author-packet object membership or order differs from the exclusive list")
for expected_index, (index, name, expected_hash) in enumerate(author_rows, start=1):
    if int(index) != expected_index:
        fail(f"author-packet manifest index mismatch: {index}")
    if digest(ROOT / name) != expected_hash:
        fail(f"author-packet object hash mismatch: {name}")

expected_manifest_sidecar = f"{digest(AUTHOR_MANIFEST)}  {AUTHOR_MANIFEST.name}"
if AUTHOR_MANIFEST_SIDECAR.read_text(encoding="utf-8").strip() != expected_manifest_sidecar:
    fail("author-packet manifest digest sidecar mismatch")

expected_archive_sidecar = f"{digest(AUTHOR_ARCHIVE)}  {AUTHOR_ARCHIVE.name}"
if AUTHOR_ARCHIVE_SIDECAR.read_text(encoding="utf-8").strip() != expected_archive_sidecar:
    fail("author-packet archive digest sidecar mismatch")
expected_archive_bytes = f"{AUTHOR_ARCHIVE.stat().st_size}  {AUTHOR_ARCHIVE.name}"
if AUTHOR_ARCHIVE_BYTES_SIDECAR.read_text(encoding="utf-8").strip() != expected_archive_bytes:
    fail("author-packet archive byte-count sidecar mismatch")

expected_historical_sidecar = (
    f"{digest(HISTORICAL_R0_1_ARCHIVE)}  {HISTORICAL_R0_1_ARCHIVE.name}"
)
if (
    HISTORICAL_R0_1_ARCHIVE_SIDECAR.read_text(encoding="utf-8").strip()
    != expected_historical_sidecar
):
    fail("historical r0.1 archive digest sidecar mismatch")

expected_historical_r0_2_sidecar = (
    f"{digest(HISTORICAL_R0_2_ARCHIVE)}  {HISTORICAL_R0_2_ARCHIVE.name}"
)
if (
    HISTORICAL_R0_2_ARCHIVE_SIDECAR.read_text(encoding="utf-8").strip()
    != expected_historical_r0_2_sidecar
):
    fail("historical r0.2 archive digest sidecar mismatch")
expected_historical_r0_2_bytes = (
    f"{HISTORICAL_R0_2_ARCHIVE.stat().st_size}  {HISTORICAL_R0_2_ARCHIVE.name}"
)
if (
    HISTORICAL_R0_2_BYTES_SIDECAR.read_text(encoding="utf-8").strip()
    != expected_historical_r0_2_bytes
):
    fail("historical r0.2 archive byte-count sidecar mismatch")

expected_archive_members = {
    f"{AUTHOR_ARCHIVE_PREFIX}/{name}"
    for name in AUTHOR_OBJECTS
    + [AUTHOR_MANIFEST.name, AUTHOR_MANIFEST_SIDECAR.name]
}
with tarfile.open(AUTHOR_ARCHIVE, mode="r:gz") as author_archive:
    members = author_archive.getmembers()
    if {member.name for member in members} != expected_archive_members:
        fail("author-packet archive membership differs from the exclusive manifest")
    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            fail(f"author-packet archive has an unsafe path: {member.name}")
        if not member.isfile() or member.issym() or member.islnk():
            fail(f"author-packet archive has a non-regular member: {member.name}")
        relative = member_path.relative_to(AUTHOR_ARCHIVE_PREFIX).as_posix()
        extracted = author_archive.extractfile(member)
        if extracted is None or extracted.read() != (ROOT / relative).read_bytes():
            fail(f"author-packet archive byte mismatch: {relative}")

readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
readme_words = re.sub(r"\s+", " ", readme)
scope = ROOT.joinpath("SCOPE_BRIEF.md").read_text(encoding="utf-8")
ledger = ROOT.joinpath("SCIENTIFIC_OWNER_DECISION_LEDGER.md").read_text(
    encoding="utf-8"
)

for phrase in (
    "Stage B not authorized",
    "working wheel",
    "blank-field",
    "SCI-BEAM",
):
    if phrase.lower() not in readme_words.lower():
        fail(f"README lacks required boundary phrase: {phrase}")

for family in ("MAP", "JINC", "FLT-FIXED", "FLT-MATCHED"):
    if family not in scope:
        fail(f"scope does not distinguish candidate parent family {family}")

if "FRUIT is not a separate\n  POINT parent family" not in scope:
    fail("scope does not preserve FRUIT as lineage on a terminal map type")
if "exclude coadd parents from base v0.1" not in scope:
    fail("scope does not preserve the base-v0.1 coadd exclusion")

odqs = sorted(set(re.findall(r"SCI-POINT-ODQ-(\d{3})", ledger)))
if odqs != [f"{i:03d}" for i in range(1, 10)]:
    fail(f"owner decision ledger is not contiguous ODQ-001--009: {odqs}")

if "SCI-POINT-ODQ-001` — Does POINT own a cross-array aggregate? — **decided**" not in ledger:
    fail("ODQ-001 is not marked decided")
if "SCI-POINT-ODQ-002` — Measurement versus correction construction — **decided**" not in ledger:
    fail("ODQ-002 is not marked decided")
if "SCI-POINT-ODQ-003` — Admitted observation-local parent routes — **decided**" not in ledger:
    fail("ODQ-003 is not marked decided")
if "SCI-POINT-ODQ-004` — Compatibility estimator — **decided**" not in ledger:
    fail("ODQ-004 is not marked decided")
if "SCI-POINT-ODQ-005` — Center, search, support, and constraints — **decided**" not in ledger:
    fail("ODQ-005 is not marked decided")
if "SCI-POINT-ODQ-006` — Per-array acceptance and partial success — **decided**" not in ledger:
    fail("ODQ-006 is not marked decided")
if "SCI-POINT-ODQ-007` — Formal covariance baseline — **decided**" not in ledger:
    fail("ODQ-007 is not marked decided")
if "SCI-POINT-ODQ-008` — Amplitude and effective shape — **decided**" not in ledger:
    fail("ODQ-008 is not marked decided")
if "SCI-POINT-ODQ-009` — Named-use VAL profiles — **decided**" not in ledger:
    fail("ODQ-009 is not marked decided")
for subdecision in ("SCI-POINT-ODQ-003A", "SCI-POINT-ODQ-003B"):
    if f"`{subdecision}` | decided" not in ledger:
        fail(f"{subdecision} is not marked decided")
if "ODQ-001 through ODQ-009 are decided; no bounded ODQ remains open" not in readme_words:
    fail("README does not preserve the remaining open decision set")

owner_methods = ROOT.joinpath(
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md"
).read_text(encoding="utf-8")
for method in (
    "POINT-COMPATIBILITY-METHOD v0.1",
    "POINT-FORMAL-ERROR-METHOD v0.1",
    "POINT-FULL-MAP-RMS-METHOD v0.1",
):
    if method not in owner_methods:
        fail(f"owner method-authority response omits {method}")
if owner_methods.count("unavailable_pending_separate_owner_approval") < 3:
    fail("owner method-authority table does not retain three unavailable gates")

lifecycle = ROOT.joinpath(
    "AUTHOR_LIFECYCLE_AND_NAMED_USE_DISPOSITIONS.md"
).read_text(encoding="utf-8")
for phrase in (
    "request",
    "applicability",
    "eligibility",
    "realization",
    "diagnostic_display_only",
    "not used as a producer state or\nSCI-VAL eligibility value",
):
    if phrase not in lifecycle:
        fail(f"named-use typing is incomplete: {phrase}")

policy = ROOT.joinpath("AUTHOR_POLICY_PROFILE_RECORDS.md").read_text(encoding="utf-8")
if "Eligibility is exactly `eligible`, `ineligible`, or `decision_unavailable`" not in policy:
    fail("SCI-VAL eligibility domain is not exactly three-valued")

dependency = ROOT.joinpath(
    "AUTHOR_PRODUCT_AND_CLAIM_DEPENDENCY_MATRIX.md"
).read_text(encoding="utf-8")
for product in (
    "`POINT-FIT-RESULT`",
    "`POINT-SOURCE-ASSOCIATION-STATE`",
    "processed-map displacement measurement",
    "formal errors",
    "dynamic-range diagnostic",
    "photometric transfer",
):
    if product not in dependency:
        fail(f"product/claim dependency matrix omits {product}")

parent_roles = ROOT.joinpath(
    "AUTHOR_PARENT_SIGNAL_ROLE_AND_BOUNDARY_STATUS.md"
).read_text(encoding="utf-8")
for role in (
    "MAP-SIGNAL/OBSERVATION-LEVEL-NORMALIZED@1",
    "JINC-SIGNAL/NORMALIZED-JINC-MAP@1",
    "FLT-FIXED-SIGNAL/TRANSFORMED-MAP@1",
    "FLT-MATCHED-SIGNAL/MATCHED-TEMPLATE-AMPLITUDE-FIELD@1",
):
    if role not in parent_roles:
        fail(f"parent signal-role table omits {role}")

for boundary in (
    "AUTHOR_MAP_TO_POINT_BOUNDARY.md",
    "AUTHOR_JINC_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_FIXED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_MATCHED_TO_POINT_BOUNDARY.md",
):
    text = ROOT.joinpath(boundary).read_text(encoding="utf-8")
    if "Status: draft boundary requirements" not in text:
        fail(f"parent boundary is not explicitly draft requirements: {boundary}")
    if "source\ndigest not bound" not in text or "numerical route unavailable" not in text:
        fail(f"parent boundary overstates numerical binding: {boundary}")

source_boundary = ROOT.joinpath(
    "AUTHOR_SOURCE_REFERENCE_AND_DISPLACEMENT_BOUNDARY.md"
).read_text(encoding="utf-8")
if "`Delta_POINT = mu_fitted - mu_expected`" not in source_boundary:
    fail("source-reference boundary lacks exact displacement definition")

for name in AUTHOR_OBJECTS:
    text = ROOT.joinpath(name).read_text(encoding="utf-8")
    if "complete, diagnostic-only, or unavailable" in text:
        fail(f"author object retains diagnostic_only as producer state: {name}")

link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
for path in sorted(ROOT.glob("*.md")):
    text = path.read_text(encoding="utf-8")
    for target in link_pattern.findall(text):
        if "://" in target or target.startswith("#"):
            continue
        link_target = target.split("#", 1)[0]
        if not link_target:
            continue
        if not (path.parent / link_target).resolve().exists():
            fail(f"broken local link in {path.name}: {target}")

print(
    "SCI-POINT Stage A verification passed: "
    f"{len(REQUIRED)} required files, placeholders only, no PDFs, "
    "ODQ-001--009 contiguous and decided, 37-object SHA-bound author packet "
    "and deterministic archive verified, three asymmetric method gates, "
    "zero unsafe archive members, owner parity, and local links resolved."
)
