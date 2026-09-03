#!/usr/bin/env python3
"""Verify SCI-POINT v0.1 r0.3 content, parity, links, and archive."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
import tarfile


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = Path(__file__).resolve().parent
MANIFEST = ROOT / "AUTHOR_PACKET_MANIFEST.md"
MANIFEST_SIDECAR = ROOT / "AUTHOR_PACKET_MANIFEST.sha256"
ARCHIVE = PACKET_DIR / "SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz"
ARCHIVE_HASH = Path(f"{ARCHIVE}.sha256")
ARCHIVE_BYTES = Path(f"{ARCHIVE}.bytes")
PREFIX = "SCI-POINT-v0.1-r0.3-stage-b-author-packet"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


manifest = MANIFEST.read_text(encoding="utf-8")
if "SCI-POINT_AUTHOR_PACKET_MANIFEST v0.1/r0.3" not in manifest:
    raise SystemExit("wrong author-packet manifest identity")
if "HASH_PENDING" in manifest or "Stage B not launched" not in manifest:
    raise SystemExit("manifest pending hash or Stage B gate failure")

rows = re.findall(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$",
    manifest,
    re.MULTILINE,
)
if len(rows) != 37:
    raise SystemExit(f"expected 37 author objects, found {len(rows)}")

objects: list[str] = []
for expected_index, (index, name, expected_hash) in enumerate(rows, start=1):
    if int(index) != expected_index or name in objects:
        raise SystemExit(f"manifest index or uniqueness failure: {index} {name}")
    objects.append(name)
    path = ROOT / name
    if not path.is_file() or digest(path) != expected_hash:
        raise SystemExit(f"author-object hash mismatch: {name}")
    text = path.read_text(encoding="utf-8")
    for forbidden in (
        "include/citlali/",
        "src/citlali/",
        "validation/",
        "tests/",
        "INTERNAL_DOSSIER.md",
        "PRIOR_WORK.md",
        "SCIENTIFIC_OWNER_ODQ_",
    ):
        if forbidden in text:
            raise SystemExit(f"author object {name} leaks prohibited token {forbidden}")

owner = (ROOT / "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md").read_text(
    encoding="utf-8"
)
for method in (
    "POINT-COMPATIBILITY-METHOD v0.1",
    "POINT-FORMAL-ERROR-METHOD v0.1",
    "POINT-FULL-MAP-RMS-METHOD v0.1",
):
    if method not in owner or method not in manifest:
        raise SystemExit(f"owner-decision/manifest parity failure: {method}")
if owner.count("unavailable_pending_separate_owner_approval") < 3:
    raise SystemExit("owner method table does not preserve three unavailable gates")

lifecycle = (ROOT / "AUTHOR_LIFECYCLE_AND_NAMED_USE_DISPOSITIONS.md").read_text(
    encoding="utf-8"
)
if "`eligible`, `ineligible`, or `decision_unavailable`" not in (
    ROOT / "AUTHOR_POLICY_PROFILE_RECORDS.md"
).read_text(encoding="utf-8"):
    raise SystemExit("SCI-VAL eligibility domain is not exactly three-valued")
if "diagnostic_display_only" not in lifecycle:
    raise SystemExit("diagnostic consumer action is missing")
if re.search(r"eligibility[^\n]*diagnostic_only", lifecycle):
    raise SystemExit("diagnostic_only leaked into eligibility")

for name in (
    "AUTHOR_MAP_TO_POINT_BOUNDARY.md",
    "AUTHOR_JINC_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_FIXED_TO_POINT_BOUNDARY.md",
    "AUTHOR_FLT_MATCHED_TO_POINT_BOUNDARY.md",
):
    text = (ROOT / name).read_text(encoding="utf-8")
    for phrase in (
        "Status: draft boundary requirements",
        "source\ndigest not bound",
        "owner approval pending",
        "numerical route unavailable",
    ):
        if phrase not in text:
            raise SystemExit(f"boundary status failure in {name}: {phrase}")

link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
for name in objects:
    path = ROOT / name
    for target in link_pattern.findall(path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        link_target = target.split("#", 1)[0]
        if link_target and not (path.parent / link_target).resolve().exists():
            raise SystemExit(f"unresolved bundle-local link in {name}: {target}")

expected_manifest_sidecar = f"{digest(MANIFEST)}  {MANIFEST.name}"
if MANIFEST_SIDECAR.read_text(encoding="utf-8").strip() != expected_manifest_sidecar:
    raise SystemExit("manifest digest sidecar mismatch")

expected_archive_hash = f"{digest(ARCHIVE)}  {ARCHIVE.name}"
if ARCHIVE_HASH.read_text(encoding="utf-8").strip() != expected_archive_hash:
    raise SystemExit("archive digest sidecar mismatch")
expected_archive_bytes = f"{ARCHIVE.stat().st_size}  {ARCHIVE.name}"
if ARCHIVE_BYTES.read_text(encoding="utf-8").strip() != expected_archive_bytes:
    raise SystemExit("archive byte-count sidecar mismatch")

expected_members = {
    f"{PREFIX}/{name}" for name in objects + [MANIFEST.name, MANIFEST_SIDECAR.name]
}
with tarfile.open(ARCHIVE, mode="r:gz") as archive:
    members = archive.getmembers()
    if {member.name for member in members} != expected_members:
        raise SystemExit("archive membership differs from exclusive manifest payload")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if not member.isfile() or member.issym() or member.islnk():
            raise SystemExit(f"unsafe/non-regular archive member: {member.name}")
        relative = path.relative_to(PREFIX).as_posix()
        extracted = archive.extractfile(member)
        if extracted is None or extracted.read() != (ROOT / relative).read_bytes():
            raise SystemExit(f"archive byte mismatch: {relative}")

print("sci_point_author_packet=PASS")
print(f"author_packet_objects={len(objects)}")
print(f"author_packet_archive_files={len(expected_members)}")
print(f"author_packet_manifest_sha256={digest(MANIFEST)}")
print(f"author_packet_archive_bytes={ARCHIVE.stat().st_size}")
print(f"author_packet_archive_sha256={digest(ARCHIVE)}")
print("unsafe_archive_members=0")
print("unresolved_bundle_local_links=0")
print("owner_decision_manifest_parity=PASS")
print("compatibility_method_available=false")
print("formal_error_method_available=false")
print("full_map_rms_method_available=false")
print("stage_b_authorized=false")
