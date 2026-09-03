#!/usr/bin/env python3
"""Create the deterministic SCI-POINT v0.1 r0.3 author packet archive."""

from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
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


rows = re.findall(
    r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$",
    MANIFEST.read_text(encoding="utf-8"),
    re.MULTILINE,
)
if len(rows) != 37 or len(rows) != len({name for name, _ in rows}):
    raise RuntimeError(f"expected 37 unique manifest rows, found {len(rows)}")

for name, expected in rows:
    path = ROOT / name
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError(f"author-object hash mismatch: {name}")

expected_manifest_sidecar = f"{digest(MANIFEST)}  {MANIFEST.name}"
if MANIFEST_SIDECAR.read_text(encoding="utf-8").strip() != expected_manifest_sidecar:
    raise RuntimeError("author-packet manifest sidecar mismatch")

payload = [(ROOT / name, name) for name, _ in rows]
payload.extend([(MANIFEST, MANIFEST.name), (MANIFEST_SIDECAR, MANIFEST_SIDECAR.name)])

tar_bytes = BytesIO()
with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
    for source, relative in sorted(payload, key=lambda item: item[1]):
        data = source.read_bytes()
        info = tarfile.TarInfo(f"{PREFIX}/{relative}")
        info.size = len(data)
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = "root"
        info.gname = "root"
        info.mtime = 0
        tar.addfile(info, BytesIO(data))

with ARCHIVE.open("wb") as output:
    with GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        compressed.write(tar_bytes.getvalue())

archive_hash = digest(ARCHIVE)
archive_bytes = ARCHIVE.stat().st_size
ARCHIVE_HASH.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8")
ARCHIVE_BYTES.write_text(f"{archive_bytes}  {ARCHIVE.name}\n", encoding="utf-8")
print(f"author_packet_files={len(payload)}")
print(f"author_packet_bytes={archive_bytes}")
print(f"author_packet_sha256={archive_hash}")
