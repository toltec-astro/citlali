#!/usr/bin/env python3
"""Create the deterministic SCI-FLT-MATCHED v0.1 r0.6 output bundle."""

from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(__file__).resolve().parent
MANIFEST = ROOT / "STAGE_B_DRAFT_MANIFEST.md"
BUNDLE_NOTE = BUNDLE_DIR / "BUNDLE_README_R0.6.md"
ARCHIVE = BUNDLE_DIR / "SCI-FLT-MATCHED-v0.1-r0.6-stage-b-output-bundle.tar.gz"
PREFIX = "SCI-FLT-MATCHED-v0.1-r0.6"


def manifest_paths() -> list[str]:
    rows = re.findall(
        r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*`[^`]+`\s*\|\s*\d+\s*\|",
        MANIFEST.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not rows:
        raise RuntimeError("authority manifest contains no object rows")
    if len(rows) != len(set(rows)):
        raise RuntimeError("authority manifest contains duplicate object paths")
    return rows


def add_file(archive: tarfile.TarFile, source: Path, relative: str) -> None:
    data = source.read_bytes()
    info = tarfile.TarInfo(f"{PREFIX}/{relative}")
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    archive.addfile(info, BytesIO(data))


paths = manifest_paths()
payload = [(ROOT / relative, relative) for relative in paths]
payload.extend(
    [
        (MANIFEST, "STAGE_B_DRAFT_MANIFEST.md"),
        (ROOT / "STAGE_B_DRAFT_MANIFEST.sha256", "STAGE_B_DRAFT_MANIFEST.sha256"),
        (BUNDLE_NOTE, "BUNDLE_README_R0.6.md"),
    ]
)
if len(payload) != len({relative for _, relative in payload}):
    raise RuntimeError("bundle payload contains duplicate paths")
for source, _ in payload:
    if not source.is_file():
        raise FileNotFoundError(source)

tar_bytes = BytesIO()
with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
    for source, relative in sorted(payload, key=lambda item: item[1]):
        add_file(tar, source, relative)

with ARCHIVE.open("wb") as output:
    with GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        compressed.write(tar_bytes.getvalue())

archive_hash = sha256(ARCHIVE.read_bytes()).hexdigest()
Path(f"{ARCHIVE}.sha256").write_text(
    f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8"
)
print(f"bundle_files={len(payload)}")
print(f"bundle_sha256={archive_hash}")
