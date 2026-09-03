#!/usr/bin/env python3
"""Build the deterministic self-contained WP-7 successor-auditor archive."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[3]
SOURCE_COMMIT = "170ecea9de1ee810da7d7e45a489a4545ccd623d"
ARCHIVE = PACKET / "WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D.tar.gz"
ROOT_NAME = "WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D"

PACKET_FILES = (
    "AUDITOR_START_HERE.md",
    "AUDIT_THREAD_LAUNCH_PROMPT.md",
    "CLEAN_ROOM_CHARTER.md",
    "READABLE_SOURCE_ALLOWLIST.md",
    "SANITIZED_COMPOSITION_NOTES.md",
    "SOURCE_MANIFEST.md",
    "verify_packet.py",
    "build_handoff.py",
)

INTEGRITY_PATHS = (
    "doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md",
    "doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md",
    "doc/scientific_contracts/packages/SCI-AST/v0.1/README.md",
    "doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md",
    "doc/scientific_contracts/packages/SCI-RTC/v0.1/SOURCE_MANIFEST_CORRECTED_2026-08-25.md",
    "doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md",
    "doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md",
    "doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md",
    "doc/scientific_contracts/boundaries/v0.1/SOURCE_MANIFEST.md",
    "doc/scientific_contracts/producer_interfaces/v0.1/SOURCE_MANIFEST.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_SOURCE_MANIFEST.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_SOURCE_HYGIENE_MANIFEST.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_SOURCE_MANIFEST.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_REFERENCE_BINDING_REGISTER.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_RTC_NOTATION_SEMANTIC_MAPPING.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_AST_NOTATION_SEMANTIC_MAPPING.md",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md",
    "doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_REPAIR_AUTHORITY_MANIFEST_2026-08-25.md",
)


def git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def readable_paths() -> list[str]:
    text = (PACKET / "READABLE_SOURCE_ALLOWLIST.md").read_text()
    entries = sorted(set(re.findall(r"`(doc/[^`]+)`", text)))
    tree = git_output("ls-tree", "-r", "--name-only", SOURCE_COMMIT).decode().splitlines()
    resolved: set[str] = set()
    for entry in entries:
        if "*" in entry:
            matches = {path for path in tree if fnmatch.fnmatch(path, entry)}
            if not matches:
                raise RuntimeError(f"allowlist pattern resolves no source: {entry}")
            resolved.update(matches)
        else:
            if entry not in tree:
                raise RuntimeError(f"allowlist source is absent: {entry}")
            resolved.add(entry)
    if any("/history/" in path or "packages/SCI-MAP" in path for path in resolved):
        raise RuntimeError("resolved allowlist contains a prohibited source")
    return sorted(resolved)


def source_object(path: str) -> bytes:
    return git_output("show", f"{SOURCE_COMMIT}:{path}")


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name=f"{ROOT_NAME}/{name}")
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    paths = readable_paths()
    source_data = {path: source_object(path) for path in paths}
    integrity_data = {path: source_object(path) for path in INTEGRITY_PATHS}
    checksums = "".join(
        f"{hashlib.sha256(source_data[path]).hexdigest()}  {path}\n"
        for path in paths
    ).encode()
    integrity_checksums = "".join(
        f"{hashlib.sha256(integrity_data[path]).hexdigest()}  {path}\n"
        for path in INTEGRITY_PATHS
    ).encode()

    with tempfile.NamedTemporaryFile(
        prefix=".wp7-handoff-", suffix=".tar.gz", dir=PACKET, delete=False
    ) as stream:
        staged = Path(stream.name)

    try:
        with staged.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for name in PACKET_FILES:
                        add_bytes(tar, name, (PACKET / name).read_bytes())
                    add_bytes(tar, "SOURCE_OBJECT_SHA256SUMS.txt", checksums)
                    add_bytes(
                        tar,
                        "INTEGRITY_OBJECT_SHA256SUMS.txt",
                        integrity_checksums,
                    )
                    for path in paths:
                        add_bytes(tar, f"sources/{path}", source_data[path])
                    for path in INTEGRITY_PATHS:
                        add_bytes(tar, f"bindings/{path}", integrity_data[path])
        os.replace(staged, ARCHIVE)
    finally:
        if staged.exists():
            staged.unlink()

    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"archive={ARCHIVE}")
    print(f"sha256={digest}")
    print(f"source_objects={len(paths)}")
    print(f"integrity_objects={len(INTEGRITY_PATHS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
