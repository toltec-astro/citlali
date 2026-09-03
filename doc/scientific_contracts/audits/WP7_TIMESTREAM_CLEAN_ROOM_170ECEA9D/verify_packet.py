#!/usr/bin/env python3
"""Verify the WP-7 sanitized clean-room successor packet."""

from __future__ import annotations

import hashlib
import fnmatch
from pathlib import Path
import re
import subprocess


PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[3]
SOURCE_COMMIT = "170ecea9de1ee810da7d7e45a489a4545ccd623d"

BOUND_SOURCE_OBJECTS = {
    "doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md":
        "be51b2347f04237ed5ae5773efb6978405f76666b3a92647721a482d25f7f9e0",
    "doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md":
        "26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac",
    "doc/scientific_contracts/packages/SCI-AST/v0.1/README.md":
        "f722589fb39df1d75c12c6f5a99797ee9bd1f304088edada8cf4788311b8b257",
    "doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md":
        "b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601",
    "doc/scientific_contracts/packages/SCI-RTC/v0.1/SOURCE_MANIFEST_CORRECTED_2026-08-25.md":
        "a5c06bd46cd8514e67ea77a7a728e3decb8c415cf486c4ec121927212bf22994",
    "doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md":
        "413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22",
    "doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md":
        "8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66",
    "doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md":
        "2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6",
    "doc/scientific_contracts/boundaries/v0.1/SOURCE_MANIFEST.md":
        "ce813e0adab8270daf713b30db8a271185227048fb79a71abe4b9e4a6ae2ab4a",
    "doc/scientific_contracts/producer_interfaces/v0.1/SOURCE_MANIFEST.md":
        "a417fb3d22aa46ad7d7f1134b6d804b9d3c3f5a7f601dbb53c19f10a23e72912",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_SOURCE_MANIFEST.md":
        "d407228bfbbdbe8be994e7e84e4945fc6868365c2d045c18ac7ce1e5c40ae9aa",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_SOURCE_HYGIENE_MANIFEST.md":
        "57dacf3a5847a24a85b754e878306bd5efb088f571c354f650d0961bdd3ca9a0",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_SOURCE_MANIFEST.md":
        "365de9715c7b0fb3ef7390a07caf53a8b7c89d1bb6939f2fad36db0a816261cd",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP3_REFERENCE_BINDING_REGISTER.md":
        "96338b56aa57211d2e59664f6eeb9514d0846d51aca94bcb9cbf11d076402efe",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_RTC_NOTATION_SEMANTIC_MAPPING.md":
        "c5466abe6386715c52edbf1632972b2ec5c1f39ae684612a5b47813a1d919a09",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP4_AST_NOTATION_SEMANTIC_MAPPING.md":
        "514e05c2b4e3d7053ee23e9e1a11f269252d97c63c8be69354e330b0a1310636",
    "doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md":
        "c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c",
    "doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_REPAIR_AUTHORITY_MANIFEST_2026-08-25.md":
        "f5f9f903e52a979339c04cd741686d267c739e5530e364ab49e3bb612b37b26f",
}

PACKET_FILES = (
    "AUDITOR_START_HERE.md",
    "AUDIT_THREAD_LAUNCH_PROMPT.md",
    "CLEAN_ROOM_CHARTER.md",
    "READABLE_SOURCE_ALLOWLIST.md",
    "SANITIZED_COMPOSITION_NOTES.md",
    "verify_packet.py",
    "build_handoff.py",
)

FIREWALL_FILES = (
    "AUDITOR_START_HERE.md",
    "AUDIT_THREAD_LAUNCH_PROMPT.md",
    "CLEAN_ROOM_CHARTER.md",
    "READABLE_SOURCE_ALLOWLIST.md",
    "SANITIZED_COMPOSITION_NOTES.md",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_object(path: str) -> bytes:
    for root in ("sources", "bindings"):
        bundled = PACKET / root / path
        if bundled.is_file():
            return bundled.read_bytes()
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def verify_bound_sources() -> None:
    for path, expected in BOUND_SOURCE_OBJECTS.items():
        actual = sha256(git_object(path))
        if actual != expected:
            raise RuntimeError(
                f"source-object hash mismatch for {path}: {actual} != {expected}"
            )


def verify_firewall() -> None:
    corpus = "\n".join((PACKET / name).read_text() for name in FIREWALL_FILES)
    forbidden = (
        r"\bF-\d{3}\b",
        r"\bXOD-\d{3}\b",
        r"\bTS-CLAR-\d{3}\b",
        r"\bWP\d*-OWNER-D\d+\b",
        r"SCENARIO_TRACE_REPORT",
        r"TIMESTREAM_FINDING_SCOPE_AND_CLOSURE_REGISTER",
    )
    for pattern in forbidden:
        if re.search(pattern, corpus):
            raise RuntimeError(f"clean-room packet leaks forbidden token: {pattern}")

    allowlist = (PACKET / "READABLE_SOURCE_ALLOWLIST.md").read_text()
    if "packages/SCI-MAP" in allowlist or "SIX_PACKAGE_WIDE_SCALE" in allowlist:
        raise RuntimeError("readable allowlist admits a prohibited source")


def verify_readable_sources() -> int:
    allowlist = (PACKET / "READABLE_SOURCE_ALLOWLIST.md").read_text()
    entries = sorted(set(re.findall(r"`(doc/[^`]+)`", allowlist)))
    resolved: set[str] = set()
    bundled_root = PACKET / "sources"
    for entry in entries:
        if "*" in entry:
            if bundled_root.is_dir():
                candidates = {
                    path.relative_to(bundled_root).as_posix()
                    for path in bundled_root.rglob("*") if path.is_file()
                }
            else:
                prefix = entry.split("*", 1)[0]
                result = subprocess.run(
                    ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT, prefix],
                    cwd=REPO,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr)
                candidates = set(result.stdout.splitlines())
            matches = {path for path in candidates if fnmatch.fnmatch(path, entry)}
            if not matches:
                raise RuntimeError(f"allowlist pattern resolves no source: {entry}")
            resolved.update(matches)
        else:
            git_object(entry)
            resolved.add(entry)
    if any("/history/" in path or "packages/SCI-MAP" in path for path in resolved):
        raise RuntimeError("resolved allowlist contains a prohibited source")
    return len(resolved)


def verify_bundle_checksums() -> None:
    checksum_path = PACKET / "SOURCE_OBJECT_SHA256SUMS.txt"
    source_root = PACKET / "sources"
    if not source_root.is_dir():
        return
    if not checksum_path.is_file():
        raise RuntimeError("bundle source checksum file is missing")
    expected_paths: set[str] = set()
    for line in checksum_path.read_text().splitlines():
        expected, path = line.split("  ", 1)
        actual = sha256((source_root / path).read_bytes())
        if actual != expected:
            raise RuntimeError(f"bundled source hash mismatch for {path}")
        expected_paths.add(path)
    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("bundle source inventory differs from its checksum list")


def verify_binding_checksums() -> None:
    binding_root = PACKET / "bindings"
    if not binding_root.is_dir():
        return
    checksum_path = PACKET / "INTEGRITY_OBJECT_SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise RuntimeError("bundle integrity checksum file is missing")
    expected_paths: set[str] = set()
    for line in checksum_path.read_text().splitlines():
        expected, path = line.split("  ", 1)
        actual = sha256((binding_root / path).read_bytes())
        if actual != expected:
            raise RuntimeError(f"bundled integrity hash mismatch for {path}")
        expected_paths.add(path)
    if expected_paths != set(BOUND_SOURCE_OBJECTS):
        raise RuntimeError("bundle integrity inventory differs from bound objects")
    actual_paths = {
        path.relative_to(binding_root).as_posix()
        for path in binding_root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("bundle integrity tree differs from its checksum list")


def verify_packet_manifest() -> None:
    manifest = PACKET / "SOURCE_MANIFEST.md"
    if not manifest.is_file():
        raise RuntimeError("SOURCE_MANIFEST.md is missing")
    rows = re.findall(
        r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
        manifest.read_text(),
        re.MULTILINE,
    )
    expected_names = set(PACKET_FILES)
    if {name for name, _ in rows} != expected_names:
        raise RuntimeError("packet manifest does not bind the exact packet files")
    for name, expected in rows:
        actual = sha256((PACKET / name).read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"packet hash mismatch for {name}: {actual} != {expected}"
            )


def main() -> int:
    verify_bound_sources()
    verify_firewall()
    readable_count = verify_readable_sources()
    verify_bundle_checksums()
    verify_binding_checksums()
    verify_packet_manifest()
    print(f"OK: source commit {SOURCE_COMMIT}")
    print(f"OK: bound source objects {len(BOUND_SOURCE_OBJECTS)}")
    print(f"OK: readable source objects {readable_count}")
    print(f"OK: sanitized packet files {len(PACKET_FILES)}")
    print("OK: clean-room firewall and SCI-MAP exclusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
