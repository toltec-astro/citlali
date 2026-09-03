#!/usr/bin/env python3
"""Verify the immutable SCI-POINT r0.3 input and Stage B contract artifacts."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tarfile
from urllib.parse import unquote

from pypdf import PdfReader


ROOT = pathlib.Path(__file__).resolve().parent
ARCHIVE = ROOT / "author_packet" / "SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz"
ARCHIVE_BYTES = 33_262
ARCHIVE_SHA256 = "d74de333b87c66fb74c04c1346beb5ea0956bb25b61efcf9f770bb19818ee00f"
MANIFEST_SHA256 = "c0df54ea420404cb6b100f3e478b89bd81e9d1c50b211a723230177d180df894"
PACKET_PREFIX = "SCI-POINT-v0.1-r0.3-stage-b-author-packet/"
PDFS = {
    "SCI-POINT-SCIENTIFIC-RATIONALE-v0.1.pdf": "Scientific Rationale",
    "SCI-POINT-ENGINEERING-CONFORMANCE-v0.1.pdf": "Engineering Conformance",
}
COMMON = (
    "notation.tex",
    "definitions.tex",
    "equations.tex",
    "assumptions.tex",
    "requirements.tex",
    "edge_cases.tex",
)


class CheckFailure(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def verify_packet() -> dict[str, bytes]:
    raw = ARCHIVE.read_bytes()
    require(len(raw) == ARCHIVE_BYTES, f"archive byte count: {len(raw)}")
    require(sha256(raw) == ARCHIVE_SHA256, "archive SHA-256 mismatch")

    admitted: dict[str, bytes] = {}
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        members = bundle.getmembers()
        require(len(members) == 39, f"archive member count: {len(members)}")
        for member in members:
            pure = pathlib.PurePosixPath(member.name)
            require(not pure.is_absolute(), f"absolute archive path: {member.name}")
            require(".." not in pure.parts, f"traversal archive path: {member.name}")
            require(member.isfile(), f"non-regular archive member: {member.name}")
            require(not (member.issym() or member.islnk()), f"link archive member: {member.name}")
        manifest_member = bundle.getmember(PACKET_PREFIX + "AUTHOR_PACKET_MANIFEST.md")
        manifest_bytes = bundle.extractfile(manifest_member).read()
        require(sha256(manifest_bytes) == MANIFEST_SHA256, "manifest SHA-256 mismatch")
        manifest = manifest_bytes.decode("utf-8")
        rows = re.findall(
            r"^\|\s*([0-9]+)\s*\|\s*`([^`]+)`\s*\|.*?\|\s*`([0-9a-f]{64})`\s*\|$",
            manifest,
            re.MULTILINE,
        )
        require(len(rows) == 37, f"admitted object count: {len(rows)}")
        require([int(row[0]) for row in rows] == list(range(1, 38)), "manifest numbering")
        for _, name, expected_hash in rows:
            member = bundle.getmember(PACKET_PREFIX + name)
            data = bundle.extractfile(member).read()
            require(sha256(data) == expected_hash, f"admitted object hash: {name}")
            admitted[name] = data

    admitted_names = set(admitted)
    unresolved: list[str] = []
    link_pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    for name, data in admitted.items():
        for target in link_pattern.findall(data.decode("utf-8")):
            local = unquote(target.split("#", 1)[0].strip())
            if not local or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", local):
                continue
            if local not in admitted_names:
                unresolved.append(f"{name} -> {target}")
    require(not unresolved, "unresolved bundle-local links: " + ", ".join(unresolved))
    return admitted


def verify_owner_parity(admitted: dict[str, bytes]) -> None:
    decisions = admitted["SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md"].decode("utf-8")
    expected_odqs = {
        "SCI-POINT-ODQ-001",
        "SCI-POINT-ODQ-002",
        "SCI-POINT-ODQ-003",
        "SCI-POINT-ODQ-003A",
        "SCI-POINT-ODQ-003B",
        "SCI-POINT-ODQ-004",
        "SCI-POINT-ODQ-005",
        "SCI-POINT-ODQ-006",
        "SCI-POINT-ODQ-007",
        "SCI-POINT-ODQ-008",
        "SCI-POINT-ODQ-009",
    }
    found_odqs = set(re.findall(r"SCI-POINT-ODQ-[0-9]{3}[AB]?", decisions))
    require(found_odqs == expected_odqs, f"owner-decision parity: {sorted(found_odqs)}")
    for method in (
        "POINT-COMPATIBILITY-METHOD v0.1",
        "POINT-FORMAL-ERROR-METHOD v0.1",
        "POINT-FULL-MAP-RMS-METHOD v0.1",
    ):
        require(method in decisions, f"owner method disposition missing: {method}")


def verify_sources() -> str:
    src = ROOT / "src"
    mains = (src / "scientific-rationale.tex", src / "engineering-conformance.tex")
    for name in COMMON:
        require((src / "common" / name).is_file(), f"missing common source: {name}")
    for main in mains:
        text = main.read_text(encoding="utf-8")
        for name in COMMON:
            require(f"\\input{{common/{name[:-4]}}}" in text, f"{main.name} does not import {name}")

    shared = "\n".join((src / "common" / name).read_text(encoding="utf-8") for name in COMMON)
    all_source = shared + "\n" + "\n".join(main.read_text(encoding="utf-8") for main in mains)
    normalized_source = all_source.replace("\\_", "_")
    reqs = set(re.findall(r"SCI-POINT-REQ-[0-9]{3}", shared))
    preds = set(re.findall(r"SCI-POINT-PRED-[0-9]{3}", shared))
    unavs = set(re.findall(r"SCI-POINT-UNAV-[0-9]{3}", shared))
    require(reqs == {f"SCI-POINT-REQ-{i:03d}" for i in range(1, 39)}, "requirement identity set")
    require(preds == {f"SCI-POINT-PRED-{i:03d}" for i in range(1, 32)}, "prediction identity set")
    require(unavs == {f"SCI-POINT-UNAV-{i:03d}" for i in range(1, 22)}, "unavailable identity set")

    required_tokens = (
        "fitted_amplitude_over_full_map_rms",
        "sig2noise",
        "diagnostic_display_only",
        "request",
        "applicability",
        "eligibility",
        "realization",
        "POINT-FIXED-BRANCH-RESPONSE-STATE",
        "POINT-FULL-PROCEDURE-RESPONSE-STATE",
        "POINT-OBSERVATIONAL-BIAS-ACCURACY-STATE",
        "unavailable_pending_separate_owner_approval",
    )
    for token in required_tokens:
        require(token in normalized_source, f"required source token: {token}")
    require("peak\\_over\\_full\\_map\\_rms} is not an admitted alias" in all_source,
            "peak-over-RMS non-alias guard")

    source_manifest = json.loads((ROOT / "STAGE_B_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(source_manifest.get("schema_version") == 1, "source manifest schema")
    require(source_manifest.get("author_packet_sha256") == ARCHIVE_SHA256,
            "source manifest author-packet binding")
    listed_paths = set()
    for entry in source_manifest.get("files", []):
        relative = entry["path"]
        listed_paths.add(relative)
        require(sha256((ROOT / relative).read_bytes()) == entry["sha256"],
                f"source manifest hash: {relative}")
    expected_paths = {
        "src/scientific-rationale.tex",
        "src/engineering-conformance.tex",
        *(f"src/common/{name}" for name in COMMON),
        "verify_stage_b.py",
    }
    require(listed_paths == expected_paths, "source manifest path set")
    return all_source


def verify_pdfs() -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    for filename, title_fragment in PDFS.items():
        path = ROOT / "pdf" / filename
        require(path.is_file(), f"missing PDF: {filename}")
        reader = PdfReader(path)
        require(len(reader.pages) >= 10, f"unexpectedly short PDF: {filename}")
        title = str((reader.metadata or {}).get("/Title", ""))
        require(title_fragment in title, f"PDF title metadata: {filename}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        compact_text = re.sub(r"\s+", "", text)
        for fragment in (
            "SCI-POINT",
            "POINT-COMPATIBILITY-METHOD v0.1",
            "POINT-FORMAL-ERROR-METHOD v0.1",
            "POINT-FULL-MAP-RMS-METHOD v0.1",
            "diagnostic_display_only",
            "SCI-POINT-PRED-031",
        ):
            require(re.sub(r"\s+", "", fragment) in compact_text,
                    f"PDF text fragment {fragment!r}: {filename}")
        results.append((filename, len(reader.pages), sha256(path.read_bytes())))
    build_manifest = json.loads((ROOT / "STAGE_B_BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    require(build_manifest.get("schema_version") == 1, "build manifest schema")
    require(build_manifest.get("visual_qa", {}).get("pages_inspected") == 29,
            "build manifest visual-QA page count")
    require(build_manifest.get("visual_qa", {}).get("result") == "pass",
            "build manifest visual-QA result")
    by_name = {entry["path"].split("/")[-1]: entry for entry in build_manifest.get("pdfs", [])}
    require(set(by_name) == set(PDFS), "build manifest PDF set")
    for filename, pages, digest in results:
        require(by_name[filename]["pages"] == pages, f"build manifest pages: {filename}")
        require(by_name[filename]["sha256"] == digest, f"build manifest hash: {filename}")
    return results


def main() -> int:
    try:
        admitted = verify_packet()
        verify_owner_parity(admitted)
        verify_sources()
        pdfs = verify_pdfs()
    except (CheckFailure, KeyError, tarfile.TarError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS archive_bytes=33262 archive_sha256=" + ARCHIVE_SHA256)
    print("PASS manifest_sha256=" + MANIFEST_SHA256)
    print("PASS members=39 admitted_objects=37 unsafe_members=0 unresolved_bundle_local_links=0")
    print("PASS owner_decision_parity=11_ids method_dispositions=3")
    print("PASS shared_sources=6 requirements=38 predictions=31 unavailable_states=21")
    for filename, pages, digest in pdfs:
        print(f"PASS pdf={filename} pages={pages} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
