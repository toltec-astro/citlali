#!/usr/bin/env python3
"""Verify SCI-POINT v0.1 r0.4 view separation and delivery parity."""

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
SCIENTIFIC_DIRECTIVE = ROOT / "SCIENTIFIC_OWNER_TARGETED_STAGE_B_R0_3_DIRECTIVE_2026-09-03.md"
SCIENTIFIC_DIRECTIVE_SHA256 = "9f445480fd42311ebe21f4da772e4e32db400c2da08a119b699f6bd8e13a54d4"
PRESENTATION_DIRECTIVE = ROOT / "SCIENTIFIC_OWNER_R0_4_VIEW_SEPARATION_DIRECTIVE_2026-09-03.md"
PRESENTATION_DIRECTIVE_SHA256 = "3f2331d4dc2a926ebd840cddb9bd85c9bc7d2a88be28718074313e262c259193"
COMMON_CORE_SHA256 = "c0ca71bd457b8e6d37a425eb3ead76400dba3a5e29c869420807928201cdcdbd"
COMMON_HASHES = {
    "notation.tex": "e4579b755cb5fbdc71a061f5718e4b9ab29e040e472d7774d2457383a29695d8",
    "definitions.tex": "74846e0fa6b19613a7cf66aaa8867068a5ed6b3260a772f836e8a6913da7beaf",
    "equations.tex": "dc45043f1ba0a03a5aaf2b1afd602fd57202392fcd625beb524f12f51d43d752",
    "assumptions.tex": "07cd085ceabb6d423fd85be2d3a76f213cfbdfe3230b090e34ae8cf1c6918ad7",
    "requirements.tex": "7d8076a8ae92556a38fbb2cdf0ff023dad25901d5ef4b5b4a8c3cd5343db443d",
    "edge_cases.tex": "42783d6652553ddfba62aa9893544a7e45c2ed2746955992989e1465788e2835",
}
COMMON = tuple(COMMON_HASHES)
DELIVERY = ROOT / "delivery" / "SCI-POINT-v0.1-r0.4-stage-b-delivery-packet.tar.gz"
DELIVERY_PREFIX = "SCI-POINT-v0.1-r0.4-stage-b-delivery/"
PDFS = {
    "SCI-POINT-NORMATIVE-CORE-v0.1.pdf": {
        "title": "SCI-POINT Normative Core v0.1 r0.4",
        "pages": 18,
        "source": "normative-core.tex",
        "source_macro": "NormativeSourceSHA",
        "required": ("SCI-POINT-REQ-038", "SCI-POINT-PRED-032", "SCI-POINT-UNAV-023"),
    },
    "SCI-POINT-SCIENTIFIC-RATIONALE-v0.1.pdf": {
        "title": "SCI-POINT Scientific Rationale v0.1 r0.4",
        "pages": 8,
        "source": "scientific-rationale.tex",
        "source_macro": "RationaleSourceSHA",
        "required": ("SCI-POINT-REQ-001--038", "SCI-POINT-PRED-001--032"),
    },
    "SCI-POINT-ENGINEERING-CONFORMANCE-v0.1.pdf": {
        "title": "SCI-POINT Engineering Conformance v0.1 r0.4",
        "pages": 9,
        "source": "engineering-conformance.tex",
        "source_macro": "EngineeringSourceSHA",
        "required": ("REQ-038", "PRED-032", "Eight stage-specific logical records"),
    },
}


class CheckFailure(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def compact(text: str) -> str:
    normalized = text.replace("–", "--").replace("—", "--").replace("ﬁ", "fi")
    return re.sub(r"\s+", "", normalized)


def macro(text: str, name: str) -> str:
    match = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]+)}}", text)
    require(match is not None, f"binding macro missing: {name}")
    return match.group(1)


def verify_safe_tar(path: pathlib.Path, expected_prefix: str) -> list[tarfile.TarInfo]:
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            pure = pathlib.PurePosixPath(member.name)
            require(member.name.startswith(expected_prefix), f"archive prefix: {member.name}")
            require(not pure.is_absolute(), f"absolute archive path: {member.name}")
            require(".." not in pure.parts, f"traversal archive path: {member.name}")
            require(member.isfile(), f"non-regular archive member: {member.name}")
            require(not (member.issym() or member.islnk()), f"link archive member: {member.name}")
        return members


def verify_packet() -> dict[str, bytes]:
    raw = ARCHIVE.read_bytes()
    require(len(raw) == ARCHIVE_BYTES, f"archive byte count: {len(raw)}")
    require(sha256(raw) == ARCHIVE_SHA256, "archive SHA-256 mismatch")
    members = verify_safe_tar(ARCHIVE, PACKET_PREFIX)
    require(len(members) == 39, f"archive member count: {len(members)}")

    admitted: dict[str, bytes] = {}
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        manifest_bytes = bundle.extractfile(
            bundle.getmember(PACKET_PREFIX + "AUTHOR_PACKET_MANIFEST.md")
        ).read()
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
            data = bundle.extractfile(bundle.getmember(PACKET_PREFIX + name)).read()
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
        "SCI-POINT-ODQ-001", "SCI-POINT-ODQ-002", "SCI-POINT-ODQ-003",
        "SCI-POINT-ODQ-003A", "SCI-POINT-ODQ-003B", "SCI-POINT-ODQ-004",
        "SCI-POINT-ODQ-005", "SCI-POINT-ODQ-006", "SCI-POINT-ODQ-007",
        "SCI-POINT-ODQ-008", "SCI-POINT-ODQ-009",
    }
    require(set(re.findall(r"SCI-POINT-ODQ-[0-9]{3}[AB]?", decisions)) == expected_odqs,
            "owner-decision parity")
    require(sha256(SCIENTIFIC_DIRECTIVE.read_bytes()) == SCIENTIFIC_DIRECTIVE_SHA256,
            "scientific-directive SHA-256 mismatch")
    require(sha256(PRESENTATION_DIRECTIVE.read_bytes()) == PRESENTATION_DIRECTIVE_SHA256,
            "presentation-directive SHA-256 mismatch")


def verify_sources() -> tuple[str, dict[str, str]]:
    src = ROOT / "src"
    common_paths = tuple(src / "common" / name for name in COMMON)
    for path in common_paths:
        require(path.is_file(), f"missing common source: {path.name}")
        require(sha256(path.read_bytes()) == COMMON_HASHES[path.name],
                f"accepted r0.3 bytes changed: {path.name}")
    common_digest = sha256(b"".join(path.read_bytes() for path in common_paths))
    require(common_digest == COMMON_CORE_SHA256, "common-core digest")

    bindings = (src / "common" / "bindings.tex").read_text(encoding="utf-8")
    require(macro(bindings, "NormativeCoreID") == "SCI-POINT-NORMATIVE-CORE/v0.1-r0.4",
            "normative-core identity")
    require(macro(bindings, "NormativeCoreSHA") == common_digest,
            "normative-core digest binding")
    require(macro(bindings, "AcceptedCommonCoreSHA") == common_digest,
            "accepted-common-core digest binding")
    require(macro(bindings, "StageAPacketSHA") == ARCHIVE_SHA256, "Stage A binding")
    require(macro(bindings, "ScientificDirectiveSHA") == SCIENTIFIC_DIRECTIVE_SHA256,
            "scientific-directive binding")
    require(macro(bindings, "PresentationDirectiveSHA") == PRESENTATION_DIRECTIVE_SHA256,
            "presentation-directive binding")

    mains = {
        "normative-core.tex": (src / "normative-core.tex", "NormativeSourceSHA"),
        "scientific-rationale.tex": (src / "scientific-rationale.tex", "RationaleSourceSHA"),
        "engineering-conformance.tex": (src / "engineering-conformance.tex", "EngineeringSourceSHA"),
    }
    source_hashes: dict[str, str] = {}
    for name, (path, digest_macro) in mains.items():
        text = path.read_text(encoding="utf-8")
        digest = sha256(path.read_bytes())
        source_hashes[name] = digest
        require("\\input{common/bindings}" in text, f"{name} lacks bindings")
        require(macro(bindings, digest_macro) == digest, f"{name} source binding")
        for cover_macro in (
            "StageAPacketSHA", "ScientificDirectiveSHA", "OwnerDirectiveSHA",
            "NormativeCoreSHA", digest_macro, "BuildRecordID",
        ):
            require(f"\\{cover_macro}" in text, f"{name} lacks cover binding {cover_macro}")

    core_text = mains["normative-core.tex"][0].read_text(encoding="utf-8")
    rationale_text = mains["scientific-rationale.tex"][0].read_text(encoding="utf-8")
    ecs_text = mains["engineering-conformance.tex"][0].read_text(encoding="utf-8")
    for name in COMMON:
        command = f"\\input{{common/{name[:-4]}}}"
        require(core_text.count(command) == 1, f"core import count: {name}")
        require(command not in rationale_text, f"rationale duplicates common source: {name}")
        require(command not in ecs_text, f"ECS duplicates common source: {name}")
    require("\\setcounter{tocdepth}{1}" in rationale_text, "rationale top-level contents")
    require("\\setcounter{tocdepth}{1}" in ecs_text, "ECS top-level contents")

    common_text = "\n".join(path.read_text(encoding="utf-8") for path in common_paths)
    normalized = common_text.replace("\\_", "_")
    require(set(re.findall(r"SCI-POINT-REQ-[0-9]{3}", common_text)) ==
            {f"SCI-POINT-REQ-{i:03d}" for i in range(1, 39)}, "requirement IDs")
    require(set(re.findall(r"SCI-POINT-PRED-[0-9]{3}", common_text)) ==
            {f"SCI-POINT-PRED-{i:03d}" for i in range(1, 33)}, "prediction IDs")
    require(set(re.findall(r"SCI-POINT-UNAV-[0-9]{3}", common_text)) ==
            {f"SCI-POINT-UNAV-{i:03d}" for i in range(1, 24)}, "unavailable IDs")
    require(len(re.findall(r"(?m)^REQ-[0-9]{3}\s*&", ecs_text)) == 38,
            "ECS requirement evidence index")
    require(len(re.findall(r"(?m)^PRED-[0-9]{3}\s*&", ecs_text)) == 32,
            "ECS prediction fixture index")
    require(len(re.findall(r"(?m)^[A-H]\s*&", ecs_text)) == 8, "ECS logical stage rows")
    require("SCI-POINT-REQ-001--038" in rationale_text and
            "SCI-POINT-PRED-001--032" in rationale_text,
            "rationale compact normative index")
    require(len(set(re.findall(r"SCI-POINT-REQ-[0-9]{3}", rationale_text))) <= 1,
            "rationale repeats requirement register")
    require(len(set(re.findall(r"SCI-POINT-PRED-[0-9]{3}", rationale_text))) <= 1,
            "rationale repeats prediction register")

    required_tokens = (
        "applicability_unknown", "decision_unavailable", "incomplete", "not_produced",
        "applied", "fit_realized", "complete_publication_candidate",
        "POINT-FIT-AMPLITUDE-COMPONENT",
        "POINT-SOURCE-ASSOCIATED-AMPLITUDE-DIAGNOSTIC",
        "POINT-DYNAMIC-RANGE-DIAGNOSTIC/FITTED-AMPLITUDE-OVER-FULL-MAP-RMS@1",
        "POINT-FORMAL-AMPLITUDE-STANDARDIZATION@1", "diagnostic_display_only",
        "approximately_centered", "C_{\\Delta}", "R^{\\mathrm{fixed}}",
        "R^{\\mathrm{POINT\\mbox{-}FP}\\mid\\mathrm{parent\\mbox{-}fixed}}",
        "R^{\\mathrm{chain\\mbox{-}FP}}",
        "POINT-FORMAL-AMPLITUDE-MAGNITUDE-STANDARDIZATION@1", "no proposition",
        "inherited parent observation/exposure lineage",
    )
    for token in required_tokens:
        require(token in normalized, f"required common token: {token}")
    for stale in ("not_realized", "approximately-centered", "POINT-AMPLITUDE-DIAGNOSTIC"):
        require(stale not in normalized, f"stale common token: {stale}")

    records = load_json("STAGE_B_R0_3_RECORDS.json")
    require(records["document_revision"] == "r0.3", "scientific-record baseline")
    require(records["sci_val"]["applicability"][-1] == "applicability_unknown",
            "SCI-VAL applicability")
    require(records["sci_val"]["realization"] ==
            ["realized", "incomplete", "failed", "not_produced"], "SCI-VAL realization")
    require(records["sci_val"]["dispositions"]["not_requested"] ==
            ["not_requested", None, None, "not_produced"], "absence of proposition")
    require(len(records["producer_lifecycle"]) == 13, "producer lifecycle")
    require(len(records["logical_records"]) == 8, "logical record count")
    require(len(records["point_boundary_roles"]) == 4, "parent family count")
    require(len(records["required_method_record_templates"]) == 3, "method authority count")
    require(records["response_families"]["families_alias"] is False,
            "response-family nonaliasing")
    require(records["derived_diagnostics"][0]["signed"] is True and
            records["derived_diagnostics"][1]["signed"] is True, "signed diagnostics")
    require(records["publication_semantics"]["downstream_may_mutate_fit"] is False,
            "downstream decision separation")
    require(records["published_product_lineage"]["point_creates_exposure_identity"] is False,
            "published lineage")

    parity = load_json("STAGE_B_R0_4_PARITY_REPORT.json")
    require(parity["result"] == "pass" and parity["semantic_change"] is False,
            "r0.4 parity report")
    require(parity["common_core"]["byte_equal"] is True, "r0.4 common byte parity")

    source_manifest = load_json("STAGE_B_SOURCE_MANIFEST.json")
    require(source_manifest["document_revision"] == "r0.4", "source manifest revision")
    require(source_manifest["author_packet_sha256"] == ARCHIVE_SHA256,
            "source manifest Stage A binding")
    require(source_manifest["presentation_directive"]["sha256"] ==
            PRESENTATION_DIRECTIVE_SHA256, "source manifest presentation directive")
    require(source_manifest["normative_core"]["sha256"] == common_digest,
            "source manifest core binding")
    require(source_manifest["normative_core"]["scientific_bytes_changed"] is False,
            "source manifest scientific nonchange")
    for entry in source_manifest["files"]:
        require(sha256((ROOT / entry["path"]).read_bytes()) == entry["sha256"],
                f"source manifest hash: {entry['path']}")
    return common_digest, source_hashes


def verify_pdfs(common_digest: str, source_hashes: dict[str, str]) -> list[tuple[str, int, str]]:
    build = load_json("STAGE_B_BUILD_MANIFEST.json")
    require(build["document_revision"] == "r0.4", "build manifest revision")
    by_name = {pathlib.Path(entry["path"]).name: entry for entry in build["pdfs"]}
    require(set(by_name) == set(PDFS), "build manifest PDF set")
    bindings = (ROOT / "src/common/bindings.tex").read_text(encoding="utf-8")
    results: list[tuple[str, int, str]] = []
    for filename, expected in PDFS.items():
        path = ROOT / "pdf" / filename
        reader = PdfReader(path)
        require(len(reader.pages) == expected["pages"] == by_name[filename]["pages"],
                f"PDF pages: {filename}")
        require(not (reader.get_fields() or {}), f"unexpected PDF form fields: {filename}")
        metadata = reader.metadata or {}
        require(str(metadata.get("/Title", "")) == expected["title"], f"PDF title: {filename}")
        require(str(metadata.get("/Author", "")) ==
                "SCI-POINT scientific-contract program; Grant Wilson, scientific owner",
                f"PDF author metadata: {filename}")
        for page in reader.pages:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            require(abs(width - 612) < 0.1 and abs(height - 792) < 0.1,
                    f"PDF page size: {filename}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        flat = compact(text)
        source_hash = source_hashes[expected["source"]]
        for fragment in (
            "SCI-POINT", "r0.4", ARCHIVE_SHA256, SCIENTIFIC_DIRECTIVE_SHA256,
            PRESENTATION_DIRECTIVE_SHA256, common_digest, source_hash, *expected["required"],
        ):
            require(compact(fragment) in flat, f"PDF fragment {fragment!r}: {filename}")
        digest = sha256(path.read_bytes())
        require(digest == by_name[filename]["sha256"], f"PDF digest: {filename}")
        require(len(path.read_bytes()) == by_name[filename]["bytes"], f"PDF bytes: {filename}")
        require(macro(bindings, expected["source_macro"]) == source_hash,
                f"PDF/source binding macro: {filename}")
        results.append((filename, len(reader.pages), digest))

    total_pages = sum(row[1] for row in results)
    require(total_pages == 35, "publication page count")
    require(build["visual_qa"]["pages_inspected"] == total_pages and
            build["visual_qa"]["result"] == "pass", "build visual-QA record")
    require(build["source_manifest_sha256"] ==
            sha256((ROOT / "STAGE_B_SOURCE_MANIFEST.json").read_bytes()),
            "build/source manifest binding")
    qa = load_json("STAGE_B_R0_4_PDF_QA_REPORT.json")
    require(qa["result"] == "pass" and qa["pages_inspected"] == total_pages,
            "PDF QA report")
    build_parity = load_json("STAGE_B_R0_4_BUILD_PARITY_REPORT.json")
    require(build_parity["result"] == "pass", "build parity report")
    for row in build_parity["results"]:
        pdf_path = ROOT / row["pdf"]
        source_path = ROOT / row["source"]
        reader = PdfReader(pdf_path)
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        require(row["source_sha256"] == sha256(source_path.read_bytes()),
                f"build parity source: {row['source']}")
        require(row["pdf_sha256"] == sha256(pdf_path.read_bytes()),
                f"build parity PDF: {row['pdf']}")
        require(row["extracted_text_sha256"] == sha256(extracted.encode()),
                f"build parity text: {row['pdf']}")
    return results


def verify_delivery() -> tuple[int, int, str]:
    from build_stage_b_delivery import FILES

    raw = DELIVERY.read_bytes()
    require(DELIVERY.with_suffix(DELIVERY.suffix + ".sha256").read_text(
        encoding="utf-8").strip() == sha256(raw), "delivery hash sidecar")
    require(int(DELIVERY.with_suffix(DELIVERY.suffix + ".bytes").read_text(
        encoding="utf-8").strip()) == len(raw), "delivery byte sidecar")
    members = verify_safe_tar(DELIVERY, DELIVERY_PREFIX)
    relative_names = {
        str(pathlib.PurePosixPath(member.name).relative_to(
            pathlib.PurePosixPath(DELIVERY_PREFIX))) for member in members
    }
    require(relative_names == set(FILES), "delivery member set")
    with tarfile.open(DELIVERY, "r:gz") as bundle:
        for member in members:
            relative = pathlib.PurePosixPath(member.name).relative_to(
                pathlib.PurePosixPath(DELIVERY_PREFIX))
            local = ROOT.joinpath(*relative.parts)
            require(local.is_file(), f"delivery source exists: {relative}")
            require(bundle.extractfile(member).read() == local.read_bytes(),
                    f"delivery byte parity: {relative}")
    return len(members), len(raw), sha256(raw)


def main() -> int:
    try:
        admitted = verify_packet()
        verify_owner_parity(admitted)
        common_digest, source_hashes = verify_sources()
        pdfs = verify_pdfs(common_digest, source_hashes)
        delivery_members, delivery_bytes, delivery_digest = verify_delivery()
    except (CheckFailure, KeyError, ValueError, tarfile.TarError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS author_packet bytes={ARCHIVE_BYTES} sha256={ARCHIVE_SHA256}")
    print(f"PASS author_manifest sha256={MANIFEST_SHA256}")
    print("PASS author_members=39 admitted=37 unsafe=0 unresolved_links=0")
    print("PASS owner_parity=11_ids scientific_and_presentation_directives=2")
    print(f"PASS common_core files=6 byte_equal_r0.3=true sha256={common_digest}")
    print("PASS requirements=38 predictions=32 unavailable_states=23")
    print("PASS view_separation=core_complete,rationale_concise,engineering_indexed")
    print("PASS lifecycle=13 logical_records=8 parent_families=4 method_authorities=3")
    print("PASS semantics=equations,roles,SCI_VAL,responses,diagnostics,weights,lineage")
    for filename, pages, digest in pdfs:
        print(f"PASS pdf={filename} pages={pages} sha256={digest}")
    print("PASS visual_qa pages=35 defects=0")
    print(f"PASS delivery members={delivery_members} bytes={delivery_bytes} sha256={delivery_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
