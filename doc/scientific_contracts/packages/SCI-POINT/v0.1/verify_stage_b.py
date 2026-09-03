#!/usr/bin/env python3
"""Verify the immutable SCI-POINT input and final targeted Stage B r0.3 closure."""

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
OWNER_DIRECTIVE = ROOT / "SCIENTIFIC_OWNER_TARGETED_STAGE_B_R0_3_DIRECTIVE_2026-09-03.md"
OWNER_DIRECTIVE_SHA256 = "9f445480fd42311ebe21f4da772e4e32db400c2da08a119b699f6bd8e13a54d4"
DELIVERY = ROOT / "delivery" / "SCI-POINT-v0.1-r0.3-stage-b-delivery-packet.tar.gz"
DELIVERY_PREFIX = "SCI-POINT-v0.1-r0.3-stage-b-delivery/"
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
        "SCI-POINT-ODQ-001", "SCI-POINT-ODQ-002", "SCI-POINT-ODQ-003",
        "SCI-POINT-ODQ-003A", "SCI-POINT-ODQ-003B", "SCI-POINT-ODQ-004",
        "SCI-POINT-ODQ-005", "SCI-POINT-ODQ-006", "SCI-POINT-ODQ-007",
        "SCI-POINT-ODQ-008", "SCI-POINT-ODQ-009",
    }
    found_odqs = set(re.findall(r"SCI-POINT-ODQ-[0-9]{3}[AB]?", decisions))
    require(found_odqs == expected_odqs, f"owner-decision parity: {sorted(found_odqs)}")
    for method in (
        "POINT-COMPATIBILITY-METHOD v0.1",
        "POINT-FORMAL-ERROR-METHOD v0.1",
        "POINT-FULL-MAP-RMS-METHOD v0.1",
    ):
        require(method in decisions, f"owner method disposition missing: {method}")


def macro(text: str, name: str) -> str:
    match = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]+)}}", text)
    require(match is not None, f"binding macro missing: {name}")
    return match.group(1)


def verify_sources() -> tuple[str, str]:
    src = ROOT / "src"
    common_paths = tuple(src / "common" / name for name in COMMON)
    mains = (src / "scientific-rationale.tex", src / "engineering-conformance.tex")
    for path in common_paths:
        require(path.is_file(), f"missing common source: {path.name}")
    bindings_path = src / "common" / "bindings.tex"
    require(bindings_path.is_file(), "missing source bindings")
    bindings = bindings_path.read_text(encoding="utf-8")
    require(sha256(OWNER_DIRECTIVE.read_bytes()) == OWNER_DIRECTIVE_SHA256,
            "owner-directive SHA-256 mismatch")

    for main in mains:
        text = main.read_text(encoding="utf-8")
        require("\\input{common/bindings}" in text, f"{main.name} lacks bindings")
        for name in COMMON:
            require(f"\\input{{common/{name[:-4]}}}" in text, f"{main.name} lacks {name}")
        for cover_macro in ("StageAPacketSHA", "CommonCoreSHA", "OwnerDirectiveSHA", "BuildRecordID"):
            require(f"\\{cover_macro}" in text, f"{main.name} lacks cover binding {cover_macro}")

    common_bytes = b"".join(path.read_bytes() for path in common_paths)
    common_digest = sha256(common_bytes)
    require(macro(bindings, "CommonCoreSHA") == common_digest, "common-core cover binding")
    require(macro(bindings, "StageAPacketSHA") == ARCHIVE_SHA256, "Stage A cover binding")
    require(macro(bindings, "OwnerDirectiveSHA") == OWNER_DIRECTIVE_SHA256,
            "owner-directive cover binding")
    require(macro(bindings, "RationaleSourceSHA") == sha256(mains[0].read_bytes()),
            "rationale-source cover binding")
    require(macro(bindings, "EngineeringSourceSHA") == sha256(mains[1].read_bytes()),
            "engineering-source cover binding")

    shared = "\n".join(path.read_text(encoding="utf-8") for path in common_paths)
    all_source = shared + "\n" + "\n".join(main.read_text(encoding="utf-8") for main in mains)
    normalized = all_source.replace("\\_", "_")
    reqs = set(re.findall(r"SCI-POINT-REQ-[0-9]{3}", shared))
    preds = set(re.findall(r"SCI-POINT-PRED-[0-9]{3}", shared))
    unavs = set(re.findall(r"SCI-POINT-UNAV-[0-9]{3}", shared))
    require(reqs == {f"SCI-POINT-REQ-{i:03d}" for i in range(1, 39)}, "requirement IDs")
    require(preds == {f"SCI-POINT-PRED-{i:03d}" for i in range(1, 33)}, "prediction IDs")
    require(unavs == {f"SCI-POINT-UNAV-{i:03d}" for i in range(1, 24)}, "unavailable IDs")

    required_tokens = (
        "applicability_unknown", "decision_unavailable", "incomplete", "not_produced",
        "applied", "fit_realized", "complete_publication_candidate",
        "POINT-FIT-AMPLITUDE-COMPONENT",
        "POINT-SOURCE-ASSOCIATED-AMPLITUDE-DIAGNOSTIC",
        "POINT-DYNAMIC-RANGE-DIAGNOSTIC/FITTED-AMPLITUDE-OVER-FULL-MAP-RMS@1",
        "POINT-FORMAL-AMPLITUDE-STANDARDIZATION@1",
        "diagnostic_display_only", "approximately_centered",
        "C_{\\Delta}", "R^{\\mathrm{fixed}}",
        "R^{\\mathrm{POINT\\mbox{-}FP}\\mid\\mathrm{parent\\mbox{-}fixed}}",
        "R^{\\mathrm{chain\\mbox{-}FP}}",
        "POINT-FORMAL-AMPLITUDE-MAGNITUDE-STANDARDIZATION@1",
        "no proposition", "inherited parent observation/exposure lineage",
    )
    for token in required_tokens:
        require(token in normalized, f"required source token: {token}")
    for stale in ("not_realized", "approximately-centered", "POINT-AMPLITUDE-DIAGNOSTIC"):
        require(stale not in normalized, f"stale source token: {stale}")
    require("realization: \\path{realized}, \\path{incomplete}, \\path{failed}, or"
            in all_source, "exact realization enumeration")
    require("applicability: \\path{applicable}, \\path{inapplicable}, or"
            in all_source, "exact applicability enumeration")
    require("Use not requested & \\path{not_requested} & no proposition & no proposition & \\path{not_produced}"
            in all_source, "source not-requested SCI-VAL row")
    require("Requested and known inapplicable; decision artifact written & \\path{requested} & \\path{inapplicable} & no proposition & \\path{realized}"
            in all_source, "source inapplicable SCI-VAL row")
    require("POINT-FULL-PROCEDURE-RESPONSE-STATE" not in normalized,
            "ambiguous generic full-procedure response role")

    records = json.loads((ROOT / "STAGE_B_R0_3_RECORDS.json").read_text(encoding="utf-8"))
    require(records["document_revision"] == "r0.3", "records revision")
    require(records["sci_val"]["applicability"][-1] == "applicability_unknown",
            "records applicability types")
    require(records["sci_val"]["realization"] ==
            ["realized", "incomplete", "failed", "not_produced"],
            "records realization types")
    require(len(records["logical_records"]) == 8, "logical record count")
    require(len(records["draft_named_use_profiles"]) == 4, "draft profile count")
    require(len(records["point_boundary_roles"]) == 4, "boundary-role count")
    require(len(records["required_method_record_templates"]) == 3, "method templates")
    dispositions = records["sci_val"]["dispositions"]
    require(dispositions["not_requested"] == ["not_requested", None, None, "not_produced"],
            "not-requested SCI-VAL semantics")
    require(dispositions["requested_known_inapplicable_artifact_written"] ==
            ["requested", "inapplicable", None, "realized"],
            "inapplicable SCI-VAL semantics")
    for key in ("missing_profile_artifact_written", "missing_source_binding_artifact_written",
                "unresolved_structural_scope_artifact_written"):
        require(dispositions[key][-1] == "realized", f"written accounting realization: {key}")
    require(records["response_families"]["families_alias"] is False,
            "response families do not alias")
    require(records["response_families"]["finite_difference_jacobian_composition_without_theorem"] is False,
            "finite-difference composition gate")
    require(records["derived_diagnostics"][0]["signed"] is True and
            records["derived_diagnostics"][1]["signed"] is True,
            "signed diagnostics")
    require(records["publication_semantics"]["downstream_success_required_for_base_fit"] is False,
            "downstream/base-fit separation")
    require(records["published_product_lineage"]["point_creates_exposure_identity"] is False,
            "POINT exposure ownership")

    parity = json.loads((ROOT / "STAGE_B_R0_3_PARITY_REPORT.json").read_text(encoding="utf-8"))
    require(parity["document_revision"] == "r0.3", "parity report revision")
    require(parity["targeted_semantic_audit"]["status"] == "pass",
            "targeted semantic audit")

    source_manifest = json.loads((ROOT / "STAGE_B_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(source_manifest["document_revision"] == "r0.3", "source manifest revision")
    require(source_manifest["author_packet_sha256"] == ARCHIVE_SHA256,
            "source manifest Stage A binding")
    require(source_manifest["owner_directive"]["sha256"] == OWNER_DIRECTIVE_SHA256,
            "source manifest owner-directive binding")
    require(source_manifest["common_core"]["sha256"] == common_digest,
            "source manifest common-core binding")
    for entry in source_manifest["files"]:
        require(sha256((ROOT / entry["path"]).read_bytes()) == entry["sha256"],
                f"source manifest hash: {entry['path']}")
    return all_source, common_digest


def verify_pdfs(common_digest: str) -> list[tuple[str, int, str]]:
    build = json.loads((ROOT / "STAGE_B_BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    require(build["document_revision"] == "r0.3", "build manifest revision")
    by_name = {pathlib.Path(entry["path"]).name: entry for entry in build["pdfs"]}
    require(set(by_name) == set(PDFS), "build manifest PDF set")
    results: list[tuple[str, int, str]] = []
    for filename, title_fragment in PDFS.items():
        path = ROOT / "pdf" / filename
        reader = PdfReader(path)
        require(len(reader.pages) == by_name[filename]["pages"], f"PDF pages: {filename}")
        require(not (reader.get_fields() or {}), f"unexpected PDF form fields: {filename}")
        title = str((reader.metadata or {}).get("/Title", ""))
        require(title_fragment in title and "r0.3" in title, f"PDF title: {filename}")
        require("Grant Wilson, scientific owner" in str((reader.metadata or {}).get("/Author", "")),
                f"PDF author metadata: {filename}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        compact = re.sub(r"\s+", "", text)
        for fragment in (
            "SCI-POINT", "r0.3", ARCHIVE_SHA256, OWNER_DIRECTIVE_SHA256, common_digest,
            "POINT-FIT-AMPLITUDE-COMPONENT",
            "POINT-SOURCE-ASSOCIATED-AMPLITUDE-DIAGNOSTIC",
            "applicability_unknown", "SCI-POINT-PRED-032",
        ):
            require(re.sub(r"\s+", "", fragment) in compact,
                    f"PDF fragment {fragment!r}: {filename}")
        digest = sha256(path.read_bytes())
        require(digest == by_name[filename]["sha256"], f"PDF digest: {filename}")
        results.append((filename, len(reader.pages), digest))
    require(build["visual_qa"]["pages_inspected"] == sum(row[1] for row in results),
            "visual-QA page count")
    require(build["visual_qa"]["result"] == "pass", "visual-QA result")
    require(build["source_manifest_sha256"] ==
            sha256((ROOT / "STAGE_B_SOURCE_MANIFEST.json").read_bytes()),
            "build/source manifest binding")
    qa = json.loads((ROOT / "STAGE_B_R0_3_PDF_QA_REPORT.json").read_text(encoding="utf-8"))
    require(qa["result"] == "pass" and qa["pages_inspected"] == sum(row[1] for row in results),
            "PDF QA report")
    clean = json.loads((ROOT / "STAGE_B_R0_3_CLEAN_BUILD_REPORT.json").read_text(encoding="utf-8"))
    require(clean["result"] == "pass" and
            all(row["extracted_text_matches_delivered_pdf"] for row in clean["results"]),
            "clean-build report")
    return results


def verify_delivery() -> tuple[int, int, str]:
    sidecar_hash = DELIVERY.with_suffix(DELIVERY.suffix + ".sha256")
    sidecar_bytes = DELIVERY.with_suffix(DELIVERY.suffix + ".bytes")
    raw = DELIVERY.read_bytes()
    require(sidecar_hash.read_text(encoding="utf-8").strip() == sha256(raw),
            "delivery hash sidecar")
    require(int(sidecar_bytes.read_text(encoding="utf-8").strip()) == len(raw),
            "delivery byte sidecar")
    members = verify_safe_tar(DELIVERY, DELIVERY_PREFIX)
    names = {pathlib.PurePosixPath(member.name).name for member in members}
    for required in (
        ARCHIVE.name, "AUTHOR_PACKET_MANIFEST.md", "bindings.tex",
        "scientific-rationale.tex", "engineering-conformance.tex",
        "SCIENTIFIC_OWNER_TARGETED_STAGE_B_R0_3_DIRECTIVE_2026-09-03.md",
        "STAGE_B_R0_3_RECORDS.json", "STAGE_B_R0_3_TARGETED_AMENDMENTS.md",
        "STAGE_B_R0_3_PARITY_REPORT.json", "STAGE_B_R0_3_SEMANTIC_CHANGE_REPORT.md",
        "PROPOSED_SCIENTIFIC_OWNER_FREEZE_R0_3.md", "STAGE_B_SOURCE_MANIFEST.json",
        "STAGE_B_BUILD_MANIFEST.json", "STAGE_B_R0_3_CLEAN_BUILD_REPORT.json",
        "STAGE_B_R0_3_PDF_QA_REPORT.json", "verify_stage_b.py",
        *PDFS.keys(),
    ):
        require(required in names, f"delivery member: {required}")
    with tarfile.open(DELIVERY, "r:gz") as bundle:
        for member in members:
            relative = pathlib.PurePosixPath(member.name).relative_to(
                pathlib.PurePosixPath(DELIVERY_PREFIX)
            )
            local = ROOT.joinpath(*relative.parts)
            require(local.is_file(), f"delivery source exists: {relative}")
            archived = bundle.extractfile(member).read()
            require(archived == local.read_bytes(), f"delivery byte parity: {relative}")
    return len(members), len(raw), sha256(raw)


def main() -> int:
    try:
        admitted = verify_packet()
        verify_owner_parity(admitted)
        _, common_digest = verify_sources()
        pdfs = verify_pdfs(common_digest)
        delivery_members, delivery_bytes, delivery_digest = verify_delivery()
    except (CheckFailure, KeyError, ValueError, tarfile.TarError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS author_packet bytes={ARCHIVE_BYTES} sha256={ARCHIVE_SHA256}")
    print(f"PASS author_manifest sha256={MANIFEST_SHA256}")
    print("PASS author_members=39 admitted=37 unsafe=0 unresolved_links=0")
    print("PASS owner_parity=11_ids method_dispositions=3")
    print(f"PASS owner_directive sha256={OWNER_DIRECTIVE_SHA256}")
    print(f"PASS common_core files=6 sha256={common_digest}")
    print("PASS requirements=38 predictions=32 unavailable_states=23")
    print("PASS sci_val_axes=4 logical_records=8 boundary_roles=4 profiles=4 method_templates=3")
    print("PASS targeted_semantics=SCI_VAL,response_types,signed_diagnostics,dependencies,weights,lineage")
    for filename, pages, digest in pdfs:
        print(f"PASS pdf={filename} pages={pages} sha256={digest}")
    print(f"PASS delivery members={delivery_members} bytes={delivery_bytes} sha256={delivery_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
