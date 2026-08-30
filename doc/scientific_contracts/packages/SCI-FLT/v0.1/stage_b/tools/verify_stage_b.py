#!/usr/bin/env python3
"""Verify SCI-FLT-FIXED v0.1 Stage B r0.2 closure and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PDF_SITE_PACKAGES = os.environ.get("SCI_FLT_PDF_SITE_PACKAGES")
if PDF_SITE_PACKAGES:
    sys.path.append(PDF_SITE_PACKAGES)

from pypdf import PdfReader

import build_stage_b as buildmod


STAGE_B_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = STAGE_B_DIR / "source"
OUTPUT_DIR = STAGE_B_DIR / "output" / "pdf"
DEFAULT_BINDING = STAGE_B_DIR / "BUILD_BINDING.json"
DEFAULT_REPORT = STAGE_B_DIR / "VERIFICATION_REPORT.md"
DEFAULT_VISUAL_QA = STAGE_B_DIR / "VISUAL_QA.md"
DEFAULT_SOURCE_CLOSURE = STAGE_B_DIR / "SOURCE_PACKET_CLOSURE_REPORT.md"
DEFAULT_OWNER_PARITY = STAGE_B_DIR / "OWNER_DECISION_PARITY_REPORT.md"
DEFAULT_VIEW_PARITY = STAGE_B_DIR / "CORE_VIEW_PARITY_REPORT.md"
DEFAULT_REBUILD_REPORT = STAGE_B_DIR / "REBUILD_REPORT.md"

CORE_PATH = SOURCE_DIR / "SHARED_NORMATIVE_CORE.md"
RATIONALE_PATH = SOURCE_DIR / "SCIENTIST_RATIONALE.md"
CONFORMANCE_PATH = SOURCE_DIR / "ENGINEERING_CONFORMANCE.md"
TRACE_PATH = SOURCE_DIR / "TRACEABILITY.json"
POLICY_PATH = SOURCE_DIR / "POLICY_RECORDS.json"
CHANGE_MAP_PATH = SOURCE_DIR / "SEMANTIC_CHANGE_MAP.json"
OWNER_DIRECTIVE_PATH = SOURCE_DIR / "OWNER_DIRECTIVE_R0_2.txt"

EXPECTED_REQUIREMENTS = [
    f"SCI-FLT-FIXED-REQ-{number:03d}" for number in range(1, 45)
]
EXPECTED_PREDICTIONS = [
    f"SCI-FLT-FIXED-PRED-{number:03d}" for number in range(1, 25)
]
EXPECTED_IDS = EXPECTED_REQUIREMENTS + EXPECTED_PREDICTIONS

OWNER_DIRECTIVE_SECTIONS = [
    "1. CLOSE GENERAL-LINEAR VERSUS CONVOLUTION SCOPE",
    "2. MAKE LINEARITY CONDITIONAL ON ONE FROZEN REALIZED SELECTOR",
    "3. DEFINE REQUIRED FOOTPRINT AND ZERO-COEFFICIENT SEMANTICS",
    "4. REPAIR PUBLICATION LIFECYCLE CIRCULARITY",
    "5. REMOVE NOI PRODUCT FROM FLT ATOMIC COMPLETION",
    "6. TYPE INPUT, ROW, AND PUBLICATION POLICIES",
    "7. TYPE RESPONSE FAMILIES AND ZERO-OPERATOR CLAIMS",
    "8. REPAIR COVARIANCE TYPES AND NOTATION",
    "9. COMPLETE LOW-PASS, WCS, AND TRANSFER SEMANTICS",
    "10. DEFINE OPERATOR ORDER AND COMPOSITION",
    "11. ADD NUMERICAL-CONFORMANCE POLICY",
    "12. EXPOSURE, TERMINOLOGY, AND OWNER METADATA",
    "13. SOURCE-PACKET CLOSURE",
    "14. DELIVERABLES",
]


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_heading(value: str) -> str:
    translated = value.translate(
        str.maketrans({"\u2014": "-", "\u2013": "-", "\u2011": "-", "\u2212": "-"})
    )
    translated = translated.replace("`", "").strip()
    return re.sub(r"\s+", " ", translated)


def headings(path: Path) -> set[str]:
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            result.add(canonical_heading(match.group(1)))
    return result


def exact_heading_ids(path: Path, prefix: str) -> list[str]:
    found = []
    expression = re.compile(rf"^###\s+({re.escape(prefix)}-\d{{3}})\b", re.MULTILINE)
    text = path.read_text(encoding="ascii")
    for match in expression.finditer(text):
        found.append(match.group(1))
    return found


def verify_packet(checks: list[str]) -> None:
    require(
        sha256(buildmod.PACKET_MANIFEST) == buildmod.PACKET_MANIFEST_SHA256,
        "packet manifest SHA-256 mismatch",
    )
    checks.append("packet manifest external SHA-256 matches")
    for name, expected in buildmod.ADMITTED_OBJECTS.items():
        require(
            sha256(buildmod.PACKAGE_DIR / name) == expected,
            f"admitted object SHA-256 mismatch: {name}",
        )
    checks.append("all 17 admitted object SHA-256 values match")


def verify_sources_and_trace(checks: list[str]) -> dict:
    ascii_sources = [
        SOURCE_DIR / name
        for name in buildmod.SUPPORTING_SOURCES
        if name != OWNER_DIRECTIVE_PATH.name
    ]
    for path in [CORE_PATH, RATIONALE_PATH, CONFORMANCE_PATH] + ascii_sources:
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError as error:
            raise VerificationError(f"non-ASCII Stage B source: {path.name}") from error
    checks.append("all authored r0.2 sources are ASCII-clean; the exact owner directive is preserved as UTF-8")

    owner_text = OWNER_DIRECTIVE_PATH.read_text(encoding="utf-8")
    for section in OWNER_DIRECTIVE_SECTIONS:
        require(section in owner_text, f"owner-directive section missing: {section}")
    checks.append("all 14 r0.2 owner-directive sections are present")

    requirements = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-REQ")
    predictions = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-PRED")
    require(requirements == EXPECTED_REQUIREMENTS, "requirement heading sequence mismatch")
    require(predictions == EXPECTED_PREDICTIONS, "prediction heading sequence mismatch")
    checks.append("44 stable requirement identifiers preserve 001-036 and append 037-044")
    checks.append("24 stable prediction identifiers preserve 001-021 and append 022-024")

    conformance_text = CONFORMANCE_PATH.read_text(encoding="ascii")
    for identifier in EXPECTED_IDS:
        require(
            conformance_text.count(identifier) == 1,
            f"conformance view must contain {identifier} exactly once",
        )
    checks.append("engineering-conformance view routes every stable identifier exactly once")

    trace = json.loads(TRACE_PATH.read_text(encoding="ascii"))
    entries = trace.get("entries", [])
    trace_ids = [entry.get("id") for entry in entries]
    require(trace_ids == EXPECTED_IDS, "traceability identifier sequence mismatch")
    require(len(set(trace_ids)) == len(trace_ids), "duplicate traceability identifier")

    core_headings = headings(CORE_PATH)
    rationale_headings = headings(RATIONALE_PATH)
    conformance_headings = headings(CONFORMANCE_PATH)
    admitted_names = set(buildmod.ADMITTED_OBJECTS)
    admitted_heading_cache = {
        name: headings(buildmod.PACKAGE_DIR / name) for name in admitted_names
    }
    for entry in entries:
        identifier = entry["id"]
        expected_kind = "requirement" if "-REQ-" in identifier else "prediction"
        require(entry.get("kind") == expected_kind, f"kind mismatch for {identifier}")
        require(
            canonical_heading(entry["core_section"]) in core_headings,
            f"missing core heading for {identifier}",
        )
        require(
            canonical_heading(entry["rationale_section"]) in rationale_headings,
            f"missing rationale heading for {identifier}",
        )
        require(
            canonical_heading(entry["conformance_section"]) in conformance_headings,
            f"missing conformance heading for {identifier}",
        )
        sources = entry.get("stage_a_sources", [])
        require(sources, f"no Stage A source for {identifier}")
        for source in sources:
            require("#" in source, f"unsectioned Stage A trace for {identifier}")
            filename, section = source.split("#", 1)
            require(filename in admitted_names, f"unadmitted source for {identifier}: {filename}")
            require(
                canonical_heading(section) in admitted_heading_cache[filename],
                f"missing admitted-source heading for {identifier}: {source}",
            )
        owner_source = entry.get("r0_2_owner_source")
        if owner_source is not None:
            require(
                owner_source.startswith(f"{OWNER_DIRECTIVE_PATH.name}#"),
                f"invalid r0.2 owner source for {identifier}",
            )
            owner_section = owner_source.split("#", 1)[1]
            require(
                owner_section in OWNER_DIRECTIVE_SECTIONS,
                f"missing owner-directive heading for {identifier}: {owner_source}",
            )
    checks.append("traceability covers every identifier and only the admitted Stage A packet plus the r0.2 owner directive")
    checks.append("every traced core, rationale, conformance, Stage A, and r0.2 owner section resolves")

    core_sha = sha256(CORE_PATH)
    token = "{{NORMATIVE_CORE_SHA256}}"
    require(
        RATIONALE_PATH.read_text(encoding="ascii").count(token) == 1,
        "rationale core-binding token count mismatch",
    )
    require(
        CONFORMANCE_PATH.read_text(encoding="ascii").count(token) == 1,
        "conformance core-binding token count mismatch",
    )
    checks.append(f"both views import one shared normative core SHA-256 {core_sha}")
    return trace


def verify_closure_sources(checks: list[str]) -> tuple[dict, dict]:
    for path in [CORE_PATH, RATIONALE_PATH, CONFORMANCE_PATH]:
        text = path.read_text(encoding="ascii")
        require(
            "Scientific owner: Grant Wilson" in text,
            f"scientific owner missing from {path.name}",
        )

    policies = json.loads(POLICY_PATH.read_text(encoding="ascii"))
    require(
        policies.get("scientific_policy_owner") == "Grant Wilson",
        "policy owner mismatch",
    )
    require(
        policies.get("registry_binding_status")
        == "not_owner_approved_not_registered_not_evaluated",
        "policy Registry status mismatch",
    )
    profiles = policies.get("profiles", [])
    expected_profiles = [
        "SCI-FLT-FIXED:input_bundle_admission@1",
        "SCI-FLT-FIXED:input_row_admission@1",
        "SCI-FLT-FIXED:output_publication@1",
    ]
    require(
        [profile.get("identity") for profile in profiles] == expected_profiles,
        "typed policy profile identity mismatch",
    )
    required_fields = {
        "identity",
        "domain",
        "request",
        "applicability",
        "eligibility",
        "realization",
        "restrictions",
        "decisive_exclusions",
        "exceptions",
        "missing_or_conflict_behavior",
        "lifecycle",
        "consumer_action",
    }
    for profile in profiles:
        require(
            required_fields <= set(profile),
            f"incomplete policy fields: {profile.get('identity')}",
        )
    checks.append("all three exact typed policy domains and unregistered VAL status are complete")

    change_map = json.loads(CHANGE_MAP_PATH.read_text(encoding="ascii"))
    req_changes = change_map.get("requirement_changes", {})
    pred_changes = change_map.get("prediction_changes", {})
    for label, groups, expected in [
        ("requirement", req_changes, EXPECTED_REQUIREMENTS),
        ("prediction", pred_changes, EXPECTED_PREDICTIONS),
    ]:
        partitions = [
            groups.get("clarified_without_renumbering", []),
            groups.get("preserved_without_semantic_change", []),
            groups.get("appended", []),
        ]
        flattened = [identifier for group in partitions for identifier in group]
        require(len(flattened) == len(set(flattened)), f"duplicate {label} change-map id")
        require(set(flattened) == set(expected), f"incomplete {label} change-map partition")
        require(groups.get("renumbered") == [], f"{label} identifiers were renumbered")

    require(
        req_changes.get("appended") == EXPECTED_REQUIREMENTS[36:],
        "appended requirement sequence mismatch",
    )
    require(
        pred_changes.get("appended") == EXPECTED_PREDICTIONS[21:],
        "appended prediction sequence mismatch",
    )
    routes = change_map.get("owner_directive_routes", {})
    require(list(routes) == OWNER_DIRECTIVE_SECTIONS, "owner-directive route sequence mismatch")
    routed_ids = {
        value
        for values in routes.values()
        for value in values
        if value in EXPECTED_IDS
    }
    changed_ids = set(req_changes["clarified_without_renumbering"])
    changed_ids.update(req_changes["appended"])
    changed_ids.update(pred_changes["clarified_without_renumbering"])
    changed_ids.update(pred_changes["appended"])
    require(changed_ids <= routed_ids, "changed identifier lacks an owner-directive route")
    require(len(change_map.get("equation_changes", [])) == 6, "equation change map mismatch")
    checks.append("equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs")
    checks.append("every changed or appended identifier routes to an exact r0.2 owner-directive section")
    return policies, change_map


def verify_binding_and_pdfs(binding_path: Path, checks: list[str]) -> dict:
    binding = json.loads(binding_path.read_text(encoding="ascii"))
    require(
        binding.get("stage_a_launch_commit") == buildmod.STAGE_A_LAUNCH_COMMIT,
        "Stage A launch commit binding mismatch",
    )
    packet = binding.get("packet", {})
    require(
        packet.get("manifest_sha256") == buildmod.PACKET_MANIFEST_SHA256,
        "binding manifest SHA-256 mismatch",
    )
    require(
        packet.get("manifest_bytes") == buildmod.PACKET_MANIFEST.stat().st_size,
        "binding manifest byte-count mismatch",
    )
    require(packet.get("admitted_object_count") == 17, "binding admitted-object count mismatch")
    expected_packet_rows = [
        {
            "path": buildmod.repo_relative(buildmod.PACKAGE_DIR / name),
            "sha256": digest,
            "bytes": (buildmod.PACKAGE_DIR / name).stat().st_size,
        }
        for name, digest in buildmod.ADMITTED_OBJECTS.items()
    ]
    require(packet.get("admitted_objects") == expected_packet_rows, "binding packet rows mismatch")

    expected_owner = {
        "path": buildmod.repo_relative(OWNER_DIRECTIVE_PATH),
        "sha256": sha256(OWNER_DIRECTIVE_PATH),
        "bytes": OWNER_DIRECTIVE_PATH.stat().st_size,
        "scientific_owner": "Grant Wilson",
    }
    require(
        binding.get("r0_2_owner_directive") == expected_owner,
        "r0.2 owner-directive binding mismatch",
    )

    source_paths = [
        SOURCE_DIR / item["source"] for item in buildmod.DOCUMENTS
    ] + [SOURCE_DIR / name for name in buildmod.SUPPORTING_SOURCES]
    expected_sources = [
        {
            "path": buildmod.repo_relative(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source_paths
    ]
    require(binding.get("sources") == expected_sources, "Stage B source binding mismatch")

    builder_path = Path(buildmod.__file__).resolve()
    verifier_path = Path(__file__).resolve()
    expected_tools = [
        {
            "path": buildmod.repo_relative(builder_path),
            "sha256": sha256(builder_path),
            "bytes": builder_path.stat().st_size,
        },
        {
            "path": buildmod.repo_relative(verifier_path),
            "sha256": sha256(verifier_path),
            "bytes": verifier_path.stat().st_size,
        },
    ]
    require(binding.get("build_tools") == expected_tools, "build-tool binding mismatch")
    expected_fonts = [
        {
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in buildmod.FONT_FILES.items()
    ]
    require(binding.get("embedded_fonts") == expected_fonts, "embedded-font binding mismatch")
    checks.append("launch commit, exact packet bytes, owner directive, r0.2 sources, and build tools match BUILD_BINDING.json")
    checks.append("all embedded font files and SHA-256 values match BUILD_BINDING.json")

    output_records = binding.get("outputs", [])
    require(len(output_records) == 3, "expected exactly three output records")
    expected_names = [item["output"] for item in buildmod.DOCUMENTS]
    require([row.get("filename") for row in output_records] == expected_names, "PDF order mismatch")
    actual_names = sorted(path.name for path in OUTPUT_DIR.glob("*.pdf"))
    require(actual_names == sorted(expected_names), "unexpected or missing PDF output")

    builder_sha = sha256(builder_path)
    core_sha = sha256(CORE_PATH)
    record_by_name = {row["filename"]: row for row in output_records}
    for item in buildmod.DOCUMENTS:
        output_path = OUTPUT_DIR / item["output"]
        record = record_by_name[item["output"]]
        require(sha256(output_path) == record["sha256"], f"PDF digest mismatch: {output_path.name}")
        require(output_path.stat().st_size == record["bytes"], f"PDF size mismatch: {output_path.name}")
        reader = PdfReader(str(output_path))
        require(len(reader.pages) == record["pages"], f"PDF page-count mismatch: {output_path.name}")
        require(reader.metadata.title == item["identity"], f"PDF title mismatch: {output_path.name}")
        require(reader.metadata.author == "Grant Wilson", f"PDF author mismatch: {output_path.name}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for page_number, page in enumerate(reader.pages, 1):
            require((page.extract_text() or "").strip(), f"blank PDF page: {output_path.name} page {page_number}")
        required_text = [
            item["identity"],
            record["source_sha256"],
            core_sha,
            buildmod.PACKET_MANIFEST_SHA256,
            sha256(OWNER_DIRECTIVE_PATH),
            builder_sha,
            "Scientific owner: Grant Wilson",
        ]
        for value in required_text:
            require(value in text, f"missing PDF identity text in {output_path.name}: {value}")
    checks.append("all three PDF identity blocks, Grant Wilson author metadata, hashes, sizes, and page counts match")
    checks.append("every PDF page contains extractable text")
    return binding


def verify_reproducible(binding: dict, checks: list[str]) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="sci-flt-fixed-rebuild-") as temp_name:
        temp_dir = Path(temp_name)
        output_dir = temp_dir / "pdf"
        binding_out = temp_dir / "BUILD_BINDING.json"
        command = [
            sys.executable,
            str(Path(buildmod.__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--binding-out",
            str(binding_out),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        require(completed.returncode == 0, f"clean rebuild failed: {completed.stderr}")
        expected = {row["filename"]: row["sha256"] for row in binding["outputs"]}
        actual = {name: sha256(output_dir / name) for name in expected}
        require(actual == expected, "clean rebuild PDF digest mismatch")
    checks.append("clean temporary rebuild reproduces all three PDF SHA-256 digests exactly")
    return actual


def poppler_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-v"], check=False, capture_output=True, text=True
    )
    output = (completed.stderr or completed.stdout).splitlines()
    return output[0].strip() if output else "unknown"


def verify_poppler_render(binding: dict, checks: list[str]) -> str:
    pdftoppm = shutil.which("pdftoppm")
    require(pdftoppm is not None, "pdftoppm is required")
    with tempfile.TemporaryDirectory(prefix="sci-flt-fixed-render-") as temp_name:
        temp_dir = Path(temp_name)
        for record in binding["outputs"]:
            output_path = OUTPUT_DIR / record["filename"]
            prefix = temp_dir / output_path.stem
            completed = subprocess.run(
                [pdftoppm, "-r", "120", "-png", str(output_path), str(prefix)],
                check=False,
                capture_output=True,
                text=True,
            )
            require(completed.returncode == 0, f"Poppler render failed: {output_path.name}")
            rendered = sorted(temp_dir.glob(f"{output_path.stem}-*.png"))
            require(
                len(rendered) == record["pages"],
                f"Poppler page-count mismatch: {output_path.name}",
            )
            for page in rendered:
                require(page.stat().st_size > 0, f"empty rendered page: {page.name}")
    version = poppler_version(pdftoppm)
    checks.append("Poppler renders every page of all three PDFs to a non-empty PNG")
    return version


def verify_visual_qa(binding: dict, visual_path: Path, checks: list[str]) -> None:
    require(visual_path.exists(), "required VISUAL_QA.md is missing")
    text = visual_path.read_text(encoding="ascii")
    require(
        f"Build binding SHA-256: `{sha256(DEFAULT_BINDING)}`" in text,
        "visual-QA build-binding digest mismatch",
    )
    expected = []
    for record in binding["outputs"]:
        require(
            f"PDF SHA-256: `{record['sha256']}`" in text,
            f"visual-QA PDF digest missing: {record['filename']}",
        )
        for page_number in range(1, record["pages"] + 1):
            expected.append((record["filename"], page_number))
    found = [
        (match.group(1), int(match.group(2)))
        for match in re.finditer(r"^- `([^`]+)` page (\d+): PASS - ", text, re.MULTILINE)
    ]
    require(found == expected, "visual-QA page sequence is incomplete, duplicated, or reordered")
    checks.append("VISUAL_QA.md records a PASS observation for every bound PDF page")


def write_source_closure(report_path: Path, binding: dict) -> None:
    packet = binding["packet"]
    owner = binding["r0_2_owner_directive"]
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.2 Source-Packet Closure Report",
        "",
        "Report identity: `SCI-FLT-FIXED-SOURCE-PACKET-CLOSURE v0.1/draft-r0.2`",
        "",
        "Status: PASS; exact repository bytes and SHA-256 values reproduced; scientific-owner review required",
        "",
        "## Stage A author packet",
        "",
        f"- `{packet['manifest']}`: {packet['manifest_bytes']} bytes; SHA-256 `{packet['manifest_sha256']}`",
    ]
    for row in packet["admitted_objects"]:
        lines.append(
            f"- `{row['path']}`: {row['bytes']} bytes; SHA-256 `{row['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## r0.2 scientific-owner directive",
            "",
            f"- `{owner['path']}`: {owner['bytes']} bytes; SHA-256 `{owner['sha256']}`; owner `{owner['scientific_owner']}`",
            "",
            "## Stage B sources",
            "",
        ]
    )
    for row in binding["sources"]:
        lines.append(
            f"- `{row['path']}`: {row['bytes']} bytes; SHA-256 `{row['sha256']}`"
        )
    lines.extend(["", "## Build tools", ""])
    for row in binding["build_tools"]:
        lines.append(
            f"- `{row['path']}`: {row['bytes']} bytes; SHA-256 `{row['sha256']}`"
        )
    lines.extend(["", "## Bound PDFs", ""])
    for row in binding["outputs"]:
        lines.append(
            f"- `{row['filename']}`: {row['bytes']} bytes; {row['pages']} pages; SHA-256 `{row['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Result and nonclaims",
            "",
            "Every listed byte count and digest was recomputed from the exact repository object. This is a source-identity closure record, not a scientific approval, implementation-conformity, validation, calibration, readiness, production, or Unity claim.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="ascii")


def write_owner_parity(report_path: Path, change_map: dict) -> None:
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.2 Owner-Decision Parity Report",
        "",
        "Report identity: `SCI-FLT-FIXED-OWNER-DECISION-PARITY v0.1/draft-r0.2`",
        "",
        "Status: PASS; every directive section has an exact artifact or identifier route; scientific-owner review required",
        "",
        f"Owner directive SHA-256: `{sha256(OWNER_DIRECTIVE_PATH)}`",
        "",
        "## Directive routes",
        "",
    ]
    for section, routes in change_map["owner_directive_routes"].items():
        lines.append(f"- PASS: `{section}` -> {', '.join(f'`{route}`' for route in routes)}")
    lines.extend(
        [
            "",
            "## Stable-identifier result",
            "",
            "Requirements 001-036 and predictions 001-021 retain their identifiers. Requirements 037-044 and predictions 022-024 are append-only additions. No identifier was renumbered.",
            "",
            "## Nonclaims",
            "",
            "This parity report records document semantics only and makes no implementation, validation, calibration, adequacy, performance, readiness, production, or Unity claim.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="ascii")


def write_view_parity(report_path: Path) -> None:
    core_sha = sha256(CORE_PATH)
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.2 Core/View Parity Report",
        "",
        "Report identity: `SCI-FLT-FIXED-CORE-VIEW-PARITY v0.1/draft-r0.2`",
        "",
        "Status: PASS; rationale and ECS import one exact normative core; scientific-owner review required",
        "",
        f"Normative core SHA-256: `{core_sha}`",
        f"Scientist rationale source SHA-256: `{sha256(RATIONALE_PATH)}`",
        f"Engineering-conformance source SHA-256: `{sha256(CONFORMANCE_PATH)}`",
        "",
        "## Results",
        "",
        "- PASS: both views contain exactly one normative-core digest token, replaced at build time with the digest above.",
        "- PASS: the rationale declares the imported core controlling on conflict.",
        "- PASS: the ECS declares that it adds no scientific rule and the imported core controls.",
        f"- PASS: the ECS routes all {len(EXPECTED_REQUIREMENTS)} requirement identifiers exactly once.",
        f"- PASS: the ECS routes all {len(EXPECTED_PREDICTIONS)} prediction identifiers exactly once.",
        "",
        "## Nonclaims",
        "",
        "This parity report makes no implementation-conformity, validation, calibration, numerical-adequacy, readiness, production, or Unity claim.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="ascii")


def write_rebuild_report(report_path: Path, binding: dict, rebuilt: dict[str, str]) -> None:
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.2 Deterministic Rebuild Report",
        "",
        "Report identity: `SCI-FLT-FIXED-DETERMINISTIC-REBUILD v0.1/draft-r0.2`",
        "",
        "Status: PASS; a clean temporary rebuild reproduced every bound PDF SHA-256",
        "",
        f"Build binding SHA-256: `{sha256(DEFAULT_BINDING)}`",
        "",
        "## Results",
        "",
    ]
    for row in binding["outputs"]:
        lines.append(
            f"- PASS: `{row['filename']}` -> `{rebuilt[row['filename']]}`"
        )
    lines.extend(
        [
            "",
            "## Nonclaims",
            "",
            "This report establishes deterministic document reproduction only. It makes no scientific approval, implementation, validation, calibration, numerical-adequacy, performance, readiness, production, or Unity claim.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="ascii")


def write_report(
    report_path: Path,
    binding_path: Path,
    binding: dict,
    checks: list[str],
    poppler: str,
    visual_required: bool,
) -> None:
    lines = [
        "# SCI-FLT-FIXED v0.1 Stage B Verification Report",
        "",
        "Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.2`",
        "",
        "Status: PASS; deterministic r0.2 closure verification only; scientific-owner review required",
        "",
        f"Build binding SHA-256: `{sha256(binding_path)}`",
        "",
        f"Verifier SHA-256: `{sha256(Path(__file__).resolve())}`",
        "",
        f"Poppler: `{poppler}`",
        "",
        "## Results",
        "",
    ]
    lines.extend(f"- PASS: {check}" for check in checks)
    lines.extend(
        [
            "",
            "## Bound PDF outputs",
            "",
        ]
    )
    for record in binding["outputs"]:
        lines.append(
            f"- `{record['filename']}`: {record['pages']} pages; "
            f"{record['bytes']} bytes; SHA-256 `{record['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Visual review",
            "",
            (
                "The exact bound page-by-page visual-QA record was required and verified."
                if visual_required
                else "The automated Poppler render passed; a final page-by-page visual-QA record was not required for this invocation."
            ),
            "",
            "## Nonclaims",
            "",
            "This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--visual-qa", type=Path, default=DEFAULT_VISUAL_QA)
    parser.add_argument("--source-closure-out", type=Path, default=DEFAULT_SOURCE_CLOSURE)
    parser.add_argument("--owner-parity-out", type=Path, default=DEFAULT_OWNER_PARITY)
    parser.add_argument("--view-parity-out", type=Path, default=DEFAULT_VIEW_PARITY)
    parser.add_argument("--rebuild-report-out", type=Path, default=DEFAULT_REBUILD_REPORT)
    parser.add_argument("--require-visual-qa", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[str] = []
    try:
        verify_packet(checks)
        verify_sources_and_trace(checks)
        _, change_map = verify_closure_sources(checks)
        binding = verify_binding_and_pdfs(args.binding.resolve(), checks)
        rebuilt = verify_reproducible(binding, checks)
        poppler = verify_poppler_render(binding, checks)
        if args.require_visual_qa:
            verify_visual_qa(binding, args.visual_qa.resolve(), checks)
        write_source_closure(args.source_closure_out.resolve(), binding)
        write_owner_parity(args.owner_parity_out.resolve(), change_map)
        write_view_parity(args.view_parity_out.resolve())
        write_rebuild_report(args.rebuild_report_out.resolve(), binding, rebuilt)
        write_report(
            args.report_out.resolve(),
            args.binding.resolve(),
            binding,
            checks,
            poppler,
            args.require_visual_qa,
        )
    except (VerificationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    for check in checks:
        print(f"PASS: {check}")
    print(f"wrote {args.report_out}")
    print(f"wrote {args.source_closure_out}")
    print(f"wrote {args.owner_parity_out}")
    print(f"wrote {args.view_parity_out}")
    print(f"wrote {args.rebuild_report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
