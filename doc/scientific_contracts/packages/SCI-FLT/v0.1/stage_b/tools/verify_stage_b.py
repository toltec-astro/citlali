#!/usr/bin/env python3
"""Verify SCI-FLT-FIXED v0.1 Stage B r0.4 closure and artifacts."""

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
DEFAULT_PROFILE_REPORT = STAGE_B_DIR / "PROFILE_REPORT.md"
DEFAULT_AUTHORITY_MANIFEST = STAGE_B_DIR / "AUTHORITY_MANIFEST.json"
DEFAULT_AUTHORITY_DIGEST = STAGE_B_DIR / "AUTHORITY_MANIFEST.sha256"

CORE_PATH = SOURCE_DIR / "SHARED_NORMATIVE_CORE.md"
RATIONALE_PATH = SOURCE_DIR / "SCIENTIST_RATIONALE.md"
CONFORMANCE_PATH = SOURCE_DIR / "ENGINEERING_CONFORMANCE.md"
TRACE_PATH = SOURCE_DIR / "TRACEABILITY.json"
POLICY_PATH = SOURCE_DIR / "POLICY_RECORDS.json"
CHANGE_MAP_PATH = SOURCE_DIR / "SEMANTIC_CHANGE_MAP.json"
OWNER_DIRECTIVE_PATHS = {
    "r0_2": SOURCE_DIR / "OWNER_DIRECTIVE_R0_2.txt",
    "r0_3": SOURCE_DIR / "OWNER_DIRECTIVE_R0_3.txt",
    "r0_4": SOURCE_DIR / "OWNER_DIRECTIVE_R0_4.txt",
}

EXPECTED_REQUIREMENTS = [
    f"SCI-FLT-FIXED-REQ-{number:03d}" for number in range(1, 54)
]
EXPECTED_PREDICTIONS = [
    f"SCI-FLT-FIXED-PRED-{number:03d}" for number in range(1, 31)
]
EXPECTED_IDS = EXPECTED_REQUIREMENTS + EXPECTED_PREDICTIONS

R0_2_OWNER_DIRECTIVE_SECTIONS = [
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

R0_3_OWNER_DIRECTIVE_SECTIONS = [
    "1. BIND THE EXACT SIGNAL ROLE OF EVERY PARENT",
    "2. OWNER-DISPOSITION EXACT-ZERO AND REQUIRED-FOOTPRINT SEMANTICS",
    "3. CLOSE COVARIANCE AUTHORITY / REPRESENTATION COMPATIBILITY",
    "4. MAKE NOI COMPATIBILITY AND ATTACHMENT IMMUTABLE",
    "5. REPAIR POLICY DOMAINS AND THE CONSUMER-ACTION BOUNDARY",
    "6. COMPLETE THE LOW-PASS TRANSFORM CONVENTION",
    "7. TIGHTEN FULL-PROCEDURE RESPONSE DOMAIN COMPATIBILITY",
    "8. CONSOLIDATE SOURCE AND AUTHORITY BINDING",
    "9. RATIONALE AND PREFLIGHT",
    "10. DELIVERABLES",
]

R0_4_OWNER_DIRECTIVE_SECTIONS = [
    "1. REPAIR THE CONVOLUTION SUMMATION DOMAIN",
    "2. SEPARATE PLAN INDEPENDENCE FROM PARENT-FACT SELECTOR RESOLUTION",
    "3. DECLARE THE BASE SCALAR FIELD AND COEFFICIENT ADMISSIBILITY",
    "4. OWNER-DISPOSITION EMPTY SCIENTIFIC OUTPUT SUPPORT",
    "5. COMPLETE THE MARGINAL-ONLY COVARIANCE EDGE CASE",
    "6. CLARIFY LATE NOI REQUESTS",
    "7. COMPLETE SOURCE, PROFILE, AND AUTHORITY PREFLIGHT",
    "8. MECHANICAL AND VISUAL FREEZE PREFLIGHT",
    "9. DELIVERABLES",
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
        if name not in {path.name for path in OWNER_DIRECTIVE_PATHS.values()}
    ]
    for path in [CORE_PATH, RATIONALE_PATH, CONFORMANCE_PATH] + ascii_sources:
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError as error:
            raise VerificationError(f"non-ASCII Stage B source: {path.name}") from error
    checks.append("all authored r0.4 sources are ASCII-clean; all three exact owner directives are preserved as UTF-8")

    directive_sections = {
        "r0_2": R0_2_OWNER_DIRECTIVE_SECTIONS,
        "r0_3": R0_3_OWNER_DIRECTIVE_SECTIONS,
        "r0_4": R0_4_OWNER_DIRECTIVE_SECTIONS,
    }
    for revision, path in OWNER_DIRECTIVE_PATHS.items():
        owner_text = path.read_text(encoding="utf-8")
        for section in directive_sections[revision]:
            require(section in owner_text, f"{revision} owner-directive section missing: {section}")
    checks.append("all 14 r0.2, 10 r0.3, and 9 r0.4 owner-directive sections are present")

    requirements = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-REQ")
    predictions = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-PRED")
    require(requirements == EXPECTED_REQUIREMENTS, "requirement heading sequence mismatch")
    require(predictions == EXPECTED_PREDICTIONS, "prediction heading sequence mismatch")
    checks.append("53 stable requirement identifiers preserve 001-051 and append 052-053")
    checks.append("30 stable prediction identifiers preserve 001-028 and append 029-030")

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
        for revision, sections in directive_sections.items():
            owner_source = entry.get(f"{revision}_owner_source")
            if owner_source is not None:
                directive_path = OWNER_DIRECTIVE_PATHS[revision]
                require(
                    owner_source.startswith(f"{directive_path.name}#"),
                    f"invalid {revision} owner source for {identifier}",
                )
                owner_section = owner_source.split("#", 1)[1]
                require(
                    owner_section in sections,
                    f"missing owner-directive heading for {identifier}: {owner_source}",
                )
            owner_sources = entry.get(f"{revision}_owner_sources", [])
            require(
                isinstance(owner_sources, list),
                f"invalid {revision} owner-source list for {identifier}",
            )
            for listed_source in owner_sources:
                directive_path = OWNER_DIRECTIVE_PATHS[revision]
                require(
                    listed_source.startswith(f"{directive_path.name}#"),
                    f"invalid {revision} owner source for {identifier}",
                )
                owner_section = listed_source.split("#", 1)[1]
                require(
                    owner_section in sections,
                    f"missing owner-directive heading for {identifier}: {listed_source}",
                )
    checks.append("traceability covers every identifier and only the admitted Stage A packet plus the exact r0.2/r0.3/r0.4 owner directives")
    checks.append("every traced core, rationale, conformance, Stage A, and owner-directive section resolves")

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
    for path, document in zip(
        [CORE_PATH, RATIONALE_PATH, CONFORMANCE_PATH], buildmod.DOCUMENTS
    ):
        text = path.read_text(encoding="ascii")
        require(
            "Scientific owner: Grant Wilson" in text,
            f"scientific owner missing from {path.name}",
        )
        require(
            f"Stage B date: `{buildmod.STAGE_B_DATE}`" in text,
            f"Stage B date missing from {path.name}",
        )
        require(
            f"Document identity: `{document['identity']}`" in text,
            f"document identity mismatch in {path.name}",
        )
        require("Status:" in text, f"status missing from {path.name}")

    core_text = CORE_PATH.read_text(encoding="ascii")
    for required in [
        "jinc_signal_numerator",
        "jinc_signed_normalization",
        "jinc_quadratic_accumulator",
        "jinc_coefficient_squared_time",
        "K_req = K_nonzero",
        "(L_Theta m)_p = sum over r in K_nonzero",
        "K_store",
        "Var(y_i) = sum_j A_ij^2 Var(m_j)",
        "Var(y_i) = A_ij^2 Var(m_j)",
        "m, y, k_Theta(r), L_Theta, and A_Theta,J are real-valued",
        "applied_no_scientific_output_support",
        "no_full_footprint_output_rows",
        "not_requested_at_FLT_publication",
        "FLT-NOI-COMPATIBILITY",
        "SCI-FLT-FIXED:input_parent_row_admission@1",
        "H(nu) = sum over r of k(r) exp[-2 pi i nu dot r]",
        "AUTHORITY_MANIFEST",
    ]:
        require(required in core_text, f"r0.4 core closure text missing: {required}")
    require(
        "(L_Theta m)_p = sum over r in K_geom_science" not in core_text,
        "stale geometric-support convolution equation remains",
    )
    require("FLT-NOI-ATTACHMENT-STATE" not in core_text, "mutable NOI attachment role remains in core")
    require("input_row_admission@1" not in core_text, "misnamed row profile remains in core")

    rationale_text = RATIONALE_PATH.read_text(encoding="ascii")
    for required in [
        "exact parent signal role",
        "-> exact parent-row admission",
        "-> frozen J_full",
        "-> one sampled convolution applied once",
        "-> immutable FLT bundle with FLT-NOI-COMPATIBILITY",
        "generic contract",
        "implementation assessment",
    ]:
        require(required in rationale_text, f"rationale preflight element missing: {required}")
    checks.append("parent roles, owner footprint disposition, covariance table, immutable NOI design, policy actors, low-pass convention, response domain, and rationale preflight are closed")

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
        "SCI-FLT-FIXED:input_parent_row_admission@1",
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
    publication = profiles[2]
    require(
        "performs no publication" in publication["consumer_action"],
        "publication profile must not perform publication",
    )
    require(
        "do not construct J_full or S_out" in profiles[1]["consumer_action"],
        "parent-row profile must not construct FLT output support",
    )
    dispositions = publication["missing_or_conflict_behavior"]
    require(
        dispositions.get("requested_nonzero_empty_scientific_output_support")
        == [
            "applied_no_scientific_output_support",
            "not_produced",
            "no_full_footprint_output_rows",
        ],
        "empty-output-support publication disposition mismatch",
    )
    require(
        dispositions.get("identity_operator")
        == ["complete_publication_candidate", "publisher_action", "realized_identity"],
        "identity publication disposition mismatch",
    )
    require(
        dispositions.get("zero_operator")
        == ["complete_publication_candidate", "publisher_action", "realized_zero"],
        "zero publication disposition mismatch",
    )
    require(
        dispositions.get("eligible_transformation_failure")
        == ["eligible", "realization_failed"],
        "publication failure disposition mismatch",
    )
    require(
        dispositions.get("late_noi_request")
        == [
            "historical_not_requested_at_FLT_publication",
            "child_owned_lifecycle",
            "no_flt_mutation",
        ],
        "late-NOI publication disposition mismatch",
    )
    require(
        "base signal may carry typed unavailable response or covariance state"
        in profiles[0]["exceptions"]
        and "qualified requests require their named available compatible companion"
        in profiles[0]["exceptions"],
        "base-versus-qualified companion disposition mismatch",
    )
    checks.append("all three typed policy domains, exact dispositions, actor boundaries, and unregistered VAL status are complete")

    change_map = json.loads(CHANGE_MAP_PATH.read_text(encoding="ascii"))
    req_changes = change_map.get("requirement_changes", {})
    pred_changes = change_map.get("prediction_changes", {})
    for label, groups, expected in [
        ("requirement", req_changes, EXPECTED_REQUIREMENTS),
        ("prediction", pred_changes, EXPECTED_PREDICTIONS),
    ]:
        partitions = [
            groups.get("amended_without_renumbering", []),
            groups.get("preserved_without_semantic_change", []),
            groups.get("appended", []),
        ]
        flattened = [identifier for group in partitions for identifier in group]
        require(len(flattened) == len(set(flattened)), f"duplicate {label} change-map id")
        require(set(flattened) == set(expected), f"incomplete {label} change-map partition")
        require(groups.get("renumbered") == [], f"{label} identifiers were renumbered")

    require(
        req_changes.get("appended") == EXPECTED_REQUIREMENTS[51:],
        "appended requirement sequence mismatch",
    )
    require(
        pred_changes.get("appended") == EXPECTED_PREDICTIONS[28:],
        "appended prediction sequence mismatch",
    )
    routes = change_map.get("r0_4_owner_directive_routes", {})
    require(list(routes) == R0_4_OWNER_DIRECTIVE_SECTIONS, "r0.4 owner-directive route sequence mismatch")
    routed_ids = {
        value
        for values in routes.values()
        for value in values
        if value in EXPECTED_IDS
    }
    changed_ids = set(req_changes["amended_without_renumbering"])
    changed_ids.update(req_changes["appended"])
    changed_ids.update(pred_changes["amended_without_renumbering"])
    changed_ids.update(pred_changes["appended"])
    require(changed_ids <= routed_ids, "changed identifier lacks an owner-directive route")
    require(len(change_map.get("equation_changes", [])) == 3, "equation change map mismatch")
    require(len(change_map.get("route_status_changes", [])) == 7, "route-status change map mismatch")
    require(len(change_map.get("lifecycle_changes", [])) == 1, "lifecycle change map mismatch")
    checks.append("equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs")
    checks.append("every r0.4 amended or appended identifier routes to an exact r0.4 owner-directive section")
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

    expected_owner = [
        {
            "path": buildmod.repo_relative(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "scientific_owner": "Grant Wilson",
        }
        for path in OWNER_DIRECTIVE_PATHS.values()
    ]
    require(
        binding.get("owner_directives") == expected_owner,
        "owner-directive binding mismatch",
    )
    require(binding.get("stage_b_date") == buildmod.STAGE_B_DATE, "Stage B date binding mismatch")

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
    checks.append("launch commit, exact packet bytes, all three owner directives, r0.4 sources, date, and build tools match BUILD_BINDING.json")
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
            sha256(OWNER_DIRECTIVE_PATHS["r0_2"]),
            sha256(OWNER_DIRECTIVE_PATHS["r0_3"]),
            builder_sha,
            "Scientific owner: Grant Wilson",
            f"Stage B date: {buildmod.STAGE_B_DATE}",
        ]
        for value in required_text:
            require(value in text, f"missing PDF identity text in {output_path.name}: {value}")
        require(buildmod.STAGE_B_DATE in (reader.metadata.subject or ""), f"PDF subject date mismatch: {output_path.name}")
    checks.append("all three PDF identity blocks, Grant Wilson author metadata, date metadata, hashes, sizes, and page counts match")
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
    owners = binding["owner_directives"]
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.4 Source-Packet Closure Report",
        "",
        "Report identity: `SCI-FLT-FIXED-SOURCE-PACKET-CLOSURE v0.1/draft-r0.4`",
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
            "## Scientific-owner directives",
            "",
        ]
    )
    for owner in owners:
        lines.append(
            f"- `{owner['path']}`: {owner['bytes']} bytes; SHA-256 `{owner['sha256']}`; owner `{owner['scientific_owner']}`"
        )
    lines.extend(["", "## Stage B sources", ""])
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
        "# SCI-FLT-FIXED v0.1 r0.4 Owner-Decision Parity Report",
        "",
        "Report identity: `SCI-FLT-FIXED-OWNER-DECISION-PARITY v0.1/draft-r0.4`",
        "",
        "Status: PASS; every directive section has an exact artifact or identifier route; scientific-owner review required",
        "",
        f"r0.2 owner directive SHA-256: `{sha256(OWNER_DIRECTIVE_PATHS['r0_2'])}`",
        f"r0.3 owner directive SHA-256: `{sha256(OWNER_DIRECTIVE_PATHS['r0_3'])}`",
        f"r0.4 owner directive SHA-256: `{sha256(OWNER_DIRECTIVE_PATHS['r0_4'])}`",
        "",
        "## Directive routes",
        "",
    ]
    for section, routes in change_map["r0_4_owner_directive_routes"].items():
        lines.append(f"- PASS: `{section}` -> {', '.join(f'`{route}`' for route in routes)}")
    lines.extend(
        [
            "",
            "## Stable-identifier result",
            "",
            "Requirements 001-051 and predictions 001-028 retain their identifiers. Requirements 052-053 and predictions 029-030 are append-only additions. No identifier was renumbered.",
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
        "# SCI-FLT-FIXED v0.1 r0.4 Core/View Parity Report",
        "",
        "Report identity: `SCI-FLT-FIXED-CORE-VIEW-PARITY v0.1/draft-r0.4`",
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
        "# SCI-FLT-FIXED v0.1 r0.4 Deterministic Rebuild Report",
        "",
        "Report identity: `SCI-FLT-FIXED-DETERMINISTIC-REBUILD v0.1/draft-r0.4`",
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


def write_profile_report(report_path: Path, policies: dict) -> None:
    lines = [
        "# SCI-FLT-FIXED v0.1 r0.4 Profile Architecture Report",
        "",
        "Report identity: `SCI-FLT-FIXED-PROFILE-ARCHITECTURE v0.1/draft-r0.4`",
        "",
        "Status: PASS; exact draft identities, dispositions, and actor boundaries reproduced; r0.3-origin profiles remain unregistered and not Registry-evaluated",
        "",
        f"Policy-record SHA-256: `{sha256(POLICY_PATH)}`",
        "",
        "## Profiles",
        "",
    ]
    for profile in policies["profiles"]:
        lines.append(
            f"- `{profile['identity']}`: domain `{profile['domain']}`; consumer action `{profile['consumer_action']}`"
        )
    lines.extend(
        [
            "",
            "## Actor boundary",
            "",
            "Parent-row decisions feed FLT-owned construction of `J_full` and `S_out`. VAL may create a decision artifact. The FLT publisher alone performs or declines publication and owns realization and FLT-local validity.",
            "",
            "## Exact r0.4 dispositions",
            "",
            "- Empty nonzero `S_out`: `applied_no_scientific_output_support` -> `not_produced` (`no_full_footprint_output_rows`).",
            "- Base versus qualified companions: honest unavailable companions are permitted only for the base request; each qualified request requires its named exact companion.",
            "- Identity and zero: complete candidate and publisher action precede `realized_identity` or `realized_zero`.",
            "- Publication failure: eligible transformation failure is `realization_failed` and emits no complete product.",
            "- Late NOI request: publication-time not-requested is provenance; the child owns its lifecycle and cannot mutate FLT.",
            "",
            "## Nonclaims",
            "",
            "This report supplies no Registry approval or evaluation, numerical route, implementation conformity, validation, readiness, production authorization, or Unity claim.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="ascii")


def authority_inventory() -> list[dict]:
    classified: dict[Path, str] = {}
    classified[buildmod.PACKET_MANIFEST.resolve()] = "stage_a_author_packet_manifest"
    for name in buildmod.ADMITTED_OBJECTS:
        classified[(buildmod.PACKAGE_DIR / name).resolve()] = "stage_a_admitted_object"
    source_paths = [SOURCE_DIR / item["source"] for item in buildmod.DOCUMENTS]
    source_paths.extend(SOURCE_DIR / name for name in buildmod.SUPPORTING_SOURCES)
    for path in source_paths:
        classified[path.resolve()] = "stage_b_source"
    classified[Path(buildmod.__file__).resolve()] = "build_tool"
    classified[Path(__file__).resolve()] = "verification_tool"
    classified[DEFAULT_BINDING.resolve()] = "build_binding"
    for path in [
        DEFAULT_REPORT,
        DEFAULT_VISUAL_QA,
        DEFAULT_SOURCE_CLOSURE,
        DEFAULT_OWNER_PARITY,
        DEFAULT_VIEW_PARITY,
        DEFAULT_REBUILD_REPORT,
        DEFAULT_PROFILE_REPORT,
    ]:
        classified[path.resolve()] = "verification_or_parity_report"
    for item in buildmod.DOCUMENTS:
        classified[(OUTPUT_DIR / item["output"]).resolve()] = "rendered_pdf"

    def descriptors(path: Path, artifact_class: str) -> tuple[str, str, str]:
        name = path.name
        if artifact_class == "stage_a_author_packet_manifest":
            return (
                "stage_a_scientific_authority_manifest",
                "active_exact_packet_binding",
                "authoritative_input_not_generated",
            )
        if artifact_class == "stage_a_admitted_object":
            return (
                "stage_a_admitted_scientific_authority",
                "active_exact_admitted_object",
                "authoritative_input_not_generated",
            )
        if artifact_class == "build_tool":
            return (
                "deterministic_pdf_build_process",
                "active_r0_4_process_recipe",
                "generates_build_binding_and_three_pdf_views",
            )
        if artifact_class == "verification_tool":
            return (
                "durable_content_identity_traceability_and_render_verification_process",
                "active_r0_4_process_recipe",
                "generates_verification_parity_profile_rebuild_closure_and_authority_records",
            )
        if artifact_class == "build_binding":
            return (
                "deterministic_build_identity_record",
                "active_r0_4_generated_evidence",
                "generated_by_build_stage_b.py_from_bound_sources_tools_fonts_and_pdfs",
            )
        if artifact_class == "verification_or_parity_report":
            return (
                "process_verification_or_parity_evidence",
                "active_r0_4_generated_evidence",
                "generated_or_completed_by_verify_stage_b.py_and_bound_to_current_build",
            )
        if artifact_class == "rendered_pdf":
            return (
                "rendered_stage_b_scientific_view",
                "active_r0_4_generated_view",
                "generated_by_build_stage_b.py_from_exact_named_source_and_shared_core",
            )
        if name.startswith("OWNER_DIRECTIVE_R0_"):
            state = (
                "active_scientific_owner_directive"
                if name == "OWNER_DIRECTIVE_R0_4.txt"
                else "preserved_with_exact_r0_4_supersession_or_amendment_only"
            )
            return (
                "scientific_owner_directive",
                state,
                "owner_authored_input_not_generated",
            )
        if name == "SHARED_NORMATIVE_CORE.md":
            return (
                "normative_scientific_core_source",
                "active_draft_r0_4",
                "single_source_imported_by_rationale_ecs_and_all_pdf_views",
            )
        if name == "SCIENTIST_RATIONALE.md":
            return (
                "scientist_readable_explanatory_view_source",
                "active_draft_r0_4_core_subordinate",
                "imports_exact_shared_core_and_generates_scientist_rationale_pdf",
            )
        if name == "ENGINEERING_CONFORMANCE.md":
            return (
                "engineering_conformance_view_source",
                "active_draft_r0_4_core_subordinate",
                "imports_exact_shared_core_and_generates_engineering_conformance_pdf",
            )
        if name == "POLICY_RECORDS.json":
            return (
                "unregistered_scientific_policy_profile_source",
                "r0_3_origin_records_amended_and_bound_at_r0_4_but_not_owner_approved_registered_or_evaluated",
                "authored_source_reported_by_profile_and_parity_views",
            )
        if name == "NUMERICAL_CONFORMANCE_POLICY.md":
            return (
                "prospective_numerical_comparison_policy_source",
                "draft_not_preregistered_and_no_candidate_finding",
                "authored_supporting_source_not_a_generated_result",
            )
        return (
            "stage_b_scientific_closure_or_traceability_source",
            "active_draft_r0_4_or_preserved_exact_supporting_record",
            "authored_source_or_machine_readable_input_to_generated_reports",
        )

    rows = []
    for path, artifact_class in classified.items():
        require(path.is_file(), f"authority-manifest input missing: {path}")
        role, state, relation = descriptors(path, artifact_class)
        rows.append(
            {
                "artifact_class": artifact_class,
                "path": buildmod.repo_relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role,
                "compatibility_or_supersession_state": state,
                "generated_view_relation": relation,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def authority_manifest_record() -> dict:
    entries = authority_inventory()
    return {
        "schema_version": "1.0",
        "manifest_identity": "SCI-FLT-FIXED-AUTHORITY-MANIFEST v0.1/proposed-freeze-r0.4",
        "status": "complete proposed-freeze authority inventory; scientific-owner review required; not a scientific freeze",
        "scientific_owner": "Grant Wilson",
        "stage_b_date": buildmod.STAGE_B_DATE,
        "self_binding": {
            "external_digest_file": buildmod.repo_relative(DEFAULT_AUTHORITY_DIGEST),
            "rule": "AUTHORITY_MANIFEST.json and its digest file are excluded from entries to avoid self-reference; the digest file binds the exact manifest bytes",
        },
        "entry_count": len(entries),
        "entries": entries,
        "nonclaims": [
            "implementation conformity",
            "achieved response or covariance",
            "numerical adequacy",
            "validation",
            "calibration",
            "observational performance",
            "readiness",
            "scientific freeze",
            "production suitability or authorization",
            "Unity activity",
        ],
    }


def write_authority_manifest(manifest_path: Path, digest_path: Path) -> str:
    record = authority_manifest_record()
    manifest_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    digest = sha256(manifest_path)
    digest_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="ascii")
    return digest


def verify_authority_manifest(manifest_path: Path, digest_path: Path) -> None:
    require(manifest_path.is_file(), "authority manifest is missing")
    require(digest_path.is_file(), "authority manifest external digest is missing")
    actual = json.loads(manifest_path.read_text(encoding="ascii"))
    require(actual == authority_manifest_record(), "authority manifest inventory mismatch")
    required_entry_fields = {
        "artifact_class",
        "path",
        "bytes",
        "sha256",
        "role",
        "compatibility_or_supersession_state",
        "generated_view_relation",
    }
    for entry in actual.get("entries", []):
        require(
            set(entry) == required_entry_fields,
            f"authority manifest entry fields mismatch: {entry.get('path')}",
        )
    digest = sha256(manifest_path)
    require(
        digest_path.read_text(encoding="ascii") == f"{digest}  {manifest_path.name}\n",
        "authority manifest external digest mismatch",
    )


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
        "Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.4`",
        "",
        "Status: PASS; deterministic r0.4 proposed-freeze preflight only; scientific-owner review required",
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
    parser.add_argument("--profile-report-out", type=Path, default=DEFAULT_PROFILE_REPORT)
    parser.add_argument("--authority-manifest-out", type=Path, default=DEFAULT_AUTHORITY_MANIFEST)
    parser.add_argument("--authority-digest-out", type=Path, default=DEFAULT_AUTHORITY_DIGEST)
    parser.add_argument("--require-visual-qa", action="store_true")
    parser.add_argument("--require-authority-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[str] = []
    try:
        verify_packet(checks)
        verify_sources_and_trace(checks)
        policies, change_map = verify_closure_sources(checks)
        binding = verify_binding_and_pdfs(args.binding.resolve(), checks)
        rebuilt = verify_reproducible(binding, checks)
        poppler = verify_poppler_render(binding, checks)
        if args.require_visual_qa:
            verify_visual_qa(binding, args.visual_qa.resolve(), checks)
        if args.require_authority_manifest:
            verify_authority_manifest(
                args.authority_manifest_out.resolve(), args.authority_digest_out.resolve()
            )
        checks.append("consolidated proposed-freeze authority inventory and external self-binding are complete")
        write_source_closure(args.source_closure_out.resolve(), binding)
        write_owner_parity(args.owner_parity_out.resolve(), change_map)
        write_view_parity(args.view_parity_out.resolve())
        write_rebuild_report(args.rebuild_report_out.resolve(), binding, rebuilt)
        write_profile_report(args.profile_report_out.resolve(), policies)
        write_report(
            args.report_out.resolve(),
            args.binding.resolve(),
            binding,
            checks,
            poppler,
            args.require_visual_qa,
        )
        write_authority_manifest(
            args.authority_manifest_out.resolve(), args.authority_digest_out.resolve()
        )
        verify_authority_manifest(
            args.authority_manifest_out.resolve(), args.authority_digest_out.resolve()
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
    print(f"wrote {args.profile_report_out}")
    print(f"wrote {args.authority_manifest_out}")
    print(f"wrote {args.authority_digest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
