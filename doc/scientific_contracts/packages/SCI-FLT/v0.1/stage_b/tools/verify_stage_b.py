#!/usr/bin/env python3
"""Verify SCI-FLT-FIXED v0.1 Stage B content, identity, and traceability."""

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

CORE_PATH = SOURCE_DIR / "SHARED_NORMATIVE_CORE.md"
RATIONALE_PATH = SOURCE_DIR / "SCIENTIST_RATIONALE.md"
CONFORMANCE_PATH = SOURCE_DIR / "ENGINEERING_CONFORMANCE.md"
TRACE_PATH = SOURCE_DIR / "TRACEABILITY.json"

EXPECTED_REQUIREMENTS = [
    f"SCI-FLT-FIXED-REQ-{number:03d}" for number in range(1, 37)
]
EXPECTED_PREDICTIONS = [
    f"SCI-FLT-FIXED-PRED-{number:03d}" for number in range(1, 22)
]
EXPECTED_IDS = EXPECTED_REQUIREMENTS + EXPECTED_PREDICTIONS


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
    for path in [CORE_PATH, RATIONALE_PATH, CONFORMANCE_PATH, TRACE_PATH]:
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError as error:
            raise VerificationError(f"non-ASCII Stage B source: {path.name}") from error
    checks.append("all Stage B authored sources are ASCII-clean")

    requirements = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-REQ")
    predictions = exact_heading_ids(CORE_PATH, "SCI-FLT-FIXED-PRED")
    require(requirements == EXPECTED_REQUIREMENTS, "requirement heading sequence mismatch")
    require(predictions == EXPECTED_PREDICTIONS, "prediction heading sequence mismatch")
    checks.append("36 stable requirement identifiers are complete, unique, and ordered")
    checks.append("21 stable prediction identifiers are complete, unique, and ordered")

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
    checks.append("traceability covers every identifier and only admitted Stage A objects")
    checks.append("every traced core, rationale, conformance, and Stage A section resolves")

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
    require(packet.get("admitted_object_count") == 17, "binding admitted-object count mismatch")
    expected_packet_rows = [
        {
            "path": buildmod.repo_relative(buildmod.PACKAGE_DIR / name),
            "sha256": digest,
        }
        for name, digest in buildmod.ADMITTED_OBJECTS.items()
    ]
    require(packet.get("admitted_objects") == expected_packet_rows, "binding packet rows mismatch")

    source_paths = [
        SOURCE_DIR / item["source"] for item in buildmod.DOCUMENTS
    ] + [TRACE_PATH]
    expected_sources = [
        {"path": buildmod.repo_relative(path), "sha256": sha256(path)}
        for path in source_paths
    ]
    require(binding.get("sources") == expected_sources, "Stage B source binding mismatch")

    builder_path = Path(buildmod.__file__).resolve()
    verifier_path = Path(__file__).resolve()
    expected_tools = [
        {"path": buildmod.repo_relative(builder_path), "sha256": sha256(builder_path)},
        {"path": buildmod.repo_relative(verifier_path), "sha256": sha256(verifier_path)},
    ]
    require(binding.get("build_tools") == expected_tools, "build-tool binding mismatch")
    expected_fonts = [
        {"name": name, "path": str(path), "sha256": sha256(path)}
        for name, path in buildmod.FONT_FILES.items()
    ]
    require(binding.get("embedded_fonts") == expected_fonts, "embedded-font binding mismatch")
    checks.append("launch commit, packet, sources, and build tools match BUILD_BINDING.json")
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
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for page_number, page in enumerate(reader.pages, 1):
            require((page.extract_text() or "").strip(), f"blank PDF page: {output_path.name} page {page_number}")
        required_text = [
            item["identity"],
            record["source_sha256"],
            core_sha,
            buildmod.PACKET_MANIFEST_SHA256,
            builder_sha,
        ]
        for value in required_text:
            require(value in text, f"missing PDF identity text in {output_path.name}: {value}")
    checks.append("all three PDF identity blocks, metadata, hashes, sizes, and page counts match")
    checks.append("every PDF page contains extractable text")
    return binding


def verify_reproducible(binding: dict, checks: list[str]) -> None:
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
        "Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.1`",
        "",
        "Status: PASS; deterministic draft verification only; scientific-owner review required",
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
    parser.add_argument("--require-visual-qa", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[str] = []
    try:
        verify_packet(checks)
        verify_sources_and_trace(checks)
        binding = verify_binding_and_pdfs(args.binding.resolve(), checks)
        verify_reproducible(binding, checks)
        poppler = verify_poppler_render(binding, checks)
        if args.require_visual_qa:
            verify_visual_qa(binding, args.visual_qa.resolve(), checks)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
