#!/usr/bin/env python3
"""Verify SCI-NOI Stage B sources, traceability, PDFs, and deterministic build."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

import build_stage_b_pdfs as builder


HERE = Path(__file__).resolve().parent

SOURCES = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.1",
    "NORMATIVE_CORE.md": "SCI-NOI_NORMATIVE_CORE v0.1/draft-r0.1",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md": (
        "SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.1"
    ),
}

ALLOWED_AUTHORITIES = {
    "SCOPE",
    "COVER",
    "CORE1",
    "CORE2",
    "CONV",
    "TAX",
    "MAPB",
    "JINCB",
    "PTCB",
    "GRAPH",
    "DESIGN",
    "UNCT",
    "STDT",
    "VALP",
    "FILTER",
    "LIFE",
    "OWNER",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sources() -> tuple[set[str], set[str]]:
    for name, identity in SOURCES.items():
        text = (HERE / name).read_text(encoding="utf-8")
        require(f"Document identity: `{identity}`" in text, f"identity mismatch: {name}")
        require("\t" not in text, f"tab found: {name}")
        require("\u2011" not in text and "\u2013" not in text and "\u2014" not in text,
                f"non-ASCII dash found: {name}")

    normative = (HERE / "NORMATIVE_CORE.md").read_text(encoding="utf-8")
    rationale = (HERE / "SCIENTIFIC_RATIONALE.md").read_text(encoding="utf-8")
    requirement_list = re.findall(r"^`(NOI-REQ-\d{3})`", normative, flags=re.MULTILINE)
    prediction_list = re.findall(r"^- `(NOI-PRED-\d{3})`", rationale, flags=re.MULTILINE)
    expected_requirements = {f"NOI-REQ-{index:03d}" for index in range(1, 38)}
    expected_predictions = {f"NOI-PRED-{index:03d}" for index in range(1, 16)}
    require(set(requirement_list) == expected_requirements, "requirement ID set mismatch")
    require(len(requirement_list) == 37, "duplicate normative requirement ID")
    require(set(prediction_list) == expected_predictions, "prediction ID set mismatch")
    require(len(prediction_list) == 15, "duplicate rationale prediction ID")
    return expected_requirements, expected_predictions


def verify_traceability(requirements: set[str], predictions: set[str]) -> None:
    path = HERE / "REQUIREMENT_PREDICTION_TRACEABILITY.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    require(len(rows) == len(requirements), "traceability row count mismatch")
    row_requirements = [row["requirement_id"] for row in rows]
    require(set(row_requirements) == requirements, "traceability requirement coverage mismatch")
    require(len(row_requirements) == len(set(row_requirements)), "duplicate traceability requirement")
    traced_predictions: set[str] = set()
    for row in rows:
        row_predictions = {item for item in row["prediction_ids"].split("|") if item}
        row_authorities = {item for item in row["authority_codes"].split("|") if item}
        require(row_predictions <= predictions, f"unknown prediction in {row['requirement_id']}")
        require(bool(row_predictions), f"missing prediction in {row['requirement_id']}")
        require(bool(row_authorities), f"missing authority in {row['requirement_id']}")
        require(row_authorities <= ALLOWED_AUTHORITIES,
                f"unlisted authority in {row['requirement_id']}")
        require(bool(row["engineering_section"]), f"missing section in {row['requirement_id']}")
        require(bool(row["verification_kind"]), f"missing review kind in {row['requirement_id']}")
        traced_predictions |= row_predictions
    require(traced_predictions == predictions, "traceability prediction coverage mismatch")


def verify_manifest_and_pdfs() -> dict:
    manifest_path = HERE / "SOURCE_AND_BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["author_packet_manifest_sha256"] == builder.MANIFEST_SHA256,
        "build manifest author binding mismatch",
    )
    builder.verify_author_packet()
    for source_name in list(SOURCES) + [
        "REQUIREMENT_PREDICTION_TRACEABILITY.csv",
        "build_stage_b_pdfs.py",
    ]:
        require(
            manifest["stage_b_source_sha256"][source_name] == sha256(HERE / source_name),
            f"stage source hash mismatch: {source_name}",
        )
    for pdf_name, record in manifest["pdf_outputs"].items():
        path = HERE / pdf_name
        require(path.is_file(), f"missing PDF: {pdf_name}")
        require(record["sha256"] == sha256(path), f"PDF hash mismatch: {pdf_name}")
        reader = PdfReader(path)
        require(len(reader.pages) == record["pages"], f"PDF page mismatch: {pdf_name}")
        require(len(reader.pages) >= 3, f"unexpectedly short PDF: {pdf_name}")
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        require("implementation-blind Stage B draft" in text, f"missing draft marker: {pdf_name}")
        require("SCI-NOI" in text, f"missing identity text: {pdf_name}")
    return manifest


def verify_determinism(manifest: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="sci-noi-stage-b-pdf-") as tmp:
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        result = subprocess.run(
            [
                os.environ.get("SCI_NOI_PYTHON", os.sys.executable),
                str(HERE / "build_stage_b_pdfs.py"),
                "--output-dir",
                tmp,
                "--no-manifest",
            ],
            cwd=HERE,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        require(bool(result.stdout.strip()), "deterministic rebuild produced no report")
        for pdf_name, record in manifest["pdf_outputs"].items():
            require(
                sha256(Path(tmp) / pdf_name) == record["sha256"],
                f"nondeterministic PDF bytes: {pdf_name}",
            )


def main() -> None:
    requirements, predictions = verify_sources()
    verify_traceability(requirements, predictions)
    manifest = verify_manifest_and_pdfs()
    verify_determinism(manifest)
    print(
        "SCI-NOI Stage B verification passed: "
        f"{len(requirements)} requirements, {len(predictions)} predictions, "
        f"{len(manifest['pdf_outputs'])} deterministic PDFs"
    )


if __name__ == "__main__":
    main()
