#!/usr/bin/env python3
"""Verify SCI-NOI Stage B r0.2 authority, traceability, PDFs, and determinism."""

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

import build_stage_b_r0_2_pdfs as builder


HERE = Path(__file__).resolve().parent
DOCUMENTS = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.2",
    "NORMATIVE_CORE.md": "SCI-NOI_NORMATIVE_CORE v0.1/draft-r0.2",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        "SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.2",
}
ALLOWED_AUTHORITIES = {
    "SCOPE", "COVER", "CORE1", "CORE2", "CONV", "TAX", "MAPB", "JINCB",
    "PTCB", "GRAPH", "DESIGN", "UNCT", "STDT", "VALP", "FILTER", "LIFE",
    "OWNER", "REG_BIND",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_modules_and_documents() -> tuple[set[str], set[str]]:
    builder.verify_bindings()
    binding = json.loads(builder.MODULE_BINDING.read_text(encoding="utf-8"))
    for item in binding["ordered_modules"]:
        require(sha256(HERE / item["path"]) == item["sha256"],
                f"module digest mismatch: {item['path']}")
    binding_marker = (
        "5c6f954b457a546abcf74a1dc6dae190f2b22ea43c14edf34d2e4d2a8a704268"
    )
    for name, identity in DOCUMENTS.items():
        text = (HERE / name).read_text(encoding="utf-8")
        require(f"Document identity: `{identity}`" in text, f"identity mismatch: {name}")
        require(binding_marker in text, f"binding digest absent: {name}")
        require("\t" not in text, f"tab found: {name}")
    requirements_text = (HERE / "normative/REQUIREMENTS.md").read_text(encoding="utf-8")
    predictions_text = (HERE / "normative/PREDICTIONS.md").read_text(encoding="utf-8")
    requirements = re.findall(r"^`(SCI-NOI-REQ-\d{3})`", requirements_text, re.MULTILINE)
    predictions = re.findall(r"^`(SCI-NOI-PRED-\d{3})`", predictions_text, re.MULTILINE)
    expected_req = {f"SCI-NOI-REQ-{i:03d}" for i in range(1, 43)}
    expected_pred = {f"SCI-NOI-PRED-{i:03d}" for i in range(1, 18)}
    require(set(requirements) == expected_req and len(requirements) == 42,
            "requirement set or uniqueness mismatch")
    require(set(predictions) == expected_pred and len(predictions) == 17,
            "prediction set or uniqueness mismatch")
    active_text = "\n".join(
        (HERE / path).read_text(encoding="utf-8")
        for path in builder.MODULE_ORDER
    )
    require(re.search(r"(?<!SCI-)NOI-REQ-\d{3}", active_text) is None,
            "unqualified active requirement ID")
    require(re.search(r"(?<!SCI-)NOI-PRED-\d{3}", active_text) is None,
            "unqualified active prediction ID")
    return expected_req, expected_pred


def verify_traceability(requirements: set[str], predictions: set[str]) -> None:
    rows = list(csv.DictReader((HERE / "REQUIREMENT_PREDICTION_TRACEABILITY.csv").open(
        newline="", encoding="utf-8"
    )))
    require(len(rows) == 42, "traceability row count mismatch")
    ids = [row["requirement_id"] for row in rows]
    require(set(ids) == requirements and len(ids) == len(set(ids)),
            "traceability requirement coverage mismatch")
    covered: set[str] = set()
    for row in rows:
        row_predictions = set(filter(None, row["prediction_ids"].split("|")))
        authorities = set(filter(None, row["authority_codes"].split("|")))
        require(bool(row_predictions) and row_predictions <= predictions,
                f"prediction trace mismatch: {row['requirement_id']}")
        require(bool(authorities) and authorities <= ALLOWED_AUTHORITIES,
                f"authority trace mismatch: {row['requirement_id']}")
        require(bool(row["engineering_section"] and row["verification_kind"]),
                f"missing engineering trace: {row['requirement_id']}")
        covered |= row_predictions
    require(covered == predictions, "prediction coverage mismatch")


def verify_targeted_reports() -> None:
    parity = (HERE / "OWNER_DECISION_PARITY_MATRIX.md").read_text(encoding="utf-8")
    for decision in ["ODQ-101", "ODQ-102A", "ODQ-102B", "ODQ-102C", "ODQ-102D",
                     "ODQ-103", "ODQ-104", "ODQ-105A", "ODQ-105B", "ODQ-106",
                     "ODQ-107", "ODQ-108", "ODQ-109", "ODQ-110A", "ODQ-110B",
                     "ODQ-110C", "ODQ-111"]:
        require(decision in parity, f"owner parity row absent: {decision}")
    questions = (HERE / "FINITE_DESIGN_OWNER_QUESTIONS.md").read_text(encoding="utf-8")
    for question in ["SCI-NOI-OWNER-Q-102D-01", "SCI-NOI-OWNER-Q-SCOPE-01",
                     "SCI-NOI-OWNER-Q-107-NAME-01"]:
        require(question in questions, f"owner question absent: {question}")
    equations = (HERE / "normative/EQUATIONS.md").read_text(encoding="utf-8")
    for exact in ["a_pi = G_pi gamma_i", "Q_p = sum_{i in C_p} a_pi",
                  "R_b^fixed = A_MAP,Pi D_epsilon_b H_PTC^fixed"]:
        require(exact in equations, f"canonical operator expression absent: {exact}")
    requirement_text = (HERE / "normative/REQUIREMENTS.md").read_text(encoding="utf-8")
    require(re.search(r"no\s+complement pair shall be forced", requirement_text) is not None,
            "complement occurrence boundary absent")


def verify_manifest_and_pdfs() -> dict:
    manifest = json.loads((HERE / "SOURCE_AND_BUILD_MANIFEST.json").read_text(
        encoding="utf-8"
    ))
    require(manifest["author_packet_manifest_sha256"] == builder.BASE.MANIFEST_SHA256,
            "author manifest binding mismatch")
    require(manifest["normative_module_binding_sha256"] == builder.MODULE_BINDING_SHA256,
            "normative binding mismatch")
    for source, digest in manifest["stage_b_source_sha256"].items():
        path = builder.STAGE_B / source[3:] if source.startswith("../") else HERE / source
        require(path.is_file() and sha256(path) == digest, f"source hash mismatch: {source}")
    for pdf_name, record in manifest["pdf_outputs"].items():
        path = HERE / pdf_name
        require(path.is_file() and sha256(path) == record["sha256"],
                f"PDF hash mismatch: {pdf_name}")
        reader = PdfReader(path)
        require(len(reader.pages) == record["pages"] and len(reader.pages) >= 3,
                f"PDF page count mismatch: {pdf_name}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        require("Implementation-blind Stage B draft" in text,
                f"PDF draft marker absent: {pdf_name}")
        require("SCI-NOI" in text, f"PDF identity absent: {pdf_name}")
        metadata = reader.metadata
        require(metadata.get("/Creator") == "SCI-NOI deterministic matplotlib builder",
                f"PDF creator mismatch: {pdf_name}")
    return manifest


def verify_determinism(manifest: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="sci-noi-stage-b-r02-") as tmp:
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(
            [os.environ.get("SCI_NOI_PYTHON", os.sys.executable),
             str(HERE / "build_stage_b_r0_2_pdfs.py"),
             "--output-dir", tmp, "--no-manifest"],
            cwd=HERE, env=env, check=True, capture_output=True, text=True,
        )
        for pdf_name, record in manifest["pdf_outputs"].items():
            require(sha256(Path(tmp) / pdf_name) == record["sha256"],
                    f"nondeterministic PDF bytes: {pdf_name}")


def main() -> None:
    requirements, predictions = verify_modules_and_documents()
    verify_traceability(requirements, predictions)
    verify_targeted_reports()
    manifest = verify_manifest_and_pdfs()
    verify_determinism(manifest)
    print(
        "SCI-NOI Stage B r0.2 verification passed: "
        f"{len(requirements)} requirements, {len(predictions)} predictions, "
        f"{len(manifest['pdf_outputs'])} deterministic PDFs"
    )


if __name__ == "__main__":
    main()
