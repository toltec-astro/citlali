#!/usr/bin/env python3
"""Verify SCI-NOI Stage B r0.4 authority, traceability, and deterministic PDFs."""

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

import build_stage_b_r0_4_pdfs as builder


HERE = Path(__file__).resolve().parent
DOCUMENTS = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.4",
    "NORMATIVE_CORE.md": "SCI-NOI_NORMATIVE_CORE v0.1/draft-r0.4",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        "SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def verify_authority_and_modules() -> tuple[set[str], set[str]]:
    builder.verify_bindings()
    binding = json.loads(builder.MODULE_BINDING.read_text(encoding="utf-8"))
    require(binding["authority_bindings"]["r0_3_owner_supplement_sha256"] ==
            builder.OWNER_SUPPLEMENT_SHA256, "owner supplement binding mismatch")
    require(binding["authority_bindings"]["r0_4_owner_directive_sha256"] ==
            builder.OWNER_DIRECTIVE_SHA256, "owner directive binding mismatch")
    for row in binding["ordered_modules"]:
        require(sha256(HERE / row["path"]) == row["sha256"],
                f"module hash mismatch: {row['path']}")
    for name, identity in DOCUMENTS.items():
        text = (HERE / name).read_text(encoding="utf-8")
        require(f"Document identity: `{identity}`" in text, f"identity mismatch: {name}")
        require("Scientific owner: Grant Wilson" in text, f"owner absent: {name}")
        require(builder.MODULE_BINDING_SHA256 in text, f"binding absent: {name}")
    req_text = (HERE / "normative/REQUIREMENTS.md").read_text(encoding="utf-8")
    pred_text = (HERE / "normative/PREDICTIONS.md").read_text(encoding="utf-8")
    req_list = re.findall(r"^`(SCI-NOI-REQ-\d{3})`", req_text, re.MULTILINE)
    pred_list = re.findall(r"^`(SCI-NOI-PRED-\d{3})`", pred_text, re.MULTILINE)
    req = {f"SCI-NOI-REQ-{i:03d}" for i in range(1, 48)}
    pred = {f"SCI-NOI-PRED-{i:03d}" for i in range(1, 23)}
    require(set(req_list) == req and len(req_list) == 47, "stable requirement set mismatch")
    require(set(pred_list) == pred and len(pred_list) == 22, "stable prediction set mismatch")
    active = "\n".join((HERE / p).read_text(encoding="utf-8") for p in builder.MODULE_ORDER)
    require(re.search(r"(?<!SCI-)NOI-REQ-\d{3}", active) is None,
            "unqualified active requirement ID")
    require(re.search(r"(?<!SCI-)NOI-PRED-\d{3}", active) is None,
            "unqualified active prediction ID")
    return req, pred


def verify_scientific_repairs() -> None:
    req = (HERE / "normative/REQUIREMENTS.md").read_text(encoding="utf-8")
    eq = (HERE / "normative/EQUATIONS.md").read_text(encoding="utf-8")
    definitions = (HERE / "normative/DEFINITIONS.md").read_text(encoding="utf-8")
    require("`N_requested = 0` shall mean only disabled/not requested" in req,
            "cardinality closure absent")
    require("`A_UNC = B_resolved`" in req and "shall not be admitted or renormalized" in req,
            "UNC membership closure absent")
    require("canonical reduced\narbitrary-precision rational pairs" in req,
            "exact rational authority absent")
    require("epsilon_b iid ~ Uniform(A)" in req and "domain separation" in req,
            "joint member law absent")
    require("shall not be inferred from member count" in req,
            "effective-information typing absent")
    require("delta zeta_fixed_scale" in req and "full STD response" in req,
            "STD response boundary absent")
    req13 = req[req.index("`SCI-NOI-REQ-013`"):req.index("`SCI-NOI-REQ-014`")]
    require("RNG" not in req13 and "retry cap" not in req13,
            "rejection-only field leaked into universal design")
    require("A_UNC=B_resolved" in eq and "1/N_resolved" in eq,
            "UNC equation identity absent")
    require("exact integer comparison" in eq, "exact comparison absent")
    require("delta Vhat_cond/(2 sigma_cond^3)" in eq, "full STD derivative absent")
    require("conditional_detector_sign_randomization_marginal_second_moment" in definitions,
            "primary UNC identity absent")
    require("Atomic UNC membership" in definitions, "atomic UNC definition absent")
    require("one independent observation-level" in req, "approved scope absent")


def verify_traceability(requirements: set[str], predictions: set[str]) -> None:
    rows = list(csv.DictReader((HERE / "REQUIREMENT_PREDICTION_OWNER_TRACEABILITY.csv").open(
        newline="", encoding="utf-8"
    )))
    require(len(rows) == 47, "traceability row count mismatch")
    ids = [row["requirement_id"] for row in rows]
    require(set(ids) == requirements and len(ids) == len(set(ids)),
            "traceability requirement coverage mismatch")
    covered: set[str] = set()
    for row in rows:
        row_pred = set(filter(None, row["prediction_ids"].split("|")))
        require(bool(row_pred) and row_pred <= predictions,
                f"prediction mismatch: {row['requirement_id']}")
        for field in ["owner_decision_ids", "owner_source_sha256", "approval_state",
                      "source_profile_dependency", "parity_result", "ecs_section",
                      "verification_kind"]:
            require(bool(row[field]), f"missing {field}: {row['requirement_id']}")
        for digest in row["owner_source_sha256"].split("|"):
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    f"invalid owner digest: {row['requirement_id']}")
        covered |= row_pred
    require(covered == predictions, "prediction trace coverage mismatch")


def verify_profile_blocks() -> None:
    source = (builder.PACKAGE / "SCI-NOI_VAL_PROFILE_DRAFTS.md").read_text(encoding="utf-8")
    table = (HERE / "PROFILE_AND_REGISTRY_SUCCESSOR_TABLE.md").read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^## `SCI-NOI:[^`]+`\n", source))
    end = source.index("## Supersession And Claim Boundary")
    expected = {
        "generation_input_admission@1": "373b8496f6553527da937ae44422734b00db16517964ca239dbbefba6f59c5b3",
        "uncertainty_member_admission@1": "85c6641060ebb21f4246995d8f471888992b2b29b013457254ba1cc5938cfaa3",
        "uncertainty_ensemble_admission@1": "61e3da9349f55110ee8cd46114c980dbda6b173cc24514e79404db3ede9d18d6",
        "standardization_admission@1": "f6979949b907ffebe97c1a807381398dfb342dafcfe3def477f1c08dbcd77021",
    }
    starts = [m.start() for m in matches] + [end]
    for match, start, stop in zip(matches, starts[:-1], starts[1:]):
        identity = re.search(r"SCI-NOI:([^`]+)", match.group()).group(1)
        digest = hashlib.sha256((source[start:stop].rstrip() + "\n").encode()).hexdigest()
        require(expected[identity] == digest and digest in table,
                f"profile block digest mismatch: {identity}")
    proposed = builder.PROPOSED_PROFILES.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^## (SCI-NOI:[^\n]+)\n", proposed))
    end = proposed.index("\nNo reciprocal-use successor")
    proposed_expected = {
        "SCI-NOI:generation_input_admission@2":
            "53361ec1df6006e8b0d80b6c59391165fd1df678a777811680d24856cc18a697",
        "SCI-NOI:uncertainty_member_admission@2":
            "554e608d3eb3f5a89a3bc9d07504f8f48673ed768ae85ad1970a59ff51a240b0",
        "SCI-NOI:uncertainty_ensemble_admission@2":
            "b19e0a19782cd42cb24203fd96fe67b89e014aa999c34decf49e09412f2d4e3c",
        "SCI-NOI:standardization_admission@2":
            "d825f97c3d89f6f6f73a83bdf2d3fc2034c949f881b29533e8eff8353df419b4",
    }
    starts = [m.start() for m in matches] + [end]
    for match, start, stop in zip(matches, starts[:-1], starts[1:]):
        identity = match.group(1)
        digest = hashlib.sha256((proposed[start:stop].rstrip() + "\n").encode()).hexdigest()
        require(proposed_expected[identity] == digest and digest in table,
                f"proposed profile block digest mismatch: {identity}")


def verify_freeze_manifest() -> None:
    freeze = json.loads((HERE / "PROPOSED_FREEZE_MANIFEST.json").read_text(
        encoding="utf-8"
    ))
    require(freeze["disposition"] == "scientifically freezeable conditional; not frozen",
            "freeze disposition mismatch")
    require(bool(freeze["missing_freeze_gates"]), "freeze gates absent")
    require(freeze["immutable_r0_18_manifest"]["sha256"] == builder.BASE.MANIFEST_SHA256,
            "freeze r0.18 manifest mismatch")
    require(freeze["normative_authority"]["binding"][1] ==
            builder.MODULE_BINDING_SHA256, "freeze module binding mismatch")
    for path, digest in freeze["normative_authority"]["ordered_modules"]:
        require(sha256(HERE / path) == digest, f"freeze module mismatch: {path}")
    require(freeze["profile_authority_and_proposals"]["proposed_r0_4_source"][1] ==
            builder.PROPOSED_PROFILES_SHA256, "freeze profile source mismatch")
    for path, digest in freeze["scientist_and_engineering_views"]:
        require(sha256(HERE / path) == digest, f"freeze view mismatch: {path}")
    for path, digest in freeze["completion_and_parity_records"]:
        require(sha256(HERE / path) == digest, f"freeze record mismatch: {path}")
    for path, digest, pages in freeze["deterministic_pdf_views"]:
        pdf = HERE / path
        require(sha256(pdf) == digest and len(PdfReader(pdf).pages) == pages,
                f"freeze PDF mismatch: {path}")


def verify_manifest_and_pdfs() -> dict:
    manifest = json.loads((HERE / "SOURCE_AND_BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["scientific_owner"] == "Grant Wilson", "build owner mismatch")
    require(manifest["owner_decision_supplement_sha256"] ==
            builder.OWNER_SUPPLEMENT_SHA256, "build owner binding mismatch")
    require(manifest["owner_directive_sha256"] == builder.OWNER_DIRECTIVE_SHA256,
            "build directive binding mismatch")
    require(manifest["proposed_profile_successors_sha256"] ==
            builder.PROPOSED_PROFILES_SHA256, "build profile binding mismatch")
    for source, digest in manifest["stage_b_source_sha256"].items():
        path = builder.STAGE_B / source[3:] if source.startswith("../") else HERE / source
        require(path.is_file() and sha256(path) == digest, f"source mismatch: {source}")
    for name, record in manifest["pdf_outputs"].items():
        path = HERE / name
        require(path.is_file() and sha256(path) == record["sha256"], f"PDF hash: {name}")
        reader = PdfReader(path)
        require(len(reader.pages) == record["pages"] and len(reader.pages) >= 3,
                f"PDF pages: {name}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        require("Scientific owner: Grant Wilson" in text, f"PDF owner text: {name}")
        require("SCI-NOI" in text and "Stage B draft" in text, f"PDF identity: {name}")
        require(reader.metadata.get("/Author") == "Grant Wilson", f"PDF author: {name}")
        require(reader.metadata.get("/Creator") == "SCI-NOI deterministic matplotlib builder",
                f"PDF creator: {name}")
    return manifest


def verify_determinism(manifest: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="sci-noi-r04-") as tmp:
        env = dict(os.environ); env.setdefault("MPLBACKEND", "Agg")
        subprocess.run([os.environ.get("SCI_NOI_PYTHON", os.sys.executable),
                        str(HERE / "build_stage_b_r0_4_pdfs.py"), "--output-dir", tmp,
                        "--no-manifest"], cwd=HERE, env=env, check=True,
                       capture_output=True, text=True)
        for name, record in manifest["pdf_outputs"].items():
            require(sha256(Path(tmp) / name) == record["sha256"],
                    f"nondeterministic PDF: {name}")


def main() -> None:
    requirements, predictions = verify_authority_and_modules()
    verify_scientific_repairs(); verify_traceability(requirements, predictions)
    verify_profile_blocks(); verify_freeze_manifest()
    manifest = verify_manifest_and_pdfs()
    verify_determinism(manifest)
    print(f"SCI-NOI Stage B r0.4 verification passed: {len(requirements)} requirements, "
          f"{len(predictions)} predictions, {len(manifest['pdf_outputs'])} deterministic PDFs")


if __name__ == "__main__":
    main()
