#!/usr/bin/env python3
"""Build deterministic SCI-NOI Stage B r0.2 PDFs from byte-bound sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE_B = HERE.parent
PACKAGE = STAGE_B.parent
REPO = PACKAGE.parents[4]

BASE_PATH = STAGE_B / "build_stage_b_pdfs.py"
SPEC = importlib.util.spec_from_file_location("sci_noi_stage_b_r01_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load package-local Stage B PDF renderer")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

MODULE_BINDING = HERE / "NORMATIVE_MODULE_BINDING.json"
MODULE_BINDING_SHA256 = "5c6f954b457a546abcf74a1dc6dae190f2b22ea43c14edf34d2e4d2a8a704268"

PROCESS_BINDINGS = {
    REPO / "doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER_NOI_STAGE_A_R0_18_2026-08-30.md":
        "04eca2da9ce76afacf18ae90dc2dbcb702fedbf55e03acb28e14e7dbc459a7c3",
    REPO / "doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md":
        "5994f4dff49dff3a9c9da6fbb494671b14a2f926f325f1c7c4a9603a6c2a38c1",
    PACKAGE / "SCI_VAL_REGISTRY_BINDING_2026-08-30.md":
        "739b5c7d7818a4292ae4b0beeab5a2d0356d77f0525bd0198e67181ae6d28a2e",
}

SOURCE_TO_PDF = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_v0.1_STAGE_B_SCIENTIFIC_RATIONALE_DRAFT_r0.2.pdf",
    "NORMATIVE_CORE.md": "SCI-NOI_v0.1_STAGE_B_NORMATIVE_CORE_DRAFT_r0.2.pdf",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        "SCI-NOI_v0.1_STAGE_B_ENGINEERING_CONFORMANCE_DRAFT_r0.2.pdf",
}

MODULE_ORDER = [
    "normative/NOTATION.md",
    "normative/DEFINITIONS.md",
    "normative/EQUATIONS.md",
    "normative/ASSUMPTIONS.md",
    "normative/REQUIREMENTS.md",
    "normative/PREDICTIONS.md",
]

BUILD_MANIFEST = HERE / "SOURCE_AND_BUILD_MANIFEST.json"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bindings() -> tuple[dict[str, str], dict[str, str]]:
    author_hashes = BASE.verify_author_packet()
    if sha256_path(MODULE_BINDING) != MODULE_BINDING_SHA256:
        raise RuntimeError("normative module binding digest mismatch")
    binding = json.loads(MODULE_BINDING.read_text(encoding="utf-8"))
    if [item["path"] for item in binding["ordered_modules"]] != MODULE_ORDER:
        raise RuntimeError("normative module order mismatch")
    for item in binding["ordered_modules"]:
        observed = sha256_path(HERE / item["path"])
        if observed != item["sha256"]:
            raise RuntimeError(f"normative module digest mismatch: {item['path']}")
    process_hashes: dict[str, str] = {}
    for path, expected in PROCESS_BINDINGS.items():
        observed = sha256_path(path)
        if observed != expected:
            raise RuntimeError(f"process binding digest mismatch: {path.name}")
        process_hashes[str(path.relative_to(REPO))] = observed
    return author_hashes, process_hashes


def table_to_readable_markdown(source: str) -> str:
    """Turn Markdown table rows into PDF-readable bullets without changing sources."""
    output: list[str] = []
    for line in source.splitlines():
        if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            output.append("- " + " | ".join(cells))
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def exact_pdf_source(source_name: str) -> str:
    source = (HERE / source_name).read_text(encoding="utf-8").rstrip() + "\n"
    if source_name == "NORMATIVE_CORE.md":
        source += "\n# Appendix A. Exact Bound Normative Modules\n\n"
        for module in MODULE_ORDER:
            source += (HERE / module).read_text(encoding="utf-8").rstrip() + "\n\n"
    if source_name == "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        rows = list(csv.DictReader((HERE / "REQUIREMENT_PREDICTION_TRACEABILITY.csv").open(
            newline="", encoding="utf-8"
        )))
        source += "\n# Appendix A. Exact Requirement/Prediction Traceability\n\n"
        for row in rows:
            source += (
                f"## {row['requirement_id']}\n\n"
                f"Predictions: {row['prediction_ids']}  \n"
                f"Authorities: {row['authority_codes']}  \n"
                f"ECS section: {row['engineering_section']}  \n"
                f"Review: {row['verification_kind']}\n\n"
            )
    return table_to_readable_markdown(source)


def stage_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pdf" or path.name == BUILD_MANIFEST.name:
            continue
        if "__pycache__" in path.parts:
            continue
        result[str(path.relative_to(HERE))] = sha256_path(path)
    result["../build_stage_b_pdfs.py"] = sha256_path(BASE_PATH)
    return result


def build(output_dir: Path, write_manifest: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    author_hashes, process_hashes = verify_bindings()
    BASE.plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    outputs: dict[str, dict[str, object]] = {}
    for source_name, pdf_name in SOURCE_TO_PDF.items():
        source = exact_pdf_source(source_name)
        title_match = re.search(r"^#\s+(.+)$", source, flags=re.MULTILINE)
        identity_match = re.search(
            r"^Document identity:\s+`([^`]+)`", source, flags=re.MULTILINE
        )
        if title_match is None or identity_match is None:
            raise RuntimeError(f"missing title or identity: {source_name}")
        output = output_dir / pdf_name
        pages = BASE.MarkdownPdfRenderer(
            output, title_match.group(1), identity_match.group(1)
        ).render(source)
        outputs[pdf_name] = {"pages": pages, "sha256": sha256_path(output)}
    result = {
        "manifest_identity": "SCI-NOI_STAGE_B_SOURCE_AND_BUILD_MANIFEST v0.1/draft-r0.2",
        "fixed_build_date": "2026-08-30T00:00:00Z",
        "author_packet_manifest_sha256": BASE.MANIFEST_SHA256,
        "author_object_sha256": author_hashes,
        "normative_module_binding_sha256": MODULE_BINDING_SHA256,
        "process_binding_sha256": dict(sorted(process_hashes.items())),
        "stage_b_source_sha256": stage_source_hashes(),
        "pdf_outputs": dict(sorted(outputs.items())),
    }
    if write_manifest:
        BUILD_MANIFEST.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args()
    result = build(args.output_dir.resolve(), not args.no_manifest)
    print(json.dumps(result["pdf_outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1788048000")
    main()
