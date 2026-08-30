#!/usr/bin/env python3
"""Build deterministic SCI-NOI Stage B r0.4 PDFs from byte-bound sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE_B = HERE.parent
PACKAGE = STAGE_B.parent
REPO = PACKAGE.parents[4]
R02_PATH = STAGE_B / "r0.2/build_stage_b_r0_2_pdfs.py"
SPEC = importlib.util.spec_from_file_location("sci_noi_r02_builder", R02_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load package-local r0.2 builder")
R02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R02)
BASE = R02.BASE

OWNER_SUPPLEMENT = STAGE_B / "r0.3/SCIENTIFIC_OWNER_DECISION_SUPPLEMENT.md"
OWNER_SUPPLEMENT_SHA256 = "4b622de96e72860e544bf3699e3a131bc9cca5d2be0b0fe5bb41096327e977e5"
OWNER_DIRECTIVE = HERE / "SCIENTIFIC_OWNER_DIRECTIVE.md"
OWNER_DIRECTIVE_SHA256 = "f155c1e70c1a8431bf4baf68f111960456ef24e9e77544837ff2efbecb86f423"
MODULE_BINDING = HERE / "NORMATIVE_MODULE_BINDING.json"
MODULE_BINDING_SHA256 = "5c6de3bd5180c9231c79cabe5f5918938571340a9f836f437962779e0410d55a"
PROPOSED_PROFILES = HERE / "PROPOSED_PROFILE_SUCCESSORS.md"
PROPOSED_PROFILES_SHA256 = "b0ca8c26da0f04d9acb26823f5dce207dcb1a766621cb40770f2482b9b15355c"
R02_BINDING_SHA256 = "5c6f954b457a546abcf74a1dc6dae190f2b22ea43c14edf34d2e4d2a8a704268"
MODULE_ORDER = [
    "normative/NOTATION.md", "normative/DEFINITIONS.md", "normative/EQUATIONS.md",
    "normative/ASSUMPTIONS.md", "normative/REQUIREMENTS.md",
    "normative/PREDICTIONS.md",
]
SOURCE_TO_PDF = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_v0.1_STAGE_B_SCIENTIFIC_RATIONALE_DRAFT_r0.4.pdf",
    "NORMATIVE_CORE.md": "SCI-NOI_v0.1_STAGE_B_NORMATIVE_CORE_DRAFT_r0.4.pdf",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        "SCI-NOI_v0.1_STAGE_B_ENGINEERING_CONFORMANCE_DRAFT_r0.4.pdf",
}
BUILD_MANIFEST = HERE / "SOURCE_AND_BUILD_MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bindings() -> tuple[dict[str, str], dict[str, str]]:
    author_hashes = BASE.verify_author_packet()
    if sha256(OWNER_SUPPLEMENT) != OWNER_SUPPLEMENT_SHA256:
        raise RuntimeError("owner supplement hash mismatch")
    if sha256(OWNER_DIRECTIVE) != OWNER_DIRECTIVE_SHA256:
        raise RuntimeError("owner directive hash mismatch")
    if sha256(PROPOSED_PROFILES) != PROPOSED_PROFILES_SHA256:
        raise RuntimeError("proposed profile-successor hash mismatch")
    if sha256(MODULE_BINDING) != MODULE_BINDING_SHA256:
        raise RuntimeError("r0.4 normative binding hash mismatch")
    binding = json.loads(MODULE_BINDING.read_text(encoding="utf-8"))
    if [row["path"] for row in binding["ordered_modules"]] != MODULE_ORDER:
        raise RuntimeError("r0.4 module order mismatch")
    for row in binding["ordered_modules"]:
        if sha256(HERE / row["path"]) != row["sha256"]:
            raise RuntimeError(f"r0.4 module hash mismatch: {row['path']}")
    if sha256(STAGE_B / "r0.2/NORMATIVE_MODULE_BINDING.json") != R02_BINDING_SHA256:
        raise RuntimeError("r0.2 normative binding hash mismatch")
    process: dict[str, str] = {}
    for path, expected in R02.PROCESS_BINDINGS.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"process binding hash mismatch: {path.name}")
        process[str(path.relative_to(REPO))] = observed
    return author_hashes, process


class OwnerRenderer(BASE.MarkdownPdfRenderer):
    def __init__(self, output: Path, title: str, identity: str) -> None:
        super().__init__(output, title, identity)
        self.pdf.infodict()["Author"] = "Grant Wilson"

    def _cover(self) -> None:
        self.page = 1
        self.fig = BASE.plt.figure(figsize=(8.5, 11), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_axis_off(); self.ax.set_xlim(0, 8.5); self.ax.set_ylim(0, 11)
        self.ax.add_patch(BASE.Rectangle((0, 0), 8.5, 11, facecolor="#f7faf9", edgecolor="none"))
        self.ax.add_patch(BASE.Rectangle((0.72, 1.02), 0.16, 8.96,
                                         facecolor="#26707a", edgecolor="none"))
        self.ax.text(1.22, 9.68, "SCI-NOI v0.1", fontsize=14, color="#26707a",
                     family="DejaVu Sans", weight="bold", va="top")
        y = 8.95
        for line in textwrap.wrap(self.title, width=34):
            self.ax.text(1.22, y, line, fontsize=25, color="#17383e",
                         family="DejaVu Sans", weight="bold", va="top")
            y -= 0.47
        self.ax.text(1.22, y - 0.20, "Implementation-blind Stage B draft",
                     fontsize=13, color="#476a70", family="DejaVu Sans", va="top")
        self.ax.text(1.22, y - 0.56, "Scientific owner: Grant Wilson",
                     fontsize=11, color="#31545a", family="DejaVu Sans", va="top")
        self.ax.text(1.22, y - 0.92, self.identity, fontsize=9.5, color="#60777b",
                     family="DejaVu Sans Mono", va="top")
        warning = ("Not frozen. No implementation conformity, validation, calibration, "
                   "physical-noise, covariance-completeness, significance, performance, "
                   "readiness, or production claim.")
        self.ax.add_patch(BASE.Rectangle((1.18, 1.28), 6.32, 1.24,
                                         facecolor="#e8f1f1", edgecolor="#b7cbce"))
        yy = 2.20
        for line in textwrap.wrap(warning, width=76):
            self.ax.text(1.40, yy, line, fontsize=10, color="#29484e",
                         family="DejaVu Sans", va="top"); yy -= 0.20
        self.ax.text(1.22, 0.72, "r0.18 packet b6f8e725...77ca0 | r0.4 owner f155c1e7...f423",
                     fontsize=8, color="#60777b", family="DejaVu Sans Mono", va="bottom")
        self.pdf.savefig(self.fig, dpi=144)
        BASE.plt.close(self.fig); self.fig = None; self.ax = None


def exact_pdf_source(name: str) -> str:
    source = (HERE / name).read_text(encoding="utf-8").rstrip() + "\n"
    if name == "NORMATIVE_CORE.md":
        source += "\n# Appendix A. Exact Bound Normative Modules\n\n"
        for module in MODULE_ORDER:
            source += (HERE / module).read_text(encoding="utf-8").rstrip() + "\n\n"
    if name == "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
        rows = list(csv.DictReader((HERE / "REQUIREMENT_PREDICTION_OWNER_TRACEABILITY.csv").open(
            newline="", encoding="utf-8"
        )))
        source += "\n# Appendix A. Exact Requirement/Prediction/Owner Traceability\n\n"
        for row in rows:
            source += (
                f"## {row['requirement_id']}\n\nPredictions: {row['prediction_ids']}  \n"
                f"Owner decisions: {row['owner_decision_ids']}  \n"
                f"Owner digest: {row['owner_source_sha256']}  \n"
                f"Approval: {row['approval_state']}  \n"
                f"Dependency: {row['source_profile_dependency']}  \n"
                f"Parity: {row['parity_result']}  \n"
                f"ECS section/review: {row['ecs_section']} / {row['verification_kind']}\n\n"
            )
    return R02.table_to_readable_markdown(source)


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.suffix != ".pdf" and path.name != BUILD_MANIFEST.name \
                and "__pycache__" not in path.parts:
            result[str(path.relative_to(HERE))] = sha256(path)
    for relative in ["r0.2/NORMATIVE_MODULE_BINDING.json",
                     "r0.2/normative/NOTATION.md", "r0.2/normative/DEFINITIONS.md",
                     "r0.2/normative/EQUATIONS.md", "r0.2/normative/ASSUMPTIONS.md",
                     "r0.2/normative/REQUIREMENTS.md", "r0.2/normative/PREDICTIONS.md",
                     "r0.2/build_stage_b_r0_2_pdfs.py", "build_stage_b_pdfs.py"]:
        result[f"../{relative}"] = sha256(STAGE_B / relative)
    return dict(sorted(result.items()))


def build(output_dir: Path, write_manifest: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    author_hashes, process_hashes = verify_bindings()
    BASE.plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.compression": 9,
                              "pdf.fonttype": 42, "axes.unicode_minus": False})
    outputs: dict[str, dict[str, object]] = {}
    for source_name, pdf_name in SOURCE_TO_PDF.items():
        source = exact_pdf_source(source_name)
        title = re.search(r"^#\s+(.+)$", source, re.MULTILINE)
        identity = re.search(r"^Document identity:\s+`([^`]+)`", source, re.MULTILINE)
        if title is None or identity is None:
            raise RuntimeError(f"missing title or identity: {source_name}")
        output = output_dir / pdf_name
        pages = OwnerRenderer(output, title.group(1), identity.group(1)).render(source)
        outputs[pdf_name] = {"pages": pages, "sha256": sha256(output)}
    result = {
        "manifest_identity": "SCI-NOI_STAGE_B_SOURCE_AND_BUILD_MANIFEST v0.1/draft-r0.4",
        "fixed_build_date": "2026-08-30T00:00:00Z",
        "scientific_owner": "Grant Wilson",
        "author_packet_manifest_sha256": BASE.MANIFEST_SHA256,
        "author_object_sha256": author_hashes,
        "owner_decision_supplement_sha256": OWNER_SUPPLEMENT_SHA256,
        "owner_directive_sha256": OWNER_DIRECTIVE_SHA256,
        "proposed_profile_successors_sha256": PROPOSED_PROFILES_SHA256,
        "r0_2_normative_binding_sha256": R02_BINDING_SHA256,
        "r0_4_normative_binding_sha256": MODULE_BINDING_SHA256,
        "process_binding_sha256": dict(sorted(process_hashes.items())),
        "stage_b_source_sha256": source_hashes(),
        "pdf_outputs": dict(sorted(outputs.items())),
    }
    if write_manifest:
        BUILD_MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir.resolve(), not args.no_manifest)["pdf_outputs"],
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1788048000")
    main()
