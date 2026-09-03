#!/usr/bin/env python3
"""Mechanical verifier and canonical PDF builder for SCI-VAL v0.1/r0.3.

The verifier protects the original implementation-blind packet, the approved
r0.2/r0.3 revision authority, stable normative identity, Core/Registry and
aggregate-profile boundaries, dual-view genre split, and canonical PDF
artifacts. It inspects no Citlali implementation, schema, test, audit,
validation, Unity, or adjacent-package source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


PACKAGE = Path(__file__).resolve().parent.parent
SRC = PACKAGE / "src"
COMMON = SRC / "common"
PDF = PACKAGE / "pdf"

ORIGINAL_PACKET = {
    PACKAGE / "SCOPE_BRIEF.md":
        "98510ed385164ee2f3339284a3b15434da4821b85a43b19de1f9f691186594f9",
    PACKAGE / "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md":
        "32dc62160dff5dcb15e4af83d0df3311024494f30de075784603d4b4bfb4a52c",
    PACKAGE / "AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md":
        "7296112f48fd1edc8eb4b4527883aad86b3dbade19509ab8268e9c6f8b7e4964",
    PACKAGE / "DECISION_LOG.md":
        "29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60",
}

REVISION_AUTHORITY = {
    PACKAGE / "REVISION_DIRECTIVE_R0.2.md":
        "5b8f36288917bb12c342ada192d2dee0b87bb40f8f9868acdcc11eff489d8ef0",
    PACKAGE / "REVISION_DIRECTIVE_R0.3.md":
        "c33e07121dcb2979a28463eecbfe61025e4bd4b9c310b733f8f8e5ebe5c9da0e",
}

R03_FEEDBACK_SHA256 = (
    "9e04c73f8cad5731536720e741d78c53541fe8a378490e16d9aefda9a9c56635"
)

MODULES = (
    "notation",
    "definitions",
    "equations",
    "assumptions",
    "requirements",
    "edge_cases",
)

VIEWS = {
    "scientific-rationale.tex": "SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf",
    "engineering-conformance.tex": "SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf",
}

REQ_DECL = re.compile(r"\\ContractRequirement\{(\d{3})\}")
PRED_DECL = re.compile(r"\\ContractPrediction\{(\d{3})\}")
CROSSWALK_ROW = re.compile(
    r"^\|\s*(SCI-VAL-(?:REQ|PRED)-\d{3})\s*\|", re.MULTILINE
)


class VerificationError(RuntimeError):
    """A deterministic contract check failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_text(path: Path) -> str:
    require(path.is_file(), f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def verify_bound_authority() -> None:
    for path, expected in {**ORIGINAL_PACKET, **REVISION_AUTHORITY}.items():
        actual = sha256(path)
        require(
            actual == expected,
            f"authority hash mismatch for {path.name}: expected {expected}, "
            f"got {actual}",
        )

    directive = read_text(PACKAGE / "REVISION_DIRECTIVE_R0.3.md")
    require(R03_FEEDBACK_SHA256 in directive,
            "r0.3 directive is not bound to the approved feedback digest")

    registry = read_text(PACKAGE / "PROFILE_REGISTRY.md")
    require("SCI-VAL:independent_exposure@1" in registry,
            "mandatory canonical profile is absent from PROFILE_REGISTRY.md")
    require("Scientific owner | Grant Wilson" in registry,
            "canonical registry record lost its exact owner metadata")
    require("not registered" in registry and "has no compatibility alias" in registry,
            "former draft profile key must be explicitly unregistered with no alias")
    require("SCI-MAP:map_upstream_admission" in registry and
            "Unbound" in registry,
            "MAP reserved profile must remain explicitly unbound")
    for role in ("structural_gate", "required_permission",
                 "decisive_exclusion", "advisory"):
        require(role in registry, f"profile registry is missing role {role}")
    for marker in ("Aggregate Profile Rule", "atomic source-profile",
                   "cannot reuse the atomic profile identity"):
        require(marker in registry,
                f"profile registry is missing aggregate rule marker: {marker}")
    source_register = read_text(PACKAGE / "SOURCE_BINDING_REGISTER.md")
    for token in ("SCI-RTC", "SCI-CAL", "SCI-PTC", "SCI-MAP"):
        require(token in source_register,
                f"source-binding register is missing {token}")
    require("continuing r0.3 source-binding authority" in source_register,
            "source register is not marked as the continuing r0.3 authority")

    active_paths = [
        PACKAGE / "README.md",
        PACKAGE / "SOURCE_BINDING_REGISTER.md",
        *(COMMON / f"{name}.tex" for name in MODULES),
        *(SRC / name for name in VIEWS),
    ]
    active_corpus = "\n".join(read_text(path) for path in active_paths)
    require("VAL.core.independent_exposure@1" not in active_corpus and
            r"VAL.core.independent\_exposure@1" not in active_corpus,
            "former draft profile key remains in active contract content")

    ledger = read_text(PACKAGE / "SCIENTIFIC_OWNER_DECISION_LEDGER.md")
    qb_rows = [line for line in ledger.splitlines()
               if "SCI-VAL-OWNER-QB" in line]
    require(len(qb_rows) == 6, "owner ledger must contain six QB dispositions")
    require(all("OPEN" not in line for line in qb_rows),
            "a general SCI-VAL QB disposition remains open")
    require("ENGINEERING DEFERRED" in ledger and "PROFILE-LOCAL" in ledger,
            "ledger must preserve engineering-deferred and profile-local work")


def verify_view_architecture() -> None:
    engineering = read_text(SRC / "engineering-conformance.tex")
    expected = [f"common/{name}.tex" for name in MODULES]
    engineering_imports = re.findall(
        r"\\input\{(common/[^}]+\.tex)\}", engineering
    )
    require(
        engineering_imports == expected,
        "engineering view must import all six formal modules exactly once "
        f"and in canonical order; got {engineering_imports}",
    )
    require(not REQ_DECL.search(engineering) and not PRED_DECL.search(engineering),
            "engineering wrapper declares normative IDs outside formal modules")

    rationale = read_text(SRC / "scientific-rationale.tex")
    require(not re.search(r"\\input\{common/", rationale),
            "scientist-facing rationale must remain a standalone narrative")
    require(not REQ_DECL.search(rationale) and not PRED_DECL.search(rationale),
            "scientist-facing rationale declares normative IDs")
    require("Grant Wilson" not in rationale,
            "scientist-facing narrative hard-codes registry owner metadata")
    for marker in ("Hypothetical complete", "Would be eligible",
                   "own registered profile identity", "No general SCI-VAL"):
        require(marker in rationale,
                f"rationale is missing r0.3 narrative marker: {marker}")
    for label in (
        "sec:rat-worked",
        "sec:rat-authority",
        "sec:rat-axes",
        "sec:rat-influence",
        "sec:rat-aggregation",
        "sec:rat-profiles",
        "sec:rat-validation",
    ):
        require(f"\\label{{{label}}}" in rationale,
                f"rationale is missing crosswalk anchor {label}")
    for view_name in VIEWS:
        view = read_text(SRC / view_name)
        require("Producer Facts and Use-Specific Eligibility" in view and
                "Not Final Map Validity" in view,
                f"{view_name} is missing the approved clarifying subtitle")
    require("r0.3 registry snapshot" in rationale,
            "rationale source table is not labeled as an r0.3 snapshot")
    definitions = read_text(COMMON / "definitions.tex")
    require("r0.3 registry snapshot" in definitions and
            "continuing source-binding authority" in definitions,
            "engineering source table lacks snapshot/continuing-authority labels")

    equations = read_text(COMMON / "equations.tex")
    for marker in (r"\rho_\Gamma", "atomic source-profile", "exception cannot neutralize",
                   "structural\\_gate", "required\\_permission",
                   "decisive\\_exclusion", "advisory"):
        require(marker in equations,
                f"formal equations are missing r0.3 marker: {marker}")

    for module in MODULES:
        text = read_text(COMMON / f"{module}.tex")
        require(not re.search(r"\\input\{common/", text),
                f"formal module {module}.tex recursively imports shared science")


def declared_ids() -> tuple[list[str], list[str]]:
    corpus = "\n".join(read_text(COMMON / f"{name}.tex") for name in MODULES)
    req_numbers = REQ_DECL.findall(corpus)
    pred_numbers = PRED_DECL.findall(corpus)
    require(len(req_numbers) == len(set(req_numbers)), "duplicate requirement ID")
    require(len(pred_numbers) == len(set(pred_numbers)), "duplicate prediction ID")
    expected_req = [f"{number:03d}" for number in range(1, 50)]
    expected_pred = [f"{number:03d}" for number in range(1, 25)]
    require(req_numbers == expected_req,
            f"expected preserved+appended REQ-001--049, got {req_numbers}")
    require(pred_numbers == expected_pred,
            f"expected preserved+appended PRED-001--024, got {pred_numbers}")
    return (
        [f"SCI-VAL-REQ-{number}" for number in req_numbers],
        [f"SCI-VAL-PRED-{number}" for number in pred_numbers],
    )


def verify_crosswalk(req_ids: list[str], pred_ids: list[str]) -> None:
    rows = CROSSWALK_ROW.findall(read_text(PACKAGE / "CROSSWALK.md"))
    require(len(rows) == len(set(rows)), "duplicate identifier row in CROSSWALK.md")
    declared = set(req_ids + pred_ids)
    crossed = set(rows)
    require(
        declared == crossed,
        "crosswalk coverage mismatch; "
        f"missing={sorted(declared - crossed)}, extra={sorted(crossed - declared)}",
    )


def verify_authored_files() -> None:
    expected = [
        PACKAGE / "CROSSWALK.md",
        PACKAGE / "AUTHOR_DRAFT_DECISIONS.md",
        PACKAGE / "PROFILE_REGISTRY.md",
        PACKAGE / "SOURCE_BINDING_REGISTER.md",
        PACKAGE / "REVISION_DIRECTIVE_R0.2.md",
        PACKAGE / "REVISION_DIRECTIVE_R0.3.md",
        PACKAGE / "MANAGER_REVIEW_R0.3.md",
        SRC / "verify_contract.py",
        *(COMMON / f"{name}.tex" for name in MODULES),
        *(SRC / name for name in VIEWS),
    ]
    for path in expected:
        require(path.is_file(), f"missing authored artifact: {path}")


def run_command(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def build_pdfs() -> None:
    tectonic = shutil.which("tectonic")
    require(tectonic is not None, "tectonic is required for --build")
    PDF.mkdir(parents=True, exist_ok=True)
    build_env = os.environ.copy()
    build_env.setdefault("SOURCE_DATE_EPOCH", "1787198400")

    with tempfile.TemporaryDirectory(prefix=".sci-val-r03-build-", dir=PDF) as temp:
        outdir = Path(temp)
        for source_name, canonical_name in VIEWS.items():
            output = run_command(
                [tectonic, "--keep-logs", "--outdir", str(outdir), source_name],
                cwd=SRC,
                env=build_env,
            )
            built = outdir / source_name.replace(".tex", ".pdf")
            require(built.is_file(), f"tectonic did not produce {built}")
            log = outdir / source_name.replace(".tex", ".log")
            if log.is_file():
                log_text = log.read_text(encoding="utf-8", errors="replace").lower()
                require("undefined references" not in log_text,
                        f"undefined references in {log.name}")
                require("undefined citations" not in log_text,
                        f"undefined citations in {log.name}")
                require("overfull \\hbox" not in log_text,
                        f"overfull horizontal box in {log.name}")
            destination = PDF / canonical_name
            staged = PDF / f".{canonical_name}.new"
            shutil.copyfile(built, staged)
            os.replace(staged, destination)
            if output.strip():
                print(f"tectonic {source_name}: completed")


def pdf_page_count(pdf: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is not None:
        output = run_command([pdfinfo, str(pdf)], cwd=PACKAGE)
        for line in output.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
        raise VerificationError(f"pdfinfo did not report page count for {pdf}")
    from pypdf import PdfReader
    return len(PdfReader(str(pdf)).pages)


def pdf_text(pdf: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is not None:
        return run_command([pdftotext, str(pdf), "-"], cwd=PACKAGE)
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def verify_pdfs(req_ids: list[str], pred_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    rationale_path = PDF / VIEWS["scientific-rationale.tex"]
    engineering_path = PDF / VIEWS["engineering-conformance.tex"]
    for path in (rationale_path, engineering_path):
        require(path.is_file(), f"missing canonical PDF: {path}")
        text = pdf_text(path)
        require("??" not in text, f"{path.name} contains unresolved references")
        require("v0.1" in text and "r0.3" in text,
                f"{path.name} does not expose both version axes")
        counts[path.name] = pdf_page_count(path)

    rationale_text = pdf_text(rationale_path)
    require("SCI-VAL-REQ-" not in rationale_text and
            "SCI-VAL-PRED-" not in rationale_text,
            "scientist-facing rationale leaks formal stable identifiers")
    require(7 <= counts[rationale_path.name] <= 9,
            f"rationale must be 7--9 pages, got {counts[rationale_path.name]}")

    engineering_text = re.sub(r"SCI-V\s*AL-", "SCI-VAL-", pdf_text(engineering_path))
    missing = [identifier for identifier in req_ids + pred_ids
               if identifier not in engineering_text]
    require(not missing, f"engineering PDF is missing identifiers: {missing}")
    require(counts[engineering_path.name] > counts[rationale_path.name],
            "engineering specification must remain the complete longer formal view")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", action="store_true",
        help="build and atomically install canonical PDFs",
    )
    parser.add_argument(
        "--check-pdfs", action="store_true",
        help="verify existing canonical PDFs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_bound_authority()
        verify_authored_files()
        verify_view_architecture()
        req_ids, pred_ids = declared_ids()
        verify_crosswalk(req_ids, pred_ids)
        if args.build:
            build_pdfs()
        page_counts: dict[str, int] = {}
        if args.build or args.check_pdfs:
            page_counts = verify_pdfs(req_ids, pred_ids)
    except (VerificationError, ImportError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("OK: original packet hashes verified (4)")
    print("OK: r0.2/r0.3 revision authority hashes verified (2)")
    print("OK: canonical profile, aggregate schema, roles, and source bindings present")
    print(f"OK: sequential requirements={len(req_ids)}, predictions={len(pred_ids)}")
    print("OK: dual-view genre split and exact crosswalk coverage verified")
    for name, pages in page_counts.items():
        print(f"OK: {name}: {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
