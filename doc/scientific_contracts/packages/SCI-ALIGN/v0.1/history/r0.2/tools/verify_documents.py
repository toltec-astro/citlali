#!/usr/bin/env python3
"""Build and verify the two SCI-ALIGN v0.1 document views.

This verifier checks document integrity and layout prerequisites.  It does not
perform scientific approval, implementation conformity, or observational
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PDF_DIR = ROOT / "pdf"

MODULES = [
    "notation",
    "definitions",
    "assumptions",
    "equations",
    "requirements",
    "edge_cases",
]
WRAPPERS = ["scientific-rationale", "engineering-conformance"]

BUNDLED_TOOLS = {
    "pdfinfo": Path(
        "/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/bin/override/pdfinfo"
    ),
    "pdftoppm": Path(
        "/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/bin/override/pdftoppm"
    ),
    "pdftotext": Path(
        "/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/native/poppler/poppler/bin/pdftotext"
    ),
}

TECTONIC_BUNDLE_HASH = (
    "6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c"
)
TECTONIC_CACHE_BUNDLE = (
    Path.home()
    / "Library"
    / "Caches"
    / "Tectonic"
    / "bundles"
    / "data"
    / TECTONIC_BUNDLE_HASH
)

EXPECTED_BOUNDARY_SHA256 = (
    "359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf"
)


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        candidate = BUNDLED_TOOLS.get(name)
        if candidate is not None and candidate.is_file():
            resolved = str(candidate)
    if resolved is None:
        raise VerificationError(f"required tool is unavailable: {name}")
    return resolved


def run(command: list[str], *, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "1787270400"
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.returncode != 0:
        rendered = " ".join(command)
        raise VerificationError(
            f"command failed ({process.returncode}): {rendered}\n{process.stdout}"
        )
    return process.stdout


def assert_sequence(label: str, actual: list[str], stop: int) -> None:
    expected = [f"{value:03d}" for value in range(1, stop + 1)]
    if actual != expected:
        raise VerificationError(
            f"{label} IDs are not exact sequential 001-{stop:03d}: {actual}"
        )


def verify_sources() -> list[Path]:
    paths: list[Path] = []
    for module in MODULES:
        path = SRC / "common" / f"{module}.tex"
        if not path.is_file():
            raise VerificationError(f"missing shared module: {path}")
        paths.append(path)

    rationale = SRC / "scientific-rationale.tex"
    engineering = SRC / "engineering-conformance.tex"
    for path in (rationale, engineering):
        if not path.is_file():
            raise VerificationError(f"missing document wrapper: {path}")
        paths.append(path)

    rationale_text = rationale.read_text(encoding="utf-8")
    if re.findall(r"\\input\{common/([^}]+)\}", rationale_text):
        raise VerificationError("scientific rationale must be a concise narrative, not the complete modules")
    if rationale_text.count(r"\begin{figure}") < 2:
        raise VerificationError("scientific rationale must contain at least two explanatory figures")
    if "origin: original / synthesized / unavailable" not in rationale_text:
        raise VerificationError("Figure 2 must show canonical origin states")
    if re.search(r"origin:[^\\\n]*interpolat", rationale_text, flags=re.IGNORECASE):
        raise VerificationError("Figure 2 must not classify interpolation as origin")
    if "method: exact / linear / circular / held / surrogate / none" not in rationale_text:
        raise VerificationError("Figure 2 must show interpolation classes on the method axis")
    rationale_sections = re.findall(r"\\section\{([^}]+)\}", rationale_text)
    if len(rationale_sections) != 9:
        raise VerificationError(
            f"scientific rationale must contain exactly 9 narrative sections: {rationale_sections}"
        )

    expected_inputs = [f"\\input{{common/{name}}}" for name in MODULES]
    found = re.findall(r"\\input\{common/([^}]+)\}", engineering.read_text(encoding="utf-8"))
    if found != MODULES:
        raise VerificationError(
            f"engineering-conformance.tex does not include the six modules exactly once and in order: {found}"
        )
    engineering_text = engineering.read_text(encoding="utf-8")
    for marker in expected_inputs:
        if engineering_text.count(marker) != 1:
            raise VerificationError(f"engineering-conformance.tex: expected exactly one {marker}")
    for path in (rationale, engineering):
        text = path.read_text(encoding="utf-8").lower()
        if "no conformity" not in text and "conformity" not in text:
            raise VerificationError(f"{path.name}: missing explicit conformity disclaimer")

    requirement_text = (SRC / "common" / "requirements.tex").read_text(
        encoding="utf-8"
    )
    prediction_text = (SRC / "common" / "edge_cases.tex").read_text(
        encoding="utf-8"
    )
    equation_text = (SRC / "common" / "equations.tex").read_text(
        encoding="utf-8"
    )
    assumption_text = (SRC / "common" / "assumptions.tex").read_text(
        encoding="utf-8"
    )
    assert_sequence("requirement", re.findall(r"\\Req\{(\d{3})\}", requirement_text), 55)
    assert_sequence(
        "prediction", re.findall(r"\\Prediction\{(\d{3})\}", prediction_text), 26
    )
    assert_sequence(
        "equation",
        re.findall(r"\\tag\{SCI-ALIGN-EQ-(\d{3})\}", equation_text),
        20,
    )
    assert_sequence(
        "assumption",
        re.findall(r"\\Assumption\{(\d{3})\}", assumption_text),
        12,
    )

    owner_register = ROOT / "OWNER_DECISION_REGISTER.md"
    crosswalk = ROOT / "CROSSWALK.md"
    author_decisions = ROOT / "AUTHOR_DRAFT_DECISIONS.md"
    additional_records = [
        ROOT / "NOTATION_AND_SYMBOL_CHANGE_MAP.md",
        ROOT / "REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md",
        ROOT / "AVAILABILITY_REGISTER.md",
        ROOT / "SOURCE_MANIFEST.md",
        ROOT / "PDF_VISUAL_QA_REPORT.md",
    ]
    for path in (owner_register, crosswalk, author_decisions, *additional_records):
        if not path.is_file():
            raise VerificationError(f"missing Stage B record: {path}")
        paths.append(path)
    paths.append(Path(__file__).resolve())

    owner_text = owner_register.read_text(encoding="utf-8")
    preserved_decisions = [
        "ALIGN-OD1--OD2",
        "ALIGN-SCOPE-D007--D008",
        "ALIGN-OD3",
        "ALIGN-OD4--OD6",
        "ALIGN-OD7",
        "ALIGN-C001",
        "ALIGN-SCOPE-D015--D016",
        "ALIGN-SCOPE-D017",
        "ALIGN-SCOPE-D018",
        "ALIGN-SCOPE-D019",
        "ALIGN-SCOPE-D005",
        "ALIGN-OD8",
    ]
    for identity in preserved_decisions:
        if identity not in owner_text:
            raise VerificationError(f"owner register lost decision identity: {identity}")
    for value in range(101, 111):
        identity = f"SCI-ALIGN-ODQ-{value}"
        if identity not in owner_text:
            raise VerificationError(f"owner register lost question identity: {identity}")

    crosswalk_text = crosswalk.read_text(encoding="utf-8")
    for module in MODULES:
        if f"src/common/{module}.tex" not in crosswalk_text:
            raise VerificationError(f"crosswalk does not name module: {module}")
    for marker in ("SCI-ALIGN-REQ-001", "SCI-ALIGN-REQ-055", "SCI-ALIGN-PRED-026"):
        if marker not in crosswalk_text:
            raise VerificationError(f"crosswalk missing terminal marker: {marker}")

    manifest = ROOT / "SOURCE_MANIFEST.md"
    manifest_text = manifest.read_text(encoding="utf-8")
    manifested_paths = [
        *(f"src/common/{module}.tex" for module in MODULES),
        "src/scientific-rationale.tex",
        "src/engineering-conformance.tex",
        "SCI-ALIGN_TO_SCI-AST_BOUNDARY.md",
        "CROSSWALK.md",
        "OWNER_DECISION_REGISTER.md",
        "AUTHOR_DRAFT_DECISIONS.md",
        "NOTATION_AND_SYMBOL_CHANGE_MAP.md",
        "REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md",
        "AVAILABILITY_REGISTER.md",
        "PDF_VISUAL_QA_REPORT.md",
        "tools/verify_documents.py",
        "pdf/scientific-rationale.pdf",
        "pdf/engineering-conformance.pdf",
    ]
    for relative in manifested_paths:
        line_match = re.search(
            rf"^\| `{re.escape(relative)}` \|.*`([0-9a-f]{{64}})` \|$",
            manifest_text,
            flags=re.MULTILINE,
        )
        if line_match is None:
            raise VerificationError(f"source manifest missing digest row: {relative}")
        actual = sha256(ROOT / relative)
        if line_match.group(1) != actual:
            raise VerificationError(
                f"source manifest digest mismatch for {relative}: "
                f"recorded {line_match.group(1)}, actual {actual}"
            )

    boundary = ROOT / "SCI-ALIGN_TO_SCI-AST_BOUNDARY.md"
    actual_boundary_sha256 = sha256(boundary)
    if actual_boundary_sha256 != EXPECTED_BOUNDARY_SHA256:
        raise VerificationError(
            "ALIGN-to-AST boundary digest changed: "
            f"expected {EXPECTED_BOUNDARY_SHA256}, got {actual_boundary_sha256}"
        )
    boundary_text = boundary.read_text(encoding="utf-8")
    if "SHA-256" in boundary_text or EXPECTED_BOUNDARY_SHA256 in boundary_text:
        raise VerificationError("ALIGN-to-AST boundary body must not contain a self-hash")
    if "SCI-ALIGN_TO_SCI-AST" not in boundary_text or "v0.1/r0.1" not in boundary_text:
        raise VerificationError("ALIGN-to-AST boundary lost exact profile identity")

    canonical_paths = [
        *(SRC / "common" / f"{module}.tex" for module in MODULES),
        rationale,
        engineering,
        boundary,
    ]
    canonical_text = "\n".join(path.read_text(encoding="utf-8") for path in canonical_paths)
    banned_patterns = {
        "old stable slot (o,n)": r"\(o,n\)|\(o,\s*n\)",
        "reference interface r=D": r"\br\s*=\s*D\b",
        "old corrected-time superscript": r"t_\{ik\}\^\{\(r\)\}",
        "old circular interval": r"\(-P/2,P/2\]",
        "raw x called Stokes-I": r"Stokes-I use only|Stokes-I detector support",
        "old product-state assignment": r"(?:^|\s)Q\s*=\s*\(",
        "generic field endpoint x_a/x_b": r"\bx_[ab]\b",
        "generic stacked bold x input": r"\\boldsymbol\s+x(?:\b|_)",
        "generic input covariance C_x": r"\bC_x\b|C_\{x(?:\\mid|\})",
        "generic capital-X input container": r"\bX_D\b|X_\{(?:Tv|Hv)\}",
    }
    for label, pattern in banned_patterns.items():
        if re.search(pattern, canonical_text):
            raise VerificationError(f"notation audit failed: {label}")

    required_semantics = [
        "(o,s)",
        r"i_{\rm ref}=D",
        r"\delta_{i\rightarrow\rm ref}",
        r"\mathcal S_{\rm ALIGN}",
        r"[-P/2,P/2)",
        r"v_a",
        r"\boldsymbol v",
        r"C_v",
    ]
    for marker in required_semantics:
        if marker not in canonical_text:
            raise VerificationError(f"canonical sources missing targeted notation: {marker}")

    return paths


def build_pdfs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sci-align-tex-") as temporary:
        build_dir = Path(temporary)
        local_bundle: Path | None = None
        if shutil.which("pdflatex") is None:
            if not TECTONIC_CACHE_BUNDLE.is_dir():
                raise VerificationError(
                    f"cached Tectonic bundle is unavailable: {TECTONIC_CACHE_BUNDLE}"
                )
            local_bundle = build_dir / "tectonic-bundle"
            shutil.copytree(TECTONIC_CACHE_BUNDLE, local_bundle)
            (local_bundle / "SHA256SUM").write_text(
                TECTONIC_BUNDLE_HASH + "\n", encoding="ascii"
            )
        for wrapper in WRAPPERS:
            source = SRC / f"{wrapper}.tex"
            pdflatex = shutil.which("pdflatex")
            if pdflatex is not None:
                command = [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-output-directory={build_dir}",
                    source.name,
                ]
                first_output = run(command, cwd=SRC)
                second_output = run(command, cwd=SRC)
                compiler_output = first_output + second_output
            else:
                tectonic = require_tool("tectonic")
                if local_bundle is None:
                    raise VerificationError("internal error: local Tectonic bundle not prepared")
                command = [
                    tectonic,
                    "--bundle",
                    str(local_bundle),
                    "--chatter",
                    "minimal",
                    "--reruns",
                    "1",
                    "-Z",
                    "deterministic-mode",
                    "--outdir",
                    str(build_dir),
                    source.name,
                ]
                compiler_output = run(command, cwd=SRC)
            if re.search(
                r"warning:|overfull|underfull|undefined (?:reference|citation)",
                compiler_output,
                flags=re.IGNORECASE,
            ):
                raise VerificationError(
                    f"{source.name}: compiler warning detected\n{compiler_output}"
                )
            built = build_dir / f"{wrapper}.pdf"
            if not built.is_file():
                raise VerificationError(f"pdflatex did not produce {built}")
            shutil.copyfile(built, PDF_DIR / built.name)


def page_count(pdf: Path) -> int:
    pdfinfo = require_tool("pdfinfo")
    metadata = run([pdfinfo, str(pdf)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", metadata, flags=re.MULTILINE)
    if match is None:
        raise VerificationError(f"could not determine page count: {pdf}")
    pages = int(match.group(1))
    if pages < 1:
        raise VerificationError(f"PDF has no pages: {pdf}")
    if "Page size:" not in metadata:
        raise VerificationError(f"PDF lacks page-size metadata: {pdf}")
    return pages


def verify_pdf(pdf: Path, render_root: Path | None) -> tuple[int, int]:
    pdftotext = require_tool("pdftotext")
    if not pdf.is_file():
        raise VerificationError(f"missing PDF: {pdf}")
    pages = page_count(pdf)
    extracted = run([pdftotext, "-layout", str(pdf), "-"])
    searchable = extracted.translate(
        str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
    )
    searchable = re.sub(r"\s+", " ", searchable)
    if pdf.stem == "engineering-conformance":
        required_text = [
            "Notation and identity",
            "Scientific definitions and ownership boundary",
            "Assumptions, conditional scope, and typed unavailability",
            "Canonical equations",
            "Normative requirements",
            "Edge cases and falsifiable predictions",
            "SCI-ALIGN-REQ-055",
            "SCI-ALIGN-PRED-026",
            "SCI-ALIGN_TO_SCI-AST",
        ]
        minimum_text = 20_000
    else:
        required_text = [
            "What ALIGN scientifically establishes",
            "Native event time versus nominal detector-reference slot",
            "Tune/readout before paired",
            "Field-specific mapping",
            "Observing-state fields versus pointing-correction records",
            "Physical scans versus processing and science windows",
            "What AST and RTC receive",
            "Owner decision register summary",
            "SCI-ALIGN_TO_SCI-AST",
        ]
        minimum_text = 8_000
    for marker in required_text:
        if marker not in searchable:
            raise VerificationError(f"{pdf.name}: extracted text missing {marker!r}")
    if len(extracted.strip()) < minimum_text:
        raise VerificationError(f"{pdf.name}: unexpectedly little extracted text")
    if pdf.stem == "scientific-rationale" and not 7 <= pages <= 9:
        raise VerificationError(
            f"scientific rationale must be 7-9 pages, got {pages}"
        )

    nonblank_pages = 0
    for page in range(1, pages + 1):
        text = run(
            [pdftotext, "-f", str(page), "-l", str(page), str(pdf), "-"]
        )
        if len(re.sub(r"\s+", "", text)) < 40:
            raise VerificationError(f"{pdf.name}: page {page} is blank or nearly blank")
        nonblank_pages += 1

    if render_root is not None:
        pdftoppm = require_tool("pdftoppm")
        destination = render_root / pdf.stem
        destination.mkdir(parents=True, exist_ok=True)
        prefix = destination / "page"
        run([pdftoppm, "-png", "-r", "144", str(pdf), str(prefix)])
        rendered = sorted(destination.glob("page-*.png"))
        if len(rendered) != pages:
            raise VerificationError(
                f"{pdf.name}: rendered {len(rendered)} pages, expected {pages}"
            )

    return pages, nonblank_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        action="store_true",
        help="build both PDFs with pdflatex before verification",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="render every PDF page with Poppler into this directory",
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="with --build, rebuild and require byte-identical PDFs",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        source_paths = verify_sources()
        if arguments.build:
            build_pdfs()
            if arguments.check_determinism:
                first_hashes = {
                    wrapper: sha256(PDF_DIR / f"{wrapper}.pdf")
                    for wrapper in WRAPPERS
                }
                build_pdfs()
                second_hashes = {
                    wrapper: sha256(PDF_DIR / f"{wrapper}.pdf")
                    for wrapper in WRAPPERS
                }
                if first_hashes != second_hashes:
                    raise VerificationError(
                        f"PDF build is not deterministic: {first_hashes} != {second_hashes}"
                    )
        summaries: list[tuple[Path, int, int]] = []
        for wrapper in WRAPPERS:
            pdf = PDF_DIR / f"{wrapper}.pdf"
            pages, nonblank = verify_pdf(pdf, arguments.render_dir)
            summaries.append((pdf, pages, nonblank))
    except VerificationError as error:
        print(f"VERIFY FAIL: {error}", file=sys.stderr)
        return 1

    print("VERIFY PASS: narrative/formal views, stable IDs, targeted notation, owner identities, and boundary digest")
    for path in source_paths:
        print(f"SHA256 {sha256(path)}  {path.relative_to(ROOT)}")
    for pdf, pages, nonblank in summaries:
        print(
            f"PDF {pdf.name}: pages={pages}, nonblank_pages={nonblank}, "
            f"sha256={sha256(pdf)}"
        )
    if arguments.render_dir is not None:
        print(f"RENDERED_ALL_PAGES {arguments.render_dir.resolve()}")
    if arguments.check_determinism:
        print("DETERMINISTIC_REBUILD byte-identical PDFs")
    print("No scientific approval, conformity, validation, freeze, readiness, or production claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
