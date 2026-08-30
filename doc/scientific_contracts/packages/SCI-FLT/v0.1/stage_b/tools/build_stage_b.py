#!/usr/bin/env python3
"""Build the deterministic SCI-FLT-FIXED v0.1 Stage B r0.2 PDF set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from html import escape
from pathlib import Path

PDF_SITE_PACKAGES = os.environ.get("SCI_FLT_PDF_SITE_PACKAGES")
if PDF_SITE_PACKAGES:
    sys.path.append(PDF_SITE_PACKAGES)

import reportlab
from pypdf import PdfReader
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


STAGE_B_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = STAGE_B_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[7]
SOURCE_DIR = STAGE_B_DIR / "source"
DEFAULT_OUTPUT_DIR = STAGE_B_DIR / "output" / "pdf"
DEFAULT_BINDING = STAGE_B_DIR / "BUILD_BINDING.json"

PACKET_MANIFEST = PACKAGE_DIR / "AUTHOR_PACKET_MANIFEST.md"
PACKET_MANIFEST_SHA256 = (
    "7f2d03f182258ac9770f7dba869e9ae0b5018efdcdb93b18b299a9b9c6df1e4d"
)
STAGE_A_LAUNCH_COMMIT = "cd55752e716051383da54356833ef0fac20b083a"
OWNER_DIRECTIVE = SOURCE_DIR / "OWNER_DIRECTIVE_R0_2.txt"

ADMITTED_OBJECTS = {
    "SCOPE_BRIEF.md": "b66dbca45edc758e1fc29f9f14313deb52473527acec8ed4d8ce93e725e32468",
    "AUTHOR_SUPERSESSION_COVER.md": "68bb884513754375eba67d19881b092564e2c49a0345ca1b0ee2cbc9f0d55ef0",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md": "3daa40ca90fc290a91452d0493e92366ca784372daeeea96b92d2ddd602dc30f",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md": "118baee3555d7fb498c7f9a479a74a98b619e4eb6147cbe9c252bbd0f54831a2",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md": "214dad66b93d021a0c67014756dcf746e3bba46c624142298542330b5be9a659",
    "AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md": "8ec6bc2bf1c64a136203cbdc9db7c0ad2531bf1098022b7b813403d47ae9279f",
    "SCI-MAP_TO_SCI-FLT-FIXED_BOUNDARY.md": "2c04689734359a6fa8139b502a691238a118002ae27d8cd58fe82c3d0dddfbca",
    "SCI-JINC_TO_SCI-FLT-FIXED_BOUNDARY.md": "8c9cffe3641311ece334827136eafd47752a13750df1dcf2f55107ecc115892f",
    "SCI-FLT-FIXED_TO_SCI-NOI_BOUNDARY.md": "a349064e5bd0711eec54cd4f63ab02f934a60c2b1b6d5eccae0b64c02b47acd8",
    "FIXED_LINEAR_OPERATOR_AND_CONVOLUTION_SPECIFICATION.md": "91f9dd40d6784ff8544f88062645e022435cecac0319332a9a89f61365398ca6",
    "WCS_KERNEL_DISCRETIZATION_DECISION_TABLE.md": "d81b022f7a7feae8d465de3c09904222247a2b40debf6bb484e9825585cc9275",
    "EDGE_MISSING_NONFINITE_METHOD_DECISION_TABLE.md": "fa910834db9a3e9ee068f49d779cdc4a46ae4e7f67ef30e5a21bdb2f497f5ed4",
    "NORMALIZATION_UNIT_BEAM_DECISION_TABLE.md": "7b0090c4782abfe8d86dd15c286015aa9d8d9fd4e75a02cde97d0d99c017e828",
    "RESPONSE_NULLSPACE_COVARIANCE_PRODUCT_TABLE.md": "468d04343dfd715e97df001ca59db996a75f68c644506724d7220c6b86b3b991",
    "OBSERVATION_COADD_NONCOMMUTATION_TABLE.md": "4483fd81d690caf90f16953cd21fb1199cc68c562f42e354e95876e1521b816a",
    "SCI_FLT_VAL_PROFILE_DRAFTS.md": "9546082e4defcc3d83ce65969b44e35926fed7fe0185d306b9679701c0ce7976",
    "FIXED_PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md": "0c10c8e7cf46b80c94bc2454090ab918f7f440bcd0d4bc8b7e94cf53df66efed",
}

DOCUMENTS = [
    {
        "source": "SHARED_NORMATIVE_CORE.md",
        "output": "SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-draft-r0.2.pdf",
        "identity": "SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.2",
        "short": "SCI-FLT-FIXED Normative Core",
    },
    {
        "source": "SCIENTIST_RATIONALE.md",
        "output": "SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-draft-r0.2.pdf",
        "identity": "SCI-FLT-FIXED-SCIENTIST-RATIONALE v0.1/draft-r0.2",
        "short": "SCI-FLT-FIXED Scientist Rationale",
    },
    {
        "source": "ENGINEERING_CONFORMANCE.md",
        "output": "SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-draft-r0.2.pdf",
        "identity": "SCI-FLT-FIXED-ENGINEERING-CONFORMANCE v0.1/draft-r0.2",
        "short": "SCI-FLT-FIXED Engineering Conformance",
    },
]

SUPPORTING_SOURCES = [
    "TRACEABILITY.json",
    "FORMAL_CLOSURE_RECORD.md",
    "POLICY_RECORDS.json",
    "NUMERICAL_CONFORMANCE_POLICY.md",
    "SEMANTIC_CHANGE_MAP.json",
    "OWNER_DIRECTIVE_R0_2.txt",
]

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#197278")
PALE = colors.HexColor("#EAF2F4")
MID = colors.HexColor("#506575")
LIGHT = colors.HexColor("#D8E2E8")

FONT_FILES = {
    "StageBSans": Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
    "StageBSans-Bold": Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
    "StageBMono": Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "matplotlib"
    / "mpl-data"
    / "fonts"
    / "ttf"
    / "DejaVuSansMono.ttf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def verify_packet() -> None:
    if sha256(PACKET_MANIFEST) != PACKET_MANIFEST_SHA256:
        raise RuntimeError("author packet manifest SHA-256 mismatch")
    for name, expected in ADMITTED_OBJECTS.items():
        actual = sha256(PACKAGE_DIR / name)
        if actual != expected:
            raise RuntimeError(f"admitted object SHA-256 mismatch: {name}")


def ensure_ascii(path: Path) -> None:
    raw = path.read_bytes()
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"non-ASCII content in {repo_relative(path)}") from error


def register_fonts() -> None:
    for font_name, font_path in FONT_FILES.items():
        if not font_path.is_file():
            raise RuntimeError(f"required build font is unavailable: {font_path}")
        pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        "StageBSans",
        normal="StageBSans",
        bold="StageBSans-Bold",
        italic="StageBSans",
        boldItalic="StageBSans-Bold",
    )


def inline_markup(text: str) -> str:
    rendered = escape(text, quote=False)
    rendered = re.sub(
        r"`([^`]+)`",
        lambda match: f'<font name="StageBMono">{match.group(1)}</font>',
        rendered,
    )
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", rendered)
    return rendered


def make_styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="StageBSans-Bold",
            fontSize=23,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=18,
        ),
        "cover_identity": ParagraphStyle(
            "CoverIdentity",
            parent=base["BodyText"],
            fontName="StageBMono",
            fontSize=9,
            leading=13,
            textColor=MID,
            spaceAfter=8,
        ),
        "cover_status": ParagraphStyle(
            "CoverStatus",
            parent=base["BodyText"],
            fontName="StageBSans-Bold",
            fontSize=10,
            leading=14,
            textColor=TEAL,
            spaceAfter=18,
        ),
        "cover_owner": ParagraphStyle(
            "CoverOwner",
            parent=base["BodyText"],
            fontName="StageBSans-Bold",
            fontSize=9.2,
            leading=12.5,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "binding": ParagraphStyle(
            "Binding",
            parent=base["BodyText"],
            fontName="StageBMono",
            fontSize=7.2,
            leading=10,
            textColor=NAVY,
            leftIndent=8,
            rightIndent=8,
            borderColor=LIGHT,
            borderWidth=0.6,
            borderPadding=8,
            backColor=PALE,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="StageBSans-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="StageBSans-Bold",
            fontSize=12.5,
            leading=16,
            textColor=TEAL,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="StageBSans-Bold",
            fontSize=10.2,
            leading=13,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="StageBSans",
            fontSize=9.15,
            leading=12.7,
            textColor=colors.HexColor("#1F2A33"),
            alignment=TA_LEFT,
            spaceAfter=6.2,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="StageBSans",
            fontSize=8.95,
            leading=12.4,
            textColor=colors.HexColor("#1F2A33"),
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="StageBMono",
            fontSize=8.15,
            leading=11,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=10,
            borderColor=LIGHT,
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#F6F8F9"),
            spaceBefore=3,
            spaceAfter=8,
        ),
    }


class InvariantCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        kwargs["invariant"] = 1
        kwargs["pageCompression"] = 1
        super().__init__(*args, **kwargs)


def first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise RuntimeError("document has no level-one title")


def status_line(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("Status:"):
            return line.strip()
    raise RuntimeError("document has no status line")


def markdown_flowables(markdown: str, styles) -> list:
    lines = markdown.splitlines()
    story = []
    index = 0
    skipped_title = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index == len(lines):
                raise RuntimeError("unterminated fenced block")
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            index += 1
            continue
        if line.startswith("# "):
            if not skipped_title:
                skipped_title = True
            else:
                story.append(Paragraph(inline_markup(line[2:].strip()), styles["h1"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:].strip()), styles["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:].strip()), styles["h3"]))
            index += 1
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines):
                current = lines[index]
                if not current.startswith("- "):
                    break
                parts = [current[2:].strip()]
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if not continuation.strip():
                        break
                    if continuation.startswith(("- ", "#", "```")):
                        break
                    parts.append(continuation.strip())
                    index += 1
                items.append(" ".join(parts))
                while index < len(lines) and not lines[index].strip():
                    index += 1
            for item in items:
                story.append(
                    Paragraph(
                        '<font color="#197278">-</font>&nbsp;&nbsp;'
                        + inline_markup(item),
                        styles["bullet"],
                    )
                )
            story.append(Spacer(1, 4))
            continue
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if not next_line.strip():
                break
            if next_line.startswith(("#", "- ", "```")):
                break
            paragraph_lines.append(next_line.strip())
            index += 1
        story.append(
            Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"])
        )
    return story


def draw_page(canvas_obj, document, short_title: str) -> None:
    canvas_obj.saveState()
    width, height = letter
    canvas_obj.setStrokeColor(LIGHT)
    canvas_obj.setLineWidth(0.45)
    canvas_obj.line(0.72 * inch, height - 0.52 * inch, width - 0.72 * inch, height - 0.52 * inch)
    canvas_obj.setFont("StageBSans-Bold", 7.2)
    canvas_obj.setFillColor(NAVY)
    canvas_obj.drawString(0.72 * inch, height - 0.41 * inch, short_title)
    canvas_obj.setFont("StageBSans", 7.1)
    canvas_obj.setFillColor(MID)
    canvas_obj.drawRightString(width - 0.72 * inch, height - 0.41 * inch, "Stage B r0.2 - owner review required")
    canvas_obj.line(0.72 * inch, 0.48 * inch, width - 0.72 * inch, 0.48 * inch)
    canvas_obj.setFont("StageBSans", 7.2)
    canvas_obj.drawString(0.72 * inch, 0.33 * inch, "SCI-FLT-FIXED v0.1")
    canvas_obj.drawRightString(width - 0.72 * inch, 0.33 * inch, f"Page {document.page}")
    canvas_obj.restoreState()


def build_pdf(
    source_path: Path,
    output_path: Path,
    identity: str,
    short_title: str,
    source_sha: str,
    core_sha: str,
    builder_sha: str,
) -> None:
    markdown = source_path.read_text(encoding="ascii")
    expanded = markdown.replace("{{NORMATIVE_CORE_SHA256}}", core_sha)
    if "{{" in expanded or "}}" in expanded:
        raise RuntimeError(f"unresolved template token in {source_path.name}")

    styles = make_styles()
    title = first_heading(expanded)
    status = status_line(expanded)
    binding_lines = [
        f"Document identity: {identity}",
        f"Source SHA-256: {source_sha}",
        f"Shared normative core SHA-256: {core_sha}",
        f"Author packet manifest SHA-256: {PACKET_MANIFEST_SHA256}",
        f"Owner directive SHA-256: {sha256(OWNER_DIRECTIVE)}",
        f"Build recipe SHA-256: {builder_sha}",
    ]

    story = [
        Spacer(1, 0.58 * inch),
        Paragraph(inline_markup(title), styles["cover_title"]),
        Paragraph(inline_markup(identity), styles["cover_identity"]),
        Paragraph(inline_markup(status), styles["cover_status"]),
        Paragraph("Scientific owner: Grant Wilson", styles["cover_owner"]),
        Spacer(1, 0.18 * inch),
        Paragraph("<br/>".join(inline_markup(line) for line in binding_lines), styles["binding"]),
        Spacer(1, 0.14 * inch),
        Paragraph(
            "Implementation-blind scientific-contract draft. This artifact reports no implementation, validation, calibration, achieved-response, achieved-covariance, performance, readiness, freeze, production, or Unity result.",
            styles["body"],
        ),
        PageBreak(),
    ]
    story.extend(markdown_flowables(expanded, styles))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.63 * inch,
        title=identity,
        author="Grant Wilson",
        subject=f"{identity}; source {source_sha}; core {core_sha}",
        creator="SCI-FLT-FIXED deterministic Stage B builder",
        keywords="SCI-FLT-FIXED, scientific contract, Stage B draft r0.2",
    )
    document.build(
        story,
        onFirstPage=lambda c, d: draw_page(c, d, short_title),
        onLaterPages=lambda c, d: draw_page(c, d, short_title),
        canvasmaker=InvariantCanvas,
    )


def build(output_dir: Path, binding_path: Path) -> dict:
    verify_packet()
    register_fonts()
    source_paths = [SOURCE_DIR / item["source"] for item in DOCUMENTS]
    source_paths.extend(SOURCE_DIR / name for name in SUPPORTING_SOURCES)
    for path in source_paths:
        if path == OWNER_DIRECTIVE:
            continue
        ensure_ascii(path)

    builder_path = Path(__file__).resolve()
    verifier_path = builder_path.parent / "verify_stage_b.py"
    core_path = SOURCE_DIR / "SHARED_NORMATIVE_CORE.md"
    core_sha = sha256(core_path)
    builder_sha = sha256(builder_path)
    verifier_sha = sha256(verifier_path)

    outputs = []
    for item in DOCUMENTS:
        source_path = SOURCE_DIR / item["source"]
        source_sha = sha256(source_path)
        output_path = output_dir / item["output"]
        build_pdf(
            source_path,
            output_path,
            item["identity"],
            item["short"],
            source_sha,
            core_sha,
            builder_sha,
        )
        reader = PdfReader(str(output_path))
        outputs.append(
            {
                "document_identity": item["identity"],
                "filename": item["output"],
                "source": repo_relative(source_path),
                "source_sha256": source_sha,
                "shared_normative_core_sha256": core_sha,
                "sha256": sha256(output_path),
                "bytes": output_path.stat().st_size,
                "pages": len(reader.pages),
            }
        )

    binding = {
        "schema_version": "1.0",
        "record_identity": "SCI-FLT-FIXED-STAGE-B-BUILD-BINDING v0.1/draft-r0.2",
        "status": "deterministic Stage B r0.2 draft build record; scientific-owner review required; no implementation approval claim",
        "scientific_owner": "Grant Wilson",
        "stage_a_launch_commit": STAGE_A_LAUNCH_COMMIT,
        "packet": {
            "manifest": repo_relative(PACKET_MANIFEST),
            "manifest_identity": "SCI-FLT-FIXED_AUTHOR_PACKET v0.1/r0.1",
            "manifest_sha256": PACKET_MANIFEST_SHA256,
            "manifest_bytes": PACKET_MANIFEST.stat().st_size,
            "admitted_object_count": len(ADMITTED_OBJECTS),
            "admitted_objects": [
                {
                    "path": repo_relative(PACKAGE_DIR / name),
                    "sha256": digest,
                    "bytes": (PACKAGE_DIR / name).stat().st_size,
                }
                for name, digest in ADMITTED_OBJECTS.items()
            ],
        },
        "r0_2_owner_directive": {
            "path": repo_relative(OWNER_DIRECTIVE),
            "sha256": sha256(OWNER_DIRECTIVE),
            "bytes": OWNER_DIRECTIVE.stat().st_size,
            "scientific_owner": "Grant Wilson",
        },
        "sources": [
            {
                "path": repo_relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
        "build_tools": [
            {
                "path": repo_relative(builder_path),
                "sha256": builder_sha,
                "bytes": builder_path.stat().st_size,
            },
            {
                "path": repo_relative(verifier_path),
                "sha256": verifier_sha,
                "bytes": verifier_path.stat().st_size,
            },
        ],
        "embedded_fonts": [
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in FONT_FILES.items()
        ],
        "deterministic_build": {
            "python": sys.version.split()[0],
            "reportlab": reportlab.Version,
            "pypdf": __import__("pypdf").__version__,
            "pdf_site_packages": PDF_SITE_PACKAGES,
            "reportlab_invariant": True,
            "page_size": "US Letter",
            "pdf_count": len(DOCUMENTS),
        },
        "outputs": outputs,
        "nonclaims": [
            "implementation conformity",
            "algorithm change",
            "validation",
            "calibration",
            "achieved response or covariance",
            "performance",
            "readiness",
            "scientific freeze",
            "production",
            "Unity activity",
        ],
    }
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return binding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binding-out", type=Path, default=DEFAULT_BINDING)
    return parser.parse_args()


def main() -> int:
    rl_config.invariant = 1
    args = parse_args()
    binding = build(args.output_dir.resolve(), args.binding_out.resolve())
    for output in binding["outputs"]:
        print(
            f"built {output['filename']} pages={output['pages']} "
            f"sha256={output['sha256']}"
        )
    print(f"wrote {args.binding_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
