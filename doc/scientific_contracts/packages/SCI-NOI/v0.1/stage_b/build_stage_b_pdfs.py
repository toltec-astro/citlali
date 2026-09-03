#!/usr/bin/env python3
"""Build deterministic SCI-NOI Stage B draft PDFs from package-local sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = HERE.parents[5]

MANIFEST = PACKAGE / "AUTHOR_PACKET_MANIFEST.md"
MANIFEST_SHA256 = "b6f8e7252e7f61f4506899cb3e8e26cf939887bb48464852713f8ce81ac77ca0"

LOCAL_AUTHOR_OBJECTS = {
    "SCOPE_BRIEF.md": "044abcd528b46d41338a01cc1613df45d1898f96cd2546d85234bf006669b403",
    "AUTHOR_SUPERSESSION_COVER.md": "07dd18d45adc08572480b3141aabf324e9a9834ab806ab4dc661cc210f7b47cf",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md": "aca85dd1d5ea5e6a3123e4e544b2e844b78a774929e6d34dc2f651c4735cd966",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md": "a6fd83a19f373c83d26355255aa6a8878ec7a4f37dd306351d7e36e57acfda46",
    "SCI-MAP_TO_SCI-NOI_BOUNDARY.md": "4273c5a75ff10d00506e5aa8732690cd3f398ff5afbaa561af8f1434ec467e29",
    "SCI-JINC_TO_SCI-NOI_BOUNDARY.md": "7bf0ff489957943cee5abcd581b6b6b1fea0840969d62ced4d73072cff8b51f8",
    "SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md": "0a6484058569930cee62e80e04ca2045c107fde67603f662473ae471406f905c",
    "NOI_GEN_PARENT_OPERATOR_GRAPH.md": "d7cacc667f479965ab7d1a7c3acb453f0195224bf0df1051b319d424ac04e5ac",
    "ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md": "f0927dad70bb8487cbdd798efb1c7d82f35755721cbcf49f24dcbd3c62e89b9f",
    "FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md": "cad960d438810942ee630b63dff05211ad831977d3fcd81fb8ab40042b820976",
    "STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md": "ff24c6946104d0653469fe0f2921efcda39fbbc2a431e673528ce622f388d28e",
    "SCI-NOI_VAL_PROFILE_DRAFTS.md": "c89883d8c20f72aea05f0ae62464daee3a3ee6e81543ff313888ac318a192d6b",
    "FILTER_AND_FRUIT_SCOPE.md": "08eba55f840e8f8aa265e1d2f1a981e16351a1c2460e74907cb4beb5ccb7df77",
    "PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md": "3d138d769c9629f93b0a493955d5a76f537d2d832c2b58de33fae595653e6212",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md": "272ac939b8a7109a123073b1a39fcdd7ac4129c603683ee81257b94ab2f55a0b",
}

GIT_AUTHOR_OBJECTS = {
    "SCI-NOI-001_INDEPENDENT_CORE_R3.tex": (
        "5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508",
        "doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex",
        "27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da",
    ),
    "SCI-NOI-002_INDEPENDENT_CORE.tex": (
        "4f1fec36f7802f3b5e8ac067377679946930983c",
        "doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex",
        "36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d",
    ),
}

SOURCE_TO_PDF = {
    "SCIENTIFIC_RATIONALE.md": "SCI-NOI_v0.1_STAGE_B_SCIENTIFIC_RATIONALE_DRAFT.pdf",
    "NORMATIVE_CORE.md": "SCI-NOI_v0.1_STAGE_B_NORMATIVE_CORE_DRAFT.pdf",
    "ENGINEERING_CONFORMANCE_SPECIFICATION.md": (
        "SCI-NOI_v0.1_STAGE_B_ENGINEERING_CONFORMANCE_DRAFT.pdf"
    ),
}

TRACEABILITY = HERE / "REQUIREMENT_PREDICTION_TRACEABILITY.csv"
BUILD_MANIFEST = HERE / "SOURCE_AND_BUILD_MANIFEST.json"

FIXED_DATE = datetime(2026, 8, 30, 0, 0, 0, tzinfo=timezone.utc)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def verify_author_packet() -> dict[str, str]:
    observed = sha256_path(MANIFEST)
    if observed != MANIFEST_SHA256:
        raise RuntimeError(f"author manifest hash mismatch: {observed}")
    result = {"AUTHOR_PACKET_MANIFEST.md": observed}
    for name, expected in LOCAL_AUTHOR_OBJECTS.items():
        value = sha256_path(PACKAGE / name)
        if value != expected:
            raise RuntimeError(f"author object hash mismatch: {name}: {value}")
        result[name] = value
    for name, (commit, path, expected) in GIT_AUTHOR_OBJECTS.items():
        data = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO,
            check=True,
            capture_output=True,
        ).stdout
        value = sha256_bytes(data)
        if value != expected:
            raise RuntimeError(f"author object hash mismatch: {name}: {value}")
        result[f"{commit}:{path}"] = value
    return result


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    return text


def append_traceability(source: str) -> str:
    rows = list(csv.DictReader(TRACEABILITY.open(newline="", encoding="utf-8")))
    lines = [source.rstrip(), "", "# Appendix A. Exact Traceability Rows", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['requirement_id']}",
                "",
                f"Predictions: {row['prediction_ids']}",
                f"Admitted authorities: {row['authority_codes']}",
                f"Engineering sections: {row['engineering_section']}",
                f"Review kind: {row['verification_kind']}",
                "",
            ]
        )
    return "\n".join(lines)


class MarkdownPdfRenderer:
    def __init__(self, output: Path, title: str, identity: str) -> None:
        self.output = output
        self.title = title
        self.identity = identity
        self.pdf = PdfPages(
            output,
            metadata={
                "Title": title,
                "Author": "SCI-NOI implementation-blind Stage B author",
                "Subject": "Draft scientific contract; no conformity or readiness claim",
                "Keywords": "SCI-NOI, Stage B, scientific contract, draft",
                "Creator": "SCI-NOI deterministic matplotlib builder",
                "Producer": "SCI-NOI deterministic matplotlib builder",
                "CreationDate": FIXED_DATE,
                "ModDate": FIXED_DATE,
            },
        )
        self.fig = None
        self.ax = None
        self.y = 0.0
        self.page = 0
        self.in_code = False

    def _new_page(self, section: str = "") -> None:
        if self.fig is not None:
            self._finish_page()
        self.page += 1
        self.fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_axis_off()
        self.ax.set_xlim(0, 8.5)
        self.ax.set_ylim(0, 11)
        self.ax.add_patch(
            Rectangle((0.64, 10.26), 7.22, 0.035, facecolor="#26707a", edgecolor="none")
        )
        self.ax.text(
            0.75,
            10.48,
            "SCI-NOI v0.1 | implementation-blind Stage B draft",
            fontsize=7.5,
            color="#31545a",
            family="DejaVu Sans",
            va="top",
        )
        if section:
            self.ax.text(
                7.75,
                10.48,
                section[:54],
                fontsize=7.2,
                color="#60777b",
                family="DejaVu Sans",
                ha="right",
                va="top",
            )
        self.y = 10.03

    def _finish_page(self) -> None:
        assert self.fig is not None and self.ax is not None
        self.ax.add_patch(
            Rectangle((0.64, 0.57), 7.22, 0.018, facecolor="#a9bfc2", edgecolor="none")
        )
        self.ax.text(
            0.75,
            0.38,
            self.identity,
            fontsize=7,
            color="#60777b",
            family="DejaVu Sans",
            va="bottom",
        )
        self.ax.text(
            7.75,
            0.38,
            str(self.page),
            fontsize=7.5,
            color="#31545a",
            family="DejaVu Sans",
            ha="right",
            va="bottom",
        )
        self.pdf.savefig(self.fig, dpi=144)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def _ensure(self, height: float, section: str = "") -> None:
        if self.fig is None:
            self._new_page(section)
        if self.y - height < 0.78:
            self._new_page(section)

    def _write_lines(
        self,
        lines: list[str],
        *,
        fontsize: float,
        leading: float,
        color: str = "#172326",
        family: str = "DejaVu Sans",
        weight: str = "normal",
        left: float = 0.82,
        section: str = "",
        background: bool = False,
    ) -> None:
        height = leading * len(lines) + 0.04
        self._ensure(height, section)
        assert self.ax is not None
        if background:
            self.ax.add_patch(
                Rectangle(
                    (left - 0.10, self.y - height + 0.02),
                    7.00 - (left - 0.82),
                    height,
                    facecolor="#f2f6f6",
                    edgecolor="#d4e0e1",
                    linewidth=0.5,
                )
            )
        for line in lines:
            self.ax.text(
                left,
                self.y,
                line,
                fontsize=fontsize,
                color=color,
                family=family,
                weight=weight,
                va="top",
            )
            self.y -= leading
        self.y -= 0.04

    def _cover(self) -> None:
        self.page = 1
        self.fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_axis_off()
        self.ax.set_xlim(0, 8.5)
        self.ax.set_ylim(0, 11)
        self.ax.add_patch(
            Rectangle((0, 0), 8.5, 11, facecolor="#f7faf9", edgecolor="none")
        )
        self.ax.add_patch(
            Rectangle((0.72, 1.02), 0.16, 8.96, facecolor="#26707a", edgecolor="none")
        )
        self.ax.text(
            1.22,
            9.68,
            "SCI-NOI v0.1",
            fontsize=14,
            color="#26707a",
            family="DejaVu Sans",
            weight="bold",
            va="top",
        )
        title_lines = textwrap.wrap(self.title, width=34)
        y = 8.95
        for line in title_lines:
            self.ax.text(
                1.22,
                y,
                line,
                fontsize=25,
                color="#17383e",
                family="DejaVu Sans",
                weight="bold",
                va="top",
            )
            y -= 0.47
        self.ax.text(
            1.22,
            y - 0.20,
            "Implementation-blind Stage B draft",
            fontsize=13,
            color="#476a70",
            family="DejaVu Sans",
            va="top",
        )
        self.ax.text(
            1.22,
            y - 0.66,
            self.identity,
            fontsize=9.5,
            color="#60777b",
            family="DejaVu Sans Mono",
            va="top",
        )
        warning = (
            "Not owner-accepted or frozen. No implementation conformity, validation, "
            "calibration, physical-noise, covariance-completeness, significance, "
            "performance, readiness, or production claim."
        )
        yy = 2.20
        self.ax.add_patch(
            Rectangle((1.18, 1.28), 6.32, 1.24, facecolor="#e8f1f1", edgecolor="#b7cbce")
        )
        for line in textwrap.wrap(warning, width=76):
            self.ax.text(
                1.40,
                yy,
                line,
                fontsize=10,
                color="#29484e",
                family="DejaVu Sans",
                va="top",
            )
            yy -= 0.20
        self.ax.text(
            1.22,
            0.72,
            "Exact author-packet binding: b6f8e725...77ca0 | 2026-08-30",
            fontsize=8,
            color="#60777b",
            family="DejaVu Sans Mono",
            va="bottom",
        )
        self.pdf.savefig(self.fig, dpi=144)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def render(self, markdown: str) -> int:
        self._cover()
        current_section = ""
        paragraph: list[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph
            if not paragraph:
                return
            text = clean_inline(" ".join(x.strip() for x in paragraph))
            wrapped = textwrap.wrap(
                text,
                width=100,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            self._write_lines(
                wrapped,
                fontsize=9.0,
                leading=0.165,
                section=current_section,
            )
            paragraph = []

        self._new_page()
        for raw in markdown.splitlines():
            line = raw.rstrip()
            if line.startswith("```"):
                flush_paragraph()
                self.in_code = not self.in_code
                self.y -= 0.03
                continue
            if self.in_code:
                wrapped = textwrap.wrap(
                    line,
                    width=92,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                self._write_lines(
                    wrapped,
                    fontsize=8.1,
                    leading=0.15,
                    family="DejaVu Sans Mono",
                    color="#23474d",
                    left=0.94,
                    section=current_section,
                    background=True,
                )
                continue
            if not line.strip():
                flush_paragraph()
                self.y -= 0.03
                continue
            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                text = clean_inline(heading.group(2))
                if level == 1 and self.page > 2:
                    self._new_page(text)
                current_section = text
                size = {1: 16.5, 2: 12.0, 3: 10.2}[level]
                lead = {1: 0.31, 2: 0.24, 3: 0.20}[level]
                color = {1: "#17383e", 2: "#26707a", 3: "#31545a"}[level]
                wrapped = textwrap.wrap(text, width={1: 55, 2: 76, 3: 86}[level])
                self._write_lines(
                    wrapped,
                    fontsize=size,
                    leading=lead,
                    color=color,
                    weight="bold",
                    section=current_section,
                )
                self.y -= 0.05
                continue
            bullet = re.match(r"^\s*(-|\d+\.)\s+(.*)$", line)
            if bullet:
                flush_paragraph()
                prefix, body = bullet.groups()
                marker = "-" if prefix == "-" else prefix
                wrapped = textwrap.wrap(
                    clean_inline(body),
                    width=94,
                    subsequent_indent="  ",
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                wrapped[0] = f"{marker} {wrapped[0]}"
                self._write_lines(
                    wrapped,
                    fontsize=8.9,
                    leading=0.16,
                    left=0.98,
                    section=current_section,
                )
                continue
            paragraph.append(line)
        flush_paragraph()
        if self.fig is not None:
            self._finish_page()
        self.pdf.close()
        return self.page


def build(output_dir: Path, write_manifest: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    author_hashes = verify_author_packet()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    stage_sources = {}
    outputs = {}
    for source_name, pdf_name in SOURCE_TO_PDF.items():
        source_path = HERE / source_name
        source = source_path.read_text(encoding="utf-8")
        if source_name == "ENGINEERING_CONFORMANCE_SPECIFICATION.md":
            source = append_traceability(source)
        title_match = re.search(r"^#\s+(.+)$", source, flags=re.MULTILINE)
        identity_match = re.search(r"^Document identity:\s+`([^`]+)`", source, flags=re.MULTILINE)
        if not title_match or not identity_match:
            raise RuntimeError(f"missing title or identity in {source_name}")
        output_path = output_dir / pdf_name
        pages = MarkdownPdfRenderer(
            output_path,
            title_match.group(1),
            identity_match.group(1),
        ).render(source)
        stage_sources[source_name] = sha256_path(source_path)
        outputs[pdf_name] = {"sha256": sha256_path(output_path), "pages": pages}
    stage_sources[TRACEABILITY.name] = sha256_path(TRACEABILITY)
    stage_sources[Path(__file__).name] = sha256_path(Path(__file__))
    result = {
        "manifest_identity": "SCI-NOI_STAGE_B_SOURCE_AND_BUILD_MANIFEST v0.1/draft-r0.1",
        "fixed_build_date": "2026-08-30T00:00:00Z",
        "author_packet_manifest_sha256": MANIFEST_SHA256,
        "author_object_sha256": author_hashes,
        "stage_b_source_sha256": dict(sorted(stage_sources.items())),
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
