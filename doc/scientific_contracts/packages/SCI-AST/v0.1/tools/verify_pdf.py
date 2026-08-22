#!/usr/bin/env python3
"""Durable structural verifier for the two SCI-AST Stage B r0.3 draft PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader


def dereference(value):
    return value.get_object() if hasattr(value, "get_object") else value


def font_is_embedded(font) -> bool:
    font = dereference(font)
    subtype = font.get("/Subtype")
    if subtype == "/Type0":
        descendants = dereference(font.get("/DescendantFonts", []))
        return bool(descendants) and all(font_is_embedded(item) for item in descendants)

    descriptor = font.get("/FontDescriptor")
    if descriptor is None:
        return False
    descriptor = dereference(descriptor)
    return any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))


def expected_ids(prefix: str, count: int) -> list[str]:
    return [f"SCI-AST-{prefix}-{number:03d}" for number in range(1, count + 1)]


def verify(pdf_path: Path) -> dict[str, object]:
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise ValueError(f"missing or empty PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path), strict=True)
    if not reader.pages:
        raise ValueError("PDF contains no pages")

    page_character_counts: list[int] = []
    all_text: list[str] = []
    fonts_seen: dict[str, bool] = {}

    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"page {page_number} has a non-positive media box")
        rotation = int(page.get("/Rotate", 0)) % 360
        if rotation not in (0, 90, 180, 270):
            raise ValueError(f"page {page_number} has invalid rotation {rotation}")

        text = page.extract_text() or ""
        visible_count = len("".join(text.split()))
        if visible_count < 40:
            raise ValueError(
                f"page {page_number} appears blank or unreadable ({visible_count} characters)"
            )
        page_character_counts.append(visible_count)
        all_text.append(text)

        resources = dereference(page.get("/Resources", {}))
        fonts = dereference(resources.get("/Font", {}))
        for font_name, font_reference in fonts.items():
            font = dereference(font_reference)
            base_name = str(font.get("/BaseFont", font_name))
            embedded = font_is_embedded(font)
            fonts_seen[base_name] = fonts_seen.get(base_name, True) and embedded

    if not fonts_seen:
        raise ValueError("PDF exposes no font resources")
    unembedded = sorted(name for name, embedded in fonts_seen.items() if not embedded)
    if unembedded:
        raise ValueError(f"unembedded fonts: {', '.join(unembedded)}")

    normalized_text = "".join("".join(all_text).split())
    document = pdf_path.stem
    counts: dict[str, int] = {}
    if document == "engineering-conformance":
        groups = {
            "requirements": expected_ids("REQ", 90),
            "predictions": expected_ids("PRED", 50),
            "assumptions": expected_ids("ASM", 15),
        }
        for group_name, identifiers in groups.items():
            missing = [identifier for identifier in identifiers if identifier not in normalized_text]
            duplicates = [
                identifier for identifier in identifiers if normalized_text.count(identifier) != 1
            ]
            if missing:
                raise ValueError(f"missing {group_name}: {', '.join(missing)}")
            if duplicates:
                raise ValueError(
                    f"non-unique {group_name} in extracted text: {', '.join(duplicates)}"
                )
            counts[group_name] = len(identifiers)
    elif document == "scientific-rationale":
        if not 8 <= len(reader.pages) <= 10:
            raise ValueError(
                f"scientist-facing rationale must be 8-10 pages, got {len(reader.pages)}"
            )
        required_phrases = (
            "SCI-ALIGN_TO_SCI-ASTv0.1/r0.1",
            "Theordered,noncommutingASTchain",
            "Pointingtransferrequiresthesamerealization",
            "ALIGN-gridcoordinates,RTC-output-gridcoordinates,andMAPdeposition",
            "Implementationconformity",
        )
        missing_phrases = [phrase for phrase in required_phrases if phrase not in normalized_text]
        if missing_phrases:
            raise ValueError(f"missing rationale content: {', '.join(missing_phrases)}")
        counts["figures"] = 3
    else:
        raise ValueError(f"unrecognized SCI-AST PDF role: {document}")

    for phrase in (
        "ordinarynonpolarimetriccoordinatepath",
        "Stokesreconstruction",
    ):
        if phrase not in normalized_text:
            raise ValueError(f"missing ordinary-path scope phrase: {phrase}")

    return {
        "pdf": str(pdf_path),
        "pages": len(reader.pages),
        "minimum_page_characters": min(page_character_counts),
        "maximum_page_characters": max(page_character_counts),
        "fonts": sorted(fonts_seen),
        "all_fonts_embedded": True,
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.pdf), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
