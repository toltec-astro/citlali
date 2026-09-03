#!/usr/bin/env python3
"""Assemble labeled four-page contact sheets from Poppler page renders."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAGE_PATTERN = re.compile(r"page-(\d+)\.png$")


def page_number(path: Path) -> int:
    match = PAGE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"unrecognized page filename: {path}")
    return int(match.group(1))


def make_sheets(render_dir: Path, output_dir: Path, document: str) -> list[Path]:
    pages = sorted((render_dir / document).glob("page-*.png"), key=page_number)
    if not pages:
        raise ValueError(f"no rendered pages for {document}")

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=20)
    cell_width = 760
    label_height = 34
    gutter = 18
    sheet_paths: list[Path] = []

    for group_start in range(0, len(pages), 4):
        group = pages[group_start : group_start + 4]
        prepared: list[tuple[Image.Image, str]] = []
        max_cell_height = 0
        for path in group:
            with Image.open(path) as source:
                page = source.convert("RGB")
            scale = cell_width / page.width
            page = page.resize((cell_width, round(page.height * scale)), Image.Resampling.LANCZOS)
            label = f"{document} - page {page_number(path)}"
            prepared.append((page, label))
            max_cell_height = max(max_cell_height, label_height + page.height)

        sheet_width = 2 * cell_width + 3 * gutter
        sheet_height = 2 * max_cell_height + 3 * gutter
        sheet = Image.new("RGB", (sheet_width, sheet_height), "#d9d9d9")
        draw = ImageDraw.Draw(sheet)
        for position, (page, label) in enumerate(prepared):
            row, column = divmod(position, 2)
            x = gutter + column * (cell_width + gutter)
            y = gutter + row * (max_cell_height + gutter)
            draw.rectangle((x, y, x + cell_width, y + label_height), fill="white")
            draw.text((x + 8, y + 6), label, fill="black", font=font)
            sheet.paste(page, (x, y + label_height))

        first = page_number(group[0])
        last = page_number(group[-1])
        output = output_dir / f"{document}-{first:03d}-{last:03d}.png"
        sheet.save(output, optimize=True)
        sheet_paths.append(output)

    return sheet_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("documents", nargs="+")
    args = parser.parse_args()
    for document in args.documents:
        for output in make_sheets(args.render_dir, args.output_dir, document):
            print(output)


if __name__ == "__main__":
    main()
