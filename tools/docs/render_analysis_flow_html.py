#!/usr/bin/env python3
"""Render standalone Mermaid HTML documents for the analysis flow note."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MD = REPO_ROOT / "doc" / "ANALYSIS_FLOW_RAW_TO_SCIENCE_PRODUCTS_2026-07-01.md"
OUTPUT_DIR = REPO_ROOT / "doc" / "analysis_flow_html"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"


@dataclass(frozen=True)
class Diagram:
    title: str
    slug: str
    intro_html: str
    mermaid: str
    outro_html: str


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "diagram"


def markdown_text_to_html(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    output: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(f"<p>{escape(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            items = "".join(f"<li>{escape(item)}</li>" for item in bullets)
            output.append(f"<ul>{items}</ul>")
            bullets = []

    for line in lines:
        if not line:
            flush_paragraph()
            flush_bullets()
            continue
        if line.startswith("!["):
            continue
        if line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:])
            continue
        flush_bullets()
        paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    return "\n".join(output)


def extract_diagrams(markdown: str) -> list[Diagram]:
    sections = list(re.finditer(r"^## (?P<title>.+)$", markdown, flags=re.MULTILINE))
    diagrams: list[Diagram] = []
    for index, section in enumerate(sections):
        title = section.group("title").strip()
        start = section.end()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(markdown)
        body = markdown[start:end]
        match = re.search(r"```mermaid\n(?P<mermaid>.*?)\n```", body, flags=re.DOTALL)
        if match is None:
            continue
        intro = body[:match.start()]
        outro = body[match.end():]
        diagrams.append(
            Diagram(
                title=title,
                slug=slugify(title),
                intro_html=markdown_text_to_html(intro),
                mermaid=match.group("mermaid").strip(),
                outro_html=markdown_text_to_html(outro),
            )
        )
    return diagrams


def page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #223041;
      --muted: #536579;
      --line: #d4dde8;
      --panel: #f7f9fc;
      --accent: #215f9a;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.55;
    }}
    main {{
      width: min(1320px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      margin-bottom: 28px;
      padding-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 10px;
      font-size: 25px;
      letter-spacing: 0;
    }}
    p {{
      max-width: 900px;
      margin: 8px 0 14px;
    }}
    a {{
      color: var(--accent);
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      margin-top: 16px;
      font-size: 14px;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
    }}
    .diagram-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin: 18px 0 34px;
      background: var(--panel);
      overflow-x: auto;
    }}
    .mermaid {{
      min-width: 760px;
      text-align: center;
    }}
    details {{
      margin-top: 16px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    pre.source {{
      overflow-x: auto;
      background: #101820;
      color: #e8eef6;
      border-radius: 6px;
      padding: 14px;
      font-size: 13px;
      line-height: 1.45;
    }}
    ul {{
      margin-top: 8px;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
<script src="{MERMAID_CDN}"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    securityLevel: "loose",
    theme: "default",
    flowchart: {{ htmlLabels: true, curve: "basis" }},
    sequence: {{ mirrorActors: false }}
  }});
</script>
</body>
</html>
"""


def render_diagram_block(diagram: Diagram) -> str:
    source = escape(diagram.mermaid)
    return f"""
<section id="{diagram.slug}">
  <h2>{escape(diagram.title)}</h2>
  {diagram.intro_html}
  <div class="diagram-card">
    <pre class="mermaid">{source}</pre>
    <details>
      <summary>Mermaid source</summary>
      <pre class="source"><code>{source}</code></pre>
    </details>
  </div>
  {diagram.outro_html}
</section>
"""


def render_index(diagrams: list[Diagram]) -> str:
    links = "\n".join(
        f'<a href="#{diagram.slug}">{escape(diagram.title)}</a>'
        for diagram in diagrams
    )
    page_links = "\n".join(
        f'<a href="{diagram.slug}.html">{escape(diagram.title)} page</a>'
        for diagram in diagrams
    )
    sections = "\n".join(render_diagram_block(diagram) for diagram in diagrams)
    body = f"""
<header>
  <h1>Citlali Analysis Flow</h1>
  <p class="note">Standalone Mermaid HTML export from {escape(SOURCE_MD.name)}. Open this file in a browser; JavaScript must be enabled and the Mermaid runtime is loaded from jsDelivr.</p>
  <nav>{links}</nav>
  <nav>{page_links}</nav>
</header>
{sections}
"""
    return page_shell("Citlali Analysis Flow", body)


def render_single(diagram: Diagram, diagrams: list[Diagram]) -> str:
    nav = "\n".join(
        f'<a href="{other.slug}.html">{escape(other.title)}</a>'
        for other in diagrams
        if other.slug != diagram.slug
    )
    body = f"""
<header>
  <h1>{escape(diagram.title)}</h1>
  <p class="note">Standalone Mermaid HTML export. <a href="index.html">All diagrams</a></p>
  <nav>{nav}</nav>
</header>
{render_diagram_block(diagram)}
"""
    return page_shell(f"Citlali Analysis Flow: {diagram.title}", body)


def main() -> None:
    markdown = SOURCE_MD.read_text()
    diagrams = extract_diagrams(markdown)
    if not diagrams:
        raise RuntimeError(f"no Mermaid diagrams found in {SOURCE_MD}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "index.html").write_text(render_index(diagrams))
    for diagram in diagrams:
        (OUTPUT_DIR / f"{diagram.slug}.html").write_text(render_single(diagram, diagrams))

    for path in sorted(OUTPUT_DIR.glob("*.html")):
        print(path)


if __name__ == "__main__":
    main()
