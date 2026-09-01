#!/usr/bin/env python3
"""Audit Markdown links for a standalone SCI-FLT-MATCHED bundle tree."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def markdown_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def target_path(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    files = markdown_files(root)
    unresolved: list[dict[str, str]] = []
    checked = 0
    for source in files:
        for match in LINK_RE.finditer(source.read_text(encoding="utf-8")):
            raw_target = match.group(1)
            resolved = target_path(source, raw_target)
            if resolved is None:
                continue
            checked += 1
            try:
                resolved.relative_to(root)
            except ValueError:
                unresolved.append(
                    {
                        "source": str(source.relative_to(root)),
                        "target": raw_target,
                        "reason": "target escapes standalone root",
                    }
                )
                continue
            if not resolved.exists():
                unresolved.append(
                    {
                        "source": str(source.relative_to(root)),
                        "target": raw_target,
                        "reason": "target absent",
                    }
                )
    return {
        "status": "PASS" if not unresolved else "FAIL",
        "scope": "standalone Markdown links only; external URLs are excluded",
        "root": str(root),
        "markdown_files": len(files),
        "local_links_checked": checked,
        "unresolved_local_links": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
