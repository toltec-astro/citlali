#!/usr/bin/env python3
"""Audit process exits exposed by the session boundary and core sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "citlali-session-exit-audit-v1"
DEFAULT_ENTRY = "include/citlali/core/cli/standard_reduction_execution.h"
DEFAULT_BASELINE = "tools/refactor/session_exit_baseline.json"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
EXIT_RE = re.compile(r"\b(?:std::)?exit\s*\(")


def resolve_project_include(repo: Path, source: Path, include: str) -> Path | None:
    candidates = [source.parent / include]
    if include.startswith("citlali/") or include.startswith("citlali_config/"):
        candidates.append(repo / "include" / include)
    else:
        candidates.append(repo / include)
    for candidate in candidates:
        candidate = candidate.resolve()
        try:
            candidate.relative_to(repo)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def dependency_files(repo: Path, entries: Iterable[str]) -> list[Path]:
    pending = [(repo / entry).resolve() for entry in entries]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"session audit entry does not exist: {path}")
        visited.add(path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            dependency = resolve_project_include(repo, path, match.group(1))
            if dependency is not None and dependency not in visited:
                pending.append(dependency)
    return sorted(visited)


def core_library_sources(repo: Path) -> list[Path]:
    source_root = repo / "src/citlali/core"
    if not source_root.is_dir():
        return []
    return sorted(path.resolve() for path in source_root.rglob("*.cpp"))


def audit(repo: Path, entries: list[str]) -> dict[str, object]:
    repo = repo.resolve()
    files = sorted(set(dependency_files(repo, entries) + core_library_sources(repo)))
    rows: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(repo).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not EXIT_RE.search(line):
                continue
            rows.append(
                {
                    "file": relative,
                    "line": line_number,
                    "source": line.strip(),
                    "boundary": (
                        "cli" if relative.startswith("include/citlali/core/cli/")
                        else "library"
                    ),
                }
            )
    library_counts = Counter(
        str(row["file"]) for row in rows if row["boundary"] == "library"
    )
    cli_counts = Counter(
        str(row["file"]) for row in rows if row["boundary"] == "cli"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "dependency_file_count": len(files),
        "exit_count": len(rows),
        "library_exit_count": sum(library_counts.values()),
        "cli_exit_count": sum(cli_counts.values()),
        "library_exit_counts_by_file": dict(sorted(library_counts.items())),
        "cli_exit_counts_by_file": dict(sorted(cli_counts.items())),
        "exits": rows,
        "scope_note": (
            "This combines conservative project-header dependency reachability "
            "with every core library source file. It is not proof that every "
            "reported definition is runtime-call reachable."
        ),
    }


def baseline_growth(audit_result: dict[str, object], baseline: dict[str, object]) -> list[str]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        return ["baseline schema version does not match the audit"]
    expected = baseline.get("library_exit_counts_by_file", {})
    actual = audit_result["library_exit_counts_by_file"]
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return ["baseline or audit library-exit counts are invalid"]
    problems: list[str] = []
    for path, count in actual.items():
        allowed = int(expected.get(path, 0))
        if int(count) > allowed:
            problems.append(f"{path}: exits grew from {allowed} to {count}")
    return problems


def markdown_report(result: dict[str, object], growth: list[str]) -> str:
    lines = [
        "# Session Exit Boundary Audit",
        "",
        f"- Dependency files: {result['dependency_file_count']}",
        f"- Dependency-reachable exits: {result['exit_count']}",
        f"- Library exits: {result['library_exit_count']}",
        f"- CLI exits: {result['cli_exit_count']}",
        f"- Baseline growth: {'yes' if growth else 'no'}",
        "",
        str(result["scope_note"]),
        "",
        "## Library Exits By File",
        "",
        "| File | Count |",
        "| --- | ---: |",
    ]
    counts = result["library_exit_counts_by_file"]
    assert isinstance(counts, dict)
    for path, count in counts.items():
        lines.append(f"| `{path}` | {count} |")
    if growth:
        lines.extend(["", "## Growth", ""])
        lines.extend(f"- {problem}" for problem in growth)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-growth", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    entries = args.entry or [DEFAULT_ENTRY]
    result = audit(repo, entries)
    baseline_path = args.baseline
    if baseline_path is None:
        baseline_path = repo / DEFAULT_BASELINE
    elif not baseline_path.is_absolute():
        baseline_path = repo / baseline_path
    growth: list[str] = []
    if baseline_path.is_file():
        growth = baseline_growth(
            result, json.loads(baseline_path.read_text(encoding="utf-8")))
    elif args.fail_on_growth:
        growth = [f"baseline does not exist: {baseline_path}"]

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = markdown_report(result, growth)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report, encoding="utf-8")
    print(
        "session exit audit: "
        f"dependencies={result['dependency_file_count']} "
        f"library_exits={result['library_exit_count']} "
        f"cli_exits={result['cli_exit_count']} growth={len(growth)}"
    )
    for problem in growth:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if args.fail_on_growth and growth else 0


if __name__ == "__main__":
    raise SystemExit(main())
