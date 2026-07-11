#!/usr/bin/env python3
"""Audit the transitional YAML boundary in PTCProc::get_config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_PATH_COUNT = 171
EXPECTED_PATH_SHA256 = (
    "b2bc50c1cf73064a1279c0335bf03a3033938ec1589095f390fdcac9c0fc67b6"
)


def parser_body(path: Path) -> str:
    text = path.read_text()
    start = text.index("void PTCProc::get_config")
    end = text.find("\ntemplate", start + 1)
    if end < 0:
        raise ValueError(f"unable to find end of PTCProc::get_config in {path}")
    return text[start:end]


def literal_paths(body: str) -> list[str]:
    tuple_pattern = re.compile(r"std::tuple\s*\{([^}]*)\}", re.DOTALL)
    string_pattern = re.compile(r'"([^"]+)"')
    paths = {
        ".".join(string_pattern.findall(match.group(1)))
        for match in tuple_pattern.finditer(body)
        if string_pattern.findall(match.group(1))
    }
    return sorted(paths)


def family(path: str) -> str:
    parts = path.split(".")
    if parts[:2] == ["timestream", "fruit_loops"]:
        return "fruit_loops"
    if parts[:3] == ["timestream", "processed_time_chunk", "clean"]:
        return "clean"
    if parts[:3] == ["timestream", "processed_time_chunk", "weighting"]:
        return "weighting"
    if parts[:3] == ["timestream", "processed_time_chunk", "flagging"]:
        return "flagging"
    return "unclassified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    source = repo_root / "include/citlali/core/timestream/ptc/ptcproc.h"
    body = parser_body(source)
    paths = literal_paths(body)
    digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    direct_exit = "std::exit" in body or re.search(r"(?<![\w:])exit\s*\(", body)
    counts = dict(sorted(Counter(map(family, paths)).items()))
    drift = len(paths) != EXPECTED_PATH_COUNT or digest != EXPECTED_PATH_SHA256
    result = {
        "schema_version": "citlali-processed-config-boundary-audit-v1",
        "source": str(source.relative_to(repo_root)),
        "literal_path_count": len(paths),
        "literal_path_sha256": digest,
        "expected_path_count": EXPECTED_PATH_COUNT,
        "expected_path_sha256": EXPECTED_PATH_SHA256,
        "family_counts": counts,
        "path_drift": drift,
        "direct_process_exit": bool(direct_exit),
        "paths": paths,
        "note": (
            "Literal tuple paths freeze the legacy boundary; dynamic tuple "
            "components are represented by their literal prefix."
        ),
    }
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"| {name} | {count} |" for name, count in counts.items()
        )
        output.write_text(
            "# Processed Timestream Boundary Audit\n\n"
            f"- Literal paths: `{len(paths)}`\n"
            f"- Path digest: `{digest}`\n"
            f"- Path drift: `{drift}`\n"
            f"- Direct process exit: `{bool(direct_exit)}`\n\n"
            "| Family | Paths |\n| --- | ---: |\n"
            f"{rows}\n"
        )
    print(
        "processed config boundary: "
        f"paths={len(paths)} drift={drift} direct_exit={bool(direct_exit)} "
        f"families={counts}"
    )
    if args.fail_on_drift and (drift or direct_exit):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
