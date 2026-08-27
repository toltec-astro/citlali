#!/usr/bin/env python3
"""Prepare pinned first-party sources for Citlali's independent Spack lane."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    branch: str
    commit: str


CommandRunner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def load_sources(manifest: Path) -> list[SourceSpec]:
    """Load complete source records from the accepted revision manifest."""
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported upstream revision manifest schema")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise ValueError("upstream revision manifest has no repositories")

    sources = []
    for name, record in repositories.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid repository record for {name}")
        url = record.get("url")
        branch = record.get("branch")
        commit = record.get("commit")
        if not isinstance(url, str) or not url.startswith("https://github.com/"):
            raise ValueError(f"invalid repository URL for {name}")
        if not isinstance(branch, str) or not branch:
            raise ValueError(f"invalid branch for {name}")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"invalid full commit for {name}")
        sources.append(SourceSpec(name, url, branch, commit))
    return sources


def prepare_source(
    root: Path,
    source: SourceSpec,
    *,
    refresh: bool,
    runner: CommandRunner = _run,
) -> None:
    """Clone a missing source or advance a clean checkout when authorized."""
    target = root / source.name
    cloned = False
    if not target.exists():
        root.mkdir(parents=True, exist_ok=True)
        runner(
            [
                "git",
                "clone",
                "--branch",
                source.branch,
                "--single-branch",
                source.url,
                str(target),
            ]
        )
        cloned = True
    elif not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a Git checkout")

    status = runner(["git", "-C", str(target), "status", "--porcelain"])
    if status:
        raise RuntimeError(f"refusing to modify dirty checkout {target}")
    actual = runner(["git", "-C", str(target), "rev-parse", "HEAD"])
    if actual == source.commit:
        return
    if not refresh and not cloned:
        raise RuntimeError(
            f"{source.name} is at {actual}, expected {source.commit}; "
            "rerun with --refresh to update the clean build-only checkout"
        )

    if not cloned:
        runner(["git", "-C", str(target), "fetch", "origin", source.branch])
    runner(["git", "-C", str(target), "checkout", "--detach", source.commit])
    actual = runner(["git", "-C", str(target), "rev-parse", "HEAD"])
    if actual != source.commit:
        raise RuntimeError(f"failed to prepare {source.name} at {source.commit}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=source_root / "build/spack-sources",
        help="ignored directory that owns the pinned first-party checkouts",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=source_root / "spack/upstream-revisions.json",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="update clean existing checkouts to the accepted revisions",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    for source in load_sources(args.manifest.expanduser().resolve()):
        prepare_source(source_root, source, refresh=args.refresh)
        print(f"prepared {source.name}={source.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
