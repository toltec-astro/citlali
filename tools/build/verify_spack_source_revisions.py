#!/usr/bin/env python3
"""Verify exact first-party source revisions used by the Spack build lane."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class RevisionResult:
    repository: str
    expected: str
    actual: str
    clean: bool


CommandRunner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def load_revisions(manifest: Path) -> dict[str, str]:
    """Load and validate repository revisions from the JSON manifest."""
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported upstream revision manifest schema")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise ValueError("upstream revision manifest has no repositories")

    revisions: dict[str, str] = {}
    for name, record in repositories.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid repository record for {name}")
        commit = record.get("commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"invalid full commit for {name}")
        revisions[name] = commit
    return revisions


def inspect_revisions(
    workspace_root: Path,
    revisions: dict[str, str],
    *,
    runner: CommandRunner = _run,
) -> list[RevisionResult]:
    """Inspect every declared sibling checkout without changing it."""
    results = []
    for repository, expected in revisions.items():
        source = workspace_root / repository
        actual = runner(["git", "-C", str(source), "rev-parse", "HEAD"])
        status = runner(["git", "-C", str(source), "status", "--porcelain"])
        results.append(
            RevisionResult(
                repository=repository,
                expected=expected,
                actual=actual,
                clean=not status,
            )
        )
    return results


def require_accepted_revisions(results: Sequence[RevisionResult]) -> None:
    """Reject revision drift or dirty first-party dependency sources."""
    failures = []
    for result in results:
        if result.actual != result.expected:
            failures.append(
                f"{result.repository}: expected {result.expected}, "
                f"got {result.actual}"
            )
        if not result.clean:
            failures.append(f"{result.repository}: checkout is dirty")
    if failures:
        raise RuntimeError("unaccepted first-party sources:\n" + "\n".join(failures))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=source_root / "build/spack-sources",
        help="directory containing the pinned first-party build sources",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=source_root / "spack/upstream-revisions.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    revisions = load_revisions(args.manifest.expanduser().resolve())
    results = inspect_revisions(args.workspace_root.expanduser().resolve(), revisions)
    require_accepted_revisions(results)
    for result in results:
        print(f"{result.repository}={result.actual} clean=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
