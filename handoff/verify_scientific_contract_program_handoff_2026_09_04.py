#!/usr/bin/env python3
"""Verify the 2026-09-04 scientific-contract manager handoff.

The verifier binds immutable snapshot commits, trees, candidate artifact
bytes, handoff content, and the handoff-only repository diff.  Moving branch
tips and worktree states are reported for the incoming manager to compare;
their later movement does not rewrite the dated snapshot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HANDOFF = HERE / "HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM.md"
MANIFEST = HERE / "HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM_MANIFEST.json"
README = HERE / "README.md"
BASE = "9f42d348298d76c5d5145aaf0c3eace1f3e154c1"
BASE_TREE = "e51f22760c64454ce7233c45dd740aa710777bae"
EXPECTED_DIFF = {
    "handoff/HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM.md",
    "handoff/HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM_MANIFEST.json",
    "handoff/README.md",
    "handoff/verify_scientific_contract_program_handoff_2026_09_04.py",
}


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def git(*args: str, text: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=text, check=False
    )
    if result.returncode:
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        stdout = result.stdout if text else result.stdout.decode(errors="replace")
        raise Failure(f"git {' '.join(args)} failed: {(stderr or stdout).strip()}")
    return result


def git_text(*args: str) -> str:
    return git(*args).stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commit_bytes(commit: str, path: str) -> bytes:
    return git("show", f"{commit}:{path}", text=False).stdout


def verify_handoff_diff() -> str:
    head = git_text("rev-parse", "HEAD")
    require(git_text("rev-parse", f"{BASE}^{{tree}}") == BASE_TREE,
            "canonical snapshot tree mismatch")
    status = git("status", "--porcelain=v1", "--untracked-files=all").stdout
    status_paths = {
        line[3:].split(" -> ", 1)[-1]
        for line in status.splitlines()
        if len(line) >= 4
    }
    if head == BASE:
        require(status_paths == EXPECTED_DIFF,
                f"pre-commit handoff path set mismatch: {sorted(status_paths)}")
        return "pre_commit"
    require(git_text("rev-parse", "HEAD^") == BASE,
            "handoff candidate is not a direct child of the canonical snapshot")
    changed = set(git_text("diff", "--name-only", f"{BASE}..HEAD").splitlines())
    require(changed == EXPECTED_DIFF,
            f"post-commit handoff path set mismatch: {sorted(changed)}")
    require(not status, "post-commit handoff worktree is not clean")
    require(not git_text("diff", "--check", f"{BASE}..HEAD"),
            "handoff diff check failed")
    return "post_commit"


def main() -> int:
    try:
        mode = verify_handoff_diff()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        require(manifest["schema_version"] == 1, "manifest schema mismatch")
        canonical = manifest["canonical_snapshot"]
        require(canonical["commit"] == BASE, "manifest base mismatch")
        require(canonical["tree"] == BASE_TREE, "manifest base tree mismatch")

        candidate = manifest["review_candidate"]
        require(git_text("rev-parse", f'{candidate["commit"]}^{{tree}}') == candidate["tree"],
                "candidate tree mismatch")
        require(git_text("rev-parse", f'{candidate["commit"]}^') == candidate["parent"],
                "candidate parent mismatch")
        for path, expected in candidate["artifact_sha256"].items():
            require(sha256(commit_bytes(candidate["commit"], path)) == expected,
                    f"candidate artifact digest mismatch: {path}")

        for name, snapshot in manifest["independent_snapshots"].items():
            require(git_text("rev-parse", f'{snapshot["commit"]}^{{tree}}') == snapshot["tree"],
                    f"{name} snapshot tree mismatch")

        handoff_text = HANDOFF.read_text(encoding="utf-8")
        for phrase in (
            "Governing Approach That Must Not Drift",
            BASE,
            candidate["commit"],
            "MSP-E027",
            "MAP observation-bundle",
            "CTI-OD-007",
            "GPT-6 Astra",
            "Pickup Acceptance Gate",
        ):
            require(phrase in handoff_text, f"handoff statement missing: {phrase}")
        require("HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM.md" in
                README.read_text(encoding="utf-8"), "handoff README link missing")
    except (Failure, OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS mode={mode} head={git_text('rev-parse', 'HEAD')}")
    print(f"PASS canonical_snapshot={BASE} tree={BASE_TREE}")
    print("PASS candidate_commit_tree_parent_and_artifact_digests=true")
    print("PASS fruit_and_align_snapshot_commits_and_trees=true")
    print("PASS handoff_only_diff=true pickup_gate_present=true")
    print("scientific_contract_program_handoff=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
