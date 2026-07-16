#!/usr/bin/env python3
"""Validate Citlali's intended post-baseline science-change ledger."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "citlali-intended-science-change-ledger-v1"
FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CHANGE_CLASSES = {
    "correctness_bug_fix",
    "performance_optimization",
    "performance_and_robustness",
    "intentional_algorithm_change",
    "intentional_default_change",
    "intentional_product_or_schema_change",
}
MAPPING_METHODS = {"manual_transplant", "patch_equivalent_cherry_pick"}
EVIDENCE_KINDS = {"accepted_run", "document"}
MODES = {"point", "oof", "science", "beammap"}


class ScienceChangeLedgerError(ValueError):
    pass


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ScienceChangeLedgerError(
            f"{context}: missing required field {key!r}"
        )
    return mapping[key]


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScienceChangeLedgerError(f"{context}: expected object")
    return value


def require_nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScienceChangeLedgerError(f"{context}: expected non-empty string")
    return value


def require_nonempty_text_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ScienceChangeLedgerError(f"{context}: expected non-empty list")
    result = []
    for index, item in enumerate(value):
        result.append(require_nonempty_text(item, f"{context}[{index}]"))
    if len(result) != len(set(result)):
        raise ScienceChangeLedgerError(f"{context}: duplicate value")
    return result


def load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return require_mapping(json.load(stream), str(path))


def git_output(repo_root: Path, arguments: list[str], input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ScienceChangeLedgerError(
            f"git {' '.join(arguments)} failed: {detail}"
        )
    return result.stdout


def require_commit(repo_root: Path, commit: str, context: str) -> None:
    if not FULL_GIT_SHA_RE.fullmatch(commit):
        raise ScienceChangeLedgerError(f"{context}: expected full 40-character Git SHA")
    git_output(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])


def require_ancestor(repo_root: Path, commit: str, context: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 1:
        raise ScienceChangeLedgerError(f"{context}: commit is not an ancestor of HEAD")
    if result.returncode != 0:
        raise ScienceChangeLedgerError(
            f"{context}: unable to test ancestry: {result.stderr.decode().strip()}"
        )


def patch_id(repo_root: Path, commit: str) -> str:
    patch = git_output(
        repo_root,
        ["show", "--no-ext-diff", "--full-index", "--pretty=format:", commit],
    )
    result = subprocess.run(
        ["git", "patch-id", "--stable"],
        input=patch,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "no patch identity produced"
        raise ScienceChangeLedgerError(f"unable to compute patch-id for {commit}: {detail}")
    return result.stdout.split()[0]


def accepted_run_index(path: Path) -> dict[str, dict[str, Any]]:
    ledger = load_json_mapping(path)
    records = require(ledger, "records", str(path))
    if not isinstance(records, list):
        raise ScienceChangeLedgerError(f"{path}.records: expected list")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        item = require_mapping(record, f"{path}.records[{index}]")
        record_id = require_nonempty_text(
            require(item, "record_id", f"{path}.records[{index}]"),
            f"{path}.records[{index}].record_id",
        )
        result[record_id] = item
    return result


def product_family_ids(path: Path) -> set[str]:
    registry = load_json_mapping(path)
    families = require_mapping(require(registry, "families", str(path)), f"{path}.families")
    return set(families)


def validate_mapping(mapping: Any, context: str, repo_root: Path) -> str:
    item = require_mapping(mapping, context)
    source = require_nonempty_text(
        require(item, "source_commit", context), f"{context}.source_commit"
    )
    integration = require_nonempty_text(
        require(item, "integration_commit", context), f"{context}.integration_commit"
    )
    method = require_nonempty_text(
        require(item, "method", context), f"{context}.method"
    )
    if method not in MAPPING_METHODS:
        raise ScienceChangeLedgerError(f"{context}.method: unsupported method {method!r}")
    require_commit(repo_root, source, f"{context}.source_commit")
    require_commit(repo_root, integration, f"{context}.integration_commit")
    require_ancestor(repo_root, integration, f"{context}.integration_commit")

    if method == "patch_equivalent_cherry_pick":
        expected = require_nonempty_text(
            require(item, "patch_id", context), f"{context}.patch_id"
        )
        if not FULL_GIT_SHA_RE.fullmatch(expected):
            raise ScienceChangeLedgerError(f"{context}.patch_id: invalid patch identity")
        source_patch_id = patch_id(repo_root, source)
        integration_patch_id = patch_id(repo_root, integration)
        if source_patch_id != expected or integration_patch_id != expected:
            raise ScienceChangeLedgerError(
                f"{context}: patch identity mismatch; expected={expected} "
                f"source={source_patch_id} integration={integration_patch_id}"
            )
    elif "patch_id" in item:
        raise ScienceChangeLedgerError(
            f"{context}.patch_id: manual transplant must not claim patch equivalence"
        )
    return integration


def validate_evidence(
    evidence: Any,
    context: str,
    repo_root: Path,
    accepted_runs: dict[str, dict[str, Any]],
) -> None:
    item = require_mapping(evidence, context)
    kind = require_nonempty_text(require(item, "kind", context), f"{context}.kind")
    if kind not in EVIDENCE_KINDS:
        raise ScienceChangeLedgerError(f"{context}.kind: unsupported kind {kind!r}")
    require_nonempty_text(require(item, "finding", context), f"{context}.finding")
    if kind == "accepted_run":
        record_id = require_nonempty_text(
            require(item, "record_id", context), f"{context}.record_id"
        )
        if record_id not in accepted_runs:
            raise ScienceChangeLedgerError(
                f"{context}.record_id: unknown accepted-run record {record_id!r}"
            )
    else:
        relative_path = require_nonempty_text(
            require(item, "path", context), f"{context}.path"
        )
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ScienceChangeLedgerError(
                f"{context}.path: expected repository-relative path"
            )
        if not (repo_root / path).is_file():
            raise ScienceChangeLedgerError(
                f"{context}.path: referenced document does not exist"
            )


def validate_change(
    change: Any,
    index: int,
    repo_root: Path,
    accepted_runs: dict[str, dict[str, Any]],
    product_families: set[str],
) -> tuple[str, list[str]]:
    context = f"changes[{index}]"
    item = require_mapping(change, context)
    change_id = require_nonempty_text(
        require(item, "change_id", context), f"{context}.change_id"
    )
    require_nonempty_text(require(item, "title", context), f"{context}.title")
    require_nonempty_text(
        require(item, "recorded_date", context), f"{context}.recorded_date"
    )
    if require_nonempty_text(require(item, "status", context), f"{context}.status") != "accepted":
        raise ScienceChangeLedgerError(f"{context}.status: expected 'accepted'")
    change_class = require_nonempty_text(
        require(item, "change_class", context), f"{context}.change_class"
    )
    if change_class not in CHANGE_CLASSES:
        raise ScienceChangeLedgerError(
            f"{context}.change_class: unsupported class {change_class!r}"
        )
    require_nonempty_text(
        require(item, "source_branch", context), f"{context}.source_branch"
    )
    require_nonempty_text(require(item, "rationale", context), f"{context}.rationale")

    modes = require_nonempty_text_list(
        require(item, "affected_modes", context), f"{context}.affected_modes"
    )
    unknown_modes = sorted(set(modes) - MODES)
    if unknown_modes:
        raise ScienceChangeLedgerError(
            f"{context}.affected_modes: unsupported modes {unknown_modes}"
        )
    families = require_nonempty_text_list(
        require(item, "affected_product_families", context),
        f"{context}.affected_product_families",
    )
    unknown_families = sorted(set(families) - product_families)
    if unknown_families:
        raise ScienceChangeLedgerError(
            f"{context}.affected_product_families: unknown families {unknown_families}"
        )

    effect = require_mapping(
        require(item, "expected_effect", context), f"{context}.expected_effect"
    )
    require_nonempty_text(
        require(effect, "behavior", f"{context}.expected_effect"),
        f"{context}.expected_effect.behavior",
    )
    require_nonempty_text(
        require(effect, "numerical_or_schema", f"{context}.expected_effect"),
        f"{context}.expected_effect.numerical_or_schema",
    )

    mappings = require(item, "commit_mappings", context)
    if not isinstance(mappings, list) or not mappings:
        raise ScienceChangeLedgerError(f"{context}.commit_mappings: expected non-empty list")
    integrations = [
        validate_mapping(mapping, f"{context}.commit_mappings[{mapping_index}]", repo_root)
        for mapping_index, mapping in enumerate(mappings)
    ]
    if len(integrations) != len(set(integrations)):
        raise ScienceChangeLedgerError(f"{context}.commit_mappings: duplicate integration commit")

    evidence = require(item, "validation_evidence", context)
    if not isinstance(evidence, list) or not evidence:
        raise ScienceChangeLedgerError(
            f"{context}.validation_evidence: expected non-empty list"
        )
    for evidence_index, evidence_item in enumerate(evidence):
        validate_evidence(
            evidence_item,
            f"{context}.validation_evidence[{evidence_index}]",
            repo_root,
            accepted_runs,
        )
    if not any(
        isinstance(evidence_item, dict) and evidence_item.get("kind") == "accepted_run"
        for evidence_item in evidence
    ):
        raise ScienceChangeLedgerError(
            f"{context}.validation_evidence: at least one accepted-run record is required"
        )

    limitations = require(item, "limitations", context)
    if not isinstance(limitations, list):
        raise ScienceChangeLedgerError(f"{context}.limitations: expected list")
    for limitation_index, limitation in enumerate(limitations):
        require_nonempty_text(limitation, f"{context}.limitations[{limitation_index}]")
    return change_id, integrations


def validate_ledger(
    path: Path,
    *,
    repo_root: Path,
    accepted_runs_path: Path,
    product_contracts_path: Path,
) -> tuple[int, int]:
    ledger = load_json_mapping(path)
    if require(ledger, "schema_version", str(path)) != SCHEMA_VERSION:
        raise ScienceChangeLedgerError(f"{path}: unsupported schema_version")

    baseline = require_mapping(
        require(ledger, "refactor_baseline", str(path)), f"{path}.refactor_baseline"
    )
    baseline_commit = require_nonempty_text(
        require(baseline, "commit", f"{path}.refactor_baseline"),
        f"{path}.refactor_baseline.commit",
    )
    require_nonempty_text(
        require(baseline, "definition", f"{path}.refactor_baseline"),
        f"{path}.refactor_baseline.definition",
    )
    require_commit(repo_root, baseline_commit, f"{path}.refactor_baseline.commit")
    require_ancestor(repo_root, baseline_commit, f"{path}.refactor_baseline.commit")

    policy = require_mapping(require(ledger, "policy", str(path)), f"{path}.policy")
    for field in ("scope", "future_change_rule", "baseline_rule", "comparison_rule"):
        require_nonempty_text(
            require(policy, field, f"{path}.policy"), f"{path}.policy.{field}"
        )

    changes = require(ledger, "changes", str(path))
    if not isinstance(changes, list) or not changes:
        raise ScienceChangeLedgerError(f"{path}.changes: expected non-empty list")
    accepted_runs = accepted_run_index(accepted_runs_path)
    product_families = product_family_ids(product_contracts_path)
    validated = [
        validate_change(change, index, repo_root, accepted_runs, product_families)
        for index, change in enumerate(changes)
    ]
    change_ids = [change_id for change_id, _ in validated]
    if len(change_ids) != len(set(change_ids)):
        raise ScienceChangeLedgerError(f"{path}: duplicate change_id")
    integration_commits = [commit for _, commits in validated for commit in commits]
    if len(integration_commits) != len(set(integration_commits)):
        raise ScienceChangeLedgerError(f"{path}: integration commit appears in multiple changes")
    return len(changes), len(integration_commits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=Path("validation/intended_science_changes.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--accepted-runs",
        type=Path,
        default=Path("validation/accepted_runs.json"),
    )
    parser.add_argument(
        "--product-contracts",
        type=Path,
        default=Path("validation/product_contracts.json"),
    )
    args = parser.parse_args()
    try:
        changes, commits = validate_ledger(
            args.ledger,
            repo_root=args.repo_root,
            accepted_runs_path=args.accepted_runs,
            product_contracts_path=args.product_contracts,
        )
    except (
        OSError,
        json.JSONDecodeError,
        ScienceChangeLedgerError,
    ) as error:
        print(f"science-change ledger invalid: {error}", file=sys.stderr)
        return 1
    print(f"science-change ledger valid: changes={changes} integration_commits={commits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
