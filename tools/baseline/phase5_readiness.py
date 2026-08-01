#!/usr/bin/env python3
"""Validate and render the prepared Phase 5 validation-epoch readiness record."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import validate_reduction
    from .validation_profiles import profile_by_id, validate_registry
except ImportError:
    import validate_reduction
    from validation_profiles import profile_by_id, validate_registry


SCHEMA_VERSION = "citlali-phase5-validation-readiness-v1"
MODES = {"point", "oof", "science", "beammap"}
GATE_NAMES = {"audit", "config", "contract", "products"}
GATE_STATES = {"pass", "blocked", "fail"}
EVIDENCE_ROLES = {"fixture_smoke", "promotion_candidate"}


class ReadinessError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReadinessError(f"{path}: expected JSON object")
    return value


def nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReadinessError(f"{context}: expected non-empty string")
    return value


def validate_readiness(
    manifest_path: Path,
    registry_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReadinessError(f"{manifest_path}: unsupported schema_version")
    registry = validate_registry(registry_path, ledger_path)
    epoch_id = nonempty_text(manifest.get("epoch_id"), "epoch_id")
    if epoch_id not in registry["preparing_epoch_ids"]:
        raise ReadinessError(
            f"epoch_id {epoch_id!r} is not a registered preparing epoch; "
            f"expected one of {registry['preparing_epoch_ids']!r}"
        )
    if manifest.get("status") != "preparing":
        raise ReadinessError("status must be 'preparing'")

    binding_policy_id = nonempty_text(
        manifest.get("config_binding_policy_id"),
        "config_binding_policy_id",
    )
    evaluated_on = nonempty_text(
        manifest.get("evaluated_on"),
        "evaluated_on",
    )
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise ReadinessError("fixtures must be a list")

    seen_modes: set[str] = set()
    gate_counts: Counter[str] = Counter()
    fixture_results: list[dict[str, Any]] = []
    for index, fixture_value in enumerate(fixtures):
        context = f"fixtures[{index}]"
        if not isinstance(fixture_value, dict):
            raise ReadinessError(f"{context}: expected object")
        mode = nonempty_text(fixture_value.get("mode"), f"{context}.mode")
        if mode not in MODES:
            raise ReadinessError(f"{context}.mode: unsupported mode {mode!r}")
        if mode in seen_modes:
            raise ReadinessError(f"duplicate fixture mode {mode!r}")
        seen_modes.add(mode)

        profile_id = nonempty_text(
            fixture_value.get("profile_id"), f"{context}.profile_id"
        )
        profile = profile_by_id(registry, profile_id)
        if profile["status"] != "preparing" or profile["epoch_id"] != epoch_id:
            raise ReadinessError(
                f"{context}.profile_id: profile is not in the preparing epoch"
            )
        if profile["mode"] != mode:
            raise ReadinessError(
                f"{context}.profile_id: profile mode {profile['mode']!r} "
                f"does not match {mode!r}"
            )
        if profile["config"].get("binding_policy_id") != binding_policy_id:
            raise ReadinessError(
                f"{context}.profile_id: config binding policy mismatch"
            )

        nonempty_text(fixture_value.get("citlali_sha"), f"{context}.citlali_sha")
        nonempty_text(
            fixture_value.get("citlali_version"),
            f"{context}.citlali_version",
        )
        nonempty_text(fixture_value.get("local_path"), f"{context}.local_path")
        nonempty_text(
            fixture_value.get("baseline_path"),
            f"{context}.baseline_path",
        )
        evidence_role = nonempty_text(
            fixture_value.get("evidence_role"),
            f"{context}.evidence_role",
        )
        if evidence_role not in EVIDENCE_ROLES:
            raise ReadinessError(
                f"{context}.evidence_role: unsupported role {evidence_role!r}"
            )
        gate_values = fixture_value.get("gates")
        if not isinstance(gate_values, dict) or set(gate_values) != GATE_NAMES:
            raise ReadinessError(
                f"{context}.gates: expected exactly {sorted(GATE_NAMES)}"
            )
        for gate, state in gate_values.items():
            if state not in GATE_STATES:
                raise ReadinessError(
                    f"{context}.gates.{gate}: unsupported state {state!r}"
                )
            gate_counts[f"{gate}:{state}"] += 1

        blockers = fixture_value.get("promotion_blockers")
        if not isinstance(blockers, list) or not all(
            isinstance(item, str) and item.strip() for item in blockers
        ):
            raise ReadinessError(f"{context}.promotion_blockers: expected string list")
        if any(state != "pass" for state in gate_values.values()) and not blockers:
            raise ReadinessError(
                f"{context}: non-passing gates require a promotion blocker"
            )
        fixture_results.append(
            {
                "mode": mode,
                "profile_id": profile_id,
                "citlali_sha": fixture_value["citlali_sha"],
                "citlali_version": fixture_value["citlali_version"],
                "local_path": fixture_value["local_path"],
                "baseline_path": fixture_value["baseline_path"],
                "evidence_role": evidence_role,
                "gates": gate_values,
                "promotion_blockers": blockers,
                "promotion_ready": (
                    evidence_role == "promotion_candidate"
                    and all(state == "pass" for state in gate_values.values())
                    and not blockers
                ),
            }
        )

    if seen_modes != MODES:
        raise ReadinessError(
            f"fixtures must cover {sorted(MODES)}; got {sorted(seen_modes)}"
        )

    global_blockers = manifest.get("global_blockers")
    if not isinstance(global_blockers, list):
        raise ReadinessError("global_blockers must be a list")
    blocker_ids: set[str] = set()
    for index, blocker in enumerate(global_blockers):
        if not isinstance(blocker, dict):
            raise ReadinessError(f"global_blockers[{index}]: expected object")
        blocker_id = nonempty_text(
            blocker.get("blocker_id"),
            f"global_blockers[{index}].blocker_id",
        )
        if blocker_id in blocker_ids:
            raise ReadinessError(f"duplicate global blocker {blocker_id!r}")
        blocker_ids.add(blocker_id)
        nonempty_text(
            blocker.get("description"),
            f"global_blockers[{index}].description",
        )
        nonempty_text(
            blocker.get("exit_condition"),
            f"global_blockers[{index}].exit_condition",
        )

    promotion_candidate_shas = {
        fixture["citlali_sha"]
        for fixture in fixture_results
        if fixture["evidence_role"] == "promotion_candidate"
    }
    same_sha = len(promotion_candidate_shas) == 1 and all(
        fixture["evidence_role"] == "promotion_candidate" for fixture in fixture_results
    )
    promotion_ready = (
        all(fixture["promotion_ready"] for fixture in fixture_results)
        and same_sha
        and not global_blockers
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "status": "ready" if promotion_ready else "preparing",
        "promotion_ready": promotion_ready,
        "config_binding_policy_id": binding_policy_id,
        "evaluated_on": evaluated_on,
        "fixture_count": len(fixture_results),
        "same_sha": same_sha,
        "gate_counts": dict(sorted(gate_counts.items())),
        "fixtures": sorted(fixture_results, key=lambda item: item["mode"]),
        "global_blockers": global_blockers,
        "source_manifest": str(manifest_path.resolve()),
    }


def verify_fixtures(
    readiness: dict[str, Any],
    registry: dict[str, Any],
    output_dir: Path,
    product_contracts: Path,
    binding_policies: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for fixture in readiness["fixtures"]:
        mode = fixture["mode"]
        baseline = Path(fixture["baseline_path"]).expanduser().resolve()
        candidate = Path(fixture["local_path"]).expanduser().resolve()
        if not baseline.is_dir():
            raise ReadinessError(f"{mode}: baseline fixture does not exist: {baseline}")
        if not candidate.is_dir():
            raise ReadinessError(
                f"{mode}: candidate fixture does not exist: {candidate}"
            )
        profile = profile_by_id(registry, fixture["profile_id"])
        case_dir = output_dir / mode
        validation = validate_reduction.run_validation(
            profile,
            baseline,
            candidate,
            case_dir,
            product_contracts,
            binding_policies,
        )
        validate_reduction.write_text(
            case_dir / "validation.json",
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
        )
        validate_reduction.write_text(
            case_dir / "validation.md",
            validate_reduction.render_markdown(validation),
        )
        actual_pass = {
            gate["name"]: bool(gate["passed"]) for gate in validation["gates"]
        }
        expected_pass = {
            gate: state == "pass" for gate, state in fixture["gates"].items()
        }
        mismatches = [
            gate
            for gate in sorted(GATE_NAMES)
            if actual_pass.get(gate) != expected_pass[gate]
        ]
        cases.append(
            {
                "mode": mode,
                "profile_id": fixture["profile_id"],
                "actual_pass": actual_pass,
                "expected_states": fixture["gates"],
                "mismatched_gates": mismatches,
                "matched": not mismatches,
                "report": str((case_dir / "validation.md").resolve()),
            }
        )
    return {
        "schema_version": "citlali-phase5-fixture-verification-v1",
        "matched": all(case["matched"] for case in cases),
        "case_count": len(cases),
        "cases": cases,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 Validation Readiness",
        "",
        f"- Epoch: `{result['epoch_id']}`",
        f"- Status: **{result['status']}**",
        f"- Promotion ready: `{result['promotion_ready']}`",
        f"- One promotion-candidate SHA: `{result['same_sha']}`",
        f"- Config binding policy: `{result['config_binding_policy_id']}`",
        f"- Evaluated: `{result['evaluated_on']}`",
        "",
        "## Fixture Matrix",
        "",
        "| Mode | Commit | Evidence | Audit | Config | Contract | Products | Promotion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fixture in result["fixtures"]:
        gates = fixture["gates"]
        lines.append(
            f"| `{fixture['mode']}` | `{fixture['citlali_sha']}` | "
            f"{fixture['evidence_role']} | "
            f"{gates['audit']} | {gates['config']} | {gates['contract']} | "
            f"{gates['products']} | "
            f"{'ready' if fixture['promotion_ready'] else 'blocked'} |"
        )
    lines.extend(["", "## Fixture Promotion Blockers", ""])
    for fixture in result["fixtures"]:
        if fixture["promotion_blockers"]:
            lines.append(
                f"- `{fixture['mode']}`: "
                + "; ".join(
                    blocker.rstrip(".") for blocker in fixture["promotion_blockers"]
                )
                + "."
            )
        else:
            lines.append(f"- `{fixture['mode']}`: none")
    lines.extend(["", "## Global Blockers", ""])
    for blocker in result["global_blockers"]:
        lines.append(
            f"- `{blocker['blocker_id']}`: {blocker['description']} "
            f"Exit: {blocker['exit_condition']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "validation/phase5_validation_readiness.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo_root / "validation/validation_profiles.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=repo_root / "validation/accepted_runs.json",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--verify-fixtures",
        action="store_true",
        help="Rerun all four declared profile checks and verify recorded gate states.",
    )
    parser.add_argument(
        "--fixture-output-dir",
        type=Path,
        default=Path("/tmp/citlali-phase5-fixtures"),
    )
    parser.add_argument(
        "--product-contracts",
        type=Path,
        default=repo_root / "validation/product_contracts.json",
    )
    parser.add_argument(
        "--binding-policies",
        type=Path,
        default=repo_root / "validation/config_binding_policies.json",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return nonzero until all promotion gates and global blockers close.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = validate_readiness(
            args.manifest.expanduser().resolve(),
            args.registry.expanduser().resolve(),
            args.ledger.expanduser().resolve(),
        )
        fixture_verification = None
        if args.verify_fixtures:
            registry = validate_registry(
                args.registry.expanduser().resolve(),
                args.ledger.expanduser().resolve(),
            )
            fixture_verification = verify_fixtures(
                result,
                registry,
                args.fixture_output_dir.expanduser().resolve(),
                args.product_contracts.expanduser().resolve(),
                args.binding_policies.expanduser().resolve(),
            )
    except (
        OSError,
        json.JSONDecodeError,
        ReadinessError,
        validate_reduction.ValidationError,
    ) as error:
        print(f"Phase 5 readiness record invalid: {error}", file=sys.stderr)
        return 2
    report = render_markdown(result)
    if args.json_out:
        write_text(
            args.json_out.expanduser(),
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
    if args.report_out:
        write_text(args.report_out.expanduser(), report)
    print(report, end="")
    if fixture_verification is not None:
        fixture_dir = args.fixture_output_dir.expanduser().resolve()
        print(
            "Fixture verification: "
            f"{'matched' if fixture_verification['matched'] else 'MISMATCH'} "
            f"({fixture_verification['case_count']} cases); "
            f"reports={fixture_dir}"
        )
        if not fixture_verification["matched"]:
            return 3
    return 1 if args.require_ready and not result["promotion_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
