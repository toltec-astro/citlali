#!/usr/bin/env python3
"""Run the profile-pinned Citlali audit, config, contract, and product gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .validation_profiles import (
        RegistryError,
        ledger_records,
        profile_by_id,
        validate_registry,
    )
except ImportError:
    from validation_profiles import (
        RegistryError,
        ledger_records,
        profile_by_id,
        validate_registry,
    )


SCHEMA_VERSION = "citlali-profile-validation-result-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCT_CONTRACTS = REPO_ROOT / "validation/product_contracts.json"
DEFAULT_BINDING_POLICIES = REPO_ROOT / "validation/config_binding_policies.json"
PROVENANCE_FLAGS = {
    "runtime": "--require-runtime-provenance",
    "processed": "--require-processed-provenance",
    "raw": "--require-raw-provenance",
    "mapmaking": "--require-mapmaking-provenance",
    "coadd": "--require-coadd-provenance",
    "noise_products": "--require-noise-products-provenance",
    "pointing": "--require-pointing-provenance",
    "post_processing": "--require-post-processing-provenance",
    "beammap": "--require-beammap-provenance",
    "kids_external": "--require-kids-external-provenance",
    "polarimetry": "--require-polarimetry-provenance",
    "astrometry": "--require-astrometry-provenance",
    "config_source_manifest": "--require-config-source-manifest",
}


class ValidationError(ValueError):
    pass


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def find_lowlevel_config(reduction: Path) -> Path:
    matches = sorted(reduction.glob("citlali_o*.yaml"))
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one citlali_o*.yaml in {reduction}; found {len(matches)}"
        )
    return matches[0]


def build_audit_command(
    profile: dict[str, Any], candidate: Path, json_out: Path
) -> list[str]:
    audit = profile["audit"]
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/baseline/audit_reduction_run.py"),
        str(candidate),
        "--json-out",
        str(json_out),
    ]
    expected_mode = audit.get("expected_mode", profile["mode"])
    if expected_mode:
        command.extend(["--expected-mode", expected_mode])
    if audit.get("expected_label"):
        command.extend(["--expected-label", audit["expected_label"]])
    for name in audit["required_provenance"]:
        command.append(PROVENANCE_FLAGS[name])
    return command


def build_config_command(
    profile: dict[str, Any],
    baseline: Path,
    candidate: Path,
    json_out: Path,
    binding_policies: Path = DEFAULT_BINDING_POLICIES,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/config/compare_lowlevel_yaml.py"),
        str(find_lowlevel_config(baseline)),
        str(find_lowlevel_config(candidate)),
        "--json-out",
        str(json_out),
    ]
    for path in profile["config"].get("ignore_paths", []):
        command.extend(["--ignore", path])
    if profile["config"]["policy"] == "exact_except_bindings":
        command.extend(
            [
                "--binding-policy-registry",
                str(binding_policies),
                "--binding-policy",
                profile["config"]["binding_policy_id"],
            ]
        )
    return command


def build_contract_command(
    profile: dict[str, Any],
    candidate: Path,
    json_out: Path,
    registry_path: Path = DEFAULT_PRODUCT_CONTRACTS,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools/baseline/validate_product_contract.py"),
        str(candidate),
        "--contract",
        profile["product_contract_id"],
        "--registry",
        str(registry_path),
        "--json-out",
        str(json_out),
    ]


def build_product_command(
    profile: dict[str, Any], baseline: Path, candidate: Path, json_out: Path
) -> list[str]:
    products = profile["products"]
    comparator = products["comparator"]
    if comparator == "reduction_products":
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/baseline/compare_reduction_products.py"),
            str(baseline),
            str(candidate),
            "--mode",
            profile["mode"],
            "--max-array-elements",
            str(products["max_array_elements"]),
            "--frac-floor",
            str(products["frac_floor"]),
            "--atol",
            str(products["atol"]),
            "--rtol",
            str(products["rtol"]),
            "--json-out",
            str(json_out),
        ]
        for pattern in products.get("include", []):
            command.extend(["--include", pattern])
        for pattern in products.get("exclude", []):
            command.extend(["--exclude", pattern])
        if products.get("include_timestream"):
            command.append("--include-timestream")
        if products.get("strict"):
            command.append("--strict")
        return command

    script_names = {
        "science_scientific_equivalence": "compare_science_scientific_equivalence.py",
        "beammap_scientific_equivalence": "compare_beammap_scientific_equivalence.py",
    }
    return [
        sys.executable,
        str(REPO_ROOT / "tools/baseline" / script_names[comparator]),
        str(baseline),
        str(candidate),
        "--profile",
        str(resolve_repo_path(products["scientific_profile"])),
        "--json-out",
        str(json_out),
    ]


def _read_gate_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else None


def run_gate(name: str, command: list[str], json_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "name": name,
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": command,
        "result": _read_gate_result(json_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_validation(
    profile: dict[str, Any],
    baseline: Path,
    candidate: Path,
    output_dir: Path,
    product_contracts: Path = DEFAULT_PRODUCT_CONTRACTS,
    binding_policies: Path = DEFAULT_BINDING_POLICIES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_json = output_dir / "audit.json"
    config_json = output_dir / "config.json"
    contract_json = output_dir / "contract.json"
    products_json = output_dir / "products.json"
    gates = [
        run_gate(
            "audit",
            build_audit_command(profile, candidate, audit_json),
            audit_json,
        ),
        run_gate(
            "config",
            build_config_command(
                profile,
                baseline,
                candidate,
                config_json,
                binding_policies,
            ),
            config_json,
        ),
        run_gate(
            "contract",
            build_contract_command(
                profile, candidate, contract_json, product_contracts
            ),
            contract_json,
        ),
        run_gate(
            "products",
            build_product_command(profile, baseline, candidate, products_json),
            products_json,
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "profile_status": profile["status"],
        "epoch_id": profile["epoch_id"],
        "mode": profile["mode"],
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
    }


def gate_detail(gate: dict[str, Any]) -> str:
    result = gate.get("result")
    if isinstance(result, dict):
        if gate["name"] == "audit":
            issue_counts = result.get("log", {}).get("issue_counts", {})
            return f"logged issues={sum(issue_counts.values())}"
        if gate["name"] == "config":
            summary = result.get("summary", {})
            detail = (
                f"leaves={summary.get('candidate_leaf_count', 'unknown')} "
                f"differences={summary.get('diff_count', 'unknown')}"
            )
            if summary.get("binding_policy_id"):
                detail += (
                    f" binding_matches="
                    f"{summary.get('binding_match_count', 'unknown')}"
                )
            return detail
        if gate["name"] == "contract":
            return (
                f"classified={result.get('classified_product_count', 'unknown')}/"
                f"{result.get('product_count', 'unknown')} "
                f"families={len(result.get('entry_results', []))} "
                f"errors={len(result.get('errors', []))}"
            )
        if gate["name"] == "products":
            if "changed_record_count" in result:
                return (
                    f"products={result.get('common_product_count', 'unknown')} "
                    f"records={result.get('record_count', 'unknown')} "
                    f"changed={result.get('changed_record_count', 'unknown')} "
                    f"skipped={result.get('skipped_record_count', 'unknown')}"
                )
            failures = result.get("failures", [])
            return (
                f"scientific verdict={result.get('verdict', 'unknown')} "
                f"failures={len(failures)}"
            )
    diagnostic = gate.get("stderr", "").strip() or gate.get("stdout", "").strip()
    return diagnostic.splitlines()[-1] if diagnostic else "no result produced"


def render_markdown(result: dict[str, Any]) -> str:
    if result["passed"] and result.get("profile_status") == "preparing":
        verdict = "prepared gates pass (not accepted)"
    elif result["passed"]:
        verdict = "accepted"
    else:
        verdict = "rejected"
    lines = [
        "# Citlali Profile Validation",
        "",
        f"- Profile: `{result['profile_id']}`",
        f"- Epoch: `{result['epoch_id']}`",
        f"- Mode: `{result['mode']}`",
        f"- Baseline: `{result['baseline']}`",
        f"- Candidate: `{result['candidate']}`",
        f"- Profile status: `{result.get('profile_status', 'active')}`",
        f"- Verdict: **{verdict}**",
        "",
        "## Gates",
        "",
    ]
    for gate in result["gates"]:
        state = "pass" if gate["passed"] else "FAIL"
        lines.append(
            f"- `{gate['name']}`: **{state}** (exit {gate['exit_code']}); "
            f"{gate_detail(gate)}"
        )
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--profile", default="")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "validation/validation_profiles.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "validation/accepted_runs.json",
    )
    parser.add_argument(
        "--product-contracts",
        type=Path,
        default=DEFAULT_PRODUCT_CONTRACTS,
    )
    parser.add_argument(
        "--binding-policies",
        type=Path,
        default=DEFAULT_BINDING_POLICIES,
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--list-preparing-profiles", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Retain the four delegated gate results in this directory.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        registry = validate_registry(args.registry, args.ledger)
        if args.list_profiles or args.list_preparing_profiles:
            expected_status = (
                "preparing" if args.list_preparing_profiles else "active"
            )
            selected_epochs = (
                set(registry["preparing_epoch_ids"])
                if args.list_preparing_profiles
                else {registry["active_epoch_id"]}
            )
            for profile in sorted(
                registry["profiles"],
                key=lambda item: (item["epoch_id"], item["mode"]),
            ):
                if (
                    profile["status"] == expected_status
                    and profile["epoch_id"] in selected_epochs
                ):
                    print(
                        f"{profile['epoch_id']}\t{profile['profile_id']}\t"
                        f"{profile['mode']}\t"
                        f"{profile['baseline_record_id'] or 'pending'}"
                    )
            return 0
        if not args.profile:
            raise ValidationError("--profile is required")
        if args.candidate is None:
            raise ValidationError("candidate reduction directory is required")
        profile = profile_by_id(registry, args.profile)
        if profile["status"] == "superseded":
            raise ValidationError(f"profile {args.profile!r} is superseded")
        records = ledger_records(args.ledger)
        baseline_id = profile.get("baseline_record_id")
        if args.baseline is not None:
            baseline_value = args.baseline
        elif baseline_id:
            baseline_record = records[baseline_id]
            baseline_value = Path(
                baseline_record["artifacts"]["candidate_local_path"]
            )
        else:
            raise ValidationError(
                f"preparing profile {args.profile!r} has no accepted baseline; "
                "pass --baseline explicitly"
            )
        baseline = baseline_value.expanduser().resolve()
        candidate = args.candidate.expanduser().resolve()
        if not baseline.is_dir():
            raise ValidationError(
                f"baseline directory not found: {baseline}; pass --baseline for this host"
            )
        if not candidate.is_dir():
            raise ValidationError(f"candidate directory not found: {candidate}")

        if args.output_dir:
            result = run_validation(
                profile,
                baseline,
                candidate,
                args.output_dir.expanduser().resolve(),
                args.product_contracts.expanduser().resolve(),
                args.binding_policies.expanduser().resolve(),
            )
        else:
            with tempfile.TemporaryDirectory(prefix="citlali-validation-") as directory:
                result = run_validation(
                    profile,
                    baseline,
                    candidate,
                    Path(directory),
                    args.product_contracts.expanduser().resolve(),
                    args.binding_policies.expanduser().resolve(),
                )
        report = render_markdown(result)
        if args.json_out:
            write_text(
                args.json_out.expanduser(),
                json.dumps(result, indent=2, sort_keys=True) + "\n",
            )
        if args.report_out:
            write_text(args.report_out.expanduser(), report)
        print(report, end="")
        return 0 if result["passed"] else 1
    except (OSError, json.JSONDecodeError, RegistryError, ValidationError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
