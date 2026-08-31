#!/usr/bin/env python3
"""Run the bounded WP-7 D0/D1 fixture census over declared local cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import yaml


SCHEMA = "citlali-wp7-rtc-filter-fixture-corpus-v3"
RESULT_SCHEMA = "citlali-wp7-rtc-filter-fixture-census-v3"
NUMERICAL_POLICY_ID = "wp7-rtc-scan-array-numerical-policy-v2"
SPEED_ADMISSION_POLICY_ID = "wp7-rtc-occurrence-speed-admission-v1"
AST_POLICY_ID = "wp7-ast-scan-motion-v2"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(source_root: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_cases(path: pathlib.Path) -> list[dict]:
    with path.open() as stream:
        document = yaml.safe_load(stream)
    if document.get("schema") != "citlali-wp7-rtc-filter-fixture-cases-v1":
        raise RuntimeError("fixture case manifest schema is not supported")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("fixture case manifest has no cases")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise RuntimeError("fixture case id is missing")
    if len(set(ids)) != len(ids):
        raise RuntimeError("fixture case ids are not unique")
    supported_profiles = {
        "science-lissajous",
        "oof-lissajous",
        "pointing-lissajous",
        "rectilinear-continuous-beammap",
    }
    if any(case.get("ast_route_profile") not in supported_profiles for case in cases):
        raise RuntimeError("fixture case AST route profile is missing or unsupported")
    return cases


def resolve_file(data_root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    path = (data_root / relative).resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    return path


def resolve_directory(
    data_root: pathlib.Path, relative: str, label: str
) -> pathlib.Path:
    path = (data_root / relative).resolve()
    if not path.is_dir():
        raise RuntimeError(f"{label} is absent: {path}")
    return path


def auxiliary_inputs(data_root: pathlib.Path, case: dict) -> list[tuple[str, pathlib.Path]]:
    result: list[tuple[str, pathlib.Path]] = []
    seen: set[pathlib.Path] = set()
    for declaration in case.get("auxiliary_inputs", []):
        role = declaration.get("role")
        pattern = declaration.get("pattern")
        minimum = declaration.get("minimum_count")
        if not isinstance(role, str) or not role or not isinstance(pattern, str):
            raise RuntimeError(f"{case['id']} has a malformed auxiliary declaration")
        if not isinstance(minimum, int) or minimum < 0:
            raise RuntimeError(f"{case['id']} has an invalid auxiliary minimum")
        matches = sorted(path.resolve() for path in data_root.glob(pattern) if path.is_file())
        if len(matches) < minimum:
            raise RuntimeError(
                f"{case['id']} auxiliary role {role} found {len(matches)}, "
                f"requires at least {minimum}: {pattern}"
            )
        for path in matches:
            if path in seen:
                raise RuntimeError(f"{case['id']} repeats auxiliary input {path}")
            seen.add(path)
            result.append((role, path))
    return result


def validate_case_result(case: dict, record: dict) -> None:
    if record.get("schema") != RESULT_SCHEMA:
        raise RuntimeError(f"{case['id']} output schema is not supported")
    if record.get("numerical_policy_id") != NUMERICAL_POLICY_ID:
        raise RuntimeError(f"{case['id']} numerical policy identity changed")
    if record.get("speed_admission_policy_id") != SPEED_ADMISSION_POLICY_ID:
        raise RuntimeError(f"{case['id']} speed-admission policy identity changed")
    telescope_ast = record.get("telescope_ast", {})
    if telescope_ast.get("policy_id") != AST_POLICY_ID:
        raise RuntimeError(f"{case['id']} AST policy identity changed")
    if telescope_ast.get("route_profile") != case.get("ast_route_profile"):
        raise RuntimeError(f"{case['id']} AST route profile changed")
    if telescope_ast.get("maximum_available") is not True:
        raise RuntimeError(f"{case['id']} lacks a complete AST maximum")
    if telescope_ast.get("maximum_causes") != 0:
        raise RuntimeError(f"{case['id']} has unexpected AST maximum causes")
    if telescope_ast.get("physical_scan_member_count", 0) <= 0:
        raise RuntimeError(f"{case['id']} has no physical scan members")
    if telescope_ast.get("physical_segment_count", 0) <= 0:
        raise RuntimeError(f"{case['id']} has no physical segments")
    if telescope_ast.get("chunk_record_mismatch_count") != 0 or telescope_ast.get(
        "chunk_summary_matches"
    ) is not True:
        raise RuntimeError(f"{case['id']} AST product is not chunk invariant")
    expected = (case["observation"], case["subobservation"], case["scan"])
    actual = (record["observation"], record["subobservation"], record["scan"])
    if actual != expected:
        raise RuntimeError(f"{case['id']} output scope {actual} != {expected}")
    if record.get("common_analysis_grid_requested") is not False:
        raise RuntimeError(f"{case['id']} unexpectedly requested a common grid")
    if record.get("rtc_route_activated") is not False:
        raise RuntimeError(f"{case['id']} unexpectedly activated RTC")
    if record.get("mapping_checks", {}).get("identity_mismatch_count") != 0:
        raise RuntimeError(f"{case['id']} lost native timing identity")
    if record.get("mapping_checks", {}).get("missing_support_count") != 0:
        raise RuntimeError(f"{case['id']} lost available AST support")
    apt_bundle = record.get("apt_bundle", {})
    if apt_bundle.get("bundle_kind") not in {"baseline", "matched"}:
        raise RuntimeError(f"{case['id']} canonical APT bundle kind is invalid")
    if apt_bundle.get("canonical_bundle_verified") is not True:
        raise RuntimeError(f"{case['id']} canonical APT bundle is not verified")
    if apt_bundle.get("detector_raw_inventory_complete") is not True:
        raise RuntimeError(f"{case['id']} detector/raw inventory is incomplete")
    if record.get("d0_fixture_identity_ready") is not record.get(
        "source_clean_asserted"
    ):
        raise RuntimeError(f"{case['id']} D0 readiness disagrees with custody")
    if record.get("automatic_factor_selection_authorized") is not False:
        raise RuntimeError(f"{case['id']} unexpectedly authorized factor selection")
    domains = record.get("candidate_mode_domains")
    if not isinstance(domains, list) or not domains:
        raise RuntimeError(f"{case['id']} has no candidate mode domains")
    for domain in domains:
        if domain.get("automatic_factor_selection_authorized") is not False:
            raise RuntimeError(
                f"{case['id']} candidate domain authorized factor selection"
            )
        factors = domain.get("factor_candidates")
        if not isinstance(factors, list) or [item.get("factor") for item in factors] != list(range(1, 257)):
            raise RuntimeError(f"{case['id']} factor census is not exactly 1..256")
        for candidate in factors:
            if "planned_speed_arcsec_per_sec" in candidate:
                raise RuntimeError(f"{case['id']} retained scan-maximum planning")
            ceiling = candidate.get("upper_speed_ceiling_arcsec_per_sec")
            if not isinstance(ceiling, (int, float)) or ceiling <= 0:
                raise RuntimeError(f"{case['id']} has an invalid structural ceiling")
            if candidate.get("upper_boundary_inclusive") is not True:
                raise RuntimeError(f"{case['id']} made the upper boundary exclusive")
            if candidate.get("upper_speed_typed_cause") != "scan_speed_above_mode_support":
                raise RuntimeError(f"{case['id']} changed the upper-speed cause")
            networks = candidate.get("occurrence_admission_by_network")
            if not isinstance(networks, list) or not networks:
                raise RuntimeError(f"{case['id']} lacks network occurrence evidence")
            for network in networks:
                if (
                    network.get("base_admitted_count")
                    != network.get("upper_speed_admitted_count", 0)
                    + network.get("scan_speed_above_mode_support_count", 0)
                ):
                    raise RuntimeError(
                        f"{case['id']} occurrence admission does not partition the base"
                    )
            support = candidate.get("support_erosion", {})
            if candidate["factor"] == 1:
                if support.get("status") != "exact-occurrence-local-m1-no-filter" or support.get("support_eroded_output_count") != 0:
                    raise RuntimeError(f"{case['id']} M=1 support accounting is not exact")
            elif support.get("status") != "pending-exact-filter-coefficients-and-half-support" or support.get("support_eroded_output_count", "absent") is not None:
                raise RuntimeError(f"{case['id']} guessed M>1 support erosion")


def main() -> int:
    parser = argparse.ArgumentParser()
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser.add_argument("--executable", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--case-manifest",
        type=pathlib.Path,
        default=source_root / "validation" / "wp7_rtc_filter_fixture_cases_v1.yaml",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        raise RuntimeError(f"census executable is absent: {executable}")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise RuntimeError(f"data root is absent: {data_root}")
    cases = load_cases(args.case_manifest.resolve())
    if args.selected_cases:
        requested = set(args.selected_cases)
        known = {case["id"] for case in cases}
        unknown = requested - known
        if unknown:
            raise RuntimeError(f"unknown fixture cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in requested]

    revision = git_output(source_root, "rev-parse", "HEAD")
    source_clean = not git_output(
        source_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in cases:
        output = args.output_dir / f"{case['id']}.json"
        command = [
            str(executable),
            "--dataset-id", case["id"],
            "--data-dir", str(resolve_directory(data_root, case["raw_data_directory"], "raw data directory")),
            "--telescope", str(resolve_file(data_root, case["telescope"], "telescope")),
            "--apt-manifest", str(resolve_file(data_root, case["apt_manifest"], "APT manifest")),
            "--config", str(resolve_file(data_root, case["config"], "config")),
            "--output", str(output),
            "--source-revision", revision,
        ]
        if source_clean:
            command.append("--source-clean")
        auxiliaries = auxiliary_inputs(data_root, case)
        for role, path in auxiliaries:
            command.extend(["--auxiliary-input", role, str(path)])
        subprocess.run(command, check=True)
        with output.open() as stream:
            record = json.load(stream)
        validate_case_result(case, record)
        summaries.append({
            "id": case["id"],
            "role": case["role"],
            "observation": case["observation"],
            "result": output.name,
            "result_sha256": sha256_file(output),
            "apt_bundle_kind": record["apt_bundle"]["bundle_kind"],
            "canonical_apt_bundle_verified": record["apt_bundle"][
                "canonical_bundle_verified"
            ],
            "detector_raw_inventory_complete": record["apt_bundle"][
                "detector_raw_inventory_complete"
            ],
            "matched_detector_relation_available": record["apt_bundle"][
                "matched_detector_relation_available"
            ],
            "d0_fixture_identity_ready": record["d0_fixture_identity_ready"],
            "ast_policy_id": record["telescope_ast"]["policy_id"],
            "ast_route_profile": record["telescope_ast"]["route_profile"],
            "ast_maximum_available": record["telescope_ast"]["maximum_available"],
            "ast_maximum_causes": record["telescope_ast"]["maximum_causes"],
            "ast_physical_scan_member_count": record["telescope_ast"][
                "physical_scan_member_count"
            ],
            "ast_physical_segment_count": record["telescope_ast"][
                "physical_segment_count"
            ],
            "ast_derivative_valid_record_count": record["telescope_ast"][
                "derivative_valid_record_count"
            ],
            "ast_maximum_speed_arcsec_per_sec": record["telescope_ast"][
                "maximum_speed_arcsec_per_sec"
            ],
            "network_count": len(record["network_native_census"]),
            "all_cadence_uncertainties_within_100ppm": all(
                item["cadence_uncertainty_within_100ppm"]
                for item in record["network_native_census"]
            ),
            "auxiliary_input_count": len(auxiliaries),
            "candidate_mode_domain_count": len(record["candidate_mode_domains"]),
            "m1_raw_upper_speed_excluded_by_array": {
                domain["array"]: sum(
                    network["scan_speed_above_mode_support_count"]
                    for network in domain["factor_candidates"][0]["occurrence_admission_by_network"]
                )
                for domain in record["candidate_mode_domains"]
            },
        })

    corpus = {
        "schema": SCHEMA,
        "source_revision": revision,
        "source_clean": source_clean,
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "case_manifest": str(args.case_manifest.resolve()),
        "case_manifest_sha256": sha256_file(args.case_manifest.resolve()),
        "common_analysis_grid_requested": False,
        "rtc_route_activated": False,
        "automatic_factor_selection_authorized": False,
        "case_count": len(summaries),
        "cases": summaries,
    }
    corpus_path = args.output_dir / "corpus.json"
    with corpus_path.open("w") as stream:
        json.dump(corpus, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(corpus_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"WP-7 RTC filter corpus census failed: {error}", file=sys.stderr)
        raise SystemExit(1)
