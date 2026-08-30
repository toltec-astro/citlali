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


SCHEMA = "citlali-wp7-rtc-filter-fixture-corpus-v1"


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
            "matched_detector_relation_available": record["apt_bundle"]["matched_detector_relation_available"],
            "ast_maximum_available": record["telescope_ast"]["maximum_available"],
            "ast_maximum_causes": record["telescope_ast"]["maximum_causes"],
            "network_count": len(record["network_native_census"]),
            "all_cadence_uncertainties_within_100ppm": all(
                item["cadence_uncertainty_within_100ppm"]
                for item in record["network_native_census"]
            ),
            "auxiliary_input_count": len(auxiliaries),
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
