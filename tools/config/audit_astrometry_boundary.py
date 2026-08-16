#!/usr/bin/env python3
"""Audit the observation-indexed astrometry config and provenance boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


READER_SOURCE = "include/citlali/core/pipeline/pointing_offsets_config_read.h"
INSTALL_SOURCE = "include/citlali/core/pipeline/observation_calibration_config.h"
PLAN_SOURCE = "include/citlali/core/pipeline/astrometry_execution_plan.h"
APPLICATION_SOURCE = (
    "include/citlali/core/pipeline/telescope_pointing_operations.h"
)
INTERPOLATION_SOURCE = (
    "include/citlali/core/engine/detail/todproc_pointing_impl.h"
)
NATIVE_POINTING_SOURCE = "include/citlali/core/pipeline/telescope_pointing_operations.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/astrometry_provenance.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"


def count(text: str, token: str) -> int:
    return text.count(token)


def audit(repo_root: Path) -> dict[str, object]:
    reader = (repo_root / READER_SOURCE).read_text(encoding="utf-8")
    install = (repo_root / INSTALL_SOURCE).read_text(encoding="utf-8")
    plan = (repo_root / PLAN_SOURCE).read_text(encoding="utf-8")
    application = (repo_root / APPLICATION_SOURCE).read_text(encoding="utf-8")
    interpolation = (repo_root / INTERPOLATION_SOURCE).read_text(
        encoding="utf-8"
    )
    native_pointing = (repo_root / NATIVE_POINTING_SOURCE).read_text(
        encoding="utf-8"
    )
    provenance = (repo_root / PROVENANCE_SOURCE).read_text(encoding="utf-8")
    cli = (repo_root / CLI_SOURCE).read_text(encoding="utf-8")

    boundary = {
        "typed_reader_definitions": count(reader, "read_astrometry_config("),
        "typed_reader_calls": count(install, "read_astrometry_config("),
        "typed_validation_calls": count(
            install, "require_valid_astrometry_config("
        ),
        "forward_adapter_calls": count(install, "install_astrometry_config("),
        "request_record_calls": count(install, "record_astrometry_request("),
        "install_record_calls": count(install, "record_astrometry_installed("),
        "application_record_calls": count(
            application, "record_astrometry_applied("
        ),
        "completion_record_calls": count(
            cli, "record_astrometry_reduction_completed("
        ),
        "provenance_write_calls": count(
            cli, "write_astrometry_provenance_file("
        ),
        "reverse_mirror_occurrences": sum(
            text.count("mirror_typed_pointing_offsets")
            for text in (reader, install, plan, application, interpolation)
        ),
        "process_exit_occurrences": sum(
            text.count(token)
            for text in (reader, install, plan, application, interpolation)
            for token in ("std::exit", "EXIT_FAILURE")
        ),
    }
    boundary["exact"] = bool(
        boundary
        == {
            "typed_reader_definitions": 1,
            "typed_reader_calls": 1,
            "typed_validation_calls": 1,
            "forward_adapter_calls": 1,
            "request_record_calls": 1,
            "install_record_calls": 1,
            "application_record_calls": 1,
            "completion_record_calls": 1,
            "provenance_write_calls": 1,
            "reverse_mirror_occurrences": 0,
            "process_exit_occurrences": 0,
        }
    )

    contract_tokens = (
        "AstrometryApplicationMode::constant",
        "AstrometryApplicationMode::observation_span_linear",
        "AstrometryApplicationMode::explicit_mjd_linear",
        "astrometry observations must be registered in order",
        "astrometry observation lifecycle is incomplete",
    )
    plan_state = {
        "contract_tokens_present": all(token in plan for token in contract_tokens),
        "observation_indexed": "std::vector<AstrometryObservationPlan>" in plan,
        "requested_effective_realized": all(
            token in plan
            for token in ("requested", "effective", "resolution", "realized")
        ),
    }
    plan_state["exact"] = all(plan_state.values())

    provenance_state = {
        "schema": "citlali-astrometry-provenance-v1" in provenance,
        "tolteca_selection_authority": (
            'root["authority"]["calibration_selection"] = "tolteca"'
            in provenance
        ),
        "citlali_application_authority": (
            'root["authority"]["application"] = "citlali"' in provenance
        ),
        "origin_not_overclaimed": (
            'root["authority"]["support_origin_metadata_available"] = false'
            in provenance
        ),
        "atomic": "write_yaml_file_atomic" in provenance,
        "completion_required": "!plan.reduction_completed" in provenance,
    }
    provenance_state["exact"] = all(provenance_state.values())

    interpolation_state = {
        "native_offset_model": "make_native_pointing_offset_model(" in interpolation,
        "native_trajectory_boundary": all(
            token in native_pointing
            for token in (
                "native_consumer_plan",
                "native_pointing_plan",
                "build_native_pointing_plan_candidate",
                "NativePointingPlan",
            )
        ),
        "incomplete_native_state_fails_closed": (
            "network-native pointing state is incomplete before evaluation" in native_pointing
            and "has_native_alignment != has_raw_telescope" in native_pointing
        ),
        "no_common_time_compatibility_fallback": (
            "no common-time compatibility" in native_pointing
            and "candidate_native_pointing" in native_pointing
        ),
    }
    interpolation_state["exact"] = all(interpolation_state.values())

    drift = not bool(
        boundary["exact"]
        and plan_state["exact"]
        and provenance_state["exact"]
        and interpolation_state["exact"]
    )
    return {
        "schema_version": "citlali-astrometry-boundary-audit-v1",
        "boundary": boundary,
        "execution_plan": plan_state,
        "provenance": provenance_state,
        "interpolation": interpolation_state,
        "drift": drift,
    }


def markdown(result: dict[str, object]) -> str:
    return "\n".join(
        (
            "# Astrometry Boundary Audit",
            "",
            f"- Drift: `{result['drift']}`",
            f"- Boundary exact: `{result['boundary']['exact']}`",
            f"- Execution plan exact: `{result['execution_plan']['exact']}`",
            f"- Required provenance exact: `{result['provenance']['exact']}`",
            f"- Legacy interpolation contract exact: `{result['interpolation']['exact']}`",
            "",
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(Path(args.repo_root).resolve())
    report = markdown(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(report)
    print(
        "astrometry boundary: "
        f"plan={result['execution_plan']['exact']} "
        f"provenance={result['provenance']['exact']} "
        f"drift={result['drift']}"
    )
    return int(args.fail_on_drift and result["drift"])


if __name__ == "__main__":
    raise SystemExit(main())
