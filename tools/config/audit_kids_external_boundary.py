#!/usr/bin/env python3
"""Audit the deliberately external KIDs config and provenance boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


AUTHORITY_SOURCE = "tools/config/config_authority_inventory.json"
MODEL_SOURCE = "include/citlali/core/pipeline/kids_external_config.h"
PROVENANCE_SOURCE = (
    "include/citlali/core/pipeline/kids_external_provenance.h"
)
EXTERNAL_READER_SOURCE = "include/citlali/core/pipeline/kids_metadata.h"
IDENTITY_BOUNDARY_SOURCE = (
    "include/citlali/core/pipeline/kids_external_config.h"
)
ENGINE_BOUNDARY_SOURCE = (
    "include/citlali/core/engine/detail/citlali_config_impl.h"
)
PROCESSOR_SOURCE = "include/citlali/core/engine/kidsproc.h"
CLI_SOURCE = "include/citlali/core/cli/reduction_execution.h"
EXPECTED_TYPES = ("xs", "rs", "is", "qs")


def count(text: str, token: str) -> int:
    return text.count(token)


def audit(repo_root: Path) -> dict[str, object]:
    model = (repo_root / MODEL_SOURCE).read_text(encoding="utf-8")
    provenance = (repo_root / PROVENANCE_SOURCE).read_text(encoding="utf-8")
    external_reader = (repo_root / EXTERNAL_READER_SOURCE).read_text(
        encoding="utf-8"
    )
    identity_boundary = (repo_root / IDENTITY_BOUNDARY_SOURCE).read_text(
        encoding="utf-8"
    )
    engine_boundary = (repo_root / ENGINE_BOUNDARY_SOURCE).read_text(
        encoding="utf-8"
    )
    processor = (repo_root / PROCESSOR_SOURCE).read_text(encoding="utf-8")
    cli = (repo_root / CLI_SOURCE).read_text(encoding="utf-8")
    inventory = json.loads((repo_root / AUTHORITY_SOURCE).read_text())
    domain = next(
        item for item in inventory["domains"] if item["id"] == "kids-external"
    )

    supported_block = model.split(
        "supported_kids_tod_types{{", 1
    )[1].split("}};", 1)[0]
    type_counts = {
        name: count(
            supported_block,
            f"citlali::config::TodType::{name}",
        )
        for name in EXPECTED_TYPES
    }
    boundary = {
        "external_reader_calls": count(
            external_reader, 'get_config("kids")'
        ),
        "identity_reader_calls": count(
            identity_boundary, 'get_config("kids")'
        ),
        "plan_initialization_calls": count(
            engine_boundary, "make_kids_external_config_plan("
        ),
        "provenance_write_calls": count(
            cli, "write_kids_external_provenance_file("
        ),
        "type_counts": type_counts,
        "legacy_extra_output_global_count": count(
            processor, "bool extra_output"
        ),
        "effective_extra_output_constant_uses": count(
            processor, "kids_solver_extra_output_effective"
        ),
    }
    boundary["exact"] = bool(
        boundary["external_reader_calls"] == 1
        and boundary["identity_reader_calls"] == 1
        and boundary["plan_initialization_calls"] == 1
        and boundary["provenance_write_calls"] == 1
        and all(value == 1 for value in type_counts.values())
        and boundary["legacy_extra_output_global_count"] == 0
        and boundary["effective_extra_output_constant_uses"] == 1
    )
    schema = {
        "config": "citlali-kidscpp-bridge-v1" in model,
        "provenance": "citlali-kids-external-provenance-v1" in provenance,
        "atomic": "write_yaml_file_atomic" in provenance,
        "dependency_identity": 'root["dependency"]' in provenance,
        "requested_effective": all(
            token in provenance
            for token in ('root["requested"]', 'root["effective"]')
        ),
    }
    schema["exact"] = all(schema.values())
    inventory_state = {
        "domain": domain,
        "exact": bool(
            domain["execution_authority"] == "external"
            and domain["adapter_direction"] == "external"
            and domain["migration_status"] == "external-boundary"
            and domain["provenance_status"] == "complete"
            and domain["loader"] == IDENTITY_BOUNDARY_SOURCE
        ),
    }
    drift = not (
        boundary["exact"] and schema["exact"] and inventory_state["exact"]
    )
    return {
        "boundary": boundary,
        "schema": schema,
        "inventory": inventory_state,
        "drift": drift,
    }


def markdown(result: dict[str, object]) -> str:
    boundary = result["boundary"]
    return "\n".join(
        (
            "# KIDs External Config Boundary Audit",
            "",
            f"- Drift: `{result['drift']}`",
            f"- External reader calls: `{boundary['external_reader_calls']}`",
            f"- Identity reader calls: `{boundary['identity_reader_calls']}`",
            f"- Supported TOD types: `{', '.join(EXPECTED_TYPES)}`",
            f"- Required provenance write calls: `{boundary['provenance_write_calls']}`",
            f"- Inventory provenance: `{result['inventory']['domain']['provenance_status']}`",
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
        "KIDs external boundary: "
        f"types={len(EXPECTED_TYPES)} provenance=required-atomic-v1 "
        f"drift={result['drift']}"
    )
    return int(args.fail_on_drift and result["drift"])


if __name__ == "__main__":
    raise SystemExit(main())
