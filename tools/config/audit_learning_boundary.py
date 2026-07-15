#!/usr/bin/env python3
"""Audit the typed learning request, adapter, and provenance boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/learning_config_paths.json"
LEAF_CONTRACT_SOURCE = "tools/config/config_leaf_contract_resolved.json"
READER_SOURCE = "include/citlali/core/pipeline/learning_config_read.h"
ADAPTER_SOURCE = "include/citlali/core/pipeline/learning_config_adapter.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/learning_config_impl.h"
PLAN_SOURCE = "include/citlali/core/pipeline/processed_timestream_execution_plan.h"
SERIALIZATION_SOURCE = "include/citlali/core/pipeline/processed_timestream_config_serialization.h"
EXPECTED_SCHEMA = "citlali-frozen-learning-config-paths-v1"
EXPECTED_DIGEST = "adfc35ff0c164dc9744b7585c97e10102b8d1ff2a3ff7c3e333e03cfff673652"


def digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def tuple_paths(source: str) -> list[str]:
    paths: set[str] = set()
    for match in re.finditer(r"std::tuple\s*\{(.*?)\}", source, re.DOTALL):
        parts = re.findall(r'"([^"]+)"', match.group(1))
        path = ".".join(parts)
        if path.startswith("timestream.learning."):
            paths.add(path)
    return sorted(paths)


def audit(repo_root: Path) -> dict[str, object]:
    manifest = json.loads((repo_root / MANIFEST_SOURCE).read_text())
    paths = manifest.get("paths", [])
    leaf_contract = json.loads((repo_root / LEAF_CONTRACT_SOURCE).read_text())
    contract_paths = sorted(
        row["path"]
        for row in leaf_contract["leaves"]
        if row["authority"] == "learning"
    )
    reader = (repo_root / READER_SOURCE).read_text()
    adapter = (repo_root / ADAPTER_SOURCE).read_text()
    boundary = (repo_root / BOUNDARY_SOURCE).read_text()
    plan = (repo_root / PLAN_SOURCE).read_text()
    serialization = (repo_root / SERIALIZATION_SOURCE).read_text()
    reader_paths = tuple_paths(reader)
    adapter_members = set(re.findall(r"options\.([A-Za-z0-9_]+)", adapter))
    serialized_names = set(re.findall(r'node\["([A-Za-z0-9_]+)"\]', serialization))
    expected_names = {path.rsplit(".", 1)[-1] for path in paths}

    checks = {
        "manifest_exact": bool(
            manifest.get("schema_version") == EXPECTED_SCHEMA
            and manifest.get("path_count") == len(paths) == 28
            and paths == sorted(set(paths))
            and digest(paths) == manifest.get("path_sha256") == EXPECTED_DIGEST
        ),
        "leaf_contract_exact": contract_paths == paths,
        "typed_reader_exact": reader_paths == paths
        and "read_optional_mirrored_config_value" not in reader,
        "one_way_adapter_exact": len(adapter_members) == 28
        and "adapt_learning_config_one_way" in adapter
        and "make_learning_options" in adapter,
        "boundary_exact": boundary.count("read_learning_config(") == 1
        and boundary.count("adapt_learning_config_one_way(") == 1
        and "learning.configure(options)" not in boundary,
        "plan_exact": "TimestreamLearningConfig learning" in plan
        and "config.learning" in plan,
        "serialization_exact": expected_names <= serialized_names
        and 'node["learning"] = learning_config_node(snapshot.learning)' in serialization,
    }
    return {"checks": checks, "drift": not all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    result = audit(repo_root)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "learning config boundary: "
        f"paths=28 typed_reader={result['checks']['typed_reader_exact']} "
        f"adapter={result['checks']['one_way_adapter_exact']} "
        f"provenance={result['checks']['serialization_exact']} "
        f"drift={result['drift']}"
    )
    return 2 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
