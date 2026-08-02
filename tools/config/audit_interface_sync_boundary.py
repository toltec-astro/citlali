#!/usr/bin/env python3
"""Audit the typed interface-sync request, adapter, and provenance boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SOURCE = "tools/config/interface_sync_config_paths.json"
LEAF_CONTRACT_SOURCE = "tools/config/config_leaf_contract_resolved.json"
MODEL_SOURCE = "include/citlali/core/config/interface_sync_config.h"
READER_SOURCE = "include/citlali/core/pipeline/citlali_config_read_sync_offsets.h"
ADAPTER_SOURCE = "include/citlali/core/pipeline/interface_sync_config_adapter.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/citlali_config_impl.h"
PLAN_SOURCE = "include/citlali/core/pipeline/raw_timestream_execution_plan.h"
PROVENANCE_SOURCE = "include/citlali/core/pipeline/raw_timestream_provenance.h"
EXPECTED_SCHEMA = "citlali-frozen-interface-sync-config-paths-v1"
EXPECTED_DIGEST = "5b16c3e0faa1fa9f841c04c616c12513a3d6f935816d3e9274837dfc3b6c4e94"


def digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def audit(repo_root: Path) -> dict[str, object]:
    manifest = json.loads((repo_root / MANIFEST_SOURCE).read_text())
    paths = manifest.get("paths", [])
    contract = json.loads((repo_root / LEAF_CONTRACT_SOURCE).read_text())
    contract_rows = [
        row for row in contract["leaves"]
        if row["authority"] == "interface-sync"
    ]
    model = (repo_root / MODEL_SOURCE).read_text()
    reader = (repo_root / READER_SOURCE).read_text()
    adapter = (repo_root / ADAPTER_SOURCE).read_text()
    boundary = (repo_root / BOUNDARY_SOURCE).read_text()
    plan = (repo_root / PLAN_SOURCE).read_text()
    provenance = (repo_root / PROVENANCE_SOURCE).read_text()

    read_at = boundary.find("read_interface_sync_offsets(")
    adapt_at = boundary.find("adapt_interface_sync_config_one_way(")
    checks = {
        "manifest_exact": bool(
            manifest.get("schema_version") == EXPECTED_SCHEMA
            and manifest.get("path_count") == len(paths) == 14
            and paths == sorted(set(paths))
            and digest(paths) == manifest.get("path_sha256") == EXPECTED_DIGEST
        ),
        "leaf_contract_exact": sorted(row["path"] for row in contract_rows)
        == paths
        and all(
            row["allowed_domain"]
            == {"kind": "typed-real", "finite_required": True}
            for row in contract_rows
        ),
        "typed_model_exact": "toltec_interface_count = 13" in model
        and "toltec_offset_sec" in model
        and "hwpr_offset_sec" in model,
        "atomic_reader_exact": "InterfaceSyncOffsetConfig candidate" in reader
        and "if (clean)" in reader
        and "request = candidate" in reader
        and "entry.size() != 1" in reader
        and "configured_keys.insert" in reader
        and "std::isfinite" in reader,
        "one_way_adapter_exact": "adapt_interface_sync_config_one_way" in adapter
        and "offsets.clear()" in adapter
        and "toltec_offset_sec" in adapter
        and "hwpr_offset_sec" in adapter,
        "boundary_exact": 0 <= read_at < adapt_at
        and boundary.count("read_interface_sync_offsets(") == 1
        and boundary.count("adapt_interface_sync_config_one_way(") == 1,
        "plan_exact": "interface_sync_requested" in plan
        and "interface_sync_effective" in plan,
        "provenance_exact": "citlali-raw-timestream-provenance-v3" in provenance
        and "interface_sync_offset_config_node" in provenance
        and "interface_offset_lifecycle_node" in provenance
        and "interface_sync_requested" in provenance
        and "interface_sync_effective" in provenance
        and "observation_resolved_sec" in provenance
        and "realized_sec" in provenance
        and "applied_exactly_once" in provenance,
    }
    return {"checks": checks, "drift": not all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    result = audit(repo_root)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "interface-sync config boundary: "
        f"paths=14 reader={result['checks']['atomic_reader_exact']} "
        f"adapter={result['checks']['one_way_adapter_exact']} "
        f"provenance={result['checks']['provenance_exact']} "
        f"drift={result['drift']}"
    )
    return 2 if args.fail_on_drift and result["drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
