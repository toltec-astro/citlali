#!/usr/bin/env python3
"""Audit the typed polarimetry capability boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST_SOURCE = "tools/config/polarimetry_config_paths.json"
READER_SOURCE = "include/citlali/core/pipeline/polarimetry_config_read.h"
PLAN_SOURCE = "include/citlali/core/pipeline/polarimetry_execution_plan.h"
ADAPTER_SOURCE = (
    "include/citlali/core/pipeline/timestream_config_adapter_polarimetry.h"
)
SERIALIZER_SOURCE = (
    "include/citlali/core/pipeline/polarimetry_config_serialization.h"
)
PROVENANCE_SOURCE = "include/citlali/core/pipeline/polarimetry_provenance.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/rtc_config_impl.h"
CLASSIFICATION_SOURCE = "tools/config/config_key_classification.yaml"
EXPECTED_SCHEMA = "citlali-frozen-polarimetry-config-paths-v1"
EXPECTED_PATHS = [
    "timestream.polarimetry.enabled",
    "timestream.polarimetry.grouping",
    "timestream.polarimetry.ignore_hwpr",
]
EXPECTED_PATH_SHA256 = (
    "07e30fc112500049a6564619a7a15c607ca86acdc661e40eff1d65faedc7448d"
)
RETIRED_TOKENS = (
    "read_legacy_polarimetry_runtime_config",
    "adapt_legacy_polarimetry_runtime",
    "mirror_polarimetry_config",
)


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    paths = data.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError(f"invalid paths in {path}")
    if paths != sorted(set(paths)):
        raise ValueError(f"paths must be sorted and unique in {path}")
    return data


def manifest_state(data: dict[str, object]) -> dict[str, object]:
    paths = list(data["paths"])
    digest = path_digest(paths)
    exact = (
        data.get("schema_version") == EXPECTED_SCHEMA
        and paths == EXPECTED_PATHS
        and data.get("path_count") == len(paths) == len(EXPECTED_PATHS)
        and data.get("path_sha256") == digest == EXPECTED_PATH_SHA256
    )
    return {
        "schema_version": data.get("schema_version"),
        "path_count": len(paths),
        "path_sha256": digest,
        "paths": paths,
        "exact": exact,
    }


def tuple_paths(source_text: str) -> list[str]:
    paths = []
    for body in re.findall(r"std::tuple\{([^}]+)\}", source_text):
        components = re.findall(r'"([^"]+)"', body)
        if components:
            paths.append(".".join(components))
    return sorted(set(paths))


def count_call(source_text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source_text))


def audit(repo_root: Path) -> dict[str, object]:
    manifest = manifest_state(load_manifest(repo_root / MANIFEST_SOURCE))
    reader_text = (repo_root / READER_SOURCE).read_text()
    plan_text = (repo_root / PLAN_SOURCE).read_text()
    adapter_text = (repo_root / ADAPTER_SOURCE).read_text()
    serializer_text = (repo_root / SERIALIZER_SOURCE).read_text()
    provenance_text = (repo_root / PROVENANCE_SOURCE).read_text()
    boundary_text = (repo_root / BOUNDARY_SOURCE).read_text()
    classification_text = (repo_root / CLASSIFICATION_SOURCE).read_text()

    reader_paths = tuple_paths(reader_text)
    reader_exact = reader_paths == EXPECTED_PATHS
    plan_exact = (
        "enabled_polarimetry_available = false" in plan_text
        and '"planned-unavailable"' in plan_text
        and "request.enabled && !enabled_polarimetry_available" in plan_text
    )
    adapter_exact = all(
        token in adapter_text
        for token in (
            "config.enabled",
            "config.grouping",
            "config.hwpr_policy",
            "stokes_params",
            "calib.ignore_hwpr",
        )
    )
    serializer_exact = all(
        f'node["{leaf}"]' in serializer_text
        for leaf in ("enabled", "grouping", "ignore_hwpr")
    )
    provenance_exact = all(
        token in provenance_text
        for token in (
            "citlali-polarimetry-provenance-v1",
            'root["requested"]',
            'root["effective"]',
            'root["realized"]',
            "write_yaml_file_atomic",
        )
    )
    classification_exact = bool(
        re.search(
            r"pattern:\s*timestream\.polarimetry\*\s+"
            r"classification:\s*expert\s+owner:\s*citlali",
            classification_text,
        )
    )
    retired_occurrences = {
        token: sum(
            text.count(token)
            for text in (reader_text, adapter_text, boundary_text)
        )
        for token in RETIRED_TOKENS
    }
    boundary_counts = {
        "typed_request_reads": count_call(
            boundary_text, "read_polarimetry_request_config"
        ),
        "plan_resets": count_call(boundary_text, "reset_from_request"),
        "forward_adapters": count_call(
            boundary_text, "adapt_polarimetry_config"
        ),
        "capability_diagnostics": count_call(
            boundary_text, "add_invalid_config_key"
        ),
    }
    boundary_exact = (
        boundary_counts
        == {
            "typed_request_reads": 1,
            "plan_resets": 1,
            "forward_adapters": 1,
            "capability_diagnostics": 1,
        }
        and not any(retired_occurrences.values())
    )
    drift = not all(
        (
            manifest["exact"],
            reader_exact,
            plan_exact,
            adapter_exact,
            serializer_exact,
            provenance_exact,
            classification_exact,
            boundary_exact,
        )
    )
    return {
        "schema_version": "citlali-polarimetry-boundary-audit-v1",
        "manifest": manifest,
        "reader": {
            "source": READER_SOURCE,
            "paths": reader_paths,
            "exact": reader_exact,
        },
        "capability_plan": {"source": PLAN_SOURCE, "exact": plan_exact},
        "adapter": {"source": ADAPTER_SOURCE, "exact": adapter_exact},
        "serializer": {
            "source": SERIALIZER_SOURCE,
            "exact": serializer_exact,
        },
        "provenance": {
            "source": PROVENANCE_SOURCE,
            "exact": provenance_exact,
        },
        "exposure_policy": {
            "source": CLASSIFICATION_SOURCE,
            "classification": "expert",
            "exact": classification_exact,
        },
        "boundary": {
            "source": BOUNDARY_SOURCE,
            "counts": boundary_counts,
            "retired_token_occurrences": retired_occurrences,
            "exact": boundary_exact,
        },
        "path_or_boundary_drift": drift,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[2]
    )
    result = audit(repo_root)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "# Polarimetry Boundary Audit\n\n"
            f"- Frozen paths: `{result['manifest']['path_count']}`\n"
            f"- Enabled capability available: `false`\n"
            f"- Typed reader exact: `{result['reader']['exact']}`\n"
            f"- One-way adapter exact: `{result['adapter']['exact']}`\n"
            f"- Required provenance exact: `{result['provenance']['exact']}`\n"
            f"- Exposure policy exact: `{result['exposure_policy']['exact']}`\n"
            f"- Drift: `{result['path_or_boundary_drift']}`\n"
        )
    print(
        "polarimetry boundary: "
        f"paths={result['manifest']['path_count']} "
        f"reader_exact={result['reader']['exact']} "
        f"adapter_exact={result['adapter']['exact']} "
        f"provenance_exact={result['provenance']['exact']} "
        f"boundary_exact={result['boundary']['exact']} "
        f"drift={result['path_or_boundary_drift']}"
    )
    return 1 if args.fail_on_drift and result["path_or_boundary_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
