#!/usr/bin/env python3
"""Audit the retired raw-timestream parser and typed authority boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


MANIFEST_SOURCE = "tools/config/raw_timestream_legacy_paths.json"
PARSER_SOURCE = "include/citlali/core/timestream/rtc/rtcproc.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/rtc_config_impl.h"
EXPECTED_MANIFEST_SCHEMA = "citlali-frozen-raw-config-paths-v1"
EXPECTED_PATH_COUNT = 171
EXPECTED_RAW_PATH_COUNT = 169
EXPECTED_POLARIMETRY_PATH_COUNT = 2
EXPECTED_PATH_SHA256 = (
    "5f10271aae40942ae1be587a105b70229b86c885f4d0bc4b02edcf312bc088c0"
)
TYPED_READER_SOURCES = (
    "include/citlali/core/pipeline/raw_filtering_config_read.h",
    "include/citlali/core/pipeline/raw_flagging_config_read.h",
    "include/citlali/core/pipeline/raw_line_audit_config_read.h",
)
RAW_SERIALIZER_SOURCE = (
    "include/citlali/core/pipeline/raw_timestream_config_serialization.h"
)
RAW_ADAPTER_SOURCES = (
    "include/citlali/core/pipeline/timestream_config_adapter_raw.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_filtering.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_flagging.h",
    "include/citlali/core/pipeline/timestream_config_adapter_raw_line_audit.h",
)
EXPECTED_DECLARED_TYPED_READER_PATH_COUNT = 156
EXPECTED_DECLARED_TYPED_READER_PATH_SHA256 = (
    "39f59b97de6ec9ae52b718c4ab8971485576b9d51264144e80678800a3e89a05"
)
COMPATIBILITY_ALIASES = {
    "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate."
    "candidate_sigma_scale":
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate."
        "candidate_rel_sigma_scale",
}
ADAPTER_MEMBER_ALIASES = {
    "timestream.raw_time_chunk.IIR_filter": "iir_filter",
    "timestream.raw_time_chunk.despike.legacy": "legacy_enabled",
}
LEGACY_PARSER_CALL = "read_processor_config"
TYPED_REQUEST_READ_CALL = "read_raw_timestream_request_config"
TYPED_AUTHORITY_INIT_CALL = "initialize_raw_timestream_authority"
TYPED_AUTHORITY_COMPARE_CALL = "compare_raw_timestream_authority"
POLARIMETRY_COMPATIBILITY_READ_CALL = (
    "read_legacy_polarimetry_runtime_config"
)
POLARIMETRY_RUNTIME_ADAPTER_CALL = "adapt_legacy_polarimetry_runtime"


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def family(path: str) -> str:
    if path.startswith("timestream.raw_time_chunk."):
        return "raw_timestream"
    if path.startswith("timestream.polarimetry."):
        return "polarimetry"
    return "unclassified"


def load_frozen_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    paths = manifest.get("paths")
    if not isinstance(paths, list) or not all(
        isinstance(item, str) for item in paths
    ):
        raise ValueError(f"invalid paths in {path}")
    if paths != sorted(set(paths)):
        raise ValueError(f"paths must be sorted and unique in {path}")
    return manifest


def manifest_state(manifest: dict[str, object]) -> dict[str, object]:
    paths = list(manifest["paths"])
    digest = path_digest(paths)
    counts = dict(sorted(Counter(map(family, paths)).items()))
    exact = (
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("retired_parser_source") == PARSER_SOURCE
        and manifest.get("path_count") == len(paths) == EXPECTED_PATH_COUNT
        and manifest.get("path_sha256") == digest == EXPECTED_PATH_SHA256
        and counts.get("raw_timestream") == EXPECTED_RAW_PATH_COUNT
        and counts.get("polarimetry") == EXPECTED_POLARIMETRY_PATH_COUNT
        and set(counts) == {"polarimetry", "raw_timestream"}
    )
    return {
        "source": MANIFEST_SOURCE,
        "schema_version": manifest.get("schema_version"),
        "retired_parser_source": manifest.get("retired_parser_source"),
        "literal_path_count": len(paths),
        "literal_path_sha256": digest,
        "family_counts": counts,
        "exact": exact,
        "paths": paths,
    }


def retired_parser_state(source_text: str) -> dict[str, object]:
    definition_count = len(
        re.findall(r"\bRTCProc::get_config\s*\(", source_text)
    )
    declaration_count = len(
        re.findall(r"\bvoid\s+get_config\s*\(", source_text)
    )
    return {
        "source": PARSER_SOURCE,
        "definition_count": definition_count,
        "declaration_count": declaration_count,
        "retired": definition_count == 0 and declaration_count == 0,
    }


def declared_typed_reader_paths(repo_root: Path) -> list[str]:
    paths: set[str] = set()
    for source in TYPED_READER_SOURCES:
        paths.update(
            re.findall(
                r'"(timestream\.raw_time_chunk\.[^"]+)"',
                (repo_root / source).read_text(),
            )
        )
    return sorted(paths)


def typed_reader_coverage(
    frozen_paths: list[str], declared_paths: list[str]
) -> tuple[list[str], list[str], list[str]]:
    declared = set(declared_paths)
    covered: list[str] = []
    uncovered: list[str] = []
    for frozen_path in frozen_paths:
        typed_path = COMPATIBILITY_ALIASES.get(frozen_path, frozen_path)
        destination = (
            covered
            if typed_path in declared
            or any(path.startswith(typed_path + ".") for path in declared)
            else uncovered
        )
        destination.append(frozen_path)
    frozen_typed_paths = {
        COMPATIBILITY_ALIASES.get(path, path) for path in frozen_paths
    }
    stale = sorted(
        path
        for path in declared
        if path not in frozen_typed_paths and path not in COMPATIBILITY_ALIASES
    )
    return sorted(covered), sorted(uncovered), stale


def serializer_coverage(
    frozen_paths: list[str], repo_root: Path
) -> tuple[list[str], list[str]]:
    source_text = (repo_root / RAW_SERIALIZER_SOURCE).read_text()
    covered: list[str] = []
    uncovered: list[str] = []
    for frozen_path in frozen_paths:
        typed_path = COMPATIBILITY_ALIASES.get(frozen_path, frozen_path)
        leaf = typed_path.rsplit(".", 1)[-1]
        destination = covered if f'"{leaf}"' in source_text else uncovered
        destination.append(frozen_path)
    return sorted(covered), sorted(uncovered)


def unsafe_yaml_string_view_assignment_lines(source_text: str) -> list[int]:
    pattern = re.compile(r"\]\s*=\s*citlali::config::to_string\s*\(")
    return [
        source_text.count("\n", 0, match.start()) + 1
        for match in pattern.finditer(source_text)
    ]


def adapter_coverage(
    frozen_paths: list[str], repo_root: Path
) -> tuple[list[str], list[str]]:
    source_text = "\n".join(
        (repo_root / source).read_text() for source in RAW_ADAPTER_SOURCES
    )
    covered: list[str] = []
    uncovered: list[str] = []
    for frozen_path in frozen_paths:
        typed_path = COMPATIBILITY_ALIASES.get(frozen_path, frozen_path)
        leaf = ADAPTER_MEMBER_ALIASES.get(
            frozen_path, typed_path.rsplit(".", 1)[-1]
        )
        destination = (
            covered
            if re.search(rf"\b{re.escape(leaf)}\b", source_text)
            else uncovered
        )
        destination.append(frozen_path)
    return sorted(covered), sorted(uncovered)


def call_positions(source_text: str, name: str) -> list[int]:
    return [
        match.start()
        for match in re.finditer(
            rf"\b{re.escape(name)}\s*\(", source_text
        )
    ]


def authority_boundary(source_text: str) -> dict[str, object]:
    typed_reads = call_positions(source_text, TYPED_REQUEST_READ_CALL)
    authority_inits = call_positions(source_text, TYPED_AUTHORITY_INIT_CALL)
    polarimetry_reads = call_positions(
        source_text, POLARIMETRY_COMPATIBILITY_READ_CALL
    )
    polarimetry_adapters = call_positions(
        source_text, POLARIMETRY_RUNTIME_ADAPTER_CALL
    )
    parser_calls = call_positions(source_text, LEGACY_PARSER_CALL)
    compare_calls = call_positions(source_text, TYPED_AUTHORITY_COMPARE_CALL)
    mirror_calls = re.findall(
        r"\b(mirror_raw_[A-Za-z0-9_]+)\s*\(", source_text
    )
    authority_order_exact = (
        len(typed_reads) == 1
        and len(polarimetry_reads) == 1
        and len(authority_inits) == 1
        and len(polarimetry_adapters) == 1
        and typed_reads[0] < polarimetry_reads[0]
        and polarimetry_reads[0] < authority_inits[0]
        and authority_inits[0] < polarimetry_adapters[0]
    )
    exact = (
        authority_order_exact
        and not parser_calls
        and not compare_calls
        and not mirror_calls
    )
    return {
        "source": BOUNDARY_SOURCE,
        "typed_request_read_call_count": len(typed_reads),
        "typed_authority_init_call_count": len(authority_inits),
        "legacy_parser_call_count": len(parser_calls),
        "typed_authority_compare_call_count": len(compare_calls),
        "legacy_to_typed_mirror_call_counts": dict(
            sorted(Counter(mirror_calls).items())
        ),
        "polarimetry_compatibility_read_call_count": len(
            polarimetry_reads
        ),
        "polarimetry_runtime_adapter_call_count": len(
            polarimetry_adapters
        ),
        "authority_order_exact": authority_order_exact,
        "exact": exact,
        "current_direction": (
            "requested_yaml -> typed_plan -> production_rtcproc; "
            "polarimetry_yaml -> named_compatibility_reader -> "
            "production_rtcproc"
        ),
        "target_direction": (
            "requested_yaml -> typed_plan -> production_rtcproc; "
            "migrate polarimetry under its separate authority decision"
        ),
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
    manifest = manifest_state(
        load_frozen_manifest(repo_root / MANIFEST_SOURCE)
    )
    paths = list(manifest["paths"])
    raw_paths = [path for path in paths if family(path) == "raw_timestream"]
    parser = retired_parser_state((repo_root / PARSER_SOURCE).read_text())
    boundary = authority_boundary((repo_root / BOUNDARY_SOURCE).read_text())
    declared_reader_paths = declared_typed_reader_paths(repo_root)
    declared_reader_digest = path_digest(declared_reader_paths)
    reader_covered, reader_uncovered, stale_reader_paths = (
        typed_reader_coverage(raw_paths, declared_reader_paths)
    )
    serializer_text = (repo_root / RAW_SERIALIZER_SOURCE).read_text()
    serialized, unserialized = serializer_coverage(raw_paths, repo_root)
    unsafe_yaml_assignments = unsafe_yaml_string_view_assignment_lines(
        serializer_text
    )
    adapted, unadapted = adapter_coverage(raw_paths, repo_root)
    drift = (
        not manifest["exact"]
        or not parser["retired"]
        or not boundary["exact"]
        or len(declared_reader_paths)
        != EXPECTED_DECLARED_TYPED_READER_PATH_COUNT
        or declared_reader_digest
        != EXPECTED_DECLARED_TYPED_READER_PATH_SHA256
        or bool(stale_reader_paths)
        or bool(reader_uncovered)
        or bool(unserialized)
        or bool(unsafe_yaml_assignments)
        or bool(unadapted)
    )
    result = {
        "schema_version": "citlali-raw-config-boundary-audit-v2",
        "manifest": manifest,
        "retired_parser": parser,
        "typed_authority_boundary": boundary,
        "path_or_boundary_drift": drift,
        "declared_typed_reader_path_count": len(declared_reader_paths),
        "declared_typed_reader_path_sha256": declared_reader_digest,
        "typed_reader_sources": list(TYPED_READER_SOURCES),
        "typed_reader_covered_frozen_path_count": len(reader_covered),
        "typed_reader_uncovered_paths": reader_uncovered,
        "stale_declared_typed_reader_paths": stale_reader_paths,
        "serializer_source": RAW_SERIALIZER_SOURCE,
        "serialized_path_count": len(serialized),
        "unserialized_paths": unserialized,
        "unsafe_yaml_string_view_assignment_lines": unsafe_yaml_assignments,
        "adapter_sources": list(RAW_ADAPTER_SOURCES),
        "adapter_covered_path_count": len(adapted),
        "unadapted_paths": unadapted,
        "note": (
            "The versioned manifest preserves the retired 169-path raw "
            "surface and two adjacent polarimetry paths. Production raw "
            "configuration is one-way typed authority. Polarimetry remains "
            "a separate named compatibility boundary."
        ),
    }
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        family_rows = "\n".join(
            f"| {name} | {count} |"
            for name, count in manifest["family_counts"].items()
        )
        output.write_text(
            "# Raw Timestream Boundary Audit\n\n"
            f"- Frozen paths: `{len(paths)}`\n"
            f"- Path digest: `{manifest['literal_path_sha256']}`\n"
            f"- Legacy parser retired: `{parser['retired']}`\n"
            f"- Typed authority boundary exact: `{boundary['exact']}`\n"
            f"- Frozen paths covered by typed readers: "
            f"`{len(reader_covered)}/{len(raw_paths)}`\n"
            f"- Frozen paths covered by request serializer: "
            f"`{len(serialized)}/{len(raw_paths)}`\n"
            f"- Frozen paths covered by typed-to-RTC adapter: "
            f"`{len(adapted)}/{len(raw_paths)}`\n"
            f"- Drift: `{drift}`\n\n"
            "| Family | Paths |\n| --- | ---: |\n"
            f"{family_rows}\n"
        )
    print(
        "raw config boundary: "
        f"paths={len(paths)} "
        f"raw={manifest['family_counts'].get('raw_timestream', 0)} "
        f"polarimetry={manifest['family_counts'].get('polarimetry', 0)} "
        f"parser_retired={parser['retired']} "
        f"typed_request_reads={boundary['typed_request_read_call_count']} "
        f"typed_authority_inits={boundary['typed_authority_init_call_count']} "
        f"legacy_parser_calls={boundary['legacy_parser_call_count']} "
        f"legacy_oracle_mirrors="
        f"{len(boundary['legacy_to_typed_mirror_call_counts'])} "
        f"polarimetry_reads="
        f"{boundary['polarimetry_compatibility_read_call_count']} "
        f"polarimetry_adapters="
        f"{boundary['polarimetry_runtime_adapter_call_count']} "
        f"typed_reader_coverage={len(reader_covered)}/{len(raw_paths)} "
        f"serialized={len(serialized)}/{len(raw_paths)} "
        f"adapted={len(adapted)}/{len(raw_paths)} drift={drift}"
    )
    return 1 if args.fail_on_drift and drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
