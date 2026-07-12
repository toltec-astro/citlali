#!/usr/bin/env python3
"""Audit the legacy-authoritative raw-timestream config boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PARSER_SOURCE = "include/citlali/core/timestream/rtc/rtcproc.h"
BOUNDARY_SOURCE = "include/citlali/core/engine/detail/rtc_config_impl.h"
EXPECTED_PATH_COUNT = 171
EXPECTED_RAW_PATH_COUNT = 169
EXPECTED_POLARIMETRY_PATH_COUNT = 2
EXPECTED_PATH_SHA256 = (
    "5f10271aae40942ae1be587a105b70229b86c885f4d0bc4b02edcf312bc088c0"
)
EXPECTED_DIRECT_EXIT_COUNT = 0
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
TYPED_SHADOW_READ_CALL = "read_raw_timestream_request_config"
TYPED_SHADOW_COMPARE_CALL = "compare_raw_timestream_shadow"
LEGACY_TO_TYPED_MIRROR_CALLS = (
    "mirror_raw_despike_config",
    "mirror_raw_flagging_config",
    "mirror_raw_kernel_config",
    "mirror_raw_altaz_destripe_config",
    "mirror_raw_line_audit_config",
    "mirror_raw_downsample_config",
    "mirror_raw_filter_config",
    "mirror_raw_iir_filter_config",
    "mirror_raw_correction_flags",
    "mirror_raw_filter_edge_guard_config",
)


def parser_body(path: Path) -> str:
    text = path.read_text()
    start = text.index("void RTCProc::get_config")
    end = text.find("\ntemplate", start + 1)
    if end < 0:
        raise ValueError(f"unable to find end of RTCProc::get_config in {path}")
    return text[start:end]


def literal_paths(body: str) -> list[str]:
    tuple_pattern = re.compile(r"std::tuple\s*\{([^}]*)\}", re.DOTALL)
    string_pattern = re.compile(r'"([^"]+)"')
    paths = {
        ".".join(string_pattern.findall(match.group(1)))
        for match in tuple_pattern.finditer(body)
        if string_pattern.findall(match.group(1))
    }
    return sorted(paths)


def path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def family(path: str) -> str:
    if path.startswith("timestream.raw_time_chunk."):
        return "raw_timestream"
    if path.startswith("timestream.polarimetry."):
        return "polarimetry"
    return "unclassified"


def direct_exit_count(body: str) -> int:
    return len(re.findall(r"\bstd::exit\s*\(", body))


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
        path for path in declared
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
    pattern = re.compile(
        r"\]\s*=\s*citlali::config::to_string\s*\("
    )
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


def legacy_boundary(source_text: str) -> dict[str, object]:
    parser_positions = [
        match.start()
        for match in re.finditer(
            rf"\b{re.escape(LEGACY_PARSER_CALL)}\s*\(", source_text
        )
    ]
    shadow_read_positions = [
        match.start()
        for match in re.finditer(
            rf"\b{re.escape(TYPED_SHADOW_READ_CALL)}\s*\(", source_text
        )
    ]
    shadow_compare_positions = [
        match.start()
        for match in re.finditer(
            rf"\b{re.escape(TYPED_SHADOW_COMPARE_CALL)}\s*\(", source_text
        )
    ]
    observed_mirrors = Counter(
        re.findall(r"\b(mirror_raw_[A-Za-z0-9_]+)\s*\(", source_text)
    )
    expected_mirrors = set(LEGACY_TO_TYPED_MIRROR_CALLS)
    missing = sorted(expected_mirrors - observed_mirrors.keys())
    unexpected = sorted(observed_mirrors.keys() - expected_mirrors)
    repeated = {
        name: count for name, count in sorted(observed_mirrors.items())
        if count != 1
    }
    first_mirror = min(
        (
            source_text.index(name)
            for name in LEGACY_TO_TYPED_MIRROR_CALLS
            if name in source_text
        ),
        default=-1,
    )
    parser_precedes_mirrors = (
        len(parser_positions) == 1
        and first_mirror >= 0
        and parser_positions[0] < first_mirror
    )
    shadow_order_exact = (
        len(shadow_read_positions) == 1
        and len(parser_positions) == 1
        and first_mirror >= 0
        and len(shadow_compare_positions) == 1
        and shadow_read_positions[0] < parser_positions[0]
        and parser_positions[0] < first_mirror
        and first_mirror < shadow_compare_positions[0]
    )
    exact = (
        len(parser_positions) == 1
        and not missing
        and not unexpected
        and not repeated
        and parser_precedes_mirrors
        and shadow_order_exact
    )
    return {
        "source": BOUNDARY_SOURCE,
        "legacy_parser_call_count": len(parser_positions),
        "typed_shadow_read_call_count": len(shadow_read_positions),
        "typed_shadow_compare_call_count": len(shadow_compare_positions),
        "legacy_to_typed_mirror_call_counts": dict(sorted(observed_mirrors.items())),
        "missing_mirror_calls": missing,
        "unexpected_mirror_calls": unexpected,
        "non_unit_mirror_call_counts": repeated,
        "parser_precedes_mirrors": parser_precedes_mirrors,
        "shadow_order_exact": shadow_order_exact,
        "exact": exact,
        "current_direction": (
            "requested_yaml -> typed_shadow; requested_yaml -> legacy_rtcproc "
            "-> typed_snapshot; typed_shadow == legacy_snapshot gate"
        ),
        "target_direction": "requested_yaml -> typed_plan -> legacy_rtcproc",
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
    source = repo_root / PARSER_SOURCE
    body = parser_body(source)
    paths = literal_paths(body)
    digest = path_digest(paths)
    counts = dict(sorted(Counter(map(family, paths)).items()))
    exits = direct_exit_count(body)
    boundary = legacy_boundary((repo_root / BOUNDARY_SOURCE).read_text())
    raw_paths = [
        path for path in paths if family(path) == "raw_timestream"
    ]
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
        len(paths) != EXPECTED_PATH_COUNT
        or counts.get("raw_timestream") != EXPECTED_RAW_PATH_COUNT
        or counts.get("polarimetry") != EXPECTED_POLARIMETRY_PATH_COUNT
        or set(counts) != {"polarimetry", "raw_timestream"}
        or digest != EXPECTED_PATH_SHA256
        or exits != EXPECTED_DIRECT_EXIT_COUNT
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
        "schema_version": "citlali-raw-config-boundary-audit-v1",
        "source": PARSER_SOURCE,
        "literal_path_count": len(paths),
        "literal_path_sha256": digest,
        "expected_path_count": EXPECTED_PATH_COUNT,
        "expected_path_sha256": EXPECTED_PATH_SHA256,
        "family_counts": counts,
        "direct_process_exit_count": exits,
        "expected_direct_process_exit_count": EXPECTED_DIRECT_EXIT_COUNT,
        "path_or_boundary_drift": drift,
        "legacy_boundary": boundary,
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
        "paths": paths,
        "note": (
            "This is a characterization gate, not an approval of the current "
            "direction. The two polarimetry paths are recorded as an adjacent "
            "domain and are excluded from the raw-timestream migration. Direct "
            "parser exits are forbidden; legacy-to-typed mirrors remain known "
            "removal debt."
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
            f"| {name} | {count} |" for name, count in counts.items()
        )
        mirror_rows = "\n".join(
            f"| `{name}` | {count} |"
            for name, count in boundary[
                "legacy_to_typed_mirror_call_counts"
            ].items()
        )
        output.write_text(
            "# Raw Timestream Boundary Audit\n\n"
            f"- Literal paths: `{len(paths)}`\n"
            f"- Path digest: `{digest}`\n"
            f"- Direct parser exits: `{exits}`\n"
            f"- Boundary exact: `{boundary['exact']}`\n"
            f"- Typed shadow order exact: "
            f"`{boundary['shadow_order_exact']}`\n"
            f"- Declared direct typed-reader paths: "
            f"`{len(declared_reader_paths)}`\n"
            f"- Frozen paths covered by typed readers: "
            f"`{len(reader_covered)}/{len(raw_paths)}`\n"
            f"- Frozen paths covered by request serializer: "
            f"`{len(serialized)}/{len(raw_paths)}`\n"
            f"- Frozen paths covered by typed-to-RTC adapter: "
            f"`{len(adapted)}/{len(raw_paths)}`\n"
            f"- Drift: `{drift}`\n\n"
            "| Family | Paths |\n| --- | ---: |\n"
            f"{family_rows}\n\n"
            "| Legacy-to-typed mirror | Calls |\n| --- | ---: |\n"
            f"{mirror_rows}\n"
        )
    print(
        "raw config boundary: "
        f"paths={len(paths)} raw={counts.get('raw_timestream', 0)} "
        f"polarimetry={counts.get('polarimetry', 0)} exits={exits} "
        f"legacy_parser_calls={boundary['legacy_parser_call_count']} "
        f"typed_shadow_reads={boundary['typed_shadow_read_call_count']} "
        f"typed_shadow_compares={boundary['typed_shadow_compare_call_count']} "
        f"legacy_to_typed_mirrors={len(boundary['legacy_to_typed_mirror_call_counts'])} "
        f"typed_reader_coverage={len(reader_covered)}/{len(raw_paths)} "
        f"serialized={len(serialized)}/{len(raw_paths)} "
        f"adapted={len(adapted)}/{len(raw_paths)} "
        f"drift={drift}"
    )
    return 1 if args.fail_on_drift and drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
