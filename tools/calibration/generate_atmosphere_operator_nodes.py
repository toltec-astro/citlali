#!/usr/bin/env python3
"""Validate SCI-CAL-001 source artifacts and generate the embedded node table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


CONTRACT_SHA256 = "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a"
NODES_SHA256 = "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f"
OPERATOR_ID = "am12_fixed_djf25_piecewise_linear_los_tau_v1"
PASSBAND_SET_ID = (
    "toltec-passband-set-v1:sha256:"
    "5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433"
)
PROFILE_ID = (
    "LMT_DJF_25.amc:sha256:"
    "aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866"
)
ARRAYS = ("a1100", "a1400", "a2000")
ALPHAS = (-1, 0, 2, 4)
ANCHORS = (
    ("am_q25", "0.0504874104674104401", tuple(range(20, 81, 2))),
    ("am_q50", "0.0883393725904400573", tuple(range(20, 81, 2))),
    ("tau015", "0.15", (25, 35, 45, 55, 65, 75, 80)),
    ("am_q75", "0.158313198574890929", tuple(range(20, 81, 2))),
    ("tau020", "0.20", (25, 35, 45, 55, 65, 75, 80)),
    ("tau025", "0.25", (25, 35, 45, 55, 65, 75, 80)),
)
COLUMNS = (
    "anchor_id",
    "source_profile",
    "tau225",
    "elevation_deg",
    "passband_id",
    "array",
    "alpha",
    "line_of_sight_optical_depth",
    "extinction_correction",
    "provenance",
)


class ArtifactError(ValueError):
    """Raised when frozen SCI-CAL-001 bytes or schema are invalid."""


@dataclass(frozen=True)
class Node:
    anchor_id: str
    tau225: float
    elevation_deg: float
    array: str
    alpha: int
    los_tau: float


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def validate_contract(path: Path) -> dict[str, object]:
    _require(sha256(path) == CONTRACT_SHA256, "operator contract SHA-256 mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    _require(contract.get("operator_id") == OPERATOR_ID, "unexpected operator_id")
    _require(
        contract.get("schema_version")
        == "sci-cal-001-fixed-djf25-full-domain-operator-contract-v1",
        "unexpected contract schema_version",
    )
    domain = contract.get("domain", {})
    _require(domain.get("tau225") == "0 <= tau225 <= 0.25", "unexpected tau225 domain")
    _require(
        domain.get("elevation_deg") == "25 <= elevation_deg <= 80",
        "unexpected elevation domain",
    )
    interpolation = contract.get("interpolation", {})
    _require(
        interpolation.get("opacity")
        == "piecewise linear in LOS optical depth through ordered anchors",
        "unexpected opacity interpolation",
    )
    spectral = contract.get("spectral_reference", {})
    _require(spectral.get("supported_values") == list(ALPHAS), "unexpected alpha set")
    provenance = contract.get("provenance", {})
    _require(
        provenance.get("operator_nodes_csv_sha256") == NODES_SHA256,
        "contract node digest mismatch",
    )
    return contract


def validate_nodes(path: Path) -> list[Node]:
    _require(sha256(path) == NODES_SHA256, "operator node SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == COLUMNS, "unexpected node schema")
        rows = list(reader)
    _require(len(rows) == 1368, "unexpected node row count")

    by_key: dict[tuple[str, str, int], list[Node]] = {}
    anchor_tau = {anchor: float(tau) for anchor, tau, _ in ANCHORS}
    expected_elevations = {anchor: elevations for anchor, _, elevations in ANCHORS}
    for row_number, row in enumerate(rows, start=2):
        anchor = row["anchor_id"]
        array = row["array"]
        try:
            alpha = int(row["alpha"])
            tau225 = float(row["tau225"])
            elevation = float(row["elevation_deg"])
            los_tau = float(row["line_of_sight_optical_depth"])
            correction = float(row["extinction_correction"])
        except ValueError as error:
            raise ArtifactError(f"non-numeric node field at row {row_number}") from error
        _require(anchor in anchor_tau, f"unexpected anchor at row {row_number}")
        _require(array in ARRAYS, f"unexpected array at row {row_number}")
        _require(alpha in ALPHAS, f"unexpected alpha at row {row_number}")
        _require(row["source_profile"] == "LMT_DJF_25", "unexpected profile")
        _require(math.isclose(tau225, anchor_tau[anchor], rel_tol=0.0, abs_tol=0.0), "anchor tau mismatch")
        _require(math.isfinite(elevation), "non-finite elevation")
        _require(math.isfinite(los_tau) and los_tau > 0.0, "invalid LOS optical depth")
        _require(math.isfinite(correction) and correction > 1.0, "invalid correction")
        _require(
            math.isclose(math.exp(los_tau), correction, rel_tol=5.0e-15, abs_tol=0.0),
            "correction/LOS optical-depth mismatch",
        )
        node = Node(anchor, tau225, elevation, array, alpha, los_tau)
        by_key.setdefault((anchor, array, alpha), []).append(node)

    expected_keys = {
        (anchor, array, alpha)
        for anchor, _, _ in ANCHORS
        for array in ARRAYS
        for alpha in ALPHAS
    }
    _require(set(by_key) == expected_keys, "node surface inventory mismatch")
    ordered: list[Node] = []
    for array in ARRAYS:
        for alpha in ALPHAS:
            previous: list[float] | None = None
            for anchor, _, _ in ANCHORS:
                nodes = sorted(by_key[(anchor, array, alpha)], key=lambda item: item.elevation_deg)
                actual = tuple(int(node.elevation_deg) for node in nodes)
                _require(actual == expected_elevations[anchor], "elevation lattice mismatch")
                _require(len({node.elevation_deg for node in nodes}) == len(nodes), "duplicate node")
                values = [node.los_tau for node in nodes]
                _require(all(a > b for a, b in zip(values, values[1:])), "LOS tau is not elevation-monotone")
                if previous is not None and len(previous) == len(values):
                    _require(all(a < b for a, b in zip(previous, values)), "LOS tau is not opacity-monotone")
                previous = values
                ordered.extend(nodes)
    _require(len(ordered) == len(rows), "ordered node row count mismatch")
    return ordered


def _cpp_float(value: float) -> str:
    return f"{value:.17e}"


def render_header(nodes: list[Node]) -> str:
    descriptors: list[str] = []
    elevations: list[str] = []
    optical_depths: list[str] = []
    offset = 0
    cursor = 0
    for array_index, array in enumerate(ARRAYS):
        for alpha in ALPHAS:
            for anchor, tau_text, expected in ANCHORS:
                count = len(expected)
                series = nodes[cursor : cursor + count]
                cursor += count
                _require(
                    all(
                        node.array == array and node.alpha == alpha and node.anchor_id == anchor
                        for node in series
                    ),
                    "render order mismatch",
                )
                descriptors.append(
                    "        {"
                    f"{array_index}, {alpha}, {_cpp_float(float(tau_text))}, "
                    f"{offset}U, {count}U, \"{anchor}\""
                    "},"
                )
                elevations.extend(f"        {_cpp_float(node.elevation_deg)}," for node in series)
                optical_depths.extend(f"        {_cpp_float(node.los_tau)}," for node in series)
                offset += count
    _require(cursor == len(nodes), "unrendered nodes")
    return f'''#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace timestream::atmosphere_nodes {{

// Generated by tools/calibration/generate_atmosphere_operator_nodes.py from
// the immutable SCI-CAL-001 contract and node artifact.
inline constexpr std::string_view contract_sha256 = "{CONTRACT_SHA256}";
inline constexpr std::string_view nodes_sha256 = "{NODES_SHA256}";
inline constexpr std::string_view operator_id = "{OPERATOR_ID}";
inline constexpr std::string_view passband_set_id = "{PASSBAND_SET_ID}";
inline constexpr std::string_view reference_profile_id = "{PROFILE_ID}";

struct SeriesDescriptor {{
    int array_index;
    int alpha;
    double tau225;
    std::size_t offset;
    std::size_t count;
    std::string_view anchor_id;
}};

inline constexpr std::array<SeriesDescriptor, {len(descriptors)}> series{{{{
{chr(10).join(descriptors)}
    }}}};

inline constexpr std::array<double, {len(elevations)}> elevation_deg{{{{
{chr(10).join(elevations)}
    }}}};

inline constexpr std::array<double, {len(optical_depths)}> los_optical_depth{{{{
{chr(10).join(optical_depths)}
    }}}};

}}  // namespace timestream::atmosphere_nodes
'''


def generate(contract_path: Path, nodes_path: Path) -> str:
    validate_contract(contract_path)
    return render_header(validate_nodes(nodes_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    contract = root / "data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_contract.json"
    nodes = root / "data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv"
    output = root / "include/citlali/core/timestream/atmosphere_operator_nodes_generated.h"
    content = generate(contract, nodes)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            print(f"generated atmosphere node header is stale: {output}")
            return 1
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    print(
        f"SCI-CAL-001 artifacts verified: contract={CONTRACT_SHA256} "
        f"nodes={NODES_SHA256} rows=1368 series=72"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
