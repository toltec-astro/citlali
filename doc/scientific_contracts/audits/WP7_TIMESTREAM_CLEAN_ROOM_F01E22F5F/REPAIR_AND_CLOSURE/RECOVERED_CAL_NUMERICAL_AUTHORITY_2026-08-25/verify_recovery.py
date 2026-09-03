#!/usr/bin/env python3
"""Verify the exact WP-7 recovered CAL numerical-authority staging bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
CAL_ROOT = (
    SOURCES
    / "citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01"
)
PASSBAND_ROOT = SOURCES / "tolteca/tolteca/data/cal/toltec_passband"

EXPECTED = {
    "sources/citlali/licenses/LICENSE":
        "8f46574eb73aa5ca78636c21f83a5cc2bbdf32793a6f563d6463b4103ca2df9b",
    "sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/SCI_CAL_001_FIXED_DJF25_FULL_DOMAIN_OWNER_DECISION.md":
        "c43aa932c633e336497547730f73278d3a5cf70d2a5fcfb19049d967c79dd469",
    "sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_contract.json":
        "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a",
    "sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv":
        "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f",
    "sources/tolteca/licenses/LICENSE.rst":
        "9959c063acc1cb1030dc86d7efbd6591d9a759c7787721dee6320ebd4d20a41e",
    "sources/tolteca/tolteca/data/cal/toltec_passband/data/a1100_passband.ecsv":
        "13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72",
    "sources/tolteca/tolteca/data/cal/toltec_passband/data/a1400_passband.ecsv":
        "a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e",
    "sources/tolteca/tolteca/data/cal/toltec_passband/data/a2000_passband.ecsv":
        "77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff",
    "sources/tolteca/tolteca/data/cal/toltec_passband/index.yaml":
        "74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5",
}

PASSBAND_MEMBERS = (
    "data/a1100_passband.ecsv",
    "data/a1400_passband.ecsv",
    "data/a2000_passband.ecsv",
    "index.yaml",
)
PASSBAND_SET_SHA256 = (
    "5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433"
)
PASSBAND_TOTAL_BYTES = 1_297_803
NODE_TABLE_SHA256 = (
    "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_inventory() -> None:
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in SOURCES.rglob("*")
        if path.is_file()
    }
    require(actual == set(EXPECTED), "staged source inventory mismatch")

    checksum_rows = {}
    for line in (ROOT / "SOURCE_OBJECT_SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(relative not in checksum_rows, f"duplicate checksum row: {relative}")
        checksum_rows[relative] = expected
    require(checksum_rows == EXPECTED, "checksum-file inventory mismatch")

    for relative, expected in EXPECTED.items():
        actual_digest = sha256(ROOT / relative)
        require(
            actual_digest == expected,
            f"SHA-256 mismatch for {relative}: {actual_digest} != {expected}",
        )


def verify_contract() -> None:
    contract = json.loads(
        (CAL_ROOT / "sci_cal_001_fixed_djf25_full_domain_operator_contract.json")
        .read_text()
    )
    require(
        contract["operator_id"]
        == "am12_fixed_djf25_piecewise_linear_los_tau_v1",
        "operator identity mismatch",
    )
    require(
        contract["provenance"]["operator_nodes_csv_sha256"]
        == NODE_TABLE_SHA256,
        "machine contract does not bind the recovered node table",
    )
    require(
        contract["domain"]["outside"] == "fail_closed",
        "machine contract support behavior mismatch",
    )

    node_path = CAL_ROOT / "sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv"
    with node_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    require(len(rows) == 1_368, f"node-table row count mismatch: {len(rows)}")
    require(
        set(reader.fieldnames or ())
        == {
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
        },
        "node-table schema mismatch",
    )


def verify_passband_set() -> None:
    aggregate = hashlib.sha256()
    total_bytes = 0
    for relative in sorted(PASSBAND_MEMBERS):
        path = PASSBAND_ROOT / relative
        member_digest = sha256(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(member_digest))
        aggregate.update(b"\0")
        total_bytes += path.stat().st_size
    require(total_bytes == PASSBAND_TOTAL_BYTES, "passband byte total mismatch")
    require(
        aggregate.hexdigest() == PASSBAND_SET_SHA256,
        "passband-set aggregate mismatch",
    )


def main() -> int:
    verify_inventory()
    verify_contract()
    verify_passband_set()
    print("OK: exact staged source inventory 9 files")
    print("OK: atmosphere machine contract and 1,368-row node table")
    print(f"OK: TolTECA-v1 passband set {PASSBAND_SET_SHA256}")
    print(f"OK: passband member bytes {PASSBAND_TOTAL_BYTES}")
    print("NOTE: WVR policy authority is outside this exact-byte verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
