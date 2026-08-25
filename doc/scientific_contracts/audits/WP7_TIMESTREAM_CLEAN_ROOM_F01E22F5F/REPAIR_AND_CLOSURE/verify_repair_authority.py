#!/usr/bin/env python3
"""Verify the exact WP-7 repair authority publication set."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
MANIFEST = ROOT / "WP7_REPAIR_AUTHORITY_MANIFEST_2026-08-25.md"
STAGING = ROOT / "RECOVERED_CAL_NUMERICAL_AUTHORITY_2026-08-25"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name: str) -> Path:
    if name.startswith("doc/"):
        return REPO / name
    if name.startswith("WP7_") or name == "verify_repair_authority.py":
        return ROOT / name
    return STAGING / name


def verify_manifest_rows() -> int:
    rows = re.findall(
        r"^\| `([^`]+)` \| [^|]+ \| `([0-9a-f]{64})` \|$",
        MANIFEST.read_text(),
        re.MULTILINE,
    )
    if len(rows) != 21:
        raise RuntimeError(f"expected 21 bound authority objects, found {len(rows)}")

    names: set[str] = set()
    for name, expected in rows:
        if name in names:
            raise RuntimeError(f"duplicate manifest object: {name}")
        names.add(name)
        path = resolve(name)
        if not path.is_file():
            raise RuntimeError(f"missing authority object: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"authority hash mismatch for {name}: {actual} != {expected}"
            )
    return len(rows)


def verify_readable_authority() -> None:
    text = (ROOT / "WP7_APPROVED_SCIENTIFIC_AUTHORITY_ADDENDUM_2026-08-25.md").read_text()
    compact = " ".join(text.split())
    required = (
        "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969",
        "7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a",
        "fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f",
        "5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433",
        "cal_wvr_tau225_linear_detector_time_v1",
        "cal_wvr_tau225_unavailable_v1",
        "cal_wvr_observation_quality_mean_peak_v1",
        "RTC logical-stream terminal completion",
        "not physical materialization or serialization",
    )
    missing = [token for token in required if token not in compact]
    if missing:
        raise RuntimeError(f"readable authority omits required content: {missing}")
    if re.search(r"\b(?:F|XOD|TS-CLAR)-\d{3}\b", text):
        raise RuntimeError("readable authority leaks a prior finding identifier")


def verify_recovery() -> None:
    result = subprocess.run(
        [sys.executable, str(STAGING / "verify_recovery.py")],
        cwd=STAGING,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    print(result.stdout, end="")


def main() -> int:
    count = verify_manifest_rows()
    verify_readable_authority()
    verify_recovery()
    print(f"OK: repair authority objects {count}")
    print("OK: sanitized authority content and finding firewall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
