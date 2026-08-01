#!/usr/bin/env python3
"""Generate or verify the deterministic SCI-CAL-001 package SHA-256 list."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


DIGEST_NAME = "SHA256SUMS"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_bytes(package_dir: Path) -> bytes:
    files = sorted(
        path
        for path in package_dir.iterdir()
        if path.is_file() and path.name != DIGEST_NAME
    )
    lines = [f"{sha256_path(path)}  {path.name}" for path in files]
    return ("\n".join(lines) + "\n").encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify SHA256SUMS instead of rewriting it",
    )
    args = parser.parse_args()

    package_dir = Path(__file__).resolve().parent
    digest_path = package_dir / DIGEST_NAME
    expected = expected_bytes(package_dir)
    if args.check:
        if not digest_path.exists() or digest_path.read_bytes() != expected:
            print(f"stale or missing digest list: {digest_path}", file=sys.stderr)
            return 1
        print(f"verified {digest_path}")
        return 0

    digest_path.write_bytes(expected)
    print(f"wrote {digest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
