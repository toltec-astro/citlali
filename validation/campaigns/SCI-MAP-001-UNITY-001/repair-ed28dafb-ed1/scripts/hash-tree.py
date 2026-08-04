#!/usr/bin/env python3
"""Write or verify a deterministic SHA-256 inventory without following links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"inventory root is not a directory: {root}")
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        stat = path.lstat()
        common = {
            "path": relative,
            "mtime_ns": stat.st_mtime_ns,
            "mode": oct(stat.st_mode & 0o7777),
        }
        if path.is_symlink():
            records.append({**common, "type": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            records.append(
                {
                    **common,
                    "type": "file",
                    "size": stat.st_size,
                    "sha256": sha256(path),
                }
            )
        elif path.is_dir():
            records.append({**common, "type": "directory"})
        else:
            raise ValueError(f"unsupported filesystem entry: {path}")
    return {
        "schema_version": "sci-map-path-inventory-v1",
        "root": str(root),
        "records": records,
    }


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("select exactly one of --output or --verify")
    current = inventory(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(current))
        print(f"wrote {args.output}")
        return 0
    expected = json.loads(args.verify.read_text())
    # The evidence can move as a complete tree. The content-relative inventory,
    # not the absolute root string, is the comparison authority.
    current["root"] = expected.get("root")
    if canonical_bytes(current) != canonical_bytes(expected):
        print("inventory mismatch", file=sys.stderr)
        return 2
    print(f"verified {len(current['records'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
