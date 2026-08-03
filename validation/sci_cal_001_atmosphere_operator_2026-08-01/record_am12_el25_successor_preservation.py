#!/usr/bin/env python3
"""Record and verify the byte-preserving AM12 EL25 stopped-cache preservation.

This program only reads the source and durable preserved cache.  It rejects
symlinks, nonregular files, altered contents, altered relative paths, and any
writable mode in the preserved tree.  The generated manifest is canonical JSON
and contains a SHA-256 identity for every preserved regular file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/private/tmp/sci_cal_001_am12_el25_confirmation_v1_20260802_root")
DEFAULT_PRESERVED = Path(
    "/Users/gwilson/work_toltec/local_data/citlali-validation/v1/evidence/"
    "sci_cal_001_am12_el25_confirmation_5d1597ca"
)
DEFAULT_OUTPUT = PACKAGE_DIR / "am12_el25_successor_preservation_manifest.json"
FRAGMENT_DIR = PACKAGE_DIR / "am12_el25_successor_preservation_fragments"
SUBTREES = (
    ".am12_el25_confirmation.lock",
    "execution_context.json",
    "scale_traces",
    "failed_attempts",
    "execution_records",
    "raw_outputs",
    *(f"am_spectral_cache/shard_{index:02d}" for index in range(8)),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def inventory(root: Path, *, require_read_only: bool) -> dict[str, Any]:
    if not root.is_dir():
        raise RuntimeError(f"missing cache directory: {root}")
    root_mode = root.stat().st_mode
    if require_read_only and root_mode & 0o222:
        raise RuntimeError(f"preserved cache root is writable: {root}")
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for directory, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in [*directories, *filenames]:
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symlink forbidden in cache: {path}")
            if stat.S_ISDIR(mode):
                if require_read_only and mode & 0o222:
                    raise RuntimeError(f"writable preserved directory: {path}")
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"nonregular cache entry: {path}")
            if require_read_only and mode & 0o222:
                raise RuntimeError(f"writable preserved file: {path}")
            relative = path.relative_to(root).as_posix()
            digest = sha256_path(path)
            size = path.stat().st_size
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(digest))
            aggregate.update(b"\0")
            total_bytes += size
            entries.append({"path": relative, "size_bytes": size, "sha256": digest})
    return {
        "algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }


def build_manifest(source: Path, preserved: Path) -> bytes:
    source_inventory = inventory(source, require_read_only=False)
    preserved_inventory = inventory(preserved, require_read_only=True)
    if source_inventory != preserved_inventory:
        raise RuntimeError("source and preserved cache inventories differ")
    return json_bytes(
        {
            "schema_version": "sci-cal-001-am12-el25-successor-preservation-v1",
            "record_id": "SCI-CAL-001-AM12-EL25-PRESERVATION-001",
            "source_cache": {
                "path": str(source),
                "durable": False,
                "read_only": False,
                "inventory": source_inventory,
            },
            "preserved_cache": {
                "path": str(preserved),
                "durable": True,
                "read_only": True,
                "inventory": preserved_inventory,
            },
            "byte_preservation_verified": True,
        }
    )


def subtree_inventory(root: Path, subtree: str, *, require_read_only: bool) -> dict[str, Any]:
    if subtree not in SUBTREES:
        raise RuntimeError(f"unregistered preservation subtree: {subtree}")
    path = root / subtree
    if path.is_file():
        if require_read_only and path.stat().st_mode & 0o222:
            raise RuntimeError(f"writable preserved file: {path}")
        digest = sha256_path(path)
        return {
            "algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
            "file_count": 1,
            "total_bytes": path.stat().st_size,
            "aggregate_sha256": hashlib.sha256(
                subtree.encode("utf-8") + b"\0" + bytes.fromhex(digest) + b"\0"
            ).hexdigest(),
            "files": [{"path": subtree, "size_bytes": path.stat().st_size, "sha256": digest}],
        }
    nested = inventory(path, require_read_only=require_read_only)
    files = [
        {**item, "path": f"{subtree}/{item['path']}"}
        for item in nested["files"]
    ]
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(item["sha256"]))
        aggregate.update(b"\0")
    return {**nested, "aggregate_sha256": aggregate.hexdigest(), "files": files}


def write_fragment(source: Path, preserved: Path, subtree: str, output: Path) -> bytes:
    source_inventory = subtree_inventory(source, subtree, require_read_only=False)
    preserved_inventory = subtree_inventory(preserved, subtree, require_read_only=True)
    if source_inventory != preserved_inventory:
        raise RuntimeError(f"source and preserved subtree differ: {subtree}")
    data = json_bytes({"subtree": subtree, "inventory": source_inventory})
    atomic_write(output, data)
    return data


def assemble_fragments(source: Path, preserved: Path, fragment_dir: Path) -> bytes:
    inventories = []
    for subtree in SUBTREES:
        path = fragment_dir / (subtree.replace("/", "__") + ".json")
        if not path.is_file():
            raise RuntimeError(f"missing preservation fragment: {path}")
        raw = path.read_bytes()
        fragment = json.loads(raw)
        if raw != json_bytes(fragment) or fragment.get("subtree") != subtree:
            raise RuntimeError(f"noncanonical preservation fragment: {path}")
        inventories.append(fragment["inventory"])
    files = [item for inventory_item in inventories for item in inventory_item["files"]]
    if [item["path"] for item in files] != sorted(item["path"] for item in files):
        files = sorted(files, key=lambda item: item["path"])
    if len({item["path"] for item in files}) != len(files):
        raise RuntimeError("duplicate preserved relative path")
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(item["sha256"]))
        aggregate.update(b"\0")
    full_inventory = {
        "algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }
    return json_bytes(
        {
            "schema_version": "sci-cal-001-am12-el25-successor-preservation-v1",
            "record_id": "SCI-CAL-001-AM12-EL25-PRESERVATION-001",
            "source_cache": {"path": str(source), "durable": False, "read_only": False, "inventory": full_inventory},
            "preserved_cache": {"path": str(preserved), "durable": True, "read_only": True, "inventory": full_inventory},
            "fragment_set": {"subtrees": list(SUBTREES), "directory": str(fragment_dir)},
            "byte_preservation_verified": True,
        }
    )


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--preserved-cache", type=Path, default=DEFAULT_PRESERVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subtree", choices=SUBTREES)
    parser.add_argument("--fragment-dir", type=Path, default=FRAGMENT_DIR)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.parent != PACKAGE_DIR:
        raise RuntimeError("output must remain in the SCI-CAL-001 task package")
    source = args.source_cache.expanduser().resolve()
    preserved = args.preserved_cache.expanduser().resolve()
    fragment_dir = args.fragment_dir.expanduser().resolve()
    if args.subtree and args.assemble:
        raise RuntimeError("choose one preservation mode")
    if args.subtree:
        fragment_dir.mkdir(parents=True, exist_ok=True)
        fragment = fragment_dir / (args.subtree.replace("/", "__") + ".json")
        data = write_fragment(source, preserved, args.subtree, fragment)
        print(json.dumps({"fragment": args.subtree, "sha256": hashlib.sha256(data).hexdigest()}))
        return 0
    data = assemble_fragments(source, preserved, fragment_dir) if args.assemble else build_manifest(source, preserved)
    if args.check:
        if not output.is_file() or output.read_bytes() != data:
            raise RuntimeError("preservation manifest differs from live verification")
        print(json.dumps({"check": True, "sha256": hashlib.sha256(data).hexdigest()}))
        return 0
    atomic_write(output, data)
    print(json.dumps({"written": str(output), "sha256": hashlib.sha256(data).hexdigest()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
