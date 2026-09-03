#!/usr/bin/env python3
"""Build the deterministic SCI-POINT v0.1 r0.2 Stage B delivery archive."""

from __future__ import annotations

import gzip
import hashlib
import io
import pathlib
import tarfile


ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT = ROOT / "delivery" / "SCI-POINT-v0.1-r0.2-stage-b-delivery-packet.tar.gz"
PREFIX = "SCI-POINT-v0.1-r0.2-stage-b-delivery"
FILES = (
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz",
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz.bytes",
    "author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz.sha256",
    "AUTHOR_PACKET_MANIFEST.md",
    "AUTHOR_PACKET_MANIFEST.sha256",
    "src/common/notation.tex",
    "src/common/definitions.tex",
    "src/common/equations.tex",
    "src/common/assumptions.tex",
    "src/common/requirements.tex",
    "src/common/edge_cases.tex",
    "src/common/bindings.tex",
    "src/scientific-rationale.tex",
    "src/engineering-conformance.tex",
    "STAGE_B_R0_2_RECORDS.json",
    "STAGE_B_R0_2_PARITY_REPORT.json",
    "PROPOSED_SCIENTIFIC_OWNER_DISPOSITION_R0_2.md",
    "STAGE_B_SOURCE_MANIFEST.json",
    "STAGE_B_BUILD_MANIFEST.json",
    "verify_stage_b.py",
    "build_stage_b_delivery.py",
    "pdf/SCI-POINT-SCIENTIFIC-RATIONALE-v0.1.pdf",
    "pdf/SCI-POINT-ENGINEERING-CONFORMANCE-v0.1.pdf",
)


def add_file(bundle: tarfile.TarFile, relative: str) -> None:
    data = (ROOT / relative).read_bytes()
    info = tarfile.TarInfo(f"{PREFIX}/{relative}")
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    bundle.addfile(info, io.BytesIO(data))


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for relative in FILES:
                    add_file(bundle, relative)
    raw = OUTPUT.read_bytes()
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )
    OUTPUT.with_suffix(OUTPUT.suffix + ".bytes").write_text(
        str(len(raw)) + "\n", encoding="utf-8"
    )
    print(f"WROTE {OUTPUT} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
