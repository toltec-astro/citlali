"""Owner-run reconstruction of deleted CAL transports under original identities.

No health learning, POINT map, new model or archive modification. A single
original-executable CAL invocation precedes the 54-call diagnostic cycle budget.
Only exact-hash bytes can enter the shadow parent; failures preserve both stages.
"""

from pathlib import Path
import os
import shutil

from configured_point_cli import bind, native_call, write_json


def materialize(restoration, destination, timeout):
    stage = destination / "restore-cal"
    print("Restoring original CAL inputs; exact saved hashes are required.", flush=True)
    receipt = native_call(
        Path(restoration["native"]["path"]),
        Path(restoration["request"]["path"]),
        stage,
        terminal="cal",
        workers=restoration["workers"],
        timeout=timeout,
    )
    shadow = destination / "original-cal"
    shadow.mkdir()
    roots = {}
    count = 0
    size = 0
    for network in restoration["networks"]:
        name = "network" + str(network["network"])
        fresh = stage / "products" / name
        original = Path(network["original_root"])
        target = shadow / name
        target.mkdir()
        for name in network["metadata"]:
            out = target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / name, out)
        for name, expected in network["payloads"].items():
            source = fresh / name
            if bind(source)["sha256"] != expected:
                raise ValueError("restored original CAL differs: " + str(source))
            out = target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            os.link(
                source, out
            )  # One owned immutable byte set; no duplicate transport storage.
            count += 1
            size += source.stat().st_size
        roots[str(original)] = str(target)
    write_json(
        destination / "RESTORATION.json",
        dict(
            status="exact_saved_payloads_verified",
            upstream_invocations=1,
            receipt=receipt,
            payload_files=count,
            payload_bytes=size,
            original_archives_untouched=True,
            shadow_roots=roots,
        ),
    )
    return roots
