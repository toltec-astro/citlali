#!/usr/bin/env python3
"""Execute exactly one registered EL-F12 trajectory with retained receipts.

No implicit retry, candidate search, new input, or extra trajectory is allowed.
Registration is prepared and frozen separately, before any observation run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import numpy as np

from tools.fruit_loops.analyze_response_intervention import (
    ARRAYS, compare_trajectories, file_record, iteration_dirs, product_path,
    require_map_identity, write_json, decision_assignments,
    SCIENCE_PLANES, image,
)

ORDER = [f"{arm}/{injection}" for arm in ("H0", "H", "Half", "Hold")
         for injection in ("uninjected", "injected")]
RESTART_ORDER = [f"{arm}/restart-{injection}" for arm in ("Half", "Hold")
                 for injection in ("uninjected", "injected")]
GIB = 1 << 30


def verify_artifacts(records: list[dict]) -> None:
    for expected in records:
        actual = file_record(Path(expected["path"]))
        if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
            raise ValueError(f"registered input/implementation identity changed: {expected['path']}")


def tree_bytes(root: Path) -> tuple[int, int]:
    retained = spool = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            retained += size
            if path.name.startswith("fruit_response_") and path.suffix == ".bin":
                spool += size
    return retained, spool


def resource_violation(elapsed, aggregate, rss, retained, spool) -> str | None:
    for exceeded, reason in ((elapsed > 3600, "one-hour trajectory limit"),
                             (aggregate > 43200, "12-hour aggregate limit"),
                             (rss > 16 * GIB, "16-GiB process RSS limit"),
                             (retained > 64 * GIB, "64-GiB retained-output limit"),
                             (spool > 32 * GIB, "32-GiB temporary-spool limit")):
        if exceeded:
            return reason
    return None


def compress_completed_spools(root: Path, guard=lambda: None) -> list[dict]:
    """Losslessly retain successful temporary spools, verifying byte identity.

    Called only after the entire trajectory has exited successfully. Failed
    attempts remain untouched. The compressed file replaces no older product.
    """
    records = []
    for path in sorted(root.rglob("fruit_response_*.bin")):
        before = file_record(path)
        target = path.with_suffix(".bin.gz")
        started = time.monotonic()
        with path.open("rb") as source, target.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
                last_check = time.monotonic()
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    compressed.write(chunk)
                    if time.monotonic() - last_check > 1:
                        guard()
                        last_check = time.monotonic()
        digest = hashlib.sha256()
        count = 0
        with gzip.open(target, "rb") as restored:
            for chunk in iter(lambda: restored.read(1 << 20), b""):
                digest.update(chunk)
                count += len(chunk)
                if time.monotonic() - last_check > 1:
                    guard()
                    last_check = time.monotonic()
        if digest.hexdigest() != before["sha256"] or count != before["size_bytes"]:
            raise ValueError("lossless spool retention verification failed")
        # Exact verified replacement of this run's temporary storage only.
        path.unlink()
        records.append({"original": before, "lossless_retained": file_record(target),
                        "compression_and_verification_seconds": time.monotonic() - started})
    return records


def parse_time_log(text: str) -> dict:
    record = {}
    match = re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", text)
    if match:
        record.update(zip(("wall_seconds", "user_seconds", "system_seconds"), map(float, match.groups())))
    match = re.search(r"(\d+)\s+maximum resident set size", text)
    if match:
        record["peak_rss_bytes"] = int(match.group(1))
    if set(record) != {"wall_seconds", "user_seconds", "system_seconds", "peak_rss_bytes"}:
        raise ValueError("required operating-system resource receipt missing")
    return record


def completed_gate(root: Path, name: str, registration: dict) -> dict:
    arm, injection = name.split("/")
    restart = injection.startswith("restart-")
    injection = injection.removeprefix("restart-")
    reduced = root / name / "reduced"
    iterations = iteration_dirs(reduced, 123424)
    expected = set(range(4, 7)) if restart else set(range(7))
    if set(iterations) != expected:
        raise ValueError("missing or extra absolute iteration")
    for iteration, directory in iterations.items():
        if not (directory / "citlali_restart_checkpoint.nc").is_file():
            raise ValueError("required checkpoint absent")
        if arm != "H0" and len(list(directory.rglob("*_fruit_response.nc"))) != 1:
            raise ValueError("required causal decision ledger absent")
        for array in ARRAYS:
            path = product_path(directory, 123424, array)
            # Self-comparison also confirms all four mandatory science planes.
            require_map_identity(path, path)
            if any(not np.isfinite(image(path, extension)).all() for extension in SCIENCE_PLANES):
                raise ValueError("nonfinite ordinary science product")
    result = {}
    if restart:
        result["own_iteration_3_restart"] = compare_trajectories(root / arm / injection / "reduced", reduced)
        return result
    if name == "H0/uninjected":
        result["EL_F2_alpha_one"] = compare_trajectories(Path(registration["retained_control"]), reduced, maps_only=True)
    if arm == "H":
        result["H0_neutrality"] = compare_trajectories(root / "H0" / injection / "reduced", reduced, audit_added=True)
    if arm in ("Half", "Hold"):
        assignments = decision_assignments(iterations[6] / "citlali_restart_checkpoint.nc")
        first_action = min((row["first_application"] for row in assignments if row["first_application"] >= 0), default=7)
        historical = iteration_dirs(root / "H" / injection / "reduced", 123424)
        for iteration in range(first_action):
            for array in ARRAYS:
                require_map_identity(product_path(historical[iteration], 123424, array),
                                     product_path(iterations[iteration], 123424, array))
        result["pre_first_action"] = {"first_application": first_action if first_action < 7 else None,
                                       "bitwise_through_iteration": first_action - 1}
    if injection == "injected":
        uninjected = iteration_dirs(root / arm / "uninjected" / "reduced", 123424)[0]
        for array in ARRAYS:
            require_map_identity(product_path(uninjected, 123424, array), product_path(iterations[0], 123424, array))
        result["paired_iteration_zero"] = "bitwise"
    return result


def run_one(registration_path: Path, name: str) -> dict:
    registration = json.loads(registration_path.read_text())
    if registration["decision"] != "SCI-FRUIT-EL-F12-RESPONSE-AWARE-INTERVENTION-SCREEN-R0.1+CAP-001":
        raise ValueError("unknown registration")
    if registration["order"] != ORDER or name not in ORDER + RESTART_ORDER:
        raise ValueError("unregistered run order or trajectory")
    root = Path(registration["output_root"])
    restart = name in RESTART_ORDER
    for earlier in ORDER if restart else ORDER[:ORDER.index(name)]:
        receipt = root / earlier / "EXECUTION_RECEIPT.json"
        if not receipt.exists() or json.loads(receipt.read_text())["status"] != "pass":
            raise ValueError(f"earlier trajectory/gate has not passed: {earlier}")
    if restart:
        arm = name.split("/")[0]
        analysis = json.loads((root / "analysis" / f"{arm}_RESULT.json").read_text())
        if not analysis["promising_before_required_restarts"] or not analysis["selected_opportunities"]:
            raise ValueError("conditional restart has no scientifically promising candidate")
    if (root / name / "STARTED.json").exists():
        raise ValueError("trajectory already attempted; no automatic retry")
    verify_artifacts(registration["artifacts"])
    first = root / "H0/uninjected/STARTED.json"
    aggregate_start = json.loads(first.read_text())["unix_time"] if first.exists() else time.time()
    retained, spool = tree_bytes(root)
    violation = resource_violation(0, time.time() - aggregate_start, 0, retained, spool)
    if violation:
        raise ValueError(violation)
    case = registration["trajectories"][name]
    started = time.time()
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "STARTED.json", {"unix_time": started, "name": name,
               "registered_command": case["command"], "registration": file_record(registration_path),
               "reserved_passes": 3 if restart else 7, "primary_attempt": None if restart else ORDER.index(name) + 1})
    environment = os.environ.copy()
    environment.update(registration["environment"])
    log = run_dir / "reduction.log"
    receipt = {"name": name, "status": "unavailable", "command": case["command"], "started_unix": started}
    process = None
    peak_sampled_rss = 0
    try:
        with log.open("xb") as stream:
            process = subprocess.Popen(["/usr/bin/time", "-l", *case["command"]],
                cwd=registration["repository"], env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                for child in [monitored, *monitored.children(recursive=True)]:
                    try:
                        rss = max(rss, child.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                peak_sampled_rss = max(peak_sampled_rss, rss)
                retained, spool = tree_bytes(root)
                violation = resource_violation(time.time() - started, time.time() - aggregate_start,
                                               rss, retained, spool)
                if violation:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise ValueError(violation)
                time.sleep(.25)
        text = log.read_text(errors="replace")
        receipt["exit_code"] = process.returncode
        receipt["resources"] = parse_time_log(text)
        if process.returncode:
            raise ValueError(f"Citlali returned {process.returncode}")
        if re.search(r"\[(?:error|critical)\]", re.sub(r"\x1b\[[0-9;]*m", "", text), re.I):
            raise ValueError("unexpected error/critical log record")
        if receipt["resources"]["peak_rss_bytes"] > 16 * GIB:
            raise ValueError("16-GiB process RSS limit")
        receipt["compatibility"] = completed_gate(root, name, registration)
        def retention_guard():
            retained, spool = tree_bytes(root)
            reason = resource_violation(time.time() - started, time.time() - aggregate_start,
                                        psutil.Process().memory_info().rss, retained, spool)
            if reason:
                raise ValueError(reason)
        receipt["compressed_spools"] = compress_completed_spools(run_dir, retention_guard)
        retained, spool = tree_bytes(root)
        violation = resource_violation(time.time() - started, time.time() - aggregate_start,
                                       peak_sampled_rss, retained, spool)
        if violation:
            raise ValueError(violation)
        receipt["status"] = "pass"
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
    finally:
        receipt["elapsed_including_gates_and_retention_seconds"] = time.time() - started
        receipt["aggregate_elapsed_seconds"] = time.time() - aggregate_start
        receipt["sampled_peak_process_rss_bytes"] = peak_sampled_rss
        receipt["retained_bytes"], receipt["temporary_spool_bytes"] = tree_bytes(root)
        if log.exists():
            receipt["log"] = file_record(log)
        write_json(run_dir / "EXECUTION_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("trajectory", choices=ORDER + RESTART_ORDER)
    args = parser.parse_args()
    outcome = run_one(args.registration.resolve(), args.trajectory)
    print(json.dumps({key: outcome[key] for key in ("name", "status", "elapsed_including_gates_and_retention_seconds")}, indent=2))
    if outcome["status"] != "pass":
        print(outcome.get("failure", "required gate failed"))
        raise SystemExit(1)
