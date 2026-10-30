#!/usr/bin/env python3
"""Owner-only bounded recovery runner. Default is read-only readiness checking."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import subprocess
import signal

import yaml

import numpy as np
from astropy.io import fits

from configured_point_cli import bind, native_call, write_json
from configured_point_feedback import initialize, update, measure_only
from point_recovery_restore import materialize
from point_recovery_products import ARRAYS, ROLES, embed, measurements
from point_recovery_evidence import (
    segment_check,
    publish_seed,
    record_cycle,
    final_accounting,
    load_map,
    increments,
    seed_initial,
)


def read(path):
    return json.loads(Path(path).read_text())


def check(manifest):
    if manifest.get("schema") != "citlali-point-recovery-trial-v1":
        raise ValueError("unexpected trial schema")
    for ref in manifest["bindings"]:
        if bind(Path(ref["path"])) != ref:
            raise ValueError("trial input changed: " + ref["path"])
    if manifest["roles"] != list(ROLES) or manifest["budget"] != 10:
        raise ValueError("trial scope changed")
    refs = {r["path"]: r for r in manifest["bindings"]}
    for name in ("native", "base_request", "runtime_manifest", "review"):
        if refs.get(manifest[name]["path"]) != manifest[name]:
            raise ValueError("unbound " + name)
    for group in [
        manifest["cases"],
        manifest["geometry"],
        *manifest["rasters"].values(),
    ]:
        for ref in group.values():
            if refs.get(ref["path"]) != ref:
                raise ValueError("unbound operand")
    runtime = read(manifest["runtime_manifest"]["path"])
    runtime_root = Path(manifest["runtime_manifest"]["path"]).parent
    if runtime["source"]["revision"] != manifest["source_revision"]:
        raise ValueError("runtime source differs")
    for entry in runtime["files"]:
        path = runtime_root / entry["path"]
        if (
            not path.resolve().is_relative_to(runtime_root.resolve())
            or bind(path)["sha256"] != entry["sha256"]
        ):
            raise ValueError("runtime member changed")
    if bind(Path(runtime["python"]["path"]))["sha256"] != runtime["python"]["sha256"]:
        raise ValueError("Python executable changed")
    if not (
        0 < manifest["max_seconds"] <= 14400
        and 0 < manifest["max_output_bytes"] <= 20 * 1024**3
    ):
        raise ValueError("resource budget changed")
    review = read(manifest["review"]["path"])
    if review.get("candidate") != manifest["source_revision"] or review.get(
        "verdict"
    ) not in ("pass", "pass with recorded limitations"):
        raise ValueError("exact-source independent review absent")
    for key in ("native", "request"):
        ref = manifest["restoration"][key]
        if refs.get(ref["path"]) != ref:
            raise ValueError("unbound original restoration input")
    if not manifest.get("ready", False):
        raise ValueError("trial is not release-ready; preparation/review gates remain")


def parity(products, reference):
    for array in ARRAYS:
        with (
            fits.open(products / array / "map.fits") as a,
            fits.open(reference / array / "map.fits") as b,
        ):
            if len(a) != len(b):
                raise ValueError("REF plane count differs")
            for x, y in zip(a, b, strict=True):
                if x.name != y.name or not np.array_equal(
                    x.data, y.data, equal_nan=True
                ):
                    raise ValueError(
                        "REF scientific plane differs: " + array + " " + x.name
                    )


class CampaignStop(RuntimeError):
    """A shared budget failure stops the entire matrix, never retries a branch."""


def execute(manifest):
    check(manifest)
    root = Path(manifest["output"])
    root.mkdir(parents=False, exist_ok=False)  # Explicit no overwrite or resume.
    start = time.monotonic()
    rows = {}
    invocations = 0
    ref_contexts = ref_models = ref_entry = None
    geometries = {}
    for array in ARRAYS:
        with np.load(manifest["geometry"][array]["path"]) as n:
            geometries[array] = {k: n[k] for k in n.files}
    truths = {
        r: {
            a: embed(
                np.array(
                    read(
                        manifest["rasters"][r if r != "SEED" else "SHOULDER"][a]["path"]
                    )["values"]
                ).reshape(41, 41),
                geometries[a],
            )
            for a in ARRAYS
        }
        for r in ROLES
    }
    ideal_measurements = {}
    ref_final = None

    def budget():
        remaining = manifest["max_seconds"] - (time.monotonic() - start)
        if remaining <= 0:
            raise CampaignStop("campaign wall ceiling reached")
        if shutil.disk_usage(root).free < manifest["minimum_free_bytes"]:
            raise CampaignStop("campaign free-space reserve reached")
        # Only the compact campaign output tree, once per cycle; no input tracing.
        if (
            sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            >= manifest["max_output_bytes"]
        ):
            raise CampaignStop("campaign output storage ceiling reached")
        return remaining

    record = dict(
        schema="citlali-point-recovery-run-v1",
        status="running",
        cases=rows,
        calibration_eligible=False,
        invocations=0,
    )
    write_json(root / "RUN.json", record)

    def alarm(signum, frame):
        raise CampaignStop("campaign wall ceiling reached")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, manifest["max_seconds"])
    try:
        restored = materialize(manifest["restoration"], root, budget())
    except Exception as error:
        record.update(status="stopped", error=str(error), phase="restore_original_cal")
        write_json(root / "RUN.json", record)
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise
    record["restoration"] = bind(root / "RESTORATION.json")
    base = read(manifest["base_request"]["path"])
    for network in base["networks"]:
        network["root"] = restored[network["root"]]
    base.pop("weighting_control", None)
    base["verbose"] = False
    base["weighting_control"] = "inverse-noise"
    for role in ROLES:
        work = root / role
        work.mkdir()
        row = dict(status="running", cycles=[])
        rows[role] = row
        models = {}
        contexts = {}
        first = 0
        parents = None
        if role == "SEED":
            if rows["REF"]["status"] != "completed":
                break
            models = copy.deepcopy(ref_models)
            contexts = ref_contexts.copy()
            row["cycles"].append(copy.deepcopy(ref_entry))
            first = 1
            case = read(manifest["cases"][role]["path"])
        try:
            if role == "SEED":
                row["seed_operand"] = publish_seed(
                    models, case["arrays"], work / "seed"
                )
                row["shared_m0"] = seed_initial(root, work, geometries, truths, models)
            for cycle in range(first, 11):
                remaining = budget()
                if invocations >= 54:
                    raise CampaignStop("native invocation ceiling reached")
                request = dict(
                    base,
                    stage="cycle",
                    cycle_index=cycle,
                    models=models,
                    recovery=manifest["cases"][role],
                )
                path = work / f"cycle-{cycle:02d}-request.json"
                write_json(path, request)
                stage = work / f"cycle-{cycle:02d}"
                invocations += 1
                record["invocations"] = invocations
                receipt = native_call(
                    Path(manifest["native"]["path"]),
                    path,
                    stage,
                    terminal="map",
                    timeout=remaining,
                )
                products = stage / "products"
                if role == "REF":
                    parity(
                        products,
                        Path(manifest["reference_work"])
                        / f"weighted-{cycle:02d}/products",
                    )
                reference_stage = root / "REF" / f"cycle-{cycle:02d}"
                reference_products = (
                    Path(manifest["reference_work"]) / f"weighted-{cycle:02d}/products"
                    if role == "REF"
                    else reference_stage / "products"
                )
                parents = segment_check(products, parents, reference_products)
                if role == "SEED":
                    ref_plan = {
                        a: yaml.safe_load(
                            (reference_products / a / "plan.yaml").read_text()
                        )["segments"]
                        for a in ARRAYS
                    }
                    if any(
                        [s["recovery_parent_sha256"] for s in parents[a]]
                        != [s["recovery_parent_sha256"] for s in ref_plan[a]]
                        for a in ARRAYS
                    ):
                        raise ValueError("SEED original data differs from REF")
                entry = dict(
                    cycle=cycle,
                    native=bind(products / "cycle.yaml"),
                    receipt=receipt,
                    selection={},
                )
                row["cycles"].append(entry)
                models = {}
                guards = {}
                for array in ARRAYS:
                    dest = stage / f"feedback-{array}"
                    if cycle == 10:
                        guard = measure_only(
                            products / array / "map.fits", contexts[array]
                        )
                        entry["selection"][array] = dict(
                            status="final-measurement-only", guard=guard
                        )
                        guards[array] = guard
                        if not guard["accepted"]:
                            raise ValueError("final source guard unavailable: " + array)
                        continue
                    if cycle == 0:
                        contexts[array], selection = initialize(
                            products / array / "map.fits", dest
                        )
                    else:
                        selection = update(
                            products / array / "map.fits", contexts[array], dest
                        )
                    entry["selection"][array] = selection
                    guards[array] = selection["guard"]
                    if selection["status"] != "ready":
                        raise ValueError("source feedback unavailable: " + array)
                    models[array] = dict(
                        candidate=bind(Path(selection["model_path"])),
                        feedback=bind(dest / "feedback.json"),
                        bundle=bind(products / array / "bundle.yaml"),
                    )
                measured = record_cycle(
                    manifest,
                    role,
                    cycle,
                    stage,
                    reference_stage,
                    guards,
                    geometries,
                    truths,
                )
                entry["measurements"] = bind(stage / "measurements.json")
                if role == "REF" and cycle == 10:
                    ref_final = measured
                    for perturbation in ("CORE+", "CORE-", "SHOULDER"):
                        ideal_measurements[perturbation] = {}
                        for array in ARRAYS:
                            budget()
                            path = products / array / "map.fits"
                            guard = measure_only(
                                path, contexts[array], truths[perturbation][array]
                            )
                            data, valid = load_map(path)
                            ideal_measurements[perturbation][array] = measurements(
                                data + truths[perturbation][array],
                                valid,
                                geometries[array],
                                guard,
                            )
                    write_json(root / "IDEAL_CONTROLS.json", ideal_measurements)
                elif role in ("CORE+", "CORE-", "SHOULDER") and cycle == 10:
                    write_json(
                        work / "IDEAL_COMPARISON.json",
                        {
                            a: increments(
                                measured[a]["total"],
                                ideal_measurements[role][a],
                                ref_final[a]["total"],
                            )
                            for a in ARRAYS
                        },
                    )
                # M10 has no candidate and no further application.
                if role == "REF" and cycle == 0:
                    ref_contexts = contexts.copy()
                    ref_models = copy.deepcopy(models)
                    ref_entry = copy.deepcopy(entry)
                write_json(root / "RUN.json", record)
                print(f"{role} M{cycle}: complete", flush=True)
            row["status"] = "completed"
        except Exception as error:
            row.update(status="stopped", error=str(error))
            write_json(root / "RUN.json", record)
            if role == "REF" or isinstance(
                error, (CampaignStop, subprocess.TimeoutExpired)
            ):
                record["status"] = "stopped"
                write_json(root / "RUN.json", record)
                raise
        write_json(root / "RUN.json", record)
    final_accounting(root, rows, geometries, truths)
    record.update(
        status="completed"
        if len(rows) == 5 and all(r["status"] == "completed" for r in rows.values())
        else "completed_with_stops",
        seconds=time.monotonic() - start,
    )
    write_json(root / "RUN.json", record)
    if record["status"] != "completed":
        raise RuntimeError("one or more recovery cases stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Owner authorization to run the five reductions",
    )
    parser.add_argument(
        "--expected-sha256", required=True, help="Frozen trial manifest identity"
    )
    args = parser.parse_args()
    if bind(args.manifest)["sha256"] != args.expected_sha256:
        parser.error("trial manifest changed")
    manifest = read(args.manifest)
    if args.run:
        try:
            execute(manifest)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    else:
        check(manifest)
        print("Ready; no reductions launched.")


if __name__ == "__main__":
    main()
