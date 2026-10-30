#!/usr/bin/env python3
"""Prepare the approved five-case 152418 experiment; never process observations."""

import argparse
import json
from pathlib import Path
import shlex

import numpy as np
from astropy.io import fits
import yaml

from configured_point_cli import bind, write_json
from point_recovery_products import ARRAYS, ROLES, signals, geometry
from inspect_point_source import gaussian

PROJECT = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027"
)
REFERENCE = PROJECT / "pointings-weight-control-152418-20261030/reduced/redu00"


def read(path):
    return json.loads(Path(path).read_text())


def prepare(runtime, review, anchors, destination):
    runtime = runtime.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("trial destination already exists")
    frozen = read(anchors)
    work = REFERENCE / ".point-work"
    bindings = {}

    def add(path, expected=None):
        key = str(Path(path).resolve())
        if key in bindings:
            ref = bindings[key]
            if expected is not None and ref["sha256"] != expected:
                raise ValueError("conflicting original binding")
            return ref
        ref = bind(Path(path))
        if expected is not None and ref["sha256"] != expected:
            raise ValueError("changed original input: " + str(path))
        bindings[ref["path"]] = ref
        return ref

    def nested_refs(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                add(value["path"], value["sha256"])
            for v in value.values():
                nested_refs(v)
        elif isinstance(value, list):
            for v in value:
                nested_refs(v)

    nested_refs(frozen)
    rm = read(runtime / "manifest.json")
    revision = rm["source"]["revision"]
    verdict = read(review)
    if verdict.get("candidate") != revision or verdict.get("verdict") not in (
        "pass",
        "pass with recorded limitations",
    ):
        raise ValueError("runtime has no passing exact-source independent review")
    runtime_binding = add(runtime / "manifest.json")
    review_binding = add(review)
    for e in rm["files"]:
        add(runtime / e["path"], e["sha256"])
    add(rm["python"]["path"], rm["python"]["sha256"])
    basepath = work / "weighted-00-request.json"
    base = read(basepath)
    nested_refs(base)
    basebinding = add(basepath)
    config = yaml.safe_load(Path(base["effective_config"]["path"]).read_text())[
        "successor"
    ]
    if (
        config["map"]["pixel_size_arcsec"] != 2
        or config["map"]["coverage_cut"] != 0.15
        or config["cleaning"]["ranks"] != dict(a1100=3, a1400=4, a2000=6)
        or config["feedback"] != dict(enabled=True, max_iters=10)
        or config["detector_response"]["min_relative_response"] != 0.1
    ):
        raise ValueError("saved recipe differs from approved fixed experiment")
    restoration = []
    for network in base["networks"]:
        root = Path(network["root"])
        inp = read(network["input"]["path"])
        nested_refs(inp)
        apt = Path(inp["manifest"]["path"]).parent
        for path in apt.rglob("*"):
            if path.is_file():
                add(path)
        metadata = [
            "receipt.yaml",
            "learning-receipt.yaml",
            "processing-scans.yaml",
            "donor-continuity/apply-receipt.yaml",
            "donor-continuity/cal/receipt.yaml",
        ]
        for name in metadata:
            add(root / name)
        # Native-time identity lives in the retained learning receipt. Avoid
        # parsing the full scientific receipt more than once during preparation.
        learn = yaml.load(
            (root / "learning-receipt.yaml").read_text(), Loader=yaml.CSafeLoader
        )
        cal = yaml.load(
            (root / "donor-continuity/cal/receipt.yaml").read_text(),
            Loader=yaml.CSafeLoader,
        )
        payloads = {"native-time.f64": learn["native_time_sha256"]}
        for d in cal["detectors"]:
            for name, sha in d["files"].items():
                payloads["donor-continuity/cal/" + name] = sha
        restoration.append(
            dict(
                network=inp["network"],
                original_root=str(root),
                metadata=metadata,
                payloads=payloads,
            )
        )
    upstream = add(work / "inputs/observation.json")
    oldruntime = Path(frozen["runtime"]["executable"]["path"]).parent.parent
    oldmanifest = read(oldruntime / "manifest.json")
    add(oldruntime / "manifest.json")
    for e in oldmanifest["files"]:
        add(oldruntime / e["path"], e["sha256"])
    # Every saved reference plane and model used by replay/measurement is frozen.
    for i in range(11):
        for a in ARRAYS:
            for name in ("map.fits", "plan.yaml", "bundle.yaml"):
                add(work / f"weighted-{i:02d}/products" / a / name)
            feedback = work / f"weighted-{i:02d}/feedback-{a}/feedback.json"
            if feedback.exists():
                add(feedback)
    destination.mkdir(parents=True)
    inputs = destination / "inputs"
    inputs.mkdir()
    rasters = {r: {} for r in ROLES if r != "SEED"}
    geometries = {}
    shapes = {}
    for array in ARRAYS:
        final = read(frozen["final_fits"][array]["path"])["guard"][
            "accepted_parameters"
        ]
        initial = read(work / f"weighted-00/feedback-{array}/feedback.json")["guard"][
            "accepted_parameters"
        ]
        major = max(initial[3:5]) * np.sqrt(8 * np.log(2))
        with fits.open(work / f"weighted-00/products/{array}/map.fits") as h:
            g = geometry(h[0].header, h[0].data.shape, initial)
            valid = h["SCI_SUPPORT"].data.astype(bool)
            required = (
                g["disk"]
                | g["background"]
                | (g["template"] != 0)
                | np.any(g["apertures"], axis=0)
            )
            if not valid[required].all():
                raise ValueError("fixed measurement support unavailable: " + array)
        gp = inputs / (array + "-geometry.npz")
        np.savez_compressed(gp, **g)
        geometries[array] = add(gp)
        values = signals(final, major)
        yy, xx = np.indices((41, 41))
        xx = (20 - xx) * 2.0
        yy = (yy - 20) * 2.0
        summaries = {}
        for role, raster in values.items():
            p = inputs / (array + "-" + role + ".json")
            write_json(
                p,
                dict(
                    schema="citlali-point-recovery-raster-v1",
                    array=array,
                    frame="delta-source-AltAz",
                    unit="mJy/nominal beam",
                    values=raster.ravel().tolist(),
                ),
            )
            rasters[role][array] = add(p)
            q = np.asarray(final).copy()
            q[6] = 0
            if role == "SHOULDER":
                q[0] *= 0.05
                q[1] += 0.5 * major
                q[3:5] *= 2
            elif role == "REF":
                q[0] = 0
            else:
                q[0] *= 0.1 if role == "CORE+" else -0.1
            before = gaussian(q, xx, yy)
            summaries[role] = dict(
                parameters=q.tolist(),
                signed_extreme=float(raster.min() if role == "CORE-" else raster.max()),
                brightness_area=float(raster.sum() * 4),
                untapered_grid_brightness_area=float(before.sum() * 4),
                taper_removed_brightness_area=float((before - raster).sum() * 4),
            )
        shapes[array] = dict(
            saved_M10_parameters=final,
            saved_M0_parameters=initial,
            major_M0_arcsec=major,
            summaries=summaries,
            profile_edges_arcsec=list(range(0, 62, 2)),
            aperture_factors=[1.0, 1.5, 2.0],
            template_radius_major_fwhm=3.0,
            pixel_area_arcsec2=4.0,
        )
    write_json(inputs / "signal-definitions.json", shapes)
    add(inputs / "signal-definitions.json")
    health = yaml.load(
        Path(base["health"]["path"]).read_text(), Loader=yaml.CSafeLoader
    )
    observation = health["networks"][0]["observation_identity"]
    cases = {}
    for role in ROLES:
        case = dict(
            schema="citlali-point-recovery-v1",
            case=role,
            observation=observation,
            config_sha256=base["effective_config"]["sha256"],
            health_sha256=base["health"]["sha256"],
            arrays=rasters["SHOULDER" if role == "SEED" else role],
        )
        if role == "REF":
            case["weighted_references"] = {
                a: [rasters["CORE+"][a], rasters["SHOULDER"][a]] for a in ARRAYS
            }
        path = inputs / (role + ".json")
        write_json(path, case)
        cases[role] = add(path)
    manifest = dict(
        schema="citlali-point-recovery-trial-v1",
        ready=True,
        source_revision=revision,
        runtime_manifest=runtime_binding,
        review=review_binding,
        native=add(runtime / "bin/citlali-native"),
        base_request=basebinding,
        reference_work=str(work),
        roles=list(ROLES),
        budget=10,
        output=str(destination / "run-001"),
        max_seconds=14400,
        max_output_bytes=20 * 1024**3,
        minimum_free_bytes=50 * 1024**3,
        cases=cases,
        rasters=rasters,
        geometry=geometries,
        restoration=dict(
            native=frozen["runtime"]["executable"],
            request=upstream,
            workers=3,
            purpose="one original-executable CAL materialization; exact saved payload hashes required",
            networks=restoration,
        ),
        scientific_acceptance="descriptive only; no release thresholds set",
        bindings=list(bindings.values()),
    )
    path = destination / "trial.json"
    write_json(path, manifest)
    command = [
        rm["python"]["path"],
        "-B",
        str(runtime / "tools/timestream_successor/run_point_recovery.py"),
        str(path),
        "--expected-sha256",
        bind(path)["sha256"],
    ]
    launcher = "#!/bin/bash\n(\nset -euo pipefail\n"
    launcher += "export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1\n"
    launcher += "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE\n"
    launcher += (
        "export MPLCONFIGDIR="
        + shlex.quote(str(destination / "cache/matplotlib"))
        + "\n"
    )
    launcher += (
        "export XDG_CACHE_HOME=" + shlex.quote(str(destination / "cache")) + "\n"
    )
    launcher += "exec " + shlex.join(command) + ' "$@"\n)\n'
    (destination / "02_redu.sh").write_text(launcher)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True, type=Path)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--anchors", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    print(prepare(args.runtime, args.review, args.anchors, args.destination))


if __name__ == "__main__":
    main()
