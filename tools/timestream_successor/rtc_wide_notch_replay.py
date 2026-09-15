#!/usr/bin/env python3
"""Prepare and run the two explicitly chosen wider RTC trials, without maps."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from rtc_coherent_science_campaign import sky
from rtc_coherent_trial import erode
from rtc_wide_notch_experiment import COHERENT_SEAL, POPULATION_SEAL, IDS, sealed_reader
from run_rtc_notch_recovery import binding, digest, write_json

ARMS = ("lowpass", "w0.5-t6", "w1-t3", "w2-t1")
ASEC = 206264.80624709636


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    old, old_used = sealed_reader(a.prior, COHERENT_SEAL)
    pop, pop_used = sealed_reader(a.population, POPULATION_SEAL)
    chosen = json.loads(a.selection.read_text())
    if (
        chosen["new_replays"] != list(ARMS[2:])
        or digest(a.screen / "screen.json") != chosen["screen_sha256"]
    ):
        raise ValueError("incompatible explicit replay selection")
    screen = json.loads((a.screen / "screen.json").read_text())
    cfg0 = json.loads(pop("replay-plans/valuable.json").read_text())
    trials = [dict(id="lowpass", notch=False, reject=False)]
    for name in ARMS[1:]:
        r = next(c for c in screen["candidates"] if c["id"] == name)
        if digest(r["trial"]["path"]) != r["trial"]["sha256"]:
            raise ValueError("changed screened coefficient vector")
        trials.append(json.loads(Path(r["trial"]["path"]).read_text()))
    reference = next(t for t in cfg0["trials"] if t["id"] == "finite-6s")
    if not np.array_equal(reference["finite_notch"], trials[1]["finite_notch"]):
        raise ValueError("narrow suppression reference was altered")
    receipt_path = old("native-full/receipt.json")
    receipt = json.loads(receipt_path.read_text())
    n = receipt["rows"]
    t = np.fromfile(old("native-full/native-time.f64"), "<f8")
    gpath = pop("campaign/valuable/geometry.f64")
    g = np.fromfile(gpath, "<f8").reshape(n, 4)
    if not np.allclose(t, g[:, 0] - g[0, 0], rtol=0, atol=2e-7):
        raise ValueError("native and AST time axes differ")
    narrow = np.fromfile(pop("campaign/valuable/finite-6s-causes.u8"), "u1") == 0
    # Use the fastest previously reference-supported center with a conservative
    # 20-native-cell interior margin, fixed before any new source response.
    interior = np.flatnonzero(erode(narrow, 20))
    fast = int(interior[np.argmax(g[interior, 1])])
    xy269 = np.fromfile(old("geometry-02/pointing-269.f64"), "<f8").reshape(n, 2) * ASEC
    design = json.loads(old("prepare-03/science-design.json").read_text())
    components = []
    for i, c in enumerate(design["crossings"]):
        components.append(
            dict(c, id=f"compact-{i + 1}", kind="compact", crossing_role=c["kind"])
        )
    for phase in (0.0, 0.5):
        center_time = t[fast] + phase * cfg0["nominal_interval"]
        point = [float(np.interp(center_time, t, xy269[:, k])) for k in range(2)]
        components.append(
            dict(
                id=f"fast-{phase:g}",
                kind="compact",
                time=float(center_time),
                center_arcsec=point,
                native_row=fast,
                phase=phase,
                speed_arcsec_per_second=float(g[fast, 1]),
            )
        )
    excluded = int(np.nanargmax(g[:, 1]))
    components.append(
        dict(
            id="fast-original-excluded",
            kind="compact",
            time=float(t[excluded]),
            center_arcsec=xy269[excluded].tolist(),
            native_row=excluded,
            speed_arcsec_per_second=float(g[excluded, 1]),
        )
    )
    components.append(
        dict(
            id="extended",
            kind="extended",
            time=design["crossings"][0]["time"],
            center_arcsec=design["crossings"][0]["center_arcsec"],
            native_row=design["crossings"][0]["native_row"],
            apparent_FWHM_arcsec=design["extended_apparent_FWHM_arcsec"],
        )
    )
    for c in components:
        c["peak_mJy_per_beam"] = (
            design["extended_peak_mJy_per_beam"]
            if c["kind"] == "extended"
            else design["compact_peak_mJy_per_beam"]
        )
    apt = np.fromfile(old("geometry-02/apt.f64"), "<f8").reshape(491, 6)
    line = np.fromfile(old("real-01/known-added-line-selected.f64"), "<f8").reshape(
        n, 12, 2
    )
    records = {
        r["detector"]: r
        for r in map(
            json.loads, gzip.open(pop("population-final/detectors.jsonl.gz"), "rt")
        )
        if str(r["observation"]) == "152390" and r["network"] == 12
    }
    # All 12 original occurrences are retained; only the predetermined eight
    # receive optional notches. Controls and the other 479 channels stay LPF.
    configs = []
    for j, d in enumerate(IDS):
        cfg = json.loads(json.dumps(cfg0))
        for k in (
            "conditional_transient_accounting",
            "audit_record",
            "learning_overlay",
        ):
            cfg.pop(k, None)
        cfg.update(
            channel=d,
            case=f"wide-{d}",
            samples=binding(old(f"native-full/samples-{d}.f64")),
            audit_receipt=binding(receipt_path),
            trials=trials if j < 8 else trials[:1],
            injections=[],
            prior_population_record=records[d],
            exact_screen=binding(a.screen / "screen.json"),
            explicit_selection=binding(a.selection),
            scientific_admission=False,
        )
        xy_path = old(f"geometry-02/pointing-{d}.f64")
        xy = np.fromfile(xy_path, "<f8").reshape(n, 2) * ASEC
        for c in components:
            sd = dict(design)
            sd.update(
                crossings=[] if c["kind"] == "extended" else [c],
                extended_peak_mJy_per_beam=c["peak_mJy_per_beam"]
                if c["kind"] == "extended"
                else 0.0,
                extended_center_arcsec=c["center_arcsec"],
            )
            value = sky(t, xy, sd) / apt[d, 2]
            delta = np.column_stack([value, design["r_injection_native_ratio"] * value])
            p = a.output / f"{d}-{c['id']}.f64"
            delta.astype("<f8").tofile(p)
            cfg["injections"].append(
                dict(
                    **binding(p),
                    identity=c["id"],
                    geometry_binding=binding(gpath),
                    own_detector_pointing=binding(xy_path),
                    source=c,
                    r_ratio=design["r_injection_native_ratio"],
                    scope="physically consistent sky, fixed original plan; no learning or source admission",
                )
            )
        p = a.output / f"{d}-known-line.f64"
        line[:, j].astype("<f8").tofile(p)
        cfg["injections"].append(
            dict(
                **binding(p),
                identity="known-line",
                geometry_binding=binding(gpath),
                source="previous constructed envelope; not independent evidence of physical contamination",
            )
        )
        write_json(a.output / f"{d}.json", cfg)
        configs.append(binding(a.output / f"{d}.json"))
    write_json(
        a.output / "design.json",
        dict(
            components=components,
            configs=configs,
            interval_seconds=cfg0["measured_interval"],
            nominal_interval=cfg0["nominal_interval"],
            fixed_feature_edges_hz=screen["fixed_feature_edges_hz"],
            fixed_broad_diagnostic_hz=[9.75, 12.25],
            arms=list(ARMS),
            targets=IDS[:8],
            controls=IDS[8:],
            own_pointing=True,
            paired_xr_operator=True,
            r_ratio=design["r_injection_native_ratio"],
            calibration="same matched-APT reference scale, not new production calibration",
            readout="same provisional centered uniform average,64-midpoint,linear within-native pointing",
            extended="direct source-only timestream at first supported compact center; require complete5sigma domain for integrated response; not prior poorly covered map aperture",
            bindings=old_used + pop_used,
            prior_root=str(a.prior),
            population_root=str(a.population),
            production=False,
            downstream_campaign=False,
            FRUIT_qualification=False,
        ),
    )
    print(
        "prepared",
        len(configs),
        "original detector occurrences; fast center",
        fast,
        float(g[fast, 1]),
    )


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepared / "design.json").read_text())
    records = []
    for b in design["configs"]:
        if digest(b["path"]) != b["sha256"]:
            raise ValueError("changed frozen replay specification")
        cfg = json.loads(Path(b["path"]).read_text())
        d = cfg["channel"]
        start = time.perf_counter()
        command = [str(a.executable), b["path"], str(a.output / str(d))]
        with (a.output / f"{d}.log").open("w") as f:
            subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
        receipt = json.loads((a.output / str(d) / "receipt.json").read_text())
        if (
            not receipt["original_pair_unchanged"]
            or receipt["production_filtering_active"]
        ):
            raise ValueError("original or activation invariant failed")
        original = np.fromfile(cfg["samples"]["path"], "<f8").reshape(-1, 4)[:, :2]
        if not np.array_equal(
            original,
            np.fromfile(a.output / str(d) / "original.f64", "<f8").reshape(-1, 2),
            equal_nan=True,
        ):
            raise ValueError("replayed original pair differs")
        records.append(
            dict(
                detector=d,
                command=command,
                specification=b,
                executable=binding(a.executable),
                receipt=receipt,
                seconds=time.perf_counter() - start,
            )
        )
        print("completed", d, receipt["source_revision"], flush=True)
    write_json(
        a.output / "receipt.json",
        dict(
            records=records,
            design=binding(a.prepared / "design.json"),
            selected_new_candidates=list(ARMS[2:]),
            production=False,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    for name in ("prior", "population", "screen", "selection", "output"):
        q.add_argument("--" + name, type=Path, required=True)
    q = sub.add_parser("run")
    for name in ("prepared", "executable", "output"):
        q.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a) if a.command == "prepare" else run(a)


if __name__ == "__main__":
    main()
