#!/usr/bin/env python3
"""Exact native frozen-Apply and injection-before-Learn conformance witnesses."""

import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import signal
from run_rtc_coherent_science import IDS
from run_rtc_notch_recovery import binding, write_json


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.evidence / "native-full/receipt.json").read_text())
    n = receipt["rows"]
    sky = np.fromfile(a.real / "sky-selected.f64", "<f8").reshape(n, 12, 2)
    original = np.empty((n, 12, 2))
    for j, d in enumerate(IDS):
        original[:, j] = np.fromfile(
            a.evidence / f"native-full/samples-{d}.f64", "<f8"
        ).reshape(n, 4)[:, :2]
    incoherent = np.fromfile(a.real / "real/relearned-selected.f64", "<f8").reshape(
        n, 12, 2
    )
    coherent = np.fromfile(a.real / "real/coherent-selected.f64", "<f8").reshape(
        n, 12, 2
    )
    frozen = coherent - original
    fresh = incoherent - original - sky
    template = json.loads(a.trial.read_text())
    geometry = binding(
        Path(template["samples"]["path"]).parent.parent
        / "campaign/valuable/geometry.f64"
    )
    # Both AST geometry and all own-detector sky dependencies remain separately bound.
    results = []
    for j, d in enumerate(IDS[:9]):
        for injected in (False, True):
            label = f"{d}-" + ("sky-before-Learn" if injected else "original")
            cfg = json.loads(json.dumps(template))
            cfg.update(channel=d, injections=[])
            cfg["samples"] = binding(a.evidence / f"native-full/samples-{d}.f64")
            cfg["audit_receipt"] = binding(a.evidence / "native-full/receipt.json")
            cfg["trials"] = [v for v in cfg["trials"] if v["id"] == "lowpass"]
            if injected:
                sp = a.output / (label + "-sky.f64")
                sky[:, j].tofile(sp)
                cfg["learning_overlay"] = binding(sp)
            if j < 8:
                dp = a.output / (label + "-subtraction.f64")
                (fresh if injected else frozen)[:, j].tofile(dp)
                cfg["injections"] = [
                    dict(
                        **binding(dp),
                        identity="coherent-diagnostic-subtraction",
                        geometry_binding=geometry,
                        own_detector_geometry=binding(
                            a.evidence / f"geometry-02/pointing-{d}.f64"
                        ),
                        complete_coherent_plan=binding(
                            a.real
                            / "real"
                            / ("injected-plan.json" if injected else "plan.json")
                        ),
                        support_eligibility=False,
                    )
                ]
            spec = a.output / (label + ".json")
            write_json(spec, cfg)
            command = [str(a.executable), str(spec), str(a.output / label)]
            with (a.output / (label + ".log")).open("w") as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
            r = json.loads((a.output / label / "receipt.json").read_text())
            causes = np.fromfile(a.output / label / "lowpass-causes.u8", np.uint8)
            record = dict(
                label=label,
                specification=binding(spec),
                command=command,
                executable=binding(a.executable),
                runtime_transient_spectral_Learn_repeated=True,
                receipt=r,
                cause_counts={
                    str(v): int((causes == v).sum()) for v in np.unique(causes)
                },
            )
            if j < 8:
                actual = np.fromfile(
                    a.output / label / "lowpass-coherent-diagnostic-subtraction.f64",
                    "<f8",
                ).reshape(n, 2)
                z = (incoherent if injected else coherent)[:, j]
                expected = signal.fftconvolve(
                    z, np.array(cfg["fir"])[:, None], axes=0, mode="same"
                )
                valid = causes == 0
                record["conditional_population_replay_max_abs_difference"] = float(
                    np.max(abs(actual[valid] - expected[valid]))
                )
                record["rms_original_scale"] = float(
                    np.sqrt(np.mean(original[:, j] ** 2))
                )
                # Python population uses the same composed extra coherent support;
                # the C++ witness itself remains an explicit overlay LPF replay.
                cg = np.memmap(
                    a.prepare / "coherent-good.u8", np.bool_, "r", shape=(n, 491)
                )[:, d]
                if np.any(cg & ~valid):
                    record[
                        "additional_runtime_exclusions_inside_population_support"
                    ] = int(np.count_nonzero(cg & ~valid))
                else:
                    record[
                        "additional_runtime_exclusions_inside_population_support"
                    ] = 0
            if injected:
                old = np.fromfile(
                    a.output / (f"{d}-original") / "lowpass-causes.u8", np.uint8
                )
                record["changed_native_causes"] = int(np.count_nonzero(old != causes))
            results.append(record)
            write_json(a.output / "receipt.json", results)
            print(
                label,
                r["transient_excluded_cells"],
                record.get("changed_native_causes"),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["evidence", "prepare", "real", "trial", "executable", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    run(p.parse_args())
