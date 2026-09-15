#!/usr/bin/env python3
"""Fixed experiment campaign; no parameter search or production admission."""

import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy import signal, special
from rtc_coherent_trial import CoherentEvidence, CoherentPlan
from run_rtc_coherent_science import evidence, plan_record, IDS, GROUPS, TARGETS, ASEC
from run_rtc_notch_recovery import binding, write_json

ARMS = ("lowpass", "finite", "coherent", "reject")


def sky(time_axis, xy, design):
    """One sky at own-detector pointing; provisional centered readout integral.

    Compact Airy intensity uses the same 50 m / 150 GHz definition as the
    accepted AST experiment. Extended Gaussian is an explicitly apparent
    (beam-convolved) template, not an intrinsic-source deconvolution claim.
    """
    value = np.zeros(len(time_axis))
    dt = np.median(np.diff(time_axis))
    v = np.gradient(xy, time_axis, axis=0)
    # The offset is at most half one native interval. This is a separately
    # declared linear-within-sample evaluation of measured native pointing.
    for cross in design["crossings"]:
        center = np.asarray(cross["center_arcsec"])
        radius = np.linalg.norm(xy - center, axis=1)
        active = (
            radius < 80
        )  # Airy response beyond80arcsec retained below via center evaluation.
        u = np.pi * 50 / (299792458 / 150e9) * radius / ASEC
        response = np.ones(len(u))
        nz = u != 0
        response[nz] = (2 * special.j1(u[nz]) / u[nz]) ** 2
        value += design["compact_peak_mJy_per_beam"] * response
        integrated = np.zeros(active.sum())
        for offset in (np.arange(64) + 0.5) / 64 - 0.5:
            radius = np.linalg.norm(
                xy[active] + v[active] * (offset * dt) - center, axis=1
            )
            u = np.pi * 50 / (299792458 / 150e9) * radius / ASEC
            z = np.ones(len(u))
            nz = u != 0
            z[nz] = (2 * special.j1(u[nz]) / u[nz]) ** 2
            integrated += z / 64
        value[active] += design["compact_peak_mJy_per_beam"] * (
            integrated - response[active]
        )
    center = np.asarray(design["extended_center_arcsec"])
    sigma = design["extended_apparent_FWHM_arcsec"] / np.sqrt(8 * np.log(2))
    extended = np.zeros(len(time_axis))
    for offset in (np.arange(64) + 0.5) / 64 - 0.5:
        radius2 = np.sum((xy + v * (offset * dt) - center) ** 2, axis=1)
        extended += np.exp(-0.5 * radius2 / sigma**2) / 64
    return value + design["extended_peak_mJy_per_beam"] * extended


def filtered_block(values, kernel, rows):
    half = len(kernel) // 2
    lo = int(rows[0]) - half
    hi = int(rows[-1]) + half + 1
    if lo < 0 or hi > len(values):
        raise ValueError("unavailable real convolution support")
    z = signal.fftconvolve(
        values[lo:hi], np.asarray(kernel)[:, None], axes=0, mode="valid"
    )
    return z[rows - rows[0]]


def make_plan(x, t, good, train, val, name):
    ev = CoherentEvidence.learn(x, t, good, GROUPS, train, 11.006255844577186, 244, val)
    return CoherentPlan.consider(ev, x, t, good, TARGETS, val, name)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    t, receipt, pop, original, good, p0 = evidence(a.evidence)
    n = len(t)
    design = json.loads((a.prepare / "science-design.json").read_text())
    apt = np.fromfile(a.evidence / "geometry-02/apt.f64", "<f8").reshape(491, 6)
    telescope = np.fromfile(a.evidence / "geometry-02/telescope.f64", "<f8").reshape(
        n, 6
    )
    params = json.loads(a.trial.read_text())
    fir = np.array(params["fir"])
    notch = np.array(
        next(v["finite_notch"] for v in params["trials"] if v["id"] == "finite-6s")
    )
    kernels = {
        "lowpass": fir,
        "finite": np.convolve(notch, fir),
        "coherent": fir,
        "reject": fir,
    }
    masks = {
        k: np.memmap(
            a.prepare / (k + "-good.u8"), dtype=np.bool_, mode="r", shape=(n, 491)
        )
        for k in ARMS
    }
    common = np.logical_and.reduce([v for v in masks.values()])
    rows = {
        str(c): np.flatnonzero(
            (t >= c * 10) & (t < (c + 1) * 10) & (np.arange(n) % 2 == 0)
        )
        for c in design["chunks"]
    }
    sky_selected = np.zeros_like(original)
    for j, d in enumerate(IDS):
        xy = (
            np.fromfile(a.evidence / f"geometry-02/pointing-{d}.f64", "<f8").reshape(
                n, 2
            )
            * ASEC
        )
        s = sky(t, xy, design) / apt[d, 2]
        sky_selected[:, j, 0] = s
        sky_selected[:, j, 1] = design["r_injection_native_ratio"] * s
    sky_selected.tofile(a.output / "sky-selected.f64")
    baseline_coherent, _ = p0.apply(original, p0.evidence.val_identity)
    known_line = (
        original - baseline_coherent
    )  # named injected component, not a cleaned-noise truth
    # No added line to quiet controls. Coherence is learned anew for each realization.
    known_line[:, 8:] = 0
    known_line.tofile(a.output / "known-added-line-selected.f64")
    inputs = []
    summary = []
    chosen_seeds = (
        [-1]
        if a.real_only
        else ([-10, -11, -12, -13, -14] if a.phase_controls else design["noise_seeds"])
    )
    for seed in chosen_seeds:
        start = time.perf_counter()
        label = (
            "real"
            if seed == -1
            else (f"control-{-seed - 10}" if seed <= -10 else f"noise-{seed}")
        )
        folder = a.output / label
        folder.mkdir()
        phase = np.ones(n // 2 + 1, complex)
        if seed >= 0:
            phase = np.exp(
                1j * np.random.default_rng(seed).uniform(-np.pi, np.pi, n // 2 + 1)
            )
            phase[0] = 1
        phase.tofile(folder / "shared-frequency-phase.c128")
        selected = original.copy()
        if seed <= -10:
            selected = np.zeros_like(original)
            angle = (-seed - 10) * np.pi / 2
            if seed != -14:
                for gi, targets in enumerate(TARGETS):
                    # Controlled line truth: exactly declared common-envelope
                    # phase rotation, separate from predictive real-data evidence.
                    b = p0.evidence.bases[gi]
                    rot = np.array(
                        [
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                        ]
                    )
                    for ti, j in enumerate(targets):
                        cc = np.einsum(
                            "ij,jk->ik", rot, p0.coefficients[gi][ti], optimize=False
                        )
                        selected[:, j] = np.einsum("ij,jk->ik", b, cc, optimize=False)
        elif seed != -1:
            for j in range(len(IDS)):
                selected[:, j] = np.fft.irfft(
                    np.fft.rfft(original[:, j] - original[:, j].mean(0), axis=0)
                    * phase[:, None],
                    n=n,
                    axis=0,
                )
            selected += known_line
        val = f"scenario-{label}:initialVAL0:fixed-original-declarations"
        if seed == -14:
            # Source-only response of the ORIGINAL frozen projection. Preserve
            # its actual donor/input provenance; never relabel a learned basis
            # as evidence produced by an all-zero controlled input.
            plan = p0
            before, _ = p0.apply(original, p0.evidence.val_identity)
            projected, _ = p0.apply(
                original,
                p0.evidence.val_identity,
                injection=sky_selected,
                response="projection",
            )
            coherent = np.zeros_like(selected)
            template = sky_selected.copy()
            projection = projected - before
            relearned = projection
            write_json(folder / "plan.json", plan_record(p0))
            write_json(
                folder / "injected-plan.json",
                dict(
                    available=False,
                    reason="source-only frozen original projection response; no absent oscillator learned or admitted",
                    original_plan=binding(folder / "plan.json"),
                ),
            )
        else:
            plan = make_plan(
                selected, t, good, p0.evidence.training, val, label + "-uninjected"
            )
            injected_plan = make_plan(
                selected + sky_selected,
                t,
                good,
                p0.evidence.training,
                val,
                label + "-sky-before-Learn",
            )
            write_json(folder / "plan.json", plan_record(plan))
            write_json(folder / "injected-plan.json", plan_record(injected_plan))
            coherent, _ = plan.apply(selected, val)
            template, _ = plan.apply(selected, val, injection=sky_selected)
            projection, _ = plan.apply(
                selected, val, injection=sky_selected, response="projection"
            )
            relearned, _ = injected_plan.apply(selected + sky_selected, val)
        # Preserve full native selected-pair responses for numerical and learning audits.
        for name, z in [
            ("coherent", coherent),
            ("template", template),
            ("projection", projection),
            ("relearned", relearned),
        ]:
            z.astype("<f8").tofile(folder / (name + "-selected.f64"))
        variants = {arm: {"none": None, "sky": None} for arm in ARMS}
        if seed == -1:
            variants["coherent"].update(template=None, projection=None)
        arrays = {}
        for arm, iv in variants.items():
            for inj in iv:
                for chunk, rr in rows.items():
                    for coord in [0, 1] if seed == -1 else [0]:
                        name = f"{arm}-{inj}-{chunk}-{coord}"
                        arrays[name] = np.memmap(
                            folder / (name + ".f64"), "<f8", "w+", shape=(len(rr), 491)
                        )
        for d in range(491):
            raw = np.fromfile(
                a.evidence / f"native-full/samples-{d}.f64", "<f8"
            ).reshape(n, 4)
            admitted = raw[:, 2:].all(1)
            if not np.isfinite(raw[admitted, :2]).all():
                raise ValueError("unexpected nonfinite admitted original")
            values = np.where(admitted[:, None], raw[:, :2], 0)
            # Already invalid channels remain excluded. Their placeholders never enter PCA/map.
            if seed <= -10:
                values = np.zeros_like(values)
            elif seed != -1:
                values = np.fft.irfft(
                    np.fft.rfft(values - values.mean(0), axis=0) * phase[:, None],
                    n=n,
                    axis=0,
                )
            if d in IDS:
                j = IDS.index(d)
                values = selected[:, j]
                sx = sky_selected[:, j]
            else:
                xy = (
                    np.fromfile(
                        a.evidence / f"geometry-02/pointing-{d}.f64", "<f8"
                    ).reshape(n, 2)
                    * ASEC
                )
                # Only these four chunks and the maximum real FIR halo are needed.
                sx = np.zeros((n, 2))
                if apt[d, 4] == 0:
                    for rr in rows.values():
                        lo = int(rr[0]) - 519
                        hi = int(rr[-1]) + 520
                        s = sky(t[lo:hi], xy[lo:hi], design) / apt[d, 2]
                        sx[lo:hi, 0] = s
                        sx[lo:hi, 1] = s * design["r_injection_native_ratio"]
            for arm, iv in variants.items():
                k = kernels[arm] if d in IDS[:8] else fir
                for inj in iv:
                    z = values if inj == "none" else values + sx
                    if arm == "coherent" and d in IDS[:8]:
                        j = IDS.index(d)
                        z = {
                            "none": coherent,
                            "sky": relearned,
                            "template": template,
                            "projection": projection,
                        }[inj][:, j]
                    for chunk, rr in rows.items():
                        f = filtered_block(z, k, rr) * apt[d, 2]
                        for coord in [0, 1] if seed == -1 else [0]:
                            arrays[f"{arm}-{inj}-{chunk}-{coord}"][:, d] = f[:, coord]
            if d % 100 == 0:
                print(
                    label,
                    "detector",
                    d,
                    "seconds",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
        for arr in arrays.values():
            arr.flush()
        jobs = []
        for chunk, rr in rows.items():
            telescope[rr].astype("<f8").tofile(folder / (chunk + "-telescope.f64"))
            common[rr].astype("<f8").tofile(folder / (chunk + "-common.f64"))
            for arm in ARMS:
                masks[arm][rr].astype("<f8").tofile(
                    folder / (chunk + "-" + arm + "-good.f64")
                )
        for name in arrays:
            arm, inj, chunk, coord = name.split("-")
            gj = binding(folder / (chunk + "-" + arm + "-good.f64"))
            cj = binding(folder / (chunk + "-common.f64"))
            jobs.append(
                dict(
                    name=name,
                    rows=len(rows[chunk]),
                    data=binding(folder / (name + ".f64")),
                    good=gj,
                    telescope=binding(folder / (chunk + "-telescope.f64")),
                    pca=True,
                    map_supports=[
                        dict(name="own", good=gj),
                        dict(name="common", good=cj),
                    ],
                )
            )
        cfg = dict(
            mode="downstream",
            apt=binding(a.evidence / "geometry-02/apt.f64"),
            output_hz=1 / (2 * receipt["nominal_interval"]),
            pixel_arcsec=design["pixel_arcsec"],
            map_pixels=design["map_pixels"],
            jobs=jobs,
        )
        write_json(folder / "downstream.json", cfg)
        exe = binding(a.executable)
        before = time.perf_counter()
        command = [
            str(a.executable),
            str(folder / "downstream.json"),
            str(folder / "downstream"),
        ]
        inputs.append(
            dict(
                command=command,
                executable=exe,
                specification=binding(folder / "downstream.json"),
            )
        )
        with (folder / "downstream.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary.append(
            dict(
                scenario=label,
                total_seconds=time.perf_counter() - start,
                downstream_seconds=time.perf_counter() - before,
                jobs=len(jobs),
                noise_independent_realization=seed >= 0,
                controlled_line_phase_rad=(
                    (-seed - 10) * np.pi / 2 if seed <= -10 and seed != -14 else None
                ),
                source_only_control=seed == -14,
                training_support="plan.json",
                source_injection_all_donors_targets=True,
                transient_snapshot="fixed existing exclusions in this conditional population simulation; separate exact runtime injection witness tests rerun transient/spectral Learn",
                production=False,
            )
        )
        write_json(
            a.output / "receipt.json",
            dict(
                invocations=inputs,
                scenarios=summary,
                design=binding(a.prepare / "science-design.json"),
                prepare=binding(a.prepare / "receipt.json"),
                trial=binding(a.trial),
                conditional=True,
                production=False,
            ),
        )
        print(label, "complete", summary[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["evidence", "prepare", "output", "trial", "executable"]:
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--real-only", action="store_true")
    p.add_argument("--phase-controls", action="store_true")
    run(p.parse_args())
