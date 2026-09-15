#!/usr/bin/env python3
"""Bounded n12 science experiment; all outputs are conditional, never admitted."""

import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from rtc_coherent_trial import CoherentEvidence, CoherentPlan, erode, runs, fingerprint
from run_rtc_notch_recovery import binding, write_json

ASEC = 206264.80624709636
IDS = [269, 402, 296, 300, 366, 438, 406, 307, 193, 231, 226, 330]
GROUPS = [[0, 2, 4, 6], [1, 3, 5, 7]]
TARGETS = [GROUPS[1], GROUPS[0]]


def mask_from(n, spans):
    m = np.zeros(n, bool)
    for lo, hi in spans:
        m[lo:hi] = True
    return m


def evidence(e):
    native = e / "native-full"
    receipt = json.loads((native / "receipt.json").read_text())
    t = np.fromfile(native / "native-time.f64", "<f8")
    n = len(t)
    pop = {
        r["detector"]: r
        for r in map(json.loads, (native / "population.jsonl").read_text().splitlines())
    }
    original = np.empty((n, len(IDS), 2))
    good = np.zeros((n, len(IDS)), bool)
    for j, d in enumerate(IDS):
        a = np.fromfile(native / f"samples-{d}.f64", "<f8").reshape(n, 4)
        original[:, j] = a[:, :2]
        good[:, j] = mask_from(
            n, [(lo, hi) for lo, hi, cause in pop[d]["support"] if cause == 0]
        ) & a[:, 2:].all(1)
    train = t < t[-1] / 3 - 4
    ev = CoherentEvidence.learn(
        original,
        t,
        good,
        GROUPS,
        train,
        11.006255844577186,
        244,
        "original:VAL0:" + receipt["original_pair_fingerprint_before"],
    )
    plan = CoherentPlan.consider(
        ev, original, t, good, TARGETS, ev.val_identity, "one-common-envelope-original"
    )
    return t, receipt, pop, original, good, plan


def plan_record(p):
    return dict(
        input=p.evidence.input_sha256,
        VAL=p.evidence.val_identity,
        time=p.evidence.time_sha256,
        support=p.evidence.support_sha256,
        plan=p.plan_id,
        donor_groups=[[IDS[j] for j in g] for g in GROUPS],
        target_groups=[[IDS[j] for j in g] for g in TARGETS],
        basis_sha256=[fingerprint(b) for b in p.evidence.bases],
        basis_support=[runs(g) for g in p.evidence.basis_good],
        coefficients=[[c.tolist() for c in cs] for cs in p.coefficients],
        inverse_gram=[[c.tolist() for c in cs] for cs in p.inverse_gram],
        training_support=[[runs(f) for f in fs] for fs in p.fit_support],
        donor_weights=[
            [[float(w.real), float(w.imag)] for w in a]
            for a in p.evidence.donor_weights
        ],
        numerical_parameters="one donor narrow complex weight vector and two fixed target coefficients per coordinate; common envelope sampled, correlated, not independent parameters",
        response="frozen template, frozen projection and full relearning are distinct",
        five_second_projection=False,
        scientific_admission=False,
        production=False,
    )


def prepare(a):
    out = a.output
    out.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    t, receipt, pop, x, good, p = evidence(a.evidence)
    write_json(out / "coherent-plan.json", plan_record(p))
    y, available = p.apply(x, p.evidence.val_identity)
    y.astype("<f8").tofile(out / "coherent-native.f64")
    available.tofile(out / "coherent-available.u8")
    (x - y).astype("<f8").tofile(out / "subtraction.f64")
    summary = []
    for gi, targets in enumerate(TARGETS):
        b = p.evidence.bases[gi]
        for ti, d in enumerate(targets):
            # Held-out simultaneous quadrature power, not broad-band residual.
            evalmask = (t > t[-1] / 3 + 4) & available[:, d]
            bt = b[evalmask]
            inv = np.linalg.inv(np.einsum("ij,ik->jk", bt, bt, optimize=False))
            before = np.einsum(
                "ij,jk->ik",
                inv,
                np.einsum("ij,ik->jk", bt, x[evalmask, d], optimize=False),
                optimize=False,
            )
            after = np.einsum(
                "ij,jk->ik",
                inv,
                np.einsum("ij,ik->jk", bt, y[evalmask, d], optimize=False),
                optimize=False,
            )
            summary.append(
                dict(
                    detector=IDS[d],
                    heldout_quadrature_before=before.tolist(),
                    heldout_quadrature_after=after.tolist(),
                    coefficient=p.coefficients[gi][ti].tolist(),
                    fit_cells=int(p.fit_support[gi][ti].sum()),
                    fit_dof=2,
                )
            )
    # Physical treatment intervals are the exact prior bound finite domain.
    prior = {
        r["detector"]: r
        for r in map(json.loads, gzip.open(a.population, "rt"))
        if str(r["observation"]) == "152390" and r["network"] == 12
    }
    apt = np.fromfile(a.evidence / "geometry-02/apt.f64", "<f8").reshape(491, 6)
    inputs = np.zeros((len(t), 491), bool)
    for d in range(491):
        if prior[d]["bound_chain"] is not None and apt[d, 4] == 0:
            inputs[:, d] = mask_from(len(t), prior[d]["bound_chain"]["input_runs"])
    inputs.tofile(out / "input-good.u8")
    base = np.column_stack([erode(inputs[:, d], 153) for d in range(491)])
    finite = base.copy()
    coherent = base.copy()
    reject = base.copy()
    for j, d in enumerate(IDS[:8]):
        finite[:, d] = erode(inputs[:, d], 519)
        # Direct donor window and target input all stay in the physical domain.
        gi = 0 if j in TARGETS[0] else 1
        donor_good = np.all(inputs[:, [IDS[v] for v in GROUPS[gi]]], axis=1)
        cg = erode(donor_good, 244) & inputs[:, d] & available[:, j]
        coherent[:, d] = erode(cg, 153)
        reject[:, d] = False
    for name, g in [
        ("lowpass", base),
        ("finite", finite),
        ("coherent", coherent),
        ("reject", reject),
    ]:
        g.tofile(out / (name + "-good.u8"))
    # Four existing 10s chunks, selected only from support and temporal thirds.
    # Actual chunk origin is explicitly bound to this native network diagnostic.
    choices = []
    candidates = [
        (lo, hi) for lo, hi in runs(finite[:, 269] & coherent[:, 269]) if hi - lo > 300
    ]
    for wanted in [260.0, 650.0, 1000.0]:
        lo, hi = min(candidates, key=lambda r: abs(t[(r[0] + r[1]) // 2] - wanted))
        mid = (lo + hi) // 2
        block = int(t[mid] // 10)
        xy = (
            np.fromfile(a.evidence / "geometry-02/pointing-269.f64", "<f8").reshape(
                -1, 2
            )
            * ASEC
        )
        choices.append(
            dict(
                kind="supported",
                time=float(t[mid]),
                native_row=mid,
                chunk=block,
                center_arcsec=xy[mid].tolist(),
                support_run=[lo, hi],
            )
        )
    lo, hi = min(candidates, key=lambda r: abs(t[r[0]] - 800))
    # Start of actual finite retained support, deliberately challenged by source width.
    mid = lo
    xy = (
        np.fromfile(a.evidence / "geometry-02/pointing-269.f64", "<f8").reshape(-1, 2)
        * ASEC
    )
    choices.append(
        dict(
            kind="boundary-challenge",
            time=float(t[mid]),
            native_row=mid,
            chunk=int(t[mid] // 10),
            center_arcsec=xy[mid].tolist(),
            support_run=[lo, hi],
        )
    )
    write_json(
        out / "science-design.json",
        dict(
            crossings=choices,
            beam=dict(
                model="50m150GHz Airy",
                FWHM_arcsec=8.483936,
                readout="provisional uniform centered integration;64 midpoint quadrature",
            ),
            compact_peak_mJy_per_beam=100.0,
            extended_peak_mJy_per_beam=25.0,
            extended_apparent_FWHM_arcsec=3 * 8.483936,
            extended_center_arcsec=[0.0, 0.0],
            r_injection_native_ratio=0.2,
            r_leakage_authority="explicit conditional response probe, not measured calibration or dark-channel assumption",
            chunks=sorted(set(v["chunk"] for v in choices)),
            chunk_definition="10s forced reference duration, native n12 origin; conditional diagnostic partition, not admitted PTC binding",
            reference="fixed10-mode existing Cleaner plus existing ordinary naive map projection; all419 originally APT-good detectors,72 already flagged; matched-APT fluxscale/static weights; no JINC or FRUIT qualification",
            map_pixels=401,
            pixel_arcsec=2,
            noise_seeds=list(range(8101, 8109)),
            added_line_phases=[0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
            line_truth="known added copy of the measured common envelope, amplitudes/relative detector phases fixed; empirical original background itself still contains unknown sky/line contributions",
        ),
    )
    write_json(
        out / "receipt.json",
        dict(
            original_pair_sha256=fingerprint(x),
            original_unchanged=fingerprint(x) == p.evidence.input_sha256,
            heldout=summary,
            seconds=time.perf_counter() - start,
            geometry=binding(a.evidence / "geometry-02/receipt.json"),
            population=binding(a.population),
            support={
                name: dict(
                    cells=int(g[::2].sum()),
                    target_cells=int(g[::2][:, IDS[:8]].sum()),
                    static_weight_seconds=float(
                        np.sum(g[::2] / apt[:, 3] ** 2)
                        * 2
                        * receipt["nominal_interval"]
                    ),
                )
                for name, g in [
                    ("lowpass", base),
                    ("finite", finite),
                    ("coherent", coherent),
                    ("reject", reject),
                ]
            },
            support_admission=False,
        ),
    )
    print(
        json.dumps(
            dict(seconds=time.perf_counter() - start, crossings=choices), indent=2
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare"])
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a)
