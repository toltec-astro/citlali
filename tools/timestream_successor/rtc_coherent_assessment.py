#!/usr/bin/env python3
"""Bounded n12 held-out oscillator evidence; no treatment or source admission."""

import argparse
import json
from pathlib import Path
import numpy as np
from rtc_disturbance_burden import records
from run_rtc_notch_recovery import binding, write_json


def fit_coupling(oscillator, targets):
    """Coordinate-local complex least squares, with explicit training arrays."""
    q = np.asarray(oscillator)
    y = np.asarray(targets)
    if q.ndim != 1 or y.ndim != 2 or y.shape[0] != len(q) or len(q) < 3:
        raise ValueError("incomplete complex training support")
    if not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite complex training support")
    power = np.vdot(q, q).real
    if power <= 0:
        raise ValueError("zero oscillator training power")
    return np.einsum("w,wj->j", np.conj(q), y, optimize=False) / power


def group_prediction(values, training, evaluation, donors, targets):
    """One donor-derived complex mode; target coefficients use training only."""
    values = np.asarray(values)
    training, evaluation = np.asarray(training), np.asarray(evaluation)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or training.dtype != bool
        or evaluation.dtype != bool
        or training.shape != (len(values),)
        or evaluation.shape != training.shape
        or np.any(training & evaluation)
        or training.sum() < 3
        or not evaluation.any()
        or not donors
        or not targets
        or set(donors).intersection(targets)
        or len(set(donors)) != len(donors)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("invalid held-out donor/target partition")
    train = values[training][:, donors]
    _, singular, right = np.linalg.svd(train, full_matrices=False)
    if not singular[0] > 0 or np.any(
        np.sum(abs(values[evaluation][:, targets]) ** 2, axis=0) == 0
    ):
        raise ValueError("unavailable zero-power prediction")
    weights = right[0].conj()
    q = np.einsum("wj,j->w", values[:, donors], weights, optimize=False)
    coefficients = fit_coupling(q[training], values[training][:, targets])
    prediction = q[evaluation, None] * coefficients[None, :]
    truth = values[evaluation][:, targets]
    qphase = q / np.maximum(abs(q), np.finfo(float).tiny)
    phase_coefficients = fit_coupling(qphase[training], values[training][:, targets])
    phase_prediction = qphase[evaluation, None] * phase_coefficients[None, :]

    def result(pred):
        return (
            np.sum(abs(truth - pred) ** 2, axis=0) / np.sum(abs(truth) ** 2, axis=0)
        ).tolist()

    return dict(
        training_singular_values=singular.tolist(),
        training_rank1_power_fraction=float(singular[0] ** 2 / np.sum(singular**2)),
        complex_coefficients=[[float(c.real), float(c.imag)] for c in coefficients],
        relative_phase_rad=np.angle(coefficients).tolist(),
        amplitude_ratio=abs(coefficients).tolist(),
        heldout_complex_power_error_ratio=result(prediction),
        heldout_phase_only_power_error_ratio=result(phase_prediction),
    ), q


def assess(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads(a.design.read_text())
    native = a.native
    receipt = json.loads((native / "receipt.json").read_text())
    if (
        receipt["Apply"]
        or receipt["VAL_generation"] != 0
        or receipt["original_pair_fingerprint_before"]
        != receipt["original_pair_fingerprint_after"]
    ):
        raise ValueError("not unchanged original native evidence")
    ids = design["affected"] + design["quiet"]
    n = receipt["fft_samples"]
    dt = receipt["measured_interval"]
    t = np.fromfile(native / "native-time.f64", "<f8")
    meta = list(records(native / "spectra.jsonl"))
    windows = np.memmap(
        native / "windows.f64", "<f8", "r", shape=(receipt["window_records"], 14)
    )
    common = None
    for d in ids:
        for c in (0, 1):
            m = meta[2 * d + c]
            w = windows[m["window_offset"] : m["window_offset"] + m["window_count"]]
            spans = {(int(r[2]), int(r[3])) for r in w if r[3] - r[2] == n}
            common = spans if common is None else common & spans
    chosen = []
    end = -1
    for lo, hi in sorted(common):
        if lo >= end:
            chosen.append((lo, hi))
            end = hi
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) < 24:
        raise ValueError("insufficient common disjoint windows")
    center = np.array([(t[lo] + t[hi - 1]) / 2 for lo, hi in chosen])
    f0 = design["target_initial_hz"]
    h = np.hanning(n)
    z = np.empty((len(chosen), len(ids), 2), complex)
    # Demodulation retains native phase on the original axis. Existing full
    # Hann windows/two-median convention; no interpolation or invalid filling.
    for j, d in enumerate(ids):
        s = np.fromfile(native / f"samples-{d}.f64", "<f8").reshape(-1, 4)
        for k, (lo, hi) in enumerate(chosen):
            if not s[lo:hi, 2:].all() or not np.isfinite(s[lo:hi, :2]).all():
                raise ValueError("invalid original paired support")
            carrier = np.exp(-2j * np.pi * f0 * t[lo:hi])
            for c in (0, 1):
                v = s[lo:hi, c] - meta[2 * d + c]["population_median"]
                v -= np.median(v)
                z[k, j, c] = 2 * np.dot(h * v, carrier) / h.sum()
    z.tofile(a.output / "complex-demod.c128")
    chosen.astype("<i8").tofile(a.output / "windows.i64")
    center.astype("<f8").tofile(a.output / "centers.f64")
    boundaries = [t[-1] * v for v in (1 / 3, 2 / 3)]
    guard = design["between_partition_guard_seconds"]
    training = t[chosen[:, 1] - 1] < boundaries[0] - guard
    evaluations = [
        (t[chosen[:, 0]] > boundaries[0] + guard)
        & (t[chosen[:, 1] - 1] < boundaries[1] - guard),
        t[chosen[:, 0]] > boundaries[1] + guard,
    ]
    output = []
    for c in (0, 1):
        values = z[:, :, c]
        x = values[training, :8]
        scale = np.sqrt(np.mean(abs(x) ** 2, axis=0))
        _, sv, _ = np.linalg.svd(x / scale, full_matrices=False)
        coherence = abs(
            np.einsum("wi,wj->ij", values.conj(), values, optimize=False)
        ) ** 2 / np.outer(
            np.sum(abs(values) ** 2, axis=0), np.sum(abs(values) ** 2, axis=0)
        )
        phase = np.angle(np.einsum("wi,wj->ij", values.conj(), values, optimize=False))
        record = dict(
            coordinate="x" if c == 0 else "r",
            normalized_training_singular_values=sv.tolist(),
            normalized_rank1_fraction=float(sv[0] ** 2 / np.sum(sv**2)),
            coherence=coherence.tolist(),
            cross_phase_rad=phase.tolist(),
            predictions=[],
        )
        for label, donor_ids in design["groups"].items():
            donors = [ids.index(d) for d in donor_ids]
            targets = [j for j, d in enumerate(ids) if d not in donor_ids]
            for ei, ev in enumerate(evaluations):
                p, q = group_prediction(values, training, ev, donors, targets)
                p.update(
                    donor_group=label,
                    donors=donor_ids,
                    targets=[ids[j] for j in targets],
                    evaluation_partition=ei,
                    training_window_indices=np.flatnonzero(training).tolist(),
                    evaluation_window_indices=np.flatnonzero(ev).tolist(),
                )
                angle = np.unwrap(np.angle(q))
                fit = np.polyfit(center[training], angle[training], 1)
                error = np.angle(np.exp(1j * (angle[ev] - np.polyval(fit, center[ev]))))
                p["constant_frequency_fit_hz"] = float(f0 + fit[0] / (2 * np.pi))
                p["constant_frequency_holdout_phase_rms_rad"] = float(
                    np.sqrt(np.mean(error**2))
                )
                delta = np.diff(angle) / (2 * np.pi * np.diff(center))
                p["phase_increment_frequency_quantiles_hz"] = np.quantile(
                    f0 + delta, [0.05, 0.5, 0.95]
                ).tolist()
                p["donor_oscillator_amplitude_quantiles"] = np.quantile(
                    abs(q), [0.05, 0.5, 0.95]
                ).tolist()
                record["predictions"].append(p)
        output.append(record)
    write_json(
        a.output / "assessment.json",
        dict(
            design=binding(a.design),
            input_bindings={
                n: binding(native / n)
                for n in [
                    "receipt.json",
                    "spectra.jsonl",
                    "windows.f64",
                    "native-time.f64",
                ]
                + [f"samples-{d}.f64" for d in ids]
            },
            ids=ids,
            windows=len(chosen),
            training=int(training.sum()),
            evaluations=[int(e.sum()) for e in evaluations],
            target_hz=f0,
            learning="one complex donor mode learned on first third; target couplings fitted on first third only; two later guard-separated thirds held out; donor groups disjoint from prediction targets",
            phase_prediction="stationary-frequency forecast versus simultaneous donor-informed complex trajectory are distinct diagnostics",
            training_support_span_seconds=float(
                center[training][-1] - center[training][0] + n * dt
            ),
            global_model_is_not_a_five_second_operator=True,
            Apply=False,
            physical_origin=None,
            source_protection="unknown and retained",
            results=output,
        ),
    )
    print(
        json.dumps(
            dict(
                windows=len(chosen),
                training=int(training.sum()),
                evaluations=[int(e.sum()) for e in evaluations],
                rank1=[r["normalized_rank1_fraction"] for r in output],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["native", "design", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    assess(p.parse_args())
