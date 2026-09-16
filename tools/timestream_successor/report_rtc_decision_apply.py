#!/usr/bin/env python3
"""Report bounded RTC decisions and paired support; finite x is not exposure."""

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import yaml

ARMS = ("exclusion-control", "donor-continuity")
CAUSES = (
    "retained",
    "producer-invalid",
    "transient-excluded",
    "motion-unavailable",
    "motion-outside-domain",
    "filter-boundary",
    "rejection",
    "below-minimum-speed",
    "insufficient-output-sampling",
    "event-policy-unavailable",
    "accepted-event-excluded",
    "processing-support-unavailable",
)


def load(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def arm_data(root, channel):
    stem = str(channel)
    return dict(
        pair=np.fromfile(root / f"{stem}-filtered.f64", dtype="<f8").reshape(-1, 2),
        conditioned=np.fromfile(root / f"{stem}-conditioned.f64", dtype="<f8").reshape(
            -1, 2
        ),
        state=np.fromfile(root / f"{stem}-state.u8", dtype="u1"),
        cause=np.fromfile(root / f"{stem}-causes.u8", dtype="u1"),
        input_cause=np.fromfile(root / f"{stem}-input-causes.u8", dtype="u1"),
        rows=np.fromfile(root / f"{stem}-rows.i64", dtype="<i8"),
    )


def support_counts(a):
    schedule = np.zeros(len(a["state"]), dtype=bool)
    schedule[a["rows"]] = True
    state = a["state"]
    return dict(
        native_pair_cells=len(state),
        input_causes={
            name: int(np.sum(a["input_cause"] == i)) for i, name in enumerate(CAUSES)
        },
        realized_replaced_native=int(np.sum((state & 4) != 0)),
        scheduled_output_rows=int(schedule.sum()),
        paired_available_output=int(np.sum(schedule & ((state & 3) == 3))),
        x_available_output=int(np.sum(schedule & ((state & 1) != 0))),
        representative_excluded_output=int(np.sum(schedule & ((state & 8) != 0))),
        representative_replaced_output=int(np.sum(schedule & ((state & 4) != 0))),
        nonrepresentative_influence_output=int(
            np.sum(schedule & ((state & 16) != 0) & ((state & 4) == 0))
        ),
        pair_available_representative_not_excluded=int(
            np.sum(schedule & ((state & 3) == 3) & ((state & 8) == 0))
        ),
        filter_boundary_native=int(np.sum(a["cause"] == 5)),
    )


def compare(a, b, coordinate=0, region=None):
    mask_a = np.zeros(len(a["state"]), dtype=bool)
    mask_b = mask_a.copy()
    mask_a[a["rows"]] = True
    mask_b[b["rows"]] = True
    mask_a &= (a["state"] & (1 << coordinate)) != 0
    mask_b &= (b["state"] & (1 << coordinate)) != 0
    if region is not None:
        mask_a &= region
        mask_b &= region
    both = mask_a & mask_b
    delta = a["pair"][both, coordinate] - b["pair"][both, coordinate]
    if not np.all(np.isfinite(delta)):
        raise ValueError("availability claims nonfinite output")
    return dict(
        matched=int(both.sum()),
        only_first=int((mask_a & ~mask_b).sum()),
        only_second=int((mask_b & ~mask_a).sum()),
        rms=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
        maximum=float(np.max(np.abs(delta))) if len(delta) else None,
    )


def report(config_path, natural, injected, output):
    output.mkdir(exist_ok=False)
    cfg = json.loads(config_path.read_text())
    receipt = load(natural / "receipt.yaml")
    decisions = load(natural / "event-decisions.yaml")
    dt = receipt["cadence_interval_seconds"]
    summary = dict(
        events=dict(Counter(d["class"] for d in decisions)),
        proposed_prerequisites=[
            d["event"] for d in decisions if d["proposed_isolation_prerequisites"]
        ],
        natural_donor_execution=False,
        channels={},
        denominator_pair_cells=0,
        cadence_seconds=dt,
    )
    for detector in cfg["detectors"]:
        ch = detector["channel"]
        raw = np.fromfile(detector["samples"]["path"], dtype="<f8").reshape(-1, 4)
        summary["denominator_pair_cells"] += int(
            np.sum((raw[:, 2] == 1) & (raw[:, 3] == 1))
        )
        a, b = [arm_data(natural / arm, ch) for arm in ARMS]
        summary["channels"][ch] = dict(
            control=support_counts(a),
            donor=support_counts(b),
            x_comparison=compare(a, b),
            r_comparison=compare(a, b, 1),
        )
    summary["denominator_detector_seconds"] = (
        summary["denominator_pair_cells"] * receipt["native_integration_seconds"]
    )
    fixture_receipt = load(injected / "receipt.yaml")
    model = fixture_receipt["declared_contaminant"]
    ch, row = model["channel"], model["native_row"]
    fixture_decisions = load(injected / "event-decisions.yaml")
    accepted = [
        d for d in fixture_decisions if d["class"] == "accepted-declared-contaminant"
    ]
    summary["fixture"] = dict(
        channel=ch, native_row=row, accepted_events=accepted, arms={}
    )
    reference = arm_data(natural / ARMS[0], ch)
    region = np.abs(np.arange(receipt["rows"]) - row) <= int(5 / dt)
    affected = np.zeros(receipt["rows"], dtype=bool)
    for d in accepted:
        first, last = d["affected"]
        affected[first:last] = True
    for arm in ARMS:
        a = arm_data(injected / arm, ch)
        details = support_counts(a)
        details["outside_replaced_x_vs_reference"] = compare(
            a, reference, region=region & ~affected
        )
        details["outside_replaced_r_vs_reference"] = compare(
            a, reference, 1, region=region & ~affected
        )
        # Native filtered residuals include ringing outside replaced cells.
        available = (
            region
            & ~affected
            & np.isfinite(a["pair"][:, 0])
            & np.isfinite(reference["pair"][:, 0])
        )
        residual = a["pair"][available, 0] - reference["pair"][available, 0]
        details["native_x_residual_energy_outside_fill"] = float(
            np.sum(residual**2) * dt
        )
        delta = a["conditioned"][:, 0] - reference["conditioned"][:, 0]
        differences = np.diff(delta)
        valid = np.isfinite(differences) & (region[:-1] | region[1:])
        details["maximum_conditioned_residual_adjacent_change"] = (
            float(np.max(np.abs(differences[valid]))) if valid.any() else None
        )
        # Accepted runtime D2 products retain exact per-arm window populations.
        learned = load(injected / arm / "apply-receipt.yaml")
        details["relearned_windows"] = [
            dict(
                after_lowpass=stage["after_lowpass"],
                available=s["available"],
                cause=s["cause"],
                coordinate=s["coordinate"],
                windows=len(s.get("windows", [])),
            )
            for stage in learned["relearned"]
            for s in stage["spectra"]
            if s["channel"] == ch
        ]
        summary["fixture"]["arms"][arm] = details
    summary["fixture"]["arm_x_comparison"] = compare(
        arm_data(injected / ARMS[0], ch),
        arm_data(injected / ARMS[1], ch),
        region=region,
    )
    # Spectral comparison only over exactly shared realized D2 window support.
    for arm in ARMS:
        ar = load(injected / arm / "apply-receipt.yaml")
        rr = load(natural / ARMS[0] / "apply-receipt.yaml")
        changes = []
        for stage, refstage in zip(ar["relearned"], rr["relearned"]):
            for s, r in zip(stage["spectra"], refstage["spectra"]):
                if s["channel"] != ch:
                    continue
                same = [w["rows"] for w in s.get("windows", [])] == [
                    w["rows"] for w in r.get("windows", [])
                ]
                changes.append(
                    dict(
                        after_lowpass=stage["after_lowpass"],
                        coordinate=s["coordinate"],
                        identical_window_population=same,
                    )
                )
        summary["fixture"]["arms"][arm]["D2_comparability"] = changes
    # Paired diagnostics on the intersection of the actual runtime D2 windows.
    # This is an inspection of realized products, not fresh classification evidence.
    all_arms = [arm_data(injected / arm, ch) for arm in ARMS]
    receipts = [load(injected / arm / "apply-receipt.yaml") for arm in ARMS]
    refreceipt = load(natural / ARMS[0] / "apply-receipt.yaml")
    diagnostics = []
    for stage_index in range(2):
        for coordinate in range(2):
            spectra = [
                next(
                    s
                    for s in rec["relearned"][stage_index]["spectra"]
                    if s["channel"] == ch and s["coordinate"] == coordinate
                )
                for rec in [*receipts, refreceipt]
            ]
            supports = [
                {tuple(w["rows"]) for w in spec.get("windows", [])} for spec in spectra
            ]
            common = sorted(set.intersection(*supports))
            key = "pair" if stage_index else "conditioned"
            powers = [[] for _ in range(3)]
            nfft = int(round(4 / dt))
            window = np.hanning(nfft)
            freq = np.fft.rfftfreq(nfft, dt)
            for first, last in common:
                if last <= row - 5 / dt or first >= row + 5 / dt:
                    continue
                for population, data in zip(powers, [*all_arms, reference]):
                    y = data[key][first:last, coordinate].copy()
                    if not np.all(np.isfinite(y)):
                        raise ValueError(
                            "runtime spectral window includes unavailable numerical input"
                        )
                    y -= np.median(y)
                    y = np.pad(y, (0, nfft - len(y)))
                    psd = np.abs(np.fft.rfft(y * window)) ** 2 / (
                        np.sum(window**2) / dt
                    )
                    psd[1:-1] *= 2
                    population.append(psd)
            d = dict(
                after_lowpass=bool(stage_index),
                coordinate=coordinate,
                actual_windows=[len(x) for x in supports],
                local_matched_windows=len(powers[0]),
                convention="same-window median/symmetric-Hann/native-fs PSD; diagnostic only",
            )
            if powers[0]:
                average = [np.mean(p, axis=0) for p in powers]
                d["bands"] = []
                for low, high in [(0, 5), (5, 20), (20, 40), (40, 1 / (2 * dt) + 1)]:
                    mask = (freq >= low) & (freq < high)
                    df = 1 / (nfft * dt)
                    d["bands"].append(
                        dict(
                            hz=[low, high],
                            control_power=float(np.sum(average[0][mask]) * df),
                            donor_power=float(np.sum(average[1][mask]) * df),
                            reference_power=float(np.sum(average[2][mask]) * df),
                        )
                    )
            diagnostics.append(d)
    summary["fixture"]["matched_spectral_diagnostics"] = diagnostics
    summary["runtime"] = {}
    for name, folder in [("natural", natural), ("injected", injected)]:
        r = load(folder / "receipt.yaml")
        summary["runtime"][name] = {
            k: v for k, v in r.items() if k.endswith("_seconds")
        }
        for arm in ARMS:
            a = load(folder / arm / "apply-receipt.yaml")
            summary["runtime"][name][arm] = {
                k: v for k, v in a.items() if k.endswith("_seconds")
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "| Event | Ch | Native row | Class | Paired extent | Operation unavailable | Background / target domain | Cohort coincidences | Treatment / donor reason |",
        "|---:|---:|---:|---|---|---|---|---|---|",
    ]
    treatments = {
        "admitted-level-shift": "Full existing processing-scan exclusion; shift, never donor-fill",
        "no-resolved-excursion": "Guarded operation unavailable; no seeded extent to admit",
        "unresolved-extent": "Physical-run operation unavailable; no finite extent",
        "isolated-admission-unavailable": "Paired guarded/affected operation unavailable; isolated-admission predicate missing",
    }
    for d in decisions:
        lines.append(
            f"| {d['event']} | {d['channel']} | {d['seed_earlier_row']} | {d['class']} | {d['affected']} | {d['operation_unavailable']} | {d['background_available']} / {d['target_domain_available']} | {d['cohort_coincident_events']} | {treatments[d['class']]} |"
        )
    (output / "EVENTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    t = np.fromfile(natural / "native-time.f64", dtype="<f8")
    t -= t[row]
    detector = next(d for d in cfg["detectors"] if d["channel"] == ch)
    raw = np.fromfile(detector["samples"]["path"], dtype="<f8").reshape(-1, 4)
    for c in range(2):
        axes[0, c].plot(
            t[region], raw[region, c], color=".7", lw=0.6, label="untouched original"
        )
        for arm, color in zip(ARMS, ("tab:blue", "tab:orange")):
            a = arm_data(injected / arm, ch)
            axes[0, c].plot(
                t[region], a["conditioned"][region, c], color=color, lw=0.9, label=arm
            )
            residual = a["pair"][:, c] - reference["pair"][:, c]
            axes[1, c].plot(t[region], residual[region], color=color, lw=0.9, label=arm)
        axes[0, c].set_title(f"Native conditioned {'xr'[c]}; gaps remain gaps")
        axes[1, c].set_title(f"Filtered {'xr'[c]} minus untouched reference")
        axes[1, c].set_xlabel("Seconds from declared disturbance")
        for ax in axes[:, c]:
            ax.axvspan(-dt / 2, dt / 2, color="gold", alpha=0.35)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"152390 / n12 / channel {ch}: one known paired contaminant; x continuity is not independent exposure"
    )
    fig.savefig(output / "contaminant-comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--natural", type=Path, required=True)
    p.add_argument("--injected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report(a.config, a.natural, a.injected, a.output)
