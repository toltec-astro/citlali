"""Inert r-continuity candidates; no modification of accepted RTC Apply/policy."""

from pathlib import Path
import hashlib
import json
import subprocess
import argparse
import numpy as np
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--replay-binary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
E = args.output
E.mkdir(exist_ok=False)
np.seterr(all="raise")
P = args.reference
N = P / "natural-d92cd3a51"


def load(p):
    return yaml.load(p.read_text(), Loader=yaml.CSafeLoader)


cfg = load(P / "input.json")
receipt = load(N / "exclusion-control/apply-receipt.yaml")
plans = {d["channel"]: d for d in receipt["realized_detectors"]}
channels = [d["channel"] for d in cfg["detectors"]]
samples = np.stack(
    [np.fromfile(d["samples"]["path"], "<f8").reshape(-1, 4) for d in cfg["detectors"]],
    axis=1,
)
n = len(samples)
time = np.fromfile(N / "native-time.f64", "<f8")
base_r = samples[:, :, 1].copy()
common = np.ones(n, bool)
for ch in channels:
    common &= (np.fromfile(N / f"exclusion-control/{ch}-state.u8", "u1") & 3) == 3
guard = int(np.ceil(5 / 0.008192))
count = np.r_[0, np.cumsum(~common)]
q = np.arange(guard, n - guard)
eligible = q[(q % 2 == 0) & ((count[q + guard + 1] - count[q - guard]) == 0)]
assert len(eligible) > 0
cases = [
    dict(
        id="controlled-269-19340",
        channel=269,
        center=19340,
        length=1,
        kind="existing controlled paired contaminant",
    )
]
for fraction, length in [(1 / 3, 1), (2 / 3, 3)]:
    center = int(eligible[np.argmin(np.abs(eligible - fraction * (n - 1)))])
    for ch in [193, 269, 307]:
        cases.append(
            dict(
                id=f"heldout-{ch}-{center}-n{length}",
                channel=ch,
                center=center,
                length=length,
                kind="r-only hidden sample test; original x unchanged",
            )
        )
(E / "case-selection.json").write_text(json.dumps(cases, indent=2) + "\n")


def predict(masked, case):
    """Uses no hidden target values, no x, no flxscale, no output error ranking."""
    ch = case["channel"]
    d = channels.index(ch)
    c = case["center"]
    length = case["length"]
    a = c - length // 2
    b = a + length
    t = time - time[c]
    train = (abs(t) >= 0.05) & (abs(t) <= 2.05)
    assert np.all(np.isnan(masked[a:b, d]))
    assert np.all(np.isfinite(masked[train]))
    assert np.all(common[train]) and np.all(common[a - 1 : b + 1])
    # Same two-second cubic-background family, normalized for numerical scale.
    design = np.polynomial.polynomial.polyvander(t[train] / 2.05, 3)
    coef = np.linalg.lstsq(design, masked[train], rcond=None)[0]
    assert np.all(np.isfinite(coef))
    residual = masked[train] - np.einsum("ij,jk->ik", design, coef, optimize=False)
    rows = np.arange(a - 1, b + 1)
    u = (time[rows] - time[a - 1]) / (time[b] - time[a - 1])
    population = []
    fits = []
    for peer in range(len(channels)):
        if peer == d:
            continue
        z = residual[:, peer]
        y = residual[:, d]
        den = float(np.sum(z * z))
        if not den > 0:
            raise ValueError("r transfer unavailable: zero residual variance")
        slope = float(np.sum(z * y) / den)
        z_eval = masked[rows, peer] - np.polynomial.polynomial.polyval(
            t[rows] / 2.05, coef[:, peer]
        )
        population.append(slope * z_eval)
        fits.append(
            dict(
                channel=channels[peer],
                r_residual_slope=slope,
                training_correlation=float(np.corrcoef(z, y)[0, 1]),
            )
        )
    median = np.median(population, axis=0)
    donor_line = np.interp(u, [0, 1], [median[0], median[-1]])
    bg = np.polynomial.polynomial.polyval(t[rows] / 2.05, coef[:, d])
    taper = 16 * (u * (1 - u)) ** 2
    donor = (bg + taper * (median - donor_line))[1:-1]
    linear = np.interp(
        time[a:b], [time[a - 1], time[b]], [masked[a - 1, d], masked[b, d]]
    )
    cubic = bg[1:-1]
    return dict(target_bracketing=linear, r_donor=donor, target_cubic=cubic), dict(
        training_native_ranges=[
            [
                int(np.flatnonzero(train)[0]),
                int(np.flatnonzero(train & (t < 0))[-1]) + 1,
            ],
            [
                int(np.flatnonzero(train & (t > 0))[0]),
                int(np.flatnonzero(train)[-1]) + 1,
            ],
        ],
        training_samples=int(train.sum()),
        donors=fits,
        hidden_range=[a, b],
        target_cubic=coef[:, d].tolist(),
    )


def replay(values, ch, runs, label, skip=True):
    root = E / "products"
    root.mkdir(exist_ok=True)
    raw = root / f"{label}-raw-r.f64"
    raw.write_bytes(values.astype("<f8").tobytes())
    lp = root / f"{ch}-lp.f64"
    lp.write_bytes(np.array(plans[ch]["lowpass_FIR"], "<f8").tobytes())
    notch = root / f"{ch}-notch.f64"
    notch.write_bytes(np.array(plans[ch]["finite_notch_FIR"] or [], "<f8").tobytes())
    rpath = root / f"{label}-runs.i64"
    rpath.write_bytes(np.array(runs, "<i8").tobytes())
    out = root / f"{label}-filtered-r.f64"
    subprocess.run(
        [
            str(args.replay_binary),
            str(raw),
            str(lp),
            str(notch),
            str(rpath),
            str(out),
            "1" if skip else "0",
        ],
        check=True,
    )
    return np.fromfile(out, "<f8")


references = {}
parity = []
for ch in [193, 269, 307]:
    expected = np.fromfile(N / f"exclusion-control/{ch}-filtered.f64", "<f8").reshape(
        -1, 2
    )[:, 1]
    actual = replay(
        base_r[:, channels.index(ch)],
        ch,
        plans[ch]["admitted_runs"],
        f"reference-{ch}",
        skip=False,
    )
    assert np.array_equal(np.isfinite(actual), np.isfinite(expected))
    assert np.array_equal(actual[np.isfinite(actual)], expected[np.isfinite(expected)])
    references[ch] = actual
    parity.append(
        dict(
            channel=ch,
            exact_finite_value_equality=True,
            exact_availability_equality=True,
        )
    )

records = []
for case in cases:
    ch = case["channel"]
    d = channels.index(ch)
    c = case["center"]
    length = case["length"]
    a = c - length // 2
    b = a + length
    masked = base_r.copy()
    masked[a:b, d] = np.nan
    estimates, fit = predict(masked, case)
    # The fitting boundary refuses an unmasked target: reference values are
    # available only to the separate, subsequent error-accounting path.
    try:
        predict(base_r, case)
    except AssertionError:
        pass
    else:
        raise RuntimeError("predictor accepted exposed held-out target values")
    runs = plans[ch]["admitted_runs"]
    split = []
    for first, last in runs:
        if first < b and a < last:
            if first < a:
                split.append([first, a])
            if b < last:
                split.append([b, last])
        else:
            split.append([first, last])
    excluded = replay(base_r[:, d], ch, split, case["id"] + "-excluded", skip=False)
    missing = replay(masked[:, d], ch, runs, case["id"] + "-x-only")
    if case["id"].startswith("controlled"):
        for name, value in [
            ("exclusion-control", excluded),
            ("donor-continuity", missing),
        ]:
            expected = np.fromfile(
                P / f"injected-d92cd3a51/{name}/{ch}-filtered.f64", "<f8"
            ).reshape(-1, 2)[:, 1]
            assert np.array_equal(np.isfinite(value), np.isfinite(expected))
            assert np.array_equal(
                value[np.isfinite(value)], expected[np.isfinite(expected)]
            )
    ref = references[ch]
    schedule = np.arange(n) % 2 == 0
    x_reference = np.fromfile(
        N / f"exclusion-control/{ch}-filtered.f64", "<f8"
    ).reshape(-1, 2)[:, 0]
    if case["id"].startswith("controlled"):
        fixed_x = np.fromfile(
            P / f"injected-d92cd3a51/donor-continuity/{ch}-filtered.f64", "<f8"
        ).reshape(-1, 2)[:, 0]
    else:
        fixed_x = x_reference
    hidden = np.zeros(n, bool)
    hidden[a:b] = True
    local = (abs(time - time[c]) <= 5) & schedule
    neighbor = local & ~hidden & np.isfinite(ref) & ~np.isfinite(missing)
    assert np.all(np.isfinite(fixed_x[neighbor]))
    relation_rows = local & ~hidden & np.isfinite(ref)
    record = dict(
        **case,
        fit=fit,
        reference_hidden_values=base_r[a:b, d].tolist(),
        x_unchanged_between_experimental_r_arms=True,
        unmasked_target_rejected=True,
        additional_paired_original_representatives=int(neighbor.sum()),
        replaced_scheduled_representatives=int(np.count_nonzero(schedule & hidden)),
        screening_and_other_masks_unchanged=True,
        methods={},
    )
    for method, estimate in estimates.items():
        values = masked[:, d].copy()
        values[a:b] = estimate
        out = replay(values, ch, runs, case["id"] + "-" + method)
        assert np.array_equal(np.isfinite(out), np.isfinite(ref))
        err = out - ref
        e = err[neighbor]
        rms = float(np.sqrt(np.mean(e * e)))
        maximum = float(np.max(abs(e)))
        if len(e) > 1 and np.std(e) > 0 and np.std(e[1:]) > 0:
            corr = float(np.corrcoef(e[:-1], e[1:])[0, 1])
        else:
            corr = None
        x = fixed_x[relation_rows]
        r0 = ref[relation_rows]
        r1 = out[relation_rows]
        relation = dict(
            reference_xr_correlation=float(np.corrcoef(x, r0)[0, 1]),
            untouched_pair_xr_correlation=float(
                np.corrcoef(x_reference[relation_rows], r0)[0, 1]
            ),
            reconstructed_xr_correlation=float(np.corrcoef(x, r1)[0, 1]),
            reference_xr_covariance=float(np.cov(x, r0, ddof=0)[0, 1]),
            reconstructed_xr_covariance=float(np.cov(x, r1, ddof=0)[0, 1]),
        )
        record["methods"][method] = dict(
            estimate_r=estimate.tolist(),
            hidden_error_r=(estimate - base_r[a:b, d]).tolist(),
            neighboring_count=int(neighbor.sum()),
            neighbor_r_error_RMS=rms,
            neighbor_r_error_max=maximum,
            neighbor_error_lag1_correlation=corr,
            reference_neighbor_r_std=float(np.std(ref[neighbor])),
            outside_replacement_error_RMS=float(
                np.sqrt(np.mean(err[local & ~hidden & np.isfinite(ref)] ** 2))
            ),
            exact_full_support_restored=True,
            xr_relation=relation,
            label="experimental reconstructed r; influenced diagnostic is not independent evidence",
            no_consumer_eligibility_selected=True,
        )
    records.append(record)
summary = dict(
    schema="rtc-r-continuity-diagnostic-v1",
    runtime_policy_changed=False,
    qualification_threshold_selected=False,
    parity=parity,
    cases=records,
    original_files_unchanged=all(
        hashlib.sha256(Path(d["samples"]["path"]).read_bytes()).hexdigest()
        == d["samples"]["sha256"]
        for d in cfg["detectors"]
    ),
)
(E / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(
    [
        (
            r["id"],
            r["additional_paired_original_representatives"],
            {m: round(v["neighbor_r_error_RMS"], 12) for m, v in r["methods"].items()},
        )
        for r in records
    ]
)
