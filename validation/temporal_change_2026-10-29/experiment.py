"""Declared synthetic Monte Carlo. Telescope data and Citlali are never used."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from temporal import assess, multiply, prepare


NULL_CASES = ("iid", "ar1", "heteroskedastic_ar1", "offsets", "limited_excitation",
              "gaps", "heavy_tails", "quiet")
STRESS_CASES = ("long_memory", "reference_drift", "reference_noise", "nonlinear",
                "outlier", "no_excitation", "constant_last_block")
ALTERNATIVES = ("step_005", "step_010", "step_030", "step_080", "drift_010",
                "drift_030", "temporary_030", "late_weak", "late_reversed")


def fixture(case, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(240)
    blocks = t//40
    phase = 2*np.pi*(t % 40)/40
    x = np.repeat([-3., -2., -1., 1., 2., 3.], 40) + .7*np.sin(phase)
    if case == "limited_excitation":
        x[blocks == 5] = 3 + .002*np.sin(phase[blocks == 5])
    if case == "constant_last_block":
        x[blocks == 5] = 3
    if case == "no_excitation":
        x[:] = 1
    gain = np.ones(240)
    if case.startswith("step_"):
        gain[blocks >= 3] += int(case.split("_")[1])/100
    if case.startswith("drift_"):
        gain += int(case.split("_")[1])/100*t/239
    if case == "temporary_030":
        gain[blocks == 3] += .3
    if case == "late_weak":
        gain[blocks >= 3] = .03
    if case == "late_reversed":
        gain[blocks >= 3] = -1
    mean = 20 + gain*x
    if case == "offsets":
        mean += np.array([0, 1, -2, 3, -1, 2])[blocks]
    if case == "nonlinear":
        mean += .05*x*x  # Stationary nonlinear coupling, not time-varying gain.
    scale = .1
    if case in {"iid", "quiet"}:
        rho = 0.
    elif case == "long_memory":
        rho = .95
    else:
        rho = .6
    if case == "quiet":
        scale = .0001
    innovations = rng.normal(size=240)*scale
    if case == "heavy_tails":
        innovations = rng.standard_t(3, size=240)*scale/np.sqrt(3)
    if case == "heteroskedastic_ar1":
        innovations *= np.where(blocks < 3, .5, 2.)
    keep = np.ones(240, bool)
    if case == "gaps":
        keep[[18, 19, 61, 62, 135, 136, 200]] = False
    noise = innovations.copy()
    for i in range(1, 240):
        if blocks[i] == blocks[i-1] and keep[i] and keep[i-1]:
            noise[i] += rho*noise[i-1]
    if case == "outlier":
        noise[179] += 10
    y = mean+noise
    panel_b = x + .0001*np.cos(3*phase+.2)
    if case == "reference_drift":
        panel_b = x*(1 + .2*t/239)
    if case == "reference_noise":
        panel_b = x + .25*rng.normal(size=240)
    return t[keep], blocks[keep], [x[keep], panel_b[keep]], y[keep], mean[keep]


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def run_case(case, repetitions, draws):
    counts = {"available": 0, "flagged_at_001": 0, "projection_interval_covered": 0}
    p_values, midpoint_width, fitted_rho = [], [], []
    failures = {}
    # Stable, named, independent random stream between different scenarios.
    offset = int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)
    for i in range(repetitions):
        t, b, panels, y, mean = fixture(case, offset+i)
        designs = [prepare(x, t, b) for x in panels]
        result = assess(designs, y, draws=draws, seed=offset+100000+i)
        if not result["available"]:
            why = result["reason"]
            failures[why] = failures.get(why, 0)+1
            continue
        counts["available"] += 1
        counts["flagged_at_001"] += result["p_value"] <= .01
        truth = np.concatenate([multiply(d.contrast, mean[:, None])[:, 0]
                                for d in designs])
        counts["projection_interval_covered"] += bool(
            np.all(truth >= result["lower"]) and np.all(truth <= result["upper"]))
        p_values.append(result["p_value"])
        midpoint_width.append(result["upper"][2]-result["lower"][2])
        fitted_rho.append(result["rho_full"][0])
    n = counts["available"]
    return {"case": case, "repetitions": repetitions, **counts, "unavailable": failures,
            "flag_rate_interval_95": wilson(counts["flagged_at_001"], n),
            "coverage_interval_95": wilson(counts["projection_interval_covered"], n),
            "p_quantiles": np.quantile(p_values, [.01, .5, .99]).tolist() if n else None,
            "median_midpoint_interval_width": float(np.median(midpoint_width)) if n else None,
            "median_fitted_rho": float(np.median(fitted_rho)) if n else None}


def old_counterexamples(draws):
    # Import only the earlier synthetic fixture, never application code.
    import importlib.util
    import sys
    previous = Path(__file__).resolve().parent.parent/"optical_decision_separation_2026-10-29"
    sys.path.insert(0, str(previous))
    spec = importlib.util.spec_from_file_location("previous_synthetic", previous/"synthetic_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for seed in (7, 15):
        b, panels, y = module.fixture("healthy", seed)
        d = [prepare(x, np.arange(len(y)), b) for x in panels]
        records.append({"old_seed": seed, "result": assess(d, y, draws=draws, seed=seed+40000)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-repetitions", type=int, default=1000)
    parser.add_argument("--other-repetitions", type=int, default=250)
    parser.add_argument("--draws", type=int, default=999)
    args = parser.parse_args()
    start = time.monotonic()
    cases = []
    for case in NULL_CASES + STRESS_CASES + ALTERNATIVES:
        n = args.null_repetitions if case in NULL_CASES else args.other_repetitions
        record = run_case(case, n, args.draws)
        cases.append(record)
        print(case, record["flagged_at_001"], "/", record["available"], flush=True)
    result = {"method": "experimental-fixed-reference-ar1-wild-max-v1",
              "optical_cut": .1, "harm_threshold": None, "draws": args.draws,
              "null_cases": NULL_CASES, "stress_cases": STRESS_CASES,
              "alternative_cases": ALTERNATIVES, "cases": cases,
              "previous_counterexamples": old_counterexamples(args.draws),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform()},
              "elapsed_seconds": time.monotonic()-start}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
