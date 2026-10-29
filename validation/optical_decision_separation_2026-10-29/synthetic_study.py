"""Independent synthetic experiment; never reads telescope data or runs Citlali.

Exact finite multinomial bootstrap, rather than native finite-draw RNG. This
tests policy mechanisms, not native learner conformance or statistical coverage.
"""
from dataclasses import asdict
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial
from pathlib import Path
import argparse
import json

import numpy as np

from decision_rule import Interval, consider


@lru_cache(None)
def bootstrap_counts(n):
    counts, probabilities = [], []
    for sample in combinations_with_replacement(range(n), n):
        c = np.bincount(sample, minlength=n)
        counts.append(c)
        probabilities.append(factorial(n) / np.prod([factorial(int(v)) for v in c]) / n**n)
    return np.asarray(counts), np.asarray(probabilities)


def solve(x, y, weights):
    total = np.sum(weights)
    mx, my = np.sum(weights * x) / total, np.sum(weights * y) / total
    variance = np.sum(weights * (x - mx)**2)
    if not np.isfinite(variance) or variance <= 32 * np.finfo(float).eps * total:
        return None
    slope = np.sum(weights * (x - mx) * (y - my)) / variance
    return slope, my - slope * mx


def conditional_interval(x, y, blocks):
    """Frozen robust-weight response fit; all contributing blocks retained."""
    unique = np.unique(blocks)
    if len(unique) < 2:
        return Interval(None, None)
    weights = np.ones(len(x))
    fit = solve(x, y, weights)
    if fit is None:
        return Interval(None, None)
    for _ in range(20):
        residual = y - (fit[0] * x + fit[1])
        scale = 1.482602218505602 * np.median(np.abs(residual - np.median(residual)))
        if scale <= 0:
            break
        weights = np.minimum(1., 1.345 * scale / np.maximum(np.abs(residual), np.finfo(float).tiny))
        updated = solve(x, y, weights)
        if updated is None:
            return Interval(None, None)
        done = all(abs(a-b) <= 1e-10 * (1+abs(b)) for a, b in zip(updated, fit))
        fit = updated
        if done:
            break
    # Center before forming moments; no block is discarded for low excitation.
    x, y = x - np.mean(x), y - np.mean(y)
    moments = np.array([[np.sum(weights[m]), np.sum(weights[m]*x[m]),
                         np.sum(weights[m]*y[m]), np.sum(weights[m]*x[m]**2),
                         np.sum(weights[m]*x[m]*y[m])]
                        for b in unique for m in [blocks == b]])
    counts, probabilities = bootstrap_counts(len(unique))
    # Tiny bounded sums: avoid dependence on host BLAS FP status flags.
    # No warnings are suppressed; non-finite results remain an explicit error.
    combined = np.einsum("ij,jk->ik", counts, moments, optimize=False)
    if not np.all(np.isfinite(combined)):
        raise ArithmeticError("non-finite resampled moments")
    w, sx, sy, xx, xy = combined.T
    denominator = w * xx - sx * sx
    if np.any(denominator <= 32*np.finfo(float).eps*np.maximum(w*xx, sx*sx)):
        return Interval(None, None)
    slopes = (w * xy - sx * sy) / denominator
    order = np.argsort(slopes)
    cumulative = np.cumsum(probabilities[order])
    lower, upper = [float(slopes[order[min(np.searchsorted(cumulative, q), len(order)-1)]])
                    for q in (.005, .995)]
    if not np.isfinite(lower + upper) or upper-lower <= 32*np.finfo(float).eps*(1+abs(lower)+abs(upper)):
        return Interval(None, None)
    return Interval(lower, upper)


def fixture(name, seed=42):
    """Declared dimensionless synthetic signal; true array-normal response=1."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(6), 40)
    phase = np.tile(np.linspace(0, 2*np.pi, 40, endpoint=False), 6)
    centers = np.repeat([-3., -2., -1., 1., 2., 3.], 40)
    within = np.full(len(blocks), .7)
    if name == "limited_excitation":
        within[blocks == 5] = .002
    x = centers + within * np.sin(phase)
    gain = np.ones(len(x))
    noise = .02
    if name == "quiet_healthy":
        noise = .0001
    elif name == "noisy_healthy":
        noise = .3
    elif name == "weak_quiet":
        gain[:] = .03
        noise = .0001
    elif name == "reversed":
        gain[:] = -1
    elif name == "positive_gain_change":
        gain[blocks >= 3] = 1.8
    elif name == "late_weak":
        gain[blocks >= 3] = .03
    elif name == "late_reversed":
        gain[blocks >= 3] = -1
    elif name == "no_excitation":
        x[:] = 1
    elif name == "constant_block":
        x[blocks == 5] = 3
    elif name == "ambiguous_full":
        gain[:] = .1
        noise = .3
    elif name not in {"healthy", "limited_excitation"}:
        raise ValueError(name)
    y = 20 + gain * x + noise * rng.normal(size=len(x))
    if name == "limited_excitation":
        # A small additive nuisance can dominate the tiny within-block change.
        # Physical optical gain remains one. This does NOT prove a real block safe.
        y[blocks == 5] -= .015 * np.sin(phase[blocks == 5])
    panels = [x.copy(), x + .0001*np.cos(phase * 3 + .2)]
    if name in {"no_excitation", "constant_block"}:
        panels = [x.copy(), x.copy()]
    return blocks, panels, y


def evaluate(name, seed=42):
    blocks, panels, y = fixture(name, seed)
    selections = [np.ones(len(y), bool), blocks < 3, blocks >= 3]
    checks = tuple(conditional_interval(p[m], y[m], blocks[m])
                   for m in selections for p in panels)
    return {"case": name, "seed": seed, "checks": [asdict(c) for c in checks],
            "decision": asdict(consider(checks)),
            "old_six_interval_admission": all(c.available and c.lower > .1 for c in checks)}


CASES = ("healthy", "quiet_healthy", "noisy_healthy", "weak_quiet", "reversed",
         "limited_excitation", "positive_gain_change", "late_weak", "late_reversed",
         "no_excitation", "constant_block", "ambiguous_full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "optical-decision-synthetic-study-v1", "threshold": .10,
              "confidence": .99, "interval_method": "exact-multinomial-frozen-Huber-reference",
              "coverage_claim": False, "native_conformance_claim": False,
              "cases": [evaluate(name) for name in CASES],
              "seed_sweep": [evaluate(name, seed) for name in CASES for seed in range(16)]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
