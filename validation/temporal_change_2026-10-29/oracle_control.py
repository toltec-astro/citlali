"""Known-noise-law control: diagnostic only, not an implementable learned test."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment import fixture, wilson
from temporal import multiply, prepare


def known_factor(blocks, times, rho, sigma):
    """e = L z for exact fixture AR(1), reset at block boundaries/gaps."""
    n = len(blocks)
    factor = np.eye(n)*sigma
    for i in range(1, n):
        if blocks[i] == blocks[i-1] and times[i] == times[i-1]+1:
            factor[i] += rho*factor[i-1]
    return factor


def control(case, rho, sigma):
    t, b, panels, _, _ = fixture(case, 0)
    designs = [prepare(x, t, b) for x in panels]
    h = np.vstack([d.contrast for d in designs])
    transform = multiply(h, known_factor(b, t, rho, sigma))
    sd = np.sqrt(np.sum(transform**2, axis=1))
    rng = np.random.default_rng(923814)
    # Independent calibration draws; fixed before evaluating target realizations.
    maxima = []
    for _ in range(50):
        draws = multiply(transform, rng.normal(size=(len(t), 1000)))
        maxima.extend(np.max(np.abs(draws/sd[:, None]), axis=0))
    q = float(np.quantile(maxima, .99, method="higher"))
    covered = 0
    trials = 1000
    for seed in range(700000, 700000+trials):
        _, _, _, y, mean = fixture(case, seed)
        error = multiply(h, (y-mean)[:, None])[:, 0]
        covered += bool(np.all(np.abs(error) <= q*sd))
    return {"case": case, "known_rho": rho, "known_innovation_sigma": sigma,
            "calibration_draws": 50000, "realizations": trials, "covered": covered,
            "coverage_interval_95": wilson(covered, trials), "critical_max": q,
            "operationally_available": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"purpose": "oracle dependence diagnostic, not a production repair",
              "cases": [control("iid", 0, .1), control("ar1", .6, .1),
                        control("long_memory", .95, .1)]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
