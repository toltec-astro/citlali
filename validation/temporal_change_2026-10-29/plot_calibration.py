"""Render measured synthetic calibration; no scientific thresholds selected."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.results.read_text())
    oracle = json.loads(args.oracle.read_text())
    selected = [c for c in r["cases"] if c["case"] in r["null_cases"] or c["case"] == "long_memory"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(selected))
    for ax, key, interval, nominal in [
            (axes[0], "flagged_at_001", "flag_rate_interval_95", 1),
            (axes[1], "projection_interval_covered", "coverage_interval_95", 99)]:
        values = np.array([100*c[key]/c["available"] for c in selected])
        bounds = 100*np.array([c[interval] for c in selected])
        ax.errorbar(x, values, yerr=np.abs(bounds.T-values), fmt="o", capsize=3,
                    color="#264c73", label="Estimated dependence")
        ax.axhline(nominal, color="#a43d38", linestyle="--", label=f"Nominal {nominal}%")
        ax.set_xticks(x, [c["case"].replace("_", "\n") for c in selected], rotation=35, ha="right")
        ax.grid(axis="y", alpha=.2)
    for c in oracle["cases"]:
        i = next(i for i, v in enumerate(selected) if v["case"] == c["case"])
        axes[1].plot(i, 100*c["covered"]/c["realizations"], "s", color="#35846b")
    axes[1].plot([], [], "s", color="#35846b", label="Known-noise oracle (not deployable)")
    axes[0].set(ylabel="Constant-response flag rate (%)", title="False alarms at the declared 1% test point")
    axes[1].set(ylabel="Simultaneous interval coverage (%)", title="99% intervals are too narrow")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic temporal study — error bars show 95% Monte Carlo intervals")
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
