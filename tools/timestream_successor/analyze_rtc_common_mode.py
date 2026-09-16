#!/usr/bin/env python3
"""Report a bounded native common-mode diagnostic; never emits science flags."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_yaml(path):
    with path.open() as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    health = args.replay / "health"
    receipt = read_yaml(health / "receipt.yaml")
    parent = read_yaml(args.replay / "receipt.yaml")
    table = pd.read_csv(health / "detectors.csv")
    fits = pd.read_csv(health / "fits.csv")
    intervals = read_yaml(health / "intervals.yaml")
    reference = np.fromfile(health / "reference.f64", dtype="<f8").reshape(-1, 3)
    time = reference[:, 0] - reference[0, 0]
    dt = parent.get("native_integration_seconds", .008192)
    paired = np.array([s["original_paired_rows"] for s in intervals])
    eligible = np.array([s["eligible_target_rows"] for s in intervals])
    reasons = np.array([s["reference_reasons"] for s in intervals])
    contributors = np.array([s["contributors"] for s in intervals])
    table["original_paired_seconds"] = paired.sum(axis=0) * dt
    table["adaptive_eligible_seconds"] = eligible.sum(axis=0) * dt
    table["reference_intervals"] = (reasons == 0).sum(axis=0)
    table["fit_fraction_of_adaptive_support"] = table.fitted_rows / np.maximum(eligible.sum(axis=0), 1)
    selected = receipt["self_excluded_channels"]
    checks = {}
    for channel in selected:
        check = pd.read_csv(health / f"self-excluded-{channel}.csv")
        good = check[check.cause == 0]
        checks[channel] = {
            "available_segments": len(good),
            "gain_median": float(good.gain.median()),
            "gain_segment_mad": float(1.4826 * (good.gain - good.gain.median()).abs().median()),
            "correlation_median": float(good.correlation.median()),
            "residual_scatter_median": float(good.residual_scatter.median()),
            "fitted_rows": int((good.past_last - good["first"]).sum()),
        }
    table.to_csv(args.output / "detector-census.csv", index=False)
    ordinary_denominator = float(table.loc[table.reference_population == 1, "adaptive_eligible_seconds"].sum())
    compact = table.set_index("channel").loc[selected].copy()
    compact["loo_gain"] = [checks[c]["gain_median"] for c in selected]
    compact["loo_correlation"] = [checks[c]["correlation_median"] for c in selected]
    compact["prospective_eligible_seconds"] = np.where(compact.reference_population == 1, compact.adaptive_eligible_seconds, 0.)
    compact["prospective_population_percent"] = compact.prospective_eligible_seconds / ordinary_denominator * 100
    compact.to_csv(args.output / "inspection-table.csv")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    axes[0].plot(time, reference[:, 1], linewidth=.5)
    axes[0].set(ylabel="Common-mode candidate (native x)", title=f"152390 / network {parent.get('network', 12)} / all {len(table)} detectors — original SCIENCE")
    centers = [time[min(len(time)-1, (s["rows"][0]+s["rows"][1])//2)] for s in intervals]
    axes[1].step(centers, contributors, where="mid")
    axes[1].set(ylabel="Fixed reference contributors")
    axes[2].scatter(table.relative_gain, table.correlation, s=10, c=table.fit_fraction_of_adaptive_support, cmap="viridis", vmin=0, vmax=1)
    for c in selected:
        a = compact.loc[c]
        if np.isfinite(a.relative_gain) and np.isfinite(a.correlation):
            axes[2].annotate(str(c), (a.relative_gain, a.correlation), fontsize=8)
    axes[2].set(xlabel="Median calibrated relative gain (signed)", ylabel="Median raw correlation")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(args.output / "network-overview.png", dpi=150)
    plt.close(fig)

    # Eight preselected inspections: no new data-dependent reference pruning.
    for page, group in enumerate((selected[:4], selected[4:])):
        if not group:
            continue
        fig, axes = plt.subplots(len(group), 2, figsize=(13, 2.6*len(group)), squeeze=False, layout="constrained")
        for row, channel in enumerate(group):
            data = np.fromfile(health / f"diagnostic-{channel}.f64", dtype="<f8").reshape(-1, 4)
            check = pd.read_csv(health / f"self-excluded-{channel}.csv")
            good = check[check.cause == 0]
            # Choose the middle available processing interval, then its longest
            # contiguous fit. This is not selected for largest anomaly score.
            if good.empty:
                continue
            middle = sorted(good.interval.unique())[len(good.interval.unique())//2]
            candidates = good[good.interval == middle]
            chosen = candidates.loc[(candidates.past_last-candidates["first"]).idxmax()]
            lo, hi = int(chosen["first"]), int(chosen.past_last)
            axes[row, 0].plot(time[lo:hi], data[lo:hi, 0], color=".45", linewidth=.7, label="original x")
            axes[row, 0].plot(time[lo:hi], data[lo:hi, 1], color="tab:orange", linewidth=1.2, label="signed affine response")
            axes[row, 1].plot(time[lo:hi], data[lo:hi, 2], color="tab:blue", linewidth=.7)
            axes[row, 0].set_title(f"Channel {channel}; gain {chosen.gain:.3g}; correlation {chosen.correlation:.3f}")
            axes[row, 1].set_title(f"Residual; robust scatter {chosen.residual_scatter:.3g}")
            axes[row, 0].set_ylabel("Native x units")
            axes[row, 0].legend(fontsize=8, loc="best")
            for ax in axes[row]:
                ax.grid(alpha=.2)
                ax.set_xlabel("Native seconds from observation start")
        fig.suptitle("Self-excluded references; no rescaling to conceal signed response; no science correction")
        fig.savefig(args.output / f"inspections-{page+1}.png", dpi=150)
        plt.close(fig)

    result = {
        "source_revision": receipt["source_revision"],
        "detectors": len(table), "initial_population": int(table.reference_population.sum()),
        "targets_with_fits": int((table.available_segments > 0).sum()),
        "processing_intervals": len(intervals),
        "contributors_min_median_max": [int(contributors.min()), float(np.median(contributors)), int(contributors.max())],
        "native_rows": len(reference), "speed_eligible_rows": int(reference[:, 2].sum()),
        "available_reference_rows": int(np.isfinite(reference[:, 1]).sum()),
        "fit_causes": {str(k): int(v) for k, v in fits.cause.value_counts().items()},
        "initial_population_adaptive_detector_seconds": ordinary_denominator,
        "negative_relative_median_channels": table.loc[table.relative_gain < 0, "channel"].tolist(),
        "self_excluded": checks,
        "runtime": {k: v for k, v in receipt.items() if k.endswith("seconds") or k.endswith("bytes")},
        "Apply_performed": parent["Apply_performed"], "original_pair_unchanged": parent["original_pair_unchanged"],
        "prospective_cost_scope": "original paired adaptive-eligible time, within APT/Tune reference population; before detailed event treatment; no exclusion applied",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
