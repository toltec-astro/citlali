#!/usr/bin/env python3
"""Export fixed scientific diagnostic figures; never launch a GUI from Codex."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot(a):
    a.output.mkdir(parents=True, exist_ok=False)
    design = json.loads((a.prepare / "science-design.json").read_text())
    comparison = json.loads((a.analysis / "comparison.json").read_text())["results"]
    arms = ["lowpass", "finite", "coherent", "reject"]
    labels = [
        "LPF / F2",
        "6 s notch + LPF",
        "Coherent + LPF\n(support-ineligible)",
        "Reject eight detectors",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for j, (arm, label) in enumerate(zip(arms, labels)):
        m = np.load(a.analysis / f"{arm}-own-without_sky.npy")
        delta = np.load(a.analysis / f"{arm}-own-paired_response.npy")
        im0 = axes[0, j].imshow(
            m,
            origin="lower",
            extent=[-401, 401, -401, 401],
            vmin=-100,
            vmax=100,
            cmap="RdBu_r",
        )
        im1 = axes[1, j].imshow(
            delta,
            origin="lower",
            extent=[-401, 401, -401, 401],
            norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=80),
            cmap="RdBu_r",
        )
        axes[0, j].set_title(label)
        for ax in axes[:, j]:
            for i, c in enumerate(design["crossings"]):
                x, y = c["center_arcsec"]
                ax.add_patch(plt.Circle((x, y), 20, fill=False, color="black", lw=0.7))
                ax.text(x + 20, y + 8, str(i + 1), fontsize=8)
            ax.set_xlabel("Tangent-plane RA (arcsec)")
            ax.set_ylabel("Dec (arcsec)")
    fig.colorbar(
        im0,
        ax=axes[0, :],
        shrink=0.8,
        label="Original after PCA (matched-APT mJy/beam)",
    )
    fig.colorbar(
        im1, ax=axes[1, :], shrink=0.8, label="Injected − uninjected (mJy/beam)"
    )
    fig.suptitle(
        "152390 / n12: four fixed 10 s chunks, full network, PCA relearned per arm\nConditional ordinary projection; fixed color scales; missing coverage remains blank; production inactive"
    )
    fig.savefig(a.output / "science-comparison.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for j, arm in enumerate(arms):
        r = next(r for r in comparison if r["arm"] == arm and r["support"] == "own")
        x = np.arange(4) + j * 0.16 - 0.24
        axs[0].errorbar(
            x,
            [v["peak_mJy_per_beam"]["mean"] for v in r["compact"]],
            yerr=[v["peak_mJy_per_beam"]["uncertainty_sd"] for v in r["compact"]],
            fmt="o",
            ms=4,
            label=labels[j].replace("\n", " "),
        )
        axs[1].plot(
            np.arange(4) + 1,
            [v["peak_mJy_per_beam"]["MSE"] for v in r["compact"]],
            "o-",
            label=arm,
        )
        axs[2].errorbar(
            j, r["extended"]["mean"], yerr=r["extended"]["uncertainty_sd"], fmt="o"
        )
    axs[0].axhline(100, color="k", ls="--", lw=1)
    axs[0].set_xticks(np.arange(4), ["1", "2", "3", "4 boundary"])
    axs[0].set_ylabel("Recovered compact peak (mJy/beam)")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("Fixed source crossing")
    axs[1].set_ylabel("Peak MSE ((mJy/beam)²)")
    axs[1].set_xticks([1, 2, 3, 4])
    axs[2].axhline(25, color="k", ls="--", lw=1)
    axs[2].set_xticks(range(4), ["LPF", "Notch", "Coherent", "Reject"], rotation=20)
    axs[2].set_ylabel("Extended peak (mJy/beam)")
    fig.suptitle(
        "Eight independent phase-randomized background realizations with cross-detector spectra preserved\nError bars: sample standard deviation; own retained support; known added line; no common score across units"
    )
    fig.savefig(a.output / "science-error.png", dpi=170)
    plt.close(fig)
    residual = json.loads((a.residuals / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for j, (arm, label) in enumerate(zip(arms[:3], labels[:3])):
        r = [
            next(
                r
                for r in residual
                if r["coordinate"] == "x" and r["detector"] == d and r["arm"] == arm
            )
            for d in [269, 402]
        ]
        ax.bar(
            np.arange(2) + j * 0.24 - 0.24,
            [r0["rtc"]["mean_band_power"] for r0 in r],
            width=0.22,
            alpha=0.3,
            color=f"C{j}",
        )
        ax.bar(
            np.arange(2) + j * 0.24 - 0.24,
            [r0["pca"]["mean_band_power"] for r0 in r],
            width=0.22,
            color=f"C{j}",
            label=label.replace("\n", " "),
        )
    ax.set_xticks([0, 1], ["n12 / 269", "n12 / 402"])
    ax.set_ylabel("9.75–12.25 Hz power ((mJy/beam)²)")
    ax.legend(fontsize=8)
    ax.set_title(
        "Six identical complete 4 s windows per detector\nPale: after RTC; solid: after independently relearned 10-mode PCA"
    )
    fig.savefig(a.output / "downstream-line-power.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["prepare", "analysis", "residuals", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    plot(p.parse_args())
