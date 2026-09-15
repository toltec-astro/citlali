#!/usr/bin/env python3
"""Inspect explicit event indices from the simultaneous caller; no classification."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from run_rtc_pipeline_replay import digest


def plot(config_path, learning, indices, output):
    config = json.loads(config_path.read_text())
    receipt = yaml.safe_load((learning / "receipt.yaml").read_text())
    if digest(config_path) != receipt["configuration_sha256"]:
        raise ValueError("configuration differs from exact learning input")
    if digest(learning / "events.yaml") != receipt["events_sha256"]:
        raise ValueError("event evidence changed")
    if digest(learning / "native-time.f64") != receipt["native_time_sha256"]:
        raise ValueError("native time changed")
    events = yaml.safe_load((learning / "events.yaml").read_text())
    times = np.fromfile(learning / "native-time.f64", dtype="<f8")
    inputs = {d["channel"]: d["samples"] for d in config["detectors"]}
    fig, axes = plt.subplots(len(indices), 2, figsize=(13, 3.0 * len(indices)), squeeze=False)
    for axes_row, index in zip(axes, indices, strict=True):
        event = events[index]
        source = inputs[event["channel"]]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("original sample projection changed")
        samples = np.fromfile(source["path"], dtype="<f8").reshape((-1, 4))
        relative = times - event["origin_unix_seconds"]
        selected = np.flatnonzero(np.abs(relative) <= 2.1)
        for c, ax in enumerate(axes_row):
            evidence = event["coordinates"][c]
            valid = samples[selected, c + 2] != 0
            ax.plot(relative[selected][valid], samples[selected, c][valid], color="0.5", lw=0.65, label="original")
            if evidence["background_available"]:
                u = relative[selected] / event["time_scale_seconds"]
                cubic = np.polynomial.polynomial.polyval(u, evidence["cubic"])
                with_offset = np.polynomial.polynomial.polyval(u, evidence["cubic_with_offset"])
                with_offset += evidence["offset"] * (u >= 0)
                ax.plot(relative[selected], cubic, color="#208bb0", label="shared cubic")
                ax.plot(relative[selected], with_offset, color="#d66644", label="cubic + offset")
            a, b = evidence["affected"]
            if 0 <= a < b <= times.size:
                ax.axvspan(relative[a], relative[b - 1] + receipt["cadence_interval_seconds"],
                           color="#e0b64c", alpha=0.25, label="measured affected support")
            a, b = evidence["confirmation"]
            if 0 <= a < b <= times.size:
                ax.axvspan(relative[a], relative[b - 1] + receipt["cadence_interval_seconds"],
                           color="#64b883", alpha=0.2, label="recovery confirmation")
            kind = "jump rule passes" if event["jump_admitted"] else "candidate only"
            ax.set_title(f"Event {index} · n12/ch{event['channel']} · {'xr'[c]} · {kind}", fontsize=10)
            ax.set_xlabel("Native seconds relative to event center")
            ax.set_ylabel("Original coordinate value")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.grid(alpha=0.12)
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("152390 · simultaneous RTC learning · no spatial source mask\n"
                 "Original data and existing fits; no spike acceptance or reconstruction", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output.exists():
        raise ValueError("preserve previous review plot")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learning", type=Path, required=True)
    parser.add_argument("--events", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.config, args.learning, args.events, args.output)
