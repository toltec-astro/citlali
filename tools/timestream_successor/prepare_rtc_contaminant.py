#!/usr/bin/env python3
"""Choose one usable real interval reproducibly; declare test contamination only."""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from run_rtc_pipeline_replay import digest


def prepare(config_path, reference, output):
    config = json.loads(config_path.read_text())
    receipt_path = reference / "receipt.yaml"
    receipt = yaml.load(receipt_path.read_text(), Loader=yaml.CSafeLoader)
    if not receipt["Apply_performed"] or not receipt["original_pair_unchanged"]:
        raise ValueError("an untouched applied reference is required")
    if receipt["configuration_sha256"] != digest(config_path):
        raise ValueError("reference configuration differs")
    if "declared_contaminant" in config:
        raise ValueError("a contaminant cannot be added cumulatively")
    rows = receipt["rows"]
    dt = receipt["cadence_interval_seconds"]
    radius = int(np.ceil(5 / dt))
    usable = np.ones(rows, dtype=bool)
    for detector in config["detectors"]:
        channel = detector["channel"]
        state = np.fromfile(
            reference / "exclusion-control" / f"{channel}-state.u8", dtype="u1"
        )
        if len(state) != rows:
            raise ValueError("reference state cardinality mismatch")
        usable &= (state & 3) == 3
        usable &= (state & 60) == 0
    # All twelve peers have full paired filter support in this bounded window.
    good = (
        np.convolve(usable.astype(int), np.ones(2 * radius + 1, dtype=int), "same")
        == 2 * radius + 1
    )
    choices = np.flatnonzero(good)
    if not len(choices):
        raise ValueError("no originally usable interval supports the declared test")
    row = int(choices[0])
    # Exercise the already-admitted notch baseline; no new filter selection.
    detector = next(d for d in config["detectors"] if d["filter"] == "w1-t3")
    samples = detector["samples"]
    if digest(samples["path"]) != samples["sha256"]:
        raise ValueError("original sample identity changed")
    data = np.fromfile(samples["path"], dtype="<f8").reshape(rows, 4)
    residual = np.diff(data[row - radius : row + radius + 1, :2], axis=0)
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("test amplitude unavailable")
    config["declared_contaminant"] = dict(
        channel=detector["channel"],
        native_row=row,
        x_delta=float(30 * scale[0]),
        r_delta=float(30 * scale[1]),
        reference_configuration=dict(path=str(config_path), sha256=digest(config_path)),
        reference_receipt=dict(path=str(receipt_path), sha256=digest(receipt_path)),
        placement_rule="earliest center with five seconds of paired output support on each side for all 12 cohort detectors",
        model="one native occurrence: additive 30 local difference-MAD scales independently in x and r; no peers changed; test truth only",
    )
    with output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.config, a.reference, a.output)
