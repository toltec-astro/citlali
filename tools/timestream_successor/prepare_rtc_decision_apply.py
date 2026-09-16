#!/usr/bin/env python3
"""Bind the existing processing generation; never select event IDs or supports."""

import argparse
import json
from pathlib import Path
import yaml
from run_rtc_pipeline_replay import digest


def prepare(prior, raw_root, provenance, output):
    seal = prior / "EVIDENCE_SHA256SUMS"
    if (
        digest(seal)
        != "f1de4f0095cd3344f1de75ccab0e08a2b9d8658acb659e320ed60246f7785397"
    ):
        raise ValueError("unexpected predecessor")
    sealed = {line[66:]: line[:64] for line in seal.read_text().splitlines()}
    if digest(prior / "input.json") != sealed["input.json"]:
        raise ValueError("previous input changed")
    config = json.loads((prior / "input.json").read_text())
    effective = config["effective_config"]
    if digest(effective["path"]) != effective["sha256"]:
        raise ValueError("effective configuration changed")
    cfg = yaml.safe_load(Path(effective["path"]).read_text())
    observation = cfg["inputs"][0]
    timing = []
    for entry in observation["data_items"]:
        interface = entry["meta"]["interface"]
        if not interface.startswith("toltec"):
            continue
        path = raw_root / Path(entry["filepath"]).name
        timing.append(
            dict(network=int(interface[6:]), path=str(path), sha256=digest(path))
        )
    if sorted(t["network"] for t in timing) != [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12]:
        raise ValueError(
            "the existing processing generation has eleven exact network inputs"
        )
    config["decision_apply"] = dict(
        policy="existing-authorities-with-unavailable-isolated-admission-v1",
        timing_inputs=timing,
        processing_provenance=dict(path=str(provenance), sha256=digest(provenance)),
    )
    with output.open("x") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.prior, a.raw_root, a.provenance, a.output)
