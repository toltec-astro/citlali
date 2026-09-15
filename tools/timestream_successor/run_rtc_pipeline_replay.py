#!/usr/bin/env python3
"""Replay the fixed 152390 population through RTC assembly; no new selections.

Uses sealed wider-notch inputs and coefficients. LPF -> selected notch -> LPF
is an explicit test schedule, not a convergence or classification policy.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import yaml

AFFECTED = (269, 402, 296, 300, 366, 438, 406, 307)
CONTROLS = (193, 231, 226, 330)
BASELINE_MANIFEST = "cb09214f637e65fe91b109e911c1d48422440daea2de09da9b19c59d3a4b1f1d"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_config(config, channel):
    if channel not in AFFECTED + CONTROLS or config["channel"] != channel:
        raise ValueError("channel is outside the fixed population")
    if config["observation"] != 152390 or config["network"] != 12:
        raise ValueError("replay requires the bound original observation/network")
    out = copy.deepcopy(config)
    trials = {t["id"]: t for t in out["trials"]}
    selected = [trials["lowpass"]]
    if channel in AFFECTED:
        notch = trials["w1-t3"]
        if notch["requested_width_hz"] != 1 or notch["requested_span_seconds"] != 3:
            raise ValueError("selected baseline changed")
        selected.append(notch)
    reset = copy.deepcopy(trials["lowpass"])
    reset["id"] = "lowpass-replay"
    out["trials"] = selected + [reset]
    out["rtc_assembly"] = True
    return out


def run(args):
    prior = args.prior.resolve(strict=True)
    manifest = prior / "EVIDENCE_SHA256SUMS"
    if digest(manifest) != BASELINE_MANIFEST:
        raise ValueError("prior experiment manifest differs")
    sealed = {line[66:]: line[:64] for line in manifest.read_text().splitlines()}
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    for channel in args.channels:
        source_config = prior / "prepared-01" / f"{channel}.json"
        if digest(source_config) != sealed[str(source_config.relative_to(prior))]:
            raise ValueError("prior prepared configuration changed")
        config = prepare_config(json.loads(source_config.read_text()), channel)
        cfg_path = args.output / f"{channel}.json"
        cfg_path.write_text(json.dumps(config, indent=2) + "\n")
        output = args.output / str(channel)
        began = time.monotonic()
        with (args.output / f"{channel}.log").open("w") as log:
            subprocess.run([str(binary), str(cfg_path), str(output)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        receipt = json.loads((output / "receipt.json").read_text())
        if not receipt["original_pair_unchanged"] or receipt["production_filtering_active"]:
            raise ValueError("original/activation invariant failed")
        comparisons = []
        timings = []
        for trial in config["trials"]:
            stem = trial["id"]
            reference = "lowpass" if stem == "lowpass-replay" else stem
            reference_dir = prior / "campaign-01" / str(channel)
            # Known source transfer is retained exactly, including missing
            # support, rather than choosing a new tolerance or science budget.
            suffixes = ["native.f64", "conditioned.f64", "causes.u8", "rows.i64"]
            suffixes += [inj["identity"] + ".f64" for inj in config["injections"]]
            for suffix in suffixes:
                old = reference_dir / f"{reference}-{suffix}"
                new = output / f"{stem}-{suffix}"
                if digest(old) != sealed[str(old.relative_to(prior))]:
                    raise ValueError(f"old reference changed: {old}")
                if digest(new) != digest(old):
                    raise ValueError(f"replay differs from exact saved reference: {new}")
                comparisons.append(new.name)
            stages = yaml.safe_load((output / f"{stem}-assembly.yaml").read_text())
            for stage in ("post-notch", "post-lowpass"):
                data = stages[stage]
                if digest(output / data["numerical_parent"]) != data["numerical_parent_sha256"]:
                    raise ValueError("conditioned evidence numerical parent changed")
            timings.append({"trial": stem, "apply_seconds": stages["complete_Apply_seconds"],
                            "reconsider_learn_seconds": stages["reconsider_Learn_seconds"],
                            "consider_seconds": stages["complete_Consider_seconds"],
                            "two_stage_learn_and_export_seconds": stages["post_apply_Learn_seconds"],
                            "two_stage_native_product_seconds": sum(stages[s]["native_product_seconds"] for s in ("post-notch", "post-lowpass")),
                            "two_stage_spectral_learn_seconds": sum(stages[s]["spectral_Learn_seconds"] for s in ("post-notch", "post-lowpass")),
                            "post_notch_windows": [c["windows"] for c in stages["post-notch"]["coordinates"]],
                            "post_lowpass_windows": [c["windows"] for c in stages["post-lowpass"]["coordinates"]]})
        records.append({"channel": channel, "source_revision": receipt["source_revision"],
                        "wall_seconds": time.monotonic()-began,
                        "byte_identical_reference_files": comparisons, "timings": timings})
        (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"channel {channel}: {len(comparisons)} reference artifacts unchanged", flush=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"disposition": "PASS", "records": records, "binary": str(binary),
              "binary_sha256": digest(binary), "prior_manifest_sha256": digest(manifest),
              "wall_seconds": time.monotonic()-start,
              "child_cpu_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
              "cumulative_peak_child_rss_native_units": after.ru_maxrss,
              "rss_units": "bytes on macOS; KiB on Linux",
              "scope": "fixed single-detector replays; not full-network donor admission or production throughput",
              "new_automatic_decisions": False, "FRUIT": False, "production": False}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=list(AFFECTED + CONTROLS))
    run(parser.parse_args())
