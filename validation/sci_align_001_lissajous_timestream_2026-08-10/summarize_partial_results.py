#!/usr/bin/env python3
"""Project checksum-valid observation results into the bounded stop package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table


OPENED = [131920, 131926, 133542, 133544, 135396, 135398, 136278, 136280, 150818]
UNOPENED = [148669, 148671, 150820, 151125, 151127, 151599, 151601, 151949, 151951, 152450, 152452, 152881, 152883]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.strip()
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {path}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_partial_results.py RESULT_ROOT OUTPUT")
    result_root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    identities = []
    stop_pileup_count = None
    for obsnum in OPENED:
        root = result_root / f"o{obsnum}"
        verify(root)
        result_path = root / "result.json"
        doc = json.loads(result_path.read_text())
        bootstrap = doc["bootstrap"]
        timestream = bootstrap["timestream_summary"]
        delta = bootstrap["paired_delta_summary"]
        wins = doc["blocked_prediction"]["winner_counts"]
        accepted = not (
            bool(timestream["multimodal"])
            or bootstrap["timestream_convergence"]["status"] != "pass"
            or bool(doc["model_sensitivity"]["model_sensitive"])
        )
        if obsnum == 136280:
            with np.load(root / "bootstrap_work.npz") as work:
                values = np.asarray(work["timestream"], dtype=float)
            stop_pileup_count = int(np.count_nonzero(
                np.isfinite(values)
                & np.isclose(
                    values, float(doc["primary_tau_ms"]), rtol=0.0, atol=1e-9
                )
            ))
        rows.append({
            "pointing_obsnum": obsnum,
            "beammap_obsnum": int(doc["beammap_obsnum"]),
            "brightness_stratum": doc["brightness_stratum"],
            "gate_status": (
                "accepted_observation_level"
                if accepted else "stopped_persistent_multimodality"
            ),
            "tau_point_ms": float(doc["primary_tau_ms"]),
            "tau_bootstrap_median_ms": float(timestream["median"]),
            "tau_bootstrap_68_low_ms": float(timestream["interval_68"][0]),
            "tau_bootstrap_68_high_ms": float(timestream["interval_68"][1]),
            "tau_bootstrap_95_low_ms": float(timestream["interval_95"][0]),
            "tau_bootstrap_95_high_ms": float(timestream["interval_95"][1]),
            "p_tau_negative": float(timestream["p_negative"]),
            "timestream_bootstrap_count": int(bootstrap["timestream_target_count"]),
            "timestream_kde_peak_count": int(timestream["kde_peak_count"]),
            "timestream_multimodal": bool(timestream["multimodal"]),
            "map_tau_point_ms": float(doc["map_coordinate_shift_tau_ms"]),
            "point_delta_timestream_minus_map_ms": float(doc["point_difference_ms"]),
            "paired_bootstrap_count": int(bootstrap["paired_successful_count"]),
            "paired_delta_median_ms": float(delta["median"]),
            "paired_delta_95_low_ms": float(delta["interval_95"][0]),
            "paired_delta_95_high_ms": float(delta["interval_95"][1]),
            "paired_covariance_ms2": float(bootstrap["timestream_map_covariance_ms2"]),
            "paired_correlation": float(bootstrap["timestream_map_correlation"]),
            "derivative_tau_ms": float(doc["derivative_crosscheck"]["tau_ms"]),
            "model_sensitive": bool(doc["model_sensitivity"]["model_sensitive"]),
            "scan_count": int(doc["support"]["scan_count"]),
            "detector_count": int(doc["support"]["eligible_uid_count"]),
            "common_support_sample_count": int(doc["support"]["common_support_sample_count"]),
            "scored_value_count": int(doc["support"]["scored_value_count"]),
            "blocked_constant_wins": int(wins["constant"]),
            "blocked_lag_wins": int(wins["lag"]),
            "blocked_hysteresis_wins": int(wins["hysteresis"]),
            "blocked_joint_wins": int(wins["joint"]),
            "result_sha256": digest(result_path),
            "result_sha256s_sha256": digest(root / "SHA256SUMS"),
        })
        identities.append({
            "pointing_obsnum": obsnum,
            "ptc_path": doc["input"]["ptc_path"],
            "ptc_sha256": doc["input"]["ptc_sha256"],
            "ppt_path": doc["input"]["ppt_path"],
            "ppt_sha256": doc["input"]["ppt_sha256"],
            "protocol_sha256": doc["input"]["protocol_sha256"],
            "selection_sha256": doc["input"]["selection_sha256"],
            "map_result": doc["input"]["map_result"],
            "result_sha256": digest(result_path),
            "result_sha256s_sha256": digest(root / "SHA256SUMS"),
        })
    Table(rows=rows).write(
        output / "partial_observation_results.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    (output / "partial_observation_results.json").write_text(json.dumps({
        "schema": "sci-align-001-lissajous-timestream-partial-results-v1",
        "opened_count": len(rows),
        "accepted_observation_level_count": sum(
            row["gate_status"] == "accepted_observation_level" for row in rows
        ),
        "stopped_observation": 136280,
        "corpus_inference_permitted": False,
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n")
    (output / "partial_input_identities.json").write_text(json.dumps({
        "schema": "sci-align-001-lissajous-timestream-partial-input-identities-v1",
        "rows": identities,
    }, indent=2, sort_keys=True) + "\n")
    (output / "partial_stop_summary.json").write_text(json.dumps({
        "schema": "sci-align-001-lissajous-timestream-stop-summary-v1",
        "status": "stopped_at_pre_specified_gate",
        "opened_observations": OPENED,
        "unopened_observations": UNOPENED,
        "stop_observation": 136280,
        "stop_condition": "bootstrap remains multimodal at maximum count",
        "timestream_bootstrap_count": 1500,
        "kde_peak_count": 2,
        "exact_optimizer_start_pileup_count": stop_pileup_count,
        "corpus_level_summary_computed": False,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote partial package for {len(rows)} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
