#!/usr/bin/env python3
"""Export one legacy Citlali TOD file for explicitly discovery-only D2 use.

Legacy RTC/PTC NetCDF files use a rectangular telescope-time container and may
already have been filtered or downsampled.  This adapter therefore cannot
create conforming network-native D2 evidence.  Its manifest always declares
``legacy_rectangular_discovery`` and a legacy stream stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from tools.wp7.rtc_filter_psd_line_evidence import (
    DISCOVERY_TIMING_DOMAIN,
    ESTABLISHED_LINE_STRATEGY,
    INPUT_SCHEMA,
    sha256_file,
)


ARRAY_NAMES = {0: "a1100", 1: "a1400", 2: "a2000"}


def _scalar_string(variable: Any) -> str:
    value = np.asarray(variable[:]).reshape(-1)
    if value.size != 1:
        raise RuntimeError("legacy TOD type is not scalar")
    return str(value[0])


def _write_array(output_dir: Path, name: str, value: np.ndarray) -> str:
    path = output_dir / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return path.name


def export_legacy_tod(
    input_path: Path,
    output_dir: Path,
    *,
    case_id: str,
    route_family: str,
    network: int,
) -> Path:
    input_path = input_path.resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    if route_family not in {"beammap", "science", "oof", "pointing"}:
        raise RuntimeError("route family is not supported")
    with netCDF4.Dataset(input_path) as dataset:
        required = {
            "tod_output_type",
            "obsnum",
            "scan_indices",
            "output_scan_index",
            "signal",
            "flags",
            "apt_array",
            "apt_nw",
            "apt_uid",
            "TelUTC",
        }
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise RuntimeError(f"legacy TOD variables are missing: {', '.join(missing)}")
        output_type = _scalar_string(dataset.variables["tod_output_type"])
        if output_type not in {"rtc", "ptc"}:
            raise RuntimeError("legacy TOD type is neither rtc nor ptc")
        stage = f"legacy_{output_type}_output"
        networks = np.asarray(dataset.variables["apt_nw"][:], dtype=np.int64)
        detector_selection = np.where(networks == network)[0]
        if detector_selection.size == 0:
            raise RuntimeError(f"network {network} has no detectors")
        arrays = np.asarray(dataset.variables["apt_array"][:], dtype=np.int64)[
            detector_selection
        ]
        if np.unique(arrays).size != 1 or int(arrays[0]) not in ARRAY_NAMES:
            raise RuntimeError("network does not have one recognized array identity")
        detector_id = np.asarray(
            dataset.variables["apt_uid"][:], dtype=np.int64
        )[detector_selection]
        scan_indices = np.asarray(dataset.variables["scan_indices"][:], dtype=np.int64)
        output_scan_index = np.asarray(
            dataset.variables["output_scan_index"][:], dtype=np.int64
        )

        rows: list[np.ndarray] = []
        run_ids: list[np.ndarray] = []
        for scan, (first, last) in enumerate(scan_indices):
            if first < 0 or last < first:
                raise RuntimeError(f"legacy scan index {scan} is invalid")
            current = np.arange(int(first), int(last) + 1, dtype=np.int64)
            rows.append(current)
            run_ids.append(np.full(current.shape, int(output_scan_index[scan]), dtype=np.int64))
        selected_rows = np.concatenate(rows)
        physical_run_id = np.concatenate(run_ids)
        signal = np.asarray(
            dataset.variables["signal"][selected_rows, detector_selection],
            dtype=np.float64,
        )
        flags = np.asarray(
            dataset.variables["flags"][selected_rows, detector_selection],
            dtype=np.int8,
        )
        time_sec = np.asarray(
            dataset.variables["TelUTC"][selected_rows], dtype=np.float64
        )
        observation = int(np.asarray(dataset.variables["obsnum"][:]))
        producer_configuration = {
            name: np.asarray(dataset.variables[name][:]).reshape(-1).tolist()
            for name in (
                "CONFIG.TODFILTERED",
                "CONFIG.DOWNSAMPLED",
                "CONFIG.RTC.LINE_AUDIT.ENABLED",
                "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED",
                "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED",
                "SAMPRATE",
            )
            if name in dataset.variables
        }

    same_run = physical_run_id[1:] == physical_run_id[:-1]
    intervals = np.diff(time_sec)[same_run]
    nominal_interval = float(np.median(intervals))
    maximum_deviation = float(
        np.max(np.abs(intervals - nominal_interval)) / nominal_interval
    )
    output_dir.mkdir(parents=True)
    declarations = {
        "occurrence_id": _write_array(output_dir, "occurrence_id", selected_rows),
        "time_sec": _write_array(output_dir, "time_sec", time_sec),
        "physical_run_id": _write_array(
            output_dir, "physical_run_id", physical_run_id
        ),
        "detector_id": _write_array(output_dir, "detector_id", detector_id),
        "signal": _write_array(output_dir, "signal", signal),
        "valid": _write_array(output_dir, "valid", flags == 0),
        "source_excluded": _write_array(
            output_dir,
            "source_excluded",
            np.zeros(time_sec.shape, dtype=bool),
        ),
    }
    manifest = {
        "schema": INPUT_SCHEMA,
        "identity": {
            "case_id": case_id,
            "route_family": route_family,
            "observation": observation,
            "subobservation": None,
            "scan": "all-published-output-scans",
            "network": network,
            "array": ARRAY_NAMES[int(arrays[0])],
            "stream_stage": stage,
            "timing_domain": DISCOVERY_TIMING_DOMAIN,
            "signal_units": "legacy_tod_native_units",
            "cadence_domain_id": (
                f"legacy-observed-{1.0 / nominal_interval:.9f}hz-discovery"
            ),
        },
        "cadence_domain": {
            "nominal_interval_sec": nominal_interval,
            "maximum_fractional_deviation": maximum_deviation + 1.0e-12,
            "authority": "observed_legacy_container_not_D1_native_census",
        },
        "source_mask": {
            "policy_id": "legacy-discovery-no-source-mask",
            "status": "absent_discovery",
            "meaning": "all false; not suitable for a D2 envelope candidate",
        },
        "line_mask": {
            "policy_id": "legacy-discovery-no-line-mask",
            "strategy_id": ESTABLISHED_LINE_STRATEGY,
            "status": "pending",
            "intervals_hz": [],
        },
        "producer": {
            "adapter": "export_legacy_tod_psd_line_discovery.py",
            "source_filename": input_path.name,
            "source_sha256": sha256_file(input_path),
            "source_tod_output_type": output_type,
            "source_configuration": producer_configuration,
            "scientific_limitation": (
                "rectangular legacy TelUTC container; network-native timing and "
                "native-rate prefilter/post-cleaning stage are not established"
            ),
        },
        "arrays": declarations,
    }
    manifest_path = output_dir / "input.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--route-family",
        choices=("beammap", "science", "oof", "pointing"),
        required=True,
    )
    parser.add_argument("--network", type=int, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest = export_legacy_tod(
        args.input,
        args.output_dir.resolve(),
        case_id=args.case_id,
        route_family=args.route_family,
        network=args.network,
    )
    print(manifest)


if __name__ == "__main__":
    main()
