#!/usr/bin/env python3
"""Audit the bounded SCI-ALIGN acquisition event-semantics evidence.

This diagnostic is deliberately non-corrective.  It verifies the frozen
three-map evidence, inspects already-local Beammap 148670 detector files, and
tests whether whole-row integer reassociation can reproduce the accepted
nearest-half-cadence labels.  It cannot identify the physical integration
event without producer-side FPGA authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import netCDF4
import numpy as np


EXPECTED_HEAD = "92cfa670a33255250895d68aaf26e8b01aa057bd"
EXPECTED_PARENT = "77c8a1a71cc79eb3aeacbd596c42b6dae33b3aa4"
EXPECTED_TREE = "908825af674e3ea19c03cbb54441680dd4d6ad12"
EXPECTED_GROUP = "roach-t0:44cf69da97d473965ef6"
EXPECTED_MAPS = {
    148670: "map:912d0ccf8b3539501f6c",
    150819: "map:1971b4dfddbc99932afb",
    151126: "map:d5fec4dcd0f16b424fb6",
}
EXPECTED_NETWORKS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12)
EXPECTED_HALF_STATES = {148670: 0, 150819: -3, 151126: -2}
EXPECTED_TS_LONG_NAME = (
    "ClockTime (sec), PpsCount (pps ticks), ClockCount (clock ticks), "
    "PacketCount (packet ticks), PpsTime (clock ticks), "
    "ClockTimeNanoSec (nsec)"
)
CADENCE_SEC = 0.008192
HALF_CADENCE_SEC = CADENCE_SEC / 2.0


class AuditError(RuntimeError):
    """Raised when an audit identity or integrity gate fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def u32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(np.uint32).astype(np.uint64)


def modular_difference(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    modulus = np.uint64(2**32)
    return (after + modulus - before) % modulus


def reconstruct_legacy_timestamp(ts: np.ndarray, fpga_hz: float) -> np.ndarray:
    fields = np.asarray(ts, dtype=np.float64)
    require(fields.ndim == 2 and fields.shape[1] == 6, "Ts must have [row,6] shape")
    anchor = int(fields[0, 0] + fields[0, 5] * 1e-9 - 0.5)
    delta = fields[:, 2] - fields[:, 4]
    delta[fields[:, 2] < fields[:, 4]] += 4294967295.0
    result = anchor + fields[:, 1] + delta / fpga_hz
    require(bool(np.all(np.isfinite(result))), "reconstructed timestamp is nonfinite")
    require(bool(np.all(np.diff(result) > 0)), "reconstructed timestamp is not increasing")
    return result


def labels_are_integer_row_compatible(
    pair_labels: Mapping[tuple[int, int], int],
) -> bool:
    """Whole-row shifts are even in half-cadence label units."""

    return all(int(label) % 2 == 0 for label in pair_labels.values())


def transitive_states(
    pair_labels: Mapping[tuple[int, int], int], reference: int
) -> dict[int, int]:
    states = {reference: 0}
    remaining = dict(pair_labels)
    changed = True
    while changed:
        changed = False
        for (left, right), delta in list(remaining.items()):
            if left in states and right not in states:
                states[right] = states[left] + int(delta)
                changed = True
            elif right in states and left not in states:
                states[left] = states[right] - int(delta)
                changed = True
    for (left, right), delta in pair_labels.items():
        require(left in states and right in states, "pair graph is disconnected")
        require(states[right] - states[left] == int(delta), "pair labels are not transitive")
    return states


def load_phase0_raw_manifest(path: Path) -> list[dict[str, str]]:
    selected = [
        row
        for row in read_csv(path)
        if row["mode"] == "beammap" and int(row["obsnum"]) == 148670
    ]
    require(len(selected) == len(EXPECTED_NETWORKS), "148670 raw manifest is incomplete")
    actual = tuple(sorted(int(row["interface_id"].removeprefix("toltec")) for row in selected))
    require(actual == EXPECTED_NETWORKS, "148670 raw manifest network set changed")
    return sorted(selected, key=lambda row: int(row["interface_id"].removeprefix("toltec")))


def inspect_raw_148670(
    manifest_rows: Sequence[Mapping[str, str]], verify_full_sha256: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        network = int(manifest["interface_id"].removeprefix("toltec"))
        path = Path(manifest["path"])
        require(path.is_file(), f"required local raw file is unavailable: {path}")
        require(path.stat().st_size == int(manifest["size_bytes"]), f"raw size mismatch: {path}")
        full_sha = sha256_file(path) if verify_full_sha256 else manifest["sha256"]
        require(full_sha == manifest["sha256"], f"raw SHA-256 mismatch: {path}")
        with netCDF4.Dataset(path) as dataset:
            for name in (
                "Header.Toltec.RoachIndex",
                "Header.Toltec.ObsNum",
                "Header.Toltec.FpgaFreq",
                "Header.Toltec.AccumLen",
                "Header.Toltec.SampleFreq",
                "Data.Toltec.Is",
                "Data.Toltec.Qs",
                "Data.Toltec.Ts",
                "Data.Toltec.RecvTime",
            ):
                require(name in dataset.variables, f"{path} lacks {name}")
            roach = int(dataset["Header.Toltec.RoachIndex"][...].item())
            obsnum = int(dataset["Header.Toltec.ObsNum"][...].item())
            fpga = float(dataset["Header.Toltec.FpgaFreq"][...].item())
            accum = int(dataset["Header.Toltec.AccumLen"][...].item())
            sample_rate = float(dataset["Header.Toltec.SampleFreq"][...].item())
            is_var = dataset["Data.Toltec.Is"]
            qs_var = dataset["Data.Toltec.Qs"]
            ts_var = dataset["Data.Toltec.Ts"]
            recv_var = dataset["Data.Toltec.RecvTime"]
            require(roach == network, f"RoachIndex mismatch in {path}")
            require(obsnum == 148670, f"ObsNum mismatch in {path}")
            require(math.isclose(fpga, 256_000_000.0, rel_tol=0.0, abs_tol=0.0), "FpgaFreq changed")
            require(accum == 2_097_152, "AccumLen changed")
            require(math.isclose(accum / fpga, CADENCE_SEC, rel_tol=0.0, abs_tol=1e-15), "cadence changed")
            require(math.isclose(sample_rate, 1.0 / CADENCE_SEC, rel_tol=0.0, abs_tol=1e-12), "SampleFreq changed")
            require(tuple(ts_var.dimensions) == ("time", "tlen"), "Ts dimensions changed")
            require(tuple(is_var.dimensions[:1]) == ("time",), "Is time dimension changed")
            require(tuple(qs_var.dimensions[:1]) == ("time",), "Qs time dimension changed")
            require(tuple(recv_var.dimensions) == ("time",), "RecvTime dimension changed")
            require(getattr(ts_var, "long_name", "") == EXPECTED_TS_LONG_NAME, "Ts long_name changed")
            native_rows = int(ts_var.shape[0])
            require(is_var.shape[0] == native_rows, "Is/Ts row mismatch")
            require(qs_var.shape[0] == native_rows, "Qs/Ts row mismatch")
            require(recv_var.shape[0] == native_rows, "RecvTime/Ts row mismatch")
            ts = np.asarray(ts_var[:], dtype=np.int64)
            recv = np.asarray(recv_var[:], dtype=np.float64)
        reconstructed = reconstruct_legacy_timestamp(ts, fpga)
        packet_step = modular_difference(u32(ts[1:, 3]), u32(ts[:-1, 3]))
        clock_step = modular_difference(u32(ts[1:, 2]), u32(ts[:-1, 2]))
        recv_delta = recv - reconstructed
        rows.append(
            {
                "observation_number": 148670,
                "network_id": network,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": full_sha,
                "full_sha256_verified_now": str(bool(verify_full_sha256)).lower(),
                "timing_projection_sha256": manifest["timing_projection_sha256"],
                "native_rows": native_rows,
                "is_qs_ts_recvtime_share_time_dimension": "true",
                "fpga_hz": format(fpga, ".17g"),
                "accumulation_ticks": accum,
                "cadence_sec": format(accum / fpga, ".17g"),
                "packet_increment_mismatch_count": int(np.sum(packet_step != 1)),
                "clock_increment_mismatch_count": int(np.sum(clock_step != accum)),
                "recvtime_strictly_increasing": str(bool(np.all(np.diff(recv) > 0))).lower(),
                "recv_minus_reconstructed_min_sec": format(float(np.min(recv_delta)), ".17g"),
                "recv_minus_reconstructed_median_sec": format(float(np.median(recv_delta)), ".17g"),
                "recv_minus_reconstructed_max_sec": format(float(np.max(recv_delta)), ".17g"),
                "recv_minus_reconstructed_std_sec": format(float(np.std(recv_delta)), ".17g"),
                "physical_event_semantics": "unresolved",
            }
        )
    return rows


def verify_three_map_inputs(result_root: Path, aggregate_root: Path) -> dict[str, Any]:
    joined = read_csv(result_root / "joined_network_records.csv")
    pairs = read_csv(result_root / "pair_summary.csv")
    require(len(joined) == 33, "joined three-map record count changed")
    expected_keys = {
        (obsnum, network) for obsnum in EXPECTED_MAPS for network in EXPECTED_NETWORKS
    }
    actual_keys = {(int(row["observation_number"]), int(row["network_id"])) for row in joined}
    require(actual_keys == expected_keys, "joined map/network identity changed")
    for row in joined:
        obsnum = int(row["observation_number"])
        require(row["map_id"] == EXPECTED_MAPS[obsnum], "map identity changed")
        require(row["association_class"] == "same_row_only", "association class changed")
        require(row["raw_linkage_status"] == "proved_original_row_one_to_one", "raw linkage changed")
        require(row["raw_timestamp_physical_semantics"] == "unresolved", "physical semantics unexpectedly resolved")
        require(row["variable_metadata_latency_observed"] == "False", "variable latency flag changed")

    pair_labels = {
        (int(row["observation_a"]), int(row["observation_b"])): int(row["modal_half_cadence_index"])
        for row in pairs
    }
    require(len(pair_labels) == 3, "pair set changed")
    states = transitive_states(pair_labels, 148670)
    require(states == EXPECTED_HALF_STATES, "transitive half-cadence state changed")

    sessions = [
        row
        for row in read_csv(aggregate_root / "session_registry.csv")
        if int(row["obsnum"]) in EXPECTED_MAPS
    ]
    require(len(sessions) == 3, "session registry three-map selection changed")
    for row in sessions:
        obsnum = int(row["obsnum"])
        require(row["map_id"] == EXPECTED_MAPS[obsnum], "session map identity changed")
        require(row["t0_session_key"] == EXPECTED_GROUP, "frozen T0 group changed")

    occurrence = [
        row
        for row in read_csv(aggregate_root / "pps_time_increment_occurrence.csv")
        if int(row["observation_number"]) in EXPECTED_MAPS
    ]
    require(len(occurrence) == 33, "PpsTime occurrence record count changed")
    anomaly_rows = [
        row
        for row in read_csv(aggregate_root / "raw_pps_time_increment_anomalies.csv")
        if int(row["observation_number"]) in EXPECTED_MAPS
    ]
    max_anomaly_sec = max(
        (abs(float(row["delivered_reconstructed_timestamp_step_residual_sec"])) for row in anomaly_rows),
        default=0.0,
    )
    require(max_anomaly_sec <= 1.0 / 256_000_000.0 + 1e-18, "unexpected PpsTime anomaly magnitude")

    return {
        "joined": joined,
        "pairs": pairs,
        "pair_labels": pair_labels,
        "states": states,
        "occurrence": occurrence,
        "anomaly_rows": anomaly_rows,
        "max_anomaly_sec": max_anomaly_sec,
    }


def build_semantic_matrix() -> list[dict[str, str]]:
    return [
        {
            "layer": "physical_detector_integration",
            "field_or_object": "analog/digital integration represented by D[n]",
            "proved_event": "none",
            "proved_association": "none upstream of delivered packet",
            "unresolved": "start, end, effective centroid, and integration-to-metadata association",
            "authority": "producer FPGA source required; recorded unavailable",
        },
        {
            "layer": "producer_udp",
            "field_or_object": "T0, incremental PPS counter, internal-clock counter",
            "proved_event": "fields carried in UDP under owner-supplied producer authority",
            "proved_association": "shared packet presence only",
            "unresolved": "atomic capture and which detector integration each value labels",
            "authority": "frozen producer clarification; FPGA source unavailable",
        },
        {
            "layer": "delivered_netcdf_row",
            "field_or_object": "Is[n,*], Qs[n,*], Ts[n,0:6], RecvTime[n]",
            "proved_event": "one shared NetCDF time index",
            "proved_association": "equal cardinality and common time dimension; direct for 148670",
            "unresolved": "whether the producer placed metadata from the same physical integration",
            "authority": "retained raw schema and compatibility input contract",
        },
        {
            "layer": "detector_counters",
            "field_or_object": "Ts[:,1] PpsCount; Ts[:,2] ClockCount; Ts[:,3] PacketCount; Ts[:,4] PpsTime",
            "proved_event": "delivered counter geometry and increments only",
            "proved_association": "same delivered row; all three maps have same-row PpsCount/PpsTime transitions",
            "unresolved": "physical simultaneity, ISR capture latency, and metadata-to-integration association",
            "authority": "compact three-map counter diagnostics",
        },
        {
            "layer": "consumer_native_time",
            "field_or_object": "anchor + PpsCount + (ClockCount-PpsTime)/FpgaFreq",
            "proved_event": "exact Citlali delivered-coordinate constructor",
            "proved_association": "same Ts[n] row retained with D[n] downstream",
            "unresolved": "physical event denoted by the constructed coordinate",
            "authority": "timestream_alignment_helpers.h and Stage-A lineage",
        },
        {
            "layer": "consumer_assigned_slot",
            "field_or_object": "nearest 8.192-ms common-grid slot",
            "proved_event": "software assignment coordinate",
            "proved_association": "one-to-one increasing delivered-row mapping on admitted support",
            "unresolved": "absolute physical sky-placement time",
            "authority": "Citlali implementation and Stage-A lineage",
        },
        {
            "layer": "telescope_ingress_orthogonal",
            "field_or_object": "Data.TelescopeBackend.PpsTime and 20-ms telescope rows",
            "proved_event": "not evaluated in this audit",
            "proved_association": "none claimed",
            "unresolved": "retained for SCI-TEL-INPUT-001",
            "authority": "scope boundary",
        },
    ]


def build_hypothesis_rows(result: Mapping[str, Any]) -> list[dict[str, str]]:
    odd_pairs = sorted(
        f"{left}->{right}:{label:+d}"
        for (left, right), label in result["pair_labels"].items()
        if label % 2
    )
    return [
        {
            "hypothesis": "consumer-side D/Ts row permutation or off-by-one",
            "disposition": "falsified within delivered-to-retained lineage",
            "evidence": "33/33 records have proved one-to-one raw linkage; Stage A found no row permutation",
            "limit": "does not inspect upstream FPGA association",
        },
        {
            "hypothesis": "changed delivered PpsCount/PpsTime same-vs-adjacent-row association",
            "disposition": "falsified for the frozen three maps",
            "evidence": "33/33 records are same_row_only; zero pairwise association-class changes",
            "limit": "same delivered row is not physical simultaneity",
        },
        {
            "hypothesis": "whole-row integer reassociation alone explains all modal labels",
            "disposition": "falsified as an exact label generator",
            "evidence": "one row equals two half steps, but odd pair labels remain: " + ", ".join(odd_pairs),
            "limit": "integer reassociation may still contribute to even components",
        },
        {
            "hypothesis": "retained PpsTime increment anomalies are the primary cause",
            "disposition": "strongly disfavored",
            "evidence": "anomaly-free controls retain the bands; individual delivered timestamp step residuals are at most one 256-MHz tick",
            "limit": "compact controls are descriptive and not statistically independent",
        },
        {
            "hypothesis": "stable native detector-frame phase alone explains map changes",
            "disposition": "strongly disfavored",
            "evidence": "accepted median absolute pairwise native-phase change is 0.0167 ms versus 8.477 ms timing change",
            "limit": "native phase has no established physical event semantics",
        },
        {
            "hypothesis": "host RecvTime latency is the Citlali timing coordinate",
            "disposition": "falsified at the consumer boundary",
            "evidence": "RecvTime is retained on the raw row but the timestamp constructor does not consume it",
            "limit": "host buffering before NetCDF insertion remains producer-packaging context, not a physical event clock",
        },
        {
            "hypothesis": "half-cadence association or map-varying start/end/centroid semantics",
            "disposition": "survives; descriptively compatible, not identified",
            "evidence": "transitive states {148670:0,150819:-3,151126:-2} exactly reproduce pair modes",
            "limit": "producer FPGA event authority is unavailable; half lattice nests full lattice",
        },
        {
            "hypothesis": "upstream non-atomic or adjacent integration/metadata packaging",
            "disposition": "survives",
            "evidence": "delivered row lineage begins after the producer packet association",
            "limit": "requires producer source or equivalent acquisition-level event trace",
        },
        {
            "hypothesis": "map fitting creates the descriptive bands",
            "disposition": "survives as an undistinguished alternative",
            "evidence": "audit does not refit maps and accepted labels remain descriptive",
            "limit": "bounded scope prohibits new reduction or map fitting",
        },
    ]


def build_source_counter_registry(aggregate_root: Path) -> list[dict[str, str]]:
    selected_names = {
        "SHA256SUMS",
        "input_manifest.csv",
        "input_manifest.json",
        "raw_counter_transitions.csv",
        "raw_phase_summary.csv",
        "raw_pps_time_increment_anomalies.csv",
    }
    selected_map_ids = set(EXPECTED_MAPS.values())
    rows = []
    for row in read_csv(aggregate_root / "input_digests.csv"):
        path = Path(row["path"])
        if row["map_id"] not in selected_map_ids or path.name not in selected_names:
            continue
        rows.append(
            {
                "map_id": row["map_id"],
                "observation_number": next(
                    obsnum for obsnum, map_id in EXPECTED_MAPS.items() if map_id == row["map_id"]
                ),
                "source_path_recorded_in_owner_return": row["path"],
                "sha256": row["sha256"],
                "exact_source_file_local": "false",
                "local_use": "identity/digest record only; aggregate projection used for analysis",
            }
        )
    rows.sort(key=lambda row: (int(row["observation_number"]), row["source_path_recorded_in_owner_return"]))
    require(len(rows) == 18, "three-map source counter artifact registry is incomplete")
    return rows


def report_text(summary: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]) -> str:
    medians_ms = [1000.0 * float(row["recv_minus_reconstructed_median_sec"]) for row in raw_rows]
    maxima_ms = [1000.0 * float(row["recv_minus_reconstructed_max_sec"]) for row in raw_rows]
    return f"""# SCI-ALIGN-001 acquisition-boundary event-semantics audit

Date: 2026-08-08

## Stopping result

**STOP — essential producer evidence is unavailable.** The physical event
represented by a detector integration row cannot be identified from the
retained evidence. The accepted producer clarification says that the FPGA
source is not presently available, and no authoritative event-level
specification binds integration accumulation, counter capture, PPS ISR update,
UDP assembly, and NetCDF row insertion. This is the required stopping result,
not an authorization to retrieve source, contact a producer, run a reduction,
or design a correction.

No timing correction is proposed or authorized. The accepted modal labels
remain descriptive only.

## Frozen identity and inspected evidence

- entry commit: `{EXPECTED_HEAD}`;
- parent: `{EXPECTED_PARENT}`;
- tree: `{EXPECTED_TREE}`;
- frozen group: `{EXPECTED_GROUP}`;
- maps: 148670, 150819, and 151126;
- networks: {', '.join(str(value) for value in EXPECTED_NETWORKS)}.

All 11 retained Beammap 148670 raw detector files were inspected directly and
their full-file SHA-256 values were reverified. `Is`, `Qs`, `Ts`, and
`RecvTime` share the NetCDF `time` dimension in every file. PacketCount and
ClockCount have zero increment mismatches. `RecvTime` is strictly increasing;
its per-network median minus the reconstructed detector coordinate spans
{min(medians_ms):.6f}--{max(medians_ms):.6f} ms, with rare maxima up to
{max(maxima_ms):.6f} ms. This is host-delivery/packaging evidence only:
Citlali does not use `RecvTime` to construct detector time, and its physical
event and clock remain undocumented.

For all three maps, the checksum-bound compact evidence retains 33/33
map/network records with one-to-one delivered raw linkage, same-row-only
PpsCount/PpsTime transition association, no association-class changes, and no
reported variable PpsCount/PpsTime transition latency. The exact campaign
per-map raw-counter artifact files are represented locally only by
owner-returned paths and SHA-256 digests; the accepted aggregate projection is
local. Direct detector raw is local only for 148670. This audit did not retrieve
any missing source file.

## Event mapping

The strongest supported mapping is:

1. Producer authority says each UDP packet carries T0, the incremental PPS
   counter, and the internal-clock counter. PPS does not restart the detector
   integration cadence.
2. The delivered NetCDF places `Is[n,*]`, `Qs[n,*]`, and the six `Ts[n,*]`
   fields on one row. This proves delivered packaging, not FPGA atomicity.
3. Citlali reconstructs the native coordinate from the first-row integer
   anchor plus PpsCount and `(ClockCount-PpsTime)/FpgaFreq`; it does not use
   RecvTime.
4. The same delivered detector row is carried through native-time
   reconstruction and nearest-slot assignment without a demonstrated
   permutation.
5. Which integration event the row represents—start, end, effective centroid,
   counter-capture instant, or another event—remains unresolved upstream of
   the delivered pair.

`semantic_boundary_matrix.csv` records the field-by-field boundary.

## Integer- and half-cadence test

A whole detector-row reassociation changes time by 8.192 ms, exactly two
4.096-ms half-step labels. It therefore preserves half-step parity. The frozen
pair modes include odd changes, 148670->150819 = -3 and 150819->151126 = +1.
No common additive offset changes those pair parities. Consequently a
whole-row integer reassociation **alone** cannot reproduce the exact three-map
modal label system. It may still contribute an even component and is not
excluded upstream.

The transitive half-step states
`{{148670: 0, 150819: -3, 151126: -2}}` reproduce all three modal pair labels,
so a half-cadence association or map-varying start/end/centroid convention is
descriptively compatible. It is not physically identified: the half lattice
nests the full lattice, only three maps are present, and producer event
semantics are unavailable.

The 1,437 retained PpsTime increment-anomaly rows each perturb the delivered
timestamp step by at most one 256-MHz tick (3.90625 ns), while anomaly-free
controls retain the timing bands. They are strongly disfavored as the primary
cause and do not authorize a repair or mask.

## Hypothesis disposition

Falsified or strongly disfavored within this boundary:

- a Citlali delivered-row permutation or off-by-one;
- a change in delivered same-row versus adjacent-row PpsCount/PpsTime class;
- whole-row integer reassociation as the sole exact generator of all labels;
- retained PpsTime increment anomalies as the primary cause;
- stable native detector-frame phase alone;
- host RecvTime as the coordinate consumed by Citlali.

Surviving without preference:

- acquisition-hardware integration phase or capture state;
- upstream non-atomic or adjacent integration/metadata packaging;
- map-varying timestamp start/end/effective-centroid semantics;
- map fitting.

`hypothesis_disposition.csv` gives the evidence and limit for each statement.

## Evidence limits and smallest next step

Unavailable essential evidence includes the exact FPGA/packetizer source or an
equivalent authoritative event specification, direct local raw detector files
for 150819 and 151126, and an independent physical integration-event marker.
The raw absence is not repaired by using telescope data, map fitting, or a new
reduction.

The smallest next step is an owner decision on owner-mediated acquisition of
the exact producer FPGA/packetizer source revision or an authoritative
event-level specification. That later material must bind integration
accumulation, counter capture, PPS ISR update, UDP assembly, and NetCDF row
insertion. This audit stops before retrieval, execution, external contact,
scientific policy choice, or correction.

The orthogonal 20-ms telescope-file ingress boundary was not evaluated or
reinterpreted. `SCI_TEL_INPUT_001_HANDOFF.md` contains only the bounded facts
relevant to that separate audit.
"""


def sci_tel_handoff_text() -> str:
    return """# Handoff to SCI-TEL-INPUT-001

Date: 2026-08-08

This handoff does not launch or broaden the 20-ms TolTECA/telescope-file
ingress audit.

- Detector `Data.Toltec.Ts[:,4]` is the TolTEC raw `PpsTime` clock-tick field.
  It must not be conflated with telescope
  `Data.TelescopeBackend.PpsTime`.
- The detector acquisition audit establishes delivered `D[n]/Ts[n]` row
  lineage but not the physical integration event represented by that row.
  Detector time therefore cannot serve as an absolute physical oracle for the
  telescope-ingress audit without later producer authority.
- No telescope row, timestamp, interpolation, 20-ms association, or
  recomputation was inspected in this audit.
- The three descriptive same-T0 labels remain non-physical and must not be
  imported into SCI-TEL-INPUT-001 as a correction or prior.
"""


def write_sha256sums(root: Path) -> None:
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.name}\n" for path in paths]
    (root / "SHA256SUMS").write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-raw-file-sha256", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    aggregate = args.aggregate_root.resolve()
    output = args.output_root.resolve()
    require(output.parent.is_dir(), f"output parent is unavailable: {output.parent}")
    require(not output.exists(), f"output already exists: {output}")

    result_root = repo / "validation/sci_align_001_split_direction_beammap_2026-08-06/same_t0_cadence_lattice_result_2026-08-08"
    three_map = verify_three_map_inputs(result_root, aggregate)
    protocol = json.loads(
        (repo / "validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json").read_text()
    )
    producer_lines = protocol["producer_authority"]["clock_architecture"]
    require("FPGA source is not presently available" in producer_lines, "producer-source availability changed")
    raw_manifest_path = repo / "validation/sci_align_001_phase0_2026-08-01/selected_detector_input_manifest.csv"
    raw_rows = inspect_raw_148670(
        load_phase0_raw_manifest(raw_manifest_path), args.verify_raw_file_sha256
    )
    require(all(int(row["packet_increment_mismatch_count"]) == 0 for row in raw_rows), "148670 packet mismatch found")
    require(all(int(row["clock_increment_mismatch_count"]) == 0 for row in raw_rows), "148670 clock mismatch found")

    output.mkdir()
    semantic_rows = build_semantic_matrix()
    hypothesis_rows = build_hypothesis_rows(three_map)
    source_counter_rows = build_source_counter_registry(aggregate)
    summary = {
        "schema": "sci-align-001-acquisition-event-semantics-audit-v1",
        "identity": {
            "head": EXPECTED_HEAD,
            "parent": EXPECTED_PARENT,
            "tree": EXPECTED_TREE,
            "frozen_group": EXPECTED_GROUP,
            "maps": EXPECTED_MAPS,
            "networks": list(EXPECTED_NETWORKS),
        },
        "scope": "bounded read-only acquisition-boundary event-semantics audit",
        "result": "STOP_REQUIRED_ESSENTIAL_PRODUCER_EVIDENCE_UNAVAILABLE",
        "physical_event_identified": False,
        "correction_authorized": False,
        "producer_fpga_source_available": False,
        "delivered_boundary": {
            "direct_raw_maps": [148670],
            "compact_counter_maps": [148670, 150819, 151126],
            "direct_raw_network_count": len(raw_rows),
            "joined_compact_record_count": len(three_map["joined"]),
            "all_joined_raw_linkage_proved": True,
            "all_delivered_pps_association_same_row": True,
            "variable_metadata_latency_observed": False,
        },
        "cadence_tests": {
            "cadence_sec": CADENCE_SEC,
            "half_cadence_sec": HALF_CADENCE_SEC,
            "pair_modal_half_labels": {
                f"{left}->{right}": value
                for (left, right), value in sorted(three_map["pair_labels"].items())
            },
            "transitive_half_states": three_map["states"],
            "integer_whole_row_only_exactly_compatible": labels_are_integer_row_compatible(three_map["pair_labels"]),
            "half_cadence_descriptively_compatible": True,
            "half_cadence_physical_mechanism_identified": False,
        },
        "pps_time_anomalies": {
            "row_count": len(three_map["anomaly_rows"]),
            "maximum_absolute_delivered_timestamp_step_residual_sec": three_map["max_anomaly_sec"],
            "primary_cause_disposition": "strongly_disfavored",
        },
        "surviving_alternatives": [
            "acquisition hardware integration-phase or capture state",
            "upstream non-atomic or adjacent integration/metadata packaging",
            "map-varying timestamp start/end/effective-centroid semantics",
            "map fitting",
        ],
        "smallest_next_step": (
            "Owner-mediated acquisition of the exact producer FPGA/packetizer source revision "
            "or an authoritative event-level specification that binds integration accumulation, "
            "counter capture, PPS ISR update, UDP assembly, and NetCDF row insertion. Stop before "
            "retrieval, execution, correction, or policy choice."
        ),
        "orthogonal_telescope_ingress": "not launched; facts handed to SCI-TEL-INPUT-001 only",
    }
    (output / "audit_summary.json").write_text(canonical_json(summary))
    write_csv(
        output / "semantic_boundary_matrix.csv",
        semantic_rows,
        ("layer", "field_or_object", "proved_event", "proved_association", "unresolved", "authority"),
    )
    write_csv(
        output / "hypothesis_disposition.csv",
        hypothesis_rows,
        ("hypothesis", "disposition", "evidence", "limit"),
    )
    write_csv(
        output / "raw_148670_packaging_summary.csv",
        raw_rows,
        tuple(raw_rows[0].keys()),
    )
    write_csv(
        output / "source_counter_artifact_registry.csv",
        source_counter_rows,
        (
            "map_id",
            "observation_number",
            "source_path_recorded_in_owner_return",
            "sha256",
            "exact_source_file_local",
            "local_use",
        ),
    )
    (output / "REPORT.md").write_text(report_text(summary, raw_rows))
    (output / "SCI_TEL_INPUT_001_HANDOFF.md").write_text(sci_tel_handoff_text())

    inputs = [
        repo / "AGENTS.md",
        repo / "validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json",
        repo / "validation/sci_align_001_phase0_2026-08-01/selected_detector_input_manifest.csv",
        repo / "validation/sci_align_001_sample_lineage_phase_2026-08-03/stage_a_conclusion.json",
        repo / "validation/sci_align_001_sample_lineage_phase_2026-08-03/lineage_source_trace.csv",
        repo / "validation/sci_align_001_split_direction_beammap_2026-08-06/SAME_T0_CADENCE_LATTICE_EVIDENCE_2026-08-08.md",
        result_root / "joined_network_records.csv",
        result_root / "pair_summary.csv",
        result_root / "pairwise_network_differences.csv",
        aggregate / "corpus_summary.json",
        aggregate / "network_map_results.csv",
        aggregate / "pps_time_increment_occurrence.csv",
        aggregate / "raw_pps_time_increment_anomalies.csv",
        aggregate / "session_registry.csv",
        aggregate / "input_digests.csv",
        repo / "include/citlali/core/engine/detail/sci_align_netcdf_input_contract.h",
        repo / "include/citlali/core/engine/detail/todproc_alignment_impl.h",
        repo / "include/citlali/core/pipeline/timestream_alignment_helpers.h",
    ]
    input_rows = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "authoritative_or_accepted_input",
        }
        for path in inputs
    ]
    write_csv(output / "input_manifest.csv", input_rows, ("path", "size_bytes", "sha256", "role"))
    write_sha256sums(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
