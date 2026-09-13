"""Account preserved RTC evidence in native detector-time; never run or apply RTC.

This intentionally consumes one sealed corpus generation. Exposure uses the
original producer/initial-VAL paired eligibility, not peer eligibility or a
successful fit. All intervals are half-open integer microseconds relative to
the first native center of one network input. JSON records retain native rows,
coordinate origins and source identities. No processing scan is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from netCDF4 import Dataset

SOURCE = "81a35ced25d45eeccafabc3243b5aa3010893625"
CLOSURE = "8fff882dd0e8891512b01e6065cdb442981b1e5d"
SEAL = "c4b0bf26be49a9a400286ee90a48d0d742f627b98cec7015708e8e04893169ba"
US = 1_000_000
MINIMUM_SECONDS = (1, 5, 10)
EXTRA_GUARD_MS = (0, 50, 100)
RECOVERY = ("recovered", "background_unavailable", "onset_unavailable",
            "search_limit", "observation_end", "acquisition_gap",
            "invalid_support", "nonfinite")
SETS = ("transition", "finite_recovery", "direct", "fitting_guard",
        "candidate_edge", "candidate_inclusive", "noise_screening_required")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def compressed_writer(path):
    # No filename or wall clock enters the compressed scientific artifact.
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                yield stream


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        require(isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))
                and a < b, f"invalid integer interval {(a, b)}")
        a, b = int(a), int(b)
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def intersect(left, right):
    left, right = union(left), union(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(support, exclusions):
    support, exclusions = union(support), union(exclusions)
    result, j = [], 0
    for a, b in support:
        while j < len(exclusions) and exclusions[j][1] <= a:
            j += 1
        k, cursor = j, a
        while k < len(exclusions) and exclusions[k][0] < b:
            lo, hi = exclusions[k]
            if cursor < lo:
                result.append((cursor, min(lo, b)))
            cursor = max(cursor, hi)
            k += 1
        if cursor < b:
            result.append((cursor, b))
    return result


def measure(intervals):
    return sum(b - a for a, b in union(intervals))


def qualify(eligible, transitions, finite, minimum_us, extra_us=0):
    """Native boundary-delimited units, with transient holes retained as holes.

    Only transitions delimit new level units. A localized transient removes
    support inside its unit, but does not create a new level-shift boundary.
    The caller supplies acquisition/invalidity/scan boundaries explicitly by
    calling this per unit. No corpus call supplies or invents a PCA scan.
    """
    require(minimum_us >= 0 and extra_us >= 0, "negative scenario parameter")
    expanded = union((a - extra_us, b + extra_us) for a, b in intersect(transitions, eligible))
    units = subtract(eligible, expanded)
    segments, retained = [], []
    for a, b in units:
        remaining = subtract([(a, b)], finite)
        available = measure(remaining)
        good = available >= minimum_us and available > 0
        segments.append(dict(interval_us=[a, b], remaining_us=available,
                             duration_qualified=good))
        if good:
            retained.extend(remaining)
    lengths = [s["remaining_us"] for s in segments if s["duration_qualified"]]
    return dict(segments=segments, retained=union(retained), all_us=sum(lengths),
                longest_us=max(lengths, default=0))


def concurrency(detector_sets, domain):
    """A histogram in network wall-time, unioning each detector before sweeping."""
    changes = Counter()
    for intervals in detector_sets:
        for a, b in intersect(intervals, domain):
            changes[a] += 1
            changes[b] -= 1
    histogram = Counter()
    points = sorted(changes)
    active, previous = 0, None
    for point in points:
        if previous is not None and active:
            histogram[active] += point - previous
        active += changes[point]
        require(active >= 0, "negative concurrency")
        previous = point
    require(active == 0, "unterminated concurrency")
    histogram[0] = measure(domain) - sum(histogram.values())
    require(histogram[0] >= 0, "concurrency outside native support")
    return dict(sorted(histogram.items()))


@dataclass
class Axis:
    centers: np.ndarray
    begin: np.ndarray
    end: np.ndarray
    runs: list
    legacy_begin: np.ndarray
    legacy_end: np.ndarray
    metadata: dict

    def intervals(self, rows):
        a, b = rows
        require(0 <= a < b <= len(self.centers), f"invalid native rows {rows}")
        # Split every physical support gap; do not replace acquired support by
        # the enclosing first/last time interval.
        splits = np.flatnonzero(self.begin[a + 1:b] > self.end[a:b - 1]) + a + 1
        bounds = [a, *map(int, splits), b]
        return union((int(self.begin[x]), int(self.end[y - 1]))
                     for x, y in zip(bounds, bounds[1:]))

    def bound(self, bound):
        require(bound["available"], "unavailable transition used as support")
        rows = bound.get("affected", bound.get("rows"))
        intervals = self.intervals(rows)
        a, b = rows
        require(abs(bound["begin"] - self.legacy_begin[a]) <= 1e-6 and
                abs(bound["end"] - self.legacy_end[b - 1]) <= 1e-6,
                "saved bound does not bind the original timing cells")
        return intervals


def read_axis(raw, inventory, receipt):
    with Dataset(raw, "r") as dataset:
        ts = np.asarray(dataset["Data.Toltec.Ts"][:], dtype=np.int64)
        freq = float(dataset["Header.Toltec.FpgaFreq"][...])
        accum = int(dataset["Header.Toltec.AccumLen"][...])
        sample_hz = float(dataset["Header.Toltec.SampleFreq"][...])
        for name, key in (("ObsNum", "observation"), ("SubObsNum", "subobservation"),
                          ("ScanNum", "scan")):
            require(int(dataset["Header.Toltec." + name][...]) == inventory[key],
                    "raw producer scope mismatch: " + key)
    require(freq.is_integer() and freq > 0 and accum > 0, "unsupported clock units")
    frequency = int(freq)
    require(len(ts) == receipt["rows"], "native timing cardinality mismatch")
    require(math.isclose(sample_hz, freq / accum, rel_tol=1e-12), "cadence mismatch")
    phase = ts[:, 2] - ts[:, 4]
    phase = np.where(phase < 0, phase + 2**32 - 1, phase)
    ticks = ts[:, 1] * frequency + phase
    relative = ticks - ticks[0]
    require(np.all(np.diff(relative) > 0), "nonmonotone producer clock")
    # Exact integer tick arithmetic before rounding avoids subtracting Unix
    # doubles. The source adapter treats Ts columns as signed int32 values.
    center_us = np.floor_divide(relative * US + frequency // 2, frequency)
    begin_num, end_num = (2 * relative - accum) * US, (2 * relative + accum) * US
    begin = np.floor_divide(begin_num + frequency, 2 * frequency)
    end = np.floor_divide(end_num + frequency, 2 * frequency)
    error = max(float(np.max(np.abs(begin - begin_num / (2 * frequency)))),
                float(np.max(np.abs(end - end_num / (2 * frequency)))))
    require(error <= .500001 and np.all(end > begin), "time representation failed")
    counter_breaks = np.flatnonzero(np.diff(ts[:, 3]) != 1) + 1
    require(len(counter_breaks) + 1 == receipt["physical_runs"], "physical-run mismatch")
    # Also retain an actual physical-time hole even if a packet counter did
    # not signal it. Sub-microsecond rounding is declared in the output.
    breaks = sorted(set(map(int, counter_breaks)) |
                    set(map(int, np.flatnonzero(begin[1:] > end[:-1]) + 1)))
    endpoints = [0, *breaks, len(ts)]
    runs = list(zip(endpoints, endpoints[1:]))
    epoch = float(int(ts[0, 0] + ts[0, 5] * 1e-9 - .5))
    legacy_dt = ts[:, 2] / freq - ts[:, 4] / freq
    legacy_dt = np.where(legacy_dt < 0, legacy_dt + (2.**32 - 1) / freq, legacy_dt)
    legacy = epoch + ts[:, 1] + legacy_dt
    metadata = dict(units="integer microseconds", origin="first native integration center",
                    producer_clock="producer-native-clock:no-telescope-sync:census-only",
                    epoch_authority="not established", cross_network_relation="unavailable",
                    first_center_recorded_unix_seconds=float(legacy[0]),
                    fpga_hz=frequency, accumulation_length=accum,
                    sample_hz=sample_hz, integration_role="center",
                    averaging="provisional uniform averaging, inherited",
                    maximum_endpoint_rounding_us=error, physical_run_rows=runs,
                    producer_counter_run_count=int(len(counter_breaks) + 1),
                    processing_scan_relation="unavailable")
    return Axis(center_us, begin, end, runs, legacy - accum / freq / 2,
                legacy + accum / freq / 2, metadata)


def verify_seal(root):
    seal = root / "SHA256SUMS"
    require(digest(seal) == SEAL, "wrong preserved evidence generation")
    hashes = {}
    for line in seal.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix("*").removeprefix("./")
        path = (root / name).resolve()
        require(root in path.parents, "manifest path escapes evidence root")
        require(name not in hashes and digest(path) == expected, "evidence mismatch: " + name)
        hashes[name] = expected
    binding = json.loads((root / "source-binding.json").read_text())
    require(binding["revision"] == SOURCE, "wrong event source")
    return hashes


def coordinate_evidence(event, initial, audit, check, coordinate, axis):
    """Select the saved final transition once; never reuse a superseded bound.

    Initial finite-recovery support is usable when its fit was not reassessed.
    A reassessed recovery has only a cause in this generation's serialization,
    not its new affected rows. Preserve that absence rather than reuse an old
    fit's extent or invent an export by rerunning the science.
    """
    original = event["coordinates"][coordinate]
    final = audit["coordinates"][coordinate] if audit else None
    retained = final is not None and final["diagnostic_cause"] in (0, 7)
    refit = bool(audit and audit["refit_requested"])
    result = dict(coordinate="xr"[coordinate], seeded=original["seeded"],
                  initial_background_available=original["available"],
                  initial_recovery_cause=RECOVERY[original["recovery_cause"]],
                  initial_recovery_rows=original["affected"],
                  initial_transition_available=initial["available"],
                  final_reassessment_cause=final["diagnostic_cause"] if final else None,
                  refit_requested=refit, sigma_delta=check["sigma_delta"],
                  sigma_delta_meaning="empirical difference scale, not offset uncertainty",
                  transition_us=[], finite_recovery_us=[], unresolved=False)
    if retained:
        bound = initial if final["diagnostic_cause"] == 0 else final["transition"]
        result.update(state="transition_supported", bound_version=(
            "initial" if final["diagnostic_cause"] == 0 else "remeasured"),
            native_rows=bound.get("affected", bound.get("rows")),
            transition_us=axis.bound(bound),
            recorded_physical_bound_seconds=[bound["begin"], bound["end"]],
            offset=(final["primary_offset"] if refit else original["with_offset"])["offset"])
    elif refit:
        result.update(state=("reassessed_recovery_extent_unavailable"
                             if final["recovery_cause"] == 0 else "unresolved_after_reassessment"),
                      unresolved=True)
    elif original["available"] and original["recovery_cause"] == 0:
        if original["affected"] == [-1, -1]:
            result["state"] = "no_resolved_excursion"
        else:
            result.update(state="finite_recovery_supported", native_rows=original["affected"],
                          finite_recovery_us=axis.intervals(original["affected"]))
    else:
        result.update(state="unresolved", unresolved=True)
    return result


def verify_validity(folder, detectors, axis):
    """Prove all initial samples valid, or stop instead of filling missing masks.

    A maximum admitted-edge count covers both endpoints of every sample in
    each physical run. Counts alone cannot recover a partial invalidity mask.
    This fixed corpus has complete coverage; a future partial case needs an
    explicit validity export and is deliberately rejected here.
    """
    coverage = defaultdict(list)
    noise = defaultdict(list)
    noise_rows = Counter()
    run_by_row = np.empty(len(axis.centers), dtype=np.int64)
    for a, b in axis.runs:
        run_by_row[a:b] = a
    for block in records(folder / "health-blocks.jsonl"):
        detector, first, end = block["detector"], block["first"], block["end"]
        require(detector in detectors and 0 <= first < end <= len(axis.centers),
                "malformed noise block")
        run_first = int(run_by_row[first])
        require(run_by_row[end - 1] == run_first, "noise block spans acquisition break")
        coverage[detector].append((first, end))
        if detectors[detector]["tune_valid"]:
            require(end > run_first + 1, "isolated sample validity cannot be reconstructed")
            expected = end - max(first, run_first + 1)
            require(all(c["admitted_differences"] == expected for c in block["coordinates"]),
                    "partial validity unavailable; exact per-cell mask export required")
        else:
            require(all(c["admitted_differences"] == 0 for c in block["coordinates"]),
                    "Tune-invalid stream has admitted differences")
        if any(c["scale_cause"] != 0 for c in block["coordinates"]):
            noise[detector].extend(axis.intervals((first, end)))
            noise_rows[detector] += end - first
        else:
            require(all(c["scale"] is not None and c["scale"] > 0
                        for c in block["coordinates"]), "available nonpositive noise scale")
    for detector, meta in detectors.items():
        blocks = sorted(coverage[detector])
        require(blocks and blocks[0][0] == 0 and blocks[-1][1] == len(axis.centers)
                and all(x[1] == y[0] for x, y in zip(blocks, blocks[1:])),
                "noise blocks do not partition native rows exactly")
        require(noise_rows[detector] == meta["pair_screening_excluded_rows"],
                "noise-screening row count mismatch")
    return {d: union(v) for d, v in noise.items()}


def array_name(network):
    # Exact scope already verified by the original canonical target manifest;
    # SCIENTIFIC_CONVENTIONS.md fixes the present interface-to-array mapping.
    require(0 <= network <= 12, "unknown instrument network")
    return "a1100" if network <= 6 else "a1400" if network <= 10 else "a2000"


def program_metadata(root, selected):
    """Read only telescope identity/program headers, never map products or times."""
    result, sources = defaultdict(list), []
    if root is None:
        return {}, []
    for path in sorted(root.rglob("tel*_recomputed.nc")):
        with Dataset(path, "r") as dataset:
            required = ["Header.Dcs." + k for k in ("ObsNum", "SubObsNum", "ScanNum", "ObsGoal", "ObsPgm")]
            if not all(k in dataset.variables for k in required):
                continue
            key = tuple(int(dataset[k][...]) for k in required[:3])
            if key not in selected:
                continue
            def string(name):
                value = np.asarray(dataset[name][:])
                return value.tobytes().decode("utf-8").rstrip("\x00 ")
            labels = dict(goal=string(required[3]), program=string(required[4]))
        record = dict(path=str(path), sha256=digest(path), scope=list(key), **labels)
        sources.append(record)
        result[key].append(record)
    labels = {}
    for key, values in result.items():
        distinct = {(v["goal"], v["program"]) for v in values}
        labels[key] = (dict(goal=next(iter(distinct))[0], program=next(iter(distinct))[1],
                            state="verified producer headers", sources=[v["sha256"] for v in values])
                       if len(distinct) == 1 else dict(state="unavailable: conflicting program headers"))
    return labels, sources


def process_network(entry, inventory, root, output, event_stream, detector_stream,
                    duration_stream, programs):
    obs, network = entry["observation"], entry["network"]
    folder = root / "campaign-01" / f"{obs}-{network:02d}"
    receipt = json.loads((folder / "receipt.json").read_text())
    require(entry["status"] == "completed" and receipt == entry["receipt"], "incomplete invocation")
    require(receipt["source_protection"] == "unavailable" and
            receipt["spectral_context"] == "owner-deferred", "unexpected scientific profile")
    inputs, verification_started = [], time.monotonic()
    for path, field in zip(entry["command"][1:4], ("raw_sha256", "tune_sha256", "manifest_sha256")):
        actual = digest(path)
        require(actual == receipt[field], "original input changed: " + path)
        inputs.append(dict(path=path, sha256=actual, bytes=Path(path).stat().st_size))
    input_seconds = time.monotonic() - verification_started
    axis = read_axis(entry["command"][1], inventory, receipt)
    with compressed_writer(output / "native-time" / (folder.name + ".jsonl.gz")) as stream:
        emit(stream, dict(observation=obs, network=network, raw_sha256=receipt["raw_sha256"],
                          **axis.metadata))
        for row in range(len(axis.centers)):
            emit(stream, [row, int(axis.centers[row]), int(axis.begin[row]), int(axis.end[row])])
    detectors = {d["detector"]: d for d in records(folder / "detectors.jsonl")}
    require(sorted(detectors) == list(range(receipt["channels"])), "detector sequence mismatch")
    require(len({d["occurrence"] for d in detectors.values()}) == len(detectors), "duplicate occurrence")
    noise = verify_validity(folder, detectors, axis)
    seeds = {c["candidate"]: c for c in records(folder / "candidates.jsonl")}
    require(sorted(seeds) == list(range(receipt["candidate_edges"])), "candidate sequence mismatch")
    audits = {a["event"]: a for a in records(folder / "jump-reassessment.jsonl")}
    sets = {d: defaultdict(list) for d in detectors}
    counts = {d: Counter() for d in detectors}
    seen_candidates = set()
    event_count = retained_count = retained_groups = 0
    coordinate_states = Counter()
    for event, initial, check in zip(records(folder / "events.jsonl"),
                                    records(folder / "jump-transitions.jsonl"),
                                    records(folder / "jump-consistency.jsonl"), strict=True):
        eid, detector = event["event"], event["detector"]
        require(eid == event_count == initial["event"] == check["event"], "event sequence mismatch")
        require(detector == initial["detector"] == check["detector"] and detector in detectors,
                "event detector mismatch")
        require(detectors[detector]["tune_valid"], "event on initially invalid detector")
        require(not event["hard_event_accepted"] and not event["apply_authorized"], "unexpected Apply")
        count, support = counts[detector], sets[detector]
        count["candidate_groups"] += 1
        count["candidate_edges"] += len(event["candidates"])
        candidate_by_coordinate = [[], []]
        for candidate in event["candidates"]:
            require(candidate not in seen_candidates, "duplicate candidate membership")
            seen_candidates.add(candidate)
            c = seeds[candidate]
            require(c["detector"] == detector and c["later_row"] == c["earlier_row"] + 1,
                    "candidate relation mismatch")
            candidate_by_coordinate[c["seed_coordinate"]].extend(
                axis.intervals((c["earlier_row"], c["later_row"] + 1)))
        edges = union(candidate_by_coordinate[0] + candidate_by_coordinate[1])
        support["candidate_edge"].extend(edges)
        guard_rows = [event["trial_exclusion"], *event["neighbor_exclusions"]]
        guards = union(x for rows in guard_rows for x in axis.intervals(rows))
        support["fitting_guard"].extend(guards)
        audit = audits.get(eid)
        if audit:
            require(audit["detector"] == detector and audit["network"] == network
                    and audit["scan_assignment"].startswith("unavailable:"), "unexpected reassessment binding")
        coordinates = [coordinate_evidence(event, initial["coordinates"][c], audit,
                                           check["coordinates"][c], c, axis) for c in range(2)]
        found, finite, unresolved = [], [], []
        for c, value in enumerate(coordinates):
            coordinate_states[value["state"]] += 1
            if value["state"] == "transition_supported":
                found.append("xr"[c])
                retained_count += 1
                count["retained_coordinates"] += 1
                support["transition"].extend(value["transition_us"])
                support["transition_" + "xr"[c]].extend(value["transition_us"])
            if value["state"] == "finite_recovery_supported":
                finite.append("xr"[c])
                count["finite_recovery_coordinates"] += 1
                support["finite_recovery"].extend(value["finite_recovery_us"])
            if value["unresolved"]:
                unresolved.append("xr"[c])
                count["unresolved_coordinates"] += 1
                count["extent_unavailable_after_refit"] += value["state"] == "reassessed_recovery_extent_unavailable"
        if found:
            retained_groups += 1
            count["transition_groups"] += 1
            count["transition_groups_" + "".join(found)] += 1
        count["finite_recovery_groups"] += bool(finite)
        count["unresolved_groups"] += bool(unresolved)
        count["partner_unresolved_with_transition"] += bool(found and unresolved)
        count["no_resolved_excursion_groups"] += all(c["state"] == "no_resolved_excursion" for c in coordinates)
        emit(event_stream, dict(observation=obs, network=network, detector=detector,
                                producer_group=eid, candidates=event["candidates"],
                                physical_event_identity="unresolved", coordinates=coordinates,
                                candidate_edge_us_by_coordinate=[union(x) for x in candidate_by_coordinate],
                                fitting_guard_us=guards, original_review_disposition=event["review_disposition"],
                                source_protection="unavailable", spectral_context="unavailable",
                                hard_event_accepted=False, apply_authorized=False))
        event_count += 1
    require(event_count == receipt["assessed_events"] and seen_candidates == set(seeds),
            "incomplete original population")
    require(set(audits).issubset(range(event_count)), "orphan reassessment")
    totals = Counter(network_files=1, detector_streams=len(detectors),
                     paired_samples=receipt["rows"] * len(detectors),
                     candidate_edges=len(seeds), candidate_groups=event_count,
                     retained_coordinates=retained_count, transition_groups=retained_groups)
    scope = (obs, inventory["subobservation"], inventory["scan"])
    program = programs.get(scope, dict(state="unavailable: producer program metadata absent"))
    native_runs = [axis.intervals(rows) for rows in axis.runs]
    acquired = union(x for run in native_runs for x in run)
    acquired_us = measure(acquired)
    grid = {f"minimum={m}s,extra={g}ms": Counter() for m in MINIMUM_SECONDS for g in EXTRA_GUARD_MS}
    concurrent_sets = defaultdict(list)
    for detector, meta in sorted(detectors.items()):
        eligible = acquired if meta["tune_valid"] else []
        support, count = sets[detector], counts[detector]
        require(count["candidate_edges"] == sum(meta["candidate_counts"]), "per-detector candidate mismatch")
        support["noise_screening_required"] = noise.get(detector, [])
        support["direct"] = union(support["transition"] + support["finite_recovery"])
        support["candidate_inclusive"] = union(support["direct"] + support["candidate_edge"])
        support = {k: intersect(v, eligible) for k, v in support.items()}
        for key in SETS:
            support.setdefault(key, [])
        durations = {k + "_us": measure(support[k]) for k in SETS}
        eligible_us = measure(eligible)
        durations.update(eligible_us=eligible_us, acquired_us=acquired_us,
                         initial_excluded_us=acquired_us - eligible_us,
                         transition_xr_overlap_us=measure(intersect(support.get("transition_x", []),
                                                                   support.get("transition_r", []))),
                         noise_direct_overlap_us=measure(intersect(support["direct"], support["noise_screening_required"])),
                         guard_beyond_direct_us=measure(subtract(support["fitting_guard"], support["direct"])),
                         candidate_beyond_direct_us=measure(subtract(support["candidate_inclusive"], support["direct"])),
                         unresolved_stream_exposure_us=eligible_us if count["unresolved_groups"] else 0,
                         no_candidate_stream_exposure_us=eligible_us if not count["candidate_edges"] else 0)
        totals.update(durations)
        totals["eligible_detector_streams"] += bool(eligible_us)
        totals["initial_excluded_detector_streams"] += not bool(eligible_us)
        totals["health_concern_streams"] += meta["health_review_concern"]
        for category, key in (("candidate", "candidate_edges"), ("transition", "transition_groups"),
                              ("finite_recovery", "finite_recovery_groups"), ("unresolved", "unresolved_groups"),
                              ("no_resolved_excursion", "no_resolved_excursion_groups")):
            totals[category + "_streams"] += count[key] > 0
        for key in ("finite_recovery_coordinates", "finite_recovery_groups", "unresolved_coordinates",
                    "unresolved_groups", "partner_unresolved_with_transition", "extent_unavailable_after_refit",
                    "transition_groups_x", "transition_groups_r", "transition_groups_xr"):
            totals[key] += count[key]
        totals["no_candidate_eligible_streams"] += bool(eligible_us) and not count["candidate_edges"]
        for kind in ("direct", "transition", "finite_recovery", "candidate_inclusive"):
            concurrent_sets[kind].append(support[kind])
        for run_id, run in enumerate(native_runs if eligible_us else []):
            base = qualify(run, support["transition"], support["finite_recovery"], 0)
            for segment in base["segments"]:
                emit(duration_stream, dict(observation=obs, network=network, detector=detector,
                                           native_run=run_id, **segment,
                                           scientific_adequacy="unassessed", processing_scan=None))
            for minimum in MINIMUM_SECONDS:
                for guard in EXTRA_GUARD_MS:
                    value = qualify(run, support["transition"], support["finite_recovery"], minimum * US, guard * 1000)
                    require(value["all_us"] >= value["longest_us"], "qualification ordering failed")
                    row = grid[f"minimum={minimum}s,extra={guard}ms"]
                    row["retained_us"] += value["all_us"]
                    row["excluded_us"] += measure(run) - value["all_us"]
                    row["qualified_intervals"] += sum(s["duration_qualified"] for s in value["segments"])
        emit(detector_stream, dict(observation=obs, network=network, array=array_name(network),
                                   acquisition_scope=list(scope), detector=detector, occurrence=meta["occurrence"],
                                   program_metadata=program, tune_valid=meta["tune_valid"],
                                   peer_eligible=meta["peer_eligible"], health_review_concern=meta["health_review_concern"],
                                   counts=dict(count), durations_us=durations,
                                   eligible_intervals_us=eligible, intervals_us=support,
                                   unresolved_extent_us=None if count["unresolved_groups"] else 0,
                                   scan_losses="unavailable: native-to-PCA-scan relation absent"))
    concurrency_rows = {k: concurrency(v, acquired) for k, v in concurrent_sets.items()}
    for key, histogram in concurrency_rows.items():
        require(sum(int(n) * dt for n, dt in histogram.items()) == totals[key + "_us"],
                "concurrency detector-time does not reconcile")
    return dict(observation=obs, network=network, array=array_name(network),
                acquisition_scope=list(scope), program_metadata=program,
                totals=dict(totals), coordinate_states=dict(coordinate_states), native_sensitivity=grid,
                within_network_concurrency_wall_us=concurrency_rows, timing=axis.metadata,
                inputs=inputs, input_verification_seconds=input_seconds)


def partition(networks, key):
    groups = defaultdict(Counter)
    for row in networks:
        groups[str(key(row))].update(row["totals"])
    return {k: dict(v) for k, v in sorted(groups.items())}


def summarize(networks, deferred, inventory_count):
    totals = Counter()
    states = Counter()
    grid = defaultdict(Counter)
    for row in networks:
        totals.update(row["totals"])
        states.update(row["coordinate_states"])
        for key, value in row["native_sensitivity"].items():
            grid[key].update(value)
    expected = dict(network_files=143, detector_streams=71734, paired_samples=2148911461,
                    candidate_edges=910241, candidate_groups=242014,
                    retained_coordinates=11150, transition_groups=8389,
                    eligible_detector_streams=70024, initial_excluded_detector_streams=1710)
    for key, value in expected.items():
        require(totals[key] == value, f"fixed corpus reconciliation failed: {key}: {totals[key]} != {value}")
    require(len(deferred) == 155 and inventory_count == 298, "deferred population changed")
    require(sum(states.values()) == 2 * totals["candidate_groups"], "coordinate population changed")
    require(totals["eligible_us"] + totals["initial_excluded_us"] == totals["acquired_us"], "baseline partition failed")
    partitions = dict(observation=partition(networks, lambda r: r["observation"]),
                      network=partition(networks, lambda r: r["network"]),
                      array=partition(networks, lambda r: r["array"]),
                      program=partition(networks, lambda r: (
                          r["program_metadata"].get("goal", "unavailable") + " / " +
                          r["program_metadata"].get("program", "unavailable"))))
    require(len(partitions["observation"]) == 13, "observation cohort changed")
    for name, rows in partitions.items():
        summed = Counter()
        for values in rows.values():
            summed.update(values)
        require(summed == totals, "hierarchical totals differ: " + name)
    for row in grid.values():
        require(row["excluded_us"] + row["retained_us"] == totals["eligible_us"], "scenario exposure not conserved")
    for guard in EXTRA_GUARD_MS:
        values = [grid[f"minimum={m}s,extra={guard}ms"]["retained_us"] for m in MINIMUM_SECONDS]
        require(values == sorted(values, reverse=True), "minimum-duration sensitivity ordering failed")
    for minimum in MINIMUM_SECONDS:
        values = [grid[f"minimum={minimum}s,extra={g}ms"]["retained_us"] for g in EXTRA_GUARD_MS]
        require(values == sorted(values, reverse=True), "guard sensitivity ordering failed")
    denominator = totals["eligible_us"]
    return dict(schema="rtc-disturbance-burden-census-v1", scope="fixed preserved corpus; review only",
                totals=dict(totals), coordinate_states=dict(states), partitions=partitions,
                native_sensitivity=dict(grid),
                fractions={k: totals[k + "_us"] / denominator for k in SETS},
                measurement_rates_per_detector_hour={k: totals[k] / (denominator / US / 3600)
                    for k in ("candidate_edges", "candidate_groups", "retained_coordinates", "transition_groups")},
                deferred=dict(entries=len(deferred), exposure_us=None,
                              cause="missing bindings; file counts are not an exposure estimate"),
                scan_scenarios={k: dict(excluded_us=None, retained_us=None, status="unavailable",
                                      cause="verified native-time-to-existing-PCA-scan relation absent")
                    for k in ("full_detector_scan", "longest_segment_per_scan", "all_segments_per_scan",
                              "releveling_opportunity_inventory")},
                interpretation=[
                    "Initial VAL / Tune-valid finite paired input exposure, before RTC disturbance exclusions.",
                    "Exposure is detector-time, not detector sensitivity or scientific information.",
                    "Coordinate measurements and original candidate groups are not physical-event identities.",
                    "Direct support combines final retained transition bounds with available unreassessed finite recoveries.",
                    "Reassessed recovery extent is unavailable in the saved export; earlier fits are not substituted.",
                    "Unresolved-stream exposure is a population denominator, not the duration of unknown disturbances.",
                    "Candidate-inclusive support adds recorded candidate edge cells, not invented event extents.",
                    "Candidate-inclusive and measurement-supported cases are not bounds on true prevalence.",
                    "Fitting guards do not define physical extent or a future Apply operator.",
                    "Native duration sensitivity uses available bounds only and excludes unresolved extent from its claim.",
                    "Transient holes remove support within a level-delimited unit; they do not create level-shift identities.",
                    "Duration qualification, including 50 ms local confirmation, does not establish scientific adequacy.",
                    "No detected candidate is not proof of clean data; known gradual-ramp misses are not corrected statistically.",
                    "Source protection, spectra and cross-network concurrency remain unavailable.",
                    "Full-detector/full-existing-PCA-scan level-0 policy is preserved; this census selects no Apply treatment."])


def make_plots(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    figures = output / "figures"
    figures.mkdir()
    values = defaultdict(list)
    for row in records(output / "detector-accounting.jsonl.gz"):
        if row["durations_us"]["eligible_us"]:
            for key in ("direct", "transition", "candidate_inclusive"):
                values[key].append(row["durations_us"][key + "_us"])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for key, label in (("direct", "Available direct support"), ("transition", "Retained transition bounds"),
                       ("candidate_inclusive", "Direct + candidate edge cells")):
        ranked = np.sort(values[key])[::-1]
        ax.plot(np.arange(1, len(ranked) + 1) / len(ranked) * 100,
                np.cumsum(ranked) / max(1, sum(ranked)) * 100, label=label)
    ax.plot([0, 100], [0, 100], color="0.7", linestyle=":")
    ax.set(xlabel="Eligible detector streams, ranked separately for each curve (%)",
           ylabel="Cumulative share of that support (%)", xlim=(0, 100), ylim=(0, 101),
           title="Is the detected burden concentrated?")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(figures / "concentration.png"); plt.close(fig)

    span, remaining = [], []
    for row in records(output / "native-intervals.jsonl.gz"):
        span.append((row["interval_us"][1] - row["interval_us"][0]) / US)
        remaining.append(row["remaining_us"] / US)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for data, label in ((span, "Boundary-delimited span"), (remaining, "Acquired support after finite-transient exclusions")):
        ordered = np.sort(data)
        ax.plot(np.maximum(ordered, 1 / US), np.arange(1, len(ordered) + 1) / len(ordered) * 100, label=label)
    ax.set(xscale="log", xlabel="Duration (seconds; zero shown at 1 microsecond)",
           ylabel="Cumulative share of intervals (%)", ylim=(0, 101),
           title="Native interval lengths — scientific adequacy unassessed")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(figures / "interval-durations.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for guard in EXTRA_GUARD_MS:
        loss = [100 * summary["native_sensitivity"][f"minimum={m}s,extra={guard}ms"]["excluded_us"] /
                summary["totals"]["eligible_us"] for m in MINIMUM_SECONDS]
        ax.plot(MINIMUM_SECONDS, loss, marker="o", label=f"Additional transition guard: {guard} ms per side")
    ax.axhline(summary["fractions"]["direct"] * 100, color="0.3", linestyle=":", label="Available direct support only")
    ax.set(xlabel="Minimum remaining support per native boundary-delimited interval (s)",
           ylabel="Excluded initial detector-time (%)", xticks=MINIMUM_SECONDS,
           title="Native duration sensitivity — available bounds only\nPCA-scan costs and unresolved extents remain unavailable")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(figures / "native-sensitivity.png"); plt.close(fig)


def write_report(output, summary, networks, elapsed):
    t, f = summary["totals"], summary["fractions"]
    def pct(n):
        return f"{100 * n / t['eligible_us']:.5f}%"
    lines = ["# RTC disturbance burden: preserved native-time census", "",
             "This measures evidence exposed by the fixed method, not the complete physical transient population. "
             "No Apply treatment is selected or executed.", "",
             f"143 network files / 13 observations; {t['eligible_detector_streams']:,} initially eligible detector streams. "
             f"Fixed denominator: **{t['eligible_us'] / US / 3600:,.3f} detector-hours**. "
             "The 155 deferred entries have unavailable exposure.", "",
             "| Accounting quantity | Detector-seconds | Fraction of fixed denominator |", "|---|---:|---:|"]
    for name, key in (("Retained transition bounds", "transition"), ("Bounded finite recoveries", "finite_recovery"),
                      ("Union of available direct support", "direct"), ("Original fitting exclusions", "fitting_guard"),
                      ("Direct support + candidate edge cells (sensitivity)", "candidate_inclusive"),
                      ("Existing noise-screening-required exclusion", "noise_screening_required")):
        lines.append(f"| {name} | {t[key + '_us'] / US:,.3f} | {100 * f[key]:.5f}% |")
    lines += ["", "Rows overlap and must not be added. Noise-screening support is shown separately from initial "
              "producer eligibility; peer-only APT cuts and health-review concerns do not alter the denominator.", "",
              f"Initial producer exclusions account for {t['initial_excluded_us'] / US / 3600:,.3f} detector-hours "
              f"across {t['initial_excluded_detector_streams']:,} detector streams. "
              f"Direct/noise-screen overlap is {t['noise_direct_overlap_us'] / US:,.3f} detector-seconds.", "",
              "| Evidence incidence | Eligible detector streams | Share of eligible streams |", "|---|---:|---:|"]
    for label, key in (("Any candidate", "candidate_streams"), ("Retained transition measurement", "transition_streams"),
                       ("Bounded finite recovery", "finite_recovery_streams"), ("Unresolved coordinate evidence", "unresolved_streams"),
                       ("No detected candidate", "no_candidate_eligible_streams")):
        lines.append(f"| {label} | {t[key]:,} | {100 * t[key] / t['eligible_detector_streams']:.3f}% |")
    lines += ["", f"Final measurements reconcile to **{t['retained_coordinates']:,} coordinates in "
              f"{t['transition_groups']:,} original groups**. These are not unique physical events. "
              f"{t['partner_unresolved_with_transition']:,} groups retain a transition in one coordinate while another remains unresolved.", "",
              f"Streams containing unresolved evidence represent {pct(t['unresolved_stream_exposure_us'])} of initial detector-time. "
              "This is the exposure of that population, not the unknown disturbance duration. "
              f"{t['extent_unavailable_after_refit']:,} coordinate records report a reassessed recovery without its new extent; "
              "these remain unavailable rather than borrowing earlier bounds.", "",
              "## What the hypothetical accounting does and does not establish", "",
              "The direct-support row is an incomplete exclusion benchmark. The native sensitivity grid varies only "
              "minimum remaining duration (1/5/10 s) and extra transition guards (0/50/100 ms per side). "
              "Localized-transient holes remove support without becoming new level-shift boundaries. "
              "Neither these intervals nor the local 50 ms post-level confirmation certify long stable plateaus.", "",
              "**Full detector-scan rejection, longest/all qualifying segments per PCA scan, and releveling opportunity "
              "exposure remain unavailable.** No verified native-time-to-existing-PCA-scan relation is saved. "
              "Acquisition ScanNum and network files are not substituted for processing scans. "
              "The settled level-0 policy remains full affected-detector exclusion for each intersected existing scan.", "",
              "This census can prioritize where to inspect burden, but cannot yet decide whether full-scan exclusion "
              "is inexpensive or whether segmentation/releveling preserves enough exposure to justify its complexity. "
              "The next prerequisite for that comparison is the exact native-cell/PCA-scan association. "
              "Source safety and scientific adequacy remain prerequisites for selecting any treatment.", "",
              "## Concentration, fragmentation and sensitivity", "",
              "![Concentration](figures/concentration.png)", "", "![Native interval durations](figures/interval-durations.png)", "",
              "![Native duration sensitivity](figures/native-sensitivity.png)", "",
              "## Evidence and reproducibility", "",
              "`summary.json` contains exact integer totals, per-observation/network/array/program partitions, rates, "
              "scenario assumptions and unavailable states. `detector-accounting.jsonl.gz` preserves identities and "
              "unioned intervals. `evidence-intervals.jsonl.gz` preserves coordinate origins, native rows, original "
              "groups, offsets and sigma_delta. `native-intervals.jsonl.gz` preserves boundary-delimited durations; "
              "`native-time/` binds every timing cell to the verified raw artifact.", "",
              "Timing uses integration-center support with the inherited provisional uniform averaging assumption, "
              "rounded to integer microseconds relative to each network's first native center. "
              f"Maximum endpoint rounding observed: {max(n['timing']['maximum_endpoint_rounding_us'] for n in networks):.6f} microseconds. "
              "Acquisition gaps and initial invalidity are retained. Within-network concurrency reconciles exactly "
              "to detector-time; cross-network concurrency is unavailable without a verified clock relation.", "",
              "Program labels are independently read, scope-checked producer goal/program headers, not inferred from "
              "directory names. All partitions sum integer numerators and denominators. Amplitudes retain original "
              "dimensionless detector-coordinate meaning; sigma_delta is not a fitted-offset uncertainty.", "",
              "The known injection ramp misses remain explicit; no completeness correction or new threshold is used. "
              "Spectra and source membership/protection remain unavailable. No map product is read. "
              "The report is a census of this selected corpus, not a population estimate for TolTEC operations.", "",
              "Runtime Learn/Consider are unchanged producers. This tool is an engineering evidence consumer and "
              "creates no runtime plan or Apply action. Source and sealed evidence identities are in `run-binding.json`; "
              "this local Python result does not claim Unity/Spack or production performance.", "",
              f"Run time through report preparation: {elapsed:.2f} s; complete stage times and process RSS are in `timing.json`.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path)
    args = parser.parse_args()
    root, output = args.evidence.resolve(strict=True), args.output.resolve()
    require(root != output and root not in output.parents and output not in root.parents,
            "output must be separate from immutable evidence")
    require(not output.exists(), "output exists; preserve prior run")
    repo = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    require(not git("status", "--porcelain"), "run requires a clean committed worktree")
    candidate, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", CLOSURE, candidate]).returncode == 0,
            "census must preserve reviewed closure ancestry")
    started = time.monotonic()
    output.mkdir(parents=True)
    (output / "native-time").mkdir()
    try:
        sealed = verify_seal(root)
        seal_seconds = time.monotonic() - started
        inventory = json.loads((root / "input-inventory.json").read_text())["records"]
        entries = list(records(root / "campaign-01/invocations.jsonl"))
        require(len(entries) == 143 and all(e["status"] == "completed" for e in entries), "incomplete corpus")
        inventory_by_path = {r["path"]: r for r in inventory}
        require(len(inventory_by_path) == len(inventory), "duplicate raw inventory identity")
        selected = [inventory_by_path[e["command"][1]] for e in entries]
        require(len({(r["observation"], r["network"]) for r in selected}) == 143, "duplicate invocation scope")
        deferred = json.loads((root / "campaign-01/deferred-inputs.json").read_text())
        require({r["path"] for r in selected}.isdisjoint(r["path"] for r in deferred) and
                {r["path"] for r in selected + deferred} == set(inventory_by_path), "inventory not partitioned")
        programs, program_sources = program_metadata(args.metadata_root,
            {(r["observation"], r["subobservation"], r["scan"]) for r in selected})
        write_json(output / "program-metadata.json", program_sources)
        write_json(output / "deferred-inputs.json", dict(exposure_us=None, records=deferred))
        binding = dict(work_order="RTC-DISTURBANCE-BURDEN-CENSUS-001", candidate=candidate, tree=tree,
                       implementation_base=CLOSURE, preserved_event_source=SOURCE,
                       preserved_seal_sha256=SEAL, preserved_files_verified=len(sealed),
                       tool_sha256=digest(__file__), evidence=str(root),
                       command=sys.argv, python=sys.version, interpreter=sys.executable,
                       platform=platform.platform(), numpy=np.__version__,
                       timing_unit="integer microseconds relative to first native center",
                       minimum_duration_seconds=MINIMUM_SECONDS, extra_transition_guard_ms=EXTRA_GUARD_MS,
                       scan_association="unavailable; not inferred", runtime_apply=False)
        write_json(output / "run-binding.json", binding)
        networks = []
        with compressed_writer(output / "evidence-intervals.jsonl.gz") as events, \
                compressed_writer(output / "detector-accounting.jsonl.gz") as detectors, \
                compressed_writer(output / "native-intervals.jsonl.gz") as durations:
            for index, entry in enumerate(sorted(entries, key=lambda e: (e["observation"], e["network"]))):
                network = process_network(entry, inventory_by_path[entry["command"][1]], root, output,
                                          events, detectors, durations, programs)
                networks.append(network)
                if (index + 1) % 11 == 0:
                    print(f"Accounted {index + 1}/143 files; {sum(n['totals']['retained_coordinates'] for n in networks)} retained coordinates", flush=True)
        summary = summarize(networks, deferred, len(inventory))
        write_json(output / "networks.json", networks)
        write_json(output / "summary.json", summary)
        analysis_seconds = time.monotonic() - started - seal_seconds
        make_plots(output, summary)
        write_report(output, summary, networks, time.monotonic() - started)
        require(git("rev-parse", "HEAD") == candidate and not git("status", "--porcelain"), "source changed during census")
        write_json(output / "timing.json", dict(wall_seconds=time.monotonic() - started,
            sealed_evidence_verification_seconds=seal_seconds, accounting_and_input_verification_seconds=analysis_seconds,
            original_input_verification_seconds=sum(n["input_verification_seconds"] for n in networks),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            scope="local serial Python, includes hashing/output/plots; no solver or event-analysis rerun; RSS is whole process"))
        write_json(output / "final-status.json", dict(status="PASS", candidate=candidate,
            unexpected_errors=0, source_clean=True, integer_partition_checks="PASS", fixed_population_checks="PASS"))
        names = sorted(p for p in output.rglob("*") if p.is_file())
        (output / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(output)}\n" for p in names))
        print(json.dumps(dict(status="PASS", totals=summary["totals"], fractions=summary["fractions"]), indent=2))
    except Exception as exc:
        write_json(output / "final-status.json", dict(status="FAIL", candidate=candidate, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
