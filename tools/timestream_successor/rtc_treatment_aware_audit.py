#!/usr/bin/env python3
"""Bounded population accounting from preserved RTC Learn exports.

These offline corpus records are not runtime handles or production decisions.
Unweighted native-cell seconds and static APT weight proxies remain distinct.
No FFT, filter design, Apply, interpolation or scan construction occurs here.
"""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np

from analyze_rtc_real_data_audit import describe_coordinate, row_support
from rtc_disturbance_burden import (
    compressed_writer,
    digest,
    emit,
    intersect,
    measure,
    records,
    subtract,
    union,
    write_json,
)


def erode(runs, half):
    """Complete real centered support; never join physically separate runs."""
    if half < 0 or int(half) != half:
        raise ValueError("noninteger/negative footprint")
    return [(int(a + half), int(b - half)) for a, b in runs if b - a > 2 * half]


def rows_for_intervals(intervals, cells):
    result = []
    for a, b in union(intervals):
        first = int(np.searchsorted(cells[:, 2], a))
        last = int(np.searchsorted(cells[:, 3], b, side="right"))
        if first < last:
            result.append((first, last))
    if measure(row_support(result, cells)) != measure(intervals):
        raise ValueError("interval does not consist of exact original cells")
    return result


def bool_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def split_runs(runs, allowed):
    # Keep each physical run separate even when adjacent storage rows straddle
    # an acquisition gap. A union in row space would erase that boundary.
    return [piece for run in runs for piece in intersect([run], allowed)]


def family(frequency):
    if frequency is None:
        return "no_selected_narrow_feature"
    if 9 <= frequency <= 13:
        return "around11"
    if 27 <= frequency <= 32:
        return "around29"
    if frequency > 40:
        return "above40"
    return "other"


def simultaneous(intervals):
    """Sweep native physical interval endpoints, without a sampled grid."""
    events = defaultdict(int)
    for spans in intervals:
        for a, b in union(spans):
            events[a] += 1
            events[b] -= 1
    count = maximum = 0
    last = None
    histogram = defaultdict(int)
    for t, delta in sorted(events.items()):
        if last is not None and count:
            histogram[count] += t - last
        count += delta
        maximum = max(maximum, count)
        last = t
    if count:
        raise ValueError("unclosed time support")
    return dict(maximum=maximum, duration_us_by_count=dict(sorted(histogram.items())))


def fixed_weight(apt_good, sensitivity):
    if (
        apt_good
        and sensitivity is not None
        and math.isfinite(sensitivity)
        and sensitivity > 0
    ):
        return 1 / sensitivity**2
    return None


def sealed_verifier(root, manifest_name, expected_manifest_digest):
    """Bind reused evidence to its recorded accepted seal before consuming it."""
    manifest = root / manifest_name
    if digest(manifest) != expected_manifest_digest:
        raise ValueError("changed accepted evidence seal: " + str(manifest))
    sealed = {
        line[66:].removeprefix("./"): line[:64]
        for line in manifest.read_text().splitlines()
    }
    checked = {manifest_name: expected_manifest_digest}

    def verify(path):
        rel = str(path.relative_to(root))
        actual = digest(path)
        if actual != sealed.get(rel):
            raise ValueError("changed or unsealed evidence: " + rel)
        checked[rel] = actual

    return verify, checked


def main(a):
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    # Exact accepted audit/short-trial/census seals recorded in their handoffs.
    verify, checked = sealed_verifier(
        a.audit,
        "AUDIT_SHA256SUMS",
        "6b631e36bb1532c7bf21c3eb51a2395acd7f30de20495f31ba13861321fa286b",
    )
    verify_short, short_bindings = sealed_verifier(
        a.short,
        "EVIDENCE_SHA256SUMS",
        "c14769f45b56ed4eb1b65bee86b3cbdddb0454d686fcb82bb65518ab50abf8a5",
    )
    verify_burden, burden_bindings = sealed_verifier(
        a.burden,
        "SHA256SUMS",
        "e8156f63a96b21f045e571ff784d1c246994acd83f8e1cf0647e4c4ef33b95b4",
    )
    for name in (
        "analysis-final/summary.json",
        "analysis-final/detectors.jsonl.gz",
        "campaign-01/invocations.jsonl",
    ):
        verify(a.audit / name)
    audit_summary = json.loads((a.audit / "analysis-final/summary.json").read_text())
    burden_path = a.burden / "detector-accounting.jsonl.gz"
    verify_burden(burden_path)
    if digest(burden_path) != audit_summary["bindings"]["burden_accounting_sha256"]:
        raise ValueError("wrong prior transient accounting")
    burden = {
        (r["observation"], r["network"], r["detector"]): r for r in records(burden_path)
    }
    previous = {
        (r["observation"], r["network"], r["detector"]): r
        for r in records(a.audit / "analysis-final/detectors.jsonl.gz")
    }
    # Geometry is reused only for its exact original network/time authority.
    # Other population rows retain unavailable motion/filter applicability.
    geometry = {}
    for case, nw in [("inband", 11), ("quiet", 12)]:
        for path in (
            a.short / "plans" / f"{case}.json",
            a.short / "campaign" / case / "geometry.f64",
            a.short / "campaign" / case / "lowpass-predicted.f64",
        ):
            verify_short(path)
        cfg = json.loads((a.short / "plans" / f"{case}.json").read_text())
        geometry[nw] = (
            cfg,
            np.fromfile(a.short / "campaign" / case / "geometry.f64", "<f8").reshape(
                -1, 4
            ),
        )
    # Reuse the runtime Consider export's exact coefficient response on its
    # unchanged frequency grid; no second filter-response implementation.
    reference_prediction = np.fromfile(
        a.short / "campaign/inband/lowpass-predicted.f64", "<f8"
    ).reshape(2, -1, 5)[0]
    if np.any(reference_prediction[:, 2] <= 0):
        raise ValueError("reference response unavailable at a zero-power bin")
    lowpass_power_response = reference_prediction[:, 3] / reference_prediction[:, 2]
    totals = defaultdict(lambda: defaultdict(float))
    concurrency = defaultdict(list)
    ranking = []
    rows_count = 0
    with compressed_writer(a.output / "detectors.jsonl.gz") as out:
        for entry in records(a.audit / "campaign-01/invocations.jsonl"):
            if entry["exit_status"] or not entry["exact_prior_input_binding"]:
                raise ValueError("failed prior invocation")
            obs, nw = entry["observation"], entry["network"]
            folder = a.audit / f"campaign-01/{obs}-{nw:02d}"
            for name in (
                "receipt.json",
                "spectra.jsonl",
                "spectra.f64",
                "windows.f64",
                "native-time.f64",
            ):
                verify(folder / name)
            receipt = json.loads((folder / "receipt.json").read_text())
            if (
                receipt["VAL_generation"] != 0
                or receipt["Apply"]
                or receipt["original_pair_fingerprint_before"]
                != receipt["original_pair_fingerprint_after"]
            ):
                raise ValueError("prior evidence identity changed")
            metas = list(records(folder / "spectra.jsonl"))
            psds = np.memmap(
                folder / "spectra.f64",
                dtype="<f8",
                mode="r",
                shape=(receipt["channels"], 2, 4, receipt["bins"]),
            )
            windows = np.memmap(
                folder / "windows.f64",
                dtype="<f8",
                mode="r",
                shape=(receipt["window_records"], 14),
            )
            native_cells = a.burden / "native-time" / f"{obs}-{nw:02d}.jsonl.gz"
            verify_burden(native_cells)
            with gzip.open(native_cells, "rt") as stream:
                clock = json.loads(next(stream))
                cells = np.array([json.loads(line) for line in stream], dtype=np.int64)
            if (
                clock["raw_sha256"] != receipt["raw_sha256"]
                or len(cells) != receipt["rows"]
            ):
                raise ValueError("foreign native clock")
            relative_time = np.fromfile(folder / "native-time.f64", "<f8")
            if np.max(abs(relative_time * 1e6 - cells[:, 1])) > 1:
                raise ValueError("native-cell timing disagrees with sealed audit axis")
            dt = receipt["measured_interval"]
            frequency = np.arange(receipt["bins"]) / (receipt["fft_samples"] * dt)
            # A footprint-only ceiling is cheap across the corpus. It does not
            # transplant coefficients, a2000 beam authority or qualification.
            halves = {
                "baseline_307tap": 153,
                "existing_6s_footprint": 153 + int(np.floor(6 / (2 * dt))),
            }
            motion = None
            if obs == 152390 and nw in geometry:
                cfg, g = geometry[nw]
                relative = np.fromfile(folder / "native-time.f64", "<f8")
                if (
                    cfg["audit_receipt"]["sha256"]
                    != checked[str((folder / "receipt.json").relative_to(a.audit))]
                    or len(g) != len(cells)
                    or np.max(abs((g[:, 0] - g[0, 0]) - relative)) > 1e-6
                ):
                    raise ValueError("geometry is not on this exact native axis")
                limit = cfg["science"]["raw_four_sample_speed_ceiling_arcsec_per_sec"]
                motion = dict(
                    unavailable=bool_runs(~np.isfinite(g[:, 1])),
                    below_minimum=bool_runs(np.isfinite(g[:, 1]) & (g[:, 1] < 1)),
                    sampling_excluded=bool_runs(
                        np.isfinite(g[:, 1]) & (g[:, 1] > limit)
                    ),
                    admitted=bool_runs(
                        np.isfinite(g[:, 1])
                        & (g[:, 1] >= 1)
                        & (g[:, 1] <= limit)
                        & (
                            g[:, 1] * 1.05
                            <= cfg["science"]["speed_ceiling_arcsec_per_sec"]
                        )
                    ),
                )
            for d in range(receipt["channels"]):
                key = obs, nw, d
                old, prior = burden[key], previous[key]
                if (
                    old["occurrence"] != metas[2 * d]["occurrence"]
                    or old["occurrence"] != prior["occurrence"]
                ):
                    raise ValueError("detector occurrence mismatch")
                eligible = old["eligible_intervals_us"]
                noise = old["intervals_us"]["noise_screening_required"]
                direct = old["intervals_us"]["direct"]
                retained = subtract(eligible, noise)
                rr = rows_for_intervals(retained, cells)
                info = prior["coordinates"][0]
                selected = info["narrow_fraction"] >= 0.1 and prior["eligible_us"] > 0
                category = (
                    family(info["narrow_frequency_hz"])
                    if selected
                    else "not_x10_selected"
                )
                mm = metas[2 * d]
                ww = windows[
                    mm["window_offset"] : mm["window_offset"] + mm["window_count"]
                ]
                description, activity = describe_coordinate(
                    mm, psds[d, 0], ww, frequency
                )
                if description != info:
                    raise ValueError("changed accepted descriptor computation")
                active = row_support(activity.get(0.1, []), cells) if selected else []
                unknown = selected and info["recurrence"] is None
                features = []
                for region in mm["regions_2"]:
                    (
                        first,
                        last,
                        peak,
                        excess,
                        fraction,
                        contrast,
                        incomplete,
                        clipped,
                    ) = region
                    if (
                        frequency[peak] < 2
                        or fraction is None
                        or fraction < 0.1
                        or contrast is None
                        or contrast < 10
                        or incomplete
                        or (last - first) * (frequency[1] - frequency[0]) > 2
                    ):
                        continue
                    original_band = float(
                        psds[d, 0, 0, first:last].sum() * (frequency[1] - frequency[0])
                    )
                    mathematical_prediction = None
                    if np.array_equal(frequency, reference_prediction[:, 0]):
                        predicted = float(
                            np.dot(
                                psds[d, 0, 0, first:last],
                                lowpass_power_response[first:last],
                            )
                            * (frequency[1] - frequency[0])
                        )
                        mathematical_prediction = dict(
                            original_band_power=original_band,
                            lowpass_band_power=predicted,
                            lowpass_to_original_ratio=predicted / original_band
                            if original_band
                            else None,
                            response_reference="shorter-notch/campaign/inband/lowpass-predicted.f64",
                            applicability="bound experimental a2000 domain"
                            if motion is not None
                            else "coefficient-grid arithmetic only; motion/array applicability unavailable",
                            support="pooled original accepted Learn windows; not a finite-record residual measurement",
                        )
                    features.append(
                        dict(
                            peak_hz=float(frequency[peak]),
                            first_hz=float(frequency[first]),
                            last_hz=float(frequency[last - 1]),
                            bin_span_hz=(last - first)
                            * float(frequency[1] - frequency[0]),
                            positive_excess_power=excess,
                            stored_fraction=fraction,
                            contrast=contrast,
                            family=family(float(frequency[peak])),
                            uncertainty="window/bin resolution; intrinsic width and drift not resolved by this descriptor",
                            lowpass_prediction=mathematical_prediction,
                        )
                    )
                weight = fixed_weight(prior["apt_good"], prior["sens"])
                group = (obs, old["program_metadata"]["goal"], nw, prior["array"])
                r = dict(
                    observation=obs,
                    network=nw,
                    detector=d,
                    array=prior["array"],
                    occurrence=old["occurrence"],
                    program=group[1],
                    raw_sha256=receipt["raw_sha256"],
                    audit_receipt_sha256=checked[
                        str((folder / "receipt.json").relative_to(a.audit))
                    ],
                    initial_VAL_generation=0,
                    stage="original-native-audit",
                    offline_corpus_context=True,
                    original_pair_fingerprint=receipt[
                        "original_pair_fingerprint_before"
                    ],
                    source_protection="unknown-retained",
                    scan_binding=None,
                    coordinates=prior["coordinates"],
                    features_x=features,
                    selected_x10=selected,
                    family=category,
                    weight=weight,
                    weight_identity="exact-compact-APT:finite-positive-sens:flag=flag2=0:static-inverse-square",
                    prior_health_concern=prior["prior_health_concern"],
                    independent_selection=prior["independent_selection"],
                    existing_exclusions=dict(
                        producer_paired_invalid_us=old["durations_us"][
                            "initial_excluded_us"
                        ],
                        screening_us=measure(intersect(eligible, noise)),
                        conditional_direct_us=measure(intersect(eligible, direct)),
                        screening_direct_overlap_us=measure(
                            intersect(intersect(eligible, noise), direct)
                        ),
                    ),
                    prefilter_runs=rr,
                    eligible_us=measure(eligible),
                    baseline_before_filter_us=measure(retained),
                    activity_unknown=unknown,
                    activity_intervals_us=active,
                    active_lower_us=measure(intersect(retained, active)),
                    active_upper_us=measure(retained)
                    if unknown
                    else measure(intersect(retained, active)),
                    footprint_only_upper_bound_us={
                        name: measure(row_support(erode(rr, h), cells))
                        for name, h in halves.items()
                    },
                    bound_chain=None,
                    candidate_coverage="unavailable: measured line width/drift and source-response domain not qualified",
                    admitted_recovery=None,
                    production=False,
                )
                if motion is not None:
                    mr = split_runs(rr, motion["admitted"])
                    br = erode(mr, halves["baseline_307tap"])
                    nr = erode(mr, halves["existing_6s_footprint"])
                    base, notch = row_support(br, cells), row_support(nr, cells)
                    r["bound_chain"] = dict(
                        domain="152390:a2000:exact-native-AST:F2:307tap:experimental-only",
                        input_runs=mr,
                        baseline_runs=br,
                        finite6_runs=nr,
                        baseline_us=measure(base),
                        finite6_potential_us=measure(notch),
                        additional_footprint_loss_us=measure(base) - measure(notch),
                        baseline_active_lower_us=measure(intersect(base, active)),
                        baseline_active_upper_us=measure(base)
                        if unknown
                        else measure(intersect(base, active)),
                        finite6_active_lower_us=measure(intersect(notch, active)),
                        finite6_active_upper_us=measure(notch)
                        if unknown
                        else measure(intersect(notch, active)),
                        motion_causes={
                            cause: measure(
                                row_support(
                                    split_runs(
                                        rows_for_intervals(eligible, cells), spans
                                    ),
                                    cells,
                                )
                            )
                            for cause, spans in motion.items()
                            if cause != "admitted"
                        },
                        motion_screening_overlap_us=measure(
                            intersect(
                                noise,
                                row_support(
                                    subtract(
                                        rows_for_intervals(eligible, cells),
                                        motion["admitted"],
                                    ),
                                    cells,
                                ),
                            )
                        ),
                        direct_intersection_baseline_us=measure(
                            intersect(base, direct)
                        ),
                        direct_intersection_finite6_us=measure(
                            intersect(notch, direct)
                        ),
                        source_response_qualification="unavailable beyond exact prior source fixtures",
                        residual_allowance=None,
                        cleaned_noise_reference=None,
                    )
                total = totals[group]
                total["occurrences"] += 1
                total["eligible_occurrences"] += bool(r["eligible_us"])
                for name in ("eligible_us", "baseline_before_filter_us"):
                    total[name] += r[name]
                if weight is not None:
                    total["weighted_coverage_us"] += r["baseline_before_filter_us"]
                    total["reference_weight_seconds"] += (
                        r["baseline_before_filter_us"] / 1e6 * weight
                    )
                if selected:
                    total[category + "_occurrences"] += 1
                    total[category + "_observation_rejection_us"] += r[
                        "baseline_before_filter_us"
                    ]
                    total[category + "_active_lower_us"] += r["active_lower_us"]
                    total[category + "_active_upper_us"] += r["active_upper_us"]
                    if weight is not None:
                        total[category + "_reference_weight_seconds"] += (
                            r["baseline_before_filter_us"] / 1e6 * weight
                        )
                    concurrency[group + (category,)].append(intersect(retained, active))
                if r["bound_chain"]:
                    b = r["bound_chain"]
                    total["bound_baseline_us"] += b["baseline_us"]
                    total["bound_weighted_coverage_us"] += (
                        b["baseline_us"] if weight is not None else 0
                    )
                    total["bound_reference_weight_seconds"] += (
                        b["baseline_us"] / 1e6 * (weight or 0)
                    )
                    if selected:
                        for name in (
                            "baseline_us",
                            "finite6_potential_us",
                            "baseline_active_lower_us",
                            "baseline_active_upper_us",
                            "finite6_active_lower_us",
                            "finite6_active_upper_us",
                        ):
                            total[category + "_bound_" + name] += b[name]
                            total[category + "_bound_weight_" + name] += (
                                b[name] / 1e6 * (weight or 0)
                            )
                if selected:
                    ranking.append(
                        dict(
                            observation=obs,
                            network=nw,
                            detector=d,
                            array=prior["array"],
                            family=category,
                            weighted=weight is not None,
                            apt_good=prior["apt_good"],
                            weight_seconds=(weight or 0)
                            * r["baseline_before_filter_us"]
                            / 1e6,
                            bound_finite6_weight_seconds=None
                            if not r["bound_chain"] or weight is None
                            else weight
                            * r["bound_chain"]["finite6_potential_us"]
                            / 1e6,
                            baseline_us=r["baseline_before_filter_us"],
                            bound=r["bound_chain"],
                            spectral=info,
                            prior_health_concern=r["prior_health_concern"],
                        )
                    )
                emit(out, r)
                rows_count += 1
            print(f"accounted {obs}-{nw:02d}", flush=True)
    write_json(
        a.output / "groups.json",
        [
            dict(observation=k[0], program=k[1], network=k[2], array=k[3], **v)
            for k, v in sorted(totals.items())
        ],
    )
    write_json(
        a.output / "simultaneous.json",
        [
            dict(
                observation=k[0],
                program=k[1],
                network=k[2],
                array=k[3],
                family=k[4],
                **simultaneous(v),
            )
            for k, v in sorted(concurrency.items())
        ],
    )
    ranking.sort(key=lambda r: -(r["bound_finite6_weight_seconds"] or 0))
    write_json(a.output / "ranking.json", ranking)
    write_json(
        a.output / "receipt.json",
        dict(
            rows=rows_count,
            checked_audit_files=checked,
            burden_accounting_sha256=digest(burden_path),
            wall_seconds=time.monotonic() - started,
            short_evidence_bindings=short_bindings,
            burden_evidence_bindings=burden_bindings,
            original_estimators_unchanged=True,
            filtering_executed=False,
            assumptions="x10 exploratory screen unchanged; six-second footprint potential is not candidate coverage; actual AST/filter domain only where exact prior geometry binds; no pooled cross-array sensitivity",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "burden", "short", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
