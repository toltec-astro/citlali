"""Audit fixed targeted replays and report truth error with explicit denominators."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import yaml


VERSIONS = ("original", "current", "truth_fit")


def expand(sides):
    return [np.array([i for a, b in side for i in range(a, b)], dtype=int) for side in sides]


def mask_set(ranges):
    return {i for a, b in ranges for i in range(a, b)}


def audit_fit(fit, sides, samples, origin, time_scale, coordinate, unmodified=False):
    if not fit["available"]:
        return 0
    rows = np.concatenate(sides)
    assert all(len(s) >= 64 for s in sides)
    col = (8 + coordinate) if unmodified else (4 + 2 * coordinate)
    values = samples[rows, col]
    t = (samples[rows, 1] - origin) / time_scale
    side = np.concatenate([np.zeros(len(sides[0])), np.ones(len(sides[1]))])
    model = np.polynomial.polynomial.polyval(t, fit["coefficients"]) + side * fit["offset"]
    z = np.abs((values - model) / fit["scale"])
    loss = np.where(z <= 1.345, .5 * z**2, 1.345 * (z - .5 * 1.345)).sum()
    assert math.isclose(loss, fit["huber_loss"], rel_tol=3e-6, abs_tol=2e-5), (loss, fit["huber_loss"])
    return 1


def boundary_metrics(t, d, samples):
    if not t["available"]:
        return dict(available=False, cause=t["cause"])
    lo, hi = t["rows"]
    assert 0 < lo < hi < len(samples)
    assert t["begin"] == samples[lo - 1, 3] and t["end"] == samples[hi, 2]
    out = dict(available=True, cause=t["cause"], rows=t["rows"], begin=t["begin"], end=t["end"])
    if not d["jump_injected"] and not d["pulse_injected"]:
        out["truth_boundary_error"] = None
        return out
    a, b = d["truth_cells"]
    # Sample coordinates are cell-boundary positions: a sharp midpoint is a+1/2.
    def position(time):
        i = int(np.searchsorted(samples[:, 2], time, side="right") - 1)
        i = min(max(i, 0), len(samples) - 1)
        return i + (time - samples[i, 2]) / (samples[i, 3] - samples[i, 2])
    begin, end = d["truth_begin"], d["truth_end"]
    out.update(boundary_error_seconds=[t["begin"] - begin, t["end"] - end],
               boundary_error_samples=[lo - position(begin), hi - position(end)],
               affected_cell_edge_error=[lo - a, hi - b],
               physical_support_covered=t["begin"] <= begin and t["end"] >= end,
               affected_cells_covered=lo <= a and hi >= b,
               excess_seconds=max(0., begin - t["begin"]) + max(0., t["end"] - end),
               excess_cells=max(0, a-lo) + max(0, hi-b))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control_root", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.resolve() == args.control_root.resolve() or args.control_root.resolve() in args.output.resolve().parents:
        p.error("output must be outside the sealed control")
    args.output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    all_docs, metrics = [], []
    for index in range(3):
        with (args.replay / f"{index:02d}-diagnostic.out").open() as stream:
            docs = list(yaml.safe_load_all(stream))
        expected = {d["trial"]: d for d in map(json.loads, (args.control_root / f"injection-final-{index:02d}.jsonl").read_text().splitlines())}
        assert len(docs) == (8 if index == 0 else 6)
        for d in docs:
            samples = np.array(d["samples"], dtype=float)
            assert np.isfinite(samples).all() and np.all(samples[:, 3] > samples[:, 2])
            assert np.all(samples[:, (5, 7)] == 1)  # fixed prior fixture validity, not a new admission rule
            if index:
                folder = "152385-04" if index == 1 else "152430-08"
                raw = np.loadtxt(args.control_root / "campaign-01" / folder / "injection-background.txt")
                assert np.array_equal(samples[:, :4], raw[:, :4])
                assert np.array_equal(samples[:, 8:], raw[:, (4, 6)])
                assert np.array_equal(samples[:, (5, 7)], raw[:, (5, 7)])
            center, last = d["truth_cells"]
            expected_values = samples[:, 8:].copy()
            for c in range(2):
                sigma = d["sigma_delta_unmodified"][c]
                if d["jump_injected"]:
                    n = last - center
                    ramp = np.minimum(1., (np.arange(len(samples)-center)+.5)/n) if d["trial"].startswith("finite_") else np.r_[.5, np.ones(len(samples)-center-1)]
                    expected_values[center:, c] += (20 if c == 0 else -20) * sigma * ramp
                if d["trial"] in ("sharp_plus_neighbor_spike", "spike_only_control"):
                    expected_values[center + (25 if d["jump_injected"] else 0), c] += 12 * sigma
                if d["pulse_injected"]:
                    f = np.ones(last-center); f[[0,-1]] = .5
                    expected_values[center:last, c] += (20 if c == 0 else -20) * sigma * f
            # Independent NumPy ramp arithmetic differs by one ULP from the
            # source-bound C++ expression on a few cells. Legacy replay is also
            # byte-exact; this two-ULP formula check is numerical, not scientific.
            ulps = np.abs(samples[:, (4, 6)] - expected_values) / np.abs(np.spacing(expected_values))
            assert np.all(ulps <= 2), (d["trial"], float(ulps.max()))
            counts["maximum_formula_difference_ulps"] = max(counts["maximum_formula_difference_ulps"], int(ulps.max()))
            counts["sample_value_checks"] += len(samples) * 2
            assert d["hard_event_accepted"] is False and d["apply_authorized"] is False
            counts["explicit_anchor_parity_checks"] += d["explicit_anchor_parity_checks"]
            if d["trial"] in expected:
                old = expected[d["trial"]]
                assert d["background_identity"] == old["background_identity"]
                assert sum(d["candidate_edges_at_truth"]) == old["candidate_edges_touching_truth_cells"]
                assert d["all_candidate_edges"] == old["all_candidate_edges"]
                if d["known_location_diagnostic_only"]:
                    assert not old["groups"]
                else:
                    assert [g["event"] for g in d["groups"]] == [g["event"] for g in old["groups"]]
                    for group, previous in zip(d["groups"], old["groups"], strict=True):
                        for coord, original in zip(group["coordinates"], previous["coordinates"], strict=True):
                            for version, key in (("original", "before"), ("current", "after")):
                                v = coord[version]
                                bound = [v["actual_transition"]["begin"], v["actual_transition"]["end"]] if v["retained"] else None
                                assert bound == original[key], (d["trial"], version)
                counts["legacy_trial_matches"] += 1
            for group in d["groups"]:
                truth_mask = mask_set(group["truth_exclusions"])
                # Reconstruct the established candidate guards from original
                # native midpoints; retain all candidates outside scored truth.
                expected_mask = set(range(center, last))
                for candidate in d["all_candidates"]:
                    a, b = candidate["earlier"], candidate["later"]
                    if a < last and b >= center:
                        continue
                    middle = samples[a, 1] + (samples[b, 1] - samples[a, 1]) / 2
                    first = min(a, int(np.searchsorted(samples[:, 3], middle - .05, side="right")))
                    past = max(b + 1, int(np.searchsorted(samples[:, 2], middle + .05, side="left")))
                    expected_mask.update(range(first, past))
                assert truth_mask == expected_mask
                revised_mask = mask_set(group["inferred_reassessment_exclusions"])
                for c, coord in enumerate(group["coordinates"]):
                    counts["coordinate_comparisons"] += 1
                    for name in VERSIONS:
                        v = coord[name]
                        primary, short = expand(v["primary_rows"]), expand(v["short_rows"])
                        if name == "current" and group["refit_requested"]:
                            for kind, new in (("primary_rows", primary), ("short_rows", short)):
                                previous = expand(coord["original"][kind])
                                assert all(set(n) == set(o) - revised_mask for n,o in zip(new,previous))
                        if name == "truth_fit":
                            for flank, sides in ((2, primary), (1, short)):
                                a, b = group["trial_exclusion"]
                                low, high = samples[a, 2]-flank, samples[b-1, 3]+flank
                                for side, rows in enumerate(sides):
                                    desired = {i for i in range(len(samples)) if samples[i,2] >= low and samples[i,3] <= high
                                               and i not in truth_mask and (i < center if side == 0 else i >= last)}
                                    assert set(rows) == desired
                        for fit, sides, base in (("primary_offset", primary, False), ("short_offset", short, False),
                                                 ("matched_uninjected_primary", primary, True), ("matched_uninjected_short", short, True)):
                            counts["fit_loss_audits"] += audit_fit(v[fit], sides, samples, group["origin"], group["time_scale"], c, base)
                        # These scales have separate accepted roles; none is a parameter SE.
                        assert v["sigma_delta_comparison_tolerance"] == coord["original"]["sigma_delta_comparison_tolerance"]
                        if v["primary_offset"]["available"]:
                            assert v["primary_offset"]["scale"] == coord["original"]["primary_pre_scale_fit"]["scale"]
                        if v["short_offset"]["available"]:
                            assert v["short_offset"]["scale"] == coord["original"]["short_pre_scale_fit"]["scale"]
                        fit, base = v["primary_offset"], v["matched_uninjected_primary"]
                        injected = d["injected_offset"][c]
                        raw_error = fit["offset"]-injected if fit["available"] else None
                        increment_error = fit["offset"]-base["offset"]-injected if fit["available"] and base["available"] else None
                        row = dict(background=index, background_identity=d["background_identity"], trial=d["trial"],
                            event=group["event"], coordinate=coord["coordinate"], version=name,
                            jump_injected=d["jump_injected"], pulse_injected=d["pulse_injected"],
                            forced_location=group["forced_location"], candidate_detected=d["candidate_edges_at_truth"][c]>0,
                            end_to_end_retained=(not group["forced_location"] and v["retained"]),
                            diagnostic_retained=v["retained"], primary_available=v["primary_available"], short_available=v["short_available"],
                            offset_fit_available=fit["available"], primary_counts=list(map(len, primary)), short_counts=list(map(len, short)),
                            injected_offset=injected, fitted_offset=fit["offset"], fitted_short_offset=v["short_offset"]["offset"],
                            matched_uninjected_offset=base["offset"], raw_fit_minus_injected=raw_error,
                            raw_fit_minus_injected_sigma=raw_error/d["sigma_delta_unmodified"][c] if raw_error is not None else None,
                            recovered_injection_error=increment_error,
                            recovered_injection_error_sigma=increment_error/d["sigma_delta_unmodified"][c] if increment_error is not None else None,
                            consistency_cause=v["consistency_cause"], first_decisive_stage=v["first_decisive_stage"],
                            recovery_cause=v["recovery"]["cause"], primary_fit_cause=fit["cause"], short_fit_cause=v["short_offset"]["cause"],
                            actual_transition=boundary_metrics(v["actual_transition"], d, samples),
                            ungated_transition_probe=boundary_metrics(v["ungated_transition_probe"], d, samples),
                            additional_evaluated=dict(primary_overlap=v["primary_overlap"], short_overlap=v["short_overlap"],
                                confirmed_recovery=v["recovery"]["cause"] == 0))
                        metrics.append(row)
            all_docs.append(d)
    assert counts["legacy_trial_matches"] == 18 and len(all_docs) == 20
    summary = {}
    for name in VERSIONS:
        rs = [r for r in metrics if r["version"] == name and r["jump_injected"]]
        assert len(rs) == 24
        available = [r for r in rs if not r["forced_location"] and r["offset_fit_available"]]
        summary[name] = dict(jump_coordinate_cases=24, end_to_end_candidate_detected=sum(r["candidate_detected"] for r in rs),
            end_to_end_retained=sum(r["end_to_end_retained"] for r in rs),
            candidate_associated_offset_fits=len(available),
            median_absolute_fit_minus_injected_sigma=float(np.median([abs(r["raw_fit_minus_injected_sigma"]) for r in available])) if available else None,
            forced_location_coordinate_cases=sum(r["forced_location"] for r in rs),
            forced_location_diagnostic_retained=sum(r["forced_location"] and r["diagnostic_retained"] for r in rs))
    result = dict(status="PASS", audits=dict(counts), injection_summary=summary, records=metrics,
        definitions=dict(raw_fit_minus_injected="Includes any offset/background-model bias already present in the real background; not total truth error for an unknown real event.",
            recovered_injection_error="Fitted injected offset minus same-support/frozen-scale uninjected fit minus known injected offset; describes recovery of the added signal only.",
            boundary_error_samples="Signed bound minus known physical boundary on the native integration-cell axis; midpoint sharp truth has half-cell position. Integer affected-cell errors are separate.",
            truth_fit="Test-only exclusion using supplied truth. Current empirical decisions and remaining-overlap rule are still reported; a fit or ungated probe is not retained admission.",
            denominators="All 12 injected jumps /24 coordinate cases remain included. Forced-location probes never repair end-to-end misses. No-jump/pulse controls and real unmodified inputs stay in full records."))
    (args.output / "truth-analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output / "replays.json").write_text(json.dumps(all_docs,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],audits=result["audits"],summary=summary),indent=2))


if __name__ == "__main__":
    main()
