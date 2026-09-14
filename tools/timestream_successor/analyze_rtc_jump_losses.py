"""Read the sealed reassessment census; no detector data or policy is changed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


FIT = ("none", "not_attempted", "insufficient_samples", "additional_candidate",
       "nonfinite", "zero_scale", "rank_deficient", "iteration_limit")
CONSISTENCY = ("primary_not_passed", "short_fit_unavailable", "arithmetic_nonfinite",
               "short_below_threshold", "sign_disagreement", "offset_disagreement", "passes")
TRANSITION = ("not_requested", "measured", "background_unavailable", "nonfinite",
              "pre_confirmation_missing", "post_confirmation_missing", "ambiguous_reference",
              "invalid_transition_support", "competing_exclusion", "support_geometry_unavailable")
DISPOSITION = ("retained_without_refit", "original_unavailable", "refit_unavailable",
               "consistency_failed", "confirmed_recovery", "transition_unavailable",
               "remaining_overlap", "retained_after_refit")


def rows(sides):
    return [sum(b - a for a, b in side) for side in sides]


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def diagnose(row, paired_overlap):
    """Refine a saved terminal cause; separately retain other evaluated facts.

    The production availability predicate joins primary/short failures with OR.
    If both insufficient support and a numerical failure occur, support takes
    precedence solely for mutually exclusive reporting. Both facts are retained.
    Defaults of stages that did not run are never counted as evaluated failures.
    """
    terminal = DISPOSITION[row["diagnostic_cause"]]
    conditions = []
    detail = None
    if row["diagnostic_cause"] in (0, 1):
        return terminal, conditions, detail
    counts = {kind: rows(row[f"new_{kind}_rows"]) for kind in ("primary", "short")}
    for kind, count in counts.items():
        for side, n in zip(("pre", "post"), count):
            if n < 64:
                conditions.append(f"{kind}_{side}_insufficient_support")
    for name in ("primary_cubic", "primary_offset", "short_offset"):
        cause = row[name]["cause"]
        if cause not in (0, 1):
            conditions.append(f"{name}:{FIT[cause]}")
    for name in ("frozen_primary_scale", "frozen_short_scale"):
        if row[name] is None or row[name] <= 0:
            conditions.append(f"{name}:unavailable")
    if row["primary_available"] and row["short_available"]:
        if row["consistency_cause"] != 6:
            conditions.append("consistency:" + CONSISTENCY[row["consistency_cause"]])
    if row["recovery_cause"] == 0:
        conditions.append("confirmed_recovery_changes_persistent_interpretation")
    if row["remeasure_requested"] and not row["transition"]["available"]:
        conditions.append("transition:" + TRANSITION[row["transition"]["cause"]])
    if paired_overlap:
        conditions.append("paired_measured_support_overlaps_successor_fit")
    if terminal == "refit_unavailable":
        terminal = ("insufficient_fitting_support" if any(n < 64 for v in counts.values() for n in v)
                    else "fit_failure")
        detail = dict(primary_support_cause=FIT[row["primary_support_cause"]],
                      primary_cubic_cause=FIT[row["primary_cubic"]["cause"]],
                      primary_offset_cause=FIT[row["primary_offset"]["cause"]],
                      short_cause=row["short_cause"], short_offset_cause=FIT[row["short_offset"]["cause"]],
                      support_counts=counts)
    elif terminal == "consistency_failed":
        terminal = {0: "primary_amplitude_recheck_failed", 2: "comparison_nonfinite",
                    3: "short_amplitude_recheck_failed", 4: "short_long_sign_inconsistent",
                    5: "short_long_offset_inconsistent"}[row["consistency_cause"]]
    elif terminal == "transition_unavailable":
        detail = TRANSITION[row["transition"]["cause"]]
    return terminal, conditions, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control, output = args.control_root.resolve(), args.output.resolve()
    if output == control or control in output.parents:
        parser.error("outputs must be outside the immutable control root")
    output.mkdir(parents=True, exist_ok=True)
    totals, terminal_counts, conditions, pair_counts, signatures = (Counter() for _ in range(5))
    detectors, observations = defaultdict(Counter), defaultdict(Counter)
    health = {}
    used = {}
    with (output / "coordinate-losses.jsonl").open("w") as coords, (output / "paired-groups.jsonl").open("w") as groups:
        for folder in sorted((control / "campaign-01").iterdir()):
            if not folder.is_dir():
                continue
            obs, network = map(int, folder.name.split("-"))
            totals["input_files"] += 1
            for name in ("detectors.jsonl", "jump-reassessment.jsonl", "jump-transitions.jsonl"):
                used[str((folder / name).relative_to(control))] = digest(folder / name)
            dmeta = {d["detector"]: d for d in records(folder / "detectors.jsonl")}
            totals["all_channel_streams"] += len(dmeta)
            totals["health_concern_channel_streams"] += sum(d["health_review_concern"] for d in dmeta.values())
            for d, info in dmeta.items():
                health[(obs, network, d)] = {k: info[k] for k in (
                    "health_review_concern", "health_assessment_available", "coordinate_concern",
                    "concerning_blocks", "complete_blocks", "tune_valid", "occurrence")}
            audit = {a["event"]: a for a in records(folder / "jump-reassessment.jsonl")}
            for transition in records(folder / "jump-transitions.jsonl"):
                original = [c for c, t in enumerate(transition["coordinates"]) if t["available"]]
                a = audit.get(transition["event"])
                if a and a["refit_requested"]:
                    totals["requested_paired_refits"] += 1
                if not original:
                    if a:
                        totals["new_partner_measurements"] += sum(r["diagnostic_cause"] == 7 for r in a["coordinates"])
                    continue
                assert a is not None and a["detector"] == transition["detector"]
                key = (obs, network, a["detector"])
                totals["original_paired_groups"] += 1
                pair_overlap = any(r["new_primary_overlap"] or r["new_short_overlap"] for r in a["coordinates"])
                reasons = []
                remaining = []
                for c, r in enumerate(a["coordinates"]):
                    measured = r["diagnostic_cause"] in (0, 7)
                    if measured:
                        remaining.append(c)
                    if c not in original:
                        totals["new_partner_measurements"] += measured
                        continue
                    terminal, extra, detail = diagnose(r, pair_overlap)
                    reasons.append(terminal)
                    terminal_counts[terminal] += 1
                    totals["original_coordinate_measurements"] += 1
                    totals["health_concern_original_coordinates"] += dmeta[a["detector"]]["health_review_concern"]
                    totals["health_unavailable_original_coordinates"] += not dmeta[a["detector"]]["health_assessment_available"][c]
                    detectors[key]["original_coordinates"] += 1
                    observations[obs]["original_coordinates"] += 1
                    if a["refit_requested"]:
                        totals["original_coordinates_in_requested_groups"] += 1
                    lost = not measured
                    totals["unresolved_coordinates"] += lost
                    detectors[key]["unresolved_coordinates"] += lost
                    observations[obs]["unresolved_coordinates"] += lost
                    detectors[key][terminal] += 1
                    if lost:
                        conditions.update(extra)
                        if dmeta[a["detector"]]["health_review_concern"]:
                            totals["unresolved_coordinates_health_concern"] += 1
                    record = dict(observation=obs, network=network, detector=a["detector"],
                                  event=a["event"], coordinate=("x", "r")[c], terminal=terminal,
                                  original_terminal=DISPOSITION[r["diagnostic_cause"]],
                                  additional_evaluated_conditions=extra, detail=detail,
                                  health_review_concern=dmeta[a["detector"]]["health_review_concern"])
                    coords.write(json.dumps(record, allow_nan=False) + "\n")
                retained_original = set(original) & set(remaining)
                if not a["refit_requested"]:
                    pair_status = "retained_without_refit"
                elif len(retained_original) == len(original):
                    pair_status = "refit_all_original_coordinates_retained"
                elif retained_original:
                    pair_status = "refit_partial_original_loss"
                else:
                    pair_status = "refit_all_original_coordinates_lost"
                pair_counts[pair_status] += 1
                signatures[" | ".join(sorted(set(reasons)))] += 1
                lost_pair = len(retained_original) < len(original)
                detectors[key]["original_groups"] += 1
                detectors[key]["groups_with_any_original_loss"] += lost_pair
                observations[obs]["original_groups"] += 1
                observations[obs]["groups_with_any_original_loss"] += lost_pair
                totals["groups_with_any_remaining_bound"] += bool(remaining)
                groups.write(json.dumps(dict(observation=obs, network=network, detector=a["detector"],
                    event=a["event"], refit_requested=a["refit_requested"], original_coordinates=original,
                    remaining_coordinates=remaining, original_coordinate_terminal_reasons=reasons,
                    status=pair_status, health_review_concern=dmeta[a["detector"]]["health_review_concern"])) + "\n")
    old = json.loads((control / "reconciliation.json").read_text())["totals"]
    assert totals["input_files"] == 143
    assert totals["original_coordinate_measurements"] == old["original_measured_coordinates"] == 11830
    assert totals["requested_paired_refits"] == old["refit_requested_groups"] == 3226
    assert totals["unresolved_coordinates"] == 2758
    assert totals["original_paired_groups"] == old["groups_with_original_bound"] == 8980
    assert totals["groups_with_any_remaining_bound"] == old["groups_with_remaining_numerical_bound"] == 6758
    assert totals["new_partner_measurements"] == 28
    totals.setdefault("unresolved_coordinates_health_concern", 0)
    assert sum(pair_counts.values()) == 8980 and sum(terminal_counts.values()) == 11830
    ranked = [dict(observation=k[0], network=k[1], detector=k[2], counts=dict(v), health=health[k])
              for k, v in sorted(detectors.items(), key=lambda kv: (-kv[1]["unresolved_coordinates"], kv[0]))]
    result = dict(control_root=str(control), control_source="aaa86ea1b006cb11aa740adce2429375b13e5026",
        totals=dict(totals), mutually_exclusive_coordinate_terminal_reasons=dict(terminal_counts),
        evaluated_conditions_among_unresolved=dict(conditions), mutually_exclusive_paired_group_counts=dict(pair_counts),
        paired_terminal_reason_signatures=dict(signatures), detectors=ranked,
        observations=[dict(observation=k, counts=dict(v)) for k,v in sorted(observations.items())],
        concentration={f"top_{n}_detectors": sum(d["counts"]["unresolved_coordinates"] for d in ranked[:n]) for n in (1, 5, 10, 20)},
        consumed_control_file_sha256=used,
        caveats=["Paired groups retain candidate membership; they are not accepted physical-event identities.",
                 "Confirmed recovery changes persistent-jump interpretation; it does not establish that no disturbance occurred.",
                 "Support takes reporting precedence within the combined fit-availability gate; simultaneous numerical failures remain recorded.",
                 "Unevaluated decision defaults are not failures. Additional conditions overlap and must not be summed as disjoint populations.",
                 "Health concerns are the saved observation-time review signal, not a new detector rejection rule."])
    (output / "census-losses.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("totals", "mutually_exclusive_coordinate_terminal_reasons",
        "mutually_exclusive_paired_group_counts", "concentration")}, indent=2))


if __name__ == "__main__":
    main()
