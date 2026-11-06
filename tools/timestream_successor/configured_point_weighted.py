"""Sequence the configured empirical POINT owners; no timestream arithmetic.

Original completed CAL is the parent of every cycle. Health is learned once,
and source models replace their predecessors. A stopped source selector is an
explicit outcome, never a convergence or ten-pass success claim.
"""

from collections import Counter
import hashlib
import re
from pathlib import Path

import yaml

from configured_point_types import PreparedPoint

ARRAYS = ("a1100", "a1400", "a2000")
STATES = ("prior_unavailable", "retained", "weak_response", "sign_mismatch", "unresolved")


def health_summary(path):
    return detector_summary(yaml.safe_load(Path(path).read_text()), "citlali-point-optical-health-v2")


def detector_summary(health, schema):
    if health["schema"] != schema:
        raise ValueError("unexpected optical health publication")
    seed_identity = health.get("bootstrap_seed_identity", "")
    if (
        health.get("bootstrap_seed_policy") != "point-optical-learning-input-seed-v1"
        or not isinstance(seed_identity, str)
        or len(seed_identity) != 71
        or not seed_identity.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in seed_identity[7:])
    ):
        raise ValueError("unexpected optical bootstrap seed binding")
    summary = dict(
        schema="citlali-point-detector-health-summary-v1",
        min_relative_response=health["min_relative_response"],
        method=health["method"],
        bootstrap_seed_policy=health["bootstrap_seed_policy"],
        bootstrap_seed_identity=seed_identity,
        uncertainty_limitations=health["uncertainty_limitations"],
        arrays={},
        exceptions=[],
    )
    totals = {array: Counter({state: 0 for state in STATES}) for array in ARRAYS}
    seen_networks, seen_detectors = set(), set()
    for network in health["networks"]:
        if (
            type(network["array"]) is not int
            or not 0 <= network["array"] < 3
            or network["network"] in seen_networks
        ):
            raise ValueError("optical health array/network domain differs")
        seen_networks.add(network["network"])
        array = ARRAYS[network["array"]]
        for detector in network["detectors"]:
            if not detector["detector_id"] or detector["detector_id"] in seen_detectors:
                raise ValueError("optical health detector domain differs")
            seen_detectors.add(detector["detector_id"])
            state = detector["state"]
            if state not in STATES:
                raise ValueError("unknown detector health state")
            totals[array][state] += 1
            if state not in ("retained", "prior_unavailable"):
                summary["exceptions"].append(
                    dict(
                        array=array,
                        network=network["network"],
                        **{
                            key: detector[key]
                            for key in (
                                "detector_id",
                                "state",
                                "reason",
                                "relative_response",
                                "interval",
                            )
                        },
                    )
                )
    summary["arrays"] = {array: dict(counts) for array, counts in totals.items()}
    return summary


def execute(
    prepared: PreparedPoint,
    work,
    reduction,
    native,
    request,
    record,
    *,
    native_call,
    bind,
    write_json,
    shared_work=None,
    shared_health=None,
    weighting_control="inverse-noise",
    array_executor=None,
    memory_handoff=False,
):
    from configured_point_feedback import initialize, update

    if weighting_control not in ("inverse-noise", "equal-control") or (
        weighting_control == "equal-control"
        and (not request.weight_comparison or shared_health is None)
    ):
        raise ValueError("equal control requires explicit comparison and shared health")
    science = getattr(request, "reduction_type", "pointing") == "science"
    label = "SCIENCE" if science else "POINT"
    upstream_work = work if shared_work is None else shared_work
    if memory_handoff:
        if shared_work is not None or shared_health is not None or request.weight_comparison:
            raise ValueError("live POINT products require their original ordinary observation")
        networks = record["stages"][0]["point_handoff"]
        if len(networks) != len(prepared["networks"]):
            raise ValueError("incomplete live POINT network cohort")
        for nw, row in zip(prepared["networks"], networks):
            root = work / "upstream/products" / f"network{nw}"
            if (row["root"] != str(root) or row["transport"] != "observation-memory"
                    or row["input"] != bind(prepared["effective_config"].parent / f"network{nw}-input.json")
                    or not re.fullmatch("[0-9a-f]{64}", row["handoff_sha256"])):
                raise ValueError("live POINT network/input generation differs")
    else:
        networks = []
        for nw in prepared["networks"]:
            root = upstream_work / "upstream/products" / f"network{nw}"
            networks.append(
                dict(
                    root=str(root),
                    input=bind(prepared["effective_config"].parent / f"network{nw}-input.json"),
                    receipt_sha256=bind(root / "receipt.yaml")["sha256"],
                    cal_receipt_sha256=bind(root / "donor-continuity/cal/receipt.yaml")["sha256"],
                    processing_sha256=bind(root / "processing-scans.yaml")["sha256"],
                )
            )
    base = dict(
        schema="citlali-configured-weighted-science-v1" if science else "citlali-configured-weighted-point-v1",
        effective_config=bind(prepared["effective_config"]),
        networks=networks,
        verbose=request.verbose,
    )
    if shared_health is None:
        health_request = work / "health-request.json"
        write_json(health_request, dict(base, stage="health"))
        record["stages"].append(
            native_call(
                native, health_request, work / "health", terminal="map", workers=request.workers, verbose=request.verbose
            )
        )
        health = None if memory_handoff else work / "health/products/health.yaml"
    else:
        health = Path(shared_health["path"])
        if bind(health) != shared_health:
            raise ValueError("comparison shared health changed")
    if memory_handoff:
        execution = record["stages"][-1]["execution"]
        generation = execution["health_generation_sha256"]
        if not re.fullmatch("[0-9a-f]{64}", generation):
            raise ValueError("invalid frozen POINT health identity")
        frozen_health = dict(transport="observation-memory", sha256=generation)
        decisions = (work / "health/products/detector-decisions.yaml").read_bytes()
        if hashlib.sha256(decisions).hexdigest() != execution["detector_decisions_sha256"]:
            raise ValueError("POINT detector decisions differ from health publication")
        summary = detector_summary(yaml.safe_load(decisions), "citlali-point-optical-decisions-v1")
    else:
        frozen_health = bind(health)
        summary = health_summary(health)
    summary_path = reduction / "detector-health.json"
    write_json(summary_path, summary)
    record["detector_health"] = bind(summary_path)
    record["frozen_health"] = frozen_health
    if request.weight_comparison:
        record["weighting_control"] = weighting_control
        base["weighting_control"] = weighting_control
    for array, counts in summary["arrays"].items():
        print(
            f"{array} optical response: kept {counts['retained']}; "
            f"weak {counts['weak_response']}, reversed {counts['sign_mismatch']}, "
            f"unresolved {counts['unresolved']}; prior unavailable {counts['prior_unavailable']}",
            flush=True,
        )

    budget = request.feedback_iterations if request.feedback_enabled else 0
    trajectory = dict(
        requested_feedback_applications=budget,
        applied_feedback_applications=0,
        stopping_reason="feedback_disabled" if not budget else "budget_not_yet_reached",
        convergence_claim=False,
        cycles=[],
    )
    record["feedback"] = trajectory
    contexts, models = {}, {}
    products = None
    for cycle in range(budget + 1):
        if not memory_handoff and bind(health) != frozen_health:
            raise ValueError("frozen optical health changed during feedback")
        stage_dir = work / f"weighted-{cycle:02d}"
        native_request = work / f"weighted-{cycle:02d}-request.json"
        write_json(
            native_request,
            dict(base, stage="cycle", health=frozen_health, cycle_index=cycle, models=models),
        )
        record["stages"].append(
            native_call(native, native_request, stage_dir, terminal="map", workers=request.workers, verbose=request.verbose)
        )
        products = stage_dir / "products"
        result = yaml.safe_load((products / "cycle.yaml").read_text())
        if science and result.get("schema") != "citlali-weighted-science-cycle-v1":
            raise ValueError("SCIENCE cycle publication mode differs")
        from configured_point_products import EMPIRICAL_FAMILY, CONTROL_FAMILY

        family = CONTROL_FAMILY if weighting_control == "equal-control" else EMPIRICAL_FAMILY
        if (
            (not memory_handoff and bind(health) != frozen_health)
            or (memory_handoff and record["stages"][-1]["execution"].get("health_generation_sha256") != frozen_health["sha256"])
            or result["cycle_index"] != cycle
            or result["frozen_health_identity"] != "sha256:" + frozen_health["sha256"]
            or result["coefficient_family"] != family
        ):
            raise ValueError("weighted cycle identity differs")
        if set(result["arrays"]) != set(ARRAYS):
            raise ValueError("weighted cycle must produce all three arrays")
        entry = dict(cycle_index=cycle, native=bind(products / "cycle.yaml"), arrays={})
        for array in ARRAYS:
            bundle = result["arrays"][array]
            actual = yaml.safe_load((products / array / "bundle.yaml").read_text())
            if (
                actual != bundle
                or bundle["array"] != array
                or bundle["observation"] != f"obs:{prepared['observation']}_0_2"
                or bundle["cycle_index"] != cycle
                or bundle["frozen_health_identity"] != result["frozen_health_identity"]
                or bundle["coefficient_family"] != family
            ):
                raise ValueError("weighted array publication identity differs")
            entry["arrays"][array] = {
                k: bundle[k]
                for k in ("contributing_detectors", "contributing_samples", "product_identity")
            }
        trajectory["cycles"].append(entry)
        trajectory["applied_feedback_applications"] = cycle
        print(
            f"{label} {weighting_control} map {cycle}/{budget}: all three arrays complete", flush=True
        )
        if cycle == budget:
            trajectory["stopping_reason"] = (
                "fixed_budget_completed" if budget else "feedback_disabled"
            )
            break
        models, selections = {}, {}
        jobs = [(products / array / "map.fits", stage_dir / f"feedback-{array}",
                 request.verbose, contexts.get(array)) for array in ARRAYS]
        if array_executor is None:
            # Direct library callers retain their explicit serial execution.
            results = []
            for path, destination, verbose, context in jobs:
                if context is None:
                    context, selection = initialize(path, destination, verbose)
                else:
                    selection = update(path, context, destination, verbose)
                results.append((context, selection))
        else:
            results = array_executor.run("feedback", jobs, phase=f"feedback-M{cycle}")
        for array, (context, selection) in zip(ARRAYS, results, strict=True):
            contexts[array] = context
            selections[array] = selection
            if selection["status"] == "ready":
                models[array] = dict(
                    candidate=bind(Path(selection["model_path"])),
                    feedback=bind(stage_dir / f"feedback-{array}" / "feedback.json"),
                    bundle=bind(products / array / "bundle.yaml"),
                )
        entry["source_selection"] = selections
        if len(models) != len(ARRAYS):
            trajectory["stopping_reason"] = "source_feedback_unavailable"
            print(
                "FRUIT stopped: a required source model is unavailable; retaining the last complete all-array maps.",
                flush=True,
            )
            break
        write_json(reduction / "RUN.json", record)
    diagnostic = "weighted-science.json" if science else "weighted-point.json"
    write_json(reduction / diagnostic, trajectory)
    record["weighted_science_diagnostic" if science else "weighted_point_diagnostic"] = bind(reduction / diagnostic)
    record["final_cycle"] = trajectory["applied_feedback_applications"]
    return products
