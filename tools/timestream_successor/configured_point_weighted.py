"""Sequence the configured empirical POINT owners; no timestream arithmetic.

Original completed CAL is the parent of every cycle. Health is learned once,
and source models replace their predecessors. A stopped source selector is an
explicit outcome, never a convergence or ten-pass success claim.
"""

from collections import Counter
from pathlib import Path

import yaml

from configured_point_types import PreparedPoint

ARRAYS = ("a1100", "a1400", "a2000")
STATES = ("prior_unavailable", "retained", "weak_response", "sign_mismatch", "unresolved")


def health_summary(path):
    health = yaml.safe_load(Path(path).read_text())
    if health["schema"] != "citlali-point-optical-health-v1":
        raise ValueError("unexpected optical health publication")
    summary = dict(
        schema="citlali-point-detector-health-summary-v1",
        min_relative_response=health["min_relative_response"],
        method=health["method"],
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
):
    from configured_point_feedback import initialize, update

    if weighting_control not in ("inverse-noise", "equal-control") or (
        weighting_control == "equal-control"
        and (not request.weight_comparison or shared_health is None)
    ):
        raise ValueError("equal control requires explicit comparison and shared health")
    upstream_work = work if shared_work is None else shared_work
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
        schema="citlali-configured-weighted-point-v1",
        effective_config=bind(prepared["effective_config"]),
        networks=networks,
        verbose=request.verbose,
    )
    if shared_health is None:
        health_request = work / "health-request.json"
        write_json(health_request, dict(base, stage="health"))
        record["stages"].append(
            native_call(
                native, health_request, work / "health", terminal="map", verbose=request.verbose
            )
        )
        health = work / "health/products/health.yaml"
    else:
        health = Path(shared_health["path"])
        if bind(health) != shared_health:
            raise ValueError("comparison shared health changed")
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
        if bind(health) != frozen_health:
            raise ValueError("frozen optical health changed during feedback")
        stage_dir = work / f"weighted-{cycle:02d}"
        native_request = work / f"weighted-{cycle:02d}-request.json"
        write_json(
            native_request,
            dict(base, stage="cycle", health=frozen_health, cycle_index=cycle, models=models),
        )
        record["stages"].append(
            native_call(native, native_request, stage_dir, terminal="map", verbose=request.verbose)
        )
        products = stage_dir / "products"
        result = yaml.safe_load((products / "cycle.yaml").read_text())
        from configured_point_products import EMPIRICAL_FAMILY, CONTROL_FAMILY

        family = CONTROL_FAMILY if weighting_control == "equal-control" else EMPIRICAL_FAMILY
        if (
            bind(health) != frozen_health
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
            f"POINT {weighting_control} map {cycle}/{budget}: all three arrays complete", flush=True
        )
        if cycle == budget:
            trajectory["stopping_reason"] = (
                "fixed_budget_completed" if budget else "feedback_disabled"
            )
            break
        models, selections = {}, {}
        for array in ARRAYS:
            map_path = products / array / "map.fits"
            destination = stage_dir / f"feedback-{array}"
            if cycle == 0:
                contexts[array], selection = initialize(map_path, destination, request.verbose)
            else:
                selection = update(map_path, contexts[array], destination, request.verbose)
            selections[array] = selection
            if selection["status"] == "ready":
                models[array] = dict(
                    candidate=bind(Path(selection["model_path"])),
                    feedback=bind(destination / "feedback.json"),
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
    write_json(reduction / "weighted-point.json", trajectory)
    record["weighted_point_diagnostic"] = bind(reduction / "weighted-point.json")
    record["final_cycle"] = trajectory["applied_feedback_applications"]
    return products
