"""Paired weighting ablation: shared CAL/health, independent identical FRUIT recipes.

Only orchestration and exact association/support comparison live here. A
completed pair is a controlled experiment, never photometric qualification.
"""

from pathlib import Path

import yaml

from configured_point_products import ARRAYS, EMPIRICAL_FAMILY, CONTROL_FAMILY

SUPPORT_KEYS = (
    "ordered_domain_sha256",
    "input_support_sha256",
    "source_free_sha256",
    "residual_support_sha256",
    "cleaned_support_sha256",
    "mapped_support_sha256",
)


def compare_support(weighted, equal, bind):
    """Authenticate all cycles and require the same sample domain, not just counts."""
    if weighted["frozen_health"] != equal["frozen_health"]:
        raise ValueError("comparison frozen detector health differs")
    budget = weighted["feedback"]["requested_feedback_applications"]
    for record in (weighted, equal):
        trajectory = record["feedback"]
        if (
            trajectory["requested_feedback_applications"] != budget
            or trajectory["applied_feedback_applications"] != budget
            or trajectory["stopping_reason"]
            != ("fixed_budget_completed" if budget else "feedback_disabled")
            or [e["cycle_index"] for e in trajectory["cycles"]]
            != list(range(budget + 1))
        ):
            raise ValueError(
                "comparison requires both complete identical FRUIT budgets"
            )
    checks = []
    for cycle in range(budget + 1):
        plans = []
        for record, family in ((weighted, EMPIRICAL_FAMILY), (equal, CONTROL_FAMILY)):
            ref = record["feedback"]["cycles"][cycle]["native"]
            path = Path(ref["path"])
            if bind(path) != ref:
                raise ValueError("comparison cycle publication changed")
            result = yaml.safe_load(path.read_text())
            if (
                result["cycle_index"] != cycle
                or result["coefficient_family"] != family
                or result["frozen_health_identity"]
                != "sha256:" + record["frozen_health"]["sha256"]
                or set(result["arrays"]) != set(ARRAYS)
            ):
                raise ValueError("comparison cycle/arm/health identity differs")
            arm = {}
            for array in ARRAYS:
                bundle = result["arrays"][array]
                plan_path = path.parent / array / "plan.yaml"
                if bind(plan_path)["sha256"] != bundle["files"]["plan.yaml"]:
                    raise ValueError("comparison map plan changed")
                plan = yaml.safe_load(plan_path.read_text())
                if (
                    plan["array"] != array
                    or plan["cycle_index"] != cycle
                    or plan["frozen_health_identity"]
                    != result["frozen_health_identity"]
                    or plan["WCS"] != bundle["WCS"]
                ):
                    raise ValueError("comparison map plan identity differs")
                arm[array] = plan
            plans.append(arm)
        for array in ARRAYS:
            a, b = (p[array] for p in plans)
            if a["observation"] != b["observation"] or a["WCS"] != b["WCS"]:
                raise ValueError(
                    f"comparison observation/grid differs: {array} cycle {cycle}"
                )
            if not a["segments"] or len(a["segments"]) != len(b["segments"]):
                raise ValueError("comparison segment domains differ")
            for left, right in zip(a["segments"], b["segments"], strict=True):
                for key in ("network", "scan", "rank", *SUPPORT_KEYS):
                    if key not in left or key not in right or left[key] != right[key]:
                        raise ValueError(
                            f"comparison {key} differs: {array} cycle {cycle}, network {left['network']}"
                        )
                if not left["converged"] or not right["converged"]:
                    raise ValueError("comparison contains an unconverged cleaner")
            checks.append(
                dict(
                    cycle=cycle,
                    array=array,
                    segments=len(a["segments"]),
                    support_equal=True,
                )
            )
    return checks


def execute_equal_arm(
    prepared,
    work,
    reduction,
    native,
    request,
    weighted_record,
    *,
    native_call,
    bind,
    write_json,
    publish_maps,
):
    from configured_point_weighted import execute

    root = reduction / "comparison/equal"
    root.mkdir(parents=True)
    equal_work = work / "equal-control"
    equal_work.mkdir()
    record = dict(
        schema="citlali-configured-point-equal-control-run-v1",
        status="running",
        received_config=weighted_record["received_config"],
        executable=weighted_record["executable"],
        version=weighted_record["version"],
        stages=[],
        calibration_eligible=False,
    )
    comparison = dict(
        schema="citlali-point-weight-comparison-v1",
        status="running",
        changed="detector coefficients in both cleaning and mapmaking",
        shared_upstream=True,
        shared_health=weighted_record["frozen_health"],
        fruit="same fixed recipe and budget; source trajectories are independent consequences",
        calibration_eligible=False,
    )
    comparison_path = reduction / "weight-comparison.json"
    write_json(comparison_path, comparison)
    try:
        trajectory = weighted_record["feedback"]
        if (
            trajectory["applied_feedback_applications"]
            != trajectory["requested_feedback_applications"]
        ):
            raise ValueError(
                "noise-weighted arm stopped before the requested budget; equal arm not launched"
            )
        products = execute(
            prepared,
            equal_work,
            root,
            native,
            request,
            record,
            native_call=native_call,
            bind=bind,
            write_json=write_json,
            shared_work=work,
            shared_health=weighted_record["frozen_health"],
            weighting_control="equal-control",
        )
        # Preserve the complete equal-arm maps even if the controlled support gate fails.
        record["products"] = publish_maps(
            products,
            root / str(prepared["observation"]) / "raw",
            prepared["observation"],
            request.arrays,
            fit_diagnostics=request.fit_diagnostics,
            verbose=request.verbose,
        )
        record["status"] = "completed"
        comparison["checks"] = compare_support(weighted_record, record, bind)
        comparison["status"] = "matched_support_complete"
        comparison["equal_run"] = str(root / "RUN.json")
        print(
            "Weight comparison: both arms complete; every cycle has identical detector/sample support.",
            flush=True,
        )
    except Exception as exc:
        record["status"] = "comparison_failed"
        comparison.update(status="not_a_matched_complete_comparison", error=str(exc))
        raise
    finally:
        write_json(root / "RUN.json", record)
        write_json(comparison_path, comparison)
        weighted_record["weight_comparison"] = bind(comparison_path)
