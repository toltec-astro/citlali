"""Compact adapter for native scientific exclusion decisions; no classification."""

from pathlib import Path
import hashlib
import yaml

from configured_point_config import ARRAYS, DISTURBANCE_POLICY

COUNTS = (
    "input_detectors",
    "previously_unavailable",
    "evaluated_detectors",
    "selected_for_exclusion",
    "newly_dropped",
    "retained_after_decision",
)


def collect(root, observation, networks):
    result = dict(
        schema="citlali-detector-drops-observation-v1",
        observation=observation,
        policy=DISTURBANCE_POLICY,
        evaluation_state="complete",
        networks=[],
        arrays={},
        final_map_contributors=None,
        final_map_contributors_status="not_measured_by_RTC_diagnostic",
    )
    seen = set()
    for nw in networks:
        array = ARRAYS[0 if nw < 7 else 1 if nw < 11 else 2]
        path = Path(root) / f"network{nw}" / "detector-drops.yaml"
        row = dict(
            network=nw,
            array=array,
            evaluation_state="not_evaluated",
            application_state="not_applied",
            detectors=[],
            **{k: None for k in COUNTS},
        )
        if path.exists():
            data = yaml.safe_load(path.read_text())
            if (
                data.get("schema") != "citlali-detector-drops-network-v1"
                or data.get("policy") != DISTURBANCE_POLICY
                or data.get("observation") != observation
                or data.get("network") != nw
                or data.get("array") != ARRAYS.index(array)
            ):
                raise ValueError("foreign detector-drop evidence: " + str(path))
            row.update(
                data,
                array=array,
                source=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            )
            for key in COUNTS:
                value = row[key]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("invalid detector-drop count: " + key)
            if row["evaluation_state"] == "complete":
                if any(row[k] is None for k in COUNTS if k != "newly_dropped"):
                    raise ValueError("complete exclusion decision has missing counts")
                if (
                    row["input_detectors"]
                    != row["previously_unavailable"] + row["evaluated_detectors"]
                    or row["evaluated_detectors"]
                    != row["selected_for_exclusion"] + row["retained_after_decision"]
                    or len(row["detectors"]) != row["selected_for_exclusion"]
                ):
                    raise ValueError("detector-drop counts do not reconcile")
            for detector in row["detectors"]:
                key = (nw, detector["detector_occurrence"])
                if (
                    key in seen
                    or detector["network"] != nw
                    or not key[1]
                    or detector["reason"] != "repeated_level_shifts_unassessable_remainder"
                ):
                    raise ValueError("duplicate or invalid excluded-detector identity")
                seen.add(key)
            if row["application_state"] == "applied" and (
                row["evaluation_state"] != "complete"
                or row["newly_dropped"] != row["selected_for_exclusion"]
            ):
                raise ValueError("applied exclusion does not match the frozen decision")
        if row["evaluation_state"] != "complete" or row["application_state"] != "applied":
            result["evaluation_state"] = "incomplete"
        result["networks"].append(row)
    for array in ARRAYS:
        rows = [r for r in result["networks"] if r["array"] == array]
        result["arrays"][array] = {
            k: sum(r[k] for r in rows) if rows and all(r[k] is not None for r in rows) else None
            for k in COUNTS
        }
    result["totals"] = {
        k: sum(r[k] for r in result["arrays"].values())
        if all(r[k] is not None for r in result["arrays"].values())
        else None
        for k in COUNTS
    }
    return result


def console_summary(summary, path):
    def value(v):
        return "unavailable" if v is None else str(v)

    arrays = ", ".join(f"{a}={value(summary['arrays'][a]['newly_dropped'])}" for a in ARRAYS)
    return f"Detectors dropped for disturbances: {value(summary['totals']['newly_dropped'])} ({arrays}); {summary['evaluation_state']}; details: {path}"
