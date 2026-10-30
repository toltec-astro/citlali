"""Comparison controls and proof gates; synthetic fixtures only."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import yaml

import configured_point_cli as cli
import configured_point_comparison as comparison
import configured_point_products as products
import configured_point_weighted as weighted
from test_configured_point_weighted import orchestration as orchestration


def test_equal_arm_reuses_exact_upstream_and_health_without_relearning(request):
    s = request.getfixturevalue("orchestration")
    s.request.weight_comparison = True
    s.execute()
    prior = deepcopy(s.calls)
    work = s.work / "equal-control"
    work.mkdir()
    root = s.reduction / "equal"
    root.mkdir()
    record = dict(stages=[])

    def native(*args, **kwargs):
        result = s.native_call(*args, **kwargs)
        directory = args[2] / "products"
        path = directory / "cycle.yaml"
        cycle = yaml.safe_load(path.read_text())
        cycle["coefficient_family"] = products.CONTROL_FAMILY
        for array, bundle in cycle["arrays"].items():
            bundle["coefficient_family"] = products.CONTROL_FAMILY
            (directory / array / "bundle.yaml").write_text(yaml.safe_dump(bundle))
        path.write_text(yaml.safe_dump(cycle))
        return result

    weighted.execute(
        s.prepared,
        work,
        root,
        Path("/synthetic/native"),
        s.request,
        record,
        native_call=native,
        bind=cli.bind,
        write_json=cli.write_json,
        shared_work=s.work,
        shared_health=s.record["frozen_health"],
        weighting_control="equal-control",
    )
    calls = s.calls[len(prior) :]
    assert len(calls) == 3 and all(q["stage"] == "cycle" for q in calls)
    assert all(q["networks"] == prior[0]["networks"] for q in calls)
    assert all(q["health"] == s.record["frozen_health"] for q in calls)
    assert all(q["weighting_control"] == "equal-control" for q in calls)
    assert record["frozen_health"] == s.record["frozen_health"]
    # Models after M0 come from this arm, never from the noise-weighted arm.
    assert all(
        str(work) in q["models"]["a1100"]["candidate"]["path"] for q in calls[1:]
    )
    outputs = products.publish_maps(
        work / "weighted-02/products",
        root / "published",
        152389,
        products.ARRAYS,
        fit_diagnostics=False,
        verbose=False,
    )
    assert all(v["coefficient_family"] == products.CONTROL_FAMILY for v in outputs)
    assert all("equal-control" in v["normalization"] for v in outputs)


@pytest.fixture
def pair(tmp_path):
    result = []
    for name, family in [
        ("weighted", products.EMPIRICAL_FAMILY),
        ("equal", products.CONTROL_FAMILY),
    ]:
        root = tmp_path / name
        root.mkdir()
        arrays = {}
        for array in products.ARRAYS:
            directory = root / array
            directory.mkdir()
            plan = dict(
                array=array,
                cycle_index=0,
                frozen_health_identity="sha256:" + "a" * 64,
                observation="obs:152418_0_2",
                WCS={"grid": "same"},
                segments=[
                    dict(
                        network=0,
                        scan=1,
                        rank=3,
                        converged=True,
                        **{k: "b" * 64 for k in comparison.SUPPORT_KEYS},
                    )
                ],
            )
            path = directory / "plan.yaml"
            path.write_text(yaml.safe_dump(plan))
            arrays[array] = dict(WCS=plan["WCS"], files={"plan.yaml": cli.digest(path)})
        cycle = dict(
            cycle_index=0,
            coefficient_family=family,
            frozen_health_identity="sha256:" + "a" * 64,
            arrays=arrays,
        )
        path = root / "cycle.yaml"
        path.write_text(yaml.safe_dump(cycle))
        result.append(
            dict(
                frozen_health={"path": "shared-health", "sha256": "a" * 64},
                feedback=dict(
                    requested_feedback_applications=0,
                    applied_feedback_applications=0,
                    stopping_reason="feedback_disabled",
                    cycles=[dict(cycle_index=0, native=cli.bind(path))],
                ),
            )
        )
    return result


def test_exact_support_gate_accepts_matching_domains(pair):
    assert len(comparison.compare_support(*pair, cli.bind)) == 3


@pytest.mark.parametrize("key", comparison.SUPPORT_KEYS)
def test_exact_support_gate_rejects_same_counts_but_changed_domains(pair, key):
    ref = pair[1]["feedback"]["cycles"][0]["native"]
    path = Path(ref["path"])
    plan_path = path.parent / "a1100/plan.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    plan["segments"][0][key] = "c" * 64
    plan_path.write_text(yaml.safe_dump(plan))
    cycle = yaml.safe_load(path.read_text())
    cycle["arrays"]["a1100"]["files"]["plan.yaml"] = cli.digest(plan_path)
    path.write_text(yaml.safe_dump(cycle))
    pair[1]["feedback"]["cycles"][0]["native"] = cli.bind(path)
    with pytest.raises(ValueError, match=key):
        comparison.compare_support(*pair, cli.bind)


def test_stopped_feedback_cannot_be_a_completed_comparison(pair):
    pair[1]["feedback"]["stopping_reason"] = "source_feedback_unavailable"
    with pytest.raises(ValueError, match="budgets"):
        comparison.compare_support(*pair, cli.bind)


@pytest.mark.parametrize("stopped", [False, True])
def test_pair_coordinator_publishes_honest_completion_or_failure(
    pair, tmp_path, monkeypatch, stopped
):
    work = tmp_path / "run/.point-work"
    work.mkdir(parents=True)
    record = pair[0]
    record.update(
        received_config={"path": "config"},
        executable={"path": "native"},
        version="test",
    )
    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        args[5].update(pair[1])
        return tmp_path / "native-products"

    monkeypatch.setattr(weighted, "execute", execute)
    if stopped:
        record["feedback"]["requested_feedback_applications"] = 10

    def run():
        comparison.execute_equal_arm(
            dict(observation=152418),
            work,
            work.parent,
            Path("/native"),
            SimpleNamespace(
                arrays=products.ARRAYS, fit_diagnostics=False, verbose=False
            ),
            record,
            native_call=None,
            bind=cli.bind,
            write_json=cli.write_json,
            publish_maps=lambda *a, **k: [],
        )

    if stopped:
        with pytest.raises(ValueError, match="not launched"):
            run()
        assert not calls
    else:
        run()
        assert (
            calls[0]["shared_work"] == work
            and calls[0]["shared_health"] == record["frozen_health"]
        )
    outcome = json.loads((work.parent / "weight-comparison.json").read_text())
    assert outcome["status"] == (
        "not_a_matched_complete_comparison" if stopped else "matched_support_complete"
    )
    assert (work.parent / "comparison/equal/RUN.json").is_file()
