import json

import numpy as np
import pytest
import yaml

from prepare_rtc_contaminant import prepare
from report_rtc_decision_apply import compare, support_counts
from run_rtc_pipeline_replay import digest


def test_counts_distinguish_replacement_influence_and_pair_availability():
    a = dict(
        state=np.array([3, 29, 17, 3], dtype="u1"),
        rows=np.arange(4),
        cause=np.zeros(4),
        input_cause=np.zeros(4),
        pair=np.zeros((4, 2)),
    )
    counts = support_counts(a)
    assert counts["realized_replaced_native"] == 1
    assert counts["representative_excluded_output"] == 1
    assert counts["nonrepresentative_influence_output"] == 1
    assert counts["paired_available_output"] == 2
    assert counts["x_available_output"] == 4
    b = {**a, "state": np.full(4, 3, dtype="u1")}
    assert compare(a, b, 1)["only_second"] == 2
    assert compare(a, b)["only_second"] == 0


def test_contaminant_placement_requires_real_common_paired_support(tmp_path):
    reference = tmp_path / "reference"
    (reference / "exclusion-control").mkdir(parents=True)
    samples = tmp_path / "samples.f64"
    rng = np.random.default_rng(1)
    np.column_stack((rng.normal(size=(2000, 2)), np.ones((2000, 2)))).astype(
        "<f8"
    ).tofile(samples)
    config = dict(
        detectors=[
            dict(
                channel=4,
                filter="w1-t3",
                samples=dict(path=str(samples), sha256=digest(samples)),
            )
        ]
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    receipt = dict(
        Apply_performed=True,
        original_pair_unchanged=True,
        rows=2000,
        cadence_interval_seconds=0.01,
        configuration_sha256=digest(path),
    )
    (reference / "receipt.yaml").write_text(yaml.safe_dump(receipt))
    state = np.full(2000, 3, dtype="u1")
    state.tofile(reference / "exclusion-control" / "4-state.u8")
    output = tmp_path / "injected.json"
    prepare(path, reference, output)
    result = json.loads(output.read_text())["declared_contaminant"]
    assert result["reference_configuration"]["sha256"] == digest(path)
    assert result["native_row"] == 500
    assert result["x_delta"] > 0 and result["r_delta"] > 0
    assert digest(samples) == config["detectors"][0]["samples"]["sha256"]
    state[:] = 1  # finite x alone cannot supply a usable paired interval
    state.tofile(reference / "exclusion-control" / "4-state.u8")
    with pytest.raises(ValueError, match="no originally usable interval"):
        prepare(path, reference, tmp_path / "unavailable.json")
    with pytest.raises(FileExistsError):
        state[:] = 3
        state.tofile(reference / "exclusion-control" / "4-state.u8")
        prepare(path, reference, output)


def test_availability_inconsistent_with_values_is_not_a_success():
    a = dict(
        state=np.array([3], dtype="u1"),
        rows=np.array([0]),
        pair=np.array([[np.nan, 1.0]]),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        compare(a, a)
