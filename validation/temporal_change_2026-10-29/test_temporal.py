"""Mathematical and boundary checks; Monte Carlo is the scientific assessment."""
import numpy as np
import pytest
from experiment import fixture
from temporal import OPTICAL_CUT, assess, multiply, prepare, residual
from map_tradeoffs import clean, impact, project, setup
from oracle_control import known_factor


def design_case(case="iid", seed=2):
    t, b, x, y, mean = fixture(case, seed)
    return [prepare(p, t, b) for p in x], y, mean


def test_optical_policy_not_temporal_admission():
    d, y, _ = design_case()
    result = assess(d, y, draws=199)
    assert OPTICAL_CUT == .1
    assert result["admission"] == "not_decided"
    assert result["harm_threshold"] is None


def test_fwl_matches_independent_nested_least_squares():
    t, b, panels, y, _ = fixture("step_030", 8)
    d = prepare(panels[0], t, b)
    x = panels[0]
    offsets = (b[:, None] == np.unique(b)).astype(float)
    patterns = [(b >= i).astype(float) for i in range(1, 6)] + [t/239]
    for j, pattern in enumerate(patterns):
        columns = np.column_stack((offsets, x, x*pattern))
        beta = np.linalg.lstsq(columns, y, rcond=None)[0][-1]
        assert multiply(d.contrast, y[:, None])[j, 0] == pytest.approx(beta, abs=1e-11)


def test_contrast_annihilates_null_and_offset_changes():
    d, _, _ = design_case()
    assert np.max(np.abs(multiply(d[0].contrast, d[0].null))) < 1e-12


def test_residual_orthogonality():
    d, y, _ = design_case()
    r = residual(d[0].null, y[:, None])
    assert np.max(np.abs(multiply(d[0].null.T, r))) < 1e-12


def test_deterministic_and_input_unchanged():
    d, y, _ = design_case()
    before = y.copy()
    assert assess(d, y, draws=199, seed=13) == assess(d, y, draws=199, seed=13)
    np.testing.assert_array_equal(y, before)


def test_duplicate_panel_not_independent_confirmation():
    d, y, _ = design_case()
    one = assess([d[0]], y, draws=199)
    two = assess([d[0], d[0]], y, draws=199)
    assert one["p_value"] == two["p_value"]
    np.testing.assert_allclose(one["lower"], two["lower"][:6])


@pytest.mark.parametrize("case", ["no_excitation", "constant_last_block"])
def test_missing_excitation_is_unavailable(case):
    d, y, _ = design_case(case)
    result = assess(d, y, draws=199)
    assert not result["available"]
    assert result["reason"] == "change_design_unidentifiable"


def test_offsets_do_not_change_test():
    t, b, x, y, _ = fixture("iid", 2)
    d = [prepare(p, t, b) for p in x]
    shifted = y + np.array([2., -1., 4., 0., 9., 5.])[b]
    a, z = assess(d, y, draws=199), assess(d, shifted, draws=199)
    assert a["p_value"] == z["p_value"]
    np.testing.assert_allclose(a["estimate"], z["estimate"], atol=1e-11)


def test_shared_target_scaling_invariant_statistic():
    d, y, _ = design_case()
    a, z = assess(d, y, draws=199), assess(d, 3*y, draws=199)
    assert a["p_value"] == z["p_value"]
    np.testing.assert_allclose(3*np.array(a["estimate"]), z["estimate"], atol=1e-11)


def test_gap_resets_are_retained():
    t, b, x, _, _ = fixture("gaps", 1)
    d = prepare(x[0], t, b)
    np.testing.assert_array_equal(d.linked[1:], (np.diff(t) == 1) & (np.diff(b) == 0))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_target_rejected(bad):
    d, y, _ = design_case()
    y[4] = bad
    with pytest.raises(ValueError):
        assess(d, y)


def test_no_excitation_blocks_are_not_deleted():
    t, b, x, _, _ = fixture("constant_last_block", 1)
    assert prepare(x[0], t, b) is None


def test_map_zero_and_linear_contamination():
    s = setup()
    _, a = impact(s, 0, .2, "step_0.5", "atmospheric_only")
    assert np.max(np.abs(a)) == 0
    _, a = impact(s, .1, .01, "step_0.5", "atmospheric_only")
    _, b = impact(s, .2, .05, "step_0.5", "atmospheric_only")
    np.testing.assert_allclose(b, 10*a, atol=1e-12)


def test_fixed_cleaner_removes_constant_atmospheric_coupling():
    s = setup()
    assert np.max(np.abs(clean(s, 3+2*s["atmosphere"]))) < 1e-11


def test_map_constant_and_scan_association():
    s = setup()
    np.testing.assert_allclose(project(s, np.ones(len(s["scan"]))), 1)
    np.testing.assert_allclose(project(s, s["source"].ravel()[s["scan"]]), s["source"])


def test_source_gain_is_distinct_from_atmosphere_only():
    s = setup()
    a, _ = impact(s, .3, .2, "step_0.5", "atmospheric_only")
    b, _ = impact(s, .3, .2, "step_0.5", "all_signal")
    assert b["amplitude_bias_percent"] > a["amplitude_bias_percent"] + 2


def test_oracle_covariance_factor_matches_direct_recursion():
    times = np.array([0, 1, 2, 4, 5, 6, 7])
    blocks = np.array([0, 0, 0, 0, 0, 1, 1])
    z = np.array([1., -2., 3., -1., 2., .5, -1.])
    expected = .1*z.copy()
    for i in range(1, len(z)):
        if times[i] == times[i-1]+1 and blocks[i] == blocks[i-1]:
            expected[i] += .6*expected[i-1]
    actual = multiply(known_factor(blocks, times, .6, .1), z[:, None])[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_oracle_covariance_matches_stationary_limit_and_reset():
    t = np.arange(80)
    blocks = t//40
    factor = known_factor(blocks, t, .6, .1)
    covariance = multiply(factor, factor.T)
    assert covariance[39, 39] == pytest.approx(.1**2/(1-.6**2))
    assert covariance[39, 40] == 0
    assert covariance[40, 40] == pytest.approx(.1**2)
