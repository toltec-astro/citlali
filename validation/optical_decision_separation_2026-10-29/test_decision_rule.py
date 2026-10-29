"""Synthetic physical predictions and failure-boundary tests, not reductions."""
from dataclasses import FrozenInstanceError
import math

import pytest

from decision_rule import Interval as I, consider
from synthetic_study import bootstrap_counts, evaluate


def six(full=(.8, 1.2), early=(.7, 1.3), late=(.6, 1.4)):
    return tuple(I(*bounds) for bounds in (full, full, early, early, late, late))


def test_normal_full_response_survives_uncertain_short_half():
    evidence = six(late=(-.3, 1.4))
    before = tuple(evidence)
    decision = consider(evidence)
    assert decision.eligible and decision.optical == "responsive"
    assert decision.temporal == "subset_uncertainty_only"
    assert decision.reasons == ("eligible_with_temporal_uncertainty",)
    assert evidence == before
    with pytest.raises(FrozenInstanceError):
        decision.eligible = False


@pytest.mark.parametrize("bounds,optical", [((.02,.05),"weak"), ((-1.2,-.8),"reversed"), ((.05,.15),"unresolved")])
def test_full_response_failure_cannot_be_rescued_by_good_halves(bounds, optical):
    d = consider(six(full=bounds))
    assert d.optical == optical and not d.eligible


@pytest.mark.parametrize("upper", [.099, .10, -.2])
def test_known_late_failure_is_not_mistaken_for_imprecision(upper):
    d = consider(six(late=(-.5,upper)))
    assert not d.eligible and "subset_response_at_or_below_cut" in d.reasons


def test_positive_temporal_change_requires_review_without_new_tolerance():
    d = consider(six(early=(.8,1.1), late=(1.4,1.8)))
    assert d.optical == "responsive" and not d.eligible
    assert d.temporal == "difference_requires_review"


def test_temporal_endpoint_contact_is_overlap():
    assert consider(six(early=(.8,1.1),late=(1.1,1.8))).eligible


@pytest.mark.parametrize("index", range(6))
def test_missing_is_not_a_wide_interval(index):
    evidence = list(six()); evidence[index] = I(None,None)
    assert not consider(evidence).eligible


def test_missing_half_does_not_hide_another_known_failure():
    evidence = list(six()); evidence[2] = I(None,None); evidence[4] = I(-.8,-.5)
    d = consider(evidence)
    assert not d.eligible
    assert "temporal_interval_unavailable" in d.reasons
    assert "subset_response_at_or_below_cut" in d.reasons


@pytest.mark.parametrize("key", ["prior_available","reference_available"])
def test_prior_or_reference_failure_cannot_be_rescued(key):
    assert not consider(six(),**{key:False}).eligible


def test_disturbance_exclusion_preserved():
    d = consider(six(),disturbance_excluded=True)
    assert not d.eligible and d.reasons == ("disturbance_excluded",)


def test_full_cut_contact_and_tunable_threshold():
    assert not consider(six(full=(.1,.2))).eligible
    assert consider(six(full=(.1,.2)),threshold=.05).eligible
    assert not consider(six(full=(.8,1.2)),threshold=.9).eligible


def test_one_disagreeing_full_panel_withholds():
    evidence = list(six()); evidence[1] = I(-.3,.4)
    assert not consider(evidence).eligible


@pytest.mark.parametrize("cut", [True,0,-.1,1.1,math.nan,math.inf])
def test_invalid_threshold(cut):
    with pytest.raises(ValueError): consider(six(),cut)


@pytest.mark.parametrize("bounds", [(None,1),(0,None),(math.nan,1),(0,math.inf),(2,1)])
def test_malformed_interval(bounds):
    with pytest.raises(ValueError): I(*bounds)


def test_wrong_interval_shape():
    with pytest.raises(ValueError): consider(six()[:5])


@pytest.mark.parametrize("name", ["healthy","quiet_healthy","noisy_healthy"])
def test_generated_responsive_detectors_are_useful_despite_noise_differences(name):
    assert evaluate(name)["decision"]["eligible"]


@pytest.mark.parametrize("name", ["weak_quiet","reversed","positive_gain_change","late_weak","late_reversed","no_excitation","constant_block","ambiguous_full"])
def test_generated_bad_or_unresolved_cases_withheld(name):
    assert not evaluate(name)["decision"]["eligible"]


def test_generated_limited_excitation_exposes_old_subset_veto():
    result = evaluate("limited_excitation")
    assert result["decision"]["eligible"]
    assert result["decision"]["temporal"] == "subset_uncertainty_only"
    assert not result["old_six_interval_admission"]


def test_three_block_bootstrap_has_no_hidden_extra_information():
    counts, probability = bootstrap_counts(3)
    assert counts.shape == (10,3)
    assert sum(probability) == pytest.approx(1)
    for i in range(3):
        assert probability[counts[:,i]==3][0] == pytest.approx(1/27)


def test_deterministic_reference_experiment():
    assert evaluate("limited_excitation") == evaluate("limited_excitation")


@pytest.mark.parametrize("seed", [7,15])
@pytest.mark.xfail(strict=True, reason="Known scientific gate failure: non-overlap veto falsely withholds stationary response; candidate is not activation-ready")
def test_stationary_response_counterexamples_require_policy_revision(seed):
    # Assertion states the scientific goal, not the candidate's faulty result.
    assert evaluate("healthy", seed)["decision"]["eligible"]
