from __future__ import annotations

import pytest

from tools.fruit_loops import analyze_penalty_counterfactual as analysis


def test_reversal_fraction_has_registered_directions() -> None:
    assert analysis.reversal_fraction(0.89, 0.90, 0.80, 1) == pytest.approx(
        0.9
    )
    assert analysis.reversal_fraction(0.011, 0.010, 0.020, -1) == pytest.approx(
        0.9
    )


@pytest.mark.parametrize(
    ("recovery", "annular", "expected"),
    (
        (0.8, 0.5, "substantial_causal_contribution"),
        (0.4, 0.2, "partial_causal_contribution"),
        (0.2, -0.1, "mixed_effect"),
        (0.0, 0.0, "no_support_for_causal_contribution"),
    ),
)
def test_classify_effect(
    recovery: float,
    annular: float,
    expected: str,
) -> None:
    assert analysis.classify_effect(recovery, annular) == expected


def test_reversal_fraction_rejects_non_loss() -> None:
    with pytest.raises(ValueError, match="do not define a loss"):
        analysis.reversal_fraction(0.9, 0.8, 0.9, 1)
