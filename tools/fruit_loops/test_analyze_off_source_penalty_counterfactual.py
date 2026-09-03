from __future__ import annotations

import pytest

from tools.fruit_loops import analyze_off_source_penalty_counterfactual as analysis


def test_reversal_fraction_uses_lower_is_better_direction() -> None:
    assert analysis.reversal_fraction(0.4, 0.2, 0.6) == pytest.approx(0.5)
    assert analysis.reversal_fraction(0.1, 0.2, 0.6) == pytest.approx(1.25)


def test_reversal_fraction_rejects_non_loss() -> None:
    with pytest.raises(ValueError, match="do not define a loss"):
        analysis.reversal_fraction(0.3, 0.6, 0.2)


@pytest.mark.parametrize(
    ("kernel", "annular", "expected"),
    (
        (0.5, 0.8, "substantial_causal_contribution"),
        (0.49, 0.2, "partial_causal_contribution"),
        (0.2, -0.1, "mixed_effect"),
        (0.0, 0.0, "no_support_for_causal_contribution"),
    ),
)
def test_classify_effect(kernel: float, annular: float, expected: str) -> None:
    assert analysis.classify_effect(kernel, annular, 0.5) == expected


def test_read_execution_accepts_bsd_and_gnu_time_order(tmp_path) -> None:
    common = "citlali is done!\n12345 maximum resident set size\n"
    (tmp_path / "untouched-injected-sham.log").write_text(
        common + "real 30.5\nuser 29.1\nsys 0.6\n"
    )
    (tmp_path / "injected-without-uid4460.log").write_text(
        common + "31.5 real\n29.2 user\n0.7 sys\n"
    )

    rows = analysis.read_execution(tmp_path)

    assert rows[0]["wall_seconds"] == 30.5
    assert rows[1]["wall_seconds"] == 31.5
    assert all(row["maximum_resident_bytes"] == 12345 for row in rows)
