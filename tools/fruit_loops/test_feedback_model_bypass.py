from __future__ import annotations

import pytest

from tools.fruit_loops import analyze_feedback_model_bypass as analysis


def screen() -> dict:
    return {
        "maximum_central_recovery_absolute_error_increase": 0.01,
        "maximum_major_width_absolute_error_increase": 0.01,
        "maximum_minor_width_absolute_error_increase": 0.01,
        "maximum_centroid_error_increase_arcsec": 0.1,
        "maximum_annular_residual_ratio": 1.10,
        "maximum_kernel_residual_ratio": 1.10,
    }


def response() -> dict:
    return {
        "kernel_normalized_central_recovery": 0.98,
        "major_fwhm_over_kernel": 1.01,
        "minor_fwhm_over_kernel": 0.99,
        "centroid_error_arcsec": 0.05,
        "annular_residual_over_truth": 0.02,
        "kernel_residual_relative_rms": 0.04,
    }


def test_reversal_fraction_has_registered_directions() -> None:
    assert analysis.reversal_fraction(0.90, 0.82, 0.89, True) == pytest.approx(
        8.0 / 7.0
    )
    assert analysis.reversal_fraction(0.003, 0.021, 0.006, False) == pytest.approx(
        1.2
    )


def test_regression_screen_uses_strict_registered_limits() -> None:
    complete = response()
    candidate = response()
    candidate["annular_residual_over_truth"] = 0.022
    candidate["kernel_residual_relative_rms"] = 0.044

    assert analysis.regression_failures(complete, candidate, screen()) == []

    candidate["kernel_residual_relative_rms"] = 0.0441
    assert analysis.regression_failures(complete, candidate, screen()) == [
        "kernel_residual"
    ]


def test_regression_details_report_the_failed_metric() -> None:
    complete = response()
    candidate = response()
    candidate["annular_residual_over_truth"] = 0.024

    assert analysis.regression_failure_details(
        complete, candidate, screen()
    ) == [
        {
            "metric": "annular_residual",
            "complete_value": 0.02,
            "candidate_value": 0.024,
            "comparison": "candidate_over_complete_ratio",
            "measured": pytest.approx(1.2),
            "limit": 1.1,
        }
    ]


def test_penalty_comparison_distinguishes_timing_and_removal() -> None:
    common = {
        "obsnum": 1,
        "variant": "injected",
        "scan": 5,
        "array": 1,
        "score": 4.0,
        "factor": 0.0,
        "scan_local": True,
    }
    rows = [
        {
            **common,
            "evidence_view": "complete_map",
            "iteration": 3,
            "uid": 10,
        },
        {
            **common,
            "evidence_view": "feedback_excluded",
            "iteration": 4,
            "uid": 10,
        },
        {
            **common,
            "evidence_view": "complete_map",
            "iteration": 2,
            "uid": 11,
        },
    ]

    comparison = analysis.compare_penalties(rows)

    assert [row["disposition"] for row in comparison] == [
        "timing_changed",
        "removed",
    ]
