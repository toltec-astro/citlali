from __future__ import annotations

import math

import numpy as np
import pytest

from tools.fruit_loops import analyze_causal_diagnostic_discovery as analysis


def test_finite_rms_and_cosine_use_declared_finite_support() -> None:
    left = np.array([[1.0, 2.0], [math.nan, 4.0]])
    right = np.array([[2.0, 4.0], [8.0, math.nan]])
    mask = np.ones((2, 2), dtype=bool)

    assert analysis.finite_rms(left, mask) == math.sqrt(7.0)
    assert analysis.finite_cosine(left, right, mask) == pytest.approx(1.0)


def test_new_penalty_counts_are_array_explicit() -> None:
    baseline = {("producer", "reason", 1, 2, 100, 0)}
    current = baseline | {
        ("producer", "reason", 4, 5, 4460, 1),
        ("producer", "reason", 4, 7, 5000, 1),
        ("producer", "reason", 4, 9, 6000, -1),
    }

    assert analysis.new_penalty_counts(baseline, current) == {
        "a1100": 0,
        "a1400": 2,
        "a2000": 0,
    }


def test_array_screen_pass_keeps_science_protections_conjunctive() -> None:
    reference = {
        "kernel_normalized_central_recovery": 0.90,
        "major_fwhm_over_kernel": 1.00,
        "minor_fwhm_over_kernel": 1.00,
        "centroid_error_arcsec": 0.01,
        "annular_residual_over_truth": 0.01,
        "kernel_residual_relative_rms": 0.10,
    }
    candidate = dict(reference)
    screen = {
        "max_absolute_recovery_error_degradation": 0.01,
        "max_width_fractional_error": 0.03,
        "max_centroid_error_arcsec": 0.1,
        "max_residual_ratio_to_reference": 1.1,
    }

    assert analysis.array_screen_pass(candidate, reference, screen)
    candidate["annular_residual_over_truth"] = 0.012
    assert not analysis.array_screen_pass(candidate, reference, screen)


def test_annular_mask_uses_declared_radii() -> None:
    class Variable:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)

        def __getitem__(self, key):
            return self.values[key]

    class DatasetStub:
        variables = {
            "fruit_feedback_n_rows": Variable([3]),
            "fruit_feedback_n_cols": Variable([3]),
            "fruit_feedback_wcs_cdelt": Variable([10.0, 10.0]),
            "fruit_feedback_wcs_crpix": Variable([1.0, 1.0]),
        }

    mask = analysis.annular_mask_from_checkpoint(DatasetStub(), 9.0, 11.0)

    assert np.array_equal(
        mask,
        np.array(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ]
        ),
    )
