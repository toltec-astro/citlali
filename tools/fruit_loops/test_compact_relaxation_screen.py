from __future__ import annotations

import numpy as np
import pytest

from tools.fruit_loops import analyze_compact_relaxation_screen as analysis


def screen_manifest() -> dict:
    return {
        "test_id": "test",
        "screen": {
            "reference_alpha": 1.0,
            "reference_target_iteration": 5,
            "candidate_target_deadline_iteration": 4,
            "final_iteration": 6,
            "max_absolute_recovery_error_degradation": 0.01,
            "max_width_fractional_error": 0.03,
            "max_centroid_error_arcsec": 0.1,
            "max_residual_ratio_to_control": 1.10,
        },
    }


def row(alpha: float, iteration: int, array: str) -> dict:
    reference = alpha == 1.0
    return {
        "alpha": alpha,
        "iteration": iteration,
        "array": array,
        "kernel_normalized_central_recovery": (
            0.95 if reference and iteration == 5 else 0.97
        ),
        "major_fwhm_over_kernel": 1.01,
        "minor_fwhm_over_kernel": 0.99,
        "centroid_error_arcsec": 0.05,
        "annular_residual_over_truth": 0.02 if reference else 0.021,
        "kernel_residual_relative_rms": 0.04 if reference else 0.043,
    }


def promising_rows() -> list[dict]:
    rows = []
    for array in analysis.ARRAYS:
        rows.extend((row(1.0, 5, array), row(1.0, 6, array)))
        for iteration in range(1, 7):
            candidate = row(1.25, iteration, array)
            candidate["kernel_normalized_central_recovery"] = (
                0.96 if iteration >= 4 else 0.90
            )
            if iteration == 6:
                candidate["kernel_normalized_central_recovery"] = 0.98
            rows.append(candidate)
    return rows


def test_common_support_rejects_finite_mask_mismatch() -> None:
    with pytest.raises(ValueError, match="finite-support mismatch"):
        analysis.common_support(
            {
                "signal": np.array([[1.0, np.nan]]),
                "kernel": np.array([[1.0, 2.0]]),
            },
            context="test",
        )


def test_annulus_uses_geometric_map_center() -> None:
    mask = analysis.annulus_mask((5, 5), 1.0, 1.0, 1.0)
    assert int(mask.sum()) == 4
    assert not mask[2, 2]


def test_classifies_candidate_as_restart_pending_when_all_checks_pass() -> None:
    result = analysis.classify(promising_rows(), screen_manifest())

    assert result["classification"] == "restart_pending_for_promising_candidate"
    assert result["promising_candidates_requiring_restart"] == ["1.25"]


def test_classifies_candidate_not_promising_when_one_array_fails() -> None:
    rows = promising_rows()
    failed = analysis.select_row(rows, 1.25, 6, "a1400")
    failed["centroid_error_arcsec"] = 0.2

    result = analysis.classify(rows, screen_manifest())

    assert result["classification"] == "not_promising_on_this_compact_case"
    checks = result["candidate_outcomes"]["1.25"]["array_checks"]["a1400"]
    assert not checks["final_centroid_within_limit"]
