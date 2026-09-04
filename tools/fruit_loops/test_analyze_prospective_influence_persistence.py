from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from tools.fruit_loops import analyze_prospective_influence_persistence as analysis


def test_persistence_summary_exact_scaled_case() -> None:
    d4 = np.array([[1.0, -2.0], [3.0, -4.0]])
    d5 = 2.0 * d4
    selected = np.ones_like(d4, dtype=bool)

    result = analysis.persistence_summary(d4, d5, selected)

    assert result["normalized_inner_product"] == 1.0
    assert result["pearson_correlation"] == 1.0
    assert result["spearman_rank_correlation"] == 1.0
    assert result["beta_d5_on_d4"] == 2.0
    assert result["scaled_residual_fraction"] == 0.0
    assert result["sign_agreement_fraction"] == 1.0
    assert abs(result["energy_identity_residual"]) < 1e-12


def test_stable_top_response_uses_row_major_tie_break() -> None:
    values = np.array([[5.0, -5.0], [4.0, 3.0]])
    selected = np.ones_like(values, dtype=bool)

    result = analysis.stable_top_indices(values, selected, 1)

    assert np.array_equal(result, [0])


def test_top_response_summary_reports_exact_population_fraction() -> None:
    d4 = np.arange(1.0, 11.0).reshape(2, 5)
    d5 = d4.copy()
    selected = np.ones_like(d4, dtype=bool)

    [result] = analysis.top_response_summary(d4, d5, selected, [0.1])

    assert result["selected_count"] == 1
    assert result["intersection_count"] == 1
    assert result["overlap_fraction_each_set"] == 1.0
    assert result["jaccard_fraction"] == 1.0
    assert result["d5_squared_response_captured_by_d4_selection"] == 100 / 385


def test_equal_netcdf_ignores_container_layout(tmp_path) -> None:
    left = tmp_path / "left.nc"
    right = tmp_path / "right.nc"
    for path, compression in ((left, 1), (right, 4)):
        with Dataset(path, "w", format="NETCDF4") as dataset:
            dataset.createDimension("row", 2)
            dataset.setncattr("identity", "same")
            variable = dataset.createVariable(
                "value", "f8", ("row",), zlib=True, complevel=compression
            )
            variable.setncattr("unit", "mJy/beam")
            variable[:] = [1.0, np.nan]

    result = analysis.require_equal_netcdf(left, right)

    assert result["structure_attributes_and_values_identical"] is True
    assert result["whole_file_hash_identical"] is False


def test_derive_iteration_maps_closes_deletion_identity() -> None:
    shape = (1, 2)
    planes = {
        "total_N": np.array([[20.0, 30.0]]),
        "total_C": np.array([[10.0, 10.0]]),
        "total_Q": np.array([[5.0, 5.0]]),
        "target_N": np.array([[4.0, 6.0]]),
        "target_C": np.array([[2.0, 2.0]]),
        "target_Q": np.array([[1.0, 1.0]]),
        "target_abs_C_terms": np.array([[2.0, 2.0]]),
        "total_abs_C_terms": np.array([[10.0, 10.0]]),
        "total_unique_detector_count": np.array([[5, 5]]),
    }
    for name in (
        "total_abs_N_terms",
        "target_abs_N_terms",
        "total_occurrence_pixel_count",
        "target_occurrence_pixel_count",
        "target_unique_detector_count",
        "formal_coefficient",
        "empirical_coefficient",
        "normalization_support",
        "science_policy_support",
    ):
        planes[name] = np.ones(shape)

    maps, _ = analysis.derive_iteration_maps(planes, 0.0)

    assert np.all(maps["conditioned"])
    assert np.allclose(maps["deletion_response"], maps["predicted_deletion"])
    assert np.max(np.abs(maps["deletion_identity_residual"])) < 1e-15
