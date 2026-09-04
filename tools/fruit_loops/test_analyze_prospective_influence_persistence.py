from __future__ import annotations

import csv

import numpy as np
import pytest
from netCDF4 import Dataset

from tools.fruit_loops import analyze_prospective_influence_persistence as analysis


LEARNING_HEADER = ["record_type", "iter", "uid", "score"]


def write_learning(path, rows: list[list[str]], header=LEARNING_HEADER) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_learning_iteration_rows_compare_exact_registered_iteration(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    replay = tmp_path / "replay.csv"
    prior = [
        ["penalty", str(iteration), "10", str(iteration)] for iteration in range(4)
    ]
    current = [
        ["penalty", "4", "4460", "4"],
        ["application", "4", "4460", "0"],
    ]
    write_learning(reference, [*prior, *current])
    write_learning(replay, current)

    result = analysis.require_learning_iteration_rows(
        reference, replay, 4, {0, 1, 2, 3}
    )

    assert result["headers_identical"] is True
    assert result["reference_iteration_counts"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 2,
    }
    assert result["replay_iteration_counts"] == {"4": 2}
    assert result["ordered_raw_rows_identical"] is True


def test_learning_iteration_rows_reject_header_difference(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    replay = tmp_path / "replay.csv"
    write_learning(reference, [["penalty", "4", "4460", "4"]])
    write_learning(
        replay,
        [["penalty", "4", "4460", "4"]],
        ["record_type", "iter", "score", "uid"],
    )

    with pytest.raises(ValueError, match="headers differ"):
        analysis.require_learning_iteration_rows(reference, replay, 4, set())


def test_learning_iteration_rows_reject_replay_history(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    replay = tmp_path / "replay.csv"
    current = [["penalty", "4", "4460", "4"]]
    write_learning(reference, current)
    write_learning(replay, [["penalty", "3", "4460", "3"], *current])

    with pytest.raises(ValueError, match="outside the completed iteration"):
        analysis.require_learning_iteration_rows(reference, replay, 4, set())


@pytest.mark.parametrize(
    "replay_rows",
    [
        [
            ["application", "4", "4460", "0"],
            ["penalty", "4", "4460", "4"],
        ],
        [
            ["penalty", "4", "4460", "5"],
            ["application", "4", "4460", "0"],
        ],
    ],
)
def test_learning_iteration_rows_reject_order_or_field_change(
    tmp_path, replay_rows
) -> None:
    reference = tmp_path / "reference.csv"
    replay = tmp_path / "replay.csv"
    expected = [
        ["penalty", "4", "4460", "4"],
        ["application", "4", "4460", "0"],
    ]
    write_learning(reference, expected)
    write_learning(replay, replay_rows)

    with pytest.raises(ValueError, match="count, order, or raw fields"):
        analysis.require_learning_iteration_rows(reference, replay, 4, set())


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
