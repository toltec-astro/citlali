from __future__ import annotations

import numpy as np

from tools.fruit_loops import analyze_jinc_accounting as analysis


class FakeVariable:
    def __init__(self, value: object) -> None:
        self.value = value

    def __getitem__(self, key: object) -> object:
        return self.value


class FakeDataset:
    def __init__(self, value: object) -> None:
        self.variables = {"value": FakeVariable(value)}


def test_scalar_returns_native_string() -> None:
    dataset = FakeDataset(np.asarray(["schema-v1"], dtype=object))

    assert analysis.scalar(dataset, "value") == "schema-v1"


def test_scalar_decodes_bytes() -> None:
    dataset = FakeDataset(np.asarray([b"schema-v1"], dtype=object))

    assert analysis.scalar(dataset, "value") == "schema-v1"


def test_scalar_unwraps_numpy_numeric_scalar() -> None:
    dataset = FakeDataset(np.asarray([np.int64(17)]))

    value = analysis.scalar(dataset, "value")

    assert value == 17
    assert isinstance(value, int)


def test_receipt_plane_reverses_internal_columns_to_fits_orientation() -> None:
    dataset = FakeDataset(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    result = analysis.receipt_plane(dataset, "value")

    assert np.array_equal(result, [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])


def test_positive_order_threshold_matches_registered_algorithm() -> None:
    values = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])

    threshold, count, index = analysis.positive_order_threshold(values, 0.1)

    assert count == 4
    assert index == 3
    assert threshold == 0.4


def test_finalize_jinc_handles_signed_cancelled_and_small_denominators() -> None:
    numerator = np.array([[2.0, 2.0, 2.0, 2.0]])
    denominator = np.array([[2.0, -2.0, 0.0, 1e-9]])
    variance = np.full((1, 4), 4.0)

    result = analysis.finalize_jinc(numerator, denominator, variance, 0.0)

    assert np.array_equal(result["support"], [[True, True, False, False]])
    assert np.array_equal(result["signal"], [[1.0, -1.0, 0.0, 0.0]])
    assert np.array_equal(result["coefficient"], [[1.0, 1.0, 0.0, 0.0]])


def test_accumulator_bound_includes_total_target_without_and_subtraction() -> None:
    unit_roundoff = 2.0**-53
    bound = analysis.accumulator_difference_bound(
        np.array([[10.0]]),
        np.array([[2.0]]),
        np.array([[12.0]]),
        np.array([[2.0]]),
        np.array([[5]]),
        np.array([[1]]),
        unit_roundoff,
    )

    expected = (
        analysis.gamma(np.array([[5]]), unit_roundoff) * 12.0
        + analysis.gamma(np.array([[1]]), unit_roundoff) * 2.0
        + analysis.gamma(np.array([[4]]), unit_roundoff) * 10.0
        + unit_roundoff * 12.0
    )
    assert np.array_equal(bound, expected)


def test_binned_response_is_deterministic_and_complete() -> None:
    predictor = np.arange(12.0).reshape(3, 4)
    response = predictor * 2.0
    support = np.ones((3, 4), dtype=bool)

    rows = analysis.binned_response(predictor, response, support, 3)

    assert [row["count"] for row in rows] == [4, 4, 4]
    assert sum(row["count"] for row in rows) == 12
    assert rows[0]["predictor_min"] == 0.0
    assert rows[-1]["predictor_max"] == 11.0
