from __future__ import annotations

import unittest

import numpy as np

from tools.fruit_loops import stratify_pointing_quality as quality


class PointingQualityStratificationTest(unittest.TestCase):
    def test_badness_direction_and_nonfinite_policy(self) -> None:
        high_good = quality.badness_percentile(
            np.asarray([1.0, 2.0, 3.0, np.nan]),
            higher_is_better=True,
        )
        low_good = quality.badness_percentile(
            np.asarray([1.0, 2.0, 3.0, np.nan]),
            higher_is_better=False,
        )

        np.testing.assert_allclose(high_good[:3], [1.0, 0.5, 0.0])
        np.testing.assert_allclose(low_good[:3], [0.0, 0.5, 1.0])
        self.assertEqual(high_good[3], 1.0)
        self.assertEqual(low_good[3], 1.0)

    def test_invalid_positive_ratio_becomes_missing(self) -> None:
        self.assertTrue(np.isnan(quality.absolute_log_ratio(0.0)))
        self.assertTrue(np.isnan(quality.absolute_log_ratio(1.0, np.nan)))
        self.assertAlmostEqual(
            quality.absolute_log_ratio(2.0, 1.0), np.log(2.0)
        )

    def test_observation_score_penalizes_one_bad_array(self) -> None:
        rows = []
        for obsnum, array_badness, centroid in (
            (1, [0.1, 0.1, 0.1], [(0.0, 0.0)] * 3),
            (
                2,
                [0.1, 0.1, 0.9],
                [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ),
            (
                3,
                [0.9, 0.9, 0.9],
                [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
            ),
        ):
            for array, badness, (x, y) in zip(
                quality.ARRAYS, array_badness, centroid, strict=True,
            ):
                rows.append(
                    {
                        "obsnum": obsnum,
                        "source": "test",
                        "observation_date": "2026-01-01",
                        "mjd": 61000.0,
                        "array": array,
                        "array_quality_badness": badness,
                        "fit_x_arcsec": x,
                        "fit_y_arcsec": y,
                        "fit_s2n": 10.0,
                        "amplitude_to_background_sigma": 10.0,
                        "measured_to_kernel_fwhm_ratio": 1.0,
                        "fit_axis_ratio": 1.0,
                        "map_roughness_fraction": 0.1,
                    }
                )

        observations = quality.observation_rows(rows)
        by_obsnum = {row["obsnum"]: row for row in observations}

        self.assertLess(
            by_obsnum[1]["quality_score"],
            by_obsnum[2]["quality_score"],
        )
        self.assertLess(
            by_obsnum[2]["quality_score"],
            by_obsnum[3]["quality_score"],
        )

    def test_108_observations_receive_frozen_stratum_counts(self) -> None:
        rows = []
        for obsnum in range(100000, 100108):
            badness = (obsnum - 100000) / 107.0
            for array_index, array in enumerate(quality.ARRAYS):
                rows.append(
                    {
                        "obsnum": obsnum,
                        "source": "test",
                        "observation_date": "2026-01-01",
                        "mjd": 61000.0,
                        "array": array,
                        "array_quality_badness": badness,
                        "fit_x_arcsec": float(array_index) * badness,
                        "fit_y_arcsec": 0.0,
                        "fit_s2n": 10.0,
                        "amplitude_to_background_sigma": 10.0,
                        "measured_to_kernel_fwhm_ratio": 1.0,
                        "fit_axis_ratio": 1.0,
                        "map_roughness_fraction": 0.1,
                    }
                )

        observations = quality.observation_rows(rows)
        counts = {
            stratum: sum(
                row["quality_stratum"] == stratum
                for row in observations
            )
            for stratum in ("normal", "marginal", "stress")
        }

        self.assertEqual(
            counts, {"normal": 54, "marginal": 38, "stress": 16}
        )


if __name__ == "__main__":
    unittest.main()
