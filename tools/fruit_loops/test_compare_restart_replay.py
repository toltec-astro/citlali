from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from netCDF4 import Dataset

from tools.fruit_loops import compare_restart_replay as compare


class RestartReplayComparisonTest(unittest.TestCase):
    @staticmethod
    def write_product(
        path: Path, iteration: int, signal: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        primary = fits.PrimaryHDU()
        primary.header["HIERARCH FRUITLOOPS_ITER"] = iteration
        hdus = [primary]
        for extension, value in (
            ("signal_I", signal),
            ("kernel_I", 2.0),
            ("weight_I", 3.0),
        ):
            hdus.append(
                fits.ImageHDU(
                    np.full((3, 4), value, dtype=float), name=extension
                )
            )
        fits.HDUList(hdus).writeto(path)

    @staticmethod
    def write_checkpoint(path: Path, uids: list[int]) -> None:
        with Dataset(path, "w") as dataset:
            dataset.createDimension("count_dim", 1)
            dataset.createDimension("observation", 1)
            dataset.createDimension("effective_detector_penalty", len(uids))
            count = dataset.createVariable(
                "effective_detector_penalty_count", "i8", ("count_dim",)
            )
            count[:] = [len(uids)]
            obs = dataset.createVariable("observation_id", str, ("observation",))
            obs[0] = "42"
            fields = {
                "penalty_observation_index": ("i4", [0] * len(uids)),
                "penalty_producer": (str, ["mapdiag:raw_obs"] * len(uids)),
                "penalty_reason": (
                    str, ["map_pixel_outlier_detector_dominance"] * len(uids)
                ),
                "penalty_iteration": ("i4", list(range(len(uids)))),
                "penalty_scan": ("i4", [2] * len(uids)),
                "penalty_uid": ("i4", uids),
                "penalty_network": ("i4", [-1] * len(uids)),
                "penalty_array": ("i4", [0] * len(uids)),
                "penalty_factor": ("f8", [0.0] * len(uids)),
                "penalty_score": ("f8", [8.0] * len(uids)),
                "penalty_scan_local": ("i4", [1] * len(uids)),
            }
            for name, (dtype, values) in fields.items():
                variable = dataset.createVariable(
                    name, dtype, ("effective_detector_penalty",)
                )
                if dtype is str:
                    for index, value in enumerate(values):
                        variable[index] = value
                else:
                    variable[:] = values

    def test_reports_first_delayed_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            replay = root / "replay"
            for iteration in (5, 6):
                for array in compare.ARRAYS:
                    self.write_product(
                        compare.product_path(
                            reference / f"redu{iteration:02d}", 42, array
                        ),
                        iteration,
                        float(iteration),
                    )
                    self.write_product(
                        compare.product_path(
                            replay / f"redu{iteration:02d}", 42, array
                        ),
                        iteration,
                        float(iteration),
                    )
            with fits.open(
                compare.product_path(replay / "redu06", 42, "a1100"),
                mode="update",
            ) as hdul:
                hdul["signal_I"].data[0, 0] += 1.0

            rows = compare.comparison_rows(reference, replay, 42)

            differences = [row for row in rows if not row["exact"]]
            self.assertEqual(len(differences), 1)
            self.assertEqual(differences[0]["iteration"], 6)
            self.assertEqual(differences[0]["array"], "a1100")
            self.assertEqual(differences[0]["extension"], "signal_I")

    def test_reports_missing_effective_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.nc"
            replay = root / "replay.nc"
            self.write_checkpoint(reference, [987, 1489])
            self.write_checkpoint(replay, [987])

            result = compare.checkpoint_comparison(reference, replay)

            self.assertIn(
                "effective_detector_penalty_count",
                result["differing_variables"],
            )
            self.assertEqual(
                [row["uid"] for row in result["reference_penalties"]],
                [987, 1489],
            )
            self.assertEqual(
                [row["uid"] for row in result["replay_penalties"]], [987]
            )

    def test_compares_every_replayed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = {}
            replay = {}
            for iteration in (5, 6, 7):
                reference[iteration] = root / "reference" / f"redu{iteration:02d}"
                replay[iteration] = root / "replay" / f"redu{iteration:02d}"
                reference[iteration].mkdir(parents=True)
                replay[iteration].mkdir(parents=True)
                self.write_checkpoint(
                    reference[iteration] / "citlali_restart_checkpoint.nc",
                    [987, 1489],
                )
                self.write_checkpoint(
                    replay[iteration] / "citlali_restart_checkpoint.nc",
                    [987, 1489] if iteration != 7 else [987],
                )

            results = compare.checkpoint_trajectory_comparisons(
                reference, replay
            )

            self.assertEqual(
                [result["iteration"] for result in results], [5, 6, 7]
            )
            self.assertEqual(
                [result["exact"] for result in results], [True, True, False]
            )
            self.assertIn(
                "effective_detector_penalty_count",
                results[-1]["differing_variables"],
            )


if __name__ == "__main__":
    unittest.main()
