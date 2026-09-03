from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from tools.fruit_loops.analyze_off_source_injection import (
    classify_location_control,
    injection_specific_hard_penalties,
    write_complete_response_maps,
)
from tools.fruit_loops.compare_injected_source_pair import ARRAYS, product_path


def penalty(iteration: int, scan: int, uid: int, array: int, factor: float = 0.0):
    return {
        "iteration": iteration,
        "scan": scan,
        "uid": uid,
        "array": array,
        "score": 4.0,
        "factor": factor,
        "scan_local": True,
    }


def responses(loss_array: str | None = None) -> dict[str, dict]:
    return {
        array: {"registered_response_loss_direction": array == loss_array}
        for array in ("a1100", "a1400", "a2000")
    }


class OffSourceAnalysisTest(unittest.TestCase):
    target = {"iteration": 4, "scan": 5, "uid": 4460, "array": 1}

    def test_finds_only_injection_specific_hard_penalties(self) -> None:
        retained = penalty(4, 1, 100, 0)
        added = penalty(4, 5, 4460, 1)
        late = penalty(5, 2, 200, 2)
        soft = penalty(4, 3, 300, 2, 0.5)

        result = injection_specific_hard_penalties(
            [retained], [retained, added, late, soft], iteration=4, factor=0.0
        )

        self.assertEqual(result, [added])

    def test_classifies_same_event_only_with_response_loss(self) -> None:
        target = penalty(4, 5, 4460, 1)
        self.assertEqual(
            classify_location_control([target], responses("a1400"), self.target),
            "same_event_replicated_off_source",
        )
        self.assertEqual(
            classify_location_control([target], responses(), self.target),
            "inconclusive",
        )

    def test_classifies_absent_or_different_event(self) -> None:
        self.assertEqual(
            classify_location_control([], responses("a1400"), self.target),
            "centered_event_not_replicated",
        )
        other = penalty(4, 2, 99, 2)
        self.assertEqual(
            classify_location_control([other], responses("a2000"), self.target),
            "different_penalty_association",
        )

    def test_writes_every_complete_response_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"
            injected = root / "injected"
            for iteration in range(2):
                for array in ARRAYS:
                    for trajectory, value in ((control, 2.0), (injected, 5.5)):
                        path = product_path(
                            trajectory / f"redu{iteration:02d}", 123424, array
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        primary = fits.PrimaryHDU()
                        primary.header["HIERARCH FRUITLOOPS_ITER"] = iteration
                        signal = fits.ImageHDU(
                            np.full((3, 4), value), name="signal_I"
                        )
                        signal.header["CTYPE1"] = "AZOFFSET"
                        signal.header["CTYPE2"] = "ELOFFSET"
                        fits.HDUList([primary, signal]).writeto(path)
            manifest = {
                "test_id": "test",
                "obsnum": 123424,
                "trajectory_start_iteration": 0,
                "stop_iteration_exclusive": 2,
                "az_offset_arcsec": 0.0,
                "el_offset_arcsec": -60.0,
            }

            records = write_complete_response_maps(
                control, injected, root / "responses", manifest
            )

            self.assertEqual(len(records), 2 * len(ARRAYS))
            with fits.open(records[0]["path"]) as hdul:
                np.testing.assert_array_equal(hdul["RESPONSE_I"].data, 3.5)
                self.assertEqual(hdul[0].header["SCI.RESPONSE"], "injected-control")


if __name__ == "__main__":
    unittest.main()
