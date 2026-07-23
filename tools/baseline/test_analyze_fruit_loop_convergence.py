import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from tools.baseline import analyze_fruit_loop_convergence as convergence


class AnalyzeFruitLoopConvergenceTest(unittest.TestCase):
    def test_requires_two_consecutive_stable_transitions(self) -> None:
        with self._study() as manifest:
            result = convergence.analyze(manifest)

        self.assertTrue(result["protocol_complete"])
        rule = result["candidate_rule_results"][0]
        self.assertEqual(rule["stop_iteration_id"], 2)
        self.assertEqual(rule["saved_iteration_count"], 1)
        self.assertEqual(rule["estimated_saved_seconds"], 10.0)

    def test_reports_unexpected_config_difference(self) -> None:
        with self._study(changed_config=True) as manifest:
            result = convergence.analyze(manifest)

        self.assertFalse(result["protocol_complete"])
        self.assertIn("config differs", result["protocol_errors"][0])

    class _Study:
        def __init__(self, changed_config: bool) -> None:
            self.changed_config = changed_config
            self.directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.directory = tempfile.TemporaryDirectory()
            root = Path(self.directory.name)
            entries = []
            values = (1.0, 1.001, 1.002, 1.003)
            for iteration_id, value in enumerate(values):
                reduction = root / f"redu{iteration_id:02d}"
                maps = reduction / "coadded" / "raw"
                maps.mkdir(parents=True)
                config = {
                    "runtime": {"output_dir": str(root)},
                    "mapmaking": {
                        "method": (
                            "changed"
                            if self.changed_config and iteration_id == 2
                            else "naive"
                        )
                    },
                }
                (reduction / "citlali_test.yaml").write_text(
                    yaml.safe_dump(config), encoding="utf-8"
                )
                for array in ("a1100", "a1400", "a2000"):
                    self._write_fits(maps / f"test_{array}_citlali.fits", value)
                with gzip.open(reduction / "citlali.log.gz", "wt") as stream:
                    stream.write(
                        "[info] citlali version: test-version\n"
                        "[info] reduction learning finalize: enabled=1 "
                        f"iter={iteration_id} phase=apply reduction_type=science "
                        "source_model_available=1 "
                        "effective_sample_mask_intervals=10 "
                        "effective_detector_penalties=2\n"
                        "[info] profile stage=reduction.iteration "
                        f"context=fruit_iter={iteration_id} elapsed_s=10.0\n"
                    )
                entries.append(
                    {
                        "iteration_id": iteration_id,
                        "reduction_dir": reduction.name,
                    }
                )
            manifest = {
                "schema_version": convergence.MANIFEST_SCHEMA_VERSION,
                "study_id": "unit-test",
                "config_ignore_paths": ["runtime.output_dir"],
                "product": {
                    "path_pattern": "coadded/raw/*_{array}_citlali.fits",
                    "arrays": ["a1100", "a1400", "a2000"],
                    "signal_hdu": "signal_I",
                    "weight_hdu": "weight_I",
                    "coverage_hdu": "coverage_bool_I",
                    "aperture_center": "map_center",
                    "aperture_radius_arcsec": 2.0,
                },
                "iterations": entries,
                "candidate_rules": [
                    {
                        "rule_id": "test-rule",
                        "minimum_completed_iteration_id": 1,
                        "consecutive_passes": 2,
                        "require_learning_state_stable": True,
                        "support_jaccard_min": 1.0,
                        "map_relative_l2_delta_max": 0.01,
                        "aperture_relative_l2_delta_max": 0.01,
                        "peak_fractional_change_max": 0.01,
                        "weight_relative_l2_delta_max": 0.01,
                    }
                ],
            }
            path = root / "study.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

        @staticmethod
        def _write_fits(path: Path, value: float) -> None:
            shape = (8, 8)
            signal = np.full(shape, value, dtype=np.float64)
            weight = np.ones(shape, dtype=np.float64)
            coverage = np.ones(shape, dtype=np.uint8)
            header = fits.Header({"BUNIT": "mJy/beam", "CDELT1": 1.0 / 3600.0})
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(signal, header=header, name="signal_I"),
                    fits.ImageHDU(
                        weight,
                        header=fits.Header({"BUNIT": "1/(mJy/beam)^2"}),
                        name="weight_I",
                    ),
                    fits.ImageHDU(coverage, name="coverage_bool_I"),
                ]
            ).writeto(path)

    def _study(self, changed_config: bool = False) -> _Study:
        return self._Study(changed_config)


if __name__ == "__main__":
    unittest.main()
