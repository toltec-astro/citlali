import tempfile
import unittest
from pathlib import Path

from tools.baseline import compare_reduction_products as compare


class CompareReductionProductsTest(unittest.TestCase):
    def test_accepts_oof_as_validation_mode(self) -> None:
        args = compare.parse_args(["--base-root", "/tmp/root", "--mode", "oof"])

        self.assertEqual(args.mode, "oof")

    def test_resolves_oof_base_root_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "oof" / "citlali" / "reduced" / "redu00"
            candidate = root / "oof" / "refactor" / "reduced" / "redu00"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            args = compare.parse_args(
                [
                    "--base-root",
                    str(root),
                    "--mode",
                    "oof",
                    "--baseline-redu",
                    "redu00",
                    "--candidate-redu",
                    "redu00",
                ]
            )

            self.assertEqual(
                compare.resolve_from_base_root(args),
                (baseline.resolve(), candidate.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
