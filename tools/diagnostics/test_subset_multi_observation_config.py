import tempfile
import unittest
from pathlib import Path

from subset_multi_observation_config import make_subset


def sample_config():
    return {
        "inputs": [
            {"meta": {"name": f"{obsnum}_0_2"}}
            for obsnum in ("100", "200", "300", "400")
        ],
        "runtime": {"output_dir": "/original/", "use_subdir": False},
    }


class SubsetMultiObservationConfigTest(unittest.TestCase):
    def test_selects_contiguous_history_through_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = make_subset(
                sample_config(),
                terminal_obsnum="400",
                output_dir=Path(temporary),
                history_count=2,
            )
        self.assertEqual(
            [item["meta"]["name"] for item in result["inputs"]],
            ["200_0_2", "300_0_2", "400_0_2"],
        )
        self.assertTrue(result["runtime"]["use_subdir"])
        self.assertTrue(result["runtime"]["output_dir"].endswith("/"))

    def test_repeats_one_observation_before_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = make_subset(
                sample_config(),
                terminal_obsnum="400",
                output_dir=Path(temporary),
                repeated_obsnum="200",
                repeat_count=3,
            )
        self.assertEqual(
            [item["meta"]["name"] for item in result["inputs"]],
            ["200_0_2", "200_0_2", "200_0_2", "400_0_2"],
        )

    def test_rejects_history_before_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "outside"):
                make_subset(
                    sample_config(),
                    terminal_obsnum="200",
                    output_dir=Path(temporary),
                    history_count=2,
                )


if __name__ == "__main__":
    unittest.main()
