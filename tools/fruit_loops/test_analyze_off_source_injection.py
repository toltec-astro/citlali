from __future__ import annotations

import unittest

from tools.fruit_loops.analyze_off_source_injection import (
    classify_location_control,
    injection_specific_hard_penalties,
)


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


if __name__ == "__main__":
    unittest.main()

