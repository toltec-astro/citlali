#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "audit_sci_align_001_acquisition_event_semantics.py"
)
SPEC = importlib.util.spec_from_file_location("acquisition_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AcquisitionEventSemanticsAuditTest(unittest.TestCase):
    def test_integer_rows_are_even_half_steps(self):
        self.assertTrue(
            MODULE.labels_are_integer_row_compatible(
                {(148670, 151126): -2, (151126, 150819): -2}
            )
        )
        self.assertFalse(
            MODULE.labels_are_integer_row_compatible(
                {(148670, 150819): -3, (148670, 151126): -2}
            )
        )

    def test_frozen_labels_are_transitive(self):
        labels = {
            (148670, 150819): -3,
            (148670, 151126): -2,
            (150819, 151126): 1,
        }
        self.assertEqual(
            MODULE.transitive_states(labels, 148670),
            {148670: 0, 150819: -3, 151126: -2},
        )

    def test_inconsistent_labels_fail(self):
        labels = {
            (148670, 150819): -3,
            (148670, 151126): -2,
            (150819, 151126): 0,
        }
        with self.assertRaises(MODULE.AuditError):
            MODULE.transitive_states(labels, 148670)


if __name__ == "__main__":
    unittest.main()
