from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.config import audit_kids_external_boundary as audit


REPO_ROOT = Path(__file__).resolve().parents[2]


class KidsExternalBoundaryAuditTest(unittest.TestCase):
    def test_current_boundary_is_exact(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertFalse(result["drift"])
        self.assertTrue(result["boundary"]["exact"])
        self.assertTrue(result["schema"]["exact"])
        self.assertEqual(
            result["boundary"]["type_counts"],
            {name: 1 for name in audit.EXPECTED_TYPES},
        )

    def test_inventory_records_unity_acceptance(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertTrue(result["inventory"]["exact"])
        self.assertEqual(
            result["inventory"]["domain"]["provenance_status"],
            "complete",
        )


if __name__ == "__main__":
    unittest.main()
