from __future__ import annotations

import unittest
from pathlib import Path

from tools.config import audit_astrometry_boundary as audit


class AstrometryBoundaryAuditTest(unittest.TestCase):
    def test_counts_exact_tokens(self) -> None:
        self.assertEqual(audit.count("f(); f ();", "f("), 1)

    def test_repo_boundary_is_exact(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        result = audit.audit(repo_root)
        self.assertFalse(result["drift"])
        self.assertTrue(result["boundary"]["exact"])
        self.assertTrue(result["provenance"]["origin_not_overclaimed"])


if __name__ == "__main__":
    unittest.main()
