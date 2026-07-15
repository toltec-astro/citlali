from __future__ import annotations

import unittest
from pathlib import Path

from tools.config import audit_config_source_manifest as audit


REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigSourceManifestAuditTest(unittest.TestCase):
    def test_current_contract_is_exact(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertFalse(result["drift"])
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
