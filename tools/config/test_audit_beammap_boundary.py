from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.config import audit_beammap_boundary as audit


REPO_ROOT = Path(__file__).resolve().parents[2]


class BeammapBoundaryAuditTest(unittest.TestCase):
    def test_current_characterization_is_exact(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertFalse(result["drift"])
        self.assertTrue(result["manifest"]["exact"])
        self.assertEqual(result["provenance"]["status"], "missing")

    def test_rejects_manifest_digest_drift(self) -> None:
        manifest = audit.load_manifest(REPO_ROOT / audit.MANIFEST_SOURCE)
        with patch.object(audit, "EXPECTED_PATH_SHA256", "wrong"):
            self.assertFalse(audit.manifest_state(manifest)["exact"])

    def test_rejects_default_surface_drift(self) -> None:
        manifest = audit.load_manifest(REPO_ROOT / audit.MANIFEST_SOURCE)
        changed = copy.deepcopy(manifest)
        changed["paths"] = [*changed["paths"], "beammap.unexpected"]
        state = audit.default_surface_state(REPO_ROOT, changed["paths"])
        self.assertFalse(state["exact"])
        self.assertEqual(state["missing_paths"], ["beammap.unexpected"])

    def test_boundary_has_one_typed_install_and_one_fitter_adapter(self) -> None:
        result = audit.audit(REPO_ROOT)["authority_boundary"]
        self.assertTrue(result["exact"])
        self.assertEqual(result["call_counts"], audit.EXPECTED_BOUNDARY_CALLS)
        self.assertTrue(result["adapter_exact"])

    def test_config_literals_remain_at_declared_boundary(self) -> None:
        result = audit.literal_boundary_state(REPO_ROOT)
        self.assertTrue(result["exact"])
        self.assertEqual(result["unexpected_files"], [])

    def test_inventory_does_not_claim_complete_provenance(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertTrue(result["inventory"]["exact"])
        self.assertTrue(result["provenance"]["expected_missing"])


if __name__ == "__main__":
    unittest.main()
