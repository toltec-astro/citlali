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
        self.assertEqual(
            result["provenance"]["status"], "required-atomic-v2"
        )

    def test_reader_covers_all_frozen_paths(self) -> None:
        state = audit.audit(REPO_ROOT)["reader_coverage"]
        self.assertTrue(state["exact"])
        self.assertEqual(state["root_count"], 59)
        self.assertEqual(state["covered_path_count"], 74)
        self.assertEqual(state["missing_paths"], [])
        self.assertEqual(state["extra_roots"], [])
        self.assertTrue(state["mutation_helpers_retired"])
        self.assertEqual(
            state["retired_mutation_helper_counts"],
            {helper: 0 for helper in audit.RETIRED_READER_MUTATION_HELPERS},
        )

    def test_serializer_covers_all_frozen_paths(self) -> None:
        state = audit.audit(REPO_ROOT)["serializer_coverage"]
        self.assertTrue(state["exact"])
        self.assertEqual(state["root_count"], 59)
        self.assertEqual(state["covered_path_count"], 74)
        self.assertEqual(state["missing_paths"], [])
        self.assertEqual(state["extra_roots"], [])

    def test_execution_plan_is_wired_with_compatibility_consumers(self) -> None:
        state = audit.audit(REPO_ROOT)["execution_plan"]
        self.assertTrue(state["exact"])
        self.assertEqual(
            state["status"], "wired-realized-provenance"
        )
        self.assertEqual(
            state["production_references"],
            state["expected_production_references"],
        )
        self.assertTrue(state["wired_at_boundary"])
        self.assertEqual(
            state["serializer_production_references"],
            [audit.PROVENANCE_SOURCE],
        )

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

    def test_provenance_and_lifecycle_are_required_and_exact(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertTrue(result["provenance"]["exact"])
        self.assertEqual(result["provenance"]["cli_write_count"], 1)
        self.assertEqual(result["provenance"]["cli_completion_count"], 1)
        self.assertTrue(result["provenance"]["completion_before_write"])
        self.assertTrue(result["lifecycle"]["exact"])
        self.assertEqual(
            result["lifecycle"]["call_counts"],
            audit.EXPECTED_LIFECYCLE_CALLS,
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
