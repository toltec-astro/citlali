#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import yaml

from tools.config import audit_post_processing_boundary as audit


REPO_ROOT = Path(__file__).resolve().parents[2]


class PostProcessingBoundaryAuditTest(unittest.TestCase):
    def test_current_characterization_is_exact(self) -> None:
        self.assertFalse(audit.audit(REPO_ROOT)["drift"])

    def test_rejects_manifest_digest_drift(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / audit.MANIFEST_SOURCE).read_text()
        )
        manifest["path_sha256"] = "wrong"
        self.assertFalse(audit.manifest_state(manifest)["exact"])

    def test_rejects_missing_wiener_filter_prefix(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / audit.MANIFEST_SOURCE).read_text()
        )
        manifest["config_prefixes"] = ["post_processing"]
        self.assertFalse(audit.manifest_state(manifest)["exact"])

    def test_typed_request_has_no_known_path_gaps(self) -> None:
        result = audit.audit(REPO_ROOT)
        self.assertEqual(result["manifest"]["known_typed_gaps"], [])
        self.assertTrue(
            result["mixed_boundary"]["checks"]
            ["complete_request_reader_present"]
        )
        self.assertTrue(
            result["mixed_boundary"]["checks"]["source_model_typed"]
        )
        self.assertTrue(
            result["mixed_boundary"]["checks"]["kernel_tail_typed"]
        )

    def test_map_filter_uses_one_way_effective_adapter(self) -> None:
        checks = audit.audit(REPO_ROOT)["mixed_boundary"]["checks"]
        self.assertEqual(checks["direct_request_reader_call_count"], 1)
        self.assertEqual(checks["shadow_comparison_call_count"], 1)
        self.assertTrue(checks["shadow_report_present"])
        self.assertTrue(checks["execution_plan_present"])
        self.assertEqual(checks["execution_plan_reset_call_count"], 1)
        self.assertEqual(checks["execution_plan_accessor_count"], 2)
        self.assertTrue(checks["serial_filter_parser_retired"])
        self.assertTrue(checks["omp_filter_parser_retired"])
        self.assertEqual(checks["legacy_filter_boundary_call_count"], 0)
        self.assertEqual(checks["reverse_filter_mirror_call_count"], 0)
        self.assertTrue(checks["typed_filter_adapter_present"])
        self.assertEqual(checks["typed_filter_adapter_call_count"], 1)
        self.assertEqual(checks["effective_filter_accessor_call_count"], 1)
        self.assertTrue(checks["filter_output_policy_is_effective"])
        self.assertFalse(checks["reverse_filter_mirror_present"])

    def test_source_finding_uses_one_way_effective_adapter(self) -> None:
        checks = audit.audit(REPO_ROOT)["mixed_boundary"]["checks"]
        self.assertTrue(checks["source_finding_parser_retired"])
        self.assertEqual(checks["source_finding_parser_call_count"], 0)
        self.assertTrue(checks["typed_source_finding_adapter_present"])
        self.assertEqual(
            checks["typed_source_finding_adapter_call_count"], 1
        )
        self.assertTrue(checks["effective_source_finding_policy_used"])
        self.assertTrue(
            checks["source_finding_output_policy_is_effective"]
        )
        self.assertTrue(checks["source_finding_shadow_details_retired"])
        self.assertFalse(checks["source_finding_reverse_mirror_present"])

    def test_rejects_default_surface_drift(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / audit.MANIFEST_SOURCE).read_text()
        )
        config = yaml.safe_load(
            (REPO_ROOT / audit.DEFAULT_CONFIG_SOURCE).read_text()
        )
        changed = copy.deepcopy(config)
        changed["post_processing"]["unexpected"] = True
        state = audit.default_surface_state(changed, manifest["paths"])
        self.assertFalse(state["exact"])
        self.assertEqual(
            state["extra_paths"], ["post_processing.unexpected"]
        )


if __name__ == "__main__":
    unittest.main()
