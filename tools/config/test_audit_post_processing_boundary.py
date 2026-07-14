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
