from __future__ import annotations

import unittest

from tools.config import audit_coadd_boundary as audit


class CoaddBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_manifest(self) -> None:
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                "path_count": 1,
                "path_sha256": audit.EXPECTED_PATH_SHA256,
                "paths": audit.EXPECTED_PATHS,
            }
        )
        self.assertTrue(state["exact"])

    def test_rejects_manifest_drift(self) -> None:
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                "path_count": 1,
                "path_sha256": "wrong",
                "paths": audit.EXPECTED_PATHS,
            }
        )
        self.assertFalse(state["exact"])

    def test_accepts_direct_reader(self) -> None:
        source = 'read_config_value(config, request.enabled, d, std::tuple{"coadd", "enabled"});'
        self.assertTrue(audit.reader_state(source)["exact"])

    def test_rejects_mirrored_reader(self) -> None:
        source = 'read_mirrored_config_value(std::tuple{"coadd", "enabled"});'
        self.assertFalse(audit.reader_state(source)["exact"])

    def test_accepts_one_way_authority_boundary(self) -> None:
        boundary = "\n".join(
            [
                "read_coadd_request_config();",
                "coadd_plan.reset_from_request();",
            ]
        )
        accessor = "return engine.coadd_plan.effective;"
        state = audit.authority_state("", boundary, accessor)
        self.assertTrue(state["exact"])

    def test_rejects_requested_config_mutation(self) -> None:
        boundary = "\n".join(
            [
                "read_coadd_request_config();",
                "coadd_plan.reset_from_request();",
            ]
        )
        state = audit.authority_state(
            "set_coadd_enabled();",
            boundary,
            "return engine.coadd_plan.effective;",
        )
        self.assertFalse(state["exact"])


if __name__ == "__main__":
    unittest.main()
