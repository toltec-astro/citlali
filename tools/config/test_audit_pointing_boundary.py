from __future__ import annotations

import unittest

from tools.config import audit_pointing_boundary as audit


class PointingBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_manifest(self) -> None:
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                "path_count": 5,
                "path_sha256": audit.EXPECTED_PATH_SHA256,
                "paths": audit.EXPECTED_PATHS,
            }
        )
        self.assertTrue(state["exact"])

    def test_rejects_manifest_drift(self) -> None:
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                "path_count": 5,
                "path_sha256": "wrong",
                "paths": audit.EXPECTED_PATHS,
            }
        )
        self.assertFalse(state["exact"])

    def test_accepts_direct_reader(self) -> None:
        source = "\n".join(
            [
                "bool read_optional_pointing_enum() {}",
                "read_optional_pointing_enum();",
                "read_optional_pointing_enum();",
                "read_optional_config_value();",
                "read_optional_config_value();",
                "read_optional_config_value();",
                *[
                    "const auto sample_key = std::tuple{" +
                    ", ".join(
                        f'\"{part}\"' for part in path.split(".")
                    ) +
                    "};"
                    for path in audit.EXPECTED_PATHS
                ],
            ]
        )
        self.assertTrue(audit.reader_state(source)["exact"])

    def test_rejects_mirrored_reader(self) -> None:
        source = "read_optional_mirrored_config_value();"
        self.assertFalse(audit.reader_state(source)["exact"])

    def test_accepts_one_way_authority_boundary(self) -> None:
        boundary = "\n".join(
            [
                "read_pointing_request_config();",
                "pointing_plan.reset_from_request();",
                "adapt_pointing_config_one_way();",
            ]
        )
        accessor = "return engine.pointing_plan.effective;"
        adapter = "const citlali::config::PointingConfig &effective"
        state = audit.authority_state(boundary, accessor, adapter, "")
        self.assertTrue(state["exact"])

    def test_rejects_bidirectional_reader(self) -> None:
        boundary = "\n".join(
            [
                "read_pointing_request_config();",
                "pointing_plan.reset_from_request();",
                "adapt_pointing_config_one_way();",
            ]
        )
        state = audit.authority_state(
            boundary,
            "return engine.pointing_plan.effective;",
            "const citlali::config::PointingConfig &effective",
            "read_pointing_source_strategy_config();",
        )
        self.assertFalse(state["exact"])


if __name__ == "__main__":
    unittest.main()
