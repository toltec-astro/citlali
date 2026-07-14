from __future__ import annotations

import unittest

from tools.config import audit_noise_products_boundary as audit


class NoiseProductsBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_manifest(self) -> None:
        state = audit.manifest_state({
            "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
            "path_count": 6,
            "path_sha256": audit.EXPECTED_PATH_SHA256,
            "paths": audit.EXPECTED_PATHS,
        })
        self.assertTrue(state["exact"])

    def test_rejects_manifest_drift(self) -> None:
        state = audit.manifest_state({
            "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
            "path_count": 6,
            "path_sha256": "wrong",
            "paths": audit.EXPECTED_PATHS,
        })
        self.assertFalse(state["exact"])

    def test_accepts_direct_reader(self) -> None:
        source = "\n".join([
            'read_config_value(std::tuple{"noise_maps", "enabled"});',
            'read_config_value(std::tuple{"noise_maps", "n_noise_maps"});',
            'read_config_value(std::tuple{"noise_maps", "randomize_dets"});',
            'read_optional_config_value(std::tuple{"noise_maps", "write_realizations"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "enabled"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "apply_empirical_weights"});',
        ])
        self.assertTrue(audit.reader_state(source)["exact"])

    def test_rejects_mirrored_reader(self) -> None:
        source = "\n".join([
            'read_config_value(std::tuple{"noise_maps", "enabled"});',
            'read_config_value(std::tuple{"noise_maps", "n_noise_maps"});',
            'read_config_value(std::tuple{"noise_maps", "randomize_dets"});',
            'read_optional_config_value(std::tuple{"noise_maps", "write_realizations"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "enabled"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "apply_empirical_weights"});',
            "read_mirrored_config_value();",
        ])
        self.assertFalse(audit.reader_state(source)["exact"])

    def test_rejects_overlapping_but_incomplete_reader_paths(self) -> None:
        source = "\n".join([
            'read_config_value(std::tuple{"noise_maps", "enabled"});',
            'read_config_value(std::tuple{"noise_maps", "n_noise_maps"});',
            'read_config_value(std::tuple{"noise_maps", "randomize_dets"});',
            'read_optional_config_value(std::tuple{"noise_maps", "write_realizations"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "apply_empirical_weights"});',
            'read_optional_config_value(std::tuple{"noise_maps", "products", "apply_empirical_weights"});',
        ])
        self.assertFalse(audit.reader_state(source)["exact"])

    def test_accepts_one_way_authority_boundary(self) -> None:
        boundary = "\n".join([
            "read_noise_request_config();",
            "noise_plan.reset_from_request();",
            "adapt_noise_config_one_way();",
        ])
        state = audit.authority_state(
            "", boundary, "return engine.noise_plan.effective;",
            "const citlali::config::NoiseConfig &effective"
        )
        self.assertTrue(state["exact"])

    def test_rejects_requested_config_mutation(self) -> None:
        boundary = "\n".join([
            "read_noise_request_config();",
            "noise_plan.reset_from_request();",
            "adapt_noise_config_one_way();",
        ])
        state = audit.authority_state(
            "set_noise_maps_enabled();", boundary,
            "return engine.noise_plan.effective;",
            "const citlali::config::NoiseConfig &effective"
        )
        self.assertFalse(state["exact"])

    def test_accepts_explicit_rng_identity(self) -> None:
        state = audit.rng_state([
            "boost::random::mt19937 eng{noise_random_seed};"
            for _ in range(3)
        ])
        self.assertTrue(state["exact"])

    def test_accepts_effective_execution_policy(self) -> None:
        state = audit.execution_policy_state(
            "noise_maps_active(noise_config)",
            "noise_config(*this)",
        )
        self.assertTrue(state["exact"])

    def test_rejects_requested_execution_policy_read(self) -> None:
        state = audit.execution_policy_state(
            "noise_maps_active(reduction_config.noise)",
            "noise_config(*this)",
        )
        self.assertFalse(state["exact"])


if __name__ == "__main__":
    unittest.main()
