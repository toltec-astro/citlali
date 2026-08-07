from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.config import audit_mapmaking_boundary as audit


class MapmakingBoundaryAuditTest(unittest.TestCase):
    def test_accepts_canonical_manifest(self) -> None:
        paths = sorted(
            audit.CORE_PATHS | audit.METHOD_PATHS | audit.OUTPUT_PATHS
        )
        state = audit.manifest_state(
            {
                "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                "path_count": len(paths),
                "path_sha256": audit.path_digest(paths),
                "paths": paths,
            }
        )
        self.assertTrue(state["exact"])

    def test_rejects_unsorted_or_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "manifest.json"
            source.write_text('{"paths":["mapmaking.z","mapmaking.a"]}')
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                audit.load_frozen_manifest(source)

    def test_accepts_exact_authority_boundary(self) -> None:
        lines = [
            "mapmaking_plan();",
            "read_mapmaking_enabled_config();",
            "read_map_grouping_config();",
            "read_map_method_config();",
            "read_map_pixel_axes_config();",
            "read_mapmaking_output_request_config();",
            "adapt_mapmaking_output_config_one_way();",
            "adapt_mapmaking_output_config_one_way();",
            "read_mapmaking_method_request_config();",
            "adapt_jinc_filter_config_one_way();",
            "adapt_maximum_likelihood_config_one_way();",
            "mapmaking_plan.reset_from_request();",
        ]
        self.assertTrue(audit.authority_boundary("\n".join(lines))["exact"])

    def test_rejects_duplicate_or_misordered_boundary_calls(self) -> None:
        duplicate = "\n".join(
            [
                "read_mapmaking_enabled_config();",
                "read_mapmaking_enabled_config();",
            ]
        )
        self.assertFalse(audit.authority_boundary(duplicate)["exact"])
        misordered = "\n".join(
            [
                "adapt_mapmaking_output_config_one_way();",
                "read_mapmaking_output_request_config();",
            ]
        )
        self.assertFalse(audit.authority_boundary(misordered)["exact"])

    def test_reader_coverage_reports_missing_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            for source in (
                audit.CORE_READER_SOURCE,
                audit.METHOD_SOURCE,
                audit.MODEL_SOURCE,
                audit.OUTPUT_SOURCE,
            ):
                path = repo_root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            result = audit.reader_coverage(repo_root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["covered_count"], 0)

    def test_manifest_digest_detects_drift(self) -> None:
        paths = sorted(
            audit.CORE_PATHS | audit.METHOD_PATHS | audit.OUTPUT_PATHS
        )
        with patch.object(audit, "EXPECTED_PATH_SHA256", "wrong"):
            state = audit.manifest_state(
                {
                    "schema_version": audit.EXPECTED_MANIFEST_SCHEMA,
                    "path_count": len(paths),
                    "path_sha256": audit.path_digest(paths),
                    "paths": paths,
                }
            )
        self.assertFalse(state["exact"])

    def test_accepts_science_map_product_contract_surface(self) -> None:
        registry = {
            "schema_version": audit.EXPECTED_PRODUCT_REGISTRY_SCHEMA,
            "science_map_contracts": {
                audit.EXPECTED_SCIENCE_MAP_CONTRACT: {
                    "audit_state": audit.EXPECTED_SCIENCE_MAP_AUDIT_STATE,
                    "planes": [
                        {"name": name}
                        for name in sorted(audit.EXPECTED_SCIENCE_MAP_PLANES)
                    ],
                    "aliases": {
                        "coverage_I": {
                            "canonical": "retained_exposure_I",
                            "relationship": "bitwise_equal",
                        },
                        "coverage_bool_I": {
                            "canonical": "science_policy_support_I",
                            "relationship": "bitwise_equal",
                            "deprecated": True,
                            "validity_authority": False,
                        },
                    },
                }
            },
            "contracts": [
                {"contract_id": contract_id}
                for contract_id in audit.EXPECTED_SUCCESSOR_PRODUCT_CONTRACTS
            ],
        }

        self.assertTrue(audit.product_contract_state(registry)["exact"])

    def test_rejects_science_map_alias_promotion(self) -> None:
        registry = {
            "schema_version": audit.EXPECTED_PRODUCT_REGISTRY_SCHEMA,
            "science_map_contracts": {
                audit.EXPECTED_SCIENCE_MAP_CONTRACT: {
                    "audit_state": audit.EXPECTED_SCIENCE_MAP_AUDIT_STATE,
                    "planes": [
                        {"name": name}
                        for name in sorted(audit.EXPECTED_SCIENCE_MAP_PLANES)
                    ],
                    "aliases": {
                        "coverage_I": {
                            "canonical": "retained_exposure_I",
                            "relationship": "bitwise_equal",
                        },
                        "coverage_bool_I": {
                            "canonical": "science_policy_support_I",
                            "relationship": "bitwise_equal",
                            "deprecated": True,
                            "validity_authority": True,
                        },
                    },
                }
            },
            "contracts": [
                {"contract_id": contract_id}
                for contract_id in audit.EXPECTED_SUCCESSOR_PRODUCT_CONTRACTS
            ],
        }

        self.assertFalse(audit.product_contract_state(registry)["exact"])

    def test_rejects_pre_acceptance_science_map_audit_state(self) -> None:
        registry = {
            "schema_version": audit.EXPECTED_PRODUCT_REGISTRY_SCHEMA,
            "science_map_contracts": {
                audit.EXPECTED_SCIENCE_MAP_CONTRACT: {
                    "audit_state": "addressed_pending_reaudit",
                    "planes": [
                        {"name": name}
                        for name in sorted(audit.EXPECTED_SCIENCE_MAP_PLANES)
                    ],
                    "aliases": {
                        "coverage_I": {
                            "canonical": "retained_exposure_I",
                            "relationship": "bitwise_equal",
                        },
                        "coverage_bool_I": {
                            "canonical": "science_policy_support_I",
                            "relationship": "bitwise_equal",
                            "deprecated": True,
                            "validity_authority": False,
                        },
                    },
                }
            },
            "contracts": [
                {"contract_id": contract_id}
                for contract_id in audit.EXPECTED_SUCCESSOR_PRODUCT_CONTRACTS
            ],
        }

        self.assertFalse(audit.product_contract_state(registry)["exact"])


if __name__ == "__main__":
    unittest.main()
