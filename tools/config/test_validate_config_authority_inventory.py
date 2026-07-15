from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.config import validate_config_authority_inventory as inventory


class ConfigAuthorityInventoryValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        loader = self.repo_root / "loader.h"
        loader.write_text("// test loader\n")
        self.data = {
            "schema_version": inventory.SCHEMA_VERSION,
            "contract": {
                "target_adapter_direction": (
                    "requested_yaml -> typed_config -> legacy_runtime"
                )
            },
            "domains": [
                {
                    "id": "raw-timestream",
                    "config_prefixes": ["timestream.raw_time_chunk"],
                    "typed_owner": "RawTimeChunkConfig",
                    "loader": "loader.h",
                    "execution_authority": "legacy",
                    "legacy_targets": ["RTCProc"],
                    "adapter_direction": "legacy-to-typed",
                    "migration_status": (
                        "legacy-authoritative-with-typed-mirror"
                    ),
                    "provenance_status": "partial",
                    "exit_gate": "Migrate authority.",
                }
            ],
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_accepts_explicit_legacy_authority_characterization(self) -> None:
        self.assertEqual(inventory.validate(self.data, self.repo_root), [])

    def test_accepts_legacy_authority_without_typed_mirror(self) -> None:
        domain = self.data["domains"][0]
        domain["legacy_targets"] = []
        domain["adapter_direction"] = "none"
        domain["migration_status"] = "legacy-authoritative"

        self.assertEqual(inventory.validate(self.data, self.repo_root), [])

    def test_rejects_legacy_authority_with_forward_adapter_label(self) -> None:
        domain = self.data["domains"][0]
        domain["adapter_direction"] = "typed-to-legacy"

        errors = inventory.validate(self.data, self.repo_root)

        self.assertIn(
            "domains[0]: legacy authority requires no adapter or a legacy-to-typed mirror",
            errors,
        )

    def test_rejects_legacy_mirror_without_legacy_target(self) -> None:
        self.data["domains"][0]["legacy_targets"] = []

        errors = inventory.validate(self.data, self.repo_root)

        self.assertIn(
            "domains[0]: legacy-to-typed mirror requires legacy_targets",
            errors,
        )

    def test_rejects_migration_label_that_disagrees_with_authority(self) -> None:
        domain = self.data["domains"][0]
        domain["execution_authority"] = "mixed"
        domain["adapter_direction"] = "typed-to-legacy"
        domain["migration_status"] = "typed-authoritative-with-adapter"

        errors = inventory.validate(self.data, self.repo_root)

        self.assertIn(
            "domains[0]: migration status "
            "'typed-authoritative-with-adapter' requires "
            "execution_authority='typed' and "
            "adapter_direction='typed-to-legacy'",
            errors,
        )

    def test_accepts_typed_authority_with_one_way_adapter(self) -> None:
        domain = self.data["domains"][0]
        domain["execution_authority"] = "typed"
        domain["adapter_direction"] = "typed-to-legacy"
        domain["migration_status"] = "typed-authoritative-with-adapter"

        self.assertEqual(inventory.validate(self.data, self.repo_root), [])


if __name__ == "__main__":
    unittest.main()
