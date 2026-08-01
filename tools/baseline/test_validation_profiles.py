import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.baseline import validate_product_contract as contracts
from tools.baseline import validation_profiles as profiles


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "validation/validation_profiles.json"
LEDGER = REPO_ROOT / "validation/accepted_runs.json"
PRODUCT_CONTRACTS = REPO_ROOT / "validation/product_contracts.json"
BINDING_POLICIES = REPO_ROOT / "validation/config_binding_policies.json"


class ValidationProfilesTest(unittest.TestCase):
    def test_checked_in_registry_is_valid_and_complete(self) -> None:
        registry = profiles.validate_registry(REGISTRY, LEDGER)

        active = [
            profile
            for profile in registry["profiles"]
            if profile["status"] == "active"
            and profile["epoch_id"] == registry["active_epoch_id"]
        ]
        self.assertEqual({profile["mode"] for profile in active}, profiles.SUPPORTED_MODES)
        self.assertEqual(len(active), len(profiles.SUPPORTED_MODES))
        self.assertGreaterEqual(len(registry["preparing_epoch_ids"]), 2)
        self.assertIn(
            "phase5-v2.1-candidate-2026-07-24",
            registry["preparing_epoch_ids"],
        )
        self.assertIn(
            "sci-map-001-repair-2026-07-31",
            registry["preparing_epoch_ids"],
        )
        for epoch_id in registry["preparing_epoch_ids"]:
            preparing = [
                profile
                for profile in registry["profiles"]
                if profile["status"] == "preparing"
                and profile["epoch_id"] == epoch_id
            ]
            self.assertEqual(
                {profile["mode"] for profile in preparing},
                profiles.SUPPORTED_MODES,
            )
            self.assertTrue(
                all(profile["baseline_record_id"] is None for profile in preparing)
            )

        product_registry = contracts.load_registry(PRODUCT_CONTRACTS)
        contracts_by_id = {
            contract["contract_id"]: contract
            for contract in product_registry["contracts"]
        }
        for profile in registry["profiles"]:
            contract = contracts_by_id[profile["product_contract_id"]]
            self.assertEqual(contract["mode"], profile["mode"])

    def test_missing_product_contract_id_is_rejected(self) -> None:
        registry = self._portable_registry()
        del registry["profiles"][0]["product_contract_id"]

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(
                profiles.RegistryError, "product_contract_id"
            ):
                profiles.validate_registry(path, LEDGER)

    def test_unknown_baseline_record_is_rejected(self) -> None:
        registry = self._portable_registry()
        registry["profiles"][0]["baseline_record_id"] = "missing-record"

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(profiles.RegistryError, "unknown ledger record"):
                profiles.validate_registry(path, LEDGER)

    def test_duplicate_profile_id_is_rejected(self) -> None:
        registry = self._portable_registry()
        registry["profiles"][1]["profile_id"] = registry["profiles"][0]["profile_id"]

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(profiles.RegistryError, "duplicate values"):
                profiles.validate_registry(path, LEDGER)

    def test_intentional_changes_require_successor_epoch(self) -> None:
        registry = self._portable_registry()
        registry["evolution_policy"][
            "successor_epoch_required_for_intentional_change"
        ] = False

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(profiles.RegistryError, "successor epoch"):
                profiles.validate_registry(path, LEDGER)

    def test_preparing_profile_may_defer_baseline_record(self) -> None:
        registry = self._portable_registry()
        preparing = next(
            profile
            for profile in registry["profiles"]
            if profile["status"] == "preparing"
        )
        self.assertIsNone(preparing["baseline_record_id"])

        with self._write_registry(registry) as path:
            profiles.validate_registry(path, LEDGER)

    def test_active_profile_must_name_accepted_baseline(self) -> None:
        registry = self._portable_registry()
        registry["profiles"][0]["baseline_record_id"] = None

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(
                profiles.RegistryError, "required outside preparing"
            ):
                profiles.validate_registry(path, LEDGER)

    def test_registry_may_have_no_successor_in_preparation(self) -> None:
        registry = self._portable_registry()
        preparing_epochs = set(registry["preparing_epoch_ids"])
        registry["preparing_epoch_ids"] = []
        registry["epochs"] = [
            epoch
            for epoch in registry["epochs"]
            if epoch["epoch_id"] not in preparing_epochs
        ]
        registry["profiles"] = [
            profile
            for profile in registry["profiles"]
            if profile["epoch_id"] not in preparing_epochs
        ]

        with self._write_registry(registry) as path:
            profiles.validate_registry(path, LEDGER)

    def test_each_preparing_epoch_requires_all_four_modes(self) -> None:
        registry = self._portable_registry()
        epoch_id = registry["preparing_epoch_ids"][-1]
        registry["profiles"] = [
            profile
            for profile in registry["profiles"]
            if not (
                profile["epoch_id"] == epoch_id
                and profile["mode"] == "science"
            )
        ]

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(
                profiles.RegistryError, "must contain exactly one"
            ):
                profiles.validate_registry(path, LEDGER)

    def test_unlisted_preparing_epoch_is_rejected(self) -> None:
        registry = self._portable_registry()
        registry["preparing_epoch_ids"].pop()

        with self._write_registry(registry) as path:
            with self.assertRaisesRegex(
                profiles.RegistryError, "agree with preparing_epoch_ids"
            ):
                profiles.validate_registry(path, LEDGER)

    def _portable_registry(self) -> dict:
        registry = copy.deepcopy(json.loads(REGISTRY.read_text(encoding="utf-8")))
        for profile in registry["profiles"]:
            products = profile["products"]
            if "scientific_profile" in products:
                products["scientific_profile"] = str(
                    REPO_ROOT / products["scientific_profile"]
                )
        return registry

    class _RegistryFile:
        def __init__(self, registry: dict) -> None:
            self.registry = registry
            self.directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.directory = tempfile.TemporaryDirectory()
            path = Path(self.directory.name) / "registry.json"
            path.write_text(json.dumps(self.registry), encoding="utf-8")
            (path.parent / "config_binding_policies.json").write_text(
                BINDING_POLICIES.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

    def _write_registry(self, registry: dict) -> _RegistryFile:
        return self._RegistryFile(registry)


if __name__ == "__main__":
    unittest.main()
