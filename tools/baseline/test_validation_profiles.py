import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.baseline import validation_profiles as profiles


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "validation/validation_profiles.json"
LEDGER = REPO_ROOT / "validation/accepted_runs.json"


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
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

    def _write_registry(self, registry: dict) -> _RegistryFile:
        return self._RegistryFile(registry)


if __name__ == "__main__":
    unittest.main()
