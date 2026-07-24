import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.baseline import phase5_readiness as readiness


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "validation/phase5_validation_readiness.json"
REGISTRY = REPO_ROOT / "validation/validation_profiles.json"
LEDGER = REPO_ROOT / "validation/accepted_runs.json"


class Phase5ReadinessTest(unittest.TestCase):
    def test_checked_record_is_valid_but_not_promotion_ready(self) -> None:
        result = readiness.validate_readiness(MANIFEST, REGISTRY, LEDGER)

        self.assertFalse(result["promotion_ready"])
        self.assertFalse(result["same_sha"])
        self.assertEqual(result["status"], "preparing")
        self.assertEqual(
            {fixture["mode"] for fixture in result["fixtures"]},
            readiness.MODES,
        )
        self.assertTrue(
            all(
                fixture["evidence_role"] == "fixture_smoke"
                for fixture in result["fixtures"]
            )
        )
        self.assertEqual(result["gate_counts"]["audit:blocked"], 4)
        self.assertEqual(result["gate_counts"]["config:pass"], 4)
        self.assertEqual(result["gate_counts"]["contract:pass"], 4)
        self.assertEqual(result["gate_counts"]["products:pass"], 4)

    def test_duplicate_mode_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["fixtures"][1]["mode"] = manifest["fixtures"][0]["mode"]

        with self._write_manifest(manifest) as path:
            with self.assertRaisesRegex(readiness.ReadinessError, "duplicate"):
                readiness.validate_readiness(path, REGISTRY, LEDGER)

    def test_nonpassing_gate_requires_blocker(self) -> None:
        manifest = self._manifest()
        manifest["fixtures"][0]["promotion_blockers"] = []

        with self._write_manifest(manifest) as path:
            with self.assertRaisesRegex(
                readiness.ReadinessError,
                "require a promotion blocker",
            ):
                readiness.validate_readiness(path, REGISTRY, LEDGER)

    def test_smoke_fixture_cannot_be_promotion_ready(self) -> None:
        manifest = self._manifest()
        fixture = manifest["fixtures"][0]
        fixture["gates"] = {gate: "pass" for gate in readiness.GATE_NAMES}
        fixture["promotion_blockers"] = []

        with self._write_manifest(manifest) as path:
            result = readiness.validate_readiness(path, REGISTRY, LEDGER)

        point = next(item for item in result["fixtures"] if item["mode"] == "point")
        self.assertFalse(point["promotion_ready"])

    def test_duplicate_global_blocker_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["global_blockers"].append(
            copy.deepcopy(manifest["global_blockers"][0])
        )

        with self._write_manifest(manifest) as path:
            with self.assertRaisesRegex(
                readiness.ReadinessError,
                "duplicate global blocker",
            ):
                readiness.validate_readiness(path, REGISTRY, LEDGER)

    def test_fixture_verification_matches_declared_gate_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            fixture = {
                "mode": "point",
                "profile_id": "phase5-point-152389-v2",
                "baseline_path": str(baseline),
                "local_path": str(candidate),
                "gates": {
                    "audit": "blocked",
                    "config": "pass",
                    "contract": "pass",
                    "products": "pass",
                },
            }
            validation = {
                "passed": False,
                "profile_id": fixture["profile_id"],
                "profile_status": "preparing",
                "epoch_id": "epoch",
                "mode": "point",
                "baseline": str(baseline),
                "candidate": str(candidate),
                "gates": [
                    {
                        "name": name,
                        "passed": passed,
                        "exit_code": 0 if passed else 1,
                        "result": None,
                        "stdout": "",
                        "stderr": "",
                    }
                    for name, passed in (
                        ("audit", False),
                        ("config", True),
                        ("contract", True),
                        ("products", True),
                    )
                ],
            }
            registry = readiness.validate_registry(REGISTRY, LEDGER)
            with mock.patch.object(
                readiness.validate_reduction,
                "run_validation",
                return_value=validation,
            ):
                result = readiness.verify_fixtures(
                    {"fixtures": [fixture]},
                    registry,
                    root / "reports",
                    REPO_ROOT / "validation/product_contracts.json",
                    REPO_ROOT / "validation/config_binding_policies.json",
                )

        self.assertTrue(result["matched"])
        self.assertEqual(result["cases"][0]["mismatched_gates"], [])

    def _manifest(self) -> dict:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    class _ManifestFile:
        def __init__(self, manifest: dict) -> None:
            self.manifest = manifest
            self.directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.directory = tempfile.TemporaryDirectory()
            path = Path(self.directory.name) / "manifest.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            return path

        def __exit__(self, *args: object) -> None:
            assert self.directory is not None
            self.directory.cleanup()

    def _write_manifest(self, manifest: dict) -> _ManifestFile:
        return self._ManifestFile(manifest)


if __name__ == "__main__":
    unittest.main()
