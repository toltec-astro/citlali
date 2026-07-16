import json
import tempfile
import unittest
from pathlib import Path

from tools.baseline import validate_reduction as validate
from tools.baseline.validation_profiles import profile_by_id, validate_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "validation/validation_profiles.json"
LEDGER = REPO_ROOT / "validation/accepted_runs.json"


class ValidateReductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = validate_registry(REGISTRY, LEDGER)

    def test_point_command_is_zero_tolerance_and_complete(self) -> None:
        profile = profile_by_id(self.registry, "phase4-point-152389-v1")
        command = validate.build_product_command(
            profile, Path("/baseline"), Path("/candidate"), Path("/result.json")
        )

        self.assertIn("--include-timestream", command)
        self.assertIn("--strict", command)
        self.assertEqual(command[command.index("--atol") + 1], "0.0")
        self.assertEqual(command[command.index("--rtol") + 1], "0.0")
        self.assertEqual(command[command.index("--max-array-elements") + 1], "0")

    def test_science_command_uses_pinned_scientific_profile(self) -> None:
        profile = profile_by_id(
            self.registry, "phase4-science-152390-152392-v1"
        )
        command = validate.build_product_command(
            profile, Path("/baseline"), Path("/candidate"), Path("/result.json")
        )

        self.assertTrue(command[1].endswith("compare_science_scientific_equivalence.py"))
        pinned = Path(command[command.index("--profile") + 1])
        self.assertEqual(
            pinned,
            REPO_ROOT / "validation/profiles/science_refactor_snapshot_v1.json",
        )

    def test_audit_command_requires_profile_provenance(self) -> None:
        profile = profile_by_id(self.registry, "phase4-beammap-148670-v1")
        command = validate.build_audit_command(
            profile, Path("/candidate"), Path("/result.json")
        )

        self.assertIn("--require-beammap-provenance", command)
        self.assertIn("--require-raw-provenance", command)
        self.assertIn("--require-config-source-manifest", command)

    def test_contract_command_uses_profile_contract(self) -> None:
        profile = profile_by_id(self.registry, "phase4-oof-152385-152387-v1")
        command = validate.build_contract_command(
            profile, Path("/candidate"), Path("/result.json")
        )

        self.assertTrue(command[1].endswith("validate_product_contract.py"))
        self.assertEqual(
            command[command.index("--contract") + 1],
            "phase4-oof-products-v1",
        )
        self.assertEqual(
            Path(command[command.index("--registry") + 1]),
            REPO_ROOT / "validation/product_contracts.json",
        )

    def test_config_command_requires_one_lowlevel_file(self) -> None:
        profile = profile_by_id(self.registry, "phase4-point-152389-v1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            (baseline / "citlali_o1.yaml").write_text("a: 1\n", encoding="utf-8")
            (candidate / "citlali_o1.yaml").write_text("a: 1\n", encoding="utf-8")

            command = validate.build_config_command(
                profile, baseline, candidate, root / "result.json"
            )
            self.assertEqual(Path(command[2]), baseline / "citlali_o1.yaml")
            self.assertEqual(Path(command[3]), candidate / "citlali_o1.yaml")

            (candidate / "citlali_o2.yaml").write_text("a: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(validate.ValidationError, "exactly one"):
                validate.build_config_command(
                    profile, baseline, candidate, root / "result.json"
                )

    def test_report_rejects_any_failed_gate(self) -> None:
        result = {
            "profile_id": "profile",
            "epoch_id": "epoch",
            "mode": "point",
            "baseline": "/baseline",
            "candidate": "/candidate",
            "passed": False,
            "gates": [
                {
                    "name": "audit",
                    "passed": True,
                    "exit_code": 0,
                    "result": {"log": {"issue_counts": {}}},
                    "stdout": "",
                    "stderr": "",
                },
                {
                    "name": "products",
                    "passed": False,
                    "exit_code": 4,
                    "result": None,
                    "stdout": "",
                    "stderr": "changed product\n",
                },
            ],
        }

        report = validate.render_markdown(result)

        self.assertIn("Verdict: **rejected**", report)
        self.assertIn("`products`: **FAIL**", report)
        self.assertIn("changed product", report)


if __name__ == "__main__":
    unittest.main()
