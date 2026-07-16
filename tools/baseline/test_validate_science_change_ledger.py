import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.baseline import validate_science_change_ledger as science_ledger


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "validation" / "intended_science_changes.json"
ACCEPTED_RUNS_PATH = REPO_ROOT / "validation" / "accepted_runs.json"
PRODUCT_CONTRACTS_PATH = REPO_ROOT / "validation" / "product_contracts.json"


class ValidateScienceChangeLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checked_in = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    def validate(self, value: dict[str, object]) -> tuple[int, int]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return science_ledger.validate_ledger(
                path,
                repo_root=REPO_ROOT,
                accepted_runs_path=ACCEPTED_RUNS_PATH,
                product_contracts_path=PRODUCT_CONTRACTS_PATH,
            )

    def modified(self) -> dict[str, object]:
        return copy.deepcopy(self.checked_in)

    def test_checked_in_ledger_is_valid(self) -> None:
        self.assertEqual(self.validate(self.modified()), (3, 5))

    def test_rejects_duplicate_change_id(self) -> None:
        value = self.modified()
        value["changes"][1]["change_id"] = value["changes"][0]["change_id"]

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "duplicate change_id"):
            self.validate(value)

    def test_rejects_missing_expected_numerical_effect(self) -> None:
        value = self.modified()
        del value["changes"][0]["expected_effect"]["numerical_or_schema"]

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "numerical_or_schema"):
            self.validate(value)

    def test_rejects_unknown_validation_record(self) -> None:
        value = self.modified()
        value["changes"][0]["validation_evidence"][1]["record_id"] = "not-a-run"

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "unknown accepted-run"):
            self.validate(value)

    def test_rejects_missing_document(self) -> None:
        value = self.modified()
        value["changes"][0]["validation_evidence"][0]["path"] = "doc/missing.md"

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "does not exist"):
            self.validate(value)

    def test_rejects_abbreviated_commit(self) -> None:
        value = self.modified()
        value["changes"][0]["commit_mappings"][0]["source_commit"] = "991428e70"

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "40-character"):
            self.validate(value)

    def test_rejects_false_patch_identity(self) -> None:
        value = self.modified()
        value["changes"][1]["commit_mappings"][0]["patch_id"] = "0" * 40

        with self.assertRaisesRegex(science_ledger.ScienceChangeLedgerError, "patch identity mismatch"):
            self.validate(value)


if __name__ == "__main__":
    unittest.main()
