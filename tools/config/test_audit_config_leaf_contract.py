from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.config import audit_config_leaf_contract as audit


class ConfigLeafContractTest(unittest.TestCase):
    def test_matching_rule_rejects_uncovered_leaf(self) -> None:
        with self.assertRaisesRegex(audit.ContractError, "uncovered"):
            audit.matching_rule("unknown.path", [])

    def test_matching_rule_uses_ordered_specific_rule(self) -> None:
        rules = [
            {"id": "specific", "patterns": ["value.exact"]},
            {"id": "broad", "patterns": ["value*"]},
        ]
        self.assertEqual(
            audit.matching_rule("value.exact", rules)["id"], "specific"
        )

    def test_numeric_domain_requires_finite_values(self) -> None:
        self.assertEqual(
            audit.allowed_domain({"int", "float"}, True),
            {"kind": "typed-real", "finite_required": True},
        )

    def test_non_executable_domain_is_explicit(self) -> None:
        self.assertEqual(
            audit.allowed_domain({"bool"}, False),
            {"kind": "ignored", "finite_required": False},
        )

    def test_rule_can_strengthen_observed_numeric_domain(self) -> None:
        rule = {
            "id": "offset",
            "allowed_domain": {
                "kind": "typed-real",
                "finite_required": True,
            },
        }
        self.assertEqual(
            audit.resolved_allowed_domain(rule, {"int"}, True),
            {"kind": "typed-real", "finite_required": True},
        )

    def test_semantic_view_ignores_machine_local_missing_inputs(self) -> None:
        value = {
            "summary": {"leaf_count": 1, "missing_optional_inputs": ["/tmp/x"]},
            "leaves": [],
        }
        self.assertEqual(
            audit.semantic_view(value),
            {"summary": {"leaf_count": 1}, "leaves": []},
        )


if __name__ == "__main__":
    unittest.main()
