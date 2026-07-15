from __future__ import annotations

import unittest

from tools.config import generate_config_schema_header as generator


class GenerateConfigSchemaHeaderTest(unittest.TestCase):
    def test_expands_mapping_and_sequence_prefixes(self) -> None:
        self.assertEqual(
            generator.node_paths(["inputs[].cal_items[].meta.interface"]),
            [
                "inputs",
                "inputs[]",
                "inputs[].cal_items",
                "inputs[].cal_items[]",
                "inputs[].cal_items[].meta",
                "inputs[].cal_items[].meta.interface",
            ],
        )

    def test_preserves_empty_container_paths(self) -> None:
        self.assertEqual(
            generator.yaml_node_paths({"clean": {"grouping": []}}),
            {"clean", "clean.grouping", "clean.grouping[]"},
        )


if __name__ == "__main__":
    unittest.main()
