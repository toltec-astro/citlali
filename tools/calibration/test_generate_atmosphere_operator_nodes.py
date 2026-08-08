from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.calibration import generate_atmosphere_operator_nodes as generator


class AtmosphereOperatorArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.contract = cls.root / "data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_contract.json"
        cls.nodes = cls.root / "data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv"
        cls.header = cls.root / "include/citlali/core/timestream/atmosphere_operator_nodes_generated.h"

    def test_exact_digests_schema_and_generated_parity(self) -> None:
        content = generator.generate(self.contract, self.nodes)
        self.assertEqual(content, self.header.read_text(encoding="utf-8"))
        self.assertIn("std::array<SeriesDescriptor, 72>", content)
        self.assertIn("std::array<double, 1368>", content)

    def test_tampered_node_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nodes.csv"
            path.write_bytes(self.nodes.read_bytes().replace(b"am_q25", b"am_q24", 1))
            with self.assertRaisesRegex(generator.ArtifactError, "SHA-256 mismatch"):
                generator.validate_nodes(path)

    def test_tampered_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_bytes(self.contract.read_bytes().replace(b"0.25", b"0.26", 1))
            with self.assertRaisesRegex(generator.ArtifactError, "SHA-256 mismatch"):
                generator.validate_contract(path)


if __name__ == "__main__":
    unittest.main()
