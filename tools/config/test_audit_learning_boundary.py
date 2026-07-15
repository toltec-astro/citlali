from __future__ import annotations

import unittest

from tools.config import audit_learning_boundary as audit


class LearningBoundaryAuditTest(unittest.TestCase):
    def test_extracts_multiline_learning_tuple_paths(self) -> None:
        source = '''
        std::tuple{"timestream", "learning",
                   "enabled"};
        std::tuple{"runtime", "verbose"};
        '''
        self.assertEqual(
            audit.tuple_paths(source), ["timestream.learning.enabled"]
        )

    def test_digest_is_order_sensitive(self) -> None:
        self.assertNotEqual(
            audit.digest(["a", "b"]), audit.digest(["b", "a"])
        )


if __name__ == "__main__":
    unittest.main()
