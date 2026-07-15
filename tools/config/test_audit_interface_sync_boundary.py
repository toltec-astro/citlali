from __future__ import annotations

import unittest

from tools.config import audit_interface_sync_boundary as audit


class InterfaceSyncBoundaryAuditTest(unittest.TestCase):
    def test_digest_is_order_sensitive(self) -> None:
        self.assertNotEqual(
            audit.digest(["a", "b"]), audit.digest(["b", "a"])
        )


if __name__ == "__main__":
    unittest.main()
