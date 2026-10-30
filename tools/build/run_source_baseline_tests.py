#!/usr/bin/env python3
"""Run baseline tests that do not require a compiled Citlali executable.

The native CI job runs the complete baseline suite, including the explicitly
identified CanonicalAptProtocolV1Test class. This selector reports that boundary
and fails if it no longer matches discovery; missing native code is not a skip.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE_CLASS = "test_validate_product_contract.CanonicalAptProtocolV1Test"


def partition(suite: unittest.TestSuite) -> tuple[list, list]:
    source, native = [], []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            child_source, child_native = partition(item)
            source.extend(child_source)
            native.extend(child_native)
        elif item.id().startswith(NATIVE_CLASS + "."):
            native.append(item)
        else:
            source.append(item)
    return source, native


def main() -> int:
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tools/baseline"), pattern="test_*.py"
    )
    source, native = partition(suite)
    if not source or not native:
        raise RuntimeError("baseline source/native test partition no longer matches discovery")
    print(f"Source baseline tests: {len(source)}; native CI owns {len(native)} tests in {NATIVE_CLASS}", flush=True)
    return 0 if unittest.TextTestRunner().run(unittest.TestSuite(source)).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
