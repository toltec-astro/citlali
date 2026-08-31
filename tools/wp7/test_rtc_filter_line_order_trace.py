#!/usr/bin/env python3
"""Source-order guards for the bounded WP-7 D2 line audit."""

from __future__ import annotations

import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class RtcFilterLineOrderTraceTest(unittest.TestCase):
    def test_rtc_prefilter_line_path_precedes_sample_removal(self) -> None:
        source = (
            SOURCE_ROOT / "include/citlali/core/timestream/rtc/rtcproc.h"
        ).read_text()
        prefilter = source.index(
            "if (line_audit.enabled && line_audit.pre_filter_enabled)"
        )
        fixed_notch = source.index("apply_rtc_line_audit_fixed_notches", prefilter)
        downsample = source.index("if (run_downsample)", fixed_notch)
        self.assertLess(prefilter, fixed_notch)
        self.assertLess(fixed_notch, downsample)

    def test_science_ptc_line_audit_is_later_than_rtc(self) -> None:
        source = (
            SOURCE_ROOT / "include/citlali/core/engine/detail/lali_run_impl.h"
        ).read_text()
        rtc = source.index("rtcproc.run(")
        ptc_line = source.index("apply_model_protected_ptc_line_audit(")
        self.assertLess(rtc, ptc_line)

    def test_pointing_ptc_line_audit_is_later_than_rtc(self) -> None:
        source = (
            SOURCE_ROOT / "include/citlali/core/engine/detail/pointing_run_impl.h"
        ).read_text()
        rtc = source.index("rtcproc.run(")
        ptc_line = source.index("apply_model_protected_ptc_line_audit(")
        self.assertLess(rtc, ptc_line)


if __name__ == "__main__":
    unittest.main()
