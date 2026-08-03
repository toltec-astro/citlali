#!/usr/bin/env python3
"""Focused no-AM checks for the SCI-CAL-001 TAU025 runner."""

from pathlib import Path
import tempfile

import numpy as np

import run_tau025_engineering_extension as runner


def main() -> int:
    profiles = runner.profile_inventory()
    inventory = runner.full_inventory(profiles)
    assert len(inventory) == 1275
    assert len(runner.scale_inventory(profiles)) == 225
    assert all("nextafter" not in item.run_id for item in inventory)
    provenance = runner.derived_provenance()
    assert len(provenance) == 9
    assert max(float(item["absolute_difference"]) for item in provenance) < 1e-12
    warning_runner = runner.CacheRunner(Path(tempfile.gettempdir()), {}, {"LMT_DJF_5.amc": {"sha256": "x"}})
    admitted_warning = runner.p1.ParsedOutput(
        samples=np.zeros((50001, 4)), version_identity="am version test",
        warning_count=86, numeric_text_sha256="x", normalized_output_sha256="x",
        unresolved_column_warning_line_count=1,
        unresolved_summary_warning_line_count=1,
        other_warning_line_count=0, error_line_count=0,
    )
    warning_runner._warn_ok(1, admitted_warning, 50001)
    try:
        warning_runner._warn_ok(1, admitted_warning, 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("WARN-001 admitted a non-full scale-search grid")
    # A non-approved root is rejected before any path is made.
    with tempfile.TemporaryDirectory(prefix="sci_cal_001_tau025_runner_test_") as temporary:
        wrong = Path(temporary) / runner.CACHE_BASENAME
        try:
            runner.cache_admission(wrong)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unapproved root admitted")
        assert not wrong.exists()
    # The selected root must remain absent through all no-AM checks.
    assert not runner.CACHE_ROOT.exists()
    print("TAU025 runner focused no-AM tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
