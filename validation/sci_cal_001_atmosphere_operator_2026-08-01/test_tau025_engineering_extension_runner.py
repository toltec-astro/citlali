#!/usr/bin/env python3
"""Focused no-AM checks for the SCI-CAL-001 TAU025 runner."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

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
    # Exercise the actual admission path with a temporary stand-in for the
    # one selected root.  It must reject an existing target, admission lock,
    # unwritable parent, or insufficient provisioned storage.
    with tempfile.TemporaryDirectory(prefix="sci_cal_001_tau025_runner_test_") as temporary:
        admitted = Path(temporary) / runner.CACHE_BASENAME
        with patch.object(runner, "CACHE_ROOT", admitted):
            admission = runner.cache_admission(admitted)
            assert admission["target_absent"] and admission["parent_writable"]
            assert admission["admission_lock_absent"]
            assert admission["retry_root_authorization"]["decision_id"] == "CAL-ATM-D007-RETRY-ROOT-001"
            admission_lock = admitted.parent / f".{admitted.name}.admission.lock"
            admission_lock.touch()
            try:
                runner.cache_admission(admitted)
            except RuntimeError:
                pass
            else:
                raise AssertionError("admission lock was accepted")
            admission_lock.unlink()
            admitted.mkdir()
            try:
                runner.cache_admission(admitted)
            except RuntimeError:
                pass
            else:
                raise AssertionError("existing target was accepted")
            admitted.rmdir()
            with patch.object(runner.os, "access", return_value=False):
                try:
                    runner.cache_admission(admitted)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unwritable parent was accepted")
            with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=runner.MIN_FREE_BYTES - 1)):
                try:
                    runner.cache_admission(admitted)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("underprovisioned target was accepted")
        for rejected in (runner.FORENSIC_CACHE_ROOT, Path(temporary) / "arbitrary_sibling"):
            try:
                runner.cache_admission(rejected)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"unapproved root admitted: {rejected}")
        assert not admitted.exists()
    # The exact committed JSON path is restored before first would-be AM argv
    # construction.  subprocess.run is trapped to prove this coverage cannot
    # invoke AM; the sentinel leaf remains absent throughout.
    with tempfile.TemporaryDirectory(prefix="sci_cal_001_tau025_deserialize_") as temporary:
        sentinel = Path(temporary) / runner.CACHE_BASENAME
        real_subprocess_run = runner.subprocess.run

        def reject_am_invocation(argv, *args, **kwargs):
            if Path(argv[0]).resolve() == runner.AM_EXECUTABLE.resolve():
                raise AssertionError("AM must not be invoked")
            return real_subprocess_run(argv, *args, **kwargs)

        with patch.object(runner.subprocess, "run", side_effect=reject_am_invocation):
            context = runner.preflight(sentinel, dry_run_sentinel=True)
            restored = runner.deserialize_full_inventory(context["full_inventory"])
            construction = next(item for item in restored if item.node.role == "construction")
            heldout = next(item for item in restored if item.node.role == "heldout")
            profiles_by_filename = {row["filename"]: row for row in context["inputs"]["profiles"]}
            command_builder = runner.CacheRunner(sentinel, context, profiles_by_filename)
            construction_spec = runner.full_grid_specification(construction, "1.00000000000000000")
            heldout_spec = runner.full_grid_specification(heldout, "1.00000000000000000")
            construction_argv = command_builder.argv(construction_spec)
            heldout_argv = command_builder.argv(heldout_spec)
        assert len(restored) == 1275
        assert construction_spec.zenith_angle_deg == 90 - construction.elevation_deg
        assert heldout_spec.zenith_angle_deg == 90 - heldout.elevation_deg
        assert construction_argv[1].endswith(f"{construction.profile}.amc")
        assert heldout_argv[1].endswith(f"{heldout.profile}.amc")
        assert construction_argv[-1] == heldout_argv[-1] == "1.00000000000000000"
        assert not sentinel.exists()
    print("TAU025 runner focused no-AM tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
