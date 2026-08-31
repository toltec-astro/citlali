#!/usr/bin/env python3
"""Focused tests for the explicitly discovery-only legacy TOD adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from tools.wp7 import export_legacy_tod_psd_line_discovery as adapter
from tools.wp7 import rtc_filter_psd_line_evidence as evidence


class LegacyTodAdapterTest(unittest.TestCase):
    def make_tod(self, path: Path, output_type: str = "ptc") -> None:
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("one", 1)
            dataset.createDimension("sample", 1280)
            dataset.createDimension("detector", 4)
            dataset.createDimension("scan", 2)
            dataset.createDimension("bound", 2)
            output = dataset.createVariable("tod_output_type", str, ("one",))
            output[:] = np.asarray([output_type], dtype=object)
            obsnum = dataset.createVariable("obsnum", "i4")
            obsnum.assignValue(152391)
            scan_indices = dataset.createVariable(
                "scan_indices", "i8", ("scan", "bound")
            )
            scan_indices[:] = np.asarray([[0, 639], [640, 1279]])
            output_scan = dataset.createVariable("output_scan_index", "i8", ("scan",))
            output_scan[:] = np.asarray([3, 4])
            apt_nw = dataset.createVariable("apt_nw", "i8", ("detector",))
            apt_nw[:] = np.asarray([7, 7, 7, 7])
            apt_array = dataset.createVariable("apt_array", "i8", ("detector",))
            apt_array[:] = np.asarray([0, 0, 0, 0])
            apt_uid = dataset.createVariable("apt_uid", "i8", ("detector",))
            apt_uid[:] = np.asarray([100, 101, 102, 103])
            time = dataset.createVariable("TelUTC", "f8", ("sample",))
            time[:] = 1000.0 + np.arange(1280) / 64.0
            signal = dataset.createVariable(
                "signal", "f8", ("sample", "detector")
            )
            signal[:] = np.sin(2.0 * np.pi * 18.0 * np.arange(1280)[:, None] / 64.0)
            flags = dataset.createVariable(
                "flags", "i1", ("sample", "detector")
            )
            flags[:] = 0
            for name, value in (
                ("CONFIG.TODFILTERED", 1),
                ("CONFIG.DOWNSAMPLED", 1),
                ("CONFIG.RTC.LINE_AUDIT.ENABLED", 0),
                ("CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED", 1),
                ("CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED", 0),
            ):
                variable = dataset.createVariable(name, "i4", ("one",))
                variable[:] = np.asarray([value])

    def test_adapter_cannot_claim_native_timing_or_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tod = root / "legacy.nc"
            self.make_tod(tod)
            manifest = adapter.export_legacy_tod(
                tod,
                root / "export",
                case_id="pointing-152391-discovery",
                route_family="pointing",
                network=7,
            )
            loaded = evidence.load_input(manifest)
            self.assertEqual(
                loaded.manifest["identity"]["timing_domain"],
                evidence.DISCOVERY_TIMING_DOMAIN,
            )
            self.assertEqual(
                loaded.manifest["identity"]["stream_stage"], "legacy_ptc_output"
            )
            result = evidence.build_evidence(loaded, root / "result")
            self.assertEqual(result["disposition"], "discovery_only_non_native_timing")
            self.assertEqual(
                result["line_inventory"]["ordering_relevance"],
                "diagnostic_only_postcleaning_stream",
            )

    def test_missing_network_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tod = root / "legacy.nc"
            self.make_tod(tod, output_type="rtc")
            with self.assertRaisesRegex(RuntimeError, "no detectors"):
                adapter.export_legacy_tod(
                    tod,
                    root / "export",
                    case_id="missing",
                    route_family="science",
                    network=0,
                )


if __name__ == "__main__":
    unittest.main()
