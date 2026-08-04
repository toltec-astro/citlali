#!/usr/bin/env python3
"""Focused contracts for the SCI-ALIGN-001 148670 replay preparation tool."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.diagnostics.prepare_sci_align_001_148670_reproduction import (
    APT_NAME,
    ARCHIVED_APT_SHA256,
    ARCHIVED_RAW_SHA256,
    ARCHIVED_TELESCOPE_SHA256,
    NETWORKS,
    PRIOR_RELATIVE,
    TELESCOPE_NAME,
    PreparationError,
    input_paths,
    prepare,
    reproduction_config,
    verify_archived_inputs,
)


class ReproductionConfigurationTest(unittest.TestCase):
    def test_binds_the_exact_148670_inputs_and_requests_detector_tod(self) -> None:
        config = reproduction_config(
            analysis_root=Path("/analysis"),
            raw_root=Path("/raw"),
            repo_root=Path("/repo"),
            output_root=Path("/output/reduced"),
            threads=6,
        )
        observation = config["inputs"][0]
        items = observation["data_items"]
        self.assertEqual(items[0]["filepath"], f"/analysis/reduced/{TELESCOPE_NAME}")
        self.assertEqual(
            [item["meta"]["interface"] for item in items[1:]],
            [f"toltec{network}" for network in NETWORKS],
        )
        self.assertEqual(
            [item["filepath"] for item in items[1:]],
            [f"/raw/toltec{network}_148670_000_0002_2026_01_13_11_59_10.nc" for network in NETWORKS],
        )
        self.assertEqual(
            observation["cal_items"][2]["filepath"],
            f"/analysis/reduced/{APT_NAME}",
        )
        self.assertEqual(config["beammap"]["priors"]["filepath"], f"/repo/{PRIOR_RELATIVE}")
        self.assertEqual(config["kids"]["solver"]["fitreportdir"], "/raw")
        self.assertTrue(config["beammap"]["detector_tod_output"]["enabled"])
        self.assertEqual(config["beammap"]["detector_tod_output"]["subdir_name"], "source_crossing_tod")
        self.assertEqual(config["runtime"]["output_dir"], "/output/reduced")
        self.assertEqual(config["runtime"]["n_threads"], 6)

    def test_requires_positive_thread_count(self) -> None:
        with self.assertRaisesRegex(PreparationError, "threads must be positive"):
            reproduction_config(
                analysis_root=Path("/analysis"),
                raw_root=Path("/raw"),
                repo_root=Path("/repo"),
                output_root=Path("/output/reduced"),
                threads=0,
            )

    def test_input_identity_requires_the_archived_apt_telescope_and_raw_files(self) -> None:
        rows = [
            {"role": "matched_input_apt", "sha256": ARCHIVED_APT_SHA256},
            {"role": "telescope", "sha256": ARCHIVED_TELESCOPE_SHA256},
            *[
                {"role": f"toltec{network}", "sha256": digest}
                for network, digest in ARCHIVED_RAW_SHA256.items()
            ],
        ]
        verify_archived_inputs(rows)
        rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PreparationError, "input identity mismatch"):
            verify_archived_inputs(rows)

    def test_path_inventory_has_only_the_expected_networks(self) -> None:
        paths = input_paths(Path("/analysis"), Path("/raw"), Path("/repo"))
        self.assertEqual(tuple(ARCHIVED_RAW_SHA256), NETWORKS)
        self.assertEqual(
            [role for role in paths if role.startswith("toltec")],
            [f"toltec{network}" for network in NETWORKS],
        )

    def test_prepare_publishes_a_checksum_bound_replay_directory(self) -> None:
        rows = [
            {"role": "matched_input_apt", "sha256": ARCHIVED_APT_SHA256},
            {"role": "telescope", "sha256": ARCHIVED_TELESCOPE_SHA256},
            *[
                {"role": f"toltec{network}", "sha256": digest}
                for network, digest in ARCHIVED_RAW_SHA256.items()
            ],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "citlali"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            output = root / "replay"
            arguments = argparse.Namespace(
                analysis_root=Path("/analysis"),
                raw_root=Path("/raw"),
                repo_root=Path("/repo"),
                citlali_bin=executable,
                output_root=output,
                threads=6,
                dry_run=False,
            )
            with mock.patch(
                "tools.diagnostics.prepare_sci_align_001_148670_reproduction.input_manifest",
                return_value=rows,
            ):
                preparation = prepare(arguments)
            self.assertTrue(preparation["detector_tod_requested"])
            self.assertEqual(preparation["fitreport_directory"], "/raw")
            checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertIn("config/citlali_o148670_0_2_c1_sci_align_reproduction.yaml", checksums)
            self.assertIn("run_148670_reproduction.sh", checksums)
            self.assertIn("submit_148670_reproduction.sbatch", checksums)
            run_script = output / "run_148670_reproduction.sh"
            self.assertIn("! -name SHA256SUMS", run_script.read_text())
            subprocess.run(["bash", "-n", str(run_script)], check=True)


if __name__ == "__main__":
    unittest.main()
