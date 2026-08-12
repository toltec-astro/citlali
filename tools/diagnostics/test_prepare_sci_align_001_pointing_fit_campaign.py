#!/usr/bin/env python3
"""Focused tests for the staged SCI-ALIGN-001 pointing-fit campaign."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from astropy.table import Table

import analyze_sci_align_001_lissajous_pointing as map_space
import prepare_sci_align_001_pointing_fit_campaign as campaign


class CampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, root: Path, names: list[str]) -> None:
        campaign.write_checksums(root, names, "SHA256SUMS")

    def quality_package(self) -> tuple[Path, Path]:
        package = self.root / "quality"
        frozen = package / "frozen"
        result = package / "result"
        frozen.mkdir(parents=True)
        result.mkdir()
        product = self.root / "products"
        product.mkdir()
        frozen_rows = []
        audit_rows = []
        selected_rows = []
        for obsnum in (123424, campaign.PILOT_OBSNUM):
            ptc = product / f"pointing_{obsnum}_ptc.nc"
            ppt = product / f"pointing_{obsnum}_ppt.ecsv"
            ptc.write_bytes(f"ptc-{obsnum}".encode())
            ppt.write_bytes(f"ppt-{obsnum}".encode())
            ppt_digest = campaign.sha256_file(ppt)
            frozen_rows.append({
                "obsnum": obsnum,
                "ppt_path": str(ppt),
                "ppt_sha256": ppt_digest,
            })
            audit_rows.append({
                "obsnum": obsnum,
                "status": "ready",
                "size_bytes": ptc.stat().st_size,
                "ptc_path": str(ptc),
                "ppt_path": str(ppt),
            })
            selected_rows.append({
                "obsnum": obsnum,
                "snr_pass": True,
                "a1100_map_sha256": "a" * 64,
                "strongest_abs_smoothed_residual_fraction_peak": 0.1,
                "strongest_positive_secondary_peak_fraction": 0.0,
                "coherent_residual_component_count": 1,
            })
        (frozen / "frozen_input.json").write_text(json.dumps({
            "rows": frozen_rows
        }))
        self.write_manifest(frozen, ["frozen_input.json"])
        Table(rows=selected_rows).write(
            result / "snr_selected_pointings.ecsv", format="ascii.ecsv"
        )
        self.write_manifest(result, ["snr_selected_pointings.ecsv"])
        (package / "human_morphology_review.json").write_text(json.dumps({
            "schema": "sci-align-001-pointing-human-morphology-review-v1",
            "accepted_observation_count": 2,
        }))
        self.write_manifest(package, [
            "frozen/SHA256SUMS", "result/SHA256SUMS",
            "human_morphology_review.json",
        ])
        audit = self.root / "schema.csv"
        with audit.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)
        return package, audit

    def test_freeze_generates_staged_scripts_and_bound_selection(self) -> None:
        quality, audit = self.quality_package()
        output = self.root / "campaign"
        repo = Path(__file__).resolve().parents[2]
        args = argparse.Namespace(
            schema_audit=audit,
            quality_package=quality,
            map_protocol_template=(
                repo / "validation/sci_align_001_lissajous_pointing_2026-08-10"
                / "frozen_protocol.json"
            ),
            timestream_protocol_template=(
                repo / "validation/sci_align_001_lissajous_timestream_2026-08-10"
                / "frozen_protocol.json"
            ),
            repo_root=repo,
            output_root=output,
            python="python",
            array_concurrency=2,
        )
        summary = {
            "ptc_sample_count": 100,
            "ptc_scan_count": 12,
            "detector_count": 5000,
            "mean_elevation_deg": 50.0,
            "median_speed_arcsec_s": 35.0,
            "p95_speed_arcsec_s": 80.0,
            "populated_velocity_sector_count": 8,
        }
        with (
            mock.patch.object(campaign, "git_commit", return_value="f" * 40),
            mock.patch.object(
                campaign.map_space, "ppt_a1100",
                return_value={
                    "snr_a1100": 70.0, "ppt_x_arcsec": 1.0,
                    "ppt_y_arcsec": -1.0,
                },
            ),
            mock.patch.object(campaign.map_space, "ptc_summary", return_value=summary),
        ):
            campaign.freeze(args)
        selection = json.loads(
            (output / "frozen/selected_pointings.json").read_text()
        )
        protocol = json.loads(
            (output / "frozen/timestream_protocol.json").read_text()
        )
        self.assertEqual(selection["row_count"], 2)
        self.assertEqual(protocol["scope"]["pointing_count"], 2)
        self.assertEqual(
            protocol["input_authority"]["selection_manifest_sha256"],
            campaign.sha256_file(output / "frozen/selected_pointings.json"),
        )
        self.assertEqual(
            len((output / "jobs/fit_gate_remaining.commands.txt").read_text().splitlines()),
            1,
        )
        self.assertIn(
            "--owner-review-approved",
            (output / "jobs/resume.commands.txt").read_text(),
        )
        self.assertIn(
            "--existing-observation-root",
            (output / "jobs/run_map_aggregate.sbatch").read_text(),
        )
        for path in (output / "jobs").glob("*.sbatch"):
            subprocess.run(["bash", "-n", path], check=True)
        campaign.verify_manifest(output, "PREPARATION_SHA256SUMS")
        campaign.verify_manifest(output / "jobs", "JOB_SHA256SUMS")

    def test_map_run_one_owns_only_observation_directory(self) -> None:
        selection_dir = self.root / "selection"
        selection_dir.mkdir()
        ptc = self.root / "input.nc"
        ppt = self.root / "input.ecsv"
        ptc.write_text("ptc")
        ppt.write_text("ppt")
        row = {
            "pointing_obsnum": 42,
            "ptc_path": str(ptc), "ptc_sha256": campaign.sha256_file(ptc),
            "ppt_path": str(ppt), "ppt_sha256": campaign.sha256_file(ppt),
        }
        (selection_dir / "selected_pointings.json").write_text(json.dumps({
            "rows": [row]
        }))
        (selection_dir / "frozen_protocol.json").write_text("{}")
        campaign.write_checksums(selection_dir, [
            "selected_pointings.json", "frozen_protocol.json"
        ], "SHA256SUMS")
        output = self.root / "maps"
        with mock.patch.object(map_space, "analyze_pointing_isolated") as analyze:
            map_space.run_one(argparse.Namespace(
                selection_dir=selection_dir, output_root=output, obsnum=42
            ))
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[2], output.resolve() / "o42")

    def test_map_aggregate_accepts_authenticated_array_root(self) -> None:
        selection_dir = self.root / "aggregate_selection"
        selection_dir.mkdir()
        row = {"pointing_obsnum": 42}
        (selection_dir / "selected_pointings.json").write_text(json.dumps({
            "rows": [row]
        }))
        (selection_dir / "frozen_protocol.json").write_text("{}")
        campaign.write_checksums(selection_dir, [
            "selected_pointings.json", "frozen_protocol.json"
        ], "SHA256SUMS")
        output = self.root / "aggregate_maps"
        observation = output / "o42"
        observation.mkdir(parents=True)
        (observation / "result.json").write_text("{}")
        campaign.write_checksums(observation, ["result.json"], "SHA256SUMS")
        with mock.patch.object(map_space, "aggregate_results") as aggregate:
            map_space.aggregate(argparse.Namespace(
                selection_dir=selection_dir, output=output,
                existing_observation_root=output,
            ))
        aggregate.assert_called_once_with(selection_dir.resolve(), output.resolve(), [row])

    def test_gate_audit_counts_complete_missing_and_invalid(self) -> None:
        selection = self.root / "selected.json"
        selection.write_text(json.dumps({"rows": [
            {"pointing_obsnum": 1}, {"pointing_obsnum": 2},
            {"pointing_obsnum": 3},
        ]}))
        fit_root = self.root / "fits"
        complete = fit_root / "o1"
        complete.mkdir(parents=True)
        models = {}
        for name in ("constant", "lag", "hysteresis", "joint"):
            models[name] = {
                "status": "success", "tau_ms": 1.0,
                "parameters": {"h_az_arcsec": 0.1, "h_el_arcsec": -0.2},
            }
        (complete / "fit_gate.json").write_text(json.dumps({
            "quality_gate": {"automatic_structural_status": "pass"},
            "point_model_results": models,
        }))
        campaign.write_checksums(
            complete, ["fit_gate.json"], "FIT_GATE_SHA256SUMS"
        )
        invalid = fit_root / "o3"
        invalid.mkdir(parents=True)
        output = self.root / "gate_audit"
        campaign.audit_gates(argparse.Namespace(
            selection=selection, fit_root=fit_root, output=output
        ))
        summary = json.loads((output / "manifest.json").read_text())
        self.assertEqual(summary["complete_count"], 1)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["invalid_count"], 1)
        campaign.verify_manifest(output)


if __name__ == "__main__":
    unittest.main()
