#!/usr/bin/env python3
"""Local identity and immutability tests for the ED2 successor package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import unittest

import jsonschema
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[3]
STOP_COMMIT = "3e014f11decbcf17ad372391e5e960e6c0c54461"
CANDIDATE = "ed28dafb37f9113c0d3c95297148157129a90886"
CANDIDATE_TREE = "cf75c36557178f351fb62781108a6f4b41b19225"
SUCCESSOR_REL = PACKAGE.relative_to(REPO).as_posix()
PREDECESSOR = PACKAGE.parent / "repair-ed28dafb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(REPO), *args), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.rstrip()


class PackageContractTest(unittest.TestCase):
    def test_candidate_and_resume_identity(self) -> None:
        self.assertEqual(git("rev-parse", f"{CANDIDATE}^{{tree}}"), CANDIDATE_TREE)
        self.assertEqual(
            git("rev-parse", f"{STOP_COMMIT}^{{tree}}"),
            "6db187a3f0f976cbd16dafbe17078438c0af1733",
        )
        self.assertEqual(
            git("rev-parse", f"{STOP_COMMIT}^"),
            "1b824f138754eeb1856ae5f102027db4b31598be",
        )

    def test_exact_ed2_coordination_authorities(self) -> None:
        authority = load("campaign.json")["authority"]
        self.assertEqual(
            authority["ed2_content_commit"],
            "ae2188dd4761afa85a772a1edd6b9d9571fa9d4b",
        )
        self.assertEqual(
            authority["ed2_identity_binding_head"],
            "c35333d4090e2bebae422538cb40fc063f7cb71a",
        )
        self.assertEqual(
            authority["ed2_stop_return_review_sha256"],
            "ec98c2f3b8475e7aa4842363780cc247143ecca05053ea61e22e0a9d8e22f83d",
        )
        self.assertEqual(
            authority["ed2_owner_decision_sha256"],
            "b03e410bf246fd4e3218d1114b59cf96f6019a901112fadea6074af0003a026a",
        )
        self.assertEqual(
            authority["ed2_continuation_handoff_sha256"],
            "709873e1c3e325d9e1a0a2a85d6acd647b9a31b44f4074014abe674878ffa058",
        )
        provenance = (PACKAGE / "PROVENANCE.md").read_text(encoding="utf-8")
        for token in (
            "a38ec92f28d63d543ad80d463bc99b5ec4606e52",
            "85998ea7c078208ba6bcae939dd97919f5189cf776f727bd00651cf6ef07d8c4",
            "23ddec55b6bede06cb27342d00fd96bb9a919019",
            "bb9fba34f6122a24268fd9fba3e92d8775b1c678fb908a4cd019e491b3a3b73b",
        ):
            self.assertIn(token, provenance)

    def test_no_application_or_build_surface_changed(self) -> None:
        paths = git("diff", "--name-only", f"{CANDIDATE}..HEAD").splitlines()
        forbidden_prefixes = ("src/", "include/", "config/", "tools/")
        forbidden_names = {"CMakeLists.txt", "CMakePresets.json", "CMakeUserPresets.json"}
        bad = [
            path for path in paths
            if path.startswith(forbidden_prefixes) or Path(path).name in forbidden_names
        ]
        self.assertEqual(bad, [])

    def test_current_edits_are_confined_to_successor(self) -> None:
        changed = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        bad = []
        for row in changed:
            path = row[3:].split(" -> ")[-1]
            if not path.startswith(SUCCESSOR_REL + "/"):
                bad.append(row)
        self.assertEqual(bad, [])

    def test_frozen_predecessor(self) -> None:
        self.assertEqual(
            git("rev-parse", f"{STOP_COMMIT}:{PREDECESSOR.relative_to(REPO)}"),
            "dbf486e30c9b78ca16e05bccafc2d027562d0746",
        )
        self.assertEqual(
            digest(PREDECESSOR / "SHA256SUMS"),
            "ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69",
        )
        subprocess.run(
            ("shasum", "-a", "256", "-c", "SHA256SUMS"),
            cwd=PREDECESSOR, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def test_stop_artifacts_are_immutable(self) -> None:
        expected = {
            "decision-brief.json": "6b2d061332f62fd6316c37c3efade3196a181377e219730923e05ae0b1062b92",
            "INDEPENDENT_READ_ONLY_REVIEW_2026-08-02.md": "ab1949e738c22544ede6ae9af449bfbe219f5e33794346426d8a81eb76bdca6d",
            "MAP-UNITY-ED1_BOUNDED_DECISION_BRIEF_2026-08-02.md": "a1de515081bee6169811ac9a9f7ec14ab4e07135b6a30858384c7325e676d2bb",
            "SHA256SUMS": "293b21ec162d407496c22db0b022cc512e8e4ebc8ac0c6d15765e8bbd844cc60",
        }
        self.assertEqual({name: digest(PACKAGE / name) for name in expected}, expected)

    def test_fixed_seven_case_matrix_is_unchanged(self) -> None:
        successor = load("campaign.json")
        predecessor = json.loads((PREDECESSOR / "campaign.json").read_text())
        self.assertEqual(successor["cases"], predecessor["cases"])
        self.assertEqual(
            [row["id"] for row in successor["cases"]],
            ["P-SEQ", "P-OMP", "S-C-SEQ", "S-C-OMP", "S-E-SEQ", "S-E-OMP", "S-X-SEQ"],
        )

    def test_ed2_capture_contract(self) -> None:
        campaign = load("campaign.json")
        self.assertEqual(campaign["revision"], "repair-sha-ed28dafb-ed1-2026-08-02")
        self.assertEqual(campaign["candidate_sha"], CANDIDATE)
        self.assertEqual(campaign["auxiliary_capture_contract"]["binary_count"], 1)
        captures = campaign["auxiliary_capture_contract"]["captures"]
        self.assertEqual([row["id"] for row in captures], ["CAP-POINT", "CAP-SCIENCE"])
        self.assertEqual(captures[1]["science_observations"], [152390, 152392])
        self.assertEqual(captures[1]["pointing_support_observations"], [152389, 152391, 152393])
        self.assertEqual(campaign["resource_contract"]["cumulative_ceiling_bytes"], 214748364800)
        self.assertEqual(campaign["compact_evidence_contract"]["groups"], [
            "152389:a1100", "152389:a1400", "152389:a2000",
            "152390:a1100", "152390:a1400", "152390:a2000",
            "152392:a1100", "152392:a1400", "152392:a2000",
        ])

    def test_exact_tolproj_specs(self) -> None:
        self.assertEqual(load("tolproj-point-source.json"), {
            "description": "SCI-MAP-001 Point source project",
            "project_id": "SCI-MAP-001-POINT-SOURCE",
            "obsnums": [152389],
            "1146+399": {"obsnums": [152389]},
        })
        self.assertEqual(load("tolproj-science-source.json"), {
            "description": "SCI-MAP-001 Science source project",
            "project_id": "SCI-MAP-001-SCIENCE-SOURCE",
            "obsnums": [152389, 152390, 152391, 152392, 152393],
            "NGC4449": {"obsnums": [152390, 152392]},
            "1146+399": {"obsnums": [152389, 152391, 152393]},
        })

    def test_exact_capture_overlay(self) -> None:
        overlay = yaml.safe_load((PACKAGE / "processed-time-chunk-full-overlay.yaml").read_text())
        self.assertEqual(overlay, {"reduce": {"steps": {0: {"config": {"low_level": {
            "timestream": {"processed_time_chunk": {"output": {
                "enabled": True, "mode": "full", "indices": "all",
            }}},
        }}}}}})

    def test_owner_operational_values_remain_unfilled(self) -> None:
        values = load("owner-values.template.json")
        fixed = {
            "schema_version": "sci-map-unity-owner-values-v1",
            "unity_host_alias": "unity_toltec",
            "slurm_account": "", "slurm_qos": "", "slurm_constraint": "", "slurm_reservation": "",
        }
        for key, value in values.items():
            if key in fixed:
                self.assertEqual(value, fixed[key])
            else:
                self.assertIsNone(value, key)

    def test_all_json_schema_documents_are_well_formed(self) -> None:
        schemas = sorted(PACKAGE.glob("*.schema.json"))
        self.assertGreaterEqual(len(schemas), 8)
        for path in schemas:
            with self.subTest(path=path.name):
                jsonschema.Draft202012Validator.check_schema(
                    json.loads(path.read_text(encoding="utf-8"))
                )

    def test_driver_blocks_remote_and_scheduler_execution(self) -> None:
        path = PACKAGE / "scripts" / "unity-campaign.py"
        spec = importlib.util.spec_from_file_location("ed2_unity_campaign", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for executable in ("ssh", "scp", "rsync", "sbatch", "srun"):
            with self.subTest(executable=executable):
                with self.assertRaises(module.CampaignError):
                    module.require_local_command((executable, "--version"))

    def test_human_runbook_remote_and_staging_safety(self) -> None:
        text = (PACKAGE / "OWNER_RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn("Unity (after the owner has logged in", text)
        self.assertIn("unity_toltec", text)
        self.assertNotRegex(text, r"(?m)^\s*(?:ssh|scp|rsync)\b.*$")
        self.assertNotRegex(text, r"(?m)^\s*ssh unity(?:\s|$)")
        self.assertIn('"$tolproj" copy-raw', text)
        self.assertIn("generated `citlali_*.yaml`", text)
        self.assertNotIn('--config "$site"', text)
        self.assertNotIn('--config "$TOLPROJ_SITE_CONFIG"', text)
        self.assertIn("ordinary Unity default", text)
        self.assertIn("full ptc remains on unity", text.lower())
        self.assertIn("Do not run cleanup", text)

    def test_local_resource_arithmetic(self) -> None:
        report = load("resource-report.json")
        measured = report["local_metadata_measurement"]
        by_obs = measured["processed_terms_by_observation"]
        self.assertEqual(sum(by_obs.values()), measured["processed_terms_total"])
        envelope = report["projected_unity_envelope"]
        total = sum(row["bytes"] for row in envelope["incremental_stage_bytes"])
        self.assertEqual(total, envelope["projected_total_bytes"])
        self.assertEqual(
            total + envelope["projected_headroom_bytes"],
            report["ceiling"]["bytes"],
        )
        self.assertFalse(report["retention"]["automatic_cleanup"])
        self.assertIn("not a component-wise full/all-PTC serialization bound",
                      envelope["warning"])
        runbook = (PACKAGE / "OWNER_RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn('RESOURCE_RECORDS="$COMPACT_EVIDENCE_ROOT/_campaign/resource-records"',
                      runbook)
        self.assertIn("resource_record PREPARE-STAGING pre", runbook)
        self.assertNotIn("$REQUEST_ROOT/resource-records", runbook)


if __name__ == "__main__":
    unittest.main()
