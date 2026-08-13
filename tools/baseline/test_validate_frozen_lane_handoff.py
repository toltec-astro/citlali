import contextlib
import copy
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.baseline import validate_frozen_lane_handoff as handoff


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "validation" / "frozen_lane_handoff_packet.schema.json"


class ValidateFrozenLaneHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "base")
        self.git("config", "user.name", "Citlali Test")
        self.git("config", "user.email", "citlali-test@example.invalid")

        (self.repo / "validation").mkdir()
        (self.repo / "authority.md").write_text("authority\n", encoding="utf-8")
        (self.repo / "app.txt").write_text("base\n", encoding="utf-8")
        (self.repo / "validation" / "intended_science_changes.json").write_text(
            json.dumps(
                {
                    "schema_version": "citlali-intended-science-change-ledger-v1",
                    "changes": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.base = self.git("rev-parse", "HEAD")
        self.base_tree = self.git("rev-parse", "HEAD^{tree}")

        self.git("switch", "-c", "candidate")
        (self.repo / "app.txt").write_text("candidate\n", encoding="utf-8")
        self.git("add", "app.txt")
        self.git("commit", "-m", "candidate application")
        self.candidate = self.git("rev-parse", "HEAD")
        self.candidate_tree = self.git("rev-parse", "HEAD^{tree}")

        self.git("switch", "-c", "audit")
        (self.repo / "review.md").write_text("independent review\n", encoding="utf-8")
        self.git("add", "review.md")
        self.git("commit", "-m", "independent review")
        self.audit = self.git("rev-parse", "HEAD")
        self.audit_tree = self.git("rev-parse", "HEAD^{tree}")
        self.git("switch", "candidate")

        self.packet = self.make_packet()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *arguments: str, binary: bool = False):
        result = subprocess.run(
            ["git", "-C", str(self.repo), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                f"git {' '.join(arguments)} failed: "
                f"{result.stderr.decode(errors='replace')}"
            )
        return result.stdout if binary else result.stdout.decode().strip()

    def blob_oid(self, commit: str, path: str) -> str:
        return self.git("rev-parse", f"{commit}:{path}")

    def blob_sha256(self, commit: str, path: str) -> str:
        return hashlib.sha256(self.git("show", f"{commit}:{path}", binary=True)).hexdigest()

    def artifact(self, artifact_id: str) -> dict[str, object]:
        return {
            "artifact_id": artifact_id,
            "location_kind": "repository_blob",
            "path_or_uri": "app.txt",
            "source_commit_sha": self.candidate,
            "originating_candidate_sha": self.candidate,
            "sha256": self.blob_sha256(self.candidate, "app.txt"),
        }

    def gate(self, gate_id: str) -> dict[str, object]:
        scope = "cal" if gate_id.startswith("CAL-") else "combined"
        if gate_id == "EXTERNAL-APT-001":
            scope = "external"
        return {
            "gate_id": gate_id,
            "gate_version": "v1",
            "domain": "synthetic-focused-validation",
            "scope": scope,
            "required": True,
            "timing": ["lane_freeze"],
            "blocking_stage": "lane_handoff",
            "candidate": {
                "sha": self.candidate,
                "tree": self.candidate_tree,
                "base_sha": self.base,
            },
            "inputs": [self.artifact(f"{gate_id}-input")],
            "action": {
                "kind": "local_command",
                "command_argv": ["synthetic-gate", gate_id],
                "procedure": "",
            },
            "outputs": [self.artifact(f"{gate_id}-output")],
            "criteria": ["synthetic exact-SHA criterion passes"],
            "result": "pass",
            "omission": {"authority": "", "reason": ""},
            "owners": {
                "execution": "test execution owner",
                "architectural": "test architecture owner",
                "scientific": ["test scientific owner"],
                "evidence": "test evidence owner",
            },
            "evidence_reference": f"synthetic:{gate_id}",
            "claim_constraints": [],
            "started_at": "2026-08-13T12:00:00-04:00",
            "finished_at": "2026-08-13T12:00:01-04:00",
            "metrics": {
                "exit_status": 0,
                "unexpected_error_count": 0,
                "unexplained_required_output_failure_count": 0,
                "missing_required_output_count": 0,
                "skipped_required_comparison_count": 0,
            },
            "interface_contract": {
                "applicable": False,
                "interface_id": "",
                "producer_repository": "",
                "consumer_repository": "",
                "producer_commit_sha": None,
                "producer_tree_sha": None,
                "consumer_commit_sha": None,
                "consumer_tree_sha": None,
                "owner_repositories": [],
                "producer_artifact_schema": "",
                "consumer_preflight": "",
                "stable_scoped_keys": [],
                "exact_artifact_sha256": None,
                "mapping_sha256": None,
                "counterexamples": [],
                "readiness_status": "not_applicable",
                "blocking_dependencies": [],
                "mode_routes": [],
            },
            "apt_phase_contract": self.empty_apt_phase_contract(),
        }

    @staticmethod
    def empty_apt_phase_contract() -> dict[str, object]:
        return {
            "applicable": False,
            "phase_id": "",
            "readiness_status": "not_applicable",
            "software_revisions": [],
            "generation_id": "",
            "generation_root": "",
            "software_revision_set_sha256": None,
            "config_manifest_sha256": None,
            "raw_data_manifest_sha256": None,
            "cohort_manifest_sha256": None,
            "artifact_manifest_sha256": None,
            "component_manifest_sha256": None,
            "membership_sha256": None,
            "mapping_sha256": None,
            "transformation_sha256": None,
            "application_sha256": None,
            "quarantine_manifest_sha256": None,
            "rollback_manifest_sha256": None,
            "network_count": 0,
            "artifact_scope_count": 0,
            "complete_case_count": 0,
            "permutation_case_count": 0,
            "rejection_case_count": 0,
            "legacy_input_count": 0,
            "mixed_generation_count": 0,
            "selected_artifacts_all_contract_generated": False,
            "immutable_generation": False,
            "historical_evidence_only": False,
            "blocking_dependencies": [],
        }

    def recorded_boundary_gate(self, gate_id: str) -> dict[str, object]:
        row = self.gate(gate_id)
        row["scope"] = "external"
        row["required"] = False
        row["blocking_stage"] = "production_end_to_end"
        row["result"] = "conditioned"
        row["omission"] = {
            "authority": "test cross-repository owner",
            "reason": "synthetic packet preserves the unresolved external boundary",
        }
        producer, consumer = handoff.APT_INTERFACE_GATES[gate_id]
        allowed_endpoints = handoff.APT_ROUTE_ENDPOINTS[gate_id]
        route_producer, route_consumer = sorted(allowed_endpoints)[0]
        if gate_id == "APT-C-BEAMMAP-MATCHING-001":
            tolapt_role = "offline_downstream"
        elif gate_id == "APT-D-TOLAPT-TOLPROJ-PACKAGE-001":
            tolapt_role = "offline_package_exchange"
        elif gate_id in {
            "APT-E-TOLPROJ-TOLTECA-SELECTION-001",
            "APT-F-TOLTECA-CITLALI-TRANSPORT-001",
            "APT-G-CITLALI-ADMISSION-001",
        }:
            tolapt_role = "precomputed_input"
        else:
            tolapt_role = "not_in_path"
        beammap_only = gate_id in {
            "APT-A-RAW-KMP-CITLALI-AXIS-001",
            "APT-B-CITLALI-BEAMMAP-EXPORT-001",
            "APT-C-BEAMMAP-MATCHING-001",
        }
        routes = []
        for mode in sorted(handoff.MODES):
            applicable = mode == "beammap" if beammap_only else True
            routes.append(
                {
                    "mode": mode,
                    "applicable": applicable,
                    "actual_direction": (
                        f"{route_producer}->{route_consumer}"
                        if applicable
                        else "not_applicable"
                    ),
                    "route_producer_repository": route_producer if applicable else "",
                    "route_consumer_repository": route_consumer if applicable else "",
                    "tolapt_role": tolapt_role if applicable else "not_in_path",
                    "nonapplicability_authority": "" if applicable else "test mode owner",
                    "nonapplicability_reason": (
                        "" if applicable else "interface executes only in Beammap production"
                    ),
                }
            )
        row["interface_contract"] = {
            "applicable": True,
            "interface_id": gate_id,
            "producer_repository": producer,
            "consumer_repository": consumer,
            "producer_commit_sha": None,
            "producer_tree_sha": None,
            "consumer_commit_sha": None,
            "consumer_tree_sha": None,
            "owner_repositories": sorted({route_producer, route_consumer, producer, consumer}),
            "producer_artifact_schema": f"synthetic schema for {gate_id}",
            "consumer_preflight": f"synthetic preflight for {gate_id}",
            "stable_scoped_keys": ["artifact_id", "network_id", "typed_member_id"],
            "exact_artifact_sha256": None,
            "mapping_sha256": None,
            "counterexamples": [
                "permuted rows",
                "missing or duplicate member",
                "conflicting mapping",
            ],
            "readiness_status": "conditioned",
            "blocking_dependencies": ["APT-SAMPLE-NEW-CONTRACT-FIXTURES-001"],
            "mode_routes": routes,
        }
        return row

    def recorded_apt_phase_gate(self, gate_id: str) -> dict[str, object]:
        row = self.gate(gate_id)
        timing, stage = handoff.APT_LIBRARY_GENERATION_GATES[gate_id]
        row.update(
            {
                "scope": "external",
                "required": False,
                "timing": [timing],
                "blocking_stage": stage,
                "result": "blocked",
                "omission": {"authority": "", "reason": ""},
            }
        )
        contract = self.empty_apt_phase_contract()
        contract.update(
            {
                "applicable": True,
                "phase_id": gate_id,
                "readiness_status": "blocked",
                "historical_evidence_only": True,
                "blocking_dependencies": ["SYNTHETIC-TOLTECA"],
            }
        )
        row["apt_phase_contract"] = contract
        return row

    def make_packet(self) -> dict[str, object]:
        required_gates = sorted(
            handoff._required_gate_ids("cal_lane", False, {"point"})
        )
        patch_digest = hashlib.sha256(
            self.git("diff", "--binary", self.base, self.candidate, "--", binary=True)
        ).hexdigest()
        name_status_digest = hashlib.sha256(
            self.git(
                "diff",
                "--name-status",
                "--no-renames",
                "-z",
                self.base,
                self.candidate,
                "--",
                binary=True,
            )
        ).hexdigest()
        candidate_parent = self.git("rev-parse", f"{self.candidate}^")
        history = {
            "commit_sha": self.candidate,
            "parent_shas": [candidate_parent],
            "tree_sha": self.candidate_tree,
            "purpose": "synthetic application candidate",
            "categories": ["application", "test", "validation"],
            "import_disposition": "include_application",
        }
        approvals = [
            {
                "role": role,
                "owner": f"{role} test owner",
                "status": "approved",
                "candidate_sha": self.candidate,
                "candidate_tree": self.candidate_tree,
                "recorded_at": "2026-08-13T12:10:00-04:00",
                "conditions": [],
            }
            for role in (
                "lane_owner",
                "scientific_owner",
                "independent_auditor",
                "coordinator",
            )
        ]
        return {
            "schema_version": handoff.SCHEMA_VERSION,
            "packet_identity": {
                "packet_id": "SYNTHETIC-CAL-HANDOFF-001",
                "lane_id": "SYNTHETIC-CAL",
                "packet_kind": "cal_lane",
                "recorded_at": "2026-08-13T12:15:00-04:00",
                "lifecycle_state": "frozen",
                "target_stage": "lane_handoff",
            },
            "implementation_candidate": {
                "source_ref": "refs/heads/candidate",
                "snapshot_started_at": "2026-08-13T12:00:00-04:00",
                "snapshot_finished_at": "2026-08-13T12:01:00-04:00",
                "start_tip_sha": self.candidate,
                "end_tip_sha": self.candidate,
                "commit_sha": self.candidate,
                "parent_shas": [candidate_parent],
                "tree_sha": self.candidate_tree,
                "authorized_base_sha": self.base,
                "authorized_base_tree": self.base_tree,
                "merge_base_sha": self.base,
                "ahead_count": 1,
                "behind_count": 0,
                "standard_binary_patch_sha256": patch_digest,
                "name_status_sha256": name_status_digest,
                "embedded_version": "synthetic-candidate",
                "implementation_frozen": True,
                "worktree_clean": True,
            },
            "packet_container": {
                "kind": "uncommitted_packet",
                "commit_sha": None,
                "tree_sha": None,
                "separate_from_implementation": True,
            },
            "freeze_snapshot": {
                "refs": [
                    {
                        "name": "refs/heads/candidate",
                        "availability": "available",
                        "start_sha": self.candidate,
                        "end_sha": self.candidate,
                        "verify_local": True,
                    },
                    {
                        "name": "live-origin-candidate",
                        "availability": "unavailable",
                        "start_sha": None,
                        "end_sha": None,
                        "verify_local": False,
                    },
                ],
                "tip_moved": False,
            },
            "authority": {
                "convergence_base_decision": "test owner selected exact base",
                "owner_decision_refs": ["synthetic owner decision"],
                "authority_paths": [
                    {
                        "path": "authority.md",
                        "blob_sha": self.blob_oid(self.base, "authority.md"),
                    }
                ],
            },
            "repository_scope": {
                "citlali": "repairable_in_current_authorized_repository_scope",
                "tolproj": "repairable_only_in_separately_reviewed_repository_lane",
                "tolteca": "blocked_deferred_read_only",
                "compensation_elsewhere_allowed": False,
            },
            "ancestry": {
                "application_history": [history],
                "excluded_history": [
                    {
                        "commit_sha": self.audit,
                        "tree_sha": self.audit_tree,
                        "category": "audit",
                        "reason": "independent review remains outside application ancestry",
                    }
                ],
                "source_dependencies": [],
            },
            "changed_scope": {
                "paths": [
                    {
                        "status": "M",
                        "path": "app.txt",
                        "blob_sha": self.blob_oid(self.candidate, "app.txt"),
                        "category": "application",
                        "owner": "test application owner",
                    }
                ],
                "interfaces": [
                    {
                        "interface": "synthetic application boundary",
                        "path": "app.txt",
                        "architectural_owner": "test architecture owner",
                        "scientific_owners": ["test scientific owner"],
                        "lifecycle_owner": "test lifecycle owner",
                        "classification": "additive",
                        "required_evidence": ["synthetic focused gate"],
                        "future_stage_owner": "test combined owner",
                    }
                ],
                "affected_modes": ["point"],
                "governed_change_kinds": ["structural"],
            },
            "independent_disposition": {
                "review_commit_sha": self.audit,
                "review_tree_sha": self.audit_tree,
                "report_path": "review.md",
                "report_sha256": self.blob_sha256(self.audit, "review.md"),
                "axes": {
                    "scientific_contract": "approved",
                    "implementation": "conformant",
                    "validation_readiness": "complete",
                    "historical_fixture": "not_applicable",
                    "production": "fail_closed",
                    "verdict": "accept",
                },
                "findings": [],
            },
            "scientific_change": {
                "state": "none",
                "owner_basis": "synthetic structural change only",
                "ledger_path": "validation/intended_science_changes.json",
                "ledger_blob_sha": self.blob_oid(
                    self.candidate, "validation/intended_science_changes.json"
                ),
                "change_ids": [],
                "predecessor_epoch_id": None,
                "successor_epoch_id": None,
                "successor_epoch_status": "none",
                "profile_ids": [],
            },
            "gate_policy": {
                "required_gate_ids": required_gates,
                "required_modes": ["point"],
                "unity_required": False,
                "unity_omission": {
                    "authority": "test scientific owner",
                    "reason": "synthetic unit test uses no external system",
                },
            },
            "gate_results": [self.gate(gate_id) for gate_id in required_gates]
            + [
                self.recorded_boundary_gate(gate_id)
                for gate_id in sorted(handoff.APT_INTERFACE_GATES)
            ]
            + [
                self.recorded_apt_phase_gate(gate_id)
                for gate_id in sorted(handoff.APT_LIBRARY_GENERATION_GATES)
            ],
            "generated_evidence": [],
            "local_evidence": {
                "candidate_sha": self.candidate,
                "candidate_tree": self.candidate_tree,
                "gate_ids": required_gates,
                "clean_after_gates": True,
                "evidence_references": ["synthetic local evidence"],
            },
            "unity_evidence": {
                "required": False,
                "human_mediated_only": True,
                "codex_accessed_unity": False,
                "dependency_environment_sha256": None,
                "omission": {
                    "authority": "test scientific owner",
                    "reason": "synthetic unit test uses no external system",
                },
                "rows": [],
            },
            "external_dependencies": [
                {
                    "dependency_id": "SYNTHETIC-TOLAPT",
                    "repository": "tolapt",
                    "classification": "external_owner_dependency",
                    "status": "open",
                    "owner": "TolAPT owner",
                    "boundary": "immutable matching run and endpoint mapping",
                    "evidence_authority": "synthetic audit",
                    "finding_ids": ["TA-TEST"],
                    "exit_condition": "owner-reviewed producer contract",
                    "blocking_stage": "production_end_to_end",
                    "read_only": True,
                    "compensation_elsewhere_allowed": False,
                    "resolved_commit_sha": None,
                    "resolved_tree_sha": None,
                    "closure_evidence_sha256": None,
                },
                {
                    "dependency_id": "SYNTHETIC-TOLPROJ",
                    "repository": "tolproj",
                    "classification": "repairable_only_in_separately_reviewed_repository_lane",
                    "status": "open",
                    "owner": "TolProj owner",
                    "boundary": "selection transport",
                    "evidence_authority": "synthetic audit",
                    "finding_ids": ["TP-TEST"],
                    "exit_condition": "separate owner-reviewed repair",
                    "blocking_stage": "production_end_to_end",
                    "read_only": False,
                    "compensation_elsewhere_allowed": False,
                    "resolved_commit_sha": None,
                    "resolved_tree_sha": None,
                    "closure_evidence_sha256": None,
                },
                {
                    "dependency_id": "SYNTHETIC-TOLTECA",
                    "repository": "tolteca",
                    "classification": "blocked_deferred_at_tolteca",
                    "status": "deferred",
                    "owner": "TolTECA owner",
                    "boundary": "lossless selection and transport",
                    "evidence_authority": "synthetic audit",
                    "finding_ids": ["TV2-TEST"],
                    "exit_condition": "owner repair or replacement contract",
                    "blocking_stage": "production_end_to_end",
                    "read_only": True,
                    "compensation_elsewhere_allowed": False,
                    "resolved_commit_sha": None,
                    "resolved_tree_sha": None,
                    "closure_evidence_sha256": None,
                },
                {
                    "dependency_id": "SYNTHETIC-BEAMMAP-CONSUMER",
                    "repository": "toltec_beammap",
                    "classification": "external_owner_dependency",
                    "status": "open",
                    "owner": "toltec_beammap owner",
                    "boundary": "declared Beammap detector-binding consumer",
                    "evidence_authority": "synthetic audit",
                    "finding_ids": ["BM-R1"],
                    "exit_condition": "consumer uses declared producer mapping and fails closed",
                    "blocking_stage": "production_end_to_end",
                    "read_only": True,
                    "compensation_elsewhere_allowed": False,
                    "resolved_commit_sha": None,
                    "resolved_tree_sha": None,
                    "closure_evidence_sha256": None,
                },
            ],
            "claims": {
                "supported": ["source_static", "unit", "synthetic_counterexample"],
                "conditioned": ["external production boundary"],
                "prohibited": ["production end-to-end APT contract"],
                "cross_repository_apt_conformance": False,
                "production_end_to_end_apt_contract": False,
                "refactor_apt_generation_selected": False,
                "refactor_reductions_regenerated": False,
                "legacy_lineage_used_as_refactor_input": False,
                "legacy_selection_equivalence_required": False,
                "new_contract_sample_artifact_milestone_met": False,
                "real_end_to_end_apt_chain_conformance": False,
                "scientific_readiness": False,
                "production_readiness": False,
                "refactor_apt_library_validated": False,
            },
            "attestations": {
                "application_history_separated": True,
                "zero_unexplained_required_output_failures": True,
                "zero_unexpected_error_logs": True,
                "no_skipped_required_comparisons": True,
                "requested_effective_observation_realized_checked": True,
                "product_inventory_checked": True,
                "scientific_conventions_checked": True,
                "same_sha_local": True,
                "same_sha_local_unity": False,
                "compensating_identity_or_admission_weakening": False,
            },
            "approvals": approvals,
        }

    def validate(self, packet=None, *, require_ready: bool = False):
        return handoff.validate_packet(
            copy.deepcopy(self.packet if packet is None else packet),
            repo_root=self.repo,
            expected_sha=self.candidate if require_ready else None,
            require_ready=require_ready,
            candidate_worktree=self.repo if require_ready else None,
        )

    def write_packet(self, packet: dict[str, object], name: str = "packet.json") -> Path:
        path = Path(self.temporary.name) / name
        path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        return path

    def test_schema_is_valid_json_with_expected_identity(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            handoff.SCHEMA_VERSION,
        )
        self.assertEqual(
            set(schema["$defs"]["blocking_stage"]["enum"]),
            handoff.BLOCKING_STAGES,
        )
        self.assertEqual(
            set(
                schema["$defs"]["gate_result"]["properties"]["timing"]["items"][
                    "enum"
                ]
            ),
            handoff.GATE_TIMINGS,
        )
        self.assertIn(
            "apt_phase_contract", schema["$defs"]["gate_result"]["required"]
        )
        for claim in (
            "new_contract_sample_artifact_milestone_met",
            "real_end_to_end_apt_chain_conformance",
            "scientific_readiness",
            "production_readiness",
            "refactor_apt_library_validated",
        ):
            self.assertIn(claim, schema["$defs"]["claims"]["required"])

    def test_valid_ready_packet_is_derived_ready(self) -> None:
        self.assertEqual(
            handoff._commit_facts(self.repo, self.base, "synthetic root"),
            ([], self.base_tree),
        )
        self.assertEqual(
            handoff._commit_facts(self.repo, self.candidate, "synthetic non-root"),
            ([self.base], self.candidate_tree),
        )
        merge_commit = self.git(
            "commit-tree",
            self.audit_tree,
            "-p",
            self.candidate,
            "-p",
            self.audit,
            "-m",
            "synthetic two-parent facts",
        )
        self.assertEqual(
            handoff._commit_facts(self.repo, merge_commit, "synthetic merge"),
            ([self.candidate, self.audit], self.audit_tree),
        )
        result = self.validate(require_ready=True)
        self.assertTrue(result["ready"])
        self.assertEqual(result["blockers"], [])

    def test_valid_preparing_packet_is_not_ready(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["packet_identity"]["lifecycle_state"] = "preparing"
        result = self.validate(packet)
        self.assertFalse(result["ready"])
        self.assertIn("lifecycle:preparing", result["blockers"])

    def test_required_nonpass_gate_is_valid_but_blocking(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["gate_results"][0]["result"] = "not_run"
        result = self.validate(packet)
        self.assertFalse(result["ready"])
        self.assertTrue(any(item.startswith("gate:") for item in result["blockers"]))

    def test_rejects_unknown_property(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["unexpected"] = True
        with self.assertRaisesRegex(handoff.PacketError, "unknown fields"):
            self.validate(packet)

    def test_rejects_duplicate_json_key(self) -> None:
        path = Path(self.temporary.name) / "duplicate.json"
        path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
        with self.assertRaisesRegex(handoff.PacketError, "duplicate JSON key"):
            handoff.load_packet(path)

    def test_rejects_nonfinite_json_number(self) -> None:
        path = Path(self.temporary.name) / "nan.json"
        path.write_text('{"value": NaN}', encoding="utf-8")
        with self.assertRaisesRegex(handoff.PacketError, "non-finite"):
            handoff.load_packet(path)

    def test_rejects_placeholder(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["authority"]["owner_decision_refs"] = ["TBD"]
        with self.assertRaisesRegex(handoff.PacketError, "placeholder"):
            self.validate(packet)

    def test_rejects_abbreviated_candidate_sha(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["implementation_candidate"]["commit_sha"] = self.candidate[:9]
        with self.assertRaisesRegex(handoff.PacketError, "40-character"):
            self.validate(packet)

    def test_rejects_false_parent_and_tree(self) -> None:
        for field in ("parent_shas", "tree_sha"):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet["implementation_candidate"][field] = (
                    [self.audit] if field == "parent_shas" else self.audit_tree
                )
                with self.assertRaisesRegex(handoff.PacketError, field):
                    self.validate(packet)

    def test_rejects_false_patch_digests(self) -> None:
        for field in ("standard_binary_patch_sha256", "name_status_sha256"):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet["implementation_candidate"][field] = "0" * 64
                with self.assertRaisesRegex(handoff.PacketError, field):
                    self.validate(packet)

    def test_rejects_incomplete_history(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["ancestry"]["application_history"] = []
        with self.assertRaisesRegex(handoff.PacketError, "base..candidate"):
            self.validate(packet)

    def test_rejects_excluded_candidate_ancestor(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["ancestry"]["excluded_history"][0].update(
            {
                "commit_sha": self.base,
                "tree_sha": self.base_tree,
            }
        )
        with self.assertRaisesRegex(handoff.PacketError, "candidate ancestor"):
            self.validate(packet)

    def test_contaminating_ancestor_blocks_readiness(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["ancestry"]["source_dependencies"] = [
            {
                "dependency_id": "CONTAMINATION",
                "repository": "citlali",
                "commit_sha": self.candidate,
                "classification": "contaminating",
                "disposition": "imported",
                "owner": "test owner",
                "reason": "synthetic contaminated ancestry",
            }
        ]
        result = self.validate(packet)
        self.assertFalse(result["ready"])
        self.assertIn("dependency:contaminating_ancestor", result["blockers"])

    def test_reconstruct_required_history_blocks_readiness(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["ancestry"]["application_history"][0]["import_disposition"] = (
            "reconstruct_required"
        )
        result = self.validate(packet)
        self.assertFalse(result["ready"])

    def test_rejects_changed_path_blob_mismatch(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["changed_scope"]["paths"][0]["blob_sha"] = self.base_tree
        with self.assertRaisesRegex(handoff.PacketError, "blob_sha"):
            self.validate(packet)

    def test_rejects_missing_required_gate_row(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["gate_results"].pop()
        with self.assertRaisesRegex(handoff.PacketError, "missing mandatory gate rows"):
            self.validate(packet)

    def test_rejects_missing_sample_artifact_milestone_row(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["gate_results"] = [
            row
            for row in packet["gate_results"]
            if row["gate_id"] != "APT-SAMPLE-NEW-CONTRACT-FIXTURES-001"
        ]
        with self.assertRaisesRegex(handoff.PacketError, "missing mandatory gate rows"):
            self.validate(packet)

    def test_rejects_gate_bound_to_other_sha(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["gate_results"][0]["candidate"]["sha"] = self.base
        with self.assertRaisesRegex(handoff.PacketError, "differs from implementation candidate"):
            self.validate(packet)

    def test_rejects_passing_gate_with_error_or_required_output_failure(self) -> None:
        for field in (
            "unexpected_error_count",
            "unexplained_required_output_failure_count",
            "missing_required_output_count",
            "skipped_required_comparison_count",
        ):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet["gate_results"][0]["metrics"][field] = 1
                with self.assertRaisesRegex(handoff.PacketError, field):
                    self.validate(packet)

    def test_rejects_omitted_gate_without_authority(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["gate_results"][0]["result"] = "omitted"
        with self.assertRaisesRegex(handoff.PacketError, "requires authority and reason"):
            self.validate(packet)

    def test_science_governed_change_requires_declaration(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["changed_scope"]["governed_change_kinds"] = ["schema"]
        with self.assertRaisesRegex(handoff.PacketError, "requires declared"):
            self.validate(packet)

    def test_rejects_unknown_science_change_id(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["changed_scope"]["governed_change_kinds"] = ["schema"]
        packet["scientific_change"].update(
            {
                "state": "declared",
                "change_ids": ["unknown-change"],
                "predecessor_epoch_id": "epoch-0",
                "successor_epoch_id": "epoch-1",
                "successor_epoch_status": "preparing",
                "profile_ids": ["profile-1"],
            }
        )
        with self.assertRaisesRegex(handoff.PacketError, "unknown accepted ledger IDs"):
            self.validate(packet)

    def test_rejects_tolteca_scope_relaxation(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["external_dependencies"][2]["classification"] = (
            "external_owner_dependency"
        )
        with self.assertRaisesRegex(handoff.PacketError, "blocked_deferred_at_tolteca"):
            self.validate(packet)

    def test_rejects_tolproj_scope_relaxation(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["external_dependencies"][1]["classification"] = (
            "repairable_in_current_authorized_repository_scope"
        )
        with self.assertRaisesRegex(handoff.PacketError, "separate reviewed lane"):
            self.validate(packet)

    def test_rejects_end_to_end_apt_claim(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["claims"]["production_end_to_end_apt_contract"] = True
        with self.assertRaisesRegex(handoff.PacketError, "must be false"):
            self.validate(packet)

    def test_rejects_cross_repository_apt_claim(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["claims"]["cross_repository_apt_conformance"] = True
        with self.assertRaisesRegex(handoff.PacketError, "must be false"):
            self.validate(packet)

    def test_rejects_missing_beammap_bm_r1_dependency(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["external_dependencies"][3]["finding_ids"] = ["OTHER"]
        with self.assertRaisesRegex(handoff.PacketError, "BM-R1"):
            self.validate(packet)

    def test_rejects_coarse_or_incomplete_apt_interface_row(self) -> None:
        packet = copy.deepcopy(self.packet)
        row = next(
            item
            for item in packet["gate_results"]
            if item["gate_id"] == "APT-D-TOLAPT-TOLPROJ-PACKAGE-001"
        )
        row["interface_contract"]["mode_routes"].pop()
        with self.assertRaisesRegex(handoff.PacketError, "all modes"):
            self.validate(packet)

    def test_rejects_false_apt_route_direction_or_inline_role(self) -> None:
        for mutation in ("direction", "inline"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.packet)
                row = next(
                    item
                    for item in packet["gate_results"]
                    if item["gate_id"] == "APT-D-TOLAPT-TOLPROJ-PACKAGE-001"
                )
                route = row["interface_contract"]["mode_routes"][0]
                if mutation == "direction":
                    route["actual_direction"] = "tolproj->citlali"
                    pattern = "explicit endpoint direction"
                else:
                    route["tolapt_role"] = "inline"
                    pattern = "unsupported value"
                with self.assertRaisesRegex(handoff.PacketError, pattern):
                    self.validate(packet)

    def test_rejects_legacy_or_mixed_input_in_refactor_generation(self) -> None:
        for field in ("legacy_input_count", "mixed_generation_count"):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                row = next(
                    item
                    for item in packet["gate_results"]
                    if item["gate_id"] == "APT-LIB-IMMUTABLE-GENERATION-001"
                )
                row["apt_phase_contract"][field] = 1
                with self.assertRaisesRegex(handoff.PacketError, "prohibit"):
                    self.validate(packet)

    def test_rejects_wrong_apt_phase_timing_or_stage(self) -> None:
        for field, value, pattern in (
            ("timing", ["lane_freeze"], "requires"),
            ("blocking_stage", "production_end_to_end", "requires"),
        ):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                row = next(
                    item
                    for item in packet["gate_results"]
                    if item["gate_id"] == "APT-SAMPLE-NEW-CONTRACT-FIXTURES-001"
                )
                row[field] = value
                with self.assertRaisesRegex(handoff.PacketError, pattern):
                    self.validate(packet)

    def test_rejects_currently_forbidden_readiness_claims(self) -> None:
        for field in (
            "new_contract_sample_artifact_milestone_met",
            "real_end_to_end_apt_chain_conformance",
            "scientific_readiness",
            "production_readiness",
            "refactor_apt_library_validated",
            "refactor_apt_generation_selected",
            "refactor_reductions_regenerated",
            "legacy_lineage_used_as_refactor_input",
            "legacy_selection_equivalence_required",
        ):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet["claims"][field] = True
                with self.assertRaisesRegex(handoff.PacketError, "must be false"):
                    self.validate(packet)

    def test_noninline_tolapt_mode_requires_explicit_route(self) -> None:
        packet = copy.deepcopy(self.packet)
        row = next(
            item
            for item in packet["gate_results"]
            if item["gate_id"] == "APT-D-TOLAPT-TOLPROJ-PACKAGE-001"
        )
        route = row["interface_contract"]["mode_routes"][0]
        route.update(
            {
                "applicable": False,
                "actual_direction": "not_applicable",
                "route_producer_repository": "",
                "route_consumer_repository": "",
                "tolapt_role": "not_in_path",
                "nonapplicability_authority": "mode workflow owner",
                "nonapplicability_reason": "mode uses a direct TolProj path",
            }
        )
        result = self.validate(packet)
        self.assertTrue(result["ready"])

    def test_rejects_mixed_sha_human_unity_row_even_when_optional(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["unity_evidence"]["rows"] = [
            {
                "mode": "point",
                "profile_id": "synthetic-profile",
                "run_id": "synthetic-run",
                "candidate_sha": self.base,
                "candidate_tree": self.base_tree,
                "embedded_version": "synthetic",
                "binary_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "input_manifest_sha256": "3" * 64,
                "output_manifest_sha256": "4" * 64,
                "report_sha256": "5" * 64,
                "log_sha256": "6" * 64,
                "retrieved_at": "2026-08-13T12:20:00-04:00",
                "retrieved_by": "authorized human",
                "retrieval_source": "human supplied synthetic report",
                "provided_by_authorized_human": True,
                "result": "pass",
                "unexpected_error_count": 0,
                "unexplained_required_output_failure_count": 0,
                "missing_required_output_count": 0,
                "skipped_required_comparison_count": 0,
            }
        ]
        with self.assertRaisesRegex(handoff.PacketError, "mixed-SHA"):
            self.validate(packet)

    def test_require_ready_rejects_dirty_candidate_worktree(self) -> None:
        (self.repo / "app.txt").write_text("dirty\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(handoff.PacketError, "not clean"):
                self.validate(require_ready=True)
        finally:
            (self.repo / "app.txt").write_text("candidate\n", encoding="utf-8")

    def test_recorded_command_is_never_executed(self) -> None:
        sentinel = Path(self.temporary.name) / "must-not-exist"
        packet = copy.deepcopy(self.packet)
        packet["gate_results"][0]["action"]["command_argv"] = [
            "touch",
            str(sentinel),
        ]
        result = self.validate(packet)
        self.assertTrue(result["ready"])
        self.assertFalse(sentinel.exists())

    def test_cli_exit_codes_ready_blocked_and_invalid(self) -> None:
        ready_path = self.write_packet(self.packet, "ready.json")
        blocked = copy.deepcopy(self.packet)
        blocked["gate_results"][0]["result"] = "not_run"
        blocked_path = self.write_packet(blocked, "blocked.json")
        invalid = copy.deepcopy(self.packet)
        invalid["implementation_candidate"]["tree_sha"] = "0" * 40
        invalid_path = self.write_packet(invalid, "invalid.json")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            ready_code = handoff.main(
                [
                    str(ready_path),
                    "--repo-root",
                    str(self.repo),
                    "--candidate-worktree",
                    str(self.repo),
                    "--expected-sha",
                    self.candidate,
                    "--require-ready",
                ]
            )
            blocked_code = handoff.main(
                [
                    str(blocked_path),
                    "--repo-root",
                    str(self.repo),
                    "--candidate-worktree",
                    str(self.repo),
                    "--expected-sha",
                    self.candidate,
                    "--require-ready",
                ]
            )
            invalid_code = handoff.main(
                [str(invalid_path), "--repo-root", str(self.repo)]
            )
        self.assertEqual((ready_code, blocked_code, invalid_code), (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
