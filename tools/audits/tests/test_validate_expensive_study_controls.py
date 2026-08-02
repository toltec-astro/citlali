"""Adversarial launch-gate tests for costly numerical-study controls.

Every negative case starts from the same launchable synthetic bundle.  The
test rewrites the register/preflight digest chain after mutation so a semantic
failure cannot be mistaken for a stale fixture binding unless stale identity
or digest binding is the behavior under test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "tools" / "audits" / "validate_expensive_study_controls.py"
POSITIVE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "positive"
PYTHON = Path.home() / "tolteca" / "bin" / "python"
DocumentSet = dict[str, dict[str, Any]]
Mutation = Callable[[DocumentSet], None]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_fingerprint(comparison: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in comparison.items()
        if key != "comparison_fingerprint"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition(documents: DocumentSet, condition_id: str) -> dict[str, Any]:
    return next(
        item
        for item in documents["register"]["conditions"]
        if item["condition_id"] == condition_id
    )


def _condition_coverage(documents: DocumentSet, condition_id: str) -> dict[str, Any]:
    return next(
        item
        for item in documents["preflight"]["coverage"]["condition_coverage"]
        if item["condition_id"] == condition_id
    )


def _discovered_condition(documents: DocumentSet, condition_id: str) -> dict[str, Any]:
    return next(
        item
        for item in documents["preflight"]["guard_inventory"]["discovered_conditions"]
        if item["condition_id"] == condition_id
    )


def _synchronize_comparison(documents: DocumentSet, condition_id: str) -> None:
    condition = _condition(documents, condition_id)
    comparison = condition["comparison"]
    comparison["comparison_fingerprint"] = _comparison_fingerprint(comparison)
    observation = _condition_coverage(documents, condition_id)["comparison_observation"]
    observation["comparison_fingerprint"] = comparison["comparison_fingerprint"]
    observation["observed_threshold"] = copy.deepcopy(comparison["threshold"])


class ExpensiveStudyControlAdversarialTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="citlali-expensive-study-gate-test-"
        )
        self.bundle = Path(self._temporary.name) / "bundle"
        shutil.copytree(POSITIVE_FIXTURE, self.bundle)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _load(self) -> DocumentSet:
        return {
            name: yaml.safe_load((self.bundle / f"{name}.yaml").read_text(encoding="utf-8"))
            for name in ("register", "preflight", "certificate")
        }

    def _write_with_fresh_control_bindings(self, documents: DocumentSet) -> None:
        register_path = self.bundle / "register.yaml"
        preflight_path = self.bundle / "preflight.yaml"
        certificate_path = self.bundle / "certificate.yaml"

        register_path.write_text(
            yaml.safe_dump(documents["register"], sort_keys=False), encoding="utf-8"
        )
        register_digest = _sha256(register_path)
        documents["preflight"]["bindings"]["register"]["sha256"] = register_digest
        documents["certificate"]["bindings"]["register"]["sha256"] = register_digest

        preflight_path.write_text(
            yaml.safe_dump(documents["preflight"], sort_keys=False), encoding="utf-8"
        )
        documents["certificate"]["bindings"]["preflight_report"]["sha256"] = _sha256(
            preflight_path
        )
        certificate_path.write_text(
            yaml.safe_dump(documents["certificate"], sort_keys=False), encoding="utf-8"
        )

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(PYTHON),
                str(VALIDATOR),
                "--register",
                str(self.bundle / "register.yaml"),
                "--preflight",
                str(self.bundle / "preflight.yaml"),
                "--certificate",
                str(self.bundle / "certificate.yaml"),
                "--root",
                str(self.bundle),
                "--launch-gate",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _mutate_and_run(self, mutation: Mutation) -> subprocess.CompletedProcess[str]:
        documents = self._load()
        mutation(documents)
        self._write_with_fresh_control_bindings(documents)
        return self._run()

    def _assert_rejected(
        self, mutation: Mutation, expected_message: str
    ) -> subprocess.CompletedProcess[str]:
        result = self._mutate_and_run(mutation)
        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(expected_message, result.stderr)
        return result

    def test_positive_synthetic_bundle_passes_launch_gate(self) -> None:
        result = self._run()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("EXPENSIVE STUDY CONTROL LAUNCH GATE: PASS", result.stdout)

    def test_hidden_source_hard_stop_cannot_be_registered_as_warning(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-D-DIAGNOSTIC")["source_sites"][0][
                "implemented_action"
            ] = "hard_stop"
            _discovered_condition(documents, "SYN-D-DIAGNOSTIC")[
                "implemented_actions"
            ] = ["hard_stop"]

        self._assert_rejected(mutate, "source-site implemented action(s)")

    def test_false_data_dependent_classification_is_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-C-SCIENCE")["data_dependent"] = False

        self._assert_rejected(mutate, "deterministic classification conflicts")

    def test_data_dependent_guard_requires_fault_injection(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition_coverage(documents, "SYN-C-SCIENCE")[
                "synthetic_or_fault_injection"
            ] = None

        self._assert_rejected(mutate, "lacks a passing, registered synthetic/fault-injection")

    def test_arbitrary_none_nonfinite_and_negative_tolerances_are_rejected(self) -> None:
        cases: tuple[tuple[str, Any, str, bool], ...] = (
            ("arbitrary", {"canonical_literal": "arbitrary", "representation": "text", "units": "ULP"}, "not valid under any of the given schemas", False),
            ("none", None, "numerical comparison lacks a threshold", True),
            ("nonfinite", {"canonical_literal": "NaN", "representation": "binary64", "units": "ULP"}, "not valid under any of the given schemas", False),
            ("negative", {"canonical_literal": "-1", "representation": "integer", "units": "ULP"}, "must be nonnegative", True),
        )
        baseline = self._load()
        for label, threshold, expected, synchronize in cases:
            with self.subTest(label=label):
                documents = copy.deepcopy(baseline)
                _condition(documents, "SYN-B-ULP")["comparison"]["threshold"] = threshold
                if synchronize:
                    _synchronize_comparison(documents, "SYN-B-ULP")
                self._write_with_fresh_control_bindings(documents)
                result = self._run()
                self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)

    def test_changed_canonical_threshold_requires_new_fingerprint(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["comparison"]["threshold"][
                "canonical_literal"
            ] = "3"

        self._assert_rejected(mutate, "comparison fingerprint mismatch")

    def test_preflight_threshold_literal_must_match_frozen_register(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition_coverage(documents, "SYN-B-ULP")["comparison_observation"][
                "observed_threshold"
            ]["canonical_literal"] = "3"

        self._assert_rejected(mutate, "observed threshold literal")

    def test_preflight_fingerprint_must_match_frozen_register(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition_coverage(documents, "SYN-B-ULP")["comparison_observation"][
                "comparison_fingerprint"
            ] = "1" * 64

        self._assert_rejected(mutate, "comparison fingerprint does not match frozen register")

    def test_incomplete_registered_branch_coverage_is_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition_coverage(documents, "SYN-B-ULP")["observed_branches"] = [
                "within_bound"
            ]

        self._assert_rejected(mutate, "required branch coverage is incomplete")

    def test_source_site_digest_must_match_source_file(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["source_sites"][0][
                "source_sha256"
            ] = "1" * 64

        self._assert_rejected(mutate, "source SHA-256 mismatch")

    def test_source_identity_must_match_every_frozen_binding(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["register"]["study"]["runner"][
                "source_identity"
            ] = "different-runner-identity"

        self._assert_rejected(mutate, "source identity does not match the frozen binding")

    def test_unregistered_abort_capable_guard_is_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["preflight"]["guard_inventory"][
                "unregistered_abort_capable_ids"
            ].append("HIDDEN-HARD-STOP")

        self._assert_rejected(mutate, "unregistered abort-capable guards present")

    def test_partial_frozen_tuple_coverage_is_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            coverage = documents["preflight"]["coverage"]
            coverage["exercised_tuple_count"] = 0
            coverage["all_frozen_tuples_complete"] = False

        self._assert_rejected(mutate, "exercised tuple count is incomplete")

    def test_partial_condition_coverage_is_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            coverage = documents["preflight"]["coverage"]["condition_coverage"]
            coverage[:] = [
                item for item in coverage if item["condition_id"] != "SYN-D-DIAGNOSTIC"
            ]

        self._assert_rejected(mutate, "IDs do not equal frozen register")

    def test_preflight_cannot_make_scientific_model_calls(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["preflight"]["execution"]["scientific_model_calls"] = 1

        self._assert_rejected(mutate, "dry-run made scientific model calls")

    def test_class_a_rejects_approximate_numerical_semantics(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            comparison = _condition(documents, "SYN-A-IDENTITY")["comparison"]
            comparison["semantics"] = "absolute"
            comparison["operator"] = "less_than_or_equal"
            comparison["threshold"] = {
                "canonical_literal": "0",
                "representation": "exact zero",
                "units": "fractional",
            }
            _synchronize_comparison(documents, "SYN-A-IDENTITY")

        self._assert_rejected(mutate, "Class A uses incompatible absolute semantics")

    def test_class_b_rejects_unsupported_derivation(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["threshold_derivation"][
                "kind"
            ] = "exact_identity"

        self._assert_rejected(mutate, "Class B lacks an analytic")

    def test_class_b_derived_bound_without_metric_mapping_passes(self) -> None:
        documents = self._load()
        effect = _condition(documents, "SYN-B-ULP")["maximum_propagated_effect"]
        effect["status"] = "not_applicable_derived_correctness"
        effect["affected_metrics"] = []
        effect["exact_integrity_rationale"] = None
        effect["derived_correctness_rationale"] = (
            "The registered ULP derivation proves the numerical invariant directly; no "
            "scientific metric propagation is claimed."
        )
        self._write_with_fresh_control_bindings(documents)
        result = self._run()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_class_b_derived_only_rejects_none_diagnostic(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            condition = _condition(documents, "SYN-B-ULP")
            condition["threshold_derivation"]["kind"] = "none_diagnostic"
            effect = condition["maximum_propagated_effect"]
            effect["status"] = "not_applicable_derived_correctness"
            effect["affected_metrics"] = []
            effect["exact_integrity_rationale"] = None
            effect["derived_correctness_rationale"] = "Unsupported assertion of correctness."

        self._assert_rejected(mutate, "Class B lacks an analytic")

    def test_class_b_cannot_masquerade_as_scientific_failure(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            condition = _condition(documents, "SYN-B-ULP")
            condition["action"] = "scientific_failure"
            condition["source_sites"][0]["implemented_action"] = "scientific_failure"
            _discovered_condition(documents, "SYN-B-ULP")["implemented_actions"] = [
                "scientific_failure"
            ]

        self._assert_rejected(mutate, "Class B may not masquerade")

    def test_class_c_cannot_be_an_execution_hard_stop(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            condition = _condition(documents, "SYN-C-SCIENCE")
            condition["action"] = "hard_stop"
            condition["source_sites"][0]["implemented_action"] = "hard_stop"
            _discovered_condition(documents, "SYN-C-SCIENCE")["implemented_actions"] = [
                "hard_stop"
            ]

        self._assert_rejected(mutate, "Class C must yield scientific_failure")

    def test_class_d_unquantified_diagnostic_has_no_veto(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            condition = _condition(documents, "SYN-D-DIAGNOSTIC")
            condition["action"] = "hard_stop"
            condition["source_sites"][0]["implemented_action"] = "hard_stop"
            condition["approval"] = {
                "required": True,
                "role": "audit_manager",
                "status": "approved",
                "record": "SYN-APPROVAL-D",
            }
            _discovered_condition(documents, "SYN-D-DIAGNOSTIC")[
                "implemented_actions"
            ] = ["hard_stop"]
            documents["preflight"]["guard_inventory"][
                "abort_capable_condition_ids"
            ].append("SYN-D-DIAGNOSTIC")

        self._assert_rejected(mutate, "Class D conditions are warning-only")

    def test_class_d_cannot_be_scientific_failure(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            condition = _condition(documents, "SYN-D-DIAGNOSTIC")
            condition["action"] = "scientific_failure"
            condition["source_sites"][0]["implemented_action"] = "scientific_failure"
            condition["approval"] = {
                "required": True,
                "role": "scientific_owner",
                "status": "approved",
                "record": "SYN-INVALID-D-APPROVAL",
            }
            _discovered_condition(documents, "SYN-D-DIAGNOSTIC")[
                "implemented_actions"
            ] = ["scientific_failure"]
            documents["preflight"]["guard_inventory"][
                "abort_capable_condition_ids"
            ].append("SYN-D-DIAGNOSTIC")

        self._assert_rejected(mutate, "Class D conditions are warning-only")

    def test_approval_role_rejects_arbitrary_authority(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["approval"]["role"] = "project_owner"

        self._assert_rejected(mutate, "is not one of")

    def test_class_c_requires_scientific_owner_approval(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-C-SCIENCE")["approval"]["role"] = "audit_manager"

        self._assert_rejected(mutate, "requires approved role scientific_owner")

    def test_warning_approval_must_be_not_required(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-D-DIAGNOSTIC")["approval"]["role"] = "audit_manager"

        self._assert_rejected(mutate, "warning approval must be not_required")

    def test_nonwarning_guard_requires_approved_authority(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            approval = _condition(documents, "SYN-B-ULP")["approval"]
            approval["status"] = "pending"
            approval["record"] = None

        self._assert_rejected(mutate, "required approval must be approved")

    def test_quantified_effect_requires_valid_complete_metric_bounds(self) -> None:
        def empty_metrics(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["maximum_propagated_effect"][
                "affected_metrics"
            ] = []

        def unknown_metric(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["maximum_propagated_effect"][
                "affected_metrics"
            ][0]["metric_id"] = "UNKNOWN-METRIC"

        def negative_bound(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["maximum_propagated_effect"][
                "affected_metrics"
            ][0]["bound"]["canonical_literal"] = "-1e-15"

        def nonfinite_bound(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["maximum_propagated_effect"][
                "affected_metrics"
            ][0]["bound"]["canonical_literal"] = "NaN"

        cases: tuple[tuple[str, Mutation, str], ...] = (
            ("empty", empty_metrics, "quantified effect must enumerate every affected metric"),
            ("unknown", unknown_metric, "unknown affected metric UNKNOWN-METRIC"),
            ("negative", negative_bound, "numeric bound must be nonnegative"),
            ("nonfinite", nonfinite_bound, "does not match"),
        )
        baseline = self._load()
        for label, mutation, expected in cases:
            with self.subTest(label=label):
                documents = copy.deepcopy(baseline)
                mutation(documents)
                self._write_with_fresh_control_bindings(documents)
                result = self._run()
                self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)

    def test_ulp_threshold_must_be_an_integer(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            _condition(documents, "SYN-B-ULP")["comparison"]["threshold"][
                "canonical_literal"
            ] = "1.5"
            _synchronize_comparison(documents, "SYN-B-ULP")

        self._assert_rejected(mutate, "ULP threshold must be a nonnegative integer")

    def test_raw_parser_evaluator_and_decision_states_must_be_separable(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            separation = documents["register"]["study"]["raw_evaluator_separation"]
            separation["raw_model_stage_separable"] = False
            separation["parser_admission_validity_state_artifact"] = separation[
                "raw_validity_state_artifact"
            ]

        result = self._assert_rejected(mutate, "raw model stage must be separable")
        self.assertIn("state artifacts must be distinct", result.stderr)

    def test_duplicate_condition_ids_are_rejected(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["register"]["conditions"][-1]["condition_id"] = documents[
                "register"
            ]["conditions"][0]["condition_id"]

        self._assert_rejected(mutate, "condition_id values must be unique")

    def test_guard_census_site_count_must_match_registered_sites(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["register"]["guard_site_census"]["site_count"] += 1

        self._assert_rejected(mutate, "does not equal the number of registered source sites")

    def test_guard_census_condition_ids_must_match_registered_conditions(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["register"]["guard_site_census"]["condition_ids"].pop()

        self._assert_rejected(mutate, "condition IDs do not equal registered conditions")

    def test_new_or_revised_harness_requires_condition_id_dispatcher(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["register"]["guard_site_census"]["routing"] = "static_inventory"
            documents["preflight"]["guard_inventory"]["method"] = "static_inventory"

        self._assert_rejected(mutate, "new or revised harness must use the condition-ID dispatcher")

    def test_legacy_static_inventory_requires_approved_exception(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            census = documents["register"]["guard_site_census"]
            census["harness_kind"] = "legacy"
            census["routing"] = "static_inventory"
            census["legacy_static_exception"] = {
                "required": True,
                "status": "pending",
                "manager_record": None,
            }
            documents["preflight"]["guard_inventory"]["method"] = "static_inventory"

        self._assert_rejected(mutate, "requires an approved manager exception record")

    def test_placeholder_and_all_zero_digests_are_rejected(self) -> None:
        cases = (
            ("placeholder", "TODO", "does not match"),
            ("all_zero", "0" * 64, "all-zero SHA-256 placeholder is forbidden"),
        )
        baseline = self._load()
        for label, digest, expected in cases:
            with self.subTest(label=label):
                documents = copy.deepcopy(baseline)
                documents["register"]["study"]["protocol"]["sha256"] = digest
                self._write_with_fresh_control_bindings(documents)
                result = self._run()
                self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)

    def test_timestamp_order_requires_review_after_preflight(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["certificate"]["independent_review"][
                "completed_at_utc"
            ] = "2026-08-02T11:00:00Z"

        self._assert_rejected(mutate, "completion predates the preflight report")

    def test_timestamp_order_requires_approval_after_review(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["certificate"]["manager_approval"][
                "approved_at_utc"
            ] = "2026-08-02T12:30:00Z"

        self._assert_rejected(mutate, "approval predates independent review completion")

    def test_ready_certificate_rejects_false_attestation(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["certificate"]["attestations"][
                "engineering_diagnostics_have_no_veto"
            ] = False

        self._assert_rejected(mutate, "all readiness attestations must be true")

    def test_readiness_certificate_cannot_claim_launch_authorization(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["certificate"]["readiness_scope"]["launch_authorization"] = True

        self._assert_rejected(mutate, "False was expected")

    def test_ready_certificate_requires_approved_manager(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            approval = documents["certificate"]["manager_approval"]
            approval["decision"] = "pending"
            approval["approved_at_utc"] = None

        self._assert_rejected(mutate, "audit manager has not approved readiness")

    def test_ready_certificate_requires_clean_independent_review(self) -> None:
        def mutate(documents: DocumentSet) -> None:
            documents["certificate"]["independent_review"]["status"] = "pending"

        self._assert_rejected(mutate, "independent review is not cleanly approved")


if __name__ == "__main__":
    unittest.main()
