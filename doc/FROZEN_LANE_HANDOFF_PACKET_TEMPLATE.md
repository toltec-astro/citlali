# Frozen-Lane Handoff Packet Template

Use this template only after an implementation candidate is frozen and an
independent review has dispositioned that exact commit and tree. It records a
candidate; it does not accept, merge, promote, or release one. A later
documentation-only packet-container commit is not the tested implementation
SHA.

The machine record is JSON conforming to
[`frozen_lane_handoff_packet.schema.json`](../validation/frozen_lane_handoff_packet.schema.json).
Validate it with:

```sh
$HOME/tolteca/bin/python \
  tools/baseline/validate_frozen_lane_handoff.py PACKET.json \
  --repo-root . \
  --candidate-worktree CANDIDATE_WORKTREE \
  --expected-sha FULL_IMPLEMENTATION_SHA \
  --require-ready
```

The checker is read-only. It validates recorded commands and evidence; it
never executes packet commands, accesses Unity, moves refs, or derives
scientific approval. Exit `0` means structurally valid and, with
`--require-ready`, mechanically ready for the packet's bounded target stage;
exit `1` means valid but not ready; exit `2` means invalid.

## Packet construction order

1. Freeze the candidate ref and every relevant dependency/audit ref at start
   and end. Record unavailable live refs as unavailable; never invent their
   values.
2. Bind the full candidate, parents, tree, authorized base, merge base,
   divergence, embedded version, standard binary-patch SHA-256, and exact
   name-status SHA-256.
3. Inventory every `base..candidate` commit in topological order. Separate
   application history from audit, failed/abandoned repair, generated
   evidence, coordination, and contaminating dependency history.
4. Inventory every changed path and material interface. Name architectural,
   scientific, lifecycle, evidence, and future combined-stage owners.
5. Bind the independent disposition to its exact commit/tree, report path and
   report digest. Carry every closed, accepted, conditioned, open, and blocked
   finding.
6. Declare either no intentional scientific change, with reviewed owner
   basis, or accepted change-ledger IDs plus predecessor/successor epoch and
   profile identities. A preparing epoch is not acceptance evidence.
7. Instantiate all required gate rows. Every gate has its own exact
   SHA/tree/base, timing, inputs, action, outputs, criteria, result, owners,
   evidence reference, metrics, and blocking stage.
8. Bind generated evidence to its originating candidate, source commit,
   command argv, input/output hashes, environment and retention location.
9. Record local evidence and human-mediated Unity evidence separately. All
   required Unity rows must identify one candidate SHA/tree and one dependency
   environment. Codex never retrieves Unity evidence.
10. Preserve external repository boundaries and derive readiness. Do not set a
    free-standing `ready` field.

The repository-standard patch digest is the SHA-256 of the exact bytes from
`git diff --binary BASE_SHA CANDIDATE_SHA` (without `--full-index`). The
name-status digest is the SHA-256 of the exact NUL-delimited bytes from
`git diff --name-status --no-renames -z BASE_SHA CANDIDATE_SHA --`. Record the
full SHAs used; a displayed short name is never an input identity.

## Canonical JSON skeleton

The following is structural notation, not a valid packet. Angle-bracketed
values must be replaced, optional arrays may be empty only where the schema
and checker allow it, and no unresolved placeholder may remain in the final
JSON.

```json
{
  "schema_version": "citlali-frozen-lane-handoff-packet-v1",
  "packet_identity": {
    "packet_id": "<unique packet ID>",
    "lane_id": "<lane ID>",
    "packet_kind": "<cal_lane | align_lane | combined>",
    "recorded_at": "<ISO-8601 timestamp with timezone>",
    "lifecycle_state": "<preparing | frozen | returned_for_repair | blocked | authority_decision_required>",
    "target_stage": "<lane_handoff | combined_acceptance>"
  },
  "implementation_candidate": {
    "source_ref": "refs/heads/<source ref>",
    "snapshot_started_at": "<timestamp>",
    "snapshot_finished_at": "<timestamp>",
    "start_tip_sha": "<40-hex>",
    "end_tip_sha": "<40-hex>",
    "commit_sha": "<40-hex>",
    "parent_shas": ["<40-hex>"],
    "tree_sha": "<40-hex>",
    "authorized_base_sha": "<40-hex>",
    "authorized_base_tree": "<40-hex>",
    "merge_base_sha": "<40-hex>",
    "ahead_count": 0,
    "behind_count": 0,
    "standard_binary_patch_sha256": "<64-hex>",
    "name_status_sha256": "<64-hex>",
    "embedded_version": "<version text>",
    "implementation_frozen": false,
    "worktree_clean": false
  },
  "packet_container": {
    "kind": "uncommitted_packet",
    "commit_sha": null,
    "tree_sha": null,
    "separate_from_implementation": true
  },
  "freeze_snapshot": {
    "refs": [
      {
        "name": "refs/heads/<candidate>",
        "availability": "available",
        "start_sha": "<40-hex>",
        "end_sha": "<40-hex>",
        "verify_local": true
      }
    ],
    "tip_moved": false
  },
  "authority": {
    "convergence_base_decision": "<owner decision>",
    "owner_decision_refs": ["<decision reference>"],
    "authority_paths": [
      {"path": "doc/ARCHITECTURE.md", "blob_sha": "<40-hex blob>"}
    ]
  },
  "repository_scope": {
    "citlali": "repairable_in_current_authorized_repository_scope",
    "tolproj": "repairable_only_in_separately_reviewed_repository_lane",
    "tolteca": "blocked_deferred_read_only",
    "compensation_elsewhere_allowed": false
  },
  "ancestry": {
    "application_history": [
      {
        "commit_sha": "<40-hex>",
        "parent_shas": ["<40-hex>"],
        "tree_sha": "<40-hex>",
        "purpose": "<bounded purpose>",
        "categories": ["application", "test"],
        "import_disposition": "include_application"
      }
    ],
    "excluded_history": [
      {
        "commit_sha": "<40-hex>",
        "tree_sha": "<40-hex>",
        "category": "audit",
        "reason": "<why it remains outside application ancestry>"
      }
    ],
    "source_dependencies": [
      {
        "dependency_id": "<ID>",
        "repository": "citlali",
        "commit_sha": "<40-hex or null>",
        "classification": "<base_present | lane_local | cross_lane | separately_promoted | external_blocked | contaminating>",
        "disposition": "<imported | reconstructed | independently_required | excluded | deferred>",
        "owner": "<owner>",
        "reason": "<evidence-backed reason>"
      }
    ]
  },
  "changed_scope": {
    "paths": [
      {
        "status": "<A | M | D | T>",
        "path": "<repository-relative path>",
        "blob_sha": "<40-hex or null for deletion>",
        "category": "<application/test/validation/etc.>",
        "owner": "<path owner>"
      }
    ],
    "interfaces": [
      {
        "interface": "<interface>",
        "path": "<path or null>",
        "architectural_owner": "<owner>",
        "scientific_owners": ["<owner>"],
        "lifecycle_owner": "<owner>",
        "classification": "<additive | textual_conflict | semantic_conflict | coupled>",
        "required_evidence": ["<criterion>"],
        "future_stage_owner": "<stage owner>"
      }
    ],
    "affected_modes": ["point"],
    "governed_change_kinds": ["structural"]
  },
  "independent_disposition": {
    "review_commit_sha": "<40-hex>",
    "review_tree_sha": "<40-hex>",
    "report_path": "<repository-relative path>",
    "report_sha256": "<64-hex>",
    "axes": {
      "scientific_contract": "<axis value>",
      "implementation": "<axis value>",
      "validation_readiness": "<axis value>",
      "historical_fixture": "<axis value>",
      "production": "<axis value>",
      "verdict": "<axis value>"
    },
    "findings": [
      {
        "finding_id": "<ID>",
        "status": "<closed | accepted | conditioned | open | blocked>",
        "owner": "<owner>",
        "blocking_stage": "<stage>",
        "rationale": "<rationale>",
        "evidence_ids": ["<evidence ID>"],
        "changed_at_candidate": false
      }
    ]
  },
  "scientific_change": {
    "state": "<none | declared>",
    "owner_basis": "<reviewed basis>",
    "ledger_path": "validation/intended_science_changes.json",
    "ledger_blob_sha": "<40-hex blob>",
    "change_ids": [],
    "predecessor_epoch_id": null,
    "successor_epoch_id": null,
    "successor_epoch_status": "none",
    "profile_ids": []
  },
  "gate_policy": {
    "required_gate_ids": ["<gate ID>"],
    "required_modes": ["point"],
    "unity_required": false,
    "unity_omission": {
      "authority": "<owner authority>",
      "reason": "<stage-bounded reason>"
    }
  },
  "gate_results": ["<gate rows using the schema below>"],
  "generated_evidence": [],
  "local_evidence": {
    "candidate_sha": "<40-hex>",
    "candidate_tree": "<40-hex>",
    "gate_ids": ["<local gate ID>"],
    "clean_after_gates": false,
    "evidence_references": ["<evidence reference>"]
  },
  "unity_evidence": {
    "required": false,
    "human_mediated_only": true,
    "codex_accessed_unity": false,
    "dependency_environment_sha256": null,
    "omission": {
      "authority": "<owner authority>",
      "reason": "<stage-bounded reason>"
    },
    "rows": []
  },
  "external_dependencies": [
    {
      "dependency_id": "TOLAPT-PACKAGE-MAPPING",
      "repository": "tolapt",
      "classification": "external_owner_dependency",
      "status": "open",
      "owner": "TolAPT owner",
      "boundary": "immutable matching run/package and measured/design endpoint mapping",
      "evidence_authority": "<exact audit authority>",
      "finding_ids": ["<finding ID>"],
      "exit_condition": "owner-reviewed exact package/mapping conformance",
      "blocking_stage": "production_end_to_end",
      "read_only": true,
      "compensation_elsewhere_allowed": false,
      "resolved_commit_sha": null,
      "resolved_tree_sha": null,
      "closure_evidence_sha256": null
    },
    {
      "dependency_id": "TOLPROJ-APT-TRANSPORT",
      "repository": "tolproj",
      "classification": "repairable_only_in_separately_reviewed_repository_lane",
      "status": "open",
      "owner": "TolProj owner",
      "boundary": "selected APT transport",
      "evidence_authority": "<exact audit authority>",
      "finding_ids": ["<finding ID>"],
      "exit_condition": "<owner-reviewed closure>",
      "blocking_stage": "production_end_to_end",
      "read_only": false,
      "compensation_elsewhere_allowed": false,
      "resolved_commit_sha": null,
      "resolved_tree_sha": null,
      "closure_evidence_sha256": null
    },
    {
      "dependency_id": "TOLTECA-APT-SELECTION-TRANSPORT",
      "repository": "tolteca",
      "classification": "blocked_deferred_at_tolteca",
      "status": "deferred",
      "owner": "TolTECA owner",
      "boundary": "lossless exactly-one APT selection and transport",
      "evidence_authority": "<exact audit authority>",
      "finding_ids": ["TV2-01", "TV2-02"],
      "exit_condition": "owner repair or owner-approved replacement contract",
      "blocking_stage": "production_end_to_end",
      "read_only": true,
      "compensation_elsewhere_allowed": false,
      "resolved_commit_sha": null,
      "resolved_tree_sha": null,
      "closure_evidence_sha256": null
    },
    {
      "dependency_id": "BEAMMAP-CONSUMER-BM-R1",
      "repository": "toltec_beammap",
      "classification": "external_owner_dependency",
      "status": "open",
      "owner": "toltec_beammap owner",
      "boundary": "declared producer detector-binding consumer",
      "evidence_authority": "<exact audit authority>",
      "finding_ids": ["BM-R1"],
      "exit_condition": "consumer uses the declared mapping and fails closed when absent or conflicting",
      "blocking_stage": "production_end_to_end",
      "read_only": true,
      "compensation_elsewhere_allowed": false,
      "resolved_commit_sha": null,
      "resolved_tree_sha": null,
      "closure_evidence_sha256": null
    }
  ],
  "claims": {
    "supported": ["schema_contract"],
    "conditioned": [],
    "prohibited": ["production end-to-end APT conformance while external blockers remain"],
    "cross_repository_apt_conformance": false,
    "production_end_to_end_apt_contract": false,
    "refactor_apt_generation_selected": false,
    "refactor_reductions_regenerated": false,
    "legacy_lineage_used_as_refactor_input": false,
    "legacy_selection_equivalence_required": false,
    "new_contract_sample_artifact_milestone_met": false,
    "real_end_to_end_apt_chain_conformance": false,
    "scientific_readiness": false,
    "production_readiness": false,
    "refactor_apt_library_validated": false
  },
  "attestations": {
    "application_history_separated": false,
    "zero_unexplained_required_output_failures": false,
    "zero_unexpected_error_logs": false,
    "no_skipped_required_comparisons": false,
    "requested_effective_observation_realized_checked": false,
    "product_inventory_checked": false,
    "scientific_conventions_checked": false,
    "same_sha_local": false,
    "same_sha_local_unity": false,
    "compensating_identity_or_admission_weakening": false
  },
  "approvals": [
    {
      "role": "lane_owner",
      "owner": "<owner>",
      "status": "pending",
      "candidate_sha": "<40-hex>",
      "candidate_tree": "<40-hex>",
      "recorded_at": "<timestamp>",
      "conditions": ["<condition>"]
    }
  ]
}
```

## Gate-row skeleton

Every gate row, including document reviews and human Unity procedures, uses
the same shape:

```json
{
  "gate_id": "<stable gate ID>",
  "gate_version": "v1",
  "domain": "<domain>",
  "scope": "<cal | align | overlap | combined | external>",
  "required": true,
  "timing": ["lane_freeze"],
  "blocking_stage": "lane_handoff",
  "candidate": {
    "sha": "<implementation SHA>",
    "tree": "<implementation tree>",
    "base_sha": "<authorized base>"
  },
  "inputs": [
    {
      "artifact_id": "<input ID>",
      "location_kind": "<repository_blob | local_artifact | human_supplied_external>",
      "path_or_uri": "<path or scoped evidence locator>",
      "source_commit_sha": "<40-hex or null>",
      "originating_candidate_sha": "<implementation SHA>",
      "sha256": "<64-hex>"
    }
  ],
  "action": {
    "kind": "<local_command | human_mediated_unity | document_review>",
    "command_argv": ["<argv element>"],
    "procedure": "<human or review procedure; empty for local command>"
  },
  "outputs": ["<artifact records>"],
  "criteria": ["<mechanically checkable criterion>"],
  "result": "<not_run | pass | fail | blocked | conditioned | not_applicable | omitted>",
  "omission": {"authority": "", "reason": ""},
  "owners": {
    "execution": "<owner>",
    "architectural": "<owner>",
    "scientific": ["<owner>"],
    "evidence": "<owner>"
  },
  "evidence_reference": "<reference; nonempty for pass>",
  "claim_constraints": [],
  "started_at": null,
  "finished_at": null,
  "metrics": {
    "exit_status": null,
    "unexpected_error_count": null,
    "unexplained_required_output_failure_count": null,
    "missing_required_output_count": null,
    "skipped_required_comparison_count": null
  },
  "interface_contract": {
    "applicable": false,
    "interface_id": "",
    "producer_repository": "",
    "consumer_repository": "",
    "producer_commit_sha": null,
    "producer_tree_sha": null,
    "consumer_commit_sha": null,
    "consumer_tree_sha": null,
    "owner_repositories": [],
    "producer_artifact_schema": "",
    "consumer_preflight": "",
    "stable_scoped_keys": [],
    "exact_artifact_sha256": null,
    "mapping_sha256": null,
    "counterexamples": [],
    "readiness_status": "not_applicable",
    "blocking_dependencies": [],
    "mode_routes": []
  },
  "apt_phase_contract": {
    "applicable": false,
    "phase_id": "",
    "readiness_status": "not_applicable",
    "software_revisions": [],
    "generation_id": "",
    "generation_root": "",
    "software_revision_set_sha256": null,
    "config_manifest_sha256": null,
    "raw_data_manifest_sha256": null,
    "cohort_manifest_sha256": null,
    "artifact_manifest_sha256": null,
    "component_manifest_sha256": null,
    "membership_sha256": null,
    "mapping_sha256": null,
    "transformation_sha256": null,
    "application_sha256": null,
    "quarantine_manifest_sha256": null,
    "rollback_manifest_sha256": null,
    "network_count": 0,
    "artifact_scope_count": 0,
    "complete_case_count": 0,
    "permutation_case_count": 0,
    "rejection_case_count": 0,
    "legacy_input_count": 0,
    "mixed_generation_count": 0,
    "selected_artifacts_all_contract_generated": false,
    "immutable_generation": false,
    "historical_evidence_only": false,
    "blocking_dependencies": []
  }
}
```

## Mandatory APT boundary rows

Every packet records all seven rows below, even when the packet targets only a
bounded lane handoff and the row is conditioned at a later stage. Each row's
`interface_contract` names the producer and consumer, owner repositories,
versioned artifact/schema, consumer preflight, stable scoped keys, exact
artifact and mapping digests, counterexamples, readiness, and blocking IDs.
Its four `mode_routes` state the actual producer and consumer for Pointing,
OOF, Science, and Beammap. A non-applicable route needs owner authority and a
reason. TolAPT is offline: `inline` is not an allowed role.

| Gate | Exact producer/consumer contract |
|---|---|
| `APT-A-RAW-KMP-CITLALI-AXIS-001` | Raw/KMP coordinates to Citlali Beammap's internal ordered application axis. Component, network and local-column identity are explicit; row position is not identity. |
| `APT-B-CITLALI-BEAMMAP-EXPORT-001` | Citlali Beammap to `toltec_beammap`: versioned component/artifact plus exact HDU locator and producer slot maps to `BEAMMAP.UID`, the full typed APT member and raw coordinate. `det_N` and `EXTNAME N` are never UID. |
| `APT-C-BEAMMAP-MATCHING-001` | `toltec_beammap` to the actual TolAPT or TolProj consumer: exact Beammap, fit-QC and source-APT artifacts; complete one-to-one slot-to-member update; declared partial policy. Overlap, basename, adjacency, last-wins and array-wide local-tone inference are forbidden. BM-R1 must fail closed if B's mapping is absent or conflicting. |
| `APT-D-TOLAPT-TOLPROJ-PACKAGE-001` | The actual offline TolAPT/TolProj direction for each production mode, using immutable run/package inputs, typed measured/design endpoint maps, hashes, membership/mapping digests, exclusions and ambiguity. Absence of a TolAPT step is explicit. Paths and source rows do not bind components. |
| `APT-E-TOLPROJ-TOLTECA-SELECTION-001` | TolProj publishes exactly one observation-qualified selected refactor product with expected package, component, artifact, membership and mapping identities. `_apt_library` is the curated selection layer, never the generation workspace. |
| `APT-F-TOLTECA-CITLALI-TRANSPORT-001` | TolTECA v2 transports the configured artifact byte- and identity-preservingly. The repository is read-only/deferred; `_fix_apt` UID rewriting, positional regeneration, fallback `_make_apt`, and last-wins multiple selection are production-prohibited until separately repaired and authorized. |
| `APT-G-CITLALI-ADMISSION-001` | Citlali recomputes exact artifact and semantic membership/mapping/application identities, is permutation invariant, and fails closed on missing, extra, duplicate, ambiguous, stale, forged, partial, or conflicting input. Presentation/source row remains provenance only. |

Overall cross-repository conformance is the conjunction of every applicable
A--G route. No row can be satisfied by a downstream heuristic repair. TolProj
repairs remain a separately reviewed owner-repository lane; TolTECA remains
`blocked_deferred_read_only`.

## Blocked new-contract sample milestone

`APT-SAMPLE-NEW-CONTRACT-FIXTURES-001` is a mandatory recorded, currently
blocked pre-production milestone. It does not require the full library
campaign. An authorized human will eventually generate the evidence on Unity;
this template grants no such authorization. A pass requires exact frozen
software/config/raw identities and at least:

- one Beammap product, its detector-binding manifest, and a candidate APT and
  fit-QC package;
- at least two networks and enough members to exercise scoped UID, producer
  slot, and tone mappings;
- at least two observations or other distinct artifact scopes to prove
  cross-artifact isolation;
- one complete valid case, a controlled permutation that changes no scientific
  value, and explicit rejection cases; and
- component, artifact, membership, mapping, transformation, and application
  digests joined through `toltec_beammap`, the actual TolAPT route (if any),
  TolProj, TolTECA pass-through, and Citlali admission.

Until that milestone passes, evidence is limited to `source_static`, `unit`,
`synthetic_counterexample`, `schema_contract`, and
`historical_cross_generation_comparison`. The packet must keep real chain,
scientific-readiness, production-readiness, and library-validation claims
false.

## Dedicated refactor APT generation

The following rows are mandatory records but later-stage blockers in a current
lane or combined-software packet. They do not authorize a reduction, shared
library write, or selected-generation change:

- `APT-LIB-SOFTWARE-FREEZE-001`: exact accepted software revision set.
- `APT-LIB-COHORT-MANIFEST-001`: owner-approved observation cohort and BEAM
  campaign manifest with exact config and raw-data identities.
- `APT-LIB-BEAM-CAMPAIGN-001`: human-run new-contract BEAM campaign.
- `APT-LIB-CANDIDATE-CONFORMANCE-001`: producer, consumer, and scientific
  disposition of every candidate.
- `APT-LIB-IMMUTABLE-GENERATION-001`: construct a new immutable/versioned
  generation under a dedicated `citlali-refactor` library root; never overwrite
  or silently replace an entry.
- `APT-LIB-COMPLETENESS-QUARANTINE-001`: account for accepted, rejected, and
  quarantined candidates with zero unexplained candidate.
- `APT-LIB-PROVENANCE-001`: bind artifact/component/membership/mapping/
  transformation/application digests.
- `APT-LIB-NO-MIXED-LINEAGE-001`: zero legacy inputs and zero mixed-generation
  selection or downstream products.
- `APT-LIB-HISTORICAL-IMMUTABILITY-001`: legacy libraries, manifests, and runs
  remain addressable, pinned, and immutable comparison evidence only.
- `APT-LIB-SHADOW-COMPARISON-001`: optional cross-generation scientific
  comparison. Legacy selection equivalence and backward-compatible admission
  are not promotion requirements.
- `APT-LIB-ACTIVATION-ROLLBACK-001`: owner-approved atomic selected-generation
  reference change with an exact rollback manifest; generation roots never
  change.
- `APT-LIB-SELECTED-CONTRACT-001`: every selected APT was generated and
  accepted under the frozen A--G contract.
- `APT-REFACTOR-REDUCTIONS-001`: regenerate Pointing, OOF where applicable,
  and Science from scratch using only the selected refactor generation and the
  exact frozen software/config/raw-data identities.

The relationship is conjunctive: new-contract BEAM products *and* derived
TolAPT/TolProj packages are required. Historical accepted runs are never
rewritten or promoted. Promotion later selects a fully validated refactor
generation by changing only its explicit selected-generation reference.

## Readiness and invalidation

The checker derives readiness from exact Git facts, ancestry, disposition,
gate results, findings, attestations and approvals. A packet can be valid while
blocked. A TolTECA blocker may coexist with a bounded lane handoff only when it
blocks `production_end_to_end` and both cross-repository and production
end-to-end APT claims remain false.

Any source, generated input, dependency, packet-bound evidence, candidate ref,
or implementation-tree change after freeze creates a new candidate and
invalidates every affected local and Unity row. Never combine passing rows
from different SHAs.
