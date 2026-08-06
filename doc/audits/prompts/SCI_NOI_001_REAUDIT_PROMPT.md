# SCI-NOI-001 narrow independent re-audit dispatch prompt — 2026-08-06

This is a narrow independent re-audit of the completed bounded repair. It does
not authorize repair, merge/cherry-pick, application integration, evidence
execution, Unity, local reduction, recipient dispatch, push, or production
change.

## Execution profile

- MODEL: `gpt-5.6-sol`
- EFFORT: `ultra`
- TASK SHAPE: `cross_path_determinism_and_provenance_reaudit`
- MISSION: Independently assess exact repair commit
  `38ef72860743636f59d226c9e1ff5ff776d0e9c0` against only approved SCI-NOI-001
  F001/F002/F003/F005/F008 and the held F007 evidence policy.
- ULTRA TRIGGER: RNG namespace, sequential/OpenMP determinism, Beammap
  lifecycle, compact provenance, configuration admission, and persisted-output
  joins cross multiple paths and require one independent synthesis. Ultra ends
  at the re-audit report/commit stop boundary.
- PARALLELISM: serial; no delegation or subagents.
- STOP RULE: Commit documentation-only re-audit artifacts and proposed ledger
  update, report exact disposition/commit/digests/clean state, and stop for
  coordinator review.

## Exact entry and frozen authority

1. Create a fresh isolated re-audit worktree from exact repair commit
   `38ef72860743636f59d226c9e1ff5ff776d0e9c0`, not from the repair worktree,
   prior audit worktree, or coordination branch. Create
   `codex/reaudit-sci-noi-001` only if absent and the new worktree is clean.
   Verify its parent is exact application
   `d5015fe716971bf8ea617e8a187311bf5af05185`; stop on mismatch.
2. Verify every authority entry in
   `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REAUDIT_AUTHORITY_MANIFEST_2026-08-06.yaml`,
   SHA-256 `fbe4cf5c4c783f6bc878ef1037aae7806386294124af1689dd071e04d9abb49e`.
3. Read `AGENTS.md`, the TolTEC context skill and routed repository
   authorities, `doc/audits/README.md`, `doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md`,
   and only the frozen manifest/authority Git objects before trace work.
4. Return a concise scope checkpoint before substantive inspection. It must
   identify worktree/branch/HEAD/parent/clean state, manifest hashes, included
   and excluded findings, focused gates, and the stop point. Do not expand
   scope without coordinator authorization.

## Included re-audit scope

Inspect the exact repair diff and affected implementation/tests/configuration/
provenance/output boundaries only for:

- F001 deterministic versioned realization namespace and distinct
  cross-observation assignments;
- F002 compact reproducibility provenance, completion/mode identity, digest
  joins, and required-write/error behavior without dense products;
- F003 truthful `source_imprinted_current` product/mode metadata only;
- F005 enabled-positive-count and disabled-only zero-work semantics, including
  Pointing/OOF quicklook; and
- F008 once-per-named-pass/iteration Beammap signs reused across active slots,
  invariant to active-map ordering/history.

Independently verify the reported local gate record, including focused
sequential/OpenMP/scheduling, namespace, Beammap, admission/no-work,
provenance, writer, CTest/config/header, and root rerun results, proportionally
and only through existing facilities. The disabled external-corpus test is
unrelated and is not a success substitute or a re-audit blocker.

F007 is held policy only: do not request or execute runtime, astronomical, or
Unity evidence. The exact-d501 evidence design remains a reference; any later
study requires an exact repaired SHA and its separately approved
FRAMEWORK-NUM-001 admission. F004 filter-edge work, F006 NOI-002
estimator/consumer/count policy, residual FRUIT work, and all RTC/PTC/MAP/JINC/
FLT/FRUIT algorithm changes are excluded.

## Deliverables and prohibitions

Create documentation-only re-audit report, focused finding dispositions,
proposed handoffs only if supported, and a proposed YAML ledger update in the
established package/proposal locations. Do not edit the canonical ledger or
handoff registry from the re-audit branch. Do not alter application code or
tests; rerun no local reduction; do not contact Unity, execute evidence, push,
integrate, launch another task, or authorize production. Production remains
`existing_use_only` unless a later coordinator/owner decision changes it.
