# SCI-NOI-002 Cycle 2 bounded repair prompt — 2026-08-06

Continue the existing SCI-NOI-002 repair task. This is a bounded successor
repair, not a new derivation, re-audit, application integration, Unity or local
astronomical reduction, evidence request, production change, or count/default
selection task.

## Execution profile

- MODEL: current resource-balanced repair profile; increase effort only if a
  genuinely ambiguous cross-lifecycle correctness issue is encountered.
- PARALLELISM: serial; no delegation or subagents.
- EXACT START: clean `codex/repair-sci-noi-002` at
  `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`.
- STOP RULE: commit one bounded Cycle 2 repair and proportional local tests,
  report the exact result, and stop before push, integration, evidence, or a
  fresh re-audit.

## Entry gate

1. Read `AGENTS.md`, the TolTEC context skill and routed authorities,
   `doc/REFACTOR_STATUS.md`, `doc/audits/README.md`, and
   `doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md`.
2. Verify the frozen Cycle 2 authority manifest and every named digest from the
   supplied Git objects. Do not assume coordination documents are present in
   the application worktree.
3. Verify that the worktree is clean, the branch is
   `codex/repair-sci-noi-002`, and HEAD is exactly
   `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`. Stop on mismatch.
4. Return a concise scope checkpoint before editing: exact worktree/HEAD/clean
   state, verified authorities, planned paths/tests, prohibited scope, and
   return gate. Wait for coordinator continuation.

## Authorized work

Implement only the six owner decisions in
`doc/audits/packages/SCI-NOI-002_CYCLE2_OWNER_DECISION_AND_REPAIR_HANDOFF_2026-08-06.md`:

- explicit run-scoped successfully published package membership, validated
  joins, and atomic final package-complete publication;
- active auditor support for disabled available-zero/no-work semantics while
  retaining enabled-zero rejection;
- separate plan-derived expected counts and inexpensive aggregate observed
  realization/publication completion counters at existing lifecycle boundaries;
- filtered stack-scatter validity based on actual calculated validity with
  distinct invalid reasons and identical alias metadata;
- removal of candidate-added duplicate canonical HDUs while retaining one
  legacy-named plane with canonical metadata identity; and
- bounded Mapdiag description/identity correction without duplicate variables
  or numerical changes.

Prefer direct changes to the existing first-repair paths and direct validators/
tests. A necessary direct reference or existing test may be added only after a
scope checkpoint. Do not add a general publication framework, helper subsystem,
new product class, dense covariance, sign stream, per-sample ID, or persistent
realization ledger.

## Frozen scientific boundary

Do not change the conditional finite-stack estimator, global nonprecision-scale
diagnostic, filtering mathematics, source-finder mathematics, MAP/JINC/RTC/PTC/
MODE algorithms, FRUIT behavior, configuration defaults, or realization counts.
F005 remains conditioned on SCI-FLT; F006 remains SCI-FRUIT-001-owned. Do not
claim physical-noise variance, inverse variance/precision, calibrated
significance, aperture/photometric uncertainty, or count adequacy.

## Required verification and return

Add exact fixtures for the repaired lifecycle, package, validation, alias, and
no-duplicate-plane contracts. Run the active baseline auditor against enabled
and disabled records, the required config preflight, focused tests, the full
CTest suite if the touched shared paths warrant it, and a proportionate
`citlali_cli` build. No local data or Unity reduction is authorized.

Commit only the bounded application/test repair on
`codex/repair-sci-noi-002`. Return exact parent and repair commit, changed paths,
commands/results, finding disposition, exclusions, and clean state. Stop before
push, integration, Unity, astronomical evidence, production action, or re-audit.
