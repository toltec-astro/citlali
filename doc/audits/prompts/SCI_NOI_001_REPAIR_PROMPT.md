# SCI-NOI-001 bounded repair dispatch prompt — 2026-08-06

This is a bounded implementation-repair task, not a re-audit, application
integration, evidence request, numerical reduction, Unity task, production
change, or recipient-audit dispatch.

## Execution profile

- MODEL: `gpt-5.6-sol`
- EFFORT: `ultra`
- TASK SHAPE: `bounded_cross_path_repair_and_determinism_synthesis`
- MISSION: Implement only the approved SCI-NOI-001 realization-identity,
  compact-provenance, truthful-current-mode, zero-stack-admission, and
  Beammap-pass identity repairs at exact application base `d5015fe716971bf8ea617e8a187311bf5af05185`.
- ULTRA TRIGGER: RNG identity, sequential/OpenMP scheduling determinism,
  Beammap lifecycle, and compact provenance cross distinct paths and require
  one reconciled deterministic design; Ultra ends after the repair/test commit
  and coordinator return.
- PARALLELISM: serial; no delegation or subagents.
- STOP RULE: Commit only the bounded repair and focused tests, report exact
  commit/digests/gates/clean state, and stop for coordinator review.

## Frozen authority and entry gate

1. The coordinator has prepared this record but has **not** launched the task.
   On launch, create a fresh isolated repair worktree from detached or branch
   state exactly `d5015fe716971bf8ea617e8a187311bf5af05185`
   (`origin/codex/refactor-mainline`), never from an audit or coordination
   branch. Create `codex/repair-sci-noi-001` only if it is absent and the new
   worktree is clean. Stop on any mismatch.
2. Verify every authority entry and digest in
   `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml`,
   SHA-256 `44c6ac459f058ba39e18d01f35e64bfb37bcfec9c116a12a2839f64beac71bae`.
3. Read `AGENTS.md`, the TolTEC context skill and routed repository
   authorities, `doc/REFACTOR_STATUS.md`, `doc/audits/README.md`,
   `doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md`, the frozen authority manifest,
   and only the named SCI-NOI-001 authority Git objects before code inspection.
4. Return a concise scope checkpoint before the first code or test edit. It
   must give worktree path, starting HEAD, branch state, clean state, verified
   authority digests, allowed findings, prohibited scope, first focused tests,
   and planned return point. Do not proceed past that checkpoint without
   explicit coordinator continuation.

## Included repair contract

Implement only these open findings and approved owner policies:

- **F001 / D001:** a compact, versioned deterministic realization key with
  observation identity, ensemble mode, conditioning iteration/pass,
  realization number, and coherence-unit identity; stable observation-scoped
  realized channel identity; independent assignments for distinct
  observations; exact sequential/OpenMP/scheduling repeatability.
- **F002 / D001:** compact reconstruction provenance containing key
  policy/version, stable ordering/partition, completed realization IDs,
  ensemble mode, and digest joins. Do not persist dense sign vectors,
  per-sample IDs, or N-by-N sign-correlation matrices.
- **F003 / D003:** truthful `source_imprinted_current` identity and metadata
  for the ordinary randomization of cleaned `x=s+n`. Individual realizations
  and moments may retain deterministic signal, including negative-source
  realizations. This is restricted diagnostic metadata only; it does not
  validate an estimator or product.
- **F005 / D002:** enabled realizations require requested/effective/realized
  count at least one. Disabled is the sole zero-stack state: zero effective and
  realized counts, no promised realization products/weights/diagnostics, and
  no realization-generation or downstream noise-product work. Pointing and
  OOF quicklook must be able to use that disabled minimal-computation lane.
- **F008 / D001:** generate Beammap signs once per named mapmaking
  pass/iteration and reuse them across active map slots; active-map order or
  history must not alter the assignment.

Use a versioned deterministic design and explain why its RNG/seed namespace
and ordering satisfy the contract. Do not introduce an arbitrary unversioned
seed choice. Preserve ordinary naive-map operator behavior except the
intentional sign-identity changes. Any JINC-path conclusion remains explicitly
conditioned on SCI-MAP-002.

## Explicit exclusions

Do not resolve F004 filter-edge preprocessing, F006 NOI-002
finite-N/variance/weight/S/N/significance/threshold/feedback/aperture/count
work, or F007 astronomical evidence. Do not implement a
`final_pre_readdition_residual` mode or alter the FRUIT loop. Do not alter RTC,
PTC, MAP, JINC, FLT, or FRUIT algorithms; choose a production realization
count/default; request or run evidence; use Unity; run a reduction; push;
re-audit; integrate; or change production status.

## Required local repair gates

Begin with focused contract tests and retain/add only proportional test support.
Before the repair return, run the applicable existing gates:

1. deterministic repeatability across sequential, OpenMP, and scheduling
   arrangements;
2. distinct-observation namespace/collision tests;
3. Beammap named-pass and active-map ordering/history invariance tests;
4. enabled-positive-count rejection/admission and disabled zero-work tests,
   including Pointing and OOF quicklook;
5. metadata/provenance reconstruction and completion/digest-join round trips;
6. required-write failure/error propagation tests;
7. focused CTest, required config preflight, and `citlali_cli` build where
   proportionate to touched paths.

The repair must preserve reproducibility. Do not make a full local astronomical
reduction or add a helper, schema, verifier, framework, campaign, or dense
provenance product. Record every test omitted as `not_applicable` with a
scope-based reason.

## Required return

Commit application and focused-test changes only on `codex/repair-sci-noi-001`.
Report the exact repair commit and parent, all changed paths, test commands and
results, deterministic design/version and provenance fields, any JINC
conditioning, current open findings/exclusions, clean state, and a concise
repair handoff for coordinator review. Stop before re-audit, evidence, push,
integration, or production authorization.
