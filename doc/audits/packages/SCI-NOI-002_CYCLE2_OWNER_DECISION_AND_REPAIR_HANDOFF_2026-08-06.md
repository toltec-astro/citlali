# SCI-NOI-002 Cycle 2 owner decision and bounded repair handoff — 2026-08-06

Status: `owner_approved_cycle_2_repair_ready_for_frozen_dispatch`. This record
accepts the independent re-audit as the governing assessment of exact repair
candidate `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`. It authorizes a bounded
successor repair only; it does not integrate application code, request Unity or
astronomical evidence, change a configured realization count/default, launch a
fresh re-audit, or alter production status.

## Accepted re-audit disposition

The owner accepts the re-audit's `amend` verdict and all four P1 findings. The
owner also accepts both P2 findings with the following explicit product-policy
decision: do not duplicate a physical data plane merely to provide a canonical
`EXTNAME`. Preserve one physical plane under its compatible legacy `EXTNAME`
and carry canonical identity through `NOIPRID`, semantic version and digest,
validity/restriction metadata, and the package join. Registry and tests must
verify that identity without requiring or emitting a duplicate canonical HDU.

The bounded Mapdiag terminology repair is approved as engineering cleanup:
preserve compatible stored variable names where changing them would break
readers, but correct their descriptions and identity/restriction metadata. Do
not create duplicate canonical variables. This decision changes labels and
publication contracts, not the estimator mathematics.

## Cycle 2 repair decisions

1. **RA-B001 — realized publication membership and atomic completion.** Build
   the noise-package inventory from the explicit, run-scoped list of members
   successfully published by this reduction. Do not discover membership by a
   recursive extension scan. Normalize and validate relative paths; reject
   duplicate, outside-root, non-regular, or symlink members; verify existence,
   hashes, inventory digest, and FITS/ECSV/NetCDF joins. Publish the final
   package-complete sidecar/marker atomically so interruption cannot leave an
   older or partial package looking current and complete. Retain full SHA-256
   post-write verification for now. Do not create a general publication
   framework.
2. **RA-B002 — disabled available-zero validation.** The active baseline
   auditor and its fixtures must accept and require the approved disabled
   representation: requested count retained, effective count zero, generation
   not executed, every realized count available and zero,
   `outputs_completed=true`, and completion basis
   `effective_disabled_zero_work`. Enabled-zero remains invalid.
3. **RA-B003 — observed completion.** Keep plan-derived expected counts
   separate from inexpensive aggregate observed completion counters updated at
   existing realization/publication lifecycle boundaries. Final package truth
   must reflect observed completion; incomplete or partially published work
   must not appear complete. Do not add per-sample identifiers, a sign stream,
   or a large persistent realization ledger.
4. **RA-B004 — filtered scatter validity.** Derive validity from the actual
   calculated filtered stack scatter, response, and support, not merely from a
   positive requested/effective noise count. Unavailable values are NaN with
   distinct reasons for `R_lt_2`, scatter unavailable/nonfinite, response
   invalid, and support invalid. Canonical identity metadata and every legacy
   alias must report the same truthful state.
5. **RA-R001 — one physical plane.** Remove the candidate-added duplicate
   canonical FITS HDUs. Preserve one legacy-named physical plane and the
   canonical metadata identity described above. Update registry and exact
   product tests to reject duplicate physical planes.
6. **RA-R002 — Mapdiag terminology.** Correct descriptions and semantic
   metadata that currently imply physical variance, precision weight, or
   calibrated significance. Preserve compatible stored names when necessary;
   no duplicate variables or numerical algorithm change is authorized.

## Findings and boundaries

F001, F002, and F008 remain accepted as closed by the first repair and its
deterministic fixtures. F003, F004, and F007 remain open for this bounded Cycle
2 repair. F005 remains open and conditioned on SCI-FLT closure even after its
local validity correction. F006 remains wholly SCI-FRUIT-001-owned except for
the previously admitted interface boundary; Cycle 2 must not change FRUIT
mathematics, thresholds, defaults, iteration, add-back, stopping law, or choose
an uncertainty estimator.

No new scientific estimator, physical-noise claim, significance claim,
configured realization count, default, mapmaking/filter algorithm, Unity run,
or astronomical evidence is admitted. This is an engineering-correctness
repair of publication, validation, lifecycle truth, validity metadata, and
storage behavior on top of the first candidate.

## Exact continuation and return gate

Continue the existing isolated repair task on `codex/repair-sci-noi-002` from
exact candidate `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`, never from the
re-audit commit or coordination branch. The successor must be a child of that
candidate. Run proportional local fixtures, the active baseline auditor, the
required configuration preflight, focused/full CTest as warranted, and a
proportionate `citlali_cli` build. Commit the bounded repair and return its
exact parent/commit, changed paths, gate results, and clean state. Stop before
push, integration, Unity, astronomical evidence, production action, or fresh
re-audit.
