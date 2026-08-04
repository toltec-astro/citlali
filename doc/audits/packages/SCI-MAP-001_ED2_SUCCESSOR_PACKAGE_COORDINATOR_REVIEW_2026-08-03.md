# SCI-MAP-001 ED2 successor-package coordinator review — 2026-08-03

Status: locally verified successor package accepted for future human-run Unity
preparation. This is not Unity evidence, a repair acceptance, re-audit, or
production authorization.

## Verified identity

The reviewed task package is commit
`49e21ea90cd663370aa797f1295e8ee65ad4341c` on
`codex/map-unity-ed1`, with commit tree
`7804c6855371757bfcad5b00e133c8262bbdbd1d` and package tree
`d5fa2a2219cfc91b53621c896a5d34a734f3121f`.

Its active 42-file inventory has SHA-256
`ff33b659376c0535ae1b27bd02ecc9dba2c841bd7b6eaff5619f5fa87183cbf7`.
The worktree is clean and `git show --check` passes. The application candidate
remains `ed28dafb37f9113c0d3c95297148157129a90886`, tree
`cf75c36557178f351fb62781108a6f4b41b19225`; no application source changed.

## Review disposition

The successor correctly applies the owner-approved Unity-only capacity rule:

- the inclusive 200-GiB cap governs the five Unity campaign roots, not local
  package preparation;
- `144.244` GiB is retained only as digest-bound planning context, not as a
  serialization bound or capacity guarantee;
- no-submit pre- and post-stage records capture logical and allocated use,
  filesystem capacity, and planning context below the governed compact root;
- the owner must inspect each pre-record before separately invoking staging,
  CAP-POINT, CAP-SCIENCE, seven-case submission, analysis, or return work; and
- no automatic submission, continuation, cleanup, deletion, cache reuse, or
  CAP transfer is introduced.

The independent read-only review is included in the inventory and passes after
correction of its two operational findings. The package's complete verifier,
checksum checks, JSON/shell parsing, and 47 local tests pass.

## Boundaries and next step

The full/all-PTC route, frozen candidate, CAP identities, fixed seven-case
matrix, scientific gates, retention policy, and all repair/re-audit restrictions
remain unchanged. The package is now eligible for a later owner-operated Unity
campaign only after the owner explicitly elects to begin that external work and
the exact package commit is made available to the intended Unity checkout.
Until then, `SCI-MAP-001` remains nonconformant, its findings remain open, and
its production status remains existing-use-only.
