# SCI-FLT-MATCHED — Optimal Matched-Template Map Filtering

Status: implementation-blind Stage B r0.1 draft returned and manager-reviewed
on `2026-08-31`; 17 scientific-owner questions remain open

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package follows the [Scientific Contract Library Program](../../../README.md),
the [pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
Package-specific recovery and dispositions are in [`PRIOR_WORK.md`](PRIOR_WORK.md).
Implementation-informed material is quarantined by [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md)
and is not author input.

The closed [`SCI-FLT-INF` holding study](../../../studies/SCI-FLT-INF_STAGE_A_2026-08-30/README.md)
resolved ODQ-001 through ODQ-013 and approved this package identity. This
package-local packet sanitizes those decisions without admitting the study's
implementation evidence, manager analysis, or historical mechanics.

## Scientific Boundary

`SCI-FLT-MATCHED` defines a map-domain optimal matched-template amplitude
estimator. It publishes a filtered map, not detected sources or a posterior
sky reconstruction. Ordinary convolution, genuine Wiener/posterior
reconstruction, source analysis, data-thresholded filtering, and FRUIT are
separate methods or future packages.

## Current Gate

- Prior-work recovery and owner decisions: complete.
- Sanitized Scope Brief and exact author packet: candidate prepared.
- Exact bytes and hashes: owner-approved through
  [`SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md).
- Stage B: fresh implementation-blind author returned the two-view r0.1 draft.
- Authored option sets: six families and 21 alternatives produced in both
  views; all remain unselected.
- Draft inventory: 39 requirements, 18 predictions, 15 assumptions, a
  seven-part uncertainty budget, a 78-ID crosswalk, and 17 open owner
  questions.
- Manager review and build/PDF consistency: passed; see
  [`STAGE_B_MANAGER_REVIEW_2026-08-31.md`](STAGE_B_MANAGER_REVIEW_2026-08-31.md).
- Scientific authority, numerical availability, implementation conformity,
  validation, achieved performance, readiness, and production status: not
  established.

## Stage A Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): package-local recovery and dispositions
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantine and recovery pointer
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): sanitized Stage B assignment
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): reusable and excluded material
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md): shared identities and ownership
- [`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md): content-bound owner direction
- [`AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_STATE_AND_PRODUCT_TAXONOMY.md): estimator, state, and products
- [`AUTHOR_BOUNDARIES.md`](AUTHOR_BOUNDARIES.md): MAP, NOI, VAL, CAL/BEAM, FRUIT, and exclusions
- [`REQUIRED_AUTHORED_OPTION_SETS.md`](REQUIRED_AUTHORED_OPTION_SETS.md): stable option assignments
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exclusive content-bound author inputs
- [`SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-31.md): exact-byte approval and launch authority
- [`verify_stage_a.py`](verify_stage_a.py): packet and firewall verification
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): 17 open Stage B dispositions
- [`CROSSWALK.md`](CROSSWALK.md): complete 78-ID source/view traceability
- [`STAGE_B_MANAGER_REVIEW_2026-08-31.md`](STAGE_B_MANAGER_REVIEW_2026-08-31.md): manager firewall, content, and PDF review
- [`STAGE_B_DRAFT_MANIFEST.md`](STAGE_B_DRAFT_MANIFEST.md): exact draft source/PDF binding
- [`DECISION_LOG.md`](DECISION_LOG.md): package process decisions and preserved nonclaims

## Stop Boundary

The Stage B draft is not scientific authority and no option is selected.
Scientific-owner disposition starts with `SCI-FLT-MATCHED-SODL-001`; each
dependent route remains unavailable until its exact option and parameters are
disposed and the resulting contract bytes are reviewed. No implementation,
conformity, validation, performance, readiness, production, or freeze claim
follows from this draft.
