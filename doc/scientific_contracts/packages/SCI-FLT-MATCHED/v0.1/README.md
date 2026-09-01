# SCI-FLT-MATCHED — Optimal Matched-Template Map Filtering

Status: owner-authorized Stage B r0.3 independent-review repair draft prepared
on `2026-09-01` for a second independent review; 14 owner questions remain
open, two are decided, and one is superseded

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
- Stage B r0.1: fresh implementation-blind author returned the two-view draft;
  its immutable review baseline is Git commit `bbbef9fe2`.
- Stage B r0.2: the scientific-owner closure directive was incorporated without
  implementation inspection. Collision-free notation, coordinate-basis
  algebra, local constrained GLS, anchor lattice, general-sky response,
  Learn--Resolve--Apply, realized/reference separation, option refactor,
  lifecycle, boundary, and route-status records are present.
- Independent ChatGPT Pro review of the exact r0.2 PDFs returned
  `major repair required` with no P0 findings. The owner authorized all directed
  P1--P3 repairs. r0.3 closes those findings without implementation input or a
  new scientific selection; see
  [`CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md`](CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md)
  and [`SEMANTIC_CHANGE_MAP_R0.3.md`](SEMANTIC_CHANGE_MAP_R0.3.md).
- Authored option identities: six families and 21 r0.1-stable alternatives in
  both views; no weighting, covariance, representation, named-use, or numerical
  route is selected.
- Draft inventory: 50 requirements, 24 predictions, 15 assumptions, a
  seven-part uncertainty budget, a 95-ID crosswalk, and 17 retained SODL IDs
  (14 open, 2 decided, 1 superseded).
- r0.3 build/source/PDF consistency and visual QA are the current closure gate;
  see [`build/BUILD_VERIFICATION.md`](build/BUILD_VERIFICATION.md) and
  [`PDF_QA_R0.3.md`](PDF_QA_R0.3.md).
- Scientific authority, numerical availability, implementation conformity,
  validation, achieved performance, readiness, and production status: not
  established.

## Package Contents And Current Records

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
- [`SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md`](SCIENTIFIC_OWNER_R0.2_DIRECTIVE_2026-08-31.md): hash-bound targeted closure direction
- [`CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md`](CHATGPT_PRO_INDEPENDENT_REVIEW_R0.2_2026-08-31.md): exact r0.2 independent-review record
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md): retained 17-ID disposition ledger
- [`SEMANTIC_CHANGE_MAP_R0.3.md`](SEMANTIC_CHANGE_MAP_R0.3.md): directed-review repair map
- [`NOTATION_CROSSWALK_R0.3.md`](NOTATION_CROSSWALK_R0.3.md): complete collision-free notation map
- [`REPRESENTATION_CROSSWALK_R0.3.md`](REPRESENTATION_CROSSWALK_R0.3.md): authority and representation closure
- [`ROUTE_STATUS_R0.3.md`](ROUTE_STATUS_R0.3.md): generic and realized route availability
- [`OWNER_DECISION_PARITY_R0.3.md`](OWNER_DECISION_PARITY_R0.3.md): repair and two-view parity
- [`CROSSWALK.md`](CROSSWALK.md): complete 95-ID source/view traceability
- [`STAGE_B_MANAGER_REVIEW_2026-08-31.md`](STAGE_B_MANAGER_REVIEW_2026-08-31.md): historical r0.1 manager review
- [`STAGE_B_DRAFT_MANIFEST.md`](STAGE_B_DRAFT_MANIFEST.md): exact draft source/PDF binding
- [`DECISION_LOG.md`](DECISION_LOG.md): package process decisions and preserved nonclaims

## Stop Boundary

The Stage B r0.3 draft is not frozen scientific authority. The anchor lattice,
exact-science/numerical-profile separation, and seven role semantics are owner
disposed, but no weighting or realized scientific route is selected. Each
dependent route remains unavailable until the remaining exact dispositions and
declarations exist and the resulting contract bytes are reviewed. No
implementation, conformity, response/covariance fidelity, observational
validation, performance, readiness, production, or freeze claim follows.
