# SCI-NOI — Noise Realizations And Empirical Uncertainty

Status: repaired Stage A owner-review candidate; Stage B not commissioned

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It starts from the frozen SCI-MAP v0.1/r0.7.1 and SCI-JINC v0.1/r0.3
authorities without modifying either package.

Work began with the package-specific [`PRIOR_WORK.md`](PRIOR_WORK.md) and the
quarantined [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md). Recovery located and
classified:

- the earlier implementation-independent `SCI-NOI-001` conditional-sign
  ensemble core;
- the earlier implementation-independent `SCI-NOI-002` finite-ensemble,
  covariance-calibration, and consumer-validity core;
- their later owner decisions, application-integration record, and held
  cross-package questions;
- the internal Citlali noise-estimation derivation;
- the current noise configuration, realization, observation/coadd,
  filtering, product, and persistence surfaces; and
- historical tests and validation strictly as implementation evidence, never
  as scientific authority.

The repaired proposed Stage B packet reuses the two independent cores under one
[`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md). The cover keeps
their mathematics while superseding their old two-package organization,
fixed-state-only premise, collision-prone bare family symbols, and any language
that could conflate realization
generation, uncertainty inference, empirical weights, standardized signal,
or statistical significance. Exact allowed inputs and the information
firewall are in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

This opening has not yet received scientific-owner approval. No Stage B
scientific author may be commissioned until Grant Wilson approves the exact
Scope Brief, proposed taxonomy, unresolved-question disposition, and
content-bound author packet. The final Stage A repair was performed only from
the supplied owner directive and existing scientific-authority/Stage A records;
it did not inspect implementation, configuration, schemas, tests, audits,
validation, reductions, Unity, defaults, or historical behavior.

## Current Gate

- Program, roadmap, and frozen-parent adherence: documented.
- Package-specific prior-work recovery: complete for Stage A owner review.
- Implementation-informed inventory: complete and quarantined.
- Ownership and typed-boundary classification: repaired and proposed.
- Collision-free operator/product taxonomy and exact DAGs: proposed; ODQ-101
  conditioning family approved, all numerical routes unavailable.
- Exact MAP, JINC, and conditional pre-MAP PTC boundaries: proposed; numerical
  availability remains fail-closed.
- NOI-owned GEN/UNC-member/UNC-ensemble/STD profile drafts: proposed; not
  registered; GEN completion ownership repaired.
- Sanitized owner-decision artifact: ODQ-101 approved; granular route and later
  decisions remain open.
- Sanitized Scope Brief and author packet: repaired and content-bound candidate;
  not owner-approved.
- Implementation-blind scientific rationale: not commissioned and not
  drafted.
- Engineering conformance specification: not commissioned and not drafted.
- Scientific authority: not frozen.
- Implementation conformity, representation fidelity, validation, achieved
  uncertainty/performance, readiness, and production state: not assessed.

## Scientific Boundary

SCI-NOI contains two primary scientific roles and one separately identified
derived operation:

1. `NOI-GEN`: generation of a declared realization ensemble;
2. `NOI-UNC`: empirical uncertainty inference from that exact ensemble; and
3. `NOI-STD`: construction of a standardized-signal companion from an immutable MAP or
   JINC parent and an authorized uncertainty product.

The boundaries are hard and typed. An ensemble does not become variance,
covariance, a calibrated weight, signal-to-noise, statistical significance,
or detection probability merely because it is randomized, sign-flipped,
jackknifed, or signal-suppressed. A standardized-signal product is not itself
an uncertainty estimate. Unless separately justified and validated, it means
only “standardized by the stated empirical scale.”

MAP and JINC parents remain immutable. NOI products are versioned companions
with exact parent, method, conditioning, domain, support, and availability
identity. An empirical NOI weight is not a PTC/MAP gridding coefficient and
cannot cross that boundary without explicit future authority.

The current frozen authorities do not supply a generally authorized numerical
MAP, pre-MAP-to-MAP, or JINC parent route. The exact boundary artifacts preserve
those typed unavailable states and do not create numerical products.

## Stage A Contents

- [`SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-08-29.md`](SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-08-29.md): durable owner launch direction
- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery, classification, digests,
  dispositions, and non-repetition synthesis
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation and
  evidence inventory
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): proposed scientist-readable Stage B input
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): proposed
  treatment of both recovered mathematical cores
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized parent, ownership, and typed-boundary extract
- [`AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md):
  collision-free roles, exact operator DAGs, ensemble/source-imprint,
  estimator/covariance, STD, and atomic lifecycle tables
- [`NOI_GEN_PARENT_OPERATOR_GRAPH.md`](NOI_GEN_PARENT_OPERATOR_GRAPH.md):
  conditioning classes and distinct route-specific complete method candidates
- [`ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md`](ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md):
  complete finite-design, canonical-key, rank, and source-imprint identity
- [`FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md`](FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md):
  exact UNC decision surface
- [`STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md`](STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md):
  exact numerator/scale compatibility and unit-`1` claim boundary
- [`PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md`](PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md):
  atomic roles, producer truth, admissions, and lifecycle
- [`SCI-MAP_TO_SCI-NOI_BOUNDARY.md`](SCI-MAP_TO_SCI-NOI_BOUNDARY.md): exact
  sanitized MAP parent boundary
- [`SCI-JINC_TO_SCI-NOI_BOUNDARY.md`](SCI-JINC_TO_SCI-NOI_BOUNDARY.md): exact
  sanitized JINC parent boundary and numerical-unavailability state
- [`SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md`](SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md):
  conditional exact pre-MAP GEN boundary
- [`SCI-NOI_VAL_PROFILE_DRAFTS.md`](SCI-NOI_VAL_PROFILE_DRAFTS.md): four
  NOI-owned use-specific VAL policy drafts
- [`FILTER_AND_FRUIT_SCOPE.md`](FILTER_AND_FRUIT_SCOPE.md): deterministic FLT,
  Wiener, and FRUIT inclusion/deferral record
- [`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md):
  one granular sanitized decision artifact with ODQ-101 approved
- [`SCIENTIFIC_OWNER_ODQ_101_APPROVAL_2026-08-29.md`](SCIENTIFIC_OWNER_ODQ_101_APPROVAL_2026-08-29.md):
  durable manager-facing owner approval and conflict check
- [`OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md`](OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md):
  complete Stage A ownership matrix
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact proposed
  allowed and prohibited inputs
- [`AUTHOR_PACKET_MANIFEST.sha256`](AUTHOR_PACKET_MANIFEST.sha256): external
  SHA-256 binding for the manifest bytes
- [`BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.md`](BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.md):
  exact closure-packet, source, and manifest verification
- [`BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.sha256`](BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.sha256):
  external binding for that report
- [`DECISION_LOG.md`](DECISION_LOG.md): launch decisions and applied recovery
  and final-repair dispositions
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  recovered decisions and bounded genuine owner questions
- [`CROSSWALK.md`](CROSSWALK.md): reserved Stage B traceability surface
- [`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md): exact final repair and
  preserved claim boundary
- `src/` and `pdf/`: canonical paths reserved without normative or rendered
  Stage B content

## Stop Boundary

Stop after presenting these final Stage A closure artifacts and continue the
owner-decision walkthrough at `SCI-NOI-ODQ-102A`; ODQ-101 is approved. Do not
draft the implementation-blind scientific rationale, shared normative core,
engineering conformance specification, or PDFs until the exact conditional
Stage B gate in the Scope Brief is satisfied. Do not implement or modify
SCI-NOI algorithms, and do not alter frozen MAP, JINC, or PTC contracts. Make no
implementation-conformity, validation, achieved-performance, readiness, or
production claim under this launch.
