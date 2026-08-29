# SCI-NOI — Noise Realizations And Empirical Uncertainty

Status: Stage A owner-review candidate; Stage B not commissioned

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

The proposed Stage B packet reuses the two independent cores under one
[`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md). The cover keeps
their mathematics while superseding their old two-package organization,
fixed-state-only premise, and any language that could conflate realization
generation, uncertainty inference, empirical weights, standardized signal,
or statistical significance. Exact allowed inputs and the information
firewall are in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

This opening has not yet received scientific-owner approval. No Stage B
scientific author may be commissioned until Grant Wilson approves the exact
Scope Brief, proposed taxonomy, unresolved-question disposition, and
content-bound author packet.

## Current Gate

- Program, roadmap, and frozen-parent adherence: documented.
- Package-specific prior-work recovery: complete for Stage A owner review.
- Implementation-informed inventory: complete and quarantined.
- Ownership and typed-boundary classification: proposed.
- Initial operator/product taxonomy: proposed.
- Sanitized Scope Brief and author packet: proposed; not owner-approved.
- Implementation-blind scientific rationale: not commissioned and not
  drafted.
- Engineering conformance specification: not commissioned and not drafted.
- Scientific authority: not frozen.
- Implementation conformity, representation fidelity, validation, achieved
  uncertainty/performance, readiness, and production state: not assessed.

## Scientific Boundary

SCI-NOI contains two primary scientific families and one separately identified
derived operation:

1. generation of a declared realization ensemble;
2. empirical uncertainty inference from that exact ensemble; and
3. construction of a standardized-signal companion from an immutable MAP or
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
  proposed initial method and product taxonomy
- [`OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md`](OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md):
  complete Stage A ownership matrix
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact proposed
  allowed and prohibited inputs
- [`DECISION_LOG.md`](DECISION_LOG.md): launch decisions and applied recovery
  dispositions
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  recovered decisions and bounded genuine owner questions
- [`CROSSWALK.md`](CROSSWALK.md): reserved Stage B traceability surface
- `src/` and `pdf/`: canonical paths reserved without normative or rendered
  Stage B content

## Stop Boundary

Stop after presenting these Stage A artifacts for owner review and opening the
owner-decision walkthrough. Do not draft the implementation-blind scientific
rationale, shared normative core, engineering conformance specification, or
PDFs before explicit owner approval. Do not implement or modify SCI-NOI
algorithms, and do not alter frozen MAP, JINC, or PTC contracts. Make no
implementation-conformity, validation, achieved-performance, readiness, or
production claim under this launch.
