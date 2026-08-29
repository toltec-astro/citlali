# SCI-JINC — Signed-Coefficient JINC Observation Mapmaker

Status: **Scientific authority frozen; implementation conformity not yet
assessed under this contract.**

Version: `v0.1`

Frozen revision: `r0.3`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It begins from the exact scientific-contract library authority
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`
and preserves the
[frozen SCI-MAP v0.1/r0.7.1 predecessor boundary](../../SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md)
without editing or broadening it.

Work began with the package's reviewed
[`PRIOR_WORK.md`](PRIOR_WORK.md) record. That recovery:

- cites the frozen implementation-independent `SCI-MAP-002` JINC core rather
  than asking a new author to repeat its signed-estimator derivation;
- adopts, through a sanitized supersession cover, the eight approved JINC
  support, subpixel, conditioning, admission, mask, coverage, kernel, and
  provenance decisions;
- abstracts the later accepted destination-ownership boundary without
  importing its implementation, tests, audit verdict, or validation results;
- supersedes the core's radial-support and pixel-area-integration branches
  with the later owner-approved square-cache and point-phase conventions;
- admits the owner-supplied Schloerb LMT OTF/JINC method only through a
  page-exact excerpt and cover, using it for the generic analytic family while
  excluding its 3-mm/FCRAO values and simulations from TolTEC authority;
- admits the exact post-freeze PTC coefficient-registry predecessor only under
  a JINC-specific cover, adding explicit per-family named-consumer permission
  without altering frozen PTC r0.5 or MAP r0.7.1;
- defers implementation, audit, repair, re-audit, Unity, achieved-performance,
  integration, and production records to later separately authorized work;
  and
- excludes ordinary positive-coefficient SCI-MAP predicates and products from
  JINC by analogy.

The predecessor author inputs were approved at
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f`. The ODQ-101/102B/103/104/105/106/107/109/110 successor candidate
updates the scientist-readable
[`SCOPE_BRIEF.md`](SCOPE_BRIEF.md), the frozen independent core paired with
[`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md), the page-exact
Schloerb method excerpt paired with
[`AUTHOR_LMT_JINC_REFERENCE_COVER.md`](AUTHOR_LMT_JINC_REFERENCE_COVER.md),
the exact PTC registry predecessor paired with
[`AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md`](AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md),
and the exact sanitized decision, boundary, notation, geometry, fixed product
and ownership artifacts. The recovered response/covariance table is now a
deferred manager-side reference excluded from author inputs. The successor byte
identities and complete firewall are recorded in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

The completed Stage B work was limited to reconciling the recovered JINC
authority with the frozen upstream quantity, coordinate, validity, response,
and covariance boundaries; resolving the explicitly listed owner questions;
and rendering the retained science in the library's shared two-view house
form. Implementation-derived material remained outside the
implementation-blind author channel.

Grant Wilson approved the exact predecessor Stage A candidate at
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f` on `2026-08-28`; see
[`SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-28.md`](SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-28.md).
Grant Wilson then approved the complete exact successor packet represented by
the manifest at commit `88dcce8b0f7b1d78053b25831b39cf370afd47cc`
under `SCI-JINC-STAGE-A-Q002`; see
[`SCIENTIFIC_OWNER_STAGE_A_Q002_APPROVAL_2026-08-28.md`](SCIENTIFIC_OWNER_STAGE_A_Q002_APPROVAL_2026-08-28.md).
The approved author-input bytes remain unchanged. The separately versioned
SCI-VAL binding is recorded in
[`SCI_VAL_REGISTRY_BINDING_2026-08-28.md`](SCI_VAL_REGISTRY_BINDING_2026-08-28.md).
The implementation-blind Stage B authoring, bounded owner repairs, independent
two-view consistency checks, exact center-tie approval, freeze, mechanical
verification, and post-freeze horizontal audit are complete. The controlling
entry point is
[`FREEZE_AUTHORITY_MANIFEST_R0.3.md`](FREEZE_AUTHORITY_MANIFEST_R0.3.md).

## Current Gate

- Program, roadmap, and frozen-predecessor adherence: documented.
- Package-specific prior-work recovery: complete for Stage A owner review.
- Implementation-informed internal dossier: complete and quarantined.
- ODQ-101 coefficient ownership/permission/selection architecture: owner-
  approved and incorporated; no exact family is registered here.
- ODQ-102B parameter semantics/no-numerical-route disposition: owner-approved
  and incorporated; inherited TolTEC values remain quarantined evidence and
  parameter optimization is deferred.
- ODQ-103 AST/JINC association, ownership, admission and cause disposition:
  owner-approved and incorporated as `SCI-JINC:jinc_map_contribution@1`,
  AST-to-JINC r0.2 and PTC-to-JINC r0.3.
- ODQ-104 time-support disposition: owner-approved and incorporated;
  `jinc_coefficient_squared_time` is the sole base-v0.1 time-support product,
  while physical exposure is deferred until an identified scientific use.
- ODQ-105 observation/coadd disposition: owner-approved and incorporated;
  base v0.1 defines one complete observation bundle, permits incremental
  same-observation accumulation under one exact realization, and defines no
  cross-observation combination semantics.
- ODQ-106 per-array grouping/cardinality disposition: owner-approved and
  incorporated; an observation may produce zero through three independent
  bundles, absent arrays create no placeholders, and contributions never merge
  across array or destination identities.
- ODQ-107 fixed product-schema disposition: owner-approved and incorporated;
  every produced bundle contains only required `N_p`, `C_p`, `Q_p`, derived
  `m_p` with local support/validity and `jinc_coefficient_squared_time`.
  ODQ-108 response/covariance products and every other role are deferred.
- ODQ-109 scientific-conditioning/numerical-adequacy disposition: owner-
  approved and incorporated through the exact-discrete-oracle-only certificate
  boundary. The upper-bin phase lattice and positive-axis half-pixel center
  tie are separately bound by `SCI-JINC-DEC-PHASE-CENTER-001`; no numerical-
  adequacy profile or matching certificate is supplied by this package.
- ODQ-110 finite-map center-admission disposition: owner-approved and
  incorporated; an occurrence contributes only when its resolved rounded
  cache center is in the finite destination domain. An outside center
  contributes nowhere even if its square overlaps the map; ordinary in-map
  edge crop remains and JINC-then-crop equivalence is not required.
- Numbered scientific-scope ODQs: complete.
- Exact-byte Stage A owner gate: closed by `SCI-JINC-STAGE-A-Q002` for commit
  `88dcce8b0f7b1d78053b25831b39cf370afd47cc` and manifest SHA-256
  `52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`.
- Sanitized Scope Brief: approved exact author input.
- Exact author-input manifest and firewall: approved; all sixteen object
  digests verified.
- SCI-VAL source/profile registry binding: satisfied by immutable successors
  `SCI-VAL_SOURCE_BINDING_REGISTER
  v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28` and
  `SCI-VAL_PROFILE_REGISTRY
  v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28`.
- Stage B implementation-blind authorship and bounded owner review: complete.
- Shared canonical authority: frozen at `SCI-JINC v0.1/r0.3`, with 44 stable
  requirements and 36 stable predictions shared by both views.
- Scientific rationale and engineering conformance specification: complete,
  compiled, mechanically checked, and visually inspected.
- Scientific authority: frozen at commit
  `a9f43877e01a661db13bd85b2e7f34ea5ac82fb7` and tag
  `sci-jinc-v0.1-r0.3`; later scientific correction requires a versioned
  successor and shall not change the tagged bytes.
- Post-freeze horizontal authority audit: complete with no material finding;
  no successor opened.
- Numerical TolTEC JINC route: typed unavailable pending the separately owned
  JINC-permitted PTC coefficient family, TolTEC array parameter set, and,
  where numerical support is claimed, exact adequacy profile and matching
  certificate.
- Implementation conformity, representation fidelity, validation, achieved
  response/performance, readiness, and production state: not assessed.

## Frozen Stage B Authority And Evidence

- [`FREEZE_AUTHORITY_MANIFEST_R0.3.md`](FREEZE_AUTHORITY_MANIFEST_R0.3.md):
  single controlling manifest for the complete frozen authority set
- [`AUTHOR_DRAFT_DECISION_RECORD.md`](AUTHOR_DRAFT_DECISION_RECORD.md):
  implementation-blind author and owner-review disposition record
- [`PHASE_LATTICE_OWNER_DISPOSITION_R0.3.md`](PHASE_LATTICE_OWNER_DISPOSITION_R0.3.md):
  stable `SCI-JINC-DEC-PHASE-CENTER-001` decision binding
- [`CROSSWALK.md`](CROSSWALK.md): 44-row frozen requirement traceability
- [`src/scientific-rationale.tex`](src/scientific-rationale.tex) and
  [`src/engineering-conformance.tex`](src/engineering-conformance.tex): two
  views importing one six-module shared authority
- [`pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf`](pdf/SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf)
  and
  [`pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf`](pdf/SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf):
  canonical frozen renderings
- [`FREEZE_VERIFICATION_R0.3.md`](FREEZE_VERIFICATION_R0.3.md): exact hashes,
  identifier/reference checks, PDF metadata, render, and visual verification
- [`HORIZONTAL_AUTHORITY_COHERENCE_AUDIT_R0.3.md`](HORIZONTAL_AUTHORITY_COHERENCE_AUDIT_R0.3.md):
  post-freeze read-only audit with no material finding

## Scientific Boundary

SCI-JINC owns a distinct signed-coefficient observation-map estimator. PTC
owns the positive analysis/gridding coefficient registry and each family's
named-consumer permission. SCI-JINC authority includes its signed spatial
coefficient, signed deposition, normalization, support, unit-invariant
conditioning, formal-support validity, coefficient-squared temporal
accounting, fixed five-role bundle and atomic whole-bundle publication.
Recovered response/covariance mathematics is deferred future science, not a
base-v0.1 product. No general product-availability or provenance framework is
authorized.

It does not inherit ordinary SCI-MAP's positive-coefficient contribution
predicate, one-hot placement, F010 product bundle, coaddition rule, support
aliases, or validity policy. It consumes upstream quantities only through an
explicit JINC-specific boundary and does not redefine CAL, RTC, PTC, AST, or
VAL facts. NOI, FLT, BEAM, MODE, SRC, and FRUIT remain downstream or adjacent
authorities.

## Stage A Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery, classification, digests,
  dispositions, and non-repetition synthesis
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation-
  informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): successor scientist-readable Stage B
  input candidate
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): successor
  owner-decision cover for the recovered core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  successor sanitized upstream/downstream boundary extract
- [`AUTHOR_DECISIONS_AND_OWNERSHIP.md`](AUTHOR_DECISIONS_AND_OWNERSHIP.md):
  exact sanitized table of eight inherited owner decisions
- [`AUTHOR_LMT_JINC_REFERENCE_COVER.md`](AUTHOR_LMT_JINC_REFERENCE_COVER.md)
  and [`ANALYTIC_JINC_IDENTITY.md`](ANALYTIC_JINC_IDENTITY.md): generic
  Schloerb formula, notation reconciliation, SCI-JINC supersessions, parameter
  semantics, and typed TolTEC numerical unavailability
- [`AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md`](AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md):
  controlled author use of the exact post-freeze PTC registry predecessor
- [`SCIENTIFIC_OWNER_ODQ_101_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_101_DECISION_2026-08-28.md):
  owner-approved registry, permission, selection and no-fallback disposition
- [`SCIENTIFIC_OWNER_ODQ_102B_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_102B_DECISION_2026-08-28.md):
  owner-approved generic-scale, parameter-semantics, evidence-only baseline,
  no-hidden-default and deferred-optimization disposition
- [`SCIENTIFIC_OWNER_ODQ_103_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_103_DECISION_2026-08-28.md):
  owner-approved AST/sample association, JINC map-contribution admission,
  local geometry/coefficient ownership and cause disposition
- [`SCIENTIFIC_OWNER_ODQ_104_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_104_DECISION_2026-08-28.md):
  owner-approved sole base-v0.1 time-support product and deferred physical-
  exposure disposition
- [`SCIENTIFIC_OWNER_ODQ_105_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_105_DECISION_2026-08-28.md):
  owner-approved observation-only base, same-observation incremental
  accumulation and future complete-bundle coadd-boundary disposition
- [`SCIENTIFIC_OWNER_ODQ_106_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_106_DECISION_2026-08-28.md):
  owner-approved per-array bundle identity, zero-through-three cardinality,
  absent-array and no-cross-destination-merge disposition
- [`SCIENTIFIC_OWNER_ODQ_107_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_107_DECISION_2026-08-28.md):
  owner-approved fixed five-role schema, whole-product fail-closed rule and
  deferral of ODQ-108 response/covariance and every other companion role
- [`SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md):
  owner-approved scientific-conditioning and instrument-relevant numerical-
  adequacy disposition
- [`SCIENTIFIC_OWNER_ODQ_110_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_110_DECISION_2026-08-28.md):
  owner-approved finite-map center-admission and no-overlap-fallback
  disposition
- [`SCI-PTC_TO_SCI-JINC_BOUNDARY.md`](SCI-PTC_TO_SCI-JINC_BOUNDARY.md) and
  [`SCI-AST_TO_SCI-JINC_BOUNDARY.md`](SCI-AST_TO_SCI-JINC_BOUNDARY.md): exact
  proposed upstream quantity and coordinate boundaries
- [`SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md`](SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md):
  exact JINC-owned upstream-occurrence profile bound by the immutable SCI-VAL
  successor records
- [`NOTATION_AND_UNITS.md`](NOTATION_AND_UNITS.md),
  [`GEOMETRY_DECISION_TABLE.md`](GEOMETRY_DECISION_TABLE.md),
  and [`GROUPING_AND_PRODUCT_ROLES.md`](GROUPING_AND_PRODUCT_ROLES.md): exact
  sanitized author tables and resolved numerical/finite-map policies
- [`RESPONSE_AND_COVARIANCE_FAMILIES.md`](RESPONSE_AND_COVARIANCE_FAMILIES.md):
  recovered manager-side scientific reference deferred by ODQ-107 and excluded
  from the base-v0.1 author packet
- [`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md): owner-feedback repair map
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exact proposed
  successor allowed and prohibited inputs
- [`DECISION_LOG.md`](DECISION_LOG.md): applied Stage A decisions and approval
  gate
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  recovered decisions, completed numbered scope ODQs and remaining approval
  gate
- [`SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-28.md`](SCIENTIFIC_OWNER_STAGE_A_APPROVAL_2026-08-28.md):
  exact candidate binding, approved scope, and preserved dispatch blockers
- [`SCIENTIFIC_OWNER_STAGE_A_Q002_APPROVAL_2026-08-28.md`](SCIENTIFIC_OWNER_STAGE_A_Q002_APPROVAL_2026-08-28.md):
  exact successor-packet approval, firewall interpretation, and bounded
  SCI-VAL registry-binding assessment
- [`SCI_VAL_REGISTRY_BINDING_2026-08-28.md`](SCI_VAL_REGISTRY_BINDING_2026-08-28.md):
  immutable successor identities, hashes, firewall treatment, and Stage B
  dispatch consequence
- [`CROSSWALK.md`](CROSSWALK.md): final 44-row frozen requirement traceability
- `src/` and `pdf/`: frozen shared authority, both source views, and canonical
  rendered Stage B PDFs

## Stop Boundary

SCI-JINC v0.1/r0.3 is complete and at rest as conditional,
implementation-independent scientific authority. Do not modify the bytes
bound by tag `sci-jinc-v0.1-r0.3`; a later scientific correction requires an
explicitly authorized versioned successor.

Any implementation-conformity audit, parameter/family authorization,
numerical-adequacy exercise, validation program, readiness decision, or
production decision is separate future work and requires its own authority.
Until then, do not inspect an implementation candidate under this package or
claim implementation conformity, numerical-route availability, validation,
achieved performance, readiness, or production authorization.
