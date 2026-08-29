# SCI-JINC — Signed-Coefficient JINC Observation Mapmaker

Status: complete exact-byte Stage A successor packet approved under
`SCI-JINC-STAGE-A-Q002`; Stage A closed; versioned SCI-VAL binding satisfied;
fresh implementation-blind Stage B dispatch authorized but not yet recorded

Version: `v0.1`

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

Genuinely new work is limited to reconciling the recovered JINC authority with
the now-frozen upstream quantity, coordinate, validity, response, and
covariance boundaries; resolving the explicitly listed owner questions; and
rendering the retained science in the library's shared two-view house form.
Implementation-derived material remains outside the implementation-blind
author channel.

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
All Stage B dispatch prerequisites are satisfied; no Stage B normative content
has yet been created in this manager increment.

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
  approved and incorporated; total numerical error must be negligible against
  the approximately `10^-3` relative instrument-fidelity scale. No prescribed
  summation/count formula, exact adequate tie/bin/cache choice, bitwise
  reproducibility or stronger precision is scientifically required.
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
- Stage B dispatch gate: open; fresh implementation-blind Ultra authorship is
  authorized from the exact approved packet without another owner question.
- Implementation-blind scientific rationale: not commissioned and not drafted.
- Engineering conformance specification: not commissioned and not drafted.
- Scientific authority: not frozen.
- Implementation conformity, representation fidelity, validation, achieved
  response/performance, readiness, and production state: not assessed.

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
  JINC-owned profile draft awaiting a versioned VAL registry binding
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
- [`CROSSWALK.md`](CROSSWALK.md): reserved Stage B traceability surface
- `src/` and `pdf/`: canonical package paths reserved without normative or
  rendered Stage B content

## Stop Boundary

The exact ODQ-101/102B/103/104/105/106/107/109/110 successor packet is approved
under Q002 without changing its bytes. Do not dispatch or draft the
implementation-blind scientific rationale, shared normative core, engineering
conformance specification, or PDFs in the registry-binding increment. The
binding is now recorded, so a fresh isolated Ultra Stage B author may be
dispatched from the exact approved manifest without another scientific-owner
question. Do not inspect an implementation candidate for conformity or make
any implementation-conformity, validation, achieved-performance, readiness,
or production claim under this delta.
