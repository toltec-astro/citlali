# SCI-NOI — Noise Realizations, Empirical Uncertainty, And Standardization Scope Brief

Status: repaired Stage A owner-review candidate; exact bytes not owner-approved

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-29`

Starting source identifier:
`codex/scientific-contract-library@5f206cf46bb2868aadb00f37dbbbc3944ac4ec8c`

Approved source identifier: unavailable until owner approval

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
Scientific authorship begins after frozen SCI-MAP v0.1/r0.7.1 and frozen
SCI-JINC v0.1/r0.3, but a numerical GEN method may insert at an earlier exact
immutable parent. Scientific sequencing is therefore not the same as one
serial numerical pipeline.

Stage A recovered the two implementation-independent NOI cores and classified
prior work once in [`PRIOR_WORK.md`](PRIOR_WORK.md). The proposed packet retains
their coherence-unit/sign-law, conditional-moment, source-imprint,
fixed-operator propagation, finite-design, centering, rank, covariance-domain,
projected-uncertainty, exact-regeneration, and use-specific-adequacy reasoning.
It supersedes their old two-package split, fixed-state-only premise, and every
conflation among generation, uncertainty, weights, standardized signal, and
significance.

The future implementation-blind author may inspect only exact content-bound
items in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). Current or
historical implementation, schemas, configuration, tests, audits, repairs,
validation, reductions, Unity, accepted runs, defaults, historical behavior,
and production status remain prohibited. No implementation-blind Stage B work
has been launched.

## 1. Scientific Purpose And Collision-Free Roles

SCI-NOI defines three separate scientific operations:

1. `NOI-GEN`: generation of a declared finite realization ensemble;
2. `NOI-UNC`: empirical uncertainty inference from one exact admitted
   ensemble; and
3. `NOI-STD`: a derived standardized-signal product joining one immutable
   signal parent to one authorized uncertainty scale.

No operation automatically realizes the next. Bare NOI family symbols `G`,
`U`, and `Z` are prohibited because they collide with MAP and PTC notation.
`Z_i^PTC` remains reserved for the transformed PTC sample. Scientific identity
uses semantic names everywhere.

## 2. GEN: Exact Parent, Insertion Point, And Operator Graph

Every GEN method declares:

- earliest immutable parent;
- exact insertion point;
- complete parent and operator DAG;
- which adjacent-package state is fixed, rerun, not applicable, or unavailable;
- assignment design and finite member population;
- source-imprint state;
- output support, unit, validity, member QC, lifecycle, and provenance; and
- persistence, sufficient-statistic, or exact reconstruction state.

Fixed-state generation has the graph

```text
realization_b = O_Theta0(R_b(parent)).
```

Relearned generation has

```text
Theta_b       = LearnResolve_b(R_b(parent))
realization_b = O_Theta_b(R_b(parent)).
```

The exact rerun stages, `Theta_b`, changed-state record, response, lifecycle,
and failure are member identity. Fixed and relearned methods answer different
questions and cannot share an ensemble. A later UNC calculation, filtering
choice, or selection cannot change an earlier GEN member.

The scientific owner approved fixed-state conditional-sign GEN as the ordinary
base-v0.1 **conditioning family** in `SCI-NOI-ODQ-101`. Relearned GEN remains a
separate method class and is numerically unavailable until a complete rerun
graph is owner-approved. Fixed-state and relearned members shall never be
combined in one uncertainty estimate.

That conditioning family is not one complete numerical method. Each exact
parent/insertion/host combination has a separate method identity. The complete
candidate graph in [`NOI_GEN_PARENT_OPERATOR_GRAPH.md`](NOI_GEN_PARENT_OPERATOR_GRAPH.md)
separates PTC-to-frozen-MAP, PTC-to-frozen-JINC, realized-MAP,
realized-JINC, and filtered routes. ODQ-101 selected none of them. Until
`SCI-NOI-ODQ-102A` is resolved, every numerical route remains unavailable.

## 3. Exact Parent Boundaries And Current Availability

The author packet includes three sanitized exact boundaries:

- [`SCI-MAP_TO_SCI-NOI_BOUNDARY.md`](SCI-MAP_TO_SCI-NOI_BOUNDARY.md), identity
  `SCI-MAP_TO_SCI-NOI v0.1/r0.1`;
- [`SCI-JINC_TO_SCI-NOI_BOUNDARY.md`](SCI-JINC_TO_SCI-NOI_BOUNDARY.md), identity
  `SCI-JINC_TO_SCI-NOI v0.1/r0.1`; and
- [`SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md`](SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md),
  identity `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.1`, containing separately
  identified PTC-to-frozen-MAP and PTC-to-frozen-JINC candidates.

Each boundary binds exact package/revision, product/application generation,
observation/detector/array/group identity, stable RTC `n` where applicable,
signal quantity/unit/beam/response/WCS/frame, coefficient/projection/
normalization/support/boundary/grouping, validity/causes/influence,
response/covariance/availability, operator state, source/filter state,
lifecycle, failure, and immutable provenance.

A boundary describes a route only when its exact parent is scientifically
realized. It never creates a numerical parent. Presently:

- ordinary numerical MAP and pre-MAP-to-MAP routes remain unavailable because
  no exact PTC MAP-facing coefficient family and no exact owner-admitted
  numerical `coverage_cut` state/value are bound; and
- the numerical JINC route remains unavailable because its required PTC
  coefficient, TolTEC array parameter, and applicable numerical-adequacy
  authority are unavailable.

Stage B may remain conditional on those states. It may not infer numerical
parents, hidden defaults, or implementation behavior.

## 4. Finite Assignment Design And Cardinality

Every GEN method binds

```text
S = {s_bg},  b = 1,...,B,  g in the declared coherence-unit set,
```

including marginal and joint assignment law; exact probabilities or
deterministic design weights; coherence partition and stable coherence-unit
ordering; balance, complement, replacement, duplicate, and cross-observation
rules; scheduling-independent key derivation; exact key fields and canonical
serialization; the `B_unique` equivalence relation; complement treatment for
every count and rank; exact duplicate detection; exact design-rank definition
and domain; and exact algorithm/version reconstruction identity. Worker count,
scheduling, traversal, and container order cannot change the design.

The following remain distinct:

- `B_requested`;
- `B_resolved`;
- `B_completed`;
- `B_unique`;
- `B_admitted_for_UNC`; and
- applicable design rank.

Complement-pair count and effective information are additional distinct
quantities. None may be substituted for another.

Enabled GEN requires a positive resolved design and a method-valid completed
design. Disabled GEN is explicit zero-member/no-work. A specific UNC method
owns its minimum positive count/rank. Balance, complements, and large count do
not prove independent physical draws.

## 5. Source Imprint And Cancellation

Every GEN method declares parent source content, exact cancellation target,
assumptions, finite balance residual, support/coefficient/projection/filter
variation, scan-synchronous residual, source-model use/error, known leakage,
and resulting claim limits. Global assignment balance is not pixelwise source
cancellation under varying operator support or membership.

The proposed ordinary label is
`source_imprinted_conditional_randomization_ensemble`. It establishes neither
repeated physical-noise sampling nor a calibrated/source-free null. A fixed
source-residual ensemble and a full relearned FRUIT procedure remain different
methods.

## 6. UNC: Target, Estimator, Rank, And Covariance Domain

Every UNC method defines or types unavailable:

- exact target law;
- admitted GEN method, parent graph, completed membership, design, and
  source-imprint state;
- known, empirical, fitted, or other exact center;
- second moment versus covariance;
- exact finite-design normalization/correction;
- missingness, duplicates, complements, dependence, and effective information;
- domain, WCS, support, response reference, units, and use-specific minimum
  count/rank;
- diagonal, retained-ensemble, stationary/kernel, structured, projected, full,
  or unavailable representation;
- rank, null space, unresolved modes, regularization, inverse domain and bias;
- external calibration, uncertainty of the estimate, and omitted terms; and
- exact conditional/empirical claim ceiling.

No universal `1/B` or `1/(B-1)` rule is authorized. A multivariate covariance
requires common completed membership or an exact missing-data estimator with
declared symmetry, positive-semidefinite, rank, and domain properties. Missing
blocks are unknown, not zero. A numerical inverse is not automatically
precision. Realization count is not exposure or independent astronomical
sample count, and increasing it does not reduce parent-map noise as
`1/sqrt(B)`.

No full pixel covariance is universally required. The representation must be
adequate only for its exact claim and honest about rank, null space, domain,
regularization, and omissions.

## 7. STD: Numerator, Positive Scale, And Claim

Every STD method binds:

- exact immutable MAP or JINC numerator product and estimand;
- exact UNC product and whether it supplies standard deviation, standard
  error, projected uncertainty, calibrated scale, or another quantity;
- every square root, projection, or calibration needed to produce the direct
  denominator;
- numerator/denominator units and exact estimator, response-reference, WCS,
  support, validity, and lifecycle compatibility;
- numerator/scale dependence;
- zero, negative, nonfinite, unavailable, incompatible, and outside-support
  behavior; and
- exact output support and claim class.

The direct denominator is an authorized positive scale in the numerator's
signal unit. Variance, covariance, inverse variance, precision, and consumer
weight are not direct denominators without an exact method-specific transform.
The proposed product identity is `empirical_scale_standardized_signal`, with
the ordinary claim “standardized by the stated empirical scale,” and

```text
unit(empirical_scale_standardized_signal) = 1.
```

It is dimensionless. Dimensionlessness does not strengthen the claim.

`sig2noise` is not a scientific identity. Studentized statistics, Gaussian
z-scores, N-sigma claims, detection probability, false-alarm rate,
completeness, purity, and catalog decisions require separate null, selection,
search, multiplicity, and validation authority.

## 8. Typed Roles And Atomic Products

The contract keeps separate:

1. producer-owned sample validity and causes;
2. PTC/MAP analysis or gridding coefficients;
3. GEN assignments;
4. GEN-owned member completion, availability, support, and QC;
5. empirical variance/covariance;
6. marginal inverse variance, precision, and consumer-effective weight; and
7. standardized signal.

Numerical resemblance, inverse-square units, filenames, or historical names
cannot create an identity join. An empirical NOI product is never a MAP-facing
PTC coefficient without separately authorized successor/feedback authority.

Each realization member is atomic. GEN publishes a complete method/plan,
design, member inventory, requested/resolved/completed/unique/UNC-admitted
counts, terminal states, source imprint, QC, persistence/reconstruction,
lifecycle, failure, and provenance—or types the required result unavailable or
failed. GEN owns completion truth. An ensemble with failed members is usable only if its method defines
the remaining completed design as valid.

UNC and STD likewise publish their complete exact role or a typed
unavailable/failed state. No partial or finite payload creates false realized
success, and no product automatically realizes the next operation.

## 9. NOI-Owned VAL Profiles

[`SCI-NOI_VAL_PROFILE_DRAFTS.md`](SCI-NOI_VAL_PROFILE_DRAFTS.md) proposes:

- `SCI-NOI:generation_input_admission@1`;
- `SCI-NOI:uncertainty_member_admission@1`;
- `SCI-NOI:uncertainty_ensemble_admission@1`; and
- `SCI-NOI:standardization_admission@1`.

NOI owns each policy. A later SCI-VAL Registry successor binds approved exact
bytes and VAL Core evaluates them. Each profile separately records named
request, applicability, eligibility, and realization fields; required facts,
restrictions, exceptions, source-imprint/response/uncertainty roles,
missing/conflict behavior, aggregation, lifecycle, and one exact consumer
action. A generic flag, finite payload, completed realization, or another-use
pass has no universal veto or rescue effect.

The member-admission policy consumes GEN completion, failure,
duplicate/equivalence, support, source-imprint, QC,
persistence/reconstruction, and lifecycle facts without redefining them. GEN
owns completion truth; NOI's named-use policy owns admission to the exact UNC
use; VAL binds and evaluates that policy but authors neither fact nor policy.

The drafts are not registered or evaluable. Registry binding is a Stage B
dispatch prerequisite after owner approval, not scientific approval by VAL.

## 10. FLT, Wiener, And FRUIT Boundary

The proposed base disposition is recorded exactly in
[`FILTER_AND_FRUIT_SCOPE.md`](FILTER_AND_FRUIT_SCOPE.md):

- a deterministic held-fixed FLT may be used only through a future exact
  content-bound FLT boundary with complete signal/realization parity;
- re-estimated Wiener filtering is unavailable without a complete inference/
  feedback contract; and
- fixed FRUIT residual and complete relearned FRUIT methods remain separate and
  unavailable until an exact FRUIT boundary exists.

If `UNC_k` selects or constructs a later filter, the successor graph is
`UNC_k -> FLT_(k+1) -> GEN_(k+1) -> UNC_(k+1)`. It does not mutate or validate
`UNC_k`, and an uncertainty input to a Wiener operator is not independent
evidence validating that operator in the same generation.

## 11. Persistence And Immutable Companions

Individual members may be persisted, generated transiently with exact
regeneration, or reduced through mathematically equivalent streaming sufficient
statistics. The plan records exact immutable parents, algorithm/version,
seed/key/configuration, completed membership, estimator state, and resulting
audit/reconstruction limitation. Dense signs and per-sample provenance are not
universally required.

MAP and JINC parents remain immutable. NOI products attach as new versioned
companions with exact parent and method identity. Absence of complete
covariance does not invalidate the parent or prohibit later analysis.

## 12. Scientific-Owner Decisions And Stage A Gate

[`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md)
is the one sanitized decision candidate. It records approved ODQ-101 and keeps
the following scientifically independent choices separate:

- exact ordinary numerical route;
- initial coherence family and finite assignment law;
- source-imprint claim;
- fixed state and replay graph;
- initial UNC targets/estimators;
- covariance and rank policy;
- empirical inverse/weight products;
- STD numerator/scale/claim;
- persistence/reconstruction;
- deterministic FLT, Wiener, and FRUIT scope; and
- NOI-owned VAL identities and exact actions.

ODQ-101 is approved. The next walkthrough question is
`SCI-NOI-ODQ-102A`: select one exact ordinary parent/insertion route, or keep
all numerical routes explicitly unavailable. Every later decision and the
artifact's final hash still require explicit owner review.

## 13. Non-Goals And Claim Boundary

Stage A does not:

- draft or freeze the scientific rationale, normative core, engineering
  conformance specification, requirements, predictions, or PDFs;
- inspect or change implementation, schemas, configuration, tests, audits,
  repairs, validation, reductions, or frozen adjacent authority;
- create unavailable MAP, JINC, PTC, FLT, or FRUIT numerical parents;
- select an implementation default or infer historical behavior;
- require dense covariance, dense assignment provenance, or persisted members;
- promote NOI products into PTC/MAP coefficients;
- infer physical-noise validity, calibrated null behavior, Gaussian
  significance, detection probability, or feedback validity; or
- claim implementation conformity, representation fidelity, empirical
  calibration, achieved performance, readiness, or production authorization.

Conditional Stage B authorship may begin only after Grant Wilson approves the
exact repaired Scope Brief and packet bytes; every decision is either approved
or explicitly open with its dependent method unavailable; boundaries are
content-bound; profile bytes are approved; required source/profile Registry
successors are exact; and the manifest firewall is verified. If the packet is
insufficient, the future author must return one precise scientific question
rather than inspect prohibited material or fill a gap from memory.
