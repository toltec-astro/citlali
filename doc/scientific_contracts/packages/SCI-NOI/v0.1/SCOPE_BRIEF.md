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

The scientifically consequential rerun/relearn stages, `Theta_b`, and resulting
state that may differ from the real-observation reduction are method/member
identity. This method-definition requirement does not require exhaustive
implementation provenance. Fixed and relearned methods answer different
questions and cannot share an ensemble. A later UNC calculation, filtering
choice, or selection cannot change an earlier GEN member.

The scientific owner approved fixed-state conditional-sign GEN as the ordinary
base-v0.1 **conditioning family** in `SCI-NOI-ODQ-101`. Relearned GEN remains a
separate method class and is numerically unavailable until its consequential
rerun/relearn graph and resulting changed-state identity are fully specified.
Fixed-state and relearned members shall never be
combined in one uncertainty estimate.

That conditioning family is not one complete numerical method. Each exact
parent/insertion/host combination has a separate method identity. ODQ-102A
selects `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` as the ordinary route:
NOI applies its assignment at the exact PTC-to-MAP numerical boundary, and MAP
may consume the modifier inline during otherwise ordinary frozen accumulation.
No materialized randomized timestream is required. NOI owns assignment,
ensemble design, realization identity, and the resulting NOI realization map;
MAP owns only conforming application of the modifier. The result is not an
ordinary MAP science product.

The complete graph in
[`NOI_GEN_PARENT_OPERATOR_GRAPH.md`](NOI_GEN_PARENT_OPERATOR_GRAPH.md)
keeps PTC-to-frozen-JINC, realized-MAP, realized-JINC, and filtered routes
separate and unselected. The selected ordinary route remains numerically
unavailable until its frozen PTC/MAP coefficient and exact numerical
`coverage_cut` state/value and failure-policy gates are realized.

## 3. Exact Parent Boundaries And Current Availability

The author packet includes three sanitized exact boundaries:

- [`SCI-MAP_TO_SCI-NOI_BOUNDARY.md`](SCI-MAP_TO_SCI-NOI_BOUNDARY.md), identity
  `SCI-MAP_TO_SCI-NOI v0.1/r0.2`;
- [`SCI-JINC_TO_SCI-NOI_BOUNDARY.md`](SCI-JINC_TO_SCI-NOI_BOUNDARY.md), identity
  `SCI-JINC_TO_SCI-NOI v0.1/r0.2`; and
- [`SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md`](SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md),
  identity `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.6`, containing separately
  identified PTC-to-frozen-MAP and PTC-to-frozen-JINC candidates.

Each boundary binds exact package/revision, product/application generation,
observation/detector/array/group identity, stable RTC `n` where applicable,
signal quantity/unit/beam/response/WCS/frame, coefficient/projection/
normalization/support/boundary/grouping, validity/causes/influence,
response/covariance/availability, operator state, source/filter state,
lifecycle, failure, and immutable provenance.

A boundary describes a route only when its exact parent is scientifically
realized. Selecting route identity never creates a numerical parent. Presently:

- the selected ordinary pre-MAP-to-frozen-MAP route remains unavailable because
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

For the ordinary route, ODQ-102B fixes the coherence partition: one unit is one
stable realized detector/channel within one exact observation. A member's
assignment applies to every admitted PTC occurrence of that detector throughout
the observation. Scan, subscan, chunk, sample/time, traversal, worker,
container, and accumulation order cannot change it. The same detector identity
in another observation is a different unit.

ODQ-102C selects a network-stratified, coefficient-balanced randomized sign
family. Within each stable readout network and exact observation, the design
balances detector coefficient masses
`B_d = sum_p sum_{i in C_p, detector(i)=d} G_pi gamma_i` derived from the exact
frozen MAP-admitted positive contributions. One network, array, or observation
cannot balance another. Admissibility and probability are complement-symmetric,
so each detector retains marginal sign probability `1/2`; detector signs are
not product-independent after balance conditioning, and equal detector counts
are not required. `B_d` is an NOI design coefficient, not precision, empirical
uncertainty weight, exposure, validity, or a replacement MAP-facing
coefficient. Exact tolerance, sampling/search, count, failure, member
dependence, pairing, duplicate/equivalence, and rank mechanics are delegated
under ODQ-102D to the implementation-blind contract author. The proposed
tolerance-conditioned construction is nonbinding guidance only, so numerical
design remains unavailable until exact Stage B mechanics are later accepted.

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

The randomization is intended to suppress source signal; it does not, by
construction alone, establish that the resulting maps are source-free. The
Stage-B scientific author may select clear terminology preserving that exact
meaning. `source_imprinted_conditional_randomization_ensemble` is a nonbinding
terminology suggestion, not a required name. The product establishes neither
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
requires common completed membership under its exact estimator. A future
separately authorized method may define an exact within-product/domain
missingness estimator with declared member population, symmetry, positive-
semidefinite, rank, and domain properties, but it cannot rescue an admitted-
member failure or create a survivor ensemble. Missing blocks are unknown, not
zero. A numerical inverse is not automatically precision. Realization count is not exposure or independent astronomical
sample count, and increasing it does not reduce parent-map noise as
`1/sqrt(B)`.

ODQ-105B approves the initial pointwise estimand on the exact domain where
every admitted realization supplies a valid finite value:

```text
V_hat_cond(p) = sum_b omega_b M_b(p)^2,
sum_b omega_b = 1.
```

The exact finite design supplies the nonnegative weights. The declared center
is zero, the finite ensemble mean is not subtracted, and no `B-1` correction
applies. The result is a conditional randomization second moment that retains
source imprint and structured residual content; it is not automatically
physical-noise variance or MAP covariance. Locations outside the common all-
member domain are unavailable for this initial method.

No full pixel covariance is universally required. The representation must be
adequate only for its exact claim and honest about rank, null space, domain,
regularization, and omissions.

ODQ-106 establishes the initial pointwise field above as the ordinary primary
uncertainty representation, not as covariance by virtue of its pointwise or
diagonal-like shape. Retained ensembles and separately identified fixed-
projection, stationary/kernel, block/spectral/sparse/low-rank or other
structured, and full covariance methods are permitted; no dense full
covariance is universally required. A retained ensemble is not automatically
an estimated covariance. Every covariance method declares exact estimator,
member population, domain/support/response, rank or rank limit, null or
unresolved modes, regularization/approximation, omissions, and uncertainty/
calibration state. Unreported covariance is unknown or unavailable rather than
zero or independence. No representation implies an inverse or precision.

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
failed. GEN owns completion truth. Rejected finite-design candidates are not
failures or members. Once an assignment is admitted, every requested member
must complete through the declared frozen operator. Failure of any admitted
member fails the ensemble closed for all UNC use; completed survivors cannot be
reinterpreted as a partial ensemble. Diagnostic cause reporting is required,
but exhaustive implementation provenance is not.

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
is the one sanitized decision candidate. It records approved ODQ-101 and
ODQ-102A/B/C and keeps
the following scientifically independent choices separate:

- exact ordinary numerical route, ownership, and inline/materialized
  representation rule;
- initial coherence and balance families plus exact finite assignment mechanics;
- source-imprint claim;
- fixed state and replay graph;
- initial UNC targets/estimators;
- covariance and rank policy;
- empirical inverse/weight products;
- STD numerator/scale/claim;
- persistence/reconstruction;
- deterministic FLT, Wiener, and FRUIT scope; and
- NOI-owned VAL identities and exact actions.

ODQ-101 and ODQ-102A/B/C are approved. ODQ-102D delegates exact finite-design
selection and rationale to the implementation-blind contract author; its
tolerance-conditioned construction is a nonbinding suggestion, not advance
approval. ODQ-103 approves the source-suppression intent and no-source-free-by-
construction boundary while delegating exact terminology to the Stage-B
scientific author. ODQ-105A approves fail-closed completion for every admitted
member. ODQ-105B approves the initial zero-centered conditional randomization
second moment. ODQ-106 approves the covariance representation and rank policy.
The next walkthrough question is `SCI-NOI-ODQ-107`. Every later
decision and the artifact's final hash still require explicit owner review.

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
