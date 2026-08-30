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
  `SCI-MAP_TO_SCI-NOI v0.1/r0.3`;
- [`SCI-JINC_TO_SCI-NOI_BOUNDARY.md`](SCI-JINC_TO_SCI-NOI_BOUNDARY.md), identity
  `SCI-JINC_TO_SCI-NOI v0.1/r0.3`; and
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

ODQ-107 authorizes the exact finite-positive reciprocal

```text
D_inv = {p in D_common : V_hat_cond(p) is finite and strictly positive},
W_hat_cond(p) = 1 / V_hat_cond(p),
```

as `NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE`. The result has inverse
squared signal units and is not inverse variance or precision. Zero, negative,
nonfinite, unavailable, and outside-parent-domain input yields unavailable, not
a numerical zero. Any floor, cap, clipping, epsilon, shrinkage, or other
regularization is a separately identified method. Marginal inverse variance
requires separately authorized marginal variance; precision requires an
authorized covariance inverse/generalized inverse on an exact domain/subspace;
and consumer-effective weight is use-specific. None is validity, support,
exposure, a PTC/MAP coefficient, or a parent-mutation instruction. Future
cross-boundary use requires explicit scientific authority.

ODQ-108 selects `NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`. The exact
immutable normalized real-observation MAP signal associated with the same
frozen MAP operator state is divided by canonical `sqrt(V_hat_cond)` on their
exact compatible finite-positive valid-domain intersection:

```text
sigma_cond(p) = sqrt(V_hat_cond(p)),
S_cond(p) = q_MAP(p) / sigma_cond(p).
```

The output `empirical_scale_standardized_signal` has unit `1` and means only
“MAP signal standardized by the stated conditional randomization second-moment
scale.” Exact estimator/generation, parent, response, unit/beam, WCS, support,
validity, lifecycle, UNC generation, transformation, and numerator/scale
dependence are bound. No interpolation, domain extension, substitution, or
implicit alternate route applies. Invalid or incompatible scale yields
unavailable rather than zero or infinity. Algebraic `1/sqrt(W_hat_cond)` is
not a second implicit method. No Gaussian/Student/z/N-sigma significance,
probability, detection, completeness, purity, or catalog claim follows. JINC
standardization remains a separate future method with a JINC-specific scale.

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

The ODQ-110A/B-approved ownership/parity and Wiener rules and the still-open ODQ-110C
dispositions are recorded exactly in
[`FILTER_AND_FRUIT_SCOPE.md`](FILTER_AND_FRUIT_SCOPE.md):

- NOI does not choose or define a deterministic transformation; the appropriate
  upstream/downstream scientific process owns it, and NOI must apply exactly
  that transformation to every admitted compatible randomization when
  estimating uncertainty for the exact transformed scientific product;
- the transformed route remains unavailable until the owner process supplies a
  content-bound authority with complete transformation and realization parity;
- an owner-defined Wiener transformation learned once and frozen before
  realization application follows ODQ-110A, while use of an NOI product to
  learn/select/update it creates a separate successor transformation/science/
  GEN/UNC generation and per-member learning is a separate ODQ-104 method;
- every numerical Wiener route remains unavailable until its exact owner
  authority, inference/relearning contract where applicable, and NOI boundary
  exist; and
- fixed FRUIT residual and complete relearned FRUIT methods remain separate and
  unavailable until an exact FRUIT boundary exists.

If the transformation owner uses `UNC_k` to learn, select, or update a Wiener
transformation, `UNC_k` is an immutable dependent input to a new owner-
transformation, transformed-science, GEN, and UNC generation. It is not
independent evidence validating the successor, and the successor cannot mutate
or retroactively validate it.

## 11. Persistence And Immutable Companions

ODQ-109 admits three plan-selected modes: persisted ensemble, compact
deterministic regeneration, and streaming sufficient statistics. The plan
records requested/effective/applied/realized mode; there is no universal
default or silent fallback. Persisted mode retains every required member with
exact identity. Compact mode binds immutable parents, exact method/algorithm
versions, frozen operator state, canonical unit ordering, finite design,
admitted membership, assignment key/seed/counter, and full configuration, plus
its byte-identical or numerical reproducibility class and limitation. Dense
signs, per-sample sign provenance, randomized timestreams, and member maps are
not universally required when the compact identity reconstructs the declared
scientific product.

Streaming mode retains mathematically sufficient state for every published
product and claim. For the initial second moment this includes exact weighted
accumulation, common-all-member domain/availability, and required design/
dependence/effective-information/estimator-uncertainty state. It declares every
unsupported later member diagnostic, estimator, covariance, projection, or
reanalysis and does not claim a retained ensemble.

ODQ-105A applies in all modes: every admitted member completes, failed
ensembles yield no survivor or partial streaming estimate, and partial
accumulators carry no UNC authority. Plan-required persistence failure is
product failure; planned transience is not. Persistence/regeneration does not
establish adequacy, covariance completeness, calibration, significance,
conformity, performance, readiness, or production authority.

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
second moment. ODQ-106 approves the covariance representation and rank policy,
and ODQ-107 approves the initial inverse conditional second-moment scale while
keeping inverse variance, precision, and consumer weights separately typed.
ODQ-108 approves the initial MAP standardized-signal method and keeps JINC
separate. ODQ-109 approves the three plan-controlled persistence/regeneration
modes and their fail-closed/audit limits. ODQ-110A keeps transformation
definition with the appropriate scientific process and requires exact
application to admitted randomizations for transformed-product uncertainty.
ODQ-110B classifies owner-frozen Wiener transforms under ODQ-110A, requires a
new immutable generation when NOI products inform owner learning/update, and
keeps per-member learning separate under ODQ-104. The next walkthrough question
is `SCI-NOI-ODQ-110C`. Every later
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
