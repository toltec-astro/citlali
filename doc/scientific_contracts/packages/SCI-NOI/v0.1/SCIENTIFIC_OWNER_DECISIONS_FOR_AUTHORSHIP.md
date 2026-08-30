# SCI-NOI v0.1 — Sanitized Scientific-Owner Decisions For Authorship

Artifact identity: `SCI-NOI_OWNER_DECISIONS v0.1/r0.16`

Status: ODQ-101, ODQ-102A, ODQ-102B, ODQ-102C, ODQ-103, ODQ-104, ODQ-105A, ODQ-105B, ODQ-106, ODQ-107, ODQ-108, ODQ-109, ODQ-110A, and ODQ-110B owner-approved; exact
finite-design mechanics delegated to the scientific-contract author under
ODQ-102D; later decisions remain open; not yet an allowed Stage B authority

Scientific owner: Grant Wilson

Prepared: `2026-08-29`

This is the sanitized decision artifact for a future implementation-blind
author. Approval of one decision changes only the bytes listed for it. It does
not approve another decision or make an unresolved numerical route available.

## Walkthrough Order And Stable IDs

### `SCI-NOI-ODQ-101` — ordinary conditioning family

- **Exact question:** Shall fixed-state conditional-sign GEN be the ordinary
  base-v0.1 conditioning family, with every relearned method kept separate?
- **Approved disposition:** Yes. Reduce the real observation first; freeze the
  state learned by Citlali; generate the ordinary noise ensemble through that
  fixed reduction. A method that relearns any pipeline state is separately
  identified. Fixed-state and relearned members shall never be combined in one
  uncertainty estimate.
- **Alternatives:** make relearning ordinary; mix both member classes; or leave
  the conditioning family unspecified.
- **Scientific consequences:** ordinary GEN estimates variation under its
  declared assignment design conditional on realized learned state; it omits
  learning-procedure variation.
- **Conservative state:** approved conditioning family; ODQ-102A separately
  selects its ordinary route, which remains numerically unavailable at its
  declared gates.
- **Affected artifacts:** GEN graph, Scope Brief, taxonomy, decision ledger,
  parent boundaries, manifest.
- **Exact bytes changed by approval:** the ODQ-101 approval record and explicit
  `ODQ-101 approved` dispositions only. No route, estimator, or profile byte is
  approved by implication.

### `SCI-NOI-ODQ-102A` — exact ordinary parent and insertion route

- **Exact question:** Which complete parent/insertion route, if any, is the
  ordinary numerical GEN route?
- **Approved disposition:** select
  `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`. Apply the NOI-defined
  realization assignment at the exact PTC-to-MAP numerical boundary. MAP may
  consume and apply the modifier inline during ordinary frozen accumulation;
  no materialized randomized timestream is required. NOI owns assignment,
  ensemble design, member/ensemble identity, and the realization product. MAP
  owns only conforming application within its ordinary accumulation arithmetic.
  The output is an NOI realization map produced through the frozen MAP
  operator, not an ordinary MAP science product.
- **Alternatives:** select another single route, several separately named
  routes, or none.
- **Scientific consequences:** fixes randomized quantity, insertion point,
  frozen host, output/unit/WCS, support, response, ownership, and product class;
  materialized versus inline modifier application is representation only.
- **Conservative availability state:** the route identity is approved but
  remains numerically unavailable until the selected PTC MAP-facing
  coefficient family/value/QC and exact numerical `coverage_cut` state/value
  and failure policy are realized. Other routes remain unselected and
  unavailable.
- **Affected artifacts:** three parent boundaries, GEN graph, Scope Brief,
  taxonomy, profile inputs, products.
- **Exact bytes changed by approval:** selected ordinary method identity;
  PTC-to-MAP insertion and optional inline-application representation; NOI/MAP
  ownership; NOI realization-product classification; route-specific
  availability wording; and downstream next-question status only.

### `SCI-NOI-ODQ-102B` — initial coherence-unit family

- **Exact question:** Which stable coherence partition is initially admitted
  for the selected GEN route?
- **Approved disposition:** use one observation-scoped realized
  detector/channel as each coherence unit. One assignment value is constant
  over every admitted PTC occurrence belonging to that detector/channel within
  that observation. Order units lexicographically by canonical observation UID
  and stable realized detector/channel UID; never by container position.
- **Alternatives:** detector-by-scan, one scan/chunk shared across detectors,
  subscan/residual block, whole observation, per-sample, or no admitted
  partition.
- **Scientific consequences:** preserves each detector's within-observation
  temporal structure while allowing detector contributions to be randomized.
  It does not by itself specify independence, balance, or the sign law, and it
  does not preserve detector-detector correlations under every later law.
- **Conservative availability state:** the coherence partition is approved;
  numerical assignment remains unavailable until ODQ-102C supplies an exact
  sign law and finite design and the selected route's numerical gates pass.
- **Affected artifacts:** design specification, selected boundary, GEN graph,
  GEN admission profile.
- **Exact bytes changed by approval:** canonical observation-scoped
  detector/channel partition, constant-within-observation rule, stable unit
  identity/order, applicability, route compatibility, and next-question state.

### `SCI-NOI-ODQ-102C` — ordinary sign and balance family

- **Exact question:** What detector sign range and marginal law, balance
  quantity and domain, detector-count relation, and future-family allowance
  govern the ordinary design? The earlier aggregate design question was
  decomposed before approval; ODQ-102D owns its exact finite mechanics.
- **Approved disposition:** use a network-stratified,
  coefficient-balanced randomized sign family. For each exact observation,
  derive detector coefficient mass
  `B_d = sum_p sum_{i in C_p, detector(i)=d} a_pi` from exact frozen
  MAP-admitted positive contributions `a_pi = G_pi gamma_i`. Balance the
  positive- and negative-sign coefficient totals separately inside every
  stable readout network; no network, array, or observation may balance
  another. The admissible design is complement-symmetric and gives
  complementary assignments equal probability, preserving marginal detector
  sign probability `1/2`. Detector-count balance is not additionally required.
  `B_d` is an NOI design coefficient, not precision, empirical uncertainty
  weight, exposure, validity, or a replacement MAP-facing coefficient.
- **Alternatives:** independent Bernoulli, balanced sampling, exhaustive
  enumeration, count balance, observation-global coefficient balance,
  pixel-vector or source-template balance, paired complements, or unavailable.
- **Scientific consequences:** reduces declared network-local MAP coefficient
  imbalance without claiming pixelwise source cancellation. Assignments are
  not product-independent, so later UNC must consume the realized design and
  dependence rather than assume independent draws. Future design families
  remain available as separately named studies.
- **Conservative state:** the balance family is approved, but no numerical
  design is realizable until ODQ-102D selects its exact mechanics.
- **Affected artifacts:** design specification, GEN graph, UNC table, profiles,
  persistence identity.
- **Exact bytes changed by approval:** detector sign range and marginal law;
  network-local coefficient-mass definition and balance domain; complement
  symmetry; no count-balance requirement; non-promotion of the design
  coefficient; future-family allowance; conditional-dependence and claim
  boundary; and next-question state.

### `SCI-NOI-ODQ-102D` — exact balanced finite-design mechanics

- **Exact question:** What network-local imbalance functional and tolerance,
  conditional randomization/search algorithm, canonical key/version,
  requested/resolved count, retry/failure rule, member dependence,
  complement-pairing, replacement, equivalence, duplicate, and rank rules make
  the approved ODQ-102C family numerically realizable?
- **Owner disposition:** delegate selection and justification of the exact
  mechanics to the implementation-blind scientific-contract author. The
  tolerance-conditioned scheme below is approved only as a suggestion, not as
  normative mechanics. The author may adopt, revise, or reject it while
  preserving every ODQ-102A/B/C boundary and claim restriction, and shall
  return one precise scientific question if the admitted packet is
  insufficient.
- **Nonbinding author suggestion:** use
  `Delta_h = abs(sum_{d in h} s_d B_d) / sum_{d in h} B_d`; generate symmetric
  candidates independently within each network; require every network to meet
  an explicitly requested tolerance with no inferred value; bind a
  deterministic versioned counter-based generator and retry cap; fail closed
  rather than relax tolerance; force neither complement pairs nor detector-
  count balance; and retain duplicates and realized rank as design facts.
- **Alternatives:** exact optimum partition, best-of-fixed-candidate pool,
  tolerance-conditioned sampling, deterministic balanced block design, or
  unavailable.
- **Scientific consequences:** completes numerical reproducibility, finite
  dependence, failure, rank, and effective-information semantics.
- **Conservative state:** the authorship disposition is decided, but no
  numerical assignments or completed design exist. The ODQ-102C family remains
  unavailable until exact Stage B mechanics are authored and later accepted.
- **Affected artifacts:** design specification, GEN graph, UNC table, profiles,
  persistence identity.
- **Exact bytes changed by this disposition:** author-selection authority;
  nonbinding suggested mechanics; required author disclosures and preserved
  constraints; precise-question fallback; and no-advance-acceptance state.

### `SCI-NOI-ODQ-103` — source imprint and cancellation claim

- **Exact question:** What source content and cancellation claim may accompany
  ordinary GEN?
- **Approved scientific disposition:** randomization is intended to suppress
  source signal; it does not, by construction alone, establish that the
  resulting maps are source-free. Bind source content, suppression/cancellation
  target and assumptions, finite residual, consequential support/operator
  effects, structured residuals, source-model error, and leakage. Claim no
  physical-noise ensemble, source-free null, or calibrated null absent separate
  scientific authority and evidence.
- **Terminology disposition:** the Stage-B scientific author may select less
  painful scientist-readable terminology. The provisional
  `source_imprinted_conditional_randomization_ensemble` phrase is nonbinding;
  any replacement must preserve the approved scientific meaning and must not
  imply guaranteed source removal.
- **Alternatives:** narrower route-specific imprint, separately demonstrated
  source suppression, or unavailable.
- **Scientific consequences:** balance alone cannot imply pixelwise source
  cancellation or null calibration.
- **Conservative state:** scientific interpretation approved; exact Stage-B
  terminology and numerical realization remain unavailable.
- **Affected artifacts:** design/imprint specification, GEN graph/products,
  UNC claim ceiling, VAL facts.
- **Exact bytes changed by approval:** source-suppression intent; no-source-free-
  by-construction boundary; exact disclosure fields; prohibited claim ceiling;
  and Stage-B terminology freedom without semantic weakening.

### `SCI-NOI-ODQ-104` — fixed/relearned state classification

- **Exact question:** Must every route classify scientifically consequential
  adjacent reduction state as fixed, rerun/relearned, not applicable, or
  unavailable?
- **Approved disposition:** Yes, by explicit owner approval. A relearned method
  identifies the scientifically consequential stages it reruns or relearns and
  the resulting state that may differ from the real-observation reduction. No
  generic or partially specified “relearned” label defines an authorized NOI
  method. This is a scientific method-definition requirement, not a requirement
  for exhaustive implementation provenance.
- **Alternatives:** infer state from execution, permit partial replay under the
  fixed label, or share one identity across different replay graphs.
- **Scientific consequences:** different conditioning questions cannot share
  an ensemble identity or be mixed in one UNC estimator.
- **Conservative state:** any route without the complete consequential-state
  classification and scientific rerun/relearn graph is unavailable.
- **Affected artifacts:** GEN graph, all route boundaries, taxonomy.
- **Exact bytes changed by approval:** explicit owner approval; the four-state
  classification; consequential rerun/relearn stage and resulting-state
  identity; and the limit against exhaustive implementation-provenance scope.
  Exact ordinary route selection remains ODQ-102A.

### `SCI-NOI-ODQ-105A` — enabled, disabled, and partial completion

- **Exact question:** What completed design is valid when GEN is enabled,
  disabled, or only partly completes?
- **Approved disposition:** disabled is explicit zero-member/no-work and enabled
  requires a positive resolved admitted design. Candidate assignments rejected
  during finite-design construction are not failures and never become members.
  Once admitted, every requested realization must complete through the declared
  frozen operator. Failure of any admitted member fails the whole GEN ensemble
  closed for every NOI-UNC use; surviving members cannot be retained as a
  partial ensemble. GEN reports enough cause/context for diagnosis without
  exhaustive implementation provenance.
- **Alternatives:** admit a declared survivor minimum/rank, allow a UNC method
  to reconstruct a partial design, or treat candidate rejection as failure.
- **Scientific consequences:** finite payload or VAL admission cannot fabricate
  producer completion.
- **Conservative state:** approved fail-closed completion; no partial ensemble
  is available to UNC.
- **Affected artifacts:** product-role table, design specification, GEN
  products, UNC-member-admission profile.
- **Exact bytes changed by approval:** enabled/disabled terminal rules;
  candidate-versus-admitted boundary; all-admitted-member completion rule;
  ensemble-wide UNC invalidation; no survivor reinterpretation; and bounded
  diagnostic cause reporting without exhaustive implementation provenance.

### `SCI-NOI-ODQ-105B` — initial UNC target law and estimator

- **Exact question:** What target law, center, moment/covariance estimator,
  normalization, finite-design correction, and effective-information meaning
  define the first UNC method?
- **Approved disposition:** authorize the zero-centered conditional detector-
  sign-randomization second moment for one exact all-members-successful design.
  On the common domain where every admitted realization supplies a valid finite
  `M_b(p)`, use `V_hat_cond(p)=sum_b omega_b M_b(p)^2`, with exact nonnegative
  normalized weights supplied by the finite design. Do not subtract the finite
  ensemble mean and use no `B-1` correction. Report design dependence, rank,
  effective information, and estimator uncertainty/unavailability; infer
  neither independent observations nor physical repeated-observation noise.
- **Alternatives:** fixed projected uncertainty, second moment about a known
  center, empirical covariance, or unavailable.
- **Scientific consequences:** determines the exact estimated quantity and
  downstream claim ceiling.
- **Conservative state:** the estimand is approved but numerically unavailable
  until GEN, finite-design/weight, all-member-completion, common-domain, and
  use-specific adequacy gates are realized.
- **Affected artifacts:** UNC table/taxonomy/products, ensemble-admission
  profile, STD eligibility.
- **Exact bytes changed by approval:** conditional assignment-law target; known
  zero center; weighted second-moment estimator; no empirical recentering or
  `B-1`; common all-member domain; dependence/rank/effective-information and
  uncertainty-of-estimate reporting; and physical-noise/covariance claim limit.

### `SCI-NOI-ODQ-106` — covariance representation and rank

- **Exact question:** Which covariance representation, domain, rank,
  null-space, regularization, and unavailable states are authorized?
- **Approved disposition:** the ordinary initial representation is the
  ODQ-105B pointwise conditional randomization second-moment field; it is not
  covariance merely because it is pointwise or diagonal-like. Permit retained
  ensemble, named fixed projection, stationary/kernel, block/spectral/sparse/
  low-rank or other exact structured covariance, full covariance, and explicit
  unavailable states as separately identified methods. Never universally
  require dense full covariance. Every covariance method declares its exact
  estimator, member population, domain/support/response, rank or rank limit,
  null space/unresolved modes, regularization/approximation, omissions, and
  uncertainty/calibration state. A retained ensemble is not automatically an
  estimated covariance. Unreported off-diagonal entries or blocks are unknown
  or unavailable, not zero or independence. The ordinary initial method keeps
  common all-member membership and cannot use pairwise populations, survivor
  subsets, or a generic missing-data estimator to rescue a failed member or
  unavailable location. A future separately authorized method may govern exact
  within-product/domain missingness without overriding ODQ-105A.
- **Alternatives:** diagonal-only, retained ensemble, structured, dense, or
  unavailable.
- **Scientific consequences:** useful limited products do not imply
  completeness, invertibility, or zero missing blocks.
- **Conservative state:** the initial second-moment representation is approved
  but numerically unavailable at its existing gates. Every additional
  covariance method is unavailable until exactly defined and admitted. No
  inverse or precision claim follows.
- **Affected artifacts:** UNC table/products, profiles, STD compatibility,
  consumer claims.
- **Exact bytes changed by approval:** initial representation classification;
  optional additional representation families; retained-ensemble non-meaning;
  exact covariance method fields; common-membership and missingness limits;
  unknown-not-zero off-diagonal policy; rank/null/regularization disclosure;
  and no-inverse implication.

### `SCI-NOI-ODQ-107` — empirical inverse and weight products

- **Exact question:** Which inverse-variance, precision, or consumer-effective
  weight products, if any, are authorized?
- **Approved disposition:** authorize
  `NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE` with
  `W_hat_cond(p)=1/V_hat_cond(p)` only where the exact ODQ-105B parent is
  available, finite, and strictly positive. Its role is
  `inverse_conditional_second_moment_scale`, with inverse squared signal units;
  it is not inverse variance or precision. Zero, negative, nonfinite,
  unavailable, or outside-parent-domain input yields unavailable, not a
  numerical zero. Any floor, cap, clipping, epsilon, shrinkage, or other
  regularization is a separately identified method. Marginal inverse variance
  requires a separately authorized finite positive marginal variance.
  Precision requires an authorized covariance and an exact inverse or
  generalized inverse on a declared domain/subspace with rank, null,
  conditioning, and regularization semantics; reciprocal covariance diagonals
  are not precision by default. Consumer-effective weight requires one exact
  named estimator/projection/response/domain and is not portable. None is
  sample validity, support, exposure, a PTC/MAP coefficient, or a parent-
  mutation instruction. Cross-boundary use requires explicit scientific
  authority. A consumer-side numerical zero may represent omission only under
  an authorized application contract that preserves unavailability; it is not
  an estimated inverse value.
- **Alternatives:** authorize a subset, none, or defer to consumers.
- **Scientific consequences:** numerical resemblance cannot create a false
  ownership or coefficient join.
- **Conservative state:** the initial reciprocal method identity is approved
  but numerically unavailable until its parent and strictly-positive-domain
  gates pass. All other inverse, precision, and consumer-effective-weight
  methods remain unavailable until exactly defined and admitted.
- **Affected artifacts:** UNC products, role table, consumer profiles, PTC/MAP
  non-promotion boundary.
- **Exact bytes changed by approval:** initial inverse-scale identity,
  reciprocal transform/domain/unit; unavailable-not-zero and no-implicit-
  regularization behavior; separately typed inverse-variance, precision, and
  consumer-effective-weight requirements; consumer-side omission distinction;
  and PTC/MAP non-promotion/cross-boundary restrictions.

### `SCI-NOI-ODQ-108` — STD numerator and scale

- **Exact question:** Which immutable signal numerator and authorized positive
  same-unit empirical scale define the first NOI-STD method?
- **Approved disposition:** select
  `NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`. Use the exact immutable
  normalized real-observation MAP signal associated with the same frozen MAP
  operator state as numerator. On the exact compatible valid-domain
  intersection where `V_hat_cond` is finite and strictly positive, define
  `sigma_cond=sqrt(V_hat_cond)` and `S_cond=q_MAP/sigma_cond`. Bind exact MAP
  estimator/generation, immutable parent, response, unit/beam, WCS, support,
  validity, lifecycle, UNC method/generation, and transformation. Do not
  interpolate, extend domains, or substitute parents/responses/generations.
  Invalid, unavailable, incompatible, or nonpositive scale makes STD
  unavailable, not zero or infinity. `sqrt(V_hat_cond)` is the canonical
  initial scale route; algebraic `1/sqrt(W_hat_cond)` does not create another
  implicit method. Record numerator/scale dependence. Output
  `empirical_scale_standardized_signal` has unit `1` and means only “MAP signal
  standardized by the stated conditional randomization second-moment scale.”
  It is not Gaussian/Student/z/N-sigma significance, probability, completeness,
  purity, or catalog authority. `jinc_map` requires a separately identified
  future method and exact compatible JINC-specific scale; it cannot inherit the
  MAP route by analogy.
- **Alternatives:** MAP, JINC, or another exact numerator; a different
  authorized scale; or unavailable.
- **Scientific consequences:** dimensionless output remains neither an
  uncertainty estimate nor Gaussian/Student/z/N-sigma/detection significance.
- **Conservative state:** the initial MAP STD method is approved but numerically
  unavailable until all MAP, GEN/UNC, compatibility, finite-positive-domain,
  and admission gates pass. JINC STD remains unselected and unavailable.
- **Affected artifacts:** STD table/products/profile and MAP/JINC boundary.
- **Exact bytes changed by approval:** initial method and MAP numerator
  identity; canonical square-root scale/formula/domain; exact compatibility and
  unavailable behavior; inverse-route non-duplication; dependence disclosure;
  unit/claim ceiling; and separately unavailable JINC method.

### `SCI-NOI-ODQ-109` — persistence and exact regeneration

- **Exact question:** Which persisted-member, transient-regeneration, or
  streaming-sufficient-statistic modes are admitted, with what audit limits?
- **Approved disposition:** admit three plan-selected modes: persisted
  ensemble, compact deterministic regeneration, and streaming sufficient
  statistics. Record requested/effective/applied/realized mode with no silent
  fallback and no universal default. Persisted mode retains every required
  completed member with exact identity. Compact mode binds immutable parents,
  exact method/algorithm versions, frozen operator state, canonical unit
  ordering, finite design, admitted membership, assignment key/seed/counter,
  and full configuration; it declares byte-identical or exact numerical
  reproducibility class and limitation. Dense signs, per-sample sign
  provenance, randomized timestreams, and realization maps are unnecessary
  when that compact record reconstructs the approved assignments/product.
  Streaming mode retains mathematically sufficient state for every published
  product/claim, including initial weighted second-moment accumulation, common-
  member domain, and required dependence/effective-information/estimator-
  uncertainty state, and declares every unavailable later reconstruction or
  reanalysis. ODQ-105A remains absolute: every admitted member completes,
  failed ensembles yield no survivor or partial streaming estimate, and
  partial accumulators carry no UNC authority. Required persistence failure is
  product failure; planned transience is not. All modes retain immutable
  completion/membership/mode/audit identity. Persistence or regenerability
  establishes no adequacy, covariance, calibration, significance, conformity,
  performance, readiness, or production claim.
- **Alternatives:** require all members, regeneration-only, streaming-only,
  selected modes, or unavailable.
- **Scientific consequences:** bounded storage without overstated
  reproducibility.
- **Conservative state:** all three mode families are approved but no mode is
  realized until an exact plan and its required identity/sufficiency record
  exist. Missing mode or reconstruction facts make the affected product
  unavailable.
- **Affected artifacts:** GEN/UNC lifecycle, role table, profile, provenance.
- **Exact bytes changed by approval:** three mode families; requested/effective/
  applied/realized state; no-default/no-fallback rules; compact regeneration
  identity and reproducibility class; streaming sufficiency and limitation
  disclosures; required-persistence failure; fail-closed partial-state rule;
  and no-adequacy implications.

### `SCI-NOI-ODQ-110A` — externally owned deterministic transformation parity

- **Exact question:** When an upstream or downstream scientific process defines
  a deterministic transformation of a scientific product, what does NOI own
  when estimating uncertainty for that transformed product?
- **Approved disposition:** NOI neither chooses nor defines the transformation.
  The appropriate scientific process owns and defines it. NOI binds that exact
  owner-defined transformation and applies exactly it to every admitted
  compatible randomization used to estimate uncertainty for the exact
  transformed scientific product.
- **Alternatives rejected:** NOI selection or definition of the transformation;
  approximate/substitute transformation; assumed commutation or relocation;
  silent omission; or attachment of transformed uncertainty to another parent.
- **Scientific consequences:** transformation ownership remains external to
  NOI; exact owner/version/state/parameter/order/domain/support/edge/response/
  lifecycle/failure parity is method identity. A member-specific rerun or
  relearning is a separate ODQ-104 method.
- **Conservative availability state:** every transformed-product uncertainty
  route remains unavailable until the owning scientific process supplies an
  exact content-bound authority and the NOI method satisfies its parity
  interface.
- **Affected artifacts:** filter scope, GEN graph, route, response, imprint,
  profiles.
- **Exact bytes changed by approval:** external transformation ownership;
  exact realization-parity requirement; fixed/relearned separation;
  transformed-product-only uncertainty scope; and unavailable-until-bound
  state.

### `SCI-NOI-ODQ-110B` — Wiener scope

- **Exact question:** How does NOI treat a Wiener transformation that is fixed,
  learned from prior NOI products, or relearned for individual realizations?
- **Approved disposition:** an exact owner-defined Wiener transformation frozen
  before application to randomizations is governed by ODQ-110A, even if its
  owner originally derived it from data. A Wiener transformation learned,
  selected, or updated from an NOI product begins a separately versioned
  successor scientific-product/GEN/UNC generation. Per-realization learning is
  a separate ODQ-104 relearned method. NOI owns none of these transformation or
  learning definitions.
- **Alternatives rejected:** NOI-owned Wiener selection/definition; treating
  data-derived as necessarily per-member relearned; same-generation feedback;
  mutation of prior UNC; or mixing fixed and relearned members without an
  authorized mixture estimand.
- **Scientific consequences:** `UNC_k` may be a declared input to the owner-
  defined successor transformation but is not independent evidence validating
  that transformation or `UNC_(k+1)`. Exact transformation/product/generation
  identity and dependence are preserved.
- **Conservative availability state:** a fixed Wiener route remains unavailable
  pending ODQ-110A owner authority/parity; feedback or per-member routes remain
  unavailable pending complete owner-authored inference/relearning contracts
  and exact NOI boundaries.
- **Affected artifacts:** filter scope, graph, lifecycle, response boundaries.
- **Exact bytes changed by approval:** fixed-state classification; external
  authority; successor-generation graph; dependence/nonvalidation rule;
  per-realization separation; and typed unavailability.

### `SCI-NOI-ODQ-110C` — FRUIT scope

- **Exact question:** Is a fixed FRUIT residual or partly/fully relearned FRUIT
  route admitted?
- **Recommended disposition:** keep them separate and unavailable until an
  exact FRUIT boundary and complete route graph exist.
- **Alternatives:** fixed residual only, one relearning graph, exclude FRUIT,
  or retain unavailability.
- **Scientific consequences:** source-model residual sampling is not merged
  with learning-procedure variation.
- **Conservative state while open:** all FRUIT routes unavailable.
- **Affected artifacts:** FRUIT scope, graph, imprint, lifecycle, profiles.
- **Exact bytes changed by approval:** method, parent/rerun graph, state,
  response, and availability.

### `SCI-NOI-ODQ-111` — VAL profile identities and actions

- **Exact question:** Are the four NOI-owned profile identities and one exact
  consumer action each approved for later Registry binding?
- **Recommended disposition:** approve `generation_input_admission`,
  `uncertainty_member_admission`, `uncertainty_ensemble_admission`, and
  `standardization_admission` at `@1`. GEN owns completion truth; NOI policy
  owns use admission; VAL binds/evaluates but authors neither fact nor policy.
- **Alternatives:** revise identity/action, split a profile, or leave
  unavailable.
- **Scientific consequences:** producer facts, policy, and evaluation remain
  separate and use-specific.
- **Conservative state while open:** drafts unregistered and unevaluable; every
  dependent use unavailable.
- **Affected artifacts:** profile drafts, role table, GEN/UNC/STD gates, future
  SCI-VAL Registry/source bindings.
- **Exact bytes changed by approval:** profile identity and exact consumer
  action bytes only; Registry binding remains separate.

## Preserved Architecture And Stage B Gate

The three operator roles remain separate; no GEN-to-UNC-to-STD implication is
allowed; validity, coefficients, assignments, member QC, covariance/inverses,
weights, and standardized signal remain distinct; parents remain immutable;
and no implementation, calibration, significance, performance, readiness, or
production claim is made.

A conditional Stage B contract may be authored while named routes remain
unavailable. Dispatch nevertheless requires approval of the Scope Brief and
every decision either as approved or explicitly open with its dependent method
unavailable; exact parent-boundary binding; approved profile bytes; exact
SCI-VAL Registry/source-binding successors; a complete manifest-bound packet;
and a clean implementation-evidence firewall.
