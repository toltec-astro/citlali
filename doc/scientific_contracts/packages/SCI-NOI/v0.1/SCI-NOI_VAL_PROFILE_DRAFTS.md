# SCI-NOI v0.1 — NOI-Owned VAL Profile Drafts

Status: proposed sanitized Stage A profiles; exact bytes await scientific-owner
approval and later versioned SCI-VAL Registry binding

Scientific-policy owner: Grant Wilson for SCI-NOI

SCI-NOI owns every policy below. A future SCI-VAL Registry successor may bind
approved immutable bytes, and VAL Core may evaluate them. SCI-VAL does not
author, broaden, aggregate, or execute these policies. Until owner approval and
exact Registry binding, applicability and decisions are unavailable.

GEN owns realization-member completion, terminal state, duplicate/equivalence,
support, source-imprint, QC, persistence/reconstruction, lifecycle, cause, and
failure facts. NOI's named-use policy owns admission of those immutable facts
to one exact UNC use. VAL binds and evaluates that policy but authors neither
the producer fact nor the NOI policy.

Every profile preserves four separately named fields:

- request: `requested` or `not_requested`;
- applicability: `applicable`, `inapplicable`, or `applicability_unknown`;
- eligibility: `eligible`, `ineligible`, or `decision_unavailable`; and
- realization: `realized`, `incomplete`, `failed`, or `not_produced`.

Only `requested + applicable + eligible + realized` projects to the profile's
exact consumer action. The four fields are not a positional tuple and cannot be
replaced by one generic pass bit.

## `SCI-NOI:generation_input_admission@1`

| Field | Draft binding |
| --- | --- |
| Object/domain | One exact input occurrence or complete parent bundle named by one requested GEN method, including earliest immutable parent and insertion point |
| Exact source packages | For the ODQ-102A/B/C ordinary route, exact proposed `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.6` and its bound frozen `SCI-PTC_TO_SCI-MAP v0.1/r0.1` source; another separately approved method uses only its exact boundary, never an adjacent package by analogy |
| Required facts | Exact parent package/version, product/application generation, observation and stable realized detector/channel/array/readout-network/group identity, ODQ-102B observation-scoped detector coherence identity, ODQ-102C network-local coefficient-mass and complement-symmetry design identity, exact Stage B finite-design mechanics authored under ODQ-102D and later accepted, signal quantity/unit/beam, response/WCS/frame, coefficient/projection/normalization/support/boundary, parent validity/causes/influence, lifecycle, operator route, assignment-method identity, and immutable provenance required by the selected boundary |
| Decisive exclusions | Nonexistent or incomplete parent; typed parent unavailability; incompatible ancestry/generation; missing exact route/profile identity; unresolved required MAP/JINC numerical gate; absent required signal/support/coordinate/permission; or a parent restriction explicitly prohibiting the GEN use |
| Exception authority | None for exact identity, ancestry, required parent completion, source binding, quantity/unit/beam, WCS/frame, lifecycle, or route compatibility. Any other exception requires a versioned NOI profile successor |
| Source-imprint role | Parent source content and causes remain immutable inputs; admission creates no source-cancellation claim |
| Response/uncertainty role | Carried exactly as parent facts; neither is upgraded, synthesized, or made a universal admission prerequisite unless the selected GEN method explicitly requires it |
| Missing/conflict behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a decisive false restriction yields `ineligible`; no row/time/shape/filename/unit-only fallback |
| Aggregation/propagation | Atomic only. No detector, observation, ensemble, or reverse-propagated decision is implied. A generic flag or another-use pass has no veto or rescue effect |
| Lifecycle | Evaluation binds requested/effective/observation-resolved/applied/realized GEN and parent generations plus exact profile/source versions |
| Exact consumer action | Admit the exact object as a candidate occurrence for `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` only. NOI assignment, MAP's conforming inline or materialized application, GEN completion/QC, ensemble completion, and UNC admission remain separate |

## `SCI-NOI:uncertainty_member_admission@1`

| Field | Draft binding |
| --- | --- |
| Object/domain | One atomic GEN realization member `b`, with immutable GEN-owned completion/QC facts, presented to one exact UNC method/use |
| Exact source packages | Exact GEN route-specific method/generation, parent boundary, generation-input profile evaluation, operator graph, assignment design/version, GEN product identity, and proposed UNC method identity |
| Required producer facts consumed without redefinition | Exact member/ensemble IDs; GEN completion and terminal state; assignment/equivalence/duplicate state; earliest parent; fixed `Theta_0` or relearned `Theta_b`; required operator terminal states; NOI realization-product identity; inline/materialized representation state when applicable; exact requested/effective/applied/realized persisted, compact-regeneration, or streaming mode; required payload or exact regeneration/sufficient-statistic state; reproducibility class; audit/reanalysis limitations; unit/WCS/support; source-imprint/QC; lifecycle; causes, failure scope, and provenance |
| Named-use restrictions | The exact UNC method's permitted terminal states, equivalence/duplicate treatment, domain/support, source-imprint class, QC, persistence/reconstruction, response reference, lifecycle, and parent/method compatibility |
| Decisive exclusions | The member's GEN ensemble has any incomplete/failed/unavailable admitted member; missing/ambiguous member or parent identity; disallowed duplicate/equivalence state; incompatible unit/WCS/support/response; missing/conflicting mode or source-imprint/QC facts; silent mode fallback; failed required persistence; inadequate reconstruction identity; insufficient streaming state for the exact use; partial streaming accumulator; mixed method generation; or false/missing source binding |
| Exception authority | None for an admitted-member failure or failed GEN ensemble. The exact UNC-owned policy may interpret another named producer limitation for that use, but cannot relabel completion, failure, duplicate/equivalence, support, QC, or lifecycle facts |
| Source-imprint role | Consumed exactly from GEN; admission creates no source-cancellation, physical-noise, or calibrated-null claim |
| Response/uncertainty role | GEN response state is consumed as a compatibility fact. Member admission supplies no variance, covariance, physical-noise, or adequacy claim |
| Missing/conflict behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; a decisive named-use restriction yields `ineligible`; no finite-payload, completion, generic-QC, or another-use-pass rescue |
| Aggregation/propagation | Atomic member-to-one-UNC-use only. No ensemble admission, producer-fact change, universal veto/rescue, or reverse propagation is implied |
| Lifecycle | Evaluation binds immutable GEN completion generation, requested UNC method/use, and exact profile/source versions; the evaluation has its own request/applicability/eligibility/realization fields |
| Exact consumer action | Admit this exact GEN member as a candidate member for the named UNC method. Ensemble-design admission remains separately required |

## `SCI-NOI:uncertainty_ensemble_admission@1`

| Field | Draft binding |
| --- | --- |
| Object/domain | One exact GEN ensemble presented to one exact UNC method and target/domain |
| Exact source packages | Exact GEN method/generation, approved parent boundary, `SCI-NOI:uncertainty_member_admission@1` bytes/evaluations, and proposed UNC method identity |
| Required facts | `S={s_bg}` and joint law; exact admitted membership; successful completion of every admitted member; `B_requested`, `B_resolved`, `B_completed`, `B_unique`, proposed `B_admitted_for_UNC`, design rank; complement/duplicate/dependence state; fixed/relearned graph; source-imprint state; exact persistence mode generation and requested/effective/applied/realized state; for compact mode, complete regeneration identity and reproducibility class/limitation; for streaming mode, target-specific mathematical sufficiency, common-member/domain and dependence state, and unsupported reconstruction/reanalysis list; for an ODQ-110A transformed-product method, exact external scientific owner/authority, immutable transformed science product, transformation version/generation/operator/state/parameters/order/domain/support/edge/missing-data/normalization/unit/response/lifecycle/failure identity, and exact parity for every admitted compatible randomization; for an ODQ-110B Wiener successor, exact immutable prior NOI input, external owner learning/selection/update target, inputs/population/dependence/regularization/state/response/support/stopping/failure/lifecycle, resulting transformation and transformed-science generation, transformed GEN and successor UNC generation, and explicit nonindependence; for the ODQ-105B method, known zero center, exact normalized finite-design weights, no empirical recentering or `B-1`, and the exact common all-member finite-value domain; target, estimator, representation, response reference, effective information, estimator-uncertainty state, calibration, limitations, lifecycle, and provenance; for any separately identified covariance method, exact member population, domain/support, rank or rank limit, null/unresolved modes, regularization/approximation, omissions, and missing/unreported-covariance state; for ODQ-107, exact parent second-moment generation, finite strictly positive reciprocal domain, inverse squared signal unit, no-implicit-regularization identity, and distinct inverse/precision/consumer-weight role |
| Decisive exclusions | Any incomplete/failed/unavailable admitted member; failed GEN ensemble; survivor or partial-ensemble substitution; partial streaming accumulator; silent persistence-mode fallback; failed required persistence; compact mode without complete regeneration/reproducibility identity; streaming mode without sufficient state for the exact product/claim; mixed fixed/relearned graph; mixed parent/method generation; method-invalid completed design; below the UNC method's positive minimum cardinality/rank; for an ODQ-110A transformed-product method, NOI-selected or NOI-defined transformation, missing/conflicting external authority, transformation mismatch/substitution/relocation/omission, member-specific relearning inside a fixed-state method, or attachment to an untransformed/differently transformed product; for an ODQ-110B Wiener successor, same-generation feedback, mutation of prior UNC, missing owner-learning or successor-generation identity, independent-validation claim for the prior UNC input, or fixed/per-member-relearned mixture without an authorized estimand; for ODQ-105B, empirical-mean subtraction, `B-1`, inferred equal weights, member-subset use outside the common domain, or promotion of the pointwise second moment to covariance by shape; for covariance methods, inferred zero/independent missing blocks or undeclared member/domain/rank/null/regularization state; for ODQ-107, zero/negative/nonfinite/unavailable parent, numerical-zero substitution, implicit floor/cap/epsilon/regularization, inverse-variance or precision relabeling, portable consumer-weight inference, or PTC/MAP promotion; incompatible unit/WCS/response/support; unavailable target/estimator identity; or missing source-imprint disclosure |
| Exception authority | None for admitted-member failure, failed GEN ensemble, or survivor/partial-ensemble substitution. A future separately authorized covariance method may govern exact within-product/domain missingness only; it cannot rescue member failure or silently change member populations |
| Source-imprint role | Admission preserves the exact Stage-B-authored source-imprint terminology and the owner-approved meaning that randomization intends source suppression but does not by construction establish source-free maps; it does not promote physical-noise or calibrated-null meaning |
| Response/uncertainty role | Exact target/response reference is required. The decision authorizes only the stated UNC calculation, not every covariance representation or consumer use. The approved reciprocal remains an inverse conditional second-moment scale, not inverse variance or precision |
| Missing/conflict behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; decisive design/compatibility failure yields `ineligible`; large count, balance, or completed payload cannot rescue the decision |
| Aggregation/propagation | One ensemble-to-one-UNC-method decision. No universal detector/pixel/consumer aggregate and no reverse modification of members or parent |
| Lifecycle | Binds exact GEN and UNC requested/effective/observation-resolved/applied/realized generations and immutable completed membership |
| Exact consumer action | Admit the exact complete all-members-successful GEN ensemble to exactly the named UNC estimator/domain; for ODQ-105B, authorize only the zero-centered design-weighted second moment on the common all-member domain. For any separate covariance method, authorize only its exact representation/domain/rank/null/regularization claim; unreported covariance remains unknown/unavailable. For ODQ-107, authorize only `W_hat_cond=1/V_hat_cond` on the finite strictly positive parent domain with unavailable-not-zero behavior and no PTC/MAP promotion. No UNC product exists until estimator realization and atomic publication succeed |

## `SCI-NOI:standardization_admission@1`

| Field | Draft binding |
| --- | --- |
| Object/domain | For the initial method, one exact immutable normalized real-observation MAP signal and its exact compatible ODQ-105B conditional second moment presented to `NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1` |
| Exact source packages | Exact approved MAP parent boundary, exact `V_hat_cond` UNC product/method generation, canonical square-root transformation, and exact STD method identity; JINC requires a separate future method/profile revision |
| Required facts | Exact MAP signal estimand/product/generation and same-frozen-operator relation; exact `V_hat_cond` method/generation; `sigma_cond=sqrt(V_hat_cond)`; finite strictly positive direct denominator in the MAP signal unit; exact valid-domain intersection; parent/method, unit/beam, response, WCS/support/validity/lifecycle compatibility; numerator/scale dependence; local invalid/unavailable behavior; output unit `1`; and exact conditional-scale-standardized claim |
| Decisive exclusions | Incomplete/unavailable MAP parent; JINC or substituted numerator; direct use of variance/covariance/inverse scale/precision/weight as denominator; algebraic inverse-scale route treated as an implicit second method; nonpositive/nonfinite/unavailable scale; interpolation or domain extension; incompatible estimator, response, unit/beam, WCS, support, validity, lifecycle, or generation; or absent dependence/claim identity |
| Exception authority | None for direct denominator positivity, unit compatibility, exact parent/method identity, or immutable parentage. Stronger statistical claims require a separately versioned policy and authority |
| Source-imprint role | Signal and scale source-imprint/dependence states remain explicit; standardization does not remove source leakage |
| Response/uncertainty role | The scale's exact UNC target and transformation are required; standardization creates neither a new uncertainty estimate nor a MAP/JINC covariance/response product |
| Missing/conflict behavior | Missing/conflicting required facts yield `applicability_unknown` and `decision_unavailable`; local invalid scale yields unavailable on its exact domain, never numeric zero or infinity |
| Aggregation/propagation | Exact numerator/scale rows only. No automatic interpolation, domain extension, aggregate pass, or reverse change to either parent |
| Lifecycle | Binds exact signal, UNC, and STD requested/effective/applied/realized generations plus immutable parent joins |
| Exact consumer action | Permit construction of `S_cond=q_MAP/sqrt(V_hat_cond)` as `empirical_scale_standardized_signal`, unit `1`, on the exact compatible finite-positive intersection with claim “MAP signal standardized by the stated conditional randomization second-moment scale” only |

## Supersession And Claim Boundary

Any change to a profile's object/domain, source binding, required fact,
restriction, exception, source-imprint/response/uncertainty role, missing or
conflict behavior, propagation rule, lifecycle, or consumer action requires a
new immutable profile version and new evaluation generation. Registration or
evaluation establishes no implementation conformity, calibration,
physical-noise validity, Gaussian significance, performance, readiness, or
production authorization.
