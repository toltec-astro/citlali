# SCI-NOI v0.1 Stage B Shared Normative Core

Document identity: `SCI-NOI_NORMATIVE_CORE v0.1/draft-r0.1`

Status: implementation-blind Stage B draft; not owner-accepted or frozen.

Normative terms: `shall` is required, `shall not` is prohibited, `may` is
permitted only inside the stated boundary, and `unavailable` is an explicit
scientific state rather than a numerical sentinel.

## A. Scope, identity, and ownership

`NOI-REQ-001` SCI-NOI shall expose `NOI-GEN`, `NOI-UNC`, and `NOI-STD` as
separate scientific operations. No operation shall automatically realize,
authorize, or mutate another.

`NOI-REQ-002` The ordinary GEN method shall be
`NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`, with the exact retained PTC
occurrence as earliest immutable parent, the exact PTC-to-MAP numerical
boundary as insertion point, and the complete frozen MAP accumulation as host.
The output shall be an NOI realization map, not an ordinary MAP science
product.

`NOI-REQ-003` Every GEN method shall classify every scientifically
consequential adjacent state as immutable/fixed, rerun/relearned, not
applicable, or unavailable. Fixed and relearned members, or members from
different replay graphs, shall not share an ensemble.

## B. Ordinary finite assignment design

`NOI-REQ-004` One ordinary coherence unit shall be one stable realized
detector/channel within one exact observation. Its sign shall apply to every
admitted PTC occurrence of that detector in that observation. Canonical order
shall be lexicographic by canonical observation UID and stable realized
detector/channel UID.

`NOI-REQ-005` For every observation/readout-network stratum `h`, the design
shall derive each detector mass
`B_d=sum_p sum_{i in C_p,detector(i)=d} G_pi gamma_i` from the exact frozen
MAP-admitted positive contribution population. One stratum shall not balance
another. `B_d` shall not be relabeled as precision, empirical uncertainty
weight, exposure, support, validity, or a PTC/MAP coefficient.

`NOI-REQ-006` Each stratum shall bind an explicit exact rational tolerance
`tau_h` satisfying `0 <= tau_h < 1`, with no default. A candidate `s_h` shall
be admissible exactly when
`abs(sum_d s_d B_d) <= tau_h sum_d B_d`. The comparison shall use exact
rational arithmetic over the persisted numerical representations of the
positive contributions and tolerance; traversal-dependent floating summation
shall not decide admission.

`NOI-REQ-007` The target one-member law shall be uniform over every admissible
sign vector in each stratum and the product of those laws across strata and
observations. Candidate signs before conditioning shall be independent uniform
values in `{-1,+1}`. The admissible law shall remain complement-symmetric; it
shall not imply detector independence or detector-count balance.

`NOI-REQ-008` Resolved members shall be sampled with replacement from the
target law. Cross-member streams and observation/network strata shall use
disjoint canonical key domains. Complement pairing shall not be forced.
Duplicates and complements shall not be silently rejected, replaced, or
collapsed.

`NOI-REQ-009` Every plan shall explicitly bind a content-identified random-bit
generator and algorithm version, opaque seed/key bytes and namespace,
`B_requested`, and a positive per-member/per-stratum retry cap. No generator,
seed, count, cap, or tolerance shall be inferred. Missing or conflicting plan
facts shall make design resolution unavailable.

`NOI-REQ-010` Canonical assignment-key serialization shall use
`NOI-KEY-C14N@1`: type-tagged, length-prefixed UTF-8 NFC fields in this order:
SCI-NOI version; route method/version; design generation; earliest immutable
parent product/application generation; observation UID; stable array/group;
stratum type and owner identity; stratum identity; stable unit identity;
member identity; candidate counter; seed/key namespace; random-bit generator
and version. Integers shall be canonical base-10 without leading zeros except
`0`; Boolean and unavailable tokens shall be fixed lowercase; no locale,
platform path, whitespace normalization, container index, floating display, or
implicit absent value shall participate.

`NOI-REQ-011` The design shall use bounded conditional rejection. Candidate
rejections before admission shall be construction outcomes, not members or
failures. If any required stratum has no admissible assignment or no candidate
is accepted within its retry cap, the complete design resolution shall fail
closed without tolerance relaxation, cross-stratum balancing, fallback, or a
partial design.

`NOI-REQ-012` Assignment equality shall require byte-identical canonical sign
serialization on the identical ordered coherence domain and design generation.
Duplicate detection shall compare a SHA-256 digest first and then require byte
equality. Complements shall be paired-but-distinct for full-distribution
identity. The product shall report `B_requested`, `B_resolved`, `B_completed`,
`B_unique`, complement-orbit count, `B_admitted_for_UNC`, design rank, and
use-specific effective information separately.

`NOI-REQ-013` For the initial method, design rank shall be the exact real rank
of the uncentered admitted member-by-coherence sign matrix on its canonical
domain. The product shall report its null space or an exact basis/identifier.
The known-zero second-moment method shall require at least one admitted member
and rank at least one; estimator uncertainty shall be reported or explicitly
unavailable. Count shall not substitute for rank or independence.

`NOI-REQ-014` A successfully resolved design shall set
`B_resolved=B_requested>0` and assign every resolved member the exact weight
`omega_b=1/B_resolved`. Failed design resolution shall publish no member
weights. A later design with unequal weights shall require a new method
version.

## C. GEN product and lifecycle

`NOI-REQ-015` The exact scientist-readable product class shall be
`source-bearing conditional randomization ensemble`. Its declaration shall
record parent source content, suppression target and assumptions, finite
balance residual, support/coefficient/projection/filter variation, structured
residuals, source-model use/error, leakage, and claim limits. It shall not be
called source-free, a repeated physical-noise ensemble, a calibrated null,
variance, covariance, precision, or significance by existence.

`NOI-REQ-016` Every admitted assignment shall complete through the declared
frozen operator. Any incomplete, failed, or unavailable admitted member shall
fail the whole ensemble for all UNC use. Completed survivors and partial
streaming accumulators shall carry no UNC authority. A retry or replacement
shall use a new exact design/generation identity.

`NOI-REQ-017` The plan shall select exactly one persistence mode: persisted
ensemble, compact deterministic regeneration, or streaming sufficient
statistics. Requested, effective, applied, and realized mode shall be distinct;
no default or silent fallback is permitted. Each mode shall retain the exact
identity, completion, limitations, and sufficiency facts required by its
published products and claims.

`NOI-REQ-018` An externally owned deterministic transformation shall be used
only when its scientific owner supplies exact content-bound purpose, operator,
state, parameters, order, domain, support/edge/missing-data behavior,
normalization, units, response, validity, lifecycle, and failure authority.
NOI shall apply it identically to every compatible admitted realization and
shall not choose, tune, substitute, relocate, simplify, or silently omit it.

`NOI-REQ-019` A Wiener or FRUIT transformation frozen before member application
shall remain a fixed owner-transformed route. Use of an NOI product to learn,
select, continue, or update the owner process shall create new immutable
transformation, science-product, GEN, and UNC generations. Per-member learning
or replay shall be a distinct relearned method. Fixed and relearned members
shall not mix.

## D. UNC target, estimator, representation, and inverse

`NOI-REQ-020` Every UNC method shall bind its exact target law, admitted GEN
method/generation and complete membership, center, estimator, design
normalization, missingness/dependence treatment, domain, WCS, support, response,
unit/beam, representation, rank/null space, regularization, inverse domain,
calibration/omissions, estimator uncertainty, lifecycle, claim, and named use.

`NOI-REQ-021` UNC shall consume only an all-members-successful GEN ensemble
admitted by both the exact member and ensemble policies for the named method.
Rejected candidates, failed ensembles, survivor subsets, mixed methods,
partial persistence, and partial streaming state shall not be admitted.

`NOI-REQ-022` The initial UNC method shall compute, on its exact admitted
domain, `V_hat_cond(p)=sum_b omega_b M_b(p)^2` with the design weights required
above. Its center shall be known zero; it shall not subtract the finite
ensemble mean or apply `B-1`.

`NOI-REQ-023` The initial estimator domain shall be
`D_common={p: every admitted M_b supplies a valid finite value at p}`. Outside
`D_common` it shall be unavailable. It shall not use a smaller member subset,
pairwise population, zero fill, interpolation, or domain extension.

`NOI-REQ-024` `V_hat_cond` shall have squared signal units and shall be labeled
a conditional randomization second moment retaining source imprint and
structured residual content. It shall report dependence, complements, all
counts, exact design rank, use-specific effective information, and estimator
uncertainty or its unavailable state. It shall not be promoted to physical-
noise variance, MAP covariance, precision, or significance.

`NOI-REQ-025` Retained ensemble, fixed projection, marginal variance,
stationary/kernel, block, spectral, sparse, low-rank, full covariance, and
unavailable representations shall be separately versioned methods. Every
covariance method shall declare estimator, common member population, domain,
support, response, rank/null modes, regularization/approximation, omissions,
and uncertainty/calibration. Unreported covariance shall remain unknown or
unavailable, never zero or independence.

`NOI-REQ-026` The only initial inverse product shall be
`NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE`, with
`W_hat_cond=1/V_hat_cond` on the exact finite strictly positive parent domain.
It shall have inverse squared signal units and shall be unavailable for zero,
negative, nonfinite, unavailable, or outside-domain input. It shall not be
inverse variance, precision, validity, support, exposure, a PTC/MAP coefficient,
or an implicitly regularized value.

## E. STD product

`NOI-REQ-027` The initial STD method shall be
`NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`. It shall bind the exact
immutable normalized real-observation MAP signal from the same frozen operator
state and compute `sigma_cond=sqrt(V_hat_cond)` and
`S_cond=q_MAP/sigma_cond` only on the exact compatible finite-positive valid
intersection. No interpolation, substitution, implicit inverse route, or
regularization is permitted.

`NOI-REQ-028` The STD output shall be
`empirical_scale_standardized_signal` with unit exactly `1`, explicit
numerator/scale dependence, and the sole claim "MAP signal standardized by the
stated conditional randomization second-moment scale." It shall not claim a
Gaussian, Student, z, N-sigma, probability, detection, completeness, purity,
catalog, uncertainty, or JINC meaning.

## F. Use-specific admission profiles

`NOI-REQ-029` Every NOI-owned admission evaluation shall retain separately
named request, applicability, eligibility, and realization fields. Only
`requested + applicable + eligible + realized` shall project to the exact
profile action. A generic flag or another-use pass shall have no universal
veto, rescue, or propagation effect.

`NOI-REQ-030` `SCI-NOI:generation_input_admission@1` shall permit only the
exact candidate occurrence for the selected PTC-to-frozen-MAP GEN method.
Assignment, host application, member completion, ensemble completion, and UNC
admission shall remain separate.

`NOI-REQ-031` `SCI-NOI:uncertainty_member_admission@1` shall admit one exact
GEN member only as a candidate for one named UNC method. It shall consume but
shall not redefine GEN completion, failure, duplicate/equivalence, support,
source-imprint, QC, persistence/reconstruction, lifecycle, cause, or provenance
facts. Ensemble admission shall remain separate.

`NOI-REQ-032` `SCI-NOI:uncertainty_ensemble_admission@1` shall admit only one
exact complete all-members-successful ensemble to one exact UNC estimator and
domain. For the initial method it shall authorize only the zero-centered
design-weighted second moment on the common all-member domain, and for the
initial inverse only the exact finite-positive reciprocal domain.

`NOI-REQ-033` `SCI-NOI:standardization_admission@1` shall permit only
construction of the exact initial MAP `S_cond` product on its compatible
finite-positive intersection with the claim in `NOI-REQ-028`.

## G. Unavailability, immutability, and claim ceiling

`NOI-REQ-034` Missing, conflicting, incomplete, unsupported, invalid, or
failed required facts shall produce the typed state applicable to their scope.
No undocumented zero, NaN, infinity, empty object, filename, equal shape,
approximate WCS, or hidden default shall create a successful identity join.

`NOI-REQ-035` PTC, MAP, JINC, external transformation, Wiener, and FRUIT
parents shall remain immutable. Every NOI or successor result shall be a new
versioned companion with exact parent, method, generation, dependence, and
lifecycle identity.

`NOI-REQ-036` No NOI assignment, second moment, covariance, inverse, precision,
scale, or weight shall become validity, support, exposure, or a PTC/MAP
analysis, gridding, or coadd coefficient without separately authorized
successor or feedback authority.

`NOI-REQ-037` This draft shall establish no implementation conformity,
representation fidelity, numerical validation, calibration, physical-noise
validity, covariance completeness, Gaussian significance, achieved
performance, readiness, freeze, production suitability, or production
authorization. Every unavailable numerical route shall remain unavailable.
