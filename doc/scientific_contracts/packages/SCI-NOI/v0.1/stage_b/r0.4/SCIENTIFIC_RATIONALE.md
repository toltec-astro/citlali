# SCI-NOI v0.1 r0.4 Scientific Rationale

Document identity: `SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.4`

Scientific owner: Grant Wilson

Date: 2026-08-30

Status: implementation-blind proposed final Stage B draft; scientifically
freezeable conditional; not frozen.

Normative authority: the six ordered modules in
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.4`, binding-file SHA-256
`5c6de3bd5180c9231c79cabe5f5918938571340a9f836f437962779e0410d55a`.
This rationale explains that authority and authors no independent normative
equation, requirement, assumption, definition, or prediction.

# 1. Scientific path

```text
Exact retained PTC parent
          |
          v
Active detector sign assignment
          |
          v
Frozen ordinary MAP operator
          |
          v
NOI-GEN realization ensemble
          |
          v
NOI-UNC conditional marginal second moment
          |
          v
NOI-STD independently realized MAP / conditional scale
```

Each arrow is an explicit identity and admission boundary, not automatic
realization. The sign enters once, only on the PTC signal occurrence. The
unsigned denominator, projection, WCS, support, coverage, response, validity,
and every other MAP gate remain fixed.

# 2. Scope, cardinality, and finite design

The owner selected one independent ensemble per exact observation and TolTEC
array. Base v0.1 contains no cross-observation randomization or randomized
coadd.

`N_requested=0` is only an explicit disabled/not-requested state and creates no
design, member, successful empty ensemble, UNC, or STD. An enabled request is a
positive integer. Resolution is atomic: all requested members resolve, giving
`N_resolved=N_requested>0`, or the complete design fails. Candidate rejection
belongs to search construction and is never a member or member failure.

Within an observation/array, positive coefficient-mass detectors are grouped
by exact readout network. Zero-mass detectors remain typed facts outside the
active sign matrix; zero-total strata are not applicable; ambiguous positive-
mass network membership and singleton active strata make design resolution
unavailable. No stratum is silently exempted or cross-balanced.

The accepted design draws independent symmetric sign candidates and retains
the first that satisfies every active network's exact plan-bound tolerance. It
fails closed under its positive finite cap, draws members with replacement,
forces no complement pair, and reports duplicates and complement orbits
separately.

The admitted set is reproducible because every balance quantity is a canonical
reduced arbitrary-precision rational. Exact accumulation follows stable
scientific identity order, and exact integer cross multiplication decides the
boundary. A floating reduction, epsilon, relaxed retry, or best failed
candidate cannot change membership.

Conditional on successful resolution, every admitted vector has the same base
probability. Summing over the common geometric sequence of earlier rejected
candidates gives the same conditional probability for each admitted vector.
The finite cap therefore changes resolution probability, not the conditional
accepted-vector law.

On complete resolution, member assignments are independent uniform draws from
the full admitted set. Candidate streams are disjoint across strata, members,
and attempts under a content-bound generator/key identity. Consequently the
draws, including valid duplicates and possible complements, do not depend on
scheduling, worker count, traversal, storage layout, or persistence mode.

# 3. UNC meaning and information typing

Complement symmetry gives active detectors known target-law sign mean zero.
Through the fixed linear operator, the member-map target mean is zero on exact
available rows. Initial UNC therefore computes the equal-weight marginal
second moment about that known center on the common all-member domain. Its
primary name is
`conditional_detector_sign_randomization_marginal_second_moment`. Equality to
target-law marginal variance is a separate consequence of the known center,
not a license to call it map variance or noise variance.

Initial UNC is an all-members identity: the admitted and resolved member sets
are identical and every resolved member completes, passes policy, shares the
common domain, and enters the estimator. One unavailable or ineligible member
makes the complete estimator unavailable; there is no survivor renormalization
or changed divisor.

The parent can contain source and deterministic residual structure, which can
remain visible in members and their squares. The result is not repeated
physical-noise variance, total uncertainty, covariance completeness,
precision, or calibrated significance.

`N_resolved` counts conditional Monte Carlo draws. It is not exposure,
independent astronomical observations, or an automatic effective sample size.
Member count, unique assignments, complement orbits, and sign/map ranks remain
distinct from estimator-specific effective Monte Carlo information and from
numerical estimator uncertainty.

# 4. STD meaning and response boundary

STD divides an independently realized ordinary MAP signal by the square root
of the compatible conditional second moment. NOI cannot manufacture the MAP
numerator through an all-`+1` assignment. The unit is `1`; the claim is only
that the MAP signal is standardized by the stated conditional scale.

Because numerator and scale descend from the same observed parent, STD is a
nonlinear data-dependent product. Dividing MAP response by the realized scale
is only a fixed-scale conditional derivative. The full response also contains
the response of the conditional second moment and therefore remains
unavailable until that response has separate exact authority. No significance
or detection claim follows.

# 5. Route status

| Route/surface | Method/source state | Exact gates before the next state |
| --- | --- | --- |
| Ordinary parent route | Identity approved; source closed. | Exact PTC coefficient value/QC, canonical rational coefficient, numerical `coverage_cut`, MAP admission, support/WCS/response, and every frozen parent/operator fact. Missing facts make numerical realization unavailable. |
| Product scope | Owner-approved and source closed. | One ensemble per exact observation and TolTEC array; no cross-observation ensemble or randomized coadd. |
| Detailed design | Owner-approved bounded first-accepted rejection family. | Positive request; exact populations and rational arithmetic; exact tolerance/cap; complete generator/key plan; complete assignment resolution. Cap exhaustion fails the design. |
| Numerical GEN | Scientific method identified; not numerically realized here. | All parent/design gates, positive complete resolution, exact generation-input `@2` owner approval and Registry/source binding, response-bearing parent when required, and successful realization. Missing profile authority gives decision unavailable, not ineligibility or empty success. |
| Initial UNC | Estimand and normalization identified; not numerically realized here. | Exact complete GEN; member `@2` and ensemble `@2` owner approval and Registry/source binding; all positive member decisions; equality of member sets/counts; common domain; estimator uncertainty explicit or unavailable. |
| Reciprocal | Excluded from the ordinary base bundle. | New exact owner disposition, method, finite-positive domain, profile, consumer use, and Registry/source binding. Otherwise unavailable. |
| Initial STD | Method and bounded claim identified; full response unavailable. | Independently governed `m_MAP`; compatible finite-positive `Vhat_cond`; standardization `@2` owner approval and Registry/source binding; exact support/response intersection and dependence/cause. Full response additionally requires separately authorized `delta Vhat_cond`. |
| Profiles | Four r0.18 records are immutable and registered; four complete `@2` records are proposed only. | Exact owner approval of successor bytes plus exact SCI-VAL source and Registry binding. A proposed name implies no decision. |
| External transforms | Interfaces bounded; base numerical transformed routes unavailable. | Exact external authority/operator parity; immutable parent/product identity; and new generations for learning, replay, or changed state. |

Scientific method identity and source closure do not establish plan resolution
or numerical realization. A response-bearing realization is a further state;
empirical calibration, implementation conformity, validation, and production
authorization are separate gates and none is supplied here.

# 6. Claim ceiling

This draft defines scientific identities and prospective evidence duties. It
reports no implementation conformity, numerical validation, calibration,
physical-noise validity, covariance completeness, Gaussian significance,
achieved performance, readiness, freeze, production suitability, or
production authorization.
