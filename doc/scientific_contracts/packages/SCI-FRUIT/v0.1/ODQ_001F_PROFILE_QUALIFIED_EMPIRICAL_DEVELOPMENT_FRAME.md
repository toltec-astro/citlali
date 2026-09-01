# SCI-FRUIT v0.1 — ODQ-001F Profile-Qualified Empirical-Development Frame

Status: **Stage A repaired owner-review candidate; Disposition B approved in
principle but final ODQ-001F approval remains open; no research,
implementation, qualification, or Stage B launch is authorized**

## Purpose

The owner-approved ODQ-001E gate defines scientific quality as a constrained
multi-objective comparison with exact historical Citlali. It does not imply
that one globally optimal, uniquely correct, or universally best FRUIT
recurrence exists or can be identified.

This frame parameterizes the *process by which one or more exact methods may
earn baseline-relative scientific qualification over declared domains*. It
does not parameterize or select a method itself.

The proposed sequence is:

\[
  \text{candidate-neutral scientific framework}
  \rightarrow
  \text{controlled method development}
  \rightarrow
  \text{frozen held-out qualification}
  \rightarrow
  \text{method-specific Stage B contract}
  \rightarrow
  \text{production conformance}.
\]

The two empirical steps would be a separately authorized and quarantined
method-development and qualification lane. They are neither Stage B authorship
nor implementation conformity. Creating that lane requires an explicit owner
decision; this file does not create it.

## Revised ODQ-001F

**Which development and qualification architecture shall instantiate the
approved ODQ-001E comparison framework without selecting a recurrence in
advance?**

### Disposition A — Complete A Priori Universal Parameterization

Require one universal science objective, metric set, validity domain, and
numerical acceptance thresholds before any candidate method is developed.

This preserves the ordinary contract-first sequence. It is scientifically
honest only if those quantities can be justified independently of candidate
behavior and the intended observation class.

### Disposition B — Baseline-Relative, Profile-Qualified Staged Development

Approve the bounded architecture below. Its purpose is to identify
demonstrable improvements over exact historical Citlali within prospectively
declared profiles and applicability domains, not to claim a uniquely optimal
algorithm. Define candidate-neutral declaration and claim invariants first;
develop methods on an identified development population; freeze the method,
policies, metrics, thresholds, guardrails, condition strata, and untouched
qualification population; qualify only on that frozen protocol; and admit
only qualified method records to method-specific Stage B authorship.

This disposition allows thresholds that are not presently knowable to be set
from a declared development procedure. It does not allow them to be changed
after qualification data are examined. The owner approves this disposition in
principle subject to the baseline-relative repair directed on `2026-09-01`.
Final approval is not recorded until the owner accepts the repaired language.

### Disposition C — Historical-Compatibility v0.1 With Separate Successor R&D

Preserve the recovered historical recurrence as the only v0.1 method candidate
and conduct intentional-new-method research in a separately versioned future
tranche. This avoids changing the present package lifecycle but defers the
owner's provisional Choice 3 direction.

Disposition B is the conditional owner preference. It is specified below so
the owner can review the repaired safeguards and remaining launch decisions
before final approval is recorded.

## Baseline-Relative Qualification, Not Global Optimization

The empirical program explores a scientific trade space. It is not required to
produce one algorithm that dominates every other method on every metric,
profile, or observing condition. The legitimate outcomes include:

1. one method that performs adequately across all tested profiles and domains;
2. one recurrence with profile- or condition-dependent parameter, adaptation,
   or stopping policies;
3. multiple separately qualified methods occupying materially different and
   scientifically useful points in the trade space;
4. a candidate qualified only over a restricted signal, angular-scale, or
   observing-condition domain; or
5. no new method qualifying, leaving exact historical Citlali as the v0.1
   production method and program fallback.

None is a failure of the empirical-development program. A defensible finding
that no candidate clears the prospectively frozen goalposts is a valid
scientific result.

The program recognizes a Pareto trade space rather than collapsing all
dimensions into a scalar score. Operationally, it prefers the simplest broadly
adequate production policy. Multiple methods or policies are retained only
when prospectively evaluated evidence establishes scientifically meaningful
specialization, not merely small benchmark fluctuations.

## Candidate-Neutral Invariants

These are required *declarations and claim rules* for every candidate. They do
not force every candidate to use the same numerical state or operator order.
Each candidate must bind:

1. the astronomical estimand and science profile;
2. the versioned accepted feedback-state schema `F_k` and every causal state
   capable of changing a future output;
3. the model removed from the observation, the residual presented to each
   learning/processing operator, the operations bypassed by the accepted model,
   its rejoin point, and its realized response in the next map;
4. the distinct identities of parent, feedback model, applied model, residual,
   update contribution, complete iteration product, and terminal product;
5. the fixed, learned, observation-resolved, applied, carried, reset, and
   relearned policy/state and its generation semantics;
6. deterministic replay conditions and the complete exact-checkpoint state;
7. response, support, uncertainty, null-space, validity, and failure claims,
   including typed scientific unavailability outside the qualified domain;
8. per-iteration diagnostics sufficient to evaluate the declared metric and
   stopping families; and
9. exact method, profile, parameter-policy, stopping-policy, data, software,
   and evidence lineage.

This is deliberate pushback on freezing one concrete `F_k` representation or
one historical add-back order before alternative methods are studied. The
semantic questions are universal; their candidate-specific answers must be
explicit, versioned, tested, and ultimately made normative for a qualified
method.

## Conditional Qualification Identity

A qualification claim is never simply “FRUIT is qualified.” Each separately
qualified method/domain has the form

\[
  K=(M,P,S,Q,D,H,\Pi,E),
\]

where:

- `M` is the exact recurrence and model/operator definition;
- `P` is the exact fixed or bounded-adaptive parameter policy;
- `S` is the exact stopping and terminal-selection policy;
- `Q` is one science-profile identity;
- `D` is its observation, signal, nuisance, support, and validity domain;
- `H` is the exact historical Citlali control profile;
- `Pi` is the frozen experiment and qualification protocol; and
- `E` is the immutable evidence/result record.

Changing any scientifically consequential element creates a new candidate or
qualification generation. Evidence from different `K` identities cannot be
pooled as though it qualified one method.

More than one `K` identity may qualify when the evidence establishes material
specialization. Qualification of one identity neither invalidates another nor
authorizes either outside its own `Q,D`. A final production policy must map an
eligible request to an exact qualified identity or to the historical fallback;
it must not search among methods after seeing the desired output.

## Initial Profile Questions

The owner's examples motivate two deliberately distinct candidate profiles:

| Candidate profile | FRUIT-owned recovery objective | Boundary that remains outside FRUIT | Status |
| --- | --- | --- | --- |
| Bright compact/PSF-shape recovery | Core and scientifically relevant wing response, integrated and peak flux, centroid stability, shape/width fidelity, local false structure, high-S/N convergence | OOF inference, wavefront interpretation, and telescope correction remain future OOF authority | motivating candidate only; identity and metrics open |
| Faint extended-emission recovery | Transfer across declared extended spatial modes, integrated-signal recovery, morphology fidelity, nuisance leakage, false large-scale structure, and low-S/N convergence | SZE astrophysical inference, cluster modeling, and source claims remain downstream authority | motivating candidate only; identity and metrics open |

FRUIT may qualify a recovery transformation for inputs resembling these
objectives without defining OOF or SZE science. The owner must approve the
profile identities and validity domains separately.

The study must test the simpler hypotheses first:

1. one recurrence and one parameter/stopping policy spans the profiles;
2. one recurrence spans them with separately versioned profile policies; or
3. distinct recurrence families are scientifically necessary.

Profile or method proliferation requires held-out evidence; it is not assumed
from the motivating examples.

## Truth, Control, And Metric Families

Historical Citlali is the mandatory paired control and scientific-performance
goalpost, not the truth. Known injected astronomical signal supplies truth for
recovery quantities within the injection validity domain. Nuisance-only or
otherwise justified null constructions supply the complementary check that
increased astronomical recovery is not residual atmosphere or other nuisance
promoted into the sky model.

For each metric `G_l`, the protocol must define whether larger or smaller is
scientifically preferable. Let `s_l=+1` for larger-is-better metrics and
`s_l=-1` for smaller-is-better metrics. A paired oriented improvement is

\[
  \Delta_{l r}=s_l
  \left(
    G_{l r}^{\mathrm{candidate}}
    -G_{l r}^{\mathrm{historical}}
  \right),
\]

so positive `Delta_{lr}` always favors the candidate. The candidate and
historical values must use the same admitted input/truth realization,
conditioning, metric definition, support rule, and comparison population
wherever a scientifically valid pairing exists.

Absolute `G_l` values remain mandatory. They expose common-mode failures,
retain physical meaning, and prevent a candidate from qualifying merely because
both methods are poor. Qualification does not require an unsupported absolute
standard of perfect recovery.

Before development begins, each profile must freeze the equations,
normalization, conditioning, and priority order for at least:

- astronomical transfer versus declared two-dimensional spatial mode or
  morphology;
- recovered flux or amplitude bias and dispersion by mode and realization;
- atmosphere and other nuisance leakage, kept distinct from ordinary noise;
- unsupported or false astronomical structure;
- per-mode convergence trajectory, oscillation, terminal bias, and time or
  iterations to a declared quality region;
- fixed-state and complete-procedure response and uncertainty honesty;
- deterministic replay and exact-restart equivalence; and
- end-to-end resource use and time to comparable scientific quality.

### Candidate Metric Skeleton

Let `j` identify a declared astronomical mode or morphology, `r` a realization,
`k` an absolute iteration, `a_{jr}` its nonzero injected amplitude or flux under
a frozen estimator, and `ahat_{jrk}` the recovered value in common declared
units, grid, kernel/response convention, and support. The proposed minimum
truth-referenced quantities are

\[
  \rho_{jrk}=\frac{\widehat a_{jrk}}{a_{jr}},
  \qquad
  T_j(k)=\frac{\sum_r w_{jr}\rho_{jrk}}{\sum_r w_{jr}},
  \qquad
  B_j(k)=T_j(k)-1,
\]

and

\[
  D_j^2(k)=
  \frac{\sum_r w_{jr}\left(\rho_{jrk}-T_j(k)\right)^2}
       {\sum_r w_{jr}}.
\]

The realization weights `w_{jr}`, exclusion rule, small-sample treatment,
uncertainty interval, and any robust or stratified successor to these
illustrative moments must be frozen before candidate-method development. A
zero or unavailable truth amplitude does not enter a ratio by numerical
convention.

For a nuisance family `u` with declared injected amplitude `b_{ur}`, a paired
full-procedure coupling candidate is

\[
  L_{u\rightarrow j}(k)=
  \frac{\sum_r v_{ur}
    \left[
      \widehat a_{jrk}(m+n+b_{ur}u)
      -\widehat a_{jrk}(m+n)
    \right]/b_{ur}}
  {\sum_r v_{ur}},
\]

where `m` is the injected astronomical realization, `n` is the paired
background/noise realization, and the two evaluations use the same base
realization under a prospectively declared pairing rule. This is distinct from
the distribution of false recovered structure on a nuisance-only/null input.
Both are required because low average coupling can coexist with unacceptable
localized or extreme false structure.

Convergence must keep truth error and inter-iteration stability separate, for
example

\[
  e_{jr}(k)=
  \frac{|\widehat a_{jrk}-a_{jr}|}{s_{jr}},
  \qquad
  d_{jr}(k)=
  \frac{|\widehat a_{jrk}-\widehat a_{jr,k-1}|}{s_{jr}},
\]

with a frozen positive scale `s_{jr}`. Small `d` is not evidence of small `e`.
The terminal metric must report both, oscillation or monotonic drift, and the
iterations and end-to-end time to a prospectively declared multi-dimensional
quality region.

For shape-sensitive profiles, amplitude/flux, centroid, normalized morphology,
and support-weighted map residual remain separate estimands. Any map residual
must bind the comparison grid, response/kernel convention, support and edge
domain, weighting measure, background treatment, and missing/non-finite rule.
No one norm may silently replace those disclosures.

Uncertainty qualification must compare declared interval or region coverage
and calibration with truth for fixed-state and complete-procedure targets
separately. Replay/restart qualification compares uninterrupted and checkpoint-
continued realized state, outputs, terminal decision, and failure state under
the exact method identity; a map-only seed is not a restart member.

These equations are an owner-review skeleton, not approved metric authority.
They establish what must be made exact before an empirical lane begins and
which numerical acceptance thresholds may later be informed by development
data. Changing an estimand, metric equation, normalization, or priority after
candidate behavior is examined creates a new study generation and requires an
independent development/qualification split.

The truth-referenced quantities above and paired baseline-relative quantities
answer different questions. `rho`, `T`, `B`, `D`, absolute leakage, and false-
structure statistics state what either method recovered. `Delta` states how the
candidate changed performance relative to historical Citlali. Both must be
reported.

### Candidate Profile Priority Structure

No scalar score or numerical weight is proposed. Each profile instead declares
primary improvement dimensions, protected non-inferiority dimensions, and
descriptive/challenge dimensions:

| Candidate profile | Primary improvement candidates | Protected non-inferiority candidates | Descriptive/challenge candidates |
| --- | --- | --- | --- |
| Bright compact/PSF-shape recovery | Core/wing transfer, normalized shape and width, flux/peak and centroid truth error, stable time-to-quality | Nuisance leakage, false local structure, response/uncertainty honesty, restart/replay, support/failure | Saturation/brightness range, crowding, off-axis/edge support, weather and coverage strata |
| Faint extended-emission recovery | Two-dimensional extended-mode transfer, integrated-signal and morphology truth error, low-S/N convergence | Atmosphere/other leakage, false large-scale structure, noise/covariance behavior, compact-signal non-regression where declared, response/uncertainty honesty, restart/replay | Orientation, scale, amplitude, scan/cross-linking, coverage, weather, foreground/bright-contaminant strata |

The owner must decide which entries are actually primary or protected before
candidate-method development and which have an externally imposed scientific
threshold. The development population may inform feasible prospective
thresholds but may not reclassify a poor dimension after candidate outcomes
are seen.

No universal maximum recoverable scale is primitive. A reported boundary must
be derived from a frozen response criterion for an exact mode family and
validity domain.

## Prospective Meaning Of “Better Under Most Conditions”

A candidate need not improve every realization or dominate every condition,
but a favorable average is insufficient. Before the qualification population
is opened, each proposed `Q,D` must freeze:

1. the applicability-domain population and its inclusion, exclusion, missing,
   non-finite, and failure rules;
2. the target distribution or explicit scientific weighting of observing
   conditions within that population;
3. signal profiles, morphologies, amplitudes, orientations, angular scales,
   supports, and nuisance families;
4. the paired aggregate estimator for each metric, such as a declared mean,
   median, stratified estimator, or other exact functional;
5. uncertainty and statistical-credibility construction for every aggregate;
6. lower-tail, adverse-case, and worst-declared-stratum summaries;
7. prospective bands defining improved, practically unchanged, degraded, and
   failed cases, and the reported fraction in each class;
8. condition-stratified results and rules for scientifically important strata;
   and
9. catastrophic-regression definitions and maximum permitted counts or rates.

The protocol must state whether inference targets the declared finite
qualification population or a superpopulation represented by it. Sampling,
weights, uncertainty, and claims must match that target.

For a weighted finite-population mean example,

\[
  A_l=\frac{\sum_r \omega_r\Delta_{lr}}{\sum_r\omega_r},
\]

but the mean is not a default: a median, stratified functional, model-based
contrast, or other estimator may be selected prospectively. The protocol must
pair `A_l` with its exact uncertainty construction and with a declared adverse
quantile or lower-tail functional `Q_l(alpha)`.

Given prospectively frozen metric-specific practical-change bands, each
admitted case is classified as improved, practically unchanged, degraded, or
failed. The protocol reports the weighted or unweighted fractions

\[
  (p_l^{+},p_l^{0},p_l^{-},p_l^{\mathrm{fail}}),
  \qquad
  p_l^{+}+p_l^{0}+p_l^{-}+p_l^{\mathrm{fail}}=1,
\]

using the same population target as the aggregate. The same summaries are
reported for every prospectively important condition stratum. A catastrophic-
regression indicator and its maximum permitted count or rate are frozen
separately; it is not absorbed into `A_l` or the ordinary degraded fraction.

If a favorable aggregate conceals a protected-dimension failure, catastrophic
regression, or material degradation in a scientifically important condition
class, the candidate does not receive broad qualification. The outcome must
instead fail, narrow `D`, stratify the qualification, or bind an explicit
condition-dependent policy according to rules frozen before the data are
opened.

## Population Separation And Anti-Leakage Rules

The protocol must bind disjoint, immutable population identities:

1. **Development population** — may be inspected repeatedly and used for
   hypothesis testing, tuning, adaptation design, and threshold feasibility.
2. **Qualification population** — remains untouched by method and threshold
   selection until the complete candidate and qualification protocol are
   frozen; it supplies the primary ordinary qualification evidence.
3. **Challenge population** — probes predeclared edge and near-boundary cases;
   it characterizes or narrows the validity domain and counts toward
   qualification only when its decision role was frozen in advance.

Observation, injected-signal, nuisance realization, and derived-product
lineage must prevent near-duplicate or descendant leakage across populations.
Population exclusions and failures remain visible and cannot be replaced
post-hoc by favorable cases.

## Tuning And Adaptation Classes

| Class | Permitted role | Qualification consequence |
| --- | --- | --- |
| Offline profile tuning | Explore only on the development population; freeze the resulting parameter and stopping policy before qualification | The frozen policy is part of `P` and `S`; later retuning creates a new candidate generation |
| Bounded automatic adaptation | Derive parameters from declared observation diagnostics through a deterministic, bounded, versioned mapping | The diagnostic inputs, mapping, bounds, learned state, failure behavior, and domain are part of the qualified method and must be replayable |
| Research/expert override | Support exploratory work or unusual observations through an explicit override | The result is experimental and does not inherit the ordinary qualified-product claim unless separately qualified |

Unrecorded manual adjustment until a map looks plausible is prohibited from
ordinary qualification. An out-of-domain observation must not be silently
assigned to the nearest profile.

## Staged Threshold Freeze

The framework distinguishes quantities that must be fixed at different times:

1. **Before development:** metric equations, conditioning, priority order,
   pairing and sign conventions, applicability-population construction/split
   rules, target condition distribution/weighting, condition strata,
   comparison control, and any externally imposed scientific limits.
2. **After development but before qualification:** exact candidate recurrence,
   parameter/adaptation and stopping policies, protected non-inferiority
   dimensions and tolerances, material-improvement rule, qualification data
   identity, paired aggregate and uncertainty rules, lower-tail and
   outcome-fraction summaries, catastrophic-regression guardrails,
   missing/failure handling, and computational protocol.
3. **After qualification begins:** no method, threshold, profile, population,
   or decision-rule change is allowed within that qualification generation.
   A change creates a new frozen generation and requires a fresh untouched
   qualification population or an owner-approved independent replacement.

This permits empirically informed but prospectively frozen thresholds. It does
not permit qualification thresholds or condition weights to be invented after
candidate-versus-control qualification outcomes are known.

## Qualification, Compatibility, And Out-Of-Domain Policy

Under the already approved ODQ-001E rule and the `2026-09-01` repair direction,
a candidate qualifies for one `Q,D` only if it:

1. satisfies the frozen validity, response/uncertainty, restart, reproducibility,
   failure, and disclosure gates;
2. remains within the frozen baseline-relative non-inferiority or safety
   guardrail on every protected scientific dimension, including adverse strata
   and catastrophic-regression limits;
3. shows a frozen material and statistically credible paired improvement over
   exact historical Citlali in at least one owner-prioritized scientific
   dimension, with absolute truth/null metrics retaining acceptable scientific
   meaning; and
4. reports computational performance separately unless a scientific-resource
   trade was explicitly approved before qualification.

The candidate need not dominate every metric, realization, or condition. A
qualification claim may be restricted to a signal family, angular-scale range,
or observing-condition domain. A method that improves a population aggregate
but violates a protected tail, important stratum, failure, or catastrophic
guardrail does not qualify broadly.

Historical Citlali remains the paired control and program fallback throughout
development and qualification. No candidate displaces it merely by improving
an average. A specialized candidate does not replace it outside the exact
qualified `Q,D`, and a candidate with an unacceptable failure mode does not
replace it inside that domain. This fallback statement does not newly claim
that every refactored numerical route is available or authorize a production
change.

When a request lies outside all qualified domains, the realized action must be
one of: use an explicitly approved historical/compatibility route, require an
explicit experimental override with downgraded claims, or return typed
scientific unavailability. Silent fallback or nearest-profile selection is
forbidden.

## Quarantine And Entry Into Stage B

If authorized, the empirical lane may inspect implementation, prototype
candidate algorithms, and use simulations, injections, observations, and
performance evidence. Its code, searches, tuning history, failed candidates,
and raw evidence remain implementation-informed material outside the later
implementation-blind author packet.

The only candidate-specific scientific inputs eligible for Stage B are one or
more owner-approved, sanitized, exact qualified-method records, each binding:

- `K=(M,P,S,Q,D,H,Pi,E)` and its version;
- the recurrence, model construction, operator order, and causal state;
- parameter/adaptation and stopping policies;
- response, uncertainty, support, validity, failure, and checkpoint claims;
- the frozen qualification protocol and exact evidence identity;
- compatibility and out-of-domain behavior; and
- known limitations and forbidden claims.

Stage B then authors the method-specific scientific contract or explicitly
bounded family of method/profile contracts. It must not manufacture one
universal recurrence when the qualification evidence supports multiple
domains. Production code may be assessed for conformance only after the
applicable contract is frozen.

## Repaired Candidate Owner Decision

> Approve a staged, profile- and domain-aware empirical qualification program
> whose purpose is to identify demonstrable improvements over historical
> Citlali, not to claim a uniquely optimal FRUIT algorithm. Qualification may
> yield a universal method, conditional policies, multiple specialized
> methods, restricted-domain improvements, or no replacement method.

The owner has approved this direction in principle but required the present
repair before approval is recorded. ODQ-001F therefore remains open pending
explicit acceptance of this exact repaired language.

## Remaining Owner Decisions Before The Empirical Lane Can Launch

Even after final ODQ-001F approval, the owner must separately approve:

1. the package-specific empirical lane and its exact relation to the governing
   contract-program lifecycle;
2. the exact historical-control identity, build/configuration, route, grouping,
   stopping behavior, and paired-execution protocol;
3. the initial FRUIT recovery-profile identities, objectives, applicability
   domains, and downstream OOF/SZE exclusions;
4. the applicability-population construction, target condition distribution or
   weights, signal/nuisance families, condition strata, and immutable
   development/qualification/challenge split;
5. the candidate-neutral declaration/claim invariants, exact metric equations,
   sign conventions, paired estimators, uncertainty, tail summaries, outcome
   fractions, and truth/null strategy;
6. the protected and prioritized dimensions, externally imposed limits,
   development-informed threshold-setting procedure, final non-inferiority
   margins, material-improvement thresholds, failure rules, and catastrophic-
   regression guardrails;
7. the bounded candidate-hypothesis family, simplicity/broad-adequacy rule,
   specialization evidence rule, tuning, automatic adaptation, expert override,
   and candidate-generation policy;
8. the conditional qualification identity, historical fallback, profile
   selection, out-of-domain, narrowing/stratification, and no-replacement
   policies; and
9. the exact authorization, scope, repository/branch, inputs, outputs,
   provenance, review cadence, resource bound, and stop rule for bounded
   research execution.

Until those decisions are recorded, ODQ-001F remains open and no empirical
lane, method study, or implementation is authorized.
