# SCI-FRUIT v0.1 — ODQ-001F Profile-Qualified Empirical-Development Frame

Status: **Stage A owner-review candidate; ODQ-001F remains open; no research,
implementation, qualification, or Stage B launch is authorized**

## Purpose

The owner-approved ODQ-001E gate defines scientific quality as a constrained
multi-objective comparison with exact historical Citlali. It does not require
FRUIT to pretend that the winning recurrence, parameter policy, stopping rule,
or numerical thresholds are knowable before controlled experiments.

This frame parameterizes the *process by which a method may earn scientific
authority*. It does not parameterize or select the method itself.

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

### Disposition B — Profile-Qualified Staged Empirical Development

Approve the bounded architecture below. Define candidate-neutral declaration
and claim invariants first; develop methods on an identified development
population; freeze the method, policies, metrics, thresholds, and untouched
qualification population; qualify only on that frozen protocol; and admit
only a qualified method record to method-specific Stage B authorship.

This disposition allows thresholds that are not presently knowable to be set
from a declared development procedure. It does not allow them to be changed
after qualification data are examined. It is the Stage A response suggested
by the owner's empirical-method concern, but it is not approved by this file.

### Disposition C — Historical-Compatibility v0.1 With Separate Successor R&D

Preserve the recovered historical recurrence as the only v0.1 method candidate
and conduct intentional-new-method research in a separately versioned future
tranche. This avoids changing the present package lifecycle but defers the
owner's provisional Choice 3 direction.

No disposition is selected. Disposition B is specified below so the owner can
review its exact safeguards and consequences rather than approve a slogan.

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

A qualification claim is never simply “FRUIT is qualified.” It has the form

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

Historical Citlali is the mandatory control, not the truth. Known injected
astronomical signal supplies truth for recovery quantities within the injection
validity domain. Nuisance-only or otherwise justified null constructions supply
the complementary check that increased astronomical recovery is not residual
atmosphere or other nuisance promoted into the sky model.

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
   population construction/split rules, comparison control, and any externally
   imposed scientific limits.
2. **After development but before qualification:** exact candidate recurrence,
   parameter/adaptation and stopping policies, protected non-inferiority
   dimensions and tolerances, material-improvement rule, qualification data
   identity, uncertainty rule, missing/failure handling, and computational
   protocol.
3. **After qualification begins:** no method, threshold, profile, population,
   or decision-rule change is allowed within that qualification generation.
   A change creates a new frozen generation and requires a fresh untouched
   qualification population or an owner-approved independent replacement.

This permits empirically informed but prospectively frozen thresholds. It does
not permit qualification thresholds to be invented after outcomes are known.

## Qualification, Compatibility, And Out-Of-Domain Policy

Under the already approved ODQ-001E rule, a candidate qualifies for one `Q,D`
only if it:

1. satisfies the frozen validity, response/uncertainty, restart, reproducibility,
   failure, and disclosure gates;
2. is non-inferior to exact historical Citlali on every frozen protected
   scientific dimension;
3. is materially closer to injected truth in at least one owner-prioritized
   scientific dimension; and
4. reports computational performance separately unless a scientific-resource
   trade was explicitly approved before qualification.

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

The only candidate-specific scientific input eligible for Stage B is an
owner-approved, sanitized, exact qualified-method record binding:

- `K=(M,P,S,Q,D,H,Pi,E)` and its version;
- the recurrence, model construction, operator order, and causal state;
- parameter/adaptation and stopping policies;
- response, uncertainty, support, validity, failure, and checkpoint claims;
- the frozen qualification protocol and exact evidence identity;
- compatibility and out-of-domain behavior; and
- known limitations and forbidden claims.

Stage B then authors the method-specific scientific contract. Production code
may be assessed for conformance only after that contract is frozen.

## Owner Decisions Required To Select Disposition B

The owner must separately approve:

1. the package-specific empirical lane and its relation to the governing
   program lifecycle;
2. the initial profile identities, objectives, and downstream exclusions;
3. the candidate-neutral declaration and claim invariants;
4. the metric equations/priority freeze and truth/null strategy;
5. population construction, split, leakage-prevention, tuning, adaptation,
   threshold-freeze, failure, and replacement rules;
6. the qualification claim identity and acceptance/out-of-domain policy; and
7. the exact authorization, scope, repository/branch, inputs, and stop rule for
   any later bounded research execution.

Until those decisions are recorded, ODQ-001F remains open and no empirical
lane, method study, or implementation is authorized.
