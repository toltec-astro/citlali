# SCI-FRUIT v0.1 — ODQ-001F Profile-Qualified Empirical-Development Frame

Revision: `r0.8`

Status: **exact Stage A final candidate for owner review; Disposition B is
approved in principle but ODQ-001F remains open; no method-development lane,
qualification, Stage B, implementation, readiness, production, or Unity work
is authorized**

## Purpose And Claim Layer

ODQ-001E defines scientific quality as a constrained multi-objective
comparison with exact historical Citlali. It does not imply one globally
optimal or universally best FRUIT recurrence. This frame defines the process
by which exact methods could later earn baseline-relative qualification over
declared domains; it does not parameterize or select a method.

The complete claim-layer sequence is:

```text
candidate-neutral scientific framework
  -> separately authorized quarantined method development
  -> frozen held-out qualification
  -> scientific-owner-approved qualified-method record
  -> sealed implementation-blind author packet
  -> method-specific Stage B authorship
  -> scientific-owner acceptance and freeze
  -> implementation conformity
  -> empirical/observational validation and achieved performance
  -> readiness
  -> production authorization
```

Each transition is a separate gate. Qualification does not automatically
supply a scientific contract, implementation conformity, validation, achieved
performance, readiness, production authority, or operational fallback. The
exact boundary is stated in
[`PROGRAM_CLAIM_LAYER_SEQUENCE.md`](PROGRAM_CLAIM_LAYER_SEQUENCE.md).

## Revised ODQ-001F

**Which development and qualification architecture shall instantiate the
approved ODQ-001E comparison framework without selecting a recurrence in
advance?**

### Disposition A — Complete A Priori Universal Parameterization

Require one universal science objective, metric set, validity domain, and
numerical acceptance thresholds before any candidate is developed. This is
honest only if those quantities can be justified independently of candidate
behavior and intended recovery regime.

### Disposition B — Baseline-Relative, Profile-Qualified Staged Development

Approve the bounded architecture in this frame. Its purpose is to identify
demonstrable improvements over the exact historical paired control within
prospectively declared profiles and applicability domains, not to find a
global optimum. Candidate-neutral declaration and claim invariants come first;
method development, if separately authorized, occurs only on an identified
development population; complete methods and decision rules freeze before
held-out qualification outcomes are opened; only owner-approved qualified-
method records may later be considered for a sealed Stage B author packet.

Disposition B permits thresholds that are not presently knowable to be set by
a declared development procedure. It forbids changing them after qualification
outcomes are opened. The owner has approved this disposition in principle and
directed the present repair. Final approval has not yet been recorded.

### Disposition C — Historical Compatibility v0.1 With Separate Successor R&D

Limit v0.1 method consideration to a historical-compatibility candidate and
defer intentional-new-method research to a successor tranche. This avoids an
empirical-development lane in the present lifecycle but defers the owner's
provisional Choice 3 direction.

Disposition B is the exact final candidate presented for owner review.

## Baseline-Relative Qualification, Not Global Optimization

Exact historical Citlali is the mandatory paired scientific control, not
truth. Known injected astronomical signal supplies truth only within its
declared validity domain. Nuisance-only and justified null constructions test
whether apparent recovery is atmosphere or another nuisance promoted into the
sky model.

The program operates in a Pareto trade space and has no default scalar score.
Legitimate results include:

1. one method broadly adequate across every tested profile/domain;
2. one recurrence with exact condition- or profile-dependent policies;
3. multiple exact methods with materially useful specialization;
4. qualification restricted to a declared signal, scale, support, or
   observing-condition domain; or
5. no qualifying replacement.

No-replacement is a valid scientific result. It leaves the historical control
and any separately considered compatibility candidate in their typed roles;
it does not create an operational fallback or production decision. The program
prefers the simplest broadly adequate method/policy and retains specialization
only when prospectively evaluated held-out evidence establishes material value.

## Candidate-Neutral Scientific Invariants

Every candidate must bind, without forcing one representation or recurrence:

1. the astronomical estimand, generic science-profile identity, and exact
   applicability domain;
2. the versioned accepted feedback state `F_k` and every causal state capable
   of changing a future result;
3. the model removed from the observation, the residual presented to each
   learning/processing operator, the operations bypassed by the accepted model,
   the rejoin point, and the restored model response in the next map;
4. the distinct parent, feedback model, applied model, residual, update
   contribution, complete iteration product, and terminal product identities;
5. policy and state that are fixed, learned, observation-resolved, applied,
   carried, reset, or relearned, including generation semantics;
6. deterministic replay conditions and complete exact-checkpoint state;
7. response, support, uncertainty, null-space, validity, failure, and typed
   scientific-unavailability claims;
8. causal per-iteration diagnostics sufficient to evaluate the declared metric
   and stopping families; and
9. exact method, claim, evidence, decision, data, software, and environment
   lineage.

Alternative map-domain or fused subtraction/add-back implementations are
admissible only if they demonstrate equivalence to the exact scientific
operator-ordering and bypass semantics. A literal array operation is not
prescribed; the scientific remove/residual/bypass/rejoin/response contract is.

## Method, Claim, Evidence, And Decision Identities

The former `K=(M,P,S,Q,D,H,Pi,E)` tuple is retired. The exact typed structure is:

```text
METHOD_ID = (
  parent_and_reduction_route,
  recurrence,
  feedback_state_schema,
  parameter_or_adaptation_policy,
  stopping_and_terminal_policy
)

CLAIM_ID = (
  science_profile,
  applicability_domain,
  exact_historical_control,
  frozen_qualification_protocol
)

EVIDENCE_ID = (
  population_split,
  execution_generation,
  software_and_environment,
  paired_results,
  uncertainty_and_failure_record
)

QUALIFICATION_DECISION = (
  METHOD_ID,
  CLAIM_ID,
  EVIDENCE_ID,
  disposition,
  owner_and_date
)
```

Evidence generation is not part of `METHOD_ID`. Independent evidence
generations may address the same frozen claim only under a prospectively
declared combination rule. A changed recurrence, feedback schema,
parameter/adaptation policy, or stopping/terminal policy creates a new
`METHOD_ID`. A changed profile, domain, historical control, or qualification
protocol creates a new `CLAIM_ID`. The exact ordinary-MAP, JINC, FLT-FIXED, or
future owner-admitted FLT-MATCHED parent route belongs in `METHOD_ID`; evidence
is not pooled across parent routes by implication. See
[`METHOD_CLAIM_EVIDENCE_DECISION_IDENTITY_TAXONOMY.md`](METHOD_CLAIM_EVIDENCE_DECISION_IDENTITY_TAXONOMY.md).

Stable contribution identity does not require permanent retention. An
increment is not an independently calibrated or scientifically interpretable
sky product unless separately authorized.

## Generic Recovery Profiles And Downstream Boundary

The motivating, not approved, generic profile identities are:

| Candidate profile | FRUIT-owned recovery objective | Outside FRUIT authority |
| --- | --- | --- |
| `compact_high_snr_response_recovery` | Core/wing response, integrated and peak flux, centroid, morphology/width, local false structure, convergence | OOF inference, wavefront interpretation, telescope correction |
| `extended_low_snr_mode_recovery` | Transfer over declared two-dimensional extended modes, integrated-signal and morphology recovery, nuisance leakage, false large-scale structure, convergence | SZE astrophysics, cluster modeling, source claims |

OOF and SZE are motivating populations only unless a later owner decision adds
observation identity to the applicability domain. Profile selection must use
pre-output, owner-authorized facts; it cannot run several methods and choose
the best-looking result. Specialization requires frozen materiality/equivalence
bands, domain prevalence, held-out evidence, and the simplicity/broad-adequacy
rule. See
[`PROFILE_NAMING_AND_SPECIALIZATION_RULE.md`](PROFILE_NAMING_AND_SPECIALIZATION_RULE.md).

## Absolute And Baseline-Relative Evidence

For every metric `G_l`, the protocol declares whether larger or smaller is
scientifically preferable. With `s_l=+1` for larger-is-better and `s_l=-1` for
smaller-is-better, the paired oriented contrast is

\[
  \Delta_{lr}=s_l\left(G_{lr}^{\mathrm{candidate}}
  -G_{lr}^{\mathrm{historical}}\right),
\]

so positive `Delta` favors the candidate. Candidate and historical values use
the same admitted input/truth realization, metric definition, conditioning,
support rule, and target population wherever a valid pair exists.

Absolute truth-referenced and nuisance/null quantities remain mandatory. They
retain physical meaning, expose common-mode failures, and prevent qualification
merely because both methods are poor. Baseline-relative qualification does not
claim historical truth or impose an unsupported perfect-recovery standard.

Before development, each profile must freeze equations, units, normalization,
conditioning, support, and priority for at least astronomical transfer,
mode/flux bias and dispersion, atmosphere and other nuisance leakage,
unsupported/false structure, convergence and time-to-quality, fixed-state and
complete-procedure response/uncertainty, restart/replay, and end-to-end
resources to comparable scientific quality.

The exact candidate metric skeleton and its near-zero, weight, nuisance,
dispersion, and censoring safeguards are in
[`REPAIRED_METRIC_SKELETON.md`](REPAIRED_METRIC_SKELETON.md).

## Profile Priority And Protected Gates

No scalar score or default numerical weight is authorized. Each profile must
prospectively separate:

- owner-prioritized dimensions in which material improvement may qualify;
- protected scientific dimensions subject to non-inferiority/safety gates;
- descriptive/challenge dimensions that may characterize or narrow a claim;
- lower-tail and worst-important-stratum summaries;
- failure, scientific-unavailability, support-loss, and catastrophic-regression
  safeguards; and
- computational metrics, kept separate unless an explicit science/resource
  trade is authorized before qualification.

No universal maximum recoverable scale is primitive. Any reported boundary is
derived from a frozen response criterion, exact mode family, support, and
validity domain.

## Population Separation, Access, And Multiplicity

The protocol binds disjoint immutable development, qualification, and challenge
populations. Development data may support tuning and threshold feasibility.
Qualification data remain untouched until candidate methods, claims, metrics,
thresholds, populations, and decision rules are frozen. Challenge data probe
predeclared edges and count toward a decision only when their role was frozen.
Observation, source, injection, nuisance, and descendant-product lineage must
prevent duplicate, near-duplicate, and descendant leakage.

Before unblinding, the protocol also freezes the maximum candidate/hypothesis
set, submission deadline, candidate/control label blinding, exact unblinding
event, historical-control outcome access, one-shot/sequential design,
population reuse, sequential monitoring, multiplicity across methods/profiles/
metrics/scales/strata/stopping policies, joint confidence or gatekeeping,
minimum effect, statistical credibility, and replacement-population rule.
Winner selection cannot use nominal single-candidate uncertainty. See
[`QUALIFICATION_ACCESS_AND_MULTIPLICITY_RULES.md`](QUALIFICATION_ACCESS_AND_MULTIPLICITY_RULES.md).

A post-unblinding change to a method, claim, metric, threshold, population,
support, failure rule, or policy creates a development candidate requiring
fresh untouched qualification evidence.

## Replication And Inference Target

Each protocol declares a finite-population or superpopulation target, primary
independent sampling unit, observation/source/weather/scan/detector clustering,
within-unit repeats, candidate/control pairing unit, cluster-aware covariance
or resampling unit, metric/stratum dependence, nominal/effective sample size,
missingness/exclusion mechanisms, and supported claim domain.

The conservative default is qualification only for the exact frozen finite
held-out population. A superpopulation claim needs separate sampling or
generative-model authority and matching uncertainty. Multiple modes,
injections, amplitudes, orientations, or iterations in one observation are not
independent astronomical observations by default. See
[`REPLICATION_DEPENDENCE_AND_INFERENCE_TARGET.md`](REPLICATION_DEPENDENCE_AND_INFERENCE_TARGET.md).

## Outcome, Failure, Unavailable, And Support Accounting

The frozen paired outcome matrix distinguishes both methods succeeding,
candidate-only rescue, candidate regression, joint failure, scientific
unavailability of either method, and known inapplicability excluded only by the
prospective target-population rule. Complete-pair selection after outcomes is
forbidden.

On the same target population, report `p_improved`,
`p_practically_unchanged`, `p_degraded`, `p_failed`, and `p_unavailable`, plus
known-inapplicable accounting and exact denominators. See
[`PAIRED_OUTCOME_FAILURE_UNAVAILABLE_MATRIX.md`](PAIRED_OUTCOME_FAILURE_UNAVAILABLE_MATRIX.md).

Spatial and morphological reports separately retain common-support recovery,
candidate support, historical support, support gained/lost, and failure/
unavailable causes. A method cannot improve apparent accuracy by withholding
difficult regions. See [`SUPPORT_COMPARISON_RULE.md`](SUPPORT_COMPARISON_RULE.md).

## Operational Tuning, Adaptation, And Stopping

Offline tuning is confined to the development population and freezes before
qualification. Bounded automatic adaptation uses only causal declared
diagnostics through a deterministic, bounded, versioned mapping. Research or
expert overrides are explicitly experimental and inherit no ordinary claim.

A qualified stop policy may use only diagnostics available during an ordinary
reduction. Injected truth, historical output, future iterations, final desired
quality, and oracle choices may evaluate but cannot drive it. Actual terminal,
oracle best, time-to-quality, censored/nonconvergent time, hard-cap termination,
oscillation, and drift remain separate. Every cap capable of changing the
terminal scientific output belongs to `METHOD_ID`. See
[`OPERATIONAL_STOPPING_AND_ADAPTATION_BOUNDARY.md`](OPERATIONAL_STOPPING_AND_ADAPTATION_BOUNDARY.md).

## Prospective Freeze

The framework distinguishes three times:

1. **Before development:** freeze metric estimands/equations and sign,
   normalization, support, pairing, candidate-neutral priorities, population-
   split construction, target condition distribution/weights, declared strata,
   historical-control identity plan, and externally imposed scientific limits.
2. **After development and before qualification unblinding:** freeze exact
   `METHOD_ID`, `CLAIM_ID`, candidate set, qualification population,
   protected margins, material-improvement rule, uncertainty/multiplicity,
   tail/stratum/outcome/support/failure/unavailable/catastrophic rules, and
   computational protocol.
3. **After qualification begins:** no within-generation method, claim, metric,
   threshold, support, population, or decision-rule changes. Changes require a
   new generation and fresh untouched evidence.

This permits development-informed, prospectively frozen values. It forbids
moving goalposts after candidate/control outcomes are known.

## Qualification Rule

One exact `METHOD_ID` may qualify one exact `CLAIM_ID` only if its frozen
evidence:

1. satisfies validity, absolute truth/null meaning, fixed-state and complete-
   procedure response/uncertainty, support, restart/replay, failure,
   unavailable-state, and disclosure gates;
2. establishes multiplicity-aware non-inferiority on every protected
   scientific dimension, including lower tails, important strata, support,
   failures, and catastrophic-regression limits;
3. establishes a material and statistically credible paired improvement over
   the exact historical control in at least one owner-prioritized scientific
   dimension; and
4. reports computational performance separately unless an explicit
   science/resource trade was frozen before qualification.

The candidate need not dominate every metric or realization. A failing broad
claim may fail entirely or may narrow/stratify only by a rule prospectively
authorized before outcomes. No post-hoc favorable domain carving is allowed.

## Historical Roles And Out-Of-Domain Behavior

`historical_scientific_control`, `historical_compatibility_candidate`, and
`authorized_operational_fallback` are separate identities. The first is
mandatory here. The second is only a compatibility role. The third remains
unavailable until separately approved through scientific contract,
implementation conformity, validation, readiness, and production decisions.
See
[`HISTORICAL_CONTROL_COMPATIBILITY_FALLBACK_TAXONOMY.md`](HISTORICAL_CONTROL_COMPATIBILITY_FALLBACK_TAXONOMY.md).

When a request lies outside all qualified domains or a historical route is
unavailable, the result is an explicitly authorized alternate route, an
experimental override with downgraded claims, or typed scientific
unavailability. Silent fallback, nearest-profile selection, and best-looking-
output selection are forbidden.

## Quarantine And Possible Future Stage B Entry

Only if separately authorized, an empirical lane may inspect implementation,
prototype candidates, and use simulations, injections, observations, and
performance evidence. Its code, searches, tuning history, failed candidates,
and raw evidence remain implementation-informed and outside a later
implementation-blind author packet.

Only a scientific-owner-approved qualified-method record binding exact
`METHOD_ID`, `CLAIM_ID`, `EVIDENCE_ID`, and `QUALIFICATION_DECISION`, plus its
recurrence, operator order, causal state, response/uncertainty/support/validity/
failure/checkpoint claims, compatibility, limitations, and forbidden claims,
could later be sanitized for owner consideration. No such record exists.

## Exact Final Candidate Owner Decision

> Approve the repaired ODQ-001F Disposition B architecture as an exact,
> baseline-relative, profile/domain-aware development and qualification
> framework: exact historical Citlali is the paired scientific control rather
> than truth; separately authorized development, frozen held-out qualification,
> and challenge populations are disjoint and protected against duplicate and
> descendant leakage; methods, claims, populations, metrics, thresholds,
> multiplicity rules, and decision safeguards freeze prospectively; candidates
> must demonstrate protected scientific non-inferiority and material,
> statistically credible improvement in at least one owner-prioritized
> scientific dimension while passing truth/null, lower-tail, important-stratum,
> support, failure, unavailable-state, and catastrophic-regression safeguards;
> the result remains a Pareto comparison that prefers the simplest broadly
> adequate policy and permits material specialization, restricted
> qualification, or no replacement; out-of-domain behavior is explicit; and
> implementation-informed development remains quarantined from later
> implementation-blind Stage B authorship.

**ODQ-001F approval selects a development and qualification architecture.**

**It does not select ODQ-001 recurrence treatment, ODQ-001A–D state/update
semantics, a parent route, or a numerical method.**

**It does not launch the empirical lane.**

The exact content-bound candidate decision record is
[`SCIENTIFIC_OWNER_ODQ_001F_FINAL_CANDIDATE_DECISION_R0.8.md`](SCIENTIFIC_OWNER_ODQ_001F_FINAL_CANDIDATE_DECISION_R0.8.md).
ODQ-001F remains open until the scientific owner explicitly accepts that exact
record.

## Decisions Still Required Before Any Empirical Lane

Even after ODQ-001F approval, separate owner decisions must authorize and bind:

1. the empirical lane and its program lifecycle;
2. exact historical-control build/configuration, parent route, grouping,
   stopping, and paired protocol;
3. generic profile identities, applicability domains, and downstream
   exclusions;
4. sampling/inference target, population construction, strata, and immutable
   split;
5. metrics, truth/null strategy, support, pairing, uncertainty, multiplicity,
   tail/outcome/failure/unavailable/catastrophic rules;
6. protected/prioritized dimensions, materiality/equivalence bands, thresholds,
   and statistical-credibility construction;
7. candidate-family bound, submission/access/unblinding rules, specialization,
   adaptation, stopping, override, and generation policy;
8. out-of-domain, fallback/unavailability, no-replacement, narrowing, and
   evidence-combination policies; and
9. execution repository/branch, inputs, outputs, provenance, review cadence,
   resource bound, and stop rule.

No item above is selected or authorized by this Stage A repair.
