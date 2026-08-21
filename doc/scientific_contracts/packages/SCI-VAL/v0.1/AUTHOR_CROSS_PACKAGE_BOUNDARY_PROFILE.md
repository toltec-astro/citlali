# SCI-VAL v0.1 — Sanitized RTC/CAL/PTC/MAP Boundary Profile

Status: owner-approved implementation-blind author reference for
`SCI-VAL-v0.1-StageB-packet-r0.2`

Date: `2026-08-20`

## Purpose And Authority Limit

SCI-VAL interchanges producer-owned facts/supports and executes exact
use-owner-supplied policies without taking ownership of either. This profile
supplies the exact cross-package meanings the author must preserve without
exposing full adjacent drafts, implementation, schemas, audits, repairs,
tests, validation, or production state.

This file is an owner-approved sanitized extract. It becomes dispatchable
author authority only after content binding in the packet manifest. It does
not freeze RTC, CAL, PTC, or MAP and does not make an unavailable adjacent
fact available.

## Chain Boundary

The selected v0.1 route is:

`aligned raw paired x/r -> RTC conditioned x -> CAL calibrated x -> PTC
transformed x -> VAL evaluation of the use-owner policy -> optional MAP`.

A direct CAL-to-MAP route is separate authority. Auxiliary conditioned-`r`
analysis is diagnostic-only and inert/advisory relative to calibrated `x` in
base PTC v0.1.

## RTC Facts Supplied To VAL

RTC owns and, when required by a consumer, supplies:

- the exact conditioned-`x` parent, sample grid, detector occurrence, and
  representative source occurrence;
- original, ALIGN-synthesized, RTC-replaced, and other producer-owned origin
  facts;
- typed direct causes, operator masks as controls, edge/state facts, complete
  local support, and transitive causal influence;
- exact versus conservative influence representation, when available;
- RTC-local response and uncertainty availability; and
- immutable plan/lifecycle and parentage.

The producer fact “the exact representative occurrence was synthesized or
replaced” is distinct from its downstream use-specific consequence. RTC owns
the truth and any RTC-local composite/support derived from its facts. RTC does
not provide one universal eligibility bit for nonrepresentative influence or
decide a scientific use owned by another package.

## CAL Facts Supplied To VAL

CAL owns and, when required by a consumer, supplies:

- exact detector/sample identity and admitted RTC parent;
- calibration-factor and target-atmosphere role identity;
- factor/domain, detector-binding, atmosphere, response, and calibration
  availability/validity;
- conditional uncertainty and nuisance/correlation scope; and
- complete upstream response status ending on the CAL detector-time grid.

An invalid or unavailable required CAL fact cannot be repaired or promoted by
VAL. A finite calibrated payload does not establish valid calibration.

## PTC Facts Supplied To VAL

PTC owns and, when required by a consumer, supplies:

- the exact immutable CAL parent and selected fit/application/output role;
- basis-fit, loading-fit, operator-application, output-retention,
  coefficient/QC, response, empirical/simulation, and downstream-support
  facts;
- distinct `fit_invalid`, `fit_excluded_apply_allowed`,
  `postfit_output_reject`, `weight_only`, and advisory states;
- removed-subspace, local response, complete-response availability,
  covariance/uncertainty availability, and coefficient-family identity; and
- fit/refinement/pass lifecycle and parentage.

PTC also owns the Boolean/truth-domain composition that constructs each of its
fit, application, output, coefficient/QC, response, and empirical/simulation
supports. VAL may carry those resolved supports or evaluate a complete
PTC-owned policy; it does not map raw PTC causes into those supports.

Only a fit-support change requires a new PTC fit or fitted-state invalidation.
An output-only or coefficient-only decision does not rewrite the earlier fit
membership decision. Later producer facts create a new VAL decision parent.

## MAP Facts And Authority Retained By MAP

MAP supplies and owns the exact upstream-admission policy for the named
`analysis_or_gridding_contribution` use and may use VAL evaluation mechanics
to evaluate it. MAP retains ownership of:

- geometric incidence and projection;
- estimator contribution;
- upstream-eligible and retained exposure;
- normalization support and its threshold/policy;
- science-policy support and its threshold/policy;
- response, covariance, coaddition, and required companion identity; and
- final raw map-product validity.

The VAL-evaluated upstream-admission result is necessary but not sufficient for MAP
contribution, supported output, or final validity. A consumer-local finite
value, positive coefficient, or supported pixel cannot promote an invalid or
unavailable parent.

## Shared Named-Use Vocabulary

V0.1 shall define a mandatory structural interchange gate plus these reusable
profile names:

| Profile | Scientific question |
| --- | --- |
| `independent_exposure` | May this occurrence count as an original independent astronomical measurement? |
| `estimator_fit` | May it influence learning or fitting of an estimator? |
| `operator_application` | May an already resolved operator be applied to it? |
| `output_retention` | May its resulting value remain in a product? |
| `analysis_or_gridding_contribution` | May it numerically contribute to an analysis or map estimator, subject to consumer-local conditions? |
| `response_companion` | May it contribute to or receive a response calculation? |
| `empirical_or_simulation_population` | May it enter a noise realization, surrogate, or simulation population? |
| `diagnostic_display` | May it be retained for review without implying stronger scientific use? |

These profile names are vocabulary, not policies. They do not transfer
ownership of PTC fitting, MAP support, NOI realizations, or another consumer
estimator to VAL. Each realized policy declares its scientific-use owner and
complete predicates. Package-specific versions may impose stricter declared
predicates while preserving the shared logical contract.

## Multiple Causes, Supports, And Restrictions

Adjacent producers accumulate their asserted causes without erasure and own
their local composite flags/supports. VAL does not AND or OR raw causes into a
producer-owned support. When an exact use-owner policy supplies multiple
applicable restrictions, permission is conjunctive and exclusion is
disjunctive: no permitting restriction rescues an occurrence excluded by
another. An explicit exception belongs to that same policy and preserves the
underlying causes. “Permitted for diagnostic display” is a disposition under
the diagnostic policy, not another cause.

## Required Nonretroactive Lifecycle

For each exact decision identity:

\[
  F_k \longrightarrow V_k \longrightarrow C_k \longrightarrow F_{k+1},
\]

where `F_k` is one immutable producer-fact set, `V_k` one immutable VAL
decision, `C_k` the consumer stage using it, and later facts form a new fact
set with a new VAL decision identity. A later decision cannot rewrite the
facts, applicability, disposition, or consumer action of an earlier decision.

## Missing And Unavailable Adjacent Facts

This profile supplies meanings, not numerical availability or a use-owner
policy. If a required RTC, CAL, PTC, or MAP-owned fact/support or supplied
policy element is missing, contradictory, out of domain, or not yet
authoritative, VAL applies the approved shared decision-domain and
missing-fact logic. It may not substitute a default, an identity response,
zero uncertainty, a clean cause assertion, a policy predicate, or a finite
payload.
