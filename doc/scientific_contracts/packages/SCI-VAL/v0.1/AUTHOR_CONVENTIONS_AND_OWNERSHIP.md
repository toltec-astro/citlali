# SCI-VAL v0.1 — Sanitized Conventions And Ownership

Status: owner-approved implementation-blind author reference for
`SCI-VAL-v0.1-StageB-packet-r0.2`

Date: `2026-08-20`

## Purpose And Sanitization

This file abstracts stable project conventions and owner-approved adjacent
boundaries needed by SCI-VAL. It intentionally contains no source path,
implementation behavior, finding, repair, test result, validation evidence,
production status, or numerical default.

## Stable Identity And Shape

- Primary detector timestreams are matrices with samples on rows and
  detectors on columns.
- Observation, scan/coherent segment, sample/time occurrence, detector
  occurrence, stable detector UID, array, network/group, stream/product role,
  stage, parent, and lifecycle are distinct identities.
- A dense row, column, detector index, map index, or file order is not a stable
  external identity.
- Cross-product detector joins require the declared UID/occurrence relation.
- Actual timestamps and gaps retain authority over nominal cadence.

## Stable State Distinctions

- Missing, disabled, automatic, unavailable, inapplicable, invalid,
  non-finite, rejected, and numeric zero are distinct.
- A flag describes a cause. It is not automatically a mask, invalidity,
  weight, support decision, or universal eligibility action.
- Acquisition support, direct validity, numerical validity, coordinate/time
  validity, origin/synthesis, replacement, operator mask, operator support,
  transitive influence, response status, uncertainty availability, consumer
  eligibility, and provenance are distinct facts.
- Requested, effective, observation-resolved, learned/resolved, and realized
  state flow one way.
- Unknown or unavailable uncertainty and response are not zero.
- Request state, policy applicability, eligibility disposition, and
  decision-product realization are independent. `Not requested`,
  `inapplicable`, `decision_unavailable`, `ineligible`, and `not_produced`
  are not aliases.
- No cause record is not an explicit negative assertion. Explicit absence is
  valid only when the owning producer asserts it under a declared complete
  cause family.
- Each producer owns the truth and Boolean composition of its own facts and
  producer-local supports. A producer-local composite declares its owner,
  exact inputs, truth-domain/Boolean rule, missing-state behavior, scope, use,
  and version. VAL does not reconstruct that composite from raw causes.

## Approved Composition Boundaries

1. **Direct representative origin.** An occurrence whose exact representative
   source is synthesized by ALIGN or replaced by RTC is not an independent
   detector exposure. It may remain usable for an explicitly declared
   continuity, diagnostic, operator-state, or other non-independent role.
2. **Nonrepresentative influence.** Causal influence from synthesis,
   replacement, filtering, masking boundaries, or other producer operations
   remains traceable. It is not automatically converted into universal
   downstream ineligibility.
3. **PTC decision stage.** `fit_invalid`, `postfit_output_reject`, and
   `weight_only` are distinct. Only a change to fit support requires refit or
   fitted-state invalidation. Fit exclusion and application availability may
   also be distinct when the selected PTC family defines that behavior.
4. **MAP fact hierarchy.** Upstream eligibility, estimator contribution,
   geometric incidence, exposure, normalization support, science-policy
   support, and final raw MAP validity are distinct. MAP owns its estimator
   contribution, support predicates, response, covariance, coaddition, and
   final output validity.
5. **Invalid and non-finite payloads.** A producer- or policy-declared invalid
   occurrence is excluded before its payload is numerically evaluated. A
   required occurrence that remains eligible but has a non-finite numerical
   payload causes failure or unavailability at the declared scope. Finiteness
   alone does not establish eligibility.
6. **Immutable parent validity.** A downstream operator may establish its own
   local validity, but a finite or locally valid descendant cannot rewrite or
   promote an invalid parent.
7. **Mask boundary.** A source mask, edge guard, fit mask, or other processing
   control is an operator input, not acquisition validity or confidence.
8. **Cause preservation.** Direct causes and inherited influence accumulate as
   an order-independent, idempotent set or graph. Producers own their
   assertions and local composites. A supplied use-owner policy selects
   decisive predicates and the disposition but never deletes other causes.
9. **Influence precision.** Exact/confirmed influence and
   conservative/possible influence are distinct facts with support and
   approximation-rule identity. Each named policy states whether possible
   influence rejects, permits, requests review, or makes the decision
   unavailable.
10. **Nonretroactivity.** One immutable producer-fact set produces one
    immutable VAL decision used by one consumer stage. Later producer facts
    create a new fact-set and decision identity; they do not rewrite the
    earlier decision.
11. **No-rescue restriction logic.** For one exact use, applicable
    restrictions compose conjunctively in permission, equivalently
    disjunctively in exclusion. One permission cannot rescue an occurrence
    excluded by another applicable restriction. An override, supersession, or
    exception is valid only when explicit in the same use-owner policy and it
    preserves the underlying causes. A permission is a disposition, not a
    cause.

## Producer–VAL–Consumer Ownership

### Producers own facts

- ALIGN/AST/TEL producers own time, coordinate, association, origin,
  synthesis/recomputation, detector binding, frame, and producer-local
  validity.
- RTC owns replacement, temporal conditioning, operator masks, edge/state
  behavior, representative occurrence, direct causes, causal influence,
  support, response, and RTC-local uncertainty availability.
- CAL owns calibration factor/domain, atmosphere, detector binding, response,
  uncertainty, and calibration-product validity.
- PTC owns fit/application/output/coefficient supports, staged decisions,
  transformed-product state, coefficient roles, covariance availability, and
  PTC response.

### VAL owns shared interchange and evaluation semantics

VAL consumes typed producer facts/supports and deterministically executes an
exact owner-approved policy supplied by the named-use owner. It owns shared
types, four-axis and knowledge-state logic, immutable identity/provenance,
cause preservation, deterministic evaluation mechanics, and preservation of
the reasons used. It owns no producer Boolean composite, global cause ranking,
or scientific-use policy. VAL does not change producer facts, reinterpret a
producer-local support, or manufacture missing authority.

### Consumers own their estimators and products

PTC, MAP, NOI, FLT, BEAM, SRC/MODE, and FRUIT supply and own the policy for
each scientific use they own, including their estimator supports, numerical
thresholds, responses, covariances, recurrences, and final product validity.
They may reuse VAL evaluation mechanics but may not erase causes or promote an
unavailable/invalid parent.

## Required Logical Output Distinction

The scientific decision product has four independent axes:

- request: `requested` or `not_requested`;
- applicability: `applicable`, `inapplicable`, or
  `applicability_unknown`;
- eligibility: `eligible`, `ineligible`, or `decision_unavailable`; and
- realization: `realized`, `incomplete`, `failed`, or `not_produced`.

Every realized applicability/eligibility result is qualified by exact policy
and use. The eligibility axis means:

- **eligible:** all facts required by the named policy/use are present and the
  occurrence or detector is admitted for that exact use;
- **ineligible:** the required facts are known and a declared cause or rule
  excludes the occurrence or detector from that use; and
- **decision unavailable:** the decision cannot be scientifically made because a
  required fact, identity, policy, parent, influence description, response,
  or availability state is missing, contradictory, or out of domain.

After identity, parent, policy, and applicability identify the decision
domain, a known decisive false predicate may establish `ineligible` despite an
unrelated non-gating unknown. If no decisive predicate is false and a required
predicate is unknown, the result is `decision_unavailable`. All false and
unknown facts remain in the cause/reason record.

## Shared V0.1 Named-Use Vocabulary

The mandatory structural interchange gate is paired with these shared
vocabulary names:

- `independent_exposure`;
- `estimator_fit`;
- `operator_application`;
- `output_retention`;
- `analysis_or_gridding_contribution`;
- `response_companion`;
- `empirical_or_simulation_population`; and
- `diagnostic_display`.

Profile identity supplies no scientific predicate by itself and does not
transfer ownership of the producer composite, consumer estimator, numerical
support, threshold, response, covariance, or final validity to VAL. Every
realized policy declares its scientific-use owner.

## Aggregation And Propagation

Every occurrence-to-detector or detector-to-occurrence decision states the
scientific owner of the aggregation plus the exact population,
observation/scan/segment/time support, four-axis counts, denominator,
missing-decision treatment, all/any/fraction/threshold/quantile or other
operator, threshold and boundary polarity, advisory or binding role,
propagation authority, and learned/data-dependent uncertainty. VAL may carry
or execute this supplied rule but does not originate it. An empty or unknown
denominator is unavailable or inapplicable under policy, never a valid zero
fraction. No detector state automatically applies to every occurrence.

The Stage B author must derive the precise formal model and edge behavior.
This document does not select an implementation representation.

## Capability Limits

- Enabled polarimetry has no accepted current execution contract.
- Numerical auxiliary measured-R scientific execution is deferred.
- No package may infer a stronger eligibility, response, uncertainty,
  significance, or production claim from the existence of a finite value.
