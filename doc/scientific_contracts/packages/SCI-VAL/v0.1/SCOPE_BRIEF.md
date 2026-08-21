# SCI-VAL — Sample And Detector Validity, Flags, And Map Eligibility Scope Brief

Status: Stage A r0.2 owner-approved scope; exact packet content binding in
progress; approved for fresh implementation-blind Stage B after binding

Scientific owner: Grant Wilson

Proposed version/date: `v0.1`, `2026-08-20`

Approved source identifier: `SCI-VAL-v0.1-StageB-packet-r0.2`; exact file
hashes are recorded in `AUTHOR_PACKET_MANIFEST.md`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[accepted pilot process](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager, `2026-08-20`; first external scope
  review absorbed in [`SCOPE_REVIEW_R0.1.md`](SCOPE_REVIEW_R0.1.md); ownership
  corrected and approved in
  [`SCOPE_OWNERSHIP_DECISION_R0.2.md`](SCOPE_OWNERSHIP_DECISION_R0.2.md)
- Existing material approved for adoption: the approved separation of
  producer causes, direct validity, operator support, consumer eligibility,
  and MAP final validity; the direct representative synthesis/replacement
  rule; and the PTC decision-stage distinction
- Existing material abstracted, deferred, or excluded: eleven historical VAL
  handoffs, current implementation and schemas, audit findings, repairs,
  tests, validation, Unity evidence, production status, and current Boolean
  encodings
- Genuinely new scientific work: define the exact typed fact/policy/evaluation
  interchange, producer-owned cause and local-support boundaries,
  use-owner-owned admission policy, four-axis decision semantics,
  conjunctive restriction behavior, fail-closed missing-fact rules, eight
  shared use-profile names, nonretroactive lifecycle, occurrence/detector
  aggregation interchange, exact versus conservative influence,
  uncertainty/response coupling, and falsifiable edge behavior
- Approved author references: this Scope Brief;
  [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md);
  [`AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`](AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md);
  and the owner-approved form of [`DECISION_LOG.md`](DECISION_LOG.md)
- Author-packet exclusions: every unlisted file and all implementation,
  handoff, audit, repair, test, validation, reduction, Unity, and production
  material

Confirm that this opening was reviewed before launching scientific authorship:
`yes — owner-approved r0.2 on 2026-08-20`.

## 1. Package Name And Scientific Purpose

**SCI-VAL — Sample and Detector Validity, Flags, and Map Eligibility** defines
the shared scientific interchange and deterministic evaluation rules by which
an exact producer-owned fact set and an exact use-owner-owned policy yield a
cause-preserving decision record.

**Authority limit:** SCI-VAL does not originate producer-local validity,
producer-local Boolean composites/supports, a named-use scientific policy, or
final product validity. A producer owns the facts and local supports it
publishes. The scientific-use owner owns the policy that interprets admitted
facts for that use. VAL may execute that exact supplied policy reproducibly
and preserve its result. “Map eligibility” therefore means the result of a
MAP-owned upstream-admission policy; it does not mean VAL owns MAP numerical
pixel support, estimator contribution, or final map validity.

The package exists because `finite`, `unflagged`, `positive weight`, `inside a
mask`, `computable`, `retained`, and `scientifically eligible` are not
synonyms. A producer may replace a sample for continuity, exclude it from a
fit, retain it for diagnostics, or carry transitive influence from an invalid
source. A consumer needs an explicit policy for the exact use rather than one
universal Boolean.

The organizing abstraction is:

\[
  (\text{producer-owned facts/supports},\ \text{typed causes},
   \ \text{influence},\ \text{named use},\ \text{use-owner policy})
  \longrightarrow
  (\text{eligibility disposition},\ \text{preserved reasons}).
\]

Every evaluated result is qualified as **eligible or ineligible under
use-owner policy (P) for use (U), evaluated with the VAL contract**. VAL is
the evaluation/interchange authority, not the scientific-policy owner. An
unqualified “valid” or “eligible” token is insufficient, and the consumer may
still reject an admitted occurrence when a separate numerical or
estimator-specific condition it owns fails.

The science-team rationale shall begin with one concrete occurrence: its
payload is finite and retained, but its exact representative source was
replaced. The rationale asks separately whether it may stabilize a continuity
operator, be displayed diagnostically, count as independent exposure, enter
an estimator fit, receive a frozen operator, or contribute to a map. This
example must make clear why one Boolean cannot answer all uses.

Stage B must derive this abstraction precisely and produce falsifiable rules.
The expression above is a scope statement, not a selected equation or truth
table.

## 2. Scientific Boundary

SCI-VAL begins only after upstream producers have supplied typed facts for an
exact occurrence or detector, including the applicable identity, origin,
direct validity, numerical state, availability, causes, local support,
transitive influence, response status, uncertainty availability, producer
decision stage, and parentage.

SCI-VAL checks and deterministically executes an exact versioned policy
supplied by the named-use owner. It ends with a cause-preserving evaluated
decision product containing four independent axes:

1. request state: `requested` or `not_requested`;
2. policy applicability: `applicable`, `inapplicable`, or
   `applicability_unknown`;
3. eligibility disposition: `eligible`, `ineligible`, or
   `decision_unavailable`; and
4. decision-product realization: `realized`, `incomplete`, `failed`, or
   `not_produced`.

`Not requested` does not manufacture a decision. `Inapplicable` is not
`decision_unavailable`, and a failed or unproduced decision product is not an
eligibility disposition. For every requested applicable decision, the output
also states:

- the exact occurrence or detector and product role;
- the named scientific use, policy owner, and policy identity/version;
- which required producer facts were present and applicable;
- the complete four-axis state and “eligible/ineligible under policy (P) for
  use (U)” meaning where realized;
- the direct causes, inherited influence, decision stage, and reasons;
- which producer-local support or named-use decision role is represented,
  without making VAL the owner of fit, application, retention, coefficient,
  response, empirical-noise, simulation, independent exposure, or another
  use policy; and
- the exact parent and lifecycle state.

The package does not alter the signal, repair data, synthesize missing facts,
choose an upstream operator, or determine a downstream estimator's numerical
support or final product validity.

## 3. Legitimate Inputs

The contract may admit only explicit, typed inputs appropriate to the named
scientific use:

1. **Occurrence identity:** observation, scan or coherent segment, exact
   sample/time occurrence, detector occurrence and stable detector identity,
   array/network/group, stream/product role, stage, parent, and lifecycle.
2. **Origin and transformation state:** original, synthesized, replaced,
   derived, recomputed, or unavailable origin, with every causal source link
   required by the producer contract.
3. **Producer-local validity:** acquisition/structural validity,
   detector-binding validity, numerical/domain validity, coordinate/time
   validity, response validity or availability, uncertainty availability, and
   producer completion state, without collapsing them into one bit.
4. **Typed causes and producer composites:** direct causes and their owning
   producer, decision stage,
   scope, severity/class where scientifically meaningful, and whether the
   cause is asserted, explicitly absent, inapplicable, or unknown. No cause
   record is not an explicit assertion that the cause is absent. Explicit
   absence is accepted only from the owning producer under a declared complete
   cause family. Any producer-local composite flag or support supplies its
   owner, exact inputs, Boolean/truth-domain rule, missing-state behavior,
   scope, use, and version. VAL does not reconstruct or reinterpret that
   composite from raw causes.
5. **Support and influence:** producer-local operator support, representative
   occurrence, full transitive causal influence or an explicitly conservative
   over-approximation, mask/control state, edge/context state, and exact
   support identity. Exact versus conservative and confirmed versus possible
   influence, the over-approximation rule, and false-positive consequence are
   distinct inputs.
6. **Producer decision stages:** for PTC-like consumers, distinctions such as
   fit-invalid, fit-excluded/application-available, post-fit output rejection,
   weight-only noncontribution, and advisory-only state. These names do not
   prescribe one representation.
7. **Named scientific use:** one of the shared vocabulary
   `independent_exposure`, `estimator_fit`, `operator_application`,
   `output_retention`, `analysis_or_gridding_contribution`,
   `response_companion`, `empirical_or_simulation_population`, or
   `diagnostic_display` profiles, or an owner-approved package-specific
   successor name. These names do not define or transfer policy ownership.
8. **Use-owner-supplied policy:** exact scientific owner, version,
   applicability domain, required authoritative producer facts/supports,
   decisive predicates, conjunctive applicable-restriction behavior, explicit
   overrides/exceptions, missing-fact behavior, aggregation rules, and failure
   scope for the named use. A producer severity or quality class remains
   diagnostic unless the use-owner policy supplies the threshold, comparison,
   domain, and uncertainty treatment that makes it decisive.
9. **Decision parent and lifecycle:** the immutable producer-fact-set identity,
   prior VAL decision if any, consumer stage, and successor fact-set identity.
10. **Aggregation specification, when requested:** exact occurrence
    population, observation/scan/segment/time support, per-axis counts,
    denominator, missing-decision treatment, aggregation operator and
    threshold, boundary polarity, propagation authority, advisory versus
    binding role, and learned/data-dependent uncertainty.

A bare flag, finite value, zero, positive coefficient, detector row, mask, or
product name is not a complete legitimate input.

## 4. Required Outputs

For every requested decision, SCI-VAL must define a logical output containing:

1. exact occurrence/detector and parent identity;
2. named use and policy owner/identity/version;
3. the independent request, applicability, eligibility, and realization axes;
4. all direct causes, explicit negative assertions, unknown cause states, and
   transitive influences material to the decision, without evidence-erasing
   short circuit;
5. the producer-local validity, support, response, uncertainty, and
   availability facts consumed;
6. decision-stage and action scope, without retroactively changing an earlier
   stage;
7. any aggregation from sample to detector, detector to sample population, or
   parent to descendant, including population/time support, all four-axis
   counts, denominator, missing semantics, operator/threshold, boundary
   polarity, propagation authority, and data-dependence/uncertainty;
8. exact failure or unavailability reason when required facts are absent,
   contradictory, ambiguous, or out of domain; and
9. requested/effective/observation-resolved/realized policy lineage;
10. exact versus conservative and confirmed versus possible influence state,
    support identity, approximation rule, and policy consequence; and
11. immutable fact-set, VAL-decision, consumer-stage, and successor-fact-set
    identities that prevent retroactive rewriting.

The base disposition rule shall distinguish structural decision-domain gates
from use-specific predicates:

- missing or contradictory identity, parent, policy, or applicability makes
  the decision `decision_unavailable`;
- after the domain is identified, any known decisive false predicate may
  establish `ineligible`, even if an unrelated non-gating fact is unknown;
- if no decisive predicate is false but a required predicate is unknown, the
  decision is `decision_unavailable`; and
- only all required true predicates establish `eligible`.

Stage B shall formalize a rule equivalent to

\[
D(P,U,F)=
\begin{cases}
\mathrm{decision\_unavailable}, &
  \text{identity, parent, policy, or applicability is missing or contradictory},\\
\mathrm{ineligible}, &
  \text{at least one known decisive admissibility predicate is false},\\
\mathrm{decision\_unavailable}, &
  \text{none is false and at least one required predicate is unknown},\\
\mathrm{eligible}, &
  \text{all required predicates are true}.
\end{cases}
\]

This eligibility rule is evaluated only on the applicability axis for a
requested decision; request and realization remain separate axes.

Every false and unknown fact remains in the reasons. For one exact use,
applicable restrictions compose conjunctively in permission, equivalently
disjunctively in exclusion: one permitting fact cannot rescue an occurrence
excluded by another applicable restriction. An override, supersession, or
exception is valid only when explicit in the same use-owner policy and cannot
delete the underlying causes or influence. A use-specific permission is a
disposition, not a cause.

The scientific contract defines these logical products. It does not select a
bit mask, enum, table, file, or in-memory class.

## 5. Upstream And Downstream Responsibilities

### Upstream producers

- **ALIGN/AST/TEL input** own time, coordinate, association, origin,
  synthesis/recomputation, detector binding, frame, and producer-local
  validity facts.
- **SCI-RTC** owns replacement, filters, masks as operator controls, edge and
  state behavior, selected representative occurrence, typed causes, local
  response, complete support, and transitive causal influence.
- **SCI-CAL** owns calibration factor/domain, atmosphere, detector join,
  response, uncertainty, and calibration-product validity.
- **SCI-PTC** owns its fit/application/output/coefficient supports,
  fit-invalid and post-fit decisions, removed-subspace response, coefficient
  roles, and transformed-product validity facts.

VAL consumes those facts and producer-local composite/support decisions; it
may not repair, strengthen, infer, recompute, or redefine them. Producers own
their atomic facts and Boolean composition. VAL owns no global cause ranking
and no use-owner policy. It preserves the order-independent, idempotent cause
set or graph and executes only the exact supplied predicates and composition
rules of the named-use owner.

### Downstream consumers

- **SCI-MAP** owns projection, contribution, normalization, numerical
  support, science-policy support, response, covariance, coaddition, and final
  raw map validity.
- **SCI-NOI** owns empirical noise-realization admission, covariance/scatter,
  and significance authority.
- **SCI-FLT** owns filter-local support and filtered-output validity while
  preserving raw-parent validity.
- **SCI-BEAM, SRC/MODE, and FRUIT** own their estimator-specific admission,
  fit, response, recurrence, and output-validity rules.

Consumers must supply or bind the exact policy they own for a named use. They
may not relabel `decision_unavailable` as eligible, erase causes, or promote an
invalid parent because a downstream value is finite. A successfully evaluated
upstream-admission result remains necessary but not sufficient for a separate
consumer-local numerical admission or final product validity decision.

## 6. Externally Imposed Conventions

- Sample-by-detector matrices use samples on rows and detectors on columns.
- Detector joins use stable UID/occurrence identity, not row position.
- Array, network, detector, observation, scan, sample, map, and lifecycle
  identities are distinct.
- Missing, disabled, automatic, unavailable, inapplicable, invalid,
  non-finite, rejected, and numeric zero are distinct states.
- Request, applicability, eligibility, and decision-product realization are
  independent axes. `Not requested`, `inapplicable`,
  `decision_unavailable`, `ineligible`, and `not_produced` cannot share one
  token.
- Requested, effective, observation-resolved, learned/resolved, and realized
  policy state flow one way.
- Invalid payloads are excluded before their numerical values are evaluated.
  Eligible non-finite required inputs fail at the declared scope or make the
  affected result unavailable.
- Operator masks and source-protection controls are not acquisition validity.
- A finite downstream product cannot promote an invalid or unavailable
  parent.
- Producer assertions and local Boolean composites retain producer authority.
  Cause preservation by VAL is order-independent and idempotent. No record is
  not an explicit negative assertion; absence requires the owning producer and
  a declared complete cause family.
- For one use-owner policy, applicable restrictions are conjunctive in
  permission/disjunctive in exclusion. One permission cannot rescue another
  applicable exclusion; explicit exceptions remain policy-owned and
  cause-preserving.
- Exact/confirmed influence and conservative/possible influence are distinct.
- Later facts create a new decision identity; they do not rewrite an earlier
  decision or the consumer action that used it.
- Current enabled polarimetry and numerical measured-R execution remain
  outside the active contract inventory.

## 7. Questions The Contract Must Answer

1. What are the exact logical domains for request state, applicability,
   producer facts, causes, knowledge state, named use, eligibility
   disposition, and decision-product realization?
2. How are multiple direct causes and transitive influences composed without
   erasing information or making one flag a universal action? Which Boolean
   composites remain producer-owned, and which supplied restrictions are
   evaluated for one named use?
3. Which facts are universally required for any scientific decision, and
   which are use-specific?
4. How do `ineligible` and `decision_unavailable` differ, especially when a
   known decisive predicate is false while an unrelated fact is unknown?
5. Which direct-origin states prohibit independent exposure while still
   permitting continuity, diagnostic, operator-state, or other declared use?
6. How does nonrepresentative transitive influence enter a consumer-specific
   policy without either disappearing or causing blanket invalidation?
7. How are fit-invalid, fit-excluded/application-available, post-fit output
   rejection, weight-only, and advisory states prevented from acting
   retroactively on the wrong stage?
8. How are sample-level and detector-level facts aggregated, and when does a
   detector decision apply to every occurrence? Which population, time
   support, denominator, missing rule, operator/threshold, polarity,
   propagation authority, and data-dependent uncertainty are required?
9. What must happen when identities, parentage, supports, responses, causes,
   or policy versions conflict?
10. How are invalid payload exclusion and eligible non-finite failure ordered
    so masked arithmetic cannot leak into a decision?
11. How are source masks, edge guards, replacement supports, fit supports,
    coefficient supports, map contribution, numerical support, and final
    product validity kept distinct?
12. What policy and lineage state is required to replay an exact eligibility
    decision?
13. How do response and uncertainty availability constrain eligibility for a
    claim without being treated as numeric zero?
14. Which fallbacks are scientifically admissible, and when must the result be
    unavailable rather than merely ineligible?
15. What analytic, truth-table, property, and end-to-end predictions would
    falsify the contract?
16. How are cause-order invariance, idempotent composition, determinism,
    nonretroactivity, fail-closed missing-fact behavior, and monotonicity under
    an explicitly declared stricter-profile relation defined and tested?
17. How do exact/confirmed influence and conservative/possible influence
    produce different allowed policy outcomes without being collapsed?
18. How does the immutable lifecycle
    (F_k\rightarrow V_k\rightarrow C_k\rightarrow F_{k+1}) create a new
    decision rather than retroactively changing (V_k)?
19. How does the contract prove the no-rescue property: for one exact use,
    adding an applicable exclusion cannot promote an ineligible occurrence,
    while an explicit use-owner exception remains visible and reproducible?

## 8. Non-Goals

SCI-VAL v0.1 does not:

- design or implement flags, bit layouts, enums, masks, schemas, classes, or
  file formats;
- inspect, audit, repair, test, validate, optimize, or authorize production
  code;
- replace samples, interpolate gaps, filter timestreams, calibrate signals,
  fit correlated modes, estimate noise, grid maps, filter maps, fit sources,
  infer beams, or run fruit-loop recurrence;
- choose package-local thresholds, detector-quality metrics, fit objectives,
  map support rules, or uncertainty estimators;
- define physical event timing or invent missing producer facts;
- make all influenced descendants universally invalid;
- combine raw causes into a producer-local flag or support owned by another
  package;
- originate PTC fit/application/output support, MAP contribution/support, or
  another scientific-use policy;
- establish one global cause-precedence order or treat silence as an explicit
  negative cause assertion;
- reinterpret positive weight, finiteness, mask membership, or numerical
  computability as eligibility;
- own MAP final validity or another consumer's output validity;
- promote a detector-level disposition to every occurrence without an
  explicit population, time support, aggregation, and propagation rule;
- authorize enabled polarimetry or measured-R scientific execution; or
- claim implementation conformity, representation fidelity, observational
  performance, validation, or production readiness.

## 9. Allowed References

The approved implementation-blind packet contains only:

1. this owner-approved Scope Brief;
2. the owner-approved
   [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md);
3. the owner-approved
   [`AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`](AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md);
4. the owner-approved [`DECISION_LOG.md`](DECISION_LOG.md); and
5. subsequent owner answers recorded through the manager, if any.

The exact files and SHA-256 values will be frozen in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md) after approval. No
source, current product schema, historical handoff, audit, repair, test,
validation record, Unity evidence, full adjacent-package draft, or production
status is allowed.

## 10. Owner Decisions And Stage B Direction

The owner-approved package decisions are recorded in
[`DECISION_LOG.md`](DECISION_LOG.md), and `VAL-OWNER-Q001--Q005` are resolved
in the owner ledger. The r0.2 decision narrows VAL to shared
interchange/evaluation authority: producers own their facts and local
composites/supports; each scientific-use owner supplies and owns its admission
policy; VAL executes without inventing either side. For one use, applicable
restrictions obey the no-rescue rule unless the same use-owner policy declares
an explicit cause-preserving exception.

No numerical threshold, producer Boolean expression, scientific-use
predicate, or implementation representation was approved.

The science-team rationale shall use this order:

1. why validity, flags, and eligibility differ;
2. producer facts/supports, use-owner policy, VAL evaluation, and consumer
   final admission;
3. applicability and the three eligibility dispositions;
4. one occurrence under several shared named-use profiles owned by different
   scientific consumers;
5. direct origin versus transitive influence;
6. decision stages and nonretroactivity;
7. sample-to-detector aggregation;
8. response- and uncertainty-dependent eligibility;
9. conflicting or missing facts and fail-closed behavior; and
10. replay, provenance, and validation.

The formal view shall derive truth tables and the properties of determinism,
cause-order invariance, idempotent cause preservation, conjunctive
applicable-restriction/no-rescue behavior, declared stricter-profile
monotonicity, nonretroactivity, and fail-closed missing-fact behavior. It shall
make policy ownership an input and shall not turn shared profile vocabulary
into VAL-owned science.

## 11. Independence Statement

This brief defines a scientific problem and ownership boundary without
prescribing current Citlali flag encodings, masks, branch order, file schema,
or implementation behavior as the answer. The author packet contains only the
approved brief, sanitized conventions/ownership, sanitized exact
RTC/CAL/PTC/MAP boundary profile, and approved decisions.
`INTERNAL_DOSSIER.md`, `PRIOR_WORK.md`, full adjacent drafts, source,
handoffs, audits, repairs, tests, validation evidence, and production status
remain outside the Stage B channel.
