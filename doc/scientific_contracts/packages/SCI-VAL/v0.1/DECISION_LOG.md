# SCI-VAL v0.1 — Scientific Decision Log

Status: Stage A r0.2 package-scope decisions owner-approved for
implementation-blind Stage B derivation

Scientific owner: Grant Wilson

Date: `2026-08-20`

## Owner-Approved Package-Scope Decisions

| Decision | Status | Disposition |
| --- | --- | --- |
| `VAL-SCOPE-D001` | approved `2026-08-20` | SCI-VAL owns shared fact/policy interchange types, knowledge-state logic, immutable identity/provenance, cause preservation, and deterministic evaluation mechanics. It may execute an exact supplied policy but does not originate producer facts, producer-local supports, or a scientific-use policy. |
| `VAL-SCOPE-D002` | approved `2026-08-20` | Every realized result identifies exact policy owner, policy version, and use. VAL never emits an unqualified scientifically-valid token. MAP eligibility means only the result of a MAP-owned upstream-admission policy, not pixel support, estimator contribution, or final map validity. |
| `VAL-SCOPE-D003` | approved `2026-08-20` | Each producer owns its atomic causes, explicit negative assertions, and producer-local Boolean composites/supports. Direct causes and inherited influence accumulate as an order-independent, idempotent set or graph without erasure. No record is not explicit absence; a negative assertion requires the owning producer and a declared complete cause family. |
| `VAL-SCOPE-D004` | approved `2026-08-20` | V0.1 defines a mandatory structural interchange gate plus the shared vocabulary `independent_exposure`, `estimator_fit`, `operator_application`, `output_retention`, `analysis_or_gridding_contribution`, `response_companion`, `empirical_or_simulation_population`, and `diagnostic_display`. Each realized profile policy is owned by the scientific-use owner, not VAL. |
| `VAL-SCOPE-D005` | approved `2026-08-20` | Request state, policy applicability, eligibility disposition, and decision-product realization are independent. `Not requested`, `inapplicable`, `decision_unavailable`, `ineligible`, and `not_produced` cannot share one token. |
| `VAL-SCOPE-D006` | approved `2026-08-20` | Every occurrence-to-detector or detector-to-occurrence rule declares its owner, population, observation/scan/segment/time support, four-axis counts, denominator, missing semantics, operator/threshold, boundary polarity, propagation authority, advisory/binding role, and learned/data-dependent uncertainty. No detector state automatically applies to all occurrences. |
| `VAL-SCOPE-D007` | approved `2026-08-20` | Exact/confirmed influence and conservative/possible influence remain distinct, with producer, support, and approximation-rule identity. The named-use owner declares whether possible influence rejects, permits, requests review, or makes the decision unavailable. |
| `VAL-SCOPE-D008` | approved `2026-08-20` | The producer owns the exact representative synthesis/replacement fact. Under the shared `independent_exposure` invariant that fact is disqualifying; it is not automatically disqualifying for continuity, frozen operator application, diagnostic retention, or another use-owner policy. Nonrepresentative influence remains visible and use-owner governed. |
| `VAL-SCOPE-D009` | approved `2026-08-20` | Invalid payload exclusion precedes payload evaluation. An admitted non-finite required payload fails or makes the affected result unavailable; finiteness never establishes admission. Operator masks, positive weights, producer-local support, retention, contribution, and final validity remain distinct. |
| `VAL-SCOPE-D010` | approved `2026-08-20` | One immutable producer-fact set `F_k` and one immutable use-owner policy produce one immutable evaluated decision `V_k` used by consumer stage `C_k`; later facts form `F_{k+1}` and a new decision identity. Post-stage facts cannot rewrite an earlier decision. Only a PTC-owned fit-support change invokes a new PTC fit/invalidation. |
| `VAL-SCOPE-D011` | approved `2026-08-20` | After identity, parent, policy owner/version, and applicability establish the decision domain, a known decisive false predicate may establish `ineligible` despite an unrelated non-gating unknown. If no decisive predicate is false and a required predicate is unknown, the result is `decision_unavailable`. All false and unknown reasons remain preserved. |
| `VAL-SCOPE-D012` | approved `2026-08-20` | Every decision binds exact occurrence/detector, immutable fact-set parent, use, policy owner/version, four-axis state, causes, influence precision, lifecycle, reasons, and aggregation if any. Representation remains engineering-owned. |
| `VAL-SCOPE-D013` | approved `2026-08-20` | PTC owns PTC fit/application/output/coefficient support policies. MAP retains contribution, normalization support, science-policy support, response/covariance, coaddition, and final raw map validity. VAL supplies shared evaluation mechanics and an immutable evaluated record only. |
| `VAL-SCOPE-D014` | approved `2026-08-20` | Required identity, parentage, policy/applicability, cause/influence, response, or availability conflict fails closed at the declared scope before consumer scientific-state mutation. Empty or unknown aggregation denominator is unavailable or inapplicable under the owner policy, never a valid zero fraction. |
| `VAL-SCOPE-D015` | approved `2026-08-20` | Stage B may derive the shared logical algebra, four-axis truth tables, supplied-policy evaluation, influence precision, nonretroactive lifecycle, aggregation interchange, uncertainty/response gating, and falsifiable properties, but may not choose producer composites, consumer scientific predicates, numerical estimators, thresholds, replacements, filters, mapmakers, encodings, schemas, or production policy. |
| `VAL-SCOPE-D016` | approved `2026-08-20` | For one exact use, applicable restrictions compose conjunctively in permission, equivalently disjunctively in exclusion: one permitting fact cannot rescue an occurrence excluded by another applicable restriction. Any override, supersession, or exception belongs explicitly to the same use-owner policy and preserves the underlying causes. A use-specific permission is a disposition, not a cause. |

## Inherited Approved Scientific Facts

These inputs are approved for adoption without re-derivation:

| Source boundary | Preserved fact |
| --- | --- |
| SCI-RTC owner-approved scope | Direct representative synthesis/replacement is not independent exposure; noncenter influence remains cause-preserving and consumer-policy dependent. |
| SCI-PTC owner decision D002 and approved Stage A decisions | Fit-invalid, post-fit output rejection, and weight-only are distinct; only a fit-support change invokes refit/invalidation. |
| Accepted SCI-MAP F010/ADR boundary | Upstream eligibility, contribution, exposure, two support predicates, and final raw validity are distinct; downstream finiteness cannot promote raw-invalid state. |
| Program conventions | Missing, disabled, unavailable, invalid, non-finite, and zero are distinct; state flows requested to realized in one direction. |

These inherited facts and their bounded use above were approved by the
scientific owner for the Stage B packet on `2026-08-20`.

## Decision Discipline

Stage A approval authorizes only implementation-blind contract derivation. It
does not freeze SCI-VAL, modify adjacent packages, validate an implementation,
authorize repair, or expand production use.
