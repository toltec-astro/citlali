# Prediction and evidence crosswalk

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Status: **sixteen proposed predictions and ten worked decision cases; no implementation or observational evidence performed**

## Predictions

| Prediction | Proposed statement | Required discriminating evidence |
| --- | --- | --- |
| UFC-PRED-001 | For every finite nonempty exact support and every member, raw and normalized coefficients equal exact one; all pairwise coefficient ratios and the arithmetic mean are exactly one. | Algebraic conformance bound to exact source, q, g, and support. |
| UFC-PRED-002 | A publication support containing exactly one occurrence is valid and produces one exact coefficient. | Singleton algebraic and publication conformance. |
| UFC-PRED-003 | Empty support produces no mean, coefficient realization, payload, or eligibility claim. | Empty-support structural and lifecycle conformance. |
| UFC-PRED-004 | Completed segment, scan, or bounded-chunk publication units can be handed off independently without an observation-wide barrier. Different actual units create new q,g realizations under the same family rule. | Publication completion evidence for each unit plus an independent handoff before unrelated units finish. |
| UFC-PRED-005 | Repetition from the same exact inputs is numerically identical, yet each q,g and evaluation retains immutable identity; unbound cross-generation substitution is forbidden. | Replay and immutable-identity conformance. |
| UFC-PRED-006 | When the named profile is explicitly requested, a selected constant override or different family reaches UQ-01 and is ineligible rather than silently ignored. | Named request with independently varied selection value. |
| UFC-PRED-007 | A false scalar, unit, integrity, or membership fact makes UQ-05 false even with unrelated unknown/conflict. The payload is producer-invalid, while the completed negative profile artifact is realized and ineligible. | Mixed-fact truth composition and four-axis lifecycle conformance. |
| UFC-PRED-008 | With no fidelity false, absent evidence or unexplained unreadability leaves UQ-05 unknown; same-fact contradiction remains conflict. The decision is unavailable without unity fallback. | Unknown-only and same-fact-conflict cases with retained causes. |
| UFC-PRED-009 | A stored compact R-hat_g exactly matched to target R_q establishes membership without a duplicated list. A wrong stored reference is payload false; missing/conflicting target or corrupt shared binding has shared structural scope; a wrong pair in one request remains request-local. | Independent target/reference match, mismatch, missing target, shared corruption, and request-local pairing cases. |
| UFC-PRED-010 | An invalid signal occurrence can still have a faithful coefficient one. It contributes only if independent producer and consumer gates admit the signal and occurrence. | Cross-profile evidence varying signal validity independently of coefficient fidelity. |
| UFC-PRED-011 | MAP consumption gives gamma_i=1 and removes only variation attributable to the PTC coefficient; other MAP factors and gates remain. | MAP boundary conformance with independent admission and numerical classification. |
| UFC-PRED-012 | JINC consumption gives w_ip=kappa_ip, including JINC's sign; final JINC weights need not be positive or uniform. | JINC boundary conformance over signed spatial factors. |
| UFC-PRED-013 | An authoritative MAP denial affects only MAP; missing or conflicting MAP permission is decision unavailable absent another false. Neither state determines JINC or payload fidelity. | Separate MAP/JINC permission and payload-fidelity evaluations. |
| UFC-PRED-014 | With no named-profile request there is no evaluation. With no selected family and no authorized default, neither consumer may infer this family from a stored one, a label, or another route. | Request-axis, missing-selection, and no-fallback conformance. |
| UFC-PRED-015 | On an otherwise valid exact MAP host route, a_pi=G_pi; missing lossless rational identity for that exact parent still leaves NOI design resolution unavailable. | Exact NOI rational-parent binding evidence. |
| UFC-PRED-016 | On the admitted first ordinary route, missing valid positive integer rank, noninteger/zero rank, or unrealizable positive rank prevents a completed q, coefficient, and downstream route and does not become PTC-disabled or another route. PTC-disabled separately produces no PTC coefficient. No identity or zero-rank route family is inferred. | Exact rank, group realizability, route identity, failure cause, and absence of fallback. |

## Worked mixed-evidence and scope cases

| Case | Proposed result | Required discriminating evidence |
| --- | --- | --- |
| UFC-MIX-001 | Known scalar two plus unknown membership makes UQ-05 F; the payload is producer-invalid, its handoffs are ineligible, and membership U is retained. | Independent scalar and membership states for the same payload. |
| UFC-MIX-002 | Known omitted target member plus unavailable integrity check makes UQ-05 F; integrity U is retained. | Exact target/enumeration comparison and separately absent integrity check. |
| UFC-MIX-003 | Contradictory authoritative scalar claims with all other facts true make scalar and UQ-05 C; eligibility is unavailable and neither claim is selected. | Two same-payload authoritative scalar claims and retained conflict. |
| UFC-MIX-004 | The scalar conflict plus independently repeated membership makes UQ-05 F; eligibility is ineligible and the scalar conflict remains recorded. | Same-fact scalar conflict and independent enumerated-member violation. |
| UFC-MIX-005 | Faithful payload plus complete MAP denial makes only MAP ineligible; no JINC result is inferred. | Complete named MAP permission record and separate JINC state. |
| UFC-MIX-006 | Faithful payload plus missing/unresolved MAP permission and no other false makes MAP decision unavailable, without denial. | Missing or unresolved permission evidence distinct from a complete denial. |
| UFC-MIX-007 | Malformed occurrence reference or wrong otherwise-authoritative q,g pair in one request affects only that request. | Valid shared objects plus one malformed request reference. |
| UFC-MIX-008 | Ambiguous authoritative R_q identity or corrupt/misbound shared q-to-g object prevents structural establishment for every reference; UQ-05 is not evaluated. | Shared-object identity/binding failure observed through multiple requests. |
| UFC-MIX-009 | Exact compact one plus stored R-hat_g independently matched to valid R_q makes membership T without a member list; wrong R-hat_g instead makes payload F. | Exact independent source/generation/scope/digest match and mismatch. |
| UFC-MIX-010 | An enumerated form with correct length but one repeated and one omitted identity makes membership F and the payload producer-invalid. | Exact identity multiset comparison rather than cardinality alone. |

Every future evidence record must bind the exact approved source, Registry record, application, q, g, support reference or enumeration, payload, request, consumer, evaluation, and evidence layer. The specification itself supplies no completed evidence.
