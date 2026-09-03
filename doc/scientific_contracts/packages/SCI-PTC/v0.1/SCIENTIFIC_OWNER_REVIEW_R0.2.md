# SCI-PTC v0.1 -- Scientific Owner Review of r0.2

Status: accepted bounded revision directive for document revision `r0.3`

Date received: `2026-08-20`

Exact supplied review SHA-256:

`bd4aa11330b477f628b62118c13f2274d625dff2a873c13f025f9d932d01aac8`

## Review Judgment

The review finds that r0.2 resolves the central frozen-subspace projection
ambiguity, has the correct scientist-facing architecture, and is close to a
freeze candidate. It explicitly rejects another broad rewrite and directs one
bounded r0.3 pass before rationale freeze consideration.

## Bounded r0.3 Disposition

| Review item | r0.3 disposition |
| --- | --- |
| Distinguish the correlated component from its estimate | Adopt. Use latent `U_*`, fitted `Uhat`, and realized removed subspace `Uhat_space` consistently in the shared authority and rationale. |
| Clarify what the CAL-grid response contains | Adopt. Rename the response `K_up->CAL` and define it as the complete admitted upstream response carried by the CAL product, including every applicable admitted parent operator rather than only the CAL-owned multiplier. |
| Add a compact scientific validation program | Adopt in the rationale as study families, variations, and observables without thresholds, results, or validation claims. |
| Explain estimator-family scientific motivations | Adopt as concise guidance, without creating a methods catalogue or default-family policy. |
| Define application availability for fit-excluded occurrences | Adopt normatively. Add one definition, assumption, requirement, and prediction; preserve all existing IDs and append only new IDs. |
| Explain shifted/null surrogate purpose | Adopt as explanatory rationale text; preserve the existing surrogate contract. |
| Align lifecycle terminology with RTC | Adopt `Learn, select, resolve, and apply`; retain `freeze` as the resolved-model property. |
| Move the orphaned decision-register heading | Adopt as the only specifically requested layout correction in this content-focused revision. |
| Check the companion engineering specification mechanically | Adopt. Extend the durable verifier for the r0.3 obligations and exact identifier coverage; do not perform an implementation audit. |

## Effort And Scope Decision

High effort is sufficient. The review introduces no new algorithm family or
unresolved owner choice. Most companion-formal obligations already exist in
r0.2; r0.3 strengthens notation, response naming, and application-availability
semantics and adds scientist-facing validation guidance. Ultra effort was not
requested because no substantive ambiguity remained after comparison with the
shared formal authority.

Final cosmetic pagination and layout polish remain deferred to the final
editorial revision except for the named orphan-heading correction. No
implementation conformity, validation, achieved performance, scientific
freeze, or production-readiness claim follows from this review or revision.
