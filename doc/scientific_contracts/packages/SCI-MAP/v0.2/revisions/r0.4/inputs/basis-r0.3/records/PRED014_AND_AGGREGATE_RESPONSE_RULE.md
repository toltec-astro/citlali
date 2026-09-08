# SCI-MAP v0.2/r0.3 PRED-014 and aggregate-response rule

Status: **CANDIDATE review extraction**. The exact normative prediction is in
`../src/common/edge_cases.tex`; the aggregate policy is in
`../profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.3_CANDIDATE.md`.

## Amended `SCI-MAP-PRED-014`

> A required companion incompatibility rejects the complete observation
> before coadd mutation. Under `SCI-MAP:base_coadd@1`, unavailable or
> basis-incompatible response does not change signal membership; it yields
> unavailable coadd response with exact cause and no hidden subset or zero
> companion. Under a selected stronger role, an incompatible required response
> or covariance blocks that role for an existing base parent; any separately
> requested admission may reject the observation only before constructing its
> own explicitly identified base/signal parent. Other required identity, unit,
> WCS, shape, policy, and parent incompatibilities remain atomic at their
> declared role scope.

## Discriminating cases

| Case | Base-coadd membership | Response/covariance result | Required behavior |
| --- | --- | --- | --- |
| Member response unavailable; base coadd selected | Unchanged | Coadd response unavailable with exact cause | Admit the complete base signal bundle; do not drop the member, fill a response with zero, or infer identity. |
| Member response basis incompatible; base coadd selected | Unchanged | Coadd response unavailable with exact incompatibility | Preserve exact signal membership and disclose the limitation. |
| Named member response incompatible; response-bearing coadd requested over an existing base parent | Identical to the existing parent | Stronger role blocked/unavailable | Do not mutate or subset the parent. Every admitted member must carry the same exact response family `R`. |
| Named covariance block absent; covariance-qualified coadd requested over an existing base parent | Identical to the existing parent | Stronger role blocked/unavailable; block remains unknown | Do not treat the missing block as zero or independence. |
| Separately requested stronger-role admission detects a required companion incompatibility | No parent exists yet for that request | Observation rejected before mutation | If a base/signal parent is subsequently realized for this request, it has its own explicit identity and exactly the same ordered membership as the stronger product. |
| Response family absent but not named by selected role | Unchanged | No block from that unrelated absence | Family separation is preserved; whole-chain absence does not block a satisfied fixed-state role, for example. |
| Same-family responses but coadd state may change under perturbation | Unchanged for the base coadd | Fixed-state composition does not apply | Require a separately authorized coadd-procedure family, exact composition theorem, and its own `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE` record. |

The aggregate profile performs admission only. MAP owns arithmetic. VAL may
register and evaluate the MAP-authored policy only after exact owner
acceptance, canonical Registry/source binding, and activation; it does not
aggregate, place, or define it. The current profile semantics are candidate,
approval/registration/source binding are pending, and evaluation and an
object-specific decision are unavailable.
