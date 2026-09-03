# SCI-NOI v0.1 — Scientific-Owner ODQ-101 Approval

Decision identity: `SCI-NOI-ODQ-101`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

State: approved

## Exact Owner Direction

> For the normal noise-realization machinery, reduce the real observation
> first, freeze what Citlali learned, and generate the noise ensemble through
> that fixed reduction. If later we want Monte Carlo realizations that relearn
> some or all of the pipeline, define those explicitly as a different NOI
> method. Never combine the two kinds of realizations into one uncertainty
> estimate.

## Sanitized Scientific Disposition

Fixed-state conditional-sign GEN is the ordinary SCI-NOI v0.1 conditioning
class. Relearned GEN is a separate method class and remains numerically
unavailable until its complete exact rerun graph is owner-approved. Fixed-state
and relearned members, and relearned members with different rerun graphs, shall
never be pooled into one uncertainty estimate.

This decision selects a conditioning class, not a complete numerical method or
parent route. PTC-to-frozen-MAP, PTC-to-frozen-JINC, realized-MAP,
realized-JINC, and filtered routes remain separate method identities. None is
selected or made numerically available by ODQ-101.

## Compatibility And Conflict Review

The subsequent final Stage A owner-review directive expressly preserves this
disposition and clarifies the class-versus-route distinction. It does not
conflict with ODQ-101. Any future Scope Brief or decision that would make
relearned GEN ordinary, mix fixed and relearned members, or merge distinct
parent/insertion routes by numerical coincidence conflicts with this decision
and requires renewed owner discussion before editing.

This record establishes no implementation conformity, empirical calibration,
physical-noise validity, significance, performance, readiness, or production
authorization.
