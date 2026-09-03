# SCI-FLT-INF-ODQ-010 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-010`

Decision date: `2026-08-31`

Scientific owner: G. Wilson

Status: **Option 1 approved; ODQ-010 closed**

## Approved state and NOI generation graph

Base v0.1 uses one exact immutable realized application state. Each
scientifically consequential component is classified separately:

- the ODQ-005 template is declared fixed;
- an authoritatively supplied noise/covariance or spectral-weighting state is
  declared fixed;
- a state estimated from the exact real observation or coadd parent is learned
  once from that parent and frozen;
- parent-owned support and validity are fixed input facts, while ODQ-007
  complete-support admission is a deterministic application rule rather than
  filter-owned learning;
- regularization defining `Q` is immutable ODQ-004 state;
- approximation envelope and tolerances are declared before application;
- realized convergence/approximation facts are recorded application state;
- `D` is derived from the fixed template, weighting, and support rather than
  an independently learned state; and
- method choice is declared fixed.

When a PSD or `Q` is learned from the real parent, the exact graph is parent,
one learning generation, one immutable frozen-state artifact, science
application, and identical frozen-state application to every admitted NOI
member. Observation and coadd learning remain separate. Their state is not
reused or presumed equivalent.

The frozen-state response and uncertainty are conditional on the realized
state. They do not include PSD/state-estimation uncertainty or constitute
full-procedure response or covariance.

Base v0.1 excludes NOI-informed state updates and per-member relearning. Those
are separately versioned future methods requiring a complete rerun graph,
estimand, response, uncertainty population, lifecycle, and NOI contract.
Fixed-state and relearned members cannot mix.

The implementation-blind author shall develop matching options in both
contract views for materialized, exact structured, or lineage-resolvable
immutable-state representation. Representation cannot change the lifecycle or
scientific conditioning selected here. Owner disposition is required before
freeze where an option changes numerical availability or reconstructability.

This decision does not select an ODQ-004 noise model, authorize a numerical
route, define a relearned method, or launch Stage B.
