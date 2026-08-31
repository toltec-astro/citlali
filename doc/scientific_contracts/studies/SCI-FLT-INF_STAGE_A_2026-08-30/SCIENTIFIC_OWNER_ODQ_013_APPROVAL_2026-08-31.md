# SCI-FLT-INF-ODQ-013 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-013`

Decision date: `2026-08-31`

Scientific owner: G. Wilson

Status: **Option 1 approved; ODQ-013 closed**

## Approved tiered atomic product policy

Base v0.1 uses a role-complete atomic signal bundle with independently atomic
conditional companions. Every realized signal bundle contains an exact record
of:

1. immutable parent identity and digest, including observation/coadd identity;
2. requested, effective, observation-resolved, resolved, applied, and realized
   method identities;
3. template, weighting/PSD, regularization, support, approximation, and frozen-
   state lineage;
4. the matched-filtered amplitude map, quantity, units, WCS/frame, grouping,
   calibration provenance, and FLT→FRUIT interface;
5. normalization `D`, its domain, and its nonprecision meaning through an
   authorized materialized, structured, or reconstructable representation;
6. fixed-state response identity and authorized representation, including
   matching-template response and response-derived beam availability;
7. complete support, validity, null, missing, edge, and cause state;
8. covariance/uncertainty identity and truthful availability state;
9. exact/approximate realization, regularization, convergence, and selection
   records; and
10. atomic completion, failure, publication, lineage, and provenance.

Numerical covariance, transformed-NOI uncertainty, calibration/nuisance
uncertainty, response-derived beam, and authorized projection products are
conditionally required only for the named product role or request that needs
them. Diagnostics may be optional but cannot replace a required scientific
role.

Atomicity is role-scoped. Failure of a required signal role yields no realized
signal product. Typed covariance unavailability does not fail a signal-only
product. Failure of a requested covariance-, NOI-, calibration-, or other
qualified companion fails that companion. A composite request is not realized
when a required companion fails; any separately retained signal carries only
its base signal identity.

Approved lifecycle states are `not_requested`, `requested`, `effective`,
`disabled`, `unavailable`, `resolved`, `applied`, `failed`, `realized`, and
`superseded`. Disabled and unavailable produce no scientific product. Applied
is intermediate until bundle completion. Supersession never mutates the prior
generation. A valid numerical amplitude of zero is ordinary realized data,
not a lifecycle or missing-data state.

FLT owns its named-use policies; SCI-VAL may register and evaluate them but
does not author FLT admission, validity, response, uncertainty, or publication
science. Required policy families cover parent/application admission,
output-location scientific validity, transformed-NOI parity/admission when
requested, and covariance-qualified named-consumer admission.

The implementation-blind author shall develop matching bounded options in both
contract views for response/covariance/state persistence, materialized versus
lineage-resolvable roles, and exact VAL-profile granularity. Owner disposition
is required before freeze where an option changes scientific availability,
consumer permission, or reconstructability.

This decision rejects both a universal monolithic signal-plus-uncertainty
bundle and best-effort independent-plane publication. It authorizes no
numerical route, conformity claim, Stage B launch, or scientific freeze.
