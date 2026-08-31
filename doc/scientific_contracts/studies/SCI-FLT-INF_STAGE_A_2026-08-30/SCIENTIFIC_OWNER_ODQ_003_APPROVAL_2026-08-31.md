# SCI-FLT-INF-ODQ-003 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-003`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: approved; closes ODQ-003 only

## Approved parents and grouping

The matched-filter map package admits both of the following ordinary MAP
parents in v0.1:

1. one exact immutable normalized ordinary-MAP **observation bundle**; and
2. one exact immutable normalized ordinary-MAP **coadd bundle**.

They are distinct parent and grouping identities. Observation-map filtering is
observation-local: the exact observation bundle is the parent, and any
admitted state learning and application are bound to that observation
generation. Coadd-map filtering is coadd-local: the exact coadd bundle,
including its contributing-observation set and coadd generation, is the
parent, and any admitted state learning and application are bound to that
coadd generation.

The two cases may share a parameterized scientific method only where later
decisions establish identical applicable operator facts. Every realized
product must nevertheless preserve whether its parent is an observation map
or a coadd map, the exact parent digest, grouping, map identity, WCS/grid,
support/validity, response/covariance declarations, and lifecycle generation.

## Non-equivalence and order

No equivalence, commutation, or cross-observation combination rule is
approved. In particular, filtering an ordinary MAP coadd is not presumed
equivalent to filtering its contributing observation maps and then combining
the filtered results. No response, normalization, support, covariance,
uncertainty, state, or validity fact may be transferred between those ordered
graphs without separate authority.

This package performs no independent cross-observation combination. Any
coaddition already represented by an admitted coadd parent remains the exact
upstream MAP-owned operation and generation.

## Deferred parents

The following are not admitted in v0.1:

- JINC observation bundles;
- SCI-FLT-FIXED derivatives;
- other matched-filtered, inference-bearing, or derived map parents; and
- any implicit serialization or substitution for the two admitted ordinary
  MAP parent roles.

Those parent families remain deferred and unavailable until separately
recovered and owner-approved. This decision does not bypass the frozen PTC
coefficient or numerical `coverage_cut` gates that still make the ordinary MAP
numerical route unavailable.

## Consequences

- `SCI-FLT-INF-ODQ-003` is closed with two admitted ordinary MAP parent roles.
- Observation-local and coadd-local filtering are distinct realized method/
  product identities even when later mathematics is shared.
- The package publishes matched-filtered maps in both cases and retains the
  ODQ-002 exclusion of source-analysis behavior.
- JINC and derived-map parent routes remain deferred and unavailable.
- ODQ-004 is the next owner gate: exact noise/covariance model, weighting, and
  parent-coefficient authority for each admitted parent role.

## Nonclaims

This decision does not approve observation/coadd equivalence, a filtered-map
coadd operator, a final package name, author packet, Stage B launch, numerical
operator, noise/covariance object, parent coefficient meaning, template,
normalization, approximation, regularization, edge/support method, units,
response, uncertainty, NOI lifecycle, detailed product bundle,
implementation conformity, validation, performance, readiness, production,
freeze, or Unity action. It changes no SCI-FLT-FIXED or frozen SCI-NOI byte.
