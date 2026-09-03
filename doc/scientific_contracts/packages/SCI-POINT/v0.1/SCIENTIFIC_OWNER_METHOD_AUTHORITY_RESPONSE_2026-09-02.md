# SCI-POINT v0.1 Scientific-Owner Method-Authority Response

Date: `2026-09-02`

Scientific owner: Grant Wilson

Status: binding Stage A disposition; no numerical method or Stage B launch
approved

## Decision

The `r0.1` packet does not contain sufficient scientific authority to select
or reconstruct the published legacy width convention, fitting objective,
fit-weight meaning/source, complete search/initialization/tie/fallback/solution
procedure, or marginal formal-error calculation. None may be inferred from
practice, implementation, configuration, optimizer defaults, Beammap
similarity, or reviewer preference.

Two scientifically and lifecycle-distinct authorities are required:

1. `POINT-COMPATIBILITY-METHOD v0.1` shall define the complete numerical
   point-estimation method for
   `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`.
2. `POINT-FORMAL-ERROR-METHOD v0.1` shall define marginal formal parameter
   errors and, where applicable, joint parameter covariance.

Both have current state
`unavailable_pending_separate_owner_approval`.

## Dependency Scope

While `POINT-COMPATIBILITY-METHOD` is unavailable, the numerical fit,
centroid/displacement, amplitude, widths, angle, residual, and fit-derived QC
quantities are unavailable. Generic ownership, boundaries, symbolic model,
roles, lifecycle, typed unavailability, downstream envelopes, and required
method-record contents may still be contracted.

When the compatibility method is approved but the formal-error method remains
unavailable, numerical fit values may be realized, but marginal formal errors,
joint covariance unless independently authorized, formal-amplitude
standardization, and named uses requiring formal uncertainty remain
unavailable. Their absence does not erase products that do not require them.

The amplitude/full-map-RMS diagnostic has its own exact RMS-population and
denominator authority and does not become available merely because a fit is
available.

## Lifecycle Repair

`diagnostic_only` is a named-use disposition, not a producer lifecycle state.
Producer realization, component identifiability, and named-use disposition are
separate typed axes. Independent per-array atomicity remains unchanged.

## Recovery And Stage B Firewall

Method recovery occurs in a separately authorized, quarantined effort. Its
output must be implementation-independent and distinguish established
compatibility behavior, scientific interpretation, solver representation,
historical non-authority, and unresolved owner choices. A Stage B author may
receive only sanitized, content-bound method records and exact dispositions.

An implementation-blind Stage B author may conditionally describe the generic
contract while either method remains unavailable, but it may not invent the
missing method or claim an affected numerical route. Exact repaired packet
bytes require owner approval and explicit fresh-task dispatch.

No implementation conformity, uncertainty coverage, pointing performance,
validation, readiness, production suitability, production authorization, or
Unity activity is established.
