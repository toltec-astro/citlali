# SCI-RTC To SCI-AST Sample-Grid Boundary

Boundary identity: `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY v0.1/r0.1`

Status: owner-approved boundary authority; implementation conformity not
assessed

Prepared: `2026-08-23`

Scientific owner: Grant Wilson

Joint boundary owners: SCI-RTC and SCI-AST scientific owners

## Purpose And Authority

This package-neutral boundary defines how an exact SCI-RTC output-grid sample
is attached to the SCI-AST role
`SCI-AST:rtc_output_grid_coordinates@1`. It supplies the approved boundary
authority required for later re-audit of finding `F-006` without introducing
another signal, temporal, or astrometric operator.

It composes, without superseding:

- SCI-RTC v0.1/r0.12, especially Equation 11 and Requirements 028--029,
  037--041, 046, 048--052, 111, 114, and 117;
- SCI-AST v0.1/r0.3, especially the RTC parent in the canonical notation,
  Definitions of ALIGN-grid and RTC-output-grid coordinates, Equations for
  `theta^A` and `theta^RTC`, and Requirements 073--079; and
- the exact `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` occurrence/time/mapping boundary.

Similar array shape, cadence, field spelling, numerical coordinate equality,
or implementation behavior does not establish compatibility.

## Ownership

SCI-RTC owns the immutable RTC product, resolved plan, realized output grid,
stable output sample, phase-zero selection, selected-point time, representative
ALIGN occurrence, segmentation, decimation, temporal response, transitive
support, correction state, signal-state facts, and RTC provenance.

SCI-AST consumes those facts unchanged. It owns realization of detector sky
direction and every requested tangent, continuous-WCS, or pixel-coordinate
role on that exact RTC grid. AST does not select an RTC representative
occurrence, reconstruct an RTC plan, or apply temporal signal coefficients to
angular coordinates.

## Stable Identity And Coordinate Anchor

For observation `o`, detector `d`, and stable RTC output sample `n`, the v0.1
phase-zero relation is

```text
y^RTC[d,n] = v^preD[d,M n]
rho[d,n]   = (d,M n)
t_out[n]   = t[M n].
```

`M` is the exact segment-local decimation factor. `rho[d,n]` is the exact
representative occurrence on the admitted aligned parent. It is not a response
centroid, maximum-coefficient input, donor occurrence, support midpoint, or
synthetic sample.

AST attaches the RTC-grid coordinate to this exact selected occurrence and
selected-point time. Numerical equality with the corresponding ALIGN-grid
coordinate does not collapse the identities: `(o,s)` remains an ALIGN slot and
`(o,d,n)` remains an RTC output occurrence with its RTC product, plan, grid,
and realized-state parents.

## One Coordinate Grid, Separate Signal States

Conditioned `x` and requested conditioned `r` share one RTC-grid coordinate
bundle indexed by `(o,d,n)`. They share grid, cardinality, selected times,
phase-zero slots, representative occurrences, and the canonical pair-level
operator state.

The shared coordinate does not merge their numerical signal states. The
following remain separately typed for `x` and `r` where applicable:

- payload availability and numerical validity;
- modification, cause, and publication state;
- response and uncertainty availability; and
- downstream scientific-use disposition.

Unavailable `r` neither invalidates the coordinate nor otherwise valid `x`.
A valid coordinate does not validate either signal. A future differently
gridded `r` descendant requires a separately named product and coordinate
parent.

## Exact Boundary Bundle

The boundary instance for `(o,d,n)` binds the following as one immutable
relation. A compact or generative representation is permitted only when every
item remains exactly recoverable.

| Role | Required exact content |
| --- | --- |
| Product identity | Observation, detector occurrence, RTC product and version, application context, immutable resolved-plan identity, realized-record identity, generation, and parent aligned-pair identity. |
| Grid identity | Stable output sample `n`, output-grid identity, segment, input and output grids/rates, cardinality, decimation factor `M`, zero phase, selected-point rule, edge/output disposition, and compatibility state. |
| Coordinate anchor | Representative stable ALIGN slot `M n`, exact representative occurrence `rho[d,n]`, selected-point time `t_out[n]=t[M n]`, and the exact available ALIGN direction/tangent/continuous-pixel parent selected for extension. |
| Timing and phase | Phase, delay, and any separately authorized time-shift record with sign, units, value or typed unavailability, applicability, uncertainty status, parent, and application count. |
| Signal support | Complete RTC-local transitive support through replacement, filters, masks, state boundaries, edges, and phase-zero selection, with exact cause preservation. Optional native-acquisition support is separately identified and composes ALIGN exactly once. |
| Response authority | Exact realized RTC-local response authority for conditioned `x` and the requested conditioned-`r` state, including acting domain/codomain, support, phase/delay, boundary/state behavior, availability, and provenance. |
| Correction history | Every upstream or RTC-related coordinate/time correction identity, sign, units, parent, applicability, and application count, including an explicit zero count when not applied. |
| State and uncertainty | Coordinate-independent RTC signal validity/cause facts, coordinate-parent availability, response availability, uncertainty availability, status, exact reasons, and prohibited claims. |
| Provenance | Source package/revision bindings, plan and realized-state lineage, reconstruction method/version when generative, exceptions, compatibility/supersession state, and immutable parent links. |

## Response Authority And Recoverability

The boundary requires complete response semantics and recoverability, not
gratuitous dense serialization of the formal Jacobian

```text
K^RTC[dn,qj] = partial y^RTC[d,n] / partial y^A[q,j].
```

An exact composed operator, segment-specific kernels plus immutable
boundary/state records, a sparse or factored operator, a replay-complete plan
and realized state, or materialized blocks may satisfy the boundary. The
representation must allow the realized response and its complete support to
be recovered without undocumented defaults. If it cannot, response is
explicitly `response_unavailable`; it is never approximated by a partial
kernel labeled complete.

The coordinate anchor does not replace the full temporal support. A
response-aware consumer jointly retains the RTC operator and every
contributing ALIGN-grid coordinate. In general,

```text
sky_response(L[n,s], {theta^A[d,s]})
    != sum_s L[n,s] theta^A[d,s].
```

AST shall not linearly filter circular, spherical, or nonlinear angular
coordinates as though they were detector signal.

## Delay And Coordinate-Correction Rule

The assigned RTC-grid time remains the selected-point time `t[M n]`. Filter
phase or group delay is response information and is not an automatic pointing
time shift. A scalar delay may be inapplicable at response zeros, edges,
resets, masks, donor influence, state changes, detector mixing, or other
non-LTI support.

Any required time/pointing correction is a separately authorized adapter. It
binds exact sign, units, numerical value or operator, domain, support, parent
response, uncertainty state, and application count. AST applies no inferred
delay correction, and no admitted correction is applied twice. Unavailable
delay is not silently replaced by zero.

## Failure Semantics

If the required pointing/coordinate authority for the RTC grid cannot be
recovered, the observation reduction halts. No CAL, PTC, ordinary science, or
companion-ML handoff may be published from that attempt. Raw inputs remain
preserved, and typed failure diagnostics may be retained; existing facts are
not rewritten as false.

An explicit `response_unavailable` state is a complete response-status fact
and need not erase an otherwise recoverable coordinate, but it blocks every
response-dependent product or claim. An omitted response-status field is an
incomplete boundary. Local pointing loss may be handled only by a separately
authorized sample-rejection policy; this boundary supplies no such default.

## Compatibility And Supersession

Compatibility requires this exact boundary identity, compatible bound package
revisions, identical stable-identity semantics, the phase-zero representative
relation, complete required roles, and preservation of typed unavailable and
hard-failure behavior. A successor shall name this revision, map every changed
role, and state whether old instances remain readable and scientifically
equivalent.

This boundary does not assess implementation conformity, validate numerical
response recovery, authorize production use, or introduce MAP projection or
deposition semantics.
