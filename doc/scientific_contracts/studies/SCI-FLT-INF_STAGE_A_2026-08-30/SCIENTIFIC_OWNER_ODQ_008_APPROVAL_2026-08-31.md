# SCI-FLT-INF-ODQ-008 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-008`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: approved and closed; template-defined amplitude units and exact
fixed-state response selected

## Selected output interpretation and units

The base-v0.1 signal product remains the ODQ-002 matched-filtered map. Its
value at each admitted location is the ODQ-001 amplitude estimate for the
exact ODQ-005 template-response product under the ODQ-006 normalized
estimator. The signal quantity is therefore **template amplitude**, not a
posterior sky reconstruction, ordinary convolved sky value, detected source,
selected peak, fitted-source quantity, or catalog entry.

For parent-map signal `m`, template response `t`, and declared template
amplitude `A`, the unit relation is

```text
unit(A_hat) = unit(A) = unit(m) / unit(t).
```

This template-amplitude unit is authoritative for the filtered signal. It is
not automatically the parent signal unit or `mJy/beam`. Equality with a
parent unit, flux-density unit, peak convention, integrated convention, or
other label exists only when the exact template scaling and compatible
CAL/BEAM lineage explicitly establish it.

The output preserves the applicable parent map-domain spatial structure and
semantics: WCS/frame, location indexing, array/band identity, observation or
coadd grouping, parent and contributor lineage, and inherited support,
validity, and calibration provenance. It does not automatically preserve the
parent signal quantity, nominal-beam interpretation, DC response, integrated
flux, surface brightness, extended-source fidelity, or calibration
covariance. Those properties require exact authority for the selected
template-amplitude convention and response.

## Exact fixed-state response

For each admitted output location `x`, define the fixed-state row functional

```text
L_x u = <t_x, Q_x u_x> / <t_x, Q_x t_x>,
```

using the exact realized ODQ-004 weighting object, ODQ-005 template,
ODQ-006 discrete conventions and conformant realization, and ODQ-007 complete
support. The exact response to any deterministic declared parent-domain
perturbation `delta m` is

```text
delta A_hat(x) = L_x delta m.
```

For the unit-amplitude declared template placed at location `y`, the exact
template-response matrix or operator is

```text
R_t(x,y) = L_x t_y
           = <t_x, Q_x t_y> / <t_x, Q_x t_x>.
```

At every admitted matching location, under the exact fixed state, support,
phase, boundary, and validity assumptions,

```text
R_t(y,y) = 1.
```

This is the ODQ-001 amplitude-unbiased matching-template response. Response
to any other declared mode `u` is `R_u(x)=L_x u`; its mode amplitude, units,
phase, domain, and support must be stated before a transfer claim is made.

The off-diagonal template response need not be symmetric, stationary,
isotropic, translation invariant, or representable by one convolution
kernel. Spatial variation in `Q_x`, WCS/grid geometry, subpixel phase,
support, validity, or boundary admission may make it position dependent. The
complete-support rule means response is scientifically defined only over
admitted input/output locations; an unavailable location is not a zero-
response measurement.

## Response representations and state distinction

A uniformly processed kernel is not a universal response for the
matched-filtered map. It may represent the exact response only on a declared
domain where translation invariance, identical weighting, identical complete
support and validity, identical centering/subpixel phase, the same boundary
convention, and the same normalization are established. Otherwise the exact
location-indexed operator above remains the response authority.

The selected response is conditional on one exact fixed realized state.
If `Q`, support, method selection, approximation state, or any other
consequential state is learned or re-estimated when the parent is perturbed,
the full-procedure response is a different object. Its learning graph,
perturbation family, state-change record, and response interpretation remain
for ODQ-010. A fixed-state response may not be relabeled as full-procedure
response, and a future full-procedure response does not replace the exact
fixed-state operator.

The persisted representation of the response—explicit matrix, response map,
kernel plus a proved invariance domain, executable operator identity, or
another exact encoding—and its required or conditional product role remain
for ODQ-013. Representation may not weaken the scientific response defined
here.

## Point-source, beam, and calibration consequences

When the exact template is the parent-bound point-source response, the output
is a matched point-source amplitude field and `R_t` is its matched point-
source response footprint. A literal calibrated point-source flux-density
interpretation is permitted only when the template amplitude convention and
the exact inherited CAL/BEAM lineage establish that quantity and unit. For a
different scientific template, the output remains the amplitude of that
specified shape and must not be labeled with point-source or beam terminology.

The parent's originating nominal-beam identity remains provenance; it is not
automatically the effective beam of the matched estimator. Any effective
matched-filter beam, area, or solid angle must be derived from the exact
`R_t` under a stated coordinate measure, domain, normalization, and validity
convention. It may not be inherited from the parent nominal beam or inferred
from a generic kernel label.

Parent-signal and template calibration dependence must be represented jointly.
No independence or cancellation of shared calibration factors may be assumed.
Where exact lineage proves cancellation or another dependence, that result
may be propagated; otherwise the calibration covariance contribution is
unavailable. Missing calibration covariance is not zero, is not supplied by
the estimator denominator `D`, and does not become precision. Numerical and
total uncertainty products remain governed by ODQ-009.

## Consequences

- `SCI-FLT-INF-ODQ-008` is approved and closed.
- The filtered signal has the exact template-amplitude quantity and unit
  defined by the declared template scaling.
- The exact fixed-state response is the location-indexed linear functional
  `L_x`, with unity response to the matching template at each admitted
  location.
- No stationary or universal response kernel is presumed.
- Parent spatial identity and provenance are retained, while signal, beam,
  transfer, and calibration meanings are retained only when explicitly
  established for the matched estimator.
- Fixed-state and full-procedure response remain distinct.
- ODQ-009 uncertainty and covariance products are the next owner gate.

## Nonclaims

This approval selects no ODQ-004 noise/covariance option, quantitative
ODQ-006 approximation envelope, numerical response representation, template
instance, beam model, effective solid angle, flux-conversion factor,
calibration-covariance value, uncertainty product, NOI generation graph,
public product bundle, VAL profile, implementation, conformity, validation,
performance, readiness, production, freeze, or Unity action. It changes no
SCI-FLT-FIXED or frozen SCI-NOI byte.
