# SCI-CAL-001 opacity-model decision amendment — 2026-07-31

## Authority and scope

The project owner, acting as calibration and atmosphere scientist, supplied
this bounded specialization of `CAL-D001` on 2026-07-31. It is additive to
the approved
[`SCI-CAL-001` coordinator and scientific-owner decision](SCI-CAL-001_COORDINATOR_DECISION_2026-07-31.md)
at commit `e8bd929008140e2ea8b44bfdc80b0a531b488765`. All other `CAL-D001`--
`CAL-D005` decisions, restrictions, and closure gates remain unchanged.

This decision specifies the previously open low-opacity operator and a
mandatory pre-implementation boundary check. It does not approve the assessed
implementation, authorize production, waive `SCI-ALIGN-001`, request Unity
evidence, or launch a re-audit.

## CAL-D001-OPACITY-001 — Low-opacity transmission

For each TolTEC band `b`, let `tau_q25` be the exact `tau225` reference value
at the first nonzero atmospheric-model boundary on the selected repair-base
SHA. Let

```text
T_q25,b(X) = transmission from the existing am_q25 model
             at tau225 = tau_q25 and sample airmass X.
```

For `0 <= tau225 <= tau_q25`, the approved transmission is

```text
T_b(tau225, X) = exp[(tau225 / tau_q25) * log(T_q25,b(X))]
               = T_q25,b(X) ** (tau225 / tau_q25).
```

This is geometric interpolation in transmission, equivalently linear
interpolation in line-of-sight optical depth, between unity at zero opacity
and the existing `am_q25` transmission at its exact reference `tau225`.

The implementation must:

- apply the full sample airmass under the already approved top-of-atmosphere
  pivot `X_ref = 0`;
- preserve exact endpoint equality, with `T_b(0, X) = 1` and
  `T_b(tau_q25, X) = T_q25,b(X)`;
- require the reference opacity, interpolation state, airmass, anchor
  transmission, derived transmission, logarithm, and applied correction to be
  finite and in their approved positive domains; and
- fail closed for negative or non-finite opacity, invalid airmass/elevation,
  nonpositive or non-finite transmission, invalid logarithm, missing model
  support, or opacity outside the approved model domain.

The exact `tau_q25` value and `am_q25` coefficient/model provenance must be
retained at full precision in the realized calibration record. Rounded values
from the audit narrative are not implementation authority.

## Mandatory q-model continuity preflight

Before modifying application code, the repairer must evaluate the existing
model on the proposed exact repair-base SHA at every q-model selection
boundary. The evidence must use the exact source coefficients and thresholds,
cover every TolTEC band, and cover the declared valid airmass domain at its
boundaries and representative interior values. For each boundary it must
report the selected models, both one-sided transmission limits, both
line-of-sight optical depths, and absolute and relative jumps without rounding
away a difference.

The new low-opacity definition makes the zero and q25 endpoints part of the
approved repair. At every boundary strictly above `tau_q25`, any analytic
mismatch or numerical jump exceeding documented floating-point evaluation
roundoff is a stop condition. The repairer must make no application-code
change, must preserve the preflight evidence, and must return to the project
owner for an explicit scope decision. The repairer may proceed with the
bounded implementation only if all above-q25 boundaries are continuous under
that criterion.

The preflight does not silently broaden the repair to alter q25, q50, q75,
q95, or any other existing atmospheric model. A discontinuity above q25
requires a successor owner decision that names the affected models and repair
scope.

