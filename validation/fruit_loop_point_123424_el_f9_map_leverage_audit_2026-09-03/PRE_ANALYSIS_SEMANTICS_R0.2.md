# FRUIT EL-F9 pre-analysis weight-semantics resolution r0.2

Status: **registered correction before new numerical measurement**

`TEST_DEFINITION.md` used the generic terms `W_all` and `W_without` and
required source/provenance confirmation of their map-weight meaning before
subtraction. The required source/schema inspection found two published weight
planes:

- `weight_formal_I` is the finalized additive mapmaking normalization
  coefficient copied before noise-product rescaling; and
- `weight_I` is the active output coefficient and is empirically rescaled
  when configured noise-weight calibration is applied.

The exact EL-F9 identities therefore use **`weight_formal_I` only**.
`weight_I` may be reported as a noise-product diagnostic but must not be
subtracted to infer UID 4460's mapmaking contribution.

The normalization code also applies a support threshold. The reconstruction
of `M4460` and `C4460` is admitted only where both paired formal weights are
finite and positive and where their positive difference exceeds the declared
roundoff floor. Pixels with positive `W_all` but zero `W_without` are retained
as a separate support-loss category; no missing retained-map numerator is
invented for them. Negative differences beyond roundoff stop the leverage
interpretation.

The source and schema checks made before this correction did not inspect any
new EL-F9 pixel value, distribution, correlation, or result. All other
questions, measurements, availability rules, bounds, and claim limits in
`TEST_DEFINITION.md` remain unchanged.
