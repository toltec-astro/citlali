# SCI-CAL v0.1 Rationale r0.5 / Engineering r0.4 Consistency Report

Date: `2026-08-20`

Scope: implementation-blind consistency review after incorporation of the
scientific owner's Q01--Q09 decisions.

## Assessment

The two views are consistent. They share one included authority for notation,
definitions, assumptions, equations, requirements, and edge predictions. The
science view explains the model and achievable evidence; the engineering view
retains the exact normative IDs and observable conformance consequences.

The review specifically confirmed:

- identical dimensionless `xs` meaning, sign, Tune premise, and no-additive-CAL
  boundary;
- identical CAL-before-PTC ordering;
- identical SCI-BEAM source-factor and TolProj selection/rescale ownership;
- identical 272/214/150 GHz references, spectrum/color-correction boundary,
  passband limitations, and content-bound atmosphere operator;
- identical separation between whole-observation quality tolerance and
  sample-level numerical support;
- identical downstream noise boundary, within-array systematic scope, and
  explicitly unavailable uncertainty products;
- identical distinction between same-Beammap closure and independent
  associated-pointing transfer; and
- identical treatment of 1%, 5%, and 5--10% as reporting benchmarks, with
  achieved-performance acceptance reserved to the scientific owner.

No arbitrary validation matrix or minimum sample size appears in either view.
No implementation, executed validation, science qualification, or production
readiness is asserted.
