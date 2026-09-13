# Fixed-template versus free-shape readout: authorized saved-map experiment

2026-09-13 · r0.1 · SCI-FRUIT v0.1 development

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the exact [single-experiment proposal](../fruit_point_peak_failure_diagnosis_2026-09-13/NEXT_EXPERIMENT.md).
The owner answered **“Give this a go”** to that proposal. Decision identity:
`SCI-FRUIT-POINT-FIXED-TEMPLATE-READOUT-2026-09-13`. This authorizes its 48
linear diagnostic fits and no follow-on experiment. Starlet remains parked
for POINT, all registered outcomes remain unchanged, and the earlier numerical
image-stability qualification remains failed. This is outside scientific
contract authorship and production.

Read exactly the saved pass-6 H and D totals: two seeds (20260911, 20260912),
two arms (pixelwise P, nominal starlet C), three arrays. Use the original
supported radius-60 domain and its radius-52 inner probe, with the same finite
pixels as each retained free Gaussian-plus-plane fit. This gives 24 array maps
and 48 linear fits, with no alternative terminal selection.

For each map/domain, solve unweighted least squares for four unrestricted,
signed coefficients:

    map = amplitude * (saved_truth / declared_analytic_peak)
          + b0 + bx*x/90 + by*y/90 + residual.

The declared peak is 100 for H and 90 for D. Do not normalize by the sampled
pixel maximum. Shape, centroid and orientation are deliberately supplied by
injection truth to this diagnostic ruler. Neither fitted amplitude nor fitted
plane is fixed to truth. They never enter feedback or a usability rule.

Use NumPy least squares with its default numerical rank convention (`rcond=None`).
Retain coefficients, rank, singular values, condition number, residual norm,
normal-equation residual and runtime. Mark rank-deficient or nonfinite results
unavailable and retain their status; no alternate solver or substituted value.
Make exactly 48 registered calls. Verification uses residual/orthogonality and
stored-model arithmetic, never a second fit. No PTC or feedback code is imported.

The comparator is the already saved free Gaussian-plus-plane fit on each exact
domain. Re-evaluate its saved model only to check its recorded residual cost;
do not refit it. Retain original outer-plane background and original usability
labels separately. There is no template-derived availability policy.

Report absolute peak errors, all four same/crossed-seed H/D ratios, H and D
seed changes, domain sensitivity, central fitted-plane shifts and residual
costs for both readouts. Compare numerical errors against the existing 5%
budget descriptively, including all withheld cases. Crossed pairs are dependent
recombinations of two reused realizations, not additional independent trials.
Do not choose a preferred readout/domain using truth and claim qualification.

Interpretation is fixed by the proposal: persistence of crossed-noise errors
in the fixed template localizes a substantial component to source-aligned
content in the saved maps; improvement localizes sensitivity to the free-shape
readout. A mixed result remains mixed. Neither finding separates all residual
nuisance, transfer bias, morphology mismatch and source-fit numerical effects.
Do not infer a deployable template, detector policy, uncertainty, OOF suitability
or production readiness.

Hash-bind source and inputs before execution and preserve every prior result
manifest and opaque archive status. No pre-PTC input reads, new cleaning passes,
feedback fits, free-Gaussian refits, thresholds, reserved observations, Unity,
production changes, OOF campaign or bundled next experiment is authorized.
