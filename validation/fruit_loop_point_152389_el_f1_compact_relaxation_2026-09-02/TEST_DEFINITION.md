# FRUIT EL-F1 compact-relaxation feasibility screen

Status: **execution stopped; screen invalid before candidate comparison**

Test ID: `SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1`

## Bound question

Does fixed over-relaxation at `alpha=1.25` or `1.50` reach the same-build
`alpha=1.00` iteration-5 compact-source recovery by iteration 4 in all three
arrays without violating the predeclared shape, centroid, residual, support,
stability, or restart protections?

The scientific and execution contract is the exact owner-approved
`EL_F1_COMPACT_RELAXATION_OWNER_REVIEW_R0.1.md`; this file records its local
realization and may not revise its screen.

## Frozen inputs

- admitted root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`
- fresh source YAML SHA-256:
  `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909`
- inherited completed-pair control YAML SHA-256:
  `fa0ad45d269eed9248913a0e9e8e9231cd4481a69a6ffd2395808af30268c847`
- inherited completed-pair injected YAML SHA-256:
  `cf8899b0c9348c3a1b61fe1a00ee8aefdaa2422cecc90f63ad5eda19c921b007`
- method ID: `SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1`
- fixed alpha values: `1.00`, `1.25`, `1.50`
- injection: disabled/enabled pair, with `100 mJy/beam` in each array from
  absolute iteration 1
- iteration range: fresh absolute iteration 0 through absolute iteration 6
- execution: local sequential GRPPI policy and one configured thread
- optional raw/processed timestream output: disabled by `COMMON_LOCAL.yaml`

The six small variant YAML files in this directory differ only in their new
output root, fixed alpha, and injection switch. They are merged after the
unchanged fresh source YAML and common local overlay.

## Technical preflight

Focused tests must pass before opening the real development data. They cover:

- exact alpha-one no-op behavior for signal, kernel, and weight;
- common signal/kernel update and newest-product weight/RMS retention;
- fail-closed WCS/grid and finite-support mismatch handling;
- content-bound alpha and pointing/observation/raw scope validation; and
- complete EL-F1 checkpoint round-trip and method-mismatch rejection.

A read-only inspection of the already completed six-iteration pair found the
same `371 x 375` grid, WCS, and 139,125-pixel finite support in every array and
iteration. That is favorable preflight evidence, not a substitute for runtime
checks in the new trajectories.

## Stop conditions

The experiment stops on alpha-one numerical incompatibility, unavailable
required state, non-finite output, route/grid/support mismatch, checkpoint
incompatibility, unexpected error-level logging, or a resource-envelope
breach. An unfavorable scientific result is retained and is not rerun.

## Result

The alpha-one pair completed, but alpha 1.25 stopped before iteration-1 scan
processing. Two bounded replacements isolated first a non-spatial WCS-state
representation mismatch and then a bit-level pre-/post-FITS `MEDRMS`
serialization mismatch. The replacement allowance is exhausted, the
remaining trajectories were not run, and no scientific candidate comparison
is available. See `EXECUTION_RESULT_R0.1.md` and
`EXECUTION_ATTEMPTS_R0.1.csv`.
