# SCI-FRUIT EL-F1 compact-relaxation execution result r0.1

Result: **invalid screen; no scientific candidate comparison is available**

The owner-authorized test was
`SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1`, bound by the exact
`EL_F1_BUNDLE_MANIFEST_R0.1.md` with SHA-256
`5d9f8c31a019f20fb969879c0e29187a5353111c2b53faab9cbc118eef411a2d`.

## What completed

The isolated prototype preserves the ordinary complete map as the measured
iteration product and gives non-unity alpha a separate feedback state. The
method is disabled by default and restricted to the owner-authorized pointing,
raw-observation, save-all, diagnostic development route. Focused tests passed
for:

- the bitwise-no-op alpha-one path;
- joint signal/kernel relaxation and newest-product weight/RMS carriage;
- fail-closed spatial WCS, grid, and finite-support identity;
- configuration scope and method identity; and
- complete experimental checkpoint round-trip and method/alpha mismatch
  rejection.

The alpha-one uninjected and injected trajectories both completed absolute
iterations 0 through 6. Their logs contain no unexpected error- or
critical-level message. The uninjected reduction loop took 279.045 seconds;
the injected run took 289.47 seconds wall and 287.02 seconds CPU. Reported
maximum resident footprints were 839,042,368 and 928,235,520 bytes. These are
technical execution observations only because the same-build candidate pair
was not completed.

The retained experimental tree, including the stopped attempts, occupies
960,096 KiB. The original source data and original reduction products were not
modified.

## Why the screen is invalid

The first alpha-1.25 control stopped before iteration-1 scan processing. Its
initial diagnostic combined route, grouping, WCS, and grid identity. A second
allowed attempt named the differing fields: the ordinary map loader retains
the two spatial WCS axes and deliberately zeroes or omits unused spectral and
Stokes entries, while the prototype had copied all four output-map entries.
The state was corrected to bind the exact two-dimensional sampling WCS; map
grouping and ordered planes continue to bind array identity. The focused tests
passed after that correction.

The second and final replacement then passed the spatial identity and every
per-pixel weight comparison but stopped on exact median-RMS identity. The
checkpoint held the in-memory values before FITS serialization, while the next
ordinary pass read decimal values from the FITS `MEDRMS` headers:

| Array | In-memory value | Reloaded FITS value | Relative difference |
| --- | ---: | ---: | ---: |
| a1100 | 15.426979746586087 | 15.4269797465861 | `+9.21e-16` |
| a1400 | 8.310331728331906 | 8.31033172833191 | `+4.28e-16` |
| a2000 | 10.878518198372534 | 10.8785181983725 | `-3.10e-15` |

These tiny differences are serialization evidence, not a measured scientific
effect. They nevertheless violate the prototype's exact state rule. The
approved allowance of two replacement attempts is exhausted, so no further
trajectory was run. In addition, diagnostic rebuilds after the completed
alpha-one pair mean a future candidate cannot use that pair as an exact
same-binary control. A valid retry must rerun all six primary trajectories
from a newly frozen executable.

No candidate map was opened for scientific ranking, no alpha-1.25 recurrence
was executed beyond construction of iteration 0, and alpha 1.50 was not run.
The result is therefore neither promising nor unpromising. It is the
predeclared **invalid** outcome.

## Narrow repair recommended for a successor decision

The ordinary complete map `Q_k` already persists and supplies the next pass's
weights and `MEDRMS`. The relaxed state `F_k` only needs to replace the signal
and kernel planes used as the accepted feedback model. A successor should
therefore define exact restart state as the composite of:

1. checkpoint-bound ordinary `Q_k`, including its reloaded weights and RMS;
2. separately identified relaxed `F_k` signal and kernel plus spatial route,
   grid, method, alpha, and iteration identity; and
3. the already checkpointed learned operational state.

That avoids claiming that pre-serialization and reloaded RMS values are the
same object, while still requiring the exact checkpoint-bound `Q_k` values to
drive the next selection step. If duplicate RMS storage is retained instead,
it must be populated from the serialized product that the next pass actually
reads.

Any retry requires a new owner-approved execution allowance, one frozen and
hashed executable before the first trajectory, and a fresh six-trajectory
same-build run. The scientific metrics and thresholds were not exercised and
need not change merely because of this technical stop.

## Repository verification

- focused EL-F1/config/restart C++ tests: 35 passed;
- complete CTest gate: 610 passed, zero failed, one pre-existing disabled;
- baseline and fruit-loop Python tools: 188 passed;
- focused prospective-analysis and injected-pair tools: 11 passed;
- full configuration preflight: 127 Python tests plus all mode-kit,
  compatibility, coverage, schema, authority, and boundary audits passed;
- `git diff --check` and Ruff checks: passed.
