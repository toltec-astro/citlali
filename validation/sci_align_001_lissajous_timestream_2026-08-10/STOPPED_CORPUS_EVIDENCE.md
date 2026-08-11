# Stopped corpus evidence

## Trigger

ObsNum 136280 was the ninth and final real observation opened. Its initial
500-realization exact whole-scan bootstrap had two qualifying KDE modes even
though the scalar checkpoint changes were small. The frozen revision-5 rule
therefore extended the same deterministic scan draws in increments of 250 to
the declared maximum of 1,500.

At 1,500 nominally successful realizations:

- median tau: `+11.7284374024 ms`;
- 68% interval: `[+3.2038861816, +21.3719257577] ms`;
- 95% interval: `[-9.2901185343, +27.9567837336] ms`;
- `P(tau < 0) = 0.1126666667`;
- qualifying KDE peaks: `2`;
- convergence status: `extend` at the maximum count;
- exact fits equal to the original optimizer start to machine precision:
  `361 / 1500`.

The full-data objective profile is smooth with one visible minimum. The
bootstrap histogram has a narrow pile-up at the original optimizer start plus
broad support on both sides. The successor audit in
`REAL_BOOTSTRAP_OPTIMIZER_FAILURE_02.md` proved that the 361-point spike is a
numerical artifact: these fits stopped abnormally at iteration zero but were
accepted because their objectives were finite. It is not evidence for two
physical timing states.

## Paired map comparison at the stop

The same deterministic whole-scan draws were used for the paired estimators.
The map side had already passed its declared convergence gate at 250 paired
realizations:

- point estimates: timestream `+11.7284374024 ms`, map coordinate-shift
  `+7.7903394636 ms`;
- point difference: `+3.9380979388 ms`;
- paired median difference: `+6.5152560602 ms`;
- paired 95% difference interval: `[-27.6135656899, +25.9146281100] ms`;
- covariance: `118.5475533380 ms^2`;
- correlation: `0.7081427428`.

The map estimate is not treated as uncertainty-free. The successor optimizer
audit invalidates the numerical interpretation of this paired interval,
covariance, and correlation because the affected timestream values enter the
paired arithmetic. These values are retained only as authenticated history.

## Disposition

The frozen stop condition is satisfied: "bootstrap remains multimodal or
fails convergence at its maximum count." No later observation was opened.
The remaining pointings are 148669, 148671, 150820, 151125, 151127, 151599,
151601, 151949, 151951, 152450, 152452, 152881, and 152883.

The eight other opened observations passed their observation-level gates, but
they are not a frozen representative subset and must not be promoted to an
all-corpus or high-S/N result. Their values are retained only to document work
completed before the stop.

That bounded audit is now complete and found a confirmed zero-iteration
single-start acceptance defect. After the repair passes every synthetic
regime and is checksum-refrozen, the smallest possible successor is a fresh
ObsNum 136280-only bootstrap in a new output root. It must not resume the
contaminated checkpoint, inspect an unopened observation, or tune an answer
toward the map estimate or the prior approximately 11-ms description.
