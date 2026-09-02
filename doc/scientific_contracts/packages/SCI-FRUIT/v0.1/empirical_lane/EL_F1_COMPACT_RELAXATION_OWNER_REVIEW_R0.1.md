# SCI-FRUIT v0.1 — First Recurrence Feasibility Review r0.1

Decision candidate: `SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1`

Status: **owner-review proposal; no prototype or run is authorized**

## The Short Version

The proposed first test asks a deliberately small question:

> Can FRUIT recover the known compact-source flux in fewer iterations if its
> map-to-map update is made moderately more aggressive, without damaging the
> recovered source shape, position, background, or numerical stability?

The test uses only the already exposed development copy of pointing 152389 and
the existing centered 100 mJy/beam injection. It is a screen for whether one
new-recurrence idea deserves broader development. It is not a test of extended
emission and cannot qualify a method.

## One Point Of Pushback

The executable preserved with the downloaded pointing reduction reports
Citlali `v4.0.0-3753-gc31a60a0`. It is a valuable refactor-era artifact, but it
is not the exact historical executable corresponding to the recovered
historical source commit `f70701ad...`. The latter still lacks a reproducibly
bound executable and dependency environment.

Therefore this first test must use an **exact same-build compatibility
control**, not something labeled historical Citlali. That control is the
ordinary complete-product recurrence with relaxation factor `alpha = 1` in
the same prototype binary. A successful result would justify continued
development only. Gate D and any historical-superiority claim remain blocked
until the exact historical control is recovered or the owner explicitly
changes that accepted requirement.

## The Proposed Recurrence Idea

An ordinary pass first creates the complete next map candidate
`Q_tilde_(k+1)` using the established subtraction, residual processing,
model-bypass, restoration, weighting, and mapmaking order. The feedback state
for the following pass is then updated by

\[
F_{k+1}=F_k+\alpha\left(Q^{\mathstrut\sim}_{k+1}-F_k\right).
\]

For this first test:

- `alpha = 1.00` is the compatibility control and must reproduce the existing
  recurrence exactly;
- `alpha = 1.25` is the first candidate; and
- `alpha = 1.50` is the second candidate.

`alpha` is fixed for an entire trajectory. No automatic tuning is allowed.
The values are intentionally modest probes of over-relaxation, not proposed
scientific defaults.

The complete map `Q_tilde_(k+1)` remains the iteration's measured output. The
relaxed `F_(k+1)` is a separately identified feedback state used by the next
iteration; it is not an independently calibrated sky product.

## Technical Closure Before Real Data

The formula is executable only after the prototype gives every field an
honest meaning. Before a pointing run, focused synthetic tests must establish
all of the following:

1. Signal and kernel/response planes are updated together on an exactly common
   route, grouping, WCS, grid, units, normalization, and finite support.
2. The first tranche permits no remapping and no support mismatch. Either
   condition makes that trajectory unavailable rather than invoking a hidden
   fill or fallback.
3. The newest complete product supplies the weight/RMS fields used by the
   next selection step. Those fields describe the newest measurement, not the
   uncertainty of the relaxed state; no uncertainty claim is allowed.
4. Selection, subtraction, residual-only processing, model bypass, restoration,
   and response propagation retain the recovered historical operator order.
5. The parent is the observation/raw JINC map on the existing array grouping.
   RTC, PTC, weighting, masks, detector penalties, learning cadence, source
   selection, filtering, and stopping cap are identical across all three
   `alpha` values. Learned operational state carries forward normally; it is
   neither blended nor reset by this proposal.
6. The method identity, `alpha`, feedback-state identity, and every causal
   field needed for exact continuation are checkpointed.
7. `alpha = 1.00` is bitwise identical to the unmodified recurrence in focused
   tests. If it is not, the experiment stops before real data are run.

These rules make the proposal an intentional new recurrence rather than an
unsupported claim of mathematical equivalence.

## Data And Runs

The only admitted data root is:

`/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`

The source data and original reduction products remain read-only. New configs,
checkpoints, logs, and products go under a new child directory of
`fruit-injection-development`.

The new configs inherit the exact settings of the completed six-iteration
pair whose control and injected config SHA-256 values are respectively
`fa0ad45d269eed9248913a0e9e8e9231cd4481a69a6ffd2395808af30268c847`
and
`cf8899b0c9348c3a1b61fe1a00ee8aefdaa2422cecc90f63ad5eda19c921b007`.
Only the new output roots, fresh iteration-0 start, fixed recurrence identity
and `alpha`, and executable/provenance identities may differ. Existing local
single-thread execution and I/O suppression are retained for direct
comparison.

Each of the three fixed `alpha` values gets a pair of trajectories:

- injection disabled; and
- the same centered 100 mJy/beam source enabled from absolute iteration 1.

Every trajectory starts at absolute iteration 0 and saves through absolute
iteration 6. Running from iteration 0 avoids treating a checkpoint created
under a different method identity as an exact continuation. The six
trajectories run sequentially on the local machine. Unity is out of scope.

The bounded execution envelope is:

- at most 6 primary trajectories and 42 iteration passes;
- at most 2 replacements for environmental or interrupted-run failures, never
  for an unfavorable scientific outcome;
- exactly 1 local CPU thread per trajectory and no concurrent trajectories;
- 4 hours wall time per trajectory and 24 hours aggregate wall time;
- 64 GiB memory per trajectory;
- 250 GiB total new retained output; and
- automatic stop on non-finite products, route/grid/support mismatch,
  checkpoint incompatibility, unexpected error-level logging, or a breached
  resource limit.

The implementation and runs remain quarantined development work. They may not
change production defaults or accepted configuration kits.

## What Will Be Measured

At every iteration and for every array, form the injected-minus-uninjected map
and measure:

- central and full-kernel recovered flux fraction;
- fitted major/minor width relative to the same-iteration processed kernel;
- centroid error;
- residual structure after subtracting the best kernel-shaped source;
- response-map change from the preceding iteration;
- common valid support and any unavailable product; and
- background/noise behavior in the existing 40--120 arcsec annulus.

For performance, record complete wall time, CPU time when available, peak
resident memory, retained bytes, and iterations and elapsed time to the
declared compact-source target. Performance remains separate from scientific
quality.

## Prospective Development Screen

This is a feasibility screen, not a qualification threshold. The outcome is
classified before any candidate map is opened:

- **invalid** if `alpha = 1` fails exact compatibility, any required state or
  output is unavailable, restart replay is inexact, or a stop condition fires;
- **not promising on this compact case** if neither candidate reaches the
  same-build control's iteration-5 kernel-normalized central recovery by
  iteration 4 in all three arrays; or
- **promising enough for broader development** if at least one candidate does
  so, while at iteration 6 its absolute central-recovery error is no more than
  one percentage point worse than the control in every array, both fitted
  widths are within 3 percent of the processed kernel, centroid error is at
  most 0.1 arcsec, and the annular and kernel-residual structure metrics are no
  more than 10 percent worse than the control.

One exact restart split is also required for every scientifically promising
candidate. It must reproduce three subsequent iterations bit-for-bit in signal,
kernel, weight, and checkpoint state.

The screen may reject a candidate family. Passing it does not select an
operational method, stopping rule, or value of `alpha`.

## What This Cannot Answer

Even a clean positive result cannot establish:

- superiority or non-inferiority to exact historical Citlali;
- response before the injection point in the pipeline;
- recovery of faint, off-center, negative, crowded, or extended emission;
- atmospheric or other nuisance leakage over a representative population;
- a production stopping rule or uncertainty model;
- a qualified method, Stage B input, or production default.

The next scientific step after a positive screen would be a Gate-D packet with
an exact historical control, a larger development population, and compact plus
extended-signal experiments. A negative screen would retire these two fixed
over-relaxation candidates without ruling out other recurrence families.

## Owner Choices

### Choice A — Approve The Bounded Feasibility Screen (Recommended)

Approve `SCI-FRUIT-EL-F1-COMPACT-RELAXATION-R0.1` exactly as bound by its
manifest. This authorizes only the isolated prototype, focused tests, six local
development trajectories, one exact-restart check per promising candidate,
and the frozen analysis above.

### Choice B — Recover The Historical Executable First

Do not prototype a candidate yet. Continue only with artifact or pinned-build
work needed to create an exact historical scientific control.

### Choice C — Revise The First Test

Return a new proposal with different candidate values, data, metrics, bounds,
or recurrence family. No work in Choice A is authorized.

Silence, a general request to continue, or approval of the surrounding
analysis is not approval of Choice A. The exact decision identifier and
manifest must be approved.
