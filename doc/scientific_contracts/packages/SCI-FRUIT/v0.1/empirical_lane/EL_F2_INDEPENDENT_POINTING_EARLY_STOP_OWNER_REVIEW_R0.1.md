# SCI-FRUIT v0.1 — Independent-Pointing Early-Stop Review r0.1

Decision candidate: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.1`

Status: **owner-review proposal; no run or code change is authorized**

## The short version

The first relaxation test taught us two things on pointing observation 152389:

- `alpha = 1.25` recovered the injected compact source faster than
  `alpha = 1.00`; and
- continuing `alpha = 1.25` through iteration 6 made the a1100 background
  residual worse than the allowed limit.

At iteration 5, before that degradation, `alpha = 1.25` had compact-source
recovery and residuals comparable to the `alpha = 1.00` iteration-6 result.
That was noticed after the first test, so it is only a reason to design this
new test. It is not evidence that the early-stop rule works.

The proposed test asks one simple, prospective question on a different
pointing observation:

> Does fixed `alpha = 1.25`, stopped after iteration 5, preserve the compact-
> source result of fixed `alpha = 1.00` through iteration 6 while saving one
> complete iteration?

## Why observation 123424

Observation 123424 is a Neptune Lissajous pointing from 2024-11-27. It was
chosen before inspecting any of its FRUIT outcomes: it is the first/lowest
observation number in the existing single-observation multiyear pointing
configuration collection. It is separated in date from observation 152389,
which was taken on 2026-02-19.

The required raw files, recomputed telescope file, and matched APT are now
available locally. Their exact paths, hashes, and identity checks are recorded
in `INPUT_INVENTORY_R0.1.md`.

The matched APT is a legacy ECSV, not an APT v2 run envelope. That is acceptable
for this development screen because the exact same, immutable file is used in
all four trajectories. Its calibration limitations are therefore shared by
both methods. This does limit the claim: the result cannot qualify the APT,
the recurrence, or a production method.

## Four primary runs

All four trajectories start from the original raw observation at iteration 0.
No checkpoint from 152389 is reused.

| Method | Injection | Saved iterations |
| --- | --- | --- |
| `alpha = 1.00` | off | 0 through 6 |
| `alpha = 1.00` | centered 100 mJy/beam in every array from iteration 1 | 0 through 6 |
| `alpha = 1.25` | off | 0 through 5 |
| `alpha = 1.25` | centered 100 mJy/beam in every array from iteration 1 | 0 through 5 |

The control/injected pair removes the real Neptune signal and isolates the
known injection. The same raw data, telescope data, APT, photometry, pointing
offsets, mapmaking, cleaning, learning, masks, weighting, filtering, thread
count, and output policy are used in every trajectory. Only `alpha`, the
injection switch, output path, and planned terminal iteration differ.

This test does not tune a stopping rule after looking at 123424. The terminal
iterations above are fixed before any new reduction product is opened.

## What counts as a useful result

The terminal `alpha = 1.25` iteration-5 injected-minus-control result is
compared directly with the terminal `alpha = 1.00` iteration-6 result. In all
three arrays, the candidate must satisfy the same compact-source protections
used in EL-F1:

- its absolute central-flux recovery error may be no more than one percentage
  point worse than the reference;
- fitted major and minor widths must each be within 3 percent of the processed
  injection kernel;
- centroid error must be no more than 0.1 arcsec;
- annular residual structure from 40 through 120 arcsec must be no more than
  10 percent worse than the reference; and
- full-map residual structure after subtracting the fitted kernel-shaped
  source must be no more than 10 percent worse than the reference.

Full-kernel flux recovery, response change, successive transfer-map change,
finite support, wall time, CPU time when available, peak memory, and retained
bytes are also recorded. Shape, WCS/grid, configuration, or finite-support
mismatch invalidates the affected comparison.

The performance target is at least 10 percent lower pair-mean wall time for
the two `alpha = 1.25` trajectories than for the two `alpha = 1.00`
trajectories. Scientific and performance results are reported separately.

The primary classification is one of:

- **invalid** — a required input, product, identity check, or trajectory is
  unavailable or a stop rule fires;
- **does not replicate** — any scientific protection fails;
- **scientifically replicates but misses the performance target** — all
  scientific protections pass but wall-time improvement is below 10 percent;
  or
- **promising early-stop result** — every scientific protection passes and
  pair-mean wall time improves by at least 10 percent.

An unfavorable result is retained and is not rerun. If the result is
promising, one exact restart replay of the injected `alpha = 1.25` trajectory
must restart from its iteration-2 checkpoint and reproduce iterations 3
through 5 bit-for-bit in signal, kernel, weight, and complete checkpoint state.

## Limits and stop rules

The execution is local and sequential with one configured thread. The source
inputs and all existing reduction products remain read-only. New products go
only under:

`/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1`

The bounds are:

- 4 primary trajectories and 26 primary iteration passes;
- at most 2 replacements for genuine environmental or interrupted-run
  failures, never for an unfavorable result;
- one 3-pass restart replay only after a promising primary result;
- no concurrent trajectories and exactly 1 configured thread per trajectory;
- 4 hours wall time and 64 GiB memory per trajectory;
- 20 hours aggregate wall time and 200 GiB total new retained output; and
- immediate stop for non-finite required products, route/grid/support or
  configuration mismatch, checkpoint incompatibility, unexpected error-level
  logging, or a breached resource limit.

The existing experimental recurrence is reused unchanged. Only the analysis
tool needs a bounded extension to compare trajectories having different
predeclared terminal iterations. Focused tests must prove that it rejects
missing/extra iterations and selects exactly reference iteration 6 and
candidate iteration 5 before any real-data run.

## What this cannot establish

This test concerns one centered compact-source injection in a pointing
observation. It cannot establish extended-source recovery, maximum recoverable
angular scale, atmospheric separation over a population, performance on faint
or negative signals, equivalence to historical Citlali, a general stopping
rule, method qualification, or a production default.

## Owner choices

### Choice A — Run the bounded comparison (recommended)

Approve `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.1` exactly as
bound by its manifest. This authorizes the small analysis-tool extension and
tests, one frozen executable, the four local primary trajectories, frozen
analysis, and the conditional exact-restart replay described above.

### Choice B — Use 123424 only as an ordinary repeat

Run both methods through iteration 6, repeating EL-F1 on the new observation
instead of testing the early-stop hypothesis.

### Choice C — Do not use the legacy APT

Pause this experiment until an APT v2 product is available for a suitable
independent pointing observation.

A general request to continue is not approval of Choice A. The decision ID and
exact bundle manifest must be approved.
