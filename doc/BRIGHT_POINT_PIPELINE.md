# Bright-POINT downstream development reference

This is the recommended successor route for RTC/PTC evaluation. Its downstream
method is frozen at ordinary MAP + BASE-POINT v1. It is a bright-POINT development
reference, not an all-mode or production qualification. CORR, DIAG, JINC, CUT3,
shoulder refinements, seeds and extra cycles are parked. Historical drivers and
compact evidence remain for explicit reproduction only.

## Build and run

Use a durable checkout of `codex/fruit-pipeline-freeze-20260926` (or its final local
checkpoint tag recorded in the handoff). The supplemental local environment is
AppleClang/Homebrew with the existing cached dependency sources and
`$HOME/tolteca/bin/python`. This is separate from Unity/Spack qualification. The
recorded KIDs dependency identifies itself as `04088da-dirty`; no dependency was
modified in this task. No Unity access or input transfer was used.

```sh
cd /Users/gwilson/.codex/worktrees/fruit-pipeline-freeze/citlali-refactor
$HOME/tolteca/bin/python tools/timestream_successor/configure_point_local.py \
  --dependency-root /Users/gwilson/.codex/worktrees/4c31/citlali-refactor/build/_deps \
  --prefix /opt/homebrew
cmake --build build --target citlali_cli -j 8
```

The dependency root is explicit and must contain the cached sources listed in
`data/development/bright_point_base_v1/local-build-options.json`. No private
`/tmp` build or network download supplies the candidate. Other build environments
retain their existing owners; this helper does not establish their qualification.

Set roots for your environment. The example acquisition is exactly POINT
152430/0/2, tune 152430/0/1, a2000 networks 11/12. Its original inputs are enumerated
with hashes relative to `INPUT_ROOT` in `point_152430_inputs.json`. They are verified
against canonical APT raw/KMP associations and raw headers. Missing originals
cause preflight failure; recover only those exact manifest objects from the normal
input archive. Reporting never downloads or regenerates them.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/point-reference-mpl XDG_CACHE_HOME=/tmp/point-reference-cache
POINT_REPO=/Users/gwilson/.codex/worktrees/fruit-pipeline-freeze/citlali-refactor
INPUT_ROOT=$HOME/work_toltec/local_data
POINT_OUTPUT=$HOME/work_toltec/local_data/citlali-validation/my-bright-point-run
POINT_CONFIG=$POINT_REPO/data/development/bright_point_base_v1/reference.json
POINT_BINARY=$POINT_REPO/build/bin/citlali

$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" preflight \
  --config "$POINT_CONFIG" --input-root "$INPUT_ROOT" \
  --output "$POINT_OUTPUT" --binary "$POINT_BINARY"
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" feedback \
  --config "$POINT_CONFIG" --input-root "$INPUT_ROOT" \
  --output "$POINT_OUTPUT" --binary "$POINT_BINARY"
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" report \
  --output "$POINT_OUTPUT" > "$POINT_OUTPUT/saved-report.json"
```

`preflight` requires a fresh destination and starts no science. `feedback` accepts
that prepared destination, or performs the same preflight on a fresh destination.
For M0 alone use `forward` in place of `feedback`, with a separate fresh output.
Both call the same `point_processing.upstream`; do not run both to demonstrate M0
and feedback. A started execution cannot be implicitly resumed or retried. A
source rejection is a scientific outcome with a retained diagnostic map and empty
accepted candidate; a processing failure is separately recorded and returns a
nonzero exit. Read `EXECUTION.json` even after a zero exit.

The runner requires a clean committed checkout and its matching native executable.
After a source or documentation commit, rebuild to refresh its revision identity.
The frozen numerical regression source and later reporting commits are recorded
separately. Saved reporting needs only compact run products and works after moving
them; historical absolute bindings remain provenance, never automatic input paths.

## Flow and ownership

```text
Original paired detector acquisition + exact tune / canonical APT
Telescope / auxiliary + shared observation identity / native timing
                    |                          |
             paired ingress / ALIGN            AST
                    |                     native motion + coordinates
                   RTC <------------------ timing / matched output grid
                    |                          |
                   CAL <------------------- calibration / pointing
                    |
         immutable post-RTC/CAL PTC parent y + VAL / x,r context
                    |
          fresh per-network, per-segment PTC
                    |                          |
             ordinary MAP <--------------- same-sample AST projection
                    |
                  total M0 -> POINT-EXPECTED / BASE -> m0
                    |
           y - project(m_k) -> fresh PTC Learn/Consider/Apply
                    |
          restore exactly the subtracted native operand
                    |
             ordinary MAP -> total T_(k+1) -> replacement m_(k+1)
```

AST and VAL are coordinated side paths. Native occurrences, physical gaps,
paired x/r, eligibility, causes, calibrated units and support remain attached.
RTC-matched grids do not merge physical runs. x is the science stream; r retains
its defined diagnostic role through the parent. The parent is prepared once,
while PTC relearns on the current sky-subtracted residual for every processing
segment/network. Memory residency across the run is not observation-wide learning.
RTC is never reapplied to a filtered residual, and previous candidates are not
accumulated. The measured total map is separate from its admitted feedback map.

Scientific owners remain the typed ingress/ALIGN, RTC/FLT plan, CAL, AST, VAL,
PTC, MAP and FruitPointModel/Residual/Learned components. The CLI adapters bind
files/identities; `point_processing.py` sequences calls. POINT measurement and
admission live in `point_expected.py` / `point_guarded_feedback.py`;
`point_reference.py` owns the existing observation-local noise initialization and
conditional endpoint calculation using `point_additive.py`. No numerical algorithm
is imported from a dated validation/report directory. Historical wrappers redirect
to these shared owners.

FLT's existing array low-pass plan and support/response records remain. Local PTC
operator/control accounting is available; a complete realized upstream response
and NOI ensemble are unavailable and remain explicitly so. The empirical additive
companion is not an inverse-variance map weight, a realized NOI product, or a full
adaptive error law. No unavailable companion is replaced with fabricated zeros.

## Frozen downstream recipe and upstream controls

`reference.json` binds the selected observation/array/networks, original input
manifest, effective acquisition YAML, RTC plan, PTC request and resource limits.
For a new acquisition provide its own versioned input manifest and effective YAML;
never reuse learned M0 masks/noise/bases. The current saved MAP adapter supports
subobservation 0 / scan 2, fixed J2000 source/zero interface offsets and POINT only.
All three array selectors and their nominal beams remain callable (5, 7.5, 10
arcsec for a1100/a1400/a2000); this closeout's cold-start evidence covers a2000 only.
Other observing modes, arbitrary interface offsets and all-mode calibration are
not added or qualified.

- Ordinary MAP: 2 arcsec containing pixels, gamma=1, existing support and 0.5
  coverage cut. FK5/J2000; units mJy/nominal beam. Nominal calibration beams remain
  distinct from measured PSFs.
- POINT-EXPECTED: the existing seven-parameter Gaussian plus constant, W=I,
  existing optimizer/bounds; finite accepted fit, positive amplitude, strict
  fitted A/sigma_A >10, both principal sigmas >1.5 arcsec, centroid radius <=20
  arcsec. Preserve the implemented strictness at the boundary (see owner code).
- BASE: source-connected threshold 3 within 40 arcsec, measured brightness minus
  the current fitted constant, gain 1, complete replacement. No Gaussian feedback.
- M0 plus ten applications: T10 uses m9; m10 is saved but never applied. This is
  an iteration limit, not a convergence claim.
- M0 C0: existing four-FWHM source exclusion and 49-pixel covariance stamp; BASE
  amplitude scale uses three-FWHM exclusion. Relearn these for every observation
  or changed upstream configuration; freeze them and original geometry/support
  only within that trajectory. PTC bases/means are relearned every residual.
- Above-training coverage is informational, using the unchanged finite terminal
  bin continuation. Low coverage, real numerical defects, covariance/projection
  and geometric support still gate use.
- T10 gets one separate conditional endpoint covariance/sensitivity analysis,
  including the fitted constant. It does not feed back or change weights.

Upstream changes belong in the existing effective YAML / declared RTC array-plan
owner and `reference.json`'s `ptc` method/rank request, consumed by
`execute_connected_ptc`. The checked-in reference remains observed-entry ALS rank 10,
100 iterations, objective tolerance1e-5, rank tolerance1e-10, per network/segment.
The native feedback adapter reads solver settings from the current PTC receipt and
checks the configured rank; downstream code need not change for a future authorized
rank experiment. Pairwise-covariance comparison is an existing separate native
capability, not this feedback reference. No mode-count comparison ran here.

The existing native pre-PTC projection/injection controls, `FruitPointModel`,
`point_response_cd` compact/shoulder definitions and their retained tests remain
available for future PTC work. Historical C/D schemas remain explicit comparison
interfaces, outside the ordinary route. They bypass RTC/CAL. A pre-RTC injection
is deferred: it must define native measured x/r units, calibration/response,
physical occurrence/validity, and AST source projection before implementation.

## Products and storage

`M0/products/<array>/map.fits` retains the six MAP planes. `freeze/<array>` stores
this observation's guard, M0 C0/BASE numerical operands, masks and continuation
metadata. Each `run/BASE-POINT/REF/<array>/cycle-NN` stores a separate total map,
candidate/admission, fit/template/apertures, guard, accounting and learning receipts.
`SUMMARY.json` records all stages; `endpoint` contains conditional uncertainty
operators and scales. No seventh uncertainty plane replaces an original MAP plane.

The measured run is bounded by native-cell storage estimates and explicit free
space. Only one cycle runs at a time; native network workers are explicit and
numeric libraries use one thread. Per-cycle transient operands are removed after
receipts/measurements. Shared RTC/CAL/PTC arrays remain until the trajectory ends.
After compact evidence is durably saved, delete only the enumerated inactive
native payloads in that run. No automatic raw-input deletion, broad `git clean`,
checkpoint restoration or hidden recomputation is part of reporting.

See `validation/fruit_pipeline_freeze_2026-09-26` for the exact isolated regression,
limits and cleanup record; the retained reference is linked in place under
`validation/point_152430_no_upper_coverage_veto_2026-09-26/compact`, not duplicated.

## Bounded PTC rank experiment (2026-09-26)

The owner-authorized rank 5/10/15 comparison uses the same entry point and shared
processing owners, with explicit `citlali-point-ptc-rank-v1` identity. The reference
configuration above remains unchanged. On the experiment branch, with the same
thread/build environment and a fresh `RANK_OUTPUT` under local validation:

```sh
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" rank-preflight \
  --input-root "$INPUT_ROOT" --output "$RANK_OUTPUT"
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" rank-run \
  --output "$RANK_OUTPUT" --binary "$POINT_BINARY" --review "$RANK_REVIEW"
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" rank-report \
  --output "$RANK_OUTPUT"
```

`RANK_REVIEW` is the independent preflight JSON binding the exact clean source SHA
and `verdict: pass`. Reporting reads compact products, never restarts science.
Attempt counters forbid implicit retries. Rank10 REF and injection regression gates
precede dependent work. See [experiment record](../validation/ptc_rank_response_2026-09-26/PREFLIGHT.md).

The completed [rank assessment](../validation/ptc_rank_response_2026-09-26/ASSESSMENT.md)
and [handoff](../handoff/PTC_RANK_RESPONSE_NOISE_2026-09-26.md) keep rank10 as the reference.
