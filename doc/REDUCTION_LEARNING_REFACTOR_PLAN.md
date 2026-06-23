# Reduction Learning Refactor Plan

This note tracks the phased refactor for source-aware reduction learning across
pointing, holography, science, and beammap-style reductions. Edit this document
as implementation and test reductions change the plan.

## Status

- Phase 1: complete.
- Phase 2: complete.
- Phase 3: complete, including the phase-boundary fix that prevents apply
  iterations from adding new learned records.
- Phase 4: implemented; pending reduction-test review.
- Phase 5: implemented; pending reduction-test review.
- Phase 6: implemented as diagnostic-only map-pixel outlier reporting;
  per-sample contributor tracing is opt-in because it is too expensive for
  normal jinc mapmaking.
- Phase 7: implemented for the shared source-mask interface used by RTC/PTC and
  mapmaking diagnostics; current active mode is center-radius protection.
- Phase 8: implemented for current conservative defaults and pointing-test
  configuration coverage.
- Phase 9: in progress. Local build verification passes; reduction-test
  verification is pending the next Unity run.
- Follow-up cleanup: learning now consumes per-scan RTC/PTC summary snapshots
  rather than live shared diagnostic maps. This fixes nondeterministic learned
  sample-mask summaries seen in otherwise identical pointing reductions and
  captures high-weight detector summaries after the selected final weight pass.

## Phase 1: Shared Learning State

Add a shared reduction-learning state, probably in/near PTCProc, with records for:

- learned spike/sample masks
- detector/scan penalties
- high-weight detector warnings/caps
- map-pixel outlier contributors
- source-protection metadata
- iteration phase: learn, learn_with_model, apply

No behavior change yet except writing diagnostics.

## Phase 2: Iteration Phase Wiring

Use the existing fruitloops iteration boundary in `src/citlali/cli/main.cpp` for
science/pointing/holography-style reductions.

Rules:

- iter 0: learn
- iter 1: learn with fruitloops model if available
- iter 2+: apply learned state

Beammap keeps its internal loop, but calls the same learning-state
begin/finalize/apply hooks.

## Phase 3: Pre-RTC Learned Spike Masks

Implement learned spike masks that can be applied before RTC filtering.

This means storing spike locations in scan/sample coordinates and injecting them
into `rtcdata.flags` before `rtcproc.run`. Only high-confidence, off-source
events get this treatment. Ambiguous or source-adjacent events remain diagnostic
until source subtraction confirms them.

## Phase 4: Fix Busy-Network Despiking

Change the current busy-veto policy from "flag nothing" to selective handling:

- accept high-confidence compact off-source clusters
- cap accepted clusters per network/scan
- promote severe survivors to detector/scan penalties
- record accepted/rejected/protected reasons

This is the direct fix for the artifact class seen in pointing reductions.

Implementation note:

- `PTCProc::apply_second_pass_local` now keeps the busy-network diagnostic flag
  but selectively accepts capped high-confidence off-source clusters instead of
  accepting no clusters.
- Source-overlapping candidate rows are counted as protected and are not
  converted into accepted flags.
- PTC diag products and learning CSV rows now carry accepted, rejected, and
  source-protected counts.
- Only accepted non-source PTC candidate events become learned sample-mask
  records.

## Phase 5: High-Weight Detector Validation

During learning iterations, compare detector weights to robust array/network
distributions.

Response:

- cap extreme high weights unless validated
- prevent spike-pathological detectors from being upweighted
- penalize detectors with high weight plus poor atmospheric agreement or
  residual pathology

This addresses the misidentified-tone failure mode.

Implementation note:

- `PTCProc` now compares approximate detector weights to robust group
  distributions (`array`, `nw`, or `all`) during validated weighting.
- Extreme high weights are recorded in the learning CSV. During apply
  iterations, unvalidated extremes can be capped to a configurable multiple of
  the group median.
- Validation still comes from learned source-subtracted/full-vs-approx and
  atmospheric-agreement information; source flux is not treated as a detector
  gain estimator.

## Phase 6: Map-Pixel Outlier Diagnostics

Add mapmaking-side diagnostics that track extreme map pixels using effective
sample count. Per-sample contributor tracking remains an opt-in debug mode until
we have a cheaper replay/provenance path.

Start diagnostic-only, then promote only very clear off-source single-detector
events into learned masks for the next iteration.

Implementation note:

- Per-sample contributor tracing can record the largest weighted sample
  contribution per map pixel, with detector UID, scan, and PTC sample index, but
  it is disabled by default because it adds inner-loop work and lock contention
  during jinc mapmaking.
- `write_mapdiag` now records off-source extreme map pixels in the learning CSV,
  using robust z on the core support, an effective-sample cut when coverage is
  available, and a center-radius source exclusion.
- This phase is intentionally diagnostic-only. It does not yet promote map-pixel
  contributors into learned sample masks.

## Phase 7: Source Protection Generalization

Provide one source-mask interface consumed by RTC/PTC/mapmaking diagnostics:

- pointing: center-radius mask
- holography: configurable/support-based empirical mask
- science: fruitloops support mask
- beammap: detector source centers/priors

No Gaussian assumption in the shared path.

Implementation note:

- A shared `calc_source_protection_mask` interface now provides source masks to
  RTC despiking, PTC second-pass despiking, learned-mask application, and
  mapmaking diagnostics.
- The currently active shared mode is `map_center_radius`, which is appropriate
  for compact pointing/holography tests where the source is assumed near map
  center. Unsupported modes return an all-false diagnostic mask rather than
  imposing a Gaussian or empirical support assumption.

## Phase 8: Config And Defaults

Add conservative config keys, likely under `timestream.learning` or similar:

- `enabled`
- `apply_start_iter`
- source mask mode/radius/support thresholds
- learned spike pre-filter application
- busy-network selective acceptance thresholds
- high-weight cap thresholds
- map-pixel outlier thresholds

Pointing/holography defaults should be source-aware. Science defaults should be
conservative unless fruitloops support is credible.

Implementation note:

- `data/config.yaml` exposes the active learning controls under
  `timestream.learning`, including iteration phase controls, learned sample-mask
  application limits, map-pixel outlier thresholds, and the opt-in expensive
  map-pixel contributor tracing switch.
- Source protection remains explicit on the RTC despiker and PTC second-pass
  local despiker. It is activated only for the pointing-style reduction path,
  which also covers focus and holography reductions that use this pipeline.
- High-weight validation and caps remain under
  `timestream.processed_time_chunk.weighting.validation`, so detector-weight
  policy stays attached to weighting rather than buried in generic learning
  state.
- The current pointing test config keeps contributor tracing disabled by
  default because full per-sample provenance in jinc mapmaking is intentionally
  an expensive debug mode.

## Phase 9: Verification

After implementation:

- build
- review touched paths manually
- update `point_test/70_reduce.yaml`
- inspect diagnostics from the next run
- compare the known bad pixel behavior, FWHM, flux recovery, and spike diagnostics

Current verification note:

- `cmake --build build` passes after moving RTC/PTC learning summary collection
  to per-scan snapshots.
- The next pointing run should check that repeated reductions now produce
  stable `learning_iter_*.csv` counts, especially `rtc_despike` sample masks,
  and that `high_weight_detector` rows are present when validated weighting
  reports high-weight caps/recommendations in the logs.
