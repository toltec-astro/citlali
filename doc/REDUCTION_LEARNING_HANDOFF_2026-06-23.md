# Reduction Learning Refactor Handoff - 2026-06-23

This note summarizes the pointing/holography/science reduction-learning
refactor work done on 2026-06-23. It is intended as a restart point for a new
Codex thread.

The working branch at the time this note was written was `gw_dev`. The code
changes through commit `a638888c` had already been pushed before this handoff
note was added.

## Motivation

The original problem was source-biased and artifact-prone pointing map
production. The test case was the local pointing reduction in:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/point_test`

The key observed failure mode was an a1100 map with troublesome off-source
negative and positive structure. Later diagnostics showed that some of these
features were not normal source leakage and were not simply solved by
fruitloops. They were often dominated by one detector in one scan.

The agreed philosophy is:

- Iteration 0 learns from the data without assuming a source model.
- Iteration 1 can learn with the previous fruitloops source model subtracted.
- Iterations 2 and later apply the learned state.
- Source flux is not a valid detector-gain estimator for pointing/holography,
  because not every detector crosses the source.
- The low-order atmosphere is broadly common across detectors, so disagreement
  with the array/network population is a better detector-quality handle.
- Source protection must be explicit, and the shared path must not impose a
  Gaussian PSF. This matters especially for very out-of-focus holography.
- The ideas are general enough for pointing, holography, and science
  reductions, not just pointing.

## Main Code Areas

- Shared learning state:
  `include/citlali/core/engine/learning.h`
- Engine learning config, application hooks, diagnostics, learning CSV writer,
  and mapdiag diagnostics:
  `include/citlali/core/engine/engine.h`
- Fruitloops iteration-phase wiring:
  `src/citlali/cli/main.cpp`
- Pointing reduction path:
  `include/citlali/core/engine/pointing.h`
- Science/lali reduction path:
  `include/citlali/core/engine/lali.h`
- PTC weighting, high-weight validation, second-pass despiking diagnostics:
  `include/citlali/core/timestream/ptc/ptcproc.h`
- Mapmaking contributor tracing:
  `include/citlali/core/mapmaking/map.h`
  `include/citlali/core/mapmaking/jinc_mm.h`
  `src/citlali/core/mapmaking/map.cpp`
- Config defaults:
  `data/config.yaml`
- Running phase plan:
  `doc/REDUCTION_LEARNING_REFACTOR_PLAN.md`

## Implemented Refactor Pieces

### Shared Learning State

`ReductionLearningState` now carries the cross-stage learning records:

- learned sample masks
- detector penalties
- high-weight detector diagnostics
- map-pixel outliers
- busy-network summaries
- source-protection summaries
- learned-mask and detector-exclusion application summaries
- current iteration phase: `learn`, `learn_with_model`, or `apply`

The learning CSV files are written as `learning_iter_*.csv` in each reduction
directory.

### Iteration Phase Wiring

The main fruitloops loop now calls `begin_iteration`, `finalize_iteration`, and
`write_learning_summary`. Iteration phase is controlled by:

- `timestream.learning.learn_iters`
- `timestream.learning.apply_start_iter`

Current default behavior is conservative. Learning is disabled in
`data/config.yaml` unless a reduction config opts in.

### Source-Aware Learned Sample Masks

Learned sample masks can be applied before RTC filtering and before PTC
cleaning. They are gated by iteration phase, scan, obsnum, detector UID, and
source protection.

The important config keys are:

- `timestream.learning.apply_sample_masks_enabled`
- `timestream.learning.apply_max_new_flagged_fraction`

The current source-protection mode is center-radius based. This is appropriate
for the pointing test, where the source is assumed to be within about 20-30
arcsec of map center. It is not yet the full final source-mask abstraction for
all holography/science cases.

### Busy-Network Despiking

The previous busy-network behavior could veto all candidates in a busy network.
That was changed toward selective handling:

- accept compact high-confidence off-source clusters
- cap accepted clusters per network/scan
- protect source-adjacent clusters
- record accepted, rejected, protected, and busy-vetoed reasons
- record severe residuals as diagnostic detector penalties

This improved the philosophy but did not catch all map artifacts, because the
remaining a1100/a1400 features were not necessarily narrow one-sample spikes.

### High-Weight Detector Validation

Validated/hybrid weighting compares detector weights to robust network/array
distributions and atmospheric-agreement diagnostics. The intent is to avoid the
approximate-weighting failure mode where a misidentified tone can receive a high
weight while being assigned to the wrong detector.

High-weight rows appear in the learning CSV as `high_weight_detector`.

In the latest inspected pre-exclusion reduction, uid 4072 was capped strongly
but still left an a1400 feature, showing that capping alone is not always enough.

### Map-Pixel Diagnostics

Mapdiag now records off-source extreme map pixels in the learning CSV as
`map_pixel_outlier`. The selection uses robust z on the off-source core support,
an effective sample cut when coverage is available, and a central source
protection radius.

Two contributor modes exist:

- Full contributor tracing:
  `map_pixel_outlier_contributor_diagnostics_enabled`
- Low-overhead targeted contributor tracing:
  `map_pixel_outlier_targeted_contributor_diagnostics_enabled`

Full tracing is expensive and remains off by default. Targeted tracing uses
previous-iteration outlier pixels as targets for the next mapmaking pass. Jinc
targeted tracing records the same numerator/denominator/variance-weight
arithmetic used by map normalization, so leave-one-out diagnostics are
consistent with the final map.

### Scan-Local Detector Exclusion

The most recent implemented response is an opt-in learned scan-local detector
exclusion.

When mapdiag emits retained off-source outlier pixels with contributor
provenance, it groups them by `(uid, scan)`. If a single detector/scan owns at
least `map_pixel_outlier_detector_exclusion_min_pixels` retained outlier pixels,
it records a scan-local detector penalty:

- `record_type=detector_penalty`
- `reason=map_pixel_outlier_detector_dominance`
- `factor=0`
- `scan_local=1`

In later apply iterations, the pointing and lali paths consume those penalties
before PTC cleaning. The detector is flagged for that scan, its scan-local apt
flag is set, and any current weight is zeroed. The application rows are written
as:

- `record_type=detector_penalty_application`
- `application_stage=pre_ptc_detector_exclusion`
- `reason=apply_learned_detector_exclusion`

The repo default is disabled:

- `timestream.learning.map_pixel_outlier_detector_exclusion_enabled: false`

The local point-test config has it enabled:

- `map_pixel_outlier_detector_exclusion_enabled: true`
- `map_pixel_outlier_detector_exclusion_min_pixels: 4`

This is intentionally scan-local, not observation-global. Observation-level
detector exclusion should require recurrence across scans and/or reductions.

## Latest Inspected Reduction State

The most recent detailed inspection before scan-local detector exclusion used
the reset `redu00` set in `point_test/reduced`.

Notable findings:

- Obsnum: `152389`
- Final a1100 mapdiag outliers were dominated by uid 982, scan 2.
- The dominant a1100 pixels had values around 278-400 mJy/beam.
- Their leave-one-out z scores were modest, about 0.95-2.57, which argued
  against a single ultra-narrow sample spike and toward scan-local detector
  contamination.
- Earlier large negative a1100 structure was associated after targeted tracing
  with uid 681, scan 2. It was severe in early iterations but was no longer a
  top final a1100 outlier in the inspected final iteration.
- a1400 retained a feature dominated by uid 4072, scan 2. This detector had a
  high-weight cap applied but still produced an artifact-like map feature.
- a2000 had no comparable final map-pixel outlier issue in that inspection.

Pointing fit values from that inspected final reduction:

- a1100 FWHM: about 6.55 x 6.58 arcsec
- a1400 FWHM: about 7.21 x 7.64 arcsec
- a2000 FWHM: about 10.48 x 10.38 arcsec

Scan numbering note: learning CSV `scan` values are zero-based. Log messages
often print scan number as one-based.

## Current Local Test Config

The active local test config is outside the repo:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/point_test/70_reduce.yaml`

At handoff, the important learning settings were:

```yaml
timestream:
  learning:
    enabled: true
    diagnostics_enabled: true
    learn_iters: 2
    apply_start_iter: 2
    apply_sample_masks_enabled: true
    apply_max_new_flagged_fraction: 0.02
    map_pixel_outlier_diagnostics_enabled: true
    map_pixel_outlier_contributor_diagnostics_enabled: false
    map_pixel_outlier_targeted_contributor_diagnostics_enabled: true
    map_pixel_outlier_detector_exclusion_enabled: true
    map_pixel_outlier_targeted_contributor_max_pixels: 32
    map_pixel_outlier_detector_exclusion_min_pixels: 4
    map_pixel_outlier_top_n: 8
    map_pixel_outlier_min_abs_z: 8.0
    map_pixel_outlier_min_n_eff: 4.0
    map_pixel_outlier_source_radius_arcsec: 30.0
```

## What To Test Next

Run the point-test reduction with the current branch and config above.

After the run, inspect:

```sh
rg "map_pixel_outlier_detector_dominance|detector_penalty_application|pre_ptc_detector_exclusion" \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/point_test/reduced/redu*/learning_iter_*.csv
```

Expected diagnostic pattern:

- Iteration 0/1 may still show map outliers while learning.
- Once targeted tracing has contributor provenance, mapdiag should nominate
  detector penalties for repeated off-source outlier ownership.
- From apply iterations onward, matching detector-penalty application rows
  should appear for the affected scan.
- The final a1100 uid 982 scan-2 feature should be reduced or gone if that
  detector/scan is the real cause.
- The a1400 uid 4072 scan-2 feature should also be checked, since weight capping
  alone did not remove it previously.

Also compare:

- final a1100/a1400/a2000 maps
- pointing offsets
- FWHM and PSF shape
- source flux recovery
- `learning_iter_*.csv` counts
- PTC diagnostics for the culled detector/scan
- whether the detector-exclusion application cap ever rejects a cull

## Remaining Work

Short-term:

- Run the new detector-exclusion configuration on Unity.
- Confirm that `detector_penalty` and `detector_penalty_application` rows appear
  as expected.
- Verify that scan-local exclusion improves the known off-source artifacts
  without harming the central source, FWHM, or flux recovery.
- Decide whether `map_pixel_outlier_detector_exclusion_min_pixels: 4` is the
  right threshold. It may need tuning after one or two reductions.
- Inspect whether repeated detector penalties duplicate excessively in the CSV.
  This is probably harmless but may be worth compacting later.

Medium-term:

- Promote observation-level detector exclusion only if the same UID is
  pathological across multiple scans or multiple observations.
- Generalize source protection beyond `map_center_radius`:
  pointing can use center-radius; holography likely needs support/empirical
  masks; science should prefer credible fruitloops support.
- Continue the map-pixel outlier work: define "extreme" robustly when sampling
  varies strongly across a map, and decide when a map-pixel contributor should
  become a sample mask versus a detector/scan cull.
- Audit whether beammap's internal iteration loop should call the same
  begin/finalize/apply hooks, or whether it should remain separate because it
  already has beammap-specific source priors.
- Add regression tests or at least small diagnostic tests for learning-state
  CSV output and detector-exclusion application.

Longer-term:

- Revisit full vs approximate vs validated/hybrid weighting once the refactor
  behavior is stable.
- Keep approximate weighting as a diagnostic comparison, but avoid relying on it
  as the solution because of the misidentified tone-matching failure mode.
- Consider a richer detector-quality score that combines high weight,
  atmospheric disagreement, busy-network residuals, map-pixel ownership, and
  recurrence across scans.

## Build Status

Before this handoff note, the code built successfully with:

```sh
cmake --build build
```

`git diff --check` was also clean before adding this documentation-only commit.

