# Fruit-loop 108-observation population extension

Date: 2026-07-26

Status: independent quality baseline complete; 16-observation Unity Stage A
launch bundle prepared but not uploaded or submitted

## Immediate finding

The original five-observation fruit-loop sample is not representative of the
full quality range:

- 4 of the 5 are in the normal half of the 108-observation population;
- 1 is marginal; and
- 0 are in the lowest-quality stress stratum.

The five observations have quality ranks 7, 10, 20, 39, and 60 of 108. This
explains why the first investigation is useful mechanistic evidence but cannot
establish a population-wide calibration-reference policy.

The 108 RC1 pointings comprise 65 observations of 3C273, 17 of Uranus, 16 of
Neptune, 6 of 3C279, 2 of 3C345, and 2 of 3C84. All 324 array maps and their
processed kernels are present in the existing TolAPT evidence tables.

## Quality definition

Quality is frozen before examining any new fruit-loop trajectory. It comes
from the existing fruit-loop-disabled RC1 pointing products and is therefore
independent of the convergence outcome.

For each array, five diagnostics are converted to within-array badness
percentiles:

1. low fitted S/N;
2. low fitted-amplitude/background-sigma contrast;
3. high map roughness fraction;
4. large absolute log mismatch between measured and processed-kernel FWHM;
5. large absolute log departure of the fitted axis ratio from unity.

Each array receives the equal-weight mean of these components. Observation
map badness is:

```text
0.5 * median(array badness) + 0.5 * worst(array badness)
```

The observation score is:

```text
0.8 * population rank(map badness)
    + 0.2 * population rank(cross-array centroid RMS)
```

Absolute centroid offset is not penalized because it may be a real pointing
offset. Only cross-array inconsistency contributes to the quality score.
Non-finite diagnostics receive worst badness.

The strata are descriptive experiment-design labels, not rejection rules:

| Stratum | Rank range | Observations | Existing fruit-loop sample |
|---|---:|---:|---:|
| Normal | 1–54 | 54 | 4 |
| Marginal | 55–92 | 38 | 1 |
| Stress | 93–108 | 16 | 0 |

Quantile labels avoid inventing absolute quality thresholds before the
relationship between these diagnostics and fruit-loop behavior is measured.
Every observation remains in the population analysis.

## Evidence package

`validation/fruit_loop_population_quality_2026-07-26/` contains:

- `array_quality_metrics.csv`: 324 source rows and component percentiles;
- `observation_quality_inventory.csv`: all 108 observation scores, ranks,
  strata, and absolute diagnostics;
- `quality_stratum_summary.csv`: stratum counts and ranges;
- `population_run_matrix.csv`: the ordered two-stage Unity plan;
- three quality-distribution plots; and
- `manifest.json`: input hashes and the exact scoring definition.

Reproduce it with:

```bash
MPLCONFIGDIR=/tmp/citlali-fruitloop-mpl \
  $HOME/tolteca/bin/python \
  tools/fruit_loops/stratify_pointing_quality.py \
  --hero-metrics \
    ../tolapt/outputs/hero-pointing-comparison/v4-vs-modeled-frequency-rc1-multiyear-full108/hero_reduction_metrics.ecsv \
  --kernel-metrics \
    ../tolapt/outputs/hero-pointing-comparison/v4-vs-modeled-frequency-rc1-multiyear-kernels/kernel_metrics.ecsv \
  --output validation/fruit_loop_population_quality_2026-07-26
```

## Two-stage real-source run

Do not start with one 108-input Citlali process. The earlier long-process
`SIGBUS` investigation and the exact standalone/batched equivalence result
favor one observation per process with a unique output root. A scheduler array
may coordinate the tasks.

### Stage A: 16 sentinels

Run 16 observations through ten saved iterations using one immutable current
binary and the same frozen policy:

- the original five observations, rerun for a common binary, checkpoint-v2
  schema, and ten-iteration horizon;
- three additional anchors from each quality stratum; and
- extra observations needed to cover 3C345 and 3C84.

The 11 new anchors selected reproducibly from the frozen quality ranks are:

| Obsnum | Source | Rank | Stratum | Selection reason |
|---:|---|---:|---|---|
| 129081 | 3C273 | 9 | normal | lower-badness anchor |
| 151594 | 3C273 | 28 | normal | median anchor |
| 133542 | 3C273 | 47 | normal | higher-badness anchor |
| 152990 | 3C84 | 54 | normal | missing-source coverage |
| 130921 | 3C273 | 61 | marginal | lower-badness anchor |
| 150818 | 3C273 | 74 | marginal | median anchor |
| 134546 | 3C345 | 75 | marginal | missing-source coverage |
| 148719 | Uranus | 87 | marginal | higher-badness anchor |
| 151951 | 3C273 | 95 | stress | lower-badness anchor |
| 142578 | Uranus | 101 | stress | median anchor |
| 123426 | Neptune | 106 | stress | higher-badness anchor |

Stage A answers whether the unchanged ten-iteration horizon is adequate across
quality, whether poor fits cause false convergence/no-op behavior, and whether
the candidate diagnostics remain numerically meaningful in stress maps.

Gate before Stage B:

- all 16 jobs have ten contiguous saved products and zero unexpected
  error-level messages;
- metrics are finite or explicitly classified as fit/no-op failures;
- at least two observations in every stratum have interpretable trajectories;
- the 1%, 2%, 5%, and 10% two-transition assessments can be computed without
  changing their definitions; and
- no quality-dependent failure requires a setup or measurement correction.

### Stage B: remaining 92

If Stage A passes, run the remaining 92 observations with the same executable,
policy, ten-iteration horizon, and per-observation workspace. Report all 108;
do not discard the stress stratum or failed fits from yield accounting.

For each observation/array, record the first iteration satisfying each
candidate tolerance for two successive transitions. If iteration 10 is not
enough, add checkpoint-v2 continuation in three-iteration blocks only for the
unresolved observations. Do not reinterpret a fixed ten-iteration cap as
convergence.

Population summaries must be split by quality stratum and source, while
retaining the full per-observation table. At minimum report:

- convergence yield and failure mode;
- first stable iteration at 1%, 2%, 5%, and 10%;
- amplitude and kernel-normalized amplitude trajectory;
- major/minor FWHM relative to the processed kernel;
- per-step and cumulative centroid change;
- ordinary fit S/N and cumulative degradation;
- weights, robust background, roughness, and whole-map relative RMS; and
- relationships with the independently frozen quality score.

## Controlled transfer subset

Real-source trajectories cannot establish photometric transfer. Use the same
immutable binary to create fresh ten-iteration checkpoint-v2 references and
exact control/injected pairs for:

| Obsnum | Stratum | Reason |
|---:|---|---|
| 133410 | normal | existing transfer benchmark |
| 151718 | marginal | existing real trajectory and highest tau in the original five |
| 142578 | stress | middle of the stress stratum |

This is the smallest subset that tests whether the measured recovery fraction
and plateau iteration depend on baseline pointing quality. Use a common
synthetic per-array amplitude for cross-observation comparison; do not use an
unmatched APT or Beammap configured flux as truth. Each pair must pass exact
uninterrupted-versus-restarted equality at its first iteration.

Amplitude-linearity and off-center tests remain separate axes. They should not
be multiplied across all three quality strata until the three representative
transfer pairs show a quality dependence.

## Interpretation policy

The four calibration-reference verdicts remain separate:

- **Astrometry:** compare endpoint centroid stability and cross-array
  consistency by quality stratum. A stable final step does not erase a
  material cumulative shift.
- **Effective PSF:** require both FWHM axes and kernel-relative shape to settle;
  report poor-fit yield rather than forcing a width.
- **Photometry:** use only exact injected-source recovery and independently
  established external flux truth. Real-source monotonic growth alone is not a
  calibration.
- **Science response:** remains unmeasured until an associated science
  observation and approved science-mode injection exist.

A population stopping policy is eligible for consideration only if it reaches
an explicitly selected yield across normal, marginal, and stress observations
without scientifically material S/N loss. Median convergence alone is
insufficient.

## Unity handoff

The owner-run Stage A bundle is now prepared at
`validation/fruit_loop_population_stage_a_2026-07-26/`. The exact upload,
immutable-binary snapshot, preflight, Slurm-array launch, monitoring, download,
and cleanup commands are recorded in
`handoff/FRUIT_LOOP_POPULATION_STAGE_A_UNITY_HANDOFF_2026-07-26.md`.

The bundle implements the following launch contract:

1. freeze one current Citlali executable and record version/SHA256;
2. generate 16 single-observation configs from the frozen RC1 input entries
   and fruit-loop policy;
3. set `max_iters: 10`, `save_all_iters: true`, diagnostics and kernel output
   on, no restart path, and injection disabled;
4. give every observation an independent output workspace;
5. submit as a scheduler array with one observation per task; and
6. download all products and logs before evaluating the Stage B gate.

Production defaults, PTC/RTC algorithms, and build integration remain
unchanged.
