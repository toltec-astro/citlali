# Fruit-Loop Population Continuation Proposal

Date: 2026-07-27

Status: proposed from complete 108-observation evidence; do not launch until
the checkpoint-v2 setup bundle and preflight are generated and reviewed

## Why Another Run Is Bounded

The ten-iteration population is complete and audited. The candidate V0 rule
resolves 57/108 observations. The 51 non-stops divide into two scientifically
different classes:

- 23 are measurement-limited, primarily by a censored pointing Gaussian width;
- 28 have interpretable measurements but have not satisfied all V0 guards.

Only the 28 trajectory-unresolved observations are proposed for more
fruit-loop iterations. Their authoritative machine-readable list is
`validation/fruit_loop_population_full_analysis_2026-07-27/trajectory_continuation_candidates.csv`.
The excluded measurement class is
`measurement_limited_observations.csv` in the same directory.

The continuation set comprises 15 unresolved radio sources and 13 planetary
disks. By source it contains eleven 3C273, two 3C279, one 3C345, one 3C84,
four Neptune, and nine Uranus observations. It contains 17 normal, eight
marginal, and three stress observations.

## Questions Resolved By One Block

One three-iteration block, absolute iterations 10 through 12, answers:

1. whether the remaining amplitude and whole-map motion contracts enough for
   two consecutive all-array passes;
2. whether the provisional 10%-over-seed background guard returns to a stable
   state or marks a persistent alternate trajectory;
3. whether planet disks merely need a longer horizon than unresolved sources;
4. whether any trajectory reaches a measured plateau while a different guard
   remains systematically failed.

The block does not resolve censored PSF measurements, absolute source flux,
the residual injected-source attenuation, or pointing-versus-science transfer.

## Run Contract

For every selected observation:

1. use the exact Stage A/Stage B executable SHA256
   `0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330`;
2. use the existing absolute-iteration-9 checkpoint schema v2 at that
   observation's `redu09`;
3. preserve the low-level science configuration, inputs, APT, learning policy,
   kernel production, diagnostics, and injection-disabled state;
4. set the absolute fruit-loop maximum to 13, producing iterations 10, 11, and
   12;
5. write to a new, observation-unique continuation root rather than appending
   to or sharing the ten-iteration workspace;
6. preserve absolute iteration identity from the FITS and checkpoint metadata;
7. record config, checkpoint, input, APT, and executable checksums; and
8. fail on missing products, non-contiguous absolute iteration identity,
   changed restart policy, or unexpected error-level messages.

Use one observation per Slurm array task and retain the established
four-observation concurrency limit. The bundle preflight must verify all 28
restart checkpoints before submission and refuse every nonempty destination.

## Evaluation And Stop Rule

Append the new absolute transitions 9→10, 10→11, and 11→12 to the existing
trajectory. Reapply the unchanged offline V0 definitions:

- morphology-aware amplitude below 3%;
- morphology-aware major and minor FWHM change below 5%;
- centroid step below 0.1 arcsec;
- successive whole-map change below 5%;
- map-weight change below 5%;
- valid-mask symmetric difference below 1%;
- learning in the `apply` phase; and
- source-free background sigma no more than 10% above the seed.

Require two consecutive passing transitions in all three arrays. Retain the
strict unchanged-mask/penalty-count variant as a sensitivity result rather
than making it the core rule.

After this block:

- stop a trajectory when V0 passes;
- classify a contracting but persistently guarded trajectory as a measured
  plateau for owner review;
- request another three-iteration block only when motion remains material and
  continues to contract; and
- retain a predeclared absolute safety cap before launch.

## Download Scope

The continuation analysis needs FITS signal, weight, coverage, and kernel
extensions; pointing/learning tables; compressed Citlali logs; copied configs;
restart/provenance metadata; and the Slurm wrapper logs. Processed and raw
timestream products are not required for the convergence analysis and need
not be downloaded.

## Decision Before Launch

Approve or revise this single 28-observation, three-iteration block. Once
approved, generate and test a dedicated checkpoint-v2 continuation setup
bundle and provide exact upload, preflight, `sbatch`, monitoring, and selective
download commands. Do not reuse the fresh-start Stage B generator unchanged:
the continuation bundle must validate restart identity and absolute iteration
numbering explicitly.
