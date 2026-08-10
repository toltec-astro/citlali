# RTC learned sampling Stage A metrics

Date: 2026-08-09

Status: implemented and locally validated; observe-only diagnostic candidate;
no scientific tolerances, recommendation, selection, or apply behavior

## Authority and implementation identity

This implementation starts from the clean application branch
`origin/codex/refactor-mainline` at
`46ad23888a40f5102cdfd50c06e49a549bdf8a20` (parent
`4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, tree
`ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`). The isolated implementation
branch is `codex/rtc-learned-sampling-stage-a`. The nonconformant
phase-independent RTC repair candidate `24f28ea9de6b4a1a3ff81d07944fa5fc2565f240`
is not an ancestor and was not used as the base.

The owner-approved design remains the documentation-only commit
`cbb676d84bc58da4239a906a420a04a326968309`. This Stage A subset implements
only its metrics path. Existing requested configuration, fixed execution,
audit findings, and learned-sampling evidence remain separate states.

Relevant existing typed configuration is
`timestream.raw_time_chunk.filter.{enabled,a_gibbs,freq_low_Hz,freq_high_Hz,n_terms}`
and `timestream.raw_time_chunk.downsample.*`. The diagnostic reads the typed
effective raw-time-chunk configuration. It does not mutate it and it adds no
execution consumer.

## What is measured

The existing atomic `rtcdiag` writer now emits an unranked candidate matrix
for every scan and array for which telescope motion and admitted APT beam
metadata are available. It measures:

- the native detector sample rate and assigned native compatibility-grid
  semantics;
- valid physical-AltAz motion intervals from `TelTime`, `az_phys`, and
  `alt_phys`, with a one-telescope-sample boundary guard;
- the exact maximum valid scan speed as the primary scalar motion authority;
- p50, p95, and p99.5 speed values as diagnostics only;
- the scan-direction cross-section of the elliptical Gaussian array beam and
  the valid interval with the shortest beam-crossing time;
- every positive integer factor whose output Nyquist is not below the
  configured FIR high-frequency edge;
- the exact centered FIR coefficients produced by the current
  `timestream::Filter` configuration, or the explicit identity response when
  RTC FIR filtering is disabled; and
- the exact analytical beam, FIR, and phase-zero decimator composition.

For a circular beam, samples per FWHM is exactly

```text
N_beam = f_sample,out * theta_FWHM / v_max.
```

For an elliptical beam and varying scan direction, the implementation uses
the more informative limiting crossing time across all valid intervals. The
separate maximum-speed variable remains present and is never replaced by a
percentile.

The phase-zero decimator calculation explicitly sums every native-band alias
image that folds onto each output-baseband frequency. The candidate metrics
therefore use the realized FIR rather than an ideal brick-wall assumption.
The integrated astronomical alias metric is folded compact-source power
divided by desired compact-source power over the positive output baseband.

## Diagnostic schema

Observation identity, `SAMPRATE`, array identity, filter configuration, and
the APT path are already serialized in `rtcdiag` or are added as the beam
authority. Stage A adds:

- `scan_speed_altaz_max_arcsec_s` and the three percentile diagnostics;
- beam axes, position angle, limiting projected FWHM and limiting speed;
- the unranked `rtc_sampling_candidate_factor` axis;
- `rtc_sampling_realized_fir_coefficients`;
- output rate and Nyquist;
- samples per projected beam FWHM;
- compact-source peak attenuation and FWHM broadening;
- FIR response at the beam temporal half-power frequency;
- astronomical alias-power ratio;
- FIR stopband rejection and transition margin; and
- raw FIR support delay and realized centered-software group delay.

An availability status of `0` means computed and `-1` means required motion or
beam metadata were unavailable. It is not a safe/unsafe label. Signed FIR
"attenuation" may be slightly negative where passband ripple produces gain.
No rank, selected-candidate field, recommendation, or acceptance field exists.

Every file states:

> This is a metrics-only diagnostic. No candidate was selected and no RTC
> behavior was changed.

Physical detector integration-event semantics and absolute timing placement
remain unavailable. The calculation is valid only on the assigned
compatibility grid and does not authorize a timing correction.

## What is intentionally not decided

Stage A does not define response-loss, broadening, alias, stopband, sampling,
or computational-cost tolerances. It does not call any candidate safe or
unsafe, choose an optimal factor, generate learned/resolved/applied state, or
change the factor, FIR, samples, flags, timestamps, time grid, RTC/PTC/map
inputs, weights, maps, or products. It does not add source injection because
the approved evidence path is analytical.

The following remain separate owner decisions:

1. numerical scientific tolerances and the allowed factor/filter-cost policy;
2. the candidate-specific FIR design policy if it is to differ from the
   currently configured realized FIR family;
3. native-cadence fallback versus failure by reduction role;
4. generation and serialization of authoritative learned/resolved state;
5. common-observation Stage B application and restart behavior; and
6. any per-array/per-scan cadence, noise-aware objective, or downstream
   heterogeneous-transfer handling.

## Validation and example

The focused unit suite covers valid maximum-speed extraction with the boundary
guard, invalid-gap rejection, elliptical beam cross-sections, exact FIR
response, beam-times-FIR composition, phase-zero alias folding, deterministic
factor enumeration/results, identity broadening, and the persisted unranked
NetCDF schema and disclaimer.

Local validation on the changed tree passed:

- both `citlali_test` and `citlali_cli` builds;
- 9/9 focused Stage A tests;
- 632/632 runnable CTests, with one pre-existing disabled test;
- 172/172 baseline-tool tests; and
- the complete 127-test config preflight, all four mode kits, all eight compact
  compatibility cases, 100% compact-surface coverage, all authority/boundary
  audits, and the unchanged 45-record raw-execution census at digest
  `09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`.

No representative science dataset was consumed because no local or Unity
science reduction was authorized. The checked-in
[`candidate_metrics.csv`](../validation/rtc_learned_sampling_stage_a_example_2026-08-09/candidate_metrics.csv)
is a deterministic analytical example using 488.28125 Hz native sampling, a
10 arcsec circular Gaussian beam, 100 arcsec/s maximum scan speed, and the
existing 16 Hz, 32-term, 50 dB-Gibbs FIR family. It is an output-format
example, not scientific evidence and not a recommendation.
