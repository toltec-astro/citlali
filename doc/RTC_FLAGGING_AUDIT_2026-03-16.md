# RTC Flagging Audit

This note audits the current RTC-side contamination handling path in Citlali as
of the local code state on 2026-03-16.

It is focused on:

- RTC despiking changes
- level-shift / step masking
- impulsive-event instrumentation
- the actual order in which these checks run

## Audit Findings

### 1. Raw-gate candidate threshold is not aligned with the pre-widening local-raw threshold

Status:

- fixed in the local code state after this audit
- raw candidate nomination now uses `candidate_rel_sigma_scale * sigma_scale *
  min_spike_sigma`
- legacy `candidate_sigma_scale` is deprecated and mapped onto the new relative
  scale during config parsing

Code:

- `local_cutoff = local_residual.sigma_scale * min_spike_sigma * resid_sigma`
  in [despike.h](../include/citlali/core/timestream/rtc/despike.h)
- `candidate_z = local_residual.compact_raw_gate.candidate_sigma_scale * min_spike_sigma`
  in [despike.h](../include/citlali/core/timestream/rtc/despike.h)

Issue:

- the accepted local-raw threshold is expressed as `sigma_scale * min_spike_sigma`
- the raw candidate threshold is expressed as `candidate_sigma_scale * min_spike_sigma`
- these are only equivalent when `candidate_sigma_scale == sigma_scale`

Implication:

- setting `candidate_sigma_scale: 1.0` does not recover the old pre-widening
  raw candidate behavior when `sigma_scale: 0.75`
- with `min_spike_sigma: 8` and `sigma_scale: 0.75`, the old local-raw
  nomination scale was effectively `6 sigma`, while `candidate_sigma_scale: 1.0`
  nominates only above `8 sigma`

Severity:

- moderate

Why it matters:

- it makes survey configuration more error-prone
- it can create a false sense that a config-only revert restored the prior raw
  path when it did not

### 2. `network_step_mask` runs after `altaz_destripe`

Code order in [rtcproc.h](../include/citlali/core/timestream/rtc/rtcproc.h):

1. despike + replace
2. FIR / notch / IIR filtering
3. downsample or copy to RTC output chunk
4. `altaz_destripe`
5. seed RTC despike summaries
6. `capture_rtc_diagnostics(...)`
7. `apply_network_step_mask(...)`
8. `capture_rtc_diagnostics(...)` again

Issue:

- the step mask protects downstream PTC cleaning
- but it does not protect the earlier `altaz_destripe` fit from step-like
  intervals

Implication:

- aligned level shifts can still influence the az/el template regression before
  the mask is applied
- if `altaz_destripe` reacts badly to those intervals, the later step mask may
  be too late to prevent some contamination transfer

Severity:

- moderate

Why it matters:

- this can blur the intended separation between "template subtraction" and
  "mask before PTC clean"

### 3. Raw-gate and delta-gate diagnostics are not directly comparable

Code:

- `local_raw_candidate_count` / `local_raw_reject_count` are event counts
- `local_exceed_count` is a flagged-sample count after raw-gate acceptance
- `local_delta_candidate_count` / `local_delta_exceed_count` /
  `local_delta_reject_count` are event counts

Implication:

- raw and delta metrics with similar names do not have the same units
- naive comparisons in analysis can be misleading

Severity:

- low

Why it matters:

- analysis tooling must treat the raw accepted quantity as sample-domain, not
  event-domain

## Current RTC Search / Flag Order

The current RTC path is:

1. APT-level detector flags are already available before RTC processing.
2. Flux calibration and extinction correction run first when enabled.
3. RTC despiking runs on the raw calibrated timestream.
4. Within RTC despiking, per detector:
   - global raw MAD outliers are found first
   - optional local-residual raw candidates are generated and shape-gated
   - optional local-residual delta candidates are generated and shape-gated
   - native global delta-domain spike finding then runs recursively
5. Flagged samples are replaced by interpolation / grouped replacement in
   `replace_spikes(...)`.
6. FIR / notch / IIR filtering runs if configured.
7. Downsampling runs if configured; otherwise the filtered RTC chunk is copied.
8. Optional `altaz_destripe` regression runs on the RTC output chunk.
9. RTC diagnostic summaries are computed on the output chunk:
   - final flagged fractions / run lengths
   - detector step metrics
   - impulsive metrics
   - network-level step alignment and common-mode metrics
10. If enabled, `network_step_mask` uses those diagnostic summaries to flag a
    network-wide time window around aligned step-like events.
11. RTC diagnostics are recomputed after masking for final writeout, but the
    trigger step metrics are intentionally preserved from the pre-mask pass.
12. RTC netCDF writeout persists:
   - the timestream itself
   - detector/network diagnostics
   - optional impulsive snippets
   - provenance for despike and mask settings

## Technique Summary

### APT / hard detector flags

Purpose:

- remove detectors known bad from calibration or metadata

Where:

- APT flags are honored in RTC despike / replacement and later detector removal

Role:

- hard exclusion, not event detection

### Global raw-sample despiking

Purpose:

- catch very large single-sample or short raw-amplitude outliers using a
  detector-level robust scale on the full scan

Where:

- first raw pass in [despike.h](../include/citlali/core/timestream/rtc/despike.h)

Good for:

- obvious cosmic-ray-like bursts
- gross amplitude outliers

Weak for:

- smaller bursts riding on drift or line structure
- compact events that do not stand out in the full-scan sigma

### Local-residual raw gate

Purpose:

- catch short raw-like bursts after subtracting a local baseline, while
  rejecting broader level shifts / drifts

Where:

- `local_residual.compact_raw_gate` in
  [despike.h](../include/citlali/core/timestream/rtc/despike.h)

Search logic:

- smooth the detector timestream
- subtract the local baseline
- nominate local raw candidates
- cluster adjacent candidate samples into events
- reject events that are too wide or show too large a pre/post baseline shift

Good for:

- compact raw-like bursts
- short local excursions that are not extreme in the global scan variance

Current weak spot:

- candidate nomination still needs tuning to avoid both under-nominating and
  over-firing

### Local-residual delta gate

Purpose:

- catch compact adjacent-sample transients after local detrending, while
  rejecting mini-steps and broader excursions

Where:

- `local_residual.compact_delta_gate` in
  [despike.h](../include/citlali/core/timestream/rtc/despike.h)

Search logic:

- compute locally detrended adjacent-sample deltas
- nominate high local-delta candidate edges
- cluster adjacent candidate edges into events
- accept only compact events with small pre/post baseline shift

Good for:

- compact, edge-like impulsive events

Weak for:

- broader raw-like bursts
- events that are impulsive in appearance but not strongly delta-dominated

### Native global delta despiker

Purpose:

- catch classic spikes via adjacent-sample differences on the full detector scan

Where:

- `spike_finder(...)` path in [despike.h](../include/citlali/core/timestream/rtc/despike.h)

Search logic:

- build adjacent-sample deltas
- iteratively flag large delta outliers
- merge nearby spikes into a representative central flag

Good for:

- classic sharp spikes with strong delta signatures

Weak for:

- structured local bursts riding on low-frequency contamination
- level-shift residuals

### Level-shift / step diagnostics

Purpose:

- identify detector-local and network-aligned step-like behavior

Where:

- `capture_rtc_diagnostics(...)` in
  [rtcproc.h](../include/citlali/core/timestream/rtc/rtcproc.h)

Search logic:

- compute per-detector step scores from left/right window mean jumps
- compute per-network active fraction, alignment fraction, and dominant step
  sample

Good for:

- aligned level shifts affecting a subset of detectors / networks

### Network step mask

Purpose:

- mask a shared time window around aligned network-level step events before PTC
  cleaning

Where:

- `apply_network_step_mask(...)` in
  [rtcproc.h](../include/citlali/core/timestream/rtc/rtcproc.h)

Search logic:

- use detector step metrics and network alignment metrics
- require enough good detectors, enough step-active fraction, and enough timing
  consensus
- reject masks that would cost too much data

Good for:

- coherent step-like / level-shift families

Weak for:

- impulsive families
- any contamination already absorbed by earlier template fits

### Impulsive capture / snippets

Purpose:

- persist compact examples of the strongest impulsive RTC events for diagnosis

Where:

- `impulsive_capture` in
  [rtcproc.h](../include/citlali/core/timestream/rtc/rtcproc.h)

Role:

- instrumentation only
- does not change flagging by itself

Good for:

- checking whether bad survivors are raw-like, delta-like, step-like, or mixed

## Practical Interpretation

The current search order is best thought of as three branches:

1. hard detector exclusion
2. sample / event despiking
3. network-aligned step masking

Within the event-despike branch, the present search sequence is:

1. global raw
2. local raw
3. local delta
4. global delta

That is a sensible structure, but the raw branch is still the least mature.
The level-shift branch is now more coherent and targeted than the impulsive
branch.

## Recommended Near-Term Use

For the 13-obsnum survey:

- keep `network_step_mask` enabled
- keep impulsive capture enabled
- use conservative raw-gate nomination, not the aggressive widened setting from
  `redu31`

For further code work after the survey:

- align raw-gate candidate threshold semantics with `sigma_scale`
- decide whether `altaz_destripe` should remain before or move after the step
  mask
- keep raw-gate and delta-gate counters in event units or document the sample /
  event asymmetry more explicitly in analysis tooling
