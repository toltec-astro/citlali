# Beammap Debug Office Addendum (2026-02-27)

## Purpose
This is the office-side addendum to the home-machine handoff in:
- `DEBUG_NOTES_2026-02-27_BEAMMAP_HANDOFF.md`

That file is being committed unchanged. This addendum captures the extra local analysis and decisions made afterward so the two notes can be merged later without losing context.

## Branch / Code State
- Branch: `gw_dev`
- Code state before this note:
  - `a2dd5c38 prior based beammapping`
- No additional code changes were made in this office session.
- This commit is note-only.

## Main Conclusion From The Latest Checks
The beammap priors path is no longer failing in the same way it did when:
- `beammap.priors.enabled: true`
- `beammap.priors.fallback_blind: false`

The change to:
- `beammap.priors.fallback_blind: true`

removed the dominant hard-failure mode where detectors were skipped before fitting due to:
- `no prior-guided init candidate and fallback_blind=false`

In the current best reference run:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/gw-52915062-3c273.out`

there are:
- `0` occurrences of `no prior-guided init candidate`
- `3098` unique detectors that use blind init at least once

Of those blind-fallback detectors:
- `2501` end `flag==0`
- `597` end `flag==1`
- only `7` end `good_fit==0`

Interpretation:
- blind fallback is rescuing detectors,
- not just hiding a new failure mode.

## Remaining Failure Mode
The remaining losses are now mostly:
- wrong-peak fits,
- broadened fits,
- or weak off-source fits that still converge in Ceres but fail final QC cuts.

For a1100 nw2 relative to `redu03`:
- `flag==0`: `403 -> 386`
- remaining net deficit: `17`

Among the remaining nw2 `good -> bad` losses:
- `39` total
- `37 / 39` are `good_fit == 1`
- only `2 / 39` are `good_fit == 0`

Dominant failure bits are:
- `AzFWHM + ElFWHM`
- `ElFWHM + Sig2Noise`
- `AzFWHM + ElFWHM + Sig2Noise`
- `Sig2Noise`

This means the current problem is mostly post-fit quality, not seed absence.

## Important Detector-Specific Reminder
`uid=273` is still a useful example:
- it remains in `flag1_bad`
- it is not a Ceres failure (`good_fit == 1`)
- the visible compact peak is not where the stored final fit landed

This is the clearest "looks good by eye, fails because the accepted solution is elsewhere" example.

Because of split beammap output:
- `uid=273` is in:
  - `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/151600/raw/toltec_commissioning_a1100_beammap_151600_citlali_flag1_bad.fits`
- not in:
  - `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/151600/raw/toltec_commissioning_a1100_beammap_151600_citlali_flag0_good.fits`

This matters when opening detector map HDUs by `signal_det_<uid>_I`.

## Zeroed Pixels In Detector Maps
The "source location is zeroed" issue is still most consistent with weight-zero coverage holes, not a new direct signal-zeroing path in priors code.

Relevant implementation fact:
- map normalization writes signal/error as zero where `weight <= 0`

So a source-centered hole can happen if the fitted center lands on a pixel with no accumulated map weight.

This is still worth checking against RTC/PTC masking when convenient, but the current evidence does not point to priors directly zeroing signal maps.

## Map Size / Bound Reduction
The quick bounds exercise using the latest `redu04` fit-QC and `flag==0` detector positions (`x_t_raw`, `y_t_raw`) with a `10"` pad on all sides gave:

### All arrays
- raw x extent: `[-98.64, 118.38]"`
- raw y extent: `[-110.92, 119.37]"`
- padded symmetric size about `(0,0)`:
  - `x_size_pix = 257`
  - `y_size_pix = 259`

### a1100 only
- raw x extent: `[-98.26, 116.85]"`
- raw y extent: `[-95.53, 119.37]"`
- padded symmetric size about `(0,0)`:
  - `x_size_pix = 255`
  - `y_size_pix = 259`

Practical recommendation:
- if no recentering is desired, the safe immediate test is:
  - `x_size_pix: 257`
  - `y_size_pix: 259`

That trims map pixels by about 26.5% relative to the current `301 x 301` maps.

## Side Quest Outcome
I also built local coadd / stack-sum products from the split `flag0_good` a1100 detector maps under:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/151600/raw/`

Those files were useful for a quick look, but the user decided this was not the right way to judge row trimming, so there is no code follow-up attached to that path.

## Recommended Next Step From Home
The next technically useful step is still a targeted audit of the remaining `flag1` detectors in the current best run, split into:
1. wrong-peak acceptance,
2. broadened / row-artifact fits,
3. true low-SNR failures.

That is now more valuable than more work on prior-init fallback, because the hard prior-miss path has already been neutralized by `fallback_blind: true`.
