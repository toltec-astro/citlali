# Beammap Debug Handoff Notes (2026-02-27)

## Scope and Intent
These notes capture the current beammap debugging state after the prior-guided beammap work that was integrated from the home machine and exercised against the latest local `3c273` test products.

This note is meant to let a later Codex session or a separate machine resume without reconstructing:
- what code landed after the 2026-02-26 handoff,
- what the key beammap config traps are,
- what the latest `redu04` runs showed,
- what remains broken vs what is now fixed.

## Branch / Code State
- Branch: `gw_dev`
- Current head at write time:
  - `a2dd5c38 prior based beammapping`
- Repository was clean before this note was added.

## Code Changes Integrated Since The 2026-02-26 Handoff
These are already in the tree before this note.

### 1) Despike hardening
Commit:
- `a0aa4df9 despike indices issues`

Relevant file:
- `include/citlali/core/timestream/rtc/despike.h`

Purpose:
- Fix invalid index / window handling in despiking.
- Add guardrails and logging around degenerate or invalid spike windows.

### 2) Beammap FITS splitting by detector quality
Commit:
- `58870741 new beammap fits output options`

Relevant files:
- `include/citlali/core/engine/beammap.h`
- `include/citlali/core/engine/engine.h`

Purpose:
- Add config support to write separate beammap FITS products by final detector flag.
- Current outputs use:
  - `..._flag0_good.fits`
  - `..._flag1_bad.fits`

This is what the Python inspection work is using now.

### 3) Beammap priors pipeline and data products
Commits:
- `1f9e06b1 prior based beammapping`
- `0084ec1d prior based beammapping`
- `acff79fc prior based beammapping`
- `1da44317 prior based beammapping`
- `a2dd5c38 prior based beammapping`

Relevant files:
- `data/beammap_priors/beammap_slot_priors_soft_v1.ecsv`
- `data/beammap_priors/beammap_network_priors_v1.ecsv`
- `data/beammap_priors/build_beammap_slot_priors_soft.py`
- `data/beammap_priors/build_beammap_network_priors.py`
- `data/beammap_priors/README.md`
- `include/citlali/core/engine/beammap.h`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/utils/fitting.h`
- `include/citlali/core/utils/utils.h`

Purpose:
- Add soft beammap priors under version control.
- Add prior-guided source initialization in beammap detector mode.
- Add `fallback_blind` behavior so prior failure can optionally fall back to the old blind search.
- Reject `previous` seeds if the seed pixel is invalid / no-weight.
- Add supporting diagnostics around beammap fitting.

## Important Config / Behavior Facts

### 1) `mapmaking.grouping: auto` becomes detector mode in beammap
Code path:
- `include/citlali/core/engine/todproc.h`

For `runtime.reduction_type: beammap`, `auto` resolves to `detector`.

Implication:
- Beammap priors are active with `grouping: auto`.
- Beammap `rfi_mask` is active only because beammap resolves to detector grouping.

### 2) Priors and beammap `rfi_mask` are detector-mode only
Code path:
- `include/citlali/core/engine/beammap.h`

If grouping is not detector, priors are disabled.

### 3) `fallback_blind: false` is dangerous
If priors fail to find a valid candidate and `fallback_blind` is false:
- the detector can be skipped before fitting,
- yielding the old sentinel output (`x_t_raw = y_t_raw = -150`) and `good_fit = 0`.

This was the main reason some runs got worse even when RTC masking was relaxed.

### 4) Map zeros at source locations are often just zero-weight pixels
Map normalization explicitly zeros pixels with `weight <= 0`.
This means "holes" in detector maps are not by themselves proof of signal corruption; they can be coverage loss.

## Local Analysis Products Used
All local quantitative checks in this session used:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu03`
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04`

Main files:
- `redu03/gw-52841352-3c273.out`
- `redu04/gw-52909886-3c273.out` (priors on, `fallback_blind: no`)
- `redu04/gw-52915062-3c273.out` (priors on, `fallback_blind: yes`)
- `redu04/151600/raw/apt_commissioning_beammap_151600_citlali_fit_qc.ecsv`
- `redu04/151600/raw/apt_commissioning_beammap_151600_citlali.ecsv`

## What The Latest `redu04` Runs Showed

### Run A: priors on, `fallback_blind: no`, `rfi_mask: no`, `despike: no`
Run log:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/gw-52909886-3c273.out`

Key findings:
- This run did not just "turn off masking"; it also had priors enabled and `fallback_blind: no`.
- Many detectors were lost because priors produced no acceptable candidate and the code was forbidden from blind fallback.

Quantitative summary:
- a1100 `flag==0`: `2338 / 3163`
- a1400 `flag==0`: `812 / 1233`
- a2000 `flag==0`: `900 / 959`

a1100 per-network vs `redu03`:
- nw0: `489 -> 491`
- nw1: `415 -> 305`
- nw2: `403 -> 353`
- nw3: `416 -> 463`
- nw4: `343 -> 396`
- nw5: `279 -> 330`

The dramatic losses in nw1 / nw2 were traced mostly to prior-init misses, not RTC/PTC masking itself.

No-prior-candidate stats:
- `292` unique detectors had no accepted prior candidate.
- By array:
  - a1100: `208`
  - a1400: `84`
- In a1100 nw2:
  - `39` detectors had no accepted prior candidate.

Important examples from that run:
- `uid=1105`
- `uid=1111`

Those had strong maps but were skipped because:
- `no prior-guided init candidate and fallback_blind=false`

### Run B: priors on, `fallback_blind: yes`, `rfi_mask: no`, `despike: no`
Run log:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/gw-52915062-3c273.out`

This is the current best reference run.

Direct log confirmation:
- config dump contains `fallback_blind: yes`
- there are `0` occurrences of:
  - `no prior-guided init candidate`
- there are many occurrences of:
  - `init mode=blind row=-99 col=-99`

Meaning:
- prior failures now go through blind fallback instead of becoming hard skips.

Quantitative summary vs `redu03`:
- a1100 `flag==0`: `2345 -> 2524` (`+179`)
- a1400 `flag==0`: `606 -> 861` (`+255`)
- a2000 `flag==0`: `851 -> 900` (`+49`)

Blind-fallback usage:
- `3098` unique detectors used blind init at least once.
- Of those:
  - `2501` end as `flag==0`
  - `597` end as `flag==1`
  - only `7` end as `good_fit==0`

This is the clearest indication that blind fallback is rescuing detectors rather than merely hiding failure.

Rescued detectors that were previously lost with `fallback_blind: no`:
- `uid=1105`
- `uid=1111`
- `uid=1245`
- `uid=1317`
- `uid=1321`
- `uid=1330`

These now fit successfully and land in `flag0_good`.

## Current Remaining Failure Mode (Latest `fallback_blind: yes` Run)

### a1100 overall
- `flag==0`: `2524 / 3163`
- `flag==1`: `639 / 3163`
- Sentinel `(-150, -150)` rows still exist in the full a1100 table, but the formerly hard-skipped nw2 losses are no longer sentinel-driven.

### a1100 nw2
- `flag==0`: `386 / 514`
- This is still `17` below the `redu03` nw2 value (`403`), but far better than the `fallback_blind: no` run.

Remaining nw2 `good -> bad` losses vs `redu03`:
- `39`
- Sentinel losses among those: `0`
- Non-sentinel losses: all `39`
- `37 / 39` are `good_fit == 1`
- only `2 / 39` are `good_fit == 0`

Interpretation:
- The remaining nw2 deficit is no longer an initialization failure problem.
- It is now mostly a post-fit quality-cut problem.

Dominant nw2 failure reason combinations:
- `AzFWHM + ElFWHM` (largest group)
- `ElFWHM + Sig2Noise`
- `AzFWHM + ElFWHM + Sig2Noise`
- `Sig2Noise`

This is consistent with:
- wrong-peak selection,
- broadened / distorted fits,
- weak off-source fits that still converge in Ceres.

## Key Detector Examples Worth Remembering

### `uid=273`
- Still in `flag1_bad`.
- `good_fit == 1`, not a Ceres failure.
- The bright compact-looking peak exists, but the stored fit is on a different solution and fails:
  - `AzFWHM`
  - `ElFWHM`
  - `Sig2Noise`
  - `Position`

This is the canonical example of "looks fine by eye, but final fit is elsewhere."

### `uid=4` (earlier fallback-off diagnostics)
- Another wrong-peak case:
  - `good_fit == 1`
  - but final stored solution landed on a bad prior-guided branch and failed post-fit cuts.

### `uid=258`
- The "zero patch at source" concern for this detector was traced to weight-zero coverage holes, not to beammap `rfi_mask`.

## Map Size / Bounds Measurement
Using the latest `redu04` fit-QC, with `flag==0` detectors and `x_t_raw`, `y_t_raw`, plus a `10"` pad on all sides:

### All arrays
- raw x extent: `[-98.64, 118.38]"`
- raw y extent: `[-110.92, 119.37]"`
- Minimal padded box if recentring is allowed:
  - `x_size_pix = 239`
  - `y_size_pix = 251`
  - center about `(9.87", 4.23")`
- Symmetric about `(0,0)`:
  - `x_size_pix = 257`
  - `y_size_pix = 259`

### a1100 only
- raw x extent: `[-98.26, 116.85]"`
- raw y extent: `[-95.53, 119.37]"`
- Minimal padded box if recentring is allowed:
  - `x_size_pix = 237`
  - `y_size_pix = 235`
  - center about `(9.30", 11.92")`
- Symmetric about `(0,0)`:
  - `x_size_pix = 255`
  - `y_size_pix = 259`

Practical recommendation:
- safest immediate test:
  - `x_size_pix: 257`
  - `y_size_pix: 259`

## Local Helper Products Created During Analysis
These are local analysis artifacts only, not part of the repo:
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/151600/raw/toltec_commissioning_a1100_beammap_151600_citlali_flag0_good_coadd.fits`
- `/Users/gwilson/work_toltec/local_data/beammaps/3c273/redu04/151600/raw/toltec_commissioning_a1100_beammap_151600_citlali_flag0_good_stacksum.fits`

These were created to test whether a coadd of good detector maps could help define row limits.
Conclusion:
- not very useful for the intended purpose,
- because summing across all detector-centered maps does not yield a clean direct footprint proxy for trimming.

## Recommended Next Steps
Priority order:

1. Keep `fallback_blind: yes`
- This is now required. It fixed the worst regression from priors.

2. Focus on the remaining post-fit failures, not prior-miss failures
- The dominant residual issue is now:
  - wrong peak / broad peak / weak off-source fit that still converges.

3. Use FITS inspection on the remaining nw2 bad set
- Sort the remaining `39` nw2 losses into:
  - wrong peak,
  - broad / row-corrupted fit,
  - genuine low-SNR miss.

4. Test tighter map size next
- Try the symmetric global bounds first:
  - `x_size_pix: 257`
  - `y_size_pix: 259`

5. If fit stability remains poor, revisit fit-selection strategy rather than RTC masking first
- Blind fallback has shown the larger issue is fit target selection and downstream QC, not simply raw/processed masking.

## Home Computer Resume Context
If continuing from the home machine:
- start from branch `gw_dev` at or after `a2dd5c38`
- use this note plus `DEBUG_NOTES_2026-02-26_BEAMMAP_HANDOFF.md`
- the current reference validation run is:
  - `redu04` with `fallback_blind: yes`
  - log: `gw-52915062-3c273.out`
- the important conclusion to carry forward is:
  - priors are useful,
  - `fallback_blind` must stay on,
  - remaining losses are mostly post-fit QC failures, not prior-init misses.
