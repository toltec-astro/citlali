# GOODS-N Blank-Sky / MP Cleaner Handoff

Date: 2026-03-13

This note is the handoff for two parallel threads:

1. the GOODS-N blank-sky reduction / diagnosis work in
   `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N`
2. the Citlali cleaner implementation work in this repo

It is intended so another Codex session can resume without rebuilding the full conversation.

## Executive Summary

- The GOODS-N `a1100` problem does not look like a bulk astrometry / pointing failure.
- The main issue is persistent non-astronomical residual structure that survives cleaning and projects coherently to the sky.
- For `a1100`, the relevant networks are `nw0-5`.
- The residual contamination is not one thing:
  - `nw2`: coherent subgroup / common-mode residuals
  - `nw4`: heavy-tail / impulsive detector contamination
  - `nw0` and `nw1`: residual low-frequency / scan-synchronous leakage
- The best single empirical improvement so far was stronger RTC despiking (`min_spike_sigma: 9`), then a deeper `a1100` PCA cut (`18` instead of `12`).
- The new direction is to replace manual PCA depth tuning with an optional Marchenko-Pastur mode selector in PTC, while keeping RTC cleanup intact.

## Main Science Notes

The primary science report is:

- `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/DEEP_DIVE_2026-03-11.md`

That file already contains the earlier map-domain and reduction-by-reduction analysis. This note summarizes the state after the follow-up null tests and the MP-cleaner implementation work.

### High-level conclusions so far

- The raw `jinc` vs `naive` comparison showed that the mapmaker matters, but it is not the root cause by itself.
- `jinc` raw maps gave better compact-source punch for GN20 but also amplified clutter.
- `naive + Wiener` was the best-behaved comparison product among the earlier map tests.
- The clutter is distributed across the dataset, not caused by one or two bad maps.
- The dominant observations (`151930`, `151937`) amplify the clutter because they carry most of the weight, but they do not uniquely create it.

### Why the interpretation changed

The decisive shift was treating these blank-sky observations as a null experiment:

- astrophysical sky signal per sample is negligible compared to detector noise
- cleaned timestreams should be close to uncorrelated Gaussian noise
- any strong departures from Gaussianity, independence, or stationarity should be treated as contamination

That made the map a projection product, not the primary diagnostic. From that point on, the analysis focused on per-scan / per-network null failures in cleaned TOD.

## Important Paths

### GOODS-N reduction area

- root: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N`
- main report: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/DEEP_DIVE_2026-03-11.md`

### Blank-sky tooling added to Citlali

- `/Users/gwilson/GitHub/citlali/tools/blank_sky/README.md`
- `/Users/gwilson/GitHub/citlali/tools/blank_sky/blank_sky_null_audit.py`
- `/Users/gwilson/GitHub/citlali/tools/blank_sky/localize_detector_clusters.py`
- `/Users/gwilson/GitHub/citlali/tools/blank_sky/mp_mode_estimator.py`

### Key outputs already produced

Blank-sky null audits:

- `redu03 / 151930`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu03/151930/raw/blank_sky_null_audit/BLANK_SKY_NULL_AUDIT.md`
- `redu04 / 151930 / a1100 subset`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu04/151930/raw/blank_sky_null_audit_smoketest/BLANK_SKY_NULL_AUDIT.md`
- `redu05`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu05/151930/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`
- `redu06`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu06/151930/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`
- `redu07`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu07/151930/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`
- `redu08`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu08/151930/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`
- `redu09`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu09/151930/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`
- `redu10 / 152524`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu10/152524/raw/blank_sky_null_audit_a1100/BLANK_SKY_NULL_AUDIT.md`

Cluster localization:

- worst `nw2` coherence scan: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu03/151930/raw/localize_scan046_nw02/cluster_localization_summary.csv`
- worst `nw4` tail scan: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu03/151930/raw/localize_scan098_nw04/cluster_localization_summary.csv`

RTC/PTC residual checks:

- `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu04/analysis_timestream_residuals_a1100/RTC_PTC_RESIDUALS_REPORT.md`

Correlation summaries:

- `redu04 a1100`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu04/151930/raw/corr_analysis_a1100/obs151930_corr_summary.csv`
- `redu03 a1100 quick`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu03/151930/raw/corr_analysis_a1100_quick/obs151930_corr_summary.csv`

MP prototype outputs:

- `redu09 / 151930 / low-band`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu09/151930/raw/mp_mode_estimate_lowband/MP_MODE_ESTIMATE.md`
- `redu10 / 152524 / low-band`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu10/152524/raw/mp_mode_estimate_lowband/MP_MODE_ESTIMATE.md`
- `redu09 / 151930 / full-band`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu09/151930/raw/mp_mode_estimate_fullband/MP_MODE_ESTIMATE.md`
- `redu10 / 152524 / full-band`: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu10/152524/raw/mp_mode_estimate_fullband/MP_MODE_ESTIMATE.md`

## Reduction Sequence and Interpretation

### Earlier map-oriented runs

- `redu01`: simplified baseline cleaning, `jinc`
  - improved global `a1100` AzTEC correlation over the more aggressive original run
  - recovered GN20 better
  - still had substantial clutter

- `redu02`: same cleaning baseline, `naive`, plus Wiener products
  - raw `naive` looked statistically cleaner than raw `jinc`
  - `jinc` raw preserved more compact-source punch but more interpolation/noise structure
  - `naive + Wiener` was the most stable comparison product

### Targeted timestream runs

- `redu03`: single-observation PTC mini output for `151930`
  - used to audit the full observation cheaply

- `redu04`: `151930`, RTC + PTC full outputs for selected scans `[2, 5, 8]`
  - established that for `a1100` / `nw0-5`
    - `nw2` was the strongest coherent residual network
    - `nw0` had weak low-frequency suppression
    - `nw1` and `nw4` were next-tier issues

### One-knob diagnostic tests relative to `redu04`

- `redu05`: enable `null_model`
  - clear reject
  - made coherence and low-frequency metrics worse
  - interpretation: surrogate null was misclassifying modes in this non-Gaussian mixture

- `redu06`: lower RTC despike threshold from `12` to `9`
  - clear improvement
  - helped `nw4` tail excess substantially
  - also improved `nw2` coherence metrics some
  - became the new baseline

- `redu07`: raise RTC high-pass from `0.1` to `0.2 Hz`
  - mixed
  - some low-frequency metrics improved
  - coherence metrics got worse in important networks
  - not adopted

- `redu08`: disable `altaz_destripe`
  - mixed-to-negative
  - low-frequency leakage generally worsened
  - interpretation: destriping is helping more than hurting

### Follow-up diagnostic baseline

- `redu09`: `redu06` + `a1100 n_eig_to_cut = 18`
  - net improvement over `redu06`
  - especially helped `nw2`, `nw4`, and `nw1`
  - not uniformly better in every network
  - current best empirical diagnostic baseline

### Second observation check

- `redu10`: `152524`, intended as `redu09`-like independent check
  - the downloaded copy present during analysis was not actually map-enabled
  - however, the null audit was still useful
  - contamination pattern was consistent with `151930`
    - `nw2` still main coherent-residual family
    - `nw4` still main heavy-tail family

## Network-by-Network Interpretation

This is the current best working model for `a1100` (`nw0-5`):

- `nw2`
  - strongest coherent residual structure
  - not especially dominant in low-band MP atmosphere-like mode counts
  - likely not “just atmosphere”
  - more likely residual subgroup coherence / electronics/common-mode structure that fixed whole-network PCA can miss

- `nw4`
  - strongest non-Gaussian / heavy-tail / impulsive behavior
  - localization suggested many single-detector offenders rather than a clean coherent network mode
  - stronger despiking helped here

- `nw0`
  - more of a low-frequency / scan-synchronous leakage case
  - not the worst detector-correlation network

- `nw1`
  - intermediate case, with noticeable low-frequency contamination

- Overall
  - contamination is distributed
  - it is not one catastrophic network
  - it is not one bad observation
  - repeated scan geometry and repeated detector/network behavior can project non-astronomical residuals to the same sky locations over many observations

## MP Prototype Results

The prototype MP estimator in `tools/blank_sky/mp_mode_estimator.py` was run before integrating the algorithm into Citlali.

Main takeaways:

- low-band MP (`0.05 - 0.5 Hz`) produced small adaptive cuts, roughly `k_mp ~ 2-5`
- those results were fairly consistent between `151930` and `152524`
- full-band MP produced very large cuts, roughly `k_mp ~ 70-112`
- therefore:
  - low-band MP is promising as an atmosphere/common-mode selector
  - full-band MP is too broad for direct use as “the” cleaning rule
  - MP should not replace despiking or other RTC-side cleanup

This is why the first integrated PTC implementation uses optional band-limited covariance.

## Citlali Code State

### New blank-sky tools in repo

These were added earlier in this debugging thread:

- `tools/blank_sky/blank_sky_null_audit.py`
- `tools/blank_sky/localize_detector_clusters.py`
- `tools/blank_sky/mp_mode_estimator.py`
- `tools/blank_sky/README.md`

### New PTC cleaner structure

The current in-progress cleaner design in code is:

- `standard_pca`
- `null_model`
- `marchenko_pastur`

with the intent that exactly one is enabled when `clean.enabled: true`.

Relevant files:

- `/Users/gwilson/GitHub/citlali/include/citlali/core/timestream/ptc/clean.h`
- `/Users/gwilson/GitHub/citlali/include/citlali/core/timestream/ptc/ptcproc.h`
- `/Users/gwilson/GitHub/citlali/include/citlali/core/engine/engine.h`
- `/Users/gwilson/GitHub/citlali/data/config.yaml`

### What was implemented

In `clean.h`:

- added `MarchenkoPasturOptions`
- added `StandardPCAOptions`
- added `sample_rate_Hz` to the cleaner
- added helper methods for:
  - active cleaner label
  - per-group MP activation
  - hard cleaner failures via thrown runtime error
- added an MP-based adaptive mode selector:
  - robust centering
  - robust scaling
  - optional low-band covariance via FFT filtering
  - covariance eigenspectrum
  - MP bulk fitting from quantile matching
  - adaptive `k_mp`

In `ptcproc.h`:

- added parsing of `clean.standard_pca.*`
- added parsing of `clean.marchenko_pastur.*`
- kept legacy top-level `stddev_limit`, `n_calc`, `n_eig_to_cut` as fallback parse paths for compatibility
- enforced that exactly one cleaner is enabled
- routed runtime mode selection to:
  - standard fixed/stddev PCA
  - `null_model`
  - `marchenko_pastur`

In `engine.h`:

- passed `telescope.d_fsmp` into `ptcproc.cleaner.sample_rate_Hz`
- added provenance keys:
  - `CONFIG.CLEANED.MODESEL`
  - `CONFIG.CLEANED.MP.ENABLED`
  - `CONFIG.CLEANED.MP.BANDLOW_HZ`
  - `CONFIG.CLEANED.MP.BANDHIGH_HZ`
  - `CONFIG.CLEANED.MP.MAXMODES`

In `data/config.yaml`:

- added `standard_pca` block
- added `marchenko_pastur` block
- documented that exactly one cleaner block should be enabled

## Important Behavior Decisions

### Catastrophic failure policy

The user explicitly asked that cleaner failures should be catastrophic, not silent. The code was changed in that direction:

- `null_model` failures now throw
- `marchenko_pastur` failures now throw
- this avoids silently producing an ill-defined output

This means there is no current fallback-to-fixed-cut behavior if MP fails. That was deliberate after the user request.

### Packaging direction

The user asked whether the cleaners should be packaged cleanly and separately. The answer was yes, and the code was moved toward:

```yaml
clean:
  standard_pca:
    enabled: false
    ...
  marchenko_pastur:
    enabled: true
    ...
  null_model:
    enabled: false
    ...
```

There is still an argument that a single selector key like

```yaml
clean:
  method: standard_pca | null_model | marchenko_pastur
```

would be even cleaner than multiple booleans. That has not been implemented yet.

## Example MP Config Intended for the Next Test

The intended first MP test configuration is:

```yaml
processed_time_chunk:
  clean:
    enabled: true
    mask_radius_arcsec: 0
    tau: 0.0
    grouping: [nw]

    standard_pca:
      enabled: false
      stddev_limit: 0
      n_calc: 0
      n_eig_to_cut:
        a1100: [18]
        a1400: [8]
        a2000: [8]

    marchenko_pastur:
      enabled: true
      min_good_frac: 0.8
      max_modes: 64
      max_samples: 20000
      band_low_Hz: 0.05
      band_high_Hz: 0.5
      clip_z: 12.0
      bulk_keep_frac: 0.8
      q_grid_size: 64
      grouping: [nw]

    null_model:
      enabled: false
```

## Known Caveats / Unresolved Items

1. Build verification has not been done yet.

- The shell used in this Codex session did not have `cmake` on `PATH`.
- Because of that, the new cleaner code has not been compile-verified in this session.
- A real build is the first required next step on a machine with the full toolchain.

2. The FITS/netCDF `CONFIG.CLEANED.NEIG*` provenance keys still reflect configured fixed cuts, not the actual per-scan adaptive `k`.

- The new provenance keys do record the cleaner method and MP config.
- But the actual chosen `k_mp` is not yet written out as a per-scan diagnostic.
- Adding per-scan adaptive-`k` diagnostics would be a good next improvement.

3. The cleaner packaging is improved but not perfect.

- It now supports separate `standard_pca`, `null_model`, and `marchenko_pastur` blocks.
- A future `clean.method` selector would likely be cleaner than the current “exactly one enabled” pattern.

4. The MP implementation was designed from the Python prototype, but numerical behavior still needs real reduction validation.

- especially for low-rank / narrow-band covariance edge cases
- especially to confirm the FFT inverse call compiles as written with the project’s Eigen version

## Recommended Immediate Next Steps

1. Build Citlali on a machine with the actual toolchain.

- confirm the new cleaner code compiles
- fix any Eigen FFT or template issues that appear

2. Run a short MP test modeled on `redu04`.

- single observation
- `151930`
- `a1100` focus
- `grouping: [nw]`
- `marchenko_pastur.enabled: true`
- low-band covariance `0.05 - 0.5 Hz`

3. Audit the MP-cleaned PTC output with the existing blank-sky tools.

- compare against `redu09`
- check whether `nw2` coherence improves without reintroducing `null_model`-style over-cleaning
- check whether `nw4` remains dominated by impulsive behavior, as expected

4. Repeat on `152524`.

- this is the independent-observation consistency check

5. Only after the timestream null metrics look good, run map-enabled tests again.

- likely `naive`
- filtered/Wiener products on

## Recommended Starting Files for the Next Codex Session

If a new session has to restart quickly, open these first:

1. `/Users/gwilson/GitHub/citlali/DEBUG_NOTES_2026-03-13_GOODSN_MP_HANDOFF.md`
2. `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/DEEP_DIVE_2026-03-11.md`
3. `/Users/gwilson/GitHub/citlali/include/citlali/core/timestream/ptc/clean.h`
4. `/Users/gwilson/GitHub/citlali/include/citlali/core/timestream/ptc/ptcproc.h`
5. `/Users/gwilson/GitHub/citlali/tools/blank_sky/mp_mode_estimator.py`

