# Citlali Branch Audit - 2026-05-14

Audit target: `gw_dev` in `/Users/gwilson/GitHub/citlali`.

This report records the read-only audit results from the branch review so the
findings can be prioritized and closed incrementally.

## Executive Summary

Top risks found:

1. Non-`mJy/beam` calibration was likely wrong because per-detector flux
   conversion factors were later indexed by array id.
2. `uK` conversion appeared physically wrong: beam area was commented out, and
   callers passed inconsistent FWHM units.
3. Negative source finding could silently flip output maps if no source
   survived detection.
4. Polarimetric HWPR interpolation could use the telescope sample count instead
   of the HWP sample count when `interp_over_gaps=false`.
5. Test coverage was far below the risk level of this codebase; current tests
   did not protect calibration, alignment, mapmaking, metadata, or
   branch-specific features.

Overall confidence: medium-high for the static findings, medium for end-to-end
science safety because no TolTEC regression products were reduced during the
audit.

Merge assessment at audit time: targeted follow-up required before treating the
branch as production-safe for calibrated science products.

## Repository Map

- `CMakeLists.txt`: CMake project, static library, CLI executable.
- `src/citlali/cli/main.cpp`: CLI orchestration, config merge, reduction
  dispatch.
- `include/citlali/core/engine`: reduction engine, IO model, calibration,
  telescope alignment, beammap/pointing/science pipelines.
- `include/citlali/core/timestream`: RTC/PTC processing, filtering, despiking,
  downsampling, calibration, cleaning.
- `include/citlali/core/mapmaking`: naive, jinc, ML, Wiener, map buffers and
  products.
- `data/config.yaml`: default runtime/science configuration embedded into
  generated `default_config.h`.
- `tests`: one GTest target with minimal smoke coverage.

Inferred data flow:

```text
YAML config + SeqIO rawobs
  -> KIDs raw netCDF + telescope netCDF + optional HWPR + APT/calibration ECSV
  -> Calib/Telescope setup
  -> time alignment and optional gap interpolation
  -> scan chunking
  -> KidsDataProc raw TOD solve
  -> RTCProc: despike/filter/downsample/calibrate/extinction/polarization/kernel diagnostics
  -> PTCProc: cleaning, flags, weights, sensitivity, optional fruit-loops subtraction/addback
  -> mapmaker: naive / jinc / maximum_likelihood
  -> MapBuffer normalization, PSD/hist/noise/source products
  -> optional coadd and filtering
  -> FITS, netCDF TOD, stats, diagnostics, ECSV products
```

## Branch-Specific Review

`git merge-base HEAD origin/master` and `git merge-base HEAD origin/v4.x` did
not return a merge base, so normal three-dot branch review was unavailable.
Tree-level two-dot comparisons showed:

- Versus `origin/v4.x`: 147 files changed, 49,303 insertions, 2,384 deletions.
- Versus `origin/master`: 187 files changed, 77,880 insertions, 370 deletions.

Highest-risk changed areas:

- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/todproc.h`
- `include/citlali/core/engine/kidsproc.h`
- `src/citlali/core/engine/calib.cpp`
- `include/citlali/core/timestream/rtc/calibrate.h`
- `include/citlali/core/utils/utils.h`
- `src/citlali/core/mapmaking/map.cpp`
- `include/citlali/core/engine/beammap.h`

## Findings

| ID | Severity | Confidence | Evidence | Problem | Recommended fix | Test |
| --- | --- | --- | --- | --- | --- | --- |
| F-001 | P1 | High | `src/citlali/core/engine/calib.cpp:167`, `include/citlali/core/timestream/rtc/calibrate.h:181` | `calc_flux_calibration()` created per-detector factors, but `calibrate_tod()` read them by array id. | Use detector index and validate vector length. | Synthetic multi-array APT with distinct FWHMs. |
| F-002 | P1 | High | `include/citlali/core/utils/utils.h:77`, `src/citlali/core/engine/calib.cpp:201`, `include/citlali/core/engine/engine.h:1825` | `mJy_beam_to_uK()` ignored beam area and callers mixed FWHM units. | Define FWHM in arcsec, include beam solid angle, use one helper consistently. | Analytic `dB/dT * Omega_beam` checks. |
| F-003 | P1 | High | `src/citlali/core/mapmaking/map.cpp:790`, `:827`, `:871`, `:948`, `:953` | Negative source finding negated maps in place and early-returned without restoring sign. | Use non-mutating local signal or guaranteed restoration. | Negative-mode no-detection map remains unchanged. |
| F-004 | P2 | High | `src/citlali/core/mapmaking/map.cpp:840`, `:851` | Source finder searched edge neighborhoods without bounds checks. | Clamp neighbor windows. | Edge hot-pixel map under ASan or no-crash test. |
| F-005 | P1 | High | `include/citlali/core/engine/todproc.h:684`, `:712`, `:717` | No-gap HWPR interpolation reused telescope length for HWP arrays. | Use `hwpr_recvt.size()` and validate HWP angle/time lengths. | Telescope and HWP streams with different lengths. |
| F-006 | P2 | High | `include/citlali/core/engine/engine.h:1970` | TOD netCDF wrote `JINC_A/B/C` all from shape parameter index 0. | Write B from index 1 and C from index 2. | Metadata parity test for JINC params. |
| F-007 | P2 | High | `include/citlali/core/engine/engine.h:1271`, `include/citlali/core/engine/config.h:23` | Beammap prior quantile intended upper bound was passed as a second min value. | Pass max value via `max_val`. | Config with quantile `0.9` fails. |
| F-008 | P2 | High | `include/citlali/core/engine/engine.h:1300`, `:1321`, `:1338`, `:965` | Fixed-length config vectors were indexed without length validation. | Add exact-length config helper. | Short vector config fails before reduction. |
| F-009 | P2 | Medium-high | `src/citlali/core/engine/calib.cpp:247`, `:261`, `:302`, `:366` | APT grouping assumed sorted contiguous rows and divided by zero for all-flagged groups. | Validate grouping or group by explicit index lists; reject zero-good groups. | Shuffled and all-flagged APT fixtures. |
| F-010 | P2 | Medium-high | `include/citlali/core/engine/todproc.h:864`, `:869`, `include/citlali/core/engine/kidsproc.h:462`, `:467` | Time alignment and scan extraction had weak overlap/bounds checks. | Validate overlap and index bounds. | No-overlap and boundary-gap fixtures. |
| F-011 | P2 | High | `include/citlali/core/timestream/rtc/despike.h:1035`, `:1039`, `:1261` | Despike replacement used random-device seeding with no configured seed/provenance. | Add deterministic configurable seed or deterministic replacement. | Same despike-enabled reduction twice. |
| F-012 | P2 | High | `tests/test_utils.cpp:11`, `:21` | Test suite was mostly smoke tests with no science-path coverage. | Add focused unit and synthetic regression tests. | Calibration, alignment, source, metadata, and E2E tests. |

## Assumptions Needing Domain Review

- Exact `uK` convention: thermodynamic CMB vs Rayleigh-Jeans, beam vs
  steradian, and TolTEC bandpass assumptions.
- Tangent-plane sign conventions for RA/Dec, alt/az, galactic, parallactic
  angle, and detector offset rotation.
- Whether APT rows are guaranteed sorted by network and array.
- Whether `runtime.interp_over_gaps=false` is supported with HWPR/polarimetry.
- Scientific acceptability of random-noise spike replacement.

## Highest-Value Tests

1. Calibration unit tests for `mJy/beam`, `MJy/sr`, `uK`, and `Jy/pixel`.
2. Analytic `mJy/beam` to `uK` checks for all TolTEC bands.
3. Source finder tests for negative mode and edge pixels.
4. HWPR no-gap alignment with different telescope/HWP sample counts.
5. Config validation tests for fixed-vector lengths and quantile bounds.
6. APT grouping tests for sorted, shuffled, and all-flagged inputs.
7. Time-overlap tests for non-overlapping interfaces.
8. JINC metadata parity between netCDF and FITS.
9. Despike reproducibility with fixed seed.
10. One small end-to-end synthetic reduction.

## Quick Wins

- Fix detector-indexed flux conversion.
- Standardize `mJy_beam_to_uK()` on FWHM arcsec and include beam area.
- Make source finding non-mutating and edge-safe.
- Fix HWPR interpolation length in no-gap alignment.
- Correct JINC metadata indices.
- Add fixed-length config validation.
- Remove tracked `.DS_Store` and `__pycache__/*.pyc`.
