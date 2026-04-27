# Beammap Prior / Fitting Audit Handoff - 2026-04-26

This note captures the beammap pipeline review, implementation work, and first
3c273 test reduction from 2026-04-26. It is intended as a next-morning handoff
for continuing the work on another machine.

## Scientific Goal

The beammap pipeline should make per-detector maps of a point source for each
detector in all three TolTEC arrays, then extract:

- detector-relative PSF center positions,
- PSF shape,
- detector response/amplitude.

The difficult cases are RFI, level shifts, cosmic rays/glitches, atmosphere, and
PCA cleaning. The specific operational failures being targeted are:

- the source itself triggering flagging or de-weighting,
- iterative fitting locking onto a wrong peak,
- unreliable fits after PCA cleaning,
- biased map RMS/SNR estimates from zero-filled unsupported map pixels,
- Gaussian fit residuals ignoring pixel weights.

## Code Changes Made

The working tree was clean when this handoff note was written, so these changes
appear to already be in the current local commit history. Confirm on the work
machine with `git log`/`git show` if needed.

### Hard Beammap Config Errors

Beammap reductions now fail fast for invalid core assumptions:

- beammap source flux must be positive and finite for all arrays,
- beammap map pixel axes must be `altaz`.

Rationale: a zero-flux beammap reduction is scientifically unusable, and
beammapping only makes sense in the alt/az frame.

### Detector Weighting Mode

Added `beammap.detector_weighting.mode` with modes:

- `const`,
- `ptc`,
- `ptc_after_iter0`.

The 3c273 test used `ptc_after_iter0`, so iteration 0 stays permissive and later
iterations use detector-mode PTC weights with source-aware masking.

### Source-Aware PTC Weight Masking

PTC weight calculation can now use previous beammap fit centers even when fruit
loops is disabled. This prevents the source itself from inflating noise estimates
or causing source deweighting after there is a previous fit.

Important naming caveat: this reuses fields named
`fruit_loops_source_lat/lon/valid` as plumbing. In the 3c273 test, fruit loops
was disabled; those fields were populated from previous Gaussian fit centers.

### Weighted Gaussian Fit Residuals

Beammap Gaussian fitting now weights residuals by positive map weights. The map
sigma is estimated on positive-weight support pixels, and each residual is
scaled by the relative pixel weight.

Rationale: zero-filled unsupported map pixels and nonuniform coverage should not
drive the fit.

### Support-Only RMS/SNR QC

Map RMS/SNR QC now uses positive-weight support rather than the whole
zero-filled image. Where possible, the fitted source core is excluded from the
RMS estimate.

Rationale: previous S/N values were biased by unsupported map pixels.

### Peak-Lock Guard

Iterative initialization now compares the previous-iteration seed against the
current weighted peak. If a much stronger peak appears far from the previous
seed, the previous seed can be rejected. This is guarded by prior compatibility:
when priors are available, a prior-compatible previous seed should not be
displaced by a prior-incompatible weighted peak.

### Dynamic / Empirical Priors

Added prior configuration and code paths for:

- different prior strength on iter 0 vs later iterations,
- per-array empirical translation after iter 0,
- optional residual rotation fit,
- prior-slot matching in a transformed prior frame,
- diagnostics in fit-QC metadata.

The soft prior catalog used in the test was:

`data/beammap_priors/beammap_slot_priors_soft_v1.ecsv`

New config controls include:

- `max_d2_iter0`,
- `max_d2_after_iter0`,
- `score_lambda_iter0`,
- `score_lambda_after_iter0`,
- `align_after_iter0`,
- `alignment_min_matches`,
- `alignment_max_d2`,
- `alignment_fit_rotation`,
- `alignment_max_rotation_deg`.

### Build Fix

A const-correctness build error was fixed in the prior elevation helper. The
const method no longer uses `operator[]` on `telescope.tel_data`; it uses
`find("TelElAct")` and falls back to zero elevation if missing.

### flag2 Metadata

The `flag2` metadata legend was corrected to describe bit values rather than
ordinal values:

- `Good=0`,
- `BadFit=1`,
- `AzFWHM=2`,
- `ElFWHM=4`,
- `Sig2Noise=8`,
- `Sens=16`,
- `Position=32`,
- `PriorDist=64`,
- `NetworkPos=128`.

## Test Reduction

Test source: `3c273`

Obsnum: `152307`

Generated/used config:

`~/work_toltec/local_data/beammaps/3c273/70_reduce.yaml`

Reduction output inspected:

`~/work_toltec/local_data/beammaps/3c273/reduced/redu02`

Use the TolTEC venv for local Python checks:

`~/toltec/bin/python`

The test config used:

- `beammap.detector_weighting.mode: ptc_after_iter0`,
- priors enabled,
- `beammap.flagging.max_prior_d2: 0.0` so prior-distance diagnostics were
  recorded but not used for flagging in the first test,
- source regime set to `source_dominant`,
- 3c273 fluxes from APT metadata:
  - a1100: `3896.3679050705327`,
  - a1400: `4512.609202551715`,
  - a2000: `5609.315775446418`.

## Reduction Runtime

The 3c273 reduction took about `78 min` wall time.

From `citlali.log.gz`:

- pre-iteration setup: `3.7 min`,
- beammap iterative loop: `72.3 min`,
- final sensitivity/QC/writeout: `1.9 min`.

Inside the iterative loop:

- scan-level processing before fitting: `69.2 min`,
- Gaussian/prior init/fitting/QC: `3.2 min`.

The bulk of the time is not Gaussian fitting or prior search. It is repeated
per-scan processing:

- TOD copy,
- Gaussian subtract/add for iterations greater than 0,
- PTC/PCA cleaning,
- detector-mode PTC weight calculation,
- mapmaking.

Counts from the run:

- `199` scans,
- `10` beammap iterations,
- `1990` scan-processing passes,
- `1791` detector-mode PTC weight recalculations (`199` scans times 9 post-iter0
  iterations).

The last five iterations alone cost roughly `34 min`.

## Iteration Type

For the 3c273 `redu02` run, the beammap iteration was Gaussian fit
subtraction/add-back, not fruit loops.

The loop is:

1. Iteration 0 has no source subtraction.
2. Make detector maps after cleaning.
3. Fit one Gaussian PSF per detector.
4. For later iterations, copy original TOD chunks.
5. Subtract the previous iteration's per-detector Gaussian model from the TOD.
6. Run PTC/PCA cleaning.
7. Add the same Gaussian model back.
8. Remake maps and refit.
9. Stop when convergence passes or `beammap.iter_max` is reached.

Fruit loops was disabled in the test config:

```yaml
timestream:
  fruit_loops:
    enabled: no
```

If fruit loops is enabled, the loop subtracts/adds a map model instead of the
Gaussian model, and convergence is based on map changes rather than Gaussian
parameter changes.

## Priors / Alignment Observations

The prior system behaved mostly as intended in the first test.

Iteration 0:

- prior-seeded maps: `5465`,
- blind maps: `26`.

After iteration 0:

- all three arrays got empirical alignment,
- no blind fallback after iteration 3,
- final prior distances were tight.

Final median `final_prior_d2`:

- all detectors: `0.245`,
- final-good detectors: `0.221`,
- final-flagged detectors: `0.521`.

### Important Open Concern: Array Transforms

Wilson noted at the end of the day that the transformations for the three arrays
should be the same because the arrays are co-located. The test logs showed
different fitted per-array empirical alignments, for example at iteration 9:

- a1100: `dx=-10.5 arcsec`, `dy=-0.12 arcsec`, `rot=-1.05 deg`,
  RMS `14.4 arcsec`,
- a1400: `dx=+9.9 arcsec`, `dy=+0.09 arcsec`, `rot=-0.79 deg`,
  RMS `13.8 arcsec`,
- a2000: `dx=+1.0 arcsec`, `dy=+0.04 arcsec`, `rot=-0.37 deg`,
  RMS `24.4 arcsec`.

This is suspicious. Possible explanations to investigate:

- the per-array center estimate is absorbing array-specific catalog bias,
- the prior catalog is centered per array, not in one common focal-plane frame,
- raw vs derotated coordinate handling differs somewhere in the matching path,
- network/array geometry priors have residual offsets from the biased historical
  library products used to build them,
- the empirical alignment should be solved as one common transform using all
  arrays, perhaps with per-array fixed geometry offsets, rather than three
  independent transforms.

Do not assume the current per-array transform is physically correct. This should
be one of the first next checks.

## Test Product Comparison

Compared new `redu02` fit-QC against the existing `_apt_library` product for the
same obsnum.

Final-good fractions:

| product | total good | a1100 | a1400 | a2000 |
|---|---:|---:|---:|---:|
| new `redu02` | `4767/5491 = 86.8%` | `81.8%` | `92.2%` | `95.8%` |
| old `_apt_library` | `4937/5491 = 89.9%` | `89.5%` | `87.6%` | `94.5%` |

The new reduction improves a1400/a2000 but loses many a1100 detectors.

Transitions by detector:

- both good: `4580`,
- new only good: `187`,
- old only good: `357`,
- both bad: `367`.

The loss is almost entirely a1100:

- old-good/new-bad total: `357`,
- a1100 old-good/new-bad: `336`.

Among those a1100 old-good/new-bad detectors, new flags were mostly FWHM:

- `232` AzFWHM,
- `105` ElFWHM,
- `11` Sig2Noise,
- `3` Position,
- `4` NetworkPos.

Prior-distance flags were disabled in this test and did not drive the losses.

## a1100 FITS Inspection

After the a1100 FITS files were downloaded, the split products were inspected:

- `toltec_commissioning_a1100_beammap_152307_citlali_flag0_good.fits`,
- `toltec_commissioning_a1100_beammap_152307_citlali_flag1_bad.fits`.

The FITS maps line up with `x_t_raw/y_t_raw`, with the FITS x-axis flipped
because `CDELT1=-1`. Using derotated `x_t/y_t` makes the source appear
misplaced in FITS pixel coordinates.

For the `336` a1100 old-good/new-bad detectors:

- median local source S/N near the fitted center: `38.7`,
- median fit-to-local-peak distance: `1.54 arcsec`,
- median center/local-peak ratio: `0.86`,
- median prior distance: `0.31`,
- median fitted widths:
  - `a_fwhm=10.5 arcsec`,
  - `b_fwhm=7.23 arcsec`.

For the a1100 AzFWHM subset:

- median local source S/N: `38.6`,
- median fit-to-local-peak distance: `1.29 arcsec`,
- median center/local-peak ratio: `0.891`,
- median `a_fwhm=11.2 arcsec`,
- median `b_fwhm=6.95 arcsec`.

Conclusion: the a1100 losses are mostly not wrong-peak failures. The source is
usually centered and detected with good S/N, but the Gaussian fit is broad
relative to the current a1100 upper FWHM threshold (`10 arcsec`) and many fits
hit the fit upper bound (`15 arcsec`).

This is probably a fit-model/QC-threshold/residual-wing question rather than a
prior-search question.

## Known Oddities / Bugs To Follow Up

### Flag Count Log Timing

The log said `709 detectors were flagged`, while the final fit-QC table had
`724` flagged. The difference is exactly the `15` `NetworkPos` flags, so the log
line appears to be emitted before network-position flags are added.

### Convergence Semantics

`386` detectors ended with `converged=0`, but many were still final-good. The
loop ran to `iter_max=10`.

This suggests convergence should be revisited:

- convergence may be too strict,
- convergence may include parameters that do not matter for final science,
- convergence may need to ignore final-bad detectors,
- the loop could stop when the good detector set and prior frame are stable.

### Efficiency

The expensive path is repeated scan processing, especially detector-mode PTC
weight recalculation every post-iter0 iteration.

Promising speedups:

- stop earlier once fit/prior state stabilizes,
- freeze detector PTC weights after the source-mask centers stabilize,
- add fine-grained timers around:
  - Gaussian subtract/add,
  - `ptcproc.run`,
  - detector-mode weight calculation,
  - map projection,
  - Gaussian fitting.

## Suggested Next Steps

1. Investigate why empirical transforms differ by array even though arrays are
   co-located. Consider fitting one common transform across all arrays.

2. Make a small montage of representative a1100 maps:
   - both-good,
   - old-good/new-bad high-S/N centered broad fits,
   - old-good/new-bad weak/far local detections,
   - both-bad.

3. Decide whether the a1100 upper FWHM QC threshold of `10 arcsec` is still
   appropriate after weighted residual fitting and support-only S/N.

4. Check whether Gaussian fits are being broadened by residual PCA wings or
   atmospheric structure. If so, consider a fit window, robust loss, background
   term, or PSF/template model rather than a free broad Gaussian over the current
   support.

5. Add detailed timing instrumentation before trying to optimize further.

6. Consider lowering `beammap.iter_max` for test reductions or adding a stable
   prior-frame/good-set early stopping criterion.

