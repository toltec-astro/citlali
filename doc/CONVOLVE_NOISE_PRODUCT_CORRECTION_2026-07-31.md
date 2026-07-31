# Convolve Noise-Product Correction

Date: 2026-07-31  
Branch: `codex/convolve-noise-correction`  
Base: `codex/refactor-mainline` at `9aae0e669384c5c0c0dda93debc194d6b8dac787`  
Status: local gates complete; Unity validation pending; not accepted on application mainline

## Scope

This is a bounded numerical and product-semantics correction for post-processing
that executes the fixed unit-sum convolution path: explicit `convolve` and
Wiener `lowpass_only`. It does not change the convolved signal, raw maps,
RTC/PTC algorithms, calibration, source fitting, or Conan 2 build adaptation.

## Corrected Contracts

For output pixel `p` produced by the fixed convolution

`y_p = sum_i k_i x_(p-i)`, with `sum_i k_i = 1`,

and diagonal input covariance, the propagated variance is

`Var(y_p) = sum_i k_i^2 Var(x_(p-i))`.

The former implementation divided this result by the locally valid
`sum(k^2)`. On uniform coverage that cancellation left the formal weight
unchanged by smoothing. The correction retains the valid-support check but
uses the unnormalized `k^2` convolution as the output variance.

For `N` empirical noise realizations with an estimated mean, the persisted
central sample variance is

`V = sum_j (n_j - mean(n))^2 / (N - 1)`.

Persisted empirical uncertainty products therefore require `N >= 2`.
The distinct known-zero-mean second moment `sum_j n_j^2 / N` remains available
to library callers that explicitly disable mean subtraction.

## Product Interpretation

The convolved signal and its jackknife uncertainty retain the map signal unit,
currently `mJy/beam` in accepted products. Unit-integral smoothing does not
apply a point-source response normalization. To preserve the current FITS HDU
shape, the historical `point_source_*` aliases remain, but convolve-mode
metadata identifies them as `convolved_amplitude`, describes the lack of
point-source response normalization, and omits the false `RESPNORM=1.0` claim.
Their ratio remains dimensionless.

## Expected Numerical Effects

- Convolved `weight_formal` changes wherever fixed convolution is active.
  With uniform input variance, `W_out = W_in / sum(k^2)`.
- Mean-subtracted `noise_variance` is multiplied by `N/(N-1)` relative to the
  previous estimator. Its uncertainty increases by `sqrt(N/(N-1))`, and the
  corresponding empirical S/N decreases by the reciprocal factor.
- Empirical weighting remains a scalar calibration of formal spatial weight;
  this change does not redefine it as pixelwise inverse jackknife variance.
- One-realization uncertainty-product requests fail validation instead of
  publishing zero variance.
- Convolved signal samples are unchanged.

## Validation Plan

Required local evidence:

1. Synthetic sequential and OpenMP tests for uniform diagonal variance and
   incomplete valid support.
2. Synthetic jackknife tests for `N - 1`, the `N = 1` failure boundary, and
   the explicit known-zero-mean second moment.
3. Metadata tests for the unnormalized convolved-amplitude aliases.
4. `citlali_cli` build, complete CTest, complete configuration preflight,
   baseline-tool tests, and validation-ledger checks.

Required operational evidence before mainline integration:

1. A same-SHA Unity point or OOF run with fixed convolution, empirical noise
   products, and at least five realizations.
2. Zero unexpected error-level messages and complete provenance/product
   contracts.
3. Exact raw-map agreement with the control and expected changes confined to
   filtered formal weights, empirical variance/weight/SNR products, and the
   corrected metadata.
4. A science-change-ledger entry linked to the accepted run. The immutable
   historical accepted run remains evidence of the former behavior and is not
   relabeled.

## Local Validation Snapshot

Both `CITLALI_USE_WIENER_FILTER_OMP=OFF` and `ON` configurations built the CLI
and test target. In each configuration all 541 enabled CTests passed; one
pre-existing lifecycle sequence test remained explicitly disabled. The
focused convolve/noise/metadata subset passed in both configurations. The
complete configuration preflight passed 123 tests, the baseline-tool suite
passed 135 tests, and the validation and accepted-science-change ledgers both
validated. The normal local build cache was restored to the prior
`CITLALI_USE_WIENER_FILTER_OMP=OFF` setting after OpenMP validation.
