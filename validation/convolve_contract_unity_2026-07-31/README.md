# Convolve statistical-contract Unity handoff

This bundle freezes the human-run Unity evidence request for implementation
commit `b294802a5e339f9ba5e0e323980cec3a4bd00249`. It does not record scientific
acceptance. Codex must not connect to Unity; the authorized operator uses the
SSH alias `unity_toltec` and returns the evidence listed in `manifest.yaml`.

## Runs

Use one clean Release executable built from the exact candidate SHA for both
runs. Verify its SHA256 and record the compiler, CMake flags, dependency SHAs,
and clean source state before execution. The retained merged point config is:

```text
/work/toltec/commissioning2025-test/2026-refactor/point/refactor/reduced/redu64/citlali_o152389_0_2_c1.yaml
```

Its required SHA256 is
`b494d8671fb162f47d6eaadc1299755594db5c8c27353d7e8b4ac8ffe8566ed8`.
Copy the two overlay files byte-for-byte to Unity and record their remote
paths and hashes. The low-level CLI applies YAML files left-to-right, so run:

```text
CITLALI_EXE BASE_CONFIG noise_products_overlay.yaml disable_filter_overlay.yaml
CITLALI_EXE BASE_CONFIG noise_products_overlay.yaml
```

The first is the raw control and the second is the unit-sum convolve
candidate. The control overlay disables both map filtering and its dependent
source-finding stage; pointing-mode source fitting remains active on the raw
map. The retained base config has `runtime.use_subdir: yes`; confirm that each
invocation acquires a new reduction directory and record both exact IDs and
paths. Never overwrite `redu64` or another retained reduction.

## Exact implementation checks

For each array, inventory every HDU named by the manifest and verify its
TYPE, BUNIT, estimator, filter, conditioning, covariance, calibration, and
fruit-loop-feedback metadata. Independently recompute the ten-realization
central sample variance with `ddof=1`, its square-root uncertainty, direct
amplitude/uncertainty S/N, the global empirical weight scale, and the
conditional formal diagonal variance obtained from the exact squared signal
coefficients and strict `1e-6` relative numerical guard. The filtered noise
realizations must use the same fixed mask, fill, edge window, and convolution
operator as the science amplitude.

Compare only the raw products between the control and candidate, exactly:

```text
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  CONTROL_REDU CONVOLVE_REDU \
  --mode point \
  --include '*/raw/*' \
  --include-timestream \
  --max-array-elements 0 \
  --atol 0 \
  --rtol 0 \
  --strict
```

Also run the normal product inventory, provenance, configuration, log, and
reduction-audit checks with no skipped required data and no unexpected
error-level messages.

## Scientific evidence, not yet acceptance

Return formal-versus-empirical residuals, covariance/correlation maps,
histograms and PSDs, edge-distance tables, false-S/N exceedances, and
interior/edge ratios for all arrays. Blank controls and source injections are
still required to decide the conditional median-fill approximation, response
normalization, support/confidence definition, and multi-pixel covariance
contract. No numerical acceptance bounds have been approved here; they must
be preregistered before interpreting those experiments. Until that review,
unit-sum filtered products remain fail-closed for fruit-loop feedback and the
topic commit must not be integrated.
