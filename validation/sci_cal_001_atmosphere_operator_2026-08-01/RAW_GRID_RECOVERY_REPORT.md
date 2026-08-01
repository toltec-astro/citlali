# SCI-CAL-001 recovered raw atmosphere-grid evidence

## Recovery result

The complete local q25, q50, and q75 NPZ grids are identified by both SHA-256 and the MD5 values in TolTECA's static-data registry. The q95 bytes are not local. No remote endpoint or Unity system was contacted.

Each recovered NPZ has 31 elevations from 20 through 80 degrees in 2-degree steps and 50,001 frequency samples from 0 through 500 GHz in 0.01-GHz steps. It contains dimensionless transmission (`atmTtx`) and Rayleigh-Jeans atmosphere temperature (`atmTRJ`).

## Exact legacy-fit reconstruction

For every q25/q50/q75 model and TolTEC band, the current Citlali coefficients are reproduced exactly after eight-decimal rounding by fitting degree six in elevation radians to:

```text
atmTtx(nu_band, elevation) / atmTtx(225.00 GHz, elevation)
```

with `nu_band = 272.73, 214.29, 150.00 GHz` for a1100, a1400, and a2000. All 31 raw elevation nodes participate in `numpy.polyfit`. The legacy operator is therefore monochromatic at recovered nominal-wavelength frequencies; no TolTEC passband is integrated in this lineage.

The exact 80-degree 225-GHz raw transmissions and source-derived selector tau225 values are:

The selector coordinate is zenith optical depth: `tau225 = -log(T225 at 80 deg) / X(80 deg)` with the repair-base modified-secant airmass, not the unscaled 80-degree slant optical depth.

| Model | T225 at 80 deg | selector tau225 |
| --- | ---: | ---: |
| `am_q25` | `0.9500275` | `5.04874104674104401e-02` |
| `am_q50` | `0.9142065` | `8.83393725904400573e-02` |
| `am_q75` | `0.8515054` | `1.58313198574890929e-01` |

The unrounded-to-source coefficient differences are all below half of the final decimal unit, and every rounded coefficient is exactly equal to the repair-base literal. `recovered_fit_coefficients.csv` preserves all 63 comparisons.

## Raw-node representation fidelity

`raw_anchor_fit_metrics.csv` isolates the degree-six transmission-ratio fit from the 225-GHz slant-path reconstruction. Its worst fractional correction error is `0.024111%` for `am_q75/a2000`. `raw_anchor_operator_metrics.csv` evaluates the owner-required top-of-atmosphere-pivot, full-sample-airmass anchor reconstruction using the repair-base coefficients. It is not the current application correction, whose missing sample-airmass factor remains separate mandatory repair scope. Its worst raw-anchor correction error is `0.427091%` for `am_q75/a1100`. These are real q25--q75 raw-anchor results, but not the full successor-domain gate.

A post-hoc raw leave-one-model-out check is possible at q50: interpolate raw LOS optical depth between raw q25 and q75 using the exact selector tau225 coordinates, then compare with the recovered raw q50 calculation. q50 was already inspected during provenance recovery, so this is not a preregistered or blinded holdout. Its worst correction error is `0.012264%` in `a1400`. Interpolating the full-airmass q25/q75 anchor reconstructions instead and comparing with raw q50 gives worst error `0.243563%` in `a1400`. Both pass one percent in this single post-hoc q50 check only. They do not validate q75--q95, preregistered intermediate profiles in every interval, or a declared operational domain.

Across the recovered q25/q50/q75 nominal-frequency raw surfaces, `raw_grid_physical_metrics.csv` records 0 increasing-opacity violations and 0 increasing-elevation wrong-way cells at tolerance `1e-12`.

## Provenance still missing

The local evidence names Scott Paine's `am` model and historical LMT percentile grids, but it does not preserve the exact `am` executable/version, atmosphere-profile files, percentile construction, generation command, or site/geometry directives. The q95 request is exactly TolTECA datafile ID `461`, expected MD5 `0ca7b331823237767d26016d19bffb3d`; those bytes must be supplied locally, inspected, and SHA-256 identified before the full operator decision.

The nearby modeled passband artifact and TolTECA's versioned passband tables are not inputs to the recovered Citlali coefficients. A band-integrated successor would be a new, explicitly approved spectral convention, not a faithful rerun of this monochromatic lineage.

## Disposition

This partial recovery materially narrows the owner request, but no final atmosphere operator or operational domain is selected. Full q95 raw evidence, intermediate-opacity generation rules/runs, and the missing generator/profile provenance remain required.
