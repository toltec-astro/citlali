# Full Fruit-Loop Population Analysis

- Expected observations: `108`
- Analyzed observations: `108`
- Missing observations: `none`
- Stage A audited jobs: `16/16`
- Stage B audited jobs: `92/92`
- Per-observation convergence plots: `108`
- Empirical blank-sky point-source S/N included: `true`
- Production stopping policy changed: `false`

## Morphology-aware amplitude-only simulation

| Morphology | Tolerance | Resolved | Median stop | Median absolute residual | P90 residual | Maximum residual |
|---|---:|---:|---:|---:|---:|---:|
| all | 2% | 265/324 | 6.0 | 0.59% | 1.30% | 3.57% |
| all | 2.5% | 287/324 | 6.0 | 0.67% | 1.87% | 3.57% |
| all | 3% | 302/324 | 6.0 | 0.78% | 2.08% | 4.38% |
| all | 4% | 314/324 | 6.0 | 1.15% | 3.04% | 7.09% |
| all | 5% | 320/324 | 6.0 | 1.29% | 4.30% | 7.94% |
| planetary_disk | 2% | 54/99 | 7.0 | 0.64% | 1.45% | 3.08% |
| planetary_disk | 2.5% | 66/99 | 7.0 | 0.74% | 2.24% | 3.13% |
| planetary_disk | 3% | 77/99 | 7.0 | 0.98% | 2.47% | 4.38% |
| planetary_disk | 4% | 89/99 | 7.0 | 1.76% | 4.41% | 7.09% |
| planetary_disk | 5% | 95/99 | 6.0 | 2.39% | 5.74% | 7.94% |
| unresolved | 2% | 211/225 | 6.0 | 0.58% | 1.24% | 3.57% |
| unresolved | 2.5% | 221/225 | 6.0 | 0.66% | 1.64% | 3.57% |
| unresolved | 3% | 225/225 | 6.0 | 0.75% | 1.87% | 3.57% |
| unresolved | 4% | 225/225 | 6.0 | 0.96% | 2.38% | 4.19% |
| unresolved | 5% | 225/225 | 6.0 | 0.97% | 3.39% | 7.12% |

## Trajectory behavior

| Morphology | Array | Amplitude monotonic | Tail positive steps | Endpoint amplitude / seed | Background monotonic | Endpoint background / seed | Formal S/N / seed | Empirical S/N / seed | Legacy dynamic range / seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| planetary_disk | a1100 | 29/33 | 96.2% | 2.40 | 1/33 | 0.99 | 4.19 | 2.78 | 0.79 |
| planetary_disk | a1400 | 25/33 | 87.9% | 2.65 | 0/33 | 0.89 | 4.86 | 3.83 | 1.05 |
| planetary_disk | a2000 | 29/33 | 93.9% | 2.54 | 0/33 | 0.82 | 5.10 | 3.67 | 0.85 |
| unresolved | a1100 | 43/75 | 77.3% | 1.59 | 2/75 | 0.86 | 2.29 | 1.96 | 0.99 |
| unresolved | a1400 | 64/75 | 95.7% | 2.06 | 0/75 | 0.86 | 3.11 | 2.88 | 1.13 |
| unresolved | a2000 | 56/75 | 88.0% | 1.95 | 0/75 | 0.77 | 3.39 | 3.07 | 0.85 |

The formal template-fit S/N rises while the historical peak/full-map-RMS dynamic range can fall. Background sigma is not monotonically increasing; the historical apparent S/N loss is therefore not evidence of worsening source-free noise.

## Planet-disk correction

| Array | Median/max major-axis broadening | Median/max minor-axis broadening | P90/max tail amplitude-metric difference |
|---|---:|---:|---:|
| a1100 | 5.20% / 8.38% | 5.63% / 8.51% | 1.44% / 7.01% |
| a1400 | 3.07% / 5.72% | 3.66% / 6.98% | 1.19% / 3.21% |
| a2000 | 1.88% / 2.58% | 1.97% / 3.06% | 0.99% / 10.69% |

## Candidate V0 all-array simulation

The core rule uses 3% morphology-aware amplitude, 5% morphology-aware FWHM, 0.1 arcsec centroid, 5% successive map change, 5% weight change, less than 1% valid-mask symmetric difference, apply-phase learning, and a background sigma no more than 10% above seed. The strict-state variant additionally requires unchanged effective learning mask/penalty counts.

| Morphology | Resolved observations | Median stop | Amplitude P90/max | Centroid P90/max | FWHM P90/max | Source-aperture residual P90/max |
|---|---:|---:|---:|---:|---:|---:|
| all | 57/108 | 7.0 | 1.31% / 2.52% | 0.016 / 0.043 arcsec | 1.02% / 1.81% | 2.84% / 4.28% |
| planetary_disk | 6/33 | 8.0 | 0.73% / 1.01% | 0.010 / 0.011 arcsec | 0.80% / 0.90% | 1.74% / 2.25% |
| unresolved | 51/75 | 7.0 | 1.37% / 2.52% | 0.017 / 0.043 arcsec | 1.04% / 1.81% | 2.89% / 4.28% |

### Core-rule yield by frozen quality stratum

| Stratum | Resolved observations | Median stop |
|---|---:|---:|
| marginal | 21/38 | 7.0 |
| normal | 35/54 | 8.0 |
| stress | 1/16 | 8.0 |

### Core-rule yield by source

| Source | Resolved observations | Median stop |
|---|---:|---:|
| 3c273 | 46/65 | 7.0 |
| 3c279 | 4/6 | 9.0 |
| 3c345 | 1/2 | 6.0 |
| 3c84 | 0/2 | nan |
| Neptune | 5/16 | 8.0 |
| Uranus | 1/17 | 9.0 |

## Continuation review

`51` observations require continuation or trajectory review under the provisional residual guards.

- Measurement-limited: `23`
- Trajectory unresolved with individually measurable criteria: `28`

Radio sources and planetary disks are assessed independently. Planet amplitudes use each realized kernel convolved with the epoch-specific JPL Horizons uniform disk; planet widths are normalized by that disk-convolved template rather than the bare point-source kernel.
