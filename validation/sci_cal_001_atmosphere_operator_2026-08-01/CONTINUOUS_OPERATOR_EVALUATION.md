# SCI-CAL-001 continuous-operator candidate evaluation

## Status

**No successor operator or operational domain is selected.** Exact q25/q50/q75 raw grids and the legacy monochromatic fit have been recovered, while q95, intermediate-opacity model runs, and the original `am` execution/profile provenance remain missing. This report deliberately uses only the exact repair-base legacy q-model polynomials as surrogate evidence; `RAW_GRID_RECOVERY_REPORT.md` contains the bounded raw-grid checks. The full-domain provisional one-percent fractional extinction-correction fidelity gate cannot yet be evaluated.

The evidence is bound to repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and repair-line evidence head `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`. Frozen input digests are recorded in `legacy_anchor_manifest.json`.

## Diagnostic support

The analysis spans tau225 `0` through the source-derived q95 selector anchor `3.04868387190534607e-01` and elevation `30.0` through `80.0` degrees at `0.1`-degree spacing. This is the phase-0 diagnostic range, not an approved operational domain. Values outside the tau range are undefined and would fail closed.

Source-derived selector anchors are:

| Model | tau225 binary64 |
| --- | ---: |
| `am_q0` | `0.00000000000000000e+00` |
| `am_q25` | `5.04874104674104401e-02` |
| `am_q50` | `8.83393725904400573e-02` |
| `am_q75` | `1.58313198574890929e-01` |
| `am_q95` | `3.04868387190534607e-01` |

## Candidate definitions

All candidates preserve the owner-approved zero-to-q25 operator exactly: LOS optical depth is `(tau225/tau_q25) * LOS_tau_q25(elevation)`. Above q25 they interpolate the exact legacy fitted LOS-optical-depth anchor surfaces in tau225. The candidates are diagnostic `v0` surfaces, not implementation contracts.

- `piecewise_linear_los_tau_v0`: piecewise affine in tau225; minimal and C0.
- `pchip_los_tau_v0`: shape-preserving PCHIP in tau225 through q25--q95; C0 where it meets the fixed low-opacity segment.
- `cubic_through_anchors_los_tau_v0`: one unconstrained barycentric cubic through q25--q95; included as an exact-anchor stress candidate.

Every surface is evaluated as `T = exp(-LOS_tau)` and correction `C = exp(LOS_tau)`. Fractional correction error against a truth value is `abs(exp(LOS_tau_candidate - LOS_tau_truth) - 1)`.

## Structural and surrogate checks

| Candidate | Max anchor LOS-tau error | Max low-opacity LOS-tau error | Opacity violations | Elevation violations | Max wrong-way correction excursion | Worst q50/q75 leave-one-out correction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `piecewise_linear_los_tau_v0` | `0.00000000000000000e+00` | `0.00000000000000000e+00` | 0 | 25811 | `0.839827%` | `0.313498%` |
| `pchip_los_tau_v0` | `1.11022302462515654e-16` | `0.00000000000000000e+00` | 0 | 19108 | `0.839827%` | `0.172500%` |
| `cubic_through_anchors_los_tau_v0` | `0.00000000000000000e+00` | `0.00000000000000000e+00` | 0 | 14406 | `0.839827%` | `0.135693%` |

All candidates are finite, have `0<T<=1` and correction `C>=1` on the diagnostic grid, preserve the exact q anchors to binary64 evaluation precision, and preserve the approved low-opacity identity. The detailed per-band results, nextafter continuity checks, minima, and violation counts are in `candidate_surface_metrics.csv`.

The legacy q anchors are monotone with increasing opacity throughout the diagnostic elevation grid. The unconstrained cubic is retained only to expose possible between-anchor behavior; exact anchors alone do not establish a valid interpolant.

Every exact-anchor candidate necessarily inherits the legacy q95/a2000 elevation feature. On the 0.1-degree grid that anchor has 81 wrong-way steps and a maximum running-minimum-to-later correction excursion of `0.839827%`. This is the owner-identified sub-percent diagnostic; it is not converted here into an absolute-photometry claim.

## Surrogate holdout and intermediate-opacity evidence

The leave-one-anchor-out table withholds q50 or q75 and predicts its legacy fitted surface from the other q anchors. This tests interpolation sensitivity, not an independent atmosphere run. It must not be used as the provisional one-percent raw-grid fidelity result.

The largest pairwise candidate difference at the arithmetic midpoint of an above-q25 interval is `0.259200%` for `a2000` between `am_q75` and `am_q95` (`piecewise_linear_los_tau_v0` versus `cubic_through_anchors_los_tau_v0`). No truth value exists locally at those midpoints.

## Decision disposition

The evidence is insufficient to choose a versioned successor operator or declare an operational opacity/elevation domain. An exact-anchor surface built from the legacy fits cannot simultaneously remove the q95/a2000 elevation feature, and candidate agreement, legacy-fit leave-one-out performance, or the post-hoc raw q50 leave-one-model-out check is not a substitute for preregistered full-domain raw-model fidelity.

After the requested raw grid is supplied, evaluate at least the piecewise-linear LOS-tau baseline and monotone PCHIP against preregistered withheld tau/elevation model nodes. Select the simplest candidate that preserves exact approved anchors, positivity, continuity, opacity monotonicity, fail-closed support, and no more than one-percent fractional correction error over the owner-declared domain. Elevation monotonicity must either pass or receive an explicit owner scientific disposition supported by recovered raw q95 and independent model evidence. The 0.839827% q95/a2000 feature is diagnostic rather than automatically release-blocking, but it may not be silently waived. Observational 5--10% absolute accuracy and approximately 5% repeatability remain separate later gates.
