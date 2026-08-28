# SCI-JINC v0.1 — Square Cache, Point Phase, And Edge Decisions

Status: final Stage A repair candidate; unresolved rows block Stage B

Prepared: `2026-08-28`

| Geometry facet | Stage A disposition | Consequence / blocker |
| --- | --- | --- |
| Coordinate input | **Bound:** frozen `SCI-AST:rtc_output_grid_coordinates@1` continuous FITS pixel for the same RTC `n` and exact target JINC WCS. | [`SCI-AST_TO_SCI-JINC_BOUNDARY.md`](SCI-AST_TO_SCI-JINC_BOUNDARY.md) controls identity, validity and provenance. |
| Center rule | **Partly bound:** the inherited decision says the sample center is rounded before residual phase is binned. | Exact nearest-center expression and relation between one-based FITS coordinates and zero-based local indices are unresolved under `SCI-JINC-ODQ-109`. |
| Half-pixel tie | **Unresolved.** | Owner must select the exact tie direction or a content-bound symmetric rule. Implementation behavior may not decide it. |
| Subpixel phase | **Partly bound:** residual between continuous coordinate and rounded center, evaluated separately on each axis. | Exact phase interval, wrapping at interval ends and sign/order convention are unresolved under `SCI-JINC-ODQ-109`. |
| Phase bins and representatives | **Unresolved:** phase-quantized point evaluation is binding, but bin edges and representative locations are absent. | Owner must content-bind the bin formula and below/equal/above-edge behavior. |
| Effective `subpixel_n` | **Partly bound:** integer and `>=1`; increasing it refines the point-phase representation. | Defaulting/clamping and convergence/acceptance bound are unresolved; no area-integration target is available. |
| Square-cache support | **Bound:** fully populated square support; no radial predicate. | Every pixel in the resolved square is evaluated, including corners beyond radial `r_max`, subject only to finite-map crop and coefficient evaluation success. |
| Cache extent | **Partly bound:** `r_max` fixes the square half-width and the first zero of the second JINC factor. | Exact integer-index extent as a function of pixel size, TolTEC per-array scale and `r_max`, including equality/rounding, remains part of `SCI-JINC-ODQ-102B/109`. |
| Analytic coefficient | **Generic family bound:** the Schloerb method excerpt supplies `J(2 pi r'/a) exp[-(2r'/b)^c] J(3.831706 r'/RMAX)`; point evaluation of finite signed `kappa_ip`; analytic zeros and negative lobes are valid. | Exact TolTEC per-array scale and parameter source/value provenance remain blocked by `SCI-JINC-ODQ-102B`. |
| Radial cutoff | **Unavailable.** | The independent core's radial predicate is superseded. No circular mask or `radius <= r_max` test survives. |
| Pixel-area integration | **Unavailable.** | The independent core's pixel-average branch is superseded. No pixel quadrature or pixel-integrated convergence claim survives. |
| In-map edge crop | **Bound:** square pixels outside the finite map are absent; no wrap, reflection or full-interior normalizer. | Retained pixels use their actual `N_p`, `C_p`, `Q_p`, response and covariance. Truncation can make response/covariance asymmetric and must be recorded. |
| Rounded center outside but square overlaps map | **Unresolved owner decision `SCI-JINC-ODQ-110`.** | Select exactly one: **A** center-required, occurrence contributes nowhere; **B** overlap-admitted, every exact overlapping square pixel is evaluated; or a fully specified successor rule. Stage B cannot choose. |

## Required Geometry Predictions

The future contract must state falsifiable predictions for center coordinates
just below, exactly at and just above every half-pixel tie; phase-bin edges;
`subpixel_n=1`; analytic zeros; negative lobes; square corners beyond radial
`r_max`; each finite-map edge and corner; a center immediately outside each
edge with overlapping square support; and equality at every cache-extent
rounding boundary.

For every retained edge pixel, fixed-state response and covariance use the
same truncated membership as the signal. A rejected occurrence or unavailable
pixel has no finite substitute response/covariance; each requested role is
typed unavailable with the same primary cause and its role-specific
consequence.
