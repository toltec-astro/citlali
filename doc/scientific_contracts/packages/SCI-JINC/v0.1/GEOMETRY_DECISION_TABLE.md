# SCI-JINC v0.1 — Square Cache, Point Phase, And Edge Decisions

Status: ODQ-107/109/110 dispositions incorporated; no open geometry decision

Prepared: `2026-08-28`

| Geometry facet | Stage A disposition | Consequence / blocker |
| --- | --- | --- |
| Coordinate input | **Bound:** frozen `SCI-AST:rtc_output_grid_coordinates@1` continuous FITS pixel associated with the exact same processed sample realization entering JINC and the exact target JINC WCS. | [`SCI-AST_TO_SCI-JINC_BOUNDARY.md`](SCI-AST_TO_SCI-JINC_BOUNDARY.md) controls scientific association, identity, validity, causes and provenance without prescribing a data-model join. |
| Sample admission vs pixel support | **Bound:** `SCI-JINC:jinc_map_contribution@1` admits a sample for consideration; JINC finite support separately decides each sample-pixel contribution. | Outside support and contract-defined zero are ordinary no-contribution states. Missing/ambiguous AST association prevents geometry evaluation; finite negative `kappa_ip` is valid. |
| Center rule | **Bound scientifically:** the sample center is rounded before residual phase is binned. | The realization must be single-valued, internally consistent with FITS/pixel indexing and numerically adequate under ODQ-109. No particular adequate nearest-center expression is a separate owner decision. |
| Half-pixel tie | **Engineering realization under ODQ-109.** | Any single-valued tie rule is permitted if it preserves the point-phase operator and its numerical effect is negligible compared with the approximately `10^-3` relative instrument-fidelity scale. Bitwise cross-realization identity is not required. |
| Subpixel phase | **Bound scientifically:** residual between continuous coordinate and rounded center, evaluated separately on each axis. | Interval, wrapping and sign/order choices must be internally consistent with the selected point-phase realization and meet ODQ-109; no particular adequate representation is separately authorized. |
| Phase bins and representatives | **Bound by adequacy:** phase-quantized point evaluation is required. | Bin edges and representatives are engineering choices when single-valued and their total numerical effect is negligible against the ODQ-109 scale. |
| Effective `subpixel_n` | **Bound:** integer and `>=1`; increasing it refines the point-phase representation. | The selected value must make phase-quantization error negligible against the ODQ-109 scale. No pixel-area-integration target, fixed default or stronger precision guarantee is authorized. |
| Square-cache support | **Bound:** fully populated square support; no radial predicate. | Every pixel in the resolved square is evaluated, including corners beyond radial `r_max`, subject only to finite-map crop and coefficient evaluation success. |
| Cache extent | **Bound scientifically:** `r_max` fixes the square half-width and the first zero of the second JINC factor. | Integer-index equality/rounding is an engineering realization that must preserve the fully populated square operator and meet ODQ-109. Evaluation also requires an authorized array-associated `s_a`, `(r_max)_a`, and pixel size; without an authorized parameter set the numerical route is unavailable. |
| Analytic coefficient | **Generic family and semantics bound:** the Schloerb method excerpt supplies `J(2 pi r'/a) exp[-(2r'/b)^c] J(3.831706 r'/RMAX)`, with `r'_a=r/s_a`; point evaluation of finite signed `kappa_ip`; analytic zeros and negative lobes are valid. | `s_a` is an explicit array-associated angular scale and all shape parameters may be array-associated. No TolTEC numerical realization is authorized in v0.1 and no hidden default is permitted. |
| Radial cutoff | **Unavailable.** | The independent core's radial predicate is superseded. No circular mask or `radius <= r_max` test survives. |
| Pixel-area integration | **Unavailable.** | The independent core's pixel-average branch is superseded. No pixel quadrature or pixel-integrated convergence claim survives. |
| Center-domain admission | **Bound by ODQ-110:** the resolved rounded center used for cache placement must lie in the finite destination pixel domain before any footprint evaluation. | An outside center makes `I_ip=0` for every destination pixel, so the occurrence contributes zero to `N_p`, `C_p`, `Q_p` and `T_p^(kappa^2)`, even if its square would overlap the map. Footprint-overlap admission is prohibited. |
| In-map edge crop | **Bound:** for an admitted in-map center, square pixels outside the finite map are absent; no wrap, reflection, full-interior normalizer, footprint completion or edge correction. | Retained pixels use their actual `N_p`, `C_p`, `Q_p` and `T_p^(kappa^2)` membership. JINC-then-crop need not equal direct smaller-map construction. |

## Required Geometry Predictions

The future conformance contract must test representative center coordinates
around half-pixel ties and phase-bin edges, `subpixel_n=1`, analytic zeros,
negative lobes, square corners beyond radial `r_max`, cache-extent boundaries,
and each finite-map edge/corner, including centers immediately inside and
outside every boundary with overlapping square support. These tests
demonstrate the ODQ-110 center gate, a single-valued operator and ODQ-109
adequacy; they do not impose bitwise identity or a scientifically preferred
adequate tie/bin realization.

For every retained edge pixel, all fixed-bundle accumulators use the same
truncated membership. Zero, insufficient or invalid local support is recorded
within the `jinc_map` role under the JINC support and validity rules; it does
not make a whole product role unavailable. A required whole-product
accumulator that cannot be formed prevents publication of the complete bundle;
no placeholder role is synthesized.

An occurrence rejected by the center-domain gate is ordinary no-contribution,
not a bundle failure or a demand for an edge-specific cause, provenance or
diagnostic product.
