# SCI-ALIGN-001-LR-BEAMMAP retained-product result

## Outcome

The no-code retained-product path was sufficient. No Citlali process or new reduction was launched, and no application source changed. The frozen direction registry contains 99 left and 99 right windows with 0 exclusions.

The pooled retained-TOD estimator gives **-12.1380 ms** with delete-one-scan SE **0.2727 ms** and 95% interval **[-12.6725, -11.6036] ms**. The right-minus-left centroid is **-1.18499 arcsec** parallel and **0.03408 arcsec** perpendicular to the frozen scan axis. The confirmatory matched detector population is 4809/4965.

The three arrays span **-15.200 to -11.070 ms** (ordered by signed value), and network estimates span **-16.386 to -9.296 ms**. The network estimates correlate **-0.966** with the known raw-timestamp-minus-assigned-slot residual; the fitted slope is **-0.849**, compared with -1 for a shared raw-event lag. Under that algebra only, `measured + raw residual` has mean **-13.054 ms** and population scatter **0.743 ms**. This is not an integration-centroid correction because the raw timestamp event is unproved, and the remaining interface scatter plus first/second-half difference still preclude a universal correction.

Disposition: **direction-odd residual detected; interface dependence tracks engineering slot residuals, but no physical correction is authorized** in this bounded local retained-product diagnostic. This is not SCI-ALIGN acceptance and does not provide absolute physical timestamp correctness.

## Scope and limitations

- Eight feasibility-pilot UIDs [0, 5, 10, 15, 20, 25, 30, 35] were viewed before protocol freeze and are excluded from all confirmatory results.
- The native speed distribution is narrow, so scaling with speed is not independently identified. The 50/100/200 arcsec/s values in `timing_estimates.csv` are dimensional translations.
- Detector timestamp start/end/effective integration-centroid semantics remain unproved; absolute sky correctness remains unresolved.
- The retained split uses final-iteration per-crossing PTC products and a bounded 1-arcsec diagnostic accumulation. It does not alter or validate SCI-MAP implementation.
- A human Unity exact-9aae/exact-candidate left/right campaign remains required for definitive governing-versus-candidate evidence.
