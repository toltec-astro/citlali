# SCI-FRUIT v0.1 — Scientific-Owner Comparative-Quality Direction

Date: `2026-08-31`

Status: **owner Stage A analytical direction; comparison domains identified;
exact metrics, thresholds, and recurrence remain open**

Direction ID: `SCI-FRUIT-DIR-2026-08-31-COMPARATIVE-QUALITY`

## Owner Direction

The scientific owner directs that "better" be defined by controlled comparison
with the existing Citlali FRUIT approach. The existing method appears
anecdotally to perform very well and is therefore a serious benchmark, but that
anecdotal assessment is motivation rather than validation evidence.

The comparison must distinguish two metric families:

1. **scientific quality**, including the angular scales of recoverable
   astronomical signal, the recovered-flux fraction for declared astronomical
   modes, atmospheric-fluctuation and other residual leakage, flux convergence,
   and related response/uncertainty behavior; and
2. **computational and operational performance**, including the resource and
   scaling measures needed for a fair end-to-end comparison.

The historical benchmark must be exact and versioned. A bare reference to
"existing Citlali" is not sufficient for a reproducible comparison: recurrence,
route, effective configuration, input cohort, stopping/terminal rule, build,
hardware, and measurement protocol must be bound where applicable.

## Scientific Interpretation

No single metric may silently stand for scientific quality. In particular:

- maximum recoverable angular scale depends on the declared signal family,
  response threshold, support, orientation, and validity domain;
- recovered-flux fraction requires an exact mode basis, normalization, input
  amplitude, estimator, response convention, bias, scatter, and uncertainty;
- atmospheric or other residual leakage requires a declared nuisance family
  and a metric that distinguishes leakage into astronomical estimates from
  ordinary map noise;
- flux convergence requires an iteration trajectory and terminal-selection
  rule and does not by itself establish correct flux; and
- computational speed does not compensate for a scientific degradation unless
  the owner later approves that explicit trade.

The comparison should therefore preserve a vector of scientific and performance
results rather than collapse them into an unapproved scalar score.

## Decisions Still Required

Stage A must still present for owner approval:

- the exact versioned historical benchmark profile;
- the astronomical signal/mode families, angular-scale domain, amplitudes,
  morphologies, orientations, and support conditions;
- the atmosphere and other nuisance families and leakage estimands;
- exact metric definitions, uncertainty treatment, comparison tolerances, and
  failure/unavailable rules;
- protected scientific non-inferiority constraints and the improvement needed
  to justify an intentional compatibility break;
- the computational/resource metrics, hardware/build controls, and scaling
  domain; and
- the treatment of tradeoffs when one scientific metric improves and another
  degrades.

This direction does not approve a recurrence, benchmark execution, metric
threshold, parent route, validation run, Stage B dispatch, implementation, or
production use.
