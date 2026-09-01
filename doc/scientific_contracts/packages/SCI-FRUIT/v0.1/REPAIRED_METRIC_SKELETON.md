# SCI-FRUIT v0.1 — Repaired Metric Skeleton

Status: **Stage A candidate skeleton; no estimand, metric, threshold, or
population is approved**

## Common Identity And Weight Rules

Let `j` identify a declared astronomical mode or morphology, `r` a realization,
and `k` an absolute iteration. Every metric must bind its units, estimator,
response/kernel convention, grid/WCS, support, masks, conditioning, failure and
missing rules, independent sampling unit, pairing unit, and inference target.

Every realization or aggregate weight must be finite, nonnegative,
prospectively fixed, independent of candidate outcome, and common to candidate
and historical control wherever pairing is claimed. Each weighted denominator
must be strictly positive. Outcome-dependent reweighting is prohibited.

## Truth-Referenced Recovery

For injected truth amplitude `a_jr`, recovered value `ahat_jrk`, and a
prospectively declared minimum valid ratio amplitude `a_min,jr > 0`, ratio
metrics are defined only when `|a_jr| >= a_min,jr`:

\[
  \rho_{jrk}=\frac{\widehat a_{jrk}}{a_{jr}},
  \qquad
  T_j(k)=\frac{\sum_r w_{jr}\rho_{jrk}}{\sum_r w_{jr}},
  \qquad
  B_j(k)=T_j(k)-1.
\]

Near zero, the protocol must use an absolute-error metric or an externally
scaled metric with a positive scale that is independent of candidate outcome.
It may not create a numerical ratio by convention.

An illustrative weighted population dispersion is

\[
  D_j^2(k)=
  \frac{\sum_r w_{jr}\left(\rho_{jrk}-T_j(k)\right)^2}
       {\sum_r w_{jr}}.
\]

`D_j^2` is a **weighted population dispersion**, not a sampling variance or
estimator-uncertainty estimate. Variance or uncertainty language requires a
separate sampling theorem matched to the declared independent unit,
clustering, and inference target.

The final protocol may replace these illustrative moments with prospectively
frozen robust, quantile, stratified, or model-based functionals, but it must
retain absolute candidate/historical quantities and their paired contrast.

## Paired Baseline Contrast

For metric `G_l` and preference sign `s_l`,

\[
  \Delta_{lr}=s_l\left(G_{lr}^{\mathrm{candidate}}
  -G_{lr}^{\mathrm{historical}}\right),
\]

with positive `Delta` favoring the candidate. This contrast is available only
when both methods have a scientifically valid result on prospectively declared
common support. Rescue, regression, failure, and scientific unavailability are
reported through the complete outcome matrix rather than imputed into
`Delta`.

## Nuisance Coupling And Null Evidence

For every applicable nuisance family, the protocol declares the nuisance
identity and support, positive and negative injections, multiple amplitudes,
the astronomical/background realization, the independent and pairing units,
and a linearity/nonlinearity assessment. For nuisance `u` with nonzero injected
amplitude `b_ur`, an illustrative full-procedure coupling is

\[
  L_{u\rightarrow j}(k)=
  \frac{\sum_r v_{ur}
    \left[\widehat a_{jrk}(m+n+b_{ur}u)
          -\widehat a_{jrk}(m+n)\right]/b_{ur}}
       {\sum_r v_{ur}}.
\]

Positive/negative and multi-amplitude results must be examined for asymmetry
and nonlinearity. Fixed-state coupling and complete-procedure coupling are
separate targets. Neither substitutes for false astronomical structure on a
nuisance-only/null input; both coupling and null evidence are required where
scientifically applicable.

## Convergence And Time To Quality

With a prospectively fixed positive scale `s_jr`, retain truth error and
inter-iteration stability separately:

\[
  e_{jr}(k)=\frac{|\widehat a_{jrk}-a_{jr}|}{s_{jr}},
  \qquad
  d_{jr}(k)=\frac{|\widehat a_{jrk}-\widehat a_{jr,k-1}|}{s_{jr}}.
\]

Small `d` is not evidence of small `e`. Reports distinguish actual terminal,
oracle best, hard-cap termination, oscillation, drift, and time/iterations to a
frozen multidimensional quality region. A method that never reaches the region
remains censored or failed according to the frozen protocol; it is not omitted
from a mean time-to-quality.

## Response, Uncertainty, And Morphology

Fixed-state and complete-procedure response, nuisance coupling, and uncertainty
are distinct. Shape-sensitive profiles keep amplitude/flux, centroid,
normalized morphology, support, and map residual as separate estimands. Map
residuals bind response/kernel, comparison grid, edge/support domain, weighting,
background, masks, filtering, and missing/non-finite rules.

## Freeze Rule

No method-specific or outcome-favorable metric may be introduced after
candidate outcomes are seen. Changing an estimand, equation, normalization,
weight, support, priority, threshold, or failure rule creates a new development
and qualification generation with fresh untouched evidence.
