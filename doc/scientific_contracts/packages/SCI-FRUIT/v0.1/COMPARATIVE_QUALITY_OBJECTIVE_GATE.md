# SCI-FRUIT v0.1 — Comparative Quality Objective Gate

Status: **owner-approved Stage A comparison framework; exact metric
parameterization, thresholds, and recurrence remain open**

## Purpose

This gate turns the provisional Choice 3 direction into a falsifiable
comparison question:

> Over an owner-approved scientific validity domain, does a candidate FRUIT
> recurrence provide a scientifically preferable result to the exact
> historical Citlali benchmark, with computational and operational performance
> reported separately and no hidden compatibility trade?

The benchmark is a scientific control, not scientific authority. Its anecdotal
quality motivates a demanding comparison but supplies no acceptance threshold.

## Exact Paired-Comparison Unit

A comparison is defined only for a bound tuple

\[
  C=(H,N,D,P,V,T,E),
\]

where `H` is the exact historical benchmark profile, `N` is one exact candidate
new recurrence, `D` is the common input/truth cohort, `P` is the controlled
upstream/effective policy, `V` is the scientific validity domain, `T` is the
terminal-selection/comparison rule, and `E` is the execution environment for
computational metrics. Results from different tuples must not be pooled or
ranked as though they were one controlled comparison.

`H` must bind the historical recurrence, parent route and grouping, effective
configuration, input and calibration identity, learned/fixed-state policy,
stop/terminal rule, and all other state that can affect a scientific result.
For resource comparisons it must additionally bind the build, hardware,
concurrency, storage/cache conditions, and measurement method. The exact
benchmark profile remains an open owner decision.

## Scientific Quality Vector

No weighted scalar quality score is authorized. The approved minimum
comparison vector is:

| Dimension | Quantity to define | Required conditioning and disclosure | Current state |
| --- | --- | --- | --- |
| Angular-scale recovery | Response as a function of declared angular scale or spatial frequency; any maximum recoverable scale is derived from an owner-approved response threshold | Array/band, two-dimensional mode or morphology, orientation, amplitude, support/edge domain, map response/kernel, and uncertainty | owner-approved dimension; exact definition open |
| Per-mode flux recovery | Recovered-to-input amplitude or integrated-flux ratio for each declared astronomical mode, including bias and dispersion across realizations | Mode basis and normalization, estimator/aperture, input amplitude, crowding/background, response convention, validity, and covariance | owner-approved dimension; exact definition open |
| Residual leakage | Atmospheric-fluctuation and other declared nuisance response in the recovered astronomical product, separated from ordinary noise | Nuisance family/spectrum, coupling metric, scan/array/condition stratum, null/truth construction, support, and uncertainty | owner-approved dimension; exact definition open |
| Flux convergence | Per-mode recovery trajectory versus absolute iteration and terminal result; report bias, stability/oscillation, and iterations or time to an accepted band | Initialization, state/learning policy, stopping rule, measurement floor, missing/non-finite behavior, and terminal selector | owner-approved dimension; exact definition open |
| Noise and false structure | Change in map-noise/covariance behavior and creation or amplification of unsupported astronomical structure | Exact NOI target, conditioning, null space, support, multiple-comparison domain, and uncertainty | owner-approved companion dimension; exact definition open |
| Response and uncertainty honesty | Whether fixed-state and complete-procedure response, bias, null space, and uncertainty are correctly disclosed | Exact generation/state, linearization or ensemble definition, route, support, and unavailable-state policy | owner-approved gate; numerical authority unavailable |

A "mode" is not sufficiently identified by angular size alone. Its basis,
normalization, morphology, orientation, support, amplitude, response convention,
and grouping must be explicit. Fourier modes, compact sources, Gaussian or
other extended morphologies, and arbitrary injected templates are not
interchangeable tests.

Maximum recoverable scale is likewise not a primitive universal number. It is
the boundary derived from a declared response/recovery criterion within a
specific mode family and validity domain.

## Computational And Operational Performance Vector

The computational comparison must include at least total wall time, CPU/GPU
time where applicable, peak resident memory, read/write volume, checkpoint and
intermediate storage, and scaling with samples, detectors, observations, map
size, and iterations. Setup and terminal-product costs must not be silently
excluded.

Per-iteration timing is diagnostic but is not the primary efficiency result
when recurrence and stopping behavior differ. The meaningful end-to-end
quantity is resource use and elapsed time to reach the same declared scientific
quality target, together with the result when a method never reaches that
target.

## Owner-Approved Comparison And Acceptance Logic

The owner approves a constrained, multi-objective comparison rather than a
weighted scalar score:

1. define protected scientific dimensions and owner-approved non-inferiority
   tolerances;
2. require the candidate to satisfy validity, response/uncertainty honesty,
   restart, and failure-disclosure gates;
3. require a declared material scientific improvement over the historical
   benchmark in at least one owner-prioritized domain to justify a new
   incompatible recurrence; and
4. report computational/operational performance as a separate vector unless
   the owner explicitly approves a scientific-versus-resource trade.

This framework is approved; its numerical parameters are not. A candidate that
improves recoverable scale while worsening leakage, flux bias, noise, or
stability is not simply "better". Computational performance is reported
separately and cannot compensate for scientific degradation unless the owner
later approves that explicit trade.

## Evidence Layers

- Controlled simulations or injections establish truth-referenced recovery,
  bias, leakage, response, and convergence for their declared domain.
- Representative observations establish behavior on real instrument data but
  do not supply unavailable sky truth; nulls, splits, external references, and
  repeatability must be interpreted within their actual authority.
- Computational benchmarks establish resource performance only for the bound
  build/hardware/input protocol.
- Historical anecdotes may select stress cases but do not establish numerical
  acceptance.

Each result must identify whether it establishes fixed-state behavior, the
complete adaptive procedure, or only one realized trajectory.

## Remaining Parameterization Gate

Before recurrence candidates are ranked, the owner must approve:

1. the exact historical benchmark profile;
2. the signal-mode and nuisance families, validity domain, and truth/null
   construction;
3. exact metric estimators, uncertainty, missing/failure rules, and protected
   non-inferiority dimensions and tolerances;
4. the owner-prioritized improvement domain and material-improvement threshold;
   and
5. exact computational metrics, hardware/build controls, and scaling protocol.

No scientific-versus-resource trade is admissible unless separately approved.

No validation execution or algorithm implementation is authorized by this
gate.
