# Temporal change and map-impact synthetic study

## Program adherence and prior-work recovery

This owner-authorized experimental study follows the [contract-library
charter](../../doc/scientific_contracts/README.md) boundary: it is not a new
frozen contract, application implementation or implementation-blind derivation.
Grant requested development and synthetic testing of the proposed temporal
method, preserving the 0.10 optical cut and showing map tradeoffs before harm
threshold selection. No reductions are authorized.

Recovery dispositions: adopt the configurable 0.10 owner decision in
`doc/OPTICAL_RESPONSE_POLICY_2026-10-29.md`; cite the weighted POINT implementation
and its frozen original-CAL health/FRUIT boundaries; cite and retain all of
`validation/optical_decision_separation_2026-10-29/`, especially stationary
counterexamples 7 and 15. Supersede interval non-overlap only as the experimental
temporal test. Defer production admission, gain correction, harm budgets,
reference-population uncertainty, full FRUIT and real-map qualification.

## Preflight and scope

- Owner: Grant; same pointing-study worktree and branch
  `codex/configurable-point-20261028`; clean base
  `9bcbfcf539355a36faf18c5114492fd56c600fb0`.
- Read Engineering Governance, Timestream Successor Governance, Review and
  Conformance, AGENTS, TolTEC context, current status, optical policy, previous
  study and contract charter. Governance acceptance/effectiveness recorded at
  `06a3ade51c1b3f38887295433d913811bf25cd14` /
  `77507836325eff9f469062d5884481ea37599594`; canonical integration recorded as
  `e32c4584dec07617d6caa857d39bf13047b6c4b0` in preceding study.
- Tier 2 synthetic evidence, no application WIP slot. Owned paths: this directory
  and a brief living-status entry. Figure outputs in task visualization storage.
- Pure NumPy/SciPy prototype; dimensionless synthetic normalized response;
  standalone toy map in beam-FWHM coordinates and arbitrary brightness units.
  No native imports, telescope inputs or operational entry points.
- Statistical method: fixed-design null residual bootstrap with estimated AR(1)
  and wild innovations; maximum over all five block-boundary steps and a drift.
  Same block offsets in null and alternatives. Conditional 99% simultaneous
  contrast intervals from encompassing-model residuals. Two panels share draws;
  combined search includes both panels, never treats them as independent evidence.
- Prespecified test point p <= 0.01 is a diagnostic false-alarm target, not an
  admission or harm threshold. Do not tune it after outcomes. Preserve failures.
- Synthetic design: six blocks of 40 bins; independent, AR(1), heteroskedastic,
  low-excitation, offset, gaps, heavy-tail, reference-error and model-misspecification
  controls; step/drift/temporary-change alternatives. Revisit old seeds 7/15.
  Main null Monte Carlo: 1,000 realizations per named case; alternative and
  misspecification: 250 per case; 999 bootstrap draws. No stopping on significance.
- Map grid: synthetic 41x41 raster, four cross-linked passes, unit beam FWHM,
  source peak 20, atmosphere RMS 30, benchmark uncleaned map noise 1. Fixed
  source-masked atmospheric regression; no learned ALS or FRUIT. Sweep changes
  0.01/0.05/0.10/0.30/0.80 and affected weight shares 0.001/0.01/0.05/0.20,
  steps at 1/4,1/2,3/4 and drift. Compare atmospheric-only and all-signal gain.
  Coadd scaling shown as assumptions, not measured telescope outcomes.
- Gates: focused independent mathematical checks, synthetic Monte Carlo and
  finite-value checks, deterministic reproduction, visual inspection, exact-SHA
  independent review. No full CTest/config/build gate: application untouched.
- Local Python venv; single-thread numerical libraries where supported;
  warnings as errors and headless rendering. Retain compact results only.
- Stop/reassess numerical failures, inability to identify change, false-alarm
  or interval failures; report rather than tune. No automatic restoration.
- Authority: local coherent commits and study only. No production activation,
  integration, push, Unity, executable replacement, observation reprocessing,
  new harm threshold, cleanup or unrelated edits.

Method background (motivation, not proof of this prototype):
https://doi.org/10.1111/boer.12243 discusses fixed-regressor/sieve-wild resampling
with changing and correlated errors. The present AR(1) approximation does not
inherit its general asymptotic guarantees. Practical impact and statistical
significance remain separate, as in https://doi.org/10.1177/1948550617697177.

## Evidence-triggered reassessment

The prespecified 12,000-realization study exposed undercoverage of simultaneous
99% intervals (about90–92% for several rho0.6 cases), elevated null flags in some
cases, and poor power for temporary changes. Preserve all results, confidence
levels and seeds. Continue within synthetic scope with a diagnostic **oracle**
control: supply the exact known Gaussian noise dependence/scale to the same
linear contrasts and maximum search. This is not an available operational
estimator or a replacement chosen to make the candidate pass. It isolates the
dependence-estimation limitation. No thresholds, trials or runtime are changed.
