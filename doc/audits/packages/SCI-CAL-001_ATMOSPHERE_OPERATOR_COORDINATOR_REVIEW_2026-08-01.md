# SCI-CAL-001 atmosphere-operator coordinator review — 2026-08-01

## Reviewed identity and scope

The coordinator independently reviewed the completed evidence-only package on
`codex/sci-cal-001-atmosphere-operator` at
`8b2fc27513eabcac1f4862b18026879f94c34d69`.

- Parent: CAL phase-zero evidence commit
  `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`.
- Repair base: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Package root:
  `validation/sci_cal_001_atmosphere_operator_2026-08-01`.
- Owner brief SHA-256:
  `b3fb2a5c0588bfc3ad12ee0d5bf93e637008d5c402fed8156934f858bf650714`.
- Machine-readable owner request SHA-256:
  `067f23786aac8f59b966b1941f0786cbccbcdad05e2a4f83475aa273c1fb80d8`.
- Package digest-manifest SHA-256:
  `c7d6448758cb4cebe6a57a318d2610ef32ff441e9e4d78f076bec2abcbda7af6`.

The branch was clean at the reviewed commit. Its diff contains the
reproducible validation package and a status note only; no Citlali application
source, tests, product contract, configuration, or estimator implementation
was changed. The package verifier passed with the required local TolTEC Python
environment and no network or Unity access.

## Evidence accepted for coordination

The following results are accepted as reproducible phase-zero evidence, not
as an approved scientific operator or observational-accuracy claim:

- The exact q25, q50, and q75 raw NPZ grids were recovered with digest and
  TolTECA registry identity. The q95 grid is still missing.
- The legacy coefficients are exactly reproduced after eight-decimal rounding
  by degree-six `numpy.polyfit` fits in elevation radians over 31 nodes from
  20 through 80 degrees in two-degree steps.
- The legacy spectral convention uses monochromatic samples at 272.73,
  214.29, and 150.00 GHz relative to 225.00 GHz; it is not TolTEC-passband
  integration.
- Full-sample-airmass reconstruction from the repair-base anchors has worst
  recovered raw-node fractional extinction-correction error `0.427091%` over
  the available grids.
- The q50 prediction from q25 and q75 has worst corresponding error
  `0.243563%`, but is post-hoc rather than preregistered or blinded and cannot
  close the fidelity gate.
- Piecewise-linear line-of-sight optical depth is a bounded primary candidate;
  monotone PCHIP is a challenger. Neither is selected or versioned.
- Exact-anchor candidates inherit the q95/a2000 fitted elevation feature,
  with a `0.839827%` maximum wrong-way correction excursion on the diagnostic
  grid. The missing raw q95 artifact must be inspected before disposition.
- The hard selector remains materially discontinuous, so a continuous
  successor remains scientifically motivated.

These numerical results are below the provisional one-percent representation
target where raw evidence exists. They do not establish approximately five
percent observational repeatability, 5--10 percent absolute flux accuracy,
or validity in the missing q75--q95 and intermediate-opacity domains.

## Coordinator disposition

The package is **verified and accepted as evidence; owner input required**.
Do not select or implement a successor atmosphere operator yet.

`SCI-CAL-001` remains:

- contract `approved`;
- implementation `nonconformant`;
- validation `in_progress`;
- production `fail_closed`;
- verdict `amend`; and
- re-audit `required`.

The CAL repair remains paused after phase zero. `SCI-CAL-001-UNITY-001` is not
eligible because there is still no application-repair SHA. The approved ALIGN
decisions constrain the eventual input boundary, but ALIGN implementation is
still nonconformant and its repair/re-audit are incomplete; CAL F010 remains
open.

## Owner inputs and decisions required

### CAL-ATM-D001 — complete atmosphere evidence

Supply locally, without Codex contacting Unity or the recorded HTTP endpoint:

1. the original q95 NPZ registered as TolTECA datafile 461 with expected MD5
   `0ca7b331823237767d26016d19bffb3d` and SHA-256 custody provenance;
2. the historical `am` executable/source, arguments, configuration, q
   profiles, MERRA percentile construction, site/geometry, grids, and packer
   if full regeneration is required; and
3. physical, preregistered intermediate-opacity calculations in each
   q25--q50, q50--q75, and q75--q95 interval.

Recommended disposition: require all three before final operator selection.
Partial recovery is sufficient to keep work moving but not to freeze a
production scientific model.

### CAL-ATM-D002 — spectral lineage

Choose either:

- preserve the recovered monochromatic 272.73/214.29/150.00 GHz relative-to-
  225.00-GHz convention for the bounded CAL repair; or
- authorize a separately versioned band-integrated scientific model with
  immutable passbands, detector/array aggregation, normalization, source
  spectrum, and quadrature rules.

Recommended disposition: preserve the monochromatic convention for lineage
closure. Treat passband integration as a later separately audited model.

### CAL-ATM-D003 — candidate selection gate

Approve piecewise-linear interpolation in line-of-sight optical depth as the
primary candidate and monotone PCHIP as the challenger. Freeze their
definitions before inspecting preregistered intermediate runs. Select the
simplest candidate that preserves exact anchors, positivity, continuity,
physical monotonicity, fail-closed support, and no more than one-percent
fractional extinction-correction error over the approved domain.

Recommended disposition: approve this validation gate, not an operator yet.

### CAL-ATM-D004 — operational domain

Approve exact inclusive zenith-opacity and aligned-elevation endpoints only
after q95 and held-out model support are available. The maximum candidate
domain is `0 <= tau225 <= tau_q95` and `20 <= elevation <= 80 deg`; no
extrapolation is authorized, and invalid/out-of-domain requests fail closed.

Recommended disposition: adopt that maximum-domain rule now while deferring
the final endpoints until the evidence gate passes.

## Next sequence

1. Record owner decisions `CAL-ATM-D001`--`CAL-ATM-D004`.
2. Stage and verify the requested owner-supplied inputs locally.
3. Freeze the candidate definitions and blinded validation design before
   generating or inspecting intermediate-opacity results.
4. Return a final operator/domain decision package to the coordinator.
5. Only after explicit approval, amend the CAL repair handoff and resume
   application fixtures and implementation on `codex/repair-sci-cal-001`.
6. Hold Unity evidence and fresh CAL re-audit until the exact application
   repair SHA, local gates, and applicable ALIGN disposition exist.
