# SCI-CAL-001 copied-AM protocol addendum after lineage check

## Triggering result

Study A found that the copied AM 12.2 suite is internally complete but is not
the registered legacy q-model lineage. None of its 25 full-grid NPZ products
matches the q25/q50/q75 TolTECA identities, and neither copied q95 product
matches datafile ID 461's expected MD5. The closest same-percentile family is
DJF, but its numerical differences are material provenance evidence and must
not be renamed as the legacy source.

This addendum preserves the frozen preregistration rather than rewriting it
after results were seen.

## Disposition of the original studies

- Study A continues as an exact custody, workflow, and mismatch report.
- Study B continues. It tests whether a clean native build of the supplied AM
  12.2 source reproduces the copied AM 12.2 output matrix, not whether that
  matrix generated the historical legacy q artifacts.
- Study C's proposed annual-profile anchor training is stopped before operator
  selection. Adopting annual MERRA-2 profiles as new production anchors would
  be a scientific-model change requiring an owner choice. No annual-anchor
  operator is selected or assigned an operational domain here.

## Added diagnostic C1: legacy-surface AM-12.2-family stress test

Evaluate the already defined legacy-anchor `piecewise_linear_los_tau_v1` and
`pchip_los_tau_v1` surfaces against the 25 copied AM 12.2 physical profiles.
Use each profile's own zenith `tau225` coordinate, the frozen monochromatic
frequencies, the 20--80 degree raw grid, full sample modified-secant airmass,
and the top-of-atmosphere pivot. Exclude out-of-support rows rather than
extrapolating.

This analysis is post-discovery and not blinded. It may expose whether the
legacy surfaces generalize to distinct seasonal/profile structures indexed by
`tau225`, but it cannot close the missing q95 artifact or prove the historical
generator. Report the one-percent representation threshold as a stress-test
diagnostic only. Do not use a pass to authorize an operator, and do not use a
failure to rewrite the copied profiles.

## Added diagnostic P1: documented H2O-scale provenance hypothesis

The supplied AMC contract has one explicit scalar generation parameter:
`Nscale troposphere h2o`. Test, without altering any profile bytes, whether a
copied LMT profile plus a non-unit H2O scale can reproduce the legacy grids.

For q25/q50/q75:

1. For each of the 25 copied profiles, solve only the H2O scale so that the AM
   12.2 transmission at 225 GHz and elevation 80 degrees equals the exact
   legacy value.
2. Freeze that scale from the 225-GHz match; do not optimize it against any
   other frequency or elevation.
3. Generate the historical 0--500 GHz, 10-MHz grid at elevations 20--80
   degrees and compare it against the recovered legacy NPZ over the complete
   frequency/elevation grid and at the three nominal TolTEC frequencies.
4. Rank hypotheses by full-grid transmission and Rayleigh-Jeans residuals,
   while reporting the source profile identity and fitted scale explicitly.

For q95, the registered raw grid is absent. Solve the scale against the source
T225 literal and compare the resulting nominal-frequency, elevation-ratio
degree-six coefficients against the legacy q95 literals. This is weaker than
raw-grid comparison and must remain labeled as a provenance hypothesis.

An exact or near-exact numerical reconstruction is evidence for a candidate
input recipe, not proof of historical custody, because profile selection and
scale inference are post hoc. A failure means the remaining profile/version
facts still require owner or producer input; no further atmospheric degrees of
freedom may be introduced silently.

## Added diagnostic R1: frequency-resolution convergence

Every copied 10-MHz AM output is numerically complete but reports unresolved
in-band lines and returns status 1. Before proposing a new AM 12.2 grid,
compare center-frequency optical depth and transmission at 150.00, 214.29,
225.00, and 272.73 GHz using a common 140--280 GHz interval and frequency
steps 10, 5, 2, and 1 MHz. Use DJF q5 and DJF q95 at elevations 80 and 20
degrees as low/high-opacity and low/high-airmass brackets. All four center
frequencies must be exact grid nodes at every resolution.

Report the warning count, exit status, and fractional extinction-correction
difference relative to the 1-MHz result. Treat a maximum difference no larger
than 0.1% as a bounded numerical-resolution diagnostic within the overall 1%
representation budget; it does not make a warning-status run operationally
clean. If the warning persists or the diagnostic fails, a finer-grid policy
requires owner approval before a separately versioned successor grid is
frozen. The 10-MHz grid remains the immutable historical copied-suite
lineage regardless of this diagnostic.

## Unchanged stop boundary

The copied matrix may support a separately versioned AM 12.2 successor study,
but selecting that model family, constructing new intermediate profiles, or
declaring an operational opacity/elevation domain requires owner scientific
direction. No Citlali application code, Unity access, repair, re-audit, or
coordination-registry change is in scope.
