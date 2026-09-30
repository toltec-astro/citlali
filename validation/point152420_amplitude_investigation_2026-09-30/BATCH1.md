# Batch 1: calibration at the saved M0 checkpoint

Three serial full-profile diagnostic starts, charged conservatively to the new
six-planned/two-replacement envelope. No starts yet. This is a frozen-operator
experiment, not a historical reproduction or a repaired five-feedback baseline.
Parent: completed B-REF numerical source 518e05891, all four networks 1–4,
a1100, POINT152420_0_2; exact saved checkpoint bindings are frozen by prepare.

| Role | Question / changed block | Fixed state / expected discriminator |
| --- | --- | --- |
| M0-replay | Can the diagnostic evaluator reproduce native masked PTC and MAP? No scientific block changes. | Original calibrated input, eligibility, rank-10 basis, AST pixel membership and normalization. All native cleaned values must agree within 1e-5 mJy and science-map values within 1e-6; exact causes/counts/science support. Failure stops before treatments. |
| common-current | What does restricting to detectors with valid calibration and flag=0 in both APTs do? | B RTC/sample validity, current CAL, original basis restricted to the remaining observed entries, same grid. Recompute means, masked coefficients and actual MAP denominator/support. Changes are population plus frozen-basis restriction, not detector count multiplication. |
| common-historical-conversion | What is the effect of historical flxscale with its extinction correction disabled? | Exactly the common-current membership and RTC/AST/basis; replace y by y × historical_flxscale / recorded_current_total_multiplier **before** centering and PTC projection. Expected change is a larger source response if conversion is responsible; it can also alter removed atmosphere and morphology. No factor applied to the final map. |

The full calibrated detector factor distribution is heterogeneous. A median
factor is not a map correction and is not used as one. Exact identity matches
use unique (network, native tone), array and both recorded frequencies; actual
CAL row selection and factor are checked against the successor APT. The audit
checks every available CAL detector against the saved post-RTC x and multiplier;
representative conversions additionally multiply x by the historical factor
independently. This validates application arithmetic, not either APT's absolute
calibration or the passband/atmospheric physics. Current atmospheric gain is
about 1.0333, far smaller than the APT differences.

This checkpoint is complete for this operation: original PTC input/eligibility,
slots, detector domains, basis, cleaned/cause arrays, source-probe calibration
multipliers, MAP incidence, plan and FITS remain. Removed intermediate FRUIT
scratch is not needed. Means and coefficients are recomputed; the learned basis
is not changed. Rank-deficient patterns are unavailable and reported. MAP
numerator and denominator use retained contributions; coverage uses the native
87.5%-order-statistic reference and 0.5 cut, not a maximum-count proxy. Compare
native and common-population results separately, retain lost/gained support,
and refuse a fit with incomplete 40-arcsec coverage.

Measurements: repaired Gaussian+constant40 fit, native centroid, peak, widths,
constant, objective, angular integral; fixed radii4/8/12/20/30/40/50/60 and
independent45–60 arcsec median background; actual contributions and per-segment
rank losses. Save cleaned and uncleaned numerators on the same contribution
domain to allow source-coherent removed-signal analysis. Raw uncleaned level is
not source truth. No rotations, invented registration, uniform missing-pixel
fills, independently normalized map addition, injections or FRUIT here.

This isolates a conditional calibration effect at M0. It cannot alone allocate
the entire B T5/redu09 gap, reproduce historical PCA/JINC/weights/network0, or
establish the adaptive consequence of changing calibration. Follow-on controls
will be chosen after assessment, using at most three remaining planned starts;
replacement starts need a diagnosed defect. No blind retry or automatic follow-on.
The earlier campaign's allowance is not additive. Q stays parked, photometry open.

One process, one numerical thread; networks/segments serial, below the approved
two-network ceiling. Only compact maps/JSON are written (100MB reservation).
The shared existing roots, 30GB cap and 2GB disk reserve are checked at prepare,
before each charged start, and completion. No deletion. Input/source/environment
hash verification and an exclusive lock precede the append-and-fsync start
ledger. A pre-existing ledger refuses a rerun. Failures preserve evidence.

No persistent launcher is established for this worktree. After clean-source
freeze and gates, use the owner foreground command in RESUME.md, then return
for assessment. There is no model polling, background agent or automatic resume.
