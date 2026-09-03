# SCI-FRUIT EL-F10 — Targeted JINC Accounting Design r0.1

Status: **implementation-informed development design; not scientific
authority, implementation authorization, or a JINC-conformity finding**

## Plain-language purpose

EL-F8 showed that withholding UID 4460 directly from the final a1400 map
creates most of the large scan-shaped response near Neptune. EL-F9 could not
tell whether that happens because the detector has unusually large local
mapmaking leverage, unusually different processed signal, or both. The
published JINC coefficient is nonlinear, so subtracting the two published
coefficient maps does not recover the missing detector's weight.

EL-F10 would run the already understood iteration-5 case once more and keep a
diagnostic receipt for the JINC accumulation. The receipt must be invisible to
the science calculation. Its first job is to prove that turning it on leaves
the science maps bit for bit unchanged. Only then may it be used to explain
the existing result.

## Authority and evidence boundary

The frozen SCI-JINC v0.1/r0.3 authority supplies collision-free symbols and
the generic signed-estimator algebra. It does **not** assert that the present
Citlali implementation conforms to that contract, authorize a TolTEC
numerical JINC route, or add these diagnostic quantities to the fixed
scientific product bundle.

The checked-out implementation and EL-F6--F9 products supply non-authoritative
development evidence about the present executable. EL-F10 may compare that
evidence with the frozen algebra, but it must not convert the comparison into
a conformity, calibration, readiness, or production claim.

The proposed accounting files are development sidecars. They are not sky
maps, independently calibrated detector maps, checkpoint state, or new
SCI-JINC products.

## Exact accounting identity

For an admitted processed occurrence `i` and a1400 map pixel `p`, use the
frozen SCI-JINC notation

\[
N_p=\sum_i I_{ip}\,\omega_i\kappa_{ip}z_i,\qquad
C_p=\sum_i I_{ip}\,\omega_i\kappa_{ip},\qquad
Q_p=\sum_i I_{ip}\,\omega_i\kappa_{ip}^2,
\]

with normalized map

\[
m_p=N_p/C_p
\]

only on the implementation's accepted, conditioned support. In current
implementation names these are the pre-normalization signal accumulator,
grid denominator, and variance accumulator, historically described as `S`,
`G`, and `V`. EL-F10 uses `N`, `C`, and `Q` to avoid inventing a second
scientific notation.

Let `t` be the exact target subset:

- observation `123424`;
- array `a1400`;
- detector UID `4460`;
- zero-based scan index `5`; and
- only occurrences admitted to iteration-5 JINC accumulation after the
  unchanged RTC/PTC and ordinary flag/weight decisions.

Retain the corresponding target sums `N_t,p`, `C_t,p`, and `Q_t,p`. The
without-target accumulators are then

\[
N_{-t,p}=N_p-N_{t,p},\quad
C_{-t,p}=C_p-C_{t,p},\quad
Q_{-t,p}=Q_p-Q_{t,p}.
\]

These are additive identities for the pre-normalization accumulators. They do
not imply that the published coefficient `C_p^2/Q_p` is additive.

Where `C_p`, `C_t,p`, and `C_-t,p` are independently well conditioned, define

\[
\lambda_{C,p}=C_{t,p}/C_p,\qquad
m_{t,p}=N_{t,p}/C_{t,p},\qquad
m_{-t,p}=N_{-t,p}/C_{-t,p}.
\]

The exact-arithmetic deletion identity is

\[
m_{-t,p}-m_p
=\lambda_{C,p}\left(m_{-t,p}-m_{t,p}\right).
\]

This provides the desired leverage-versus-signal-contrast decomposition, but
only where all denominators pass the registered conditioning checks.
`lambda_C` is signed because the JINC kernel is signed. It may be negative or
larger than one and must not be called a hit fraction, probability, or
effective detector count.

## Additional quantities needed for honest interpretation

Signed cancellation can make `C_t/C` small even when the target has a large
operator footprint. The sidecar therefore also retains these explicitly
diagnostic construction quantities for the total and target subsets:

\[
B_p=\sum_i I_{ip}|\omega_i\kappa_{ip}|,
\qquad
A^N_p=\sum_i I_{ip}|\omega_i\kappa_{ip}z_i|.
\]

They support the distinct summaries

\[
f_{B,p}=B_{t,p}/B_p,\qquad
f_{Q,p}=Q_{t,p}/Q_p,\qquad
\rho_p=|C_p|/B_p.
\]

`f_B` is an absolute coefficient-mass share, `f_Q` is a quadratic-support
share, and `rho` describes signed cancellation. None is automatically a
scientific weight, precision, or detector count.

The receipt also records exact admitted occurrence-pixel count and exact
unique-detector count per pixel for the total and target subsets. Counts and
absolute sums remain diagnostics/construction state under frozen SCI-JINC;
they are retained here only because EL-F9 specifically needs map-local
redundancy and roundoff evidence.

## Required diagnostic receipt

When the opt-in diagnostic is enabled for the one selected a1400 observation
map, write an atomic NetCDF sidecar containing:

| Plane or fact | Meaning |
|---|---|
| total and target `N`, `C`, `Q` | additive JINC accumulators before normalization |
| total and target `A_N`, `B` | absolute term sums used for cancellation and roundoff bounds |
| total and target occurrence-pixel counts | exact `I_ip=1` contribution count |
| total and target unique-detector counts | exact number of distinct contributing UIDs |
| realized normalization and science-policy masks | exact support used by the diagnostic-on science map |
| realized thresholds and algorithms | complete finalization identity needed for reconstruction |
| WCS, grid, array, map-slot, units, iteration, scan, UID, kernel, and coefficient identities | interpretation and lineage |

Write a separate compact target-sample table for every final PTC occurrence
of UID 4460 in zero-based scan 5, including stable sample index, processed
signal, analysis coefficient, final flag/admission state and reason, projected
continuous map position, rounded center, subpixel phase, and whether any
pixel received a JINC contribution. The expected accounting from EL-F8 is 305
proposed processed occurrences, of which 34 were already unavailable and 271
were newly withheld by the map-only record. The diagnostic-on no-record replay
must recover those categories rather than assume them.

All identities and units must be explicit. A missing required sidecar field is
a failed diagnostic run, not permission to substitute a proxy.

## Numerical closure

The total `N`, `C`, and `Q` planes should be snapshots of the actual
accumulators consumed by normalization, not independent re-sums. They must
reproduce the diagnostic-on map and its unscaled formal coefficient exactly
under the realized finalization rules.

Target accumulators are separate sums, so `total - target` need not be bitwise
identical to a run that never accumulated the target. Compare the reconstructed
without-target result with the existing EL-F8 `A5-map` using a registered
binary64 forward-error bound based on occurrence counts, `A_N`, `B`, and the
nonnegative `Q` term sum. The bound must include total accumulation, target
accumulation, subtraction, division, and `C^2/Q` propagation. It must be fixed
in the analysis registration before result values are opened.

Reconstruction must reapply the exact existing denominator gate,
normalization coefficient, positive-order-statistic threshold,
normalization-support mask, empirical coefficient rescale, science-policy
threshold, and finite-validity rules. Report common, gained, and lost support
separately. A support difference outside its registered rounding/threshold
explanation is a stop, not a result to interpret.

## Proposed software boundary

Add one typed, disabled-by-default `mapmaking.jinc_accounting` diagnostic
configuration with a single explicit target `(array, uid, zero-based scan)`.
It is valid only for JINC raw-observation mapmaking and fails closed for an
ambiguous or absent target map. The implementation must be generic; UID 4460,
observation 123424, and iteration 5 belong only in the EL-F10 test
configuration.

The diagnostic state belongs to the map-buffer/output lifecycle. It must not
add process-lifetime state, make YAML authoritative inside numerical kernels,
or expose new cross-cutting `Engine` state. Science accumulation and
normalization must not read diagnostic values. With diagnostics disabled, no
diagnostic arrays or sample rows are allocated.

This design deliberately leaves storage representation and internal helper
layout to implementation, provided the exact identities, arithmetic, bounded
scope, and neutrality gates are met.
