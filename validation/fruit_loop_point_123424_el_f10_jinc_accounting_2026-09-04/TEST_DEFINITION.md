# SCI-FRUIT EL-F10 — Registered JINC accounting closure

Status: **authorized development test definition; no result yet**

Decision identity:
`SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1`

## Fixed question and target

Replay the exact EL-F6 no-record iteration-4 checkpoint once, advancing
absolute FRUIT iteration 4 to 5. The diagnostic target is observation 123424,
a1400, UID 4460, and zero-based scan 5. The ordinary map calculation is
unchanged. The target accounting contains only final-PTC occurrences that the
ordinary JINC mapmaker admits.

The result may explain the already observed direct a1400 response in this one
trajectory. It may not judge the detector, select a safeguard, establish a
generic mechanism, qualify JINC or FRUIT, or change an algorithm.

## Gate order

1. Revalidate every registered input hash.
2. Require bitwise identity of the nine a1100/a1400/a2000 signal, kernel, and
   realized-coefficient planes between the diagnostic replay and EL-F6 N5.
   Also require bitwise identity of all three unscaled formal-coefficient
   planes.
3. Require scientific checkpoint equality, allowing only the registered
   creator-version and diagnostic-configuration provenance differences.
4. Require exact re-finalization of the diagnostic a1400 map and unscaled
   formal coefficient from the captured total `N`, `C`, and `Q` snapshots.
5. Require the target ledger to contain 305 proposed final-PTC occurrences,
   34 already unavailable, and 271 otherwise admitted occurrences. Geometry
   exclusions are reported separately and cannot be silently combined with
   prior flagging.
6. Only after gates 1--5 pass, subtract target `N_t`, `C_t`, and `Q_t`, apply
   the existing finalization, and compare with EL-F8 A5-map under the fixed
   forward-error bound below.

A failure at any step stops scientific interpretation.

## Existing finalization

For every pixel, the reconstruction uses the current implementation's exact
rules:

- denominator is usable only when finite and `abs(C) > 1e-8`;
- `Q` is usable only when finite and positive;
- raw formal coefficient is `C*C/max(Q, 1e-30)` on that usable support and
  zero elsewhere;
- the positive values are sorted ascending;
- `j=floor((floor(0.75*n_positive)+n_positive)/2)`;
- normalization threshold is `positive[j] * coverage_cut/10`;
- normalization support is finite, positive coefficient greater than or equal
  to that threshold;
- signal is `N/C` on normalization support and zero elsewhere;
- the already existing A5-map empirical scale is used only when checking the
  later empirical coefficient/science-policy support; it is not treated as an
  independently reconstructed noise realization.

## Frozen binary64 forward-error bound

Let binary64 unit roundoff be exactly `u = 2^-53`. For an accumulator `X`,
term absolute sum `A_X`, and occurrence count `h`, define

`gamma(h) = max(h,1)*u / (1-max(h,1)*u)`.

For separately rounded total, target, subtraction, and reference-without-
target sums, the registered accumulator comparison bound is

`b_X = gamma(h) A_X + gamma(h_t) A_Xt + gamma(h-h_t)(A_X-A_Xt)
       + u (abs(X)+abs(X_t))`.

Negative `A_X-A_Xt` caused only by rounded absolute sums is clipped to zero.
For `N`, use the recorded absolute numerator-term sums. For signed `C`, use
the recorded `B=sum(abs(omega*kappa))`. For nonnegative `Q`, use the absolute
captured total and target `Q` values.

Define the reconstructed values `n=N-N_t`, `c=C-C_t`, `q=Q-Q_t`. A pixel is
conditioned for the comparison only when

`abs(c)-b_C > 1e-8` and `q-b_Q > 0`.

With fixed safety factor `s=16`, the signal bound is

`b_m = s * ((b_N + abs(n/c)b_C)/(abs(c)-b_C)
             + u*max(1,abs(n/c)))`.

The unscaled formal-coefficient bound is

`b_W = s * ((2 abs(c)b_C+b_C^2)/(q-b_Q)
             +(abs(c)+b_C)^2 b_Q/(q(q-b_Q))
             +u*max(1,abs(c^2/q)))`.

Every common-conditioned-support difference must be no larger than its
per-pixel bound. No observed value may be used to enlarge `u`, `s`, or these
formulas.

For a support mismatch, the threshold uncertainty allowance is the largest
finite registered `b_W` multiplied by `coverage_cut/10`. A changed pixel is
roundoff-explainable only when its reconstructed raw coefficient lies within
its own `b_W` plus that threshold allowance of the reconstructed threshold.
Any other support mismatch stops interpretation.

## Fixed summaries

Use five regions: complete normalization support; 20-arcsec apertures about
the registered injected-source and Neptune world positions; the registered
40--120 arcsec injection-centered annulus excluding 25 arcsec about Neptune;
and the full UID-4460 scan-5 contribution footprint. Retain the four EL-F8
trigger pixels.

Report distributions of signed `C_t/C`, absolute `B_t/B`, quadratic `Q_t/Q`,
total and target signed cancellation, occurrence counts, unique-detector
counts, target-only and without-target signal, signal contrast, deletion
response, and deletion-identity residual. The response-versus-predictor
summaries use ten stable equal-count bins, including signed leverage,
absolute-mass share, quadratic share, total cancellation, detector
redundancy, and absolute signal contrast. These summaries are descriptive;
there is no dominance or detector-quality cutoff.

Retain complete derived maps for the deletion identity and the signed
coefficient-square expansion `C^2 - 2 C C_t + C_t^2`. None is a calibrated
standalone detector sky product.

## Resource and execution boundary

Use one configured thread and `--grppiex seq`, at most one hour, 64 GiB peak
memory, and 8 GiB retained output. A replacement replay is allowed only for an
environmental interruption. Preserve all external inputs and prior products;
use an isolated copied checkpoint and output root. No Unity activity is
authorized.
