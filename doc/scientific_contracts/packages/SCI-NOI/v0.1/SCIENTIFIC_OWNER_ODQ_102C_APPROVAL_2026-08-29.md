# SCI-NOI v0.1 — Scientific-Owner ODQ-102C Approval

Decision identity: `SCI-NOI-ODQ-102C`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved ordinary-method balance family; exact finite-design mechanics
remain open

## Exact Owner Decision

> Yes, this choice, which I agree with, will open up other studies down the
> road. That's fine. Let's do the balance within networks originally

## Sanitized Disposition

For `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`, the initial ordinary
assignment family is a network-stratified, coefficient-balanced randomized
sign design. Detector signs remain `-1` or `+1`; ODQ-102B's assignment remains
constant over all admitted samples of that detector throughout the exact
observation.

For one exact observation and one stable readout-network identity, define the
ordinary detector contribution coefficient

```text
B_d = sum_p sum_{i in C_p, detector(i)=d} a_pi,
a_pi = G_pi gamma_i > 0,
```

from the exact frozen MAP-admitted contribution population. The balance design
compares the positive- and negative-sign totals of `B_d` separately within
each network. A detector in one network cannot balance a detector in another
network, and one array or observation cannot balance another.

The admitted assignment set shall be sign-symmetric: if assignment vector
`s` is admissible, its global complement `-s` is admissible under the same
network-local balance facts. The randomized law shall assign equal probability
to complementary assignments, preserving marginal probability `1/2` for each
detector sign. The ordinary method does not additionally require equal counts
of positive and negative detectors.

`B_d` is a derived NOI design coefficient tied to the exact frozen MAP route.
It is not inverse variance, precision, empirical NOI weight, exposure,
validity, or a replacement for the PTC-owned MAP-facing coefficient. A changed
coefficient family, admitted population, network identity, observation, or MAP
plan changes the NOI design identity.

This choice intentionally leaves separately named future NOI methods free to
study observation-global, count-balanced, pixel-vector-balanced,
source-template-balanced, complement-paired, or other declared designs.

## Non-Implications And Remaining Exact Mechanics

This approval does not yet select:

- the numerical balance discrepancy, norm, tolerance, or exact-feasibility
  rule;
- candidate-generation, conditional-sampling, optimization, retry-cap, or
  failure behavior;
- canonical key algorithm/version or requested/resolved member count;
- cross-member dependence, forced complement pairing, replacement,
  equivalence, duplicate treatment, or design-rank rule; or
- a pixelwise source-cancellation, source-free-null, physical-noise,
  covariance, or significance claim.

Those finite-design mechanics remain the next bounded owner decision. The
ordinary route remains numerically unavailable until they and the frozen PTC
coefficient and numerical `coverage_cut` gates are resolved.
