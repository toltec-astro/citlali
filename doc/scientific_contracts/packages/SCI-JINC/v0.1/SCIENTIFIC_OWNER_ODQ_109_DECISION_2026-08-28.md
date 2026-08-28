# SCI-JINC-ODQ-109 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

## Approved Scientific Disposition

SCI-JINC conditioning and numerical realization are governed by scientific
adequacy, not by floating-point pathology as an end in itself. Numerical error
from accumulation, reduction order, finite arithmetic, analytic-function
evaluation, phase quantization, and cache/index realization must remain
negligible compared with the approximately `10^-3` relative fidelity relevant
to the instrument. No stronger precision or reproducibility guarantee is a
scientific requirement of SCI-JINC v0.1.

The word **negligible** is comparative: numerical error must be materially
smaller than the approximately `10^-3` instrument-fidelity scale, not merely
equal to it. This decision does not invent a sharper universal constant. A
later engineering conformance procedure must use a scientifically meaningful
comparison and demonstrate that the chosen realization meets this principle;
the particular algorithm, test construction and handling of comparisons near
zero are engineering-conformance matters rather than new scientific product
semantics.

## Conditioning Semantics Preserved

The following recovered scientific rules remain binding:

- contributors and the required accumulators must satisfy their finite-state
  requirements;
- `Q_p>0` and `C_p!=0` are required for local formal JINC support;
- exact signed cancellation is locally invalid rather than zero sky;
- finite negative `C_p` is scientifically admissible;
- `rho_p=abs(C_p)/sum_i I_ip abs(omega_i kappa_ip)` remains a useful
  dimensionless conditioning indicator;
- conditioning must remain invariant under signal-unit changes and stable
  common coefficient rescaling; and
- no unit-bearing absolute `C_p` or `Q_p` floor, silent clipping, or substitute
  sky value is authorized.

A finite nonzero normalization is usable only when the realized calculation
can establish adequate accuracy under the scientific principle above. If
near-cancellation prevents that showing, the affected pixel is locally
invalid under the existing support/validity rules. SCI-JINC v0.1 does not
require a universal `rho_p` cutoff or a contributor-count/machine-epsilon
formula as scientific authority.

## Numerical And Discrete Realization

The phase-quantized point-evaluation and fully populated square-support
conventions remain binding. Effective `subpixel_n` is an integer at least one,
and its realized phase approximation must satisfy the same scientific
accuracy principle. The center rounding, half-pixel tie, phase-bin edge and
representative, cache-extent rounding, summation algorithm, reduction order,
and parallel organization must each be single-valued and internally
consistent, but no particular choice is a separate scientific-owner decision
when the accepted operator semantics and accuracy budget are preserved.

SCI-JINC v0.1 therefore does not require:

- a prescribed summation algorithm or contributor-count error formula;
- serialization of a machine-specific floating-point bound;
- bitwise identity, a fixed reduction order, or exact sequential/parallel
  reproducibility;
- a machine-epsilon-level proof or precision materially stronger than the
  instrument-relevant need; or
- owner selection among numerically adequate tie, bin-edge, representative,
  cache-rounding, or accumulation realizations.

## Supersession And Stage Consequence

This disposition supersedes the machine-specific portion of recovered
`SCI-MAP-002-D003-CONDITIONING-001` that required a documented bound derived
from the realized summation method and contributor count, including its
serialization requirement. It also closes the remaining numerical-policy
questions attached to `SCI-MAP-002-D003-SUBPIXEL-001` by replacing exact
owner-selected tie/bin/convergence rules with the scientific-adequacy
criterion above. The estimator, signed-cancellation, point-phase, square-
support and no-unit-bearing-floor decisions are preserved.

`SCI-JINC-ODQ-109` is closed for Stage A. The next unresolved scientific-owner
question is `SCI-JINC-ODQ-110`, the finite-map rule for a rounded sample center
outside the map whose square support overlaps it. This decision changes
sanitized Stage A author-control bytes and remains subject to renewed
exact-byte approval under `SCI-JINC-STAGE-A-Q002`. It does not launch Stage B,
prescribe an implementation, perform validation, or establish achieved
fidelity, conformity, readiness, or production status.
