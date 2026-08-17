# SCI-RTC v0.1 — Supersession Cover For The Reusable RTC Core

Status: owner-approved and content-bound author reference

Prepared: `2026-08-17`

This cover accompanies only:

`3319d7424c732c1c9fc300c336e4d428e6f91068:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex`

Verified content SHA-256:
`d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7`.

The core is reusable implementation-independent science. Its associated audit,
source inspection, findings, repairs, tests, validation, re-audits,
conformity claims, and production status are not author references.

## Binding V0.1 Specializations

The following owner-approved decisions control wherever the reusable core is
broader, older, or ambiguous.

1. **Product-role signal domains.** V0.1 does not impose one unit on every RTC
   output. The primary frozen Beammap input is raw fractional frequency shift
   `Delta f/f`. A separately authorized CAL path may use `mJy/beam`. Signal
   quantity, unit, sign, reference/baseline, and response are explicit for
   every role; raw and calibrated products never substitute silently.
2. **Imported calibration, owned ordering.** SCI-BEAM/SCI-CAL own calibration
   factor production and meaning. When a role imports the CAL operator, RTC
   owns its exact ordered application and response consequences but does not
   derive or repair it. Calibration precedes cross-detector replacement unless
   a different order proves complete unit, factor, additive-offset, response,
   uncertainty, and validity equivalence. For raw donor `q` and target `d`,
   valid compatible factors under the declared convention
   `z_i = flxscale_i x_i` authorize replacement scale
   `flxscale_q / flxscale_d`. Both factors must be valid for the exact detector
   occurrences under the same calibration convention and domain, and the
   target factor must be nonzero. Frozen SCI-BEAM assigns legacy
   `responsivity` no canonical role, and this route does not require it.
3. **Transitive influence ineligibility.** Any signal corrupted by ALIGN
   synthesis or RTC replacement, and any downstream output influenced by it,
   is ineligible for scientific analysis. Compact cause/support bookkeeping is
   mandatory; dense per-sample provenance is not.
4. **Complete response or unavailable.** Every enabled response-changing RTC
   stage, detector mixing, mask, state, edge rule, and sampling phase is
   represented in the realized local/factorized response. A partial kernel is
   never called the complete conditioned response. If truthful economical
   representation is not available, the response is explicitly unavailable.
5. **Immutable stage identities.** Outer, inner, full, mini, diagnostic,
   simulated, and processed products have distinct immutable stage identities
   and explicit parent/processing links. This does not require duplicate
   computation.
6. **Bounded filter and multirate semantics.** Exact coefficients,
   normalization, coefficient convention, precision, state, edge rule, phase,
   rate, representative time/support, and alias status are serialized at full
   precision once per coherent observation or processing segment. The
   supported approximately 8-ms lattice relationships (`0.5x`, `2x`, and
   `4x`) are conditional software-grid facts, not physical event timing.
7. **Phase-zero selection.** The authoritative v0.1 downsampling operator is
   point selection `y[n]=u[M n]` on the declared zero phase. Arithmetic-mean
   or other block-aggregate downsampling is not authorized. The exact selected
   input/output phase, support represented by each output, time-grid identity,
   flag/validity/influence propagation, realized transfer, and unavailable
   states are retained.
8. **Fixed and learned modes.** RTC sampling has fixed and optional learned
   requests. Learned mode has distinct requested, bootstrap, learned,
   resolved, and applied states. Apply consumes one immutable resolved plan
   and cannot retune or silently fall back.
9. **Maximum safe reduction.** The first learned applied scope selects the
   largest admitted integer factor only after every astronomical-transfer,
   alias-rejection, sampling, filter-realizability, and downstream-compatibility
   constraint passes. It uses the smallest admitted beam, maximum valid
   in-scan speed, exact realized FIR response, and phase-zero multirate
   response. Percentile speeds are diagnostic, not safety authority.
10. **Common learned plan.** The first learned applied scope uses one common
    observation cadence and filter across arrays and scans. Noise-aware,
    per-array, per-scan, heterogeneous-cadence, and continuously adaptive
    execution require successor authority. Numeric tolerances and fallback
    policy remain explicit owner decisions.
11. **No source-injection requirement for the analytic operator.** Exact
    analytical beam-times-filter and alias response is the scientific
    authority for learned-plan admission. Deterministic vectors may verify an
    implementation; source injection is not required to define the linear
    operator.
12. **Coordinate-dependent controls.** A source mask or coordinate-dependent
    response consumes an admitted AST coordinate identity, frame, topology,
    detector binding, and validity. Invalid or unavailable coordinates are not
    outside-source values and fail the affected operation. A mask is an
    operator control, not acquisition validity.
13. **Selected policy, not imported defaults.** Despike/donor, source-mask,
    FIR/IIR/notch, state, edge, non-finite, and recovery choices are versioned
    selected policies subject to the contract. The core's generic equations
    do not select current implementation defaults or universal numerical
    thresholds.
14. **Atomic output.** TOD values alone are insufficient. The role-specific
    output also carries identity, complete-response status, support/influence,
    typed causes and flags, validity inputs, conditional uncertainty
    availability, one-way state/provenance, required-output completion, and
    scientifically named diagnostics.
15. **Claim separation.** Algebraic contract correctness, implementation
    conformity, representation fidelity, observational performance, and
    production readiness are separate claims. The reusable core establishes
    none of the achieved implementation or observational claims.

## Corrections To Read Literally

- In the displayed total-covariance equation labeled `RTC-23`, the first term
  is `Sigma_y^stat`; the missing typesetting escape is a transcription error,
  not a distinct variable.
- Core language allowing a block aggregate as an alternative to point
  selection is superseded by binding item 7.
- Core language assigning only zero direct weight to a synthesized or replaced
  cell is strengthened by binding item 3 to include every influenced output.
- Core language that initially assumes calibrated `mJy/beam` is restricted by
  binding item 1 and cannot govern the raw Beammap role.
- Audit target, branch, finding, validation, and production prose in the core
  records provenance of its derivation and is not scientific status for this
  library package.

## Permitted Use Of The Core

The implementation-blind author may:

- reuse its definitions, equations, conditional reasoning, response and
  covariance derivations, limiting cases, and falsifiable predictions;
- consolidate them under the library's shared-core and science-team rationale
  house standard;
- reconcile them with the binding specializations above; and
- identify a precise remaining scientific ambiguity rather than inventing an
  answer.

The author may not:

- consult or cite the associated audit, source trace, findings, repairs,
  re-audits, tests, reductions, validation, or status;
- repeat the central operator, multirate, response, or covariance derivation
  merely to appear new;
- infer a current filter, threshold, donor, mask, edge, unit, or product policy
  from implementation;
- weaken a binding owner decision to match remembered behavior; or
- broaden v0.1 to PTC, VAL, MAP, FLT, source fitting, or fruit-loop science.

This cover was approved by Grant Wilson on `2026-08-17` and is binding through
the exact content hash in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).
