# ADR 0005: Defer Measured R-Channel Execution

- **Status:** Accepted
- **Recorded:** 2026-07-16
- **Decision owners:** Citlali project owner, scientific owner, and engineering

## Context

The KIDs solver exposes the primary science phase stream `x` and a measured
quadrature stream `r`. The R channel can diagnose detector/readout noise,
electronics modes, tuning, leakage, and null behavior. It is not a synthetic
kernel. Applying the primary optical calibration, PCA subtraction, or mapmaking
policy to R without an explicit scientific contract could produce
plausible-looking but physically ambiguous results.

The refactor contains structure for an optional auxiliary measured stream, but
no active RTC/PTC operator, product contract, or validated dataset establishes
enabled R execution.

## Decision

Preserve a first-class distinction among the primary measured science stream,
an optional measured quadrature stream, and synthetic/model kernels. Do not
implement or enable R-channel execution during this refactor.

Before enabled execution, the scientific owner must approve a contract for:

- channel identity and source selection;
- sample/detector shape and alignment with X through gap handling and
  downsampling;
- native and calibrated units;
- optical calibration and extinction policy;
- flag propagation and missing-data behavior;
- which linear RTC/PTC operations share the primary transfer function;
- independent R diagnostics/PCA and any conditions for influencing X;
- learning-state use and source-leakage safeguards;
- optional TOD, diagnostic, and null-map product identity; and
- requested/effective/realized provenance.

The first implementation must have a disabled/no-cost path and an enabled
reference dataset. R-derived modes do not clean X merely because they exist;
any such operation requires measured coherence/leakage evidence and a separate
intentional-science decision.

## Consequences

- Current science behavior and performance remain unchanged.
- The architecture does not close the door to future polarimetric or auxiliary
  control-channel development.
- The existing scaffold is structure-only and may be revised before execution;
  its fields are not a validated public scientific contract.
- Selecting `rs` as the primary legacy TOD type does not prove the semantics of
  simultaneous X/R processing.
- Future R work is a separate scientifically owned project with its own
  products, validation profile, and science-change ledger entries.

## Rejected Alternatives

- **Treat R as a kernel sidecar:** conflates measured data with a synthetic
  transfer/model quantity.
- **Calibrate and clean R exactly like X by default:** the optical meaning is
  not established.
- **Blindly subtract R-derived modes from X:** may inject R noise or remove
  real optical signal when leakage is present.
- **Delete the structural concept:** would unnecessarily harden the pipeline
  around one measured matrix and make later work more invasive.

## Supersession

A successor ADR is required before R execution. It must reference the approved
measured-channel contract, disabled/no-cost tests, an enabled reference
dataset, product/provenance contracts, and the allowed influence on primary
science processing.

## Evidence

- [`../R_ANALYSIS_AUXILIARY_CHANNEL_NOTE_2026-07-08.md`](../R_ANALYSIS_AUXILIARY_CHANNEL_NOTE_2026-07-08.md)
- [`../SCIENTIFIC_CONVENTIONS.md`](../SCIENTIFIC_CONVENTIONS.md), capability
  boundary
- `include/citlali/core/timestream/auxiliary_stream.h`
