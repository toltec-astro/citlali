# Coadd Config Authority

This document fixes the bounded Phase 2 contract for `coadd.enabled`.

## Authority Flow

The merged YAML value is read once into `CoaddConfig`, preserving the user’s
request. `CoaddExecutionPlan` copies that request and resolves effective
activation. Coadd is effectively enabled only when both the user requested it
and mapmaking is effectively enabled. Resolution never writes back into the
requested config.

Execution predicates consume only the effective plan. Existing coadd buffer
allocation, accumulation, normalization, diagnostics, filtering, and output
algorithms are unchanged.

## Provenance

Every successful CLI reduction must atomically publish
`coadd_provenance.yaml` using schema `citlali-coadd-provenance-v1`. It records:

- requested activation;
- effective activation and whether mapmaking disabled the request;
- whether coaddition executed;
- realized map and required logical write cardinality; and
- successful output and reduction completion.

Realized cardinality is copied one way from the mapmaking output lifecycle at
successful CLI completion. The reduction audit cross-checks both sidecars and
rejects disagreement. A required coadd-provenance write failure fails the
reduction.

## Gate

Local acceptance requires the CLI/test builds, full CTest suite, config
preflight including the frozen one-path boundary audit, and focused
reduction-audit tests. Unity point validates the disabled/no-output case.
Science validates enabled raw/filtered coadd cardinality and unchanged
products.
