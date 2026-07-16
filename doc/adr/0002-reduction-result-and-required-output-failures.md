# ADR 0002: Reduction Results And Required Output Failures

- **Status:** Accepted
- **Recorded:** 2026-07-16
- **Decision owners:** Citlali project owner and engineering

## Context

Historical library paths could log a write error, call `exit()`, or allow a
partially written product to coexist with apparent reduction success. Ordered
concurrent writers could leave other workers waiting after one stream failed.
This made CLI status, product completeness, and scientific success different
and sometimes contradictory facts.

The project owner decided that a required write failure must fail the
reduction. Such failures indicate an invalid delivered result and should not be
hidden as best-effort behavior.

## Decision

Reusable reduction execution returns a structured
`citlali::session::ReductionResult`. It contains a classified status,
path-aware diagnostics, published product roots, and published provenance
artifacts. Canonical config, I/O, output, runtime, and internal failures
propagate to the session boundary. Only the CLI reports process diagnostics and
maps the result to a process exit code.

Library code does not call `exit()` and does not log-and-forget a required
failure. A required FITS, NetCDF, ECSV, CSV, config manifest, or required
provenance failure makes the result unsuccessful. An optional diagnostic must
be explicitly classified as optional in its product contract; optionality is
not inferred from a catch block.

Ordered writers record the first failure, cancel the output domain, and wake
all waiters before the owning thread rethrows. Expected product/write
cardinality is checked independently of individual append success. Atomic
publication removes temporary files on failure where the format permits it.
Any partial product has an explicit failed-run disposition and is never enough
to make the run successful.

## Consequences

- CLI exit status, run audit, and required product completeness have one
  definition of success.
- Worker exceptions are drained and reported from an owning thread rather than
  escaping a parallel worker.
- Failure injection must test cancellation, wake-up, partial-product state,
  nonzero CLI status, and successful recovery in the same process.
- Product contracts determine requested, required, forbidden, and genuinely
  optional families.
- `ReductionResult` may evolve to carry richer realized records, but process
  policy remains outside it.

## Rejected Alternatives

- **Log and continue:** can deliver plausible but incomplete science products.
- **Call `exit()` at the failure site:** prevents reuse, cleanup, testing, and
  caller control.
- **Treat every diagnostic as optional:** hides product-contract violations.
- **Let each writer cancel only itself:** can deadlock workers waiting on
  another ordered stream.

## Supersession

A future result/error mechanism may supersede the concrete type, but it must
retain structured non-process failure, fatal required outputs, explicit
optional policy, cancellation safety, and CLI-only exit selection.

## Evidence

- [`../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md`](../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md)
- [`../PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md`](../PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md)
- `tests/test_ordered_writer.cpp`
- `tests/test_output_schema.cpp`
- `tests/test_reduction_session.cpp`
- `validation/product_contracts.json`
