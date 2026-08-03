# SCI-MAP-002 JINC coverage owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-COVERAGE-001`

Authority: project owner

## Decision

The existing JINC coverage plane is retained with the defined identity
**coefficient-squared effective integration time**:

\[
T_{c^2} = \sum_{\mathrm{eligible\ samples}} \frac{c_i^2}{f_{s,i}}.
\]

Its units are seconds. It measures response-weighted temporal support and is
not geometric exposure, a hit count, or a validity mask. A sample with an
analytic JINC coefficient of zero contributes zero to this plane even if its
footprint geometrically reaches a pixel.

The plane must be joined to the realized coefficient/phase convention and
sample-frequency provenance. Downstream edge/support logic may use it only in
combination with the authoritative formal-support validity mask; it may never
replace that mask. No geometric-exposure product is required unless a future
approved consumer has a concrete need for one.

This decision preserves the existing diagnostic and corrects its scientific
labeling. It does not authorize code changes, Unity evidence, repair,
re-audit, production-status change, or a new output format.
