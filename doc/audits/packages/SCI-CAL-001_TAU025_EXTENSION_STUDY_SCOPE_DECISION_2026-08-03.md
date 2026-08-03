# SCI-CAL-001 tau225 engineering-extension study-scope decision — 2026-08-03

Status: owner-approved study design; exact execution request remains required

Package: `SCI-CAL-001`

Decision ID: `CAL-ATM-D007`

Authority: project owner

## Approved study design

The owner approves the `SCI-CAL-001-TAU025-ENGINEERING-EXTENSION-001`
planning design, as corrected by coordinator review. This approval applies
only to a future direct-AM engineering-domain evidence study; it neither
adopts an atmosphere operator nor changes Citlali behavior.

### Profile matrix

Use all 25 existing copied AM 12.2 profiles: the `annual`, `DJF`, `MAM`,
`JJA`, and `SON` families at percentiles `5`, `25`, `50`, `75`, and `95`.
The future execution request must name each immutable AMC filename and
SHA-256 from the copied-AM inventory. The generic q95 product remains
ineligible as an extension profile, target, or substitute.

### Direct-AM lattice

Use the following decimal target coordinate design:

- construction opacity nodes: `.15`, `.20`, `.25` at elevations `25`, `35`,
  `45`, `55`, `65`, `75`, and `80` degrees;
- independent held-out opacity nodes: `.1625`, `.175`, `.1875`, `.2125`,
  `.225`, and `.2375` at elevations `29`, `41`, `53`, `67`, and `79` degrees;
- a no-AM post-candidate evaluator diagnostic at `nextafter(.15, -inf)`,
  `.15`, and `nextafter(.15, +inf)` to demonstrate that the same candidate
  identity is continuous at the quality boundary.

The binary64 diagnostic is not an AM target or scale-search coordinate. The
exact parsed-transmission literals, achieved coordinates, scale traces, and
anti-join showing that held-out coordinates are not used to fit or tune a
candidate remain mandatory contents of the later execution request.

### Engineering screen

For this evidence study, approve a maximum absolute held-out fractional
extinction-correction error of **5%** over the complete approved profile,
opacity, elevation, array, and source-index grid. Report the maximum, p95,
RMS, signed extrema, and locations without pooling away a maximum.

This is an engineering-availability numerical screen only. It is neither the
science `<=1%` representation-fidelity criterion nor a 5--10% photometric
accuracy or production-calibration claim.

### Preserved requirements

The existing content-bound TolTECA v1 ECSV passband authority, full
eligible-sample modified-secant airmass, top-of-atmosphere pivot `X_ref=0`,
and `WARN-001` bounded warning-bearing evidence policy remain binding. Any
new warning class, unknown warning, cache mutation, or error fails closed.
The compact observation/declared-segment quality policy remains unchanged:
one declared operator and quality state per unit, classified by its maximum
eligible tau225, with no sample-by-sample switch at `.15` and no per-sample
quality tag.

## Next permitted preparation and stop boundary

CAL may now prepare a documentation-only exact execution request: a complete
tuple/run inventory, immutable cache and provenance layout, AM invocation and
scale-search plan, expected resource estimate, cache/warning stop register,
and preflight/readiness gates. It must stop and return that request for owner
approval before creating a cache, executing AM, fitting an operator, or
interpreting a numerical result.

No Citlali or TolTECA source change, Unity action, repair, re-audit, operator
adoption, operational-domain adoption, production-status change, or new output
format is authorized.
