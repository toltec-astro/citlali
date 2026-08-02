# SCI-CAL-001 EL25 Confirmation Failure Record

## Decision

The preregistered confirmation is **invalid**, not a numerical pass or fail. Execution stopped at the frozen achieved-coordinate floating-point guard before complete evidence existed. Numerical representation fidelity is invalid/unavailable, observational performance was not evaluated, and this record authorizes neither operator adoption nor an operational domain.

No partial band integration, candidate error metric, ranking, maximum-error search, or observational inference was performed by this recorder.

## Exact Stop

- Case: `q50_q75_trisect_2/LMT_DJF_50`
- Trace: `scale_traces/q50_q75_trisect_2_LMT_DJF_50.json` with 99 evaluations
- Expected achieved tau225 float: `1.34988558021834681e-01`
- Recomputed tau225 float: `1.34988558021834626e-01`
- Absolute discrepancy: `5.55111512312578270e-17` (2 ULPs)
- Frozen guard threshold: `4.99999999999999990e-17`
- Excess over threshold: `5.51115123125782807e-18`
- The cache contains no captured exception log. The discrepancy is reconstructed from the SHA-bound frozen runner expression, the canonical 99-evaluation trace, and the frozen preregistration operand; it is not presented as a verbatim log.

## Preserved Coverage

- Complete cases: 12 / 16
- Complete full grids: 672 / 896
- Missing full grids: 224
- The failing case has its completed scale trace and zero full grids. Three later cases were not started.

## Cache Integrity

- Execution context SHA-256: `a867df7b05ea590c498e41932bb1b3f9520e635d2534f7c8fcc539cfd4a12ecf`
- Raw outputs and matched sidecars: 1953
- Scale traces: 13
- Rejected AM failed-attempt files: 0
- Evidence aggregate SHA-256: `25ee2a1b2f793f5273e714dc0094bb7e39ebc76615e1fe424d2d65a95013956d`
- Internal AM cache: 21637 files across 8 shards, aggregate `9141c5fb61f8d6a7265ef6b6fd0d70b8a7ed113d0950a9f787dc62480bda087d`

The decision JSON contains the complete per-file SHA-256 inventory for raw outputs, execution sidecars, scale traces, and failed attempts. Internal AM cache members are summarized by deterministic per-shard and whole-cache aggregates.
