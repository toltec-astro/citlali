# Fit gate, checkpoint, and runtime audit 04

## Trigger and scope

The owner stopped further real-observation execution after the instrumented
ObsNum 150818 run spent one hour without reaching a scientific result. This is
a diagnostic lifecycle and performance repair only. It does not change the
frozen signal model, coordinate reconstruction, support, optimizer starts,
objective arithmetic, bootstrap draws, or any production Citlali code.

The exact stopped-run progress log has SHA256
`78e8eabc602e61258a213a942a4238e844f979efc9bab948787fd80b5610a189`.
It contains 723 events and ends when the 3,600-second deadline is detected in
bootstrap timestream attempt 83. No `result.json` was published; the retained
75 finite bootstrap values are an incomplete computational checkpoint and are
not scientific evidence.

## Read-only speed audit

The new `audit-runtime` projection was run against that already-retained log.
Its checksum-valid output is local at
`sci_align_001_lissajous_runtime_audit_2026-08-11/o150818_revision8` beneath the
thread visualization root. `runtime_audit.json` has SHA256
`71b07498e844203805c9a328bb3799d295570df25767b81ac826a74f4d03d297`;
its `SHA256SUMS` has SHA256
`da3fcf01072705902cbea8c35bd0a509a0d7f5330b4288b9aac137fa126d424a`.

The observed stage durations were:

| Stage | Seconds | Disposition |
| --- | ---: | --- |
| full model fits | 422.745 | new mandatory owner-review gate |
| objective profile | 132.834 | checkpoint after completion |
| derivative cross-check | 0.145 | checkpoint after completion |
| held-out model comparison | 2252.377 | dominant completed cost; checkpoint |
| sensitivity fits | 26.113 | checkpoint |
| network sensitivity | 169.890 | checkpoint |
| map scan accumulators | 0.690 | inexpensive reconstruction |
| bootstrap before deadline | 594.821 | existing realization checkpoint |

The audit records 307 completed optimizer attempts and 26 fallback events.
Held-out fitting alone accounts for 2,247.6 optimizer-attempt seconds; the
bootstrap accounts for 588.4 seconds before interruption. Historical events do
not contain function-evaluation counts, so those counts are reported as
unavailable rather than reconstructed.

## Mandatory first-phase fit gate

Direct full execution through `analyze-observation` is now prohibited. The
`fit-gate` command performs only authenticated input preparation, coordinate
reconstruction, authenticated map-result loading, and the four full-data model
fits. On normal ObsNum 150818 timing this is approximately seven minutes. It
then stops and writes a separate checksum-bound package containing:

- the complete full-model results and optimizer census;
- per-scan best-lag, zero-lag, and constant-model residual metrics;
- a three-page PDF with model objectives, multistart convergence, per-scan
  residual/timing-leverage diagnostics, and the four highest-leverage aggregate
  source profiles;
- the exact progress slice and input, support, coordinate, map-result, and
  implementation identities; and
- an automatic structural status that never tests the fitted value of tau.

A failed primary fit also produces a review package. In that case the PDF
states which detailed diagnostics are unavailable and resume is prohibited.
Passing structural checks is necessary but not sufficient: the owner must
inspect source compactness, residual structure, multistart behavior, scan
leverage, and scan/network diversity. The measured tau is descriptive at this
gate and cannot justify an implementation change.

## Explicit resume and durable stage checkpoints

`resume-observation` requires both a checksum-valid fit-gate package and the
explicit `--owner-review-approved` flag. It reauthenticates every frozen input,
support identity, coordinate reconstruction result, map result, implementation
digest, and gate digest. It loads the serialized full fits without refitting
them. A synthetic round-trip regression proves that the optimizer vector and
objective are bit-identical before and after JSON serialization.

Objective profiling, the derivative cross-check, held-out comparison,
sensitivity fits, and network sensitivity each write an atomic, checksum-bound
stage checkpoint immediately after success. A later resume verifies and reuses
those exact values. Tampering, source drift, incomplete manifest/state pairs,
unknown stages, and gate-identity drift fail closed. Bootstrap retains its
existing deterministic scan-draw and paired map/timestream realization
checkpoint.

## Scientific disposition and stop

This repair recovers no new real-observation result and changes no prior
scientific conclusion. It only makes the first interpretable phase visible and
prevents already-completed expensive stages from being discarded. No further
real observation was opened. The next permitted human decision is whether to
run the new fit gate on the already-known successful bright anchor ObsNum
150818, inspect that package, and separately authorize resume. ObsNum 136280
remains deferred.
