# Real-data runtime observability stop 03

## Trigger

On 2026-08-11 the owner authorized a fresh revision-6 ObsNum 136280 run after
the historical bootstrap optimizer defect was repaired. The process remained
before its first `bootstrap_work.npz` checkpoint after 66 minutes. A read-only
process audit found approximately one CPU second per wall second, 98--100%
CPU use, and active NumPy interpolation/arithmetic in a sampled stack. It was
computing rather than blocked, but the executable emitted no stage identity,
optimizer-attempt count, fallback count, or enforceable runtime bound.

The owner stopped PID 3520 after its command line was reauthenticated as the
new-root ObsNum 136280 diagnostic. The process terminated after `SIGTERM` and
left the output root empty: no result, checkpoint, or partial scientific
artifact exists.

## Why this is a real diagnostic failure

The revision-6 fallback is invoked by every fit supplied with an inherited
initial condition, not only bootstrap fits. Before bootstrap, a 12-scan
observation executes four full fits, 48 leave-one-scan-out fits, two model
sensitivities, and up to 14 network sensitivities. If every inherited fit
requires fallback, the deterministic pre-bootstrap optimizer-attempt ceiling
rises from approximately 88 to 424. The weak, structured ObsNum 136280 source
can therefore incur a materially larger workload than the previously
successful pointings.

That workload explains why an hour is plausible; it does not make an
unobservable, unbounded execution acceptable. CPU occupancy alone cannot
distinguish expected multistart work from pathological repeated fallback.

## Narrow instrumentation repair

The diagnostic-only runner now:

- writes timestamped stage, optimizer-attempt, fallback, and bootstrap-progress
  events immediately to `progress.jsonl`;
- writes a machine-readable `run_state.json` with running, failed, or complete
  status and aggregate optimizer counters;
- reports stage transitions, fallbacks, and 25-attempt bootstrap checkpoints
  to stderr without printing every optimizer attempt;
- accepts `--maximum-wall-seconds` and checks that deadline inside every
  timestream objective evaluation, so one long optimizer cannot silently
  overrun it; and
- checksum-binds the completed progress and run-state records.

With no deadline supplied, the numerical objective, optimizer options, starts,
support, scan draws, and result arithmetic are unchanged. When the deadline is
reached, the diagnostic fails closed and preserves its progress and run state;
it does not publish a completed result.

## Visualization installation

The generalized `visualize_sci_align_001_lissajous_fit.py` tool consumes a
checksum-valid completed result for any frozen selected pointing. It rebuilds
the exact fixed support and model evaluation, profiles a same-support tau=0
comparator, records support and input identities, and creates the detailed
fit-unit, residual-atlas, source-footprint, scan-profile, leverage, and standard
map context products requested by the owner. It labels retained PTC-weight
residuals as `sqrt(weight)`-scaled rather than sigma-standardized because the
available product does not authenticate a per-sample uncertainty.

## Next bounded real-data action

ObsNum 136280 remains deferred as potentially pathological. The next real run
is the previously successful bright anchor ObsNum 150818, in a new output root,
with an explicit 1,800-second wall limit. Its purpose is to validate normal
instrumented lifecycle and the installed renderer, not to reopen or extend the
corpus inference. No timing correction follows from this run.

This planned action was superseded after the bounded anchor attempts by the
mandatory seven-minute gate and resumable-stage contract in
`FIT_GATE_CHECKPOINT_RUNTIME_AUDIT_04.md`.
