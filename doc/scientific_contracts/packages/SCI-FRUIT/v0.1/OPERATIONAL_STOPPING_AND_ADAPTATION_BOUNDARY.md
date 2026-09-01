# SCI-FRUIT v0.1 — Operational Stopping And Adaptation Boundary

Status: **Stage A candidate boundary; no stopping or adaptation policy selected**

## Causal Operational Policy

A qualified stopping policy may use only prospectively declared diagnostics
available causally during an ordinary reduction. Injected truth,
historical-control output, future iterations, final desired quality, or an
oracle-selected result may evaluate a stopping policy but cannot drive it.

Reports keep the following distinct:

- the actual terminal iteration under the candidate policy;
- the oracle best iteration under truth, for evaluation only;
- iterations and wall time to a declared quality region;
- nonconvergence or censored time-to-quality;
- hard-cap termination; and
- oscillation and drift.

Any iteration, wall-time, memory, resource, or convergence cap capable of
changing the terminal scientific output is part of `METHOD_ID`, specifically
its `stopping_and_terminal_policy`. Reaching a cap is not scientific
convergence by definition.

## Bounded Adaptation

Bounded automatic adaptation may use only declared causal diagnostics through
one deterministic, bounded, versioned mapping. Its inputs, cadence, state,
bounds, failure behavior, and replay/checkpoint requirements are part of the
method. Unrecorded manual tuning and output-based selection are experimental
overrides and do not inherit ordinary qualification.
