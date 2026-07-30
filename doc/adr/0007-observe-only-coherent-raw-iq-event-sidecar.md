# ADR 0007: Observe-only coherent raw-I/Q event sidecar

Date: 2026-07-30

Status: Accepted

## Context

Late-night pointing data contain large phase-level transitions that affect
many tones in selected readout networks. Tone-level analysis shows that each
network response is well described, to varying degrees, by a stable
network-specific low-rank loading pattern excited with different amplitudes
and signs. Existing RTC diagnostics and learned masks represent these events
as many detector-local records and discard the simultaneous raw-I/Q event
identity.

The available templates have only been tested within one night and observing
corpus. They are not yet known to generalize across retunes, LO placement,
firmware, IF state, or observing conditions. Automatic subtraction or masking
would therefore risk biasing astronomical and atmospheric signals.

## Decision

Citlali will first integrate this model as an opt-in, observation-local,
observe-only sidecar.

When enabled, the sidecar:

- retains the existing RTC step and impulsive summaries without enabling
  their masking paths;
- clusters those summaries into shared candidate epochs;
- attempts a raw-I/Q mode score for every network present at every shared
  epoch, regardless of which networks seeded the candidate;
- requires one versioned, network-specific, fail-closed compatible template
  for each score;
- rereads only bounded raw-I/Q windows after RTC rather than retaining
  observation-length complex data in the science pipeline; and
- writes a required atomic diagnostic product with requested configuration,
  realized counts, template hashes, compatibility status, and network-event
  scores.

The observer does not change samples, flags, weights, learned state, or maps.
It has no network allow-list. Missing or incompatible templates are explicit
diagnostic outcomes.

## Consequences

The implementation adds bounded raw-file I/O and diagnostic computation when
explicitly enabled. This cost is accepted for the initial validation phase
because it keeps the science path isolated and avoids adding mutable
cross-cutting state to `Engine`.

Candidate selection remains dependent on existing RTC step/impulsive
summaries. Expanding every shared candidate across all present networks lets
the observer measure sub-threshold network responses, but it cannot discover
an event that seeds no RTC candidate anywhere. That limitation must be
measured before the observer is considered complete.

No score threshold has masking authority. Event-level masking may be proposed
later through the existing source-protected mask proposal path. Model
subtraction remains experimental until cross-state template validation and
astronomical signal-injection tests pass.

## Superseding evidence

This decision should be revisited if continuous raw-I/Q projection is shown to
recover materially important events missed by RTC seeding, or if the bounded
reread becomes a demonstrated runtime bottleneck. Any masking or subtraction
authority requires a separate ADR and validation record.
