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

- copies threshold-passing RTC step and impulsive seeds into a compact
  observation-local cache before standard detailed-diagnostic cleanup,
  without enabling their masking paths;
- clusters those summaries into shared candidate epochs;
- attempts a raw-I/Q mode score for every network present at every shared
  epoch, regardless of which networks seeded the candidate;
- requires one versioned, network-specific, fail-closed compatible template
  for each score;
- opens each present network once, reads its receive-time and tone-coordinate
  vectors once, and then reads only bounded raw-I/Q windows for all of that
  network's candidates; and
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

An observation-wide network-event score budget is part of that bound. If the
projected candidate-by-network workload exceeds the configured budget, the
observer writes an explicit `skipped_workload_budget` sidecar instead of
silently truncating an order-dependent subset. Progress and realized raw-file
open/time-vector-read counts make the observer visible in logs and provenance.
Required science products and raw provenance are published before the
observe-only sidecar is evaluated, so observer cost or failure cannot prevent
those products from being written.

The compact seed cache is owned by RTC diagnostics and cleared after the
required observation sidecar is written. It is deliberately independent of
the detailed per-detector QA cache, which remains free to publish and clear
scan by scan.

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

The first corrected observation-152433 Unity smoke demonstrated that the
original reread was not operationally bounded: 12,177 projected
network-event scores produced 2.46 TB of logical reads before the job was
cancelled. Inspection found one raw-file open and full receive-time read per
score plus quadratic coincidence comparison. The 2026-07-31 scaling repair
therefore makes the accepted bounded-reread language concrete through
network-batched I/O, event-keyed coincidence, progress, and a global budget;
it does not change event selection or score meaning.

The first completed scaling-repair smoke processed 12,408 network-event
scores with 11 raw-file opens and 11 receive-time-vector reads, validating the
new I/O lifecycle. All scores initially reported `incompatible_tone_map`
because the matched runtime APT preserves flagged unmatched raw-tone rows with
placeholder UID zero. These rows are required to preserve raw column order but
do not carry usable detector identity. The scoring join therefore excludes
rows with no finite usable phase before checking UID uniqueness. A duplicate
among usable rows remains an explicit fail-closed incompatibility.

This decision should be revisited if continuous raw-I/Q projection is shown to
recover materially important events missed by RTC seeding, or if the bounded
reread becomes a demonstrated runtime bottleneck. Any masking or subtraction
authority requires a separate ADR and validation record.
