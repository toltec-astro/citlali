# SCI-RTC-001 core phase-zero downsampling owner amendment

Date: 2026-08-11

Decision ID: `SCI-RTC-001-CORE-DOWNSAMPLING-PHASE-ZERO-001`

Status: owner-approved phase-independent core RTC authority; documentation
only; core repair not launched

## Authority and separation

This amendment applies only to the future phase-independent core RTC repair
under
[`SCI-RTC-001_PHASE_INDEPENDENT_BOUNDED_REPAIR_HANDOFF_2026-08-09.md`](SCI-RTC-001_PHASE_INDEPENDENT_BOUNDED_REPAIR_HANDOFF_2026-08-09.md),
SHA-256
`872b7045d0eba1263c7615529004ee74ae16d649f1ad666aa441626537cddcb7`.

It is separate from the learned-sampling Stage A repair and does not add,
close, or reinterpret any Stage A finding.

## Owner decision

Phase-zero point selection is the authoritative RTC downsampling operator.
Arithmetic-mean downsampling is not authorized.

A future core RTC repair must preserve and truthfully publish the exact
phase-zero selection contract, including:

- selected input/output phase and factor identity;
- the exact input support represented by each selected output;
- flag, validity, eligibility, and cause propagation associated with that
  support;
- the selected representative time and output time-grid identity;
- the exact realized filter/multirate state and resulting transfer/response;
- requested, effective, observation-resolved, and realized provenance; and
- explicit unavailability whenever the complete support, time-grid, flag, or
  transfer identity cannot be represented truthfully.

This authority does not prove physical integration-event timing. The
ALIGN-deferred assigned-time compatibility interface remains binding, and
physical event semantics, absolute timing, sub-sample placement, and timing
correction remain unavailable.

## Non-authorization

This amendment does not launch or implement core RTC repair, authorize
arithmetic averaging, change Stage A, change production, authorize Stage B,
access Unity, run reductions, merge, push, or launch downstream work.
