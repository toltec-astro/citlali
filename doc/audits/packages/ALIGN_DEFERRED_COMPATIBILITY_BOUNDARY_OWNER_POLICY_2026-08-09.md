# ALIGN-deferred compatibility boundary owner policy

Date: 2026-08-09

Policy ID: `ALIGN-ASSIGNED-TIME-COMPAT-001`

Authority: project owner

Status: approved cross-audit compatibility authority; no implementation,
audit, repair, evidence execution, or production launch authorized

## Compatibility interface

The existing assigned-time grid may be consumed only as an exact identified
compatibility interface. It is not evidence of the physical integration event
represented by a detector row and must not be described as physical event
timing, absolute detector timing, or a physical sample centroid.

Every affected downstream contract and realized product must bind the exact
ALIGN/assigned-time identity used. At minimum that identity resolves the
application and ALIGN implementation authority, observation/scan or coherent
segment, detector/sample ordering, realized cadence/rate lattice, assigned-time
values, requested/effective/realized offset or resampling state, parent stage,
and processing identity. The compact record must also carry
`physical_event_semantics: unavailable` or an equivalent typed state until
producer authority is accepted.

No guessed half-sample or whole-sample correction, physical centroid claim,
or detector-time-as-absolute-oracle claim is allowed. Detector `PpsTime` and
telescope `PpsTime` remain distinct facts. The descriptive same-T0 and
acquisition-event evidence at `92cfa670...` and `08f0a673...` is post-core
evidence, not physical timing authority or a correction.

## Fail-closed scientific boundary

Phase-independent implementation and mathematical work may proceed against
the exact identified assigned grid. The following remain unavailable or fail
closed until exact producer authority returns and is accepted:

- absolute detector/telescope physical phase;
- sub-sample astrometric placement and physical sample centroid;
- timing-sensitive source-mask fidelity and source-crossing accuracy;
- timing corrections, reassociation, or offset priors; and
- any precision, accuracy, or response claim that materially depends on those
  physical timing semantics.

When producer authority arrives, re-audit only the timing-sensitive seams and
consumers materially affected by the returned event ordering. Do not reopen
phase-independent mathematics, identity, validity, response bookkeeping, or
implementation solely because the dependency becomes available.

## Queue disposition

1. `SCI-MAP-002` and `SCI-CAL-001` continue only under their already approved
   bounded scopes.
2. `SCI-TEL-INPUT-001` may be frozen as a structural audit of file selection,
   row identity, allowed mutation, cache/atomicity, and Citlali ingress. ALIGN
   `08f0a673...` remains quarantined post-core evidence and the missing
   producer event meaning remains an unavailable dependency. A separate owner
   launch is still required.
3. A phase-independent `SCI-RTC-001` repair handoff may be prepared under
   approved D001--D004 for replacement/influence eligibility, complete
   signal/response parity on the assigned grid, filter/edge/support behavior,
   immutable stage identity, provenance, and local production tests. Absolute
   phase, timing correction, timing-sensitive mask accuracy, and final AST
   placement remain conditioned. A separate owner launch is required.
4. `SCI-AST-001` coordinate mathematics and `SCI-PTC-001` internal estimator
   work may proceed against the identified assigned grid, but this policy
   launches neither task and grants no physical timing or absolute-placement
   claim.
5. `SCI-BEAM-001` remains held.

## Evidence dependency

The owner-mediated producer-authority request
`SCI-ALIGN-001-PRODUCER-AUTHORITY-REQUEST-001` remains active. The requested
return is the exact FPGA/packetizer/backend source revision or authoritative
event-level specification binding accumulation, counter capture, PPS update,
UDP assembly, and NetCDF row insertion. Acquisition is read-only and does not
itself confer acceptance or trigger a correction.

## Non-authorizations

This policy does not authorize application/config/test edits, Unity, local or
external reductions, external contact by the coordinator, repair, re-audit,
TEL/RTC/AST/PTC/BEAM launch, production expansion, merge, or push.
