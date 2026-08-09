# SCI-ALIGN-001 producer-authority evidence request authorization

Date: 2026-08-09

Record ID: `SCI-ALIGN-001-PRODUCER-AUTHORITY-REQUEST-001`

Authority: project owner

Status: owner-mediated evidence acquisition and read-only analysis authorized;
evidence not yet supplied or accepted

## Exact dependency context

The accepted bounded acquisition-event-semantics package is commit
`08f0a6733d1cb523ae78ccf9348ac6832b834e52` (parent
`92cfa670a33255250895d68aaf26e8b01aa057bd`, tree
`cb46a319270e9a05a1763100ad3075bbf61b65dc`). Its dedicated TEL handoff is
`validation/sci_align_001_acquisition_event_semantics_2026-08-08/SCI_TEL_INPUT_001_HANDOFF.md`
at that commit, SHA-256
`28a39458872e00d0c36c48eac79a3f1f1f154c1d5bb815159094fa60883257d2`.

That package preserves delivered detector `D[n]`/`Ts[n]` row lineage through
Citlali but does not identify the physical integration event represented by a
detector row. It inspected neither exact FPGA/packetizer producer authority
nor telescope-row/20-ms ingress behavior. Detector `PpsTime` is not telescope
`PpsTime`, and the package authorizes no timing correction or prior.

Accordingly, `08f0a673...` remains bounded post-core evidence and an explicit
unavailable upstream dependency. It is not physical-state identification and
must not be promoted to producer authority.

## Authorized owner-mediated request

The owner may acquire one exact, immutable producer-authority package through
the responsible instrument/backend authority. The returned authority must be
either:

1. the exact FPGA, packetizer, and backend source revision(s) governing the
   relevant acquired data; or
2. an authoritative event-level specification bound to the exact applicable
   hardware/firmware/software revision and data epoch.

The source or specification must bind, in execution order and with explicit
counter/time semantics:

- detector integration accumulation and its boundary event;
- counter capture relative to accumulation completion;
- PPS-state update ordering and the counter/PPS values attached to a record;
- UDP packet construction, segmentation, and assembly into a logical detector
  record; and
- backend NetCDF row insertion, including which event/time state each stored
  row represents.

The return must identify exact revision objects or immutable specification
identities, applicable hardware/data scope, artifact paths, digests, provenance,
and any unavailable or ambiguous boundary. Read-only source tracing and
deterministic static analysis are authorized only as necessary to produce that
binding.

## Admission and interpretation boundary

Acquisition does not equal acceptance. The returned package must stop for
coordinator and owner review before it can become ALIGN authority or enter any
TEL pre-core/post-core manifest. Until then:

- producer authority remains `unavailable`;
- physical detector-row integration-event identity remains unresolved;
- no integer-row or half-cadence timing interpretation is selected;
- no timing correction, reassociation, interpolation policy, or prior follows;
  and
- no ALIGN, AST, TEL, RTC, CAL, PTC, MAP, VAL, or BEAM conclusion is expanded.

## Explicit exclusions and stop

This authorization permits evidence acquisition and read-only analysis only.
It does not authorize Unity access or requests, reductions, application/test/
config changes, correction or repair, re-audit, downstream launch, production
change, merge, push, or external scientific interpretation beyond the exact
producer event binding. Any unavailable revision, scope conflict, ambiguous
event ordering, or need for active experimentation stops for owner review.
