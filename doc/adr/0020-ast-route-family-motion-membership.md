# ADR 0020: AST route-family motion membership

Status: accepted bounded scientific-owner authority 2026-08-30; clean
implementation and representative execution pass; exact-SHA conformance
review pending

Decision owners: Citlali project owner and scientific owner

## Context

ADR 0018 closed the scan-motion estimator only for the exact real TolTEC
`Science/Lissajous`, 50 Hz, J2000 family. The WP-7 filter/downsampling
certification matrix also requires Beammap and OOF witnesses. Applying the v1
operator to those files without an authorized route profile would invent
physical-scan membership locally; treating their unavailable motion as zero
loss would be scientifically false.

Read-only profiling established that OOF and Pointing Lissajous files have the
same producer cadence and realized J2000 trajectory contract as the accepted
Science family. The rectilinear Beammap instead contains producer hold states
and repeated scan/turn intervals, so it requires an exact physical-membership
predicate before the unchanged estimator can be applied.

## Decision

The controlling successor authority is
[`wp7-ast-scan-motion-v2`](../WP7_AST_ROUTE_FAMILY_MOTION_OWNER_DECISION_PACKET_2026-08-30.md).
It supersedes only the v1 supported-family and physical-membership clauses.
Every v1 numerical field, frame, continuity boundary, telemetry-defect rule,
eleven-record derivative, scalar speed, maximum, validity, ownership, and
network-specific mapping rule is preserved.

V2 admits these exact profiles:

- `Science/Lissajous`, `Oof/Lissajous`, and `Pointing/Lissajous`, with the
  complete native telescope observation scope as physical membership; and
- the bounded zero-offset, no-hold-during-turns, azimuth-coordinate,
  continuous rectilinear `BeamMap/Map` profile named by the owner packet.

For the Beammap profile, membership first requires producer `Hold == 0`. The
finite realized corrected horizon offset is then rotated by the configured
scan angle and admitted on the inclusive configured rectangle. The exact
producer fields and formula are fixed by the owner packet. A maximal native-
contiguous member run is one compact physical segment, and no defect or
derivative support may cross its boundary.

Nonzero hold and a finite outside-footprint occurrence are truthful expected
non-members, not missing AST coverage. A non-finite required membership field
makes membership unavailable and prevents a complete scan maximum. V2 retains
short physical segments and does not adopt the legacy first-record trim or
two-second range rejection as AST authority.

The product remains network-independent at telescope time. ALIGN maps it to
each network's independent occurrence/time axis. No common-analysis-grid
relation is requested or constructed.

## Consequences

- The AST product identity becomes `SCI-AST:scan_motion_planning@2` with policy
  `wp7-ast-scan-motion-v2`.
- The immutable source carrier may hold the seven additional Beammap
  membership planes once; the compact derived record does not duplicate them.
- Product records expose membership and compact segment identity separately
  from continuity runs, derivative validity, and processing chunks.
- Expected Beammap non-members remain locally inspectable with typed causes and
  do not make the physical-scan maximum incomplete.
- The existing v1 observation-152390 maximum remains unchanged and must be
  reverified as a regression gate.
- Filter choice, downsampling, RTC activation, persistent TOD publication,
  CAL, PTC, MAP/JINC, and legacy Beammap range behavior remain outside this
  decision.

## Supersession

This ADR partially supersedes ADR 0018 only for its exact supported input
family, producer-state interpretation, and physical-scan membership. ADR 0018
remains the numerical and product-semantics authority underneath v2.

A future profile, nonzero-offset map interpretation, different producer
cadence, changed membership predicate, or changed numerical estimator requires
another named authority and representative evidence.

## Evidence and authority

- [Approved route-family owner decision](../WP7_AST_ROUTE_FAMILY_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
- [V1 estimator owner decision](../WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
- [V1 durable numerical decision](0018-ast-scan-motion-velocity-and-validity.md)
- [Network-specific timing decision](0015-network-specific-timing-and-common-analysis-grid.md)
- [Filtering/downsampling certification plan](../WP7_RTC_FILTER_DOWNSAMPLING_CERTIFICATION_TEST_PLAN_2026-08-30.md)
- [Clean implementation and representative evidence](../../handoff/WP7_AST_ROUTE_FAMILY_MOTION_ACCEPTANCE_PACKAGE_2026-08-30.md)
