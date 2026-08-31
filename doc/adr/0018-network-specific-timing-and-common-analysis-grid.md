# ADR 0018: Network-specific timing and explicit common analysis grids

> Canonical numbering note: this decision first appeared as divergent ADR
> 0015 at `doc/adr/0015-network-specific-timing-and-common-analysis-grid.md`,
> introduced by `50aa0409e559dfd983c236bc6933687c52f9ad03`. It entered
> canonical application ancestry as ADR 0018.
> The historical number remains a provenance locator only.

Status: accepted 2026-08-29; bounded implementation and exact-SHA conformance
review passed

Decision owners: Citlali project owner and scientific owner

## Context

ADR 0017 correctly separated the first identity RTC witness from ALIGN's
common-slot machinery, but its ordinary-route diagram still placed a singular
ALIGN common-slot projection before complete RTC. The frozen SCI-ALIGN and
SCI-RTC packages likewise used a singular observation-wide detector-reference
grid. That model conflated two distinct timing responsibilities:

1. reconstructing and preserving each network's occurrences and times; and
2. associating multiple networks with common analysis epochs for an operation
   that mathematically couples simultaneous measurements.

TolTEC networks have independent acquisition occurrence/time vectors. Ordinary
RTC, CAL, AST association, MAP/JINC accumulation, and network-level PTC/PCA do
not jointly estimate an output from multiple networks at one epoch merely
because they process several networks or use rectangular storage.

## Decision

Network-specific timing is authoritative by default. Ordinary scientific
identity is conceptually

```text
(observation, network, occurrence) -> reconstructed time
```

Paired `x/r` shares the same occurrence within its originating network.
Ordinary RTC consumes and produces network-keyed streams. An `M=1` operation
preserves each network's occurrence identity and time exactly. A sampling
operation produces a new per-network occurrence/time/support relation unless
its named mathematics explicitly couples multiple networks synchronously.

A cross-network **common analysis grid** is a separate, explicitly requested,
derived relation. ALIGN constructs it because ALIGN owns the timing knowledge.
The relation preserves its source network occurrence, source time, validity,
causes, origin, and support and does not destructively replace the network
axis. Rectangular storage, shared configuration, pooling, or convenience cannot
request it.

The successor route is therefore:

```text
network-native paired x/r -> network-timed ALIGN relation
  -> network-keyed RTC -> CAL / ordinary AST / MAP-JINC
  -> network-level PTC-PCA

network-keyed product
  -> explicitly requested ALIGN common-analysis-grid relation
  -> array-wide PTC-PCA or an authorized cross-network RTC method
```

"Common analysis grid" is reserved for the second path. A paired x/r axis
within one network is not called a common grid.

## Relationship To ADR 0017

This ADR partially supersedes ADR 0017's ordinary-route diagram and its wording
that could make common-slot projection a prerequisite for complete RTC. ADR
0014 remains authoritative for bounded subsystem succession, paired x/r,
explicit routing, one-way adapters, immutable lifecycle state, evidence gates,
technology direction, and legacy-route activation policy.

## Consequences

- Ordinary RTC cannot depend on the shared-slot carrier or common-slot
  association helpers.
- A gap in one network does not manufacture a slot or absence state in another
  network.
- Network-level PTC remains network-local. Array-wide PTC must explicitly
  request and bind a common-analysis-grid relation.
- AST evaluates ordinary occurrence-local state at each network time.
- Existing common-slot code may remain only behind an explicitly named
  common-analysis-grid interface with source-network facts preserved.
- The legacy route remains production-authoritative until the repaired
  successor passes its gates.

## Evidence And Authority

- [Network-timing owner correction](../WP7_NETWORK_TIMING_OWNER_AUTHORITY_CORRECTION_2026-08-29.md)
- [Network-timing authority crosswalk](../WP7_NETWORK_TIMING_AUTHORITY_CROSSWALK_2026-08-29.md)
- [WP-7 implementation baseline](../WP7_TIMESTREAM_SUCCESSOR_IMPLEMENTATION_BASELINE.md)
- [`validation/wp7_timestream_successor_authority.json`](../../validation/wp7_timestream_successor_authority.json)
