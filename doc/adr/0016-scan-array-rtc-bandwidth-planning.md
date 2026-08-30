# ADR 0016: Scan/array RTC bandwidth and decimation planning

Status: accepted scientific structure 2026-08-29; numerical closure and
implementation pending

Decision owners: Citlali project owner and scientific owner

## Context

The first nonidentity proposal after the accepted WP-7 identity RTC route was a
fixed `M=2` conformance witness. It isolated useful support and response
questions, but selected a factor independently of the astronomical workload.
Frozen SCI-RTC text also deferred per-array/per-scan factors in favor of one
immutable observation plan.

The controlling science is instead set by how fast the telescope scans each
array's diffraction-limited beam. TolTEC arrays have different reference beam
sizes, scans have different realized speeds, and ordinary products remain on
network-specific time axes under ADR 0015. One historical cutoff, factor, or
output cadence is therefore not scientific authority.

## Decision

RTC applies an exact science occurrence-admission boundary at valid realized
on-sky speed `v >= 1 arcsec/s`. Slower valid occurrences are unavailable as
independent astronomical measurements with the typed cause
`below_minimum_science_scan_speed`. Invalid AST or telemetry facts retain
their distinct causes. Admitted occurrences form bounded filter runs; invalid
or slow occurrences cannot influence retained outputs.

For each scan, use the actual maximum scalar on-sky speed over AST-valid
science occurrences meeting the threshold. For each array, use one immutable
circular diffraction-limited reference beam at its nominal center frequency.
Derive the required astronomical temporal band from the scanned beam and
approved product-level distortion tolerances.

For each scan, array, and exact input cadence, select the largest factor from
an approved finite integer set for which the simplest permitted low-pass
realization satisfies the complete astronomical passband, phase, alias,
sampling, support, edge, and paired-operator constraints. If no factor greater
than one passes, select `M=1` without a sampling change while retaining the
planner's occurrence-admission dispositions; never narrow the required science
band to make a factor fit. The separate accepted identity-RTC conformance
context remains unchanged.

Plans are immutable before Apply and independent of detector values and chunk
partitions. Different arrays may realize different factors, filters, and
cadences. Every sampling-changing result creates a new network-scoped
occurrence/time/support relation. Equal plan values do not create a common
grid; ADR 0015 continues to govern explicit cross-network relations.

## Numerical gate

This ADR approves the structure, not numerical defaults. Implementation awaits
an authoritative array-model artifact and exact universal values for
astronomical distortion, phase/centroid, alias error, beam sampling, factor
set, filter families/tie rule, support/edge loss, arithmetic precision, and
uncertainty margins. Existing code constants and the legacy `32 Hz` cutoff are
evidence only.

## Consequences

- The pushed fixed-`M=2` packet commit remains useful historical design work,
  but its fixed-witness recommendation is superseded.
- The first nonidentity RTC method now has an AST dependency for immutable
  trajectory, derivative-validity, and science-scan facts. This does not move
  timing or pointing authority into RTC.
- Slow-motion admission is a pair-wide astronomical decision while `x/r`
  producer validity and member-local causes remain distinct.
- Filtering is run-bounded; edge and invalid-support consequences are explicit
  scientific availability, not hidden padding or chunk behavior.
- A deterministic planner may share compact immutable plans and reusable
  workspaces, but architecture and kernel choices remain benchmark-driven.
- No nonidentity numerical implementation begins until the numerical gate is
  closed.

## Evidence and authority

- [Scan/array planning owner authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Scan/array planning authority crosswalk](../WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md)
- [Network-timing decision](0015-network-specific-timing-and-common-analysis-grid.md)
- [Revised RTC planning packet](../WP7_RTC_FIXED_DECIMATION_OWNER_DECISION_PACKET_2026-08-29.md)
