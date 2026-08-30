# ADR 0016: Scan/array RTC bandwidth and decimation planning

Status: accepted scientific structure 2026-08-29; numerical-policy portion
partially superseded by ADR 0017 on 2026-08-30; AST authority closed by ADR
0018; bounded AST implementation locally gated; representative AST
conformance, certified filter bank, and RTC implementation pending

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
than one passes, select `M=1` without a sampling change only when the input
cadence itself still meets the approved science-band and beam-sampling
requirements. Otherwise produce no admitted ordinary astronomical product
with typed cause `input_cadence_inadequate_for_science_band`. Retain the
planner's occurrence-admission dispositions and never narrow the required
science band to make a factor fit. The separate accepted identity-RTC
conformance context remains unchanged.

Plans are immutable before Apply and independent of detector values and chunk
partitions. Different arrays may realize different factors, filters, and
cadences. Every sampling-changing result creates a new network-scoped
occurrence/time/support relation. Equal plan values do not create a common
grid; ADR 0015 continues to govern explicit cross-network relations.

## Numerical policy

The scientific owner originally approved
[`wp7-rtc-scan-array-numerical-policy-v1`](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
on 2026-08-30. Later that day,
[ADR 0017](0017-precertified-rtc-filter-bank-and-science-error-budgets.md)
and
[`wp7-rtc-scan-array-numerical-policy-v2`](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
superseded v1's astronomical-response budget, generic folded-alias bound, and
runtime filter-construction rule. V1 remains historical authority for the
unchanged clauses named by v2; its prototype tap counts are evidence only.

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
- No nonidentity numerical implementation begins until the approved ADR 0018
  AST role is implemented and conforming, the v2 certified-bank prerequisites
  are satisfied, and the bounded implementation gates pass.

## Evidence and authority

- [Scan/array planning owner authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Scan/array planning authority crosswalk](../WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md)
- [Current v2 filter-bank policy](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [Historical v1 numerical packet](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
- [Network-timing decision](0015-network-specific-timing-and-common-analysis-grid.md)
- [AST scan-motion decision](0018-ast-scan-motion-velocity-and-validity.md)
- [Revised RTC planning packet](../WP7_RTC_FIXED_DECIMATION_OWNER_DECISION_PACKET_2026-08-29.md)
