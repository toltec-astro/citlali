# ADR 0019: Scan/array RTC bandwidth and decimation planning

> Canonical numbering note: this decision first appeared as divergent ADR
> 0016 at `doc/adr/0016-scan-array-rtc-bandwidth-planning.md`, introduced by
> `38644343a4f8cfa213c8cab87c06753377704e12`. It entered canonical application
> ancestry as ADR 0019.
> The historical number remains a provenance locator only.

Status: partially superseded by ADR 0020 and ADR 0022; AST authority closed by
ADR 0021; bounded AST implementation passes local, representative-data, and
fresh exact-SHA gates; certified filter bank and RTC implementation pending

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
network-specific time axes under ADR 0018. One historical cutoff, factor, or
output cadence is therefore not scientific authority.

## Decision

RTC applies an exact science occurrence-admission boundary at valid realized
on-sky speed `v >= 1 arcsec/s`. Slower valid occurrences are unavailable as
independent astronomical measurements with the typed cause
`below_minimum_science_scan_speed`. Invalid AST or telemetry facts retain
their distinct causes. Admitted occurrences form bounded filter runs; invalid
or slow occurrences cannot influence retained outputs.

AST retains the actual maximum scalar on-sky speed over AST-valid science
occurrences meeting the threshold. ADR 0022 supersedes use of that maximum as a
whole-scan planning boundary: each array/cadence/mode declares a physical upper
speed and excludes only occurrences above it. For each array, use one immutable
circular diffraction-limited reference beam at its nominal center frequency.

For each scan, array, mode, and exact input cadence, enumerate candidate
occurrence admission and complete-support consequences while preserving all
astronomical passband, phase, alias, sampling, support, edge, and paired-
operator constraints. ADR 0022 suspends the largest-factor and scan-wide
inadequate-input fallback rules until representative retained-support evidence
supports a separate owner selection decision. Never narrow the required
science band or use a percentile to make a factor fit. The separate accepted
identity-RTC conformance context remains unchanged.

Plans are immutable before Apply and independent of detector values and chunk
partitions. Different arrays may realize different factors, filters, and
cadences. Every sampling-changing result creates a new network-scoped
occurrence/time/support relation. Equal plan values do not create a common
grid; ADR 0018 continues to govern explicit cross-network relations.

## Numerical policy

The scientific owner originally approved
[`wp7-rtc-scan-array-numerical-policy-v1`](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
on 2026-08-30. Later that day,
[ADR 0020](0020-precertified-rtc-filter-bank-and-science-error-budgets.md)
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
- No nonidentity numerical implementation begins until the approved ADR 0021
  AST role is implemented and conforming, the v2 certified-bank prerequisites
  are satisfied, and the bounded implementation gates pass.
- ADR 0022 retains the raw AST maximum but moves upper-speed consequences to
  candidate-specific occurrences and defers automatic factor selection.

## Evidence and authority

- [Scan/array planning owner authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Scan/array planning authority crosswalk](../WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md)
- [Current v2 filter-bank policy](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [Historical v1 numerical packet](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
- [Network-timing decision](0018-network-specific-timing-and-common-analysis-grid.md)
- [AST scan-motion decision](0021-ast-scan-motion-velocity-and-validity.md)
- [Occurrence-level upper-speed decision](0022-occurrence-level-rtc-upper-speed-admission.md)
- [Revised RTC planning packet](../WP7_RTC_FIXED_DECIMATION_OWNER_DECISION_PACKET_2026-08-29.md)
