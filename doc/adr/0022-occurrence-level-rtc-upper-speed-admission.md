# ADR 0022: Occurrence-level RTC upper-speed admission

> Canonical numbering note: this decision first appeared as divergent ADR
> 0019 at `doc/adr/0019-occurrence-level-rtc-upper-speed-admission.md`, introduced by
> `4e8eae0237df492a67223fda365980319f3be71a`. It entered canonical application
> ancestry as ADR 0022.
> The historical number remains a provenance locator only.

Status: accepted scientific-owner correction 2026-08-30; census harness
conforming; representative evidence partial; production implementation pending

Decision owners: Citlali project owner and scientific owner

## Context

ADR 0019 planned each scan/array mode from `1.05` times the maximum valid scan
speed. Representative observation 152390 proved that its accepted
`221.40490828695155 arcsec/s` maximum is real but extremely sparse: 62 of
62,067 derivative-valid telescope records exceed `200 arcsec/s`, and only five
exceed `220 arcsec/s`. Treating that tail as representative made the complete
scan fail the shortest-array native-cadence screen.

A percentile replacement would be observation-dependent and physically
incorrect. Conversely, processing unsupported high-speed astronomical
content through a mode would make the preservation claim false.

## Decision

AST retains the full valid velocity product and truthful raw maximum. Each
array, exact cadence family, and certified RTC mode instead declares an
inclusive physical upper-speed ceiling after the approved velocity and cadence
margins. A network occurrence above that ceiling is unavailable for that mode
with pair-wide RTC cause `scan_speed_above_mode_support`; its AST value,
validity, and member-local producer causes remain intact.

`M=1` admission is occurrence-local. A filtered or decimated output is valid
only when its complete input footprint is admitted. No invalid, slow,
upper-speed-excluded, gap, slew, or non-science occurrence contributes across
a support boundary. Network-specific timing remains authoritative.

Scan-wide `v_max` no longer selects or rejects an ordinary RTC mode. The old
largest-factor rule is suspended because candidate modes retain different
support. The certification program measures raw and support-eroded occurrence,
duration, weighted-exposure, coverage, map/OOF, response, noise, and
performance consequences. Automatic factor selection requires a later bounded
owner decision; it is not replaced with a percentile or arbitrary retained-
sample fraction here.

## Consequences

- The current D0/D1 custody, cadence, AST, and network-mapping evidence remains
  valid, but scan-maximum factor dispositions become superseded diagnostics.
- Candidate certification uses a native-rate reference with the same
  candidate admission/support domain, while separately reporting the cost of
  that domain relative to all valid native data.
- A bounded high-speed tail does not by itself cause
  `input_cadence_inadequate_for_science_band`; the final no-product cause and
  selection policy remain pending evidence and owner closure.
- The conforming harness enumerates occurrence-level structural bounds and raw
  admission without activating a numerical RTC route or selecting a production
  factor. Its first exact-SHA execution completes the authorized Science/
  Lissajous motion census and truthfully leaves Beammap/OOF motion evidence
  pending.

## Evidence and authority

- [Occurrence-speed owner authority](../WP7_RTC_OCCURRENCE_SPEED_ADMISSION_OWNER_AUTHORITY_2026-08-30.md)
- [Scan/array planning authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Filter-bank numerical policy](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [AST scan-motion decision](0021-ast-scan-motion-velocity-and-validity.md)
- [Network timing decision](0018-network-specific-timing-and-common-analysis-grid.md)
- [First exact-SHA D0/F0 evidence](../../handoff/WP7_RTC_OCCURRENCE_SPEED_CENSUS_EVIDENCE_2026-08-30.md)
