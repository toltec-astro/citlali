# ADR 0020: Pre-certified RTC filter bank and science error budgets

> Canonical numbering note: this decision first appeared as divergent ADR
> 0017 at
> `doc/adr/0017-precertified-rtc-filter-bank-and-science-error-budgets.md`,
> introduced by `a8c7924e998f29eed9e4b26f965b05eafcae3c7b`. It entered
> canonical application ancestry as ADR 0020.
> The historical number remains a provenance locator only.

Status: accepted scientific-owner budgets 2026-08-30; scan-maximum lookup
superseded by ADR 0022; artifacts and implementation pending

Decision owners: Citlali project owner and scientific owner

## Context

ADR 0019 correctly made RTC bandwidth and factor planning scan- and
array-specific, but its first numerical closure treated filter choice too much
like a generic signal-processing optimization. It imposed sub-percent generic
response bounds and a runtime-constructible Kaiser rule without connecting the
cost directly to mapped astronomical distortion or the noise expected to
survive cleaning.

The relevant scientific risks are additional distortion of a beam-convolved
source and additional retained noise from broadband aliasing. They can be
certified offline for a small set of immutable filters. Ordinary reduction
setup then needs only an authoritative scan velocity, array, and cadence.

Narrow sub-input-Nyquist lines are owned separately by the established line-
detection/mitigation strategy. They do not justify making every generic
anti-alias filter longer.

## Decision

The current numerical policy is
[`wp7-rtc-scan-array-numerical-policy-v2`](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md).
It supersedes only the v1 astronomical-response budget, generic folded-alias
bound, and runtime filter-construction rule.

Each filter-bank entry must keep the additional mapped astronomical distortion
within `1%` and the broadband alias contribution within `1%` of the retained
cleaned-product noise variance. Certification compares native-rate and
filtered/decimated point-source and OOF cases through both naive and JINC, and
through fruitloops for the OOF route. A `1%` passband-magnitude limit is a
conservative engineering surrogate, not a separate tighter science claim.

Broadband alias certification uses representative photon, detector, readout,
and residual-atmosphere PSD envelopes appropriate to the noise expected after
cleaning. It must not normalize only against removable raw atmospheric power.
Sub-input-Nyquist lines are excluded from that broadband envelope and handled
by line detection/mitigation; any line that could alias must be effectively
mitigated before decimation.

Filter family, design method, and tap count are offline engineering choices.
Each immutable entry declares an array, exact cadence domain, factor, and
margin-aware physical upper-speed ceiling. ADR 0022 supersedes scan-maximum
lookup and suspends automatic largest-factor selection pending representative
retained-support evidence. Production still may not synthesize or optimize a
filter or estimate PSDs during an ordinary reduction.

All unlisted ADR 0019 and v1 authority remains unchanged, including the array
beam, full optical temporal support, factor universe, beam sampling, support,
arithmetic, safety margins, network timing, paired semantics, and inadequate-
input `M=1` disposition.

## Consequences

- The v1 Kaiser calculator, candidate factors, and tap counts are historical
  feasibility evidence and cannot select a production plan.
- A candidate filter is accepted initially only if the same entry passes naive,
  JINC, and OOF/fruitloops certification; mapmaker-specific RTC policy is not
  introduced.
- The runtime planner is simpler, but the project must build and version the
  representative PSD, map/OOF comparison, and coefficient-certification
  artifacts before nonidentity RTC acceptance.
- ADR 0021 closes AST velocity/validity authority; its conforming
  implementation remains a prerequisite.
- The line path remains a distinct scientific and implementation concern; this
  decision changes neither its algorithm nor its source-protection semantics.
- ADR 0022 separates candidate occurrence/support loss from numerical transfer
  and defers automatic factor selection; the response and alias budgets here
  remain unchanged.

## Supersedes

This ADR partially supersedes the numerical-policy portion of
[ADR 0019](0019-scan-array-rtc-bandwidth-planning.md). ADR 0019 remains the
authority for occurrence admission, per-scan/per-array planning, network-
specific output timing, and its other unmodified decisions.

## Evidence and authority

- [Filter-bank owner authority](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [Historical v1 numerical packet](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
- [Scan/array planning owner authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Network-timing decision](0018-network-specific-timing-and-common-analysis-grid.md)
- [AST scan-motion decision](0021-ast-scan-motion-velocity-and-validity.md)
- [Occurrence-level upper-speed decision](0022-occurrence-level-rtc-upper-speed-admission.md)
