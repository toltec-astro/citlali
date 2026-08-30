# ADR 0017: Pre-certified RTC filter bank and science error budgets

Status: accepted scientific-owner correction 2026-08-30; artifacts and
implementation pending

Decision owners: Citlali project owner and scientific owner

## Context

ADR 0016 correctly made RTC bandwidth and factor planning scan- and
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
Production Citlali performs a bounded lookup of the largest certified factor
whose immutable entry admits the array, exact cadence, and margin-adjusted scan
velocity. It does not synthesize or optimize a filter or estimate PSDs during
an ordinary reduction.

All unlisted ADR 0016 and v1 authority remains unchanged, including the array
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
- Conforming AST velocity/validity authority remains a prerequisite.
- The line path remains a distinct scientific and implementation concern; this
  decision changes neither its algorithm nor its source-protection semantics.

## Supersedes

This ADR partially supersedes the numerical-policy portion of
[ADR 0016](0016-scan-array-rtc-bandwidth-planning.md). ADR 0016 remains the
authority for occurrence admission, per-scan/per-array planning, network-
specific output timing, and its other unmodified decisions.

## Evidence and authority

- [Filter-bank owner authority](../WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [Historical v1 numerical packet](../WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
- [Scan/array planning owner authority](../WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [Network-timing decision](0015-network-specific-timing-and-common-analysis-grid.md)
