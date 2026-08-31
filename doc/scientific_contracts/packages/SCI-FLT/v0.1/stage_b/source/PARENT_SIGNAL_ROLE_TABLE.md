# SCI-FLT-FIXED v0.1 Parent Signal-Role Table

Record identity: `SCI-FLT-FIXED-PARENT-SIGNAL-ROLE-TABLE v0.1/draft-r0.3`

Status: implementation-blind Stage B closure draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-30`

## Exact parent roles

```text
Parent role             Exact m used by FLT-SIG
FLT-PARENT-MAP-OBS      base/unfiltered MAP observation signal
FLT-PARENT-MAP-COADD    base/unfiltered MAP coadd signal
FLT-PARENT-JINC-OBS     normalized jinc_map on admitted local support
```

## Atomic JINC binding

The JINC parent retains all exact boundary roles:

```text
jinc_signal_numerator
jinc_signed_normalization
jinc_quadratic_accumulator
jinc_map with local support/validity
jinc_coefficient_squared_time
```

Only `jinc_map` is transformed as `FLT-SIG`. The other four roles remain
parent diagnostics or accounting facts. Parent response and covariance follow
their typed compositions. Support and validity remain predicates and records.
Exposure remains lineage. No other map-shaped role is transformed without a
separately named method.

## Nonclaims

This table supplies no numerical parent, implementation, validation,
calibration, performance, readiness, freeze, production, or Unity claim.
