# WP-7 RTC Occurrence-Level Upper-Speed Admission Authority

Date: 2026-08-30

Scientific owner: Grant Wilson

Authority identity: `wp7-rtc-occurrence-speed-admission-v1`

Status: **approved bounded scientific-owner correction; evidence and
implementation pending**

## Governing correction

The accepted AST velocity product and its truthful raw physical-scan maximum
remain unchanged. The raw maximum is a diagnostic fact; it no longer forces
one RTC bandwidth/factor disposition on every occurrence in a scan.

For each TolTEC array `a`, exact native cadence family `c`, and certified RTC
mode `m`, define an inclusive maximum supported realized celestial-sphere
speed

```text
v_limit(a, c, m).
```

The ceiling is physical, fixed by the mode, and independent of an
observation's velocity percentile or detector values. It is the greatest
unmargined realized speed for which the existing approved requirements all
hold after applying the `1.05` velocity margin and `0.9999` cadence margin,
including:

- full ideal-aperture temporal support inside the entry's certified passband;
- at least four output samples per approved Airy intensity FWHM;
- certified mapped-response, phase, alias, support, and edge behavior; and
- exact paired `x/r`, network timing, and arithmetic semantics.

An upper-bound equality is admitted. An otherwise AST-valid network
occurrence with

```text
v(q) > v_limit(a, c, m)
```

is scientifically unavailable for that array and mode with exact typed RTC
cause `scan_speed_above_mode_support`. This is a pair-wide astronomical action
for the `x/r` occurrence. It does not change AST validity, erase the realized
speed, or replace member-local producer validity and causes.

The existing inclusive lower boundary remains unchanged: valid science motion
is independently admissible only at `v(q) >= 1 arcsec/s`; lower valid speeds
retain cause `below_minimum_science_scan_speed`. Invalid AST state,
telemetry defects, gaps, slews, and non-science motion retain their distinct
causes.

## Support and output validity

AST velocity is mapped onto each network's native occurrence axis. Networks
remain independent; neither the upper-speed rule nor equal array ceilings
create a common analysis grid. Because native occurrence times differ, the
exact excluded occurrences may differ between networks in the same array.

For `M=1` with no numerical filter, upper-speed admission is occurrence-local.
For a filtering or sampling-changing mode, a candidate output is available
only when its complete exact source footprint is producer-valid, AST-valid,
lower-speed admitted, and upper-speed admitted. Filter support may not cross
any such boundary. No interpolation, replacement, padding, or support
renormalization makes incomplete support available. Paired `x` and `r` use
the same occurrence action, operator, and support relation.

The realization reports, by array, network, mode, and typed cause:

- raw excluded occurrence count and duration;
- support-eroded output count and duration;
- weighted exposure removed where weights are scientifically defined;
- retained run lengths and boundary loss; and
- downstream spatial coverage/support consequences.

These are output diagnostics and acceptance evidence. They do not define or
tune `v_limit`.

## Planning and selection consequence

The former scan-wide rule

```text
v_plan,s = 1.05 * v_max,s
```

is superseded as the ordinary RTC mode-admission and whole-scan failure rule.
AST continues to publish `v_max,s`; RTC may retain it in compact diagnostics,
but a rare valid maximum does not make every other occurrence unavailable.

The former instruction to select the largest certified factor admitting the
scan-wide maximum is suspended. Occurrence-level admission makes every
candidate's retained support part of the scientific trade. This authority
does not invent a percentage-of-samples threshold, use an observation
percentile as a cutoff, or authorize the largest factor that leaves merely
nonzero data.

The certification program shall measure candidate-specific loss after full
support erosion and through the required map/OOF products. A separate bounded
owner decision will close automatic factor selection after those results show
the available retained-support, weighted-exposure, spatial-coverage,
response, noise, and performance tradeoffs. Until then the program may
enumerate and test candidates, but it may not freeze a production selection
policy.

Whole-product failure is no longer caused solely by one occurrence exceeding
a mode ceiling. It occurs when no candidate mode produces a conforming and
scientifically usable retained product under the subsequently approved
selection/support policy, or when an existing downstream product contract
fails. The exact terminal cause for that later decision remains to be closed;
`input_cadence_inadequate_for_science_band` is not used merely because a
bounded high-speed tail is excluded.

## Certification comparison

For each candidate mode, separate two effects:

1. **Admission/support cost:** compare the complete AST-valid native
   population with the candidate's occurrence-admitted and support-eroded
   population.
2. **Numerical filter/downsampling effect:** compare the candidate with a
   native-rate reference using the same candidate occurrence admission and
   equivalent complete-support domain.

The existing independent `1%` mapped-response and retained broadband-noise
variance limits apply to the second comparison. They do not conceal the first.
Count fraction alone is not a sensitivity claim: weighting, correlations,
source position, and spatial coverage remain part of representative
acceptance.

## Preserved authority and scope

This correction does not change:

- `wp7-ast-scan-motion-v1`, its validity, derivative, cause, support, or raw
  maximum semantics;
- the exact array frequencies, aperture, Airy model, full optical temporal
  support, four-samples-per-FWHM rule, margins, or factor universe;
- the mapped-response, broadband-alias, line-ownership, phase, DC, support,
  arithmetic, paired, network-timing, or chunk-invariance rules;
- the accepted identity-RTC route; or
- any CAL, VAL, PTC/PCA, MAP/JINC, OOF, publication, or production authority.

No nonidentity RTC production implementation is authorized by this correction.
The immediate work is authority conformance, occurrence/support census repair,
and representative evidence.

## Superseded clauses

This decision narrowly supersedes the scan-wide upper-speed planning,
scan-wide inadequate-input `M=1` consequence, and largest-factor lookup clauses
of:

- `wp7-rtc-scan-array-planning-v1` and ADR 0019;
- `wp7-rtc-scan-array-numerical-policy-v1` items 13--14 where they bind the
  whole scan to `v_max,s`;
- `wp7-rtc-scan-array-numerical-policy-v2` only where it retains those v1
  clauses or performs lookup from the scan-wide maximum; and
- the associated authority crosswalk, test-plan U2/U3 interpretation, status,
  and historical fixed-decimation packet language.

All unlisted authority remains in force.
