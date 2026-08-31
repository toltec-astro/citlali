# WP-7 AST Scan-Motion Authority Crosswalk

Date: 2026-08-30

Controlling bounded authority:
[`wp7-ast-scan-motion-v2`](WP7_AST_ROUTE_FAMILY_MOTION_OWNER_DECISION_PACKET_2026-08-30.md),
approved by the scientific owner on 2026-08-30 and recorded in
[ADR 0020](adr/0020-ast-route-family-motion-membership.md). The v1 numerical
operator remains governed by
[ADR 0018](adr/0018-ast-scan-motion-velocity-and-validity.md).

Source authority is the frozen SCI-AST v0.1/r0.3 package bound by
`validation/wp7_timestream_successor_authority.json`, together with the
accepted network-timing correction and RTC scan/array planning authority.
Frozen text remains historical and is not silently edited. This crosswalk
states only the bounded successor dispositions needed by the AST scan-motion
planning role.

## Authority map

| Frozen entry or topic | Bounded successor disposition |
| --- | --- |
| SCI-AST exact telescope field registry, including open question Q001 | `Data.TelescopeBackend.SourceRaAct` and `SourceDecAct` remain the realized boresight direction differentiated in equatorial J2000 radians for every v2 profile. The exact seven-field horizon/hold registry in the v2 owner packet is used only to determine rectilinear Beammap membership; it is not differentiated or substituted for celestial motion. |
| Producer `Hold`/state interpretation and physical scan membership | Science/OOF/Pointing Lissajous use the complete exact observation scope. The exact bounded rectilinear Beammap uses `Hold == 0` and the inclusive realized corrected in-footprint predicate. Maximal native-contiguous member runs have compact physical-segment identity. Legacy first-record trim and two-second rejection are not AST authority. |
| Native timing continuity and gaps | Exact producer times remain authoritative. A continuity run admits finite strictly increasing adjacent intervals `0 < dt <= 0.030 s`; equality is continuous and a larger interval splits support. The bound is not rescaled from chunk size. |
| Telescope telemetry defect | An eleven-record, canonical tangent-plane, component-wise median pair-slope/intercept test marks a record defective only for radial residual strictly greater than `2.0 arcsec`; equality is valid. The raw direction is preserved and receives a typed defect cause rather than being clipped or moved. |
| Scan-motion derivative and scalar speed | New bounded successor authority: use the same eleven records in one valid continuity run for an unweighted quadratic least-squares fit in exact time. The two `b1` coefficients are J2000 east/north velocity; their norm is speed in arcseconds per second. No one-sided endpoints, direct-difference fallback, extra smoothing, percentile, or commanded-rate substitution is authorized. |
| Frozen spherical topology | Preserved. Circular differences use `[-pi, pi)`, and exact antipodes remain typed unavailable rather than resolved by local unwrapping. |
| Frozen role-local validity and dependency-limited failure | Preserved and specialized with exact scan-motion causes for unsupported family, scope/time/direction failure, topology, gaps, defect support, defects, derivative support, rank/nonfinite failure, network mapping, and incomplete maximum. Causes remain local and do not overwrite ALIGN, paired-member, or RTC causes. |
| Raw physical-scan maximum | AST owns the uninflated actual maximum over complete valid raw motion records with `v >= 1 arcsec/s`, including the maximizing record. It is diagnostic and does not select an RTC mode. RTC separately owns each network occurrence's inclusive lower-speed admission and `below_minimum_science_scan_speed` cause plus mode-specific inclusive upper-speed admission and `scan_speed_above_mode_support` cause. An incomplete candidate population cannot be labeled the scan maximum. |
| Network timing and ALIGN mapping | ADR 0015 is preserved. ALIGN maps adjacent valid raw AST records to each network's exact occurrence/time axis without extrapolation or invalidity crossing. Each view retains grid-independent source time and support. Ordinary RTC does not request or depend on a cross-network common analysis grid. |
| Product ownership and lifecycle | The AST role owns only derived compact facts and references immutable producer axes, directions, optional Beammap membership planes, scan identity, and ALIGN mappings through bounded typed handles. Requested, effective, observation-resolved, and realized identities bind the exact v2 policy without full-plane hashing, per-cell identity duplication, or generalized provenance. |

## Preserved and unaffected authority

- Ordinary SCI-AST coordinate realization, pointing corrections, frames not in
  the bounded J2000 role, exact ALIGN parentage, and dependency-limited failure
  remain governed by their existing authority.
- Paired `x/r` identity, independent member validity and causes, conservative
  pair-wide RTC action, and the accepted identity RTC route are unchanged.
- Network-specific timing and explicit placement of a requested common
  analysis grid remain governed by ADR 0015.
- RTC array/scan planning and filter-bank numerical policy remain governed by
  ADRs 0016 and 0017. AST publishes motion facts; it does not select filters or
  inspect detector values.
- CAL, VAL, PTC/PCA, MAP/JINC, persistent TOD publication, and production
  activation are not opened.

## Still pending

The bounded v2 implementation now passes exact synthetic profile/membership
boundaries, chunk/order invariance, network independence, clean Science/OOF/
Pointing/Beammap execution, and the full local repository gates at revision
`672f907355a3f15f3ee987d92a5f7e95bbdc38b5`. Fresh independent exact-SHA
review remains required before final conformance. Another map profile,
cadence, or simulation family requires a later named successor.

The separate versioned PSD/filter-bank, naive/JINC, OOF/fruitloops, and line-
path prerequisites also remain pending before nonidentity RTC acceptance.
