# SCI-JINC v0.1 — Observation Grouping And Product Roles

Status: ODQ-105/106/107/109/110 owner-approved Stage A successor candidate; awaiting
renewed exact-byte approval

Prepared: `2026-08-28`

## Observation-Level Grouping

Base v0.1 defines the estimator and complete product bundle for one
observation and authorizes no cross-observation JINC combination semantics. A
future coadd requires a separately authorized boundary over complete
observation-level JINC bundles; no ordinary SCI-MAP coaddition rule,
accumulator-plane addition or normalized-map combination is imported or
inferred.

The base scientific bundle identity is

```text
observation
x stable TolTEC array
x exact JINC plan
x target WCS
x product role
x lifecycle generation.
```

The supported array identities are exactly `a1100`, `a1400`, and `a2000`.
For one observation, base v0.1 may produce at most one bundle for each stable
array admitted and requested under the exact JINC realization. The produced-
bundle cardinality is zero through three; this is not a requirement to
synthesize all three. Missing, unavailable, or unrequested arrays produce no
placeholder bundle, empty-array product, or synthetic failure product, and
their absence does not invalidate a different produced bundle. Cross-array,
frequency-combined, network-combined, or shared-destination JINC products are
unavailable unless separately authorized.

Observation is the scientific grouping boundary, not a streaming, processing-
chunk, process, container or memory boundary. Samples or chunks from the same
observation may accumulate incrementally into the one observation bundle only
when the exact observation, stable array, JINC plan and realization, target
WCS, admission/parameter/coefficient state and lifecycle generation match.
Chunk identity neither creates a JINC product nor licenses cross-observation
combination.

Each produced bundle is scientifically independent. Contributions whose
stable array or exact destination map identity differs must not be merged.
Requested/effective/resolved/realized state and any absence cause remain at
the existing plan/bundle provenance granularity; no additional per-
contribution provenance is required.

For one array bundle, the population is the union of exact occurrence
candidates whose stable array identity equals the bundle array and whose
observation, JINC plan, target WCS and lifecycle generation match. Each member
retains its exact detector/UID, RTC `n`, PTC group/segment/application
generation, coefficient generation, AST parent and admission evaluation. A
PTC network/group is lineage and an admission domain; it is not a substitute
for stable array identity and does not create a separate JINC bundle unless a
future owner-approved plan says so.

Workers, threads, processes, containers, filenames, map-vector positions and
network order are not product identity. One complete destination identity is
resolved before mutation. Ambiguous, duplicate or conflicting destination
ownership fails the affected bundle atomically; it does not select a winner.

## Fixed Closed Product Schema

Every produced bundle contains exactly the five scientific roles below.
“Required” is a whole-product rule: failure to form any role suppresses the
entire affected bundle. Pixel-level zero, insufficient, cancelled or invalid
support is ordinary content under existing support/validity rules and does not
make a whole role unavailable. No placeholder role is synthesized.

| Role | Mathematical quantity | Participation | Status |
| --- | --- | --- | --- |
| `jinc_signal_numerator` | `N_p=sum_i I_ip omega_i kappa_ip z_i` | Signed numerator used with `C_p` to construct the map. | **Required** |
| `jinc_signed_normalization` | `C_p=sum_i I_ip omega_i kappa_ip` | Signed denominator of `m_p=N_p/C_p`; negative finite values remain admissible. | **Required** |
| `jinc_quadratic_accumulator` | `Q_p=sum_i I_ip omega_i kappa_ip^2` | Required by formal support (`Q_p>0`) and preserves the accepted distinct quadratic statistic; it is not automatically precision, variance, exposure or validity. | **Required** |
| `jinc_map` | `m_p=N_p/C_p` on accepted local JINC support | Published signal-unit map with its local pixel support/validity state; that state is not a separate role-availability object. | **Required** |
| `jinc_coefficient_squared_time` | `T_p^(kappa^2)=sum_i I_ip kappa_ip^2/f_s,i` | Authorized method-specific temporal-support product in seconds; not physical exposure, precision or validity. | **Required** |

The accepted cancellation treatment retains
`rho_p=abs(C_p)/sum_i I_ip abs(omega_i kappa_ip)` as a dimensionless
conditioning indicator. ODQ-109 requires the total numerical error to remain
negligible compared with the approximately `10^-3` relative fidelity relevant
to the instrument. Any absolute-term sum, count, error estimate or diagnostic
used to demonstrate that adequacy is construction state, not a persistent
bundle role; no prescribed summation algorithm or machine-specific bound is a
scientific requirement.

The observation, stable array, exact JINC realization, destination geometry
and lifecycle identify the bundle under ODQ-106. They are not additional
numerical roles or a general provenance product. Existing sample admission,
support, validity and causes govern accumulator contents. Operational
diagnostics may be logged for debugging but are not required products.

ODQ-110 applies one occurrence-level center-domain gate before sample-pixel
support. If the rounded cache center is outside the finite destination map,
`I_ip=0` for all `p`, so that occurrence changes none of the four fixed
accumulators. An admitted in-map center uses ordinary cropped square
membership. No overlap admission, edge correction, role, provenance or
diagnostic is added, and JINC-then-crop equivalence is not required.

## Outside Or Deferred

| Role or family | Base-v0.1 disposition |
| --- | --- |
| Standalone formal-support, per-role availability or detailed-cause products | **Outside/deferred.** Local support/validity stays with `jinc_map` and the accumulator contents; no general availability framework is authorized. |
| Formal weight/variance and covariance products | **Outside/deferred.** `Q_p` remains required, but `C_p^2/Q_p`, `Q_p/C_p^2` and `A C_PTC A^T` are not base products. |
| Response companions | **Outside/deferred.** Recovered response mathematics is preserved for a future concrete scientific use. |
| Physical exposure | **Outside/deferred.** ODQ-104 authorizes only coefficient-squared time. |
| Empirical noise, weight or significance | **Outside/deferred.** These remain SCI-NOI or other separately approved authority. |
| Diagnostics, optional companions and generalized provenance | **Outside/deferred.** No generic optional/conditional-required machinery, persistent operational-reason archive, per-pixel/per-contribution provenance or placeholder product is authorized. |

This table is closed for base v0.1. A new role requires a concrete scientific
use and explicit successor authority; implementation convenience is
insufficient.
