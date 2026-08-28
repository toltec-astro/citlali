# SCI-JINC v0.1 — Observation Grouping And Product Roles

Status: final Stage A base-v0.1 candidate; awaiting scientific-owner approval

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
Base v0.1 produces independent observation bundles for each requested present
array. Cross-array, frequency-combined, or network-combined JINC products are
unavailable unless separately authorized.

Observation is the scientific grouping boundary, not a streaming, processing-
chunk, process, container or memory boundary. Samples or chunks from the same
observation may accumulate incrementally into the one observation bundle only
when the exact observation, stable array, JINC plan and realization, target
WCS, admission/parameter/coefficient state and lifecycle generation match.
Chunk identity neither creates a JINC product nor licenses cross-observation
combination.

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

## Product-Role Table

“Required” means absence or publication/join failure prevents realized bundle
success. “Conditional required” means the role becomes required when the
exact JINC plan requests it or a named consumer contract requires it.

| Role | Requirement | Exact meaning and publication rule | Failure scope |
| --- | --- | --- | --- |
| Normalized signal `m_p` | Required | Publish `N_p/C_p` in `U` only on formal JINC support, with typed unavailable state elsewhere. | Affected array observation bundle cannot realize success without the signal role. |
| Numerator `N_p` | Required | Publish the distinct signed signal numerator or an exact lossless role from which its value and unit are available; never alias it to signal or normalization. | Missing/join failure blocks the bundle. |
| Normalization `C_p` | Required | Publish the signed normalization, including finite negative values on accepted pixels and typed cancellation state; never require positivity. | Missing/join failure blocks the bundle. |
| Quadratic accumulator `Q_p` | Required | Publish the nonnegative quadratic accumulator separately from `C_p`, formal weight and time accounting. | Missing/join failure blocks the bundle. |
| Formal-support state | Required | Publish the authoritative JINC formal-support proposition and exact cause, distinct from upstream admission, AST validity, time accounting and empirical policy. | Missing/join failure blocks the bundle. |
| WCS/operator/parameter identity | Required | Publish exact array, plan, WCS/frame, analytic identity/version, ordered array-associated parameter-set identity/source, point-phase, square extent, edge rule, conditioning policy and lifecycle generation. | Any missing scientifically authorized numerical parameter set makes the numerical route unavailable; no hidden default may complete the bundle. |
| Upstream cause and lineage | Required | Publish exact PTC/AST parents, retention/coefficient/admission evaluations, direct/transitive causes and immutable joins. | Missing/conflicting parent or join blocks the bundle. |
| `jinc_coefficient_squared_time` | Required; sole base-v0.1 time-support product | Publish `sum I_ip kappa_ip^2/f_s,i` in seconds with coefficient/phase/frequency provenance and its prohibited interpretations. | Missing requested plane or join blocks the bundle; it never substitutes for formal support or physical exposure. |
| Coupled-contribution identity | Required | Signal, normalization, quadratic/time, response, covariance, and related accumulators use the same admitted sample-pixel pair, AST coordinate realization, and signed `kappa_ip` identity while applying only their own contract-defined algebra. | Inconsistent admission, coordinate, or coefficient realization blocks the affected required bundle; no new per-contribution provenance payload is implied. |
| Fixed-state response | Conditional required | When requested, publish the processing-filtered source-template response transformed by the exact fixed JINC operator once. Otherwise publish typed `not_requested`/unavailable state, not a default kernel. | Failure blocks only a plan that requires this role; base signal may coexist with an explicitly optional unavailable response. |
| Covariance / conditional formal weight | Conditional required | Publish exact `A C_PTC A^T` domain when available and any permitted diagonal view with assumptions/omissions. If the coefficient family or covariance evidence is insufficient, publish typed unavailable; never substitute `Q`, time or hits. | Failure blocks claims/plans that require uncertainty, but not an explicitly uncertainty-optional signal role. |
| Response/covariance limitation record | Required | Publish the family, domain, assumptions, edge membership and every omitted or unavailable calibration, correlation, selection, nuisance, response and parameter term. | Missing limitation identity blocks any response/covariance publication. |
| Requested/effective/resolved/realized provenance | Required | One compact atomic record with exact product identities, destinations, joins and failure state; no per-sample/pixel payload required. | Required publication failure suppresses realized success. |
| Diagnostics | Optional | Contributor counts, cancellation summaries and numerical diagnostics may be published only under named identities; they never control validity by existence alone. | Optional absence does not block unless a plan promotes an exact diagnostic to required. |
| Empirical noise/significance | Outside SCI-JINC | SCI-NOI-owned future companion only. SCI-JINC publishes no substitute. | No effect on base JINC signal validity; downstream claim remains unavailable. |
| Physical exposure | Deferred outside base v0.1 | No product is defined until an identified scientific use separately authorizes exact original-occurrence lineage, membership, units, semantics, availability, provenance and consumer meaning. | Never reinterpret `jinc_coefficient_squared_time` or distribute one physical integration through JINC lobes. |

An optional unavailable companion may coexist with valid signal only under the
exact role above and with its cause. A requested-required companion, atomic
join, or destination failure prevents realized success. Required roles are
published as one coherent bundle generation; partial output is not success.
