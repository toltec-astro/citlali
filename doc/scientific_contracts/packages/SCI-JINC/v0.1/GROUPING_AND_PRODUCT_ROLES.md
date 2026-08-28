# SCI-JINC v0.1 — Observation Grouping And Product Roles

Status: final Stage A base-v0.1 candidate; awaiting scientific-owner approval

Prepared: `2026-08-28`

## Observation-Level Grouping

Base v0.1 is bounded to one observation-level bundle and defines no JINC
coadd. A future coadd requires a separately authorized boundary over complete
JINC bundles; no ordinary SCI-MAP coaddition rule is imported.

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
| WCS/operator/parameter identity | Required | Publish exact array, plan, WCS/frame, analytic identity/version, ordered parameters/source, point-phase, square extent, edge rule, conditioning policy and lifecycle generation. | Any missing exact identity blocks the bundle; the analytic gap currently blocks Stage B. |
| Upstream cause and lineage | Required | Publish exact PTC/AST parents, retention/coefficient/admission evaluations, direct/transitive causes and immutable joins. | Missing/conflicting parent or join blocks the bundle. |
| `jinc_coefficient_squared_time` | Required | Publish `sum I_ip kappa_ip^2/f_s,i` in seconds with coefficient/phase/frequency provenance and its prohibited interpretations. | Missing requested plane or join blocks the bundle; it never substitutes for formal support. |
| Fixed-state response | Conditional required | When requested, publish the processing-filtered source-template response transformed by the exact fixed JINC operator once. Otherwise publish typed `not_requested`/unavailable state, not a default kernel. | Failure blocks only a plan that requires this role; base signal may coexist with an explicitly optional unavailable response. |
| Covariance / conditional formal weight | Conditional required | Publish exact `A C_PTC A^T` domain when available and any permitted diagonal view with assumptions/omissions. If the coefficient family or covariance evidence is insufficient, publish typed unavailable; never substitute `Q`, time or hits. | Failure blocks claims/plans that require uncertainty, but not an explicitly uncertainty-optional signal role. |
| Response/covariance limitation record | Required | Publish the family, domain, assumptions, edge membership and every omitted or unavailable calibration, correlation, selection, nuisance, response and parameter term. | Missing limitation identity blocks any response/covariance publication. |
| Requested/effective/resolved/realized provenance | Required | One compact atomic record with exact product identities, destinations, joins and failure state; no per-sample/pixel payload required. | Required publication failure suppresses realized success. |
| Diagnostics | Optional | Contributor counts, cancellation summaries and numerical diagnostics may be published only under named identities; they never control validity by existence alone. | Optional absence does not block unless a plan promotes an exact diagnostic to required. |
| Empirical noise/significance | Outside SCI-JINC | SCI-NOI-owned future companion only. SCI-JINC publishes no substitute. | No effect on base JINC signal validity; downstream claim remains unavailable. |

An optional unavailable companion may coexist with valid signal only under the
exact role above and with its cause. A requested-required companion, atomic
join, or destination failure prevents realized success. Required roles are
published as one coherent bundle generation; partial output is not success.
