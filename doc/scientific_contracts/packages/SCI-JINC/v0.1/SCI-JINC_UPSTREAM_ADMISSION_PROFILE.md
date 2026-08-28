# SCI-JINC v0.1 — Upstream Admission Profile Candidate

Registry key: `SCI-JINC:upstream_admission@1`

Status: final Stage A profile draft; awaiting scientific-owner approval and a
versioned SCI-VAL registry binding; not currently evaluable

Prepared: `2026-08-28`

## Ownership And Registry Rule

SCI-JINC owns this named-use policy. SCI-VAL Registry binds the immutable
profile, and VAL Core evaluates it; neither authors JINC policy. Frozen
SCI-VAL v0.1/r0.3 is not edited by this Stage A repair. A versioned registry
successor must bind the approved profile bytes before the profile is usable.
Until then, profile applicability and decision are unavailable and no
numerical JINC route exists.

| Field | Proposed binding |
| --- | --- |
| Registry key | `SCI-JINC:upstream_admission@1` |
| Named use and action | `upstream_admission`; decide whether one exact realized PTC occurrence may enter the JINC route-candidate population before JINC square placement and pixel-local numerical gates. |
| Scientific owner | Grant Wilson, SCI-JINC scientific-policy owner. |
| Authoritative source | Owner-approved successor bytes of this profile; exact `SCI-PTC_TO_SCI-JINC v0.1/r0.2`; controlled ODQ-101 source/cover with source SHA-256 `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`; exact `SCI-AST_TO_SCI-JINC v0.1/r0.1`; frozen SCI-PTC v0.1/r0.5 freeze record SHA-256 `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`; frozen SCI-AST v0.1/r0.3 source manifest SHA-256 `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`; and the future exact SCI-VAL registry successor. |
| Applicability and object | Requested observation-level SCI-JINC route; one exact occurrence binding observation, detector occurrence/UID, stable RTC output sample `n`, PTC product/application generation, segment, stable TolTEC array, exact JINC plan, target WCS and complete parent chain. |
| Required permissions | Exact realized PTC product exists; `SCI-PTC:output_retention@1` is requested, applicable, eligible and realized; transformed signal and source identity are available; requested/effective/observation-resolved/realized identities select one exact registered PTC positive analysis/gridding family by user request or authorized versioned mode default; that family explicitly permits `SCI-JINC`; its compatible value is typed available; its separate coefficient/QC record permits JINC use; the exact same-`n` AST RTC-grid continuous coordinate is structurally bound and AST-valid; all required source, generation and lifecycle bindings are compatible. |
| Decisive exclusions | PTC-disabled/no-product; direct CAL fallback or inferred no-op PTC; output-retention ineligible; direct synthesized or replaced representative source for ordinary JINC signal support; missing selection with no authorized default; unregistered family; missing `SCI-JINC` permission; unavailable or mismatched coefficient payload/QC; incompatible signal/coordinate parents; incompatible generations; absent stable array identity; or boundary/profile/source-version mismatch. |
| Direct and inherited influence | Direct causes and complete transitive influence are preserved. Only a fact named by an exact restriction affects this profile. Reachability, inherited influence, a generic flag, or an optional unavailable fact has no universal veto. |
| Response role | `advisory` for base numerical JINC signal admission. Exact response family/state and causes are carried. A requested response companion applies its own product-role requirement. |
| Uncertainty role | `advisory` for base numerical JINC signal admission. Exact covariance/uncertainty state, assumptions, omissions and causes are carried; admission neither fabricates nor upgrades them. |
| Exceptions | None for exact occurrence/parent/generation binding, realized PTC route, output-retention permission, direct synthesized/replaced exclusion, registry/family identity, JINC consumer permission, coefficient identity/QC permission, same-`n` AST binding, stable array identity, or exact source/profile identity. |
| Four decision fields | Request: `requested`/`not_requested`; applicability: `applicable`/`inapplicable`/`applicability_unknown`; eligibility: `eligible`/`ineligible`/`decision_unavailable`; realization: `realized`/`incomplete`/`failed`/`not_produced`. Only requested + applicable + eligible + realized projects to upstream-admission pass. |
| Missing/conflicting behavior | Missing or conflicting applicability, identity, parent, generation, selection/default authority, registration, named-consumer permission, coefficient family/value/QC, coordinate, boundary, source or registry binding yields `applicability_unknown` and `decision_unavailable` where evaluable. A decisive false restriction yields `ineligible`; all restrictions true yields `eligible`. Causes and scopes remain exact. No alternate-family or unity fallback is permitted. |
| Lifecycle and consumer action | The evaluation binds requested, effective, observation-resolved, applied and realized identities plus exact source/profile versions. Pass creates a JINC route candidate only. JINC still requires an exact scientifically authorized array-associated parameter-set identity and evaluates payload finiteness, positive `omega_i`, point-phase `kappa_ip`, square placement, edge policy, conditioning, required companions and final bundle validity. Missing parameter-set authority makes the numerical route unavailable without a hidden default; it does not retroactively change upstream admission. |
| Aggregation and propagation | `atomic_only`. No pixel, detector, observation, exposure or coadd aggregate and no reverse propagation are implied. Producer facts and earlier decisions are immutable. |
| Supersession | Any changed source digest, occurrence domain, restriction, exception, response/uncertainty role, lifecycle, direct/inherited influence rule or consumer action requires a new immutable profile version and evaluation generation. |

## Separately Typed Gates

The profile does not collapse the following propositions:

1. PTC signal availability;
2. PTC output-retention disposition;
3. coefficient availability;
4. PTC coefficient/QC disposition;
5. JINC upstream admission;
6. AST coordinate validity;
7. finite signal;
8. finite positive upstream coefficient;
9. authorized array-associated JINC parameter-set availability;
10. finite signed-kernel placement;
11. cancellation and formal JINC support;
12. required-companion availability; and
13. final JINC product validity.

An optional unavailable companion may coexist with an available base signal
only when the exact JINC product-role table permits it. A requested-required
companion or join failure prevents realized success.

## Registration Blocker

This candidate is not yet present in the frozen SCI-VAL registry. Owner
approval must authorize its exact bytes and a separately controlled SCI-VAL
registry successor. Merely reserving the name does not create an evaluable
policy.
