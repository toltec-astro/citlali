# SCI-JINC v0.1/r0.3 Horizontal Authority Coherence Audit

Date: `2026-08-29`

Status: complete; no material coherence finding

Audit mode: post-freeze, read-only, implementation-blind

## Frozen Audit Boundary

The SCI-JINC authority was frozen before this audit at local commit
`a9f43877e01a661db13bd85b2e7f34ea5ac82fb7`, tree
`70c750b1fd003a4f71894e04d3c55391a9ed7d28`, and lightweight tag
`sci-jinc-v0.1-r0.3`.  Its superseding freeze manifest has SHA-256
`ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2`.
The frozen bytes were not modified during or after the audit.

The audit was limited to the following exact scientific authorities and
source-lock controls:

| Audited object | Exact identity or SHA-256 |
| --- | --- |
| Frozen SCI-PTC authority | `SCI-PTC v0.1/r0.5`, content-bound commit `8f0ecccfacbdce0543141c4289ec06c702065f5e`, owner-freeze SHA-256 `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| SCI-PTC shared normative modules | `469219ed1c2927fa9cca9f4ec6d249af3ea49afabe7bb95f0254ba014648d6cc`, `904ffa5292aa0ec0ca89cd6b69567d50dfcd9578d6cf7729d96d01ec78ab63c5`, `af4418116e2fd8d827dd8f149bf16af36a95f2155976644ea6148224faf86f9d`, `dfa6d9b40a506f00ed618142ff2e9debe899e831104c31cfcdfb314fc4ee73d8`, `f2047600cc06c234a78aa3ddf6a575abf2f9592b3e3da810491f6db0150fe21c`, `1541a9916f702949c74703211ef181aa260d39b92a767e88034a24b4d47cde04` |
| PTC-to-JINC boundary | `SCI-PTC_TO_SCI-JINC v0.1/r0.3`, SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e` |
| AST-to-JINC boundary | `SCI-AST_TO_SCI-JINC v0.1/r0.2`, SHA-256 `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5` |
| JINC scientist-readable upstream profile | `SCI-JINC:jinc_map_contribution@1`, SHA-256 `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a` |
| Exact SCI-VAL source-binding snapshot | `v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28`, SHA-256 `0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88` |
| Exact SCI-VAL profile-registry snapshot | same revision, SHA-256 `4b9a1ebecfc847c83b59da772afd9b031ab1830e8febbb12d1a47f70ce5a1110` |
| Frozen SCI-JINC authority | `SCI-JINC v0.1/r0.3`, commit/tag and manifest digest above |

No implementation, configuration, test, validation result, audit of an
implementation candidate, manager record, ambient registry, or web source was
consulted.

## Audit Results

| Required audit axis | Read-only coherence result |
| --- | --- |
| Occurrence and generation identity | Coherent.  The boundaries, profile, SCI-VAL snapshot, and frozen JINC bind one exact atomic occurrence by observation, detector occurrence/UID, stable RTC output sample, PTC product/application generation, segment, stable array, exact plan/WCS where applicable, and immutable parent chain.  Row position, nearest time, tolerance, shape, detector order, and numerical equality are not substitutes. |
| Transformed quantity and unit | Coherent.  Frozen PTC defines the transformed signal `Z` in the admitted CAL physical unit.  The PTC boundary and JINC retain the exact transformed quantity, unit, availability, reference-loss state, and parents.  JINC applies a spatial transform and creates no calibration, Stokes, conditioning, astrometric, noise, or source interpretation. |
| Coefficient ownership and permission | Coherent.  PTC owns registration, selection, lifecycle, payload, normalization, support, QC, and named-consumer permission for an analysis/gridding coefficient family.  JINC requires one exact registered positive family that explicitly permits JINC and has a compatible realized value/QC state.  JINC neither invents nor selects a family.  No such family or payload is supplied by the frozen JINC packet, so the numerical route remains typed unavailable. |
| Same-sample AST association | Coherent.  The AST coordinate is the exact `SCI-AST:rtc_output_grid_coordinates@1` realization for the same processed sample and compatible parent/generation chain.  Missing, duplicate, or ambiguous association is unavailable; there is no row/order/time/tolerance fallback. |
| Profile semantics | Coherent.  `SCI-JINC:jinc_map_contribution@1` means only that an exact upstream occurrence may be considered by JINC.  A pass does not mean pixel contribution, kernel membership, accumulator membership, support, or bundle validity; all JINC-owned gates remain separate and later. |
| Causes and influence | Coherent.  Producer facts, direct causes, transitive influence, and scopes are preserved without erasure.  Only a fact named by an exact restriction affects this profile; reachability, inherited influence, a generic flag, or an optional unavailable fact has no universal veto. |
| Source-lock lifecycle | Coherent.  Boundary, profile, source-register, and profile-register identities and digests are exact-source locked.  Requested, effective, observation-resolved, applied, realized, and evaluation-generation identities remain distinct.  A changed digest or semantic binding requires a new immutable version/generation rather than an adjacent-source substitution. |
| Response/covariance producer-fact carriage | Coherent.  Typed producer facts and limitations are carried neutrally and are advisory for atomic upstream admission.  Missing response/covariance is not zero, identity, independence, or permission.  Base SCI-JINC v0.1 creates no response, covariance, uncertainty, exposure, or availability product and makes no fidelity claim. |
| Failure and typed unavailability | Coherent and fail-closed.  Missing or conflicting identity, ancestry, generation, family selection/permission/payload/QC, AST association, or exact source binding yields the declared unavailable/ineligible state and preserves causes.  No unity, alternate-family, MAP, hidden-parameter, inherited-scale, or approximate-fidelity fallback is authorized.  The TolTEC numerical route remains unavailable without the separately owned coefficient family, authorized array parameter set, and, where numerical support is claimed, exact adequacy profile and matching certificate. |
| Compact replay | Coherent.  The JINC bundle-level replay state `G_a`, together with exact upstream parents, identities, lifecycle, decisions, plan/WCS, parameters, and required summaries, is sufficient to regenerate the authorized estimator inputs and decisions.  It is not a sixth numerical product role, a dense per-contribution product, or a generalized provenance framework. |

## Disposition

Result: **PASS -- no material horizontal-authority incoherence found.**

No successor is opened by this audit, and no frozen byte is changed.  This is
an implementation-blind document-coherence result only.  It makes no claim of
implementation conformity, representation fidelity, response/covariance
fidelity, numerical adequacy, achieved performance, observational validation,
readiness, production, or production authorization, and it does not make the
numerical TolTEC JINC route available.
