# Proposed final MAP/VAL Profile Registry generation

Proposed identity: `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.4-2026-09-08`.
Paired source-register identity: `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.4-2026-09-08`.
Status: proposed for owner approval; unregistered, not formally source-bound,
not activated, unevaluable. Policy owner: Grant Wilson for SCI-MAP.
SCI-VAL remains evaluator/registry; it does not author policy or create products.

## Complete immutable inheritance

Incorporate the entire [PROFILE_REGISTRY_PTC_UNIFORM_R0_4_2026-09-06.md](../../packages/SCI-VAL/v0.1/PROFILE_REGISTRY_PTC_UNIFORM_R0_4_2026-09-06.md); SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f`. It incorporates the entire
[PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md](../../packages/SCI-VAL/v0.1/PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md); SHA-256 `5994f4dff49dff3a9c9da6fbb494671b14a2f926f325f1c7c4a9603a6c2a38c1`. This preserves all registration and aggregate rules, fifteen
complete named-use profiles, unsupported names, exceptions, source references,
missing/conflict behavior, ownership, limitations and historical decisions.
The inheritance is a raw-byte binding, not a summary-only replacement.
All eight historical Registry/source-register files and all fifteen inherited
keys are enumerated and hashed in [BINDING_MANIFEST.json](BINDING_MANIFEST.json)
and [IDENTITY_ALLOCATION.json](IDENTITY_ALLOCATION.json).

Only the explicit MAP associations and coadd successor below are proposed.
Every other profile, including the uniform handoff, JINC and NOI, retains its
original source and permissions. Earlier bindings and evaluations remain exact.
Historical retention does not extend predecessor production authorization.

## Complete occurrence-profile association

Profile key: `SCI-MAP:map_upstream_admission@2`.
Proposed binding: `SCI-MAP:map_upstream_admission@2 / SCI-MAP-v0.2-r0.4-2026-09-08`.

The complete inherited `@2` table in the content-bound NOI Stage A Registry
remains the predicate basis, with its September 6 uniform association preserved.
This new binding incorporates the entire exact [SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.4_CANDIDATE.md](../../packages/SCI-MAP/v0.2/revisions/r0.4/profiles/SCI-MAP_OCCURRENCE_ADMISSION_PROFILE_v0.2_r0.4_CANDIDATE.md); SHA-256 `5e3786f41286305677fb3cac7b791722f5935869d0a7f8229610f4e65cd89265` and the r0.4
core/role composition in [MAP_BINDING_AND_ROLES.md](MAP_BINDING_AND_ROLES.md),
SHA-256 `3229f3ef7aa62c0803dba2f6f8210083d664182988e19050d78c7dbeb5a38882`. The source-generation change requires explicit owner
admission; a bare `@2` key never selects it. It does not infer a general waiver
of the inherited versioning rule: the precise compatible association and
unchanged-predicate declaration below are the proposed owner disposition.

| Required Registry component | Complete authority for this proposed association |
| --- | --- |
| Key/version and named use | Exact `@2` plus the new binding and paired Registry/register; one occurrence's MAP route-candidate admission. |
| Actual owner | Grant Wilson, SCI-MAP policy owner; unchanged from both exact source records. |
| Authoritative source | Entire inherited `@2` table, exact r0.4 occurrence record and shared core; source register binds the exact tuple. |
| Domain/object | Full occurrence identity and ordinary positive-rank PTC route from Object and exact axes / Exact predicate set and order. |
| Restrictions/missing facts | All inherited required permissions, decisive exclusions, classification/influence rules, missing/conflicting behavior, and ordered A–E gates remain required. |
| Exceptions | The inherited nonexceptionable occurrence/parent/generation, PTC retention, direct-origin, coefficient/QC and same-n AST/binding invariants remain exact; none added. |
| Response/uncertainty roles | Advisory at occurrence-level base signal admission; the selected stronger role separately requires its exact companion. All source statuses and causes remain. |
| Aggregation/propagation | Inherited atomic_only; no pixel/observation/coadd aggregate, reverse propagation or producer rewrite. |
| Lifecycle/action | Independent four VAL axes, exact five-stage MAP references, requested/applicable/eligible/realized pass only; eligible decision creates a route candidate only. |
| Supersession/incompatibility | Explicit new Registry/register/binding tuple; earlier tuples immutable. No further source or policy change is licensed by this compatibility declaration. |

## Complete coadd successor

Proposed profile key: `SCI-MAP:observation_coadd_admission@2`.
Incorporate the entire exact [SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.4_CANDIDATE.md](../../packages/SCI-MAP/v0.2/revisions/r0.4/profiles/SCI-MAP_COADD_AGGREGATE_PROFILE_v0.2_r0.4_CANDIDATE.md); SHA-256 `290292888accabd4aab9afce1c2ceb471f3b3f275072724ffbe5caed4e3d1130`, exact six-role record, and r0.4
core. Historical `@1` is retained, never aliased or rebound. This is the
already reviewed r0.4 successor policy presented for separate profile approval.

| Required Registry component | Complete authority for this proposed successor |
| --- | --- |
| Key/version and named use | `@2`; atomically admit one complete observation bundle for its selected coadd role before mutation. |
| Actual owner | Grant Wilson for SCI-MAP; VAL evaluates, MAP performs the scientific action. |
| Authoritative source | Entire r0.4 coadd profile and role record, exact shared core and this paired binding generation. |
| Domain/object | Object, population, and axes plus Request and applicability: complete base/unfiltered bundle, exact support rows, selected coadd role, centered-integer plan and ordered population. |
| Restrictions/missing facts | Required base compatibility, Role-qualified companion rules, Aggregation/output/failure are incorporated completely; all causes remain and no unsupported substitute is supplied. |
| Exceptions | Retain the inherited nonexceptionable quantity/beam, grid/frame, centered-integer, required-companion, generation, atomicity and selected-role restrictions; r0.4 introduces none. |
| Response/uncertainty roles | Base-role numerical response/covariance availability remains advisory; required companions are required_permission for the selected stronger role. The complete r0.4 role-local rules govern. |
| Aggregation/propagation | Exact centered-integer, equal-observation aggregate and declared forward companion propagation; no reverse propagation, crop, pad, interpolation, reprojection, GLS or mosaic. |
| Lifecycle/action | Independent VAL decision realization; full five-stage MAP lifecycle and distinct attempt/product outcomes; exact membership and no-support outcome remain required. |
| Supersession/incompatibility | New `@2` and exact paired generation; no `@1` alias, retroactive evaluation or unstated source replacement. Any further policy/source change needs an immutable successor. |

## Method and product-role associations

The two unchanged method-policy keys and six independently requestable roles
are bound by [MAP_BINDING_AND_ROLES.md](MAP_BINDING_AND_ROLES.md). Method keys
do not create new VAL named-use propositions. Observation roles use the exact
occurrence binding; coadd roles use exact coadd `@2`. Their complete role-local
requirements never mutate an already-realized base parent or select a hidden
subset. No profile declaration realizes an input, decision, map or companion.

## Finality and activation gates

The paired register names this file's raw hash, and the final manifest binds
both; this avoids a mutual file-hash cycle. New decision identity must explicitly
name both exact generations, their hashes, exact profile/binding, object and
role. Missing/contradictory identities cannot be repaired by ambient-current
lookup. The finite administrative mapping is in IDENTITY_ALLOCATION.json;
candidate delivery-time status is never silently upgraded.

Owner approval, canonical registration/formal binding, activation/evaluability,
and object-specific decision realization remain separate. This proposal performs
none. Even after a later activation, a new decision requires the exact producer
facts and resulting decision-artifact verification before MAP consumes it.
