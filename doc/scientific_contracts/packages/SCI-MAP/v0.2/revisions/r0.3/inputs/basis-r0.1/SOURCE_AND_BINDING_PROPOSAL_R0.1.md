# SCI-MAP v0.2/r0.1 source and binding proposal

Status: concrete candidate binding proposal; **not activated, integrated, or
frozen**. Known approved identities and hashes are exact. New local source
hashes are bound in `SOURCE_MANIFEST_R0.1.md`; identities that cannot exist
until integration remain explicitly pending.

## Proposed successor source binding

| Field | Proposed exact value |
| --- | --- |
| Contract/source generation | `SCI-MAP v0.2/r0.1-candidate-2026-09-07` |
| Shared wrapper and order | `src/SCI-MAP-v0.2_SHARED_AUTHORITY_r0.1.tex`, then notation, definitions, equations, assumptions, requirements, edge cases |
| Local byte identities | Exact per-file and aggregate SHA-256 values in `SOURCE_MANIFEST_R0.1.md` |
| Canonical Git commit/tree | **PENDING -- not supplied to the implementation-blind author** |
| Immutable final source-manifest identity and external digest | **PENDING -- allocate and bind after the imported candidate is exact** |
| Scientific status | Scope-approved candidate; complete owner disposition, consistency review, and freeze pending |

The local hashes are actual draft bytes, not fabricated canonical identities.
Any later scientific edit requires recomputation and a new reviewed candidate.

## Exact imported producer and boundary bindings

| Role | Exact binding | SHA-256 / source constraint |
| --- | --- | --- |
| Frozen predecessor | SCI-MAP v0.1/r0.7.1 freeze | `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1` |
| Predecessor PTC boundary | `SCI-PTC_TO_SCI-MAP v0.1/r0.1` | `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` |
| Uniform boundary successor | `SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4` | `58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8`; imports predecessor and fills only the uniform slot |
| Coefficient Registry | `SCI-PTC:analysis_gridding_coefficients@draft-0.1`, admission `PTC-UNIFORM-ADMISSION r0.4/2026-09-06` | Registry record `433309b21d17751ce3972d196f9ba73a5f910f36e88052046928a0163fb7ca76`; admission manifest `500494b7814d3f13306581a6ca0ddfb232928046a6a1c15b9ecc19bb63989f11` |
| Uniform family source | `SCI-PTC:uniform_constant@draft-0.1`; `SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1/r0.4` | Common source `8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685`; freeze manifest `c6d23f7f20f35089eba9b5f6dd6e65aaf05561ab3da2d0f7d18e5b8256a2f016` |
| Uniform handoff profile | `SCI-PTC:uniform_coefficient_handoff@draft-0.1` | Complete record in exact uniform VAL Registry successor; MAP permission granted under UFC-OD-008 |
| Uniform VAL Registry | `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-ptc-uniform-r0.4-2026-09-06` | File SHA-256 `a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f` |
| Uniform VAL source register | `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-ptc-uniform-r0.4-2026-09-06` | `fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e` |
| Uniform consumer composition | `PTC-UNIFORM-CONSUMER-COMPOSITION r0.4/2026-09-06` | Allows exact MAP `map_upstream_admission@2` composition without rewriting its predicates or old evaluations |
| AST original-footprint boundary | `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1` | `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4`; unchanged |

## Proposed MAP profile/source composition

| MAP policy identity | Successor treatment | Required immutable binding action |
| --- | --- | --- |
| `SCI-MAP:map_upstream_admission@2` | Reuse exact predicates and identity; compose the admitted uniform handoff only through the exact uniform consumer composition. | Create a new v0.2-compatible source-binding generation naming the exact local/final MAP source digest and the existing uniform Registry/register/boundary. Do not rebind any earlier evaluation. |
| `SCI-MAP:one_hot_containing_pixel@1` | Reuse exact one-hot half-open rule. | Bind the unchanged policy to the new MAP v0.2 source generation; no alias or changed projection. |
| `SCI-MAP:uniform_observation_coadd_coefficient@1` | Reuse exact equal-observation coefficient. | Bind unchanged arithmetic to the new source generation. |
| `SCI-MAP:observation_coadd_admission@1` | Reuse the complete-bundle and compatibility predicates. The v0.2 source explicitly records logically equivalent in-memory/persisted forms and preconstruction grid identity. | A future VAL owner/reviewer must decide the exact immutable Registry/source-binding successor identity after byte binding. Existing record and evaluations stay unchanged. If VAL determines that the clarified route/storage fields change the profile record rather than only its compatible source generation, allocate a versioned profile successor without changing the approved science. |
| Pointing/OOF consumer profiles | No exact mode profile was supplied. | **PENDING from the respective mode owner.** Operator permission does not make either route available or authorize a default. |

### Pending exact VAL successor identity

The proposal requires one immutable VAL Registry/source-binding generation
that binds the final SCI-MAP v0.2 shared-source digest, exact uniform boundary
and consumer composition, and the unchanged MAP policy semantics above. Its
literal Registry/register identity and digest are **PENDING** because no such
artifact is present in the approved packet. This record deliberately does not
invent either value. The required content is reviewable here; activation and
canonical placement remain later owner/manager actions.

## Binding invariants

1. Explicit family selection is distinct from Registry presence and mode
   policy; no selection and no authorized mode default makes the route
   unavailable.
2. Complete PTC publication, exact `(q,g,i,consumer)` association, payload
   integrity/membership, producer QC, MAP permission, the four-axis handoff,
   independent MAP admission, and MAP finite-positive classification all
   remain separate.
3. Old source/profile evaluations are immutable. Numerical equality, common
   names, equal shapes, or the literal value one cannot establish compatibility.
4. Existing uniform, AST, PTC, JINC, NOI, VAL, and frozen MAP records remain
   byte-identical. No additional coefficient family, NOI permission, default,
   or realized application route is inferred.
5. A changed successor source byte, predicate, owner, profile, boundary,
   exception, lifecycle, or failure scope requires a new exact identity and
   review before use.
