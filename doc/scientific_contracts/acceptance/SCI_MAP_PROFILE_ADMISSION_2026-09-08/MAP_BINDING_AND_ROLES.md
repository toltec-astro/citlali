# Proposed final MAP source binding and role Registry

Proposed identities: `SCI-MAP_SOURCE_BINDING v0.2/r0.4-2026-09-08` and
`SCI-MAP_PRODUCT_ROLE_REGISTRY v0.2/r0.4-2026-09-08`. Both are pending owner allocation and admission.
Owner: Grant Wilson for SCI-MAP. Current activation/evaluability is unavailable.

## Exact source and complete role incorporation

Shared authority: `SCI-MAP-v0.2-SHARED-AUTHORITY/r0.4`, aggregate SHA-256
`3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31`. Source/length/order inventory is in the exact
[SOURCE_MANIFEST.json](../../packages/SCI-MAP/v0.2/revisions/r0.4/SOURCE_MANIFEST.json); SHA-256 `5668162f42a83d653a8a8cb8219d04706150cf48374576e7000cb250baad31c4`.
Acceptance/freeze status is supplied by [FREEZE_BINDING_R0.4.json](../../packages/SCI-MAP/v0.2/FREEZE_BINDING_R0.4.json); SHA-256 `37e4af6d556efb637cdef27337bc78cd13380aa053978d15012139edaf323ca7`.

The complete six-role policy is incorporated by exact content from
[SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.4_CANDIDATE.md](../../packages/SCI-MAP/v0.2/revisions/r0.4/profiles/SCI-MAP_PRODUCT_ROLE_REGISTRY_v0.2_r0.4_CANDIDATE.md); SHA-256 `d42d93a9f46f4f22c62d010172a20bad4f43e400e8271cbdffa655d689bd9086`. This includes every parent, required-content, missing-companion,
membership, request, five-stage lifecycle, attempt/product-realization and
no-support clause. The source record remains unchanged and historical.
The exact six role keys are:

- `SCI-MAP:base_observation_map@1`.
- `SCI-MAP:response_bearing_observation_map@1`.
- `SCI-MAP:covariance_qualified_observation_map@1`.
- `SCI-MAP:base_coadd@1`.
- `SCI-MAP:response_bearing_coadd@1`.
- `SCI-MAP:covariance_qualified_coadd@1`.

For this proposed successor only, [IDENTITY_ALLOCATION.json](IDENTITY_ALLOCATION.json)
maps the five administrative candidate generation references to explicit new
generation identities. It changes no profile/role key, scientific source,
predicate, exception, owner, object, membership, response/covariance rule,
lifecycle, failure scope or consumer action. This is an explicit new association;
it is not a rewrite or alias of an earlier record or evaluation. Inherited
candidate/pending status prose describes the original snapshot. The current
proposal is also pending; any later approval and registration require exact
separate status records. No string substitution in frozen sources is authorized.

The unchanged method-policy associations are
`SCI-MAP:one_hot_containing_pixel@1` and
`SCI-MAP:uniform_observation_coadd_coefficient@1`, bound to the same r0.4 core.
They provide the already approved projection and coadd coefficient rules;
they are not additional atomic/aggregate VAL eligibility propositions.

All seven core modules, three view entries, owner-register source, eight
frozen profile/binding records and imported producer authorities are bound in
[BINDING_MANIFEST.json](BINDING_MANIFEST.json). Their raw hashes are checked;
the shared core remains the sole normative scientific authority. This wrapper
adds only explicit proposed generation associations and status conditions.
