# SCI-NOI v0.1 — SCI-VAL Registry Binding

Date: `2026-08-30`

Status: Stage B Registry/source prerequisite satisfied; approved Stage A
author-input bytes unchanged; final exact-packet owner approval remains open;
no Stage B normative content created by this binding

## Authority

The scientific owner approved every bounded SCI-NOI Stage A decision through
ODQ-111 at commit `2f7076e0c7a51320413a86cc6be74c2d3e8f1537`.
`SCI-NOI_OWNER_DECISIONS v0.1/r0.18` has SHA-256
`272ac939b8a7109a123073b1a39fcdd7ac4129c603683ee81257b94ab2f55a0b`;
the ODQ-111 approval record has SHA-256
`4e377dba46f8aead91ce14291ff6ae41de46476ff6cf3eab732d3aa29b503e67`.
The approved author manifest remains
`SCI-NOI_AUTHOR_PACKET_MANIFEST v0.1/r0.18`, SHA-256
`b6f8e7252e7f61f4506899cb3e8e26cf939887bb48464852713f8ce81ac77ca0`.

Frozen SCI-VAL v0.1/r0.3 preserves continuing immutable source and profile
registries. `SCI-VAL-REQ-025`, `SCI-VAL-REQ-043`, and `SCI-VAL-REQ-044`
authorize the closed response/uncertainty roles, separation between VAL
mechanics and use-owner policy, and complete immutable profile bindings. This
record applies those already approved mechanics; it creates no NOI policy.

## Immutable Successor Objects

The existing MAP- and JINC-bound registry files remain byte-identical so their
replay authorities are undisturbed. SCI-NOI is bound through new immutable
successor objects:

| Control | Exact identity | SHA-256 |
| --- | --- | --- |
| Source-binding successor | `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-2026-08-30` | `04eca2da9ce76afacf18ae90dc2dbcb702fedbf55e03acb28e14e7dbc459a7c3` |
| Profile-registry successor | `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-2026-08-30` | `5994f4dff49dff3a9c9da6fbb494671b14a2f926f325f1c7c4a9603a6c2a38c1` |

The source-binding successor preserves every earlier row and adds one exact
SCI-NOI row binding the r0.18 manifest, owner-approved policy bytes, sanitized
decisions, ODQ-111 approval, three NOI boundary extracts, and frozen upstream
authorities.

The profile successor preserves every earlier record and registers:

- `SCI-NOI:generation_input_admission@1`;
- `SCI-NOI:uncertainty_member_admission@1`;
- `SCI-NOI:uncertainty_ensemble_admission@1`; and
- `SCI-NOI:standardization_admission@1`.

Each registration binds one exact object/domain and one exact consumer action,
producer-fact ownership, NOI policy/action ownership, four separately named
decision fields, missing/conflict behavior, lifecycle, propagation limits, and
supersession. VAL binds/evaluates but authors neither fact nor policy. No
profile automatically realizes the next GEN, UNC or STD operation.

## Dispatch Consequence

The versioned SCI-VAL Registry/source prerequisite in the SCI-NOI Scope Brief
is satisfied. The author-input manifest and all 17 admitted objects remain
byte-identical to r0.18; this process-only record and both Registry successors
remain outside the normative author packet. Stage B remains prohibited until
the scientific owner approves the exact final packet bytes/hashes and
explicitly launches it.

Registration makes the four supplied profiles evaluable only when their exact
request, source, applicability and producer facts exist. It does not select or
authorize missing finite-design mechanics, a PTC MAP coefficient, an admitted
numerical `coverage_cut`, a MAP/JINC numerical parent, or a transformed,
Wiener, or FRUIT route. It establishes no implementation conformity,
validation, calibration, physical-noise meaning, covariance completeness,
Gaussian significance, achieved performance, readiness or production claim.
