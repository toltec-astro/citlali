# WP-5 PTC Profile Registry Review r0.1

Date: `2026-08-24`

Status: exact integration packet prepared from approved
`WP5-OWNER-D003--D011`; F-004 and F-020 remain open pending owner review and
clean-room re-audit

Scope: non-MAP PTC named-use admission only

## Exact packet

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| `PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md` | PTC-owned non-evaluable common semantics fragment | `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |
| `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | Continuing VAL registry containing the five complete profiles and two unsupported dispositions | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| `packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | Unchanged exact adjacent-source authority | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |

The decision source is
`WP5_VAL_SCIENTIFIC_OWNER_DECISION_PACKET.md` at commit `44662a36b`, file
SHA-256 `9bc101e8447173836380e00ea58185fc2e67cbcbac5077ff1578ca5dc27139fd`.
Frozen PTC authority is v0.1/r0.5, freeze-record SHA-256
`8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`.

## Materialized dispositions

Complete registered propositions:

1. `SCI-PTC:basis_fit_admission@1`;
2. `SCI-PTC:loading_fit_admission@1`;
3. `SCI-PTC:operator_application@1`;
4. `SCI-PTC:output_retention@1`; and
5. `SCI-PTC:response_companion@1`.

Explicitly unsupported and unbound:

1. `SCI-PTC:coefficient_qc_population@1`; and
2. `SCI-PTC:empirical_or_simulation_population@1`.

MAP and coadd profiles remain deferred and unbound.

## Review assertions

1. The common fragment grants no permission and contains no use-specific
   restriction.
2. Each registered profile is independently complete and transfers no
   permission to another use.
3. Only facts explicitly declared scientifically relevant by the named use
   affect its decision. Unrelated metadata, including its missing, available,
   unknown, or conflicting state, is admission-neutral.
4. CAL `engineering-only` remains preserved. It is not a universal veto on
   PTC mathematics and is not upgraded by a PTC decision.
5. Among the five PTC profiles, direct-origin exclusion is applied only to
   basis fit, loading fit, and ordinary output retention. It is not inferred
   as a veto on mathematical operator application.
6. The application profile makes the exact configured-rank group-time guard
   fail closed with no silent numerical substitute.
7. The response profile admits only the already-frozen tracked-kernel
   propagation. It creates no new response computation or dense-serialization
   duty.
8. The unsupported coefficient/QC and empirical/simulation identities are not
   represented by vacuous profiles.
9. The source-binding register and canonical independent-exposure proposition
   are unchanged.
10. No MAP, coadd, generic usable exposure, runtime policy object, sidecar, or
    duplicated provenance payload is introduced.

## Next gate

Scientific-owner approval of this exact packet may be recorded as
`WP5-OWNER-D012`. Closure still requires the WP-7 clean-room re-audit; this
packet alone does not claim implementation conformity, validation,
performance, production readiness, or MAP availability.
