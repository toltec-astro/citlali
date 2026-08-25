# WP-7 VAL Successor-Binding Compatibility Proposal

Status: **approved binding scientific-owner authority**

Prepared: `2026-08-25`

Scientific owner: Grant Wilson

If the scientific owner replaces the status above with
`approved binding scientific-owner authority`, every proposed rule below
becomes binding for the exact sources named in Section 1. No separate
compatibility inference is authorized.

## 1. Proposed exact binding set

| Source | SHA-256 |
| --- | --- |
| Continuing SCI-VAL source-binding register | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| Continuing SCI-VAL profile registry containing the five immutable PTC profiles | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| WP-7 approved scientific-authority addendum | `7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577` |
| Approved passband-identity compatibility authority | `ef92c8d2d91fcf9c8f3707275d8de606b469de7bb0d17f73a7c407d38f71f6bb` |
| Approved CAL quality-token compatibility authority | `93a285433879f6df8c2f4bcd1cbe7f1a1c9639f8506edbfae7f15fb24051b222` |
| SCI-RTC explanatory source-correction record | `e79efa61cff5cb8c733a8718eb4ff91ecfd3a3826063e506cb2e628929cd5a3e` |
| SCI-RTC corrected source manifest | `a5c06bd46cd8514e67ea77a7a728e3decb8c415cf486c4ec121927212bf22994` |

The frozen SCI-VAL register remains the base adjacent-source binding. The
approved WP-7 authorities above would be an exact additive compatibility
generation for successor replay; they would not silently replace the base
register with an unspecified “current” source.

## 2. Proposed compatibility determination

The exact WP-7 successor authorities are compatibility-preserving for all five
registered SCI-PTC profiles:

```text
SCI-PTC:basis_fit_admission@1
SCI-PTC:loading_fit_admission@1
SCI-PTC:operator_application@1
SCI-PTC:output_retention@1
SCI-PTC:response_companion@1
```

The compatibility basis is:

1. The native paired-readout promotion changes authority status and source
   closure only. It does not change the ALIGN/RTC facts or restrictions used
   by any profile.
2. The recovered CAL numerical objects and exact WVR interpolation and
   unavailable-opacity rules complete how upstream CAL facts are produced.
   They do not change any profile's named action, restriction, exception,
   response/uncertainty role, aggregation rule, or consequence.
3. The exact observation classifier resolves an upstream CAL producer fact.
   The approved quality-token crosswalk preserves the registered
   `engineering-only` consequence exactly as canonical `engineering_only`:
   it does not itself prohibit the named PTC mathematics and never creates
   CAL science qualification.
4. The passband-identity authority resolves exact CAL provenance and join
   identity only. It changes no CAL value or profile predicate.
5. RTC logical-stream terminal completion governs the consumer-neutral
   PTC-disabled route and RTC terminal lifecycle. It does not change the
   representative-origin, synthesis/replacement, support, operator, response,
   uncertainty, or lifecycle facts imported by the five ordinary PTC
   profiles.
6. The RTC source correction is explicitly explanatory. Its normative core,
   engineering source, crosswalk, and controlling owner ledger are unchanged.

## 3. Proposed replay and identity rule

The five immutable profile records retain their existing keys, versions,
meanings, and bytes. For a decision evaluated under the WP-7 successor
generation, the effective adjacent-source identity is the ordered pair:

```text
(base_source_binding_register_sha256,
 wp7_successor_compatibility_authority_sha256)
```

The first member must be exactly
`ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430`.
The second member is the SHA-256 of this file after direct scientific-owner
approval. A decision record must retain both digests, the exact immutable
profile key, and the exact profile-registry digest
`5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893`.

This additive binding is part of successor replay identity. Omitting it does
not authorize a successor-generation decision under the old source identity.
It also does not retroactively alter a decision made under the earlier base
generation.

## 4. Failure and supersession

Any missing or changed digest in Section 1, conflict between the successor
authority and a profile consequence, changed profile predicate or action,
changed RTC fact imported by a profile, or changed CAL classification
consequence requires either a new owner-approved compatibility determination
or a new immutable profile version. Until then, the dependent decision is
`decision_unavailable`; no “latest source” substitution is permitted.

## 5. Proposed precedence and limitations

Approval would supplement the continuing SCI-VAL source-binding register only
for the exact WP-7 successor generation in Section 1. It would resolve the
addendum/profile compatibility decision while preserving the register's
fail-closed change rule and all five profile consequences.

Approval would not establish implementation conformity, observational
validation, achieved performance, production readiness, MAP admission,
science qualification, or downstream consumer acceptance. Final audit-finding
closure remains the responsibility of a fresh independent successor audit.
