# SCI-FLT-FIXED v0.1 Policy and Action Architecture

Record identity: `SCI-FLT-FIXED-POLICY-ACTION-ARCHITECTURE v0.1/draft-r0.3`

Status: unregistered Stage B profile architecture draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-30`

## Policy domains

```text
SCI-FLT-FIXED:input_bundle_admission@1
  evaluates one complete parent bundle and resolved plan

SCI-FLT-FIXED:input_parent_row_admission@1
  evaluates each exact parent row for the named FLT use

SCI-FLT-FIXED:output_publication@1
  evaluates one complete candidate and defines disposition plus action
```

`J_full` and `S_out` are deterministic FLT-owned constructions from parent-row
decisions, `K_req`, parent domain, availability, finiteness, and predicates.
Profiles do not perform convolution or author output-row arithmetic support.

SCI-VAL may bind and evaluate a future owner-approved immutable profile and
produce a decision artifact. The FLT publisher performs or declines the
prescribed action. FLT owns realization and FLT-local validity. A policy or VAL
evaluation does not perform publication.

## Registration state

All three r0.3 profile identities are drafts, not owner-approved Registry
entries, not registered, and not Registry-evaluated. Their names are corrected
before any approval.
